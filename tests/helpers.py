"""Helpers several test files share: tiny MLX models, a stand-in tokenizer, waiting for saved traces, a fake GPU
child, a fake profile, synthetic expert traces, and the steering tests' vectors, norms and middleware runs."""

import asyncio
import json
import sys
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import llama, qwen3_moe
from mlx_lm.models.base import create_attention_mask

from fastapi import FastAPI
from fastapi.testclient import TestClient

from metalscope import concepts as C, gpulock, middleware, models, vectors, web
from metalscope.learn import gpu, job as J, plan as P
from metalscope.omlx import backend as omlx_backend
from metalscope.store import Trace


D = 64


E, K = 32, 4


VOCAB: dict[str, int] = {}   # shared across traces so equal texts get equal token ids


TOKENS = mx.array([[1, 5, 9, 2, 7]])


def tiny_moe():
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
        num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1,
        mlp_only_layers=[], moe_intermediate_size=32, rms_norm_eps=1e-5,
        vocab_size=100, head_dim=16, rope_theta=10000.0,
        tie_word_embeddings=False, max_position_embeddings=512, norm_topk_prob=True,
    )
    mx.random.seed(0)
    model = qwen3_moe.Model(args)
    mx.eval(model.parameters())
    return model


def tiny_llama(layers=3):
    args = llama.ModelArgs(
        model_type="llama", hidden_size=64, num_hidden_layers=layers,
        intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=100, tie_word_embeddings=False,
    )
    mx.random.seed(0)
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


def hc_write(streams, branch, gate):
    """A pending write applied, as oMLX's qwen4_exp does (adapters.qwen4_exp._apply_write)."""
    return streams + (branch[..., None, :] * gate[..., None]).reshape(*streams.shape)


class Qwen4ExpFakeLayer(nn.Module):
    """A llama block as a Flash Next layer: it reads its streams' mean and writes its change into each stream by a gate
    of that stream's own, left pending for the next layer as (streams, branch, gate) when ``defer_write``."""

    def __init__(self, block, gate):
        super().__init__()
        self.block, self.gate = block, gate

    def __call__(self, x, inputs=None, mask=None, cache=None, write=None, defer_write=False):
        if write is not None:
            x = hc_write(x, *write)
        B, T, _ = x.shape
        hc = self.gate.shape[0]
        h = x.reshape(B, T, hc, -1).mean(axis=2)
        out = (x, self.block(h, mask, cache) - h, mx.broadcast_to(self.gate, (B, T, hc)))
        return out if defer_write else hc_write(*out)


class Qwen4ExpFakeMixer(nn.Module):
    def __init__(self, norm, hc):
        super().__init__()
        self.norm, self.hc = norm, hc

    def __call__(self, x):
        return self.norm(x.reshape(*x.shape[:-1], self.hc, -1).mean(axis=-2))


class Qwen4ExpFakeModel(nn.Module):
    """Flash Next's shape around a tiny llama (the qwen4_exp adapter takes it): ``hc`` streams, the embedding tiled into
    them, each layer's write deferred into the next (``defer``, oMLX's fused writes) or applied at once."""

    def __init__(self, base, hc, defer):
        super().__init__()
        self.args = SimpleNamespace(hc_count=hc, hidden_size=base.args.hidden_size)
        self.embed_tokens = base.embed_tokens
        self.layers = [Qwen4ExpFakeLayer(b, mx.random.uniform(0.5, 1.5, (hc,), key=mx.random.key(i)))
                       for i, b in enumerate(base.layers)]
        self.hyper_connection_mixer = Qwen4ExpFakeMixer(base.norm, hc)
        self.defer = defer

    def __call__(self, inputs, cache=None):
        h = mx.tile(self.embed_tokens(inputs), (1, 1, self.args.hc_count))
        cache = cache or [None] * len(self.layers)
        mask, write = create_attention_mask(h, cache[0]), None
        for layer, c in zip(self.layers, cache):
            h = layer(h, inputs, mask=mask, cache=c, write=write, defer_write=self.defer)
            if self.defer:
                h, write = h[0], h[1:]
        return self.hyper_connection_mixer(h if write is None else hc_write(h, *write))


class FlashNextLike(nn.Module):
    def __init__(self, layers=3, hc=4, defer=True):
        super().__init__()
        base = tiny_llama(layers)
        self.model = Qwen4ExpFakeModel(base.model, hc, defer)
        self.lm_head = base.lm_head
        mx.eval(self.parameters())

    @property
    def layers(self):       # make_prompt_cache's count
        return self.model.layers

    def __call__(self, inputs, cache=None):
        return self.lm_head(self.model(inputs, cache))


def flash_next_writes(model, ids, sizes=False):
    """A ``FlashNextLike`` model's layers run by hand over ``ids`` (no cache): each layer's output streams less the
    streams it read (pending writes applied), a norm per stream, [L, B, T, hc] (CaptureSession's resid_write_norm);
    ``sizes``: with each layer's output streams' norms beside them, the same shape (hc_stream_norm)."""
    inner = model.model
    hc = inner.args.hc_count
    h, write, out, size = mx.tile(inner.embed_tokens(ids), (1, 1, hc)), None, [], []
    mask = "causal" if ids.shape[1] > 1 else None
    per_stream = lambda x: np.linalg.norm(np.array(x, np.float32).reshape(*x.shape[:-1], hc, -1), axis=-1)
    for layer in inner.layers:
        before = h if write is None else hc_write(h, *write)
        r = layer(h, mask=mask, write=write, defer_write=True)
        h, write = r[0], r[1:]
        out.append(per_stream(hc_write(*r) - before))
        size.append(per_stream(hc_write(*r)))
    return (np.stack(out), np.stack(size)) if sizes else np.stack(out)


def drain(loop, q):
    import asyncio
    loop.run_until_complete(asyncio.sleep(0))
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def wait_saved(mgr, n, timeout=5):
    end = time.time() + timeout
    while mgr.saved < n and time.time() < end:
        time.sleep(0.02)
    assert mgr.saved >= n


def fake_child(monkeypatch, lines, seen, rc=0, interrupt=False):
    """subprocess.Popen for a job's GPU child (``python -m CHILD``): checks the command is this Python on a module with
    arguments every child accepts (gpu.child_parser), records it, who holds the lock when it starts and the descriptors
    it is passed, then prints ``lines`` and exits with ``rc`` (or, ``interrupt``, the job is cancelled while it runs)."""
    real = J.subprocess.Popen

    class Child:
        pid = 987654321

        def __new__(cls, cmd, **kw):           # only the GPU child is faked (subprocess.run uses Popen too)
            return super().__new__(cls) if cmd[1:2] == ["-m"] else real(cmd, **kw)

        def __init__(self, cmd, **kw):
            assert cmd[0] == sys.executable, cmd
            try:
                gpu.child_parser(cmd[2]).parse_known_args(cmd[3:])
            except SystemExit:
                raise AssertionError(f"a GPU child would refuse its arguments: {cmd}") from None
            seen.update(cmd=cmd, holder=gpulock.held(), pass_fds=kw.get("pass_fds"))
            self.done = False

            def out():
                yield from lines
                if interrupt:
                    raise KeyboardInterrupt("stopped")
                self.done = True
            self.stdout = out()

        def wait(self, timeout=None):
            self.done = True
            return rc

        def poll(self):
            return rc if self.done else None
    monkeypatch.setattr(J.subprocess, "Popen", Child)


class Tok:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def fake_profile(home, model="tiny", name="toy", layers=(0, 1, 2), peak=1):
    rng = np.random.default_rng(0)
    vec = vectors.Vector(name, {l: vectors.unit(rng.standard_normal(D)) for l in layers}, dict(hc_count=1))
    vectors.save_vector(model, vec)
    stats = {}
    for l in layers:
        stats[str(l)] = dict(heldout=4.0 if l == peak else 2.0, dprime=5.0, perm_p95=0.5, p_value=0.005, permutations=200,
                             streams=[], fold_repeats=5)
    prof = dict(concept=name, title="Toy", description="d", model=model, stats=stats,
                adjacent_cosine={}, n_pos=3, n_neg=3, position="mean",
                inputs=dict(n_pos=3, n_neg=3, position="mean", n_layers=len(layers), pos_example="soft", neg_example="hard", words=["soft"],
                            word_tokens=["soft"], target_words=["cat"], target_tokens=["cat"]),
                writer=dict(first=0, jump=1, rule="p < 0.01"),
                reader=dict(layer=1, base_p=0.3, base_chosen=0.25, best=dict(strength=0.5, concept=0.8, control=0.1, effect=0.7, concept_mass_ratio=0.97,
                                                                         control_mass_ratio=1.0, chosen=0.6, control_chosen=0.3)),
                generation=dict(layer=1, strength=0.5, instruction="why", max_tokens=4, shifts=[0], task_items=[], items=[],
                                ladder_instruction="why, at length", ladder_tokens=160, reader_strength=0.5, below_edge=True,
                                summary={k: dict(n=0, target=0.5, lettered=1.0, concept_words=0.2, stopped=1.0, distinct=0.9)
                                         for k in ("base", "concept", "control")},
                                ladder=[generation_rung(0.0), generation_rung(0.5), generation_rung(2.0, text="", n=0)]))
    d = models.concept_dir(model, name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.json").write_text(json.dumps(prof))
    return vec, prof


def generation_rung(strength, text="A) fine, and said to the end.", n=12):
    """One stored ladder reply, as ``generation`` records it."""
    return dict(item=0, strength=strength, target_letter="A", letter="A" if text else None, hits=[], text=text, n_tokens=n, stopped=True,
                distinct=1.0 if n else 0.0, distinct_window=1.0 if n else 0.0, early_stop=False, short=n < C.SHORT_TOKENS)


def random_ids(rng, L, T, k=K, n=E):
    return np.stack([np.stack([rng.choice(n, k, replace=False) for _ in range(T)]) for _ in range(L)])


def make_trace(texts, ids, prompt_len=None, probs=None, title="t"):
    """Trace with router records ids [L, T, k] and one token id per distinct text."""
    for t in texts:
        VOCAB.setdefault(t, len(VOCAB))
    arrays = {}
    for l in range(ids.shape[0]):
        arrays[f"router_ids/{l}"] = ids[l].astype(np.int32)
        arrays[f"router_probs/{l}"] = (np.full(ids[l].shape, 0.1, np.float32) if probs is None else probs[l])
    return Trace(arrays, [VOCAB[t] for t in texts], list(texts),
                 len(texts) // 2 if prompt_len is None else prompt_len, dict(num_experts=E, title=title))


def run_middleware(scope_headers, body, app_response=200, resolve_model=None):
    from metalscope.middleware import TraceContextMiddleware, REQUEST_CTX
    seen = {}

    async def app(scope, receive, send):
        seen["ctx"] = REQUEST_CTX.get()
        msg = await receive()
        seen["body"] = msg["body"]
        await send({"type": "http.response.start", "status": app_response, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})

    sent = []

    async def send(m):
        sent.append(m)

    messages = iter([{"type": "http.request", "body": body, "more_body": False}])

    async def receive():
        return next(messages)

    scope = dict(type="http", method="POST", path="/v1/chat/completions",
                 headers=[(k.encode(), v.encode()) for k, v in scope_headers.items()])
    middleware = TraceContextMiddleware(app, resolve_model=resolve_model)
    asyncio.new_event_loop().run_until_complete(middleware(scope, receive, send))
    return seen, sent


def manual_logits(model, tokens, add=None):
    """Hook-free reference forward; ``add``: {layer: [D] vector} added after that layer."""
    h = model.model.embed_tokens(tokens)
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask="causal", cache=None)
        if add and i in add:
            h = h + mx.array(add[i]).astype(h.dtype)[None, None]
    return model.lm_head(model.model.norm(h))


def manual_residuals(model, ids, layers):
    """Hook-free residuals after each of ``layers``: {layer: [T, D] float32} for one row."""
    h, out = model.model.embed_tokens(mx.array([ids])), {}
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask="causal", cache=None)
        if i in layers:
            out[i] = np.array(h[0].astype(mx.float32))
    return out


def learn_job(name, description, model_path, model_name, engine="mlx", corpus=None, writer=None, **kw):
    """A LearnJob as `concept learn` makes one, from its plan (P.learn_plan); a plan that refuses is ValueError."""
    plan = P.learn_plan(name, model_name, description, corpus, writer, engine)
    if plan["taken"]:
        raise ValueError(plan["taken"])
    return J.LearnJob(plan, model_path, model_name, **kw)


def write_norms(model, layers, value=10.0, streams=None):
    p = models.norms_path(model)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(
        model=model, traces=["t"], layers={str(l): value for l in layers},
        streams={str(l): streams for l in layers} if streams else {}, hc_count=len(streams) if streams else 1)))


def rand_vec(name, layers, width=D, seed=1):
    rng = np.random.default_rng(seed)
    return vectors.Vector(name, {l: vectors.unit(rng.standard_normal(width)) for l in layers},
                        dict(source="test", hc_count=1, natural_norm={l: 3.0 for l in layers}))


def scope_client(backend=None) -> TestClient:
    """The /scope routes on a fresh app, as oMLX mounts them (``backend``: another web.Backend), the local-only check
    off."""
    app = FastAPI()
    web.mount(app, backend or omlx_backend.BACKEND)
    app.dependency_overrides[middleware._local_only] = lambda: None
    return TestClient(app)

"""The engine interface (engines/): the measurements on a numpy FakeEngine (sweep, reader, generation ladder,
the lens steps skipped by capability), the MLX engine against hook-free references, and the hyper-connection
count read from a layer output's shape."""
import json
import re
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from metalscope import concepts as C, vectors
from metalscope import models, steer
from metalscope.engines import NEXT_TOP, SINK, Engine, Shape, Tokenizer, hc_of, samples
from metalscope.engines.mlx import MlxEngine
from tests.helpers import tiny_llama, tiny_moe, write_norms
from tests.test_concepts import Tok, write_concept
from tests.helpers import TOKENS, manual_logits, manual_residuals


# ---------------------------------------------------------------- a numpy engine

class FakeTok(Tokenizer):
    """test_concepts' Tok as an engine tokenizer; a chat turn is the text after a marker id."""
    eos = {0}

    def __init__(self):
        self.t = Tok()

    def encode(self, text):
        return self.t.encode(text)

    def chat(self, user, thinking=False):
        return [1] + self.t.encode(user)

    def decode(self, ids):
        return self.t.decode(ids)


class FakeEngine(Engine):
    """A linear residual stream: layer l writes the previous residual plus a bias plus ``gain[l]`` times its add, so
    an add at a layer reaches every later one, and layer 2's counts most; logits are the last residual times U."""
    kind = "fake"

    def __init__(self, width=16, vocab=100, gain=(0.2, 0.5, 2.0, 1.0), reply_cap=None):
        rng = np.random.default_rng(0)
        self.served = self.checkpoint = "fake"
        self.gain, self.reply_cap = gain, reply_cap
        self.shape = Shape(len(gain), width, 1, width, vocab)
        self.tok = FakeTok()
        self.E = rng.standard_normal((vocab, width)).astype(np.float32)
        self.B = 0.1 * rng.standard_normal((len(gain), width)).astype(np.float32)
        self.U = rng.standard_normal((width, vocab)).astype(np.float32) / np.sqrt(width)
        self.calls = 0

    def _hidden(self, ids, adds):
        h, out = self.E[np.asarray(ids)], []
        for l, g in enumerate(self.gain):
            h = h + self.B[l] + g * (adds or {}).get(l, 0.0)
            out.append(h)
        return out                                                      # [L] x [T, W]

    def _lp(self, ids, adds):
        z = self._hidden(ids, adds)[-1][-1] @ self.U
        return z - (z.max() + np.log(np.exp(z - z.max()).sum()))

    def residuals(self, prompts_ids, layers, sink=SINK, adds=None):
        hs = [self._hidden(ids, adds) for ids in prompts_ids]
        mean = np.stack([[h[l][min(sink, len(h[l]) - 1):].mean(0) for l in layers] for h in hs])
        last = np.stack([[h[l][-1] for l in layers] for h in hs])
        return mean, last

    def logprobs(self, ids, token_ids, adds=None):
        self.calls += 1
        return self._lp(ids, adds)[list(token_ids)]

    def generate(self, ids, adds, max_tokens):
        seq, toks = list(ids), []
        for _ in range(max_tokens):
            nxt = int(np.argmax(self._lp(seq, adds)))
            if nxt in self.tok.eos:
                return toks, True
            if self.reply_cap is not None and len(toks) == self.reply_cap:
                return toks, False                                      # cut short (a context limit): not stopped on its own
            toks.append(nxt)
            seq.append(nxt)
        return toks, False


def _prompts(engine, n=4):
    """Prompts whose target is always letter A (slot 0), so a push toward A's logit is a push toward the target."""
    rng = np.random.default_rng(1)
    return [dict(item=i, shift=0, ids=[1] + rng.integers(2, 80, size=6).tolist(), letters=[90, 91, 92], slot=0) for i in range(n)]


def test_sweep_on_a_numpy_engine_finds_the_layer_that_counts_most(home):
    engine = FakeEngine()
    write_norms("fake", range(4), value=1.0)
    U = engine.U
    toward = U[:, 90] - U[:, [91, 92]].mean(1)                         # raises A against B and C at the last position
    vec = vectors.Vector("v", {l: vectors.unit(toward) for l in range(4)}, dict(hc_count=1))
    prompts = _prompts(engine)
    sw = C.sweep(engine, vec, steer.norm_table("fake"), prompts, layers=[0, 1, 2, 3], strengths=[0.0, 0.5, 1.0], seeds=(1, 2))
    assert engine.calls == len(prompts) + 1 + 4 * 3 * len(prompts) * 3     # base, the determinism check, concept + 2 controls
    assert sw["determinism"] == 0.0
    np.testing.assert_allclose(sw["concept_logodds"][:, 0], np.broadcast_to(sw["base_logodds"], (4, len(prompts))), atol=1e-6)
    lp = engine.logprobs(prompts[0]["ids"], prompts[0]["letters"])
    assert sw["base_logodds"][0] == pytest.approx(C.choice_scores(lp, 0)[0])
    assert (sw["concept_logodds"][:, 2].mean(-1) > sw["base_logodds"].mean()).all()
    rd = C.reader_layer(sw, mass_floor=0.0)
    assert rd["layer"] == 2 and rd["best"]["strength"] == 1.0                # gain 2.0: the layer whose add counts most
    effects = {l: rd["per_layer"][l]["best"]["effect"] for l in range(4)}
    assert effects[2] > effects[3] > effects[1] > effects[0] > 0


def test_generation_ladder_on_a_numpy_engine_uses_the_engines_stop(home):
    engine = FakeEngine(reply_cap=3)
    write_norms("fake", range(4), value=1.0)
    write_concept("fake", n=4)
    concept = C.load_concept("fake", "toy")
    vec = vectors.Vector("v", {1: vectors.unit(np.random.default_rng(0).standard_normal(16))}, dict(hc_count=1))
    g = C.generation(engine, concept, vec, steer.norm_table("fake"), layer=1, strength=0.5, max_tokens=6,
                     ladder=(0.5, 2.0), ladder_items=2)
    assert len(g["items"]) == 3 and set(g["items"][0]["replies"]) == {"base", "concept", "control"}
    assert [x["strength"] for x in g["ladder"]] == [0.0, 0.5, 2.0] * 2
    # Cut short at 3 tokens, under the budget: the engine says it did not stop on its own, and that is what counts.
    replies = [r for rec in g["items"] for r in rec["replies"].values()] + g["ladder"]
    cut = [r for r in replies if not r["stopped"]]
    assert cut and all(r["n_tokens"] == 3 for r in cut) and all(r["n_tokens"] <= 3 for r in replies)
    assert g["fluency"]["reference"]["n"] == 2 and [x["strength"] for x in g["fluency"]["rungs"]] == [0.5, 2.0]
    assert C.measure._fluency([5, 6], 10, "x", stopped=False)["stopped"] is False and C.measure._fluency([5, 6], 10, "x")["stopped"] is True


def test_a_profile_is_the_checkpoints_and_says_where_it_was_measured(home):
    """Presence needs only the engine's means; the profile is the checkpoint's, measured where."""
    write_norms("fake", range(4), value=1.0)
    write_concept("fake", n=6)
    prof = C.profile(FakeEngine(), C.load_concept("fake", "toy"), log_fn=lambda s: None)
    assert prof["stats"] and vectors.find_vector("fake", "toy") is not None
    assert (prof["model"], prof["measured_on"]) == ("fake", dict(engine="fake", served="fake"))


def test_a_block_vector_is_pushed_over_its_block_as_the_header_pushes_it(home):
    """A vector whose meta names a ``block`` (Inkling 2's concepts): a measurement's push at a layer adds at the rest
    of its block too, each layer its own direction and norm, the control's over the same layers; without one, the
    layer alone (every other model, as before)."""
    write_norms("fake", range(4), value=2.0)
    table = steer.norm_table("fake")
    rng = np.random.default_rng(3)
    v = vectors.Vector("v", {l: vectors.unit(rng.standard_normal(16)) for l in range(4)}, dict(hc_count=1, block=2))
    adds = C.measure.adds_for(v, 2, 0.5, table)
    assert sorted(adds) == [2, 3] and all(np.allclose(adds[l], 1.0 * v.vectors[l]) for l in (2, 3))
    assert sorted(C.measure.adds_for(vectors.Vector("p", v.vectors, dict(hc_count=1)), 2, 0.5, table)) == [2]
    ctrl = steer.control_direction(v, [2, 3], seed=1)
    k = C.measure.adds_for(v, 2, 0.5, table, ctrl)
    assert sorted(k) == [2, 3] and np.allclose(k[3], ctrl[3])


def test_an_engine_with_a_block_and_its_own_steer_layers_learns_over_them(home):
    """concept learn on an engine that names its block and steer layers (MlxEngine on Inkling 2): the vector saved
    with the block, the selection sweep at those layers alone, every push over the block."""
    from metalscope.learn import gpu
    pushed = []

    class Spy(FakeEngine):
        block, steer_layers = 2, [2]

        def logprobs(self, ids, token_ids, adds=None):
            pushed.append(tuple(sorted(adds or ())))
            return super().logprobs(ids, token_ids, adds)

        def generate(self, ids, adds, max_tokens):
            pushed.append(tuple(sorted(adds or ())))
            return super().generate(ids, adds, max_tokens)
    write_norms("fake", range(4), value=1.0)
    write_concept("fake", n=6)
    prof = gpu.run(Spy(), "toy", strengths=(0.5,))
    assert vectors.load_vector("fake", "toy").meta["block"] == 2
    assert prof["learn"]["sweep_layers"] == [2] and set(pushed) == {(), (2, 3)}


def test_samples_take_the_last_position_or_the_mean_from_the_sink():
    engine = FakeEngine()
    texts = ["1 2 3 4 5 6 7 8", "9 8 7"]
    last, mean = samples(engine, texts, [0, 3], "last"), samples(engine, texts, [0, 3], "mean")
    h = engine._hidden(engine.tok.encode(texts[0]), None)
    np.testing.assert_allclose(last[3][0], h[3][-1])
    np.testing.assert_allclose(mean[3][0], h[3][SINK:].mean(0), rtol=1e-6)
    np.testing.assert_allclose(mean[0][1], engine._hidden(engine.tok.encode(texts[1]), None)[0][-1])   # shorter than the sink
    chat = samples(engine, texts, [0], "last", chat=True)
    np.testing.assert_allclose(chat[0][0], engine._hidden(engine.tok.chat(texts[0]), None)[0][-1])


# ---------------------------------------------------------------- the MLX engine

def test_mlx_engine_matches_hook_free_references():
    model = tiny_llama()
    engine = MlxEngine(model, Tok(), "tiny")
    assert engine.shape == Shape(3, 64, 1, 64, 100)
    ids = TOKENS[0].tolist()
    ref = manual_logits(model, TOKENS)[0, -1]
    want = np.array(ref - mx.logsumexp(ref))[[3, 7, 9]]
    np.testing.assert_allclose(engine.logprobs(ids, [3, 7, 9]), want, rtol=1e-5, atol=1e-5)
    v = 7.0 * vectors.unit(np.random.default_rng(1).standard_normal(64))
    ref = manual_logits(model, TOKENS, {1: v})[0, -1]
    steered = engine.logprobs(ids, [3, 7, 9], {1: v})
    np.testing.assert_allclose(steered, np.array(ref - mx.logsumexp(ref))[[3, 7, 9]], rtol=1e-4, atol=1e-4)
    assert not np.allclose(steered, want, atol=1e-3)
    grab = manual_residuals(model, ids, [0, 2])
    mean, last = engine.residuals([ids], [0, 2])
    np.testing.assert_allclose(last[0, 1], grab[2][-1], rtol=1e-6)
    np.testing.assert_allclose(mean[0, 0], grab[0][SINK:].mean(0), rtol=1e-6)
    toks, stopped = engine.generate(ids, {1: v}, 4)
    assert len(toks) <= 4 and stopped == (len(toks) < 4)
    from metalscope.hooks import is_hooked
    assert not any(is_hooked(l) for l in model.model.layers)


@pytest.mark.parametrize("n, split", [(23, [5, 5, 5, 5, 3]), (20, [5, 5, 5, 5, 1])])  # 20: no empty last chunk
def test_mlx_generate_prefills_a_long_prompt_in_chunks_to_the_same_reply(monkeypatch, n, split):
    from metalscope.engines import mlx as mlx_engine
    engine = MlxEngine(tiny_llama(), Tok(), "tiny")
    ids = np.random.default_rng(3).integers(10, 100, n).tolist()
    v = 7.0 * vectors.unit(np.random.default_rng(1).standard_normal(64))
    whole = [engine.generate(ids, adds, 6) for adds in (None, {1: v})]
    calls = []
    real = engine.model.__call__
    monkeypatch.setattr(mlx_engine, "PREFILL_CHUNK", 5)
    monkeypatch.setattr(type(engine.model), "__call__", lambda self, x, *a, **k: calls.append(x.shape[1]) or real(x, *a, **k))
    chunked = [engine.generate(ids, adds, 6) for adds in (None, {1: v})]
    assert chunked == whole and calls[:5] == split


class _Ple(nn.Module):
    """A stand-in n-gram lookup: what it adds to the residual depends on the token (its own small table)."""

    def __init__(self):
        super().__init__()
        self.table = nn.Linear(64, 64)

    def __call__(self, h):
        return 3.0 * mx.tanh(self.table(h))


def _with_ple(model, at=1):
    """``model`` with a lookup child ``ple`` on decoder layer ``at`` whose output is added to the layer's input,
    as Flash Next's PLE layer is; ``off`` (a list) leaves its write out when it holds True."""
    layer, off = model.model.layers[at], [False]
    mx.random.seed(5)
    layer.ple = _Ple()
    mx.eval(layer.ple.parameters())
    cls = type(layer)

    class WithPle(cls):
        def __call__(self, x, *a, **k):
            add = self.ple(x)
            return cls.__call__(self, x + (mx.zeros_like(add) if off[0] else add), *a, **k)
    layer.__class__ = WithPle
    return model, off


@pytest.mark.parametrize("start", [1, 4, 22])
def test_mlx_next_logprobs_match_one_whole_forward_with_and_without_the_lookup(monkeypatch, start):
    from metalscope.engines import mlx as mlx_engine
    from metalscope.hooks import is_hooked
    model, off = _with_ple(tiny_llama())
    engine = MlxEngine(model, Tok(), "tiny")
    assert engine.lookups == [model.model.layers[1].ple]
    ids = np.random.default_rng(4).integers(10, 100, 23).tolist()

    def ref(lookup_off):
        off[0] = lookup_off
        z = model(mx.array([ids]))[0].astype(mx.float32)
        off[0] = False
        lp = np.array(z - mx.logsumexp(z, axis=-1, keepdims=True))[start - 1:len(ids) - 1]
        top = np.argsort(-lp, axis=-1)[:, :NEXT_TOP]
        return lp[np.arange(len(lp)), ids[start:]], top, np.take_along_axis(lp, top, -1)
    monkeypatch.setattr(mlx_engine, "LOOKUP_CHUNK", 5)               # five chunks through one cache
    monkeypatch.setattr(mlx_engine, "TOP_ROWS", 2)                   # the top choices picked two rows at a time
    chunks = []
    run = lambda fn: chunks.append(1) or fn()
    on, gone = engine.next_logprobs(ids, start, run=run), engine.next_logprobs(ids, start, lookup=False, run=run)
    for got, want in ((on, ref(False)), (gone, ref(True))):
        np.testing.assert_allclose(got.lp, want[0], rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(got.ids, want[1])              # likeliest first, every row from start - 1
        np.testing.assert_allclose(got.top, want[2], rtol=1e-4, atol=1e-4)
    assert not np.allclose(on.lp, gone.lp, atol=1e-2) and len(on.lp) == len(ids) - start == len(on.ids)
    assert len(chunks) == 10 and not is_hooked(model.model.layers[1].ple)    # zeroed inside each chunk only
    with pytest.raises(ValueError):
        engine.next_logprobs(ids, len(ids))


def test_mlx_next_logprobs_take_the_lookups_zeroing_off_when_a_chunk_fails():
    from metalscope.hooks import is_hooked
    model, _off = _with_ple(tiny_llama())
    later = model.model.layers[2]

    class Broken(type(later)):
        def __call__(self, *a, **k):
            raise RuntimeError("out of memory")
    later.__class__ = Broken
    with pytest.raises(RuntimeError, match="out of memory"):
        MlxEngine(model, Tok(), "tiny").next_logprobs([11, 12, 13, 14], 1, lookup=False)
    assert not is_hooked(model.model.layers[1].ple)                       # the served model is left as it was


def test_mlx_next_logprobs_need_a_lookup_to_leave_out():
    engine = MlxEngine(tiny_llama(), Tok(), "tiny")
    assert engine.lookups == []
    with pytest.raises(ValueError, match="no n-gram lookup"):
        engine.next_logprobs([11, 12, 13], 1, lookup=False)
    assert len(engine.next_logprobs([11, 12, 13], 1).lp) == 2
    assert engine.chunk_bytes(512) == 100 * 8 * (512 + 128)         # the chunk's logits and a top pick's, vocab 100


class _Offloaded(nn.Module):
    """oMLX's OffloadSwitchGLU as lookup_chunk reads it: the experts in a cache of ``capacity`` slots; ``seen`` gets
    each call's routes (tokens * k)."""

    def __init__(self, glu, capacity, seen, **floor):
        super().__init__()
        self.glu, self.seen = glu, seen
        self.cache = SimpleNamespace(moe_offload_cache=True, capacity=capacity, **floor)

    def __call__(self, x, indices):
        self.seen.append(indices.size)
        return self.glu(x, indices)


def _offloaded(model, capacities, seen=None):
    for layer, c in zip(model.model.layers, capacities):
        layer.mlp.switch_mlp = _Offloaded(layer.mlp.switch_mlp, c, [] if seen is None else seen)
    return model


@pytest.mark.parametrize("capacities, k, rows", [((384, 512), 10, 38), ((256, 256), 10, 25), ((8, 512), 10, 1),
                                                 ((), 10, 512), ((5000, 5000), 2, 512)])
def test_mlx_lookup_chunk_keeps_each_moe_call_within_the_smallest_slots(capacities, k, rows):
    model = _offloaded(tiny_moe(), capacities)
    for layer in model.model.layers:
        layer.mlp.top_k = k
    engine = MlxEngine(model, Tok(), "tiny")
    assert engine.lookup_chunk() == rows
    if capacities:                                           # read when called: oMLX shrinks the slots under pressure
        model.model.layers[1].mlp.switch_mlp.cache.capacity = 100
        assert engine.lookup_chunk() == min(rows, 100 // k)
    assert MlxEngine(tiny_llama(), Tok(), "tiny").lookup_chunk() == 512       # no offload


@pytest.mark.parametrize("floor, rows", [(dict(floor=10), 38), ({}, 1)])
def test_mlx_lookup_chunk_without_a_routing_rule_reads_the_slots_floor_else_one_row(floor, rows):
    model = tiny_llama(2)
    for layer in model.model.layers:
        layer.mlp.switch_mlp = _Offloaded(None, 384, [], **floor)
    assert MlxEngine(model, Tok(), "tiny").lookup_chunk() == rows


def test_mlx_next_logprobs_read_in_the_rows_the_expert_slots_hold():
    from mlx_lm.models.cache import KVCache

    def moe(capacities=(), seen=None):
        model = _offloaded(tiny_moe(), capacities, seen)
        model.make_cache = lambda: [KVCache() for _ in model.layers]
        return MlxEngine(model, Tok(), "tiny")
    ids = np.random.default_rng(6).integers(10, 100, 9).tolist()
    whole = moe().next_logprobs(ids, 1)
    seen, chunks = [], []
    engine = moe((7, 5), seen)                                                # k 2: 2 rows a forward
    got = engine.next_logprobs(ids, 1, run=lambda fn: chunks.append(1) or fn())
    assert len(chunks) == 5 and max(seen) <= 5
    np.testing.assert_allclose(got.lp, whole.lp, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(got.ids, whole.ids)


# ---------------------------------------------------------------- the hyper-connection count, from the shape

def test_hc_is_read_from_the_layer_output_shape():
    assert hc_of((2, 3, 64), 64) == (64, 1)                    # [B, T, D]
    assert hc_of((2, 3, 4, 16), 16) == (64, 4)                 # [B, T, hc, D]: DeepSeek V4's streams
    assert hc_of((2, 3, 64), 16) == (64, 4)                    # [B, T, hc*D]: Flash Next's
    with pytest.raises(ValueError, match="not a whole number"):
        hc_of((1, 1, 60), 16)


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def __call__(self, h):
        return h + mx.tanh(self.lin(h))


class _Inner(nn.Module):
    def __init__(self, hc, d, vocab, n):
        super().__init__()
        self.hc = hc
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = [_Block(d) for _ in range(n)]

    def __call__(self, x, add=None):
        h = mx.repeat(self.embed_tokens(x)[:, :, None, :], self.hc, axis=2)          # [B, T, hc, D]
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if add and i in add:
                h = h + mx.array(add[i]).reshape(self.hc, -1)
        return h


class Streams(nn.Module):
    """DeepSeek V4's shape: every layer carries ``hc`` streams ``[B, T, hc, D]``, summed down before the head."""

    def __init__(self, hc=2, d=8, vocab=20, n=3):
        super().__init__()
        mx.random.seed(0)
        self.model = _Inner(hc, d, vocab, n)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def __call__(self, x, add=None):
        return self.lm_head(self.model(x, add).sum(-2))


def test_four_d_streams_are_measured_and_steered_flat():
    model = Streams()
    engine = MlxEngine(model, Tok(), "streams")
    assert engine.shape == Shape(3, 16, 2, 8, 20)
    ids = [3, 5, 7, 2, 9, 4]
    mean, last = engine.residuals([ids], [1])
    h = mx.repeat(model.model.embed_tokens(mx.array([ids]))[:, :, None, :], 2, axis=2)
    for layer in model.model.layers[:2]:
        h = layer(h)
    assert last.shape == (1, 1, 16)
    np.testing.assert_allclose(last[0, 0], np.array(h[0, -1]).reshape(-1), rtol=1e-6)               # streams flattened, hc-major
    v = vectors.unit(np.random.default_rng(2).standard_normal(16)) * 3.0
    ref = model(mx.array([ids]), {1: v})[0, -1]
    np.testing.assert_allclose(engine.logprobs(ids, [0, 4], {1: v}), np.array(ref - mx.logsumexp(ref))[[0, 4]], rtol=1e-5, atol=1e-5)
    # SteerSession itself: a 4-D output is flattened for the add and comes back 4-D.
    s = steer.SteerSession(lambda *_: None)
    s.install(model)
    s.set_rows([(0, "r", steer.SteerPlan({0: v}, "all", {}))], 1, "decode")
    x = mx.ones((1, 2, 2, 8))
    out = model.model.layers[0](x)
    s.close()
    assert out.shape == (1, 2, 2, 8)
    np.testing.assert_allclose(np.array(out), np.array(model.model.layers[0](x) + mx.array(v).reshape(2, 8)), rtol=1e-6)


DS4_CHILD = '''
"""concept learn's GPU child on ds4 as a real process: the real Ds4Engine in front of a stand-in ds4-server whose
measurement request writes its capture file, says so, then takes two minutes."""
import json, sys, time
from pathlib import Path
import httpx
from metalscope.engines import ds4 as E
from metalscope.learn import gpu

ready, argv = Path(sys.argv[1]), sys.argv[2:]
E.model_shape = lambda g: dict(family="qwen4", n_layer=4, n_hc=1, n_embd=16, k=2, n_expert=16)
E.tokenizer_for = lambda g: type("V", (), {"decode": lambda self, ids: "", "__len__": lambda self: 100})()
E.read_gguf_meta = lambda g: {"tokenizer.ggml.eos_token_id": 0}


def ds4_server(request):
    body = json.loads(request.content)
    if request.url.path == "/v1/scope/tokenize":
        return httpx.Response(200, json={"tokens": [5, 6]})
    Path(body["scope"]["trace"]).write_bytes(b"capture so far")
    ready.write_text(body["scope"]["trace"])
    time.sleep(120)
    return httpx.Response(500)


E.Ds4Engine.open = classmethod(lambda cls, served, url=None: cls(served, Path("/m.gguf"), "http://ds4",
                                                                 transport=httpx.MockTransport(ds4_server)))
gpu.main(argv)
'''


def test_the_ds4_child_as_a_process_holds_the_lock_and_cleans_up_when_stopped(home, tmp_path):
    """The child inherits the job's lock (pass_fds) and keeps it while it runs; a cancel (SIGTERM) unwinds, so the
    engine removes its request files; a child whose job died exits on its own (the watchdog; its files are left for
    the engine's sweep)."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path
    import metalscope
    from metalscope import gpulock
    write_concept("fake-ds4", n=6)                  # the texts of the checkpoint the child measures
    script = tmp_path / "ds4_child.py"
    script.write_text(DS4_CHILD)
    env = dict(os.environ, METALSCOPE_HOME=str(home), PYTHONPATH=str(Path(metalscope.__file__).parents[1]),
               PYTHONUNBUFFERED="1")
    argv = ["fake-ds4", "toy", "--engine", "ds4", "--url", "http://proxy.invalid"]

    def wait_for(cond, timeout=60.0):
        end = time.monotonic() + timeout
        while not cond() and time.monotonic() < end:
            time.sleep(0.05)
        return cond()

    # A cancel: SIGTERM while a measurement request is in flight.
    ready = tmp_path / "ready-1"
    lock = gpulock.GpuLock(gpulock.Holder("learning toy")).acquire()
    child = subprocess.Popen([sys.executable, str(script), str(ready), *argv], env=env, cwd=tmp_path, pass_fds=(lock.fd,),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        lock.release()                                                   # the job's own descriptor closed
        assert wait_for(ready.exists), child.stdout.read1().decode() if child.poll() is not None else "no request"
        trace = Path(ready.read_text())
        assert trace.exists() and gpulock.held().what == "learning toy"   # the child still holds the lock
        os.kill(child.pid, signal.SIGTERM)
        assert child.wait(timeout=30) != 0
        assert not trace.exists() and gpulock.held() is None             # its files removed, the lock free
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    # The job dies (SIGKILL, no finally): the child notices its parent is gone and exits.
    ready = tmp_path / "ready-2"
    job = subprocess.Popen([sys.executable, "-c", "import pathlib, subprocess, sys, time\n"
                            "p = subprocess.Popen(sys.argv[1:], start_new_session=True)\n"
                            "print(p.pid, flush=True)\n"
                            f"while not pathlib.Path({str(ready)!r}).exists(): time.sleep(0.05)\n",
                            sys.executable, str(script), str(ready), *argv], env=env, cwd=tmp_path,
                           stdout=subprocess.PIPE, text=True)
    pid = int(job.stdout.readline())

    def gone():
        try:
            os.kill(pid, 0)
            return False
        except ProcessLookupError:
            return True
    try:
        assert job.wait(timeout=60) == 0 and ready.exists()
        assert wait_for(gone, timeout=15)                                 # the watchdog checks once a second
        assert Path(ready.read_text()).exists()                           # os._exit: left for the sweep
    finally:
        if not gone():
            os.kill(pid, signal.SIGKILL)


def test_the_learn_child_on_ds4_measures_the_served_model(home, monkeypatch, capsys):
    """concept learn's GPU child with --engine ds4 (the model ds4 serves, here the numpy engine): no copy loaded, the
    whole run on one engine, a profile with its learn section; every stage it reports is one the ds4 window weighs;
    an engine that cannot open is one line."""
    import signal
    from metalscope import engines
    from metalscope.engines import ds4 as ds4_engine
    from metalscope.learn import gpu, job as J, state
    write_norms("fake", range(4), value=1.0)
    write_concept("fake", n=6)
    opened, handlers = [], []
    engine = FakeEngine()
    monkeypatch.setattr(ds4_engine.Ds4Engine, "open", classmethod(lambda cls, served, url=None: opened.append((served, url)) or engine))
    monkeypatch.setattr(gpu.signal, "signal", lambda sig, fn: handlers.append(sig))      # not this test process's SIGTERM
    gpu.main(["fake", "toy", "--engine", "ds4", "--url", "http://proxy", "--sweep-layers", "0,2", "--strengths", "0.5,1"])
    assert opened == [("fake", "http://proxy")] and handlers == [signal.SIGTERM]
    evs = [json.loads(l[2:]) for l in capsys.readouterr().out.splitlines() if l.startswith("@@")]
    stages = [e["stage"] for e in evs if "stage" in e]
    assert stages[0] == "connecting" and stages[-1] == "done" and "loading the model" not in stages
    assert set(stages) - {"done"} <= set(J.LearnJob.DS4_WEIGHTS)
    prof = json.loads(models.profile_path("fake", "toy").read_text())
    assert prof["learn"]["sweep_layers"] == [0, 2] and prof["inputs"]["strengths"] == [0.5, 1.0]
    assert evs[-1]["profile"] == str(models.profile_path("fake", "toy"))

    # A push that works one way only (pushing away raises it too): learn still calibrates in conversation, whose verdict
    # is the concept's; no keep/drop anywhere.
    from metalscope.learn import calibrate as K
    calibrated = []

    def calibrate(name, model, server=None, log=None, on_progress=None):
        calibrated.append(name)
        return dict(version=4, default_dose=0.4, good=15, turns=18, reason="15/18 good @0.4")
    monkeypatch.setattr(K, "calibrate", calibrate)
    p = models.profile_path("fake", "toy")
    prof = json.loads(p.read_text())
    assert prof["learn"]["layer"] is not None                              # a steer layer was found
    row = lambda s, lo, chosen: dict(strength=s, concept=lo, control=0.1, effect=lo - 0.1, mass_kept=0.95, chosen=chosen,
                                     control_chosen=0.25, base_chosen=0.25, flips_over_control=chosen - 0.25)
    prof["learn"]["push"] = dict(dose=0.4, n_against=8, at_dose=row(0.4, 1.2, 0.75), at_quarter=row(0.1, 0.3, 0.25),
                                 away=row(-0.4, 0.8, 0.5), curve=[])
    p.write_text(json.dumps(prof))
    # Only after_gpu runs here: a plan for texts it wrote (these 6 pairs would not pass learn_plan's checks).
    plan = dict(name="toy", model="fake", source="write", texts=None, description="d", writer="fake", corpus=None,
                engine="ds4", read=None, taken=None, notes=[])
    j = J.LearnJob(plan, None, "fake", url="http://proxy", job="toy-1")
    j.state["profile"] = str(p)
    j.after_gpu()
    res = j.state["result"]
    assert calibrated == ["toy"] and res["steerable"] is True and res["reason"] is None
    assert res["calibration"]["reason"] == "15/18 good @0.4"
    assert res["push"]["away"] == "Steering away from toy: picked more often too, 25% to 50% (expected less often)."
    assert any(e.startswith("push at dose 0.4 on 8 held-out prompts: +1.10 log-odds") for e in res["evidence"])
    assert not re.search(r"keep|drop|decision|ok: |fails: ", json.dumps(res))
    assert "status" not in json.loads((models.concept_dir("fake", "toy") / "concept.json").read_text())
    assert state.outcome_text(dict(outcome="done", result=res)) == "15/18 good @0.4"

    # Every item saturated (2 already chosen, 1 never): no sweep, ladder or test; the profile keeps presence and the
    # reason, and learn stops there: can't be steered, no calibration.
    write_concept("fake", name="sure", n=6)
    monkeypatch.setattr(gpu.decide, "item_probs", lambda items, logodds, n: [0.99, 0.999, 0.01])
    gpu.main(["fake", "sure", "--engine", "ds4", "--url", "http://proxy"])
    out = capsys.readouterr().out
    stages = [e["stage"] for e in (json.loads(l[2:]) for l in out.splitlines() if l.startswith("@@")) if "stage" in e]
    assert stages[-2:] == ["measuring the items", "done"]
    why = ("can't be steered: fake already picks the sure option on 2 of 3 items unsteered; rephrase the contrast so "
           "the model doesn't already do it (or the other side), then learn it again under a new name")
    assert f"[sure] {why}; no sweep, ladder or test" in out
    prof = json.loads(models.profile_path("fake", "sure").read_text())
    assert prof["learn"]["untestable"] == why and not prof.get("reader") and prof["stats"]
    plan = dict(plan, name="sure")
    j = J.LearnJob(plan, None, "fake", url="http://proxy", job="sure-1")
    j.state["profile"] = str(models.profile_path("fake", "sure"))
    j.after_gpu()
    res = j.state["result"]
    assert res["steerable"] is False and res["reason"] == why and res["testable"] is False and "calibration" not in res
    assert calibrated == ["toy"] and j.step("calibrate")["summary"] == why and res["steer"] is None
    assert res["evidence"][0].startswith("presence: held-out d′ ")
    assert not re.search(r"keep|drop|decision", json.dumps(res))
    assert "status" not in json.loads((models.concept_dir("fake", "sure") / "concept.json").read_text())
    assert state.outcome_text(dict(outcome="done", result=res)) == "can't be steered"

    def refuse(cls, served, url=None):
        raise engines.EngineUnavailable("the ds4 proxy at http://proxy serves other, not fake")
    monkeypatch.setattr(ds4_engine.Ds4Engine, "open", classmethod(refuse))
    with pytest.raises(SystemExit, match="^the ds4 proxy at http://proxy serves other, not fake$"):
        gpu.main(["fake", "toy2", "--engine", "ds4"])
    with pytest.raises(SystemExit):
        gpu.main(["fake", "toy"])                                    # mlx needs --model

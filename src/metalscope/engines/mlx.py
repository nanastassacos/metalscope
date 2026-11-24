"""The MLX engine: a copy of the checkpoint loaded in this process, one row per forward.

Every forward is one prompt from an empty cache: deterministic, where a batched row is not (on Flash Next the row
count picks another bf16 matmul kernel, and routing turns that rounding into other experts by L0). Steering
is a SteerSession (steer.py) installed for the call with one plan, ``positions=all``: the adds act on the prompt
forward and on every generated token, as when served. Residuals are grabbed by a post-hook on each decoder layer,
after the steering add there, in the space the hook writes (the streams on hyper-connection models, the deferred
write applied on Flash Next), flattened to ``[T, W]``. A layer is what a trace records (adapters.layer_count): on
Inkling 2 a layer visit, a layer call named by its visit as the steer session names it (adapters.visit_counter),
its cache the model's own (``make_cache``: each visit's state).
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from .. import config, models, routing
from ..hooks import add_hook
from ..jlens.fit import PREFIX_CHUNK, _arrays
from . import END_TOKENS, NEXT_TOP, SINK, Engine, NextLp, Shape, Tokenizer, checked, hc_of

PREFILL_CHUNK = PREFIX_CHUNK  # generate's prompt tokens per prefill forward (only longer prompts are split)
LOOKUP_CHUNK = 512            # next_logprobs' most tokens per forward: [512, vocab] float32 logits, 0.5 GB on Flash Next
TOP_ROWS = 128                # rows per top-choices pick: [128, vocab] float32 negation and uint32 indices, 0.25 GB on Flash Next


def load_model(path: str, loader: str = "mlx_lm"):
    """The model and tokenizer, loaded by mlx_lm or (``omlx``) oMLX's own loader (models only oMLX can load; run
    under scripts/omlx-python)."""
    if loader == "omlx":
        from ..omlx.load import load_omlx_model
        return load_omlx_model(path)
    from mlx_lm import load
    return load(path)


def _eos_ids(tok, more=()) -> set[int]:
    ids = set()
    v = getattr(tok, "eos_token_id", None)
    if isinstance(v, int):
        ids.add(v)
    for s in (*END_TOKENS, *more):
        try:
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        except Exception:
            pass
    return ids


def token_ids(out) -> list[int]:
    """Token ids from whatever apply_chat_template returned (a list, or a BatchEncoding / dict with ``input_ids``)."""
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    out = list(out)
    if out and isinstance(out[0], (list, tuple)):
        out = list(out[0])
    return [int(i) for i in out]


class MlxTokenizer(Tokenizer):
    """A Hugging Face tokenizer (mlx_lm's wrapper) as the engine's: chat turns through its chat template. ``answer``:
    a template with no thinking switch (adapters.answer_marks): the text that opens the reply's answer and the token
    that ends it (an end of reply here). That token also ends a thinking message: ``generate`` runs only on prompts
    opened at the answer (corpus.render_item, thinking=False); one with thinking left on would stop after the thought."""

    def __init__(self, hf, answer: Optional[tuple[str, str]] = None):
        self.hf, self.answer = hf, answer
        self.eos = _eos_ids(hf, answer[1:] if answer else ())

    def encode(self, text: str) -> list[int]:
        return list(self.hf.encode(text, add_special_tokens=False))

    def chat(self, user: str, thinking: Optional[bool] = False) -> list[int]:
        """``thinking=False`` asks the template to switch thinking off, so the next token is the answer (the task's
        prompts), or where it has no switch opens the answer after its prompt (``answer``); None leaves the template's
        default (corpus texts with ``--chat``). Without a chat template the text is encoded as is."""
        if not hasattr(self.hf, "apply_chat_template"):
            return self.encode(user)
        turn = [{"role": "user", "content": user}]
        if thinking is None:
            return token_ids(self.hf.apply_chat_template(turn, add_generation_prompt=True, tokenize=True))
        if self.answer and not thinking:
            return token_ids(self.hf.apply_chat_template(turn, add_generation_prompt=True, tokenize=True)) + \
                self.encode(self.answer[0])
        try:
            out = self.hf.apply_chat_template(turn, add_generation_prompt=True, tokenize=True, enable_thinking=thinking)
        except TypeError:                  # a tokenizer whose template call takes no such switch
            out = self.hf.apply_chat_template(turn, add_generation_prompt=True, tokenize=True)
        return token_ids(out)

    def decode(self, ids: list[int]) -> str:
        return self.hf.decode(list(ids))


class MlxEngine(Engine):
    kind = "mlx"

    def __init__(self, model, tokenizer, served: str):
        from .. import adapters
        from ..adapters.generic import text_model
        self.model, self.served, self.checkpoint = model, served, models.checkpoint_of(served)
        self.tok = MlxTokenizer(tokenizer, adapters.answer_marks(model))
        self.layers = text_model(model).layers
        self.n_layers = adapters.layer_count(model)                   # Inkling 2: its layer visits
        self.block, self.steer_layers = adapters.steer_block(model), adapters.steer_layers(model)

    @classmethod
    def load(cls, path: str, served: str, loader: str = "mlx_lm") -> "MlxEngine":
        return checked(cls(*load_model(path, loader), served))

    @cached_property
    def shape(self) -> Shape:
        """From one token's forward: W from layer 0's output (``hc_of``), d_model from the config's hidden size, else
        from what the embedding returns, the vocabulary from the logits."""
        from ..adapters.generic import d_model as d_model_of, text_model
        d_model = d_model_of(text_model(self.model))
        seen = {}

        def grab(_m, _a, _k, out):
            seen["shape"] = tuple((out[0] if isinstance(out, tuple) else out).shape)
        remove = add_hook(self.layers[0], grab)
        try:
            logits, _ = self._forward([0])
        finally:
            remove()
        width, hc = hc_of(seen["shape"], int(d_model))
        return Shape(self.n_layers, width, hc, int(d_model), int(logits.shape[-1]))

    @contextmanager
    def _steering(self, adds: Optional[dict[int, np.ndarray]]):
        """The block's forwards steered with ``adds``; fails closed: raises if the hook could not apply them."""
        if not adds:
            yield
            return
        from ..steer import SteerPlan, SteerSession
        fails: list[str] = []
        session = SteerSession(lambda _rid, why: fails.append(why))
        session.install(self.model)
        try:
            session.set_rows([(0, "measure", SteerPlan(dict(adds), "all", {}))], 1, "decode")
            yield
            if fails:
                raise RuntimeError(f"steering failed: {fails[-1]}")
        finally:
            session.close()

    def _forward(self, ids: list[int], grab: Optional[list[int]] = None, logits: bool = True):
        """``[V]`` logits at the last position (None without ``logits``) and, for ``grab`` layers, the residual each
        wrote ``[T, W]`` float32, as seen after every hook installed before this call (so a steering add is included).
        Every layer call is counted (adapters.visit_counter), so a layer run several times a token (Inkling 2) is
        grabbed at the visit asked."""
        from ..adapters import visit_counter
        from ..adapters.qwen4_exp import _apply_write
        grabbed, removers, want, visit = {}, [], set(grab or ()), visit_counter(self.model)
        for i, layer in enumerate(self.layers if want else ()):
            def hook(_m, _a, _k, out, i=i):
                if (l := visit(i)) in want:
                    h = out[0] if isinstance(out, tuple) else out
                    grabbed[l] = _apply_write(out) if isinstance(out, tuple) and len(out) == 3 else h
            removers.append(add_hook(layer, hook))
        m = self.model
        try:
            out = m(mx.array([ids]), cache=m.make_cache()) if hasattr(m, "make_cache") else m(mx.array([ids]))
        finally:
            for r in removers:
                r()
        last = getattr(out, "logits", out)[0, -1] if logits else None
        mx.eval(*([last] if logits else []), *grabbed.values())
        return last, {l: np.array(v[0].reshape(v.shape[1], -1).astype(mx.float32)) for l, v in grabbed.items()}

    def residuals(self, prompts_ids, layers, sink=SINK, adds=None):
        layers = list(layers)
        mean = np.zeros((len(prompts_ids), len(layers), self.shape.width), np.float32)
        last = np.zeros_like(mean)
        with self._steering(adds):
            for i, ids in enumerate(prompts_ids):
                _, grabbed = self._forward(ids, layers, logits=False)
                for j, l in enumerate(layers):
                    a = grabbed[l]
                    last[i, j], mean[i, j] = a[-1], a[min(sink, len(a) - 1):].mean(0)
        return mean, last

    def logprobs(self, ids, token_ids, adds=None):
        with self._steering(adds):
            row, _ = self._forward(ids)
        r = row.astype(mx.float32)
        return np.array((r - mx.logsumexp(r))[mx.array(list(token_ids))])

    @cached_property
    def lookups(self) -> list:
        """The model's n-gram lookup modules (Flash Next's PLE, a ``ple`` child of a decoder layer)."""
        return [layer["ple"] for layer in self.layers if "ple" in layer]

    def lookup_chunk(self) -> int:
        """next_logprobs' rows per forward: LOOKUP_CHUNK, or under oMLX's expert offload as many as keep every MoE call
        within the smallest layer's resident slots (rows * top-k), its exact path: over them a call can go expert-major
        (when its distinct experts outnumber the slots) and the reading then depends on which experts are resident
        (two 512-row readings 0.14 nats apart on Flash Next, 2026-11-18). Read when called: oMLX shrinks the slots
        under memory pressure."""
        caches = [m.cache for _, m in self.model.named_modules()
                  if getattr(getattr(m, "cache", None), "moe_offload_cache", False)]
        if not caches:
            return LOOKUP_CHUNK
        rule = routing.model_rule(enumerate(self.layers))
        # Top-k unknown: oMLX's floor where the cache has one (at least the config's top-k), else one row a forward.
        return max(1, min(LOOKUP_CHUNK, *(c.capacity // (rule.k if rule else getattr(c, "floor", c.capacity))
                                          for c in caches)))

    def chunk_bytes(self, rows: int) -> int:
        """The most next_logprobs holds for a chunk of ``rows`` beyond the model and its cache: the chunk's [rows, vocab]
        logits and their float32 copy, and a TOP_ROWS pick's float32 negation and uint32 indices."""
        from ..adapters.generic import vocab_size
        return vocab_size(self.model) * 8 * (rows + TOP_ROWS)

    def next_logprobs(self, ids, start, lookup=True, run=None, rows=None):
        """In chunks of ``rows`` (default lookup_chunk()) through one fresh cache of its own (rows before ``start - 1``
        only fill it: their logits are never computed). ``run(fn)`` runs each chunk (oMLX: on its engine's executor,
        between its own steps; default: here); with ``lookup`` False each lookup module's output is zeroed inside it
        only. Each row's top choices are picked inside its chunk: no [rows, vocab] array outlives one."""
        if not 0 < start < len(ids):
            raise ValueError(f"start {start} is not inside the {len(ids)} ids")
        if not lookup and not self.lookups:
            raise ValueError("this model has no n-gram lookup")
        cache = self.model.make_cache()
        hits = []

        def zeroed(_m, _a, _k, out):
            hits.append(1)
            return mx.zeros_like(out)

        def chunk(lo: int, hi: int):
            removers = [] if lookup else [add_hook(m, zeroed) for m in self.lookups]
            try:
                out = self.model(mx.array([ids[lo:hi]]), cache=cache)
            finally:
                for r in removers:
                    r()
            a, b = max(lo, start - 1), min(hi, len(ids) - 1)       # rows read: their next token is in ids
            if a >= b:
                mx.eval(_arrays([c.state for c in cache]))
                return None
            z = getattr(out, "logits", out)[0, a - lo:b - lo].astype(mx.float32)
            lse = mx.logsumexp(z, axis=-1)
            lp = mx.take_along_axis(z, mx.array(ids[a + 1:b + 1])[:, None], axis=-1)[:, 0] - lse
            mx.eval(z, lse, lp, _arrays([c.state for c in cache]))
            # The NEXT_TOP likeliest a row, TOP_ROWS rows at a time: argpartition's [rows, vocab] indices stay small.
            k, tops = min(NEXT_TOP, z.shape[-1]), []
            for s in range(0, z.shape[0], TOP_ROWS):
                zs = z[s:s + TOP_ROWS]
                i = mx.argpartition(-zs, k - 1, axis=-1)[:, :k]
                v = mx.take_along_axis(zs, i, axis=-1)
                o = mx.argsort(-v, axis=-1)
                i, v = mx.take_along_axis(i, o, axis=-1), mx.take_along_axis(v, o, axis=-1) - lse[s:s + TOP_ROWS, None]
                mx.eval(i, v)
                tops.append((np.array(i), np.array(v)))
            return np.array(lp), np.concatenate([t[0] for t in tops]), np.concatenate([t[1] for t in tops])

        run, rows = run or (lambda fn: fn()), rows or self.lookup_chunk()
        got = [run(lambda lo=lo: chunk(lo, min(lo + rows, len(ids)))) for lo in range(0, len(ids), rows)]
        if not lookup and not hits:
            raise RuntimeError("the lookup's output was never reached: nothing was left out")
        got = [g for g in got if g is not None]
        return NextLp(*(np.concatenate([g[j] for g in got]) for j in range(3)))

    def generate(self, ids, adds, max_tokens):
        """Greedy, with a KV cache; the adds act on the prompt forward and on every generated token."""
        eos = self.tok.eos
        toks: list[int] = []
        with self._steering(adds):
            cache = self.model.make_cache()
            # A long prompt (learn's grading prompt, ~5k tokens; every measurement prompt is far shorter) is prefilled
            # in chunks: one forward would hold [T, vocab] logits, 1.5-3 GB at 5k tokens. Evaluating only the cache leaves
            # each early chunk's logits uncomputed.
            start = 0
            while len(ids) - start > PREFILL_CHUNK:
                self.model(mx.array([ids[start:start + PREFILL_CHUNK]]), cache=cache)
                mx.eval(_arrays([c.state for c in cache]))
                start += PREFILL_CHUNK
            out = self.model(mx.array([ids[start:]]), cache=cache)
            for _ in range(max_tokens):
                nxt = int(mx.argmax(getattr(out, "logits", out)[0, -1]).item())
                if nxt in eos:
                    return toks, True
                toks.append(nxt)
                out = self.model(mx.array([[nxt]]), cache=cache)
        return toks, False


def local_mlx_lm() -> bool:
    """Whether this Python has mlx_lm (a checkout's does; a plain install's does not, and a job then runs mlx_lm
    under oMLX's bundled Python, which has it)."""
    return importlib.util.find_spec("mlx_lm") is not None


def pick_loader(path: str) -> str:
    """"mlx_lm" when mlx_lm has the checkpoint's architecture (the 2B, Qwen3.6), this Python's or else oMLX's
    bundled one's; else "omlx": oMLX's own loader under its bundled Python, as every Flash Next profile has run
    (mlx_lm has no ``qwen4_exp``). oMLX's loader takes its fast kernels, which a lens fit cannot differentiate
    unless an adapter switches them off (Flash Next's does)."""
    model_type = config.read_json(Path(path).expanduser() / "config.json", {}).get("model_type")
    if not model_type:
        return "omlx"
    module = f"mlx_lm.models.{model_type}"
    try:
        if local_mlx_lm():
            return "mlx_lm" if importlib.util.find_spec(module) is not None else "omlx"
    except (ImportError, ValueError):
        return "omlx"
    from ..omlx import install as omlx_install
    try:
        r = omlx_install.bundled_run(omlx_install.run_app(),
                                     f"import importlib.util as u; print(u.find_spec({module!r}) is not None)")
    except OSError:
        return "omlx"
    return "mlx_lm" if r.returncode == 0 and r.stdout.strip() == "True" else "omlx"

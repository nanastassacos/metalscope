"""Fitting a Jacobian lens in MLX (the estimator of anthropics/jacobian-lens).

    J_l = mean over prompts and valid source positions p of
          sum_{p' >= p valid} d h_final[p'] / d h_l[p]

For each output dimension i a one-hot cotangent is placed on dimension i at
every valid target position at once; one backward pass then yields row i of
every J_l. ``dim_batch`` output dimensions run together by replicating the
prompt along the batch axis.

To get d h_final / d h_l for all layers from one backward pass without
re-implementing any architecture's forward, a zero "probe" tensor is added to
each source layer's output through the instance hooks (hooks.py); the gradient
with respect to probe l is exactly d h_final / d h_l. The same code therefore
fits conventional decoders and Flash Next (whose layer output is a deferred
hyper-connection write; the probe is added to its residual streams).

Deep positions. A prompt may be {"text" or "ids", "window": W}: only the last
W tokens are sources and targets, and the prefix before them is run once,
without gradients, into a cache that every backward pass starts from. The
model is causal, so nothing in the prefix depends on the window and the
gradients are exactly those of the full sequence restricted to the window,
while a pass costs O(W) instead of O(T). This is what lets the lens see
positions thousands of tokens into a chat for the price of a short prompt.

What "h_final" is depends on the adapter:
  generic    output of the last decoder block (pre final norm), as in the
             reference; readout = final norm + LM head.
  qwen4_exp  the mixer output the LM head reads; readout = LM head. Sources
             are the 4-stream residuals, so J_l is [hidden, hc * hidden].
  generic, DeepSeek V4 (hyper-connection streams, ``hyper_streams``): layers
             return [B, T, hc, D]; h_final is ``hc_head``'s output, the
             residual the final norm + LM head read. Sources are the streams
             flattened, so J_l is [hidden, hc * hidden], as for Flash Next,
             and capture applies it to the streams.
  inkling2   layers are layer visits; h_final is what the final norm reads.
             Sources are each visit's state (finished blocks but the first,
             running sum, the loop's message bank), so J_v is [hidden, n_v * hidden]
             and its width differs by visit. The adapter runs the forward
             (its ``differentiable``).
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from .. import adapters
from ..adapters.generic import hyper_streams, text_model
from ..hooks import add_hook
from . import SKIP_FIRST_N_POSITIONS, Lens

PREFIX_CHUNK = 1024


class _Probes:
    """Adds per-layer probe tensors to decoder outputs while a fit runs; ``shapes``: layer -> (its output's width per
    position, dtype), set by each forward."""

    def __init__(self, model, sources: list[int], hyper: bool):
        self.inner = text_model(model)
        self.eps: dict[int, mx.array] = {}
        self.shapes: dict[int, tuple] = {}
        self.final: Optional[mx.array] = None
        self.removers = []
        last = len(self.inner.layers) - 1
        for l in sources:
            self.removers.append(add_hook(self.inner.layers[l], self._adder(l)))
        if not hyper:
            def grab(_m, _a, _k, out):
                self.final = out[0] if isinstance(out, tuple) else out
            # DeepSeek V4: the target is what hc_head mixes the last layer's streams down to.
            target = self.inner["hc_head"] if hyper_streams(self.inner) is not None else self.inner.layers[last]
            self.removers.append(add_hook(target, grab))

    def _adder(self, l: int):
        def add(_m, _a, _k, out):
            o = out[0] if isinstance(out, tuple) else out
            self.shapes[l] = (int(np.prod(o.shape[2:])), o.dtype)        # V4's [B, T, hc, D]: hc * D
            e = self.eps.get(l)
            if e is None:
                return None
            if isinstance(out, tuple):  # deferred hyper-connection write: (streams, branch, gate)
                return (out[0] + e,) + tuple(out[1:])
            # DeepSeek V4's [B, T, hc, D] streams take the probe flattened, as [B, T, hc * D].
            return out + (e.reshape(out.shape) if e.ndim != out.ndim else e)
        return add

    def close(self):
        for r in self.removers:
            r()


def unfuse_rope(model) -> Callable[[], None]:
    """Switch mlx-vlm's fused mRoPE kernel to its pure-MLX path; returns an undo.

    The kernel indexes position ids as [3, B, T], but the text path passes
    [3, 1, T] (one row, broadcast), so with B > 1 every row (and row 0's
    second and third axes) is rotated by out-of-range reads. Serving runs one
    row or per-row positions; the fit replicates prompts along the batch, so
    it must not use the kernel. The pure path broadcasts correctly.
    """
    mods = [m for _, m in text_model(model).named_modules() if getattr(m, "fused_apply", False)]
    for m in mods:
        m.fused_apply = False

    def undo():
        for m in mods:
            m.fused_apply = True
    return undo


@contextmanager
def differentiable(model, sources: list[int], hyper: bool):
    """A model set up for gradients through probes at ``sources`` (adapters.differentiable takes an adapter's own
    instead when it has one: Inkling 2): yields the probes and ``forward(ids, cache=None)``, which returns the residual
    the readout reads (the last layer's output, captured by the probes, or, with ``hyper``, the model's own output).
    Custom Metal kernels have no VJP: mlx-lm switches its kernels (e.g. the GatedDeltaNet recurrence) to plain ops in
    training mode, and these models have no dropout, so training mode changes nothing else; replicated rows need the
    fused mRoPE kernel off (``unfuse_rope``). Both are restored on exit, and the probes removed."""
    inner = text_model(model)
    probes = _Probes(model, sources, hyper)
    was_training = model.training
    model.train(True)
    restore_rope = unfuse_rope(model)
    def forward(ids: mx.array, cache=None) -> mx.array:
        # Some architectures (Flash Next's attention, oMLX's Qwen3.5) need a cache
        # object even for one fresh pass; give each forward its own empty one.
        if cache is None and hasattr(model, "make_cache"):
            cache = model.make_cache()
        # mx.vjp refuses mx.async_eval inside the forward it differentiates ("[async_eval] Not allowed inside a graph
        # transformation"), so async_eval is a no-op while the graph is built, for every model: the backward needs that
        # graph, and the prefix's forwards evaluate their cache themselves. oMLX's DeepSeek V4 is a model that calls
        # it, at the end of every forward on its cache arrays (_materialize_cache_arrays).
        cut = mx.async_eval
        mx.async_eval = lambda *_a, **_k: None
        try:
            out = inner(ids, cache=cache)
        finally:
            mx.async_eval = cut
        return out if hyper else probes.final

    try:
        yield probes, forward
    finally:
        probes.close()
        model.train(was_training)
        restore_rope()


def _make_cache(model):
    if hasattr(model, "make_cache"):
        return model.make_cache()
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(model)


def _fresh(st):
    """A cache state with fresh lists all the way down. ArraysCache.state is
    its live slot list (or, in newer mlx-lm, a tuple holding it), which every
    later forward writes into; sharing it would corrupt a snapshot."""
    return type(st)(_fresh(x) for x in st) if isinstance(st, (list, tuple)) else st


def _arrays(st) -> list:
    return [a for x in st for a in _arrays(x)] if isinstance(st, (list, tuple)) else \
        [st] if isinstance(st, mx.array) else []


def prefix_cache(model, prefix: list[int], b: int, chunk: int = PREFIX_CHUNK):
    """Run the prefix once (no gradients, no probes set) replicated ``b``
    times; return the cache and a snapshot to ``restore`` before each pass."""
    inner = text_model(model)
    cache = _make_cache(model)
    # Inkling 2's text model takes its embeddings and draft captures with no default (oMLX 0.8.4 on).
    extra = dict(input_embeddings=None, capture={}) if adapters.find_adapter(model).NAME == "inkling2" else {}
    for a in range(0, len(prefix), chunk):  # bounded attention temporaries
        inner(mx.array([prefix[a:a + chunk]] * b), cache=cache, **extra)
        mx.eval(_arrays([c.state for c in cache]))
    snap = [_fresh(c.state) for c in cache]
    mx.eval(_arrays(snap))
    return cache, snap


def restore(cache, snap):
    # The setter installs the prefix arrays. KV caches reallocate on the next
    # update; ArraysCache writes into the list it was given, so give it a copy.
    for c, st in zip(cache, snap):
        c.state = _fresh(st)


def _prompt_ids(tokenizer, prompt, max_seq_len: int) -> tuple[list[int], int]:
    """(token ids, index of the first source/target position)."""
    if isinstance(prompt, dict):
        ids = prompt.get("ids") or tokenizer.encode(prompt["text"], add_special_tokens=False)
        if prompt.get("window"):
            return list(ids), max(0, len(ids) - int(prompt["window"]))
        return list(ids)[:max_seq_len], 0
    return tokenizer.encode(prompt, add_special_tokens=False)[:max_seq_len], 0


def checkpoint_done(checkpoint: str | Path) -> int:
    """How many prompts a checkpoint's sums hold: the count saved inside fit_state.safetensors; 0 without a state."""
    ck = Path(checkpoint)
    if not (ck / "fit_state.safetensors").exists():
        return 0
    _, info = mx.load(str(ck / "fit_state.safetensors"), return_metadata=True)
    return int(info["done"])


def fit(model, tokenizer, prompts: list, *, dim_batch: int = 64, max_seq_len: int = 128,
        source_layers: Optional[list[int]] = None, skip_first: int = SKIP_FIRST_N_POSITIONS,
        checkpoint: Optional[str | Path] = None, log: Callable[[str], None] = print,
        on_prompt: Optional[Callable[[dict, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None, dim_limit: Optional[int] = None,
        prefix_chunk: int = PREFIX_CHUNK) -> Lens:
    """``should_stop`` is asked before each prompt (e.g. a deadline); the fit
    then ends cleanly at a checkpoint. ``dim_limit`` fits only the first that
    many output dims (benchmarking batch sizes; the lens is then partial).
    ``prefix_chunk``: tokens per no-gradient prefix forward; smaller lowers the
    peak memory of deep windows at large ``dim_batch``."""
    adapter = adapters.find_adapter(model).NAME
    sources = source_layers if source_layers is not None else list(range(adapters.layer_count(model) - 1))
    sums: dict[int, mx.array] = {}
    done = 0
    ck = Path(checkpoint) if checkpoint else None
    if ck and (ck / "fit_state.safetensors").exists():
        state = mx.load(str(ck / "fit_state.safetensors"))
        sums = {int(k.split(".")[1]): v for k, v in state.items() if k.startswith("S.")}
        done = checkpoint_done(ck)
        log(f"resuming from {done} prompts")

    stats: dict = {}
    with adapters.differentiable(model, sources, adapter == "qwen4_exp") as (probes, forward):
        for n in range(done, len(prompts)):          # by index: ``prompts`` may build each one on demand (lenscorpus.Windows)
            if should_stop is not None and should_stop():
                log(f"stopping before prompt {n + 1}")
                break
            prompt = prompts[n]
            all_ids, w0 = _prompt_ids(tokenizer, prompt, max_seq_len)
            prefix, ids = all_ids[:w0], all_ids[w0:]
            T = len(ids)
            first = 0 if prefix else skip_first  # deep windows have no attention-sink positions
            if T < first + 2:
                continue
            t0 = time.perf_counter()
            mx.reset_peak_memory()
            t_prefix, passes = 0.0, 0
            valid = np.zeros(T, bool)
            valid[first:T - 1] = True
            vmask = mx.array(valid.astype(np.float32))[None, :, None]
            n_valid = float(valid.sum())

            # Shapes and dtypes of the sources (each forward sets them) and the target.
            probes.eps = {}
            h_final = forward(mx.array([(prefix + ids)[:32]]))
            mx.eval(h_final)
            shapes = probes.shapes
            d_out, out_dtype = h_final.shape[-1], h_final.dtype
            if not sums:
                sums = {l: mx.zeros((d_out, shapes[l][0]), dtype=mx.float32) for l in sources}

            pre = {}  # batch size -> (cache, snapshot) for the prefix
            for start in range(0, min(d_out, dim_limit or d_out), dim_batch):
                b = min(dim_batch, d_out - start)
                if prefix and b not in pre:
                    tp = time.perf_counter()
                    probes.eps = {}
                    pre = {b: prefix_cache(model, prefix, b, prefix_chunk)}
                    t_prefix += time.perf_counter() - tp
                x = mx.array([ids] * b)
                cot = np.zeros((b, T, d_out), np.float32)
                cot[np.arange(b)[:, None], np.nonzero(valid)[0][None, :], (start + np.arange(b))[:, None]] = 1.0
                cot = mx.array(cot).astype(out_dtype)
                zeros = [mx.zeros((b, T, shapes[l][0]), dtype=shapes[l][1]) for l in sources]

                def f(*eps):
                    cache = None
                    if prefix:
                        cache, snap = pre[b]
                        restore(cache, snap)
                    probes.eps = dict(zip(sources, eps))
                    try:
                        return [forward(x, cache)]
                    finally:
                        probes.eps = {}

                _, grads = mx.vjp(f, zeros, [cot])
                for l, g in zip(sources, grads):
                    # Mean over valid source positions -> rows start:start+b of J_l.
                    rows = (g.astype(mx.float32) * vmask).sum(axis=1) / n_valid
                    sums[l][start:start + b] = sums[l][start:start + b] + rows
                mx.eval(list(sums.values()))
                passes += 1
            done = n + 1
            where = f" at depth {w0}" if prefix else ""
            dt = time.perf_counter() - t0
            stats = dict(seconds=round(dt, 2), prefix_s=round(t_prefix, 2), passes=passes,
                         per_pass_s=round((dt - t_prefix) / max(1, passes), 3), peak_gb=round(mx.get_peak_memory() / 1e9, 2),
                         d_out=d_out, tokens=T, depth=w0)
            log(f"prompt {done}/{len(prompts)}: {T} tokens{where}, {dt:.1f}s "
                f"(prefix {t_prefix:.1f}s, {passes} passes x {(dt - t_prefix) / max(1, passes):.2f}s at B={dim_batch}; "
                f"peak {mx.get_peak_memory() / 1e9:.1f} GB, active {mx.get_active_memory() / 1e9:.1f} GB)")
            if dim_limit:
                continue   # benchmark: no checkpoint
            if ck:
                # Written beside and renamed over, with the count inside: a process killed mid-write (a cancel, a
                # deadline) leaves the last complete state, never a torn one or sums that disagree with the count.
                ck.mkdir(parents=True, exist_ok=True)
                tmp = ck / "fit_state.tmp.safetensors"
                mx.save_safetensors(str(tmp), {f"S.{l}": v for l, v in sums.items()}, metadata=dict(done=str(done)))
                os.replace(tmp, ck / "fit_state.safetensors")
                # The count beside it too: what lens_learn_plan reads to say a fit resumes, without loading the sums.
                (ck / "fit_state.json").write_text(json.dumps(dict(done=done)))
            if on_prompt:
                on_prompt(sums, done)
    count = max(1, done)
    # ``last``: the last prompt's time, passes and peak memory (what a batch-size probe reads).
    return Lens({l: (s / count).astype(mx.float16) for l, s in sums.items()},
                dict(n_prompts=count, adapter=adapter, max_seq_len=max_seq_len, dim_batch=dim_batch, last=stats))


_INDEX_OPS = ("argpartition", "argsort", "argmax", "argmin")


def no_index_gradients() -> None:
    """Expert routing and sparse-attention selection derive integer indices from differentiable scores, and MLX
    refuses to differentiate through the ops that make them. Indices never need gradients, so in a fitting process
    (only: this patches ``mlx.core`` for the whole process) they return stop_gradient'ed results. Dense models never
    call them on the gradient path, so it changes nothing there. Idempotent."""
    for name in _INDEX_OPS:
        orig = getattr(mx, name, None)
        if orig is None or getattr(orig, "_jfit_patched", False):
            continue

        def patched(*a, _f=orig, **k):
            return mx.stop_gradient(_f(*a, **k))
        patched._jfit_patched = True
        setattr(mx, name, patched)

"""Capture sessions: what to record, and how records travel off the GPU.

Hooks never call ``mx.eval``. They compute small lazy reductions and append
them to the session's pending list. The caller evaluates them together with
the step's own outputs (``session.pending_arrays()`` goes into the same
``mx.async_eval`` as the next token), so tracing adds GPU work but no extra
CPU/GPU synchronisation. Host transfer happens later: in the server once the
next step is submitted (live.TraceManager._host), offline in ``drain()``.

Decode is bound by kernel launches, not FLOPs, so for short steps (decode)
hooks only stash each layer's tensor; ``pending_arrays()`` then runs every
reduction once over all layers stacked, a handful of kernels instead of a
dozen per layer. Long steps (prefill chunks) reduce per layer as they go,
since stacking them would cost a lot of memory for little gain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np

from .hooks import add_hook


@dataclass
class CaptureConfig:
    # Decoder layer indices to record residual reductions for; None = all.
    layers: Optional[list[int]] = None
    residual_norm: bool = True
    # MoE router: the experts each token went to and the weights they were
    # mixed with, by the model's own rule and k (routing.py; the adapter sets
    # ``session.routing``).
    router: bool = False
    # Final logits: top-k token ids and log-probs, plus entropy.
    logits_topk: int = 0
    # Logit lens: read each recorded residual out through the adapter's
    # readout (final norm + LM head); records the top-k per layer.
    logit_lens_topk: int = 0
    # Restrict the (expensive) lens to every Nth layer plus the last K, or to
    # ``logit_lens_layers`` when given (``reads_lens``).
    lens_every: int = 1
    lens_last: int = 0
    logit_lens_layers: Optional[list[int]] = None
    # Jacobian lens (jlens.py): layer -> J_l [d, d]; full top-k readout at
    # ``jlens_layers`` (default: every layer with a J).
    jlens: Optional[dict] = None
    jlens_topk: int = 0
    jlens_layers: Optional[list[int]] = None
    # A step longer than this (a prompt chunk) reads both lenses on its last
    # ``lens_tail`` rows only, the rows before recorded empty; 0 reads every row.
    lens_tail: int = 0

    def reads_lens(self, i: int, n_layers: Optional[int]) -> bool:
        """Whether the logit lens reads layer ``i``: one of ``logit_lens_layers`` when given, else every
        ``lens_every``-th and the last ``lens_last``."""
        if self.logit_lens_layers is not None:
            return i in self.logit_lens_layers
        return i % self.lens_every == 0 or (n_layers is not None and i >= n_layers - self.lens_last)


STACKED = -2  # Record.layer for a value stacked over ``Record.layers``
BATCH_MAX_TOKENS = 16
# Vocab-sized reductions run in token chunks: a 2k-token prefill over a 248k
# vocabulary would otherwise create several 2 GB float32 temporaries.
LOGIT_CHUNK_TOKENS = 256
# Lens reads (every read layer, both lenses) go through the LM head together, up to this many rows (layers x batch x
# tokens) at a time: decode reads the ~0.7 GB head once per token instead of once per layer (8 ms vs 51 ms per token for
# both lenses on Flash Next, M5 Max, in isolation). A step with this many rows or more reads one layer per call.
LENS_BATCH_ROWS = 64

# A vocabulary-wide top-k (argpartition over ~250k) dominated a lens read: ~0.13 ms per row on Flash Next, so a depth-4
# MTP verify window's 180 reads took 24 ms against 6 ms for a plain step's 36 (the head alone, in isolation).
# _topk_ids finds the same k from the TOPK_BLOCK-wide blocks with the largest maxima: 9 ms for those 180 rows.
# docs/measurements.md's bench table has what tracing costs a served model.
TOPK_BLOCK = 512


def _topk_ids(z: mx.array, k: int) -> mx.array:
    """The ids of the k largest entries of z's last axis (unordered; ties may pick a different equal entry than a
    plain argpartition). The k largest lie in the k blocks with the largest maxima, so only those are searched."""
    V = z.shape[-1]
    if V < 8 * TOPK_BLOCK or k * TOPK_BLOCK >= V:
        return mx.argpartition(-z, kth=k - 1, axis=-1)[..., :k]
    lead = z.shape[:-1]
    flat = z.reshape(-1, V)
    pad = (-V) % TOPK_BLOCK
    if pad:
        flat = mx.pad(flat, [(0, 0), (0, pad)], constant_values=-mx.inf)
    blocks = flat.reshape(flat.shape[0], -1, TOPK_BLOCK)
    best = mx.argpartition(-blocks.max(axis=-1), kth=k - 1, axis=-1)[..., :k]                     # [R, k] blocks
    cand = mx.take_along_axis(blocks, best[..., None], axis=1).reshape(flat.shape[0], k * TOPK_BLOCK)
    j = mx.argpartition(-cand, kth=k - 1, axis=-1)[..., :k]
    ids = mx.take_along_axis(best, j // TOPK_BLOCK, axis=-1) * TOPK_BLOCK + j % TOPK_BLOCK
    return ids.reshape(*lead, k)


def _stacked(xs: list, missing: Optional[mx.array] = None) -> Optional[mx.array]:
    """``xs`` stacked, each None among them as zeros shaped like ``missing``; None when every one is None, or when one
    is and no ``missing`` is given."""
    if all(x is None for x in xs) or (missing is None and any(x is None for x in xs)):
        return None
    zero = mx.zeros_like(missing) if any(x is None for x in xs) else None
    return mx.stack([zero if x is None else x for x in xs])


@dataclass
class Record:
    kind: str          # "resid_norm", "router_ids", ...
    layer: int         # -1 for model-level records (logits), STACKED for [L, ...] values
    step: int
    value: Any         # mx.array until drained, np.ndarray after
    # Who each batch row of ``value`` belongs to (request ids); None = unowned.
    owners: Optional[list] = None
    layers: Optional[list[int]] = None

    def expand(self) -> list["Record"]:
        """Split a stacked record into one record per layer (host values)."""
        if self.layer != STACKED:
            return [self]
        return [Record(self.kind, layer, self.step, self.value[n], self.owners)
                for n, layer in enumerate(self.layers)]


class CaptureSession:
    """Owns installed hooks and the records they produce."""

    def __init__(self, config: CaptureConfig):
        self.config = config
        self.enabled = True
        self.step = 0
        self._pending: list[Record] = []
        self.records: list[Record] = []
        self._removers: list[Callable[[], None]] = []
        # Set by the adapter: residual -> logits, as the model's head reads it.
        self.readout: Optional[Callable[[mx.array], mx.array]] = None
        # Batch-row selection. None records every row; a list records only
        # those rows (an empty list records nothing). ``owners`` labels them.
        self.n_layers: Optional[int] = None  # set by the adapter
        self.rows: Optional[list[int]] = None
        self.owners: Optional[list] = None
        self._row_index: Optional[mx.array] = None
        self._identity = False   # rows are 0..n-1 in order: a batch of n rows is kept whole (select)
        # Hyper-connection models: streams [.., hc*D] -> residual [.., D].
        self.streams_to_resid: Optional[Callable[[mx.array], mx.array]] = None
        self.hc_count: int = 1
        # Per-step stash for batched reductions: kind -> {layer: (tensor, its extra or None)} (_defer).
        self._stash: dict[str, dict[int, tuple[mx.array, Optional[mx.array]]]] = {}
        # An adapter's state from one hook to the next within a forward (qwen4_exp: the last layer's streams; inkling2:
        # what its read shares need), handed to the ``on_flush`` functions and emptied with the stash, so nothing of a
        # forward outlives its step even when the forward stops part way.
        self.carry: dict = {}
        self._flushers: list[Callable[[dict], None]] = []
        # MoE routing rule (routing.py), set by the adapter; None = no router to read.
        self.routing = None
        self._router_tokens: Optional[mx.array] = None
        # Told when one of this session's hooks raises (the tracer sets it: TraceManager.capture_failed).
        self.on_error: Optional[Callable[[BaseException], None]] = None

    @property
    def active(self) -> bool:
        """Hooks check this first; False makes them return immediately."""
        return self.enabled and (self.rows is None or len(self.rows) > 0)

    def set_rows(self, rows: Optional[list[int]], owners: Optional[list] = None) -> None:
        self.rows, self.owners = rows, owners
        self._row_index = None if rows is None else mx.array(rows, dtype=mx.int32)
        self._identity = rows is not None and rows == list(range(len(rows)))

    def select(self, x: mx.array) -> mx.array:
        """Keep only the selected batch rows (axis 0) before any reduction."""
        if self._row_index is None or (self._identity and len(self.rows) == x.shape[0]):
            return x
        return mx.take(x, self._row_index, axis=0)

    # -- hook management -------------------------------------------------
    def hook(self, module, fn) -> None:
        self._removers.append(add_hook(module, fn, self._hook_failed))

    def on_close(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` when the session closes, with the hook removals (an adapter's other change put back)."""
        self._removers.append(fn)

    def _hook_failed(self, exc: BaseException) -> None:
        if self.on_error is not None:
            self.on_error(exc)

    def close(self) -> None:
        for remove in self._removers:
            remove()
        self._removers.clear()

    def wants_layer(self, i: int) -> bool:
        return self.config.layers is None or i in self.config.layers

    # -- recording -------------------------------------------------------
    def emit(self, kind: str, layer: int, value: mx.array) -> None:
        self._pending.append(Record(kind, layer, self.step, value))

    def emit_stacked(self, kind: str, layers: list[int], value: mx.array) -> None:
        self._pending.append(Record(kind, STACKED, self.step, value, layers=list(layers)))

    def _defer(self, kind: str, layer: int, x: mx.array, extra: Optional[mx.array] = None) -> bool:
        """Keep ``x``, with ``extra`` (what its reduction reads beside it, or None), for the step's end, where the
        layers' records of a kind are stacked and reduced together (``_flush``); False when ``x`` has too many tokens
        to stack: the caller reduces it now."""
        if x.shape[1] > BATCH_MAX_TOKENS:
            return False
        self._stash.setdefault(kind, {})[layer] = (x, extra)
        return True

    def record_residual(self, layer: int, h: mx.array, state: Optional[mx.array] = None) -> None:
        """h: [batch, tokens, hidden] residual as the next block would read it. ``state`` [batch, tokens, W]: what a
        J-lens W wide reads at this layer instead (Inkling 2's state at a visit; its width differs by layer)."""
        if not self._defer("resid", layer, h, state):
            self._resid_reductions([layer], h[None], states=[state])

    def record_steered(self, layer: int, x: mx.array) -> None:
        """x: [batch, tokens, W], what a steer adds to at this layer where that is not the residual recorded (Inkling
        2's running sum): its norm, ``steer_norm``, the dose unit (steer.norm_table)."""
        if self.config.residual_norm and not self._defer("steer", layer, x):
            self._steer_reductions([layer], x[None])

    def _steer_reductions(self, layers: list[int], X: mx.array) -> None:
        self.emit_stacked("steer_norm", layers, mx.linalg.norm(X.astype(mx.float32), axis=-1))

    def record_streams(self, layer: int, streams: mx.array, before: Optional[mx.array] = None) -> None:
        """Hyper-connection residual streams [batch, tokens, hc * hidden]; ``before``, the streams the layer read (the
        same shape and rows): what the layer wrote into each stream is recorded too (``resid_write_norm``)."""
        if not self._defer("streams", layer, streams, before):
            self._stream_reductions([layer], streams[None], None if before is None else before[None])

    def record_write(self, layer: int, out: mx.array, before: Optional[mx.array] = None) -> None:
        """A residual that is one stream [batch, tokens, hidden] (a conventional decoder's; Inkling 2's running sum):
        the layer's output less its input ``before`` (None: the layer starts the residual afresh, so its whole output
        is its write), ``resid_write_norm`` with one stream."""
        if not self._defer("wrote", layer, out, before):
            self._write_reductions([layer], out[None], None if before is None else before[None])

    def on_flush(self, fn: Callable[[dict], None]) -> None:
        """Run ``fn(carry)`` as each step's records are gathered, with what the adapter's hooks kept in ``carry`` that
        step: an adapter's own reductions over the whole step (Inkling 2's read shares)."""
        self._flushers.append(fn)

    def record_router(self, layer: int, logits: mx.array, tokens: Optional[mx.array] = None) -> None:
        """Router logits [batch, tokens, n_experts]; ``tokens`` [batch, tokens]
        the step's token ids, for rules that route some layers by token id."""
        if not self._defer("router", layer, logits):
            self._router_reductions([layer], logits[None], tokens)
        elif tokens is not None:
            self._router_tokens = tokens

    def _stream_reductions(self, layers: list[int], S: mx.array, before: Optional[mx.array] = None) -> None:
        L, B, T, width = S.shape
        per_stream = S.reshape(L, B, T, self.hc_count, -1).astype(mx.float32)
        self.emit_stacked("hc_stream_norm", layers, mx.linalg.norm(per_stream, axis=-1))
        if before is not None:
            self._write_reductions(layers, S, before)
        H = self.streams_to_resid(S.reshape(L * B, T, width))
        # A lens fitted on the streams (J: hidden x hc*hidden) reads S, not H.
        self._resid_reductions(layers, H.reshape(L, B, T, -1), S)

    def _write_reductions(self, layers: list[int], S: mx.array, before: Optional[mx.array]) -> None:
        """Each layer's output streams [L, B, T, hc * hidden] less its input's (None: none), a norm per stream."""
        L, B, T, _ = S.shape
        wrote = (S if before is None else S - before).reshape(L, B, T, self.hc_count, -1).astype(mx.float32)
        self.emit_stacked("resid_write_norm", layers, mx.linalg.norm(wrote, axis=-1))

    def _resid_reductions(self, layers: list[int], H: mx.array, S: Optional[mx.array] = None,
                          states: Optional[list] = None) -> None:
        """H: residual the head reads [L, B, T, D]; S: hyper-connection streams if any; ``states``: each layer's
        state [B, T, W] or None (``record_residual``). J-lens matrices apply to whichever matches their input width, a
        layer's state before the residual (an Inkling 2 visit whose state is one sum is as wide as its residual)."""
        cfg = self.config

        def source(n: int, width: int) -> Optional[mx.array]:
            if states is not None and states[n] is not None and states[n].shape[-1] == width:
                return states[n]
            if H.shape[-1] == width:
                return H[n]
            if S is not None and S.shape[-1] == width:
                return S[n]
            return None
        H32 = H.astype(mx.float32)
        if cfg.residual_norm:
            self.emit_stacked("resid_norm", layers, mx.linalg.norm(H32, axis=-1))
        reads = []   # the lens reads of every layer here, read out together (_lens_reads)
        T = H.shape[-2]
        tail = cfg.lens_tail if cfg.lens_tail and T > cfg.lens_tail else 0
        # Lens rules from this step's cfg, never self.config: /jlens use swaps the config whole between forwards
        # (ModelTracer.refresh_lens), and a step under way finishes on the lens it started with.
        jlens = cfg.jlens if cfg.jlens and cfg.jlens_topk and self.readout is not None else None
        lens = cfg.logit_lens_topk and self.readout is not None
        for n, layer in enumerate(layers):
            J = jlens.get(layer) if jlens and (cfg.jlens_layers is None or layer in cfg.jlens_layers) else None
            if J is not None:
                X = source(n, J.shape[-1])
                if X is not None:
                    X = X[..., T - tail:, :] if tail else X
                    reads.append(("jlens", layer, cfg.jlens_topk, (X.astype(J.dtype) @ J.T).astype(H.dtype), T - tail if tail else 0))
            if lens and cfg.reads_lens(layer, self.n_layers):
                reads.append(("lens", layer, cfg.logit_lens_topk, H[n][..., T - tail:, :] if tail else H[n], T - tail if tail else 0))
        self._lens_reads(reads)

    def _lens_reads(self, reads: list) -> None:
        """Each (prefix, layer, k, x [B, T, D], lead) through the readout and its top k emitted (no entropy), as
        ``_emit_topk`` does; up to ``LENS_BATCH_ROWS`` rows share one readout call (the LM head's weights read once for
        them all). ``lead``: x is only the step's last rows, and that many rows before them are recorded empty
        (``_pad_lead``). The reads of one call share x's shape and lead (``_resid_reductions``)."""
        if not reads:
            return
        # Keep the readout's own LM-head call out of the "logits" records.
        self.enabled = False
        try:
            B, T = reads[0][3].shape[0], reads[0][3].shape[-2]
            per = max(1, LENS_BATCH_ROWS // max(1, B * T))
            for i in range(0, len(reads), per):
                chunk = reads[i:i + per]
                if len(chunk) == 1 and not chunk[0][4] and T > LOGIT_CHUNK_TOKENS:
                    prefix, layer, k, x, _ = chunk[0]
                    self._emit_topk(prefix, layer, self.readout(x), k, entropy=False)
                    continue
                x = chunk[0][3] if len(chunk) == 1 else mx.concatenate([r[3] for r in chunk], axis=0)
                ids, top, _ = self._topk_stats(self.readout(x), max(r[2] for r in chunk), entropy=False)
                for j, (prefix, layer, k, _, lead) in enumerate(chunk):
                    rows = slice(j * B, (j + 1) * B)
                    for kind, v in self._pad_lead(ids[rows, ..., :k], top[rows, ..., :k], lead).items():
                        self.emit(f"{prefix}_{kind}", layer, v)
        finally:
            self.enabled = True

    @staticmethod
    def _pad_lead(ids: mx.array, top: mx.array, lead: int) -> dict:
        """A read of a step's last rows, with the ``lead`` rows before them recorded empty (ids -1, log-p -inf, as a
        position with no record), so the record keeps one row per position."""
        if not lead:
            return dict(ids=ids, logp=top)
        lead_shape, k = ids.shape[:-2], ids.shape[-1]
        return dict(ids=mx.concatenate([mx.full((*lead_shape, lead, k), -1, ids.dtype), ids], axis=-2),
                    logp=mx.concatenate([mx.full((*lead_shape, lead, k), -mx.inf, top.dtype), top], axis=-2))

    def _router_reductions(self, layers: list[int], G: mx.array, tokens: Optional[mx.array] = None) -> None:
        """G: router logits [L, B, T, E] -> the experts the model picks and the
        weights it mixes them with (``router_probs``), by its own rule."""
        if tokens is None:
            tokens, self._router_tokens = self._router_tokens, None
        ids, w = self.routing.select(G, layers, tokens)
        self.emit_stacked("router_ids", layers, ids)
        self.emit_stacked("router_probs", layers, w)

    def _flush(self) -> None:
        stash, self._stash = self._stash, {}
        carry, self.carry = self.carry, {}
        for fn in self._flushers:
            fn(carry)
        # Each kind's records stacked by layer, with each layer's extra (_defer): a state is passed as a list (widths
        # differ by layer); the streams a layer read go only when every layer has them (record_streams: no write
        # without); what a layer wrote from is zeros where it started afresh (record_write: its whole output).
        handlers = {"resid": lambda ls, H, states: self._resid_reductions(ls, H, states=states),
                    "streams": lambda ls, S, befores: self._stream_reductions(ls, S, _stacked(befores)),
                    "wrote": lambda ls, S, befores: self._write_reductions(ls, S, _stacked(befores, S[0])),
                    "router": lambda ls, G, _: self._router_reductions(ls, G),
                    "steer": lambda ls, X, _: self._steer_reductions(ls, X)}
        for kind, by_layer in stash.items():
            layers = sorted(by_layer)
            handlers[kind](layers, mx.stack([by_layer[l][0] for l in layers]), [by_layer[l][1] for l in layers])

    def record_logits(self, logits: mx.array) -> None:
        self._emit_topk("logit", -1, logits, self.config.logits_topk)

    def _emit_topk(self, prefix: str, layer: int, logits: mx.array, k: int, entropy: bool = True) -> None:
        """The top k (``prefix``_ids, _logp) and, with ``entropy``, the entropy (the output's; a lens read has none)."""
        T = logits.shape[-2] if logits.ndim >= 2 else 1
        if T <= LOGIT_CHUNK_TOKENS:
            ids, top, ent = self._topk_stats(logits, k, entropy)
        else:
            parts = [self._topk_stats(logits[..., i:i + LOGIT_CHUNK_TOKENS, :], k, entropy)
                     for i in range(0, T, LOGIT_CHUNK_TOKENS)]
            ids, top = (mx.concatenate([p[n] for p in parts], axis=-2) for n in range(2))
            ent = mx.concatenate([p[2] for p in parts], axis=-1) if entropy else None
        self.emit(f"{prefix}_ids", layer, ids)
        self.emit(f"{prefix}_logp", layer, top)
        if entropy:
            self.emit(f"{prefix}_entropy", layer, ent)

    @staticmethod
    def _topk_stats(logits: mx.array, k: int, entropy: bool = True):
        """(top-k ids sorted by prob, their log-probs, entropy or None) over the last axis. A choice masked out with
        -inf (Inkling 2's padded vocabulary) adds nothing to the entropy (0 x -inf would make it NaN, which the dock's
        JSON parser refuses)."""
        z = logits.astype(mx.float32)
        lse = mx.logsumexp(z, axis=-1, keepdims=True)
        ids = _topk_ids(z, k)
        top = mx.take_along_axis(z, ids, axis=-1) - lse
        # Sort the k entries so index 0 is the argmax.
        order = mx.argsort(-top, axis=-1)
        return (mx.take_along_axis(ids, order, axis=-1).astype(mx.int32),
                mx.take_along_axis(top, order, axis=-1),
                lse[..., 0] - (mx.exp(z - lse) * mx.where(z > -mx.inf, z, 0)).sum(axis=-1) if entropy else None)

    # -- step boundary ---------------------------------------------------
    def pending_arrays(self) -> list[mx.array]:
        """Lazy arrays to fold into the caller's async_eval for this step."""
        self._flush()
        return [r.value for r in self._pending]

    def end_step(self) -> None:
        self._flush()
        if self.owners is not None:
            for r in self._pending:
                r.owners = self.owners
        self.records.extend(self._pending)
        self._pending = []
        self.step += 1

    def drain(self) -> list[Record]:
        """Move finished records to host. Blocks only on already-scheduled work."""
        out = []
        for r in self.records:
            if isinstance(r.value, mx.array):
                r.value = np.array(r.value)
            out.extend(r.expand())
        self.records = []
        return out


# ---------------------------------------------------------------- what a traced request records

LENS_PROMPT_TAIL = 64


def capture_config() -> CaptureConfig:
    """What a traced request records."""
    # router: each MoE layer's experts by the model's own rule and k (routing.py).
    # The logit lens's top 10, beside the J-lens when the model has one (live.py).
    # lens_tail: a prompt chunk reads both lenses on its last 64 rows only; every prompt row cost ~2.7 ms per lens
    # per token (Flash Next, LM head alone).
    # The layers the lenses read: the lens step's (tracer.lens_steps), set by the tracer.
    return CaptureConfig(router=True, logits_topk=10, logit_lens_topk=10, lens_tail=LENS_PROMPT_TAIL)

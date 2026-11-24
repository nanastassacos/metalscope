"""Adapter for Inkling 2 (``inkling2``, looped ``inkling2_loop``) as oMLX >= 0.8.3 vendors it (omlx/patches/inkling2;
an older oMLX's is refused: ``matches``). docs/measurements.md has what was measured with it.

Three things differ from a conventional decoder:

* Layer visits: a looped checkpoint runs its core layers several times a token (the text model's ``_schedule``; 48
  visits of 24 layers on Inkling 2 Mini), each later pass restarting from the prelude's output. The layer axis a
  trace records is the visit (``session.n_layers`` = visits), and ``_Visits`` maps each layer call to its visit.
* No single residual stream: a layer returns its attention-residual block's running sum ``h`` (a block's first layer
  appends the unit-RMS sum it was handed to the list of finished ``blocks``), and the head reads a learned mix of the
  finished blocks and ``h`` (``final_input``). The residual recorded at a visit is that mix: what the final norm and
  head would read had the model stopped there (as DeepSeek V4's ``hc_head``). The J-lens reads the state after the
  visit (``state``): the finished blocks but the first, ``h``, and inside a later loop pass the previous pass's
  message bank (its blocks and its sum); ``lens_sources`` counts them. The first block is left out: it is the token's
  own embedding, the same at every visit, and through the head it reads as the token just read. A steer writes the
  layer's output ``h`` (the running sum), not the mix: the edit below.
* Decode and verify steps (up to 16 tokens) run a layer's second half, the MoE included, as one ``mx.compile``d call
  (row-exact MTP verify rows: a compiled call of their own, ``_post_exact``), inside which a hook runs only while the
  call is traced. While a session records, that half runs uncompiled (``_eager_while_recording``), so the router hook
  sees each call.

A J-lens fit runs the model's loop itself (``differentiable``): a visit's cache is ``visit_caches``'s (a sliding visit
one entry, a full-attention visit two), and the full-attention visits share one ``tau``.

A concept steers whole replies (thinking included) through the running sum ``h``: its direction at a visit is the
diff of means of each text's mean sum there, and a push at a visit covers that visit and the rest of its block
(``steer_block``: the vector's meta ``block``, steer.push_layers), each visit with its own direction and dose x its own
typical sum norm (``steer_norm``, recorded by the hook: the sum's norm restarts at each block, about 47 to 205 over
visits 30-35 on the Mini). Its steer layer is the selection sweep's pick of two (``steer_layers``): the prelude's
first visit (0), whose sum every pass restarts from, or the first visit of the last pass (30 on the Mini). A forced
answer (engines/mlx.py) opens the reply's answer channel after the prompt (``ANSWER_MARKS``): the template has no
switch for thinking.

A word (its lens direction) changes one token only, the dock's edit, as on every model (steer.WORD_EDIT_ONLY). The
edit acts on the running sum ``h`` alone, at the visit (``visit_counter``: the steer session's own ``_Visits``). A
word's vector at a visit is its lens direction cut to the sum's columns of the state (``lens_columns``), made unit; a
removal projects it out over the last block's fitted visits (``edit_band``: 42-46 on the Mini), a swap pushes the new
word at the band's middle.

Measured on Inkling 2 Mini (2026-11-22) and not in docs/measurements.md:
* What the lenses read (6 fit prompts, 135 held-out positions, the model's own next token in the lens's top 10 / top
  1, mean over the 48 visits, the first block still read): the logit lens on ``h`` alone 0.314 / 0.140, on the head's
  mix 0.363 / 0.190; a J-lens on ``h`` alone 0.141 / 0.081, on the blocks and ``h`` 0.293 / 0.146, on the whole state
  0.379 / 0.182.
* The first block in the state (a lens fitted on 8 windows): the token just read was the lens's top word at 61% of 992
  held-out positions (the model writes it next at 0.1%), 8% without those columns; the model's next token in the top
  10 at 0.25 (top 1 0.08) against 0.31 (0.14) without them.
* The sweep's forced-choice items cannot tell a middle block from the two steer layers (a thought that never ends is
  not seen: over all eight blocks it picked 18 for shakespearean and 6 for fluffy); between the two it picked 0 for
  shakespearean (+2.7 log-odds over a random direction against +1.2) and 30 for fluffy (+1.4 against +0.8).
* Concept pushes: over the prelude a voice breaks replies from 0.4; decoded greedily (pirate) the prelude and pass 1 did
  nothing and pass 2 was weak; one visit alone in 30-35 takes 1.3-2x its norm.
* The word edit: a removal over 30-41 changed the top word in 24%, at 46 alone 74%, over 0-46 78% (3.7 of the next 5
  kept); a swap's dose unit is the sum's norm at 44, about 64 (with the head mix's, about 38: 96% and 3.1 kept at 0.7);
  a word pushed over block 30-35 loops or does nothing at every dose tried; a removal held on every token of a reply
  left the word in (milk 1 to 1, cat 5 to 5, blue 10 to 9).
"""

from __future__ import annotations

import sys
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .. import routing
from ..capture import BATCH_MAX_TOKENS, CaptureSession
from . import generic

NAME = "inkling2"
MODEL_TYPES = ("inkling2", "inkling2_loop")
# After the template's prompt (it has no generation prompt and no thinking switch: the model opens its own message):
# what opens the reply's answer, so the next token is the answer's first, and the token that ends the answer.
ANSWER_MARKS = ("<|message_model|><|content_text|>", "<|end_message|>")


def matches(model: nn.Module) -> bool:
    """Inkling 2's text model. ValueError on an older oMLX's (no ``visit_caches``: a cache entry a visit), which every
    adapter call would read wrong, generic's too."""
    try:
        inner = generic.text_model(model)
    except ValueError:
        return False
    if routing._cls(inner).__name__ != "Inkling2TextModel":
        return False
    if not hasattr(inner, "visit_caches"):
        raise ValueError("Inkling 2 needs oMLX 0.8.3 or newer")
    return True


def layer_count(model: nn.Module) -> int:
    return len(generic.text_model(model)._schedule)


def _top(model: nn.Module):
    """The Inkling 2 model (it owns ``logits_from_hidden``): the one served, or the one oMLX's VLM adapter wraps."""
    for m in (model, getattr(model, "_vlm_model", None)):
        if callable(getattr(m, "logits_from_hidden", None)):
            return m
    raise ValueError("no Inkling 2 output head (logits_from_hidden)")


def readout(model: nn.Module):
    """The head's input (a visit's mix, ``state``) -> logits, as the head reads it: the final norm in float32, then the
    LM head, softcap and vocabulary mask (``logits_from_hidden``)."""
    top, norm = _top(model), generic.text_model(model)["norm"]
    return lambda h: top.logits_from_hidden(norm(h.astype(mx.float32)).astype(h.dtype))


def unembed(model: nn.Module, ids: list[int]) -> tuple[mx.array, mx.array]:
    """The tokens' output-head rows [n, H] float32 (a quantised head's rows alone dequantised: jlens.head_rows) and the
    final norm's gain [H], for a word's lens direction (ModelTracer.unembed)."""
    from ..jlens import head_rows
    inner = generic.text_model(model)
    H = int(inner.args.hidden_size)
    return head_rows(_top(model).lm_head, ids), inner["norm"](mx.ones((1, H), dtype=mx.float32))[0]


def lens_columns(model: nn.Module) -> Callable[[int, int], slice]:
    """(visit, its lens Jacobian's width) -> the running sum's columns in that visit's state (``state``): source
    ``layer // k`` (the blocks past the first before it), H wide. Inkling 2 Mini: source 0 at visits 0-5, 1 at 6-11, 2
    at 12-17, then 1, 2 each later pass's halves, 3 at 42-47. ValueError when the width is not the state's
    (lens_sources x H): a lens of another layout (the token's embedding block in) would cut the wrong columns."""
    inner = generic.text_model(model)
    H, k = int(inner.args.hidden_size), int(inner.args.n_layers_per_attn_res_block)
    src, n = [int(i) // k for i, _ in inner._schedule], lens_sources(vars(inner.args))

    def cut(v: int, width: int) -> slice:
        if int(width) != n[v] * H:
            raise ValueError(f"this lens is not as wide as the model's state at visit {v}: fit it again (/jlens learn)")
        return slice(src[v] * H, (src[v] + 1) * H)
    return cut


def edit_band(model: nn.Module) -> tuple[int, int]:
    """The visits the word edit removes over: the last block's fitted ones (the last visit has no lens), 42-46 of 48
    on the Mini, where a removal from the sum changed the top word in 74% and kept 4.0 of the next 5 candidates (the
    workspace band 18-44: 62%; docs/measurements.md)."""
    inner = generic.text_model(model)
    n, k = len(inner._schedule), int(inner.args.n_layers_per_attn_res_block)
    return n - k, n - 2


def steer_block(model: nn.Module) -> int:
    """The visits a concept's push spans: its attention-residual block (``n_layers_per_attn_res_block``, 6 on the
    Mini; blocks are aligned with the visits, since the loop's core starts and ends on block boundaries). A push at
    a visit covers it and the rest of its block (steer.push_layers)."""
    return int(generic.text_model(model).args.n_layers_per_attn_res_block)


def steer_layers(model: nn.Module) -> list[int]:
    """The selection sweep's candidate steer layers, the two places where a concept steered whole replies (module
    docstring): the prelude's first visit (0, with a prelude) and the first visit of the last loop pass (30 of 48 on
    the Mini, core_start + (num_passes - 1) x the core's length); the every-4th default (learn/gpu.py) without a
    loop."""
    inner = generic.text_model(model)
    schedule = inner._schedule
    last = max(p for _, p in schedule)
    if not last:
        return []
    last_pass = next(v for v, (_, p) in enumerate(schedule) if p == last)
    return ([0] if inner._core_start else []) + [last_pass]


def visit_counter(model: nn.Module) -> Callable[[int], int]:
    """A steer session's layer hooks: a layer call -> its visit. A ``_Visits`` of its own: it sees every layer call
    as the capture's does, so the two stay in step."""
    return _Visits(generic.text_model(model)._schedule).done


def _module(inner: nn.Module):
    """The module the text model's class is from (oMLX's model.py): its ``_unit_rms``, ``kernels`` and
    ``_COMPILED_MAX_TOKENS``, read when used (a fit swaps them)."""
    return sys.modules[routing._cls(inner).__module__]


def lens_sources(cfg: dict) -> list[int]:
    """How many hidden-wide sources the state after each layer visit holds (``state``), from config.json (its
    ``text_config``): the finished blocks but the first and the running sum, and in a later loop pass the previous
    pass's message bank, its blocks past the prelude's and its sum. A ``loop_num_passes`` beside the loop_config sets
    the passes, as oMLX's ``TextArgs.from_dict`` reads it.
    Inkling 2 Mini: [1]*6 + [2]*6 + [3]*6 + ([4]*6 + [5]*6)*2 + [4]*6."""
    t = cfg.get("text_config") or cfg
    n, k = int(t.get("num_hidden_layers") or 0), int(t.get("n_layers_per_attn_res_block") or 6)
    loop = t.get("loop_config")
    if not loop:
        return [i // k + 1 for i in range(n)]
    start, end = int(loop["core_start"]), int(loop["core_end"])
    passes = int(t["loop_num_passes"] if "loop_num_passes" in t else loop["num_passes"])
    bank = (end - 1) // k - start // k + 1
    schedule = ([(i, 0) for i in range(start)] + [(i, p) for p in range(passes) for i in range(start, end)]
                + [(i, 0) for i in range(end, n)])
    return [i // k + 1 + (bank if p else 0) for i, p in schedule]


def state(inner: nn.Module, blocks: list, h: mx.array, bank=None, select=lambda x: x) -> tuple[mx.array, mx.array]:
    """After a layer visit, with ``blocks`` as the layer left them, ``h`` its output and ``bank`` the messages it was
    handed (None outside a later loop pass), rows by ``select``: (the head's own mix of the blocks and ``h``, in
    ``h``'s dtype; the state ``[*blocks past the first, h, *bank blocks, bank sum]`` float32, side by side on the last
    axis: the first block is the token's embedding, see the module docstring)."""
    blocks, h = [select(b) for b in blocks], select(h)
    mix = inner.final_input(blocks, _module(inner)._unit_rms(h.astype(mx.float32)))[:, :, 0].astype(h.dtype)
    parts = [*blocks[1:], h] + ([*(select(b) for b in bank[0]), select(bank[1])] if bank is not None else [])
    return mix, mx.concatenate([p.astype(mx.float32) for p in parts], axis=-1)


class _Visits:
    """Which visit of ``schedule`` (``(layer, pass)`` in order) a layer call is: the next visit when that is this
    layer's, else this layer's first. A forward begins at layer 0, whose only visit is the first (or, on a model
    with no loop, at a layer whose only visit is its own), so a count left part-way by a forward that raised is put
    right at the next forward's first call."""

    def __init__(self, schedule):
        self.layers = [int(i) for i, _ in schedule]
        self.first: dict[int, int] = {}
        for v, i in enumerate(self.layers):
            self.first.setdefault(i, v)
        self.next = 0

    def current(self, layer: int) -> int:
        """The visit of the call of ``layer`` under way (hooks inside the layer ask this)."""
        n = self.next
        return n if self.layers[n] == layer else self.first[layer]

    def done(self, layer: int) -> int:
        """A call of ``layer`` returned: its visit. The next call is expected at the visit after it."""
        v = self.current(layer)
        self.next = (v + 1) % len(self.layers)
        return v


def _eager_while_recording(session: CaptureSession, layer: nn.Module) -> None:
    """``layer``'s compiled second halves (``_post_compiled``, and ``_post_exact`` on row-exact verify rows) run as plain
    ``_post`` while ``session`` records, so the MoE hook inside runs on every call (``_post`` reads the row-exact flag
    as it runs: the same arithmetic); untraced forwards keep the compiled ones. Put back when the session closes."""
    plain = weakref.proxy(layer)                                               # a proxy: the layer does not hold itself
    for name in ("_post_compiled", "_post_exact"):
        compiled = layer.__dict__[name]
        setattr(layer, name, lambda *a, c=compiled: plain._post(*a) if session.active else c(*a))
        session.on_close(lambda name=name, c=compiled: setattr(layer, name, c))


class _Reads:
    """Where each visit's inputs read from, ``resid_read_share`` [visits, B, T, K]: each source's share in the softmax
    mixes (``AttnResQuery``) that feed the visit's attention and its MLP, the mean over those of its queries that read
    the visit's sources (at a block's first layer the attention's alone: its MLP reads one source more, the block it
    just finished; at layer 0, which has no attention mix, the MLP's), slot j the j-th finished block (0 the
    embedding's; 0 where not finished yet) and slot K - 1 the running sum (K: 5 on the Mini). As the dock's design was
    measured offline.

    The attention's mix runs inside the layer's compiled first half, where no hook runs: its shares are worked out here
    from what the layer hook has (the blocks, the incoming sum) and the layer's queries. The MLP's reads the sum
    halfway through the layer, which only its own call sees: a hook on ``mlp_input``, called in the second half, which
    runs uncompiled while the session records (``_eager_while_recording``, so only with the router recorded, as served).
    A step's visits are reduced together as its records are gathered (``on_flush``): about fifteen ops a step, none a
    visit; a prompt chunk longer than BATCH_MAX_TOKENS each visit on its own."""

    def __init__(self, session: CaptureSession, inner: nn.Module):
        self.session, self.inner = session, inner
        self.layers = inner.layers
        self.K = -(-len(self.layers) // int(inner.args.n_layers_per_attn_res_block)) + 1
        self.qa = [int(layer.attn_input.query.shape[0]) if i else 0 for i, layer in enumerate(self.layers)]
        self.C = max(self.qa) + 1           # the attention's queries, then the MLP's in the last column
        self.tables: dict = {}

    def mlp_hook(self, visit: Callable[[], int]):
        def on_mlp_input(_m, args, _k, _out):
            if self.session.active and self.session.wants_layer(v := visit()):
                self.session.carry.setdefault("mlp_read", {})[v] = args[1]
        return on_mlp_input

    def keep(self, v: int, i: int, blocks: list, h: mx.array) -> None:
        """Visit ``v`` (layer ``i``) returned with ``blocks`` as it left them, having read ``h``."""
        r = self.session.carry.get("mlp_read", {}).pop(v, None)      # the sum its MLP read
        if r is None:
            return
        entry = (v, i, list(blocks), h, r)
        if h.shape[1] > BATCH_MAX_TOKENS:
            self._emit([entry])
        else:
            self.session.carry.setdefault("reads", []).append(entry)

    def flush(self, carry: dict) -> None:
        if carry.get("reads"):
            self._emit(sorted(carry["reads"], key=lambda e: e[0]))

    def _table(self, sig: tuple):
        """For visits (layer, finished blocks after it): their queries side by side, scaled [n, H, C], which sources
        each column reads (0, else -inf: [n, K + 1, 1, C]) and each column's weight in the mean [n, 1, 1, C]. Sources:
        the blocks in slots 0 .. K - 2, the incoming sum (unit-RMS) at K - 1, the sum the MLP read at K."""
        K, C, H = self.K, self.C, int(self.inner.args.hidden_size)
        W, mask, cw = [], np.zeros((len(sig), K + 1, 1, C), np.float32), np.zeros((len(sig), 1, 1, C), np.float32)
        for r, (i, n_after) in enumerate(sig):
            layer, qa = self.layers[i], self.qa[i]
            q = mx.zeros((C, H), dtype=mx.float32)
            if qa:
                q[:qa] = layer.attn_input.query.astype(mx.float32) * layer.attn_input.scale
            q[C - 1:] = layer.mlp_input.query.astype(mx.float32) * layer.mlp_input.scale
            W.append(q.T)
            n_before = n_after - bool(layer.is_boundary)
            attn = np.full(K + 1, -np.inf, np.float32)
            attn[:n_before], attn[K - 1] = 0, 0
            mlp = np.full(K + 1, -np.inf, np.float32)
            mlp[:min(n_after, K - 1)], mlp[K] = 0, 0
            mask[r, :, 0, :qa], mask[r, :, 0, C - 1] = attn[:, None], mlp
            counted = list(range(qa)) + ([] if qa and layer.is_boundary else [C - 1])
            cw[r, 0, 0, counted] = 1 / len(counted)
        return mx.stack(W), mx.array(mask), mx.array(cw)

    def _emit(self, entries: list) -> None:
        K = self.K
        sig = tuple((i, len(blocks)) for _, i, blocks, _, _ in entries)
        if sig not in self.tables:
            self.tables[sig] = self._table(sig)
        W, mask, cw = self.tables[sig]
        parts = []
        for _, _, blocks, _, r in entries:          # missing blocks: any source, masked out
            parts += blocks[:K - 1] + [r] * (K - len(blocks[:K - 1]))
        select = self.session.select
        X = select(mx.stack(parts, axis=1))                                          # [B, n * K, T, H]
        B, _, T, H = X.shape
        X = X.reshape(B, len(entries), K, T, H)
        h_n = _module(self.inner)._unit_rms(select(mx.stack([e[3] for e in entries], axis=1)).astype(mx.float32))
        src = mx.concatenate([X[:, :, :K - 1], h_n[:, :, None], X[:, :, K - 1:]], axis=2)   # [B, n, K + 1, T, H]
        w = mx.softmax(src @ W[None, :, None] + mask, axis=2, precise=True)           # [B, n, K + 1, T, C]
        w = (w * cw).sum(-1)
        w = mx.concatenate([w[:, :, :K - 1], w[:, :, K - 1:K] + w[:, :, K:]], axis=2)  # both sums in the sum's slot
        self.session.emit_stacked("resid_read_share", [e[0] for e in entries], w.transpose(1, 0, 3, 2))


def install(session: CaptureSession, model: nn.Module) -> None:
    cfg = session.config
    inner = generic.text_model(model)
    visits = _Visits(inner._schedule)
    session.n_layers = len(visits.layers)
    session.readout = readout(model)
    reads = _Reads(session, inner) if cfg.router else None      # needs the second half uncompiled (_Reads)

    # One hook a layer, counting its visits (even untraced, so the count stays in step); it records the visits asked:
    # the head's mix, the state for a J-lens, the sum's norm (a steer's dose unit), what the visit wrote into the sum (a
    # block's first layer starts the sum afresh) and where it read from.
    for i, layer in enumerate(inner.layers):
        def on_layer(_m, args, kwargs, out, i=i, boundary=bool(layer.is_boundary)):
            v = visits.done(i)
            if session.active and session.wants_layer(v):
                session.record_residual(v, *state(inner, args[1], out, kwargs.get("bank"), session.select))
                session.record_steered(v, session.select(out))
                session.record_write(v, session.select(out), None if boundary else session.select(args[0]))
                if reads is not None:
                    reads.keep(v, i, args[1], args[0])
        session.hook(layer, on_layer)

    if reads is not None:
        for i, layer in enumerate(inner.layers):
            session.hook(layer["mlp_input"], reads.mlp_hook(lambda i=i: visits.current(i)))
        session.on_flush(reads.flush)

    if cfg.router:
        # The model's rule (routing.py), its blocks by visit; each MoE call recorded at its layer's visit under way.
        blocks = [layer["mlp"] for layer in inner.layers]
        rule = session.routing = routing.rule_for(blocks[0])
        rule.blocks = {v: blocks[i] for v, i in enumerate(visits.layers)}
        for i, block in enumerate(blocks):
            session.hook(block, generic.router_hook(session, lambda i=i: visits.current(i)))
        for layer in inner.layers:
            _eager_while_recording(session, layer)

    if cfg.logits_topk:
        session.hook(model, generic.output_hook(session, inner["embed"].weight.shape[0]))


class _PlainMx:
    """``mlx.core`` as oMLX's kernels.py sees it during a fit: a default device that is not the GPU, so each of its
    Metal kernels (no gradients) takes its own plain-MLX path. Everything else is mlx.core's."""

    def __getattr__(self, name):
        return getattr(mx, name)

    def default_device(self):
        return mx.cpu


@contextmanager
def differentiable(model: nn.Module):
    """The model set up for a J-lens fit (jlens/fit.py): yields ``probes`` (``eps``: visit -> a probe [B, T, n * H]
    added to the state after that visit, its n sources in ``state``'s order; ``shapes``: visit -> (n * H, dtype), set by
    each forward) and ``forward(ids, cache=None)``, the model's own loop that returns what the final norm reads.

    A probe stays in the state for everything that reads it later: blocks are carried across the loop's restarts, the
    bank through its pass. Gradients need the kernels' plain paths (``_PlainMx``), the layers' uncompiled halves, the
    experts looked up by MLX's own ``gather_qmm`` (a copy oMLX's engine loaded has it rerouted to a Metal kernel:
    "[Primitive::vjp] Not implemented for CustomKernel"), and the sums' unit-RMS with its scale held fixed (else
    d unit_rms(h)/dh applied to h is zero and the lens loses each sum's own direction). ``forward`` switches these
    while it builds its graph and puts them back: the model's own calls between passes (a window's prefix, the
    held-out scoring) run as served."""
    inner = generic.text_model(model)
    mod = _module(inner)
    H = int(inner.args.hidden_size)
    real = mod._unit_rms
    lookup = sys.modules.get("omlx.patches.m5_gather_qmm")       # its reroute asks _nax_host (None: not asked yet)

    def frozen(x, eps=1e-6):
        # The model's own value (``real``, to the bit: the forward computes what the model does on the same paths)
        # with the gradient of x times its scale held fixed (``y - stop_gradient(y)`` is zero-valued).
        if x.shape[-1] != H:
            return real(x, eps)
        y = x * mx.stop_gradient(mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps))
        return mx.stop_gradient(real(x, eps)) + (y - mx.stop_gradient(y))

    @contextmanager
    def plain():
        was = mod.kernels.mx, mod._COMPILED_MAX_TOKENS, mod._unit_rms, getattr(lookup, "_nax_host", None)
        mod.kernels.mx, mod._COMPILED_MAX_TOKENS, mod._unit_rms = _PlainMx(), 0, frozen
        if lookup is not None:
            lookup._nax_host = False
        try:
            yield
        finally:
            mod.kernels.mx, mod._COMPILED_MAX_TOKENS, mod._unit_rms = was[:3]
            if lookup is not None:
                lookup._nax_host = was[3]

    probes = SimpleNamespace(eps={}, shapes={})

    def loop(ids: mx.array, cache) -> mx.array:
        h = inner.embed_norm(inner.embed(ids))
        visits = [None] * len(inner._schedule) if cache is None else inner.visit_caches(cache)
        blocks, prelude, n_pre, bank, tau = [], None, 0, None, None
        for visit, ((idx, loop_pass), c) in enumerate(zip(inner._schedule, visits)):
            layer = inner.layers[idx]
            if idx == inner._core_start and prelude is None:
                prelude, n_pre = h, len(blocks)
            elif idx == inner._core_start:
                bank = (blocks[n_pre + 1:], frozen(h.astype(mx.float32)))
                # The restart keeps the probes: the prelude's blocks as they are now. blocks[n_pre] is unit_rms(prelude
                # sum), which the boundary layer reads only through unit_rms of its incoming sum: hand that sum
                # blocks[n_pre]'s gradient at the sum's scale (``carried`` is zero-valued).
                p32 = prelude.astype(mx.float32)
                scale = mx.stop_gradient(mx.sqrt(mx.mean(mx.square(p32), axis=-1, keepdims=True) + 1e-6))
                carried = (blocks[n_pre] - mx.stop_gradient(blocks[n_pre])) * scale
                h = (mx.stop_gradient(prelude).astype(mx.float32) + carried).astype(prelude.dtype)
                blocks = blocks[:n_pre]
            if not layer.sliding and tau is None:        # the full-attention visits share one log scaling, as served
                tau = layer.attn.log_tau(None if c is None else c[0], h.shape[1])
            live = bank if loop_pass else None
            h = layer(h, blocks, cache=c, bank=live, tau=None if layer.sliding else tau)
            n = len(blocks) + (len(live[0]) + 1 if live is not None else 0)
            probes.shapes[visit] = (n * H, mx.float32)
            e = probes.eps.get(visit)
            if e is not None:
                e, k = mx.split(e, n, axis=-1), len(blocks) - 1
                blocks = blocks[:1] + [b + x for b, x in zip(blocks[1:], e[:k])]
                h = h + e[k].astype(h.dtype)
                if live is not None:
                    bank = ([b + x for b, x in zip(bank[0], e[k + 1:-1])], bank[1] + e[-1])
        return inner.final_input(blocks, frozen(h.astype(mx.float32)))[:, :, 0]

    def forward(ids: mx.array, cache=None) -> mx.array:
        with plain():
            return loop(ids, cache)

    yield probes, forward

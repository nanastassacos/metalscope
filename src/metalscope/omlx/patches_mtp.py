"""oMLX patches for MTP (multi-token prediction) decode.

With MTP on, oMLX runs its verify forwards outside the scheduler's normal decode
step and keeps primed prompt snapshots of its own. These patches keep those
forwards traced and steered, and keep steered KV out of shared caches.
patches._patch_engine applies them when an MTP-decoding model attaches
(mtp.mtp_decode), after oMLX's own MTP patches are in place.
patch_mtp_verify is for the Qwen3.5 family only; the rest serve every MTP
model (Qwen3.5 family, Flash Next).
"""

from __future__ import annotations

import functools
import logging

from ..hooks import run_hooks
from ..live import MANAGER
from .patches import _apply, _batch_rows, _current, _patched, _safe

log = logging.getLogger("metalscope")


def patch_mtp_next() -> None:
    """GenerationBatch.next: select the traced rows once for the whole call.

    oMLX's Lightning MTP replaces next() and runs its verify forwards itself,
    never through _step, so the _step wrapper alone would record nothing.
    Applied after oMLX's MTP patch has wrapped next(), so this wraps the MTP
    version. Models without MTP never get it.
    """
    from mlx_lm.generate import GenerationBatch as batch_cls
    if _patched(batch_cls, "patch_mtp_next"):
        return
    orig_next = batch_cls.next

    @functools.wraps(orig_next)
    def next(self, *a, **k):
        found = _batch_rows(self)
        if found is None or not found[0].mtp:
            return orig_next(self, *a, **k)
        tracer, rids, sched = found

        def prompts():      # the prompt tail can be prefilled inside next(), with logits
            for rid in rids:
                req = sched.requests.get(rid) if MANAGER.is_traced(rid) else None
                if req is not None:
                    MANAGER.note_prompt(rid, len(req.prompt_token_ids or ()))
        _safe(prompts)
        # Verify windows count as decode for steer.py's positions. A window can
        # carry a prefilled prompt tail, which is then steered as decode too.
        _safe(MANAGER.rows_for, tracer, rids, "decode")
        _current.outer_rows = True
        try:
            return orig_next(self, *a, **k)
        finally:
            _current.outer_rows = False
            _safe(MANAGER.clear_rows, tracer)

    _apply("patch_mtp_next", (batch_cls, dict(next=next)))
    log.info("metalscope: MTP decode tracing enabled")


def patch_mtp_verify(cls=None) -> None:
    """Qwen3_5BatchInvariantForward._layer (mlx-vlm), for the Qwen3.5 family (``qwen3_5``, ``qwen3_5_moe``; e.g.
    Qwen3.8 27B, dense). Its MTP verify windows compute each decoder layer's output in ``_layer`` without calling
    the layer, so the layer's hooks (capture, steering) never run. This runs them on what ``_layer`` returns, as the
    layer's own call would. oMLX wraps ``_layer`` before the model loads and this wraps it at attach, so oMLX's fused
    path is covered too. Flash Next's verify windows call their layers and do not need it."""
    if cls is None:
        from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward as cls
    if _patched(cls, "patch_mtp_verify"):
        return
    orig = cls._layer

    @functools.wraps(orig)
    def _layer(self, layer, hidden, *a, **k):
        return run_hooks(layer, (hidden, *a), k, orig(self, layer, hidden, *a, **k))

    _apply("patch_mtp_verify", (cls, dict(_layer=_layer)))


def patch_mtp_priming(pp=None) -> None:
    """prompt_priming._publish_boundary_candidate (oMLX's MTP patch). MTP prompt priming publishes a prefix snapshot
    that a later request with the same prompt can restore. The store itself (``store_mtp_prefix_snapshot``) has no
    request id, but this function's context does, so steered requests' snapshots are skipped here."""
    if pp is None:
        from omlx.patches.mlx_lm_mtp import prompt_priming as pp
    if _patched(pp, "patch_mtp_priming"):
        return
    orig = pp._publish_boundary_candidate

    @functools.wraps(orig)
    def _publish_boundary_candidate(ctx, *a, **k):
        if _safe(MANAGER.kv_isolated, getattr(ctx, "request_id", None)):
            return None
        return orig(ctx, *a, **k)

    _apply("patch_mtp_priming", (pp, dict(_publish_boundary_candidate=_publish_boundary_candidate)))


def patch_mtp_reconcile(bg=None) -> None:
    """batch_generator._reconcile_mtp_to_standard (oMLX's MTP patch), which re-prefills a request's history into a
    fresh cache. Those tokens are not served the way the steer asked: outside a row-set forward they are not steered
    at all, and inside MTP next() they run as decode forwards and get the decode-time add. So each steered request
    among them is marked partial (steer.py)."""
    if bg is None:
        from omlx.patches.mlx_lm_mtp import batch_generator as bg
    if _patched(bg, "patch_mtp_reconcile"):
        return
    orig = bg._reconcile_mtp_to_standard

    def partial(gen_batch):
        sched = getattr(_current, "scheduler", None)
        if sched is not None:
            for uid in gen_batch.uids:
                MANAGER.steer_partial(sched.uid_to_request_id.get(uid))

    @functools.wraps(orig)
    def _reconcile_mtp_to_standard(gen_batch, state, *a, **k):
        if MANAGER.requests:
            _safe(partial, gen_batch)
        return orig(gen_batch, state, *a, **k)

    _apply("patch_mtp_reconcile", (bg, dict(_reconcile_mtp_to_standard=_reconcile_mtp_to_standard)))

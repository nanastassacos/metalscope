"""Patches that connect metalscope to a running oMLX (the versions in install.TESTED_OMLX).

boot.py applies each patch function as its oMLX module finishes importing. Every
wrapper calls the original; when nothing is traced it adds a dict lookup or two
per scheduler step. A patch function reads every name it wraps before it assigns
any (_apply), so if oMLX dropped one the module is left as oMLX wrote it. When a
wrapper cannot read what it needs on a request's path, the request is served
untraced.

The MTP decode patches are in patches_mtp.py.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import sys
import threading
import time

from .. import config, web
from ..live import MANAGER, new_request_id
from ..middleware import REQUEST_CTX, TraceContextMiddleware, patch_uvicorn_shutdown

log = logging.getLogger("metalscope")

# The scheduler stepping on this thread (set in Scheduler.step); maps a batch's uids to request ids.
_current = threading.local()

APPLIED: set[str] = set()      # patch functions that took (/scope/api/status "patches")
FAILED: set[str] = set()       # patch functions tried at a model's attach that raised (patch_state "missing")
# Without both, a tracked request could reuse or store KV, or run forwards with no rows selected: nothing is tracked.
_CORE = {"patch_scheduler", "patch_prefix_cache"}
_FLAG = "_metalscope_patched_"
_logged: dict = {}             # _due: when each name last logged


def stop_ids(scheduler, request) -> set:
    """The token ids that end ``request``'s reply, built as oMLX's scheduler builds them: the tokenizer's EOS and
    end-of-turn ids, the output parser's (gpt-oss: <|return|>, <|call|>) and the request's own. With ``ignore_eos``
    only the request's own count, as in oMLX. Empty if oMLX's internals moved."""
    out = set()
    params = getattr(request, "sampling_params", None)
    model_eos = not getattr(params, "ignore_eos", False)
    for get in (lambda: scheduler._get_stop_tokens() if model_eos else (),
                lambda: getattr(getattr(scheduler, "_output_parser_factory", None), "stop_token_ids", None)
                if model_eos else (),
                lambda: getattr(params, "stop_token_ids", None)):
        try:
            out |= {int(t) for t in (get() or ())}
        except Exception:   # never raise: this runs outside _safe
            pass
    return out


def _due(what) -> bool:
    """True at most once every 30 s per name. Something that fails once tends to fail on every step or request."""
    now = time.monotonic()
    if now - _logged.get(what, -30.0) < 30:
        return False
    _logged[what] = now
    return True


def _failed(what) -> None:
    """Log the exception being handled, at most every 30 s (_due)."""
    if _due(what):
        log.exception("metalscope: %s failed; serving continues untraced", what)


def _safe(fn, *args, **kwargs):
    """Run metalscope bookkeeping; never let it break the scheduler."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        _failed(getattr(fn, "__name__", fn))
        return None


def _unselected(exc) -> None:
    """A forward's rows could not be selected because oMLX's internals moved. The forward runs untraced and every
    tracked request is told: its records are partial, and a steered one fails (fail closed)."""
    _failed("selecting a forward's rows")
    reason = f"rows: {type(exc).__name__}: {exc}"[:200]
    for rid in list(MANAGER.requests):
        MANAGER.mark_partial(rid, reason)
        MANAGER.steer_failed(rid, reason)


def _patched(obj, name: str) -> bool:
    """True if patch ``name`` is already on ``obj``. A module reload makes new classes, without the flag."""
    return getattr(obj, _FLAG + name, False)


def _apply(name: str, *groups) -> None:
    """Assign a patch function's wrappers, each group ``(object, {attribute: wrapper})``. All or nothing: the caller
    has read every original first, so a name oMLX dropped has raised there, before anything is assigned."""
    for obj, wrappers in groups:
        for attr, fn in wrappers.items():
            setattr(obj, attr, fn)
    setattr(groups[0][0], _FLAG + name, True)
    APPLIED.add(name)


def patch_state() -> dict:
    """Which patch functions took, and which are missing: their oMLX module is imported, or a model's attach tried
    them, yet they did not take (boot or the attach logged why)."""
    from . import boot
    return dict(applied=sorted(APPLIED),
                missing=sorted({fn for mod, fn in boot._TARGETS.items() if mod in sys.modules and fn not in APPLIED}
                               | (FAILED - APPLIED)))


# --------------------------------------------------------------- engines

def _mtp_patches(model) -> None:
    """patches_mtp, when ``model`` decodes with MTP (mtp.mtp_decode); raises when one the traces need did not take."""
    from ..mtp import mtp_decode
    from . import patches_mtp
    if not mtp_decode(model):
        return
    for fn in ("patch_mtp_next", "patch_mtp_verify", "patch_mtp_priming", "patch_mtp_reconcile"):
        try:
            getattr(patches_mtp, fn)()
        except Exception:
            FAILED.add(fn)              # patch_state "missing": status and doctor say which
            if fn != "patch_mtp_verify":
                raise
            # Qwen3.5 family only: the rest is still traced
            log.exception("metalscope: patch_mtp_verify not applied: a Qwen3.5-family model's MTP "
                          "verify windows go without lens rows or residual norms, and unsteered")


def _patch_engine(name: str, cls, model_of) -> None:
    """Wrap an engine class's start and stop: attach a tracer to the model the engine hands its scheduler once it
    has started, and detach it before it stops. ``model_of(engine)`` is that model, the object EngineCore.add_request
    and the scheduler see (requests find their tracer by its id). An MTP-decoding model also gets patches_mtp: at the
    start, or when oMLX gives it its drafter afterwards (``set_dflash_drafter``: DSpark and DFlash drafts, which the
    engine pool loads once the engine runs; the tracer reads ``mtp`` live, so it needs no new hooks)."""
    if _patched(cls, name):
        return
    orig_start, orig_stop, orig_drafter = cls.start, cls.stop, cls.set_dflash_drafter

    @functools.wraps(orig_start)
    async def start(self, *a, **k):
        result = await orig_start(self, *a, **k)
        try:
            model = model_of(self)
            if model is not None:
                _mtp_patches(model)         # first: if one raises, the model gets no tracer
                MANAGER.attach(model, self._model_name, self.tokenizer)
        except Exception:
            log.exception("metalscope: attach failed for %s", getattr(self, "_model_name", "?"))
        return result

    @functools.wraps(orig_drafter)
    def set_dflash_drafter(self, drafter):
        result = orig_drafter(self, drafter)
        try:
            _mtp_patches(model_of(self))
        except Exception:
            log.exception("metalscope: MTP patches failed for %s: its traces will not line up",
                          getattr(self, "_model_name", "?"))
        return result

    def detach(engine):
        model = model_of(engine)
        if model is not None:
            MANAGER.detach(model)

    @functools.wraps(orig_stop)
    async def stop(self, *a, **k):
        _safe(detach, self)
        return await orig_stop(self, *a, **k)

    _apply(name, (cls, dict(start=start, stop=stop, set_dflash_drafter=set_dflash_drafter)))


def patch_vlm_engine(module) -> None:
    """VLMBatchedEngine (vision-language models): the scheduler runs oMLX's VLMModelAdapter around the model."""
    _patch_engine("patch_vlm_engine", module.VLMBatchedEngine, lambda e: e._adapter)


def patch_batched_engine(module) -> None:
    """BatchedEngine (text-only models on mlx_lm's own classes: gpt-oss, GLM, DeepSeek, Qwen text MoE): the
    scheduler runs the loaded model itself."""
    _patch_engine("patch_batched_engine", module.BatchedEngine, lambda e: e._model)


# --------------------------------------------------------------- engine core and edits

NO_EDIT = "oMLX cannot edit this model's replies yet"
_DETOKENIZERS = False          # set by patch_scheduler: oMLX's scheduler keeps a detokenizer per request (_continued)


def edit_caps() -> frozenset:
    """``{"edit_at"}`` once an edit can run, else empty. It needs patch_engine_core and patch_scheduler (the engine
    gets token ids, replies are kept) and a detokenizer per request to start the new text in (omlx/edit.py)."""
    return frozenset({"edit_at"}) if _DETOKENIZERS and {"patch_engine_core", "patch_scheduler"} <= APPLIED \
        else frozenset()


SEEDED_PARSERS = ("inkling",)  # oMLX output parsers an edit's request is seeded through (_continued): Inkling 2's


def _refuses_edit(sched) -> bool:
    """True if oMLX decodes this scheduler's replies through a protocol parser _continued does not seed (DeepSeek
    V4, gpt-oss, Gemma 4 and others: its own detokenizer and state). An edit is then refused (status ``chat`` says
    ``edit`` False). Inkling 2's parser (SEEDED_PARSERS) is seeded."""
    factory = getattr(sched, "_output_parser_factory", None)
    return factory is not None and getattr(factory, "kind", None) not in SEEDED_PARSERS


def _continued(sched, request_id, ids) -> str | None:
    """Seed the detokenizer of a request that continues a reply (an edit, edit.py) with the reply's tokens so far.
    Its first piece then keeps a leading space (a fresh detokenizer trims one) and no character is split at the
    join. Returns the text of those tokens, as the client got it. On a parser oMLX decodes through (SEEDED_PARSERS:
    Inkling 2's, its channel and think state, markers out and ``<think>``/``</think>`` in) the request's parser
    session is seeded instead: the kept tokens' text is its ``stream_text``, and the request's own tokens continue in
    the channel they were in. None when the edit is refused (_refuses_edit) or oMLX has no detokenizer or parser
    session for the request."""
    if _refuses_edit(sched):
        return None
    if getattr(sched, "_output_parser_factory", None) is not None:
        s = sched._get_output_parser_session(request_id)
        return None if s is None else "".join(s.process_token(int(t)).stream_text for t in ids)
    d = sched._get_detokenizer(request_id)
    if d is None:
        return None
    for t in ids:
        d.add_token(int(t))
    _ = d.last_segment                           # consumed: the request's text starts after it
    return d.text


def _forget(core, request_id) -> None:
    """Drop the detokenizer or parser session _continued made for a request oMLX never took (oMLX drops one only
    with its request)."""
    _safe(core.scheduler._cleanup_detokenizer, request_id)
    if callable(drop := getattr(core.scheduler, "_cleanup_output_parser_session", None)):
        _safe(drop, request_id)


async def _continues(core, request_id, ctx: dict) -> None:
    """An edit's completion (omlx/edit.py) reaching oMLX: put its request id and its kept tokens' text in its
    context, then tell the edit it started. The text comes from _continued, run off the event loop (a detokenizer
    reads the vocabulary). If cancelled meanwhile, the detokenizer is dropped once made."""
    ctx["rid"] = request_id
    made = asyncio.ensure_future(asyncio.to_thread(_safe, _continued, core.scheduler, request_id, ctx["base"][1]))
    try:
        ctx["text"] = await asyncio.shield(made)
    except asyncio.CancelledError:
        made.add_done_callback(lambda _f: _forget(core, request_id))
        raise
    finally:
        ctx["started"].set()


def _keep(rid, req) -> None:
    """Keep a traced reply's token ids for an edit (edit.py). Called before finish, so an edit waiting for it finds
    it. The prompt as it came in (a memory requeue appends the reply so far to it) and the reply (oMLX leaves the
    stop token out)."""
    rt = MANAGER.requests.get(rid)
    if rt is None or rt.level is None:
        return
    prompt = list(req.prompt_token_ids or ())
    prompt = prompt[:len(prompt) - (getattr(req, "requeued_output_tokens", 0) or 0)]
    MANAGER.kept.put(rid, prompt, list(req.output_token_ids or ()), rt.base, rt.meta.get("model"))


def patch_engine_core(module) -> None:
    """EngineCore.add_request and _cleanup_request, for every model: register traced and steered API requests,
    give an edit's completion its token ids, and release a request oMLX dropped without finishing it."""
    cls = module.EngineCore
    if _patched(cls, "patch_engine_core"):
        return
    orig, orig_cleanup = cls.add_request, cls._cleanup_request

    @functools.wraps(orig)
    async def add_request(self, prompt, sampling_params=None, request_id=None, *a, **k):
        # No context: the middleware never saw it. oMLX's benchmark and evaluation routes call the engine directly;
        # they stay untraced, so they keep their prefix reuse.
        ctx = REQUEST_CTX.get()
        level = MANAGER.resolve_level(ctx.get("header")) if ctx is not None else None
        spec = ctx.get("steer") if ctx is not None else None    # steer.SteerSpec, validated by the middleware
        if level is None and spec is None:
            return await orig(self, prompt, sampling_params, request_id, *a, **k)
        request_id = request_id or new_request_id()
        base = ctx.get("base")
        if base is None:
            return await tracked(self, prompt, sampling_params, request_id, ctx, level, spec, a, k)
        # An edit's completion (omlx/edit.py): oMLX's route had only the text, so give oMLX the token ids. The
        # request's text must start after the kept tokens' text, or the edit is refused.
        await _continues(self, request_id, ctx)
        try:
            if ctx["text"] is None:
                from fastapi import HTTPException
                raise HTTPException(400, NO_EDIT)
            return await tracked(self, [*base[0], *base[1]], sampling_params, request_id, ctx, level, spec, a, k)
        except BaseException:
            if request_id not in (getattr(self.scheduler, "requests", None) or ()):
                _forget(self, request_id)          # refused before oMLX took it: drop its detokenizer too
            raise

    async def tracked(self, prompt, sampling_params, request_id, ctx, level, spec, a, k):
        try:
            model = getattr(self, "model", None)
            if not _CORE <= APPLIED:
                raise RuntimeError(f"oMLX is not fully patched (missing {', '.join(sorted(_CORE - APPLIED))})")
            if spec is not None and MANAGER.tracer_for(model) is None:
                raise RuntimeError("metalscope has no hooks on this model")
            MANAGER.register(request_id, level, dict(ctx.get("meta") or {}), model=model, steer=spec)
            if ctx.get("base") is not None:
                MANAGER.requests[request_id].base = ctx["base"]
        except Exception as exc:
            # register leaves nothing behind when it raises. Fail closed: a steered request is refused (a 503 with
            # the reason; an open stream ends with oMLX's error event), never served unsteered.
            if spec is not None:
                log.warning("metalscope: steered request refused: %s", exc)   # one line: install logged the traceback
                from fastapi import HTTPException
                raise HTTPException(503, f"{config.NO_STEERING}: {exc}") from exc
            if _due("an untraced request"):
                log.warning("metalscope: request served untraced: %s", exc)
            return await orig(self, prompt, sampling_params, request_id, *a, **k)
        if spec is not None:
            # oMLX's own flag (its benchmark probes set it): no KV store is prepared for a steered request at its
            # end or on cancel. patch_prefix_cache refuses the stores this flag does not reach.
            k["skip_cache_store"] = True
        try:
            return await orig(self, prompt, sampling_params, request_id, *a, **k)
        except BaseException:
            # oMLX refused it (prefill capacity, a failed insert, the client left): it never runs.
            _safe(MANAGER.release, request_id)
            raise

    # The consumer's own cleanup (stream_outputs' finally, generate's end, a refused add_request) runs on every path
    # once oMLX is done with a request. A finished one was saved by _cleanup_finished. One the scheduler dropped
    # without finishing (a failed prefill, an engine-loop error, an abort) is released here, so it is not left
    # active. One the scheduler still holds (a pending deferred abort, a timed-out stream) is left to the scheduler:
    # _cleanup_finished saves it or _do_abort_request releases it.
    @functools.wraps(orig_cleanup)
    def _cleanup_request(self, request_id, *a, **k):
        try:
            return orig_cleanup(self, request_id, *a, **k)
        finally:
            if MANAGER.requests and MANAGER.is_traced(request_id):
                held = getattr(getattr(self, "scheduler", None), "requests", None)
                if not held or request_id not in held:
                    _safe(MANAGER.release, request_id)

    _apply("patch_engine_core", (cls, dict(add_request=add_request, _cleanup_request=_cleanup_request)))


# --------------------------------------------------------------- scheduler

def _prefill(sched, requests_of, run):
    """Run a prefill forward ``run()`` of the requests ``requests_of()`` gives. One request: its rows are selected
    when it is tracked. Several share one packed forward, which row selection cannot split: it runs untraced and
    each tracked request in it is marked partial."""
    tracer = None
    if MANAGER.requests:
        try:
            requests = requests_of()
            rids = [r.request_id for r in requests]
            if len(rids) > 1:
                for rid in rids:
                    MANAGER.steer_partial(rid)
                    MANAGER.mark_partial(rid, "a prefill chunk ran packed with other requests, untraced")
            elif any(map(MANAGER.is_traced, rids)):
                tracer = MANAGER.tracer_for(sched.model)
                if tracer is not None:
                    MANAGER.note_prompt(rids[0], len(requests[0].prompt_token_ids or ()), stop_ids(sched, requests[0]))
                    MANAGER.rows_for(tracer, rids, "prefill")
        except Exception as exc:
            _unselected(exc)
    if tracer is None:
        return run()
    try:
        return run()
    finally:
        _safe(MANAGER.clear_rows, tracer)


def _batch_rows(batch):
    """``(tracer, request id of each row, scheduler)`` for a decode batch that holds a tracked request, else None."""
    sched = getattr(_current, "scheduler", None)
    if sched is None or not MANAGER.requests:
        return None
    try:
        rids = [sched.uid_to_request_id.get(uid) for uid in batch.uids]
        tracer = MANAGER.tracer_for(batch.model) if any(map(MANAGER.is_traced, rids)) else None
        return (tracer, rids, sched) if tracer is not None else None
    except Exception as exc:
        return _unselected(exc)


def patch_scheduler(module) -> None:
    """Scheduler and GenerationBatch, for every model: select the traced rows for each prefill and decode forward,
    keep traced requests out of packed and helper prefill, and save or release them when oMLX finishes, aborts or
    fails them."""
    global _DETOKENIZERS
    sched_cls, batch_cls = module.Scheduler, module.GenerationBatch
    if _patched(sched_cls, "patch_scheduler"):
        return
    orig_step, orig_prefill, orig_run = sched_cls.step, sched_cls._do_external_prefill, sched_cls._run_prefill_chunks
    orig_add, orig_cleanup = sched_cls.add_request, sched_cls._cleanup_finished
    orig_abort, orig_fail_all = sched_cls._do_abort_request, sched_cls.fail_all_requests
    orig_helper = sched_cls._start_helper_prefill
    orig_batch_step = batch_cls._step       # already oMLX's wrapper (rope deltas, row realignment): ours goes outside

    @functools.wraps(orig_step)
    def step(self, *a, **k):
        _current.scheduler = self
        try:
            return orig_step(self, *a, **k)
        finally:
            _current.scheduler = None

    @functools.wraps(orig_prefill)
    def _do_external_prefill(self, request, *a, **k):
        return _prefill(self, lambda: [request], lambda: orig_prefill(self, request, *a, **k))

    # Chunked prefill (one chunk per scheduler step, used whenever another request is decoding) runs every chunk
    # through here. On models with packed prefill (qwen3_5, qwen3_5_moe, qwen4_exp, glm5_next and hy_v3, unless
    # expert offload or a block drafter is on) that includes the chunks _advance_chunked_prefills plans itself and
    # the generation-prompt tail. One plan is one request's chunk. add_request below keeps traced requests out of
    # packed forwards; if a pack holds one anyway, _prefill marks it partial.
    @functools.wraps(orig_run)
    def _run_prefill_chunks(self, plans, *a, **k):
        return _prefill(self, lambda: [p.state.request for p in plans], lambda: orig_run(self, plans, *a, **k))

    # Keep traced requests out of packed prefill with oMLX's own flag (set for a request that failed in a pack):
    # they prefill alone, through the wrappers above. An edit's request whose parser session _continued seeded: its
    # kept text already holds the thinking's opener, so oMLX's own (``needs_think_prefix``, when the kept tokens end
    # just after Inkling's <|content_thinking|>) is marked sent and never added.
    @functools.wraps(orig_add)
    def add_request(self, request, *a, **k):
        rid = getattr(request, "request_id", None)
        if MANAGER.requests and MANAGER.is_traced(rid):
            _safe(setattr, request, "packed_prefill_excluded", True)
            if rid in (getattr(self, "_output_parser_sessions", None) or ()):
                _safe(setattr, request, "think_prefix_sent", True)
        return orig_add(self, request, *a, **k)

    # oMLX 0.8's Prefill Accelerator (off unless a model's settings name a helper Mac) runs each prompt chunk's first
    # layers on a paired Mac and the rest here without a top-level forward: the helper's layers go unrecorded and
    # unsteered, the rest's records land in the next forward. Tracked requests prefill on this Mac (False is oMLX's
    # own answer when no helper takes a request).
    @functools.wraps(orig_helper)
    def _start_helper_prefill(self, request, *a, **k):
        if MANAGER.requests and MANAGER.is_traced(getattr(request, "request_id", None)):
            return False
        return orig_helper(self, request, *a, **k)

    @functools.wraps(orig_cleanup)
    def _cleanup_finished(self, finished_ids, *a, **k):
        def tracked():
            return [(rid, self.requests.get(rid)) for rid in finished_ids if MANAGER.is_traced(rid)]
        done = _safe(tracked) if MANAGER.requests else None
        result = orig_cleanup(self, finished_ids, *a, **k)
        if done:
            tracer = _safe(lambda: MANAGER.tracer_for(self.model))
            for rid, req in done:
                if req is not None:
                    _safe(_keep, rid, req)
                    _safe(MANAGER.finish, rid, req.prompt_token_ids, req.output_token_ids, tracer)
                else:
                    _safe(MANAGER.release, rid)
        return result

    # Requests the scheduler drops without _cleanup_finished: an abort (the client left, memory pressure) and the
    # engine loop's safety net after step() raised. These run once per abort or failure, not per step. An abort
    # always releases. A running request kept for a resend was already saved by the _cleanup_finished it ran inside
    # the abort (the release is then a no-op). One left in oMLX's requests for a deferred cache store (a cancel
    # mid-prefill) never comes back through us; a steered one stays tombstoned for that store.
    @functools.wraps(orig_abort)
    def _do_abort_request(self, request_id, *a, **k):
        try:
            return orig_abort(self, request_id, *a, **k)
        finally:
            if MANAGER.is_traced(request_id):
                _safe(MANAGER.release, request_id)

    @functools.wraps(orig_fail_all)
    def fail_all_requests(self, *a, **k):
        failed = orig_fail_all(self, *a, **k)
        for rid in failed or ():
            if MANAGER.is_traced(rid):
                _safe(MANAGER.release, rid)
        return failed

    @functools.wraps(orig_batch_step)
    def _step(self, *a, **k):
        # Inside MTP next() (patches_mtp) the rows are already selected.
        found = None if getattr(_current, "outer_rows", False) else _batch_rows(self)
        if found is None:
            return orig_batch_step(self, *a, **k)
        _safe(MANAGER.rows_for, found[0], found[1], "decode")
        try:
            return orig_batch_step(self, *a, **k)
        finally:
            _safe(MANAGER.clear_rows, found[0])

    _apply("patch_scheduler",
           (sched_cls, dict(step=step, _do_external_prefill=_do_external_prefill, _run_prefill_chunks=_run_prefill_chunks,
                            add_request=add_request, _start_helper_prefill=_start_helper_prefill,
                            _cleanup_finished=_cleanup_finished,
                            _do_abort_request=_do_abort_request, fail_all_requests=fail_all_requests)),
           (batch_cls, dict(_step=_step)))
    _DETOKENIZERS = all(callable(getattr(sched_cls, n, None)) for n in ("_get_detokenizer", "_cleanup_detokenizer"))


# --------------------------------------------------------------- prefix cache

def _kv_isolated(request_id) -> bool:
    """MANAGER.kv_isolated for a prefix-cache store id. oMLX stores some of a request's KV under its id plus a
    suffix, ``<id>#requeue<n>`` (a request requeued under memory pressure) and ``<id>#system`` (the shared
    system-prompt tail); those are the request's own KV too."""
    if MANAGER.kv_isolated(request_id):
        return True
    return isinstance(request_id, str) and "#" in request_id and MANAGER.kv_isolated(request_id.split("#", 1)[0])


def patch_prefix_cache(module) -> None:
    """BlockAwarePrefixCache, for every model. Traced requests skip prefix reuse, so cached tokens never bypass the
    layers. Steered requests also never store their KV and never take an exact prefix. oMLX's ``skip_cache_store``
    (set in patch_engine_core) stops the stores it prepares at a request's end or cancel. ``store_cache`` is refused
    here for the one store that flag misses (``<id>#requeue<n>``, a memory requeue's) and as a backstop for the
    rest: a store can run on a worker after the request finished, hence the tombstone in MANAGER.kv_isolated.
    ``store_exact_prefix`` stores through ``store_cache``."""
    cls = module.BlockAwarePrefixCache
    if _patched(cls, "patch_prefix_cache"):
        return
    orig_fetch, orig_store, orig_exact = cls.fetch_cache, cls.store_cache, cls.fetch_exact_prefix

    @functools.wraps(orig_fetch)
    def fetch_cache(self, request_id, tokens, *a, **k):
        # oMLX's system-tail store first fetches under "<id>#system" to anchor its block table. Refusing that for a
        # steered request stops the store there (its store_cache is refused too).
        if MANAGER.is_traced(request_id) or _kv_isolated(request_id):
            return None, tokens
        return orig_fetch(self, request_id, tokens, *a, **k)

    def guarded(orig_fn):
        @functools.wraps(orig_fn)
        def wrapper(self, request_id, *a, **k):
            return None if _kv_isolated(request_id) else orig_fn(self, request_id, *a, **k)
        return wrapper

    _apply("patch_prefix_cache", (cls, dict(fetch_cache=fetch_cache, store_cache=guarded(orig_store),
                                            fetch_exact_prefix=guarded(orig_exact))))


# --------------------------------------------------------------- engine pool (GPU lock)

# Set by patch_engine_pool. /scope/api/status reports it; concept learn requires it before unloading a model.
GPU_LOCK_ENFORCED = False
# The model get_engine is getting, in a list (emptied when it returns; tasks it started keep the context).
_ADMITTING: contextvars.ContextVar = contextvars.ContextVar("metalscope_admitting", default=None)


def _admission_size(pool, model_id, settings) -> int:
    """The size oMLX's own admission counts for this load: the resident size under the model's settings (expert
    offload, drafters), not the files' size."""
    if settings is None and pool._settings_manager is not None:
        settings = pool._settings_manager.get_settings(model_id)
    return int(pool._entry_runtime_resident_size(pool.get_entry(model_id), settings))


def _job_elsewhere(pool, model_id) -> str | None:
    """The running job, as the refusal says it, when it runs on the other backend: its model is not one of this
    pool's (a calibration on ds4 holds no GPU lock, and a model loaded here would sit beside ds4's). A job on one of
    oMLX's own models refuses nothing: its requests load its model, a learn job reloads it."""
    from ..learn import state
    job = state.running()
    model = config.short_name((job or {}).get("model"))      # a job from the terminal keeps the name as typed (org/NAME)
    if not model or model == config.short_name(model_id) or pool.get_entry(model) is not None:
        return None
    return state.running_text(job)


def patch_engine_pool(module) -> None:
    """EnginePool, for every model: refuse model loads while the GPU lock is held (gpulock.py) or a job runs on the
    other backend (``_job_elsewhere``), with oMLX's own ModelUnavailableError (a 409 with the message). After a run
    that was killed rather than released (``gpulock.unclean``), also refuse while free memory is below what the load
    needs plus a margin: the dead run's memory may still be coming back.

    Every load goes through ``_load_engine``: chat requests (via ``get_engine``), the admin load route, the pinned
    preload at startup and the settings auto-reload. Inside ``get_engine`` an unload can make room for a load (the
    model's own engine for a runtime-settings variant or force_lm, other models by LRU), so while the lock is held
    ``_unload_engine`` is refused there too, first. A held lock never leaves a served model unloaded. The memory
    check stays with the load, since an eviction is what frees memory. Requests to a loaded model load nothing and
    pass. Raising before oMLX's code runs leaves the pool entry untouched (no cached load failure), so the model
    loads normally once the lock is free.
    """
    global GPU_LOCK_ENFORCED
    from .. import gpulock
    cls, unavailable = module.EnginePool, module.ModelUnavailableError
    if _patched(cls, "patch_engine_pool"):
        return
    orig_load, orig_get, orig_unload = cls._load_engine, cls.get_engine, cls._unload_engine

    def refuse_if_held(pool, model_id) -> None:
        holder = _safe(gpulock.held)
        if holder is not None:
            raise unavailable(model_id, f"Model '{model_id}' cannot load now: {holder.message()}")
        job = _safe(_job_elsewhere, pool, model_id)
        if job:
            raise unavailable(model_id, f"Model '{model_id}' cannot load now: {job}; models load again when it is done")

    @functools.wraps(orig_load)
    async def _load_engine(self, model_id, *a, **k):
        refuse_if_held(self, model_id)
        need = _safe(_admission_size, self, model_id, k.get("runtime_settings")) if _safe(gpulock.unclean) else None
        avail = await asyncio.to_thread(_safe, gpulock.available) if need else None     # vm_stat: off the event loop
        if avail is not None and avail < need + 8 * gpulock.GB:
            raise unavailable(
                model_id, f"Model '{model_id}' cannot load yet: a run that held the GPU stopped without releasing it, and only "
                          f"{avail / gpulock.GB:.0f} GB is free; try again in a minute")
        return await orig_load(self, model_id, *a, **k)

    @functools.wraps(orig_get)
    async def get_engine(self, model_id, *a, **k):
        admitting = [model_id]
        token = _ADMITTING.set(admitting)
        try:
            return await orig_get(self, model_id, *a, **k)
        finally:
            admitting.clear()
            _ADMITTING.reset(token)

    @functools.wraps(orig_unload)
    async def _unload_engine(self, model_id, *a, **k):
        for admitting in _ADMITTING.get() or ():
            refuse_if_held(self, admitting)
        return await orig_unload(self, model_id, *a, **k)

    _apply("patch_engine_pool", (cls, dict(_load_engine=_load_engine, get_engine=get_engine, _unload_engine=_unload_engine)))
    GPU_LOCK_ENFORCED = True


# --------------------------------------------------------------- server: /scope

def patch_server(module) -> None:
    """oMLX's app: mount the /scope routes and add the trace middleware. Must run before the app starts."""
    app = module.app
    if _patched(app, "patch_server"):
        return
    if app.middleware_stack is not None:
        raise RuntimeError("oMLX's app has started: too late to add the middleware")
    from starlette.middleware import Middleware
    from . import backend       # here, not at the top: backend imports this module
    config.private_home()
    patch_uvicorn_shutdown()
    web.mount(app, backend.BACKEND)   # /scope, ahead of any catch-all
    backend._SERVER = module
    # Last, so nothing is traced until the rest is in. Innermost, inside every middleware oMLX adds: its
    # RequestBodySizeLimitMiddleware has refused or cut an oversized body before ours buffers one
    # (app.add_middleware would put ours outside it).
    from .edit import answer
    app.user_middleware.append(Middleware(TraceContextMiddleware, resolve_model=backend.resolve_model, edits=answer))
    _apply("patch_server", (app, dict()))
    log.info("metalscope: active (mode=%s); routes at %s", MANAGER.mode(), web.PREFIX)

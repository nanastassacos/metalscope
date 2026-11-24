"""The oMLX backend of the /scope routes (web.Backend): the models oMLX serves, a reading on the served model, oMLX's
API key check, and which patches took. Read through oMLX's server module, which patch_server hands it.
"""

from __future__ import annotations

import asyncio
import functools

from .. import config, models, web
from . import patches
from .patches import _refuses_edit, _safe, edit_caps, patch_state

_SERVER = None                 # oMLX's server module (set by patch_server); its engine pool lists the served models


@functools.lru_cache(maxsize=256)
def _chat_template(model_path: str) -> str | None:
    """A model directory's chat template: chat_template.jinja, else tokenizer_config.json's (a string, or a list of
    named templates, all joined)."""
    from pathlib import Path
    d = Path(model_path)
    try:
        return (d / "chat_template.jinja").read_text()
    except OSError:
        pass
    t = config.read_json(d / "tokenizer_config.json", {}).get("chat_template")
    if isinstance(t, list):
        t = "\n".join(str(x.get("template") or "") for x in t if isinstance(x, dict))
    return t if isinstance(t, str) else None


def chat_format(template: str | None) -> dict:
    """A model's thinking controls, from its chat template. One that reads ``enable_thinking`` takes it in
    ``chat_template_kwargs`` (Pi's thinking format "qwen-chat-template"). One that only writes <think> always
    thinks (reasoning, no switch). Otherwise none."""
    t = template or ""
    if "enable_thinking" in t:
        return dict(reasoning=True, thinking_format="qwen-chat-template")
    return dict(reasoning="<think>" in t, thinking_format=None)


def _scheduler(entry):
    """A pool entry's scheduler while its model is loaded, else None. The same path oMLX takes to it
    (BatchedEngine's supports_early_tool_call_streaming)."""
    return getattr(getattr(getattr(getattr(entry, "engine", None), "_engine", None), "engine", None), "scheduler", None)


def chat_formats() -> dict:
    """Every model oMLX serves, loaded or not: its thinking controls (chat_format), whether metalscope was tested on
    its family (its config_model_type in models.TESTED) and, when loaded, whether it takes an edit (not
    _refuses_edit: Inkling 2's parser is seeded, other parsers refuse; a model not loaded has no ``edit``, since oMLX
    has not decided yet). Empty before the server module is in, or when its pool has moved."""
    def edit(e) -> dict:
        s = _scheduler(e)
        return {} if s is None else dict(edit=not _refuses_edit(s))
    try:
        pool = _SERVER.get_server_state().engine_pool
        entries = ((mid, pool.get_entry(mid)) for mid in pool.get_model_ids())
        return {mid: dict(chat_format(_chat_template(str(e.model_path)) if e.model_path else None),
                          tested=models.tested("omlx", getattr(e, "config_model_type", None)), **edit(e))
                for mid, e in entries if e is not None}
    except Exception:
        return {}


def resolve_model(name: str) -> str:
    """The model id oMLX serves under ``name``. Its /v1/models lists aliases too (a model's settings alias, a
    provider prefix), and steering lookups and saved traces need the served model's own id. ``name`` itself when
    oMLX has no resolver or does not know it."""
    resolve = getattr(_SERVER, "resolve_model_id", None)
    out = _safe(resolve, name) if callable(resolve) else None
    return out if isinstance(out, str) and out else name


ROOM_MARGIN_GB = 2.0      # free memory an engram inspection leaves beside its own (MLX's buffer pool, Python's heap)


def _room_for(scheduler, eng, n: int, rows: int) -> None:
    """Refuse a reading of ``n`` tokens in chunks of ``rows`` that would leave less than ROOM_MARGIN_GB free. Its
    cost: its own KV cache and attention, priced as oMLX's memory guard prices an n-token prompt (the scheduler's
    MemoryMonitor; Flash Next is 24.6 KB a token), plus a chunk's logits (MlxEngine.chunk_bytes). Free memory is
    what the GPU jobs read (gpulock.available) plus MLX's buffer pool in this process, which the reading uses first.
    Not checked when there is no price (no monitor, or it does not know the model)."""
    from .. import gpulock
    from ..lookup import Refused
    price = getattr(getattr(scheduler, "memory_monitor", None), "estimate_prefill_peak_bytes", None)
    kv = _safe(price, n, rows) if callable(price) else None
    if not kv:
        return
    import mlx.core as mx
    need, free = int(kv) + eng.chunk_bytes(rows) + int(ROOM_MARGIN_GB * gpulock.GB), gpulock.available()
    if free is None:
        return
    free += mx.get_cache_memory()
    if free < need:
        raise Refused(409, f"not enough free memory: this needs about {need / gpulock.GB:.1f} GB, "
                           f"{free / gpulock.GB:.1f} GB is free")


async def lookup_lp(tracer, ids: list, start: int, on: bool, job):
    """lookup.Measure on the served model itself, in an MlxEngine (engines/mlx.py) with a fresh cache of its own.
    Each chunk runs on the engine's MLX executor between oMLX's own steps, under an engine lease so oMLX cannot
    unload the model meanwhile (oMLX's speech engines run their model work the same way). Before each chunk ``job``
    is told the share done (Run.part), or, if called off, the reading stops there. oMLX gives back expert slots on
    the same executor under memory pressure: if they no longer hold a chunk's rows (MlxEngine.lookup_chunk), the
    reading is refused, never read expert-major. Before the first pass, a reading that would not fit in free memory
    is refused (_room_for)."""
    from ..engines.mlx import MlxEngine
    from ..lookup import Refused, whole
    pool = _SERVER.get_server_state().engine_pool
    # The tracer is named by the model's path (the engine's model_name), the pool by its id.
    mid = next((m for m in pool.get_model_ids() if str(getattr(pool.get_entry(m), "model_path", "")) == tracer.name),
               None)
    if mid is None:
        raise RuntimeError(f"oMLX's pool has no model at {tracer.name}")
    async with pool.acquire(mid) as engine:
        core = getattr(getattr(engine, "_engine", None), "engine", None)
        executor = getattr(core, "_mlx_executor", None)
        if executor is None:
            raise RuntimeError("oMLX's engine has no MLX executor")
        eng = MlxEngine(tracer.model, tracer.tokenizer, config.short_name(tracer.name))
        rows = eng.lookup_chunk()
        if on:          # the first pass; the second reuses its memory (MLX keeps it cached)
            await asyncio.to_thread(_room_for, getattr(core, "scheduler", None), eng, len(ids), rows)
        n, k = -(-len(ids) // rows), 0

        def chunk(fn):
            nonlocal k
            job.check()
            job.part(k / n, (k + 1) / n)
            k += 1
            return executor.submit(fitted, fn).result()

        def fitted(fn):     # on the executor: no slots are given back between this check and the forward
            if eng.lookup_chunk() < rows:
                raise Refused(409, "oMLX gave back expert slots during the reading: ask again")
            return fn()
        # If cancelled (the client left), it still waits for the pass: the lease and the route's locks outlast it.
        return await whole(eng.next_logprobs, ids, start, on, chunk, rows)


async def check_key(request) -> None:
    """oMLX's own API key check in front of every /scope route, while oMLX has a key set: its verify_api_key (the main
    key or a sub key, as Bearer or x-api-key; its loopback-only skip setting honoured). Nothing without a key. A
    missing or wrong key: 401 saying what to send; an oMLX whose check moved: 503, never let through."""
    from fastapi import HTTPException
    server = _SERVER
    if server is None:                   # not patched into a server (a test app)
        return
    try:
        if server.get_server_state().api_key is None:
            return
        verify, bearer = server.verify_api_key, server.security
    except AttributeError:
        raise HTTPException(status_code=503, detail="metalscope cannot check oMLX's API key in this oMLX version: "
                                                    "run `metalscope doctor`")
    try:
        await verify(request, await bearer(request))
    except HTTPException as exc:
        if exc.status_code == 401:
            raise HTTPException(status_code=401, detail=web.NEEDS_KEY) from None
        raise


# The oMLX backend of the /scope routes. oMLX runs the model in this process, so a GPU job can unload it
# (patch_engine_pool guards the loads); a reply can be answered again from one token once the patches that needs
# took (edit_caps, omlx/edit.py); a saved reply can be read again without the n-gram lookup (lookup_lp, lookup.py);
# every route needs oMLX's API key while one is set (check_key).
BACKEND = web.Backend(engine="omlx", caps=frozenset({"load_copy"}), chat_formats=chat_formats,
                      gpu_lock_enforced=lambda: patches.GPU_LOCK_ENFORCED, patches=patch_state, more_caps=edit_caps,
                      lookup=lookup_lp, credential=check_key)

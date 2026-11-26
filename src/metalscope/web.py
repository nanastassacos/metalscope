"""HTTP routes under /scope, mounted on the backend's server: oMLX's FastAPI
app (omlx/patches.py patch_server) or the ds4 proxy (ds4/proxy.py).

The routes, in this order below: status, mode and the live stream; a saved reply's routing (/api/traces/{id}/routing)
and its n-gram lookup's help (/api/lookup); the experts; concepts and steering; the GPU jobs and the J-lens (/api/jobs,
/api/concept/learn|calibrate|delete, /api/lens/...). Each route's docstring says what it answers.

Every route here is the same on both backends (the ds4 proxy adds /scope/api/ds4 of its own); what one cannot do it
declares by capability (``Backend.caps``), and a route that needs a capability the backend lacks answers 409 saying
what is missing (never by engine name).

Traces contain conversation text, so every route is loopback-only regardless
of the address the server binds to. The client address alone is not enough:
oMLX answers every origin (CORS ``*``), so a web page open in the user's
browser could read traces or switch tracing on. Browser requests therefore
also need loopback ``Host`` and, when present, ``Origin`` headers, and must
not be marked cross-site; see ``middleware._local_only``. A route that reads a body
takes it only as ``application/json`` (FastAPI refuses a text/plain one), so
a page cannot send it without a CORS preflight. The ds4 proxy puts the same
check in front of every route it serves (ds4/proxy.py ``LocalOnly``).

After those checks, on oMLX with an API key set, every route also needs that key, as oMLX's own API does
(``Backend.credential``). The ds4 proxy has no key: oMLX's is never sent to its port (client.omlx_key).
"""

from __future__ import annotations

import asyncio
import threading
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from . import config, models, vectors
from .store import read_index
from .live import MANAGER
from .middleware import _local_only


NEEDS_KEY = "oMLX's API key required: send Authorization: Bearer <key>, the key in ~/.omlx/settings.json"


async def _credential(request: Request) -> None:
    """The backend's key check (``Backend.credential``), after the local-only one: any local process could otherwise
    read the live stream, switch steering or start GPU jobs."""
    check = _backend(request).credential
    if check is not None:
        await check(request)


PREFIX = "/scope"
# No prefix of its own: ``mount`` includes it under PREFIX.
router = APIRouter(dependencies=[Depends(_local_only), Depends(_credential)], include_in_schema=False)


@dataclass(frozen=True)
class Backend:
    """The server the routes are mounted on (omlx/backend.py BACKEND, ds4/proxy.py BACKEND), as /scope/api/status
    reports it: ``engine`` its name; ``caps`` what it can do beyond tracing and steering (which both do):

        load_copy      a GPU job may unload the served model and load its own MLX copy (concept learn, J-lens learn)
        measure_served a GPU job may measure the served model where it is, chat refused meanwhile (concept learn on
                       the engine's own requests, engines/ds4.py)
        token_prompts  /v1/completions feeds a prompt of token ids as given (ids from POST /v1/scope/tokenize); a
                       text prompt would be wrapped in the chat template (learn/calibrate.py sends its rendered
                       conversations so)
        edit_at        a streamed chat completion with X-Scope-Edit answers an earlier reply again from one token, the
                       edit at that token only (edit.py; Pi's dock word edit)
        lookup         POST /api/lookup reads a saved reply again with and without the model's n-gram lookup
                       (lookup.py); listed while a served model has one and the backend can measure it (``lookup``)

    ``chat_formats()`` each served model's thinking controls and flags, ``{id: {reasoning, thinking_format, tested,
    edit}}`` (Pi reads them): ``tested`` whether metalscope was run on its family here (models.TESTED), ``edit``
    whether the backend takes the edit for it (left out: not known yet; ``status`` adds the ``edit_at`` cap);
    ``gpu_lock_enforced()`` whether its model loads check the GPU lock (oMLX's, once patch_engine_pool
    ran: a GPU job refuses to unload without it); ``patches()`` which of the engine's patches took and which did not
    (oMLX: ``{applied, missing}``; doctor fails on a missing one); ``more_caps()`` caps that hold once something
    took (oMLX's ``edit_at``, patches.edit_caps); ``lookup(tracer, ids, start, on, job)`` its lookup.Measure, or None;
    ``credential(request)`` its key check in front of every route (oMLX's API key, backend.check_key; raises
    HTTPException), or None; ``can()`` all of them."""
    engine: str
    caps: frozenset = frozenset()
    chat_formats: Callable[[], dict] = dict
    gpu_lock_enforced: Callable[[], bool] = lambda: False
    patches: Callable[[], dict] = dict
    more_caps: Callable[[], frozenset] = frozenset
    lookup: Callable | None = None
    credential: Callable | None = None

    def can(self) -> frozenset:
        from .lookup import caps
        return self.caps | frozenset(self.more_caps()) | (caps(MANAGER.loaded()) if self.lookup else frozenset())


def _backend(request: Request) -> Backend:
    return request.app.state.metalscope_backend


def _require(request: Request, cap: str, why: str) -> None:
    """409 with ``why`` when the backend lacks ``cap``."""
    if cap not in _backend(request).can():
        raise HTTPException(status_code=409, detail=f"{why}; not available on this server")


def mount(app, backend: Backend) -> None:
    """Include the routes under /scope for ``backend``, ahead of any catch-all the server
    registers (FastAPI matches routes in order)."""
    app.state.metalscope_backend = backend
    n = len(app.router.routes)
    app.include_router(router, prefix=PREFIX)   # appends
    ours = app.router.routes[n:]
    del app.router.routes[n:]
    app.router.routes[0:0] = ours


def _loaded() -> list[str]:
    return [t.name for t in MANAGER.loaded()]


def _checked(name: str, what: str) -> str:
    """``name`` when it is one file name (models.NAME_RE), else 400: a request's names become paths under the data
    home and words on a job's command line (``..``, ``--fresh``)."""
    if not models.NAME_RE.fullmatch(name):
        raise HTTPException(status_code=400, detail=f"bad {what} name {name!r}: letters, digits, _ . - only, not starting with a dot")
    return name


def _pick_model(model: str | None, prefer=None) -> str | None:
    """The model a route acts on, by its short name: ``model`` when the request names one (400 unless it is one file
    name, ``_checked``); else the first loaded model that ``prefer`` accepts (e.g. one with concept profiles); else
    the first loaded model. None when nothing qualifies."""
    if model:
        return _checked(config.short_name(model), "model")
    loaded = _loaded()
    model = (next((n for n in loaded if prefer(n)), None) if prefer else None) or next(iter(loaded), None)
    return config.short_name(model) if model else None


def _find(trace_id: str) -> dict:
    for e in read_index():
        if e["id"] == trace_id:
            return e
    raise HTTPException(status_code=404, detail="unknown trace")


# -- status, mode and the live stream


# The package as this process imported it: oMLX through its hook's link (unresolved, so a link repointed by `metalscope
# omlx install` is followed), the ds4 proxy from where it runs.
IMPORTED_FROM = Path(__file__).parent


def code_stamp(root: Path = IMPORTED_FROM) -> int:
    """The package's Python source on disk now: a hash of each *.py file's path, size and mtime (a missing package:
    an empty one)."""
    files = []
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("node_modules", "__pycache__")]
        for n in names:
            if n.endswith(".py"):
                try:
                    st = os.stat(os.path.join(d, n))
                except OSError:
                    continue
                files.append((os.path.relpath(os.path.join(d, n), root), st.st_size, st.st_mtime_ns))
    return hash(tuple(sorted(files)))


_BOOT = code_stamp()
# What loads the installed code into a running server, by its engine (status ``stale``).
RESTART = {"omlx": "metalscope omlx restart", "ds4": "metalscope ds4 stop, then metalscope ds4"}


@router.get("/api/status")
def status(request: Request) -> dict:
    """Mode, loaded models, active traces and what tracing costs a step; the backend: its engine, what it can do
    (``caps``, see Backend) and each served model's thinking controls and flags (``chat``; ``edit``: the server takes
    the dock's one-token edit for it: False without ``edit_at``, left out while the backend does not know); ``stale``:
    the restart that loads the installed code, when it changed since the server started (code_stamp); else None."""
    backend = _backend(request)
    try:
        chat = backend.chat_formats()
    except Exception:                    # the backend's internals moved: no thinking controls rather than no status
        chat = {}
    can = backend.can()
    return dict(
        engine=backend.engine,
        caps=sorted(can),
        chat={m: c if "edit_at" in can else dict(c, edit=False) for m, c in chat.items()},
        mode=MANAGER.mode(),
        models=[dict(name=t.name, installed=t.installed, adapter=t.adapter_name,
                     **({"install_error": t.install_error} if getattr(t, "install_error", None) else {}))
                for t in MANAGER.loaded()],
        active=len(MANAGER.requests),
        traced_steps=MANAGER.traced_steps,
        bookkeeping_ms_per_step=round(1000 * MANAGER.overhead_s / MANAGER.traced_steps, 3) if MANAGER.traced_steps else None,
        bookkeeping_ms_by_part={k: round(1000 * v / MANAGER.traced_steps, 3) for k, v in MANAGER.phase_s.items()}
        if MANAGER.traced_steps else {},
        traces_dir=str(config.traces_dir()),
        gpu_lock_enforced=bool(backend.gpu_lock_enforced()),
        patches=backend.patches(),
        stale=RESTART.get(backend.engine, "restart it") if code_stamp() != _BOOT else None,
    )


@router.post("/api/mode")
def set_mode(body: dict) -> dict:
    """``{"mode": "off"|"on"}``: switch tracing."""
    mode = body.get("mode")
    if mode not in config.MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {config.MODES}")
    return MANAGER.set_mode(mode)


SHUTTING_DOWN = threading.Event()


@router.get("/api/live")
async def live(request: Request, client: str | None = None, experts: bool = False, lens: bool = False,
               writes: bool = False):
    """Server-sent events, one per token (livestream.py), of every client's traced replies or ``client``'s;
    ``experts`` adds each token's MoE routing, ``lens`` each read layer's top 10, ``writes`` what each layer wrote into
    each residual stream."""
    async def events():
        # uvicorn's graceful shutdown waits for open connections, so a stream
        # must end by itself once shutdown starts (SHUTTING_DOWN, set from
        # uvicorn's signal handler by middleware.patch_uvicorn_shutdown).
        # Subscribed here, not in the route: a response closed before its first step never runs this body, so only
        # here does the finally below always follow the subscription.
        queue = MANAGER.hub.subscribe(asyncio.get_running_loop(), client, experts, lens, writes)
        idle = 0
        try:
            yield ": connected\n\n"
            while not SHUTTING_DOWN.is_set() and not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1)
                except asyncio.TimeoutError:
                    idle += 1
                    if idle >= 15:
                        idle = 0
                        yield ": ping\n\n"
                    continue
                idle = 0
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
        finally:
            MANAGER.hub.unsubscribe(queue)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# -- a saved reply: its routing and its n-gram lookup's help; the experts


@router.get("/api/traces/{trace_id}/routing")
def reply_routing(trace_id: str, last: int | None = None) -> dict:
    """One saved reply's routing per reply token as the live stream sends it (layout.reply_routing), the last ``last``
    tokens when given: Pi's dock's experts for a reply traced before the dock asked the stream for routing. Reads the
    router arrays alone."""
    from . import layout
    entry = _find(trace_id)
    if last is not None and last < 1:
        raise HTTPException(status_code=400, detail="last must be 1 or more")
    try:
        return layout.reply_routing(entry["path"], last)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:   # missing or corrupt: not a 500
        raise HTTPException(status_code=404, detail=f"no routing saved for this reply: {exc}")


@router.post("/api/lookup")
async def reply_lookup(body: dict, request: Request) -> dict:
    """``{trace, client, ask}``: how much the model's n-gram lookup helped each fed token of that saved reply (lookup.py;
    the last sampled token is not fed, so it has none): ``ids``, ``lp_on``, ``lp_off`` and each pass's top choices
    (``top_ids_on`` …, ``vocab``), measured on the served model at the first ask, then kept beside the trace; where the
    passes are goes to the live feed's ``client`` (lookup.Run). The same ask again while it runs, or with ``again``
    (Pi's, past its idle timeout), waits for that run and answers as it did, never starting one. ``{trace, ask, stop:
    true}`` calls that ask off, waiting or under way (it is answered 409, nothing kept). 409 when the server cannot (no
    ``lookup``), the reply is another model's or not whole, or a reply, GPU job or lookup is under way (one called off
    is waited for). The GPU lock is held for both passes."""
    from . import gpulock, lookup
    trace, ask = str(body.get("trace") or ""), body.get("ask") or None
    if ask is not None and not isinstance(ask, str):
        raise HTTPException(status_code=400, detail="ask: text")
    if body.get("stop"):
        return dict(stopping=lookup.stop(trace, ask))
    entry = _find(trace)
    key = (trace, ask)
    was = lookup.ASKS.get(key) if ask else None
    if was is not None or body.get("again"):
        while was is not None and not was.ended.is_set() and not await request.is_disconnected():
            await asyncio.sleep(0.1)
        got = await asyncio.to_thread(lookup.cached, entry)
        if got is not None:
            return got
        failed = was.failed if was is not None else None
        raise HTTPException(status_code=failed.status_code if failed else 409,
                            detail=failed.detail if failed else "the engram inspection is not under way: ask again")
    job, held = lookup.Run(entry["id"], body.get("client"), ask), False
    if ask:
        lookup.ASKS[key] = job
    try:
        lock = gpulock.GpuLock(gpulock.Holder(lookup.HOLDER))
        try:
            got = await asyncio.to_thread(lookup.cached, entry)
            if got is not None:
                return got
            _require(request, "lookup", "engram inspection needs a served model with an engram")
            tracer = _tracer_named(entry.get("model"))
            if tracer is None:
                raise HTTPException(status_code=409, detail=f"that reply is {entry.get('model')}'s, not served here")
            if not getattr(tracer, "lookup", False):
                raise HTTPException(status_code=409, detail=f"{tracer.name} has no engram")
            if not (held := await lookup.take(job)):
                raise HTTPException(status_code=409,
                                    detail="an engram inspection is under way: ask again when it is done")
            job.check()
            if MANAGER.requests:
                raise HTTPException(status_code=409, detail="a reply is being served: ask again when it is done")
            try:
                await lookup.whole(lock.acquire, 1.0)            # cancelled meanwhile, still taken, then let go below
            except gpulock.LockBusy as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            return await lookup.measure(entry, tracer, _backend(request).lookup, job)
        except lookup.Refused as exc:
            job.failed = HTTPException(status_code=exc.status, detail=str(exc))
        except lookup.Stopped as exc:
            job.failed = HTTPException(status_code=409, detail=str(exc))
        except (RuntimeError, ValueError, OSError) as exc:
            job.failed = HTTPException(status_code=503, detail=f"the engram could not be inspected: {exc}")
        except HTTPException as exc:
            job.failed = exc
        finally:
            if held:
                try:
                    lock.release()
                finally:
                    lookup.RUN = None
                    lookup.ONE.release()
        raise job.failed
    finally:
        if ask and lookup.ASKS.get(key) is job:
            del lookup.ASKS[key]
        job.ended.set()


@router.get("/api/experts/grid")
def expert_grid(model: str | None = None, layer: int = 0) -> dict:
    """Pi's dock's EXPERTS grid for one MoE layer of ``model`` (default: the first loaded): each expert's cell, map
    neighbours near each other (layout.expert_grid). Without an expert map (or one
    that does not read) ``pos`` is null and the dock draws the experts in number order (metalscope experts layout
    builds one); a loaded model whose hooks found no MoE layers (its routing rule None) is said to have none."""
    from . import layout
    name = _pick_model(model)
    if not name:
        raise HTTPException(status_code=404, detail="no model loaded or named")
    try:
        g = layout.expert_grid(name, layer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    t = _tracer_named(name)
    if g["pos"] is None and t is not None and t.installed and getattr(t.capture or t, "routing", None) is None:
        g["note"] = f"{g['model']} has no experts"
    return g


def _tally_layer(model: str | None, layer: int, client: str | None):
    """The tally's layer for the expert routes (tally.layer_view: ``all``, or ``client``'s session) and a token decoder
    from the model's own tokenizer (a loaded copy of its checkpoint; ``<id>`` without one)."""
    from . import tally
    name = _pick_model(model)
    if not name:
        raise HTTPException(status_code=404, detail="no model loaded or named")
    try:
        view = tally.layer_view(name, layer, client or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    tracer = _tracer_named(name) or next((t for t in MANAGER.loaded()
                                          if models.checkpoint_of(t.name) == models.checkpoint_of(name)), None)
    tok = getattr(tracer, "tokenizer", None)

    def text(i: int) -> str:
        try:
            return tok.decode([int(i)]) if tok is not None else f"<{i}>"
        except Exception:
            return f"<{i}>"
    head = dict(model=models.checkpoint_of(name), layer=layer, source="session" if client else "all",
                tokens=view.N if view else 0, against=view.against if view else None)
    return view, text, head


@router.get("/api/experts/specialists")
def expert_specialists(layer: int, model: str | None = None, client: str | None = None) -> dict:
    """The dock's experts' tokens at one MoE layer, from the stored tally (``all``) or ``client``'s session: every
    expert over the evidence floor (tally.py; one under it is left out), by number, each with its top 3 tokens by
    lift. Empty without a tally or a session. About 18 KB for 400 experts."""
    view, text, out = _tally_layer(model, layer, client)
    out["experts"] = [dict(e=x, top=[text(t) for t, _, _ in view.tops(x, 3)]) for x in sorted(view.order)] if view else []
    return out


@router.get("/api/experts/expert")
def expert_one(layer: int, e: int, model: str | None = None, client: str | None = None) -> dict:
    """One expert at one MoE layer, from the tally (``all``) or ``client``'s session: its tokens counted (``n``) and
    their share of the tokens, its rank of ``ranked`` (null under the evidence floor), its top tokens [text, lift, count] and its
    same-layer overlaps by co-fire lift (each with its top tokens)."""
    view, text, out = _tally_layer(model, layer, client)
    out["ranked"] = len(view.order) if view else 0
    if not view or not 0 <= e < view.E:
        return dict(out, e=e, n=0, share=0.0, rank=None, top=[], overlaps=[])
    return dict(out, e=e, n=int(view.ne[e]), share=float(view.ne[e] / max(view.N, 1)),
                rank=view.order.index(e) + 1 if e in view.order else None,
                top=[[text(t), round(lift, 2), c] for t, lift, c in view.tops(e, 10)],
                overlaps=[dict(e=b, lift=round(lift, 2), co=co, top=[text(t) for t, _, _ in view.tops(b, 3)])
                          for b, lift, co in view.overlaps(e)[:8]])


# -- concepts and steering: what Pi's /steer and /scope need


def _tracer_named(model: str | None):
    short = config.short_name(model)
    for t in MANAGER.loaded():
        if config.short_name(t.name) == short or t.name == model:
            return t
    return None


@router.get("/api/concepts")
def concepts(model: str | None = None) -> dict:
    """Every profiled concept of ``model`` (default: the loaded model that
    has profiles) with what Pi reads of it (concepts.summarize) and its chat calibration
    (``calibration``: this served model's, None without one; its default dose is the ``steer`` dose) and who wrote its
    texts (``texts``, one line); also whether steering is
    enabled on the server and ``norms_note``, a warning when the dose units came from few or short traces
    (steer.norms_note); ``library``: the concepts not learned on it yet (``_library``), what /concept offers to
    learn.
    The profiles are the checkpoint's, the norms and calibrations the served model's (models.py)."""
    from . import concepts as C
    from .concepts.corpus import texts_line
    loaded = _loaded()
    model = _pick_model(model, prefer=models.profile_paths)
    tracer = _tracer_named(model)
    table = _norm_table(tracer)
    items = []
    for p in models.profile_paths(model or ""):
        try:
            # This served model's chat calibration, if one was made (learn/calibrate.py): its default dose becomes /steer's.
            cal = C.calibration_of(model or "", p.parent.name)
            prof = json.loads(p.read_text())
            item = C.summarize(prof, cal)
            item["calibration"] = cal
            where = C.texts_dir(model or "", p.parent.name)    # who wrote its texts, for /concept NAME (not the request)
            item["texts"] = texts_line(where, request=False) if where is not None else None
        except Exception as exc:        # its files malformed (a pulled pack is not checked field by field): said on it
            item = dict(name=p.parent.name, error=str(exc) or type(exc).__name__)
        items.append(item)
    from .steer import norms_note
    return dict(model=model or "", checkpoint=models.checkpoint_of(model or ""), loaded=[config.short_name(n) for n in loaded], steer_enabled=MANAGER.steer_enabled(),
                norms_note=norms_note(table), concepts=items, library=_library(model or ""))


def _library(model: str) -> list[dict]:
    """The concepts not learned on ``model``'s checkpoint yet, as ``concept list`` shows them (``list_concepts``: the
    shipped ones, the defined ones and the model's own texts): name, title, the first sentence of its description, who
    wrote its texts (``writer``: concept.json's, None without texts or a record; ``texts``: the one line,
    corpus.texts_line). Only those learn would take (``learn.plan.name_taken``: no vector or profile of it on the
    checkpoint already, e.g. a vector measured before learn jobs)."""
    from .concepts.corpus import list_concepts
    from .learn.plan import name_taken
    learned = {p.parent.name for p in models.profile_paths(model)} if model else set()
    out = []
    for c in list_concepts(model or None):
        if c["name"] in learned or (model and name_taken(c["name"], model) is not None):   # learn would refuse it here
            continue
        desc = " ".join(str(c["description"]).split())
        cut = desc.find(". ")
        meta = config.read_json(Path(c["path"]) / "concept.json")
        wrote = meta.get("learned") if isinstance(meta, dict) else None
        out.append(dict(name=c["name"], title=c["title"], description=desc if cut < 0 else desc[:cut + 1],
                        writer=wrote.get("writer") if isinstance(wrote, dict) else None, texts=c["texts"]))
    return out


def _norm_table(tracer):
    """The tracer's served model's strength table, the one the steering path uses (MANAGER.strength_table: read again
    when its norms.json changes); None when it has no saved traces of its own to measure on."""
    if tracer is None:
        return None
    try:
        return MANAGER.strength_table(tracer.name)
    except ValueError:
        return None


@router.get("/api/concepts/{name}/replies")
def concept_replies(name: str, model: str | None = None) -> dict:
    """One user message answered plain and steered at the default dose, from the concept's last chat calibration
    (calibrate.sample_replies), for Pi's /concept NAME overview; ``sample`` is None without a fitting calibration or a
    turn where both answered."""
    from . import concepts as C
    from .learn import calibrate as K
    name = _checked(name, "concept")
    model = _pick_model(model, prefer=models.profile_paths)
    cal = C.calibration_of(model or "", name)
    # The served model's own calibration and its conversations (models.calibration_dir).
    return dict(sample=K.sample_replies(model, name, cal["default_dose"]) if cal else None)


def _served_for(model: str, served: str | None = None) -> str:
    """The served copy whose chat calibrations the steering routes answer for ``model``: ``served`` when it is one of the
    checkpoint's served names; ``model`` itself when it is one; else the first of them with calibrations, else the
    first."""
    names = models.served_names(model)
    for n in (served, model):
        if n and config.short_name(n) in names:
            return config.short_name(n)
    return next((n for n in names if models.served_copy(n)["calibrations"]), names[0])


def _chat(cal: dict | None, served: str, name: str) -> dict | None:
    """A chat calibration as Pi's Steer page draws it (learn/calibrate.py's calibration.json, the parts it shows), each
    strength and plain with its mean style over the turns graded at it (``style``, from the last grading, the one its
    good replies and mean coherence come from; None without grades). The file is only read."""
    from .learn import calibrate as K
    if cal is None:
        return None
    grades = K.grades_of(served, name)
    styled = lambda r: dict(r, style=K.mean_grade(grades, r["dose"], "style")) if isinstance(r, dict) and "dose" in r else r
    keys = ("default_dose", "good", "turns", "layer")
    return dict({k: cal.get(k) for k in keys}, plain=styled(cal.get("plain")),
                strengths=[styled(r) for r in cal.get("strengths") or []])


@router.get("/api/steering")
def steering(model: str | None = None, served: str | None = None) -> dict:
    """Pi's Steer page: every learned concept of ``model``'s checkpoint (default: the loaded model with profiles) as
    its chat calibration on the served copy ``served`` (``_served_for``; ``chat``, None when it has none that fits
    its profile, ``chat_stale`` when it has one made before the profile changed; each strength with its mean style,
    ``_chat``), with its title and description; calibrated concepts first, by good replies at their dose, then the
    lower dose, then name."""
    from . import concepts as C
    model = _pick_model(model, prefer=models.profile_paths) or ""
    copy = _served_for(model, served)
    ranked = []
    for p in models.profile_paths(model):
        name = p.parent.name
        try:
            prof = json.loads(p.read_text())
            cal = C.calibration_of(copy, name)
            chat = _chat(cal, copy, name)
            item = dict(name=name, title=prof.get("title") or name, description=prof.get("description", ""), chat=chat,
                        chat_stale=cal is None and models.is_calibration(models.calibration_path(copy, name)))
            rank = (chat is None, -float((chat or {}).get("good") or 0), float((chat or {}).get("default_dose") or 0))
        except Exception as exc:        # its files malformed (a pulled pack is not checked field by field): said on it
            item, rank = dict(name=name, error=str(exc) or type(exc).__name__, chat=None, chat_stale=False), (True, 0.0, 0.0)
        ranked.append((rank, name, item))
    ranked.sort(key=lambda r: r[:2])
    return dict(model=model, checkpoint=models.checkpoint_of(model), served=copy, concepts=[r[2] for r in ranked])


@router.get("/api/steering/{name}/chats")
def steering_chats(name: str, model: str | None = None, dose: float | None = None) -> dict:
    """One concept's chat calibration conversations on the served copy ``model`` (``_served_for``), plain beside
    ``dose`` (default: its default dose), with the last grading's grades (calibrate.conversations), for Pi's Steer
    page. 404 without a calibration that fits the profile, or at a dose it did not try."""
    from . import concepts as C
    from .learn import calibrate as K
    name = _checked(name, "concept")
    model = _pick_model(model, prefer=models.profile_paths) or ""
    copy = _served_for(model, model)
    cal = C.calibration_of(copy, name)
    if cal is None:
        raise HTTPException(status_code=404, detail=f"no chat calibration of {name!r} on {copy!r}")
    dose = cal["default_dose"] if dose is None else dose
    if not any(abs(s["dose"] - dose) < 1e-9 for s in cal.get("strengths") or []):
        raise HTTPException(status_code=404, detail=f"{name!r} was not tried at dose {dose:g} on {copy!r}")
    return dict(name=name, served=copy, dose=float(dose), default_dose=cal["default_dose"],
                conversations=K.conversations(copy, name, dose))


@router.post("/api/steer")
def set_steer(body: dict) -> dict:
    """Enable or disable steering on the server (the X-Scope-Steer header is
    refused while it is off), as `metalscope steering on|off` does."""
    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled must be true or false")
    config.save_settings(steer=enabled)
    return dict(steer_enabled=MANAGER.steer_enabled())


@router.post("/api/steer/words")
def steer_from_words(body: dict) -> dict:
    """A word vector for the dock's one-token edit (the only steer a word takes:
    steer.WORD_EDIT_ONLY): the Jacobian-lens direction of the words' tokens
    (vectors.from_lens) on the loaded model, saved under ``name`` so the edit's
    header can use it; ``ids`` (one a word): the tokens themselves, as the dock
    shows them. The response gives the edit band (a removal's layers, a swap
    pushing at its middle: ``_edit_band``) and the words cut to their first token. A plain
    ``def``: FastAPI runs it in the threadpool, off the event loop that serves
    the live stream. The lens is the tracer's, never one loaded for the call
    (gigabytes): without one yet, the tracer installs as a steered request
    would, loading it once under its lock however many calls wait; the output
    head read through the tracer (``unembed``: the MLX model's, the GGUF's on
    ds4), the direction cut to the columns a steer writes where the model has
    them (``lens_columns``: Inkling 2's running sum). Names that belong to a
    concept are refused, so a word vector cannot replace a profiled direction."""
    from . import concepts as C
    name, words, ids = body.get("name"), body.get("words"), body.get("ids")
    if not name or not isinstance(words, list) or not words:
        raise HTTPException(status_code=400, detail="name and a non-empty words list are required")
    if ids is not None and not (isinstance(ids, list) and len(ids) == len(words) and all(type(i) is int and i >= 0 for i in ids)):
        raise HTTPException(status_code=400, detail="ids: one token id a word")
    if not vectors.NAME_RE.fullmatch(str(name)):
        raise HTTPException(status_code=400, detail="name: letters, digits, _ . - only")
    tracer = _tracer_named(body.get("model"))
    if tracer is None:
        raise HTTPException(status_code=404, detail=f"model {body.get('model')!r} is not loaded")
    if C.shipped_dir(str(name)) or C.definition(str(name)) or C.texts_dir(tracer.name, str(name)):
        raise HTTPException(status_code=409, detail=f"{name!r} is a concept: learn it (/concept learn {name} ...) and steer it by name")
    if models.profile_path(tracer.name, str(name)).exists():
        raise HTTPException(status_code=409, detail=f"{name!r} has a stored profile: its vector is the profiled direction")
    if tracer.lens is None:
        try:
            tracer.install(MANAGER)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
    setup = tracer.lens                                             # once: /jlens use may swap it meanwhile
    if setup is None:
        raise HTTPException(status_code=400, detail=f"no Jacobian lens for {config.short_name(tracer.name)}")
    try:
        vec = vectors.from_lens(tracer.unembed, tracer.tokenizer, tracer.name, str(name), [str(w) for w in words],
                                lens=setup.lens, ids=ids, columns=getattr(tracer, "lens_columns", None))
    except (ValueError, OverflowError) as exc:                      # OverflowError: an id no tokenizer has
        raise HTTPException(status_code=400, detail=str(exc))
    vectors.save_vector(tracer.name, vec)
    layers = vec.layers
    return dict(name=vec.name, model=config.short_name(tracer.name), layers=layers, words=vec.meta.get("words"),
                cut=vec.meta["cut"], band=_edit_band(tracer))


def _edit_band(tracer) -> tuple[int, int]:
    """The layers a word removal (the dock's edit) projects over on ``tracer``'s model (a swap pushes at
    its middle: Pi's pushLayer): the words route's ``band``, the lens steps' ``edit_band``. The workspace band, or the
    adapter's (ModelTracer.edit_band; Inkling 2: the last block's visits, 42-46 of 48)."""
    from . import jlens
    own = getattr(tracer, "edit_band", None)                        # Ds4Tracer has none: the workspace band
    return own() if own else jlens.workspace_band(tracer.n_layers)


# -- the GPU jobs (learn/) and the J-lens


def _job_model(model: str | None, what: str, prefer=None) -> str:
    """The model a job acts on (``_pick_model``); 409 saying it cannot ``what`` when none is loaded."""
    model = _pick_model(model, prefer)
    if not model:
        raise HTTPException(status_code=409, detail=f"no model is loaded to {what}")
    return model


async def _off_loop(fn: Callable, *args, status: int = 409, errors=(ValueError,)):
    """``fn(*args)`` in a thread, off the event loop that serves the live stream; ``errors`` answer ``status`` with
    their message (a job that cannot start says why)."""
    try:
        return await asyncio.to_thread(fn, *args)
    except errors as exc:
        raise HTTPException(status_code=status, detail=str(exc))


def _served_here(request: Request, model: str) -> str:
    """``model``, when this server serves it (its ``chat_formats``: oMLX's every model, loaded or not; the ds4
    proxy's own); else 409 saying how to start it. A server that cannot list its models lets it through."""
    from . import client
    try:
        here = {config.short_name(m) for m in _backend(request).chat_formats()}
    except Exception:                    # the backend's internals moved: no list to check against
        here = set()
    if here and config.short_name(model) not in here:
        raise HTTPException(status_code=409, detail=client.not_served(model))
    return model


def _self_url(request: Request) -> str:
    """This server's URL, from the socket the request came in on: a job started here asks it. Never from the Host
    header, which a local client could set to a port of its own listener. 409 when it has no TCP address."""
    host, port = request.scope.get("server") or (None, None)
    if not host or port is None:
        raise HTTPException(status_code=409, detail="this server listens on a Unix socket, which a job cannot ask: "
                                                    "start the job with the metalscope command and its --url")
    return f"http://{f'[{host}]' if ':' in host else host}:{port}"


def _needs_load_copy(request: Request, what: str) -> None:
    """A job that loads its own MLX copy of the model and unloads the served one (the J-lens's learn) needs a backend that can unload it (``load_copy``)."""
    _require(request, "load_copy", f"{what} unloads the served model and loads its own copy in a GPU job")


def _job_calibration(j: dict) -> dict | None:
    """A job's chat calibration in brief (its default dose, its good replies of how many), version 4; None without
    one."""
    cal = (j.get("result") or {}).get("calibration")
    if not isinstance(cal, dict) or cal.get("error") or cal.get("version") != models.CALIBRATION_VERSION:
        return None
    return {k: cal.get(k) for k in ("default_dose", "good", "turns")}


def _job_fit(j: dict) -> dict | None:
    """A lens fit in brief: windows fitted and the last held-out hit@10, J-lens and logit lens; None for other jobs."""
    if j.get("kind") != "lens-learn":
        return None
    r, w = j.get("result") or {}, j.get("fit") or {}
    h = r.get("heldout") or {}
    return dict(windows=r.get("windows", w.get("n")), hit10=h.get("hit10", w.get("hit10")),
                logit_hit10=h.get("logit_hit10", w.get("logit_hit10")), lens=r.get("lens", w.get("lens")))


@router.get("/api/jobs")
def jobs() -> dict:
    """Every GPU job Pi or the command line started (``learn/state.py``; ``kind``
    tells them apart: learn, calibrate, lens-learn), for Pi's /scope jobs and its widget:
    ``latest`` (the newest job's full state: kind, phase, progress, eta, log
    tail, and when done its result), ``jobs`` (the last ten, briefly; a learn
    job's ``steerable`` (False: it stopped early), a calibration's (a learn job's too) ``calibration``, a lens fit's ``fit``:
    its windows and held-out hit@10; None for the other kinds) and ``gpu``
    (who holds the GPU lock now, with its message; None when free)."""
    from dataclasses import asdict
    from . import gpulock
    from .learn import state
    jobs = [j for j in state.list_jobs() if j.get("job") and j.get("kind")]     # a damaged state: that job left out
    holder = gpulock.held()
    brief = [dict(job=j["job"], kind=j["kind"], name=j.get("name"), phase=j.get("phase"), outcome=j.get("outcome"),
                  started=j.get("started"), steerable=(j.get("result") or {}).get("steerable"),
                  calibration=_job_calibration(j), fit=_job_fit(j)) for j in jobs[-10:]]
    return dict(latest=jobs[-1] if jobs else None, jobs=brief,
                gpu=dict(asdict(holder), message=holder.message()) if holder else None)


@router.post("/api/jobs/{job}/seen")
async def job_seen(job: str) -> dict:
    """Its result was shown."""
    from .learn import state
    return dict(seen=await _off_loop(state.mark_seen, job, status=400))


@router.post("/api/jobs/{job}/cancel")
async def job_cancel(job: str) -> dict:
    """Stop a running job; a job holding the GPU stops its run, frees the lock and reloads the model."""
    from .learn import state
    return dict(cancelled=await _off_loop(state.cancel, job, status=400))


@router.get("/api/lens")
def lens_status(model: str | None = None) -> dict:
    """Pi's /jlens: the model's Jacobian lens from its files (``jlens.status``): installed or not and where, its
    source, the other lenses ranked. ``model``: default the loaded model. 404 with no model loaded or named."""
    from . import jlens
    model = _pick_model(model)
    if not model:
        raise HTTPException(status_code=404, detail="no model loaded or named")
    return jlens.status(model)


LENS_SPEED_REPLIES = 5     # a lens step's speed: the mean of this model's last replies read at it


def _lens_tracer(model: str | None):
    """The tracer the lens steps are for: 409 without it, as /api/steer/words answers without a model loaded (Pi reads
    a 404 here as a server without this route), and 409 with the install error on a model the hooks cannot read."""
    tracer = _tracer_named(_pick_model(model))
    if tracer is None:
        raise HTTPException(status_code=409, detail=f"model {model!r} is not loaded" if model else "no model is loaded")
    try:
        tracer.lens_steps()                 # a model the hooks cannot read: why, as /api/steer/words answers
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return tracer


def _lens_layers(model: str | None) -> dict:
    """LENS's - + in the dock: the step the next traced reply reads the lenses at (``at``; -1 none, the lenses off)
    and, a step each, fewest first: its name (``name``: small, medium, large), how many layers it reads (``n``, the
    J-lens's where there is one), every layer either lens reads (``layers``), and the decode speed of this model's last
    unsteered replies read at it on this server (``tps``, tok/s, over ``replies``; None without one): at its name and as
    many layers (a reply saved without the count, or read at the sizes before, is not this size's). With a J-lens (Inkling 2's too), ``edit_band``: where the
    dock's word edit acts (``_edit_band``), shown before it is sent."""
    tracer = _lens_tracer(model)
    short = config.short_name(tracer.name)
    seen = [e for e in read_index() if e.get("model") == short and not e.get("steer") and e.get("decode_tps")]   # newest first
    out = []
    for i, (jl, logit) in enumerate(tracer.lens_steps()):
        n = len(jl or logit)
        last = [e["decode_tps"] for e in seen if e.get("lens_step") == config.lens_name(i) and e.get("lens_layers") == n][:LENS_SPEED_REPLIES]
        out.append(dict(name=config.lens_name(i), n=n, layers=sorted({*jl, *logit}),
                        tps=round(sum(last) / len(last), 1) if last else None, replies=len(last)))
    return dict(model=short, at=MANAGER.lens_step(), steps=out,
                **({"edit_band": list(_edit_band(tracer))} if tracer.lens is not None else {}))


@router.get("/api/lens/layers")
def lens_layers(model: str | None = None) -> dict:
    return _lens_layers(model)


@router.post("/api/lens/layers")
def set_lens_layers(body: dict) -> dict:
    """Read the lenses at step ``{step}`` (0 the fewest layers; -1 none, the lenses off) from the next traced reply on:
    settings' lens_step, for every model; LENS's - + in the dock. What GET answers then; 409 without the model loaded,
    nothing saved."""
    _lens_tracer(body.get("model"))
    if not config.lens_step_ok(step := body.get("step")):
        raise HTTPException(status_code=400, detail=f"step must be 0 to {config.LENS_STEPS - 1}, or {config.LENS_OFF} (off)")
    try:
        config.save_settings(lens_step=config.lens_name(step))
    except ValueError as exc:                     # another setting invalid (edited by hand): said, not a 500
        raise HTTPException(status_code=400, detail=str(exc))
    return _lens_layers(body.get("model"))


@router.get("/api/lens/learn")
async def lens_learn_plan(request: Request, model: str | None = None, until: str | None = None,
                          windows: int | None = None, path: str | None = None) -> dict:
    """What ``/jlens learn`` would do, for Pi's confirm text (``lens_learn_plan``): the deadline, the memory it waits
    for, the estimated minutes a window and windows by the deadline (from the served checkpoint's config), the corpus,
    whether an earlier fit resumes, the lens directory. 400 for a bad time, count or path; 409 with no model."""
    from .learn import command, spawn
    name = _served_here(request, _job_model(model, "learn a lens for"))
    _needs_load_copy(request, "/jlens learn")
    path_ = await asyncio.to_thread(command.model_path_for, name)
    return await _off_loop(spawn.lens_learn_plan, name, path_, until, windows, path, status=400)


@router.post("/api/lens/learn")
async def lens_learn(body: dict, request: Request) -> dict:
    """Start a lens-learn job for ``{model?, path?, until?, windows?, fresh?}`` (model: default the loaded model): it
    unloads the model until the deadline, so Pi asks first. 409 with the reason when it cannot start."""
    from .learn import spawn
    model = _served_here(request, _job_model(body.get("model"), "learn a lens for"))
    windows = body.get("windows")
    _needs_load_copy(request, "/jlens learn")
    # A bad window count is a 409 too: it is read in the thread.
    return await _off_loop(lambda: spawn.lens_learn(model, body.get("path") or None, body.get("until") or None,
                                                    int(windows) if windows else None, bool(body.get("fresh"))),
                           errors=(ValueError, TypeError))


@router.post("/api/lens/use")
async def lens_use(body: dict) -> dict:
    """Make lens ``{name}`` the J-lens of the model's checkpoint (``jlens.use``: checkpoint.json records it, for every
    served name of the checkpoint; ``logit``: the logit lens); what changed. A loaded model with its hooks installed reads the new lens from its next forward
    (``ModelTracer.refresh_lens``); ``live`` is the lens it now reads, which differs from ``lens`` only when that one
    failed to load (the server log says why); None when the hooks are not installed (they read it when tracing
    starts). 409 for an unknown lens or no model."""
    from . import jlens
    model = _job_model(body.get("model"), "give a lens")
    out = await _off_loop(jlens.use, str(body.get("name") or ""), model)
    out["live"] = await _relens(model)
    return out


async def _relens(model: str) -> str | None:
    """A loaded model with its hooks installed reads its lens again (``refresh_lens``: the checkpoint's choice, now),
    from its next forward: the lens it reads (``LOGIT`` for none); None when it is not loaded or has no hooks yet
    (they read it when tracing starts)."""
    from . import jlens
    tracer = _tracer_named(model)
    if tracer is None or not tracer.installed:
        return None
    fresh = await asyncio.to_thread(tracer.refresh_lens)
    return fresh.name if fresh is not None else jlens.LOGIT


@router.get("/api/model/link")
def model_link_offer(request: Request, model: str | None = None) -> dict:
    """The dock's link offer for ``model`` (default the loaded model; one this server serves): models.link_offer,
    the same-shape models with data it could be a copy of. 409 with no model."""
    return models.link_offer(_served_here(request, _job_model(model, "link")))


@router.post("/api/model/link")
async def model_link(body: dict, request: Request) -> dict:
    """``model link`` for ``{model?, checkpoint}`` from the dock: only one of the offer's candidates (the command
    line links anything else). Once linked it answers 200 with what the model's checkpoint now holds and the J-lens
    it reads (``lens``, None for the logit lens): a loaded model reads it again now (``_relens``), its vectors and
    concepts at the next request. When that re-read raises, or reads no lens though the checkpoint has one, ``lens``
    is None and ``note`` says the lens loads with the next model load and why (``note`` None otherwise). 409 with
    link's reason when it refuses (all before anything moves) or for a model that is not a candidate, 400 for a bad
    name, 500 for anything else: "nothing was linked" before the link starts, else it may be part way and running
    it again finishes it (models.link)."""
    from . import jlens
    model = _served_here(request, _job_model(body.get("model"), "link"))
    ckpt = _checked(str(body.get("checkpoint") or ""), "checkpoint")
    started = []

    def run() -> None:
        models.check_link(model, ckpt)
        if ckpt not in {c["name"] for c in models.link_offer(model)["candidates"]}:
            raise ValueError(f"{ckpt} is not a model of the same shape with data")
        started.append(True)
        models.link(model, ckpt)
    try:
        await asyncio.to_thread(run)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        why = f"{type(exc).__name__}: {exc}"
        raise HTTPException(status_code=500, detail=f"the link may have stopped part way: {why}; run it again to finish it"
                            if started else f"nothing was linked: {why}")
    out = dict(model=model, checkpoint=ckpt, concepts=models.concept_count(model), lenses=len(jlens.lens_dirs(model)),
               lens=None, note=None)
    later = "J-lens loads at the next model load"     # under Pi's "linked to ...", on two lines of its narrowest column
    try:
        live, lens = await _relens(model), jlens.find_for(model)
        if live == jlens.LOGIT and lens is not None:
            out["note"] = later + " (the server log says why)"
        else:
            out["lens"] = lens.name if lens is not None else None
    except Exception as exc:
        out["note"] = later + f": {type(exc).__name__}: {exc}"
    return out


@router.get("/api/concept/learn")
def concept_learn_plan(request: Request, name: str, model: str | None = None, corpus: str | None = None,
                       description: str | None = None) -> dict:
    """What ``/concept learn NAME [--corpus DIR]`` would do (learn.plan.learn_plan, the start's own checks), for Pi's
    confirm text: where NAME's texts come from (``source``) and who wrote existing ones, or why
    it cannot be learned here (with ``corpus``, also why DIR's texts cannot be used; with ``description``, also a
    blank one), and its notes. The texts themselves stay out."""
    from .learn.plan import learn_plan
    model = _served_here(request, _job_model(model, "learn with", prefer=models.profile_paths))
    plan = learn_plan(name, model, description, corpus or None)
    plan.pop("read")
    return plan


@router.post("/api/concept/learn")
async def concept_learn(body: dict, request: Request) -> dict:
    """Start ``concept learn`` for ``{name, description?, model?, corpus?}``
    (model: default the loaded model with concept profiles; description: not
    needed for a library concept, or with corpus,
    a directory of texts written elsewhere). On a ``load_copy`` server
    it unloads that model for the GPU part; on a ``measure_served`` one (ds4) it
    measures the served model and chat is refused meanwhile; so Pi asks before
    calling this. 409 with the reason when it cannot start."""
    from .learn import spawn
    model = _served_here(request, _job_model(body.get("model"), "learn with", prefer=models.profile_paths))
    caps = _backend(request).can()
    engine = "mlx" if "load_copy" in caps else "ds4" if "measure_served" in caps else None
    if engine is None:
        raise HTTPException(status_code=409, detail="/concept learn measures the model on its own copy or where it is "
                                                    "served; not available on this server")
    # On ds4 the job asks this proxy; on oMLX the server oMLX's settings name (the job's default).
    url = _self_url(request) if engine == "ds4" else None
    return await _off_loop(lambda: spawn.learn(str(body.get("name") or ""), str(body.get("description") or ""), model,
                                               None, engine, url, corpus=str(body.get("corpus") or "") or None))


@router.post("/api/concept/calibrate")
async def concept_calibrate(body: dict, request: Request) -> dict:
    """Start ``concept calibrate`` for ``{name, model?}`` (model: default
    the loaded model with concept profiles): chat requests only, nothing is
    unloaded. 409 with the reason when it cannot start."""
    from .learn import spawn
    model = _job_model(body.get("model"), "calibrate with", prefer=models.profile_paths)
    if not str(body.get("name") or "").strip():
        raise HTTPException(status_code=400, detail="name the concept to calibrate")
    _served_here(request, model)
    # The job asks this server (oMLX's or the ds4 proxy's), whichever started it.
    return await _off_loop(spawn.calibrate, str(body["name"]), model, _self_url(request))


@router.post("/api/concept/delete")
def concept_delete(body: dict, request: Request) -> dict:
    """``concept delete`` for ``{name, model?}`` (model: default the loaded model with concept profiles; one this
    server serves): the concept's folder for the model's checkpoint moved to the trash (models.trash), for every
    served copy of it; ``checkpoint`` and ``trash`` say which and where. When no model has it, its description goes
    instead (``definition``: true). 400 for a name that is not one folder, 409 while a learn or calibrate job of it
    runs, 404 when it has nothing there (naming the models that have it)."""
    from .learn import state
    name = str(body.get("name") or "")
    model = _served_here(request, _job_model(body.get("model"), "delete from", prefer=models.profile_paths))
    why = state.busy_with(name) if models.NAME_RE.fullmatch(name) else None
    if why:
        raise HTTPException(status_code=409, detail=why)
    try:
        definition = models.is_definition(models.deletable(model, name))
        where = models.trash(model, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"nothing moved: {exc}")
    return dict(name=name, model=model, checkpoint=models.checkpoint_of(model), trash=str(where),
                **({"definition": True} if definition else {}))

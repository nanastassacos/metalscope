"""metalscope in front of ds4-server.

``metalscope ds4 serve`` starts ds4-server (the metalscope build, which takes a
request's "scope" options) on a private port and this proxy on the public one.
The proxy answers the OpenAI routes by passing them through; a request that is
traced (``X-Scope-Trace``, or the global mode) or steered (``X-Scope-Steer``)
gets ds4's scope options and is followed into a saved trace (backend.py). It
also serves the /scope routes, so Pi uses a ds4 model as it uses
an oMLX one (Pi finds it on the default port when oMLX serves no model; ``METALSCOPE_URL`` points it elsewhere).

Chat and text completions, Anthropic messages and Responses requests are
traced and steered alike (ds4-server takes the scope options on all four).
A streamed chat completion with ``X-Scope-Edit`` answers an earlier traced
reply again from one of its tokens, the edit at that token only (metalscope/edit.py).

Every route is for this machine's own clients (``LocalOnly``): a request a web
page of another site sent is refused, and a POST to /v1 must be JSON.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anyio
import httpx
from fastapi import FastAPI, Request
from starlette.datastructures import Headers

from .. import config, edit as editing, gpulock, lookup, web
from ..backends import ds4_chat
from ..capture import capture_config
from ..engines.ds4 import Ds4Engine
from ..edit import HOP_HEADERS, refuse, relay
from ..live import MANAGER, new_request_id
from ..middleware import OTHER_SITE, TRACED_PATHS, _reject, from_other_site
from . import scope_dir
from .backend import Ds4Tracer, Run, runs_dir
from .frames import write_steering
from .vocab import model_shape, tokenizer_for

log = logging.getLogger("metalscope")
# ds4-server runs the model in its own process: nothing here can unload it (its output head is read from the GGUF:
# Ds4Tracer.unembed); a completion's
# prompt of token ids is fed as given (POST /v1/scope/tokenize makes them); a GPU job measures the served model
# itself (engines/ds4.py), while chat is refused (``v1``); a reply is answered again from one token (``Ds4Edit``);
# a saved reply is read again without the n-gram lookup (``lookup_lp``, lookup.py).
# Each model's thinking controls and flags: backends.ds4_chat (ds4-server reads chat_template_kwargs.enable_thinking
# for every model it serves).
BACKEND = web.Backend(engine="ds4", caps=frozenset({"token_prompts", "measure_served", "edit_at"}),
                      chat_formats=lambda: {t.name: ds4_chat(t.family) for t in MANAGER.loaded()})


@dataclass
class Served:
    name: str             # the model id clients see, and metalscope's name for it (traces; its checkpoint: models.py)
    gguf: Path
    upstream: str         # ds4-server's base URL
    tracer: Ds4Tracer


def make_served(gguf: Path, name: Optional[str], upstream: str) -> Served:
    """The served model: its tracer, with its J-lens loaded now (and the J-lens written as ds4 reads it),
    not on the first traced request. Refused (SystemExit) when the GGUF's shape is not the name's checkpoint's
    (models.check_shape): its vectors and lens would not fit."""
    from .. import models
    shape = model_shape(gguf)
    name = name or f"{gguf.stem}-ds4"
    try:
        models.check_shape(name, models.shape_of(shape["n_layer"], shape["n_hc"], shape["n_embd"]))
    except models.ShapeMismatch as exc:
        raise SystemExit(str(exc)) from None
    tracer = Ds4Tracer(name, tokenizer_for(gguf), **shape, gguf=gguf)
    MANAGER.tracers[id(tracer)] = tracer
    tracer.install(MANAGER)
    tracer.use_lens_step(MANAGER.lens_step())     # the saved step's J-lens file is the one kept
    keep = tracer.jlens_file()
    # J-lens files of other lenses or layer sets, and a write cut short, from earlier runs (never removed while
    # serving: a request in flight may name one). The lenses off, all kept: + reads them again without writing one.
    off = tracer.lens_step == config.LENS_OFF
    for old in [] if off else scope_dir().glob("jlens-*.bin*"):
        if old != keep:
            old.unlink(missing_ok=True)
    return Served(name, gguf, upstream.rstrip("/"), tracer)


def scope_files(tracer: Ds4Tracer) -> dict:
    """A traced request's scope options for its J-lens. Blocking (a J-lens is written once, ~0.8 GB): off the event
    loop."""
    jl = tracer.jlens_file()
    return {"jlens": str(jl)} if jl is not None else {}


def client_body(body: bytes) -> Optional[dict]:
    """A client's request body without a "scope" object; None when it is not a JSON object. ds4-server takes its
    scope options from any request (they name files it writes and reads), so only this proxy sets them: the body is
    always sent on as it parses here, never as the client's bytes."""
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):        # not JSON, not UTF-8, or nested too deep to parse
        return None
    if not isinstance(data, dict):
        return None
    # ds4-server decodes \u0000 to a NUL and compares keys as C strings: "scope\u0000" would be its "scope".
    if any("\x00" in k for k in data):
        return None
    data.pop("scope", None)
    return data


RUNS_STALE_S = 600.0                    # a request file this old when serve starts is left over (not another serve's)


def is_json(content_type: Optional[str]) -> bool:
    """``Content-Type: application/json``, with or without parameters (``; charset=utf-8``)."""
    return (content_type or "").split(";", 1)[0].strip().lower() == "application/json"


class LocalOnly:
    """Pure ASGI, in front of every proxy route: a request a web page of another site sent (from_other_site: a
    Host or Origin that is not this machine, or Sec-Fetch-Site: cross-site) is refused with 403, and a POST to /v1
    whose body is not declared JSON with 415. A page can send a text/plain POST without asking (no CORS preflight);
    it cannot send application/json so. Pi, curl, the OpenAI client and same-origin fetches pass."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            if from_other_site(headers):
                return await _reject(send, 403, OTHER_SITE)
            if scope.get("method") == "POST" and scope["path"].startswith("/v1/") and not is_json(headers.get("content-type")):
                return await _reject(send, 415, "send the request body as JSON, with Content-Type: application/json")
        return await self.app(scope, receive, send)


def build_app(served: Served, transport: Optional[httpx.AsyncBaseTransport] = None):
    """The proxy's ASGI app (``transport``: tests stand a fake ds4-server behind it)."""
    from ..middleware import REQUEST_CTX, TraceContextMiddleware

    client = httpx.AsyncClient(base_url=served.upstream, timeout=httpx.Timeout(None, connect=10.0), transport=transport)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        await client.aclose()

    app = FastAPI(title="metalscope ds4 proxy", lifespan=lifespan)
    engine: list = []         # the Ds4Engine the lookup measures on, made at its first ask

    async def lookup_lp(_tracer, ids: list, start: int, on: bool, job: lookup.Run):
        """lookup.Measure on ds4-server itself (engines/ds4.py), one request at a time as the engine sends them: ``job``
        told as each prefill chunk ends (V4.1: then each row fed alone; lookup.Run.fed); called off, the request is
        dropped (ds4-server stops it between its chunks)."""
        def run():
            try:
                if not engine:
                    engine.append(Ds4Engine(served.name, served.gguf, served.upstream))
                return engine[0].next_logprobs(ids, start, on, job.stop, job.fed)
            except httpx.HTTPError as exc:
                raise RuntimeError(f"ds4-server did not answer ({type(exc).__name__}: {exc})") from None
        return await lookup.whole(run)          # cancelled, it still waits for ds4-server to end the request
    web.mount(app, dataclasses.replace(BACKEND, lookup=lookup_lp))

    @app.get("/v1/models")
    async def models():
        model = {"id": served.name, "object": "model", "owned_by": "ds4", "created": int(time.time())}
        # ds4-server's --ctx, under the name oMLX gives it (Pi's extension reads it as the context window). No answer
        # from ds4-server: 503, not the model listed as served (`metalscope status`, Pi and `ds4 start` read this).
        try:
            r = await client.get("/v1/models", timeout=5.0)
            r.raise_for_status()
            ctx = (r.json().get("data") or [{}])[0].get("context_length")
        except (httpx.HTTPError, ValueError, AttributeError, IndexError) as exc:
            why = f"ds4-server is not answering for {served.name}: {exc}"
            return refuse(503, why)
        if ctx:
            model["max_model_len"] = ctx
        return {"object": "list", "data": [model]}

    @app.get("/scope/api/ds4")
    async def ds4_info():
        """What a measurement needs to reach ds4-server itself (engines/ds4.py): the model's name, its GGUF and
        ds4-server's own URL (requests there skip the proxy, which would replace their scope options)."""
        return {"name": served.name, "gguf": str(served.gguf), "upstream": served.upstream}

    recorded = MANAGER.kept      # the newest traced requests' token ids, for an edit (edit.py)

    async def send(request: Request, path: str, body: bytes):
        """ds4-server's streamed answer to ``body`` at ``path``, with the client's headers less the hop ones; a 502
        JSONResponse when it cannot be reached."""
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        req = client.build_request(request.method, path, content=body, headers=headers,
                                   params=dict(request.query_params))
        try:
            return await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            return refuse(502, f"ds4-server: {exc}")

    async def forward(request: Request, path: str, body: bytes):
        return relay(await send(request, path, body))

    @app.api_route("/v1/{rest:path}", methods=["GET", "POST", "DELETE"])
    async def v1(request: Request, rest: str):
        path = f"/v1/{rest}"
        if request.method == "POST" and path != "/v1/scope/tokenize":
            # A GPU job measuring this model (learn/window.py Ds4Window) sends its requests to ds4-server directly, which
            # answers one at a time: chat waits for the job, refused with its message as oMLX refuses a load (an
            # OpenAI error body, as oMLX's).
            holder = gpulock.held()
            if holder is not None:
                why = f"Model '{served.name}' cannot answer now: {holder.doing()}; chat works again when it is done"
                return refuse(409, why)
        body = await request.body()
        data = None
        if request.method == "POST" or body:
            # Every body, traced or not (a POST's is declared JSON: LocalOnly): a client's "scope" object never
            # reaches ds4-server, and a body that is not a JSON object is refused (ds4-server's own parser might
            # still find one in it). ds4-server's own port, which a page can post to as well, opens scope files
            # only in scope_dir() (its --scope-dir).
            data = client_body(body)
            if data is None:
                return refuse(400, "the request body is not a JSON object")
            body = json.dumps(data).encode()
        ctx = REQUEST_CTX.get()
        if request.method != "POST" or path not in TRACED_PATHS or ctx is None:
            return await forward(request, path, body)
        level = MANAGER.resolve_level(ctx.get("header"))
        spec = ctx.get("steer")
        if ctx.get("edit") is not None:
            return await editing.answer(ctx["edit"], path, data, ctx.get("meta") or {}, level, spec, Ds4Edit(request))
        if level is None and spec is None:
            return await forward(request, path, body)
        upstream, _rid, done, chunk = await scoped(request, path, data, level, spec, ctx.get("meta") or {})
        return relay(upstream, done, chunk)

    async def scoped(request: Request, path: str, data: dict, level, spec, meta: dict, base=None):
        """A traced or steered request sent on to ds4-server with its scope options: (``send``'s answer or a refusal,
        its request id, ``done``, ``chunk``). ``done(served_ok)`` (None when nothing was registered) must run once the
        exchange is over (``relay`` hands it to RelayResponse): it saves a traced request, keeps its token ids
        (``recorded``; ``base``, an edit's (prompt, reply so far): kept as that prompt and the reply so far then its
        own) and removes its files. ``chunk`` times a traced reply's text as it passes."""
        rid = new_request_id()
        meta = dict(meta, model=served.name)
        try:
            MANAGER.register(rid, level, meta, steer=spec)
        except Exception as exc:
            # A steer plan that cannot be built refuses the request, as oMLX's 503 does; a traced one is served untraced.
            MANAGER.release(rid)          # in case it raised after taking the request
            if spec is None:
                log.warning("metalscope: request served untraced: %s", exc)
                return await send(request, path, json.dumps(data).encode()), rid, None, None
            return refuse(503, f"{config.NO_STEERING}: {exc}"), rid, None, None
        files: list[Path] = []
        run = None
        ended = []      # done() ran (once): it saves the request, or releases it, and removes its files

        def done(served_ok: bool) -> None:
            if ended:
                return
            ended.append(served_ok)
            if run is not None:
                recorded.finishing(rid)

            def wrap_up():
                try:
                    if run is not None:
                        run.finish(served_ok)
                        if run.reader.end is None:
                            recorded.missing(rid, "ds4 recorded no end for that reply: it cannot be edited")
                        else:
                            recorded.put(rid, run.reader.prompt or [], run.reader.end.sampled, base, served.name)
                    else:
                        MANAGER.finish(rid, [], [], served.tracer)
                except Exception:
                    log.exception("metalscope: saving %s failed", rid[:8])
                finally:
                    recorded.drop(rid)            # not kept: an edit waiting for it stops waiting
                    MANAGER.release(rid)          # finish raised before it took the request: not left active
                    for f in files:
                        f.unlink(missing_ok=True)
            threading.Thread(target=wrap_up, name=f"ds4-finish-{rid[:8]}", daemon=True).start()

        try:
            rt = MANAGER.requests.get(rid)
            # The layers to read the lenses at: the step register gave the tracer, kept while this request is traced.
            scope: dict = {"topk": capture_config().logits_topk, "lens": served.tracer.read_layers()[1]}
            plan = rt.steer if rt is not None else None
            # Read once: nothing before write_steering below marks the request failed.
            steering = spec is not None and plan is not None and rt.steer_failed is None
            if level is not None:
                trace_path = runs_dir() / f"{rid}.bin"
                files.append(trace_path)
                scope["trace"] = str(trace_path)
                try:
                    scope.update(await anyio.to_thread.run_sync(scope_files, served.tracer))
                except Exception:
                    # Traced without the J-lens rather than not at all (the server log says why).
                    log.exception("metalscope: %s's J-lens could not be prepared", rid[:8])
                run = Run(rid, served.tracer, trace_path)
                run.start()
            if steering and (plan.vectors or plan.remove):
                steer_path = runs_dir() / f"{rid}.steer"
                files.append(steer_path)          # before it is written: a failed write is removed too
                try:
                    write_steering(steer_path, plan.vectors, served.tracer.width, plan.remove)
                    scope["steer"] = str(steer_path)
                    scope["steer_decode_only"] = plan.positions == "decode"
                except ValueError as exc:
                    MANAGER.steer_failed(rid, str(exc))
            # A traced reply's decode speed: timed by its pieces of text as they pass (TraceManager.note_chunk).
            chunk = (lambda c: MANAGER.note_chunk(rid, c)) if level is not None else None
            return await send(request, path, json.dumps(dict(data, scope=scope)).encode()), rid, done, chunk
        except BaseException:
            # Failed or cancelled before ds4-server answered (a step above raised, the client left): nothing to save,
            # and the request must not stay active.
            if not ended:
                ended.append(False)
                if run is not None:
                    run.stop()
                MANAGER.release(rid)
                for f in files:
                    f.unlink(missing_ok=True)
            raise

    class Ds4Edit:
        """ds4 for an edit (edit.answer): its completions are ``scoped`` ones, prompts of token ids as given."""
        name, model, tokenizer = "ds4", served.name, served.tracer.tokenizer
        stop_in_reply = True          # ds4's recorded reply ends with its stop token

        def __init__(self, request: Request):
            self.request = request

        async def complete(self, body: dict, level, spec, meta: dict, base):
            return await scoped(self.request, "/v1/completions", body, level, spec, meta, base=base)

        async def budget(self) -> Optional[int]:
            """ds4-server's own max_tokens."""
            try:
                r = await client.get("/v1/models", timeout=5.0)
                return r.json()["data"][0]["top_provider"]["max_completion_tokens"]
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
                return None

        async def kept_bytes(self, kept: list, rest) -> bytes:
            """The kept tokens' bytes: ds4-server streams the rest's as bytes too, so a character split at the join
            is decoded whole."""
            return self.tokenizer.decode_bytes(kept)

    # Steering looks up the served copy, whatever name the client sent; a page of another site is refused first.
    return LocalOnly(TraceContextMiddleware(app, resolve_model=lambda _: served.name, edits=True))


# ---------------------------------------------------------------- launching

def start_ds4(ds4_dir: Path, gguf: Path, port: int, extra: list[str], log_path: Path) -> subprocess.Popen:
    """ds4-server from its checkout (it reads metal/*.metal relative to the working directory), opening a request's
    scope files only in metalscope's scope directory (--scope-dir)."""
    exe = ds4_dir / "ds4-server"
    if not exe.exists():
        raise SystemExit(f"{exe} not found: build the metalscope ds4 branch there first (make)")
    # Last, so it is the one ds4-server keeps whatever the options passed on hold.
    cmd = [str(exe), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), *extra,
           "--scope-dir", str(scope_dir().resolve())]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(config.rotated_log(log_path), "ab")
    out.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
    out.flush()
    return subprocess.Popen(cmd, cwd=ds4_dir, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(url: str, proc: subprocess.Popen, timeout: float) -> None:
    """Until ds4-server answers, and answers metalscope's tokenize route: a ds4-server without --scope-dir (or without
    the patch) serves a scoped request as any other, so a steered reply would come back unsteered with a 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"ds4-server exited with {proc.returncode} while loading (see its log)")
        try:
            if httpx.get(f"{url}/v1/models", timeout=2.0).status_code == 200:
                if httpx.post(f"{url}/v1/scope/tokenize", json={"text": ""}, timeout=10.0).status_code == 200:
                    return
                raise SystemExit("this ds4-server takes no scope requests (no --scope-dir, or not metalscope's "
                                 "build): run `metalscope ds4 install`, then start again")
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise SystemExit(f"ds4-server did not answer within {timeout:.0f}s")


def serve(gguf: Path, name: Optional[str], port: int, ds4_dir: Path, ds4_port: int, extra: list[str]) -> None:
    import uvicorn

    from .. import backends
    from ..engines.ds4 import sweep_runs
    from ..middleware import patch_uvicorn_shutdown
    # Before the model loads (minutes): a taken port would otherwise be found only once it had.
    backends.require_free_ports({port: "the ds4 proxy (--port)", ds4_port: "ds4-server (--ds4-port)"})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gguf = gguf.expanduser().resolve()
    upstream = f"http://127.0.0.1:{ds4_port}"
    served = make_served(gguf, name, upstream)
    # Request files an earlier serve left behind (it was killed, or ds4-server wrote after the request's files went).
    sweep_runs(runs_dir(), stale_s=RUNS_STALE_S)
    ctx = backends.strip_ctx(extra)[1] or backends.DEFAULT_CTX
    log_path = config.home() / "ds4" / "ds4-server.log"
    stopping = threading.Event()      # serve itself stops ds4-server (Ctrl-C, `ds4 stop`, the proxy failed)
    proc: Optional[subprocess.Popen] = None

    def stop(*_):
        stopping.set()
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    exited = []                       # ds4-server's exit code, when it exited while the proxy served
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # SIGTERM as Ctrl-C: the finally below runs
    lock = backends.free_for_ds4(served.name, log.info)         # oMLX's models unloaded; held until ds4-server answers
    try:
        backends.write_serving(served.name, ctx)
        proc = start_ds4(ds4_dir.expanduser().resolve(), gguf, ds4_port, extra, log_path)
        # Its pid recorded: `ds4 stop` stops this ds4-server, never another one on the machine.
        backends.write_serving(served.name, ctx, server=proc.pid)
        log.info("metalscope: ds4-server pid %d loading %s (log %s)", proc.pid, gguf.name, log_path)
        wait_ready(upstream, proc, backends.DS4_LOAD_TIMEOUT)
        backends.register(served.name, gguf, extra)              # loaded: these options work, Pi's /model lists it
        lock.release()                # loaded: oMLX may load again (its own memory check now sees ds4's)
        log.info("metalscope: serving %s on http://127.0.0.1:%d (ds4-server on %d)", served.name, port, ds4_port)
        patch_uvicorn_shutdown()
        server = uvicorn.Server(uvicorn.Config(build_app(served), host="127.0.0.1", port=port, log_level="warning"))

        def watch():
            # ds4-server gone (a crash, killed): the proxy stops too, so nothing reports the model as served and
            # `metalscope ds4 MODEL` starts it again.
            code = proc.wait()
            if not stopping.is_set() and not server.should_exit:
                exited.append(code)
                log.error("metalscope: ds4-server exited with %s; stopping (see %s)", code, log_path)
                server.handle_exit(signal.SIGTERM, None)
        threading.Thread(target=watch, name="ds4-server-watch", daemon=True).start()
        server.run()
    finally:
        stop()
        lock.release(clean=proc is None or proc.poll() is not None)   # still running (killed, not reaped): memory not back
        backends.clear_serving()
    if exited:
        raise SystemExit(f"ds4-server exited with {exited[0]} while serving {served.name} (see {log_path})")

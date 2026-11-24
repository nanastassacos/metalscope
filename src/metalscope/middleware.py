"""The request side every backend shares: the local-only guard (``_local_only`` on the /scope routes, ``from_other_site``
in the ds4 proxy too), the ASGI middleware that reads a request's trace and steer headers into ``REQUEST_CTX``, and
ending the live streams when uvicorn shuts down.

oMLX's app gets the middleware inside its own (omlx/patches.py patch_server) and its EngineCore patch reads
``REQUEST_CTX``; the ds4 proxy wraps its app in it (ds4/proxy.py) and reads it in its relay.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import re

import anyio
from fastapi import HTTPException, Request

from . import config, models
from .live import MANAGER

# Set by the ASGI middleware for the duration of one API request.
REQUEST_CTX: contextvars.ContextVar[dict | None] = contextvars.ContextVar("metalscope_ctx", default=None)

TRACED_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses")


async def _reject(send, status: int, message: str) -> None:
    body = json.dumps({"error": {"message": message, "type": "metalscope"}}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


LOOPBACK = ("127.0.0.1", "::1", "localhost")


_HOST = r"(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?"
# A Host header or an Origin URL that names this machine and nothing else: no user info, nothing after the port but an
# Origin's path (http://127.0.0.1@evil.example is evil.example to a browser).
LOOPBACK_URL = re.compile(rf"(?i)::1|{_HOST}|https?://{_HOST}(?:/.*)?")


def _loopback_url_host(value: str) -> bool:
    """True if a ``Host`` header or an ``Origin`` URL names this machine (LOOPBACK_URL)."""
    return bool(LOOPBACK_URL.fullmatch(value.strip()))


OTHER_SITE = "metalscope answers only this machine's own clients, not another site's requests"


def from_other_site(headers) -> bool:
    """True when a browser sent the request for a page of another site: a ``Host`` that is not this machine (DNS
    rebinding), an ``Origin`` that is not (a cross-origin page; oMLX allows every CORS origin), or ``Sec-Fetch-Site:
    cross-site``. Clients that are not browsers (Pi, curl, the OpenAI client) send a loopback Host and no Origin.
    ``headers``: a request's headers (anything with ``.get``)."""
    host, origin = headers.get("host"), headers.get("origin")
    return bool((host and not _loopback_url_host(host)) or (origin and not _loopback_url_host(origin))
                or (headers.get("sec-fetch-site") or "").lower() == "cross-site")


def _local_only(request: Request) -> None:
    client = request.client.host if request.client else ""
    if client not in LOOPBACK:
        raise HTTPException(status_code=403, detail="metalscope is only available from this machine")
    # A page from another site can still make the local browser send the request.
    if from_other_site(request.headers):
        raise HTTPException(status_code=403, detail=OTHER_SITE)


def conversation_meta(body: dict) -> dict:
    """Group turns of one chat: clients resend the whole history every turn.
    Reads chat completions (``messages``), Anthropic messages (``messages``,
    a top-level ``system``) and Responses bodies (``input``, ``instructions``)."""
    messages = body.get("messages")
    if messages is None and isinstance(body.get("input"), list):
        messages = body["input"]
    if not isinstance(messages, list):
        messages = []
    prompt = body.get("prompt") if isinstance(body.get("prompt"), str) else (
        body["input"] if isinstance(body.get("input"), str) else "")

    def flat(c):
        if isinstance(c, list):  # content parts: OpenAI, Anthropic and Responses blocks with "text"
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and isinstance(p.get("text", ""), str))
        return c if isinstance(c, str) else ""

    def text(m):
        return flat(m.get("content") if isinstance(m, dict) else None)

    def asks(m):
        """A user turn, not an Anthropic user message made only of tool results."""
        if not isinstance(m, dict) or m.get("role") != "user":
            return False
        c = m.get("content")
        return not (isinstance(c, list) and c and all(isinstance(p, dict) and p.get("type") == "tool_result" for p in c))

    users = [text(m) for m in messages if asks(m)]
    system = next((text(m) for m in messages if isinstance(m, dict) and m.get("role") == "system"), "") \
        or flat(body.get("system")) or flat(body.get("instructions"))
    anchor = json.dumps([body.get("model"), system, users[0] if users else prompt])
    return dict(
        conversation=hashlib.sha1(anchor.encode()).hexdigest()[:10],
        turn=len(users),
        last_user=users[-1] if users else prompt,
        model=body.get("model"),
    )


def _steer_of(name: str, value: str, model: str):
    """A steer header's spec or an edit header's edit, read and checked (ValueError, steer.NotReady): every part's
    vector exists, and the served copy has its strength table. Run off the event loop: it reads profiles and vectors."""
    from . import steer
    if name == config.STEER_HEADER:
        spec = steer.with_defaults(steer.parse_header(value), model)
        steer.check_spec(spec, model)
        return spec
    edit = steer.parse_edit(value, model)
    steer.check_spec(edit["steer"], model)
    return edit


class TraceContextMiddleware:
    """Pure ASGI middleware: read the trace and steer headers and the request body.

    Pure ASGI (not BaseHTTPMiddleware) so the contextvar set here is visible
    to the endpoint and to the streaming generator it returns. Any X-Scope header sent for a page of another site
    (from_other_site) is refused with 403 before anything else. A steer header
    (steer.py) is refused with 403 while steering is off in the settings and
    with 400 when it does not parse, names no stored vector (or a layer it lacks) for the body's
    model, or the served copy has no strength table yet (steer.check_spec); an accepted one is echoed back, its parts'
    layers included (``SteerSpec.accepted``), in ``X-Scope-Steer-Accepted``. ``resolve_model`` maps the body's model
    to the served copy's name (oMLX: an alias to its model id) before any of that. An edit header (``X-Scope-Edit``,
    steer.parse_edit) is checked as a steer is, into ``REQUEST_CTX["edit"]``; refused with 400
    unless ``edits``: True, the app answers it (the ds4 proxy); a function, it does
    (``edits(scope, body, receive, send, app)``, the body a JSON object: oMLX's, omlx/edit.py).
    """

    def __init__(self, app, resolve_model=None, edits=False):
        self.app = app
        self.resolve_model = resolve_model
        self.edits = edits

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or not scope["path"].startswith(TRACED_PATHS):
            return await self.app(scope, receive, send)
        header = client = steer_header = edit_header = None
        ours = False
        for k, v in scope.get("headers") or []:
            name = k.decode("latin-1").lower()
            ours = ours or name.startswith("x-scope-")
            if name == config.HEADER:
                header = v.decode("latin-1")
            elif name == config.CLIENT_HEADER:
                client = v.decode("latin-1")[:128]
            elif name == config.STEER_HEADER:
                # A blank value means "no steer" (/replay's unsteered column sends one), not a bad header.
                steer_header = v.decode("latin-1").strip() or None
            elif name == config.EDIT_HEADER:
                edit_header = v.decode("latin-1").strip() or None
        if ours:
            # oMLX allows every CORS origin and header: a page of another site could steer, or learn the steering
            # state and vector names from the refusals below.
            from starlette.datastructures import Headers
            if from_other_site(Headers(scope=scope)):
                return await _reject(send, 403, OTHER_SITE)
        if edit_header is not None and not self.edits:
            return await _reject(send, 400, "this server cannot answer again from one token (X-Scope-Edit)")
        # Off the event loop (every reply it serves stalls meanwhile): the settings read again, and on a change the
        # hooks installed (a J-lens load, seconds); a steer's strength table measured (check_spec); the hooks a
        # traced or steered request needs (MANAGER.prepare). The engine's register then finds them done.
        mode = await anyio.to_thread.run_sync(MANAGER.mode)
        if header is None and steer_header is None and edit_header is None and mode == "off":
            return await self.app(scope, receive, send)

        # The body, as the messages it came in (the app gets them again below). No size limit of our own: oMLX's
        # is outside this middleware (patch_server) and has refused or cut an oversized body by now.
        messages, more = [], True
        while more:
            messages.append(await receive())
            more = messages[-1]["type"] == "http.request" and messages[-1].get("more_body", False)
        queued = iter(messages)

        async def replay():
            return next(queued, None) or await receive()

        if messages[-1]["type"] != "http.request":      # the client left, or the body was cut: not ours to trace
            return await self.app(scope, replay, send)
        body = b"".join(m.get("body", b"") for m in messages)
        try:
            parsed = json.loads(body or b"{}")
        except (ValueError, RecursionError):        # not JSON, or nested too deep to parse
            parsed = None
        # Not a JSON object (a list, a number): no metadata here; the engine answers it.
        meta = conversation_meta(parsed) if isinstance(parsed, dict) else {}
        if meta.get("model") and self.resolve_model is not None:
            # An alias the server lists (oMLX's model aliases) names the served copy its vectors, doses and traces
            # are filed under; the conversation anchor above keeps the name the client sent.
            try:
                meta["model"] = self.resolve_model(meta["model"]) or meta["model"]
            except Exception:
                pass
        if client:
            meta["client"] = client

        spec = edit = None
        if steer_header is not None or edit_header is not None:
            from . import steer
            if not MANAGER.steer_enabled():
                return await _reject(send, 403, "steering is off; `metalscope steering on` enables it")
            model = meta.get("model") or ""
            if not models.NAME_RE.fullmatch(config.short_name(model)):
                # Not one file name (``..``, ``qwen3:8b``): metalscope files no data under it, so it has none.
                return await _reject(send, 400, f"metalscope has no steering data for model {model!r}")
            for name, value in ((config.STEER_HEADER, steer_header), (config.EDIT_HEADER, edit_header)):
                if value is None:
                    continue
                try:
                    got = await anyio.to_thread.run_sync(_steer_of, name, value, model)
                except steer.NotReady as exc:
                    return await _reject(send, 400, f"metalscope cannot steer yet: {exc}")
                except ValueError as exc:
                    return await _reject(send, 400, f"bad {name} header: {exc}")
                spec, edit = (got, edit) if name == config.STEER_HEADER else (spec, got)
        if MANAGER.resolve_level(header) is not None or spec is not None or edit is not None:
            await anyio.to_thread.run_sync(MANAGER.prepare, meta.get("model"))
        token = REQUEST_CTX.set(dict(header=header, meta=meta, steer=spec, edit=edit))

        async def send_with_steer(message):
            if message["type"] == "http.response.start" and message.get("status", 200) < 400:   # not on a refusal
                headers = list(message.get("headers") or [])
                headers.append((config.STEER_RESPONSE_HEADER.encode(), spec.accepted.encode("latin-1", "replace")))
                message = {**message, "headers": headers}
            await send(message)

        out = send_with_steer if spec is not None else send
        try:
            if edit is not None and callable(self.edits):
                if not isinstance(parsed, dict):
                    return await _reject(send, 400, "the request body is not a JSON object")
                return await self.edits(scope, parsed, replay, out, self.app)
            return await self.app(scope, replay, out)
        finally:
            REQUEST_CTX.reset(token)


def patch_uvicorn_shutdown() -> None:
    """End live SSE streams when uvicorn starts shutting down; otherwise its
    graceful shutdown waits forever on a client (Pi's dock) that keeps /scope/api/live
    open (and a second SIGTERM does not force it)."""
    try:
        from uvicorn.server import Server
    except ImportError:
        return
    if getattr(Server, "_metalscope_patched_handle_exit", False):
        return
    Server._metalscope_patched_handle_exit = True
    orig = Server.handle_exit

    @functools.wraps(orig)
    def handle_exit(self, sig, frame):
        from .web import SHUTTING_DOWN
        SHUTTING_DOWN.set()
        return orig(self, sig, frame)

    Server.handle_exit = handle_exit

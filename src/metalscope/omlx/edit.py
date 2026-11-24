"""The edit at one token (metalscope/edit.py) on oMLX: its two completions run in this process through oMLX's own
/v1/completions route (its API key check, model lease and disconnect handling, as for any request). That route reads
a prompt as text; patches.py's EngineCore wrapper gives oMLX the token ids instead (``REQUEST_CTX["base"]``) and
starts the request's text after the kept tokens (patches._continued)."""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from starlette.datastructures import Headers
from starlette.responses import Response

from .. import config, edit
from ..live import MANAGER
from ..middleware import REQUEST_CTX

CLOSE_WAIT_S = 10.0        # oMLX notices a client that left within 2 s (its keepalive poll); then cancelled


class Answer:
    """oMLX's answer to a request this process made of its app (``call``): what edit.py reads of an httpx
    response, and the request id and kept text the EngineCore wrapper put in its context."""

    def __init__(self, ctx: dict, start: dict, queue: asyncio.Queue, gone: asyncio.Event, task: asyncio.Task):
        self.ctx, self._q, self._gone, self._task, self._more = ctx, queue, gone, task, True
        self.status_code = start["status"]
        self.headers = Headers(raw=list(start.get("headers") or []))

    @property
    def rid(self) -> Optional[str]:
        return self.ctx.get("rid")

    async def aiter_raw(self):
        while self._more:
            msg = await self._q.get()
            if msg is None:                                    # the app returned
                self._more = False
            elif msg["type"] == "http.response.body":
                self._more = msg.get("more_body", False)
                if msg.get("body"):
                    yield msg["body"]
            elif msg["type"] == "error":                       # it raised mid-stream: an error event, the end
                self._more = False
                yield b"data: " + json.dumps({"error": {"message": msg["message"], "type": "server_error"}}).encode() + b"\n\n"

    async def aread(self) -> bytes:
        return b"".join([c async for c in self.aiter_raw()])

    async def aclose(self) -> None:
        """Done with it: the request's receive says the client left (oMLX aborts the request), and the app is waited
        for, so its own cleanup (the model lease) runs to the end; cancelled only if it does not finish."""
        self._gone.set()
        if not self._task.done():
            await asyncio.wait({self._task}, timeout=CLOSE_WAIT_S)
        if not self._task.done():
            self._task.cancel()

    async def kept_text(self) -> Optional[str]:
        """The kept tokens' text as oMLX's detokenizer or output parser for this request holds it (patches._continued),
        once the request reached the engine; None when it never did or the edit is refused it."""
        started = asyncio.ensure_future(self.ctx["started"].wait())
        await asyncio.wait({started, self._task}, return_when=asyncio.FIRST_COMPLETED)
        started.cancel()
        return self.ctx.get("text")


async def call(app, scope: dict, ctx: Optional[dict], body: Optional[dict] = None,
               path: str = "/v1/completions") -> Answer:
    """``body`` POSTed to ``path`` of oMLX's app (inside the middleware) as a request of this process (no body: a
    GET), with the client's headers (its API key) and ``ctx`` as its REQUEST_CTX; returns once the response
    starts."""
    raw = json.dumps(body).encode() if body is not None else b""
    headers = [(k, v) for k, v in scope.get("headers") or [] if k.lower() not in (b"content-length", b"content-type")]
    if body is not None:
        headers += [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]
    sub = dict(scope, method="POST" if body is not None else "GET", path=path, raw_path=path.encode(),
               query_string=b"", headers=headers)
    q: asyncio.Queue = asyncio.Queue()
    gone, sent = asyncio.Event(), []

    async def receive():
        if not sent:
            sent.append(True)
            return {"type": "http.request", "body": raw, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def run():
        try:
            await app(sub, receive, q.put)
        except Exception as exc:
            await q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            await q.put(None)

    token = REQUEST_CTX.set(ctx)
    try:
        task = asyncio.ensure_future(run())          # in ``ctx``: the EngineCore wrapper reads it
    finally:
        REQUEST_CTX.reset(token)
    while True:
        msg = await q.get()
        if msg is None or msg["type"] == "error":
            why = msg["message"] if msg is not None else "no response"
            return Answer(ctx or {}, {"status": 500, "headers": [(b"content-type", b"application/json")]},
                          _ended(json.dumps({"error": {"message": why, "type": "server_error"}}).encode()), gone, task)
        if msg["type"] == "http.response.start":
            return Answer(ctx or {}, msg, q, gone, task)


def _ended(body: bytes) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait({"type": "http.response.body", "body": body, "more_body": False})
    return q


class Omlx:
    """oMLX for an edit (edit.answer): its completions are requests of this process to oMLX's own route."""
    name = "oMLX"
    stop_in_reply = False         # oMLX leaves the stop token out of a reply's tokens

    def __init__(self, app, scope: dict, tracer):
        self.app, self.scope, self.tokenizer = app, scope, tracer.tokenizer
        self.model = config.short_name(tracer.name)

    async def complete(self, body: dict, level, spec, meta: dict, base):
        ids = body["prompt"]
        ctx = dict(header=level, meta=meta, steer=spec, edit=None, base=(list(base[0]), list(base[1])),
                   rid=None, text=None, started=asyncio.Event())
        # oMLX's route reads the text (its length checks); the engine is given ``base``'s ids.
        return await call(self.app, self.scope, ctx, dict(body, prompt=self.tokenizer.decode(ids))), None, None, None

    async def budget(self) -> Optional[int]:
        return None              # oMLX's own default for the rest

    async def kept_bytes(self, kept: list, held) -> bytes:
        """The kept tokens' text as the detokenizer (or Inkling 2's parser) of ``held``'s request holds it (the rest's,
        or the edited token's when it ended the reply), so the rest's text joins it exactly and no marker is shown;
        decoded here when there is none."""
        text = await held.kept_text() if held is not None else None
        return (text if text is not None else self.tokenizer.decode(kept)).encode()


async def _key_refused(app, scope: dict) -> Optional[Response]:
    """oMLX's own API key check, before anything about the reply is answered: GET /v1/models, which checks the key
    as the completion route does and runs no model. Its refusal, or None."""
    gate = await call(app, scope, None, path="/v1/models")
    try:
        raw = await gate.aread()
    finally:
        await gate.aclose()
    if gate.status_code < 400:
        return None
    return Response(raw, status_code=gate.status_code, media_type=gate.headers.get("content-type"))


async def answer(scope, body: dict, receive, send, app) -> None:
    """The middleware's ``edits`` on oMLX (middleware.TraceContextMiddleware): the edit answered here, its
    completions by oMLX's app."""
    from ..web import _tracer_named
    from .patches import edit_caps
    ctx = REQUEST_CTX.get() or {}
    meta = ctx.get("meta") or {}
    tracer = _tracer_named(meta.get("model"))
    response = await _key_refused(app, scope)
    if response is None and "edit_at" not in edit_caps():
        response = edit.refuse(400, "this server cannot answer again from one token (X-Scope-Edit)")
    elif response is None and tracer is None:
        response = edit.refuse(409, f"{config.short_name(meta.get('model')) or 'that model'} is not loaded: "
                                    "send it a message, then edit")
    elif response is None:
        response = await edit.answer(ctx["edit"], scope["path"], body, meta, MANAGER.resolve_level(ctx.get("header")),
                                     ctx.get("steer"), Omlx(app, scope, tracer))
    await response(scope, receive, send)
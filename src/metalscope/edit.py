"""An earlier reply answered again from one token (``X-Scope-Edit``, steer.parse_edit), on either engine: ``answer``
puts it together, the engine (ds4/proxy.py, omlx/edit.py) runs its two completions.

The reply's tokens before the edited one stay as the engine sampled them, ids and all (``Recorded``, the trace
manager's ``kept``, keeps the newest traced requests' ids). The edited token is one greedy completion of a prompt of
token ids: the reply's prompt and its tokens before the edit, the edit's steer at the prompt's last row only
(``positions=decode``, ``max_tokens=1``). The rest is a completion from there with no edit (the steer on, if any).
Both completions are traced: the edited token's trace holds the lens at the edited row, the rest's the continuation.
Each is kept as the whole reply the dock shows (the original prompt; the kept tokens, then its own), so an edit of an
edited reply names its token as the dock counts it. The client asked for a chat completion, so the reply goes back as
one: the kept tokens' text first, then the rest as it comes (decoded across the join: a character can be split
between two tokens), split into reasoning and content at ``</think>`` (``ChatText``) as the engines' chat streams
split it. The rest stops where a tool call would start (``TOOL_OPENERS``): its markup is the engine's chat parser's
to read, and a token after one is not edited. Inkling 2's opener is one token, literal in its raw tokens: oMLX
stops the rest at it as a stop sequence of tokens, before its parser reads it. Only a reply's first thinking is read
as reasoning (ChatText): a second thinking block's <think> would show in the text.
"""

from __future__ import annotations

import codecs
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import AsyncIterator, Optional

import anyio
import numpy as np
from starlette.responses import JSONResponse, Response, StreamingResponse

log = logging.getLogger("metalscope")

THINK_START, THINK_END = "<think>", "</think>"
# ds4-server's tool call openers (ds4_server.c find_any_tool_start): Qwen's, DeepSeek V4.1's and V4's DSML; then
# Inkling 2's, one token, literal in its raw tokens (oMLX stops at it as a stop sequence of tokens).
TOOL_OPENERS = ["<tool_call>", "<tool_calls>", "<｜DSML｜ calls>", "<｜DSML｜tool_calls>", "<DSML｜tool_calls>",
                "<|content_invoke_tool_json|>"]
KEEP = 16          # traced requests whose ids are kept for an edit
WAIT_S = 40.0      # a reply still served or wrapped up is waited for this long (ds4's END record: up to 30 s)
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding", "host"}


NOT_KEPT = "that reply's tokens are not kept here (untraced, or older than the last 16): edit a newer one"
OTHER_MODEL = "that reply is another model's: edit one of this model's"


class Recorded:
    """The token ids of the newest traced requests, (prompt, reply), by request id, with the model that wrote them,
    or why a request's are not (``missing``). A request still being wrapped up is waited for (``get``): an edit of a
    reply cut short arrives while its capture is still read."""

    def __init__(self, keep: int = KEEP):
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._keep = keep

    def finishing(self, rid: str) -> None:
        self._set(rid, None)

    def put(self, rid: str, prompt, reply, base=None, model: Optional[str] = None) -> None:
        """``base``, an edit's (prompt, reply before this request): kept as that prompt and that reply, then its own."""
        if base is not None:
            prompt, reply = base[0], [*base[1], *reply]
        self._set(rid, (np.asarray(prompt, np.int32), np.asarray(reply, np.int32), model))

    def missing(self, rid: str, why: str) -> None:
        self._set(rid, why)

    def drop(self, rid: str) -> None:
        with self._lock:
            if rid in self._d and self._d[rid] is None:
                del self._d[rid]

    def _set(self, rid, value) -> None:
        with self._lock:
            self._d[rid] = value
            self._d.move_to_end(rid)
            while len(self._d) > self._keep:
                self._d.popitem(last=False)

    async def get(self, rid: str, active, wait_s: float, model: Optional[str] = None) -> tuple[list[int], list[int]] | str:
        """(prompt, reply) of request ``rid``; while it is still served (``active(rid)``) or wrapped up, waited for up
        to ``wait_s``; else why not (``missing``, ``NOT_KEPT``, ``OTHER_MODEL`` when ``model`` did not write it)."""
        end = time.monotonic() + wait_s
        while True:
            with self._lock:
                got = self._d.get(rid, False)
            if isinstance(got, str):
                return got
            if got:
                if model is not None and got[2] is not None and got[2] != model:
                    return OTHER_MODEL
                return got[0].tolist(), got[1].tolist()
            if (got is False and not active(rid)) or time.monotonic() > end:
                return NOT_KEPT
            await anyio.sleep(0.05)


def _held(buf: str, marker: str) -> int:
    """How many characters at the end of ``buf`` could begin ``marker``: held back until the next piece."""
    for k in range(min(len(buf), len(marker) - 1), 0, -1):
        if buf.endswith(marker[:k]):
            return k
    return 0


class ChatText:
    """A reply's raw text as a chat completion's deltas: ``reasoning_content`` up to ``</think>`` while the prompt
    opened the thinking (``thinking``) or the text itself does (it begins with ``<think>``, dropped: Inkling 2's on
    oMLX, its parser's text), ``content`` after it (its leading newlines dropped)."""

    def __init__(self, thinking: bool):
        self.thinking, self.buf, self.started, self.fresh = thinking, "", False, True

    def feed(self, text: str, final: bool = False) -> list[tuple[str, str]]:
        self.buf += text
        out = []
        if self.fresh and self.buf:                  # nothing read yet: does the text open the thinking?
            if self.buf.startswith(THINK_START):
                self.buf, self.thinking = self.buf[len(THINK_START):], True
            elif not final and THINK_START.startswith(self.buf):
                return out                           # it may yet: held until the next piece
            self.fresh = False
        if self.thinking:
            i = self.buf.find(THINK_END)
            if i < 0:
                n = len(self.buf) - (0 if final else _held(self.buf, THINK_END))
                if n > 0:
                    out.append(("reasoning_content", self.buf[:n]))
                    self.buf = self.buf[n:]
                return out
            if i:
                out.append(("reasoning_content", self.buf[:i]))
            self.buf, self.thinking = self.buf[i + len(THINK_END):], False
        if not self.started:
            self.buf = self.buf.lstrip("\n")
            self.started = bool(self.buf)
        if self.buf:
            out.append(("content", self.buf))
            self.buf = ""
        return out


def thinking_open(tail: str) -> bool:
    """Whether a prompt ending in ``tail`` (its last few tokens' text) leaves the reply in its thinking."""
    return tail.rstrip().endswith("<think>")


def chunk(cid: str, model: str, delta: Optional[dict] = None, finish: Optional[str] = None,
          usage: Optional[dict] = None) -> bytes:
    """One chat completion chunk as ds4-server streams it."""
    body = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
            "choices": [] if usage is not None else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n".encode()


def raw_text(s: str) -> bytes:
    """A completion event's text as the bytes ds4-server sent (``event``): a character split between two tokens comes
    as its bytes, not UTF-8."""
    try:
        return s.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:
        return s.encode("utf-8", "replace")


def event(line: bytes):
    """An SSE ``data:`` line's JSON, its bytes that are not UTF-8 kept as surrogates (``raw_text``); None when it is
    not JSON."""
    try:
        return json.loads(line.decode("utf-8", "surrogateescape"))
    except ValueError:
        return None


async def chat_stream(raw: Optional[AsyncIterator[bytes]], kept: bytes, text: ChatText, cid: str, model: str,
                      done_tokens: int, prompt_tokens: int, usage: bool, failed: Optional[list] = None,
                      ) -> AsyncIterator[bytes]:
    """The edited reply as a chat completion stream: the role, ``kept`` (the reply's bytes up to and with the edited
    token, ``done_tokens`` tokens), then the completion stream ``raw`` (the engine's SSE; None: the edited token
    ended the reply) as deltas, its finish reason, the usage when asked for (``prompt_tokens`` the reply's prompt).
    An error event from the engine is passed on and ends the stream (``failed`` then holds it)."""
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")

    def deltas(piece: bytes, final: bool = False) -> list[bytes]:
        return [chunk(cid, model, {field: s}) for field, s in text.feed(utf8.decode(piece, final), final)]

    yield chunk(cid, model, {"role": "assistant"})
    for c in deltas(kept):
        yield c
    finish, used, buf = None, 0, b""
    if raw is not None:
        async for piece in raw:
            *lines, buf = (buf + piece).split(b"\n")
            for line in lines:
                line = line.strip()
                if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                    continue
                ev = event(line[5:])
                if not isinstance(ev, dict):
                    continue
                if ev.get("error") is not None:            # ds4-server's error event: passed on, the end
                    if failed is not None:
                        failed.append(ev["error"])
                    for c in deltas(b"", True):
                        yield c
                    yield b"data: " + json.dumps(ev).encode() + b"\n\n"
                    yield chunk(cid, model, finish="error")
                    yield b"data: [DONE]\n\n"
                    return
                used = int((ev.get("usage") or {}).get("completion_tokens") or used)
                for ch in ev.get("choices") or []:
                    for c in deltas(raw_text(ch.get("text") or "")):
                        yield c
                    finish = ch.get("finish_reason") or finish
    for c in deltas(b"", True):
        yield c
    yield chunk(cid, model, finish=finish or "stop")
    if usage:
        n = done_tokens + used
        yield chunk(cid, model, usage={"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                       "total_tokens": prompt_tokens + n})
    yield b"data: [DONE]\n\n"


# ---------------------------------------------------------------- relaying an engine's answer

def refuse(status: int, why: str) -> JSONResponse:
    """An OpenAI error body, as oMLX's."""
    return JSONResponse({"error": {"message": why, "type": "metalscope"}}, status_code=status)


class RelayResponse(StreamingResponse):
    """An engine's response passed through as it arrives (``upstream``: an httpx response, or omlx/edit.py's
    in-process one; ``convert``: rewritten on the way, an async generator of the raw chunks). Whatever ends it (the
    end of the stream, the client leaving, a cancelled task), the upstream is closed and ``on_done(served_ok)`` runs
    exactly once: a traced request is then saved and its files removed."""

    def __init__(self, upstream, on_done, on_chunk=None, convert=None, **kw):
        self._upstream, self._on_done, self._on_chunk, self._ok = upstream, on_done, on_chunk, False
        super().__init__(self._body(convert), **kw)

    async def _raw(self):
        async for chunk in self._upstream.aiter_raw():
            if self._on_chunk is not None:
                self._on_chunk(chunk)
            yield chunk

    async def _body(self, convert):
        async for chunk in (convert(self._raw()) if convert is not None else self._raw()):
            yield chunk
        self._ok = self._upstream.status_code < 400

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self._upstream.aclose()
                except Exception:
                    log.exception("metalscope: closing the engine's response failed")
            if self._on_done is not None:
                done, self._on_done = self._on_done, None
                done(self._ok)


def relay(upstream, on_done=None, on_chunk=None) -> Response:
    """An engine's answer as the client's response (``on_done(False)`` at once when it is a refusal of ours)."""
    if isinstance(upstream, Response):
        if on_done:
            on_done(False)
        return upstream
    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS}
    return RelayResponse(upstream, on_done, on_chunk, status_code=upstream.status_code, headers=out_headers,
                         media_type=upstream.headers.get("content-type"))


# ---------------------------------------------------------------- the edit

async def answer(e: dict, path: str, data: dict, meta: dict, level, spec, eng) -> Response:
    """A streamed chat completion with ``X-Scope-Edit`` (``e``: steer.parse_edit's): reply ``e["rid"]`` answered
    again from its token ``e["at"]``, the edit at that token only, the rest under the steer on (``spec``), as one chat
    completion stream. Two traced completions: the edited token's and the rest's (each live start names the reply it
    continues), each kept as the whole reply so far.

    ``eng``, the engine: ``name`` (in messages), ``model`` (the served copy's name), ``stop_in_reply`` (whether the
    stop token that ends a reply is among its kept tokens), ``tokenizer`` (``decode``),
    ``complete(body, level, spec, meta, base)`` (a text completion of ``body``, its prompt token ids, traced at
    ``level``, steered by ``spec``, kept as ``base``, a (prompt, reply so far): its answer or a refusal, its request id
    (None: the answer's ``rid`` once read), ``done(served_ok)`` or None, a chunk timer or None), ``budget()`` (the
    server's own max_tokens, or None) and ``kept_bytes(kept, held)`` (what the client gets for the kept tokens;
    ``held`` the answer whose request continued from them: the rest's, or the edited token's when it ended the reply).
    The text opens the thinking where the prompt did or where it begins with ``<think>`` (ChatText). Refusals are 4xx:
    Pi does not send them again."""
    from .live import MANAGER
    from .steer import SteerSpec
    if path != "/v1/chat/completions" or not data.get("stream"):
        return refuse(400, "an edit answers a streamed chat completion")
    kept_ids = MANAGER.kept
    active = lambda rid: rid in MANAGER.requests
    got = await kept_ids.get(e["rid"], active, WAIT_S, eng.model)
    if isinstance(got, str):
        return refuse(409, got)
    prompt, reply = got
    at = e["at"]
    if at >= len(reply):
        return refuse(400, f"that reply has no token {at}")
    tok = eng.tokenizer
    if any(o in tok.decode(reply[:at]) for o in TOOL_OPENERS):
        return refuse(400, "that token comes after a tool call: edit one before it")
    parts = (spec.parts if spec is not None else ()) + e["steer"].parts
    if len({(p.control, p.name) for p in parts}) < len(parts):
        return refuse(400, "the edit names a vector the steer on has too")
    level = level or "on"           # traced: the edited token's id is read from its trace
    label = e["steer"].label
    # The edited token, greedy: the edit (and the steer on) at the prompt's last row only. Marked an edit's own steer
    # (``SteerSpec.edit``: the only steer a word vector takes, steer.WORD_EDIT_ONLY).
    one = dict(model=data.get("model"), prompt=prompt + reply[:at], max_tokens=1, temperature=0, stream=False)
    upstream, rid_a, done, _chunk = await eng.complete(one, level, SteerSpec(parts, positions="decode", edit=True),
                                                       dict(meta, edit=dict(of=e["rid"], at=at, steer=label)),
                                                       (prompt, reply[:at]))
    if isinstance(upstream, Response):
        return relay(upstream, done)
    ok = False
    try:
        raw = await upstream.aread()
        ok = upstream.status_code < 400
    finally:
        with anyio.CancelScope(shield=True):
            await upstream.aclose()
        if done is not None:
            done(ok)
    if not ok:
        return Response(raw, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))
    rid_a = rid_a or getattr(upstream, "rid", None)
    try:
        ended = json.loads(raw)["choices"][0].get("finish_reason") == "stop"   # it ended the reply
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        ended = False
    mine = await kept_ids.get(rid_a, active, WAIT_S) if rid_a else NOT_KEPT
    own = mine[1][at:] if isinstance(mine, tuple) else []
    # The stop token that ended the reply: ds4 keeps it in a reply's tokens (none recorded is a failure), oMLX leaves
    # it out (none is the stop).
    if not own and (eng.stop_in_reply or not (ended and isinstance(mine, tuple))):
        return refuse(502, f"{eng.name} recorded no edited token")
    kept = reply[:at] + ([] if ended else own)
    text = ChatText(thinking_open(tok.decode(prompt[-4:])))
    model = data.get("model") or eng.model
    usage = bool((data.get("stream_options") or {}).get("include_usage"))
    failed: list = []

    def body(raw, kept_text: bytes):
        return chat_stream(raw, kept_text, text, f"chatcmpl-{rid_a}", model, len(kept), len(prompt), usage, failed)
    if ended:                       # the kept tokens are the edited token's request's: its text of them
        return StreamingResponse(body(None, await eng.kept_bytes(kept, upstream)), media_type="text/event-stream")
    # The rest: no edit, the steer on as it was; it stops before a tool call. Its budget: the client's, else the
    # server's own, less the tokens kept.
    rest = dict(model=data.get("model"), prompt=prompt + kept, stream=True, stream_options={"include_usage": True},
                stop=TOOL_OPENERS, **{k: data[k] for k in ("temperature", "top_p", "top_k", "min_p", "seed") if k in data})
    cap = data.get("max_tokens") or data.get("max_completion_tokens")
    if not isinstance(cap, int):
        cap = await eng.budget()
    if isinstance(cap, int):
        rest["max_tokens"] = max(1, cap - len(kept))
    upstream, _rid, done, chunk = await eng.complete(rest, level, spec,
                                                     dict(meta, edit=dict(of=rid_a, at=len(kept), steer=label)),
                                                     (prompt, kept))
    if isinstance(upstream, Response) or upstream.status_code >= 400:
        return relay(upstream, done, chunk)
    kept_text = await eng.kept_bytes(kept, upstream)
    # An error event mid-stream: saved as not served (chat_stream's ``failed``).
    finished = (lambda ok: done(ok and not failed)) if done is not None else None
    return RelayResponse(upstream, finished, chunk, convert=lambda raw: body(raw, kept_text),
                         media_type="text/event-stream")

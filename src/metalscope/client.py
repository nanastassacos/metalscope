"""Talking to a metalscope server: oMLX's (metalscope runs inside it) or the ds4 proxy (``metalscope ds4 serve``).

One place for which servers there are and which one serves a model, and for the requests every caller makes: JSON
under ``/scope/api``, a rendered prompt as the server's own token ids, and completions traced under one client id
with that client's per-token events from the live feed (``Session``). Pi's provider.ts mirrors ``candidates`` in
TypeScript. oMLX's admin API (load, unload, restart) is omlx/admin.py's.

    candidates()      METALSCOPE_URL when set, else oMLX's URL (~/.omlx/settings.json), then the ds4 proxy's
    serving()         those answering with a served model, as Clients with their status
    for_model(name)   the Client of the server serving ``name`` (else the first serving, else the first candidate)
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from . import config
from .omlx import admin as omlx_admin

DS4_URL = "http://127.0.0.1:8009"   # metalscope ds4 serve's default port
# A request that never got a whole answer: refused or reset, timed out, or its reply cut short (IncompleteRead).
BROKEN = (OSError, ValueError, http.client.HTTPException)


# A job's own request (its grader, its writer, a steered calibration reply): not traced whatever the server's mode, so
# it costs no trace's memory and is no chat in the tally.
UNTRACED = {config.HEADER: "off"}


class ServerError(RuntimeError):
    """A request the server refused (``status``: its HTTP code, ``detail``: its reason) or could not be reached for or
    gave no whole answer to (``status`` None; the transport's error as its ``__cause__``)."""

    def __init__(self, message: str, status: Optional[int] = None, detail: Optional[str] = None):
        super().__init__(message)
        self.status, self.detail = status, detail


def candidates() -> list[str]:
    url = config.env("URL")
    return [url.rstrip("/")] if url else [omlx_admin.server_url(), DS4_URL]


def _where(url: str) -> tuple:
    """A URL's host and port, every name of this machine as one."""
    u = urllib.parse.urlsplit(url)
    host = u.hostname or ""
    return ("127.0.0.1" if host in ("127.0.0.1", "::1", "localhost") else host, u.port or 80)


def omlx_key(url: str) -> Optional[str]:
    """oMLX's API key when ``url`` is oMLX's own address, else None: whatever listens on another port (the ds4 proxy's
    while ds4 is down, METALSCOPE_URL's) never gets it."""
    settings = omlx_admin.read_settings()
    if _where(url) != _where(omlx_admin.server_url(settings)):
        return None
    return (settings.get("auth") or {}).get("api_key")


class Client:
    """Requests to the server at ``url``, with oMLX's API key when it is oMLX's (``omlx_key``)."""

    def __init__(self, url: str, api_key: Optional[str] = None):
        self.url = url.rstrip("/")
        self.api_key = api_key if api_key is not None else omlx_key(self.url)
        self._status: Optional[dict] = None
        self.status_error: Optional[ServerError] = None     # why status() is None: its .status 401, the key refused

    def __repr__(self) -> str:
        return f"Client({self.url!r})"

    def _open(self, method: str, path: str, body, timeout: float, headers: Optional[dict]):
        """The open response; ServerError as ``request`` says."""
        data = None if body is None else json.dumps(body).encode()
        headers = {**({"Content-Type": "application/json"} if body is not None else {}),
                   **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}), **(headers or {})}
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read().decode(errors="replace")
            except BROKEN:                  # its body cut short (oMLX out of memory): its status still says enough
                raw = ""
            try:
                parsed = json.loads(raw)
                err = parsed.get("error") if isinstance(parsed, dict) else None
                detail = (parsed.get("detail") if isinstance(parsed, dict) else None) or \
                    (err.get("message") if isinstance(err, dict) else err if isinstance(err, str) else None)
            except ValueError:
                detail = None
            detail = detail or raw[:300]
            raise ServerError(f"{method} {path}: HTTP {exc.code}: {detail}", exc.code, detail) from None
        except BROKEN as exc:
            raise ServerError(f"no metalscope server at {self.url} ({exc})") from exc

    def request(self, method: str, path: str, body=None, timeout: float = 60, headers: Optional[dict] = None,
                as_text: bool = False):
        """JSON in and out (``as_text``: the answer's text as it came, e.g. a whole event stream). ServerError with
        the server's reason (its ``detail``) when it refuses, with no status when it cannot be reached."""
        r = self._open(method, path, body, timeout, headers)
        try:
            with r:
                raw = r.read() or b"{}"
        except BROKEN as exc:
            raise ServerError(f"{method} {path}: the reply broke off ({exc})") from exc
        # Leniently: ds4-server sends a reply cut at max_tokens as its bytes, which can end inside a UTF-8 character,
        # and the ds4 proxy passes them through.
        text = raw.decode("utf-8", "replace")
        if as_text:
            return text
        try:
            return json.loads(text)
        except ValueError:
            raise ServerError(f"{method} {path}: the reply is not JSON: {raw[:300]!r}") from None

    def stream(self, path: str, body: dict, timeout: float = 900, headers: Optional[dict] = None):
        """A streamed POST's events as they arrive: (seconds since it was sent, the event's JSON). ServerError as
        ``request`` says."""
        t0 = time.time()
        r = self._open("POST", path, body, timeout, headers)
        try:
            with r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if line.startswith("data:") and line[5:].strip().startswith("{"):
                        yield time.time() - t0, json.loads(line[5:])
        except BROKEN as exc:
            raise ServerError(f"POST {path}: the stream broke ({exc})") from exc

    def get(self, path: str, timeout: float = 60) -> dict:
        return self.request("GET", path, timeout=timeout)

    def post(self, path: str, body: dict, timeout: float = 60, headers: Optional[dict] = None) -> dict:
        return self.request("POST", path, body, timeout=timeout, headers=headers)

    def status(self, timeout: float = 3, fresh: bool = False) -> Optional[dict]:
        """/scope/api/status (engine, caps, models, mode...), None when nothing answers or it refuses (``status_error``
        says which: status None, nothing answered; 401, oMLX refused the key); kept after the first ask (``fresh``:
        asked again)."""
        if self._status is None or fresh:
            try:
                self._status, self.status_error = self.get("/scope/api/status", timeout=timeout), None
            except ServerError as exc:
                self.status_error = exc
                return None
        return self._status

    def caps(self) -> list[str]:
        return list((self.status() or {}).get("caps") or [])

    def served(self) -> list[str]:
        return [m["name"] for m in (self.status() or {}).get("models") or []]

    def tokenize(self, text: str) -> list[int]:
        """A rendered prompt as the server's own tokenizer reads it, chat markers included (POST /v1/scope/tokenize on
        a ``token_prompts`` server): the token ids it feeds as given."""
        return [int(t) for t in self.post("/v1/scope/tokenize", {"text": text, "specials": True})["tokens"]]


def not_served(name: str) -> str:
    """What a job says for a model no server serves or can load (backends.require_served, web's job routes)."""
    return f"{name} is not being served: start it with `metalscope omlx {name}` or `metalscope ds4 {name}`"


def serving() -> list[Client]:
    """Each candidate answering with a served model, oMLX's first."""
    out = []
    for url in candidates():
        c = Client(url)
        if c.served():
            out.append(c)
    return out


def for_model(name: Optional[str]) -> Client:
    """The server serving ``name`` (by short name); for a name none serves, the first candidate (oMLX's, which loads
    a model on demand; a running ds4 proxy serves only its own); with no name, the first serving one."""
    found = serving()
    if name:
        want = config.short_name(name)
        for c in found:
            if want in (config.short_name(m) for m in c.served()):
                return c
    return found[0] if found and not name else Client(candidates()[0])


class Session:
    """Completions with the raw rendered prompt (as token ids on a ``token_prompts`` server), traced (level on) under
    one client id, and the live feed's per-token events for that client (the server publishes them only while someone
    subscribes)."""

    def __init__(self, server: Client, client: str, wait: float = 10.0, experts: bool = False, lens: bool = False,
                 feed: bool = True):
        """``wait``: how long to wait for the live feed to connect; ``experts``, ``lens``: the feed's routing and top
        10s too, as the dock asks for them (web.live); ``feed`` False: the replies' text only, no feed subscribed
        (the server then publishes nothing for it)."""
        self.server, self.client = server, client
        self.asks = "".join(f"&{k}=true" for k, on in (("experts", experts), ("lens", lens)) if on)
        self.token_prompts = "token_prompts" in server.caps()
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.alive = False
        self.stop = False
        if not feed:
            return
        threading.Thread(target=self._listen, daemon=True).start()
        t0 = time.time()
        while not self.alive and time.time() - t0 < wait:
            time.sleep(0.1)
        if wait:
            time.sleep(0.5)      # slack only: the server subscribes this client before it answers (web.live)

    def _listen(self) -> None:
        path = f"/scope/api/live?client={urllib.parse.quote(self.client)}{self.asks}"
        while not self.stop:
            try:
                with self.server._open("GET", path, None, 600, None) as r:     # with oMLX's key, as every request
                    self.alive = True
                    for raw in r:
                        line = raw.decode(errors="replace").strip()
                        if line.startswith("data:"):
                            try:
                                ev = json.loads(line[5:])
                            except ValueError:
                                continue
                            with self.lock:
                                self.events.append(ev)
            except Exception:
                self.alive = False
                time.sleep(1.0)

    def complete(self, model: str, prompt: str, steer: str, max_tokens: int, timeout: float = 900,
                 temperature: Optional[float] = None, traced: bool = True) -> dict:
        """The reply's text and its token events: their texts, entropies and probabilities, and the events
        themselves (``events``) with the reply's request id (``rid``). ``steer`` blank: unsteered; ``temperature`` None:
        the server's default, as Pi sends none; ``traced`` False: the text only, the server keeps no trace of it.
        ServerError when the server refuses."""
        with self.lock:
            start = len(self.events)
        body = dict(model=model, prompt=self.server.tokenize(prompt) if self.token_prompts else prompt,
                    max_tokens=max_tokens, stream=False)
        if temperature is not None:
            body["temperature"] = temperature
        t0 = time.time()
        out = self.server.request("POST", "/v1/completions", body, timeout=timeout,
                                  headers={config.HEADER: "on" if traced else "off", config.CLIENT_HEADER: self.client,
                                           config.STEER_HEADER: steer})
        seconds = time.time() - t0
        if not out.get("choices"):          # oMLX aborted it (its model unloaded under it): its reason, if it gave one
            err = out.get("error")
            why = (err.get("message") if isinstance(err, dict) else err) or out.get("detail")
            raise ServerError(f"the server ended the request: {why}" if why else
                              "the server ended the request without a reply", 200, why)
        toks, rid = [], None
        while traced and self.alive and time.time() - t0 < seconds + 20:      # no feed, or no trace: no events to wait for
            with self.lock:
                evs = [e for e in self.events[start:] if e.get("client") == self.client]
            ends = [e for e in evs if e.get("type") == "end"]
            if ends:
                rid = ends[-1].get("rid")
                toks = sorted((e for e in evs if e.get("type") == "token" and e.get("rid") == rid),
                              key=lambda e: e.get("i", 0))
                break
            time.sleep(0.2)
        with self.lock:              # read: a session that runs for hours keeps none of them (the next reply's come after)
            self.events.clear()
        return dict(text=out["choices"][0].get("text", ""), seconds=round(seconds, 1),
                    tok_text=[e.get("text") or "" for e in toks],
                    entropy=[e.get("entropy") for e in toks], p=[e.get("p") for e in toks], events=toks, rid=rid)

    def close(self) -> None:
        self.stop = True

"""The ds4 engine: measurements on the model ds4-server serves (``metalscope ds4 serve``).

Every call is one request straight to ds4-server's own port, which the proxy names at /scope/api/ds4 (through the
proxy a request's scope options would be replaced): the prompt as token ids, fed as given (ds4-server applies no
chat template to them), greedy, traced to a capture file ``frames.ScopeReader`` reads, and the adds in a DS4STEER
file, added after their layers at every position, prompt and reply (``steer_decode_only`` false). Both files are
removed after the request. One prompt per request, one request at a time; each reads its whole prompt (scope
``fresh``: no reuse of what the session holds).

- ``residuals``: the request's residual sums (scope ``rsum``, RSUM): per layer the mean of the prompt rows from the
  sink on and the prompt's last row, as ds4 sums them on the GPU;
- ``logprobs``: the prompt's last row's log-probabilities at the ids (scope ``logit_ids``, STEP flag 64);
- ``generate``: the reply's tokens (END_), ``stopped`` when ds4 finished on a stop token;
- ``next_logprobs``: each prompt row's next-token log-probability and top 10 (scope ``next_lp``, NXLP and NXTK), the
  lookup left out with scope ``lookup`` false. Called off (``stop``), its request is dropped: ds4-server watches each
  request's socket and cancels a dropped one between its prefill chunks (8,192 tokens on Qwen3.8; on V4.1, whose
  sweep reads no row's head, the rows from ``start - 1`` on go one at a time). Meanwhile the capture is read as it
  grows: each forward's STEP record says how far it is (``fed``).

Encoding and chat turns are ds4's own tokenizer's (POST /v1/scope/tokenize), decoding the GGUF's vocabulary
(ds4/vocab.py). No per-position residuals: ds4 does not ship rows to the host.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

from .. import client, models
from ..ds4 import scope_dir
from ..ds4.frames import VERSION, ScopeReader, Step, write_steering
from ..ds4.vocab import model_shape, read_gguf_meta, tokenizer_for
from . import END_TOKENS, SINK, Engine, EngineUnavailable, NextLp, Shape, Stopped, Tokenizer, checked

MAX_IDS = 256                         # logit ids ds4 reads per request (ds4.c, DS4_SCOPE_MAX_IDS)
END_WAIT = 30.0                       # s a capture may take to end after ds4-server's reply (it ends just after; the proxy's bound)
STALE_S = 3600.0                      # a request's files this old are left over (no measurement request takes an hour)
STOP_POLL = 0.1                       # s between looks at a request's stop (ds4-server looks at its socket as often)
BEGIN_WAIT = 0.3                      # s a dropped request's capture may take to appear (begun as it was dropped)


def sweep_runs(runs: Path, now: Optional[float] = None, stale_s: float = STALE_S) -> None:
    """Remove request files a run left behind: a cancelled or killed run's last request, which ds4-server may still
    write after the run removed its files (the proxy's runs too, as `ds4 serve` starts: proxy.py)."""
    now = time.time() if now is None else now
    for f in [*runs.glob("*.bin"), *runs.glob("*.steer")]:
        try:
            if now - f.stat().st_mtime > stale_s:
                f.unlink()
        except OSError:
            pass


class Ds4Tokenizer(Tokenizer):
    """ds4-server's tokenizer for text and chat turns; the GGUF's vocabulary for decoding."""

    def __init__(self, engine: "Ds4Engine", gguf: Path):
        self._engine, self._vocab = engine, tokenizer_for(gguf)
        eos = read_gguf_meta(gguf).get("tokenizer.ggml.eos_token_id")
        eos = set() if eos is None else {int(eos)}
        for text in END_TOKENS:
            ids = engine.tokenize({"text": text, "specials": True})
            if len(ids) == 1:
                eos.add(ids[0])
        self.eos = eos

    def __len__(self) -> int:
        return len(self._vocab)

    def encode(self, text: str) -> list[int]:
        return self._engine.tokenize({"text": text})

    def chat(self, user: str, thinking: Optional[bool] = False) -> list[int]:
        """The prompt ds4-server renders for one user turn; ``thinking=None`` leaves its default."""
        body: dict = {"messages": [{"role": "user", "content": user}]}
        if thinking is not None:
            body["thinking"] = bool(thinking)
        return self._engine.tokenize(body)

    def decode(self, ids: list[int]) -> str:
        return self._vocab.decode(ids)


class Ds4Engine(Engine):
    kind = "ds4"

    def __init__(self, served: str, gguf: Path, upstream: str, transport: Optional[httpx.BaseTransport] = None):
        """``transport``: tests stand a fake ds4-server behind the engine."""
        self.served, self.checkpoint = served, models.checkpoint_of(served)
        self.gguf, self.upstream = Path(gguf), upstream.rstrip("/")
        self._transport = transport
        self._http = httpx.Client(base_url=self.upstream, timeout=httpx.Timeout(None, connect=10.0), transport=transport)
        self.runs = scope_dir() / "engine"           # ds4-server opens scope files only in its --scope-dir
        self.runs.mkdir(parents=True, exist_ok=True)
        sweep_runs(self.runs)
        self.model: Optional[str] = None      # the model ds4-server's first measurement reply named
        s = model_shape(self.gguf)
        self.tok = Ds4Tokenizer(self, self.gguf)
        self.shape = Shape(s["n_layer"], s["n_hc"] * s["n_embd"], s["n_hc"], s["n_embd"], len(self.tok))

    @classmethod
    def open(cls, served: str, url: Optional[str] = None) -> "Ds4Engine":
        """The engine on the model the ds4 proxy at ``url`` (default ``client.DS4_URL``) serves, which must be ``served``."""
        url = url or client.DS4_URL
        try:
            r = httpx.get(f"{url.rstrip('/')}/scope/api/ds4", timeout=10.0)
            r.raise_for_status()
            info = r.json()
        except httpx.HTTPError as exc:
            raise EngineUnavailable(f"no metalscope ds4 proxy at {url} ({exc}); start one with `metalscope ds4 serve`") from None
        if info["name"] != served:
            raise EngineUnavailable(f"the ds4 proxy at {url} serves {info['name']}, not {served}")
        return checked(cls(info["name"], Path(info["gguf"]), info["upstream"]))

    def close(self) -> None:
        self._http.close()

    # ---- ds4-server

    def _post(self, path: str, body: dict, stop: Optional[threading.Event] = None, watch=None) -> dict:
        r = self._http.post(path, json=body) if stop is None else asyncio.run(self._post_until(path, body, stop, watch))
        if r.status_code >= 400:
            raise RuntimeError(f"ds4-server {path}: HTTP {r.status_code}: {r.text[:300]}")
        # A reply cut at max_tokens can end inside a multi-byte character, and ds4-server sends those bytes as they
        # are: not valid UTF-8. The measurements read tokens from the capture, never this text.
        try:
            return json.loads(r.content.decode("utf-8", "replace"))
        except ValueError:
            raise RuntimeError(f"ds4-server {path}: the reply is not JSON: {r.content[:300]!r}") from None

    async def _post_until(self, path: str, body: dict, stop: threading.Event, watch=None) -> httpx.Response:
        """The POST, its connection dropped once ``stop`` is set (Stopped); ``watch()`` between the looks."""
        async with httpx.AsyncClient(base_url=self.upstream, timeout=httpx.Timeout(None, connect=10.0),
                                     transport=self._transport) as c:
            sent = asyncio.ensure_future(c.post(path, json=body))
            while not sent.done():
                if stop.is_set():
                    sent.cancel()
                    await asyncio.wait({sent})
                    raise Stopped("the request was called off")
                await asyncio.wait({sent}, timeout=STOP_POLL)
                if watch:
                    watch()
            return sent.result()

    def tokenize(self, body: dict) -> list[int]:
        """POST /v1/scope/tokenize: ``{"text", "specials"}`` or a chat body -> its token ids."""
        return [int(t) for t in self._post("/v1/scope/tokenize", body)["tokens"]]

    def _ended(self, reader: ScopeReader) -> list:
        """The capture's records (those ``reader`` has not read yet) once ds4-server has ended it (END_), END_WAIT s
        at most."""
        records = reader.poll()
        deadline = time.monotonic() + END_WAIT
        while reader.end is None and time.monotonic() < deadline:
            time.sleep(0.005)
            records += reader.poll()
        return records

    def _run(self, ids: list[int], max_tokens: int, adds: Optional[dict[int, np.ndarray]],
             stop: Optional[threading.Event] = None, fed=None, **scope) -> tuple[dict, ScopeReader, list]:
        """One traced, greedy completion of ``ids``: ds4-server's answer, the capture's reader and its forwards' records
        holding the logits at ``logit_ids`` (the rest are dropped as read: a long prompt's are large); ``stop`` set, its
        request dropped (Stopped). Meanwhile ``fed(rows, chunk)`` (with ``stop``) as each forward's record comes: the
        prompt's rows read so far, ``chunk`` in the last forward (a prefill chunk: 8,192 rows on Qwen3.8)."""
        rid = uuid.uuid4().hex
        trace = self.runs / f"{rid}.bin"
        files = [trace]
        reader, records = ScopeReader(trace), []

        def keep(got: list) -> None:
            records.extend(s for s in got if isinstance(s, Step) and s.out_at_ids is not None)

        def watch():
            got = reader.poll()
            keep(got)
            for s in got:
                if fed and isinstance(s, Step):
                    fed(s.pos0 + s.T, s.T)
        scope = dict(trace=str(trace), topk=1, lens=[], fresh=True, **scope)
        if adds:
            steer = self.runs / f"{rid}.steer"
            files.append(steer)
            write_steering(steer, {int(l): np.asarray(v, np.float32) for l, v in adds.items()}, self.shape.width)
            scope.update(steer=str(steer), steer_decode_only=False)
        body = dict(prompt=[int(i) for i in ids], max_tokens=int(max_tokens), temperature=0, stream=False, scope=scope)
        try:
            try:
                out = self._post("/v1/completions", body, stop, watch)
            except httpx.HTTPError:
                raise
            except Exception:
                # A request dropped (called off: Stopped; or the watch failed): ds4-server ends the capture of one it
                # began ("cancelled") once it has stopped, waited for so the caller's locks outlast its work. One it
                # had not begun (no file, a moment after) never runs. A process told to end (SIGTERM) does not wait.
                if not trace.exists():
                    time.sleep(BEGIN_WAIT)
                if trace.exists():
                    self._ended(reader)
                raise
            # Every measurement on one model: ds4-server restarted on another mid-run would mix two models' numbers.
            model = out.get("model")
            if model is not None:
                if self.model is None:
                    self.model = model
                elif model != self.model:
                    raise RuntimeError(f"ds4-server now serves {model}, not {self.model}: the model changed mid-run")
            # ds4-server answers from inside the job and ends the capture (RSUM, END_) just after: read until it
            # has, or the reply came before the file was done (once in about 2,400 requests on V4.1).
            keep(self._ended(reader))
        finally:
            for f in files:
                f.unlink(missing_ok=True)
        if reader.end is None:
            raise RuntimeError("ds4-server's capture of a measurement request ended early (see ds4-server.log)")
        if reader.end.finish != "done":
            raise RuntimeError(f"ds4-server's capture of a measurement request ended {reader.end.finish!r}, not done")
        if reader.prompt != [int(i) for i in ids]:
            raise RuntimeError(f"ds4-server fed a different prompt than the ids sent (is it the format-{VERSION} build?)")
        return out, reader, records

    # ---- the measurements

    def residuals(self, prompts_ids, layers, sink=SINK, adds=None):
        layers = [int(l) for l in layers]
        mean = np.zeros((len(prompts_ids), len(layers), self.shape.width), np.float32)
        last = np.zeros_like(mean)
        for i, ids in enumerate(prompts_ids):
            _out, reader, _records = self._run(ids, 1, adds, rsum=dict(sink=int(sink)))
            s = reader.rsum
            if s is None:
                raise RuntimeError(f"ds4-server wrote no residual sums (is it the format-{VERSION} build?)")
            want = len(ids) - min(int(sink), len(ids) - 1)
            short = [l for l in layers if int(s.n[l]) != want]
            if short:
                raise RuntimeError(f"ds4 summed fewer than the prompt's {want} rows at layers {short} (a V4.1 prompt "
                                   f"of 8192 tokens or more runs its deep layers on a suffix)")
            mean[i], last[i] = s.mean[layers], s.last[layers]
        return mean, last

    def logprobs(self, ids, token_ids, adds=None):
        token_ids = [int(t) for t in token_ids]
        if not 0 < len(token_ids) <= MAX_IDS:
            raise ValueError(f"ds4 reads 1 to {MAX_IDS} logit ids per request, not {len(token_ids)}")
        _out, reader, records = self._run(ids, 1, adds, logit_ids=token_ids)
        row = next((s.out_at_ids[-1] for s in records if s.pos0 + s.T == len(ids)), None)
        if row is None or reader.logit_ids != token_ids:
            raise RuntimeError("ds4-server's capture holds no output at the prompt's last row")
        return np.asarray(row, np.float32)

    def next_logprobs(self, ids, start, lookup=True, stop=None, fed=None):
        """Each prompt row's log-probability of the next token from row ``start - 1`` on (scope ``next_lp``, NXLP) and
        its top 10 (NXTK); ``lookup`` False: the n-gram lookup adds nothing, Qwen3.8's PLE or V4.1's Engram (scope
        ``lookup``; ds4 drops the session's rows before and after). The same request on both: ds4 feeds V4.1's rows
        from ``start - 1`` on one at a time, each through the output head. ``stop`` (a threading.Event) set: the
        request dropped, Stopped; ``fed(rows, chunk)`` as each forward's record comes (_run)."""
        if not 0 < start < len(ids):
            raise ValueError(f"start {start} is not inside the {len(ids)} ids")
        _out, reader, _records = self._run(ids, 1, None, stop, fed, next_lp=int(start), **({} if lookup else {"lookup": False}))
        rows = range(start - 1, len(ids) - 1)
        got = [reader.next_lp.get(p) for p in rows]
        if None in got:
            raise RuntimeError("ds4-server read no next-token log-probabilities (is it the build with next_lp?)")
        top = [reader.next_top.get(p) for p in rows]
        if None in top:
            raise RuntimeError("ds4-server read no next-token top choices (is it the build with NXTK?)")
        return NextLp(np.asarray(got, np.float32), np.stack([t[0] for t in top]), np.stack([t[1] for t in top]))

    def generate(self, ids, adds, max_tokens):
        out, reader, _records = self._run(ids, max_tokens, adds)
        stopped = (out.get("choices") or [{}])[0].get("finish_reason") == "stop"
        sampled = list(reader.end.sampled)
        return (sampled[:-1] if stopped and sampled else sampled), stopped

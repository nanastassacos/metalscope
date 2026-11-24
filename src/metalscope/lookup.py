"""How much a model's n-gram lookup helped each token of one finished reply (Flash Next's PLE at layer index 1: bigram
and trigram hash tables whose rows are added to the residual; DeepSeek V4.1's Engram at layers 1 and 14: 2- to
4-grams).

On demand (POST /scope/api/lookup), the served model reads the saved reply's fed tokens again (every reply token but
the last sampled one, which is never fed: ``measure``), teacher-forced, twice:
as served, and with the lookup's write left out at every position (``Engine.next_logprobs``: ds4's scope ``lookup``,
oMLX's lookup module zeroed inside the pass only). Normal serving never changes. Both log-probabilities, and each
row's top choices in either pass (the dock's TOKENS without and with), come from these two passes, not the trace: the trace keeps only each step's top 10, and its numbers come from decode steps and MTP
verifies, so on 2-7% of tokens they and a fresh pass disagree about whether the lookup helped (measured 2026-11-16).
Two fresh passes on the same path differ only by the lookup. The fresh on pass costs one prefill. The answer is kept
beside the trace (``lookup.json``), so a second ask is instant.

While the passes run, where they are goes to the asking Pi on the live feed (``Run``), and the ask can be called off
(``stop``): between the passes always, within one where the engine can (Engine.next_logprobs). Nothing is kept then.

A server lists the ``lookup`` cap while it serves a model with a lookup (``caps``; the tracer's ``lookup``) and has a
way to measure it (web.Backend ``lookup``). A token the lookup helped: ``lp_on - lp_off >= HELPED``. What an
inspection needs to know of the lookup is the tracer's ``lookup_kind`` (Kind): the answer's ``ngram``, the tokens the
dock marks before a helped one; and, where the engine feeds rows one at a time (V4.1 on ds4), what that costs, for the
expected time and the refusal of an inspection expected or found to pass INSPECT_MAX_S.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

from . import config
from .engines import NextLp, Stopped
from .store import read_index

HELPED = 0.1                 # nats: the lookup helped a token whose log-probability it raised this much
FILE = "lookup.json"
VERSION = 3                  # 2: the top choices of both passes; 3: ngram. A file of another version is measured again
ONE = threading.Lock()       # one measurement at a time
HOLDER = "inspecting the engram"     # the GPU lock's holder while it runs (gpulock.Holder)
STOP_WAIT_S = 30.0           # an ask waits this long at most for a called-off measurement to end (ds4: a prefill chunk)
PACE = 1e-3                  # s a conversation token a pass on a model never measured (ds4 Flash Next: 0.8-1.0 ms)
PACE_FROM = 1000             # tokens a measurement needs to set the pace: a shorter one is mostly a request's overhead
PACES: dict = {}             # the pace by model (``pace``)
INSPECT_MAX_S = 600.0        # s both passes may take where the engine feeds rows one at a time (Kind.row_s),
                             # expected and run: they hold the GPU lock, and the dock waits 15 min at most
TELL_EVERY = (0.25, 0.05)    # ds4's progress (Run.fed) goes out once this many s or this share of the pass have passed
                             # since it last did: V4.1's rows fed one at a time would send 14 a second
RUN: Optional["Run"] = None  # the measurement under way (the route's, while it holds ONE)
ASKS: dict = {}              # the measurements by (trace, Pi's ask), from their route's start to its end (stop, again)


@dataclass(frozen=True)
class Kind:
    """A model's n-gram lookup as an inspection reads it (the tracer's ``lookup_kind``). ``ngram``: its longest n-gram,
    the tokens the dock marks before a helped one (Flash Next's PLE: bigrams and trigrams). Where the engine feeds rows
    one at a time (ds4 on V4.1: its sweep reads no row's output head), ``row_s``: the seconds a pass spends on each, the
    reply's rows and the prompt's last; ``alone``: the most rows before them it may also feed one at a time (a worst
    case the estimate allows); ``pace``: the seconds a conversation token a pass before any measurement."""
    ngram: int = 3
    row_s: float = 0.0
    alone: int = 0
    pace: float = PACE


# (tracer, ids, start, lookup on, run) -> each token's log-probability from ``start`` on and each row's top choices
# (Engine.next_logprobs); where the pass is goes to ``run`` (Run.part); called off (Run.stop), it raises Stopped
Measure = Callable[[object, list, int, bool, "Run"], Awaitable[NextLp]]


class Refused(Exception):
    """A lookup the route answers with ``status`` and the reason."""

    def __init__(self, status: int, why: str):
        super().__init__(why)
        self.status = status


class Run:
    """One measurement under way (``RUN``), of reply ``trace``, asked by Pi's live-feed ``client`` as ``ask``. Where it
    is goes to that client on the live feed, an event of ``type`` "lookup": the pass (``reading`` 0 with the lookup, 1
    without), the share of it known done and the share at the next checkpoint (``done``, ``next``: an MLX chunk,
    MlxEngine.lookup_chunk; a ds4 prefill chunk of 8,192, on V4.1 then each reply token; told at most every TELL_EVERY),
    its seconds so far and the seconds it is expected to take (``elapsed``, ``expect``). ``stop``: called off (``over``:
    by fed, past ``limit``); ``ended``: its route is done, the answer kept or ``failed`` (the route's refusal: the same
    ask again gets it too). ``swept``: the seconds and rows of ds4's forwards of more than one row, both passes."""

    def __init__(self, trace: str, client: Optional[str] = None, ask: Optional[str] = None):
        self.trace, self.client, self.ask, self.stop, self.ended = trace, client, ask, threading.Event(), threading.Event()
        self.failed: Optional[Exception] = None
        self.reading, self.done, self.next, self.t0, self.expect = 0, 0.0, 1.0, time.monotonic(), 0.0
        self.told = (-math.inf, -math.inf)       # when fed last told and its share (TELL_EVERY)
        self.plan: Optional[tuple] = None         # (kind, model, tokens, start) of the passes (measure; fed)
        self.last, self.swept, self.limit, self.over = self.t0, [0.0, 0], math.inf, False

    def check(self) -> None:
        if self.stop.is_set():
            raise Stopped("the engram inspection was called off")

    def begin(self, reading: int, expect: float) -> None:
        """Pass ``reading`` starts, expected to take ``expect`` s."""
        self.check()
        self.reading, self.done, self.next, self.t0, self.expect = reading, 0.0, 1.0, time.monotonic(), expect
        self.told, self.last = (-math.inf, -math.inf), self.t0
        self.tell()

    def part(self, done: float, nxt: float, tell: bool = True) -> None:
        """``done`` of the pass is done, ``nxt`` will be at the next checkpoint (an MLX chunk, a ds4 prefill chunk); the
        pass expected to take as long as its part done took so far."""
        self.done, self.next = min(1.0, done), min(1.0, nxt)
        if done > 0:
            self.expect = (time.monotonic() - self.t0) / done
        if tell:
            self.tell()

    def fed(self, rows: int, chunk: int) -> None:
        """ds4's progress (engines/ds4.py ``fed``): ``rows`` read, ``chunk`` of them in its last forward (a prefill
        chunk; on V4.1 a row read alone), as shares of the pass's expected time (``plan``, share); told at most as
        often as TELL_EVERY lets. A chunk's time since the last forward goes to ``swept``; past ``limit``, stopped."""
        kind, model, n, start = self.plan
        self.part(share(kind, model, n, start, rows), share(kind, model, n, start, rows + chunk), tell=False)
        now = time.monotonic()
        if chunk > 1:
            self.swept[0] += now - self.last
            self.swept[1] += chunk
        self.last = now
        if now > self.limit:
            self.over = True
            self.stop.set()
        if now - self.told[0] >= TELL_EVERY[0] or abs(self.done - self.told[1]) >= TELL_EVERY[1]:
            self.told = (now, self.done)
            self.tell()

    def tell(self) -> None:
        if self.client:
            from .live import MANAGER
            MANAGER.hub.publish(dict(type="lookup", client=self.client, ask=self.ask, reading=self.reading,
                                     done=round(self.done, 4), next=round(self.next, 4),
                                     elapsed=round(time.monotonic() - self.t0, 2), expect=round(self.expect, 2)))


async def take(job: Run) -> bool:
    """ONE for ``job`` (then RUN), at once; or, the measurement holding it called off, once that one has ended
    (STOP_WAIT_S at most). False: another is under way. ``job`` called off meanwhile: Stopped."""
    global RUN
    end = time.monotonic() + STOP_WAIT_S
    while not ONE.acquire(blocking=False):
        job.check()
        if RUN is None or not RUN.stop.is_set() or time.monotonic() > end:
            return False
        await asyncio.sleep(0.05)
    RUN = job
    return True


def stop(trace: str, ask: Optional[str]) -> bool:
    """Call the measurement of ``trace`` asked as ``ask`` off, waiting or under way; whether there was one."""
    run = ASKS.get((trace, ask)) if ask else None
    if run is None:
        return False
    run.stop.set()
    return True


def pace(model: str, kind: Kind) -> float:
    """Seconds a conversation token a pass on ``model``: its newest measurement of PACE_FROM tokens or more's (kept
    beside its trace, read once; then each one's since; _pace), else ``kind.pace``."""
    if model not in PACES:
        PACES[model] = kind.pace
        for e in read_index():
            got = cached(e) if e.get("model") == model else None
            if got and "seconds" in got and (n := got["prompt_tokens"] + len(got["ids"])) >= PACE_FROM \
                    and (p := _pace(got["seconds"], n, kind)) is not None:
                PACES[model] = p
                break
    return PACES[model]


def _swept(kind: Kind, n: int, start: int) -> int:
    """The rows of a pass over ``n`` tokens fed from ``start`` read in prefill chunks: all, or where the engine feeds
    rows one at a time (Kind.row_s) the conversation's but its last ``kind.alone`` and the prompt's last row."""
    return n if not kind.row_s else max(0, start - 1 - kind.alone)


def _pace(seconds: dict, n: int, kind: Kind) -> Optional[float]:
    """A measurement's (its answer's ``seconds``) seconds a conversation token a pass over ``n`` tokens; where rows are
    fed one at a time (Kind.row_s), those of its forwards read in chunks (``swept``: seconds, rows; Run.fed), None
    without any."""
    if not kind.row_s:
        return (seconds["on"] + seconds["off"]) / 2 / n
    s, rows = seconds.get("swept", (0.0, 0))
    return s / rows if rows else None


def _too_long(about: str, seconds: float) -> Refused:
    return Refused(409, f"too long to inspect on this model: {about} {seconds / 60:.0f} min for both readings, "
                        f"at most {INSPECT_MAX_S / 60:.0f}")


def reading_s(kind: Kind, model: str, n: int, start: int, rows: Optional[int] = None) -> float:
    """Seconds a pass over ``n`` tokens fed from ``start`` is expected to take (its first ``rows`` only: that part):
    the rows read in chunks at the model's pace, the rest at ``kind.row_s`` each (_swept)."""
    swept, rows = _swept(kind, n, start), n if rows is None else min(rows, n)
    return min(rows, swept) * pace(model, kind) + max(0, rows - swept) * kind.row_s


def share(kind: Kind, model: str, n: int, start: int, rows: int) -> float:
    """The share of a pass done once its first ``rows`` are (reading_s): what ds4's progress reports (Run.fed)."""
    total = reading_s(kind, model, n, start)
    return reading_s(kind, model, n, start, rows) / total if total > 0 else rows / n


async def whole(fn, *args):
    """``fn(*args)`` on a worker thread, waited for to its end even when the caller is cancelled (then cancelled
    after): the model lease and the locks around a pass must outlast it. asyncio.to_thread and anyio's run_sync both
    return at a task's cancel while their thread runs on."""
    done = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(done)
    except asyncio.CancelledError:
        while not done.done():
            try:
                await asyncio.wait({done})
            except asyncio.CancelledError:
                pass
        raise


def caps(tracers) -> frozenset:
    """``{"lookup"}`` while a loaded model has a lookup."""
    return frozenset({"lookup"}) if any(getattr(t, "lookup", False) for t in tracers) else frozenset()


def cached(entry: dict) -> Optional[dict]:
    """The answer kept beside the trace, or None."""
    got = config.read_json(Path(entry["path"]) / FILE, None)
    return got if isinstance(got, dict) and got.get("version") == VERSION else None


async def measure(entry: dict, tracer, run: Measure, job: Run) -> dict:
    """The answer for saved reply ``entry`` on ``tracer``'s model: for every fed reply token its id, ``lp_on`` and
    ``lp_off`` (natural log), the two passes' seconds, and from each pass the row's NEXT_TOP likeliest next tokens
    (``top_ids_on``, ``top_lp_on``, ``top_ids_off``, ``top_lp_off``: [tokens][NEXT_TOP], likeliest first) with the text
    of every id among them (``vocab``); kept beside the trace.

    The trace keeps the tokens it fed (prompt, then the reply but its last sampled token, which is never fed), so
    ``ids[i]`` is the reply's token i (sampled at reply step i, from 0), the trace's token at ``prompt_tokens + i``,
    and ``lp_on[i]``, ``lp_off[i]`` are its log-probability given everything before it. The last sampled token has
    none: ``len(ids)`` is the reply's tokens less one. A trace lacking the conversation's first tokens (``tokens_cut``)
    is refused: read without them, every number would be another context's. Where it is goes to ``job``; called off
    (Stopped), between the passes or in one, nothing is kept. Where rows are fed one at a time (Kind.row_s), both
    passes are held to INSPECT_MAX_S: expected (reading_s), twice the first, and as they run (Run.fed); past it,
    refused."""
    try:
        meta = json.loads(await asyncio.to_thread((Path(entry["path"]) / "meta.json").read_text))
        ids, start = [int(i) for i in meta["tokens"]], int(meta["prompt_len"])
    except (OSError, ValueError, KeyError, TypeError):
        raise Refused(409, "that trace keeps no token ids") from None
    if meta.get("tokens_cut"):
        raise Refused(409, "that trace lacks the conversation's first tokens")
    if not 0 < start < len(ids):
        raise Refused(409, "that reply has no fed tokens")
    model, kind = entry.get("model"), tracer.lookup_kind
    job.plan = (kind, model, len(ids), start)
    expect = await asyncio.to_thread(reading_s, kind, model, len(ids), start)
    if kind.row_s and 2 * expect > INSPECT_MAX_S:
        raise _too_long("about", 2 * expect)
    t0 = time.monotonic()
    if kind.row_s:
        job.limit = t0 + INSPECT_MAX_S
    job.begin(0, expect)
    try:
        on = await run(tracer, ids, start, True, job)
        t1 = time.monotonic()
        if kind.row_s and 2 * (t1 - t0) > INSPECT_MAX_S:
            raise _too_long("about", 2 * (t1 - t0))
        job.begin(1, t1 - t0)
        off = await run(tracer, ids, start, False, job)
        t2 = time.monotonic()
        job.check()
    except Stopped:
        if job.over:
            raise _too_long("over", INSPECT_MAX_S) from None
        raise
    took = dict(on=t1 - t0, off=t2 - t1, swept=job.swept)
    if len(ids) >= PACE_FROM and (p := _pace(took, len(ids), kind)) is not None:
        PACES[model] = p
    seconds = dict(on=round(t1 - t0, 2), off=round(t2 - t1, 2))
    if kind.row_s:
        seconds["swept"] = [round(job.swept[0], 2), job.swept[1]]
    out = dict(version=VERSION, trace=entry["id"], model=tracer.name, prompt_tokens=start, ids=ids[start:],
               ngram=kind.ngram,
               lp_on=[round(float(x), 4) for x in on.lp], lp_off=[round(float(x), 4) for x in off.lp],
               seconds=seconds)
    for side, got in (("on", on), ("off", off)):
        out[f"top_ids_{side}"] = got.ids.astype(int).tolist()
        out[f"top_lp_{side}"] = np.round(got.top.astype(np.float64), 3).tolist()
    seen = sorted({int(i) for got in (on, off) for i in np.unique(got.ids) if i >= 0})
    out["vocab"] = {str(i): tracer.tokenizer.decode([i]) for i in seen}
    await asyncio.to_thread(config.atomic_write, Path(entry["path"]) / FILE, json.dumps(out, separators=(",", ":")))
    return out


def table(answer: dict, token_text: list[str], n: int = 10) -> list[str]:
    """The ``n`` tokens the lookup helped most, each after the tokens it looked up (the answer's ``ngram``), with both
    probabilities; ``token_text`` the trace's (prompt and reply). First a line on the whole reply."""
    on, off = np.asarray(answer["lp_on"], np.float64), np.asarray(answer["lp_off"], np.float64)
    d, p0 = on - off, answer["prompt_tokens"]
    se = d.std() / np.sqrt(len(d)) if len(d) > 1 else 0.0
    out = [f"{len(d)} tokens: lookup {d.mean():+.3f} ± {se:.3f} nats a token, helped {int((d >= HELPED).sum())}"]
    for i in np.argsort(-d)[:n]:
        if d[i] < HELPED:
            break
        at = p0 + int(i)
        before = "".join(token_text[max(0, at - answer["ngram"]):at])
        out.append(f"{d[i]:+6.2f}  {np.exp(on[i]):4.0%} on {np.exp(off[i]):4.0%} off  {before!r} -> {token_text[at]!r}")
    return out

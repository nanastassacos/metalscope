"""The expert tally: per checkpoint, how often each expert takes each token, counted from the traced chats, and the
scores the dock's expert pages rank by.

One rule counts everything: a token counts for an expert when the expert is among that token's K_TOP strongest
routing weights (of the 10 it mixes). Per checkpoint (the served copies of one pool: their expert ids agree), at
``models/<checkpoint>/experts-tally.npz`` under the data home, per MoE layer (its place among the MoE layers, as the
live stream orders them):

    k<l>, c<l>    expert x token id counts, sparse: keys expert << 20 | token id (sorted, stored as differences)
    q<l>, d<l>    expert x expert co-fire counts in one token's K_TOP, sparse: keys a << 32 | b with a < b
    rows          the token positions counted per layer
    seen          the prompt-prefix hashes counted (sorted): a prompt's prefix already counted is not counted again
    ids           the trace ids counted: nothing is counted twice, by the live update or a rebuild
    E, N          experts a layer; token positions counted

Each saved chat grows it (``Updater``: after the save, from a thread of its own, while no request runs): its reply
and the prompt tokens not already counted (the longest prefix of its tokens that an earlier counted trace had, and
the rows its engine did not read, ``read_from``). Automated runs, steered replies (steering changes the routing) and
models without routing are left out. A file lock orders the writers (oMLX and the ds4 proxy are separate processes);
the file is replaced atomically, so a reader sees the old one or the new one. A failure logs once and skips.

The tally is derived from the user's chats (token counts carry names and project words): it lives only in the data
home, never in the repo, a package or a concept pack; ``metalscope experts tally --clear`` deletes it.
No token text is stored: the routes decode ids with the model's own tokenizer.

Scores: an expert's token mix P~(t|e) = (c + a P(t)) / (n_e + a),
shrunk toward the background P(t) by ``a = floor_of(N)`` pseudo-tokens; its score the KL divergence from the
background; experts under the same floor are not ranked. ``all``: the background is the layer's own mix. ``session``
(one Pi client's traces): the background is each expert's own usual mix in the stored tally (the session's counts
taken out of it), so a session's experts rank by what is unusual for them; with no stored tally, the session's own
layer mix. Tokens rank by smoothed lift (c + 2) / (n_e P + 2).
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import queue
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import config, models
from .store import is_automated, read_index, write_npz

log = logging.getLogger("metalscope")

K_TOP = 3                 # a token counts for its K_TOP strongest experts
SHIFT = 20                # token ids below 2**20 (a key is expert << SHIFT | token)
FILE = "experts-tally.npz"
VERSION = 1
LIFT_PRIOR = 2.0          # the smoothed lift's pseudo-count, on observed and expected alike
MIN_COUNT = {"all": 3, "session": 2}   # a token listed for an expert: at least this many of it
COMPACT_ROWS = 300_000    # an Acc compacts past this many rows uncompacted (48 bytes a row: 14 MB)
SESSION_FRESH = 16        # a session's newest replies kept apart until the stored tally holds them (session)
_B_INV = np.uint64(pow(1000003, -1, 2 ** 64))


def tally_path(model: str) -> Path:
    return models.model_dir(model) / FILE


def counted(entry: dict) -> bool:
    """A saved trace the tally counts: routed, not steered, not an automated run."""
    return bool(entry.get("experts")) and not entry.get("steer") and not is_automated(entry.get("client"))


def floor_of(n: float, E: int, k: int = K_TOP) -> float:
    """The shrinkage and the evidence floor for ``n`` tokens: a quarter of an expert's mean load (k n / E), at least 10."""
    return max(k * n / max(E, 1) / 4, 10.0)


# -- counting ---------------------------------------------------------------------------------------------------------


def prefix_hashes(tok: np.ndarray) -> np.ndarray:
    """A hash of every prefix of ``tok`` (uint64, position-weighted sums): equal prefixes hash equal."""
    n = len(tok)
    if not n:
        return np.zeros(0, np.uint64)
    with np.errstate(over="ignore"):
        pw = np.full(n, _B_INV, np.uint64)
        pw[0] = 1
        pw = np.cumprod(pw, dtype=np.uint64)
        return np.cumsum(np.asarray(tok, np.uint64) * pw + np.uint64(1), dtype=np.uint64)


def _seen_prefix(g: np.ndarray, seen: np.ndarray) -> int:
    """The longest prefix whose hash is in ``seen`` (sorted): prefixes nest, so a binary search."""
    def has(i):
        j = np.searchsorted(seen, g[i])
        return j < len(seen) and seen[j] == g[i]
    lo, hi = 0, len(g)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if has(mid - 1):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _merge(a: tuple, b: tuple, sign: int = 1) -> tuple:
    """Two sparse count tables (sorted keys, counts) added (``sign`` -1: ``b`` taken off, never below 0)."""
    k = np.concatenate([a[0], b[0]])
    c = np.concatenate([a[1], sign * np.asarray(b[1], np.int64)])
    if not len(k):
        return k.astype(np.uint64), c.astype(np.int64)
    u, inv = np.unique(k, return_inverse=True)
    s = np.bincount(inv, weights=c).astype(np.int64)
    keep = s > 0
    return u[keep], s[keep]


_EMPTY = (np.zeros(0, np.uint64), np.zeros(0, np.int64))


class Acc:
    """Counts gathered in memory (a rebuild, one trace, a session), per layer, compacted as they grow."""

    def __init__(self):
        self.E = 0
        self.N = 0
        self.rows: list[int] = []
        self.ids: list[str] = []
        self.seen = np.zeros(0, np.uint64)
        self._k: list[list] = []
        self._q: list[list] = []
        self._kd: list[tuple] = []
        self._qd: list[tuple] = []
        self._pending = 0

    @property
    def L(self) -> int:
        return len(self.rows)

    def _grow(self, L: int):
        while len(self.rows) < L:
            self.rows.append(0)
            self._k.append([])
            self._q.append([])
            self._kd.append(_EMPTY)
            self._qd.append(_EMPTY)

    def add_trace(self, path: Path, trace_id: str, reply_only: bool = False) -> bool:
        """One saved trace's tokens not already counted (``seen``; ``reply_only``: its reply alone): every MoE layer's
        K_TOP of each. False when it has no routing. Raises when its files do not read."""
        meta = json.loads((path / "meta.json").read_text())
        tok = np.asarray(meta["tokens"], np.int64)
        P, n = int(meta["prompt_len"]), len(tok)
        g = None if reply_only else prefix_hashes(tok)
        start = P if reply_only else max(min(_seen_prefix(g, self.seen), P), int(meta.get("read_from") or 0))
        with np.load(path / "arrays.npz") as z:
            layers = sorted(int(k.split("/")[1]) for k in z.files if k.startswith("router_ids/"))
            if not layers:
                return False
            self._grow(len(layers))
            E = int(meta.get("num_experts") or 0)
            for li, l in enumerate(layers):
                ids = z[f"router_ids/{l}"]
                T = ids.shape[0]
                off = n - T                                   # records are right-aligned to the tokens
                pos = np.arange(max(start, off, 0), n)
                rec = pos - off
                ids = ids[rec].astype(np.int64)
                w = (np.nan_to_num(z[f"router_probs/{l}"][rec].astype(np.float32)) if f"router_probs/{l}" in z.files
                     else np.ones(ids.shape, np.float32))
                w = np.where(ids < 0, -1.0, w)
                top = np.argsort(-w, axis=1, kind="stable")[:, :K_TOP]
                sel = np.take_along_axis(ids, top, 1)
                ok = (np.take_along_axis(w, top, 1) >= 0).all(1) & (tok[pos] < (1 << SHIFT))
                sel, tk = sel[ok], tok[pos][ok]
                if len(sel):
                    E = max(E, int(sel.max()) + 1)
                self._k[li].append((sel << SHIFT | tk[:, None]).ravel().astype(np.uint64))
                s = np.sort(sel, 1)
                pairs = [(s[:, a], s[:, b]) for a in range(s.shape[1]) for b in range(a + 1, s.shape[1])]
                self._q[li].append(np.concatenate([a << 32 | b for a, b in pairs]).astype(np.uint64) if pairs
                                   else np.zeros(0, np.uint64))
                self.rows[li] += int(ok.sum())
                self._pending += len(sel)
                if self._pending > COMPACT_ROWS:              # a layer at a time: a long trace's keys never all at once
                    self.compact()
        self.E = max(self.E, E)
        self.N += max(0, n - start)
        if not reply_only:                                   # a session's replies never read it
            # 8 bytes a token position counted (the tally's N): 80 MB at 10M tokens. Not capped: a prefix dropped
            # from it would be counted again when a later chat repeats it.
            self.seen = np.union1d(self.seen, g)
        self.ids.append(trace_id)
        return True

    def compact(self):
        for li in range(self.L):
            for raw, done in ((self._k, self._kd), (self._q, self._qd)):
                if raw[li]:
                    u, c = np.unique(np.concatenate(raw[li]), return_counts=True)
                    done[li] = _merge(done[li], (u, c))
                    raw[li] = []
        self._pending = 0

    def layer(self, li: int) -> tuple:
        self.compact()
        return self._kd[li]

    def cofire(self, li: int) -> tuple:
        """Co-fire keys a << 32 | b (a < b) and counts."""
        self.compact()
        return self._qd[li]


# -- the stored file --------------------------------------------------------------------------------------------------


class Stored:
    """A tally file, read a layer at a time."""

    def __init__(self, path: Path):
        self.path = path
        self._z = np.load(path)
        if int(self._z["version"]) != VERSION:
            raise ValueError(f"{path.name}: version {int(self._z['version'])}, not {VERSION}")
        self.E, self.N = int(self._z["E"]), int(self._z["N"])
        self.rows = self._z["rows"].astype(np.int64)
        self.L = len(self.rows)

    @functools.cached_property
    def ids(self) -> set:
        return set(self._z["ids"].tolist())

    @property
    def seen(self) -> np.ndarray:
        return np.cumsum(self._z["seen"], dtype=np.uint64)

    def layer(self, li: int) -> tuple:
        return np.cumsum(self._z[f"k{li}"], dtype=np.uint64), self._z[f"c{li}"].astype(np.int64)

    def cofire(self, li: int) -> tuple:
        return np.cumsum(self._z[f"q{li}"], dtype=np.uint64), self._z[f"d{li}"].astype(np.int64)


def _diff(k: np.ndarray) -> np.ndarray:
    return np.diff(k, prepend=np.uint64(0)).astype(np.uint64) if len(k) else k.astype(np.uint64)


def _save(path: Path, acc: Acc, old: Optional[Stored]) -> None:
    """``acc`` added to ``old`` (None: the file is ``acc`` alone), written to ``path``."""
    E = max(acc.E, old.E if old else 0)
    L = max(acc.L, old.L if old else 0)
    rows = np.zeros(L, np.int64)
    rows[:acc.L] += acc.rows
    if old:
        rows[:old.L] += old.rows

    def parts():
        yield "version", np.array(VERSION)
        yield "E", np.array(E)
        yield "N", np.array(acc.N + (old.N if old else 0))
        yield "rows", rows
        yield "ids", np.array(sorted(set(acc.ids) | (old.ids if old else set())), dtype=str)
        yield "seen", _diff(np.union1d(acc.seen, old.seen) if old else acc.seen)
        for li in range(L):
            k = acc.layer(li) if li < acc.L else _EMPTY
            if old and li < old.L:
                k = _merge(old.layer(li), k)
            yield f"k{li}", _diff(k[0])
            yield f"c{li}", k[1].astype(np.uint32)
            q = acc.cofire(li) if li < acc.L else _EMPTY
            if old and li < old.L:
                ok, oc = old.cofire(li)
                q = _merge((ok, oc), q)
            yield f"q{li}", _diff(q[0])
            yield f"d{li}", q[1].astype(np.uint32)
    write_npz(path, parts())


def _stored_or_none(path: Path) -> Optional[Stored]:
    try:
        return Stored(path) if path.exists() else None
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        log.warning("metalscope: %s does not read; it is counted afresh", path)
        return None


def add(entry: dict) -> bool:
    """Count one saved trace (an index entry) into its checkpoint's tally, unless it is not ``counted`` or already
    counted. True when it was added."""
    if not counted(entry) or not entry.get("model"):
        return False
    path = tally_path(entry["model"])
    with config.file_lock(path.with_suffix(".lock")):
        old = _stored_or_none(path)
        if old is not None and entry["id"] in old.ids:
            return False
        acc = Acc()
        if old is not None:
            acc.seen = old.seen
        if not acc.add_trace(Path(entry["path"]), entry["id"]) or (old is not None and acc.L != old.L):
            return False
        _save(path, acc, old)
    return True


def rebuild(model: str) -> dict:
    """The checkpoint's tally counted afresh from its saved traces, oldest first (``counted`` ones only). Its size."""
    ckpt, of = models.checkpoint_of(model), models.checkpoint_lookup()
    entries = [e for e in reversed(read_index()) if counted(e) and of(e.get("model")) == ckpt]
    path = tally_path(model)
    with config.file_lock(path.with_suffix(".lock")):
        acc = Acc()
        for e in entries:
            try:
                acc.add_trace(Path(e["path"]), e["id"])
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
                log.warning("metalscope: %s not counted: %s", e["id"], exc)
        if acc.ids:
            _save(path, acc, None)
        elif path.exists():
            path.unlink()
    return size(model)


def _clear(path: Path) -> bool:
    """The tally at ``path`` deleted with its lock and any temp file, and its folder when that leaves it empty; nothing
    made where there is none. Whether there was one."""
    tmps = lambda: list(path.parent.glob(f".{path.name}.*.tmp")) if path.parent.is_dir() else []
    lock, had = path.with_suffix(".lock"), path.exists()
    if not had and not lock.exists() and not tmps():
        return False
    with config.file_lock(path.with_suffix(".lock")):
        for x in [path, *tmps()]:
            with contextlib.suppress(OSError):
                x.unlink()
    for x in (lock, path.parent):
        with contextlib.suppress(OSError):
            x.unlink() if x is lock else x.rmdir()
    return had


def clear(model: str) -> bool:
    return _clear(tally_path(model))


def move(served: str, checkpoint: str) -> bool:
    """``model link``: a served name's own tally (counted while it was a checkpoint of its own) added into
    ``checkpoint``'s, then deleted. Whether there was one."""
    src, dst = (models.models_dir() / n / FILE for n in (served, checkpoint))
    if not src.exists():
        return False
    with config.file_lock(dst.with_suffix(".lock")), config.file_lock(src.with_suffix(".lock")):
        mine, old = _stored_or_none(src), _stored_or_none(dst)
        if mine is not None and (old is None or old.L == mine.L):
            _save(dst, mine, old)
    return _clear(src)


def size(model: str) -> dict:
    """The tally's tokens, traces and bytes (all 0 without one)."""
    path = tally_path(model)
    st = _stored_or_none(path)
    if st is None:
        return dict(checkpoint=models.checkpoint_of(model), tokens=0, traces=0, bytes=0, path=str(path))
    return dict(checkpoint=models.checkpoint_of(model), tokens=st.N, traces=len(st.ids), bytes=path.stat().st_size,
                path=str(path))


class Updater:
    """Counts each saved trace into the tally from a thread of its own, while ``busy()`` is false. The server's busy
    check (live.py) sees traced and steered requests only: an update may run beside an untraced chat (measured: a
    2.3 ms gap in its stream). The first failure is logged in full, later ones in one line; none reaches the save."""

    IDLE_WAIT = 0.5

    def __init__(self, busy: Callable[[], bool] = lambda: False):
        self.busy = busy
        self._q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._failed = False
        self._lock = threading.Lock()

    def submit(self, entry: dict) -> None:
        try:
            if not counted(entry):
                return
            catch_up(entry["model"], self)
            self._q.put(dict(entry))
            with self._lock:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, name="metalscope-tally", daemon=True)
                    self._thread.start()
        except Exception:
            log.exception("metalscope: the expert tally skipped %s", entry.get("id"))

    def _run(self):
        while True:
            try:
                entry = self._q.get(timeout=5)
            except queue.Empty:
                with self._lock:                  # a submit from now on starts another thread
                    if self._q.empty():
                        self._thread = None
                        return
                continue
            while self.busy():
                time.sleep(self.IDLE_WAIT)
            try:
                add(entry)
            except Exception:
                if not self._failed:
                    log.exception("metalscope: the expert tally skipped %s", entry.get("id"))
                else:
                    log.warning("metalscope: the expert tally skipped %s", entry.get("id"))
                self._failed = True
            finally:
                self._q.task_done()

    def drain(self, timeout: float = 10.0) -> None:
        """Wait until every entry queued is counted or skipped (tests); not for the thread's idle exit."""
        end = time.time() + timeout
        while time.time() < end and self._q.unfinished_tasks:
            time.sleep(0.01)


UPDATER: Optional[Updater] = None   # the server's (live.py): catch_up queues on it
CATCH_UP = 200
_CAUGHT: set = set()


def catch_up(model: str, on: Optional[Updater] = None) -> None:
    """Once per checkpoint and data home in this process (its first tally read or update): the newest CATCH_UP saved
    traces the tally would count but has not, queued on ``on`` (the server's UPDATER), oldest first. ``--rebuild``
    counts all."""
    key, on = (str(config.home()), models.checkpoint_of(model)), on or UPDATER
    if on is None or key in _CAUGHT:
        return
    _CAUGHT.add(key)
    st, of = stored(model), models.checkpoint_lookup()
    have = st.ids if st else set()
    todo = [e for e in read_index() if counted(e) and e["id"] not in have and of(e.get("model")) == key[1]][:CATCH_UP]
    for e in reversed(todo):
        on.submit(e)


# -- scoring ----------------------------------------------------------------------------------------------------------


@dataclass
class Layer:
    """One layer of one source, scored: its counts (expert, token, count per entry), the score and n per expert, each
    entry's lift, the co-fire matrix, and the ranking."""
    source: str
    E: int
    N: int
    traces: int
    e: np.ndarray
    t: np.ndarray
    c: np.ndarray
    score: np.ndarray
    ne: np.ndarray
    lift: np.ndarray
    co: np.ndarray
    against: Optional[str] = None
    order: list = field(default_factory=list)
    _tops: dict = field(default_factory=dict)

    @property
    def mean_load(self) -> float:
        return K_TOP * self.N / max(self.E, 1)

    def tops(self, x: int, n: int = 10) -> list[tuple[int, float, int]]:
        """Expert ``x``'s tokens by lift (at least MIN_COUNT of each): (token id, lift, count)."""
        if not self._tops:
            idx = np.where(self.c >= MIN_COUNT[self.source])[0]
            idx = idx[np.lexsort((-self.lift[idx], self.e[idx]))]
            ue, first = np.unique(self.e[idx], return_index=True)
            ends = list(first[1:]) + [len(idx)]
            self._tops = {int(a): idx[f:g] for a, f, g in zip(ue, first, ends)}
        return [(int(self.t[i]), float(self.lift[i]), int(self.c[i])) for i in self._tops.get(x, [])[:n]]

    def overlaps(self, x: int) -> list[tuple[int, float, int]]:
        """Same-layer partners of ``x`` by co-fire lift (observed over expected), at least ``min_co`` co-fires:
        (expert, lift, co-fires)."""
        if not (0 <= x < self.E) or self.ne[x] <= 0:
            return []
        row = self.co[x]
        min_co = max(3, int(np.ceil(self.mean_load / 100)))
        b = np.where((row >= min_co) & (self.ne > 0))[0]
        b = b[b != x]
        lift = row[b] * self.N / (self.ne[x] * self.ne[b])
        o = np.lexsort((b, -row[b], -lift))
        return [(int(b[i]), float(lift[i]), int(row[b[i]])) for i in o]



def _score(e, c, E, bg, a):
    ne = np.bincount(e, weights=c, minlength=E)
    p = (c + a * bg) / (ne[e] + a)
    kl = np.bincount(e, weights=p * np.log(p / bg), minlength=E)
    cov = np.bincount(e, weights=bg, minlength=E)
    rest = a * (1 - cov) / (ne + a)
    kl += np.where(rest > 0, rest * np.log(a / (ne + a)), 0)
    kl = np.where(ne >= a, kl, np.nan)
    lift = (c + LIFT_PRIOR) / (ne[e] * bg + LIFT_PRIOR)
    return kl, ne, lift


def _split(k: np.ndarray):
    return (k >> np.uint64(SHIFT)).astype(np.int64), (k & np.uint64((1 << SHIFT) - 1)).astype(np.int64)


def _co_matrix(E: int, keys: np.ndarray, counts: np.ndarray) -> np.ndarray:
    M = np.zeros((E, E), np.int64)
    a, b = (keys >> np.uint64(32)).astype(np.int64), (keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
    ok = (a < E) & (b < E)
    M[a[ok], b[ok]] = counts[ok]
    return M + M.T


def _layer(source, E, N, traces, kc, co, bg_of, a, against=None) -> Layer:
    k, c = kc
    e, t = _split(k)
    c = c.astype(np.float64)
    bg = bg_of(e, t, c)
    score, ne, lift = _score(e, c, E, bg, a) if len(c) else (np.full(E, np.nan), np.zeros(E), np.zeros(0))
    order = [int(x) for x in np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable") if not np.isnan(score[x])]
    return Layer(source, E, N, traces, e, t, c, score, ne, lift, _co_matrix(E, *co), against, order)


def _mix(e, t, c):
    """The layer's own token mix P(t), per entry."""
    if not len(t):
        return np.zeros(0)
    pt = np.bincount(t, weights=c)
    return pt[t] / pt.sum()


def score_all(st: Stored, li: int) -> Layer:
    """Layer ``li`` of the stored tally, each expert against the layer's mix."""
    N = int(st.rows[li]) if li < st.L else 0
    return _layer("all", st.E, N, len(st.ids), st.layer(li), st.cofire(li), _mix, floor_of(N, st.E))


def _sum(parts: list) -> tuple:
    """Sparse count tables added."""
    if not parts:
        return _EMPTY
    u, inv = np.unique(np.concatenate([k for k, _ in parts]), return_inverse=True)
    return u, np.bincount(inv, weights=np.concatenate([c for _, c in parts]).astype(np.float64)).astype(np.int64)


def _joined(accs: list) -> Acc:
    """One Acc with the counts of ``accs`` (None skipped), none of them changed."""
    out = Acc()
    for x in filter(None, accs):
        out._grow(x.L)
        out.E, out.N = max(out.E, x.E), out.N + x.N
        for li in range(x.L):
            out.rows[li] += x.rows[li]
            out._kd[li], out._qd[li] = _merge(out._kd[li], x.layer(li)), _merge(out._qd[li], x.cofire(li))
    return out


@dataclass
class Session:
    """One Pi client's replies of a checkpoint, each counted once: the ones the stored tally holds merged in one Acc
    (``held``, their ids ``held_ids``: what the reference takes out of the tally), the newest others a reply each
    (``fresh``, trace id -> its counts), older others merged (``apart``, ids ``apart_ids``: never taken out), and every
    trace id read (``read``, those that did not read included). Kept a reply each, the counts grew with every reply of
    a long session; merged, they grow with the (expert, token) pairs it meets."""
    held: Optional[Acc] = None
    fresh: dict = field(default_factory=dict)
    read: set = field(default_factory=set)
    apart: Optional[Acc] = None
    held_ids: set = field(default_factory=set)
    apart_ids: set = field(default_factory=set)

    @property
    def parts(self) -> list:
        return [a for a in (self.held, self.apart, *self.fresh.values()) if a is not None]


def _reference(held: Optional[Acc], li: int, st: Stored) -> tuple:
    """The stored tally's layer ``li`` with the session's replies it holds (``held``) taken out: keys, counts, tokens."""
    x = held if held is not None and li < held.L else None
    rk, rc = _merge(st.layer(li), x.layer(li) if x else _EMPTY, -1)
    return rk, rc, max(int(st.rows[li]) - (x.rows[li] if x else 0), 1)


def score_session(s: Session, li: int, st: Optional[Stored]) -> Layer:
    """Layer ``li`` of a session's replies (``session``), each expert against its own usual mix in the stored tally
    ``st`` with the session's replies it holds taken out (one not yet tallied is not); without one, against the
    session's own layer mix (``against`` "session")."""
    mine = [a for a in s.parts if li < a.L]
    N, E = sum(a.rows[li] for a in mine), max([a.E for a in mine] + [st.E if st else 0])
    kc, co = _sum([a.layer(li) for a in mine]), _sum([a.cofire(li) for a in mine])
    a = floor_of(N, E)
    if st is None or li >= st.L:
        return _layer("session", E, N, len(s.read), kc, co, _mix, a, "session")
    rk, rc, n_ref = _reference(s.held, li, st)
    re_, rt = _split(rk)
    g = floor_of(n_ref, E)

    def own(e, t, c):
        V = int(max(rt.max() if len(rt) else 0, t.max() if len(t) else 0)) + 1
        tf = np.bincount(rt, weights=rc, minlength=V)
        vocab = len(np.union1d(np.unique(rt), np.unique(t)))
        pbg = (tf + 0.5) / (tf.sum() + 0.5 * vocab)
        nre = np.bincount(re_, weights=rc, minlength=E)
        k = (e.astype(np.uint64) << np.uint64(SHIFT)) | t.astype(np.uint64)
        j = np.clip(np.searchsorted(rk, k), 0, max(len(rk) - 1, 0))
        cref = np.where(rk[j] == k, rc[j], 0.0) if len(rk) else np.zeros(len(k))
        return (cref + g * pbg[t]) / (nre[e] + g)
    return _layer("session", E, N, len(s.read), kc, co, own, a, "tally")


# -- what the routes read ---------------------------------------------------------------------------------------------

_STORED: "OrderedDict[tuple, Stored]" = OrderedDict()
_SCORED: "OrderedDict[tuple, Layer]" = OrderedDict()
_SESSIONS: "OrderedDict[tuple, Session]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_SESSION_LOCK = threading.Lock()


def _keep(d: OrderedDict, key, make, n: int):
    with _CACHE_LOCK:
        if key in d:
            d.move_to_end(key)
            return d[key]
    v = make()
    with _CACHE_LOCK:
        d[key] = v
        while len(d) > n:
            d.popitem(last=False)
    return v


def stored(model: str) -> Optional[Stored]:
    """The model's checkpoint's tally as it is on disk now (None without one), kept while the file is unchanged."""
    path = tally_path(model)
    try:
        s = path.stat()
    except OSError:
        return None
    return _keep(_STORED, (str(path), s.st_ino, s.st_mtime_ns, s.st_size), lambda: _stored_or_none(path), 2)


def session(model: str, client: str, st: Optional[Stored] = None) -> Session:
    """One Pi client's saved traces of the model's checkpoint (``counted`` ones), each reply counted in memory once,
    new ones added on each call; those the stored tally ``st`` holds merged into ``held``, all but the newest
    SESSION_FRESH of the others into ``apart``. A copy to score: a merge makes a new Acc, never changes one a copy
    has. When the tally no longer holds a reply in ``held`` (a clear, a rebuild, a pruned trace) or now holds one in
    ``apart`` (the catch-up counted it), the session is counted again, so the tally's replies are always the ones
    taken out."""
    key = (str(config.home()), models.checkpoint_of(model), client)
    have = st.ids if st is not None else set()
    of = models.checkpoint_lookup()
    with _SESSION_LOCK:                                                   # one request adds a trace at a time
        s = _keep(_SESSIONS, key, Session, 4)
        if st is not None and (s.held_ids - have or s.apart_ids & have):
            s = Session()
            with _CACHE_LOCK:
                _SESSIONS[key] = s
        for e in reversed(read_index()):
            if e.get("client") != client or not counted(e) or e["id"] in s.read or of(e.get("model")) != key[1]:
                continue
            s.read.add(e["id"])                                           # not asked again, read or not
            acc = Acc()
            try:
                if acc.add_trace(Path(e["path"]), e["id"], reply_only=True):
                    acc.compact()
                    s.fresh[e["id"]] = acc
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
                log.warning("metalscope: %s not in the session's tally: %s", e["id"], exc)
        done = [i for i in s.fresh if i in have]
        old = [i for i in s.fresh if i not in have][:max(0, len(s.fresh) - len(done) - SESSION_FRESH)]
        for ids, acc, merged in ((done, "held", s.held_ids), (old, "apart", s.apart_ids)):
            if ids:
                setattr(s, acc, _joined([getattr(s, acc), *(s.fresh.pop(i) for i in ids)]))
                merged.update(ids)
        return Session(s.held, dict(s.fresh), set(s.read), s.apart, set(s.held_ids), set(s.apart_ids))


def layer_view(model: str, li: int, client: Optional[str] = None) -> Optional[Layer]:
    """Layer ``li`` scored: ``all`` (the stored tally) without ``client``, else that client's session. None when
    there is nothing to score (no tally, an empty session)."""
    if li < 0:
        raise ValueError("layer must be 0 or more")
    catch_up(model)
    st = stored(model)
    if client:
        s = session(model, client, st)
        L = max([a.L for a in s.parts], default=0)
        if not L:
            return None
        if li >= L:
            raise ValueError(f"layer must be 0 to {L - 1}")
        key = ("session", str(config.home()), models.checkpoint_of(model), client, li, len(s.read), tuple(s.fresh),
               len(s.held_ids), st.path if st else None, len(st.ids) if st else 0, st.N if st else 0)
        return _keep(_SCORED, key, lambda: score_session(s, li, st), 4)
    if st is None:
        return None
    if li >= st.L:
        raise ValueError(f"layer must be 0 to {st.L - 1}")
    return _keep(_SCORED, ("all", str(st.path), st.N, len(st.ids), li), lambda: score_all(st, li), 4)

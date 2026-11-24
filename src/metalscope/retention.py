"""Trace retention: the saved traces are kept under a size budget (10 GB unless ``metalscope traces keep SIZE``).

What goes first when the total passes the budget:

    1. automated runs (store.is_automated: calibrate-*, ladder-*, bench ...), oldest first
    2. chats (Pi's pi-* clients, and requests with no client), oldest first

Never removed:

    pinned traces            metalscope traces pin ID (kept in pinned.json)
    the strength table's     the newest store.NORM_TRACES unsteered traces of each served model: what
                             steer.norm_table would measure dose 1.0 on next (store.norm_picks)
    traces being written     a trace joins the index only once its folder is complete; prune reads the index.
                             A save cut short leaves a dot folder (store.PART): prune removes one a day old
    traces the index lost    a <day>/<trace id> folder with a meta.json that no line names (a save stopped before
                             its line, a line lost or damaged): prune gives it its line once a day old, and it is
                             pruned in its turn; anything else under traces/ is left alone
    folders outside traces/  an index line may point anywhere; only folders under the traces directory are deleted

Each index entry records its folder's size when saved (``bytes``). The running total (``Usage``) reads only what was
appended to the index since it last looked, so the check after each save costs a few hundred bytes of reading.
Servers check from a thread of their own (``Keeper``), never on a request or on oMLX's scheduler thread, and prune
only when the total is over the budget.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from . import config
from .models import NAME_RE
from .store import PART, _entry, append_index, index_entry, is_automated, named, norm_picks, rewrite_index

log = logging.getLogger("metalscope")
GB = 1000 ** 3
DEFAULT_BUDGET = 10 * GB
CUT_SHORT_AGE = 86400   # a save's dot folder untouched this long was cut short (a save takes a minute at most)
SLACK = 0.9          # the automatic prune goes down to this share of the budget: one rewrite, not one per save
_UNITS = {"": GB, "k": 1000, "kb": 1000, "m": 1000 ** 2, "mb": 1000 ** 2, "g": GB, "gb": GB, "t": 1000 ** 4, "tb": 1000 ** 4}


# -- the budget ------------------------------------------------------------
def parse_size(text: str) -> int:
    """``20GB``, ``500MB``, ``1.5TB`` or a bare number of GB, in bytes (1 GB = 10^9 bytes, as macOS counts)."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", str(text))
    unit = m and m.group(2).lower()
    if not m or unit not in _UNITS:
        raise ValueError(f"not a size: {text!r} (e.g. 20GB, 500MB, 1TB)")
    n = int(float(m.group(1)) * _UNITS[unit])
    if n <= 0:
        raise ValueError("the budget must be more than 0 (metalscope off stops tracing)")
    return n


def parse_age(text: str) -> float:
    """An age such as 7d, 2w or 12h, in seconds (a bare number: days). ValueError otherwise."""
    t = text.strip().lower()
    unit = {"h": 3600, "d": 86400, "w": 7 * 86400}.get(t[-1:]) if t else None
    try:
        n = float(t[:-1] if unit else t)
    except ValueError:
        raise ValueError(f"{text!r} is not an age: write it like 7d, 2w or 12h") from None
    if not math.isfinite(n) or n <= 0:
        raise ValueError(f"{text!r} is not an age: it must be a number more than 0")
    return n * (unit or 86400)


def fmt_size(n: float) -> str:
    return f"{n / GB:,.1f} GB" if n >= GB / 10 else f"{n / 1000 ** 2:,.1f} MB"


def budget(settings: Optional[dict] = None) -> int:
    """The traces' size budget in bytes (settings.json ``trace_budget``)."""
    n = (settings if settings is not None else config.load_settings()).get("trace_budget")
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else DEFAULT_BUDGET


def auto(settings: Optional[dict] = None) -> bool:
    """Whether the server keeps the traces under the budget by itself (settings.json ``trace_auto``, on unless
    turned off with ``metalscope traces auto off``). Off, nothing is removed until ``traces prune``."""
    return (settings if settings is not None else config.load_settings()).get("trace_auto", True) is not False


# -- pins ------------------------------------------------------------------
def pins_path() -> Path:
    return config.home() / "pinned.json"


def pinned() -> set[str]:
    ids = config.read_json(pins_path(), [])
    return {str(i) for i in ids} if isinstance(ids, list) else set()


def pin(trace_id: str, on: bool = True) -> set[str]:
    """Pin a trace (never pruned), or unpin it with ``on=False``; the pins after."""
    ids = pinned() | {trace_id} if on else pinned() - {trace_id}
    config.atomic_write(pins_path(), json.dumps(sorted(ids), indent=1))
    return ids


# -- sizes -----------------------------------------------------------------
def folder_bytes(path) -> int:
    """The bytes of the files under ``path`` (0 when it is gone)."""
    total, stack = 0, [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for d in it:
                    try:
                        if d.is_dir(follow_symlinks=False):
                            stack.append(d.path)
                        elif d.is_file(follow_symlinks=False):
                            total += d.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def size_of(e: dict) -> int:
    """An index entry's recorded size (``bytes``), 0 when not recorded."""
    n = e.get("bytes")
    return n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else 0


class Usage:
    """The running total of the index: bytes and traces. ``refresh`` reads only the lines appended since the last
    call (anyone's: both servers write the same index); a rewritten index (a prune) is read again from the start."""

    def __init__(self):
        self._key = None
        self._offset = 0
        self.bytes = 0
        self.count = 0

    def refresh(self) -> "Usage":
        path = config.index_path()
        try:
            st = path.stat()
        except OSError:
            self._key, self._offset, self.bytes, self.count = None, 0, 0, 0
            return self
        key = (str(path), st.st_ino)
        if key != self._key or st.st_size < self._offset:
            self._key, self._offset, self.bytes, self.count = key, 0, 0, 0
        if st.st_size == self._offset:
            return self
        try:
            with open(path, "rb") as f:
                f.seek(self._offset)
                data = f.read()
        except OSError:
            return self
        cut = data.rfind(b"\n") + 1                       # a line still being written is read next time
        for e in map(_entry, data[:cut].decode(errors="replace").splitlines()):
            if e and e.get("path"):
                self.bytes += size_of(e)
                self.count += 1
        self._offset += cut
        return self


# -- pruning ---------------------------------------------------------------
@dataclasses.dataclass
class Result:
    budget: int
    before_bytes: int = 0
    before_count: int = 0
    removed: list = dataclasses.field(default_factory=list)     # the entries removed (or that would be), in order
    freed: int = 0
    auto_count: int = 0
    auto_bytes: int = 0
    chat_count: int = 0
    chat_bytes: int = 0
    protected: dict = dataclasses.field(default_factory=dict)   # why -> traces kept for that reason
    stale: int = 0                                              # index lines whose folder was gone (dropped)
    dry_run: bool = False
    before: Optional[float] = None                              # a cutoff instead of the budget (inf: everything)
    adopted: int = 0                                            # folders the index lost, given their line first

    @property
    def after_bytes(self) -> int:
        return self.before_bytes - self.freed

    @property
    def after_count(self) -> int:
        return self.before_count - len(self.removed)


def plan(entries: list[dict], limit: int, pins: Optional[set] = None,
         before: Optional[float] = None) -> tuple[Result, Optional[list]]:
    """What a prune down to ``limit`` bytes removes from ``entries`` (index lines, any order): the result, and the
    entries to keep (stale lines dropped), or None when the index would not change. ``before`` (a
    time; inf for all): every trace saved before it goes instead, whatever the budget. The protected stay either way.
    Decides only; deletes nothing."""
    pins = pinned() if pins is None else pins
    res = Result(budget=limit, before=before)
    root = config.traces_dir().resolve()
    live = []
    for e in entries:
        p = e.get("path")
        if not p or not Path(p).is_dir():
            res.stale += 1
            continue
        live.append(e)
    res.before_count, res.before_bytes = len(live), sum(map(size_of, live))
    why: dict[int, str] = {}
    newest = sorted(live, key=lambda e: e.get("time", 0), reverse=True)        # read_index's order
    for model in {config.short_name(e.get("model")) for e in live}:
        for e in norm_picks(newest, model):
            why[id(e)] = "the strength table's newest unsteered"
    for e in live:
        if str(e.get("id")) in pins:
            why[id(e)] = "pinned"
    candidates = sorted((e for e in live if id(e) not in why),
                        key=lambda e: (not is_automated(e.get("client")), e.get("time", 0)))
    total, gone = res.before_bytes, set()
    for e in candidates:
        if before is None and total <= limit:
            break
        if before is not None and e.get("time", 0) >= before:
            continue
        if not _inside(e["path"], root):         # checked only for what would go: resolving every path is slow
            why[id(e)] = "outside the traces folder"
            continue
        n = size_of(e)
        total -= n
        gone.add(id(e))
        res.removed.append(e)
        res.freed += n
        if is_automated(e.get("client")):
            res.auto_count, res.auto_bytes = res.auto_count + 1, res.auto_bytes + n
        else:
            res.chat_count, res.chat_bytes = res.chat_count + 1, res.chat_bytes + n
    for reason in why.values():
        res.protected[reason] = res.protected.get(reason, 0) + 1
    if not (gone or res.stale):
        return res, None
    return res, [e for e in live if id(e) not in gone]


def _inside(path, root: Path) -> bool:
    try:
        p = Path(path).resolve()
    except OSError:
        return False
    return p != root and p.is_relative_to(root)


def prune(limit: Optional[int] = None, dry_run: bool = False, before: Optional[float] = None) -> Result:
    """Remove traces until the total is at most ``limit`` (default: the budget), or with ``before`` every trace
    saved before that time (inf: all of them), the protected kept either way. The index is rewritten first
    (atomically, under its lock), then the folders deleted. ``dry_run``: decide and report, change nothing."""
    limit = budget() if limit is None else limit
    root = config.traces_dir().resolve()
    if not dry_run:
        _remove_cut_short(root)
    try:
        raw = [e for e in map(_entry, config.index_path().read_text().splitlines()) if e]
    except OSError:
        raw = []
    lost = _unindexed(root, raw)
    pins = pinned()
    if dry_run:
        return dataclasses.replace(plan(raw + lost, limit, pins, before)[0], dry_run=True, adopted=len(lost))
    adopted = len(append_index(*lost, new_only=True)) if lost else 0     # read again under the lock: no second line
    out: list = []

    def keep(entries):
        res, kept = plan(entries, limit, pins, before)
        out.append((res, {e["path"] for e in kept or []}))
        return kept
    rewrite_index(keep)
    if not out:
        return Result(budget=limit, before=before, adopted=adopted)
    res, still = out[0]
    res.adopted = adopted
    for e in res.removed:
        # Kept by another line naming the same folder; or not under traces/ now (plan checked it under the lock: a
        # folder swapped for a link since must not take a folder elsewhere with it).
        if e["path"] in still or not _inside(e["path"], root):
            continue
        shutil.rmtree(e["path"], ignore_errors=True)
        parent = Path(e["path"]).parent
        if _inside(parent, root):
            try:
                parent.rmdir()          # the day's folder, once its last trace is gone
            except OSError:
                pass
        time.sleep(0)                   # let the server's threads in between folders
    return res


def _remove_cut_short(root: Path) -> None:
    """The dot folders of saves cut short (killed, or failed past their own cleanup) under ``root``, none touched in
    the last CUT_SHORT_AGE: no index line names them, so nothing else ever removes them."""
    cutoff = time.time() - CUT_SHORT_AGE
    for d in root.glob(f"*/.*{PART}"):
        try:
            if d.is_symlink() or not _inside(d, root) or \
                    max(x.stat().st_mtime for x in (d, *d.iterdir())) >= cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(d, ignore_errors=True)


DAY = re.compile(r"\d{4}-\d{2}-\d{2}")       # a trace's day folder (live.py: traces/<%Y-%m-%d>/<trace id>)


def _unindexed(root: Path, entries: list[dict]) -> list[dict]:
    """The index lines (store.index_entry) of the traces under ``root`` that ``entries`` do not name: a
    ``<day>/<trace id>`` folder, no link, holding a meta.json object, untouched in the last CUT_SHORT_AGE (a save
    under way is newer). Anything else under traces/ is not a trace: never listed, never removed."""
    have, cutoff, out = named(entries), time.time() - CUT_SHORT_AGE, []
    for d in sorted(root.glob("*/*")):
        if (d.parent.name, d.name) in have or not DAY.fullmatch(d.parent.name) or not NAME_RE.fullmatch(d.name):
            continue
        try:
            if d.is_symlink() or not d.is_dir() or not _inside(d, root) or \
                    max(x.stat().st_mtime for x in (d, *d.iterdir())) >= cutoff:
                continue
            meta = config.read_json(d / "meta.json")
            if isinstance(meta, dict):
                out.append(index_entry(d, meta))
        except OSError:
            continue
    return out


def summary(res: Result) -> list[str]:
    """A prune's result in plain lines, one fact each."""
    verb = "would remove" if res.dry_run else "removed"
    lines = [f"before: {res.before_count:,} traces, {fmt_size(res.before_bytes)}",
             f"budget: {fmt_size(res.budget)}" if res.before is None else
             "removing: every trace" if res.before == math.inf else
             f"removing: every trace saved before {time.strftime('%Y-%m-%d %H:%M', time.localtime(res.before))}"]
    if not res.removed:
        lines.append("nothing to remove: under the budget" if res.before is None and res.before_bytes <= res.budget else
                     "nothing to remove: what is left is protected" if res.before is None else
                     "nothing to remove: nothing older, or only protected traces")
    else:
        lines.append(f"{verb}: {len(res.removed):,} traces, {fmt_size(res.freed)}")
        if res.auto_count:
            lines.append(f"  automated runs: {res.auto_count:,}, {fmt_size(res.auto_bytes)}")
        if res.chat_count:
            lines.append(f"  chats: {res.chat_count:,}, {fmt_size(res.chat_bytes)}")
    lines.append(f"{'would keep' if res.dry_run else 'kept'}: {res.after_count:,} traces, {fmt_size(res.after_bytes)}")
    for reason, n in sorted(res.protected.items()):
        lines.append(f"protected ({reason}): {n:,}")
    if res.before is None and res.after_bytes > res.budget:
        lines.append(f"still over the budget: the protected traces alone take {fmt_size(res.after_bytes)}")
    if res.adopted:
        lines.append(f"{'would adopt' if res.dry_run else 'adopted'}: {res.adopted:,} trace folders the index did not list")
    if res.stale:
        lines.append(f"index lines for folders already gone: {res.stale:,} ({'would be dropped' if res.dry_run else 'dropped'})")
    return lines


class Keeper:
    """Keeps the traces under the budget from a thread of its own. ``poke`` (the trace writer calls it after each
    save) only queues a check when none is waiting; the check reads what was appended to the index and prunes only
    when the total is over the budget."""

    def __init__(self):
        self.usage = Usage()
        self.last: Optional[Result] = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metalscope-prune")
        self._lock = threading.Lock()
        self._waiting = False

    def poke(self) -> None:
        with self._lock:
            if self._waiting:
                return
            self._waiting = True
        self._pool.submit(self._check)

    def _check(self) -> None:
        with self._lock:
            self._waiting = False          # a save from now on queues another check
        try:
            settings = config.load_settings()
            limit = budget(settings)
            if not auto(settings) or self.usage.refresh().bytes <= limit:
                return
            res = self.last = prune(int(limit * SLACK))
            if res.removed or res.stale:
                log.warning("metalscope: traces over the %s budget (%s, %d traces): removed %d (%s freed; %d "
                            "automated, %d chats), kept %d (%s)", fmt_size(limit), fmt_size(res.before_bytes),
                            res.before_count, len(res.removed), fmt_size(res.freed), res.auto_count, res.chat_count,
                            res.after_count, fmt_size(res.after_bytes))
        except Exception:
            log.exception("metalscope: keeping the traces under the budget failed")


def status(week: bool = False) -> dict:
    """The traces' total, count, budget and pins (``metalscope traces``); ``week``: also ``old_bytes`` and
    ``old_count``, what ``prune --older-than 7d`` would remove (doctor)."""
    u = Usage().refresh()
    out = dict(bytes=u.bytes, count=u.count, budget=budget(), auto=auto(), pinned=len(pinned()))
    if week:
        res = prune(dry_run=True, before=time.time() - 7 * 86400)
        out.update(old_bytes=res.freed, old_count=len(res.removed))
    return out

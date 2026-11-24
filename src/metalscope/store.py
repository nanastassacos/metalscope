"""Traces on disk: one directory per request.

    <dir>/meta.json     tokens, decoded token strings, prompt length, model, config; read_from: the first
                        position the request fed (0 unless it reused an earlier request's prompt prefix);
                        capped: the tokens kept when the trace reached its cap (live.TRACE_MAX_BYTES)
    <dir>/arrays.npz    "<kind>/<layer>" -> array with the token axis concatenated
                        across steps (prefill contributes T_prompt, decode 1 each); rows before
                        read_from are empty (-1 / NaN)

Plots read these and never touch the model.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import config

# read_index's last parse: (the index file's path, inode, size, mtime) -> its entries. Every trace route and the
# strength table read the index; with thousands of entries, parsing it and checking each folder on every call was
# the slow part, so it is done again only when the file changed (a save appends, a prune rewrites it).
_parsed: tuple = (None, [])


def read_index() -> list[dict]:
    """Saved traces (index.jsonl entries whose directory existed when the index last changed), newest first.

    Entries store absolute paths. A folder deleted by hand stays listed until the index changes again."""
    global _parsed
    path = config.index_path()
    try:
        st = path.stat()
    except OSError:
        return []
    key = (str(path), st.st_ino, st.st_size, st.st_mtime_ns)
    if _parsed[0] != key:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return []
        out = [e for e in map(_entry, lines) if e and e.get("id") and e.get("path") and Path(e["path"]).exists()]
        _parsed = (key, sorted(out, key=lambda e: e.get("time", 0), reverse=True))
    return [dict(e) for e in _parsed[1]]       # copies: a caller's change never reaches the next one


def index_entry(path: Path, meta: dict) -> dict:
    """A saved trace's index line, from its folder and its meta.json's fields (what Trace.save writes there): what a
    save appends, and what retention.prune adopts for a folder no line names."""
    from .retention import folder_bytes
    st, started = meta.get("steer"), meta.get("started")
    try:
        with zipfile.ZipFile(path / "arrays.npz") as z:
            experts = any(n.startswith("router_ids/") for n in z.namelist())    # whether it routed (the expert tally)
    except (OSError, zipfile.BadZipFile):
        experts = False
    return dict(id=path.name, path=str(path),
                time=started if isinstance(started, (int, float)) else (path / "meta.json").stat().st_mtime,
                model=meta.get("model"), level=meta.get("level"), conversation=meta.get("conversation"),
                turn=meta.get("turn"), client=meta.get("client"), title=meta.get("title"),
                prompt_tokens=meta.get("prompt_len"), generated_tokens=meta.get("generated_tokens"),
                steer=f"{st.get('label')}{' FAILED' if st.get('failed') else ''}{' PARTIAL' if st.get('partial') else ''}"
                if isinstance(st, dict) else None,
                lens_step=meta.get("lens_step"), decode_tps=meta.get("decode_tps"), experts=experts,
                bytes=folder_bytes(path))       # the running total against the budget reads it


def named(entries: list[dict]) -> set[tuple[str, ...]]:
    """The ``(<day>, <trace id>)`` folders the index ``entries`` name, wherever their paths say the data home is."""
    return {Path(e["path"]).parts[-2:] for e in entries if isinstance(e.get("path"), str)}


def append_index(*entries: dict, new_only: bool = False) -> list[dict]:
    """Add saved traces' lines to the index, under its lock; ``new_only``: only those whose folder no line names yet
    (``named``, read under the lock). The lines added."""
    path = config.index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with config.file_lock(config.home() / "index.lock"), open(path, "a+b") as f:
        if new_only:
            f.seek(0)
            have = named([e for e in map(_entry, f.read().decode(errors="replace").splitlines()) if e])
            entries = tuple(e for e in entries if Path(e["path"]).parts[-2:] not in have)
        if f.seek(0, os.SEEK_END):
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")             # a line cut short by a crash: this one starts on its own
        f.write(b"".join(json.dumps(e).encode() + b"\n" for e in entries))
    return list(entries)


def rewrite_index(keep: Callable[[list[dict]], Optional[list[dict]]]) -> None:
    """Replace the index with ``keep(entries)``, atomically (a temp file renamed over it) and under the lock.

    ``keep`` gets every complete entry and returns the ones to write (None: leave the file as it is); it runs under
    the lock, so an append waits for it."""
    path = config.index_path()
    with config.file_lock(config.home() / "index.lock"):
        try:
            old = open(path, "rb")
        except FileNotFoundError:
            return
        with old:
            entries = [e for e in map(_entry, old.read().decode(errors="replace").splitlines()) if e]
            kept = keep(entries)
            if kept is None:
                return
            text = "".join(json.dumps(e) + "\n" for e in kept)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(text)
                os.chmod(tmp, os.fstat(old.fileno()).st_mode & 0o777)
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise


def write_npz(path: Path, parts) -> None:
    """``parts`` (name, array) pairs into an npz at ``path``, through a temp file and a rename (a reader sees the old
    file or the new one); written one at a time, so a caller can make each array only when it is its turn. zlib level
    1: about twice numpy's default speed for a few percent more disk, so a long trace's save ends while the dock waits
    for it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f, zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for name, a in parts:
                with zf.open(name + ".npy", "w", force_zip64=True) as m:
                    np.lib.format.write_array(m, np.asanyarray(a), allow_pickle=False)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


NORM_TRACES = 20          # traces the strength table is measured on (steer.norm_table)


def norm_picks(entries: list[dict], model: str, n: int = NORM_TRACES) -> list[dict]:
    """The traces served copy ``model``'s strength table is measured on (steer.norm_table), which retention never
    removes: the first ``n`` of ``entries`` (newest first) that are unsteered and of that served copy."""
    short = config.short_name(model)
    return [e for e in entries if not e.get("steer") and config.short_name(e.get("model")) == short][:n]


PART = ".part"      # Trace.save writes trace folder NAME as .NAME.part, then renames it


def is_automated(client: str | None) -> bool:
    """A session's client is an automated run (calibrate-NAME, ladder-*, bench, live-check ...), not a chat: anything
    but Pi's ``pi-<id>`` or no client at all."""
    return bool(client) and not client.startswith("pi-")


def _entry(line: str):
    try:
        e = json.loads(line)
    except ValueError:
        return None                      # a line cut short by a crash mid-append
    return e if isinstance(e, dict) else None


@dataclass
class Trace:
    arrays: dict[str, np.ndarray]
    tokens: list[int]
    token_text: list[str]
    prompt_len: int
    meta: dict[str, Any] = field(default_factory=dict)

    # -- access ----------------------------------------------------------
    @property
    def read_from(self) -> int:
        """The first position this request read: the rows before it (a prompt prefix an earlier request read, whose
        state ds4 reused) have their tokens but no records (-1 / NaN). 0 when not recorded."""
        return int(self.meta.get("read_from") or 0)

    def kinds(self) -> set[str]:
        return {k.split("/")[0] for k in self.arrays}

    def layers(self, kind: str) -> list[int]:
        return sorted(int(k.split("/")[1]) for k in self.arrays if k.startswith(kind + "/"))

    def get(self, kind: str, layer: int = -1) -> np.ndarray:
        return self.arrays[f"{kind}/{layer}"]

    def by_layer(self, kind: str) -> np.ndarray:
        """[n_layers, T, ...] for a per-layer record kind."""
        return np.stack([self.get(kind, i) for i in self.layers(kind)])

    # -- io --------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Written into a dot folder beside ``path`` (PART) and renamed into place once whole: a folder under
        its own name is always complete, and a save cut short leaves only the dot folder (retention.prune removes
        old ones)."""
        path = Path(path)
        part = path.parent / f".{path.name}{PART}"
        part.mkdir(parents=True, exist_ok=True)
        try:
            write_npz(part / "arrays.npz", self.arrays.items())
            meta = dict(self.meta, tokens=self.tokens, token_text=self.token_text,
                        prompt_len=self.prompt_len)
            (part / "meta.json").write_text(json.dumps(meta, indent=1))
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)             # the same name saved again: the newer save replaces it
            os.rename(part, path)
        except BaseException:
            shutil.rmtree(part, ignore_errors=True)
            raise
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        with np.load(path / "arrays.npz") as z:
            arrays = {k: z[k] for k in z.files}
        return cls(arrays, meta.pop("tokens"), meta.pop("token_text"),
                   meta.pop("prompt_len"), meta)

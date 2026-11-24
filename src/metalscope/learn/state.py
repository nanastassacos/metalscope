"""The jobs' state: one directory per job under ``<home>/learn/``, its ``state.json`` rewritten atomically at every
step (job.py), which the server's ``/scope/api/jobs`` serves to Pi and ``metalscope jobs`` prints. A job without
an outcome whose process no longer holds its live lock (``hold``) reads as "interrupted"; one at a time (``busy``).
"""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Optional

from .. import config, gpulock
from ..vectors import NAME_RE
from .decide import CANT_STEER


LAST_LINE = 200                 # a dead job's last printed line, as its error: at most this many characters
LIVE = "live.lock"              # in the job's directory: held while the job runs (hold)
INHERITED = "METALSCOPE_JOB_LOCK"   # the descriptor of the live lock spawn passed to the job's process
SEEN = "seen"                   # in the job's directory: Pi has announced its result (mark_seen)


def span(seconds: float) -> str:
    """A duration for a job's lines: "40 min", "8.0 h"."""
    return f"{seconds / 3600:.1f} h" if seconds >= 5400 else f"{max(1, round(seconds / 60))} min"


def learn_dir() -> Path:
    return config.home() / "learn"


def new_job_id(name: str, kind: str = "", unique: bool = False) -> str:
    """A job's id from a concept's name: ``NAME[-KIND]-YYYYmmdd-HHMMSS``, ``unique``: then ``-`` and 4 hex characters
    (two started in one second differ). The longest, ``NAME-calibrate-...-abcd``, is 31 characters past the name:
    vectors.NAME_MAX keeps it a job id (``job_dir``)."""
    return "-".join([name, *([kind] if kind else []), time.strftime("%Y%m%d-%H%M%S"), *([os.urandom(2).hex()] if unique else [])])


def job_dir(job: str) -> Path:
    """A job's directory; the id arrives in URLs, so it must be a plain name (no dots first, no slashes)."""
    if not NAME_RE.fullmatch(job or ""):
        raise ValueError(f"not a job id: {job!r}")
    return learn_dir() / job


def take(job: str) -> gpulock.GpuLock:
    """Take the job's live lock (``hold``); ValueError when another process runs this job."""
    try:
        return gpulock.GpuLock(gpulock.Holder(f"job {job}"), job_dir(job) / LIVE).acquire(wait=1.0)
    except gpulock.LockBusy:
        raise ValueError(f"job {job} is already running") from None


def hold(job: str) -> gpulock.GpuLock:
    """The job's live lock, which its process holds for as long as it runs: liveness is the kernel's word (released at
    exit, a kill -9 too), not a pid that another process may have reused nor a command line typed in any order. A job
    the server starts adopts the descriptor ``spawn`` took and passed (``INHERITED``) when it is this job's live.lock,
    so the job reads as running from its first state; anything else in that variable is ignored and the lock taken
    here (``take``)."""
    path = job_dir(job) / LIVE
    try:
        fd = int(os.environ.pop(INHERITED))
        if os.path.samestat(os.fstat(fd), os.stat(path)):
            os.set_inheritable(fd, False)     # nothing the job starts may keep it running after it ends
            lock = gpulock.GpuLock(gpulock.Holder(f"job {job}"), path)
            lock.fd = fd
            return lock
    except (KeyError, ValueError, OSError):
        pass
    return take(job)


def _alive(job: str) -> bool:
    return gpulock.held(job_dir(job) / LIVE) is not None


def _last_line(job: str) -> Optional[str]:
    """The last line a job printed (spawn's stdout.txt), without an ``error:`` in front, at most LAST_LINE characters
    and its control characters shown as "·" (it reaches the dock); None without one."""
    from ..share import CONTROL
    try:
        lines = (job_dir(job) / "stdout.txt").read_text(errors="replace").splitlines()
    except OSError:
        return None
    line = next((l.strip().removeprefix("error: ") for l in reversed(lines) if l.strip()), None)
    if line is None:
        return None
    line = CONTROL.sub("·", line)
    return line if len(line) <= LAST_LINE else line[:LAST_LINE - 1] + "…"


def read_state(job: str) -> Optional[dict]:
    """A job's state; a job whose process is gone without an outcome is marked "interrupted", with the last line it
    printed as the error when it died before writing a state of its own (spawn's first one has no steps)."""
    st = config.read_json(job_dir(job) / "state.json")
    if not isinstance(st, dict):         # missing, or a list or a number in the file
        return None
    st["seen"] = (job_dir(job) / SEEN).exists() or bool(st.get("seen"))    # or an older state.json's own flag
    if st.get("outcome") is None and not _alive(job):
        st["outcome"], st["phase"] = "interrupted", "interrupted"
        st["error"] = (st.get("error") or ("steps" not in st and _last_line(job))
                       or "the job stopped without finishing (killed, crashed or the machine restarted)")
    return st


def list_jobs() -> list[dict]:
    d = learn_dir()
    jobs = []
    for p in d.glob("*/state.json") if d.exists() else []:
        try:
            st = read_state(p.parent.name)
        except ValueError:
            continue
        if st:
            jobs.append(st)
    return sorted(jobs, key=lambda s: s.get("started") or 0)


def outcome_text(st: dict) -> str:
    """How a job ended, in a few words: a calibration's one line (calibrate.summary_text, made from its numbers; a
    learn job's calibration too), a learn job that stopped early ("can't be steered"), else the outcome (or the phase
    while it runs)."""
    if st.get("outcome") != "done":
        return st.get("outcome") or f"running ({st.get('phase')})"
    r = st.get("result") or {}
    if r.get("steerable") is False:
        return CANT_STEER
    if "windows" in r:
        h = r.get("heldout") or {}
        return f"{r['windows']} windows" + (f", hit@10 {h['hit10']:.3f} vs logit {h['logit_hit10']:.3f}" if h.get("hit10") is not None else "")
    cal = r.get("calibration") if isinstance(r.get("calibration"), dict) else {}
    if not cal.get("error") and all(isinstance(cal.get(k), (int, float)) for k in ("default_dose", "good", "turns")):
        from .calibrate import summary_text
        return summary_text(cal)
    return cal.get("error") or "done"


def mark_seen(job: str) -> bool:
    """A file of its own, not a field in state.json: the job rewrites that whole from what it holds, which would undo
    the mark."""
    d = job_dir(job)
    if not (d / "state.json").exists():
        return False
    (d / SEEN).touch()
    return True


def busy(exclude: Optional[str] = None) -> Optional[str]:
    """Why no new job can start now (the GPU is held, or a job runs: one at a time, since a learn job unloads the
    model a calibration talks to), else None. ``exclude``: the asking job's own id (its state is already written)."""
    held = gpulock.held()
    if held:
        return f"the GPU is in use: {held.message()}"
    job = running(exclude)
    return running_text(job) if job else None


def running(exclude: Optional[str] = None) -> Optional[dict]:
    """The running job's state (the last started), else None; ``exclude``: a job id not to count."""
    jobs = [j for j in list_jobs() if j.get("outcome") is None and j.get("job") != exclude]
    return jobs[-1] if jobs else None


def running_text(st: dict) -> str:
    return f"a job is running: {doing(st)} (/scope jobs)"


def busy_with(concept: str) -> Optional[str]:
    """Why ``concept``'s data cannot be deleted now (a learn or calibrate job of it running), else None."""
    jobs = [j for j in list_jobs() if j.get("outcome") is None and j.get("kind") in ("learn", "calibrate")
            and j.get("name") == concept]
    return running_text(jobs[-1]) if jobs else None


# What a running job of each kind is doing, for messages ("a job is running: learning crunchy").
DOING = {"learn": "learning", "calibrate": "calibrating", "lens-learn": "learning a lens for"}


def doing(st: dict) -> str:
    return f"{DOING.get(st['kind'], st['kind'])} {st.get('name')}"


def cancel(job: str) -> bool:
    """Stop a running job: SIGTERM, which runs its ``finally`` (child stopped,
    lock released, model reloaded). False when it is not running, or is starting and has no pid yet."""
    st = read_state(job)
    if not st or st.get("outcome") is not None or not st.get("pid"):   # no outcome: its process holds the lock
        return False
    os.kill(st["pid"], signal.SIGTERM)
    return True

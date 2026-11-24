"""The GPU lock and the memory checks around a run that loads its own model copy.

Why: a served model and a run's own copy of it (``concept learn``, ``jlens learn``) do not both fit in memory
(Flash Next is near 100 GB of a 128 GB Mac), so the served model is unloaded first. oMLX loads a model on demand when a request asks for it, so without a lock a single chat message
mid-run would load it again beside the run's copy.

The lock is an ``flock`` on ``<home>/gpu.lock``:

- The run takes it before unloading and holds it until it has finished and
  its memory is back. A child process that inherits the file descriptor
  shares it.
- The kernel releases it only when every process holding it is gone
  (exit, SIGKILL, a memory kill, a reboot). It never expires on a timer: a
  stuck run still holds its memory, so it must still hold the lock.
- The holder writes who it is into the file (``Holder``), so a refusal can
  say what is running and how long is left.
- Inside oMLX, ``patch_engine_pool`` (omlx/patches.py) refuses every model load
  while the lock is held (``held()``), with that message.

The memory helpers read the kernel directly (``vm_stat``-style page counts,
the memory-pressure level and swap use through ``sysctlbyname``).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import fcntl
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import config

GB = 1024 ** 3


def lock_path() -> Path:
    return config.home() / "gpu.lock"


@dataclass
class Holder:
    """Who holds the lock, for messages ("learning crunchy: profiling, about 11 min left")."""
    what: str                       # e.g. "learning crunchy", "concept split angry"
    pid: int = field(default_factory=os.getpid)
    since: float = field(default_factory=time.time)
    stage: str = ""                 # updated as the run goes
    eta: Optional[float] = None     # epoch seconds when the run expects to finish

    def doing(self) -> str:
        """What runs, its stage and the time left ("learning crunchy: presence, about 11 min left")."""
        text = self.what + (f": {self.stage}" if self.stage else "")
        if self.eta:
            text += (", " if self.stage else ": ") + f"about {max(1, round((self.eta - time.time()) / 60))} min left"
        return text

    def message(self) -> str:
        return self.doing() + "; models load again when it is done"


class GpuLock:
    """Held for the lifetime of the open file descriptor ``fd``.

    ``acquire`` retries for ``wait`` seconds: a look (``held()``, ``unclean()``) takes
    a shared lock for an instant, so a free lock can look busy to it for a moment.
    """

    def __init__(self, holder: Holder, path: Optional[Path] = None):
        self.holder = holder
        self.path = path or lock_path()
        self.fd: Optional[int] = None

    def acquire(self, wait: float = 5.0) -> "GpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    other = read_holder(self.path)
                    os.close(fd)
                    raise LockBusy(other)
                time.sleep(0.1)
        self.fd = fd
        try:
            self.update()
        except BaseException:
            self.fd = None               # closed: an open descriptor would hold the lock until this process ends
            os.close(fd)
            raise
        return self

    def update(self, **changes) -> None:
        """Change the holder's stage or eta and rewrite it into the lock file."""
        for k, v in changes.items():
            setattr(self.holder, k, v)
        if self.fd is not None:
            self._write(self.fd)

    def _write(self, fd: int, **extra) -> None:
        """The holder (and ``extra``) as the lock file's whole content."""
        os.ftruncate(fd, 0)
        os.pwrite(fd, json.dumps(dict(asdict(self.holder), **extra)).encode(), 0)

    def release(self, clean: bool = True) -> None:
        """Close this process's descriptor. Never ``LOCK_UN``: an flock lives on
        the open file description, which a child shares through ``pass_fds``,
        so unlocking here would free the lock while the child still holds its
        memory. Closing frees it only when the last holder is gone. ``clean``
        (the caller knows the run's memory is back) marks the file released,
        so ``unclean()`` does not treat it as a crash."""
        fd, self.fd = self.fd, None          # first, so an interrupt mid-release never touches a reused descriptor
        if fd is None:
            return
        if clean:
            try:
                self._write(fd, released=True)
            except OSError:
                pass
        os.close(fd)


class LockBusy(RuntimeError):
    def __init__(self, holder: Optional[Holder]):
        super().__init__(f"the GPU is in use: {holder.message() if holder else 'another run holds the lock'}")


def _read(path: Optional[Path] = None) -> Optional[dict]:
    data = config.read_json(path or lock_path())
    return data if isinstance(data, dict) else None


UNCLEAN_WINDOW_S = 300.0      # a killed run's memory is back well within this


def unclean(path: Optional[Path] = None, now: Optional[float] = None) -> Optional[Holder]:
    """The last holder when the lock is free but was not released cleanly
    (the run was killed or crashed): its memory may still be coming back.
    oMLX's load guard then checks free memory before loading. None otherwise,
    and None once that cannot be so: the lock was written before the machine
    last booted, or it was first found dead over ``UNCLEAN_WINDOW_S`` ago.
    Without the expiry a large model (more than free memory less the guard's
    margin) could never load again after one killed run.

    When it was first found dead is written into the file (``dead_since``),
    under the lock, so every process (a restarted server too) counts from the
    same moment. Not the file's mtime: a run's last write can be hours before
    it died (a run by hand writes only when it starts)."""
    p = path or lock_path()
    try:
        fd = os.open(p, os.O_RDWR)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)      # shared, as held(): looks never block each other
        except BlockingIOError:
            return None                      # held: a run is going, not dead
        try:
            data = json.loads(os.pread(fd, 1 << 16, 0) or b"null")
            if not isinstance(data, dict) or data.get("released"):
                return None
            booted = boot_time()
            if booted is not None and os.fstat(fd).st_mtime < booted:
                return None
            now = time.time() if now is None else now
            if "dead_since" not in data:
                data["dead_since"] = now
                try:                         # written alone; another look holding it now writes it itself
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.ftruncate(fd, 0)
                    os.pwrite(fd, json.dumps(data).encode(), 0)
                except BlockingIOError:
                    pass
            if now - data["dead_since"] > UNCLEAN_WINDOW_S:
                return None
            return Holder(**{k: data[k] for k in Holder.__dataclass_fields__ if k in data})
        except (ValueError, TypeError, OSError):
            return None
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def read_holder(path: Optional[Path] = None) -> Optional[Holder]:
    data = _read(path)
    if data is None:
        return None
    try:
        return Holder(**{k: data[k] for k in Holder.__dataclass_fields__ if k in data})
    except TypeError:
        return None


def held(path: Optional[Path] = None) -> Optional[Holder]:
    """The holder if the lock is held now, else None. Never blocks."""
    p = path or lock_path()
    if not p.exists():
        return None
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)      # shared: another look at the same moment still gets in
        except BlockingIOError:
            return read_holder(p) or Holder("another run")
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


# ------------------------------------------------------------------ memory

_libc = None


def _sysctl_raw(name: str, size: int) -> Optional[bytes]:
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    buf = ctypes.create_string_buffer(size)
    n = ctypes.c_size_t(size)
    if _libc.sysctlbyname(name.encode(), buf, ctypes.byref(n), None, ctypes.c_size_t(0)) != 0:
        return None
    return buf.raw[: n.value]


def boot_time() -> Optional[float]:
    """When the machine last booted, in epoch seconds (``kern.boottime``: a struct timeval)."""
    raw = _sysctl_raw("kern.boottime", 16)
    return float(int.from_bytes(raw[:8], "little")) if raw and len(raw) >= 8 else None


def pressure_level() -> Optional[int]:
    """The kernel's memory-pressure level: 1 normal, 2 warning, 4 critical."""
    raw = _sysctl_raw("kern.memorystatus_vm_pressure_level", 8)
    return int.from_bytes(raw, "little") if raw else None


def swap_used() -> Optional[int]:
    """Bytes of swap in use (``vm.swapusage``: total, avail, used, ...)."""
    raw = _sysctl_raw("vm.swapusage", 32)
    return int.from_bytes(raw[16:24], "little") if raw and len(raw) >= 24 else None


def available() -> Optional[int]:
    """Free plus inactive memory in bytes, as the standing rule reads ``vm_stat``."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page, counts = 16384, {}
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        elif ":" in line:
            k, v = line.split(":", 1)
            try:
                counts[k.strip()] = int(v.strip().rstrip("."))
            except ValueError:
                pass
    if "Pages free" not in counts:
        return None
    return (counts.get("Pages free", 0) + counts.get("Pages inactive", 0) + counts.get("Pages speculative", 0)) * page


def wait_available(need_gb: float, timeout: float = 120.0, log: Callable[[str], None] = print,
                   reader: Callable[[], Optional[int]] = available, sleep: float = 2.0,
                   settle: Optional[float] = None, clock: Callable[[], float] = time.monotonic) -> bool:
    """Poll until free plus inactive memory is at least ``need_gb``; False on timeout. With ``settle``, also True
    once it has not risen by a GB for ``settle`` seconds: a finished run's memory is back as far as it comes (other
    programs may have grown meanwhile)."""
    deadline = clock() + timeout
    last = None
    base, rose = None, clock()          # the reading when it last rose by a GB, and when
    while True:
        avail = reader()
        now = clock()
        if avail is not None and avail >= need_gb * GB:
            log(f"memory: {avail / GB:.0f} GB free or inactive (need {need_gb:.0f})")
            return True
        if avail is not None and (base is None or avail >= base + GB):
            base, rose = avail, now
        if settle is not None and avail is not None and now - rose >= settle:
            log(f"memory: {avail / GB:.0f} GB free or inactive, steady for {settle:.0f}s (need {need_gb:.0f})")
            return True
        if now >= deadline:
            log(f"memory: only {(avail or 0) / GB:.0f} GB free or inactive after {timeout:.0f}s (need {need_gb:.0f})")
            return False
        if avail is not None and (last is None or abs(avail - last) > GB):
            log(f"memory: {avail / GB:.0f} GB free or inactive, waiting for {need_gb:.0f}")
            last = avail
        time.sleep(sleep)


class Watchdog(threading.Thread):
    """Kills the process when memory gets dangerous, before the machine does.

    Checks once a second: the kernel's pressure level reaching critical, swap
    growing by more than ``max_swap_growth_gb`` since start, or the parent
    process dying (the child must not outlive the job that will reload the
    served model). On any of them it writes the reason with ``log`` and calls
    ``os._exit(3)``: no cleanup that could allocate.
    """

    def __init__(self, log: Callable[[str], None] = print, max_swap_growth_gb: float = 4.0,
                 interval: float = 1.0, parent: Optional[int] = None, exit_fn: Callable[[int], None] = os._exit,
                 pressure: Callable[[], Optional[int]] = pressure_level, swap: Callable[[], Optional[int]] = swap_used):
        super().__init__(daemon=True, name="metalscope-watchdog")
        self.log, self.interval, self.exit_fn = log, interval, exit_fn
        self.pressure, self.swap = pressure, swap
        self.parent = parent if parent is not None else os.getppid()
        self.swap0 = swap() or 0
        self.max_growth = max_swap_growth_gb * GB
        self._halt = threading.Event()        # not _stop: Thread.join() and is_alive() call their own _stop()

    def check(self) -> Optional[str]:
        level = self.pressure()
        if level is not None and level >= 4:
            return "memory pressure critical"
        s = self.swap()
        if s is not None and s - self.swap0 > self.max_growth:
            return f"swap grew by {(s - self.swap0) / GB:.1f} GB"
        if self.parent > 1 and os.getppid() != self.parent:
            return "the parent job died"
        return None

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            reason = self.check()
            if reason:
                try:
                    self.log(f"watchdog: {reason}; stopping now")
                finally:
                    self.exit_fn(3)
                return

    def stop(self) -> None:
        self._halt.set()


# ------------------------------------------------------------------ unload

def wait_unloaded(admin, model_id: str, timeout: float = 180.0, sleep: float = 1.0,
                  log: Callable[[str], None] = print) -> bool:
    """Poll oMLX until ``model_id`` is neither loaded nor loading. oMLX's
    unload answers 202 while it aborts active requests, before the model is
    gone, so the answer alone does not mean the memory is free."""
    deadline = time.monotonic() + timeout
    while True:
        state = next((m for m in admin.admin_models() if m.get("id") == model_id), None)
        if state is None or (not state.get("loaded") and not state.get("is_loading")):
            return True
        if time.monotonic() >= deadline:
            log(f"{model_id} still {'loading' if state.get('is_loading') else 'loaded'} after {timeout:.0f}s")
            return False
        time.sleep(sleep)


def wait_no_loading(admin, timeout: float = 300.0, sleep: float = 2.0, log: Callable[[str], None] = print) -> bool:
    """Poll until no model is loading. A load that passed oMLX's guard just
    before the lock was taken keeps going; its weights are not resident yet,
    so a memory reading taken meanwhile would be too high."""
    deadline = time.monotonic() + timeout
    while True:
        loading = [m["id"] for m in admin.admin_models() if m.get("is_loading")]
        if not loading:
            return True
        if time.monotonic() >= deadline:
            log(f"still loading after {timeout:.0f}s: {', '.join(loading)}")
            return False
        time.sleep(sleep)


def model_bytes(path: str) -> int:
    p = Path(path).expanduser()
    return sum(f.stat().st_size for f in p.glob("*.safetensors")) if p.is_dir() else 0


def need_gb_for(path: str) -> float:
    """Memory the run needs free before it loads: about the weights' size on
    disk (Flash Next: 99 GB on disk, about 100 GB in use by a profile run),
    and never less than 6 GB."""
    return max(6.0, round(model_bytes(path) / GB * 0.97))

"""The GPU window a job's child runs in: on oMLX (``OmlxWindow``) the served model gives way to the child's own copy
and comes back after; on ds4 (``Ds4Window``) the child measures the model ds4-server serves.

The oMLX window's steps:

- lock: take the GPU lock (``gpulock``); from here oMLX refuses every model load with the lock's message.
- unload: when the server has the model loaded, unload it and wait until it is gone (oMLX answers 202 before it is).
- memory: wait until free plus inactive memory covers the run.
- gpu: the child (``CHILD``) inherits the lock and does every measurement on one load; its ``@@`` lines become the
  state.
- release: wait for the child's memory to come back, release the lock, reload the model if it was loaded before.

On ds4: the same lock (the ds4 proxy then refuses chat with its message), the child on ``--engine ds4``, nothing
unloaded or reloaded.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .. import client, gpulock
from ..omlx import admin as omlx_admin, install as omlx_install
from ..engines.mlx import local_mlx_lm, pick_loader

if TYPE_CHECKING:                        # for the annotations only: job.py imports this module
    from .job import GpuJob


# The oMLX window's steps, in every GPU job's list on oMLX.
UNLOAD = ("unload", "unload", ("locking", "unloading", "waiting for memory"))
LOAD = ("load", "load model", ("loading the model",))
RELOAD = ("reload", "reload", ("releasing", "reloading"))
_sleep = time.sleep          # the reload's wait, for tests to replace: time.sleep itself is every thread's


class OmlxWindow:
    """The window on oMLX (the module docstring's lock, unload, memory, gpu and release): the served model unloaded,
    the child an MLX copy of ``model_path`` under a memory limit, the model reloaded after. Its steps are the kind's
    ``STEPS`` (with ``UNLOAD``, ``LOAD``, ``RELOAD``) and ``WEIGHTS``."""
    name = "omlx"

    def __init__(self, model_path: str, loader: str = "auto", need_gb: Optional[float] = None, admin=None):
        self.model_path = str(Path(model_path).expanduser())
        self.loader = loader
        self.need_gb = need_gb if need_gb is not None else gpulock.need_gb_for(model_path)
        self.admin = admin or omlx_admin.Admin()
        self.lock: Optional[gpulock.GpuLock] = None
        self.unloaded = False
        self.reloaded = False

    def layout(self, kind) -> tuple[tuple, dict]:
        """The kind's steps and weights in this window."""
        return kind.STEPS, kind.WEIGHTS

    def fields(self) -> dict:
        """What the job's state records of the window."""
        return dict(need_gb=self.need_gb, served_was_loaded=None)

    def server(self) -> client.Client:
        """The server the job's chat requests go to (the corpus writer, the calibration)."""
        return client.Client(self.admin.base_url)

    def check(self, job: "GpuJob") -> None:
        """What must hold before the job writes anything: nothing here (oMLX's state is read under the lock)."""

    def take(self, job: "GpuJob") -> None:
        job.phase("locking")
        self.lock = gpulock.GpuLock(gpulock.Holder(job.holder(), stage="starting")).acquire(wait=10)
        served ={m.get("id"): m for m in self.admin.admin_models()}
        st = served.get(job.model_name) or {}
        job.state["served_was_loaded"] = bool(st.get("loaded") or st.get("is_loading"))
        if job.state["served_was_loaded"]:
            # Unloading is only safe if oMLX will refuse to load it back while the run holds the lock.
            status = self.admin.trace_status() or {}
            if not status.get("gpu_lock_enforced"):
                raise RuntimeError("the oMLX server does not enforce the GPU lock (metalscope not loaded, or not restarted "
                                   "since the update: metalscope omlx restart); not unloading")
            job.phase("unloading")
            job.log(f"unloading {job.model_name} (chat pauses until it is back)")
            omlx_admin.unload(self.admin, job.model_name, log=job.log)
            self.unloaded = True
        job.phase("waiting for memory")
        self.memory_ready(job)
        free = (gpulock.available() or 0) / gpulock.GB
        job.summarize("unload", (f"{job.model_name} unloaded" if self.unloaded else "nothing to unload") + f" · {free:.0f} GB free")

    def memory_ready(self, job: "GpuJob") -> None:
        """No load in flight (one that passed oMLX's guard before the lock keeps going) and enough memory free."""
        if not gpulock.wait_no_loading(self.admin, timeout=300, log=job.log):
            raise RuntimeError("a model is still loading on the server")
        if not gpulock.wait_available(self.need_gb, timeout=120, log=job.log):
            raise RuntimeError(f"not enough free memory for the run ({self.need_gb:.0f} GB needed)")

    def child_args(self, job: "GpuJob", loader: str, limit_gb: float) -> list[str]:
        """The child's arguments after ``-m CHILD``: every child's (``gpu.child_parser``), then the kind's."""
        return [job.model_name, "--model", self.model_path, "--loader", loader, "--memory-limit-gb", f"{limit_gb:.0f}",
                *job.extra_args()]

    def command(self, job: "GpuJob") -> tuple[list[str], dict]:
        """The child's command line and environment."""
        job.phase("loading the model")
        self.memory_ready(job)                    # again, right before the child starts
        avail = gpulock.available() or 0
        limit = max(self.need_gb, avail / gpulock.GB - 6)
        loader = pick_loader(self.model_path) if self.loader == "auto" else self.loader
        # oMLX's loader needs oMLX's bundled Python and its packages, with this metalscope importable; so does mlx_lm
        # when this Python has none (a plain install: oMLX's Python has it).
        bundled = loader == "omlx" or not local_mlx_lm()
        python, env = omlx_install.bundled_command() if bundled else ([sys.executable], dict(os.environ))
        cmd = python + ["-m", job.CHILD, *self.child_args(job, loader, limit)]
        job.log(f"starting the GPU run: {job.model_name} from {self.model_path} with the {loader} loader "
                f"(memory limit {limit:.0f} GB)")
        # METALSCOPE_DISABLE: a plain model copy. oMLX's Python runs metalscope's boot hook at startup, which would
        # otherwise patch the copy's engine like the server's (the CLI's --loader omlx sets it too, but later).
        env.update(PYTHONUNBUFFERED="1", METALSCOPE_DISABLE="1")
        return cmd, env

    def guard(self, job: "GpuJob", child: subprocess.Popen) -> Optional[gpulock.Watchdog]:
        """The job watches memory too, from its own process: the child's watchdog is a Python thread that a long
        MLX call (loading the weights) can hold off; this one kills the child's whole group."""
        def kill_child(_code):
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        guard = gpulock.Watchdog(log=job.log, parent=0, exit_fn=kill_child)
        guard.start()
        return guard

    def release(self, job: "GpuJob", ran: bool) -> None:
        """Memory back (``ran``: a child ran), then the lock, then the model. Safe to call twice."""
        memory_back = True
        if self.lock is not None and self.lock.fd is not None:
            job.phase("releasing")
            if ran:
                # Back once steady, not only at need_gb: a run's memory can settle a GB or so under what was
                # free before it, and waiting for need_gb alone then leaves the lock unclean.
                memory_back = gpulock.wait_available(self.need_gb, timeout=300, settle=60, log=job.log)
            # Not clean when the memory was still coming back: oMLX's guard then checks free memory before any load.
            self.lock.release(clean=memory_back)
        if self.unloaded and not self.reloaded:
            if not memory_back:
                self.reloaded = True
                job.log(f"memory is not back yet: not reloading {job.model_name}; the next chat loads it once it is free")
                job.summarize("reload", "not reloaded: memory not back yet; the next chat loads it")
                return
            job.phase("reloading")
            for _ in range(10):
                try:
                    omlx_admin.load(self.admin, job.model_name, log=job.log)
                    failed = None
                except omlx_admin.ServerError as exc:
                    failed = exc
                # Once the server has answered: a cancel cutting the load request short leaves the reload to the
                # next release, one during the wait below leaves it to the next chat (no new round of retries).
                self.reloaded = True
                if failed is None:
                    job.summarize("reload", f"{job.model_name} is back")
                    break
                if "409" in str(failed) or "already" in str(failed).lower():
                    job.log(f"{job.model_name} is already loading or loaded")     # a chat request got there first
                    break
                job.log(f"reload failed ({failed}); retrying in 60s")
                _sleep(60)
            else:
                job.log(f"could not reload {job.model_name}; the next chat request will load it")
                job.summarize("reload", "could not reload; the next chat loads it")


# The ds4 window's steps (a kind's ``DS4_STEPS``).
HOLD = ("hold", "hold ds4", ("locking", "connecting"))
FREE = ("release", "release", ("releasing",))


class Ds4Window:
    """The window on ds4: the child measures on the model ds4-server serves, so nothing is unloaded, loaded or
    reloaded, and no memory is waited for.

    - lock: the GPU lock; while it is held the ds4 proxy refuses chat with its message (ds4-server answers one request
      at a time, and the run's requests go to it directly, past the proxy); a reply in flight finishes first.
    - connecting: the proxy at ``url`` serves the job's model.
    - gpu: the child (``--engine ds4``) inherits the lock.
    - release: the lock.

    Its steps are the kind's ``DS4_STEPS`` (with ``HOLD`` and ``FREE``) and ``DS4_WEIGHTS``."""
    name = "ds4"

    def __init__(self, url: Optional[str] = None):
        self.url = (url or client.DS4_URL).rstrip("/")
        self.lock: Optional[gpulock.GpuLock] = None

    def layout(self, kind) -> tuple[tuple, dict]:
        if not getattr(kind, "DS4_STEPS", None):
            raise ValueError(f"a {kind.KIND} job does not run on ds4")
        return kind.DS4_STEPS, kind.DS4_WEIGHTS

    def fields(self) -> dict:
        return dict(window="ds4")

    def server(self) -> client.Client:
        return client.Client(self.url)

    def check(self, job: "GpuJob") -> None:
        """The proxy at ``url`` serves the job's model (RuntimeError else): before anything is written, and again
        under the lock (a switch is refused only from then on)."""
        try:
            info = self.server().get("/scope/api/ds4", timeout=10)
        except client.ServerError as exc:
            raise RuntimeError(f"no metalscope ds4 proxy at {self.url} ({exc})") from None
        if info.get("name") != job.model_name:
            raise RuntimeError(f"the ds4 proxy at {self.url} serves {info.get('name')}, not {job.model_name}")

    def take(self, job: "GpuJob") -> None:
        job.phase("locking")
        self.lock = gpulock.GpuLock(gpulock.Holder(job.holder(), stage="starting")).acquire(wait=10)
        job.phase("connecting")
        self.check(job)
        job.summarize("hold", f"ds4 keeps serving {job.model_name} · chat refused while it is measured")

    def child_args(self, job: "GpuJob") -> list[str]:
        return [job.model_name, "--engine", "ds4", "--url", self.url, *job.extra_args()]

    def command(self, job: "GpuJob") -> tuple[list[str], dict]:
        job.log(f"starting the GPU run on the model ds4 serves: {job.model_name} at {self.url}")
        return [sys.executable, "-m", job.CHILD, *self.child_args(job)], dict(os.environ, PYTHONUNBUFFERED="1")

    def guard(self, job: "GpuJob", child: subprocess.Popen) -> None:
        """None: the memory is ds4-server's, which this run does not change."""

    def release(self, job: "GpuJob", ran: bool) -> None:
        if self.lock is not None and self.lock.fd is not None:
            job.phase("releasing")
            self.lock.release(clean=True)
            job.summarize("release", "chat works again")

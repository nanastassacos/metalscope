"""The lens-learn job (``LensLearnJob``), which ``/jlens learn`` starts: a Jacobian lens fitted on a copy of the served
model in an oMLX window.
"""
from __future__ import annotations

import re
import time
from typing import Optional

from .. import config
from . import lens_fit_gpu as F
from .job import GpuJob
from .state import span
from .window import LOAD, RELOAD, UNLOAD, OmlxWindow


def lens_job_id(model_name: str, kind: str) -> str:
    return f"{kind}-{re.sub(r'[^A-Za-z0-9_.-]', '-', config.short_name(model_name)).lstrip('.-')}-{time.strftime('%Y%m%d-%H%M%S')}"


RELOAD_S = 180                       # the release and reload after a fit, kept before the deadline


class LensLearnJob(GpuJob):
    """``/jlens learn``: a Jacobian lens fitted for the served model on its own copy (``lens_fit_gpu``), window by
    window until the deadline (``until``, HH:MM) or a number of windows, then the model reloaded. The lens is written
    after every window to its own directory, used by nothing until ``/jlens use``. The result: the lens, the windows
    fitted, why it stopped, the held-out scores."""
    KIND = "lens-learn"
    CHILD = "metalscope.learn.lens_fit_gpu"
    NO_RESULT = "a lens"
    # The window, the preparation on the loaded model, then hours of fitting: for one bar over an overnight run, the
    # fit is nearly all of it.
    WEIGHTS = {"locking": 0.0, "unloading": 0.005, "waiting for memory": 0.005, "loading the model": 0.01,
               "building the corpus": 0.005, "scoring the logit lens": 0.01, "sizing the batch": 0.01, "fitting": 0.945,
               "releasing": 0.005, "reloading": 0.005}
    STEPS = (UNLOAD, LOAD, ("corpus", "corpus", ("building the corpus",)), ("compare", "compare", ("scoring the logit lens",)),
             ("batch", "batch", ("sizing the batch",)), ("fit", "fit", ("fitting",)), RELOAD)

    def __init__(self, model_path: str, model_name: str, source: Optional[str] = None, until: Optional[str] = None,
                 windows: Optional[int] = None, fresh: bool = False, loader: str = "auto", need_gb: Optional[float] = None,
                 admin=None, job: Optional[str] = None, origin: str = "cli", now: Optional[float] = None):
        if not until and not windows:
            raise ValueError("say until when (HH:MM) or how many windows")
        self.source, self.until, self.windows, self.fresh = source, until, windows, fresh
        self.deadline = F.parse_until(until, now) if until else None
        window = OmlxWindow(model_path, loader, need_gb if need_gb is not None else F.fit_need_gb(model_path), admin)
        super().__init__(job or lens_job_id(model_name, "lens-learn"), model_name, window, origin, name=model_name, description=None, source=source, until=until, deadline=self.deadline,
                         windows=windows, fit=None, window_s=None)

    def holder(self) -> str:
        return f"learning a lens for {self.model_name}" + (f" until {self.until}" if self.until else "")

    def extra_args(self) -> list[str]:
        args = []
        if self.source:
            args += ["--source", self.source]
        if self.deadline is not None:
            # Stop a few minutes early: the reload is part of what "until" promises.
            args += ["--deadline", f"{self.deadline - RELOAD_S:.0f}"]
        if self.windows:
            args += ["--windows", str(self.windows)]
        return args + (["--fresh"] if self.fresh else [])

    def before_gpu(self) -> None:
        """The default corpus on disk before the window: its first use downloads it, which must not keep the model
        unloaded (and a failed download then fails the job with nothing unloaded)."""
        if not self.source:
            from ..jlens import corpus as LC
            LC.documents(None, log=self.log)

    def child_event(self, ev: dict) -> None:
        """The time a window takes, once the batch is sized (the job fails when none fits before the deadline), then
        each window's scores."""
        if "window_s" in ev:
            if ev.get("none_fits"):
                raise RuntimeError(f"a window takes about {span(ev['window_s'])}: none fits before {self.until}")
            self.state["window_s"] = ev["window_s"]
            self.save()
        if isinstance(ev.get("window"), dict):
            self.state["fit"] = ev["window"]
            self.state["window_s"] = 60 * ev["window"]["minutes"]
            self.step("fit")["detail"] = F.fit_text(ev["window"])
            self.save()

    WARN_SLOW = False                    # the fit runs to its deadline or windows: hours by design

    def phase(self, name: str, n: Optional[int] = None, total: Optional[int] = None) -> None:
        super().phase(name, n, total)
        if name == "fitting":
            if self.state.get("fit"):            # the last window's scores stay on the fit line between windows
                self.step("fit")["detail"] = F.fit_text(self.state["fit"])
            if self.deadline is not None:        # the deadline is the end, whatever the bar says
                self.state["eta"] = self.deadline
            self.save()

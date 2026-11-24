"""Starting a job from the server: ``metalscope concept learn|calibrate`` or ``jlens learn`` with ``--job``,
detached in its own session (so it outlives the server and Pi) under the repository's interpreter, once no other
job runs and the GPU is free; each raises ValueError with a readable reason when it cannot start.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .. import config, models
from ..omlx import install as omlx_install
from ..vectors import check_name
from . import lens_fit_gpu as F
from .lens_job import lens_job_id
from .plan import learn_plan
from .state import INHERITED, busy, job_dir, learn_dir, new_job_id, take


REPO = Path(__file__).resolve().parents[3]          # src/metalscope/learn/spawn.py -> the repository (in a checkout)
CHECKOUT = (REPO / "pyproject.toml").exists() and (REPO / "src" / "metalscope").is_dir()
KEEP_ENV = ("HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "SHELL", "METALSCOPE_HOME")


def repo_python() -> Path:
    """The interpreter a job runs in, never oMLX's bundled one (it cannot import mlx once the job's environment is
    cleaned, KEEP_ENV): a checkout's own (it has mlx_lm, which loads the checkpoint); inside oMLX's server (its hook
    loaded) the one that ran `omlx install` for that hook (ValueError when none is recorded); elsewhere (the ds4
    proxy, run by the install's own Python) this one."""
    py = REPO / ".venv" / "bin" / "python"
    if CHECKOUT and py.exists():
        return py
    # oMLX's process with the hook loaded, whatever the app is named (a test process imports the hook alone).
    if not (omlx_install.BOOT_MODULE in sys.modules and "omlx" in sys.modules):
        return Path(sys.executable)
    recorded = omlx_install.install_python()
    if recorded is None:
        raise ValueError("metalscope does not know which Python runs it: run metalscope omlx install again")
    return recorded


def learn(name: str, description: str, model_name: str, writer: Optional[str] = None, engine: Optional[str] = None,
          url: Optional[str] = None, strengths: Optional[list[float]] = None, corpus: Optional[str] = None) -> dict:
    """Start ``concept learn`` as a detached process (its own session, so it
    outlives the server and Pi) and return ``{job, pid}``: on ``engine`` (mlx, a
    copy on oMLX; ds4, the model ds4 serves; default the command's auto),
    asking the server at ``url`` (the one that started it). What it learns from,
    and what it refuses, is ``plan.learn_plan``'s (ValueError with its reason);
    the GPU in use or another job running refuse too."""
    plan = learn_plan(name, model_name, description, corpus, writer, engine, url)
    if plan["taken"]:
        raise ValueError(plan["taken"])
    name, corpus = plan["name"], plan["corpus"]          # corpus absolute: the job runs from another directory
    job = new_job_id(name)
    # The description as given: the job's own plan reads it again (and notes one given for existing texts). Free text
    # goes as --option=VALUE, so a value starting with "-" is never read as an option.
    return _launch(job, ["concept", "learn", name, model_name] + ([f"--description={description}"] if description.strip() else [])
                   + (["--corpus", corpus] if corpus else [])
                   + ([f"--writer={writer}"] if writer else []) + (["--engine", engine] if engine else [])
                   + (["--url", url] if url else [])
                   + (["--strengths", ",".join(f"{s:g}" for s in strengths)] if strengths else []),
                   "learn", name, model_name, description=plan["description"],
                   **({"window": "ds4"} if engine == "ds4" else {}))


def calibrate(name: str, model_name: str, url: Optional[str] = None) -> dict:
    """Start ``concept calibrate`` as a detached job (chat requests to the served model at ``url``, the server that
    started it; nothing is unloaded) and return ``{job, pid}``. Raises ValueError when it cannot start: no profile for
    the model, the GPU in use or another job running."""
    name = name.strip()
    if not name:
        raise ValueError("name the concept to calibrate")
    check_name(name)
    if not models.profile_path(model_name, name).exists():
        raise ValueError(f"{name} has no profile for {model_name}: /concept list shows the profiled concepts")
    job = new_job_id(name, "calibrate")
    return _launch(job, ["concept", "calibrate", name, model_name] + (["--url", url] if url else []),
                   "calibrate", name, model_name)


def lens_learn_plan(model_name: str, model_path: Optional[str], until: Optional[str] = None, windows: Optional[int] = None,
                    source: Optional[str] = None, now: Optional[float] = None) -> dict:
    """What ``/jlens learn`` would do, for its confirm text: the deadline, the memory it waits for, the estimated
    minutes a window and windows by the deadline (``F.fit_estimate``), the corpus (its kind and, when its files are at
    hand, its id and the windows an earlier fit of this model on it got through: those resume), and the lens
    directory it writes. ValueError for a bad time, count or path."""
    from ..jlens import corpus as LC
    now = time.time() if now is None else now
    if windows is not None and not 1 <= int(windows) <= 10000:
        raise ValueError("windows: a number from 1 to 10000")
    if not until and not windows:
        raise ValueError("say until when (until HH:MM) or how many windows (windows N)")
    deadline = F.parse_until(until, now) if until else None
    kind, corpus_id, resume = "default", None, 0
    if source:
        p = Path(source).expanduser()
        if not p.exists():
            raise ValueError(f"{p} does not exist")
        if not (p.is_dir() or (p.is_file() and p.suffix.lower() == ".txt")):
            raise ValueError(f"{p}: a directory of text files or one .txt")
        source, kind = str(p.resolve()), "dir" if p.is_dir() else "txt"
    if source or (LC.corpus_dir() / LC.WIKITEXT_FILE).exists():       # never a download just to plan
        k, docs = LC.documents(source)
        corpus_id = LC.corpus_id(k, docs, LC.WINDOW, LC.MAX_DEPTH, 0)
        try:
            resume = int(json.loads((F.fit_dir(model_name, corpus_id) / "fit_state.json").read_text())["done"])
        except (OSError, ValueError, KeyError):
            resume = 0
    est = F.fit_estimate(model_path)
    n_est = None
    if est and deadline is not None:
        n_est = F.windows_by(now + 60 * est["overhead_min"], deadline, 60 * est["min_per_window"])
    if windows:
        n_est = min(n_est, int(windows)) if n_est is not None else int(windows)
    return dict(model=config.short_name(model_name), path=model_path, until=until, deadline=deadline, windows=windows,
                need_gb=F.fit_need_gb(model_path) if model_path else None, estimate=est, windows_estimate=n_est,
                source=source, kind=kind, shallow_only=kind == "txt", corpus_id=corpus_id, resume=resume,
                lens=f"{config.short_name(model_name)}-learned-{time.strftime('%Y%m%d', time.localtime(now))}")


def lens_learn(model_name: str, source: Optional[str] = None, until: Optional[str] = None,
                     windows: Optional[int] = None, fresh: bool = False) -> dict:
    """Start ``jlens learn NAME [PATH] --until HH:MM --job`` as a detached job (``LensLearnJob``: the model is unloaded
    until the deadline) and return ``{job, pid}``. Raises ValueError when it cannot start: no deadline or count, a
    bad time or path, the GPU in use or another job running."""
    plan = lens_learn_plan(model_name, None, until, windows, source)
    args = ["jlens", "learn", model_name] + ([plan["source"]] if plan["source"] else [])
    args += (["--until", until] if until else []) + (["--windows", str(windows)] if windows else []) + (["--fresh"] if fresh else [])
    return _launch(lens_job_id(model_name, "lens-learn"), args, "lens-learn", model_name, model_name, until=until,
                   windows=windows, source=plan["source"])


def _launch(job: str, args: list[str], kind: str, name: str, model_name: str, **fields) -> dict:
    """Run ``metalscope ARGS --job JOB --origin pi`` detached in the repository's interpreter, once no other job runs
    and the GPU is free and no backend switch is under way (ValueError else), with a first state until the job writes
    its own. The check, the job's lock and that first state are one step under ``launch.lock`` and the switch lock
    (backends.switching), so of two starters at once only one starts, and a switch sees the job."""
    d = job_dir(job)
    cmd = [str(repo_python()), "-m", "metalscope.cli", *args, "--job", job, "--origin", "pi"]
    env = {k: os.environ[k] for k in KEEP_ENV if k in os.environ}
    env.update(PYTHONUNBUFFERED="1")
    if CHECKOUT:                         # an installed metalscope is on its interpreter's path already
        env.update(PYTHONPATH=str(REPO / "src"))
    # No pid until the process exists: the folder reads as running once the lock is held, and cancel signals nothing
    # without one (an older run's pid in a reused folder may be another process's by now).
    stub = dict(dict(kind=kind, name=name, description=None, model=model_name), **fields)
    stub.update(job=job, phase="starting", progress=0.0, started=time.time(), updated=time.time(), pid=None, log=[],
                outcome=None, origin="pi")
    from ..backends import switching
    with config.file_lock(learn_dir() / "launch.lock"), switching(wait=False):
        why = busy()
        if why:
            raise ValueError(why)
        d.mkdir(parents=True, exist_ok=True)
        live = take(job)                 # taken here and passed on: the job reads as running from its first state
        env[INHERITED] = str(live.fd)
        try:
            config.atomic_write(d / "state.json", json.dumps(stub))
            with open(d / "stdout.txt", "w") as out:
                proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                        cwd=str(REPO if CHECKOUT else d), start_new_session=True, close_fds=True,
                                        pass_fds=(live.fd,))
        except BaseException as exc:
            config.atomic_write(d / "state.json", json.dumps(dict(stub, outcome="failed", phase="failed",
                                                                  error=str(exc) or type(exc).__name__)))
            raise
        finally:
            live.release(clean=False)    # the job's process holds it now
    # Reap it when it ends, so it never lingers as a zombie of the server.
    threading.Thread(target=proc.wait, daemon=True, name=f"metalscope-job-{job}").start()
    config.atomic_write(d / "state.json", json.dumps(dict(stub, pid=proc.pid)))
    return dict(job=job, pid=proc.pid)

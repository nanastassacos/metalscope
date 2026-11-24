"""The jobs: what every kind keeps (``Job``), a job with a GPU window (``GpuJob``), and the concept jobs ``learn``
(``LearnJob``) and ``calibrate`` (``CalibrateJob``, chat requests only), which Pi's ``/concept`` starts.

Every kind keeps the same state (state.py reads it, ``kind`` tells them apart; ``lens-learn``, which ``/jlens learn``
starts, is lens_job.py's), so one route and one widget serve them; one runs at a time (``state.busy``). Their command
lines: command.py.

A GPU job (``GpuJob``) runs its child in one window (window.py): a copy of the model on oMLX (``OmlxWindow``), or on
ds4 the model ds4-server serves (``Ds4Window``).

A learn job writes the corpus (or copies supplied or shipped texts; the model's own texts are used as they are,
nothing written) into the model's concept folder before the window (``writer``) and after it calibrates the
chat dose in conversation (``calibrate``), whose good replies rank the concept. The only runs it does not
calibrate are the ones that can't be steered: no item had room to move, or no layer to steer at (``decide``).

Whatever happens, the ``finally`` kills the child, releases the lock and reloads the model if this job unloaded it.
A job that died without its ``finally`` (SIGKILL) reads as "interrupted" (``state.read_state``); the kernel has
released its locks, and the next chat request loads the model again.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from .. import client, config, models
from ..concepts.corpus import TEXTS
from ..vectors import find_vector, list_vectors, load_vector
from . import calibrate as K, decide, writer as W
from .state import SEEN, busy, hold, job_dir, new_job_id, span
from .window import FREE, HOLD, LOAD, RELOAD, UNLOAD, Ds4Window, OmlxWindow

LOG_TAIL = 20


class Job:
    """What every kind of job keeps: its state file, its step list and progress, and the failure handling (``_run``).
    A kind sets ``KIND``, its ``STEPS`` (key, label and the phases each covers: Pi's widget lists one line each) and
    ``WEIGHTS`` (the share of the whole run each phase takes, for one progress bar), and runs a body of phases with
    ``_run``.

    A phase far past its usual time gets one warning line (the log and ``warning``, until the next phase); it never
    stops the job. Its usual time is its weight at the run's own pace so far (the seconds before it over the weights
    of the phases before it), at least ``SLOW_FLOOR``; its estimate is its time so far, or with a count, that time
    scaled to the whole count (from 5% of it); it warns once the estimate passes ``SLOW`` times the usual time and
    the time so far passes the usual time itself (early items can be slow, so an estimate from the first few runs
    long)."""
    KIND = ""
    STEPS: tuple = ()
    WEIGHTS: dict = {}
    ETA_FROM = 0.15                  # the progress from which an ETA is shown (before it, the weights are too rough)
    SLOW, SLOW_FLOOR = 4, 300        # the warning: an estimate over 4x the usual time, taken as at least 5 min
    WARN_SLOW = True
    stamp = True                     # log lines on stdout with their time; a foreground calibration's are as they were

    def __init__(self, job: str, model_name: str, origin: str = "cli", **fields):
        self.job, self.model_name = job, model_name
        self.dir = job_dir(self.job)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.live = hold(self.job)                    # before the first state: no outcome reads as running only while held
        (self.dir / SEEN).unlink(missing_ok=True)     # a reused id: Pi saw the earlier run, not this one
        self.order = list(self.WEIGHTS)
        self.entered: dict[str, float] = {}           # each phase's first start
        self.usual: dict[str, Optional[float]] = {}    # its usual seconds at the run's pace (None: nothing to go by)
        self.warned: set[str] = set()
        self.step_of = {phase: key for key, _, phases in self.STEPS for phase in phases}
        self.state = dict(kind=self.KIND, job=self.job, **fields, model=model_name, origin=origin,
                          phase="starting", n=None, total=None, progress=0.0, started=time.time(), updated=time.time(),
                          eta=None, warning=None, pid=os.getpid(), log=[], outcome=None, error=None, result=None,
                          steps=[dict(key=k, label=lab, status="pending", summary=None, detail=None, n=None, total=None,
                                      started=None, ended=None) for k, lab, _ in self.STEPS])
        self.save()

    def save(self) -> None:
        self.state["updated"] = time.time()
        config.atomic_write(self.dir / "state.json", json.dumps(self.state, indent=1, default=str))

    def log(self, msg: str, echo: bool = True) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        if echo:
            print(line if self.stamp else msg, flush=True)
        with open(self.dir / "log.txt", "a") as f:
            f.write(line + "\n")
        self.state["log"] = (self.state["log"] + [msg])[-LOG_TAIL:]
        self.save()

    def step(self, key: str) -> dict:
        return next(s for s in self.state["steps"] if s["key"] == key)

    def summarize(self, key: str, text: str) -> None:
        """A finished step's one-line result, for the widget."""
        self.step(key)["summary"] = text
        self.save()

    def _advance(self, name: str, n: Optional[int], total: Optional[int]) -> None:
        """The step list follows the phase: earlier running steps are done, this one runs, with its progress."""
        key = self.step_of.get(name)
        if key is None:
            return
        now = time.time()
        keys = [k for k, _, _ in self.STEPS]
        for st in self.state["steps"]:
            if keys.index(st["key"]) < keys.index(key) and st["status"] == "running":
                st.update(status="done", ended=now, n=None, total=None)
        cur = self.step(key)
        if cur["status"] == "pending":
            cur.update(status="running", started=now)
        cur.update(n=n, total=total, detail=name if name != cur["label"] else None)

    def _close_steps(self, failed: bool) -> None:
        now = time.time()
        for st in self.state["steps"]:
            if st["status"] == "running":
                st.update(status="failed" if failed else "done", ended=now, n=None, total=None)
            elif st["status"] == "pending" and not failed:
                st["status"] = "skipped"

    def _pace(self, name: str, n: Optional[int], total: Optional[int], now: float) -> None:
        """The warning for a phase far past its usual time (the class docstring)."""
        if name not in self.entered:
            before = sum(self.WEIGHTS.get(p, 0.0) for p in self.entered)
            w = self.WEIGHTS.get(name, 0.0)
            self.usual[name] = w * (now - self.state["started"]) / before if before > 0 and w > 0 else None
            self.entered[name] = now
        if name != self.state["phase"]:
            self.state["warning"] = None
        usual = self.usual[name]
        if not self.WARN_SLOW or usual is None or name in self.warned:
            return
        ran = now - self.entered[name]
        est = ran * total / n if n and total and n >= 0.05 * total else ran
        if est > self.SLOW * max(usual, self.SLOW_FLOOR) and ran > max(usual, self.SLOW_FLOOR):
            self.warned.add(name)
            self.state["warning"] = (f"{name}: estimated {span(est)}, expected about {span(usual)}; something is off, "
                                     "consider stopping it")
            self.log(self.state["warning"])

    def phase(self, name: str, n: Optional[int] = None, total: Optional[int] = None) -> None:
        self._pace(name, n, total, time.time())
        self._advance(name, n, total)
        self.state.update(phase=name, n=n, total=total)
        done = sum(self.WEIGHTS[p] for p in self.order[:self.order.index(name)]) if name in self.WEIGHTS else self.state["progress"]
        frac = (n / total) if (n is not None and total) else 0.0
        self.state["progress"] = min(1.0, done + self.WEIGHTS.get(name, 0.0) * frac)
        elapsed = time.time() - self.state["started"]
        p = self.state["progress"]
        self.state["eta"] = time.time() + elapsed * (1 - p) / p if p > self.ETA_FROM else None
        self.save()

    def claim(self) -> None:
        """The job's turn, asked once its own state is written (a job started meanwhile sees it): RuntimeError while
        another job runs or the GPU is held. One started from a terminal also refuses while a backend switch is under
        way, which looked for jobs before this one was there; the server's launch held the switch lock itself
        (spawn._launch)."""
        from ..backends import switching
        try:
            with switching(wait=False) if self.state["origin"] == "cli" else contextlib.nullcontext():
                why = busy(exclude=self.job)
        except ValueError as exc:
            why = str(exc)
        if why:
            raise RuntimeError(why)

    def release(self) -> None:
        """What the job holds, given back (``GpuJob``: the child, the lock, the model). Safe to call twice."""

    def cleanup_failed(self) -> None:
        """What a failed run undoes, once ``release`` ran."""

    def _run(self, body) -> dict:
        """Run ``body`` (the kind's phases); a failure (a cancel too) is the outcome, then ``cleanup_failed``. Whatever
        happens, ``release`` runs: for a GPU job the child stopped, the lock released, the model reloaded."""
        try:
            body()
            self.state.update(outcome="done", phase="done", progress=1.0, eta=None)
            self._close_steps(failed=False)
        except BaseException as exc:      # KeyboardInterrupt and SIGTERM too: the finally must run
            why = "stopped" if isinstance(exc, KeyboardInterrupt) else str(exc) or type(exc).__name__   # Ctrl-C, SIGTERM
            self.state.update(outcome="failed", phase="failed", error=why)
            self._close_steps(failed=True)
            self.log(f"failed: {self.state['error']}", echo=self.stamp)     # unstamped: the command says it
            try:
                self.release()            # first: the child must be gone before cleanup_failed moves what it made
            finally:
                self.cleanup_failed()
        finally:
            try:
                self.release()            # again: a second cancel can cut the first release short
            finally:                      # and a third this one: the final state is written whatever it raised
                for st in self.state["steps"]:          # the reload after a failure ran after the failed step closed
                    if st["status"] == "running":
                        st.update(status="done", ended=time.time(), n=None, total=None)
                if self.state["outcome"]:               # the release's phases came after the outcome: it stands
                    self.state.update(phase=self.state["outcome"], eta=None)
                self.save()
                self.live.release()                     # after the outcome is saved, so it never reads as interrupted
        return self.state


class GpuJob(Job):
    """A job with a GPU window (``window``: ``OmlxWindow`` or ``Ds4Window``). The window gives the kind's steps and weights
    (``layout``), takes the GPU, makes the child's command and gives everything back; the job runs the child (it
    inherits the lock) and turns its ``@@`` lines into the state. A kind sets the child's ``CHILD`` module and its own
    arguments (``extra_args``), the GPU lock's ``holder`` text, what the child's last line (``child_done``) means and
    what must hold first (``check``); ``NO_RESULT`` names what a child that ended without its result failed to make."""
    CHILD = ""
    NO_RESULT = ""

    def __init__(self, job: str, model_name: str, window, origin: str = "cli", **fields):
        self.window = window
        self.STEPS, self.WEIGHTS = window.layout(type(self))
        self.child: Optional[subprocess.Popen] = None
        super().__init__(job, model_name, origin, **fields, **window.fields())

    # ---- what a kind sets
    def holder(self) -> str:
        """Who holds the GPU lock, for the refusals it causes (oMLX's loads, the ds4 proxy's chat): "learning crunchy"."""
        raise NotImplementedError

    def extra_args(self) -> list[str]:
        """The kind's own arguments to its child, after the ones every child takes (the window's ``child_args``)."""
        return []

    def child_done(self, ev: dict) -> None:
        """The child's ``{"stage": "done", ...}`` line: its result."""
        self.state["result"] = ev.get("result")

    def child_event(self, ev: dict) -> None:
        """Any other ``@@`` line of the child's (a lens fit's per-window scores)."""

    def check(self) -> None:
        """What must hold before the job starts (RuntimeError else)."""

    def before_gpu(self) -> None:
        """The kind's phases before the window."""

    def after_gpu(self) -> None:
        """The kind's phases after it."""

    def run(self) -> dict:
        def body():
            self.check()
            self.claim()
            self.window.check(self)
            self.before_gpu()
            self.window.take(self)
            self.run_gpu()
            self.release()
            if self.NO_RESULT and not self.state.get("result"):
                raise RuntimeError(f"the GPU run ended without {self.NO_RESULT}")
            self.after_gpu()
        return self._run(body)

    def phase(self, name: str, n: Optional[int] = None, total: Optional[int] = None) -> None:
        super().phase(name, n, total)
        lock = self.window.lock
        if lock is not None and lock.fd is not None:
            lock.update(stage=name, eta=self.state["eta"])

    # ---- the child
    def run_gpu(self) -> None:
        cmd, env = self.window.command(self)
        # SIGTERM (cancel) is held off from fork to the assignment, so release() always knows the child it must
        # stop; the child unblocks it for itself before it runs anything.
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            self.child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                          pass_fds=(self.window.lock.fd,), start_new_session=True,
                                          preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM}))
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
        self.state["child_pid"] = self.child.pid
        self.save()
        guard = self.window.guard(self, self.child)
        try:
            self._stream_child()
        finally:
            if guard is not None:
                guard.stop()

    def _stream_child(self) -> None:
        for line in self.child.stdout:
            line = line.rstrip("\n")
            if line.startswith("@@"):
                try:
                    ev = json.loads(line[2:])
                except ValueError:
                    continue
                if ev.get("summary") and self.step_of.get(ev.get("step")):
                    self.summarize(self.step_of[ev["step"]], ev["summary"])
                elif ev.get("stage") in self.WEIGHTS:
                    self.phase(ev["stage"], ev.get("n"), ev.get("total"))
                elif ev.get("stage") == "done":
                    self.child_done(ev)
                else:
                    self.child_event(ev)
            elif line.strip():
                self.log(line)
        rc = self.child.wait()
        if rc != 0:
            # The exception the child died of: the first error line after its last traceback (shutdown noise, its own
            # "Exception ignored in" tracebacks included, can follow it); else its last lines.
            log_tail = self.state["log"]
            tb = max((i for i, m in enumerate(log_tail) if m.startswith("Traceback")), default=None)
            errors = [] if tb is None else [m for m in log_tail[tb:] if re.match(r"[\w.]*(Error|Exception)\b", m.strip())
                                            and not m.strip().startswith("Exception ignored")]
            tail = errors[0].strip() if errors else " / ".join(log_tail[-3:])
            raise RuntimeError(f"the GPU run exited with code {rc}" + (" (stopped by the memory watchdog)" if rc == 3 else "")
                               + f": {tail}")

    def release(self) -> None:
        """The child gone, then the window's (memory back, the lock, the model). Safe to call twice."""
        if self.child is not None and self.child.poll() is None:
            self.log("stopping the GPU run")
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
                self.child.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.child.wait()
        self.window.release(self, ran=self.child is not None)


def _with_calibrating(rest: dict, calibrating: float) -> dict:
    """Phase weights: ``rest`` scaled to share what ``calibrating`` (the last phase) leaves."""
    total = sum(rest.values())
    return {**{k: v * (1 - calibrating) / total for k, v in rest.items()}, "calibrating": calibrating}


class LearnJob(GpuJob):
    """``concept learn``: the corpus, the GPU window (``gpu``: every measurement on one load, or on the model ds4
    serves), then the chat calibration whenever there is a layer to steer at."""
    KIND = "learn"
    CHILD = "metalscope.learn.gpu"
    # Flash Next's timings, roughly: the rest about 15 min; calibrating 5-15 min a strength, 5-6 sets and grading.
    WEIGHTS = _with_calibrating({"writing": 0.06, "locking": 0.0, "unloading": 0.02, "waiting for memory": 0.01,
                                 "loading the model": 0.04, "presence": 0.06, "measuring the items": 0.03,
                                 "selection sweep": 0.39, "fluency ladder": 0.12, "test sweep": 0.08,
                                 "releasing": 0.02, "reloading": 0.02}, 0.8)
    STEPS = (("corpus", "corpus", ("writing",)), UNLOAD, LOAD, ("presence", "presence", ("presence",)),
             ("items", "items", ("measuring the items",)), ("selection", "selection", ("selection sweep",)),
             ("ladder", "fluency", ("fluency ladder",)), ("test", "test", ("test sweep",)),
             RELOAD, ("calibrate", "calibrate", ("calibrating",)))
    # On ds4 (Ds4Window): one measurement request at a time on the served model, no load; the sweeps are most of the
    # rest (V4.1: 1-1.6 h). Calibrating V4.1 takes 3-6 h (4.5-6 h inside a learn), about three times the rest.
    DS4_WEIGHTS = _with_calibrating({"writing": 0.02, "locking": 0.0, "connecting": 0.0, "presence": 0.04,
                                     "measuring the items": 0.05, "selection sweep": 0.53, "fluency ladder": 0.03,
                                     "test sweep": 0.11, "releasing": 0.0}, 0.75)
    DS4_STEPS = (STEPS[0], HOLD) + STEPS[3:8] + (FREE,) + STEPS[9:]

    def __init__(self, plan: dict, model_path: Optional[str], model_name: str, loader: str = "auto",
                 need_gb: Optional[float] = None, sweep_layers: str = "", seed: int = 0, admin=None,
                 job: Optional[str] = None, origin: str = "cli", url: Optional[str] = None,
                 strengths: Optional[list[float]] = None):
        """``plan``: ``learn_plan``'s, one that holds: the name, description, writer, engine and texts. On engine mlx, a
        copy of ``model_path`` on oMLX (``OmlxWindow``); on ds4, the model the ds4 proxy at ``url`` serves
        (``Ds4Window``). ``url``: the server the corpus writer and the calibration ask (default the window's).
        ``strengths``: the selection sweep's doses (default ``gpu.SELECT_STRENGTHS``). The plan's ``source`` says where
        the texts come from; ``write_corpus`` puts them in the model's concept folder."""
        name, description, writer_model, engine = plan["name"], plan["description"], plan["writer"], plan["engine"]
        self.plan = plan
        self.supplied = plan["read"] if plan["corpus"] else None
        self.source = plan["source"]
        self.name, self.description, self.writer_model = name, description, writer_model
        self.sweep_layers, self.seed, self.strengths = sweep_layers, seed, strengths
        self.url = url
        self.made: list[Path] = []      # the text files this run put in the concept's folder (written, or copied)
        self.defined: Optional[Path] = None     # the definition this run stored (corpus.define)
        v = find_vector(model_name, name)
        self.vector_mtime = v.stat().st_mtime_ns if v is not None else None   # a vector this run did not write stays
        self.existed = models.concept_dir(model_name, name).exists()
        window = Ds4Window(url) if engine == "ds4" else OmlxWindow(model_path, loader, need_gb, admin)
        super().__init__(job or new_job_id(name), model_name, window, origin,
                         name=name, description=description, writer=writer_model,
                         **({"supplied": self.supplied["log"]["supplied"]} if self.supplied else {}))

    @property
    def server(self) -> client.Client:
        """The server the corpus writer and the calibration ask: ``url``'s, else the window's."""
        return client.Client(self.url) if self.url else self.window.server()

    def holder(self) -> str:
        return f"learning {self.name}"

    def extra_args(self) -> list[str]:
        return ([self.name, "--seed", str(self.seed)] + (["--sweep-layers", self.sweep_layers] if self.sweep_layers else [])
                + (["--strengths", ",".join(f"{s:g}" for s in self.strengths)] if self.strengths else []))

    def child_done(self, ev: dict) -> None:
        self.state["profile"] = ev.get("profile")

    def before_gpu(self) -> None:
        for note in self.plan["notes"]:
            self.log(f"warning: {note}")
        self.write_corpus()

    def after_gpu(self) -> None:
        from ..concepts import calibration_of, steer_dose
        self.state["result"] = self.measured()
        if self.state["result"]["steerable"]:
            self.save()
            self.state["result"]["calibration"] = self.run_calibration()
        else:
            self.summarize("calibrate", self.state["result"]["reason"])
        # The dose /steer NAME now uses (the calibration's; None without one), for the result's "try it".
        prof = config.read_json(Path(self.state["profile"]))
        self.state["result"]["steer"] = steer_dose(prof, calibration_of(self.model_name, self.name)) if prof else None

    def _corpus_stage(self, what: str) -> None:
        """The writer's sub-steps ("writing the pairs") show on the corpus line while it runs."""
        self.step("corpus")["detail"] = what
        self.log(what)

    # ---- the learn job's own phases
    def _put_texts(self, files: dict[str, str]) -> Path:
        """Write the texts (file name -> contents) into the model's concept folder, made now when it is not there, whole
        or not at all: each under a dot-temp name, then renamed in, concept.json last (texts count from it: a run killed
        before leaves no texts, and the next run clears what it left). Never into one with texts: FileExistsError when
        another run or the user put them there meanwhile. From here, a failed run takes back what it put there
        (cleanup_failed), and only that."""
        d = models.concept_dir(self.model_name, self.name)
        d.mkdir(parents=True, exist_ok=True)
        if (d / "concept.json").exists():
            raise FileExistsError(f"{d} holds texts already")
        for n in TEXTS:                                  # a killed run's: temps, and files renamed in before concept.json
            (d / f".{n}.tmp").unlink(missing_ok=True)
            (d / n).unlink(missing_ok=True)
        order = sorted(files, key=lambda n: n == "concept.json")
        for fname in order:
            (d / f".{fname}.tmp").write_text(files[fname])
        for fname in order:
            os.rename(d / f".{fname}.tmp", d / fname)
            self.made.append(d / fname)
        return d

    def write_corpus(self) -> Path:
        """The corpus, in the model's concept folder: its own texts as they are (nothing written), the shipped ones
        copied in, or the supplied or written ones, with writing.json when there are writer's instructions to keep. A
        new name's definition is stored then (corpus.define)."""
        from ..concepts import corpus as C
        self.phase("writing")
        if self.source in ("own", "shipped"):
            out, d = self.plan["read"], Path(self.plan["read"]["path"])
            learned = out["concept"].get("learned") or {}
            self._corpus_stage(f"{'its own' if self.source == 'own' else 'shipped'} texts, written by "
                               f"{learned.get('writer') or 'unknown'}" + (f" ({learned['date']})" if learned.get("date") else ""))
            if self.source == "shipped":
                d = self._put_texts({n: (d / n).read_text() for n in TEXTS if (d / n).is_file()})
                self.log(f"shipped texts copied into {d}")
        else:
            if self.supplied:
                out = self.supplied
                learned = out["concept"].get("learned") or {}
                who = (f"written by {learned['writer']}" + (f" ({learned['date']})" if learned.get("date") else "")
                       if learned.get("writer") else "no record of who wrote them")
                self._corpus_stage(f"supplied from {out['log']['supplied']}, {who}")
            else:
                wr = W.Writer(self.server, self.writer_model, log=self.log)
                out = W.write(wr, self.name, self.description, on_stage=self._corpus_stage)
            d = self._put_texts({"concept.json": json.dumps(out["concept"], indent=1) + "\n",
                                 "pos.txt": "\n".join(out["pos"]) + "\n", "neg.txt": "\n".join(out["neg"]) + "\n",
                                 "task.json": json.dumps(out["task"], indent=1) + "\n",
                                 **({"writing.json": json.dumps(out["writing"], indent=1) + "\n"} if out.get("writing") else {})})
            self.defined = C.define(self.name, self.description)
        (self.dir / "corpus-log.json").write_text(json.dumps(out["log"], indent=1))
        source = {"own": "own", "shipped": "shipped", "corpus": "supplied"}.get(self.source)
        kind, contrast = out["concept"].get("kind") or "-", out["concept"].get("contrast") or "-"
        how = source or f"{out['log']['requests']} requests, {out['log']['seconds']}s"
        warnings = out["log"].get("warnings") or []          # existing texts: pair-rule breaks are kept, only reported
        for w in warnings:
            self.log(f"kept, though it breaks the writer's rules: {w}")
        self.log(f"corpus: {len(out['pos'])} pairs, {len(out['task']['items'])} items, kind {kind}, contrast: {contrast} ({how})")
        self.summarize("corpus", f"{len(out['pos'])} pairs, {len(out['task']['items'])} items · {kind} · contrast: {contrast}"
                                  + (f" · {source}" if source else "")
                                  + (f" · {len(warnings)} pair{'s' * (len(warnings) != 1)} break the writer's rules (kept)"
                                     if warnings else ""))
        self.state["corpus"] = dict(path=str(d), pairs=len(out["pos"]), items=len(out["task"]["items"]), kind=kind,
                                    contrast=contrast, example=dict(pos=out["pos"][0], neg=out["neg"][0]),
                                    stored=self.source in ("own", "shipped"), source=self.source, texts=C.texts_line(d))
        self.save()
        return d

    def measured(self) -> dict:
        """The GPU run's measurements as the job's result: ``steerable`` (a layer to steer at was found; False: learn
        stops, ``reason`` saying why, "can't be steered: ..."), ``evidence`` (``decide.evidence``'s lines), the push
        both ways in words, its curve, the nearest existing concepts, samples."""
        from ..concepts import peak_layer
        from ..concepts.summary import push_words
        prof = json.loads(Path(self.state["profile"]).read_text())
        stats = config.int_keys(prof["stats"])
        peak = peak_layer(stats)
        learn = prof.get("learn") or {}
        new = load_vector(self.model_name, self.name)
        existing = {}
        for v in list_vectors(self.model_name):
            if v["name"] == self.name:
                continue
            try:
                other = load_vector(self.model_name, v["name"])
            except ValueError:
                continue
            if peak in other.vectors:
                existing[v["name"]] = other.vectors[peak]
        ov = decide.overlap(new.vectors[peak], existing)
        why = learn.get("untestable") or (decide.NO_STEER_LAYER if learn.get("no_reader") else None)
        push = learn.get("push") or {}
        heldout, p_value = stats[peak]["heldout"], stats[peak].get("p_value")
        result = dict(steerable=why is None, reason=why,
                      evidence=decide.evidence(learn, heldout, p_value, peak, prof.get("reader")),
                      layer=learn.get("layer"), dose=learn.get("dose"), edge=learn.get("edge"),
                      heldout=heldout, p_value=p_value, peak_layer=peak,
                      items=(learn.get("selection") or {}).get("counts"), testable=(learn.get("selection") or {}).get("testable"),
                      curve=push.get("curve"), at_dose=push.get("at_dose"), away=push.get("away"),
                      push=push_words(push, self.name),
                      overlap=ov, samples=learn.get("samples"), profile=self.state["profile"])
        for line in result["evidence"]:
            self.log(line)
        if result["push"]["toward"]:
            self.log(" ".join(w for w in (result["push"]["toward"], result["push"]["away"]) if w))
        if why:
            self.log(why)
        return result

    def run_calibration(self) -> dict:
        """The chat calibration in conversation (``calibrate``: the default /steer dose and its good replies). An
        error is the step's summary and the result's ``error``; a cancel stops the job ("stopped")."""
        self.phase("calibrating", 0, 1)
        try:
            out = K.calibrate(self.name, self.model_name, server=self.server, log=self.log,
                              on_progress=lambda n, total: self.phase("calibrating", n, total))
        except Exception as exc:
            why = str(exc) or type(exc).__name__
            self.log(f"calibration: {why}")
            self.summarize("calibrate", f"not calibrated: {why}")
            return dict(error=why)
        self.summarize("calibrate", K.summary_text(out))
        return out

    def cleanup_failed(self) -> None:
        """A failed run leaves no half concept, and touches only what it made: the texts it wrote or was given move
        into the job directory and shipped texts it copied are removed; then the model's concept folder goes if this
        run made it (else only the vector, if this run wrote it: absent at the start, or rewritten since), and so does
        the definition it stored. The model's own texts from before, another model's folder and an earlier definition
        are never touched. Nothing happens once the profile is written (the run got far enough to keep), or when it
        made nothing: no texts and no GPU run."""
        if models.profile_path(self.model_name, self.name).exists() or not (self.made or self.child is not None):
            return
        if self.source == "shipped":
            for f in self.made:
                f.unlink(missing_ok=True)
        elif self.made:
            (self.dir / "concept").mkdir(exist_ok=True)
            for f in self.made:
                if f.exists():
                    shutil.move(str(f), str(self.dir / "concept" / f.name))
        d = models.concept_dir(self.model_name, self.name)
        if not self.existed and d.exists():
            shutil.rmtree(d)
            for up in (d.parent, d.parent.parent):          # no empty folder that reads as a checkpoint's data
                with contextlib.suppress(OSError):
                    up.rmdir()
        elif (v := find_vector(self.model_name, self.name)) is not None and v.stat().st_mtime_ns != self.vector_mtime:
            for f in (v, v.with_suffix(".json")):
                f.unlink(missing_ok=True)
        if self.defined is not None:
            self.defined.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                self.defined.parent.rmdir()


class CalibrateJob(Job):
    """``concept calibrate --job``: the calibration (chat requests only: nothing is unloaded) with one step that Pi's
    widget follows, and its summary and samples as the result. A failure or a cancel is the outcome; nothing else is
    touched."""
    KIND = "calibrate"
    WEIGHTS = {"calibrating": 1.0}
    STEPS = (("calibrate", "calibrate", ("calibrating",)),)
    ETA_FROM = 0.0                   # the calibration's own count of requests: its ETA holds from the first

    def __init__(self, name: str, model_name: str, job: str, origin: str = "cli", stamp: bool = True, **kwargs):
        """``stamp``: False for ``concept calibrate`` without ``--job``, whose stdout is the calibration's own lines."""
        self.name, self.kwargs, self.stamp = name, kwargs, stamp
        super().__init__(job, model_name, origin, name=name, description=None)

    def run(self) -> dict:
        def body():
            self.claim()
            self.phase("calibrating")
            try:
                out = K.calibrate(self.name, self.model_name, log=self.log,
                                  on_progress=lambda n, total: self.phase("calibrating", n, total), **self.kwargs)
            except BaseException as exc:
                self.summarize("calibrate", "stopped" if isinstance(exc, KeyboardInterrupt) else str(exc) or type(exc).__name__)
                raise
            self.state["result"] = dict(calibration=out)
            self.summarize("calibrate", K.summary_text(out))
        return self._run(body)

"""The jobs as commands: ``concept learn`` (``main``), ``concept calibrate`` (``run_calibrate_job``) and ``jlens learn``
(``main_lens_learn``), SIGTERM their cancel, each ending with its outcome on stdout.
"""
from __future__ import annotations

import json
import signal
from typing import Optional

from ..omlx import admin as omlx_admin
from . import calibrate as K
from .job import CalibrateJob, LearnJob
from .lens_job import LensLearnJob
from .plan import learn_plan
from .state import job_dir


def _sigterm(*_):
    raise KeyboardInterrupt("stopped")


def run_calibrate_job(name: str, model_name: str, job: str, origin: str = "cli", **kwargs) -> dict:
    """``concept calibrate --job``: a ``CalibrateJob``, SIGTERM its cancel."""
    signal.signal(signal.SIGTERM, _sigterm)
    return CalibrateJob(name, model_name, job, origin, **kwargs).run()


def served_model_path(model_name: str, admin=None) -> Optional[str]:
    """The checkpoint oMLX serves ``model_name`` from; None when it does not serve it (ServerError without a server)."""
    return next((m["model_path"] for m in (admin or omlx_admin.Admin()).admin_models()
                 if m.get("id") == model_name and m.get("model_path")), None)


def model_path_for(model_name: str) -> Optional[str]:
    """The same, None without a server too."""
    try:
        return served_model_path(model_name)
    except (omlx_admin.ServerError, OSError):
        return None


def _main(args, make_job, report, copy: bool = True, words=None) -> None:
    """A job from the command line: SIGTERM its cancel, the model path by default the served model's own (``copy``:
    the job loads one; else ``make_job(None)``); prints the outcome and ``report(state)`` as JSON, and exits 1 unless
    it is done. With ``words``: a job done prints ``words(state)`` instead, and one not done exits with its error."""
    signal.signal(signal.SIGTERM, _sigterm)
    try:                                            # what can be checked before a job folder exists
        if args.job:
            job_dir(args.job)
        model_path = None
        if copy:
            model_path = args.model or served_model_path(args.name)
            if not model_path:
                raise SystemExit(f"oMLX does not serve {args.name}; pass --model PATH")
        job = make_job(model_path)
    except (ValueError, omlx_admin.ServerError) as exc:
        raise SystemExit(f"error: {exc}")
    st = job.run()
    if words is not None:
        if st["outcome"] != "done":
            raise SystemExit(f"error: {st['error'] or st['outcome']}")
        print(words(st))
        return
    print(json.dumps(dict(job=st["job"], outcome=st["outcome"], error=st["error"], **report(st))))
    if st["outcome"] != "done":
        raise SystemExit(1)


def main_lens_learn(args) -> None:
    """``jlens learn NAME [PATH] --until HH:MM | --windows N [--fresh] [--job ID]``: a lens fit as a job
    (``LensLearnJob``)."""
    _main(args, lambda path: LensLearnJob(path, args.name, source=args.source, until=args.until, windows=args.windows,
                                          fresh=args.fresh, loader=args.loader, need_gb=args.need_gb, job=args.job,
                                          origin=args.origin),
          lambda st: dict(result=st.get("result")))


def learn_words(result: dict, concept: str, model_name: str) -> str:
    """How ``concept learn`` ends on the command line, in words, one fact per line: what was learned and its chat
    count (calibrate.summary_text, as ``concept calibrate`` prints it), then the next command."""
    cal = result.get("calibration") if isinstance(result.get("calibration"), dict) else None
    if not result.get("steerable"):
        why = (result.get("reason") or "").removeprefix("can't be steered: ")
        return "\n".join([f"{concept} can't be steered on {model_name}" + (f": {why}" if why else ""),
                          f"what was measured: metalscope concept show {concept} {model_name}"])
    lines = [f"learned {concept} on {model_name}" + (f": steer layer L{result['layer']}" if result.get("layer") is not None else "")]
    if not cal or cal.get("error") or cal.get("default_dose") is None:
        why = (cal or {}).get("error") or "no calibration"
        return "\n".join(lines + [f"in chat: not calibrated ({why})", f"next: metalscope concept calibrate {concept} {model_name}"])
    return "\n".join(lines + [f"in chat: {K.summary_text(cal)}",
                              f"next: metalscope steering on, then send the header X-Scope-Steer: {concept} (in Pi: /steer {concept})",
                              f"details: metalscope concept show {concept} {model_name}"])


def main(args) -> None:
    """``concept learn CONCEPT NAME``: a ``LearnJob`` on ``--engine`` (auto: mlx with ``--model``, else ``pick_engine``),
    its corpus written by ``--writer``, read from ``--corpus``, or CONCEPT's existing texts (learn_plan's ``source``). It ends with the chat count in
    words and the next command (``learn_words``); Pi reads the job's state, not this. The model is checked first, as
    Pi's routes do (every later check is about it), then ``learn_plan``'s, all before a job folder exists."""
    if not (args.model or args.url or args.job):     # a job Pi started: its server checked the model
        from .. import backends
        try:
            backends.require_served(args.name)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")
    # A checkpoint path given means a copy of it: mlx.
    plan = learn_plan(args.concept, args.name, args.description or "", args.corpus, args.writer,
                      "mlx" if args.engine == "auto" and args.model else args.engine, args.url)
    if plan["taken"]:
        raise SystemExit(f"error: {plan['taken']}")

    def make(model_path: Optional[str]) -> LearnJob:
        return LearnJob(plan, model_path, args.name, loader=args.loader, need_gb=args.need_gb,
                        sweep_layers=args.sweep_layers, seed=args.seed, job=args.job, origin=args.origin, url=args.url,
                        strengths=args.strengths or None)
    _main(args, make, None, copy=plan["engine"] == "mlx",
          words=lambda st: learn_words(st.get("result") or {}, plan["name"], args.name))

"""metalscope: trace, steer and read the models a local backend serves on Apple Silicon (in Pi: /scope help).

Backends (one serves a model at a time; starting one's model frees the other's):

    metalscope status [--json]             each backend: running or not, its models (and context), what is loaded
    metalscope omlx [MODEL] · omlx start|stop      start oMLX (MODEL: ds4 stopped first, MODEL loaded) · stop it
    metalscope ds4 [MODEL|GGUF] · ds4 start|stop   serve a ds4 model in the background at its full context (oMLX's
                                           model unloaded first) · stop it
    metalscope omlx install | uninstall    hook into oMLX.app (then metalscope omlx restart) · remove the hook
    metalscope omlx status | restart [--model ID] | load ID | unload ID | settings ID key=value
    metalscope ds4 install | serve GGUF [-- OPTS]   build ds4 with metalscope's patch · serve in the foreground
    metalscope uninstall                   back to stock: ds4 stopped, the oMLX hook and Pi package removed (data kept)

Front ends: this command line and Pi:

    metalscope pi install | remove         the Pi extension (/model lists both backends' models; picking one starts it)

Every day:

    metalscope on | off · steering on|off  trace every client's requests · let requests steer (Pi's /steer does)
    metalscope doctor [--app PATH] [--live]  check every piece is wired up (--live: trace and steer on each backend)
    metalscope bench [--backend omlx|ds4] [--quick] [--json FILE]   time the served model, tracing off and on
    metalscope traces | status | keep SIZE | auto | prune | pin|unpin ID   traces under a size budget (10GB default)
    metalscope jobs | jobs cancel [JOB]    the GPU job running or last, then the last ten · stop one
    metalscope model list | show NAME | link SERVED CKPT   checkpoints: the data every served copy shares

The J-lens (in Pi: /jlens):

    metalscope jlens status [NAME]      a model's J-lens, the other lenses ranked
    metalscope jlens on|off NAME        the J-lens on, or the logit lens alone
    metalscope jlens use NAME LENS      make lenses/LENS the model's J-lens
    metalscope jlens learn NAME [PATH] --until HH:MM | --windows N
                                        fit one as a job; the model is unloaded until then

Concepts (in Pi: /concept):

    metalscope concept list [NAME]      the concept library and NAME's calibrations
    metalscope concept show CONCEPT NAME   one profile
    metalscope concept learn CONCEPT NAME [--description TEXT] [--corpus DIR]
                                        make a concept: the model writes its texts (or DIR supplies them; a library
                                        CONCEPT needs no description), a GPU job labels it, then calibrates it
    metalscope concept calibrate CONCEPT NAME · delete CONCEPT NAME [--yes]
                                        its chat dose: the most good replies · its data on NAME to the trash
    metalscope concept pack CKPT DIR --base-model ID --license ID | pull DIR|REPO[@REV] SERVED
                                        share (docs/steering.md: Sharing concepts)

Research, by hand (Pi does not need these):

    metalscope experts tally [--rebuild|--clear]   the expert pages' tally
    metalscope experts layout [NAME] [--rebuild]   the expert map the dock's grids are laid out by
    metalscope vector list|norms NAME   steering vectors: stored, and what dose 1.0 means per layer

metalscope alone says what runs (status); metalscope help [COMMAND] is this list or one command's options.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

from . import client, config, doctor
from .omlx import admin as omlx_admin, install as omlx_install

def _set_mode(mode: str) -> None:
    # Every server that answers applies it at once; the settings file is shared, and a server re-reads it on its
    # next request, so with none answering the file is enough.
    told = False
    for url in client.candidates():
        try:
            client.Client(url).post("/scope/api/mode", {"mode": mode}, timeout=2)
            told = True
        except client.ServerError:
            pass
    if not told:
        config.save_settings(mode=mode)
    print(f"tracing: {mode}")


def cmd_status(args) -> None:
    """Both backends: running or not, their models with the context each runs at, what is loaded; tracing, memory.
    --json is the same for programs (Pi's /model reads it)."""
    from . import backends
    st = backends.status()
    if args.json:
        print(json.dumps(st))
        return
    o, d = st["omlx"], st["ds4"]
    ctx = lambda n: f"{n:,}" if n else "?"
    print(f"oMLX  {'running' if o['up'] else 'stopped'}  {o['url']}" + (f" · tracing {o['tracing']}" if o.get("tracing") else ""))
    for m in o["models"]:
        print(f"  {m['name']:<36} {ctx(m['ctx']):>9}" + ("  loaded" if m.get("loaded") else ""))
    state = (f"serving {d['serving']}" if d["ready"] else f"loading {d['serving'] or ''}".rstrip()) if d["up"] else \
        f"not serving: ds4-server pid {', '.join(map(str, d['orphaned']))} left running" if d["orphaned"] else "stopped"
    print(f"ds4   {state}  {d['url']}")
    for m in d["models"]:
        print(f"  {m['name']:<36} {ctx(m['ctx']):>9}" + ("  " + ("serving" if d["ready"] else "loading") if m["loaded"] else ""))
    if not d["models"]:
        print("  no models yet (metalscope ds4 PATH.gguf serves one)")
    mem = f"memory {st['memory_free']}% free · " if st["memory_free"] is not None else ""
    print(f"{mem}tracing {st['mode']} (global) · traces {config.traces_dir()}")
    on = [name for name, up in (("omlx", o["up"]), ("ds4", d["up"] or d["orphaned"])) if up]
    print("\nswitch  metalscope omlx MODEL · metalscope ds4 MODEL   (the other backend's model is freed first)")
    if on:
        print("stop    " + " · ".join(f"metalscope {name} stop" for name in on))
    print("Pi      /model lists every model here; picking one starts it")
    print("help    metalscope help (every command) · metalscope help ds4 (one command's options)")


def cmd_pi(args) -> None:
    """The Pi front end: `pi install` or `pi remove` the extension as a Pi package (listed by `pi list`)."""
    action = args.action
    try:
        subprocess.run(["pi", action, str(config.PI_PACKAGE)], check=True)
    except FileNotFoundError:
        raise SystemExit("pi is not on PATH: install Pi, then run this again")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"pi {action} failed ({exc})")
    if action == "install":
        # How Pi runs this metalscope (status --json for /model, starting a backend): this Python, whatever PATH holds.
        config.atomic_write(config.home() / "cli.json", json.dumps(dict(python=sys.executable)))
        print(f"installed the Pi package {config.PI_PACKAGE}; in Pi: /model lists oMLX's and ds4's models (picking one "
              "starts its backend), then /scope")
    else:
        print(f"removed the Pi package {config.PI_PACKAGE}")


def cmd_bench(args) -> None:
    from . import bench
    if not 0 <= (args.rest or 0) < float("inf"):       # also nan
        raise SystemExit("--rest: seconds, 0 or more")
    bench.main(args.backend, 1 if args.quick else args.repeats, [1000] if args.quick else args.contexts,
               args.max_tokens, args.json, args.model,
               0 if args.quick and args.rest is None else args.rest)     # None: bench's defaults


def cmd_jobs(args) -> None:
    """The jobs Pi and ``concept learn|calibrate`` start, from their state files (learn/state.py), as /scope jobs
    shows them: the running or last one, then the last ten; ``cancel`` stops one (its cleanup reloads the model)."""
    from .learn import state
    jobs = state.list_jobs()
    if args.action == "cancel":
        job = args.job or next((j["job"] for j in reversed(jobs) if j.get("outcome") is None), None)
        if job is None:
            raise SystemExit("no job is running")
        try:
            stopped = state.cancel(job)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")
        kind = next((j.get("kind") for j in jobs if j.get("job") == job), None)
        reloads = "" if kind == "calibrate" else ": the model it unloaded is reloaded when it has stopped"
        print(f"stopping {job}{reloads}" if stopped else f"{job} is not running")
        return
    if not jobs:
        print("no jobs yet: concept learn, concept calibrate and jlens learn start one "
              "(in Pi: /concept learn, /concept calibrate, /jlens learn)")
        return
    def why(j: dict) -> str:
        """A job that did not finish: its reason, clipped."""
        e = j.get("error") if j.get("outcome") not in (None, "done") else None
        return f" ({e if len(e) <= 60 else e[:59] + '…'})" if e else ""

    st = jobs[-1]
    if st.get("outcome") is None:
        step = f" {st['n']}/{st['total']}" if st.get("n") is not None and st.get("total") else ""
        left = f" · about {max(1, round((st['eta'] - time.time()) / 60))} min left" if st.get("eta") else ""
        per = f" · about {state.span(st['window_s'])} a window" if st.get("window_s") else ""
        print(f"running: {st['kind']} {st.get('name')} · {st.get('phase')}{step} · "
              f"{round(100 * (st.get('progress') or 0))}%{per}{left}  ({st['job']}; metalscope jobs cancel stops it)")
    else:
        print(f"last: {st['kind']} {st.get('name')}: {state.outcome_text(st)}{why(st)}  ({st['job']})")
    print(f"last {min(10, len(jobs))}:")
    for j in reversed(jobs[-10:]):
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(j["started"])) if j.get("started") else "?"
        print(f"  {when}  {j['kind']:13s} {str(j.get('name')):20s} {state.outcome_text(j):32s} {j['job']}{why(j)}")


def _lens_use(lens: str, name: str) -> dict:
    """``jlens.use`` through the server when it listens, which then reads the new lens at once (``live``); else on
    the files, read when the server next loads the model. Only for the server's own data home: with
    METALSCOPE_HOME set the files are another home's. ValueError for an unknown lens."""
    from . import jlens
    if not config.env("HOME"):
        try:
            return client.for_model(name).post("/scope/api/lens/use", dict(model=name, name=lens), timeout=120)
        except client.ServerError as exc:
            if exc.status == 409:
                raise ValueError(exc.detail or "refused") from None
    return jlens.use(lens, name)


def lens_use_lines(out: dict) -> list[str]:
    """What ``jlens use|on|off`` did (``jlens.use``, and ``live`` from the server: the lens the loaded model now reads).
    Pi's /jlens says more (jlens.ts)."""
    from . import jlens
    model, lens, logit = out["model"], out["lens"], out["lens"] == jlens.LOGIT
    was = f" (was {out['was']})" if out.get("was") and out["was"] != lens else ""
    lines = [f"{model} now reads through the logit lens{was}" if logit else f"{lens} is now the lens for {model}{was}"]
    if "live" not in out:
        lines.append(f"the server reads it when it next loads {model}")
    elif out["live"] not in (None, lens):
        lines.append(f"{lens} did not load on the server ({model} reads through {out['live']}): the server log says why")
    if logit:
        lines.append(f"no word vectors; metalscope jlens on {model} switches back")
    return lines


def lens_status_lines(st: dict) -> list[str]:
    """``jlens.status`` as text: the lens and where it came from, then the lenses that could be the model's, best
    held-out hit@10 first (none when the lens in use is the only one)."""
    cs = st.get("candidates") or []
    if st["installed"]:
        src = [x for x in (st.get("source"), st.get("hf_model") and f"hf_model {st['hf_model']}",
                           st.get("n_prompts") and f"{st['n_prompts']} prompts", st.get("layers") and f"{st['layers']} layers") if x]
        others = [s for s in st.get("served") or [] if s != st["model"]]
        lines = [f"{st['model']}: lens at {st['path']}", *(["  " + " · ".join(src)] if src else []),
                 *([f"  checkpoint {st['checkpoint']}" + (f", also served as {', '.join(others)}" if others else "")]
                   if st.get("checkpoint") not in (None, st["model"]) or others else [])]
    elif any(c.get("logit") and c["in_use"] for c in cs):
        lines = [f"{st['model']}: the logit lens (no J-lens in use: no word vectors)"]
    else:
        lines = [f"{st['model']}: no lens (lenses go in {st['lenses_dir']}/<name>/; metalscope jlens learn fits one)"]
    if not cs or (len(cs) == 1 and cs[0]["in_use"]):
        return lines
    lines.append("  lenses for it, best held-out hit@10 first (each on its own held-out windows):")
    for c in cs:
        score = ("not scored" if c.get("hit10") is None
                 else f"hit@10 {c['hit10']:.3f}" + (f" on {c['scored_on']}'s windows" if c.get("logit") and c.get("scored_on") else "")
                 + (f" vs logit lens {c['logit_hit10']:.3f}" if c.get("logit_hit10") is not None else ""))
        lines.append(f"    {'*' if c['in_use'] else '-'} {c['name']}: {c['why']} · {score}"
                     + (f" · {c['n_prompts']} windows" if c.get("learned_for") and c.get("n_prompts") else ""))
    return lines + ["  metalscope jlens use NAME LENS switches, jlens off|on NAME the J-lens"]


def cmd_jlens(args) -> None:
    from . import jlens, models
    if args.action == "status":
        names = [args.name] if args.name else sorted({d.parent.parent.name for d in jlens.lens_dirs()})
        if not names:
            print(f"no lenses under {models.models_dir()}/*/lenses (published ones for about 40 open models: neuronpedia/jacobian-lens "
                  "on Hugging Face, converted by scripts/convert_lens.py in a metalscope checkout)")
        for name in names:
            print("\n".join(lens_status_lines(jlens.status(name))))
        return
    if args.action == "learn":
        # Always a job: it unloads the served model until the deadline and reloads it, as Pi's /jlens learn does.
        if not args.until and not args.windows:
            raise SystemExit("--until HH:MM or --windows N: when to stop (the model is unloaded until then)")
        from .learn import command
        command.main_lens_learn(args)
        return
    # use, on and off: on and off are the server's names for the model's J-lens and for the logit lens alone
    # (jlens.JLENS, jlens.LOGIT).
    lens = args.lens if args.action == "use" else jlens.JLENS if args.action == "on" else jlens.LOGIT
    try:
        out = _lens_use(lens, args.name)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    print("\n".join(lens_use_lines(out)))


def _copy_args(sp) -> None:
    """The memory wait's override, for the jobs that load their own model copy (``gpulock.wait_available``)."""
    sp.add_argument("--need-gb", type=float, default=None,
                    help="free memory needed before loading (default: about the weights' size); the run refuses with less")


def cmd_steering(args) -> None:
    """``steering on|off``, the backends' switch: whether requests may steer at all."""
    data = config.save_settings(steer=args.state == "on")
    print(f"steering {'allowed' if data['steer'] else 'refused'} for requests with the {config.STEER_HEADER} header "
          f"({config.settings_path()})")


def cmd_traces(args) -> None:
    """The saved traces against their size budget: total and budget (alone or status), set the budget (keep),
    free space down to it now (prune), protect one trace (pin) or stop protecting it (unpin)."""
    from . import retention as R
    if args.action == "auto":
        if args.state:
            config.save_settings(trace_auto=args.state == "on")
        st = R.status()
        print(f"automatic cleanup: {'on' if st['auto'] else 'off'}")
        if st["auto"]:
            print(f"past the {R.fmt_size(st['budget'])} budget the server removes the oldest traces by itself")
        else:
            print("nothing is removed until you run metalscope traces prune (doctor says when it is over the budget)")
        return
    if args.action == "keep":
        try:
            n = R.parse_size(args.size)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")
        config.save_settings(trace_budget=n)
        print(f"budget: {R.fmt_size(n)}")
        st = R.status()
        print(f"traces: {st['count']:,}, {R.fmt_size(st['bytes'])}")
        if st["bytes"] > n:
            print("over the budget: the next saved trace frees space down to it" if st["auto"] else
                  "over the budget: automatic cleanup is off, so nothing goes until metalscope traces prune")
            print("to free it now: metalscope traces prune (--dry-run first shows what goes)")
        return
    if args.action == "prune":
        try:
            before = math.inf if args.all else time.time() - R.parse_age(args.older_than) if args.older_than else None
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")
        res = R.prune(dry_run=args.dry_run, before=before)
        print("\n".join(R.summary(res)))
        if args.dry_run:
            print("nothing deleted (--dry-run)")
        return
    if args.action in ("pin", "unpin"):
        from .store import read_index
        if args.action == "pin" and not any(e["id"] == args.id for e in read_index()):
            raise SystemExit(f"error: no saved trace {args.id}")
        ids = R.pin(args.id, on=args.action == "pin")
        print(f"{args.id}: {'pinned (never pruned)' if args.id in ids else 'not pinned'}")
        print(f"pinned traces: {len(ids)}")
        return
    st = R.status()
    print(f"traces: {st['count']:,}")
    print(f"total: {R.fmt_size(st['bytes'])}")
    print(f"budget: {R.fmt_size(st['budget'])} (metalscope traces keep SIZE changes it)")
    print(f"automatic cleanup: {'on' if st['auto'] else 'off'} (metalscope traces auto on|off)")
    print(f"pinned: {st['pinned']}")
    if st["bytes"] > st["budget"]:
        print(f"over the budget by {R.fmt_size(st['bytes'] - st['budget'])}: " +
              ("the next saved trace frees it" if st["auto"] else "nothing goes until metalscope traces prune"))
        print("automated runs go first, then chats, oldest first (metalscope traces prune --dry-run shows the totals)")
    print(f"folder: {config.traces_dir()}")


def cmd_experts(args) -> None:
    """``experts tally``: each checkpoint's expert tally (tally.py), its size; --rebuild counts it afresh from the saved
    traces, --clear deletes it. NAME: a served name or checkpoint (default: every tally, or every checkpoint with
    traced chats for --rebuild). ``experts layout``: the expert map (layout.py), built where there is none or with
    --rebuild; NAME as for the tally (default: every checkpoint with a routed saved reply)."""
    from . import models, tally
    from .retention import fmt_size
    from .store import read_index
    if args.action == "layout":
        return _experts_layout(args)
    if args.name:
        names = [models.checkpoint_of(args.name)]
    elif args.rebuild:
        names = sorted({models.checkpoint_of(e.get("model") or "") for e in read_index() if tally.counted(e)})
    else:
        names = sorted(p.parent.name for p in models.models_dir().glob(f"*/{tally.FILE}"))
    if not names:
        print("no expert tally yet: each traced chat grows one (metalscope experts tally --rebuild counts the saved ones)")
    for name in names:
        if args.clear:
            print(f"{name}: {'deleted' if tally.clear(name) else 'no tally'}")
            continue
        st = tally.rebuild(name) if args.rebuild else tally.size(name)
        print(f"{name}: {st['tokens']:,} tokens from {st['traces']:,} traces, {fmt_size(st['bytes'])}" if st["traces"]
              else f"{name}: no tally")
        if st["traces"]:
            print(f"  {st['path']} (from your chats: it stays on this machine; --clear deletes it)")


def _experts_layout(args) -> None:
    from . import layout, models
    from .store import read_index
    names = ([models.checkpoint_of(args.name)] if args.name else
             sorted({models.checkpoint_of(e.get("model") or "") for e in read_index() if e.get("experts")}))
    if not names:
        print("no routed reply saved yet: trace a chat on an MoE model first")
    for name in names:
        if not args.rebuild and layout.usable_layout(name):
            print(f"{name}: built already, {layout.layout_path(name)} (--rebuild builds it again)")
            continue
        try:
            lay = layout.build_layout(name)
        except ValueError as e:
            print(f"{name}: no expert map ({e})")
            continue
        print(f"{name}: {lay['tokens']:,} tokens from {lay['traces']:,} traces, {layout.layout_path(name)}")


def cmd_vector(args) -> None:
    """Steering vectors: the stored ones, and the strength table (what dose 1.0 means per layer)."""
    from . import models, steer, vectors
    if args.action == "list":
        vecs = vectors.list_vectors(args.name)
        if not vecs:
            print(f"no steering vectors for {args.name} under {models.model_dir(args.name) / 'concepts'}")
        for v in vecs:
            print(f"{v['name']:20s} layers {v['layers']}  width {v['width']}  {v.get('source') or ''}")
        return
    from .concepts import current_calibrations
    before = current_calibrations(args.name) if args.refresh else []
    try:
        table = steer.norm_table(args.name, refresh=args.refresh)
    except ValueError as exc:
        raise SystemExit(str(exc))
    stale = sorted(set(before) - set(current_calibrations(args.name))) if args.refresh else []
    print(f"{table['model']}: median residual norm per layer over {len(table['traces'])} traces "
          f"(dose 1.0 adds a vector of this norm){'; per stream' if table.get('streams') else ''}")
    for l in sorted(table["layers"]):
        per = table.get("streams", {}).get(l)
        print(f"  L{l:2d}  {table['layers'][l]:8.2f}" + (f"   streams {[round(x, 1) for x in per]}" if per else ""))
    note = steer.norms_note(table)
    if note:
        print(f"warning: {note}")
    if stale:
        served, n = config.short_name(args.name), len(stale)
        print(f"{n} chat calibration{'s' * (n != 1)} on {served} {'are' if n != 1 else 'is'} out of date now (new dose "
              f"units): metalscope concept calibrate {stale[0] if n == 1 else 'NAME'} {served}"
              + (f" ({', '.join(stale)})" if n != 1 else ""))


def _calibration_text(served: str, concept: str) -> str:
    """One served copy's chat calibration of a concept as a tag (calibrate.summary_text: "14/18 good @0.4"), "out of
    date" (the profile or the dose units changed after it), or "not calibrated" (an older calibration counts as none)."""
    from . import concepts as C
    from . import models
    from .learn import calibrate as K
    cal = C.calibration_of(served, concept)
    if cal is None:
        path = models.calibration_path(served, concept)
        return "out of date (the profile or the dose units changed after it)" if models.is_calibration(path) else "not calibrated"
    return K.summary_text(cal)


def _concept_list(args) -> None:
    """The concept library, and with NAME its own texts and each one's chat calibration on that served model."""
    from . import concepts as C, models
    if args.name:
        print(f"profiles of checkpoint {models.checkpoint_of(args.name)}; chat calibrations on {config.short_name(args.name)}")
    rows = C.list_concepts(args.name)
    for c in rows:
        on = ""
        if args.name:
            on = (_calibration_text(args.name, c["name"]) if models.profile_path(args.name, c["name"]).exists()
                  else "no profile")
            on = f"{on[:36]:36s} "
        print(f"{c['name']:18s} {c['title'][:28]:28s} {'task' if c['task'] else '    '} "
              f"{on}{c['description'][:80]}")
        print(f"{'':18s} {c['texts']}")
    if not args.name:                            # the models' own concepts are listed per model
        listed = {c["name"] for c in rows}
        more = {p.name for p in models.models_dir().glob("*/concepts/[!.]*") if p.name not in listed and (
            (p / "profile.json").is_file() or C.texts_dir(p.parent.parent.name, p.name))}
        if more:
            print(f"({len(more)} more on models: metalscope concept list NAME)")


def _concept_show(args) -> None:
    """A stored profile's summary, the concept's folder and its chat calibrations on every served copy."""
    from . import concepts as C, models, share, steer
    p = models.profile_path(args.name, args.concept)
    c = args.concept
    if not p.exists():
        if C.shipped_dir(c) or C.definition(c) or C.texts_dir(args.name, c):
            raise SystemExit(f"no profile at {p}; learn it: metalscope concept learn {c} {args.name}")
        raise SystemExit(f"no profile at {p}; learn it: metalscope concept learn {args.concept} {args.name} "
                         "--description TEXT (or --corpus DIR)")
    served = config.short_name(args.name)
    prof = config.read_json(p)
    print("\n".join(C.summary_lines(prof, C.calibration_of(served, args.concept))))
    if prof.get("package"):
        print(f"  pulled from {share.source_text(prof.get('source')) or prof['package']}; measured on "
              f"{(prof.get('measured_on') or {}).get('served') or 'the packer copy'}")
    print(f"  folder {models.concept_dir(args.name, c)}")
    where = C.texts_dir(args.name, c)
    print(f"  {C.texts_line(where) if where is not None else 'texts: none on this model'}")
    print(f"  in chat on {served}: {_calibration_text(served, args.concept)} ({models.calibration_path(served, args.concept)})")
    for other in models.calibrated_copies(args.name, args.concept):
        if other != served:
            print(f"  in chat on {other}: {_calibration_text(other, args.concept)}")
    note = steer.norms_note(config.read_json(models.norms_path(served)))
    if note:
        print(f"  warning: {note}")


def _concept_calibrate(args) -> None:
    """The chat dose, from graded conversations with the served model only (learn/calibrate.py): nothing is loaded or
    unloaded."""
    from . import backends
    from .learn import calibrate as K, command, state
    try:
        if args.job:
            state.job_dir(args.job)
        why = state.busy(exclude=args.job)       # one job at a time: a learn job unloads the model this talks to
        if why:
            raise ValueError(why)
        if not (args.url or args.job):           # a job Pi started: its server checked the model
            backends.require_served(args.name)
        server = client.Client(args.url) if args.url else client.for_model(args.name)
        # Without --job the run is a job of its own all the same (origin cli), so while it runs Pi sees a job running
        # and starts no learn job that would unload the model under it; its outcome, Ctrl-C included, ends it.
        job = args.job or state.new_job_id(args.concept, "calibrate", unique=True)
        st = command.run_calibrate_job(args.concept, args.name, job, origin=args.origin, max_tokens=args.max_tokens,
                                       server=server, stamp=bool(args.job))  # a foreground run prints as it always did
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}")
    if st["outcome"] != "done":
        raise SystemExit(1 if args.job else f"error: {st['error']}")
    if not args.job:
        print(K.summary_text(st["result"]["calibration"]))


def _concept_delete(args) -> None:
    """``concept delete CONCEPT NAME [--yes]``: the concept's folder for NAME's checkpoint to the trash (models.trash),
    or its description when no model has it, asked first; refused while a learn or calibrate job of it runs."""
    from . import models
    from .learn import state
    try:
        where = models.deletable(args.name, args.concept)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}")
    c, ckpt = args.concept, models.checkpoint_of(args.name)
    why = state.busy_with(c)
    if why:
        raise SystemExit(f"error: {why}")
    if models.is_definition(where):
        question, done = f"{c}'s description (no model has it)", f"{c}'s description deleted"
    else:
        copies = sorted(p.name for p in where.iterdir() if p.is_dir())
        holds = [w for w, f in (("texts", "concept.json"), ("texts", "pos.txt"), ("vector", "vector.safetensors"),
                                ("profile", "profile.json")) if (where / f).is_file()]
        holds = list(dict.fromkeys(holds)) + ([f"calibrations on {', '.join(copies)}"] if copies else [])
        question, done = f"{c} on {ckpt} (its {', '.join(holds) or 'empty folder'})", f"{c} deleted on {ckpt}"
    if not args.yes and input(f"y deletes {question}? [y/N] ").strip().lower() != "y":
        raise SystemExit("nothing deleted")
    why = state.busy_with(c)                     # a job started while it asked
    if why:
        raise SystemExit(f"error: {why}")
    try:
        print(f"{done}: moved to {models.trash(args.name, c)}")
    except OSError as exc:
        raise SystemExit(f"error: nothing moved: {exc}")


def _concept_learn(args) -> None:
    """The whole run: the served model writes the corpus (or --corpus supplies it), then one GPU job measures and labels
    it (learn/job.py)."""
    from .learn import command
    command.main(args)


def _concept_pack(args) -> None:
    """A package of a checkpoint's concepts (share.py): their texts, vectors and profiles, and each served copy's
    calibration as a reference."""
    from . import models, share
    try:
        m = share.pack(args.name, args.out_dir, args.base_model, args.concepts, license=args.license,
                       license_name=args.license_name, license_file=args.license_file, all_=args.all)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    print(f"packed {len(m['concepts'])} concepts of {m['checkpoint']} ({models.shape_text(m['shape'])}) for "
          f"{m['base_model']} into {args.out_dir}: {', '.join(c['name'] for c in m['concepts'])}")
    if m["left_out"]:
        print(f"  left out (not shipped with metalscope; name them, or --all): {', '.join(m['left_out'])}")
    if m["no_texts"]:
        print(f"  packed without texts (none on {m['checkpoint']}): {', '.join(m['no_texts'])}")
    print(f"  licence {m.get('license_name') or m['license']} (LICENSE"
          f"{'' if args.license_file else ': a note naming it; --license-file PATH copies the text in'})")
    print(f"  upload the folder yourself, one repo per base model (e.g. hf upload OWNER/REPO {args.out_dir})")


def _concept_pull(args) -> None:
    """A package's concepts placed for a served model (share.py): checked first (the shape, the files), then
    written; a concept the model has already is kept; then what to do next."""
    from . import share
    try:
        out = share.pull(args.source, args.name)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    name = config.short_name(args.name)
    if out["linked"]:
        print(f"{name} is now a copy of checkpoint {out['checkpoint']} (metalscope model show {name})")
    print(f"pulled {share.source_text(out['source'])} for {name} (checkpoint {out['checkpoint']})")
    print(f"  the package is for {out.get('base_model') or 'an unnamed base model'}; you serve {name}")
    for what, names in (("placed", out["placed"]), ("already there", out["unchanged"]),
                        ("kept yours (the package's is not placed)", out["kept"]),
                        ("kept (placed by another package)", out["kept_other"])):
        if names:
            print(f"  {what}: {', '.join(names)}")
    print("next:")
    print("\n".join(share.next_steps(out, name)))


def cmd_concept(args) -> None:
    {"list": _concept_list, "show": _concept_show, "calibrate": _concept_calibrate,
     "learn": _concept_learn, "pack": _concept_pack, "pull": _concept_pull,
     "delete": _concept_delete}[args.action](args)


def cmd_omlx(args) -> None:
    """The oMLX backend: its hook (install, uninstall) and its server (status, restart, load, unload, settings)."""
    if args.action == "install":
        omlx_install.install(Path(args.app))
        for line in omlx_install.untested(omlx_install.omlx_version(Path(args.app))):
            print(f"warning: {line}")
        print("restart oMLX to activate (metalscope omlx restart); then `metalscope on`")
        from . import backends
        for line in backends.model_lines(("omlx",)):
            print(line)
        return
    if args.action == "uninstall":
        print("restart oMLX" if omlx_install.uninstall(Path(args.app)) else "not installed")
        return
    if args.action in ("start", "stop"):
        from . import backends
        if args.action == "start":
            backends.omlx_start(args.model)
        elif not backends.omlx_stop():
            print("oMLX was not running")
        return
    admin = omlx_admin.Admin(base_url=args.url)
    try:
        if args.action == "status":
            omlx_admin.status(admin)
        else:           # settings, restart, load, unload: as a backend switch (backends.omlx_stop)
            from . import backends
            with backends.switching():
                if args.action == "settings":
                    # oMLX reloads or unloads a loaded model to apply them.
                    if any(m.get("id") == args.model_id and (m.get("loaded") or m.get("is_loading"))
                           for m in admin.admin_models()):
                        backends.gpu_free(f"change {args.model_id}'s settings")
                    omlx_admin.settings(admin, args.model_id, args.pairs)
                elif args.action == "restart":
                    backends.gpu_free("restart oMLX")
                    omlx_admin.restart(admin, model=args.model)
                else:
                    backends.gpu_free(f"{args.action} {args.model_id}")
                    (omlx_admin.load if args.action == "load" else omlx_admin.unload)(admin, args.model_id)
    except omlx_admin.ServerError as exc:
        raise SystemExit(f"error: {exc}")


def cmd_ds4(args) -> None:
    """The ds4 backend: building it (install), serving a model with it (start in the background, serve in the
    foreground), stopping it."""
    if args.action in ("start", "stop"):
        from . import backends
        if args.action == "start":
            backends.ds4_start(args.model, name=args.name, ctx=args.ctx, args=args.ds4_args, wait=not args.no_wait)
        elif not backends.ds4_stop():
            others = backends.other_ds4_servers()
            print("metalscope's ds4 was not running" + (
                f" (left alone: ds4-server pid {', '.join(map(str, others))}, not started by metalscope)" if others else ""))
        return
    if args.action == "install":
        from .ds4 import build
        build.install(args.dir, source=args.source, force=args.force)
        from . import backends
        for line in backends.model_lines(("ds4",)):
            print(line)
    elif args.action == "serve":
        from .backends import require_ds4_extra
        from .ds4.build import find_dir
        require_ds4_extra()                         # before the proxy's imports: one line, not a traceback
        from .ds4.proxy import serve
        serve(args.gguf, args.name, args.port, find_dir(args.ds4_dir), args.ds4_port, args.ds4_args or [])


def cmd_uninstall(args) -> None:
    """Back to stock engines: ds4 stopped, the oMLX hook removed (a running oMLX restarted without it), the Pi package
    removed. Deletes no data. Refused while a job runs; a step that fails is said, the rest still run (exit 1)."""
    from . import backends
    from .ds4 import build
    backends.gpu_free("uninstall")                  # before anything changes: names the job and how to stop it
    failed = []

    def fail(line: str) -> None:
        print(line)
        failed.append(line)
    try:
        if not backends.ds4_stop():
            print("metalscope's ds4: not running")
    except SystemExit as exc:
        fail(f"ds4: could not stop it ({exc})")
    try:
        removed = omlx_install.uninstall(omlx_install.DEFAULT_APP, log=lambda line: None)
    except SystemExit:                              # no oMLX.app and no manifest: no hook
        removed = False
    except (OSError, subprocess.CalledProcessError) as exc:
        removed = None
        fail(f"oMLX hook: could not remove it ({exc})")
    if removed is not None:
        print("oMLX hook removed" if removed else "oMLX hook: not installed")
        admin = omlx_admin.Admin()
        # A hook removed earlier leaves a server that still runs metalscope until it restarts.
        if admin.listening() and (removed or admin.trace_status() is not None):
            print("restarting oMLX without metalscope", flush=True)
            try:
                with backends.switching():
                    backends.gpu_free("restart oMLX")
                    omlx_admin.restart(admin, log=lambda line: None)
                print("oMLX restarted")
            except (SystemExit, omlx_admin.ServerError) as exc:
                fail(f"oMLX restart failed ({exc}): restart oMLX by hand to finish")
    listed = doctor.pi_listed()
    if not listed:
        print("Pi: not on PATH" if listed is None else "Pi package: not in `pi list`")
    else:
        try:
            cmd_pi(argparse.Namespace(action="remove"))
        except SystemExit as exc:
            fail(f"Pi package: {exc}")
    print(f"kept: your data in {doctor.tilde(str(config.home()))}, the ds4 build and its models in "
          f"{doctor.tilde(str(build.build_dir()))}")
    print("last step: uv tool uninstall metalscope")
    if failed:
        raise SystemExit(1)


def _copy_text(s: dict) -> str:
    """A served copy's data in a few words (models.served_copy)."""
    norms = f"norms from {s['norm_traces']} traces" if s["norms"] else "no norms (no unsteered traces measured)"
    cals = f"{len(s['calibrations'])} calibrations ({', '.join(s['calibrations'])})" if s["calibrations"] else "no calibrations"
    return f"{norms}, {cals}"


def cmd_model(args) -> None:
    """Checkpoints (models.py): list them, show one served name's, link a served name to one."""
    from . import models
    if args.action == "list":
        cs = models.checkpoints()
        if not cs:
            print(f"no checkpoints under {models.models_dir()} (an engine or the ds4 proxy records one when it opens)")
        for c in cs:
            print(f"{c['name']}: {models.shape_text(c['shape'])} · served as {', '.join(c['served']) or '(none linked)'} · "
                  f"lens {c.get('lens') or '(by name)'} · {c['vectors']} vectors, {c['profiles']} profiles")
            for s in c["copies"]:
                print(f"  {s['name']}: {_copy_text(s)}")
        return
    if args.action == "show":
        rec = models.record(args.name)
        print(f"{config.short_name(args.name)} is checkpoint {rec['name']} ({models.shape_text(rec['shape'])})")
        print(f"  served as {', '.join(rec['served']) or '(none linked)'}; lens {rec.get('lens') or '(by name)'}")
        print(f"  {len(models.vector_paths(args.name))} vectors, {len(models.profile_paths(args.name))} profiles "
              f"({models.model_dir(args.name)})")
        print("  each served copy's strength table and chat calibrations:")
        for name in dict.fromkeys(rec["served"] + [config.short_name(args.name)]):
            s = models.served_copy(name)
            print(f"    {name}: {_copy_text(s)} ({s['path']})")
        name = config.short_name(args.name)
        for c in models.link_offer(name)["candidates"][:3]:
            print(f"  same shape as {c['name']} ({c['concepts']} concept{'s' * (c['concepts'] != 1)}, {c['lenses']} "
                  f"lens{'es' * (c['lenses'] != 1)}): metalscope model link {name} {c['name']} if it is the same model")
        return
    try:
        rec = models.link(args.served, args.checkpoint)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    except OSError as exc:
        raise SystemExit(f"error: {exc} (run the link again once that is fixed)")
    print(f"{config.short_name(args.served)} is now a copy of {rec['name']} ({models.shape_text(rec['shape'])}): its vectors, "
          f"profiles and J-lens (the strength table and calibrations stay each served copy's own); "
          f"served as {', '.join(rec['served'])}")


def _model_cmd(sub, action: str, concept=False, name_help: str = "served model name", **kw):
    """A subcommand on one served model: its NAME, after a CONCEPT when ``concept`` (True, or that argument's help)."""
    sp = sub.add_parser(action, **kw)
    if concept:
        sp.add_argument("concept", help=None if concept is True else concept)
    sp.add_argument("name", help=name_help)
    return sp


def _job_args(sp, help: str = argparse.SUPPRESS) -> None:
    """A run as a job: its id (Pi's; default made up) and who started it (pi: Pi announces the result)."""
    sp.add_argument("--job", default=None, help=help)
    sp.add_argument("--origin", default="cli", choices=["cli", "pi"], help=argparse.SUPPRESS)


def _csv(cast):
    """An argparse type: a comma-separated list, empty items dropped."""
    def parse(text: str) -> list:
        return [cast(x.strip()) for x in text.split(",") if x.strip()]
    parse.__name__ = f"comma-separated {cast.__name__}"           # argparse's "invalid ... value"
    return parse


def _positive(text: str) -> int:
    """An argparse type: a whole number above 0."""
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{text} is not above 0")
    return n


class _Version(argparse.Action):
    """--version: the installed package's version (as a shared concept pack records it), read only when asked."""

    def __call__(self, parser, namespace, values, option_string=None):
        from .share import _version
        print(f"metalscope {_version()}")
        parser.exit()


def main(argv=None) -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")    # a terminal that is not UTF-8 shows ? for a mark, no traceback
    from .backends import DS4_SERVER_PORT
    from .ds4.build import DS4_REPO
    ap =argparse.ArgumentParser(prog="metalscope", usage="metalscope [-h] COMMAND ...", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action=_Version, nargs=0, help="the installed version")
    # The docstring lists the commands; prog keeps the root's usage out of each command's own.
    sub = ap.add_subparsers(required=True, metavar="COMMAND", help=argparse.SUPPRESS, prog="metalscope")
    sub.add_parser("on", help="trace every client's requests: experts, top 10 and both lenses",
                   description="Trace every client's requests: each token's experts and top 10, "
                               "and the logit lens with the J-lens beside it (metalscope jlens off NAME: the logit lens "
                               "alone).").set_defaults(func=lambda a: _set_mode("on"))
    sub.add_parser("off").set_defaults(func=lambda a: _set_mode("off"))
    p = sub.add_parser("status", help="what runs: each backend, its models and their context, what is loaded, memory")
    p.set_defaults(func=cmd_status)
    p.add_argument("--json", action="store_true", help="the same as JSON (Pi's /model reads it)")
    p = sub.add_parser("jobs", help="the GPU jobs: the running or last one, then the last ten; cancel stops one")
    p.set_defaults(func=cmd_jobs)
    p.add_argument("action", nargs="?", choices=["cancel"])
    p.add_argument("job", nargs="?", help="job id (default: the running job)")
    p = sub.add_parser("jlens", help="the Jacobian lens (J-lens): a model's lens (status), fitting a new lens (learn), "
                                     "switching (use), turning it off and on")
    p.set_defaults(func=cmd_jlens)
    lens_sub = p.add_subparsers(dest="action", required=True)
    lens_sub.add_parser("status", help="a model's J-lens, the other lenses ranked").add_argument(
        "name", nargs="?", help="served model name (default: every installed lens)")
    sp = _model_cmd(lens_sub, "learn", help="fit a Jacobian lens for a served model, as a job that unloads it until the deadline")
    sp.add_argument("source", nargs="?", default=None, help="a directory of text files or one .txt (default: the default corpus)")
    sp.add_argument("--until", default=None, help="start no window that would end after HH:MM (local time)")
    sp.add_argument("--windows", type=int, default=None, help="fit at most this many windows")
    sp.add_argument("--fresh", action="store_true", help="start over instead of resuming this model's fit on this corpus")
    sp.add_argument("--model", default=None, help="checkpoint path (default: the served model's own)")
    sp.add_argument("--loader", default="auto", choices=["auto", "mlx_lm", "omlx"])
    _copy_args(sp)
    _job_args(sp)
    sp = _model_cmd(lens_sub, "use", help="make a lens directory the served model's J-lens")
    sp.add_argument("lens", help="the lens directory's name under lenses/ (metalscope jlens status NAME lists them); "
                                 "metalscope jlens off|on NAME turns the J-lens off (the logit lens alone) or back on")
    _model_cmd(lens_sub, "off", help="turn the served model's J-lens off: traces read the logit lens alone")
    _model_cmd(lens_sub, "on", help="turn the served model's J-lens back on: the one it used last, else the best scored")
    p = sub.add_parser("traces", help="saved traces: total and budget, set the budget (keep), free space now (prune), "
                                      "protect one (pin)",
                       description="Traces are kept under a size budget (10GB unless kept otherwise). Past it, the "
                                   "oldest automated runs go first, then the oldest chats; pinned traces and each "
                                   "model's newest unsteered traces (what steering sizes its doses from) are kept.")
    p.set_defaults(func=cmd_traces)
    traces_sub = p.add_subparsers(dest="action")
    traces_sub.add_parser("status", help="total, count and budget (the same as traces alone)")
    traces_sub.add_parser("auto", help="automatic cleanup past the budget: on (the default) or off").add_argument(
        "state", nargs="?", choices=("on", "off"), help="on or off (none: say which)")
    traces_sub.add_parser("keep", help="set the budget").add_argument("size", help="e.g. 20GB, 500MB, 1TB (a bare number: GB)")
    prune = traces_sub.add_parser("prune", help="free space down to the budget now, or remove older traces",
                                  description="Pinned traces and each model's newest unsteered traces are kept "
                                              "whichever way you prune.")
    prune.add_argument("--dry-run", action="store_true", help="say what would go; delete nothing")
    which = prune.add_mutually_exclusive_group()
    which.add_argument("--older-than", metavar="AGE", help="remove every trace older than AGE (7d, 2w, 12h), whatever the budget")
    which.add_argument("--all", action="store_true", help="remove unprotected traces")
    for name, text in (("pin", "keep a trace whatever the budget"), ("unpin", "let a pinned trace be pruned again")):
        traces_sub.add_parser(name, help=text).add_argument("id", help="the trace id (its folder's name)")
    p = sub.add_parser("experts", help="the expert tally the dock's expert pages read, and the map its grids are laid out by")
    p.set_defaults(func=cmd_experts)
    experts_sub = p.add_subparsers(dest="action", required=True)
    sp = experts_sub.add_parser(
        "tally", help="each checkpoint's tally of which experts take which tokens, counted from the traced chats")
    sp.add_argument("name", nargs="?", help="served model name or checkpoint (default: every one)")
    how = sp.add_mutually_exclusive_group()
    how.add_argument("--rebuild", action="store_true", help="count it afresh from the saved traces")
    how.add_argument("--clear", action="store_true", help="delete it (it is made from your chats)")
    sp = experts_sub.add_parser("layout", help="the expert map the dock's grids are laid out by, from the newest traces")
    sp.add_argument("name", nargs="?", help="served model name or checkpoint (default: every one with a routed reply)")
    sp.add_argument("--rebuild", action="store_true", help="build it again where one is built")
    p = sub.add_parser("vector", help="steering vectors: list them, the strength table (norms)")
    p.set_defaults(func=cmd_vector)
    vector_sub = p.add_subparsers(dest="action", required=True)
    _model_cmd(vector_sub, "list")
    _model_cmd(vector_sub, "norms").add_argument("--refresh", action="store_true", help="recompute from the latest traces")
    p = sub.add_parser("doctor")
    p.set_defaults(func=doctor.cmd_doctor)
    p.add_argument("--app", default=str(omlx_install.DEFAULT_APP))
    p.add_argument("--live", action="store_true",
                   help="also a traced and a steered completion on each backend serving a model (after an engine update)")
    p = sub.add_parser("bench", help="time the served model: writing speed with tracing off and at each lens step, a "
                                     "one-word edit, the engram inspection (best effort: other requests to the server "
                                     "skew the numbers)")
    p.set_defaults(func=cmd_bench)
    p.add_argument("--backend", choices=["omlx", "ds4"], default=None, help="when both serve a model")
    p.add_argument("--model", default=None, help="when the server serves several (default: the first)")
    p.add_argument("--repeats", type=_positive, default=None, help="replies timed per setting, after one warm-up "
                                                                    "(median; default 3)")
    p.add_argument("--max-tokens", type=_positive, default=None, help="each timed reply's length (default 300)")
    p.add_argument("--contexts", type=_csv(int), default=None,
                   help="conversation lengths in tokens the engram inspection is timed after (default "
                        "1000,4000,16000; \"\": none)")
    p.add_argument("--rest", metavar="SECONDS", type=float, default=None,
                   help="idle before each measurement: the Mac slows under sustained load (default 60)")
    p.add_argument("--quick", action="store_true", help="a smoke run: --repeats 1 --contexts 1000 --rest 0")
    p.add_argument("--json", metavar="FILE", default=None, help="every raw number, each repeat included")
    p = sub.add_parser("concept", help="the concept library: list, show, learn a new one, calibrate its chat dose, "
                                           "pack concepts to share, pull a package")
    p.set_defaults(func=cmd_concept)
    concept_sub = p.add_subparsers(dest="action", required=True)
    concept_sub.add_parser("list", help="the concept library (shipped and yours); with NAME, each one's chat "
                                        "calibration on it").add_argument(
        "name", nargs="?", default=None, help="served model name")
    _model_cmd(concept_sub, "show", concept=True, help="summary of a stored profile")
    sp = _model_cmd(concept_sub, "learn", concept="the new concept's name (letters, digits, _ . -)",
                    name_help="served model name (vectors, norms and profile are stored under it; on oMLX unloaded during "
                              "the run if loaded, on ds4 chat refused while it is measured)",
                    help="a new concept from one line: the served model writes the corpus, then one GPU job "
                         "profiles, tests and labels it, and a kept one is calibrated in chat; a library concept "
                         "(concept list) needs no description")
    sp.add_argument("--description", default=None, help="one line: what the concept means (with --corpus, default its own; "
                                                        "a library concept has its own)")
    sp.add_argument("--corpus", metavar="DIR", default=None,
                    help="concept.json, pos.txt, neg.txt, task.json written elsewhere, instead of the model writing them: "
                         "copied into the model's folder; their structure must hold, pair-rule breaks are only reported")
    sp.add_argument("--engine", default="auto", choices=["auto", "mlx", "ds4"],
                    help="mlx: a copy of the checkpoint, oMLX's model unloaded meanwhile; ds4: the model ds4 serves as "
                         "NAME, chat refused while it is measured; auto: mlx with --model, else ds4 when it serves "
                         "NAME, else mlx")
    sp.add_argument("--url", default=None, help="the server the corpus writer and the calibration ask, and on ds4 its proxy "
                                                "(default: oMLX's, or the ds4 proxy's)")
    sp.add_argument("--model", default=None, help="checkpoint path to load (--engine mlx; default: the served model's own, "
                                                  "from oMLX)")
    sp.add_argument("--loader", default="auto", choices=["auto", "mlx_lm", "omlx"], help="auto: mlx_lm when it has the architecture, else oMLX's loader under its Python")
    sp.add_argument("--writer", default=None, help="served model that writes the corpus (default: the same as name; on ds4 "
                                                   "only that)")
    _copy_args(sp)
    # A string: the job hands it on to its GPU run's command line (learn/job.py).
    sp.add_argument("--sweep-layers", default="", help="layers for the selection sweep (default every 4th, or the model's own)")
    sp.add_argument("--strengths", type=_csv(float), default="",
                    help="doses for the selection sweep (default 0.1,0.2,0.4,0.8)")
    sp.add_argument("--seed", type=int, default=0)
    _job_args(sp)
    sp = concept_sub.add_parser("calibrate", help="the chat dose (/steer CONCEPT's default): conversations behind a "
                                                  "plain system prompt, graded by the served model")
    sp.add_argument("concept")
    sp.add_argument("name", help="served model name")
    sp.add_argument("--max-tokens", type=int, default=4000, help="per reply, thinking included (another budget runs its "
                                                                    "own plain conversations)")
    sp.add_argument("--url", default=None, help="the server to ask (default: the one serving NAME, oMLX's or the ds4 proxy's)")
    _job_args(sp, help="run as a job with a state Pi follows (/concept calibrate starts it so)")
    _model_cmd(concept_sub, "delete", concept=True, help="move the concept's texts, vector, profile and calibrations on NAME's "
               "checkpoint to ~/.metalscope/trash (moving the folder back restores it); when no model has it, its "
               "description").add_argument(
        "--yes", action="store_true", help="do not ask first")
    sp = _model_cmd(concept_sub, "pack", name_help="the checkpoint (or a served name of it) whose concepts to pack",
                    help="a package of a checkpoint's concepts to share: texts, vectors, profiles, its calibrations as "
                         "references, a README (no upload: that is yours)")
    sp.add_argument("out_dir", metavar="OUT_DIR", help="a new or empty folder, or an earlier package to replace")
    sp.add_argument("concepts", nargs="*", metavar="CONCEPT",
                    help="the concepts to pack (default: the ones shipped with metalscope that have a vector and "
                         "profile on the checkpoint; yours need naming, or --all)")
    sp.add_argument("--all", action="store_true", help="every concept with a vector and profile, yours too")
    sp.add_argument("--base-model", required=True, metavar="ID",
                    help="the base model's Hugging Face id, as its publisher names it (e.g. Qwen/Qwen3.6-35B-A3B): the "
                         "model card's base_model")
    sp.add_argument("--license", required=True, metavar="ID",
                    help="the package's licence as a Hugging Face licence id, usually the base model's (apache-2.0, "
                         "mit, other)")
    sp.add_argument("--license-name", default="", metavar="NAME",
                    help="with --license other: the licence's name (e.g. qwen-community-1.0)")
    sp.add_argument("--license-file", default=None, metavar="PATH",
                    help="the licence's text, copied in as LICENSE (default: a note naming the licence; needed with other)")
    sp = concept_sub.add_parser("pull", help="a package's concepts placed for a served model: only a shape mismatch "
                                             "refuses; calibrate them on your copy for your own dose")
    sp.add_argument("source", metavar="DIR|OWNER/REPO[@REV]",
                    help="a package folder, or a Hugging Face repo at a branch, tag or commit (default: its default "
                         "branch; needs huggingface_hub)")
    sp.add_argument("name", metavar="SERVED", help="the served model to use them with (as clients see it)")
    p = sub.add_parser("steering", help="allow requests to steer (the X-Scope-Steer header), or refuse them",
                       description="steering on|off: the backends' switch. The vector tools are metalscope vector; "
                                   "concepts are metalscope concept.")
    p.set_defaults(func=cmd_steering)
    p.add_argument("state", choices=["on", "off"])
    p = sub.add_parser("omlx", help="the oMLX backend: its hook (install, uninstall) and its server (status, restart, "
                                    "load/unload models, model settings)")
    p.set_defaults(func=cmd_omlx)
    omlx_sub = p.add_subparsers(dest="action", required=True)
    for name in ("install", "uninstall"):
        omlx_sub.add_parser(name, help="hook metalscope into oMLX.app's Python (restart oMLX after)" if name == "install"
                            else "remove the hook").add_argument("--app", default=str(omlx_install.DEFAULT_APP))
    sp = omlx_sub.add_parser("start", help="start oMLX's server; with MODEL, ds4 stopped first and MODEL loaded "
                                          "(metalscope omlx [MODEL] for short)")
    sp.add_argument("model", nargs="?", default=None)
    omlx_sub.add_parser("stop", help="stop oMLX's server")
    server = []          # the server's subcommands, which take its URL
    server.append(omlx_sub.add_parser("status"))
    sp = omlx_sub.add_parser("restart")
    sp.add_argument("--model", default=None, help="load this model once the server answers")
    server.append(sp)
    for name in ("load", "unload"):
        server.append(omlx_sub.add_parser(name))
        server[-1].add_argument("model_id")
    sp = omlx_sub.add_parser("settings")
    sp.add_argument("model_id")
    sp.add_argument("pairs", nargs="+", metavar="key=value")
    server.append(sp)
    for sp in server:
        sp.add_argument("--url", default=None, help="oMLX base URL (default from ~/.omlx/settings.json)")
    p = sub.add_parser("pi", help="the Pi front end: install or remove the extension")
    p.set_defaults(func=cmd_pi)
    p.add_argument("action", choices=["install", "remove"])
    sub.add_parser("uninstall", help="back to stock engines; your data stays",
                   description="Back to stock engines: stop ds4, remove the oMLX hook (restarting a running oMLX), "
                               "remove the Pi package. Deletes no data. Then: uv tool uninstall metalscope"
                   ).set_defaults(func=cmd_uninstall)
    p = sub.add_parser("model", help="checkpoints: list them, show a served name's, link a served name to one")
    p.set_defaults(func=cmd_model)
    model_sub = p.add_subparsers(dest="action", required=True)
    model_sub.add_parser("list", help="every checkpoint: its served names, shape, lens and what it holds")
    model_sub.add_parser("show", help="a served name's checkpoint and where its data is").add_argument("name")
    sp = model_sub.add_parser("link", help="make a served name a copy of a checkpoint (its vectors, profiles, J-lens)",
                              description="SERVED reads CHECKPOINT's vectors, profiles and J-lens from now on; its "
                                          "strength table and plain chats move to ~/.metalscope/models/CHECKPOINT/served/"
                                          "SERVED/ and its /jlens learn fits to models/CHECKPOINT/fits/. Chat "
                                          "calibrations do not move: calibrate it again under CHECKPOINT. Refused when "
                                          "their recorded shapes differ, when SERVED has data of its own (a checkpoint "
                                          "itself), while a job runs on SERVED, or when CHECKPOINT has a folder by the "
                                          "same name already. A link that stopped part way finishes when run again.")
    sp.add_argument("served", help="the served model's name (as clients see it)")
    sp.add_argument("checkpoint", help="the checkpoint's name (metalscope model list)")
    p = sub.add_parser("ds4", help="the ds4 backend (antirez/ds4): build it, serve a model with it")
    p.set_defaults(func=cmd_ds4)
    ds4_sub = p.add_subparsers(dest="action", required=True)
    sp = ds4_sub.add_parser("install", help="fetch ds4 at the commit metalscope's patch is for, apply it, build ds4-server",
                            description="Fetch antirez/ds4 at the commit metalscope's patch is made for, apply the patch "
                                        "and build ds4-server (git, make and Xcode's command-line tools). Done once; "
                                        "again only when metalscope's patch changed.")
    sp.add_argument("--dir", type=Path, default=None,
                    help="where to build (default ~/.local/share/metalscope/ds4, or METALSCOPE_DS4_DIR)")
    sp.add_argument("--source", default=DS4_REPO, help="ds4's repository, or a local clone of it")
    sp.add_argument("--force", action="store_true", help="build again even when the build is current")
    sp = ds4_sub.add_parser("start", help="serve a ds4 model in the background (metalscope ds4 [MODEL] for short)",
                            description="Serve a ds4 model in the background at its full context: oMLX's loaded model "
                                        "unloaded and a different ds4 model stopped first; waits until it answers. MODEL "
                                        "is a ds4 model's name (metalscope status) or a GGUF file; none: the one served "
                                        "last. Options after -- go to ds4-server and are kept for the model once it loads; -- alone clears them.")
    sp.add_argument("model", nargs="?", default=None)
    sp.add_argument("--name", default=None, help="for a GGUF file: the model id clients see (default: the file name and -ds4)")
    sp.add_argument("--ctx", type=int, default=None, help="context size (default: the model's maximum)")
    sp.add_argument("--no-wait", action="store_true", help="return once it is loading")
    ds4_sub.add_parser("stop", help="stop ds4 (the proxy and ds4-server)")
    sp = ds4_sub.add_parser("serve", help="start ds4-server and the metalscope proxy in front of it, in the foreground",
                            description="Start ds4-server (the metalscope build) on --ds4-port and the metalscope proxy "
                                        "on --port; options after -- go to ds4-server (e.g. -- --ctx 32768 --ssd-streaming).")
    sp.add_argument("gguf", type=Path)
    sp.add_argument("--name", default=None, help="the model id clients see (default: the file name and -ds4)")
    sp.add_argument("--port", type=int, default=urllib.parse.urlsplit(client.DS4_URL).port)
    sp.add_argument("--ds4-dir", type=Path, default=None,
                    help="a ds4 build with metalscope's patch (default: metalscope ds4 install's)")
    sp.add_argument("--ds4-port", type=int, default=DS4_SERVER_PORT)
    words, ds4_args = list(sys.argv[1:] if argv is None else argv), None      # None: no --, the remembered options
    # `metalscope` alone: what runs; `metalscope help [COMMAND ...]`: the overview, or that command's options.
    if not words:
        words = ["status"]
    elif words[0] == "help":                     # its options only: nothing after -- (that would be ds4-server's,
        rest = words[1:words.index("--")] if "--" in words else words[1:]   # and the command would run)
        words = [w for w in rest if w != "help"] + ["--help"]
    # `metalscope omlx` / `metalscope ds4 MODEL`: start (a word that is not one of their subcommands is the model).
    if words and words[0] in ("omlx", "ds4"):
        acts = (omlx_sub if words[0] == "omlx" else ds4_sub).choices
        if len(words) == 1 or (words[1] not in acts and words[1] not in ("-h", "--help")):
            words.insert(1, "start")
    # ds4 serve / start: what follows -- is ds4-server's, not ours (argparse would take its flags).
    if words[:1] == ["ds4"] and words[1:2] in (["serve"], ["start"]) and "--" in words:
        cut = words.index("--")
        words, ds4_args = words[:cut], words[cut + 1:]
    args = ap.parse_args(words)
    args.ds4_args = ds4_args
    config.private_home()
    try:
        args.func(args)
    except KeyboardInterrupt:           # Ctrl-C: the shell's exit code for it, no traceback
        raise SystemExit(130)


if __name__ == "__main__":
    main()

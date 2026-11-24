"""The GPU half of ``concept learn``: one model load, every measurement.

Run as a child process of the job (``python -m metalscope.learn.gpu ...``),
which holds the GPU lock and passes it down by inheriting its file
descriptor. Steps, on one engine (an MLX copy of the model, engines/mlx.py, or
the model ds4-server serves, engines/ds4.py):

1. presence: ``concepts.profile`` (held-out d′ per layer,
   the writer and carrier, the vector, saved under the concept's name);
2. items: every task item's unsteered P(target), averaged over its option
   rotations; ``decide.select_items`` picks and splits them. No item with
   room to move (the model already picks one side everywhere): the run stops
   here, the profile written with presence and ``learn.untestable``, why the
   concept can't be steered (``decide.untestable``);
3. selection sweep on the selection half (every usable item when too few for a test half): every 4th layer (the
   engine's ``steer_layers`` where it names them: Inkling 2's two) x a few doses,
   concept against two random controls; ``reader_layer`` gives the steer
   layer L and dose d. None (no layer and dose kept the answers well formed):
   the run stops here too (``learn.no_reader``);
4. the fluency ladder (``generation`` on four selection items): the edge,
   the test dose D (``generation_dose``: d, or half the edge), and samples;
5. test sweep at L on the test half and the favour items, at -D, D/4, D/2
   and D: the reported effect and the curve.

Writes a normal profile (``write_profile``) with a ``learn`` section, so
``/steer`` and the concept page read it like any other. Progress goes to
stdout as lines starting with ``@@`` followed by JSON, which the job turns
into its state file. Safety: MLX memory and cache limits, and a watchdog
that exits on critical memory pressure, swap growth or the job's death (on
ds4 the job's death only: the memory is ds4-server's).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import replace

from .. import config, gpulock
from . import decide

SELECT_STRENGTHS = (0.1, 0.2, 0.4, 0.8)
LADDER_ITEMS = 4
SEEDS = (1, 2)


def emit(**kw) -> None:
    print("@@" + json.dumps(kw), flush=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def progress(stage: str):
    last = [0.0]

    def cb(n, total):
        if time.monotonic() - last[0] > 2 or n == total:
            last[0] = time.monotonic()
            emit(stage=stage, n=n, total=total)
    return cb


def _limit_memory(limit_gb: float, cache_gb: float = 4.0) -> None:
    import mlx.core as mx
    for owner in (mx, getattr(mx, "metal", None)):
        if owner is not None and hasattr(owner, "set_memory_limit"):
            owner.set_memory_limit(int(limit_gb * gpulock.GB))
            owner.set_cache_limit(int(cache_gb * gpulock.GB))
            log(f"mlx memory limit {limit_gb:.0f} GB, cache limit {cache_gb:.0f} GB")
            return


def child_parser(prog: str, model_required: bool = True) -> argparse.ArgumentParser:
    """The arguments every GPU child takes from its job (``OmlxWindow.child_args``); a child adds its own.
    ``model_required`` False: a child that can run on the model ds4 serves (``--engine ds4``, no copy)."""
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("model_name")
    ap.add_argument("--model", required=model_required, default=None)
    ap.add_argument("--loader", default="mlx_lm", choices=["mlx_lm", "omlx"])
    ap.add_argument("--memory-limit-gb", type=float, default=None)
    return ap


def _stopped(*_):
    raise KeyboardInterrupt("stopped")


@contextmanager
def child(args):
    """What every GPU child runs in: the watchdog, MLX's memory limits, and a plain copy under oMLX's loader (no
    server patches). On ds4 (``args.engine``) the memory is ds4-server's: the watchdog only follows the job, and
    SIGTERM (a cancel) unwinds, so the engine removes its request files."""
    if getattr(args, "engine", "mlx") == "ds4":
        signal.signal(signal.SIGTERM, _stopped)
        dog = gpulock.Watchdog(log=log, pressure=lambda: None, swap=lambda: None)
    else:
        dog = gpulock.Watchdog(log=log)
    dog.start()
    if args.memory_limit_gb:
        _limit_memory(args.memory_limit_gb)
    if args.loader == "omlx":
        os.environ.setdefault("METALSCOPE_DISABLE", "1")
    yield
    dog.stop()


def sublist(task, idx: list[int]):
    from ..concepts import Task
    return Task(task.instruction, [task.items[i] for i in idx], task.letters)


def run(engine, name: str, seed: int = 0, sweep_layers: list[int] | None = None, strengths=SELECT_STRENGTHS) -> dict:
    """Every measurement of ``concept learn`` on ``engine``, stored under its checkpoint (models.py)."""
    from .. import concepts as C
    from ..steer import norm_table
    from ..vectors import load_vector

    model_name, tok = engine.checkpoint, engine.tok
    concept = C.load_concept(model_name, name)       # the model's own texts, the only ones it learns from
    if concept.task is None:
        raise SystemExit(f"{name} has no task.json")
    t0 = time.perf_counter()

    # 1. Presence.
    emit(stage="presence")
    prof = C.profile(engine, concept, log_fn=log)
    vec = load_vector(model_name, name)
    table = norm_table(engine.served)              # the served copy's own strength table
    stats = config.int_keys(prof["stats"])
    peak = C.peak_layer(stats)
    pv = stats[peak].get("p_value")
    emit(step="presence", summary=f"held-out d′ {stats[peak]['heldout']:.2f} at L{peak}"
         + (f" (p {pv:.3f})" if pv is not None else ""))

    # 2. The items' unsteered probabilities, then the selection.
    emit(stage="measuring the items")
    all_prompts = C.prompts_for(tok, concept.task)
    base = C.sweep(engine, vec, table, all_prompts, [], [], seeds=())
    probs = decide.item_probs([p["item"] for p in all_prompts], base["base_logodds"].tolist(), len(concept.task.items))
    sel = decide.select_items(probs, seed=seed)
    sel["probs"] = probs
    log(f"[{name}] items: {sel['counts']}; selection {len(sel['selection'])}, test {len(sel['test'])}, favour {len(sel['favour'])}"
        + ("" if sel["testable"] else " (push not testable)"))
    c = sel["counts"]
    emit(step="measuring the items", summary=f"{c['against'] + c['favour']} usable of {c['items']} ({c['against']} against, {c['favour']} favour)"
         f" · {c['saturated_high']} already chosen, {c['saturated_low']} never" + ("" if sel["testable"] else " · push not testable"))

    if not sel["testable"] and not sel["fallback"]:
        why = decide.untestable(c, name, engine.served)
        log(f"[{name}] {why}; no sweep, ladder or test")
        prof["learn"] = dict(selection=sel, seed=seed, untestable=why)
    else:
        measure_push(engine, concept, prof, vec, table, sel, seed, sweep_layers, strengths)
    prof["seconds"] = time.perf_counter() - t0
    out = C.write_profile(prof)
    log(f"[{name}] profile written to {out}")
    emit(stage="done", profile=str(out / "profile.json"))
    return prof


def measure_push(engine, concept, prof: dict, vec, table: dict, sel: dict, seed: int, sweep_layers, strengths) -> None:
    """Steps 3 to 5 of ``run`` into ``prof``: the selection sweep, the ladder, the test sweep."""
    from .. import concepts as C
    from ..concepts.run import sweep_inputs, sweep_json
    from ..concepts.summary import push_toward, push_words
    name, tok, n_layers = concept.name, engine.tok, engine.shape.n_layers

    # 3. The selection sweep: steer layer and dose.
    choose = sel["selection"] if sel["testable"] else sel["fallback"]
    sweep_layers = sweep_layers or engine.steer_layers or \
        [l for l in range(n_layers) if l % 4 == 0 and l + 1 < n_layers]
    sel_prompts = C.prompts_for(tok, sublist(concept.task, choose))
    emit(stage="selection sweep", n=0, total=len(sweep_layers) * len(strengths) * len(sel_prompts) * (1 + len(SEEDS)))
    sw = C.sweep(engine, vec, table, sel_prompts, sweep_layers, list(strengths), SEEDS, on_progress=progress("selection sweep"))
    rd = C.reader_layer(sw)
    prof["coarse"] = sweep_json(sw)
    prof["coarse"]["reader"] = prof["reader"] = rd
    prof["inputs"].update(sweep_inputs(table, sweep_layers, strengths, SEEDS, len(choose)))
    learn = dict(selection=sel, seed=seed, sweep_layers=sweep_layers)
    prof["learn"] = learn
    if rd["layer"] is None or rd["best"] is None:
        log(f"[{name}] {decide.NO_STEER_LAYER}: no ladder, no test")
        learn["no_reader"] = True
        emit(step="selection sweep", summary="no layer and dose kept the answers well formed: no steer layer")
    else:
        L, d = int(rd["layer"]), float(rd["best"]["strength"])
        log(f"[{name}] steer layer L{L} @ {d:g} (effect {rd['best']['effect']:+.2f} over control on {decide.picked_on(sel)})")
        emit(step="selection sweep", summary=f"steer at L{L}, dose {d:g} · {rd['best']['effect']:+.2f} log-odds over control on {len(choose)} items")

        # 4. The ladder: fluency edge, test dose, samples.
        gtask = sublist(concept.task, choose[:LADDER_ITEMS])
        gconcept = replace(concept, task=gtask)
        n_gen = len(gtask.items) * 3 + len(gtask.items) * len(C.LADDER)
        emit(stage="fluency ladder", n=0, total=n_gen)
        gen = C.generation(engine, gconcept, vec, table, L, d, ladder_items=LADDER_ITEMS, on_progress=progress("fluency ladder"))
        C.attach_generation(prof, gen)
        D = float(C.generation_dose(prof)["strength"])
        edge = (C.fluency_rungs(gen) or {}).get("edge")
        log(f"[{name}] fluency edge {edge}; test dose D = {D:g}")
        emit(step="fluency ladder", summary=(f"replies break from dose {edge:g}" if edge is not None else "no break up to dose 3.2")
             + f" · test dose {D:g}")
        learn["dose"], learn["layer"], learn["edge"] = D, L, edge

        # 5. The test sweep on held-out items.
        if sel["testable"]:
            idx = sel["test"] + sel["favour"]
            kinds_by_item = {i: "against" for i in sel["test"]} | {i: "favour" for i in sel["favour"]}
            test_prompts = C.prompts_for(tok, sublist(concept.task, idx))
            kinds = [kinds_by_item[idx[p["item"]]] for p in test_prompts]
            doses = [-D, D / 4, D / 2, D]
            emit(stage="test sweep", n=0, total=len(doses) * len(test_prompts) * (1 + len(SEEDS)))
            tw = C.sweep(engine, vec, table, test_prompts, [L], doses, SEEDS, on_progress=progress("test sweep"))
            test = sweep_json(tw, per_prompt=True)["prompts"]
            test.update(strengths=doses, base_mass=tw["base_mass"].tolist())
            learn["test"] = dict(test, kinds=kinds, items=idx)
            learn["push"] = decide.push_result(test, kinds, D)
            p, words = learn["push"]["at_dose"], push_words(learn["push"], name)
            plain = " ".join(w for w in (words["toward"], words["away"]) if w)
            log(f"[{name}] test half at D: effect {p['effect']:+.2f} log-odds over control. {plain}")
            toward = push_toward(learn["push"])[0]
            emit(step="test sweep", summary=f"{p['effect']:+.2f} log-odds over control" + (f" · {toward}" if toward else ""))
        else:
            emit(step="test sweep", summary="skipped: too few items had room to move")
        samples = []
        for rec in gen["items"][:2]:
            samples.append(dict(prompt=rec["prompt"], **{k: rec["replies"][k]["text"] for k in ("base", "concept", "control")}))
        learn["samples"] = samples


def main(argv=None) -> None:
    ap = child_parser("metalscope.learn.gpu", model_required=False)
    ap.add_argument("name")
    ap.add_argument("--engine", default="mlx", choices=["mlx", "ds4"],
                    help="mlx: a copy of --model; ds4: the model the ds4 proxy at --url serves as model_name")
    ap.add_argument("--url", default=None, help="the ds4 proxy (default client.DS4_URL)")
    ap.add_argument("--sweep-layers", default="")
    ap.add_argument("--strengths", default="", help="the selection sweep's doses, comma-separated (default SELECT_STRENGTHS)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.engine == "mlx" and not args.model:
        ap.error("--model is required with --engine mlx")
    with child(args):
        layers = [int(x) for x in args.sweep_layers.split(",") if x.strip()] or None
        strengths = tuple(float(x) for x in args.strengths.split(",") if x.strip()) or SELECT_STRENGTHS
        if args.engine == "ds4":
            from ..engines import EngineUnavailable
            from ..engines.ds4 import Ds4Engine
            emit(stage="connecting")
            try:
                engine = Ds4Engine.open(args.model_name, args.url)
            except EngineUnavailable as exc:
                raise SystemExit(str(exc)) from None
            try:
                run(engine, args.name, args.seed, layers, strengths)
            finally:
                engine.close()
            return
        from ..engines.mlx import MlxEngine
        emit(stage="loading the model")
        run(MlxEngine.load(args.model, args.model_name, args.loader), args.name, args.seed, layers, strengths)


if __name__ == "__main__":
    sys.exit(main())

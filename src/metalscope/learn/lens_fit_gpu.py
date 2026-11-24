"""The GPU half of ``/jlens learn``: fit a Jacobian lens for the served model on its own copy, window by window,
until a deadline.

Run as a child of the lens-learn job (``lens_job.LensLearnJob``), which holds the GPU lock and passes it down, as
``gpu`` is for a learn job. On one load of the model:

1. the corpus (``jlens.corpus.build``): the fit windows and the held-out ones;
2. the held-out windows read by the logit lens and by the lens in use, once: what the new lens is compared with;
3. the batch (output dims per backward pass): one pass of the deepest upcoming window at growing sizes, the
   fastest per dim whose peak stays under the memory cap;
4. the fit (``jfit.fit``), checkpointed in ``<home>/models/<checkpoint>/fits/lens-<model>-<corpus id>/``, so a
   later run on the same model and corpus resumes (``--fresh`` starts over). After every window: the held-out set scored (hit@10 in the
   workspace band), the lens exported with its scores (so a deadline or a cancel leaves a usable lens), an ``@@``
   line with the window, its minutes and the scores, and MLX's cache cleared (scoring leaves buffers of other
   shapes behind; kept, they push the next window's passes into swap). No window starts that would end after the
   deadline (the last one's time, +15%); when the first cannot, the run fails at once, nothing written.

The lens goes to ``<home>/models/<checkpoint>/lenses/<model>-learned-<date>/``, never into a lens the checkpoint
chose: its lens.json names the checkpoint (and ``learned_for``, the served model fitted on). The checkpoint reads it
once ``/jlens use`` chooses it (``jlens.use``), or with no choice recorded while it is the checkpoint's only lens.

Model-specific setup: the adapter's ``fit_setup`` after the load (Flash Next: oMLX's fast kernels off) and
``jfit.no_index_gradients`` (routing indices). The model runs in training mode while fitting (jfit) and in eval
mode while scoring; Inkling 2's fit forward switches its own kernels a pass at a time (adapters/inkling2.py), so its
scoring runs as served too.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from .. import config, gpulock, models
from ..adapters import fit_setup, layer_count, lens_sources
from ..engines.mlx import load_model
from ..jlens import corpus as LC
from .gpu import child, child_parser, emit, log
from .state import span

GROW = (1, 2, 4, 8, 16, 32, 64, 128)   # batch sizes tried, while the peak stays under the cap
PREFIX_CHUNK = 256                     # prefix tokens per forward: smaller keeps a deep window's peak down
REPEAT = 1.15                          # a window takes about as long as the last one, +15%
UPCOMING = 20                          # the batch is sized on the deepest of the next this many windows


# ---------------------------------------------------------------- the deadline

def parse_until(hhmm: str, now: Optional[float] = None) -> float:
    """The next HH:MM (local time) after ``now``, as a timestamp: 07:30 at 23:00 is tomorrow's. ValueError else."""
    try:
        h, m = (int(x) for x in hhmm.strip().split(":"))
        if not (0 <= h < 24 and 0 <= m < 60) or len(hhmm.strip().split(":")[1]) != 2:
            raise ValueError
    except ValueError:
        raise ValueError(f"not a time: {hhmm!r} (HH:MM, 24-hour)") from None
    base = dt.datetime.fromtimestamp(time.time() if now is None else now)
    t = base.replace(hour=h, minute=m, second=0, microsecond=0)
    if t <= base:
        t += dt.timedelta(days=1)
    return t.timestamp()


def should_stop(now: float, deadline: Optional[float], next_s: float) -> bool:
    """Whether a window that takes about ``next_s`` seconds would end after the deadline."""
    return deadline is not None and now + REPEAT * next_s > deadline


def windows_by(now: float, deadline: Optional[float], per_window_s: float) -> Optional[int]:
    """How many more windows fit before the deadline at ``per_window_s`` each (None without a deadline)."""
    if deadline is None:
        return None
    return max(0, int((deadline - now) / (REPEAT * max(per_window_s, 1.0))))


# ---------------------------------------------------------------- where things go

def fit_dir(model_name: str, corpus_id: str) -> Path:
    """The fit state of one model on one corpus: what a later run resumes."""
    return models.model_dir(model_name) / "fits" / f"lens-{config.short_name(model_name)}-{corpus_id}"


def lens_target(model_name: str, previous: Optional[str], fit_id: str, today: Optional[dt.date] = None) -> str:
    """The lens directory a fit exports to: the one it exported to before (a resumed fit keeps its lens) unless that
    one is now chosen (``/jlens use`` made it the checkpoint's choice) or is not this fit's; else
    ``<model>-learned-<date>``, with -2, -3.. when that is taken. Never a lens the checkpoint chose."""
    from .. import jlens
    if previous:
        d = jlens.lenses_dir(model_name) / previous
        if not d.exists() or (jlens.lens_meta(d).get("fit") == fit_id and not jlens.in_use(previous, model_name)):
            return previous
    base = f"{config.short_name(model_name)}-learned-{(today or dt.date.today()):%Y%m%d}"
    name, k = base, 2
    while (jlens.lenses_dir(model_name) / name).exists():
        name, k = f"{base}-{k}", k + 1
    return name


def eval_layers(n_layers: int) -> tuple[list[int], list[int]]:
    """Every 4th layer and the last source layer, and those in the paper's workspace band (38-92% of depth)."""
    layers = sorted(set(range(0, n_layers - 1, 4)) | {n_layers - 2})
    return layers, [l for l in layers if 0.38 * n_layers <= l <= 0.92 * n_layers] or layers     # a tiny model: all


# ---------------------------------------------------------------- sizing a fit (the job asks before it loads)

# A fit takes every layer but the last as a source while their running sums stay under this; past it, only the
# layers the lens is scored at (eval_layers). Flash Next's 4.6 GB fits; DeepSeek V4's would be
# 11.3 GB (42 x 4096 x 4 streams x 4096), twice over with a window's Jacobians beside them, on a 79 GB copy.
FULL_STATE_GB = 8.0


def _fit_shape(path: str) -> Optional[tuple[int, list[float]]]:
    """(hidden size, GB of each recorded layer's running sums: a [hidden x hidden * sources] float32 matrix, the
    sources its residual streams, or what the model's adapter counts, ``adapters.lens_sources``) from config.json;
    None if it cannot be read."""
    cfg = config.read_json(Path(path).expanduser() / "config.json")
    if cfg is None:
        return None
    t = cfg.get("text_config") or {}
    get = lambda k: t.get(k) or cfg.get(k)                  # text_config's, else the top level's
    d = get("hidden_size") or 0
    sources = lens_sources(cfg) or [int(get("hc_count") or get("hc_mult") or 1)] * (get("num_hidden_layers") or 0)
    return d, [d * d * n * 4 / gpulock.GB for n in sources]


def fit_sources(path: str) -> Optional[list[int]]:
    """The layers a fit of this checkpoint takes as sources: None for every layer but the last, or, when their sums
    would pass ``FULL_STATE_GB``, the layers the lens is scored at."""
    shape = _fit_shape(path)
    if shape is None or sum(shape[1][:-1]) <= FULL_STATE_GB:
        return None
    return eval_layers(len(shape[1]))[0]


def fit_state_gb(path: str) -> float:
    """The fit's running sums (one [width x source width] float32 matrix per source layer, which every window adds
    into), from config.json: 4.6 GB on Flash Next (47 layers x 2560 x 4 streams x 2560), 0.4 GB on Qwen3.5-2B, 3.0 GB
    on DeepSeek V4-Flash (its 12 scored layers, ``fit_sources``), 3.2 GB on Inkling 2 Mini (47 visits, 164 sources of
    2304 between them)."""
    shape = _fit_shape(path)
    if shape is None:
        return 0.0
    sources = fit_sources(path)
    return sum(shape[1][l] for l in sources) if sources is not None else sum(shape[1][:-1])


# The fit's copy of the weights against their size on disk, measured: the served Flash Next oQ4e-mtp (99 GB on
# disk) was 75.0 GB at a 1-dim pass with its 4.6 GB of sums, Flash Next 4-bit (104 GB) 79 GB once loaded with
# them: about 0.71 of the disk size either way (the MTP head and other unused tensors stay on disk). 0.75 leaves a
# margin. A small model's copy is a larger share.
FIT_WEIGHTS_SHARE = 0.75


def fit_need_gb(path: str) -> float:
    """Memory a lens fit needs free before it loads: its copy of the weights (``FIT_WEIGHTS_SHARE`` of their size on
    disk), the running sums and one window's Jacobians beside them (``fit_state_gb`` each), and 2 GB for a pass at
    the smallest batch; the child grows the batch only while a window's peak stays under the job's memory limit
    (``pick_batch``). GB as gpulock counts them, 2^30 bytes.

    The served oQ4e-mtp: 74 + 2 x 4.6 + 2 = 85 GB, against the measured 75 GB at 1 dim a pass and 95.5 GB for a
    window at 4. Counting the weights at their whole disk size (103 GB) refused a run that fit in the 99 GB a
    desktop session leaves free. Never less than 6 GB."""
    weights = gpulock.model_bytes(path) / gpulock.GB * FIT_WEIGHTS_SHARE
    return max(6.0, round(weights + 2 * fit_state_gb(path) + 2))


# A 128-token window is one backward pass per output dim (d_model of them), each about DIM_S x layers x d_model
# seconds, plus 3 min to export and score: a constant picked from three fits (Flash Next 4-bit, 48 layers x 2560:
# ~33 min; Qwen3.6-35B-A3B, 40 x 2048: 34 min; Qwen3.8-27B-oQ4e-mtp, 64 x 5120: 4.9 h), not a model of the pace. A
# rough guess: the dims a pass and the pace are known only once the job has loaded the model and timed a pass.
DIM_S = 8e-6
# The job's fixed minutes (unload, load, held-out scoring, batch sizing, reload): ~15 on Flash Next 4-bit's 111.5 GB
# of safetensors, scaled by the weights' size on disk.
FIXED_POINT_GB = 111.5


def fit_estimate(path: Optional[str]) -> Optional[dict]:
    """Minutes a window takes (``DIM_S``, from config.json's layers, or Inkling 2's layer visits, and d_model) and the
    job's fixed minutes (``FIXED_POINT_GB``). None without a readable config."""
    shape = _fit_shape(path) if path else None
    if not shape or not shape[0] or not shape[1]:
        return None
    d, layers = shape[0], len(shape[1])
    k = gpulock.model_bytes(path) / 1e9 / FIXED_POINT_GB
    return dict(d_model=d, layers=layers, min_per_window=round(d * DIM_S * layers * d / 60 + 3), overhead_min=round(3 + 12 * k, 1))


# ---------------------------------------------------------------- held-out scoring

def heldout_scores(model, tok, prompts: list[dict], jacobians, layers: list[int]) -> dict:
    """``{"layers": {layer: {"top1", "hit10"}}, "kinds": {kind: {layer: hit10}}}`` over the held-out windows: how
    often the model's own next token is the lens's top word (top1) or in its top 10 (hit10), for the J-lens
    (``jacobians``) or, None, the logit lens. A deep window's prefix runs first, uncaptured."""
    import mlx.core as mx
    import numpy as np
    from ..adapters import install
    from ..capture import CaptureConfig, CaptureSession
    from ..jlens.fit import _arrays, _make_cache
    from ..jlens import SKIP_FIRST_N_POSITIONS as SKIP
    if jacobians is not None:
        cfg = CaptureConfig(residual_norm=False, logits_topk=1, jlens=jacobians, jlens_topk=10, jlens_layers=layers)
        kind = "jlens_ids"
    else:
        cfg = CaptureConfig(residual_norm=False, logits_topk=1, logit_lens_topk=10, logit_lens_layers=layers)
        kind = "lens_ids"
    hits: dict = {l: [] for l in layers}
    by_kind: dict = {}
    was = model.training
    model.train(False)
    try:
        for p in prompts:
            ids = list(p["ids"])
            w0, skip = (len(ids) - int(p["window"]), 0) if p.get("window") else (0, SKIP)
            cache = _make_cache(model)
            for a in range(0, w0, 1024):
                model(mx.array([ids[a:min(w0, a + 1024)]]), cache=cache)
                mx.eval(_arrays([c.state for c in cache]))
            session = CaptureSession(cfg)
            install(session, model)
            try:
                model(mx.array([ids[w0:]]), cache=cache)
                mx.eval(session.pending_arrays())
                session.end_step()
                recs = {(r.kind, r.layer): r.value for r in session.drain()}
            finally:
                session.close()
            final = recs[("logit_ids", -1)][0, skip:, 0]
            for l in layers:
                lens = recs[(kind, l)][0, skip:]
                h = np.stack([lens[:, 0] == final, (lens == final[:, None]).any(-1)], 1)
                hits[l].append(h)
                by_kind.setdefault(p.get("kind", "text"), {}).setdefault(l, []).append(h[:, 1])
    finally:
        model.train(was)
    out = {}
    for l, h in hits.items():
        h = np.concatenate(h)
        out[l] = dict(top1=round(float(h[:, 0].mean()), 4), hit10=round(float(h[:, 1].mean()), 4))
    kinds = {k: {l: round(float(np.concatenate(v).mean()), 4) for l, v in ls.items()} for k, ls in by_kind.items()}
    return dict(layers=out, kinds=kinds)


def band(scores: dict, layers: list[int]) -> dict:
    """The workspace band's mean top1 and hit10, and hit10 per kind of window."""
    import numpy as np
    q = scores["layers"]
    ls = [l for l in layers if l in q]
    return dict(top1=round(float(np.mean([q[l]["top1"] for l in ls])), 4), hit10=round(float(np.mean([q[l]["hit10"] for l in ls])), 4),
                kinds={k: round(float(np.mean([v[l] for l in ls])), 4) for k, v in scores["kinds"].items()})


# ---------------------------------------------------------------- the batch

def pick_batch(model, tok, prompt: dict, cap_gb: float, max_seq_len: int, sizes=GROW, window_gb: float = 0.0,
               source_layers: Optional[list[int]] = None) -> dict:
    """Time one backward pass of ``prompt`` at growing batch sizes until the peak (or, from two sizes, its linear
    extrapolation) would pass ``cap_gb``; the fastest per output dim under the cap wins, else the smallest tried
    (the cap is a preference: the smallest still ran). ``{"B", "runs": [...], "d_out", "window_s"}``, window_s the
    estimated time of a window like ``prompt`` at that size.

    A probe computes only B rows of the window's Jacobians (``dim_limit``); a whole window also holds all of them
    until they go into the running sums, ``window_gb`` more (``fit_state_gb``). On the served Flash Next a
    4-dim probe peaked at 91.4 GB and the window at 95.5 GB, 4.1 of its 4.6 GB, so each peak is counted with it.

    A probe can pass the memory limit (MLX's is no hard cap: Qwen3.8-27B's 2-dim probe peaked at 96.4 GB under 91);
    the watchdog (swap growth, critical pressure) is the guard."""
    import mlx.core as mx
    from ..jlens.fit import fit
    runs = []
    # The process's first backward pass also builds its kernels: run the smallest size once before timing.
    fit(model, tok, [prompt], dim_batch=sizes[0], max_seq_len=max_seq_len, dim_limit=sizes[0], log=lambda _m: None,
        prefix_chunk=PREFIX_CHUNK, source_layers=source_layers)
    for B in sizes:
        if len(runs) >= 2:
            a, b = runs[-2], runs[-1]
            slope = (b["peak_gb"] - a["peak_gb"]) / max(1, b["B"] - a["B"])
            if b["peak_gb"] + slope * (B - b["B"]) + window_gb > cap_gb:
                break
        lens = fit(model, tok, [prompt], dim_batch=B, max_seq_len=max_seq_len, dim_limit=B, log=lambda _m: None,
                   prefix_chunk=PREFIX_CHUNK, source_layers=source_layers)
        st = lens.meta["last"]
        del lens
        mx.clear_cache()
        runs.append(dict(B=B, per_pass_s=st["per_pass_s"], prefix_s=st["prefix_s"], peak_gb=round(st["peak_gb"] * 1e9 / gpulock.GB, 1),
                         d_out=st["d_out"]))
        log(f"batch {B}: {st['per_pass_s']:.2f}s a pass, prefix {st['prefix_s']:.1f}s, peak {runs[-1]['peak_gb']:.1f} GB")
        if runs[-1]["peak_gb"] + window_gb > cap_gb or B >= st["d_out"]:
            break
    ok = [r for r in runs if r["peak_gb"] + window_gb <= cap_gb]
    best = min(ok, key=lambda r: (r["per_pass_s"] / r["B"], -r["B"])) if ok else runs[0]
    d_out = best["d_out"]
    return dict(B=best["B"], runs=runs, d_out=d_out, cap_gb=round(cap_gb, 1), peak_gb=round(best["peak_gb"] + window_gb, 1),
                per_pass_s=best["per_pass_s"], window_s=round(-(-d_out // best["B"]) * best["per_pass_s"] + best["prefix_s"], 1))


# ---------------------------------------------------------------- the fit

def _export(lens_dir: Path, J: dict, meta: dict) -> None:
    """The lens, written beside and renamed over: a process stopped mid-write leaves the last whole lens. ``meta``
    names the checkpoint it is for."""
    import mlx.core as mx
    if not meta.get("checkpoint"):
        raise ValueError("a lens names its checkpoint (lens.json checkpoint)")
    lens_dir.mkdir(parents=True, exist_ok=True)
    tmp = lens_dir / "lens.tmp.safetensors"
    mx.save_safetensors(str(tmp), {f"J.{l}": v for l, v in J.items()})
    os.replace(tmp, lens_dir / "lens.safetensors")
    first = next(iter(J.values()))
    config.atomic_write(lens_dir / "lens.json", json.dumps(dict(meta, d_model=int(first.shape[0]), layers=sorted(J)), indent=1))


def fit_text(w: dict) -> str:
    """The last window in a few words, for the fit line under its bar (which has the count): "35 min a window ·
    hit@10 0.214 vs logit 0.148, in use 0.207", held-out hit@10 in the workspace band."""
    m = w["minutes"]
    cmp = f", in use {w['compared_hit10']:.3f}" if w.get("compared_hit10") is not None else ""
    return (f"{m:.0f}" if m >= 10 else f"{m:.1f}") + f" min a window · hit@10 {w['hit10']:.3f} vs logit {w['logit_hit10']:.3f}{cmp}"


def run(model_path: str, model_name: str, *, loader: str = "mlx_lm", source: Optional[str] = None,
        deadline: Optional[float] = None, windows: Optional[int] = None, fresh: bool = False, seed: int = 0,
        window: int = LC.WINDOW, max_depth: int = LC.MAX_DEPTH, batch: Optional[int] = None, mem_cap_gb: Optional[float] = None,
        memory_limit_gb: Optional[float] = None, max_heldout: Optional[int] = None) -> dict:
    """The steps in the module docstring; the result (the ``done`` line's): the lens's name and path, the windows
    fitted in all and in this run, why it stopped (deadline, windows, corpus), the held-out scores, the batch."""
    import mlx.core as mx
    from .. import jlens
    from ..jlens import fit as jfit
    if deadline is None and not windows:
        raise SystemExit("give a deadline (--until HH:MM) or a number of windows (--windows N)")
    short = config.short_name(model_name)
    jfit.no_index_gradients()
    emit(stage="loading the model")
    model, tok = load_model(model_path, loader)
    off = fit_setup(model_path)
    if off:
        log(f"fused kernels off for the fit ({len(off)} switches)")
    n_layers = layer_count(model)

    emit(stage="building the corpus")
    corpus = LC.build(source, tok, seed=seed, window=window, max_depth=max_depth, log=log)
    fdir = fit_dir(model_name, corpus.id)
    done0 = 0 if fresh else jfit.checkpoint_done(fdir)
    held = corpus.heldout()[:max_heldout] if max_heldout else corpus.heldout()
    if len(held) < 1:
        raise SystemExit("no held-out window: the held-out documents are all under 48 tokens")
    log(f"corpus {corpus.id}: {corpus.description}; {len(corpus.held)} documents held out ({len(held)} windows)"
        + ("; a .txt gives shallow windows only (each paragraph's first tokens), so the lens learns nothing about deep positions"
           if corpus.shallow_only else ""))
    log(f"resuming the fit in {fdir} after {done0} windows" if done0 else f"a new fit in {fdir}")
    emit(step="building the corpus", summary=f"{corpus.description} · {len(held)} held-out windows"
         + (f" · resuming after {done0} windows" if done0 else ""))

    layers, ws = eval_layers(n_layers)
    sources = fit_sources(model_path)
    if sources is not None:
        log(f"fitting the {len(sources)} layers the lens is scored at ({', '.join(map(str, sources))}): "
            f"every layer's sums would pass {FULL_STATE_GB:.0f} GB")
    emit(stage="scoring the logit lens")
    logit = band(heldout_scores(model, tok, held, None, layers), ws)
    compared = None
    in_use = jlens.find_for(model_name)
    if in_use is not None:
        try:
            old = jlens.load(in_use)
            compared = dict(lens=in_use.name, **band(heldout_scores(model, tok, held, old.jacobians, layers), ws))
            del old
        except Exception as exc:                  # a lens for other weights (wrong width or layers): no comparison
            log(f"could not score {in_use.name} on the held-out windows: {exc}")
        mx.clear_cache()
    emit(step="scoring the logit lens", summary=f"held-out hit@10 in the workspace band: logit lens {logit['hit10']:.3f}"
         + (f", {compared['lens']} (in use) {compared['hit10']:.3f}" if compared else ""))

    emit(stage="sizing the batch")
    wins = corpus.windows(done0 + (windows or LC.MAX_WINDOWS))
    if done0 >= len(wins):
        raise SystemExit(f"this corpus has {len(wins)} windows and all are fitted: --fresh starts over")
    upcoming = [wins[i] for i in range(done0, min(len(wins), done0 + UPCOMING))]
    deepest = max(upcoming, key=lambda w: len(w["ids"]))
    # The cap: given, else under the job's memory limit, else under what is free now.
    cap = mem_cap_gb or (memory_limit_gb - 2 if memory_limit_gb else max(4.0, (gpulock.available() or 0) / gpulock.GB - 8))
    if batch:
        probe = dict(B=batch, runs=[], d_out=None, cap_gb=cap, window_s=0.0, peak_gb=None, per_pass_s=None)
        emit(step="sizing the batch", summary=f"{batch} output dims a pass (given)")
    else:
        probe = pick_batch(model, tok, deepest, cap, window, window_gb=fit_state_gb(model_path), source_layers=sources)
        a_window = f"about {span(probe['window_s'])} a window"
        log(f"{a_window} ({probe['d_out']} dims, {probe['B']} a pass)")
        emit(step="sizing the batch", summary=f"{probe['B']} output dims a pass · {probe['per_pass_s']:.2f}s a pass · peak "
             f"{probe['peak_gb']:.0f} GB (cap {cap:.0f}) · {a_window} {deepest.get('depth', 0)} tokens deep")
        # No window fits before the deadline: the job fails with that, before anything is written.
        none_fits = should_stop(time.time(), deadline, probe["window_s"])
        emit(window_s=probe["window_s"], none_fits=none_fits)
        if none_fits:
            raise SystemExit(f"{a_window}: none fits before the deadline")
    B = probe["B"]
    if fresh and fdir.exists():
        log(f"--fresh: starting over (removing {fdir})")
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True, exist_ok=True)
    meta_path = fdir / "fit.json"
    fmeta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(model=short, corpus=corpus.info(), started=time.time())
    fmeta["lens"] = lens_target(model_name, fmeta.get("lens"), fdir.name)
    config.atomic_write(meta_path, json.dumps(fmeta, indent=1))
    lens_dir = jlens.lenses_dir(model_name) / fmeta["lens"]
    log(f"exporting to {lens_dir} after every window (the model reads it from its next load if it is its only lens; else /jlens use {fmeta['lens']})")
    hf = jlens.lens_meta(in_use).get("hf_model") if in_use else None

    st = dict(last_s=probe["window_s"], t=time.time(), n=0, result=None)

    def planned(done: int) -> int:
        """Windows in all by the end of this run: the count asked for, the ones the deadline leaves room for at the
        last window's pace, the corpus's, whichever is fewest."""
        ends = [len(wins)] + ([done0 + windows] if windows else [])
        if deadline is not None:
            ends.append(done + windows_by(time.time(), deadline, st["last_s"]))
        return max(done, min(ends))

    def stop() -> bool:
        if windows and st["n"] >= windows:
            st["stopped"] = "windows"
            return True
        if should_stop(time.time(), deadline, st["last_s"]):
            st["stopped"] = "deadline"
            return True
        return False

    def after(sums, done):
        w = wins[done - 1]
        J = {l: (s / done).astype(mx.float16) for l, s in sums.items()}
        q = band(heldout_scores(model, tok, held, J, layers), ws)
        heldout = dict(hit10=q["hit10"], top1=q["top1"], kinds=q["kinds"], logit_hit10=logit["hit10"], logit_top1=logit["top1"],
                       logit_kinds=logit["kinds"], band=ws, windows=len(held), compared=compared)
        lens_meta = dict(source=f"metalscope /jlens learn, {done} windows of {corpus.description}", n_prompts=done,
                         max_seq_len=window, dim_batch=B, learned_for=short, checkpoint=models.checkpoint_of(short),
                         fitted=time.strftime("%Y-%m-%d"),
                         fit=fdir.name, corpus=corpus.info(), heldout=heldout, **({"hf_model": hf} if hf else {}))
        _export(lens_dir, J, lens_meta)
        del J
        now = time.time()
        st["last_s"], st["t"], st["n"] = now - st["t"], now, st["n"] + 1
        ev = dict(n=done, of=planned(done), approx=deadline is not None, minutes=round(st["last_s"] / 60, 2), hit10=q["hit10"],
                  logit_hit10=logit["hit10"], compared=compared["lens"] if compared else None, compared_hit10=compared["hit10"] if compared else None,
                  kind=w.get("kind"), depth=w.get("depth", 0), lens=lens_dir.name, path=str(lens_dir), B=B,
                  peak_gb=round(mx.get_peak_memory() / gpulock.GB, 1))
        st["result"] = ev
        # "window 3/~11: 35 min a window · hit@10 ...": ~ when the deadline may end it sooner.
        log(f"window {done}/{'~' if deadline is not None else ''}{ev['of']}: {fit_text(ev)} · {ev['kind']} at depth {ev['depth']} "
            f"· peak {ev['peak_gb']} GB")
        emit(window=ev)
        emit(stage="fitting", n=done, total=ev["of"])
        mx.clear_cache()

    emit(stage="fitting", n=done0, total=planned(done0))
    jfit.fit(model, tok, wins, dim_batch=B, max_seq_len=window, checkpoint=fdir, log=log, on_prompt=after, should_stop=stop,
             prefix_chunk=PREFIX_CHUNK, source_layers=sources)
    # The window count also ends the list, so jfit can run out of windows before stop() is asked: count it here.
    stopped = "windows" if windows and st["n"] >= windows else st.get("stopped") or "corpus"
    done = jfit.checkpoint_done(fdir)
    why = {"deadline": "at the deadline", "windows": f"after the {windows} asked for", "corpus": "at the end of the corpus"}[stopped]
    result = dict(lens=lens_dir.name, path=str(lens_dir), windows=done, this_run=st["n"], stopped=stopped,
                  heldout=jlens.lens_meta(lens_dir).get("heldout"), last=st["result"], corpus=corpus.info(), batch=B, model=short)
    emit(step="fitting", summary=(f"{st['n']} window{'' if st['n'] == 1 else 's'} this run, {done} in all · stopped {why}"
                                  + (f" · held-out hit@10 {result['heldout']['hit10']:.3f} vs logit lens {logit['hit10']:.3f}"
                                     if result["heldout"] else "")))
    emit(stage="done", result=result)
    return result


def main(argv=None) -> None:
    ap = child_parser("metalscope.learn.lens_fit_gpu")
    ap.add_argument("--source", default=None, help="a directory of text files or one .txt (default: the default corpus)")
    ap.add_argument("--deadline", type=float, default=None, help="start no window that would end after this timestamp")
    ap.add_argument("--windows", type=int, default=None, help="fit at most this many windows this run")
    ap.add_argument("--fresh", action="store_true", help="start over instead of resuming this model's fit on this corpus")
    args = ap.parse_args(argv)
    with child(args):
        run(args.model, args.model_name, loader=args.loader, source=args.source, deadline=args.deadline,
            windows=args.windows, fresh=args.fresh, memory_limit_gb=args.memory_limit_gb)


if __name__ == "__main__":
    sys.exit(main())

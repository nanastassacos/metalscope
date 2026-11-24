"""The measurements that run the model for ``concept learn``'s GPU stage (learn/gpu.py), on any engine (engines/):
one prompt per call.

``sweep`` and ``reader_layer`` (where injecting the direction moves the
choice) and open ``generation`` with its dose ladder,
fluency signals and the edge against the unsteered rung (``fluency_rungs``).
The choice is read from the
option letters' log-probabilities (``engine.logprobs``), replies from
``engine.generate``.
"""

from __future__ import annotations

import itertools
import re
import time
from typing import Callable, Optional

import numpy as np

from ..steer import control_direction, push_layers
from ..vectors import Vector, unit
from .corpus import Concept, Task, option_letters, render_body, render_item

MASS_FLOOR = 0.8              # a dose is "well formed" while the letter mass keeps 80% of its baseline
METRICS = ("logodds", "mass", "p", "chosen")  # what ``_choice`` reads of a prompt, in its order


# ---------------------------------------------------------------- the choice and the adds

def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.exp(x - m).sum()))


def choice_scores(lp: np.ndarray, target_slot: int) -> tuple[float, float, float]:
    """From the option letters' log-probabilities at the answer position (slot
    order, ``engine.logprobs``): log-odds of the target letter against the
    other option letters (renormalised within the options), the probability
    mass on all option letters (how well formed the answer still is) and the
    target letter's probability."""
    sel = np.asarray(lp, np.float64)
    others = np.concatenate([sel[:target_slot], sel[target_slot + 1:]])
    return float(sel[target_slot] - _logsumexp(others)), float(np.exp(sel).sum()), float(np.exp(sel[target_slot]))


def target_chosen(lp: np.ndarray, target_slot: int) -> bool:
    """Whether the target is the most likely of the option letters (argmax
    over them): the answer the model would choose. The mean log-odds can rise
    while this rate falls (fluffy's reader on Qwen3.5-2B),
    so the sweep records both."""
    return int(np.argmax(lp)) == target_slot


def adds_for(vec: Vector, layer: int, strength: float, table: dict,
             dirs: Optional[dict[int, np.ndarray]] = None) -> dict[int, np.ndarray]:
    """A push at ``layer``, as the steering header's: at each of its layers (``layer``, and the rest of its block for
    a vector with a ``block``: steer.push_layers) ``strength`` times that layer's typical residual norm along
    ``vec``'s direction there (``dirs``: a control's instead, control_direction over those layers)."""
    dirs = vec.vectors if dirs is None else dirs
    out = {}
    for l in push_layers(vec.meta, [layer]):
        if l not in table["layers"]:
            raise ValueError(f"no residual norm for layer {l} in the strength table")
        out[l] = (strength * table["layers"][l] * unit(dirs[l])).astype(np.float32)
    return out


def _choice(engine, pr: dict, adds: Optional[dict]) -> tuple[float, float, float, bool]:
    """A rendered prompt's (log-odds, letter mass, target p, chosen) under ``adds``."""
    lp = engine.logprobs(pr["ids"], pr["letters"], adds)
    return (*choice_scores(lp, pr["slot"]), target_chosen(lp, pr["slot"]))


def _ticker(on_progress: Optional[Callable[[int, int], None]], total: int) -> Callable[[], None]:
    """Called after each forward: ``on_progress(n, total)`` with n counting from 1."""
    count = itertools.count(1)
    return (lambda: on_progress(next(count), total)) if on_progress else (lambda: None)


# ---------------------------------------------------------------- the sweep (reader)

def sweep(engine, vec: Vector, table: dict, prompts: list[dict], layers: list[int], strengths: list[float],
          seeds: tuple[int, ...] = (1, 2), on_progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Steer every prompt at every (layer, strength) with the concept direction
    and with norm-matched random controls (one per seed), reading the target
    log-odds, the option-letter mass and whether the target is the chosen
    (most likely) option letter at the answer position. Returns the baseline
    per prompt, the determinism of two baseline forwards, and ``concept`` /
    ``control`` arrays ``[layers, strengths, prompts]`` (the control's
    ``chosen`` is the share of seeds under which the target was chosen)."""
    L, S, P = len(layers), len(strengths), len(prompts)
    res = dict(layers=list(layers), strengths=list(strengths), n_prompts=P, seeds=list(seeds),
               **{f"base_{m}": np.zeros(P) for m in METRICS},
               **{f"{k}_{m}": np.full((L, S, P), np.nan) for k in ("concept", "control") for m in METRICS},
               items=[pr["item"] for pr in prompts], shifts=[pr["shift"] for pr in prompts])
    controls = {s: control_direction(vec, push_layers(vec.meta, layers), seed=s) for s in seeds}
    t0 = time.perf_counter()
    for pi, pr in enumerate(prompts):
        res["base_logodds"][pi], res["base_mass"][pi], res["base_p"][pi], res["base_chosen"][pi] = _choice(engine, pr, None)
    res["determinism"] = abs(_choice(engine, prompts[0], None)[0] - res["base_logodds"][0])
    tick = _ticker(on_progress, L * S * P * (1 + len(seeds)))

    def measure(adds) -> np.ndarray:            # [prompts, METRICS]
        vals = np.zeros((P, len(METRICS)))
        for pi, pr in enumerate(prompts):
            vals[pi] = _choice(engine, pr, adds)
            tick()
        return vals

    for li, l in enumerate(layers):
        for si, s in enumerate(strengths):
            c = measure(adds_for(vec, l, s, table))
            k = np.mean([measure(adds_for(vec, l, s, table, controls[seed])) for seed in seeds], axis=0)
            for j, m in enumerate(METRICS):
                res[f"concept_{m}"][li, si], res[f"control_{m}"][li, si] = c[:, j], k[:, j]
    res["seconds"] = time.perf_counter() - t0
    return res


def reader_layer(sw: dict, mass_floor: float = MASS_FLOOR) -> dict:
    """Per layer: the effect of the concept over the control (mean change in
    target log-odds, concept minus control) at the dose with the largest such
    effect among the doses whose option-letter mass keeps ``mass_floor`` of the
    baseline (usually, not necessarily, the strongest of them); the reader is
    the layer with the largest effect. ``dose`` tables give the full curves.
    Each dose also carries the target choice rate (share of prompts whose
    most likely option letter is the target) for the concept (``chosen``) and
    the control (``control_chosen``); ``base_chosen`` is the unsteered rate."""
    base = sw["base_logodds"].mean()
    bmass = sw["base_mass"].mean()
    per_layer = {}
    for li, l in enumerate(sw["layers"]):
        rows = []
        for si, s in enumerate(sw["strengths"]):
            c, r = sw["concept_logodds"][li, si].mean() - base, sw["control_logodds"][li, si].mean() - base
            cm, rm = sw["concept_mass"][li, si].mean() / max(bmass, 1e-9), sw["control_mass"][li, si].mean() / max(bmass, 1e-9)
            rows.append(dict(strength=s, concept=float(c), control=float(r), effect=float(c - r),
                             concept_mass_ratio=float(cm), control_mass_ratio=float(rm),
                             chosen=float(sw["concept_chosen"][li, si].mean()), control_chosen=float(sw["control_chosen"][li, si].mean())))
        ok = [r for r in rows if r["concept_mass_ratio"] >= mass_floor]
        best = max(ok, key=lambda r: r["effect"]) if ok else None
        per_layer[l] = dict(dose=rows, best=best)
    cands = {l: v["best"]["effect"] for l, v in per_layer.items() if v["best"] is not None}
    reader = max(cands, key=cands.get) if cands else None
    out = dict(layer=reader, best=per_layer[reader]["best"] if reader is not None else None,
               base_logodds=float(base), base_mass=float(bmass), base_p=float(sw["base_p"].mean()),
               base_chosen=float(sw["base_chosen"].mean()), mass_floor=mass_floor, per_layer=per_layer)
    return out


# ---------------------------------------------------------------- open generation: what the steered model says

EXPLAIN = "Reply with the letter of your choice, then one sentence saying why."
LADDER = (0.0, 0.2, 0.4, 0.8, 1.6, 3.2)   # 0 = the unsteered rung, same instruction and budget: the reference the edge is measured against
EXPLAIN_LONG = "Reply with the letter of your choice, then explain your choice in a detailed paragraph."
LADDER_TOKENS = 160          # long enough to see a steered reply stop early (in chat, fluffy @ 0.8 stopped after ~100 tokens)
SHORT_TOKENS = 8             # a ladder reply under this many tokens (a bare letter, a letter and a word) wrote no paragraph
FLUENCY_MARGIN = 0.25        # a rung is broken when a damage rate exceeds the unsteered rung's by more than this (absolute)...
DISTINCT_DROP = 0.15         # ...or its mean distinct-token ratio is more than this below the unsteered rung's
DISTINCT_WINDOW = 32         # that ratio is the mean over sliding windows of this many tokens, so it does not fall with length


def _letter_of(text: str, letters: list[str]) -> Optional[str]:
    """The option letter (one of ``letters``) the reply opens with, if any."""
    alts = "|".join(re.escape(x) for x in sorted(letters, key=len, reverse=True))
    m = re.match(rf"\s*\(?({alts})[\)\.:,\s]", text + " ")
    return m.group(1) if m else None


def _concept_hits(text: str, words: list[str], exclude: str = "") -> list[str]:
    """Concept words the reply uses on its own: words that already occur in
    the prompt or its options (``exclude``) do not count, or quoting the
    option "Substantially changed" would count as using "change"."""
    low, ex = text.lower(), exclude.lower()
    return sorted({w for w in words if w.lower() in low and w.lower() not in ex})


def _generation_summary(items: list[dict], conds=("base", "concept", "control")) -> dict:
    """Per condition: how often the reply opens with the target letter,
    mentions a concept word, stops on its own; mean distinct."""
    summ = {}
    for k in conds:
        rs = [rec["replies"][k] for rec in items]
        hit = np.array([r["letter"] == rec["target_letter"] for r, rec in zip(rs, items)], dtype=float)
        summ[k] = dict(n=len(rs), target=float(hit.mean()),
                       lettered=float(np.mean([r["letter"] is not None for r in rs])),
                       concept_words=float(np.mean([bool(r["hits"]) for r in rs])),
                       stopped=float(np.mean([r["stopped"] for r in rs])),
                       distinct=float(np.mean([r["distinct"] for r in rs])))
    return summ


_SENTENCE_END = re.compile(r'[.!?…"”\')\]*]\s*$')


def _fluency(toks: list[int], max_tokens: int, text: str = "", stopped: Optional[bool] = None) -> dict:
    """Cheap damage signals of one reply:
    ``stopped``: it ended on its own (the engine's ``stopped``; None: under budget, as a greedy reply that ended on
    its end-of-turn token is);
    ``early_stop``: it stopped in the first half of the budget, not at the end of a sentence (steering can raise the
    chance of the end-of-turn token; the forced-choice sweep never sees that);
    ``short``: under SHORT_TOKENS tokens, a bare letter or a letter and a word;
    ``distinct``: the share of distinct tokens in the whole reply, which falls with length even in fluent text;
    ``distinct_window``: the mean share over sliding windows of DISTINCT_WINDOW tokens (the whole reply when shorter),
    which does not, and which the ladder compares."""
    distinct = len(set(toks)) / max(len(toks), 1)
    stopped = len(toks) < max_tokens if stopped is None else bool(stopped)
    early = bool(stopped and 0 < len(toks) < max_tokens // 2 and not _SENTENCE_END.search(text.strip()))
    return dict(n_tokens=len(toks), stopped=stopped, distinct=float(distinct), distinct_window=windowed_distinct(toks),
                early_stop=early, short=len(toks) < SHORT_TOKENS)


def windowed_distinct(toks: list[int], window: int = DISTINCT_WINDOW) -> float:
    """The moving-average distinct-token ratio: the mean share of distinct
    tokens over every run of ``window`` consecutive tokens (the plain ratio
    for a shorter reply). A loop shorter than the window scores low wherever
    it sits; a long fluent reply scores like a short one."""
    if len(toks) <= window:
        return float(len(set(toks)) / max(len(toks), 1))
    return float(np.mean([len(set(toks[i:i + window])) / window for i in range(len(toks) - window + 1)]))


def _is_short(x: dict) -> bool:
    """A reply under SHORT_TOKENS tokens."""
    return x["n_tokens"] < SHORT_TOKENS


def generation(engine, concept: Concept, vec: Vector, table: dict, layer: int, strength: float, max_tokens: int = 60,
               ladder: tuple[float, ...] = LADDER, ladder_items: int = 4, seed: int = 1,
               on_progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """What the model says, not just which letter it weighs. Every task item
    (its options in their own order) is answered three ways under the EXPLAIN
    instruction: unsteered, steered with the concept direction at ``layer``
    and ``strength`` through prompt and generation, and steered with the
    matched random control. Recorded per reply: the text, the letter it
    opens with, which of the concept's words it uses, and two damage signals.
    Then a dose ladder on the first ``ladder_items`` items under the
    long-answer instruction (EXPLAIN_LONG) with LADDER_TOKENS of room: the
    concept direction at each strength, to see where fluency breaks, including
    replies that stop early. The ladder always starts with the unsteered rung
    (dose 0, added if ``ladder`` lacks it), which ``fluency_rungs`` measures
    every steered rung against."""
    assert concept.task is not None
    ladder = (0.0,) + tuple(s for s in ladder if s != 0)
    task = Task(EXPLAIN, concept.task.items, concept.task.letters)
    tok = engine.tok
    ctrl = control_direction(vec, push_layers(vec.meta, [layer]), seed=seed)
    conds = dict(base=None, concept=adds_for(vec, layer, strength, table),
                 control=adds_for(vec, layer, strength, table, ctrl))
    out = dict(layer=layer, strength=strength, instruction=EXPLAIN, max_tokens=max_tokens, shifts=[0], seed=seed,
               task_items=[dict(prompt=it.prompt, options=it.options, target=it.target) for it in task.items],   # the items as run, for rescoring
               items=[], ladder=[])
    if task.letters:
        out["letters"] = list(task.letters)          # for rescoring; absent means A, B, C, ...
    n_items = len(task.items)
    tick = _ticker(on_progress, n_items * len(conds) + min(ladder_items, n_items) * len(ladder))
    t0 = time.perf_counter()

    def reply(ids, adds, n_opt, body, budget=max_tokens):
        toks, stopped = engine.generate(ids, adds, budget)
        text = tok.decode(toks)
        return dict(text=text, letter=_letter_of(text, option_letters(task, n_opt)), hits=_concept_hits(text, concept.words, body),
                    **_fluency(toks, budget, text, stopped))

    for i, item in enumerate(task.items):
        ids, _, slot = render_item(tok, task, item, 0)
        body = render_body(task, item, 0)
        rec = dict(item=i, shift=0, target_letter=option_letters(task, len(item.options))[slot], prompt=body, replies={})
        for k, adds in conds.items():
            rec["replies"][k] = reply(ids, adds, len(item.options), body)
            tick()
        out["items"].append(rec)
    long_task = Task(EXPLAIN_LONG, concept.task.items, concept.task.letters)
    out["ladder_instruction"], out["ladder_tokens"] = EXPLAIN_LONG, LADDER_TOKENS
    for i, item in enumerate(task.items[:ladder_items]):
        ids, _, slot = render_item(tok, long_task, item, 0)
        body = render_body(long_task, item, 0)
        for s in ladder:
            adds = adds_for(vec, layer, s, table) if s else None
            r = reply(ids, adds, len(item.options), body, LADDER_TOKENS)
            out["ladder"].append(dict(item=i, strength=s, target_letter=option_letters(long_task, len(item.options))[slot], **r))
            tick()
    out["summary"] = _generation_summary(out["items"])
    out["fluency"] = fluency_rungs(out)
    out["seconds"] = time.perf_counter() - t0
    return out


# The shares that break a dose. Overrun is shown but breaks nothing: a concept that makes replies longer or more
# dramatic runs past the budget while still fluent (anime-nerd at 0.2 to 0.8).
# Loops and gibberish fall on distinct; silence on short or early.
BREAKING_RATES = ("short", "early")


def _rung(replies: list[dict], distinct_fallback: Optional[float] = None) -> dict:
    """One ladder dose's fluency over its replies: the share that ran to the
    budget without stopping (``overrun``), that are short (``short``), that
    stopped early (``early``), and the mean distinct-token ratio over the
    replies that are not short (an empty reply or a bare letter is counted by
    ``short``, not here), over windows of DISTINCT_WINDOW tokens
    (``distinct_window``). With no reply that is not short, ``distinct`` is
    ``distinct_fallback`` (the unsteered rung's value; None for that rung)."""
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    long = [float(x["distinct_window"]) for x in replies if not _is_short(x)]
    return dict(n=len(replies), overrun=mean([not x["stopped"] for x in replies]),
                short=mean([_is_short(x) for x in replies]), early=mean([bool(x["early_stop"]) for x in replies]),
                distinct=float(np.mean(long)) if long else distinct_fallback)


def fluency_rungs(gen: Optional[dict]) -> Optional[dict]:
    """The ladder's fluency per dose, measured against the unsteered rung
    (dose 0, same instruction and budget). A steered rung is broken when any
    of its BREAKING_RATES exceeds the unsteered rung's by more than FLUENCY_MARGIN
    (absolute) or its mean distinct-token ratio (``_rung``: replies that are
    not short, windowed) is more than DISTINCT_DROP below the unsteered
    rung's; ``why`` lists which (no distinct test when the unsteered rung has
    only short replies). ``distinct_window`` is the window. Every nonzero dose is a steered rung, in increasing
    order, wherever the ladder lists dose 0. ``edge`` is the first
    broken dose, None if none is. None without a ladder, or for a ladder
    without its unsteered rung (``generation`` always runs one): nothing to
    measure against."""
    if not gen or not gen.get("ladder"):
        return None
    doses = sorted({float(x["strength"]) for x in gen["ladder"]})
    if 0.0 not in doses:
        return None
    at = lambda s: [x for x in gen["ladder"] if float(x["strength"]) == s]
    ref = _rung(at(0.0))
    rungs = []
    for s in [d for d in doses if d != 0.0]:
        m = _rung(at(s), distinct_fallback=ref["distinct"])
        why = [k for k in BREAKING_RATES if m[k] - ref[k] > FLUENCY_MARGIN + 1e-9]
        if ref["distinct"] is not None and ref["distinct"] - m["distinct"] > DISTINCT_DROP + 1e-9:
            why.append("distinct")
        rungs.append(dict(strength=s, **m, broken=bool(why), why=why))
    return dict(edge=next((r["strength"] for r in rungs if r["broken"]), None), reference=ref, rungs=rungs,
                margin=FLUENCY_MARGIN, distinct_drop=DISTINCT_DROP, distinct_window=DISTINCT_WINDOW)

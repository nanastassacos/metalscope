"""What ``concept learn`` measures with, as pure functions of the numbers: which items, the push, when it stops.

- ``select_items``: which task items the push is tested on. An item is
  useful when the unsteered model is against it or torn (P(target) between
  ``AGAINST``), so a push toward the concept has room to show. A few items it
  already favours (``FAVOUR``) are kept as a check that the push still raises
  them. The against items are split at random into a selection half (picks
  the steer layer and dose) and a test half (reports the effect), so the
  choice of layer and dose does not inflate the reported effect.
- ``untestable`` and ``NO_STEER_LAYER``: learn's only early stops, "can't be
  steered": no item has room to move, or no layer and dose kept the answers
  well formed. Anything else goes on to the conversations (calibrate.py),
  whose verdict is the concept's.
- ``push_result``: the test sweep's numbers at the chosen dose D, its
  quarter and half, and the opposite push, away from the concept (−D);
  concepts/summary.py ``push_words`` says it in plain words.
- ``evidence``: the measurements as plain lines, for the job's result.
- ``overlap``: the new direction's cosine with every existing vector at the
  peak layer (held-out d'), nearest first.

All of it is as the model understood the concept: the corpus, items and
words all come from the same model.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

AGAINST = (0.03, 0.5)        # unsteered P(target): against or torn, room to rise
FAVOUR = (0.5, 0.95)         # already leaning to the target: kept as a check
MAX_AGAINST, MAX_FAVOUR = 16, 4
MIN_AGAINST = 8              # fewer usable items: the push is "not testable" (the test half gets at least 4 items, 8 prompts)

CANT_STEER = "can't be steered"
NO_STEER_LAYER = f"{CANT_STEER}: no layer and dose kept the answers well formed, so there is no layer to steer at"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -60 else 0.0


def item_probs(prompt_items: list[int], logodds: list[float], n_items: int) -> list[Optional[float]]:
    """Each item's unsteered P(target), averaged over its option rotations,
    from the per-prompt target log-odds against the other options."""
    acc: dict[int, list[float]] = {}
    for i, lo in zip(prompt_items, logodds):
        acc.setdefault(int(i), []).append(sigmoid(float(lo)))
    return [float(np.mean(acc[i])) if i in acc else None for i in range(n_items)]


def select_items(probs: list[Optional[float]], seed: int = 0) -> dict:
    """``selection`` and ``test`` (against items, split in half at random),
    ``favour`` (a few items already leaning to the target), ``testable``
    (at least MIN_AGAINST against items) and ``fallback``: when the push is
    not testable, the items to pick a layer and dose on anyway (every item
    that is neither certain nor impossible), so ``/steer`` still gets a
    default. No such item (``fallback`` empty): nothing to measure the push
    on, and learn stops after the items (``untestable``)."""
    rng = random.Random(seed)
    against = [i for i, p in enumerate(probs) if p is not None and AGAINST[0] <= p <= AGAINST[1]]
    favour = [i for i, p in enumerate(probs) if p is not None and FAVOUR[0] < p <= FAVOUR[1]]
    rng.shuffle(against)
    rng.shuffle(favour)
    against, favour = sorted(against[:MAX_AGAINST]), sorted(favour[:MAX_FAVOUR])
    shuffled = list(against)
    rng.shuffle(shuffled)
    half = (len(shuffled) + 1) // 2
    out = dict(against=against, favour=favour, selection=sorted(shuffled[:half]), test=sorted(shuffled[half:]),
               testable=len(against) >= MIN_AGAINST,
               counts=dict(items=len(probs), against=len(against), favour=len(favour),
                           saturated_high=sum(1 for p in probs if p is not None and p > FAVOUR[1]),
                           saturated_low=sum(1 for p in probs if p is not None and p < AGAINST[0])))
    if not out["testable"]:
        out["fallback"] = [i for i, p in enumerate(probs) if p is not None and AGAINST[0] <= p <= FAVOUR[1]]
    return out


def picked_on(sel: dict) -> str:
    """What the steer layer and dose were picked on: the selection half, or with too few against items for a held-out
    test, every usable item (``select_items``' ``fallback``)."""
    if sel.get("testable", True):
        return "the selection half"
    return f"all {len(sel.get('fallback') or [])} usable items (no held-out test)"


def untestable(counts: dict, name: str, model: str) -> str:
    """Why learn stops when no item has room to move (``select_items``' ``fallback`` empty), as one line a person
    can act on."""
    high, low, n = counts["saturated_high"], counts["saturated_low"], counts["items"]
    did = (f"already picks the {name} option on {high}" if high >= low else f"never picks the {name} option on {low}")
    return (f"{CANT_STEER}: {model} {did} of {n} items unsteered; rephrase the contrast so the model doesn't "
            "already do it (or the other side), then learn it again under a new name")


def push_result(test: dict, kinds: list[str], dose: float) -> dict:
    """The test sweep at one layer: ``test`` holds per-prompt arrays
    (``base_logodds`` / ``base_chosen`` [P], ``concept_logodds`` / ``control_logodds`` /
    ``concept_mass`` / ``concept_chosen`` / ``control_chosen`` [1, S, P],
    ``base_mass`` [P], ``strengths`` [S]); ``kinds`` says per prompt whether
    its item is "against" or "favour". Effects are mean log-odds changes over
    the unsteered prompt, concept minus control, on the against prompts (the
    favour prompts stay only in the raw sweep, ``learn.test``)."""
    s = [float(x) for x in test["strengths"]]
    idx = {round(v, 6): k for k, v in enumerate(s)}
    a = np.array([k == "against" for k in kinds])
    base = np.asarray(test["base_logodds"], float)
    bmass = np.asarray(test["base_mass"], float)
    cl = np.asarray(test["concept_logodds"], float)[0]
    rl = np.asarray(test["control_logodds"], float)[0]
    cm = np.asarray(test["concept_mass"], float)[0]
    cc = np.asarray(test["concept_chosen"], float)[0]
    rc = np.asarray(test["control_chosen"], float)[0]
    bc = np.asarray(test["base_chosen"], float)

    def at(dose_value: float) -> Optional[dict]:
        k = idx.get(round(dose_value, 6))
        if k is None or not a.any():
            return None
        row = dict(strength=dose_value,
                   concept=float((cl[k][a] - base[a]).mean()), control=float((rl[k][a] - base[a]).mean()),
                   mass_kept=float(cm[k][a].mean() / max(bmass[a].mean(), 1e-9)),
                   chosen=float(cc[k][a].mean()), control_chosen=float(rc[k][a].mean()),
                   base_chosen=float(bc[a].mean()))
        row["effect"] = row["concept"] - row["control"]
        row["flips_over_control"] = row["chosen"] - row["control_chosen"]
        return row

    curve = [at(v) for v in s]
    return dict(dose=dose, curve=[c for c in curve if c is not None], at_dose=at(dose), at_quarter=at(dose / 4),
                away=at(-dose), n_against=int(a.sum()))


def overlap(new_dir: np.ndarray, existing: dict[str, np.ndarray]) -> dict:
    """``new_dir`` against every existing direction at the same layer: the five nearest by |cosine|."""
    unit = lambda v: np.asarray(v, float) / max(float(np.linalg.norm(v)), 1e-12)
    n = unit(new_dir)
    cos = {k: float(n @ unit(v)) for k, v in existing.items()}
    top = sorted(cos.items(), key=lambda kv: -abs(kv[1]))[:5]
    return dict(nearest=[dict(name=k, cosine=v) for k, v in top], n_existing=len(cos))


def evidence(learn: dict, heldout: Optional[float], p_value: Optional[float], peak: Optional[int],
             reader: Optional[dict] = None) -> list[str]:
    """What learn measured, one plain line each: presence, the items, the steer layer (``reader``: the profile's), the
    ladder, the push on the held-out items (its sentences are concepts/summary.py ``push_words``')."""
    out = []
    if heldout is not None:
        out.append(f"presence: held-out d′ {heldout:.2f} at L{peak}" + (f", p {p_value:.3f}" if p_value is not None else ""))
    c = (learn.get("selection") or {}).get("counts")
    if c:
        out.append(f"items: {c['against']} against or torn and {c['favour']} leaning to it, of {c['items']} "
                   f"({c['saturated_high']} already chosen, {c['saturated_low']} never)")
    best = (reader or {}).get("best")
    if best and reader.get("layer") is not None:
        out.append(f"steer layer: L{reader['layer']} at dose {best['strength']:g}, {best['effect']:+.2f} log-odds over a "
                   f"random direction on {picked_on(learn.get('selection') or {})}")
    if learn.get("layer") is not None:
        edge = learn.get("edge")
        out.append((f"fluency: replies break from dose {edge:g}" if edge is not None else "fluency: no break up to dose 3.2")
                   + (f"; test dose {learn['dose']:g}" if learn.get("dose") is not None else ""))
    push = learn.get("push") or {}
    d, q = push.get("at_dose"), push.get("at_quarter")
    if d:
        n = push.get("n_against")
        out.append(f"push at dose {d['strength']:g}" + (f" on {n} held-out prompts" if n else "")
                   + f": {d['effect']:+.2f} log-odds over a random direction"
                   + (f" ({q['effect']:+.2f} at a quarter of it)" if q else "")
                   + f"; answers still letters: {100 * d['mass_kept']:.0f}% of the mass")
    return out

"""What a profile says, in short: for the server API and Pi, and for the terminal.

``summarize`` gives the headline numbers /steer needs,
``summary_lines`` the few lines the CLI prints, ``push_words`` learn's push in plain words (learn's job result
too). Where a model's profiles live: models.py.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from .. import config, models
from .measure import fluency_rungs
from .stats import floor_of, p_text, peak_layer, reader_best

def steer_layer(prof) -> Optional[int]:
    """The layer a concept steers at: its profile's reader layer; None without a profile (``prof`` not a dict) or a
    reader."""
    layer = reader_best(prof)[0].get("layer") if isinstance(prof, dict) else None
    return None if layer is None else int(layer)


def steer_dose(prof: dict, calibration: Optional[dict] = None) -> Optional[dict]:
    """The default dose for ``/steer <concept>``: the chat calibration's (``calibration``, learn/calibrate.py version
    4) when it steered at the reader layer. ``{layer, strength, basis: "chat"}``; None otherwise: a dose is a default
    only when measured in conversation."""
    layer = steer_layer(prof)
    cal = calibration or {}
    if (layer is None or cal.get("version") != models.CALIBRATION_VERSION or cal.get("default_dose") is None
            or cal.get("layer") != layer):
        return None
    return dict(layer=int(layer), strength=float(cal["default_dose"]), basis="chat")


def generation_dose(prof: dict) -> Optional[dict]:
    """Where learn's GPU stage runs its test sweep (learn/gpu.py), before any calibration: at the reader layer, the
    stored generation's dose when it ran there, else the reader's; half the fluency edge when that is at or past it.
    ``{layer, strength}``; None without a reader. Never a /steer default (``steer_dose``)."""
    r, b = reader_best(prof)
    if r.get("layer") is None or b.get("strength") is None:
        return None
    g = prof.get("generation")
    ran_here = g and g.get("layer") == r["layer"] and g.get("strength") is not None
    strength = float(g["strength"] if ran_here else b["strength"])
    edge = (fluency_rungs(g) or {}).get("edge")
    if edge is not None and strength >= edge - 1e-9:
        strength = float(edge) / 2
    return dict(layer=int(r["layer"]), strength=strength)


def generation_dose_why(prof: dict) -> str:
    """Why the stored generation ran at its dose: the reader's dose; below it
    because the reader's is at or past the fluency edge; or a dose or layer
    given by hand; plus a warning when the dose is itself at or past the edge
    (``below_edge`` false). Empty without a generation."""
    g = prof.get("generation")
    if not g:
        return ""
    r = prof.get("reader") or {}
    rs = g["reader_strength"]
    edge = (fluency_rungs(g) or {}).get("edge")
    s = float(g["strength"])
    below = g["below_edge"]
    if r.get("layer") is None:
        why = "given by hand (the profile has no reader)"
    elif g.get("layer") != r.get("layer"):
        why = f"at a layer other than the reader's (L{r['layer']}), given by hand"
    elif rs is None or abs(s - float(rs)) < 1e-9:
        why = "the reader's dose"
    elif edge is not None and float(rs) >= edge - 1e-9 and s < float(rs):
        why = f"below the reader's dose {float(rs):g}, which is at or past the fluency edge {edge:g}"
    else:
        why = f"not the reader's dose {float(rs):g}"
    tail = "" if below or edge is None else f"; at or past the fluency edge ({edge:g}): expect damaged replies"
    return why + tail


def summarize(prof: dict, calibration: Optional[dict] = None) -> dict:
    """What Pi reads of a profile (/scope/api/concepts): the reader's layer and dose with its task log-odds, the
    ladder's fluency edge, ``steer`` (the default dose, ``steer_dose``; None when not calibrated) and learn's push
    both ways in words (``push_words_of``)."""
    r, b = reader_best(prof)
    g = prof.get("generation")
    return dict(
        name=prof["concept"], title=prof.get("title") or prof["concept"], description=prof.get("description", ""),
        model=prof["model"],
        reader=dict(layer=r.get("layer"), strength=b.get("strength"), concept=b.get("concept"), control=b.get("control"))
        if b else None,
        generation=dict(edge=(fluency_rungs(g) or {}).get("edge")) if g else None,
        steer=steer_dose(prof, calibration),
        push=push_words_of(prof),
    )


def push_words_of(prof: dict) -> Optional[dict]:
    """``concept learn``'s push on held-out items in plain words, both ways (``push_words``); None without one."""
    push =(prof.get("learn") or {}).get("push")
    words = push_words(push, prof["concept"]) if push else None
    return words if words and words["toward"] else None


def push_away(push: dict) -> Optional[dict]:
    """The push away from the concept (−D): ``away``, or the curve's point at −D (profiles learned before ``away``)."""
    if push.get("away"):
        return push["away"]
    dose = push.get("dose")
    return next((c for c in push.get("curve") or [] if dose is not None and abs(c["strength"] + dose) < 1e-6), None)


def _pct(x) -> Optional[int]:
    return None if x is None else round(100 * float(x))


def _often(a: int, b: int) -> str:
    return "more often" if b > a else "less often" if b < a else "as often"


def push_toward(push: Optional[dict]) -> tuple[Optional[str], Optional[bool]]:
    """The push toward the concept at the test dose in a few words, and whether it went up: ("picked more often, 25%
    to 44% (a random direction: 28%)", True); without the shares, "made more / less likely"; (None, None) when not
    measured. ``push_words``' toward sentence and learn's test step (learn/gpu.py) say it."""
    d = (push or {}).get("at_dose")
    if not d:
        return None, None
    a, b, ctl = _pct(d.get("base_chosen")), _pct(d.get("chosen")), _pct(d.get("control_chosen"))
    if a is not None and b is not None:
        return (f"picked {_often(a, b)}, {a}% to {b}%" + (f" (a random direction: {ctl}%)" if ctl is not None else "")), b > a
    if d.get("concept") is not None:
        up = d["concept"] > 0
        return f"made {'more' if up else 'less'} likely", up
    return None, None


def push_words(push: Optional[dict], name: str = "the concept") -> dict:
    """The push on the held-out items in plain words: ``toward`` and ``away``, one sentence each on how often the
    model picks the concept's option (the share of prompts where its letter is the likeliest), unsteered to steered;
    None when not measured. The away sentence says whether that was expected; without the shares, "more / less
    likely"."""
    r = push_away(push or {})
    words, up = push_toward(push)
    toward = f"Steering toward {name}: {words}." if words else None
    away = None
    if r:
        lo, a, c = r.get("concept"), _pct(r.get("base_chosen")), _pct(r.get("chosen"))
        went = None if lo is None else "less" if lo < 0 else "more"
        too = " too" if up else ""
        what = None
        if a is not None and c is not None:
            verb = _often(a, c)
            ok = lo < 0 if lo is not None else c < a
            what = (f"picked {verb}{too if verb == 'more often' else ''}, {a}% to {c}%"
                    + (f", but made {went} likely" if went and verb != f"{went} often" else ""))
            expected = "less often"
        elif went:
            ok = lo < 0
            what, expected = f"made {went} likely{too if went == 'more' else ''}", "less likely"
        if what:
            away = f"Steering away from {name}: {what}" + (", as expected." if ok else f" (expected {expected}).")
    return dict(toward=toward, away=away)


def norms_stamp(served: str) -> Optional[dict]:
    """Which strength table the served copy's doses are in now (steer.norm_table's norms.json): ``{made, tokens}``;
    None without one."""
    table = config.read_json(models.norms_path(served))
    return dict(made=table.get("made"), tokens=table.get("tokens")) if isinstance(table, dict) else None


def calibration_of(served: str, concept: str) -> Optional[dict]:
    """The served copy's chat calibration of ``concept`` (models.calibration_path, written by ``concept calibrate``,
    learn/calibrate.py): version 4 has each strength's good replies and the default dose (``steer_dose`` uses it), the
    one with the most. None when there is none, it is of an older version, it cannot be read, the checkpoint's
    profile changed after it was made (``profile_mtime``: re-profiling may move the vector and the layer), or the
    served copy's strength table did (``norms``, ``norms_stamp``: a new table rescales every dose). A calibration
    without ``norms`` counts as made in the table there is while that table was made before it. Another served
    copy's calibration is never this one's."""
    path = models.calibration_path(served, concept)
    try:
        cal = json.loads(path.read_text())
        mtime = models.profile_path(served, concept).stat().st_mtime
        written = path.stat().st_mtime
    except (OSError, ValueError):
        return None
    if (not isinstance(cal, dict) or cal.get("version") != models.CALIBRATION_VERSION
            or abs(float(cal.get("profile_mtime") or 0) - mtime) > 1e-3):
        return None
    units = norms_stamp(served)
    if "norms" in cal:
        return cal if cal["norms"] == units else None
    try:
        made = time.mktime(time.strptime(str((units or {}).get("made")), "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return cal                                   # no table, or one without its date: nothing to go by
    return cal if made < written else None


def current_calibrations(served: str) -> list[str]:
    """The concepts with a calibration on the served copy that fits (``calibration_of``)."""
    return [p.parent.name for p in models.profile_paths(served)
            if calibration_of(served, p.parent.name) is not None]


def reader_choice(prof: dict) -> Optional[dict]:
    """The target choice rate at the reader layer and dose (share of prompts
    whose most likely option letter is the target): ``base`` unsteered,
    ``concept`` and ``control`` steered. None without a reader dose."""
    r, b = reader_best(prof)
    if not b:
        return None
    return dict(base=r["base_chosen"], concept=b["chosen"], control=b["control_chosen"])


def summary_lines(prof: dict, calibration: Optional[dict] = None) -> list[str]:
    """A few lines for the terminal; ``calibration``: the served copy's (``calibration_of``), for the /steer default."""
    w, c, r = prof.get("writer") or {}, prof.get("carrier"), prof.get("reader")
    on = prof.get("measured_on") or {}
    via = f" (measured on {on['served']}, {on['engine']})" if on.get("served") and on["served"] != prof["model"] else ""
    lines = [f"{prof['title']} on {prof['model']}{via}: {prof['n_pos']} vs {prof['n_neg']} texts, {prof['position']} position"]
    stats = config.int_keys(prof["stats"])
    best = peak_layer(stats)
    ptxt = p_text(stats[best])
    lines.append(f"  writer: first layer that is significant ({w['rule']}) L{w.get('first')}, largest rise at L{w.get('jump')}; "
                 f"held-out d' peaks at L{best} ({stats[best]['heldout']:.1f}; shuffled-label floor {floor_of(stats[best]):.1f}, {ptxt})")
    if c:
        lines.append(f"  carrier: L{c['start']}-{c['end'] - 1} (adjacent cosine >= {c['threshold']}, mean {c['mean_cos']:.2f})")
    else:
        lines.append("  carrier: no band of adjacent layers keeps the direction")
    if r and r.get("layer") is not None and r.get("best"):
        b, ch = r["best"], reader_choice(prof)
        lines.append(f"  reader: L{r['layer']} at strength {b['strength']:g}: target log-odds {b['concept']:+.2f} (control {b['control']:+.2f}; "
                     f"baseline p {r['base_p']:.2f}), letter mass kept {100 * b['concept_mass_ratio']:.0f}%; target chosen "
                     f"{100 * ch['base']:.0f}% → {100 * ch['concept']:.0f}% (control {100 * ch['control']:.0f}%) as the most likely option letter")
    elif r:
        lines.append("  reader: no layer moved the choice at a dose that kept the answer well formed")
    pw = push_words_of(prof)
    if pw:
        lines.append(f"  push on held-out items: {pw['toward']}" + (f" {pw['away']}" if pw["away"] else ""))
    g = prof.get("generation")
    if g:
        s, fr = g["summary"], fluency_rungs(g)
        edge = "" if not fr else ((f"; fluency edge {fr['edge']:g}" if fr["edge"] is not None else "; no fluency edge within the ladder")
                                  + " (against the unsteered rung)")
        lines.append(f"  generation at L{g['layer']} @ {g['strength']:g}: opens with the target on {100 * s['base']['target']:.0f}% / "
                     f"{100 * s['concept']['target']:.0f}% / {100 * s['control']['target']:.0f}% of replies (unsteered / concept / control), "
                     f"uses a concept word on {100 * s['base']['concept_words']:.0f}% / {100 * s['concept']['concept_words']:.0f}% / {100 * s['control']['concept_words']:.0f}%"
                     + edge + f" (dose: {generation_dose_why(prof)})")
    sd = steer_dose(prof, calibration)
    if sd:
        lines.append(f"  /steer default: L{sd['layer']} @ {sd['strength']:g} (the most good replies in conversation: "
                     f"{calibration.get('good')}/{calibration.get('turns')})")
    elif r and r.get("layer") is not None:
        lines.append("  /steer default: not calibrated (concept calibrate sets it)")
    return lines


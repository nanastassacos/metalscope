"""A profile's presence, and writing a profile: ``profile`` and ``write_profile``.

``profile`` measures the direction and its statistics at every layer (stats) on an engine (engines/) and saves the
vector; ``concept learn``'s GPU stage (learn/gpu.py) adds the items, sweeps, ladder and test to it.
``write_profile`` saves ``profile.json``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .. import config, models
from ..engines import SINK, samples
from ..vectors import Vector, check_name, save_vector, unit
from .corpus import Concept, option_letters, render_body
from .measure import fluency_rungs
from .stats import adjacent_cosine, carrier_band, contrast_stats, reader_best, writer_layer


def profile(engine, concept: Concept, log_fn: Callable[[str], None] = print) -> dict:
    """The concept's presence as a JSON-ready dict: the diff-of-means direction at every layer,
    saved as the concept's vector (with the engine's ``block`` where a push spans one), with its
    statistics, the writer and the carrier band, and what went in (``inputs``). The vector and profile are the
    checkpoint's (``engine.checkpoint``); ``measured_on`` names the engine and the served model the measurements ran
    on."""
    check_name(concept.name)                    # the vector is saved under it; refuse before the GPU work
    model_name, tok = engine.checkpoint, engine.tok
    n_layers = engine.shape.n_layers
    layers = list(range(n_layers))
    position, chat = "mean", False              # each text's mean residual, the text alone (no chat turn)
    t0 = time.perf_counter()
    out = dict(concept=concept.name, model=model_name, measured_on=dict(engine=engine.kind, served=engine.served),
               title=concept.meta.get("title", concept.name),
               description=concept.meta.get("description", ""), inspiration=concept.meta.get("inspiration", ""),
               n_pos=len(concept.pos), n_neg=len(concept.neg), position=position)

    log_fn(f"[{concept.name}] residuals of {len(concept.pos)} + {len(concept.neg)} texts at {len(layers)} layers ({position} position)")
    P = samples(engine, concept.pos, layers, position, chat)
    N = samples(engine, concept.neg, layers, position, chat)
    hc = engine.shape.hc
    vecs, natural, stats = {}, {}, {}
    for l in layers:
        st = contrast_stats(P[l], N[l], hc_count=hc)
        stats[l] = st
        d = P[l].mean(0) - N[l].mean(0)
        natural[l], vecs[l] = float(np.linalg.norm(d)), unit(d)
    block = engine.block                        # a push spans its block (steer.push_layers; Inkling 2)
    vec = Vector(concept.name, vecs, dict(source=f"concept {concept.name}: diff-of-means, {len(concept.pos)} vs {len(concept.neg)} texts, "
                                           f"{position} position", n_pos=len(concept.pos), n_neg=len(concept.neg), position=position,
                                           chat=chat, hc_count=hc, natural_norm=natural, dprime={l: stats[l]["dprime"] for l in layers},
                                           **({"block": block} if block else {})))
    save_vector(model_name, vec)
    cos = adjacent_cosine(vec)
    out["stats"] = {l: stats[l] for l in layers}
    out["adjacent_cosine"] = cos
    out["writer"] = writer_layer(stats)
    out["carrier"] = carrier_band(cos, start=out["writer"]["first"])
    log_fn(f"[{concept.name}] writer: first significant layer ({out['writer']['rule']}) {out['writer']['first']}, largest jump at {out['writer']['jump']}; "
           f"carrier band {out['carrier']}")
    # The concept's and the target's word tokens, for the inputs card.
    out["inputs"] = describe_inputs(concept, vec, position, n_layers, _word_ids(tok, concept.words),
                                    _word_ids(tok, concept.target_words), tok)
    out["seconds"] = time.perf_counter() - t0
    return jsonable(out)   # string keys throughout, as read back from profile.json


def attach_generation(prof: dict, gen: dict) -> dict:
    """Store a fresh ``generation`` run in ``prof`` (in place): records
    ``reader_strength`` (the reader's dose, for saying why the generation ran
    where it did) and ``below_edge`` (whether its dose is under the fluency
    edge of its own ladder; True when nothing broke)."""
    gen = dict(gen)
    gen["reader_strength"] = reader_best(prof)[1].get("strength")
    edge = (fluency_rungs(gen) or {}).get("edge")
    gen["below_edge"] = bool(edge is None or float(gen["strength"]) < edge - 1e-9)
    prof["generation"] = jsonable(gen)
    return prof


def describe_inputs(concept: Concept, vec: Vector, position: str, n_layers: int, word_ids: list[int],
                    target_ids: list[int], tok) -> dict:
    """What went in: corpus sizes and an example pair, the
    task rendered as the model sees it, the vector's space, the lens words."""
    task = concept.task
    out = dict(n_pos=len(concept.pos), n_neg=len(concept.neg), pos_example=concept.pos[0], neg_example=concept.neg[0],
               position=position, sink=SINK, width=vec.width, hc_count=vec.hc_count, n_layers=n_layers,
               words=list(concept.words), target_words=list(concept.target_words),
               n_word_tokens=len(word_ids), n_target_tokens=len(target_ids),
               word_tokens=[tok.decode([i]) for i in word_ids], target_tokens=[tok.decode([i]) for i in target_ids])
    if task is not None:
        item = task.items[0]
        out.update(n_items=len(task.items), options_per_item=sorted({len(it.options) for it in task.items}),
                   n_prompts=sum(len(it.options) for it in task.items), instruction=task.instruction,
                   prompt_example=render_body(task, item, 0), example_target=item.target,
                   letters=option_letters(task, max(len(it.options) for it in task.items)))
    return out


def _word_ids(tok, words: list[str]) -> list[int]:
    """Token ids for the words, with and without a leading space and
    capitalised: a form that is one token, or whose first piece is at least
    three characters (so "kitten" -> "k" is dropped: single letters would
    collide with the answer letters and with everything else)."""
    ids = []
    for w in words:
        for form in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            t = tok.encode(form)
            if not t:
                continue
            piece = tok.decode([int(t[0])]).strip()
            if (len(t) == 1 or len(piece) >= 3) and int(t[0]) not in ids:
                ids.append(int(t[0]))
    return ids


def sweep_inputs(table: dict, layers: list[int], strengths, seeds, n_items: int) -> dict:
    """What a sweep put in, for the profile's ``inputs``: the strength table's norms at ``layers`` and how many traces
    it came from, the strengths, the control seeds and the number of task items swept."""
    return dict(norms={l: table["layers"][l] for l in layers if l in table["layers"]}, norm_traces=len(table.get("traces") or []),
                strengths=list(strengths), seeds=list(seeds), coarse_items=n_items)


def sweep_json(sw: dict, per_prompt: bool = False) -> dict:
    """A ``sweep`` for profile.json: the baseline means, the concept and control curves [layers, strengths] (means over
    the prompts; ``chosen`` is the target choice rate) and, with ``per_prompt``, every prompt's numbers for the
    per-item view: [layers, strengths, prompts]."""
    out = dict(layers=sw["layers"], strengths=sw["strengths"], n_prompts=sw["n_prompts"], seeds=sw["seeds"],
               determinism=float(sw["determinism"]), seconds=float(sw["seconds"]),
               **{f"base_{m}": float(sw[f"base_{m}"].mean()) for m in ("logodds", "mass", "p", "chosen")},
               **{f"{k}_{m}": np.nanmean(sw[f"{k}_{m}"], axis=-1).tolist() for m in ("logodds", "mass", "chosen")
                  for k in ("concept", "control")})
    if per_prompt:
        out["prompts"] = dict(items=sw["items"], shifts=sw["shifts"], **{k: sw[k].tolist() for k in (
            "base_logodds", "base_p", "concept_logodds", "concept_p", "control_logodds", "control_p", "concept_mass",
            "base_chosen", "concept_chosen", "control_chosen")})
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_profile(prof: dict, out_dir: Optional[Path] = None) -> Path:
    d = out_dir or models.concept_dir(prof["model"], prof["concept"])
    d.mkdir(parents=True, exist_ok=True)
    prof = jsonable(prof)
    config.atomic_write(d / "profile.json", json.dumps(prof, indent=1, ensure_ascii=False))
    return d

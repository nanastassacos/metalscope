"""What ``concept learn NAME`` would do, or why it cannot (``learn_plan``): one set of checks for the command line,
Pi's confirm and start, and the job's own start.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .. import client, config, models
from ..vectors import check_name, find_vector
from . import writer as W
from .state import list_jobs


def unfinished_vector(name: str, model_name: str) -> bool:
    """Whether ``name``'s vector on ``model_name``'s checkpoint is one a learn run saved and never profiled (it was
    killed between the presence and the profile): no profile beside it, its metadata says ``profile`` made it
    (concepts/run.py), and a learn job for that name on that checkpoint started before it was written and ended
    interrupted or failed (learn/state.py). Profile-less vectors made otherwise (from the J-lens, or measured before
    learn jobs) have no such record and stay the user's. A new learn run overwrites it."""
    v = find_vector(model_name, name)
    if v is None or models.profile_path(model_name, name).exists():
        return False
    meta = config.read_json(v.with_suffix(".json"), {})
    if not (isinstance(meta, dict) and str(meta.get("source") or "").startswith(f"concept {name}:")):
        return False
    made, ckpt = v.stat().st_mtime, models.checkpoint_of(model_name)
    return any(j.get("kind") == "learn" and j.get("name") == name and j.get("outcome") in ("interrupted", "failed")
               and isinstance(j.get("model"), str) and models.checkpoint_of(j["model"]) == ckpt
               and isinstance(j.get("started"), (int, float)) and j["started"] < made for j in list_jobs())


def name_taken(name: str, model_name: str) -> Optional[str]:
    """Why ``name`` cannot be learned on ``model_name``'s checkpoint, else None: a vector or profile of it is there
    already (not a learn run's unfinished vector, ``unfinished_vector``)."""
    if find_vector(model_name, name) is not None and not unfinished_vector(name, model_name):
        return f"a steering vector named {name!r} already exists for {model_name}"
    if models.profile_path(model_name, name).exists():
        return f"a profile for {name!r} already exists for {models.checkpoint_of(model_name)}"
    return None


def no_dose_units(model_name: str) -> Optional[str]:
    """Why a learn run on the served ``model_name`` would fail once the model is unloaded, for want of dose units,
    else None: its sweeps size their doses with the served copy's strength table (steer.norm_table, from its own
    unsteered traced replies), and there is none to make it from."""
    from ..steer import NORM_MIN_TOKENS, norm_table
    try:
        norm_table(model_name)
    except ValueError:
        served = config.short_name(model_name)
        return (f"{served} has no traced replies yet, and a concept's doses are sized from them (about "
                f"{NORM_MIN_TOKENS} tokens of unsteered replies; 0 so far): chat a few unsteered messages with {served} "
                "first (metalscope on), then learn again")
    return None


def dose_units_note(model_name: str) -> Optional[str]:
    """steer.norms_note for the served copy's strength table: a warning when its dose units came from few or short
    traces; None otherwise, or without a table."""
    from ..steer import norms_note
    return norms_note(config.read_json(models.norms_path(model_name)))


# Words a new concept may not be named (Pi's RESERVED_NAMES): /concept and /steer would read them as their own.
RESERVED_NAMES = {"list", "show", "learn", "calibrate", "status", "cancel", "jobs", "replay", "help",
                  "off", "none"}


# A name JavaScript's Number() reads as a number (Pi's parseNumber): /steer takes it as a dose, never as a concept.
NUMBER_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?|0x[0-9a-f]+|0b[01]+|0o[0-7]+", re.I)


def check_new_name(name: str) -> None:
    """A usable name for a new concept, else ValueError."""
    check_name(name)
    if name.lower() in RESERVED_NAMES:
        raise ValueError(f"{name!r} is a command word, so it cannot name a concept: pick another name")
    if NUMBER_RE.fullmatch(name):
        raise ValueError(f"{name!r} reads as a number, and /steer takes a number as a dose, so it cannot name a concept: "
                         "pick a name with a letter in it")


def learn_plan(name: str, model_name: str, description: Optional[str] = None, corpus: Optional[str] = None,
               writer: Optional[str] = None, engine: Optional[str] = None, url: Optional[str] = None) -> dict:
    """What ``concept learn NAME`` would do on the served ``model_name``: the one set of checks and the one choice of
    where the texts come from, for the command line, Pi's confirm and start, and the job's own start.

    The texts (``source``): the model's own (``own``: a re-run, learned as they are), else the ones in ``corpus``
    (``corpus``: a folder written elsewhere), else the shipped ones (``shipped``: copied into the model's folder), else
    the writer (``writer``, default the served model) writes them (``write``) from ``description`` or NAME's stored
    definition. ``texts``: who wrote existing ones (``concepts.corpus.texts_line``). ``read``: the existing or
    supplied texts as read (``writer.write``'s shape), None when they are to be written. ``description`` and
    ``writer``: what the job uses. ``description`` None: not known yet (Pi's confirm), so not asked for; the stored
    definition, if any, is in the plan's. ``engine``: mlx, ds4, or auto (``pick_engine``, asking the ds4 proxy at
    ``url``, after every other check); None: not known.

    ``taken``: why it cannot run, else None; checked cheapest first (a taken name before a missing
    description). ``notes``: warnings that do not stop it."""
    from ..concepts import corpus as C
    name = name.strip()
    description = None if description is None else " ".join(description.split())
    corpus = str(Path(corpus).expanduser().resolve()) if corpus else None
    plan = dict(name=name, model=config.short_name(model_name), source="write", texts=None, description=description or "",
                writer=writer or model_name, corpus=corpus, engine=engine, read=None, taken=None, notes=[])
    try:
        check_new_name(name)
        if writer and corpus:
            raise ValueError("--writer and --corpus: a supplied corpus has no writer to pick")
        own = C.texts_dir(model_name, name)
        d = own or (None if corpus else C.shipped_dir(name))
        plan.update(source="own" if own else "corpus" if corpus else "shipped" if d else "write",
                    texts=C.texts_line(d, request=False) if d else None)
        why = name_taken(name, model_name)
        if why:                                  # first: nothing else would help
            raise ValueError(why)
        folder, ckpt = models.concept_dir(model_name, name), models.checkpoint_of(model_name)
        if (folder / "concept.json").is_file() and not all((folder / f).is_file() for f in C.TEXTS_NEEDED):
            raise ValueError(f"{name}'s texts on {ckpt} are incomplete ({folder}): metalscope concept delete {name} "
                             f"{config.short_name(model_name)}, then learn again")
        if own is not None and corpus:
            raise ValueError(f"{name} has its own texts on {ckpt}: learn it without --corpus")
        if d is not None and writer:
            raise ValueError(f"{name}'s texts exist, so nothing is written: no --writer to pick")
        if plan["source"] == "write":            # typed now, else the stored definition; "" asked and none: refused
            description = description or C.definition(name) or ("" if description == "" else None)
            plan["description"] = description or ""
            if description == "":
                raise ValueError(f"{name} needs a description (--description): one line saying what it means")
        read = W.read_corpus(corpus, name) if corpus else W.read_texts(d, name) if d is not None else None
        why = no_dose_units(model_name)          # the child needs them after the model is unloaded: refuse now
        if why:
            raise ValueError(why)
        if engine == "auto":
            engine = pick_engine(model_name, url)
        if read is None and engine == "ds4" and writer and writer != model_name:
            raise ValueError(f"on ds4 the corpus writer is the served model ({model_name}), not {writer}: "
                             "ds4 serves one model")
    except ValueError as exc:
        return dict(plan, taken=str(exc))
    learned = (read or {}).get("concept", {}).get("learned") or {}
    if corpus:
        plan.update(description=description or W.description_of(read["concept"]), writer=learned.get("writer") or "unknown")
    elif read is not None:
        plan.update(description=W.description_of(read["concept"]) or name, writer=learned.get("writer") or "unknown")
    notes = [f"the description given is ignored: {name}'s texts have their own" if d is not None and description else None,
             dose_units_note(model_name)]
    return dict(plan, engine=engine, read=read, notes=[n for n in notes if n])


def pick_engine(model_name: str, url: Optional[str] = None) -> str:
    """``concept learn --engine auto``: ds4 when the ds4 proxy (at ``url``, default ``client.DS4_URL``) serves
    ``model_name``, else mlx (a copy on oMLX)."""
    try:
        info = client.Client(url or client.DS4_URL).get("/scope/api/ds4", timeout=3)
    except client.ServerError:
        return "mlx"
    return "ds4" if isinstance(info, dict) and info.get("name") == model_name else "mlx"

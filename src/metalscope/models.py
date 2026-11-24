"""Where a model's data lives: all of it in its checkpoint's folder, a concept's by concept and then by served copy,
what belongs to one served copy alone by its name.

A checkpoint is the model a measurement is about (Qwen3.8-Flash-Next), whichever engine and quantisation serve it
(oMLX's ``Qwen3.8-Flash-Next-oQ4e-mtp``, ds4's ``Qwen3.8-Flash-Next-ds4-q4k``): a steering vector is a direction in
its residual stream, so every served copy reads the same vectors, profiles and J-lens. The registry says which
served names are which checkpoint; a name it does not list is its own checkpoint. What depends on the engine and
quantisation is the served copy's own: the strength table (so "dose 0.4" is the same relative push on every copy)
and the chat calibrations. A copy is never given another's: one without unsteered traces has no table, and
steering it says so.

    ~/.metalscope/models/<checkpoint>/       everything of one checkpoint
        checkpoint.json                      {name, shape: {n_layers, width, hc, d_model}, served: [...], lens, jlens}
                                             (checkpoint.lock orders its writers)
        experts-tally.npz                    the expert pages' tally (tally.py; ``link`` moves a served name's in)
        concepts/<concept>/                  one per concept learned on it
            concept.json, pos.txt, neg.txt,  the texts its vector was learned from and who wrote them
            task.json, writing.json          (concepts/corpus.py; none for a vector made outside learn)
            vector.safetensors, vector.json  the steering vector (keys v.<layer>) and its metadata (vectors.py)
            profile.json                     the profile (``measured_on`` names the engine and served copy),
                                             generation files
            <served>/                        one per served copy calibrated: calibration.json (fits the profile by its
                                             mtime), replies.jsonl (its conversations), grades.json (their grades)
        served/<served>/                     one served copy's (SERVED)
            norms.json                       its strength table, from its unsteered traces (steer.norm_table)
            plain-chats/<key>.jsonl          the unsteered conversations calibration compares with
        lenses/<lens>/                       its J-lenses (jlens), fits/<fit id>/ the /jlens learn fits (lens_fit_gpu)
    ~/.metalscope/concepts/<concept>/concept.json   a user concept's definition, {name, description}
    ~/.metalscope/trash/<when>-<concept>-<checkpoint>/   a concept's folder on one checkpoint ``trash`` moved
    ~/.metalscope/trash/<when>-<concept>-definition/     a definition ``trash`` moved (no checkpoint had the concept)
    ~/.metalscope/traces/, index.jsonl   the traces, each under its served name
    ~/.metalscope/expert_layouts/<checkpoint>.json   the expert map (layout.py), a cache kept outside the folders

A served copy's calibrations are under the checkpoint they were made on: after ``link`` gives it another, it has
none there until it is calibrated again. ``checkpoint.json``'s ``lens`` is the J-lens the checkpoint reads
(``"logit"``: the logit lens alone), ``jlens`` the one ``/jlens on`` goes back to. Identity is the declared name plus
the shape guard: ``check_shape`` compares what an engine or the ds4 proxy opens, and what oMLX installs its hooks
on (tracer.py: a mismatch is logged, the model served), against the recorded shape, recording it the first time
(the link offer needs it); ``link`` refuses a served name whose known shape differs.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import time
from pathlib import Path

from . import config

RECORD = "checkpoint.json"
LOCK = "checkpoint.lock"        # orders the record's writers (oMLX, the ds4 proxy, the command line)
SHAPE_KEYS = ("n_layers", "hc", "d_model")      # what the guard compares (width is hc * d_model; vocabularies are padded differently)


class ShapeMismatch(ValueError):
    """A served model's shape is not its checkpoint's."""


def models_dir() -> Path:
    return config.home() / "models"


def _records() -> dict[str, dict]:
    """Every checkpoint's record, by checkpoint name."""
    root = models_dir()
    out = {}
    for p in sorted(root.glob(f"*/{RECORD}")) if root.exists() else []:
        rec = config.read_json(p)
        if isinstance(rec, dict):
            out[p.parent.name] = rec
    return out


def checkpoint_of(model: str) -> str:
    """The checkpoint a served name (or a checkpoint's own name) is: the one whose record lists it, else itself."""
    name = config.short_name(model)
    for ckpt, rec in _records().items():
        if name in (rec.get("served") or []):
            return ckpt
    return name


def checkpoint_lookup():
    """checkpoint_of for a pass over many names (the index's entries): each name worked out once, as each lookup
    reads every checkpoint's record. None reads as no name."""
    return functools.lru_cache(maxsize=None)(lambda model: checkpoint_of(model or ""))


def model_dir(model: str) -> Path:
    return models_dir() / checkpoint_of(model)


def record(model: str) -> dict:
    """The checkpoint's record: ``{name, shape, served, lens, jlens}`` (shape None and served empty when nothing
    recorded them yet)."""
    ckpt = checkpoint_of(model)
    rec = config.read_json(models_dir() / ckpt / RECORD, {})
    if not isinstance(rec, dict):        # a list or a number in the file: every route reads the record
        rec = {}
    return dict(rec, name=ckpt, shape=rec.get("shape"), served=list(rec.get("served") or []))


def write_record(rec: dict) -> Path:
    path = models_dir() / rec["name"] / RECORD
    config.atomic_write(path, json.dumps(rec, indent=1) + "\n")
    return path


def _locked(ckpt: str):
    """The lock around a read, change and write of ``ckpt``'s record (one record's at a time: never two held)."""
    return config.file_lock(models_dir() / ckpt / LOCK)


@contextlib.contextmanager
def _record_locked(model: str):
    """``_locked`` for ``model``'s checkpoint as it is once the lock is held: a link that moved ``model`` meanwhile
    (or dropped its folder and the lock file with it) sends it to the new checkpoint's lock."""
    while True:
        ckpt = checkpoint_of(model)
        with _locked(ckpt):
            if checkpoint_of(model) == ckpt:
                yield ckpt
                return


def update(model: str, **changes) -> dict:
    """Set fields of the checkpoint's record (``lens``, ``jlens``...) and write it."""
    with _record_locked(model):
        rec = dict(record(model), **changes)
        write_record(rec)
    return rec


def served_names(model: str) -> list[str]:
    """The served names of the model's checkpoint (the checkpoint's own name when nothing is linked to it)."""
    rec = record(model)
    return rec["served"] or [rec["name"]]


# ---------------------------------------------------------------- per concept, by checkpoint

def concepts_dir() -> Path:
    return config.home() / "concepts"


def definition_path(concept: str) -> Path:
    """A user concept's definition, shared by every checkpoint (concepts.corpus.define)."""
    return concepts_dir() / concept / "concept.json"


def concept_dir(model: str, concept: str) -> Path:
    """A concept's folder for the model's checkpoint: its texts, vector, profile and generation files, and a folder
    per served copy calibrated (``calibration_dir``)."""
    return model_dir(model) / "concepts" / concept


def vector_path(model: str, concept: str) -> Path:
    """The steering vector (vectors.py); its metadata is ``vector.json`` beside it."""
    return concept_dir(model, concept) / "vector.safetensors"


def profile_path(model: str, concept: str) -> Path:
    return concept_dir(model, concept) / "profile.json"


def vector_paths(model: str) -> list[Path]:
    """Every vector of the model's checkpoint, by concept name (the concept is ``p.parent.name``)."""
    return sorted((model_dir(model) / "concepts").glob("[!.]*/vector.safetensors"))


def profile_paths(model: str) -> list[Path]:
    """Every profile of the model's checkpoint, by concept name (the concept is ``p.parent.name``)."""
    return sorted((model_dir(model) / "concepts").glob("[!.]*/profile.json"))


# ---------------------------------------------------------------- per served model

SERVED = "served"           # the folder of a checkpoint's served copies, by served name


def served_dir(served: str) -> Path:
    """One served copy's directory, under its checkpoint's: its strength table, its plain conversations."""
    return model_dir(served) / SERVED / config.short_name(served)


def norms_path(served: str) -> Path:
    """The served copy's strength table (steer.norm_table)."""
    return served_dir(served) / "norms.json"


CALIBRATION_VERSION = 4          # learn/calibrate.py's graded conversations; older calibrations are ignored


NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}$")   # a concept or vector name: a file name and a header token; no leading dot, at most 128


def deletable(model: str, concept: str) -> Path:
    """The concept's folder for the model's checkpoint, checked to be exactly that (``trash`` moves it): ValueError for
    a name that is not one folder (``.``, ``..``, a path, a leading dot), FileNotFoundError when there is none. When
    the checkpoint has nothing of it, no checkpoint has a folder for it and it has a definition (not a shipped name):
    the definition's folder (``concept delete`` then takes the description)."""
    ckpt = checkpoint_of(model)
    for what, name in (("concept", concept), ("checkpoint", ckpt)):
        if not NAME_RE.fullmatch(name or ""):
            raise ValueError(f"bad {what} name {name!r}: letters, digits, _ . - only, not starting with a dot")
    home = models_dir() / ckpt / "concepts"
    src = home / concept
    if src.is_symlink() or not src.is_dir():
        from .concepts.corpus import shipped_dir
        on = sorted(p.parent.parent.name for p in models_dir().glob(f"*/concepts/{concept}")
                    if p.is_dir() and not p.is_symlink())
        defined = definition_path(concept).parent
        if on and shipped_dir(concept) is None:
            raise FileNotFoundError(f"{concept} has nothing on {ckpt} (it is on {', '.join(on)})")
        if on or shipped_dir(concept) is not None or defined.is_symlink() or not definition_path(concept).is_file():
            raise FileNotFoundError(f"{concept} has nothing on {ckpt}")
        return defined
    if src.resolve().parent != home.resolve():
        raise ValueError(f"{src} is not a concept folder of {ckpt}")
    return src


def is_definition(folder: Path) -> bool:
    """Whether ``deletable``'s folder is a definition's (no model has the concept) rather than a checkpoint's."""
    return folder.parent == concepts_dir()


def trash(model: str, concept: str) -> Path:
    """``concept delete``: the concept's folder for the model's checkpoint (``deletable``: its texts, vector, profile
    and every served copy's calibration) renamed into ``trash/<when>-<concept>-<checkpoint>[-n]/`` (renaming it back
    restores it); its definition and other checkpoints' folders stay, and a ``concepts/`` (then checkpoint folder) it
    left empty goes, so it does not read as data. When no model has the concept, its definition's folder instead,
    into ``trash/<when>-<concept>-definition[-n]/``. Where it went. A rename that fails (another disk, permissions)
    raises with nothing moved: never a copy and delete."""
    src = deletable(model, concept)
    ckpt = checkpoint_of(model)
    root = config.home() / "trash"
    root.mkdir(parents=True, exist_ok=True)
    base = f"{time.strftime('%Y%m%d-%H%M%S')}-{concept}-{'definition' if is_definition(src) else ckpt}"
    dest, n = root / base, 1
    while dest.exists() or dest.is_symlink():
        dest, n = root / f"{base}-{n}", n + 1
    os.rename(src, dest)
    for d in () if is_definition(src) else (src.parent, src.parent.parent):
        try:
            d.rmdir()
        except OSError:
            break
    return dest


def calibration_dir(served: str, concept: str) -> Path:
    """A concept's chat calibration on the served copy, in the concept's folder for its checkpoint: calibration.json,
    replies.jsonl and grades.json."""
    return concept_dir(served, concept) / config.short_name(served)


def calibration_path(served: str, concept: str) -> Path:
    return calibration_dir(served, concept) / "calibration.json"


def replies_path(served: str, concept: str) -> Path:
    return calibration_dir(served, concept) / "replies.jsonl"


def grades_path(served: str, concept: str) -> Path:
    return calibration_dir(served, concept) / "grades.json"


def plain_chats_path(served: str, key: str) -> Path:
    """The served copy's unsteered calibration conversations for one context and budget (calibrate.plain_key)."""
    return served_dir(served) / "plain-chats" / f"{key}.jsonl"


def is_calibration(path: Path) -> bool:
    """Whether ``path`` holds a calibration of the current version (an older one counts as none)."""
    try:
        return json.loads(path.read_text()).get("version") == CALIBRATION_VERSION
    except (OSError, ValueError, AttributeError):
        return False


def calibrated_copies(model: str, concept: str) -> list[str]:
    """The served names of ``model``'s checkpoint that hold a current calibration of ``concept`` (fitting or not)."""
    ckpt = checkpoint_of(model)
    return sorted(p.parent.name for p in concept_dir(ckpt, concept).glob("*/calibration.json")
                  if checkpoint_of(p.parent.name) == ckpt and is_calibration(p))


def served_copy(served: str) -> dict:
    """What one served copy holds: ``{name, path, norms (bool), norm_traces (how many traces its table is from),
    calibrations: [concept, ...]}`` (its calibrations under its checkpoint)."""
    name = config.short_name(served)
    try:
        traces = len(config.read_json(norms_path(served)).get("traces") or [])
    except AttributeError:                       # no table, or not a table
        traces = None
    cals = (model_dir(served) / "concepts").glob(f"[!.]*/{name}/calibration.json")
    return dict(name=name, path=str(served_dir(served)), norms=norms_path(served).exists(), norm_traces=traces,
                calibrations=sorted(p.parents[1].name for p in cals if is_calibration(p)))


# ---------------------------------------------------------------- identity

# The model families metalscope has been run on, by engine, as the README's table names them (a test pins it there).
# A family is what the engine already knows of a model: oMLX's pool entry's config_model_type (the checkpoint's
# config.json model_type), ds4's family of the GGUF's architecture (ds4/vocab.py FAMILIES). Nothing gates on it: a
# model outside it is marked untested (/scope/api/status ``chat``, Pi's dock, doctor).
TESTED = {
    "omlx": {"qwen3_5": "Qwen3.5 2B, Qwen3.8 27B (MTP)", "qwen3_5_moe": "Qwen3.6 35B", "qwen4_exp": "Qwen3.8 Flash Next (MTP)",
             "inkling2_loop": "Inkling 2 Mini (MTP)"},
    "ds4": {"qwen4": "Qwen3.8 Flash Next", "deepseek41": "DeepSeek V4.1 Flash"},
}


def tested(engine: str, family) -> bool:
    """Whether metalscope has been run on a model of ``family`` on ``engine`` (TESTED)."""
    return family in TESTED.get(engine, {})


def shape_of(n_layers: int, hc: int, d_model: int) -> dict:
    return dict(n_layers=int(n_layers), width=int(hc) * int(d_model), hc=int(hc), d_model=int(d_model))


def shape_text(s) -> str:
    return f"{s['n_layers']} layers x {s['hc']} x {s['d_model']}" if s else "shape not recorded yet"


def check_shape(served: str, shape: dict) -> dict:
    """The guard where an engine opens and the ds4 proxy starts: ``shape`` (``shape_of``) against the checkpoint's
    recorded one, recorded now (with ``served`` listed) when there is none. ShapeMismatch when they differ."""
    name = config.short_name(served)
    with _record_locked(name):
        rec = record(name)
        have = rec["shape"]
        if have and any(int(have[k]) != int(shape[k]) for k in SHAPE_KEYS):
            raise ShapeMismatch(f"{name} is {shape_text(shape)}, but its checkpoint {rec['name']} is {shape_text(have)}: "
                                f"not the same model (metalscope model link {name} CHECKPOINT names the right one)")
        if not have or name not in rec["served"]:
            rec.update(shape=have or shape_of(*(shape[k] for k in SHAPE_KEYS)),
                       served=rec["served"] + ([name] if name not in rec["served"] else []))
            write_record(rec)
    return rec


FINDER = ".DS_Store"
KEPT = (RECORD, LOCK, "experts-tally.npz", "experts-tally.lock", FINDER)   # not data: the record, the tally (link moves it)


def _holds_data(ckpt: str) -> list[str]:
    """What a checkpoint's folder holds besides its record, its expert tally and its own served copy's folder (when
    that is the one ``link`` moves, ``served_dir``): its concepts/, lenses/, fits/, any other file, by name (a folder
    with a /). Empty when nothing."""
    d = models_dir() / ckpt
    moved = served_dir(ckpt)

    def data(p: Path) -> bool:
        if p.name in KEPT or p.name.startswith(".experts-tally."):
            return False
        if p.name != SERVED or not p.is_dir() or p.is_symlink():
            return True
        return any(q.name != FINDER and q != moved for q in p.iterdir())
    return [p.name + "/" * p.is_dir() for p in sorted(d.iterdir()) if data(p)] if d.is_dir() else []


def _drop(d: Path) -> bool:
    """A served name's own checkpoint folder removed once link has moved what it held: its record, its lock, the
    tally's lock and Finder files deleted, then the folders left empty; never a file link did not account for
    (anything else keeps the folder, and link writes the record back). Whether it went."""
    for p in (d / SERVED / FINDER, d / RECORD, d / LOCK, d / "experts-tally.lock", d / FINDER):
        p.unlink(missing_ok=True)
    for p in (d / SERVED, d):
        try:
            p.rmdir()
        except OSError:
            pass
    return not d.exists()


def _fits(ckpt: str, served: str) -> list[Path]:
    """The /jlens learn fits of ``served`` under ``ckpt`` (lens_fit_gpu.fit_dir: ``lens-<served>-<12 hex>``)."""
    fit = re.compile(rf"lens-{re.escape(served)}-[0-9a-f]{{12}}")
    return sorted(p for p in (models_dir() / ckpt / "fits").glob("lens-*") if fit.fullmatch(p.name))


def check_link(served: str, checkpoint: str) -> None:
    """Why ``link(served, checkpoint)`` would be refused, raised as it raises it (ValueError, ShapeMismatch): a name
    missing, CHECKPOINT a served name of another checkpoint, a job running on SERVED, SERVED holding data of its own,
    another shape."""
    name, ckpt = config.short_name(served), config.short_name(checkpoint)
    if not name or not ckpt:
        raise ValueError("name a served model and a checkpoint")
    if checkpoint_of(ckpt) != ckpt:
        raise ValueError(f"{ckpt} is a served name of {checkpoint_of(ckpt)}, not a checkpoint")
    if name == ckpt:
        return
    from .learn import state
    job = state.running()
    if job and config.short_name(job.get("model")) == name:     # it writes to SERVED's folders as they are now
        raise ValueError(f"cannot link {name} now: {state.running_text(job)}: wait for it or stop it: "
                         f"metalscope jobs cancel {job['job']}")
    own = _holds_data(name)
    if own:
        raise ValueError(f"{name} has data of its own in {models_dir() / name} ({', '.join(own)}): to link it, move what "
                         f"is inside each into the same folder under {models_dir() / ckpt}, then link again")
    was, target = record(name)["shape"], record(ckpt)["shape"]
    if was and target and any(int(was[k]) != int(target[k]) for k in SHAPE_KEYS):
        raise ShapeMismatch(f"{name} is {shape_text(was)}, {ckpt} is {shape_text(target)}: not the same model")


def link(served: str, checkpoint: str) -> dict:
    """``metalscope model link SERVED CHECKPOINT``: SERVED's data is CHECKPOINT's from now on. Its own folder (norms,
    plain chats) and its /jlens learn fits move there first; one already there with nothing left to move is taken as
    moved, so a link that stopped part way (an error, a crash) finishes when run again. Refused (``check_link``) when
    SERVED's recorded shape is not the checkpoint's, when SERVED is itself a checkpoint with data of its own, while a
    job runs on SERVED, or when CHECKPOINT is a served name of another checkpoint; and before anything moves, when
    both CHECKPOINT and SERVED's checkpoint have a folder by one name. SERVED leaves any other checkpoint's list (a
    record left listing nothing and holding nothing else is removed, never a file link did not move)."""
    check_link(served, checkpoint)
    name, ckpt = config.short_name(served), config.short_name(checkpoint)
    target = record(ckpt)
    if name == ckpt:
        return target
    was, old = record(name), checkpoint_of(name)
    moves = [(served_dir(name), models_dir() / ckpt / SERVED / name)]
    if old != ckpt:
        moves += [(p, models_dir() / ckpt / "fits" / p.name) for p in _fits(old, name)]
    moves = [(src, dst) for src, dst in moves if src != dst and (src.exists() or src.is_symlink())]
    for src, dst in moves:                       # all checked before any moves
        if dst.exists() or dst.is_symlink():
            raise ValueError(f"{ckpt} already has {dst} and {name} has {src}: move one aside")
    for src, dst in moves:                       # before any record changes
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
        for d in (src.parent, src.parent.parent):    # nothing left of it: no folder that reads as a checkpoint
            try:
                d.rmdir()
            except OSError:
                break
    from . import tally
    tally.move(name, ckpt)
    for other, rec in _records().items():
        if other == ckpt or name not in (rec.get("served") or []):
            continue
        with _locked(other):                     # read again under its lock; one record's lock at a time
            rec = config.read_json(models_dir() / other / RECORD)
            if not isinstance(rec, dict) or name not in (rec.get("served") or []):
                continue
            rec["served"] = [s for s in rec["served"] if s != name]
            if not (other == name and not rec["served"] and not _holds_data(other) and _drop(models_dir() / other)):
                write_record(dict(rec, name=other))
    with _locked(ckpt):
        target = record(ckpt)
        target.update(shape=target["shape"] or was["shape"],
                      served=target["served"] + ([name] if name not in target["served"] else []))
        write_record(target)
    return target


def concept_count(model: str) -> int:
    """How many concepts the model's checkpoint has a profile of: what the Steer page lists (a word vector is none)."""
    return len(profile_paths(model))


def link_offer(served: str) -> dict:
    """What the dock offers a served name that may be a model already here (files only): ``{model, checkpoint,
    linked, own, shape: {n_layers, width} | None, candidates: [{name, concepts, lenses}]}``. ``linked``: another
    checkpoint lists it; ``own``: it holds data of its own (``_holds_data``, what ``link`` refuses); ``candidates``:
    the other checkpoints of its recorded shape (SHAPE_KEYS) holding a concept or a lens, most of both first, then
    by name; none when it is linked, has data of its own or no shape recorded. Nothing says they are one model:
    the user does."""
    from . import jlens
    name = config.short_name(served)
    ckpt, shape = checkpoint_of(name), record(name)["shape"]
    linked, own = ckpt != name, bool(_holds_data(name))
    found = []
    for other, rec in _records().items() if shape and not linked and not own else ():
        have = rec.get("shape")
        if other == name or not have or checkpoint_of(other) != other or any(int(have[k]) != int(shape[k]) for k in SHAPE_KEYS):
            continue
        n, lenses = concept_count(other), len(jlens.lens_dirs(other))
        if n or lenses:
            found.append(dict(name=other, concepts=n, lenses=lenses))
    return dict(model=name, checkpoint=ckpt, linked=linked, own=own,
                shape=dict(n_layers=int(shape["n_layers"]), width=int(shape["hc"]) * int(shape["d_model"])) if shape else None,
                candidates=sorted(found, key=lambda c: (-(c["concepts"] + c["lenses"]), c["name"])))


def checkpoints() -> list[dict]:
    """Every checkpoint with a directory: its record, how much it holds and, in ``copies``, each of its served
    names that holds a strength table or calibrations (``served_copy``) (``metalscope model list``)."""
    root = models_dir()
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        rec = record(d.name) if checkpoint_of(d.name) == d.name else dict(name=d.name, shape=None, served=[])
        copies = [served_copy(s) for s in (rec["served"] or [d.name])]
        out.append(dict(rec, path=str(d), vectors=len(vector_paths(d.name)), profiles=len(profile_paths(d.name)),
                        copies=[c for c in copies if c["norms"] or c["calibrations"]]))
    return out

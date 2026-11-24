"""Sharing concepts: ``concept pack`` writes a package of a checkpoint's concepts, ``concept pull`` places one under
the checkpoint of the model the user serves.

A package works like a LoRA adapter: it declares the base model it was made on (the packer types its Hugging Face
id, nothing is read from weight files) and the user names the served model to use it with. Only the shape is checked
(models.SHAPE_KEYS), the manifest's and each vector file's own tensors: a vector of another shape cannot steer, so a
mismatch refuses; anything else is the user's to try. One package is one Hugging Face repo, for one base model. Each
concept's texts are kept apart from what was learned from them on the packer's checkpoint (on disk both are in one
folder, models.concept_dir), with each served copy's calibration kept as a reference only::

    manifest.json        {format, base_model, license, checkpoint, shape, made_with, made, made_on: [{served, engine}],
                          concepts: [{name, steer_layer}]}
    README.md            the model card: YAML front matter (base_model, adapter, license, tags), then per concept what
                         it is, who wrote its texts, how it did on the packer's copy; setup; licences
    LICENSE              the file the packer gave (--license-file), else a short note naming the licence
    <concept>/           concept.json, pos.txt, neg.txt, task.json, writing.json (the texts its vector was learned from)
        <checkpoint>/    vector.safetensors, vector.json, profile.json and the other files learn wrote there
            reference/<served>/   calibration.json, replies.jsonl, grades.json: how it did on the packer's copy

Pack takes the concepts that ship with metalscope (data/concepts/) learned on the checkpoint; the packer's own need
``names`` (or ``all_``). The texts it packs are the checkpoint's own (concepts.texts_dir); a concept with none there
goes without. Pull copies the texts, vector and profile into the concept's folder on the user's checkpoint
(the profile's ``model`` rewritten to it, the packer's checkpoint kept in ``measured_on``, where it came from in
``source``); references stay in the download. A calibration belongs to the
served copy that ran it, so the user calibrates their own (``concept calibrate``): no default dose comes with a
package. Uploading is the user's (``hf upload``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from . import config, models, store
from .concepts.corpus import TEXTS

FORMAT = 1
REFERENCE = "reference"
REFERENCE_FILES = ("calibration.json", "replies.jsonl", "grades.json")
SUFFIXES = (".safetensors", ".json", ".jsonl", ".txt")   # what a package's checkpoint folder may hold
DTYPES = {"F32": 4, "F16": 2, "BF16": 2}            # the vector's tensor types load_vector reads (cast to float32)
HF_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*$")
REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-/]*$")   # a branch, tag or commit of a repo
LICENSE_ID = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")  # a Hugging Face licence id: apache-2.0, mit, other...
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")   # C0 but tab and newline, DEL, C1: what drives a terminal
TAGS = ("steering-vectors", "activation-steering", "representation-engineering", "apple-silicon", "metalscope")
ENGINE_TAGS = {"mlx": "mlx", "ds4": "gguf"}         # what the served copies' weights were


def install(extra: str = "") -> str:
    """The install line, in one place: PyPI once it is there, else the git URL."""
    pkg = f"metalscope[{extra}]" if extra else "metalscope"
    return f'`uv tool install "{pkg}"` (PyPI, coming) or `uv tool install "{pkg} @ git+{config.REPO_URL}"`'


def _name(s, what: str) -> str:
    if not isinstance(s, str) or not models.NAME_RE.fullmatch(s):
        raise ValueError(f"the package names a bad {what} {s!r}")
    return s


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("metalscope")
    except PackageNotFoundError:
        return "unknown"


def _inside(p: Path, roots: tuple[Path, ...]) -> bool:
    """Not a link, or a link to somewhere inside one of ``roots`` (resolved: the package's folder, and for a Hugging
    Face cache snapshot its repo's blobs/, where its files are links to); a link elsewhere would copy one of the
    user's files into the model's folder."""
    return not p.is_symlink() or any(p.resolve().is_relative_to(r) for r in roots)


def _files(d: Path, roots: tuple[Path, ...] = ()) -> list[Path]:
    """The files directly in ``d`` (no folders or dotfiles: temp files start with a dot), a link only into ``roots``."""
    return sorted(p for p in d.iterdir() if p.is_file() and models.NAME_RE.fullmatch(p.name) and _inside(p, roots)) if d.is_dir() else []


def shipped() -> set[str]:
    """The concepts that ship with metalscope (data/concepts/)."""
    from .concepts.corpus import DATA_DIR
    return {p.name for p in DATA_DIR.iterdir() if (p / "concept.json").is_file()} if DATA_DIR.is_dir() else set()


# ---------------------------------------------------------------- pack

def pack(model: str, out_dir, base_model: str, names: Optional[list[str]] = None, *, license: str = "",
         license_name: str = "", license_file=None, all_: bool = False) -> dict:
    """Write a package of ``model``'s checkpoint into ``out_dir``: the shipped concepts (``shipped``) with a vector
    and a profile there, ``names``, or with ``all_`` every one. ``base_model`` is the base model's Hugging Face id,
    ``license`` the package's (a Hugging Face licence id; ``other`` needs ``license_name`` and ``license_file``),
    copied in as LICENSE from ``license_file`` or written as a note naming it. ``out_dir`` is new, empty or an earlier
    package (whose concept folders, README and LICENSE are replaced). The manifest written, with ``left_out``: the
    concepts with a vector and profile not packed, and ``no_texts``: the ones packed without texts (none on the
    checkpoint). ValueError naming what is missing."""
    from .concepts import calibration_of, texts_dir
    from .concepts.stats import reader_best
    ckpt = models.checkpoint_of(model)
    rec = models.record(ckpt)
    base_model, license, license_name = (base_model or "").strip(), (license or "").strip(), (license_name or "").strip()
    if not HF_REPO.fullmatch(base_model):
        raise ValueError("--base-model is the base model's Hugging Face id, as its publisher names it "
                         f"(e.g. Qwen/Qwen3.6-35B-A3B){f', not {base_model!r}' if base_model else ''}")
    if not LICENSE_ID.fullmatch(license):
        raise ValueError("--license is the package's licence, as a Hugging Face licence id (usually the base model's: "
                         f"apache-2.0, mit, or other with --license-name and --license-file){f', not {license!r}' if license else ''}")
    lic = Path(license_file).expanduser() if license_file else None
    if lic is not None and not lic.is_file():
        raise ValueError(f"--license-file {lic} is not a file")
    if license == "other" and (not LICENSE_ID.fullmatch(license_name) or lic is None):
        raise ValueError("--license other needs --license-name (e.g. qwen-community-1.0) and --license-file (its text)")
    if names and all_:
        raise ValueError("name the concepts or pass --all, not both")
    if not rec["shape"]:
        raise ValueError(f"{ckpt}'s shape is not recorded yet: serve it once first (an engine records it when it opens)")
    have = sorted(p.parent.name for p in models.profile_paths(ckpt)
                  if models.vector_path(ckpt, p.parent.name).exists())
    if names:
        missing = [n for n in names if n not in have]
        if missing:
            raise ValueError(f"no vector and profile on {ckpt} for {', '.join(missing)} (packable: {', '.join(have) or 'none'})")
    ships = shipped()
    names = list(dict.fromkeys(names)) if names else have if all_ else [n for n in have if n in ships]
    left_out = [n for n in have if n not in names]
    if not names:
        raise ValueError(f"no concept of {ckpt} has a vector and a profile ({models.model_dir(ckpt) / 'concepts'})" if not have
                         else f"none of {ckpt}'s concepts ships with metalscope: name the ones to pack, or --all "
                              f"(packable: {', '.join(have)})")
    out = Path(out_dir).expanduser()
    old = config.read_json(out / "manifest.json")
    if out.exists() and any(out.iterdir()) and not isinstance(old, dict):
        raise ValueError(f"{out} is not empty and holds no package (manifest.json): pack into a new or empty folder")
    for c in (old or {}).get("concepts") or []:
        if isinstance(c, dict) and isinstance(c.get("name"), str) and models.NAME_RE.fullmatch(c["name"]):
            shutil.rmtree(out / c["name"], ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    made_on, entries, readme, no_texts = [], [], [], []
    for name in names:
        dest = out / name
        texts = texts_dir(ckpt, name)                # the texts its vector was learned from
        if texts is None:
            no_texts.append(name)
        dest.mkdir(parents=True, exist_ok=True)
        for f in TEXTS if texts is not None else ():
            if (texts / f).is_file():
                shutil.copyfile(texts / f, dest / f)
        meta = config.read_json(dest / "concept.json")
        if isinstance(meta, dict) and isinstance(meta.get("learned"), dict) and "corpus" in meta["learned"]:
            meta["learned"].pop("corpus")              # where the packer's texts were, on the packer's disk
            config.atomic_write(dest / "concept.json", json.dumps(meta, indent=1, ensure_ascii=False))
        src, to = models.concept_dir(ckpt, name), dest / ckpt
        to.mkdir(parents=True, exist_ok=True)
        for f in _files(src):
            if f.suffix in SUFFIXES and f.name not in TEXTS:      # the texts beside it are packed once, above
                shutil.copyfile(f, to / f.name)
        prof = config.read_json(models.profile_path(ckpt, name), {})
        on = prof.get("measured_on") or {}
        if on.get("served"):
            made_on.append(dict(served=on["served"], engine=on.get("engine")))
        refs = []
        for served in models.calibrated_copies(ckpt, name):
            cal = calibration_of(served, name)
            if cal is None:                          # out of date: made on another profile
                continue
            ref = to / REFERENCE / served
            ref.mkdir(parents=True, exist_ok=True)
            for f in REFERENCE_FILES:
                p = models.calibration_dir(served, name) / f
                if p.is_file():
                    shutil.copyfile(p, ref / f)
            refs.append((served, cal))
            if not any(m["served"] == served for m in made_on):
                made_on.append(dict(served=served, engine=None))
        layer = reader_best(prof)[0].get("layer")
        entries.append(dict(name=name, steer_layer=layer))
        readme += _readme_concept(name, prof, texts, layer, refs)
    made_on = list({m["served"]: m for m in made_on}.values())
    manifest = dict(format=FORMAT, base_model=base_model, license=license, checkpoint=ckpt, shape=rec["shape"],
                    made_with=_version(), made=time.strftime("%Y-%m-%dT%H:%M:%S"), made_on=made_on, concepts=entries)
    if license_name:
        manifest["license_name"] = license_name
    config.atomic_write(out / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    config.atomic_write(out / "README.md", _front_matter(manifest) + _readme(manifest, readme))
    if lic is not None:
        shutil.copyfile(lic, out / "LICENSE")
    else:
        config.atomic_write(out / "LICENSE", _license_note(manifest))
    return dict(manifest, left_out=left_out, no_texts=no_texts)


def _front_matter(m: dict) -> str:
    """The model card's YAML metadata (Hugging Face reads it): the base model, as an adapter of it, the licence,
    the library, tags. Strings quoted as JSON (valid YAML)."""
    q = json.dumps
    engines = sorted({ENGINE_TAGS[x["engine"]] for x in m["made_on"] if x.get("engine") in ENGINE_TAGS})
    lines = ["---", f"base_model: {q(m['base_model'])}", "base_model_relation: adapter", f"license: {q(m['license'])}"]
    if m.get("license_name"):
        lines += [f"license_name: {q(m['license_name'])}", "license_link: LICENSE"]
    lines += ["library_name: metalscope", "tags:", *(f"- {t}" for t in (*TAGS, *engines)), "---", ""]
    return "\n".join(lines) + "\n"


def _license_note(m: dict) -> str:
    """LICENSE when the packer gave no file: which licence, and where its text and the base model's are. No legal
    text of its own."""
    lic = m.get("license_name") or m["license"]
    return "\n".join([
        f"metalscope concepts for {m['base_model']}", "",
        f"Licence: {lic}.",
        "This file names the licence; it does not hold its text.",
        f"The vectors were learned on {m['base_model']}, so its licence applies to them too: "
        f"https://huggingface.co/{m['base_model']}", ""])


def _readme_concept(name: str, prof: dict, texts: Optional[Path], layer, refs: list) -> list[str]:
    """The concept's lines in the README; ``refs``: (served, its calibration). A calibration is how it did on the
    packer's copy: a reference, never a default dose."""
    from .concepts import texts_line
    lines = [f"### {name}", "", f"{prof.get('title') or name}: {prof.get('description') or '(no description)'}", ""]
    lines.append("- " + (texts_line(texts) if texts is not None else "texts: none on the packer's model"))
    lines.append(f"- steers at layer {layer}" if layer is not None else "- no steer layer: its sweep found no dose that moved the choice")
    if not refs:
        lines.append("- on the packer's copies: not calibrated in conversation")
    for served, cal in refs:
        lines.append(f"- on the packer's copy {served}: {cal.get('good')} of {cal.get('turns', 18)} replies good at dose "
                     f"{cal['default_dose']:g}, its best (a good one: style, coherence and usefulness 2 of 3); graded by "
                     f"{(cal.get('grader') or {}).get('model') or served}")
    return lines + [""]


def _readme(m: dict, concepts: list[str]) -> str:
    first = next((c["name"] for c in m["concepts"] if c.get("steer_layer") is not None), "NAME")
    copies = ", ".join(x["served"] + (f" ({x['engine']})" if x.get("engine") else "") for x in m["made_on"])
    return "\n".join([
        f"# metalscope concepts for {m['base_model']}", "",
        f"Steering vectors for metalscope, learned on {m['checkpoint']} ({models.shape_text(m['shape'])}) with "
        f"metalscope {m['made_with']}" + (f", measured on {copies}." if copies else "."),
        "Each concept's texts (the contrast pairs and the task) come with it, so you can learn it again on another model.",
        "The calibrations under `<concept>/<checkpoint>/reference/` show how each did on the packer's copy.",
        "They are references, not defaults: calibrate your own with `metalscope concept calibrate`.", "",
        "## Concepts", "", *concepts,
        "## Setup", "",
        f"1. Install metalscope with the share extra: {install('share')}.",
        "2. An engine with metalscope: oMLX with the hook (`metalscope omlx install`), or ds4 built with metalscope's "
        "patch (`metalscope ds4 install`).",
        f"3. A copy of {m['base_model']} served by it, at the shape above. Serve it once so metalscope records its shape.",
        "4. Optional: Pi with the metalscope extension (`metalscope pi install`) and the provider for that server.",
        "5. `metalscope on`, then one real chat with the served model (it measures what a dose means on your copy).",
        "6. `metalscope concept pull THIS_REPO[@REVISION] SERVED` (or the downloaded folder), where SERVED is your "
        "served model's name.",
        f"7. `metalscope concept calibrate {first} SERVED` finds your copy's dose; `/steer {first}` "
        "in Pi then uses it.",
        f"8. To try a dose by hand first: `/steer {first} DOSE`.", "",
        "## Licences", "",
        f"This package: {m.get('license_name') or m['license']} (LICENSE).",
        f"The vectors are derived from {m['base_model']}: its licence applies to them too.", ""])


# ---------------------------------------------------------------- pull

def _download(repo: str, revision: Optional[str]) -> Path:
    """A repo's snapshot folder at ``revision`` (None: its default branch), downloaded with huggingface_hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ValueError(f"pulling {repo} from Hugging Face needs huggingface_hub: {install('share')} "
                         "(or download it yourself and pull the folder)") from None
    # Only what a package holds and pull reads (huggingface_hub's * spans folders): a model repo named by mistake
    # downloads no weights, and pull then finds no manifest.
    allow = ["manifest.json", *(f"*/{f}" for f in TEXTS), *(f"*/*/*{s}" for s in SUFFIXES)]
    return Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=allow))


def fetch(source: str) -> tuple[Path, dict]:
    """A package's folder and where it came from: a local folder (``{folder}``), or a Hugging Face repo
    ``owner/name[@revision]`` downloaded at that revision (``{repo, revision, commit}``: revision as asked, None for
    the default branch; commit the snapshot's, its folder name in the download cache)."""
    d = Path(source).expanduser()
    if d.is_dir():
        return d.resolve(), dict(folder=str(d.resolve()))
    repo, at, revision = source.partition("@")
    if not HF_REPO.fullmatch(repo) or (at and (not REVISION.fullmatch(revision) or ".." in revision)):
        raise ValueError(f"{source} is neither a folder nor a Hugging Face repo (owner/name, or owner/name@revision)")
    d = _download(repo, revision or None)
    return d, dict(repo=repo, revision=revision or None, commit=d.name)


def read_manifest(d: Path) -> dict:
    m = config.read_json(d / "manifest.json")
    if not isinstance(m, dict):
        raise ValueError(f"{d} holds no package (manifest.json)")
    if m.get("format") != FORMAT:
        raise ValueError(f"{d}'s package is format {m.get('format')}; this metalscope reads format {FORMAT} (update it)")
    shape = m.get("shape")
    if not isinstance(shape, dict) or any(not isinstance(shape.get(k), int) for k in models.SHAPE_KEYS):
        raise ValueError(f"{d}'s manifest has no shape")
    _name(m.get("checkpoint"), "checkpoint")
    concepts = m.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        raise ValueError(f"{d}'s manifest lists no concepts")
    for c in concepts:
        _name((c or {}).get("name") if isinstance(c, dict) else None, "concept")
    return m


def _controls(v) -> bool:
    """Whether a JSON value (or a text) holds a control character (``CONTROL``) in any string, keys included."""
    if isinstance(v, str):
        return bool(CONTROL.search(v))
    if isinstance(v, dict):
        return any(_controls(k) or _controls(x) for k, x in v.items())
    return isinstance(v, list) and any(_controls(x) for x in v)


def _check_package(d: Path, roots: tuple[Path, ...]) -> None:
    """Before anything of the package is read: every link in it stays inside it (``_inside``), and what pull prints
    as it is holds no control character (one could drive the terminal): file names, the manifest's strings, each
    concept's concept.json strings and texts (the profile's title and description: ``_check_profile``). Vocabulary
    tokens and model replies elsewhere (a profile's lens words, a ladder's broken replies) may hold any. ValueError
    naming the file."""
    for p in sorted(d.rglob("*")):
        rel = str(p.relative_to(d))
        if CONTROL.search(rel):
            raise ValueError(f"a file name in the package holds a control character: {rel!r}")
        if not _inside(p, roots):
            raise ValueError(f"the package's {rel} links outside it: copy the file into the package instead")
    for p in sorted({d / "manifest.json", *d.glob("*/concept.json"), *d.glob("*/*.txt")}):
        if not p.is_file():
            continue
        rel, text = str(p.relative_to(d)), p.read_text(errors="replace")   # universal newlines: CRLF is a newline
        try:
            value = text if p.suffix == ".txt" else json.loads(text)
        except ValueError:
            raise ValueError(f"the package's {rel} is not JSON") from None
        if _controls(value):
            raise ValueError(f"the package's {rel} holds a control character")


def _check_profile(p: Path, name: str, where: str) -> None:
    """The package's profile as the server lists it (``concepts.summarize``, what /scope/api/concepts runs on every
    profile of a checkpoint), title and description as text, and what ``concept show`` prints of it
    (``concepts.summary_lines``, its measured-on copy included) free of control characters: ValueError when it
    cannot be read so."""
    from .concepts import summarize, summary_lines
    prof = config.read_json(p)
    try:
        if not isinstance(prof, dict):
            raise ValueError("not a JSON object")
        if any(not isinstance(prof.get(k) or "", str) for k in ("title", "description")):
            raise ValueError("its title and description are text")
        if _controls([prof.get("title") or "", prof.get("description") or ""]):
            raise ValueError("its title or description holds a control character")
        summarize(dict(prof, concept=name, model=""))          # pull sets both
        if _controls(summary_lines(dict(prof, concept=name, model=""))):
            raise ValueError("what concept show prints of it holds a control character")
    except Exception as exc:
        raise ValueError(f"the package's {where} cannot be read ({str(exc) or type(exc).__name__})") from None


def tensors(p: Path) -> dict[str, tuple[str, list, int]]:
    """The tensors a safetensors file declares, ``{key: (dtype, shape, bytes)}``, from its header alone (8 bytes of
    header length, then JSON; no tensor loaded). ValueError when it is not one."""
    size = p.stat().st_size
    with p.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        if size < 10 or not 2 <= n <= min(size - 8, 100 << 20):
            raise ValueError("no safetensors header")
        try:
            head = json.loads(f.read(n))
        except ValueError:
            raise ValueError("its header is not JSON") from None
    if not isinstance(head, dict):
        raise ValueError("its header is not a table of tensors")
    out, body = {}, size - 8 - n
    for k, t in head.items():
        if k == "__metadata__":
            continue
        t = t if isinstance(t, dict) else {}
        off, dims = t.get("data_offsets"), t.get("shape")
        if not (isinstance(t.get("dtype"), str) and isinstance(dims, list) and all(isinstance(x, int) and x >= 0 for x in dims)
                and isinstance(off, list) and len(off) == 2 and all(isinstance(x, int) for x in off)
                and 0 <= off[0] <= off[1] <= body):
            raise ValueError(f"its tensor {k!r} is not described as safetensors describes one")
        out[k] = (t["dtype"], t["shape"], off[1] - off[0])
    return out


def check_vector(p: Path, shape: dict, where: str) -> None:
    """The vector file's real tensors against the package's shape (models.SHAPE_KEYS): safetensors (the one format
    read), keys ``v.<layer>`` with a layer the model has, each [hc x d_model] wide in a float type load_vector reads.
    ValueError (models.ShapeMismatch for the shape) naming what is wrong."""
    try:
        have = tensors(p)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{where} is not a safetensors file ({exc}): metalscope reads safetensors only") from None
    width = int(shape["hc"]) * int(shape["d_model"])
    if not have:
        raise ValueError(f"{where} holds no tensors")
    for k, (dtype, dims, nbytes) in sorted(have.items()):
        m = re.fullmatch(r"v\.(\d+)", k)
        if not m:
            raise ValueError(f"{where} holds a tensor {k!r}: a vector holds one per layer, v.<layer>")
        if int(m[1]) >= int(shape["n_layers"]):
            raise models.ShapeMismatch(f"{where} has a vector for layer {m[1]}; the manifest's shape "
                                       f"({models.shape_text(shape)}) has {shape['n_layers']} layers")
        if dims != [width]:
            raise models.ShapeMismatch(f"{where}'s {k} is {' x '.join(map(str, dims)) or 'a scalar'}; the manifest's "
                                       f"shape ({models.shape_text(shape)}) needs {width}")
        if dtype not in DTYPES:
            raise ValueError(f"{where}'s {k} is {dtype}; a vector is {', '.join(DTYPES)}")
        if nbytes != width * DTYPES[dtype]:
            raise ValueError(f"{where}'s {k} holds {nbytes} bytes, not the {width * DTYPES[dtype]} its shape needs")


def _from(package: Optional[str]) -> Optional[str]:
    """``source`` of ``source@revision``."""
    return package.rsplit("@", 1)[0] if isinstance(package, str) and "@" in package else None


def check_shape(served: str, manifest: dict) -> dict:
    """``served``'s checkpoint record, when its shape is the package's: ValueError when it is not recorded yet,
    models.ShapeMismatch when it differs (the one refusal a package can't get past: its vectors can't steer)."""
    name = config.short_name(served)
    rec = models.record(name)
    if not rec["shape"]:
        raise ValueError(f"{name}'s shape is not recorded yet: serve it once first (metalscope records it when an "
                         f"engine opens the model), then pull again")
    want = manifest["shape"]
    if any(int(rec["shape"][k]) != int(want[k]) for k in models.SHAPE_KEYS):
        raise models.ShapeMismatch(f"{name} is {models.shape_text(rec['shape'])}, the package is for "
                                   f"{models.shape_text(want)} ({manifest.get('base_model')}): not the same model")
    return rec


def checkpoint_for(served: str, manifest: dict) -> tuple[str, bool]:
    """The checkpoint a package lands under for ``served`` (its shape checked) and whether ``served`` was linked to it
    now: its own checkpoint when it is linked to one or holds data of its own, else the package's (created and linked
    here, models.link)."""
    name = config.short_name(served)
    ckpt, theirs = check_shape(name, manifest)["name"], manifest["checkpoint"]
    if ckpt != name or ckpt == theirs or models._holds_data(name) or models.checkpoint_of(theirs) != theirs:
        return ckpt, False
    return models.link(name, theirs)["name"], True


def pull(source: str, served: str) -> dict:
    """``concept pull SOURCE SERVED``: the package's concepts placed for ``served`` (see the module docstring).
    ``SOURCE`` is a folder or ``owner/name[@revision]``. Everything is checked before anything is written: a shape
    mismatch refuses (the manifest's, and each vector file's tensors), and so does a vector in another format.

    By name, on the user's checkpoint: what this package placed before is replaced by its newer version; a concept the
    model has already (its profile, else its vector, else its texts, from no package or another) is kept, never
    overwritten (``kept``: the user's, ``kept_other``: another package's). ``{checkpoint, linked, package, source,
    base_model, placed, kept, kept_other, unchanged, references: {concept:
    [(served, calibration)]}, layers: {concept: steer layer}}``."""
    from .concepts import texts_dir
    d, origin = fetch(source)
    # A Hugging Face cache snapshot (pulled from the hub, or given as a folder) links into ../../blobs: its files may
    # be there too (blobs/ itself no link elsewhere), nothing else of the repo's cache folder.
    blobs = d.resolve().parent.parent / "blobs"
    hub = d.parent.name == "snapshots" and blobs.is_dir() and blobs.resolve() == blobs
    roots = (d.resolve(), blobs) if hub else (d.resolve(),)
    _check_package(d, roots)
    m = read_manifest(d)
    src = origin.get("repo") or origin["folder"]
    package = f"{src}@{origin.get('commit') or m.get('made') or ''}"
    theirs = m["checkpoint"]
    check_shape(served, m)
    for c in m["concepts"]:
        name = c["name"]
        pkg = d / name
        vec, prof = pkg / theirs / "vector.safetensors", pkg / theirs / "profile.json"
        if not vec.is_file() or not prof.is_file():
            raise ValueError(f"the package's {name} has no vector and profile under {name}/{theirs}/")
        # Texts there would land in the model's folder unchecked (_check_package checks <concept>/'s only).
        other = [f.name for f in _files(pkg / theirs, roots) if f.suffix not in SUFFIXES or f.name in TEXTS]
        if other:
            raise ValueError(f"the package's {name}/{theirs}/ holds {', '.join(other)}, which metalscope does not read")
        check_vector(vec, m["shape"], f"the package's {name}/{theirs}/vector.safetensors")
        if not isinstance(config.read_json(pkg / theirs / "vector.json", {}), dict):
            raise ValueError(f"the package's {name}/{theirs}/vector.json is not a JSON object")
        _check_profile(prof, name, f"{name}/{theirs}/profile.json")
    ckpt, linked = checkpoint_for(served, m)
    out = dict(checkpoint=ckpt, linked=linked, package=package, source=origin,
               base_model=m.get("base_model"), placed=[], kept=[], kept_other=[], unchanged=[], references={},
               layers={c["name"]: c.get("steer_layer") for c in m["concepts"]})
    with models._locked(ckpt):                   # one pull at a time on a checkpoint: each sees the other's whole
        for c in m["concepts"]:
            name, pkg = c["name"], d / c["name"]
            out["references"][name] = _references(pkg / theirs / REFERENCE)
            _unpart(models.concept_dir(ckpt, name))       # a pull killed part way gives back what it held
            mine = config.read_json(models.profile_path(ckpt, name))
            if not isinstance(mine, dict) and models.vector_path(ckpt, name).exists():
                mine = config.read_json(models.concept_dir(ckpt, name) / "vector.json")   # a vector without a profile
                mine = mine if isinstance(mine, dict) else {}
            if not isinstance(mine, dict) and texts_dir(ckpt, name) is not None:
                mine = {}                                # texts only (a run that never got a profile)
            if isinstance(mine, dict) and mine.get("package") == package:
                out["unchanged"].append(name)
                continue
            if isinstance(mine, dict) and _from(mine.get("package")) != src:     # never overwritten
                out["kept_other" if mine.get("package") else "kept"].append(name)
                continue
            _place(name, pkg, theirs, roots, ckpt, package, origin)
            out["placed"].append(name)
    return out


def _references(ref: Path) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(ref.glob("*/calibration.json")) if ref.is_dir() else []:
        cal = config.read_json(p)
        if isinstance(cal, dict) and all(isinstance(cal.get(k), (int, float)) for k in ("default_dose", "good", "turns")):
            out.append((p.parent.name, cal))
    return out


def _place(name: str, pkg: Path, theirs: str, roots: tuple[Path, ...], ckpt: str, package: str, origin: dict) -> None:
    """The package's concept ``pkg`` (``_inside`` ``roots``) into ``name``'s folder for ``ckpt``: its texts, and the
    vector, profile and other files of ``pkg/theirs``, ``origin`` (fetch's) as their ``source``. The files there before
    that metalscope places (``_files``: an earlier placement of this package, or stray files in a folder that held
    nothing of the user's: no texts, vector or profile) are replaced; everything else (the served copies' folders, the
    user's calibrations, any other file) is carried across. Written into a dot folder beside it (store.PART), the old
    folder renamed aside and this one into place: a placement cut short leaves the folder as it was, or none with
    what the user had in the dot folder, which the next pull gives back (``_unpart``). Run under the checkpoint's
    lock (pull)."""
    from .concepts import write_profile
    src, dest = pkg / theirs, models.concept_dir(ckpt, name)
    part, old = _aside(dest)
    _unpart(dest)
    part.mkdir(parents=True)
    try:
        for f in [*(f for f in _files(pkg, roots) if f.name in TEXTS),
                  *(f for f in _files(src, roots) if f.name != "profile.json" and f.name not in TEXTS)]:
            shutil.copyfile(f, part / f.name)
        meta = config.read_json(part / "vector.json", {})
        config.atomic_write(part / "vector.json", json.dumps(dict(meta, package=package, source=origin), indent=1))
        prof = json.loads((src / "profile.json").read_text())
        on = dict(prof.get("measured_on") or {}, checkpoint=prof.get("model"))
        write_profile(dict(prof, concept=name, model=ckpt, measured_on=on, package=package, source=origin), part)
        placed = set(_files(dest))
        for p in [p for p in dest.iterdir() if p not in placed] if dest.is_dir() else []:
            os.rename(p, part / p.name)
        if dest.is_dir():
            os.rename(dest, old)
        os.rename(part, dest)
    except BaseException:
        _unpart(dest)
        raise
    shutil.rmtree(old, ignore_errors=True)


def _aside(dest: Path) -> tuple[Path, Path]:
    """``_place``'s dot folders beside ``dest``: the one it builds, the one the old folder is renamed to."""
    return dest.parent / f".{dest.name}{store.PART}", dest.parent / f".{dest.name}.old"


def _unpart(dest: Path) -> None:
    """``_place``'s dot folders beside ``dest`` removed: the old folder renamed back when the new one is not in place
    (it is the concept until then), and what was carried across of the user's (anything ``_files`` does not name)
    given back to ``dest`` first."""
    part, old = _aside(dest)
    if old.is_dir() and not os.path.lexists(dest):
        os.rename(old, dest)
    if part.is_dir():
        placed = {p.name for p in _files(part)}
        for p in list(part.iterdir()):
            if p.name not in placed and not os.path.lexists(dest / p.name):
                dest.mkdir(parents=True, exist_ok=True)
                os.rename(p, dest / p.name)
        shutil.rmtree(part)
    shutil.rmtree(old, ignore_errors=True)


def source_text(origin: Optional[dict]) -> str:
    """Where a pulled vector came from, in a few words: 'owner/name at revision v1 (commit abc)', or the folder."""
    if not isinstance(origin, dict):
        return ""
    if origin.get("repo"):
        at = f" at revision {origin['revision']}" if origin.get("revision") else ""
        return f"{origin['repo']}{at} (commit {origin.get('commit')})"
    return str(origin.get("folder") or "")


def next_steps(out: dict, served: str) -> list[str]:
    """What to do after a pull, in lines: norms when the served copy has none, how each concept did on the packer's
    copies (the one with the most good replies at its default: a reference, not a dose to use), then the calibration
    that gives the user's own dose (for a concept with texts on the model: calibrating needs them)."""
    from .concepts import texts_dir
    name = config.short_name(served)
    lines = []
    if not models.norms_path(name).exists():
        lines.append(f"  {name} has no dose units yet: `metalscope on`, one real chat with it, then "
                     f"`metalscope vector norms {name}`")
    for c, refs in out["references"].items():
        if out["layers"].get(c) is None:
            lines.append(f"  {c} has no steer layer (its sweep found no dose that moved the choice)")
            continue
        top = max(refs, key=lambda x: x[1].get("good") or 0, default=None)
        lines.append(f"  {c}: on the packer's copy {top[0]}, {top[1].get('good')} of {top[1].get('turns', 18)} replies good "
                     f"at dose {top[1]['default_dose']:g}" if top else f"  {c}: not calibrated on the packer's copies")
    can = [c for c in out["references"] if texts_dir(out["checkpoint"], c) is not None]
    if can:
        lines.append(f"  calibrate your own: metalscope concept calibrate {can[0]} {name} (then /steer {can[0]} uses your dose)")
    first = next(iter(out["references"]), "NAME")
    lines.append(f"  to try a dose by hand first: /steer {first} DOSE")
    return lines

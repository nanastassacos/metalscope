"""Steering vectors: stored per checkpoint, and made.

A vector is unit-norm per layer, stored in the concept's folder for the served model's
checkpoint as ``~/.metalscope/models/<checkpoint>/concepts/<name>/vector.safetensors`` (keys
``v.<layer>``; models.py) with ``vector.json`` beside it (layers, width, natural norm,
provenance), so every served copy of a checkpoint steers with the same ones. A name is a
folder name and a steering header token (``check_name``). Made by ``concept learn``
(concepts/run.py, the diff of means) or ``from_lens`` (Jacobian-lens concept directions for
some words, the server's /api/steer/words: a concept folder with no texts). Steering with
them is steer.py's.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from . import config, models

NAME_RE = models.NAME_RE        # a file name and a header token; no leading dot (temp files)
NAME_MAX = 96                   # the longest job id a name makes adds 31 (learn/state.py new_job_id); job ids are NAME_RE's


def check_name(name: str) -> str:
    """A vector name: letters, digits, _ . -, not starting with a dot, at most NAME_MAX characters. Called
    before the expensive work (building a vector) and again when saving."""
    if not NAME_RE.fullmatch(name or "") or len(name) > NAME_MAX:
        raise ValueError(f"bad vector name {name!r}: letters, digits, _ . - only, not starting with a dot, "
                         f"at most {NAME_MAX} characters")
    return name


# ---------------------------------------------------------------- on disk

@dataclass
class Vector:
    name: str
    vectors: dict[int, np.ndarray]           # layer -> [width] float32, unit norm
    meta: dict = field(default_factory=dict)  # width, hc_count, natural_norm, source, made, ...

    @property
    def layers(self) -> list[int]:
        return sorted(self.vectors)

    @property
    def width(self) -> int:
        return int(next(iter(self.vectors.values())).shape[0])

    @property
    def hc_count(self) -> int:
        return int(self.meta.get("hc_count", 1))

    def stream_norms(self, layer: int) -> np.ndarray:
        """Norm of each hyper-connection stream block of the (unit) vector."""
        return np.linalg.norm(self.vectors[layer].reshape(self.hc_count, -1), axis=-1)


def save_vector(model_name: str, vec: Vector) -> Path:
    """Store a vector under its name (letters, digits, _ . - only: the name is
    a file name and arrives in the steering header). The sidecar first, then the
    tensor file, each written to a temp name and renamed: a steered request never
    loads a partial one, and a vector is never found (its tensor file) without its
    meta (a word's ``token_ids``: steer.check_words)."""
    check_name(vec.name)
    path = models.vector_path(model_name, vec.name)
    d = path.parent
    d.mkdir(parents=True, exist_ok=True)
    # A unique temp name ending in .safetensors (mx.save_safetensors insists on the extension).
    meta = dict(vec.meta, name=vec.name, layers=vec.layers, width=vec.width, unit=True,
                made=vec.meta.get("made") or time.strftime("%Y-%m-%dT%H:%M:%S"))
    config.atomic_write(path.with_suffix(".json"), json.dumps(meta, indent=1))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".vector.", suffix=".safetensors")
    os.close(fd)
    try:
        mx.save_safetensors(tmp, {f"v.{l}": mx.array(v.astype(np.float32)) for l, v in vec.vectors.items()})
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return d


def find_vector(model_name: str, name: str) -> Optional[Path]:
    if not NAME_RE.fullmatch(name or ""):
        return None
    p = models.vector_path(model_name, name)
    return p if p.exists() else None


def load_vector(model_name: str, name: str) -> Vector:
    path = find_vector(model_name, name)
    if path is None:
        raise ValueError(f"no steering vector {name!r} for {model_name} at {models.vector_path(model_name, name)}")
    tensors = mx.load(str(path))
    vecs = {}
    for k, v in tensors.items():
        if k.startswith("v."):
            a = np.array(v.astype(mx.float32))
            vecs[int(k.split(".", 1)[1])] = a / max(float(np.linalg.norm(a)), 1e-12)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return Vector(name, vecs, meta)


def list_vectors(model_name: str) -> list[dict]:
    out = []
    for p in models.vector_paths(model_name):
        meta = config.read_json(p.with_suffix(".json"))
        if meta is not None:
            out.append(dict(name=p.parent.name, layers=meta.get("layers"), width=meta.get("width"),
                            source=meta.get("source"), made=meta.get("made")))
    return out


def is_words(meta) -> bool:
    """Whether a vector's meta (its vector.json) is a word vector's (``from_lens``: its ``token_ids``), not a
    concept's."""
    return isinstance(meta, dict) and "token_ids" in meta


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def dprime(a: np.ndarray, b: np.ndarray) -> float:
    """Separation of two sets of scores: the gap of their means over their
    pooled standard deviation."""
    pooled = math.sqrt((float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2) + 1e-6
    return float((a.mean() - b.mean()) / pooled)


# ---------------------------------------------------------------- making vectors

def from_lens(unembed, tokenizer, model_name: str, name: str, words: list[str],
              layers: Optional[list[int]] = None, lens=None, ids: Optional[list[int]] = None,
              columns: Optional[Callable[[int, int], slice]] = None) -> Vector:
    """A steering vector from Jacobian-lens concept directions: the (unit) sum
    over ``words`` of ``J_l^T (gamma * u_t)`` at each layer (jlens.concept_vectors).
    ``columns``: (layer, the direction's width) -> the part of it a steer
    writes, cut before it is made unit (a tracer's ``lens_columns``; Inkling 2:
    the running sum's, H wide at every visit, so hc_count 1; ValueError for a
    lens not as wide as the model's state); None: all of it.
    ``unembed(ids)``: the tokens' output-head rows and the final norm's gain or
    None (a tracer's ``unembed``: the MLX model's, or the GGUF's on ds4).
    Its natural norm per layer is kept in meta. ``lens``: an already loaded
    Lens (the server passes its own; loading one is gigabytes). ``ids``: the
    words' tokens themselves (the dock's eyepiece: a word as shown, with or
    without a leading space), each checked against its word; else each word's
    first token after a space (``cut``: the words longer than that)."""
    from . import jlens
    check_name(name)
    lens_dir = jlens.find_for(model_name)
    if lens is None:
        if lens_dir is None:
            raise ValueError(f"no Jacobian lens for {model_name}")
        lens = jlens.load(lens_dir)
    if ids is None:
        pairs = jlens.first_token_ids(tokenizer, words)
        cut = [w for w, _ in pairs if len(tokenizer.encode(" " + w, add_special_tokens=False)) > 1]   # first piece only
    else:
        pairs, cut = [(w.strip(), i) for w, i in zip(words, ids, strict=True)], []
        wrong = [f"{i} is not {w!r}" for w, i in pairs if not w or tokenizer.decode([i]).strip() != w]
        if wrong:
            raise ValueError(f"token {', '.join(wrong)}")
    if not pairs:
        raise ValueError("no words")
    ids = [i for _, i in pairs]
    rows, gamma = unembed(ids)
    layers = layers or lens.layers
    vecs, natural = {}, {}
    for l in layers:
        if l not in lens.jacobians:
            raise ValueError(f"the lens has no layer {l}")
        v = np.array(jlens.concept_vectors(lens, l, rows, gamma)).sum(0)   # [width]
        if columns is not None:
            v = v[columns(l, v.shape[0])]
        natural[l] = float(np.linalg.norm(v))
        vecs[l] = unit(v)
    J = lens.jacobians[layers[0]]                                          # [d_model, W]: W / d_model streams
    hc = 1 if columns is not None else int(J.shape[1]) // int(J.shape[0])
    meta = dict(source=f"J-lens concept direction for {', '.join(w for w, _ in pairs)}",
                words=[w for w, _ in pairs], token_ids=ids, cut=cut, hc_count=hc,
                natural_norm=natural, lens=str(lens_dir.name) if lens_dir else None)
    return Vector(name, vecs, meta)


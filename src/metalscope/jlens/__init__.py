"""Jacobian lens (Anthropic, "Verbalizable Representations Form a Global
Workspace in Language Models", 2026; reference code: anthropics/jacobian-lens).

    lens_l(h) = unembed(J_l @ h),   J_l = E[d h_final / d h_l]

``h_l`` is the output of decoder block ``l`` and ``unembed`` is the model's
final norm + LM head, so the readout is our adapter's readout applied to
``J_l @ h``. The plain logit lens is the special case ``J_l = I``.

Lenses live in ``~/.metalscope/models/<checkpoint>/lenses/<name>/``: a lens is the checkpoint's (models.py) whose
folder holds it. Its lens.json's ``checkpoint`` records the one it was made for, written by whatever writes the lens
(``save``, ``/jlens learn``, scripts/convert_lens.py). The checkpoint's checkpoint.json names the lens it reads
(``use`` switches, for every served name of it; ``use("logit")`` turns it off, back to the logit lens). With no
choice recorded, a checkpoint that has exactly one lens reads it.

    lens.safetensors   "J.<layer>" -> [d_model, d_model] float16 ([hidden, hc * hidden] for hyper-connection streams,
                       [hidden, n * hidden] for Inkling 2's state at a visit: fit.py)
    lens.json          {source, hf_model, n_prompts, d_model, layers, checkpoint[, learned_for, heldout, corpus]}

``scripts/convert_lens.py`` converts the reference ``.pt`` format (needs torch,
so it runs outside oMLX).

A word's J-lens direction at layer ``l`` is ``v_t = J_l^T (gamma * u_t)`` (``u_t`` its unembedding row, ``gamma`` the
final norm weight), so ``v_t . h`` is token ``t``'s lens logit up to the norm's per-position scale (vectors.from_lens).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .. import config, models

SKIP_FIRST_N_POSITIONS = 16  # attention-sink positions are atypical: the fit (reference default) and its windows skip them
LOGIT = "logit"         # the lens name /jlens use takes for the plain logit lens (J = I): the model reads no lens directory
JLENS = "jlens"         # and for "the model's J-lens", whichever that is (pick_jlens)


@dataclass
class Lens:
    jacobians: dict[int, mx.array]          # layer -> [d, d]
    meta: dict = field(default_factory=dict)

    @property
    def layers(self) -> list[int]:
        return sorted(self.jacobians)

    @property
    def d_model(self) -> int:
        return next(iter(self.jacobians.values())).shape[0]


def lenses_dir(model: str) -> Path:
    """The lenses of the model's checkpoint."""
    return models.model_dir(model) / "lenses"


def load(path: str | Path) -> Lens:
    path = Path(path)
    tensors = mx.load(str(path / "lens.safetensors"))
    jac = {int(k.split(".", 1)[1]): v for k, v in tensors.items() if k.startswith("J.")}
    # Materialise now: MLX streams are per thread, and the lens may be loaded
    # on a web thread but used on the scheduler thread.
    mx.eval(list(jac.values()))
    meta = json.loads((path / "lens.json").read_text()) if (path / "lens.json").exists() else {}
    return Lens(jac, meta)


def save(lens: Lens, path: str | Path) -> Path:
    """Write a lens directory; its meta must name the checkpoint the lens is for (``checkpoint``)."""
    if not lens.meta.get("checkpoint"):
        raise ValueError("a lens names its checkpoint: lens.meta['checkpoint'] (metalscope model list)")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path / "lens.safetensors"),
                        {f"J.{l}": J.astype(mx.float16) for l, J in lens.jacobians.items()})
    meta = dict(lens.meta, d_model=lens.d_model, layers=lens.layers)
    (path / "lens.json").write_text(json.dumps(meta, indent=1))
    return path


def lens_meta(d: Path) -> dict:
    """A lens directory's lens.json; empty when it is missing or unreadable."""
    return config.read_json(d / "lens.json", {})


def lens_dirs(model: Optional[str] = None) -> list[Path]:
    """The lens directories (ones with lens.safetensors) of the model's checkpoint, by name; with no model, every
    checkpoint's."""
    found = models.models_dir().glob("*/lenses/*") if model is None else lenses_dir(model).glob("*")
    return sorted(d for d in found if (d / "lens.safetensors").exists())


def find_for(model_name: str) -> Optional[Path]:
    """The lens directory a served model reads: its checkpoint's choice (checkpoint.json ``lens``; ``LOGIT``: none);
    with no choice recorded, the checkpoint's lens when it has exactly one; else None (the logit lens)."""
    rec = models.record(model_name)
    choice = rec.get("lens")
    if choice == LOGIT:
        return None
    if choice:
        d = lenses_dir(model_name) / choice
        return d if (d / "lens.safetensors").exists() else None
    mine = lens_dirs(model_name)
    return mine[0] if len(mine) == 1 else None


def candidates(model_name: str) -> list[dict]:
    """Every lens directory that is or could be the served model's, for /jlens: the checkpoint's lenses (ones
    ``/jlens learn`` fitted for it, ones installed for it), the one in use first. Each with its held-out hit@10 when lens.json has one (a learned lens: on its corpus's held-out
    windows, against the logit lens and the lens in use when it was fitted). Ranked by hit@10, highest first; lenses
    without a score after, by name. When there is any, the logit lens is one more (name ``LOGIT``, ``logit`` True; in
    use when no lens is), scored by the logit lens's hit@10 on the held-out windows of the best-scored lens that has
    one (``scored_on``)."""
    current = find_for(model_name)
    out = []
    for d in lens_dirs(model_name):
        meta = lens_meta(d)
        why = "in use" if d == current else "learned for it" if meta.get("learned_for") else "its lens"
        h = meta.get("heldout") or {}
        out.append(dict(name=d.name, path=str(d), in_use=d == current, why=why, source=meta.get("source"),
                        n_prompts=meta.get("n_prompts"), learned_for=meta.get("learned_for"), fitted=meta.get("fitted"),
                        hit10=h.get("hit10"), logit_hit10=h.get("logit_hit10"), compared=h.get("compared"), logit=False))
    rank = lambda c: (c["hit10"] is None, -(c["hit10"] or 0), c["name"])
    if out:
        on = next((c for c in sorted(out, key=rank) if c["logit_hit10"] is not None), None)
        out.append(dict(name=LOGIT, path=None, in_use=current is None, why="in use" if current is None else "J = I",
                        source="the model's final norm and LM head on each layer's residual", n_prompts=None,
                        learned_for=None, fitted=None, hit10=on["logit_hit10"] if on else None, logit_hit10=None,
                        compared=None, logit=True, scored_on=on["name"] if on else None))
    return sorted(out, key=rank)


def in_use(lens: str, model: str) -> bool:
    """Whether the model's checkpoint chose lens directory ``lens``: reads it (checkpoint.json ``lens``) or goes back
    to it on ``/jlens on`` (``jlens``)."""
    rec = models.record(model)
    return lens in (rec.get("lens"), rec.get("jlens"))


def width_error(lens: str, d_model, model: str) -> Optional[str]:
    """Why lens ``lens``, ``d_model`` wide, cannot be the model's: its checkpoint's recorded shape is another width
    (None when either is not known)."""
    rec = models.record(model)
    width = (rec["shape"] or {}).get("d_model")
    if width and d_model and int(d_model) != int(width):
        return f"lens {lens} is {d_model} wide, {rec['name']} is {width}"
    return None


def pick_jlens(model_name: str) -> Optional[str]:
    """The J-lens ``use(JLENS)`` means for a served model: the one in use; else the one its checkpoint read last
    (checkpoint.json ``jlens``, kept when ``use(LOGIT)`` turned it off); else the best of ``candidates`` (held-out
    hit@10, then by name). None when there is no J-lens for it."""
    current = find_for(model_name)
    if current is not None:
        return current.name
    last = models.record(model_name).get("jlens")
    if last and (lenses_dir(model_name) / last / "lens.safetensors").exists():
        return last
    cs = [c for c in candidates(model_name) if not c["logit"]]
    return cs[0]["name"] if cs else None


def use(lens: str, model_name: str) -> dict:
    """Make lens directory ``lens`` the J-lens of the served model's checkpoint, for every served name of it: its
    checkpoint.json records ``lens`` (and ``jlens``, what ``JLENS`` goes back to). The server reads the lens when it loads the model, or at once
    through ``ModelTracer.refresh_lens`` (the /jlens use route). ``lens``
    ``LOGIT``: the checkpoint reads through the logit lens (``jlens`` kept). ``lens`` ``JLENS``: the model's J-lens
    (``pick_jlens``). ValueError when ``lens`` is not a lens directory of this checkpoint or is not as wide as it,
    or for ``JLENS`` when the model has none."""
    import re
    name = config.short_name(model_name)
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", lens or ""):
        raise ValueError(f"not a lens name: {lens!r}")
    if lens == JLENS:
        lens = pick_jlens(name)
        if lens is None:
            raise ValueError(f"no J-lens for {name} under {lenses_dir(name)}: /jlens learn fits one, or a published one "
                             f"converted for its checkpoint (scripts/convert_lens.py --checkpoint) goes there")
    rec = models.record(name)
    ckpt = rec["name"]
    target = None if lens == LOGIT else lenses_dir(name) / lens
    if target is not None:
        if not (target / "lens.safetensors").exists():
            have = ", ".join(c["name"] for c in candidates(name)) or "none"
            raise ValueError(f"no lens named {lens} under {lenses_dir(name)} (lenses for {name}: {have})")
        why = width_error(lens, lens_meta(target).get("d_model"), name)
        if why:
            raise ValueError(why)
    was = find_for(name)
    changes = dict(lens=LOGIT) if target is None else dict(lens=target.name, jlens=target.name)
    if target is None and was is not None:
        changes["jlens"] = was.name
    models.update(name, **changes)
    edited = [str(models.models_dir() / ckpt / models.RECORD)]
    return dict(model=name, checkpoint=ckpt, served=models.served_names(name), lens=LOGIT if target is None else target.name,
                path=None if target is None else str(target), was=was.name if was else LOGIT, edited=edited)


def workspace_band(n_layers: int) -> tuple[int, int]:
    """The first and last layer of the paper's workspace band (38-92% of depth): 18-44 of 48, where removing a word
    moved Flash Next's one-word answer to another 5 of 5 (24-31: 1 of 5)."""
    return int(0.38 * n_layers), min(n_layers - 2, int(0.92 * n_layers))


def concept_vectors(lens: Lens, layer: int, unembed_rows: mx.array,
                    norm_weight: Optional[mx.array]) -> mx.array:
    """[n_words, d] word directions v_t = J_l^T (gamma * u_t) for one layer."""
    U = unembed_rows.astype(mx.float32)                     # [n, d]
    if norm_weight is not None:
        U = U * norm_weight.astype(mx.float32)[None, :]
    return (U @ lens.jacobians[layer].astype(mx.float32))   # (J^T u)^T = u^T J


def first_token_ids(tokenizer, words: list[str]) -> list[tuple[str, int]]:
    """Map words to single tokens (with a leading space, as they
    usually occur mid-sentence); words that split into several tokens use the
    first piece, which is what the single-token lens can see."""
    out = []
    for w in words:
        ids = tokenizer.encode(" " + w.strip(), add_special_tokens=False)
        if ids:
            out.append((w.strip(), int(ids[0])))
    return out


def unembed_rows(model, token_ids: list[int]) -> mx.array:
    """[n, d] unembedding rows (dequantised) for the given tokens."""
    from ..adapters.generic import lm_head, text_model
    head = lm_head(model)
    if head is None:  # tied: the embedding table is the unembedding
        return text_model(model)["embed_tokens"](mx.array(token_ids)).astype(mx.float32)
    return head_rows(head, token_ids)


def head_rows(head, token_ids: list[int]) -> mx.array:
    """[n, d] float32 rows of an output head (a Linear, or a QuantizedLinear: just these rows dequantised)."""
    ids = mx.array(token_ids)
    if "scales" in head:  # QuantizedLinear: dequantise just these rows
        w = mx.dequantize(head["weight"][ids], head["scales"][ids], head.get("biases", None)[ids]
                          if "biases" in head else None, group_size=head.group_size, bits=head.bits)
        return w.astype(mx.float32)
    return head["weight"][ids].astype(mx.float32)


def norm_gain(model) -> Optional[mx.array]:
    """The final norm's effective per-dimension gain, measured rather than
    assumed (some models scale by w, others by 1 + w): norm(ones) = gain."""
    from ..adapters.generic import text_model
    inner = text_model(model)
    norm = next((inner[n] for n in ("norm", "final_layernorm", "ln_f") if n in inner), None)
    if norm is None:
        return None
    d = inner["embed_tokens"].weight.shape[-1] if "embed_tokens" in inner else None
    if d is None:
        return None
    if "scales" in inner["embed_tokens"]:  # quantised embedding: packed weight width
        d = inner["embed_tokens"].scales.shape[-1] * inner["embed_tokens"].group_size
    return norm(mx.ones((1, d), dtype=mx.float32))[0]


@dataclass
class LensSetup:
    """The J-lens a tracer reads: the lens and its directory's name."""
    lens: Lens
    name: str = ""


def setup_for(model_name: str) -> Optional[LensSetup]:
    """The J-lens the MLX tracer reads (``find_for``), loaded; None for the logit lens. ValueError when it is not as
    wide as the model's checkpoint (``width_error``: lens.json's width before loading, the weights' after)."""
    path = find_for(model_name)
    if path is None:
        return None
    why = width_error(path.name, lens_meta(path).get("d_model"), model_name)
    lens = None if why else load(path)
    why = why or width_error(path.name, lens.d_model, model_name)
    if why:
        raise ValueError(why)
    return LensSetup(lens, name=path.name)


def status(model_name: str) -> dict:
    """What Pi's /jlens shows for a served model, from the files alone (no lens is loaded): whether a lens is
    installed and where; lens.json's source, hf_model, n_prompts and layer count; ``candidates``: every lens that is
    or could be the model's, ranked (``candidates``); ``checkpoint``: the model it is a copy of, ``served``: the
    served names that read the same lens (models.py)."""
    path = find_for(model_name)
    out = dict(model=config.short_name(model_name), checkpoint=models.checkpoint_of(model_name),
               served=models.served_names(model_name), installed=path is not None, path=str(path) if path else None,
               lenses_dir=str(lenses_dir(model_name)), candidates=candidates(model_name))
    if path is None:
        return out
    meta = lens_meta(path)
    out.update({k: meta.get(k) for k in ("source", "hf_model", "n_prompts")}, layers=len(meta.get("layers") or []) or None)
    return out

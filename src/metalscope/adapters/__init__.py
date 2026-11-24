"""Architecture adapters: where the capture points live in a given model.

An adapter knows how to find decoder layers, the MoE router and the LM head,
and how to turn a decoder layer's raw output into the residual the next block
reads. Everything model-specific lives here; capture.py stays generic.

An adapter is a module. Required:

    NAME                     the adapter's name (a trace's meta, a lens fit's)
    matches(model)           True for a model it reads
    install(session, model)  its capture hooks on ``model``

Optional, each but MODEL_TYPES read through the function of the same name below (its fallback when the adapter has none):

    MODEL_TYPES              the config.json ``model_type``s it serves, for what is read from a checkpoint's config
    layer_count(model)       the layers a trace records
    steer_block(model)       how many layers a concept's push spans
    steer_layers(model)      the selection sweep's candidate steer layers
    ANSWER_MARKS             what opens a reply's answer and the token that ends it (answer_marks)
    visit_counter(model)     a steer hook's layer call -> the layer a plan names it by
    lens_columns(model)      (layer, lens width) -> the lens columns a steer writes
    edit_band(model)         the layers the dock's word edit removes over
    unembed(model, ids)      the tokens' output-head rows and the final norm's gain
    differentiable(model)    the model set up for a J-lens fit (jlens/fit.py)
    fit_setup()              what a lens fit switches off once the checkpoint is loaded (by MODEL_TYPES)
    lens_sources(cfg)        how many hidden-wide sources a J-lens reads at each layer (by MODEL_TYPES)
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.nn as nn

from ..capture import CaptureSession
from . import generic, inkling2, qwen4_exp

_ADAPTERS = [qwen4_exp, inkling2, generic]


def find_adapter(model: nn.Module):
    for adapter in _ADAPTERS:
        if adapter.matches(model):
            return adapter
    raise ValueError(f"no metalscope adapter for {type(model).__name__}")


def install(session: CaptureSession, model: nn.Module) -> str:
    """Install the matching adapter's hooks; returns the adapter name."""
    adapter = find_adapter(model)
    adapter.install(session, model)
    return adapter.NAME


def layer_count(model: nn.Module) -> int:
    """The layers a trace records on ``model``: its decoder layers, or what its adapter counts instead (Inkling 2: its
    layer visits)."""
    count = getattr(find_adapter(model), "layer_count", None)
    return count(model) if count else len(generic.text_model(model).layers)


def steer_block(model: nn.Module) -> Optional[int]:
    """How many layers a concept's push at a layer spans, from it on (Inkling 2: its block's visits), or None: the
    layer alone. A profile saves it in the vector's meta (``block``; steer.push_layers)."""
    block = getattr(find_adapter(model), "steer_block", None)
    return block(model) if block else None


def steer_layers(model: nn.Module) -> Optional[list[int]]:
    """The selection sweep's candidate steer layers, when the adapter names them (Inkling 2: the prelude's first visit
    and the first visit of the last loop pass); None (or empty) for the default (learn/gpu.py)."""
    layers = getattr(find_adapter(model), "steer_layers", None)
    return layers(model) if layers else None


def answer_marks(model: nn.Module) -> Optional[tuple[str, str]]:
    """Where the chat template has no switch for thinking: (the text after the prompt that opens the reply's answer,
    the token that ends it), so a prompt can ask for the answer straight away (Inkling 2: its channels); None."""
    return getattr(find_adapter(model), "ANSWER_MARKS", None)


def visit_counter(model: nn.Module) -> Callable[[int], int]:
    """For a steer session's layer hooks: a layer call (its index) -> the layer a plan names it by, the one a trace
    records it at: the adapter's (Inkling 2: the visit), else the index itself."""
    count = getattr(find_adapter(model), "visit_counter", None)
    return count(model) if count else (lambda i: i)


def lens_columns(model: nn.Module) -> Optional[Callable[[int, int], slice]]:
    """(layer, its lens Jacobian's width) -> the columns of that Jacobian a steer writes, when the adapter cuts them
    (Inkling 2: the running sum's in each visit's state; ValueError for a lens not as wide as the state); None for the
    rest (every column)."""
    cut = getattr(find_adapter(model), "lens_columns", None)
    return cut(model) if cut else None


def edit_band(model: nn.Module, n_layers: int) -> tuple[int, int]:
    """The layers the dock's word edit removes over: the adapter's (Inkling 2: the last block's visits), else the
    workspace band of ``n_layers``."""
    from ..jlens import workspace_band
    band = getattr(find_adapter(model), "edit_band", None)
    return band(model) if band else workspace_band(n_layers)


def unembed(model: nn.Module, ids: list[int]):
    """The tokens' output-head rows [n, d] and the final norm's gain (or None), for a word's lens direction: the
    adapter's (Inkling 2's head and norm), else jlens's."""
    from .. import jlens
    own = getattr(find_adapter(model), "unembed", None)
    return own(model, ids) if own else (jlens.unembed_rows(model, ids), jlens.norm_gain(model))


def differentiable(model: nn.Module, sources: list[int], hyper: bool):
    """The model set up for a J-lens fit, a context manager yielding the probes and ``forward``: the adapter's
    (Inkling 2: its own loop), else jlens.fit's over ``sources`` (``hyper``: the model's own output)."""
    from ..jlens import fit
    own = getattr(find_adapter(model), "differentiable", None)
    return own(model) if own else fit.differentiable(model, sources, hyper)


def _for_type(model_type: Optional[str], name: str):
    """The function ``name`` of the adapter that serves config.json's ``model_type`` (its ``MODEL_TYPES``), or None."""
    found = (getattr(a, name) for a in _ADAPTERS if model_type in getattr(a, "MODEL_TYPES", ()) and hasattr(a, name))
    return next(found, None)


def fit_setup(model_path: str) -> list[str]:
    """What a lens fit switches off in its own process once the checkpoint is loaded, by ``model_type``: an adapter's
    ``fit_setup`` (qwen4_exp: oMLX's fused kernels, which have no gradients); the names switched, [] for the rest."""
    from pathlib import Path
    from .. import config
    cfg = config.read_json(Path(model_path).expanduser() / "config.json", {})
    setup = _for_type(cfg.get("model_type"), "fit_setup")
    return setup() if setup else []


def lens_sources(cfg: dict) -> Optional[list[int]]:
    """How many hidden-wide sources a J-lens reads at each recorded layer, from config.json ``cfg``, when the adapter
    of its ``model_type`` counts them (Inkling 2: each visit's state); None for the rest."""
    count = _for_type(cfg.get("model_type"), "lens_sources")
    return count(cfg) if count else None

"""Adapter for conventional mlx-lm / mlx-vlm decoders.

Assumes ``layers[i](h, ...)`` returns the post-layer residual; a layer's MoE
block (``mlp``, ``block_sparse_moe``, ``ffn``, ...) and its routing rule are
found by routing.py.
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from .. import routing
from ..capture import CaptureSession

NAME = "generic"


def text_model(model: nn.Module) -> nn.Module:
    """The module that owns ``layers`` (and usually ``lm_head`` one level up)."""
    # "_vlm_model..." reaches through oMLX's VLMModelAdapter.
    for path in ("language_model.model", "_vlm_model.language_model.model", "model", ""):
        node = model
        for part in filter(None, path.split(".")):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and hasattr(node, "layers"):
            return node
    raise ValueError("could not locate decoder layers")


def d_model(inner: nn.Module) -> int:
    """One residual stream's width: the config's hidden size, else what the embedding returns."""
    d = getattr(getattr(inner, "args", None), "hidden_size", None)
    return int(d or inner["embed_tokens"](mx.array([[0]])).shape[-1])


def lm_head(model: nn.Module):
    vlm = getattr(model, "_vlm_model", None)
    for owner in (getattr(model, "language_model", None),
                  getattr(vlm, "language_model", None), model):
        if owner is not None and "lm_head" in owner:
            return owner["lm_head"]
    return None


def vocab_size(model: nn.Module) -> int:
    return text_model(model)["embed_tokens"].weight.shape[0]


def output_hook(session: CaptureSession, vocab: int):
    """Post-hook on the top-level model: record final logits.

    Hooking the model output (not lm_head) covers tied embeddings, and skips
    calls that return hidden states instead of logits (oMLX prefill runs with
    the LM head skipped).
    """
    def on_out(_m, args, kwargs, out):
        if not session.active:
            return
        ids = args[0] if args else kwargs.get("input_ids", kwargs.get("inputs"))
        if isinstance(ids, mx.array) and ids.ndim == 2:
            session.emit("input_ids", -1, session.select(ids).astype(mx.int32))
        logits = getattr(out, "logits", out)
        if isinstance(logits, mx.array) and logits.shape[-1] == vocab:
            session.record_logits(session.select(logits))
    return on_out


def router_hook(session: CaptureSession, layer_idx: int | Callable[[], int]):
    """Post-hook for an MoE block: recompute router logits from its input, recorded at ``layer_idx`` (an index, or a
    call that gives the one under way: Inkling 2's layer visits).

    Fused router kernels may read ``gate.weight`` without calling the gate
    module, so we apply the gate to the block's real input ourselves, as the
    model's rule computes its logits (routing.py). Rules that route some
    layers by token id (DeepSeek V4's hash layers) also get the token ids the
    block was called with.
    """
    rule = session.routing
    tokens_too = rule.needs_tokens

    def on_moe(module, args, kwargs, _out):
        if session.active:
            x = session.select(args[0] if args else kwargs["x"])
            tokens = None
            if tokens_too:
                t = args[1] if len(args) > 1 else kwargs.get("input_ids")
                tokens = session.select(t) if isinstance(t, mx.array) and t.ndim == 2 else None
            session.record_router(layer_idx() if callable(layer_idx) else layer_idx, rule.logits(module, x), tokens)
    return on_moe


def input_router_hook(session: CaptureSession, layer_idx: int, block: nn.Module):
    """Post-hook on the module whose output (or its first, for a tuple) is ``block``'s input: router logits from it."""
    rule = session.routing

    def on_input(_m, _a, _k, out):
        if session.active:
            x = out[0] if isinstance(out, tuple) else out
            session.record_router(layer_idx, rule.logits(block, session.select(x)))
    return on_input


def install_router(session: CaptureSession, inner: nn.Module, mtp_input: Optional[str] = None) -> None:
    """Find the model's MoE blocks and routing rule (``session.routing``) and
    hook every block; nothing for a dense model.

    With MTP decode on, Qwen3.5 / 3.6 MoE verify windows run through a
    speculative verifier (mlx-vlm's ``Qwen3_5BatchInvariantForward._feed_forward``,
    Flash Next's ``_VERIFIER._feed_forward``), which recomputes the routing
    without calling the block. ``mtp_input`` names the layer's module whose
    output is the MoE input on both paths (the post-attention norm; Flash
    Next's mlp_hyper_connection): for those models it is hooked instead of the block."""
    session.routing = rule = routing.model_rule(enumerate(inner.layers))
    if rule is None:
        return
    for i, block in rule.blocks.items():
        layer = inner.layers[i]
        if mtp_input is not None and rule.name == "qwen3_5_moe" and mtp_input in layer:
            session.hook(layer[mtp_input], input_router_hook(session, i, block))
        else:
            session.hook(block, router_hook(session, i))


def matches(model: nn.Module) -> bool:
    try:
        text_model(model)
        return True
    except ValueError:
        return False


def readout(model: nn.Module):
    """residual -> logits via the model's final norm and LM head."""
    inner = text_model(model)
    norm = next((inner[n] for n in ("norm", "final_layernorm", "ln_f") if n in inner), None)
    head = lm_head(model)
    if head is None:
        head = inner["embed_tokens"].as_linear
    return lambda h: head(norm(h) if norm is not None else h)


def hyper_streams(inner: nn.Module):
    """DeepSeek V4-style hyper-connections: every layer carries ``hc`` parallel residual
    streams [B, T, hc, D], which the model's ``hc_head`` mixes down to the residual its
    final norm and LM head read. Returns (hc, streams [.., hc*D] -> residual [.., D]), or
    None for a conventional residual."""
    if "hc_head" not in inner:
        return None
    head, hc = inner["hc_head"], int(inner["hc_head"].hc_mult)
    return hc, lambda S: head(S.reshape(*S.shape[:-1], hc, S.shape[-1] // hc))


def install(session: CaptureSession, model: nn.Module) -> None:
    cfg = session.config
    inner = text_model(model)
    session.readout = readout(model)
    session.n_layers = len(inner.layers)
    hyper = hyper_streams(inner)
    if hyper is not None:
        # The recorded residual is hc_head(streams): what norm + head would read had the
        # model stopped after that layer (a logit lens with no approximation).
        session.hc_count, session.streams_to_resid = hyper
    for i, layer in enumerate(inner.layers):
        if session.wants_layer(i):
            def on_layer(_m, a, _k, out, i=i):
                if session.active:
                    o = out[0] if isinstance(out, tuple) else out
                    h = session.select(o)
                    if hyper is not None:
                        session.record_streams(i, h.reshape(*h.shape[:2], -1))
                    else:
                        session.record_residual(i, h)
                        # What the layer changed: its output less the residual it read (resid_write_norm, one stream).
                        x = a[0] if a else None
                        if cfg.residual_norm and isinstance(x, mx.array) and x.shape == o.shape:
                            session.record_write(i, h, session.select(x))
            session.hook(layer, on_layer)

    if cfg.router:
        from ..mtp import mtp_decode
        install_router(session, inner, "post_attention_layernorm" if mtp_decode(model) else None)
    if cfg.logits_topk:
        session.hook(model, output_hook(session, vocab_size(model)))

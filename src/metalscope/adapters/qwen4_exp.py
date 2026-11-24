"""Adapter for Qwen3.8 Flash Next (``qwen4_exp``) as vendored by oMLX >= 0.7.

Two things differ from a conventional decoder:

* Hyper-connections: the residual is ``hc_count`` parallel streams
  (``[B, T, hc_count * hidden]``), mixed down to ``hidden`` before each block
  and before the LM head by a ``Qwen4ExpGatedResidual``.
* Deferred writes: with fused writes enabled a decoder layer returns
  ``(residual, branch, gate)`` and its output is applied inside the next
  layer's norm kernel. We apply it ourselves (one cheap elementwise op) when
  tracing.

The residual we record at layer i is ``hyper_connection_mixer(stream_i)``:
exactly what the LM head would read if the model stopped after layer i, so a
logit lens is ``lm_head(recorded)`` with no approximation.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..capture import CaptureSession
from . import generic
from ..mtp import mtp_decode

NAME = "qwen4_exp"
MODEL_TYPES = ("qwen4_exp",)       # config.json model_type of the checkpoints this adapter serves


def fit_setup() -> list[str]:
    """oMLX's fused kernels off in this process, after the model is loaded and before a fit pass; the names switched.

    Each fused path is behind a module flag that oMLX itself sets when a kernel fails, so setting them takes the same
    plain MLX paths (0.8.0: the same flags; attn_fused's now also gates the q/k prologue of prefill rows on every GPU).
    Measured 2026-11-10 on 0.7.3, oQ4e-mtp: with these off a fit pass calls no Metal kernel (before: the
    hyper-connection, short-conv and expert-lookup kernels, "[Primitive::vjp] Not implemented for CustomKernel")."""
    import sys
    done = []
    for name, mod in list(sys.modules.items()):
        if "qwen4_exp" not in (getattr(mod, "__file__", None) or ""):
            continue
        for attr, val in list(vars(mod).items()):
            if attr.endswith("_DISABLED") and val is False:                 # hc_fused, attn_fused, qsa_fast, language
                setattr(mod, attr, True)
                done.append(f"{name}.{attr}")
            elif attr == "_DEPTHWISE_CONV_STATE" and isinstance(val, dict) and val.get("enabled"):
                val["enabled"] = False
                done.append(f"{name}.{attr}")
    gqmm = sys.modules.get("omlx.patches.m5_gather_qmm")                    # the quantised expert lookup's own kernel
    if gqmm is not None and hasattr(gqmm, "_nax_host"):
        gqmm._nax_host = False
        done.append("omlx.patches.m5_gather_qmm._nax_host")
    return done


def matches(model: nn.Module) -> bool:
    try:
        inner = generic.text_model(model)
    except ValueError:
        return False
    return "hyper_connection_mixer" in inner and type(inner).__name__.startswith("Qwen4Exp")


def _apply_write(out) -> mx.array:
    if not isinstance(out, tuple):
        return out
    hyper, branch, weights = out
    injection = branch[..., None, :] * weights[..., None]
    return hyper + injection.reshape(*hyper.shape)


def install(session: CaptureSession, model: nn.Module) -> None:
    cfg = session.config
    inner = generic.text_model(model)
    session.n_layers = len(inner.layers)
    mixer = inner["hyper_connection_mixer"]
    hc_count = inner.args.hc_count
    # The recorded residual is the mixer output, which the head reads directly.
    session.readout = generic.lm_head(model)
    session.streams_to_resid = mixer
    session.hc_count = hc_count

    # A layer reads its residual (``a[0]``, the embedding's streams at layer 0) with the previous layer's write still
    # pending (the ``write`` keyword, None once applied); its output streams less the streams it read are what it wrote
    # (``resid_write_norm``). When the forward hands a layer the previous one's output as it was (the same arrays), the
    # streams recorded for that output are its input, so a decode step builds each layer's streams once. The previous
    # recorded layer's output parts and streams are kept in ``session.carry``, which the step's end empties.
    for i, layer in enumerate(inner.layers):
        if session.wants_layer(i):
            def on_layer(_m, a, k, out, i=i):
                if not session.active:
                    return
                last = session.carry
                read = (a[0], *(k.get("write") or ()))
                done = last.get("parts", ())
                same = len(read) == len(done) and all(x is y for x, y in zip(read, done))
                before = last["streams"] if same else session.select(_apply_write(read if len(read) > 1 else a[0]))
                streams = session.select(_apply_write(out))
                session.record_streams(i, streams, before)
                last.update(parts=out if isinstance(out, tuple) else (out,), streams=streams)
            session.hook(layer, on_layer)

    if cfg.router:
        # oMLX's fused decode router reads gate.weight directly, so a hook on mlp.gate misses decode steps: the
        # logits are recomputed from the MoE block's actual input (its rule: routing.py). MTP verify windows call
        # the MoE through _VERIFIER._feed_forward, bypassing mlp.__call__; its input is mlp_hyper_connection's output.
        generic.install_router(session, inner, "mlp_hyper_connection" if mtp_decode(model) else None)

    if cfg.logits_topk:
        session.hook(model, generic.output_hook(session, generic.vocab_size(model)))

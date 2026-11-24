"""Recorded routing against the code oMLX serves: its own models (DeepSeek V4 / V4.1), mlx-vlm's Qwen3.5 / 3.6 MoE
and its vendored Flash Next, with oMLX's fused router patch on. Tiny random models; skipped unless omlx is importable
(oMLX's code and its MLX build ship only inside the app, so the dev venv skips this file: run it under the app's Python):

    PYTHONPATH=.venv/lib/python3.11/site-packages scripts/omlx-python -m pytest -q tests/test_routing_omlx.py
"""

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("omlx.patches")

from metalscope import routing  # noqa: E402
from metalscope.capture import CaptureConfig, CaptureSession  # noqa: E402
from metalscope.hooks import add_hook  # noqa: E402
from metalscope.adapters import install  # noqa: E402
from metalscope.adapters.generic import text_model  # noqa: E402
from tests.test_routing import by_id, randomize_gates  # noqa: E402


def compare(rule, recs, truth, rtol=1e-5, atol=1e-6):
    assert truth and sorted(truth) == sorted(rule.blocks)
    for layer, (t_ids, t_w) in truth.items():
        ids, w = recs[("router_ids", layer)], recs[("router_probs", layer)]
        a_ids, a_w = by_id(ids, w)
        b_ids, b_w = by_id(np.array(t_ids).reshape(ids.shape), np.array(t_w.astype(mx.float32)).reshape(w.shape))
        np.testing.assert_array_equal(a_ids, b_ids, err_msg=f"layer {layer}")
        np.testing.assert_allclose(a_w, b_w, rtol=rtol, atol=atol, err_msg=f"layer {layer}")


def capture(model, forward, truth_hooks):
    session = CaptureSession(CaptureConfig(layers=[], residual_norm=False, router=True))
    install(session, model)
    truth, removers = truth_hooks(model)
    forward()
    mx.eval(session.pending_arrays())
    session.end_step()
    recs = {(r.kind, r.layer): r.value for r in session.drain()}
    session.close()
    for r in removers:
        r()
    return session.routing, recs, dict(truth)


def gate_outputs(model):
    """Reference: the (inds, weights) each MoE gate returned."""
    truth, removers = {}, []
    for i, layer in enumerate(text_model(model).layers):
        block = routing.moe_block(layer)
        if block is not None:
            removers.append(add_hook(block["gate"], lambda _m, _a, _k, out, i=i: truth.__setitem__(i, out)))
    return truth, removers


TOKENS = (mx.array([[3, 17, 5, 42, 9]]), mx.array([[(7 * i + 3) % 64 for i in range(24)]]))


def tiny_v4(hc=2):
    """oMLX's DeepSeek V4 (registered as mlx_lm.models.deepseek_v4), 3 layers, the first a hash layer."""
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    apply_deepseek_v4_patch()
    from mlx_lm.models import deepseek_v4 as v4
    args = v4.ModelArgs(vocab_size=64, hidden_size=32, moe_intermediate_size=16, num_hidden_layers=3,
                        num_attention_heads=2, n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=3,
                        routed_scaling_factor=1.5, q_lora_rank=16, qk_rope_head_dim=8, head_dim=16,
                        compress_ratios=[0, 0, 0], hc_mult=hc, num_hash_layers=1, sliding_window=8, o_groups=2,
                        o_lora_rank=16, index_n_heads=2, index_head_dim=16, index_topk=4, num_nextn_predict_layers=0)
    mx.random.seed(0)
    model = v4.Model(args)
    randomize_gates(model)
    gate0 = model.model.layers[0].ffn.gate
    assert gate0.hash
    gate0.tid2eid = mx.random.randint(0, 8, gate0.tid2eid.shape).astype(mx.int32)
    head = model.model.hc_head        # random hc_head mixing, so streams differ from their mean
    head.fn, head.base = mx.random.normal(head.fn.shape), mx.random.normal(head.base.shape)
    mx.eval(model.parameters())
    return model


def test_deepseek_v4():
    """sqrt(softplus) + bias, and a hash layer routed by token id."""
    model = tiny_v4()
    for toks in TOKENS:
        rule, recs, truth = capture(model, lambda: model(toks), gate_outputs)
        assert rule.meta() == dict(rule="deepseek_v4", k=3, exact=True, shared=1, num_experts=8)
        compare(rule, recs, truth)


def test_deepseek_v4_streams_and_lens():
    """V4's layers carry hc residual streams [B, T, hc, D]: per-stream norms, the residual hc_head mixes them to,
    and a logit lens through norm + head, whose last layer is the model's own logits."""
    model, hc = tiny_v4(hc=4), 4
    session = CaptureSession(CaptureConfig(router=True, logits_topk=5, logit_lens_topk=5))
    assert install(session, model) == "generic"
    toks = TOKENS[1]
    logits = model(toks)
    mx.eval(session.pending_arrays())
    session.end_step()
    recs = {(r.kind, r.layer): np.array(r.value) for r in session.drain()}
    session.close()
    T = toks.shape[1]
    for layer in range(3):
        assert recs[("hc_stream_norm", layer)].shape[-2:] == (T, hc)
        assert recs[("resid_norm", layer)].shape[-1] == T
        assert recs[("lens_ids", layer)].shape[-2:] == (T, 5)
    top = np.argsort(-np.array(logits[0].astype(mx.float32)), axis=-1)[:, :5]
    np.testing.assert_array_equal(recs[("lens_ids", 2)].reshape(T, 5), top)
    np.testing.assert_array_equal(recs[("logit_ids", -1)].reshape(T, 5), top)


class SpaceTok:
    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]


def v4_reference_jacobian(model, ids, layer, valid):
    """Independent J_l for V4: layer ``layer``'s output replaced outright by a variable S [1, T, hc, D] (not a probe
    added to it), differentiated through to hc_head's output, rows by one-hot cotangents; mean over valid positions."""
    inner = text_model(model)
    seen = {}
    rm = add_hook(inner.layers[layer], lambda _m, _a, _k, out: seen.__setitem__("S", out))
    model(mx.array([ids]))
    rm()
    S0 = seen["S"]
    T, D = S0.shape[1], S0.shape[-1]
    box = {}
    rms = [add_hook(inner.layers[layer], lambda _m, _a, _k, out: box["S"]),
           add_hook(inner["hc_head"], lambda _m, _a, _k, out: box.__setitem__("H", out))]

    def tail(S):
        box["S"] = S
        model(mx.array([ids]))
        return [box["H"]]

    rows = []
    for i in range(D):
        cot = np.zeros((1, T, D), np.float32)
        cot[0, valid, i] = 1
        _, (g,) = mx.vjp(tail, [S0], [mx.array(cot)])
        rows.append(np.array(g.astype(mx.float32)).reshape(1, T, -1)[0, valid].mean(0))
    for r in rms:
        r()
    return np.stack(rows)                                  # [D, hc * D]


def test_deepseek_v4_jlens_fit_matches_an_independent_jacobian():
    """jfit on V4's streams: J_l is [D, hc*D], d hc_head(out) / d streams_l, the same as differentiating with the
    layer's output replaced; capture then reads the lens on the streams (norm + head of J @ S)."""
    from metalscope.jlens.fit import fit, no_index_gradients
    no_index_gradients()          # as the fit child does: routing indices take no gradient
    model, hc = tiny_v4(hc=4), 4
    prompts = ["5 9 2 7 1 3 8 4 6 2 9 9 1", "3 3 7 1 8 2 5 5 4 7 1 2"]
    lens = fit(model, SpaceTok(), prompts, dim_batch=8, skip_first=2, log=lambda *_: None)
    D = model.args.hidden_size
    assert lens.layers == [0, 1] and lens.d_model == D
    for layer in (0, 1):
        assert lens.jacobians[layer].shape == (D, hc * D)
        refs = []
        for p in prompts:
            ids = SpaceTok().encode(p)
            valid = np.zeros(len(ids), bool)
            valid[2:len(ids) - 1] = True
            refs.append(v4_reference_jacobian(model, ids, layer, valid))
        np.testing.assert_allclose(np.array(lens.jacobians[layer].astype(mx.float32)), np.mean(refs, 0),
                                   rtol=2e-2, atol=2e-3, err_msg=f"layer {layer}")
    # Capture applies the [D, hc*D] lens to the streams, then the final norm and head.
    session = CaptureSession(CaptureConfig(residual_norm=False, logits_topk=1, jlens=lens.jacobians, jlens_topk=5))
    install(session, model)
    seen = {}
    rm = add_hook(text_model(model).layers[1], lambda _m, _a, _k, out: seen.__setitem__("S", out))
    toks = TOKENS[1]
    model(toks)
    mx.eval(session.pending_arrays())
    session.end_step()
    recs = {(r.kind, r.layer): np.array(r.value) for r in session.drain()}
    session.close()
    rm()
    from metalscope.adapters.generic import readout
    S = seen["S"].reshape(1, toks.shape[1], -1)
    J = lens.jacobians[1]                                  # as capture computes it: in the lens's dtype
    logits = readout(model)((S.astype(J.dtype) @ J.T).astype(S.dtype))[0].astype(mx.float32)
    want = np.argsort(-np.array(logits), -1)[:, :5]
    np.testing.assert_array_equal(recs[("jlens_ids", 1)].reshape(-1, 5), want)


def test_deepseek_v4_jlens_window_matches_the_full_sequence():
    """A deep window on V4 (its prefix cached, its sliding-window caches restored before every pass) gives the
    full sequence's Jacobian restricted to the window, whatever dim_batch. Up to oMLX V4's own batch numerics:
    one row and a batch of copies differ by ~5e-3 in its logits (every batched row agrees exactly), so the fit,
    which replicates the prompt along the batch, is held to 1e-2 against the one-row reference and between
    batch sizes."""
    from metalscope.jlens.fit import fit, no_index_gradients
    no_index_gradients()
    model = tiny_v4(hc=4)
    ids = [5, 9, 2, 7, 1, 3, 8, 4, 6, 2, 9, 9, 1, 4, 7, 3, 2, 8, 5, 1]     # longer than sliding_window (8)
    window = 6
    valid = np.zeros(len(ids), bool)
    valid[len(ids) - window:len(ids) - 1] = True
    a = fit(model, SpaceTok(), [{"ids": ids, "window": window}], dim_batch=8, log=lambda *_: None)
    b = fit(model, SpaceTok(), [{"ids": ids, "window": window}], dim_batch=3, log=lambda *_: None)
    for layer in (0, 1):
        ref = v4_reference_jacobian(model, ids, layer, valid)
        got = np.array(a.jacobians[layer].astype(mx.float32))
        np.testing.assert_allclose(got, ref, rtol=2e-2, atol=1e-2, err_msg=f"layer {layer}")
        np.testing.assert_allclose(np.array(b.jacobians[layer].astype(mx.float32)), got, rtol=2e-2, atol=1e-2)


def test_deepseek_v41_block():
    """oMLX's own DeepSeek V4.1 MoE (what it serves for deepseek_v41; not mlx_lm's): its Block calls
    ffn(x, image_mask), so the block hook sees x as for any MoE."""
    from omlx.patches.deepseek_v41 import apply_patch
    apply_patch()
    from omlx.patches.deepseek_v41 import language as L
    from omlx.patches.deepseek_v41.config import ModelConfig
    c = ModelConfig(dim=32, moe_inter_dim=16, n_routed_experts=8, n_activated_experts=3, gate_temp=0.7,
                    route_scale=1.5, dtype="bf16", expert_dtype=None)
    mx.random.seed(0)
    block = L.MoE(c)
    block.gate.weight = mx.random.normal(block.gate.weight.shape)
    block.gate.bias = 0.3 * mx.random.normal(block.gate.bias.shape)
    mx.eval(block.parameters())
    rule = routing.rule_for(block)
    assert rule.meta() == dict(rule="deepseek_v41", k=3, exact=True, shared=1, num_experts=8)
    rule.blocks[0] = block
    seen = {}
    remove = add_hook(block.gate, lambda _m, _a, _k, out: seen.__setitem__(0, out))
    for T in (5, 40):
        x = mx.random.normal((1, T, 32))
        block(x, None)
        ids, w = rule.select(rule.logits(block, x)[None], [0])
        compare(rule, {("router_ids", 0): np.array(ids[0]), ("router_probs", 0): np.array(w[0])}, seen)
    remove()


def fused_router_calls(R):
    """Reference for the Qwen MoE blocks oMLX patches: the picks and weights its fused router returned, the n-th
    call being the n-th MoE layer of the forward."""
    def hooks(model):
        real, calls = R.fused_router_topk, []

        def spy(gates, top_k):
            out = real(gates, top_k)
            calls.append(out)
            return out
        R.fused_router_topk = spy
        layers = [i for i, l in enumerate(text_model(model).layers) if routing.moe_block(l) is not None]
        truth = {}

        def done():
            R.fused_router_topk = real
            assert len(calls) == len(layers), "the fused router did not run for every MoE layer"
            truth.update(zip(layers, calls))
        return truth, [done]
    return hooks


def tiny_qwen35_vlm():
    from mlx_vlm.models.qwen3_5_moe import language as L
    from mlx_vlm.models.qwen3_5_moe.config import ModelConfig, TextConfig, VisionConfig
    cfg = TextConfig(model_type="qwen3_5_moe", hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                     linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=16,
                     linear_value_head_dim=16, linear_conv_kernel_dim=4, num_experts=32, num_experts_per_tok=8,
                     shared_expert_intermediate_size=32, moe_intermediate_size=16, rms_norm_eps=1e-6,
                     vocab_size=64, num_key_value_heads=2, max_position_embeddings=256, head_dim=16,
                     full_attention_interval=2,
                     rope_parameters={"type": "default", "mrope_section": [2, 1, 1], "rope_theta": 10000,
                                      "partial_rotary_factor": 0.5})
    mx.random.seed(0)
    lm = L.LanguageModel(cfg, ModelConfig(text_config=cfg, vision_config=VisionConfig(), model_type="qwen3_5_moe"))
    lm.set_dtype(mx.bfloat16)                                  # the fused router runs on bf16 rows
    mx.eval(lm.parameters())
    return lm


@pytest.mark.parametrize("path", ["decode", "mtp-verify"])
def test_qwen35_moe_vlm_fused_router(path):
    """Qwen3.5 / 3.6 MoE through mlx-vlm, oMLX's fused router on (<= 8 rows, bf16). With MTP decode on, verify
    windows go through mlx-vlm's speculative verifier, which never calls the MoE block: the post-attention norm
    is hooked instead, and every layer is still recorded."""
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    from omlx.patches import qwen35_moe_router as R
    R.apply_qwen35_moe_router_patch()
    lm = tiny_qwen35_vlm()
    if path == "mtp-verify":
        lm._omlx_mtp_decode_enabled = True
        forward = lambda toks: Qwen3_5BatchInvariantForward().verify(lm, toks, cache=lm.make_cache())  # noqa: E731
    else:
        forward = lambda toks: lm(toks)  # noqa: E731
    for toks in (mx.array([[3, 17, 5, 42]]), mx.array([[9]])):
        rule, recs, truth = capture(lm, lambda: forward(toks), fused_router_calls(R))
        assert rule.meta() == dict(rule="qwen3_5_moe", k=8, exact=True, shared=1, num_experts=32)
        # bf16: the fused kernel renormalises in fp32 with one rounding, the rule in bf16 as the composed code.
        compare(rule, recs, truth, rtol=2e-2, atol=2e-3)


def test_qwen35_moe_mlx_lm_fused_router():
    """Qwen3.5 / 3.6 MoE as a text-only checkpoint (mlx_lm's qwen3_5_moe: Qwen3NextSparseMoeBlock), oMLX's
    fused router on."""
    import importlib
    from omlx.patches import qwen35_moe_router as R
    from tests.test_routing import COMMON, FAMILIES
    R.apply_qwen35_moe_router_patch()
    mod = importlib.import_module("mlx_lm.models.qwen3_5_moe")
    cfg = dict(COMMON, model_type="qwen3_5_moe", **dict(FAMILIES["qwen3_5_moe"][0], num_experts=32,
                                                         num_experts_per_tok=8, hidden_size=64))
    mx.random.seed(0)
    model = mod.Model(mod.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    for toks in (mx.array([[3, 17, 5, 42]]), mx.array([[9]])):
        rule, recs, truth = capture(model, lambda: model(toks), fused_router_calls(R))
        assert rule.meta() == dict(rule="qwen3_next", k=8, exact=True, shared=1, num_experts=32)
        compare(rule, recs, truth, rtol=2e-2, atol=2e-3)


def test_qwen35_verify_bypasses_the_block():
    """Why the norm is hooked under MTP: hooked on the block, a verify window records nothing."""
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    lm = tiny_qwen35_vlm()
    _, recs, _ = capture(lm, lambda: Qwen3_5BatchInvariantForward().verify(lm, mx.array([[3, 17]]),
                                                                            cache=lm.make_cache()),
                         lambda m: ({}, []))
    assert not any(k == "router_ids" for k, _ in recs)


@pytest.mark.parametrize("hidden,dtype,tol", [(64, mx.float32, 1e-2), (4160, mx.bfloat16, 2e-2)])
def test_qwen35_dense_verify_window_runs_the_layer_hooks(monkeypatch, hidden, dtype, tol):
    """Qwen3.8 27B (qwen3_5, dense) under MTP: a verify window computes every layer in the verifier's ``_layer``,
    oMLX's own wrapper of it armed, and never calls the layer. With patch_mtp_verify the window records the same
    residual norms and logit-lens rows as the plain forward over the same tokens (up to the verifier's own kernels'
    rounding), and a steering add lands in it as in that forward. At 4160 wide in bf16 the window takes oMLX's fused
    add+norm path (``_add_rms_eligible``: bf16, 4096 < width <= 16384), the 27B's own, which reuses the next layer's
    input norm when the hidden it returned is unchanged."""
    from mlx_vlm.models.qwen3_5 import language as L
    from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
    from omlx.patches import qwen35_verify_qmm as Q
    from metalscope.omlx import patches_mtp
    from metalscope.steer import SteerPlan, SteerSession
    Q.apply_verify_qmm_patch()                                  # oMLX wraps _layer before the model loads
    patches_mtp.patch_mtp_verify()                              # ours at attach, around it
    fused, real = [], Q.add_rms_norm
    monkeypatch.setattr(Q, "add_rms_norm", lambda *a: fused.append(1) or real(*a))
    cfg = TextConfig(model_type="qwen3_5", hidden_size=hidden, intermediate_size=128, num_hidden_layers=4,
                     num_attention_heads=4, linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=16,
                     linear_value_head_dim=16, linear_conv_kernel_dim=4, rms_norm_eps=1e-6, vocab_size=64,
                     num_key_value_heads=2, max_position_embeddings=256, head_dim=16, full_attention_interval=2,
                     rope_parameters={"type": "default", "mrope_section": [2, 1, 1], "rope_theta": 10000,
                                      "partial_rotary_factor": 0.5})
    mx.random.seed(0)
    lm = L.LanguageModel(cfg, ModelConfig(text_config=cfg, vision_config=VisionConfig(), model_type="qwen3_5"))
    lm.set_dtype(dtype)
    mx.eval(lm.parameters())
    toks = mx.array([[3, 17, 5, 42]])

    def verify():
        Q.set_verify_qmm_armed(True)
        try:
            return lm(toks, cache=lm.make_cache(), capture_layer_ids=[], speculative_verify=True).logits
        finally:
            Q.set_verify_qmm_armed(False)

    def plain():
        return lm(toks, cache=lm.make_cache()).logits

    def recorded(forward):
        s = CaptureSession(CaptureConfig(logits_topk=5, logit_lens_topk=5))
        install(s, lm)
        forward()
        mx.eval(s.pending_arrays())
        s.end_step()
        out = {(r.kind, r.layer): np.array(r.value) for r in s.drain()}
        s.close()
        return out
    window, ref = recorded(verify), recorded(plain)
    assert bool(fused) == (hidden > 4096)                       # the fused path ran exactly where it is eligible
    for layer in range(4):
        np.testing.assert_allclose(window[("resid_norm", layer)], ref[("resid_norm", layer)], rtol=tol)
        np.testing.assert_array_equal(window[("lens_ids", layer)][..., 0], ref[("lens_ids", layer)][..., 0])
    steer = SteerSession(lambda rid, why: pytest.fail(why))
    steer.install(lm)
    steer.set_rows([(0, "s", SteerPlan({1: np.full(hidden, 0.5, np.float32)}, "all", {}))], 1, "decode")
    fused.clear()
    try:
        steered = np.array(plain().astype(mx.float32))
        np.testing.assert_allclose(np.array(verify().astype(mx.float32)), steered, rtol=tol, atol=tol)
    finally:
        steer.close()
    assert bool(fused) == (hidden > 4096)
    assert np.abs(steered - np.array(plain().astype(mx.float32))).max() > 0.5       # the add moves it


def tiny_flash_next():
    """Flash Next (oMLX's vendored qwen4_exp), 4 layers, 4 hyper-connection streams, bf16."""
    from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language as L
    from mlx_vlm.models.qwen4_exp.config import ModelConfig, TextConfig, VisionConfig
    cfg = TextConfig(
        model_type="qwen4_exp", hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, num_experts=32, num_experts_per_tok=10, shared_expert_intermediate_size=32,
        moe_intermediate_size=16, rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=2,
        max_position_embeddings=256, hc_count=4, hc_lowrank=8, head_dim=16, full_attention_interval=2,
        indexer_n_heads=2, indexer_head_dim=16, indexer_budget=8, indexer_compress_ratio=2,
        ple_embed_dim=32, heads_per_ngram=2, ple_layer_ids=[3], eos_token_id=1,
        rope_parameters={"type": "default", "mrope_section": [1, 1, 0], "rope_theta": 10000,
                         "partial_rotary_factor": 0.25, "mrope_interleaved": True})
    mx.random.seed(0)
    model = L.LanguageModel(cfg, ModelConfig(text_config=cfg, vision_config=VisionConfig(), model_type="qwen4_exp"))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def test_flash_next_fused_router():
    """Flash Next (oMLX's vendored qwen4_exp, its own adapter): mlx-vlm's Qwen3.5 MoE block, so the same rule."""
    from omlx.patches import qwen35_moe_router as R
    model = tiny_flash_next()
    R.apply_qwen35_moe_router_patch()
    rule, recs, truth = capture(model, lambda: model(mx.array([[3, 17, 5, 42]])), fused_router_calls(R))
    assert rule.meta() == dict(rule="qwen3_5_moe", k=10, exact=True, shared=1, num_experts=32)
    compare(rule, recs, truth, rtol=2e-2, atol=2e-3)


def test_flash_next_write_norms_read_the_pending_write():
    """resid_write_norm on the vendored Flash Next: a layer reads its residual with the previous layer's write applied,
    which with oMLX's fused writes on is still pending (the layer's ``write`` keyword); checked against each layer's
    own arguments and output, seen by a hook of its own."""
    from mlx_vlm.models.qwen4_exp import language as L
    model = tiny_flash_next()
    s = CaptureSession(CaptureConfig(residual_norm=False))
    assert install(s, model) == "qwen4_exp"
    seen, pending = {}, []
    full = lambda h, w: h if w is None else h + (w[0][..., None, :] * w[1][..., None]).reshape(h.shape)

    def look(_m, a, k, out, i):
        pending.append(k.get("write") is not None)
        x_out = full(out[0], out[1:]) if isinstance(out, tuple) else out
        seen[i] = np.array((x_out - full(a[0], k.get("write"))).astype(mx.float32))
    removers = [add_hook(layer, lambda _m, a, k, out, i=i: look(_m, a, k, out, i))
                for i, layer in enumerate(text_model(model).layers)]
    model(TOKENS[0])
    mx.eval(s.pending_arrays())
    s.end_step()
    recs = {(r.kind, r.layer): np.array(r.value) for r in s.drain()}
    s.close()
    for r in removers:
        r()
    assert sum(pending) == (len(seen) - 1 if L.hc_fused.write_enabled() else 0)
    for i, d in seen.items():
        want = np.linalg.norm(d.reshape(*d.shape[:-1], 4, -1), axis=-1)
        np.testing.assert_allclose(recs[("resid_write_norm", i)], want, rtol=1e-3, atol=1e-5, err_msg=f"layer {i}")


@pytest.mark.parametrize("build", ["v4", "flash_next"])
def test_both_lenses_on_hyper_connection_streams(build, monkeypatch):
    """The two models with hyper-connection streams: the J-lens reads the streams (J: D x hc*D), the logit lens the
    residual hc_head mixes them to. Both are recorded; every read of a step through one LM-head call equals reading
    each layer on its own; a prompt chunk past lens_tail reads both lenses on its end only."""
    import metalscope.capture as C
    model = tiny_v4(hc=4) if build == "v4" else tiny_flash_next()
    probe = CaptureSession(CaptureConfig(layers=[]))
    adapter = install(probe, model)
    probe.close()
    hc, D, n = probe.hc_count, text_model(model)["embed_tokens"].weight.shape[-1], probe.n_layers
    assert (adapter, hc) == (("generic", 4) if build == "v4" else ("qwen4_exp", 4))
    J = {l: (mx.random.normal((D, hc * D), key=mx.random.key(l)) * 0.05).astype(mx.bfloat16) for l in range(n)}

    def run(rows, toks):
        monkeypatch.setattr(C, "LENS_BATCH_ROWS", rows)
        s = CaptureSession(CaptureConfig(router=True, residual_norm=False, logit_lens_topk=10, lens_every=1,
                                         jlens=J, jlens_topk=10, lens_tail=6))
        install(s, model)
        model(toks)
        mx.eval(s.pending_arrays())
        s.end_step()
        out = {(r.kind, r.layer): np.array(r.value) for r in s.drain()}
        s.close()
        return out
    for toks in (mx.array([[5]]), TOKENS[0], TOKENS[1]):          # a decode step, a short and a long prompt chunk
        T = toks.shape[1]
        one, together = run(1, toks), run(64, toks)
        for l in range(n):
            j, lg = one[("jlens_ids", l)].reshape(T, 10), one[("lens_ids", l)].reshape(T, 10)
            for ids in (j, lg):                                      # both lenses: the chunk's end
                assert (ids[max(0, T - 6):] >= 0).all() and (ids[:max(0, T - 6)] == -1).all()
            for kind in ("jlens_ids", "lens_ids"):
                np.testing.assert_array_equal(one[(kind, l)], together[(kind, l)], err_msg=f"{build} {kind} L{l} T={T}")
            for kind in ("jlens_logp", "lens_logp"):
                a, b = one[(kind, l)], together[(kind, l)]
                ok = np.isfinite(a)
                np.testing.assert_array_equal(ok, np.isfinite(b))
                np.testing.assert_allclose(a[ok], b[ok], rtol=0, atol=5e-2, err_msg=f"{build} {kind} L{l} T={T}")

import mlx.core as mx
import numpy as np
import pytest

from metalscope.jlens.fit import fit
from tests.helpers import tiny_llama


class Tok:
    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]


def reference_jacobian(model, ids, layer, skip_first, valid=None):
    """Independent J_l: run layers l+1.. explicitly from h_l and differentiate."""
    inner = model.model
    h = inner.embed_tokens(mx.array([ids]))
    for lyr in inner.layers[: layer + 1]:
        h = lyr(h, mask="causal", cache=None)
    T, d = h.shape[1], h.shape[2]
    if valid is None:
        valid = np.zeros(T, bool)
        valid[skip_first:T - 1] = True

    def tail(h_l):
        x = h_l
        for lyr in inner.layers[layer + 1:]:
            x = lyr(x, mask="causal", cache=None)
        return [x]

    rows = []
    for i in range(d):
        cot = np.zeros((1, T, d), np.float32)
        cot[0, valid, i] = 1
        _, (g,) = mx.vjp(tail, [h], [mx.array(cot)])
        rows.append(np.array(g)[0, valid].mean(0))
    return np.stack(rows)


def test_fit_matches_independent_jacobian():
    model = tiny_llama()
    prompts = ["5 9 2 7 1 3 8 4 6 2 9 9 1", "3 3 7 1 8 2 5 5 4 7 1 2"]
    lens = fit(model, Tok(), prompts, dim_batch=16, skip_first=2, log=lambda *_: None)
    assert lens.layers == [0, 1] and lens.meta["n_prompts"] == 2
    for layer in (0, 1):
        ref = np.mean([reference_jacobian(model, Tok().encode(p), layer, 2) for p in prompts], axis=0)
        got = np.array(lens.jacobians[layer].astype(mx.float32))
        np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-3)


def test_window_fit_matches_full_sequence_jacobian():
    """A deep window (prefix cached, no gradients) gives the full-sequence
    Jacobian restricted to the window's positions."""
    model = tiny_llama()
    ids = [5, 9, 2, 7, 1, 3, 8, 4, 6, 2, 9, 9, 1, 4, 7, 3]
    window = 6
    lens = fit(model, Tok(), [{"ids": ids, "window": window}], dim_batch=16, log=lambda *_: None)
    valid = np.zeros(len(ids), bool)
    valid[len(ids) - window:len(ids) - 1] = True
    for layer in (0, 1):
        ref = reference_jacobian(model, ids, layer, 0, valid=valid)
        got = np.array(lens.jacobians[layer].astype(mx.float32))
        np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-3)


def tiny_qwen3_next():
    """Linear-attention (GatedDeltaNet, ArraysCache) layer then a full-attention layer."""
    from mlx_lm.models import qwen3_next
    mx.random.seed(0)
    args = qwen3_next.ModelArgs(
        model_type="qwen3_next", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2,
        decoder_sparse_step=1, shared_expert_intermediate_size=32, mlp_only_layers=[], moe_intermediate_size=32,
        rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=2, rope_theta=10000.0, partial_rotary_factor=0.5,
        max_position_embeddings=256, head_dim=8, full_attention_interval=2)
    model = qwen3_next.Model(args)
    mx.eval(model.parameters())
    return model


def test_window_fit_does_not_depend_on_dim_batch_with_recurrent_state():
    """Each backward pass of a deep window restarts from the cached prefix: one
    pass over all dims and many small passes give the same lens. (A shared
    ArraysCache slot list once let each pass start from the previous one's state.)"""
    model = tiny_qwen3_next()
    prompt = [{"ids": [5, 9, 2, 7, 1, 3, 8, 4, 6, 2, 9, 9, 1, 4, 7, 3], "window": 6}]
    one = fit(model, Tok(), prompt, dim_batch=32, log=lambda *_: None)
    many = fit(model, Tok(), prompt, dim_batch=4, log=lambda *_: None)
    for layer in one.layers:
        np.testing.assert_allclose(np.array(many.jacobians[layer].astype(mx.float32)),
                                   np.array(one.jacobians[layer].astype(mx.float32)), rtol=2e-2, atol=2e-3)


def test_fit_stops_cleanly_and_resumes(tmp_path):
    """should_stop ends the fit at a checkpoint; resuming gives the uninterrupted lens."""
    model = tiny_llama()
    prompts = ["5 9 2 7 1 3 8 4 6 2 9 9 1", "3 3 7 1 8 2 5 5 4 7 1 2", "1 2 3 4 5 6 7 8 9 1 2 3"]
    quiet = lambda *_: None
    full = fit(model, Tok(), prompts, dim_batch=16, skip_first=2, log=quiet)
    calls = []
    fit(model, Tok(), prompts, dim_batch=16, skip_first=2, log=quiet, checkpoint=tmp_path,
        should_stop=lambda: calls.append(1) or len(calls) > 2)
    import json
    assert json.loads((tmp_path / "fit_state.json").read_text())["done"] == 2
    resumed = fit(model, Tok(), prompts, dim_batch=16, skip_first=2, log=quiet, checkpoint=tmp_path)
    for l in full.layers:
        np.testing.assert_allclose(np.array(resumed.jacobians[l].astype(mx.float32)),
                                   np.array(full.jacobians[l].astype(mx.float32)), rtol=1e-2, atol=1e-3)


def test_dim_limit_fits_only_the_first_rows():
    model = tiny_llama()
    lines = []
    lens = fit(model, Tok(), ["5 9 2 7 1 3 8 4 6 2 9 9 1"], dim_batch=4, skip_first=2, dim_limit=8, log=lines.append)
    J = np.array(lens.jacobians[0].astype(mx.float32))
    assert np.abs(J[:8]).sum() > 0 and np.abs(J[8:]).sum() == 0
    assert "2 passes" in lines[-1] and "peak" in lines[-1]


def test_a_fit_forward_runs_a_model_that_async_evals_in_its_forward(monkeypatch):
    """oMLX's DeepSeek V4 ends each forward with mx.async_eval (of its cache arrays), which mx.vjp refuses: the fit's
    forward makes it a no-op while it runs, mlx's own back afterwards, after a forward that raises too."""
    from metalscope.jlens.fit import differentiable
    model, real = tiny_llama(), mx.async_eval
    inner = model.model
    own = type(inner).__call__

    def call(self, inputs, *a, **k):                    # as V4's: the forward ends with an async_eval
        if inputs.shape[1] == 0:
            raise RuntimeError("no tokens")
        out = own(self, inputs, *a, **k)
        mx.async_eval(out)
        return out
    monkeypatch.setattr(type(inner), "__call__", call)
    ids = mx.array([[3, 1, 4, 1, 5]])
    shape = inner(ids).shape
    def traced(e):                                      # what the forward does to an array the vjp traces: refused
        mx.async_eval(e * 2)
        return [e * 2]
    with pytest.raises((ValueError, RuntimeError), match="async_eval"):
        mx.vjp(traced, [mx.zeros(shape)], [mx.ones(shape)])
    with differentiable(model, [0], hyper=False) as (probes, forward):
        def f(e):
            probes.eps = {0: e}
            return [forward(ids)]
        _, (g,) = mx.vjp(f, [mx.zeros(shape)], [mx.ones(shape)])
        mx.eval(g)
        assert mx.async_eval is real
        with pytest.raises(RuntimeError, match="no tokens"):
            forward(mx.zeros((1, 0), dtype=mx.int32))
        assert mx.async_eval is real
    assert np.abs(np.array(g)).max() > 0


def test_qwen4_exp_fit_setup_switches_fused_kernels_off(monkeypatch):
    """oMLX 0.7.3 gates its fused kernels behind module flags: fit_setup sets every one in the loaded qwen4_exp
    modules, and the expert lookup's, and leaves other modules alone."""
    import sys
    import types
    from metalscope.adapters import qwen4_exp

    hc = types.ModuleType("mlx_vlm.models.qwen4_exp.hc_fused")
    hc.__file__ = "/x/vendor/mlx_vlm/models/qwen4_exp/hc_fused.py"
    hc._DISABLED, hc._WRITE_DISABLED, hc._NAX_AVAILABLE = False, False, None
    lang = types.ModuleType("mlx_vlm.models.qwen4_exp.language")
    lang.__file__ = "/x/vendor/mlx_vlm/models/qwen4_exp/language.py"
    lang._DEPTHWISE_CONV_STATE = {"enabled": True, "kernel": None}
    other = types.ModuleType("mlx_vlm.models.qwen3_5.language")
    other.__file__ = "/x/mlx_vlm/models/qwen3_5/language.py"
    other._DISABLED = False
    gqmm = types.ModuleType("omlx.patches.m5_gather_qmm")
    gqmm.__file__ = "/x/omlx/patches/m5_gather_qmm.py"
    gqmm._nax_host = None
    for m in (hc, lang, other, gqmm):
        monkeypatch.setitem(sys.modules, m.__name__, m)

    done = qwen4_exp.fit_setup()
    assert hc._DISABLED is True and hc._WRITE_DISABLED is True and hc._NAX_AVAILABLE is None
    assert lang._DEPTHWISE_CONV_STATE["enabled"] is False
    assert gqmm._nax_host is False
    assert other._DISABLED is False
    assert len(done) == 4
    assert qwen4_exp.fit_setup() == ["omlx.patches.m5_gather_qmm._nax_host"]   # idempotent: only the host flag is set again

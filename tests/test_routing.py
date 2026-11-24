"""Recorded routing is the model's own: for each MoE family, a tiny random model's hooks + rule (routing.py) record
the experts its MoE blocks actually used and the weights they mixed them with.

The reference is taken inside the model: a gate that returns (inds, scores) (DeepSeek, GLM, Granite) is hooked
directly; otherwise the experts module is made to return one-hot rows (slot j -> unit vector j) and shared experts
zeros, so the block's output is the weight vector itself, in the order of the indices the experts module was called
with."""

import importlib

import mlx.core as mx
import numpy as np
import pytest

from metalscope.capture import CaptureConfig, CaptureSession
from metalscope.hooks import add_hook
from metalscope import routing
from metalscope.adapters import install

COMMON = dict(vocab_size=64, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
              rope_theta=10000.0, max_position_embeddings=256, tie_word_embeddings=False, head_dim=8)
MLA = dict(kv_lora_rank=8, q_lora_rank=8, qk_rope_head_dim=8, v_head_dim=8, qk_nope_head_dim=8)
LINEAR_ATTN = dict(linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                   linear_conv_kernel_dim=4, full_attention_interval=2)

# family (mlx_lm module) -> (tiny config, expected rule name, the rule differs from "softmax, top k, unnormalised")
FAMILIES = {
    "qwen2_moe": (dict(num_hidden_layers=2, intermediate_size=64, num_experts=8, num_experts_per_tok=3,
                       moe_intermediate_size=16, shared_expert_intermediate_size=16, decoder_sparse_step=1,
                       mlp_only_layers=[]), "qwen2_moe", False),
    "qwen3_moe": (dict(num_hidden_layers=2, intermediate_size=64, num_experts=8, num_experts_per_tok=3,
                       moe_intermediate_size=16, decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True),
                  "qwen3_moe", True),
    "qwen3_next": (dict(num_hidden_layers=4, intermediate_size=64, num_experts=8, num_experts_per_tok=3,
                        moe_intermediate_size=16, shared_expert_intermediate_size=16, decoder_sparse_step=1,
                        mlp_only_layers=[], norm_topk_prob=False, partial_rotary_factor=0.5, **LINEAR_ATTN),
                   "qwen3_next", False),
    "qwen3_5_moe": (dict(num_hidden_layers=4, intermediate_size=64, num_experts=8, num_experts_per_tok=3,
                         moe_intermediate_size=16, shared_expert_intermediate_size=16, norm_topk_prob=True,
                         **LINEAR_ATTN), "qwen3_next", True),
    "olmoe": (dict(num_hidden_layers=2, intermediate_size=16, num_experts=8, num_experts_per_tok=3,
                   norm_topk_prob=True), "olmoe", True),
    "mixtral": (dict(num_hidden_layers=2, intermediate_size=16, num_local_experts=8, num_experts_per_tok=2),
                "mixtral", True),
    "phimoe": (dict(num_hidden_layers=2, intermediate_size=16, num_local_experts=8, num_experts_per_tok=2,
                    rope_scaling=dict(short_factor=[1.0] * 4, long_factor=[1.0] * 4, short_mscale=1.0, long_mscale=1.0),
                    original_max_position_embeddings=256), "phimoe", True),
    "gpt_oss": (dict(num_hidden_layers=2, intermediate_size=16, num_local_experts=8, num_experts_per_tok=4,
                     sliding_window=16, layer_types=["sliding_attention", "full_attention"]), "gpt_oss", True),
    "granitemoe": (dict(num_hidden_layers=2, intermediate_size=16, num_local_experts=8, num_experts_per_tok=2,
                        logits_scaling=1.0, attention_multiplier=1.0, embedding_multiplier=1.0,
                        residual_multiplier=1.0, attention_bias=False), "granitemoe", True),
    "minimax": (dict(num_hidden_layers=2, intermediate_size=16, num_local_experts=8, num_experts_per_tok=3,
                     rotary_dim=8, use_qk_norm=True, shared_intermediate_size=0), "minimax", True),
    "deepseek": (dict(num_hidden_layers=2, intermediate_size=64, moe_intermediate_size=16, n_routed_experts=8,
                      num_experts_per_tok=3, n_shared_experts=1, first_k_dense_replace=1, moe_layer_freq=1),
                 "deepseek", False),
    "deepseek_v2": (dict(num_hidden_layers=3, intermediate_size=64, moe_intermediate_size=16, n_routed_experts=8,
                         num_experts_per_tok=3, n_shared_experts=1, first_k_dense_replace=1,
                         topk_method="group_limited_greedy", n_group=4, topk_group=2, routed_scaling_factor=1.7,
                         rope_scaling=dict(factor=1.0, original_max_position_embeddings=256), **MLA),
                    "deepseek_v2", True),
    "deepseek_v3": (dict(num_hidden_layers=3, intermediate_size=64, moe_intermediate_size=16, n_routed_experts=8,
                         num_experts_per_tok=3, n_shared_experts=1, first_k_dense_replace=1, n_group=4, topk_group=2,
                         routed_scaling_factor=2.5, norm_topk_prob=True, **MLA), "deepseek_v3", True),
    "deepseek_v32": (dict(num_hidden_layers=3, intermediate_size=64, moe_intermediate_size=16, n_routed_experts=8,
                          num_experts_per_tok=3, n_shared_experts=1, first_k_dense_replace=1, n_group=4,
                          topk_group=2, routed_scaling_factor=2.5, index_head_dim=16, index_n_heads=2,
                          index_topk=4, **MLA), "deepseek_v32", True),
    "glm4_moe": (dict(num_hidden_layers=3, intermediate_size=64, moe_intermediate_size=16, n_routed_experts=8,
                      num_experts_per_tok=3, n_shared_experts=1, first_k_dense_replace=1, n_group=2, topk_group=1,
                      routed_scaling_factor=1.5, norm_topk_prob=True, partial_rotary_factor=0.5, use_qk_norm=False,
                      attention_bias=False, rope_scaling=None), "glm4_moe", True),
    "glm4_moe_lite": (dict(num_hidden_layers=3, intermediate_size=64, moe_intermediate_size=16, n_routed_experts=8,
                           num_experts_per_tok=3, n_shared_experts=1, first_k_dense_replace=1, n_group=1,
                           topk_group=1, routed_scaling_factor=1.8, **MLA), "glm4_moe_lite", True),
    "deepseek_v41": (dict(num_hidden_layers=2, moe_intermediate_size=16, n_routed_experts=8, num_experts_per_tok=3,
                          q_lora_rank=8, qk_rope_head_dim=4, o_groups=2, o_lora_rank=8, head_dim=8, index_n_heads=2,
                          index_head_dim=8, index_topk=4, rms_norm_eps=1e-6, hc_mult=2, sliding_window=8,
                          gate_temp=0.7), "deepseek_v41", True),
}


def tiny(name: str):
    mod = importlib.import_module(f"mlx_lm.models.{name}")
    cfg, _, _ = FAMILIES[name]
    args = mod.ModelArgs.from_dict(dict(COMMON, model_type=name, **cfg))
    mx.random.seed(0)
    model = mod.Model(args)
    randomize_gates(model)
    mx.eval(model.parameters())
    return model


def randomize_gates(model, seed: int = 1):
    """Spread the router logits (DeepSeek-style gates start at zero) and give every correction bias a nonzero value
    big enough to change some picks."""
    from metalscope.adapters.generic import text_model
    mx.random.seed(seed)
    for layer in text_model(model).layers:
        block = routing.moe_block(layer)
        if block is None:
            continue
        for owner in (block, block["gate"] if "gate" in block else None):
            if owner is not None and isinstance(getattr(owner, "e_score_correction_bias", None), mx.array):
                b = owner.e_score_correction_bias
                owner.e_score_correction_bias = 0.3 * mx.random.normal(b.shape).astype(b.dtype)
        gate = block["gate"] if "gate" in block else None
        if gate is not None and not hasattr(gate, "to_quantized") and isinstance(getattr(gate, "weight", None), mx.array):
            gate.weight = mx.random.normal(gate.weight.shape).astype(gate.weight.dtype)


def reference_hooks(model):
    """Hooks recording, per MoE layer, the (ids, weights) the model itself used, and the block's input. Returns
    (truth, inputs, removers)."""
    from metalscope.adapters.generic import text_model
    truth, inputs, removers = {}, {}, []
    for i, layer in enumerate(text_model(model).layers):
        block = routing.moe_block(layer)
        if block is None:
            continue
        gate = next(block[g] for g in routing.GATE_ATTRS if g in block)

        def on_gate(_m, _a, _k, out, i=i):
            if isinstance(out, tuple):
                truth[i] = ("gate", out[0], out[1])
        removers.append(add_hook(gate, on_gate))

        experts = block["switch_mlp"] if "switch_mlp" in block else block["experts"]
        seen = {}

        def one_hot(_m, args, _k, out, seen=seen):
            seen["inds"] = args[1]
            k, D = out.shape[-2], out.shape[-1]
            return mx.broadcast_to(mx.eye(k, D, dtype=out.dtype), out.shape)
        removers.append(add_hook(experts, one_hot))
        for name in ("shared_expert", "shared_experts", "shared_mlp"):
            if name in block:
                removers.append(add_hook(block[name], lambda _m, _a, _k, out: mx.zeros_like(out)))

        def on_block(_m, args, _k, out, i=i, seen=seen):
            inputs[i] = args[0]
            if i not in truth or truth[i][0] != "gate":
                k = seen["inds"].shape[-1]
                truth[i] = ("onehot", seen["inds"], out[..., :k])
        removers.append(add_hook(block, on_block))
    return truth, inputs, removers


def recorded(model, tokens):
    session = CaptureSession(CaptureConfig(layers=[], residual_norm=False, router=True))
    install(session, model)
    truth, inputs, removers = reference_hooks(model)
    model(tokens)
    mx.eval(session.pending_arrays(), {i: v[1:] for i, v in truth.items()}, inputs)
    session.end_step()
    recs = {(r.kind, r.layer): r.value for r in session.drain()}
    session.close()
    for r in removers:
        r()
    return session.routing, recs, truth, inputs


def by_id(ids, w):
    """Sort each row's (id, weight) pairs by id."""
    order = np.argsort(ids, axis=-1)
    return np.take_along_axis(ids, order, -1), np.take_along_axis(w, order, -1)


TOKENS = {"short": mx.array([[3, 17, 5, 42, 9, 11, 2, 60]]),                      # <= 16: stacked at step end
          "long": mx.array([[(7 * i + 3) % 64 for i in range(24)]])}               # > 16: reduced per layer


@pytest.mark.parametrize("name", list(FAMILIES))
def test_recorded_routing_is_the_models_own(name):
    _, rule_name, differs = FAMILIES[name]
    model = tiny(name)
    for steps, toks in TOKENS.items():
        rule, recs, truth, inputs = recorded(model, toks)
        assert rule is not None and rule.name == rule_name and rule.exact, (steps, rule)
        assert rule.k == FAMILIES[name][0]["num_experts_per_tok"]
        assert sorted(truth) == sorted(rule.blocks) and truth
        naive_differs = False
        for layer, (_, t_ids, t_w) in truth.items():
            ids, w = recs[("router_ids", layer)], recs[("router_probs", layer)]
            t_ids = np.array(t_ids).reshape(ids.shape)
            t_w = np.array(t_w.astype(mx.float32)).reshape(w.shape)
            assert ids.shape[-1] == rule.k and w.dtype == np.float32
            a_ids, a_w = by_id(ids, w)
            b_ids, b_w = by_id(t_ids, t_w)
            np.testing.assert_array_equal(a_ids, b_ids, err_msg=f"{name} {steps} layer {layer}")
            np.testing.assert_allclose(a_w, b_w, rtol=1e-5, atol=1e-6, err_msg=f"{name} {steps} layer {layer}")
            # The old recording (softmax over all experts, top k, as is) would have been wrong here.
            G = rule.logits(rule.blocks[layer], inputs[layer])
            p = np.array(mx.softmax(G.astype(mx.float32), axis=-1))
            n_ids = np.argsort(-p, axis=-1)[..., :rule.k]
            n_ids, n_w = by_id(n_ids, np.take_along_axis(p, n_ids, -1))
            naive_differs |= (n_ids != b_ids).any() or not np.allclose(n_w, b_w, rtol=1e-4)
        assert naive_differs == differs, name


def test_noaux_bias_and_groups_change_the_picks():
    """The V3 rule's correction bias and group limit both matter on the tiny model (else the comparison above
    would not test them)."""
    model = tiny("deepseek_v3")
    rule, _, _, inputs = recorded(model, TOKENS["short"])
    layer = min(rule.blocks)
    G = rule.logits(rule.blocks[layer], inputs[layer])[None]
    ids = np.sort(np.array(rule.select(G, [layer])[0]), -1)
    for change in (dict(bias=None), dict(groups=None)):
        from dataclasses import replace
        other = replace(rule, **change, _bias_cache=(None, None))
        assert (np.sort(np.array(other.select(G, [layer])[0]), -1) != ids).any(), change


def test_k_comes_from_the_model_not_ten():
    for k in (1, 3, 5):
        cfg = dict(FAMILIES["qwen3_moe"][0], num_experts_per_tok=k, num_experts=16)
        from mlx_lm.models import qwen3_moe
        model = qwen3_moe.Model(qwen3_moe.ModelArgs.from_dict(dict(COMMON, model_type="qwen3_moe", **cfg)))
        mx.eval(model.parameters())
        rule, recs, _, _ = recorded(model, TOKENS["short"])
        assert rule.meta() == dict(rule="qwen3_moe", k=k, exact=True, shared=0, num_experts=16)
        assert recs[("router_ids", 0)].shape == (1, 8, k)
        np.testing.assert_allclose(recs[("router_probs", 1)].sum(-1), 1.0, rtol=1e-5)   # norm_topk_prob


def test_shared_experts_counted():
    assert recorded(tiny("qwen3_next"), TOKENS["short"])[0].shared == 1
    assert recorded(tiny("deepseek_v3"), TOKENS["short"])[0].shared == 1
    assert recorded(tiny("mixtral"), TOKENS["short"])[0].shared == 0


def test_unknown_block_is_approximate():
    """A block we have no rule for, with a linear gate and a k: softmax, top k, marked approximate."""
    import mlx.nn as nn

    class OddMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(32, 6, bias=False)
            self.top_k = 2

        def __call__(self, x):
            return x

    rule = routing.rule_for(OddMoE())
    assert rule.meta() == dict(rule="unknown:OddMoE", k=2, exact=False, shared=0, num_experts=6)
    G = mx.random.normal((1, 1, 4, 6))
    ids, w = rule.select(G, [0])
    p = np.array(mx.softmax(G, axis=-1))
    np.testing.assert_array_equal(np.sort(np.array(ids), -1), np.sort(np.argsort(-p, -1)[..., :2], -1))


def test_dense_model_has_no_rule():
    from tests.helpers import tiny_llama
    session = CaptureSession(CaptureConfig(router=True))
    install(session, tiny_llama())
    assert session.routing is None


def _routed_trace(ids, probs, meta):
    from metalscope.store import Trace
    L, T, _ = ids.shape
    arrays = {}
    for l in range(L):
        arrays[f"router_ids/{l}"], arrays[f"router_probs/{l}"] = ids[l].astype(np.int32), probs[l].astype(np.float32)
    texts = [f" w{t}" for t in range(T)]
    return Trace(arrays, list(range(T)), texts, T // 2, dict(num_experts=16, title="t", **meta))


def test_pooling_traces_of_different_k_refuses():
    """Each trace records its model's own k: pooling two models' traces is refused, not cut to the smaller k."""
    from metalscope import experts as X
    rng = np.random.default_rng(0)
    L, T, E = 2, 12, 16
    p = rng.dirichlet(np.ones(E), size=(L, T))
    traces = []
    for k in (3, 4):
        ids = np.argsort(-p, -1)[..., :k]
        w = np.take_along_axis(p, ids, -1)
        traces.append(_routed_trace(ids, w / w.sum(-1, keepdims=True),
                                    dict(routing=dict(rule="qwen3_moe", k=k, exact=True, shared=0, num_experts=E))))
    with pytest.raises(ValueError, match="different models"):
        X.build_corpus(traces, dedupe=False)

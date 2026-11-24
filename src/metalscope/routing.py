"""MoE routing as each model family computes it: which experts a token goes to, and the weights they are mixed with.

oMLX's fused decode routers read ``gate.weight`` directly, so a hook on the gate's output misses decode steps. The
adapters recompute the router logits from the MoE block's own input instead, and a ``Rule`` turns them into the
experts and weights the model itself uses. The rule is read off the block and its gate (score function, correction
bias, group-limited selection, renormalisation, scaling, k) and copies the model's own code op for op, in the model's
own dtype, so the recorded picks are the model's picks. Families, by MoE block class (the code each copies):

- softmax, then the top k, renormalised when the model does: Qwen2 MoE (never), Qwen3 MoE, Qwen3-Next / Qwen3.5 /
  Qwen3.6 (``norm_topk_prob``), mlx-vlm's Qwen3.5 MoE (always; Flash Next's experts too), OLMoE, DeepSeek (V1);
- the top k of the raw logits, then softmax over those k: Mixtral, PhiMoE, gpt-oss (its router has a bias), Granite MoE;
- DeepSeek V2: softmax, the ``topk_group`` groups with the highest expert kept ("group_limited_greedy"), the top k,
  times ``routed_scaling_factor``, never renormalised;
- sigmoid with ``e_score_correction_bias`` for selection only ("noaux_tc"): DeepSeek V3 / V3.2, Kimi K2 (served as
  DeepSeek V3), GLM-4.5 / 4.6 / 4.7 MoE (``glm4_moe``, ``glm4_moe_lite``): groups ranked by their top two biased scores,
  the top k inside the kept groups, weighted by the unbiased scores (renormalised when k > 1 and ``norm_topk_prob``),
  times ``routed_scaling_factor``; MiniMax (no groups, always renormalised, no scaling);
- DeepSeek V4 (oMLX's own model) and V4.1: sqrt(softplus) scores (``scoring_func``) plus the correction bias, the top
  k, renormalised, times ``routed_scaling_factor``; V4's first ``num_hash_layers`` layers take their experts from a
  token-id table (``tid2eid``) instead, so the hooks pass the token ids along;
- Inkling 2 (oMLX's ``inkling2``, ``kernels.route_topk``): float32 logits, sigmoid, the top k by score plus
  ``router_bias``, weighted by the unbiased scores, renormalised, times ``route_scale``.

Anything else with a gate (``gate`` / ``router``) and a k is recorded as softmax over the experts, then the top k, and
marked approximate. A trace says which: ``meta.routing`` = {rule, k, exact, shared, num_experts}.

``router_probs`` are the weights the model mixes each chosen expert's output with, after its own renormalisation and
scaling, so they need not sum to one (DeepSeek V3: 2.5; Qwen2 MoE: under one). Shared experts (always on, unweighted
by the router) are not recorded, only counted in ``shared``; readers that want each expert's share of the routed mix
divide by the sum.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

# Where a decoder layer keeps its MoE block, and the block its router.
BLOCK_ATTRS = ("mlp", "block_sparse_moe", "feed_forward", "ffn", "moe")
GATE_ATTRS = ("gate", "router")
_K_ATTRS = ("top_k", "num_experts_per_tok", "k", "n_activated_experts")


def _softmax(x):
    return mx.softmax(x, axis=-1, precise=True)


_SCORE = {"softmax": _softmax, "sigmoid": mx.sigmoid, "sqrtsoftplus": lambda x: mx.sqrt(nn.softplus(x))}


@dataclass
class Rule:
    """One model's routing: ``select`` maps router logits [L, ..., E] (stacked over ``layers``) to the experts the
    model picks [L, ..., k] and the weights it mixes them with. ``logits(block, x)`` is the router's logits for the
    block's input x, computed as the model computes them."""
    name: str
    k: int
    num_experts: int
    logits: Callable[[nn.Module, mx.array], mx.array]
    exact: bool = True
    shared: int = 0                 # shared experts, always on
    score: str = "softmax"          # applied to every expert's logit before selection
    pick_first: bool = False        # top k of the raw logits, then softmax over the k (Mixtral, gpt-oss)
    upcast: bool = False            # the model scores in float32 (else in the logits' own dtype)
    high: bool = False              # the model writes argpartition(s, kth=-k)[..., -k:] (copied: ties fall the same way)
    groups: Optional[tuple] = None  # (n_group, topk_group, "max" | "top2"): group-limited selection
    weigh_masked: bool = False      # weights read from the group-masked scores (DeepSeek V2) rather than the scores
    renorm: bool = False
    eps: float = 0.0                # added to the renormalising sum
    scale: float = 1.0
    bias: Optional[Callable[[nn.Module], Optional[mx.array]]] = None   # block -> selection bias [E]
    table: Optional[Callable[[nn.Module], Optional[mx.array]]] = None  # block -> token-id table [V, k] (hash layers)
    blocks: dict = field(default_factory=dict)                          # layer -> MoE block (set by the adapter)
    _bias_cache: tuple = field(default=(None, None), repr=False)

    def meta(self) -> dict:
        return dict(rule=self.name, k=self.k, exact=self.exact, shared=self.shared, num_experts=self.num_experts)

    @property
    def needs_tokens(self) -> bool:
        return self.table is not None and any(self.table(b) is not None for b in self.blocks.values())

    def _top(self, s: mx.array) -> mx.array:
        k = self.k
        if self.high:
            return mx.argpartition(s, kth=-k, axis=-1)[..., -k:]
        return mx.argpartition(-s, kth=k - 1, axis=-1)[..., :k]

    def _stacked_bias(self, layers: list[int], ndim: int) -> Optional[mx.array]:
        found = [self.bias(self.blocks[l]) if l in self.blocks else None for l in layers]
        if all(b is None for b in found):
            return None
        key = (tuple(layers), tuple(id(b) for b in found), ndim)
        if self._bias_cache[0] == key:
            return self._bias_cache[1]
        zero = mx.zeros((self.num_experts,), dtype=mx.float32)
        B = mx.stack([zero if b is None else b for b in found])
        B = B.reshape(B.shape[0], *([1] * (ndim - 2)), B.shape[-1])
        self._bias_cache = (key, B)
        return B

    def select(self, G: mx.array, layers: list[int], tokens: Optional[mx.array] = None):
        """(ids int32 [L, ..., k], weights float32 [L, ..., k]) for logits G [L, ..., E]; ``tokens`` [..] are the
        token ids the step fed (hash layers only)."""
        x = G.astype(mx.float32) if self.upcast else G
        if self.pick_first:
            ids = self._top(x)
            w = _softmax(mx.take_along_axis(x, ids, axis=-1))
            return ids.astype(mx.int32), w.astype(mx.float32)
        s = _SCORE[self.score](x)
        key = s
        if self.bias is not None:
            B = self._stacked_bias(layers, s.ndim)
            if B is not None:
                key = s + B.astype(s.dtype)
        if self.groups is not None:
            n, keep, how = self.groups
            g = mx.unflatten(key, -1, (n, -1))
            gs = g.max(axis=-1, keepdims=True) if how == "max" else mx.topk(g, 2, axis=-1).sum(axis=-1, keepdims=True)
            drop = mx.argpartition(gs, kth=n - keep - 1, axis=-2)[..., :n - keep, :]
            key = mx.flatten(mx.put_along_axis(g, drop, mx.array(0.0, g.dtype), axis=-2), -2, -1)
        ids = self._top(key)
        if self.table is not None and tokens is not None:
            tabs = [self.table(self.blocks[l]) if l in self.blocks else None for l in layers]
            if any(t is not None for t in tabs):
                ids = mx.stack([ids[n] if t is None else t[tokens].astype(ids.dtype) for n, t in enumerate(tabs)])
        w = mx.take_along_axis(key if self.weigh_masked else s, ids, axis=-1)
        if self.renorm:
            w = w / (w.sum(axis=-1, keepdims=True) + self.eps) if self.eps else w / w.sum(axis=-1, keepdims=True)
        if self.scale != 1.0:
            w = w * self.scale
        return ids.astype(mx.int32), w.astype(mx.float32)


# -- logits as each family computes them ---------------------------------------

def _linear(attr: str):
    return lambda block, x: block[attr](x)


def _gate_layer(block, x):          # Granite: the router wraps its linear
    return block["router"]["layer"](x)


def _gate_f32(block, x):            # MiniMax: the gate reads float32 input
    return block["gate"](x.astype(mx.float32))


def _matmul(block, x):              # DeepSeek / GLM MoEGate: a bare weight
    return x @ block["gate"].weight.T


def _matmul_f32(temp: float):       # DeepSeek V4.1 (mlx_lm's Router, oMLX's Gate): float32, over the config's temperature
    return lambda block, x: (x.astype(mx.float32) @ block["gate"].weight.astype(mx.float32).T) / temp


def _router_weight(block, x):       # Inkling 2: as its route, float32 rows [B*L, H] through its model.py's _row_matmul
    flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    logits = sys.modules[_cls(block).__module__]._row_matmul(flat, block.router_weight.T)
    return logits.reshape(*x.shape[:-1], logits.shape[-1])


def _k(*mods) -> Optional[int]:
    for m in mods:
        for a in _K_ATTRS:
            v = getattr(m, a, None)
            if isinstance(v, int) and v > 0:
                return v
    return None


def _experts(gate) -> Optional[int]:
    if isinstance(gate, nn.Module) and "layer" in gate and not hasattr(gate, "weight"):
        gate = gate["layer"]
    w = getattr(gate, "weight", None)
    return int(w.shape[0]) if isinstance(w, mx.array) else None


def _n_shared(block) -> int:
    if "shared_expert" in block or "shared_mlp" in block:
        return 1
    if "shared_experts" in block:
        cfg = getattr(block, "config", None) or getattr(block, "args", None)
        n = getattr(cfg, "n_shared_experts", None)
        return int(n) if isinstance(n, int) and n > 0 else 1
    return 0


def _softmax_topk(name, block, renorm, high):
    return Rule(name, _k(block), _experts(block["gate"]), _linear("gate"), shared=_n_shared(block), renorm=renorm, high=high)


def _noaux(name, block, gate, eps: float = 0.0) -> Rule:
    """DeepSeek V3's group_expert_select (also V3.2, GLM-4 MoE; GLM-4 MoE lite adds 1e-20 to the sum)."""
    k = gate.top_k
    return Rule(name, k, _experts(gate), _matmul, shared=_n_shared(block), score="sigmoid", upcast=True,
                groups=(gate.n_group, gate.topk_group, "top2") if gate.n_group > 1 else None,
                renorm=k > 1 and bool(gate.norm_topk_prob), eps=eps, scale=float(gate.routed_scaling_factor),
                bias=lambda b: b["gate"].e_score_correction_bias)


def _cls(block) -> type:
    # A hooked module's class is a generated subclass (hooks.py), which keeps the original.
    return block.__dict__.get("_metalscope_orig_class", type(block))


def rule_for(block: nn.Module) -> Optional[Rule]:
    """The routing rule of one MoE block, or None if it has no router we can read."""
    cls = _cls(block)
    name, leaf = cls.__name__, cls.__module__.rsplit(".", 1)[-1]
    gate = block["gate"] if "gate" in block else None

    if name == "Qwen2MoeSparseMoeBlock":                     # mlx_lm qwen2_moe: never renormalised
        return _softmax_topk("qwen2_moe", block, renorm=False, high=False)
    if name in ("Qwen3MoeSparseMoeBlock", "Qwen3NextSparseMoeBlock"):
        return _softmax_topk("qwen3_moe" if name.startswith("Qwen3Moe") else "qwen3_next", block,
                             renorm=bool(block.norm_topk_prob), high=True)
    if name == "Qwen3_5MoeSparseMoeBlock":                   # mlx-vlm qwen3_5_moe (and qwen4_exp): always renormalised
        return _softmax_topk("qwen3_5_moe", block, renorm=True, high=True)
    if name == "OlmoeSparseMoeBlock":
        return _softmax_topk("olmoe", block, renorm=bool(block.norm_topk_prob), high=False)
    if name == "DeepseekMoE" and gate is not None:            # deepseek (V1)
        return Rule("deepseek", gate.top_k, _experts(gate), _matmul, shared=_n_shared(block))
    if name == "DeepseekV2MoE" and gate is not None:
        grouped = gate.topk_method == "group_limited_greedy"
        return Rule("deepseek_v2", gate.top_k, _experts(gate), _matmul, shared=_n_shared(block),
                    groups=(gate.n_group, gate.topk_group, "max") if grouped else None, weigh_masked=grouped,
                    scale=float(gate.routed_scaling_factor))
    if name in ("DeepseekV3MoE", "DeepseekV32MoE") and gate is not None:
        return _noaux("deepseek_v3" if name == "DeepseekV3MoE" else "deepseek_v32", block, gate)
    if name == "Glm4MoeLiteMoE" and gate is not None:
        return _noaux("glm4_moe_lite", block, gate, eps=1e-20)
    if name == "MoE" and leaf == "glm4_moe" and gate is not None:
        return _noaux("glm4_moe", block, gate)
    if name == "DeepseekV4MoE" and gate is not None and hasattr(gate, "scoring_func"):
        k = gate.top_k                                        # oMLX's deepseek_v4: _expert_select / _hash_expert_select
        return Rule("deepseek_v4", k, _experts(gate), _matmul, shared=_n_shared(block), score=gate.scoring_func,
                    upcast=True, renorm=gate.scoring_func != "softmax" and bool(gate.norm_topk_prob), eps=1e-20,
                    scale=float(gate.routed_scaling_factor),
                    bias=lambda b: getattr(b["gate"], "e_score_correction_bias", None),
                    table=lambda b: getattr(b["gate"], "tid2eid", None))
    if name == "MoE" and type(gate).__name__ == "Gate" and hasattr(getattr(gate, "_config", None), "n_activated_experts"):
        c = gate._config                                      # oMLX's own deepseek_v41 (what it serves): Gate(x, image_mask)
        k = c.n_activated_experts
        return Rule("deepseek_v41", k, _experts(gate), _matmul_f32(c.gate_temp), shared=_n_shared(block),
                    score=c.score_func, renorm=bool(c.norm_topk_prob) and k > 1, eps=1e-20,
                    scale=float(c.route_scale), bias=lambda b: b["gate"].bias)
    if name == "MoE" and leaf == "deepseek_v41" and gate is not None:     # mlx_lm's own V4.1 (oMLX's is above)
        k = gate.top_k
        return Rule("deepseek_v41", k, _experts(gate), _matmul_f32(gate.gate_temp), shared=_n_shared(block),
                    score=gate.score_func, renorm=bool(gate.norm_topk_prob) and k > 1, eps=1e-20,
                    scale=float(gate.route_scale), bias=lambda b: b["gate"].e_score_correction_bias)
    if name == "Inkling2MoE":                                 # oMLX's inkling2: route, then kernels.route_topk
        return Rule("inkling2", block.top_k, int(block.router_weight.shape[0]), _router_weight, score="sigmoid",
                    renorm=True, eps=1e-20, scale=float(block.route_scale), bias=lambda b: b.router_bias)
    if name == "MiniMaxSparseMoeBlock":
        return Rule("minimax", block.num_experts_per_tok, _experts(gate), _gate_f32, score="sigmoid",
                    renorm=True, eps=1e-20, bias=lambda b: b.e_score_correction_bias)
    if name in ("MixtralSparseMoeBlock", "PhiMoESparseMoeBlock"):
        return Rule("mixtral" if name.startswith("Mixtral") else "phimoe", _k(block), _experts(gate),
                    _linear("gate"), pick_first=True)
    if name == "MLPBlock" and leaf == "gpt_oss":
        return Rule("gpt_oss", block.num_experts_per_tok, _experts(block["router"]), _linear("router"),
                    pick_first=True, high=True)
    if name == "GraniteMoeMoE":
        r = block["router"]
        return Rule("granitemoe", r.top_k, _experts(r), _gate_layer, pick_first=True, high=True, upcast=True)
    return _fallback(block)


def _fallback(block: nn.Module) -> Optional[Rule]:
    """An MoE block we do not know: softmax over the experts, then the top k, marked approximate."""
    for attr in GATE_ATTRS:
        if attr not in block:
            continue
        gate = block[attr]
        k, E = _k(block, gate), _experts(gate)
        if not k or not E:
            return None
        if isinstance(gate, (nn.Linear, nn.QuantizedLinear)):
            fn = _linear(attr)
        elif isinstance(getattr(gate, "weight", None), mx.array):
            fn = lambda b, x, attr=attr: x @ b[attr].weight.T
        else:
            return None
        return Rule(f"unknown:{_cls(block).__name__}", k, E, fn, exact=False, shared=_n_shared(block))
    return None


def moe_block(layer: nn.Module) -> Optional[nn.Module]:
    """A decoder layer's MoE block (None for dense layers)."""
    for attr in BLOCK_ATTRS:
        if attr in layer and isinstance(layer[attr], nn.Module):
            m = layer[attr]
            if any(g in m and isinstance(m[g], nn.Module) for g in GATE_ATTRS):
                return m
    return None


def model_rule(layers) -> Optional[Rule]:
    """One rule for a model's MoE layers (``layers``: index -> decoder layer), with its blocks registered; None
    for a dense model."""
    rule = None
    for i, layer in layers:
        block = moe_block(layer)
        if block is None:
            continue
        if rule is None:
            rule = rule_for(block)
            if rule is None:
                return None
        rule.blocks[i] = block
    return rule

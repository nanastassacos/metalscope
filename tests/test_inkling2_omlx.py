"""Inkling 2's adapter on oMLX's own model code (omlx/patches/inkling2, 0.8.3 on; 0.8.4's fused kernels and LoopCD too):
a tiny random model with a loop, two full-attention layers, a sliding window shorter than the prompts and oMLX's Metal
kernels. Skipped unless omlx is
importable (oMLX's code and its MLX build ship only inside the app, so the dev venv skips this file: run it under the
app's Python):

    OMLX_APP=<the app> PYTHONPATH=.venv/lib/python3.11/site-packages scripts/omlx-python -m pytest -q \
        tests/test_inkling2_omlx.py
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map, tree_map_with_path

pytest.importorskip("omlx.patches.inkling2.cache")       # 0.8.3's cache layout

from omlx.patches.inkling2 import model as ink  # noqa: E402
from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed  # noqa: E402

from metalscope import adapters, vectors  # noqa: E402
from metalscope.adapters import inkling2  # noqa: E402
from metalscope.capture import CaptureConfig, CaptureSession  # noqa: E402
from metalscope.hooks import add_hook  # noqa: E402
from metalscope.mtp import window_starts  # noqa: E402

H, E, K, V = 32, 32, 4, 64
SHORT = mx.array([[3, 17, 5, 42, 9]])                                  # a decode or verify step: compiled halves
LONG = mx.array([[(7 * i + 3) % 60 for i in range(20)]])               # a prompt chunk: plain
DEEP = [(5 * i + 1) % 60 for i in range(40)]                            # a prefix past the sliding window, then 8
CHUNK = mx.array([DEEP])                                                # 40 tokens: past flash prefill's 32 queries


def tiny(hidden=H, head_dim=16, dtype=None):
    """8 layers in blocks of 2, layers 2-5 run 3 times (16 visits), layers 3 and 7 full attention, a window of 8. With
    ``dtype``, cast as a checkpoint's (``cast_predicate``: the mixes' queries and the router stay float32)."""
    args = ink.ModelArgs(model_type="inkling2_loop", quantization=dict(group_size=32, bits=4, mode="affine"),
                         text_config=dict(
                             hidden_size=hidden, num_hidden_layers=8, vocab_size=V, unpadded_vocab_size=60,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=head_dim, swa_num_attention_heads=2,
                             swa_num_key_value_heads=1, swa_head_dim=head_dim, sliding_window_size=8, d_rel=4, rel_extent=16,
                             log_scaling_n_floor=4, local_layer_ids=[0, 1, 2, 4, 5, 6], sconv_kernel_size=4,
                             n_layers_per_attn_res_block=2, n_routed_experts=E, num_experts_per_tok=K,
                             intermediate_size=32,
                             loop_config=dict(core_start=2, core_end=6, num_passes=3, gate_rank=4)))
    mx.random.seed(0)
    model = ink.Model(args)
    noisy = lambda p: p + 0.1 * mx.random.normal(p.shape) if mx.issubdtype(p.dtype, mx.floating) else p  # noqa: E731
    model.update(tree_map(noisy, model.parameters()))
    for layer in model.model.layers:                        # random experts, quantized as a converted checkpoint's
        for proj, shape in ((layer.mlp.experts.w13, (E, 64, hidden)), (layer.mlp.experts.w2, (E, hidden, 32))):
            proj.weight, proj.scales, proj.biases = mx.quantize(0.3 * mx.random.normal(shape), group_size=32, bits=4)
    if dtype is not None:
        cast = model.cast_predicate
        model.update(tree_map_with_path(
            lambda k, p: p.astype(dtype) if mx.issubdtype(p.dtype, mx.floating) and cast(k) else p, model.parameters()))
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def model():
    return tiny()


@pytest.fixture(scope="module")
def wide():
    """As served: bf16 activations and 128-wide heads, so 0.8.4's fused mix and norm and q/k/v's clips run
    (test_the_wide_model_reaches_0_8_4s_kernels). The NAX banded attention is reached only where its self-check
    passes (``banded_check_passed``); on the M5 Max this was run on it does not, so no test here covers the fit's NAX
    stub."""
    return tiny(256, 128, mx.bfloat16)


# The bf16 model, on 0.8.4's layout only (its fused kernels).
WIDE = pytest.param("wide", marks=pytest.mark.skipif(not hasattr(ink, "fused"), reason="0.8.4's fused kernels"))


@pytest.mark.skipif(not hasattr(ink, "fused"), reason="0.8.4's fused kernels")
def test_the_wide_model_reaches_0_8_4s_kernels(wide, monkeypatch):
    """What the bf16 model is for: its activations are bf16; a prompt chunk takes the fused mix and norm (the MoE's
    input and its float32 copy, which is the rounded input cast: the router reads the same value either way) and
    q/k/v's clips; with the kernels on a fit's plain paths but the NAX attention left on, a fit's batch of 8 rows of
    13 tokens reaches the NAX banded attention exactly where its self-check passes (``banded_check_passed``; not on
    the M5 Max this was run on, where this part holds with no NAX call)."""
    inner, hits = wide.model, {"fused": 0, "qkv": 0, "nax": 0}
    real = ink.kernels.attn_res_fused, ink.kernels.qkv_post, ink.kernels.nax_attention.nax_banded_sink_attention

    def count(name, fn):
        def run(*a, **k):
            out = fn(*a, **k)
            hits[name] += out is not None
            return out
        return run
    monkeypatch.setattr(ink.kernels, "attn_res_fused", count("fused", real[0]))
    monkeypatch.setattr(ink.kernels, "qkv_post", count("qkv", real[1]))
    monkeypatch.setattr(ink.kernels.nax_attention, "nax_banded_sink_attention", count("nax", real[2]))
    pairs = []
    real_post_attn = ink.Inkling2Layer._post_attn
    monkeypatch.setattr(ink.Inkling2Layer, "_post_attn",
                        lambda self, *a: pairs.append(real_post_attn(self, *a)) or pairs[-1])
    assert own(inner, LONG).dtype == mx.bfloat16                 # 20 tokens: the halves plain, the kernels as served
    assert hits["fused"] and hits["qkv"] and pairs
    for _, x, x32, _ in pairs:
        assert x.dtype == mx.bfloat16 and (x32 is None or mx.array_equal(x.astype(mx.float32), x32).item())
    hits["nax"] = 0
    with monkeypatch.context() as m:                             # a fit's paths, the NAX attention left on
        m.setattr(ink.kernels, "mx", inkling2._PlainMx())
        m.setattr(ink, "_COMPILED_MAX_TOKENS", 0)
        mx.eval(own(inner, mx.array([[(3 * i + 1) % 60 for i in range(13)]] * 8)))
    assert bool(hits["nax"]) == ink.kernels.nax_attention.banded_check_passed(mx.bfloat16)


class Tok:
    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]

    def decode(self, ids):
        return " ".join(map(str, ids))


def own(inner, ids, cache=None):
    """The text model's own call, as oMLX's model makes it (0.8.4: no defaults)."""
    return inner(ids, cache, None, {})


def traced(model, ids, cache=None, armed=False, **cfg):
    """One traced forward: its records, each layer call's (blocks, sum, bank) and each MoE call's (ids, weights), as
    ``kernels.route_topk`` returns them."""
    inner = model.model
    session = CaptureSession(CaptureConfig(**cfg))
    adapters.install(session, model)
    calls, routed, removers, real = [], [], [], ink.kernels.route_topk
    ink.kernels.route_topk = lambda *a: routed.append(real(*a)) or routed[-1]
    removers.append(lambda: setattr(ink.kernels, "route_topk", real))
    for layer in inner.layers:
        removers.append(add_hook(layer, lambda _m, a, k, out: calls.append((list(a[1]), out, k.get("bank")))))
    try:
        if armed:
            set_verify_qmm_armed(True, row_exact=True)
        logits = model(ids, cache=cache)
    finally:
        set_verify_qmm_armed(False)
        for r in removers:
            r()
    mx.eval(logits, session.pending_arrays())
    session.end_step()
    recs = {(r.kind, r.layer): r.value for r in session.drain()}
    session.close()
    return recs, calls, routed, logits


@pytest.mark.parametrize("which", ["model", WIDE])
@pytest.mark.parametrize("ids,armed", [(SHORT, False), (SHORT, True), (LONG, False), (CHUNK, False)],
                         ids=["decode-step", "row-exact-verify", "prompt-chunk", "long-chunk"])
def test_each_visit_is_recorded_with_its_experts_as_the_model_runs(request, which, ids, armed):
    """Every visit once at its index: the head's mix, the running sum's norm, the experts the router picked (its
    kernel's ids and weights); on a compiled step, on row-exact verify rows (``_post_exact``) and on prompt chunks;
    tracing leaves the logits as they were (bf16: to its rounding). On the bf16 model, through 0.8.4's fused kernels
    and the Metal router."""
    model = request.getfixturevalue(which)
    tol = 1e-4 if which == "model" else 5e-2
    inner, n = model.model, len(model.model._schedule)
    halves = lambda: [(l.__dict__["_post_compiled"], l.__dict__["_post_exact"]) for l in inner.layers]  # noqa: E731
    compiled = halves()
    plain = model(ids)
    if armed:
        set_verify_qmm_armed(True, row_exact=True)
        try:
            plain = model(ids)
        finally:
            set_verify_qmm_armed(False)
    recs, calls, routed, logits = traced(model, ids, armed=armed, router=True, logits_topk=3)
    assert adapters.layer_count(model) == n == 16 and len(calls) == len(routed) == n
    np.testing.assert_allclose(np.array(logits), np.array(plain), rtol=tol, atol=tol)
    for v, ((blocks, h, bank), (t_ids, t_w)) in enumerate(zip(calls, routed)):
        mix, _ = inkling2.state(inner, blocks, h, bank)
        np.testing.assert_allclose(recs[("resid_norm", v)], np.linalg.norm(np.array(mix.astype(mx.float32)), axis=-1),
                                   rtol=1e-4, err_msg=str(v))
        np.testing.assert_allclose(recs[("steer_norm", v)], np.linalg.norm(np.array(h.astype(mx.float32)), axis=-1),
                                   rtol=1e-4, err_msg=str(v))
        got_ids, got_w = recs[("router_ids", v)].reshape(-1, K), recs[("router_probs", v)].reshape(-1, K)
        want_ids, want_w = np.array(t_ids).astype(np.int64).reshape(-1, K), np.array(t_w).reshape(-1, K)
        o, t = np.argsort(got_ids, -1), np.argsort(want_ids, -1)
        np.testing.assert_array_equal(np.take_along_axis(got_ids, o, -1), np.take_along_axis(want_ids, t, -1), str(v))
        np.testing.assert_allclose(np.take_along_axis(got_w, o, -1), np.take_along_axis(want_w, t, -1), rtol=1e-5)
    assert halves() == compiled                                      # the session closed: the compiled halves back


def test_the_fit_forward_is_the_models_own(model, monkeypatch):
    """The fit's forward is the model's own loop: on the paths a fit takes (the kernels' plain ones, the halves
    uncompiled), the model's call gives the same values to the bit, from an empty cache and after a prefix past the
    sliding window (each visit's own cache, the full-attention visits' one tau at the cache's offset)."""
    inner = model.model
    prefix, window = mx.array([DEEP[:32]]), mx.array([DEEP[32:]])
    mine, cache = model.make_cache(), model.make_cache()
    own(inner, prefix, mine)
    own(inner, prefix, cache)
    with inkling2._fit_paths(ink):                               # the model's own call on the fit's paths
        want_long, want = np.array(own(inner, LONG)), np.array(own(inner, window, mine))
    with inkling2.differentiable(model) as (probes, forward):
        np.testing.assert_array_equal(np.array(ink._norm(inner.norm, forward(LONG))), want_long)
        assert sorted(probes.shapes) == list(range(16))
        np.testing.assert_array_equal(np.array(ink._norm(inner.norm, forward(window, cache))), want)
        assert not np.allclose(np.array(ink._norm(inner.norm, forward(window))), want, atol=1e-2)


@pytest.mark.parametrize("chunk,length", [(1024, 8), (10, 5)], ids=["new-buffer", "in-place"])
def test_each_pass_from_a_restored_prefix_cache_is_the_models_own(model, monkeypatch, chunk, length):
    """A fit runs a deep window's prefix once and restores its cache before each pass (``prefix_cache``, ``restore``):
    two passes, both evaluated after the second, give the model's own values from a freshly built prefix cache to the
    bit. The first pass takes up the growth buffers the prefix left (a sliding visit's slot is the snapshot's array),
    the second starts new ones from the snapshot. A prefix in chunks of 10 leaves room for 5 more keys, so the first
    pass writes them in place into a buffer the snapshot's arrays are views of: the snapshot keeps its values."""
    from metalscope.jlens.fit import prefix_cache, restore
    inner, b, prefix = model.model, 2, DEEP[:32]
    window = mx.array([DEEP[32:32 + length]] * b)
    mine = model.make_cache()
    for a in range(0, len(prefix), chunk):
        own(inner, mx.array([prefix[a:a + chunk]] * b), mine)
    with inkling2._fit_paths(ink):                               # the model's own call on the fit's paths
        want = np.array(own(inner, window, mine))
    cache, snap = prefix_cache(model, prefix, b, chunk)
    with inkling2.differentiable(model) as (_, forward):
        passes = []
        for _ in range(2):
            restore(cache, snap)
            passes.append(ink._norm(inner.norm, forward(window, cache)))
        mx.eval(passes)
    for got in passes:
        np.testing.assert_array_equal(np.array(got), want)


@pytest.mark.parametrize("which", ["model", WIDE])
def test_a_fit_takes_gradients_through_every_kernel(request, which):
    """A fit with a deep window (its prefix into a cache, restored before each pass): every Metal kernel takes its
    plain path (a kernel left on raises "[Primitive::vjp] Not implemented for CustomKernel"), and every visit's
    Jacobian has entries. The NAX stub is covered only where ``banded_check_passed`` is True (see ``wide``)."""
    from metalscope.jlens.fit import fit, no_index_gradients
    model = request.getfixturevalue(which)
    H = model.args.text.hidden_size
    no_index_gradients()
    lens = fit(model, Tok(), [dict(ids=DEEP, window=8), "5 9 2 7 1 3 8 4 6 2 9 9 1"], dim_batch=8, dim_limit=16,
               log=lambda *_: None)
    assert lens.layers == list(range(15))
    for v in lens.layers:
        J = np.array(lens.jacobians[v].astype(mx.float32))
        assert J.shape == (H, inkling2.lens_sources(dict(model.args.text_config))[v] * H)
        assert np.isfinite(J).all() and np.abs(J[:16]).max() > 0, v


def test_a_steer_lands_at_its_visit_and_the_engine_uses_the_models_cache(model):
    """The learn job's engine (the model's own make_cache): layer 2 runs at visits 2, 6 and 10; an add at 10 lands
    there alone; a greedy reply runs through the cache."""
    from metalscope.engines.mlx import MlxEngine
    engine = MlxEngine(model, Tok(), "inkling-tiny")
    ids = LONG[0].tolist()
    _, last = engine.residuals([ids], [2, 6, 10])
    add = 3.0 * vectors.unit(np.random.default_rng(4).standard_normal(H))
    _, steered = engine.residuals([ids], [2, 6, 10], adds={10: add})
    np.testing.assert_allclose(steered[0, :2], last[0, :2], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(steered[0, 2], last[0, 2] + add, rtol=1e-4, atol=1e-4)
    toks, _ = engine.generate(ids, None, 4)
    assert len(toks) <= 4


@pytest.mark.parametrize("length", [1, 3], ids=["decode-rows", "verify-rows"])
def test_the_router_rule_makes_the_logits_route_makes_on_batched_rows(model, monkeypatch, length):
    """Two batch rows, row-exact armed (MTP verify): the rule's logits are those ``route`` hands ``route_topk``, to the
    bit. ``route`` takes the rows flattened, so even a decode step's two rows are two single-row products."""
    from metalscope.routing import rule_for
    block, seen, real = model.model.layers[0].mlp, [], ink.kernels.route_topk
    monkeypatch.setattr(ink.kernels, "route_topk", lambda logits, *a: seen.append(logits) or real(logits, *a))
    x = mx.random.normal((2, length, H))
    set_verify_qmm_armed(True, row_exact=True)
    try:
        block(x)
        got = rule_for(block).logits(block, x)
        mx.eval(seen, got)
    finally:
        set_verify_qmm_armed(False)
    assert got.shape == (2, length, E)
    np.testing.assert_array_equal(np.array(got).reshape(seen[0].shape), np.array(seen[0]))


def test_mtp_windows_are_placed_by_the_first_full_attention_visit(model):
    cache = model.make_cache()
    model(SHORT, cache=cache)
    assert window_starts(cache, 5, 1) == [0]
    model(mx.array([[4, 2, 7]]), cache=cache)
    assert window_starts(cache, 3, 1) == [5]


@pytest.mark.skipif(not hasattr(ink.Model, "enable_loopcd"), reason="LoopCD is oMLX 0.8.4's")
def test_loopcds_calls_are_neither_recorded_nor_steered():
    """LoopCD on: the layers past the loop run again after the schedule (6 and 7 here), counted and recorded at no visit:
    every visit is recorded once with the experts the router picked, and the logits are LoopCD's own."""
    model = tiny()
    assert model.enable_loopcd(False)
    n, coda = len(model.model._schedule), len(model.model._reference_visits)
    plain = model(SHORT)
    recs, calls, routed, logits = traced(model, SHORT, router=True, logits_topk=3)
    assert adapters.layer_count(model) == n == 16 and len(calls) == len(routed) == n + coda == 18
    np.testing.assert_allclose(np.array(logits), np.array(plain), rtol=1e-4, atol=1e-4)
    assert sorted(v for k, v in recs if k == "router_ids") == list(range(n))
    assert sorted(v for k, v in recs if k == "resid_norm") == list(range(n))
    for v, (t_ids, _) in enumerate(routed[:n]):
        got = np.sort(recs[("router_ids", v)].reshape(-1, K), -1)
        np.testing.assert_array_equal(got, np.sort(np.array(t_ids).astype(np.int64).reshape(-1, K), -1), str(v))

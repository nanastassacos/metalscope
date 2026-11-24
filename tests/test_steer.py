"""Steering (steer.py): vectors, header, strength table, the hook, and the
server plumbing in live.py / omlx/patches.py, on tiny models."""
import json
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from metalscope import config, models, steer, vectors
from metalscope.live import TraceManager
from metalscope.store import Trace
from tests.helpers import tiny_llama
from tests.helpers import Tok, drain, wait_saved
from tests.helpers import D, TOKENS, manual_logits, rand_vec, run_middleware, write_norms


# ---------------------------------------------------------------- header, vectors, table

def test_parse_header_forms_and_errors():
    s = steer.parse_header("calm@0.5")
    p = s.parts[0]
    assert (p.name, p.strength, p.control, p.layers, s.positions) == ("calm", 0.5, False, None, "all")
    s = steer.parse_header(" random~calm@1 ; layers=24,31,24 ; positions=decode ")
    assert s.parts[0].control and s.parts[0].layers == (24, 31) and s.positions == "decode"
    assert s.label == "random~calm@1"
    bare = steer.parse_header("calm+angry@0.2")
    assert [(p.name, p.strength, p.layers) for p in bare.parts] == [("calm", None, None), ("angry", 0.2, None)]
    for bad in ("random~calm", "calm/3", "calm@x", "ca lm@1", "calm@-1", f"calm@{steer.MAX_STRENGTH + 1}", "calm@1;positions=last",
                "calm@1;layers=", "calm@1;foo=1", "calm@1;seed=3", "calm@1;space=lens", "calm@nan"):
        with pytest.raises(ValueError):
            steer.parse_header(bad)
    assert steer.stable_seed("abc") == steer.stable_seed("abc") != steer.stable_seed("abd")


def test_check_name_refuses_trailing_newline():
    import pytest
    with pytest.raises(ValueError):
        vectors.check_name("foo\n")
    assert vectors.check_name("foo.bar-2") == "foo.bar-2"


def test_save_vector_refuses_names_that_are_not_file_names(home):
    for bad in ("../escape", "a b", ".hidden", ""):
        with pytest.raises(ValueError):
            vectors.save_vector("tiny", vectors.Vector(bad, {0: vectors.unit(np.ones(4))}, {}))
    assert vectors.check_name("soft-texture.v2") == "soft-texture.v2"
    # Saving leaves no temp files behind.
    vectors.save_vector("tiny", vectors.Vector("ok", {0: vectors.unit(np.ones(4))}, {}))
    assert sorted(p.name for p in models.concept_dir("tiny", "ok").iterdir()) == ["vector.json", "vector.safetensors"]


def test_atomic_write_is_unique_per_writer_and_cleans_up(tmp_path):
    import threading
    target = tmp_path / "page.html"
    texts = [f"<html>{k}</html>" for k in range(8)]
    threads = [threading.Thread(target=config.atomic_write, args=(target, t)) for t in texts]
    for t in threads: t.start()
    for t in threads: t.join()
    assert target.read_text() in texts and [p.name for p in tmp_path.iterdir()] == ["page.html"]   # one of the writers won; no temp files remain


def test_vectors_roundtrip_and_listing(home):
    v = rand_vec("calm", [0, 2])
    v.vectors[2] = v.vectors[2] * 5  # not unit on disk: normalised on load
    vectors.save_vector("tiny", v)
    assert vectors.find_vector("tiny", "calm") is not None and vectors.find_vector("tiny", "nope") is None
    assert vectors.find_vector("tiny", "../calm") is None
    back = vectors.load_vector("tiny", "calm")
    assert back.layers == [0, 2] and back.width == D and back.hc_count == 1
    assert np.allclose(np.linalg.norm(back.vectors[2]), 1.0, atol=1e-5)
    assert np.allclose(back.vectors[0], v.vectors[0], atol=1e-6)
    assert back.meta["natural_norm"]["0"] == 3.0 or back.meta["natural_norm"][0] == 3.0
    assert [x["name"] for x in vectors.list_vectors("tiny")] == ["calm"]
    with pytest.raises(ValueError):
        vectors.load_vector("tiny", "nope")


def test_norm_table_from_traces_excludes_steered(home):
    tdir = config.traces_dir()
    entries = []
    for i, (val, label) in enumerate([(4.0, None), (6.0, None), (100.0, "calm@1")]):
        arrays = {f"resid_norm/{l}": np.full(7, val * (l + 1), np.float32) for l in range(2)}
        arrays["resid_norm/0"][0] = np.nan
        t = Trace(arrays, [1] * 7, ["x"] * 7, 3, dict(model="org/tiny"))
        p = t.save(tdir / f"t{i}")
        entries.append(dict(id=f"t{i}", path=str(p), time=i, model="org/tiny", steer=label))
    with open(config.index_path(), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in entries)
    table = steer.norm_table("tiny")
    assert table["layers"] == {0: 5.0, 1: 10.0} and table["hc_count"] == 1
    assert set(table["traces"]) == {"t0", "t1"}
    # Cached; a hyper-connection model's table uses the total stream norm.
    assert steer.norm_table("tiny")["layers"] == {0: 5.0, 1: 10.0}
    s = np.tile(np.array([[3.0, 4.0]], np.float32), (5, 1))             # per-token stream norms 3, 4 -> total 5
    t = Trace({"hc_stream_norm/0": s, "resid_norm/0": np.ones(5, np.float32)}, [1] * 5, ["x"] * 5, 2, dict(model="hyper"))
    p = t.save(tdir / "h0")
    with open(config.index_path(), "a") as f:
        f.write(json.dumps(dict(id="h0", path=str(p), time=9, model="hyper")) + "\n")
    th = steer.norm_table("hyper")
    assert th["layers"] == {0: 5.0} and th["streams"] == {0: [3.0, 4.0]} and th["hc_count"] == 2
    with pytest.raises(ValueError):
        steer.norm_table("unknown-model")


def _old_norm_table(paths):
    """norm_table's numbers as they were made before it read only the norm arrays' last rows: whole traces loaded."""
    totals, streams, tokens = {}, {}, 0
    for p in paths:
        t, n = Trace.load(p), 0
        kind = "hc_stream_norm" if "hc_stream_norm" in t.kinds() else "resid_norm"
        for l in t.layers(kind):
            a = np.asarray(t.get(kind, l), np.float32)
            if kind == "hc_stream_norm":
                a = a[np.isfinite(a).all(-1)]
                totals.setdefault(l, []).append(np.sqrt((a ** 2).sum(-1)))
                streams.setdefault(l, []).append(a)
            else:
                totals.setdefault(l, []).append(a[np.isfinite(a)])
            n = max(n, len(totals[l][-1]))
        tokens += n
    return dict(layers={l: float(np.median(np.concatenate(v))) for l, v in totals.items()},
                streams={l: np.median(np.concatenate(v), axis=0).tolist() for l, v in streams.items()}, tokens=tokens)


@pytest.mark.parametrize("kind", ["hc_stream_norm", "resid_norm"])
def test_norm_table_reads_the_last_rows_and_equals_the_whole_trace_reading_on_short_ones(home, monkeypatch, kind):
    """Traces of NORM_ROWS rows or fewer: the numbers loading whole traces gave. A longer one: its last NORM_ROWS rows
    alone are kept (its earlier rows, 1000 here, never reach the median)."""
    rng = np.random.default_rng(0)
    shape = lambda T: (T, 4) if kind == "hc_stream_norm" else (T,)
    paths, entries = [], []
    for i, T in enumerate([7, 30, 50, 12]):
        arrays = {f"{kind}/{l}": rng.uniform(1, 9, shape(T)).astype(np.float32) for l in (0, 3, 11)}
        arrays[f"{kind}/3"][:2] = np.nan                                          # rows a reused prefix left empty
        arrays["router_ids/0"] = np.zeros((T, 2), np.int32)                          # another kind: not read
        paths.append(Trace(arrays, [1] * T, ["x"] * T, 2, dict(model="tiny")).save(config.traces_dir() / f"s{i}"))
        entries.append(dict(id=f"s{i}", path=str(paths[-1]), time=i, model="tiny"))
    monkeypatch.setattr(steer, "NORM_ROWS", 50)
    table = steer.norm_table("tiny", entries=entries, refresh=True)
    old = _old_norm_table(paths)
    assert (table["layers"], table["streams"], table["tokens"]) == (old["layers"], old["streams"], old["tokens"])
    long = {f"{kind}/{l}": np.concatenate([np.full(shape(150), 1000, np.float32), np.ones(shape(50), np.float32)])
            for l in (0, 3, 11)}
    p = Trace(long, [1] * 200, ["x"] * 200, 2, dict(model="tiny")).save(config.traces_dir() / "long")
    table = steer.norm_table("tiny", entries=[dict(id="long", path=str(p), time=9, model="tiny")], refresh=True)
    assert table["tokens"] == 50 and set(table["layers"].values()) == {2.0 if kind == "hc_stream_norm" else 1.0}


def test_the_norm_table_prefers_the_steered_sums_norm_and_never_mixes_kinds(home):
    """Inkling 2's traces record ``steer_norm`` (the running sum a steer adds to) beside ``resid_norm`` (the head's
    mix): the table is the sum's. Older traces of the same copy without it are left out, not read in the mix's units;
    with none that has it, the residual's alone."""
    tdir, entries = config.traces_dir(), []

    def save(i, arrays):
        T = len(next(iter(arrays.values())))
        p = Trace(arrays, [1] * T, ["x"] * T, 2, dict(model="ink")).save(tdir / f"k{i}")
        entries.append(dict(id=f"k{i}", path=str(p), time=i, model="ink"))
    save(0, {"resid_norm/0": np.full(6, 38.0, np.float32)})                                # before steer_norm
    save(1, {"resid_norm/0": np.full(6, 38.0, np.float32), "steer_norm/0": np.full(6, 64.0, np.float32)})
    save(2, {"resid_norm/0": np.full(4, 30.0, np.float32), "steer_norm/0": np.full(4, 70.0, np.float32)})
    table = steer.norm_table("ink", entries=entries, refresh=True)
    assert table["kind"] == "steer_norm" and table["layers"] == {0: 64.0} and sorted(table["traces"]) == ["k1", "k2"]
    assert table["tokens"] == 10                                                           # k0's 6 rows not read
    old = steer.norm_table("ink", entries=entries[:1], refresh=True)
    assert old["kind"] == "resid_norm" and old["layers"] == {0: 38.0}


def test_a_push_spans_the_rest_of_its_block_for_a_vector_that_names_one(home):
    """A concept profiled on Inkling 2 (meta ``block``: 6 visits, aligned from 0): a push at 30 adds at 30-35, each
    visit its own direction at its own norm; a band given by hand adds at each layer once; a word's vector (no block)
    and every other model's push at the layers asked only."""
    assert steer.push_layers({"block": 6}, [30]) == [30, 31, 32, 33, 34, 35]
    assert steer.push_layers({"block": 6}, [33]) == [33, 34, 35]
    assert steer.push_layers({"block": 6}, range(30, 36)) == list(range(30, 36))          # no layer twice
    assert steer.push_layers({"block": 6}, [5, 30]) == [5, 30, 31, 32, 33, 34, 35]
    assert steer.push_layers({"token_ids": [5]}, [44]) == [44] and steer.push_layers({}, [3, 1]) == [1, 3]
    assert steer.push_layers(None, [2]) == [2]
    v = rand_vec("pirate", range(36))
    v.meta["block"] = 6
    vectors.save_vector("ink", v)
    vectors.save_vector("ink", rand_vec("plain", range(36), seed=2))
    norms = {l: 10.0 + l for l in range(36)}
    table = dict(layers=norms, traces=["t"])
    plan = steer.build_plan(steer.parse_header("pirate@0.5/30"), "ink", "r", table=table)
    assert sorted(plan.vectors) == list(range(30, 36)) and plan.meta["layers"] == list(range(30, 36))
    back = vectors.load_vector("ink", "pirate")
    for l in range(30, 36):
        np.testing.assert_allclose(plan.vectors[l], 0.5 * norms[l] * back.vectors[l], rtol=1e-5)
    band = steer.build_plan(steer.parse_header("pirate@0.5/30-35"), "ink", "r", table=table)
    assert sorted(band.vectors) == list(range(30, 36)) and set(band.meta["parts_at"].values()) == {1}
    for l in range(30, 36):
        np.testing.assert_allclose(band.vectors[l], plan.vectors[l], rtol=1e-6)
    assert sorted(steer.build_plan(steer.parse_header("plain@0.5/30"), "ink", "r", table=table).vectors) == [30]
    ctrl = steer.build_plan(steer.parse_header("random~pirate@0.5/30"), "ink", "r", table=table)
    assert sorted(ctrl.vectors) == list(range(30, 36))
    # The middleware's check asks for norms over the whole span.
    write_norms("ink", range(34))
    with pytest.raises(steer.NotReady, match="34, 35"):
        steer.check_spec(steer.parse_header("pirate@0.5/30"), "ink")
    steer.check_spec(steer.parse_header("plain@0.5/30"), "ink")


def test_build_plan_scales_and_controls_are_matched_and_seeded(home):
    v = rand_vec("calm", [0, 1], width=8)
    v.meta["hc_count"] = 2
    vectors.save_vector("tiny", v)
    write_norms("tiny", [0, 1], value=10.0, streams=[6.0, 8.0])
    spec = steer.parse_header("calm@0.5;layers=1")
    plan = steer.build_plan(spec, "tiny", "req-1", hc_count=2)
    assert plan.vectors.keys() == {1} and np.allclose(np.linalg.norm(plan.vectors[1]), 5.0, atol=1e-4)
    assert plan.meta["abs_norm"] == {1: 5.0} and plan.meta["seed"] is None and plan.positions == "all"
    with pytest.raises(ValueError):
        steer.build_plan(spec, "tiny", "req-1", hc_count=1)                 # made for another residual layout
    with pytest.raises(ValueError):
        steer.build_plan(steer.parse_header("calm@0.5;layers=5"), "tiny", "r", hc_count=2)
    # Control: same norm, per-stream profile of the named vector, seeded by request id.
    c1 = steer.build_plan(steer.parse_header("random~calm@0.5"), "tiny", "req-1", hc_count=2)
    c2 = steer.build_plan(steer.parse_header("random~calm@0.5"), "tiny", "req-1", hc_count=2)
    c3 = steer.build_plan(steer.parse_header("random~calm@0.5"), "tiny", "req-2", hc_count=2)
    for l in (0, 1):
        assert np.allclose(np.linalg.norm(c1.vectors[l]), 5.0, atol=1e-4)
        assert np.allclose(c1.vectors[l], c2.vectors[l]) and not np.allclose(c1.vectors[l], c3.vectors[l])
        named_profile = np.linalg.norm(v.vectors[l].reshape(2, -1), axis=-1)
        ctrl_profile = np.linalg.norm(c1.vectors[l].reshape(2, -1), axis=-1) / 5.0
        assert np.allclose(named_profile, ctrl_profile, atol=1e-4)
        assert abs(float(np.dot(vectors.unit(c1.vectors[l]), v.vectors[l]))) < 0.99
    assert c1.meta["seed"] == steer.stable_seed("req-1") and c1.meta["control"]
    # Layers are aligned across the request: one base direction, so each stream
    # block points the same way at every layer (only its scale follows the profile).
    for s in range(2):
        b0, b1 = c1.vectors[0].reshape(2, -1)[s], c1.vectors[1].reshape(2, -1)[s]
        assert np.allclose(vectors.unit(b0), vectors.unit(b1), atol=1e-5)
    # With a lens, both record the readout-space norm ||J_l v||.
    J = {l: mx.array(np.random.default_rng(l).standard_normal((3, 8)).astype(np.float32)) for l in (0, 1)}
    cl = steer.build_plan(steer.parse_header("random~calm@0.5"), "tiny", "req-1", hc_count=2, jacobians=J)
    assert all(cl.meta["readout_norm"][l] > 0 for l in (0, 1))


# ---------------------------------------------------------------- the hook

def session_on(model, plan, rows=(0,), n_rows=1, phase="decode", fails=None):
    s = steer.SteerSession(lambda rid, why: (fails if fails is not None else []).append((rid, why)))
    s.install(model)
    s.set_rows([(r, f"r{r}", plan) for r in rows], n_rows, phase)
    return s


def test_zero_strength_is_identity_and_matches_manual_add():
    model = tiny_llama()
    ref = model(TOKENS)
    v = rand_vec("x", [1]).vectors[1]
    zero = steer.SteerPlan({1: np.zeros(D, np.float32)}, "all", {})
    s = session_on(model, zero)
    assert mx.array_equal(model(TOKENS), ref)
    s.close()
    plan = steer.SteerPlan({1: 7.0 * v}, "all", {})
    s = session_on(model, plan)
    out = model(TOKENS)
    s.close()
    assert not mx.allclose(out, ref, atol=1e-4)
    np.testing.assert_allclose(np.array(out), np.array(manual_logits(model, TOKENS, {1: 7.0 * v})), rtol=1e-4, atol=1e-4)
    assert mx.array_equal(model(TOKENS), ref)                                # hooks gone


def test_rows_are_isolated_and_each_gets_its_own_vector():
    model = tiny_llama()
    batch = mx.array([[1, 5, 9, 2, 7], [3, 3, 8, 1, 4], [6, 2, 2, 9, 5]])
    ref = model(batch)
    va, vb = rand_vec("a", [0], seed=2).vectors[0] * 5, rand_vec("b", [0], seed=3).vectors[0] * 5
    s = steer.SteerSession(lambda *_: None)
    s.install(model)
    s.set_rows([(1, "ra", steer.SteerPlan({0: va}, "all", {})), (2, "rb", steer.SteerPlan({0: vb}, "all", {}))], 3, "decode")
    out = model(batch)
    s.close()
    assert mx.array_equal(out[0], ref[0])                                     # exactly untouched
    np.testing.assert_allclose(np.array(out[1]), np.array(manual_logits(model, batch[1:2], {0: va}))[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(out[2]), np.array(manual_logits(model, batch[2:3], {0: vb}))[0], rtol=1e-4, atol=1e-4)


def test_positions_decode_skips_prefill_phase():
    model = tiny_llama()
    ref = model(TOKENS)
    plan = steer.SteerPlan({1: 7.0 * rand_vec("x", [1]).vectors[1]}, "decode", {})
    s = session_on(model, plan, phase="prefill")
    assert mx.array_equal(model(TOKENS), ref)
    s.set_rows([(0, "r0", plan)], 1, "decode")
    assert not mx.allclose(model(TOKENS), ref, atol=1e-4)
    s.close()


def test_dtype_is_preserved_on_bf16_models():
    model = tiny_llama()
    model.set_dtype(mx.bfloat16)
    seen = []
    from metalscope.hooks import add_hook
    plan = steer.SteerPlan({1: 7.0 * rand_vec("x", [1]).vectors[1]}, "all", {})
    s = session_on(model, plan)
    rm = add_hook(model.model.layers[1], lambda m, a, k, out: seen.append(out.dtype))   # runs after the steer hook
    model(TOKENS)
    rm()
    s.close()
    assert seen == [mx.bfloat16]


def test_batch_mismatch_fails_closed():
    model = tiny_llama()
    ref = model(TOKENS)
    fails = []
    plan = steer.SteerPlan({1: 7.0 * rand_vec("x", [1]).vectors[1]}, "all", {})
    s = session_on(model, plan, rows=(0, 1), n_rows=2, fails=fails)            # rows say B=2, the forward has B=1
    out = model(TOKENS)
    assert mx.array_equal(out, ref) and {r for r, _ in fails} == {"r0", "r1"} and "does not match" in fails[0][1]
    assert not s._V                                                           # nothing more this forward
    s.close()
    fails.clear()
    wide = steer.SteerPlan({1: np.ones(D + 1, np.float32)}, "all", {})          # wrong residual width
    s = session_on(model, wide, fails=fails)
    assert mx.array_equal(model(TOKENS), ref) and fails
    s.close()
    fails.clear()
    s = steer.SteerSession(lambda rid, why: fails.append((rid, why)))
    s.set_rows([(3, "r3", plan)], 2, "decode")                                  # row outside the batch
    assert fails and not s._V


def test_deferred_write_tuple_is_steered_on_its_residual():
    class Layer(nn.Module):
        def __call__(self, x):
            return (x * 2, x + 1, x - 1)        # (streams, branch, gate) as Flash Next returns them

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer()]

    m = M()
    x = mx.ones((1, 3, 4))
    v = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    s = session_on(m, steer.SteerPlan({0: v}, "all", {}))
    out = m.layers[0](x)
    s.close()
    assert isinstance(out, tuple) and len(out) == 3
    np.testing.assert_allclose(np.array(out[0]), np.broadcast_to(2.0 + v[None, None], (1, 3, 4)))
    assert mx.array_equal(out[1], x + 1) and mx.array_equal(out[2], x - 1)


# ---------------------------------------------------------------- the manager and the patches

def install_vector(model_name="tiny", layers=(1,), seed=1):
    v = rand_vec("calm", list(layers), seed=seed)
    vectors.save_vector(model_name, v)
    write_norms(model_name, list(layers), value=10.0)
    return v


def run_request(mgr, tracer, model, rid, prompt, steps=2):
    mgr.rows_for(tracer, [rid], "prefill"); model(mx.array([prompt[:-1]])); mgr.clear_rows(tracer)
    outs = []
    fed = prompt[-1]
    for _ in range(steps):
        mgr.rows_for(tracer, [rid], "decode")
        logits = model(mx.array([[fed]]))
        mgr.clear_rows(tracer)
        outs.append(np.array(logits[0, -1]))
        fed = int(mx.argmax(logits[0, -1]).item())
    return outs


def test_steer_only_request_with_mode_off(home):
    install_vector()
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    assert mgr.mode() == "off" and not tracer.installed
    spec = steer.parse_header("calm@1.0")
    mgr.register("s1", None, {}, model=model, steer=spec)
    assert tracer.installed and mgr.requests["s1"].steer is not None and mgr.is_traced("s1") and mgr.kv_isolated("s1")
    plan = mgr.requests["s1"].steer
    assert plan.meta["abs_norm"] == {1: 10.0}
    steered = run_request(mgr, tracer, model, "s1", [1, 5, 9])
    mgr.finish("s1", [1, 5, 9], [2, 3], tracer)
    assert "s1" not in mgr.requests and mgr.kv_isolated("s1")                 # tombstoned for the cache store
    assert mgr.saved == 0                                                     # nothing recorded
    plain = run_request(mgr, tracer, model, "nobody", [1, 5, 9])
    assert not np.allclose(steered[0], plain[0], atol=1e-4)
    # Same forward, hook-free: the first decode step feeds the last prompt token without a cache.
    expect = manual_logits(model, mx.array([[9]]), {1: plan.vectors[1]})
    np.testing.assert_allclose(steered[0], np.array(expect)[0, -1], rtol=1e-4, atol=1e-4)
    tracer.uninstall()
    assert type(model.model.layers[1]).__name__ == "TransformerBlock" and tracer.steer is None


def test_traced_steered_request_records_steered_values_and_meta(home):
    install_vector()
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    mgr.register("plain", "on", dict(conversation="a", turn=1), model=model)
    run_request(mgr, tracer, model, "plain", [1, 5, 9])
    mgr.finish("plain", [1, 5, 9], [2, 3], tracer)
    mgr.register("steered", "on", dict(conversation="b", turn=1), model=model, steer=steer.parse_header("calm@1.0"))
    run_request(mgr, tracer, model, "steered", [1, 5, 9])
    mgr.finish("steered", [1, 5, 9], [2, 3], tracer)
    wait_saved(mgr, 2)
    index = {e["conversation"]: e for e in (json.loads(l) for l in config.index_path().read_text().splitlines())}
    assert index["a"]["steer"] is None and index["b"]["steer"] == "calm@1"
    a, b = Trace.load(index["a"]["path"]), Trace.load(index["b"]["path"])
    assert "steer" not in a.meta and b.meta["steer"]["label"] == "calm@1" and b.meta["steer"]["failed"] is None
    assert b.meta["steer"]["abs_norm"] == {"1": 10.0} or b.meta["steer"]["abs_norm"] == {1: 10.0}
    # On the positions fed the same tokens (prompt and the first decode input; greedy
    # decoding then diverges), residual norms differ from the steered layer on, not before it.
    np.testing.assert_allclose(a.get("resid_norm", 0)[:3], b.get("resid_norm", 0)[:3], rtol=1e-5)
    assert not np.allclose(a.get("resid_norm", 1)[:3], b.get("resid_norm", 1)[:3], rtol=1e-3)
    # The norm table never measures steered traces.
    from metalscope.store import read_index
    assert [e["id"] for e in read_index() if not e.get("steer")] == [index["a"]["id"]]


def test_hook_failure_marks_the_request_and_stops_steering(home):
    v = rand_vec("wide", [1], width=D + 8)                                       # wrong width for this model
    vectors.save_vector("tiny", v)
    write_norms("tiny", [1])
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    mgr.register("s1", None, {}, model=model, steer=steer.parse_header("wide@1.0"))
    assert mgr.requests["s1"].steer is not None
    out = run_request(mgr, tracer, model, "s1", [1, 5, 9])
    assert "does not match" in mgr.requests["s1"].steer_failed
    plain = run_request(mgr, tracer, model, "nobody", [1, 5, 9])
    np.testing.assert_allclose(out[1], plain[1], rtol=1e-5)                     # served unsteered after the failure
    mgr.finish("s1", [1, 5, 9], [2, 3], tracer)
    # A plan that cannot be built is refused at registration (oMLX answers 503), nothing left behind.
    with pytest.raises(RuntimeError, match="could not be built: vector wide has no layers"):
        mgr.register("s2", None, {}, model=model, steer=steer.parse_header("wide@1.0;layers=7"))
    assert "s2" not in mgr.requests


def test_steering_rows_that_cannot_be_built_fail_closed(home, monkeypatch):
    """Building the forward's steer rows raises (outside the hook): the request is marked steer_failed and served
    unsteered from there, as a hook failure is, and the forward's recording rows are still set."""
    install_vector()
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    mgr.register("s", "on", {}, model=model, steer=steer.parse_header("calm@1.0"))

    def broken(*_a, **_k):
        raise RuntimeError("rows bug")
    monkeypatch.setattr(tracer.steer, "set_rows", broken)
    mgr.rows_for(tracer, ["s"], "decode")                     # no exception reaches the scheduler
    assert mgr.requests["s"].steer_failed == "rows: rows bug"
    assert tracer.capture.owners == ["s"] and not tracer.steer._V
    out = run_request(mgr, tracer, model, "s", [1, 5, 9])
    plain = run_request(mgr, tracer, model, "nobody", [1, 5, 9])
    np.testing.assert_allclose(out[1], plain[1], rtol=1e-5)


def test_a_capture_hook_that_raises_marks_the_trace_partial(home, monkeypatch):
    from tests.helpers import wait_saved
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    # One more hook of the capture session's on a layer, the way the adapters add theirs: it raises every forward.
    def boom(*_):
        raise RuntimeError("capture bug")
    tracer.capture.hook(model.model.layers[1], boom)
    mgr.register("p", "on", {}, model=model)
    run_request(mgr, tracer, model, "p", [1, 5, 9])
    assert mgr.requests["p"].partial == ["RuntimeError: capture bug"]      # once, however many forwards
    mgr.finish("p", [1, 5, 9], [2, 3], tracer)
    wait_saved(mgr, 1)
    entry = json.loads(config.index_path().read_text().splitlines()[0])
    assert Trace.load(entry["path"]).meta["partial"] == ["RuntimeError: capture bug"]


def test_prefix_cache_guards_cover_store_and_fetch(home):
    from metalscope.omlx import patches as omlx_patches
    mgr = omlx_patches.MANAGER
    mgr.register("steered", None, {}, steer=steer.parse_header("calm@1.0"))
    mgr.register("traced", "on", {})

    class FakeCache:
        def fetch_cache(self, request_id, tokens):
            return "hit", []

        def store_cache(self, request_id, tokens, data, **kw):
            return "stored"

        def store_exact_prefix(self, request_id, tokens, data):       # oMLX's stores through store_cache
            return self.store_cache(request_id, tokens, data, hot_cache_write_back=False)

        def fetch_exact_prefix(self, request_id, tokens):
            return "hit"

    omlx_patches.patch_prefix_cache(SimpleNamespace(BlockAwarePrefixCache=FakeCache))
    c = FakeCache()
    assert c.fetch_cache("steered", [1]) == (None, [1]) and c.fetch_cache("traced", [1]) == (None, [1])
    assert c.fetch_cache("other", [1]) == ("hit", [])
    assert c.store_cache("steered", [1], None) is None and c.store_cache("traced", [1], None) == "stored"
    assert c.store_exact_prefix("steered", [1], None) is None and c.fetch_exact_prefix("steered", [1]) is None
    assert c.store_exact_prefix("traced", [1], None) == "stored" and c.fetch_exact_prefix("traced", [1]) == "hit"
    mgr.finish("steered", [], [])
    mgr.finish("traced", [], [])
    assert c.store_cache("steered", [1], None) is None                          # the async store worker runs later
    assert c.store_cache("traced", [1], None) == "stored"


def test_mtp_priming_and_reconcile_patches(home):
    from metalscope.omlx import patches as omlx_patches, patches_mtp
    mgr = omlx_patches.MANAGER
    mgr.register("steered", None, {}, steer=steer.parse_header("calm@1.0"))
    calls = []
    pp = SimpleNamespace(_publish_boundary_candidate=lambda ctx, candidate=None: calls.append(ctx.request_id) or "published")
    patches_mtp.patch_mtp_priming(pp)
    assert pp._publish_boundary_candidate(SimpleNamespace(request_id="steered")) is None
    assert pp._publish_boundary_candidate(SimpleNamespace(request_id="other")) == "published" and calls == ["other"]
    bg = SimpleNamespace(_reconcile_mtp_to_standard=lambda gen_batch, state: True)
    patches_mtp.patch_mtp_reconcile(bg)
    omlx_patches._current.scheduler = SimpleNamespace(uid_to_request_id={7: "steered"})
    try:
        assert bg._reconcile_mtp_to_standard(SimpleNamespace(uids=[7]), None) is True
    finally:
        omlx_patches._current.scheduler = None
    assert mgr.requests["steered"].steer_partial
    mgr.finish("steered", [], [])


def test_middleware_refuses_steering_until_enabled_then_echoes_it(home):
    install_vector()
    body = json.dumps({"model": "org/tiny", "messages": [{"role": "user", "content": "hi"}]}).encode()
    seen, sent = run_middleware({"x-scope-steer": "calm@0.5"}, body)
    assert sent[0]["status"] == 403 and "ctx" not in seen
    config.save_settings(steer=True)
    seen, sent = run_middleware({"x-scope-steer": "calm@0.5;positions=sideways"}, body)
    assert sent[0]["status"] == 400 and b"positions" in sent[1]["body"]
    seen, sent = run_middleware({"x-scope-steer": "nope@0.5"}, body)
    assert sent[0]["status"] == 400 and b"no steering vector" in sent[1]["body"]
    seen, sent = run_middleware({"x-scope-steer": "random~calm@0.5"}, body)
    assert sent[0]["status"] == 200 and seen["ctx"]["steer"].parts[0].control
    assert (b"x-scope-steer-accepted", b"random~calm@0.5") in sent[0]["headers"] and seen["body"] == body
    # A request the server refuses (oMLX's 503 when steering cannot be set up) is not marked accepted.
    seen, sent = run_middleware({"x-scope-steer": "calm@0.5"}, body, app_response=503)
    assert sent[0]["status"] == 503 and not any(k == b"x-scope-steer-accepted" for k, _ in sent[0]["headers"])
    # Without the header, and mode off, the request passes untouched.
    seen, sent = run_middleware({}, body)
    assert sent[0]["status"] == 200 and seen["ctx"] is None
    # A blank header is "no steer", not a 400: /replay's unsteered column sends one, and Pi's header
    # hook (which would drop it) does not run on extension-made model calls.
    seen, sent = run_middleware({"x-scope-steer": " "}, body)
    assert sent[0]["status"] == 200 and seen["ctx"] is None
    config.save_settings(steer=False)
    seen, sent = run_middleware({"x-scope-steer": ""}, body)
    assert sent[0]["status"] == 200 and seen["ctx"] is None


def test_a_part_given_twice_is_refused_by_name():
    """NAME alone has no dose yet: the message is built from the name, never a label (a TypeError, a 500)."""
    for header, name in (("calm+calm", "calm"), ("calm@0.3+calm", "calm"), ("calm+calm@0.3", "calm"),
                         ("random~calm@0.3+random~calm@0.5", "random~calm")):
        with pytest.raises(ValueError, match=f"^{name} appears twice"):
            steer.parse_header(header)
    assert len(steer.parse_header("calm@0.3+random~calm@0.3").parts) == 2         # a concept and its control: two parts


def test_middleware_refuses_a_steer_that_could_not_be_built(home):
    """No strength table yet (a new user's first steered message) or a layer the vector lacks: refused up front with
    the next step, not served unsteered under X-Scope-Steer-Accepted."""
    vectors.save_vector("tiny", rand_vec("calm", [1, 2]))
    config.save_settings(steer=True)
    body = json.dumps({"model": "org/tiny", "messages": [{"role": "user", "content": "hi"}]}).encode()
    seen, sent = run_middleware({"x-scope-steer": "calm@0.3"}, body)
    msg = json.loads(sent[1]["body"])["error"]["message"]
    assert sent[0]["status"] == 400 and "ctx" not in seen
    assert msg.startswith("metalscope cannot steer yet") and "without a steer first" in msg
    assert str(steer.NORM_MIN_TOKENS) in msg
    write_norms("tiny", [1, 2])
    seen, sent = run_middleware({"x-scope-steer": "calm@0.3/99"}, body)
    msg = json.loads(sent[1]["body"])["error"]["message"]
    assert sent[0]["status"] == 400 and "calm has no layer 99" in msg and "calm@0.3/1" in msg
    seen, sent = run_middleware({"x-scope-steer": "calm@0.3;layers=2,99"}, body)
    assert sent[0]["status"] == 400 and b"no layer 99" in sent[1]["body"]
    seen, sent = run_middleware({"x-scope-steer": "calm@0.3/2"}, body)
    assert sent[0]["status"] == 200 and seen["ctx"]["steer"].parts[0].layers == (2,)
    write_norms("tiny", [1])                                                  # a table without layer 2
    seen, sent = run_middleware({"x-scope-steer": "calm@0.3/2"}, body)
    assert sent[0]["status"] == 400 and b"no residual norm for calm at layer 2" in sent[1]["body"]


def test_a_dose_without_layers_steers_at_the_concepts_steer_layer(home):
    """NAME@DOSE with no layers goes where NAME alone does (the profile's reader layer), not at every layer its vector
    has: a learned concept's has them all, word salad at any dose. Its control goes there too. A vector with no
    profile (a word's, one made by hand) keeps its own layers; layers given are kept. The accepted header and the
    trace name the layer used."""
    from tests.helpers import fake_profile
    fake_profile(home)                                                        # toy: a vector at 0-2, read at 1
    vectors.save_vector("tiny", rand_vec("calm", [0, 2]))                     # no profile
    vectors.save_vector("tiny", rand_vec("flat", [0, 2]))
    models.profile_path("tiny", "flat").write_text(json.dumps(dict(concept="flat", reader={})))   # learn found no reader
    write_norms("tiny", [0, 1, 2])
    config.save_settings(steer=True)
    body = json.dumps({"model": "org/tiny", "messages": [{"role": "user", "content": "hi"}]}).encode()
    for header, layers, accepted in [("toy@0.4", [(1,)], "toy@0.4/1"),
                                     ("random~toy@0.4", [(1,)], "random~toy@0.4/1"),
                                     ("toy@0.4+random~toy@0.4", [(1,), (1,)], "toy@0.4/1+random~toy@0.4/1"),
                                     ("toy@0.4/0,2", [(0, 2)], "toy@0.4/0,2"),
                                     ("toy@0.4;layers=2", [(2,)], "toy@0.4/2"),
                                     ("calm@0.4", [None], "calm@0.4"),
                                     ("flat@0.4/2", [(2,)], "flat@0.4/2"),
                                     ("flat@0.4;layers=0", [(0,)], "flat@0.4/0"),
                                     ("calm@0.4/2+toy@0.2", [(2,), (1,)], "calm@0.4/2+toy@0.2/1")]:
        seen, sent = run_middleware({"x-scope-steer": header}, body)
        assert sent[0]["status"] == 200, (header, sent[1]["body"])
        assert [p.layers for p in seen["ctx"]["steer"].parts] == layers, header
        assert (b"x-scope-steer-accepted", accepted.encode()) in sent[0]["headers"], header
    spec = steer.with_defaults(steer.parse_header("toy@0.4+calm@0.2+random~toy@0.4"), "tiny")
    plan = steer.build_plan(spec, "tiny", "r")
    assert [p["layers"] for p in plan.meta["parts"]] == [[1], [0, 2], [1]] and plan.meta["layers"] == [0, 1, 2]
    with pytest.raises(ValueError, match=r"give toy@DOSE/LAYERS to pick layers"):
        steer.with_defaults(steer.parse_header("toy;layers=2"), "tiny")


def test_a_dose_on_a_concept_with_no_steer_layer_needs_its_layers(home):
    """A concept learn found no steer layer for ("can't be steered") has a profile and no reader: NAME@DOSE with no
    layers is refused (all its layers at once is word salad), on the header path and in an edit, not steered at
    every layer; NAME alone says the same, not "give a dose"; a profile that cannot be read counts as one. Layers
    given, or layers=, are kept; its control still covers the vector's layers."""
    vectors.save_vector("tiny", rand_vec("flat", [0, 2]))
    models.profile_path("tiny", "flat").write_text(json.dumps(dict(concept="flat", reader={})))   # learn found no reader
    vectors.save_vector("tiny", rand_vec("broke", [0, 2]))
    models.profile_path("tiny", "broke").write_text("{not json")
    write_norms("tiny", [0, 1, 2])
    config.save_settings(steer=True)
    want = r"flat has no steer layer on tiny: give its layers, e.g. flat@0.4/24"
    with pytest.raises(ValueError, match=want):
        steer.with_defaults(steer.parse_header("flat@0.4"), "tiny")
    with pytest.raises(ValueError, match=want):
        steer.parse_edit("3f2a@5 flat@0.4", "tiny")
    with pytest.raises(ValueError, match=r"flat has no steer layer on tiny: give its layers, e.g. flat@0.3/24"):
        steer.with_defaults(steer.parse_header("flat"), "tiny")
    with pytest.raises(ValueError, match=r"broke has no steer layer on tiny"):
        steer.with_defaults(steer.parse_header("broke@0.4"), "tiny")
    assert steer.with_defaults(steer.parse_header("random~flat@0.4"), "tiny").parts[0].layers is None
    body = json.dumps({"model": "org/tiny", "messages": [{"role": "user", "content": "hi"}]}).encode()
    seen, sent = run_middleware({"x-scope-steer": "flat@0.4"}, body)
    assert sent[0]["status"] == 400 and b"flat has no steer layer on tiny" in sent[1]["body"]
    assert steer.with_defaults(steer.parse_header("flat@0.4/2"), "tiny").parts[0].layers == (2,)
    assert steer.with_defaults(steer.parse_header("flat@0.4;layers=0"), "tiny").parts[0].layers == (0,)


def test_middleware_reads_a_body_that_is_not_an_object_as_no_metadata(home):
    """A JSON list or number is the engine's to answer (its own 400), not a 500 from conversation_meta."""
    config.save_settings(mode="on")
    for body in (b"[1, 2]", b"3", b"\"hi\"", b"not json"):
        seen, sent = run_middleware({}, body)
        assert sent[0]["status"] == 200 and seen["ctx"]["meta"] == {} and seen["body"] == body


def test_middleware_resolves_a_model_alias_before_steering(home):
    """oMLX lists aliases in /v1/models: the steer is looked up, and the request filed, under the served model."""
    install_vector()
    config.save_settings(steer=True)
    body = json.dumps({"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}).encode()
    seen, sent = run_middleware({"x-scope-steer": "calm@0.5"}, body)
    assert sent[0]["status"] == 400 and b"no steering vector" in sent[1]["body"]       # no resolver: the alias is unknown
    seen, sent = run_middleware({"x-scope-steer": "calm@0.5"}, body, resolve_model={"qwen": "tiny"}.get)
    assert sent[0]["status"] == 200 and seen["ctx"]["meta"]["model"] == "tiny"
    # A resolver that knows nothing, or raises, leaves the name as sent.
    seen, sent = run_middleware({"x-scope-trace": "on"}, body, resolve_model=lambda m: None)
    assert seen["ctx"]["meta"]["model"] == "qwen"

    def broken(m):
        raise KeyError(m)
    seen, sent = run_middleware({"x-scope-trace": "on"}, body, resolve_model=broken)
    assert sent[0]["status"] == 200 and seen["ctx"]["meta"]["model"] == "qwen"


def test_register_files_the_request_under_the_tracers_name(home):
    """The meta's model is the served copy's (the tracer's short name), whatever alias the client sent."""
    mgr = TraceManager()
    model = tiny_llama()
    mgr.attach(model, "/models/tiny", Tok())
    mgr.register("r1", "on", dict(model="qwen"), model=model)
    assert mgr.requests["r1"].meta["model"] == "tiny"
    mgr.release("r1")


def test_the_strength_table_is_read_again_when_its_file_changes(home):
    """After `metalscope vector norms X --refresh` the server's table follows the file check_spec reads, not the copy it
    read first (which would pass check_spec and fail the plan: a 503 on every request until a restart)."""
    import os
    vectors.save_vector("tiny", rand_vec("calm", [1, 2]))
    write_norms("tiny", [1])
    mgr = TraceManager()
    model = tiny_llama()
    mgr.attach(model, "tiny", Tok())
    mgr.register("a", None, {}, model=model, steer=steer.parse_header("calm@1.0/1"))
    assert mgr.strength_table("tiny")["layers"] == {1: 10.0}
    write_norms("tiny", [1, 2], value=20.0)
    later = models.norms_path("tiny").stat().st_mtime + 5
    os.utime(models.norms_path("tiny"), (later, later))                   # a distinct mtime on any filesystem
    mgr.register("b", None, {}, model=model, steer=steer.parse_header("calm@1.0/2"))
    assert mgr.requests["b"].steer.meta["abs_norm"] == {2: 20.0}
    assert mgr.strength_table("tiny") is mgr.strength_table("tiny")      # unchanged file: the table read stands
    for rid in ("a", "b"):
        mgr.release(rid)


def test_a_running_steered_request_is_never_swept_and_a_swept_traced_one_ends_live(home):
    """A steered request records nothing, so its forwards refresh ``touched``; a traced one that vanished gets its
    live end when swept."""
    import asyncio
    import time
    from metalscope import live
    install_vector()
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None)
    mgr.register("s1", None, {}, model=model, steer=steer.parse_header("calm@1.0"))
    mgr.register("gone", "on", dict(conversation="c", turn=1))
    old = time.time() - live.STALE_SECONDS - 60
    mgr.requests["s1"].touched = mgr.requests["gone"].touched = old
    mgr.rows_for(tracer, ["s1"], "decode"); mgr.clear_rows(tracer)
    assert mgr.requests["s1"].touched > old
    mgr.register("next", None, {})                                            # registering sweeps
    assert "s1" in mgr.requests and "gone" not in mgr.requests
    ends = [e for e in drain(loop, q) if e["type"] == "end"]
    assert [e["rid"] for e in ends] == ["gone"] and ends[0]["unfinished"]


# Two requests can reach norm_table together: the middleware checks a steer on a worker thread.
def test_two_first_steers_measure_the_table_once(home, monkeypatch):
    d = home / "traces" / "t1"
    d.mkdir(parents=True)
    np.savez(d / "arrays.npz", **{"resid_norm/0": np.full(8, 3.0, np.float32)})
    calls = []

    def picks(entries, short, n):
        calls.append(short)
        time.sleep(0.2)                        # long enough for the second caller to arrive
        return [dict(id="t1", path=str(d))]

    monkeypatch.setattr(steer, "norm_picks", picks)
    out = []
    threads = [threading.Thread(target=lambda: out.append(steer.norm_table("m", entries=[]))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert out[0]["made"] == out[1]["made"] and out[0]["layers"] == {0: 3.0}

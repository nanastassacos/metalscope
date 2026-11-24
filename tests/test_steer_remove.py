"""Removing a word (steer.py ``remove~NAME/LAYERS``): the header, the plan's rows, the projection on the written
residual in float32 and bf16 (pending writes too), its order with an add, and the refusals."""
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from metalscope import config, steer, vectors
from metalscope.live import TraceManager
from tests.helpers import D, TOKENS, Tok, rand_vec, run_middleware, tiny_llama


# ---------------------------------------------------------------- the header

def test_removal_header_label_round_trips():
    s = steer.parse_header("remove~a/3,7,9")
    p = s.parts[0]
    assert (p.name, p.strength, p.layers, p.remove, p.control) == ("a", None, (3, 7, 9), True, False)
    assert s.label == "remove~a/3,7,9" and steer.parse_header(s.label) == s
    s = steer.parse_header("remove~soccer/18-44+rugby@0.35/31")
    assert s.parts[0].layers == tuple(range(18, 45)) and s.label == "remove~soccer/18-44+rugby@0.35"
    assert steer.parse_header("remove~a/9,3-5,4").label == "remove~a/3-5,9"
    assert steer.parse_header("remove~a;layers=2-4").parts[0] == steer.SteerPart("a", None, layers=(2, 3, 4), remove=True)
    assert steer.parse_header("calm@0.3/24-26").parts[0].layers == (24, 25, 26)         # ranges for a push too
    assert steer.with_defaults(s, "tiny") == s                                           # no dose to look up


@pytest.mark.parametrize("bad, message", [
    ("remove~a", "needs its layers"),
    ("remove~a/", "at least one layer"),
    ("remove~a/44-18", "a range low-high"),
    ("remove~a/0-100000", "up to 1024"),
    ("remove~a/x", "bad layers"),
    ("remove~a@0.3/18-44", "no dose"),
    ("remove~../x/1", "remove~ takes NAME/LAYERS"),
    ("+".join(f"remove~v{i}/1" for i in range(steer.MAX_PARTS + 1)), "at most"),      # at most 8 directions a layer
])
def test_bad_removals_say_why(bad, message):
    with pytest.raises(ValueError, match=message):
        steer.parse_header(bad)


# ---------------------------------------------------------------- the plan

def test_a_removal_needs_no_strength_table_and_is_refused_cleanly(home):
    config.save_settings(steer=True)
    vectors.save_vector("tiny", rand_vec("soccer", [0, 1, 2]))
    vectors.save_vector("tiny", rand_vec("rugby", [1], seed=4))
    body = json.dumps({"model": "org/tiny", "messages": [{"role": "user", "content": "hi"}]}).encode()
    seen, sent = run_middleware({"x-scope-steer": "remove~soccer/0-2"}, body)          # no norms.json: still accepted
    assert sent[0]["status"] == 200 and (b"x-scope-steer-accepted", b"remove~soccer/0-2") in sent[0]["headers"]
    for header, message in (("remove~nope/1", b"no steering vector 'nope'"), ("remove~soccer/1-5", b"no layer 3, 4, 5")):
        seen, sent = run_middleware({"x-scope-steer": header}, body)
        assert sent[0]["status"] == 400 and message in sent[1]["body"], sent[1]["body"]
    seen, sent = run_middleware({"x-scope-steer": "remove~soccer/1+rugby@0.3/1"}, body)  # a push still needs one
    assert sent[0]["status"] == 400 and b"cannot steer yet" in sent[1]["body"]
    vectors.save_vector("tiny", vectors.Vector("pineapple", rand_vec("p", [1]).vectors, dict(hc_count=1, cut=["pineapple"])))
    seen, sent = run_middleware({"x-scope-steer": "remove~pineapple/1"}, body)           # refused before it is scheduled
    assert sent[0]["status"] == 400 and b"more than one token" in sent[1]["body"]


def test_plan_rows_span_the_removed_words_once(home):
    a = rand_vec("a", [0, 1], seed=2)
    vectors.save_vector("tiny", a)
    vectors.save_vector("tiny", vectors.Vector("twin", {0: a.vectors[0].copy()}, dict(hc_count=1)))
    other = rand_vec("n", [0], seed=5).vectors[0]
    near = vectors.unit(a.vectors[0] + 1e-2 * other)
    vectors.save_vector("tiny", vectors.Vector("near", {0: near}, dict(hc_count=1)))
    same = vectors.unit(a.vectors[0].astype(np.float64) + 1.4e-5 * other)                # cosine 1 - 1e-10: one direction
    assert steer.remove_rows([a.vectors[0], same]).shape == (1, D)
    plan = steer.build_plan(steer.parse_header("remove~a/0-1+remove~twin/0"), "tiny", "r", table={"layers": {}})
    assert plan.vectors == {} and sorted(plan.remove) == [0, 1]
    Q = np.array(plan.remove[0])
    assert Q.shape == (1, D) and Q.dtype == np.float32                                   # a twin is removed once
    np.testing.assert_allclose(abs(Q[0] @ a.vectors[0]), 1.0, atol=1e-6)
    Q = np.array(steer.build_plan(steer.parse_header("remove~a/0+remove~near/0"), "tiny", "r", table={"layers": {}}).remove[0])
    np.testing.assert_allclose(Q @ Q.T, np.eye(2), atol=1e-5)                            # orthonormal: no double subtraction
    m = plan.meta
    assert m["label"] == "remove~a/0-1+remove~twin/0" and m["removed"] == {0: 1, 1: 1} and m["layers"] == []
    assert [(p["name"], p["remove"], p["layers"], p["strength"]) for p in m["parts"]] == [("a", True, [0, 1], None), ("twin", True, [0], None)]
    json.dumps(m)                                                                        # saved with the trace
    with pytest.raises(ValueError, match="not finite"):
        steer.remove_rows([np.full(D, np.nan, np.float32)])
    for zero in ([np.zeros(D, np.float32)], [a.vectors[0], np.zeros(D, np.float32)]):     # alone, or beside a real one
        with pytest.raises(ValueError, match="zero"):
            steer.remove_rows(zero)
    vectors.save_vector("tiny", vectors.Vector("pineapple", {0: a.vectors[0]}, dict(hc_count=1, cut=["pineapple"])))
    with pytest.raises(ValueError, match="more than one token"):
        steer.build_plan(steer.parse_header("remove~pineapple/0"), "tiny", "r", table={"layers": {}})


# ---------------------------------------------------------------- the projection

class Fixed(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.out = out

    def __call__(self, x):
        return self.out


class Stack(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = [layer]


def apply(out, plans, B):
    """The one layer's output after the steer hook, for ``plans`` {row: SteerPlan}."""
    s = steer.SteerSession(lambda rid, why: pytest.fail(why))
    m = Stack(Fixed(out))
    s.install(m)
    s.set_rows([(r, f"r{r}", p) for r, p in plans.items()], B, "decode")
    got = m.layers[0](None)
    s.close()
    return got


def written(out):
    """streams + branch·gate (Flash Next's pending write), or the residual itself, in float64 [B, T, W]."""
    if not isinstance(out, tuple):
        return np.array(out.astype(mx.float32), np.float64).reshape(*out.shape[:2], -1)
    s, b, g = (np.array(x.astype(mx.float32), np.float64) for x in out)
    return s.reshape(*s.shape[:2], -1) + (b[..., None, :] * g[..., None]).reshape(*s.shape[:2], -1)


def dirs(W, k, seed=0):
    return [vectors.unit(np.random.default_rng(seed + i).standard_normal(W)) for i in range(k)]


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("dtype, pending", [(mx.float32, False), (mx.bfloat16, False), (mx.float32, True), (mx.bfloat16, True)])
@pytest.mark.parametrize("k, T", [(3, 3), (1, 1), (8, 64)])
def test_removal_zeroes_the_coordinates_and_keeps_the_rest(device, dtype, pending, k, T):
    """Flash Next's width (4 streams x 2560 = 10240), up to MAX_PARTS words. The coordinates on the removed words go to about 0 and the rest
    of the row moves by at most bf16 rounding (one cast back); the other batch row is untouched, bit for bit."""
    before = mx.default_device()
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    try:
        B, hc, d = 2, 4, 2560
        W = hc * d
        rng = np.random.default_rng(1)
        V = np.stack(dirs(W, k))                                                       # [k, W]
        streams = rng.standard_normal((B, T, hc, d)) + 8.0 * V[0].reshape(hc, d)
        if pending:
            out = tuple(mx.array(x.astype(np.float32)).astype(dtype) for x in
                        (streams, rng.standard_normal((B, T, d)), rng.standard_normal((B, T, hc))))
        else:
            out = mx.array(streams.astype(np.float32)).astype(dtype)
        plan = steer.SteerPlan({}, "all", {}, remove={0: steer.remove_rows(list(V))})
        got = apply(out, {1: plan}, B)
        h0, h1 = (o[0] if pending else o for o in (out, got))
        assert h1.dtype == dtype and h1.shape == h0.shape
        assert mx.array_equal(h1[0], h0[0])                                            # the other row
        if pending:
            assert mx.array_equal(got[1], out[1]) and mx.array_equal(got[2], out[2])
        w0, w1 = written(out)[1], written(got)[1]
        tol = 1e-4 if dtype == mx.float32 else 2 ** -8                                 # bf16: one rounding of each element
        assert np.abs(w0 @ V.T).max() > 4.0
        assert np.abs(w1 @ V.T).max() < (1e-4 if dtype == mx.float32 else 1e-2)         # elements about 1: bf16 leaves ~1e-3
        O = np.linalg.qr(V.T)[0].T                                                     # an orthonormal basis of span(V)
        rest = lambda w: w - (w @ O.T) @ O
        drift = np.linalg.norm(rest(w1) - rest(w0), axis=-1) / np.linalg.norm(w0, axis=-1)
        assert drift.max() < tol, drift
    finally:
        mx.set_default_device(before)


def test_removal_runs_before_the_add_at_its_layer():
    W = 16
    q, v = dirs(W, 2, seed=3)
    h = mx.array(np.random.default_rng(2).standard_normal((1, 4, W)).astype(np.float32))
    plan = steer.SteerPlan({0: 3.0 * v}, "all", {}, remove={0: steer.remove_rows([q])})
    w = np.array(apply(h, {0: plan}, 1), np.float64)[0]
    np.testing.assert_allclose(w @ q, np.full(4, 3.0 * float(v @ q)), atol=1e-5)        # removed, then pushed
    ref = np.array(h, np.float64)[0]
    np.testing.assert_allclose(w, ref - np.outer(ref @ q, q) + 3.0 * v, atol=1e-5)


def test_a_removal_on_a_real_model_and_its_trace_meta(home):
    """No strength table needed end to end; the plan's rows are built once, at register, and the trace meta lists the
    removal as a part."""
    vectors.save_vector("tiny", rand_vec("soccer", [0, 1]))
    mgr = TraceManager()
    model = tiny_llama()
    ref = np.array(model(TOKENS))
    mgr.attach(model, "tiny", Tok())
    mgr.register("s1", None, {}, model=model, steer=steer.parse_header("remove~soccer/0-1"))
    rt = mgr.requests["s1"]
    assert rt.steer_failed is None and sorted(rt.steer.remove) == [0, 1]
    tracer = next(iter(mgr.tracers.values()))
    mgr.rows_for(tracer, ["s1"], "prefill")
    out = np.array(model(TOKENS))
    mgr.clear_rows(tracer)
    assert not np.allclose(out, ref, atol=1e-4) and rt.steer_failed is None
    assert rt.steer.meta["parts"][0]["remove"] and rt.steer.meta["removed"] == {0: 1, 1: 1}


def test_a_removal_is_planned_on_ds4_as_on_omlx(home):
    """ds4 has no hooks here: register builds the same plan, and the proxy sends its rows in the steering file
    (tests/test_ds4.py)."""
    from metalscope.ds4.backend import Ds4Tracer
    vectors.save_vector("tiny-ds4", rand_vec("soccer", [0, 1]))
    mgr = TraceManager()
    t = Ds4Tracer("tiny-ds4", Tok(), "qwen4", 2, 1, D, 2, 4)
    mgr.tracers[id(t)] = t
    mgr.register("d1", None, {}, steer=steer.parse_header("remove~soccer/0-1"))
    rt = mgr.requests["d1"]
    assert rt.steer_failed is None and rt.steer.vectors == {} and rt.steer.meta["removed"] == {0: 1, 1: 1}

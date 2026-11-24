"""Steering combinations (steer.py): the header grammar, plans that sum parts,
the hook's constant add in a mixed batch, and steered live tokens, on tiny CPU
arrays."""
import asyncio
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from metalscope import config, steer, vectors
from metalscope.live import TraceManager
from tests.helpers import tiny_llama
from tests.helpers import Tok, wait_saved
from tests.helpers import TOKENS, manual_logits, rand_vec, run_middleware, write_norms


@pytest.fixture(autouse=True)
def cpu():
    """Exact float32 on the CPU: Metal's float32 matmul is off by ~1e-3 at
    these sizes, which would blur the comparisons the tests check."""
    before = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(before)


# ---------------------------------------------------------------- the header

def test_single_vector_forms_are_unchanged():
    s = steer.parse_header("calm@0.5;layers=24")
    p = s.parts[0]
    assert len(s.parts) == 1 and (p.name, p.strength, p.layers) == ("calm", 0.5, (24,))
    assert s.label == "calm@0.5" and s.positions == "all"
    assert steer.parse_header("calm@0.5/24").parts == s.parts             # the per-part form of the same request


def test_combination_parts_layers_and_controls():
    s = steer.parse_header("angry@0.4/24+robotic@0.8/20,24")
    assert [(p.name, p.strength, p.layers, p.control) for p in s.parts] == [("angry", 0.4, (24,), False), ("robotic", 0.8, (20, 24), False)]
    assert s.label == "angry@0.4+robotic@0.8"
    s = steer.parse_header("angry@0.4/24+robotic@0.8;layers=31")      # layers= fills only the parts that name none
    assert [p.layers for p in s.parts] == [(24,), (31,)]
    s = steer.parse_header("random~angry@0.4/24+random~robotic@0.8/24")
    assert all(p.control for p in s.parts) and s.label == "random~angry@0.4+random~robotic@0.8"
    assert [p.control for p in steer.parse_header("angry@0.4+random~angry@0.4").parts] == [False, True]


@pytest.mark.parametrize("bad, message", [
    ("calm@0.5+", "empty part"),
    ("+calm@0.5", "empty part"),
    ("", "empty part"),
    ("positions=decode", "NAME@DOSE"),                                    # options alone
    ("calm@0.5+random~robotic", "NAME@DOSE"),
    ("calm@0.5/", "at least one layer"),
    ("calm@0.5/x", "bad layers"),
    ("calm@0.5/-1", "none negative"),
    ("calm@0.5+calm@0.2", "appears twice"),
    ("calm@0.5:meter=0.6", "bad dose"),
    ("calm@0.5;meter=0.8", "unknown option 'meter'"),
    ("calm@0.5;foo=1", "unknown option 'foo'"),
    ("calm@0.5:dose=1", "bad dose"),
    ("calm@0.3/4;read=calm/4", "unknown option 'read'"),
    ("read=calm/4", "NAME@DOSE"),
    ("calm@0.5+robotic@9", "dose must be"),
    ("+".join(f"v{i}@0.1" for i in range(steer.MAX_PARTS + 1)), "at most"),
])
def test_bad_headers_say_why(bad, message):
    with pytest.raises(ValueError, match=message):
        steer.parse_header(bad)


# ---------------------------------------------------------------- plans

def test_combination_plan_sums_parts_and_lists_them(home):
    a, b = rand_vec("angry", [0, 1], seed=2), rand_vec("robotic", [1], seed=3)
    vectors.save_vector("tiny", a), vectors.save_vector("tiny", b)
    write_norms("tiny", [0, 1], value=10.0)
    plan = steer.build_plan(steer.parse_header("angry@0.4/1+robotic@0.8"), "tiny", "req")
    assert plan.vectors.keys() == {1}
    np.testing.assert_allclose(plan.vectors[1], 4.0 * a.vectors[1] + 8.0 * b.vectors[1], atol=1e-5)
    m = plan.meta
    assert m["label"] == "angry@0.4+robotic@0.8" and m["name"] == "angry+robotic" and m["strength"] is None
    assert [(p["name"], p["strength"], p["layers"]) for p in m["parts"]] == [("angry", 0.4, [1]), ("robotic", 0.8, [1])]
    assert m["abs_norm"] == {1: pytest.approx(12.0)} and m["parts_at"] == {1: 2} and "read" not in m
    json.dumps(m)                                                         # saved with the trace
    # Controls: one Gaussian base per part, so two control parts are not one direction twice.
    c = steer.build_plan(steer.parse_header("random~angry@0.4/1+random~robotic@0.4/1"), "tiny", "req")
    seeds = [p["seed"] for p in c.meta["parts"]]
    assert seeds == [steer.stable_seed("req"), steer.stable_seed("req") + 1] and c.meta["control"]
    d0 = steer.control_direction(a, [1], seeds[0])[1]
    d1 = steer.control_direction(b, [1], seeds[1])[1]
    assert abs(float(d0 @ d1)) < 0.5
    np.testing.assert_allclose(c.vectors[1], 4.0 * d0 + 4.0 * d1, atol=1e-5)


def test_middleware_refuses_a_bad_part_and_a_read_option(home):
    vectors.save_vector("tiny", rand_vec("calm", [1]))
    write_norms("tiny", [1])
    config.save_settings(steer=True)
    body = json.dumps({"model": "org/tiny", "messages": [{"role": "user", "content": "hi"}]}).encode()
    seen, sent = run_middleware({"x-scope-steer": "calm@0.5/1:meter=1"}, body)
    assert sent[0]["status"] == 400 and b"bad layers" in sent[1]["body"]
    seen, sent = run_middleware({"x-scope-steer": "calm@0.5/1+nope@0.2/1"}, body)
    assert sent[0]["status"] == 400 and b"no steering vector 'nope'" in sent[1]["body"]
    seen, sent = run_middleware({"x-scope-steer": "calm@0.5/1;read=calm/1"}, body)
    assert sent[0]["status"] == 400 and b"unknown option 'read'" in sent[1]["body"]
    seen, sent = run_middleware({"x-scope-steer": "calm@0.5/1"}, body)
    assert sent[0]["status"] == 200 and (b"x-scope-steer-accepted", b"calm@0.5/1") in sent[0]["headers"]


# ---------------------------------------------------------------- the hook

class Fixed(nn.Module):
    """A layer that returns whatever it is given."""
    def __call__(self, x):
        return x


class Stack(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def session(model, rows, n_rows, phase="decode"):
    s = steer.SteerSession(lambda rid, why: (_ for _ in ()).throw(AssertionError(why)))
    s.install(model)
    s.set_rows(rows, n_rows, phase)
    return s


def test_constant_combination_matches_the_summed_add():
    model = tiny_llama()
    va, vb = rand_vec("a", [1], seed=2).vectors[1], rand_vec("b", [1], seed=3).vectors[1]
    plan = steer.SteerPlan({1: 4.0 * va + 8.0 * vb}, "all", dict(parts_at={1: 2}))
    s = session(model, [(0, "r", plan)], 1)
    out = model(TOKENS)
    s.close()
    np.testing.assert_allclose(np.array(out), np.array(manual_logits(model, TOKENS, {1: 4.0 * va + 8.0 * vb})), rtol=1e-4, atol=1e-4)


def test_mixed_batch_constant_and_unsteered_rows():
    """Two rows steered at different layers and two unsteered rows in one batch."""
    W = 4
    e = np.eye(W, dtype=np.float32)
    m = Stack(Fixed(), Fixed(), Fixed())
    a = steer.SteerPlan({2: e[1]}, "all", {})                            # adds at 2
    c = steer.SteerPlan({1: e[3]}, "all", {})                            # adds at 1
    x = np.zeros((4, 2, W), np.float32)
    x[0, :, 0], x[1, :, 0] = [1.0, 2.0], [5.0, 6.0]
    s = session(m, [(0, "a", a), (2, "c", c)], 4)
    out = np.array(m(mx.array(x)))
    s.close()
    np.testing.assert_allclose(out[0, :, 1], [1.0, 1.0])
    np.testing.assert_allclose(out[0, :, 0], [1.0, 2.0])
    np.testing.assert_allclose(out[2, :, 3], [1.0, 1.0])
    assert not out[0, :, 3].any()
    assert np.array_equal(out[1], x[1]) and np.array_equal(out[3], x[3])


def test_mtp_reconcile_marks_steered_requests_partial(home):
    from types import SimpleNamespace
    from metalscope.omlx import patches as omlx_patches, patches_mtp
    mgr = omlx_patches.MANAGER
    mgr.register("steered", None, {}, steer=steer.parse_header("calm@1.0"))
    mgr.register("plain", "on", {})
    bg = SimpleNamespace(_reconcile_mtp_to_standard=lambda gen_batch, state: "reconciled")
    patches_mtp.patch_mtp_reconcile(bg)
    omlx_patches._current.scheduler = SimpleNamespace(uid_to_request_id={7: "steered", 8: "plain"})
    try:
        for outer in (False, True):                                       # outside and inside a row-set MTP next()
            mgr.requests["steered"].steer_partial = False
            omlx_patches._current.outer_rows = outer
            assert bg._reconcile_mtp_to_standard(SimpleNamespace(uids=[7, 8]), None) == "reconciled"
            assert mgr.requests["steered"].steer_partial and not getattr(mgr.requests["plain"], "steer_partial", False)
    finally:
        omlx_patches._current.scheduler = None
        omlx_patches._current.outer_rows = False
        mgr.finish("steered", [], [])
        mgr.finish("plain", [], [])


def test_steered_live_tokens_and_the_saved_label(home):
    vectors.save_vector("tiny", rand_vec("calm", [1]))
    write_norms("tiny", [0, 1], value=10.0)
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, "me")
    mgr.register("r1", "on", dict(client="me"), model=model, steer=steer.parse_header("calm@0.5/1"))
    prompt = [1, 5, 9]
    mgr.rows_for(tracer, ["r1"], "prefill"); model(mx.array([prompt[:-1]])); mgr.clear_rows(tracer)
    fed, outputs = prompt[-1], []
    for _ in range(4):
        mgr.rows_for(tracer, ["r1"], "decode")
        logits = model(mx.array([[fed]]))
        mgr.clear_rows(tracer)
        fed = int(mx.argmax(logits[0, -1]).item())
        outputs.append(fed)
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)
    loop.run_until_complete(asyncio.sleep(0))
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    tokens = [e for e in events if e["type"] == "token"]
    assert tokens and all("steer_read" not in t and "meters" not in t for t in tokens)
    assert json.loads(config.index_path().read_text().splitlines()[-1])["steer"] == "calm@0.5"

import json
from collections import defaultdict

import mlx.core as mx
import numpy as np
import pytest

from metalscope import config
from metalscope.live import READOUT_KINDS, TraceManager
from metalscope.middleware import conversation_meta
from metalscope.store import Trace
from tests.helpers import tiny_llama, tiny_moe
from tests.helpers import Tok, drain, wait_saved


def test_mode_off_installs_nothing(home):
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    assert mgr.mode() == "off" and not tracer.installed
    assert type(model.model.layers[0]).__name__ == "TransformerBlock"


def test_the_loaded_tracers_are_a_list_a_load_or_unload_meanwhile_leaves_alone(home):
    mgr = TraceManager()
    a = tiny_llama()
    ta = mgr.attach(a, "a", Tok())
    seen = []
    for t in mgr.loaded():                 # iterating the dict itself: "dictionary changed size during iteration"
        seen.append(t)
        mgr.attach(tiny_llama(), "b", Tok())
        mgr.detach(a)
    assert seen == [ta] and [t.name for t in mgr.loaded()] == ["b"]


def test_unloading_a_model_lets_go_of_the_requests_it_still_held(home):
    # oMLX aborts a request under memory pressure and unloads the model at once: the scheduler's own abort never
    # runs, and the request stayed "active" in the status until the next request swept it.
    mgr = TraceManager()
    a, b = tiny_llama(), tiny_llama()
    mgr.attach(a, "a", Tok())
    mgr.attach(b, "b", Tok())
    mgr.register("ra", "on", {}, model=a)
    mgr.register("rb", "on", {}, model=b)
    mgr.detach(a)
    assert list(mgr.requests) == ["rb"]
    assert mgr.release("ra") is False          # let go once: a later release does nothing


def test_the_middleware_reads_the_settings_installs_the_hooks_and_checks_a_steer_off_the_event_loop(home, monkeypatch):
    """Done on the event loop, a J-lens load (seconds) or a first strength table stalls every reply the server is
    streaming: the middleware does them on a worker thread, and the engine's register finds the hooks installed."""
    import threading
    from metalscope import middleware, steer
    from tests.helpers import run_middleware
    mgr = TraceManager()
    monkeypatch.setattr(middleware, "MANAGER", mgr)
    tracer = mgr.attach(tiny_llama(), "org/tiny", Tok())
    other = mgr.attach(tiny_llama(), "other", Tok())
    assert not tracer.installed                                              # tracing is off
    where, loop = {}, threading.current_thread()

    def spy(name, fn):
        return lambda *a: (where.setdefault(name, threading.current_thread()), fn(*a))[1]
    monkeypatch.setattr(mgr, "mode", spy("mode", mgr.mode))
    monkeypatch.setattr(tracer, "install", spy("install", tracer.install))
    monkeypatch.setattr(steer, "check_spec", spy("check", steer.check_spec))
    body = json.dumps({"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}).encode()
    seen, sent = run_middleware({"x-scope-trace": "on"}, body)
    assert sent[0]["status"] == 200 and tracer.installed and not other.installed
    config.save_settings(steer=True)
    seen, sent = run_middleware({"x-scope-steer": "calm@0.5"}, body)
    assert sent[0]["status"] == 400 and b"no steering vector" in sent[1]["body"]
    assert set(where) == {"mode", "install", "check"} and loop not in where.values()


def test_batched_requests_are_split_and_saved(home):
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    assert tracer.installed

    # Two traced requests (a and c) and one untraced (b) in one batch.
    mgr.register("req-a", "on", dict(conversation="c1", turn=1, last_user="hi"))
    mgr.register("req-c", "on", dict(conversation="c2", turn=1, last_user="yo"))
    prompts = {"req-a": [1, 5, 9], "req-c": [4, 4, 8]}

    # "Prefill" each traced request alone (B=1), as oMLX does.
    for rid, p in prompts.items():
        mgr.rows_for(tracer, [rid])
        model(mx.array([p[:-1]]))
        mgr.clear_rows(tracer)
    # Decode: rows [a, b, c]; the last prompt token is fed first.
    feed = {"req-a": [9, 11, 12], "req-c": [8, 21, 22]}
    for step in range(3):
        mgr.rows_for(tracer, ["req-a", "req-b", "req-c"])
        model(mx.array([[feed["req-a"][step]], [7], [feed["req-c"][step]]]))
        mgr.clear_rows(tracer)

    mgr.finish("req-a", [1, 5, 9], [11, 12, 13], tracer)
    mgr.finish("req-c", [4, 4, 8], [21, 22, 23], tracer)
    wait_saved(mgr, 2)

    index = [json.loads(l) for l in config.index_path().read_text().splitlines()]
    assert {e["conversation"] for e in index} == {"c1", "c2"}
    assert [e["experts"] for e in index] == [False, False]                  # a dense model
    by_conv = {e["conversation"]: Trace.load(e["path"]) for e in index}
    a, c = by_conv["c1"], by_conv["c2"]
    assert a.tokens == [1, 5, 9, 11, 12] and a.prompt_len == 3
    assert a.get("resid_norm", 0).shape == (5,)
    # The read-outs (the output's top k, both lenses) went out on the live stream; the trace keeps none.
    assert not (a.kinds() | c.kinds()) & READOUT_KINDS and a.meta["level"] == c.meta["level"] == "on"

    # Values match running request a on its own.
    solo = tiny_llama()
    h = solo.model.embed_tokens(mx.array([[1, 5, 9, 11, 12]]))
    for layer in solo.model.layers[:1]:
        h = layer(h, mask="causal", cache=None)
    # Prefill and decode here ran without a KV cache, so only layer-0 at
    # position 0..1 (no cross-position dependence beyond causal) is comparable.
    np.testing.assert_allclose(a.get("resid_norm", 0)[:2],
                               np.linalg.norm(np.array(h.astype(mx.float32))[0, :2], axis=-1), rtol=1e-4)
    assert not mgr.requests


def test_a_trace_missing_its_first_tokens_says_how_many(home):
    """Fewer records than fed tokens (a prefix served from a cache): the trace keeps the tokens it has records for,
    and meta.tokens_cut says how many before them it lacks (the lookup refuses such a trace); a whole one has none."""
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    for rid, prompt in (("req-cut", [1, 5, 9, 3]), ("req-whole", [2, 6])):
        mgr.register(rid, "on", dict(conversation=rid, turn=1, last_user="hi"))
        mgr.rows_for(tracer, [rid])
        model(mx.array([prompt[1:-1] if rid == "req-cut" else prompt[:-1]]))   # req-cut: its first token from a cache
        mgr.clear_rows(tracer)
        for t in (prompt[-1], 11, 12):
            mgr.rows_for(tracer, [rid])
            model(mx.array([[t]]))
            mgr.clear_rows(tracer)
        mgr.finish(rid, prompt, [11, 12, 13], tracer)
    wait_saved(mgr, 2)
    got = {e["conversation"]: Trace.load(e["path"]) for e in map(json.loads, config.index_path().read_text().splitlines())}
    assert got["req-cut"].tokens == [5, 9, 3, 11, 12] and got["req-cut"].meta["tokens_cut"] == 1
    assert got["req-whole"].tokens == [2, 6, 11, 12] and "tokens_cut" not in got["req-whole"].meta


def test_a_trace_saved_with_meters_and_read_outs_still_loads(home):
    """Traces saved before the meters were removed hold meter/<layer> arrays, meta.meter_names and a steer's
    reader (meta.steer.read); older ones the read-outs (logit_*, lens_*, jlens_* with their entropy), the watchlist's
    probe records and names, meta.vocab, and an index entry's readout flag: they load as any other, those keys read
    by nothing, and the strength table reads their norms."""
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    mgr.register("r1", "on", {})
    mgr.rows_for(tracer, ["r1"])
    model(mx.array([[1, 5, 9]]))
    mgr.clear_rows(tracer)
    mgr.finish("r1", [1, 5, 9], [11], tracer)
    wait_saved(mgr, 1)
    path = next((home / "traces").rglob("meta.json")).parent
    with np.load(path / "arrays.npz") as z:
        arrays = dict(z)
    old = {"meter/1": np.zeros((3, 2), np.float32), "logit_ids/-1": np.zeros((3, 10), np.int32),
           "logit_entropy/-1": np.ones(3, np.float32), "lens_ids/0": np.zeros((3, 10), np.int32),
           "lens_entropy/0": np.ones(3, np.float32), "jlens_ids/1": np.zeros((3, 10), np.int32),
           "jlens_entropy/1": np.ones(3, np.float32), "probe/1": np.zeros((3, 2), np.float32)}
    np.savez_compressed(path / "arrays.npz", **arrays, **old)
    meta = json.loads((path / "meta.json").read_text())
    meta.update(meter_names={"1": ["calm", "duty"]}, steer=dict(label="calm@0.3;read=calm/1", read=dict(name="calm", layer=1)),
                vocab={"ids": [0], "text": ["a"]}, probe_names={"1": ["harm", "code"]}, jlens_flags=[])
    (path / "meta.json").write_text(json.dumps(meta))
    index = config.index_path()
    index.write_text("".join(json.dumps(dict(json.loads(l), readout=True)) + "\n" for l in index.read_text().splitlines()))
    t = Trace.load(path)
    assert t.tokens == [1, 5, 9] and "meter" in t.kinds() and t.get("meter", 1).shape == (3, 2)
    assert {"lens_ids", "jlens_ids", "lens_entropy", "probe"} <= t.kinds()
    from metalscope import steer
    assert steer.norm_table("tiny")["layers"]


def test_conversation_meta_groups_turns():
    t1 = conversation_meta({"model": "m", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}]})
    t2 = conversation_meta({"model": "m", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"}, {"role": "user", "content": [{"type": "text", "text": "more"}]}]})
    assert t1["conversation"] == t2["conversation"]
    assert (t1["turn"], t2["turn"], t2["last_user"]) == (1, 2, "more")


def _assembled_as_before(parts, step_pos):
    """How finish and _save built a trace's arrays before live.assemble: a view per layer and step, concatenated (or
    scattered by position: mtp.scatter), right-aligned (_right_align), router weights cast to float16."""
    chunks = defaultdict(list)
    for key, entries in parts.items():
        for step, a, j, layers in entries:
            for n, layer in enumerate(layers or ()):
                chunks[f"{key.split('/')[0]}/{layer}"].append((step, a[n, j]))
            if layers is None:
                chunks[key].append((step, a[j]))
    blank = lambda a, n: np.full((n, *a.shape[1:]), -1 if a.dtype.kind == "i" else np.nan,
                                 a.dtype if a.dtype.kind == "i" else np.float32)
    if step_pos:
        n = max(step_pos[s] + len(a) for v in chunks.values() for s, a in v if s in step_pos)
        out = {}
        for key, v in chunks.items():
            v = sorted((p for p in v if p[0] in step_pos), key=lambda p: p[0])
            out[key] = buf = blank(v[0][1], n)
            for s, a in v:
                p = step_pos[s]
                buf[max(0, p):min(n, p + len(a))] = a[max(0, -p):min(n, p + len(a)) - p]
    else:
        out = {k: np.concatenate([c for _, c in sorted(v, key=lambda x: x[0])]) for k, v in chunks.items()}
        n = max(len(a) for a in out.values())
        out = {k: np.concatenate([blank(a, n - len(a)), a.astype(blank(a, 0).dtype)]) if len(a) < n else a
               for k, a in out.items()}
    return {k: v.astype(np.float16) if k.startswith("router_probs/") else v for k, v in out.items()}


@pytest.mark.parametrize("by_pos", [False, True])
def test_assembled_records_are_those_saved_before(by_pos):
    """live.assemble builds what the old per-layer views built, value for value and type for type: a prompt chunk's
    records a layer at a time, decode steps stacked over the layers in a batch of two, read-outs only from the first
    decode on (right-aligned, the rows before empty), routing weights as float16; filed by position, a later step
    overwrites an earlier one and a step with no position is left out."""
    from metalscope.live import assemble
    rng = np.random.default_rng(1)
    lay, parts = (0, 1, 2), defaultdict(list)
    for l in lay:
        parts["resid_norm/-2"].append((0, rng.random((1, 1, 5), np.float32), 0, (l,)))
        parts["router_ids/-2"].append((0, rng.integers(0, 8, (1, 1, 5, 2)).astype(np.int32), 0, (l,)))
        parts["router_probs/-2"].append((0, rng.random((1, 1, 5, 2), np.float32), 0, (l,)))
    parts["input_ids/-1"].append((0, rng.integers(0, 99, (1, 5)).astype(np.int32), 0, None))
    for s in range(1, 7):
        parts["resid_norm/-2"].append((s, rng.random((3, 2, 1), np.float32), 1, lay))
        parts["router_ids/-2"].append((s, rng.integers(0, 8, (3, 2, 1, 2)).astype(np.int32), 1, lay))
        parts["router_probs/-2"].append((s, rng.random((3, 2, 1, 2), np.float32), 1, lay))
        parts["input_ids/-1"].append((s, rng.integers(0, 99, (2, 1)).astype(np.int32), 1, None))
        parts["logit_ids/-1"].append((s, rng.integers(0, 99, (2, 1, 3)).astype(np.int32), 1, None))
        parts["logit_entropy/-1"].append((s, rng.random((2, 1), np.float32), 1, None))
        # A float16 record whose layer 1 starts late: that layer alone is float32, as before. One step after
        # another, a later step's wider ids widen the record (the concatenation did).
        parts["half/0"].append((s, rng.random((2, 1), np.float32).astype(np.float16), 1, None))
        parts["half/1"].append((s, rng.random((2, 1), np.float32).astype(np.float16), 1, None))
        if not by_pos:
            parts["wide/-1"].append((s, np.full((2, 1), s, np.int32 if s < 4 else np.int64), 1, None))
    parts["half/0"].append((0, rng.random((1, 5), np.float32).astype(np.float16), 0, None))
    step_pos = {0: 0, 1: 5, 2: 6, 3: 7, 4: 7, 5: 8} if by_pos else {}       # step 4 overwrites step 3; step 6 has none
    want = _assembled_as_before(parts, step_pos)
    got = assemble(parts, step_pos)
    if not by_pos:
        assert (got["half/0"].dtype, got["half/1"].dtype, got["wide/-1"].dtype) == (np.float16, np.float32, np.int64)
    assert not parts and sorted(got) == sorted(want) and got["logit_ids/-1"].shape == (9 if by_pos else 11, 3)
    for k, v in want.items():
        assert (got[k].dtype, got[k].shape) == (v.dtype, v.shape), k
        np.testing.assert_array_equal(got[k], v, err_msg=k)
    assert (got["logit_ids/-1"][:5] == -1).all() and np.isnan(got["logit_entropy/-1"][:5]).all()


def test_saving_a_long_trace_holds_its_records_about_once(home):
    """A reply decoded a token a step, finished and saved: the save peaks at about its records' size once more than
    they take in flight. Before, a view per layer and step plus two copies took over six times what was in flight."""
    import tracemalloc
    config.save_settings(mode="on")
    mgr, rng, L = TraceManager(), np.random.default_rng(0), 8
    mgr.register("r", "on", {})
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        for s in range(3000):
            T = 500 if s == 0 else 1
            mgr.file_host_step(None, "r", s, [
                ("input_ids", -1, rng.integers(0, 99, (1, T)).astype(np.int32), None),
                ("resid_norm", -1, rng.random((L, 1, T), np.float32), list(range(L))),
                ("router_ids", -1, rng.integers(0, 64, (L, 1, T, 4)).astype(np.int32), list(range(L))),
                ("router_probs", -1, rng.random((L, 1, T, 4), np.float32), list(range(L)))])
        held = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.reset_peak()
        mgr.finish("r", list(range(500)), list(range(3000)), None)
        wait_saved(mgr, 1, timeout=60)
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    assert peak < 1.6 * held, (peak, held)
    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    assert t.get("router_ids", 7).shape == (3499, 4) and t.get("router_probs", 0).dtype == np.float16


@pytest.mark.parametrize("mtp", [False, True])
def test_a_trace_stops_at_its_cap_and_keeps_its_first_tokens(home, monkeypatch, mtp):
    """Past TRACE_MAX_BYTES of records a reply keeps none (never an error in the forward): the live feed says so once
    and its tokens go on; the trace keeps the tokens up to the cap, each with the records an uncapped trace has
    there, and meta.capped says how many."""
    import asyncio
    from mlx_lm.models.cache import make_prompt_cache
    from metalscope import live
    config.save_settings(mode="on")
    prompt = list(range(1, 21))
    fed = prompt[-1:] + list(range(30, 59))

    def run(cap):
        monkeypatch.setattr(live, "TRACE_MAX_BYTES", cap)
        model = tiny_moe()
        model._omlx_mtp_decode_enabled = mtp
        mgr = TraceManager()
        tracer = mgr.attach(model, "tiny", Tok())
        loop = asyncio.new_event_loop()
        q = mgr.hub.subscribe(loop, "me")
        mgr.register("r", "on", dict(client="me"))
        cache = make_prompt_cache(model)
        for ids in [prompt[:-1]] + [fed[i:i + (3 if mtp else 1)] for i in range(0, len(fed), 3 if mtp else 1)]:
            mgr.rows_for(tracer, ["r"])
            model(mx.array([ids]), cache=cache)
            mgr.clear_rows(tracer)
        mgr.finish("r", prompt, fed[1:] + [99], tracer)
        wait_saved(mgr, 1)
        path = max((home / "traces").rglob("meta.json"), key=lambda p: p.stat().st_mtime_ns).parent
        return Trace.load(path), drain(loop, q)
    whole, _ = run(10 ** 9)
    # tiny_moe keeps 52 bytes a token (input_ids, two layers' resid_norm, resid_write_norm, router_ids and router_probs
    # at k 2)
    t, events = run(19 * 52 + 10 * 52)
    kept = 19 + (9 if mtp else 10)              # the prompt chunk and the decode steps that fit, whole windows under MTP
    assert t.meta["capped"] == kept and t.tokens == whole.tokens[:kept] and t.prompt_len == 20
    assert t.meta["completion"] == whole.meta["completion"] and t.meta["generated_tokens"] == 30
    for k, v in whole.arrays.items():
        np.testing.assert_array_equal(t.arrays[k], v[:kept], err_msg=k)
    assert [e["type"] for e in events].count("capped") == 1
    assert len([e for e in events if e["type"] == "token"]) == 30                # the live feed went on


def test_records_line_up_with_their_tokens_past_a_cached_prefix_the_stop_token_and_the_cap(home, monkeypatch):
    """One step after another, a reply whose prompt starts after a prefix served from a cache (6 tokens), with and
    without a pass past the stop token, under caps from a few tokens to none: every kept row's input id is the token
    the trace names there, tokens_cut is the prefix, and meta.capped (and its live event) comes only when the cap cut
    the conversation's own tokens, never for the pass past the stop alone. Before, a cached prefix with a pass past
    the stop was saved one row off (tokens_cut 5 for 6)."""
    import asyncio
    from mlx_lm.models.cache import make_prompt_cache
    from metalscope import live
    config.save_settings(mode="on")
    model = tiny_moe()
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, "me")
    prompt, out = list(range(1, 17)), list(range(40, 50))
    for cap in (None, 5, 12, 18, 24, 25):
        for cached in (0, 6):
            for extra in (0, 1):
                monkeypatch.setattr(live, "TRACE_MAX_BYTES", 10 ** 12 if cap is None else cap * 52)   # 52 B a token
                rid, saved = f"c{cap}-{cached}-{extra}", mgr.saved
                mgr.register(rid, "on", dict(client="me"))
                cache = make_prompt_cache(model)
                if cached:
                    model(mx.array([prompt[:cached]]), cache=cache)              # the prefix, from a cache: untraced
                for ids in [prompt[cached:]] + [[t] for t in out[:len(out) - 1 + extra]]:
                    mgr.rows_for(tracer, [rid])
                    model(mx.array([ids]), cache=cache)
                    mgr.clear_rows(tracer)
                mgr.finish(rid, prompt, out, tracer)
                real = len(prompt) - cached + len(out) - 1                       # the conversation's positions fed
                kept = real if cap is None else min(real, cap)
                if kept < len(prompt) - cached:                                  # not even the prompt chunk fit
                    assert mgr.saved == saved and not [e for e in drain(loop, q) if e["type"] == "capped"], rid
                    continue
                wait_saved(mgr, saved + 1)
                capped = [e for e in drain(loop, q) if e["type"] == "capped"]   # sent by the save
                t = Trace.load(max((home / "traces").rglob("meta.json"), key=lambda p: p.stat().st_mtime_ns).parent)
                np.testing.assert_array_equal(t.get("input_ids", -1), t.tokens, err_msg=rid)
                assert t.tokens == (prompt + out[:-1])[cached:cached + kept], rid
                assert t.meta.get("tokens_cut", 0) == cached and t.prompt_len == len(prompt) - cached, rid
                assert t.meta.get("capped") == (kept if kept < real else None), rid
                assert [e["tokens"] for e in capped] == ([kept - t.prompt_len] if kept < real else []), rid


@pytest.mark.parametrize("capped", [False, True])
def test_a_trace_filed_by_position_keeps_its_first_tokens_when_its_last_rows_are_missing(home, monkeypatch, capped):
    """Filed by position, row i is position i: a trace whose rows end before its tokens do (the cap's first step over
    it starts past them, or no step recorded the last positions) keeps the first tokens, never cuts the front. Before,
    it was saved shifted: tokens [2, 3, 4, 5] against ids [0, 1, 2, 3], tokens_cut 2."""
    from metalscope import live
    monkeypatch.setattr(live, "TRACE_MAX_BYTES", 1000)
    config.save_settings(mode="on")
    mgr = TraceManager()
    mgr.register("g", "on", {})
    mgr.file_host_step(None, "g", 0, [("input_ids", -1, np.arange(4, dtype=np.int32)[None], None),
                                      ("resid_norm", -1, np.ones((3, 1, 4), np.float32), [0, 1, 2])], pos=0)
    if capped:          # positions 4 and 5 never recorded, then a step from 6 past the cap
        mgr.file_host_step(None, "g", 1, [("input_ids", -1, np.arange(6, 8, dtype=np.int32)[None], None),
                                          ("resid_norm", -1, np.ones((3, 1, 200), np.float32), [0, 1, 2])], pos=6)
    mgr.finish("g", list(range(8)), [8, 9, 10], None)
    wait_saved(mgr, 1)
    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    assert t.tokens == [0, 1, 2, 3] and t.get("input_ids", -1).tolist() == [0, 1, 2, 3] and "tokens_cut" not in t.meta
    assert t.prompt_len == 4 and t.meta.get("capped") == (4 if capped else None)


def test_hosting_a_step_skips_a_list_a_release_empties_and_hosts_the_rest(home):
    """A release empties a parts list while _host reads that step: that entry is skipped, every other request's
    records of the step still go to host memory, and nothing is logged as a failure."""
    mgr, other = TraceManager(), []

    class Clears:                      # converting it, the release empties its list
        def __array__(self, dtype=None, copy=None):
            gone.clear()
            return np.zeros(1)
    gone = [(0, Clears(), 0, None)]
    other.append((0, mx.ones((1, 2)), 0, None))
    mgr._host([(gone, 0), (other, 0)])
    assert gone == [] and isinstance(other[0][1], np.ndarray) and not mgr._host_failed


def test_a_release_racing_a_step_that_adds_a_record_still_ends_the_reply_live(home):
    """_forget empties a copy of the parts' lists: a step adding a kind meanwhile cannot stop the release, so the live
    feed still gets the reply's end (else the dock shows it busy)."""
    import asyncio
    config.save_settings(mode="on")
    mgr = TraceManager()
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, "me")
    mgr.register("r", "on", dict(client="me"))
    rt = mgr.requests["r"]

    class Racing(list):
        def clear(self):
            rt.parts["new/-1"].append(1)      # close_step files a new kind while the release runs
            super().clear()
    rt.parts["input_ids/-1"] = Racing([(0, np.zeros((1, 1)), 0, None)])
    assert mgr.release("r")
    assert [e["type"] for e in drain(loop, q)][-1:] == ["end"]


def test_a_step_whose_filing_raises_still_hosts_the_step_before(home, monkeypatch):
    """Filing a step raises (the hook logs it, the forward goes on): the step before it is still copied to host
    memory, not left as MLX arrays for good."""
    from metalscope import live
    config.save_settings(mode="on")
    model = tiny_moe()
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny", Tok())
    mgr.register("r", "on", {})

    def step(ids):
        mgr.rows_for(tracer, ["r"])
        model(mx.array([ids]))
        mgr.clear_rows(tracer)
    step([1, 2])
    assert all(isinstance(e[1], mx.array) for v in mgr.requests["r"].parts.values() for e in v)
    monkeypatch.setattr(live, "_shared", lambda layers: 1 / 0)
    step([3])
    assert all(isinstance(e[1], np.ndarray) for v in mgr.requests["r"].parts.values() for e in v)


def test_a_step_whose_filing_raises_is_not_filed_again_with_the_next(home, monkeypatch):
    """A step whose filing raises part way leaves no records on the session: the next step files its own alone, so
    the rest of the trace stays lined up with its tokens."""
    from metalscope import live
    config.save_settings(mode="on")
    model = tiny_moe()
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny", Tok())
    mgr.register("r", "on", {})

    def step(ids):
        mgr.rows_for(tracer, ["r"])
        model(mx.array([ids]))
        mgr.clear_rows(tracer)
    step([1, 2])
    calls, shared = [0], live._shared

    def flaky(layers):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError("boom")
        return shared(layers)
    monkeypatch.setattr(live, "_shared", flaky)
    step([3])
    assert tracer.capture.records == []
    monkeypatch.setattr(live, "_shared", shared)
    step([4])
    rt = mgr.requests["r"]
    assert [s for s, *_ in rt.parts["input_ids/-1"]] == [0, 1, 2] and rt.step_tokens == [2, 1, 1]


def test_a_reply_whose_first_step_is_past_the_cap_says_so(home, monkeypatch, caplog):
    """Even the first step over the cap: nothing to save, and one line says why."""
    from metalscope import live
    monkeypatch.setattr(live, "TRACE_MAX_BYTES", 10)
    config.save_settings(mode="on")
    mgr = TraceManager()
    mgr.register("big", "on", {})
    mgr.file_host_step(None, "big", 0, [("input_ids", -1, np.arange(8, dtype=np.int32)[None], None)])
    mgr.finish("big", list(range(8)), [9], None)
    assert mgr.saved == 0
    assert [r.getMessage() for r in caplog.records if "trace cap" in r.getMessage()] == [
        "metalscope: big: its first step alone was past the trace cap (0.0 GB); nothing saved"]


@pytest.mark.parametrize("by_pos", [False, True])
def test_a_ds4_trace_stops_at_its_cap(home, monkeypatch, by_pos):
    """ds4's host steps stop at the cap too, filed by position or one after another: a 4-token prompt and 3 reply
    steps fit, the 4th step does not, and the trace keeps those 7 tokens."""
    from metalscope import live
    monkeypatch.setattr(live, "TRACE_MAX_BYTES", 7 * 8)
    config.save_settings(mode="on")
    mgr = TraceManager()
    mgr.register("d", "on", {})
    T = [4, 1, 1, 1, 1, 1]
    for s, t in enumerate(T):
        at = sum(T[:s])
        mgr.file_host_step(None, "d", s, [("input_ids", -1, np.arange(at, at + t, dtype=np.int32)[None], None),
                                          ("resid_norm", -1, np.ones((1, 1, t), np.float32), [0])],
                           pos=at if by_pos else None)
    mgr.finish("d", [0, 1, 2, 3], [4, 5, 6, 7, 8, 9], None)
    wait_saved(mgr, 1)
    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    assert t.tokens == list(range(7)) and t.meta["capped"] == 7 and t.prompt_len == 4
    np.testing.assert_array_equal(t.get("input_ids", -1), np.arange(7))


def test_live_stream_announces_generated_tokens(home):
    import asyncio
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, "me")
    other = mgr.hub.subscribe(loop, "someone-else")

    mgr.register("r1", "on", dict(client="me", conversation="c", turn=1))
    prompt = [1, 5, 9]
    mgr.rows_for(tracer, ["r1"]); model(mx.array([prompt[:-1]])); mgr.clear_rows(tracer)

    # Greedy decode: feed the last prompt token, then each argmax.
    fed, outputs, probs = prompt[-1], [], []
    for _ in range(4):
        mgr.rows_for(tracer, ["r1"])
        logits = model(mx.array([[fed]]))
        mgr.clear_rows(tracer)
        lp = logits[0, -1] - mx.logsumexp(logits[0, -1])
        fed = int(mx.argmax(lp).item())
        outputs.append(fed)
        probs.append(float(mx.exp(lp[fed]).item()))
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)

    async def drain(queue):
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return out
    loop.run_until_complete(asyncio.sleep(0))
    events = loop.run_until_complete(drain(q))
    assert loop.run_until_complete(drain(other)) == []          # filtered by client
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[1] == "prefill" and kinds[-1] == "end"
    tokens = [e for e in events if e["type"] == "token"]
    assert [t["id"] for t in tokens] == outputs                  # every generated token, in order
    for t, p in zip(tokens, probs):
        assert abs(t["p"] - p) < 1e-3 and f"<{t['id']}>" not in [a for a, _ in t["alts"]]
        # The output top 10 less the token written (greedy: always among them); the lens read-out too.
        assert len(t["alts"]) == 9 and t["entropy"] >= 0 and t["lens_by"] == "logit"
        assert t["n_layers"] == 3
    assert events[-1]["n_tokens"] == 4 and events[-1]["trace_id"].endswith("r1")


def test_prompt_tokens_fed_one_at_a_time_are_not_announced(home):
    """DeepSeek V4 prefills all but its prompt's last few tokens and feeds those one per forward
    (its chat template ends the prompt with <think>): only tokens past the prompt are the reply's."""
    import asyncio
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, "me")
    mgr.register("r1", "on", dict(client="me"))
    prompt = [1, 5, 9, 3, 7]
    mgr.note_prompt("r1", len(prompt))
    mgr.rows_for(tracer, ["r1"]); model(mx.array([prompt[:-2]])); mgr.clear_rows(tracer)
    fed, outputs = prompt[-2], []
    for step in range(5):
        mgr.rows_for(tracer, ["r1"])
        logits = model(mx.array([[fed]]))
        mgr.clear_rows(tracer)
        if step == 0:
            fed = prompt[-1]                     # still the prompt
            continue
        fed = int(mx.argmax(logits[0, -1]).item())
        outputs.append(fed)
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)
    loop.run_until_complete(asyncio.sleep(0))
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert [e["id"] for e in events if e["type"] == "token"] == outputs
    assert [e["i"] for e in events if e["type"] == "token"] == list(range(len(outputs)))
    # Each prefill event says how much of the prompt is read so far (Pi takes the largest: a copy changes nothing).
    assert [e["read"] for e in events if e["type"] == "prefill"] == [3, 4] and all("tokens" not in e for e in events if e["type"] == "prefill")


def test_a_stop_token_fed_once_more_is_not_announced(home):
    """GLM-4.5 (oMLX): the scheduler runs one more forward on the stop token it sampled
    (<|user|>), then ends the reply without it; the live stream must end there too."""
    import asyncio
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, "me")
    mgr.register("r1", "on", dict(client="me"))
    prompt, STOP = [1, 5, 9], 63
    mgr.note_prompt("r1", len(prompt), {STOP, 62})
    mgr.rows_for(tracer, ["r1"]); model(mx.array([prompt[:-1]])); mgr.clear_rows(tracer)
    fed, outputs = prompt[-1], []
    for step in range(4):                        # prompt end, g1, g2, then the stop token
        mgr.rows_for(tracer, ["r1"])
        logits = model(mx.array([[fed]]))
        mgr.clear_rows(tracer)
        fed = STOP if step == 2 else int(mx.argmax(logits[0, -1]).item()) % 60
        if step < 2:
            outputs.append(fed)
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)
    loop.run_until_complete(asyncio.sleep(0))
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert [e["id"] for e in events if e["type"] == "token"] == outputs


def test_scheduler_stop_ids_never_raise():
    from types import SimpleNamespace

    from metalscope.omlx.patches import stop_ids

    class Sched:
        _output_parser_factory = SimpleNamespace(stop_token_ids=[200002])

        def _get_stop_tokens(self):
            return {1, 2}
    req = SimpleNamespace(sampling_params=SimpleNamespace(stop_token_ids=[7]))
    assert stop_ids(Sched(), req) == {1, 2, 7, 200002}
    assert stop_ids(object(), None) == set()
    assert stop_ids(SimpleNamespace(_get_stop_tokens=lambda: 5), req) == {7}   # not iterable: skipped


def test_traced_tokens_carry_the_lens_top1_per_layer(home):
    import asyncio
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, "me")
    mgr.register("r1", "on", dict(client="me", conversation="c", turn=1))
    prompt = [1, 5, 9]
    mgr.rows_for(tracer, ["r1"]); model(mx.array([prompt[:-1]])); mgr.clear_rows(tracer)
    fed, outputs = prompt[-1], []
    for _ in range(3):
        mgr.rows_for(tracer, ["r1"])
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
    assert len(tokens) == 3
    for t in tokens:
        lens = t["lens"]
        layers = [layer for layer, _, _ in lens]
        assert layers == sorted(layers) and len(layers) == 3          # every layer of the tiny model, shallow to deep
        assert all(text == f"<{i}>" for _, i, text in lens)           # the id and its decoded text
        assert t["lens_by"] == "logit"                                  # no J-lens for the tiny model
        assert "settle" not in t                                        # Pi finds it in the lens


def test_mtp_windows_keep_only_emitted_positions(home):
    """Simulated MTP decode: verify windows with a rejected draft are filed by
    cache position, so the saved trace holds, at every position, the forward
    whose input there was the emitted token, and the live stream announces
    exactly the emitted tokens."""
    import asyncio
    from mlx_lm.models.cache import make_prompt_cache
    config.save_settings(mode="on")
    model = tiny_llama()
    model._omlx_mtp_decode_enabled = True
    prompt = list(range(1, 21))                      # > MAX_WINDOW: the prefill reads as a prefill
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny", Tok())
    assert tracer.mtp
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, "me")
    mgr.register("r1", "on", dict(client="me"))
    cache = make_prompt_cache(model)

    def fwd(ids):
        mgr.rows_for(tracer, ["r1"])
        out = model(mx.array([ids]), cache=cache)
        mgr.clear_rows(tracer)
        return [int(v) for v in mx.argmax(out[0], axis=-1).tolist()]

    fwd(prompt[:-1])                                  # prefill without the last prompt token
    g0 = fwd(prompt[-1:])[0]                          # first decode: the last prompt token
    w1 = fwd([g0, 5, 6])                              # draft 5 accepted, draft 6 rejected
    for c in cache:
        c.trim(1)
    w2 = fwd([w1[1], 7, 8])                           # bonus token + two accepted drafts
    outputs = [g0, 5, w1[1], 7, 8, w2[2]]
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)

    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    assert t.tokens == prompt + outputs[:-1]
    P = len(prompt)
    # Every position holds the forward whose input was the emitted token (the rejected draft's row overwritten): its
    # layer-0 residual norms are those of one causal pass over the kept tokens.
    solo = tiny_llama()
    norms = lambda ids: np.linalg.norm(np.array(solo.model.layers[0](solo.model.embed_tokens(mx.array([ids])), mask="causal",
                                                                     cache=None).astype(mx.float32))[0], axis=-1)
    want, drafted = norms(t.tokens), norms(t.tokens[:P + 2] + [6])[P + 2]                # 6: the rejected draft
    np.testing.assert_allclose(t.get("resid_norm", 0)[P - 1:], want[P - 1:], rtol=3e-3)   # the cached passes' rounding
    assert abs(drafted - want[P + 2]) > 10 * abs(t.get("resid_norm", 0)[P + 2] - want[P + 2])
    assert t.meta["step_tokens"][:4] == [19, 1, 3, 3]

    loop.run_until_complete(asyncio.sleep(0))
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    tokens = [e for e in events if e["type"] == "token"]
    assert [e["id"] for e in tokens] == outputs
    assert [e["i"] for e in tokens] == list(range(len(outputs)))
    assert not any(k.startswith("mtp") for e in tokens for k in e) and not any(k.startswith("mtp") for k in t.meta)


def test_mtp_live_reads_chunks_without_logits_or_past_max_window_as_prefill():
    from types import SimpleNamespace
    from metalscope.mtp import MAX_WINDOW, MtpLive

    class Streamer:
        def __init__(self):
            self.prefills = []

        def _prefill(self, rt, n):
            self.prefills.append(n)

        def _dist(self, refs, t):
            return t

    s, m, rt = Streamer(), MtpLive(), SimpleNamespace(prompt_len=0)
    ref = lambda n: (np.arange(n)[None], 0)
    m.step(s, rt, {"input_ids": ref(4), "start": 0}, None)                          # LM head skipped
    m.step(s, rt, {"input_ids": ref(MAX_WINDOW + 4), "logit_ids": 1, "start": 4}, None)
    assert s.prefills == [4, MAX_WINDOW + 4] and m.prompt_end == MAX_WINDOW + 8
    assert m.by_pos == {}
    end = m.prompt_end
    m.step(s, rt, {"input_ids": ref(3), "logit_ids": 1, "start": end - 1}, None)    # first decode + 2 drafts
    assert sorted(m.by_pos) == [end - 1, end, end + 1]


def test_live_stream_ends_when_server_shuts_down(home):
    import asyncio
    from metalscope import web

    class Req:
        async def is_disconnected(self):
            return False

    async def run():
        resp = await web.live(Req(), client="x")
        it = resp.body_iterator
        assert (await it.__anext__()).startswith(": connected")
        web.SHUTTING_DOWN.set()
        try:
            await asyncio.wait_for(it.__anext__(), timeout=3)
        except StopAsyncIteration:
            return True
        return False
    try:
        assert asyncio.run(run())
    finally:
        web.SHUTTING_DOWN.clear()


def test_a_live_stream_closed_before_it_starts_leaves_no_subscriber(home):
    """A response torn down before its first event (the client gone before it was sent) never ran its body: nothing
    may stay subscribed, or the per-token live work runs for nobody until a restart."""
    import asyncio
    from metalscope import web
    from metalscope.live import MANAGER

    class Req:
        async def is_disconnected(self):
            return False

    async def run():
        resp = await web.live(Req(), client="x", experts=True)
        await resp.body_iterator.aclose()
        return MANAGER.hub.active(), MANAGER.hub.experts
    assert asyncio.run(run()) == (False, False)


def test_the_live_route_passes_the_writes_opt_in_on(home):
    import asyncio
    from metalscope import web
    from metalscope.live import MANAGER

    class Req:
        async def is_disconnected(self):
            return False

    async def run():
        it = (await web.live(Req(), client="x", writes=True)).body_iterator
        await it.__anext__()                         # subscribed once the body runs
        on = MANAGER.hub.writes
        await it.aclose()
        return on, MANAGER.hub.writes
    assert asyncio.run(run()) == (True, False)


def _check_routing(tokens, trace, n_layers=2, k=2):
    """Each fed token's ``experts`` are its own position's routing, as saved;
    the last token was never fed, so it has none."""
    P = trace.prompt_len
    for i, e in enumerate(tokens[:-1]):
        assert e["expert_k"] == k and len(e["experts"]) == len(e["expert_w"]) == n_layers * k
        ids = np.array(e["experts"]).reshape(n_layers, k)
        for l in range(n_layers):
            assert ids[l].tolist() == trace.get("router_ids", l)[P + i].tolist()
            assert trace.get("router_probs", l).dtype == np.float16
        w = np.array(e["expert_w"]).reshape(n_layers, k)
        assert (np.abs(w.sum(-1) - 255) <= k).all()
    assert "experts" not in tokens[-1]


def test_live_tokens_carry_their_own_routing(home):
    import asyncio
    from tests.helpers import tiny_moe
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_moe()
    tracer = mgr.attach(model, "tiny-moe", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None, experts=True)
    plain = mgr.hub.subscribe(loop, None)            # e.g. Pi: never asked for routing
    mgr.register("r1", "on", dict(client="me"))
    prompt = [1, 5, 9]
    mgr.rows_for(tracer, ["r1"]); model(mx.array([prompt[:-1]])); mgr.clear_rows(tracer)
    fed, outputs = prompt[-1], []
    for _ in range(4):
        mgr.rows_for(tracer, ["r1"])
        logits = model(mx.array([[fed]]))
        mgr.clear_rows(tracer)
        fed = int(mx.argmax(logits[0, -1]).item())
        outputs.append(fed)
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)
    tokens = [e for e in drain(loop, q) if e["type"] == "token"]
    assert [e["id"] for e in tokens] == outputs
    saved = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    _check_routing(tokens, saved)
    # How the routing was recorded travels with the trace (routing.py): the tiny Qwen3 MoE's own k and rule.
    assert saved.meta["routing"] == dict(rule="qwen3_moe", k=2, exact=True, shared=0, num_experts=8)
    assert saved.meta["num_experts"] == 8
    assert len(json.dumps(tokens[0])) < 1000
    plain_tokens = [e for e in drain(loop, plain) if e["type"] == "token"]
    assert [e["id"] for e in plain_tokens] == outputs and not any("experts" in e for e in plain_tokens)
    mgr.hub.unsubscribe(q)
    assert not mgr.hub.experts                      # nobody wants routing: it is no longer computed


def test_live_mtp_tokens_carry_their_own_routing(home):
    """Under MTP each window row is a different token: row t's routing goes to
    the token fed at that row, and rejected drafts' routing is never announced."""
    import asyncio
    from mlx_lm.models.cache import make_prompt_cache
    from tests.helpers import tiny_moe
    config.save_settings(mode="on")
    model = tiny_moe()
    model._omlx_mtp_decode_enabled = True
    prompt = list(range(1, 21))
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny-moe", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None, experts=True)
    mgr.register("r1", "on", dict(client="me"))
    cache = make_prompt_cache(model)

    def fwd(ids):
        mgr.rows_for(tracer, ["r1"])
        out = model(mx.array([ids]), cache=cache)
        mgr.clear_rows(tracer)
        return [int(v) for v in mx.argmax(out[0], axis=-1).tolist()]

    fwd(prompt[:-1])
    g0 = fwd(prompt[-1:])[0]
    w1 = fwd([g0, 5, 6])                              # draft 6 rejected
    for c in cache:
        c.trim(1)
    w2 = fwd([w1[1], 7, 8])
    outputs = [g0, 5, w1[1], 7, 8, w2[2]]
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)
    tokens = [e for e in drain(loop, q) if e["type"] == "token"]
    assert [e["id"] for e in tokens] == outputs
    _check_routing(tokens, Trace.load(next((home / "traces").rglob("meta.json")).parent))


@pytest.mark.parametrize("tail", ["before_last", "with_last"])
def test_mtp_prompt_tail_with_logits_is_not_a_window(home, tail):
    """oMLX can prefill a short prompt tail (the chat template's
    ``<|im_start|>assistant ...``) as its own forward with logits. Knowing the
    prompt length, the live stream reads it as prompt, not as reply tokens;
    when the tail includes the last prompt token, its last row chooses the
    first reply token."""
    import asyncio
    from mlx_lm.models.cache import make_prompt_cache
    config.save_settings(mode="on")
    model = tiny_llama()
    model._omlx_mtp_decode_enabled = True
    prompt = list(range(1, 21))
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, "me")
    mgr.register("r1", "on", dict(client="me"))
    mgr.note_prompt("r1", len(prompt))
    cache = make_prompt_cache(model)

    def fwd(ids):
        mgr.rows_for(tracer, ["r1"])
        out = model(mx.array([ids]), cache=cache)
        mgr.clear_rows(tracer)
        return [int(v) for v in mx.argmax(out[0], axis=-1).tolist()]

    fwd(prompt[:12])                                  # <= MAX_WINDOW tokens, with logits
    if tail == "before_last":
        fwd(prompt[12:-1])
        g0 = fwd(prompt[-1:])[0]
    else:
        g0 = fwd(prompt[12:])[-1]
    w1 = fwd([g0, 5, 6])
    for c in cache:
        c.trim(1)
    w2 = fwd([w1[1], 7])
    outputs = [g0, 5, w1[1], 7, w2[1]]
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)
    tokens = [e for e in drain(loop, q) if e["type"] == "token"]
    assert [e["id"] for e in tokens] == outputs and [e["i"] for e in tokens] == list(range(5))
    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    assert t.tokens == prompt + outputs[:-1]


def test_mtp_reply_ending_on_a_correction_has_no_routing_for_it(home):
    """The last window's draft was rejected and the correction ends the reply:
    the correction was never fed, so it gets no routing."""
    import asyncio
    from mlx_lm.models.cache import make_prompt_cache
    from tests.helpers import tiny_moe
    config.save_settings(mode="on")
    model = tiny_moe()
    model._omlx_mtp_decode_enabled = True
    prompt = list(range(1, 21))
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny-moe", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None, experts=True)
    mgr.register("r1", "on", dict(client="me"))
    cache = make_prompt_cache(model)

    def fwd(ids):
        mgr.rows_for(tracer, ["r1"])
        out = model(mx.array([ids]), cache=cache)
        mgr.clear_rows(tracer)
        return [int(v) for v in mx.argmax(out[0], axis=-1).tolist()]

    fwd(prompt[:-1])
    g0 = fwd(prompt[-1:])[0]
    w1 = fwd([g0, 5, 6])
    corr = w1[1] if w1[1] != 6 else 7                  # draft 6 rejected
    mgr.finish("r1", prompt, [g0, 5, corr], tracer)
    wait_saved(mgr, 1)
    tokens = [e for e in drain(loop, q) if e["type"] == "token"]
    assert [e["id"] for e in tokens] == [g0, 5, corr]
    assert "experts" in tokens[1] and "experts" not in tokens[-1]


class Qwen35Verifier:
    """mlx-vlm's Qwen3_5BatchInvariantForward as oMLX runs a Qwen3.5-family MTP verify window (its ``_model``): every
    decoder layer computed by ``_layer``, none called, so no hook on a layer runs unless patch_mtp_verify wraps it."""

    def _layer(self, layer, hidden, mask, cache):
        h = hidden + layer.self_attn(layer.input_layernorm(hidden), mask, cache)
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def forward(self, model, ids, cache):
        from mlx_lm.models.base import create_attention_mask
        h = model.model.embed_tokens(ids)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(model.model.layers, cache):
            h = self._layer(layer, h, mask, c)
        return model.lm_head(model.model.norm(h))


def verifying_llama():
    """The tiny llama, its windows (``verify=True``) through the verifier inside the top-level call, as oMLX's
    LanguageModel does with ``speculative_verify``: the call's own hooks (the output, the step's end) still run."""
    from mlx_lm.models import llama

    class VerifyingLlama(llama.Model):
        def __call__(self, inputs, cache=None, verify=False):
            return Qwen35Verifier().forward(self, inputs, cache) if verify else super().__call__(inputs, cache)
    model = tiny_llama()
    model.__class__ = VerifyingLlama
    model._omlx_mtp_decode_enabled = True
    return model


def test_mtp_verify_windows_that_never_call_the_layers_are_traced_and_steered(home, monkeypatch):
    """Qwen3.8 27B (qwen3_5, dense) with MTP: oMLX ran its prompt and first decode through the layers and every
    verify window after through the verifier, so the live feed had logit-lens rows on 1 token of 24 and the saved
    trace residual norms up to the first decode only; steering missed every window. patch_mtp_verify runs each
    layer's hooks on what the verifier computed for it."""
    import asyncio
    from mlx_lm.models.cache import make_prompt_cache
    from metalscope.omlx import patches, patches_mtp
    from metalscope.steer import SteerPlan, SteerSession
    monkeypatch.setattr(patches, "APPLIED", set())
    patches_mtp.patch_mtp_verify(Qwen35Verifier)
    config.save_settings(mode="on")
    model = verifying_llama()
    prompt = list(range(1, 21))
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny", Tok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, "me")
    mgr.register("r1", "on", dict(client="me"))
    cache = make_prompt_cache(model)

    def fwd(ids, verify=True):
        mgr.rows_for(tracer, ["r1"])
        out = model(mx.array([ids]), cache=cache, verify=verify)
        mgr.clear_rows(tracer)
        return [int(v) for v in mx.argmax(out[0], axis=-1).tolist()]

    fwd(prompt[:-1], verify=False)                    # prefill and the first decode: through the layers
    g0 = fwd(prompt[-1:], verify=False)[0]
    w1 = fwd([g0, 5, 6])                              # verify windows: draft 6 rejected
    for c in cache:
        c.trim(1)
    w2 = fwd([w1[1], 7, 8])
    outputs = [g0, 5, w1[1], 7, 8, w2[2]]
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)
    tokens = [e for e in drain(loop, q) if e["type"] == "token"]
    assert [e["id"] for e in tokens] == outputs
    assert all([layer for layer, _, _ in e["lens"]] == [0, 1, 2] for e in tokens)      # logit-lens rows on every token
    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    solo = tiny_llama()
    want = np.linalg.norm(np.array(solo.model.layers[0](solo.model.embed_tokens(mx.array([t.tokens])), mask="causal",
                                                       cache=None).astype(mx.float32))[0], axis=-1)
    np.testing.assert_allclose(t.get("resid_norm", 0), want, rtol=3e-3)                 # every position, windows too
    assert t.meta["step_tokens"] == [19, 1, 3, 3]
    assert t.meta["fed_rows"] == 26

    # Steering: a window's adds are the plain forward's.
    steer = SteerSession(lambda rid, why: pytest.fail(why))
    steer.install(model)
    plan = SteerPlan({1: np.full(64, 0.5, np.float32)}, "all", {})
    ids = mx.array([prompt[:3]])
    run = lambda verify: np.array(model(ids, cache=make_prompt_cache(model), verify=verify).astype(mx.float32))
    plain = run(False)
    steer.set_rows([(0, "s", plan)], 1, "decode")
    steered = run(False)
    np.testing.assert_allclose(run(True), steered, rtol=1e-4, atol=1e-4)
    assert np.abs(steered - plain).max() > 1e-2
    steer.close()


@pytest.mark.parametrize("how", ["release", "sweep"])
def test_a_reply_let_go_unfinished_holds_none_of_its_records(home, how):
    """A traced reply cancelled (release) or gone stale (_sweep) lets every record go at once, its last step's too,
    which the tracer's ``fresh`` still lists until the model's next forward (up to 1.5 GB for a long one)."""
    import gc
    import weakref
    from metalscope import live
    config.save_settings(mode="on")
    model = tiny_moe()
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny", Tok())
    mgr.register("r", "on", {})
    for ids in ([1, 2, 3], [4], [5]):
        mgr.rows_for(tracer, ["r"])
        model(mx.array([ids]))
        mgr.clear_rows(tracer)
    held = [weakref.ref(e[1]) for v in mgr.requests["r"].parts.values() for e in v]
    assert tracer.fresh and len(held) == 3 * 5          # input_ids and four stacked records a step
    if how == "release":
        assert mgr.release("r")
    else:
        mgr.requests["r"].touched -= 2 * live.STALE_SECONDS
        mgr._sweep()
    gc.collect()
    assert "r" not in mgr.requests and not any(ref() is not None for ref in held)
    mgr.register("s", "on", {})                       # the next forward reads the emptied step and goes on
    mgr.rows_for(tracer, ["s"])
    model(mx.array([[6]]))
    mgr.clear_rows(tracer)
    mgr.rows_for(tracer, ["s"])
    model(mx.array([[7]]))
    mgr.clear_rows(tracer)
    assert all(isinstance(e[1], np.ndarray) for e in mgr.requests["s"].parts["input_ids/-1"][:1])


def test_a_record_that_cannot_be_copied_to_host_memory_never_breaks_the_next_step(home, monkeypatch, caplog):
    """Copying the last step's records to host memory raises: that is logged once, those records stay MLX arrays
    (saved all the same), and every step is filed in full, its own records never mixed into the next."""
    from metalscope import live
    config.save_settings(mode="on")
    model = tiny_moe()
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny", Tok())
    mgr.register("r", "on", {})
    class NoHost:
        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def array(*a, **k):
            raise RuntimeError("no host memory")
    monkeypatch.setattr(live, "np", NoHost())
    prompt, fed = [1, 2, 3], [3, 4, 5, 6]
    for ids in [prompt[:-1]] + [[t] for t in fed]:
        mgr.rows_for(tracer, ["r"])
        model(mx.array([ids]))
        mgr.clear_rows(tracer)
    monkeypatch.setattr(live, "np", np)
    assert len([r for r in caplog.records if "host memory" in r.getMessage()]) == 1
    assert sorted(len(v) for v in mgr.requests["r"].parts.values()) == [5] * 5          # every step filed whole
    assert mgr.requests["r"].step_tokens == [2, 1, 1, 1, 1] and not tracer.capture._pending
    mgr.finish("r", prompt, fed[1:] + [9], tracer)
    wait_saved(mgr, 1)
    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    assert t.tokens == prompt[:-1] + fed and t.get("router_ids", 1).shape == (6, 2)


@pytest.mark.parametrize("mtp", [False, True])
def test_a_traced_reply_keeps_one_step_of_mlx_arrays_however_long_it_runs(home, mtp):
    """Each step's records go to host memory once the next step is submitted: the MLX arrays a reply holds stay one
    step's worth however long it decodes (MLX fails every allocation past 499,000 buffers), one token a step or in
    MTP verify windows, and the saved trace has every position."""
    from mlx_lm.models.cache import make_prompt_cache
    config.save_settings(mode="on")
    model = tiny_moe()
    model._omlx_mtp_decode_enabled = mtp
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny", Tok())
    assert tracer.mtp == mtp
    mgr.register("r", "on", {})
    cache = make_prompt_cache(model)
    held = []

    def fwd(ids):
        mgr.rows_for(tracer, ["r"])
        model(mx.array([ids]), cache=cache)
        mgr.clear_rows(tracer)
        held.append(sum(isinstance(e[1], mx.array) for v in mgr.requests["r"].parts.values() for e in v))
    prompt = list(range(1, 21))
    fed = prompt[-1:] + list(range(30, 89))          # the last prompt token, then the reply
    fwd(prompt[:-1])
    for i in range(0, len(fed), 3 if mtp else 1):
        fwd(fed[i:i + 3] if mtp else fed[i:i + 1])
    # A step's own records, never more: the prompt chunk's input_ids and each layer's resid_norm, resid_write_norm,
    # router_ids and router_probs (9), then a decode step's input_ids and the four stacked over the layers (5).
    assert held[0] == 9 and set(held[1:]) == {5}
    mgr.finish("r", prompt, fed[1:] + [99], tracer)
    wait_saved(mgr, 1)
    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    assert t.tokens == prompt[:-1] + fed and t.get("router_ids", 0).shape == (len(t.tokens), 2)
    assert not tracer.fresh


def test_token_text_carries_a_split_character_on_its_last_byte():
    """A byte-level tokenizer splits an emoji over several tokens; each partial decode ends in U+FFFD. The live
    stream's text adds nothing for the partial tokens and the whole character on the last one, never U+FFFD."""
    from metalscope.livestream import token_text
    decode = lambda ids: bytes(ids).decode("utf-8", errors="replace")
    ids = list("Hi ".encode()) + list("😊".encode()) + list("!".encode())
    texts = [token_text(decode, ids[max(0, k - 8):k], t) for k, t in enumerate(ids)]
    assert "".join(texts) == "Hi 😊!" and "�" not in "".join(texts)
    assert texts[3:7] == ["", "", "", "😊"]                              # 4 bytes: nothing until the last


@pytest.mark.parametrize("pieces", [[b"\xe2\x80", b"\x93"], [b"\xe2", b"\x80", b"\x93"]])
def test_an_en_dash_split_over_two_or_three_tokens_is_shown_once_live_and_saved(pieces):
    """'$400–600' with the dash's three bytes over two or three tokens: the live text and the saved trace's both
    carry it once, on its last piece."""
    from metalscope.livestream import token_text, token_texts
    vocab = [b"$", b"400", *pieces, b"600"]
    decode = lambda ids: b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
    ids = list(range(len(vocab)))
    live = [token_text(decode, ids[max(0, k - 8):k], t) for k, t in enumerate(ids)]
    want = ["$", "400"] + [""] * (len(pieces) - 1) + ["–", "600"]
    assert live == token_texts(decode, ids) == want


def test_saved_token_texts_keep_whole_characters():
    """A saved trace's per-token texts: whole tokens as decoded alone, a split emoji on its last byte."""
    from metalscope.livestream import token_texts
    decode = lambda ids: bytes(ids).decode("utf-8", errors="replace")
    ids = list("Hi ".encode()) + list("😊".encode()) + list("!".encode())
    texts = token_texts(decode, ids)
    assert "".join(texts) == "Hi 😊!" and texts[:3] == ["H", "i", " "] and texts[3:7] == ["", "", "", "😊"]


def test_lens_flag_sends_each_layers_top10_only_to_those_who_ask():
    """``lens=1`` subscribers get ``lens10`` (each read layer's top 10, words sent once, p as a byte); everyone else
    gets the token event without it, and with no such subscriber the top 10 is not even read."""
    import asyncio
    from types import SimpleNamespace
    from metalscope.livestream import LENS_P_SCALE, LiveHub, LiveState, LiveStreamer
    vocab = {0: " the", 1: " cat", 2: " dark", 3: " moon", 4: "."}

    class Tk:
        all_special_ids = []

        def decode(self, ids):
            return "".join(vocab.get(i, f" w{i}") for i in ids)

    hub = LiveHub()
    loop = asyncio.new_event_loop()
    want, plain = hub.subscribe(loop, None, lens=True), hub.subscribe(loop, None, experts=True)
    assert hub.lens and hub.experts
    streamer = LiveStreamer(hub)
    tracer = SimpleNamespace(tokenizer=Tk(), lens=None, n_layers=None)
    rt = SimpleNamespace(request_id="r", meta={}, live=LiveState(client=None))
    top = {4: [2, 1, 3, 0, 4] + list(range(10, 15)), 8: [2, 3, 1] + list(range(10, 17))}
    logp = np.log(np.linspace(0.5, 0.01, 10))[None, None]
    refs = dict(logit_ids=(np.array([[top[8]]]), 0), logit_logp=(logp, 0), logit_entropy=(np.array([[0.3]]), 0),
                jlens_ids=[(l, np.array([[top[l]]]), 0) for l in (8, 4)],
                jlens_logp=[(l, logp, 0) for l in (4, 8)])
    streamer._announce(rt, 2, streamer._dist(refs, 0), tracer)
    got, other = drain(loop, want), drain(loop, plain)
    e = got[0]["lens10"]
    assert e["l"] == [4, 8] and e["k"] == 10 and len(e["c"]) == len(e["p"]) == 20
    assert [e["w"][c] for c in e["c"][:3]] == [" dark", " cat", " moon"] and len(e["i"]) == len(set(e["i"])) == 12
    assert e["p"][0] == round(-np.log(0.5) * LENS_P_SCALE) and set(e) == {"l", "k", "i", "w", "c", "p"}
    assert "lens10" not in other[0] and other[0]["lens"] == got[0]["lens"]   # the experts-only subscriber
    hub.unsubscribe(want)
    assert not hub.lens and "lens10" not in streamer._dist(refs, 0)
    # Both lenses (a J-lens on, the lens level): the J-lens is ``lens``, the logit lens beside it, top 10 for lens=1.
    want = hub.subscribe(loop, None, lens=True)
    logit_top = {4: [3, 2] + list(range(20, 28)), 8: [2] + list(range(20, 29))}
    refs.update(lens_ids=[(l, np.array([[logit_top[l]]]), 0) for l in (8, 4)], lens_logp=[(l, logp, 0) for l in (4, 8)])
    streamer._announce(rt, 2, streamer._dist(refs, 0), tracer)
    got, other = drain(loop, want)[0], drain(loop, plain)[0]
    assert got["lens_by"] == "jlens" and [i for _, i, _ in got["lens"]] == [2, 2]
    assert got["lens_logit"] == [[4, 3, " moon"], [8, 2, " dark"]]
    assert got["lens10_logit"]["l"] == [4, 8] and [got["lens10_logit"]["w"][c] for c in got["lens10_logit"]["c"][:2]] == [" moon", " dark"]
    assert other["lens_logit"] == got["lens_logit"] and "lens10_logit" not in other   # the top 10 only for lens=1
    del refs["jlens_ids"], refs["jlens_logp"]                                          # /jlens off: the logit lens alone
    streamer._announce(rt, 2, streamer._dist(refs, 0), tracer)
    alone = drain(loop, want)[0]
    assert alone["lens_by"] == "logit" and alone["lens"] == got["lens_logit"] and "lens_logit" not in alone
    assert alone["lens10"]["l"] == [4, 8]


def test_a_lens_row_recorded_empty_is_left_out_of_the_live_stream():
    """A row a lens did not read (ids -1, log-p -inf: before a prompt chunk's lens tail) is no word: the layers that
    read it are listed, the others left out, and nothing decodes id -1."""
    import asyncio
    from types import SimpleNamespace
    from metalscope.livestream import LiveHub, LiveState, LiveStreamer

    class Tk:
        all_special_ids = []

        def decode(self, ids):
            assert all(i >= 0 for i in ids), ids
            return "".join(f" w{i}" for i in ids)

    hub = LiveHub()
    loop = asyncio.new_event_loop()
    want = hub.subscribe(loop, None, lens=True)
    streamer = LiveStreamer(hub)
    tracer = SimpleNamespace(tokenizer=Tk(), lens=None, n_layers=None)
    rt = SimpleNamespace(request_id="r", meta={}, live=LiveState(client=None))
    ids = np.arange(10)[None, None]
    logp = np.log(np.linspace(0.5, 0.01, 10))[None, None]
    empty_ids, empty_logp = np.full((1, 1, 10), -1), np.full((1, 1, 10), -np.inf)
    refs = dict(logit_ids=(ids, 0), logit_logp=(logp, 0), logit_entropy=(np.array([[0.3]]), 0),
                lens_ids=[(4, empty_ids, 0), (8, ids, 0)], lens_logp=[(4, empty_logp, 0), (8, logp, 0)],
                jlens_ids=[(4, ids, 0), (8, empty_ids, 0)], jlens_logp=[(4, logp, 0), (8, empty_logp, 0)])
    streamer._announce(rt, 0, streamer._dist(refs, 0), tracer)
    got = drain(loop, want)[0]
    assert got["lens"] == [[4, 0, " w0"]] and got["lens_logit"] == [[8, 0, " w0"]]
    assert got["lens10"]["l"] == [4] and got["lens10_logit"]["l"] == [8]


def test_writes_flag_sends_each_layers_stream_writes_only_to_those_who_ask():
    """``writes=1`` subscribers get ``writes`` (every layer's write into each stream at the row that chose the token,
    shallow to deep, 3 significant digits) and ``write_k``; everyone else gets the token event without them, and with no
    such subscriber they are not even read. Without resid_write_norm records (a model without streams) there are none."""
    import asyncio
    from types import SimpleNamespace
    from metalscope.livestream import LiveHub, LiveState, LiveStreamer

    class Tk:
        all_special_ids = []

        def decode(self, ids):
            return "".join(f" w{i}" for i in ids)

    hub = LiveHub()
    loop = asyncio.new_event_loop()
    want, plain = hub.subscribe(loop, None, writes=True), hub.subscribe(loop, None, lens=True, experts=True)
    assert hub.writes
    streamer = LiveStreamer(hub)
    tracer = SimpleNamespace(tokenizer=Tk(), lens=None, n_layers=4)
    rt = SimpleNamespace(request_id="r", meta={}, live=LiveState(client=None))
    # Two stacked records (layers 2-3 filed before 0-1), batch row 1 is the request's, the step's row 1 chose it.
    w = np.arange(4 * 2 * 2 * 3, dtype=np.float32).reshape(4, 2, 2, 3) * 0.012345 + 0.0012345
    refs = dict(logit_ids=(np.arange(10)[None, None].repeat(2, 1), 0), logit_logp=(np.full((1, 2, 10), -2.3), 0),
                logit_entropy=(np.array([[0.3, 0.4]]), 0),
                resid_write_norm=[([2, 3], w[2:], 1), ([0, 1], w[:2], 1)])
    streamer._announce(rt, 4, streamer._dist(refs, 1), tracer)
    got, other = drain(loop, want)[0], drain(loop, plain)[0]
    assert got["write_k"] == 3 and len(got["writes"]) == 12 and "reads" not in got and "read_k" not in got   # no shares
    np.testing.assert_allclose(got["writes"], w[:, 1, 1].ravel(), rtol=5e-3)
    assert got["writes"][:2] == [float(f"{w[0, 1, 1, 0]:.3g}"), float(f"{w[0, 1, 1, 1]:.3g}")]
    assert all(x == float(f"{x:.3g}") for x in got["writes"])
    assert "writes" not in other and "write_k" not in other and other["id"] == got["id"] == 4
    del refs["resid_write_norm"]
    assert "writes" not in streamer._dist(refs, 1)
    hub.unsubscribe(want)
    refs["resid_write_norm"] = [([0, 1, 2, 3], w, 1)]
    assert not hub.writes and "writes" not in streamer._dist(refs, 1)


def test_shares_are_each_layers_whole_write_over_the_whole_residual_it_read():
    """``shares`` beside ``writes``: layer l's write (the root of the sum of its streams' squares) over layer l - 1's
    output, the streams' sizes (hc_stream_norm) where recorded, else the one residual's (resid_norm), 3 significant
    digits. Layer 0 reads the embedding, whose size is not recorded: 0; a layer that read a residual of size 0: 0. None
    with reads (Inkling 2), without the sizes, or unless both cover every layer from 0."""
    from metalscope.livestream import _writes
    rng = np.random.default_rng(0)
    w = rng.uniform(0.1, 2, (4, 2, 3, 4)).astype(np.float32)          # [L, B, T, hc]; batch row 1 the request's
    hs, r = rng.uniform(5, 50, (4, 2, 3, 4)).astype(np.float32), rng.uniform(5, 50, (4, 2, 3)).astype(np.float32)
    refs = dict(resid_write_norm=[([2, 3], w[2:], 1), ([0, 1], w[:2], 1)], hc_stream_norm=[([0, 1, 2, 3], hs, 1)],
                resid_norm=[([1, 3], r[[1, 3]], 1), ([0, 2], r[[0, 2]], 1)])
    whole = np.linalg.norm(w[:, 1, 2], axis=-1)
    got = _writes(refs, 2)["shares"]
    assert len(got) == 4 and got[0] == 0 and got == [float(f"{x:.3g}") for x in got]
    np.testing.assert_allclose(got[1:], whole[1:] / np.linalg.norm(hs[:-1, 1, 2], axis=-1), rtol=5e-3)
    del refs["hc_stream_norm"]                                          # one residual: its norm
    np.testing.assert_allclose(_writes(refs, 2)["shares"][1:], whole[1:] / r[:-1, 1, 2], rtol=5e-3)
    r[1, 1, 2] = 0
    assert _writes({**refs, "resid_norm": [([0, 1, 2, 3], r, 1)]}, 2)["shares"][2] == 0, "layer 2 read nothing"
    assert "shares" not in _writes({**refs, "resid_read_share": [([0, 1, 2, 3], w, 1)]}, 2), "reads: none"
    assert "shares" not in _writes({"resid_write_norm": refs["resid_write_norm"]}, 2), "no sizes: none"
    assert "shares" not in _writes({**refs, "resid_norm": [([0, 2], r[[0, 2]], 1)]}, 2), "some layers' sizes: none"
    assert "shares" not in _writes({**refs, "resid_write_norm": [([1, 2, 3], w[1:], 1)],
                                    "resid_norm": [([1, 2, 3], r[1:], 1)]}, 2), "not from layer 0: none"


@pytest.mark.parametrize("mtp", [False, True])
def test_flash_next_writes_reach_the_trace_and_the_live_stream(home, mtp):
    """A model with residual streams (the qwen4_exp adapter): every layer's write into each stream is saved in the trace
    (resid_write_norm, a row a position) and each token event of a ``writes`` subscriber carries those of the forward that
    chose it, the position before its own. Under MTP decode the verify windows' rows each choose a token, and a
    rejected draft's row is overwritten by the forward that fed the emitted token there."""
    import asyncio
    from mlx_lm.models.cache import make_prompt_cache
    from tests.helpers import FlashNextLike, flash_next_writes
    config.save_settings(mode="on")
    model = FlashNextLike(layers=3, hc=4)
    if mtp:
        model._omlx_mtp_decode_enabled = True
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny-hc", Tok())
    assert tracer.adapter_name == "qwen4_exp" and tracer.mtp == mtp
    loop = asyncio.new_event_loop()
    want, plain = mgr.hub.subscribe(loop, "me", writes=True), mgr.hub.subscribe(loop, "me", experts=True, lens=True)
    mgr.register("r1", "on", dict(client="me"))
    prompt = list(range(1, 21))                      # > MAX_WINDOW: the prefill reads as a prefill
    cache = make_prompt_cache(model)

    def fwd(ids):
        mgr.rows_for(tracer, ["r1"])
        out = model(mx.array([ids]), cache=cache)
        mgr.clear_rows(tracer)
        return [int(v) for v in mx.argmax(out[0], axis=-1).tolist()]

    fwd(prompt[:-1])
    if mtp:                                           # as test_mtp_windows_keep_only_emitted_positions
        g0 = fwd(prompt[-1:])[0]
        w1 = fwd([g0, 5, 6])                          # draft 5 accepted, draft 6 rejected
        for c in cache:
            c.trim(1)
        w2 = fwd([w1[1], 7, 8])
        outputs = [g0, 5, w1[1], 7, 8, w2[2]]
    else:
        fed, outputs = prompt[-1], []
        for _ in range(4):
            fed = fwd([fed])[0]
            outputs.append(fed)
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)

    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    P, n = len(prompt), len(prompt) + len(outputs) - 1
    assert t.tokens == prompt + outputs[:-1]
    W = np.stack([t.get("resid_write_norm", l) for l in range(3)], axis=1)          # [positions, L, hc]
    assert W.shape == (n, 3, 4)
    # Every position holds the writes of one causal pass over the kept tokens (under MTP, the rejected draft's row
    # overwritten by the forward that fed the emitted token there).
    # (a decode step from the cache against one pass: 0.2% apart on the GPU, measured)
    wo, so = flash_next_writes(model, mx.array([t.tokens]), sizes=True)               # [L, 1, positions, hc]
    np.testing.assert_allclose(W, wo[:, 0].transpose(1, 0, 2), rtol=1e-2)
    # Each layer's share: its whole write over the whole of the streams it read, the layer before's output.
    shares = np.linalg.norm(wo[1:, 0], axis=-1) / np.linalg.norm(so[:-1, 0], axis=-1)    # [L - 1, positions]
    tokens = [e for e in drain(loop, want) if e["type"] == "token"]
    assert [e["id"] for e in tokens] == outputs
    for i, e in enumerate(tokens):                    # token i was chosen at position P - 1 + i
        assert e["write_k"] == 4 and len(e["writes"]) == 3 * 4
        np.testing.assert_allclose(e["writes"], W[P - 1 + i].ravel(), rtol=6e-3, err_msg=f"token {i}")
        assert len(e["shares"]) == 3 and e["shares"][0] == 0
        np.testing.assert_allclose(e["shares"][1:], shares[:, P - 1 + i], rtol=1e-2, err_msg=f"token {i}")
    others = [e for e in drain(loop, plain) if e["type"] == "token"]
    assert [e["id"] for e in others] == outputs and not any(k in e for e in others for k in ("writes", "write_k", "shares"))


def test_a_conventional_decoders_writes_reach_the_trace_and_the_live_stream_as_one_stream(home):
    """The generic adapter (one residual): each layer's write is saved (resid_write_norm [positions, 1]) and a
    ``writes`` subscriber's token events carry the forward that chose it, ``write_k`` 1; no one else gets them."""
    import asyncio
    from mlx_lm.models.cache import make_prompt_cache
    config.save_settings(mode="on")
    model = tiny_llama()
    mgr = TraceManager()
    tracer = mgr.attach(model, "tiny", Tok())
    assert tracer.adapter_name == "generic"
    loop = asyncio.new_event_loop()
    want, plain = mgr.hub.subscribe(loop, "me", writes=True), mgr.hub.subscribe(loop, "me", lens=True)
    mgr.register("r1", "on", dict(client="me"))
    prompt, cache = list(range(1, 21)), make_prompt_cache(model)
    fed, outputs = prompt[-1], []

    def fwd(ids):
        mgr.rows_for(tracer, ["r1"])
        out = model(mx.array([ids]), cache=cache)
        mgr.clear_rows(tracer)
        return int(mx.argmax(out[0, -1]).item())
    fwd(prompt[:-1])
    for _ in range(4):
        fed = fwd([fed])
        outputs.append(fed)
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)
    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    W = np.stack([t.get("resid_write_norm", l) for l in range(3)], axis=1)          # [positions, L, 1]
    h, offline, sizes = model.model.embed_tokens(mx.array([t.tokens])), [], []
    for layer in model.model.layers:
        out = layer(h, mask="causal", cache=None)
        offline.append(np.linalg.norm(np.array((out - h).astype(mx.float32))[0], axis=-1))
        sizes.append(np.linalg.norm(np.array(out.astype(mx.float32))[0], axis=-1))
        h = out
    assert W.shape == (len(t.tokens), 3, 1)
    np.testing.assert_allclose(W[..., 0], np.stack(offline, axis=1), rtol=1e-2)
    shares = np.stack(offline[1:]) / np.stack(sizes[:-1])           # [L - 1, positions]: over the residual read
    tokens = [e for e in drain(loop, want) if e["type"] == "token"]
    assert [e["id"] for e in tokens] == outputs
    for i, e in enumerate(tokens):                    # token i was chosen at position P - 1 + i
        assert e["write_k"] == 1 and "reads" not in e
        np.testing.assert_allclose(e["writes"], W[len(prompt) - 1 + i].ravel(), rtol=6e-3, err_msg=f"token {i}")
        assert len(e["shares"]) == 3 and e["shares"][0] == 0, "layer 0 read the embedding: not recorded"
        np.testing.assert_allclose(e["shares"][1:], shares[:, len(prompt) - 1 + i], rtol=1e-2, err_msg=f"token {i}")
    assert not any("writes" in e or "shares" in e for e in drain(loop, plain) if e["type"] == "token")


class WordTok(Tok):
    def encode(self, text, add_special_tokens=False):
        return [sum(map(ord, text)) % 100]


def test_lens_use_switches_a_loaded_model_between_its_j_lens_and_the_logit_lens_without_a_reload(home):
    from metalscope import jlens
    eye = mx.eye(64)
    jlens.save(jlens.Lens({l: eye for l in range(3)}, dict(source="test", checkpoint="tiny")), jlens.lenses_dir("tiny") / "tiny-learned")
    jlens.use("tiny-learned", "tiny")
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", WordTok())
    assert tracer.capture is not None                                     # one capture session
    import asyncio
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None, lens=True)

    def reply(rid):
        mgr.register(rid, "on", dict(conversation=rid, turn=1))
        mgr.rows_for(tracer, [rid]); model(mx.array([[1, 5]])); mgr.clear_rows(tracer)
        for fed in (9, 11):
            mgr.rows_for(tracer, [rid]); model(mx.array([[fed]])); mgr.clear_rows(tracer)
        mgr.finish(rid, [1, 5, 9], [11, 12], tracer)

    reply("r1")
    assert tracer.lens.name == "tiny-learned" and tracer.capture.config.jlens is not None
    jlens.use(jlens.LOGIT, "tiny")
    assert tracer.refresh_lens() is None
    cfg = tracer.capture.config
    assert cfg.jlens is None and cfg.logit_lens_topk == 10
    reply("r2")
    jlens.use("tiny-learned", "tiny")
    assert tracer.refresh_lens().name == "tiny-learned" and tracer.capture.config.logit_lens_topk == 10
    reply("r3")
    wait_saved(mgr, 3)
    index = [json.loads(l) for l in config.index_path().read_text().splitlines()]
    by = {e["conversation"]: Trace.load(e["path"]) for e in index}
    # With the J-lens on, both lenses on the live stream (the logit lens beside it); /jlens off, the logit lens alone.
    read = {e["rid"]: e for e in drain(loop, q) if e["type"] == "token"}
    assert read["r1"]["lens_by"] == "jlens" and read["r1"]["lens_logit"] and read["r1"]["lens10_logit"]["k"] == 10
    assert read["r2"]["lens_by"] == "logit" and "lens_logit" not in read["r2"]
    assert read["r3"]["lens_by"] == "jlens" and read["r3"]["lens_logit"]
    assert set(by) == {"r1", "r2", "r3"}


def test_conversation_meta_reads_anthropic_and_responses_bodies():
    a = conversation_meta({"model": "m", "system": [{"type": "text", "text": "sys"}], "messages": [
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}]})
    assert a["turn"] == 1 and a["last_user"] == "first"
    b = conversation_meta({"model": "m", "system": "sys", "messages": [{"role": "user", "content": "first"}]})
    assert a["conversation"] == b["conversation"]
    r = conversation_meta({"model": "m", "instructions": "sys", "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "first"}]}]})
    assert r["conversation"] == b["conversation"] and r["last_user"] == "first"
    s = conversation_meta({"model": "m", "input": "just text"})
    assert s["turn"] == 0 and s["last_user"] == "just text"


def test_lens_steps_read_the_workspace_and_the_last_layer_and_the_default_reads_today_s_sets():
    """Flash Next's shape (48 layers, a J-lens over 0-46, workspace 24 31 38): 4, 8, 15, 24 and 47 layers; every step
    reads the workspace and the last; the default step is the sets traces always read, the others one set for both."""
    from types import SimpleNamespace
    from metalscope.capture import capture_config
    from metalscope.tracer import jlens_read_layers, lens_steps
    setup = SimpleNamespace(lens=SimpleNamespace(layers=list(range(47))), workspace=[24, 31, 38])
    steps = lens_steps(setup, 48)
    assert [len(j) for j, _ in steps] == [4, 8, 15, 24, 47]
    assert steps[0] == ([24, 31, 38, 46], [24, 38, 46, 47])        # the logit lens: its default layers among those, and the last
    assert steps[4] == (list(range(47)), list(range(48)))
    assert steps[config.LENS_STEP_DEFAULT] == (jlens_read_layers(setup), capture_config().lens_layers(48))
    assert all({24, 31, 38, 46} <= set(j) and 47 in l for j, l in steps)
    # The logit lens alone: over the model's layers, its default capture_config's.
    plain = lens_steps(None, 48)
    assert [len(l) for _, l in plain] == [3, 8, 21, 24, 48] and all(j == [] for j, _ in plain)
    assert plain[config.LENS_STEP_DEFAULT][1] == capture_config().lens_layers(48) and plain[0][1] == [24, 38, 47]
    assert lens_steps(None, 1) == [([], [0])] * 5


def test_lens_steps_only_grow_and_the_default_is_today_s_sets_at_any_shape():
    """Every depth 1-100 with a J-lens over every layer but the last, every layer, every second, the last 8, three, or
    none: each lens's set, and the layers either reads, only grow step by step (nested); the default step is the sets
    traces always read; every layer is the model's; the logit lens always reads the model's last."""
    from types import SimpleNamespace
    from metalscope import jlens
    from metalscope.capture import capture_config
    from metalscope.tracer import jlens_read_layers, lens_steps
    for n in range(1, 101):
        shapes = [None, list(range(n)), list(range(max(1, n - 1))), list(range(0, n, 2)), list(range(max(0, n - 8), n)),
                  sorted({0, n // 2, n - 1})]
        for layers in shapes:
            setup = None if layers is None else SimpleNamespace(lens=SimpleNamespace(layers=layers), workspace=[
                l for l in jlens.workspace_layers(n) if l in layers] or layers[-3:])
            steps = lens_steps(setup, n)
            assert len(steps) == config.LENS_STEPS
            assert steps[config.LENS_STEP_DEFAULT] == (jlens_read_layers(setup) if setup else [], capture_config().lens_layers(n))
            for (j0, l0), (j1, l1) in zip(steps, steps[1:]):
                assert set(j0) <= set(j1) and set(l0) <= set(l1), (n, layers)
            assert all(0 <= x < n for j, l in steps for x in (*j, *l)) and all(n - 1 in l for _, l in steps)


def test_two_replies_at_two_lens_steps_read_two_layer_sets_and_one_in_flight_keeps_its_own(home, monkeypatch):
    """oMLX: the lens step is the server's setting, taken by the next traced request when none is in flight; the
    trace reads that step's layers with both lenses and saves the step and its decode speed (a clock that moves 10 ms
    a decode forward, after a 0.5 s wait on the prompt: 100 tok/s)."""
    from metalscope import jlens, live
    clock = [0.0]
    monkeypatch.setattr(live.time, "perf_counter", lambda: clock[0])
    eye = mx.eye(64)
    jlens.save(jlens.Lens({l: eye for l in range(12)}, dict(source="test", checkpoint="deep")), jlens.lenses_dir("deep") / "deep-learned")
    jlens.use("deep-learned", "deep")
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama(12)
    tracer = mgr.attach(model, "deep", WordTok())
    steps = tracer.lens_steps()
    import asyncio
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None, lens=True)
    seen: dict = {}

    def layers(rid):
        """The layers each lens read for ``rid``'s newest token on the live stream: (the J-lens's, the logit lens's)."""
        for e in drain(loop, q):
            if e["type"] == "token":
                seen[e["rid"]] = ([l for l, _, _ in e["lens"]], [l for l, _, _ in e["lens_logit"]])
        return seen[rid]

    def start(rid):
        mgr.register(rid, "on", dict(conversation=rid, turn=1))
        mgr.rows_for(tracer, [rid], "prefill"); model(mx.array([[1, 5]])); mgr.clear_rows(tracer)

    def end(rid, n=2):
        clock[0] += 0.5
        for fed in range(n):
            clock[0] += 0.01
            mgr.rows_for(tracer, [rid], "decode"); model(mx.array([[10 + fed]])); mgr.clear_rows(tracer)
        mgr.finish(rid, [1, 5], list(range(10, 11 + n)), tracer)

    config.save_settings(lens_step=0)
    start("r1"); end("r1")
    config.save_settings(lens_step=4)
    start("r2")
    config.save_settings(lens_step=1)
    start("r3")                                        # r2 in flight: r3 reads r2's layers too
    end("r3"); end("r2", n=12)
    wait_saved(mgr, 3)
    lines = lambda: [json.loads(l) for l in config.index_path().read_text().splitlines()]
    index = {e["conversation"]: e for e in lines()}
    assert steps[0][0] != steps[4][0]
    assert layers("r1") == steps[0]
    assert layers("r2")[0] == layers("r2")[1] == steps[4][0] == list(range(12))
    assert layers("r3")[0] == steps[4][0]
    assert [index[r]["lens_step"] for r in ("r1", "r2", "r3")] == [0, 4, 4]
    assert index["r2"]["decode_tps"] == pytest.approx(100.0) and index["r1"]["decode_tps"] is None   # 2 tokens: too few
    start("r4"); end("r4")                                # none in flight: the setting now
    wait_saved(mgr, 4)
    assert layers("r4")[0] == steps[1][0]
    # A request that ends unfinished (released, aborted) holds the step no longer.
    config.save_settings(lens_step=3)
    start("r5")
    config.save_settings(lens_step=0)
    start("r6")
    assert mgr.requests["r6"].meta["lens_step"] == 3
    mgr.release("r5"); mgr.release("r6")
    start("r7")
    assert mgr.requests["r7"].meta["lens_step"] == 0 and tracer.capture.config.jlens_layers == steps[0][0]
    mgr.release("r7")


def test_lens_step_setting_is_checked_when_set_and_a_bad_stored_one_reads_as_the_default(home):
    for bad in ("3", 2.0, True, -2, 5, None):
        with pytest.raises(ValueError, match="lens_step"):
            config.save_settings(lens_step=bad)
    for k in [config.LENS_OFF, *range(config.LENS_STEPS)]:
        assert config.save_settings(lens_step=k)["lens_step"] == k
    p = config.settings_path()
    p.write_text(json.dumps(dict(json.loads(p.read_text()), lens_step=9)))      # by hand, or a later version's
    mgr = TraceManager()
    assert mgr.lens_step() == config.LENS_STEP_DEFAULT
    assert config.save_settings(mode="off")["lens_step"] == config.LENS_STEP_DEFAULT      # other changes still save
    assert json.loads(p.read_text())["lens_step"] == config.LENS_STEP_DEFAULT


def test_with_the_lenses_off_a_traced_reply_reads_no_lens_and_keeps_its_probabilities(home):
    """lens_step LENS_OFF (the dock's - past the fewest layers): the next traced requests read neither lens (no rows
    on the feed), their tokens' probabilities as ever; the step saved with each; the setting holds
    until + (the fewest layers again)."""
    import asyncio
    from metalscope import jlens
    jlens.save(jlens.Lens({l: mx.eye(64) for l in range(12)}, dict(source="test", checkpoint="deep")), jlens.lenses_dir("deep") / "deep-learned")
    jlens.use("deep-learned", "deep")
    config.save_settings(mode="on", lens_step=config.LENS_OFF)
    mgr = TraceManager()
    model = tiny_llama(12)
    tracer = mgr.attach(model, "deep", WordTok())
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None, lens=True)

    def reply(rid):
        """Greedy: each token written the likeliest (its probability on the feed)."""
        mgr.register(rid, "on", dict(conversation=rid, turn=1))
        mgr.rows_for(tracer, [rid], "prefill"); out = [int(mx.argmax(model(mx.array([[1, 5]]))[0, -1]))]; mgr.clear_rows(tracer)
        for _ in range(3):
            mgr.rows_for(tracer, [rid], "decode"); out.append(int(mx.argmax(model(mx.array([[out[-1]]]))[0, -1]))); mgr.clear_rows(tracer)
        mgr.finish(rid, [1, 5], out, tracer)

    reply("r1"); reply("r2")
    wait_saved(mgr, 2)
    assert tracer.read_layers() == ([], []) and tracer.capture.config.jlens_layers == tracer.capture.config.logit_lens_layers == []
    toks = [e for e in drain(loop, q) if e["type"] == "token"]
    assert {e["rid"] for e in toks} == {"r1", "r2"} and all(e.get("p") is not None for e in toks)
    assert not any(e.get("lens") or e.get("lens_logit") or e.get("lens10") for e in toks)
    index = [json.loads(l) for l in config.index_path().read_text().splitlines()]
    assert [e["lens_step"] for e in index] == [config.LENS_OFF] * 2
    config.save_settings(lens_step=0)                  # + : the fewest layers again
    reply("r3")
    wait_saved(mgr, 3)
    assert tracer.read_layers() == tracer.lens_steps()[0]
    assert any(e.get("lens") for e in drain(loop, q) if e["type"] == "token")


def test_a_relayed_piece_times_the_reply_only_when_it_carries_reply_text():
    from metalscope.live import REPLY_TEXT
    text = [b'data: {"choices":[{"delta":{"content":"hi"}}]}', b'{"delta":{"reasoning_content":"so"}}',
            b'{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}', b'{"type":"thinking_delta","thinking":"h"}',
            b'{"type":"response.output_text.delta","delta":"hi"}']
    opening = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}', b'{"delta":{"role":"assistant","content":""}}',
               b'{"type":"content_block_start","content_block":{"type":"text","text":""}}',
               b'{"type":"message_start","message":{"role":"assistant","content":[]}}', b"data: [DONE]"]
    assert all(REPLY_TEXT.search(c) for c in text) and not any(REPLY_TEXT.search(c) for c in opening)


def _stub_vectors(model):
    """A concept's vector (calm) and a word's (soccer, its ``token_ids``) as found on disk: the tensor file (its
    contents unread here) and the sidecar."""
    from metalscope import models
    for name, meta in (("calm", {}), ("soccer", dict(token_ids=[7], words=["soccer"], cut=[]))):
        p = models.vector_path(model, name)
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        config.atomic_write(p.with_suffix(".json"), json.dumps(dict(meta, layers=[1])))


def _planner(monkeypatch, tracer):
    """steer_failed of _ensure_plan on ``tracer``, build_plan stubbed (it records the Jacobians it was given)."""
    from types import SimpleNamespace

    from metalscope import live
    from metalscope.live import RequestTrace
    got = {}

    def plan(spec, name, rid, hc, jac, table):
        got["jac"] = jac
        return SimpleNamespace(meta=dict(label=spec.label, layers=[], abs_norm={}))
    monkeypatch.setattr(live, "build_plan", plan)
    mgr = TraceManager()
    monkeypatch.setattr(mgr, "strength_table", lambda name: {"layers": {1: 1.0}})

    def failed(spec):
        rt = RequestTrace("r", "on", {}, steer_spec=spec)
        mgr._ensure_plan(rt, tracer)
        return rt.steer_failed
    return failed, got


def test_a_word_vector_steers_only_as_the_edit_on_every_model(home, monkeypatch):
    """_ensure_plan: a word's vector (made by the lens, its ``token_ids``) in a steer that is not the edit's is refused
    with steer.WORD_EDIT_ONLY; a concept's, and the edit's own, are planned."""
    from types import SimpleNamespace

    from metalscope import steer
    _stub_vectors("m")
    failed, _ = _planner(monkeypatch, SimpleNamespace(name="m", lens=None, hc_count=1))
    assert failed(steer.parse_header("calm@0.3/1")) is None
    for header in ("soccer@0.3/1", "remove~soccer/1", "calm@0.3/1+soccer@0.2/1"):
        assert failed(steer.parse_header(header)) == f"plan: {steer.WORD_EDIT_ONLY}", header
    assert failed(steer.SteerSpec(steer.parse_header("remove~soccer/1").parts, positions="decode", edit=True)) is None


def test_a_word_vector_outside_an_edit_is_a_400_before_scheduling(home):
    """The middleware's check_spec runs the same check (steer.check_words): a steer header naming a word's vector is
    refused with 400 and the reason; the edit header's steer (parse_edit: the edit's own) passes."""
    from metalscope import middleware, steer
    from tests.helpers import run_middleware
    config.save_settings(steer=True)
    _stub_vectors("tiny")
    body = json.dumps({"model": "org/tiny", "messages": [{"role": "user", "content": "hi"}]}).encode()
    for header in ("soccer@0.3/1", "remove~soccer/1"):
        seen, sent = run_middleware({config.STEER_HEADER: header}, body)
        assert sent[0]["status"] == 400 and steer.WORD_EDIT_ONLY.encode() in sent[1]["body"], sent[1]["body"]
    edit = middleware._steer_of(config.EDIT_HEADER, "r1@0 remove~soccer/1", "tiny")   # what an edit's server checks
    assert edit["steer"].edit and edit["steer"].label == "remove~soccer/1"


def test_a_lens_of_another_layout_fails_only_an_edits_plan(home, monkeypatch):
    """Inkling 2's lens_columns refuses a Jacobian of the wrong width: an edit (its word vectors need the lens) is not
    planned; a concept's steer is, without readout norms (``jac`` None)."""
    from types import SimpleNamespace

    from metalscope import steer

    def cut(layer, width):
        raise ValueError(f"this lens is not as wide as the model's state at visit {layer}: fit it again (/jlens learn)")
    lens = SimpleNamespace(lens=SimpleNamespace(jacobians={1: np.zeros((4, 12), np.float32)}))
    _stub_vectors("m")
    failed, got = _planner(monkeypatch, SimpleNamespace(name="m", lens=lens, hc_count=1, lens_columns=cut))
    assert failed(steer.parse_header("calm@0.3/1")) is None and got["jac"] is None
    edit = steer.SteerSpec(steer.parse_header("remove~soccer/1").parts, positions="decode", edit=True)
    assert failed(edit) == "plan: this lens is not as wide as the model's state at visit 1: fit it again (/jlens learn)"

"""The lookup view's server side (lookup.py): POST /scope/api/lookup measures a saved reply on the served model with
and without its n-gram lookup, keeps the answer beside the trace, and refuses what it cannot do; the ``lookup`` cap;
the CLI table; oMLX's measure on its engine's executor under a lease."""

import asyncio
import dataclasses
import json
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from metalscope import config, gpulock, lookup
from metalscope.lookup import Kind
from metalscope.engines import NextLp
from metalscope.live import MANAGER
from metalscope.omlx import backend as omlx_backend
from metalscope.store import Trace, append_index
from tests.helpers import scope_client

PROMPT, REPLY = [5, 6, 7, 8], [9, 10, 11]


class Tracer:
    installed, adapter_name = True, "fake"
    tokenizer = SimpleNamespace(decode=lambda ids: "".join(f"<{i}>" for i in ids))

    def __init__(self, name="flash", lookup=True):
        self.name, self.lookup, self.lookup_kind = name, lookup, Kind()


def top_of(ids, start, on):
    """The fake's two likeliest next tokens a row: the written one first, then 3 (on) or 4 (off)."""
    nxt = np.array(ids[start:])
    return np.stack([nxt, np.full_like(nxt, 3 if on else 4)], 1), np.tile([-0.5, -1.5], (len(nxt), 1)).astype(np.float32)


@pytest.fixture
def served():
    """A fake served model on a server whose measure records its calls: lp on = -0.1 a token, off = -0.1 - 0.05 * i,
    each row's top two (top_of)."""
    calls = []

    async def measure(tracer, ids, start, on, job):
        calls.append((tracer.name, list(ids), start, on))
        n = len(ids) - start
        return NextLp(np.full(n, -0.1, np.float32) if on else -0.1 - 0.05 * np.arange(n, dtype=np.float32), *top_of(ids, start, on))
    tracer = Tracer()
    MANAGER.tracers[id(tracer)] = tracer
    c = scope_client(dataclasses.replace(omlx_backend.BACKEND, more_caps=frozenset, lookup=measure))
    yield c, tracer, calls
    MANAGER.tracers.pop(id(tracer), None)


def saved(tid="t1", model="flash", time=1.0):
    path = config.home() / "traces" / tid
    toks = PROMPT + REPLY
    Trace({}, toks, [f"<{i}>" for i in toks], len(PROMPT), {"model": model}).save(path)
    append_index(dict(id=tid, path=str(path), model=model, time=time))
    return path


def test_the_cap_is_listed_while_a_served_model_has_a_lookup_and_the_backend_can_measure(served):
    c, tracer, _ = served
    assert "lookup" in c.get("/scope/api/status").json()["caps"]
    tracer.lookup = False
    assert "lookup" not in c.get("/scope/api/status").json()["caps"]
    tracer.lookup = True
    assert "lookup" not in omlx_backend.BACKEND.__class__(engine="x").can()       # no way to measure: no cap
    assert lookup.caps([Tracer(lookup=False)]) == frozenset()


def test_a_reply_is_measured_once_then_read_from_beside_its_trace(served):
    c, _, calls = served
    path = saved()
    r = c.post("/scope/api/lookup", json=dict(trace="t1"))
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["ids"] == REPLY and got["prompt_tokens"] == 4 and got["model"] == "flash"
    assert got["lp_on"] == [-0.1] * 3 and got["lp_off"] == pytest.approx([-0.1, -0.15, -0.2])
    assert [(on, start) for _n, _ids, start, on in calls] == [(True, 4), (False, 4)]
    assert calls[0][1] == PROMPT + REPLY and set(got["seconds"]) == {"on", "off"}
    # Both passes' top choices, row by row, and the text of every id among them.
    assert got["top_ids_on"] == [[9, 3], [10, 3], [11, 3]] and got["top_ids_off"] == [[9, 4], [10, 4], [11, 4]]
    assert got["top_lp_on"] == got["top_lp_off"] == [[-0.5, -1.5]] * 3
    assert got["vocab"] == {"3": "<3>", "4": "<4>", "9": "<9>", "10": "<10>", "11": "<11>"}
    assert json.loads((path / lookup.FILE).read_text()) == got
    again = c.post("/scope/api/lookup", json=dict(trace="t1")).json()
    assert again == got and len(calls) == 2                       # kept: no second measurement


def test_an_answer_of_an_older_version_is_measured_again(served):
    c, _, calls = served
    path = saved()
    (path / lookup.FILE).write_text(json.dumps(dict(version=1, trace="t1", ids=REPLY, lp_on=[0, 0, 0], lp_off=[0, 0, 0])))
    got = c.post("/scope/api/lookup", json=dict(trace="t1")).json()
    assert len(calls) == 2 and got["version"] == lookup.VERSION and "top_ids_on" in got
    assert json.loads((path / lookup.FILE).read_text()) == got


@pytest.mark.parametrize("why", ["unknown", "other model", "no lookup", "serving", "gpu", "measuring", "no cap"])
def test_refusals_say_why(served, monkeypatch, why):
    c, tracer, calls = served
    saved(model="other" if why == "other model" else "flash")
    if why == "no lookup":
        tracer.lookup = False
        MANAGER.tracers[0] = Tracer("spare")                       # the server still lists the cap for another model
        monkeypatch.setattr(lookup, "caps", lambda _t: frozenset({"lookup"}))
    if why == "serving":
        monkeypatch.setitem(MANAGER.requests, "r1", object())
    job = gpulock.GpuLock(gpulock.Holder("learning calm")).acquire() if why == "gpu" else None
    if why == "measuring":
        lookup.ONE.acquire()
    if why == "no cap":
        tracer.lookup = False
    try:
        r = c.post("/scope/api/lookup", json=dict(trace="nope" if why == "unknown" else "t1"))
    finally:
        MANAGER.tracers.pop(0, None)
        if job:
            job.release()
        if why == "measuring":
            lookup.ONE.release()
    want = {"unknown": (404, "unknown trace"), "other model": (409, "other's, not served here"),
            "no lookup": (409, "flash has no engram"), "serving": (409, "a reply is being served"),
            "gpu": (409, "learning calm"), "measuring": (409, "an engram inspection is under way"),
            "no cap": (409, "not available on this server")}[why]
    assert (r.status_code, want[1] in r.json()["detail"]) == (want[0], True), r.text
    assert calls == [] and not lookup.ONE.locked() and (why == "gpu" or gpulock.held() is None)


@pytest.mark.parametrize("meta, why", [(dict(tokens_cut=3), "lacks the conversation's first tokens"),
                                       (dict(tokens=None), "keeps no token ids")])
def test_a_trace_without_its_whole_token_list_is_refused(served, meta, why):
    c, _, calls = served
    path = saved()
    m = json.loads((path / "meta.json").read_text())
    (path / "meta.json").write_text(json.dumps({k: v for k, v in {**m, **meta}.items() if v is not None}))
    r = c.post("/scope/api/lookup", json=dict(trace="t1"))
    assert r.status_code == 409 and why in r.json()["detail"] and calls == []


def test_the_gpu_lock_is_held_for_both_passes(served):
    """A GPU job starting mid-measurement finds the lock held, by the lookup; it is free again after."""
    c, _, _ = served
    saved()
    seen = []

    async def measure(tracer, ids, start, on, job):
        seen.append(gpulock.held().what)
        with pytest.raises(gpulock.LockBusy, match="inspecting the engram"):
            gpulock.GpuLock(gpulock.Holder("learning calm")).acquire(wait=0)
        return NextLp(np.zeros(len(ids) - start, np.float32), *top_of(ids, start, on))
    c.app.state.metalscope_backend = dataclasses.replace(c.app.state.metalscope_backend, lookup=measure)
    assert c.post("/scope/api/lookup", json=dict(trace="t1")).status_code == 200
    assert seen == [lookup.HOLDER] * 2 and gpulock.held() is None


def test_a_failed_measurement_is_a_503_and_frees_the_lock(served):
    c, _, _ = served
    saved()

    async def broken(*_a):
        raise RuntimeError("ds4-server read no next-token log-probabilities")
    c.app.state.metalscope_backend = dataclasses.replace(c.app.state.metalscope_backend, lookup=broken)
    r = c.post("/scope/api/lookup", json=dict(trace="t1"))
    assert r.status_code == 503 and "no next-token" in r.json()["detail"] and not lookup.ONE.locked()
    assert gpulock.held() is None


def test_the_table_ranks_the_helped_tokens_after_the_three_before_them():
    answer = dict(prompt_tokens=2, ngram=3, lp_on=[np.log(0.85), np.log(0.5), np.log(0.4)],
                  lp_off=[np.log(0.09), np.log(0.5), np.log(0.1)])
    lines = lookup.table(answer, [" the", " empire", " did", " not", " survive"])
    assert lines[0].startswith("3 tokens: lookup +") and lines[0].endswith("helped 2")
    assert "85% on   9% off  ' the empire' -> ' did'" in lines[1] and "' empire did not' -> ' survive'" in lines[2]
    assert len(lines) == 3                                          # the unhelped token is not listed


def test_omlx_measures_on_its_engines_executor_under_a_lease(monkeypatch):
    """backend.lookup_lp: the pool id found by the model's path, the lease held for the measurement, each chunk
    submitted to the engine's MLX executor (another thread)."""
    from concurrent.futures import ThreadPoolExecutor
    from metalscope.engines import mlx as mlx_engine
    executor = ThreadPoolExecutor(1, thread_name_prefix="mlx-engine")
    held, threads = [], []

    @asynccontextmanager
    async def acquire(mid):
        held.append(mid)
        yield SimpleNamespace(_engine=SimpleNamespace(engine=SimpleNamespace(_mlx_executor=executor)))
        held.append("released")
    pool = SimpleNamespace(get_model_ids=lambda: ["a", "flash"], acquire=acquire,
                           get_entry=lambda m: SimpleNamespace(model_path=f"/models/{m}"))
    monkeypatch.setattr(omlx_backend, "_SERVER", SimpleNamespace(get_server_state=lambda: SimpleNamespace(engine_pool=pool)))

    class Engine:
        def __init__(self, model, tok, name):
            assert name == "flash"

        def lookup_chunk(self):
            return 512

        def next_logprobs(self, ids, start, on, run, rows):
            return run(lambda: threads.append(threading.current_thread().name) or np.zeros(len(ids) - start))
    monkeypatch.setattr(mlx_engine, "MlxEngine", Engine)
    tracer = SimpleNamespace(name="/models/flash", model=object(), tokenizer=object())
    got = asyncio.run(omlx_backend.lookup_lp(tracer, [1, 2, 3], 1, False, lookup.Run("t1")))
    assert len(got) == 2 and held == ["flash", "released"] and threads[0].startswith("mlx-engine")
    with pytest.raises(RuntimeError, match="no model at"):
        asyncio.run(omlx_backend.lookup_lp(SimpleNamespace(name="/elsewhere"), [1, 2], 1, True, lookup.Run("t1")))
    executor.shutdown()


@pytest.mark.parametrize("free_gb, pool_gb, on, refused", [(3, 0, True, True), (3, 4, True, False),
                                                           (40, 0, True, False), (None, 0, True, False),
                                                           (3, 0, False, False)])
def test_omlx_refuses_an_inspection_that_would_not_fit_in_free_memory(monkeypatch, free_gb, pool_gb, on, refused):
    """Before its first pass, lookup_lp prices the reading's own cache and attention as oMLX's memory guard prices a
    prompt that long, adds its chunk's logits and a 2 GB margin, and refuses it with one line when free memory and
    MLX's buffer pool (which the reading takes from first: 99 GB resident leaves little else free) fall short, nothing
    run. Free memory unknown, it runs; the second pass reuses the first's memory and is not priced."""
    import mlx.core as mx
    from concurrent.futures import ThreadPoolExecutor
    from metalscope.engines import mlx as mlx_engine
    executor, priced, ran = ThreadPoolExecutor(1), [], []
    monitor = SimpleNamespace(estimate_prefill_peak_bytes=lambda n, chunk: priced.append((n, chunk)) or n * 24_600)
    core = SimpleNamespace(_mlx_executor=executor, scheduler=SimpleNamespace(memory_monitor=monitor))

    @asynccontextmanager
    async def acquire(mid):
        yield SimpleNamespace(_engine=SimpleNamespace(engine=core))
    pool = SimpleNamespace(get_model_ids=lambda: ["flash"], acquire=acquire,
                           get_entry=lambda m: SimpleNamespace(model_path="/models/flash"))
    monkeypatch.setattr(omlx_backend, "_SERVER", SimpleNamespace(get_server_state=lambda: SimpleNamespace(engine_pool=pool)))

    class Engine:
        def __init__(self, *a):
            pass

        def lookup_chunk(self):
            return 512

        def chunk_bytes(self, rows):
            return rows * 248_000 * 8

        def next_logprobs(self, ids, start, on, run, rows):
            return run(lambda: ran.append(1) or np.zeros(len(ids) - start))
    monkeypatch.setattr(mlx_engine, "MlxEngine", Engine)
    monkeypatch.setattr(gpulock, "available", lambda: None if free_gb is None else free_gb * gpulock.GB)
    monkeypatch.setattr(mx, "get_cache_memory", lambda: pool_gb * gpulock.GB)
    tracer = SimpleNamespace(name="/models/flash", model=object(), tokenizer=object())
    ask = lambda: asyncio.run(omlx_backend.lookup_lp(tracer, list(range(100_000)), 1, on, lookup.Run("t1")))
    if refused:      # its cache (100k tokens at 24.6 KB) and logits, 3.2 GB, and the 2 GB margin
        with pytest.raises(lookup.Refused, match=r"^not enough free memory: this needs about 5\.2 GB, 3\.0 GB is free$"):
            ask()
    else:
        ask()
    assert priced == ([(100_000, 512)] if on else []) and bool(ran) != refused
    executor.shutdown()


def test_omlx_waits_for_its_pass_when_the_ask_is_cancelled(monkeypatch):
    """The client went mid-pass: the lease (and the route's locks) are let go only once the chunk running on the
    engine's executor is done, never while the model is still being read."""
    from concurrent.futures import ThreadPoolExecutor
    from metalscope.engines import mlx as mlx_engine
    executor, started, order = ThreadPoolExecutor(1), threading.Event(), []

    @asynccontextmanager
    async def acquire(mid):
        try:
            yield SimpleNamespace(_engine=SimpleNamespace(engine=SimpleNamespace(_mlx_executor=executor)))
        finally:
            order.append("released")
    pool = SimpleNamespace(get_model_ids=lambda: ["flash"], acquire=acquire,
                           get_entry=lambda m: SimpleNamespace(model_path="/models/flash"))
    monkeypatch.setattr(omlx_backend, "_SERVER", SimpleNamespace(get_server_state=lambda: SimpleNamespace(engine_pool=pool)))

    def chunk():
        started.set()
        threading.Event().wait(0.3)
        order.append("chunk done")

    class Engine:
        def __init__(self, *a):
            pass

        def lookup_chunk(self):
            return 512

        def next_logprobs(self, ids, start, on, run, rows):
            return run(chunk)
    monkeypatch.setattr(mlx_engine, "MlxEngine", Engine)
    tracer = SimpleNamespace(name="/models/flash", model=object(), tokenizer=object())

    async def ask_then_leave():
        task = asyncio.create_task(omlx_backend.lookup_lp(tracer, [1, 2, 3], 1, False, lookup.Run("t1")))
        await asyncio.to_thread(started.wait)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(ask_then_leave())
    assert order == ["chunk done", "released"]
    executor.shutdown()


def test_ds4_says_when_ds4_server_does_not_answer(tmp_path, monkeypatch):
    """No ds4-server behind the proxy (even when making the engine): a 503 that says so, not a 500."""
    import httpx
    from fastapi.testclient import TestClient
    from metalscope.ds4 import proxy
    from tests.test_ds4 import PROXY, tracer

    class Down:
        def __init__(self, *a):
            raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(proxy, "Ds4Engine", Down)
    t = tracer("flash")
    MANAGER.tracers[id(t)] = t
    try:
        saved()
        app = proxy.build_app(proxy.Served("flash", tmp_path / "x.gguf", "http://ds4", t),
                              transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        with TestClient(app, base_url=PROXY, client=("127.0.0.1", 50000)) as client:
            r = client.post("/scope/api/lookup", json=dict(trace="t1"))
        assert r.status_code == 503 and "ds4-server did not answer (ConnectError" in r.json()["detail"], r.text
        assert gpulock.held() is None and not lookup.ONE.locked()
    finally:
        MANAGER.tracers.pop(id(t), None)


def test_ds4_tells_each_prefill_chunk_and_hands_the_engine_the_stop(tmp_path, monkeypatch):
    """The proxy's measure: each prefill chunk the engine sees ends (``fed``: rows read, the chunk's rows) goes to the
    asking client as the pass's share done and the share after the next chunk; the engine holds the ask's stop."""
    import httpx
    from fastapi.testclient import TestClient
    from metalscope.ds4 import proxy
    from tests.test_ds4 import PROXY, tracer
    stops, told = [], []

    class Engine:
        def __init__(self, *a):
            pass

        def next_logprobs(self, ids, start, on, stop, fed):
            stops.append(stop)
            fed(4, 4)
            return NextLp(np.zeros(len(ids) - start, np.float32), *top_of(ids, start, on))
    monkeypatch.setattr(proxy, "Ds4Engine", Engine)
    monkeypatch.setattr(MANAGER.hub, "publish", told.append)
    t = tracer("flash")
    MANAGER.tracers[id(t)] = t
    try:
        saved()
        app = proxy.build_app(proxy.Served("flash", tmp_path / "x.gguf", "http://ds4", t),
                              transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        with TestClient(app, base_url=PROXY, client=("127.0.0.1", 50000)) as client:
            r = client.post("/scope/api/lookup", json=dict(trace="t1", client="pi-1", ask="a1"))
        assert r.status_code == 200, r.text
        assert [(e["reading"], e["done"], e["next"]) for e in told] == [(0, 0, 1), (0, round(4 / 7, 4), 1), (1, 0, 1),
                                                                      (1, round(4 / 7, 4), 1)]
        assert len(stops) == 2 and stops[0] is stops[1] and not stops[0].is_set()
    finally:
        MANAGER.tracers.pop(id(t), None)


def test_where_the_passes_are_goes_to_the_asking_client_on_the_live_feed(served, monkeypatch):
    """Each pass's start (and an MLX chunk's: test_omlx_tells_each_chunk_and_stops_between_chunks) goes to the client
    that asked, with the seconds it is expected to take from the model's pace; a cached answer and an ask without a
    client tell nothing."""
    c, _, _ = served
    saved()
    told = []
    monkeypatch.setattr(MANAGER.hub, "publish", told.append)
    monkeypatch.setattr(lookup, "PACES", {"flash": 0.5})
    assert c.post("/scope/api/lookup", json=dict(trace="t1", client="pi-1", ask="a1")).status_code == 200
    assert [(e["type"], e["client"], e["ask"], e["reading"], e["done"], e["next"]) for e in told] == \
        [("lookup", "pi-1", "a1", 0, 0.0, 1.0), ("lookup", "pi-1", "a1", 1, 0.0, 1.0)]
    assert told[0]["expect"] == 3.5 and told[0]["elapsed"] == 0              # 7 tokens at 0.5 s a token
    told.clear()
    assert c.post("/scope/api/lookup", json=dict(trace="t1", client="pi-1", ask="a2")).status_code == 200
    assert told == [] and lookup.RUN is None                                 # kept: no pass, nothing told
    lookup.Run("t2").begin(0, 1.0)
    assert told == []                                                        # no client: nobody to tell


def test_the_pace_is_the_models_newest_long_measurements(served, monkeypatch):
    """The seconds a token a pass: the newest kept answer of PACE_FROM tokens or more for that model (a short one is
    mostly overhead), PACE without one; then each long measurement's own."""
    monkeypatch.setattr(lookup, "PACES", {})
    assert lookup.pace("flash", Kind()) == lookup.PACE
    monkeypatch.setattr(lookup, "PACES", {})
    for t, (tid, n, s) in enumerate([("old", 2000, 8.0), ("long", 4000, 4.0), ("short", 33, 3.0)]):  # newest last
        path = saved(tid, time=2.0 + t)
        (path / lookup.FILE).write_text(json.dumps(dict(version=lookup.VERSION, prompt_tokens=n - 3, ids=[1, 2, 3],
                                                        seconds=dict(on=s, off=s))))
    assert lookup.pace("flash", Kind()) == pytest.approx(4.0 / 4000) and lookup.pace("other", Kind()) == lookup.PACE
    c, _, _ = served
    saved()
    monkeypatch.setattr(lookup, "PACE_FROM", 5)
    assert c.post("/scope/api/lookup", json=dict(trace="t1")).status_code == 200
    assert lookup.PACES["flash"] < 0.1                                       # the fake's own, measured just now


def test_called_off_between_the_passes_nothing_is_kept_and_the_locks_are_free(served):
    """POST {trace, ask, stop}: the ask's measurement stops before its second pass, answers 409 and keeps nothing;
    another ask's or another trace's stop does nothing. Asked again, it is measured whole."""
    c, _, calls = served
    path = saved()
    stops = []

    async def measure(tracer, ids, start, on, job):
        calls.append(on)
        stops.extend(lookup.stop(t, a) for t, a in [("t1", "other"), ("t2", "a1"), ("t1", "a1")])
        return NextLp(np.zeros(len(ids) - start, np.float32), *top_of(ids, start, on))
    fixture = c.app.state.metalscope_backend
    c.app.state.metalscope_backend = dataclasses.replace(fixture, lookup=measure)
    r = c.post("/scope/api/lookup", json=dict(trace="t1", ask="a1"))
    assert r.status_code == 409 and "called off" in r.json()["detail"]
    assert calls == [True] and stops == [False, False, True]                 # the second pass never ran
    assert not (path / lookup.FILE).exists() and not lookup.ONE.locked() and gpulock.held() is None
    assert lookup.RUN is None
    assert c.post("/scope/api/lookup", json=dict(trace="t1", ask="a1", stop=True)).json() == {"stopping": False}
    c.app.state.metalscope_backend = fixture
    got = c.post("/scope/api/lookup", json=dict(trace="t1", ask="a2"))
    assert got.status_code == 200 and json.loads((path / lookup.FILE).read_text()) == got.json()


def test_a_second_ask_right_after_a_stop_waits_for_the_first_to_end(served):
    """The first ask's pass runs on after its stop (ds4 ends a dropped request at its next prefill chunk): a second ask
    waits for it to end, then measures whole; the first gets 409, and only the second's answer is kept."""
    c, _, calls = served
    path = saved()
    inside, go_on = threading.Event(), threading.Event()

    async def slow(tracer, ids, start, on, job):
        calls.append(on)
        if len(calls) == 1:
            inside.set()
            await asyncio.to_thread(go_on.wait, 5)
        return NextLp(np.full(len(ids) - start, -0.1 * len(calls), np.float32), *top_of(ids, start, on))
    c.app.state.metalscope_backend = dataclasses.replace(c.app.state.metalscope_backend, lookup=slow)
    first = {}
    t = threading.Thread(target=lambda: first.update(r=c.post("/scope/api/lookup", json=dict(trace="t1", ask="a1"))))
    t.start()
    assert inside.wait(5)
    assert c.post("/scope/api/lookup", json=dict(trace="t1", ask="a1", stop=True)).json() == {"stopping": True}
    threading.Timer(0.3, go_on.set).start()
    second = c.post("/scope/api/lookup", json=dict(trace="t1", ask="a2"))
    t.join(5)
    assert first["r"].status_code == 409 and second.status_code == 200, (first["r"].text, second.text)
    assert calls == [True, True, False] and second.json()["lp_on"] == pytest.approx([-0.2] * 3)
    assert json.loads((path / lookup.FILE).read_text()) == second.json() and not lookup.ONE.locked()


def test_an_ask_waits_only_for_a_called_off_measurement_and_not_for_ever(monkeypatch):
    async def go():
        lookup.ONE.acquire()
        run = lookup.RUN = lookup.Run("t1")
        mine = lookup.Run("t1")
        try:
            assert await lookup.take(mine) is False                          # under way, not called off: busy
            run.stop.set()
            asyncio.get_running_loop().call_later(0.2, lookup.ONE.release)
            assert await lookup.take(mine) is True and lookup.RUN is mine    # waited for it
            monkeypatch.setattr(lookup, "STOP_WAIT_S", 0.1)
            lookup.RUN = run
            assert await lookup.take(mine) is False                          # one that never ends: busy in the end
            mine.stop.set()
            with pytest.raises(lookup.Stopped):                              # called off itself while waiting
                await lookup.take(mine)
        finally:
            lookup.ONE.release()
            lookup.RUN = None
    asyncio.run(go())


def slow_first(c, calls):
    """The fixture's measure, its first pass held until the returned event is set (``inside`` once it runs)."""
    inside, go_on = threading.Event(), threading.Event()

    async def slow(tracer, ids, start, on, job):
        calls.append(on)
        if len(calls) == 1:
            inside.set()
            await asyncio.to_thread(go_on.wait, 5)
        return NextLp(np.full(len(ids) - start, -0.1 * len(calls), np.float32), *top_of(ids, start, on))
    c.app.state.metalscope_backend = dataclasses.replace(c.app.state.metalscope_backend, lookup=slow)
    return inside, go_on


def in_thread(c, out, key, **body):
    t = threading.Thread(target=lambda: out.update({key: c.post("/scope/api/lookup", json=body)}))
    t.start()
    return t


def test_a_stop_for_an_ask_still_waiting_for_the_one_before_reaches_it(served):
    """g, g (A stopped), g (B waits for A to end), g (B stopped while it waits): B never measures, both are 409, and
    the locks are free; a stop with no ask (doctor --live's measurement has none) matches nothing."""
    c, _, calls = served
    path = saved()
    inside, go_on = slow_first(c, calls)
    got = {}
    a = in_thread(c, got, "a", trace="t1", ask="a")
    assert inside.wait(5)
    assert c.post("/scope/api/lookup", json=dict(trace="t1", stop=True)).json() == {"stopping": False}
    assert c.post("/scope/api/lookup", json=dict(trace="t1", ask="a", stop=True)).json() == {"stopping": True}
    b = in_thread(c, got, "b", trace="t1", ask="b")
    for _ in range(100):
        if ("t1", "b") in lookup.ASKS:
            break
        threading.Event().wait(0.02)
    assert c.post("/scope/api/lookup", json=dict(trace="t1", ask="b", stop=True)).json() == {"stopping": True}
    go_on.set()
    a.join(5), b.join(5)
    assert (got["a"].status_code, got["b"].status_code) == (409, 409) and "called off" in got["b"].json()["detail"]
    assert calls == [True] and not (path / lookup.FILE).exists()
    assert not lookup.ONE.locked() and gpulock.held() is None and lookup.RUN is None and lookup.ASKS == {}


def test_the_same_ask_again_while_it_runs_waits_for_its_answer(served):
    """Pi asks again past its client's 300 s limit: the second POST of the ask waits for the run under way and gets
    its answer, measured once; called off, both are 409."""
    c, _, calls = served
    path = saved()
    inside, go_on = slow_first(c, calls)
    got = {}
    first = in_thread(c, got, "first", trace="t1", ask="a1")
    assert inside.wait(5)
    again = in_thread(c, got, "again", trace="t1", ask="a1")
    threading.Event().wait(0.3)
    assert "again" not in got                                                # waiting
    go_on.set()
    first.join(5), again.join(5)
    assert got["first"].status_code == got["again"].status_code == 200 and got["first"].json() == got["again"].json()
    assert calls == [True, False] and json.loads((path / lookup.FILE).read_text()) == got["again"].json()
    (path / lookup.FILE).unlink()
    calls.clear()
    inside, go_on = slow_first(c, calls)
    first = in_thread(c, got, "first", trace="t1", ask="a2")
    assert inside.wait(5)
    again = in_thread(c, got, "again", trace="t1", ask="a2")
    threading.Event().wait(0.2)
    assert c.post("/scope/api/lookup", json=dict(trace="t1", ask="a2", stop=True)).json() == {"stopping": True}
    go_on.set()
    first.join(5), again.join(5)
    assert got["first"].status_code == got["again"].status_code == 409
    assert got["again"].json()["detail"] == "the engram inspection was called off" and calls == [True]
    assert lookup.ASKS == {} and not lookup.ONE.locked()
    # Sent again after its run ended (``again``): the kept answer or a 409, never a run of its own.
    r = c.post("/scope/api/lookup", json=dict(trace="t1", ask="a2", again=True))
    assert r.status_code == 409 and calls == [True] and not (path / lookup.FILE).exists()


def test_the_same_ask_again_gets_its_runs_refusal_as_it_was(served):
    c, _, calls = served
    saved()
    inside, go_on = threading.Event(), threading.Event()

    async def broken(tracer, ids, start, on, job):
        inside.set()
        await asyncio.to_thread(go_on.wait, 5)
        raise RuntimeError("ds4-server read no next-token log-probabilities")
    c.app.state.metalscope_backend = dataclasses.replace(c.app.state.metalscope_backend, lookup=broken)
    got = {}
    first = in_thread(c, got, "first", trace="t1", ask="a1")
    assert inside.wait(5)
    again = in_thread(c, got, "again", trace="t1", ask="a1", again=True)
    threading.Event().wait(0.2)
    go_on.set()
    first.join(5), again.join(5)
    assert got["first"].status_code == got["again"].status_code == 503
    assert got["first"].json() == got["again"].json()


def test_an_ask_is_known_by_its_trace_too_and_from_its_start(served, monkeypatch):
    """The same ask string for another trace leaves the first's stop working; a stop that comes while the ask still
    reads the cache reaches it (registered before the route's first wait); a failing GPU lock release frees ONE."""
    c, _, calls = served
    saved()
    saved("t2", time=2.0)
    inside, go_on = slow_first(c, calls)
    got = {}
    first = in_thread(c, got, "first", trace="t1", ask="x")
    assert inside.wait(5)
    other = c.post("/scope/api/lookup", json=dict(trace="t2", ask="x"))
    assert other.status_code == 409 and "under way" in other.json()["detail"]
    assert c.post("/scope/api/lookup", json=dict(trace="t1", ask="x", stop=True)).json() == {"stopping": True}
    go_on.set()
    first.join(5)
    assert got["first"].status_code == 409 and lookup.ASKS == {}
    calls.clear()
    reading, go_on = threading.Event(), threading.Event()
    cached = lookup.cached

    def slow_cache(entry):
        reading.set()
        go_on.wait(5)
        return cached(entry)
    monkeypatch.setattr(lookup, "cached", slow_cache)
    late = in_thread(c, got, "late", trace="t1", ask="y")
    assert reading.wait(5)
    assert c.post("/scope/api/lookup", json=dict(trace="t1", ask="y", stop=True)).json() == {"stopping": True}
    go_on.set()
    late.join(5)
    assert got["late"].status_code == 409 and "called off" in got["late"].json()["detail"] and calls == []
    monkeypatch.setattr(lookup, "cached", cached)

    real = gpulock.GpuLock.release

    def fails(self):
        real(self)
        raise OSError("the lock file is gone")
    monkeypatch.setattr(gpulock.GpuLock, "release", fails)
    with pytest.raises(OSError):
        c.post("/scope/api/lookup", json=dict(trace="t2", ask="z"))
    assert not lookup.ONE.locked() and lookup.RUN is None and lookup.ASKS == {}


def test_omlx_tells_each_chunk_and_stops_between_chunks(monkeypatch):
    """lookup_lp: before each chunk the pass's share done and the share after it go to the job; called off, the next
    chunk never runs, the lease let go after the one running; the expert slots given back mid-pass (they no longer hold
    a chunk's rows), the reading is refused before the next chunk."""
    from concurrent.futures import ThreadPoolExecutor
    from metalscope.engines import mlx as mlx_engine
    executor, order, told = ThreadPoolExecutor(1), [], []

    @asynccontextmanager
    async def acquire(mid):
        try:
            yield SimpleNamespace(_engine=SimpleNamespace(engine=SimpleNamespace(_mlx_executor=executor)))
        finally:
            order.append("released")
    pool = SimpleNamespace(get_model_ids=lambda: ["flash"], acquire=acquire,
                           get_entry=lambda m: SimpleNamespace(model_path="/models/flash"))
    monkeypatch.setattr(omlx_backend, "_SERVER", SimpleNamespace(get_server_state=lambda: SimpleNamespace(engine_pool=pool)))
    job = lookup.Run("t1")
    monkeypatch.setattr(job, "tell", lambda: told.append((job.done, job.next)))

    class Engine:
        def __init__(self, *a):
            pass

        def lookup_chunk(self):                   # the rows its expert slots hold
            return slots[0]

        def next_logprobs(self, ids, start, on, run, rows):
            return [run(lambda k=k: order.append(k)) for k in range(0, len(ids), rows)]
    slots = [2]
    monkeypatch.setattr(mlx_engine, "MlxEngine", Engine)
    tracer = SimpleNamespace(name="/models/flash", model=object(), tokenizer=object())
    asyncio.run(omlx_backend.lookup_lp(tracer, list(range(6)), 1, True, job))
    assert order == [0, 2, 4, "released"] and told == [(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1)]
    order.clear()
    monkeypatch.setattr(job, "tell", lambda: job.stop.set())                 # called off as the first chunk starts
    with pytest.raises(lookup.Stopped):
        asyncio.run(omlx_backend.lookup_lp(tracer, list(range(6)), 1, False, job))
    assert order == [0, "released"]                                          # that chunk ran; the next did not
    order.clear()
    job = lookup.Run("t2")
    monkeypatch.setattr(job, "tell", lambda: job.done > 0 and slots.__setitem__(0, 1))  # oMLX shrinks after chunk 1
    with pytest.raises(lookup.Refused, match="ask again") as got:
        asyncio.run(omlx_backend.lookup_lp(tracer, list(range(6)), 1, True, job))
    assert got.value.status == 409 and order == [0, "released"]
    executor.shutdown()



V41 = Kind(ngram=4, row_s=0.07, alone=2, pace=0.01)     # V4.1's, at most 2 conversation rows fed alone


def test_v41s_answer_names_its_4_grams_and_its_rows_fed_alone_are_expected(served, monkeypatch):
    """DeepSeek V4.1 on ds4 (lookup.Kind: 4-grams, rows fed one at a time): the answer says 4-grams (the dock marks
    the 4 tokens before a helped one, the table too); a pass is expected to take the model's pace for the rows read in
    chunks and row_s for each fed alone (the reply's, the prompt's last, and as a worst case ``alone`` before them);
    the pace learnt leaves those out; a model never measured starts at its own pace."""
    c, tracer, _ = served
    tracer.lookup_kind = V41
    saved()
    told = []
    monkeypatch.setattr(MANAGER.hub, "publish", told.append)
    monkeypatch.setattr(lookup, "PACES", {"flash": 0.5})
    got = c.post("/scope/api/lookup", json=dict(trace="t1", client="pi-1", ask="a1")).json()
    # 7 tokens from 4: row 0 in a chunk, rows 1-2 perhaps alone, row 3 (the prompt's last) and 4-6 (the reply) alone
    assert got["ngram"] == 4 and got["version"] == 3 and told[0]["expect"] == pytest.approx(1 * 0.5 + 6 * 0.07)
    lines = lookup.table(dict(got, lp_off=[-3.0, -0.1, -0.1]), [f"<{i}>" for i in PROMPT + REPLY])
    assert "'<5><6><7><8>' -> '<9>'" in lines[1]
    assert lookup._pace(dict(on=10.0, off=12.0, swept=[9.0, 900]), 1000, V41) == 0.01      # its chunks' own
    assert lookup._pace(dict(on=10.0, off=12.0, swept=[0.0, 0]), 1000, V41) is None        # none: not learnt
    assert lookup._pace(dict(on=10.0, off=12.0), 1000, lookup.Kind()) == 11.0 / 1000       # Flash Next: as before
    assert lookup.share(V41, "flash", 7, 4, 2) == pytest.approx((0.5 + 0.07) / (0.5 + 6 * 0.07))
    assert lookup.share(lookup.Kind(), "flash", 7, 4, 4) == pytest.approx(4 / 7)  # Flash Next: rows all cost the same
    monkeypatch.setattr(lookup, "PACES", {})
    assert lookup.pace("never-measured", V41) == 0.01 and lookup.pace("other", lookup.Kind()) == lookup.PACE


def test_an_inspection_expected_to_pass_the_limit_is_refused_in_one_line(served):
    """Where rows go one at a time (V4.1), both readings' whole expected time (the conversation read at the model's
    pace too) is held against INSPECT_MAX_S, below the dock's 15-minute wait."""
    c, tracer, calls = served
    saved()
    lookup.PACES["flash"] = lookup.INSPECT_MAX_S / 2              # row 0 alone takes both readings to the limit
    try:
        tracer.lookup_kind = V41
        r = c.post("/scope/api/lookup", json=dict(trace="t1"))
        assert r.status_code == 409 and r.json()["detail"].startswith("too long to inspect on this model: about 10 min")
        assert calls == [] and not lookup.ONE.locked() and gpulock.held() is None
        tracer.lookup_kind = lookup.Kind()                         # a model read in chunks: as before, never refused
        assert c.post("/scope/api/lookup", json=dict(trace="t1")).status_code == 200
    finally:
        lookup.PACES.pop("flash", None)


def test_ds4_names_each_familys_lookup():
    """The ds4 tracer: Qwen3.8's PLE (lookup.Kind's defaults) and V4.1's Engram (4-grams, rows fed one at a time) both
    have the cap's lookup; an unknown family none."""
    from metalscope.ds4.backend import Ds4Tracer
    from tests.test_ds4 import tracer
    qwen, v41 = tracer("flash"), Ds4Tracer("v41", None, "deepseek41", 43, 4, 8, 6, 256)
    assert qwen.lookup and qwen.lookup_kind == lookup.Kind()
    assert v41.lookup and v41.lookup_kind == lookup.Kind(ngram=4, row_s=0.07, alone=1023, pace=0.01)
    assert lookup.caps([v41]) == frozenset({"lookup"})
    assert not Ds4Tracer("x", None, "other", 2, 1, 8, 2, 4).lookup


def test_ds4s_progress_is_told_at_most_4_times_a_second_unless_it_moved_5_percent(monkeypatch):
    """Run.fed (ds4's path): every row read alone would be 14 events a second; Run.part (oMLX's chunks) always tells."""
    told = []
    job = lookup.Run("t1")
    monkeypatch.setattr(job, "tell", lambda: told.append(round(job.done, 2)))
    job.plan = (lookup.Kind(), "m", 100, 50)
    monkeypatch.setattr(lookup, "PACES", {"m": 0.01})
    job.begin(0, 1.0)
    for r in range(50, 60):
        job.fed(r, 1)
    assert told == [0.0, 0.5, 0.55]                 # the begin, the first, then 5% on
    told.clear()
    job.part(0.6, 0.7)
    job.part(0.61, 0.7)
    assert told == [0.6, 0.61]


def ds4_fake(clock, sweep, alone, calls=None):
    """lookup.Measure as ds4 reads V4.1: the prompt in prefill chunks of 8,192 up to its last row, ``sweep`` s a row,
    then each row alone, ``alone`` s each; the clock moved by each forward and ``fed`` told; stopped, Stopped (ds4-server
    looks every row)."""
    async def run(tracer, ids, start, on, job):
        if calls is not None:
            calls.append(on)
        rows = 0
        while rows < len(ids):
            chunk = min(8192, start - 1 - rows) if rows < start - 1 else 1
            clock[0] += chunk * (sweep if chunk > 1 else alone)
            rows += chunk
            job.fed(rows, chunk)
            if job.stop.is_set():
                raise lookup.Stopped("the request was called off")
        return NextLp(np.zeros(len(ids) - start, np.float32), *top_of(ids, start, on))
    return run


@pytest.mark.parametrize("n,reply", [(3500, 300), (8000, 500), (20000, 800)])
def test_v41s_pace_is_learnt_from_its_chunks_alone(monkeypatch, n, reply):
    """V4.1's pace (LOOKUP's own) is measured on the forwards fed in chunks (Run.fed), not by taking the worst case of
    rows fed alone out of the whole pass: one ordinary inspection must not take it to 0 (the 10-minute guard and the
    progress bar both read it). Kept beside the trace, it is read back the same."""
    from metalscope.ds4.backend import LOOKUP
    kind, clock = LOOKUP["deepseek41"], [0.0]
    monkeypatch.setattr(lookup, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(lookup, "INSPECT_MAX_S", float("inf"))   # the pace only (20,000 tokens are expected past it)
    monkeypatch.setattr(lookup, "PACES", {})
    path, toks = config.home() / "traces" / "long", list(range(n))
    Trace({}, toks, [f"<{i}>" for i in toks], n - reply, {"model": "v41"}).save(path)
    entry = dict(id="long", path=str(path), model="v41", time=1.0)
    append_index(entry)
    tracer = SimpleNamespace(name="v41", lookup_kind=kind, tokenizer=Tracer.tokenizer)
    got = asyncio.run(lookup.measure(entry, tracer, ds4_fake(clock, 0.01, 0.07), lookup.Run("long")))
    assert got["seconds"]["swept"] == [round(2 * (n - reply - 1) * 0.01, 2), 2 * (n - reply - 1)]
    assert lookup.PACES["v41"] == pytest.approx(0.01)
    monkeypatch.setattr(lookup, "PACES", {})
    assert lookup.pace("v41", kind) == pytest.approx(0.01)


def test_an_inspection_that_runs_past_the_limit_is_refused_in_one_line(served, monkeypatch):
    """Expected inside INSPECT_MAX_S, a V4.1 inspection is still held to it as it runs: a first reading over half of
    it, and the second is never asked; either reading past it, stopped (Run.fed) and refused like the estimate."""
    c, tracer, calls = served
    saved()
    tracer.lookup_kind, clock, ran = V41, [0.0], []
    monkeypatch.setattr(lookup, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(lookup, "PACES", {"flash": 0.01})        # expected: about 1 s
    fixture = c.app.state.metalscope_backend
    c.app.state.metalscope_backend = dataclasses.replace(fixture, lookup=ds4_fake(clock, 0.01, 80.0, ran))
    r = c.post("/scope/api/lookup", json=dict(trace="t1"))            # 4 rows alone: 320 s, twice that past 600
    assert r.status_code == 409 and r.json()["detail"] == \
        "too long to inspect on this model: about 11 min for both readings, at most 10"
    assert ran == [True] and not lookup.ONE.locked() and gpulock.held() is None
    ran.clear()
    clock[0] = 0.0
    c.app.state.metalscope_backend = dataclasses.replace(fixture, lookup=ds4_fake(clock, 0.01, 200.0, ran))
    r = c.post("/scope/api/lookup", json=dict(trace="t1"))            # the 3rd row alone ends past 600 s
    assert r.status_code == 409 and r.json()["detail"] == \
        "too long to inspect on this model: over 10 min for both readings, at most 10"
    assert ran == [True] and not lookup.ONE.locked() and gpulock.held() is None
    assert not (config.home() / "traces" / "t1" / lookup.FILE).exists()


def test_an_ask_that_is_not_text_is_answered_400(served):
    c, _, calls = served
    saved()
    for body in (dict(trace="t1", ask=[1], stop=True), dict(trace="t1", ask={"a": 1})):
        r = c.post("/scope/api/lookup", json=body)
        assert r.status_code == 400 and r.json()["detail"] == "ask: text", body
    assert calls == []

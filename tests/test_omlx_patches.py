"""omlx/patches.py's wrappers on stand-ins for oMLX's classes (their signatures oMLX 0.8.0's: test_omlx_contract.py
checks the real ones), what a failed registration leaves, and what oMLX internals that moved cost."""
import asyncio
import sys
import types

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from metalscope import live, steer
from metalscope.live import MANAGER
from metalscope.tracer import ModelTracer
from metalscope import middleware
from metalscope.middleware import REQUEST_CTX, TraceContextMiddleware
from metalscope.omlx import backend, patches, patches_mtp


class RequestOutput(types.SimpleNamespace):
    pass


def stand_ins(ran=None):
    """oMLX's request path, its shape where it matters here: EngineCore.add_request puts the request in the
    scheduler (or raises), stream_outputs / generate drain its outputs and end with _cleanup_request (which raises
    the request's error to the HTTP handler: the 500), the scheduler drops a request without _cleanup_finished when
    its prefill fails, on an abort and after an engine-loop error (fail_all_requests). Chunked prefill:
    _step_prefill_chunk plans one chunk and runs it through ``_run_prefill_chunks``, which _advance_chunked_prefills
    also calls directly (one plan, or several sharing a packed forward) and which runs a row's generation-prompt
    tail through itself again; ``ran`` gets each forward's request ids. Fresh classes each call (patching changes
    them)."""
    class Scheduler:
        model = "model"
        store_pending: set = set()                    # a cancel mid-prefill whose cache store is deferred
        saved_on_abort: set = set()                   # a running request kept for a resend

        def __init__(self):
            self.requests, self.uid_to_request_id, self.detoks = {}, {}, {}

        def _get_detokenizer(self, request_id):          # one a request, made on first use
            return self.detoks.setdefault(request_id, Detok())

        def _cleanup_detokenizer(self, request_id):
            self.detoks.pop(request_id, None)

        def step(self):
            pass

        def _do_external_prefill(self, request, tokens=None, existing_cache=None, vlm_embeds=None):
            pass

        def _step_prefill_chunk(self, state):
            return self._run_prefill_chunks([types.SimpleNamespace(state=state)])[0]

        def _run_prefill_chunks(self, plans):
            ran.append(tuple(p.state.request.request_id for p in plans))
            for p in plans:
                if p.state.tail:                        # stopped at its generation-prompt tail: the rest runs at once
                    p.state.tail = False
                    self._run_prefill_chunks([p])
            return [True] * len(plans)

        def add_request(self, request):
            self.requests[request.request_id] = request

        def _start_helper_prefill(self, request, tokens, cache, sampler, logits_processors):
            return True                                # a Prefill Accelerator takes every request

        def _cleanup_finished(self, finished_ids):
            for rid in finished_ids:
                self.requests.pop(rid, None)

        def _do_abort_request(self, request_id, *, allow_cache_store=True):
            if request_id in self.saved_on_abort:
                return self._cleanup_finished({request_id})
            if request_id not in self.store_pending:  # oMLX keeps it until _drain_pending_async_removes
                self.requests.pop(request_id, None)

        def fail_all_requests(self):
            failed = list(self.requests)
            self.requests.clear()
            return failed

    class GenerationBatch:
        model = "model"

        def __init__(self, uids=()):
            self.uids = list(uids)

        def _step(self):
            return "stepped"

    class EngineCore:
        model = "model"
        refuse = None                                  # add_request raises this (a capacity check, a failed insert)

        def __init__(self):
            self.scheduler = Scheduler()
            self.outputs, self.added, self.skip_store, self.prompts = {}, [], {}, []

        async def add_request(self, prompt, sampling_params=None, request_id=None, images=None, videos=None,
                              skip_cache_store=False, tools=None):
            self.added.append(request_id)
            self.prompts.append(prompt)
            self.skip_store[request_id] = skip_cache_store
            self.outputs[request_id] = []
            if self.refuse is not None:
                self._cleanup_request(request_id)      # oMLX cleans up after a refused insert, then re-raises
                raise self.refuse
            ids = list(prompt) if isinstance(prompt, list) else [1, 2, 3]      # a text prompt: as tokenized
            self.scheduler.add_request(types.SimpleNamespace(request_id=request_id, prompt_token_ids=ids,
                                                             output_token_ids=[]))
            return request_id

        def _cleanup_request(self, request_id):
            self.outputs.pop(request_id, None)

        # what the scheduler does in a step
        def emit(self, rid, finished=False, error=None):
            self.outputs[rid].append(RequestOutput(finished=finished, error=error))

        def finish(self, rid):
            self.scheduler.requests[rid].output_token_ids = [7]
            self.emit(rid, finished=True)
            self.scheduler._cleanup_finished({rid})

        def prefill_fails(self, rid):
            self.scheduler.requests.pop(rid)             # scheduler.py's "Prefill failed for": no _cleanup_finished
            self.emit(rid, finished=True, error="[METAL] Command buffer execution failed: Insufficient Memory")

        async def stream_outputs(self, rid):
            outputs = self.outputs[rid]
            try:
                while outputs:
                    out = outputs.pop(0)
                    yield out
                    if out.error:
                        raise RuntimeError(out.error)
                    if out.finished:
                        break
            finally:
                self._cleanup_request(rid)

        async def generate(self, rid):
            final = self.outputs[rid][-1]
            self._cleanup_request(rid)
            if final.error:
                raise RuntimeError(final.error)
            return final

    return types.SimpleNamespace(Scheduler=Scheduler, GenerationBatch=GenerationBatch, EngineCore=EngineCore)


def omlx(ran=None):
    """A patched stand-in's EngineCore (``.module`` its classes)."""
    module = stand_ins(ran)
    patches.patch_scheduler(module)
    patches.patch_engine_core(module)
    core = module.EngineCore()
    core.module = module
    return core


def engine_core():
    core = omlx()
    return core, core.added


def add(core, ctx, *args):
    token = REQUEST_CTX.set(ctx)
    try:
        return asyncio.run(core.add_request([1, 2, 3], *args))
    finally:
        REQUEST_CTX.reset(token)


@pytest.fixture(autouse=True)
def hooked(monkeypatch):
    """oMLX as fully patched; the stand-in model has a tracer (a steered request on a model without one is refused),
    its hooks in and its steering plans not built."""
    monkeypatch.setattr(patches, "APPLIED", patches.APPLIED | patches._CORE)
    tracer = types.SimpleNamespace(name="m", installed=True, mtp=False, adapter_name="generic",
                                   install=lambda manager: None, lens_step=2, use_lens_step=lambda step: None)
    monkeypatch.setitem(MANAGER.tracers, id("model"), tracer)
    monkeypatch.setattr(MANAGER, "_ensure_plan", lambda rt, t: None)
    return tracer


@pytest.fixture
def failing_register(monkeypatch):
    tried = []

    def register(request_id, *a, **k):
        tried.append(request_id)
        raise ValueError("boom")
    monkeypatch.setattr(MANAGER, "register", register)
    return tried


def test_a_steered_request_that_cannot_register_is_refused(failing_register):
    core, calls = engine_core()
    with pytest.raises(HTTPException) as exc:
        add(core, dict(header=None, meta={}, steer=steer.parse_header("calm@0.5")))
    assert exc.value.status_code == 503 and "could not set up steering: boom" in exc.value.detail
    assert calls == []                               # never scheduled: no unsteered reply under the echoed steer


def test_a_traced_request_that_cannot_register_is_served_untraced(failing_register, monkeypatch, caplog):
    monkeypatch.setattr(patches, "_logged", {})
    core, calls = engine_core()
    rids = [add(core, dict(header="on", meta={}, steer=None)) for _ in range(3)]
    assert calls == rids and not any(map(MANAGER.is_traced, rids))
    assert len([r for r in caplog.records if "served untraced" in r.getMessage()]) == 1     # not a line a request


def test_a_registered_request_is_tracked():
    core, calls = engine_core()
    rid = add(core, dict(header="on", meta={}, steer=None))
    try:
        assert calls == [rid] and MANAGER.is_traced(rid)
    finally:
        MANAGER.requests.pop(rid, None)


class Tracer:
    """A tracer whose install fails: register must leave no entry and announce nothing."""
    def install(self, manager):
        raise RuntimeError("hook")


def test_register_leaves_nothing_when_install_fails(monkeypatch):
    mgr = live.TraceManager()
    started = []
    monkeypatch.setattr(mgr.streamer, "on_register", started.append)
    mgr.tracers[1] = Tracer()
    with pytest.raises(RuntimeError, match="hook"):
        mgr.register("r1", "on", {})
    assert "r1" not in mgr.requests and started == []


def test_a_live_feed_failure_does_not_fail_the_request(monkeypatch):
    mgr = live.TraceManager()

    def broken(rt):
        raise OSError("hub")
    monkeypatch.setattr(mgr.streamer, "on_register", broken)
    mgr.register("r1", "on", {})
    assert mgr.is_traced("r1")


def test_a_failed_install_removes_the_hooks_it_added(monkeypatch):
    closed = []

    class Session:
        hc_count = 1

        def __init__(self, *a, **k):
            pass

        def install(self, model):
            pass

        def set_rows(self, rows):
            pass

        def close(self):
            closed.append(self)

    tracer = ModelTracer(object(), "m", None)
    monkeypatch.setattr(tracer, "_load_lens", lambda: None)
    monkeypatch.setattr(tracer, "_capture_config", lambda: None)
    monkeypatch.setattr("metalscope.tracer.SteerSession", Session)
    monkeypatch.setattr("metalscope.tracer.CaptureSession", Session)

    def adapter_fails(session, model):
        raise RuntimeError("adapter")
    monkeypatch.setattr("metalscope.tracer.install_adapter", adapter_fails)
    mgr = live.TraceManager()
    with pytest.raises(RuntimeError, match="hooks could not be installed on m: RuntimeError: adapter"):
        tracer.install(mgr)
    assert not tracer.installed and tracer.steer is None
    assert len(closed) == 2                          # the steering session and the capture session that failed

    # Kept: later requests get the same error at once, without another install (no lens load per request), each a
    # new exception whose traceback does not grow with the requests.
    monkeypatch.setattr(tracer, "_install", lambda manager: pytest.fail("installed again"))
    raised = []
    for _ in range(3):
        with pytest.raises(RuntimeError, match="adapter") as exc:
            tracer.install(mgr)
        raised.append(exc.value)
    assert raised[0] is not raised[1] and len(list(_frames(raised[2].__traceback__))) == len(list(_frames(raised[0].__traceback__)))
    assert raised[1].__context__ is None             # nothing from an earlier request held
    # A mode switch or an attach never raises it into a request.
    mgr._try_install(tracer)


def _frames(tb):
    while tb is not None:
        yield tb
        tb = tb.tb_next


def test_hooks_removed_during_a_forward_do_not_break_it():
    import mlx.core as mx
    import mlx.nn as nn
    from metalscope.hooks import add_hook

    layer = nn.Linear(2, 2)
    removers = []
    ran = []

    def first(m, args, kwargs, out):
        for r in removers:                           # another thread removes every hook mid-forward
            r()
        ran.append("first")

    removers += [add_hook(layer, first), add_hook(layer, lambda *a: ran.append("second"))]
    layer(mx.zeros((1, 2)))
    assert ran == ["first", "second"]                # the forward's snapshot: no KeyError, no hook skipped
    assert type(layer) is nn.Linear                  # and the class is back


# ------------------------------------------------- every registered request is let go once, on every path

@pytest.fixture
def tracked(monkeypatch):
    """MANAGER with what finish and release did, and nothing tracked before or after the test."""
    seen = dict(finished=[], released=[])
    finish, release = MANAGER.finish, MANAGER.release

    def finishing(rid, *a, **k):
        seen["finished"].append(rid)
        return finish(rid, *a, **k)

    def releasing(rid):
        if release(rid):
            seen["released"].append(rid)
    monkeypatch.setattr(MANAGER, "finish", finishing)
    monkeypatch.setattr(MANAGER, "release", releasing)
    before = set(MANAGER.requests)
    yield seen
    for rid in set(MANAGER.requests) - before:
        MANAGER.requests.pop(rid, None)


TRACED = dict(header="on", meta={}, steer=None)
STEERED = dict(header=None, meta={}, steer=steer.parse_header("calm@0.5"))


def start(core, ctx=TRACED):
    rid = add(core, ctx)
    assert MANAGER.is_traced(rid)
    return rid


def drain(core, rid):
    async def go():
        return [out async for out in core.stream_outputs(rid)]
    return asyncio.run(go())


def test_a_streamed_request_whose_prefill_fails_is_not_left_active(tracked):
    """The 2026-10-22 leak: three requests failed in prefill (Metal out of memory, a 500 each) and status said
    active 3 until oMLX restarted."""
    core = omlx()
    n = len(MANAGER.requests)
    rid = start(core)
    core.prefill_fails(rid)
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        drain(core, rid)
    assert not MANAGER.is_traced(rid) and len(MANAGER.requests) == n
    assert tracked == dict(finished=[], released=[rid])


def test_a_non_streamed_request_whose_prefill_fails_is_not_left_active(tracked):
    core = omlx()
    rid = start(core)
    core.prefill_fails(rid)
    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        asyncio.run(core.generate(rid))
    assert not MANAGER.is_traced(rid) and tracked == dict(finished=[], released=[rid])


@pytest.mark.parametrize("steered", [False, True])
def test_a_request_refused_after_it_registered_is_not_left_active(tracked, steered):
    core = omlx()
    core.refuse = RuntimeError("prefill capacity")
    ctx = dict(header=None, meta={}, steer=steer.parse_header("calm@0.5")) if steered else TRACED
    with pytest.raises(RuntimeError, match="prefill capacity"):
        add(core, ctx)
    assert len(tracked["released"]) == 1 and not MANAGER.is_traced(tracked["released"][0])
    assert MANAGER.kv_isolated(tracked["released"][0]) == steered    # a steered one keeps its KV guard


def test_a_finished_request_is_saved_and_not_released_too(tracked):
    for streamed in (True, False):
        core = omlx()
        rid = start(core)
        core.emit(rid)
        core.finish(rid)
        drain(core, rid) if streamed else asyncio.run(core.generate(rid))
        assert not MANAGER.is_traced(rid)
        assert tracked["finished"][-1] == rid and rid not in tracked["released"]


def test_a_client_that_leaves_mid_stream_is_released_by_the_abort(tracked):
    core = omlx()
    rid = start(core)
    core.emit(rid)
    core.emit(rid)

    async def leave():
        stream = core.stream_outputs(rid)
        await stream.__anext__()
        await stream.aclose()                        # the client left: the consumer cleans up, the abort is pending
    asyncio.run(leave())
    assert MANAGER.is_traced(rid)                   # the scheduler still holds it: its abort decides
    core.scheduler._do_abort_request(rid)
    assert not MANAGER.is_traced(rid) and tracked == dict(finished=[], released=[rid])
    core.scheduler._do_abort_request(rid)           # a second abort: nothing more
    assert tracked["released"] == [rid]


def test_a_cancel_mid_prefill_with_a_deferred_store_is_released(tracked):
    core = omlx()
    rid = start(core)
    core.scheduler.store_pending = {rid}
    core.scheduler._do_abort_request(rid)
    assert rid in core.scheduler.requests                 # oMLX drains it later, without us
    assert not MANAGER.is_traced(rid) and tracked == dict(finished=[], released=[rid])


def test_an_abort_that_keeps_a_running_request_saves_it_and_releases_nothing(tracked):
    core = omlx()
    rid = start(core)
    core.scheduler.saved_on_abort = {rid}
    core.scheduler._do_abort_request(rid)
    assert tracked == dict(finished=[rid], released=[])


def test_an_engine_loop_error_releases_every_request_it_fails(tracked):
    core = omlx()
    rids = [start(core) for _ in range(3)]
    assert sorted(core.scheduler.fail_all_requests()) == sorted(rids)
    assert not any(MANAGER.is_traced(r) for r in rids) and sorted(tracked["released"]) == sorted(rids)
    for rid in rids:                                # the consumers' cleanup afterwards: no second release
        core._cleanup_request(rid)
    assert len(tracked["released"]) == 3


def test_release_is_once_and_finish_after_it_does_nothing():
    mgr = live.TraceManager()
    mgr.register("r1", "on", {})
    assert mgr.release("r1") is True and mgr.release("r1") is False and not mgr.requests
    mgr.finish("r1", [1], [2])                      # no entry: nothing saved, nothing raised
    assert mgr.saved == 0


def test_a_traced_request_finished_with_nothing_recorded_still_ends_live(home):
    """No forward reached the hooks: nothing is saved, but the live start gets its end (the release after finish
    finds nothing), or the page keeps the request busy."""
    import asyncio
    from tests.helpers import drain
    mgr = live.TraceManager()
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None)
    mgr.register("r1", "on", {})
    mgr.finish("r1", [1, 2], [3])
    assert mgr.release("r1") is False and mgr.saved == 0
    events = drain(loop, q)
    assert [e["type"] for e in events] == ["start", "end"] and events[-1]["trace_id"] is None


def test_server_model_names_resolve_through_omlx(monkeypatch):
    """An alias Pi sends (oMLX's /v1/models lists them) becomes the model id oMLX serves; the name as sent when oMLX
    has no resolver, does not know it, or its resolver raises."""
    monkeypatch.setattr(backend, "_SERVER", None)
    assert backend.resolve_model("qwen") == "qwen"
    monkeypatch.setattr(backend, "_SERVER", types.SimpleNamespace(resolve_model_id={"qwen": "Qwen3-8B-4bit"}.get))
    assert backend.resolve_model("qwen") == "Qwen3-8B-4bit" and backend.resolve_model("other") == "other"

    def broken(name):
        raise RuntimeError("pool")
    monkeypatch.setattr(backend, "_SERVER", types.SimpleNamespace(resolve_model_id=broken))
    assert backend.resolve_model("qwen") == "qwen"


def test_stop_ids_leave_out_the_model_eos_when_the_request_ignores_it():
    """``ignore_eos`` (oMLX 0.7.2's forced-length benchmarks; 0.7.3's chat and completions routes) drops the model's
    own stop set and, in 0.7.3, the output parser's (gpt-oss <|return|> is then decoded through and kept in the
    reply); the request's own still end it."""
    sched = types.SimpleNamespace(_get_stop_tokens=lambda: {1, 2},
                                  _output_parser_factory=types.SimpleNamespace(stop_token_ids={3}))

    def request(**params):
        return types.SimpleNamespace(sampling_params=types.SimpleNamespace(stop_token_ids=[4], **params))
    assert patches.stop_ids(sched, request()) == {1, 2, 3, 4}
    assert patches.stop_ids(sched, request(ignore_eos=False)) == {1, 2, 3, 4}
    assert patches.stop_ids(sched, request(ignore_eos=True)) == {4}
    assert patches.stop_ids(sched, types.SimpleNamespace()) == {1, 2, 3}     # no sampling params at all


def test_a_request_id_omlx_supplies_is_the_one_traced():
    """oMLX's server mints every request's id (a uuid) and passes it in; metalscope keeps it rather than minting
    its own."""
    core, calls = engine_core()
    token = REQUEST_CTX.set(dict(TRACED))
    try:
        rid = asyncio.run(core.add_request([1, 2, 3], None, "5b0d7a2e-0000-4000-8000-000000000000"))
    finally:
        REQUEST_CTX.reset(token)
    try:
        assert rid == calls[0] == "5b0d7a2e-0000-4000-8000-000000000000" and MANAGER.is_traced(rid)
    finally:
        MANAGER.requests.pop(rid, None)


# ------------------------------------------------- prefill chunks, packed prefill and suffixed store ids (oMLX 0.7.3)

def prefill_scheduler():
    ran = []
    return omlx(ran).scheduler, ran


def prefill_state(rid, tail=False):
    return types.SimpleNamespace(request=types.SimpleNamespace(request_id=rid, prompt_token_ids=[1, 2, 3]), tail=tail)


def plan(rid, tail=False):
    return types.SimpleNamespace(state=prefill_state(rid, tail))


@pytest.fixture
def rows(monkeypatch, home):
    """The row selections the scheduler wrappers make (MANAGER.rows_for / clear_rows), and nothing tracked after."""
    seen = []
    monkeypatch.setattr(MANAGER, "rows_for", lambda tracer, rids, phase="decode": seen.append((tuple(rids), phase)))
    monkeypatch.setattr(MANAGER, "clear_rows", lambda tracer: seen.append("clear"))
    before = set(MANAGER.requests)
    yield seen
    for rid in set(MANAGER.requests) - before:
        MANAGER.requests.pop(rid, None)


def test_every_prefill_chunk_of_a_traced_request_is_selected_wherever_omlx_runs_it(rows):
    """Packing-ready models (Qwen3.5/3.6) run chunks through _run_prefill_chunks without _step_prefill_chunk: those
    and the generation-prompt tail were neither recorded nor steered."""
    sched, ran = prefill_scheduler()
    MANAGER.register("t1", "on", {})
    sched._step_prefill_chunk(prefill_state("t1"))
    assert ran == [("t1",)] and rows == [(("t1",), "prefill"), "clear"]
    rows.clear()
    sched._run_prefill_chunks([plan("t1")])                   # _advance_chunked_prefills' own call
    assert rows == [(("t1",), "prefill"), "clear"]
    rows.clear()
    sched._run_prefill_chunks([plan("t1", tail=True)])        # and the tail it runs straight after
    assert ran[-2:] == [("t1",), ("t1",)] and rows == [(("t1",), "prefill"), (("t1",), "prefill"), "clear", "clear"]
    rows.clear()
    sched._run_prefill_chunks([plan("u1")])                   # untraced: nothing selected
    assert rows == []


def test_a_packed_forward_is_left_as_it_runs_and_a_steered_row_in_it_is_marked_partial(rows):
    sched, ran = prefill_scheduler()
    MANAGER.register("s1", None, {}, steer=steer.parse_header("calm@1.0"))
    sched._run_prefill_chunks([plan("s1"), plan("u1")])
    assert ran == [("s1", "u1")] and rows == [] and MANAGER.requests["s1"].steer_partial
    assert MANAGER.requests["s1"].partial == ["a prefill chunk ran packed with other requests, untraced"]
    MANAGER.register("t1", None, {})                          # traced, unsteered: its missing records are said too
    sched._run_prefill_chunks([plan("t1"), plan("u1")])
    assert MANAGER.requests["t1"].partial and not MANAGER.requests["t1"].steer_partial


def test_traced_requests_are_kept_out_of_packed_prefill(rows):
    sched, _ = prefill_scheduler()
    MANAGER.register("t1", "on", {})

    def request(rid, **flag):
        return types.SimpleNamespace(request_id=rid, **flag)
    traced, other = request("t1", packed_prefill_excluded=False), request("u1", packed_prefill_excluded=False)
    for r in (traced, other):
        sched.add_request(r)
    assert traced.packed_prefill_excluded and not other.packed_prefill_excluded
    assert set(sched.requests) == {"t1", "u1"}


def test_traced_and_steered_requests_prefill_here_not_on_a_prefill_accelerator(rows):
    sched, _ = prefill_scheduler()
    MANAGER.register("t1", "on", {})
    MANAGER.register("s1", None, {}, steer=steer.parse_header("calm@1.0"))
    took = {rid: sched._start_helper_prefill(types.SimpleNamespace(request_id=rid), [1, 2], None, None, None)
            for rid in ("t1", "s1", "u1")}
    assert took == {"t1": False, "s1": False, "u1": True}


def test_steered_kv_stored_under_a_suffixed_id_is_refused(home):
    """oMLX stores a request's KV under ``<id>#requeue<n>`` (memory-pressure requeue) and ``<id>#system``
    (0.7.3's shared system-prompt tail, which first fetches the grid under that id): for a steered request both are
    refused, during and after it; a traced or untraced request's go through."""
    MANAGER.register("s1", None, {}, steer=steer.parse_header("calm@1.0"))
    MANAGER.register("t1", "on", {})

    class FakeCache:
        def fetch_cache(self, request_id, tokens, extra_keys=None, *, record_stats=True):
            return "hit", []

        def store_cache(self, request_id, tokens, cache_data, model_cache_config=None, **kw):
            return "stored"

        def fetch_exact_prefix(self, request_id, tokens):
            return "hit"

    patches.patch_prefix_cache(types.SimpleNamespace(BlockAwarePrefixCache=FakeCache))
    c = FakeCache()
    try:
        for _ in range(2):                                    # live, then after the request ended (its tombstone)
            assert c.fetch_cache("s1#system", [1], record_stats=False) == (None, [1])
            assert c.store_cache("s1#system", [1], None, _store_shared_tail=True) is None
            assert c.store_cache("s1#requeue1", [1], None) is None
            MANAGER.release("s1")
        assert c.fetch_cache("t1#system", [1], record_stats=False) == ("hit", [])
        assert c.store_cache("t1#system", [1], None) == "stored" and c.store_cache("u1#requeue1", [1], None) == "stored"
        assert c.fetch_cache("t1", [1]) == (None, [1])         # a traced request's own prefill still skips reuse
    finally:
        MANAGER.requests.pop("t1", None)


def test_mtp_priming_takes_the_candidate_0_7_3_passes(home):
    """0.7.3 publishes a replaced off-grid candidate (the shared system tail) as ``_publish_boundary_candidate(ctx,
    previous)``: a steered request's is still skipped."""
    MANAGER.register("s1", None, {}, steer=steer.parse_header("calm@1.0"))
    calls = []
    pp = types.SimpleNamespace(
        _publish_boundary_candidate=lambda ctx, candidate=None: calls.append((ctx.request_id, candidate)) or "published")
    patches_mtp.patch_mtp_priming(pp)
    try:
        assert pp._publish_boundary_candidate(types.SimpleNamespace(request_id="s1"), "previous") is None
        assert pp._publish_boundary_candidate(types.SimpleNamespace(request_id="u1"), "previous") == "published"
        assert pp._publish_boundary_candidate(types.SimpleNamespace(request_id="u1")) == "published"
        assert calls == [("u1", "previous"), ("u1", None)]
    finally:
        MANAGER.release("s1")


# ------------------------------------------------- what is never tracked, and what a steered request asks of oMLX

def test_an_engine_call_outside_an_api_request_is_never_traced(monkeypatch):
    """oMLX's benchmark and evaluation routes call the engine directly (no middleware context): with tracing on they
    were traced, and lost their prefix reuse."""
    monkeypatch.setattr(MANAGER, "mode", lambda: "on")
    core, calls = engine_core()
    rid = asyncio.run(core.add_request([1, 2, 3], None, "bench-1"))
    assert calls == ["bench-1"] and not MANAGER.is_traced(rid)
    rid = add(core, dict(header=None, meta={}, steer=None))          # an API request under the same mode
    try:
        assert MANAGER.is_traced(rid)
    finally:
        MANAGER.requests.pop(rid, None)


def test_a_steered_request_sets_omlx_skip_cache_store(tracked):
    core = omlx()
    s, t = start(core, STEERED), start(core)
    assert core.skip_store == {s: True, t: False}


def test_a_steered_request_on_a_model_without_hooks_is_refused(monkeypatch):
    core, calls = engine_core()
    monkeypatch.delitem(MANAGER.tracers, id("model"))
    with pytest.raises(HTTPException) as exc:
        add(core, STEERED)
    assert exc.value.status_code == 503 and "no hooks on this model" in exc.value.detail and calls == []


def test_nothing_is_tracked_while_a_core_patch_is_missing(monkeypatch):
    """Without the prefix-cache patch a steered request's KV would be stored for others."""
    core, calls = engine_core()
    monkeypatch.setattr(patches, "APPLIED", {"patch_scheduler", "patch_engine_core"})
    rid = add(core, TRACED)
    assert calls == [rid] and not MANAGER.is_traced(rid)
    with pytest.raises(HTTPException) as exc:
        add(core, STEERED)
    assert "not fully patched (missing patch_prefix_cache)" in exc.value.detail and calls == [rid]


@pytest.mark.parametrize("patch,cls,name", [("patch_scheduler", "Scheduler", "fail_all_requests"),
                                            ("patch_scheduler", "GenerationBatch", "_step"),
                                            ("patch_engine_core", "EngineCore", "_cleanup_request")])
def test_a_patch_is_all_or_nothing_when_omlx_lost_a_name(monkeypatch, patch, cls, name):
    from metalscope.omlx import boot
    monkeypatch.setattr(patches, "APPLIED", set())
    module = stand_ins()
    delattr(getattr(module, cls), name)
    classes = (module.Scheduler, module.GenerationBatch, module.EngineCore)
    before = [dict(vars(c)) for c in classes]
    with pytest.raises(AttributeError, match=name):
        getattr(patches, patch)(module)
    assert [dict(vars(c)) for c in classes] == before and not patches.APPLIED
    # boot logs it and oMLX carries on; status and doctor say which patch is missing
    target = next(mod for mod, fn in boot._TARGETS.items() if fn == patch)
    monkeypatch.setattr(boot, "_supported", True)
    monkeypatch.setitem(sys.modules, target, module)
    boot._run_patch(target, module)
    assert patches.patch_state() == dict(applied=[], missing=[patch])


# ------------------------------------------------- oMLX internals that moved cost the trace, never the request

def test_decode_rows_are_selected_by_the_scheduler_uid_map(rows):
    core = omlx()
    MANAGER.register("t1", "on", {})
    core.scheduler.uid_to_request_id = {1: "t1", 2: "u1"}
    patches._current.scheduler = core.scheduler                  # as inside Scheduler.step
    try:
        assert core.module.GenerationBatch([1, 2])._step() == "stepped"
        assert rows == [(("t1", "u1"), "decode"), "clear"]
        rows.clear()
        assert core.module.GenerationBatch([2])._step() == "stepped" and rows == []      # no tracked row in it
    finally:
        patches._current.scheduler = None


def test_mtp_next_selects_the_rows_once_for_the_whole_call(rows, hooked, monkeypatch):
    core = omlx()
    batch_cls = core.module.GenerationBatch
    batch_cls.next = lambda self: self._step()                   # oMLX's MTP next() runs its forwards itself
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", types.SimpleNamespace(GenerationBatch=batch_cls))
    patches_mtp.patch_mtp_next()
    hooked.mtp = True
    MANAGER.register("t1", "on", {})
    core.scheduler.add_request(types.SimpleNamespace(request_id="t1", prompt_token_ids=[1, 2, 3]))
    core.scheduler.uid_to_request_id = {1: "t1"}
    patches._current.scheduler = core.scheduler
    try:
        assert batch_cls([1]).next() == "stepped"
    finally:
        patches._current.scheduler = None
    assert rows == [(("t1",), "decode"), "clear"] and MANAGER.requests["t1"].prompt_len == 3


def test_a_moved_uid_map_leaves_the_step_running_and_says_so_on_tracked_requests(tracked):
    """It raised AttributeError into Scheduler.step, which fails every request in the batch."""
    core = omlx()
    t, s = start(core), start(core, STEERED)
    del core.scheduler.uid_to_request_id
    patches._current.scheduler = core.scheduler
    try:
        assert core.module.GenerationBatch([1, 2])._step() == "stepped"
    finally:
        patches._current.scheduler = None
    assert MANAGER.requests[t].partial == ["rows: AttributeError: 'Scheduler' object has no attribute 'uid_to_request_id'"]
    assert MANAGER.requests[s].steer_failed == MANAGER.requests[t].partial[0]       # never unsteered without a word


def test_a_moved_scheduler_model_leaves_prefill_and_cleanup_running(tracked):
    core = omlx()
    rid = start(core)
    del core.module.Scheduler.model
    core.scheduler._do_external_prefill(core.scheduler.requests[rid])
    assert "AttributeError" in MANAGER.requests[rid].partial[0]
    core.emit(rid)
    core.finish(rid)                                              # _cleanup_finished: saved without a tracer
    assert tracked == dict(finished=[rid], released=[]) and not MANAGER.is_traced(rid)


def test_a_repeating_failure_logs_at_most_every_30_seconds(monkeypatch, caplog):
    monkeypatch.setattr(patches, "_logged", {})

    def boom():
        raise RuntimeError("moved")
    for _ in range(50):
        assert patches._safe(boom) is None
    assert len([r for r in caplog.records if "boom failed" in r.getMessage()]) == 1


# ------------------------------------------------- the server patch: inside oMLX's body limit

class BodyLimit:
    """oMLX's RequestBodySizeLimitMiddleware, as it behaves: a 413 for a Content-Length over the cap before the app
    reads anything, a chunked body cut with http.disconnect once past it."""

    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length and int(length) > self.max_bytes:
            return await JSONResponse({"error": {"type": "request_too_large"}}, 413)(scope, receive, send)
        total = 0

        async def counting():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    return {"type": "http.disconnect"}
            return message
        await self.app(scope, counting, send)


def test_the_middleware_is_inside_omlx_body_limit(monkeypatch, home):
    """app.add_middleware after oMLX's own put ours outside its limit: with tracing on (or a trace or steer header)
    it read a whole oversized body before oMLX refused it."""
    app, seen, parsed = FastAPI(), [], []

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        seen.append(REQUEST_CTX.get())
        return {"bytes": len(await request.body())}

    app.add_middleware(BodyLimit, max_bytes=64)                  # oMLX's server module adds its limit last
    monkeypatch.setattr(backend, "_SERVER", None)
    monkeypatch.setattr(patches, "patch_uvicorn_shutdown", lambda: None)
    meta = middleware.conversation_meta
    monkeypatch.setattr(middleware, "conversation_meta", lambda body: parsed.append(body) or meta(body))
    patches.patch_server(types.SimpleNamespace(app=app, get_server_state=lambda: types.SimpleNamespace(api_key=None)))
    app.add_middleware(CORSMiddleware, allow_origins=["*"])      # and its CORS at startup, after the patch ran
    assert [m.cls for m in app.user_middleware] == [CORSMiddleware, BodyLimit, TraceContextMiddleware]

    client = TestClient(app, base_url="http://127.0.0.1:8008")
    traced = {"x-scope-trace": "on"}
    assert client.post("/v1/chat/completions", json={"model": "m"}, headers=traced).json() == {"bytes": 13}
    assert seen[-1]["header"] == "on" and parsed == [{"model": "m"}]
    r = client.post("/v1/chat/completions", json={"model": "m", "pad": "x" * 100}, headers=traced)
    assert r.status_code == 413 and len(seen) == 1 and len(parsed) == 1          # refused before ours read a byte
    app.dependency_overrides[middleware._local_only] = lambda: None
    assert "patch_server" in client.get("/scope/api/status").json()["patches"]["applied"]    # the routes are mounted
    with pytest.raises(RuntimeError, match="too late"):                           # a started app takes no middleware
        late = FastAPI()
        TestClient(late).get("/")
        patches.patch_server(types.SimpleNamespace(app=late))
    assert late.user_middleware == [] and not any(r.path.startswith("/scope") for r in late.router.routes)


def run_messages(messages):
    """The middleware on a traced POST whose body arrives as ``messages``: the context the app saw, then every
    message it received."""
    got = []

    async def app(scope, receive, send):
        got.append(REQUEST_CTX.get())
        for _ in messages:
            got.append(await receive())
    queue = iter(messages)

    async def receive():
        return next(queue)
    scope = dict(type="http", method="POST", path="/v1/chat/completions", headers=[(b"x-scope-trace", b"on")])
    asyncio.run(TraceContextMiddleware(app)(scope, receive, None))
    return got


def test_a_body_cut_mid_way_passes_through_untraced():
    """What oMLX's limit hands on for a chunked body past the cap (and a client that left): the app gets the same
    messages and no context; the cut body is not replayed as a whole one."""
    cut = [{"type": "http.request", "body": b'{"model": "m", "pad', "more_body": True}, {"type": "http.disconnect"}]
    assert run_messages(cut) == [None, *cut]
    whole = [{"type": "http.request", "body": b'{"model":', "more_body": True},
             {"type": "http.request", "body": b' "m"}', "more_body": False}]
    ctx, *replayed = run_messages(whole)
    assert ctx["header"] == "on" and ctx["meta"]["model"] == "m" and replayed == whole


# ------------------------------------------------- the edit at one token (omlx/edit.py, metalscope/edit.py)

class EditTok:
    """The served model's tokenizer: 3 opens the thinking, 13 closes it; 31 and 32 are the two halves of "é"; 40 a
    tool call's opener; 50-54 Inkling 2's channel markers."""
    words = {1: "<|im_start|>", 2: "user", 3: "<think>\n", 11: "Hmm", 12: " so", 13: "</think>", 21: " well",
             22: " ok", 23: "Done", 40: "<tool_call>", 50: "<|message_model|>", 51: "<|content_thinking|>",
             52: "<|content_text|>", 53: "<|end_message|>", 54: "<|content_invoke_tool_json|>"}
    raw = {31: b"\xc3", 32: b"\xa9"}

    @classmethod
    def piece(cls, i) -> bytes:
        return cls.raw.get(int(i)) or cls.words.get(int(i), f"<{int(i)}>").encode()

    def decode(self, ids):
        return b"".join(map(self.piece, ids)).decode("utf-8", "replace")


class Detok:
    """oMLX's streaming detokenizer (mlx_lm's BPE one) where it matters at a join: a fresh one drops its first
    piece's leading space, and a character's first byte waits for the next."""

    def __init__(self):
        self.text, self.offset, self._held = "", 0, b""

    def add_token(self, t):
        self._held += EditTok.piece(t)
        s = self._held.decode("utf-8", "replace")
        if s.endswith("�"):
            return
        self.text += s[1:] if not self.text and s.startswith(" ") else s
        self._held = b""

    @property
    def last_segment(self):
        seg, self.offset = self.text[self.offset:], len(self.text)
        return seg


class InklingSession:
    """oMLX's InklingOutputParserSession (adapter/output_parser.py's _InklingChannelSplitter) where an edit meets it:
    stateful, its channel markers out of the text, a thinking block's text between ``<think>`` and ``</think>``, a
    text block's as it is, a tool call's dropped."""
    THINKING, TEXT, END, MODEL, TOOL = (EditTok.words[i] for i in (51, 52, 53, 50, 54))

    def __init__(self):
        self.channel, self.think = None, False

    def process_token(self, t):
        piece, out = EditTok.words.get(int(t), f"<{int(t)}>"), ""
        if piece == self.THINKING:
            out, self.think, self.channel = "" if self.think else "<think>", True, "thinking"
        elif piece == self.TEXT:
            out, self.think, self.channel = "</think>" if self.think else "", False, "text"
        elif piece == self.END:
            if self.channel == "thinking" and self.think:
                out, self.think = "</think>", False
            self.channel = None
        elif piece in (self.MODEL, self.TOOL):
            self.channel = "tool" if piece == self.TOOL else None
        elif self.channel in ("thinking", "text"):
            out = piece
        return types.SimpleNamespace(stream_text=out)


def parser(sched, kind="inkling"):
    """``sched`` given oMLX's output parser of ``kind``: its factory and its sessions, one a request."""
    sched._output_parser_factory = types.SimpleNamespace(kind=kind, create_session=lambda tok: InklingSession())
    sched._output_parser_sessions = {}
    sched._get_output_parser_session = lambda rid: sched._output_parser_sessions.setdefault(rid, InklingSession())
    sched._cleanup_output_parser_session = lambda rid: sched._output_parser_sessions.pop(rid, None)


class EditRig:
    """oMLX's app as an edit meets it: patch_server's middleware before a /v1/completions route that, as oMLX's,
    checks the API key, reads a text prompt, hands it to EngineCore.add_request (the patched stand-in) and streams
    the request's detokenizer pieces (its parser session's, where the scheduler has one: ``parser``). A reply
    (``reply``): prompt 1 2 3, its tokens. ``a(prompt ids)``: the edited token's (tokens, finish); ``b(prompt ids)``:
    the rest's (tokens, an error after them or None). ``seen``: each completion's body and context."""

    def __init__(self, monkeypatch, hooked, a=None, b=None):
        import json
        import uuid

        import numpy as np
        from fastapi.responses import StreamingResponse

        from metalscope import config, vectors
        config.save_settings(mode="off", steer=True)
        vec = {l: vectors.unit(np.random.default_rng(1).standard_normal(8)) for l in (1, 2)}
        vectors.save_vector("m", vectors.Vector("soccer", vec, dict(source="test", hc_count=1)))
        monkeypatch.setattr(hooked, "tokenizer", EditTok(), raising=False)
        self.core = core = omlx()
        self.a = a or (lambda ids: ([21], "length"))
        self.b = b or (lambda ids: ([22, 13, 23], None))
        self.seen = []
        app = FastAPI()

        @app.get("/v1/models")
        async def models(request: Request):
            if request.headers.get("authorization") != "Bearer k":
                return JSONResponse({"error": {"message": "bad key"}}, 401)
            return {"data": [{"id": "m"}]}

        @app.post("/v1/completions")
        async def complete(request: Request):
            if request.headers.get("authorization") != "Bearer k":
                return JSONResponse({"error": {"message": "bad key"}}, 401)
            body = await request.json()
            self.seen.append((body, REQUEST_CTX.get()))
            rid = await core.add_request(body["prompt"], None, str(uuid.uuid4()))
            req = core.scheduler.requests[rid]

            def finish(toks):
                req.output_token_ids = list(toks)
                core.scheduler._cleanup_finished({rid})
            if not body["stream"]:
                toks, reason = self.a(req.prompt_token_ids)
                finish(toks)
                return {"choices": [{"text": "", "finish_reason": reason}]}
            toks, error = self.b(req.prompt_token_ids)

            async def sse():
                parsed = hasattr(core.scheduler, "_output_parser_sessions")
                d = core.scheduler._get_output_parser_session(rid) if parsed else core.scheduler._get_detokenizer(rid)
                for t in toks:
                    if parsed:
                        text = d.process_token(t).stream_text
                    else:
                        d.add_token(t)
                        text = d.last_segment
                    yield f"data: {json.dumps({'choices': [{'text': text, 'finish_reason': None}]})}\n\n"
                if error:
                    core.scheduler.fail_all_requests()
                    yield f"data: {json.dumps({'error': {'message': error, 'type': 'server_error'}})}\n\n"
                else:
                    finish(toks)
                    yield 'data: {"choices":[{"text":"","finish_reason":"stop"}]}\n\n'
                    yield f'data: {{"choices":[],"usage":{{"completion_tokens":{len(toks)}}}}}\n\n'
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        monkeypatch.setattr(backend, "_SERVER", None)
        monkeypatch.setattr(patches, "patch_uvicorn_shutdown", lambda: None)
        patches.patch_server(types.SimpleNamespace(app=app))
        self.client = TestClient(app, base_url="http://127.0.0.1:8008")      # this machine, as Pi sends it
        self.loop = asyncio.new_event_loop()
        self.q = MANAGER.hub.subscribe(self.loop, "pi-e")

    def reply(self, tokens=(11, 12, 13)):
        """A traced reply of prompt 1 2 3 that oMLX finished, its request id."""
        rid = add(self.core, dict(header="on", meta={"client": "pi-e"}, steer=None))
        self.core.scheduler.requests[rid].output_token_ids = list(tokens)
        self.core.scheduler._cleanup_finished({rid})
        return rid

    def chat(self, edit, key="k"):
        head = {"X-Scope-Trace": "on", "X-Scope-Trace-Client": "pi-e", "Authorization": f"Bearer {key}",
                "X-Scope-Edit": edit}
        msg = dict(model="m", messages=[{"role": "user", "content": "hi"}], stream=True, max_tokens=100,
                   temperature=0.6, stream_options={"include_usage": True})
        return self.client.post("/v1/chat/completions", json=msg, headers=head)

    def starts(self):
        self.loop.run_until_complete(asyncio.sleep(0.05))
        out = []
        while not self.q.empty():
            ev = self.q.get_nowait()
            if ev["type"] == "start":
                out.append((ev["rid"], ev.get("edit")))
        return out

    def close(self):
        MANAGER.hub.unsubscribe(self.q)
        self.loop.close()


@pytest.fixture
def edit_rig(monkeypatch, hooked, home):
    made, before = [], set(MANAGER.requests)

    def make(**kw):
        made.append(EditRig(monkeypatch, hooked, **kw))
        return made[-1]
    yield make
    for rig in made:
        rig.close()
    for rid in set(MANAGER.requests) - before:
        MANAGER.requests.pop(rid, None)


def test_an_edit_on_omlx_runs_the_kept_ids_and_the_rest_joins_the_kept_text(edit_rig):
    """A traced reply (prompt 1 2 3, reply 11 12 13), then an edit of its token 1: oMLX's route reads the kept ids'
    text, its key checked, and its engine is given the ids; the edited token is one greedy token with the removal at
    its last row only, the rest a completion from there with no steer that stops before a tool call. The rest's
    text starts after the kept text: its first piece keeps its space. Each is kept as the whole reply, so the
    edited reply is edited at the token the dock shows."""
    from metalscope.edit import TOOL_OPENERS
    from tests.test_ds4 import stream_of

    rig = edit_rig()
    rid = rig.reply()
    r = rig.chat(f"{rid}@1 remove~soccer/1-2")
    assert r.status_code == 200, r.text
    reasoning, content, finish, usage, errors = stream_of(r)
    assert (reasoning, content, finish, errors) == ("Hmm well ok", "Done", "stop", [])
    assert usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    (one, ctx_a), (rest, ctx_b) = rig.seen
    assert rig.core.prompts[-2:] == [[1, 2, 3, 11], [1, 2, 3, 11, 21]]
    assert one["prompt"] == EditTok().decode([1, 2, 3, 11]) and (one["max_tokens"], one["temperature"]) == (1, 0)
    assert ctx_a["steer"].positions == "decode" and ctx_a["steer"].label == "remove~soccer/1-2"
    assert ctx_b["steer"] is None and rest["stop"] == TOOL_OPENERS
    assert (rest["max_tokens"], rest["temperature"], rest["stream"]) == (98, 0.6, True)
    (r0, e0), (ra, ea), (rb, eb) = rig.starts()
    assert (r0, e0) == (rid, None) and (ra, rb) == (ctx_a["rid"], ctx_b["rid"])
    assert ea == dict(of=rid, at=1, steer="remove~soccer/1-2") and eb == dict(of=ra, at=2, steer="remove~soccer/1-2")
    # The edited reply reads 11 21 22 13 23: its token 3 (13) is the rest's.
    assert rig.chat(f"{rb}@3 remove~soccer/1-2").status_code == 200
    assert rig.core.prompts[-2] == [1, 2, 3, 11, 21, 22]


def test_an_edit_on_omlx_takes_a_word_named_by_its_token_id(edit_rig):
    """A lens word that is no name (初步) goes by its id (tok104495, the dock's tokenName): the header stays ASCII and
    the edit runs as any word's."""
    from metalscope import vectors

    rig = edit_rig()
    vectors.save_vector("m", vectors.Vector("tok104495", vectors.load_vector("m", "soccer").vectors,
                                            dict(source="J-lens concept direction for 初步", words=["初步"], token_ids=[104495],
                                                 cut=[], hc_count=1)))
    rid = rig.reply()
    r = rig.chat(f"{rid}@1 remove~tok104495/1-2")
    assert r.status_code == 200, r.text
    (_, ctx_a), (_, ctx_b) = rig.seen
    assert ctx_a["steer"].label == "remove~tok104495/1-2" and ctx_a["steer"].positions == "decode"
    assert rig.core.prompts[-2] == [1, 2, 3, 11]


def test_an_edit_on_omlx_joins_a_character_split_at_the_join_and_ends_where_its_token_did(edit_rig):
    """The edited token is the first byte of "é", the rest's first its second: the client reads "é". An edited
    token that is a stop token (oMLX leaves it out of a reply) ends the reply there, with no rest."""
    from tests.test_ds4 import stream_of

    rig = edit_rig(a=lambda ids: ([31], "length"), b=lambda ids: ([32, 22], None))
    rid = rig.reply()
    assert stream_of(rig.chat(f"{rid}@1 remove~soccer/1-2"))[0] == "Hmmé ok"
    rig.a = lambda ids: ([], "stop")
    n = len(rig.seen)
    r = rig.chat(f"{rid}@1 remove~soccer/1-2")
    assert r.status_code == 200 and len(rig.seen) == n + 1 and stream_of(r)[:3] == ("Hmm", "", "stop")


def test_an_edit_on_omlx_is_refused_where_it_cannot_be_kept_and_an_error_mid_stream_ends_it(edit_rig):
    """Refusals are 4xx: a reply not kept, another model's, a token after a tool call, a bad key (oMLX's route
    checks it). oMLX failing the rest mid-stream: the text held back, its error, a finish and [DONE]; nothing is
    left active."""
    from tests.test_ds4 import stream_of

    rig = edit_rig(b=lambda ids: ([22], "the engine fell over"))
    before = set(MANAGER.requests)
    rid = rig.reply()
    r = rig.chat("nope@1 remove~soccer/1-2")
    assert r.status_code == 409 and "not kept here" in r.json()["error"]["message"]
    MANAGER.kept.put("other", [1, 2, 3], [11, 12], model="another")
    assert "another model's" in rig.chat("other@1 remove~soccer/1-2").json()["error"]["message"]
    tool = rig.reply((11, 40, 12))
    r = rig.chat(f"{tool}@2 remove~soccer/1-2")
    assert r.status_code == 400 and "after a tool call" in r.json()["error"]["message"]
    assert rig.chat(f"{rid}@1 remove~soccer/1-2", key="wrong").status_code == 401
    # Without the key nothing about a reply is answered (kept or not, waited for or not): oMLX's own refusal first.
    n = len(rig.seen)
    assert [rig.chat(f"{x}@1 remove~soccer/1-2", key="wrong").status_code for x in ("nope", tool)] == [401, 401]
    assert len(rig.seen) == n
    r = rig.chat(f"{rid}@1 remove~soccer/1-2")
    reasoning, content, finish, _, errors = stream_of(r)
    assert reasoning == "Hmm well ok" and errors[0]["message"] == "the engine fell over"
    assert finish == "error" and r.text.endswith("data: [DONE]\n\n")
    assert set(MANAGER.requests) <= before


def test_an_edit_on_omlx_is_refused_on_a_model_whose_replies_it_cannot_join(edit_rig):
    """A model oMLX decodes through a protocol parser (DeepSeek V4, gpt-oss, Gemma 4...: its own detokenizer and
    state): the edit is refused before anything runs, in one line, rather than answered with an unjoined rest. So is
    a model with no detokenizer for the request."""
    rig = edit_rig()
    rid = rig.reply()
    rig.core.scheduler._output_parser_factory = types.SimpleNamespace(kind="deepseek_v41")
    r = rig.chat(f"{rid}@1 remove~soccer/1-2")
    assert r.status_code == 400 and patches.NO_EDIT in r.text and "\n" not in patches.NO_EDIT
    assert len(rig.seen) == 1 and rig.core.prompts == [[1, 2, 3]] and rig.core.scheduler.detoks == {}
    del rig.core.scheduler._output_parser_factory
    rig.core.scheduler._get_detokenizer = lambda request_id: None
    assert rig.chat(f"{rid}@1 remove~soccer/1-2").status_code == 400


INKLING_REPLY = [50, 51, 11, 12, 53, 50, 52, 23]      # thinking "Hmm so", then the answer "Done"


def kinds(r) -> list[str]:
    """A chat stream's delta kinds in order, runs merged."""
    import json
    out = []
    for line in r.text.split("\n"):
        if line.startswith("data: {"):
            for c in json.loads(line[6:]).get("choices") or []:
                out += [k for k, v in c["delta"].items() if k != "role" and v and (not out or out[-1] != k)]
    return out


def test_an_edit_on_inkling_seeds_omlxs_parser_session_and_the_rest_goes_on_in_its_channel(edit_rig):
    """Inkling 2 on oMLX: its replies go through oMLX's output parser (kind "inkling": channel markers out of the
    text, the thinking between <think> and </think>). An edit's requests have their parser session seeded with the
    kept tokens: the kept text is the session's, the rest goes on in the channel those tokens left, and the client
    reads the thinking, then the answer, no marker shown and nothing doubled or lost at the join, wherever the reply
    is cut: at its first markers (0, 1, 2), inside the thinking, at its end and in the answer. The edited token here
    is the reply's own, so every edit reads as the reply did."""
    from tests.test_ds4 import stream_of

    R = INKLING_REPLY
    rig = edit_rig(a=lambda ids: ([R[len(ids) - 2]], "length"), b=lambda ids: (R[len(ids) - 2:], None))
    parser(rig.core.scheduler)
    MANAGER.kept.put("ink", [1, 2], R, model="m")                      # a prompt that does not open the thinking
    for at in range(len(R)):
        r = rig.chat(f"ink@{at} remove~soccer/1-2")
        assert r.status_code == 200, r.text
        assert stream_of(r)[:3] == ("Hmm so", "Done", "stop") and kinds(r) == ["reasoning_content", "content"], at
        (_, ctx_a), (_, ctx_b) = rig.seen[-2:]
        fresh = InklingSession()
        assert ctx_a["text"] == "".join(fresh.process_token(t).stream_text for t in R[:at])
        assert ctx_b["rid"] in rig.core.scheduler._output_parser_sessions              # the rest's, seeded
    # The edited token ending the reply: the kept text is the edited token's session's, no marker shown.
    rig.a = lambda ids: ([], "stop")
    MANAGER.kept.put("ink", [1, 2], R, model="m")                      # kept again: the edits above pushed it out
    r = rig.chat("ink@3 remove~soccer/1-2")
    assert r.status_code == 200 and stream_of(r)[:3] == ("Hmm", "", "stop")


def test_an_inkling_edits_parser_session_is_seeded_and_dropped_when_omlx_never_takes_its_request(tracked):
    """The seeded session is the one oMLX's scheduler reads the request's tokens through (it goes on in the thinking);
    its request is marked as having its think opener sent (the kept text holds it); a request oMLX never takes has
    its session dropped; another parser kind is still refused, nothing seeded."""
    def edit_ctx():
        return dict(header="on", meta={}, steer=None, base=([1, 2], [50, 51, 11]), rid=None, text=None,
                    started=asyncio.Event())

    core = omlx()
    parser(core.scheduler)
    ctx = edit_ctx()
    rid = add(core, ctx)
    s = core.scheduler._output_parser_sessions[rid]
    assert ctx["text"] == "<think>Hmm" and core.scheduler.detoks == {}
    assert [s.process_token(t).stream_text for t in (12, 53, 50, 52, 23)] == [" so", "</think>", "", "", "Done"]
    assert core.scheduler.requests[rid].think_prefix_sent is True
    plain = add(core, dict(header="on", meta={}, steer=None))                       # not an edit: oMLX's own
    assert not hasattr(core.scheduler.requests[plain], "think_prefix_sent")
    core.refuse = RuntimeError("queue full")
    with pytest.raises(RuntimeError, match="queue full"):
        add(core, edit_ctx())
    assert list(core.scheduler._output_parser_sessions) == [rid]
    core.refuse = None
    core.scheduler._output_parser_factory = types.SimpleNamespace(kind="deepseek_v41")
    with pytest.raises(HTTPException) as exc:
        add(core, edit_ctx())
    assert (exc.value.status_code, exc.value.detail) == (400, patches.NO_EDIT)
    assert list(core.scheduler._output_parser_sessions) == [rid] and core.scheduler.detoks == {}


def test_the_status_says_inkling_takes_the_edit_and_another_parsed_model_does_not(monkeypatch):
    """chat_formats' ``edit``: a loaded model whose parser the edit seeds (Inkling 2's) takes it; another parser's,
    or one with no kind, does not; a model oMLX does not parse does."""
    def entry(factory):
        sched = types.SimpleNamespace(_output_parser_factory=factory)
        return types.SimpleNamespace(model_path=None, engine=types.SimpleNamespace(
            _engine=types.SimpleNamespace(engine=types.SimpleNamespace(scheduler=sched))))
    entries = {"ink": entry(types.SimpleNamespace(kind="inkling")), "dsv4": entry(types.SimpleNamespace(kind="deepseek_v4")),
               "odd": entry(object()), "plain": entry(None)}
    pool = types.SimpleNamespace(get_model_ids=lambda: list(entries), get_entry=entries.get)
    monkeypatch.setattr(backend, "_SERVER", types.SimpleNamespace(get_server_state=lambda: types.SimpleNamespace(engine_pool=pool)))
    assert {m: c["edit"] for m, c in backend.chat_formats().items()} == dict(ink=True, dsv4=False, odd=False, plain=True)


def test_an_edits_detokenizer_is_dropped_when_omlx_never_takes_its_request(tracked, monkeypatch):
    """The detokenizer started for an edit's completion goes when the request does not get into oMLX: refused by
    oMLX (a full queue, a preflight), by us (a steer that cannot be set up: 503), or cancelled while it was made."""
    import threading

    def edit_ctx(steered=False):
        return dict(header="on", meta={}, steer=steer.parse_header("calm@0.5") if steered else None,
                    base=([1, 2, 3], [11]), rid=None, text=None, started=asyncio.Event())

    core = omlx()
    core.refuse = RuntimeError("queue full")
    with pytest.raises(RuntimeError, match="queue full"):
        add(core, edit_ctx())
    assert core.scheduler.detoks == {} and len(tracked["released"]) == 1
    core.refuse = None

    def register(*a, **k):
        raise ValueError("boom")
    with monkeypatch.context() as m:
        m.setattr(MANAGER, "register", register)
        with pytest.raises(HTTPException):
            add(core, edit_ctx(steered=True))
    assert core.scheduler.detoks == {} and len(core.added) == 1

    gate = threading.Event()
    get = core.scheduler._get_detokenizer
    core.scheduler._get_detokenizer = lambda request_id: (gate.wait(5), get(request_id))[1]

    async def cancelled():
        token = REQUEST_CTX.set(edit_ctx())
        try:
            task = asyncio.ensure_future(core.add_request([1, 2, 3], None, "late"))
        finally:
            REQUEST_CTX.reset(token)
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if "late" not in core.scheduler.detoks and gate.is_set():
                break
    asyncio.run(cancelled())
    assert core.scheduler.detoks == {} and "late" not in core.added


def test_edit_at_is_advertised_only_once_what_the_edit_needs_took(monkeypatch):
    """oMLX's status lists edit_at only when the ids reach its engine, its replies are kept (patch_engine_core,
    patch_scheduler) and its scheduler keeps a detokenizer a request; else the dock says the server cannot."""
    monkeypatch.setattr(patches, "_DETOKENIZERS", patches._DETOKENIZERS)      # put back afterwards
    monkeypatch.setattr(patches, "APPLIED", {"patch_engine_core", "patch_scheduler", "patch_server"})
    module = stand_ins()
    patches.patch_scheduler(module)                      # a scheduler with per-request detokenizers
    assert backend.BACKEND.can() == {"load_copy", "edit_at"}
    monkeypatch.setattr(patches, "APPLIED", {"patch_engine_core", "patch_server"})
    assert backend.BACKEND.can() == {"load_copy"}
    monkeypatch.setattr(patches, "APPLIED", {"patch_engine_core", "patch_scheduler", "patch_server"})
    old = stand_ins()
    del old.Scheduler._get_detokenizer                   # an oMLX without them
    patches.patch_scheduler(old)
    assert backend.BACKEND.can() == {"load_copy"}


def test_a_traced_reply_is_kept_with_the_prompt_it_came_with(tracked):
    """oMLX appends a reply's tokens so far to its prompt when it requeues it under memory pressure: the edit keeps
    the prompt as it came."""
    MANAGER.register("requeued", "on", {"model": "m"})
    req = types.SimpleNamespace(prompt_token_ids=[1, 2, 3, 11, 12], output_token_ids=[11, 12, 13],
                                requeued_output_tokens=2)
    patches._keep("requeued", req)
    assert asyncio.run(MANAGER.kept.get("requeued", lambda rid: False, 0)) == ([1, 2, 3], [11, 12, 13])

"""concept learn: the GPU lock, the engine-pool refusal, unload waiting, the writer's checks, the items and the push,
and the early stops."""
import asyncio
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from metalscope import gpulock, models
from metalscope.omlx import patches as omlx_patches, admin as omlx_admin
from metalscope.concepts.summary import push_toward, push_words
from metalscope.learn import command as Cmd, decide, job as J, plan as P, spawn, state, window as Win, writer as W
from tests.helpers import learn_job


@pytest.fixture(autouse=True)
def served(monkeypatch):
    """Every model these tests name counts as served, and no real server is asked (test_cli has the refusal)."""
    from metalscope import backends
    monkeypatch.setattr(backends, "require_served", lambda name: None)


NO_DOSE_UNITS = P.no_dose_units        # the real check, for the test of it


@pytest.fixture(autouse=True)
def dose_units(monkeypatch):
    """Every model these tests name has dose units (a strength table), but in the test of the check itself."""
    monkeypatch.setattr(P, "no_dose_units", lambda model_name: None)


# ---------------------------------------------------------------- the lock

def test_a_lock_whose_holder_cannot_be_written_is_not_left_held(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"

    def full(self, **changes):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(gpulock.GpuLock, "update", full)
    lock = gpulock.GpuLock(gpulock.Holder("learning toasty"), path)
    with pytest.raises(OSError):
        lock.acquire()
    assert lock.fd is None and gpulock.held(path) is None and _fds_on(path) == []


def test_lock_is_held_until_released(tmp_path):
    path = tmp_path / "gpu.lock"
    assert gpulock.held(path) is None
    lock = gpulock.GpuLock(gpulock.Holder("learning toasty", stage="presence"), path).acquire()
    h = gpulock.held(path)
    assert h is not None and h.what == "learning toasty" and h.stage == "presence"
    with pytest.raises(gpulock.LockBusy, match="learning toasty"):
        gpulock.GpuLock(gpulock.Holder("other"), path).acquire(wait=0.2)
    lock.update(stage="selection sweep")
    assert gpulock.held(path).stage == "selection sweep"
    lock.release()
    assert gpulock.held(path) is None


def test_a_look_at_a_free_lock_never_makes_another_look_see_it_held(tmp_path, monkeypatch):
    """Two processes look at once (oMLX's load guard, the ds4 proxy, Pi's job poll): a look must not make a free lock
    seem held to the other, while a real holder is still seen. While this process is inside ``held`` or ``unclean``
    (its flock taken), a second process calls both."""
    import fcntl
    path = tmp_path / "gpu.lock"
    gpulock.GpuLock(gpulock.Holder("learning old"), path).acquire().release()
    other = [sys.executable, "-c", "import sys; from pathlib import Path; from metalscope import gpulock; "
             "p = Path(sys.argv[1]); print(gpulock.held(p), gpulock.unclean(p))", str(path)]
    seen = []

    class MidLook:
        """fcntl with the other process's look run right after each lock this process takes."""
        def __getattr__(self, name):
            return getattr(fcntl, name)

        def flock(self, fd, op):
            fcntl.flock(fd, op)
            if not op & fcntl.LOCK_UN:
                seen.append(subprocess.run(other, capture_output=True, text=True, check=True).stdout.split())

    monkeypatch.setattr(gpulock, "fcntl", MidLook())
    assert gpulock.held(path) is None and gpulock.unclean(path) is None
    assert seen == [["None", "None"]] * 2
    monkeypatch.undo()
    lock = gpulock.GpuLock(gpulock.Holder("learning toasty"), path).acquire()
    try:
        assert "learning toasty" in subprocess.run(other, capture_output=True, text=True, check=True).stdout
    finally:
        lock.release()


def test_lock_survives_the_parent_closing_while_a_child_holds_it(tmp_path):
    """The child inherits the descriptor (pass_fds): the lock stays held while
    either process has it open, and the kernel frees it when both are gone."""
    path = tmp_path / "gpu.lock"
    lock = gpulock.GpuLock(gpulock.Holder("job"), path).acquire()
    child = subprocess.Popen([sys.executable, "-c", "import sys,time; sys.stdin.read()"], stdin=subprocess.PIPE,
                             pass_fds=(lock.fd,))
    import os
    os.close(lock.fd)          # the parent lets go without unlocking (as if it died)
    lock.fd = None
    assert gpulock.held(path) is not None
    child.stdin.close()
    child.wait()
    assert gpulock.held(path) is None


def test_release_while_the_child_runs_keeps_the_lock(tmp_path):
    """release() must never LOCK_UN: the flock is on the description the child
    shares, so unlocking would free oMLX's loads while the child holds memory."""
    path = tmp_path / "gpu.lock"
    lock = gpulock.GpuLock(gpulock.Holder("job"), path).acquire()
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE,
                             pass_fds=(lock.fd,))
    lock.release()                      # e.g. a cancel landing before the job knew its child
    assert lock.fd is None
    assert gpulock.held(path) is not None
    child.stdin.close()
    child.wait()
    assert gpulock.held(path) is None


def test_unclean_release_is_remembered(tmp_path):
    path = tmp_path / "gpu.lock"
    gpulock.GpuLock(gpulock.Holder("learning a"), path).acquire().release(clean=False)
    assert gpulock.unclean(path).what == "learning a"
    gpulock.GpuLock(gpulock.Holder("learning b"), path).acquire().release()
    assert gpulock.unclean(path) is None


def test_an_unclean_release_stops_counting_after_a_while_or_a_reboot(tmp_path, monkeypatch):
    """A killed run's memory is soon back: the load guard's wait for it must end, or a model larger than free memory
    less the margin could never load again."""
    path = tmp_path / "gpu.lock"
    gpulock.GpuLock(gpulock.Holder("learning a"), path).acquire().release(clean=False)
    monkeypatch.setattr(gpulock, "boot_time", lambda: path.stat().st_mtime - 3600)       # booted before the run
    t0 = 1e10
    assert gpulock.unclean(path, now=t0).what == "learning a"                            # first seen dead now
    assert gpulock.unclean(path, now=t0 + gpulock.UNCLEAN_WINDOW_S - 1) is not None
    assert gpulock.unclean(path, now=t0 + gpulock.UNCLEAN_WINDOW_S + 1) is None
    gpulock.GpuLock(gpulock.Holder("learning b"), path).acquire().release(clean=False)    # another killed run: a new wait
    assert gpulock.unclean(path, now=t0 + 2 * gpulock.UNCLEAN_WINDOW_S).what == "learning b"
    monkeypatch.setattr(gpulock, "boot_time", lambda: path.stat().st_mtime + 1)           # rebooted since
    assert gpulock.unclean(path) is None


def test_an_unclean_release_counts_once_across_server_restarts(tmp_path, monkeypatch):
    """A lock left at "releasing" long ago opens the wait once, not after every oMLX restart: when it was first
    found dead lives in the file, not in each process."""
    import os
    import time
    path = tmp_path / "gpu.lock"
    path.write_text(json.dumps(dict(what="baselining the lens for m", pid=99405, since=1.0, stage="releasing")))
    long_ago = time.time() - 2 * 86400
    os.utime(path, (long_ago, long_ago))           # a run's last write can be long before it died: still counts
    monkeypatch.setattr(gpulock, "boot_time", lambda: long_ago - 3600)
    t0 = time.time() - 2 * gpulock.UNCLEAN_WINDOW_S
    assert gpulock.unclean(path, now=t0).stage == "releasing"
    assert json.loads(path.read_text())["dead_since"] == t0
    assert gpulock.unclean(path, now=t0 + 1) is not None
    restarted = subprocess.run([sys.executable, "-c", "import sys; from pathlib import Path; from metalscope import gpulock; "
                                "print(gpulock.unclean(Path(sys.argv[1])))", str(path)],
                               capture_output=True, text=True, check=True)
    assert restarted.stdout.strip() == "None"       # another process counts from the same moment: over
    assert gpulock.read_holder(path).what == "baselining the lens for m"


def test_holder_message():
    import time
    h = gpulock.Holder("learning toasty", stage="profiling", eta=time.time() + 11 * 60)
    assert h.message() == "learning toasty: profiling, about 11 min left; models load again when it is done"
    assert gpulock.Holder("concept split angry").message() == "concept split angry; models load again when it is done"


def test_wait_available_and_watchdog():
    vals = iter([10 * gpulock.GB, 50 * gpulock.GB])
    assert gpulock.wait_available(40, timeout=5, log=lambda s: None, reader=lambda: next(vals), sleep=0)
    assert not gpulock.wait_available(40, timeout=0, log=lambda s: None, reader=lambda: 1 * gpulock.GB, sleep=0)
    # settle: back once steady below the need (83 then 95 GB, need 96), not while still rising
    clock = lambda: next(ticks)
    ticks, vals = iter(range(0, 1000, 10)), iter([83 * gpulock.GB] + [95 * gpulock.GB] * 99)
    assert not gpulock.wait_available(96, timeout=300, log=lambda s: None, reader=lambda: next(vals), sleep=0, clock=clock)
    ticks, vals = iter(range(0, 1000, 10)), iter([83 * gpulock.GB] + [95 * gpulock.GB] * 99)
    assert gpulock.wait_available(96, timeout=300, log=lambda s: None, reader=lambda: next(vals), sleep=0, settle=60,
                                  clock=clock)
    ticks, vals = iter(range(0, 1000, 10)), iter(g * gpulock.GB for g in range(10, 100))    # a GB more every 10 s
    assert not gpulock.wait_available(96, timeout=300, log=lambda s: None, reader=lambda: next(vals), sleep=0, settle=60,
                                      clock=clock)
    exits = []
    dog = gpulock.Watchdog(log=lambda s: None, exit_fn=exits.append, pressure=lambda: 4, swap=lambda: 0)
    assert dog.check() == "memory pressure critical"
    dog = gpulock.Watchdog(log=lambda s: None, exit_fn=exits.append, pressure=lambda: 1,
                           swap=iter([0, 5 * gpulock.GB]).__next__)
    assert dog.check() == "swap grew by 5.0 GB"


def test_a_watchdog_that_has_ended_can_be_joined_and_asked_whether_it_runs():
    """A Thread's join() and is_alive() call its own _stop() once it has ended: the watchdog's stop signal must not
    take that name."""
    dog = gpulock.Watchdog(log=lambda s: None, interval=0.01, pressure=lambda: 1, swap=lambda: 0)
    dog.start()
    dog.stop()
    dog.join(timeout=5)
    assert not dog.is_alive()


# ---------------------------------------------------------------- the engine pool refuses loads

class _Unavailable(Exception):
    def __init__(self, model_id, message=None):
        super().__init__(message or model_id)


def _fake_pool():
    """oMLX's EnginePool, its shape where the guard matters (0.7.3's signatures): get_engine hands back a loaded
    engine, but for a runtime-settings variant or force_lm it unloads that engine first, then evicts others by LRU
    to make room, and only then loads."""
    class Entry:
        def __init__(self, engine, resident=0):
            self.engine, self.resident, self.estimated_size = engine, resident, 107 * gpulock.GB

    class EnginePool:
        _settings_manager = None
        evict = ()                                   # the LRU victims the next load needs gone

        def __init__(self):
            self._entries = {"big": Entry(None), "loaded": Entry("engine"), "other": Entry("engine")}
            self.loads, self.unloads = [], []

        def get_entry(self, model_id):
            return self._entries.get(model_id)

        def _entry_runtime_resident_size(self, entry, runtime_settings, *, base_size=None,
                                         include_ane_reservation=True, log_decision=True):
            return entry.resident

        async def _unload_engine(self, model_id, *, local_only=False):
            self.unloads.append(model_id)
            self._entries[model_id].engine = None

        async def _load_engine(self, model_id, force_lm=False, runtime_settings=None):
            self.loads.append(model_id)
            self._entries[model_id].engine = "engine"

        async def get_engine(self, model_id, force_lm=False, _lease=False, runtime_settings=None):
            entry = self._entries[model_id]
            if entry.engine is not None and not force_lm and runtime_settings is None:
                return entry.engine
            if entry.engine is not None:
                await self._unload_engine(model_id)
            for victim in self.evict:
                await self._unload_engine(victim)
            await self._load_engine(model_id, force_lm=force_lm, runtime_settings=runtime_settings)
            return entry.engine

    omlx_patches.patch_engine_pool(types.SimpleNamespace(EnginePool=EnginePool, ModelUnavailableError=_Unavailable))
    return EnginePool()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_engine_pool_refuses_loads_while_locked(tmp_path, monkeypatch):
    monkeypatch.setattr(gpulock, "lock_path", lambda: tmp_path / "gpu.lock")
    pool = _fake_pool()
    assert _run(pool.get_engine("big")) == "engine" and pool.loads == ["big"]
    _run(pool._unload_engine("big"))
    lock = gpulock.GpuLock(gpulock.Holder("learning toasty", stage="presence")).acquire()
    try:
        with pytest.raises(_Unavailable, match="cannot load now: learning toasty: presence"):
            _run(pool.get_engine("big"))
        with pytest.raises(_Unavailable):
            _run(pool._load_engine("big"))           # the admin route and the startup preload call this directly
        assert _run(pool.get_engine("loaded")) == "engine"    # chat with a loaded model is not touched
        _run(pool._unload_engine("big"))             # nor an unload that is not for a load (the job's own, the admin's)
    finally:
        lock.release()
    assert _run(pool.get_engine("big")) == "engine" and pool.loads == ["big", "big"]
    assert omlx_patches.GPU_LOCK_ENFORCED


def test_engine_pool_refuses_loads_while_a_job_runs_on_the_other_backend(tmp_path, monkeypatch):
    """A job on ds4 (a calibration holds no GPU lock) refuses every oMLX load: it would sit beside ds4's model. A job
    on one of oMLX's own models refuses nothing: its requests load its model, a learn job reloads it."""
    monkeypatch.setattr(gpulock, "lock_path", lambda: tmp_path / "gpu.lock")
    pool = _fake_pool()
    pool.evict = ("other",)
    running = [dict(job="c-1", kind="calibrate", name="pirate", model="V41-ds4")]
    monkeypatch.setattr(state, "running", lambda exclude=None: running[0])
    with pytest.raises(_Unavailable, match=r"^Model 'big' cannot load now: a job is running: calibrating pirate \(/scope jobs\); "
                                           "models load again when it is done$"):
        _run(pool.get_engine("big"))
    with pytest.raises(_Unavailable):
        _run(pool._load_engine("big"))               # the admin route
    assert pool.loads == [] and pool.unloads == []   # nothing evicted for it either
    running[0] = dict(job="c-2", kind="calibrate", name="pirate", model="big")       # a job on oMLX's own model
    assert _run(pool.get_engine("big")) == "engine" and pool.loads == ["big"]
    for typed in ("loaded", "mlx-community/loaded"):         # from the terminal the job keeps the name as typed
        running[0] = dict(job="l-1", kind="learn", name="pirate", model=typed)
        _run(pool._unload_engine("other"))
        assert _run(pool.get_engine("other")) == "engine"      # its writer, another of oMLX's models


def test_a_locked_gpu_never_leaves_a_loaded_model_unloaded(tmp_path, monkeypatch):
    """oMLX unloads the served engine for a runtime-settings variant or force_lm, and evicts others, before it gets
    to _load_engine: refused only there, the lock left the served model unloaded."""
    monkeypatch.setattr(gpulock, "lock_path", lambda: tmp_path / "gpu.lock")
    pool = _fake_pool()
    pool.evict = ("other",)
    lock = gpulock.GpuLock(gpulock.Holder("learning toasty")).acquire()
    try:
        for call in (dict(runtime_settings=object()), dict(force_lm=True)):
            with pytest.raises(_Unavailable, match="'loaded' cannot load now: learning toasty"):
                _run(pool.get_engine("loaded", **call))
        with pytest.raises(_Unavailable, match="'big' cannot load now"):
            _run(pool.get_engine("big"))             # nor is another model evicted for a load that will be refused
        assert pool.unloads == [] and pool.loads == []
        assert pool._entries["loaded"].engine == pool._entries["other"].engine == "engine"
    finally:
        lock.release()
    assert _run(pool.get_engine("loaded", force_lm=True)) == "engine"
    assert pool.unloads == ["loaded", "other"] and pool.loads == ["loaded"]


def test_engine_pool_waits_for_memory_after_a_killed_run(tmp_path, monkeypatch):
    """Against what oMLX admits the load by (its resident size: 70 GB with expert offload), not the files' 107 GB;
    vm_stat is read off the event loop; and only the load waits: an eviction for it goes ahead (it frees memory)."""
    import threading
    monkeypatch.setattr(gpulock, "lock_path", lambda: tmp_path / "gpu.lock")
    pool = _fake_pool()
    pool._entries["big"].resident = 70 * gpulock.GB
    gpulock.GpuLock(gpulock.Holder("learning toasty")).acquire().release(clean=False)   # as if killed
    free, threads = [30 * gpulock.GB], []

    def available():
        threads.append(threading.current_thread())
        return free[0]
    monkeypatch.setattr(gpulock, "available", available)
    pool.evict = ("other",)
    with pytest.raises(_Unavailable, match="stopped without releasing it, and only 30 GB is free"):
        _run(pool.get_engine("big"))
    assert pool.unloads == ["other"] and pool.loads == []
    free[0] = 80 * gpulock.GB
    assert _run(pool.get_engine("big")) == "engine"
    assert threads and threading.main_thread() not in threads


# ---------------------------------------------------------------- unload waits until the model is gone

class _SlowAdmin:
    """oMLX answering 202 while requests are in flight: still loaded for two polls."""
    def __init__(self):
        self.polls = 0

    def unload(self, model_id):
        return {"status": "unloading", "message": f"Aborting active requests before unloading {model_id}"}

    def admin_models(self):
        self.polls += 1
        return [{"id": "m", "loaded": self.polls < 3, "is_loading": False}]


def test_unload_waits_for_a_202(monkeypatch):
    monkeypatch.setattr(gpulock.time, "sleep", lambda s: None)
    admin, out = _SlowAdmin(), []
    omlx_admin.unload(admin, "m", log=out.append)
    assert admin.polls == 3
    assert out == ["Aborting active requests before unloading m", "unloaded m"]


def test_unload_gives_up(monkeypatch):
    monkeypatch.setattr(gpulock.time, "sleep", lambda s: None)
    admin = _SlowAdmin()
    admin.admin_models = lambda: [{"id": "m", "loaded": True}]
    with pytest.raises(omlx_admin.ServerError, match="did not unload"):
        omlx_admin.unload(admin, "m", log=lambda s: None, timeout=0)


# ---------------------------------------------------------------- the writer's checks

def test_parse_json_takes_the_object_asked_for():
    assert W.parse_json('Sure!\n```json\n{"a": [1, 2]}\n```', "a") == {"a": [1, 2]}
    assert W.parse_json('Use {braces} sparingly. {"a": 1} Done (see {x}).', "a") == {"a": 1}
    assert W.parse_json('{"note": 1}\n{"items": [1]}', "items") == {"items": [1]}
    assert W.parse_json('[{"items": [{"prompt": "p"}]}]', "items") == {"items": [{"prompt": "p"}]}
    # The prompt's format echoed (its placeholders and example strings only), then the answer: the answer.
    spec = W.SPEC.format(name="pirate", desc="talks like a pirate")
    echo = spec.split("Reply with JSON only:", 1)[1].strip()
    assert W.parse_json(f"Format: {echo}\n\n" + '{"title": "Pirate", "kind": "style"}', "title", spec) == {
        "title": "Pirate", "kind": "style"}
    with pytest.raises(ValueError):
        W.parse_json(echo, "title", spec)                                       # a pure echo
    # One item whose target is "..." is not an echo: the reply's other items are kept (valid_items drops that one).
    task = W.TASK.format(name="pirate", desc="talks like a pirate", n=30)
    items = [dict(prompt=f"Which reply {i}?", options=["Arr", "Hello"], target="Arr") for i in range(29)]
    items.append(dict(prompt="Which reply?", options=["Arr", "Hello"], target="..."))
    got = W.parse_json(json.dumps(dict(items=items)), "items", task)
    assert len(got["items"]) == 30 and len(W.valid_items(got["items"], "pirate")) == 29
    for bad in ("no json here", '[{"prompt": "p", "options": ["x"], "target": "x"}]',       # a bare list of items
                '{"pairs": [{"pos": "a", "neg": "b"}, {"pos": "c", "ne'):                   # cut off: not its first pair
        with pytest.raises(ValueError):
            W.parse_json(bad, "items" if "prompt" in bad else "pairs")


def test_a_reply_without_the_object_is_asked_again():
    replies = iter(['[{"prompt": "p"}]', '{"items": [{"prompt": "p"}]}'])
    wr = W.Writer(None, "m", log=lambda m: None)
    wr.chat = lambda prompt, **kw: next(replies)
    assert wr.ask_json("items?", "items") == {"items": [{"prompt": "p"}]}


def test_the_writers_requests_are_not_traced():
    """The writer's and the grader's requests are a job's own, not chats: traced under the server's mode they would
    be counted in the tally and cost a trace's memory each."""
    from metalscope import client
    sent = []

    class Server:
        def post(self, path, body, timeout=60, headers=None):
            sent.append((path, headers))
            return {"choices": [{"message": {"content": "hi"}}]}
    assert W.Writer(Server(), "m", log=lambda m: None).chat("p") == "hi"
    assert sent == [("/v1/chat/completions", client.UNTRACED)] and client.UNTRACED == {"x-scope-trace": "off"}


def test_the_writer_gives_a_model_that_thinks_anyway_room_for_its_thought():
    """Inkling 2 thinks even with thinking off (grade.thinks_anyway): its requests get the thinking budget, the
    others the 4,000 they had."""
    from metalscope.learn import grade as G
    bodies = []

    class Server:
        def post(self, path, body, timeout=None, headers=None):
            bodies.append(body)
            return {"choices": [{"message": {"content": "hi"}}]}
    W.Writer(Server(), "Inkling-2-Mini-oQ4e-mtp", log=lambda m: None).chat("p")
    W.Writer(Server(), "Qwen3.8-27B-4bit", log=lambda m: None).chat("p")
    assert [b["max_tokens"] for b in bodies] == [G.MAX_TOKENS[True], 4000] and bodies[0]["reasoning_effort"] == "none"


def test_pair_problem():
    ok = ("She jabbed the lift button again and shoved past the man stepping out of the car.",
          "She pressed the lift button again and stepped aside for the man leaving the car.")
    assert W.pair_problem("angry", *ok) == ""
    assert W.pair_problem("angry", "She was angry at the lift.", ok[1]) == "length"
    assert W.pair_problem("angry", ok[0].replace("jabbed", "angrily jabbed").replace(" again", ""), ok[1]) == "name"


def test_valid_items():
    items = [dict(prompt="Which?", options=["a", "b"], target="a"),
             dict(prompt="Which?", options=["c", "d"], target="c"),          # duplicate prompt
             dict(prompt="Odd", options=["a", "a"], target="a"),             # options not distinct
             dict(prompt="Missing", options=["a", "b"], target="z"),         # target not an option
             dict(prompt="Leak", options=["the toasty one", "b"], target="b"),
             "not a dict"]
    assert W.valid_items(items, "toasty") == [dict(prompt="Which?", options=["a", "b"], target="a")]


class _FakeWriter:
    model = "fake"

    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def ask_json(self, prompt, **kw):
        reply = next(self.replies)
        self.calls.append(dict(prompt=prompt, reply=json.dumps(reply), seconds=0.1))       # as Writer.chat records it
        return reply


def test_write_builds_the_concept():
    pair = dict(pos="He bit the crisp apple and heard the sharp snap ring out in the quiet kitchen today.",
                neg="He bit the ripe apple and felt the soft flesh give way in the quiet kitchen today.")
    pairs = [dict(pos=pair["pos"].replace("apple", f"apple {i}"), neg=pair["neg"].replace("apple", f"apple {i}")) for i in range(20)]
    items = [dict(prompt=f"Which snack {i}?", options=["chips", "pudding"], target="chips") for i in range(60)]
    fw = _FakeWriter([dict(title="Crisp", kind="content", description="d.", contrast="soft textures", words=["crisp"],
                           target_words=["snap"]), dict(pairs=pairs), dict(items=items[:30]), dict(items=items[30:])])
    out = W.write(fw, "toasty", "crisp textures")
    assert len(out["pos"]) == W.N_PAIRS == len(out["neg"]) and len(out["task"]["items"]) == 60
    assert out["concept"]["kind"] == "content" and out["concept"]["learned"]["request"] == "crisp textures"
    assert "crisp textures" in fw.calls[-1]["prompt"]          # the items are written from the user's description
    # writing.json: the writer, the date, the request and every instruction sent, in order, without the replies.
    w = out["writing"]
    assert sorted(w) == ["date", "prompts", "request", "writer"] and (w["writer"], w["request"]) == ("fake", "crisp textures")
    assert w["prompts"] == [c["prompt"] for c in fw.calls] and len(w["prompts"]) == 4
    assert w["date"] == out["concept"]["learned"]["date"] and "apple 0" not in json.dumps(w)      # a reply's pair


# ---------------------------------------------------------------- the items, the push, the early stops

def test_item_selection_splits_against_items():
    probs = [0.01, 0.99] + [0.1 + 0.02 * i for i in range(12)] + [0.7, 0.8, None]
    sel = decide.select_items(probs, seed=0)
    assert sel["testable"] and len(sel["against"]) == 12 and sel["favour"] == [14, 15]
    assert sorted(sel["selection"] + sel["test"]) == sel["against"] and len(sel["selection"]) == 6
    assert sel["counts"]["saturated_low"] == 1 and sel["counts"]["saturated_high"] == 1
    few = decide.select_items([0.2, 0.3, 0.99, 0.999], seed=0)
    assert not few["testable"] and few["fallback"] == [0, 1]
    # No item with room to move: nothing to sweep on, and a reason to act on.
    none = decide.select_items([0.99] * 59 + [0.01], seed=0)
    assert not none["testable"] and none["fallback"] == [] and none["selection"] == []
    assert decide.untestable(none["counts"], "candid", "V41") == (
        "can't be steered: V41 already picks the candid option on 59 of 60 items unsteered; rephrase the contrast so "
        "the model doesn't already do it (or the other side), then learn it again under a new name")
    assert "never picks the candid option on 3 of 4" in decide.untestable(decide.select_items([0.01] * 3 + [0.99])["counts"],
                                                                          "candid", "V41")


def _test_sweep(effect_at_d=2.0, reversed_sign=-1.0, mass=0.95):
    """Four against prompts and one favour prompt, doses -D, D/4, D/2, D."""
    D = 0.4
    strengths = [-D, D / 4, D / 2, D]
    base = np.array([-2.0, -1.5, -1.0, -0.5, 1.0])
    scale = np.array([reversed_sign, 0.25, 0.5, 1.0])[:, None]
    concept = base[None] + effect_at_d * scale
    control = np.repeat(base[None] + 0.05, 4, axis=0)
    chosen = (concept > 0).astype(float)
    return dict(strengths=strengths, base_logodds=base.tolist(), base_mass=[1.0] * 5,
                concept_logodds=[concept.tolist()], control_logodds=[control.tolist()],
                concept_mass=[(np.full((4, 5), mass)).tolist()], concept_chosen=[chosen.tolist()],
                control_chosen=[(control > 0).astype(float).tolist()], base_chosen=(base > 0).astype(float).tolist()), D


def test_push_result_and_evidence_lines():
    kinds = ["against"] * 4 + ["favour"]
    t, D = _test_sweep()
    push = decide.push_result(t, kinds, D)
    assert push["at_dose"]["effect"] == pytest.approx(1.95) and push["away"]["concept"] < 0
    assert push["at_dose"]["chosen"] == 0.75 and push["at_dose"]["base_chosen"] == 0.0   # -2.0 + 2.0 lands on 0: not chosen
    learn = dict(selection=decide.select_items([0.2] * 10), layer=12, dose=D, edge=1.6, push=push)
    reader = dict(layer=12, best=dict(strength=0.4, effect=1.5))
    assert decide.evidence(learn, 4.0, 0.005, 20, reader) == [
        "presence: held-out d′ 4.00 at L20, p 0.005",
        "items: 10 against or torn and 0 leaning to it, of 10 (0 already chosen, 0 never)",
        "steer layer: L12 at dose 0.4, +1.50 log-odds over a random direction on the selection half",
        "fluency: replies break from dose 1.6; test dose 0.4",
        "push at dose 0.4 on 4 held-out prompts: +1.95 log-odds over a random direction (+0.45 at a quarter of it); "
        "answers still letters: 95% of the mass"]
    # a push that only works one way, or breaks the answers: reported as measured, never judged
    t2, _ = _test_sweep(reversed_sign=+1.0, mass=0.5)
    lines = decide.evidence(dict(learn, push=decide.push_result(t2, kinds, D)), 1.0, 0.5, 20, reader)
    assert lines[-1].endswith("answers still letters: 50% of the mass")
    assert not any(w in " ".join(lines) for w in ("ok:", "fails:", "need", "keep", "drop"))
    # too few against items for a held-out test: the layer and dose were picked on every usable item
    few = dict(learn, selection=decide.select_items([0.2] * 5 + [0.7, 0.99]), push=None)
    assert decide.evidence(few, 4.0, 0.005, 20, reader)[2] == (
        "steer layer: L12 at dose 0.4, +1.50 log-odds over a random direction on all 6 usable items (no held-out test)")
    # stopped at the items: presence and the items only
    assert decide.evidence(dict(selection=decide.select_items([0.99] * 4)), 4.0, None, 3) == [
        "presence: held-out d′ 4.00 at L3", "items: 0 against or torn and 0 leaning to it, of 4 (4 already chosen, 0 never)"]
    assert decide.NO_STEER_LAYER.startswith("can't be steered: ")


def test_push_words_both_ways():
    row = lambda base, chosen, concept=None, control=None: dict(base_chosen=base, chosen=chosen, concept=concept,
                                                                 control_chosen=control)
    # toward up, away down: as expected
    w = push_words(dict(dose=0.8, at_dose=row(0.5, 0.75, 7.6, 0.375), away=row(0.5, 0.0625, -3.7)), "humorous")
    assert w == dict(toward="Steering toward humorous: picked more often, 50% to 75% (a random direction: 38%).",
                     away="Steering away from humorous: picked less often, 50% to 6%, as expected.")
    # away up too: not expected
    w = push_words(dict(dose=0.8, at_dose=row(0.21, 0.64, 5.2, 0.32), away=row(0.21, 0.3, 2.6)), "blunt")
    assert w["away"] == "Steering away from blunt: picked more often too, 21% to 30% (expected less often)."
    # the share unchanged: the log-odds say which way (blunt on DeepSeek V4.1)
    w = push_words(dict(dose=0.8, at_dose=row(0.21, 0.64, 5.2, 0.32), away=row(0.21, 0.21, 2.65)), "blunt")
    assert w["away"] == "Steering away from blunt: picked as often, 21% to 21%, but made more likely (expected less often)."
    # no share stored: more / less likely
    w = push_words(dict(dose=0.8, at_dose=dict(concept=1.2), away=dict(concept=0.4)), "blunt")
    assert w == dict(toward="Steering toward blunt: made more likely.",
                     away="Steering away from blunt: made more likely too (expected less likely).")
    # no away row: the curve's point at -D (older profiles); nothing at all: None
    w = push_words(dict(dose=0.4, at_dose=row(0.1, 0.4), curve=[dict(strength=-0.4, **row(0.1, 0.0, -0.7))]), "x")
    assert w["away"] == "Steering away from x: picked less often, 10% to 0%, as expected."
    assert push_words(None) == dict(toward=None, away=None)
    # the toward sentence's words alone: learn's test step says them after its log-odds
    assert push_toward(dict(at_dose=row(0.25, 0.4375, 2.1, 0.28125))) == ("picked more often, 25% to 44% (a random direction: 28%)", True)
    assert push_toward(dict(at_dose=dict(concept=-0.3))) == ("made less likely", False)
    assert push_toward(None) == (None, None)


def test_overlap_lists_the_nearest():
    rng = np.random.default_rng(0)
    width = 2048                                   # the 2B's residual width; Flash Next's streams are wider
    base = {f"c{i}": rng.normal(size=width) for i in range(6)}
    common = rng.normal(size=width)
    existing = {k: v + 0.5 * common for k, v in base.items()}   # a shared component, like "told with feeling"
    near = -existing["c2"] + 0.1 * rng.normal(size=width)
    ov = decide.overlap(near, existing)
    assert ov["n_existing"] == 6 and len(ov["nearest"]) == 5
    assert ov["nearest"][0]["name"] == "c2" and ov["nearest"][0]["cosine"] < -0.9


# ---------------------------------------------------------------- the job's bookkeeping

def test_spawn_refuses_bad_or_taken_names_and_a_busy_gpu():
    with pytest.raises(ValueError):
        spawn.learn("../x", "something", "m")
    with pytest.raises(ValueError, match="description"):
        spawn.learn("fresh", "  ", "m")
    # A model's own texts need no description, but they must hold together; one with a profile here is taken.
    d = models.concept_dir("m", "taken")
    d.mkdir(parents=True)
    (d / "concept.json").write_text("{}")
    for f in ("pos.txt", "neg.txt", "task.json"):
        (d / f).write_text("")
    with pytest.raises(ValueError, match="only 0 pairs"):
        spawn.learn("taken", "", "m")
    prof = models.profile_path("m", "taken")
    prof.write_text("{}")
    with pytest.raises(ValueError, match="a profile for 'taken' already exists for m"):
        spawn.learn("taken", "something", "m")
    lock = gpulock.GpuLock(gpulock.Holder("learning other", stage="presence")).acquire()
    try:
        with pytest.raises(ValueError, match="the GPU is in use: learning other"):
            spawn.learn("fresh", "something", "m")
    finally:
        lock.release()


def test_every_job_id_a_concept_name_makes_is_a_job_id_up_to_the_longest_name(monkeypatch):
    """Job ids are names (state.job_dir); every one built from a concept's name (state.new_job_id: learn, Pi's
    calibrate, the command line's calibrate) fits for a name of NAME_MAX characters, and a longer name is refused."""
    from metalscope import vectors
    from metalscope.learn import state
    name = "a" * vectors.NAME_MAX
    for kind, unique in (("", False), ("calibrate", False), ("calibrate", True)):
        job = state.new_job_id(name, kind, unique)
        assert state.job_dir(job).name == job and job.startswith(name + "-")
    jobs = []
    monkeypatch.setattr(spawn, "_launch", lambda job, *a, **k: jobs.append(job) or dict(job=job))
    for n in (vectors.NAME_MAX, vectors.NAME_MAX + 1):
        models.concept_dir("m", "a" * n).mkdir(parents=True)
        models.profile_path("m", "a" * n).write_text("{}")
    spawn.calibrate(name, "m")
    assert jobs[0].startswith(name + "-calibrate-")
    with pytest.raises(ValueError, match=r"letters, digits, _ \. - only, not starting with a dot, at most 96 characters"):
        spawn.calibrate("a" * (vectors.NAME_MAX + 1), "m")


def test_no_job_starts_while_a_backend_switch_is_under_way(monkeypatch):
    """A switch checks for a running job and then takes minutes to stop or start a backend: a job starting meanwhile
    is refused, and one starting first holds a switch off until its state is written (the switch then sees it)."""
    from types import SimpleNamespace

    from metalscope import backends
    prof = models.profile_path("m", "joyful")
    prof.parent.mkdir(parents=True)
    prof.write_text("{}")
    switched, started = [], []

    def popen(cmd, **kw):                                       # nothing is run: the job is never started for real
        started.append(1)
        with pytest.raises(ValueError, match="under way"):      # the launch holds the switch lock meanwhile
            with backends.switching(wait=False):
                switched.append(1)
        return SimpleNamespace(pid=999999, wait=lambda: 0)
    monkeypatch.setattr(spawn.subprocess, "Popen", popen)
    with backends.switching():
        with pytest.raises(ValueError, match="^a backend switch is under way: try again in a moment$"):
            spawn.calibrate("joyful", "m")
    assert state.list_jobs() == [] and started == []
    assert spawn.calibrate("joyful", "m")["pid"] == 999999 and started == [1] and switched == []
    with backends.switching(wait=False):
        switched.append(1)
    assert switched == [1]


def test_calibrate_jobs_need_a_profile_and_no_other_job(monkeypatch):
    from metalscope import config
    with pytest.raises(ValueError, match="no profile for m: /concept list"):
        spawn.calibrate("fluffy", "m")
    prof = models.profile_path("m", "fluffy")
    prof.parent.mkdir(parents=True)
    prof.write_text("{}")
    # Another job running (a learn job unloads the model a calibration talks to): refused, naming it.
    d = state.job_dir("other-1")
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps(dict(job="other-1", kind="learn", name="other", outcome=None, pid=1, started=1)))
    monkeypatch.setattr(state, "_alive", lambda *a, **k: True)
    with pytest.raises(ValueError, match="a job is running: learning other"):
        spawn.calibrate("fluffy", "m")
    with pytest.raises(ValueError, match="a job is running: learning other"):
        spawn.learn("fresh", "something", "m")
    # A started job: its command and first state.
    (d / "state.json").write_text(json.dumps(dict(job="other-1", kind="learn", name="other", outcome="done", pid=1, started=1)))
    started = []

    class Proc:
        pid = 4242

        def wait(self):
            return 0
    monkeypatch.setattr(spawn.subprocess, "Popen", lambda cmd, **k: started.append(cmd) or Proc())
    out = spawn.calibrate("fluffy", "m", url="http://127.0.0.1:8009")
    assert out["pid"] == 4242 and out["job"].startswith("fluffy-calibrate-")
    assert started[0][-9:] == ["calibrate", "fluffy", "m", "--url", "http://127.0.0.1:8009", "--job", out["job"], "--origin", "pi"]
    st = json.loads((state.job_dir(out["job"]) / "state.json").read_text())
    assert st["kind"] == "calibrate" and "dose" not in st and st["outcome"] is None and st["origin"] == "pi"
    assert config.home() in state.job_dir(out["job"]).parents


def test_pick_loader_uses_omlx_when_mlx_lm_lacks_the_architecture(tmp_path):
    """Flash Next's qwen4_exp is not in mlx_lm: its first /learn failed there (2026-10-07)."""
    from metalscope.engines.mlx import pick_loader
    known, unknown = tmp_path / "known", tmp_path / "unknown"
    known.mkdir(), unknown.mkdir()
    (known / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (unknown / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    assert pick_loader(str(known)) == "mlx_lm"
    assert pick_loader(str(unknown)) == "omlx"
    assert pick_loader(str(tmp_path / "missing")) == "omlx"


def test_the_step_list_follows_the_phases():
    j = learn_job("x", "d", "/nonexistent", "m", admin=object(), job="x-steps")
    status = lambda: {s["key"]: s["status"] for s in j.state["steps"]}
    j.phase("writing")
    j._corpus_stage("writing the pairs")
    assert status()["corpus"] == "running" and j.step("corpus")["detail"] == "writing the pairs"
    j.summarize("corpus", "16 pairs")
    j.phase("waiting for memory")
    assert status()["corpus"] == "done" and status()["unload"] == "running"
    j.phase("selection sweep", 10, 100)
    sel = j.step("selection")
    assert status()["unload"] == "done" and sel["status"] == "running" and (sel["n"], sel["total"]) == (10, 100)
    j._close_steps(failed=True)
    assert status()["selection"] == "failed" and status()["ladder"] == "pending"
    saved = json.loads((state.job_dir("x-steps") / "state.json").read_text())
    assert saved["steps"][0]["summary"] == "16 pairs"


def test_a_phase_far_past_its_usual_time_warns_once_and_never_stops_the_job(monkeypatch):
    """The usual time: the phase's weight at the run's pace so far; the estimate: its time scaled to its count, from 5%
    of it. Over 4x, once the phase has run its usual time: one warning line, in the log and the state until the next
    phase."""
    from metalscope.learn import job as J
    clock = [1000.0]
    monkeypatch.setattr(J.time, "time", lambda: clock[0])
    j = learn_job("x", "d", None, "m", engine="ds4", job="x-slow")
    at = lambda t, *a: (clock.__setitem__(0, 1000.0 + t), j.phase(*a))
    at(0, "writing")
    at(600, "presence")
    at(660, "measuring the items")
    at(960, "selection sweep", 0, 7200)            # usual: 0.53 / 0.11 of 960 s, 77 min; warns past 4x, 5.1 h
    for t, n in ((600, 100), (1500, 1000)):        # 100 is under 5% (its time so far, 10 min); 1000: 3 h
        at(960 + t, "selection sweep", n, 7200)
        assert j.state["warning"] is None
    at(960 + 3600, "selection sweep", 1100, 7200)  # 6.5 h, but an hour in: under its usual time, no warning yet
    assert j.state["warning"] is None
    at(960 + 4800, "selection sweep", 1300, 7200)  # 7.4 h, 80 min in
    msg = "selection sweep: estimated 7.4 h, expected about 77 min; something is off, consider stopping it"
    assert j.state["warning"] == msg and j.state["log"][-1] == msg and j.state["outcome"] is None
    at(960 + 7200, "selection sweep", 1500, 7200)
    assert j.state["log"].count(msg) == 1 and j.state["warning"] == msg
    at(960 + 7300, "fluency ladder", 0, 20)        # the next phase: the warning goes, the log keeps it
    assert j.state["warning"] is None
    assert (state.job_dir("x-slow") / "log.txt").read_text().count(msg) == 1


def test_slow_early_items_raise_no_warning_before_the_usual_time(monkeypatch):
    """blunt's test sweep: expected about 11 min, its first items slow (scaled from 5% of them, 45 min), done in 20.
    No warning: the estimate counts only once the phase has run its usual time."""
    from metalscope.learn import job as J

    class Two(J.Job):
        KIND = "two"
        WEIGHTS = {"before": 1.0, "test sweep": 1.0}
        STEPS = (("before", "before", ("before",)), ("test", "test", ("test sweep",)))
    clock = [1000.0]
    monkeypatch.setattr(J.time, "time", lambda: clock[0])
    j = Two("x-blunt", "m")
    at = lambda t, *a: (clock.__setitem__(0, 1000.0 + t), j.phase(*a))
    at(0, "before")
    at(660, "test sweep", 0, 60)                     # usual: 660 s, 11 min; 4x is 44 min
    for t, n in ((135, 3), (300, 10), (600, 24), (900, 38), (1200, 60)):     # 3 of 60 in 2.25 min: 45 min scaled
        at(660 + t, "test sweep", n, 60)
        assert j.state["warning"] is None, t
    assert not any("estimated" in line for line in j.state["log"])


def test_a_learn_job_stopped_at_the_items_cant_be_steered_and_releases_the_gpu(monkeypatch):
    """The child stops after the items (nothing to push on): no sweep, ladder, grades or test, the lock released, can't
    be steered, no calibration."""
    from metalscope.learn import job as J
    asked, killed = _ds4_setup(monkeypatch)
    why = "can't be steered: m already picks the toasty option on 59 of 60 items unsteered; rephrase ..."
    monkeypatch.setattr(J.LearnJob, "measured", lambda self: dict(steerable=False, reason=why))
    _ds4_child(monkeypatch, CHILD_LINES[:4] + [_ev(step="measuring the items", summary="0 usable of 60 · push not testable"),
                                               f"[toasty] {why}; no sweep, ladder or test\n",
                                               _ev(stage="done", profile="/p/profile.json")])
    st = learn_job("toasty", "d", None, "m", engine="ds4", url=PROXY, job="toasty-none").run()
    assert st["outcome"] == "done" and st["result"]["steerable"] is False and st["result"]["reason"] == why
    steps = {s["key"]: s["status"] for s in st["steps"]}
    assert [steps[k] for k in ("selection", "ladder", "test", "calibrate")] == ["skipped"] * 4
    assert steps["items"] == steps["release"] == "done" and "label" not in steps
    assert state.outcome_text(st) == "can't be steered"
    assert gpulock.held() is None and not killed and (PROXY, "calibrate") not in asked


def test_a_dead_job_without_an_outcome_reads_as_interrupted():
    d = state.job_dir("x-1")
    d.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (d / "state.json").write_text(json.dumps(dict(job="x-1", kind="learn", name="x", phase="selection sweep", pid=dead.pid, outcome=None)))
    st = state.read_state("x-1")
    assert st["outcome"] == "interrupted" and "stopped without finishing" in st["error"]
    assert state.list_jobs()[-1]["job"] == "x-1"
    assert state.mark_seen("x-1") and state.read_state("x-1")["seen"] and not state.mark_seen("x-none")
    with pytest.raises(ValueError):
        state.job_dir("..")


def test_a_job_that_died_before_its_state_says_why_from_its_output():
    """A job the server started that failed before it wrote its own state (the server's first one has no steps):
    the last line it printed is why."""
    d = state.job_dir("x-2")
    d.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (d / "state.json").write_text(json.dumps(dict(job="x-2", kind="learn", name="x", phase="starting", pid=dead.pid,
                                                  outcome=None, log=[])))
    (d / "stdout.txt").write_text("Traceback (most recent call last):\n  ...\nerror: oMLX does not serve m; pass --model PATH\n\n")
    assert state.read_state("x-2")["error"] == "oMLX does not serve m; pass --model PATH"
    (d / "stdout.txt").write_text("ValueError: bad \x1b]52;c;aGk=\x07 " + "x" * 400 + "\n")   # it reaches the dock
    why = state.read_state("x-2")["error"]
    assert why.startswith("ValueError: bad ·]52;c;aGk=· xxx") and len(why) <= 200 and why.endswith("…")
    (d / "stdout.txt").write_text("")
    assert "stopped without finishing" in state.read_state("x-2")["error"]


# ---------------------------------------------------------------- calibration (graded conversations)

CKPT = "Qwen3.8-Flash-Next"      # the checkpoint the calibration tests' served model ``m`` is linked to: its chat format


def _row(arc, dose, turn, attempt=1, failed=None, answer="a" * 50):
    """One recorded attempt as calibrate.converse records it."""
    return dict(arc=arc, dose=dose, turn=turn, attempt=attempt, failed=failed, user=f"{arc} {turn}?", answer=answer,
                text=f"t</think>{answer}", tool=failed == "tool", seconds=1.0)


def _grade(arc, dose, turn, style=2, coherence=3, broken=False, useful=None):
    g = dict(arc=arc, dose=dose, turn=turn, kind="turn", style=style, coherence=coherence, broken=broken)
    return g if useful is None else dict(g, useful=useful)


def test_a_good_reply_is_useful_too_unless_its_grade_has_no_useful():
    from metalscope.learn import calibrate as K
    rows = [_row(a, d, t) for d in (0.0, 0.4) for a in K.ARCS for t in range(1, 7)]
    # useful 1 is not good, 2 and 3 are; a grade the grader left useful out of: style and coherence only.
    use = {1: 1, 2: 2, 3: 3, 4: None, 5: 0}
    grades = [_grade(a, 0.4, t, useful=use.get(t, 3)) for a in K.ARCS for t in range(1, 7)]
    why = K.verdicts(rows, grades, 0.4)
    assert why[("casual", 1)] == ["not useful"] and why[("casual", 5)] == ["not useful"]
    assert why[("casual", 2)] == why[("casual", 3)] == why[("casual", 4)] == why[("casual", 6)] == []
    assert K.tally(rows, grades, 0.4)["good"] == 12 and "not useful" in K.WHY_NOT
    # Plain by the same rule: an unstyled plain reply was never good; useful only adds a reason.
    plain = [_grade(a, 0.0, t, style=0, useful=1 if t == 1 else 3) for a in K.ARCS for t in range(1, 7)]
    assert K.verdicts(rows, plain, 0.0)[("casual", 1)] == ["style too low", "not useful"]
    assert K.tally(rows, plain, 0.0)["good"] == 0


def test_a_good_reply_is_styled_coherent_first_try_and_before_the_first_break():
    from metalscope.learn import calibrate as K
    rows = [_row(a, 0.4, t) for a in K.ARCS for t in range(1, 7)]
    grades = [_grade(a, 0.4, t) for a in K.ARCS for t in range(1, 7)]
    assert len(K.good_turns(rows, grades, 0.4)) == 18
    # casual turn 2 needed a retry: never good, though graded well; the conversation goes on.
    rows = [r for r in rows if (r["arc"], r["turn"]) != ("casual", 2)]
    rows += [_row("casual", 0.4, 2, attempt=1, failed="tool"), _row("casual", 0.4, 2, attempt=2)]
    # dinner breaks at turn 4: 4, 5 and 6 are not good, even graded styled; explain turn 1 is only a hint (style 1).
    grades = [_grade(a, 0.4, t, broken=(a, t) == ("dinner", 4), style=1 if (a, t) == ("explain", 1) else 2)
              for a in K.ARCS for t in range(1, 7)]
    good = K.good_turns(rows, grades, 0.4)
    assert ("casual", 2) not in good and ("casual", 3) in good
    assert [t for a, t in good if a == "dinner"] == [1, 2, 3] and ("explain", 1) not in good and len(good) == 13
    # Coherence 1 is not good either; an ungraded turn is not good and breaks nothing.
    grades = [g for g in grades if (g["arc"], g["turn"]) != ("explain", 2)]
    grades = [dict(g, coherence=1) if (g["arc"], g["turn"]) == ("explain", 3) else g for g in grades]
    assert sorted(t for a, t in K.good_turns(rows, grades, 0.4) if a == "explain") == [4, 5, 6]
    t = K.tally(rows, grades, 0.4)
    assert (t["good"], t["turns"], t["attempts"], t["failed"], t["graded"]) == (11, 18, 19, dict(tool=1, thinking=0, short=0), 17)
    assert K.failed_share(rows, 0.4) == pytest.approx(1 / 19) and K.tally(rows, grades, 0.9)["coherence"] is None
    # The mean grades over the graded turns (16 at style 2, explain 1 at style 1; one coherence 1 among 3s).
    assert K.mean_grade(grades, 0.4, "style") == round(33 / 17, 2) and t["coherence"] == K.mean_grade(grades, 0.4, "coherence") == round(49 / 17, 2)
    assert K.mean_grade(grades, 0.9, "style") is None
    assert all(w in K.WHY_NOT for why in K.verdicts(rows, grades, 0.4).values() for w in why)
    # A turn still failing after every attempt breaks its conversation, even graded as not broken.
    rows = [r for r in rows if (r["arc"], r["turn"]) != ("explain", 5)]
    rows += [_row("explain", 0.4, 5, attempt=a, failed="short") for a in (1, 2, 3)]
    assert sorted(t for a, t in K.good_turns(rows, grades, 0.4) if a == "explain") == [4]


def test_the_search_rules():
    from metalscope.learn import calibrate as K
    # 0.1 either side of the best tried (most good, ties lower), within 0.1-0.9, not run, nothing above a dose that broke.
    assert K.second_pass({0.2: 5, 0.4: 15, 0.6: 16, 0.8: 3}, set()) == [0.5, 0.7]
    assert K.second_pass({0.2: 14, 0.4: 15}, set()) == [0.3, 0.5]
    assert K.second_pass({0.1: 3, 0.2: 14}, set()) == [0.3]                    # 0.1 below was run
    assert K.second_pass({0.2: 0, 0.4: 6, 0.6: 6, 0.8: 2}, set()) == [0.3, 0.5]
    assert K.second_pass({0.2: 0, 0.4: 2, 0.6: 9, 0.8: 10}, set()) == [0.7, 0.9]
    assert K.second_pass({0.8: 4, 0.9: 9}, set()) == []                        # 1.0 is past the top
    assert K.second_pass({0.2: 8, 0.4: 3}, {0.4}) == [0.1, 0.3]
    assert K.second_pass({0.2: 8, 0.4: 3}, {0.2}) == [0.1]
    assert K.best({0.2: 5, 0.3: 14, 0.4: 17}) == 0.4 and K.best({0.2: 5, 0.4: 9, 0.6: 9}) == 0.4
    assert K.summary_text(dict(default_dose=0.3, good=14, turns=18)) == "14/18 good @0.3"


def _asks_through_a_client(monkeypatch, failures):
    """An ask that posts through a real client.Client whose requests meet ``failures`` in turn (what urlopen raises,
    or "cut": a 500 whose body is cut short, as oMLX out of memory sends), then a whole reply; and the asks made."""
    import http.client
    import io
    import urllib.error

    from metalscope import client
    asks = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Cut(io.BytesIO):
        def read(self, *a):
            raise http.client.IncompleteRead(b"x")

    def urlopen(req, timeout):
        asks.append(req.full_url)
        if len(asks) > len(failures):
            return Resp(b'{"ok": true}')
        f = failures[len(asks) - 1]
        raise urllib.error.HTTPError(req.full_url, 500, "err", {}, Cut()) if f == "cut" else f
    monkeypatch.setattr(client.urllib.request, "urlopen", urlopen)
    return (lambda prompt: client.Client("http://omlx", api_key="").post("/v1/completions", {})), asks


def test_a_dropped_reply_is_asked_again_after_a_pause_and_a_refusal_is_not(monkeypatch):
    """A reply the server dropped (out of memory: a 500 cut short, the connection reset) is asked again after a pause,
    each drop logged, in the conversations and the grading alike. Not asked again: a refusal (4xx), steering that
    could not be set up (it would fail again), no server listening, a timeout (the request may still be running)."""
    import socket
    import urllib.error

    from metalscope import client
    from metalscope.learn import calibrate as K
    monkeypatch.setattr(K, "DROP_PAUSES", (0, 0))
    prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    good = dict(text="t</think>" + "a" * 50, answer="a" * 50, tool=False, seconds=1.0)
    rows, said = [], []
    ask, asks = _asks_through_a_client(monkeypatch, ["cut", urllib.error.URLError(ConnectionResetError("reset"))])
    K.converse(lambda prompt: ask(prompt) and good, prefix, ["hi", "and?"], rows.append, log=said.append)
    assert len(asks) == 4 and [(r["turn"], r["attempt"]) for r in rows] == [(1, 1), (2, 1)]
    assert said == ["the server dropped the reply (1): asking again in 0 s",
                    "the server dropped the reply (2): asking again in 0 s"]
    steer = client.ServerError("POST /v1/completions: HTTP 503: metalscope could not set up steering: no vector", 503,
                               "metalscope could not set up steering: no vector")
    for exc in (client.ServerError("POST /v1/completions: HTTP 400: bad", 400, "bad"), steer):
        def refused(prompt, exc=exc):
            said.append("asked")
            raise exc
        said.clear()
        with pytest.raises(client.ServerError) as got:
            K.converse(refused, prefix, ["hi"], rows.append, log=said.append)
        assert got.value is exc and said == ["asked"]
    for failure, why in ((urllib.error.URLError(ConnectionRefusedError("refused")), "no metalscope server at http://omlx"),
                         (urllib.error.URLError(socket.timeout("timed out")), "timed out"),
                         (TimeoutError("timed out"), "timed out")):
        ask, asks = _asks_through_a_client(monkeypatch, [failure] * 3)
        with pytest.raises(client.ServerError, match=why):
            K.converse(ask, prefix, ["hi"], rows.append, log=said.append)
        assert len(asks) == 1, failure
    ask, asks = _asks_through_a_client(monkeypatch, ["cut"] * 3)
    with pytest.raises(K.Dropped) as got:
        K.converse(ask, prefix, ["hi"], rows.append, log=said.append)
    assert str(got.value) == "the server dropped the reply 3 times (last: POST /v1/completions: HTTP 500: )"
    assert len(asks) == 3 and isinstance(got.value.__cause__, client.ServerError)
    # The grader: a drop asked again, then graded; dropped every time, that version is left ungraded.
    grades = json.dumps([dict(turn=1, kind="turn", style=2, coherence=3, broken=False, why="")])
    ask, asks = _asks_through_a_client(monkeypatch, ["cut"])
    got = K.grade_file(lambda prompt: ask(prompt) and grades, "text", "casual", {"V1": [(1, "turn")]}, log=said.append)
    assert got["V1"][0]["style"] == 2 and len(asks) == 2
    ask, asks = _asks_through_a_client(monkeypatch, ["cut"] * 9)
    said.clear()
    assert K.grade_file(ask, "text", "casual", {"V1": [(1, "turn")]}, log=said.append) == {"V1": None}
    assert len(asks) == 3 and said[-1] == ("grading casual V1: the server dropped the reply 3 times "
                                          "(last: POST /v1/completions: HTTP 500: ); left ungraded")


def test_a_calibration_whose_replies_are_dropped_every_time_says_what_happened_and_what_to_do(monkeypatch):
    import http.client

    from metalscope import config
    from metalscope.learn import calibrate as K
    from metalscope.omlx import admin as omlx_admin
    server, _ = _calibration_setup(monkeypatch, lambda dose, turn: 2)
    monkeypatch.setattr(K, "DROP_PAUSES", (0, 0))

    def dropped(self, *a, **k):
        raise http.client.IncompleteRead(b"x")
    monkeypatch.setattr(_ChatSession, "complete", dropped)
    for engine, log in (("omlx", f"oMLX's log says why: {omlx_admin.OMLX_LOG}"),
                        ("ds4", f"ds4-server's log says why: {config.home() / 'ds4' / 'ds4-server.log'}")):
        server.status = lambda engine=engine: dict(engine=engine)
        with pytest.raises(RuntimeError) as exc:
            K.calibrate("toy", "m", server=server, log=lambda m: None)
        assert str(exc.value) == ("the server dropped the reply 3 times (last: IncompleteRead(1 bytes read)); "
                                  f"out of memory? {log}; run it again: metalscope concept calibrate toy m")
        assert isinstance(exc.value.__cause__, K.Dropped)


def test_the_qwen_opening_is_the_templates_system_message_without_tools():
    from metalscope.learn import calibrate as K
    # Qwen3.8 (Flash Next, on either engine, a linked copy too): its effort sentence at low, then the system text.
    low = ("<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
           "the conclusion without unnecessary elaboration.\n\nYou are a helpful assistant.<|im_end|>\n")
    models.link("flash", "Qwen3.8-Flash-Next")
    assert K.context("Qwen3.8-Flash-Next-oQ4e-mtp") == K.context("Qwen3.8-Flash-Next-ds4-q4k") == K.context("flash") == low
    # Qwen3.6's and Qwen3.5's templates have no effort sentence.
    plain = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    assert K.context("Qwen3.6-35B-A3B-MLX-4bit") == K.context("mlx-community/Qwen3.5-2B-4bit") == plain
    assert K.render_chat(low, [], "Hey!") == low + "<|im_start|>user\nHey!<|im_end|>\n<|im_start|>assistant\n<think>\n"
    with pytest.raises(RuntimeError, match="none of them"):
        K.context("Llama-3.1-8B")


def test_the_deepseek_v41_opening_is_its_system_message_with_thinking_on():
    from metalscope.learn import calibrate as K
    v4 = ("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough "
          "the reasoning)\n\nYou are a helpful assistant.")
    assert K.context("DeepSeek-V4.1-Flash-ds4-q2") == v4
    assert K.render_chat(v4, [], "Hey!") == v4 + "<｜User｜>Hey!<｜Assistant｜><think>"


def test_inkling_2_converses_in_its_channels_at_low_effort_and_its_parsed_replies_read_as_opened():
    """Inkling 2's opening is its template's: the system message, then its effort line at "low" (0.2); a turn closes
    the user's message and the model opens its own. oMLX's parser gives a reply as <think>THINKING</think>ANSWER (or the
    answer alone): read as if the prompt had opened the thinking, as the other formats' are."""
    from metalscope.learn import calibrate as K
    ink = ("<|message_system|><|content_text|>You are a helpful assistant.<|end_message|>"
           "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>")
    assert K.context("Inkling-2-Mini-oQ4e-mtp") == ink
    assert K.render_chat(ink, [], "Hey!") == ink + "<|message_user|><|content_text|>Hey!<|end_message|>"
    assert K.render_chat(ink, [("hi", "hm", "Hello!"), ("and?", "", "Yes.")], "more") == ink + (
        "<|message_user|><|content_text|>hi<|end_message|>"
        "<|message_model|><|content_thinking|>hm<|end_message|>"
        "<|message_model|><|content_text|>Hello!<|end_message|><|content_model_end_sampling|>"
        "<|message_user|><|content_text|>and?<|end_message|>"                               # no thinking: none rendered
        "<|message_model|><|content_text|>Yes.<|end_message|><|content_model_end_sampling|>"
        "<|message_user|><|content_text|>more<|end_message|>")
    prompt = K.render_chat(ink, [], "Hey!")
    answer = "Hello! " + "x" * 40

    def r(text):
        text = K.opened(prompt, text)
        return dict(text=text, answer=K.answer_text(text), tool=K.is_tool(text))
    assert K.opened(prompt, "<think>hm</think>" + answer) == "hm</think>" + answer
    assert r("<think>hm</think>" + answer)["answer"] == answer and K.failed(r("<think>hm</think>" + answer)) is None
    assert r(answer)["answer"] == answer and K.failed(r(answer)) is None                 # no thinking written: the answer
    assert K.failed(r("<think>still thinking")) == "thinking"                            # cut short in its thinking
    assert K.failed(r("<think>hm</think>")) == "short"                                   # a tool call alone: left out
    assert K.failed(r("<think>hm</think><|content_invoke_tool_json|>{}")) == "tool"      # unparsed, it would be seen
    assert K.history_turn("u", K.opened(prompt, "<think>hm</think>Sure.")) == ("u", "hm", "Sure.")
    assert K.history_turn("u", K.opened(prompt, "Sure.")) == ("u", "", "Sure.")
    assert K.opened("<|im_start|>system\nS<|im_end|>\n", "hm</think>Hi") == "hm</think>Hi"   # other formats: as is
    assert K._shown(K.opened(prompt, answer)) == "<think></think>" + answer


def test_conversations_render_each_template_and_fail_as_a_user_sees_it():
    from metalscope.learn import calibrate as K
    q = "<|im_start|>system\nS<|im_end|>\n"
    assert K.render_chat(q, [("hi", "hm", "Hello!")], "more") == (
        q + "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\nhm\n</think>\n\nHello!<|im_end|>\n"
        "<|im_start|>user\nmore<|im_end|>\n<|im_start|>assistant\n<think>\n")
    d = "<｜begin▁of▁sentence｜><｜System｜>S"
    assert K.render_chat(d, [("hi", "hm", "Hello!")], "more") == (
        d + "<｜User｜>hi<｜Assistant｜><think>hm</think>Hello!<｜end▁of▁sentence｜><｜User｜>more<｜Assistant｜><think>")
    with pytest.raises(RuntimeError, match="none"):
        K.render_chat("<s>[INST] S", [], "hey")
    r = lambda text: dict(text=text, answer=K.answer_text(text), tool=K.is_tool(text))
    assert K.failed(r("hm</think>\n\n<tool_call>\n<function=bash>")) == "tool"
    assert K.failed(r("still thinking")) == "thinking" and K.failed(r("hm</think>Hi!<|im_end|>")) == "short"
    assert K.failed(r("hm</think>" + "x" * 40)) is None
    assert K.history_turn("u", "hm</think>Sure.<tool_call>x") == ("u", "hm", "Sure.")
    assert K.history_turn("u", "thinking forever") == ("u", "", "(no answer)")
    assert K.header("toy", 5, 0.4) == "toy@0.4/5" and K.header("toy", 5, 0.0) == ""
    assert K.style_of({"description": "Soft, fluffy, furry textures against hard, cold, smooth ones. No animals."}) == \
        "Soft, fluffy, furry textures"
    assert K.style_of({"description": "Soft things.", "learned": {"request": "Talk like an otaku."}}) == "Talk like an otaku"
    assert K.target_side("Cooperating (giving ground against one's gain) against defecting") == "Cooperating (giving ground against one's gain)"


def test_grades_are_read_back_only_when_every_entry_is_usable():
    from metalscope.learn import calibrate as K
    want = [(1, "turn"), (2, "turn")]
    ok = [dict(version="V2", turn=1, kind="turn", style=2, coherence=3, useful=3, broken=False, why=""),
          dict(version="V2", turn=2, kind="turn", style="3", coherence=1, useful=1, broken="true", why="loops")]
    got = K.parse_chat_grades("Here you go:\n" + json.dumps(ok) + "\nDone.", want)
    assert [(g["turn"], g["style"], g["coherence"], g["broken"]) for g in got] == [(1, 2, 3, False), (2, 3, 1, True)]
    assert K.parse_chat_grades(json.dumps(ok[:1]), want) is None                                   # one missing
    assert K.parse_chat_grades(json.dumps([ok[0], dict(ok[1], style=4)]), want) is None            # out of range
    assert K.parse_chat_grades(json.dumps([ok[0], dict(ok[1], broken=None)]), want) is None
    assert K.parse_chat_grades(json.dumps({"grades": ok}), want) is not None
    # A grader that answers every version in one array: only the asked version's entries are filed under it.
    other = [dict(g, version="V1", style=0) for g in ok]
    got = K.parse_chat_grades(json.dumps(other + ok), want, "V2")
    assert [g["style"] for g in got] == [2, 3]
    assert K.parse_chat_grades(json.dumps(other), want, "V2") is None
    assert K.parse_chat_grades(json.dumps([{k: v for k, v in g.items() if k != "version"} for g in ok]), want, "V2")
    for written in (2, "2", "v2", "Version 2", " V 2 "):                # the same version, written otherwise: kept
        assert K.parse_chat_grades(json.dumps([dict(g, version=written) for g in ok]), want, "V2") is not None
    assert K.parse_chat_grades(json.dumps([dict(g, version="Version 12") for g in ok]), want, "V2") is None
    # The grader file: versions by label only, the style looked for, no dose anywhere.
    text = K.blind_text("casual", [("V1", [_row("casual", 0.6, 1, answer=None)]), ("V2", [_row("casual", 0.0, 1)])], "Fluffy")
    assert "## Version V1" in text and "## Version V2" in text and "The style being looked for: Fluffy." in text
    assert "never finished thinking" in text and "0.6" not in text and "PROBE" not in text
    prompt = K.grade_prompt(text, "casual", "V2", 1)
    assert "6-turn conversation" in prompt and "Grade only version V2 now: every entry of V2 (1 entries)" in prompt
    assert "detects" not in prompt


class _ChatSession:
    """Pi's conversations, faked: thinking then an answer naming its dose; at FAIL and above thinking never ends."""
    calls: list = []
    fail = 0.55
    ramble = staticmethod(lambda dose: False)  # the doses (under FAIL) whose answers run on
    stop_after = None

    def __init__(self, server, client, feed=True):
        assert not feed                    # calibration reads the replies' text only: no live feed


    def complete(self, model, prompt, steer, max_tokens, traced=True):
        assert traced == (not steer)       # only the plain replies are traced: the strength table may be measured on them
        dose = float(steer.split("@", 1)[1].split("/", 1)[0]) if "@" in steer else 0.0
        type(self).calls.append((dose, prompt))
        if self.stop_after is not None and len(self.calls) > self.stop_after:
            raise KeyboardInterrupt("stopped")
        user = prompt.rsplit("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        if dose >= self.fail:
            return dict(text="hmm " * 10, seconds=1.0)
        on = " and so on" * 200 if self.ramble(dose) else ""
        return dict(text=f"ok</think>A friendly reply, dose={dose:g}, to: {user}{on}<|im_end|>", seconds=1.0)

    def close(self):
        pass


def _calibration_setup(monkeypatch, style, name="toy"):
    """A served model ``m`` (a copy of CKPT, no traces) with concept ``name`` (steer layer 5); its replies from
    _ChatSession, its grades from ``style(dose, turn)`` (broken where thinking never ended). Returns the server and the
    grade requests (how many versions each file had)."""
    import re
    from types import SimpleNamespace
    from metalscope import client
    models.link("m", CKPT)
    d = models.concept_dir("m", name)               # the model's own texts: the grader's style
    d.mkdir(parents=True, exist_ok=True)
    (d / "concept.json").write_text(json.dumps(dict(description="Talk like a toy. Notes follow.")))
    (d / "pos.txt").write_text("a\n")
    prof = models.profile_path("m", name)
    prof.write_text("{}")
    listing = dict(steer_enabled=True, concepts=[dict(name=name, reader=dict(layer=5, strength=0.4))])
    monkeypatch.setattr(client, "Session", _ChatSession)
    _ChatSession.calls, _ChatSession.stop_after = [], None
    graded = []

    def post(path, body, timeout=None, headers=None):
        assert path == "/v1/chat/completions" and body["temperature"] == 0 and headers == client.UNTRACED
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        prompt = body["messages"][0]["content"]
        v = re.search(r"Grade only version (V\d+) now", prompt).group(1)
        graded.append(prompt.count("## Version "))
        section = prompt.split(f"## Version {v}\n", 1)[1].split("## Version ", 1)[0].split("</file>", 1)[0]
        out = []
        for turn, answer in re.findall(r"### V\d+ · Turn (\d+)\n\*\*User:\*\* .*\n\n\*\*Assistant:\*\* (.*)\n", section):
            if "never finished" in answer:
                out.append(dict(turn=int(turn), kind="turn", style=0, coherence=0, broken=True, why="no answer"))
            else:
                dose = float(re.search(r"dose=([\d.]+)", answer).group(1))
                out.append(dict(turn=int(turn), kind="turn", style=style(dose, int(turn)), coherence=3, broken=False, why=""))
        return {"choices": [{"message": {"content": json.dumps(out)}}]}
    return SimpleNamespace(get=lambda path: listing, post=post), graded


def test_calibration_searches_grades_everything_together_and_writes_version_4(monkeypatch):
    from metalscope.concepts import calibration_of
    from metalscope.learn import calibrate as K
    # 0.3 and 0.4 style every reply, 0.2 hints; from 0.55 thinking never ends.
    server, graded = _calibration_setup(monkeypatch, lambda dose, turn: 2 if dose >= 0.3 else 1 if dose > 0 else 0)
    progress = []
    out = K.calibrate("toy", "m", server=server, log=lambda m: None, on_progress=lambda n, of: progress.append((n, of)))
    # The progress total is exact from the first grading on (its 12 requests, 36 turns, then 18 requests).
    k = progress[-1][0]
    late = [(n, of) for n, of in progress[:-1] if n > k - 12 - 36 - 18]
    assert len(late) == 66 and all(of == max(k, n + 1) for n, of in late) and progress[-1] == (k, k)
    # Plain, 0.2, 0.4, 0.6 (every attempt failed: 0.8 skipped), then 0.1 either side of the best (0.4).
    doses = list(dict.fromkeys(d for d, _ in _ChatSession.calls))
    assert doses == [0.0, 0.2, 0.4, 0.6, 0.3, 0.5] and sum(1 for d, _ in _ChatSession.calls if d == 0.6) == 54
    assert out["passes"] == [[0.2, 0.4, 0.6], [0.3, 0.5]]
    # Every grading request holds plain and every dose run so far: 4 versions a file, then all 6 again.
    assert graded == [4] * 12 + [6] * 18
    # 0.3, 0.4 and 0.5 have 18 good each: the most, the lower on a tie. No pass or fail.
    assert (out["default_dose"], out["good"], out["turns"]) == (0.3, 18, 18)
    assert out["reason"] == "18/18 good @0.3"
    rows = {s["dose"]: s for s in out["strengths"]}
    assert (rows[0.2]["good"], rows[0.6]["good"], rows[0.6]["attempts"], rows[0.6]["failed"]["thinking"]) == (0, 0, 54, 54)
    assert out["plain"]["good"] == 0 and out["plain"]["attempts"] == 18
    # No trace on record: the built-in opening in the checkpoint's format at low reasoning effort, no tools, every
    # reply steered at its dose (thinking included: the header is per request).
    assert all(p.startswith(K.context(CKPT) + "<|im_start|>user\n") for _, p in _ChatSession.calls)
    assert all(K.QWEN_LOW in p and "# Tools" not in p for _, p in _ChatSession.calls)
    cal = json.loads(models.calibration_path("m", "toy").read_text())
    assert cal["context"].startswith("a built-in system prompt (You are a helpful assistant.), no tools")
    assert cal["version"] == 4 and cal["layer"] == 5 and cal["grader"] == dict(model="m", thinking=False)
    assert cal["default_dose"] == 0.3 and "read_layer" not in cal and cal["prefix_hash"] and cal["max_tokens"] == 4000
    assert calibration_of("m", "toy")["default_dose"] == 0.3
    d = models.calibration_dir("m", "toy")
    assert sorted(p.name for p in d.iterdir()) == ["calibration.json", "grades.json", "replies.jsonl"]
    rounds = json.loads(models.grades_path("m", "toy").read_text())["rounds"]
    assert [r["strengths"] for r in rounds] == [[0.2, 0.4, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6]] and not rounds[1]["ungraded"]
    # 0.6 broke by its failed attempts alone (no reply ran away): still graded.
    assert [r["left_out"] for r in rounds] == [[], []] and "runaway" not in rows[0.6] and rows[0.6]["graded"] == 18
    assert len(models.replies_path("m", "toy").read_text().splitlines()) == 18 * 5 + 54
    # One message plain and steered at the default: the middle good turn in conversation order.
    s = out["sample"]
    assert (s["arc"], s["turn"], s["dose"], s["good"], s["style"]) == ("dinner", 3, 0.3, True, 2)
    assert "dose=0.3" in s["steered"] and "dose=0" in s["plain"] and s["prompt"] == K.ARCS["dinner"][2]


def _graded_prompts(server) -> list[str]:
    """Every grading prompt ``server`` (_calibration_setup's) is sent, as it is sent."""
    prompts, post = [], server.post
    server.post = lambda path, body, **kw: (prompts.append(body["messages"][0]["content"]), post(path, body, **kw))[1]
    return prompts


def test_a_dose_whose_replies_run_away_skips_the_stronger_ones_and_is_not_graded(monkeypatch):
    """Replies RUNAWAY times the plain ones' length (answered, so not failed: loops to the budget) end the first pass,
    and the second adds nothing above that dose. With one dose left the first grading is skipped (0.2 would be graded
    alone); the runaway dose never reaches the grader: no grades, 0 good, marked."""
    from metalscope.learn import calibrate as K
    server, graded = _calibration_setup(monkeypatch, lambda dose, turn: 2 if dose >= 0.3 else 1 if dose > 0 else 0)
    prompts = _graded_prompts(server)
    monkeypatch.setattr(_ChatSession, "ramble", staticmethod(lambda d: d >= 0.35))
    said, progress = [], []
    out = K.calibrate("toy", "m", server=server, log=said.append, on_progress=lambda n, of: progress.append((n, of)))
    # No first grading: the best is 0.2 (0.4 counts 0 good), so the second pass is 0.1 and 0.3 (no 0.6, 0.8).
    assert list(dict.fromkeys(d for d, _ in _ChatSession.calls)) == [0.0, 0.2, 0.4, 0.1, 0.3]
    assert out["passes"] == [[0.2, 0.4], [0.1, 0.3]] and out["default_dose"] == 0.3
    assert any(m.startswith("dose 0.4 ran ") and m.endswith("times the plain replies' length: stronger doses skipped") for m in said)
    assert any(m.startswith("first grading skipped: ") for m in said)
    rows = [json.loads(l) for l in models.replies_path("m", "toy").read_text().splitlines()]
    assert K.length_ratio(rows, 0.4) >= K.RUNAWAY > K.length_ratio(rows, 0.3) and K.broke(rows, 0.3) is None
    assert K.failed_share(rows, 0.4) == 0 and K.length_ratio(rows, 0.9) == 0
    # One grading: plain, 0.1, 0.2 and 0.3; 0.4's replies in no file, under no label.
    assert graded == [4] * 12 and not any("dose=0.4" in p or "and so on" in p for p in prompts)
    rounds = json.loads(models.grades_path("m", "toy").read_text())["rounds"]
    assert [(r["strengths"], r["left_out"]) for r in rounds] == [([0.1, 0.2, 0.3, 0.4], [0.4])]
    assert 0.4 not in rounds[0]["key"].values() and all(g["dose"] != 0.4 for g in rounds[0]["grades"])
    assert sum(m == f"dose 0.4 {K.ran_words(K.ran_away(rows, 0.4))}: not graded" for m in said) == 1
    # The progress counted the 12 grading requests made, no more: its total from the first of them is the last count.
    assert progress[-1][0] == progress[-1][1] and progress[-13][1] == progress[-1][1]
    # Marked in calibration.json: 0 good, no grades; the others as before.
    cal = json.loads(models.calibration_path("m", "toy").read_text())
    s = {x["dose"]: x for x in cal["strengths"]}
    assert s[0.4]["runaway"] == K.ran_away(rows, 0.4) == round(K.length_ratio(rows, 0.4)) >= K.RUNAWAY
    assert (s[0.4]["good"], s[0.4]["graded"], s[0.4]["coherence"], s[0.4]["attempts"]) == (0, 0, None, 18)
    assert "runaway" not in s[0.3] and (s[0.3]["good"], s[0.3]["graded"]) == (18, 18) and "runaway" not in cal["plain"]
    # Its replies read as not graded.
    steered = [t["steered"] for c in K.conversations("m", "toy", 0.4) for t in c["turns"]]
    assert all(x["grade"] is None and not x["good"] and x["why_not"] == ["not graded"] for x in steered)
    assert K.sample_replies("m", "toy", 0.4)["good"] is False


def test_a_dose_that_runs_away_in_the_second_pass_is_left_out_of_the_second_grading(monkeypatch):
    """0.6 broke by failed attempts (graded), 0.5 of the second pass ran away: the first grading has every dose, the
    second all but 0.5."""
    from metalscope.learn import calibrate as K
    server, graded = _calibration_setup(monkeypatch, lambda dose, turn: 2 if dose >= 0.3 else 1 if dose > 0 else 0)
    prompts = _graded_prompts(server)
    monkeypatch.setattr(_ChatSession, "ramble", staticmethod(lambda d: d >= 0.45))
    out = K.calibrate("toy", "m", server=server, log=lambda m: None)
    assert out["passes"] == [[0.2, 0.4, 0.6], [0.3, 0.5]] and out["default_dose"] == 0.3
    assert graded == [4] * 12 + [5] * 15 and not any("dose=0.5" in p for p in prompts)
    rounds = json.loads(models.grades_path("m", "toy").read_text())["rounds"]
    assert [r["left_out"] for r in rounds] == [[], [0.5]] and any(g["dose"] == 0.6 for g in rounds[1]["grades"])
    s = {x["dose"]: x for x in out["strengths"]}
    assert s[0.5]["runaway"] >= K.RUNAWAY and s[0.5]["good"] == 0 and "runaway" not in s[0.6]


def test_a_runaway_dose_is_graded_rather_than_leave_one_dose_graded_alone(monkeypatch):
    """0.2 runs away, so 0.4 and up are skipped, the first grading too (no dose held); the second pass is 0.1 alone
    (nothing above 0.2). The final grading then holds 0.1 and 0.2: the runaway dose graded like any other, marked."""
    from metalscope.learn import calibrate as K
    server, graded = _calibration_setup(monkeypatch, lambda dose, turn: 2 if dose >= 0.2 else 1 if dose > 0 else 0)
    prompts = _graded_prompts(server)
    monkeypatch.setattr(_ChatSession, "ramble", staticmethod(lambda d: d >= 0.15))
    said = []
    out = K.calibrate("toy", "m", server=server, log=said.append)
    assert list(dict.fromkeys(d for d, _ in _ChatSession.calls)) == [0.0, 0.2, 0.1]
    assert out["passes"] == [[0.2], [0.1]] and any(m.startswith("first grading skipped: ") for m in said)
    assert graded == [3] * 9 and any("dose=0.2" in p for p in prompts)
    rounds = json.loads(models.grades_path("m", "toy").read_text())["rounds"]
    assert [(r["strengths"], r["left_out"]) for r in rounds] == [([0.1, 0.2], [])]
    # Graded, so its real numbers, and the mark that it ran away; with more good replies it is the default (ranked,
    # not gated out).
    s = {x["dose"]: x for x in out["strengths"]}
    assert s[0.2]["runaway"] >= K.RUNAWAY and (s[0.2]["good"], s[0.2]["graded"]) == (18, 18) and "runaway" not in s[0.1]
    assert out["default_dose"] == 0.2


@pytest.mark.parametrize("why", ["ran away", "grader gave nothing"])
def test_the_default_is_a_graded_dose_even_with_no_good_reply_anywhere(monkeypatch, why):
    """0.2 holds, 0.4 fails every attempt (graded); the second pass's 0.1 has no grades (it ran away and is left out,
    or it held and the grader never gave usable grades for it), 0.3 holds. No reply is good anywhere: the default is
    the lowest graded dose, not the lower ungraded one."""
    from metalscope.learn import calibrate as K
    nothing = why == "grader gave nothing"
    server, _ = _calibration_setup(monkeypatch, lambda dose, turn: (None if nothing and dose == 0.1 else 1) if dose > 0 else 0)
    monkeypatch.setattr(_ChatSession, "fail", 0.35)
    monkeypatch.setattr(_ChatSession, "ramble", staticmethod(lambda d: not nothing and 0 < d < 0.15))
    out = K.calibrate("toy", "m", server=server, log=lambda m: None)
    rounds = json.loads(models.grades_path("m", "toy").read_text())["rounds"]
    assert out["passes"] == [[0.2, 0.4], [0.1, 0.3]] and rounds[-1]["left_out"] == ([] if nothing else [0.1])
    s = {x["dose"]: x for x in out["strengths"]}
    assert s[0.1]["graded"] == 0 and ("runaway" in s[0.1]) != nothing
    assert all(x["good"] == 0 for x in out["strengths"]) and (out["default_dose"], out["good"]) == (0.2, 0)


def test_a_tie_goes_to_the_dose_that_did_not_run_away(monkeypatch):
    """0.2 holds, 0.4 runs away, then 0.1 and 0.3 too: 0.1 is graded to make two, and ties 0.2 at 9 good. The default
    is 0.2, the dose that held, though 0.1 is lower."""
    from metalscope.learn import calibrate as K
    server, _ = _calibration_setup(monkeypatch, lambda dose, turn: (2 if turn <= 3 else 1) if dose > 0 else 0)
    monkeypatch.setattr(_ChatSession, "ramble", staticmethod(lambda d: d >= 0.3 or 0 < d < 0.15))
    out = K.calibrate("toy", "m", server=server, log=lambda m: None)
    rounds = json.loads(models.grades_path("m", "toy").read_text())["rounds"]
    assert out["passes"] == [[0.2, 0.4], [0.1, 0.3]] and [r["left_out"] for r in rounds] == [[0.3, 0.4]]
    s = {x["dose"]: x for x in out["strengths"]}
    assert (s[0.1]["good"], s[0.2]["good"]) == (9, 9) and "runaway" in s[0.1] and "runaway" not in s[0.2]
    assert out["default_dose"] == 0.2


def test_runaway_and_failed_share_together_and_what_a_grading_leaves_out():
    """A dose that failed most attempts and ran long: skipped for its failures, marked as run away, left out of a
    grading; the weakest runaway doses stay in while fewer than two doses would be graded."""
    from metalscope.learn import calibrate as K
    long = "x" * 2000
    rows = [_row(a, 0.0, t) for a in K.ARCS for t in range(1, 7)]
    rows += [_row(a, d, t) for d in (0.2, 0.3) for a in K.ARCS for t in range(1, 7)]
    rows += [dict(_row(a, d, t, failed="thinking", answer=None), text=long) for d in (0.4, 0.6) for a in K.ARCS for t in range(1, 7)]
    assert K.broke(rows, 0.4) == "failed 100% of its attempts" and K.ran_away(rows, 0.4) == round(K.length_ratio(rows, 0.4)) >= K.RUNAWAY
    assert K.tally(rows, [], 0.4)["runaway"] == K.ran_away(rows, 0.4) and K.ran_away(rows, 0.3) is None
    assert K.leave_out(rows, [0.2, 0.3, 0.4, 0.6]) == [0.4, 0.6]
    assert K.leave_out(rows, [0.2, 0.4, 0.6]) == [0.6] and K.leave_out(rows, [0.4, 0.6]) == [] and K.leave_out(rows, [0.4]) == []


def test_a_calibration_is_of_the_dose_units_it_was_made_in(monkeypatch):
    """A calibration records the served copy's strength table it ran under (norms.json's made and tokens); a new
    table (`vector norms --refresh`) rescales every dose, so the calibration then reads as out of date. One made
    before calibrations recorded it counts as made in the table there is when that table is older than it."""
    import os
    import time
    from metalscope import cli, steer
    from metalscope.concepts import calibration_of
    from metalscope.learn import calibrate as K
    from .helpers import write_norms
    server, _ = _calibration_setup(monkeypatch, lambda dose, turn: 2 if dose >= 0.2 else 0)
    write_norms("m", [5])
    norms = models.norms_path("m")
    table = lambda made, tokens: norms.write_text(json.dumps(dict(json.loads(norms.read_text()), made=made, tokens=tokens)))
    table("2026-11-01T10:00:00", 300)
    logged = []
    out = K.calibrate("toy", "m", server=server, log=logged.append)
    assert out["norms"] == dict(made="2026-11-01T10:00:00", tokens=300) and calibration_of("m", "toy") is not None
    assert "warning: dose units on m measured on 1 short trace (300 tokens): `metalscope vector norms m --refresh` " \
           "after a real chat" in logged
    table("2026-11-02T10:00:00", 9000)                                   # vector norms m --refresh
    assert calibration_of("m", "toy") is None
    # Unstamped: current while the table is older than the calibration, out of date once a refresh makes a newer one.
    p = models.calibration_path("m", "toy")
    p.write_text(json.dumps({k: v for k, v in json.loads(p.read_text()).items() if k != "norms"}))
    assert calibration_of("m", "toy")["default_dose"] == out["default_dose"]
    later = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 120))
    monkeypatch.setattr(steer, "norm_table", lambda name, refresh=False: (table(later, 9000), dict(
        json.loads(norms.read_text()), layers={5: 10.0}, streams={}))[1])
    with pytest.MonkeyPatch.context() as mp:
        printed = []
        mp.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
        cli.main(["vector", "norms", "m", "--refresh"])
    assert calibration_of("m", "toy") is None
    assert "1 chat calibration on m is out of date now (new dose units): metalscope concept calibrate toy m" in printed
    os.utime(p, (time.time() + 300,) * 2)                                # written after the new table: current
    assert calibration_of("m", "toy") is not None
    # A new table during the run: part of it was in other units, so nothing is written.
    real = _ChatSession.complete

    def complete(self, *a, **k):
        if len(_ChatSession.calls) == 30:
            table("2026-11-03T10:00:00", 9000)
        return real(self, *a, **k)
    monkeypatch.setattr(_ChatSession, "complete", complete)
    _ChatSession.calls = []
    with pytest.raises(RuntimeError, match="m's dose units changed during the calibration: run it again"):
        K.calibrate("toy", "m", server=server, log=lambda m: None)
    assert len(_ChatSession.calls) == 36                 # at the end of the set it changed in (plain, then 0.2)


@pytest.mark.parametrize("name", ["toy", "calm"])       # calm ships with other texts: never the shipped ones
def test_calibration_grades_the_style_of_the_models_own_texts(monkeypatch, name):
    import re
    from metalscope import config
    from metalscope.concepts.corpus import shipped_dir
    from metalscope.learn import calibrate as K
    assert (shipped_dir(name) is not None) == (name == "calm")
    server, _ = _calibration_setup(monkeypatch, lambda dose, turn: 2 if dose >= 0.3 else 0, name)
    asked, post = [], server.post
    server.post = lambda path, body, **k: asked.append(body["messages"][0]["content"]) or post(path, body, **k)
    K.calibrate(name, "m", server=server, log=lambda m: None)
    assert asked and all("The style being looked for: Talk like a toy." in a for a in asked)
    # No texts of it on the model (a definition is not texts): refused, naming where they would be.
    texts = models.concept_dir("m", name) / "concept.json"
    texts.unlink()
    (config.home() / "concepts" / name).mkdir(parents=True)
    (config.home() / "concepts" / name / "concept.json").write_text(json.dumps(dict(name=name, description="Toy.")))
    with pytest.raises(RuntimeError, match=f"^{re.escape(f'no texts of {name} on {CKPT} ({texts.parent})')}$"):
        K.calibrate(name, "m", server=server, log=lambda m: None)
    # A concept.json without the texts beside it is not texts either.
    texts.write_text(json.dumps(dict(description="Talk like a toy.")))
    (texts.parent / "pos.txt").unlink()
    with pytest.raises(RuntimeError, match=f"^{re.escape(f'no texts of {name} on {CKPT} ({texts.parent})')}$"):
        K.calibrate(name, "m", server=server, log=lambda m: None)


def test_the_second_pass_tries_either_side_of_the_best_and_decides_from_the_regrading(monkeypatch):
    from metalscope.learn import calibrate as K
    monkeypatch.setattr(_ChatSession, "fail", 2.0)
    # Good turns per arc: 0.2: 0, 0.4: 3, 0.6: 2, 0.8: 1; then 0.3: 1 and 0.5: 4 (12 of 18, the most).
    cut = {0.2: 0, 0.4: 3, 0.6: 2, 0.8: 1, 0.3: 1, 0.5: 4}
    server, graded = _calibration_setup(monkeypatch, lambda dose, turn: 2 if dose and turn <= cut[dose] else 0)
    out = K.calibrate("toy", "m", server=server, log=lambda m: None)
    assert out["passes"] == [[0.2, 0.4, 0.6, 0.8], [0.3, 0.5]]
    assert graded == [5] * 15 + [7] * 21
    assert (out["default_dose"], out["good"]) == (0.5, 12) and out["reason"] == "12/18 good @0.5"


def test_plain_conversations_are_run_once_per_context_and_reused(monkeypatch):
    from metalscope.learn import calibrate as K
    q = K.context(CKPT)
    a = K.plain_key("m", q, 4000)
    assert a == K.plain_key("m", q, 4000) and len({a, K.plain_key("m2", q, 4000), K.plain_key("m", q + "x", 4000),
                                                    K.plain_key("m", q, 2000)}) == 4
    monkeypatch.setattr(K, "ATTEMPTS", 2)
    assert K.plain_key("m", q, 4000) != a          # the retry rules are in the key too
    monkeypatch.setattr(K, "ATTEMPTS", 3)
    server, _ = _calibration_setup(monkeypatch, lambda dose, turn: 2 if dose >= 0.2 else 0)
    K.calibrate("toy", "m", server=server, log=lambda m: None)
    assert sum(1 for d, _ in _ChatSession.calls if d == 0) == 18
    assert models.plain_chats_path("m", K.plain_key("m", K.context("m"), 4000)).exists()
    # Another concept on the same served model and context: no plain request, and its replies still hold plain's.
    models.concept_dir("m", "toy2").mkdir(parents=True)
    (models.concept_dir("m", "toy2") / "concept.json").write_text(json.dumps(dict(description="Toy two.")))
    (models.concept_dir("m", "toy2") / "pos.txt").write_text("a\n")
    models.profile_path("m", "toy2").write_text("{}")
    listing = dict(steer_enabled=True, concepts=[dict(name="toy2", reader=dict(layer=5, strength=0.4))])
    server.get = lambda path: listing
    _ChatSession.calls = []
    out = K.calibrate("toy2", "m", server=server, log=lambda m: None)
    assert not any(d == 0 for d, _ in _ChatSession.calls) and out["plain"]["attempts"] == 18
    # A first pass whose best is 0.2 (a tie with 0.4: the lower): 0.1 and 0.3, then the regrading.
    assert out["passes"] == [[0.2, 0.4, 0.6], [0.1, 0.3]] and out["default_dose"] == 0.2


def test_a_cancel_leaves_the_earlier_calibration_as_it_was(monkeypatch):
    from metalscope.learn import calibrate as K
    server, _ = _calibration_setup(monkeypatch, lambda dose, turn: 2 if dose >= 0.3 else 0)
    K.calibrate("toy", "m", server=server, log=lambda m: None)
    d = models.calibration_dir("m", "toy")
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    _ChatSession.calls, _ChatSession.stop_after = [], 30
    with pytest.raises(KeyboardInterrupt):
        K.calibrate("toy", "m", server=server, log=lambda m: None)
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before                  # no temporary file left either
    # The job ends "stopped"; the learn job's calibration lets a cancel through (an error is the step's summary).
    from metalscope.learn import command as Cmd

    def stop(*a, **k):
        raise KeyboardInterrupt("stopped")
    monkeypatch.setattr(K, "calibrate", stop)
    st = Cmd.run_calibrate_job("toy", "m", "toy-calibrate-9")
    assert (st["outcome"], st["error"], st["steps"][0]["summary"]) == ("failed", "stopped", "stopped")
    j = learn_job("x", "d", "/nonexistent", "m", admin=types.SimpleNamespace(base_url="http://x"), job="x-stop")
    with pytest.raises(KeyboardInterrupt):
        j.run_calibration()


def test_two_runs_on_one_concept_keep_their_own_temporary_files(monkeypatch):
    from metalscope.learn import calibrate as K
    server, _ = _calibration_setup(monkeypatch, lambda dose, turn: 2 if dose >= 0.3 else 0)
    d = models.calibration_dir("m", "toy")
    d.mkdir(parents=True)
    other = d / "replies.jsonl.1-abcdef.tmp"            # another run's, on the same concept and model
    other.write_text('{"theirs": 1}\n')
    seen = []
    real = _ChatSession.complete

    def complete(self, *a, **k):
        seen.append(sorted(p.name for p in d.iterdir()))
        return real(self, *a, **k)
    monkeypatch.setattr(_ChatSession, "complete", complete)
    _ChatSession.stop_after = 3
    with pytest.raises(KeyboardInterrupt):
        K.calibrate("toy", "m", server=server, log=lambda m: None)
    mine = [n for n in seen[-1] if n != other.name]
    assert len(mine) == 1 and mine[0].startswith("replies.jsonl.") and mine[0].endswith(".tmp")
    assert sorted(p.name for p in d.iterdir()) == [other.name] and other.read_text() == '{"theirs": 1}\n'


def test_a_foreground_calibration_is_a_job_while_it_runs(monkeypatch, capsys):
    from metalscope import cli
    from metalscope.learn import calibrate as K
    during = []

    def fake(name, model, log=None, on_progress=None, **k):
        during.append(state.busy())
        log("dose 0.6: 0 of 18 attempts failed (0%)")
        cal = dict(version=4, default_dose=0.6, good=15, turns=18, layer=18)
        return dict(cal, reason=K.summary_text(cal))
    monkeypatch.setattr(K, "calibrate", fake)
    cli.main(["concept", "calibrate", "fluffy", "m", "--url", "http://127.0.0.1:1"])
    assert during == ["a job is running: calibrating fluffy (/scope jobs)"]
    # Its stdout as before it was a job: the calibration's lines, no time in front, then the summary.
    assert capsys.readouterr().out == "dose 0.6: 0 of 18 attempts failed (0%)\n15/18 good @0.6\n"
    (st,) = state.list_jobs()
    assert (st["kind"], st["origin"], st["outcome"]) == ("calibrate", "cli", "done") and state.busy() is None

    def stop(*a, **k):                                              # Ctrl-C: the job ends, the machine is free
        during.append(state.busy())
        raise KeyboardInterrupt("stopped")
    monkeypatch.setattr(K, "calibrate", stop)
    with pytest.raises(SystemExit, match="error: stopped"):
        cli.main(["concept", "calibrate", "fluffy", "m", "--url", "http://127.0.0.1:1"])
    assert during[-1].startswith("a job is running") and state.busy() is None and capsys.readouterr().out == ""
    assert [j["outcome"] for j in state.list_jobs()] == ["done", "failed"]


def test_a_calibrate_job_writes_its_state(monkeypatch):
    from metalscope.learn import calibrate as K, command as Cmd

    def fake(name, model, log=None, on_progress=None, **k):
        on_progress(25, 51)
        log("halfway")
        cal = dict(version=4, default_dose=0.6, good=15, turns=18, layer=18)
        return dict(cal, reason=K.summary_text(cal))
    monkeypatch.setattr(K, "calibrate", fake)
    st = Cmd.run_calibrate_job("fluffy", "m", "fluffy-calibrate-1")
    assert st["outcome"] == "done" and st["kind"] == "calibrate" and st["result"]["calibration"]["default_dose"] == 0.6
    assert st["steps"][0]["status"] == "done" and st["steps"][0]["summary"] == "15/18 good @0.6"
    assert state.read_state("fluffy-calibrate-1")["log"] == ["halfway"]
    assert state.outcome_text(st) == "15/18 good @0.6"

    def boom(*a, **k):
        raise RuntimeError("HTTP 409: the model is unloaded")
    monkeypatch.setattr(K, "calibrate", boom)
    st = Cmd.run_calibrate_job("fluffy", "m", "fluffy-calibrate-2")
    assert st["outcome"] == "failed" and st["error"] == "HTTP 409: the model is unloaded" and st["steps"][0]["status"] == "failed"


def test_a_job_from_the_terminal_does_not_start_while_a_backend_switch_is_under_way(monkeypatch):
    """The switch looked for a running job before this one was there, and is about to stop the model under it. A job
    the server started is not asked: its launch held the switch lock itself."""
    from metalscope import backends
    from metalscope.learn import calibrate as K, command as Cmd
    ran = []

    def fake(*a, **k):
        ran.append(1)
        return dict(version=4, default_dose=0.6, good=15, turns=18, layer=18, reason="")
    monkeypatch.setattr(K, "calibrate", fake)
    with backends.switching():
        st = Cmd.run_calibrate_job("fluffy", "m", "fluffy-cli")
        assert st["outcome"] == "failed" and st["error"] == "a backend switch is under way: try again in a moment" and ran == []
        assert Cmd.run_calibrate_job("fluffy", "m", "fluffy-pi", origin="pi")["outcome"] == "done" and ran == [1]
    assert Cmd.run_calibrate_job("fluffy", "m", "fluffy-cli")["outcome"] == "done" and ran == [1, 1]


@pytest.mark.parametrize("fails", [False, True])
def test_a_job_marked_seen_while_it_runs_stays_seen(monkeypatch, fails):
    """Pi marks a job seen (the server's route) while the job still writes its state: the job's next saves and its
    final one, done or failed, keep the mark, so its result is not announced again."""
    from metalscope.learn import calibrate as K, command as Cmd

    def fake(name, model, log=None, on_progress=None, **k):
        assert state.mark_seen("fluffy-seen")
        log("after the mark")
        on_progress(1, 2)
        if fails:
            raise RuntimeError("HTTP 500: busy")
        cal = dict(version=4, default_dose=0.6, good=15, turns=18, layer=18)
        return dict(cal, reason=K.summary_text(cal))
    monkeypatch.setattr(K, "calibrate", fake)
    st = Cmd.run_calibrate_job("fluffy", "m", "fluffy-seen")
    assert st["outcome"] == ("failed" if fails else "done")
    assert state.read_state("fluffy-seen")["seen"] is True and state.list_jobs()[-1]["seen"] is True


def test_a_new_run_of_a_reused_job_id_is_unseen(monkeypatch):
    """`--job ID` on a folder whose earlier run Pi had seen: the new run's result is announced too."""
    from metalscope.learn import calibrate as K, command as Cmd
    cal = dict(version=4, default_dose=0.6, good=15, turns=18, layer=18)
    monkeypatch.setattr(K, "calibrate", lambda *a, **k: dict(cal, reason=K.summary_text(cal)))
    Cmd.run_calibrate_job("fluffy", "m", "fluffy-again")
    assert state.mark_seen("fluffy-again") and state.read_state("fluffy-again")["seen"]
    Cmd.run_calibrate_job("fluffy", "m", "fluffy-again")
    assert state.read_state("fluffy-again")["seen"] is False


def test_a_failed_calibration_is_the_steps_summary(monkeypatch):
    from metalscope.learn import calibrate as K
    j = learn_job("x", "d", "/nonexistent", "m", admin=types.SimpleNamespace(base_url="http://x"), job="x-cal")

    def boom(*a, **k):
        k["on_progress"](3, 51)
        raise RuntimeError("HTTP 500: busy")
    monkeypatch.setattr(K, "calibrate", boom)
    assert j.run_calibration() == {"error": "HTTP 500: busy"}
    assert j.step("calibrate")["summary"] == "not calibrated: HTTP 500: busy" and j.step("calibrate")["status"] == "running"
    j._close_steps(failed=False)
    assert j.step("calibrate")["status"] == "done"
    # A run that can't be steered never calibrates: the step is skipped when the job closes.
    k = learn_job("y", "d", "/nonexistent", "m", admin=object(), job="y-cal")
    k.phase("reloading")
    k._close_steps(failed=False)
    assert k.step("calibrate")["status"] == "skipped" and k.step("reload")["status"] == "done"


def test_command_words_cannot_name_a_new_concept():
    from metalscope.learn import plan as P
    for name in ("list", "Status", "off", "calibrate"):
        with pytest.raises(ValueError, match="is a command word"):
            spawn.learn(name, "something", "m")
    for name in ("1984", "0.5", "1e3", "0x1F"):                     # /steer would read them as doses
        with pytest.raises(ValueError, match="reads as a number"):
            P.check_new_name(name)
    for name in ("r2d2", "e3", "x0.5"):
        P.check_new_name(name)
    with pytest.raises(ValueError, match="name the concept"):
        spawn.calibrate("  ", "m")


def test_a_command_line_learn_ends_with_the_chat_count_in_words_and_the_next_command(capsys):
    from metalscope.learn import calibrate as K, command as Cmd
    cal = dict(version=4, default_dose=0.5, good=15, turns=18, layer=12)
    held = dict(steerable=True, reason=None, layer=12, calibration=dict(cal, reason=K.summary_text(cal)))
    assert Cmd.learn_words(held, "joyful", "Q").splitlines() == [
        "learned joyful on Q: steer layer L12",
        "in chat: 15/18 good @0.5",                                  # as `concept calibrate` prints it
        "next: metalscope steering on, then send the header X-Scope-Steer: joyful (in Pi: /steer joyful)",
        "details: metalscope concept show joyful Q"]
    worse = dict(held, calibration=dict(cal, good=8, default_dose=0.2))
    assert Cmd.learn_words(worse, "joyful", "Q").splitlines()[1] == "in chat: 8/18 good @0.2"
    assert Cmd.learn_words(dict(held, calibration=dict(error="stopped")), "joyful", "Q").splitlines()[1:] == [
        "in chat: not calibrated (stopped)", "next: metalscope concept calibrate joyful Q"]
    assert Cmd.learn_words(dict(steerable=False, reason="can't be steered: no item had room to move"), "joyful", "Q").splitlines() == [
        "joyful can't be steered on Q: no item had room to move", "what was measured: metalscope concept show joyful Q"]
    # The command's end: those words, no JSON; a job that did not finish exits with its error.
    args = types.SimpleNamespace(job=None)
    done = types.SimpleNamespace(run=lambda: dict(job="j", outcome="done", error=None, result=held))
    Cmd._main(args, lambda path: done, None, copy=False, words=lambda st: Cmd.learn_words(st["result"], "joyful", "Q"))
    out = capsys.readouterr().out
    assert out.startswith("learned joyful on Q: steer layer L12\n") and "{" not in out
    failed = types.SimpleNamespace(run=lambda: dict(job="j", outcome="failed", error="the GPU run exited with code 1", result=None))
    with pytest.raises(SystemExit, match="error: the GPU run exited with code 1"):
        Cmd._main(args, lambda path: failed, None, copy=False, words=lambda st: "")
    assert capsys.readouterr().out == ""


def test_a_padded_name_on_the_command_line_is_learned_and_named_without_its_spaces(monkeypatch, capsys):
    from metalscope import cli
    held = dict(steerable=True, reason=None, layer=12, calibration=dict(error="stopped"))
    made = []

    def run(args, make, report, copy=True, words=None):
        made.append(make(None))
        print(words(dict(result=held)))
    monkeypatch.setattr(Cmd, "_main", run)
    monkeypatch.setattr(P, "pick_engine", lambda name, url=None: "ds4")
    cli.main(["concept", "learn", " toasty ", "m", "--description", "d"])
    assert made[0].name == "toasty"
    assert state.busy() == "a job is running: learning toasty (/scope jobs)"     # made, not yet run: it holds its lock
    made[0].live.release()
    assert capsys.readouterr().out.splitlines() == ["learned toasty on m: steer layer L12",
                                                    "in chat: not calibrated (stopped)",
                                                    "next: metalscope concept calibrate toasty m"]


# The real command in its own process, its work replaced by a wait (nothing is asked of a server).
JOB_DRIVER = """
import sys, time
from metalscope.learn import calibrate as K, job as J, plan as P
from metalscope.omlx import admin
P.no_dose_units = lambda model_name: None
K.calibrate = lambda *a, **k: time.sleep(60)
J.GpuJob.run = lambda self: self._run(lambda: time.sleep(60))
admin.Admin = lambda *a, **k: object()
from metalscope import cli
cli.main(sys.argv[1:])
"""
NOWHERE = "http://127.0.0.1:1"


def _until(check, timeout=60):
    import time
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def _fds_on(path) -> list[int]:
    """This process's open descriptors on ``path``'s file."""
    import os
    want = os.stat(path)
    out = []
    for name in os.listdir("/dev/fd"):
        try:
            if os.path.samestat(os.fstat(int(name)), want):
                out.append(int(name))
        except OSError:
            pass
    return out


def _cli_job_runs(tmp_path, args, doing, env=None):
    """The command in its own process (JOB_DRIVER) reads as running from its first state: busy() names it and cancel
    stops it."""
    import os
    driver = tmp_path / "driver.py"
    driver.write_text(JOB_DRIVER)
    proc = subprocess.Popen([sys.executable, str(driver), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, **(env or {})))
    try:
        _until(lambda: state.list_jobs() or proc.poll() is not None)
        (st,) = state.list_jobs()
        assert st["outcome"] is None, (st, proc.poll())
        assert state.busy() == f"a job is running: {doing} (/scope jobs)"
        assert state.cancel(st["job"])
        assert proc.wait(timeout=30) != 0
    finally:
        proc.kill()
        proc.wait()
    st = state.read_state(st["job"])
    assert (st["outcome"], st["error"]) == ("failed", "stopped") and not state.cancel(st["job"]) and state.busy() is None


@pytest.mark.parametrize("args, doing", [
    (["concept", "learn", "--engine", "ds4", "--url", NOWHERE, "--description", "d", "joyful", "m"], "learning joyful"),
    (["concept", "learn", "joyful", "m", "--engine", "ds4", "--url", NOWHERE, "--description", "d"], "learning joyful"),
    (["concept", "calibrate", "--url", NOWHERE, "joyful", "m"], "calibrating joyful"),
    (["concept", "calibrate", "--url", NOWHERE, "joyful", "m", "--job", "joyful-cal-1"], "calibrating joyful"),
    (["jlens", "learn", "--windows", "3", "--model", "/nonexistent", "--need-gb", "1", "m"], "learning a lens for m"),
])
def test_a_command_line_job_reads_as_running_however_it_was_typed(tmp_path, args, doing):
    _cli_job_runs(tmp_path, args, doing)


@pytest.mark.parametrize("number", ["1", "987"])
def test_a_command_line_job_takes_its_own_lock_whatever_number_its_environment_names(tmp_path, number):
    """A lock number left in the environment (another open file, or none) is not the job's lock: it takes its own."""
    _cli_job_runs(tmp_path, ["concept", "calibrate", "--url", NOWHERE, "joyful", "m"], "calibrating joyful",
                  env={state.INHERITED: number})


def test_an_adopted_lock_must_be_the_jobs_own_file_and_is_never_passed_on(tmp_path, monkeypatch):
    """The descriptor spawn passed is adopted only when it is the job's live.lock, and is then closed to whatever the job
    starts (a grandchild holding it would keep a finished job running); another file's number is left alone."""
    import fcntl
    import os
    d = state.job_dir("x-adopt")
    d.mkdir(parents=True)
    fd = os.open(d / state.LIVE, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.set_inheritable(fd, True)
    monkeypatch.setenv(state.INHERITED, str(fd))
    lock = state.hold("x-adopt")
    assert lock.fd == fd and not os.get_inheritable(fd) and state.INHERITED not in os.environ
    other = os.open(tmp_path / "other", os.O_RDWR | os.O_CREAT)
    monkeypatch.setenv(state.INHERITED, str(other))
    own = state.hold("x-own")
    assert own.fd != other and _fds_on(state.job_dir("x-own") / state.LIVE) == [own.fd] and os.fstat(other)
    for f in (lock, own):
        f.release()
    os.close(other)


def test_a_job_the_server_starts_runs_from_its_first_state_and_a_kill_reads_as_interrupted(tmp_path, monkeypatch):
    """spawn's detached process holds the job's live lock from before its first state; killed -9, it reads interrupted."""
    import os
    import signal
    driver = tmp_path / "driver.py"
    driver.write_text(JOB_DRIVER)
    real = spawn.subprocess.Popen
    monkeypatch.setattr(spawn.subprocess, "Popen", lambda cmd, **kw: real([cmd[0], str(driver), *cmd[3:]], **kw))
    prof = models.profile_path("m", "joyful")
    prof.parent.mkdir(parents=True)
    prof.write_text("{}")
    out = spawn.calibrate("joyful", "m", url=NOWHERE)
    try:
        assert state.read_state(out["job"])["outcome"] is None and _fds_on(state.job_dir(out["job"]) / state.LIVE) == []
        with pytest.raises(ValueError, match="a job is running: calibrating joyful"):
            spawn.calibrate("joyful", "m", url=NOWHERE)
        _until(lambda: "steps" in state.read_state(out["job"]))           # its own state
        assert state.read_state(out["job"])["outcome"] is None and state.read_state(out["job"])["pid"] == out["pid"]
        os.kill(out["pid"], signal.SIGKILL)
        _until(lambda: state.read_state(out["job"])["outcome"] == "interrupted")
    finally:
        try:
            os.kill(out["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    assert state.busy() is None and not state.cancel(out["job"])


def test_the_launcher_never_adopts_a_lock_number_from_its_own_environment(tmp_path, monkeypatch):
    """The server's own environment naming a descriptor changes nothing: spawn takes the job's lock itself, passes it
    on and keeps no descriptor on it; the server's file stays open."""
    import os
    driver = tmp_path / "driver.py"
    driver.write_text(JOB_DRIVER)
    real = spawn.subprocess.Popen
    monkeypatch.setattr(spawn.subprocess, "Popen", lambda cmd, **kw: real([cmd[0], str(driver), *cmd[3:]], **kw))
    prof = models.profile_path("m", "joyful")
    prof.parent.mkdir(parents=True)
    prof.write_text("{}")
    servers = os.open(tmp_path / "servers-own", os.O_RDWR | os.O_CREAT)
    monkeypatch.setenv(state.INHERITED, str(servers))
    out = spawn.calibrate("joyful", "m", url=NOWHERE)
    try:
        assert os.fstat(servers) and _fds_on(state.job_dir(out["job"]) / state.LIVE) == []
        assert state.read_state(out["job"])["outcome"] is None
        _until(lambda: "steps" in state.read_state(out["job"]))
        assert state.read_state(out["job"])["outcome"] is None and state.cancel(out["job"])
        _until(lambda: state.read_state(out["job"])["outcome"] == "failed")
    finally:
        os.close(servers)
        try:
            os.kill(out["pid"], 9)
        except ProcessLookupError:
            pass


def _profiles(*names):
    for name in names:
        prof = models.profile_path("m", name)
        prof.parent.mkdir(parents=True, exist_ok=True)
        prof.write_text("{}")


def test_a_relaunch_of_an_old_id_never_signals_the_old_pid(monkeypatch):
    """A folder whose earlier run died without an outcome, its pid since taken by another process: once the new run
    holds the lock its folder reads as running, so its first state (no pid yet) must already be there, and cancel then
    signals nothing."""
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _profiles("joyful")
        monkeypatch.setattr(spawn.time, "strftime", lambda fmt, *a: "20261120-000000")
        job = "joyful-calibrate-20261120-000000"
        state.job_dir(job).mkdir(parents=True)
        (state.job_dir(job) / "state.json").write_text(json.dumps(dict(job=job, kind="calibrate", name="joyful",
                                                                       pid=sleeper.pid, outcome=None)))
        during = []

        class Proc:
            pid = 4242

            def wait(self):
                return 0

        def popen(cmd, **kw):
            during.append((state.read_state(job)["pid"], state.cancel(job)))
            return Proc()
        monkeypatch.setattr(spawn.subprocess, "Popen", popen)
        spawn.calibrate("joyful", "m", url=NOWHERE)
        assert during == [(None, False)] and sleeper.poll() is None and state.read_state(job)["pid"] == 4242
    finally:
        sleeper.kill()
        sleeper.wait()


def test_two_starters_at_once_start_one_job(tmp_path, monkeypatch):
    """Two launches overlapping (each one's check slowed): the check and the first state are one step, so one starts and
    the other is refused."""
    import os
    import threading
    import time
    driver = tmp_path / "driver.py"
    driver.write_text(JOB_DRIVER)
    real = spawn.subprocess.Popen
    monkeypatch.setattr(spawn.subprocess, "Popen", lambda cmd, **kw: real([cmd[0], str(driver), *cmd[3:]], **kw))
    check = spawn.busy
    monkeypatch.setattr(spawn, "busy", lambda: (check(), time.sleep(0.3))[0])
    _profiles("joyful", "fluffy")
    started, refused = [], []

    def start(name):
        try:
            started.append(spawn.calibrate(name, "m", url=NOWHERE))
        except ValueError as exc:
            refused.append(str(exc))
    threads = [threading.Thread(target=start, args=(n,)) for n in ("joyful", "fluffy")]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(started) == 1 and len(refused) == 1 and refused[0].startswith("a job is running: calibrating"), (started, refused)
    finally:
        for out in started:
            try:
                os.kill(out["pid"], 9)
            except ProcessLookupError:
                pass


def test_a_calibration_checks_again_once_its_state_is_written(monkeypatch):
    """A command-line calibration checked before it wrote its state; a job that started in between refuses it."""
    from metalscope.learn import calibrate as K, command as Cmd
    monkeypatch.setattr(K, "calibrate", lambda *a, **k: pytest.fail("calibrated beside another job"))
    other = state.take("other-1")
    try:
        (state.job_dir("other-1") / "state.json").write_text(json.dumps(dict(job="other-1", kind="learn", name="other",
                                                                             pid=1, outcome=None, started=1)))
        st = Cmd.run_calibrate_job("fluffy", "m", "fluffy-late")
    finally:
        other.release()
    assert (st["outcome"], st["error"]) == ("failed", "a job is running: learning other (/scope jobs)")


CHILD_FDS = """
import os, sys
def on(path):
    want = os.stat(path)
    for name in os.listdir("/dev/fd"):
        try:
            if os.path.samestat(os.fstat(int(name)), want):
                return True
        except OSError:
            pass
    return False
print(f"job lock {on(sys.argv[1])}, gpu lock {on(sys.argv[2])}")
"""


def test_the_gpu_child_inherits_the_gpu_lock_but_not_the_jobs(tmp_path, monkeypatch):
    """run_gpu's child, started as it starts it, from a job whose lock spawn passed: the GPU lock goes to the child,
    the job's own does not (a child outliving the job would keep it reading as running)."""
    import fcntl
    import os
    d = state.job_dir("x-child")
    d.mkdir(parents=True)
    fd = os.open(d / state.LIVE, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.set_inheritable(fd, True)                     # as spawn's pass_fds leaves it in the job's process
    monkeypatch.setenv(state.INHERITED, str(fd))
    j = learn_job("x", "d", None, "m", engine="ds4", url=NOWHERE, job="x-child")
    assert j.live.fd == fd
    gpu = gpulock.GpuLock(gpulock.Holder("learning x"), tmp_path / "gpu.lock").acquire()
    j.window = types.SimpleNamespace(lock=gpu, guard=lambda job, child: None,
                                     command=lambda job: ([sys.executable, "-c", CHILD_FDS, str(d / state.LIVE),
                                                           str(gpu.path)], dict(os.environ)))
    try:
        j.run_gpu()
    finally:
        gpu.release()
        j.live.release()
    assert j.state["log"][-1] == "job lock False, gpu lock True"


def test_a_launch_whose_process_cannot_start_holds_nothing(monkeypatch):
    """Popen raising: the launcher keeps no descriptor on the job's lock, nothing reads as running, and the job says why."""
    _profiles("joyful")

    def popen(cmd, **kw):
        raise OSError("no such interpreter")
    monkeypatch.setattr(spawn.subprocess, "Popen", popen)
    with pytest.raises(OSError, match="no such interpreter"):
        spawn.calibrate("joyful", "m", url=NOWHERE)
    (st,) = state.list_jobs()
    assert _fds_on(state.job_dir(st["job"]) / state.LIVE) == [] and state.busy() is None
    assert (st["outcome"], st["error"]) == ("failed", "no such interpreter")


def test_a_live_pid_without_the_jobs_lock_is_not_the_job():
    """A pid reused by another process, or a job folder from before the live lock: no lock held, so interrupted, and
    cancel signals nothing. A finished one reads as it was saved."""
    import os
    d = state.job_dir("x-old")
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps(dict(job="x-old", kind="learn", name="x", pid=os.getpid(), outcome=None)))
    assert state.read_state("x-old")["outcome"] == "interrupted" and state.busy() is None
    assert not state.cancel("x-old")
    (d / "state.json").write_text(json.dumps(dict(job="x-old", kind="learn", name="x", pid=os.getpid(), outcome="done")))
    assert state.read_state("x-old")["outcome"] == "done"


def test_each_served_copy_has_its_own_calibration_and_a_rewrite_voids_it():
    import os
    from metalscope.concepts import calibration_of
    models.link("Flash-oQ4e", "Flash")
    models.link("Flash-ds4", "Flash")
    prof = models.profile_path("Flash", "toy")
    prof.parent.mkdir(parents=True)
    prof.write_text("{}")
    for served, x in (("Flash-oQ4e", 0.5), ("Flash-ds4", 0.7)):
        cal = models.calibration_path(served, "toy")
        cal.parent.mkdir(parents=True)
        cal.write_text(json.dumps(dict(version=4, x=x, profile_mtime=prof.stat().st_mtime)))
    assert (calibration_of("Flash-oQ4e", "toy")["x"], calibration_of("Flash-ds4", "toy")["x"]) == (0.5, 0.7)
    assert calibration_of("Flash", "toy") is None                 # the checkpoint's name is not a served copy with one
    assert calibration_of("Other", "toy") is None
    # An older calibration counts as none: not read, not listed.
    stale = models.calibration_path("Flash-ds4", "toy")
    stale.write_text(json.dumps(dict(version=3, x=0.7, profile_mtime=prof.stat().st_mtime)))
    assert calibration_of("Flash-ds4", "toy") is None and models.calibrated_copies("Flash", "toy") == ["Flash-oQ4e"]
    os.utime(prof, (1, 1))                                        # the profile rewritten
    assert calibration_of("Flash-oQ4e", "toy") is None


# ---------------------------------------------------------------- learn on ds4 (Ds4Window)

PROXY = "http://127.0.0.1:8009"


def _ds4_setup(monkeypatch, proxy_name="m"):
    """A learn job's surroundings on ds4: the proxy answers /scope/api/ds4 as serving ``proxy_name`` (a list: one
    per ask, the last from then on); the corpus writer, the result and the calibration are faked; oMLX's admin must
    never be asked. Returns what went where."""
    from metalscope import client
    from metalscope.learn import calibrate as K, job as J
    asked, names = [], list(proxy_name) if isinstance(proxy_name, list) else [proxy_name]

    def get(self, path, timeout=60):
        asked.append((self.url, path))
        assert path == "/scope/api/ds4", path
        return dict(name=names.pop(0) if len(names) > 1 else names[0], gguf="/m.gguf", upstream="http://127.0.0.1:8010")
    monkeypatch.setattr(client.Client, "get", get)

    def write(wr, name, desc, on_stage=None):
        asked.append((wr.server.url, "writer", wr.model))
        return dict(concept=dict(kind="style", contrast="plain", learned=dict(writer=wr.model, request=desc, date="2026-10-28")),
                    pos=["p"], neg=["n"], task=dict(items=[1, 2]),
                    writing=dict(writer=wr.model, date="2026-10-28", request=desc, prompts=["spec?", "pairs?", "items?"]),
                    log=dict(requests=3, seconds=1))
    monkeypatch.setattr(W, "write", write)

    monkeypatch.setattr(J.LearnJob, "measured", lambda self: dict(steerable=True, reason=None))

    def calibrate(name, model, server=None, log=None, on_progress=None):
        asked.append((server.url, "calibrate"))
        return dict(version=4, default_dose=0.4, good=15, turns=18, reason="15/18 good @0.4")
    monkeypatch.setattr(K, "calibrate", calibrate)
    monkeypatch.setattr(omlx_admin, "Admin", lambda *a, **k: pytest.fail("a ds4 learn job asked oMLX"))
    killed = []
    monkeypatch.setattr(J.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    return asked, killed


def _ds4_child(monkeypatch, lines, rc=0, interrupt=False):
    """subprocess.Popen for the GPU child: records its command and who holds the lock, prints ``lines``, exits with
    ``rc`` (or the job is cancelled while it runs)."""
    from metalscope.learn import job as J
    real, seen = J.subprocess.Popen, {}

    class Child:
        pid = 987654321

        def __new__(cls, cmd, **kw):           # only the GPU child is faked (subprocess.run uses Popen too)
            return super().__new__(cls) if cmd[1:3] == ["-m", J.LearnJob.CHILD] else real(cmd, **kw)

        def __init__(self, cmd, **kw):
            seen.update(cmd=cmd, holder=gpulock.held(), pass_fds=kw.get("pass_fds"), env=kw.get("env"))
            self.done = False

            def out():
                yield from lines
                if interrupt:
                    raise KeyboardInterrupt("stopped")
                self.done = True
            self.stdout = out()

        def wait(self, timeout=None):
            self.done = True
            return rc

        def poll(self):
            return rc if self.done else None
    monkeypatch.setattr(J.subprocess, "Popen", Child)
    return seen


def _ev(**kw) -> str:
    return "@@" + json.dumps(kw) + "\n"


CHILD_LINES = [_ev(stage="connecting"), _ev(stage="presence"), _ev(step="presence", summary="held-out d′ 4.00 at L8"),
               _ev(stage="measuring the items"), _ev(stage="selection sweep", n=0, total=10),
               _ev(stage="selection sweep", n=10, total=10), _ev(stage="fluency ladder", n=0, total=4),
               _ev(step="fluency ladder", summary="replies break from dose 1.6 · test dose 0.4"),
               _ev(stage="test sweep", n=0, total=4), "a log line\n", _ev(stage="done", profile="/p/profile.json")]


def test_a_learn_job_on_ds4_holds_the_lock_and_measures_the_served_model(monkeypatch):
    import sys
    from metalscope.learn import job as J
    asked, killed = _ds4_setup(monkeypatch)
    seen = _ds4_child(monkeypatch, CHILD_LINES)
    st = learn_job("toasty", "Toasty things.", None, "m", engine="ds4", url=PROXY, job="toasty-1").run()
    assert st["outcome"] == "done", st["error"]
    assert st["window"] == "ds4" and "need_gb" not in st and st["profile"] == "/p/profile.json"
    assert seen["holder"].what == "learning toasty" and seen["pass_fds"]                 # the child ran under the lock
    assert seen["cmd"] == [sys.executable, "-m", "metalscope.learn.gpu", "m", "--engine", "ds4", "--url", PROXY,
                           "toasty", "--seed", "0"]                                          # no --model, no memory limit
    assert seen["env"]["PYTHONUNBUFFERED"] == "1" and gpulock.held() is None and not killed
    assert [s["key"] for s in st["steps"]] == ["corpus", "hold", "presence", "items", "selection", "ladder",
                                               "test", "release", "calibrate"]
    assert [s["status"] for s in st["steps"]] == ["done"] * 9
    steps = {s["key"]: s["summary"] for s in st["steps"]}
    assert steps["ladder"] == "replies break from dose 1.6 · test dose 0.4"
    assert steps["hold"] == "ds4 keeps serving m · chat refused while it is measured" and steps["release"] == "chat works again"
    # The writer, the proxy check and the calibration all asked the ds4 proxy.
    assert asked == [(PROXY, "/scope/api/ds4"), (PROXY, "writer", "m"), (PROXY, "/scope/api/ds4"), (PROXY, "calibrate")]
    assert st["result"]["calibration"]["good"] == 15 and models.concept_dir("m", "toasty").is_dir()
    assert steps["calibrate"] == "15/18 good @0.4"
    # The writer's instructions are kept beside the texts, and the corpus says who wrote them.
    writing = json.loads((models.concept_dir("m", "toasty") / "writing.json").read_text())
    assert writing["prompts"] == ["spec?", "pairs?", "items?"] and writing["writer"] == "m"
    assert st["corpus"]["texts"] == 'texts: written by m from "Toasty things." (2026-10-28)' and not st["corpus"]["stored"]
    assert st["corpus"]["source"] == "write"
    assert st["result"]["steer"] is None                          # the faked profile path holds no profile
    # Its stages are the ds4 weights' (the child's own included), and they add up.
    assert {json.loads(l[2:])["stage"] for l in CHILD_LINES if '"stage"' in l} - {"done"} <= set(J.LearnJob.DS4_WEIGHTS)
    assert sum(J.LearnJob.DS4_WEIGHTS.values()) == pytest.approx(1.0)


def test_a_learn_job_on_ds4_that_fails_releases_the_lock_and_reloads_nothing(monkeypatch):
    # The proxy serves another model: refused before any corpus is written.
    asked, killed = _ds4_setup(monkeypatch, proxy_name="other")
    seen = _ds4_child(monkeypatch, CHILD_LINES)
    st = learn_job("toasty", "d", None, "m", engine="ds4", url=PROXY, job="toasty-other").run()
    assert st["outcome"] == "failed" and st["error"] == f"the ds4 proxy at {PROXY} serves other, not m"
    assert asked == [(PROXY, "/scope/api/ds4")] and seen == {} and gpulock.held() is None
    assert not models.concept_dir("m", "toasty").exists() and not (state.job_dir("toasty-other") / "concept").exists()
    # It switched after the corpus was written: refused under the lock, before any child; the lock released, the
    # corpus moved out.
    asked, killed = _ds4_setup(monkeypatch, proxy_name=["m", "other"])
    seen = _ds4_child(monkeypatch, CHILD_LINES)
    st = learn_job("toasty", "d", None, "m", engine="ds4", url=PROXY, job="toasty-switched").run()
    assert st["outcome"] == "failed" and st["error"] == f"the ds4 proxy at {PROXY} serves other, not m"
    assert seen == {} and gpulock.held() is None and not models.concept_dir("m", "toasty").exists()
    assert (state.job_dir("toasty-switched") / "concept" / "pos.txt").exists()
    # The child dies: its error, the lock released, nothing to reload.
    asked, killed = _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES[:2] + ["Traceback (most recent call last):\n",
                                               "RuntimeError: ds4-server /v1/completions: HTTP 500: busy\n"], rc=1)
    st = learn_job("toasty", "d", None, "m", engine="ds4", url=PROXY, job="toasty-died").run()
    assert st["outcome"] == "failed" and st["error"] == ("the GPU run exited with code 1: RuntimeError: ds4-server "
                                                         "/v1/completions: HTTP 500: busy")
    steps = {s["key"]: s["status"] for s in st["steps"]}
    assert steps["presence"] == "failed" and steps["release"] == "done" and steps["calibrate"] == "pending"
    assert gpulock.held() is None and not killed and not models.concept_dir("m", "toasty").exists()
    # A cancel: the child is stopped first.
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES[:3], interrupt=True)
    st = learn_job("toasty", "d", None, "m", engine="ds4", url=PROXY, job="toasty-cancel").run()
    assert st["outcome"] == "failed" and st["error"] == "stopped" and gpulock.held() is None
    # Another job holds the GPU: refused before anything is written.
    asked, _ = _ds4_setup(monkeypatch)
    seen = _ds4_child(monkeypatch, CHILD_LINES)
    lock = gpulock.GpuLock(gpulock.Holder("learning other")).acquire()
    try:
        st = learn_job("toasty", "d", None, "m", engine="ds4", url=PROXY, job="toasty-busy").run()
    finally:
        lock.release()
    assert st["outcome"] == "failed" and st["error"].startswith("the GPU is in use: learning other")
    assert asked == [] and seen == {}


def test_the_cancelled_ds4_child_was_sent_sigterm(monkeypatch):
    import signal
    _, killed = _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES[:3], interrupt=True)
    learn_job("toasty", "d", None, "m", engine="ds4", url=PROXY, job="toasty-sig").run()
    assert killed == [(987654321, signal.SIGTERM)]


def test_on_ds4_the_writer_is_the_served_model():
    plan = P.learn_plan("toasty", "m", "d", writer="other", engine="ds4")
    assert plan["taken"].startswith("on ds4 the corpus writer is the served model (m), not other")
    assert P.learn_plan("toasty", "m", "d", writer="other", engine="mlx")["taken"] is None     # oMLX serves many


def _corpus(d, pairs=10, learned=None, pos_edit=None):
    """A supplied corpus in DIR ``d``: ``pairs`` pairs, 10 items, ``learned`` in concept.json; ``pos_edit(lines)``."""
    d.mkdir(parents=True, exist_ok=True)
    pos = [f"He bit the crisp apple {i} and heard the sharp snap ring out in the quiet kitchen today." for i in range(pairs)]
    neg = [f"He bit the ripe apple {i} and felt the soft flesh give way in the quiet kitchen today." for i in range(pairs)]
    if pos_edit:
        pos_edit(pos)
    concept = dict(title="Crisp", kind="content", contrast="soft textures", description="Crisp textures. Contrast: soft.",
                   words=["crisp"], **({"learned": learned} if learned else {}))
    (d / "concept.json").write_text(json.dumps(concept))
    (d / "pos.txt").write_text("\n".join(pos) + "\n")
    (d / "neg.txt").write_text("\n".join(neg) + "\n")
    items = [dict(prompt=f"Which snack {i}?", options=["chips", "pudding"], target="chips") for i in range(10)]
    (d / "task.json").write_text(json.dumps(dict(items=items)))
    return d


def test_a_supplied_corpus_skips_the_writer_and_records_where_it_came_from(monkeypatch, tmp_path):
    from metalscope import client
    asked, _ = _ds4_setup(monkeypatch)
    monkeypatch.setattr(client.Client, "post", lambda *a, **k: pytest.fail("a supplied corpus sent a chat request"))
    _ds4_child(monkeypatch, CHILD_LINES)
    src = _corpus(tmp_path / "blunt", learned=dict(writer="claude-opus-5-5", request="crisp textures"))
    writing = dict(writer="claude-opus-5-5", date="2026-10-27", request="crisp textures", prompts=["Write 10 pairs."])
    (src / "writing.json").write_text(json.dumps(writing))
    # The writer is not the served model, which on ds4 only a supplied corpus may be.
    st = learn_job("toasty", None, None, "m", engine="ds4", url=PROXY, job="toasty-supplied", corpus=str(src)).run()
    assert st["outcome"] == "done", st["error"]
    assert not [a for a in asked if "writer" in a] and st["description"] == "crisp textures"
    assert st["supplied"] == str(src.resolve()) and st["writer"] == "claude-opus-5-5"
    d = models.concept_dir("m", "toasty")
    meta = json.loads((d / "concept.json").read_text())
    assert meta["learned"]["writer"] == "claude-opus-5-5" and meta["learned"]["corpus"] == str(src.resolve())
    assert (d / "pos.txt").read_text() == (src / "pos.txt").read_text() and st["corpus"]["pairs"] == 10
    assert {s["key"]: s["summary"] for s in st["steps"]}["corpus"].endswith("· supplied")
    assert json.loads((d / "writing.json").read_text()) == writing          # the supplied DIR's writing.json, copied
    # Where the texts came from and who wrote them: the supplied texts' own record, never the served model.
    date = meta["learned"]["date"]
    assert (st["corpus"]["source"], st["corpus"]["stored"]) == ("corpus", False)
    assert st["corpus"]["texts"] == f'texts: written by claude-opus-5-5 from "crisp textures" ({date})'
    assert {s["key"]: s["detail"] for s in st["steps"]}["corpus"] == f"supplied from {src.resolve()}, written by claude-opus-5-5 ({date})"
    # No writer in the file: no record, none made up; every pair used, as for stored texts; no writing.json.
    plain = _corpus(tmp_path / "plain", pairs=20)
    out = W.read_corpus(plain, "toasty")
    assert "writer" not in out["concept"]["learned"] and len(out["pos"]) == 20 == len(out["neg"])
    assert out["writing"] is None
    st = learn_job("plain", None, None, "m", engine="ds4", url=PROXY, job="plain-supplied", corpus=str(plain)).run()
    assert st["outcome"] == "done", st["error"]
    assert (st["writer"], st["corpus"]["source"], st["corpus"]["texts"]) == ("unknown", "corpus", "texts: no record of who wrote them")
    assert {s["key"]: s["detail"] for s in st["steps"]}["corpus"] == f"supplied from {plain.resolve()}, no record of who wrote them"


def test_supplied_texts_follow_the_one_rule_for_existing_texts(monkeypatch, tmp_path):
    # Like stored texts: the structure must hold (else nothing runs), the writer's pair rules are only reported.
    from metalscope import cli
    _ds4_setup(monkeypatch)
    seen = _ds4_child(monkeypatch, CHILD_LINES)
    short = lambda pos: pos.__setitem__(2, "He bit the crisp apple.")
    out = W.read_corpus(_corpus(tmp_path / "short", pos_edit=short), "toasty")
    assert out["log"]["warnings"] == ["pair 3: lengths more than 15% apart in words (5 / 17 words)"]
    out = W.read_corpus(_corpus(tmp_path / "name", pos_edit=lambda p: p.__setitem__(0, p[0].replace("crisp", "toasty"))),
                        "toasty")
    assert out["log"]["warnings"][0].startswith("pair 1: uses the concept's own name")
    with pytest.raises(SystemExit, match="only 7 pairs"):
        cli.main(["concept", "learn", "toasty", "m", "--corpus", str(_corpus(tmp_path / "few", pairs=7))])
    assert seen == {} and not models.concept_dir("m", "toasty").exists()             # nothing ran, nothing written
    with pytest.raises(ValueError, match="pos.txt has 9 sentences and neg.txt 10"):   # a # line is a comment
        W.read_corpus(_corpus(tmp_path / "hash", pos_edit=lambda p: p.__setitem__(3, "# " + p[3])), "toasty")
    with pytest.raises(ValueError, match="pos.txt has 11 sentences and neg.txt 10"):
        W.read_corpus(_corpus(tmp_path / "odd", pos_edit=lambda p: p.append(p[0])), "toasty")
    d = _corpus(tmp_path / "items")
    (d / "task.json").write_text(json.dumps(dict(items=[dict(prompt="Which?", options=["a", "b"], target="a")])))
    with pytest.raises(ValueError, match="only 1 task items are usable"):
        W.read_corpus(d, "toasty")
    d = _corpus(tmp_path / "nodesc")
    (d / "concept.json").write_text(json.dumps(dict(kind="style")))
    with pytest.raises(ValueError, match="a description is needed"):
        W.read_corpus(d, "toasty")


def test_corpus_makes_the_description_optional_and_refuses_a_writer(monkeypatch, tmp_path):
    from metalscope import cli
    from metalscope.learn import command as Cmd, plan as P
    made = []
    monkeypatch.setattr(Cmd, "_main", lambda args, make, report, copy=True, words=None: made.append(make))
    monkeypatch.setattr(P, "pick_engine", lambda name, url=None: "ds4")
    src = _corpus(tmp_path / "c")
    cli.main(["concept", "learn", "toasty", "m", "--corpus", str(src)])
    job = made[-1](None)
    assert job.description == "Crisp textures. Contrast: soft." and job.writer_model == "unknown"
    with pytest.raises(SystemExit, match=r"toasty needs a description \(--description\)"):
        cli.main(["concept", "learn", "toasty", "m"])
    with pytest.raises(SystemExit, match="--writer and --corpus"):
        cli.main(["concept", "learn", "toasty", "m", "--corpus", str(src), "--writer", "other"])
    assert len(made) == 1


def test_the_engine_is_picked_by_what_the_ds4_proxy_serves(monkeypatch):
    from metalscope import client, cli
    from metalscope.learn import command as Cmd, plan as P
    _ds4_setup(monkeypatch)
    assert P.pick_engine("m") == "ds4" and P.pick_engine("other") == "mlx"

    def down(self, path, timeout=60):
        raise client.ServerError("no metalscope server")
    monkeypatch.setattr(client.Client, "get", down)
    assert P.pick_engine("m") == "mlx"
    # The command line: auto by default, ds4 needs no checkpoint path, --strengths reach the child.
    made = []
    monkeypatch.setattr(Cmd, "_main", lambda args, make, report, copy=True, words=None: made.append((copy, make)))
    monkeypatch.setattr(P, "pick_engine", lambda name, url=None: "ds4")
    cli.main(["concept", "learn", "toasty", "m", "--description", "d", "--strengths", "0.05,0.1"])
    copy, make = made[-1]
    job = make(None)
    assert copy is False and job.window.name == "ds4" and job.window.url == "http://127.0.0.1:8009"
    assert job.extra_args() == ["toasty", "--seed", "0", "--strengths", "0.05,0.1"]
    cli.main(["concept", "learn", "crunchy2", "m", "--description", "d", "--engine", "mlx"])
    assert made[-1][0] is True
    cli.main(["concept", "learn", "crunchy3", "m", "--description", "d", "--model", "/ckpt"])    # auto, a path: a copy
    assert made[-1][0] is True


def test_spawn_hands_the_engine_url_and_strengths_on(monkeypatch):
    started = []

    class Proc:
        pid = 4242

        def wait(self):
            return 0
    monkeypatch.setattr(spawn.subprocess, "Popen", lambda cmd, **k: started.append(cmd) or Proc())
    out = spawn.learn("toasty", "Toasty things.", "m", None, "ds4", PROXY, [0.05, 0.1])
    assert started[0][3:] == ["concept", "learn", "toasty", "m", "--description=Toasty things.", "--engine", "ds4",
                              "--url", PROXY, "--strengths", "0.05,0.1", "--job", out["job"], "--origin", "pi"]
    assert json.loads((state.job_dir(out["job"]) / "state.json").read_text())["window"] == "ds4"


def test_a_job_pi_starts_takes_free_text_that_begins_with_a_dash(monkeypatch):
    """A description (or writer) starting with "-" reaches the job as it was given, not as an option its command line
    refuses before the job exists."""
    from metalscope import cli
    started, made = [], []

    class Proc:
        pid = 4242

        def wait(self):
            return 0
    monkeypatch.setattr(spawn.subprocess, "Popen", lambda cmd, **k: started.append(cmd) or Proc())
    monkeypatch.setattr(spawn, "busy", lambda: None)                                 # one start after another
    monkeypatch.setattr(Cmd, "_main", lambda args, make, report, copy=True, words=None: made.append(make("/nonexistent")))
    spawn.learn("toasty", "-crisp", "m", None, "ds4", PROXY)            # one word: a space would let it through
    spawn.learn("toasted", "Toasted things.", "m", "-w", "mlx")
    for cmd in started:
        assert cmd[1:3] == ["-m", "metalscope.cli"]
        cli.main(cmd[3:])                                # the job's own command line, as it parses it
    assert [(j.name, j.description, j.writer_model) for j in made] == [
        ("toasty", "-crisp", "m"), ("toasted", "Toasted things.", "-w")]


def test_a_dsml_tool_call_counts_as_a_tool_call():
    from metalscope.learn import calibrate as K
    assert K.is_tool("hm</think>\n\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"bash\">")
    assert not K.is_tool("<｜DSML｜function_calls> while thinking") and not K.is_tool("hm</think>plain answer")


def test_a_cancel_waits_for_the_final_files():
    import os
    import signal
    from metalscope.learn import calibrate as K
    seen = []
    old = signal.signal(signal.SIGTERM, lambda n, f: seen.append("cancel"))
    try:
        with K._held():
            os.kill(os.getpid(), signal.SIGTERM)
            seen.append("swapped")
    finally:
        signal.signal(signal.SIGTERM, old)
    assert seen == ["swapped", "cancel"]


# ---------------------------------------------------------------- the texts, kept in each model's own folder

def _stored(name="humorous", pairs=10, writing=True, model="m"):
    """A concept's texts in ``model``'s own folder, written by another model, with two pairs that break the writer's
    rules (its own name, lengths far apart): a model's own texts are used as they are."""
    def edit(pos):
        pos[0] = pos[0].replace("crisp", "humorous")
        pos[2] = "He bit the crisp apple."
    d = _corpus(models.concept_dir(model, name), pairs=pairs, pos_edit=edit,
                learned=dict(writer="DeepSeek-V4.1", request="dry humour", date="2026-10-20"))
    if writing:
        (d / "writing.json").write_text(json.dumps(dict(writer="DeepSeek-V4.1", date="2026-10-20", request="dry humour",
                                                        prompts=["Write the spec.", "Write the pairs."])))
    return d


def _snapshot(d):
    """Every file under ``d``: its bytes and mtime."""
    return {str(p.relative_to(d)): (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(d.rglob("*")) if p.is_file()}


TEXT_FILES = ["concept.json", "neg.txt", "pos.txt", "task.json", "writing.json"]


def test_a_models_own_texts_win_and_take_no_corpus_or_writer(tmp_path):
    d = _stored()
    p = P.learn_plan("humorous", "m")
    assert (p["source"], p["texts"], p["description"], p["taken"]) == (
        "own", "texts: written by DeepSeek-V4.1 (2026-10-20)", "dry humour", None)
    assert "stored" not in p and p["read"]["path"] == str(d)
    other = str(_corpus(tmp_path / "c"))
    assert P.learn_plan("humorous", "m", corpus=other)["taken"] == "humorous has its own texts on m: learn it without --corpus"
    assert P.learn_plan("humorous", "m", writer="x")["taken"] == "humorous's texts exist, so nothing is written: no --writer to pick"
    assert P.learn_plan("humorous", "m", "else")["notes"] == ["the description given is ignored: humorous's texts have their own"]
    # Another model has none of them: it takes --corpus, or writes its own.
    assert P.learn_plan("humorous", "n", corpus=other)["source"] == "corpus"


def test_a_shipped_concept_is_copied_into_each_models_own_folder(monkeypatch):
    from metalscope import config
    from metalscope.concepts import corpus as C
    shipped = _snapshot(C.DATA_DIR / "fluffy")
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES)
    p = P.learn_plan("fluffy", "m")
    assert (p["source"], p["texts"]) == ("shipped", "texts: hand-written by Claude Fable 5.1 (2026-10-05)")
    st = learn_job("fluffy", "", None, "m", engine="ds4", url=PROXY, job="fluffy-1").run()
    assert st["outcome"] == "done", st["error"]
    a = models.concept_dir("m", "fluffy")
    assert {f: b for f, (b, _) in _snapshot(a).items()} == {f: b for f, (b, _) in shipped.items()}
    assert st["corpus"]["path"] == str(a) and st["corpus"]["stored"] and st["corpus"]["source"] == "shipped"
    assert any(line.endswith(f"shipped texts copied into {a}") for line in st["log"])
    assert next(s for s in st["steps"] if s["key"] == "corpus")["detail"] == (
        "shipped texts, written by Claude Fable 5.1 (by hand) (2026-10-05)")
    # Another model gets its own copy; the shipped folder and the definitions are untouched.
    _ds4_setup(monkeypatch, proxy_name="n")
    st = learn_job("fluffy", "", None, "n", engine="ds4", url=PROXY, job="fluffy-2").run()
    assert st["outcome"] == "done", st["error"]
    assert C.texts_dir("n", "fluffy") == models.concept_dir("n", "fluffy") != a
    assert _snapshot(C.DATA_DIR / "fluffy") == shipped and not (config.home() / "concepts").exists()
    # A re-run on the first model reads its own copy now.
    assert P.learn_plan("fluffy", "m")["source"] == "own"


def test_supplied_texts_go_into_the_models_folder_and_define_a_new_name_once(monkeypatch, tmp_path):
    from metalscope.concepts import corpus as C
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES)
    src = _corpus(tmp_path / "src", learned=dict(writer="claude-opus-5-5", request="crisp textures"))
    p = P.learn_plan("toasty", "m", None, str(src))
    assert (p["source"], p["texts"]) == ("corpus", None)
    st = learn_job("toasty", "", None, "m", engine="ds4", url=PROXY, job="toasty-1", corpus=str(src)).run()
    assert st["outcome"] == "done", st["error"]
    assert (models.concept_dir("m", "toasty") / "pos.txt").read_text() == (src / "pos.txt").read_text()
    assert json.loads(models.definition_path("toasty").read_text()) == dict(name="toasty", description="crisp textures")
    assert C.definition("toasty") == "crisp textures"
    # A defined name or a shipped one takes --corpus too; neither writes a definition.
    _ds4_setup(monkeypatch, proxy_name="n")
    before = models.definition_path("toasty").read_bytes()
    st = learn_job("toasty", "something new", None, "n", engine="ds4", url=PROXY, job="toasty-2", corpus=str(src)).run()
    assert st["outcome"] == "done", st["error"]
    assert models.definition_path("toasty").read_bytes() == before
    assert P.learn_plan("fluffy", "n", None, str(src))["source"] == "corpus"
    st = learn_job("fluffy", "", None, "n", engine="ds4", url=PROXY, job="fluffy-1", corpus=str(src)).run()
    assert st["outcome"] == "done", st["error"]
    assert not models.definition_path("fluffy").exists() and C.define("fluffy", "x") is None
    assert (models.concept_dir("n", "fluffy") / "pos.txt").read_text() == (src / "pos.txt").read_text()


def test_another_model_writes_its_own_texts_from_the_stored_definition(monkeypatch):
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES)
    st = learn_job("toasty", "Toasty things.", None, "m", engine="ds4", url=PROXY, job="toasty-1").run()
    assert st["outcome"] == "done", st["error"]
    a = models.concept_dir("m", "toasty")
    assert sorted(_snapshot(a)) == TEXT_FILES
    defined = models.definition_path("toasty").read_bytes()
    assert json.loads(defined) == dict(name="toasty", description="Toasty things.")
    before = _snapshot(a)
    # Model n: no description given, so the stored one; m's folder untouched, n writes its own.
    p = P.learn_plan("toasty", "n")
    assert (p["source"], p["description"], p["taken"]) == ("write", "Toasty things.", None)
    asked, _ = _ds4_setup(monkeypatch, proxy_name="n")
    st = learn_job("toasty", "", None, "n", engine="ds4", url=PROXY, job="toasty-2").run()
    assert st["outcome"] == "done", st["error"]
    assert (PROXY, "writer", "n") in asked and st["description"] == "Toasty things."
    b = models.concept_dir("n", "toasty")
    assert json.loads((b / "concept.json").read_text())["learned"]["request"] == "Toasty things."
    assert _snapshot(a) == before and models.definition_path("toasty").read_bytes() == defined
    # A description typed now is this run's; the definition stays as it was.
    _ds4_setup(monkeypatch, proxy_name="o")
    st = learn_job("toasty", "Warm bread.", None, "o", engine="ds4", url=PROXY, job="toasty-3").run()
    assert st["outcome"] == "done", st["error"]
    assert json.loads((models.concept_dir("o", "toasty") / "concept.json").read_text())["learned"]["request"] == "Warm bread."
    assert models.definition_path("toasty").read_bytes() == defined


def test_a_new_name_with_no_description_anywhere_is_refused():
    need = "toasty needs a description (--description): one line saying what it means"
    assert P.learn_plan("toasty", "m", "")["taken"] == need
    p = P.learn_plan("toasty", "m")                        # Pi's confirm: not asked yet
    assert (p["source"], p["description"], p["taken"]) == ("write", "", None)


def test_every_source_leaves_the_texts_in_the_models_folder_and_the_gpu_child_reads_only_them(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from metalscope import concepts as C
    from metalscope.learn import gpu
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES)
    _stored("own")
    src = _corpus(tmp_path / "src", learned=dict(writer="claude-opus-5-5", request="crisp textures"))
    for name, desc, corpus in (("own", "", None), ("fluffy", "", None), ("supplied", "", str(src)), ("toasty", "Toasty.", None)):
        st = learn_job(name, desc, None, "m", engine="ds4", url=PROXY, job=f"{name}-1", corpus=corpus).run()
        assert st["outcome"] == "done", st["error"]
        d = models.concept_dir("m", name)
        assert {"concept.json", "pos.txt", "neg.txt", "task.json"} <= {p.name for p in d.iterdir()}, name
        assert C.texts_dir("m", name) == d, name
        if name != "toasty":                               # the faked writer's texts are too few to load
            c = C.load_concept("m", name)
            assert c.path == d and len(c.pos) == len((d / "pos.txt").read_text().splitlines()), name
    # The GPU child reads the texts of its engine's checkpoint, nothing else.
    read = []
    monkeypatch.setattr(C, "load_concept", lambda model, name: read.append((model, name)) or pytest.fail("stop"))
    with pytest.raises(pytest.fail.Exception):
        gpu.run(SimpleNamespace(checkpoint="m", tok=None, served="m"), "own")
    assert read == [("m", "own")]
    with pytest.raises(ValueError, match=r"no texts of own on n \("):
        C.corpus.load_concept("n", "own")


def test_texts_are_written_whole_or_not_at_all(monkeypatch):
    """A run killed while it writes the texts (SIGKILL: no cleanup) leaves no texts or all of them: each file is
    written under a dot-temp name and renamed in, concept.json last. The next run clears what a killed one left."""
    import os
    from metalscope.concepts import corpus as C
    files = {n: (C.DATA_DIR / "fluffy" / n).read_text() for n in TEXT_FILES if (C.DATA_DIR / "fluffy" / n).is_file()}
    d = models.concept_dir("m", "fluffy")
    real_rename, real_write = os.rename, Path.write_text
    steps = 2 * len(files)                                    # a temp written, then renamed, for each file
    for k in range(steps):
        n = [0]

        def count():
            n[0] += 1
            if n[0] > k:
                raise KeyboardInterrupt("killed")
        monkeypatch.setattr(J.os, "rename", lambda a, b: count() or real_rename(a, b))
        monkeypatch.setattr(Path, "write_text", lambda self, t, *a, **kw: count() or real_write(self, t, *a, **kw))
        job = learn_job("fluffy", "", None, "m", engine="ds4", url=PROXY, job=f"fluffy-k{k}")
        with pytest.raises(KeyboardInterrupt):
            job._put_texts(files)
        monkeypatch.setattr(J.os, "rename", real_rename)
        monkeypatch.setattr(Path, "write_text", real_write)
        assert C.texts_dir("m", "fluffy") is None and not (d / "concept.json").exists(), k     # no texts
        p = P.learn_plan("fluffy", "m")
        assert (p["source"], p["taken"]) == ("shipped", None), k
        # The next run writes them all, nothing left of the killed one.
        learn_job("fluffy", "", None, "m", engine="ds4", url=PROXY, job=f"fluffy-n{k}")._put_texts(files)
        assert sorted(f.name for f in d.iterdir()) == sorted(files), k
        assert all((d / f).read_text() == t for f, t in files.items())
        shutil.rmtree(d)
    # Texts there already (another run's, or the user's): never written over.
    learn_job("fluffy", "", None, "m", engine="ds4", url=PROXY, job="fluffy-a")._put_texts(files)
    with pytest.raises(FileExistsError):
        learn_job("fluffy", "", None, "m", engine="ds4", url=PROXY, job="fluffy-b")._put_texts(files)


def test_a_folder_with_part_of_the_texts_is_refused_until_deleted(monkeypatch, tmp_path):
    from metalscope import cli
    src = str(_corpus(tmp_path / "src"))
    d = models.concept_dir("m", "fluffy")
    d.mkdir(parents=True)
    (d / "concept.json").write_text("{}")                          # texts from an older run, cut short
    said = f"fluffy's texts on m are incomplete ({d}): metalscope concept delete fluffy m, then learn again"
    assert P.learn_plan("fluffy", "m")["taken"] == said                           # shipped
    assert P.learn_plan("fluffy", "m", corpus=src)["taken"] == said               # corpus
    (d / "pos.txt").write_text("a\n")
    assert P.learn_plan("fluffy", "m")["taken"] == said                           # reads as its own, but is not whole
    t = models.concept_dir("m", "toasty")
    t.mkdir(parents=True)
    (t / "concept.json").write_text("{}")
    said_t = f"toasty's texts on m are incomplete ({t}): metalscope concept delete toasty m, then learn again"
    assert P.learn_plan("toasty", "m", "Toasty.")["taken"] == said_t              # written
    # Deleted, it learns again.
    cli.main(["concept", "delete", "fluffy", "m", "--yes"])
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES)
    st = learn_job("fluffy", "", None, "m", engine="ds4", url=PROXY, job="fluffy-1").run()
    assert st["outcome"] == "done", st["error"]
    assert C_texts_whole(d)


def C_texts_whole(d):
    return all((d / f).is_file() for f in ("concept.json", "pos.txt", "neg.txt", "task.json"))


def test_a_taken_name_is_said_before_what_else_is_wrong_with_the_request(tmp_path):
    _stored("done")
    models.profile_path("m", "done").write_text("{}")
    taken = "a profile for 'done' already exists for m"
    assert P.learn_plan("done", "m", corpus=str(_corpus(tmp_path / "c")))["taken"] == taken
    assert P.learn_plan("done", "m", writer="x")["taken"] == taken
    assert P.learn_plan("done", "m")["taken"] == taken


def test_another_models_texts_of_an_undefined_name_are_never_read():
    _stored("pulled", model="n")                     # on n only (a pull, or texts written before definitions)
    assert P.learn_plan("pulled", "m", "")["taken"] == "pulled needs a description (--description): one line saying what it means"
    p = P.learn_plan("pulled", "m")                  # Pi's confirm
    assert (p["source"], p["description"], p["texts"], p["read"]) == ("write", "", None, None)


def test_the_library_shows_the_models_own_copy_of_a_shipped_concept():
    from metalscope.concepts import corpus as C
    d = models.concept_dir("m", "fluffy")
    shutil.copytree(C.DATA_DIR / "fluffy", d)
    meta = json.loads((d / "concept.json").read_text())
    (d / "concept.json").write_text(json.dumps(dict(meta, learned=dict(meta["learned"], writer="Someone else (by hand)"))))
    row = next(c for c in C.list_concepts("m") if c["name"] == "fluffy")
    assert row["path"] == str(d) and row["texts"].startswith("texts: hand-written by Someone else")
    row = next(c for c in C.list_concepts("n") if c["name"] == "fluffy")
    assert row["path"] == str(C.DATA_DIR / "fluffy")


def test_the_library_says_a_concept_learned_without_texts_here_has_none_whatever_else_has_some():
    from metalscope.concepts import corpus as C
    models.definition_path("half").parent.mkdir(parents=True)
    models.definition_path("half").write_text(json.dumps(dict(name="half", description="Half a thing.")))
    for name in ("half", "fluffy", "plain"):                      # a definition, shipped texts, nothing else
        models.concept_dir("m", name).mkdir(parents=True)
        models.profile_path("m", name).write_text("{}")           # a profile on m, no texts there (a pull without them)
    rows = {c["name"]: c for c in C.list_concepts("m")}
    assert {n: rows[n]["texts"] for n in ("half", "fluffy", "plain")} == dict.fromkeys(("half", "fluffy", "plain"),
                                                                                       "texts: none on this model")
    assert rows["half"]["description"] == "Half a thing." and rows["fluffy"]["title"] == "Fluffy"
    assert {c["name"]: c["texts"] for c in C.list_concepts("n")}["half"] == "texts: each model writes its own"


def test_a_failed_run_takes_back_what_it_made_and_only_that(monkeypatch, tmp_path):
    from metalscope import config
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES[:2] + ["RuntimeError: boom\n"], rc=1)
    # Written texts: moved into the job folder; the folder and the definition this run made go.
    st = learn_job("toasty", "Toasty things.", None, "m", engine="ds4", url=PROXY, job="toasty-died").run()
    assert st["outcome"] == "failed"
    assert not models.concept_dir("m", "toasty").exists() and not models.definition_path("toasty").parent.exists()
    assert not (config.home() / "models" / "m").exists()               # no empty folder left to read as data
    assert sorted(p.name for p in (state.job_dir("toasty-died") / "concept").iterdir()) == TEXT_FILES
    # A definition there before stays, and so do another model's texts.
    models.definition_path("toasty").parent.mkdir(parents=True)
    models.definition_path("toasty").write_text(json.dumps(dict(name="toasty", description="Toasty things.")))
    theirs = _stored("toasty", model="n")
    before, defined = _snapshot(theirs), models.definition_path("toasty").read_bytes()
    st = learn_job("toasty", "", None, "m", engine="ds4", url=PROXY, job="toasty-died-2").run()
    assert st["outcome"] == "failed" and not models.concept_dir("m", "toasty").exists()
    assert models.definition_path("toasty").read_bytes() == defined and _snapshot(theirs) == before
    # Shipped texts copied: removed with the folder; the shipped ones as they were.
    st = learn_job("fluffy", "", None, "m", engine="ds4", url=PROXY, job="fluffy-died").run()
    assert st["outcome"] == "failed" and not models.concept_dir("m", "fluffy").exists()
    assert not (state.job_dir("fluffy-died") / "concept").exists()
    # A folder there before (a calibration): the texts this run put there go, the folder and calibration stay.
    cal = models.calibration_path("m", "fluffy")
    cal.parent.mkdir(parents=True)
    cal.write_text("{}")
    st = learn_job("fluffy", "", None, "m", engine="ds4", url=PROXY, job="fluffy-died-2").run()
    assert st["outcome"] == "failed" and cal.exists() and sorted(p.name for p in cal.parent.parent.iterdir()) == ["m"]
    # Own texts: never touched.
    own = _stored("humorous")
    before = _snapshot(own)
    st = learn_job("humorous", "", None, "m", engine="ds4", url=PROXY, job="humorous-died").run()
    assert st["outcome"] == "failed" and _snapshot(own) == before


def test_a_models_own_texts_are_learned_again_with_nothing_written(monkeypatch):
    from metalscope import client
    asked, _ = _ds4_setup(monkeypatch)
    monkeypatch.setattr(client.Client, "post", lambda *a, **k: pytest.fail("own texts sent a chat request"))
    seen = _ds4_child(monkeypatch, CHILD_LINES)
    d = _stored()
    before = _snapshot(d)
    # On ds4 the writer need not be the served model: nothing is written. The description given is ignored.
    st = learn_job("humorous", "something else", None, "m", engine="ds4", url=PROXY, job="humorous-1").run()
    assert st["outcome"] == "done", st["error"]
    assert not [a for a in asked if "writer" in a] and seen["cmd"][-3:] == ["humorous", "--seed", "0"]
    assert (st["description"], st["writer"]) == ("dry humour", "DeepSeek-V4.1") and "stored" not in st
    assert _snapshot(d) == before and not models.definition_path("humorous").exists()   # as they were, nothing defined
    corpus = next(s for s in st["steps"] if s["key"] == "corpus")
    assert corpus["summary"] == ("10 pairs, 10 items · content · contrast: soft textures · own · "
                                 "2 pairs break the writer's rules (kept)")
    assert corpus["detail"] == "its own texts, written by DeepSeek-V4.1 (2026-10-20)"
    assert st["corpus"]["stored"] and st["corpus"]["path"] == str(d) and st["corpus"]["source"] == "own"
    assert st["corpus"]["texts"] == 'texts: written by DeepSeek-V4.1 from "dry humour" (2026-10-20)'
    assert any("the description given is ignored" in line for line in st["log"])


def test_existing_texts_are_checked_for_their_structure_only_and_never_moved(monkeypatch, tmp_path):
    from metalscope import cli
    from metalscope.concepts import corpus as C
    from metalscope.learn import job as J
    # The shipped hand-written sets break the pair rules (fluffy uses its own name) and are learnable as they are.
    shipped = W.read_texts(C.DATA_DIR / "fluffy", "fluffy")
    assert len(shipped["pos"]) == 16 and len(shipped["task"]["items"]) == 20 and shipped["writing"] is None
    assert P.name_taken("fluffy", "m") is None
    job = learn_job("fluffy", None, None, "m", engine="ds4", url=PROXY, job="fluffy-1")
    assert job.description.startswith("Soft, fluffy, furry textures") and job.writer_model == "Claude Fable 5.1 (by hand)"
    assert len(W.read_texts(C.DATA_DIR / "stoic", "stoic")["pos"]) == 48     # every pair the GPU run reads, not 16
    with pytest.raises(ValueError, match="only 7 pairs"):
        W.read_texts(_stored("few", pairs=7, model="other"), "few")
    # A child that dies leaves the model's own texts where they are, and removes the vector it made.
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES[:2] + ["RuntimeError: boom\n"], rc=1)
    d = _stored()
    vec = models.vector_path("m", "humorous")

    def made_a_vector(self, *a, **k):
        vec.write_text("v")
        return real_run_gpu(self, *a, **k)
    real_run_gpu = J.GpuJob.run_gpu
    monkeypatch.setattr(J.GpuJob, "run_gpu", made_a_vector)
    st = learn_job("humorous", None, None, "m", engine="ds4", url=PROXY, job="humorous-died").run()
    assert st["outcome"] == "failed" and (d / "pos.txt").exists() and not (state.job_dir("humorous-died") / "concept").exists()
    assert not vec.exists()
    # A vector or profile of it on this checkpoint already: refused, as before.
    vec.write_text("v")
    assert "a steering vector named 'humorous' already exists for m" in P.learn_plan("humorous", "m")["taken"]
    # The command line: no description needed, no --writer, texts that do not hold together refused before a job.
    _stored(model="other")
    made = []
    monkeypatch.setattr(Cmd, "_main", lambda args, make, report, copy=True, words=None: made.append(make))
    monkeypatch.setattr(P, "pick_engine", lambda name, url=None: "ds4")
    cli.main(["concept", "learn", "humorous", "other"])
    assert made[-1](None).source == "own"
    with pytest.raises(SystemExit, match="texts exist, so nothing is written"):
        cli.main(["concept", "learn", "humorous", "other", "--writer", "x"])
    with pytest.raises(SystemExit, match="only 7 pairs"):
        cli.main(["concept", "learn", "few", "other"])
    assert len(made) == 1


def test_a_vector_a_killed_learn_left_without_a_profile_does_not_block_learning_again(monkeypatch):
    import os
    from metalscope.learn import job as J
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES[:2] + ["RuntimeError: boom\n"], rc=1)
    _stored()
    vec = models.vector_path("m", "humorous")
    vec.write_text("v")
    meta = vec.with_suffix(".json")
    meta.write_text(json.dumps(dict(source="concept humorous: diff-of-means, 10 vs 10 texts, mean position")))
    os.utime(vec, (2000, 2000))
    # Measured by learn's presence but no learn job behind it (made before learn jobs): the user's, taken.
    assert not P.unfinished_vector("humorous", "m") and "already exists" in P.name_taken("humorous", "m")
    # A learn job for it that started before the vector and was killed: the vector is its leftover, overwritable.
    d = state.job_dir("humorous-killed")
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps(dict(kind="learn", job="humorous-killed", name="humorous", model="m",
                                                  started=1000, pid=None, outcome=None)))
    assert P.unfinished_vector("humorous", "m") and P.name_taken("humorous", "m") is None
    # A vector made otherwise (from the J-lens), or one with its profile beside it, is taken all the same.
    meta.write_text(json.dumps(dict(source="J-lens concept direction for wit")))
    assert "a steering vector named 'humorous' already exists" in P.name_taken("humorous", "m")
    meta.write_text(json.dumps(dict(source="concept humorous: diff-of-means")))
    models.profile_path("m", "humorous").write_text("{}")
    assert "already exists" in P.name_taken("humorous", "m")
    models.profile_path("m", "humorous").unlink()
    # A run whose child dies without rewriting it leaves it as it was (it did not write it).
    st = learn_job("humorous", None, None, "m", engine="ds4", url=PROXY, job="humorous-died").run()
    assert st["outcome"] == "failed" and vec.read_text() == "v" and meta.exists()
    # The next run's child rewrites the vector and fails: the vector it wrote goes.
    def rewrites(self, *a, **k):
        vec.write_text("new")
        return real_run_gpu(self, *a, **k)
    real_run_gpu = J.GpuJob.run_gpu
    monkeypatch.setattr(J.GpuJob, "run_gpu", rewrites)
    st = learn_job("humorous", None, None, "m", engine="ds4", url=PROXY, job="humorous-again").run()
    assert st["outcome"] == "failed" and "already exists" not in st["error"] and not vec.exists()


def test_a_failed_learn_removes_only_what_its_gpu_run_wrote(monkeypatch):
    """The checkpoint's folder for the concept goes only when this run made it; a job that fails before its GPU child
    removes nothing, whatever another process put there meanwhile."""
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES[:2] + ["RuntimeError: boom\n"], rc=1)
    _stored()
    folder, vec = models.concept_dir("m", "humorous"), models.vector_path("m", "humorous")
    cal = models.calibration_path("m", "humorous")
    # The folder there already, holding only a calibration: the child's vector goes, the calibration and folder stay.
    cal.parent.mkdir(parents=True)
    cal.write_text("{}")
    real_run_gpu = J.GpuJob.run_gpu

    def writes_a_vector(self, *a, **k):
        vec.write_text("v")
        vec.with_suffix(".json").write_text("{}")
        return real_run_gpu(self, *a, **k)
    monkeypatch.setattr(J.GpuJob, "run_gpu", writes_a_vector)
    st = learn_job("humorous", None, None, "m", engine="ds4", url=PROXY, job="humorous-died").run()
    assert st["outcome"] == "failed" and cal.exists() and not vec.exists() and not vec.with_suffix(".json").exists()
    monkeypatch.setattr(J.GpuJob, "run_gpu", real_run_gpu)
    shutil.rmtree(folder)

    # Refused as busy, while another process makes the folder and writes into it: nothing of it goes.
    def busy_meanwhile(exclude=None):
        cal.parent.mkdir(parents=True)
        cal.write_text("{}")
        vec.write_text("theirs")
        return "a job is running: calibrating toy (/scope jobs)"
    monkeypatch.setattr(J, "busy", busy_meanwhile)
    st = learn_job("humorous", None, None, "m", engine="ds4", url=PROXY, job="humorous-busy").run()
    assert st["outcome"] == "failed" and "a job is running" in st["error"]
    assert cal.exists() and vec.read_text() == "theirs"
    monkeypatch.setattr(J, "busy", lambda exclude=None: None)
    vec.unlink()
    # The window refuses (the proxy switched model), while another process writes a vector there: it stays.
    def switched(self, job):
        vec.write_text("theirs")
        raise RuntimeError(f"the ds4 proxy at {PROXY} serves other, not m")
    monkeypatch.setattr(Win.Ds4Window, "check", switched)
    st = learn_job("humorous", None, None, "m", engine="ds4", url=PROXY, job="humorous-switched").run()
    assert st["outcome"] == "failed" and "serves other" in st["error"]
    assert cal.exists() and vec.read_text() == "theirs"


def test_learning_without_traced_chats_is_refused_before_anything_is_unloaded(monkeypatch):
    from metalscope import cli, steer
    from .helpers import write_norms
    monkeypatch.setattr(P, "no_dose_units", NO_DOSE_UNITS)
    why = NO_DOSE_UNITS("m")
    assert why == (f"m has no traced replies yet, and a concept's doses are sized from them (about {steer.NORM_MIN_TOKENS} "
                   "tokens of unsteered replies; 0 so far): chat a few unsteered messages with m first (metalscope on), "
                   "then learn again")
    assert P.learn_plan("humorous", "m")["taken"] == why
    _ds4_setup(monkeypatch)
    _stored()
    seen = _ds4_child(monkeypatch, CHILD_LINES)
    with pytest.raises(SystemExit, match="m has no traced replies yet"):     # the job Pi starts: refused before it is one
        cli.main(["concept", "learn", "humorous", "m", "--engine", "ds4", "--url", PROXY, "--job", "humorous-untraced"])
    assert "cmd" not in seen and gpulock.held() is None and state.list_jobs() == []
    write_norms("m", [0, 1])                             # traced: the plan goes ahead
    assert NO_DOSE_UNITS("m") is None and P.learn_plan("humorous", "m")["taken"] is None
    # Traced, but on one short reply: the plan and the job warn, and go ahead.
    p = models.norms_path("m")
    p.write_text(json.dumps(dict(json.loads(p.read_text()), tokens=300)))
    short = "dose units on m measured on 1 short trace (300 tokens): `metalscope vector norms m --refresh` after a real chat"
    assert P.learn_plan("humorous", "m")["notes"][-1] == short
    st = learn_job("humorous", None, None, "m", engine="ds4", url=PROXY, job="humorous-short").run()
    assert st["outcome"] == "done", st["error"]
    assert f"warning: {short}" in (state.job_dir("humorous-short") / "log.txt").read_text()


def test_a_stored_task_item_the_gpu_child_would_fail_on_is_refused_up_front(monkeypatch):
    from metalscope import cli
    from metalscope.learn import command as Cmd, plan as P
    d = _stored("oddtask")
    task = json.loads((d / "task.json").read_text())
    task["items"][3]["target"] = "not an option"
    (d / "task.json").write_text(json.dumps(task))
    # The structure check alone keeps 9 good items; the child uses every item as it stands, so the check fails too.
    assert len(W.valid_items(task["items"], "")) >= 8
    with pytest.raises(ValueError, match="item 4's target 'not an option' is not among its options"):
        W.read_texts(d, "oddtask")
    assert "item 4's target" in P.learn_plan("oddtask", "m")["taken"]
    monkeypatch.setattr(Cmd, "_main", lambda *a, **k: pytest.fail("a job started"))
    with pytest.raises(SystemExit, match="item 4's target"):
        cli.main(["concept", "learn", "oddtask", "m"])


def test_a_shipped_concept_learned_copies_its_texts_into_its_folder(monkeypatch):
    from metalscope import config
    from metalscope.concepts import corpus as C
    from metalscope.learn import job as J
    shipped = {p.name: p.read_text() for p in (C.DATA_DIR / "fluffy").iterdir()}
    home = models.concept_dir("m", "fluffy")
    # A child that dies takes back the copy (the shipped texts are still shipped) and the checkpoint's folder it made.
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES[:2] + ["RuntimeError: boom\n"], rc=1)
    vec = models.vector_path("m", "fluffy")

    def made_a_vector(self, *a, **k):
        vec.write_text("v")
        return real_run_gpu(self, *a, **k)
    real_run_gpu = J.GpuJob.run_gpu
    monkeypatch.setattr(J.GpuJob, "run_gpu", made_a_vector)
    st = learn_job("fluffy", None, None, "m", engine="ds4", url=PROXY, job="fluffy-died").run()
    assert st["outcome"] == "failed" and not home.exists() and not (state.job_dir("fluffy-died") / "concept").exists()
    assert {p.name: p.read_text() for p in (C.DATA_DIR / "fluffy").iterdir()} == shipped
    # A run that works leaves the texts in the model's folder, beside what it learned from them.
    monkeypatch.setattr(J.GpuJob, "run_gpu", real_run_gpu)
    _ds4_child(monkeypatch, CHILD_LINES)
    st = learn_job("fluffy", None, None, "m", engine="ds4", url=PROXY, job="fluffy-1").run()
    assert st["outcome"] == "done", st["error"]
    assert {p.name: p.read_text() for p in home.iterdir() if p.is_file()} == shipped
    assert st["corpus"]["path"] == str(home) and C.texts_dir("m", "fluffy") == home
    assert config.read_json(home / "concept.json")["learned"]["writer"] == "Claude Fable 5.1 (by hand)"


def test_new_texts_go_in_the_models_folder_and_a_failure_leaves_another_checkpoints_vector(monkeypatch):
    other = models.vector_path("elsewhere", "toasty")           # a vector made outside learn: a folder with no texts
    other.parent.mkdir(parents=True)
    other.write_text("v")
    _ds4_setup(monkeypatch)
    _ds4_child(monkeypatch, CHILD_LINES[:2] + ["RuntimeError: boom\n"], rc=1)
    st = learn_job("toasty", "Toasty things.", None, "m", engine="ds4", url=PROXY, job="toasty-died").run()
    assert st["outcome"] == "failed"
    assert not models.concept_dir("m", "toasty").exists() and other.exists()
    assert sorted(p.name for p in (state.job_dir("toasty-died") / "concept").iterdir()) == TEXT_FILES
    _ds4_child(monkeypatch, CHILD_LINES)
    st = learn_job("toasty", "Toasty things.", None, "m", engine="ds4", url=PROXY, job="toasty-1").run()
    assert st["outcome"] == "done", st["error"]
    assert (models.concept_dir("m", "toasty") / "concept.json").exists() and other.exists()
    assert sorted(p.name for p in other.parent.iterdir()) == ["vector.safetensors"]


def test_pi_learns_a_stored_concept_without_a_description(monkeypatch):
    started = []

    class Proc:
        pid = 4242

        def wait(self):
            return 0
    monkeypatch.setattr(spawn.subprocess, "Popen", lambda cmd, **k: started.append(cmd) or Proc())
    _stored()
    plan = P.learn_plan("humorous", "m")
    assert plan.pop("read")["path"] == str(_stored())
    assert plan == dict(name="humorous", model="m", source="own", taken=None,
                        texts="texts: written by DeepSeek-V4.1 (2026-10-20)", notes=[],
                        description="dry humour", writer="DeepSeek-V4.1", corpus=None, engine=None)
    out = spawn.learn("humorous", "", "m", None, "ds4", PROXY)
    assert started[0][3:7] == ["concept", "learn", "humorous", "m"] and "--description" not in started[0]
    assert P.learn_plan("toasty", "m")["source"] == "write" and P.learn_plan("list", "m")["taken"]
    assert P.learn_plan("fluffy", "m")["texts"] == "texts: hand-written by Claude Fable 5.1 (2026-10-05)"
    assert out["job"].startswith("humorous-")


def test_the_texts_line_says_who_wrote_them():
    from metalscope import config
    from metalscope.concepts import shipped_dir, texts_line
    d = _stored()
    assert texts_line(d) == 'texts: written by DeepSeek-V4.1 from "dry humour" (2026-10-20)'
    assert texts_line(d, request=False) == "texts: written by DeepSeek-V4.1 (2026-10-20)"   # /concept NAME's
    (d / "writing.json").unlink()
    assert texts_line(d) == 'texts: written by DeepSeek-V4.1 from "dry humour" (2026-10-20) (instructions not kept)'
    (d / "concept.json").write_text(json.dumps(dict(title="x")))
    assert texts_line(d) == "texts: no record of who wrote them"
    assert texts_line(config.home() / "concepts" / "toasty") == "texts: each model writes its own"   # a definition
    # Every shipped concept says it was written by hand, and when; its request is never the grader's style.
    from metalscope.learn import calibrate as K
    for name in ("angry", "doubt", "duty", "explain-reasoning", "fluffy", "flux", "robotic", "stoic"):
        where = shipped_dir(name)
        assert texts_line(where).startswith("texts: hand-written by Claude Fable 5.1 (2026-10-0"), name
        meta = json.loads((where / "concept.json").read_text())
        assert K.style_of(meta) == K.style_of({k: v for k, v in meta.items() if k != "learned"}), name
    assert config.home() not in shipped_dir("fluffy").parents


def test_every_way_in_refuses_a_bad_input_in_the_same_words_before_any_job(monkeypatch):
    """concept learn's ways in (the command line, Pi's confirm and start, the job Pi starts) take their checks from one
    plan (plan.learn_plan): a bad input gets the same words on each, and no job is made for it."""
    import re
    from metalscope import cli
    from metalscope.learn import command as Cmd, plan as P
    monkeypatch.setattr(P, "pick_engine", lambda name, url=None: "ds4")
    monkeypatch.setattr(Cmd, "_main", lambda *a, **k: pytest.fail("a job started"))
    _stored()
    for name in ("humorous", "crunchy"):
        vec = models.vector_path("m", name)
        vec.parent.mkdir(parents=True, exist_ok=True)
        vec.write_text("v")
    need = "toasty needs a description (--description): one line saying what it means"
    for name, desc, said in (("toasty", "", need), ("toasty", "   ", need),
                             ("list", "", "'list' is a command word, so it cannot name a concept: pick another name"),
                             ("humorous", "", "a steering vector named 'humorous' already exists for m"),
                             # taken, and no description: told it is taken, not to add a description first
                             ("crunchy", "", "a steering vector named 'crunchy' already exists for m")):
        assert P.learn_plan(name, "m", desc)["taken"] == said
        with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
            spawn.learn(name, desc, "m")
        for job in ([], ["--engine", "ds4", "--url", PROXY, "--job", "j-20261119-000000", "--origin", "pi"]):
            with pytest.raises(SystemExit, match=f"^error: {re.escape(said)}$"):
                cli.main(["concept", "learn", name, "m"] + (["--description", desc] if desc else []) + job)
    said = "on ds4 the corpus writer is the served model (m), not other: ds4 serves one model"
    with pytest.raises(ValueError, match=f"^{re.escape(said)}$"):
        spawn.learn("toasty", "d", "m", "other", "ds4")
    with pytest.raises(SystemExit, match=f"^error: {re.escape(said)}$"):
        cli.main(["concept", "learn", "toasty", "m", "--description", "d", "--writer", "other"])
    assert state.list_jobs() == []


def test_the_plan_says_what_will_run(tmp_path):
    """Where the texts come from is decided once, in the plan: the description and writer the job uses, the supplied
    folder as an absolute path, the engine, the texts read, and the notes."""
    from metalscope.learn import plan as P
    d = _stored()
    p = P.learn_plan("humorous", "m", "something else", engine="ds4")
    assert (p["source"], p["description"], p["writer"], p["corpus"], p["engine"]) == ("own", "dry humour", "DeepSeek-V4.1",
                                                                                       None, "ds4")
    assert p["read"]["path"] == str(d)
    assert p["notes"] == ["the description given is ignored: humorous's texts have their own"]
    src = _corpus(tmp_path / "c", learned=dict(writer="claude-opus-5-5", request="crisp textures"))
    p = P.learn_plan("toasty", "m", None, str(src))
    assert (p["source"], p["description"], p["writer"], p["corpus"]) == ("corpus", "crisp textures", "claude-opus-5-5",
                                                                          str(src.resolve()))
    assert p["read"]["log"]["supplied"] == str(src.resolve())
    p = P.learn_plan("toasty", "m", "  Toasty   things. ")
    assert (p["description"], p["writer"], p["read"], p["taken"]) == ("Toasty things.", "m", None, None)
    assert P.learn_plan("toasty", "m")["taken"] is None             # no description yet (Pi's confirm): not asked


# ---------------------------------------------------------------- a stored calibration, read

def _stored_calibration(served="m", concept="toy"):
    """A calibration of ``concept`` on ``served``: best 12 of 18 at 0.4."""
    prof = models.profile_path(served, concept)
    prof.parent.mkdir(parents=True, exist_ok=True)
    prof.write_text(json.dumps(dict(reader=dict(layer=5, best=dict(strength=0.4)))))
    table = [dict(dose=0.2, good=9, turns=18), dict(dose=0.3, good=11, turns=18), dict(dose=0.4, good=12, turns=18),
             dict(dose=0.6, good=4, turns=18)]
    cal = dict(version=4, layer=5, turns=18, default_dose=0.4, good=12, strengths=table, reason="12/18 good @0.4",
               profile_mtime=prof.stat().st_mtime)
    p = models.calibration_path(served, concept)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal))
    return p


def test_spawn_hands_the_corpus_on(monkeypatch, tmp_path):
    started = []

    class Proc:
        pid = 4242

        def wait(self):
            return 0
    monkeypatch.setattr(spawn.subprocess, "Popen", lambda cmd, **k: started.append(cmd) or Proc())
    _corpus(tmp_path / "kept")
    monkeypatch.setattr(spawn, "busy", lambda: None)                                 # one start after another
    out = spawn.learn("kept", "", "m", corpus=str(tmp_path / "kept"))
    assert started[0][3:9] == ["concept", "learn", "kept", "m", "--corpus", str(tmp_path / "kept")]
    assert "--pass" not in started[0] and out["pid"] == 4242
    _stored_calibration("m", "toy")
    spawn.calibrate("toy", "m")
    assert started[-1][3:7] == ["concept", "calibrate", "toy", "m"] and "--pass" not in started[-1]

"""The served model's lens in Pi (/jlens): its status from the files; and the GPU window every job that loads its own
copy goes through (lock, unload, memory, a child on its own copy, release, reload), on a stand-in job."""
import json
import signal

import pytest

from metalscope import cli, gpulock, jlens, models
from metalscope.learn import job as J, state, window as Win
from tests.helpers import fake_child, learn_job, scope_client, write_norms

WAIT_AVAILABLE = gpulock.wait_available        # the real one, before the window fixture replaces it


def fake_lens(home, name="m", mtime=None, checkpoint="m"):
    """A lens directory of ``checkpoint`` as convert_lens.py leaves it (the weights file only has to exist; lens.json
    names the checkpoint too)."""
    d = jlens.lenses_dir(checkpoint) / name
    d.mkdir(parents=True)
    (d / "lens.safetensors").write_bytes(b"")
    meta = dict(source="neuronpedia/jacobian-lens qwen3.5-2b", hf_model="Qwen/Qwen3.5-2B", n_prompts=283, d_model=8,
                layers=list(range(24)), checkpoint=checkpoint)
    (d / "lens.json").write_text(json.dumps(meta))
    return d


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    return scope_client(), tmp_path


# ---------------------------------------------------------------- status

def test_lens_status_route_reads_the_lens_files(client):
    c, home = client
    assert c.get("/scope/api/lens").status_code == 404                      # no model loaded or named
    r = c.get("/scope/api/lens?model=m").json()
    assert r == dict(model="m", checkpoint="m", served=["m"], installed=False, path=None, lenses_dir=str(home / "models" / "m" / "lenses"),
                     candidates=[])
    d = fake_lens(home)
    r = c.get("/scope/api/lens?model=m").json()
    assert r["installed"] and r["path"] == str(d) and r["source"] == "neuronpedia/jacobian-lens qwen3.5-2b"
    assert (r["hf_model"], r["n_prompts"], r["layers"]) == ("Qwen/Qwen3.5-2B", 283, 24)
    models.link("m-8bit", "m")
    r = c.get("/scope/api/lens?model=m-8bit").json()                         # a copy of m reads m's lens
    assert (r["path"], r["checkpoint"], r["served"]) == (str(d), "m", ["m-8bit"])


def test_lens_status_on_the_command_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    cli.main(["jlens", "status"])
    assert capsys.readouterr().out.startswith(f"no lenses under {tmp_path / 'models'}/*/lenses (published ones")
    fake_lens(tmp_path)
    cli.main(["jlens", "status", "m"])
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == [f"m: lens at {jlens.lenses_dir('m') / 'm'}",
                       "  neuronpedia/jacobian-lens qwen3.5-2b · hf_model Qwen/Qwen3.5-2B · 283 prompts · 24 layers",
                       "  lenses for it, best held-out hit@10 first (each on its own held-out windows):"]
    cli.main(["jlens", "status", "other"])
    assert capsys.readouterr().out.startswith("other: no lens (lenses go in ")


# ---------------------------------------------------------------- the job's window

class FakeAdmin:
    """oMLX's admin API: the model loaded until unloaded, back once loaded; every call recorded."""
    def __init__(self, loaded=True):
        self.loaded, self.calls = loaded, []

    def admin_models(self):
        return [dict(id="m", loaded=self.loaded, is_loading=False)]

    def trace_status(self):
        return dict(gpu_lock_enforced=True)

    def unload(self, model_id):
        self.calls.append(("unload", model_id))
        self.loaded = False
        return dict(status="unloaded")

    def load(self, model_id):
        self.calls.append(("load", model_id, gpulock.held() is None))           # the lock is released before the reload
        self.loaded = True
        return dict(message=f"loaded {model_id}")


class WindowJob(J.GpuJob):
    """A stand-in GPU job in the oMLX window: a child that loads the model and measures something."""
    KIND = "window-test"
    CHILD = "tests.window_child"
    NO_RESULT = "a measurement"
    WEIGHTS = {"locking": 0.0, "unloading": 0.1, "waiting for memory": 0.05, "loading the model": 0.35, "measuring": 0.35,
               "releasing": 0.05, "reloading": 0.1}
    STEPS = (Win.UNLOAD, Win.LOAD, ("measure", "measure", ("measuring",)), Win.RELOAD)

    def __init__(self, model_path, model_name, loader="auto", need_gb=None, admin=None, job=None):
        super().__init__(job, model_name, Win.OmlxWindow(model_path, loader, need_gb, admin), "cli", name=model_name,
                         description=None)

    def holder(self) -> str:
        return f"measuring {self.model_name}"


RESULT = dict(words=2, n_tokens=778)
LINES = [f"@@{json.dumps(ev)}\n" for ev in (dict(stage="loading the model"), dict(stage="measuring", n=1, total=2),
                                            dict(stage="measuring", n=2, total=2),
                                            dict(step="measuring", summary="2 words on 778 tokens"),
                                            dict(stage="done", result=RESULT))]


def test_a_gpu_job_takes_the_lock_unloads_runs_the_child_and_reloads(window, monkeypatch):
    home, waits, killed = window
    seen = {}
    fake_child(monkeypatch, LINES, seen=seen)
    admin = FakeAdmin()
    st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job="win-m-1").run()
    assert st["outcome"] == "done" and st["kind"] == "window-test" and st["served_was_loaded"] is True
    assert seen["holder"].what == "measuring m" and seen["pass_fds"]       # the child ran under the lock
    assert seen["cmd"][1:] == ["-m", WindowJob.CHILD, "m", "--model", "/ckpt", "--loader", "mlx_lm", "--memory-limit-gb", "94"]
    assert admin.calls == [("unload", "m"), ("load", "m", True)]                           # unloaded, then back after the lock
    assert gpulock.held() is None and admin.loaded and not killed
    assert st["result"] == RESULT
    steps = {s["key"]: (s["status"], s["summary"]) for s in st["steps"]}
    assert steps["unload"] == ("done", "m unloaded · 100 GB free") and steps["reload"] == ("done", "m is back")
    assert steps["measure"] == ("done", "2 words on 778 tokens") and steps["load"][0] == "done"
    assert state.read_state("win-m-1")["outcome"] == "done"
    assert 7 in waits                                                                     # memory checked for the run


def test_a_finished_job_releases_the_lock_cleanly_when_memory_settles_below_the_need(window, monkeypatch):
    """Memory comes back to 95 GB of the 96 needed and stays: the lock is released clean, not left at
    "releasing", which oMLX's guard would count as a killed run."""
    import itertools
    home, waits, killed = window
    seen, ticks = {}, itertools.count(0, 10)
    fake_child(monkeypatch, LINES, seen=seen)
    free = lambda: (5 if seen else 8) * gpulock.GB          # 8 GB before the child, 5 GB (of 7) after, steady
    monkeypatch.setattr(gpulock, "wait_available", lambda need, **k: WAIT_AVAILABLE(
        need, **k, reader=free, sleep=0, clock=lambda: next(ticks)))
    st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=FakeAdmin(), job="win-m-2").run()
    assert st["outcome"] == "done"
    assert json.loads(gpulock.lock_path().read_text())["released"] is True and gpulock.unclean() is None


def test_a_failed_or_cancelled_gpu_job_still_releases_the_lock_and_reloads(window, monkeypatch):
    home, waits, killed = window
    for kw, why in ((dict(rc=1), "the GPU run exited with code 1"), (dict(interrupt=True), "stopped")):
        seen, admin = {}, FakeAdmin()
        fake_child(monkeypatch, LINES[:1], seen=seen, **kw)                  # dies after "loading the model"
        st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job=f"win-m-{len(kw)}{why.split()[0]}").run()
        assert st["outcome"] == "failed" and st["error"].startswith(why), st["error"]
        assert admin.calls == [("unload", "m"), ("load", "m", True)] and gpulock.held() is None
        steps = {s["key"]: s["status"] for s in st["steps"]}
        assert steps["load"] == "failed" and steps["measure"] == "pending" and steps["reload"] == "done"
    assert killed == [(987654321, signal.SIGTERM)]                         # the cancelled child was stopped first
    # Another job holding the GPU: refused before anything is unloaded.
    admin = FakeAdmin()
    lock = gpulock.GpuLock(gpulock.Holder("learning other")).acquire()
    try:
        st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job="win-m-busy").run()
    finally:
        lock.release()
    assert st["outcome"] == "failed" and st["error"].startswith("the GPU is in use: learning other") and admin.calls == []
    # A model that was not loaded is neither unloaded nor loaded.
    seen, admin = {}, FakeAdmin(loaded=False)
    fake_child(monkeypatch, LINES, seen=seen)
    st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job="win-m-idle").run()
    assert st["outcome"] == "done" and admin.calls == [] and st["served_was_loaded"] is False


def test_an_old_lens_baseline_job_still_reads(window):
    """A lens-baseline job's state (a kind no longer made) still lists and reads."""
    home, _, _ = window
    d = state.job_dir("lens-baseline-m-1")
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps(dict(kind="lens-baseline", job="lens-baseline-m-1", name="m", outcome="done",
                                                  result=dict(words=["harm"], layers=[11], n_tokens=778, path="/x"))))
    st = state.read_state("lens-baseline-m-1")
    assert st["kind"] == "lens-baseline" and state.outcome_text(st) and state.doing(st) == "lens-baseline m"


def test_the_learn_job_keeps_its_child_and_steps():
    """The learn job runs in the same window (GpuJob); its child command and its steps are what they were."""
    write_norms("m", [0, 1])                     # dose units: the plan would refuse without them
    j = learn_job("x", "d", "/nonexistent", "m", admin=object(), job="x-same", seed=3, sweep_layers="8,16")
    assert j.CHILD == "metalscope.learn.gpu" and j.holder() == "learning x"
    assert j.window.child_args(j, "omlx", 90.4) ==["m", "--model", "/nonexistent", "--loader", "omlx", "--memory-limit-gb", "90",
                                          "x", "--seed", "3", "--sweep-layers", "8,16"]
    assert [s["key"] for s in j.state["steps"]] == [k for k, _, _ in J.LearnJob.STEPS] and j.state["kind"] == "learn"
    assert (j.state["name"], j.state["description"], j.state["writer"]) == ("x", "d", "m")

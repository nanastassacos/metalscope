"""A GPU job giving the GPU back (learn/job.py's ``release`` and learn/window.py's ``OmlxWindow.release``) when
things go wrong: memory not back, a child that ignores SIGTERM, a reload that fails or is cancelled; and gpulock's
memory reading and watchdog stop. The oMLX window's fixtures are test_lens.py's."""

import signal
import subprocess

import pytest

from metalscope import gpulock
from metalscope.learn import state, window as Win

from .test_lens import LINES, FakeAdmin, WindowJob, fake_child


def test_a_failed_job_ends_failed_with_no_eta(window, monkeypatch):
    seen = {}
    fake_child(monkeypatch, LINES[:1], rc=1, seen=seen)
    st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=FakeAdmin(), job="rel-failed").run()
    assert (st["outcome"], st["phase"], st["eta"]) == ("failed", "failed", None)
    assert (state.read_state("rel-failed")["phase"], state.read_state("rel-failed")["eta"]) == ("failed", None)


def test_memory_not_back_releases_the_lock_unclean_and_does_not_reload(window, monkeypatch):
    seen = {}
    fake_child(monkeypatch, LINES, seen=seen)
    monkeypatch.setattr(gpulock, "wait_available", lambda need, **k: not seen)     # enough before the child, not after
    admin = FakeAdmin()
    st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job="rel-memory").run()
    assert st["outcome"] == "done" and admin.calls == [("unload", "m")] and not admin.loaded
    assert next(s for s in st["steps"] if s["key"] == "reload")["summary"] == \
        "not reloaded: memory not back yet; the next chat loads it"
    assert gpulock.held() is None and gpulock.unclean() is not None                # oMLX's guard checks memory first


def test_a_jobs_own_window_unloads_and_reloads_while_it_reads_as_running(window, monkeypatch):
    """No model is taken away while a job runs (backends.gpu_free), except by the job itself: its window unloads and
    reloads through the admin API, never through that check."""
    from metalscope import backends

    class Admin(FakeAdmin):
        def unload(self, model_id):
            with pytest.raises(SystemExit, match="cannot unload m now: a job is running: window-test m"):
                backends.gpu_free("unload m")
            return super().unload(model_id)
    fake_child(monkeypatch, LINES, seen={})
    admin = Admin()
    st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job="rel-own").run()
    assert st["outcome"] == "done" and admin.calls == [("unload", "m"), ("load", "m", True)] and admin.loaded


def test_a_child_that_ignores_sigterm_is_killed(window):
    _, _, killed = window

    class Stubborn:
        pid, waits = 4242, []

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.waits.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired("child", timeout)
            return -9
    job = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=FakeAdmin(), job="rel-stubborn")
    job.child = Stubborn()
    job.release()
    assert killed == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)] and Stubborn.waits == [30, None]


def test_the_guard_kills_the_childs_group(window, monkeypatch):
    _, _, killed = window
    made = []

    class Dog:
        def __init__(self, **kw):
            made.append(kw)

        def start(self):
            pass
    monkeypatch.setattr(gpulock, "Watchdog", Dog)
    job = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=FakeAdmin(), job="rel-guard")
    job.window.guard(job, type("Child", (), {"pid": 77})())
    made[0]["exit_fn"](3)
    assert killed == [(77, signal.SIGKILL)] and made[0]["parent"] == 0


class FlakyAdmin(FakeAdmin):
    """oMLX refusing the reload ``fails`` times (``why``), then loading."""
    def __init__(self, fails, why="HTTP 500: busy"):
        super().__init__()
        self.fails, self.why = fails, why

    def load(self, model_id):
        if self.fails:
            self.fails -= 1
            self.calls.append(("refused", model_id))
            raise self.why if isinstance(self.why, BaseException) else Win.omlx_admin.ServerError(self.why)
        return super().load(model_id)


def test_a_failed_reload_is_retried_then_given_up(window, monkeypatch):
    slept = []
    monkeypatch.setattr(Win, "_sleep", slept.append)
    for fails, loaded, summary in ((2, True, "m is back"), (10, False, "could not reload; the next chat loads it")):
        seen, admin = {}, FlakyAdmin(fails)
        fake_child(monkeypatch, LINES, seen=seen)
        slept.clear()
        st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job=f"rel-retry-{fails}").run()
        assert st["outcome"] == "done" and admin.loaded is loaded and slept == [60] * fails
        assert next(s for s in st["steps"] if s["key"] == "reload")["summary"] == summary
    # Already loading or loaded (a chat request got there first): no retry.
    seen, admin = {}, FlakyAdmin(1, "HTTP 409: m is already loading")
    fake_child(monkeypatch, LINES, seen=seen)
    slept.clear()
    WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job="rel-409").run()
    assert admin.calls == [("unload", "m"), ("refused", "m")] and slept == []


def test_a_cancel_during_the_reload_still_reloads(window, monkeypatch):
    seen, admin = {}, FlakyAdmin(1, KeyboardInterrupt("stopped"))
    fake_child(monkeypatch, LINES, seen=seen)
    st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job="rel-cancel").run()
    assert st["outcome"] == "failed" and st["error"] == "stopped"
    assert admin.calls == [("unload", "m"), ("refused", "m"), ("load", "m", True)] and admin.loaded


def test_a_cancel_during_the_retry_wait_does_not_start_the_retries_again(window, monkeypatch):
    def cancel(_s):
        raise KeyboardInterrupt("stopped")
    monkeypatch.setattr(Win, "_sleep", cancel)
    seen, admin = {}, FlakyAdmin(10)
    fake_child(monkeypatch, LINES, seen=seen)
    st = WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job="rel-wait").run()
    assert (st["outcome"], st["phase"], st["eta"]) == ("failed", "failed", None)
    assert admin.calls == [("unload", "m"), ("refused", "m")]          # the next chat loads it


def test_cancels_that_cut_every_release_short_still_leave_the_outcome(window, monkeypatch):
    seen, admin = {}, FlakyAdmin(10, KeyboardInterrupt("stopped"))     # every reload cancelled mid-request
    fake_child(monkeypatch, LINES, seen=seen)
    with pytest.raises(KeyboardInterrupt):
        WindowJob("/ckpt", "m", loader="mlx_lm", need_gb=7, admin=admin, job="rel-cancels").run()
    st = state.read_state("rel-cancels")
    assert (st["outcome"], st["phase"], st["eta"]) == ("failed", "failed", None)
    assert admin.calls == [("unload", "m")] + [("refused", "m")] * 3


VM_STAT ="""Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100000.
Pages active:                            2000000.
Pages inactive:                            50000.
Pages speculative:                         25000.
Pages wired down:                         300000.
"Translation faults":                  123456789.
"""


def test_available_reads_free_inactive_and_speculative_pages(monkeypatch):
    run = lambda text: type("Done", (), {"stdout": text})()
    monkeypatch.setattr(gpulock.subprocess, "run", lambda *a, **k: run(VM_STAT))
    assert gpulock.available() == (100000 + 50000 + 25000) * 16384
    monkeypatch.setattr(gpulock.subprocess, "run", lambda *a, **k: run(VM_STAT.replace("16384", "4096")))
    assert gpulock.available() == (100000 + 50000 + 25000) * 4096
    monkeypatch.setattr(gpulock.subprocess, "run", lambda *a, **k: run("vm_stat: something else\n"))
    assert gpulock.available() is None

    def missing(*a, **k):
        raise FileNotFoundError("vm_stat")
    monkeypatch.setattr(gpulock.subprocess, "run", missing)
    assert gpulock.available() is None


def _ended(thread, timeout=5.0) -> bool:
    """Whether ``thread`` ended within ``timeout``, without join() or is_alive(): Watchdog's ``_stop`` event shadows
    Thread._stop, which both call once the thread is done."""
    import threading
    import time
    t0 = time.monotonic()
    while thread in threading.enumerate():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.01)
    return True


def test_a_stopped_watchdog_ends_without_exiting():
    import threading
    exits, checked = [], threading.Event()
    dog = gpulock.Watchdog(log=lambda s: None, exit_fn=exits.append, interval=0.01, pressure=lambda: checked.set(),
                           swap=lambda: 0)
    dog.start()
    assert checked.wait(5)
    dog.stop()
    assert _ended(dog) and exits == []
    # A reason: it logs it and exits with 3, once.
    logged = []
    dog = gpulock.Watchdog(log=logged.append, exit_fn=exits.append, interval=0.01, pressure=lambda: 4, swap=lambda: 0)
    dog.start()
    assert _ended(dog) and exits == [3] and logged == ["watchdog: memory pressure critical; stopping now"]

import pytest

# The process-wide data home is set in tests/__init__.py, before this file is imported (as tests.conftest).


@pytest.fixture(autouse=True)
def _no_leaked_methods():
    """monkeypatch.setattr on an instance (MANAGER, a process-wide object) puts back the class's method as an attribute
    of the instance, a bound method, which then hides a later test's patch of the class. Each test starts without them,
    whatever ran before."""
    from metalscope.live import MANAGER
    for obj in (MANAGER, MANAGER.hub, MANAGER.streamer):
        for name, value in list(vars(obj).items()):
            if getattr(value, "__self__", None) is obj and hasattr(type(obj), name):
                delattr(obj, name)


@pytest.fixture(autouse=True)
def _private_home(tmp_path_factory, monkeypatch):
    """Every test gets a data home of its own unless it sets one, and no ds4 build (not the machine's: whether it is
    current would decide what `ds4 start` does)."""
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path_factory.mktemp("home")))
    monkeypatch.setenv("METALSCOPE_DS4_DIR", str(tmp_path_factory.mktemp("ds4-build")))


@pytest.fixture(autouse=True)
def _no_omlx_settings(tmp_path_factory, monkeypatch):
    """oMLX's settings (its port and API key) are a file of the test's own, absent unless the test writes it: no test
    reads the machine's ~/.omlx/settings.json, and a request a test makes carries no real key."""
    from metalscope.omlx import admin
    monkeypatch.setattr(admin, "SETTINGS", tmp_path_factory.mktemp("omlx") / "settings.json")


@pytest.fixture(autouse=True)
def _sigterm_kept():
    """`ds4 serve` turns SIGTERM into KeyboardInterrupt for its process; a test that runs it here gets pytest's back."""
    import signal
    before = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, before)


@pytest.fixture(autouse=True)
def _no_port_checks(request, monkeypatch):
    """`ds4 start` and `ds4 serve` check their ports are free before loading; in tests every port reads free (8009 and
    8010 may be the machine's live servers), unless a test asks for ``real_ports`` and checks only ports it bound."""
    if "real_ports" not in request.fixturenames:
        from metalscope import backends
        monkeypatch.setattr(backends, "port_taken", lambda port: False)


@pytest.fixture(autouse=True)
def _no_omlx_listing(request, monkeypatch):
    """doctor lists oMLX's models (backends.model_lines); in tests oMLX is not running and lists none (8008 may be the
    machine's live server), unless a test asks for ``real_omlx_listing`` and stubs the admin API itself."""
    if "real_omlx_listing" not in request.fixturenames:
        from metalscope import backends
        monkeypatch.setattr(backends, "omlx_models", lambda: dict(up=False, url="http://127.0.0.1:8008", models=[], tracing=None))


@pytest.fixture
def real_omlx_listing():
    """The real backends.omlx_models, for a test that stubs oMLX's admin API."""


@pytest.fixture
def real_ports():
    """The real port check (backends.port_taken), for a test that checks only ports it bound itself."""


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A data home at tmp_path (for tests that write into it by path)."""
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def window(tmp_path, monkeypatch):
    """A GPU job's oMLX window faked (test_lens, test_gpu_release): a home at tmp_path, memory always enough, no real
    process signalled. ``(home, the memory waits asked for, the process groups signalled)``."""
    from metalscope import gpulock
    from metalscope.learn import job as J
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    waits = []
    monkeypatch.setattr(gpulock, "wait_no_loading", lambda admin, **k: True)
    monkeypatch.setattr(gpulock, "wait_available", lambda need, **k: waits.append(need) or True)
    monkeypatch.setattr(gpulock, "available", lambda: 100 * gpulock.GB)
    killed = []
    monkeypatch.setattr(J.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    return tmp_path, waits, killed


@pytest.fixture(autouse=True)
def omlx_app_left_alone(monkeypatch):
    """No test opens the real oMLX.app: `omlx start`'s helper (omlx/admin.cli_start) finds no app to open."""
    from metalscope.omlx import admin
    monkeypatch.setattr(admin, "_app_to_open", lambda: None)

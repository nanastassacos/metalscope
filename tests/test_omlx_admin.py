import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import pytest

from metalscope import cli
from metalscope.omlx import admin as omlx_admin
from metalscope.omlx.admin import Admin, ServerError

KEY = "sekret-key-123"


class Stub(ThreadingHTTPServer):
    """Mimics the oMLX 0.7.0 routes metalscope uses, including the admin session cookie."""

    def __init__(self, api_key=KEY, trace=True):
        super().__init__(("127.0.0.1", 0), Handler)
        self.api_key = api_key
        self.trace = trace
        self.models = {"alpha": dict(loaded=True, settings=dict(mtp_enabled=False, max_tokens=100)),
                       "beta": dict(loaded=False, settings=dict(mtp_enabled=False, max_tokens=100))}
        self.calls = []
        threading.Thread(target=self.serve_forever, args=(0.05,), daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, headers=()):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n)) if n else None

    def _route(self, method):
        s, path, body = self.server, self.path, self._body()
        s.calls.append((method, path))
        if path == "/v1/models" and method == "GET":
            if self.headers.get("Authorization") != f"Bearer {s.api_key}":
                return self._send(401, {"detail": "bad key"})
            return self._send(200, {"data": [{"id": m} for m in s.models]})
        if path == "/scope/api/status" and method == "GET" and s.trace:
            if s.api_key and self.headers.get("Authorization") != f"Bearer {s.api_key}":   # oMLX's key, when set
                return self._send(401, {"detail": "bad key"})
            return self._send(200, {"mode": "on", "models": [{"name": "alpha", "installed": True, "adapter": None}]})
        if path == "/admin/api/login" and method == "POST":
            if not s.api_key:
                return self._send(400, {"detail": "No API key configured. Please set up an API key first."})
            if body.get("api_key") != s.api_key:
                return self._send(401, {"detail": "Invalid API key"})
            return self._send(200, {"success": True},
                              [("Set-Cookie", "omlx_admin_session=tok; HttpOnly; Path=/; SameSite=lax")])
        if path.startswith("/admin/api/"):
            if "omlx_admin_session=tok" not in (self.headers.get("Cookie") or ""):
                return self._send(401, {"detail": "Not authenticated"})
            parts = [unquote(p) for p in path.split("/")[3:]]  # after /admin/api/
            if parts == ["models"] and method == "GET":
                return self._send(200, {"models": [dict(id=k, is_loading=False, **v) for k, v in s.models.items()]})
            if parts == ["server", "restart"] and method == "POST":
                return self._send(202, {"status": "restarting"})
            if len(parts) == 3 and parts[0] == "models":
                m = s.models.get(parts[1])
                if m is None:
                    return self._send(404, {"detail": f"Model not found: {parts[1]}"})
                if parts[2] == "load" and method == "POST":
                    m["loaded"] = True
                    return self._send(200, {"status": "ok", "message": f"Loaded {parts[1]}"})
                if parts[2] == "unload" and method == "POST":
                    if not m["loaded"]:
                        return self._send(400, {"detail": f"Model not loaded: {parts[1]}"})
                    m["loaded"] = False
                    return self._send(200, {"status": "ok", "message": f"Unloaded {parts[1]}"})
                if parts[2] == "settings" and method == "PUT":
                    m["settings"].update({k: v for k, v in body.items() if k in m["settings"]})
                    was_loaded, m["loaded"] = m["loaded"], False
                    return self._send(200, {"success": True, "settings": m["settings"], "requires_reload": True,
                                            "auto_unloaded": was_loaded, "auto_reloaded": False})
        self._send(404, {"detail": "Not Found"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.shutdown()
    s.server_close()


@pytest.fixture
def settings_file(tmp_path, monkeypatch, stub):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"server": {"port": stub.server_address[1]}, "auth": {"api_key": KEY}}))
    monkeypatch.setattr(omlx_admin, "SETTINGS", path)
    return path


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_settings_file_gives_url_and_key(settings_file, stub):
    admin = Admin()
    assert admin.base_url == stub.url and admin.api_key == KEY
    assert admin.models() == ["alpha", "beta"]


def test_status_via_cli(settings_file, stub, capsys):
    cli.main(["omlx", "status"])
    out = capsys.readouterr().out
    assert f"server {stub.url}: listening" in out
    assert "served models (2):" in out
    assert "alpha  [loaded]" in out and "  beta\n" in out
    assert "metalscope: tracing on; hooks on alpha" in out
    assert KEY not in out


def test_status_says_when_omlx_refuses_the_key(settings_file, stub, capsys):
    """oMLX running with another key than its settings file's (OMLX_API_KEY, --api-key): the /scope routes answer
    401, which is not "routes not mounted"; the line names the file, never a key."""
    stub.api_key = "another-key-456"
    cli.main(["omlx", "status"])
    out = capsys.readouterr().out
    assert (f"metalscope: oMLX at {stub.url} refused metalscope's API key: it sends the one in {settings_file}, and "
            "oMLX may run with another (OMLX_API_KEY, --api-key)\n") in out
    assert "routes not mounted" not in out and KEY not in out and "another-key" not in out


def test_status_without_trace_routes(stub, monkeypatch, tmp_path):
    """A tested oMLX without the routes: no hook, install it; a current hook, metalscope failed to load (the log)."""
    from metalscope.omlx import install
    monkeypatch.setattr(install, "omlx_version", lambda app: max(install.TESTED_OMLX))
    hook = dict(exists=False, current=False)
    monkeypatch.setattr(install, "hook_state", lambda app: hook)
    stub.trace = False
    out = []
    omlx_admin.status(Admin(base_url=stub.url, api_key=KEY), log=out.append, app=tmp_path / "oMLX.app")
    assert out[-1] == "metalscope: routes not mounted (run `metalscope omlx install`, then restart oMLX)"
    hook.update(exists=True, current=True)
    omlx_admin.status(Admin(base_url=stub.url, api_key=KEY), log=out.append, app=tmp_path / "oMLX.app")
    assert out[-1] == (f"metalscope: not loaded in this oMLX: metalscope omlx restart; if it stays, its log says why "
                       f"({omlx_admin.OMLX_LOG})")
    real = Admin._request

    def busy(self, method, path, *a, **k):              # no answer in time, or a 503: not "not loaded"
        if path == "/scope/api/status":
            raise ServerError(f"{method} {path}: HTTP 503: busy", 503)
        return real(self, method, path, *a, **k)
    monkeypatch.setattr(Admin, "_request", busy)
    omlx_admin.status(Admin(base_url=stub.url, api_key=KEY), log=out.append, app=tmp_path / "oMLX.app")
    assert "not loaded" not in out[-1]


def test_status_on_an_untested_omlx_says_so_instead_of_advising_a_reinstall(stub, monkeypatch, tmp_path):
    """oMLX updated itself to a version metalscope is not tested with: the hook loads and patches nothing, so the
    /scope routes are missing. Status says why and how to try anyway, not "run omlx install" (which changes nothing)."""
    from metalscope.omlx import install
    asked = []
    monkeypatch.setattr(install, "omlx_version", lambda app: asked.append(app) or "0.6.9")
    stub.trace = False
    out = []
    omlx_admin.status(Admin(base_url=stub.url, api_key=KEY), log=out.append, app=tmp_path / "oMLX.app")
    assert asked == [tmp_path / "oMLX.app"]
    text = "\n".join(out)
    assert "metalscope: off in this oMLX" in text and "omlx install" not in text
    assert f"oMLX 0.6.9 is not a version metalscope is tested with ({', '.join(sorted(install.TESTED_OMLX))})" in text
    assert "oMLX will run untraced (no traces or steering)" in text
    assert "install a tested oMLX version, or update metalscope once it supports oMLX 0.6.9" in text
    assert "launchctl" not in text


def test_status_when_down():
    url, out = f"http://127.0.0.1:{free_port()}", []
    omlx_admin.status(Admin(base_url=url, api_key=KEY), log=out.append)
    assert out == [f"server {url}: not listening"]


def test_server_down_is_a_clear_error():
    admin = Admin(base_url=f"http://127.0.0.1:{free_port()}", api_key=KEY)
    with pytest.raises(ServerError, match="not reachable"):
        admin.load("alpha")


def test_a_slow_answer_is_not_called_unreachable_and_http_errors_carry_their_status(stub, monkeypatch):
    """A load longer than its timeout: oMLX is up and may still be loading, so no "not reachable, restart" advice."""
    admin = Admin(base_url=stub.url, api_key=KEY)
    with pytest.raises(ServerError) as exc:
        admin.load("gamma")
    assert exc.value.status == 404
    real_open = admin._opener.open

    def slow(req, timeout=None):
        if req.full_url.endswith("/load"):
            raise TimeoutError("timed out")
        return real_open(req, timeout=timeout)
    monkeypatch.setattr(admin._opener, "open", slow)
    with pytest.raises(ServerError) as exc:
        admin.load("beta")
    assert str(exc.value) == ("oMLX did not answer POST /admin/api/models/beta/load within 900s: it may still be "
                              "working on it (metalscope omlx status)") and exc.value.status is None


def test_login_errors_never_show_the_key(stub):
    with pytest.raises(ServerError, match="oMLX has no API key set"):
        Admin(base_url=stub.url, api_key="").load("alpha")
    with pytest.raises(ServerError, match="rejected") as exc:
        Admin(base_url=stub.url, api_key="wrong-key").load("alpha")
    assert "wrong-key" not in str(exc.value)
    stub.api_key = None
    with pytest.raises(ServerError, match="oMLX has no API key set"):
        Admin(base_url=stub.url, api_key="x").load("alpha")


def test_cli_error_exit(stub, settings_file):
    settings_file.write_text(json.dumps({"server": {"port": stub.server_address[1]}}))
    with pytest.raises(SystemExit, match="error: oMLX has no API key set"):
        cli.main(["omlx", "load", "beta"])


def test_load_unload_use_the_session_cookie(stub):
    admin = Admin(base_url=stub.url, api_key=KEY)
    out = []
    omlx_admin.load(admin, "beta", log=out.append)
    assert stub.models["beta"]["loaded"] and out[0].startswith("Loaded beta (")
    omlx_admin.unload(admin, "beta", log=out.append)
    assert not stub.models["beta"]["loaded"] and out[1] == "Unloaded beta"
    assert stub.calls.count(("POST", "/admin/api/login")) == 1
    with pytest.raises(ServerError, match="HTTP 404: Model not found: gamma"):
        admin.load("gamma")


def test_model_ids_are_quoted(stub):
    stub.models["org/m x"] = dict(loaded=False, settings={})
    Admin(base_url=stub.url, api_key=KEY).load("org/m x")
    assert ("POST", "/admin/api/models/org%2Fm%20x/load") in stub.calls


def test_parse_assignments():
    assert omlx_admin.parse_assignments(["a=true", "b=4", "c=foo", "d={\"x\": 1}", "e=", "f=a=b"]) == \
        dict(a=True, b=4, c="foo", d={"x": 1}, e="", f="a=b")
    with pytest.raises(ServerError, match="key=value"):
        omlx_admin.parse_assignments(["novalue"])


def test_settings_prints_changed_keys_only(settings_file, stub, capsys):
    cli.main(["omlx", "settings", "alpha", "mtp_enabled=true", "max_tokens=100", "bogus=1"])
    out = capsys.readouterr().out.splitlines()
    assert stub.models["alpha"]["settings"]["mtp_enabled"] is True
    assert out == ["  mtp_enabled: false -> true",
                   "  bogus: not a model setting (ignored)",
                   "alpha unloaded to apply them; `metalscope omlx load alpha` to load it again"]


def test_settings_of_a_loaded_model_wait_for_a_running_job(settings_file, stub, monkeypatch, capsys):
    """oMLX reloads or unloads a loaded model to apply its settings: refused while a job runs (it may be talking to
    that model), nothing sent; a model not loaded is changed as before."""
    from metalscope.learn import state
    monkeypatch.setattr(state, "running", lambda exclude=None: dict(job="pirate-cal-1", kind="calibrate", name="pirate"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["omlx", "settings", "alpha", "mtp_enabled=true"])
    assert str(exc.value) == ("cannot change alpha's settings now: a job is running: calibrating pirate (/scope jobs); "
                              "wait for it or stop it: metalscope jobs cancel pirate-cal-1")
    assert not any(m == "PUT" for m, _ in stub.calls) and stub.models["alpha"]["loaded"]
    cli.main(["omlx", "settings", "beta", "mtp_enabled=true"])
    assert stub.models["beta"]["settings"]["mtp_enabled"] is True


def test_restart_via_the_admin_api_then_load(stub, tmp_path):
    """The API first, the CLI untouched: oMLX 0.8.0's `omlx restart` can take its menu-bar app down."""
    log_file = tmp_path / "args"
    fake = tmp_path / "omlx"
    fake.write_text(f"#!/bin/sh\necho \"$@\" >> {log_file}\n")
    fake.chmod(0o755)
    out = []
    omlx_admin.restart(Admin(base_url=stub.url, api_key=KEY), model="beta", cli=fake, down_timeout=0.2, log=out.append)
    assert ("POST", "/admin/api/server/restart") in stub.calls and not log_file.exists()
    assert stub.models["beta"]["loaded"]
    assert out[0].startswith("restart requested via admin API")
    assert out[1].startswith("server answering after") and out[2].startswith("loaded beta in")


def test_restart_runs_omlx_start_when_the_cli_restart_leaves_it_down(tmp_path):
    """No server for the API to ask: the CLI's restart, then its start (three tries) when nothing comes back (seen
    2026-10-14: `omlx restart` exited 0 having stopped the server; 2026-11-22: `omlx start` exited 0 with no server)."""
    log_file, fake = tmp_path / "args", tmp_path / "omlx"
    fake.write_text(f"#!/bin/sh\necho \"$@\" >> {log_file}\n")
    fake.chmod(0o755)
    admin = Admin(base_url=f"http://127.0.0.1:{free_port()}", api_key=KEY)
    out = []
    with pytest.raises(ServerError, match="not answering"):
        omlx_admin.restart(admin, cli=fake, timeout=0.3, down_timeout=0.1, log=out.append)
    assert log_file.read_text().splitlines() == ["restart --timeout 120", "start", "start", "start"]
    assert "running `omlx start`" in out[-1]


def test_restart_gives_up_when_server_stays_down():
    admin = Admin(base_url=f"http://127.0.0.1:{free_port()}", api_key=KEY)
    with pytest.raises(ServerError, match="not reachable"):
        omlx_admin.restart(admin, cli=None, timeout=0.5)


def test_the_server_pid_is_the_one_listening_on_its_port_not_any_omlx_server(monkeypatch):
    """A stray omlx-server (a test copy on another port) must not be taken for the server: the pid comes from the
    process listening on the server's port, pgrep only when lsof finds none."""
    import subprocess as sp
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        return sp.CompletedProcess(cmd, 0, stdout={"lsof": "91797\n", "pgrep": "7681\n91797\n"}[cmd[0]], stderr="")

    monkeypatch.setattr(omlx_admin.subprocess, "run", run)
    assert omlx_admin._pid("http://127.0.0.1:8008") == "91797"
    assert calls[0][:3] == ["lsof", "-nP", "-iTCP:8008"]
    monkeypatch.setattr(omlx_admin.subprocess, "run", lambda cmd, **k: sp.CompletedProcess(
        cmd, 0, stdout="" if cmd[0] == "lsof" else "7681\n", stderr=""))
    assert omlx_admin._pid("http://127.0.0.1:8008") == "7681"


def test_omlx_start_opens_the_menu_bar_app_first_when_it_is_not_running(tmp_path, monkeypatch):
    """`omlx start` left to open oMLX.app itself can fail to control it (0.8.2): the app quits, the server has
    nothing to restart it and the icon comes and goes at each later start. Opened first, in the background, once."""
    import types
    ran = []
    monkeypatch.setattr(omlx_admin.subprocess, "run", lambda cmd, **k: ran.append(cmd) or types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(omlx_admin, "APP_SETTLE", 0)
    app = tmp_path / "oMLX.app"
    monkeypatch.setattr(omlx_admin, "_app_to_open", lambda: app)
    omlx_admin.cli_start(tmp_path / "omlx", "--timeout", "120")
    assert ran == [["open", "-g", "-a", str(app)], [str(tmp_path / "omlx"), "start", "--timeout", "120"]]
    ran.clear()
    monkeypatch.setattr(omlx_admin, "_app_to_open", lambda: None)             # running, or not installed
    omlx_admin.cli_start(tmp_path / "omlx")
    assert ran == [[str(tmp_path / "omlx"), "start"]]


@pytest.mark.parametrize("pgrep,installed,opens", [(1, True, True), (0, True, False), (1, False, False), (2, True, False)])
def test_the_app_is_opened_only_when_installed_and_seen_not_running(tmp_path, monkeypatch, pgrep, installed, opens):
    import types
    from metalscope.omlx import install
    monkeypatch.undo()                                   # conftest's stand-in off: the real check, on stand-ins
    app = tmp_path / "oMLX.app"
    if installed:
        app.mkdir()
    monkeypatch.setattr(install, "DEFAULT_APP", app)
    monkeypatch.setattr(omlx_admin.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(returncode=pgrep))
    assert omlx_admin._app_to_open() == (app if opens else None)

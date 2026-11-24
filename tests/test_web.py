"""The /scope routes are only for this machine, and only for pages served from it."""
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from metalscope.middleware import _local_only, _loopback_url_host


def req(client="127.0.0.1", **headers) -> Request:
    scope = dict(type="http", method="GET", path="/scope/api/traces", query_string=b"",
                 client=(client, 50000) if client else None,
                 headers=[(k.replace("_", "-").lower().encode(), v.encode()) for k, v in headers.items()])
    return Request(scope)


def test_loopback_host_parsing():
    assert _loopback_url_host("127.0.0.1:8008") and _loopback_url_host("localhost") and _loopback_url_host("[::1]:8008")
    assert _loopback_url_host("http://127.0.0.1:8008") and _loopback_url_host("http://localhost:8008/scope")
    assert not _loopback_url_host("evil.example:8008") and not _loopback_url_host("http://evil.example")
    assert not _loopback_url_host("127.0.0.1.evil.example") and not _loopback_url_host("null")


def test_same_origin_browser_and_plain_clients_pass():
    _local_only(req(host="127.0.0.1:8008"))                                              # CLI, Pi: no Origin
    _local_only(req(host="localhost:8008", origin="http://localhost:8008", sec_fetch_site="same-origin"))
    _local_only(req(client="::1", host="[::1]:8008"))


@pytest.mark.parametrize("headers", [
    dict(host="127.0.0.1:8008", origin="https://evil.example"),               # cross-origin fetch (CORS is *)
    dict(host="127.0.0.1:8008", origin="null"),                               # sandboxed or file:// page
    dict(host="rebound.evil.example:8008"),                                   # DNS rebinding
    dict(host="127.0.0.1:8008", sec_fetch_site="cross-site"),
])
def test_other_sites_are_refused(headers):
    with pytest.raises(HTTPException) as e:
        _local_only(req(**headers))
    assert e.value.status_code == 403


def test_remote_client_refused_regardless_of_headers():
    with pytest.raises(HTTPException):
        _local_only(req(client="192.168.1.20", host="127.0.0.1:8008"))
    with pytest.raises(HTTPException):
        _local_only(req(client=None))


def test_a_page_cannot_change_state_through_the_scope_routes():
    """The routes that change something, on either server (oMLX mounts them as here; the ds4 proxy too, behind its own
    check): a page of another site is refused before the route runs, and a body that is not sent as JSON (a page's
    text/plain POST needs no CORS preflight) is refused as well. Pi and the CLI send JSON."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from metalscope import config, web
    from metalscope.omlx import backend as omlx_backend

    config.save_settings(mode="off")
    app = FastAPI()
    web.mount(app, omlx_backend.BACKEND)
    client = TestClient(app, base_url="http://127.0.0.1:8008", client=("127.0.0.1", 50000))
    for headers in ({"origin": "https://evil.example"}, {"host": "rebound.evil.example:8008"},
                    {"sec-fetch-site": "cross-site"}):
        assert client.post("/scope/api/mode", json={"mode": "on"}, headers=headers).status_code == 403
        assert client.post("/scope/api/steer", json={"enabled": True}, headers=headers).status_code == 403
        assert client.post("/scope/api/jobs/x-1/cancel", headers=headers).status_code == 403
    for ctype in ("text/plain", "application/x-www-form-urlencoded"):
        r = client.post("/scope/api/mode", content=b'{"mode": "on"}', headers={"content-type": ctype})
        assert r.status_code == 422, ctype
    assert config.load_settings().get("mode") == "off"
    page = {"origin": "http://127.0.0.1:8008", "sec-fetch-site": "same-origin"}
    assert client.post("/scope/api/mode", json={"mode": "on"}, headers=page).status_code == 200      # same origin
    assert client.post("/scope/api/mode", json={"mode": "off"}).status_code == 200                 # Pi, the CLI
    assert config.load_settings().get("mode") == "off"


def omlx_with_key(monkeypatch, key):
    """The /scope routes as oMLX mounts them, oMLX's key check stood in for: its verify_api_key as it reads a Bearer
    key or x-api-key (omlx/server.py), ``key`` None when none is set."""
    import types

    from fastapi import FastAPI
    from fastapi.security import HTTPBearer
    from fastapi.testclient import TestClient

    from metalscope import web
    from metalscope.omlx import backend

    async def verify_api_key(request, credentials):
        got = credentials.credentials if credentials is not None else request.headers.get("x-api-key")
        if got is None:
            raise HTTPException(status_code=401, detail="API key required")
        if got != key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return True
    state = types.SimpleNamespace(api_key=key)
    monkeypatch.setattr(backend, "_SERVER", types.SimpleNamespace(get_server_state=lambda: state, verify_api_key=verify_api_key,
                                                                  security=HTTPBearer(auto_error=False)))
    app = FastAPI()
    web.mount(app, backend.BACKEND)
    return TestClient(app, base_url="http://127.0.0.1:8008", client=("127.0.0.1", 50000))


def test_the_scope_api_needs_omlx_key_when_one_is_set(home, monkeypatch):
    """Any local process could read the live stream, switch steering or start GPU jobs: with oMLX's key set, every
    /scope route needs it, as oMLX's own API does; the local-only checks still come first; no key set, as before."""
    import types

    from metalscope import config, web
    from metalscope.omlx import backend

    config.save_settings(mode="off")
    c = omlx_with_key(monkeypatch, "k")
    for method, path, body in (("get", "/scope/api/status", None), ("get", "/scope/api/live", None),
                               ("post", "/scope/api/mode", {"mode": "on"}), ("get", "/scope/api/jobs", None),
                               ("post", "/scope/api/steer", {"enabled": True})):
        for headers in ({}, {"authorization": "Bearer wrong"}, {"x-api-key": "wrong"}):
            r = c.request(method.upper(), path, json=body, headers=headers)
            assert r.status_code == 401 and r.json()["detail"] == web.NEEDS_KEY, (path, headers, r.text)
    assert config.load_settings()["mode"] == "off" and not config.load_settings()["steer"]
    assert c.post("/scope/api/mode", json={"mode": "on"}, headers={"authorization": "Bearer k"}).status_code == 200
    assert c.get("/scope/api/jobs", headers={"x-api-key": "k"}).status_code == 200
    page = {"authorization": "Bearer k", "origin": "https://evil.example"}
    assert c.get("/scope/api/jobs", headers=page).status_code == 403                     # another site: refused first
    assert c.get("/scope/api/jobs", headers={"origin": "https://evil.example"}).status_code == 403
    assert omlx_with_key(monkeypatch, None).get("/scope/api/jobs").status_code == 200      # no key set: as before
    moved = types.SimpleNamespace(get_server_state=lambda: types.SimpleNamespace(api_key="k"))   # no verify_api_key
    monkeypatch.setattr(backend, "_SERVER", moved)
    r = c.get("/scope/api/jobs", headers={"authorization": "Bearer k"})
    assert r.status_code == 503 and "metalscope doctor" in r.json()["detail"]             # oMLX's check moved: closed


def test_a_page_of_another_site_cannot_trace_or_steer_through_omlx(home):
    """oMLX allows every CORS origin and header: a page could send X-Scope-Steer to learn the steering state and the
    vector names from the refusals, or run traced and steered replies. Any X-Scope header from another site is
    refused; a request without one is oMLX's to answer."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.testclient import TestClient

    from metalscope.middleware import TraceContextMiddleware

    app = FastAPI()
    app.post("/v1/chat/completions")(lambda: {"ok": True})
    app.add_middleware(TraceContextMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"])
    c = TestClient(app, base_url="http://127.0.0.1:8008")
    body = {"model": "m", "messages": []}
    for page in ({"origin": "https://evil.example"}, {"host": "rebound.evil.example:8008"}, {"sec-fetch-site": "cross-site"}):
        for header in ({"x-scope-steer": "calm@0.3/4"}, {"x-scope-trace": "on"}, {"x-scope-trace-client": "c"},
                       {"x-scope-edit": "r@1 calm"}):
            r = c.post("/v1/chat/completions", json=body, headers={**page, **header})
            assert r.status_code == 403 and "another site" in r.json()["error"]["message"], (page, header)
        assert c.post("/v1/chat/completions", json=body, headers=page).status_code == 200
    assert c.post("/v1/chat/completions", json=body, headers={"x-scope-trace": "off"}).status_code == 200


def test_a_job_asks_the_socket_it_was_started_on_not_the_host_header(home, monkeypatch):
    """A local client that forges the Host port must not point a detached job (its chat requests, oMLX's key) at a
    listener of its own: the job's URL is the address the request came in on."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from metalscope import web
    from metalscope.learn import spawn

    seen = []
    monkeypatch.setattr(spawn, "calibrate", lambda name, model, url: seen.append(url) or dict(job="x-1"))
    monkeypatch.setattr(spawn, "learn", lambda *a, **k: seen.append(a[5]) or dict(job="x-1"))
    for backend in (web.Backend("omlx"), web.Backend("ds4", frozenset({"measure_served"}))):
        app = FastAPI()
        web.mount(app, backend)
        c = TestClient(app, base_url="http://127.0.0.1:8008", client=("127.0.0.1", 50000))
        forged = {"host": "127.0.0.1:9999"}
        assert c.post("/scope/api/concept/calibrate", json={"name": "a", "model": "M"}, headers=forged).status_code == 200
        if backend.engine == "ds4":
            assert c.post("/scope/api/concept/learn", json={"name": "a", "model": "M"}, headers=forged).status_code == 200
    assert seen == ["http://127.0.0.1:8008"] * 3


def test_the_data_home_is_made_private_at_server_and_cli_start(tmp_path, monkeypatch):
    """Traces hold prompts and replies: the home is created 0700 and narrowed when broader, by the CLI and by the oMLX
    server as each starts; nothing outside it changes mode, nothing under it is touched."""
    import types

    from fastapi import FastAPI

    from metalscope import cli
    from metalscope.omlx import backend, patches

    parent = tmp_path / "parent"
    parent.mkdir(mode=0o755)
    home = parent / "home"
    monkeypatch.setenv("METALSCOPE_HOME", str(home))
    mode = lambda p: p.stat().st_mode & 0o777
    monkeypatch.setattr(cli, "cmd_jobs", lambda args: None)
    cli.main(["jobs"])
    assert mode(home) == 0o700 and mode(parent) == 0o755                       # created private
    home.chmod(0o755)
    (home / "index.jsonl").write_text("")
    (home / "index.jsonl").chmod(0o644)
    cli.main(["jobs"])
    assert mode(home) == 0o700 and mode(home / "index.jsonl") == 0o644          # narrowed; its files left alone
    home.chmod(0o750)
    monkeypatch.setattr(patches, "patch_uvicorn_shutdown", lambda: None)
    monkeypatch.setattr(backend, "_SERVER", None)
    patches.patch_server(types.SimpleNamespace(app=FastAPI()))
    assert mode(home) == 0o700


def test_a_server_on_a_unix_socket_says_a_job_cannot_ask_it():
    """No TCP address for a job to ask (a Unix socket): a 409 saying so, not a 500 or a URL with port None."""
    from metalscope import web
    for server in (None, ("/tmp/omlx.sock", None)):
        r = req()
        r.scope["server"] = server
        with pytest.raises(HTTPException) as e:
            web._self_url(r)
        assert e.value.status_code == 409 and "Unix socket" in e.value.detail
    r = req()
    r.scope["server"] = ("::1", 8008)
    assert web._self_url(r) == "http://[::1]:8008"


def test_making_the_home_private_never_stops_metalscope_and_narrows_only_its_own_folder(tmp_path, monkeypatch, capsys):
    """A home metalscope cannot narrow (not the user's, a volume without modes, /tmp) leaves one warning line, never a
    traceback: every CLI command would fail, and oMLX would start without metalscope. The user's own home folder, or
    one above it, named as the data home is never narrowed."""
    import os
    import types
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from metalscope import cli, config
    from metalscope.omlx import backend, patches

    mode = lambda p: p.stat().st_mode & 0o777
    user = tmp_path / "Users" / "me"
    user.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
    for named in (user, user.parent):                                         # METALSCOPE_HOME=~ (or above it)
        named.chmod(0o755)
        monkeypatch.setenv("METALSCOPE_HOME", str(named))
        config.private_home()
        assert mode(named) == 0o755, named
    other = tmp_path / "shared"
    other.mkdir(mode=0o777)
    other.chmod(0o777)
    monkeypatch.setenv("METALSCOPE_HOME", str(other))
    uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)                        # not the user's (/tmp, another account's)
    config.private_home()
    assert mode(other) == 0o777 and capsys.readouterr().err == ""
    monkeypatch.setattr(os, "getuid", lambda: uid)

    def refuse(self, m):
        raise PermissionError(1, "Operation not permitted")
    monkeypatch.setattr(Path, "chmod", refuse)                                # a volume without modes, a foreign owner
    ran = []
    monkeypatch.setattr(cli, "cmd_jobs", lambda args: ran.append(1))
    cli.main(["jobs"])
    err = capsys.readouterr().err
    assert ran and err.count("\n") == 1 and f"could not make {other} private" in err and mode(other) == 0o777
    app = FastAPI()
    monkeypatch.setattr(patches, "patch_uvicorn_shutdown", lambda: None)
    monkeypatch.setattr(backend, "_SERVER", None)
    patches.patch_server(types.SimpleNamespace(app=app, get_server_state=lambda: types.SimpleNamespace(api_key=None)))
    c = TestClient(app, base_url="http://127.0.0.1:8008", client=("127.0.0.1", 50000))
    assert c.get("/scope/api/jobs").status_code == 200                       # oMLX still gets metalscope
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "ro" / "home"))
    (tmp_path / "ro").mkdir(mode=0o500)
    config.private_home()                                                     # cannot even be created: a warning
    assert "could not make" in capsys.readouterr().err


def test_a_settings_file_that_is_not_an_object_reads_as_the_defaults(home):
    """Every chat request reads the mode: a list in the file raised AttributeError there (a 500 each)."""
    from metalscope import config
    config.settings_path().write_text("[1, 2]")
    assert config.load_settings() == dict(mode="off", steer=False)


def test_a_clients_header_picks_on_or_off_whatever_the_global_mode(home):
    """One trace level: a client's header traces at on or not at all (off: Pi's replay
    columns), whatever the global mode; without a header the global mode decides."""
    from metalscope import config
    from metalscope.live import MANAGER
    for mode, want in (("off", dict(on="on", lens="on", off=None, none=None)),
                       ("on", dict(on="on", lens="on", off=None, none="on"))):
        config.save_settings(mode=mode)
        got = {h: MANAGER.resolve_level(None if h == "none" else h) for h in want}
        assert got == want, mode


def test_code_stamp_changes_when_the_package_source_does(tmp_path):
    """A server says it runs older code (status `stale`) once a *.py file of its package changed on disk, however the
    package got there (a merge in a checkout, an upgrade); other files, a cache directory and Pi's node_modules do
    not count."""
    import os

    from metalscope import web
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "sub" / "b.py").write_text("y = 2\n")
    boot = web.code_stamp(pkg)
    (pkg / "notes.ts").write_text("ts")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "c.py").write_text("")
    (pkg / "sub" / "node_modules").mkdir()
    (pkg / "sub" / "node_modules" / "d.py").write_text("")
    assert web.code_stamp(pkg) == boot
    os.utime(pkg / "sub" / "b.py", ns=(1, 1))                              # touched
    assert web.code_stamp(pkg) != boot
    assert web.code_stamp(tmp_path / "gone") != boot                       # the package removed (an upgrade moved it)


def test_code_stamp_follows_the_link_omlx_imports_through(tmp_path):
    """oMLX imports the package through the hook's link: the stamp is of what the link points at now, so a link
    repointed at an upgraded copy (`metalscope omlx install`) reads as changed, and so does a dangling one."""
    import os
    import shutil

    from metalscope import web
    a, b, link = tmp_path / "a", tmp_path / "b", tmp_path / "metalscope"
    a.mkdir()
    (a / "web.py").write_text("x = 1\n")
    link.symlink_to(a, target_is_directory=True)
    boot = web.code_stamp(link)
    assert boot == web.code_stamp(a)
    shutil.copytree(a, b)                                                  # the same files, mtimes kept
    os.utime(b / "web.py", ns=(1, 1))                                      # one of them newer
    link.unlink()
    link.symlink_to(b, target_is_directory=True)
    assert web.code_stamp(link) != boot
    shutil.rmtree(b)
    assert web.code_stamp(link) != boot                                    # dangling


def test_a_host_or_origin_with_user_info_is_the_host_after_the_at():
    for value in ("http://127.0.0.1:80@evil.example", "http://[::1]@evil.example", "http://localhost@evil.example",
                  "http://evil.example#://127.0.0.1", "[::1", "http://[::1"):
        assert not _loopback_url_host(value), value
    assert _loopback_url_host("HTTP://LOCALHOST:8008") and _loopback_url_host(" [::1] ") and _loopback_url_host("http://[::1]:8008")


@pytest.mark.parametrize("value", ["127.0.0.1", "127.0.0.1:8008", "localhost", "localhost:8008", "LocalHost", "[::1]",
                                   "[::1]:8008", "::1", " 127.0.0.1\n", "http://localhost", "http://127.0.0.1:8008/path"])
def test_every_host_and_origin_this_machine_sends_passes(value):
    assert _loopback_url_host(value)


@pytest.mark.parametrize("value", ["http://evil.com\\@127.0.0.1", "localhost:8008:9", "localhost/", "localhost?x",
                                   "localhost#x", "http://user@localhost", "127.0.0.1@evil.example", "//localhost",
                                   "http:localhost", "localhost.", "127.0.0.1:abc", "http://[::1]x", "::1:8008", ""])
def test_a_host_or_origin_that_is_not_exactly_this_machine_is_refused(value):
    assert not _loopback_url_host(value)


def chat_client():
    """A chat route behind the request middleware, as the engines mount it."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from metalscope.middleware import TraceContextMiddleware

    app = FastAPI()
    app.post("/v1/chat/completions")(lambda: {"ok": True})
    app.add_middleware(TraceContextMiddleware)
    return TestClient(app, base_url="http://127.0.0.1:8008")


def test_a_deeply_nested_body_and_an_over_long_steer_name_are_answered_not_raised(home):
    from metalscope import config, models
    c = chat_client()
    for body in (b"[" * 100_000, b'{"a":' * 100_000):
        r = c.post("/v1/chat/completions", content=body, headers={"x-scope-trace": "on", "content-type": "application/json"})
        assert r.status_code == 200                                     # not ours to parse: the engine answers it
    config.save_settings(steer=True)
    models.concept_dir("m", "x").parent.mkdir(parents=True)          # a path reaches the name: longer than a file's
    for steer in ("a" * 300, "a" * 300 + "@0.5"):
        r = c.post("/v1/chat/completions", json={"model": "m"}, headers={"x-scope-steer": steer})
        assert r.status_code == 400 and r.json()["error"]["message"].startswith("bad x-scope-steer header: ")


def test_a_body_model_that_is_not_one_file_name_reads_no_steering_data(home, monkeypatch):
    """The body's model names the folders a steer is read from: ``..`` would read outside models/, so a name that is
    not one file name has no steering data, and the answer names it as sent; every name an engine serves passes."""
    from metalscope import config, middleware
    config.save_settings(steer=True)
    seen = []

    def steer_of(name, value, model):
        seen.append(model)
        raise ValueError("no such vector")
    monkeypatch.setattr(middleware, "_steer_of", steer_of)
    c = chat_client()
    for model in ("..", "org/..", ".", "qwen3:8b"):
        r = c.post("/v1/chat/completions", json={"model": model}, headers={"x-scope-steer": "calm"})
        assert r.status_code == 400 and r.json()["error"]["message"] == f"metalscope has no steering data for model {model!r}"
    names = ["org/Qwen3.5-2B-4bit", "Qwen3.8-Flash-Next-oQ4e-mtp", "DeepSeek-V4.1-Flash-ds4-q4k"]
    for model in names:
        assert c.post("/v1/chat/completions", json={"model": model}, headers={"x-scope-steer": "calm"}).status_code == 400
    assert seen == names

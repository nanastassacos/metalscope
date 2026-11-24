"""client.py against stand-in servers: which servers there are, which serves a model, the requests' auth and errors,
and a ds4 completion sent as the server's own token ids."""

import http.client
import io
import json
import urllib.error
import urllib.parse

import pytest

from metalscope import client
from metalscope.omlx import admin as omlx_admin


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def servers(monkeypatch):
    """Two servers by URL: oMLX's serving "omlx-model", the ds4 proxy's serving "ds4-model"; every request recorded."""
    sent = []
    status = {"http://omlx": dict(engine="omlx", caps=["load_copy"], models=[dict(name="omlx-model")]),
              client.DS4_URL: dict(engine="ds4", caps=["token_prompts"], models=[dict(name="ds4-model")])}

    def urlopen(req, timeout):
        url = urllib.parse.urlsplit(req if isinstance(req, str) else req.full_url)
        base, path = f"{url.scheme}://{url.netloc}", url.path + (f"?{url.query}" if url.query else "")
        body = None if isinstance(req, str) or req.data is None else json.loads(req.data)
        sent.append((base, path, body, {} if isinstance(req, str) else dict(req.header_items())))
        if base == "http://locked":
            raise urllib.error.HTTPError(path, 401, "unauthorized", {}, io.BytesIO(b'{"detail": "key"}'))
        if base not in status:
            raise urllib.error.URLError("refused")
        if path == "/scope/api/status":
            return Resp(json.dumps(status[base]).encode())
        if path == "/v1/scope/tokenize":
            return Resp(b'{"tokens": [0, 7, 9]}')
        if path == "/scope/api/plain-error":
            raise urllib.error.HTTPError(path, 500, "err", {}, io.BytesIO(b'{"error": "overloaded"}'))
        if path == "/scope/api/refuse":
            raise urllib.error.HTTPError(path, 409, "conflict", {}, io.BytesIO(b'{"detail": "busy"}'))
        if path == "/v1/cut":                                  # ds4-server's reply cut inside a character
            return Resp(b'{"choices": [{"text": "caf\xc3"}]}')
        if path == "/scope/api/html":
            return Resp(b"<html>gateway</html>")
        if path == "/v1/oom":                                  # oMLX out of memory: a 500 whose body is cut short
            class Cut(io.BytesIO):
                def read(self, *a):
                    raise http.client.IncompleteRead(b"x")
            raise urllib.error.HTTPError(path, 500, "err", {}, Cut())
        if path == "/v1/cut-ok":                               # a 200 whose body is cut short
            class CutResp(Resp):
                def read(self, *a):
                    raise http.client.IncompleteRead(b"x")
            return CutResp()
        if path.startswith("/scope/api/live"):
            raise urllib.error.URLError("no feed here")
        return Resp(json.dumps(dict(choices=[dict(text="hi")])).encode())

    monkeypatch.setattr(client.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(omlx_admin, "server_url", lambda settings=None: "http://omlx")
    monkeypatch.setattr(omlx_admin, "read_settings", lambda path=None: {})
    monkeypatch.delenv("METALSCOPE_URL", raising=False)
    return sent


def test_candidates_are_omlx_then_ds4_unless_metalscope_url_names_one(servers, monkeypatch):
    assert client.candidates() == ["http://omlx", client.DS4_URL]
    monkeypatch.setenv("METALSCOPE_URL", "http://elsewhere:9000/")
    assert client.candidates() == ["http://elsewhere:9000"]


def test_for_model_finds_the_server_serving_it(servers):
    assert [c.url for c in client.serving()] == ["http://omlx", client.DS4_URL]
    assert client.for_model("ds4-model").url == client.DS4_URL
    assert client.for_model("omlx-model").url == "http://omlx"
    assert client.for_model("unknown").url == "http://omlx"               # served nowhere: oMLX's, which loads on demand
    assert client.for_model(None).url == "http://omlx"


def test_requests_send_the_key_only_when_there_is_one_and_say_why_they_failed(servers):
    client.Client("http://omlx", api_key="k").get("/scope/api/x")
    assert servers[-1][3]["Authorization"] == "Bearer k"
    client.Client("http://omlx").get("/scope/api/x")                      # no key set: no "Bearer None"
    assert "Authorization" not in servers[-1][3]
    with pytest.raises(client.ServerError, match="HTTP 409: busy") as exc:
        client.Client("http://omlx").post("/scope/api/refuse", {})
    assert (exc.value.status, exc.value.detail) == (409, "busy")
    with pytest.raises(client.ServerError, match="HTTP 500: overloaded"):     # an error that is a plain string
        client.Client("http://omlx").get("/scope/api/plain-error")
    assert client.Client("http://omlx").get("/v1/cut")["choices"][0]["text"] == "caf\ufffd"
    with pytest.raises(client.ServerError, match="the reply is not JSON"):  # not "no metalscope server"
        client.Client("http://omlx").get("/scope/api/html")
    with pytest.raises(client.ServerError, match="no metalscope server at http://down") as exc:
        client.Client("http://down").get("/scope/api/x")
    assert exc.value.status is None
    with pytest.raises(client.ServerError, match="^POST /v1/oom: HTTP 500: ") as exc:      # cut short: still its status
        client.Client("http://omlx").post("/v1/oom", {})
    assert exc.value.status == 500
    with pytest.raises(client.ServerError, match=r"^GET /v1/cut-ok: the reply broke off \(IncompleteRead") as exc:
        client.Client("http://omlx").get("/v1/cut-ok")
    assert exc.value.status is None and isinstance(exc.value.__cause__, http.client.IncompleteRead)
    down = client.Client("http://down")
    assert down.status() is None and down.status_error.status is None                  # nothing answered
    up = client.Client("http://omlx")
    assert up.status() is not None and up.status_error is None
    locked = client.Client("http://locked")                                             # oMLX refusing the key
    assert locked.status() is None and locked.status_error.status == 401


def test_omlx_key_goes_to_omlx_address_only(servers, monkeypatch):
    """A process listening on the ds4 proxy's port (ds4 down) or on METALSCOPE_URL's never gets oMLX's key; oMLX's
    own address does, however this machine is named."""
    monkeypatch.setattr(omlx_admin, "read_settings", lambda path=None: {"auth": {"api_key": "k"}})
    auth = lambda c: (c.get("/scope/api/x"), servers[-1][3].get("Authorization"))[1]
    assert auth(client.Client("http://omlx")) == "Bearer k"
    assert auth(client.Client(client.DS4_URL)) is None
    assert client.Client("http://omlx:9999").api_key is None
    assert [auth(c) for c in client.serving()] == ["Bearer k", None]
    assert auth(client.for_model("ds4-model")) is None
    monkeypatch.setattr(omlx_admin, "server_url", lambda settings=None: "http://127.0.0.1:8008")
    for url, key in (("http://localhost:8008/", "k"), ("http://[::1]:8008", "k"), ("http://127.0.0.1:8009", None),
                     ("http://evil.example:8008", None)):
        assert client.Client(url).api_key == key, url


def test_no_test_reads_the_machines_omlx_key():
    """conftest points oMLX's settings at a file of the test's own: a real Client to oMLX's address gets no key."""
    assert omlx_admin.SETTINGS != omlx_admin.Path.home() / ".omlx/settings.json"
    assert omlx_admin.read_settings() == {}
    assert client.Client(omlx_admin.server_url()).api_key is None


def test_a_sessions_live_feed_sends_omlx_key_to_omlx_only(servers, monkeypatch):
    """oMLX's /scope routes need its key while one is set, the live feed's too."""
    import time
    monkeypatch.setattr(omlx_admin, "read_settings", lambda path=None: {"auth": {"api_key": "k"}})
    for url, want in (("http://omlx", "Bearer k"), (client.DS4_URL, None)):
        s = client.Session(client.Client(url), "c", wait=0)
        t0 = time.time()
        while not any(p.startswith("/scope/api/live") and b == url for b, p, *_ in servers) and time.time() - t0 < 5:
            time.sleep(0.01)
        s.stop = True
        live = next(h for b, p, _, h in servers if b == url and p.startswith("/scope/api/live"))
        assert live.get("Authorization") == want, url


def test_a_session_without_the_feed_subscribes_to_nothing_and_one_with_it_keeps_no_old_events(servers):
    s = client.Session(client.Client("http://omlx"), "c", feed=False)
    assert s.complete("omlx-model", "hey", "", 8, timeout=1)["text"] == "hi" and s.events == []
    assert not any(path.startswith("/scope/api/live") for _, path, _, _ in servers)
    # With the feed: a reply's events are its result, and gone from the session once read.
    real = s.server.request

    def request(*a, **kw):
        s.events += [dict(client="other", type="token", rid="x", i=0), dict(client="c", type="token", rid="r", i=0, text="hi"),
                     dict(client="c", type="end", rid="r")]
        return real(*a, **kw)
    s.server.request, s.alive = request, True
    for _ in range(3):
        out = s.complete("omlx-model", "hey", "", 8, timeout=1)
        assert out["rid"] == "r" and out["tok_text"] == ["hi"] and s.events == []


def test_a_reply_without_choices_is_the_servers_reason_not_a_keyerror(servers):
    """oMLX aborts a request (its model unloaded under it) with a body that has no choices: the error says so, with
    the server's own reason when the body has one."""
    s = client.Session(client.Client("http://omlx"), "c", feed=False)
    for body, said in (({"error": {"message": "Request aborted"}}, "the server ended the request: Request aborted"),
                       ({"error": "aborted"}, "the server ended the request: aborted"),
                       ({"detail": "unloading"}, "the server ended the request: unloading"),
                       ({}, "the server ended the request without a reply")):
        s.server.request = lambda *a, body=body, **k: body
        with pytest.raises(client.ServerError) as exc:
            s.complete("omlx-model", "hey", "", 8, timeout=1)
        assert str(exc.value) == said


def test_a_session_sends_a_token_prompts_server_its_own_ids(servers):
    s = client.Session(client.Client(client.DS4_URL), "c", wait=0)
    s.stop = True
    out = s.complete("ds4-model", "<｜User｜>hey", "calm@0.3/4", 8, timeout=1, temperature=0.0)
    tok = next(x for x in servers if x[1] == "/v1/scope/tokenize")
    assert tok[2] == {"text": "<｜User｜>hey", "specials": True}
    base, path, body, headers = servers[-1]
    assert (path, body["prompt"], body["temperature"]) == ("/v1/completions", [0, 7, 9], 0.0)
    assert headers["X-scope-steer"] == "calm@0.3/4" and headers["X-scope-trace-client"] == "c"
    assert headers["X-scope-trace"] == "on"
    assert out["text"] == "hi"
    s2 = client.Session(client.Client("http://omlx"), "c", wait=0)                # no token_prompts: the text as it is
    s2.stop = True
    s2.complete("omlx-model", "hey", "", 8, timeout=1)
    assert servers[-1][2]["prompt"] == "hey" and "temperature" not in servers[-1][2]
    s2.complete("omlx-model", "hey", "calm@0.3/4", 8, timeout=1, traced=False)      # steered all the same, no trace kept
    assert (servers[-1][3]["X-scope-trace"], servers[-1][3]["X-scope-steer"]) == ("off", "calm@0.3/4")
    s2.alive = True                                # with the feed: an untraced reply has no events to wait for
    t0 = __import__("time").time()
    s2.complete("omlx-model", "hey", "", 8, timeout=1, traced=False)
    assert __import__("time").time() - t0 < 5

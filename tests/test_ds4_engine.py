"""The ds4 engine (engines/ds4.py) against a stand-in ds4-server that answers as the patched build does: token-id
prompts recorded as sent, residual sums, the output at the logit ids, the reply, the tokenize route."""

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from metalscope import engines
from metalscope.ds4.frames import VERSION, ScopeReader
from metalscope.engines import ds4 as ds4_engine
from tests.test_ds4 import E, HC, L, V, ScopeWriter

W = HC * E


def resid(pos: int, layer: int) -> np.ndarray:
    """The stand-in model's residual at a position after a layer."""
    return (pos + 100 * layer + np.arange(W) / 1000).astype(np.float32)


def next_lp(pos: int, lookup: bool) -> float:
    """The stand-in's log-probability of the token after row ``pos``, a little lower without the lookup."""
    return -pos / 100 - (0 if lookup else 0.5)


def next_top(rows: list, ids: list, lookup: bool):
    """The stand-in's ten likeliest next tokens a row: the prompt's next token, then 20.., 30.. without the lookup."""
    base = 20 if lookup else 30
    got = np.array([[ids[p + 1]] + [base + j for j in range(9)] for p in rows])
    return got, np.tile(-np.arange(10, dtype=np.float32), (len(rows), 1)) - np.array(rows, np.float32)[:, None] / 100


class FakeDs4:
    """ds4-server's /v1/completions and /v1/scope/tokenize, writing the capture a request asks for."""

    def __init__(self, finish: str = "stop"):
        self.finish, self.bodies, self.steering, self.no_nxlp, self.no_nxtk = finish, [], [], False, False
        self.one_row = False                  # V4.1: next_lp's rows fed one at a time, a forward and NXLP/NXTK each
        self.model = None                     # the model a completion's reply names (ds4-server: the engine's id)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append((request.url.path, body))
        if request.url.path == "/v1/scope/tokenize":
            if "text" in body:
                special = body.get("specials") and body["text"] == "<|im_end|>"
                return httpx.Response(200, json={"tokens": [V - 1] if special else [ord(c) % V for c in body["text"]]})
            return httpx.Response(200, json={"tokens": [1, 2, 3, 4]})
        scope, ids = body["scope"], body["prompt"]
        if scope.get("steer"):
            self.steering.append(open(scope["steer"], "rb").read())
        w = ScopeWriter(scope["trace"], [], ids=scope.get("logit_ids", []))
        w.prompt(ids)
        if self.one_row and scope.get("next_lp"):          # the sweep up to the first row read, then a row a forward
            first = scope["next_lp"] - 1
            if first:
                w.step(0, ids[:first])
            for p in range(first, len(ids)):
                w.step(p, [ids[p]], out_top=11)
                if p + 1 < len(ids):
                    ids_, top = next_top([p], ids, scope.get("lookup", True))
                    w.nxlp(p, [next_lp(p, scope.get("lookup", True))])
                    w.nxtk(p, ids_, top)
        else:
            w.step(0, ids, out_top=11)
        if scope.get("next_lp") and not self.no_nxlp and not self.one_row:     # two records, as two prefill forwards
            rows = list(range(scope["next_lp"] - 1, len(ids) - 1))
            lp = [next_lp(p, scope.get("lookup", True)) for p in rows]
            ids_, top = next_top(rows, ids, scope.get("lookup", True))
            w.nxlp(rows[0], lp[:2])
            if not self.no_nxtk:
                w.nxtk(rows[0], ids_[:2], top[:2])
            w.nxlp(rows[0] + 2, lp[2:])
            if not self.no_nxtk:
                w.nxtk(rows[0] + 2, ids_[2:], top[2:])
        reply = [11, 12, 0][:body["max_tokens"] + (self.finish == "stop")]
        for j, tok in enumerate(reply[:-1]):
            w.step(len(ids) + j, [tok], out_top=reply[j + 1])
        if "rsum" in scope:
            P, sink = len(ids), min(scope["rsum"]["sink"], len(ids) - 1)
            total = np.stack([sum(resid(p, l) for p in range(sink, P)) for l in range(L)])
            last = np.stack([resid(P - 1, l) for l in range(L)])
            w.rsum(scope["rsum"]["sink"], [P - sink] * L, total, last)
        w.end(reply)
        return httpx.Response(200, json={"choices": [{"text": "", "finish_reason": self.finish}],
                                         **({"model": self.model} if self.model else {})})


@pytest.fixture
def make_engine(monkeypatch, tmp_path):
    class Vocab:
        def decode(self, ids):
            return "".join(f"<{int(i)}>" for i in ids)

        def __len__(self):
            return V

    monkeypatch.setattr(ds4_engine, "model_shape", lambda _g: dict(family="qwen4", n_layer=L, n_hc=HC, n_embd=E,
                                                                    k=2, n_expert=16))
    monkeypatch.setattr(ds4_engine, "tokenizer_for", lambda _g: Vocab())
    monkeypatch.setattr(ds4_engine, "read_gguf_meta", lambda _g: {"tokenizer.ggml.eos_token_id": 0})

    def make(finish: str = "stop"):
        fake = FakeDs4(finish)
        return ds4_engine.Ds4Engine("tiny-ds4", tmp_path / "m.gguf", "http://ds4", transport=httpx.MockTransport(fake)), fake
    return make


def test_shape_and_tokenizer_come_from_the_gguf_and_ds4(make_engine):
    eng, fake = make_engine()
    assert eng.shape == engines.Shape(L, W, HC, E, V)
    assert eng.tok.eos == {0, V - 1}                      # the GGUF's eos and <|im_end|> as ds4 reads it
    assert eng.tok.encode("ab") == [ord("a") % V, ord("b") % V] and eng.tok.decode([3, 4]) == "<3><4>"
    assert eng.tok.chat("hi") == [1, 2, 3, 4]
    path, body = fake.bodies[-1]
    assert path == "/v1/scope/tokenize" and body == {"messages": [{"role": "user", "content": "hi"}], "thinking": False}
    eng.tok.chat("hi", thinking=None)
    assert "thinking" not in fake.bodies[-1][1]


def test_residuals_are_ds4s_sums_from_the_sink_and_the_last_row(make_engine):
    eng, fake = make_engine()
    prompts = [[5, 6, 7, 8, 9, 10], [5, 6]]
    mean, last = eng.residuals(prompts, [1, 4], sink=4)
    assert mean.shape == last.shape == (2, 2, W)
    for i, ids in enumerate(prompts):
        for j, l in enumerate([1, 4]):
            rows = np.stack([resid(p, l) for p in range(len(ids))])
            np.testing.assert_allclose(mean[i, j], rows[min(4, len(ids) - 1):].mean(0), rtol=1e-6)
            np.testing.assert_array_equal(last[i, j], rows[-1])
    path, body = fake.bodies[-1]
    assert path == "/v1/completions" and body["prompt"] == [5, 6] and body["temperature"] == 0
    assert body["scope"]["rsum"] == {"sink": 4} and body["scope"]["topk"] == 1 and "steer" not in body["scope"]
    assert body["scope"]["fresh"] is True                 # a measurement reads its whole prompt
    assert not list(eng.runs.iterdir())                   # the request's files are gone


def test_logprobs_read_the_output_at_the_ids_and_adds_steer_every_position(make_engine):
    eng, fake = make_engine()
    adds = {2: np.full(W, 0.5, np.float32)}
    lp = eng.logprobs([5, 6, 7], [7, 3, 9], adds=adds)
    np.testing.assert_allclose(lp, [-1.0, -1.1, -1.2], rtol=1e-6)      # head row 0 (the stand-in's values)
    scope = fake.bodies[-1][1]["scope"]
    assert scope["logit_ids"] == [7, 3, 9] and scope["steer_decode_only"] is False
    steer = fake.steering[-1]
    assert steer[:8] == b"DS4STEER" and np.frombuffer(steer, "<f4", W, 28).tolist() == [0.5] * W
    with pytest.raises(ValueError):
        eng.logprobs([5], list(range(257)))


def test_generate_drops_the_stop_token_and_says_whether_it_stopped(make_engine):
    eng, _ = make_engine("stop")
    assert eng.generate([5, 6], None, 8) == ([11, 12], True)
    eng, _ = make_engine("length")
    assert eng.generate([5, 6], None, 2) == ([11, 12], False)


def test_a_model_change_mid_run_is_refused(make_engine):
    """ds4-server names the model it serves in each reply: a run's measurements stay on the first one."""
    eng, fake = make_engine()
    fake.model = "a"
    eng.generate([5, 6], None, 8)
    eng.logprobs([5, 6], [1])
    fake.model = "b"
    with pytest.raises(RuntimeError, match="now serves b, not a"):
        eng.logprobs([5, 6], [1])
    assert not list(eng.runs.iterdir())                   # the refused request's files are gone too


def test_files_a_run_left_behind_are_swept(make_engine, home):
    import os
    runs = home / "ds4" / "scope" / "engine"
    runs.mkdir(parents=True)
    old, new, other = runs / "a.bin", runs / "b.steer", runs / "c.txt"
    for f in (old, new, other):
        f.write_bytes(b"x")
    os.utime(old, (1, 1))
    os.utime(other, (1, 1))
    eng, _ = make_engine()
    assert eng.runs == runs and not old.exists() and new.exists() and other.exists()
    ds4_engine.sweep_runs(runs, now=os.stat(new).st_mtime + ds4_engine.STALE_S + 1)
    assert not new.exists() and other.exists()


def test_a_prompt_ds4_did_not_feed_as_sent_is_refused(make_engine, monkeypatch):
    eng, _ = make_engine()
    poll = ScopeReader.poll

    def wrapped(self):
        out = poll(self)
        if self.prompt:
            self.prompt = [1] + self.prompt                # as a chat template would have wrapped it
        return out
    monkeypatch.setattr(ScopeReader, "poll", wrapped)
    with pytest.raises(RuntimeError, match=rf"different prompt .*\(is it the format-{VERSION} build\?\)"):
        eng.logprobs([5, 6], [1])


def test_a_capture_without_residual_sums_is_refused(make_engine, monkeypatch):
    eng, _ = make_engine()
    poll = ScopeReader.poll

    def wrapped(self):
        out = poll(self)
        self.rsum = None                                   # as a build that writes no RSUM leaves it
        return out
    monkeypatch.setattr(ScopeReader, "poll", wrapped)
    with pytest.raises(RuntimeError, match=rf"no residual sums \(is it the format-{VERSION} build\?\)"):
        eng.residuals([[5, 6]], [1])


@pytest.mark.parametrize("cut", [0.3, 0.6, 0.97])
def test_a_capture_that_ends_just_after_the_reply_is_waited_for(make_engine, monkeypatch, cut):
    """ds4-server answers from inside the job and writes RSUM and END_ just after: the first read can find the file
    cut anywhere (mid-record included); the rest lands while the engine waits."""
    eng, fake = make_engine()
    want = (eng.residuals([[5, 6, 7, 8, 9, 10]], [1, 4], sink=4), eng.logprobs([5, 6, 7], [7, 3, 9]),
            eng.generate([5, 6], None, 8))
    answer, held = fake.__call__, []

    def reply_first(request):
        r = answer(request)
        scope = json.loads(request.content).get("scope")
        if scope:
            p = Path(scope["trace"])
            data = p.read_bytes()
            at = int(len(data) * cut)
            p.write_bytes(data[:at])
            held.append((p, data[at:]))
        return r
    eng._http = httpx.Client(base_url="http://ds4", transport=httpx.MockTransport(reply_first))
    sleep = ds4_engine.time.sleep

    def land(s):
        while held:
            p, rest = held.pop()
            with open(p, "ab") as f:
                f.write(rest)
        sleep(s)
    monkeypatch.setattr(ds4_engine.time, "sleep", land)
    got = (eng.residuals([[5, 6, 7, 8, 9, 10]], [1, 4], sink=4), eng.logprobs([5, 6, 7], [7, 3, 9]),
           eng.generate([5, 6], None, 8))
    for a, b in zip(want[0], got[0]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(want[1], got[1])
    assert want[2] == got[2] and not held               # every cut file was completed while the engine waited


def test_a_reply_cut_inside_a_character_is_read(make_engine):
    """ds4-server sends a reply cut at max_tokens as its bytes, which can end inside a UTF-8 character."""
    eng, fake = make_engine()
    answer = fake.__call__

    def cut(request):
        r = answer(request)
        if request.url.path == "/v1/completions":
            return httpx.Response(200, content=b'{"choices": [{"text": "caf\xc3", "finish_reason": "stop"}]}',
                                  headers={"content-type": "application/json"})
        return r
    eng._http = httpx.Client(base_url="http://ds4", transport=httpx.MockTransport(cut))
    assert eng.generate([5, 6], None, 8) == ([11, 12], True)          # the tokens come from the capture


def test_a_capture_that_did_not_finish_is_refused(make_engine, monkeypatch):
    eng, fake = make_engine()
    end = ScopeWriter.end
    monkeypatch.setattr(ScopeWriter, "end", lambda self, sampled, finish="done": end(self, sampled, "cancelled"))
    with pytest.raises(RuntimeError, match="ended 'cancelled', not done"):
        eng.generate([5, 6], None, 8)


def test_a_capture_that_never_ends_is_an_error(make_engine, monkeypatch):
    eng, _ = make_engine()
    poll = ScopeReader.poll

    def never(self):
        out = poll(self)
        self.end = None
        return out
    monkeypatch.setattr(ScopeReader, "poll", never)
    monkeypatch.setattr(ds4_engine, "END_WAIT", 0.05)
    with pytest.raises(RuntimeError, match="ended early"):
        eng.generate([5, 6], None, 8)


def test_open_finds_ds4_server_through_the_proxy(monkeypatch, make_engine):
    seen = {}

    def get(url, timeout):
        seen["url"] = url
        return httpx.Response(200, json={"name": "other-ds4", "gguf": "/m.gguf", "upstream": "http://127.0.0.1:8010"},
                              request=httpx.Request("GET", url))
    monkeypatch.setattr(ds4_engine.httpx, "get", get)
    with pytest.raises(RuntimeError, match="serves other-ds4"):
        ds4_engine.Ds4Engine.open("tiny-ds4")
    assert seen["url"] == "http://127.0.0.1:8009/scope/api/ds4"


def test_v41s_rows_fed_one_at_a_time_read_as_one_answer_and_each_tells_how_far(make_engine):
    """DeepSeek V4.1: the same request (next_lp, lookup false), ds4 feeding the rows from start - 1 on one at a time:
    a forward and an NXLP/NXTK record each, assembled into the same answer; ``fed`` as each forward's record comes."""
    eng, fake = make_engine()
    fake.one_row = True
    ids, fed = [5, 6, 7, 8, 9, 10], []
    for lookup in (True, False):
        got = eng.next_logprobs(ids, 2, lookup, threading.Event(), lambda rows, chunk: fed.append((rows, chunk)))
        np.testing.assert_allclose(got.lp, [next_lp(p, lookup) for p in range(1, 5)], rtol=1e-6)
        want_ids, want_top = next_top(list(range(1, 5)), ids, lookup)
        np.testing.assert_array_equal(got.ids, want_ids)
        np.testing.assert_allclose(got.top, want_top, rtol=1e-6)
        scope = fake.bodies[-1][1]["scope"]
        assert scope["next_lp"] == 2 and scope.get("lookup", True) is lookup and scope["fresh"] is True
        assert fake.bodies[-1][1]["max_tokens"] == 1
    assert fed == [(r, 1) for r in range(1, 8)] * 2       # row 0 swept, rows 1..5 alone, then the reply's token
    assert not list(eng.runs.iterdir())


def test_next_logprobs_read_each_row_from_start_with_the_lookup_on_or_left_out(make_engine):
    eng, fake = make_engine()
    ids = [5, 6, 7, 8, 9, 10]
    lp, top_ids, top = eng.next_logprobs(ids, 2)
    np.testing.assert_allclose(lp, [next_lp(p, True) for p in range(1, 5)], rtol=1e-6)    # rows 1..4: tokens 2..5
    want_ids, want_top = next_top(list(range(1, 5)), ids, True)
    np.testing.assert_array_equal(top_ids, want_ids)                                    # NXTK, across both records
    np.testing.assert_allclose(top, want_top, rtol=1e-6)
    assert top_ids.shape == (4, engines.NEXT_TOP)
    scope = fake.bodies[-1][1]["scope"]
    assert scope["next_lp"] == 2 and "lookup" not in scope and scope["fresh"] is True
    off = eng.next_logprobs(ids, 2, lookup=False)
    np.testing.assert_allclose(off.lp, lp - 0.5, rtol=1e-6)
    np.testing.assert_array_equal(off.ids[:, 1:], next_top(list(range(1, 5)), ids, False)[0][:, 1:])
    assert fake.bodies[-1][1]["scope"]["lookup"] is False
    fake.no_nxtk = True                               # a build before the top choices (NXTK): refused, as without NXLP
    with pytest.raises(RuntimeError, match="NXTK"):
        eng.next_logprobs(ids, 2)
    fake.no_nxtk = False
    with pytest.raises(ValueError):
        eng.next_logprobs(ids, 0)
    fake.no_nxlp = True                               # a ds4-server without next_lp ignores the key
    with pytest.raises(RuntimeError, match="next_lp"):
        eng.next_logprobs(ids, 2)
    fake.no_nxlp = False
    again = eng.next_logprobs(ids, 2, stop=threading.Event())        # one that can be called off, never called off
    np.testing.assert_array_equal(again.lp, lp)
    assert not list(eng.runs.iterdir())


@pytest.mark.parametrize("how", ["stop", "fed fails", "begun as dropped"])
def test_a_called_off_request_is_dropped_then_waited_for_until_ds4_ends_its_capture(make_engine, how):
    """Meanwhile each prefill chunk's record says how far it is (``fed``: rows read, the chunk's rows). ``stop`` set
    (or the watch failing): the request's connection is dropped (ds4-server sees its socket close and cancels the job
    between prefill chunks); the capture ds4-server began, even as it was dropped, is waited for to its end (END_
    "cancelled"), then removed."""
    eng, _ = make_engine()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    eng.upstream, eng._transport = f"http://127.0.0.1:{srv.getsockname()[1]}", None
    seen = {}

    def ds4():
        conn, _ = srv.accept()
        data = b""
        while b"\r\n\r\n" not in data:
            data += conn.recv(65536)
        head, body = data.split(b"\r\n\r\n", 1)
        n = int(next(h.split(b":")[1] for h in head.split(b"\r\n") if h.lower().startswith(b"content-length")))
        while len(body) < n:
            body += conn.recv(65536)
        req = json.loads(body)
        if how == "begun as dropped":
            seen["dropped"] = conn.recv(1) == b""
            time.sleep(0.1)
        w = ScopeWriter(req["scope"]["trace"], [])                     # the request begun
        w.prompt(req["prompt"])
        if how != "begun as dropped":
            w.step(0, req["prompt"][:2])                               # its first chunk, 2 rows
            time.sleep(0.05)
            w.step(2, req["prompt"][2:4])
            seen["dropped"] = conn.recv(1) == b""                      # until the engine goes
        time.sleep(0.3)                                                # the prefill chunk under way ends
        seen["ended"] = time.monotonic()
        w.end([], "cancelled")
        conn.close()
    t = threading.Thread(target=ds4)
    t.start()
    stop, fed = threading.Event(), []

    def tell(rows, chunk):
        fed.append((rows, chunk))
        if how == "fed fails" and rows == 4:
            raise ValueError("the watch failed")
    if how != "fed fails":
        threading.Timer(0.4, stop.set).start()
    with pytest.raises(ValueError if how == "fed fails" else engines.Stopped):
        eng.next_logprobs([5, 6, 7, 8, 9], 1, stop=stop, fed=tell)
    returned = time.monotonic()
    t.join(5)
    srv.close()
    assert fed == ([] if how == "begun as dropped" else [(2, 2), (4, 2)])
    assert seen["dropped"] and returned >= seen["ended"] and not list(eng.runs.iterdir())

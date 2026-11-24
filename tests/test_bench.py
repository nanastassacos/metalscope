"""metalscope bench against a stand-in server: what it sends, the table, the n/a paths, and that it puts back the
lens step whatever happens (and never touches the global tracing mode)."""

import io
import json

import pytest

from metalscope import bench, cli, client, config, livecheck

STEPS = [dict(n=n, layers=list(range(48 - n, 48))) for n in (4, 8, 15, 24, 47)]
CALM = dict(name="angry", reader=dict(layer=8), steer=dict(layer=8, strength=0.6, basis="chat"))


class FakeServer:
    """A client.Client's answers: status, lens steps, concepts, the lens step set, the engram inspection, and chat
    streams as a server streams them (a reply cut at max_tokens: every token; one that stopped: all but the stop
    token), pieces 0.01 s apart (traced: 0.02 s). Each traced reply leaves the trace ``last``."""

    def __init__(self, **kw):
        self.url = "http://x"
        self.s = dict(caps=["edit_at", "lookup", "token_prompts"], active=0, step=3, read_step=None, band=[18, 44],
                      steer_enabled=True, concepts=[CALM], words=["tiger"], lookup_fails=[], edit_fails=None,
                      too_long=10_000, fail_at=None, step_fails=[], finish="length", posts=[], chats=[], fresh=[])
        self.s.update(kw)

    def status(self, timeout=3, fresh=False):
        self.s["fresh"].append(fresh)
        return dict(engine="ds4", caps=self.caps(), mode="off", active=self.s["active"],
                    models=[dict(name="M")])

    def caps(self):
        return self.s["caps"]

    def served(self):
        return ["M"]

    def tokenize(self, text):
        return [0] * (len(text) // 4)

    def get(self, path, timeout=60):
        if path.startswith("/scope/api/lens/layers"):
            return dict(model="M", at=self.s["step"], steps=STEPS, **({"edit_band": self.s["band"]} if self.s["band"] else {}))
        if path.startswith("/scope/api/concepts"):
            return dict(steer_enabled=self.s["steer_enabled"], concepts=self.s["concepts"])
        raise AssertionError(path)

    def post(self, path, body, timeout=60):
        self.s["posts"].append((path, body))
        if path == "/scope/api/lens/layers":
            if self.s["step_fails"]:
                raise self.s["step_fails"].pop(0)
            self.s["step"] = body["step"]
            return {}
        if path == "/scope/api/lookup":
            if self.s["lookup_fails"]:
                why = self.s["lookup_fails"].pop(0)
                raise client.ServerError(f"HTTP 409: {why}", 409, why)
            return dict(ids=[1, 2, 3])
        raise AssertionError(path)                # never /scope/api/mode

    def stream(self, path, body, timeout=900, headers=None):
        traced = headers[config.HEADER] == "on"
        assert headers[config.CLIENT_HEADER] == "bench" and body["stream"] and body["temperature"] == 0.0
        self.s["chats"].append((traced, self.s["step"], headers.get(config.EDIT_HEADER)))
        if self.s["fail_at"] is not None and len(self.s["chats"]) > self.s["fail_at"]:
            raise self.s["fail_with"]
        if headers.get(config.EDIT_HEADER) and self.s["edit_fails"]:
            raise client.ServerError("HTTP 400", 400, self.s["edit_fails"])
        prompt = sum(len(m["content"]) for m in body["messages"]) // 4
        if prompt > self.s["too_long"]:
            raise client.ServerError("HTTP 400", 400, "the prompt is longer than the context")
        if traced:
            self.s["last"] = dict(id=f"t{len(self.s['chats'])}", decode_tps=50.0,
                                  lens_step=self.s["step"] if self.s["read_step"] is None else self.s["read_step"])
        n, dt = body["max_tokens"], 0.02 if traced else 0.01
        streamed = n if self.s["finish"] == "length" else n - 1
        for i in range(streamed):
            yield 0.5 + i * dt, dict(choices=[dict(delta=dict(content="x"))])
        yield 0.5 + (streamed - 1) * dt, dict(choices=[dict(delta={}, finish_reason=self.s["finish"])])
        yield 0.5 + (streamed - 1) * dt, dict(choices=[], usage=dict(completion_tokens=n, prompt_tokens=prompt))


SERVERS, SLEPT = [], []


@pytest.fixture(autouse=True)
def saved(monkeypatch):
    """Each traced reply saved at once (the newest server's), its request id in its meta; the rests not slept."""
    SERVERS.clear()
    SLEPT.clear()
    monkeypatch.setattr(bench.time, "sleep", SLEPT.append)
    monkeypatch.setattr(livecheck, "saved_traces", lambda t0, model, n=1, wait=20, client=livecheck.CLIENT:
                        [SERVERS[-1].s["last"]])
    monkeypatch.setattr(livecheck, "trace_meta", lambda e: dict(request_id="r1", prompt_len=5))
    real = livecheck.word_vectors           # without a J-lens: none made (a POST here is an AssertionError)
    monkeypatch.setattr(livecheck, "word_vectors", lambda server, model, say, jlens:
                        list(server.s["words"]) if jlens else real(server, model, say, jlens))
    monkeypatch.setattr(bench, "machine", lambda: "Test Mac, 64 GB")


def fake(**kw):
    SERVERS.append(FakeServer(**kw))
    return SERVERS[-1]


def run(srv, **kw):
    return bench.run(srv, {}, **dict(dict(repeats=1, contexts=[1000], max_tokens=20, log=lambda s: None), **kw))


def test_tps_counts_the_gaps_between_streamed_tokens():
    assert bench.tps(dict(tokens=300, finish="length", first_s=1.0, last_s=3.99)) == 100.0     # 299 gaps
    assert bench.tps(dict(tokens=300, finish="stop", first_s=1.0, last_s=3.98)) == 100.0       # stop token unstreamed
    assert bench.tps(dict(tokens=None, pieces=12, finish="length", first_s=0.0, last_s=1.1)) == 10.0   # no usage
    assert bench.tps(dict(tokens=1, finish="length", first_s=0.0, last_s=1.0)) is None
    assert bench.tps(dict(tokens=300, first_s=None, last_s=None)) is None


def test_the_conversation_is_deterministic_and_about_as_long_as_asked():
    a, b = bench.conversation(8000), bench.conversation(8000)
    assert a == b and a[-1]["role"] == "user" and [m["role"] for m in a[:2]] == ["user", "assistant"]
    assert 8000 <= sum(len(m["content"]) for m in a) < 8800


@pytest.mark.parametrize("finish", ["length", "stop"])
def test_a_full_run_measures_everything_and_puts_back_the_lens_step(finish):
    srv = fake(finish=finish)
    res = run(srv, repeats=2, contexts=[1000, 4000])
    assert srv.s["fresh"][0] is True                                # the idle check reads status afresh
    assert [(r["traced"], r["step"], r.get("layers")) for r in res["speed"]] == \
        [(False, None, None), (True, config.LENS_OFF, 0)] + [(True, k, s["n"]) for k, s in enumerate(STEPS)] + \
        [(False, None, None)]
    assert [r.get("again") for r in res["speed"]] == [None] * 7 + [True] and res["speed"][-1]["tps"] == 100.0
    assert all(len(r["runs"]) == 2 and "warmup" in r and "read_at" not in r for r in res["speed"])
    assert res["speed"][0]["tps"] == 100.0 and res["speed"][1]["tps"] == 50.0
    assert res["speed"][1]["runs"][0]["server_tps"] == 50.0 and res["speed"][1]["runs"][0]["lens_step"] == config.LENS_OFF
    assert res["speed"][2]["runs"][0]["lens_step"] == 0
    assert srv.s["chats"][0][:2] == (False, 3) and srv.s["chats"][3][:2] == (True, config.LENS_OFF)
    assert srv.s["chats"][6][:2] == (True, 0)
    assert all(p != "/scope/api/mode" for p, _ in srv.s["posts"])     # the global mode is left alone
    ed = res["edit"]
    assert ed["at"] == 10 and ed["of"] == 20 and ed["steer"] == "remove~tiger/18-44" and ed["written"] == 10
    assert srv.s["chats"][-3][2] == "r1@10 remove~tiger/18-44" and srv.s["chats"][-3][1] == config.LENS_STEP_DEFAULT
    rows = res["lookup"]["rows"]
    assert [r["target"] for r in rows] == [1000, 4000] and all("seconds" in r and r["read"] == 3 for r in rows)
    assert abs(rows[0]["prompt_tokens"] - 1000) < 60
    assert srv.s["step"] == 3 and res["restored"] == dict(lens_step=3)
    lines = bench.table(res)
    assert lines[0].startswith("M · ds4 · Test Mac, 64 GB") and "http://x" not in lines[0]
    text = "\n".join(lines)
    assert "—" not in text and "watch" not in text and "tracing off" in text
    assert "tracing on, 47 layers read (step 4)" in text and "fewest" not in text
    at = [k for k, line in enumerate(lines) if line.startswith(("  tracing off ", "  tracing on, lenses off ", "  tracing on, 4 layers"))]
    assert len(at) == 3 and at == list(range(at[0], at[0] + 3)) and " 50.0  " + "█" * 8 + " " * 8 + "  -50%" in lines[at[1]]
    again = [line for line in lines if line.startswith("  tracing off, again ")]
    assert len(again) == 1 and again[0].endswith("+0%") and lines.index(again[0]) == at[0] + 7
    assert "Engram inspection (the conversation read twice)" in text and "until the new word appears" in text


@pytest.mark.parametrize("exc", [client.ServerError("POST /v1/chat/completions: HTTP 500: boom", 500, "boom"),
                                 KeyboardInterrupt()])
def test_an_error_or_ctrl_c_midway_still_puts_back_the_lens_step(exc, tmp_path, monkeypatch, capsys):
    srv = fake(fail_at=6, fail_with=exc, step=1)
    monkeypatch.setattr(bench, "pick", lambda backend, model=None: (srv, "M"))
    with pytest.raises(SystemExit):
        bench.main(repeats=1, contexts=[1000], max_tokens=20, json_path=str(tmp_path / "b.json"))
    assert srv.s["step"] == 1
    saved = json.loads((tmp_path / "b.json").read_text())
    assert saved["restored"] == dict(lens_step=1) and len(saved["speed"]) == 4 and saved["error"]
    assert saved["speed"][-1]["runs"] == [] and "warmup" not in saved["speed"][-1]       # step 1 cut short
    out = capsys.readouterr().out
    assert "put back: lens step 1" in out and "stopped early" in out


def test_the_restore_survives_ctrl_c_once_and_says_what_it_could_not_put_back():
    def putter(*fails):
        calls, fails = [], list(fails)

        def put(k):
            calls.append(k)
            if fails:
                raise fails.pop(0)
        return put, calls
    logs = []
    put, calls = putter(KeyboardInterrupt())
    assert livecheck.restore(put, 3, logs.append) == dict(lens_step=3) and calls == [3, 3] and logs == []
    put, calls = putter(KeyboardInterrupt(), KeyboardInterrupt())
    assert livecheck.restore(put, 3, logs.append) == dict(lens_step="not put back to 3 (interrupted)")
    put, calls = putter(client.ServerError("HTTP 409: gone", 409))           # a refusal: not asked again
    assert livecheck.restore(put, 3, logs.append) == dict(lens_step="not put back to 3 (HTTP 409: gone)") and calls == [3]
    assert logs == ["  ! could not put the lens step back to 3 (interrupted): set it with - + in the dock's LENS",
                    "  ! could not put the lens step back to 3 (HTTP 409: gone): set it with - + in the dock's LENS"]


def test_ctrl_c_midway_then_inside_the_restore_still_puts_the_step_back():
    srv = fake(step=3, fail_at=4, fail_with=KeyboardInterrupt())     # stopped at step 0's warm-up
    put, first = srv.post, []

    def post(path, body, timeout=60):
        if body.get("step") == 3 and not first:
            first.append(1)
            raise KeyboardInterrupt()
        return put(path, body, timeout)
    srv.post = post
    res = {}
    with pytest.raises(KeyboardInterrupt):
        bench.run(srv, res, repeats=1, contexts=[1000], max_tokens=20, log=lambda s: None)
    assert srv.s["step"] == 3 and res["restored"] == dict(lens_step=3)


def test_each_measurement_starts_after_a_rest():
    res = run(fake(), rest=7, contexts=[1000, 4000])
    assert SLEPT == [7] * (8 + 1 + 2) and res["rest"] == 7          # 8 speed rows, the edit, each conversation
    SLEPT.clear()
    assert run(fake(), rest=0)["rest"] == 0 and SLEPT == []


def test_a_busy_server_is_refused_and_nothing_changed():
    srv = fake(active=1)
    with pytest.raises(SystemExit, match="tracing 1 request"):
        run(srv)
    assert srv.s["posts"] == [] and srv.s["chats"] == []


def test_a_reply_read_at_another_step_is_flagged():
    srv = fake(read_step=2)
    res = run(srv)
    assert [r.get("read_at") for r in res["speed"]] == [None, [2], [2], [2], None, [2], [2], None]
    text = "\n".join(bench.table(res))
    assert "read at step 2, not 0" in text and "read at step 2, not -1" in text and "read at step 2, not 2" not in text


def test_na_where_the_server_cannot():
    res = run(fake(caps=[]))
    assert res["lookup"] == dict(na="no engram on this model here")
    assert res["edit"]["na"].startswith("this server cannot change one word")
    srv = fake(caps=["edit_at", "lookup"], lookup_fails=["M has no engram"] * 2, too_long=2000)
    res = run(srv, contexts=[1000, 16000])
    a, b = res["lookup"]["rows"]
    assert a["na"] == "M has no engram" and b["na"] == "refused: the prompt is longer than the context"
    text = "\n".join(bench.table(res))
    assert "n/a: M has no engram" in text
    assert any("after ~16,000 tokens of conversation" in l and "n/a: refused" in l for l in text.splitlines())
    assert run(fake(steer_enabled=False))["edit"]["na"] == "steering is off (metalscope steering on)"
    assert run(fake(words=[], concepts=[]))["edit"]["na"] == "no word vector or concept to edit with"


def test_the_edit_removes_a_word_over_the_servers_band():
    """Inkling 2's band: the last block's visits, not the workspace band."""
    srv = fake(band=[42, 46])
    res = run(srv)
    assert res["edit"]["steer"] == "remove~tiger/42-46" and "first_s" in res["edit"]
    assert [c[2] for c in srv.s["chats"] if c[2]] == ["r1@10 remove~tiger/42-46"]


def test_the_edit_without_a_j_lens_still_runs_and_falls_back_to_a_concept():
    res = run(fake(band=None))          # no edit_band: no J-lens (DeepSeek V4.1 on ds4), no word vector: the concept's
    assert res["edit"]["steer"] == "angry@0.6/8" and "first_s" in res["edit"]
    assert "no J-lens on this model: no word to edit with" in res["edit"]["notes"]
    res = run(fake(words=[]))                        # a J-lens but no word vector: the concept's steer too
    assert res["edit"]["steer"] == "angry@0.6/8" and "first_s" in res["edit"]


def test_an_edit_the_server_refuses_is_na_with_its_reason():
    res = run(fake(edit_fails="that reply's tokens are not kept here"))
    assert res["edit"]["na"] == "that reply's tokens are not kept here" and res["edit"]["at"] == 10
    assert "n/a: that reply's tokens are not kept here" in "\n".join(bench.table(res))


def test_the_engram_inspection_is_asked_once_more_when_omlx_gave_back_expert_slots():
    slots = "oMLX gave back expert slots during the reading: ask again"
    srv = fake(lookup_fails=[slots])
    row = run(srv)["lookup"]["rows"][0]
    assert row["retried"] == slots and "seconds" in row and "na" not in row
    row = run(fake(lookup_fails=[slots, slots]))["lookup"]["rows"][0]          # once only
    assert row["na"] == slots and "seconds" not in row


def test_contexts_empty_skips_the_engram_inspection_and_repeats_must_be_above_0(monkeypatch, capsys):
    srv = fake()
    res = run(srv, contexts=[])
    assert res["lookup"] == dict(skipped=True) and all(p != "/scope/api/lookup" for p, _ in srv.s["posts"])
    assert "Engram" not in "\n".join(bench.table(res))
    got = []
    monkeypatch.setattr(bench, "main", lambda *a: got.append(a))
    cli.main(["bench", "--contexts", ""])
    cli.main(["bench", "--quick", "--model", "M"])
    cli.main(["bench", "--rest", "5", "--quick"])
    assert got[0][2] == [] and got[1][1:3] == (1, [1000]) and got[1][5] == "M"
    assert [a[6] for a in got] == [None, 0, 5.0]                     # --quick: no rest unless --rest says
    with pytest.raises(SystemExit):
        cli.main(["bench", "--repeats", "0"])
    assert "0 is not above 0" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="--rest"):
        cli.main(["bench", "--rest", "-1"])
    with pytest.raises(SystemExit, match="--rest"):
        cli.main(["bench", "--rest", "nan"])


def test_client_stream_yields_each_event_with_its_time(monkeypatch):
    body = (b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n: ping\n\ndata: {"choices":[]}\n\ndata: [DONE]\n\n')
    c = client.Client("http://x", api_key="")
    monkeypatch.setattr(c, "_open", lambda *a: io.BytesIO(body))
    got = list(c.stream("/v1/chat/completions", {}))
    assert [e for _, e in got] == [{"choices": [{"delta": {"content": "a"}}]}, {"choices": []}]
    assert all(t >= 0 for t, _ in got)


def test_writing_speeds_get_a_bar_against_the_fastest_and_the_change_from_tracing_off():
    res = dict(max_tokens=300, repeats=3, speed=[
        dict(traced=False, step=None, tps=16.0),
        dict(traced=True, step=config.LENS_OFF, layers=0, tps=15.7),
        dict(traced=True, step=0, layers=3, tps=15.3),
        dict(traced=True, step=1, layers=7, tps=None, read_at=[2]),
        dict(traced=False, step=None, again=True, tps=15.8)])
    assert bench.table(res) == [
        f"{'Writing speed (300 tokens, median of 3)':<46} tokens/s",
        f"  {'tracing off':<44}     16.0  " + "█" * 16,
        f"  {'tracing on, lenses off':<44}     15.7  " + "█" * 15 + "▊   -2%",
        f"  {'tracing on, 3 layers read (step 0)':<44}     15.3  " + "█" * 15 + "▎   -4%",
        f"  {'tracing on, 7 layers read (step 1)':<44}        ?" + " " * 24 + "  read at step 2, not 1",
        f"  {'tracing off, again':<44}     15.8  " + "█" * 15 + "▊   -1%"]        # the drift from the first
    assert max(map(len, bench.table(res)[:4])) < 80

"""metalscope doctor --live against a stand-in server: what each finding says, and when the check fails."""

import base64
import json
import time
import zipfile

import pytest

from metalscope import client, config, livecheck, models


def concepts(*items, steer_enabled=True):
    return dict(steer_enabled=steer_enabled, concepts=list(items))


def test_pick_concept_prefers_a_calibrated_one_at_its_dose():
    plain = dict(name="calm", reader=dict(layer=24), calibration=None)
    cal = dict(name="angry", reader=dict(layer=20), steer=dict(layer=20, strength=0.4, basis="chat"))
    stale = dict(name="calm2", reader=dict(layer=20), steer=None)                   # not calibrated: no default
    assert livecheck.pick_concept(concepts(plain, stale, cal)) == ("angry", 20, 0.4)
    assert livecheck.pick_concept(concepts(plain)) == ("calm", 24, livecheck.DOSE)
    assert livecheck.pick_concept(concepts(dict(name="x", reader=None), dict(name="err", error="bad"))) is None


def test_the_edit_steer_removes_a_word_over_the_servers_edit_band_else_steers_the_concept():
    """One spec for doctor --live and bench's edit; the band the server's (/scope/api/lens/layers' edit_band)."""
    assert livecheck.edit_steer(("angry", 20, 0.4), ["tiger", "lion"], [15, 36]) == "remove~tiger/15-36"
    assert livecheck.edit_steer(None, ["tiger"], [42, 46]) == "remove~tiger/42-46"           # Inkling 2's last block
    assert livecheck.edit_steer(("angry", 20, 0.4), [], [15, 36]) == "angry@0.4/20"
    assert livecheck.edit_steer(("angry", 20, 0.4), ["tiger"], None) == "angry@0.4/20"      # no J-lens: no band
    assert livecheck.edit_steer(None, [], [15, 36]) is None


def test_pick_words_takes_one_token_vectors_first_then_words_to_make():
    one, cut = dict(cut=[]), dict(cut=["jaguar"])
    assert livecheck.pick_words(dict(tiger=None, lion=one, Asia=one)) == ["lion", "Asia"]
    assert livecheck.pick_words(dict(tiger=one, lion=None, Asia=None)) == ["tiger", "lion"]
    assert livecheck.pick_words(dict(tiger=False, jaguar=cut, Asia=None)) == ["Asia"]     # a concept, a cut word
    assert livecheck.pick_words(dict(tiger=False)) == []


def test_the_table_puts_each_feature_by_engine():
    ds4 = {"traced reply": [True], "lens rows": [None, True], "expert routing": [True, False]}
    omlx = {"traced reply": [True], "one-token edit": [None]}
    assert [livecheck.verdict(g) for g in ([True, None], [None], [], [True, False])] == \
        ["passed", "not checked", "not checked", "failed"]
    lines = livecheck.table([("ds4", ds4), ("omlx", omlx)])
    assert len(lines) == 1 + len(livecheck.FEATURES) and lines[0].split() == ["Live", "ds4", "omlx"]
    rows = [" ".join(l.split()) for l in lines[1:]]
    assert rows[0] == "traced reply passed passed" and "lens rows passed not checked" in rows
    assert "expert routing failed not checked" in rows and "one-token edit not checked not checked" in rows
    assert lines[1].index(" passed") + 1 == lines[0].index("ds4") and lines[1].rindex(" passed") + 1 == lines[0].index("omlx")
    assert livecheck.summary("ds4", ds4) == f"ds4 · 2 passed · 1 failed · {len(livecheck.FEATURES) - 3} not checked"
    assert livecheck.summary("ds4", {f: [True] for f in livecheck.FEATURES}) == f"ds4 · {len(livecheck.FEATURES)} passed"


N_TOK = 12
P_LEN = 5


def events(rid, experts=True, lens_by="jlens", stream="shares"):
    """A reply's token events as the feed sends them to a client that asked for routing, the lens top 10s and the
    writes: ``stream`` "shares" each layer's writes (3 layers, 4 streams) and their shares, "writes" the writes alone,
    "some" shares on every other token, anything else none."""
    tops = dict(l=[24, 31], k=10, i=[1], w=["a"], c=[0] * 20, p=[0] * 20)
    writes = lambda i: dict(writes=[0.5] * 12, write_k=4, **(dict(shares=[0, 0.1, 0.2]) if stream == "shares"
                                                             or stream == "some" and i % 2 else {}))
    return [dict(type="token", rid=rid, i=i, text=f"t{i}", p=0.9, lens_by=lens_by, lens=[[24, 1, "a"], [31, 2, "b"]],
                 lens_logit=[[24, 1, "a"], [31, 2, "b"], [47, 3, "c"]], lens10=tops, lens10_logit=tops,
                 **(dict(experts=[1, 2], expert_w=[200, 55], expert_k=2) if experts else {}),
                 **(writes(i) if stream in ("shares", "writes", "some") else {})) for i in range(N_TOK)]


class FakeServer:
    """A client.Client's answers: its status, the routes doctor --live reads, a word vector made, the edit's stream."""

    def __init__(self, state):
        self.state, self.url = state, "http://x"

    def status(self):
        return dict(engine="omlx", caps=self.caps(), models=[dict(name="M")])

    def served(self):
        return ["M"]

    def caps(self):
        return self.state["caps"]

    def get(self, path):
        if path.startswith("/scope/api/concepts"):
            return self.state["concepts"]
        if path.startswith("/scope/api/lens/layers"):      # with a J-lens, the band the word edit acts over
            band = self.state.get("band", [18, 44]) if self.state.get("lens_by", "jlens") == "jlens" else None
            return dict(model="M", at=self.state.get("at", 0), steps=[dict(name="small", n=2, layers=[24, 31, 47])],
                        **({"edit_band": band} if band else {}))
        if "/routing" in path:
            L = self.state["moe"]
            n, k = N_TOK - 1, 2 if L else 0
            return dict(start=0, n=n if L else 0, L=L, k=k, ids=base64.b64encode(bytes(n * L * k * 2)).decode(), w="")
        if "/experts/specialists" in path:
            return dict(tokens=100, experts=[dict(e=7, top=["x"])])
        if "/experts/expert" in path:
            return dict(e=7, n=30, top=[["x", 2.0, 30]])
        raise AssertionError(path)

    def post(self, path, body, timeout=60):
        if path == "/scope/api/lens/layers":
            self.state.setdefault("steps_set", []).append(body["step"])
            self.state["at"] = body["step"]
            return {}
        if path == "/scope/api/lookup":
            if self.state.get("lookup_fails"):
                raise client.ServerError("POST /scope/api/lookup: HTTP 503: no next-token", 503, "no next-token")
            n = N_TOK - 1
            self.state["looked_up"] = body["trace"]
            tops = {} if self.state.get("no_tops") else dict(top_ids_on=[[0]] * n, top_ids_off=[[0]] * n)
            return dict(prompt_tokens=P_LEN, ngram=3, ids=list(range(n)), lp_on=[-0.1] * n, lp_off=[-0.1] * (n - 1) + [-2.0], **tops)
        assert path == "/scope/api/steer/words"
        self.state["made"].append(body["name"])
        return dict(name=body["name"], band=self.state.get("band", [18, 44]), cut=[])

    def request(self, method, path, body, timeout=60, headers=None, as_text=False):
        """The edit: its stream as the server sends it, and its two traces."""
        self.state["edit"] = headers[config.EDIT_HEADER]
        rid, at = headers[config.EDIT_HEADER].split()[0].split("@")
        at = int(at)
        self.state["write"]("", dict(edit=dict(of=rid, at=at), tokens=P_LEN * [0] + list(range(at)), last_token_text="NEW"), rid="ra")
        self.state["write"]("", dict(edit=dict(of="ra", at=at + 1)), rid="rb")
        kept_text = "".join(f"t{i}" for i in range(at)) if self.state["edit_keeps"] else "other"
        chunks = [dict(choices=[dict(delta=dict(role="assistant"))]), dict(choices=[dict(delta=dict(content=kept_text))]),
                  dict(choices=[dict(delta=dict(content="NEW more"), finish_reason="length")])]
        return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


@pytest.fixture
def server(monkeypatch):
    """A fake client.Session, a trace index written as the server would, and the server's concepts."""
    state = dict(concepts=concepts(dict(name="calm", reader=dict(layer=24), steer=dict(layer=24, strength=0.3, basis="chat"))),
                 steered_text="steered", steer_label="calm@0.3/24", partial=None, alive=True, sent=[], caps=["load_copy"],
                 moe=48, made=[], edit_keeps=True, stream="shares")

    def write_trace(steer, meta=None, rid=None):
        rid = rid or f"r{len(state['sent']) - 1}"
        d = config.home() / f"t{rid}"
        d.mkdir()
        meta = {**dict(request_id=rid, prompt_len=P_LEN, tokens=P_LEN * [0] + list(range(N_TOK - 1)),
                       token_text=P_LEN * ["p"] + [f"t{i}" for i in range(N_TOK - 1)]), **(meta or {})}
        (d / "meta.json").write_text(json.dumps(dict(meta, partial=state["partial"]) if state["partial"] else meta))
        if state["stream"]:                           # the trace records the writes ("lost": the feed sends none)
            with zipfile.ZipFile(d / "arrays.npz", "w") as z:
                z.writestr("resid_write_norm/0.npy", b"")
        entry = dict(id=d.name, path=str(d), time=time.time(), model="M", client=livecheck.CLIENT,
                     steer=steer or None, readout=True)
        config.index_path().parent.mkdir(parents=True, exist_ok=True)
        with open(config.index_path(), "a") as f:
            f.write(json.dumps(entry) + "\n")
    state["write"] = write_trace

    class FakeSession:
        def __init__(self, server, client_id, experts=False, lens=False, writes=False):
            assert experts and lens and writes            # the feed as the dock asks for it
            self.alive = state["alive"]

        def complete(self, model, prompt, steer, max_tokens, timeout=900, temperature=None):
            state["sent"].append((steer, temperature))
            write_trace(state["steer_label"] if steer.startswith("calm") else steer)
            rid = f"r{len(state['sent']) - 1}"
            text = state["steered_text"] if steer else "plain"
            evs = events(rid, bool(state["moe"]), state.get("lens_by", "jlens"), state["stream"])
            for i, said in state.get("said", {}).items():
                evs[i]["text"] = said
            return dict(text=text, seconds=0.1, tok_text=[e["text"] for e in evs], events=evs, rid=rid,
                        p=[e["p"] for e in evs])

        def close(self):
            pass

    monkeypatch.setattr(client, "Session", FakeSession)
    monkeypatch.setattr(livecheck, "saved_traces", lambda t0, model, n=1, wait=0: livecheck._traces_since(t0 - 1, model)[:n])
    return state


def check(state, capsys):
    found = {}
    ok = livecheck.check(FakeServer(state), found=found)
    return ok, capsys.readouterr().out, found


def test_a_working_backend_passes_every_finding(server, capsys):
    server["caps"] = ["edit_at", "lookup"]
    ok, out, found = check(server, capsys)
    assert ok and "✗" not in out, out
    assert "applied (calm@0.3/24)" in out and "differs" in out
    assert "lens layers: small 2" in out, "each size by its name"
    assert "J-lens rows on 12/12 tokens (2 layers), top 10s on 12" in out and "logit lens rows on 12/12 tokens (3 layers)" in out
    assert "routing on 12/12 tokens of the feed; 11 tokens x 48 layers x top 2" in out
    assert "  ✓ layer writes on 12/12 tokens of the feed (3 layers x 4 streams), shares beside them" in out
    assert "expert 7: 30 tokens" in out and "made a word vector for tiger" in out and server["made"] == ["tiger", "lion"]
    assert server["sent"] == [("", 0.0), ("calm@0.3/24", 0.0)]                 # a word changes one token: the edit's
    assert server["edit"] == "r0@8 remove~tiger/18-44"
    assert "one stream, the text before it kept" in out and "2 traces saved naming the edit; the reply's ids before it kept" in out
    assert "token 8: 't8' became 'NEW'" in out
    assert server["looked_up"] == "tr0" and "lookup read 11 reply tokens again" in out
    assert "+1.90   90% on  14% off  't7t8t9' -> 't10'" in out          # the one token it helped, after the three before
    assert all(livecheck.verdict(found.get(f, [])) == "passed" for f in livecheck.FEATURES), found


def test_existing_word_vectors_are_used_not_made_again_and_the_edit_goes_where_the_word_is_said(server, capsys):
    for w in ("lion", "Asia"):
        p = models.vector_path("M", w).with_suffix(".json")
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(layers=[0, 1], cut=[])))
    server.update(caps=["edit_at"], said={5: " lion"})
    ok, out, _ = check(server, capsys)
    assert ok and server["made"] == []
    assert server["edit"] == "r0@5 remove~lion/18-44" and "token 5: ' lion' became 'NEW'" in out


def test_a_server_left_with_the_lenses_off_is_checked_at_the_default_step_then_put_back(server, capsys):
    server.update(caps=["edit_at", "lookup"], at=config.LENS_OFF)
    ok, out, found = check(server, capsys)
    assert ok and "✗" not in out, out
    assert server["steps_set"] == [config.LENS_STEP_DEFAULT, config.LENS_OFF] and server["at"] == config.LENS_OFF
    assert "  · the lenses are off here: checked at medium, then off again" in out
    assert livecheck.verdict(found["lens rows"]) == "passed"


def test_no_j_lens_is_not_a_failure_and_no_word_vector_is_asked_for(server, capsys):
    server.update(caps=["edit_at"], lens_by="logit")      # DeepSeek V4.1 on ds4: the logit lens only
    ok, out, found = check(server, capsys)
    assert ok and "✗" not in out and server["made"] == [], out
    assert "  · no J-lens on this model: no word to edit with" in out and "no one-token word vector" not in out
    assert livecheck.verdict(found["one-token edit"]) == "passed"
    assert server["edit"] == "r0@8 calm@0.3/24"              # the edit with the concept's steer


def test_a_word_vector_refused_with_a_j_lens_still_fails(server, capsys, monkeypatch):
    def refuse(self, path, body, timeout=60):
        raise client.ServerError(f"POST {path}: HTTP 400: boom", 400, "boom")
    monkeypatch.setattr(FakeServer, "post", refuse)
    ok, out, _ = check(server, capsys)
    assert not ok and "  ✗ a word vector for tiger could not be made" in out


def test_the_edit_removes_the_word_over_the_servers_band(server, capsys):
    """Inkling 2's band: the last block's visits."""
    server.update(caps=["edit_at"], band=[42, 46])
    ok, out, found = check(server, capsys)
    assert ok and server["edit"] == "r0@8 remove~tiger/42-46" and livecheck.verdict(found["one-token edit"]) == "passed", out


@pytest.mark.parametrize("change, fails, says", [
    (dict(steer_label="calm@0.3/24 FAILED"), True, "  ✗ steered by"),
    (dict(steer_label="calm@0.3/24 PARTIAL"), True, "  ✗ steered by"),
    (dict(partial=["router hook: boom"]), True, "partial: router hook: boom"),
    (dict(alive=False), True, "the live feed (/scope/api/live) did not connect"),
    (dict(steered_text="plain"), False, "  ✓ steered by calm@0.3/24: applied (calm@0.3/24); the reply is the same"),
    (dict(concepts=concepts()), False, "  · no concept profiled for M"),
    (dict(concepts=concepts(dict(name="calm", reader=dict(layer=24)), steer_enabled=False)), False, "steering is off"),
    (dict(moe=0), False, "  · no experts (a dense model)"),
    (dict(stream=None), False, "  · no layer writes (a model that records none): RESIDUAL STREAM not checked"),
    (dict(stream="writes"), False, "  ✓ layer writes on 12/12 tokens of the feed (3 layers x 4 streams), no shares"),
    (dict(stream="lost"), True, "  ✗ layer writes on 0/12 tokens of the feed"),
    (dict(stream="some"), True, "  ✗ layer writes on 12/12 tokens of the feed in 2 shapes"),
    (dict(caps=["load_copy"]), False, "  · the server lists no edit_at"),
    (dict(caps=["edit_at"]), False, "  · no n-gram lookup on this model: not checked"),
    (dict(caps=["lookup"], lookup_fails=True), True, "  ✗ the lookup: POST /scope/api/lookup: HTTP 503"),
    (dict(caps=["lookup"], no_tops=True), True, "  ✗ lookup read 11 reply tokens again in"),
    (dict(caps=["edit_at"], edit_keeps=False), True, "  ✗ edit at token 8 (remove~tiger/18-44): one stream, the text before it changed"),
])
def test_what_fails_and_what_is_only_noted(server, capsys, change, fails, says):
    server.update(change)
    ok, out, _ = check(server, capsys)
    assert ok is not fails and says in out, out


def test_no_backend_serving_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(client, "serving", lambda: [])
    lines = []
    assert livecheck.run(log=lines.append) and lines[1:] == [
        "Live", "  · no backend serves a model: start oMLX with one, or metalscope ds4 serve MODEL.gguf"]


def test_two_backends_end_with_the_table(monkeypatch):
    class S:
        def __init__(self, engine):
            self.engine, self.url = engine, f"http://{engine}"

        def status(self):
            return dict(engine=self.engine)
    monkeypatch.setattr(client, "serving", lambda: [S("omlx"), S("ds4")])
    monkeypatch.setattr(livecheck, "check", lambda s, log, found: found.update({"traced reply": [s.engine == "ds4"]}) or True)
    lines = []
    results = []
    assert livecheck.run(log=lines.append, results=results) and [e for e, _ in results] == ["omlx", "ds4"]
    n = len(livecheck.FEATURES) - 1
    assert lines[:2] == [f"  omlx · 1 failed · {n} not checked", f"  ds4 · 1 passed · {n} not checked"]
    assert lines[3].split() == ["Live", "omlx", "ds4"] and lines[4].split() == ["traced", "reply", "failed", "passed"]

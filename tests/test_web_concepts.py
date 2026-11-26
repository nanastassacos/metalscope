"""The concepts and steering routes behind Pi's /steer and /scope."""

import dataclasses
import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from metalscope import config, middleware, models, vectors, web
from metalscope.omlx import backend as omlx_backend, patches as omlx_patches
from metalscope.live import MANAGER
from tests.helpers import scope_client, tiny_llama
from tests.helpers import Tok
from tests.helpers import fake_profile


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    yield scope_client(), tmp_path
    for key in list(MANAGER.tracers):
        MANAGER.tracers.pop(key)


def test_concepts_route_lists_profiles_and_steer_state(client):
    c, home = client
    fake_profile(home)
    config.save_settings(mode="on")
    MANAGER.attach(tiny_llama(), "tiny", Tok())
    MANAGER.norm_tables["tiny"] = {"layers": {1: 2.5}, "streams": {}}   # no saved traces here: stub the table
    r = c.get("/scope/api/concepts").json()
    assert r["model"] == "tiny" and r["steer_enabled"] is False
    assert r["concepts"][0]["name"] == "toy" and r["concepts"][0]["reader"]["layer"] == 1
    assert r["concepts"][0]["calibration"] is None
    assert r["concepts"][0]["steer"] is None                           # not calibrated: no default dose
    # The served model's chat calibration is passed through for Pi, only while the profile is the one it was made
    # from; another served copy's never is.
    prof = models.profile_path("tiny", "toy")
    cal = dict(version=4, concept="toy", layer=1, default_dose=0.3, good=16, turns=18, profile_mtime=prof.stat().st_mtime)
    models.link("tiny-ds4", "tiny")
    other = models.calibration_path("tiny-ds4", "toy")
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps(cal))
    assert c.get("/scope/api/concepts").json()["concepts"][0]["calibration"] is None
    mine = models.calibration_path("tiny", "toy")
    mine.parent.mkdir(parents=True)
    mine.write_text(json.dumps(cal))
    item = c.get("/scope/api/concepts").json()["concepts"][0]
    assert item["calibration"] == cal
    assert (item["steer"]["strength"], item["steer"]["basis"]) == (0.3, "chat")
    mine.write_text(json.dumps(dict(cal, version=3)))                  # an older calibration: ignored
    item = c.get("/scope/api/concepts").json()["concepts"][0]
    assert item["calibration"] is None and item["steer"] is None
    mine.write_text(json.dumps(cal))
    os.utime(prof, (1, 1))                                             # re-profiled: the vector may have moved
    assert c.get("/scope/api/concepts").json()["concepts"][0]["calibration"] is None
    for bad in ("not json", "[1, 2]"):
        mine.write_text(bad)
        assert c.get("/scope/api/concepts").json()["concepts"][0]["calibration"] is None
    # The replies route reads the served model's own conversations: one message plain and steered at the default.
    mine.write_text(json.dumps(dict(cal, profile_mtime=prof.stat().st_mtime)))
    row = lambda dose, answer: json.dumps(dict(arc="casual", dose=dose, turn=1, attempt=1, failed=None, user="Hey!", tool=False,
                                               answer=answer, text=f"t</think>{answer}"))
    models.replies_path("tiny", "toy").write_text("\n".join([row(0.0, "plain"), row(0.3, "steered")]))
    models.grades_path("tiny", "toy").write_text(json.dumps(dict(rounds=[dict(grades=[
        dict(arc="casual", dose=0.3, turn=1, kind="turn", style=2, coherence=3, broken=False)])])))
    sample = c.get("/scope/api/concepts/toy/replies").json()["sample"]
    assert (sample["steered"], sample["plain"], sample["prompt"], sample["good"], sample["style"]) == ("steered", "plain", "Hey!", True, 2)
    models.replies_path("tiny", "toy").unlink()
    models.replies_path("tiny-ds4", "toy").write_text("\n".join([row(0.0, "plain"), row(0.3, "steered")]))
    assert c.get("/scope/api/concepts/toy/replies").json()["sample"] is None          # another copy's replies: not these
    # Steering on and off from the API.
    assert c.post("/scope/api/steer", json=dict(enabled=True)).json() == dict(steer_enabled=True)
    assert config.load_settings()["steer"] is True
    assert c.post("/scope/api/steer", json=dict(enabled="yes")).status_code == 400
    assert c.post("/scope/api/steer", json=dict(enabled=False)).json() == dict(steer_enabled=False)


def test_words_route_needs_a_loaded_model_and_a_lens(client):
    c, home = client
    assert c.post("/scope/api/steer/words", json=dict(name="x", words=[])).status_code == 400
    assert c.post("/scope/api/steer/words", json=dict(model="tiny", name="../x", words=["soft"])).status_code == 400   # a file name
    assert c.post("/scope/api/steer/words", json=dict(model="tiny", name="x", words=["soft"])).status_code == 404
    config.save_settings(mode="on")
    MANAGER.attach(tiny_llama(), "tiny", Tok())
    # A concept's name: shipped, defined, or with texts on this model. Texts on another model only: a word vector.
    models.definition_path("defined").parent.mkdir(parents=True)
    models.definition_path("defined").write_text(json.dumps(dict(name="defined", description="d")))
    for model, name in (("tiny", "own"), ("other", "elsewhere")):
        d = models.concept_dir(model, name)
        d.mkdir(parents=True)
        (d / "concept.json").write_text("{}")
        (d / "pos.txt").write_text("a")
    for name in ("fluffy", "defined", "own"):
        r = c.post("/scope/api/steer/words", json=dict(model="tiny", name=name, words=["soft"]))
        assert r.status_code == 409 and r.json()["detail"].startswith(f"'{name}' is a concept: learn it"), name
    for name in ("x", "elsewhere"):
        r = c.post("/scope/api/steer/words", json=dict(model="tiny", name=name, words=["soft"]))
        assert r.status_code == 400 and "lens" in r.json()["detail"]                     # no Jacobian lens for tiny


def test_words_route_loads_no_lens_of_its_own_and_one_for_calls_at_once(client, monkeypatch):
    """Before the tracer has its lens (mode off, no traced request yet), the route installs it as a steered request
    would: two calls at once load the lens once, and never a lens of the call's own (jlens.load: gigabytes)."""
    import threading
    import time
    import mlx.core as mx
    from metalscope import jlens
    from metalscope.tracer import ModelTracer

    class Words(Tok):
        def encode(self, text, add_special_tokens=False):
            return [len(text.strip())]

    c, home = client
    config.save_settings(mode="off")
    tracer = MANAGER.attach(tiny_llama(), "tiny", Words())
    assert tracer.lens is None and not tracer.installed
    loads = []

    def load(self):
        loads.append(threading.get_ident())
        time.sleep(0.2)                                                # the other call arrives meanwhile
        jac = {l: mx.eye(64) for l in range(2)}
        mx.eval(list(jac.values()))                                    # no lazy arrays across threads
        return jlens.LensSetup(jlens.Lens(jac), name="tiny-lens")
    monkeypatch.setattr(ModelTracer, "_load_lens", load)
    monkeypatch.setattr(jlens, "load", lambda *a: pytest.fail("a lens loaded for the call"))
    out = {}
    calls = [threading.Thread(target=lambda n=n: out.update({n: web.steer_from_words(dict(model="tiny", name=n, words=["soft"]))}))
             for n in ("a", "b")]
    for t in calls:
        t.start()
    for t in calls:
        t.join()
    assert len(loads) == 1 and tracer.installed and out["a"]["layers"] == out["b"]["layers"] == [0, 1]
    assert c.post("/scope/api/steer/words", json=dict(model="tiny", name="c", words=["soft"])).status_code == 200
    assert len(loads) == 1


def test_words_route_gives_the_workspace_band_and_the_words_cut_to_a_token(client):
    """A removal's layers (jlens.workspace_band) and whether a word is one token: the dock's LENS edit needs both."""
    import mlx.core as mx
    from types import SimpleNamespace
    from metalscope import jlens

    class Pieces(Tok):
        def encode(self, text, add_special_tokens=False):          # a word of over 6 letters is two tokens
            w = text.strip()
            return [len(w)] if len(w) <= 6 else [7, 8]

    c, home = client
    tracer = MANAGER.attach(tiny_llama(), "tiny", Pieces())
    tracer.lens = SimpleNamespace(lens=jlens.Lens({l: mx.eye(64) for l in range(2)}))
    mx.eval(list(tracer.lens.lens.jacobians.values()))                                  # no lazy arrays across threads
    r = c.post("/scope/api/steer/words", json=dict(model="tiny", name="soccer", words=["soccer"])).json()
    assert r["band"] == [1, 1] and r["cut"] == [] and "suggested_layer" not in r
    assert c.post("/scope/api/steer/words", json=dict(model="tiny", name="pineapple", words=["pineapple"])).json()["cut"] == ["pineapple"]
    assert jlens.workspace_band(48) == (18, 44)                                         # Flash Next: the tested band
    # ids: the tokens themselves (the dock's eyepiece), each its word; no first piece after a space, so never cut.
    post = lambda **kw: c.post("/scope/api/steer/words", json=dict(model="tiny", name="pineapple", **kw))
    assert post(words=[" <9>"], ids=[9]).json()["cut"] == []
    assert vectors.load_vector("tiny", "pineapple").meta["token_ids"] == [9]
    r = post(words=["<9>"], ids=[8])
    assert r.status_code == 400 and "8 is not '<9>'" in r.json()["detail"]
    for bad in ([9, 9], ["9"], [-1], [True], 9):
        assert post(words=["<9>"], ids=bad).status_code == 400, bad


def test_a_lens_word_that_is_no_name_goes_by_its_token_id(client):
    """The dock's edit of a lens word such as 初步: the vector named after its id (tok9: the header and the file
    name stay ASCII), made from the token and checked against its text; the edit's header then parses and is checked
    as any word's."""
    import mlx.core as mx
    from types import SimpleNamespace
    from metalscope import jlens, steer

    class Cjk(Tok):
        def decode(self, ids):
            return "初步" if list(ids) == [9] else super().decode(ids)

    c, home = client
    tracer = MANAGER.attach(tiny_llama(), "tiny", Cjk())
    tracer.lens = SimpleNamespace(lens=jlens.Lens({l: mx.eye(64) for l in range(2)}))
    mx.eval(list(tracer.lens.lens.jacobians.values()))
    post = lambda name, **kw: c.post("/scope/api/steer/words", json=dict(model="tiny", name=name, **kw))
    assert post("初步", words=["初步"], ids=[9]).status_code == 400                       # the name itself is refused
    r = post("tok9", words=["初步"], ids=[9])
    assert r.status_code == 200 and r.json()["name"] == "tok9" and r.json()["band"] == [1, 1]
    v = vectors.load_vector("tiny", "tok9")
    assert (v.meta["words"], v.meta["token_ids"], v.meta["cut"]) == (["初步"], [9], []) and "初步" in v.meta["source"]
    assert vectors.find_vector("tiny", "tok9").relative_to(home).as_posix().isascii()
    assert post("tok8", words=["初步"], ids=[8]).status_code == 400                       # the id must be that text
    e = steer.parse_edit("3f2a@5 remove~tok9/1+vague@0.7/1", "tiny")
    assert e["steer"].label == "remove~tok9/1+vague@0.7"
    removal = steer.parse_header("remove~tok9/1")
    steer.check_spec(dataclasses.replace(removal, edit=True), "tiny")                 # as the edit's own steer it passes
    with pytest.raises(ValueError, match="one token"):                                # the same header as a steer does not
        steer.check_spec(removal, "tiny")
    assert sorted(steer.build_plan(steer.parse_header("remove~tok9/1"), "tiny", "r", table={"layers": {}}).remove) == [1]


def test_routes_answer_under_scope_only(client):
    c, home = client
    fake_profile(home)
    config.save_settings(mode="on")
    MANAGER.attach(tiny_llama(), "tiny", Tok())
    MANAGER.norm_tables["tiny"] = {"layers": {1: 2.5}, "streams": {}}
    new = c.get("/scope/api/concepts")
    assert new.status_code == 200 and new.json()["concepts"][0]["name"] == "toy"
    assert c.get("/scope").status_code == 404                                     # no page


def test_mount_puts_the_routes_ahead_of_a_catch_all(tmp_path, monkeypatch):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    app = FastAPI()

    @app.get("/{path:path}")
    def catch_all(path: str):
        return {"catch_all": path}

    web.mount(app, omlx_backend.BACKEND)
    app.dependency_overrides[middleware._local_only] = lambda: None
    c = TestClient(app)
    assert "catch_all" not in c.get("/scope/api/concepts").json()
    assert c.get("/elsewhere").json() == {"catch_all": "elsewhere"}


def test_concept_calibrate_route_says_400_for_bad_input_and_409_when_it_cannot_start(client):
    c, home = client
    for body, status, why in ((dict(name="", model="tiny"), 400, "name the concept"),
                              (dict(name="nope", model="tiny"), 409, "nope has no profile for tiny")):
        r = c.post("/scope/api/concept/calibrate", json=body)
        assert r.status_code == status, (body, r.text)
        assert why in r.json()["detail"], (body, r.text)
    assert c.get("/scope/api/jobs").json()["jobs"] == []


def test_a_concept_with_malformed_files_is_an_error_on_its_item_not_a_500(client):
    """A pulled pack's profile.json or calibration.json is not checked field by field: one bad field must not take
    Pi's whole concept list or Steer page down."""
    c, home = client
    fake_profile(home)
    MANAGER.attach(tiny_llama(), "tiny", Tok())
    for name, bad in (("reader", dict(reader="x")), ("gen", dict(generation=5)), ("listed", None), ("named", dict(concept=7))):
        _, prof = fake_profile(home, name=name)
        p = models.profile_path("tiny", name)
        p.write_text(json.dumps([1, 2] if bad is None else dict(prof, **bad)))
    cal = models.calibration_path("tiny", "reader")
    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_text(json.dumps(dict(version=4, concept="reader", layer=1, default_dose="x", good="many", turns=18,
                                   strengths=[1, "a"], plain=3, profile_mtime=models.profile_path("tiny", "reader").stat().st_mtime)))
    lib = home / "concepts" / "junk"                    # a definition that is not one
    lib.mkdir(parents=True)
    (lib / "concept.json").write_text("[1]")
    r = c.get("/scope/api/concepts")
    assert r.status_code == 200, r.text
    items = {i["name"]: i for i in r.json()["concepts"]}
    assert "error" not in items["toy"] and items["toy"]["reader"]["layer"] == 1
    assert all(items[n].get("error") for n in ("reader", "gen", "listed")), items
    r = c.get("/scope/api/steering")
    assert r.status_code == 200, r.text
    items = {i["name"]: i for i in r.json()["concepts"]}
    assert "error" not in items["toy"] and items["reader"].get("error") and items["listed"].get("error")


def test_names_that_become_paths_or_arguments_are_checked(client, monkeypatch):
    """A model or concept name a request gives becomes a path under the data home or a word on a job's command line:
    ``..`` must not reach checkpoint.json beside the home, nor ``--fresh`` the CLI as an option. 400, nothing run."""
    from metalscope.learn import spawn
    c, home = client
    monkeypatch.setattr(spawn, "_launch", lambda *a, **k: pytest.fail("a job started"))
    for bad in ("..", "--fresh", ".hidden", "a b"):
        for r in (c.get("/scope/api/concepts", params=dict(model=bad)),
                  c.get("/scope/api/steering", params=dict(model=bad)),
                  c.get("/scope/api/lens", params=dict(model=bad)),
                  c.get("/scope/api/lens/layers", params=dict(model=bad)),
                  c.get("/scope/api/experts/grid", params=dict(model=bad)),
                  c.post("/scope/api/lens/use", json=dict(model=bad, name="logit")),
                  c.get("/scope/api/lens/learn", params=dict(model=bad, until="07:30")),
                  c.post("/scope/api/lens/learn", json=dict(model=bad, until="07:30")),
                  c.post("/scope/api/concept/calibrate", json=dict(name="x", model=bad))):
            assert r.status_code == 400 and "bad model name" in r.json()["detail"], (bad, r.request.url, r.text)
    for bad in (".hidden", "-x", "%2E%2E"):
        for path in (f"/scope/api/concepts/{bad}/replies", f"/scope/api/steering/{bad}/chats"):
            r = c.get(path, params=dict(model="tiny"))
            assert r.status_code == 400 and "bad concept name" in r.json()["detail"], (path, r.text)
    assert not (home.parent / "checkpoint.json").exists() and not (home / "checkpoint.json").exists()
    assert c.get("/scope/api/concepts", params=dict(model="org/Qwen3.5-35B-A3B-4bit")).status_code == 200   # a path's last part
    listed = scope_client(web.Backend("omlx", frozenset({"load_copy"}), lambda: {"m": {}}))   # the lens fit, as the other jobs
    for r in (listed.get("/scope/api/lens/learn", params=dict(model="other", until="07:30")),
              listed.post("/scope/api/lens/learn", json=dict(model="other", until="07:30"))):
        assert r.status_code == 409 and "other is not being served" in r.json()["detail"]


def test_jobs_route_lists_every_kind_and_marks_one_seen(client):
    """/scope/api/jobs (was /scope/api/concept/jobs): the newest job in full, the last ten briefly, the GPU holder."""
    from metalscope.learn import state
    c, home = client
    for i, (kind, extra) in enumerate([("learn", dict(result=dict(steerable=False, reason="can't be steered: x"))),
                                       ("calibrate", dict(result=dict(calibration=dict(version=4, default_dose=0.3, good=15, turns=18, strengths=[]))))]):
        d = state.job_dir(f"j-{i}")
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps(dict(job=f"j-{i}", kind=kind, name=f"n{i}", phase="done", outcome="done",
                                                      started=100 + i, pid=1, **extra)))
    r = c.get("/scope/api/jobs").json()
    assert r["latest"]["job"] == "j-1" and r["latest"]["kind"] == "calibrate" and r["gpu"] is None
    assert [(j["kind"], j["steerable"]) for j in r["jobs"]] == [("learn", False), ("calibrate", None)]
    assert not any("decision" in j for j in r["jobs"])
    assert r["jobs"][1]["calibration"] == dict(default_dose=0.3, good=15, turns=18)
    assert c.post("/scope/api/jobs/j-0/seen").json() == dict(seen=True) and state.read_state("j-0")["seen"]
    assert c.post("/scope/api/jobs/j-1/cancel").json() == dict(cancelled=False)        # done: nothing to stop
    assert c.post("/scope/api/jobs/bad%20id/cancel").status_code == 400
    assert c.get("/scope/api/concept/jobs").status_code in (404, 405)                    # a clean rename: no old route


def test_status_says_what_the_omlx_backend_can_do_and_each_models_thinking(client, monkeypatch):
    from types import SimpleNamespace
    c, home = client
    monkeypatch.setattr(omlx_patches, "APPLIED", set())                  # no oMLX patched in this process
    qwen, plain, named = home / "qwen", home / "plain", home / "named"
    for d in (qwen, plain, named):
        d.mkdir()
    (qwen / "chat_template.jinja").write_text("{% if enable_thinking %}<think>{% endif %}")
    (plain / "tokenizer_config.json").write_text(json.dumps(dict(chat_template="{{ messages }}")))
    (named / "tokenizer_config.json").write_text(json.dumps(dict(chat_template=[dict(name="default", template="<think>")])))
    # A model's family is oMLX's config_model_type; loaded, its scheduler says whether oMLX parses its replies.
    sched = lambda parser: SimpleNamespace(_engine=SimpleNamespace(engine=SimpleNamespace(scheduler=SimpleNamespace(_output_parser_factory=parser))))
    entries = {"Qwen-X": SimpleNamespace(model_path=str(qwen), config_model_type="qwen3_5_moe", engine=sched(None)),
               "Plain": SimpleNamespace(model_path=str(plain), config_model_type="llama"),
               "R1": SimpleNamespace(model_path=str(named), config_model_type="deepseek_v4", engine=sched(object())),
               "remote": SimpleNamespace(model_path=None)}
    pool = SimpleNamespace(get_model_ids=lambda: list(entries), get_entry=entries.get)
    monkeypatch.setattr(omlx_backend, "_SERVER", SimpleNamespace(get_server_state=lambda: SimpleNamespace(engine_pool=pool, api_key=None)))
    from metalscope import web
    monkeypatch.setattr(web, "_BOOT", web.code_stamp())                  # the code it runs is the code installed
    st = c.get("/scope/api/status").json()
    assert (st["engine"], st["caps"]) == ("omlx", ["load_copy"])      # edit_at once oMLX is patched: test_omlx_patches
    assert st["stale"] is None
    assert st["chat"] == {"Qwen-X": dict(reasoning=True, thinking_format="qwen-chat-template", tested=True, edit=False),
                          "Plain": dict(reasoning=False, thinking_format=None, tested=False, edit=False),
                          "R1": dict(reasoning=True, thinking_format=None, tested=False, edit=False),   # always thinks: no switch
                          "remote": dict(reasoning=False, thinking_format=None, tested=False, edit=False)}
    # Patched for the edit: a loaded model takes it unless oMLX parses its replies with a parser the edit does not
    # seed (the predicate its refusal uses,
    # patches._refuses_edit); one not loaded has no parser decided yet, so no `edit` (Pi asks again once a reply starts).
    monkeypatch.setattr(omlx_patches, "APPLIED", {"patch_engine_core", "patch_scheduler"})
    monkeypatch.setattr(omlx_patches, "_DETOKENIZERS", True)
    st = c.get("/scope/api/status").json()
    assert "edit_at" in st["caps"]
    assert {m: (x["tested"], x.get("edit")) for m, x in st["chat"].items()} == {
        "Qwen-X": (True, True), "Plain": (False, None), "R1": (False, False), "remote": (False, None)}
    monkeypatch.setattr(omlx_backend, "_SERVER", None)                   # no engine pool: no thinking controls, still a status
    assert c.get("/scope/api/status").json()["chat"] == {}


def test_concept_learn_picks_its_engine_by_capability(client, monkeypatch):
    """oMLX's route starts a learn job on its own copy (load_copy), the ds4 proxy's on the served model (test_ds4); a
    server with neither refuses, saying why."""
    from metalscope.learn import spawn
    started = []
    monkeypatch.setattr(spawn, "learn", lambda *a, **k: started.append(a + tuple(k.values())) or dict(job="x-1", pid=1))
    c, _home = client
    r = c.post("/scope/api/concept/learn", json=dict(name="x", description="y", model="m"))
    assert r.status_code == 200 and started == [("x", "y", "m", None, "mlx", None, None)]       # oMLX's URL from its settings
    r = c.post("/scope/api/concept/learn", json={"name": "x", "model": "m", "corpus": "/c/x"})   # Pi's flag
    assert r.status_code == 200 and started.pop() == ("x", "", "m", None, "mlx", None, "/c/x")
    bare = scope_client(web.Backend("other", frozenset({"token_prompts"})))
    r = bare.post("/scope/api/concept/learn", json=dict(name="x", description="y", model="m"))
    assert r.status_code == 409 and "not available on this server" in r.json()["detail"] and len(started) == 1
    # No traced replies of m yet: refused in the plan, before anything is unloaded, saying what to do.
    plan = c.get("/scope/api/concept/learn", params=dict(name="fluffy", model="m")).json()
    assert plan["source"] == "shipped" and plan["taken"].startswith("m has no traced replies yet")
    assert "chat a few unsteered messages with m first" in plan["taken"]
    from tests.helpers import write_norms
    write_norms("m", [0, 1])
    # What /concept learn NAME would do, for Pi's confirm: a shipped concept's texts, copied for the model.
    plan = c.get("/scope/api/concept/learn", params=dict(name="fluffy", model="m")).json()
    assert plan == dict(name="fluffy", model="m", source="shipped", taken=None,          # the texts themselves stay out
                        texts="texts: hand-written by Claude Fable 5.1 (2026-10-05)", notes=[], corpus=None, engine=None,
                        description=plan["description"], writer="Claude Fable 5.1 (by hand)")
    assert plan["description"].startswith("Soft, fluffy, furry textures")
    plan = c.get("/scope/api/concept/learn", params=dict(name="list", model="m")).json()
    assert "command word" in plan["taken"]
    assert c.get("/scope/api/concept/learn", params=dict(name="toasty", model="m")).json()["taken"] is None
    # A defined name: the served model writes, and the confirm gets the stored description.
    models.definition_path("toasty").parent.mkdir(parents=True)
    models.definition_path("toasty").write_text(json.dumps(dict(name="toasty", description="Toasty things.")))
    plan = c.get("/scope/api/concept/learn", params=dict(name="toasty", model="m")).json()
    assert (plan["source"], plan["description"], plan["texts"], plan["taken"]) == ("write", "Toasty things.", None, None)
    models.definition_path("toasty").unlink()
    plan = c.get("/scope/api/concept/learn", params=dict(name="toasty", model="m", description=" ")).json()
    assert plan["taken"].startswith("toasty needs a description (--description)")      # as the start says
    # --corpus DIR: its texts are checked here, for the confirm text, for a library name too.
    plan = c.get("/scope/api/concept/learn", params=dict(name="kept", model="m", corpus=str(_home / "none"))).json()
    assert plan["source"] == "corpus" and "missing concept.json" in plan["taken"]
    plan = c.get("/scope/api/concept/learn", params=dict(name="fluffy", model="m", corpus=str(_home / "none"))).json()
    assert plan["source"] == "corpus" and "missing concept.json" in plan["taken"]


def test_concept_job_routes_refuse_a_model_this_server_does_not_serve(client, monkeypatch):
    """Pi's /concept learn and calibrate name the model: one this server doesn't list is a 409 saying how to start it
    (a server that can't list its models lets the name through, as the tests above show)."""
    from types import SimpleNamespace
    from metalscope.learn import spawn
    c, home = client
    pool = SimpleNamespace(get_model_ids=lambda: ["Qwen-X"], get_entry=lambda mid: SimpleNamespace(model_path=None))
    monkeypatch.setattr(omlx_backend, "_SERVER", SimpleNamespace(get_server_state=lambda: SimpleNamespace(engine_pool=pool, api_key=None)))
    monkeypatch.setattr(spawn, "learn", lambda *a, **k: dict(job="x-1", pid=1))
    monkeypatch.setattr(spawn, "calibrate", lambda *a, **k: dict(job="c-1", pid=1))
    said = "gone is not being served: start it with `metalscope omlx gone` or `metalscope ds4 gone`"
    for r in (c.get("/scope/api/concept/learn", params=dict(name="fluffy", model="gone")),
              c.post("/scope/api/concept/learn", json=dict(name="fluffy", model="gone")),
              c.post("/scope/api/concept/calibrate", json=dict(name="fluffy", model="gone"))):
        assert r.status_code == 409 and r.json()["detail"] == said, r.text
    assert c.post("/scope/api/concept/learn", json=dict(name="fluffy", model="Qwen-X")).json()["job"] == "x-1"
    assert c.post("/scope/api/concept/calibrate", json=dict(name="fluffy", model="Qwen-X")).json()["job"] == "c-1"


def test_the_concepts_route_says_who_wrote_each_concepts_texts(client):
    c, home = client
    fake_profile(home)
    MANAGER.attach(tiny_llama(), "tiny", Tok())
    assert c.get("/scope/api/concepts").json()["concepts"][0]["texts"] is None          # toy: no texts on tiny
    d = models.concept_dir("tiny", "toy")                                               # its texts, beside its vector
    (d / "pos.txt").write_text("a\n")
    (d / "concept.json").write_text(json.dumps(dict(title="Toy", learned=dict(writer="Qwen3.6", request="toys", date="2026-10-27"))))
    (d / "writing.json").write_text(json.dumps(dict(writer="Qwen3.6", date="2026-10-27", request="toys", prompts=["Write."])))
    assert c.get("/scope/api/concepts").json()["concepts"][0]["texts"] == "texts: written by Qwen3.6 (2026-10-27)"   # no request
    assert "bar" not in c.get("/scope/api/concepts").json()["concepts"][0]


def test_the_delete_route_takes_a_description_no_model_has(client):
    c, home = client
    MANAGER.attach(tiny_llama(), "tiny", Tok())
    models.definition_path("toy").parent.mkdir(parents=True)
    models.definition_path("toy").write_text(json.dumps(dict(name="toy", description="t")))
    models.concept_dir("other", "toy").mkdir(parents=True)
    r = c.post("/scope/api/concept/delete", json=dict(name="toy", model="tiny"))
    assert r.status_code == 404 and r.json()["detail"] == "toy has nothing on tiny (it is on other)"
    models.concept_dir("other", "toy").rmdir()
    r = c.post("/scope/api/concept/delete", json=dict(name="toy", model="tiny"))
    assert r.status_code == 200 and r.json()["definition"] is True and r.json()["trash"].endswith("-toy-definition")
    assert not models.definition_path("toy").exists()
    assert "toy" not in {x["name"] for x in c.get("/scope/api/concepts").json()["library"]}


def test_the_delete_route_moves_one_checkpoints_folder_to_the_trash(client, monkeypatch):
    from metalscope.learn import state
    c, home = client
    fake_profile(home)
    (models.concept_dir("tiny", "toy") / "pos.txt").write_text("a")         # its texts: they go with it
    MANAGER.attach(tiny_llama(), "tiny", Tok())
    assert len(c.get("/scope/api/concepts").json()["concepts"]) == 1
    monkeypatch.setattr(state, "busy_with", lambda name: f"a job is running: calibrating {name} (/scope jobs)")
    r = c.post("/scope/api/concept/delete", json=dict(name="toy", model="tiny"))
    assert r.status_code == 409 and r.json()["detail"] == "a job is running: calibrating toy (/scope jobs)"
    monkeypatch.setattr(state, "busy_with", lambda name: None)
    assert c.post("/scope/api/concept/delete", json=dict(name="../x", model="tiny")).status_code == 400
    # Names that are not one folder: refused, nothing moved (not the concept's texts, not every concept).
    before = sorted(str(p) for p in home.rglob("*"))
    for name, model in [("toy", "."), ("toy", ".."), ("toy", "x/.."), ("..", "tiny"), (".", "tiny"), ("../models", "tiny")]:
        r = c.post("/scope/api/concept/delete", json=dict(name=name, model=model))
        assert r.status_code == 400 and "bad" in r.json()["detail"], (name, model, r.json())
    assert sorted(str(p) for p in home.rglob("*")) == before
    r = c.post("/scope/api/concept/delete", json=dict(name="toy", model="tiny"))
    assert r.status_code == 200 and r.json()["trash"].startswith(str(home / "trash")) and r.json()["trash"].endswith("-toy-tiny")
    assert r.json()["checkpoint"] == "tiny"
    assert c.get("/scope/api/concepts").json()["concepts"] == [] and (Path(r.json()["trash"]) / "pos.txt").is_file()
    assert [x["name"] for x in c.get("/scope/api/steering", params=dict(model="tiny")).json()["concepts"]] == []
    r = c.post("/scope/api/concept/delete", json=dict(name="toy", model="tiny"))
    assert r.status_code == 404 and r.json()["detail"] == "toy has nothing on tiny"


def test_the_concepts_route_lists_the_library_not_learned_on_the_model(client):
    from metalscope.concepts.corpus import DATA_DIR
    c, home = client
    MANAGER.attach(tiny_llama(), "tiny", Tok())
    lib = {x["name"]: x for x in c.get("/scope/api/concepts").json()["library"]}
    shipped = {d.name for d in DATA_DIR.iterdir() if (d / "task.json").is_file()}
    assert set(lib) == shipped                                            # every shipped concept with its texts
    e = lib["energized"]
    assert e["title"] == "Energized" and e["writer"] == "Claude Fable 5.1 (by hand)"
    assert e["description"].startswith("Feeling energized: ") and e["description"].endswith(".") and ". " not in e["description"]
    assert e["texts"] == "texts: hand-written by Claude Fable 5.1 (2026-10-06)"
    fake_profile(home, name="energized")                                  # learned on this model: listed under concepts instead
    r = c.get("/scope/api/concepts").json()
    assert "energized" not in {x["name"] for x in r["library"]} and [x["name"] for x in r["concepts"]] == ["energized"]
    # One learn would refuse is not offered: a steering vector of that name on the checkpoint, no profile (measured
    # before learn jobs).
    from metalscope import vectors
    import numpy as np
    vectors.save_vector("tiny", vectors.Vector("calm", {1: vectors.unit(np.ones(8))}, dict(source="diff of means")))
    assert "calm" not in {x["name"] for x in c.get("/scope/api/concepts").json()["library"]}
    # A definition (learned on another model): offered, the served model writes its own texts.
    models.definition_path("half").parent.mkdir(parents=True)
    models.definition_path("half").write_text(json.dumps(dict(name="half", description="Half a thing. More.")))
    half = {x["name"]: x for x in c.get("/scope/api/concepts").json()["library"]}["half"]
    assert half == dict(name="half", title="half", description="Half a thing.", writer=None,
                        texts="texts: each model writes its own")
    # Texts of the model's own with no profile yet (a killed run): offered, with who wrote them.
    d = models.concept_dir("tiny", "own")
    d.mkdir(parents=True)
    (d / "concept.json").write_text(json.dumps(dict(title="Own", description="Mine.", learned=dict(writer="tiny", date="2026-11-20"))))
    (d / "pos.txt").write_text("a\n")
    own = {x["name"]: x for x in c.get("/scope/api/concepts").json()["library"]}["own"]
    assert (own["writer"], own["texts"]) == ("tiny", "texts: written by tiny (2026-11-20) (instructions not kept)")
    # Another model's own texts are not this model's library.
    d = models.concept_dir("other", "theirs")
    d.mkdir(parents=True)
    (d / "concept.json").write_text("{}")
    (d / "pos.txt").write_text("a\n")
    assert "theirs" not in {x["name"] for x in c.get("/scope/api/concepts").json()["library"]}


def test_steering_route_lists_a_profile_graded_before_the_ladder_grades_went(client):
    """A profile whose dose ladder was graded (generation.grades) and split (split) lists as any other: its name, title
    and description, no ladder; a broken one with its error, not a 500."""
    c, home = client
    _, prof = fake_profile(home)
    p = models.profile_path("tiny", "toy")
    prof["generation"]["grades"] = dict(grader=dict(model="judge", engine="external", thinking=False), rubric=1, made="t",
                                        seconds=1.0, items=[dict(item=0, strength=0.5, style=3, coherence=2, note="soft")])
    prof["split"] = dict(layer=1, strength=0.5, readable_share=0.4)
    p.write_text(json.dumps(prof))
    r = c.get("/scope/api/steering", params=dict(model="tiny")).json()
    assert r["checkpoint"] == "tiny" and "checkpoints" not in r
    (toy,) = r["concepts"]
    assert set(toy) == {"name", "title", "description", "chat", "chat_stale"} and toy["chat"] is None
    p.write_text("{not json")
    assert "error" in c.get("/scope/api/steering", params=dict(model="tiny")).json()["concepts"][0]


def _calibrated(served, name, prof_path, dose=0.4, good=14):
    """A chat calibration of ``name`` on ``served`` with its replies and grades: plain and ``dose``, every turn good
    at ``dose`` but casual 2 (retried after a tool call) and casual 3 (graded broken), so casual 4-6 come after a break."""
    from metalscope.learn import calibrate as K
    d = models.calibration_dir(served, name)
    d.mkdir(parents=True)
    rows, grades = [], []
    for x in (0.0, dose):
        for arc, users in K.ARCS.items():
            for t, user in enumerate(users, 1):
                if x and (arc, t) == ("casual", 2):
                    rows.append(dict(arc=arc, dose=x, turn=t, attempt=1, user=user, failed="tool", text="hm</think><tool_call>"))
                rows.append(dict(arc=arc, dose=x, turn=t, attempt=2 if x and (arc, t) == ("casual", 2) else 1, user=user,
                                 failed=None, text=f"why</think>reply {arc} {t} at {x}<|im_end|>"))
                grades.append(dict(arc=arc, dose=x, turn=t, kind="turn", style=3 if x else 0, coherence=3, useful=3,
                                   broken=bool(x) and (arc, t) == ("casual", 3), why="cut off" if x and (arc, t) == ("casual", 3) else ""))
    (d / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "grades.json").write_text(json.dumps(dict(rounds=[dict(grades=[]), dict(grades=grades)])))
    strength = dict(dose=dose, good=good, turns=18, attempts=19, failed=dict(tool=1, thinking=0, short=0), coherence=3.0, graded=18)
    plain = dict(strength, dose=0.0, good=0, attempts=18, failed=dict(tool=0, thinking=0, short=0))
    (d / "calibration.json").write_text(json.dumps(dict(version=4, concept=name, layer=1, default_dose=dose, good=good,
                                                        turns=18, plain=plain, strengths=[strength],
                                                        profile_mtime=prof_path.stat().st_mtime)))


def test_steering_route_ranks_concepts_by_their_chat_calibration(client):
    c, home = client
    fake_profile(home)
    fake_profile(home, name="soft")
    fake_profile(home, name="zed")
    models.link("tiny-ds4", "tiny")
    _calibrated("tiny-ds4", "soft", models.profile_path("tiny", "soft"), dose=0.6, good=11)
    _calibrated("tiny-ds4", "zed", models.profile_path("tiny", "zed"))
    r = c.get("/scope/api/steering", params=dict(model="tiny")).json()
    assert r["served"] == "tiny-ds4"                                              # the copy with calibrations
    assert [x["name"] for x in r["concepts"]] == ["zed", "soft", "toy"]          # calibrated by good replies, then the rest
    zed, soft, toy = r["concepts"]
    assert (zed["chat"]["good"], zed["chat"]["default_dose"]) == (14, 0.4) and zed["chat"]["strengths"][0]["failed"]["tool"] == 1
    assert (soft["chat"]["good"], soft["chat"]["default_dose"]) == (11, 0.6)
    # Each strength's mean style, from the last grading (the calibration file has none).
    assert zed["chat"]["strengths"][0]["style"] == 3.0 and zed["chat"]["plain"]["style"] == 0.0 and zed["chat"]["plain"]["coherence"] == 3.0
    assert "style" not in json.loads(models.calibration_path("tiny-ds4", "zed").read_text())["strengths"][0]   # only read
    assert toy["chat"] is None and toy["chat_stale"] is False
    os.utime(models.profile_path("tiny", "zed"), (1, 1))                        # re-profiled: not calibrated, stale
    zed = next(x for x in c.get("/scope/api/steering", params=dict(model="tiny-ds4")).json()["concepts"] if x["name"] == "zed")
    assert zed["chat"] is None and zed["chat_stale"] is True


def test_steering_chats_route_gives_each_turn_plain_and_steered_with_why_not(client):
    c, home = client
    fake_profile(home)
    models.link("tiny-ds4", "tiny")
    _calibrated("tiny-ds4", "toy", models.profile_path("tiny", "toy"))
    r = c.get("/scope/api/steering/toy/chats", params=dict(model="tiny-ds4")).json()
    assert r["dose"] == 0.4 and r["default_dose"] == 0.4 and [v["arc"] for v in r["conversations"]] == ["casual", "dinner", "explain"]
    casual = r["conversations"][0]
    assert [v["title"] for v in r["conversations"]] == ["A casual chat", "Planning a dinner party", "How a fridge works"]
    from metalscope.learn import calibrate as K
    assert set(K.ARC_TITLES) == set(K.ARCS)
    assert casual["good"] == dict(plain=0, steered=1) and r["conversations"][1]["good"]["steered"] == 6
    t1, t2, t3, t4 = casual["turns"][:4]
    assert t1["user"] == "Hey! How's it going?" and t1["steered"]["good"] and t1["steered"]["why_not"] == []
    assert t1["steered"]["text"] == "<think>why</think>reply casual 1 at 0.4"           # the end token cut, thinking opened
    assert t1["plain"]["why_not"] == ["style too low"] and t1["plain"]["grade"]["style"] == 0
    assert t2["steered"]["why_not"] == ["retried"] and t2["steered"]["attempt"] == 2
    assert t2["steered"]["earlier"] == [dict(attempt=1, failed="tool", text="<think>hm</think><tool_call>")]
    assert t3["steered"]["why_not"] == ["broken"] and t3["steered"]["grade"]["why"] == "cut off"
    assert t4["steered"]["why_not"] == ["after a break"]
    assert c.get("/scope/api/steering/toy/chats", params=dict(model="tiny-ds4", dose=0.8)).status_code == 404   # not tried
    assert c.get("/scope/api/steering/nope/chats", params=dict(model="tiny-ds4")).status_code == 404


def test_a_lens_size_set_where_another_setting_is_invalid_answers_400(client):
    """settings.json edited by hand (a trace budget no number): the lens POST says why, as a 400, not a 500."""
    import json
    c, home = client
    MANAGER.attach(tiny_llama(), "tiny", Tok())
    config.save_settings(mode="on")
    p = config.settings_path()
    p.write_text(json.dumps(dict(json.loads(p.read_text()), trace_budget="lots")))
    r = c.post("/scope/api/lens/layers", json=dict(model="tiny", step=0))
    assert r.status_code == 400 and "trace_budget" in r.json()["detail"]


def test_lens_layers_route_answers_409_without_a_loaded_model(client):
    """409, not 404: Pi reads a 404 as a server without the route (not restarted since) and shows nothing."""
    c, home = client
    for r in (c.get("/scope/api/lens/layers", params=dict(model="tiny")), c.post("/scope/api/lens/layers", json=dict(model="tiny", step=1))):
        assert r.status_code == 409 and "not loaded" in r.json()["detail"]
    assert c.get("/scope/api/lens/layers").json()["detail"] == "no model is loaded"
    MANAGER.attach(tiny_llama(), "tiny", Tok())
    got = c.get("/scope/api/lens/layers", params=dict(model="tiny"))
    assert got.status_code == 200 and len(got.json()["steps"]) == config.LENS_STEPS
    assert got.json()["at"] == config.LENS_STEP_DEFAULT, "the 409'd POST saved nothing"


def test_lens_layers_on_a_model_the_hooks_cannot_read_answer_409_with_why(client):
    """A model no adapter reads (Inkling 2 on an oMLX before 0.8.3; here one without decoder layers): the lens layers'
    GET and POST answer 409 with why the hooks cannot be installed, as the words route does, not a 500, and the POST
    saves nothing; before an install was tried (the mode is off) and after one failed (the kept install error)."""
    import mlx.nn as nn
    c, home = client
    tracer = MANAGER.attach(nn.Module(), "bare", Tok())
    why = "hooks could not be installed on bare: ValueError: no metalscope adapter for Module"
    for tried in (False, True):
        assert (tracer.install_error is not None) is tried
        for r in (c.get("/scope/api/lens/layers", params=dict(model="bare")),
                  c.post("/scope/api/lens/layers", json=dict(model="bare", step=1))):
            assert r.status_code == 409 and r.json()["detail"] == why
        words = c.post("/scope/api/steer/words", json=dict(model="bare", name="w", words=["<9>"], ids=[9]))
        assert words.status_code == 409 and words.json()["detail"] == why     # it tries the install
    assert MANAGER.lens_step() == config.LENS_STEP_DEFAULT


def test_lens_layers_say_where_the_word_edit_acts_with_a_j_lens_as_the_words_route_does(client):
    """``edit_band``: the band the dock's word edit removes over (a swap pushes at its middle), the words route's own
    ``band`` (one function); left out while the tracer has no J-lens loaded (the label is then left out; the words
    route itself may still load one from disk)."""
    import mlx.core as mx
    from types import SimpleNamespace
    from metalscope import jlens

    c, home = client
    tracer = MANAGER.attach(tiny_llama(), "tiny", Tok())
    assert "edit_band" not in c.get("/scope/api/lens/layers", params=dict(model="tiny")).json()     # no J-lens
    tracer.lens = SimpleNamespace(lens=jlens.Lens({l: mx.eye(64) for l in range(2)}))
    mx.eval(list(tracer.lens.lens.jacobians.values()))
    tracer.lens_steps = lambda: [([1], [1])] * config.LENS_STEPS
    got = c.get("/scope/api/lens/layers", params=dict(model="tiny")).json()
    made = c.post("/scope/api/steer/words", json=dict(model="tiny", name="soft", words=["<9>"], ids=[9])).json()
    from metalscope.adapters.generic import text_model
    assert got["edit_band"] == made["band"] == list(jlens.workspace_band(len(text_model(tracer.model).layers)))


# -- the dock's link offer: a build that may be a model already here

def _holding(ckpt: str, concepts=(), lenses=()) -> None:
    from tests.test_models import FLASH, holding
    holding(ckpt, concepts, lenses, FLASH)


@pytest.mark.parametrize("engine", ["omlx", "ds4"])
def test_the_link_offer_route_and_the_link_it_makes(client, engine, monkeypatch):
    from types import SimpleNamespace
    from tests.test_models import FLASH
    c = scope_client(web.Backend(engine))
    assert c.get("/scope/api/model/link").json()["detail"] == "no model is loaded to link"
    _holding("Flash", concepts=["calm", "terse"], lenses=["flash-lens"])
    models.concept_dir("Flash", "the").mkdir()
    models.vector_path("Flash", "the").write_bytes(b"x")                # a word vector: not a concept on the Steer page
    models.check_shape("Build", FLASH)
    offer =c.get("/scope/api/model/link", params=dict(model="Build")).json()
    assert offer == dict(model="Build", checkpoint="Build", linked=False, own=False, shape=dict(n_layers=48, width=10240),
                         candidates=[dict(name="Flash", concepts=2, lenses=1)])
    # Loaded with its hooks installed: it reads the shared lens from its next forward, as /jlens use does.
    relensed = []
    tracer = SimpleNamespace(name="Build", installed=True, refresh_lens=lambda: relensed.append(models.checkpoint_of("Build"))
                             or SimpleNamespace(name="flash-lens"))
    MANAGER.tracers["build"] = tracer
    r = c.post("/scope/api/model/link", json=dict(model="Build", checkpoint="Flash"))
    assert r.status_code == 200 and r.json() == dict(model="Build", checkpoint="Flash", concepts=2, lenses=1, lens="flash-lens",
                                                     note=None)
    assert models.checkpoint_of("Build") == "Flash" and relensed == ["Flash"]
    assert c.get("/scope/api/model/link", params=dict(model="Build")).json()["linked"] is True
    assert c.get("/scope/api/model/link").json()["model"] == "Build"     # the loaded model by default


def test_the_link_route_refuses_with_links_own_message_or_a_non_candidate(client, monkeypatch):
    from metalscope.learn import state
    from tests.test_models import FLASH
    c, home = client
    _holding("Flash", concepts=["calm"])
    _holding("Empty")
    models.check_shape("V4", models.shape_of(43, 4, 4096))
    models.check_shape("Build", FLASH)
    post = lambda ckpt, model="Build": c.post("/scope/api/model/link", json=dict(model=model, checkpoint=ckpt))
    for ckpt in ("Empty", "Nobody", "Build"):                               # no data, no such model, itself
        r = post(ckpt)
        assert r.status_code == 409 and r.json()["detail"] == f"{ckpt} is not a model of the same shape with data", r.text
    r = post("Flash", model="V4")
    assert r.status_code == 409 and r.json()["detail"] == "V4 is 43 layers x 4 x 4096, Flash is 48 layers x 4 x 2560: not the same model"
    models.link("Copy", "Flash")
    r = post("Copy")
    assert r.status_code == 409 and r.json()["detail"] == "Copy is a served name of Flash, not a checkpoint"
    _holding("Mine", concepts=["own"])
    r = post("Flash", model="Mine")
    assert r.status_code == 409 and r.json()["detail"].startswith("Mine has data of its own in ")
    job = dict(job="cal-1", kind="calibrate", name="calm", model="Build", outcome=None)
    monkeypatch.setattr(state, "running", lambda exclude=None: job)
    r = post("Flash")
    assert r.status_code == 409 and r.json()["detail"].startswith("cannot link Build now: a job is running: calibrating calm")
    assert r.json()["detail"].endswith("metalscope jobs cancel cal-1")
    assert c.post("/scope/api/model/link", json=dict(model="Build", checkpoint="../x")).status_code == 400
    assert models.checkpoint_of("Build") == "Build"


def test_the_link_route_answers_200_once_linked_whatever_the_lens_does_and_500_says_what_happened(client, monkeypatch):
    from types import SimpleNamespace
    from metalscope import tally
    from tests.test_models import FLASH
    c, home = client
    _holding("Flash", concepts=["calm"], lenses=["flash-lens"])
    post = lambda model: c.post("/scope/api/model/link", json=dict(model=model, checkpoint="Flash"))
    later = "J-lens loads at the next model load"                       # Pi says "linked to ..." on the line above
    # The lens re-read raises: linked all the same, no lens, a note saying why.
    models.check_shape("A", FLASH)
    MANAGER.tracers["a"] = SimpleNamespace(name="A", installed=True,
                                           refresh_lens=lambda: (_ for _ in ()).throw(MemoryError("out of memory")))
    r = post("A")
    assert r.status_code == 200 and r.json() == dict(model="A", checkpoint="Flash", concepts=1, lenses=1, lens=None,
                                                     note=later + ": MemoryError: out of memory")
    assert models.checkpoint_of("A") == "Flash"
    # It ran but read no lens though the checkpoint has one (the tracer logged why): the same.
    models.check_shape("B", FLASH)
    MANAGER.tracers["b"] = SimpleNamespace(name="B", installed=True, refresh_lens=lambda: None)
    r = post("B")
    assert r.status_code == 200 and r.json()["lens"] is None
    assert r.json()["note"] == later + " (the server log says why)"       # whole on two lines of Pi's narrowest column (link.test.ts)
    # Not loaded: nothing to re-read; it reads the lens when it loads.
    models.check_shape("C", FLASH)
    assert post("C").json() == dict(model="C", checkpoint="Flash", concepts=1, lenses=1, lens="flash-lens", note=None)
    # An unexpected error before the link: nothing linked.
    models.check_shape("D", FLASH)
    with monkeypatch.context() as m:
        m.setattr(models, "link_offer", lambda model: (_ for _ in ()).throw(RuntimeError("bad record")))
        r = post("D")
    assert r.status_code == 500 and r.json()["detail"] == "nothing was linked: RuntimeError: bad record"
    assert models.checkpoint_of("D") == "D"
    # One while it links: it may be part way, and running it again finishes it.
    models.norms_path("D").parent.mkdir(parents=True)
    models.norms_path("D").write_text("{}")
    with monkeypatch.context() as m:
        m.setattr(tally, "move", lambda a, b: (_ for _ in ()).throw(RuntimeError("tally locked")))
        r = post("D")
    assert r.status_code == 500
    assert r.json()["detail"] == "the link may have stopped part way: RuntimeError: tally locked; run it again to finish it"
    assert post("D").status_code == 200 and models.checkpoint_of("D") == "Flash" and models.norms_path("D").is_file()


def test_a_word_vector_name_ending_in_a_newline_is_refused(client):
    c, home = client
    r = c.post("/scope/api/steer/words", json=dict(model="tiny", name="ok\n", words=["soft"]))
    assert r.status_code == 400 and r.json()["detail"] == "name: letters, digits, _ . - only"


def test_a_jobs_state_of_the_wrong_shape_leaves_the_other_jobs_listed(client):
    from metalscope.learn import state
    c, home = client
    for job, st in (("j-0", [1]), ("j-1", dict(job="j-1", started=101)), ("j-2", dict(kind="learn", started=102)),
                    ("j-3", dict(job="j-3", kind="learn", phase="done", outcome="done", started=103))):
        state.job_dir(job).mkdir(parents=True)
        (state.job_dir(job) / "state.json").write_text(json.dumps(st))
    assert state.read_state("j-0") is None
    r = c.get("/scope/api/jobs")
    assert r.status_code == 200 and [j["job"] for j in r.json()["jobs"]] == ["j-3"] and r.json()["latest"]["job"] == "j-3"

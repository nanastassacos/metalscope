"""The two backends as one thing to start, stop and list (backends.py): ds4's models from what it served and the
GGUF folders, each at its own maximum context; starting one backend's model frees the other's; `metalscope omlx`
and `metalscope ds4 MODEL` start, `stop` stops, `status --json` is what Pi's /model lists."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from metalscope import backends as B, cli, config


def gguf(path: Path, arch: str, **kv) -> Path:
    """A GGUF header with general.architecture and the given integer keys (all a model's listing reads)."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    items = [s("general.architecture") + struct.pack("<I", 8) + s(arch)]
    items += [s(k.replace("__", ".")) + struct.pack("<II", 4, v) for k, v in kv.items()]
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(items)) + b"".join(items))
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "home"))
    folder = tmp_path / "gguf"
    folder.mkdir()
    monkeypatch.setenv("METALSCOPE_DS4_GGUF", str(folder))
    from metalscope.ds4 import build
    monkeypatch.setattr(build, "build_dir", lambda: tmp_path / "no-build")
    qwen = gguf(folder / "Qwen3.8-Flash-Next-Q4.gguf", "qwen4exp", qwen4exp__context_length=262144)
    v41 = gguf(folder / "V41-Q2.gguf", "deepseek41", deepseek41__max_position_embeddings=1048576)
    gguf(folder / "V4-old.gguf", "deepseek4", deepseek4__context_length=1048576)     # not ds4-capture servable
    (folder / "notes.gguf").write_text("not a gguf")
    return tmp_path, qwen, v41


def test_ds4s_models_are_the_ones_it_served_and_the_servable_files_at_their_full_context(home):
    _, qwen, v41 = home
    assert B.strip_ctx(["--mtp", "--ctx", "32768"]) == (["--mtp"], 32768)
    assert B.strip_ctx(["-c", "8192", "--ssd-streaming"]) == (["--ssd-streaming"], 8192)
    assert B.strip_ctx(["--ctx=4096"]) == ([], 4096)
    # Never served: named as `ds4 serve` names a file, with the options its kind needs.
    assert [(m["name"], m["ctx"], m["args"]) for m in B.ds4_models()] == [
        ("Qwen3.8-Flash-Next-Q4-ds4", 262144, ["--mtp"]), ("V41-Q2-ds4", 1048576, [])]
    # Served once under a name (ds4 serve records it): listed by that name, its options kept but not its --ctx.
    B.register("Qwen3.8-Flash-Next-ds4-q4k", qwen, ["--mtp", "--ctx", "32768"])
    models = B.ds4_models()
    assert [(m["name"], m["ctx"], m["args"]) for m in models] == [
        ("Qwen3.8-Flash-Next-ds4-q4k", 262144, ["--mtp"]), ("V41-Q2-ds4", 1048576, [])]
    assert B.ds4_model("Qwen3.8-Flash-Next-ds4-q4k")["gguf"] == str(qwen.resolve())
    assert B.ds4_model(str(qwen))["name"] == "Qwen3.8-Flash-Next-ds4-q4k"        # its file: the name it has
    assert B.ds4_model(str(v41), name="V41")["name"] == "V41"                      # a new file, named by the caller
    with pytest.raises(SystemExit, match="ds4's models: Qwen3.8-Flash-Next-ds4-q4k, V41-Q2-ds4"):
        B.ds4_model("nope")
    with pytest.raises(SystemExit, match="not a model metalscope's ds4 capture covers"):
        B.ds4_model(str(home[0] / "gguf" / "V4-old.gguf"))


def test_a_gguf_headers_arrays_are_read_when_named_and_skipped_otherwise(tmp_path):
    """The vocabulary (string arrays) and other arrays are skipped unread unless asked for: the keys after them still
    read right, and the tokenizer reads them in full."""
    from metalscope.ds4.vocab import read_gguf_meta, tokenizer_for

    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    toks = ["a", "b", "Ġc", "<|end|>"]
    items = [s("general.architecture") + struct.pack("<I", 8) + s("qwen4exp"),
             s("tokenizer.ggml.model") + struct.pack("<I", 8) + s("gpt2"),
             s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(toks)) + b"".join(s(t) for t in toks),
             s("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 5, len(toks)) + struct.pack("<4i", 1, 1, 1, 3),
             s("qwen4exp.context_length") + struct.pack("<II", 4, 262144)]
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(items)) + b"".join(items))
    assert read_gguf_meta(path) == {"general.architecture": "qwen4exp", "tokenizer.ggml.model": "gpt2",
                                    "qwen4exp.context_length": 262144}
    meta = read_gguf_meta(path, ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type"))
    assert meta["tokenizer.ggml.tokens"] == toks and meta["tokenizer.ggml.token_type"] == [1, 1, 1, 3]
    tok = tokenizer_for(path)
    assert tok.decode([0, 2]) == "a c" and tok.all_special_ids == [3]
    assert tokenizer_for(path).decode([1]) == "b"                       # from its cache


def test_starting_a_ds4_model_serves_it_in_the_background_at_its_full_context(home, monkeypatch):
    """`ds4 start` spawns `ds4 serve`, which frees oMLX's memory itself (free_for_ds4): the start never unloads it."""
    tmp, qwen, _ = home
    B.register("Q-ds4", qwen, ["--mtp"])
    events, started = [], []
    monkeypatch.setattr(B, "free_omlx", lambda log=print: events.append("free oMLX"))
    monkeypatch.setattr(B, "_ds4_stop", lambda log=print, timeout=90: events.append("stop ds4") or True)

    class Proc:
        pid = 4242

        def __init__(self, cmd, **kw):
            started.append(cmd)
            assert kw["start_new_session"]            # outlives the command (and Pi) that started it

    monkeypatch.setattr(B.subprocess, "Popen", Proc)
    monkeypatch.setattr(B, "_wait_ds4", lambda *a, **k: None)                    # --no-wait's wait for the load
    serving = [None]
    monkeypatch.setattr(B, "ds4_serving", lambda: serving[0])
    B.ds4_start("Q-ds4", wait=False, log=lambda s: None)
    assert events == []
    cmd = started[0]
    assert cmd[1:6] == ["-m", "metalscope.cli", "ds4", "serve", str(qwen.resolve())]
    assert cmd[cmd.index("--name") + 1] == "Q-ds4" and cmd[cmd.index("--") + 1:] == ["--mtp", "--ctx", "262144"]
    # The same model at the same context already served: nothing restarts. Another one: ds4 stopped first.
    # Its record is written as it starts (Pi and a second switch see it loading before its port opens).
    assert json.loads((B._ds4_home() / "serving.json").read_text())["name"] == "Q-ds4"
    serving[0] = dict(pid=1, name="Q-ds4", ctx=262144, ready=True)
    B.ds4_start("Q-ds4", wait=False, log=lambda s: None)
    assert len(started) == 1
    serving[0] = dict(pid=1, name="Q-ds4", ctx=65536, ready=False)              # started smaller by hand, loading:
    B.ds4_start("Q-ds4", wait=False, log=lambda s: None)                         # kept (no context asked for)
    assert len(started) == 1
    serving[0] = dict(pid=1, name="Q-ds4", ctx=262144, ready=True)
    B.ds4_start("Q-ds4", ctx=32768, wait=False, log=lambda s: None)               # a smaller context asked for
    assert events == ["stop ds4"] and started[1][-2:] == ["--ctx", "32768"]
    # No model named: the one served last.
    serving[0] = None
    B.ds4_start(None, wait=False, log=lambda s: None)
    assert started[-1][started[-1].index("--name") + 1] == "Q-ds4"


def test_a_serve_log_past_the_limit_is_moved_aside_when_ds4_starts(home, monkeypatch):
    tmp, qwen, _ = home
    B.register("Q-ds4", qwen, [])
    monkeypatch.setattr(B, "free_omlx", lambda log=print: None)
    monkeypatch.setattr(B, "ds4_serving", lambda: None)
    monkeypatch.setattr(B.subprocess, "Popen", lambda cmd, **kw: SimpleNamespace(pid=4242))
    monkeypatch.setattr(B, "_wait_ds4", lambda *a, **k: None)                    # --no-wait's wait for the load
    monkeypatch.setattr(config, "LOG_LIMIT", 100)
    log = B._ds4_home() / "serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("old\n" * 100)
    B.ds4_start("Q-ds4", wait=False, log=lambda s: None)
    assert (B._ds4_home() / "serve.log.1").read_text() == "old\n" * 100
    assert log.read_text().startswith("\n==== ") and log.stat().st_size < 1000


def test_the_serving_record_belongs_to_the_serve_process(home):
    B.write_serving("Q-ds4", 262144)
    path = B._ds4_home() / "serving.json"
    rec = json.loads(path.read_text())
    assert rec["name"] == "Q-ds4" and rec["ctx"] == 262144
    rec["pid"] = rec["pid"] + 1                         # another process's record: left alone
    path.write_text(json.dumps(rec))
    B.clear_serving()
    assert path.exists()
    B.write_serving("Q-ds4", 262144)
    B.clear_serving()
    assert not path.exists()


def test_backend_words_start_and_stop(monkeypatch):
    seen = []
    monkeypatch.setattr(B, "omlx_start", lambda model=None, log=print: seen.append(("omlx", model)))
    monkeypatch.setattr(B, "omlx_stop", lambda log=print: seen.append(("omlx stop",)) or True)
    monkeypatch.setattr(B, "ds4_start", lambda spec=None, **kw: seen.append(("ds4", spec, kw)))
    monkeypatch.setattr(B, "ds4_stop", lambda log=print, timeout=90: seen.append(("ds4 stop",)) or True)
    cli.main(["omlx"])
    cli.main(["omlx", "Qwen3.8-Flash-Next-oQ4e-mtp"])
    cli.main(["omlx", "stop"])
    cli.main(["ds4"])
    cli.main(["ds4", "Q-ds4", "--ctx", "65536"])
    cli.main(["ds4", "/m/new.gguf", "--name", "New-ds4", "--no-wait", "--", "--ssd-streaming"])
    cli.main(["ds4", "stop"])
    assert seen[:3] == [("omlx", None), ("omlx", "Qwen3.8-Flash-Next-oQ4e-mtp"), ("omlx stop",)]
    assert seen[3] == ("ds4", None, dict(name=None, ctx=None, args=None, wait=True))
    assert seen[4] == ("ds4", "Q-ds4", dict(name=None, ctx=65536, args=None, wait=True))
    assert seen[5] == ("ds4", "/m/new.gguf", dict(name="New-ds4", ctx=None, args=["--ssd-streaming"], wait=False))
    assert seen[6] == ("ds4 stop",)


def test_ds4_options_change_only_once_a_load_works_and_a_bare_double_dash_clears_them(home, monkeypatch):
    """`ds4 start` hands the options to serve, which remembers them once ds4-server answers (a failed load leaves the
    old ones); `metalscope ds4 MODEL --` with nothing after it starts the model with no options."""
    _, qwen, _ = home
    B.register("Q-ds4", qwen, ["--mtp"])
    started = []
    monkeypatch.setattr(B.subprocess, "Popen", lambda cmd, **kw: started.append(cmd) or SimpleNamespace(pid=4242))
    monkeypatch.setattr(B, "_wait_ds4", lambda *a, **k: None)
    monkeypatch.setattr(B, "ds4_serving", lambda: None)
    B.ds4_start("Q-ds4", args=["--bad-option"], wait=False, log=lambda s: None)
    assert started[-1][started[-1].index("--") + 1:] == ["--bad-option", "--ctx", "262144"]
    assert json.loads((B._ds4_home() / "models.json").read_text())["Q-ds4"]["args"] == ["--mtp"]
    cli.main(["ds4", "Q-ds4", "--no-wait", "--"])
    assert started[-1][started[-1].index("--") + 1:] == ["--ctx", "262144"]
    cli.main(["ds4", "Q-ds4", "--no-wait"])
    assert started[-1][started[-1].index("--") + 1:] == ["--mtp", "--ctx", "262144"]


def test_status_names_a_ds4_server_left_running_without_its_serve(monkeypatch, capsys):
    """serve killed (SIGKILL, a crash) leaves its ds4-server holding the model's memory: not "stopped", but the pid
    and the command that stops it."""
    monkeypatch.setattr(B, "ds4_serving", lambda: None)
    monkeypatch.setattr(B, "ds4_models", lambda: [])
    monkeypatch.setattr(B, "memory_free", lambda: 12)
    orphans = [4321]
    monkeypatch.setattr(B, "_ds4_servers", lambda: orphans)
    cli.main(["status", "--json"])
    assert json.loads(capsys.readouterr().out)["ds4"]["orphaned"] == [4321]
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "ds4   not serving: ds4-server pid 4321 left running  http://" in out and "stop    metalscope ds4 stop\n" in out
    orphans.clear()
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "ds4   stopped  http://" in out and "metalscope ds4 stop" not in out


def test_a_first_load_is_known_by_its_name_while_it_loads_and_last_only_once_it_has_loaded(home, monkeypatch):
    """A GGUF served for the first time is remembered only once it loads (its options must work), but meanwhile
    `metalscope ds4` and `ds4 start NAME` still find it loading and wait; a failed first load is not "the last"."""
    _, qwen, _ = home
    B.register("Q-ds4", qwen, ["--mtp"])                          # loaded once: the last
    started, said = [], []
    monkeypatch.setattr(B.subprocess, "Popen", lambda cmd, **kw: started.append(cmd) or SimpleNamespace(pid=4242))
    monkeypatch.setattr(B, "_wait_ds4", lambda *a, **k: None)
    serving = [None]
    monkeypatch.setattr(B, "ds4_serving", lambda: serving[0])
    new = home[0] / "gguf" / "V41-Q2.gguf"
    B.ds4_start(str(new), name="New-ds4", wait=False, log=said.append)
    assert len(started) == 1 and json.loads((B._ds4_home() / "last.json").read_text())["name"] == "Q-ds4"
    serving[0] = dict(pid=1, name="New-ds4", ctx=1048576, ready=False)          # loading, not remembered yet
    for spec in ("New-ds4", None):
        said.clear()
        B.ds4_start(spec, wait=False, log=said.append)
        assert said == ["ds4 already serves New-ds4 (loading)"] and len(started) == 1
    with pytest.raises(SystemExit, match="no ds4 model 'Other-ds4'"):
        B.ds4_start("Other-ds4", wait=False, log=said.append)
    B.register("New-ds4", new, [])                                 # serve, once it has loaded
    assert json.loads((B._ds4_home() / "last.json").read_text())["name"] == "New-ds4"


def test_ctrl_c_exits_130_without_a_traceback_and_ds4_start_says_the_load_goes_on(home, monkeypatch, capsys):
    """Ctrl-C while `metalscope ds4 MODEL` waits: serve runs on in its own session, so the start says so and how to
    stop it; the command exits 130, as a shell expects of an interrupt."""
    _, qwen, _ = home
    B.register("Q-ds4", qwen, ["--mtp"])
    monkeypatch.setattr(B.subprocess, "Popen", lambda cmd, **kw: SimpleNamespace(pid=4242))
    monkeypatch.setattr(B, "ds4_serving", lambda: None)

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(B, "_wait_ds4", interrupted)
    with pytest.raises(SystemExit) as exc:
        cli.main(["ds4", "Q-ds4"])
    assert exc.value.code == 130
    out = capsys.readouterr().out
    assert out.endswith("ds4 goes on loading Q-ds4 in the background (metalscope ds4 stop stops it)\n")


def test_status_lists_both_backends_for_people_and_for_pi(monkeypatch, capsys):
    monkeypatch.setattr(B, "omlx_models", lambda: dict(up=True, url="http://127.0.0.1:8008", tracing="on", models=[
        dict(name="Flash-oQ4e", ctx=524288, loaded=True, reasoning=True, thinking_format="qwen-chat-template")]))
    monkeypatch.setattr(B, "ds4_serving", lambda: None)
    monkeypatch.setattr(B, "ds4_models", lambda: [dict(name="Flash-ds4", gguf="/m/q.gguf", args=["--mtp"], ctx=262144,
                                                       arch="qwen4exp", family="qwen4")])
    monkeypatch.setattr(B, "memory_free", lambda: 91)
    cli.main(["status", "--json"])
    st = json.loads(capsys.readouterr().out)
    assert st["omlx"]["models"][0]["loaded"] and st["ds4"]["up"] is False
    assert st["ds4"]["models"] == [dict(name="Flash-ds4", ctx=262144, gguf="/m/q.gguf", args=["--mtp"], loaded=False,
                                        reasoning=True, thinking_format="qwen-chat-template", tested=True, edit=True)]
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "oMLX  running" in out and "Flash-oQ4e" in out and "524,288  loaded" in out
    assert "ds4   stopped" in out and "Flash-ds4" in out and "262,144" in out and "memory 91% free" in out
    # What to do next: switch (either), stop what runs (oMLX here), Pi, and where every command is.
    assert "switch  metalscope omlx MODEL · metalscope ds4 MODEL" in out and "stop    metalscope omlx stop\n" in out
    assert "metalscope ds4 stop" not in out and "help    metalscope help (every command)" in out
    cli.main([])                                                    # metalscope alone: the same
    assert capsys.readouterr().out == out


def test_omlx_models_say_tested_by_omlxs_model_type_and_edit_as_its_status_does(home, monkeypatch, real_omlx_listing):
    """`metalscope status` (Pi's /model, doctor): oMLX's chat models with tested and edit as metalscope's status says
    per model; tested, where it does not say (an older hook), by the family oMLX reads from each checkpoint (its admin
    API's config_model_type, models.TESTED); each left out where nothing says; kept for when oMLX is not running."""
    from metalscope.omlx import admin as omlx_admin
    trace = dict(caps=["edit_at", "load_copy"], chat=dict(Q=dict(reasoning=True, thinking_format="qwen-chat-template",
                                                               tested=True, edit=True), G=dict(tested=True)))
    up, admin = [True], [dict(id="Q", loaded=True, config_model_type="qwen3_5_moe"), dict(id="G", config_model_type="gemma4")]

    class Admin:
        base_url = "http://omlx"

        def listening(self):
            return up[0]

        def _request(self, method, path, **kw):
            return 200, dict(data=[dict(id="Q", max_model_len=262144), dict(id="G"), dict(id="MarkItDown")])

        def admin_models(self):
            if isinstance(admin, Exception):
                raise admin
            return admin

        def trace_status(self):
            return trace
    monkeypatch.setattr(B, "_admin", lambda: Admin())
    flags = lambda: [(m["name"], m.get("tested"), m.get("edit")) for m in B.omlx_models()["models"]]
    # The server's word first (G: tested, its edit not known while it is not loaded), whatever the admin API says.
    assert flags() == [("Q", True, True), ("G", True, None)]
    admin = omlx_admin.ServerError("no API key")                              # the admin API refused: the server's still
    assert flags() == [("Q", True, True), ("G", True, None)]
    admin = [dict(id="Q", config_model_type="qwen3_5_moe"), dict(id="G", config_model_type="gemma4")]
    trace = None                                                              # an older hook, or none: the admin API's
    assert flags() == [("Q", True, None), ("G", False, None)]                # family; the edit not known, not False
    up[0] = False                                                             # not running: as last seen
    assert flags() == [("Q", True, None), ("G", False, None)]


def test_model_lines_say_per_model_tested_lens_concepts_and_edit_loading_nothing(home, monkeypatch):
    """doctor's lines (and the installs' last ones): a model each backend has, as `status` lists it; its J-lens in use,
    turned off or none; the concepts learned for it (profiles); `?` where an older oMLX said nothing."""
    from metalscope import jlens, models
    monkeypatch.setattr(B, "omlx_models", lambda: dict(up=True, models=[
        dict(name="Qwen3.6-35B-A3B-MLX-4bit", tested=True, edit=True), dict(name="Gemma-4", tested=False, edit=False),
        dict(name="Old")]))
    for model, lens in (("Qwen3.6-35B-A3B-MLX-4bit", "q"), ("Gemma-4", "g")):
        d = jlens.lenses_dir(model) / lens
        d.mkdir(parents=True)
        (d / "lens.safetensors").write_bytes(b"")
        (d / "lens.json").write_text(json.dumps(dict(checkpoint=model)))
    models.update("Gemma-4", lens=jlens.LOGIT)                               # its J-lens turned off
    for model, concepts in (("Qwen3.6-35B-A3B-MLX-4bit", ("calm", "terse")), ("Gemma-4", ("calm",))):
        for c in concepts:
            models.profile_path(model, c).parent.mkdir(parents=True)
            models.profile_path(model, c).write_text("{}")
    assert B.model_lines() == [
        "backend  model                      tested  J-lens  concepts  edit",
        "oMLX     Qwen3.6-35B-A3B-MLX-4bit   yes     yes     2         yes",
        "oMLX     Gemma-4                    no      off     1         no",
        "oMLX     Old                        ?       no      0         ?",
        "ds4      Qwen3.8-Flash-Next-Q4-ds4  yes     no      0         yes",
        "ds4      V41-Q2-ds4                 yes     no      0         yes"]
    assert [l.split()[1] for l in B.model_lines(("ds4",))] == ["model", "Qwen3.8-Flash-Next-Q4-ds4", "V41-Q2-ds4"]
    assert [m["family"] for m in B.ds4_models()] == ["qwen4", "deepseek41"]
    # Whatever goes wrong (oMLX answering 200 with no JSON, the cache not writable): one line, never an error.
    monkeypatch.setattr(B, "omlx_models", lambda: (_ for _ in ()).throw(ValueError("Expecting value: line 1")))
    assert B.model_lines() == ["the models could not be listed: ValueError: Expecting value: line 1"]
    monkeypatch.setattr(B, "omlx_models", lambda: (_ for _ in ()).throw(PermissionError("models.json")))
    assert B.model_lines(("omlx",)) == ["the models could not be listed: PermissionError: models.json"]
    assert len(B.model_lines(("ds4",))) == 3


def test_help_is_the_overview_or_one_commands_options(capsys, monkeypatch):
    with pytest.raises(SystemExit):
        cli.main(["help"])
    assert capsys.readouterr().out.startswith("usage: metalscope [-h] COMMAND ...")
    with pytest.raises(SystemExit):
        cli.main(["help", "ds4", "start"])
    out = capsys.readouterr().out
    assert out.startswith("usage: metalscope ds4 start") and "--ctx" in out and "--no-wait" in out
    # Words for ds4-server after --: help still only shows the options (it never starts ds4).
    started = []
    monkeypatch.setattr(B, "ds4_start", lambda *a, **k: started.append(a))
    with pytest.raises(SystemExit):
        cli.main(["help", "ds4", "start", "M", "--", "--ctx", "1"])
    with pytest.raises(SystemExit):
        cli.main(["help", "ds4", "M", "--", "--mtp"])
    assert started == [] and capsys.readouterr().out.count("usage: metalscope ds4 start") == 2
    with pytest.raises(SystemExit):
        cli.main(["help", "help"])
    assert capsys.readouterr().out.startswith("usage: metalscope [-h] COMMAND ...")


def test_a_serve_process_is_known_by_its_command_not_by_words_in_it():
    assert B.is_serve("/v/bin/python3 /v/bin/metalscope ds4 serve /m/q.gguf --name Q -- --mtp")
    assert B.is_serve("/v/bin/python -m metalscope.cli ds4 serve /m/q.gguf --name Q")
    assert B.is_serve("/v/bin/python3.11 -m metalscope.cli ds4 serve /m/q.gguf")
    assert B.is_serve("/opt/homebrew/Cellar/python@3.11/3.11.9/Frameworks/Python.framework/Versions/3.11/Resources/"
                      "Python.app/Contents/MacOS/Python -m metalscope.cli ds4 serve /m/q.gguf")   # a framework Python
    assert not B.is_serve("/bin/zsh -c until ! pgrep -f 'metalscope ds4 serve'; do sleep 1; done")
    assert not B.is_serve("bash /tmp/restart.sh metalscope ds4 serve")
    assert not B.is_serve("vim notes about metalscope ds4 serve")
    assert not B.is_serve("")


def test_freeing_omlx_waits_for_a_load_in_progress_and_reports_a_refusal(monkeypatch):
    from metalscope import gpulock
    from metalscope.omlx import admin as omlx_admin
    states = [[dict(id="Big", is_loading=True, loaded=False, estimated_size=90 << 30)],
              [dict(id="Big", is_loading=False, loaded=True, estimated_size=90 << 30),
               dict(id="MarkItDown", loaded=True, estimated_size=0)]]

    class Admin:
        def listening(self):
            return True

        def admin_models(self):
            return states[0]

    unloaded, waited = [], []
    monkeypatch.setattr(B, "_admin", lambda: Admin())
    monkeypatch.setattr(gpulock, "wait_no_loading", lambda a, log=print: waited.append(1) or states.pop(0) or True)
    monkeypatch.setattr(omlx_admin, "unload", lambda a, mid, log=print: unloaded.append(mid))
    B.free_omlx(log=lambda s: None)
    assert waited == [1] and unloaded == ["Big"]                # the converter (under a gigabyte) stays

    def refuse(a, mid, log=print):
        raise omlx_admin.ServerError("POST /admin/api/models/Big/unload: HTTP 409: Model still loading")
    monkeypatch.setattr(omlx_admin, "unload", refuse)
    with pytest.raises(SystemExit, match="could not free oMLX's memory .*409"):
        B.free_omlx(log=lambda s: None)


def test_one_switch_at_a_time(home):
    import fcntl
    said = []
    B.config.home().mkdir(parents=True, exist_ok=True)
    with open(B.config.home() / "backends.lock", "a+") as other:   # another process's switch, holding the lock
        fcntl.flock(other, fcntl.LOCK_EX)
        import threading
        t = threading.Thread(target=lambda: B.switching(said.append).__enter__())
        t.start()
        t.join(0.3)
        assert t.is_alive() and said == ["waiting for another backend switch to finish"]
        fcntl.flock(other, fcntl.LOCK_UN)
        t.join(2)
        assert not t.is_alive()


def test_no_switch_while_a_gpu_job_runs(home, monkeypatch):
    """A GPU job measures the served model (ds4) or its own copy in the memory a switch would use: starting or
    stopping either backend is refused with its message, and nothing is stopped."""
    from metalscope import gpulock
    signalled, quiet = [], (lambda s: None)
    monkeypatch.setattr(B, "_serve_pids", lambda: [4242])                    # a ds4 serve runs
    monkeypatch.setattr(B, "_ds4_servers", lambda: [])
    monkeypatch.setattr(B, "_alive", lambda pid: False)
    monkeypatch.setattr(B, "_signal", lambda pids, sig: signalled.append((list(pids), sig)))
    monkeypatch.setattr(B, "ds4_serving", lambda: dict(pid=4242, name="Qwen3.8-Flash-Next-Q4-ds4", ctx=None, ready=True))

    class Admin:                                                              # oMLX answering, m not loaded
        base_url = "http://omlx"

        def answering(self):
            return True

        def admin_models(self):
            return [dict(id="m", loaded=False)]
    monkeypatch.setattr(B, "_admin", lambda: Admin())
    lock = gpulock.GpuLock(gpulock.Holder("learning toasty", stage="presence")).acquire()
    try:
        for switch in (lambda: B.ds4_stop(log=quiet), lambda: B.omlx_start("m", log=quiet),
                       lambda: B.ds4_start("V41-Q2-ds4", wait=False, log=quiet)):
            with pytest.raises(SystemExit, match="cannot switch backends now: learning toasty: presence; try again"):
                switch()
        assert signalled == []                                                # nothing was stopped
        # What changes nothing goes ahead: the model ds4 already serves, oMLX already answering, nothing to stop.
        assert B.ds4_start("Qwen3.8-Flash-Next-Q4-ds4", wait=False, log=quiet)["name"] == "Qwen3.8-Flash-Next-Q4-ds4"
        B.omlx_start(None, log=quiet)
        monkeypatch.setattr(B, "_serve_pids", lambda: [])
        assert B.ds4_stop(log=quiet) is False
    finally:
        lock.release()
    monkeypatch.setattr(B, "_serve_pids", lambda: [4242])
    assert B.ds4_stop(log=quiet) is True and signalled[0][0] == [4242]      # free again: it goes ahead


def test_omlx_model_names_a_model_oMLX_has_before_ds4_is_stopped(home, monkeypatch, tmp_path):
    """`metalscope omlx MODEL` frees ds4's memory for MODEL: a name oMLX does not have (a typo) is refused from oMLX's
    own list before ds4 is stopped. oMLX not running: it is started first (no model loads), then asked; the list
    from when it last ran only says what it may not have."""
    events, said = [], []
    running = [True]

    class Admin:
        base_url = "http://omlx"

        def answering(self):
            return running[0]

        def wait(self, up, timeout):
            running[0] = True
            return True

        def admin_models(self):
            return [dict(id="Flash-oQ4e", loaded=False)]
    monkeypatch.setattr(B, "_admin", lambda: Admin())
    monkeypatch.setattr(B, "_ds4_stop", lambda log=print, timeout=90: events.append("stop ds4") or True)
    monkeypatch.setattr(B.omlx_admin, "load", lambda a, m, log=print: events.append(("load", m)))
    cli_ = tmp_path / "omlx"
    cli_.write_text("")
    monkeypatch.setattr(B.omlx_admin, "OMLX_CLI", cli_)
    monkeypatch.setattr(B.subprocess, "run", lambda cmd, **k: events.append(("omlx", cmd[1])))
    with pytest.raises(SystemExit, match=r"^oMLX has no model 'Flash-oQ4' \(its models: Flash-oQ4e\)$"):
        B.omlx_start("Flash-oQ4", log=said.append)
    assert events == []
    B.omlx_start("Flash-oQ4e", log=said.append)
    assert events == ["stop ds4", ("load", "Flash-oQ4e")]
    running[0] = False                                       # oMLX not running, the model not in the list it last had
    events.clear()
    B._write_json(B._omlx_cache(), [dict(name="Flash-oQ4e", loaded=False)])
    with pytest.raises(SystemExit, match=r"^oMLX has no model 'Flash-oQ4' \(its models: Flash-oQ4e\)$"):
        B.omlx_start("Flash-oQ4", log=said.append)
    assert events == [("omlx", "start")]                # started to look, ds4 untouched
    assert "Flash-oQ4 was not among oMLX's models when it last ran; starting oMLX to look" in said
    running[0] = False
    events.clear()
    B._write_json(B._omlx_cache(), [])                 # a model the old list lacks (it lists chat models only)
    B.omlx_start("Flash-oQ4e", log=said.append)
    assert events == [("omlx", "start"), "stop ds4", ("load", "Flash-oQ4e")]


def test_omlx_stop_restart_load_and_unload_wait_for_a_switch_and_never_run_under_a_gpu_job(home, monkeypatch):
    """Like `ds4 stop`: one at a time with a backend switch, and refused while a GPU job holds the lock (a restart or
    stop would end a job measuring through oMLX), with nothing done."""
    from metalscope import gpulock
    done, switched = [], []

    @__import__("contextlib").contextmanager
    def switching(log=print):
        switched.append(1)
        yield
    monkeypatch.setattr(B, "switching", switching)

    class Admin:
        base_url = "http://omlx"

        def __init__(self, *a, **k):
            pass

        def listening(self):
            return True

        def wait(self, up, timeout):
            return True
    monkeypatch.setattr(B, "_admin", lambda: Admin())
    monkeypatch.setattr(B.omlx_admin, "Admin", Admin)
    monkeypatch.setattr(B.subprocess, "run", lambda cmd, **k: done.append(cmd[1]))
    for act in ("restart", "load", "unload"):
        monkeypatch.setattr(B.omlx_admin, act, lambda *a, act=act, **k: done.append(act))
    words = (["omlx", "stop"], ["omlx", "restart"], ["omlx", "load", "m"], ["omlx", "unload", "m"])
    lock = gpulock.GpuLock(gpulock.Holder("learning toasty", stage="presence")).acquire()
    try:
        for w in words:
            with pytest.raises(SystemExit, match="now: learning toasty: presence; try again when it is done"):
                cli.main(w)
    finally:
        lock.release()
    assert done == [] and len(switched) == 4
    for w in words:
        cli.main(w)
    assert done == ["stop", "restart", "load", "unload"] and len(switched) == 8


def test_no_model_is_taken_away_while_a_job_runs(home, monkeypatch, tmp_path):
    """A calibration talks to the served model and holds no GPU lock: every command that unloads, stops, restarts or
    switches a model on either backend is refused while any job runs, naming it and how to stop it, with nothing
    done; once it has stopped they go ahead."""
    import os
    import subprocess
    import sys

    from metalscope.learn import state
    from tests.test_learn import JOB_DRIVER, NOWHERE, _until
    driver = tmp_path / "driver.py"
    driver.write_text(JOB_DRIVER)
    proc = subprocess.Popen([sys.executable, str(driver), "concept", "calibrate", "--url", NOWHERE, "joyful", "m"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    done = []

    class Admin:                                                              # oMLX answering, m not loaded
        base_url = "http://omlx"

        def __init__(self, *a, **k):
            pass

        def listening(self):
            return True

        def answering(self):
            return True

        def wait(self, up, timeout):
            return True

        def admin_models(self):
            return [dict(id="m", loaded=False)]
    monkeypatch.setattr(B, "_admin", lambda: Admin())
    monkeypatch.setattr(B.omlx_admin, "Admin", Admin)
    for act in ("restart", "load", "unload"):
        monkeypatch.setattr(B.omlx_admin, act, lambda *a, act=act, **k: done.append(act))
    monkeypatch.setattr(B.subprocess, "run", lambda cmd, **k: done.append(cmd[1]))
    monkeypatch.setattr(B, "_serve_pids", lambda: [4242])                    # a ds4 serve runs
    monkeypatch.setattr(B, "_ds4_servers", lambda: [])
    monkeypatch.setattr(B, "_alive", lambda pid: False)
    monkeypatch.setattr(B, "_signal", lambda pids, sig: done.append("signal"))
    monkeypatch.setattr(B, "ds4_serving", lambda: dict(pid=4242, name="Qwen3.8-Flash-Next-Q4-ds4", ctx=None, ready=True))
    monkeypatch.setattr(B, "free_omlx", lambda log=print: done.append("free oMLX"))
    try:
        _until(lambda: state.list_jobs() or proc.poll() is not None)
        (st,) = state.list_jobs()
        assert st["outcome"] is None, (st, proc.poll())
        why = f"now: a job is running: calibrating joyful (/scope jobs); wait for it or stop it: metalscope jobs cancel {st['job']}"
        for words, act in ((["omlx", "stop"], "stop oMLX"), (["omlx", "restart"], "restart oMLX"),
                           (["omlx", "load", "m"], "load m"), (["omlx", "unload", "m"], "unload m"),
                           (["omlx", "m"], "switch backends"), (["ds4", "stop"], "switch backends"),
                           (["ds4", "V41-Q2-ds4", "--no-wait"], "switch backends")):
            with pytest.raises(SystemExit) as exc:
                cli.main(words)
            assert str(exc.value) == f"cannot {act} {why}", words
        monkeypatch.setattr(B, "_serve_pids", lambda: [os.getpid()])          # `ds4 serve` itself, by hand
        with pytest.raises(SystemExit) as exc:
            B.free_for_ds4("Q-ds4")
        assert str(exc.value) == f"cannot serve Q-ds4 {why}"
        assert done == []
        assert state.cancel(st["job"])
        assert proc.wait(timeout=30) != 0
    finally:
        proc.kill()
        proc.wait()
    cli.main(["omlx", "unload", "m"])
    B.free_for_ds4("Q-ds4").release()
    assert done == ["unload", "free oMLX"]


def test_a_failed_ds4_load_shows_ds4_servers_own_error_first(home):
    """ds4-server refusing an option prints its error (after its log's timestamp), then its whole usage text: the
    error comes first, then the log's last lines; a log whose last lines hold the reason is shown as it ends."""
    d = B._ds4_home()
    d.mkdir(parents=True)
    usage = [f"  --option-{i} N   what option {i} does" for i in range(66)]
    (d / "serve.log").write_text("starting ds4-server\nds4-server exited (2)\n")
    for said in ("1120 03:14:15 ds4-server: unknown option: --bad-option", "ds4-server: unknown option: --bad-option"):
        (d / "ds4-server.log").write_text("\n".join(["1120 03:14:14 ds4-server: loading", said,
                                                     "usage: ds4-server [options]", *usage]) + "\n")
        assert B._why_stopped() == (f"ds4-server ({d / 'ds4-server.log'}):\n  {said}\n"
                                    f"  {usage[-3]}\n  {usage[-2]}\n  {usage[-1]}\n"
                                    f"`ds4 serve` ({d / 'serve.log'}):\n  starting ds4-server\n  ds4-server exited (2)")
    (d / "ds4-server.log").write_text("ds4-server: unknown option: --x\nloading\nout of memory\n")
    assert B._why_stopped().startswith(f"ds4-server ({d / 'ds4-server.log'}):\n"
                                       "  ds4-server: unknown option: --x\n  loading\n  out of memory\n`ds4 serve`")


def test_ds4_stop_stops_only_the_ds4_server_metalscope_started(home, monkeypatch, capsys):
    """serving.json names serve's pid and its ds4-server's: `ds4 stop` signals those, never another ds4-server on the
    machine (one started by hand), and says so when nothing of metalscope's was running."""
    import signal
    alive = {100, 200, 300}                 # serve, its ds4-server, a ds4-server of someone else's
    commands = {100: "/v/bin/python -m metalscope.cli ds4 serve /m/q.gguf", 200: "/b/ds4-server -m /m/q.gguf --port 8010",
                300: "/elsewhere/ds4-server -m /m/other.gguf --port 9010"}
    signalled = []

    def sig(pids, s):
        if pids:
            signalled.append((sorted(pids), s))
        if s == signal.SIGINT and 100 in pids:     # serve stops its ds4-server and exits
            alive.difference_update({100, 200})
        if s in (signal.SIGTERM, signal.SIGKILL):
            alive.difference_update(pids)
    monkeypatch.setattr(B, "_alive", lambda pid: pid in alive)
    monkeypatch.setattr(B, "_command", lambda pid: commands.get(pid, "") if pid in alive else "")
    monkeypatch.setattr(B, "_pgrep", lambda *a: [p for p in ((200, 300, 400) if a[0] == "-x" else (100,)) if p in alive])
    monkeypatch.setattr(B, "_signal", sig)
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    quiet = lambda s: None
    B.write_serving("Q-ds4", 262144, pid=100, server=200)
    assert B._ds4_servers() == [200] and B.other_ds4_servers() == [300]
    assert B.ds4_stop(log=quiet) is True and signalled == [([100], signal.SIGINT)] and 300 in alive
    # serve killed (its record left), its ds4-server still running: that one is stopped, still not the other.
    alive.add(200)
    B.write_serving("Q-ds4", 262144, pid=100, server=200)
    signalled.clear()
    assert B.ds4_stop(log=quiet) is True and signalled == [([200], signal.SIGTERM)] and alive == {300}
    # Nothing of metalscope's: nothing signalled, and the CLI says what it left alone.
    signalled.clear()
    cli.main(["ds4", "stop"])
    assert signalled == [] and alive == {300}
    assert capsys.readouterr().out == "metalscope's ds4 was not running (left alone: ds4-server pid 300, not started by metalscope)\n"
    # An older serve's ds4-server, left running with no pid recorded: metalscope's own build's, so it is stopped.
    from metalscope.ds4 import build
    commands[400] = f"{(build.build_dir().resolve() / 'ds4-server')} -m /m/q.gguf --port 8010"
    alive.add(400)
    assert B._ds4_servers() == [400] and B.other_ds4_servers() == [300]
    assert B.ds4_stop(log=quiet) is True and signalled == [([400], signal.SIGTERM)] and alive == {300}


def test_ds4_stop_kills_a_ds4_server_that_outlives_sigterm(home, monkeypatch):
    """serve gone, its ds4-server ignoring SIGTERM until the timeout: SIGKILL, and the stop says so."""
    import signal
    alive, signalled, said = {200}, [], []

    def sig(pids, s):
        signalled.append(s) if pids else None
        if s == signal.SIGKILL:
            alive.difference_update(pids)
    monkeypatch.setattr(B, "_serve_pids", lambda: [])
    monkeypatch.setattr(B, "_ds4_servers", lambda: [200])
    monkeypatch.setattr(B, "_alive", lambda pid: pid in alive)
    monkeypatch.setattr(B, "_is_ds4_server", lambda pid: pid in alive)
    monkeypatch.setattr(B, "_signal", sig)
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    assert B.ds4_stop(log=said.append, timeout=0.05) is True
    assert signal.SIGTERM in signalled and signalled[-1] == signal.SIGKILL and alive == set()
    assert said == ["ds4 stopped (ds4-server killed)"]


def test_starting_the_served_ds4_model_with_other_options_restarts_it(home, monkeypatch):
    _, qwen, _ = home
    B.register("Q-ds4", qwen, ["--mtp"])
    events, started = [], []
    monkeypatch.setattr(B, "_ds4_stop", lambda log=print, timeout=90: events.append("stop ds4") or True)

    class Proc:
        pid = 4242

        def __init__(self, cmd, **kw):
            started.append(cmd)
    monkeypatch.setattr(B.subprocess, "Popen", Proc)
    monkeypatch.setattr(B, "_wait_ds4", lambda *a, **k: None)                    # --no-wait's wait for the load
    monkeypatch.setattr(B, "ds4_serving", lambda: dict(pid=1, name="Q-ds4", ctx=262144, ready=True))
    B.ds4_start("Q-ds4", args=["--mtp", "--ctx", "262144"], wait=False, log=lambda s: None)    # the same: kept
    assert started == [] and events == []
    B.ds4_start("Q-ds4", args=["--mtp", "--ssd-streaming"], wait=False, log=lambda s: None)
    assert events == ["stop ds4"] and started[0][started[0].index("--") + 1:] == ["--mtp", "--ssd-streaming", "--ctx", "262144"]
    assert B.ds4_model("Q-ds4")["args"] == ["--mtp"]               # serve remembers the new ones once they load


def test_a_serve_that_dies_early_says_why_from_its_own_log(home, monkeypatch):
    d = B._ds4_home()
    d.mkdir(parents=True, exist_ok=True)
    (d / "serve.log").write_text("Traceback\nSystemExit: Q-ds4 is 43 layers x 4 x 4096, the checkpoint 48 x 4 x 2560\n")
    (d / "ds4-server.log").write_text("an earlier run's last line\n")

    class Dead:
        def poll(self):
            return 1
    monkeypatch.setattr(B, "_proxy_models", lambda url=None: None)
    # ds4-server never started this time: an earlier run's lines are left out.
    with pytest.raises(SystemExit, match=r"(?s)stopped while loading Q-ds4.*43 layers") as e:
        B._wait_ds4("Q-ds4", Dead(), log=lambda s: None)
    assert "an earlier run" not in str(e.value)
    # It ran and failed: its lines, then serve's, so the last line is still serve's reason.
    (d / "serve.log").write_text("ds4-server exited with 1 while loading (see its log)\n")
    import os
    (d / "ds4-server.log").write_text("ds4: out of memory\n")
    os.utime(d / "ds4-server.log", (B.time.time() + 60,) * 2)                    # written after serve was spawned
    with pytest.raises(SystemExit, match=r"(?s)out of memory.*`ds4 serve`.*exited with 1 while loading \(see its log\)$"):
        B._wait_ds4("Q-ds4", Dead(), log=lambda s: None)


def test_a_serve_refused_before_loading_says_why_last_even_with_no_wait(home, monkeypatch):
    """oMLX not freed (free_for_ds4 refused inside the background serve): `ds4 MODEL` and `ds4 MODEL --no-wait` both
    fail with serve's reason as the last line (Pi keeps a switch's last lines). --no-wait returns once ds4-server is
    loading (serve has recorded its pid: oMLX was freed)."""
    _, qwen, _ = home
    B.register("Q-ds4", qwen, ["--mtp"])
    d = B._ds4_home()
    d.mkdir(parents=True, exist_ok=True)
    (d / "ds4-server.log").write_text("an earlier run's line\n")
    reason = "could not free oMLX's memory (HTTP 409); unload its model first (metalscope omlx unload ID)"
    monkeypatch.setattr(B, "ds4_serving", lambda: None)
    monkeypatch.setattr(B, "_proxy_models", lambda url=None: None)
    monkeypatch.setattr(B.time, "sleep", lambda s: None)

    class Refused:
        pid = 4242

        def __init__(self, cmd, stdout=None, **kw):
            stdout.write(f"{reason}\n".encode())

        def poll(self):
            return 1
    monkeypatch.setattr(B.subprocess, "Popen", Refused)
    for wait in (True, False):
        with pytest.raises(SystemExit) as e:
            B.ds4_start("Q-ds4", wait=wait, log=lambda s: None)
        assert str(e.value).endswith(reason) and "earlier run" not in str(e.value)

    class Loading:
        pid = 4242

        def __init__(self, cmd, **kw):
            pass

        def poll(self):
            B.write_serving("Q-ds4", 262144, pid=4242, server=4343)                # serve freed oMLX, started ds4-server
            return None
    monkeypatch.setattr(B.subprocess, "Popen", Loading)
    assert B.ds4_start("Q-ds4", wait=False, log=lambda s: None)["name"] == "Q-ds4"


def test_starting_ds4_on_a_build_older_than_metalscope_is_refused_before_anything_stops(home, monkeypatch):
    """`ds4 start` refuses install's out-of-date build up front (build.refuse_stale): oMLX and the ds4 model served
    are left as they are, and no serve is spawned to exit while loading."""
    import json as _json

    from metalscope.ds4 import build
    tmp, qwen, _ = home
    d = tmp / "no-build"
    (d / "metal").mkdir(parents=True)
    (d / "ds4-server").write_text("old")
    (d / build.STAMP).write_text(_json.dumps(dict(commit=build.DS4_COMMIT, patch="an-older-patch")))
    B.register("Q-ds4", qwen, ["--mtp"])
    events = []
    monkeypatch.setattr(B, "free_omlx", lambda log=print: events.append("free oMLX"))
    monkeypatch.setattr(B, "_ds4_stop", lambda log=print, timeout=90: events.append("stop ds4") or True)
    monkeypatch.setattr(B, "gpu_free", lambda: events.append("gpu"))
    class Proc:
        pid = 4242

        def __init__(self, cmd, **kw):
            events.append("spawn")

    monkeypatch.setattr(B.subprocess, "Popen", Proc)
    monkeypatch.setattr(B, "_wait_ds4", lambda *a, **k: None)                    # --no-wait's wait for the load
    monkeypatch.setattr(B, "ds4_serving", lambda: dict(pid=1, name="Other-ds4", ctx=4096, ready=True))
    with pytest.raises(SystemExit, match="older than this metalscope: run metalscope ds4 install, then start again"):
        B.ds4_start("Q-ds4", wait=False, log=lambda s: None)
    assert events == []
    (d / build.STAMP).write_text(_json.dumps(dict(commit=build.DS4_COMMIT, patch=build.patch_id())))
    B.ds4_start("Q-ds4", wait=False, log=lambda s: None)
    assert events == ["gpu", "stop ds4", "spawn"]


def test_ds4_without_its_extra_is_refused_in_one_line_before_anything_is_stopped(home, monkeypatch, capsys):
    """A plain install (no [ds4] extra): `ds4 start` and `ds4 serve` say which install line adds it, before oMLX's
    model is unloaded or a ds4 model stopped, not a ModuleNotFoundError from the proxy's imports."""
    import importlib.util
    tmp, qwen, _ = home
    B.register("Q-ds4", qwen, ["--mtp"])
    events = []
    monkeypatch.setattr(B, "free_omlx", lambda log=print: events.append("free oMLX"))
    monkeypatch.setattr(B, "_ds4_stop", lambda log=print, timeout=90: events.append("stop ds4") or True)
    monkeypatch.setattr(B, "ds4_serving", lambda: dict(pid=1, name="Other-ds4", ctx=4096, ready=True))
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name in ("fastapi", "uvicorn") else real(name, *a))
    want = r'^ds4 needs metalscope\'s \[ds4\] extra \(missing fastapi, uvicorn\): uv tool install --reinstall "metalscope\[ds4\] @ '
    with pytest.raises(SystemExit, match=want):
        B.ds4_start("Q-ds4", wait=False, log=lambda s: None)
    with pytest.raises(SystemExit, match=want):
        cli.main(["ds4", "Q-ds4"])
    from metalscope.ds4 import proxy
    monkeypatch.setattr(proxy, "serve", lambda *a: events.append("serve"))
    with pytest.raises(SystemExit, match=want):
        cli.main(["ds4", "serve", str(qwen)])
    assert events == []


def test_a_taken_port_is_named_before_omlx_is_freed_or_a_model_loads(home, monkeypatch, real_ports):
    """The proxy's port (8009) or ds4-server's (8010) held by something else: `ds4 start` says which port and what
    holds it before oMLX's model is unloaded, and `ds4 serve` before ds4-server starts loading (minutes for 80-94 GB),
    not with uvicorn's Errno 48 once the model has loaded. Only ports this test binds are checked."""
    import os
    import socket

    from metalscope.ds4 import proxy
    tmp, qwen, _ = home
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen()
    port = held.getsockname()[1]
    with socket.socket() as s:                   # a port that was free a moment ago (bound, then let go)
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    try:
        assert B.port_taken(port) and not B.port_taken(free)
        holder = B.port_holder(port)
        assert holder == "" or holder.endswith(f"(pid {os.getpid()})")        # lsof: this test's Python
        B.register("Q-ds4", qwen, ["--mtp"])
        events = []
        monkeypatch.setattr(B, "free_omlx", lambda log=print: events.append("free oMLX"))
        monkeypatch.setattr(B, "_ds4_stop", lambda log=print, timeout=90: events.append("stop ds4") or True)
        monkeypatch.setattr(B, "ds4_serving", lambda: None)
        monkeypatch.setattr(B.client, "DS4_URL", f"http://127.0.0.1:{free}")
        monkeypatch.setattr(B, "DS4_SERVER_PORT", port)
        with pytest.raises(SystemExit, match=rf"^port {port} is taken( by .+ \(pid {os.getpid()}\))?; ds4-server needs it$"):
            B.ds4_start("Q-ds4", wait=False, log=lambda s: None)
        assert events == []
        monkeypatch.setattr(proxy, "make_served", lambda *a: events.append("load"))
        monkeypatch.setattr(proxy, "start_ds4", lambda *a: events.append("ds4-server"))
        with pytest.raises(SystemExit, match=rf"^port {port} is taken.*; the ds4 proxy \(--port\) needs it$"):
            proxy.serve(qwen, None, port, tmp, free, [])
        with pytest.raises(SystemExit, match=rf"^port {port} is taken.*; ds4-server \(--ds4-port\) needs it$"):
            proxy.serve(qwen, None, free, tmp, port, [])
        assert events == []
    finally:
        held.close()


def _serve_world(monkeypatch, events, fail=None):
    """proxy.serve with oMLX, ds4-server and uvicorn faked: each step records what holds the GPU lock as it runs, and
    an oMLX load is tried while ds4-server loads. ``fail``: what the load raises instead of answering."""
    import threading

    import uvicorn

    from metalscope import gpulock
    from metalscope.ds4 import proxy
    from tests.test_learn import _Unavailable, _fake_pool, _run

    def held():
        h = gpulock.held()
        return h.what if h else None
    pool = _fake_pool()
    done = threading.Event()

    class Proc:
        pid = 31337

        def poll(self):
            return 0 if done.is_set() or fail else None

        def wait(self, timeout=None):
            done.wait()
            return 0

    def loading(*a):
        events.append(("ds4 loading", held()))
        with pytest.raises(_Unavailable, match="'big' cannot load now: ds4 loading Q-ds4; models load again"):
            _run(pool.get_engine("big"))
        if fail:
            raise fail

    class Server:
        should_exit = False

        def __init__(self, cfg):
            pass

        def run(self):
            events.append(("proxy serves", held()))
            self.should_exit = True
            done.set()

        def handle_exit(self, sig, frame):
            self.should_exit = True
    monkeypatch.setattr(B, "_serve_pids", lambda: [])
    monkeypatch.setattr(B, "free_omlx", lambda log=print: events.append(("free oMLX", held())))
    monkeypatch.setattr(proxy, "make_served", lambda gguf, name, up: events.append("GGUF read") or
                        __import__("types").SimpleNamespace(name="Q-ds4"))
    monkeypatch.setattr(proxy, "start_ds4", lambda *a: events.append(("ds4-server", held())) or Proc())
    monkeypatch.setattr(proxy, "wait_ready", loading)
    monkeypatch.setattr(proxy, "build_app", lambda served: None)
    monkeypatch.setattr(uvicorn, "Config", lambda *a, **k: None)
    monkeypatch.setattr(uvicorn, "Server", Server)
    return proxy, pool


def test_ds4_serve_frees_omlx_under_the_gpu_lock_and_holds_it_until_ds4_answers(home, monkeypatch):
    """`ds4 serve GGUF` (by hand, or spawned by `ds4 start`): the GGUF read, then the GPU lock taken and oMLX's models
    unloaded under it, then ds4-server started; while it loads an oMLX load is refused with the lock's message; once
    it answers the lock is released (oMLX's own memory check sees ds4's model from then on)."""
    from metalscope import gpulock
    tmp, qwen, _ = home
    events = []
    proxy, pool = _serve_world(monkeypatch, events)
    proxy.serve(qwen, None, 9999, tmp, 9998, ["--mtp", "--ctx", "8192"])
    assert events == ["GGUF read", ("free oMLX", "ds4 loading Q-ds4"), ("ds4-server", "ds4 loading Q-ds4"),
                      ("ds4 loading", "ds4 loading Q-ds4"), ("proxy serves", None)]
    assert pool.loads == [] and gpulock.unclean() is None
    assert json.loads((B._ds4_home() / "models.json").read_text())["Q-ds4"]["args"] == ["--mtp"]   # loaded: remembered


@pytest.mark.parametrize("fail", [SystemExit("ds4-server exited with 1 while loading"), KeyboardInterrupt()])
def test_ds4_serve_releases_the_gpu_lock_when_the_load_fails_or_is_stopped(home, monkeypatch, fail):
    """ds4-server failing to load, or Ctrl-C (`ds4 stop` sends serve SIGINT) while it loads: ds4-server stopped, then
    the lock released cleanly; oMLX loads again."""
    from metalscope import gpulock
    from tests.test_learn import _run
    tmp, qwen, _ = home
    events = []
    proxy, pool = _serve_world(monkeypatch, events, fail=fail)
    B.register("Q-ds4", qwen, ["--mtp"])
    with pytest.raises(type(fail)):
        proxy.serve(qwen, None, 9999, tmp, 9998, ["--bad-option"])
    assert events[-1] == ("ds4 loading", "ds4 loading Q-ds4")
    assert gpulock.held() is None and gpulock.unclean() is None
    assert _run(pool.get_engine("big")) == "engine"
    assert not (B._ds4_home() / "serving.json").exists()
    # Options it failed to load with are not remembered: the next start uses the ones that worked.
    assert json.loads((B._ds4_home() / "models.json").read_text())["Q-ds4"]["args"] == ["--mtp"]


def test_ds4_serve_releases_the_gpu_lock_when_ds4_server_cannot_start(home, monkeypatch):
    from metalscope import gpulock
    tmp, qwen, _ = home
    events = []
    proxy, _ = _serve_world(monkeypatch, events)
    monkeypatch.setattr(proxy, "start_ds4", lambda *a: (_ for _ in ()).throw(SystemExit("ds4-server not found")))
    with pytest.raises(SystemExit, match="not found"):
        proxy.serve(qwen, None, 9999, tmp, 9998, [])
    assert gpulock.held() is None and not (B._ds4_home() / "serving.json").exists()


def test_room_for_ds4_is_refused_while_another_ds4_or_a_gpu_job_runs_and_never_kept_on_failure(home, monkeypatch):
    """free_for_ds4 refuses before oMLX is touched while another ds4 runs (on other ports too) or a GPU job holds the
    lock; a failed unload gives the lock back."""
    import os

    from metalscope import gpulock
    freed = []
    monkeypatch.setattr(B, "free_omlx", lambda log=print: freed.append(1))
    monkeypatch.setattr(B, "_serve_pids", lambda: [os.getpid(), 999])          # itself (its record) and another
    with pytest.raises(SystemExit, match=r"^ds4 already runs \(pid 999\): metalscope ds4 stop$"):
        B.free_for_ds4("Q-ds4")
    monkeypatch.setattr(B, "_serve_pids", lambda: [os.getpid()])
    monkeypatch.setattr(gpulock.GpuLock.acquire, "__defaults__", (0.2,))
    lock = gpulock.GpuLock(gpulock.Holder("learning toasty", stage="presence"), gpulock.lock_path())
    lock.acquire()
    try:
        with pytest.raises(SystemExit, match="^cannot serve Q-ds4 now: learning toasty: presence; try again"):
            B.free_for_ds4("Q-ds4")
    finally:
        lock.release()
    assert freed == []

    def refuse(log=print):
        raise SystemExit("could not free oMLX's memory")
    monkeypatch.setattr(B, "free_omlx", refuse)
    with pytest.raises(SystemExit, match="could not free"):
        B.free_for_ds4("Q-ds4")
    assert gpulock.held() is None
    monkeypatch.setattr(B, "free_omlx", lambda log=print: None)
    B.free_for_ds4("Q-ds4").release()                                          # nothing left holding it


def test_a_switch_may_stop_ds4_while_it_loads_but_not_a_gpu_job(home, monkeypatch):
    """ds4's load holds the GPU lock, but `ds4 stop`, `omlx MODEL` and `ds4 OTHER` stop it (that frees memory);
    another holder (a GPU job, an engram inspection inside serve) still refuses them."""
    import os
    import signal

    from metalscope import gpulock
    signalled = []
    monkeypatch.setattr(B, "_serve_pids", lambda: [os.getpid()])               # this process plays the loading serve
    monkeypatch.setattr(B, "_ds4_servers", lambda: [])
    monkeypatch.setattr(B, "_alive", lambda pid: False)
    monkeypatch.setattr(B, "_signal", lambda pids, sig: signalled.append((list(pids), sig)))
    lock = gpulock.GpuLock(gpulock.Holder(f"{B.DS4_LOADING} V41-ds4")).acquire()
    try:
        B.gpu_free()
        assert B.ds4_stop(log=lambda s: None) is True and signalled[0] == ([os.getpid()], signal.SIGINT)
        monkeypatch.setattr(B, "_serve_pids", lambda: [])                       # its serve gone, the lock not
        with pytest.raises(SystemExit, match="cannot switch backends now: ds4 loading V41-ds4; try again"):
            B.gpu_free()
    finally:
        lock.release()
    monkeypatch.setattr(B, "_serve_pids", lambda: [os.getpid()])
    lock = gpulock.GpuLock(gpulock.Holder("inspecting the engram")).acquire()
    try:
        with pytest.raises(SystemExit, match="cannot switch backends now: inspecting the engram"):
            B.gpu_free()
    finally:
        lock.release()


def test_ds4_serve_refused_or_interrupted_while_freeing_omlx_leaves_no_lock_and_no_record(home, monkeypatch):
    """Ctrl-C while oMLX's models unload, or a refusal (another ds4 runs): the lock given back, no serving.json
    written, no model registered, ds4-server never started."""
    from metalscope import gpulock
    tmp, qwen, _ = home
    events = []
    proxy, _ = _serve_world(monkeypatch, events)

    def interrupted(log=print):
        raise KeyboardInterrupt
    monkeypatch.setattr(B, "free_omlx", interrupted)
    with pytest.raises(KeyboardInterrupt):
        proxy.serve(qwen, None, 9999, tmp, 9998, [])
    assert gpulock.held() is None and not (B._ds4_home() / "serving.json").exists()
    monkeypatch.setattr(B, "_serve_pids", lambda: [999])
    with pytest.raises(SystemExit, match="ds4 already runs"):
        proxy.serve(qwen, None, 9999, tmp, 9998, [])
    assert gpulock.held() is None and not (B._ds4_home() / "serving.json").exists()
    assert events == ["GGUF read", "GGUF read"]
    assert not (B._ds4_home() / "models.json").exists()          # a refused serve registers no model


SERVE = """
import sys, types
from pathlib import Path
from metalscope import backends
from metalscope.ds4 import proxy
backends._serve_pids = lambda: []
backends._ds4_servers = lambda: []
backends.free_omlx = lambda log=print: None
proxy.make_served = lambda gguf, name, up: types.SimpleNamespace(name="T-ds4")
proxy.serve(Path(sys.argv[1]), None, int(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), [])
"""


@pytest.mark.parametrize("sig", ["SIGINT", "SIGTERM"])
def test_a_signal_while_ds4_loads_stops_ds4_server_and_gives_the_lock_back(tmp_path, sig, real_ports):
    """A real `ds4 serve` process (oMLX and the GGUF faked, ds4-server a sleep, its own data home and spare ports)
    signalled while ds4-server loads: ds4-server stopped, the lock released cleanly, serving.json removed."""
    import os
    import signal
    import socket
    import subprocess
    import sys
    import time

    import metalscope
    from metalscope import config, gpulock
    assert config.home() != Path.home() / ".metalscope"
    ports = []
    for _ in range(2):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            ports.append(s.getsockname()[1])
    build = tmp_path / "ds4"
    build.mkdir()
    (build / "ds4-server").write_text("#!/bin/sh\nexec sleep 600\n")
    (build / "ds4-server").chmod(0o755)
    (tmp_path / "m.gguf").write_text("")
    env = dict(os.environ, PYTHONPATH=str(Path(metalscope.__file__).parents[1]))
    child = subprocess.Popen([sys.executable, "-c", SERVE, str(tmp_path / "m.gguf"), str(ports[0]), str(build),
                              str(ports[1])], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    record = config.home() / "ds4" / "serving.json"
    try:
        end = time.time() + 30
        while not config.read_json(record, {}).get("server") and time.time() < end and child.poll() is None:
            time.sleep(0.1)
        server = config.read_json(record, {}).get("server")
        assert server and gpulock.held().what == "ds4 loading T-ds4", child.stdout.read1().decode()
        child.send_signal(getattr(signal, sig))
        child.wait(timeout=90)
    finally:
        if child.poll() is None:
            child.kill()
    with pytest.raises(ProcessLookupError):
        os.kill(server, 0)
    assert gpulock.held() is None and json.loads(gpulock.lock_path().read_text()).get("released") is True
    assert not record.exists()

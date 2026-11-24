"""The command line without a model: listings, stored profiles, and the
argument checks that must refuse before a checkpoint is loaded."""

import argparse
import re
from pathlib import Path

import pytest

from metalscope import cli, config, doctor as D
from metalscope.engines import mlx
from metalscope.omlx import install as omlx_install
from tests.helpers import fake_profile, write_norms


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))

    def no_load(path, loader="mlx_lm"):
        raise AssertionError(f"loaded a model from {path}")
    monkeypatch.setattr(mlx, "load_model", no_load)
    return tmp_path


def test_an_empty_metalscope_home_counts_as_unset(home, monkeypatch):
    from metalscope import config
    assert config.home() == home
    monkeypatch.setenv("METALSCOPE_HOME", "")                    # as Pi reads it (backends.ts metalscopeHome)
    assert config.home() == Path.home() / ".metalscope"


def run(capsys, *argv) -> str:
    cli.main(list(argv))
    return capsys.readouterr().out


def test_two_settings_changed_at_once_both_stay(home, monkeypatch):
    """The server and the command line change settings.json at the same moment: each reads it, changes its key and
    writes it back. Forced to overlap (each read waits), neither change is lost."""
    import threading
    import time
    read = config.read_json

    def slow(path, default=None):
        out = read(path, default)
        if Path(path) == config.settings_path():
            time.sleep(0.3)
        return out
    monkeypatch.setattr(config, "read_json", slow)
    threads = [threading.Thread(target=config.save_settings, kwargs=kw) for kw in (dict(steer=True), dict(trace_auto=False))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    data = read(config.settings_path())
    assert (data["steer"], data["trace_auto"]) == (True, False)


def test_concept_list_shows_shipped_concepts(home, capsys):
    rows = {line.split()[0]: line for line in run(capsys, "concept", "list").splitlines()}
    assert {"fluffy", "stoic", "melancholic"} <= set(rows)
    assert " task " in rows["fluffy"]
    # Under each concept, one line on who wrote its texts.
    lines = run(capsys, "concept", "list").splitlines()
    i = next(i for i, l in enumerate(lines) if l.startswith("fluffy "))
    assert lines[i + 1].strip() == "texts: hand-written by Claude Fable 5.1 (2026-10-05)"


def test_concept_list_ignores_an_old_status_block(home, capsys):
    import json
    from metalscope import models
    d = models.concept_dir("tiny", "toy")               # the model's own texts
    d.mkdir(parents=True)
    (d / "pos.txt").write_text("a\nb")
    (d / "neg.txt").write_text("c\nd")
    (d / "concept.json").write_text(json.dumps(dict(title="Toy", description="t", status=dict(decision="drop", reason="no", date="2026-10-06"))))
    rows = {line.split()[0]: line for line in run(capsys, "concept", "list", "tiny").splitlines()[1:]}
    assert "toy" in rows and " drop " not in rows["toy"]
    assert "toy" not in run(capsys, "concept", "list")          # another model's texts are not the library's


def test_concept_show_prints_the_stored_profile(home, capsys):
    fake_profile(home)
    out = run(capsys, "concept", "show", "toy", "tiny")
    assert out.startswith("Toy on tiny: 3 vs 3 texts, mean position")
    assert "writer: first layer that is significant (p < 0.01) L0" in out
    assert "reader: L1 at strength 0.5" in out
    assert f"  folder {home / 'models' / 'tiny' / 'concepts' / 'toy'}" in out
    assert "  texts: none on this model\n" in out
    assert "bar" not in out and "hold" not in out


def test_concept_delete_moves_the_checkpoints_folder_to_the_trash(home, capsys, monkeypatch):
    from metalscope import models
    from metalscope.learn import state
    fake_profile(home)
    (models.concept_dir("tiny", "toy") / "pos.txt").write_text("a\nb")      # its texts go with it
    models.definition_path("toy").parent.mkdir(parents=True)
    models.definition_path("toy").write_text('{"name": "toy", "description": "t"}')   # its definition stays
    models.check_shape("tiny-ds4", models.shape_of(2, 1, 8))
    models.link("tiny-ds4", "tiny")                                        # a second served copy, calibrated too
    for copy in ("tiny", "tiny-ds4"):
        models.calibration_dir(copy, "toy").mkdir(parents=True)
    where = models.concept_dir("tiny", "toy")
    asked = []
    monkeypatch.setattr("builtins.input", lambda q: asked.append(q) or "n")
    with pytest.raises(SystemExit, match="nothing deleted"):
        cli.main(["concept", "delete", "toy", "tiny-ds4"])
    # The prompt names the checkpoint and every served copy whose calibration goes.
    assert asked == ["y deletes toy on tiny (its texts, vector, profile, calibrations on tiny, tiny-ds4)? [y/N] "]
    assert where.is_dir()
    monkeypatch.setattr(state, "busy_with", lambda c: "a job is running: calibrating toy (/scope jobs)")
    with pytest.raises(SystemExit, match="calibrating toy"):
        cli.main(["concept", "delete", "toy", "tiny", "--yes"])
    # A job that starts while it asks: checked again after the answer, nothing moved.
    jobs = iter([None, "a job is running: learning toy (/scope jobs)"])
    monkeypatch.setattr(state, "busy_with", lambda c: next(jobs))
    monkeypatch.setattr("builtins.input", lambda q: "y")
    with pytest.raises(SystemExit, match="learning toy"):
        cli.main(["concept", "delete", "toy", "tiny"])
    assert where.is_dir() and not (home / "trash").exists()
    monkeypatch.setattr(state, "busy_with", lambda c: None)
    out = run(capsys, "concept", "delete", "toy", "tiny-ds4", "--yes")     # by a served name: its checkpoint's
    gone = next((home / "trash").iterdir())
    assert out == f"toy deleted on tiny: moved to {gone}\n" and gone.name.endswith("-toy-tiny")
    assert (gone / "profile.json").is_file() and (gone / "pos.txt").is_file() and not where.exists()
    assert models.definition_path("toy").is_file()
    assert run(capsys, "concept", "delete", "toy", "tiny", "--yes").startswith("toy's description deleted: moved to ")
    with pytest.raises(SystemExit, match="toy has nothing on tiny"):
        cli.main(["concept", "delete", "toy", "tiny", "--yes"])


def test_concept_delete_takes_a_description_no_model_has(home, capsys, monkeypatch):
    import json
    from metalscope import models
    models.definition_path("toy").parent.mkdir(parents=True)
    models.definition_path("toy").write_text(json.dumps(dict(name="toy", description="t")))
    for ckpt in ("A", "B"):
        models.concept_dir(ckpt, "toy").mkdir(parents=True)
    # Other models still have it: refused, naming them; the description stays.
    with pytest.raises(SystemExit, match=r"^error: toy has nothing on m \(it is on A, B\)$"):
        cli.main(["concept", "delete", "toy", "m", "--yes"])
    for ckpt in ("A", "B"):
        cli.main(["concept", "delete", "toy", ckpt, "--yes"])
    assert models.definition_path("toy").is_file()                # deleting it on a model keeps the description
    # No model has it: the same command takes the description, asked first.
    asked = []
    monkeypatch.setattr("builtins.input", lambda q: asked.append(q) or "n")
    with pytest.raises(SystemExit, match="nothing deleted"):
        cli.main(["concept", "delete", "toy", "m"])
    assert asked == ["y deletes toy's description (no model has it)? [y/N] "] and models.definition_path("toy").is_file()
    capsys.readouterr()
    out = run(capsys, "concept", "delete", "toy", "m", "--yes")
    gone = next(p for p in (home / "trash").iterdir() if p.name.endswith("-toy-definition"))
    assert out == f"toy's description deleted: moved to {gone}\n" and (gone / "concept.json").is_file()
    assert not models.definition_path("toy").parent.exists()
    # A shipped name with nothing on the model, or nothing anywhere: as before.
    for name in ("fluffy", "toy"):
        with pytest.raises(SystemExit, match=f"^error: {name} has nothing on m$"):
            cli.main(["concept", "delete", name, "m", "--yes"])


def test_concept_delete_asks_about_what_the_folder_holds(home, monkeypatch):
    from metalscope import models
    d = models.concept_dir("tiny", "toy")
    d.mkdir(parents=True)
    (d / "concept.json").write_text("{}")
    asked = []
    monkeypatch.setattr("builtins.input", lambda q: asked.append(q) or "n")
    for add in ("", "vector.safetensors", "profile.json"):
        if add:
            (d / add).write_text("x")
        with pytest.raises(SystemExit, match="nothing deleted"):
            cli.main(["concept", "delete", "toy", "tiny"])
    (d / "concept.json").unlink()
    (d / "tiny").mkdir()
    with pytest.raises(SystemExit, match="nothing deleted"):
        cli.main(["concept", "delete", "toy", "tiny"])
    assert asked == ["y deletes toy on tiny (its texts)? [y/N] ", "y deletes toy on tiny (its texts, vector)? [y/N] ",
                     "y deletes toy on tiny (its texts, vector, profile)? [y/N] ",
                     "y deletes toy on tiny (its vector, profile, calibrations on tiny)? [y/N] "]


def test_concept_list_says_how_many_more_the_models_hold(home, capsys):
    from metalscope import models
    assert "more on models" not in run(capsys, "concept", "list")
    for ckpt, name in (("A", "pulled"), ("B", "pulled"), ("B", "mine")):
        d = models.concept_dir(ckpt, name)
        d.mkdir(parents=True)
        (d / "profile.json").write_text("{}")
    models.concept_dir("A", "tok1").mkdir(parents=True)           # a word vector's folder: not a concept
    (models.concept_dir("A", "tok1") / "vector.safetensors").write_text("v")
    models.concept_dir("A", "fluffy").mkdir(parents=True)         # listed already
    (models.concept_dir("A", "fluffy") / "profile.json").write_text("{}")
    assert run(capsys, "concept", "list").splitlines()[-1] == "(2 more on models: metalscope concept list NAME)"
    assert "more on models" not in run(capsys, "concept", "list", "A")


def test_concept_delete_refuses_a_name_that_is_not_one_folder(home):
    from metalscope import models
    fake_profile(home)
    models.models_dir().joinpath("M").mkdir(parents=True)
    before = sorted(str(p) for p in home.rglob("*"))
    for concept, name, why in [("../models", "M", "bad concept name"), ("..", "tiny", "bad concept name"),
                               (".", "tiny", "bad concept name"), ("toy", ".", "bad checkpoint name"),
                               ("toy", "..", "bad checkpoint name"), ("toy", "a/..", "bad checkpoint name"),
                               ("toy", ".hidden", "bad checkpoint name")]:
        with pytest.raises(SystemExit, match=why):
            cli.main(["concept", "delete", concept, name, "--yes"])
    assert sorted(str(p) for p in home.rglob("*")) == before                # nothing moved, no trash


def test_a_bare_name_in_the_steer_header_takes_the_calibrated_dose(home):
    # X-Scope-Steer: toy steers at what /steer toy would: the calibration's dose at its layer; none calibrated is a 400.
    import json
    import pytest
    from metalscope import models, steer
    fake_profile(home)
    with pytest.raises(ValueError, match="no calibrated dose on tiny"):
        steer.with_defaults(steer.parse_header("toy"), "tiny")
    prof = models.profile_path("tiny", "toy")
    cal = dict(version=4, default_dose=0.3, good=15, turns=18, layer=1, profile_mtime=prof.stat().st_mtime)
    models.calibration_path("tiny", "toy").parent.mkdir(parents=True)
    models.calibration_path("tiny", "toy").write_text(json.dumps(cal))
    spec = steer.with_defaults(steer.parse_header("toy;positions=decode"), "tiny")
    assert [(p.name, p.strength, p.layers) for p in spec.parts] == [("toy", 0.3, (1,))] and spec.positions == "decode"
    assert spec.label == "toy@0.3"
    with pytest.raises(ValueError, match="calibrated layer"):
        steer.with_defaults(steer.parse_header("toy;layers=2"), "tiny")


def test_concept_show_and_list_give_the_served_names_calibration(home, capsys):
    import json
    from metalscope import models
    fake_profile(home)
    models.link("tiny-ds4", "tiny")
    prof = models.profile_path("tiny", "toy")
    cal = dict(version=4, default_dose=0.3, good=6, turns=18, layer=1, profile_mtime=prof.stat().st_mtime)
    models.calibration_path("tiny-ds4", "toy").parent.mkdir(parents=True)
    models.calibration_path("tiny-ds4", "toy").write_text(json.dumps(cal))
    said = "6/18 good @0.3"
    out = run(capsys, "concept", "show", "toy", "tiny")
    assert f"  in chat on tiny: not calibrated ({models.calibration_path('tiny', 'toy')})" in out
    assert f"  in chat on tiny-ds4: {said}" in out
    out = run(capsys, "concept", "show", "toy", "tiny-ds4")
    assert f"in chat on tiny-ds4: {said} (" in out and "in chat on tiny:" not in out
    models.calibration_path("tiny-ds4", "toy").write_text(json.dumps(dict(cal, profile_mtime=1.0)))
    assert "in chat on tiny-ds4: out of date (the profile or the dose units changed after it)" in run(capsys, "concept", "show", "toy", "tiny-ds4")
    models.calibration_path("tiny-ds4", "toy").write_text(json.dumps(dict(cal, version=3)))     # an older one: none
    assert "in chat on tiny-ds4: not calibrated (" in run(capsys, "concept", "show", "toy", "tiny-ds4")
    models.calibration_path("tiny-ds4", "toy").write_text(json.dumps(cal))
    d = models.concept_dir("tiny", "toy")                          # its texts, on the model
    (d / "concept.json").write_text(json.dumps(dict(title="Toy", description="t")))
    (d / "pos.txt").write_text("a\n")
    lines = run(capsys, "concept", "list", "tiny-ds4").splitlines()
    assert lines[0] == "profiles of checkpoint tiny; chat calibrations on tiny-ds4"
    rows = {line.split()[0]: line for line in lines[1:]}
    assert "6/18 good @0.3" in rows["toy"] and "no profile" in rows["fluffy"]
    assert "not calibrated" in {line.split()[0]: line for line in run(capsys, "concept", "list", "tiny").splitlines()[1:]}["toy"]
    (d / "concept.json").write_text(json.dumps(dict(title="Toy", description="t", learned=dict(writer="Qwen3.6", date="2026-10-27"))))
    assert "\n  texts: written by Qwen3.6 (2026-10-27) (instructions not kept)\n" in run(capsys, "concept", "show", "toy", "tiny")


def test_concept_show_without_a_profile_says_how_to_make_one(home):
    with pytest.raises(SystemExit, match="no profile at .*metalscope concept learn toy tiny --description TEXT"):
        cli.main(["concept", "show", "toy", "tiny"])
    with pytest.raises(SystemExit, match="no profile at .*; learn it: metalscope concept learn fluffy tiny$"):
        cli.main(["concept", "show", "fluffy", "tiny"])


def test_steer_list(home, capsys):
    assert run(capsys, "vector", "list", "tiny").startswith("no steering vectors for tiny under ")
    fake_profile(home)
    out = run(capsys, "vector", "list", "tiny")
    assert out.split()[:4] == ["toy", "layers", "[0,", "1,"]


def test_steer_norms_without_traces_says_to_trace_first(home):
    with pytest.raises(SystemExit, match=r"no saved unsteered traces of tiny .* trace a few replies first \(metalscope on\)"):
        cli.main(["vector", "norms", "tiny"])


def _parser(monkeypatch) -> argparse.ArgumentParser:
    """The parser main() builds, caught before it parses anything."""
    class Built(Exception):
        pass

    def catch(self, *a, **k):
        raise Built(self)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", catch)
    with pytest.raises(Built) as caught:
        cli.main([])
    return caught.value.args[0]


def _commands(parser, prefix=()):
    """(command words, option strings) for every leaf command."""
    subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    opts = [o for a in parser._actions for o in a.option_strings if o not in ("-h", "--help", "--version")]   # the root's own
    if not subs:
        yield prefix, opts
        return
    for name, sp in subs[0].choices.items():
        for words, more in _commands(sp, prefix + (name,)):
            yield words, opts + more


COMMANDS = Path(__file__).resolve().parents[1] / "docs" / "commands.md"


@pytest.mark.parametrize("where", ["docstring", "docs/commands.md"])
def test_the_command_reference_lists_every_command_and_flag(monkeypatch, where):
    # Both references against the real parser: the module docstring (--help, the overview) lists every command;
    # docs/commands.md's command list every command and its flags (each command's own --help has them too).
    text = cli.__doc__ if where == "docstring" else COMMANDS.read_text()
    lines = [l.strip() for l in text.splitlines()]
    for words, opts in _commands(_parser(monkeypatch)):
        # The command's line: "metalscope" then its words in order ("a|b" alternatives, options in between).
        def matches(line):
            toks = [t.split("|") for t in line.split()[1:]]
            i = 0
            for w in words:
                while i < len(toks) and w not in toks[i]:
                    i += 1
                if i == len(toks):
                    return False
                i += 1
            return True
        hits = [i for i, l in enumerate(lines) if l.startswith("metalscope ") and matches(l)]
        assert hits, f"metalscope {' '.join(words)} is not in the {where}"
        # Its flags: on one of its lines (the reference's, not an example's) or the continuation lines under it.
        def block(h):
            end = next((j for j in range(h + 1, len(lines)) if lines[j].startswith("metalscope ") or not lines[j]), len(lines))
            return " ".join(lines[h:end])
        if where == "docs/commands.md":
            missing = min(([o for o in opts if o not in block(h)] for h in hits), key=len)
            assert not missing, f"{missing[0]} of metalscope {' '.join(words)} is not in the {where}"


def test_concept_jobs_refuse_bad_input_with_a_message(home):
    with pytest.raises(SystemExit):                                # no --dose: calibration searches the doses itself
        cli.main(["concept", "calibrate", "fluffy", "tiny", "--dose", "0.4"])
    with pytest.raises(SystemExit, match="error: not a job id"):
        cli.main(["concept", "calibrate", "fluffy", "tiny", "--job", "../x"])
    with pytest.raises(SystemExit, match="error: 'list' is a command word"):
        cli.main(["concept", "learn", "list", "tiny", "--description", "d", "--model", "/x"])
    with pytest.raises(SystemExit, match="error: bad vector name"):
        cli.main(["concept", "learn", "../x", "tiny", "--description", "d", "--model", "/x"])


def test_concept_jobs_refuse_a_model_no_backend_serves(home, monkeypatch):
    """learn and calibrate on a name neither oMLX lists nor the ds4 proxy serves: one line saying how to start it,
    before anything falls back to oMLX."""
    from metalscope import backends
    from metalscope.learn import plan as P
    lists = dict(up=True, models=[dict(name="Qwen3.6-35B-A3B-MLX-4bit")])
    serving = dict(name="DeepSeek-V4.1-Flash-ds4-q2")
    monkeypatch.setattr(backends, "omlx_models", lambda: lists)
    monkeypatch.setattr(backends, "ds4_serving", lambda: serving)
    said = "Qwen-gone is not being served: start it with `metalscope omlx Qwen-gone` or `metalscope ds4 Qwen-gone`"
    for argv in (["concept", "learn", "fluffy", "Qwen-gone"], ["concept", "calibrate", "fluffy", "Qwen-gone"]):
        with pytest.raises(SystemExit, match="^error: " + re.escape(said) + "$"):
            cli.main(argv)
    monkeypatch.setattr(P, "pick_engine", lambda *a: (_ for _ in ()).throw(AssertionError("picked an engine")))
    for served in (lists["models"][0]["name"], serving["name"]):     # dose units: the plan goes on to the engine
        write_norms(served, [0, 1])
    with pytest.raises(AssertionError, match="picked an engine"):              # served by oMLX: on to the job
        cli.main(["concept", "learn", "fluffy", "Qwen3.6-35B-A3B-MLX-4bit"])
    with pytest.raises(AssertionError, match="picked an engine"):              # served by the ds4 proxy
        cli.main(["concept", "learn", "fluffy", "DeepSeek-V4.1-Flash-ds4-q2"])
    lists["up"] = False                                                         # oMLX not running: its list is old
    with pytest.raises(SystemExit, match="Qwen3.6-35B-A3B-MLX-4bit is not being served"):
        cli.main(["concept", "calibrate", "fluffy", "Qwen3.6-35B-A3B-MLX-4bit"])


def run_doctor(capsys) -> str:
    """doctor's output; here it exits 1 (no oMLX app: FAIL lines)."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["doctor"])
    assert exc.value.code == 1
    return capsys.readouterr().out


def test_doctor_says_where_the_startup_hook_points(home, tmp_path, monkeypatch, capsys):
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.setattr(omlx_install, "user_site", lambda app: site)
    monkeypatch.setattr(D.client, "candidates", lambda: ["http://127.0.0.1:1"])
    def no_process(*a, **k):                                       # no oMLX app, no Pi
        raise OSError("not here")
    monkeypatch.setattr(D.subprocess, "run", no_process)
    monkeypatch.setattr(D.client.urllib.request, "urlopen", no_process)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    hook = omlx_install.hook_dir()
    hook.mkdir(parents=True)
    (hook / "metalscope").symlink_to(omlx_install.PACKAGE)
    (site / omlx_install.PTH_NAME).write_text(omlx_install.pth_text(omlx_install.DEFAULT_APP))
    out = run_doctor(capsys)
    assert f"  · imports   {omlx_install.PACKAGE}\n" in out and "not this metalscope" not in out
    assert "does not run" not in out and "✗" not in out.replace("✗ app", "")
    other = tmp_path / "other" / "metalscope"
    other.mkdir(parents=True)
    (hook / "metalscope").unlink()
    (hook / "metalscope").symlink_to(other)
    out = run_doctor(capsys)
    assert f"· imports   {other}\n" in out and f"  ! {'':<10}not this metalscope ({omlx_install.PACKAGE})" in out
    assert "does not run" not in out
    (site / omlx_install.PTH_NAME).write_text("# something else\n")
    out = run_doctor(capsys)
    assert f"  ✗ hook      {site / omlx_install.PTH_NAME} · does not run `{omlx_install.PTH_IMPORT}`" in out
    assert "· imports" not in out


def test_jobs_lists_the_last_ten_and_cancel_needs_a_running_job(home, capsys, monkeypatch):
    import json
    from metalscope.learn import state
    assert run(capsys, "jobs").startswith("no jobs yet")
    with pytest.raises(SystemExit, match="no job is running"):
        cli.main(["jobs", "cancel"])
    said = "16/18 good @0.4"           # made again from the numbers: an older job's stored line had a bar's verdict
    cal = dict(version=4, default_dose=0.4, good=16, turns=18, reason="holds up · dose 0.4 · 16/18 good")
    for i, (kind, result) in enumerate([("learn", dict(steerable=False, reason="can't be steered: no item had room")), ("calibrate", dict(calibration=cal))]):
        d = state.job_dir(f"j-{i}")
        d.mkdir(parents=True)
        (d / "state.json").write_text(json.dumps(dict(job=f"j-{i}", kind=kind, name=f"n{i}", phase="done", outcome="done",
                                                      started=1_700_000_000 + i, pid=1, result=result)))
    out = run(capsys, "jobs").splitlines()
    assert out[0] == f"last: calibrate n1: {said}  (j-1)" and out[1] == "last 2:"
    assert out[2].split()[2:5] == ["calibrate", "n1", "16/18"] and out[3].split()[2:6] == ["learn", "n0", "can't", "be"]
    assert run(capsys, "jobs", "cancel", "j-0") == "j-0 is not running\n"
    with pytest.raises(SystemExit, match="error: not a job id"):
        cli.main(["jobs", "cancel", "../x"])
    monkeypatch.setattr(state, "cancel", lambda job: True)
    assert run(capsys, "jobs", "cancel", "j-1") == "stopping j-1\n"                 # a calibration unloads nothing
    assert run(capsys, "jobs", "cancel", "j-0") == "stopping j-0: the model it unloaded is reloaded when it has stopped\n"


def test_the_overview_is_short_and_starts_with_how_to_start(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert out.startswith("usage: metalscope [-h] COMMAND ...")
    assert out.index("Backends") < out.index("Front ends") < out.index("Every day:") < out.index("The J-lens") \
        < out.index("Research")
    assert len(out.splitlines()) <= 60
    assert "--origin" not in out and "--need-gb" not in out and "positional arguments" not in out


def test_a_terminal_that_is_not_utf8_gets_a_question_mark_for_a_mark(monkeypatch):
    import io
    out = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr("sys.stdout", out)
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    print(config.mark("ok", "fine"))
    out.flush()
    assert out.buffer.getvalue().endswith(b"? fine\n")


def test_version_is_the_installed_packages(capsys):
    from importlib.metadata import version
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0 and capsys.readouterr().out == f"metalscope {version('metalscope')}\n"


UNTESTED_OMLX = "0.6.9"


def test_omlx_install_on_an_untested_omlx_says_it_will_run_untraced(tmp_path, monkeypatch, capsys):
    """The hook installs whatever the version (it is harmless), but on one metalscope is not tested with oMLX runs
    untraced: install says so, and to install a tested oMLX (no METALSCOPE_FORCE advice: oMLX.app's server never sees it)."""
    installed = []
    monkeypatch.setattr(omlx_install, "install", lambda app: installed.append(app))
    version = [UNTESTED_OMLX]
    monkeypatch.setattr(omlx_install, "omlx_version", lambda app: version[0])
    cli.main(["omlx", "install", "--app", str(tmp_path / "oMLX.app")])
    out = capsys.readouterr().out
    assert installed == [tmp_path / "oMLX.app"]
    assert f"warning: oMLX {UNTESTED_OMLX} is not a version metalscope is tested with (" in out
    assert "warning: oMLX will run untraced (no traces or steering)" in out
    assert f"warning: install a tested oMLX version, or update metalscope once it supports oMLX {UNTESTED_OMLX}" in out
    assert "launchctl" not in out and "METALSCOPE_FORCE" not in out
    version[0] = max(omlx_install.TESTED_OMLX)
    cli.main(["omlx", "install", "--app", str(tmp_path / "oMLX.app")])
    assert "warning" not in capsys.readouterr().out


@pytest.fixture
def uninstall(home, monkeypatch):
    """`metalscope uninstall` with ds4, the oMLX hook and server, and Pi faked as installed and running (``state``); a
    test changes one. Returns a function running it: its output lines, exit code and the steps called in order."""
    import subprocess
    import types

    from metalscope import backends
    from metalscope.omlx import admin as omlx_admin
    state = dict(ds4=True, hook=True, omlx_up=True, loaded=True, restart=None, pi=True)
    calls = []

    def ds4_stop(log=print):
        calls.append("ds4 stop")
        if state["ds4"]:
            log("ds4 stopped")
        return state["ds4"]

    def hook_uninstall(app, log=print):
        calls.append("omlx uninstall")
        assert app == omlx_install.DEFAULT_APP
        if isinstance(state["hook"], BaseException):
            raise state["hook"]
        return state["hook"]

    def restart(admin, log=print):
        calls.append("omlx restart")
        if state["restart"]:
            raise state["restart"]
    monkeypatch.setattr(backends, "ds4_stop", ds4_stop)
    monkeypatch.setattr(omlx_install, "uninstall", hook_uninstall)
    monkeypatch.setattr(omlx_admin, "Admin", lambda **kw: types.SimpleNamespace(
        listening=lambda: state["omlx_up"], trace_status=lambda: {} if state["loaded"] else None))
    monkeypatch.setattr(omlx_admin, "restart", restart)
    monkeypatch.setattr(D, "pi_listed", lambda: state["pi"])
    real_run = subprocess.run

    def run(cmd, *a, **k):
        if cmd[0] == "pi":
            calls.append(" ".join(cmd))
            return types.SimpleNamespace(returncode=0)
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(cli.subprocess, "run", run)

    def go(capsys):
        calls.clear()
        try:
            cli.main(["uninstall"])
            code = 0
        except SystemExit as exc:
            code = exc.code
        return capsys.readouterr().out.splitlines(), code, list(calls)
    go.state = state
    return go


def _kept() -> list[str]:
    from metalscope.ds4 import build
    return [f"kept: your data in {D.tilde(str(config.home()))}, the ds4 build and its models in "
            f"{D.tilde(str(build.build_dir()))}", "last step: uv tool uninstall metalscope"]


def test_uninstall_stops_ds4_removes_the_hook_restarts_omlx_and_removes_the_pi_package(uninstall, capsys):
    out, code, calls = uninstall(capsys)
    assert code == 0 and calls == ["ds4 stop", "omlx uninstall", "omlx restart", f"pi remove {config.PI_PACKAGE}"]
    assert out == ["ds4 stopped", "oMLX hook removed", "restarting oMLX without metalscope", "oMLX restarted",
                   f"removed the Pi package {config.PI_PACKAGE}", *_kept()]


def test_uninstall_refuses_while_a_job_runs_and_touches_nothing(uninstall, capsys, monkeypatch):
    from metalscope import gpulock
    from metalscope.learn import state
    monkeypatch.setattr(state, "running", lambda exclude=None: dict(job="j-1", kind="learn", name="crunchy"))
    out, code, calls = uninstall(capsys)
    assert re.fullmatch("cannot uninstall now: a job is running: learning crunchy .*metalscope jobs cancel j-1", code)
    assert out == [] and calls == []
    monkeypatch.setattr(state, "running", lambda exclude=None: None)
    monkeypatch.setattr(gpulock, "held", lambda path=None: gpulock.Holder("learning crunchy", stage="profiling"))
    out, code, calls = uninstall(capsys)
    assert code == "cannot uninstall now: learning crunchy: profiling; try again when it is done"
    assert out == [] and calls == []


def test_uninstall_with_nothing_installed_says_so_for_each_step_and_exits_0(uninstall, capsys):
    uninstall.state.update(ds4=False, hook=SystemExit("no oMLX.app at /Applications/oMLX.app"), loaded=False, pi=None)
    out, code, calls = uninstall(capsys)
    assert code == 0 and calls == ["ds4 stop", "omlx uninstall"]
    assert out == ["metalscope's ds4: not running", "oMLX hook: not installed", "Pi: not on PATH", *_kept()]
    uninstall.state.update(hook=False, pi=False)
    out, code, calls = uninstall(capsys)
    assert code == 0 and out[1:3] == ["oMLX hook: not installed", "Pi package: not in `pi list`"]


def test_uninstall_whose_omlx_restart_fails_still_removes_the_pi_package_and_exits_1(uninstall, capsys):
    from metalscope.omlx import admin as omlx_admin
    uninstall.state["restart"] = omlx_admin.ServerError("server not answering at http://127.0.0.1:8008 after 180s")
    out, code, calls = uninstall(capsys)
    assert code == 1 and calls[-1] == f"pi remove {config.PI_PACKAGE}"
    assert out[3] == ("oMLX restart failed (server not answering at http://127.0.0.1:8008 after 180s): restart oMLX "
                      "by hand to finish")
    assert out[4:] == [f"removed the Pi package {config.PI_PACKAGE}", *_kept()]


def test_uninstall_restarts_omlx_only_when_a_running_server_still_has_metalscope(uninstall, capsys):
    import subprocess
    uninstall.state["omlx_up"] = False
    out, code, calls = uninstall(capsys)
    assert code == 0 and "omlx restart" not in calls and out[1] == "oMLX hook removed"
    uninstall.state.update(omlx_up=True, hook=False, loaded=False)
    out, code, calls = uninstall(capsys)
    assert code == 0 and "omlx restart" not in calls and out[1] == "oMLX hook: not installed"
    uninstall.state["loaded"] = True                 # the hook removed earlier, the server not restarted since
    out, code, calls = uninstall(capsys)
    assert code == 0 and "omlx restart" in calls and out[1:4] == [
        "oMLX hook: not installed", "restarting oMLX without metalscope", "oMLX restarted"]
    uninstall.state["hook"] = subprocess.CalledProcessError(1, "python3")      # a damaged app: said, the rest still runs
    out, code, calls = uninstall(capsys)
    assert code == 1 and "omlx restart" not in calls and out[1].startswith("oMLX hook: could not remove it")
    assert calls[-1] == f"pi remove {config.PI_PACKAGE}"


def test_help_uninstall_prints_its_description(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["help", "uninstall"])
    assert exc.value.code == 0 and "Deletes no data" in capsys.readouterr().out


@pytest.fixture
def doctor(tmp_path, monkeypatch):
    """`metalscope doctor` with every piece faked as wired up (no oMLX.app, server, settings file or Pi is touched);
    a test breaks one. Returns a function running it: its output and exit code."""
    import subprocess
    import types

    from metalscope import client
    from metalscope.omlx import admin as omlx_admin
    state = dict(version=max(omlx_install.TESTED_OMLX),
                 hook=dict(pth=tmp_path / "metalscope.pth", exists=True, loads=omlx_install.PACKAGE, current=True),
                 pi_list=f"{config.PI_PACKAGE}\n")
    monkeypatch.setattr(omlx_install, "omlx_version", lambda app: state["version"])

    def hook_state(app):
        if isinstance(state["hook"], Exception):
            raise state["hook"]
        return state["hook"]
    monkeypatch.setattr(omlx_install, "hook_state", hook_state)
    monkeypatch.setattr(omlx_install, "bundled_run", lambda app, code: types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(omlx_admin, "read_settings", lambda path=None: {"auth": {"api_key": "k"}})
    monkeypatch.setattr(client, "candidates", lambda: [])
    real_run = subprocess.run

    def run(cmd, *a, **k):
        if cmd[:2] == ["pi", "list"]:
            return types.SimpleNamespace(stdout=state["pi_list"])
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(D.subprocess, "run", run)

    def go(capsys):
        try:
            cli.main(["doctor", "--app", str(tmp_path / "oMLX.app")])
            code = 0
        except SystemExit as exc:
            code = exc.code
        return capsys.readouterr().out, code
    go.state = state
    return go


def test_doctor_exits_0_when_nothing_fails_and_1_on_a_fail_line(doctor, capsys):
    """doctor can be scripted: exit 1 when a check fails, 0 with only passed, warning and information lines."""
    out, code = doctor(capsys)
    assert code == 0 and "✗" not in out and "  ✓ hook      " in out
    doctor.state["hook"] = dict(doctor.state["hook"], exists=False)
    out, code = doctor(capsys)
    assert code == 1 and "  ✗ hook      " in out and " · missing: metalscope omlx install\n" in out
    doctor.state["hook"] = dict(doctor.state["hook"], exists=True, current=False)
    out, code = doctor(capsys)
    assert code == 1 and "  ✗ hook      " in out and " · does not run `" in out
    doctor.state["hook"] = OSError("no such file")
    out, code = doctor(capsys)
    assert code == 1 and "  ✗ hook      could not run oMLX's bundled Python" in out


def test_doctor_warns_without_failing_when_pi_lacks_the_package(doctor, capsys):
    """Pi is optional: `pi list` without this package is a warning, exit 0."""
    doctor.state["pi_list"] = ""
    out, code = doctor(capsys)
    assert code == 0 and "  ! package   not installed in Pi: metalscope pi install\n" in out and "✗" not in out


def test_doctor_lists_each_backends_models_as_information_only(doctor, capsys, monkeypatch):
    from metalscope import backends
    monkeypatch.setattr(backends, "omlx_models", lambda: dict(up=False, models=[dict(name="Gemma-4", tested=False, edit=False)]))
    out, code = doctor(capsys)
    assert code == 0 and "\n  oMLX     Gemma-4  no      no      0         no\n" in out
    monkeypatch.setattr(backends, "omlx_models", lambda: (_ for _ in ()).throw(ValueError("not JSON")))
    out, code = doctor(capsys)
    assert code == 0 and "\n  the models could not be listed: ValueError: not JSON\n" in out


def test_doctor_fails_when_the_server_says_a_patch_did_not_apply(doctor, capsys, monkeypatch):
    from metalscope import client
    status = dict(engine="omlx", mode="off", models=[], patches=dict(applied=["patch_server"], missing=[]))
    monkeypatch.setattr(client, "candidates", lambda: ["http://127.0.0.1:8008"])
    monkeypatch.setattr(client.Client, "status", lambda self, timeout=None: status)
    out, code = doctor(capsys)
    assert code == 0 and "  ✓ server    http://127.0.0.1:8008 · tracing off · no model loaded\n" in out
    status["patches"]["missing"] = ["patch_scheduler"]
    out, code = doctor(capsys)
    assert code == 1 and f"  ✗ {'':<10}patches not applied: patch_scheduler (its log says why)\n" in out


def test_doctor_fails_when_a_tested_hooked_omlx_answers_404_for_metalscope(doctor, capsys, monkeypatch):
    """A tested oMLX with a current hook whose /scope/api/status is a 404: metalscope is not loaded in it (a restart
    after install, else its log says why). Nothing answering, a refused key (401) or a 503 is not that."""
    from metalscope import client
    from metalscope.omlx import admin as omlx_admin
    url = "http://127.0.0.1:8008"
    error = [client.ServerError("GET /scope/api/status: HTTP 404: Not Found", 404, "Not Found")]

    def status(self, timeout=None):
        self.status_error = error[0]
        return None
    monkeypatch.setattr(client, "candidates", lambda: [url])
    monkeypatch.setattr(client.Client, "status", status)
    monkeypatch.setattr(omlx_admin, "server_url", lambda settings=None: url)
    out, code = doctor(capsys)
    assert code == 1 and (f"  ✗ server    {url} · runs without metalscope: metalscope omlx restart; if it stays, its "
                          f"log says why ({D.tilde(str(omlx_admin.OMLX_LOG))})\n") in out
    assert "omlx install" not in out
    doctor.state["version"] = UNTESTED_OMLX                  # untested: its warn lines say why already
    out, code = doctor(capsys)
    assert code == 0 and "runs without metalscope" not in out
    doctor.state["version"] = max(omlx_install.TESTED_OMLX)
    for err in (None, client.ServerError("GET /scope/api/status: HTTP 503: x", 503, "x")):   # a timeout; 503
        error[0] = err
        out, code = doctor(capsys)
        assert code == 0 and "runs without metalscope" not in out and f"  · server    {url} · no metalscope server" in out
    error[0] = client.ServerError("GET /scope/api/status: HTTP 401: x", 401, "x")
    out, code = doctor(capsys)
    assert code == 1 and "runs without metalscope" not in out and "refused metalscope's API key" in out


def test_doctor_fails_when_omlx_refuses_the_key_and_says_where_it_comes_from(doctor, capsys, monkeypatch):
    """A 401 from /scope/api/status is its own FAIL line (Client.status_error), never "no metalscope server"."""
    from metalscope import client
    from metalscope.omlx import admin

    def refused(self, timeout=None):
        self.status_error = client.ServerError("GET /scope/api/status: HTTP 401: x", 401, "x")
        return None
    monkeypatch.setattr(client, "candidates", lambda: ["http://127.0.0.1:8008"])
    monkeypatch.setattr(client.Client, "status", refused)
    out, code = doctor(capsys)
    assert code == 1 and "no metalscope server" not in out
    assert (f"  ✗ server    oMLX at http://127.0.0.1:8008 refused metalscope's API key: it sends the one in "
            f"{D.tilde(str(admin.SETTINGS))}, and oMLX may run with another (OMLX_API_KEY, --api-key)\n") in out


def test_doctor_warns_when_a_server_runs_older_code_than_is_installed(doctor, capsys, monkeypatch):
    """A server whose status gives a `stale` restart (web.status) is a warning saying it, exit 0; none: nothing."""
    from metalscope import client
    status = {"http://127.0.0.1:8008": dict(engine="omlx", mode="off", models=[], stale="metalscope omlx restart"),
              "http://127.0.0.1:8009": dict(engine="ds4", mode="off", models=[], stale=None)}
    monkeypatch.setattr(client, "candidates", lambda: list(status))
    monkeypatch.setattr(client.Client, "status", lambda self, timeout=None: status[self.url])
    out, code = doctor(capsys)
    assert code == 0 and f"  ! {'':<10}runs older metalscope code than is installed: metalscope omlx restart\n" in out
    assert out.count("runs older") == 1 and out.index("runs older") < out.index("\nds4\n")
    status["http://127.0.0.1:8009"]["stale"] = "metalscope ds4 stop, then metalscope ds4"
    out, code = doctor(capsys)
    assert code == 0 and "runs older metalscope code than is installed: metalscope ds4 stop, then metalscope ds4\n" in out
    assert out.rindex("runs older") > out.index("\nds4\n")                 # under its engine


def test_doctor_on_an_untested_omlx_warns_that_it_runs_untraced_and_exits_0(doctor, capsys):
    doctor.state["version"] = UNTESTED_OMLX
    out, code = doctor(capsys)
    assert code == 0 and "✗" not in out
    assert f"  ! version   oMLX {UNTESTED_OMLX} is not a version metalscope is tested with (" in out
    assert f"  ! {'':<10}oMLX will run untraced (no traces or steering)" in out
    assert f"  ! {'':<10}install a tested oMLX version, or update metalscope once it supports oMLX {UNTESTED_OMLX}" in out
    assert "omlx install" not in out


def test_doctor_on_a_ds4_only_mac_does_not_fail_for_the_missing_omlx_app(home, monkeypatch, capsys):
    from metalscope.ds4 import build as B
    def no_process(*a, **k):
        raise OSError("not here")
    monkeypatch.setattr(D.subprocess, "run", no_process)
    monkeypatch.setattr(D.client.urllib.request, "urlopen", no_process)
    B.build_dir().mkdir(parents=True, exist_ok=True)
    (B.build_dir() / "ds4-server").write_text("")
    try:
        cli.main(["doctor"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "  · app       none at " in out and "fine with ds4 alone" in out
    assert "✗ app" not in out and "hook" not in out


def healthy(monkeypatch) -> dict:
    """Both servers up and tracing (ds4 serving a model) and the ds4 build current, for the `doctor` fixture: every
    check passes. Returns the servers' status by URL."""
    from metalscope import client
    from metalscope.ds4 import build as B
    status = {"http://127.0.0.1:8008": dict(engine="omlx", mode="on", models=[]),
              client.DS4_URL: dict(engine="ds4", mode="on", models=[dict(name="Q-ds4")])}
    monkeypatch.setattr(client, "candidates", lambda: list(status))
    monkeypatch.setattr(client.Client, "status", lambda self, timeout=None: status[self.url])
    monkeypatch.setattr(B, "built", lambda d: True)
    monkeypatch.setattr(B, "current", lambda d: True)
    return status


def test_doctor_on_a_healthy_setup_prints_each_section_in_order_and_ends_all_good(doctor, capsys, monkeypatch):
    status = healthy(monkeypatch)
    out, code = doctor(capsys)
    heads = [line for line in out.splitlines() if line and not line.startswith(" ")]
    assert code == 0 and heads == ["oMLX", "ds4", "Pi", "Models", "Traces", "all good"], out
    assert "\n\nds4\n" in out and out.endswith("\n\nall good\n") and "✗" not in out and "  ! " not in out
    assert out.startswith("oMLX\n")
    omlx, ds4, pi = (out.index(f"{h}\n") for h in ("oMLX", "\nds4", "\nPi"))
    assert omlx < out.index("  ✓ server    http://127.0.0.1:8008 · tracing on · no model loaded\n") < ds4
    assert ds4 < out.index("  ✓ build     ") < out.index("  ✓ server    http://127.0.0.1:8009 · tracing on · Q-ds4\n") < pi
    status["http://127.0.0.1:8008"]["engine"] = "ds4"         # a server goes under the engine it reports
    out, code = doctor(capsys)
    assert out.index("\nds4\n") < out.index("  ✓ server    http://127.0.0.1:8008") < out.index("\nPi\n")


def test_doctor_marks_and_counts_a_failed_check_and_a_warning_and_exits_1(doctor, capsys, monkeypatch):
    healthy(monkeypatch)
    doctor.state["hook"] = dict(doctor.state["hook"], exists=False)
    doctor.state["pi_list"] = ""
    out, code = doctor(capsys)
    assert code == 1 and out.endswith("\n\n1 failed · 1 warning\n")
    assert "  ✗ hook      " in out and "  ! package   not installed in Pi: metalscope pi install\n" in out
    assert [line[2] for line in out.splitlines() if line[2:3] in ("✗", "!")] == ["✗", "!"]


def test_doctor_prints_paths_under_the_home_folder_with_a_tilde(doctor, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    out, code = doctor(capsys)
    assert code == 0 and "  ✓ app       ~/oMLX.app · " in out and "  ✓ hook      ~/metalscope.pth\n" in out
    assert f"{tmp_path}/" not in out


def test_colour_only_on_a_terminal_with_no_color_unset(doctor, capsys, monkeypatch, tmp_path):
    import io
    import sys

    class Tty(io.StringIO):
        def isatty(self):
            return True

    def screen(stdout):
        monkeypatch.setattr(sys, "stdout", stdout)
        cli.main(["doctor", "--app", str(tmp_path / "oMLX.app")])
        return stdout.getvalue()
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = screen(Tty())
    assert "\x1b[1moMLX\x1b[0m\n" in out and "  \x1b[32m✓\x1b[0m hook " in out and "\x1b[1;32mall good\x1b[0m" in out
    assert config.mark("fail", "x") == "\x1b[31m✗\x1b[0m x" and config.mark("info", "x") == "\x1b[2m· x\x1b[0m"
    monkeypatch.setenv("NO_COLOR", "1")
    assert "\x1b" not in screen(Tty()) and config.mark("warn", "x") == "! x"
    monkeypatch.delenv("NO_COLOR")
    out = screen(io.StringIO())                                       # piped: the same characters, no codes
    assert "\x1b" not in out and "  ✓ hook " in out

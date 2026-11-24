"""models.py: which checkpoint a served name is, where its data lives (all of it under the checkpoint's folder: a
concept's, the weights', each served copy's), link, and the shape guard (engines, ds4 proxy)."""
import json
from pathlib import Path

import numpy as np
import pytest

from metalscope import cli, engines, models, steer
from metalscope.engines import Shape

FLASH = models.shape_of(48, 4, 2560)


def test_a_served_name_nothing_links_is_its_own_checkpoint(tmp_path):
    assert models.checkpoint_of("org/Qwen3.5-2B-4bit") == "Qwen3.5-2B-4bit"
    assert models.record("Qwen3.5-2B-4bit") == dict(name="Qwen3.5-2B-4bit", shape=None, served=[])
    assert models.served_names("Qwen3.5-2B-4bit") == ["Qwen3.5-2B-4bit"]
    home = models.models_dir() / "Qwen3.5-2B-4bit"
    assert models.model_dir("org/Qwen3.5-2B-4bit") == home
    assert models.definition_path("calm") == models.models_dir().parent / "concepts" / "calm" / "concept.json"   # shared
    calm = home / "concepts" / "calm"
    assert models.concept_dir("org/Qwen3.5-2B-4bit", "calm") == calm
    assert models.vector_path("Qwen3.5-2B-4bit", "calm") == calm / "vector.safetensors"
    assert models.profile_path("Qwen3.5-2B-4bit", "calm") == calm / "profile.json"
    served = home / models.SERVED / "Qwen3.5-2B-4bit"
    assert models.served_dir("org/Qwen3.5-2B-4bit") == served
    assert models.norms_path("Qwen3.5-2B-4bit") == served / "norms.json"
    cal = calm / "Qwen3.5-2B-4bit"
    assert models.calibration_path("org/Qwen3.5-2B-4bit", "calm") == cal / "calibration.json"
    assert models.replies_path("Qwen3.5-2B-4bit", "calm") == cal / "replies.jsonl"


def test_linked_names_share_the_checkpoints_data_and_keep_their_own_norms_and_calibrations():
    models.check_shape("Flash-oQ4e", FLASH)                         # opened once: recorded as its own checkpoint
    models.link("Flash-oQ4e", "Flash")
    models.link("Flash-ds4", "Flash")
    assert not (models.models_dir() / "Flash-oQ4e").exists()        # the record that held only its shape is gone
    rec = models.record("Flash-ds4")
    assert (rec["name"], rec["shape"], rec["served"]) == ("Flash", FLASH, ["Flash-oQ4e", "Flash-ds4"])
    for name in ("Flash-oQ4e", "Flash-ds4", "Flash"):
        assert models.checkpoint_of(name) == "Flash"
        assert models.vector_path(name, "calm") == models.concept_dir("Flash", "calm") / "vector.safetensors"
    assert models.served_dir("Flash-ds4") != models.served_dir("Flash-oQ4e")           # per served copy
    assert models.served_dir("Flash-ds4").parent == models.model_dir("Flash") / models.SERVED   # under the checkpoint
    assert models.norms_path("Flash-ds4") != models.norms_path("Flash-oQ4e")
    # A linked copy's calibration is in the concept's folder for the checkpoint, under the copy's own name.
    assert models.calibration_path("Flash-ds4", "calm") == models.concept_dir("Flash", "calm") / "Flash-ds4" / "calibration.json"
    assert models.calibration_dir("Flash-oQ4e", "calm") == models.concept_dir("Flash", "calm") / "Flash-oQ4e"
    prof = models.profile_path("Flash-oQ4e", "calm")
    prof.parent.mkdir(parents=True)
    prof.write_text("{}")
    assert models.profile_paths("Flash-ds4") == [prof]
    models.link("Flash-ds4", "Flash")                               # again: nothing changes
    assert models.record("Flash")["served"] == ["Flash-oQ4e", "Flash-ds4"]
    models.link("Flash-old", "Flash")
    for served in ("Flash-ds4", "Other", "Flash-old"):              # the checkpoint's, another's, an older one
        cal = models.calibration_path(served, "calm")
        cal.parent.mkdir(parents=True)
        cal.write_text(json.dumps(dict(version=3 if served == "Flash-old" else 4)))
    assert models.calibrated_copies("Flash-oQ4e", "calm") == ["Flash-ds4"]
    assert models.grades_path("Flash-ds4", "calm") == models.concept_dir("Flash", "calm") / "Flash-ds4" / "grades.json"
    assert models.served_copy("Flash-ds4")["calibrations"] == ["calm"] and models.served_copy("Other")["calibrations"] == ["calm"]
    assert models.served_copy("Flash-old")["calibrations"] == []
    assert models.plain_chats_path("Flash-ds4", "abc") == models.served_dir("Flash-ds4") / "plain-chats" / "abc.jsonl"


def test_vectors_and_profiles_are_in_each_concepts_folder_by_checkpoint():
    from metalscope import vectors
    models.link("Flash-ds4", "Flash")
    for name in ("calm", "terse"):
        vectors.save_vector("Flash-ds4", vectors.Vector(name, {3: vectors.unit(np.ones(4))}, dict(source="test")))
    vectors.save_vector("Other", vectors.Vector("calm", {3: vectors.unit(np.ones(4))}, {}))
    calm = models.concept_dir("Flash", "calm")
    assert sorted(p.name for p in calm.iterdir()) == ["vector.json", "vector.safetensors"]
    assert vectors.find_vector("Flash", "calm") == models.vector_path("Flash-ds4", "calm")
    assert vectors.find_vector("Flash", "missing") is None and vectors.find_vector("Flash", "../calm") is None
    assert vectors.load_vector("Flash-ds4", "terse").meta["source"] == "test"
    assert [(v["name"], v["layers"]) for v in vectors.list_vectors("Flash-ds4")] == [("calm", [3]), ("terse", [3])]
    assert [v["name"] for v in vectors.list_vectors("Other")] == ["calm"]
    models.profile_path("Flash", "terse").write_text("{}")
    assert models.profile_paths("Flash-ds4") == [models.profile_path("Flash", "terse")]
    assert models.vector_paths("Flash") == [models.vector_path("Flash", "calm"), models.vector_path("Flash", "terse")]
    models.check_shape("Flash-ds4", FLASH)
    assert [(c["name"], c["vectors"], c["profiles"]) for c in models.checkpoints()] == [("Flash", 2, 1), ("Other", 1, 0)]


def test_link_refuses_another_shape_a_checkpoint_with_data_and_a_served_name_as_checkpoint():
    models.check_shape("Flash-oQ4e", FLASH)
    models.link("Flash-oQ4e", "Flash")
    models.check_shape("V4", models.shape_of(43, 4, 4096))
    with pytest.raises(models.ShapeMismatch, match="V4 is 43 layers x 4 x 4096, Flash is 48 layers x 4 x 2560"):
        models.link("V4", "Flash")
    with pytest.raises(ValueError, match="Flash-oQ4e is a served name of Flash, not a checkpoint"):
        models.link("x", "Flash-oQ4e")
    models.concept_dir("Qwen-2B", "calm").mkdir(parents=True)
    with pytest.raises(ValueError, match="Qwen-2B has data of its own"):
        models.link("Qwen-2B", "Flash")
    assert models.record("Flash")["served"] == ["Flash-oQ4e"]


def test_link_moves_the_copys_own_folder_first_and_refuses_what_it_would_overwrite():
    models.check_shape("Flash-oQ4e", FLASH)                         # its own checkpoint until linked
    models.norms_path("Flash-oQ4e").parent.mkdir(parents=True)
    models.norms_path("Flash-oQ4e").write_text(json.dumps(dict(traces=["a"])))
    models.plain_chats_path("Flash-oQ4e", "abc").parent.mkdir()
    models.plain_chats_path("Flash-oQ4e", "abc").write_text("{}\n")
    models.link("Flash-oQ4e", "Flash")
    assert not (models.models_dir() / "Flash-oQ4e").exists()        # the record that held only its shape is gone...
    assert models.served_dir("Flash-oQ4e").parent == models.model_dir("Flash") / models.SERVED
    assert json.loads(models.norms_path("Flash-oQ4e").read_text())["traces"] == ["a"]   # ...its norms moved first
    assert models.plain_chats_path("Flash-oQ4e", "abc").read_text() == "{}\n"
    models.link("Flash-oQ4e", "Flash")                              # again: nothing moves
    models.link("Flash-oQ4e", "V4")                                 # to another checkpoint: the folder follows
    assert models.plain_chats_path("Flash-oQ4e", "abc").read_text() == "{}\n"
    assert not (models.model_dir("Flash") / models.SERVED).exists()
    # A copy with no record: its folder moves and nothing of it is left.
    models.norms_path("Flash-ds4").parent.mkdir(parents=True)
    taken = models.model_dir("V4") / models.SERVED / "Flash-ds4"
    taken.mkdir()
    with pytest.raises(ValueError, match=f"V4 already has {taken} and Flash-ds4 has {models.served_dir('Flash-ds4')}: "
                                         "move one aside"):
        models.link("Flash-ds4", "V4")
    assert models.checkpoint_of("Flash-ds4") == "Flash-ds4" and models.served_dir("Flash-ds4").is_dir()
    taken.rmdir()
    models.link("Flash-ds4", "V4")
    assert models.served_dir("Flash-ds4") == taken and taken.is_dir() and not (models.models_dir() / "Flash-ds4").exists()
    # A served name with a lens or a fit of its own is a checkpoint, and so is one holding another copy's folder.
    from metalscope import jlens
    from metalscope.learn import lens_fit_gpu
    for own in (jlens.lenses_dir("Own-lens") / "x", lens_fit_gpu.fit_dir("Own-fit", "abc"),
                models.served_dir("Own-copies").parent / "Own-copies-ds4"):
        own.mkdir(parents=True)
        name = own.relative_to(models.models_dir()).parts[0]
        with pytest.raises(ValueError, match=f"{name} has data of its own"):
            models.link(name, "V4")


def _norms(name: str) -> Path:
    p = models.norms_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(traces=["a"])))
    return p


def test_a_copys_norms_are_under_its_checkpoints_served_folder():
    _norms("S")
    models.link("S", "C")
    assert (models.models_dir() / "C" / "served" / "S" / "norms.json").is_file()      # the literal folder name


def test_a_link_that_fails_after_the_move_completes_when_run_again(monkeypatch):
    from metalscope import tally
    models.check_shape("C1", FLASH)
    models.check_shape("C2", FLASH)
    models.link("S", "C1")
    _norms("S")
    with monkeypatch.context() as m, pytest.raises(OSError, match="disk full"):
        m.setattr(tally, "move", lambda a, b: (_ for _ in ()).throw(OSError("disk full")))
        models.link("S", "C2")
    assert models.checkpoint_of("S") == "C1" and not models.norms_path("S").exists()     # moved, records not yet
    models.link("S", "C2")                                        # the folder already moved: adopted
    assert models.checkpoint_of("S") == "C2" and json.loads(models.norms_path("S").read_text())["traces"] == ["a"]
    # A folder for S at the target and none where S is now: adopted too.
    stale = models.models_dir() / "C1" / models.SERVED / "T"
    stale.mkdir(parents=True)
    models.link("T", "C1")
    assert models.served_dir("T") == stale and stale.is_dir()


def test_link_never_deletes_what_it_did_not_move(monkeypatch):
    # S listed by A and by itself (a hand edit, or a race with check_shape): its own folder holds S's norms.
    models.write_record(dict(name="A", shape=FLASH, served=["S"]))
    models.write_record(dict(name="S", shape=FLASH, served=["S"]))
    norms = models.models_dir() / "S" / models.SERVED / "S" / "norms.json"
    norms.parent.mkdir(parents=True)
    norms.write_text("{}")
    models.check_shape("C", FLASH)
    with pytest.raises(ValueError, match=r"S has data of its own in .*S \(served/\)"):
        models.link("S", "C")
    assert norms.is_file()
    # Even when the check misses it, the record's folder goes only when nothing else is left in it.
    monkeypatch.setattr(models, "_holds_data", lambda ckpt: [])
    models.link("S", "C")
    assert norms.is_file() and models.checkpoint_of("S") == "C"


def test_link_ignores_finder_files_and_refuses_a_served_file_cleanly():
    models.check_shape("S", FLASH)
    _norms("S")
    for d in (models.models_dir() / "S", models.models_dir() / "S" / models.SERVED):
        (d / ".DS_Store").write_text("")
    models.check_shape("C", FLASH)
    models.link("S", "C")
    assert models.norms_path("S").is_file() and not (models.models_dir() / "S").exists()
    models.check_shape("F", FLASH)
    (models.models_dir() / "F" / models.SERVED).write_text("x")
    with pytest.raises(ValueError, match="F has data of its own"):
        models.link("F", "C")


def test_the_refusal_names_the_folders_to_move_and_where():
    models.check_shape("S", FLASH)
    (models.models_dir() / "S" / "fits" / "lens-S-abc").mkdir(parents=True)
    models.concept_dir("S", "calm").mkdir(parents=True)
    models.check_shape("C", FLASH)
    with pytest.raises(ValueError) as e:
        models.link("S", "C")
    assert str(e.value) == (f"S has data of its own in {models.models_dir() / 'S'} (concepts/, fits/): to link it, move "
                            f"what is inside each into the same folder under {models.models_dir() / 'C'}, then link again")


def test_relinking_moves_the_copys_fits_with_it():
    from metalscope.learn import lens_fit_gpu as F
    models.check_shape("C1", FLASH)
    models.check_shape("C2", FLASH)
    models.link("S", "C1")
    models.link("S-ds4", "C1")
    hexid, other = "0123456789ab", "ba9876543210"
    (F.fit_dir("S", hexid) / "fit.json").parent.mkdir(parents=True)
    (F.fit_dir("S", hexid) / "fit.json").write_text("{}")
    F.fit_dir("S-ds4", hexid).mkdir(parents=True)                  # another copy's, whose name starts with S's
    clash = models.models_dir() / "C2" / "fits" / f"lens-S-{other}"
    F.fit_dir("S", other).mkdir()
    clash.mkdir(parents=True)
    with pytest.raises(ValueError, match=f"C2 already has {clash} and S has {F.fit_dir('S', other)}: move one aside"):
        models.link("S", "C2")
    assert models.checkpoint_of("S") == "C1" and F.fit_dir("S", hexid).is_dir()      # nothing moved
    clash.rmdir()
    models.link("S", "C2")
    assert (F.fit_dir("S", hexid) / "fit.json").is_file() and F.fit_dir("S", hexid).parent == models.models_dir() / "C2" / "fits"
    assert F.fit_dir("S", other).is_dir() and F.fit_dir("S-ds4", hexid).parent == models.models_dir() / "C1" / "fits"


def test_link_waits_for_a_job_on_that_served_name(monkeypatch):
    from metalscope.learn import state
    job = dict(job="cal-1", kind="calibrate", name="calm", model="org/S", outcome=None)
    monkeypatch.setattr(state, "running", lambda exclude=None: job)
    with pytest.raises(ValueError, match="cannot link S now: a job is running: calibrating calm .*: wait for it or stop it: "
                                         "metalscope jobs cancel cal-1"):
        models.link("S", "C")
    models.link("T", "C")                                           # a job on another model: no matter
    assert models.checkpoint_of("T") == "C"


def test_a_checkpoint_record_is_read_and_written_under_its_lock():
    """oMLX, the ds4 proxy and the command line may change one record at once: each change waits for the record's
    lock and keeps what another wrote meanwhile. The lock file is not data and goes with an emptied folder."""
    import threading
    from metalscope import config
    models.write_record(dict(name="C", shape=None, served=["S"]))
    models.check_shape("T", FLASH)
    assert models._holds_data("T") == []
    cases = [(lambda: models.update("S", lens="L"), lambda rec: rec.get("lens") == "L"),
             (lambda: models.check_shape("S", FLASH), lambda rec: rec["shape"] == FLASH),
             (lambda: models.link("T", "C"), lambda rec: rec["served"] == ["S", "T"])]
    for i, (change, done) in enumerate(cases):
        with config.file_lock(models.models_dir() / "C" / "checkpoint.lock"):
            th = threading.Thread(target=change)
            th.start()
            th.join(0.3)
            assert th.is_alive()                                    # waiting for the lock
            models.write_record(dict(models.record("C"), jlens=f"J{i}"))     # another writer, meanwhile
        th.join()
        rec = models.record("C")
        assert rec["jlens"] == f"J{i}" and done(rec)
    assert not (models.models_dir() / "T").exists()


def holding(ckpt: str, concepts=(), lenses=(), shape=FLASH) -> str:
    """A checkpoint with a recorded shape, profiles of ``concepts`` and lenses named ``lenses``."""
    models.check_shape(ckpt, shape)
    for c in concepts:
        models.concept_dir(ckpt, c).mkdir(parents=True)
        models.profile_path(ckpt, c).write_text("{}")
    for n in lenses:
        d = models.model_dir(ckpt) / "lenses" / n
        d.mkdir(parents=True)
        (d / "lens.safetensors").write_bytes(b"")
    return ckpt


def test_a_fresh_build_is_offered_the_same_shape_models_with_data_most_first():
    holding("Small", concepts=["calm"])
    holding("Big", concepts=["calm", "terse"], lenses=["l1"])
    holding("Tied", concepts=["a", "b"], lenses=["x"])
    holding("Empty")                                                       # nothing to share: not offered
    holding("V4", concepts=["calm"], shape=models.shape_of(43, 4, 4096))   # another shape
    models.concept_dir("Small", "vec-only").mkdir(parents=True)
    models.vector_path("Small", "vec-only").write_bytes(b"x")              # a word vector (no profile) is not a concept
    (models.concept_dir("Small", "texts-only")).mkdir()                    # texts alone are not either
    holding("Words")
    models.concept_dir("Words", "the").mkdir(parents=True)
    models.vector_path("Words", "the").write_bytes(b"x")                   # word vectors alone: not offered
    holding("Ghost", concepts=["x", "y", "z"])                             # a served name of another: never offered
    models.write_record(dict(models.record("Big"), served=["Big", "Ghost"]))
    assert models.checkpoint_of("Ghost") == "Big"
    models.check_shape("Build", FLASH)
    assert models.link_offer("org/Build") == dict(
        model="Build", checkpoint="Build", linked=False, own=False, shape=dict(n_layers=48, width=10240),
        candidates=[dict(name="Big", concepts=2, lenses=1), dict(name="Tied", concepts=2, lenses=1),
                    dict(name="Small", concepts=1, lenses=0)])


def test_no_offer_when_linked_with_data_of_its_own_or_no_shape():
    holding("Big", concepts=["calm"])
    assert models.link_offer("Nobody") == dict(model="Nobody", checkpoint="Nobody", linked=False, own=False, shape=None,
                                               candidates=[])
    models.link("Copy", "Big")
    assert models.link_offer("Copy")["linked"] and models.link_offer("Copy")["candidates"] == []
    holding("Mine", lenses=["l"], shape=models.shape_of(43, 4, 4096))     # another shape: no candidate below
    assert models.link_offer("Mine")["own"] and models.link_offer("Mine")["candidates"] == []
    holding("Alone")
    models.concept_dir("Big", "calm").joinpath("profile.json").unlink()   # the only same-shape model is now empty
    assert models.link_offer("Alone")["candidates"] == []


def test_model_show_names_up_to_three_same_shape_models_to_link(capsys):
    for i, n in enumerate(["A", "B", "C", "D"]):
        holding(f"Ckpt{n}", concepts=[f"c{k}" for k in range(4 - i)], lenses=["l"] if n == "A" else [])
    models.check_shape("Build", FLASH)
    cli.main(["model", "show", "Build"])
    lines = [l for l in capsys.readouterr().out.splitlines() if "same shape as" in l]
    assert lines == ["  same shape as CkptA (4 concepts, 1 lens): metalscope model link Build CkptA if it is the same model",
                     "  same shape as CkptB (3 concepts, 0 lenses): metalscope model link Build CkptB if it is the same model",
                     "  same shape as CkptC (2 concepts, 0 lenses): metalscope model link Build CkptC if it is the same model"]
    cli.main(["model", "show", "CkptA"])
    assert "same shape as" not in capsys.readouterr().out                  # data of its own


def test_trash_moves_one_checkpoints_folder_and_leaves_the_rest(monkeypatch):
    monkeypatch.setattr(models.time, "strftime", lambda f: "20261117-120000")
    for ckpt in ("Flash", "V4"):
        models.concept_dir(ckpt, "calm").mkdir(parents=True)
        (models.concept_dir(ckpt, "calm") / "profile.json").write_text("{}")
        (models.concept_dir(ckpt, "calm") / "pos.txt").write_text("a")          # its texts, its own
    models.definition_path("calm").parent.mkdir(parents=True)
    models.definition_path("calm").write_text("{}")
    models.check_shape("Flash-ds4", FLASH)
    models.link("Flash-ds4", "Flash")
    gone = models.trash("Flash-ds4", "calm")                      # by a served name: its checkpoint's folder
    assert gone.parent == models.config.home() / "trash" and gone.name.endswith("-calm-Flash")
    assert (gone / "profile.json").is_file() and (gone / "pos.txt").is_file() and not models.concept_dir("Flash", "calm").exists()
    assert (models.concept_dir("V4", "calm") / "pos.txt").is_file() and models.definition_path("calm").is_file()
    with pytest.raises(FileNotFoundError, match="calm has nothing on Flash"):
        models.trash("Flash", "calm")
    # The same second again: a suffix, never onto the first.
    models.concept_dir("Flash", "calm").mkdir(parents=True)
    again = models.trash("Flash", "calm")
    assert again.name == "20261117-120000-calm-Flash-1" and again.is_dir() and gone.is_dir()
    # A symlink in its place is not followed; names that are not one folder are refused.
    models.concept_dir("Flash", "calm").parent.mkdir(exist_ok=True)
    models.concept_dir("Flash", "calm").symlink_to(models.concept_dir("V4", "calm"))
    with pytest.raises(FileNotFoundError):
        models.trash("Flash", "calm")
    for model, concept in [(".", "calm"), ("..", "calm"), ("Flash", ".."), ("Flash", "a/b")]:
        with pytest.raises(ValueError, match="bad"):
            models.trash(model, concept)
    assert models.concept_dir("V4", "calm").is_dir()


def test_trash_leaves_no_empty_folder_that_reads_as_data():
    """Deleting a checkpoint's last concept leaves no empty concepts/ (and no empty checkpoint folder), which would
    count as data of its own and block model link."""
    models.check_shape("Small-ds4", FLASH)                       # a record: the folder stays, concepts/ goes
    for ckpt in ("Small-ds4", "Bare"):
        models.concept_dir(ckpt, "calm").mkdir(parents=True)
        (models.concept_dir(ckpt, "calm") / "profile.json").write_text("{}")
        models.trash(ckpt, "calm")
    assert not (models.models_dir() / "Small-ds4" / "concepts").exists() and models._holds_data("Small-ds4") == []
    assert not (models.models_dir() / "Bare").exists()
    models.check_shape("Flash", FLASH)
    models.link("Small-ds4", "Flash")                            # not blocked
    # Another concept still there: concepts/ stays.
    for c in ("calm", "joy"):
        models.concept_dir("Flash", c).mkdir(parents=True)
    models.trash("Flash", "calm")
    assert models.concept_dir("Flash", "joy").is_dir()


def test_the_shape_guard_records_once_then_refuses_another_shape():
    rec = models.check_shape("Flash-oQ4e", FLASH)
    assert rec["shape"] == FLASH and rec["served"] == ["Flash-oQ4e"]
    on_disk = json.loads((models.models_dir() / "Flash-oQ4e" / models.RECORD).read_text())
    assert on_disk == dict(name="Flash-oQ4e", shape=FLASH, served=["Flash-oQ4e"])
    models.check_shape("Flash-oQ4e", dict(FLASH, width=1))           # only layers, streams and d_model are compared
    with pytest.raises(models.ShapeMismatch, match="Flash-oQ4e is 48 layers x 1 x 2560, but its checkpoint Flash-oQ4e"):
        models.check_shape("Flash-oQ4e", models.shape_of(48, 1, 2560))


class _Engine(engines.Engine):
    kind = "fake"

    def __init__(self, served, shape):
        self.served, self.checkpoint, self.shape, self.closed = served, models.checkpoint_of(served), shape, False

    def close(self):
        self.closed = True


def test_an_engine_opens_only_on_its_checkpoints_shape():
    e = engines.checked(_Engine("Flash-ds4", Shape(48, 10240, 4, 2560, 151936)))
    assert models.record("Flash-ds4")["shape"] == FLASH and not e.closed
    wrong = _Engine("Flash-ds4", Shape(43, 16384, 4, 4096, 129280))
    with pytest.raises(engines.EngineUnavailable, match="not the same model"):
        engines.checked(wrong)
    assert wrong.closed


def test_the_ds4_proxy_refuses_a_gguf_of_another_shape_before_it_loads(monkeypatch):
    from metalscope.ds4 import proxy
    models.check_shape("Flash-ds4", FLASH)
    monkeypatch.setattr(proxy, "model_shape", lambda gguf: dict(family="deepseek41", n_layer=43, n_hc=4, n_embd=4096,
                                                                 k=6, n_expert=384))
    monkeypatch.setattr(proxy, "Ds4Tracer", lambda *a, **k: pytest.fail("the tracer must not be made"))
    with pytest.raises(SystemExit, match="Flash-ds4 is 43 layers x 4 x 4096"):
        proxy.make_served(Path("/m/V4.gguf"), "Flash-ds4", "http://127.0.0.1:8010")


@pytest.fixture
def served_on_omlx():
    """MANAGER.attach as oMLX's model load calls it (omlx/patches.py), tracing on; the tracers taken off after."""
    from metalscope import config
    from metalscope.live import MANAGER
    from tests.helpers import Tok, tiny_llama
    config.save_settings(mode="on")
    yield lambda name, model=None: MANAGER.attach(model or tiny_llama(), name, Tok())
    for key in list(MANAGER.tracers):
        MANAGER.tracers.pop(key)


def test_a_model_served_on_omlx_records_its_shape_once_when_its_hooks_install(served_on_omlx, monkeypatch):
    holding("tiny-ckpt", concepts=["calm"], shape=models.shape_of(3, 1, 64))
    assert models.record("tiny-build")["shape"] is None
    assert served_on_omlx("org/tiny-build").installed
    assert models.record("tiny-build") == dict(name="tiny-build", shape=models.shape_of(3, 1, 64), served=["tiny-build"])
    assert [c["name"] for c in models.link_offer("tiny-build")["candidates"]] == ["tiny-ckpt"]
    monkeypatch.setattr(models, "write_record", lambda rec: pytest.fail("recorded again"))
    assert served_on_omlx("tiny-build").installed                  # loaded again: the record stands


def test_a_model_with_hyper_connection_streams_records_one_streams_width_as_d_model(served_on_omlx):
    """4 streams of 64 (V4's hc_head): d_model is 64 and the width 256, never the width as d_model."""
    import mlx.nn as nn
    from tests.helpers import tiny_llama
    model, head = tiny_llama(), nn.Module()
    head.hc_mult = 4
    model.model.hc_head = head
    holding("hc-ckpt", concepts=["calm"], shape=models.shape_of(3, 4, 64))
    assert served_on_omlx("hc-build", model).installed
    assert models.record("hc-build")["shape"] == dict(n_layers=3, width=256, hc=4, d_model=64)
    assert [c["name"] for c in models.link_offer("hc-build")["candidates"]] == ["hc-ckpt"]


def test_a_served_model_of_another_shape_is_logged_and_served(served_on_omlx, caplog):
    models.check_shape("tiny", FLASH)
    before = (models.models_dir() / "tiny" / models.RECORD).read_text()
    with caplog.at_level("WARNING", logger="metalscope"):
        assert served_on_omlx("tiny").installed                    # chat goes on, traced
    assert (models.models_dir() / "tiny" / models.RECORD).read_text() == before
    assert "tiny is 3 layers x 1 x 64, but its checkpoint tiny is 48 layers x 4 x 2560" in caplog.text


def test_a_ds4_build_is_offered_the_omlx_learned_model_of_its_shape(monkeypatch):
    from metalscope.ds4 import proxy
    e = engines.checked(_Engine("Flash-oQ4e", Shape(48, 10240, 4, 2560, 151936)))   # a learn job's load on oMLX
    models.concept_dir("Flash-oQ4e", "calm").mkdir(parents=True)
    models.profile_path("Flash-oQ4e", "calm").write_text("{}")
    assert not e.closed
    monkeypatch.setattr(proxy, "model_shape", lambda gguf: dict(family="qwen4", n_layer=48, n_hc=4, n_embd=2560, k=8,
                                                                 n_expert=256))
    monkeypatch.setattr(proxy, "tokenizer_for", lambda gguf: None)
    monkeypatch.setattr(proxy, "Ds4Tracer", lambda *a, **k: (_ for _ in ()).throw(KeyError("stop here")))
    with pytest.raises(KeyError):
        proxy.make_served(Path("/m/Flash.gguf"), "Flash-ds4", "http://127.0.0.1:8010")   # records its shape first
    assert models.link_offer("Flash-ds4")["candidates"] == [dict(name="Flash-oQ4e", concepts=1, lenses=0)]


def _trace(value: float) -> str:
    """A saved trace whose two layers' residual norms are all ``value``."""
    from metalscope import config
    from metalscope.store import Trace
    arrays = {f"resid_norm/{l}": np.full(4, value, np.float32) for l in (0, 1)}
    return str(Trace(arrays, [1] * 4, ["x"] * 4, 2).save(config.traces_dir() / str(value)))


def test_each_served_copy_has_its_own_strength_table_from_its_own_traces():
    models.link("Flash-oQ4e", "Flash")
    models.link("Flash-ds4", "Flash")
    entries = [dict(id="a", model="org/Flash-oQ4e", path=_trace(10)), dict(id="b", model="Flash-ds4", path=_trace(20)),
               dict(id="c", model="Flash-ds4", path=_trace(99), steer="calm@0.5"), dict(id="d", model="Other", path=_trace(99))]
    table = steer.norm_table("Flash-ds4", entries=entries)
    assert (table["model"], table["checkpoint"], table["traces"], table["layers"]) == ("Flash-ds4", "Flash", ["b"], {0: 20.0, 1: 20.0})
    assert models.norms_path("Flash-ds4").exists() and not models.norms_path("Flash-oQ4e").exists()
    assert steer.norm_table("Flash-oQ4e", entries=entries)["traces"] == ["a"]
    assert steer.norm_table("Flash-ds4", entries=[])["traces"] == ["b"]                 # cached, per served copy
    with pytest.raises(ValueError, match="no saved unsteered traces of Flash-q8"):       # never another copy's table
        steer.norm_table("Flash-q8", entries=entries)
    with pytest.raises(ValueError, match="no saved unsteered traces of Flash "):         # nor the checkpoint's name
        steer.norm_table("Flash", entries=entries)


def test_the_model_command_lists_shows_and_links(capsys):
    models.check_shape("Flash-oQ4e", FLASH)
    cli.main(["model", "link", "Flash-ds4", "Flash-oQ4e"])
    assert capsys.readouterr().out.startswith("Flash-ds4 is now a copy of Flash-oQ4e (48 layers x 4 x 2560)")
    models.vector_path("Flash-ds4", "calm").parent.mkdir(parents=True)
    models.vector_path("Flash-ds4", "calm").write_bytes(b"x")
    cli.main(["model", "list"])
    assert capsys.readouterr().out.strip() == ("Flash-oQ4e: 48 layers x 4 x 2560 · served as Flash-oQ4e, Flash-ds4 · lens (by name) "
                                               "· 1 vectors, 0 profiles")
    # Each served copy's norms and calibrations, listed by copy.
    models.served_dir("Flash-ds4").mkdir(parents=True)
    models.norms_path("Flash-ds4").write_text(json.dumps(dict(traces=["a", "b", "c"])))
    models.calibration_path("Flash-oQ4e", "calm").parent.mkdir(parents=True)
    models.calibration_path("Flash-oQ4e", "calm").write_text(json.dumps(dict(version=4)))
    cli.main(["model", "list"])
    assert capsys.readouterr().out.splitlines()[1:] == [
        "  Flash-oQ4e: no norms (no unsteered traces measured), 1 calibrations (calm)",
        "  Flash-ds4: norms from 3 traces, no calibrations"]
    cli.main(["model", "show", "Flash-ds4"])
    out = capsys.readouterr().out
    assert out.startswith("Flash-ds4 is checkpoint Flash-oQ4e (48 layers x 4 x 2560)")
    assert f"  1 vectors, 0 profiles ({models.model_dir('Flash-oQ4e')})" in out
    assert f"    Flash-ds4: norms from 3 traces, no calibrations ({models.served_dir('Flash-ds4')})" in out
    assert f"    Flash-oQ4e: no norms (no unsteered traces measured), 1 calibrations (calm) ({models.served_dir('Flash-oQ4e')})" in out
    models.check_shape("V4", models.shape_of(43, 4, 4096))
    with pytest.raises(SystemExit, match="error: V4 is 43 layers"):
        cli.main(["model", "link", "V4", "Flash-oQ4e"])


def test_the_model_command_says_an_os_error_in_one_line(monkeypatch):
    monkeypatch.setattr(models, "link", lambda s, c: (_ for _ in ()).throw(PermissionError(13, "Permission denied", "/x")))
    with pytest.raises(SystemExit) as e:
        cli.main(["model", "link", "S", "C"])
    assert str(e.value) == "error: [Errno 13] Permission denied: '/x' (run the link again once that is fixed)"


def test_the_tested_table_is_the_readmes_and_names_families_the_engines_report():
    """models.TESTED is the one list of what metalscope has been run on: the README's Backends table is it, row for
    row; a ds4 family is one a GGUF architecture maps to (ds4/vocab.py FAMILIES), so no row is unmatchable."""
    from metalscope.ds4.vocab import FAMILIES
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    label = {"omlx": "oMLX", "ds4": "ds4"}
    rows = [f"| {label[e]} | {', '.join(names.values())} |" for e, names in models.TESTED.items()]
    assert "\n".join(["| backend | run on |", "|---|---|", *rows]) + "\n" in readme
    assert set(models.TESTED["ds4"]) <= set(FAMILIES.values())
    assert models.tested("omlx", "qwen3_5_moe") and models.tested("ds4", "deepseek41")
    assert not models.tested("omlx", "deepseek41") and not models.tested("omlx", None) and not models.tested("vllm", "qwen3_5")


def test_a_change_waiting_on_a_record_that_a_link_moved_takes_the_new_checkpoints_lock():
    import threading
    from metalscope import config
    models.check_shape("S", FLASH)
    models.check_shape("C", FLASH)
    lock = lambda ckpt: config.file_lock(models.models_dir() / ckpt / models.LOCK)
    for change, done in ((lambda: models.update("S", lens="L"), lambda rec: rec.get("lens") == "L"),
                         (lambda: models.check_shape("S", FLASH), lambda rec: "S" in rec["served"])):
        models.write_record(dict(name="S", shape=FLASH, served=["S"]))
        models.write_record(dict(name="C", shape=FLASH, served=["C"]))
        with lock("S"):
            th = threading.Thread(target=change)
            th.start()
            th.join(0.3)
            models.write_record(dict(name="S", shape=FLASH, served=[]))       # a link, meanwhile: S is C's now
            models.write_record(dict(name="C", shape=FLASH, served=["C", "S"]))
            c = lock("C")
            c.__enter__()
        th.join(0.3)
        assert th.is_alive()                                           # it waits for C's lock, not S's
        models.write_record(dict(models.record("C"), jlens="J"))
        c.__exit__(None, None, None)
        th.join()
        rec = models.record("C")
        assert rec["jlens"] == "J" and done(rec)


def test_a_record_that_is_not_a_json_object_reads_as_nothing_recorded():
    (models.models_dir() / "M").mkdir(parents=True)
    (models.models_dir() / "M" / models.RECORD).write_text("[1]")
    assert models.record("M") == dict(name="M", shape=None, served=[])
    assert models.check_shape("M", FLASH)["served"] == ["M"]                 # written afresh

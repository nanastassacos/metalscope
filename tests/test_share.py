"""Sharing concepts (share.py): ``concept pack`` writes a package of a checkpoint's concepts, ``concept pull`` places
it for a served model of the same shape under another name, in a fresh data home."""

import json
import os
import shutil

import pytest

from metalscope import cli, config, models, share, steer
from metalscope import concepts as C
from metalscope.concepts.corpus import DATA_DIR

from .helpers import fake_profile, write_norms

SHAPE = models.shape_of(3, 1, 64)           # fake_profile's: layers 0-2, width 64, one stream


def write_texts(name, pos=("a witty line",) * 8, writer="Claude Fable 5.1 (by hand)", d=None, **learned):
    d = d or models.concept_dir("V4-ds4", name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "concept.json").write_text(json.dumps(dict(title=name.title(), description="witty", kind="style",
                                                    learned=dict(writer=writer, date="2026-10-24", **learned))))
    (d / "pos.txt").write_text("\n".join(pos) + "\n")
    (d / "neg.txt").write_text("\n".join(["a plain line"] * len(pos)) + "\n")
    (d / "task.json").write_text(json.dumps(dict(items=[dict(prompt="p", options=["x", "y"], target="x")])))
    (d / "writing.json").write_text(json.dumps(dict(prompts=["write it"])))
    return d


def calibrate(served, name, dose=0.4):
    """A calibration of the profile as it is now, as ``concept calibrate`` leaves it."""
    d = models.calibration_dir(served, name)
    d.mkdir(parents=True, exist_ok=True)
    cal = dict(version=4, concept=name, model=models.checkpoint_of(served), served=served, layer=1, turns=18,
               default_dose=dose, good=15, strengths=[dict(dose=0.2, good=8), dict(dose=dose, good=15)],
               profile_mtime=models.profile_path(served, name).stat().st_mtime, grader=dict(model=served))
    (d / "calibration.json").write_text(json.dumps(cal))
    (d / "replies.jsonl").write_text("{}\n")


@pytest.fixture
def package(tmp_path, monkeypatch):
    """The packer's home: V4-ds4 with wry (texts by hand, calibrated) and fluffy (the shipped texts), packed."""
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "packer"))
    models.check_shape("V4-ds4", SHAPE)
    write_texts("wry", corpus="/Users/packer/texts/wry")              # --corpus DIR: a path on the packer's disk
    models.concept_dir("V4-ds4", "fluffy").mkdir(parents=True)       # learning a shipped concept copies its texts in
    for f in share.TEXTS:
        if (DATA_DIR / "fluffy" / f).is_file():
            (models.concept_dir("V4-ds4", "fluffy") / f).write_text((DATA_DIR / "fluffy" / f).read_text())
    for name in ("wry", "fluffy"):
        _, prof = fake_profile(None, model="V4-ds4", name=name)
        prof["measured_on"] = dict(engine="ds4", served="V4-ds4")
        prof["meter_layer"] = 1                                          # an older pack's meter fields: carried, unread
        for st in prof["stats"].values():
            st.update(scores_pos=[1.0], scores_neg=[-1.0])
        models.profile_path("V4-ds4", name).write_text(json.dumps(prof))
        models.profile_path("V4-ds4", name).with_suffix(".html").write_text("old")   # an old page: not packed
    calibrate("V4-ds4", "wry")
    out = tmp_path / "pkg"
    cli.main(["concept", "pack", "V4-ds4", str(out), "--base-model", "deepseek-ai/DeepSeek-V4.1-Flash", "--license", "mit",
              "--all"])
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "mine"))
    return out


def test_pack_writes_the_concept_folders_a_manifest_and_a_readme(package, capsys):
    m = json.loads((package / "manifest.json").read_text())
    assert m["format"] == 1 and m["base_model"] == "deepseek-ai/DeepSeek-V4.1-Flash" and m["checkpoint"] == "V4-ds4"
    assert m["license"] == "mit"
    assert m["shape"] == SHAPE and m["made_on"] == [dict(served="V4-ds4", engine="ds4")]
    assert m["concepts"] == [dict(name="fluffy", steer_layer=1), dict(name="wry", steer_layer=1)]
    h = package / "wry"
    assert {p.name for p in h.iterdir()} == {"concept.json", "pos.txt", "neg.txt", "task.json", "writing.json", "V4-ds4"}
    assert {p.name for p in (h / "V4-ds4").iterdir()} == {"vector.safetensors", "vector.json", "profile.json", "reference"}
    assert {p.name for p in (h / "V4-ds4" / "reference" / "V4-ds4").iterdir()} == {"calibration.json", "replies.jsonl"}
    assert (package / "fluffy" / "pos.txt").read_text() == (DATA_DIR / "fluffy" / "pos.txt").read_text()
    learned = json.loads((h / "concept.json").read_text())["learned"]
    assert learned == dict(writer="Claude Fable 5.1 (by hand)", date="2026-10-24")      # no local path ships
    readme = (package / "README.md").read_text()
    assert "texts: hand-written by Claude Fable 5.1 (2026-10-24)" in readme
    assert "on the packer's copy V4-ds4: 15 of 18 replies good at dose 0.4, its best (a good one: style," in readme
    assert "hold" not in readme and "bar" not in readme
    assert "fluffy" in readme and "on the packer's copies: not calibrated in conversation" in readme
    # The model card: Hugging Face's front matter, the licence named (no placeholder), no dose given as a default.
    assert readme.startswith('---\nbase_model: "deepseek-ai/DeepSeek-V4.1-Flash"\nbase_model_relation: adapter\n'
                             'license: "mit"\nlibrary_name: metalscope\ntags:\n- steering-vectors\n')
    assert "\n- gguf\n---\n" in readme and "- mlx" not in readme          # measured on ds4
    assert "## Licences\n\nThis package: mit (LICENSE).\n" in readme and "State this package" not in readme
    assert "good start" not in readme and "default dose" not in readme
    assert "They are references, not defaults: calibrate your own with `metalscope concept calibrate`." in readme
    assert f'`uv tool install "metalscope[share]"` (PyPI, coming) or `uv tool install "metalscope[share] @ git+{config.REPO_URL}"`' in readme
    lic = (package / "LICENSE").read_text()
    assert "Licence: mit." in lic and "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash" in lic
    assert "—" not in readme and "—" not in lic


def test_pull_places_the_concepts_for_another_served_name_of_the_same_shape(package, capsys):
    capsys.readouterr()
    with pytest.raises(SystemExit, match="Flash-mine's shape is not recorded yet: serve it once first"):
        cli.main(["concept", "pull", str(package), "Flash-mine"])
    models.check_shape("Flash-mine", SHAPE)                 # served once
    cli.main(["concept", "pull", str(package), "Flash-mine"])
    out = capsys.readouterr().out
    assert out.startswith("Flash-mine is now a copy of checkpoint V4-ds4")
    assert "  placed: fluffy, wry" in out
    assert "Flash-mine has no dose units yet: `metalscope on`, one real chat with it" in out
    assert "  the package is for deepseek-ai/DeepSeek-V4.1-Flash; you serve Flash-mine" in out
    assert "  wry: on the packer's copy V4-ds4, 15 of 18 replies good at dose 0.4" in out
    assert "  fluffy: not calibrated on the packer's copies" in out
    assert "  calibrate your own: metalscope concept calibrate fluffy Flash-mine (then /steer fluffy uses your dose)" in out
    assert "/steer wry 0.4" not in out and "good start" not in out
    # Created and linked: the package's checkpoint, with SERVED listed.
    assert models.checkpoint_of("Flash-mine") == "V4-ds4" and models.record("V4-ds4")["served"] == ["Flash-mine"]
    assert not (models.models_dir() / "Flash-mine").exists()
    # The texts in the model's concept folder, beside the vector; no definition is written.
    meta = json.loads((models.concept_dir("Flash-mine", "wry") / "concept.json").read_text())
    assert meta["learned"]["writer"] == "Claude Fable 5.1 (by hand)" and C.texts_dir("Flash-mine", "wry") is not None
    assert (models.concept_dir("Flash-mine", "fluffy") / "pos.txt").read_text() == (DATA_DIR / "fluffy" / "pos.txt").read_text()
    assert not models.concepts_dir().exists()
    # The vector and profile under the user's checkpoint, the packer's copy in measured_on; no references, no calibration.
    prof = json.loads(models.profile_path("Flash-mine", "wry").read_text())
    assert prof["model"] == "V4-ds4" and prof["measured_on"] == dict(engine="ds4", served="V4-ds4", checkpoint="V4-ds4")
    assert models.vector_path("Flash-mine", "wry").exists()
    assert not (models.concept_dir("Flash-mine", "wry") / "reference").exists()
    assert C.calibration_of("Flash-mine", "wry") is None and models.calibrated_copies("Flash-mine", "wry") == []
    # /steer wry 0.4 steers; /steer wry asks for a dose; a rewrite of the profile (grades) stays here.
    write_norms("Flash-mine", [0, 1, 2])
    plan = steer.build_plan(steer.parse_header("wry@0.4/1"), "Flash-mine", "r", hc_count=1)
    assert list(plan.vectors) == [1]
    with pytest.raises(ValueError, match="wry has no calibrated dose on Flash-mine"):
        steer.with_defaults(steer.parse_header("wry"), "Flash-mine")
    assert C.write_profile(prof) == models.concept_dir("Flash-mine", "wry")
    # concept show says where it came from and whose copy it was measured on.
    cli.main(["concept", "show", "wry", "Flash-mine"])
    shown = capsys.readouterr().out
    assert f"  pulled from {package.resolve()}; measured on V4-ds4" in shown
    assert json.loads((models.concept_dir("Flash-mine", "wry") / "vector.json").read_text())["source"] == dict(
        folder=str(package.resolve()))


def test_pull_onto_a_checkpoint_with_data_of_its_own_rewrites_the_profile_and_keeps_the_users_vectors(package, capsys):
    models.check_shape("Qwen-mine", SHAPE)
    fake_profile(None, model="Qwen-mine", name="wry")  # learned here by the user, from texts of its own
    mine = write_texts("wry", d=models.concept_dir("Qwen-mine", "wry"))
    before = models.profile_path("Qwen-mine", "wry").read_text()
    out = share.pull(str(package), "Qwen-mine")
    assert out["checkpoint"] == "Qwen-mine" and not out["linked"]
    assert out["placed"] == ["fluffy"] and out["kept"] == ["wry"]
    assert models.profile_path("Qwen-mine", "wry").read_text() == before
    prof = json.loads(models.profile_path("Qwen-mine", "fluffy").read_text())
    assert prof["model"] == "Qwen-mine" and prof["measured_on"]["checkpoint"] == "V4-ds4"
    assert "package" not in json.loads((mine / "concept.json").read_text())


def test_pull_keeps_a_vector_of_the_users_that_has_no_profile(package):
    models.check_shape("Flash-mine", SHAPE)
    d = models.concept_dir("Flash-mine", "wry")
    d.mkdir(parents=True)
    (d / "vector.safetensors").write_bytes(b"mine")
    (d / "vector.json").write_text(json.dumps(dict(source="the J-lens")))
    out = share.pull(str(package), "Flash-mine")
    assert out["kept"] == ["wry"] and out["placed"] == ["fluffy"]
    assert (d / "vector.safetensors").read_bytes() == b"mine" and not models.profile_path("Flash-mine", "wry").exists()


def test_a_shape_mismatch_refuses_and_writes_nothing(package):
    models.check_shape("Small", models.shape_of(4, 1, 64))
    with pytest.raises(SystemExit, match=r"error: Small is 4 layers x 1 x 64, the package is for 3 layers x 1 x 64"):
        cli.main(["concept", "pull", str(package), "Small"])
    assert not models.concepts_dir().exists() and models.checkpoint_of("Small") == "Small"


def test_a_name_the_model_has_texts_of_is_kept(package, capsys):
    models.check_shape("Flash-mine", SHAPE)
    mine = write_texts("wry", pos=("another concept entirely",) * 8, d=models.concept_dir("Flash-mine", "wry"))
    before = {f.name: f.read_bytes() for f in mine.iterdir()}
    capsys.readouterr()
    cli.main(["concept", "pull", str(package), "Flash-mine"])
    assert "  kept yours (the package's is not placed): wry" in capsys.readouterr().out
    assert {f.name: f.read_bytes() for f in mine.iterdir()} == before and not models.vector_path("Flash-mine", "wry").exists()
    assert models.vector_path("Flash-mine", "fluffy").exists()


def test_pulling_a_newer_package_replaces_its_own_concepts(package):
    models.check_shape("Flash-mine", SHAPE)
    first = share.pull(str(package), "Flash-mine")
    assert share.pull(str(package), "Flash-mine")["unchanged"] == ["fluffy", "wry"]        # the same package again
    # A newer version: other texts and another profile for wry.
    m = json.loads((package / "manifest.json").read_text())
    (package / "manifest.json").write_text(json.dumps(dict(m, made="2099-01-01T00:00:00")))
    (package / "wry" / "pos.txt").write_text("a wittier line\n" * 8)
    p = package / "wry" / "V4-ds4" / "profile.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), title="Wry, take two")))
    out = share.pull(str(package), "Flash-mine")
    assert out["placed"] == ["fluffy", "wry"] and out["package"] != first["package"]
    assert (models.concept_dir("Flash-mine", "wry") / "pos.txt").read_text() == "a wittier line\n" * 8
    assert json.loads(models.profile_path("Flash-mine", "wry").read_text())["title"] == "Wry, take two"
    assert json.loads(models.profile_path("Flash-mine", "wry").read_text())["package"].endswith("@2099-01-01T00:00:00")


def test_a_package_holds_only_what_metalscope_reads(package, tmp_path, monkeypatch):
    """An old profile page in the packer's folder is not packed; a package that holds one is refused, as any file
    metalscope does not read."""
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "packer"))
    (models.concept_dir("V4-ds4", "wry") / "profile.html").write_text("<html>an old page</html>")
    share.pack("V4-ds4", package, "x/y", license="mit", all_=True)
    assert not (package / "wry" / "V4-ds4" / "profile.html").exists()
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "mine"))
    models.check_shape("Flash-mine", SHAPE)
    out = share.pull(str(package), "Flash-mine")
    assert out["linked"] and "warning" not in out
    # The reference: the packer's copy with the most good replies at its default.
    assert "  wry: on the packer's copy V4-ds4, 15 of 18 replies good at dose 0.4" in "\n".join(share.next_steps(out, "Flash-mine"))
    (package / "wry" / "V4-ds4" / "profile.html").write_text("<html>an old page</html>")
    with pytest.raises(ValueError, match="holds profile.html, which metalscope does not read"):
        share.pull(str(package), "Flash-mine")


def test_pack_refuses_a_folder_that_is_not_a_package_and_a_missing_base_model(package, tmp_path, monkeypatch):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "packer"))
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="is not empty and holds no package"):
        share.pack("V4-ds4", tmp_path / "other", "x/y", license="mit")
    with pytest.raises(ValueError, match="--base-model"):
        share.pack("V4-ds4", tmp_path / "new", " ", license="mit")
    with pytest.raises(ValueError, match="no vector and profile on V4-ds4 for nope"):
        share.pack("V4-ds4", tmp_path / "new", "x/y", ["nope"], license="mit")
    # Packing again into the package replaces it.
    m = share.pack("V4-ds4", package, "x/y", ["wry"], license="mit")
    assert [c["name"] for c in m["concepts"]] == ["wry"] and not (package / "fluffy").exists()


def test_a_hugging_face_repo_needs_huggingface_hub(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_hub(name, *a, **kw):
        if name == "huggingface_hub":
            raise ImportError(name)
        return real(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", no_hub)
    with pytest.raises(ValueError, match=r'needs huggingface_hub: `uv tool install "metalscope\[share\]"` \(PyPI, coming\) or '
                                         r'`uv tool install "metalscope\[share\] @ git\+https://github.com/nanastassacos/metalscope"`'):
        share.fetch("someone/metalscope-v4-concepts")
    with pytest.raises(ValueError, match="neither a folder nor a Hugging Face repo"):
        share.fetch("/no/such/folder")


def test_norms_from_few_or_short_traces_warn_and_never_refuse(capsys):
    assert "measured on 1 short trace (300 tokens)" in steer.norms_note(dict(model="m", traces=["t"], tokens=300))
    assert steer.norms_note(dict(model="m", traces=["t"], tokens=5000)) is None
    assert steer.norms_note(dict(model="m", traces=["a", "b"])) is None and steer.norms_note(None) is None   # no count
    write_norms("m", [0, 1])
    p = models.norms_path("m")
    p.write_text(json.dumps(dict(json.loads(p.read_text()), tokens=300)))      # one short trace
    cli.main(["vector", "norms", "m"])
    assert "warning: dose units on m measured on 1 short trace (300 tokens): `metalscope vector norms m --refresh`" \
        in capsys.readouterr().out


def test_the_concepts_route_lists_pulled_concepts_and_warns_of_short_norms(package):
    from metalscope.live import MANAGER
    from .helpers import Tok, scope_client, tiny_llama
    models.check_shape("Flash-mine", SHAPE)
    share.pull(str(package), "Flash-mine")
    config.save_settings(mode="on")
    MANAGER.attach(tiny_llama(), "Flash-mine", Tok())
    MANAGER.norm_tables["Flash-mine"] = dict(model="Flash-mine", layers={1: 2.5}, streams={}, traces=["t"], tokens=300)
    try:
        r = scope_client().get("/scope/api/concepts?model=Flash-mine").json()
    finally:
        MANAGER.tracers.clear()
        MANAGER.norm_tables.pop("Flash-mine", None)
    assert r["norms_note"].startswith("dose units on Flash-mine measured on 1 short trace (300 tokens)")
    assert sorted(c["name"] for c in r["concepts"]) == ["fluffy", "wry"] and not any("error" in c for c in r["concepts"])


# ---------------------------------------------------------------- personal concepts, licence, real tensors, revisions, drift

def test_pack_takes_the_shipped_concepts_by_default_and_says_which_it_left_out(package, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "packer"))
    capsys.readouterr()
    cli.main(["concept", "pack", "V4-ds4", str(tmp_path / "pkg2"), "--base-model", "x/y", "--license", "mit"])
    out = capsys.readouterr().out
    assert [c["name"] for c in json.loads((tmp_path / "pkg2" / "manifest.json").read_text())["concepts"]] == ["fluffy"]
    assert "  left out (not shipped with metalscope; name them, or --all): wry" in out
    assert not (tmp_path / "pkg2" / "wry").exists()
    assert share.pack("V4-ds4", tmp_path / "pkg3", "x/y", ["wry"], license="mit")["left_out"] == ["fluffy"]
    with pytest.raises(ValueError, match="name the concepts or pass --all, not both"):
        share.pack("V4-ds4", tmp_path / "pkg4", "x/y", ["wry"], license="mit", all_=True)
    for p in models.profile_paths("V4-ds4"):                # only the packer's own left: nothing to pack by default
        if p.parent.name == "fluffy":
            p.unlink()
    with pytest.raises(ValueError, match="none of V4-ds4's concepts ships with metalscope: name the ones to pack, or --all"):
        share.pack("V4-ds4", tmp_path / "pkg5", "x/y", license="mit")


def test_pack_writes_the_licence_it_is_told(package, tmp_path, monkeypatch):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "packer"))
    with pytest.raises(SystemExit):                          # argparse: --license is required
        cli.main(["concept", "pack", "V4-ds4", str(tmp_path / "p"), "--base-model", "x/y"])
    with pytest.raises(ValueError, match="--license is the package's licence"):
        share.pack("V4-ds4", tmp_path / "p", "x/y", license="Apache 2.0")
    with pytest.raises(ValueError, match="--base-model is the base model's Hugging Face id"):
        share.pack("V4-ds4", tmp_path / "p", "DeepSeek V4", license="mit")
    with pytest.raises(ValueError, match="--license other needs --license-name"):
        share.pack("V4-ds4", tmp_path / "p", "x/y", license="other", license_name="qwen-community-1.0")
    text = tmp_path / "QWEN-LICENSE.txt"
    text.write_text("Qwen Community License 1.0\nthe text itself\n")
    with pytest.raises(ValueError, match="is not a file"):
        share.pack("V4-ds4", tmp_path / "p", "x/y", license="mit", license_file=tmp_path / "nope.txt")
    m = share.pack("V4-ds4", tmp_path / "p", "Qwen/Qwen3.8-Flash-Next", license="other",
                   license_name="qwen-community-1.0", license_file=text)
    assert m["license"] == "other" and m["license_name"] == "qwen-community-1.0"
    assert (tmp_path / "p" / "LICENSE").read_text() == text.read_text()
    readme = (tmp_path / "p" / "README.md").read_text()
    assert ('license: "other"\nlicense_name: "qwen-community-1.0"\nlicense_link: LICENSE\nlibrary_name: metalscope\n'
            in readme)
    assert "This package: qwen-community-1.0 (LICENSE)." in readme
    assert "The vectors are derived from Qwen/Qwen3.8-Flash-Next: its licence applies to them too." in readme


def write_safetensors(p, tensors, raw=None):
    """A safetensors file by hand: ``tensors`` {key: (dtype, shape)}, zeros for data; ``raw`` instead writes bytes."""
    import math
    import struct
    if raw is not None:
        p.write_bytes(raw)
        return
    head, at = {}, 0
    for k, (dtype, shape) in tensors.items():
        n = share.DTYPES.get(dtype, 1) * math.prod(shape)
        head[k] = dict(dtype=dtype, shape=list(shape), data_offsets=[at, at + n])
        at += n
    h = json.dumps(head).encode()
    p.write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * at)


@pytest.mark.parametrize("tensors,raw,match", [
    ({"v.1": ("F32", (32,))}, None, r"wry/V4-ds4/vector.safetensors's v.1 is 32; the manifest's shape \(3 layers x 1 x 64\) needs 64"),
    ({"v.5": ("F32", (64,))}, None, r"has a vector for layer 5; the manifest's shape \(3 layers x 1 x 64\) has 3 layers"),
    ({"v.1": ("I8", (64,))}, None, r"v.1 is I8; a vector is F32, F16, BF16"),
    ({"w": ("F32", (64,))}, None, r"holds a tensor 'w': a vector holds one per layer"),
    ({}, b"\x80\x02}q\x00(X\x01\x00\x00\x00vq\x01.", r"is not a safetensors file \(.*\): metalscope reads safetensors only"),
])
def test_pull_checks_the_vector_files_real_tensors(package, tensors, raw, match):
    write_safetensors(package / "wry" / "V4-ds4" / "vector.safetensors", tensors, raw)
    models.check_shape("Flash-mine", SHAPE)
    with pytest.raises(ValueError, match=match):
        share.pull(str(package), "Flash-mine")
    assert models.checkpoint_of("Flash-mine") == "Flash-mine" and not models.concepts_dir().exists()


@pytest.mark.parametrize("change,match", [
    (dict(reader="L1"), r"the package's wry/V4-ds4/profile.json cannot be read \("),
    (dict(title=["Wry"]), r"the package's wry/V4-ds4/profile.json cannot be read \(its title and description are text\)"),
    (dict(title="Wry\x1b]52;c;aGk=\x07"), r"profile.json cannot be read \(its title or description holds a control character\)"),
    (dict(description="dry\x9b2J"), r"profile.json cannot be read \(its title or description holds a control character\)"),
])
def test_pull_refuses_a_profile_the_server_could_not_list_or_one_that_would_drive_the_terminal(package, change, match):
    p = package / "wry" / "V4-ds4" / "profile.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), **change)))
    models.check_shape("Flash-mine", SHAPE)
    with pytest.raises(ValueError, match=match):
        share.pull(str(package), "Flash-mine")
    assert not models.concepts_dir().exists()


def test_pull_refuses_control_characters_in_texts_but_reads_tabs_and_windows_line_ends(package):
    models.check_shape("Flash-mine", SHAPE)
    (package / "wry" / "neg.txt").write_bytes(b"a plain\tline\r\n" * 8)
    share.pull(str(package), "Flash-mine")
    (package / "wry" / "neg.txt").write_text("a plain line\x1b[2J\n" * 8)
    with pytest.raises(ValueError, match=r"the package's wry/neg.txt holds a control character"):
        share.pull(str(package), "Flash-mine")
    (package / "wry" / "neg.txt").write_text("a plain line\n" * 8)
    meta = json.loads((package / "wry" / "concept.json").read_text())
    (package / "wry" / "concept.json").write_text(json.dumps(dict(meta, title="Wry\x1b]0;owned\x07")))
    with pytest.raises(ValueError, match=r"the package's wry/concept.json holds a control character"):
        share.pull(str(package), "Flash-mine")
    (package / "wry" / "concept.json").write_text(json.dumps(meta))
    (package / "wry" / "V4-ds4" / "notes\x1b.txt").write_text("x")
    with pytest.raises(ValueError, match=r"a file name in the package holds a control character"):
        share.pull(str(package), "Flash-mine")


ESC = "\x1b]0;owned\x07"


@pytest.mark.parametrize("change", [
    dict(position=f"mean{ESC}"), dict(n_pos=f"3{ESC}"), dict(writer=dict(first=0, jump=1, rule=f"p < 0.01{ESC}")),
    dict(measured_on=dict(served=f"V4-ds4{ESC}", engine="ds4")), dict(measured_on=dict(served="V4-ds4", engine=f"ds4{ESC}")),
])
def test_pull_refuses_a_control_character_in_what_concept_show_prints(package, change):
    p = package / "wry" / "V4-ds4" / "profile.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), **change)))
    models.check_shape("Flash-mine", SHAPE)
    with pytest.raises(ValueError, match=r"the package's wry/V4-ds4/profile.json cannot be read \(what concept show "
                                         r"prints of it holds a control character\)"):
        share.pull(str(package), "Flash-mine")
    assert not models.concepts_dir().exists()


def test_a_reference_calibration_without_numbers_is_not_reported(package):
    p = package / "wry" / "V4-ds4" / share.REFERENCE / "V4-ds4" / "calibration.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), good=f"15{ESC}")))
    models.check_shape("Flash-mine", SHAPE)
    out = share.pull(str(package), "Flash-mine")
    assert out["placed"] == ["fluffy", "wry"] and out["references"]["wry"] == []
    assert "\x1b" not in "\n".join(share.next_steps(out, "Flash-mine"))


def test_vocabulary_tokens_and_model_replies_in_a_profile_pull_as_they_are(package):
    """A profile holds the model's own tokens (a BPE token can be ESC [) and its broken high-dose replies: data,
    never printed as it is, so they pull."""
    p = package / "wry" / "V4-ds4" / "profile.json"
    prof = json.loads(p.read_text())
    prof["lens_words"] = {"1": ["\x1b[", "wry"]}
    rates = dict(target=0.1, concept_words=0.1)
    prof["generation"] = dict(layer=1, strength=0.4, reader_strength=0.4, below_edge=True,
                              summary=dict(base=rates, concept=rates, control=rates),
                              ladder=[dict(strength=3.2, text="garbled\x1c\x1c reply")])
    p.write_text(json.dumps(prof))
    models.check_shape("Flash-mine", SHAPE)
    assert share.pull(str(package), "Flash-mine")["placed"] == ["fluffy", "wry"]
    assert json.loads(models.profile_path("Flash-mine", "wry").read_text())["lens_words"]["1"][0] == "\x1b["


@pytest.mark.parametrize("bad", ["../x", "a/b", ".x"])
def test_a_manifest_naming_a_path_refuses(package, bad):
    m = json.loads((package / "manifest.json").read_text())
    for what, change in (("concept", dict(concepts=[dict(name=bad, steer_layer=1)])), ("checkpoint", dict(checkpoint=bad))):
        (package / "manifest.json").write_text(json.dumps(dict(m, **change)))
        with pytest.raises(ValueError, match=f"the package names a bad {what} "):
            share.read_manifest(package)


def test_pull_refuses_a_vector_in_another_format(package):
    (package / "wry" / "V4-ds4" / "vector.pt").write_bytes(b"\x80\x02pickle")
    models.check_shape("Flash-mine", SHAPE)
    with pytest.raises(SystemExit, match=r"error: the package's wry/V4-ds4/ holds vector.pt, which metalscope does not read"):
        cli.main(["concept", "pull", str(package), "Flash-mine"])
    assert not models.concepts_dir().exists()


def test_a_good_vector_file_passes_the_check(package):
    share.check_vector(package / "wry" / "V4-ds4" / "vector.safetensors", SHAPE, "wry")      # learn's own file
    t = share.tensors(package / "wry" / "V4-ds4" / "vector.safetensors")
    assert sorted(t) == ["v.0", "v.1", "v.2"] and t["v.1"] == ("F32", [64], 256)


def hf_download(package, tmp_path, calls, commit="0123abcd"):
    """A fake huggingface_hub download, laid out as its cache is: the snapshot folder named by its commit holds only
    links, each into the repo's ``blobs/``."""
    import os
    import shutil

    def download(repo, revision):
        calls.append((repo, revision))
        root = tmp_path / "hub" / f"models--{repo.replace('/', '--')}"
        d = root / "snapshots" / commit
        if not d.exists():
            (root / "blobs").mkdir(parents=True)
            for i, f in enumerate(sorted(p for p in package.rglob("*") if p.is_file())):
                blob, link = root / "blobs" / f"b{i}", d / f.relative_to(package)
                shutil.copyfile(f, blob)
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(os.path.relpath(blob, link.parent))
        return d
    return download


def test_pull_from_a_repo_at_a_revision_records_where_it_came_from(package, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(share, "_download", hf_download(package, tmp_path, calls))
    models.check_shape("Flash-mine", SHAPE)
    capsys.readouterr()
    cli.main(["concept", "pull", "someone/metalscope-concepts-V4@v1.0", "Flash-mine"])
    assert calls == [("someone/metalscope-concepts-V4", "v1.0")]
    out = capsys.readouterr().out
    assert "pulled someone/metalscope-concepts-V4 at revision v1.0 (commit 0123abcd) for Flash-mine" in out
    src = dict(repo="someone/metalscope-concepts-V4", revision="v1.0", commit="0123abcd")
    assert models.vector_path("Flash-mine", "wry").is_file() and not models.vector_path("Flash-mine", "wry").is_symlink()
    assert json.loads((models.concept_dir("Flash-mine", "wry") / "vector.json").read_text())["source"] == src
    prof = json.loads(models.profile_path("Flash-mine", "wry").read_text())
    assert prof["source"] == src and prof["package"] == "someone/metalscope-concepts-V4@0123abcd"
    cli.main(["concept", "show", "wry", "Flash-mine"])
    assert "  pulled from someone/metalscope-concepts-V4 at revision v1.0 (commit 0123abcd); measured on" in capsys.readouterr().out
    # No revision: the default branch; the same commit again is the same package.
    again = share.pull("someone/metalscope-concepts-V4", "Flash-mine")
    assert calls[-1] == ("someone/metalscope-concepts-V4", None) and again["unchanged"] == ["fluffy", "wry"]
    assert again["source"]["revision"] is None
    for bad in ("someone/repo@../x", "someone/repo@", "someone/repo@a b"):
        with pytest.raises(ValueError, match="neither a folder nor a Hugging Face repo"):
            share.fetch(bad)


def test_a_link_out_of_a_package_refuses_and_copies_nothing(package, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE\n")
    (package / "wry" / "pos.txt").unlink()
    (package / "wry" / "pos.txt").symlink_to(secret)
    models.check_shape("Flash-mine", SHAPE)
    with pytest.raises(ValueError, match=r"the package's wry/pos.txt links outside it"):
        share.pull(str(package), "Flash-mine")
    assert not models.models_dir().joinpath("V4-ds4").exists() and models.checkpoint_of("Flash-mine") == "Flash-mine"
    # A link to a file inside the package is a file of it.
    (package / "wry" / "pos.txt").unlink()
    (package / "wry" / "pos.txt").symlink_to("neg.txt")
    share.pull(str(package), "Flash-mine")
    assert (models.concept_dir("Flash-mine", "wry") / "pos.txt").read_text() == (package / "wry" / "neg.txt").read_text()


def test_a_hub_cache_snapshot_pulls_as_a_folder(package, tmp_path):
    """`hf download owner/repo` without --local-dir prints the cache's snapshot folder: its links into ../../blobs
    are the package's own."""
    snap = hf_download(package, tmp_path, [])("someone/pkg", None)
    models.check_shape("Flash-mine", SHAPE)
    out = share.pull(str(snap), "Flash-mine")
    assert out["placed"] == ["fluffy", "wry"] and models.vector_path("Flash-mine", "wry").is_file()
    # A link to a file beside blobs/ in the repo's cache folder, not into blobs/: outside the package.
    (snap.parent.parent / "private.txt").write_text("PRIVATE\n")
    (snap / "wry" / "pos.txt").unlink()
    (snap / "wry" / "pos.txt").symlink_to("../../../private.txt")
    with pytest.raises(ValueError, match=r"the package's wry/pos.txt links outside it"):
        share.pull(str(snap), "Flash-mine")


def test_a_hugging_face_pull_downloads_only_what_a_package_holds(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.utils import filter_repo_objects
    asked = {}
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: asked.update(kw) or str(tmp_path))
    share._download("someone/pkg", "v1")
    assert (asked["repo_id"], asked["revision"]) == ("someone/pkg", "v1")
    repo = ["manifest.json", "README.md", "LICENSE", "model-00001-of-00009.safetensors", "big.txt", "wry/pos.txt",
            "wry/concept.json", "wry/notes.bin", "wry/V4/vector.safetensors", "wry/V4/profile.json",
            "wry/V4/reference/V4/calibration.json"]
    assert list(filter_repo_objects(repo, allow_patterns=asked["allow_patterns"])) == [
        "manifest.json", "wry/pos.txt", "wry/concept.json", "wry/V4/vector.safetensors", "wry/V4/profile.json",
        "wry/V4/reference/V4/calibration.json"]


def test_a_round_trip_keeps_the_texts_with_the_vector_once(package, tmp_path, monkeypatch):
    """Pack puts the texts in <concept>/ only; pull puts them beside the vector in the model's folder, keeps a model's
    own, and replaces what this package placed before (the user's calibrations stay)."""
    assert not any(f.name in share.TEXTS for f in (package / "wry" / "V4-ds4").iterdir())
    assert (package / "wry" / "pos.txt").is_file()
    models.check_shape("Flash-mine", SHAPE)
    out = share.pull(str(package), "Flash-mine")
    assert out["placed"] == ["fluffy", "wry"] and "beside" not in out
    d = models.concept_dir("Flash-mine", "wry")
    assert sorted(f.name for f in d.iterdir() if f.is_file()) == [
        "concept.json", "neg.txt", "pos.txt", "profile.json", "task.json", "vector.json", "vector.safetensors", "writing.json"]
    calibrate("Flash-mine", "wry")                                   # the user's own calibration
    # A newer version of the package, without writing.json: replaced, nothing of the old placement left.
    m = json.loads((package / "manifest.json").read_text())
    (package / "manifest.json").write_text(json.dumps(dict(m, made="2099-01-01T00:00:00")))
    (package / "wry" / "writing.json").unlink()
    (package / "wry" / "pos.txt").write_text("a wittier line\n" * 8)
    assert share.pull(str(package), "Flash-mine")["placed"] == ["fluffy", "wry"]
    assert not (d / "writing.json").exists() and (d / "pos.txt").read_text() == "a wittier line\n" * 8
    assert models.calibration_path("Flash-mine", "wry").is_file()
    # Packed again from here: the texts the vector was learned from, once.
    share.pack("Flash-mine", tmp_path / "again", "x/y", license="mit", all_=True)
    assert (tmp_path / "again" / "wry" / "pos.txt").read_text() == "a wittier line\n" * 8
    assert not any(f.name in share.TEXTS for f in (tmp_path / "again" / "wry" / "V4-ds4").iterdir())
    # Another package with the same name: what the first placed is kept, and said so.
    other = tmp_path / "other"
    shutil.copytree(package, other)
    out = share.pull(str(other), "Flash-mine")
    assert (out["kept"], out["kept_other"]) == ([], ["fluffy", "wry"])
    assert (d / "pos.txt").read_text() == "a wittier line\n" * 8


def test_texts_in_a_packages_checkpoint_folder_are_refused_and_nothing_is_placed(package):
    """Only the texts in <concept>/ are checked for control characters; ones in <concept>/<checkpoint>/ would land
    in the model's folder unchecked, so a package holding them is refused."""
    theirs = package / "wry" / "V4-ds4"
    (theirs / "concept.json").write_text(json.dumps(dict(title="Wry", description="x", learned=dict(writer="evil\x1b[31mRED"))))
    (theirs / "pos.txt").write_text("unchecked\x07 line\n" * 8)
    models.check_shape("Flash-mine", SHAPE)
    with pytest.raises(ValueError, match=r"^the package's wry/V4-ds4/ holds concept.json, pos.txt, which metalscope does not read$"):
        share.pull(str(package), "Flash-mine")
    assert not models.models_dir().joinpath("V4-ds4").exists() and models.checkpoint_of("Flash-mine") == "Flash-mine"
    # Placing never takes a text from there either.
    (theirs / "concept.json").unlink()
    (theirs / "pos.txt").unlink()
    share.pull(str(package), "Flash-mine")
    (theirs / "pos.txt").write_text("unchecked\x07 line\n" * 8)
    share._place("wry", package / "wry", "V4-ds4", (package.resolve(),), "V4-ds4", "p@2", dict(folder="p"))
    assert (models.concept_dir("Flash-mine", "wry") / "pos.txt").read_text() == (package / "wry" / "pos.txt").read_text()


def test_a_concept_packed_without_texts_is_said_and_not_offered_to_calibrate(package, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "packer"))
    for f in share.TEXTS:
        (models.concept_dir("V4-ds4", "fluffy") / f).unlink(missing_ok=True)
    capsys.readouterr()
    cli.main(["concept", "pack", "V4-ds4", str(tmp_path / "p2"), "--base-model", "a/b", "--license", "mit", "--all"])
    assert "  packed without texts (none on V4-ds4): fluffy\n" in capsys.readouterr().out
    assert share.pack("V4-ds4", tmp_path / "p3", "a/b", license="mit", all_=True)["no_texts"] == ["fluffy"]
    readme = (tmp_path / "p2" / "README.md").read_text()
    assert "### fluffy\n" in readme and "- texts: none on the packer's model\n" in readme
    assert not (tmp_path / "p2" / "fluffy" / "pos.txt").exists()
    # Pulled: calibrating needs the model's texts, so it is not offered for fluffy.
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "mine2"))
    models.check_shape("Flash-mine", SHAPE)
    out = share.pull(str(tmp_path / "p2"), "Flash-mine")
    steps = share.next_steps(out, "Flash-mine")
    assert "  calibrate your own: metalscope concept calibrate wry Flash-mine (then /steer wry uses your dose)" in steps
    assert not any("calibrate fluffy" in line for line in steps)
    for f in share.TEXTS:
        (models.concept_dir("Flash-mine", "wry") / f).unlink(missing_ok=True)
    assert not any("concept calibrate" in line for line in share.next_steps(out, "Flash-mine"))


def test_a_vector_json_that_is_not_an_object_refuses_and_nothing_is_placed(package):
    (package / "wry" / "V4-ds4" / "vector.json").write_text("[1, 2]")
    models.check_shape("Flash-mine", SHAPE)
    with pytest.raises(ValueError, match=r"^the package's wry/V4-ds4/vector.json is not a JSON object$"):
        share.pull(str(package), "Flash-mine")
    assert not models.models_dir().joinpath("V4-ds4").exists() and models.checkpoint_of("Flash-mine") == "Flash-mine"


def test_a_placement_cut_short_leaves_nothing_under_the_concepts_name_and_the_next_pull_places_it(package, monkeypatch, capsys):
    models.check_shape("Flash-mine", SHAPE)
    share.pull(str(package), "Flash-mine")
    calibrate("Flash-mine", "wry")                                   # the user's calibration of the pulled concept
    m = json.loads((package / "manifest.json").read_text())
    (package / "manifest.json").write_text(json.dumps(dict(m, made="2099-01-01T00:00:00")))
    d = models.concept_dir("Flash-mine", "wry")
    def files():
        return {str(f.relative_to(d.parent)): f.read_bytes() for f in d.parent.rglob("*") if f.is_file()}
    before = files()

    def full(*a, **k):
        raise OSError(28, "No space left on device")
    with monkeypatch.context() as mp, pytest.raises(OSError):
        mp.setattr(C, "write_profile", full)
        share.pull(str(package), "Flash-mine")
    # The earlier placement as it was, its calibration kept, no dot folder left.
    assert files() == before and (d / "Flash-mine" / "calibration.json").is_file()
    assert sorted(p.name for p in d.parent.iterdir()) == ["fluffy", "wry"]
    # A pull killed part way leaves its dot folder: it is not a concept, and the next pull places over it.
    shutil.copytree(d, d.parent / ".wry.part")
    shutil.move(d / "Flash-mine", d.parent / ".wry.part" / "Flash-mine")
    assert [p.parent.name for p in models.profile_paths("Flash-mine")] == ["fluffy", "wry"]
    assert [p.parent.name for p in models.vector_paths("Flash-mine")] == ["fluffy", "wry"]
    assert ".wry.part" not in {c["name"] for c in C.list_concepts("Flash-mine")}
    assert models.served_copy("Flash-mine")["calibrations"] == []
    cli.main(["concept", "list"])
    assert "(1 more on models: metalscope concept list NAME)" in capsys.readouterr().out
    out = share.pull(str(package), "Flash-mine")
    assert out["placed"] == ["fluffy", "wry"] and out["kept"] == []
    assert json.loads(models.profile_path("Flash-mine", "wry").read_text())["package"].endswith("@2099-01-01T00:00:00")
    assert (d / "Flash-mine" / "calibration.json").is_file()        # given back, then kept
    assert sorted(p.name for p in d.parent.iterdir()) == ["fluffy", "wry"]


def newer(package):
    """The package as a newer version (another ``made``): pull replaces its own concepts."""
    m = json.loads((package / "manifest.json").read_text())
    (package / "manifest.json").write_text(json.dumps(dict(m, made="2099-01-01T00:00:00")))


def test_two_pulls_at_once_take_turns_and_keep_the_users_calibration(package, monkeypatch):
    import threading
    models.check_shape("Flash-mine", SHAPE)
    share.pull(str(package), "Flash-mine")
    calibrate("Flash-mine", "wry")
    newer(package)
    real, second = C.write_profile, []

    def placing(*a, **k):                    # the first pull, part way: a second pull starts
        if not second:
            th = threading.Thread(target=lambda: second.append(share.pull(str(package), "Flash-mine")))
            second.append(th)
            th.start()
            th.join(0.3)
            assert th.is_alive()                                     # waiting for the first
        return real(*a, **k)
    monkeypatch.setattr(C, "write_profile", placing)
    assert share.pull(str(package), "Flash-mine")["placed"] == ["fluffy", "wry"]
    second[0].join()
    assert second[1]["unchanged"] == ["fluffy", "wry"]
    d = models.concept_dir("Flash-mine", "wry")
    assert (d / "Flash-mine" / "calibration.json").is_file() and models.vector_path("Flash-mine", "wry").is_file()
    assert sorted(p.name for p in d.parent.iterdir()) == ["fluffy", "wry"]


def test_a_newer_pull_carries_across_what_metalscope_does_not_place(package):
    models.check_shape("Flash-mine", SHAPE)
    share.pull(str(package), "Flash-mine")
    d = models.concept_dir("Flash-mine", "wry")
    (d / "My notes.txt").write_text("mine")
    (d / ".hidden-notes").write_text("mine")
    (d / "vector.json.d").mkdir()
    newer(package)
    assert share.pull(str(package), "Flash-mine")["placed"] == ["fluffy", "wry"]
    assert {p.name for p in d.iterdir()} >= {"My notes.txt", ".hidden-notes", "vector.json.d", "profile.json"}


def test_a_pull_killed_after_moving_the_calibrations_gives_them_back_on_the_next_pull_of_any_version(package):
    models.check_shape("Flash-mine", SHAPE)
    share.pull(str(package), "Flash-mine")
    calibrate("Flash-mine", "wry")
    d = models.concept_dir("Flash-mine", "wry")
    (d / "My notes.txt").write_text("mine")
    (d.parent / ".wry.part").mkdir()
    for p in ("Flash-mine", "My notes.txt"):
        os.rename(d / p, d.parent / ".wry.part" / p)                       # carried across, then killed
    out = share.pull(str(package), "Flash-mine")
    assert out["unchanged"] == ["fluffy", "wry"] and models.served_copy("Flash-mine")["calibrations"] == ["wry"]
    assert (d / "My notes.txt").read_text() == "mine"
    assert sorted(p.name for p in d.parent.iterdir()) == ["fluffy", "wry"]


def test_a_placement_that_fails_or_is_killed_after_the_old_folder_went_aside_leaves_the_concept_as_it_was(package, monkeypatch):
    models.check_shape("Flash-mine", SHAPE)
    share.pull(str(package), "Flash-mine")
    calibrate("Flash-mine", "wry")
    d = models.concept_dir("Flash-mine", "wry")
    (d / "My notes.txt").write_text("mine")
    files = lambda: {str(p.relative_to(d)): p.read_bytes() for p in d.rglob("*") if p.is_file()}
    before, first = files(), (package / "manifest.json").read_text()
    newer(package)
    real = os.rename

    def rename(a, b):                       # the new folder's rename into place fails
        if str(a).endswith(".wry.part") and str(b) == str(d):
            raise OSError(5, "Input/output error")
        return real(a, b)
    with monkeypatch.context() as mp, pytest.raises(OSError):
        mp.setattr(share.os, "rename", rename)
        share.pull(str(package), "Flash-mine")
    assert files() == before and sorted(p.name for p in d.parent.iterdir()) == ["fluffy", "wry"]
    # Killed there instead: the old folder aside, what the user had in the dot folder. The next pull gives both back.
    (d.parent / ".wry.part").mkdir()
    for p in ("Flash-mine", "My notes.txt"):
        os.rename(d / p, d.parent / ".wry.part" / p)
    os.rename(d, d.parent / ".wry.old")
    (package / "manifest.json").write_text(first)                       # the version the old folder holds
    assert share.pull(str(package), "Flash-mine")["unchanged"] == ["wry"]       # fluffy went in before the failure
    assert files() == before and sorted(p.name for p in d.parent.iterdir()) == ["fluffy", "wry"]

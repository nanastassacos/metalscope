"""metalscope ds4 install against a stand-in for ds4: a small repository with a Makefile that makes a ds4-server,
and a patch made on its commit. The install clones it, checks the commit out, applies the patch, builds, stamps the
build with the patch's hash, does nothing again until the patch changes, and ds4 serve finds the build."""

import shutil
import subprocess

import pytest

from metalscope.ds4 import build

pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("make") is None, reason="git and make")
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main"]


def sh(*cmd, cwd=None):
    return subprocess.run(list(cmd), cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def ds4(tmp_path, monkeypatch):
    """A repository at a base commit, and a patch (format-patch) on it that the build needs."""
    src = tmp_path / "ds4-src"
    src.mkdir()
    sh(*GIT, "init", "-q", cwd=src)
    (src / "Makefile").write_text("ds4-server: server.c\n\tcp server.c ds4-server && chmod +x ds4-server\n")
    (src / "server.c").write_text("base\n")
    (src / "metal").mkdir()
    (src / "metal" / "k.metal").write_text("// kernels\n")
    sh(*GIT, "add", "-A", cwd=src)
    sh(*GIT, "commit", "-qm", "base", cwd=src)
    base = sh("git", "rev-parse", "HEAD", cwd=src)
    (src / "server.c").write_text("base\npatched\n")
    sh(*GIT, "commit", "-qam", "metalscope: the hooks", cwd=src)
    patch = tmp_path / "metalscope.patch"
    patch.write_text(sh("git", "format-patch", "--stdout", f"{base}..HEAD", cwd=src) + "\n")
    sh(*GIT, "reset", "-q", "--hard", base, cwd=src)           # upstream has no metalscope commit
    monkeypatch.setattr(build, "DS4_COMMIT", base)
    monkeypatch.setattr(build, "PATCH", patch)
    monkeypatch.setattr(build, "_check_tools", lambda: None)
    monkeypatch.setenv("METALSCOPE_DS4_DIR", str(tmp_path / "built"))
    return src, patch


def test_install_builds_once_and_again_only_when_the_patch_changes(ds4, tmp_path):
    src, patch = ds4
    log = []
    d = build.install(source=str(src), log=log.append)
    assert d == tmp_path / "built" and (d / "ds4-server").read_text() == "base\npatched\n" and build.current(d)
    assert build.find_dir() == d
    log.clear()
    build.install(source=str(src), log=log.append)
    assert log == [f"ds4 is built with this metalscope's patch at {d}"]               # nothing done
    patch.write_text(patch.read_text().replace("+patched", "+patched again"))
    assert not build.current(d)
    build.install(source=str(src), log=log.append)
    assert (d / "ds4-server").read_text() == "base\npatched again\n" and build.current(d)


def test_install_refuses_to_rebuild_while_a_ds4_server_runs_from_the_build(ds4, tmp_path, monkeypatch):
    """Rebuilding resets the checkout and replaces ds4-server and its Metal kernels under a running server."""
    from metalscope import backends
    src, patch = ds4
    d = build.install(source=str(src), log=lambda *_: None)
    patch.write_text(patch.read_text().replace("+patched", "+patched again"))
    exe = str(d.resolve() / "ds4-server")
    commands = {77: f"{exe} -m /m/q.gguf --port 8010", 88: "/elsewhere/ds4-server -m /m/q.gguf"}
    monkeypatch.setattr(backends, "_pgrep", lambda *a: list(commands) if a == ("-x", "ds4-server") else [])
    monkeypatch.setattr(backends, "_command", lambda pid: commands.get(pid, ""))
    with pytest.raises(SystemExit, match=rf"^ds4-server runs from {d} \(pid 77\): stop it first \(metalscope ds4 "
                                         r"stop\), then install again$"):
        build.install(source=str(src), log=lambda *_: None)
    assert (d / "ds4-server").read_text() == "base\npatched\n"
    del commands[77]                                                  # another build's server: no matter
    build.install(source=str(src), log=lambda *_: None)
    assert (d / "ds4-server").read_text() == "base\npatched again\n"


def test_serve_needs_a_build_and_takes_an_explicit_one(ds4, tmp_path):
    with pytest.raises(SystemExit, match="metalscope ds4 install"):
        build.find_dir()
    own = tmp_path / "own"
    with pytest.raises(SystemExit, match="has no ds4-server"):
        build.find_dir(own)
    (own / "metal").mkdir(parents=True)
    (own / "ds4-server").write_text("x")
    assert build.find_dir(own) == own.resolve() and not build.current(own)            # used, though not stamped


def test_install_says_so_when_the_pinned_commit_cannot_be_fetched(ds4, monkeypatch):
    src, _ = ds4
    monkeypatch.setattr(build, "DS4_COMMIT", "0" * 40)
    with pytest.raises(SystemExit, match=r"ds4 commit 0000000, which this metalscope's patch is made for, is not in "):
        build.install(source=str(src), log=lambda *_: None)


def test_install_refuses_a_directory_that_is_not_a_checkout(ds4, tmp_path):
    src, _ = ds4
    (tmp_path / "built").mkdir()
    (tmp_path / "built" / "something").write_text("mine")
    with pytest.raises(SystemExit, match="is not a ds4 checkout"):
        build.install(source=str(src), log=lambda *_: None)


def test_the_shipped_patch_holds_the_scope_dir_guards():
    """The patch's text only (applying it to DS4_COMMIT is CI's check): it is a format-patch, keeps ds4-server as
    upstream's without --scope-dir, and carries no commit trailers."""
    text = build.PATCH.read_text()
    assert text.startswith("From ") and "metalscope" in text and len(build.DS4_COMMIT) == 40
    assert '"--scope-dir"' in text          # the option proxy.start_ds4 passes: no scope file outside it
    assert "need version 2, width" in text  # the steering file frames.write_steering writes (removals after the adds)
    # without --scope-dir ds4-server is as upstream's: no scope object read, no token-id prompt, no tokenize endpoint
    assert "if (!scope_dir || !p ||" in text and "if (scope_dir && (**p == '-'" in text
    assert 'if (scope_dir && !strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/scope/tokenize"))' in text
    assert "Claude-Session" not in text and "Co-Authored-By" not in text     # the patch is ds4's to read: no trailers
    # macOS's open() fails with EINVAL on O_NOFOLLOW and O_NOFOLLOW_ANY together, so no trace could be created
    assert not [line for line in text.splitlines() if "O_NOFOLLOW |" in line and "O_NOFOLLOW_ANY" in line]


def test_the_patch_reads_v41s_engram_as_lookup_py_asks():
    """The engram inspection on DeepSeek V4.1 (lookup.py, engines/ds4.py next_logprobs): scope lookup false leaves the
    Engram add out in both places it enters the residual (one row, a batch); next_lp is no longer Qwen3.8's only, and
    V4.1's sync feeds its rows one at a time through the head (NXLP and NXTK, as Qwen3.8 writes them)."""
    text = build.PATCH.read_text()
    added = "\n".join(line[1:] for line in text.splitlines() if line.startswith("+"))
    assert "return !(g->scope && g->scope->lookup_off);" in added
    assert "if (ds41_engram_layer(il) && !ds41_image_at(g, g->pos) && ds41_engram_on(g)) {" in added
    assert "if (ds41_engram_layer(il) && ds41_engram_on(g)) {" in added
    assert "next_lp and lookup are read on Qwen3.8 only" not in added
    assert "nx_read ? s->logits : NULL" in added and "ds41_scope_next_lp(g->scope, s->logits, (uint32_t)i)" in added
    # read against the request's whole prompt: --batched-session syncs it in pieces, each a shorter prompt->len
    assert "(uint32_t)i + 1u < g->scope->nx_len" in added and "i + 1 < prompt->len" not in added
    assert "if (!scope_tracing(sc) || !sc->nx_prompt || pos + 1u >= sc->nx_len) return;" in added
    assert added.count("scope_put_next_lp(sc, ") == 2          # Qwen3.8's tiles and V4.1's rows: one writer
    # rows stepped alone run on the decode expert cache: the encoder's residency is given back first (nx is
    # UINT32_MAX without next_lp, so ds4's own sync is unchanged)
    assert "if (g->encoder_resident && (remaining < 768u || (uint32_t)i >= nx))" in added


def test_ds4s_licence_ships_beside_the_patch():
    """The patch carries ds4's code as diff context: ds4's MIT notice goes with it, in the package (hatch ships every
    file under src/metalscope)."""
    text = (build.PATCH.parent / "LICENSE.ds4").read_text()
    assert text.startswith("MIT License") and "The ds4.c authors" in text and "The ggml authors" in text
    assert "DeepSeek" in text and "shall be included in all" in text


def test_install_leaves_a_hand_made_checkout_alone_without_force(ds4, tmp_path):
    """A checkout install did not make (no stamp: METALSCOPE_DS4_DIR at one's own clone) is refused, not reset and
    cleaned; so is install's own with an edit git has not committed. --force resets either."""
    src, patch = ds4
    d = tmp_path / "built"
    sh("git", "clone", "-q", str(src), str(d))
    (d / "server.c").write_text("my work\n")
    (d / "notes.txt").write_text("mine\n")
    with pytest.raises(SystemExit, match=r"did not make \(no \.metalscope-build\.json\).*--force"):
        build.install(source=str(src), log=lambda *_: None)
    assert (d / "server.c").read_text() == "my work\n" and (d / "notes.txt").exists()
    shutil.rmtree(d)
    build.install(source=str(src), log=lambda *_: None)           # install's own checkout
    (d / "server.c").write_text("an edit\n")
    patch.write_text(patch.read_text().replace("+patched", "+patched again"))
    with pytest.raises(SystemExit, match=r"(?s)changes git has not committed.*server\.c.*--force"):
        build.install(source=str(src), log=lambda *_: None)
    assert (d / "server.c").read_text() == "an edit\n"
    build.install(source=str(src), force=True, log=lambda *_: None)
    assert (d / "ds4-server").read_text() == "base\npatched again\n" and build.current(d)


def test_serve_refuses_installs_build_when_it_is_older_than_this_metalscope(ds4, tmp_path):
    """A build `ds4 install` made from another patch would only exit while loading (an option serve passes, such as
    --scope-dir, unknown to it): refused up front, naming the fix. One named with --ds4-dir, or a hand-made one (no
    stamp) at METALSCOPE_DS4_DIR, is the user's and is used as it is."""
    src, patch = ds4
    d = build.install(source=str(src), log=lambda *_: None)
    build.refuse_stale()
    patch.write_text(patch.read_text().replace("+patched", "+patched again"))       # this metalscope's patch moved on
    with pytest.raises(SystemExit, match=r"metalscope's ds4 build is older than this metalscope: run metalscope "
                                         r"ds4 install, then start again \(.* is not ds4 "):
        build.find_dir()
    with pytest.raises(SystemExit, match="older than this metalscope"):
        build.refuse_stale()
    assert build.find_dir(d) == d.resolve()
    (d / build.STAMP).unlink()
    assert build.find_dir() == d


def test_the_patch_stops_a_trace_visibly():
    """What only a model run exercises, held in the patch's text: a trace that stops partway ends with END_'s finish
    then frames.STOPPED and why (the reader marks it partial); V4.1's per-row prefill fails a scoped request at the
    sweep, where the sweep decides it; next_lp's end of the trunk's batch is the forward's to fail."""
    from metalscope.ds4.frames import STOPPED
    added = "\n".join(line[1:] for line in build.PATCH.read_text().splitlines() if line.startswith("+"))
    assert f'"%s{STOPPED}%s"' in added
    assert "if (g->scope && !ds41_scope_sweep_begin(g->scope, total_count, batch_hc)) return false;" in added
    assert 'getenv("DS4_METAL_DISABLE_V41_BATCH' not in added
    assert "return flushed && begun;" in added
    assert "O_CREAT | O_EXCL" in added and "O_TRUNC" not in added

"""The oMLX startup hook: its .pth line loads metalscope only inside oMLX.app and never raises; boot stands down on
an untested oMLX; install and uninstall keep oMLX's packages out of the shared user site and clean up after
themselves. All in temp dirs: nothing here touches ~/.local or /Applications."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from metalscope import cli
from metalscope.omlx import install

INSIDE = "/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11/bin/python3"
INSIDE_PREFIX = "/Applications/oMLX.app/Contents/Resources/Python/cpython-3.11"


@pytest.fixture
def interp(monkeypatch):
    """The running interpreter as the .pth line sees it: sys.executable, sys.prefix and sys.path, restored after."""
    monkeypatch.setattr(sys, "path", list(sys.path))

    def be(executable="/usr/local/bin/python3.11", prefix="/usr/local"):
        monkeypatch.setattr(sys, "executable", executable)
        monkeypatch.setattr(sys, "prefix", prefix)
    return be


def fake_boot(tmp_path, monkeypatch, name, body="LOADED = True\n"):
    d = tmp_path / "hook"
    d.mkdir(exist_ok=True)
    (d / f"{name}.py").write_text(body)
    monkeypatch.delitem(sys.modules, name, raising=False)
    return d


def test_the_line_is_one_import_line(tmp_path):
    line = install.pth_line([tmp_path / "a", tmp_path / "b"])
    assert line.startswith("import ") and "\n" not in line          # site runs it as a .pth import line


@pytest.mark.parametrize("executable,prefix", [(INSIDE, "/x"), ("/usr/bin/python3", INSIDE_PREFIX)])
def test_inside_omlx_app_it_adds_the_paths_and_imports(tmp_path, monkeypatch, interp, executable, prefix):
    d = fake_boot(tmp_path, monkeypatch, "fakeboot_in")
    pkgs = tmp_path / "packages"
    interp(executable, prefix)
    exec(install.pth_line([d, pkgs], module="fakeboot_in"), {})
    assert sys.modules["fakeboot_in"].LOADED
    assert sys.path[-2:] == [str(d), str(pkgs)]


def test_any_other_python_gets_no_path_and_no_import(tmp_path, monkeypatch, interp):
    d = fake_boot(tmp_path, monkeypatch, "fakeboot_out")
    interp()
    before = list(sys.path)
    exec(install.pth_line([d], module="fakeboot_out"), {})
    assert "fakeboot_out" not in sys.modules and sys.path == before


def test_a_missing_or_broken_metalscope_is_silent(tmp_path, monkeypatch, interp, capsys):
    d = fake_boot(tmp_path, monkeypatch, "fakeboot_broken", body="raise RuntimeError('boom')\n")
    interp(INSIDE, INSIDE_PREFIX)
    before = list(sys.path)
    exec(install.pth_line([tmp_path / "gone"], module="fakeboot_nowhere"), {})   # metalscope removed
    exec(install.pth_line([d], module="fakeboot_broken"), {})                    # its import raises
    assert sys.path == before                                                    # the paths taken back out
    assert "fakeboot_broken" not in sys.modules
    assert capsys.readouterr() == ("", "")


def test_a_bundle_with_another_name_is_recognised_by_its_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    text = install.pth_text(tmp_path / "oMLX beta.app")
    assert f"{tmp_path}/oMLX beta.app/Contents/" in text and install.APP_MARK in text
    assert install.pth_text(Path("/Applications/oMLX.app")).count("/Contents/") == 1


def run_site(tmp_path, executable, hook_lines):
    """Python's own site processing of the .pth (no site-packages, so the only metalscope is the one given)."""
    site_dir = tmp_path / "site"
    site_dir.mkdir(exist_ok=True)
    (site_dir / install.PTH_NAME).write_text(hook_lines)
    code = ("import sys, site; sys.executable = sys.argv[1]; site.addsitedir(sys.argv[2]); "
            "print('metalscope.omlx.boot' in sys.modules, sys.argv[3] in sys.path)")
    return subprocess.run([sys.executable, "-I", "-S", "-c", code, executable, str(site_dir), str(tmp_path / "hook")],
                          capture_output=True, text=True, cwd=tmp_path)


def test_python_site_runs_the_line_only_inside_omlx_app(tmp_path):
    hook = tmp_path / "hook"
    (hook / "metalscope" / "omlx").mkdir(parents=True)
    for f in ("metalscope/__init__.py", "metalscope/omlx/__init__.py", "metalscope/omlx/boot.py"):
        (hook / f).write_text("")
    text = f"{install.PTH_HEADER}\n{install.pth_line([hook, hook / 'packages'])}\n"
    r = run_site(tmp_path, INSIDE, text)
    assert (r.stdout.split(), r.stderr) == (["True", "True"], "")
    r = run_site(tmp_path, "/usr/local/bin/python3.11", text)
    assert (r.stdout.split(), r.stderr) == (["False", "False"], "")
    (hook / "metalscope" / "omlx" / "boot.py").unlink()                          # metalscope gone: still silent
    r = run_site(tmp_path, INSIDE, text)
    assert (r.stdout.split(), r.stderr) == (["False", "False"], "")


# install and uninstall in a temp home


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A temp HOME with an oMLX.app (just its interpreter path) and a user site; nothing is run."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    app = tmp_path / "Applications" / "oMLX.app"
    py = install.bundled_python(app)
    py.parent.mkdir(parents=True)
    py.write_text("")
    site = tmp_path / "home/.local/lib/python3.11/site-packages"
    monkeypatch.setattr(install, "user_site", lambda a: site)
    monkeypatch.setattr(install.subprocess, "run", lambda cmd, *a, **k: pytest.fail(f"install ran {cmd}"))
    return dict(app=app, site=site)


def test_install_keeps_packages_out_of_the_user_site_and_uninstall_takes_only_ours(machine):
    app, site = machine["app"], machine["site"]
    (site / "requests").mkdir(parents=True)                                          # the user's own: stays
    (site / install.PTH_NAME).write_text(f"{install.hook_dir()}\n{install.PTH_IMPORT}\n")   # not this metalscope's
    assert not install.hook_state(app)["current"]

    lines = []
    install.install(app, log=lines.append)
    assert install.hook_state(app) | dict(pth=None) == dict(pth=None, exists=True, loads=install.PACKAGE, current=True)
    assert sorted(p.name for p in site.iterdir()) == [install.PTH_NAME, "requests"]
    manifest = json.loads((install.hook_dir() / install.MANIFEST).read_text())
    assert manifest == dict(pth=str(site / install.PTH_NAME), made=[], python=sys.executable)
    install.install(app, log=lines.append)                                           # again: same result
    assert install.hook_state(app)["current"]

    assert install.uninstall(app, log=lines.append)
    assert sorted(p.name for p in site.iterdir()) == ["requests"]
    assert not install.hook_dir().exists()
    assert not install.uninstall(app, log=lines.append)                              # nothing left to remove


def test_uninstall_finds_the_hook_from_the_manifest_when_omlx_app_is_gone(machine):
    app, site = machine["app"], machine["site"]
    install.install(app, log=lambda s: None)
    install.bundled_python(app).unlink()
    assert install.uninstall(app, log=lambda s: None)
    assert not (site / install.PTH_NAME).exists() and not install.hook_dir().exists()


def test_uninstall_removes_the_directories_install_made_and_the_job_links(machine):
    app, site = machine["app"], machine["site"]
    local = site.parents[2]
    assert not local.exists()                                    # no ~/.local yet: install makes the user site
    install.install(app, log=lambda s: None)
    install.install(app, log=lambda s: None)                     # a second install remembers what the first made
    assert (install.package_link() / "metalscope").is_symlink()  # a GPU job ran through its link
    (site.parent / "mine.txt").write_text("")                    # the user's, beside site-packages
    assert install.uninstall(app, log=lambda s: None)
    assert not site.exists() and sorted(p.name for p in site.parent.iterdir()) == ["mine.txt"]
    assert not (local / "share/metalscope").exists()             # the hook and the links: gone


def test_the_bundled_python_is_whichever_the_app_ships(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    app = tmp_path / "oMLX.app"
    (app / "Contents/Resources/Python/cpython-3.12/bin").mkdir(parents=True)
    (app / "Contents/Resources/Python/framework-mlx-base").mkdir()
    assert install.bundled_python(app) == app / "Contents/Resources/Python/cpython-3.12/bin/python3"
    env = install.bundled_env(app)
    assert env["PYTHONHOME"].endswith("cpython-3.12") and env["PYTHONPATH"].endswith("/lib/python3.12/site-packages")


@pytest.mark.parametrize("action", ["install", "uninstall"])
def test_no_omlx_app_is_one_line_saying_what_to_do(tmp_path, monkeypatch, action):
    monkeypatch.setenv("HOME", str(tmp_path))
    missing = tmp_path / "none.app"
    with pytest.raises(SystemExit) as exc:
        cli.main(["omlx", action, "--app", str(missing)])
    msg = str(exc.value)
    assert msg == f"no oMLX.app at {missing}: install oMLX ({install.OMLX_URL}) into /Applications, or pass --app PATH"
    assert not (tmp_path / ".local").exists()


# boot, in a Python of its own beside a stand-in omlx package

BOOT = """
import sys
import metalscope.omlx.boot
import omlx.scheduler
loader = omlx.scheduler.__spec__.loader
print(*(m for m in ("mlx", "numpy", "fastapi", "metalscope.live", "metalscope.web", "metalscope.omlx.patches")
        if m in sys.modules), "|", type(loader).__name__, loader.get_source("omlx.scheduler"))
"""


def boot_process(tmp_path, version, **env):
    pkg = tmp_path / "omlx"
    pkg.mkdir()
    for name, text in (("__init__", ""), ("_version", f"__version__ = {version!r}"), ("scheduler", "MARK = 1")):
        (pkg / f"{name}.py").write_text(text + "\n")
    src = Path(install.__file__).resolve().parents[2]
    return subprocess.run([sys.executable, "-c", BOOT], capture_output=True, text=True, cwd=tmp_path,
                          env={"PYTHONPATH": f"{src}:{tmp_path}", "HOME": str(tmp_path), "PATH": "/usr/bin:/bin", **env})


def test_boot_stands_down_on_an_untested_omlx_and_imports_nothing_of_the_stack(tmp_path):
    """It imported patches (and so mlx, numpy, fastapi, live.MANAGER with its thread pools) before it checked."""
    r = boot_process(tmp_path, "9.9.9")
    assert r.stdout.split() == ["|", "SourceFileLoader", "MARK", "=", "1"]      # nothing heavy; the loader is oMLX's own
    assert r.stderr.count("oMLX 9.9.9 is untested") == 1 and "Traceback" not in r.stderr


@pytest.mark.parametrize("version,env", [(max(install.TESTED_OMLX), {}), ("9.9.9", {"METALSCOPE_FORCE": "1"})])
def test_boot_patches_a_tested_omlx_and_a_failed_patch_leaves_the_module_imported(tmp_path, version, env):
    r = boot_process(tmp_path, version, **env)
    assert "metalscope.omlx.patches" in r.stdout.split("|")[0] and r.stdout.split()[-3:] == ["MARK", "=", "1"]
    assert "patching omlx.scheduler failed" in r.stderr and "untested" not in r.stderr   # the stand-in has no Scheduler

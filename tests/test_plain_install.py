"""A plain `pip install metalscope` (no extras) runs the everyday commands: nothing they import needs an extra."""

import subprocess
import sys
from pathlib import Path

EXTRAS = ("fastapi", "starlette", "anyio", "httpx", "uvicorn", "tokenizers", "mlx_lm")   # the [ds4] and [dev] ones

CODE = f"""
import sys
for name in {EXTRAS!r}:
    sys.modules[name] = None          # importing it raises ImportError, as on a plain install
from metalscope import backends, cli, client
from metalscope.ds4 import build
from metalscope.omlx import admin, install
assert install.TESTED_OMLX
try:
    cli.main(["help"])
except SystemExit as exc:
    assert exc.code in (0, None), exc.code
"""


def test_everyday_commands_import_without_extras(tmp_path):
    src = Path(__file__).resolve().parents[1] / "src"
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, cwd=tmp_path,
                       env={"PYTHONPATH": str(src), "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-2000:]


def test_bundled_command_imports_this_package_and_nothing_else_of_ours(tmp_path, monkeypatch):
    from metalscope.omlx import install
    monkeypatch.setenv("HOME", str(tmp_path))
    app = tmp_path / "oMLX.app"
    python, env = install.bundled_command(app, env={"HOME": str(tmp_path), "PYTHONPATH": "/elsewhere"})
    assert python == [str(app / "Contents/Resources/Python/cpython-3.11/bin/python3")]
    link_dir = Path(env["PYTHONPATH"].split(":")[-1])
    assert sorted(p.name for p in link_dir.iterdir()) == ["metalscope"]
    assert (link_dir / "metalscope").resolve() == install.PACKAGE
    assert "/elsewhere" not in env["PYTHONPATH"] and env["PYTHONHOME"].startswith(str(app))
    assert install.package_link() == link_dir                       # again: the same link, kept


def test_a_job_runs_in_the_python_of_the_install_whose_hook_the_server_runs(tmp_path, monkeypatch):
    """Inside oMLX's server (its hook loaded) a job runs in the Python that ran `omlx install` for that hook, as the
    hook's manifest records it: of two installs, the last to run `omlx install`, whatever order, and never what
    `pi install` recorded for Pi. With none recorded (a manifest from before, or none) the job is refused up front,
    naming the fix: oMLX's bundled Python cannot import mlx once the job's environment is cleaned. Outside oMLX (the
    ds4 proxy, run by the install's own Python) this Python."""
    import json
    import types

    import pytest

    from metalscope.learn import spawn, state
    from metalscope.omlx import install
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(spawn, "CHECKOUT", False)
    app = tmp_path / "Applications" / "Other.app"                     # installed under another name (--app)
    install.bundled_python(app).parent.mkdir(parents=True)
    install.bundled_python(app).write_text("")
    monkeypatch.setattr(install, "user_site", lambda a: tmp_path / "home/site")
    assert spawn.repo_python() == Path(sys.executable)                # outside oMLX: the ds4 proxy
    monkeypatch.setitem(sys.modules, install.BOOT_MODULE, types.ModuleType(install.BOOT_MODULE))
    assert spawn.repo_python() == Path(sys.executable)                # the hook alone is not oMLX's server
    monkeypatch.setitem(sys.modules, "omlx", types.ModuleType("omlx"))   # inside oMLX's server
    refused = "^metalscope does not know which Python runs it: run metalscope omlx install again$"
    with pytest.raises(ValueError, match=refused):
        spawn.repo_python()
    install.hook_dir().mkdir(parents=True)
    (install.hook_dir() / install.MANIFEST).write_text(json.dumps(dict(pth="x", made=[])))       # an older install's
    with pytest.raises(ValueError, match=refused):
        spawn.repo_python()
    prof = spawn.models.profile_path("m", "joyful")
    prof.parent.mkdir(parents=True)
    prof.write_text("{}")
    with pytest.raises(ValueError, match=refused):
        spawn.calibrate("joyful", "m")
    assert state.list_jobs() == []                                    # refused before any job was made
    installs = {}
    for name in ("a", "b"):
        (tmp_path / name / "metalscope").mkdir(parents=True)
        (tmp_path / name / "python").write_text("")
        installs[name] = (tmp_path / name / "python", tmp_path / name / "metalscope")
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "cli.json").write_text(json.dumps({"python": str(installs["a"][0])}))   # Pi's, not the server's
    for order in (("a", "b"), ("b", "a")):
        for name in order:
            with monkeypatch.context() as m:
                m.setattr(sys, "executable", str(installs[name][0]))
                m.setattr(install, "PACKAGE", installs[name][1])
                install.install(app, log=lambda line: None)
        last = installs[order[-1]]
        assert (install.hook_dir() / "metalscope").resolve() == last[1] and spawn.repo_python() == last[0]


def test_the_ds4_extra_serves_the_routes():
    """The ds4 proxy serves the same routes as oMLX's hook."""
    import tomllib
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]
    assert {"fastapi", "uvicorn", "httpx", "tokenizers"} <= set(project["optional-dependencies"]["ds4"])


def test_gpu_children_run_main_after_every_definition():
    """A job's child runs as `python -m`: its main block must come after every function it calls (jlens learn's
    child once called fit_sources, defined below the block, so every fit failed at once)."""
    import ast
    learn = Path(__file__).resolve().parents[1] / "src/metalscope/learn"
    for path in [learn / "gpu.py", *learn.glob("*_gpu.py")]:
        body = ast.parse(path.read_text()).body
        mains = [i for i, node in enumerate(body) if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)]
        assert mains and mains[-1] == len(body) - 1, f"{path.name}: the __main__ block is not last"

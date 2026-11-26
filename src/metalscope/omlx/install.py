"""``metalscope omlx install | uninstall``: the hook that makes oMLX's own Python import metalscope.

A .pth file in the user site of oMLX.app's bundled Python (``~/.local/lib/python3.11/site-packages``, which every
standalone Python 3.11 on the Mac shares; oMLX.app itself is never modified, so its signature stays intact and the hook
survives app updates). Its one executable line (``pth_line``) does nothing unless the interpreter runs from inside an
oMLX.app bundle; there it adds ``hook_dir()`` (a directory holding only a link named metalscope to this package, from
a git checkout or a pip install alike, so none of the environment metalscope is installed in comes with it) to sys.path
and imports omlx/boot.py. It never raises: with metalscope gone, Python starts as if the file were not there.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .. import config

DEFAULT_APP = Path("/Applications/oMLX.app")
OMLX_URL = "https://github.com/jundot/omlx"
TESTED_OMLX = {"0.8.0", "0.8.2", "0.8.3", "0.8.4"}   # the oMLX versions patches.py is verified against (others skipped)
PTH_NAME = "metalscope.pth"
BOOT_MODULE = "metalscope.omlx.boot"
PTH_IMPORT = f"import {BOOT_MODULE}"
PTH_HEADER = "# metalscope oMLX hook, layout 2: it loads only in oMLX.app's Python (metalscope omlx uninstall removes it)"
APP_MARK = "/oMLX.app/Contents/"
MANIFEST = "installed.json"     # in hook_dir(): the .pth's path, the directories install made, the Python it ran in
PACKAGE = Path(__file__).resolve().parents[1]       # the metalscope package the link points at


def hook_dir() -> Path:
    """What oMLX's Python adds to its path: a directory holding the metalscope link."""
    return Path("~/.local/share/metalscope/omlx").expanduser()


def python_home(app: Path) -> Path:
    """The app's bundled CPython (``cpython-3.11`` today; the name when there is no app to look in)."""
    return next(iter(sorted((app / "Contents/Resources/Python").glob("cpython-*"))),
                app / "Contents/Resources/Python/cpython-3.11")


def bundled_python(app: Path) -> Path:
    return python_home(app) / "bin/python3"


def bundled_env(app: Path) -> dict:
    res, home = app / "Contents/Resources", python_home(app)
    # HOME as ours: the user site (where the hook goes) is under it, as for the app, which runs with the user's HOME.
    # Without it Python takes the account's home from the password database, whatever HOME says.
    return dict(PYTHONHOME=str(home),
                PYTHONPATH=f"{res}:{res}/Python/framework-mlx-base/lib/python{home.name.split('-')[1]}/site-packages",
                PATH="/usr/bin:/bin", HOME=str(Path.home()))


def run_app() -> Path:
    """The oMLX.app whose Python runs GPU jobs (OMLX_APP, else /Applications/oMLX.app)."""
    return Path(os.environ.get("OMLX_APP") or DEFAULT_APP)


def run_dir() -> Path:
    """Where ``package_link()`` keeps its links (uninstall removes it)."""
    return Path("~/.local/share/metalscope/run").expanduser()


def package_link() -> Path:
    """A directory holding only a link named metalscope to this package (as ``hook_dir()``, but always this one):
    another environment's Python imports it from there and nothing else of ours. One per package (checkouts and
    installs alike), so a run from one never repoints the link a job of another is importing through."""
    import hashlib
    d = run_dir() / hashlib.sha1(str(PACKAGE).encode()).hexdigest()[:12]
    d.mkdir(parents=True, exist_ok=True)
    link = d / "metalscope"
    if not (link.is_symlink() and link.resolve() == PACKAGE):
        tmp = d / f".metalscope.{os.getpid()}"
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(PACKAGE, target_is_directory=True)
        tmp.replace(link)
    return d


def bundled_command(app: Optional[Path] = None, env: Optional[dict] = None) -> tuple[list[str], dict]:
    """The interpreter and environment that run this metalscope under oMLX's bundled Python, whose packages load a
    model as oMLX does (in a checkout, scripts/omlx-python does the same by hand). ``env`` the base (os.environ)."""
    app = app or run_app()
    base = bundled_env(app)
    out = dict(os.environ if env is None else env, PYTHONHOME=base["PYTHONHOME"], PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=f"{base['PYTHONPATH']}:{package_link()}")
    return [str(bundled_python(app))], out


def bundled_run(app: Path, code: str) -> subprocess.CompletedProcess:
    """``code`` run by oMLX's Python (OSError when the app is not there)."""
    return subprocess.run([str(bundled_python(app)), "-c", code], env=bundled_env(app), capture_output=True, text=True)


def user_site(app: Path) -> Path:
    out = bundled_run(app, "import site; print(site.getusersitepackages())")
    out.check_returncode()
    return Path(out.stdout.strip())


def omlx_version(app: Path) -> Optional[str]:
    try:
        r = bundled_run(app, "import omlx._version as v; print(v.__version__)")
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def require_app(app: Path) -> None:
    """One line instead of a traceback when there is no oMLX.app to hook into."""
    if not bundled_python(app).exists():
        raise SystemExit(f"no oMLX.app at {app}: install oMLX ({OMLX_URL}) into /Applications, or pass --app PATH")


def pth_line(paths, markers=(APP_MARK,), module: str = BOOT_MODULE) -> str:
    """The .pth file's executable line (Python runs a .pth line that starts with ``import``). Only when
    sys.executable or sys.prefix contains one of ``markers`` (inside an oMLX.app bundle) does it add ``paths`` to
    sys.path and import ``module``; when that import fails it takes the paths back out. It never raises, so a
    missing metalscope leaves every Python starting as before. No comprehensions in the code: under site's exec, a
    comprehension would not see the names the code binds."""
    cond = " or ".join(f"{m!r} in _w" for m in markers)
    code = "\n".join([
        "try:",
        "    _w = (sys.executable or '') + '\\n' + (sys.prefix or '')",
        f"    if {cond}:",
        f"        _p = {tuple(str(p) for p in paths)!r}",
        "        sys.path.extend(_p)",
        "        try:",
        f"            import {module}",
        "        except Exception:",
        "            for _q in _p:",
        "                sys.path.remove(_q)",
        "except Exception:",
        "    pass",
    ])
    return f"import sys; exec({code!r})"


def pth_text(app: Path) -> str:
    """The whole .pth: a comment saying what it is, then the guarded line. A bundle not named oMLX.app (``--app``)
    is recognised by its own path as well."""
    markers = (APP_MARK,) if app.name == "oMLX.app" else (APP_MARK, f"{os.path.abspath(app)}/Contents/")
    return f"{PTH_HEADER}\n{pth_line([hook_dir()], markers)}\n"


def _current(lines: list[str]) -> bool:
    """Whether the .pth is this metalscope's: its header and the guarded boot line."""
    return PTH_HEADER in lines and any(line.startswith("import sys; exec(") and BOOT_MODULE in line for line in lines)


def untested(version: Optional[str]) -> list[str]:
    """What install, ``omlx status`` and doctor say when oMLX ``version`` is not one patches.py is verified against:
    oMLX then runs untraced (boot.py skips every patch, the /scope routes among them), and installing again changes
    nothing. Empty for a tested version, or when the version is not known."""
    if not version or version in TESTED_OMLX:
        return []
    # No "try anyway" advice: oMLX.app starts its server with a clean environment (a launchctl setenv variable
    # reaches the app, not its server), so METALSCOPE_FORCE (boot.py) only reaches a server run by hand.
    return [f"oMLX {version} is not a version metalscope is tested with ({', '.join(sorted(TESTED_OMLX))})",
            "oMLX will run untraced (no traces or steering)",
            f"install a tested oMLX version, or update metalscope once it supports oMLX {version}"]


def hook_state(app: Path) -> dict:
    """What the installed hook does: ``pth`` its path, ``exists``, ``loads`` the package its link resolves to (None
    when missing), ``current`` whether it is this metalscope's guarded hook."""
    pth = user_site(app) / PTH_NAME
    if not pth.exists():
        return dict(pth=pth, exists=False, loads=None, current=False)
    target = hook_dir() / "metalscope"
    return dict(pth=pth, exists=True, loads=target.resolve() if target.exists() else None,
                current=_current(pth.read_text().splitlines()))


def _read_manifest() -> dict:
    try:
        return json.loads((hook_dir() / MANIFEST).read_text())
    except (OSError, ValueError):
        return {}


def install(app: Path, log=print) -> None:
    require_app(app)
    site = user_site(app)
    # The directories this install creates for the user site (an earlier install's too): uninstall removes them.
    made = sorted({*_read_manifest().get("made", []), *(str(d) for d in (site, *site.parents) if not d.exists())})
    site.mkdir(parents=True, exist_ok=True)
    hook = hook_dir()
    hook.mkdir(parents=True, exist_ok=True)
    link = hook / "metalscope"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(PACKAGE, target_is_directory=True)
    pth = site / PTH_NAME
    config.atomic_write(pth, pth_text(app))      # oMLX's Python reads it at start: never half written
    log(f"wrote {pth}: oMLX imports metalscope from {PACKAGE}")
    log("other Pythons skip it: it loads only inside oMLX.app")
    # python: what jobs the server starts run in (learn/spawn.py): this install's, whose package the link loads.
    config.atomic_write(hook / MANIFEST, json.dumps(dict(pth=str(pth), made=made, python=sys.executable), indent=1) + "\n")


def install_python() -> Optional[Path]:
    """The Python that ran `omlx install` for the hook oMLX loads (its manifest), while it is there; else None."""
    py = _read_manifest().get("python")
    return Path(py) if py and Path(py).exists() else None


def uninstall(app: Path, log=print) -> bool:
    """Remove the .pth, the link, the user-site
    directories install created (those left empty) and the links GPU jobs ran through (``run_dir()``); True when
    there was a hook to remove. With oMLX.app gone, the manifest says where the .pth is."""
    manifest = _read_manifest()
    if bundled_python(app).exists():
        site = user_site(app)
    elif manifest.get("pth"):
        site = Path(manifest["pth"]).parent
    else:
        require_app(app)
    removed = False
    pth = site / PTH_NAME
    if pth.exists():
        pth.unlink()
        log(f"removed {pth}")
        removed = True
    hook = hook_dir()
    link = hook / "metalscope"
    if link.is_symlink():
        link.unlink()
        removed = True
    (hook / MANIFEST).unlink(missing_ok=True)
    from .. import gpulock
    if gpulock.held() is None:                      # a GPU job imports metalscope through its link while it runs
        shutil.rmtree(run_dir(), ignore_errors=True)
    for d in (hook, hook.parent, *sorted(map(Path, manifest.get("made", [])), reverse=True)):     # deepest first
        try:
            d.rmdir()
        except OSError:
            pass                                    # not empty (something else lives there) or already gone
    return removed

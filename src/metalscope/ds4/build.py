"""``metalscope ds4 install``: antirez's ds4 at the commit the patch is made for, with metalscope's patch, built.

The patch (``metalscope.patch`` beside this file, ``git format-patch`` of the metalscope commits on that ds4 commit)
adds the capture and steering ds4 records for metalscope. It carries ds4's code as context, so ds4's MIT licence is
shipped beside it (``LICENSE.ds4``). The build lives outside the data home, under
``~/.local/share/metalscope/ds4`` (``METALSCOPE_DS4_DIR`` overrides it), and is redone only when the patch changed.
It needs git, make and Xcode's command-line tools (clang and the Metal framework); ds4 compiles its Metal kernels at
start, from ``metal/*.metal`` in the build directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .. import config

DS4_REPO = "https://github.com/antirez/ds4"
DS4_COMMIT = "5379c1c9ac315f1540371b06a3db88491d4059ff"   # what metalscope.patch applies to
PATCH = Path(__file__).resolve().parent / "metalscope.patch"
STAMP = ".metalscope-build.json"


def build_dir() -> Path:
    """Where ``ds4 install`` builds: METALSCOPE_DS4_DIR, else ~/.local/share/metalscope/ds4."""
    return Path(config.env("DS4_DIR") or "~/.local/share/metalscope/ds4").expanduser()


def patch_id() -> str:
    return hashlib.sha256(PATCH.read_bytes()).hexdigest()[:16]


def built(d: Path) -> bool:
    return (d / "ds4-server").exists() and (d / "metal").is_dir()


def current(d: Path) -> bool:
    """Built from this metalscope's patch (its stamp names the patch's hash)."""
    stamp = config.read_json(d / STAMP, {})
    return built(d) and stamp.get("patch") == patch_id() and stamp.get("commit") == DS4_COMMIT


def refuse_stale() -> None:
    """SystemExit when the build ``ds4 serve`` would run is one ``ds4 install`` made (its stamp) from another patch or
    commit than this metalscope's: its ds4-server may lack an option serve passes (--scope-dir) and would only exit
    while loading. A hand-made build (no stamp: METALSCOPE_DS4_DIR at one's own checkout) is the user's to keep
    current, as one --ds4-dir names."""
    d = build_dir()
    if built(d) and (d / STAMP).exists() and not current(d):
        raise SystemExit(f"metalscope's ds4 build is older than this metalscope: run metalscope ds4 install, then "
                         f"start again ({d} is not ds4 {DS4_COMMIT[:7]} + {PATCH.name} {patch_id()})")


def find_dir(explicit: Optional[Path] = None) -> Path:
    """The ds4 build ``ds4 serve`` runs: ``explicit`` (--ds4-dir), else the ``ds4 install`` build (a hand-made one:
    METALSCOPE_DS4_DIR). SystemExit naming ``metalscope ds4 install`` when there is none, or when install's build is
    older than this metalscope (refuse_stale)."""
    if explicit is not None:
        d = explicit.expanduser().resolve()
        if not built(d):
            raise SystemExit(f"{d} has no ds4-server: build it there (make), or run `metalscope ds4 install`")
        return d
    if built(build_dir()):
        refuse_stale()
        return build_dir()
    raise SystemExit("no ds4 build: `metalscope ds4 install` fetches ds4 and builds it with metalscope's patch")


def _run(cmd: list[str], cwd: Optional[Path] = None, log=print) -> None:
    log("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
        raise SystemExit(f"failed: {' '.join(cmd)}\n{tail}")


def _check_tools() -> None:
    missing = [t for t in ("git", "make") if shutil.which(t) is None]
    if missing:
        raise SystemExit(f"{', '.join(missing)} not found: install Xcode's command-line tools (xcode-select --install)")
    if subprocess.run(["xcrun", "--find", "clang"], capture_output=True).returncode != 0:
        raise SystemExit("no clang: install Xcode's command-line tools (xcode-select --install)")


def _own_checkout(d: Path, git: list[str]) -> None:
    """Refused (SystemExit saying what to do) unless ``d`` is a checkout ``ds4 install`` made (its stamp, written as it
    clones) with nothing uncommitted: installing resets it to DS4_COMMIT and removes its untracked files (ignored ones,
    the GGUFs, stay). An earlier install that stopped mid-patch is undone first (its changes are the patch's)."""
    if not (d / STAMP).exists():
        raise SystemExit(f"{d} is a ds4 checkout `metalscope ds4 install` did not make (no {STAMP}): installing "
                         f"would reset it to {DS4_COMMIT[:7]} and remove its untracked files. Build it there yourself "
                         f"(git am {PATCH}, then make ds4-server), pass another --dir, or --force to reset it")
    if (d / ".git" / "rebase-apply").exists():
        _run(git + ["-C", str(d), "am", "--abort"], log=lambda *_: None)
    # Tracked files only: what is untracked in a checkout install made is its build's (ds4-server, objects, the stamp).
    r = subprocess.run(["git", "-C", str(d), "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True)
    changed = r.stdout.splitlines()
    if r.returncode != 0 or changed:
        what = "\n".join("  " + ln for ln in changed[:10]) or (r.stderr.strip() or "git status failed")
        raise SystemExit(f"{d} has changes git has not committed, which installing would discard:\n{what}\n"
                         f"commit or move them, or pass --force to discard them")


def _has_commit(d: Path) -> bool:
    return subprocess.run(["git", "-C", str(d), "cat-file", "-e", f"{DS4_COMMIT}^{{commit}}"],
                          capture_output=True).returncode == 0


def install(d: Optional[Path] = None, source: str = DS4_REPO, force: bool = False, jobs: Optional[int] = None,
            log=print) -> Path:
    """Fetch ds4 (``source``: its repository, or a local clone) at DS4_COMMIT into ``d``, apply the patch and build
    ds4-server. Nothing is done when ``d`` already holds this patch's build (``force`` redoes it). An existing checkout
    is reset only when ``ds4 install`` made it and nothing in it is uncommitted (``force``: whatever it holds)."""
    d = (d or build_dir()).expanduser()
    if not force and current(d):
        log(f"ds4 is built with this metalscope's patch at {d}")
        return d
    from ..backends import ds4_servers_from
    running = ds4_servers_from(d)
    if running:                 # the rebuild resets the checkout and replaces its ds4-server and Metal kernels
        raise SystemExit(f"ds4-server runs from {d} (pid {running[0]}): stop it first (metalscope ds4 stop), then "
                         "install again")
    _check_tools()
    git = ["git", "-c", "user.name=metalscope", "-c", "user.email=metalscope@localhost", "-c", "advice.detachedHead=false"]
    if not (d / ".git").is_dir():
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"{d} exists and is not a ds4 checkout: remove it, or pass another directory")
        d.parent.mkdir(parents=True, exist_ok=True)
        log(f"fetching ds4 from {source}")
        _run(git + ["clone", "--quiet", source, str(d)], log=log)
        config.atomic_write(d / STAMP, json.dumps({}))          # made here: a later install may reset it (not built yet)
    else:
        if not force:
            _own_checkout(d, git)
        _run(git + ["-C", str(d), "fetch", "--quiet", source], log=log)
    if (d / ".git" / "rebase-apply").exists():          # an earlier install stopped mid-patch
        _run(git + ["-C", str(d), "am", "--abort"], log=lambda *_: None)
    if not _has_commit(d):
        # Not on a branch the clone or fetch brought (rewritten or moved): asked for by its id.
        subprocess.run(git + ["-C", str(d), "fetch", "--quiet", source, DS4_COMMIT], capture_output=True)
        if not _has_commit(d):
            raise SystemExit(f"ds4 commit {DS4_COMMIT[:7]}, which this metalscope's patch is made for, is not in "
                             f"{source}: pass --source with a ds4 clone that has it, or update metalscope")
    _run(git + ["-C", str(d), "checkout", "--quiet", "--force", DS4_COMMIT], log=log)
    _run(git + ["-C", str(d), "clean", "-fdq", "-e", "*.o", "-e", STAMP], log=log)
    log(f"applying {PATCH.name} ({patch_id()})")
    _run(git + ["-C", str(d), "am", "--quiet", str(PATCH)], log=log)
    log("building ds4-server (a few minutes)")
    _run(["make", f"-j{jobs or os.cpu_count() or 8}", "ds4-server"], cwd=d, log=log)
    if not built(d):
        raise SystemExit(f"make finished but {d / 'ds4-server'} is missing")
    config.atomic_write(d / STAMP, json.dumps(dict(commit=DS4_COMMIT, patch=patch_id())))
    log(f"ds4 built at {d}: `metalscope ds4 serve MODEL.gguf` runs it "
        f"(models: {d / 'download_model.sh'}, e.g. ./download_model.sh qwen38-q4k)")
    return d

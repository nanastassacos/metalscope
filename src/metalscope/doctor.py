"""``metalscope doctor``: is every piece wired up (oMLX's hook and key, each server, the ds4 build, Pi, traces);
``--live`` adds livecheck.py's end-to-end run on each backend serving a model."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import client, config
from .omlx import admin as omlx_admin, install as omlx_install

SECTIONS = ("oMLX", "ds4", "Pi", "Models", "Traces")
ENGINES = dict(omlx="oMLX", ds4="ds4")      # a server's engine: its section
LABEL = 10                                  # the label column


def tilde(text: str) -> str:
    """``text`` with the home folder as ~ (a screenshot shows no account name)."""
    return text.replace(str(Path.home()) + "/", "~/")


def pi_listed() -> bool | None:
    """Whether `pi list` lists metalscope's Pi package; None when pi is not on PATH."""
    try:
        return str(config.PI_PACKAGE) in subprocess.run(["pi", "list"], capture_output=True, text=True).stdout
    except OSError:
        return None


def cmd_doctor(args) -> None:
    """One screen answering "is everything wired up?": a section per piece, a line per check marked passed, failed,
    warning or information (config.MARKS), then their count. Exits 1 when a check fails (--live's too); warnings and
    information leave it at 0, so a script or CI can run it."""
    from . import backends, jlens, retention as R
    from .ds4 import build as B
    rows: list = []                 # (section, mark, label, text); no mark: a line as it is (the models' table)
    add = lambda section, kind, label, text: rows.append((section, kind, label, text))
    ok = lambda good: "ok" if good else "fail"
    app = Path(args.app)
    version = omlx_install.omlx_version(app)
    hooked = False              # oMLX runs this metalscope's current hook
    if version is None and (B.build_dir() / "ds4-server").exists():
        add("oMLX", "info", "app", f"none at {app}: fine with ds4 alone (metalscope omlx install, once oMLX is installed)")
    else:
        add("oMLX", ok(version is not None), "app",
            f"{app} · {version or 'not found'} (tested on {', '.join(sorted(omlx_install.TESTED_OMLX))})")
        for i, line in enumerate(omlx_install.untested(version)):
            add("oMLX", "warn", "" if i else "version", line)
        try:
            hook = omlx_install.hook_state(app)
            hooked = hook["exists"] and hook["current"]
            add("oMLX", ok(hooked), "hook", f"{hook['pth']}" + ("" if hooked else " · missing: metalscope omlx install"
                if not hook["exists"] else f" · does not run `{omlx_install.PTH_IMPORT}`: oMLX starts without "
                "metalscope until metalscope omlx install rewrites it"))
            if hooked:
                add("oMLX", "info", "imports", str(hook["loads"] or "nothing: the link is missing"))
                if hook["loads"] != omlx_install.PACKAGE:
                    add("oMLX", "warn", "", f"not this metalscope ({omlx_install.PACKAGE}): metalscope omlx install "
                        "from here to switch")
            if version:
                has_key = bool((omlx_admin.read_settings().get("auth") or {}).get("api_key"))
                add("oMLX", ok(has_key), "API key", "set" if has_key else f"not set: open {omlx_admin.server_url()}"
                    "/admin and set one (metalscope loads and unloads models with it)")
        except (OSError, subprocess.CalledProcessError):
            add("oMLX", "fail", "hook", "could not run oMLX's bundled Python")
    pin = f"ds4 {B.DS4_COMMIT[:7]} + {B.PATCH.name} {B.patch_id()}"
    d = B.build_dir()
    if not B.built(d):
        add("ds4", "info", "build", f"not built: metalscope ds4 install builds {pin}")
    else:
        add("ds4", ok(B.current(d)), "build",
            f"{d} · " + (pin if B.current(d) else f"not {pin}: metalscope ds4 install --force rebuilds it"))
    for url in client.candidates():
        c = client.Client(url)
        s = c.status(timeout=2)
        sec = ENGINES.get((s or {}).get("engine"), "ds4" if url == client.DS4_URL else "oMLX")
        if s is None and c.status_error is not None and c.status_error.status == 401:
            add(sec, "fail", "server", omlx_admin.key_refused(url))
            continue
        if s:
            add(sec, "ok", "server", f"{url} · tracing {s['mode']} · "
                f"{', '.join(config.short_name(m['name']) for m in s['models']) or 'no model loaded'}")
        elif hooked and version in omlx_install.TESTED_OMLX and url == omlx_admin.server_url() \
                and c.status_error is not None and c.status_error.status == 404:
            # oMLX answers but has no /scope routes: not restarted since install, or metalscope failed to load in it
            add(sec, "fail", "server", f"{url} · runs without metalscope: {omlx_admin.NOT_LOADED}")
        else:
            add(sec, "info", "server", f"{url} · no metalscope server")
        if s and (s.get("patches") or {}).get("missing"):
            add(sec, "fail", "", f"patches not applied: {', '.join(s['patches']['missing'])} (its log says why)")
        if s and s.get("stale"):
            add(sec, "warn", "", f"runs older metalscope code than is installed: {s['stale']}")
    on = pi_listed()
    if on is None:
        add("Pi", "info", "app", "not installed")
    else:               # Pi is optional: a missing package is a warning, not a failure.
        add("Pi", "ok" if on else "warn", "package", str(config.PI_PACKAGE) if on else
            "not installed in Pi: metalscope pi install")
    table = backends.model_lines()
    for i, line in enumerate(table):
        add("Models", None, None, config.paint(line, "2") if i == 0 and len(table) > 1 else line)
    for i, lens in enumerate([f"{d.parent.parent.name}/{d.name}" for d in jlens.lens_dirs()] or ["none"]):
        add("Models", None, None, f"{'' if i else 'J-lenses':<{LABEL}}{lens}")
    st = R.status(week=True)
    add("Traces", "info", None, f"{st['count']:,} traces · {R.fmt_size(st['bytes'])} of {R.fmt_size(st['budget'])} · "
        f"{R.fmt_size(st['old_bytes'])} older than a week · {config.traces_dir()}")
    add("Traces", "info", None, f"cleanup past the budget: {'on' if st['auto'] else 'off'} (metalscope traces auto on|off)")
    if not st["auto"] and st["bytes"] > st["budget"]:
        add("Traces", "warn", None, f"{R.fmt_size(st['bytes'] - st['budget'])} over the budget and cleanup is off: "
            "metalscope traces prune frees it")
    add("Traces", "info", None, "free space: metalscope traces prune --older-than 7d, or --all (--dry-run first)")

    for n, section in enumerate(SECTIONS):
        print(("\n" if n else "") + config.paint(section, "1"))
        for _, kind, label, text in (r for r in rows if r[0] == section):
            text = tilde(text if label is None else f"{label:<{LABEL}}{text}")
            print("  " + (config.mark(kind, text) if kind else text))
    failed, warned = sum(r[1] == "fail" for r in rows), sum(r[1] == "warn" for r in rows)
    if args.live:
        from . import livecheck
        results: list = []
        if not livecheck.run(log=lambda s: print(tilde(s)), results=results):
            failed += max(1, sum(livecheck.verdict(found.get(f, [])) == "failed"
                                 for _, found in results for f in livecheck.FEATURES))
    said = " · ".join(x for x in (failed and f"{failed} failed", warned and f"{warned} warning" + "s" * (warned > 1)) if x)
    print("\n" + config.paint(said or "all good", "1;31" if failed else "1;33" if warned else "1;32"))
    if failed:
        raise SystemExit(1)

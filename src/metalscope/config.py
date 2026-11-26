"""Settings and paths. Everything lives under ~/.metalscope (override with
METALSCOPE_HOME).

    settings.json   {"mode": "off" | "on", "steer": true | false, "trace_budget": bytes, "trace_auto": true | false
                     (retention.py), "lens_step": "small" | "medium" | "large" (how many layers the lenses read,
                     lens_reads, tracer.lens_steps) | "off"}
                    (settings.lock orders its writers)
    traces/         one directory per traced response
    index.jsonl     one line per saved trace (index.lock orders its writers)
    pinned.json     trace ids never pruned (metalscope traces pin)

Modes:
    off    hooks are not installed; zero cost. A request can still opt in
           with the X-Scope-Trace header, which installs hooks on demand.
    on     every request traced: residual norms, MoE routing, final-token
           predictions, and both lenses (the logit
           lens, the J-lens beside it while one is on) at the lens size's
           layers (tracer.lens_steps). Replies decode slower than
           untraced and a long prompt's first token comes later;
           `metalscope bench` measures the decode speed on the served model.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
import tempfile
from pathlib import Path

PI_PACKAGE = Path(__file__).resolve().parent / "frontends" / "pi"   # the Pi front end, shipped inside the package
MODES = ("off", "on")
HEADER = "x-scope-trace"
# Lets a client (e.g. the Pi extension) pick out its own requests on the live stream.
CLIENT_HEADER = "x-scope-trace-client"
# Steering (steer.py): refused unless settings["steer"] is true (`metalscope steering on`).
STEER_HEADER = "x-scope-steer"
STEER_RESPONSE_HEADER = "x-scope-steer-accepted"
# How a steered request refused by the server's steering setup begins (a 503; asking again cannot help it).
NO_STEERING = "metalscope could not set up steering"
# An earlier reply answered again from one token, with a steer at that token only (steer.parse_edit; the ds4 proxy).
EDIT_HEADER = "x-scope-edit"
# How many layers a trace reads the lenses at: one of LENS_SIZES, fewest first (tracer.lens_steps), by its index (a
# step: the dock's - +), each lens lens_reads layers spread over the depth; the default medium. LENS_OFF: none (a
# trace without the lenses: probabilities, routing and steering only). Settings and traces keep the step's name
# (lens_name).
LENS_SIZES = ("small", "medium", "large")
LENS_STEPS, LENS_STEP_DEFAULT, LENS_OFF = len(LENS_SIZES), 1, -1
# The layers each size reads (lens_reads): small a count; medium a fraction of the depth (to the nearest) and large one
# (rounded down), each at most its cap.
LENS_SMALL = 4
LENS_MEDIUM, LENS_MEDIUM_MAX = 1 / 3, 16
LENS_LARGE, LENS_LARGE_MAX = 1 / 2, 24
# The indexes of the five steps settings kept before 2026-11-25, each as the size nearest it in layers (save_settings).
_FIVE_STEPS = {0: "small", 1: "small", 2: "medium", 3: "medium", 4: "large", -1: "off"}


def lens_reads(n_layers: int) -> list[int]:
    """How many layers each size reads on a model of ``n_layers`` (LENS_SMALL, LENS_MEDIUM, LENS_LARGE), fewest first:
    none fewer than the size before, none more than the model has (two sizes can then read the same)."""
    want = [LENS_SMALL, min(LENS_MEDIUM_MAX, int(n_layers * LENS_MEDIUM + 0.5)), min(LENS_LARGE_MAX, int(n_layers * LENS_LARGE))]
    return [min(n_layers, max(want[:i + 1])) for i in range(len(want))]


def lens_step_ok(k) -> bool:
    """Whether ``k`` is a lens step (an int, 0 to LENS_STEPS - 1) or LENS_OFF."""
    return type(k) is int and LENS_OFF <= k < LENS_STEPS


def lens_name(k) -> str:
    """Lens step ``k``'s name: its size, or "off". Past the sizes (a server from before them, at one of its five steps)
    the size nearest that step; anything else "step k"."""
    if k == LENS_OFF:
        return "off"
    if type(k) is int and 0 <= k < LENS_STEPS:
        return LENS_SIZES[k]
    return (type(k) is int and _FIVE_STEPS.get(k)) or f"step {k}"


def lens_index(name) -> int | None:
    """The lens step named ``name`` (lens_name); None for anything else."""
    return LENS_OFF if name == "off" else LENS_SIZES.index(name) if name in LENS_SIZES else None


def stored_lens(step) -> str:
    """Settings' ``lens_step`` as a size's name (or "off"): an index of the five steps before as its nearest size;
    anything else no size is (edited by hand, a later version's) the default."""
    if lens_index(step) is not None:
        return step
    return (type(step) is int and _FIVE_STEPS.get(step)) or lens_name(LENS_STEP_DEFAULT)


def env(name: str, default: str | None = None) -> str | None:
    """``METALSCOPE_<name>``, else ``default``."""
    return os.environ.get("METALSCOPE_" + name, default)


def home() -> Path:
    """The data home: ``METALSCOPE_HOME`` when set (empty counts as unset, as in Pi), else ~/.metalscope. Resolved on
    every call (tests and background threads see the environment as it is now)."""
    override = env("HOME")
    return Path(override) if override else Path.home() / ".metalscope"


def private_home() -> Path:
    """The data home, created private (0700) and narrowed to it when broader: traces, the index and job logs hold
    prompts and replies. Called as the CLI and a server start; only the home itself changes mode, and only a folder
    the user owns that is not their home folder or one above it (METALSCOPE_HOME=~, /tmp). Never raises: a home it
    cannot create or narrow (a volume without modes) is one warning line, and metalscope runs on."""
    d = home()
    try:
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        st, user = d.stat(), Path.home().resolve()
        if st.st_mode & 0o077 and st.st_uid == os.getuid() and d.resolve() not in (user, *user.parents):
            d.chmod(0o700)
    except OSError as exc:
        print(f"metalscope: could not make {d} private ({exc.strerror or exc}); other accounts may read its traces",
              file=sys.stderr)
    return d


def traces_dir() -> Path:
    return home() / "traces"


def index_path() -> Path:
    return home() / "index.jsonl"


def settings_path() -> Path:
    return home() / "settings.json"


def load_settings() -> dict:
    data = read_json(settings_path(), {})
    if not isinstance(data, dict):       # a list or a number in the file: every request reads the mode
        data = {}
    if data.get("mode") not in MODES:
        data["mode"] = "off"
    data["steer"] = bool(data.get("steer", False))
    return data


def read_json(path, default=None):
    """``path``'s JSON, or ``default`` when the file is missing or not JSON."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic_write(path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file and rename, so a reader (the
    server polling profiles, vectors or pages) never sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique temp name: two writers of the same file (two page loads rebuilding it) must not share one.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@contextlib.contextmanager
def file_lock(path):
    """An flock on ``path`` (made when missing), held for the ``with``: around a read-modify-write of a file that
    oMLX's server, the ds4 proxy and the command line may all change at once. It orders threads too (each ``with``
    opens the file anew, and an flock belongs to the open file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


LOG_LIMIT = 50 << 20      # a server log past this many bytes is moved aside at its next start (rotated_log)


def rotated_log(path: Path) -> Path:
    """``path``, moved to ``<name>.1`` first (an older one replaced) when it is over LOG_LIMIT bytes: a server's log is
    appended at every start and was never trimmed. Called before the log is opened."""
    try:
        if path.stat().st_size > LOG_LIMIT:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass
    return path


MARKS = dict(ok=("✓", "32"), fail=("✗", "31"), warn=("!", "33"), info=("·", "2"))   # a check's mark, its colour


def paint(text: str, style: str) -> str:
    """``text`` in an ANSI ``style`` (1 bold, 2 dim, 31 red, ...) when stdout is a terminal and NO_COLOR is unset."""
    return f"\033[{style}m{text}\033[0m" if text and sys.stdout.isatty() and not os.environ.get("NO_COLOR") else text


def mark(kind: str, text: str) -> str:
    """A check's line: its mark (MARKS: ok, fail, warn, info) then ``text``; an info line dim throughout."""
    m, style = MARKS[kind]
    return paint(f"{m} {text}", style) if kind == "info" else f"{paint(m, style)} {text}"


def short_name(model) -> str:
    """The served model's short name: the last path component."""
    return str(model or "").rstrip("/").split("/")[-1]


def int_keys(d) -> dict:
    """A layer-keyed dict as read back from JSON (string keys) with its int
    keys restored; None reads as empty."""
    return {int(k): v for k, v in (d or {}).items()}


def save_settings(**changes) -> dict:
    with file_lock(home() / "settings.lock"):
        data = load_settings()
        data.update(changes)
        if data["mode"] not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not isinstance(data.get("steer", False), bool):
            raise ValueError("steer must be true or false")
        n = data.get("trace_budget")
        if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n <= 0):
            raise ValueError("trace_budget must be a number of bytes above 0")
        if not isinstance(data.get("trace_auto", True), bool):
            raise ValueError("trace_auto must be true or false")
        step = data.get("lens_step", lens_name(LENS_STEP_DEFAULT))
        if lens_index(step) is None:
            if "lens_step" in changes:
                raise ValueError(f"lens_step must be one of {', '.join(LENS_SIZES)} or off")
            data["lens_step"] = stored_lens(step)
        atomic_write(settings_path(), json.dumps(data, indent=1))
        return data

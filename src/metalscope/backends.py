"""The two backends as one thing to start, stop and list: oMLX (its app's server, which loads a model when a request
names it) and ds4 (ds4-server behind metalscope's proxy, `ds4 serve`, run in the background). One model fits at a
time on a 128 GB Mac and neither server knows of the other's memory (oMLX refuses a load, HTTP 507, while ds4 holds
its model), so starting one backend's model frees the other's first.

ds4's models are the GGUF files it has served (`ds4/models.json`: each name's file and ds4-server options, written
by `ds4 serve`) and the servable ones in the GGUF folders (`METALSCOPE_DS4_GGUF`, `ds4 install`'s `gguf/`). A model
runs at its own maximum context unless told otherwise (`--ctx`): what the GGUF says (Qwen3.8 Flash Next 262,144,
DeepSeek V4.1 1,048,576; ds4 plans 89.3 and 93.9 GiB for them). oMLX's models are what its /v1/models lists, kept
(`omlx/models.json`) for when oMLX is not running. `status()` is both, for `metalscope status` and Pi's /model."""

from __future__ import annotations

import contextlib
import fcntl
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from . import client, config
from .models import tested
from .omlx import admin as omlx_admin

Log = Callable[[str], None]
CHAT_MODEL_SKIP = ("markitdown", "embed", "rerank", "whisper", "tts")   # oMLX entries that are not chat models (as Pi's)
DS4_CHAT = dict(reasoning=True, thinking_format="qwen-chat-template")  # ds4-server's thinking switch, every model
DS4_LOAD_TIMEOUT = 1800.0                                             # V4.1's first load reads 340 GiB
DEFAULT_CTX = 32768                                                   # ds4-server's context when a GGUF names none
DS4_SERVER_PORT = 8010                                                # ds4-server behind the proxy (ds4 serve --ds4-port)
DS4_EXTRA = ("fastapi", "uvicorn", "httpx", "tokenizers")             # what `ds4 serve` imports from the [ds4] extra
DS4_LOADING = "ds4 loading"                                           # the GPU lock's holder while ds4 loads (free_for_ds4)
DS4_EXTRA_INSTALL = f'uv tool install --reinstall "metalscope[ds4] @ git+{config.REPO_URL}"'


def _ds4_home() -> Path:
    return config.home() / "ds4"


def _write_json(path: Path, value) -> None:
    # A unique temp file per write: `ds4 start` and the serve it spawns both write serving.json.
    config.atomic_write(path, json.dumps(value, indent=1))


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _pgrep(*args: str) -> list[int]:
    try:
        out = subprocess.run(["pgrep", *args], capture_output=True, text=True).stdout
    except OSError:
        return []
    return [int(p) for p in out.split() if p.isdigit() and int(p) != os.getpid()]


def _command(pid) -> str:
    try:
        return subprocess.run(["ps", "-o", "command=", "-p", str(int(pid))], capture_output=True, text=True).stdout.strip()
    except (OSError, ValueError, TypeError):
        return ""


def is_serve(command: str) -> bool:
    """A `ds4 serve` process's command line: Python (or the metalscope script) running metalscope's ds4 serve; not a
    shell, editor or monitor whose command line merely holds those words."""
    words = command.split()
    exe = Path(words[0]).name.lower() if words else ""          # a framework Python shows as .../MacOS/Python
    return bool(words) and (exe.startswith("python") or exe == "metalscope") \
        and "ds4 serve" in command and "metalscope" in command


def _is_ds4_server(pid) -> bool:
    words = _command(pid).split()
    return bool(words) and Path(words[0]).name == "ds4-server"


def _ds4_servers() -> list[int]:
    """metalscope's ds4-servers: the one our `ds4 serve` started (serving.json names it), while that pid still runs a
    ds4-server, and any run from metalscope's own build (`ds4 serve` starts it by that absolute path: one started by
    hand from it, or left running when serve was killed, has no pid recorded). Never another one on the machine
    (another build's)."""
    from .ds4.build import build_dir
    pid = config.read_json(_ds4_home() / "serving.json", {}).get("server")
    ours = [pid] if _alive(pid) and _is_ds4_server(pid) else []
    return ours + [p for p in ds4_servers_from(build_dir()) if p not in ours]


def ds4_servers_from(d: Path) -> list[int]:
    """The ds4-servers running the build in ``d`` (started by its absolute path, as `ds4 serve` starts it)."""
    exe = str(d.expanduser().resolve() / "ds4-server")
    return [p for p in _pgrep("-x", "ds4-server") if _command(p).split()[:1] == [exe]]


def other_ds4_servers() -> list[int]:
    """ds4-servers metalscope did not start (or no longer knows of): `ds4 stop` leaves them alone and says so."""
    ours = set(_ds4_servers())
    return [p for p in _pgrep("-x", "ds4-server") if p not in ours]


def require_ds4_extra() -> None:
    """SystemExit with the install line unless the [ds4] extra's packages are here (`ds4 serve` imports them): checked
    before anything is stopped or unloaded, so a plain install never frees oMLX's model only to fail at the import."""
    import importlib.util
    missing = [m for m in DS4_EXTRA if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"ds4 needs metalscope's [ds4] extra (missing {', '.join(missing)}): {DS4_EXTRA_INSTALL}")


def port_taken(port: int) -> bool:
    """Something listens on 127.0.0.1:``port`` (a bind there fails, as the server's own would)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)    # as uvicorn: a closed connection's TIME_WAIT is no holder
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def port_holder(port: int) -> str:
    """What listens on ``port``, as "NAME (pid N)" (lsof), or "" when that cannot be found."""
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    fields = {line[0]: line[1:] for line in out.splitlines() if line[:1] in ("p", "c")}
    return f"{fields['c']} (pid {fields['p']})" if "c" in fields and "p" in fields else ""


def require_free_ports(ports: dict) -> None:
    """SystemExit naming the first of ``ports`` ({port: what needs it}) that is taken, and what holds it: checked
    before a model loads (80-94 GB, minutes), not after."""
    for port, who in ports.items():
        if port_taken(port):
            holder = port_holder(port)
            raise SystemExit(f"port {port} is taken" + (f" by {holder}" if holder else "") + f"; {who} needs it")


@contextlib.contextmanager
def switching(log: Log = print, wait: bool = True):
    """One backend switch at a time, across processes (two Pi sessions, a terminal): deciding what runs, stopping the
    other, and starting the new one, so two models never load at once. ds4's load is held by the GPU lock instead
    (`free_for_ds4`): its `ds4 serve` is in serving.json (and found by is_serve) from the moment it starts. Each step
    that stops or starts something checks `gpu_free` first. A job's start holds it too (``wait`` False: ValueError
    while a switch is under way), so no job starts between a switch's check and what it stops."""
    path = config.home() / "backends.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                raise ValueError("a backend switch is under way: try again in a moment") from None
            log("waiting for another backend switch to finish")
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def gpu_free(act: str = "switch backends") -> None:
    """Whether the served model may be taken away now: refused (SystemExit with its message: "cannot ``act`` now")
    while any job runs (a calibration talks to the served model and holds no lock; a learn job between its GPU
    steps does the same) or a GPU job holds the GPU lock (gpulock.py): it measures the served model, or its own copy
    in the memory a switch would use. Checked inside ``switching``, right before a step that stops or starts
    something (a switch that changes nothing goes ahead). ds4's own load (`free_for_ds4`) is no refusal: a switch
    stops it. A job's own window unloads and reloads through the admin API, never through here."""
    from . import gpulock
    from .learn import state
    job = state.running()
    if job:
        raise SystemExit(f"cannot {act} now: {state.running_text(job)}; wait for it or stop it: "
                         f"metalscope jobs cancel {job['job']}")
    holder = gpulock.held()
    if holder is not None and not (holder.what.startswith(DS4_LOADING) and holder.pid in _serve_pids()):
        raise SystemExit(f"cannot {act} now: {holder.doing()}; try again when it is done")


def free_for_ds4(name: str, log: Log = print):
    """Room for ds4's model, the one path for `ds4 serve` (by hand or from `ds4 start`): refused while another ds4
    runs, then the GPU lock taken and oMLX's models unloaded under it. Returned held: serve releases it once
    ds4-server answers or has stopped, so meanwhile oMLX refuses a load (omlx/patches.py patch_engine_pool) that its
    own memory check might pass while ds4's weights page in."""
    from . import gpulock
    others = [p for p in _serve_pids() + _ds4_servers() if p != os.getpid()]
    if others:
        raise SystemExit(f"ds4 already runs (pid {others[0]}): metalscope ds4 stop")
    gpu_free(f"serve {name}")                     # it unloads oMLX's model, which a job may be talking to
    try:
        lock = gpulock.GpuLock(gpulock.Holder(f"{DS4_LOADING} {name}")).acquire()
    except gpulock.LockBusy as exc:
        raise SystemExit(str(exc))
    try:
        free_omlx(log)
    except BaseException:
        lock.release()
        raise
    return lock


def memory_free() -> Optional[int]:
    """The system's free-memory percentage as macOS's memory_pressure reports it."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=10).stdout
        return int(out.rsplit("percentage:", 1)[1].strip().rstrip("%"))
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------- ds4's models

def gguf_info(path: Path) -> Optional[dict]:
    """A GGUF's architecture, its family and its maximum context (its header only: a few ms however large the file);
    None when it is not a file metalscope's ds4 capture covers (ds4/vocab.py FAMILIES)."""
    from .ds4.vocab import FAMILIES, read_gguf_meta
    try:
        meta = read_gguf_meta(path)
    except (OSError, ValueError, KeyError):
        return None
    arch = meta.get("general.architecture")
    if arch not in FAMILIES:
        return None
    ctx = meta.get(f"{arch}.context_length") or meta.get(f"{arch}.max_position_embeddings")
    return dict(arch=arch, family=FAMILIES[arch], ctx=int(ctx) if ctx else None)


def ds4_chat(family: str) -> dict:
    """A ds4 model's entry in /scope/api/status ``chat`` (the proxy's, and status() here for a model not served):
    ds4-server's thinking switch, whether metalscope was run on its family on ds4, and the edit (the proxy takes one
    for every model: its ``edit_at``)."""
    return dict(DS4_CHAT, tested=tested("ds4", family), edit=True)


def _ram() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return 128 << 30


def default_args(arch: str, gguf: Path) -> list[str]:
    """ds4-server options a model needs when it was never served with its own: Qwen3.8's MTP head (faster decoding);
    V4.1 streams its experts from the SSD when the file is larger than the machine can hold (341 GiB on 128 GB)."""
    if arch == "qwen4exp":
        return ["--mtp"]
    if arch == "deepseek41" and gguf.stat().st_size > 0.6 * _ram():
        return ["--ssd-streaming"]
    return []


def strip_ctx(args: list[str]) -> tuple[list[str], Optional[int]]:
    """ds4-server options without the context size, and that size (the model's own is the default, not a remembered one)."""
    out, ctx, i = [], None, 0
    while i < len(args):
        a = args[i]
        if a in ("--ctx", "-c") and i + 1 < len(args):
            ctx, i = int(args[i + 1]), i + 2
            continue
        if a.startswith("--ctx="):
            ctx = int(a.split("=", 1)[1])
        else:
            out.append(a)
        i += 1
    return out, ctx


def register(name: str, gguf: Path, args: list[str]) -> None:
    """Remember a ds4 model: its name, file and ds4-server options, and that it was served last (`ds4 serve` calls this
    once ds4-server answers, so options that fail to load are not kept, nor a model that never loaded; Pi lists it
    from now on)."""
    path = _ds4_home() / "models.json"
    reg = config.read_json(path, {})
    entry = dict(gguf=str(Path(gguf).expanduser().resolve()), args=strip_ctx(args)[0])
    if reg.get(name) != entry:
        reg[name] = entry
        _write_json(path, reg)
    _write_json(_ds4_home() / "last.json", dict(name=name))


def gguf_dirs() -> list[Path]:
    from .ds4.build import build_dir
    env = os.environ.get("METALSCOPE_DS4_GGUF", "")
    return [Path(p).expanduser() for p in env.split(os.pathsep) if p] + [build_dir() / "gguf"]


def ds4_models() -> list[dict]:
    """Every ds4 model: the remembered ones whose file is there, then the servable GGUFs in the GGUF folders not
    remembered yet (named as `ds4 serve` names them, the file name and -ds4). Each: name, gguf, args, ctx, arch."""
    out, seen = [], set()
    for name, e in config.read_json(_ds4_home() / "models.json", {}).items():
        gguf = Path(e.get("gguf", ""))
        info = gguf_info(gguf) if gguf.is_file() else None
        if info:
            out.append(dict(name=name, gguf=str(gguf), args=list(e.get("args") or []), **info))
            seen.add(str(gguf))
    for d in gguf_dirs():
        for gguf in sorted(d.glob("*.gguf")) if d.is_dir() else []:
            gguf = gguf.resolve()
            if str(gguf) in seen:
                continue
            info = gguf_info(gguf)
            if info:
                out.append(dict(name=f"{gguf.stem}-ds4", gguf=str(gguf), args=default_args(info["arch"], gguf), **info))
                seen.add(str(gguf))
    return out


def ds4_model(spec: str, name: Optional[str] = None) -> dict:
    """A ds4 model by its name, or by a GGUF file's path (named `name`, else as `ds4 serve` names it)."""
    models = ds4_models()
    hit = next((m for m in models if m["name"] == spec), None)
    if hit:
        return hit
    path = Path(spec).expanduser()
    if path.suffix == ".gguf" or path.is_file():
        if not path.is_file():
            raise SystemExit(f"{path}: no such file")
        path = path.resolve()
        info = gguf_info(path)
        if not info:
            raise SystemExit(f"{path.name}: not a model metalscope's ds4 capture covers (Qwen3.8 Flash Next, DeepSeek V4.1)")
        known = next((m for m in models if m["gguf"] == str(path) and (name is None or m["name"] == name)), None)
        return known or dict(name=name or f"{path.stem}-ds4", gguf=str(path), args=default_args(info["arch"], path), **info)
    names = ", ".join(m["name"] for m in models) or "none yet (a GGUF file's path serves one)"
    raise SystemExit(f"no ds4 model {spec!r}; ds4's models: {names}")


# ---------------------------------------------------------------- ds4's server

def write_serving(name: str, ctx: Optional[int], pid: Optional[int] = None, server: Optional[int] = None) -> None:
    """`ds4 serve` running: which process serves what (`ds4 stop`, `status` and Pi read it; gone when serve exits).
    Written by `ds4 start` as it spawns serve (pid: serve's) and again by serve itself, then with the ds4-server it
    started (server: its pid, the one `ds4 stop` stops)."""
    rec = dict(pid=pid or os.getpid(), name=name, ctx=ctx)
    if server is not None:
        rec["server"] = server
    _write_json(_ds4_home() / "serving.json", rec)


def clear_serving() -> None:
    path = _ds4_home() / "serving.json"
    if config.read_json(path, {}).get("pid") == os.getpid():
        path.unlink(missing_ok=True)


def _serve_pids() -> list[int]:
    """`ds4 serve` processes: the recorded one (while that pid still runs serve: a stale record's pid may be another
    process's by now), and any other running one (started by hand while another serve held the record)."""
    st = config.read_json(_ds4_home() / "serving.json", {})
    pids = [st["pid"]] if _alive(st.get("pid")) and is_serve(_command(st["pid"])) else []
    return pids + [p for p in _pgrep("-f", "ds4 serve") if p not in pids and is_serve(_command(p))]


def _proxy_models(url: str = client.DS4_URL) -> Optional[list[dict]]:
    try:   # the standard library: a plain install (no ds4 extra) has no httpx
        with urllib.request.urlopen(f"{url}/v1/models", timeout=2.0) as r:
            return json.loads(r.read()).get("data", [])
    except (OSError, ValueError, http.client.HTTPException):   # URLError, HTTPError (not 200), timeouts; not HTTP
        return None


def ds4_serving() -> Optional[dict]:
    """What ds4 serves now: name, ctx, ready (the proxy answers; it opens once the model is loaded); None when no
    `ds4 serve` runs."""
    pids = _serve_pids()
    if not pids:
        return None
    st = config.read_json(_ds4_home() / "serving.json", {})
    st = st if st.get("pid") in pids else {}
    data = _proxy_models()
    if data:
        m = data[0]
        return dict(pid=pids[0], name=m.get("id"), ctx=m.get("max_model_len"), ready=True)
    return dict(pid=pids[0], name=st.get("name"), ctx=st.get("ctx"), ready=False)


def ds4_stop(log: Log = print, timeout: float = 90.0) -> bool:
    """Stop `ds4 serve` (it stops its ds4-server) and wait until ds4-server has exited and its memory is free;
    False when nothing was running."""
    with switching(log):
        return _ds4_stop(log, timeout)


def _signal(pids, sig) -> None:
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, sig)


def _ds4_stop(log: Log = print, timeout: float = 90.0) -> bool:
    pids = _serve_pids()
    servers = _ds4_servers()                          # read now: serve removes its record as it exits

    def running() -> list[int]:
        return [p for p in servers if _alive(p) and _is_ds4_server(p)]
    if not pids and not servers:
        (_ds4_home() / "serving.json").unlink(missing_ok=True)      # a stale record
        return False
    gpu_free()
    _signal(pids, signal.SIGINT)
    end = time.monotonic() + timeout
    while time.monotonic() < end and (any(_alive(p) for p in pids) or running()):
        if not any(_alive(p) for p in pids):          # serve gone (killed), its ds4-server left behind
            _signal(running(), signal.SIGTERM)
        time.sleep(0.5)
    left = running()
    _signal(left, signal.SIGKILL)
    end = time.monotonic() + 15                       # its memory is free once it has exited
    while left and running() and time.monotonic() < end:
        time.sleep(0.2)
    (_ds4_home() / "serving.json").unlink(missing_ok=True)
    log("ds4 stopped" + (" (ds4-server killed)" if left else ""))
    return True


# ds4-server's own messages, after its log's timestamp: "1120 03:14:15 ds4-server: unknown option: --x".
DS4_SAID = re.compile(r"^(?:\d{4} \d\d:\d\d:\d\d )?ds4-server: ")


def _log_tail(name: str = "ds4-server.log", n: int = 6, said: Optional[re.Pattern] = None) -> str:
    """A log's last ``n`` lines. With ``said``: when the last line it matches is not among them (ds4-server prints
    its whole usage text after a bad option), that line first, then the last 3."""
    try:
        lines = (_ds4_home() / name).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    tail = lines[-n:]
    own = [x for x in lines if said and said.match(x)]
    if own and own[-1] not in tail:
        tail = [own[-1], *lines[-3:]]
    return "\n".join("  " + x for x in tail)


def _why_stopped(since: Optional[float] = None) -> str:
    """ds4-server's last lines (a failed load; its own error first, ``_log_tail``) when it ran since ``since``, then
    serve's own (a GGUF refused, oMLX not freed, a port taken): the reason last, as Pi keeps a switch's last lines."""
    path = _ds4_home() / "ds4-server.log"
    try:
        ran = since is None or path.stat().st_mtime >= since
    except OSError:
        ran = False
    return (f"ds4-server ({path}):\n{_log_tail(said=DS4_SAID)}\n" if ran else "") + \
        f"`ds4 serve` ({_ds4_home() / 'serve.log'}):\n{_log_tail('serve.log')}"


def ds4_start(spec: Optional[str] = None, name: Optional[str] = None, ctx: Optional[int] = None,
              args: Optional[list[str]] = None, wait: bool = True, log: Log = print) -> dict:
    """Serve a ds4 model in the background: a different ds4 model stopped, then `ds4 serve` started (it unloads oMLX's
    models first: free_for_ds4) at the model's own maximum context (or `ctx`); with `wait`, until it answers, else until it loads. No
    `spec`: the model ds4 serves or last served, else its only one. Refused before anything is stopped or unloaded when
    the [ds4] extra is missing, and before oMLX's model is unloaded when the proxy's or ds4-server's port is taken."""
    require_ds4_extra()
    if spec is None:
        cur = ds4_serving()
        last = config.read_json(_ds4_home() / "last.json", {}).get("name")
        models = ds4_models()
        spec = (cur or {}).get("name") or last or (models[0]["name"] if len(models) == 1 else None)
        if spec is None:
            raise SystemExit("which model? " + (", ".join(m["name"] for m in models) or
                                                "none yet: metalscope ds4 PATH.gguf serves one"))
    try:
        m = ds4_model(spec, name)
    except SystemExit:
        # A model on its first load is remembered only once it has loaded (register): the one loading is that name.
        cur = ds4_serving()
        if not (cur and cur["name"] == spec and args is None and ctx in (None, cur["ctx"])):
            raise
        m = dict(name=spec, args=[], ctx=cur["ctx"])
    changed = False                               # other ds4-server options than the model runs with: a restart
    if args is not None:
        new, given = strip_ctx(args)
        changed, m["args"] = new != m["args"], new
        ctx = ctx or given
    asked = ctx                                   # a context asked for; else whatever the model runs at is right
    ctx = ctx or m["ctx"] or DEFAULT_CTX          # the model's maximum
    with switching(log):
        cur = ds4_serving()
        same = cur and cur["name"] == m["name"] and not changed and (asked is None or cur["ctx"] in (None, asked))
        if not same and "gguf" not in m:
            raise SystemExit(f"ds4 stopped while loading {m['name']} for the first time: give its GGUF path")
        if not same:
            from .ds4.build import refuse_stale
            refuse_stale()                        # before anything is stopped: serve would only exit while loading
            gpu_free()
            if cur:
                _ds4_stop(log)                    # its ports are ours until it has stopped
            require_free_ports({urllib.parse.urlsplit(client.DS4_URL).port: "metalscope's ds4 proxy",
                                DS4_SERVER_PORT: "ds4-server"})
            cmd = [sys.executable, "-m", "metalscope.cli", "ds4", "serve", m["gguf"], "--name", m["name"],
                   "--", *m["args"], "--ctx", str(ctx)]
            with open(config.rotated_log(_ds4_home() / "serve.log"), "ab") as out:
                out.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
                out.flush()
                proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
            write_serving(m["name"], ctx, pid=proc.pid)
            log(f"ds4 loading {m['name']} at {ctx:,} context (log {_ds4_home() / 'ds4-server.log'})")
    try:
        if same:
            log(f"ds4 already serves {m['name']}" + ("" if cur["ready"] else " (loading)"))
            if wait and not cur["ready"]:
                _wait_ds4(m["name"], None, log)
            return dict(m, ctx=cur["ctx"] or ctx)
        _wait_ds4(m["name"], proc, log, loading=not wait)
        return dict(m, ctx=ctx)
    except KeyboardInterrupt:           # serve runs in its own session: only the wait stops
        log(f"ds4 goes on loading {m['name']} in the background (metalscope ds4 stop stops it)")
        raise


def _wait_ds4(name: str, proc: Optional[subprocess.Popen], log: Log, timeout: float = DS4_LOAD_TIMEOUT,
              loading: bool = False) -> None:
    """Until ds4 answers; with ``loading``, until ds4-server is loading (oMLX freed: serve records its pid then)."""
    t0, since = time.monotonic(), (time.time() if proc is not None else None)
    said = t0
    while time.monotonic() - t0 < timeout:
        if loading and config.read_json(_ds4_home() / "serving.json", {}).get("server"):
            return
        data = _proxy_models()
        if data and data[0].get("id") == name:
            log(f"ds4 serves {name} on {client.DS4_URL} ({time.monotonic() - t0:.0f} s)")
            return
        if (proc is not None and proc.poll() is not None) or (proc is None and not _serve_pids()):
            raise SystemExit(f"ds4 stopped while loading {name}; the logs end:\n{_why_stopped(since)}")
        if time.monotonic() - said >= 20:
            log(f"  loading {name}… {time.monotonic() - t0:.0f} s")
            said = time.monotonic()
        time.sleep(1.0)
    raise SystemExit(f"ds4 did not answer within {timeout:.0f} s; the logs end:\n{_why_stopped(since)}")


# ---------------------------------------------------------------- oMLX

def _admin():
    return omlx_admin.Admin()


def _omlx_cache() -> Path:
    """oMLX's chat models as last listed (omlx_models), for when it is not running."""
    return config.home() / "omlx" / "models.json"


def free_omlx(log: Log = print) -> None:
    """Unload oMLX's loaded models (above a gigabyte: a document converter can stay) so another server's model fits."""
    from . import gpulock
    a = _admin()
    if not a.listening():
        return
    try:
        if any(m.get("is_loading") for m in a.admin_models()):     # oMLX refuses to unload a model mid-load
            log("waiting for oMLX's load to finish before unloading it")
            if not gpulock.wait_no_loading(a, log=log):
                raise SystemExit("oMLX is still loading a model; try again once it has loaded")
        loaded = [m["id"] for m in a.admin_models()
                  if (m.get("loaded") or m.get("is_loading")) and (m.get("estimated_size") or 0) > 1 << 30]
        for mid in loaded:
            omlx_admin.unload(a, mid, log=log)
    except omlx_admin.ServerError as exc:
        raise SystemExit(f"could not free oMLX's memory ({exc}); unload its model first (metalscope omlx unload ID)")


def omlx_start(model: Optional[str] = None, log: Log = print) -> None:
    """Start oMLX's server if it is not running; with a model, ds4 stopped first (its memory) and the model loaded."""
    with switching(log):
        try:
            _omlx_start(model, log)
        except omlx_admin.ServerError as exc:
            raise SystemExit(f"oMLX: {exc}")


def _omlx_start(model: Optional[str], log: Log) -> None:
    # oMLX is started (no model loads) and asked for ``model`` before ds4 is stopped: a name it does not have never
    # frees ds4's model. The list from when it last ran (chat models only, maybe old) only says why it starts.
    a = _admin()
    up = a.answering()
    if model and not up:
        last = [m["name"] for m in config.read_json(_omlx_cache(), [])]
        if last and model not in last:
            log(f"{model} was not among oMLX's models when it last ran; starting oMLX to look")
    if not up:
        if not omlx_admin.OMLX_CLI.exists():
            raise SystemExit(f"{omlx_admin.OMLX_CLI} not found: open oMLX.app once")
        gpu_free()
        log("starting oMLX")
        omlx_admin.cli_start(omlx_admin.OMLX_CLI, "--timeout", "120")
        if not a.wait(up=True, timeout=120):
            raise SystemExit(f"oMLX did not answer at {a.base_url}")
        log(f"oMLX answering at {a.base_url}")
    else:
        log(f"oMLX running at {a.base_url}")
    if model:
        try:
            state = {m["id"]: m for m in a.admin_models()}
        except omlx_admin.ServerError as exc:
            raise SystemExit(str(exc))
        if model not in state:
            raise SystemExit(f"oMLX has no model {model!r} (its models: {', '.join(state)})")
        _ds4_stop(log)
        if state[model].get("loaded"):
            log(f"{model} already loaded")
        else:
            gpu_free()
            omlx_admin.load(a, model, log=log)


def omlx_stop(log: Log = print) -> bool:
    """Stop oMLX's server (the app's `omlx stop`); False when it was not running. As ``ds4_stop``: one at a time with a
    switch, refused under a GPU job."""
    with switching(log):
        a = _admin()
        if not a.listening():
            return False
        gpu_free("stop oMLX")
        subprocess.run([str(omlx_admin.OMLX_CLI), "stop", "--timeout", "60"], capture_output=True, timeout=90)
        if not a.wait(up=False, timeout=60):
            raise SystemExit(f"oMLX still answering at {a.base_url}")
    log("oMLX stopped")
    return True


def omlx_models() -> dict:
    """oMLX's chat models (name, ctx, loaded, thinking controls; tested: metalscope was run on its family, as
    metalscope's status says, else (oMLX running without our hook, or chat_formats returning {}) by the admin API's
    config_model_type; edit: oMLX takes the dock's one-token edit for it, as the status says; each left out where
    not known) as it lists them now, or as last seen when it is not running (up False)."""
    cache = _omlx_cache()
    a = _admin()
    up = a.listening()
    try:
        data = a._request("GET", "/v1/models", bearer=True, timeout=3)[1].get("data", []) if up else None
    except omlx_admin.ServerError:
        data = None
    if data is None:
        return dict(up=up, url=a.base_url, models=config.read_json(cache, []), tracing=None)
    try:
        admin = {m["id"]: m for m in a.admin_models()}
    except omlx_admin.ServerError:
        admin = {}
    trace = a.trace_status() or {}
    chat = trace.get("chat") or {}

    def flags(mid: str) -> dict:
        c = chat.get(mid) or {}
        out = {k: c[k] for k in ("tested", "edit") if k in c}
        if "tested" not in out and mid in admin:
            out["tested"] = tested("omlx", admin[mid].get("config_model_type"))
        return out
    models = [dict(name=m["id"], ctx=m.get("max_model_len"), loaded=bool(admin.get(m["id"], {}).get("loaded")),
                   reasoning=bool((chat.get(m["id"]) or {}).get("reasoning")),
                   thinking_format=(chat.get(m["id"]) or {}).get("thinking_format"), **flags(m["id"]))
              for m in data if not any(s in m["id"].lower() for s in CHAT_MODEL_SKIP)]
    _write_json(cache, [dict(x, loaded=False) for x in models])
    return dict(up=True, url=a.base_url, models=models, tracing=trace.get("mode"))


def require_served(name: str) -> None:
    """ValueError (client.not_served) unless a backend can answer for ``name`` now: oMLX running and listing it (it
    loads a model it lists when a request names it), or the ds4 proxy serving it. ``concept learn`` and ``calibrate``
    check this first; otherwise they would fall back to oMLX and fail later with a less clear error."""
    want = config.short_name(name)
    omlx = omlx_models()
    if omlx["up"] and want in (config.short_name(m["name"]) for m in omlx["models"]):
        return
    cur = ds4_serving()
    if cur and cur.get("name") and config.short_name(cur["name"]) == want:
        return
    raise ValueError(client.not_served(name))


def model_lines(engines: tuple = ("omlx", "ds4")) -> list[str]:
    """A table of the models each backend has, nothing loaded (doctor, the end of an install), a heading then a row
    per model: tested here or not (models.TESTED), its J-lens (yes, off, no), the concepts learned for it, whether its
    server takes the dock's one-token edit (`?`: not known). oMLX's models as it lists them, or as last seen.
    Information only: whatever goes wrong is one line, never an error."""
    try:
        from . import jlens, models
        have = [("oMLX", m) for m in omlx_models()["models"]] if "omlx" in engines else []
        have += [("ds4", dict(name=m["name"], **ds4_chat(m["family"]))) for m in ds4_models()] if "ds4" in engines else []
        word = lambda v: "?" if v is None else "yes" if v else "no"
        rows = [("backend", "model", "tested", "J-lens", "concepts", "edit")] if have else []
        for engine, m in have:
            lens = "yes" if jlens.find_for(m["name"]) else "off" if jlens.pick_jlens(m["name"]) else "no"
            rows.append((engine, m["name"], word(m.get("tested")), lens, str(len(models.profile_paths(m["name"]))),
                         word(m.get("edit"))))
        wide = [max(len(r[i]) for r in rows) for i in range(6)] if rows else []
        return ["  ".join(c.ljust(w) for c, w in zip(r, wide)).rstrip() for r in rows]
    except Exception as exc:
        return [f"the models could not be listed: {type(exc).__name__}: {exc}"]


def status() -> dict:
    """Both backends: running or not, each one's models (with the context each runs at) and what is loaded; memory."""
    omlx = omlx_models()
    cur = ds4_serving()
    ds4 = dict(up=cur is not None, ready=bool(cur and cur["ready"]), url=client.DS4_URL,
               serving=cur["name"] if cur else None,
               orphaned=[] if cur else _ds4_servers(),        # its serve gone (killed), the model still in memory
               models=[dict(name=m["name"], ctx=(cur["ctx"] if cur and cur["name"] == m["name"] and cur["ctx"] else m["ctx"]),
                            gguf=m["gguf"], args=m["args"], loaded=bool(cur and cur["name"] == m["name"]),
                            **ds4_chat(m["family"]))
                       for m in ds4_models()])
    return dict(omlx=omlx, ds4=ds4, memory_free=memory_free(), mode=config.load_settings()["mode"])

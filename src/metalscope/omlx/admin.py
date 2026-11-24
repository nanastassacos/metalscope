"""Drive the oMLX server from the command line (`metalscope omlx ...`).

oMLX's interface, as used here:

    ~/.omlx/settings.json                    server.port, auth.api_key
    POST /admin/api/login {"api_key": ..}    sets the omlx_admin_session cookie
    GET  /v1/models                          served ids (Bearer key)
    GET  /admin/api/models                   every model with loaded / is_loading
    PUT  /admin/api/models/{id}/settings     partial model settings
    POST /admin/api/models/{id}/load|unload  body {}; a load can take minutes
    POST /admin/api/server/restart           202; the menu-bar app respawns it
    ~/.omlx/bin/omlx restart --timeout 120   asks the app over its control socket

The API key is sent to the server and nowhere else; it is never printed.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

from .. import config

SETTINGS = Path.home() / ".omlx/settings.json"
OMLX_CLI = Path.home() / ".omlx/bin/omlx"
OMLX_LOG = Path.home() / ".omlx/logs/server.log"
# oMLX answers 404 for /scope with the hook in place: not restarted since install, or metalscope failed to load in it.
NOT_LOADED = f"metalscope omlx restart; if it stays, its log says why ({OMLX_LOG})"
OMLX_LOAD_TIMEOUT = 900


class ServerError(Exception):
    """A failure worth showing the user as is; ``status`` the HTTP code when oMLX answered with an error."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def read_settings(path: Path | None = None) -> dict:
    return config.read_json(path or SETTINGS, {})


def server_url(settings: dict | None = None) -> str:
    server = (read_settings() if settings is None else settings).get("server") or {}
    host = server.get("host", "127.0.0.1")
    return f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{server.get('port', 8008)}"


def parse_assignments(pairs: list[str]) -> dict:
    """``key=value`` -> {key: value}; values are JSON where they parse (true, 4), else strings."""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ServerError(f"expected key=value, got {pair!r}")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def no_key_message(base_url: str) -> str:
    """oMLX's admin API answers nothing until an API key is set, which a fresh oMLX has not (its admin page asks)."""
    return (f"oMLX has no API key set, and metalscope needs one to load and unload models: open {base_url}/admin "
            "and set one (it asks on the first visit)")


def key_refused(base_url: str) -> str:
    """What doctor and `omlx status` say when oMLX answers metalscope's /scope with 401: the file the key comes from,
    never the key."""
    return (f"oMLX at {base_url} refused metalscope's API key: it sends the one in {SETTINGS}, "
            "and oMLX may run with another (OMLX_API_KEY, --api-key)")


class Admin:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, settings_path: Path | None = None):
        settings = read_settings(settings_path)
        self.base_url = (base_url or server_url(settings)).rstrip("/")
        self.api_key = api_key if api_key is not None else (settings.get("auth") or {}).get("api_key")
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        self._logged_in = False
        self.trace_error: int | None = None     # trace_status's last ask: its HTTP error (401: the key refused)

    def _request(self, method: str, path: str, body=None, timeout: float = 10, bearer: bool = False):
        headers = {"Content-Type": "application/json"} if body is not None else {}
        if bearer and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                raw = resp.read()
                return resp.status, json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise ServerError(f"{method} {path}: HTTP {exc.code}" + (f": {detail}" if detail else ""), exc.code) from None
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):      # up but slow (a load can take minutes): not a reason to restart
                raise ServerError(f"oMLX did not answer {method} {path} within {timeout:.0f}s: it may still be working "
                                  "on it (metalscope omlx status)") from None
            raise ServerError(f"oMLX server not reachable at {self.base_url} ({reason}); "
                              "start oMLX or run `metalscope omlx restart`") from None

    def listening(self) -> bool:
        url = urllib.parse.urlsplit(self.base_url)
        try:
            with socket.create_connection((url.hostname, url.port or 80), timeout=1):
                return True
        except OSError:
            return False

    def login(self) -> None:
        if self._logged_in:
            return
        if not self.api_key:
            raise ServerError(no_key_message(self.base_url))
        try:
            self._request("POST", "/admin/api/login", {"api_key": self.api_key})
        except ServerError as exc:
            if exc.status == 401:
                raise ServerError(f"admin login failed: the API key in {SETTINGS} was rejected") from None
            if exc.status == 400:
                raise ServerError(no_key_message(self.base_url)) from None
            raise
        self._logged_in = True

    def _admin(self, method: str, path: str, body=None, timeout: float = 10):
        self.login()
        return self._request(method, "/admin/api" + path, body, timeout=timeout)[1]

    def models(self) -> list[str]:
        return [m["id"] for m in self._request("GET", "/v1/models", bearer=True)[1].get("data", [])]

    def admin_models(self) -> list[dict]:
        return self._admin("GET", "/models").get("models", [])

    def trace_status(self) -> dict | None:
        """metalscope's own status route, or None when it isn't mounted or refuses the key (``trace_error``: the HTTP
        status, None when nothing answered)."""
        self.trace_error = None
        try:
            return self._request("GET", "/scope/api/status", timeout=3, bearer=True)[1]
        except ServerError as exc:
            self.trace_error = exc.status
            return None

    def load(self, model_id: str) -> dict:
        return self._admin("POST", f"/models/{_quote(model_id)}/load", {}, timeout=OMLX_LOAD_TIMEOUT)

    def unload(self, model_id: str) -> dict:
        return self._admin("POST", f"/models/{_quote(model_id)}/unload", {}, timeout=120)

    def update_settings(self, model_id: str, changes: dict) -> dict:
        return self._admin("PUT", f"/models/{_quote(model_id)}/settings", changes, timeout=120)

    def restart_api(self) -> dict:
        return self._admin("POST", "/server/restart", {})

    def answering(self) -> bool:
        """/v1/models gives an HTTP answer (an auth failure counts: it won't fix itself by waiting)."""
        try:
            self._request("GET", "/v1/models", timeout=2, bearer=True)
            return True
        except ServerError as exc:
            return exc.status in (401, 403)

    def wait(self, up: bool, timeout: float, poll: float = 0.5) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.answering() if up else not self.listening():
                return True
            time.sleep(poll)
        return False


def _quote(model_id: str) -> str:
    return urllib.parse.quote(model_id, safe="")


def _pid(base_url: str | None = None) -> str | None:
    """The pid listening on the server's port (another omlx-server, a test copy on another port say, is not it);
    pgrep's first omlx-server when lsof finds nothing to ask."""
    port = urllib.parse.urlparse(base_url or server_url()).port or 8008
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True,
                             text=True).stdout.split()
        if out:
            return out[0]
    except OSError:
        pass
    try:
        return subprocess.run(["pgrep", "-x", "omlx-server"], capture_output=True, text=True).stdout.split()[0]
    except (OSError, IndexError):
        return None


APP_SETTLE = 5.0        # seconds oMLX's menu-bar app is given before its CLI is asked to start the server


def _app_to_open() -> Path | None:
    """oMLX's menu-bar app when it is installed and not running; None too when that cannot be told."""
    from .install import DEFAULT_APP
    try:
        gone = subprocess.run(["pgrep", "-f", "/Contents/MacOS/oMLX$"], capture_output=True).returncode == 1
    except OSError:
        return None
    return DEFAULT_APP if gone and DEFAULT_APP.exists() else None


def cli_start(cli: Path, *more: str) -> None:
    """`omlx start`, oMLX's menu-bar app opened first when it is not running (in the background). Left to open the
    app itself, `omlx start` can fail to control it (oMLX 0.8.2: "Failed to control oMLX.app: timed out"): the app
    then quits within half a minute, the server is left with nothing to start it again, and each later restart shows
    the app's icon coming and going (2026-11-22). Opened first, the app stays and owns the server."""
    app = _app_to_open()
    if app:
        subprocess.run(["open", "-g", "-a", str(app)], capture_output=True)
        time.sleep(APP_SETTLE)
    subprocess.run([str(cli), "start", *more], capture_output=True, timeout=150)


def _hooked(app: Path) -> bool:
    """oMLX.app's Python runs this metalscope's current hook."""
    from . import install
    try:
        return install.hook_state(app)["current"]
    except (OSError, subprocess.CalledProcessError):
        return False


def status(admin: Admin, log=print, app: Path | None = None) -> None:
    """The server, its models and metalscope's part in it; ``app`` the oMLX.app whose version explains missing routes
    (default: install.run_app())."""
    if not admin.listening():
        log(f"server {admin.base_url}: not listening")
        return
    pid = _pid(admin.base_url)
    log(f"server {admin.base_url}: listening" + (f" (omlx-server pid {pid})" if pid else ""))
    try:
        served = admin.models()
    except ServerError as exc:
        log(f"served models: unavailable ({exc})")
        served = None
    try:
        state = {m["id"]: m for m in admin.admin_models()}
        note = ""
    except ServerError as exc:
        state, note = {}, f"  (loaded state unavailable: {exc})"
    if served is not None:
        log(f"served models ({len(served)}):" + note)
    for mid in served or []:
        m = state.get(mid, {})
        log(f"  {mid}" + ("  [loaded]" if m.get("loaded") else "  [loading]" if m.get("is_loading") else ""))
    trace = admin.trace_status()
    if trace is None and admin.trace_error == 401:
        log(f"metalscope: {key_refused(admin.base_url)}")
    elif trace is None:
        from . import install
        app = app or install.run_app()
        untested = install.untested(install.omlx_version(app))
        if untested:                  # installing again changes nothing: the hook loads and leaves oMLX alone
            log("metalscope: off in this oMLX")
            for line in untested:
                log(f"  {line}")
        elif admin.trace_error == 404 and _hooked(app):      # installing again changes nothing
            log(f"metalscope: not loaded in this oMLX: {NOT_LOADED}")
        else:
            log("metalscope: routes not mounted (run `metalscope omlx install`, then restart oMLX)")
    else:
        hooked = [t["name"] for t in trace.get("models", []) if t.get("installed")]
        log(f"metalscope: tracing {trace.get('mode')}; hooks on {', '.join(hooked) or 'no model'}")


def restart(admin: Admin, model: str | None = None, cli: Path | None = OMLX_CLI, timeout: float = 180,
            down_timeout: float = 15, log=print) -> None:
    """Restart through the admin API (oMLX's menu-bar app starts the server again), the bundled CLI when the API
    cannot; wait for /v1/models, then load ``model``. One route at a time: a second restart sent while the first is
    under way can leave the server down for minutes."""
    t0 = time.monotonic()
    old_pid = _pid(admin.base_url) if admin.base_url == server_url() else None  # pgrep only knows the local server
    cli = cli if cli and Path(cli).exists() else None
    try:
        admin.restart_api()
        admin.wait(up=False, timeout=down_timeout)  # the old process answers for a moment after the 202
        via = "admin API"
    except ServerError as exc:
        if not cli:
            raise
        log(f"the admin API could not restart it ({exc}); trying `omlx restart`")
        try:
            subprocess.run([str(cli), "restart", "--timeout", "120"], capture_output=True, timeout=150)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log(f"`omlx restart` failed ({exc})")
        via = "omlx CLI"
    log(f"restart requested via {via} ({time.monotonic() - t0:.1f}s)")
    if old_pid:  # don't let the old process satisfy the wait below
        end = time.monotonic() + down_timeout
        while _pid(admin.base_url) == old_pid and time.monotonic() < end:
            time.sleep(0.5)
    up = admin.wait(up=True, timeout=min(timeout, 30) if cli else timeout)
    # Nothing started it again (no menu-bar app behind it, or a restart that only stopped it). Up to three times: on
    # oMLX 0.8.2 `omlx start` can exit 0 with "Failed to control oMLX.app: timed out" and no server (2026-11-22).
    for _ in range(3 if cli else 0):
        if up:
            break
        log("server did not come back; running `omlx start`")
        try:
            cli_start(cli)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log(f"`omlx start` failed ({exc})")
        up = admin.wait(up=True, timeout=timeout / 3)
    if not up:
        raise ServerError(f"server not answering at {admin.base_url} after {time.monotonic() - t0:.0f}s")
    log(f"server answering after {time.monotonic() - t0:.1f}s")
    if model:
        t1 = time.monotonic()
        admin.load(model)
        log(f"loaded {model} in {time.monotonic() - t1:.1f}s (total {time.monotonic() - t0:.1f}s)")


def load(admin: Admin, model_id: str, log=print) -> None:
    t0 = time.monotonic()
    res = admin.load(model_id)
    log(f"{res.get('message') or 'loaded ' + model_id} ({time.monotonic() - t0:.1f}s)")


def unload(admin: Admin, model_id: str, log=print, timeout: float = 180.0) -> None:
    """Unload and wait until the model is gone. oMLX answers 202 ("Aborting
    active requests before unloading") while requests are in flight, before
    the memory is free; returning then let a following load overlap it."""
    from ..gpulock import wait_unloaded
    res = admin.unload(model_id)
    log(res.get("message") or f"unloading {model_id}")
    if not wait_unloaded(admin, model_id, timeout=timeout, log=log):
        raise ServerError(f"{model_id} did not unload within {timeout:.0f}s")
    if res.get("status") == "unloading":
        log(f"unloaded {model_id}")


def settings(admin: Admin, model_id: str, pairs: list[str], log=print) -> None:
    changes = parse_assignments(pairs)
    before = next((m.get("settings") or {} for m in admin.admin_models() if m.get("id") == model_id), {})
    res = admin.update_settings(model_id, changes)
    after = res.get("settings") or {}
    changed = False
    for key in changes:
        if key not in after:
            log(f"  {key}: not a model setting (ignored)")
        elif before.get(key) != after[key]:
            log(f"  {key}: {json.dumps(before.get(key))} -> {json.dumps(after[key])}")
            changed = True
    if not changed:
        log("no changes")
    if res.get("auto_reloaded"):
        log(f"{model_id} reloaded with the new settings")
    elif res.get("auto_unloaded"):
        log(f"{model_id} unloaded to apply them; `metalscope omlx load {model_id}` to load it again")
    elif res.get("reload_deferred"):
        log(f"{model_id} unloads once its current requests finish")
    elif res.get("requires_reload"):
        log("takes effect on the next load")

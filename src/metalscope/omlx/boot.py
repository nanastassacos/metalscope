"""Activate metalscope inside oMLX without editing oMLX.

``metalscope omlx install`` drops a .pth file into the user site-packages of oMLX's
bundled Python (shared by every Python 3.11 on the Mac). Python runs its guarded line
at startup, which imports this module only inside oMLX.app (install.pth_line).
It only registers an import hook (no mlx, no oMLX imports here), so it costs
nothing in oMLX's other processes. When oMLX imports one of the modules below, the
matching patch runs right after the module body executes.

On an oMLX version patches.py is not verified against (install.TESTED_OMLX) it stands
down: nothing is patched and nothing of metalscope's stack is imported (no mlx, numpy
or fastapi, no thread pools). METALSCOPE_FORCE=1 patches anyway.

Set METALSCOPE_DISABLE=1 to start oMLX without it.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys

_TARGETS = {
    "omlx.engine.vlm": "patch_vlm_engine",
    "omlx.engine.batched": "patch_batched_engine",
    "omlx.engine_core": "patch_engine_core",
    "omlx.scheduler": "patch_scheduler",
    "omlx.cache.prefix_cache": "patch_prefix_cache",
    "omlx.engine_pool": "patch_engine_pool",
    "omlx.server": "patch_server",
}
_supported: bool | None = None      # decided when the first oMLX module arrives (omlx is importable by then)


def _log():
    import logging
    return logging.getLogger("metalscope")


def omlx_supported() -> bool:
    global _supported
    if _supported is None:
        from ..config import env
        from .install import TESTED_OMLX
        try:
            from omlx._version import __version__ as version
        except ImportError:
            version = "unknown"
        _supported = version in TESTED_OMLX or env("FORCE") == "1"
        if not _supported:
            _log().warning(
                "metalscope: oMLX %s is untested (tested: %s); not patching. Set METALSCOPE_FORCE=1 to try.",
                version, ", ".join(sorted(TESTED_OMLX)))
    return _supported


def _run_patch(name: str, module) -> None:
    try:
        if not omlx_supported():
            return
        from . import patches
        getattr(patches, _TARGETS[name])(module)
    except Exception as exc:  # never take the server down
        _log().exception("metalscope: patching %s failed: %s", name, exc)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._busy: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy.discard(fullname)
        if spec is None:
            return None
        if type(spec.loader) is not importlib.machinery.SourceFileLoader:    # made for this spec: ours to change
            _log().warning("metalscope: %s has no source loader (%r); not patched", fullname, spec.loader)
            return None

        # The module's own loader, its exec_module followed by the patch (get_source and the rest stay as they are).
        def exec_module(module, _exec=spec.loader.exec_module):
            _exec(module)
            _run_patch(fullname, module)

        spec.loader.exec_module = exec_module
        return spec


def _activate() -> None:
    from ..config import env
    if env("DISABLE") == "1":
        return
    if any(isinstance(f, _PostImportFinder) for f in sys.meta_path):
        return
    sys.meta_path.insert(0, _PostImportFinder())
    # Modules already imported (e.g. boot loaded late) get patched now.
    for name in _TARGETS:
        if name in sys.modules:
            _run_patch(name, sys.modules[name])


_activate()

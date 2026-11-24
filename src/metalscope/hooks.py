"""Instance-level forward hooks for MLX modules.

MLX has no ``register_forward_hook``. Wrapping a module in another module
changes parameter paths and breaks code that reaches into children by key
(oMLX's fused paths do ``self["mlp"]``, ``"block_inject_weight" in self``).
Instead we swap the *instance's* class for a generated subclass whose
``__call__`` runs post-hooks around the original. Parameters, attribute
access and ``isinstance`` checks are unchanged, and removing the hook
restores the original class.

A hook that raises never breaks the forward: it is logged (at most every 30 s) and its ``on_error(exc)`` runs, so
its owner can mark what it was recording as incomplete (a capture session: the trace's ``meta.partial``).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import mlx.nn as nn

log = logging.getLogger("metalscope")
_last_error_log = [0.0]

# hook(module, args, kwargs, output) -> replacement output or None
PostHook = Callable[[nn.Module, tuple, dict, Any], Any]
OnError = Callable[[BaseException], None]

_HOOKS_ATTR = "_metalscope_hooks"
_ORIG_CLASS_ATTR = "_metalscope_orig_class"
_traced_classes: dict[type, type] = {}


def _traced_class(cls: type) -> type:
    sub = _traced_classes.get(cls)
    if sub is not None:
        return sub

    def __call__(self, *args, **kwargs):
        return run_hooks(self, args, kwargs, cls.__call__(self, *args, **kwargs))

    sub = type(f"Traced{cls.__name__}", (cls,), {"__call__": __call__})
    _traced_classes[cls] = sub
    return sub


def run_hooks(module: nn.Module, args: tuple, kwargs: dict, out: Any) -> Any:
    """``module``'s hooks on its output ``out``, as its call runs them; also for code that computes a hooked module's
    output without calling it (omlx/patches_mtp.patch_mtp_verify). Nothing for a module without hooks."""
    # A snapshot, read once: hooks are added and removed on another thread (oMLX's event loop, while its
    # executor runs the forward), and the last removal deletes the list.
    for hook, on_error in tuple(module.__dict__.get(_HOOKS_ATTR, ())):
        try:
            replaced = hook(module, args, kwargs, out)
        except Exception as exc:
            now = time.monotonic()
            if now - _last_error_log[0] > 30:
                _last_error_log[0] = now
                log.exception("metalscope: hook on %s failed; continuing without it",
                              module.__dict__.get(_ORIG_CLASS_ATTR, type(module)).__name__)
            if on_error is not None:
                try:
                    on_error(exc)
                except Exception:
                    log.exception("metalscope: a hook's failure callback failed")
            continue
        if replaced is not None:
            out = replaced
    return out


def add_hook(module: nn.Module, hook: PostHook, on_error: OnError | None = None) -> Callable[[], None]:
    """Attach ``hook`` to ``module`` (``on_error(exc)`` when it raises); returns a function that removes it."""
    hooks = module.__dict__.get(_HOOKS_ATTR)
    if hooks is None:
        # object.__setattr__ keeps these out of the module's parameter tree.
        hooks = []
        object.__setattr__(module, _HOOKS_ATTR, hooks)
        object.__setattr__(module, _ORIG_CLASS_ATTR, type(module))
        module.__class__ = _traced_class(type(module))
    entry = (hook, on_error)
    hooks.append(entry)

    def remove() -> None:
        if entry in hooks:
            hooks.remove(entry)
        if not hooks and _HOOKS_ATTR in module.__dict__:
            module.__class__ = module.__dict__[_ORIG_CLASS_ATTR]
            del module.__dict__[_HOOKS_ATTR]
            del module.__dict__[_ORIG_CLASS_ATTR]

    return remove


def is_hooked(module: nn.Module) -> bool:
    return bool(module.__dict__.get(_HOOKS_ATTR))

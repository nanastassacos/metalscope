"""The hooks on one loaded model (ModelTracer): steering, the capture of traced rows and the step closer, and the
layers each lens step reads (lens_steps). live.TraceManager owns the tracers and the requests they record."""

from __future__ import annotations

import dataclasses
import functools
import logging
import threading
from typing import TYPE_CHECKING, Optional

import mlx.core as mx

from . import config, jlens
from .adapters import install as install_adapter
from .capture import CaptureSession, capture_config
from .lookup import Kind
from .mtp import mtp_decode, window_starts
from .steer import SteerSession

if TYPE_CHECKING:
    from .live import TraceManager

log = logging.getLogger("metalscope")


def spread_layers(layers: list[int], k: int) -> list[int]:
    """``k`` of ``layers`` (ascending) spread evenly over them, the last always: the ((i + 1) n / k)th of the n, i
    from 0; all of them where there are no more than ``k``."""
    n = len(layers)
    return list(layers) if n <= k else [layers[(i + 1) * n // k - 1] for i in range(k)]


def lens_steps(setup: Optional["jlens.LensSetup"], n_layers: int) -> list[tuple[list[int], list[int]]]:
    """The layers each of LENS's steps (- + in the dock, settings' lens_step; config.LENS_SIZES: small, medium, large)
    reads, fewest first: (the J-lens's, the logit lens's), as many as config.lens_reads gives for the model's depth,
    spread over the depth (spread_layers), the last always. The main lens's: the J-lens's over the layers it has (all
    of them where it has fewer: 16 of 47 at medium on Flash Next, 8 of 8 on a lens fitted on 8), else the logit lens's
    over the model's. The logit lens beside a J-lens reads the same layers and the model's last (its output), so
    either lens fills the same rows (s in the dock)."""
    out = []
    for k in config.lens_reads(n_layers):
        if setup is None:
            out.append(([], spread_layers(list(range(n_layers)), k)))
        else:
            jl = spread_layers(setup.lens.layers, k)
            out.append((jl, sorted({*jl, n_layers - 1})))
    return out


class ModelTracer:
    """Hooks for one loaded model: the steering session, the capture session that records traced rows, and the
    closer that ends each step."""

    def __init__(self, model, name: str, tokenizer):
        self.model, self.name, self.tokenizer = model, name, tokenizer
        self.steer: Optional[SteerSession] = None
        self.capture: Optional[CaptureSession] = None
        self.closer: Optional[CaptureSession] = None
        self.n_layers: Optional[int] = None
        self.adapter_name: Optional[str] = None
        self.lens: Optional[jlens.LensSetup] = None
        self.lock = threading.Lock()
        self.hc_count = 1
        self.install_error: Optional[str] = None   # why install failed (raised again until a reload or a settings change)
        self.lens_step = config.LENS_STEP_DEFAULT   # the step the capture config reads the lenses at (use_lens_step)
        self.phase = "decode"   # the forward under way's (rows_for): its decode speed counts decode forwards alone
        self.fresh: list = []   # (a request's parts list, index) its last step filed, still MLX arrays (TraceManager._host)

    @property
    def installed(self) -> bool:
        return self.closer is not None      # hooked last

    @functools.cached_property
    def lookup(self) -> bool:
        """The model adds an n-gram lookup to its residual (Flash Next's PLE, a decoder layer's ``ple``): lookup.py."""
        from .adapters.generic import text_model
        try:
            return any("ple" in layer for layer in text_model(self.model).layers)
        except Exception:
            return False

    @property
    def mtp(self) -> bool:
        """MTP speculative decode: records are filed by position (mtp.py). Read live: oMLX gives a DSpark or DFlash
        model its drafter after the engine has started, and can take it away."""
        return mtp_decode(self.model)

    @functools.cached_property
    def lens_columns(self):
        """(layer, its lens Jacobian's width) -> the columns of that Jacobian a steer writes (adapters.lens_columns;
        Inkling 2: the running sum's), or None: every column (also when no adapter reads the model)."""
        from .adapters import lens_columns
        try:
            return lens_columns(self.model)
        except Exception:
            return None

    def edit_band(self) -> tuple[int, int]:
        """The layers the dock's word edit removes over (adapters.edit_band)."""
        from .adapters import edit_band
        return edit_band(self.model, self.layer_count())

    def layer_count(self) -> int:
        """The layers a trace records (the adapter's count; Inkling 2: its layer visits), before the hooks say. On a
        model no adapter reads, RuntimeError: why the hooks could not be (or cannot be) installed, as ``install``."""
        from .adapters import layer_count
        if self.n_layers:
            return self.n_layers
        if self.install_error is not None:
            raise RuntimeError(self.install_error)
        try:
            return layer_count(self.model)
        except ValueError as exc:
            raise RuntimeError(self._why(exc)) from None

    def _why(self, exc: Exception) -> str:
        return f"hooks could not be installed on {self.name}: {type(exc).__name__}: {exc}"

    @property
    def lookup_kind(self):
        """Flash Next's PLE: bigrams and trigrams, every row read in its chunks (lookup.Kind's defaults)."""
        return Kind()

    def install(self, manager: "TraceManager") -> None:
        """Every hook, or (when one fails) none: a half-installed tracer would look installed to every later request
        and serve them without a step closer, their traces never filed. A failure is kept and raised again at once
        until the model loads again (a new tracer) or the settings change: a model metalscope cannot hook would
        otherwise retry the whole install, its lens load included, on every request. Only the reason is kept and a new
        exception raised each time (re-raising one object grows its traceback, and keeps every request's frames)."""
        with self.lock:
            if self.installed:
                return
            if self.install_error is not None:
                raise RuntimeError(self.install_error)
            try:
                self._install(manager)
            except Exception as exc:
                self._close_all()
                self.install_error = self._why(exc)
                log.exception("metalscope: %s; its requests are served untraced and steered ones refused until it "
                              "loads again or the settings change", self.install_error)
                raise RuntimeError(self.install_error) from None

    def _install(self, manager: "TraceManager") -> None:
        self.lens = self._load_lens()
        # Steering hooks go on first: hooks run in install order and each
        # sees the previous one's replacement, so the capture sessions
        # below record the steered residual (steer.py).
        # Each session is kept before it hooks in, so a failed install closes whatever hooks it had added.
        self.steer = SteerSession(manager.steer_failed)
        self.steer.install(self.model)
        failed = lambda exc: manager.capture_failed(self, exc)
        session = self.capture = CaptureSession(self._capture_config())
        session.on_error = failed
        session.set_rows([])
        self.adapter_name = install_adapter(session, self.model)
        self.hc_count, self.n_layers = session.hc_count, session.n_layers
        self._record_shape()
        # Last hook on the top-level module: every forward ends a step.
        closer = self.closer = CaptureSession(capture_config())
        closer.on_error = failed
        def close(_m, args, kwargs, _out):
            ids = args[0] if args else kwargs.get("inputs")
            tokens = starts = None
            if self.mtp and isinstance(ids, mx.array) and ids.ndim == 2:
                tokens = int(ids.shape[1])
                starts = window_starts(kwargs.get("cache"), tokens, int(ids.shape[0]))
            manager.close_step(self, starts, tokens)
        closer.hook(self.model, close)
        log.info("metalscope: hooks installed on %s (%s adapter%s)", self.name, self.adapter_name,
                 ", MTP decode" if self.mtp else "")

    def _record_shape(self) -> None:
        """The served model's shape on its checkpoint's record (models.check_shape: written the first time only), at
        each load's hook install: the dock's link offer needs it. One of another shape is logged and served as it is,
        the record left alone (ds4's proxy refuses to start on one; a chat model here stays up). Never raises."""
        from . import models
        from .adapters.generic import d_model, text_model
        try:
            models.check_shape(self.name, models.shape_of(self.n_layers, self.hc_count, d_model(text_model(self.model))))
        except models.ShapeMismatch as exc:
            log.warning("metalscope: %s; served as it is, its checkpoint's vectors and lens may not fit", exc)
        except Exception:
            log.exception("metalscope: could not record the shape of %s", self.name)

    def _load_lens(self) -> Optional["jlens.LensSetup"]:
        try:
            setup = jlens.setup_for(self.name)
        except Exception:
            log.exception("metalscope: could not load a Jacobian lens for %s", self.name)
            return None
        if setup is not None:
            log.info("metalscope: Jacobian lens for %s (%d layers)", self.name, len(setup.lens.layers))
        return setup

    def unembed(self, ids: list[int]):
        """The tokens' output-head rows [n, d] and the final norm's gain (or None), for a word's lens direction
        (vectors.from_lens; adapters.unembed: Inkling 2's own); Ds4Tracer reads them from the GGUF."""
        from .adapters import unembed
        return unembed(self.model, ids)

    # What the lens sets in a capture config (_capture_config), all swapped together by refresh_lens and use_lens_step.
    LENS_CONFIG_FIELDS = ("jlens", "jlens_topk", "jlens_layers", "logit_lens_layers", "logit_lens_topk")

    def refresh_lens(self) -> Optional["jlens.LensSetup"]:
        """Read the model's lens again (``/jlens use`` switched it: another J-lens, or the logit lens) and give it to
        the capture session from its next forward, no reload. The session's config is replaced whole, so a forward
        under way finishes on the old lens; a trace in flight has the old lens's records up to the switch. Before
        the hooks are installed this does nothing: ``install`` reads the lens."""
        if not self.installed:
            return self.lens
        fresh = self._load_lens()
        with self.lock:
            self.lens = fresh
            self._relens()
        return fresh

    def use_lens_step(self, step: int) -> None:
        """Read the lenses at ``step``'s layers (lens_steps) from the next forward, as refresh_lens swaps the lens."""
        if step == self.lens_step:
            return
        with self.lock:
            self.lens_step = step
            self._relens()

    def _relens(self) -> None:
        if self.capture is not None:
            cfg = self._capture_config()
            self.capture.config = dataclasses.replace(self.capture.config, **{k: getattr(cfg, k) for k in self.LENS_CONFIG_FIELDS})

    def lens_steps(self) -> list[tuple[list[int], list[int]]]:
        """The layers each lens step reads on this model (lens_steps)."""
        return lens_steps(self.lens, self.layer_count())

    def read_layers(self) -> tuple[list[int], list[int]]:
        """The layers the J-lens and the logit lens read at the step in use (none with the lenses off)."""
        return ([], []) if self.lens_step == config.LENS_OFF else self.lens_steps()[self.lens_step]

    def _capture_config(self):
        cfg = capture_config()
        jl, cfg.logit_lens_layers = self.read_layers()
        if self.lens is not None:
            # Both lenses: the J-lens beside the logit lens (its J = I case), so the dock can show either for any
            # reply; all the reads of a step share one LM-head call (capture._lens_reads).
            cfg.jlens, cfg.jlens_topk, cfg.jlens_layers = self.lens.lens.jacobians, 10, jl
        return cfg

    def uninstall(self) -> None:
        with self.lock:
            self._close_all()

    def _close_all(self) -> None:
        """Remove every hook (a failed install's too) and forget the sessions."""
        for s in (self.steer, self.capture, self.closer):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    log.exception("metalscope: closing a hook")
        self.steer = self.capture = self.closer = None

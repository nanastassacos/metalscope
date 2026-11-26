"""Traces and steering for a model ds4 serves, through the same TraceManager
the oMLX integration uses.

ds4 records each forward pass of a traced request into a file (frames.py);
``Run`` follows that file while the request is served and files every forward
as a step (``TraceManager.file_host_step``), so the live stream, the saved
trace and every route work as they do on oMLX. ``Ds4Tracer`` stands in for the
loaded-model object the manager, the live stream and the routes read.

What a ds4 trace holds, per prompt and reply token: each layer's routed
experts and their weights, each hyper-connection stream's norm and what the layer changed in it (the norm of its
output less its input, the steer's push included), and, for the rows the head
read (every reply token, MTP verify rows included, and a prompt chunk's last
token), the residual's norm as the head reads it, the logit lens, the J-lens
(Flash Next, when the model has one and ``/jlens on``), the next-token top
10. oMLX also reads the lenses on a prompt chunk's last 64 rows and the residual
norm on every row; ds4 leaves those rows empty (ids -1, NaN), as oMLX does
for rows it does not read.

A traced request reuses the prompt prefix ds4's session already read, as an
untraced one does, when it was read under the same steering: it records the
rows it feeds, from the first new prompt token on. Its trace holds the whole
prompt's tokens; the rows before ``meta.read_from`` are empty.

The J-lens is a file ds4 loads once (frames.write_jlens, ``jlens_file``), the checkpoint's (models.py): a ds4 model
linked to the checkpoint oMLX serves too (``metalscope model link``) reads the same vectors, profiles and J-lens.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import config, jlens, lookup
from ..live import MANAGER
from ..tracer import lens_steps
from . import scope_dir
from .frames import Rewind, ScopeReader, Step, write_jlens

log = logging.getLogger("metalscope")

# The routing rule each ds4 family's router follows (routing.py names), for the
# trace's meta.routing: what router_ids / router_probs mean.
ROUTING = {"qwen4": dict(rule="qwen3_5_moe", shared=1), "deepseek41": dict(rule="deepseek_v4", shared=1)}
# Each family's n-gram lookup, which the patch can leave out (scope lookup false; lookup.Kind). Qwen3.8's PLE:
# bigrams and trigrams, read in the prefill. V4.1's Engram: 2- to 4-grams at layers 1 and 14; its sweep reads no row's
# output head, so ds4 feeds the reply's rows one at a time (0.05-0.06 s a row in the probe of 2026-11-19, M5 Max, disk
# streaming; 0.07 with the head read on each), and under disk streaming a prompt under 256 rows, or a remainder under
# 1,024 after a prefill chunk, too (ds41_prefill_count); its conversation read about 0.01 s a token (35 s at 3,500).
LOOKUP = {"qwen4": lookup.Kind(), "deepseek41": lookup.Kind(ngram=4, row_s=0.07, alone=1023, pace=0.01)}
MAX_LENS = 64        # the layers ds4 reads each lens at, at most (the patch's DS4_SCOPE_MAX_LENS)
END_WAIT_S = 30.0     # how long a finished request's END record may take to appear
POLL_S = 0.02


class _Routing:
    def __init__(self, meta: dict):
        self._meta = meta
        self.num_experts = meta["num_experts"]

    def meta(self) -> dict:
        return dict(self._meta)


class Ds4Tracer:
    """What TraceManager, the live stream and the web routes read of a loaded
    model, for one model ds4 serves. No hooks to install: ds4 records. ``lens``
    is loaded as ModelTracer loads it (the J-lens the model's name claims)."""

    adapter_name = "ds4"
    installed = True
    model = None
    mtp = False
    steer = None
    capture = None        # ds4 records; what live._save reads of a capture session (routing) is here

    def __init__(self, name: str, tokenizer, family: str, n_layer: int, n_hc: int, n_embd: int,
                 k: int, n_expert: int, gguf: Optional[Path] = None):
        self.name, self.tokenizer, self.family, self.gguf = name, tokenizer, family, gguf
        self.n_layer, self.hc_count, self.n_embd = n_layer, n_hc, n_embd
        routing = dict(ROUTING.get(family, dict(rule=family, shared=0)), k=k, exact=True, num_experts=n_expert)
        self.n_layers, self.routing = n_layer, _Routing(routing)
        self.lock = threading.Lock()
        self.lens: Optional[jlens.LensSetup] = None
        self.lens_step = config.LENS_STEP_DEFAULT      # the step a request is sent to read the lenses at (read_layers)
        self._loaded = False

    @property
    def width(self) -> int:
        """A steering vector's width: the whole residual row (every stream)."""
        return self.hc_count * self.n_embd

    @property
    def lookup(self) -> bool:
        """The model's n-gram lookup (LOOKUP: Qwen3.8's PLE, V4.1's Engram), which the patch can leave out
        (lookup.py)."""
        return self.lookup_kind is not None

    @property
    def lookup_kind(self) -> Optional[lookup.Kind]:
        return LOOKUP.get(self.family)

    def install(self, manager) -> None:
        """Load the J-lens once (TraceManager.register calls this for every traced request)."""
        if self._loaded:
            return
        with self.lock:
            if self._loaded:
                return
            self.lens = self._load_lens()
            self._loaded = True

    def uninstall(self) -> None:
        pass

    def unembed(self, ids: list[int]):
        """The tokens' output-head rows from the GGUF and no final norm gain (Flash Next's head reads the mixer
        output directly, as jlens.norm_gain finds on the MLX model): ModelTracer.unembed's pair."""
        import mlx.core as mx
        from .vocab import gguf_rows
        if self.gguf is None:
            raise ValueError(f"{self.name} has no GGUF to read the output head from")
        return mx.array(gguf_rows(self.gguf, "output.weight", ids)), None

    def _load_lens(self) -> Optional[jlens.LensSetup]:
        """The J-lens the model's name claims (jlens.find_for), when it is as wide as the model. Qwen3.8 only: ds4 reads
        no J-lens on V4.1."""
        if self.family != "qwen4" or self.gguf is None:
            return None
        path = jlens.find_for(self.name)
        if path is None:
            return None
        try:
            lens = jlens.load(path)
            shape = tuple(next(iter(lens.jacobians.values())).shape)
            if shape != (self.n_embd, self.width):
                log.warning("metalscope: %s's J-lens %s is %s, not (%d, %d): not read", self.name, path.name, shape,
                            self.n_embd, self.width)
                return None
            setup = jlens.LensSetup(lens, name=path.name)
        except Exception:
            log.exception("metalscope: could not load a Jacobian lens for %s", self.name)
            return None
        log.info("metalscope: Jacobian lens for %s: %s (%d layers)", self.name, path.name, len(lens.layers))
        return setup

    def use_lens_step(self, step: int) -> None:
        """Send the next requests to read the lenses at ``step``'s layers (ModelTracer.use_lens_step)."""
        self.lens_step = step

    def lens_steps(self) -> list[tuple[list[int], list[int]]]:
        """The layers each lens step reads on this model (ModelTracer.lens_steps)."""
        return lens_steps(self.lens, self.n_layer)

    def read_layers(self) -> tuple[list[int], list[int]]:
        """The layers the J-lens and the logit lens read at the step in use: what a request asks ds4 for, the deepest
        MAX_LENS of each (ds4 refuses more; the sizes read at most 25 today, config.LENS_LARGE_MAX and the last: kept
        should they grow); none with the lenses off (ds4 reads no lens rows, jlens_file None)."""
        if self.lens_step == config.LENS_OFF:
            return [], []
        jl, logit = self.lens_steps()[self.lens_step]
        return jl[-MAX_LENS:], logit[-MAX_LENS:]

    def refresh_lens(self):
        """Read the model's lens again (/jlens use switched it); the next traced request reads the new one."""
        fresh = self._load_lens()
        with self.lock:
            self.lens = fresh
            self._loaded = True
        return fresh

    # -- what a request sends ds4 ------------------------------------------------
    def jlens_file(self) -> Optional[Path]:
        """The model's J-lens as ds4 reads it (frames.write_jlens): the layers the trace reads (read_layers), written
        once per lens and layer set in ds4-server's scope directory (Flash Next: ~53 MB a layer, ~0.8 GB for 15; at most
        one a lens step; make_served removes the others); None without a J-lens."""
        with self.lock:                  # the lens and its layers as one (refresh_lens swaps the lens)
            setup, layers = self.lens, self.read_layers()[0]
            if setup is None or not layers:
                return None
            src = jlens.find_for(self.name)
            stamp = (src / "lens.safetensors").stat().st_mtime if src is not None else 0
            key = hashlib.sha1(f"{setup.name}:{stamp}:{layers}".encode()).hexdigest()[:12]
            out = scope_dir() / f"jlens-{setup.name}-{key}.bin"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                # The MLX arrays as they are: write_jlens converts and writes one layer at a time (47 at once: ~2.5 GB).
                write_jlens(out, {l: setup.lens.jacobians[l] for l in layers}, self.width, self.n_embd)
                log.info("metalscope: wrote %s's J-lens for ds4 (%s)", self.name, out)
        return out


@dataclass
class Run:
    """One traced request: follows ds4's capture file and files each forward."""
    request_id: str
    tracer: Ds4Tracer
    path: Path
    manager: Any = None   # the TraceManager (default live.MANAGER)
    reader: Optional[ScopeReader] = None
    next_pos: int = 0
    step: int = 0
    prompt_noted: bool = False

    def __post_init__(self):
        self.manager = self.manager or MANAGER
        self.reader = ScopeReader(self.path)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._failed: Optional[str] = None      # reading the capture raised (why): nothing more is read

    # -- following the file ---------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._follow, name=f"ds4-trace-{self.request_id[:8]}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """The request never reached ds4-server: stop following its file (nothing is saved)."""
        self._stop.set()

    def _follow(self) -> None:
        while not self._stop.is_set():
            self.pump()
            if self.reader.end is not None:
                return
            self._stop.wait(POLL_S)

    def finish(self, served: bool) -> None:
        """The HTTP exchange is over: wait for ds4's END record (it follows the
        response), file what is left and save the trace."""
        # A cancelled request (the client left) still gets ds4's END record once the server notices.
        deadline = time.time() + (END_WAIT_S if served else 10.0)
        while self.reader.end is None and self._failed is None and time.time() < deadline:
            if self._thread is None or not self._thread.is_alive():
                self.pump()     # no follower (never started, or it returned): read here
            time.sleep(POLL_S)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.pump()
        prompt = self.reader.prompt or []
        sampled = self.reader.end.sampled if self.reader.end is not None else []
        rt = self.manager.requests.get(self.request_id)
        end = self.reader.end
        if end is None or end.stopped is not None:
            # Saved as partial, with why: its last forwards (and without an end record the reply's tokens) are missing.
            why = (f"reading ds4's capture failed: {self._failed}" if self._failed else
                   "ds4 wrote no end record" if end is None else f"ds4's trace stopped: {end.stopped}")
            log.warning("metalscope: %s for %s; saving what it recorded as partial", why, self.request_id[:8])
            if rt is not None and why not in rt.partial:
                rt.partial.append(why)
        if rt is not None and self.reader.inserted:
            # ds4 put text in the reply that the model did not sample (a tool error after an invalid tool call): the
            # reply holds it, as fed; these are its (position, tokens).
            rt.meta["inserted"] = [list(map(int, x)) for x in self.reader.inserted]
        self.manager.finish(self.request_id, prompt, sampled, self.tracer)

    def pump(self) -> None:
        if self._failed is not None:      # read once to its error (logged then), not again
            return
        try:
            for st in self.reader.poll():
                self._file(st)
        except Exception as exc:
            log.exception("metalscope: reading ds4's capture for %s failed", self.request_id[:8])
            self._failed = f"{type(exc).__name__}: {exc}"[:200]
            self._stop.set()

    # -- one forward ----------------------------------------------------------
    def _file(self, st) -> None:
        if isinstance(st, Rewind):        # rows from st.pos on are fed again: those replace them
            self.next_pos = min(self.next_pos, st.pos)
            return
        P = len(self.reader.prompt) if self.reader.prompt is not None else 0
        if P and not self.prompt_noted:
            self.manager.note_prompt(self.request_id, P)
            self.prompt_noted = True
        # A forward replayed over positions already filed: keep only its new rows.
        a = max(0, self.next_pos - st.pos0)
        if a >= st.T:
            return
        # A step of its own for the prompt's last token, and for every row of a
        # forward the head read whole (a decode step, an MTP verify's kept rows:
        # reply tokens one after another): the live stream takes a multi-row step
        # as prefill (LiveStreamer._process). Other forwards (prompt chunks, a
        # suffix ds4 appends mid-reply) stay one step.
        if P and st.pos0 <= P - 1 < st.pos0 + st.T:
            own = P - 1 - st.pos0
        else:
            own = 0 if st.R == st.T else st.T
        cuts = sorted({a, *range(max(own, a), st.T), st.T})
        for lo, hi in zip(cuts, cuts[1:]):
            # filed by position: after a rewind, the rows fed again overwrite the ones they replace
            self.manager.file_host_step(self.tracer, self.request_id, self.step, self._records(st, lo, hi),
                                        pos=st.pos0 + lo)
            self.step += 1
        self.next_pos = st.pos0 + st.T

    def _records(self, st: Step, lo: int, hi: int) -> list:
        """Rows lo..hi of a forward as close_step's records: a batch axis of one,
        per-layer kinds stacked [L, 1, T, ...]; rows the head did not read (all
        but the forward's last R) empty."""
        T, L, h = hi - lo, self.tracer.n_layer, self.reader.header
        layers = list(range(L))
        K = h.topk
        # Head row r is forward row st.T - st.R + r; the ones in this slice, as (slice row, r).
        read = [(i - lo, i - (st.T - st.R)) for i in range(max(lo, st.T - st.R), hi)]

        def tail(fill, shape, dtype, value):
            a = np.full((1, T, *shape), fill, dtype=dtype)
            if value is not None:
                for t, r in read:
                    a[0, t] = value[r]
            return a
        recs = [
            ("input_ids", -1, st.tokens[None, lo:hi].astype(np.int32), None),
            ("router_ids", -1, st.experts[:, None, lo:hi], layers),
            ("router_probs", -1, st.weights[:, None, lo:hi], layers),
            ("hc_stream_norm", -1, st.stream_norm[:, None, lo:hi], layers),
            ("resid_write_norm", -1, st.write_norm[:, None, lo:hi], layers),
        ]
        resid = np.full((L, 1, T), np.nan, dtype=np.float32)
        if st.resid_norm is not None:
            for t, r in read:
                resid[:, 0, t] = st.resid_norm[r]
        recs.append(("resid_norm", -1, resid, layers))
        recs += [
            ("logit_ids", -1, tail(-1, (K,), np.int32, st.out_ids), None),
            ("logit_logp", -1, tail(np.nan, (K,), np.float32, st.out_logp), None),
            ("logit_entropy", -1, tail(np.nan, (), np.float32, st.out_entropy), None),
        ]
        for j, layer in enumerate(h.lens):
            has = st.lens_ids is not None
            recs += [
                ("lens_ids", layer, tail(-1, (K,), np.int32, st.lens_ids[:, j] if has else None), None),
                ("lens_logp", layer, tail(-np.inf, (K,), np.float32, st.lens_logp[:, j] if has else None), None),
            ]
        if self.reader.jlens:
            for j, layer in enumerate(self.reader.jlens):
                has = st.jlens_ids is not None
                recs += [
                    ("jlens_ids", layer, tail(-1, (K,), np.int32, st.jlens_ids[:, j] if has else None), None),
                    ("jlens_logp", layer, tail(-np.inf, (K,), np.float32, st.jlens_logp[:, j] if has else None), None),
                ]
        return recs


def runs_dir() -> Path:
    """The proxy's request files (capture, steering), in ds4-server's scope directory."""
    d = scope_dir() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d

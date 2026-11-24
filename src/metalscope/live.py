"""Live tracing inside a running server: which requests are traced, which
batch rows they occupy, and turning per-step records into saved traces.

This module knows nothing about oMLX internals; omlx/patches.py calls into it
from the right places. The flow for one traced request:

    register(rid, level, meta)       API layer, before the request is scheduled
                                     (level None: steered but not traced)
    rows_for(tracer, [rid, ...])     before each forward: which rows to record,
                                     and which rows to steer (steer.py)
    close_step(tracer)               after each forward: async_eval captures,
                                     hand records to their requests, the last
                                     step's to host memory
    finish(rid, prompt, output)      request done: assemble + save (background)

Steered requests (steer.py) share this registry so every scheduler wrapper
that gates on "is this request tracked" covers them; their KV is kept out of
the prefix cache (fetch and store) for as long as the id is known here or in
the finished-ids tombstone.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from . import config, models
from .edit import Recorded
from . import retention, tally
from .livestream import LIVE_KINDS, LiveHub, LiveStreamer, add_ref, token_texts
from .steer import TOMBSTONES, SteerPlan, SteerSpec, build_plan, check_words, norm_table
from .store import Trace, append_index, index_entry
from .tracer import ModelTracer

log = logging.getLogger("metalscope")
STALE_SECONDS = 1800
MIN_TIMED = 8   # a reply's decode speed needs this many tokens timed (decode_speed)
# A relayed piece of a streamed reply with reply text in it (note_chunk): OpenAI chat's content or reasoning, Anthropic's
# text, thinking or tool input deltas, Responses' delta; not the stream's opening events (empty text, no text key).
REPLY_TEXT = re.compile(rb'"(?:content|reasoning_content|text|thinking|partial_json|delta)":"[^"]')
# The read-outs: the output's top k and both lenses'. The live stream sends them (livestream.LIVE_KINDS); a trace does
# not keep them.
READOUT_KINDS = frozenset(("logit_ids", "logit_logp", "logit_entropy", "lens_ids", "lens_logp", "jlens_ids", "jlens_logp"))
# One trace keeps at most this many bytes of records (TraceManager._over_cap); past it the reply goes on unrecorded,
# saved as meta.capped. Flash Next keeps 5,572 bytes a token: its whole 262k context (1.46 GB) fits, about 269k
# tokens do; DeepSeek V4.1 on ds4 keeps 3,616 bytes a token and stops near 415k of its 1M. Small models keep far less
# a token.
TRACE_MAX_BYTES = 1_500_000_000


@dataclass
class RequestTrace:
    request_id: str
    level: Optional[str]     # "on" (traced), or None: steered only, nothing recorded
    meta: dict
    # Steering (steer.py): the header's spec, the plan built from it once the
    # model is known, and why it stopped if it did (saved with the trace).
    steer_spec: Optional[SteerSpec] = None
    steer: Optional[SteerPlan] = None
    steer_failed: Optional[str] = None
    steer_partial: bool = False
    # Capture hooks that raised while recording this request (TraceManager.capture_failed): the trace misses their
    # records for those forwards; saved as meta.partial.
    partial: list = field(default_factory=list)
    started: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    # "<kind>/<layer>" -> [(step, batch value, row index, stacked layers or None)]: numpy from the step after its own
    # (TraceManager._host), an MLX array until then
    parts: dict = field(default_factory=lambda: defaultdict(list))
    # The records' bytes kept so far; past TRACE_MAX_BYTES (_over_cap) none are kept, only the positions fed after
    # (``dropped``) or, filed by position, the first position not kept (``cap_at``).
    kept_bytes: int = 0
    capped: bool = False
    dropped: int = 0
    cap_at: Optional[int] = None
    # Tokens fed per forward pass (prefill chunks, then 1 per decode step).
    step_tokens: list = field(default_factory=list)
    live: Any = None  # livestream.LiveState
    # MTP decode only (mtp.py): step -> absolute position of its first token, and
    # the end of the last recorded forward (fallback when a forward has no cache).
    step_pos: dict = field(default_factory=dict)
    pos_end: int = 0
    # Prompt length, when the scheduler tells us (note_prompt): under MTP a
    # short prompt tail can arrive as a forward with logits, like a window.
    prompt_len: int = 0
    # The request's stop tokens (note_prompt): the scheduler may run one more forward
    # on the one it sampled, but the reply ends before it.
    stop_ids: frozenset = frozenset()
    # The reply's decode speed (TraceManager.decode_speed), the prompt's reading left out. oMLX: decode forwards filed
    # (_decoded) and (time, position fed) at the second and the newest. ds4: when the proxy relayed the first and the
    # last piece of reply text (note_chunk).
    n_dec: int = 0
    dec0: Optional[tuple] = None
    dec1: Optional[tuple] = None
    chunks: Optional[tuple] = None
    # An edit's completion (edit.py, oMLX): (the reply's prompt, its tokens before this request's own). Its prompt is
    # the two joined; it is kept as that prompt and reply, then its own tokens.
    base: Optional[tuple] = None


class TraceManager:
    def __init__(self):
        self.tracers: dict[int, ModelTracer] = {}
        self.requests: dict[str, RequestTrace] = {}
        self.lock = threading.RLock()
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metalscope-writer")
        # The traces' size budget (retention.py): checked after each save from a thread of its own.
        self.keeper = retention.Keeper()
        # The expert tally (tally.py): each saved chat counted from a thread of its own while no request runs.
        self.tally = tally.UPDATER = tally.Updater(busy=lambda: bool(self.requests))
        self._settings = config.load_settings()
        # Cache key: (settings path, mtime). The path matters: a process whose
        # trace home changes (tests, METALSCOPE_HOME) must not keep the old
        # file's values when the new home has no settings file yet.
        self._settings_key = (str(config.settings_path()), self._mtime())
        self.saved = 0
        self.hub = LiveHub()
        self.streamer = LiveStreamer(self.hub)
        # CPU time spent in our per-step bookkeeping (rows_for + close_step).
        self.overhead_s = 0.0
        self._host_failed = False   # _host logged a failure (once)
        self.traced_steps = 0
        # close_step's parts (seconds, summed over traced steps): building the capture graph, submitting it, filing
        # the records.
        self.phase_s: dict[str, float] = defaultdict(float)
        # Steering: finished steered ids (the async cache-store worker runs after
        # cleanup, so the KV guard must outlive the request), and each served
        # model's own strength table (steer.norm_table), kept until its norms.json changes (strength_table).
        self._tombstones: deque = deque(maxlen=TOMBSTONES)
        self._tomb_set: set = set()
        self.norm_tables: dict[str, dict] = {}
        self._norm_stamps: dict[str, Optional[int]] = {}   # norms.json's mtime when each table was read
        # The newest traced requests' token ids, for an edit at one token (edit.py): each engine puts them as it
        # learns them (ds4/proxy.py, omlx/patches.py).
        self.kept = Recorded()

    # -- settings ----------------------------------------------------------
    def steer_enabled(self) -> bool:
        self.mode()  # refresh settings
        return bool(self._settings.get("steer", False))

    def lens_step(self) -> int:
        """The settings' lens_step (lens_steps): the layers the next traced request reads the lenses at."""
        self.mode()
        k = self._settings.get("lens_step")
        return k if config.lens_step_ok(k) else config.LENS_STEP_DEFAULT

    @staticmethod
    def _mtime() -> float:
        try:
            return config.settings_path().stat().st_mtime
        except OSError:
            return 0.0

    def mode(self) -> str:
        key = (str(config.settings_path()), self._mtime())
        if key != self._settings_key:
            self._settings, self._settings_key = config.load_settings(), key
            for t in self.loaded():       # settings changed: a failed install is tried once more
                if getattr(t, "install_error", None) is not None:
                    t.install_error = None
                if self._settings["mode"] != "off":
                    self._try_install(t)
        return self._settings["mode"]

    def _try_install(self, tracer) -> None:
        """Install where no request waits on it (a mode switch, a model attaching): every request reads mode(), so
        a tracer that cannot be hooked must not fail them (install logged why; its requests then decide)."""
        try:
            tracer.install(self)
        except Exception as exc:
            log.warning("metalscope: hooks not installed on %s: %s", tracer.name, exc)

    def set_mode(self, mode: str) -> dict:
        data = config.save_settings(mode=mode)
        self.mode()
        return data

    # -- model lifecycle ---------------------------------------------------
    def attach(self, model, name: str, tokenizer) -> ModelTracer:
        tracer = ModelTracer(model, name, tokenizer)
        self.tracers[id(model)] = tracer
        if self.mode() != "off":
            self._try_install(tracer)
        return tracer

    def detach(self, model) -> None:
        """A model unloading: its hooks come off and the requests it still held are let go (oMLX aborts under memory
        pressure and unloads at once, so no scheduler abort ever releases them: they would count as active)."""
        tracer = self.tracers.pop(id(model), None)
        if tracer is not None:
            tracer.uninstall()
            name = config.short_name(tracer.name)
            for rid in [r for r, rt in list(self.requests.items()) if rt.meta.get("model") == name]:
                self.release(rid)

    def tracer_for(self, model) -> Optional[ModelTracer]:
        return self.tracers.get(id(model))

    def loaded(self) -> list[ModelTracer]:
        """The loaded models' tracers, as a list: iterate this, not ``tracers``, which a model loading or unloading
        changes from another thread."""
        return list(self.tracers.values())

    # -- requests ----------------------------------------------------------
    def resolve_level(self, header: Optional[str]) -> Optional[str]:
        """Header overrides the global mode; returns None for untraced. A
        header's off (0, false, no) stays off; any other value traces."""
        mode = self.mode()
        if header:
            return None if header.strip().lower() in ("0", "off", "false", "no") else "on"
        return None if mode == "off" else mode

    def register(self, request_id: str, level: Optional[str], meta: dict, model=None,
                 steer: Optional[SteerSpec] = None) -> None:
        """Track a request. The hooks and the steering plan come first: when installing raises, or the plan cannot be
        built (the middleware's check_spec refuses most such steers before they get here), nothing is left behind (no
        entry, no live "start" without an end) and the caller decides (oMLX refuses a steered request). The live
        feed's announcement cannot fail the request. Its meta's model is the tracer's (the served copy's name, not an
        alias the client sent)."""
        self._sweep()
        rt = RequestTrace(request_id, level, meta, steer_spec=steer)
        # A header-traced (or steered) request while mode is off installs hooks on demand.
        tracers = [self.tracer_for(model)] if model is not None else self.loaded()
        for t in tracers:
            if t is not None:
                t.install(self)
                if steer is not None:
                    self._ensure_plan(rt, t)
                    if rt.steer_failed is not None:
                        raise RuntimeError(f"the steer could not be built: {rt.steer_failed.removeprefix('plan: ')}")
        if model is not None and tracers[0] is not None:
            rt.meta["model"] = config.short_name(tracers[0].name)
        if level is not None:
            # The lens step takes effect when no traced request is in flight (theirs keep the layers they started
            # with: a trace reads one set); the step this one reads at is saved with it.
            with self.lock:
                idle = not any(r.level for r in self.requests.values())
            for t in tracers:
                if t is not None:
                    if idle:
                        t.use_lens_step(self.lens_step())
                    rt.meta["lens_step"] = t.lens_step
            try:
                self.streamer.on_register(rt)
            except Exception:
                log.exception("metalscope: the live feed could not announce %s", request_id[:8])
        with self.lock:
            self.requests[request_id] = rt

    def prepare(self, model: Optional[str]) -> None:
        """Install the hooks of the loaded models served as ``model`` (register's slow part: a J-lens load), for the
        middleware to run off the event loop before the engine registers the request. A failure is left for register
        to raise, with the request's own handling."""
        short = config.short_name(model)
        for t in self.loaded():
            if config.short_name(t.name) == short:
                try:
                    t.install(self)
                except Exception:
                    pass

    def is_traced(self, request_id: Optional[str]) -> bool:
        """Tracked: traced, steered or both (every scheduler wrapper gates on this)."""
        return request_id in self.requests

    # -- steering ------------------------------------------------------------
    def kv_isolated(self, request_id: Optional[str]) -> bool:
        """Steered requests' KV must neither be stored for others nor reused:
        true while the request is live and after it finished (tombstone)."""
        rt = self.requests.get(request_id)
        if rt is not None and rt.steer_spec is not None:
            return True
        return request_id in self._tomb_set

    def _tombstone(self, request_id: str) -> None:
        if len(self._tombstones) == self._tombstones.maxlen:
            self._tomb_set.discard(self._tombstones[0])
        self._tombstones.append(request_id)
        self._tomb_set.add(request_id)

    def strength_table(self, name: str) -> dict:
        """The served copy's strength table (steer.norm_table), the one the steering path and the routes use: read once
        and kept while its norms.json is unchanged, read again when it changes (``metalscope vector norms NAME
        --refresh``), so the plan agrees with the middleware's check_spec, which reads the file. A table set without
        a stamp (tests stub one) is kept. Raises ValueError as norm_table does."""
        path = models.norms_path(name)

        def stamp() -> Optional[int]:
            try:
                return path.stat().st_mtime_ns
            except OSError:
                return None
        table = self.norm_tables.get(name)
        if table is not None:
            if name not in self._norm_stamps:
                return table
            now = stamp()
            if now is None or now == self._norm_stamps[name]:      # unchanged, or gone: the table read stands
                return table
        table = norm_table(name)
        self.norm_tables[name], self._norm_stamps[name] = table, stamp()
        return table

    def _ensure_plan(self, rt: RequestTrace, tracer: ModelTracer) -> None:
        """Build the request's steering plan once the model is known. A word's vector steers only as an edit's
        one-token steer: any other steer naming one fails closed (steer.check_words, the middleware's 400 too). The lens
        Jacobians a plan reads its readout norms from are cut to the columns a steer writes (``lens_columns``); a lens
        of another layout fails an edit's plan (its word vectors need the lens) and leaves any other without readout
        norms (meta only)."""
        if rt.steer is not None or rt.steer_spec is None or rt.steer_failed is not None:
            return
        try:
            spec = rt.steer_spec
            check_words(spec, tracer.name)
            # A removal alone has no dose to size: no strength table needed.
            table = self.strength_table(tracer.name) if any(not p.remove for p in spec.parts) else {"layers": {}}
            jac = tracer.lens.lens.jacobians if tracer.lens is not None else None
            cut = getattr(tracer, "lens_columns", None)
            if jac is not None and cut is not None:
                try:
                    jac = {l: J[:, cut(l, J.shape[1])] for l, J in jac.items()}   # lazy slices
                except ValueError:
                    if spec.edit:
                        raise
                    jac = None
            rt.steer = build_plan(rt.steer_spec, tracer.name, rt.request_id, tracer.hc_count, jac, table)
            log.info("metalscope: steering %s with %s at layers %s (|v| %s)", rt.request_id[:8], rt.steer.meta["label"],
                     rt.steer.meta["layers"], {l: round(v, 1) for l, v in rt.steer.meta["abs_norm"].items()})
        except Exception as exc:
            rt.steer_failed = f"plan: {exc}"
            log.warning("metalscope: request %s not steered: %s", rt.request_id[:8], exc)

    def steer_failed(self, request_id: str, reason: str) -> None:
        """From the steer hook: the vector could not be applied; the request is
        served unsteered from here and saved as failed (fail closed)."""
        rt = self.requests.get(request_id)
        if rt is not None and rt.steer_failed is None:
            rt.steer_failed = reason
            log.warning("metalscope: steering of %s stopped: %s", request_id[:8], reason)

    def capture_failed(self, tracer: ModelTracer, exc: BaseException) -> None:
        """A capture hook raised in a forward (hooks.py carries on without it): every request the tracer was recording
        is marked partial, with why (a few distinct reasons kept)."""
        reason = f"{type(exc).__name__}: {exc}"[:200]
        session = tracer.capture
        for rid in (session.owners if session is not None else None) or ():
            self.mark_partial(rid, reason)

    def mark_partial(self, request_id: Optional[str], reason: str) -> None:
        """Records of a traced request are missing (saved as meta.partial, with why: a few distinct reasons kept)."""
        rt = self.requests.get(request_id)
        if rt is not None and reason not in rt.partial and len(rt.partial) < 8:
            rt.partial.append(reason)

    def steer_partial(self, request_id: Optional[str]) -> None:
        """A forward ran outside the wrapped paths (e.g. an MTP reconcile
        re-prefill): the request was not steered on every token."""
        rt = self.requests.get(request_id)
        if rt is not None and rt.steer_spec is not None:
            rt.steer_partial = True

    def note_prompt(self, request_id: Optional[str], n: int, stop_ids=()) -> None:
        rt = self.requests.get(request_id)
        if rt is not None and not rt.prompt_len:
            rt.prompt_len = int(n)
        if rt is not None and stop_ids:
            rt.stop_ids = frozenset(int(t) for t in stop_ids)

    def rows_for(self, tracer: Optional[ModelTracer], row_request_ids: list, phase: str = "decode") -> None:
        """Before a forward: select the traced rows and the steered
        rows. ``phase`` is "prefill" for prompt forwards and "decode" for
        decode steps and MTP verify windows (steer.py's ``positions``)."""
        if tracer is None or not tracer.installed:
            return
        t0 = time.perf_counter()
        tracer.phase = phase
        try:
            # Steering first and on its own: whatever fails in it or in the recording rows, a steered request is
            # either steered or marked failed (fail closed), never served unsteered without a word.
            if tracer.steer is not None:
                self._steer_rows(tracer, row_request_ids, phase)
            capture = tracer.capture
            if capture is not None:
                rows = [i for i, rid in enumerate(row_request_ids) if (rt := self.requests.get(rid)) is not None and rt.level]
                capture.set_rows(rows, [row_request_ids[i] for i in rows])
        finally:
            self.overhead_s += time.perf_counter() - t0

    def _steer_rows(self, tracer: ModelTracer, row_request_ids: list, phase: str) -> None:
        """The steer hook's rows for this forward. When building them fails, nothing is steered this forward and
        every steered request among the rows is marked ``steer_failed`` (saved with its trace, as a hook failure is)."""
        try:
            steered, now = [], time.time()
            for i, rid in enumerate(row_request_ids):
                rt = self.requests.get(rid)
                if rt is not None and rt.steer_spec is not None:
                    rt.touched = now          # running: a steered request records nothing to refresh it (_sweep)
                    self._ensure_plan(rt, tracer)
                    if rt.steer is not None and rt.steer_failed is None:
                        steered.append((i, rid, rt.steer))
            tracer.steer.set_rows(steered, len(row_request_ids), phase)
        except Exception as exc:
            log.exception("metalscope: the steering rows could not be built")
            tracer.steer.clear()
            for rid in row_request_ids:
                rt = self.requests.get(rid)
                if rt is not None and rt.steer_spec is not None:
                    self.steer_failed(rid, f"rows: {exc}")

    def clear_rows(self, tracer: Optional[ModelTracer]) -> None:
        if tracer is None:
            return
        if tracer.capture is not None:
            tracer.capture.set_rows([])
        if tracer.steer is not None:
            tracer.steer.clear()

    def close_step(self, tracer: ModelTracer, starts: Optional[list] = None, tokens: Optional[int] = None) -> None:
        """After a forward: schedule captures (no sync) and file them by request.

        ``tokens`` (MTP decode only): tokens per row in this forward, and
        ``starts`` each batch row's absolute first position; the step is filed
        under that position so rejected drafts are overwritten (mtp.py)."""
        t0 = time.perf_counter()
        session = tracer.capture
        if session is None:
            return
        arrays = session.pending_arrays()
        t_graph = time.perf_counter()
        if not arrays:
            return
        mx.async_eval(*arrays)
        last, tracer.fresh = tracer.fresh, []      # the last step's records, read once this one is filed (_host)
        try:
            t_submit = time.perf_counter()
            self.phase_s["graph"] += t_graph - t0
            self.phase_s["submit"] += t_submit - t_graph
            now = time.time()
            step_refs = defaultdict(dict) if self.hub.active() else None
            step = session.step
            if tokens is not None:
                self._file_positions(session, starts, tokens, step_refs)
            session.end_step()
            # Taken off the session at once: a filing that raises must not leave them to be filed again next step.
            records, session.records = session.records, []
            # Count tokens fed this step from its input ids (or, an adapter without them, its first per-layer record).
            first = next((r for r in records if r.kind == "input_ids"), None) or \
                next((r for r in records if r.kind == "resid_norm"), None)
            fed = int(first.value.shape[-1]) if first is not None else tokens or 0      # [.., B, T]
            for rid in (first.owners or ()) if first is not None else ():
                if (rt := self.requests.get(rid)) is not None:
                    rt.step_tokens.append(fed)
                    if tracer.phase == "decode":      # a token a forward, or under MTP to the window's end (pos_end)
                        self._decoded(rt, rt.pos_end if tokens is not None else rt.n_dec)
            size: dict = defaultdict(int)       # each request's bytes of this step's kept records (its row of each)
            for rec in records:
                if rec.kind not in READOUT_KINDS:
                    for rid in rec.owners or ():
                        size[rid] += rec.value.nbytes // rec.value.shape[0 if rec.layers is None else 1]
            over = {rid for rid, n in size.items()
                    if (rt := self.requests.get(rid)) is not None and self._over_cap(rt, n, step, fed)}
            for rec in records:
                layers = _shared(rec.layers)
                for j, rid in enumerate(rec.owners or ()):
                    rt = self.requests.get(rid)
                    if rt is not None:
                        if rec.kind not in READOUT_KINDS and rid not in over:
                            parts = rt.parts[f"{rec.kind}/{rec.layer}"]
                            parts.append((rec.step, rec.value, j, layers))
                            tracer.fresh.append((parts, len(parts) - 1))
                        rt.touched = now
                        if step_refs is not None and rec.kind in LIVE_KINDS:
                            add_ref(step_refs[rid], rec.kind, rec.layer, rec.layers, rec.value, j)
        finally:
            self._host(last)        # this step is submitted (and filed): the GPU stays busy while the last one is read
        if step_refs:
            for rid, refs in step_refs.items():
                rt = self.requests.get(rid)
                if rt is not None:
                    self.streamer.on_step(rt, refs, tracer)
        self.traced_steps += 1
        t_end = time.perf_counter()
        self.phase_s["file"] += t_end - t_submit
        self.overhead_s += t_end - t0

    def _host(self, fresh: list) -> None:
        """A step's records (``fresh``: (a request's parts list, index)), copied to host memory (numpy) in place: the MLX
        arrays a traced reply keeps alive stay one step's worth however long it runs (MLX fails every allocation past
        499,000 buffers). Called once the next step is submitted and filed, or at a finish, so it waits only on work
        already done or under way. A record it cannot copy stays an MLX array (saved all the same); logged once.
        One a release emptied meanwhile is skipped."""
        done: dict = {}
        for parts, i in fresh:
            try:
                step, value, j, layers = parts[i]
                a = done.get(id(value))
                if a is None:
                    a = done[id(value)] = np.array(value)
                parts[i] = (step, a, j, layers)
            except IndexError:          # a release emptied the list meanwhile
                continue
            except Exception:
                if not self._host_failed:
                    self._host_failed = True
                    log.exception("metalscope: a step's records could not be copied to host memory; kept as MLX arrays")

    def _over_cap(self, rt: RequestTrace, nbytes: int, step: int, fed: int) -> bool:
        """Whether a step's records of ``rt`` (``nbytes`` of them, ``fed`` positions) go past TRACE_MAX_BYTES and are not
        kept. From the first such step on, only the positions fed are counted (``dropped``), or, filed by position, the
        first position not kept (``cap_at``); the save says whether the reply's own tokens were cut (_save)."""
        if not rt.capped:
            if rt.kept_bytes + nbytes <= TRACE_MAX_BYTES:
                rt.kept_bytes += nbytes
                return False
            rt.capped = True
        rt.dropped += fed
        pos = rt.step_pos.pop(step, None)
        if pos is not None:
            rt.cap_at = pos if rt.cap_at is None else min(rt.cap_at, pos)
        return True

    def file_host_step(self, tracer, request_id: str, step: int, records: list, pos: Optional[int] = None) -> None:
        """A forward recorded outside MLX (the ds4 backend, ds4/backend.py): file
        it as ``close_step`` files a capture session's records. ``records``:
        [(kind, layer, value, layers)], each value a numpy array with a batch
        axis of one ([1, T, ...]), or [L, 1, T, ...] with ``layers`` the stacked
        layers; ``input_ids`` is one of them. ``pos``: the step's first position;
        given, the trace is assembled by position (a later step overwrites an
        earlier one there, as for MTP decode: assemble)."""
        rt = self.requests.get(request_id)
        if rt is None:
            return
        if pos is not None:
            rt.step_pos[step] = int(pos)
        t0 = time.perf_counter()
        refs = {} if self.hub.active() else None
        if refs is not None and pos is not None:
            refs["pos"] = int(pos)        # the live stream's position for the step (a request may start past 0)
        over = self._over_cap(rt, sum(r[2].nbytes for r in records if r[0] not in READOUT_KINDS), step,
                              next((int(r[2].shape[-1]) for r in records if r[0] == "input_ids"), 0))
        for kind, layer, value, layers in records:
            if kind not in READOUT_KINDS and not over:
                rt.parts[f"{kind}/{layer}"].append((step, value, 0, _shared(layers)))
            if kind == "input_ids":
                rt.step_tokens.append(int(value.shape[-1]))
            if refs is not None and kind in LIVE_KINDS:
                add_ref(refs, kind, layer, layers, value, 0)
        rt.touched = time.time()
        if refs:
            self.streamer.on_step(rt, refs, tracer)
        self.traced_steps += 1
        self.overhead_s += time.perf_counter() - t0

    @staticmethod
    def _decoded(rt: RequestTrace, pos: int) -> None:
        """A decode forward of ``rt`` filed now, having fed up to position ``pos``: its decode speed runs from the
        second (the first may have waited on the prompt's last chunk) to the newest."""
        rt.n_dec += 1
        rt.dec1 = (time.perf_counter(), pos)
        if rt.n_dec == 2:
            rt.dec0 = rt.dec1

    def note_chunk(self, request_id: str, chunk: bytes) -> None:
        """A piece of a streamed reply the ds4 proxy relayed: one with reply text times the reply's decode, from the
        first such piece to the last (decode_speed). ds4's capture rows reach us in bursts, its chunks as decoded."""
        rt = self.requests.get(request_id)
        if rt is not None and REPLY_TEXT.search(chunk):
            now = time.perf_counter()
            rt.chunks = (rt.chunks[0] if rt.chunks else now, now)

    @staticmethod
    def decode_speed(rt: RequestTrace, n_out: int) -> Optional[float]:
        """Tokens a second while reply ``rt`` of ``n_out`` tokens decoded; None with fewer than MIN_TIMED tokens timed.
        ds4: from its first piece of text relayed to the last, the tokens between (all but the first and the stop
        token, never streamed); a reply not streamed has none. oMLX: from its second decode forward to the newest,
        the tokens they fed."""
        if rt.chunks is not None:
            (t0, t1), n = rt.chunks, n_out - 2
        elif rt.dec0 is not None:
            (t0, p0), (t1, p1) = rt.dec0, rt.dec1
            n = p1 - p0
        else:
            return None
        return round(n / (t1 - t0), 1) if n >= MIN_TIMED and t1 > t0 else None

    def _file_positions(self, session, starts: Optional[list], tokens: int, step_refs) -> None:
        for j, rid in enumerate(session.owners or ()):
            rt = self.requests.get(rid)
            if rt is None:
                continue
            row = session.rows[j] if session.rows and j < len(session.rows) else j
            start = starts[row] if starts is not None and row < len(starts) else rt.pos_end
            rt.step_pos[session.step] = start
            rt.pos_end = start + tokens
            if step_refs is not None:
                step_refs[rid]["start"] = start

    def finish(self, request_id: str, prompt_ids: list[int], output_ids: list[int],
               tracer: Optional[ModelTracer] = None) -> None:
        with self.lock:
            rt = self.requests.pop(request_id, None)
            if rt is not None and rt.steer_spec is not None:
                self._tombstone(request_id)
        if rt is None or not rt.parts:
            if rt is not None and rt.capped:
                log.warning("metalscope: %s: its first step alone was past the trace cap (%.1f GB); nothing saved",
                            request_id[:8], TRACE_MAX_BYTES / 1e9)
            if rt is not None and rt.steer_spec is not None:
                log.info("metalscope: steered request %s finished untraced (%s%s)", request_id[:8],
                         rt.steer_spec.label, f"; FAILED: {rt.steer_failed}" if rt.steer_failed else "")
            if rt is not None and rt.level is not None:
                # Traced but nothing recorded (no forward reached the hooks): nothing to save, but its live start
                # still gets an end, or the dock shows it busy (the release after this finds nothing to end).
                try:
                    self.streamer.on_release(rt)
                except Exception:
                    log.exception("metalscope: the live feed could not end %s", request_id[:8])
            return
        if getattr(tracer, "fresh", None):          # the last step's records (earlier ones are host copies already)
            fresh, tracer.fresh = tracer.fresh, []
            self._host(fresh)
        rt.meta["decode_tps"] = self.decode_speed(rt, len(output_ids or []))
        when = datetime.fromtimestamp(rt.started)
        trace_id = f"{when:%Y%m%d-%H%M%S}-{rt.request_id[:8]}"
        try:
            self.streamer.on_finish(rt, list(output_ids or []), tracer, trace_id)
        except Exception:
            log.exception("metalscope: live stream finish failed for %s", request_id)
        # The capture session's routing rule (ds4's tracer holds it itself), read now: the model
        # may unload before the writer saves. How router_ids / router_probs were made: the rule, k, exact or
        # approximate, shared experts (routing.py).
        session = (tracer.capture or tracer) if tracer is not None else None
        routing = getattr(session, "routing", None)
        if routing is not None and any(k.startswith("router_ids/") for k in rt.parts):
            rt.meta.update(routing=routing.meta(), num_experts=routing.num_experts)
        self.writer.submit(self._save, rt, list(prompt_ids or []), list(output_ids or []), tracer,
                           config.traces_dir() / f"{when:%Y-%m-%d}" / trace_id)

    def release(self, request_id: Optional[str]) -> bool:
        """Forget a tracked request that ended without finishing (refused after registering, failed in the engine,
        aborted): nothing is saved, a steered one keeps its KV guard (tombstone), the live feed gets its end. Every
        exit path may call it: when finish (or an earlier release) already took the request it does nothing, so a
        request is let go exactly once. True if it released."""
        with self.lock:
            rt = self.requests.pop(request_id, None)
            if rt is None:
                return False
            if rt.steer_spec is not None:
                self._tombstone(request_id)
        _forget(rt)
        log.info("metalscope: request %s ended unfinished; nothing saved", str(request_id)[:8])
        if rt.level is not None:
            try:
                self.streamer.on_release(rt)
            except Exception:
                log.exception("metalscope: the live feed could not end %s", str(request_id)[:8])
        return True

    # -- persistence -------------------------------------------------------
    def _save(self, rt: RequestTrace, prompt_ids: list, output_ids: list, tracer: Optional[ModelTracer],
              out: Path) -> None:
        """On the writer thread: the request's records (host copies by now, finish) assembled and saved."""
        written = None      # the folder, saved and not yet in the index: removed if what follows fails (no prune sees it)
        try:
            arrays = assemble(rt.parts, rt.step_pos)
            # The last sampled token is never fed back, so it has no activations.
            fed = prompt_ids + output_ids[:-1]
            n_rec = next(iter(arrays.values())).shape[0] if arrays else 0
            whole, cut = len(fed), 0
            if not rt.step_pos:
                # One step after another: the records start at the first position this request fed (past a prompt
                # prefix served from a cache), and may stop at the cap or run past the stop token (pipelined decode).
                cut = _first_fed(arrays.get("input_ids/-1"), fed, len(fed) - n_rec - rt.dropped)
                fed = fed[cut:cut + n_rec]
                if cut:
                    log.warning("metalscope: %s recorded %d positions for %d tokens (prompt %d, output %d, steps %s)",
                                rt.request_id, n_rec, whole, len(prompt_ids), len(output_ids), rt.step_tokens[:6])
            else:
                # Filed by position: row i is position i (rows a reused prefix holds are empty), so the tokens past
                # the last row, or from the first position not kept at the cap, are left out; never the first ones.
                end = min(n_rec, len(fed) if rt.cap_at is None else rt.cap_at)
                if end < len(fed) and rt.cap_at is None:
                    log.warning("metalscope: %s recorded %d positions for %d tokens (prompt %d, output %d, steps %s)",
                                rt.request_id, n_rec, len(fed), len(prompt_ids), len(output_ids), rt.step_tokens[:6])
                fed = fed[:end]
            if n_rec > len(fed):
                # Pipelined decode launches the next step before it sees the stop token; those trailing passes
                # belong to no kept token. Filed by position, positions past the cap neither.
                arrays = {k: v[:len(fed)] for k, v in arrays.items()}
                n_rec = len(fed)
            # Capped: the cap cut the conversation's own tokens, not only passes past the stop token.
            capped = rt.capped and cut + n_rec < whole
            prompt_len = min(n_rec, max(0, len(prompt_ids) - cut))
            # The first position a forward fed (filed by position): the rows before it were read by an earlier request
            # whose state this one reused (ds4), so they hold no records (-1 / NaN) though their tokens are known.
            read_from = min(max(0, min(rt.step_pos.values())), prompt_len) if rt.step_pos else 0
            tok = tracer.tokenizer if tracer is not None else None
            decode = (lambda i: tok.decode([i])) if tok is not None else (lambda i: f"<{i}>")
            token_text = token_texts(tok.decode, fed) if tok is not None else [decode(i) for i in fed]
            meta = dict(rt.meta)
            meta.update(
                request_id=rt.request_id, level=rt.level, started=rt.started, finished=time.time(),
                model=meta.get("model") or (config.short_name(tracer.name) if tracer else None),
                adapter=tracer.adapter_name if tracer else None,
                completion=tok.decode(output_ids) if tok is not None else None,
                generated_tokens=len(output_ids),
                step_tokens=rt.step_tokens[:8] + (["..."] if len(rt.step_tokens) > 8 else []),
                fed_rows=sum(rt.step_tokens),          # every step's rows: the prompt's, and under MTP a window's rejected ones
                last_token_text=decode(output_ids[-1]) if output_ids else None,
                read_from=read_from,
            )
            meta["title"] = (meta.get("last_user") or meta.get("completion") or "")[:80]
            if cut:
                meta["tokens_cut"] = cut                 # the conversation's first tokens, not kept (lookup.py refuses)
            if rt.partial:
                meta["partial"] = list(rt.partial)       # some capture hook failed: records are missing (why)
            if capped:
                meta["capped"] = n_rec                   # stopped at the trace cap: the positions kept
            if rt.steer_spec is not None:
                # Saved so analyses can exclude or select steered replies (and
                # the norm table never measures them).
                meta["steer"] = dict(rt.steer.meta if rt.steer is not None else dict(label=rt.steer_spec.label),
                                     failed=rt.steer_failed, partial=rt.steer_partial)     # the index's label too
            trace = Trace(arrays, fed, token_text, prompt_len, meta)
            path = written = trace.save(out)
            if capped:
                kept = max(0, n_rec - prompt_len)
                log.warning("metalscope: %s reached the trace cap (%.1f GB of records): it keeps %d reply tokens of %d",
                            rt.request_id[:8], TRACE_MAX_BYTES / 1e9, kept, len(output_ids))
                self.hub.publish(dict(type="capped", rid=rt.request_id, client=rt.meta.get("client"), tokens=kept))
            entry = index_entry(path, dict(meta, prompt_len=prompt_len))
            append_index(entry)
            written = None
            self.saved += 1
            log.info("metalscope: saved %s (%d tokens)", path, n_rec)
            self.keeper.poke()                      # queues a budget check; never prunes on this thread
            self.tally.submit(entry)                # queues the expert tally's count; never counts on this thread
        except Exception:
            log.exception("metalscope: failed to save trace for %s", rt.request_id)
            if written is not None:
                shutil.rmtree(written, ignore_errors=True)

    def _sweep(self) -> None:
        """Drop traces for requests that vanished (aborted) without finishing; a traced one's live start gets its end.
        A running request is never stale: each forward refreshes ``touched`` (close_step for traced rows,
        _steer_rows for steered ones)."""
        cutoff = time.time() - STALE_SECONDS
        with self.lock:
            stale = [self.requests.pop(r) for r, t in list(self.requests.items()) if t.touched < cutoff]
            for rt in stale:
                if rt.steer_spec is not None:
                    self._tombstone(rt.request_id)
        for rt in stale:
            _forget(rt)
            if rt.level is not None:
                try:
                    self.streamer.on_release(rt)
                except Exception:
                    log.exception("metalscope: the live feed could not end %s", rt.request_id[:8])


PAST_STOP = 2       # passes pipelined decode may run past the stop token ("a step or two", livestream.py)


def _first_fed(ids, fed: list, at: int) -> int:
    """The position of ``fed`` the records start at when filed one step after another: ``at`` (they end where ``fed``
    ends), or up to PAST_STOP later when they run that many passes past the stop token, whichever their input ids
    (``ids``, -1 where none) match; ``at`` when none does or there are none."""
    at = max(0, at)
    if ids is None:
        return at
    for c in range(at, min(at + PAST_STOP, len(fed)) + 1):
        m = min(len(ids), len(fed) - c)
        got, want = ids[:m], np.asarray(fed[c:c + m])
        if np.array_equal(got[got >= 0], want[got >= 0]):
            return c
    return at


def _forget(rt: RequestTrace) -> None:
    """Let a request's records go at once: each list emptied too, as the tracer's last step (``fresh``) may still hold
    them until its next forward."""
    for parts in list(rt.parts.values()):      # a copy: close_step may add a key meanwhile
        parts.clear()
    rt.parts.clear()


_LAYER_SETS: dict = {}


def _shared(layers):
    """One tuple for each set of layers records are stacked over, shared by every step that keeps one (a list a record
    cost up to 1.8 KB a step on Flash Next)."""
    if layers is None:
        return None
    t = tuple(layers)
    return _LAYER_SETS.setdefault(t, t)


def assemble(parts: dict, step_pos: dict) -> dict:
    """A request's records (RequestTrace.parts) as its trace saves them: "<kind>/<layer>" -> a row a position.

    Filed by position (``step_pos``: MTP decode, ds4), each step lands at its own, a later one overwriting an earlier
    (mtp.py); steps with no position are dropped. Otherwise each layer's steps go in order, right-aligned to the
    longest: oMLX reads a prompt without the LM head, so a record may start late. Positions no step holds are -1
    (integer ids) or NaN (then float32). Routing weights are float16 (readers cast: exact for Qwen3.6/3.8 and gpt-oss
    on oMLX, else within 5e-4).

    Each kind is built in one array over its layers and its steps are let go before the next kind is built, so the
    save holds the records plus one kind's array, never a view for each layer and token. ``parts`` is emptied."""
    kinds: dict = defaultdict(list)       # kind -> [(step, batch value, row, its layers, stacked)]
    for key in list(parts):
        kind, layer = key.rsplit("/", 1)
        one = (int(layer),)
        kinds[kind] += [(s, np.asarray(a), j, ls or one, ls is not None) for s, a, j, ls in parts.pop(key)
                        if not step_pos or s in step_pos]
    span = lambda e: e[1].shape[2 if e[4] else 1]
    info, n = {}, 0
    for kind, entries in kinds.items():
        entries.sort(key=lambda e: e[0])
        rows: dict = {}             # layer -> its row
        groups: dict = {}           # id(a record's layers) -> their rows
        for e in entries:
            if id(e[3]) not in groups:
                groups[id(e[3])] = np.array([rows.setdefault(l, len(rows)) for l in e[3]])
        total = np.zeros(len(rows), np.int64)
        for e in entries:
            if step_pos:
                n = max(n, step_pos[e[0]] + span(e))
            else:
                total[groups[id(e[3])]] += span(e)
        n = n if step_pos else max(n, int(total.max(initial=0)))
        info[kind] = rows, groups, total
    out = {}
    for kind in list(kinds):
        entries = kinds.pop(kind)
        if not entries:
            continue
        rows, groups, total = info.pop(kind)
        a0, dt = entries[0][1], np.result_type(*{e[1].dtype for e in entries})     # never narrowed
        integer = np.issubdtype(dt, np.integer)
        padded = bool(step_pos) or bool((total < n).any())
        # A float record with rows no step holds is float32 (NaN there): all of them filed by position, else the
        # layers that start late (pad); routing weights are float16 either way.
        pad = dt if integer else np.promote_types(dt, np.float32)
        dtype = np.float16 if kind == "router_probs" else pad if step_pos else dt
        shape = (len(rows), n, *a0.shape[3 if entries[0][4] else 2:])
        buf = np.full(shape, -1 if integer else np.nan, dtype) if padded else np.empty(shape, dtype)
        at = n - total                # each layer's next row (one step after another)
        for step, a, j, layers, stacked in entries:
            r = a[:, j] if stacked else a[j][None]             # [layers, T, ...]
            g = groups[id(layers)]
            if step_pos:
                s, T = step_pos[step], r.shape[1]
                lo, hi = max(0, s), min(n, s + T)
                if hi > lo:
                    buf[g, lo:hi] = r[:, lo - s:hi - s]
            elif (at[g] == at[g[0]]).all():
                buf[g, at[g[0]]:at[g[0]] + r.shape[1]] = r
                at[g] += r.shape[1]
            else:
                for i, row in enumerate(g):
                    buf[row, at[row]:at[row] + r.shape[1]] = r[i]
                at[g] += r.shape[1]
        del entries
        for layer, row in rows.items():
            late = not (step_pos or integer or kind == "router_probs") and total[row] < n
            out[f"{kind}/{layer}"] = buf[row].astype(pad, copy=False) if late else buf[row]
    return out


def new_request_id() -> str:
    return str(uuid.uuid4())


MANAGER = TraceManager()

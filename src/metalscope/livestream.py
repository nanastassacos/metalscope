"""Per-token live events for traced requests, published to SSE subscribers.

Events (JSON, one per SSE message):

    start    {rid, client, model, level, conversation, turn[, edit]}
             edit (the ds4 proxy's answer from one token): {of, at, steer}, the reply is reply ``of``'s tokens
             before ``at``, then its own: its token 0 is the joined reply's token ``at``
    prefill  {rid, client, read}                       one per prefill chunk: the prompt tokens read so far
    token    {rid, client, i, id, text, p, entropy, alts, n_layers[, lens, lens10, experts, expert_w, expert_k, writes,
             write_k, reads, read_k]}
    end      {rid, client, trace_id, n_tokens, text}   authoritative final text
    capped   {rid, client, tokens}                     once saved, after its end: the trace reached its cap and keeps
                                                       the reply's first ``tokens`` (live.TraceManager._save)

A token's fields:
- ``p``, ``alts``, ``entropy``: the emitted token's probability, the other top choices, the choice's uncertainty.
- ``lens``: the lens's top-1 at every read layer, shallow to deep, [[layer, id, text], ...] (Pi finds the settle
  layer in it). The J-lens where one is installed, else the logit lens (``lens_by``: "jlens" or "logit"); with a
  J-lens, ``lens_logit`` and ``lens10_logit`` are the logit lens's too.
- ``lens10`` (for subscribers that ask, ~2 KB a token): each read layer's top 10, as {l: layers, k, i: distinct
  token ids, w: their texts, c: per cell (layer-major, rank order) an index into i/w, p: per cell round(-log p * 20)
  (255: none)}.
- ``experts`` (MoE): the top-k experts of the forward the token was fed into (as /api/traces/ID/routing gives a saved reply's),
  every MoE layer in order, flattened [L * k]; ``expert_w`` each one's share of its layer's top-k router
  probability, 0-255; ``expert_k`` is k. Absent for the last token (never fed back) and without router records.
- ``writes`` (for subscribers that ask): how much each layer of the forward that chose the token (as ``lens``)
  changed each residual stream, the norm of its output less its input per stream (``resid_write_norm``), every layer
  in order, flattened [L * hc], 3 significant digits; ``write_k`` is hc (1 for a single residual; Inkling 2: its
  running sum, by visit).
- ``shares`` (beside ``writes``, where the residual's size is recorded and there are no ``reads``): each layer's whole
  write (the root of the sum of its streams' squares) over the whole residual it read, the layer before's output
  (``hc_stream_norm``, else ``resid_norm``), [L], 3 significant digits; layer 0's is 0 (it reads the embedding, whose
  size is not recorded).
- ``reads`` (beside ``writes``, Inkling 2): where each visit's inputs read from, each source's share
  (``resid_read_share``): the finished blocks in order (the first the embedding's; 0 while not finished), the running
  sum last, flattened [L * k] the same way; ``read_k`` is k (5 on the Mini).

Under MTP decode the path (steps keyed by position, drafts that may be rejected) is mtp.py's (``MtpLive``); this
module is the one-token-per-step path and what both share.

Nothing here forces GPU work: a step's records are read on the *next* step, by which time the engine has
synchronised on them, so a token is announced about one step after it is sampled. Pipelined decoding runs a step or
two past the stop token; clients should trust ``end``.
"""

from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import numpy as np

from .mtp import MtpLive

LIVE_KINDS = frozenset(("input_ids", "logit_ids", "logit_logp", "logit_entropy", "lens_ids", "lens_logp",
                        "jlens_ids", "jlens_logp", "router_ids", "router_probs", "resid_write_norm",
                        "resid_read_share", "hc_stream_norm", "resid_norm"))
PER_LAYER = frozenset(("lens_ids", "lens_logp", "jlens_ids", "jlens_logp"))
ROUTING_FIELDS = ("experts", "expert_w", "expert_k")
LENS_FIELDS = ("lens10", "lens10_logit")   # sent only to subscribers that asked (``lens``)
WRITES_FIELDS = ("writes", "write_k", "shares", "reads", "read_k")   # sent only to subscribers that asked (``writes``)
LENS_P_SCALE = 20           # lens10 p as round(-log p * LENS_P_SCALE), the saved expert map payload's byte
STACKED_KINDS = frozenset(("router_ids", "router_probs", "resid_write_norm", "resid_read_share", "hc_stream_norm",
                           "resid_norm"))
N_ALTS = 10  # runners-up per token in the live stream: the whole output top 10 the trace reads (both engines read 10),
             # less the token written when it is among them; ~0.1 KB a token over five (Pi's TOKENS)
QUEUE_MAX = 4096



def token_text(decode, window: list[int], token: int) -> str:
    """The text ``token`` adds after the recent ``window`` of tokens: the difference of the two decodes. A
    character split over several byte-level tokens (an emoji) decodes to U+FFFD until its last byte arrives, so
    trailing U+FFFDs are held back: the partial tokens add nothing and the last one carries the whole character."""
    before, after = decode(window).rstrip("\ufffd"), decode(window + [token]).rstrip("\ufffd")
    return after[len(before):] if after.startswith(before) else decode([token])

def token_texts(decode, ids: list[int]) -> list[str]:
    """Each token's text for a saved trace: its own decode, except where that is a partial character (U+FFFD),
    which ``token_text`` resolves in context: the partial tokens get "", the last the whole character."""
    out = [decode([t]) for t in ids]
    for k, t in enumerate(ids):
        if "\ufffd" in out[k]:
            out[k] = token_text(decode, ids[max(0, k - 8):k], t)
    return out

def add_ref(refs: dict, kind: str, layer: int, layers: Optional[list], value, j: int) -> None:
    """File one of a step's records (a ``LIVE_KINDS`` kind; ``j`` the request's batch row of ``value``) in the refs
    ``LiveStreamer.on_step`` takes: per-layer kinds as [(layer, value, j)], stacked ones as [(layers, value
    [L, B, T, ..], j)], the rest as (value, j). Both backends file through it (live.TraceManager)."""
    if kind in PER_LAYER:
        refs.setdefault(kind, []).append((layer, value, j))
    elif kind in STACKED_KINDS:
        refs.setdefault(kind, []).append((layers, value, j))
    else:
        refs[kind] = (value, j)


class _Sub(NamedTuple):
    """A subscriber: its loop and queue, the client it follows (None: all), and the opt-in fields it asked for."""
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue
    client: Optional[str]
    experts: bool
    lens: bool
    writes: bool


class LiveHub:
    """Fan events out from the scheduler thread to asyncio subscribers."""

    def __init__(self):
        self._subs: list[_Sub] = []
        self._lock = threading.Lock()
        self.experts = False   # some subscriber asked for routing (``experts``: ~3 KB per token)
        self.lens = False      # some subscriber asked for each read layer's top 10 (``lens``: ~2 KB per token)
        self.writes = False    # some subscriber asked for the writes (``writes``: ~1.3 KB a token, Inkling 2 ~2)

    def active(self) -> bool:
        return bool(self._subs)

    def subscribe(self, loop: asyncio.AbstractEventLoop, client: Optional[str], experts: bool = False,
                  lens: bool = False, writes: bool = False) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        with self._lock:
            self._subs.append(_Sub(loop, q, client, experts, lens, writes))
            self._wants()
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [s for s in self._subs if s.queue is not q]
            self._wants()

    def _wants(self) -> None:
        self.experts = any(s.experts for s in self._subs)
        self.lens = any(s.lens for s in self._subs)
        self.writes = any(s.writes for s in self._subs)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        views: dict = {}   # one copy per combination of opt-in fields left out
        for loop, q, client, experts, lens, writes in subs:
            if client is None or client == event.get("client"):
                drop = ((() if experts else ROUTING_FIELDS) + (() if lens else LENS_FIELDS)
                        + (() if writes else WRITES_FIELDS))
                if drop not in views:
                    views[drop] = ({k: v for k, v in event.items() if k not in drop}
                                   if any(k in event for k in drop) else event)
                loop.call_soon_threadsafe(_offer, q, views[drop])


def _offer(q: asyncio.Queue, event: dict) -> None:
    if q.full():  # a stalled client loses its oldest events, never blocks us
        q.get_nowait()
    q.put_nowait(event)


@dataclass
class LiveState:
    client: Optional[str]
    pending: Optional[dict] = None     # this step's refs; read on the next step
    prev_dist: Optional[dict] = None   # the distribution that chose the next fed token
    decode_steps: int = 0
    fed: int = 0                       # tokens fed so far (prefill and decode): the next one's position
    announced: list = field(default_factory=list)
    mtp: Optional[MtpLive] = None  # MTP decode only: created on its first position-keyed step
    prefilled: int = 0                 # prompt tokens read so far (prefill events' ``read``: a repeated event adds nothing)


def _row(ref) -> np.ndarray:
    value, j = ref
    return np.asarray(value)[j]


def _stacked(refs: dict, kind: str, t: int) -> np.ndarray:
    """Row ``t`` of a stacked kind's records (``add_ref``) at every layer, shallow to deep: [L, ..]."""
    layers = [l for ls, _, _ in refs[kind] for l in ls]
    rows = np.concatenate([np.asarray(v)[:, j, t] for _, v, j in refs[kind]])
    return rows[np.argsort(layers, kind="stable")]


def _routing(refs: dict, t: int) -> dict:
    """Routing of the token fed at row ``t`` of a step, for its token event:
    ``experts`` top-k ids per MoE layer in layer order, flattened [L * k],
    ``expert_w`` each one's share of the routed mix (router_probs over their sum, 0-255)."""
    if not refs.get("router_ids") or not refs.get("router_probs"):
        return {}
    ids, pr = _stacked(refs, "router_ids", t), np.nan_to_num(_stacked(refs, "router_probs", t).astype(np.float32))
    w = np.rint(255 * pr / np.maximum(pr.sum(-1, keepdims=True), 1e-9))
    return dict(experts=ids.ravel().tolist(), expert_w=w.astype(np.uint8).ravel().tolist(), expert_k=int(ids.shape[-1]))


def _writes(refs: dict, t: int) -> dict:
    """What each layer of a step wrote into each stream at row ``t``, for the token that row chose: ``writes`` [L * hc]
    layer-major, 3 significant digits (they run from about 0.001 to 80), ``write_k`` hc; with read shares (Inkling 2)
    ``reads`` [L * k] the same way, ``read_k`` k; else, where every layer's output size is recorded, ``shares`` [L]."""
    out, rows = {}, {}
    sig = lambda a: [float(f"{x:.3g}") for x in a.ravel().tolist()]
    for kind, name in (("resid_write_norm", "write"), ("resid_read_share", "read")):
        if refs.get(kind):
            rows[kind] = w = np.nan_to_num(_stacked(refs, kind, t).astype(np.float32))         # [L, k]
            out[f"{name}s"], out[f"{name}_k"] = sig(w), int(w.shape[-1])
    size = "hc_stream_norm" if refs.get("hc_stream_norm") else "resid_norm"
    layers = lambda kind: sorted(l for ls, _, _ in refs.get(kind) or () for l in ls)
    L = len(rows.get("resid_write_norm", ()))
    if L and "reads" not in out and layers(size) == layers("resid_write_norm") == list(range(L)):
        w = np.linalg.norm(rows["resid_write_norm"], axis=-1)
        s = np.linalg.norm(np.nan_to_num(_stacked(refs, size, t).astype(np.float32)).reshape(L, -1), axis=-1)
        share = np.zeros_like(w)
        np.divide(w[1:], s[:-1], out=share[1:], where=s[:-1] > 0)        # layer l read layer l - 1's output
        out["shares"] = sig(np.nan_to_num(share))                         # an overflowed write: a finite share, valid JSON
    return out


class LiveStreamer:
    def __init__(self, hub: LiveHub):
        self.hub = hub

    # -- lifecycle -------------------------------------------------------
    def on_register(self, rt) -> None:
        rt.live = LiveState(client=rt.meta.get("client"))
        if self.hub.active():
            self._publish(rt, "start", model=rt.meta.get("model"), level=rt.level,
                          conversation=rt.meta.get("conversation"), turn=rt.meta.get("turn"),
                          **({"edit": rt.meta["edit"]} if rt.meta.get("edit") else {}))

    def on_step(self, rt, refs: dict, tracer) -> None:
        live = rt.live
        if live is None:
            return
        previous, live.pending = live.pending, refs
        if previous is not None:
            self._process(rt, previous, tracer)

    def on_finish(self, rt, output_ids: list[int], tracer, trace_id: str) -> None:
        live = rt.live
        if live is None or not self.hub.active():
            return
        if live.pending is not None:
            self._process(rt, live.pending, tracer)
            live.pending = None
        if live.mtp is not None:
            live.mtp.finish(self, rt, output_ids, tracer)
        # The last sampled token is never fed back; announce it from the last dist.
        elif output_ids and len(live.announced) < len(output_ids) and live.prev_dist is not None:
            self._announce(rt, int(output_ids[len(live.announced)]), live.prev_dist, tracer)
        tok = getattr(tracer, "tokenizer", None)
        self._publish(rt, "end", trace_id=trace_id, n_tokens=len(output_ids),
                      text=tok.decode(output_ids) if tok is not None else None)

    def on_release(self, rt) -> None:
        """The request ended without finishing (TraceManager.release): its start gets an end, with no trace; the
        tokens already announced stand."""
        live = rt.live
        if live is None or not self.hub.active():
            return
        self._publish(rt, "end", trace_id=None, n_tokens=len(live.announced), text=None, unfinished=True)

    # -- per step --------------------------------------------------------
    def _process(self, rt, refs: dict, tracer) -> None:
        if "input_ids" not in refs:
            return
        live = rt.live
        if "start" in refs:  # MTP decode (mtp.py): steps carry their absolute start position
            if live.mtp is None:
                live.mtp = MtpLive()
            return live.mtp.step(self, rt, refs, tracer)
        ids = _row(refs["input_ids"])
        P = rt.prompt_len       # from the scheduler (note_prompt); 0 if it never said
        if "pos" in refs:       # ds4 files each step at its position: one that reused a prompt prefix starts past 0
            live.fed = int(refs["pos"])
        pos, live.fed = live.fed, live.fed + ids.shape[0]
        # A prefill chunk, or prompt tokens fed one at a time (DeepSeek V4 feeds its prompt's last few so).
        if ids.shape[0] > 1 or (P and pos < P - 1):
            return self._prefill(rt, ids.shape[0])
        # Decode step: its input was sampled from the previous step's dist. Not announced: the last prompt token
        # (without a prompt length: the first decode input) and the stop token, fed once more before the scheduler
        # ends the reply (GLM-4.5).
        first = (pos == P - 1) if P else live.decode_steps == 0
        if not first and int(ids[0]) not in rt.stop_ids and live.prev_dist is not None:
            self._announce(rt, int(ids[0]), live.prev_dist, tracer, _routing(refs, 0) if self.hub.experts else None)
        live.decode_steps += 1
        if "logit_ids" in refs:
            live.prev_dist = self._dist(refs, 0)

    def _prefill(self, rt, n: int) -> None:
        rt.live.prefilled += int(n)
        self._publish(rt, "prefill", read=rt.live.prefilled)

    def _dist(self, refs: dict, t: int) -> dict:
        """The next-token distribution (and lens readouts) at row ``t`` of a step."""
        dist = dict(ids=_row(refs["logit_ids"])[t], logp=_row(refs["logit_logp"])[t],
                    entropy=float(_row(refs["logit_entropy"])[t]))
        if "start" in refs and self.hub.experts:  # MTP: the routing of the token fed here (MtpLive._routing)
            dist["routing"] = _routing(refs, t)
        if self.hub.writes:
            dist.update(_writes(refs, t))
        def read(kind: str) -> list:
            """[(layer, value, j)] of a lens record kind, shallow to deep, the layers that read row t: a row recorded
            empty (ids -1, before a prompt chunk's lens tail) is left out."""
            return [(layer, v, j) for layer, v, j in sorted(refs.get(kind) or [], key=lambda r: r[0])
                    if np.asarray(v)[j][t, 0] >= 0]
        # Top-1 per read layer: the J-lens where it read, else the logit lens; with both, the logit lens's beside it.
        top1 = lambda kind: [(layer, int(np.asarray(v)[j][t, 0])) for layer, v, j in read(kind)]
        main = "jlens_ids" if refs.get("jlens_ids") else "lens_ids" if refs.get("lens_ids") else None
        if main:
            dist["lens"] = top1(main)
            dist["lens_by"] = "jlens" if main == "jlens_ids" else "logit"      # which lens read it (/jlens off|on)
            if main == "jlens_ids" and refs.get("lens_ids"):
                dist["lens_logit"] = top1("lens_ids")
        def top10(prefix: str):
            logps = {layer: np.array(np.asarray(v)[j][t]) for layer, v, j in refs[f"{prefix}_logp"]}
            return [(layer, np.array(np.asarray(v)[j][t]), logps.get(layer)) for layer, v, j in read(f"{prefix}_ids")]
        if self.hub.lens and main:   # lens10 is the same lens as ``lens``; lens10_logit the logit lens beside it
            p = main[:-4]
            if refs.get(f"{p}_logp"):
                dist["lens10"] = top10(p)
            if "lens_logit" in dist and refs.get("lens_logp"):
                dist["lens10_logit"] = top10("lens")
        return dist

    def _announce(self, rt, token: int, dist: dict, tracer, extra: Optional[dict] = None) -> None:
        tok = getattr(tracer, "tokenizer", None)
        special = set(getattr(tok, "all_special_ids", []) or [])
        live = rt.live
        if token in special:
            live.announced.append(token)
            return
        ids, logp = dist["ids"], dist["logp"]
        hit = np.nonzero(ids == token)[0]
        p = float(math.exp(logp[hit[0]])) if hit.size else None
        decode = (lambda xs: tok.decode(xs)) if tok is not None else (lambda xs: "".join(f"<{x}>" for x in xs))
        alts = [[decode([int(i)]), round(float(math.exp(lp)), 4)]
                for i, lp in zip(ids, logp) if int(i) != token][:N_ALTS]
        lens = _top1_words(dist.get("lens"), decode)
        lens_logit = _top1_words(dist.get("lens_logit"), decode)
        text = token_text(decode, live.announced[-8:], token)
        i = len(live.announced)
        live.announced.append(token)
        top10 = {k: _lens10(dist[k], decode) for k in LENS_FIELDS if dist.get(k)}   # each read layer's top 10 (lens=1)
        self._publish(rt, "token", i=i, id=token, text=text, p=None if p is None else round(p, 4),
                      entropy=round(dist["entropy"], 4), alts=alts,
                      n_layers=getattr(tracer, "n_layers", None),
                      **({"lens": lens, "lens_by": dist["lens_by"]} if lens else {}), **top10,
                      **({"lens_logit": lens_logit} if lens_logit else {}),
                      **{k: dist[k] for k in WRITES_FIELDS if k in dist}, **(extra or {}))

    def _publish(self, rt, kind: str, **payload: Any) -> None:
        self.hub.publish(dict(type=kind, rid=rt.request_id, client=rt.meta.get("client"), **payload))


def _top1_words(top1s, decode):
    """[(layer, id)] shallow to deep -> [[layer, id, text]]: each read layer's top word."""
    if not top1s:
        return None
    words = {i: decode([i]) for i in {i for _, i in top1s}}
    return [[layer, i, words[i]] for layer, i in top1s]


def _lens10(rows, decode) -> dict:
    """The ``lens10`` field: [(layer, top-k ids, log-probs)] shallow to deep, each distinct
    word sent once per event."""
    ids: list[int] = []
    index: dict[int, int] = {}
    cells, ps = [], []
    for _, row, logp in rows:
        for r, tid in enumerate(row.tolist()):
            if tid not in index:
                index[tid] = len(ids)
                ids.append(int(tid))
            cells.append(index[tid])
            lp = float(logp[r]) if logp is not None and r < len(logp) else math.nan
            ps.append(int(min(254, max(0, round(-lp * LENS_P_SCALE)))) if math.isfinite(lp) else 255)
    texts = [decode([i]) if i >= 0 else "" for i in ids]
    return dict(l=[int(layer) for layer, _, _ in rows], k=int(len(rows[0][1])) if rows else 0, i=ids, w=texts,
                c=cells, p=ps)

"""Tracing under multi-token-prediction (MTP) speculative decoding.

With MTP on, each decode cycle runs the model once over a verify window
``[next, d1 .. dk]``: the token sampled last cycle plus k drafted tokens.
Every position of the window is computed, then the cache is rolled back past
the first rejected draft. Records of rejected positions are therefore wrong
and must not be saved, and a window of k+1 tokens is not a prefill chunk.

Instead of tracking which drafts were accepted, every forward is filed under
the absolute position of its first token (read from the KV cache after the
forward), and records are placed by position at save time (live.assemble), later
writes winning. A rejected draft's position is always computed again, with the token
that was actually emitted, before generation moves past it; positions beyond
the final length are trimmed. The same holds for live events (``MtpLive``, driven
by livestream.LiveStreamer): a position is final once a later forward starts
beyond it.

Only models whose MTP decode is enabled take this path (``mtp_decode``);
everything else keeps the one-token-per-decode-step path unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

MAX_WINDOW = 16         # verify windows are at most 16 tokens (a context copy's 15 + 1); longer forwards are prefill


def mtp_decode(model: Any) -> bool:
    """True if oMLX loaded this model with its MTP head and MTP decode on."""
    for m in (model, getattr(model, "language_model", None), getattr(model, "_language_model", None)):
        if m is not None and getattr(m, "_omlx_mtp_decode_enabled", False):
            return True
    return False


def window_starts(cache: Any, tokens: int, rows: int) -> Optional[list[int]]:
    """Absolute position of each batch row's first input token for the forward
    that just ran: the first attention cache's offset (already advanced by
    ``tokens``) minus ``tokens``. ``BatchKVCache.offset`` is per row and
    already excludes left padding; a ``CacheList`` (GLM-5.3's sparse-attention layers, Inkling 1's layers) carries it on
    its first cache; a state with no offset is passed over (Inkling 2's sliding visits: its first full-attention visit's
    KVCache places the window). None when no cache carries an offset."""
    for c in cache or ():
        off = getattr(c, "offset", None)
        if off is None and getattr(c, "caches", None):
            off = getattr(c.caches[0], "offset", None)
        if off is None:
            continue
        if isinstance(off, (int, np.integer)):
            return [int(off) - tokens] * rows
        try:
            vals = np.asarray(off).reshape(-1).tolist()   # tiny array; its graph is a few adds
        except Exception:
            return None
        if len(vals) == rows:
            return [int(v) - tokens for v in vals]
        if len(vals) == 1:
            return [int(vals[0]) - tokens] * rows
    return None


@dataclass
class MtpLive:
    """Live-stream state of one request under MTP decode. Each step's refs
    carry ``start``; the streamer (livestream.LiveStreamer) hands them here
    and provides the shared pieces (_prefill, _dist, _announce)."""

    by_pos: dict = field(default_factory=dict)   # position -> (input id, dist at that position)
    prompt_end: int = 0     # end of the prefilled prompt (the last prompt token is fed by the first decode)
    next_pos: int = 0       # announced up to here

    def step(self, streamer, rt, refs: dict, tracer) -> None:
        """A forward: a prefill chunk (no logits) or a window of 1..k+1 tokens
        whose drafts may be rejected. Inputs and dists are kept by position,
        later windows overwriting rejected drafts; a position is announced once
        a forward starts beyond it, so it can no longer change."""
        value, j = refs["input_ids"]
        ids = np.asarray(value)[j]
        start, T = int(refs["start"]), int(ids.shape[0])
        # The last prompt token is fed by the first decode window, at P - 1.
        last = rt.prompt_len - 1
        if "logit_ids" not in refs or T > MAX_WINDOW or start + T <= last:  # prefill chunk
            self.prompt_end = max(self.prompt_end, start + T)
            streamer._prefill(rt, ids.shape[0])
            return
        k = 0
        if start < last:
            # A prompt tail prefilled together with the last prompt token (oMLX
            # does this with logits): rows before it are prompt, the rest no draft.
            k = last - start
            streamer._prefill(rt, k)
        if last >= 0:
            self.prompt_end = max(self.prompt_end, last)
        self._commit(streamer, rt, start + k, tracer)
        for t in range(k, T):
            self.by_pos[start + t] = (int(ids[t]), streamer._dist(refs, t))

    def finish(self, streamer, rt, output_ids: list[int], tracer) -> None:
        """Announce what is left."""
        # The output ids are final; each was chosen by the dist one position earlier.
        for k in range(len(rt.live.announced), len(output_ids)):
            d = self.by_pos.get(self.prompt_end + k)
            if d is None:
                break
            q = self.prompt_end + k + 1
            # by_pos[q] may hold a rejected draft when the reply ended on a correction: that token was never fed.
            fed = q in self.by_pos and self.by_pos[q][0] == int(output_ids[k])
            streamer._announce(rt, int(output_ids[k]), d[1], tracer, self._routing(q) if fed else None)

    def _commit(self, streamer, rt, upto: int, tracer) -> None:
        """Announce reply tokens at positions below ``upto``. The first decode
        input is the last prompt token (at prompt_end), so the reply starts
        one position later; token q was chosen by the dist at q - 1."""
        q = max(self.next_pos, self.prompt_end + 1)
        while q < upto and q in self.by_pos and q - 1 in self.by_pos:
            streamer._announce(rt, self.by_pos[q][0], self.by_pos[q - 1][1], tracer, self._routing(q))
            q += 1
        self.next_pos = max(self.next_pos, q)
        for p in [p for p in self.by_pos if p < q - 1]:   # keep the dist that chooses the next token
            del self.by_pos[p]

    def _routing(self, q: int) -> dict:
        """The routing of the token fed at q (its own row's, livestream._dist), for its token event."""
        return self.by_pos[q][1].get("routing", {})

"""Reading the capture file ds4 writes for one request (the metalscope patch's ds4_scope*.inc).

The file is a header, then records appended as the request runs: the prompt
(PRMT), one per forward pass (STEP: per layer each row's routing, its streams' norms and what the layer changed in
each stream, then the head's reads), a rewind (RWND), text the server inserted into the reply (SUFX), the residual
sums (RSUM), each prompt token's log-probability and each row's top 10 (NXLP and NXTK, scope ``next_lp``) and the
reply at the end (END_).
``ScopeReader`` reads whatever has been written so far and keeps its place, so the proxy can follow a request live. PRMT is the whole prompt; the first STEP starts past 0 when the
request reused the rows ds4's session held. Any other tag is skipped by its length.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

MAGIC = b"DS4SCOPE"
VERSION = 6
FAMILIES = {1: "qwen4", 2: "deepseek41"}
MAX_REMOVE = 8   # directions a removal projects out at one layer (the patch's DS4_SCOPE_MAX_REMOVE)
OUT, LENS, RESID, JLENS, AT_IDS = 1, 2, 4, 16, 64   # STEP flags
STOPPED = "; trace stopped: "   # END_'s finish, then why, when ds4 stopped recording partway (the reply went on)


@dataclass
class Header:
    family: str
    n_layer: int
    n_hc: int
    n_embd: int
    k: int            # experts used per token
    n_expert: int
    n_vocab: int
    topk: int
    lens: list[int]   # logit-lens layers


@dataclass
class Step:
    """One forward pass: rows ``pos0 .. pos0 + T - 1``. The head read its last
    R rows (the last row, or every kept row of an MTP verify): the head kinds
    are per those rows."""
    pos0: int
    tokens: np.ndarray                    # [T] int32
    experts: np.ndarray                   # [L, T, k] int32
    weights: np.ndarray                   # [L, T, k] float32
    stream_norm: np.ndarray               # [L, T, hc] float32
    write_norm: np.ndarray                # [L, T, hc] float32: what the layer changed in each stream (out less in)
    R: int = 0
    resid_norm: Optional[np.ndarray] = None    # [R, L]: as the head reads each row
    lens_ids: Optional[np.ndarray] = None      # [R, n_lens, topk]
    lens_logp: Optional[np.ndarray] = None
    out_ids: Optional[np.ndarray] = None       # [R, topk]: each row's next-token distribution
    out_logp: Optional[np.ndarray] = None
    out_entropy: Optional[np.ndarray] = None   # [R]
    jlens_ids: Optional[np.ndarray] = None     # [R, nj, topk]
    jlens_logp: Optional[np.ndarray] = None
    out_at_ids: Optional[np.ndarray] = None    # [R, n]: the next-token log-probabilities at the LIDS ids

    @property
    def T(self) -> int:
        return int(self.tokens.shape[0])


@dataclass
class ResidualSums:
    """RSUM: per layer, the sum of its ``n`` prompt rows from position ``sink`` on (at most the prompt's last; each
    position once) and the prompt's last row (NaN where the layer did not compute it)."""
    sink: int
    n: np.ndarray        # [L] uint32
    sum: np.ndarray      # [L, W] float32
    last: np.ndarray     # [L, W] float32

    @property
    def mean(self) -> np.ndarray:
        """[L, W]: each layer's mean row (NaN for a layer with no rows)."""
        n = self.n[:, None].astype(np.float32)
        return np.where(n > 0, self.sum / np.maximum(n, 1), np.float32(np.nan))


@dataclass
class Rewind:
    """The session dropped positions pos.. (RWND): what is fed there next replaces them."""
    pos: int


@dataclass
class End:
    sampled: list[int]
    finish: str

    @property
    def stopped(self) -> Optional[str]:
        """Why ds4 stopped recording before the request ended (its records after that are missing), or None."""
        return self.finish.split(STOPPED, 1)[1] if STOPPED in self.finish else None


@dataclass
class ScopeReader:
    path: Path
    header: Optional[Header] = None
    prompt: Optional[list[int]] = None
    inserted: list[tuple[int, int]] = field(default_factory=list)   # SUFX: (position, tokens) fed, not sampled
    jlens: list[int] = field(default_factory=list)                          # JLNS: the J-lens layers read
    logit_ids: list[int] = field(default_factory=list)                     # LIDS: the ids Step.out_at_ids reads
    rsum: Optional[ResidualSums] = None
    next_lp: dict = field(default_factory=dict)   # NXLP: position p -> log P(prompt[p + 1] | prompt[..p])
    next_top: dict = field(default_factory=dict)  # NXTK: position p -> (ids [k], log-probabilities [k]), likeliest first
    end: Optional[End] = None
    _off: int = 0

    def poll(self) -> list:
        """The STEP and RWND records written since the last call, in order (complete records only), as Step and
        Rewind; JLNS, LIDS, PRMT, SUFX, RSUM, NXLP, NXTK and END_ are kept on the reader; any other tag is
        skipped."""
        try:
            with open(self.path, "rb") as f:
                f.seek(self._off)
                data = f.read()
        except FileNotFoundError:
            return []
        at, steps = 0, []
        if self.header is None:
            got = _header(data)
            if got is None:
                return []
            self.header, at = got
        while len(data) - at >= 8:
            tag, n = data[at:at + 4], struct.unpack_from("<I", data, at + 4)[0]
            if len(data) - at - 8 < n:
                break
            body = memoryview(data)[at + 8:at + 8 + n]
            at += 8 + n
            if tag == b"STEP":
                steps.append(self._step(body))
            elif tag == b"PRMT":
                m = struct.unpack_from("<I", body)[0]
                self.prompt = np.frombuffer(body, "<i4", m, 4).tolist()
            elif tag == b"RWND":
                steps.append(Rewind(struct.unpack_from("<I", body)[0]))
            elif tag == b"JLNS":
                m = struct.unpack_from("<I", body)[0]
                self.jlens = np.frombuffer(body, "<u4", m, 4).tolist()
            elif tag == b"LIDS":
                m = struct.unpack_from("<I", body)[0]
                self.logit_ids = np.frombuffer(body, "<i4", m, 4).tolist()
            elif tag == b"RSUM":
                L, W, sink = struct.unpack_from("<III", body)
                o = 12 + 4 * L
                self.rsum = ResidualSums(sink, np.frombuffer(body, "<u4", L, 12).copy(),
                                         np.frombuffer(body, "<f4", L * W, o).reshape(L, W).copy(),
                                         np.frombuffer(body, "<f4", L * W, o + 4 * L * W).reshape(L, W).copy())
            elif tag == b"NXLP":
                pos0, m = struct.unpack_from("<II", body)
                self.next_lp.update(zip(range(pos0, pos0 + m), np.frombuffer(body, "<f4", m, 8).tolist()))
            elif tag == b"NXTK":
                pos0, m, k = struct.unpack_from("<III", body)
                ids = np.frombuffer(body, "<i4", m * k, 12).reshape(m, k)
                lp = np.frombuffer(body, "<f4", m * k, 12 + 4 * m * k).reshape(m, k)
                self.next_top.update((pos0 + i, (ids[i].copy(), lp[i].copy())) for i in range(m))
            elif tag == b"SUFX":
                self.inserted.append(struct.unpack_from("<II", body))
            elif tag == b"END_":
                m = struct.unpack_from("<I", body)[0]
                sampled = np.frombuffer(body, "<i4", m, 4).tolist()
                ln = struct.unpack_from("<I", body, 4 + 4 * m)[0]
                self.end = End(sampled, bytes(body[8 + 4 * m:8 + 4 * m + ln]).decode("utf-8", "replace"))
        self._off += at
        return steps

    def _step(self, b: memoryview) -> Step:
        h = self.header
        pos0, T, flags, R = struct.unpack_from("<IIII", b)
        if flags & ~(OUT | LENS | RESID | JLENS | AT_IDS):
            raise ValueError(f"ds4 STEP record at {pos0} has flags {flags:#x} this metalscope does not read")
        at = 16
        L, k, hc, K, nl = h.n_layer, h.k, h.n_hc, h.topk, len(h.lens)

        def take(dtype: str, *shape: int) -> np.ndarray:
            nonlocal at
            n = int(np.prod(shape))
            a = np.frombuffer(b, dtype, n, at).reshape(shape).copy()
            at += 4 * n
            return a
        tokens = take("<i4", T)
        experts = take("<i4", L, T, k)
        weights = take("<f4", L, T, k)
        stream = take("<f4", L, T, hc)
        writes = take("<f4", L, T, hc)
        resid = take("<f4", R, L) if flags & RESID else None
        lens = (take("<i4", R, nl, K), take("<f4", R, nl, K)) if flags & LENS else (None,) * 2
        nj = len(self.jlens)
        jl = (take("<i4", R, nj, K), take("<f4", R, nj, K)) if flags & JLENS else (None,) * 2
        out = (take("<i4", R, K), take("<f4", R, K), take("<f4", R)) if flags & OUT else (None,) * 3
        at_ids = take("<f4", R, len(self.logit_ids)) if flags & AT_IDS else None
        if at != len(b):
            raise ValueError(f"ds4 STEP record at {pos0} is {len(b)} bytes, its flags {flags:#x} account for {at}")
        return Step(pos0, tokens, experts, weights, stream, writes, R if flags & (OUT | LENS | RESID | JLENS) else 0,
                    resid, *lens, *out, *jl, out_at_ids=at_ids)


def _header(data: bytes):
    if len(data) < 48 or data[:8] != MAGIC:
        return None
    version, fam, L, hc, E, k, ne, V, K, nl = struct.unpack_from("<10I", data, 8)
    if version != VERSION:
        raise ValueError(f"ds4 scope file version {version} (this metalscope reads {VERSION}: rebuild ds4 from "
                         f"src/metalscope/ds4/metalscope.patch)")
    end = 48 + 4 * nl
    if len(data) < end:
        return None
    lens = list(struct.unpack_from(f"<{nl}I", data, 48))
    return Header(FAMILIES.get(fam, str(fam)), L, hc, E, k, ne, V, K, lens), end


def write_jlens(path: Path, jacobians: dict[int, np.ndarray], in_width: int, out_width: int) -> None:
    """The J-lens file ds4 reads (ds4_scope_session.inc, scope_load_jlens): per layer (increasing) J [out_width, in_width] in f16.
    Written to a temporary name and renamed: ds4 caches it by path, size and modification time."""
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(b"DS4JLENS" + struct.pack("<IIII", 1, len(jacobians), in_width, out_width))
        for layer in sorted(jacobians):
            J = np.asarray(jacobians[layer], dtype="<f2")
            if J.shape != (out_width, in_width):
                raise ValueError(f"J-lens layer {layer} is {J.shape}, ds4 needs ({out_width}, {in_width})")
            f.write(struct.pack("<I", int(layer)) + J.tobytes())
    tmp.replace(path)


def write_steering(path: Path, vectors: dict[int, np.ndarray], width: int, remove: Optional[dict] = None) -> None:
    """The steering file ds4 reads (ds4_scope_session.inc, scope_load_steering; version 2): per layer, the
    vector to add to the whole residual row after it, already scaled; then per layer in ``remove`` its orthonormal
    rows [k, width] (SteerPlan.remove, k up to MAX_REMOVE), projected out of the row before the add."""
    remove = remove or {}
    with open(path, "wb") as f:
        f.write(b"DS4STEER" + struct.pack("<IIII", 2, len(vectors), width, len(remove)))
        for layer in sorted(vectors):
            v = np.asarray(vectors[layer], dtype="<f4").reshape(-1)
            if v.shape[0] != width:
                raise ValueError(f"steering vector at layer {layer} has width {v.shape[0]}, ds4 needs {width}")
            f.write(struct.pack("<I", int(layer)) + v.tobytes())
        for layer in sorted(remove):
            q = np.asarray(remove[layer], dtype="<f4")
            if q.ndim != 2 or q.shape[1] != width or not 0 < q.shape[0] <= MAX_REMOVE:
                raise ValueError(f"removal at layer {layer} is {q.shape}, ds4 needs (1 to {MAX_REMOVE}, {width})")
            f.write(struct.pack("<II", int(layer), q.shape[0]) + q.tobytes())

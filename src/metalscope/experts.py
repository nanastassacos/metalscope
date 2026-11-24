"""Expert map for MoE traces: where each layer's experts sit.

Traces record, per token and MoE layer, the experts the router picked
(``router_ids/<layer>``, [T, k], k the model's own). From those alone
this module builds what layout.build_layout lays out:

- a ``Corpus`` pooling many traces, deduplicated by context: routing is a
  deterministic function of the prefix, so a repeated prompt (or a shared
  system prompt) adds no evidence and is counted once;
- a 2D co-activation embedding of each layer's experts (classical MDS on
  numpy, no sklearn).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .store import Trace

# ---------------------------------------------------------------- routing data

def routing(trace: Trace) -> tuple[np.ndarray, np.ndarray]:
    """(ids [L, T, k], valid [T]) for a trace's MoE layers. Positions without a routing record (front padding from
    live.assemble: id -1 / NaN weight) are marked invalid rather than dropped, so indices stay aligned with
    trace.tokens."""
    layers = trace.layers("router_ids")
    ids = np.stack([trace.get("router_ids", i) for i in layers]).astype(np.int32)
    valid = (ids >= 0).all(axis=(0, 2))
    for i in layers:
        if f"router_probs/{i}" in trace.arrays:
            valid &= np.isfinite(trace.get("router_probs", i)).all(-1)
    return ids, valid


def has_routing(trace: Trace) -> bool:
    return any(k.startswith("router_ids/") for k in trace.arrays)


def find_traces(root: str | Path, model: str | None = None, checkpoint: str | None = None) -> list[Path]:
    """Trace directories under ``root`` with routing records: optionally of models whose name contains ``model``, or
    of the served copies of ``checkpoint`` (models.checkpoint_of)."""
    from . import models
    of = models.checkpoint_lookup()
    out = []
    for meta in sorted(Path(root).expanduser().rglob("meta.json")):
        if meta.parent.name.startswith("."):          # a save under way or cut short (store.Trace.save)
            continue
        m = json.loads(meta.read_text())
        if model and model not in (m.get("model") or ""):
            continue
        if checkpoint and of(m.get("model")) != checkpoint:
            continue
        if m.get("num_experts") or _npz_has_router(meta.parent / "arrays.npz"):
            out.append(meta.parent)
    return out


def _npz_has_router(path: Path) -> bool:
    try:
        with np.load(path) as z:
            return any(k.startswith("router_ids/") for k in z.files)
    except (OSError, ValueError):
        return False


@dataclass
class Corpus:
    """Routed tokens pooled across traces, one row per unique context."""
    ids: np.ndarray          # [L, N, k] expert ids
    layers: list[int]
    num_experts: int
    total_positions: int     # before deduplication

    @property
    def n(self) -> int:
        return self.ids.shape[1]

    @property
    def k(self) -> int:
        return self.ids.shape[2]


def build_corpus(traces: Iterable[Trace], dedupe: bool = True) -> Corpus:
    """Pool routing over traces. With ``dedupe``, a position whose whole prefix
    (tokens 0..t) was already seen is skipped: identical contexts route
    identically, so repeats would only inflate counts."""
    seen: set[int] = set()
    ids_l = []
    layers, n_exp, total = None, 0, 0
    for tr in (t for t in traces if has_routing(t)):
        ids, valid = routing(tr)
        if layers is None:
            layers = tr.layers("router_ids")
        elif tr.layers("router_ids") != layers:
            raise ValueError("traces have different MoE layers; pool one model at a time")
        n_exp = max(n_exp, int(tr.meta.get("num_experts") or 0), int(ids.max()) + 1)
        T = ids.shape[1]
        toks = tr.tokens[-T:]                               # records are right-aligned to tokens
        keep, h = [], 0
        for t in range(T):
            h = hash((h, int(toks[t])))
            if not valid[t] or (dedupe and h in seen):
                continue
            seen.add(h)
            keep.append(t)
        total += int(valid.sum())
        if not keep:
            continue
        keep = np.array(keep)
        ids_l.append(ids[:, keep])
    if layers is None:
        raise ValueError("no traces with routing records")
    if len({a.shape[2] for a in ids_l}) > 1:
        raise ValueError("traces of different models (each records its model's own k experts a token)")
    return Corpus(np.concatenate(ids_l, 1), layers, n_exp, total)


# ---------------------------------------------------------------- co-activation embedding

def coactivation_embedding(ids: np.ndarray, num_experts: int, min_uses: int = 3,
                           method: str = "tsne", seed: int = 0) -> tuple[np.ndarray, float]:
    """2D map of one layer's experts: experts that fire on the same tokens sit close.

    ids: [N, k]. Each expert is its binary firing vector over tokens and the
    distance between experts is 1 - correlation of those vectors. "mds" is
    classical MDS (top-2 eigenvectors of the double-centred Gram): faithful
    but, on sparse data, a crushed cloud with a few outliers. "tsne" refines
    the MDS layout with exact t-SNE (numpy, deterministic): local
    neighbourhoods are trustworthy, global distances are not.
    Experts used fewer than ``min_uses`` times get NaN.
    Returns (xy [E, 2], share of variance in the 2 MDS components).
    """
    N = ids.shape[0]
    X = np.zeros((N, num_experts), np.float32)
    X[np.repeat(np.arange(N), ids.shape[1]), ids.ravel()] = 1
    used = X.sum(0) >= min_uses
    Xs = X[:, used]
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-6)
    R = Xs.T @ Xs / N                                   # correlation, D^2 = 2 - 2R
    J = np.eye(len(R)) - 1.0 / len(R)
    B = J @ R @ J
    w, v = np.linalg.eigh(B)
    w, v = w[::-1][:2], v[:, ::-1][:, :2]
    coords = v * np.sqrt(np.maximum(w, 0))
    explained = float(w.sum() / max(np.trace(B), 1e-9))
    if method == "tsne" and len(R) > 10:
        coords = _tsne(np.maximum(2 - 2 * R, 0), coords, seed=seed)
    # Fix the arbitrary sign: the busiest expert goes on the right / top.
    busiest = np.argmax(X[:, used].sum(0))
    coords = coords - coords.mean(0)
    coords *= np.where(coords[busiest] < 0, -1, 1)
    xy = np.full((num_experts, 2), np.nan)
    xy[used] = coords
    return xy, explained


def _tsne(D2: np.ndarray, init: np.ndarray, perplexity: float = 15.0, iters: int = 350, seed: int = 0) -> np.ndarray:
    """Exact t-SNE on squared distances D2 [n, n] (n ~ 500, so O(n^2) is fine)."""
    n = len(D2)
    perplexity = min(perplexity, (n - 1) / 3)
    # Per-row precision by bisection to hit the target entropy.
    lo, hi = np.full(n, 1e-6), np.full(n, 1e6)
    beta = np.ones(n)
    D2 = D2 / np.median(D2[D2 > 0])
    off = ~np.eye(n, dtype=bool)
    for _ in range(60):
        P = np.exp(-D2 * beta[:, None]) * off
        sP = P.sum(1, keepdims=True) + 1e-12
        H = np.log(sP[:, 0]) + beta * (D2 * P).sum(1) / sP[:, 0]
        big = H > np.log(perplexity)              # too flat: sharpen
        lo = np.where(big, beta, lo)
        hi = np.where(big, hi, beta)
        beta = np.where(hi < 1e6, (lo + hi) / 2, beta * 2)
    P = P / sP
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    rng = np.random.default_rng(seed)
    Y = init / (np.abs(init).std() + 1e-9) * 1e-2 + rng.normal(0, 1e-4, init.shape)
    vel, gains = np.zeros_like(Y), np.ones_like(Y)
    P = P.astype(np.float32)
    Y = Y.astype(np.float32)
    for it in range(iters):
        exag = 12.0 if it < 100 else 1.0
        sq = (Y ** 2).sum(1)
        num = 1 / (1 + np.maximum(sq[:, None] + sq[None, :] - 2 * Y @ Y.T, 0))
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / num.sum(), 1e-12)
        W = (exag * P - Q) * num
        grad = 4 * (W.sum(1)[:, None] * Y - W @ Y)
        gains = np.where(np.sign(grad) != np.sign(vel), gains + 0.2, gains * 0.8).clip(0.01)
        vel = (0.5 if it < 250 else 0.8) * vel - 200.0 * gains * grad
        Y = Y + vel
        Y -= Y.mean(0)
    return Y

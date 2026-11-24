"""Contrast statistics: is the direction real, where is it written, how far is it carried.

Per layer, the diff-of-means direction's in-sample and held-out d' (paired
folds: a matched pair's two sentences are held out together; the mean over
FOLD_REPEATS independent fold splits), the permutation test of the held-out
d' under the paired null (labels flipped within pairs, each labelling
averaged over as many splits; ``p_value``, ``perm_p95``); the writer rule
over those; the adjacent-layer cosine and the carrier band.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..vectors import Vector, dprime, unit

WRITER_ALPHA = 0.01           # the writer is the first layer whose permutation p-value (held-out d' vs held-out d' under within-pair label flips) is below this
PERMUTATIONS = 200
CARRIER_COS = 0.7             # adjacent-layer cosine that counts as "carried unchanged"
FOLD_REPEATS = 5              # paired held-out d' is the mean over this many independent fold splits (one split at 16 pairs carries ~±0.7 of seed noise)


# ---------------------------------------------------------------- contrast statistics

def _paired_split(n: int, folds: int, rng) -> np.ndarray:
    """One paired fold assignment: pair i goes to fold i mod ``folds`` after a
    seeded shuffle of the pair indices."""
    fold = np.empty(n, dtype=int)
    fold[rng.permutation(n)] = np.arange(n) % folds
    return fold


def _grams(P: np.ndarray, N: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What the paired held-out d' needs, in pair space: with D = P - N (row i
    the pair's difference), ``P @ D.T``, ``N @ D.T`` and ``D @ D.T`` ([n, n]
    each, float64). A fold's training direction is a signed sum of D rows, so
    every projection and every norm is a product with these."""
    P64, N64 = np.asarray(P, np.float64), np.asarray(N, np.float64)
    D = P64 - N64
    return P64 @ D.T, N64 @ D.T, D @ D.T


def _paired_dprimes(grams: tuple, signs: np.ndarray, splits: np.ndarray, folds: int) -> np.ndarray:
    """Held-out d' for a batch of labellings, each averaged over its fold
    splits. ``signs`` [m, n]: +1 keeps pair i's labels, -1 swaps them (the
    paired null); ``splits`` [m, r, n]: r fold assignments per labelling.
    Fold k's direction is the diff of means over the other folds' pairs under
    that labelling, proportional to sum_i s_i D_i over those pairs; its length
    is divided out (the folds' scores are pooled) and each held-out pair is
    scored on it. Returns [m]: the mean over the r splits of each split's d'
    (``vectors.dprime``, row-wise)."""
    GP, GN, GD = grams
    m, r, n = splits.shape
    out = np.zeros((m, r))
    pos_is_p = signs > 0
    for j in range(r):
        fold = splits[:, j]                                          # [m, n]
        sp, sn = np.zeros((m, n)), np.zeros((m, n))
        for k in range(folds):
            test = fold == k
            w = signs * ~test                                         # [m, n]: weight of each D row in fold k's direction
            norm = np.sqrt(np.maximum(np.einsum("mi,ij,mj->m", w, GD, w), 0.0))
            norm = np.maximum(norm, 1e-12)[:, None]
            projP, projN = (w @ GP.T) / norm, (w @ GN.T) / norm       # [m, n]: every P row and every N row on that direction
            sp = np.where(test, np.where(pos_is_p, projP, projN), sp)
            sn = np.where(test, np.where(pos_is_p, projN, projP), sn)
        pooled = np.sqrt((sp.var(axis=1, ddof=1) + sn.var(axis=1, ddof=1)) / 2) + 1e-6
        out[:, j] = (sp.mean(axis=1) - sn.mean(axis=1)) / pooled
    return out.mean(axis=1)


def _heldout_dprime(P: np.ndarray, N: np.ndarray, folds: int, rng, repeats: int = FOLD_REPEATS) -> float:
    """Out-of-fold d' on a matched corpus: row i of ``P`` and row i of ``N``
    are a pair (the same sentence with and without the concept). Pairs, not
    rows, are assigned to folds (``_paired_split``), so a held-out sentence's
    twin is held out with it. Each fold's pairs are scored on the
    diff-of-means direction built from the other folds' pairs only. The
    result is the mean over ``repeats`` independent fold splits drawn from
    ``rng``: a single split at 16 pairs moves d' by about ±0.7 from seed to
    seed."""
    n = len(P)
    if len(N) != n:
        raise ValueError(f"paired folds need as many contrast rows as concept rows ({n} vs {len(N)})")
    folds = max(2, min(folds, n))
    splits = np.stack([_paired_split(n, folds, rng) for _ in range(repeats)])[None]
    return float(_paired_dprimes(_grams(P, N), np.ones((1, n)), splits, folds)[0])


def contrast_stats(P: np.ndarray, N: np.ndarray, hc_count: int = 1, folds: int = 4,
                   permutations: int = PERMUTATIONS, seed: int = 0, repeats: int = FOLD_REPEATS) -> dict:
    """For one layer: ``P`` [n, width] concept samples, ``N`` [n, width]
    contrast samples, row i of each a matched pair. Returns the in-sample d'
    of the diff-of-means direction; the held-out d' with paired folds (a
    pair's two sentences are held out together; ``_heldout_dprime``),
    averaged over ``repeats`` fold splits (``fold_repeats``); the
    permutation test of that held-out d' under the paired null, like for
    like: the same held-out construction, each with its own ``repeats`` fold
    splits from the same rng, on ``permutations`` labellings that flip
    pos/neg within each pair with probability 1/2 (an unrestricted shuffle
    would break the pairs: the wrong null for a matched corpus; batched in ``_paired_dprimes``), its 95th
    percentile (``perm_p95``) and the one-sided
    p-value (share of labellings at least as separated, with the +1
    correction; None with ``permutations=0``); and d' per hyper-connection
    stream block. ``load_concept`` refuses corpora that are not matched pairs."""
    rng = np.random.default_rng(seed)
    d = P.mean(0) - N.mean(0)
    u = unit(d)
    insample = dprime(P @ u, N @ u)
    heldout = _heldout_dprime(P, N, folds, rng, repeats)
    # Permutation test, like for like: paired held-out d' under within-pair label flips, each labelling
    # averaged over its own ``repeats`` fold splits.
    n, k = len(P), max(2, min(folds, len(P)))
    perm = np.array([])
    if permutations:
        signs = np.where(rng.random((permutations, n)) < 0.5, -1.0, 1.0)
        splits = np.stack([np.stack([_paired_split(n, k, rng) for _ in range(repeats)]) for _ in range(permutations)])
        perm = _paired_dprimes(_grams(P, N), signs, splits, k)
    p_value = float((1 + (perm >= heldout).sum()) / (1 + len(perm))) if len(perm) else None
    streams = []
    if hc_count > 1:
        Pb, Nb = P.reshape(len(P), hc_count, -1), N.reshape(len(N), hc_count, -1)
        for s in range(hc_count):
            us = unit(Pb[:, s].mean(0) - Nb[:, s].mean(0))
            streams.append(dprime(Pb[:, s] @ us, Nb[:, s] @ us))
    return dict(dprime=insample, heldout=heldout, streams=streams, natural_norm=float(np.linalg.norm(d)),
                p_value=p_value, perm_p95=float(np.percentile(perm, 95)) if len(perm) else None,
                perm_mean=float(perm.mean()) if len(perm) else None, permutations=int(len(perm)),
                fold_repeats=int(repeats))


def adjacent_cosine(vec: Vector) -> dict[int, float]:
    """cos(u_l, u_next) for consecutive layers of the vector."""
    layers = vec.layers
    return {l: float(np.dot(vec.vectors[l], vec.vectors[n])) for l, n in zip(layers, layers[1:])}


def writer_layer(stats: dict[int, dict]) -> dict:
    """``first``: the first layer whose permutation p-value is below WRITER_ALPHA
    (None when none is, or the stats were taken without permutations);
    ``jump``: the layer with the largest rise in held-out d' from the layer
    before."""
    layers = sorted(stats)
    first = next((l for l in layers if stats[l]["p_value"] is not None and stats[l]["p_value"] < WRITER_ALPHA), None)
    jumps = {n: stats[n]["heldout"] - stats[l]["heldout"] for l, n in zip(layers, layers[1:])}
    jump = max(jumps, key=jumps.get) if jumps else None
    return dict(first=first, jump=jump, alpha=WRITER_ALPHA, rule=f"p < {WRITER_ALPHA:g}")


def peak_layer(stats: dict[int, dict]) -> Optional[int]:
    """The layer where held-out d' peaks (``stats`` int-keyed, as from
    ``config.int_keys(prof["stats"])``); None without layers."""
    return max(stats, key=lambda l: stats[l]["heldout"]) if stats else None


def reader_best(prof: dict) -> tuple[dict, dict]:
    """The profile's reader and the reader's best dose, each {} when there is none."""
    r = prof.get("reader") or {}
    return r, r.get("best") or {}


def floor_of(st: dict) -> float:
    """The floor to draw and compare against: held-out d' under shuffled
    labels, 95th percentile (``perm_p95``)."""
    return float(st["perm_p95"])


def p_text(st: dict) -> str:
    """The layer's permutation p-value as text: 'p < 0.005' when no permutation reached the true labels."""
    p, n = st["p_value"], st["permutations"]
    return f"p < {1 / (n + 1):.1g}" if p <= 1 / (n + 1) + 1e-12 else f"p = {p:.2g}"


def carrier_band(cos: dict[int, float], start: Optional[int] = None) -> Optional[dict]:
    """Longest run of consecutive layers (at or after ``start``) whose
    adjacent cosine stays at or above CARRIER_COS; None if no two layers do."""
    layers = [l for l in sorted(cos) if start is None or l >= start]
    best, run = None, []
    for l in layers + [None]:
        if l is not None and cos[l] >= CARRIER_COS and (not run or l == run[-1] + 1):
            run.append(l)
            continue
        if run and (best is None or len(run) > len(best)):
            best = list(run)
        run = [l] if l is not None and cos[l] >= CARRIER_COS else []
    if not best:
        return None
    return dict(start=best[0], end=best[-1] + 1, threshold=CARRIER_COS,
                mean_cos=float(np.mean([cos[l] for l in best])))

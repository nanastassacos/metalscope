"""The expert map's layout: each MoE layer's experts placed where ``experts.coactivation_embedding`` puts them
(experts that fire on the same tokens are neighbours). ``build_layout`` pools the
newest saved traces of a model's checkpoint (every served copy's: models.py) and keeps the result
(``~/.metalscope/expert_layouts/<checkpoint>.json``, ``layout_path``) so the map stays put from one reply to the
next (``metalscope experts layout`` builds it); ``expert_grid`` is one layer of it as Pi's dock draws it
(/scope/api/experts/grid), read only. Also a saved reply's routing as the live stream sends it
(``reply_routing``, /scope/api/traces/ID/routing): the dock's experts for a reply traced before it asked the stream
for routing.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np

from . import config, models
from .store import read_index

LAYOUT_VERSION = 2  # 2: the model's own k
GRID_COLS = 32   # Pi's dock draws a layer's experts this many lights across (two lights a line, half-height blocks)
# A layout pools the newest saved traces up to this many tokens: enough for a stable map, and a build in under half a
# minute (Flash Next, 48 layers of 512 experts: 26 s, 2 GB, mostly the t-SNE). Pooling every trace took over ten
# minutes and 40 GB on Flash Next's 5,600 traces (10M tokens, mostly calibration chats).
LAYOUT_TOKENS = 300_000


def layout_path(model: str) -> Path:
    return config.home() / "expert_layouts" / (re.sub(r"[^\w.-]+", "_", models.checkpoint_of(model)) + ".json")


def _cached_layout(model: str) -> dict | None:
    """The model's layout as last built; None when it is not built or of an older LAYOUT_VERSION."""
    p = layout_path(model)
    out = json.loads(p.read_text()) if p.exists() else {}
    return out if out.get("version") == LAYOUT_VERSION else None


def newest_within(paths: list[Path], budget: int = LAYOUT_TOKENS) -> list[Path]:
    """The newest of ``paths`` (by meta.json time) whose tokens sum to at most ``budget``; at least one."""
    out, n = [], 0
    for p in sorted(paths, key=lambda p: (Path(p) / "meta.json").stat().st_mtime, reverse=True):
        k = len(json.loads((Path(p) / "meta.json").read_text()).get("tokens") or [])
        if out and n + k > budget:
            break
        out.append(p)
        n += k
    return out


def build_layout(model: str, root: str | Path | None = None, budget: int = LAYOUT_TOKENS) -> dict:
    """Pool the newest saved traces of a model's checkpoint, every served copy's (up to ``budget`` tokens), into a
    fixed 2D layout of every layer's experts, and keep it (``layout_path``)."""
    from . import experts as X
    from .store import Trace
    ckpt = models.checkpoint_of(model)
    paths = newest_within(X.find_traces(root or config.traces_dir(), checkpoint=ckpt), budget)
    corpus = X.build_corpus(Trace.load(p) for p in paths)
    layers = []
    for li, l in enumerate(corpus.layers):
        xy, _ = X.coactivation_embedding(corpus.ids[li], corpus.num_experts)
        ok = np.isfinite(xy[:, 0])
        xy = xy - np.nanmean(xy[ok], 0) if ok.any() else xy
        xy = xy / max(1e-9, np.nanmax(np.abs(xy[ok]))) if ok.any() else xy
        layers.append(dict(layer=int(l), xy=[[round(float(a), 3), round(float(b), 3)] if o else None for (a, b), o in zip(xy, ok)]))
    out = dict(version=LAYOUT_VERSION, model=ckpt, num_experts=corpus.num_experts, k=corpus.k, tokens=corpus.n,
               traces=len(paths), layers=layers)
    p = layout_path(model)
    p.parent.mkdir(parents=True, exist_ok=True)
    config.atomic_write(p, json.dumps(out, separators=(",", ":")))
    return out


def usable_layout(model: str) -> dict | None:
    """``_cached_layout``, None for a file that does not read."""
    try:
        return _cached_layout(model)
    except (OSError, ValueError):
        return None


def grid_positions(xy: list, cols: int = GRID_COLS) -> tuple[int, list[int]]:
    """A layer's experts on a ``cols``-wide grid, neighbours on the map near each other: (rows, pos), ``pos[e]`` the
    cell (row * cols + column) of expert ``e``. Experts sorted by x fill the columns ``rows`` at a time, left to right,
    each column sorted by y, top to bottom; experts without a position (``xy[e]`` None: never routed) go last."""
    E = len(xy)
    rows = max(1, -(-E // cols))
    order = sorted((e for e in range(E) if xy[e] is not None), key=lambda e: (xy[e][0], e))
    order += [e for e in range(E) if xy[e] is None]
    pos = [0] * E
    for c in range(cols):
        col = sorted(order[c * rows:(c + 1) * rows], key=lambda e: (xy[e] is None, xy[e][1] if xy[e] is not None else 0, e))
        for r, e in enumerate(col):
            pos[e] = r * cols + c
    return rows, pos


_SIZES: dict = {}   # (index file, checkpoint) -> (its mtime and size, experts per layer): read_index once per change


def _num_experts(model: str) -> int | None:
    """Experts per layer from the newest saved reply of ``model``'s checkpoint that routed (its meta.json)."""
    ckpt = models.checkpoint_of(model)
    index = config.index_path()
    try:
        st = index.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    key = (str(index), ckpt)
    if key in _SIZES and _SIZES[key][0] == stamp:
        return _SIZES[key][1]
    found, of = None, models.checkpoint_lookup()
    for e in reversed(read_index()):
        if not e.get("experts") or of(e.get("model")) != ckpt:
            continue
        try:
            n = json.loads((Path(e["path"]) / "meta.json").read_text()).get("num_experts")
        except (OSError, ValueError, KeyError, AttributeError):
            continue
        if isinstance(n, int) and n > 0:
            found = n
            break
    _SIZES[key] = (stamp, found)
    return found


def expert_grid(model: str, layer: int, cols: int = GRID_COLS) -> dict:
    """One MoE layer (``layer``: its place among the MoE layers, from 0, as the live stream orders them) for Pi's
    dock: ``pos`` (grid_positions) from the model's kept layout; without a usable one (not
    built, or a file that does not read), ``pos`` is None, the dock draws the experts in number order,
    and ``num_experts`` comes from a saved reply. Raises ValueError for a layer out of range."""
    if layer < 0:
        raise ValueError("layer must be 0 or more")
    try:
        lay = _cached_layout(model)
        one = None if lay is None else lay["layers"][layer] if layer < len(lay["layers"]) else False
        if one is False:
            raise ValueError(f"layer must be 0 to {len(lay['layers']) - 1}")
        if one is not None:
            rows, pos = grid_positions(one["xy"], cols)
            return dict(model=lay["model"], layer=layer, n_layers=len(lay["layers"]), num_experts=int(lay["num_experts"]),
                        cols=cols, rows=rows, pos=pos)
    except ValueError as exc:
        if str(exc).startswith("layer must"):
            raise
    except (OSError, KeyError, TypeError, IndexError, AttributeError):
        pass
    ckpt = models.checkpoint_of(model)
    E = _num_experts(model)
    return dict(model=ckpt, layer=layer, n_layers=None, num_experts=E, cols=cols, rows=max(1, -(-E // cols)) if E else None,
                pos=None, note=f"no expert map for {ckpt} yet: metalscope experts layout builds one")


def reply_routing(path: str | Path, last: int | None = None) -> dict:
    """A saved reply's MoE routing in the live stream's form (livestream._routing), for Pi's dock: reply token ``i``
    (the stream's ``i``, trace position prompt_len + i) gets the forward it was fed into, every MoE layer in order.
    ``ids`` base64 of little-endian uint16 [n, L, k] and ``w`` base64 of uint8 [n, L, k] (each expert's share of its
    layer's routed mix, 0-255; 0 where nothing was recorded) for reply tokens ``start`` .. ``start + n - 1``, the last
    ``last`` of them when given. A dense model's (no router records): L = k = n = 0. Reads only the router arrays, and
    of each only the reply's rows, a layer at a time. A file that does not read raises (OSError, ValueError, KeyError,
    zipfile.BadZipFile)."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    P, n_tok = int(meta["prompt_len"]), len(meta["tokens"])
    empty = dict(start=0, n=0, L=0, k=0, ids="", w="")
    with np.load(path / "arrays.npz") as z:
        layers = sorted(int(k.split("/")[1]) for k in z.files if k.startswith("router_ids/"))
        if not layers:
            return empty
        has_probs = f"router_probs/{layers[0]}" in z.files
        # The reply's rows only, a layer at a time: a long trace's whole router arrays would be tens of MB at once.
        rows = ok = None
        sel, w = [], []
        for l in layers:
            a = z[f"router_ids/{l}"]                                        # [T, k]
            if rows is None:
                T = a.shape[0]
                off = n_tok - T                                             # records are right-aligned to the tokens
                n_reply = max(0, n_tok - P)
                start = max(0, n_reply - last) if last else 0
                rec = P + np.arange(start, n_reply) - off                   # each reply token's record
                ok = (rec >= 0) & (rec < T)
                rows = np.clip(rec, 0, max(0, T - 1))
            sel.append(a[rows].astype(np.int64))
            w.append(np.nan_to_num(z[f"router_probs/{l}"][rows].astype(np.float32)) if has_probs else np.ones(a[rows].shape, np.float32))
    sel, w = np.stack(sel, 1), np.stack(w, 1)                               # [n, L, k]
    n, L, k = sel.shape
    bad = (sel < 0) | ~ok[:, None, None]
    w = np.where(bad, 0, np.clip(w, 0, None))
    share = np.rint(255 * w / np.maximum(w.sum(-1, keepdims=True), 1e-9)).astype(np.uint8)
    sel = np.where(bad, 0, sel).astype("<u2")
    return dict(start=int(start), n=int(n), L=int(L), k=int(k),
                ids=base64.b64encode(sel.tobytes()).decode(), w=base64.b64encode(share.tobytes()).decode())

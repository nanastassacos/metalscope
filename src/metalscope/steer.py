"""Steering: add a vector to the residual while a request is served.

A request asks for it with the ``X-Scope-Steer`` header (``config.STEER_HEADER``):

    <part>[+<part>...][;option=value...]
    part     [random~]<name>@<strength>[/<layers>], or <name> alone: its calibrated dose and layer on the served
             copy, what /steer <name> uses (``with_defaults``; none calibrated is a 400), or remove~<name>/<layers>
             (the dock's edit's own)
    layers   <layer>[,<layer>...], each item a layer or a range a-b (18-44)
    options  layers=24,31            layers for every part that names none (default: the concept's steer layer,
                                     as <name> alone; a 400 when its profile has none; the vector's layers when
                                     it has no profile: a word's, one by hand)
             positions=all|decode    prompt and reply, or decode forwards only (default all)

A combination is the per-layer sum of its parts' scaled vectors (at most ``MAX_PARTS``, each name once). Each
``random~`` part draws its own Gaussian base (a seed from the request id plus the part's index, one direction for
all its layers), matched to the named vector's per-stream norms and scaled like it, so two control parts are
nearly orthogonal, like two concepts, not one random direction counted twice. ``strength`` is a multiple of the
layer's typical residual norm on the served copy's own traces (``norm_table``). Every part is constant: the same
vector at every position it covers, thinking tokens included. A push at a layer adds there and, for a vector whose
meta names a ``block`` (a concept profiled on Inkling 2), at the rest of that block's layers too, each once
(``push_layers``).

A word's vector (vectors.from_lens: ``is_words``) steers only as the dock's one-token edit (``SteerSpec.edit``, set
by edit.answer): a word changes one token, not a whole reply (``WORD_EDIT_ONLY``; held on every token it was never
measured to work on any model, and on Inkling 2 it loops or does nothing). live.TraceManager._ensure_plan refuses
any other steer naming one.

``remove~NAME/LAYERS``, the edit's removal, projects NAME's direction out of the residual at every position it
covers, before any add at that layer: h -= (w Q^T) Q, w the written residual and Q orthonormal rows spanning the
removed directions at that layer (several near-parallel ones are removed once), in float32 a row of Q at a time
(``_remove``). No dose, so no strength table. The edit's swap is a removal plus a push:
``remove~soccer/18-44+rugby@0.7/31``. The vector must be one token's (a word vector cut to its first piece is
refused). On ds4 the plan's rows go in the steering file and its kernel does the same in float32
(ds4/frames.py ``write_steering``).

The write is a post-hook on each decoder layer (hooks.py) adding ``V[:, None, :]``, row b the scaled vector for
batch row b's request and zero for the others, in the residual's dtype. Flash Next's layer returns
``(streams, branch, gate)`` with its write pending (the written residual is ``streams + branch·gate``, applied
additively later: adapters/qwen4_exp.py), so adding to ``streams`` equals adding to the written residual. DeepSeek
V4's ``[B, T, hc, D]`` streams are flattened to ``[B, T, hc*D]`` for the add and reshaped back.

On Inkling 2 a concept's vector steers whole replies over its block. Its layer axis is the visit: the hooks name a
layer call by its visit (adapters.visit_counter), and the steered residual is the layer's output, the running sum
``h`` (its norm, ``steer_norm``, is the dose unit), a word's vector its lens direction cut to the sum's columns
(adapters/inkling2.py).

Steering fails closed: any error in the hook, a batch whose size does not match this forward's rows (nothing is
ever broadcast across rows), or a failure building those rows (live.TraceManager._steer_rows) stops steering that
request and marks it ``steer_failed``. A request whose tokens were partly served unsteered (an MTP reconcile
re-prefill, omlx/patches_mtp.py) is marked ``steer_partial``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from . import config, models
from .hooks import add_hook
from .store import NORM_TRACES, norm_picks
from .vectors import NAME_RE, Vector, find_vector, is_words, load_vector, unit

log = logging.getLogger("metalscope")

MAX_STRENGTH = 4.0
MAX_PARTS = 8             # vectors in one combination
POSITIONS = ("all", "decode")
NORM_ROWS = 10_000        # of each, its last rows only (norm_table)
NORM_MIN_TOKENS = 2000    # fewer tokens in all and the table warns (norms_note): dose units from a short trace
TOMBSTONES = 4096         # finished steered request ids kept for the cache store guard
MAX_LAYER = 1024          # no model has this many layers: bounds a range (a-b)
WORD_EDIT_ONLY = "a word changes one token (the dock's LENS edit), not a whole reply"   # Pi's steer.ts says it too
# What a dose multiplies, by trace array kind, preferred first: the hyper-connection streams' norms, the norm of what a
# steer adds to where that is not the residual the head reads (Inkling 2's running sum), the residual's (norm_table).
NORM_KINDS = ("hc_stream_norm", "steer_norm", "resid_norm")



# ---------------------------------------------------------------- the request's spec

@dataclass(frozen=True)
class SteerPart:
    """One vector of a request: ``[random~]NAME@DOSE[/LAYERS]``, ``NAME`` (strength None until ``with_defaults``) or
    ``remove~NAME/LAYERS`` (no strength)."""
    name: str
    strength: Optional[float]
    control: bool = False                     # random~<name>
    layers: Optional[tuple[int, ...]] = None  # None: layers=, else the steer layer (with_defaults), else the vector's
    remove: bool = False                      # remove~<name>: projected out

    @property
    def label(self) -> str:
        """``fluffy@0.8``, ``random~fluffy@0.8`` (no layers); a removal with the layers it changes, ``remove~a/3,7-9``."""
        if self.remove:
            return f"remove~{self.name}/{_span(self.layers)}"
        return f"{'random~' if self.control else ''}{self.name}@{self.strength:g}"


@dataclass(frozen=True)
class SteerSpec:
    parts: tuple[SteerPart, ...]
    positions: str = "all"
    edit: bool = False        # an edit's own one-token steer (edit.answer): a word's only steer

    @property
    def label(self) -> str:
        """Every part: ``angry@0.4+robotic@0.8``."""
        return "+".join(p.label for p in self.parts)

    @property
    def accepted(self) -> str:
        """Every part with its layers where it has them (``angry@0.4/24+robotic@0.8``, the second at its vector's
        layers): ``X-Scope-Steer-Accepted``, so a client sees the layer a part sent without one was given."""
        return "+".join(p.label if p.remove or p.layers is None else f"{p.label}/{_span(p.layers)}" for p in self.parts)


def _layers(v: str, what: str) -> tuple[int, ...]:
    out: set[int] = set()
    for x in filter(None, (x.strip() for x in v.split(","))):
        a, dash, b = x.partition("-")
        try:
            lo, hi = int(a), int(b if dash else a)
        except ValueError:
            raise ValueError(f"bad layers {v!r} in {what}: whole numbers, none negative")
        if not 0 <= lo <= hi <= MAX_LAYER:
            raise ValueError(f"bad layers {x!r} in {what}: up to {MAX_LAYER}, a range low-high, e.g. 18-44")
        out.update(range(lo, hi + 1))
    if not out:
        raise ValueError(f"{what} needs at least one layer, none negative")
    return tuple(sorted(out))


def _span(layers: tuple[int, ...]) -> str:
    """Layers as ``_layers`` reads them, runs as ranges: ``3,7-9``."""
    runs: list[list[int]] = []
    for l in layers:
        if runs and l == runs[-1][1] + 1:
            runs[-1][1] = l
        else:
            runs.append([l, l])
    return ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in runs)


def parse_part(text: str) -> SteerPart:
    """``[random~]NAME@DOSE[/L1,L2]``, ``NAME``, ``remove~NAME/LAYERS``."""
    head = text.strip()
    if not head:
        raise ValueError("empty part: parts are joined by +, e.g. angry@0.4+robotic@0.8")
    if head.startswith("remove~"):
        name, slash, lv = head[len("remove~"):].partition("/")
        if not NAME_RE.fullmatch(name):
            raise ValueError(f"remove~ takes NAME/LAYERS, no dose, e.g. remove~soccer/18-44 (got {head!r})")
        return SteerPart(name, None, layers=_layers(lv, repr(head)) if slash else None, remove=True)
    control = head.startswith("random~")
    if control:
        head = head[len("random~"):]
    layers = None
    if "/" in head:
        head, lv = head.split("/", 1)
        layers = _layers(lv, f"{text.strip()!r}")
    if "@" not in head and not control and layers is None and NAME_RE.fullmatch(head):
        return SteerPart(head, None)                 # NAME alone: its calibrated dose and layer (``with_defaults``)
    if "@" not in head:
        raise ValueError(f"each part must be NAME or NAME@DOSE, e.g. calm or calm@0.5 (got {text.strip()!r})")
    name, s = head.rsplit("@", 1)
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"bad vector name {name!r}")
    try:
        strength = float(s)
    except ValueError:
        raise ValueError(f"bad dose {s!r}")
    if not 0.0 <= strength <= MAX_STRENGTH:                                # NaN fails too
        raise ValueError(f"the dose must be between 0 and {MAX_STRENGTH:g} (times the typical residual norm)")
    return SteerPart(name, strength, control, layers)


def parse_header(value: str) -> SteerSpec:
    """``PART[+PART...][;key=value]*`` (module docstring); raises ValueError
    with a message fit for an HTTP 400."""
    pieces = [p.strip() for p in value.strip().split(";")]
    head, opts = pieces[0], pieces[1:]
    raw = head.split("+") if head else []
    if len(raw) > MAX_PARTS:
        raise ValueError(f"at most {MAX_PARTS} parts in one steer")
    parts = [parse_part(p) for p in raw]
    seen = set()
    for p in parts:
        if (p.control, p.name) in seen:
            name = f"{'random~' if p.control else ''}{p.name}"                # not p.label: NAME alone has no dose
            raise ValueError(f"{name} appears twice; give it once (two doses of one vector add up to one)")
        seen.add((p.control, p.name))
    kw: dict = {}
    for o in opts:
        if not o:
            continue
        if "=" not in o:
            raise ValueError(f"bad option {o!r} (expected key=value)")
        k, v = (x.strip() for x in o.split("=", 1))
        if k == "layers":
            shared = _layers(v, "layers=")
            parts = [p if p.layers is not None else replace(p, layers=shared) for p in parts]
        elif k == "positions":
            if v not in POSITIONS:
                raise ValueError(f"positions must be one of {POSITIONS}")
            kw["positions"] = v
        else:
            raise ValueError(f"unknown option {k!r}")
    if not parts:
        raise ValueError("empty part: parts are joined by +, e.g. angry@0.4+robotic@0.8")
    for p in parts:
        if p.remove and p.layers is None:
            raise ValueError(f"remove~{p.name} needs its layers, e.g. remove~{p.name}/18-44")
    return SteerSpec(tuple(parts), **kw)


def with_defaults(spec: SteerSpec, model_name: str) -> SteerSpec:
    """``spec`` with every ``NAME`` part given its calibrated dose and layer on the served copy (concepts.steer_dose,
    the dose /steer NAME uses), and every ``NAME@DOSE`` part without layers its concept's steer layer
    (concepts.steer_layer) when it has one; ValueError when a ``NAME`` has no dose, or names layers without one, or
    when a part without layers names a concept whose profile has no steer layer (a control part aside; a vector with
    no profile, a word's, keeps its own layers)."""
    from .concepts import calibration_of, steer_dose, steer_layer
    parts = []
    for p in spec.parts:
        if p.remove or (p.layers is not None and p.strength is not None):
            parts.append(p)
            continue
        path = models.profile_path(model_name, p.name)
        prof, served = config.read_json(path), config.short_name(model_name)
        layer = steer_layer(prof)
        # A learned concept's vector has every layer, and all of them at once is word salad at any dose.
        if p.layers is None and layer is None and path.exists() and not p.control:
            raise ValueError(f"{p.name} has no steer layer on {served}: give its layers, e.g. "
                             f"{p.name}@{p.strength or 0.3:g}/24")
        if p.strength is not None:
            parts.append(p if layer is None else replace(p, layers=(layer,)))
            continue
        if p.layers is not None:
            raise ValueError(f"{p.name} without a dose steers at its calibrated layer; give {p.name}@DOSE/LAYERS to "
                             "pick layers")
        dose = steer_dose(prof, calibration_of(model_name, p.name)) if isinstance(prof, dict) else None
        if dose is None:
            raise ValueError(f"{p.name} has no calibrated dose on {served}: give one, e.g. {p.name}@0.3, or run "
                             f"metalscope concept calibrate {p.name} {served}")
        parts.append(SteerPart(p.name, dose["strength"], False, (dose["layer"],)))
    return SteerSpec(tuple(parts), positions=spec.positions, edit=spec.edit)


def parse_edit(value: str, model_name: str) -> dict:
    """``X-Scope-Edit: RID@AT STEER`` (``config.EDIT_HEADER``): an earlier traced reply (its request id) answered again
    from its token AT (0 the reply's first) with STEER applied at that one position, a steer header of parts only
    (e.g. ``remove~soccer/18-44+rugby@0.7/31``). Its parts given their defaults (``with_defaults``); ValueError with
    a message fit for an HTTP 400. ``check_spec`` is the caller's."""
    head, _, rest = value.strip().partition(" ")
    rid, at_sign, at = head.rpartition("@")
    if not at_sign or not rid or not at.isdigit() or not rest.strip():
        raise ValueError("an edit is RID@TOKEN STEER, e.g. 3f2a@57 remove~soccer/18-44")
    spec = with_defaults(parse_header(rest), model_name)
    if spec.positions != "all":
        raise ValueError("an edit's steer is parts only: it acts at the one token")
    return dict(rid=rid, at=int(at), steer=replace(spec, edit=True))


def stable_seed(request_id: str) -> int:
    """Python's hash() is salted per process; the control's default seed must
    be reproducible from the saved request id."""
    return int.from_bytes(hashlib.sha1(request_id.encode()).digest()[:4], "big")


# ---------------------------------------------------------------- strength table

def norm_table(model_name: str, entries: Optional[list[dict]] = None, n_traces: int = NORM_TRACES,
               refresh: bool = False) -> dict:
    """Typical residual norm per layer for the served copy ``model_name``, from
    its own most recent unsteered saved traces:
    ``{"model": served name, "checkpoint", "kind", "layers": {l: median}, "streams":
    {l: [per-stream medians]} (hyper-connection models), "hc_count", "traces":
    [ids], "tokens" (the rows measured, at most NORM_ROWS a trace, for ``norms_note``), "made"}``. Another served
    copy of the same checkpoint (another engine or quantisation) has a table of its own, never this one.

    For hyper-connection models the vector lives in stream space, so the norm
    is the total stream norm (root of the summed squared stream norms) per
    token, not the mixed residual's; on Inkling 2 the running sum's
    (``steer_norm``), not the head's mix. The kind is the first of NORM_KINDS
    any picked trace has, and traces without it are left out (``kind``: never
    two kinds in one table). Cached in the served copy's norms.json
    (models.norms_path); ``refresh`` recomputes. Raises ValueError when the
    served copy has no usable traces of its own.

    It runs inside the server on the first steered request, so each trace's norm arrays alone are read, a layer at a
    time, and only their last NORM_ROWS rows kept (a whole Flash Next trace at 262k tokens is 1.3 GB, its stream
    norms 200 MB). That is enough: the rows before a reused prefix's ``read_from`` are empty anyway, and 20 traces
    of 10,000 rows give each layer's median from 200,000 tokens. A trace of NORM_ROWS rows or fewer is read whole."""
    cache = models.norms_path(model_name)
    data = None if refresh else _cached_norms(cache)
    if data is None:
        with _MEASURING:
            data = None if refresh else _cached_norms(cache)      # another request measured it meanwhile
            if data is None:
                data = _measure_norms(model_name, cache, entries, n_traces)
    return data


_MEASURING = threading.Lock()   # two first steers arriving together (each checked on a worker thread) measure once


def _cached_norms(cache) -> Optional[dict]:
    try:
        data = json.loads(cache.read_text())
        data["layers"] = config.int_keys(data["layers"])
        data["streams"] = config.int_keys(data.get("streams", {}))
        return data
    except (OSError, ValueError, KeyError):
        return None


def _measure_norms(model_name: str, cache, entries: Optional[list[dict]], n_traces: int) -> dict:
    """norm_table's measurement, written to ``cache``."""
    from .store import read_index
    short = config.short_name(model_name)
    picked = norm_picks(read_index() if entries is None else entries, short, n_traces)
    has = []                                                  # each picked trace's kinds (its file list only)
    for e in picked:
        try:
            with np.load(f"{e['path']}/arrays.npz") as z:
                has.append({f.split("/")[0] for f in z.files})
        except (OSError, ValueError, KeyError):
            has.append(set())
    kind = next((k for k in NORM_KINDS if any(k in h for h in has)), NORM_KINDS[-1])
    totals: dict[int, list] = {}
    streams: dict[int, list] = {}
    used, tokens = [], 0
    for e, h in zip(picked, has):
        if kind not in h:                                     # another kind's units: never mixed in
            continue
        got, n = {}, 0
        try:
            with np.load(f"{e['path']}/arrays.npz") as z:
                for l in sorted(int(f.split("/")[1]) for f in z.files if f.startswith(kind + "/")):
                    got[l] = np.array(z[f"{kind}/{l}"][-NORM_ROWS:], np.float32)   # a copy: the whole array goes
        except (OSError, ValueError, KeyError):
            continue
        if not got:
            continue
        for l, a in got.items():
            if kind == "hc_stream_norm":                                          # [T, S]
                a = a[np.isfinite(a).all(-1)]
                totals.setdefault(l, []).append(np.sqrt((a ** 2).sum(-1)))
                streams.setdefault(l, []).append(a)
            else:
                totals.setdefault(l, []).append(a[np.isfinite(a)])
            n = max(n, len(totals[l][-1]))
        used.append(e["id"])
        tokens += n
    if not totals:
        raise ValueError(f"no saved unsteered traces of {short} to measure residual norms on (each served copy "
                         f"has its own); trace a few replies first (metalscope on), then `metalscope vector norms {short}`")
    data = dict(
        model=short, checkpoint=models.checkpoint_of(short), kind=kind, traces=used, tokens=tokens,
        made=time.strftime("%Y-%m-%dT%H:%M:%S"),
        layers={l: float(np.median(np.concatenate(v))) for l, v in totals.items()},
        streams={l: np.median(np.concatenate(v), axis=0).tolist() for l, v in streams.items()},
        hc_count=int(next(iter(streams.values()))[0].shape[-1]) if streams else 1,
    )
    try:
        config.atomic_write(cache, json.dumps({**data, "layers": {str(k): v for k, v in data["layers"].items()},
                                     "streams": {str(k): v for k, v in data["streams"].items()}}, indent=1))
    except OSError:
        pass
    return data


def norms_note(table: Optional[dict]) -> Optional[str]:
    """A warning when the strength table came from few or short traces (fewer than NORM_MIN_TOKENS tokens in all), so
    dose units may be off; None when it looks measured enough. It never refuses: the table is used as it is."""
    if not isinstance(table, dict):
        return None
    n, tokens = len(table.get("traces") or []), table.get("tokens")
    if not isinstance(tokens, int) or tokens >= NORM_MIN_TOKENS:
        return None
    what = f"{n} short trace{'' if n == 1 else 's'} ({tokens} tokens)"
    return (f"dose units on {table.get('model')} measured on {what}: `metalscope vector norms {table.get('model')} "
            "--refresh` after a real chat")


class NotReady(ValueError):
    """A steer the served copy cannot take yet though the header is fine: no strength table (no unsteered traces)."""


def _one_token(name: str, meta) -> None:
    """ValueError for a word vector cut to its first token (vectors.from_lens ``cut``): a removal takes a whole word."""
    if isinstance(meta, dict) and meta.get("cut"):
        raise ValueError(f"{name} is more than one token: a removal takes a one-token word")


def push_layers(meta, layers) -> list[int]:
    """The layers a push at ``layers`` adds at, each once, in order: each layer and, for a vector whose meta names a
    ``block`` (concepts.profile on Inkling 2: its 6 visits, aligned from layer 0), the rest of that layer's block.
    Any other vector (a word's, every other model's): ``layers`` themselves."""
    k = int(meta.get("block") or 1) if isinstance(meta, dict) else 1
    return sorted({m for l in layers for m in range(l, (l // k + 1) * k)})


def _vector_meta(model_name: str, name: str, path) -> dict:
    """A stored vector's sidecar (its layers, ``block``, ...), the vector's own meta when that is unreadable."""
    meta = config.read_json(path.with_suffix(".json"))
    if isinstance(meta, dict) and isinstance(meta.get("layers"), list):
        return meta
    vec = load_vector(model_name, name)
    return dict(vec.meta, layers=vec.layers)


def check_words(spec: SteerSpec, model_name: str) -> None:
    """ValueError(WORD_EDIT_ONLY) when a steer that is not an edit's own (``spec.edit``) names a word's vector
    (vectors.is_words): check_spec's early 400, and live.TraceManager._ensure_plan's fail-closed backstop."""
    if spec.edit:
        return
    for part in spec.parts:
        path = find_vector(model_name, part.name)
        if path is not None and is_words(_vector_meta(model_name, part.name, path)):
            raise ValueError(WORD_EDIT_ONLY)


def check_spec(spec: SteerSpec, model_name: str) -> None:
    """What the middleware can check before the request is scheduled (400 on failure), so a steer that could not be
    built is refused rather than served unsteered under the echoed header: no word's vector outside an edit
    (``check_words``), every part has a stored vector with the layers it names (a removal's, of one token), and the
    served copy has a strength table (``norm_table``) with all those layers (NotReady when it has no unsteered traces
    yet), a push's whole span (``push_layers``)."""
    check_words(spec, model_name)
    model = config.short_name(model_name)
    layers_of: dict[str, list[int]] = {}
    for part in spec.parts:
        path = find_vector(model_name, part.name)
        if path is None:
            raise ValueError(f"no steering vector {part.name!r} for model {model_name!r} (metalscope vector list {model})")
        meta = _vector_meta(model_name, part.name, path)
        have = [int(l) for l in meta["layers"]]
        want = list(part.layers) if part.layers is not None else have
        if not part.remove:
            want = push_layers(meta, want)
        missing = [l for l in want if l not in have]
        if missing:
            dose = part.strength if part.strength is not None else 0.3
            eg = f"remove~{part.name}/" if part.remove else f"{part.name}@{dose:g}/"
            raise ValueError(f"{part.name} has no layer {', '.join(map(str, missing))} on {model} (it has "
                             f"{', '.join(map(str, have))}): give one of those, e.g. {eg}{have[0] if have else 0}")
        if part.remove:                                       # no dose: no strength table
            _one_token(part.name, meta)
        else:
            layers_of[part.name] = want
    if not layers_of:
        return
    try:
        table = norm_table(model_name)
    except ValueError:
        raise NotReady(f"doses are sized from {model}'s own unsteered replies, and none are traced yet: chat a few "
                       f"messages without a steer first, with tracing on (metalscope on), about {NORM_MIN_TOKENS} "
                       "tokens in all, then steer again") from None
    for what, want in layers_of.items():
        no_norm = [l for l in want if l not in table["layers"]]
        if no_norm:
            raise NotReady(f"no residual norm for {what} at layer {', '.join(map(str, no_norm))} in {model}'s traces: "
                           f"run `metalscope vector norms {model} --refresh` after a traced chat, then steer again")


# ---------------------------------------------------------------- the plan for one request

@dataclass
class SteerPlan:
    """What to add where, for one request. ``vectors``: layer -> [width]
    float32, every part scaled and summed, added at every covered position.
    ``remove``: per layer, the rows to project out (``remove_rows``), built
    once per request."""
    vectors: dict[int, np.ndarray]
    positions: str
    meta: dict
    remove: dict[int, mx.array] = field(default_factory=dict)    # layer -> Q [r, width] float32, orthonormal rows

    @property
    def width(self) -> int:
        return int(next(iter(self.vectors.values())).shape[0])


def control_direction(vec: Vector, layers: list[int], seed: int) -> dict[int, np.ndarray]:
    """Unit directions per layer for ``random~<name>``: one Gaussian base
    direction per request (aligned across layers), matching ``vec``'s
    per-stream norm profile at each layer."""
    z = np.random.default_rng(seed).standard_normal(vec.width).astype(np.float32)
    out = {}
    for l in layers:
        g = z.copy()
        hc = vec.hc_count
        if hc > 1:
            blocks = g.reshape(hc, -1)
            want = vec.stream_norms(l)                              # sums (in squares) to 1
            have = np.linalg.norm(blocks, axis=-1)
            g = (blocks * (want / np.maximum(have, 1e-12))[:, None]).reshape(-1)
        out[l] = unit(g)
    return out


def remove_rows(dirs: list[np.ndarray]) -> mx.array:
    """Orthonormal rows ``Q`` [r, width] spanning the unit directions removed at one layer: two near-parallel words are
    removed once, not twice (within 1e-3 of the largest singular value: about 2e-3 rad apart). ValueError when one is
    not finite or zero."""
    D = np.stack(dirs).astype(np.float64)
    if not np.isfinite(D).all():
        raise ValueError("a direction to remove is not finite")
    if (np.linalg.norm(D, axis=1) < 0.5).any():      # unit rows
        raise ValueError("a direction to remove is zero")
    _, s, vt = np.linalg.svd(D, full_matrices=False)
    return mx.array(vt[s > 1e-3 * s[0]].astype(np.float32))


def build_plan(spec: SteerSpec, model_name: str, request_id: str, hc_count: int = 1,
               jacobians: Optional[dict] = None, table: Optional[dict] = None) -> SteerPlan:
    """Resolve a header spec into scaled vectors: each part at its layers (a
    push over its whole span, ``push_layers``), summed per layer.
    Raises ValueError for anything the request cannot be served with (unknown
    vector, layers, no norm table). ``meta`` lists every part (``parts``);
    its top-level name/control/strength/seed/abs_norm keep their single-vector
    meaning (for a combination: names joined by +, strength None, abs_norm the
    parts' caps summed per layer)."""
    table = table if table is not None else norm_table(model_name)
    seed = stable_seed(request_id)
    summed: dict[int, np.ndarray] = {}
    parts_at: dict[int, int] = {}
    caps: dict[int, float] = {}
    removed: dict[int, list[np.ndarray]] = {}
    parts_meta = []
    for i, part in enumerate(spec.parts):
        vec = load_vector(model_name, part.name)
        if vec.hc_count != hc_count:
            raise ValueError(f"vector {part.name} was made for hc_count={vec.hc_count}, model has {hc_count}")
        layers = list(part.layers) if part.layers is not None else vec.layers
        if not part.remove:
            layers = push_layers(vec.meta, layers)
        missing = [l for l in layers if l not in vec.vectors]
        if missing:
            raise ValueError(f"vector {part.name} has no layers {missing} (has {vec.layers})")
        pseed = (seed + i) % 2 ** 32           # one Gaussian base per control part
        dirs, abs_norm = {}, {}
        if part.remove:
            _one_token(part.name, vec.meta)
            for l in layers:
                removed.setdefault(l, []).append(vec.vectors[l])
        else:
            no_norm = [l for l in layers if l not in table["layers"]]
            if no_norm:
                raise ValueError(f"no residual norm for layers {no_norm} in the traces of {model_name}")
            dirs = control_direction(vec, layers, pseed) if part.control else {l: vec.vectors[l] for l in layers}
            abs_norm = {l: float(part.strength * table["layers"][l]) for l in layers}
        for l in dirs:
            add = (abs_norm[l] * dirs[l]).astype(np.float32)
            summed[l] = add if l not in summed else summed[l] + add
            caps[l] = caps.get(l, 0.0) + abs_norm[l]
            parts_at[l] = parts_at.get(l, 0) + 1
        readout = {l: float(mx.linalg.norm(jacobians[l].astype(mx.float32) @ mx.array(dirs[l])).item())
                   for l in dirs if jacobians and l in jacobians}
        parts_meta.append(dict(
            label=part.label, name=part.name, control=part.control, remove=part.remove, strength=part.strength, layers=layers,
            seed=pseed if part.control else None, abs_norm=abs_norm,
            stream_norms={l: vec.stream_norms(l).tolist() for l in layers} if hc_count > 1 else None,
            readout_norm=readout or None, natural_norm=vec.meta.get("natural_norm"), source=vec.meta.get("source")))
    remove = {l: remove_rows(ds) for l, ds in sorted(removed.items())}
    layers = sorted(caps)
    one = parts_meta[0] if len(parts_meta) == 1 else None
    meta = dict(label=spec.label, parts=parts_meta,
                positions=spec.positions, layers=layers, parts_at=parts_at,
                name="+".join(p["label"].split("@")[0] for p in parts_meta),
                control=bool(spec.parts) and all(p.control for p in spec.parts),
                strength=one["strength"] if one else None,
                seed=seed if any(p.control for p in spec.parts) else None,
                abs_norm=caps, norm_table={l: table["layers"][l] for l in layers}, norm_traces=table.get("traces"),
                removed={l: int(q.shape[0]) for l, q in remove.items()})         # directions projected out per layer
    if one:
        meta.update(name=one["name"], stream_norms=one["stream_norms"], readout_norm=one["readout_norm"],
                    natural_norm=one["natural_norm"], source=one["source"])
    return SteerPlan(summed, spec.positions, meta, remove)


# ---------------------------------------------------------------- the hooks

class SteerSession:
    """Hooks on every decoder layer of one model that add the rows' vectors.

    ``set_rows`` is called before each forward with the (row, request id, plan)
    triples of that forward, its batch size and its phase ("prefill" or
    "decode"); it precomputes per layer one ``[B, width]`` array, row b the
    vector of the request in batch row b (zero for unsteered rows) and each
    removal's rows (the plan's, already on the GPU). ``clear`` after the
    forward. ``on_fail(request_id, reason)`` is called when the hook cannot
    apply a vector; that request is no longer steered this forward and the
    caller marks it.

    A plan's layers are the trace's: a layer call is named by ``adapters.visit_counter`` (Inkling 2: its visit, a
    count of its own kept on every call; else the layer's index)."""

    def __init__(self, on_fail: Callable[[str, str], None]):
        self.on_fail = on_fail
        self._removers: list[Callable[[], None]] = []
        self.rows: dict[int, tuple[str, SteerPlan]] = {}
        self.phase = "decode"
        self.B = 0
        self._V: dict[int, mx.array] = {}
        self._R: dict[int, list] = {}                # removals: layer -> [(row, Q [r, width])]

    def install(self, model) -> None:
        from .adapters import visit_counter
        from .adapters.generic import text_model
        visit = visit_counter(model)
        for i, layer in enumerate(text_model(model).layers):
            self._removers.append(add_hook(layer, self._hook(i, visit), lambda exc: self._fail(str(exc))))

    def close(self) -> None:
        for r in self._removers:
            r()
        self._removers.clear()
        self.clear()

    def set_rows(self, rows: list[tuple[int, str, SteerPlan]], n_rows: int, phase: str) -> None:
        self.rows = {row: (rid, plan) for row, rid, plan in rows}
        self.B, self.phase = int(n_rows), phase
        self._V, self._R = {}, {}
        if not rows:
            return
        for row, *_ in rows:
            if not 0 <= row < self.B:
                self._fail(f"row {row} outside a batch of {self.B}")
                return
        applies_now = [(row, p) for row, _, p in rows if p.positions == "all" or phase == "decode"]
        for row, p in applies_now:
            for l, Q in p.remove.items():
                self._R.setdefault(l, []).append((row, Q))
        for l in sorted({l for _, p in applies_now for l in p.vectors}):
            applies = [(row, p) for row, p in applies_now if l in p.vectors]
            width = applies[0][1].width
            V = np.zeros((self.B, width), np.float32)
            for row, p in applies:
                if p.vectors[l].shape[0] != width:
                    self._fail(f"layer {l}: vectors of widths {p.vectors[l].shape[0]} and {width} in one batch")
                    return
                V[row] += p.vectors[l]
            self._V[l] = mx.array(V)

    def clear(self) -> None:
        self.rows, self._V, self._R = {}, {}, {}
        self.B = 0

    def _fail(self, reason: str) -> None:
        self._V, self._R = {}, {}
        for rid, _ in self.rows.values():
            try:
                self.on_fail(rid, reason)
            except Exception:
                log.exception("metalscope: steer failure callback failed")

    # -- removal ------------------------------------------------------------
    @staticmethod
    def _written(out, h: mx.array) -> mx.array:
        """The residual the layer wrote (a pending write's part included), in ``h``'s dtype."""
        if isinstance(out, tuple) and len(out) == 3:
            branch, gate = (x.astype(h.dtype) for x in out[1:])
            return h + (branch[..., None, :] * gate[..., None]).reshape(h.shape)
        return h

    @staticmethod
    def _rows_of(out, h: mx.array, row: int):
        """``out`` and ``h`` cut to one batch row (pending-write parts too)."""
        if isinstance(out, tuple) and len(out) == 3:
            return (h[row:row + 1], out[1][row:row + 1], out[2][row:row + 1]), h[row:row + 1]
        return h[row:row + 1], h[row:row + 1]

    def _remove(self, out, h: mx.array, R: list) -> mx.array:
        """Each removal row's directions projected out of the residual it wrote (a pending write's part included), in
        float32 and cast back once: h - (w Q^T) Q. Other rows are left exactly as they are. A row of Q at a time,
        elementwise: Metal's float32 matmul left 0.1 of a 100 coordinate 10240 wide, this 1e-5 (M-series, 2026-11)."""
        rows = [h[i:i + 1] for i in range(self.B)]
        for row, Q in R:
            if Q.shape[1] != h.shape[-1]:
                raise ValueError(f"removal width {Q.shape[1]} does not match the residual {tuple(h.shape)}")
            o, hr = self._rows_of(out, h, row)
            x = hr.astype(mx.float32)
            w = self._written(o, x)
            for q in Q:
                x = x - (w * q).sum(-1, keepdims=True) * q
            rows[row] = x.astype(h.dtype)
        return mx.concatenate(rows, 0)

    # -- the hook ----------------------------------------------------------
    def _hook(self, i: int, visit: Callable[[int], int]):
        return lambda _m, _a, _k, out: self._apply(visit(i), out)

    def _apply(self, l: int, out):
        V, R = self._V.get(l), self._R.get(l)
        if V is None and R is None:
            return None
        try:
            h = out[0] if isinstance(out, tuple) else out
            if not isinstance(h, mx.array) or h.ndim not in (3, 4):
                raise ValueError(f"layer {l}: unexpected residual {getattr(h, 'shape', type(h))}")
            shape = h.shape
            h = h.reshape(*shape[:2], -1)                                      # [B, T, hc, D] streams: [B, T, hc*D]
            if h.shape[0] != self.B:
                raise ValueError(f"layer {l}: residual {tuple(h.shape)} does not match a batch of {self.B} steer rows")
            if R is not None:
                h = self._remove(out, h, R)                                    # before any add at this layer
            if V is not None:
                if h.shape[-1] != V.shape[1]:
                    raise ValueError(f"layer {l}: residual {tuple(h.shape)} does not match steer rows {tuple(V.shape)}")
                h = h + V[:, None, :].astype(h.dtype)
            h = h.reshape(shape)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        except Exception as exc:
            log.warning("metalscope: steering stopped: %s", exc)
            self._fail(str(exc))
            return None

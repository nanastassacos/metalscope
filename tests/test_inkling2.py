"""Inkling 2's adapter (adapters/inkling2.py) on a tiny stand-in with the call structure of oMLX 0.8.3's model
(omlx/patches/inkling2/model.py, vlm.py): a text model whose ``_schedule`` visits its core layers three times (each
later pass restarting from the prelude's output, handed the pass before's blocks and sum as its bank), layers called as
``layer(h, blocks, cache=, bank=, tau=)`` that append a block's unit-RMS incoming sum to ``blocks`` and return its
running sum, learned mixes of the blocks and the sum (``final_input``, the head's), a second half that runs
``mx.compile``d on short steps with the MoE inside it (row-exact verify rows: a compiled call of their own), an MoE
routed by ``route`` (its logits by ``_row_matmul``) as ``kernels.route_topk`` does, a head ``logits_from_hidden``, and
a ``language_model`` that is not a module, served through a wrapper like oMLX's VLMModelAdapter. Its cache is laid out
by ``visit_caches``: a sliding visit one entry (its key window, then its conv states), a full-attention visit a KVCache
and its conv states; the full-attention visits share one ``tau`` from the first one's cache offset; each visit reads its
own cache (a causal sum over the positions so far, a short conv). Like model.py, this module has the ``_unit_rms``,
``kernels`` and ``_COMPILED_MAX_TOKENS`` a fit swaps, and the row-exact flag (``is_row_exact_armed``)."""

import sys
import weakref
from collections import Counter
from types import SimpleNamespace as NS

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from metalscope import adapters
from metalscope.adapters import generic, inkling2
from metalscope.capture import CaptureConfig, CaptureSession
from metalscope.mtp import window_starts

D, V, E, K = 32, 64, 16, 4
N_LAYERS, CORE, PASSES, BLOCK = 8, (2, 6), 3, 2
FULL = (3, 7)              # the full-attention layers (the rest sliding): one in the loop's core, one after it
CONV = 2                   # a sliding visit's state: its keys and values, then its conv states (oMLX's cache.py)
SCHEDULE = tuple([(i, 0) for i in range(CORE[0])] + [(i, p) for p in range(PASSES) for i in range(*CORE)]
                 + [(i, 0) for i in range(CORE[1], N_LAYERS)])   # 16 visits: 0 1 | 2 3 4 5 | 2 3 4 5 | 2 3 4 5 | 6 7
VISITS = len(SCHEDULE)
# The sources of each visit's state: its finished blocks but the first (the token's embedding) and its sum, then in a
# later pass the bank's block and sum.
SOURCES = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4]
CONFIG = dict(model_type="inkling2_loop", text_config=dict(
    hidden_size=D, num_hidden_layers=N_LAYERS, n_layers_per_attn_res_block=BLOCK,
    loop_config=dict(core_start=CORE[0], core_end=CORE[1], num_passes=PASSES)))
_COMPILED_MAX_TOKENS = 16
kernels = NS(mx=mx)                        # what oMLX's kernels.py calls mlx.core
SHORT = mx.array([[3, 17, 5, 42, 9]])                                    # <= 16 tokens: the compiled second half
LONG = mx.array([[(7 * i + 3) % V for i in range(20)]])                  # > 16: plain
RECORD = [False]
ROUTED = []        # each MoE call's (ids, weights), in call order, while RECORD[0]
ROW_EXACT = [False]          # oMLX's row-exact flag (qwen35_verify_qmm), armed around MTP verify forwards


def is_row_exact_armed():
    return ROW_EXACT[0]


def _unit_rms(x, eps=1e-6):
    return mx.fast.rms_norm(x, None, eps)


def _row_matmul(x, w):
    """``x @ w``; armed, one single-row product a row, as model.py's."""
    rows = x.shape[-2]
    if rows > 1 and is_row_exact_armed():
        batch = np.broadcast_shapes(x.shape[:-2], w.shape[:-2])
        wb = mx.broadcast_to(w[..., None, :, :], (*batch, rows, *w.shape[-2:]))
        return (x[..., :, None, :] @ wb).squeeze(-2)
    return x @ w


class AttnResQuery(nn.Module):
    """A learned softmax mix of the blocks and the sum, as kernels.attn_res_mix's plain path; its Metal kernel (on the
    GPU) has no gradient."""

    def __init__(self):
        super().__init__()
        self.query = 0.5 * mx.random.normal((1, D))
        self.scale = D ** -0.5

    def __call__(self, blocks, partial_n):
        src = mx.stack([*blocks, partial_n], axis=-2)
        w = mx.softmax((src @ self.query.T) * self.scale, axis=-2, precise=True)
        out = w.swapaxes(-1, -2) @ src
        return out if kernels.mx.default_device() != mx.gpu else mx.stop_gradient(out)


class Inkling2MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k, self.route_scale = K, 2.5
        self.router_weight = mx.random.normal((E, D))
        self.router_bias = 0.3 * mx.random.normal((E,))
        self.out = nn.Linear(E, D, bias=False)

    def route(self, x):                         # _row_matmul, then kernels.route_topk's plain path
        scores = mx.sigmoid(_row_matmul(x.astype(mx.float32), self.router_weight.T))
        inds = mx.argpartition(-(scores + self.router_bias), kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        w = mx.take_along_axis(scores, inds, axis=-1)
        return inds, w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20) * self.route_scale

    def __call__(self, x):                      # float32 out: the caller rounds it, as oMLX's
        flat = x.reshape(-1, x.shape[-1])
        inds, w = self.route(flat)
        if RECORD[0]:
            ROUTED.append((inds, w))
        mixed = mx.put_along_axis(mx.zeros((flat.shape[0], E)), inds, w, axis=-1)
        return self.out(mixed).reshape(x.shape)


class Inkling2Attention(nn.Module):
    """Each position reads a causal sum over the positions so far, its visit's cache included: a full-attention visit's
    keys (its KVCache), a sliding visit's sum so far (its state's key slot); a full-attention visit's scaled by ``tau``,
    the log scaling of its positions (from its KVCache's offset)."""

    def __init__(self, idx):
        super().__init__()
        self.sliding = idx not in FULL
        self.proj = nn.Linear(D, D, bias=False)

    def log_tau(self, kv, length):
        if self.sliding:
            return None
        offset = kv.offset if kv is not None else 0
        return (1.0 + 0.5 * mx.log(mx.arange(length) + offset + 1.0)).reshape(1, length, 1)

    def __call__(self, x, kv, state, tau):
        length = x.shape[1]
        if kv is not None:
            x = kv.update_and_fetch(x[:, None], x[:, None])[0][:, 0]
        ctx = mx.cumsum(x, axis=1)[:, -length:]
        if kv is None and state is not None:
            if state[0] is not None:
                ctx = ctx + state[0]
            state[0] = ctx[:, -1:]
        ctx = ctx / D
        return self.proj(ctx if tau is None else ctx * tau)


class Inkling2Layer(nn.Module):
    """As oMLX's: the attention reads a mix of the finished blocks and the incoming sum's unit-RMS (layer 0 the sum
    itself); a block's first layer appends that unit-RMS sum to ``blocks`` and starts its own sum from the attention
    output, so it reads its incoming sum only through unit-RMS (what a fit's restart relies on); a later pass adds the
    bank's blocks and its sum, each through a map of its own. Its cache is ``(KVCache or None, state)``, the conv state
    (a position also reads the one before it) at ``CONV`` in a sliding visit's state, at 0 in a full-attention one's."""

    def __init__(self, idx):
        super().__init__()
        self.layer_idx, self.is_boundary = idx, idx % BLOCK == 0
        self.attn_input = AttnResQuery()
        self.attn = Inkling2Attention(idx)
        self.sliding = self.attn.sliding
        self.bank_blocks = nn.Linear(D, D, bias=False)
        self.bank_sum = nn.Linear(D, D, bias=False)
        self.mlp_input = AttnResQuery()
        self.mlp_norm = nn.RMSNorm(D)
        self.mlp = Inkling2MoE()
        layer = weakref.proxy(self)
        self._post_compiled = mx.compile(lambda *a: Inkling2Layer._post(layer, *a))
        self._post_exact = mx.compile(lambda *a: Inkling2Layer._post(layer, *a))

    def _post(self, out, gate, h, mlp_blocks):
        residual = out if self.is_boundary else h + out
        y = self.mlp_input(mlp_blocks, _unit_rms(residual.astype(mx.float32)))[:, :, 0]
        m = self.mlp(self.mlp_norm(y).astype(h.dtype))
        return (residual + gate * m).astype(h.dtype)

    def __call__(self, h, blocks, cache=None, bank=None, tau=None):
        kv, state = cache if cache is not None else (None, None)
        base = CONV if kv is None else 0
        length = h.shape[1]
        if tau is None:
            tau = self.attn.log_tau(kv, length)
        h_n = _unit_rms(h.astype(mx.float32))
        x = h if self.layer_idx == 0 else self.attn_input(blocks, h_n)[:, :, 0]
        if bank is not None:                                    # stands in for the gated messages
            x = x + sum(self.bank_blocks(b) for b in bank[0]) + self.bank_sum(bank[1])
        raw = self.attn(x.astype(h.dtype), kv, state, tau)
        before = state[base] if state is not None and state[base] is not None else mx.zeros_like(raw[:, :1])
        out = raw + 0.5 * mx.concatenate([before, raw[:, :-1]], axis=1)
        if state is not None:
            state[base] = raw[:, -1:]
        if self.is_boundary:
            blocks.append(h_n)
        compiled = length <= _COMPILED_MAX_TOKENS
        if compiled and length > 1 and is_row_exact_armed():
            post = self._post_exact
        else:
            post = self._post_compiled if compiled else self._post
        return post(out, mx.array(0.5), h, blocks)


class Inkling2TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = NS(hidden_size=D, n_layers_per_attn_res_block=BLOCK, num_hidden_layers=N_LAYERS,
                       loop_config=CONFIG["text_config"]["loop_config"])     # TextArgs' fields read here
        self.embed = nn.Embedding(V, D)
        self.embed_norm = nn.RMSNorm(D)
        self.layers = [Inkling2Layer(i) for i in range(N_LAYERS)]
        self.final_input = AttnResQuery()
        self.norm = nn.RMSNorm(D)
        self._schedule = SCHEDULE
        self._core_start = CORE[0]

    def cache_layout(self):
        layout, entry = [], 0
        for idx, _ in self._schedule:
            layout.append((None, entry) if self.layers[idx].sliding else (entry, entry + 1))
            entry += 1 if self.layers[idx].sliding else 2
        return layout

    def visit_caches(self, cache):
        layout = self.cache_layout()
        if len(cache) != layout[-1][1] + 1:
            raise ValueError(f"Inkling 2 needs {layout[-1][1] + 1} cache entries for its {len(layout)} layer visits, "
                             f"got {len(cache)}")
        return [(None if kv is None else cache[kv], cache[state]) for kv, state in layout]

    def __call__(self, inputs, cache=None, fail_at=None, grab=None):
        h = self.embed_norm(self.embed(inputs))
        visits = [None] * len(self._schedule) if cache is None else self.visit_caches(cache)
        blocks, prelude, bank, tau = [], None, None, None
        for visit, ((idx, loop_pass), c) in enumerate(zip(self._schedule, visits)):
            layer = self.layers[idx]
            if idx == self._core_start and prelude is None:
                prelude = (h, list(blocks))
            elif idx == self._core_start:
                bank = (blocks[len(prelude[1]) + 1:], _unit_rms(h.astype(mx.float32)))
                h, blocks = prelude[0], list(prelude[1])
            if visit == fail_at:
                raise RuntimeError("a forward that stops part-way")
            if not layer.sliding and tau is None:
                tau = layer.attn.log_tau(c[0] if c is not None else None, h.shape[1])
            live = bank if loop_pass else None
            h = layer(h, blocks, cache=c, bank=live, tau=None if layer.sliding else tau)
            if grab is not None:
                grab.append((list(blocks), h, live))
        x = self.final_input(blocks, _unit_rms(h.astype(mx.float32)))[:, :, 0]
        return self.norm(x).astype(h.dtype)


class _LanguageModel:                    # not a module (oMLX's vlm.py): "lm_head" in it raises TypeError
    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inkling2TextModel()
        self.lm_head = nn.Linear(D, V, bias=False)

    @property
    def language_model(self):
        return _LanguageModel(self)

    def logits_from_hidden(self, h):
        logits = self.lm_head(h).astype(mx.float32)
        return mx.tanh(logits / 30.0) * 30.0 + mx.where(mx.arange(V) < V - 4, 0.0, -mx.inf)

    def __call__(self, inputs, cache=None, **kw):
        return self.logits_from_hidden(self.model(inputs, cache, **kw))

    def make_cache(self):                # a sliding visit's state, a full-attention visit's KVCache and conv states
        cache = []
        for kv, _ in self.model.cache_layout():
            cache += [ArraysCache(7)] if kv is None else [KVCache(), ArraysCache(4)]
        return cache


class Served(nn.Module):                 # oMLX's VLMModelAdapter: what the engine hands its scheduler
    def __init__(self, vlm):
        super().__init__()
        self._vlm_model = vlm

    def __call__(self, input_ids, cache=None, **kw):
        return self._vlm_model(input_ids, cache, **kw)

    def make_cache(self):
        return self._vlm_model.make_cache()


@pytest.fixture
def served():
    mx.random.seed(0)
    s = Served(Model())
    mx.eval(s._vlm_model.parameters())      # a private attribute: not in s.parameters(); a route's thread reads them
    yield s
    ROUTED.clear()
    RECORD[0] = ROW_EXACT[0] = False


def run(session, served, ids, **kw):
    """One forward's records: {(kind, layer): value} and how often each key came."""
    served(ids, **kw)
    mx.eval(session.pending_arrays())
    session.end_step()
    recs = session.drain()
    return {(r.kind, r.layer): r.value for r in recs}, Counter((r.kind, r.layer) for r in recs)


def visits_state(served, ids):
    """Each visit's (head mix, state), by an unhooked forward: the mix of the blocks and the layer's sum, and the
    blocks past the first, the sum, the bank's blocks and its sum, side by side."""
    grab, tm = [], served._vlm_model.model
    served(ids, grab=grab)
    out = []
    for blocks, h, bank in grab:
        mix = tm.final_input(blocks, _unit_rms(h.astype(mx.float32)))[:, :, 0].astype(h.dtype)
        parts = [*blocks[1:], h] + ([*bank[0], bank[1]] if bank is not None else [])
        out.append((mix, mx.concatenate([p.astype(mx.float32) for p in parts], axis=-1)))
    return out


def test_it_matches_inkling_and_generic_does_not_win(served):
    from tests.helpers import tiny_llama
    assert adapters.find_adapter(served) is inkling2 and adapters.find_adapter(served._vlm_model) is inkling2
    assert generic.matches(served)                          # generic would take it, and fail as it did live:
    with pytest.raises(TypeError):
        generic.install(CaptureSession(CaptureConfig(logits_topk=3)), served)
    assert adapters.find_adapter(tiny_llama()) is generic
    assert adapters.layer_count(served) == VISITS and adapters.layer_count(tiny_llama()) == 3
    # A concept's push spans its block; its steer layer is the prelude's first visit or the last pass's; the answer is
    # opened.
    assert adapters.steer_block(served) == BLOCK and adapters.steer_layers(served) == [0, CORE[0] + (PASSES - 1) * 4] == [0, 10]
    assert SCHEDULE[10] == (CORE[0], PASSES - 1) and adapters.answer_marks(served) == inkling2.ANSWER_MARKS
    assert adapters.steer_block(tiny_llama()) is None and adapters.steer_layers(tiny_llama()) is None
    assert adapters.answer_marks(tiny_llama()) is None
    # A generic model: a steer names a layer call by its index, every lens column, the workspace band.
    assert [adapters.visit_counter(tiny_llama())(i) for i in (0, 2, 1)] == [0, 2, 1]
    assert adapters.lens_columns(tiny_llama()) is None
    assert adapters.edit_band(tiny_llama(), 48) == jlens_band(48) == (18, 44)


def jlens_band(n):
    from metalscope import jlens
    return jlens.workspace_band(n)


MINI = NS(layers=[], _schedule=tuple([(i, 0) for i in range(6)] + [(i, p) for p in range(3) for i in range(6, 18)]
                                     + [(i, 0) for i in range(18, 24)]),
          args=NS(hidden_size=2304, n_layers_per_attn_res_block=6, num_hidden_layers=24,
                  loop_config=dict(core_start=6, core_end=18, num_passes=3)))


def test_a_steer_writes_the_running_sum_its_columns_and_band_by_visit(served):
    """The running sum's columns in each visit's state (``lens_columns``): source layer // k, on Inkling 2 Mini's
    schedule and the stand-in's, where they hold the layer's output ``h``; a lens not as wide as a visit's state (one
    with the token's embedding block in) is refused. The edit band: the last block's fitted visits, 42-46 of the
    Mini's 48, 14 of the stand-in's 16."""
    cols, mini = inkling2.lens_columns(MINI), [1] * 6 + [2] * 6 + [3] * 6 + ([4] * 6 + [5] * 6) * 2 + [4] * 6
    want = [0] * 6 + [1] * 6 + [2] * 6 + [1] * 6 + [2] * 6 + [1] * 6 + [2] * 6 + [3] * 6
    assert [cols(v, n * 2304) for v, n in enumerate(mini)] == [slice(k * 2304, (k + 1) * 2304) for k in want]
    with pytest.raises(ValueError, match="not as wide as the model's state at visit 44"):
        cols(44, 5 * 2304)                                       # 4 sources there; 5 with the embedding block in
    assert inkling2.edit_band(MINI) == (42, 46)
    assert inkling2.edit_band(served) == adapters.edit_band(served, VISITS) == (14, 14)
    cols = adapters.lens_columns(served)
    grab, tm = [], served._vlm_model.model
    served(LONG, grab=grab)
    assert len(grab) == VISITS
    for v, (blocks, h, bank) in enumerate(grab):
        _, st = inkling2.state(tm, blocks, h, bank)
        assert st.shape[-1] == SOURCES[v] * D
        np.testing.assert_array_equal(np.array(st[..., cols(v, st.shape[-1])]), np.array(h.astype(mx.float32)),
                                      err_msg=str(v))


def test_unembed_reads_the_head_rows_and_the_final_norms_gain(served):
    """Inkling 2's own unembed (generic's cannot: oMLX's language_model is no module, and the embedding is ``embed``):
    the head's rows, a quantised head's dequantised row by row, and the final norm's gain."""
    from metalscope.tracer import ModelTracer
    m = served._vlm_model
    rows, gain = ModelTracer(served, "inkling-test", None).unembed([5, 9])
    np.testing.assert_array_equal(np.array(rows), np.array(m.lm_head.weight[mx.array([5, 9])]))
    np.testing.assert_allclose(np.array(gain), np.array(m.model.norm.weight), rtol=1e-5)
    m.lm_head = nn.QuantizedLinear.from_linear(m.lm_head, group_size=32, bits=4)
    rows, _ = adapters.unembed(served, [5, 9])
    whole = mx.dequantize(m.lm_head.weight, m.lm_head.scales, m.lm_head.biases, group_size=32, bits=4)
    np.testing.assert_allclose(np.array(rows), np.array(whole[mx.array([5, 9])]), rtol=1e-6)


def test_a_steer_session_acts_at_its_visit_only(served):
    """Removals and adds land on the visit a plan names, not every visit of its layer: layer 2 runs at visits 2, 6 and
    10, and a plan at 6 changes ``h`` there only (row 0 of 2; row 1 as it was). The capture installed beside it
    records the steered state, the two visit counts in step."""
    from metalscope import steer
    q = np.array(mx.random.normal((D,), key=mx.random.key(7)))
    q /= np.linalg.norm(q)
    ids = mx.concatenate([LONG, mx.array([[(5 * i + 1) % V for i in range(20)]])], axis=0)
    plain = []
    served(ids, grab=plain)
    for plan, check in (
            (steer.SteerPlan({}, "all", {}, remove={6: mx.array(q[None])}),
             lambda h0, h1: (h0 - (h0 @ q)[..., None] * q, np.abs(h1 @ q).max() < 1e-4)),
            (steer.SteerPlan({6: 3 * q}, "all", {}), lambda h0, h1: (h0 + 3 * q, True))):
        s = steer.SteerSession(lambda rid, why: pytest.fail(why))
        s.install(served)
        session = CaptureSession(CaptureConfig())
        adapters.install(session, served)
        s.set_rows([(0, "r", plan)], 2, "decode")
        got = []
        recs, _ = run(session, served, ids, grab=got)
        s.close()
        session.close()
        for v in range(VISITS):
            h0, h1 = np.array(plain[v][1]), np.array(got[v][1])
            np.testing.assert_allclose(h1[1], h0[1], rtol=1e-6, atol=1e-6, err_msg=str(v))     # the other row
            if v < 6:
                np.testing.assert_allclose(h1[0], h0[0], rtol=1e-6, atol=1e-6, err_msg=str(v))  # visit 2 too: same layer
        want, ok = check(np.array(plain[6][1])[0], np.array(got[6][1])[0])
        assert ok
        np.testing.assert_allclose(np.array(got[6][1])[0], want, atol=1e-5)
        assert np.abs(np.array(got[10][1])[0] @ q).max() > 1e-2                            # visit 10: not projected
        mix, _ = inkling2.state(served._vlm_model.model, *got[6])
        np.testing.assert_allclose(recs[("resid_norm", 6)], np.linalg.norm(np.array(mix), axis=-1), rtol=1e-4)


def test_the_visit_count_follows_the_schedule_and_rights_itself():
    calls = [i for i, _ in SCHEDULE]
    vis = inkling2._Visits(SCHEDULE)
    assert [vis.done(i) for i in calls] == list(range(VISITS))
    assert [vis.done(i) for i in calls] == list(range(VISITS))          # again after a forward that completed
    for i in calls[:5]:                                                 # one that raised at visit 5
        vis.done(i)
    assert vis.current(0) == 0                                          # layer 0's MoE, inside the next forward
    assert [vis.done(i) for i in calls] == list(range(VISITS))
    assert [v for v, i in enumerate(calls) if i == 2] == [2, 6, 10]     # a core layer: three visits


def test_every_visit_recorded_once_at_its_own_index(served):
    """The residual recorded at a visit is the head's own mix of the blocks and the sum there, what the final norm
    and head would read had the model stopped after that visit."""
    want = visits_state(served, LONG)
    session = CaptureSession(CaptureConfig(router=True, logits_topk=3, logit_lens_topk=3))
    assert adapters.install(session, served) == "inkling2" and session.n_layers == VISITS
    recs, seen = run(session, served, LONG)
    session.close()
    for kind in ("resid_norm", "lens_ids", "router_ids"):
        assert {l: seen[(kind, l)] for l in range(VISITS)} == {l: 1 for l in range(VISITS)}, kind
    for v, (mix, _) in enumerate(want):
        np.testing.assert_allclose(recs[("resid_norm", v)], np.linalg.norm(np.array(mix), axis=-1), rtol=1e-4)
        lens = served._vlm_model.logits_from_hidden(served._vlm_model.model.norm(mix))
        np.testing.assert_array_equal(recs[("lens_ids", v)][..., 0], np.array(mx.argmax(lens, axis=-1)))
    # A core layer's three visits are three different states.
    assert len({float(recs[("resid_norm", v)].sum()) for v in (2, 6, 10)}) == 3
    np.testing.assert_array_equal(recs[("logit_ids", -1)][..., 0], np.array(mx.argmax(served(LONG), axis=-1)))


@pytest.mark.parametrize("ids", [SHORT, LONG])
def test_each_visit_records_its_running_sums_norm_the_dose_unit(served, ids):
    """``steer_norm`` at each visit is the norm of the layer's output, the running sum a steer adds to (the dose unit,
    steer.norm_table), beside ``resid_norm`` (the head's mix); on decode-sized steps (stacked) and prompt chunks alike.
    Not recorded when the residual norms are not (a lens fit's scoring)."""
    grab = []
    served(ids, grab=grab)
    session = CaptureSession(CaptureConfig())
    adapters.install(session, served)
    recs, seen = run(session, served, ids)
    session.close()
    for v, (_, h, _) in enumerate(grab):
        assert seen[("steer_norm", v)] == 1, v
        np.testing.assert_allclose(recs[("steer_norm", v)], np.linalg.norm(np.array(h.astype(mx.float32)), axis=-1),
                                   rtol=1e-4, err_msg=str(v))
    off = CaptureSession(CaptureConfig(residual_norm=False))
    adapters.install(off, served)
    recs, _ = run(off, served, ids)
    off.close()
    assert not any(kind == "steer_norm" for kind, _ in recs)


def test_the_mlx_engine_reads_and_steers_by_visit(served):
    """The learn job's engine on Inkling 2: its layers are the visits (the trace's), a layer run three times a token is
    read at the visit asked, and an add at a visit lands there alone (not at the same layer's other visits)."""
    from metalscope import vectors
    from metalscope.engines.mlx import MlxEngine
    engine = MlxEngine(served, Tok(), "inkling-test")
    assert (engine.shape.n_layers, engine.shape.width, engine.shape.hc) == (VISITS, D, 1)
    assert engine.block == BLOCK and engine.steer_layers == [0, 10] and engine.tok.answer == inkling2.ANSWER_MARKS
    grab, ids = [], LONG[0].tolist()
    served(LONG, grab=grab)
    mean, last = engine.residuals([ids], [2, 6, 10])                      # layer 2's three visits
    for j, v in enumerate((2, 6, 10)):
        np.testing.assert_allclose(last[0, j], np.array(grab[v][1][0, -1]), rtol=1e-5, atol=1e-5, err_msg=str(v))
    assert not np.allclose(last[0, 0], last[0, 2], atol=1e-3)
    add = 3.0 * vectors.unit(np.random.default_rng(4).standard_normal(D))
    _, steered = engine.residuals([ids], [2, 6, 10], adds={10: add})
    np.testing.assert_allclose(steered[0, :2], last[0, :2], rtol=1e-5, atol=1e-5)    # visits 2 and 6: untouched
    np.testing.assert_allclose(steered[0, 2], last[0, 2] + add, rtol=1e-4, atol=1e-4)
    from metalscope.hooks import is_hooked
    assert not any(is_hooked(l) for l in served._vlm_model.model.layers)


def test_a_forced_answer_opens_the_answer_channel_after_the_template():
    """Inkling 2's template has no thinking switch and no generation prompt: thinking off appends the answer's
    opening, so the next token is the answer's first, and the answer's end stops a reply. Other templates: the
    switch, as before."""
    from metalscope.engines.mlx import MlxTokenizer

    class Hf:
        eos_token_id, kw = 9, []

        def apply_chat_template(self, turn, add_generation_prompt=True, tokenize=True, **kw):
            Hf.kw.append(kw)
            return [1, 2, 3]

        def encode(self, text, add_special_tokens=False):
            return {inkling2.ANSWER_MARKS[0]: [7, 8], inkling2.ANSWER_MARKS[1]: [6]}.get(text) or [ord(c) for c in text]
    ink = MlxTokenizer(Hf(), inkling2.ANSWER_MARKS)
    assert ink.chat("hi", thinking=False) == [1, 2, 3, 7, 8] and Hf.kw[-1] == {}
    assert ink.chat("hi", thinking=None) == [1, 2, 3] and ink.eos == {9, 6}
    plain = MlxTokenizer(Hf())
    assert plain.chat("hi", thinking=False) == [1, 2, 3] and Hf.kw[-1] == {"enable_thinking": False}
    assert plain.eos == {9}


def test_router_rows_land_on_their_visit_on_compiled_steps_too(served):
    """Decode-sized steps run the second half compiled, and row-exact verify rows a compiled half of their own
    (``_post_exact``): while traced both run plain, so each MoE call is recorded at its visit, the rule's logits made
    as the router's (``_row_matmul``: armed, a row at a time). Untraced, and once the session closes, compiled."""
    layers = served._vlm_model.model.layers
    compiled = [(l.__dict__["_post_compiled"], l.__dict__["_post_exact"]) for l in layers]
    plain = served(SHORT)                                   # compiles the second halves, untraced
    ROW_EXACT[0] = True
    served(SHORT)                                           # and the row-exact ones
    ROW_EXACT[0] = False
    session = CaptureSession(CaptureConfig(router=True))
    adapters.install(session, served)
    rule = session.routing
    assert (rule.name, rule.k, rule.num_experts, rule.exact) == ("inkling2", K, E, True)
    assert sorted(rule.blocks) == list(range(VISITS)) and rule.blocks[6] is layers[2]["mlp"]
    session.set_rows([])                                    # nothing traced: the compiled halves run
    RECORD[0] = True
    for armed in (False, True):
        ROW_EXACT[0] = armed
        served(SHORT)
    assert ROUTED == []
    session.set_rows(None)
    for armed in (False, False, True, True):                # twice each: the count starts again
        ROW_EXACT[0] = armed
        ROUTED.clear()
        recs, _ = run(session, served, SHORT)
        assert len(ROUTED) == VISITS
        for v, (t_ids, t_w) in enumerate(ROUTED):
            ids, w = recs[("router_ids", v)], recs[("router_probs", v)]
            order, t_order = np.argsort(ids, -1), np.argsort(np.array(t_ids).reshape(ids.shape), -1)
            np.testing.assert_array_equal(np.take_along_axis(ids, order, -1),
                                          np.take_along_axis(np.array(t_ids).reshape(ids.shape), t_order, -1))
            np.testing.assert_allclose(np.take_along_axis(w, order, -1),
                                       np.take_along_axis(np.array(t_w).reshape(w.shape), t_order, -1), rtol=1e-5)
    ROW_EXACT[0] = False
    np.testing.assert_allclose(np.array(served(SHORT)), np.array(plain), atol=1e-4)     # the plain half: same output
    session.close()
    assert [(l.__dict__["_post_compiled"], l.__dict__["_post_exact"]) for l in layers] == compiled


def test_a_forward_that_raises_leaves_the_next_one_right(served):
    want = visits_state(served, SHORT)
    session = CaptureSession(CaptureConfig(router=True))
    adapters.install(session, served)
    RECORD[0] = True
    with pytest.raises(RuntimeError):
        served(SHORT, fail_at=5)
    session.end_step()
    session.drain()                                         # its part-way records, dropped
    ROUTED.clear()
    recs, seen = run(session, served, SHORT)
    session.close()
    assert len(ROUTED) == VISITS
    assert {l: seen[("router_ids", l)] for l in range(VISITS)} == {l: 1 for l in range(VISITS)}
    for v, (mix, _) in enumerate(want):
        np.testing.assert_allclose(recs[("resid_norm", v)], np.linalg.norm(np.array(mix), axis=-1), rtol=1e-4)
        first = np.array(ROUTED[v][0]).reshape(-1, K)[0]                 # the first token's experts
        assert sorted(first.tolist()) == sorted(recs[("router_ids", v)][0, 0].tolist())


def test_the_lens_reads_logits_from_hidden_of_the_final_norm(served):
    session = CaptureSession(CaptureConfig())
    adapters.install(session, served)
    m = served._vlm_model
    h = mx.random.normal((2, 3, D)).astype(mx.bfloat16)
    want = m.logits_from_hidden(m.model.norm(h.astype(mx.float32)).astype(h.dtype))
    assert mx.allclose(session.readout(h), want).item()
    assert not mx.allclose(session.readout(h), m.lm_head(h).astype(mx.float32), atol=1e-3).item()
    session.close()


def test_lens_sources_follow_the_loop():
    """The state's sources at each visit, from config.json alone (the fit's sizing): Inkling 2 Mini's, the stand-in's,
    one pass without a loop_config, and a ``loop_num_passes`` that sets the passes (as oMLX's TextArgs reads it)."""
    mini = dict(model_type="inkling2_loop", text_config=dict(num_hidden_layers=24, n_layers_per_attn_res_block=6,
                                                             loop_config=dict(core_start=6, core_end=18, num_passes=3)))
    assert inkling2.lens_sources(mini) == [1] * 6 + [2] * 6 + [3] * 6 + ([4] * 6 + [5] * 6) * 2 + [4] * 6
    two = dict(mini, text_config=dict(mini["text_config"], loop_num_passes=2))
    assert inkling2.lens_sources(two) == [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6 + [4] * 6
    assert inkling2.lens_sources(CONFIG) == SOURCES
    assert inkling2.lens_sources(dict(num_hidden_layers=12, n_layers_per_attn_res_block=6)) == [1] * 6 + [2] * 6
    assert adapters.lens_sources(mini) == inkling2.lens_sources(mini)
    assert adapters.lens_sources(dict(model_type="llama", num_hidden_layers=4)) is None


def test_the_j_lens_reads_each_visits_state(served):
    """A J-lens [D, n * D] at every visit reads that visit's state (its blocks but the first, its sum, the bank in a
    later pass), on a decode step (one deferred flush, the widths differing by visit) and on a prompt chunk. In the
    first block's visits the state is the sum alone, as wide as the head's mix: the lens reads the sum."""
    m = served._vlm_model
    J = {v: 0.3 * mx.random.normal((D, n * D), key=mx.random.key(v)) for v, n in enumerate(SOURCES)}
    for ids in (SHORT, LONG):
        want = visits_state(served, ids)
        assert [st.shape[-1] for _, st in want] == [n * D for n in SOURCES]
        assert not np.allclose(np.array(want[0][1]), np.array(want[0][0]), atol=1e-3)      # the sum is not the mix
        session = CaptureSession(CaptureConfig(jlens=J, jlens_topk=5))
        adapters.install(session, served)
        recs, _ = run(session, served, ids)
        session.close()
        for v, (_, st) in enumerate(want):
            lens = m.logits_from_hidden(m.model.norm(st @ J[v].T))
            np.testing.assert_array_equal(recs[("jlens_ids", v)][..., 0], np.array(mx.argmax(lens, axis=-1)), err_msg=str(v))


def test_the_fit_forward_switches_the_kernels_only_while_it_runs(served, monkeypatch):
    """A fit's forward runs with the kernels on their plain paths, the layers on their uncompiled halves, the sums'
    unit-RMS at its frozen scale and oMLX's rerouted expert lookup (a Metal kernel on the copy its engine loads) off,
    as seen from inside a layer at each of its visits; everything is as served before and right after, including after
    a forward that raises, the lookup's switch back to "not asked yet". The forward is the model's own loop up to the
    final norm, and sets each visit's probe width."""
    from metalscope.hooks import add_hook
    me, tm = sys.modules[__name__], served._vlm_model.model
    lookup = NS(_nax_host=None)
    monkeypatch.setitem(sys.modules, "omlx.patches.m5_gather_qmm", lookup)
    real = (me._unit_rms, me.kernels.mx, me._COMPILED_MAX_TOKENS)
    served_now = lambda: (me._unit_rms, me.kernels.mx, me._COMPILED_MAX_TOKENS) == real and lookup._nax_host is None
    seen = []
    # The kernels' device is asked only of the fit's stand-in for mlx.core (as served it is the machine's: the GPU, or
    # the CPU where a test run has no Metal).
    remove = add_hook(tm.layers[3], lambda *_: seen.append((
        me.kernels.mx is mx, None if me.kernels.mx is mx else me.kernels.mx.default_device() == mx.cpu,
        me._COMPILED_MAX_TOKENS, me._unit_rms is real[0], lookup._nax_host)))
    try:
        with inkling2.differentiable(served) as (probes, forward):
            assert served_now()                                      # entering switches nothing
            x = forward(LONG)
            assert seen == [(False, True, 0, False, False)] * 3      # layer 3's three visits, each switched
            assert served_now()
            assert probes.shapes == {v: (n * D, mx.float32) for v, n in enumerate(SOURCES)}
            seen.clear()
            with pytest.raises(ValueError, match="cache entries"):
                forward(LONG, cache=[])                              # raises reading its visits' caches
            assert served_now()
            tm(LONG)                                                 # the model's own call between passes: as served
            assert seen == [(True, None, 16, True, None)] * 3
    finally:
        remove()
    # The model's own values to the bit (a frozen unit-RMS's value is the real one's; LONG runs the halves plain).
    np.testing.assert_array_equal(np.array(tm.norm(x)), np.array(tm(LONG)))


class Tok:
    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]

    def decode(self, ids):
        return " ".join(map(str, ids))


def frozen_unit_rms(x, eps=1e-6):
    """The sums' unit-RMS with its scale held fixed for the gradient, as a fit takes it; other widths as served."""
    if x.shape[-1] != D:
        return mx.fast.rms_norm(x, None, eps)
    return x * mx.stop_gradient(mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps))


N_PRE = (CORE[0] - 1) // BLOCK + 1          # the blocks finished when the core starts (the prelude's)


def rest_of_model(tm, v, n_blocks, has_bank):
    """The head's input from the state after visit v, its sources as inputs (blocks, sum, bank blocks, bank sum): the
    schedule continued from v + 1, the loop's restart written out. A restart starts the next pass from the state's
    first N_PRE blocks, hands its first layer blocks[N_PRE] (the prelude sum's unit-RMS, which is all that layer reads
    of its incoming sum) and makes the bank the pass's later blocks and its unit-RMS sum."""
    first_core = SCHEDULE.index((CORE[0], 0))

    def rest(*src):
        blocks, h = list(src[:n_blocks]), src[n_blocks]
        bank = (list(src[n_blocks + 1:-1]), src[-1]) if has_bank else None
        for w in range(v + 1, VISITS):
            idx, loop_pass = SCHEDULE[w]
            if idx == CORE[0] and w != first_core:
                bank = (blocks[N_PRE + 1:], _unit_rms(h.astype(mx.float32)))
                h, blocks = blocks[N_PRE], blocks[:N_PRE]
            h = tm.layers[idx](h, blocks, cache=None, bank=bank if loop_pass else None)
        return [tm.final_input(blocks, _unit_rms(h.astype(mx.float32)))[:, :, 0]]
    return rest


def reference_jacobian(tm, ids, v, skip):
    """J_v from ``rest_of_model``, reduced as fit.py's estimator does: each output dim's cotangent on every valid
    position (so the sum over the later ones), the mean over valid source positions; the first block's is left out,
    as the state leaves it. [D, n * D]."""
    grab = []
    tm(mx.array([ids]), grab=grab)
    blocks, h, bank = grab[v]
    src = [*blocks, h] + ([*bank[0], bank[1]] if bank is not None else [])
    T = len(ids)
    valid = np.zeros(T, bool)
    valid[skip:T - 1] = True
    cot = np.zeros((D, T, D), np.float32)
    cot[np.arange(D)[:, None], np.nonzero(valid)[0][None, :], np.arange(D)[:, None]] = 1
    _, grads = mx.vjp(rest_of_model(tm, v, len(blocks), bank is not None), [mx.repeat(s, D, axis=0) for s in src],
                      [mx.array(cot)])
    return np.concatenate([np.array(g)[:, valid].mean(1) for g in grads[1:]], axis=-1)


def test_the_fit_is_the_jacobian_of_the_rest_of_the_model_at_every_visit(served, monkeypatch):
    """A fit takes every visit but the last as a source, J_v as wide as visit v's state, and J_v is the Jacobian of
    the rest of the model from that state (``rest_of_model``, the sums' unit-RMS scale held fixed): in the prelude, at
    each pass's first and last visit, inside the passes and after the loop. The bank at a pass's last visit gets
    nothing (the restart builds a new one). A deep window's forward starts from its prefix's cache, read visit by visit
    as the model reads it (``visit_caches``, the shared ``tau`` at the cache's offset): its J is the full sequence's,
    over the window's positions. Float32 sums, before the lens's float16."""
    from metalscope.jlens.fit import fit, no_index_gradients
    no_index_gradients()                     # as the fit child does: routing indices take no gradient
    deep = dict(ids=[(7 * i + 3) % V for i in range(20)], window=8)
    prompts = ["5 9 2 7 1 3 8 4 6 2 9 9 1", "3 3 7 1 8 2 5 5 4 7 1 2", deep]
    last = {}
    lens = fit(served, Tok(), prompts, dim_batch=D, skip_first=2, log=lambda *_: None,
               on_prompt=lambda sums, done: last.update({v: np.array(s) / done for v, s in sums.items()}))
    assert lens.layers == list(range(VISITS - 1)) and lens.meta["adapter"] == "inkling2"
    assert [last[v].shape for v in lens.layers] == [(D, n * D) for n in SOURCES[:-1]]
    tm, me = served._vlm_model.model, sys.modules[__name__]
    monkeypatch.setattr(me, "_unit_rms", frozen_unit_rms)
    monkeypatch.setattr(me, "_COMPILED_MAX_TOKENS", 0)
    monkeypatch.setattr(me.kernels, "mx", inkling2._PlainMx())       # the mix's plain path: on the GPU its kernel has no gradient
    for v in lens.layers:
        ref = np.mean([reference_jacobian(tm, Tok().encode(p), v, 2) for p in prompts[:2]]
                      + [reference_jacobian(tm, deep["ids"], v, len(deep["ids"]) - deep["window"])], axis=0)
        # float32 on the GPU rounds its own way: entries agree to about 0.1%; a wrong restart or bank is off by its size
        np.testing.assert_allclose(last[v], ref, rtol=1e-2, atol=2e-3 * np.abs(ref).max(), err_msg=f"visit {v}")
        assert np.abs(ref).max() > 0
    for v in (9, 13):                        # a later pass's last visit: its two bank sources (block, sum) get nothing
        assert SCHEDULE[v] == (CORE[1] - 1, v // 4 - 1) and (last[v][:, -2 * D:] == 0).all()
    assert np.abs(last[8][:, -2 * D:]).max() > 0 and np.abs(last[12][:, -2 * D:]).max() > 0


def test_concepts_steer_whole_replies_and_a_word_steers_as_the_edit(served, home, monkeypatch):
    from metalscope import jlens, steer, vectors
    from metalscope.learn import command
    from metalscope.live import MANAGER, RequestTrace
    from metalscope.tracer import ModelTracer
    from tests.helpers import scope_client
    tracer = ModelTracer(served, "inkling-test", Tok())
    assert tracer._load_lens() is None                      # no lens yet: logit lens
    assert tracer.layer_count() == VISITS and max(tracer._capture_config().logit_lens_layers) == VISITS - 1
    # A lens as /jlens learn leaves it, J [D, n * D] a visit: the tracer reads it.
    jlens.save(jlens.Lens({v: 0.3 * mx.random.normal((D, n * D), key=mx.random.key(v)) for v, n in enumerate(SOURCES[:-1])},
                          dict(checkpoint="inkling-test")), jlens.lenses_dir("inkling-test") / "inkling-test-learned")
    setup = tracer._load_lens()
    assert setup.lens.layers == list(range(VISITS - 1)) and setup.lens.d_model == D
    # A concept (its vector profiled with the block) steers a whole reply over its block; the J-lens's learn, use and
    # layers run.
    MANAGER.tracers[id(served)] = tracer
    MANAGER.norm_tables["inkling-test"] = {"layers": {10: 1.0, 11: 3.0, 14: 2.0}}
    monkeypatch.setattr(command, "model_path_for", lambda name, admin=None: None)
    client = scope_client()
    try:
        calm = vectors.Vector("calm", {v: vectors.unit(np.random.default_rng(v).standard_normal(D)) for v in range(VISITS)},
                              dict(hc_count=1, block=BLOCK))
        vectors.save_vector("inkling-test", calm)
        rt = RequestTrace("r1", "on", {}, steer_spec=steer.parse_header("calm@0.3/10"))
        MANAGER._ensure_plan(rt, tracer)
        assert rt.steer_failed is None and sorted(rt.steer.vectors) == [10, 11]
        np.testing.assert_allclose(rt.steer.vectors[11], 0.9 * vectors.load_vector("inkling-test", "calm").vectors[11],
                                   rtol=1e-5)
        r = client.get("/scope/api/lens/learn", params=dict(model="inkling-test", until="07:30"))
        assert r.status_code == 200, r.json()
        r = client.post("/scope/api/lens/use", json=dict(model="inkling-test", name="inkling-test-learned"))
        assert r.status_code == 200 and r.json()["lens"] == "inkling-test-learned"
        tracer.lens = setup                       # the lens read: the dock's lens steps, and the band the edit acts over
        r = client.get("/scope/api/lens/layers", params=dict(model="inkling-test"))
        assert r.status_code == 200 and r.json()["edit_band"] == [14, 14], r.json()
        # The edit's word vectors: each visit's lens direction cut to the running sum's columns, made unit, H wide.
        for w in ("5", "9"):
            r = client.post("/scope/api/steer/words", json=dict(model="inkling-test", name=f"w{w}", words=[w]))
            assert r.status_code == 200 and r.json()["band"] == [14, 14], r.json()
        vec, m = vectors.load_vector("inkling-test", "w5"), served._vlm_model
        assert vec.layers == setup.lens.layers and vec.width == D and vec.hc_count == 1
        u, cols = m.lm_head.weight[5] * m.model.norm(mx.ones((1, D)))[0], inkling2.lens_columns(served)
        J = {v: np.array(setup.lens.jacobians[v].astype(mx.float32)) for v in vec.layers}
        for v in vec.layers:
            d = np.array(u[None] @ mx.array(J[v]))[0][cols(v, J[v].shape[1])]
            np.testing.assert_allclose(vec.vectors[v], d / np.linalg.norm(d), rtol=1e-4, atol=1e-6, err_msg=str(v))
        # A lens of another layout (the token's embedding block in: one source more at every visit) is refused.
        wide = {v: mx.zeros((D, (n + 1) * D)) for v, n in enumerate(SOURCES[:-1])}
        mx.eval(list(wide.values()))              # read in the route's thread: materialised in this one
        tracer.lens = jlens.LensSetup(jlens.Lens(wide), setup.workspace)
        r = client.post("/scope/api/steer/words", json=dict(model="inkling-test", name="w7", words=["7"]))
        assert r.status_code == 400 and "not as wide as the model's state at visit 0" in r.json()["detail"], r.json()
        tracer.lens = setup
        # The edit's own one-token steer (a swap: w5 out, w9 in) is planned, its readout through the cut Jacobian.
        parts = (steer.SteerPart("w5", None, layers=(14,), remove=True), steer.SteerPart("w9", 0.7, layers=(14,)))
        rt = RequestTrace("r2", "on", {}, steer_spec=steer.SteerSpec(parts, positions="decode", edit=True))
        MANAGER._ensure_plan(rt, tracer)
        assert rt.steer_failed is None and rt.steer.remove[14].shape == (1, D)
        w9 = vectors.load_vector("inkling-test", "w9").vectors[14]
        np.testing.assert_allclose(rt.steer.vectors[14], 1.4 * w9, rtol=1e-6)
        readout = np.linalg.norm(J[14][:, cols(14, J[14].shape[1])] @ w9)            # through the sum's columns alone
        np.testing.assert_allclose(rt.steer.meta["parts"][1]["readout_norm"][14], readout, rtol=1e-4)
        rt = RequestTrace("r3", "on", {}, steer_spec=steer.SteerSpec(parts, positions="decode"))   # not an edit's
        MANAGER._ensure_plan(rt, tracer)
        assert rt.steer is None and rt.steer_failed == f"plan: {steer.WORD_EDIT_ONLY}"
    finally:
        MANAGER.tracers.pop(id(served), None)
        MANAGER.norm_tables.pop("inkling-test", None)


def test_mtp_windows_are_placed_by_the_first_full_attention_visits_kvcache(served):
    """Inkling 2's cache: a sliding visit's state (first here) has no offset; the first full-attention visit's KVCache
    places a window."""
    cache = served.make_cache()
    assert isinstance(cache[0], ArraysCache) and getattr(cache[0], "offset", None) is None
    served(SHORT, cache=cache)
    assert window_starts(cache, 5, 1) == [0]
    served(mx.array([[4, 2, 7]]), cache=cache)
    assert window_starts(cache, 3, 1) == [5]
    assert window_starts([ArraysCache(7)], 3, 1) is None


def test_mtp_windows_are_placed_by_a_cache_list_first_cache():
    """GLM-5.3's sparse-attention layers (CacheList(KVCache, PoolingCache), after its linear layers' ArraysCache) and
    Inkling 1's layers (CacheList(KVCache, ArraysCache)): the CacheList has no offset of its own, its first cache does."""
    visit = NS(caches=(NS(offset=12), NS()))
    assert window_starts([NS(), visit, visit], 3, 1) == [9]
    assert window_starts([NS(caches=(NS(offset=np.array([12, 7])), NS()))], 3, 2) == [9, 4]
    assert window_starts([NS(caches=(NS(), NS()))], 3, 1) is None


def test_a_fit_forward_after_a_prefix_is_the_models_own(served, monkeypatch):
    """A deep window's fit forward reads each visit's cache as the model does (``visit_caches``: a sliding visit one
    entry, a full-attention visit two) and the full-attention visits' one ``tau`` at the cache's offset: the model's
    own values to the bit, its halves plain as the fit's; without the prefix's cache it reads another sequence."""
    me, tm, prefix, ids = sys.modules[__name__], served._vlm_model.model, LONG[:, :12], LONG[:, 12:]
    own, cache = served.make_cache(), served.make_cache()
    tm(prefix, cache=own)
    tm(prefix, cache=cache)
    with monkeypatch.context() as m:
        m.setattr(me, "_COMPILED_MAX_TOKENS", 0)
        want = np.array(tm(ids, cache=own))
    with inkling2.differentiable(served) as (_, forward):
        np.testing.assert_array_equal(np.array(tm.norm(forward(ids, cache))), want)
        assert not np.allclose(np.array(tm.norm(forward(ids))), want, atol=1e-2)


def test_an_older_omlxs_inkling_2_is_refused():
    """oMLX 0.8.1 and 0.8.2's text model (no ``visit_caches``: a cache entry a visit, each layer its own tau): every
    adapter call says Inkling 2 needs 0.8.3, where generic's adapter would read it wrong."""
    class Inkling2TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(D, D)]
            self._schedule = ((0, 0),)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inkling2TextModel()

    old = Model()
    for call in (lambda: adapters.install(CaptureSession(CaptureConfig()), old), lambda: adapters.layer_count(old),
                 lambda: adapters.visit_counter(old)):
        with pytest.raises(ValueError, match="Inkling 2 needs oMLX 0.8.3 or newer"):
            call()


def writes_and_reads(served, ids):
    """Each visit's write and where it read from, as the offline measure the dock was designed on takes them: the write
    the sum out less the sum in (a block's first layer: the sum out); every attention and MLP mix's softmax over its
    sources, the visit's calls that see as many sources as its first, their queries' mean, placed [blocks, 0 for the
    blocks not finished yet, the sum]. Run unhooked and uncompiled: ([visits, B, T, 1], [visits, B, T, 5])."""
    from metalscope.hooks import add_hook
    tm = served._vlm_model.model
    mixes = {id(l.attn_input) for l in tm.layers} | {id(l.mlp_input) for l in tm.layers}
    visit, seen, writes = [0], {}, {}
    orig = AttnResQuery.__call__

    def call(self, blocks, partial_n):
        if id(self) in mixes:
            src = mx.stack([*blocks, partial_n], axis=-2).astype(mx.float32)                  # [B, T, S, H]
            seen.setdefault(visit[0], []).append(mx.softmax((src @ self.query.T) * self.scale, axis=-2, precise=True))
        return orig(self, blocks, partial_n)

    def on_layer(m, a, _k, out):
        d = out.astype(mx.float32) - (0 if m.is_boundary else a[0].astype(mx.float32))
        writes[visit[0]] = mx.linalg.norm(d, axis=-1, keepdims=True)
        visit[0] += 1
    removers = [add_hook(layer, on_layer) for layer in tm.layers]
    AttnResQuery.__call__ = call
    mx.disable_compile()
    try:
        served(ids)
    finally:
        mx.enable_compile()
        AttnResQuery.__call__ = orig
        for r in removers:
            r()
    reads = []
    for v in range(VISITS):
        n = seen[v][0].shape[-2]
        m = mx.concatenate([w for w in seen[v] if w.shape[-2] == n], axis=-1).mean(-1)          # [B, T, n]
        reads.append(mx.concatenate([m[..., :-1], mx.zeros((*m.shape[:-1], 5 - n)), m[..., -1:]], axis=-1))
    return np.array(mx.stack([writes[v] for v in range(VISITS)])), np.array(mx.stack(reads))


def three_queries(served):
    """The full-attention layers' attention mixes with a query each for q, k and v, as the real model's (its
    attention reads the first)."""
    for i in FULL:
        mix = served._vlm_model.model.layers[i].attn_input
        mix.query = 0.5 * mx.random.normal((3, D), key=mx.random.key(i))
    mx.eval(served._vlm_model.parameters())


def test_each_visit_records_what_it_wrote_into_the_sum_and_where_it_read_from(served):
    """resid_write_norm (one stream) and resid_read_share at every visit, against the offline measure: on decode-sized
    steps (reduced together at the step's end, the second half run plain while recorded) and prompt chunks past 16
    tokens (a visit at a time), for a selected row and a subset of visits. Without the router recorded the second
    half stays compiled and only the writes are recorded."""
    three_queries(served)
    batch = mx.concatenate([LONG, mx.array([[(5 * i + 1) % V for i in range(20)]])])
    for ids in (SHORT, LONG, batch[:, :3], batch):
        rows = [1] if ids.shape[0] == 2 else None
        want_w, want_r = writes_and_reads(served, ids)
        if rows:
            want_w, want_r = want_w[:, rows], want_r[:, rows]
        for layers in (None, [0, 3, 6, 7, 12]):
            session = CaptureSession(CaptureConfig(router=True, residual_norm=False, layers=layers))
            adapters.install(session, served)
            session.set_rows(rows)
            recs, count = run(session, served, ids)
            session.close()
            for v in layers or range(VISITS):
                err = f"T={ids.shape[1]} rows={rows} visit {v}"
                assert count[("resid_write_norm", v)] == count[("resid_read_share", v)] == 1, err
                np.testing.assert_allclose(recs[("resid_write_norm", v)], want_w[v], rtol=5e-3, atol=1e-5, err_msg=err)
                np.testing.assert_allclose(recs[("resid_read_share", v)], want_r[v], rtol=0, atol=2e-3, err_msg=err)   # on the GPU 2e-4 apart, measured
                np.testing.assert_allclose(recs[("resid_read_share", v)].sum(-1), 1, rtol=1e-5)
            assert {v for k, v in count if k == "resid_read_share"} == set(layers or range(VISITS))
    # The slots: visit 0 reads the embedding's block and its sum only; the last visit all four blocks and its sum.
    assert (want_r[0][..., 1:4] == 0).all() and (want_r[-1] > 0).all() and (want_r[[0, 2, 4]][..., 3] == 0).all()
    session = CaptureSession(CaptureConfig(residual_norm=False))
    adapters.install(session, served)
    recs, _ = run(session, served, SHORT)
    session.close()
    assert ("resid_write_norm", VISITS - 1) in recs and not any(k == "resid_read_share" for k, _ in recs)


def test_inkling_writes_and_reads_reach_the_trace_and_the_live_stream_under_mtp(served, home):
    """Through the tracer, MTP decode on: every visit's write and read shares saved a row a position, and the token
    events of a ``writes`` subscriber carry those of the forward that chose the token (k = 1 write a visit, 5 shares);
    the others none."""
    import asyncio
    from metalscope import config
    from metalscope.live import TraceManager
    from metalscope.store import Trace
    from tests.helpers import drain, wait_saved
    three_queries(served)
    config.save_settings(mode="on")
    served._omlx_mtp_decode_enabled = True
    mgr = TraceManager()
    tracer = mgr.attach(served, "inkling-test", Tok())
    assert tracer.adapter_name == "inkling2" and tracer.mtp
    loop = asyncio.new_event_loop()
    want, plain = mgr.hub.subscribe(loop, "me", writes=True), mgr.hub.subscribe(loop, "me", experts=True, lens=True)
    mgr.register("r1", "on", dict(client="me"))
    prompt = [(3 * i + 2) % 60 for i in range(20)]          # > 16: read as a prefill chunk
    cache = served.make_cache()

    def fwd(ids):
        mgr.rows_for(tracer, ["r1"])
        out = served(mx.array([ids]), cache=cache)
        mgr.clear_rows(tracer)
        return [int(v) for v in mx.argmax(out[0], axis=-1).tolist()]

    fwd(prompt[:-1])
    g0 = fwd(prompt[-1:])[0]
    w1 = fwd([g0, 5, 6])                                     # two drafts, both kept
    w2 = fwd([w1[2], 7, 8])
    outputs = [g0, 5, 6, w1[2], 7, 8, w2[2]]
    mgr.finish("r1", prompt, outputs, tracer)
    wait_saved(mgr, 1)

    t = Trace.load(next((home / "traces").rglob("meta.json")).parent)
    P, n = len(prompt), len(prompt) + len(outputs) - 1
    assert t.tokens == prompt + outputs[:-1]
    W = np.stack([t.get("resid_write_norm", v) for v in range(VISITS)], axis=1)          # [positions, visits, 1]
    R = np.stack([t.get("resid_read_share", v) for v in range(VISITS)], axis=1)       # [positions, visits, 5]
    want_w, want_r = writes_and_reads(served, mx.array([t.tokens]))
    np.testing.assert_allclose(W, want_w[:, 0].transpose(1, 0, 2), rtol=1e-2, atol=1e-5)
    np.testing.assert_allclose(R, want_r[:, 0].transpose(1, 0, 2), rtol=0, atol=2e-3)
    tokens = [e for e in drain(loop, want) if e["type"] == "token"]
    assert [e["id"] for e in tokens] == outputs and W.shape == (n, VISITS, 1)
    for i, e in enumerate(tokens):                           # token i was chosen at position P - 1 + i
        assert e["write_k"] == 1 and e["read_k"] == 5 and len(e["writes"]) == VISITS and len(e["reads"]) == VISITS * 5
        assert "shares" not in e                             # reads, no shares: no layers view
        np.testing.assert_allclose(e["writes"], W[P - 1 + i].ravel(), rtol=6e-3, err_msg=f"token {i}")
        np.testing.assert_allclose(e["reads"], R[P - 1 + i].ravel(), rtol=6e-3, atol=1e-9, err_msg=f"token {i}")
        assert all(x == float(f"{x:.3g}") for x in e["reads"])
    others = [e for e in drain(loop, plain) if e["type"] == "token"]
    assert [e["id"] for e in others] == outputs
    assert not any(k in e for e in others for k in ("writes", "write_k", "shares", "reads", "read_k"))

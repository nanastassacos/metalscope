import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from metalscope.capture import CaptureConfig, CaptureSession
from metalscope.hooks import add_hook
from metalscope.hooks import is_hooked
from metalscope.adapters import install
from tests.helpers import tiny_llama, tiny_moe


TOKENS = mx.array([[1, 5, 9, 2, 7]])


def test_hook_is_transparent_and_removable():
    model = tiny_llama()
    keys_before = [k for k, _ in tree_flatten(model.parameters())]
    ref = model(TOKENS)
    layer = model.model.layers[1]
    cls = type(layer)
    seen = []
    remove = add_hook(layer, lambda m, a, k, out: seen.append(out.shape))
    out = model(TOKENS)
    assert seen == [(1, 5, 64)]
    assert mx.array_equal(out, ref)
    assert isinstance(layer, cls)
    assert [k for k, _ in tree_flatten(model.parameters())] == keys_before
    remove()
    assert type(layer) is cls
    model(TOKENS)
    assert len(seen) == 1


def test_hook_can_replace_output():
    model = tiny_llama()
    ref = model(TOKENS)
    remove = add_hook(model.model.layers[0], lambda m, a, k, out: out * 0)
    assert not mx.allclose(model(TOKENS), ref)
    remove()
    assert mx.array_equal(model(TOKENS), ref)


def test_residual_norms_match_manual_forward():
    model = tiny_llama()
    session = CaptureSession(CaptureConfig(layers=[0, 2]))
    assert install(session, model) == "generic"
    model(TOKENS)
    mx.eval(session.pending_arrays())
    session.end_step()
    recs = {(r.kind, r.layer): r.value for r in session.drain()}
    session.close()

    h = model.model.embed_tokens(TOKENS)
    expected = {}
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask="causal", cache=None)
        expected[i] = np.linalg.norm(np.array(h.astype(mx.float32)), axis=-1)
    assert set(recs) == {("resid_norm", 0), ("resid_norm", 2)}
    np.testing.assert_allclose(recs[("resid_norm", 2)], expected[2], rtol=1e-5)


def test_router_and_logits():
    model = tiny_moe()
    session = CaptureSession(CaptureConfig(layers=[], router=True, logits_topk=5))
    install(session, model)
    logits = model(TOKENS)
    mx.eval(session.pending_arrays())
    session.end_step()
    recs = {(r.kind, r.layer): r.value for r in session.drain()}
    assert recs[("router_ids", 0)].shape == (1, 5, 2)
    probs = recs[("router_probs", 1)]
    assert np.all(probs > 0) and np.all(probs.sum(-1) <= 1 + 1e-5)
    top = np.argsort(-np.array(logits.astype(mx.float32))[0, -1])[:5]
    assert set(recs[("logit_ids", -1)][0, -1]) == set(top)
    assert recs[("logit_entropy", -1)].shape == (1, 5)


def test_disabled_session_records_nothing():
    model = tiny_llama()
    session = CaptureSession(CaptureConfig())
    install(session, model)
    session.enabled = False
    model(TOKENS)
    assert session.pending_arrays() == []


def test_router_matches_model_selection():
    model = tiny_moe()
    block = model.model.layers[0].mlp
    seen = {}
    # Reference: the block's own selection, recomputed from its real input.
    remove = add_hook(block, lambda m, a, k, out: seen.setdefault("x", a[0]))
    session = CaptureSession(CaptureConfig(layers=[], router=True))
    install(session, model)
    model(TOKENS)
    mx.eval(session.pending_arrays())
    session.end_step()
    recs = {(r.kind, r.layer): r.value for r in session.drain()}
    remove()
    gates = mx.softmax(block.gate(seen["x"]).astype(mx.float32), axis=-1)
    expected = np.sort(np.array(mx.argpartition(gates, kth=-2, axis=-1)[..., -2:]), axis=-1)
    np.testing.assert_array_equal(np.sort(recs[("router_ids", 0)], axis=-1), expected)


def test_logit_lens_last_layer_matches_logits():
    model = tiny_llama()
    session = CaptureSession(CaptureConfig(layers=[2], residual_norm=False, logits_topk=3, logit_lens_topk=3))
    install(session, model)
    model(TOKENS)
    mx.eval(session.pending_arrays())
    session.end_step()
    recs = {(r.kind, r.layer): r.value for r in session.drain()}
    # Lens at the last layer is the model's own readout; head hook fired once.
    np.testing.assert_array_equal(recs[("lens_ids", 2)], recs[("logit_ids", -1)])
    np.testing.assert_allclose(recs[("lens_logp", 2)], recs[("logit_logp", -1)], rtol=1e-5)
    assert np.all(np.diff(recs[("lens_logp", 2)], axis=-1) <= 0)


def test_row_selection_and_owners():
    model = tiny_llama()
    batch = mx.array([[1, 5, 9], [2, 7, 3], [4, 4, 8]])
    session = CaptureSession(CaptureConfig(layers=[1], logits_topk=3))
    install(session, model)

    session.set_rows([])                      # nothing selected: hooks no-op
    model(batch)
    assert session.pending_arrays() == []

    session.set_rows([2, 0], owners=["req-c", "req-a"])
    model(batch)
    mx.eval(session.pending_arrays())
    session.end_step()
    recs = {(r.kind, r.layer): r for r in session.drain()}
    norm = recs[("resid_norm", 1)]
    assert norm.owners == ["req-c", "req-a"] and norm.value.shape == (2, 3)

    # Same values as recording every row and picking rows 2 and 0.
    session.set_rows(None)
    model(batch)
    mx.eval(session.pending_arrays())
    session.end_step()
    full = {(r.kind, r.layer): r.value for r in session.drain()}
    np.testing.assert_allclose(norm.value, full[("resid_norm", 1)][[2, 0]], rtol=1e-5)
    np.testing.assert_array_equal(recs[("logit_ids", -1)].value, full[("logit_ids", -1)][[2, 0]])


def test_lens_layer_subset():
    model = tiny_llama()  # 3 layers
    session = CaptureSession(CaptureConfig(residual_norm=False, logit_lens_topk=2, lens_every=2, lens_last=1))
    install(session, model)
    model(TOKENS)
    mx.eval(session.pending_arrays())
    session.end_step()
    lens_layers = sorted({r.layer for r in session.drain() if r.kind == "lens_ids"})
    assert lens_layers == [0, 2]  # every 2nd (0, 2) plus the last (2)


def test_batched_and_per_layer_paths_agree():
    """Short steps are reduced stacked at step end; long ones per layer."""
    model = tiny_moe()
    long_tokens = mx.array([list(range(1, 25))])  # 24 tokens > BATCH_MAX_TOKENS
    out = {}
    for name, toks in (("long", long_tokens), ("short", long_tokens[:, :8])):
        session = CaptureSession(CaptureConfig(router=True))
        install(session, model)
        model(toks)
        mx.eval(session.pending_arrays())
        session.end_step()
        out[name] = {(r.kind, r.layer): r.value for r in session.drain()}
        session.close()
    for key in [("resid_norm", 0), ("resid_norm", 1), ("router_probs", 1)]:
        np.testing.assert_allclose(out["short"][key], out["long"][key][:, :8], rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(np.sort(out["short"][("router_ids", 0)], -1), np.sort(out["long"][("router_ids", 0)][:, :8], -1))


def test_chunked_logit_stats_match_unchunked(monkeypatch):
    import metalscope.capture as cap
    model = tiny_llama()
    toks = mx.array([list(range(1, 41))])
    got = {}
    for chunk in (1000, 16):
        monkeypatch.setattr(cap, "LOGIT_CHUNK_TOKENS", chunk)
        session = CaptureSession(CaptureConfig(layers=[], residual_norm=False, logits_topk=4))
        install(session, model)
        model(toks)
        mx.eval(session.pending_arrays())
        session.end_step()
        got[chunk] = {r.kind: r.value for r in session.drain()}
        session.close()
    for kind in ("logit_ids", "logit_logp", "logit_entropy"):
        np.testing.assert_allclose(got[16][kind], got[1000][kind], rtol=1e-5, atol=1e-6)


def test_jlens_identity_equals_logit_lens_and_word_directions():
    """J = I reduces the Jacobian lens to the logit lens (paper, Methods); a word's direction (vectors.from_lens) dotted
    with the residual is its lens logit before the norm's per-position scale."""
    model = tiny_llama()
    d = 64
    eye = {l: mx.eye(d) for l in range(3)}
    vocab_rows = model.lm_head.weight[mx.array([3, 7])]           # two words
    gamma = model.model.norm.weight
    from metalscope.jlens import Lens, concept_vectors
    lens = Lens(eye)
    directions = concept_vectors(lens, 1, vocab_rows, gamma)
    session = CaptureSession(CaptureConfig(residual_norm=False, logit_lens_topk=4, jlens=eye, jlens_topk=4))
    install(session, model)
    model(TOKENS)
    mx.eval(session.pending_arrays())
    session.end_step()
    recs = {(r.kind, r.layer): r.value for r in session.drain()}
    for l in range(3):
        np.testing.assert_array_equal(recs[("jlens_ids", l)], recs[("lens_ids", l)])
    h = model.model.embed_tokens(TOKENS)
    for layer in model.model.layers[:2]:
        h = layer(h, mask="causal", cache=None)
    rms = mx.sqrt(mx.mean(h.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-5)
    lens_logits = model.lm_head(model.model.norm(h))[..., mx.array([3, 7])]
    score = h.astype(mx.float32) @ directions.T
    np.testing.assert_allclose(np.array(score) / np.array(rms), np.array(lens_logits), rtol=2e-2, atol=2e-3)


def test_batched_lens_reads_equal_one_layer_at_a_time(monkeypatch):
    """Every read layer of both lenses through one LM-head call (capture._lens_reads) gives what reading each layer
    on its own does, for a decode step and for a prompt chunk."""
    import metalscope.capture as C
    model = tiny_llama()
    J = {l: mx.random.normal((64, 64), key=mx.random.key(l)) * 0.2 + mx.eye(64) for l in range(3)}

    def run(rows, tokens):
        monkeypatch.setattr(C, "LENS_BATCH_ROWS", rows)
        s = CaptureSession(CaptureConfig(residual_norm=False, logit_lens_topk=10, lens_every=1, jlens=J, jlens_topk=7))
        install(s, model)
        model(tokens)
        mx.eval(s.pending_arrays())
        s.end_step()
        out = {(r.kind, r.layer): np.array(r.value) for r in s.drain()}
        s.close()
        return out
    for tokens in (mx.array([[5]]), TOKENS):
        one, together = run(1, tokens), run(64, tokens)
        lens_keys = [k for k in one if k[0].startswith(("jlens_", "lens_"))]
        assert {k[0] for k in lens_keys} == {"jlens_ids", "jlens_logp", "lens_ids", "lens_logp"}     # no lens entropy
        assert one[("jlens_ids", 0)].shape[-1] == 7 and one[("lens_ids", 0)].shape[-1] == 10
        assert set(lens_keys) == {k for k in together if k[0].startswith(("jlens_", "lens_"))}
        for k in lens_keys:   # the same words; p to the matmul kernel's rounding (one row vs many)
            if k[0].endswith("_ids"):
                np.testing.assert_array_equal(one[k], together[k], err_msg=str(k))
            else:
                np.testing.assert_allclose(one[k], together[k], rtol=0, atol=5e-3, err_msg=str(k))


def test_with_a_lens_tail_a_prompt_chunk_reads_both_lenses_only_at_its_end(monkeypatch):
    """With lens_tail, a step longer than it reads both lenses on its last rows only (the rows before empty: ids -1,
    log-p -inf), the logit lens alone too; without (the default: /jlens learn's scoring), every row."""
    import metalscope.capture as C
    monkeypatch.setattr(C, "BATCH_MAX_TOKENS", 2)                  # a 6-token prompt takes the per-layer path
    model = tiny_llama()
    eye = {l: mx.eye(64) for l in range(3)}

    def run(jlens, tail=3):
        s = CaptureSession(CaptureConfig(residual_norm=False, logit_lens_topk=4, lens_every=1, jlens=jlens,
                                         jlens_topk=4 if jlens else 0, lens_tail=tail))
        install(s, model)
        model(TOKENS)
        mx.eval(s.pending_arrays())
        s.end_step()
        out = {(r.kind, r.layer): np.array(r.value) for r in s.drain()}
        s.close()
        return out
    both, alone, every = run(eye), run(None), run(eye, tail=0)
    T = TOKENS.shape[1]
    assert T > 3
    for l in range(3):
        for out, kind in ((both, "lens"), (both, "jlens"), (alone, "lens")):
            ids, lp = out[(f"{kind}_ids", l)], out[(f"{kind}_logp", l)]
            assert ids.shape == (1, T, 4) and (ids[0, :T - 3] == -1).all() and (ids[0, T - 3:] >= 0).all(), kind
            assert np.isneginf(lp[0, :T - 3]).all() and (f"{kind}_entropy", l) not in out
            np.testing.assert_array_equal(ids[0, T - 3:], every[(f"{kind}_ids", l)][0, T - 3:])   # the end, as read whole
        assert (every[("jlens_ids", l)] >= 0).all() and (every[("lens_ids", l)] >= 0).all()


def test_tailed_lens_reads_share_one_readout_call():
    """A step's tailed reads (both lenses, every read layer) go through the LM head together, as untailed ones do,
    and read the same as one at a time."""
    import metalscope.capture as C
    model = tiny_llama()
    eye = {l: mx.eye(64) for l in range(3)}

    def run(rows):
        C.LENS_BATCH_ROWS, saved = rows, C.LENS_BATCH_ROWS
        try:
            s = CaptureSession(CaptureConfig(residual_norm=False, logit_lens_topk=4, lens_every=1, jlens=eye,
                                             jlens_topk=4, lens_tail=3))
            install(s, model)
            calls, readout = [], s.readout
            s.readout = lambda x: calls.append(x.shape) or readout(x)
            model(TOKENS)
            mx.eval(s.pending_arrays())
            s.end_step()
            out = {(r.kind, r.layer): np.array(r.value) for r in s.drain()}
            s.close()
            return calls, out
        finally:
            C.LENS_BATCH_ROWS = saved
    (calls, together), (_, one) = run(64), run(1)
    assert calls == [(6, 3, 64)]                     # 3 layers x 2 lenses, the last 3 rows each
    for k, v in one.items():
        if k[0].endswith("_ids"):
            np.testing.assert_array_equal(v, together[k], err_msg=str(k))
    assert (together[("lens_ids", 0)][0, :TOKENS.shape[1] - 3] == -1).all()


def test_failing_hook_never_breaks_the_model():
    model = tiny_llama()
    ref = model(TOKENS)
    def boom(*_):
        raise RuntimeError("capture bug")
    told = []
    remove = add_hook(model.model.layers[1], boom, told.append)
    assert mx.array_equal(model(TOKENS), ref)
    assert [str(e) for e in told] == ["capture bug"]                    # its owner is told why
    remove()
    assert not is_hooked(model.model.layers[1])


def test_lens_arrays_usable_from_another_thread(tmp_path):
    import threading
    from metalscope import jlens
    lens = jlens.Lens({0: mx.eye(4), 1: mx.eye(4) * 2}, dict(checkpoint="t"))
    jlens.save(lens, tmp_path / "L")
    loaded = {}
    t = threading.Thread(target=lambda: loaded.setdefault("l", jlens.load(tmp_path / "L")))
    t.start(); t.join()
    out = loaded["l"].jacobians[1] @ mx.ones((4,))
    assert mx.allclose(out, mx.full((4,), 2.0))


def test_block_topk_finds_the_same_k_as_a_plain_argpartition():
    """_topk_ids searches only the k blocks with the largest maxima: the same values as argpartition over the whole
    vocabulary (ids may differ only between equal entries), with or without padding, any leading shape."""
    from metalscope.capture import TOPK_BLOCK, _topk_ids
    for V in (8 * TOPK_BLOCK, 10_007):
        z = mx.random.normal((2, 3, V), key=mx.random.key(V))
        for k in (1, 10):
            ids = _topk_ids(z, k)
            assert ids.shape == (2, 3, k)
            got = np.sort(np.array(mx.take_along_axis(z, ids, axis=-1)), axis=-1)
            want = np.sort(np.array(z), axis=-1)[..., -k:]
            np.testing.assert_array_equal(got, want)
    small = mx.random.normal((4, 100))
    np.testing.assert_array_equal(np.sort(np.array(mx.take_along_axis(small, _topk_ids(small, 5), axis=-1))),
                                  np.sort(np.array(small))[..., -5:])


def test_the_entropy_of_logits_with_masked_out_choices_is_a_number():
    """Inkling 2 masks its padded vocabulary with -inf: those choices add nothing (0 x -inf is NaN, and one NaN in a
    token event makes the dock's JSON parser drop the event)."""
    z = mx.array([[[2.0, 0.5, -1.0, 0.0]]])
    masked = mx.concatenate([z, mx.full((1, 1, 3), -mx.inf)], axis=-1)
    ids, logp, ent = CaptureSession._topk_stats(masked, 2)
    _, want_logp, want = CaptureSession._topk_stats(z, 2)
    assert np.isfinite(np.array(ent)).all()
    np.testing.assert_allclose(np.array(ent), np.array(want), rtol=1e-5)
    np.testing.assert_allclose(np.array(logp), np.array(want_logp), rtol=1e-5)
    assert np.array(ids).tolist() == [[[0, 1]]]


def test_a_lens_reads_a_layers_state_when_it_is_that_wide():
    """record_residual's ``state`` (Inkling 2's state at a visit, its width by layer): a J as wide as a layer's state
    reads the state, also when that is the residual's width; with no state, one as wide as the residual reads the
    residual; in one deferred flush with mixed widths (a decode step) and on the per-layer path (a prompt chunk). The
    residual's norm is the residual's."""
    D, V, k = 8, 40, 5
    key = lambda i: mx.random.key(i)
    W = mx.random.normal((V, D), key=key(1))
    J = {0: mx.random.normal((D, 3 * D), key=key(2)), 1: mx.random.normal((D, 2 * D), key=key(3)),
         2: mx.random.normal((D, D), key=key(4)), 3: mx.random.normal((D, D), key=key(5))}
    for T in (3, 20):
        s = CaptureSession(CaptureConfig(jlens=J, jlens_topk=k))
        s.readout = lambda x: x.astype(mx.float32) @ W.T
        h = {l: mx.random.normal((1, T, D), key=key(10 + l)) for l in range(4)}
        state = {0: mx.random.normal((1, T, 3 * D), key=key(20)), 1: mx.random.normal((1, T, 2 * D), key=key(21)),
                 2: mx.random.normal((1, T, D), key=key(22))}
        for l in range(4):
            s.record_residual(l, h[l], state.get(l))
        mx.eval(s.pending_arrays())
        s.end_step()
        recs = {(r.kind, r.layer): np.array(r.value) for r in s.drain()}
        for l in range(4):
            logits = np.array(state.get(l, h[l]) @ J[l].T @ W.T)
            np.testing.assert_array_equal(recs[("jlens_ids", l)], np.argsort(-logits, axis=-1)[..., :k], err_msg=f"{T} {l}")
            np.testing.assert_allclose(recs[("resid_norm", l)], np.linalg.norm(np.array(h[l]), axis=-1), rtol=1e-5)


def test_flash_next_records_what_each_layer_wrote_into_each_stream():
    """resid_write_norm: a layer's output streams less the streams it read (the previous layer's write applied; layer 0
    reads the embedding's), a norm per stream, beside hc_stream_norm. The same with the writes deferred or applied at
    once, on the stacked path (a decode step) and the per-layer one (a prompt chunk), for selected rows, and for a layer
    whose input was not recorded (a layer subset)."""
    from tests.helpers import FlashNextLike, flash_next_writes
    model = FlashNextLike(layers=3, hc=4)
    batch = mx.array([[3, 17, 5, 42, 9] * 5, [7, 1, 8, 2, 6] * 5])
    for defer in (True, False):
        model.model.defer = defer
        for T in (1, 3, 24):                                    # 24 > BATCH_MAX_TOKENS: reduced layer by layer
            ids = batch[:, :T]
            want = flash_next_writes(model, ids)
            for layers, rows in ((None, None), ([0, 2], [1])):
                s = CaptureSession(CaptureConfig(layers=layers, residual_norm=False))
                assert install(s, model) == "qwen4_exp"
                s.set_rows(rows)
                model(ids)
                mx.eval(s.pending_arrays())
                s.end_step()
                recs = {(r.kind, r.layer): np.array(r.value) for r in s.drain()}
                s.close()
                for l in layers or range(3):
                    got = recs[("resid_write_norm", l)]
                    assert got.shape == (len(rows or (0, 1)), T, 4) and got.dtype == np.float32
                    np.testing.assert_allclose(got, want[l][rows or slice(None)], rtol=1e-4, atol=1e-6,
                                               err_msg=f"defer={defer} T={T} layer {l} rows {rows}")
                    assert ("hc_stream_norm", l) in recs
                assert {k for k, _ in recs} == {"hc_stream_norm", "resid_write_norm"}
    assert (want > 0).all() and not np.allclose(want[0], want[1])


def test_a_forward_stopped_part_way_leaves_nothing_of_its_streams_past_its_step():
    """The qwen4_exp adapter keeps the last layer's streams from one layer's hook to the next (session.carry): a forward
    that stops part way (a raise, a prefill split over the layers) leaves nothing of them once its step ends, and the
    next whole forward pairs each layer's input and output as before."""
    import gc
    import weakref
    from tests.helpers import FlashNextLike, flash_next_writes
    model = FlashNextLike(layers=3, hc=4)
    ids = mx.array([[3, 17, 5]])
    s = CaptureSession(CaptureConfig(residual_norm=False))
    install(s, model)
    s.set_rows(None)
    inner = model.model
    h, write, kept = mx.tile(inner.embed_tokens(ids), (1, 1, 4)), None, []
    for layer in inner.layers[:2]:                              # layers[:2] only: the last layer never runs
        r = layer(h, mask="causal", write=write, defer_write=True)
        kept += [weakref.ref(x) for x in r]
        h, write = r[0], r[1:]
    kept.append(weakref.ref(s.carry["streams"]))
    mx.eval(s.pending_arrays())
    s.end_step()
    s.drain()
    del h, write, r, layer
    gc.collect()
    assert s.carry == {} and not any(ref() is not None for ref in kept)
    model(ids)
    mx.eval(s.pending_arrays())
    s.end_step()
    recs = {(r.kind, r.layer): np.array(r.value) for r in s.drain()}
    s.close()
    want = flash_next_writes(model, ids)
    for l in range(3):
        np.testing.assert_allclose(recs[("resid_write_norm", l)], want[l], rtol=1e-4, atol=1e-6, err_msg=f"layer {l}")

"""The ds4 backend: reading ds4's capture file, filing it as a trace, the live
stream, the steering file, the GGUF vocabulary, and the proxy end to end
against a stand-in ds4-server."""

import asyncio
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from metalscope import config, models, vectors
from metalscope.capture import CaptureConfig, capture_config
from metalscope.ds4.backend import Ds4Tracer, Run
from metalscope.ds4.frames import ScopeReader, write_steering
from metalscope.ds4.vocab import GgufTokenizer, _byte_decoder
from metalscope.live import READOUT_KINDS, TraceManager
from metalscope.store import Trace, read_index
from tests.helpers import wait_saved, write_norms

L, HC, E, K_EXP, N_EXP, V, TOPK = 6, 4, 8, 2, 16, 50, 10
PROXY = "http://127.0.0.1:8009"     # the Host clients send the proxy (a test client's own, testserver, is refused)


class Tok:
    all_special_ids = [0]

    def decode(self, ids):
        return "".join(f"<{int(i)}>" for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [1 + sum(map(ord, text)) % (V - 1)]


def tracer(name="tiny-ds4"):
    return Ds4Tracer(name, Tok(), "qwen4", L, HC, E, K_EXP, N_EXP)


def write_norm(pos, layer, stream):
    """What ScopeWriter records layer ``layer`` changed in ``stream`` at position ``pos``: three significant digits, so
    the live stream's ``writes`` carry it exactly."""
    return (pos % 10 + layer / 10 + stream / 100).astype("<f4")


class ScopeWriter:
    """Writes ds4's capture format (ds4_scope.inc, scope_emit_forward and friends) to a file it creates, as ds4 does
    (ds4_scope_trace_create, O_EXCL): a request naming a file that exists fails, so every path the proxy and the
    measurement engine send must be fresh."""

    def __init__(self, path, lens, jlens=(), ids=()):
        """``jlens``: the J-lens layers read; ``ids``: the logit ids (LIDS)."""
        self.f, self.lens, self.jlens = open(path, "xb"), lens, list(jlens)
        self.ids = list(ids)
        self.f.write(b"DS4SCOPE" + struct.pack("<10I", 6, 1, L, HC, E, K_EXP, N_EXP, V, TOPK, len(lens)))
        self.f.write(struct.pack(f"<{len(lens)}I", *lens))
        if self.jlens:
            self.rec(b"JLNS", struct.pack(f"<I{len(self.jlens)}I", len(self.jlens), *self.jlens))
        if self.ids:
            self.rec(b"LIDS", struct.pack(f"<I{len(self.ids)}i", len(self.ids), *self.ids))
        self.f.flush()

    def rec(self, tag, payload):
        self.f.write(tag + struct.pack("<I", len(payload)) + payload)
        self.f.flush()

    def prompt(self, ids):
        self.rec(b"PRMT", struct.pack("<I", len(ids)) + np.asarray(ids, "<i4").tobytes())

    def step(self, pos0, tokens, out_top=None, head_rows=1):
        """One forward: experts (pos * 3 + layer) % N_EXP, stream norms pos + layer / 10, writes (what each layer
        changed in each stream) write_norm(pos, layer, stream), resid norms layer + 1 + r, lens top-1 = layer, J-lens
        top-1 = 100 + layer, output top ids out_top (a list for several head rows: an MTP verify's kept rows, the
        forward's last head_rows); with logit ids, head row r's log-probability at the j-th is -(r + 1) - j / 10."""
        T, R = len(tokens), head_rows
        tops = out_top if isinstance(out_top, list) else [out_top] * R
        pos = np.arange(pos0, pos0 + T)
        experts = np.stack([np.stack([(pos * 3 + l + j) % N_EXP for j in range(K_EXP)], -1) for l in range(L)])
        weights = np.full((L, T, K_EXP), 1.0 / K_EXP, "<f4")
        streams = np.broadcast_to((pos[None, :, None] + np.arange(L)[:, None, None] / 10), (L, T, HC)).astype("<f4")
        writes = write_norm(pos[None, :, None], np.arange(L)[:, None, None], np.arange(HC)[None, None, :])
        at_ids = out_top is not None and bool(self.ids)
        flags = (2 if self.lens else 0) | 4 | (1 if out_top is not None else 0) | (16 if self.jlens else 0) \
            | (64 if at_ids else 0)
        body = struct.pack("<IIII", pos0, T, flags, R) + np.asarray(tokens, "<i4").tobytes()
        body += experts.astype("<i4").tobytes() + weights.tobytes() + streams.tobytes() + writes.astype("<f4").tobytes()
        body += np.stack([np.arange(1, L + 1, dtype="<f4") + r for r in range(R)]).tobytes()
        if self.lens:
            nl = len(self.lens)
            ids = np.array([[[layer] + list(range(40, 40 + TOPK - 1)) for layer in self.lens]] * R, "<i4")
            body += ids.tobytes() + np.full((R, nl, TOPK), -2.0, "<f4").tobytes()
        if self.jlens:
            nj = len(self.jlens)
            jids = np.array([[[100 + layer] + list(range(40, 40 + TOPK - 1)) for layer in self.jlens]] * R, "<i4")
            body += jids.tobytes() + np.full((R, nj, TOPK), -1.0, "<f4").tobytes()
        if out_top is not None:
            out = np.array([[t] + list(range(40, 40 + TOPK - 1)) for t in tops], "<i4")
            logp = np.array([[np.log(0.9)] + [-5.0] * (TOPK - 1)] * R, "<f4")
            body += out.tobytes() + logp.tobytes() + np.full(R, 0.3, "<f4").tobytes()
        if at_ids:
            n = len(self.ids)
            body += (-(np.arange(R)[:, None] + 1.0) - np.arange(n)[None, :] / 10).astype("<f4").tobytes()
        self.rec(b"STEP", body)

    def rsum(self, sink, n, total, last):
        """The residual sums: per layer the row count, the sum [L, W] and the prompt's last row [L, W]."""
        self.rec(b"RSUM", struct.pack("<III", L, HC * E, sink) + np.asarray(n, "<u4").tobytes()
                 + np.asarray(total, "<f4").tobytes() + np.asarray(last, "<f4").tobytes())

    def nxlp(self, pos0, lp):
        """Next-token log-probabilities: ``lp[i]`` read at row pos0 + i."""
        self.rec(b"NXLP", struct.pack("<II", pos0, len(lp)) + np.asarray(lp, "<f4").tobytes())

    def nxtk(self, pos0, ids, lp):
        """Each row's top choices: ``ids[i]``, ``lp[i]`` ([k] each, likeliest first) read at row pos0 + i."""
        ids, lp = np.asarray(ids, "<i4"), np.asarray(lp, "<f4")
        self.rec(b"NXTK", struct.pack("<III", pos0, *ids.shape) + ids.tobytes() + lp.tobytes())

    def end(self, sampled, finish="done"):
        fb = finish.encode()
        self.rec(b"END_", struct.pack("<I", len(sampled)) + np.asarray(sampled, "<i4").tobytes()
                 + struct.pack("<I", len(fb)) + fb)
        self.f.close()


def write_request(path, prompt, replies, chunk=4):
    """A request as ds4 records it: the prompt in chunks of ``chunk`` (each chunk's last row predicts), then
    one forward per fed reply token; the last sampled token (a stop token) is never fed."""
    lens = CaptureConfig(lens_every=4, lens_last=2).lens_layers(L)
    w = ScopeWriter(path, lens)
    w.prompt(prompt)
    for i in range(0, len(prompt), chunk):
        part = prompt[i:i + chunk]
        w.step(i, part, out_top=replies[0] if i + len(part) == len(prompt) else 7)
    for j, tok in enumerate(replies[:-1]):
        w.step(len(prompt) + j, [tok], out_top=replies[j + 1])
    w.end(replies)
    return lens


def test_reader_waits_for_whole_records(tmp_path):
    path = tmp_path / "r.bin"
    write_request(path, [1, 2, 3, 4, 5], [11, 12, 0])
    data = path.read_bytes()
    part = tmp_path / "part.bin"
    part.write_bytes(data[:60])               # the header and a few bytes of the prompt record
    r = ScopeReader(part)
    assert r.poll() == [] and r.header is not None and r.prompt is None
    part.write_bytes(data)
    steps = r.poll()
    assert r.prompt == [1, 2, 3, 4, 5] and [s.pos0 for s in steps] == [0, 4, 5, 6]
    assert r.end.sampled == [11, 12, 0] and r.end.finish == "done"
    first = steps[0]
    assert first.experts.shape == (L, 4, K_EXP) and first.stream_norm.shape == (L, 4, HC)
    assert first.resid_norm[0].tolist() == list(range(1, L + 1)) and first.lens_ids.shape == (1, len(r.header.lens), TOPK)
    assert r.poll() == []


def test_reader_takes_the_logit_ids_and_the_residual_sums(tmp_path):
    """Format 5: LIDS after the header, each head row's output at those ids (STEP flag 64, after the top-k), RSUM at
    the end."""
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [1], ids=[7, 3, 42])
    w.prompt([1, 2, 3, 4, 5])
    w.step(0, [1, 2, 3, 4, 5], out_top=9)
    w.step(5, [9, 10, 11], out_top=[12, 13], head_rows=2)
    w.step(8, [12], out_top=None)                      # no output read: no flag 64
    W = HC * E
    total = np.arange(L * W, dtype=np.float32).reshape(L, W)
    last = np.full((L, W), 0.5, np.float32)
    last[3] = np.nan
    n = np.array([1, 1, 1, 0, 1, 1])
    w.rsum(4, n, total, last)
    w.end([9, 0])
    r = ScopeReader(path)
    steps = r.poll()
    assert r.logit_ids == [7, 3, 42] and r.header.lens == [1]
    first, verify, bare = steps
    np.testing.assert_allclose(first.out_at_ids, [[-1.0, -1.1, -1.2]])
    assert verify.out_at_ids.shape == (2, 3) and verify.out_at_ids[1, 0] == -2.0
    assert verify.out_ids[1, 0] == 13                                            # the ids still line up
    assert bare.out_at_ids is None and bare.out_ids is None
    s = r.rsum
    assert s.sink == 4 and s.n.tolist() == n.tolist() and s.sum.shape == (L, W)
    np.testing.assert_array_equal(s.sum, total)
    assert np.isnan(s.last[3]).all() and (s.last[0] == 0.5).all()
    mean = s.mean
    np.testing.assert_array_equal(mean[0], total[0])
    assert np.isnan(mean[3]).all()
    assert r.end.sampled == [9, 0]


def test_reader_takes_what_each_layer_changed_in_each_stream(tmp_path):
    """Format 6: a STEP's write_norm [L, T, hc] (each layer's output streams less the streams it read, a norm per
    stream) right after its stream norms, for every row of the forward as they are; the head's reads after them still
    line up."""
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [1])
    w.prompt([1, 2, 3, 4, 5])
    w.step(0, [1, 2, 3, 4, 5], out_top=9)
    w.step(5, [9, 10], out_top=[10, 11], head_rows=2)
    w.end([9, 10, 11, 0])
    first, verify = ScopeReader(path).poll()
    for st in (first, verify):
        pos = np.arange(st.pos0, st.pos0 + st.T)
        assert st.write_norm.shape == st.stream_norm.shape == (L, st.T, HC) and st.write_norm.dtype == np.float32
        np.testing.assert_array_equal(st.write_norm, write_norm(pos[None, :, None], np.arange(L)[:, None, None],
                                                                np.arange(HC)))
        np.testing.assert_allclose(st.stream_norm[:, :, 0], pos[None] + np.arange(L)[:, None] / 10, rtol=1e-6)
    assert verify.out_ids[:, 0].tolist() == [10, 11] and verify.resid_norm[1].tolist() == list(range(2, L + 2))


def test_reader_refuses_another_capture_version(tmp_path):
    """A file from a ds4 built from an older patch (format 5, without the layers' writes): rebuild ds4."""
    path = tmp_path / "r.bin"
    path.write_bytes(b"DS4SCOPE" + struct.pack("<10I", 5, 1, L, HC, E, K_EXP, N_EXP, V, TOPK, 0))
    with pytest.raises(ValueError, match=r"version 5 \(this metalscope reads 6: rebuild ds4 from src/metalscope/ds4/metalscope.patch\)"):
        ScopeReader(path).poll()


def test_a_request_is_saved_as_a_trace_with_every_row_in_place(tmp_path):
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    mgr.register("rid-1", "on", dict(conversation="c1", turn=1, last_user="hi"))
    path = tmp_path / "r.bin"
    prompt, replies = [1, 2, 3, 4, 5, 6], [11, 12, 13, 0]
    lens = write_request(path, prompt, replies)
    run = Run("rid-1", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    wait_saved(mgr, 1)
    (entry,) = read_index()
    tr = Trace.load(entry["path"])
    fed = prompt + replies[:-1]
    assert tr.tokens == fed and tr.prompt_len == len(prompt)
    assert entry["experts"] and tr.meta["adapter"] == "ds4"
    assert tr.meta["routing"] == dict(rule="qwen3_5_moe", shared=1, k=K_EXP, exact=True, num_experts=N_EXP)
    n = len(fed)
    pos = np.arange(n)
    for l in range(L):
        assert tr.get("router_ids", l).shape == (n, K_EXP)
        np.testing.assert_array_equal(tr.get("router_ids", l)[:, 0], (pos * 3 + l) % N_EXP)
        np.testing.assert_allclose(tr.get("hc_stream_norm", l)[:, 0], pos + l / 10, rtol=1e-6)
        assert tr.get("router_probs", l).dtype == np.float16 and (tr.get("router_probs", l) == np.float16(1 / K_EXP)).all()
    # The head reads a forward's last row: the prompt chunks' (positions 3 and 5) and every reply token's.
    read = np.isfinite(tr.get("resid_norm", 2))
    assert np.nonzero(read)[0].tolist() == [3, 5, 6, 7, 8]
    assert lens and not tr.kinds() & READOUT_KINDS and "vocab" not in tr.meta      # the read-outs: the live stream's only


def test_the_live_stream_announces_each_reply_token(tmp_path):
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None)
    mgr.register("rid-2", "on", dict(conversation="c2", turn=1, client="pi-1"))
    path = tmp_path / "r.bin"
    write_request(path, [1, 2, 3, 4, 5], [11, 12, 13, 0])
    run = Run("rid-2", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    loop.run_until_complete(asyncio.sleep(0.05))
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    tokens = [e for e in events if e["type"] == "token"]
    assert [e["id"] for e in tokens] == [11, 12, 13] and tokens[0]["p"] == pytest.approx(0.9, abs=1e-3)
    assert tokens[0]["lens"] and tokens[0]["lens_by"] == "logit" and events[-1]["type"] == "end"
    loop.close()


def test_a_replayed_forward_files_only_its_new_rows(tmp_path):
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    mgr.register("rid-3", "on", dict(conversation="c3", turn=1))
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [0])
    w.prompt([1, 2, 3])
    w.step(0, [1, 2, 3], out_top=9)
    w.step(1, [2, 3, 9], out_top=10)       # replay from position 1: only position 3 is new
    w.end([9, 0])
    run = Run("rid-3", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    wait_saved(mgr, 1)
    tr = Trace.load(read_index()[0]["path"])
    assert tr.tokens == [1, 2, 3, 9]
    np.testing.assert_array_equal(tr.get("router_ids", 0)[:, 0], np.arange(4) * 3 % N_EXP)


def test_a_request_that_reused_a_prompt_prefix_keeps_the_whole_prompt_and_records_from_its_first_fed_row(tmp_path):
    """ds4 reused the first 6 prompt rows its session held (an earlier turn read them): the request fed positions 6 .. on.
    The trace holds every prompt token, its records start empty (-1 / NaN) and say where reading began (read_from);
    the prompt's last row still predicts the reply's first token, and the live stream reads the reply as usual."""
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None)
    mgr.register("rid-p", "on", dict(conversation="cp", turn=2))
    path = tmp_path / "r.bin"
    prompt, replies, N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 0], 6
    lens = CaptureConfig(lens_every=4, lens_last=2).lens_layers(L)
    w = ScopeWriter(path, lens)
    w.prompt(prompt)
    w.step(N, prompt[N:], out_top=replies[0])           # the new prompt suffix, in one forward
    for j, tok in enumerate(replies[:-1]):
        w.step(len(prompt) + j, [tok], out_top=replies[j + 1])
    w.end(replies)
    run = Run("rid-p", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    wait_saved(mgr, 1)
    (entry,) = read_index()
    tr = Trace.load(entry["path"])
    fed = prompt + replies[:-1]
    assert tr.tokens == fed and tr.prompt_len == len(prompt) and entry["prompt_tokens"] == len(prompt)
    assert tr.meta["read_from"] == N and tr.read_from == N
    assert len(tr.token_text) == len(fed)
    n, pos = len(fed), np.arange(len(fed))
    for l in range(L):
        ids = tr.get("router_ids", l)
        assert ids.shape == (n, K_EXP) and (ids[:N] == -1).all()
        np.testing.assert_array_equal(ids[N:, 0], (pos[N:] * 3 + l) % N_EXP)
        sn = tr.get("hc_stream_norm", l)
        assert np.isnan(sn[:N]).all()
        np.testing.assert_allclose(sn[N:, 0], pos[N:] + l / 10, rtol=1e-6)
    assert (tr.get("input_ids")[:N] == -1).all() and tr.get("input_ids")[N:].tolist() == fed[N:]
    loop.run_until_complete(asyncio.sleep(0.05))
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    tokens = [e for e in events if e["type"] == "token"]
    assert [e["id"] for e in tokens] == [11, 12, 13] and tokens[0]["p"] == pytest.approx(0.9, abs=1e-3)
    assert tokens[0]["lens"]                                  # the prompt's last row read the reply's first token
    assert [e["read"] for e in events if e["type"] == "prefill"] == [len(prompt) - N - 1]
    loop.close()


def test_a_trace_saved_before_read_from_reads_from_its_start(tmp_path):
    Trace({"router_ids/0": np.zeros((3, 2), np.int32)}, [1, 2, 3], ["a", "b", "c"], 2, dict(model="m")).save(tmp_path / "t")
    assert Trace.load(tmp_path / "t").read_from == 0


def test_steering_file_layout(tmp_path):
    p = tmp_path / "s.steer"
    v = {3: np.arange(HC * E, dtype=np.float32), 1: np.ones(HC * E, np.float32)}
    write_steering(p, v, HC * E)
    b = p.read_bytes()
    assert b[:8] == b"DS4STEER" and struct.unpack_from("<IIII", b, 8) == (2, 2, HC * E, 0)
    first_layer = struct.unpack_from("<I", b, 24)[0]
    assert first_layer == 1 and np.frombuffer(b, "<f4", HC * E, 28).tolist() == [1.0] * (HC * E)
    assert len(b) == 24 + 2 * (4 + 4 * HC * E)
    with pytest.raises(ValueError):
        write_steering(p, {0: np.ones(5, np.float32)}, HC * E)


def save_word(model, name, layers, seed=1):
    """A stored vector over the whole row (HC streams), as a word's is."""
    rng = np.random.default_rng(seed)
    vectors.save_vector(model, vectors.Vector(name, {l: vectors.unit(rng.standard_normal(HC * E)) for l in layers},
                                              dict(source="test", hc_count=HC)))


def test_steering_file_holds_a_plans_removals_after_its_adds(tmp_path):
    """Version 2, from a plan (a swap: two words out at layers 1-2, one pushed at 2): the adds, then per removed layer
    its orthonormal rows as the plan holds them; a removal ds4 cannot take is refused before anything is sent."""
    from metalscope import steer

    W = HC * E
    for i, name in enumerate(("soccer", "ball", "rugby")):
        save_word("tiny-ds4", name, [1, 2], seed=i)
    plan = steer.build_plan(steer.parse_header("remove~soccer/1-2+remove~ball/2+rugby@0.5/2"), "tiny-ds4", "r",
                            hc_count=HC, table={"layers": {2: 4.0}})
    p = tmp_path / "s.steer"
    write_steering(p, plan.vectors, W, plan.remove)
    b = p.read_bytes()
    assert struct.unpack_from("<IIII", b, 8) == (2, 1, W, 2)
    at = 24
    assert struct.unpack_from("<I", b, at)[0] == 2
    np.testing.assert_array_equal(np.frombuffer(b, "<f4", W, at + 4), plan.vectors[2])
    at += 4 + 4 * W
    for layer, k in ((1, 1), (2, 2)):
        assert struct.unpack_from("<II", b, at) == (layer, k)
        Q = np.frombuffer(b, "<f4", k * W, at + 8).reshape(k, W)
        np.testing.assert_array_equal(Q, np.array(plan.remove[layer]))
        np.testing.assert_allclose(Q @ Q.T, np.eye(k), atol=1e-5)
        at += 8 + 4 * k * W
    assert at == len(b)
    for bad in (np.ones((1, W + 1), np.float32), np.ones((9, W), np.float32), np.ones((0, W), np.float32)):
        with pytest.raises(ValueError, match="removal at layer 3"):
            write_steering(p, {}, W, {3: bad})


def test_gguf_vocabulary_decodes_byte_level_tokens():
    dec = {b: c for c, b in _byte_decoder().items()}
    as_chars = lambda bs: "".join(dec[x] for x in bs)
    tokens = ["<|im_start|>", as_chars(b" hello"), as_chars("é".encode()[:1]), as_chars("é".encode()[1:]), "!"]
    tok = GgufTokenizer(tokens, [3, 1, 1, 1, 1])
    assert tok.decode([0, 1]) == "<|im_start|> hello" and tok.decode([2, 3, 4]) == "é!"
    assert tok.all_special_ids == [0] and tok.decode([99]) == ""


def test_proxy_passes_requests_through_and_traces_a_traced_one(tmp_path):
    """The proxy against a stand-in ds4-server: an untraced request goes through untouched; a traced one gets
    ds4's scope options (trace file, lens layers) and is saved as a trace."""
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient
    import httpx

    from metalscope.ds4.proxy import Served, build_app
    from metalscope.live import MANAGER

    config.save_settings(mode="off")
    seen = []
    fake = FastAPI()

    @fake.get("/v1/models")
    async def fake_models():
        return {"object": "list", "data": [{"id": "deepseek-v4.1-flash", "context_length": 32768}]}

    @fake.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        seen.append(body)
        scope = body.get("scope") or {}
        if scope.get("trace"):
            threading.Thread(target=write_request, args=(scope["trace"], [1, 2, 3], [11, 0])).start()

        async def sse():
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    t = tracer("proxy-ds4")
    MANAGER.tracers[id(t)] = t
    try:
        served = Served("proxy-ds4", tmp_path / "x.gguf", "http://ds4", t)
        app = build_app(served, transport=httpx.ASGITransport(app=fake))
        with TestClient(app, base_url=PROXY) as client:
            listed = client.get("/v1/models").json()["data"][0]
            assert listed["id"] == "proxy-ds4" and listed["max_model_len"] == 32768
            assert client.get("/scope/api/ds4").json() == dict(name="proxy-ds4", gguf=str(tmp_path / "x.gguf"),
                                                               upstream="http://ds4")
            msg = {"model": "proxy-ds4", "stream": True, "messages": [{"role": "user", "content": "hello"}]}
            r = client.post("/v1/chat/completions", json=msg)
            assert r.status_code == 200 and b"[DONE]" in r.content and "scope" not in seen[-1]
            saved = MANAGER.saved
            r = client.post("/v1/chat/completions", json=msg, headers={"X-Scope-Trace": "on", "X-Scope-Trace-Client": "t1"})
            assert r.status_code == 200 and b"[DONE]" in r.content
            scope = seen[-1]["scope"]
            assert scope["topk"] == 10 and scope["lens"] == capture_config().lens_layers(L) and scope["trace"].endswith(".bin")
            wait_saved(MANAGER, saved + 1)
            entry = read_index()[0]
            assert entry["model"] == "proxy-ds4" and entry["client"] == "t1"
            assert Trace.load(entry["path"]).tokens == [1, 2, 3, 11]
            assert not list((config.home() / "ds4" / "scope" / "runs").glob("*"))      # the request's files are gone
    finally:
        MANAGER.tracers.pop(id(t), None)


def test_through_the_proxy_each_layers_writes_reach_the_trace_and_a_writes_subscriber(tmp_path):
    """What ds4 records each layer changed in each stream (STEP write_norm) is saved as resid_write_norm at every row, and
    a ``writes`` subscriber's token events carry the row that chose the token (the position before it), every layer
    in order, as on oMLX; a subscriber that did not ask gets none."""
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from fastapi.testclient import TestClient
    import httpx

    from metalscope.ds4.proxy import Served, build_app
    from metalscope.live import MANAGER

    config.save_settings(mode="off")
    fake = FastAPI()
    prompt, replies = [1, 2, 3], [11, 12, 0]

    @fake.post("/v1/chat/completions")
    async def chat(request: Request):
        scope = (await request.json()).get("scope") or {}
        if scope.get("trace"):
            threading.Thread(target=write_request, args=(scope["trace"], prompt, replies)).start()

        async def sse():
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    t = tracer("writes-ds4")
    MANAGER.tracers[id(t)] = t
    loop = asyncio.new_event_loop()
    want, plain = MANAGER.hub.subscribe(loop, "w1", writes=True), MANAGER.hub.subscribe(loop, "w1", experts=True)
    try:
        app = build_app(Served("writes-ds4", tmp_path / "x.gguf", "http://ds4", t), transport=httpx.ASGITransport(app=fake))
        with TestClient(app, base_url=PROXY) as client:
            saved = MANAGER.saved
            msg = {"model": "writes-ds4", "stream": True, "messages": [{"role": "user", "content": "hello"}]}
            r = client.post("/v1/chat/completions", json=msg, headers={"X-Scope-Trace": "on", "X-Scope-Trace-Client": "w1"})
            assert r.status_code == 200
            wait_saved(MANAGER, saved + 1)
        tr = Trace.load(read_index()[0]["path"])
        fed = prompt + replies[:-1]
        assert tr.tokens == fed
        pos = np.arange(len(fed))
        for l in range(L):
            np.testing.assert_allclose(tr.get("resid_write_norm", l), write_norm(pos[:, None], l, np.arange(HC)), rtol=1e-6)
        loop.run_until_complete(asyncio.sleep(0.05))
        got = {}
        for q in (want, plain):
            got[q] = []
            while not q.empty():
                e = q.get_nowait()
                if e["type"] == "token":
                    got[q].append(e)
        tokens = got[want]
        assert [e["id"] for e in tokens] == [11, 12] and all(e["write_k"] == HC for e in tokens)
        for e, row in zip(tokens, (2, 3)):         # 11 was chosen at the prompt's last row, 12 at 11's
            assert len(e["writes"]) == L * HC
            assert e["writes"] == pytest.approx(write_norm(row, np.arange(L)[:, None], np.arange(HC)).ravel().tolist())
            # Each layer's whole write over the whole of the streams it read (the layer before's: row + (l - 1) / 10
            # each, ScopeWriter.step); layer 0's not recorded.
            whole = np.linalg.norm(write_norm(row, np.arange(L)[:, None], np.arange(HC)), axis=-1)
            assert e["shares"][0] == 0
            assert e["shares"][1:] == pytest.approx(whole[1:] / (np.sqrt(HC) * (row + np.arange(L - 1) / 10)), rel=5e-3)
        assert [e["id"] for e in got[plain]] == [11, 12] and not any("writes" in e or "shares" in e for e in got[plain])
    finally:
        MANAGER.hub.unsubscribe(want)
        MANAGER.hub.unsubscribe(plain)
        MANAGER.tracers.pop(id(t), None)
        loop.close()


def test_two_requests_at_two_lens_steps_ask_ds4_for_two_layer_sets(monkeypatch, tmp_path):
    """The lens step is the server's setting: each traced request sends ds4 the step's layers (the logit lens's in the
    scope, the J-lens's in the file it names), and its trace saves the step and its decode speed.
    The speed is the stream's: a stand-in ds4-server writes the capture rows at once and streams a token every 40 ms
    after 0.2 s on the prompt (25 tok/s)."""
    import json
    import mlx.core as mx
    from fastapi.testclient import TestClient
    import httpx

    from metalscope import jlens as J
    from metalscope.ds4 import vocab
    from metalscope.ds4.proxy import Served, build_app
    from metalscope.live import MANAGER

    J.save(J.Lens({l: mx.full((E, HC * E), 0.01) for l in range(L)}, dict(checkpoint="steps-ds4")), J.lenses_dir("steps-ds4") / "steps-learned")
    J.use("steps-learned", "steps-ds4")
    monkeypatch.setattr(vocab, "gguf_rows", lambda gguf, tensor, ids: np.ones((len(ids), E), np.float32))
    config.save_settings(mode="on")
    asked = {}

    async def chat(request: httpx.Request):
        body = json.loads(request.content)
        scope = body["scope"]
        raw, per = Path(scope["jlens"]).read_bytes(), 4 + 2 * E * HC * E
        jl = [struct.unpack_from("<I", raw, 24 + j * per)[0] for j in range(struct.unpack_from("<I", raw, 12)[0])]
        asked[body["messages"][-1]["content"]] = (jl, list(scope["lens"]))
        w = ScopeWriter(scope["trace"], scope["lens"], jlens=jl)
        w.prompt([1, 2, 3])
        w.step(0, [1, 2, 3], out_top=11)
        for j in range(12):
            w.step(3 + j, [11 + j], out_top=12 + j)
        w.end(list(range(11, 24)) + [0])

        async def sse():
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'      # before the prompt is read
            await asyncio.sleep(0.2)
            for j in range(13):                                                 # 11..23; the stop token 0 is not sent
                if j:
                    await asyncio.sleep(0.04)
                yield b'data: {"choices":[{"delta":{"content":"w"}}]}\n\n'
            yield b"data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse())

    t = Ds4Tracer("steps-ds4", Tok(), "qwen4", L, HC, E, K_EXP, N_EXP, gguf=Path("x.gguf"))
    t.install(None)
    MANAGER.tracers[id(t)] = t
    try:
        app = build_app(Served("steps-ds4", tmp_path / "x.gguf", "http://ds4", t), transport=httpx.MockTransport(chat))
        steps = t.lens_steps()
        assert steps[0][0] != steps[4][0]
        with TestClient(app, base_url=PROXY, client=("127.0.0.1", 50000)) as client:     # the /scope routes: this machine
            for k, step in enumerate((0, 4)):
                saved = MANAGER.saved
                r = client.post("/scope/api/lens/layers", json=dict(step=step))
                assert r.status_code == 200 and r.json()["at"] == step, r.text
                assert client.post("/v1/chat/completions", json={"model": "steps-ds4", "messages": [
                    {"role": "user", "content": f"turn {k}"}]}).status_code == 200
                wait_saved(MANAGER, saved + 1)
            by = {e["title"]: e for e in read_index()[:2]}
            for k, step in enumerate((0, 4)):
                e = by[f"turn {k}"]
                assert asked[f"turn {k}"] == (steps[step][0], steps[step][1])
                assert e["lens_step"] == step and e["decode_tps"] == pytest.approx(25.0, rel=0.15)
            got = client.get("/scope/api/lens/layers", params=dict(model="steps-ds4")).json()
            assert got["at"] == 4 and [s["n"] for s in got["steps"]] == [len(j) for j, _ in steps]
            assert got["steps"][4]["replies"] == 1 and got["steps"][4]["tps"] == by["turn 1"]["decode_tps"]
            assert got["steps"][1]["tps"] is None and client.post("/scope/api/lens/layers", json=dict(step=5)).status_code == 400
            from metalscope import jlens
            assert t.lens is not None and got["edit_band"] == list(jlens.workspace_band(L))      # where the word edit acts
            # A step's speed: the mean of the newest 5 unsteered replies read at it.
            from metalscope.store import append_index
            for k, tps in enumerate([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 99.0]):
                (tmp_path / f"t{k}").mkdir()
                append_index(dict(id=f"t{k}", path=str(tmp_path / f"t{k}"), time=time.time() + 10 + k, model="steps-ds4", lens_step=1,
                                  decode_tps=tps, steer="joy 0.3" if tps == 99.0 else None))
            s1 = client.get("/scope/api/lens/layers", params=dict(model="steps-ds4")).json()["steps"][1]
            assert s1 == dict(n=len(steps[1][0]), layers=steps[1][0], tps=40.0, replies=5)       # 20-60, newest first
    finally:
        MANAGER.tracers.pop(id(t), None)


def test_with_the_lenses_off_ds4_is_asked_for_no_lens_and_the_trace_keeps_routing_and_probabilities(monkeypatch, tmp_path):
    """lens_step LENS_OFF on ds4: the route says so (at -1), a traced request sends ds4 no lens layers and no J-lens
    file, and its trace keeps the routing and the output probabilities, the step saved with it. A step past off is
    refused. Started again while off, the J-lens files are kept (+ reads one again unwritten)."""
    import json
    import mlx.core as mx
    from fastapi.testclient import TestClient
    import httpx

    from metalscope import jlens as J
    from metalscope.ds4 import proxy, vocab
    from metalscope.ds4.proxy import Served, build_app
    from metalscope.live import MANAGER
    from metalscope.store import Trace

    J.save(J.Lens({l: mx.full((E, HC * E), 0.01) for l in range(L)}, dict(checkpoint="off-ds4")), J.lenses_dir("off-ds4") / "off-learned")
    J.use("off-learned", "off-ds4")
    monkeypatch.setattr(vocab, "gguf_rows", lambda gguf, tensor, ids: np.ones((len(ids), E), np.float32))
    config.save_settings(mode="on")
    asked = []

    async def chat(request: httpx.Request):
        scope = json.loads(request.content)["scope"]
        asked.append(scope)
        w = ScopeWriter(scope["trace"], scope["lens"])
        w.prompt([1, 2, 3])
        w.step(0, [1, 2, 3], out_top=11)
        for j in range(3):
            w.step(3 + j, [11 + j], out_top=12 + j)
        w.end([11, 12, 13, 14, 0])

        async def sse():
            for _ in range(4):
                yield b'data: {"choices":[{"delta":{"content":"w"}}]}\n\n'
            yield b"data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse())

    t = Ds4Tracer("off-ds4", Tok(), "qwen4", L, HC, E, K_EXP, N_EXP, gguf=Path("x.gguf"))
    t.install(None)
    MANAGER.tracers[id(t)] = t
    loop = asyncio.new_event_loop()
    q = MANAGER.hub.subscribe(loop, None, experts=True, lens=True)
    try:
        app = build_app(Served("off-ds4", tmp_path / "x.gguf", "http://ds4", t), transport=httpx.MockTransport(chat))
        with TestClient(app, base_url=PROXY, client=("127.0.0.1", 50000)) as client:
            assert client.post("/scope/api/lens/layers", json=dict(step=-2)).status_code == 400
            r = client.post("/scope/api/lens/layers", json=dict(step=config.LENS_OFF))
            assert r.status_code == 200 and r.json()["at"] == config.LENS_OFF and len(r.json()["steps"]) == config.LENS_STEPS
            assert client.get("/scope/api/lens/layers", params=dict(model="off-ds4")).json()["at"] == config.LENS_OFF
            saved = MANAGER.saved
            assert client.post("/v1/chat/completions", json={"model": "off-ds4", "messages": [
                {"role": "user", "content": "off"}]}).status_code == 200
            wait_saved(MANAGER, saved + 1)
        assert asked[0]["lens"] == [] and "jlens" not in asked[0] and t.read_layers() == ([], [])
        e = read_index()[0]
        assert e["lens_step"] == config.LENS_OFF
        assert "router_ids" in Trace.load(e["path"]).kinds()
        loop.run_until_complete(asyncio.sleep(0.05))
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        tokens = [e for e in events if e["type"] == "token"]
        assert [e["id"] for e in tokens] == [11, 12, 13, 14]
        assert all(e["p"] is not None for e in tokens) and not any(e.get("lens") for e in tokens)
        assert sum(bool(e.get("experts")) for e in tokens) >= 3
    finally:
        MANAGER.hub.unsubscribe(q)
        loop.close()
        MANAGER.tracers.pop(id(t), None)
    # Started again while off: no J-lens file named, none removed.
    kept = config.home() / "ds4" / "scope" / "jlens-off-learned-abc.bin"
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_bytes(b"x")
    monkeypatch.setattr(proxy, "model_shape", lambda gguf: dict(family="qwen4", n_layer=L, n_hc=HC, n_embd=E, k=K_EXP, n_expert=N_EXP))
    monkeypatch.setattr(proxy, "tokenizer_for", lambda gguf: Tok())
    served = proxy.make_served(Path("x.gguf"), "off-ds4", "http://ds4")
    try:
        assert served.tracer.lens_step == config.LENS_OFF and served.tracer.jlens_file() is None and kept.exists()
    finally:
        MANAGER.tracers.pop(id(served.tracer), None)


def test_the_proxy_starts_on_the_saved_lens_step_and_keeps_only_its_j_lens_file(monkeypatch):
    """A saved lens step other than the default: the tracer starts on it, its J-lens file is the one kept (others
    removed), and starting again neither removes nor rewrites it."""
    import mlx.core as mx
    from metalscope import jlens as J
    from metalscope.ds4 import proxy, vocab
    from metalscope.live import MANAGER
    J.save(J.Lens({l: mx.full((E, HC * E), 0.01) for l in range(L)}, dict(checkpoint="keep-ds4")), J.lenses_dir("keep-ds4") / "keep-learned")
    J.use("keep-learned", "keep-ds4")
    monkeypatch.setattr(vocab, "gguf_rows", lambda gguf, tensor, ids: np.ones((len(ids), E), np.float32))
    monkeypatch.setattr(proxy, "model_shape", lambda gguf: dict(family="qwen4", n_layer=L, n_hc=HC, n_embd=E, k=K_EXP, n_expert=N_EXP))
    monkeypatch.setattr(proxy, "tokenizer_for", lambda gguf: Tok())
    config.save_settings(lens_step=4)
    stale = config.home() / "ds4" / "scope" / "jlens-keep-learned-old.bin"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"x")
    served = proxy.make_served(Path("x.gguf"), "keep-ds4", "http://ds4")
    try:
        t = served.tracer
        kept = t.jlens_file()
        assert t.lens_step == 4 and not stale.exists() and kept.exists()
        assert struct.unpack_from("<I", kept.read_bytes(), 12)[0] == len(t.lens_steps()[4][0])
        stamp = kept.stat().st_mtime_ns
        MANAGER.tracers.pop(id(t), None)
        again = proxy.make_served(Path("x.gguf"), "keep-ds4", "http://ds4")
        assert again.tracer.jlens_file() == kept and kept.stat().st_mtime_ns == stamp
        MANAGER.tracers.pop(id(again.tracer), None)
    finally:
        MANAGER.tracers.pop(id(served.tracer), None)


def test_ds4_serve_gives_ds4_server_what_follows_the_dashes(monkeypatch, tmp_path):
    from metalscope import cli
    from metalscope.ds4 import proxy
    (tmp_path / "metal").mkdir()
    (tmp_path / "ds4-server").write_text("x")
    monkeypatch.setenv("METALSCOPE_DS4_DIR", str(tmp_path))       # a build, wherever this runs
    got = {}
    monkeypatch.setattr(proxy, "serve", lambda gguf, name, port, ds4_dir, ds4_port, extra: got.update(
        gguf=str(gguf), name=name, port=port, extra=extra))
    cli.main(["ds4", "serve", "m.gguf", "--name", "n-ds4", "--port", "9001", "--", "--ctx", "8192", "--name", "x"])
    assert got == dict(gguf="m.gguf", name="n-ds4", port=9001, extra=["--ctx", "8192", "--name", "x"])
    cli.main(["ds4", "serve", "m.gguf"])
    assert got["extra"] == [] and got["name"] is None


def test_an_mtp_verify_files_each_kept_row_as_a_reply_token(tmp_path):
    """A verify forward fed two tokens and kept both (R = 2 head rows): each row is its own step with its own
    prediction, in the trace and in the live stream."""
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None)
    mgr.register("rid-v", "on", dict(conversation="cv", turn=1))
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [0, 5])
    w.prompt([1, 2, 3])
    w.step(0, [1, 2, 3], out_top=11)            # prefill: predicts 11
    w.step(3, [11, 12], out_top=[12, 13], head_rows=2)   # verify of 11 and the draft 12, both kept
    w.step(5, [13], out_top=0)
    w.end([11, 12, 13, 0])
    run = Run("rid-v", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    wait_saved(mgr, 1)
    tr = Trace.load(read_index()[0]["path"])
    assert tr.tokens == [1, 2, 3, 11, 12, 13]
    rn = tr.get("resid_norm", 2)
    assert np.isnan(rn[:2]).all() and rn[2:].tolist() == [3.0, 3.0, 4.0, 3.0]     # verify rows r = 0, 1
    loop.run_until_complete(asyncio.sleep(0.05))
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    # Each row its own prediction: every token announced was the top of the row before it.
    assert [(e["id"], e["p"]) for e in events if e["type"] == "token"] == [(11, 0.9), (12, 0.9), (13, 0.9)]
    loop.close()


def test_rows_fed_again_after_a_rewind_replace_the_dropped_ones(tmp_path):
    """The server dropped a verify's second row (RWND) and fed another token there: the trace holds the new row."""
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    mgr.register("rid-w", "on", dict(conversation="cw", turn=1))
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [0])
    w.prompt([1, 2, 3])
    w.step(0, [1, 2, 3], out_top=11)
    w.step(3, [11, 12], out_top=[12, 13], head_rows=2)
    w.rec(b"RWND", struct.pack("<I", 4))
    w.step(4, [99], out_top=0)
    w.end([11, 99, 0])
    run = Run("rid-w", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    wait_saved(mgr, 1)
    tr = Trace.load(read_index()[0]["path"])
    assert tr.tokens == [1, 2, 3, 11, 99]
    assert tr.get("input_ids").tolist() == [1, 2, 3, 11, 99]
    np.testing.assert_array_equal(tr.get("router_ids", 0)[:, 0], np.arange(5) * 3 % N_EXP)


def test_text_the_server_inserts_into_the_reply_is_part_of_it(tmp_path):
    """An invalid tool call: ds4 dropped the unfed stop token, fed a tool-error suffix (SUFX) and decoded on. The
    reply as recorded is the one fed, so every row stays in place, and the trace says where the inserted text is."""
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    mgr.register("rid-s", "on", dict(conversation="cs", turn=1))
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [0])
    w.prompt([1, 2, 3])
    w.step(0, [1, 2, 3], out_top=11)
    w.step(3, [11], out_top=12)                     # 12: the stop token, never fed
    w.rec(b"SUFX", struct.pack("<II", 4, 2))
    w.step(4, [50, 51], out_top=13)
    w.step(6, [13], out_top=0)
    w.end([11, 50, 51, 13, 0])
    run = Run("rid-s", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    wait_saved(mgr, 1)
    tr = Trace.load(read_index()[0]["path"])
    assert tr.tokens == [1, 2, 3, 11, 50, 51, 13] and tr.get("input_ids").tolist() == tr.tokens
    assert tr.meta["inserted"] == [[4, 2]]
    np.testing.assert_array_equal(tr.get("router_ids", 0)[:, 0], np.arange(7) * 3 % N_EXP)   # every row in place


def test_the_proxy_traces_anthropic_messages_and_responses(tmp_path):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    import httpx

    from metalscope.ds4.proxy import Served, build_app
    from metalscope.live import MANAGER

    config.save_settings(mode="off")
    seen = {}
    fake = FastAPI()

    @fake.post("/v1/{rest:path}")
    async def any_route(request: Request, rest: str):
        body = await request.json()
        seen[rest] = body
        if (body.get("scope") or {}).get("trace"):
            write_request(body["scope"]["trace"], [1, 2, 3], [11, 0])
        return JSONResponse({"ok": True})

    t = tracer("proxy-ds4b")
    MANAGER.tracers[id(t)] = t
    try:
        app = build_app(Served("proxy-ds4b", tmp_path / "x.gguf", "http://ds4", t), transport=httpx.ASGITransport(app=fake))
        with TestClient(app, base_url=PROXY) as client:
            saved = MANAGER.saved
            on = {"X-Scope-Trace": "on"}
            r = client.post("/v1/messages", headers=on, json={
                "model": "proxy-ds4b", "system": "be brief", "max_tokens": 8,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]})
            assert r.status_code == 200 and seen["messages"]["scope"]["trace"].endswith(".bin")
            r = client.post("/v1/responses", headers=on, json={"model": "proxy-ds4b", "input": "hi there"})
            assert r.status_code == 200 and "trace" in seen["responses"]["scope"]
            wait_saved(MANAGER, saved + 2)
            titles = {e["title"] for e in read_index()[:2]}
            assert titles == {"hello", "hi there"}
    finally:
        MANAGER.tracers.pop(id(t), None)


def test_the_j_lens_reaches_the_live_stream(tmp_path):
    """ds4 read the J-lens at layers 0 and 4: the live stream has the J-lens's words (the trace keeps no read-out)."""
    import mlx.core as mx

    from metalscope import jlens as J

    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    lens = J.Lens({0: mx.zeros((E, HC * E)), 3: mx.zeros((E, HC * E)), 4: mx.zeros((E, HC * E))})
    t.lens = J.LensSetup(lens, [3], name="tiny-lens")
    t._loaded = True                      # set by hand: install() would read them from the (empty) home
    mgr.tracers[id(t)] = t
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None, lens=True)
    mgr.register("rid-m", "on", dict(conversation="cm", turn=1))
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [0, 5], jlens=[0, 4])
    w.prompt([1, 2, 3])
    w.step(0, [1, 2, 3], out_top=11)
    w.step(3, [11], out_top=12)
    w.step(4, [12], out_top=0)
    w.end([11, 12, 0])
    run = Run("rid-m", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    wait_saved(mgr, 1)
    loop.run_until_complete(asyncio.sleep(0.05))
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    tokens = [e for e in events if e["type"] == "token"]
    assert [e["id"] for e in tokens] == [11, 12]
    first = tokens[0]                                         # read on the row before it: position 2
    assert first["lens_by"] == "jlens" and first["lens"][-1][:2] == [4, 104] and first["lens10"]
    loop.close()


def test_a_ds4_tracer_reads_its_checkpoints_j_lens(monkeypatch):
    """The lens its checkpoint reads gives the ds4 model a J-lens: the layers oMLX reads and the file ds4 loads."""
    import mlx.core as mx

    from metalscope import jlens as J

    layers = [0, 1, 2, 3, 4, 5]
    J.save(J.Lens({l: mx.full((E, HC * E), 0.01 * (l + 1)) for l in layers}, dict(checkpoint="lens-ds4")),
           J.lenses_dir("lens-ds4") / "tiny-learned")
    J.use("tiny-learned", "lens-ds4")
    t = Ds4Tracer("lens-ds4", Tok(), "qwen4", L, HC, E, K_EXP, N_EXP, gguf=Path("x.gguf"))
    t.install(None)
    assert t.lens is not None and t.lens.name == "tiny-learned"
    ws = t.lens.workspace
    jl = t.read_layers()[0]
    assert jl == sorted({0, 4} | set(ws) | {5})                # the default step: the layers traces always read
    f = t.jlens_file()
    assert f.parent == config.home() / "ds4" / "scope"         # where ds4-server may open it (--scope-dir)
    head = f.read_bytes()[:24]
    assert head[:8] == b"DS4JLENS" and struct.unpack("<IIII", head[8:]) == (1, len(jl), HC * E, E)
    assert f.stat().st_size == 24 + len(jl) * (4 + 2 * E * HC * E) and t.jlens_file() == f
    t2 = Ds4Tracer("other-ds4", Tok(), "qwen4", L, HC, E, K_EXP, N_EXP, gguf=Path("x.gguf"))
    t2.install(None)
    assert t2.lens is None and t2.jlens_file() is None


def test_mtp_verify_rows_and_an_unknown_record_leave_the_reply_tokens(tmp_path):
    """Qwen3.8's MTP on ds4: the verify at 3 keeps 11 and 12, and the model's own 14 is fed at 5. A record with a tag
    the reader does not know is skipped by its length, so the trace and the live stream hold the reply tokens."""
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    loop = asyncio.new_event_loop()
    q = mgr.hub.subscribe(loop, None)
    mgr.register("rid-d", "on", dict(conversation="cd", turn=1))
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [0])
    w.prompt([1, 2, 3])
    w.step(0, [1, 2, 3], out_top=11)
    w.rec(b"XTRA", b"\0" * 12)                             # a tag the reader does not know
    w.step(3, [11, 12], out_top=[12, 14], head_rows=2)     # the verify's kept rows
    w.step(5, [14], out_top=0)
    w.end([11, 12, 14, 0])
    run = Run("rid-d", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    wait_saved(mgr, 1)
    tr = Trace.load(read_index()[0]["path"])
    assert tr.tokens == [1, 2, 3, 11, 12, 14]
    assert not any(k.startswith("mtp") for k in [*tr.kinds(), *tr.meta])
    loop.run_until_complete(asyncio.sleep(0.05))
    tokens = []
    while not q.empty():
        e = q.get_nowait()
        if e["type"] == "token":
            tokens.append(e)
    assert [e["id"] for e in tokens] == [11, 12, 14]
    assert not any(k.startswith("mtp") for e in tokens for k in e)
    loop.close()


def test_a_ds4_model_linked_to_a_checkpoint_reads_its_vectors_profiles_and_j_lens_and_keeps_its_own_norms():
    """No copies: the ds4 name linked to the checkpoint oMLX serves finds the same weights' files; its strength
    table is its own (served/<name>/norms.json), never the oMLX copy's."""
    import mlx.core as mx

    from metalscope import jlens as J, models

    models.check_shape("src-omlx", models.shape_of(L, HC, E))
    J.save(J.Lens({0: mx.zeros((E, HC * E))}, dict(checkpoint="src-omlx")), J.lenses_dir("src-omlx") / "src-omlx")
    vec = models.vector_path("src-omlx", "joy")
    vec.parent.mkdir(parents=True)
    vec.write_bytes(b"x")
    prof = models.profile_path("src-omlx", "joy")
    prof.write_text("{}")
    models.link("dst-ds4", "src-omlx")
    assert vectors.find_vector("dst-ds4", "joy") == vec == models.vector_path("dst-ds4", "joy")
    assert models.profile_path("dst-ds4", "joy") == prof and models.norms_path("dst-ds4") != models.norms_path("src-omlx")
    assert models.norms_path("dst-ds4") == models.served_dir("dst-ds4") / "norms.json"
    assert J.find_for("dst-ds4") == J.find_for("src-omlx") == J.lenses_dir("src-omlx") / "src-omlx"
    J.use(J.LOGIT, "dst-ds4")                                                    # the switch is the checkpoint's
    assert J.find_for("src-omlx") is None


def test_the_ds4_proxy_refuses_by_capability_and_says_what_it_serves(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from metalscope import backends, middleware, web
    from metalscope.ds4 import proxy
    from metalscope.learn import spawn
    from metalscope.live import MANAGER

    t = tracer("proxy-ds4c")
    MANAGER.tracers[id(t)] = t
    started = []
    monkeypatch.setattr(spawn, "learn", lambda *a, **k: started.append(a + tuple(k.values())) or dict(job="x-1", pid=1))
    try:
        app = FastAPI()
        web.mount(app, proxy.BACKEND)
        app.dependency_overrides[middleware._local_only] = lambda: None
        with TestClient(app) as client:
            for path in ("/scope/api/lens/learn",):
                r = client.post(path, json={"model": "proxy-ds4c"})
                assert r.status_code == 409 and "loads its own copy" in r.json()["detail"], path
                assert "not available on this server" in r.json()["detail"] and "ds4" not in r.json()["detail"]
            # /concept learn measures the served model here (measure_served), asking this server.
            r = client.post("/scope/api/concept/learn", json={"name": "x", "description": "y", "model": "proxy-ds4c"})
            assert r.status_code == 200 and started == [("x", "y", "proxy-ds4c", None, "ds4", "http://testserver:80", None)]
            # concept delete is the same route here (nothing of x on this checkpoint: 404, not a missing route's).
            r = client.post("/scope/api/concept/delete", json={"name": "x", "model": "proxy-ds4c"})
            assert r.status_code == 404 and r.json()["detail"] == "x has nothing on proxy-ds4c"
            r = client.post("/scope/api/steer/words", json={"name": "w", "words": ["cat"], "model": "proxy-ds4c"})
            assert r.status_code == 400 and "no Jacobian lens" in r.json()["detail"]        # served here, given a lens
            monkeypatch.setattr(web, "_BOOT", web.code_stamp())          # the code it runs is the code installed
            st = client.get("/scope/api/status").json()
            assert (st["engine"], st["caps"]) == ("ds4", ["edit_at", "measure_served", "token_prompts"])
            assert st["stale"] is None
            # The package's source changed on disk since the proxy started (web.code_stamp): its restart.
            monkeypatch.setattr(web, "_BOOT", web._BOOT + 1)
            assert client.get("/scope/api/status").json()["stale"] == "metalscope ds4 stop, then metalscope ds4"
            # Each model's flags: run on its family on ds4 (models.TESTED), and the edit (edit_at): as `metalscope
            # status` lists a ds4 model not served (backends.ds4_chat), so Pi's dock reads the same either way.
            assert st["chat"] == {"proxy-ds4c": dict(reasoning=True, thinking_format="qwen-chat-template", tested=True, edit=True)}
            assert st["chat"]["proxy-ds4c"] == backends.ds4_chat("qwen4")
            t.family = "qwen5"                                               # a family never run here: marked, not refused
            assert client.get("/scope/api/status").json()["chat"]["proxy-ds4c"]["tested"] is False
    finally:
        MANAGER.tracers.pop(id(t), None)


def test_a_typed_word_gets_its_lens_direction_on_ds4_from_the_ggufs_head(monkeypatch):
    """POST /scope/api/steer/words on the proxy: the same vector oMLX builds (vectors.from_lens), the head's rows read
    from the GGUF through the tracer; the response has a removal's band and the words cut to a token."""
    import mlx.core as mx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from metalscope import jlens as J, middleware, web
    from metalscope.ds4 import proxy, vocab
    from metalscope.live import MANAGER

    asked = []

    def rows(gguf, tensor, ids):
        asked.append((gguf, tensor, list(ids)))
        return np.stack([np.arange(E, dtype=np.float32) + i for i in ids])
    monkeypatch.setattr(vocab, "gguf_rows", rows)

    class Pieces(Tok):
        def encode(self, text, add_special_tokens=False):          # a word of over 6 letters is two tokens
            w = text.strip()
            return [len(w)] if len(w) <= 6 else [7, 8]

    t = Ds4Tracer("word-ds4", Pieces(), "qwen4", L, HC, E, K_EXP, N_EXP, gguf=Path("x.gguf"))
    jac = {l: mx.array(np.random.default_rng(l).standard_normal((E, HC * E)).astype(np.float32)) for l in (2, 3)}
    mx.eval(list(jac.values()))                                    # no lazy arrays across threads
    t.lens = J.LensSetup(J.Lens(jac), [3], name="tiny-lens")
    MANAGER.tracers[id(t)] = t
    try:
        app = FastAPI()
        web.mount(app, proxy.BACKEND)
        app.dependency_overrides[middleware._local_only] = lambda: None
        with TestClient(app) as client:
            r = client.post("/scope/api/steer/words", json={"name": "soccer", "words": ["soccer"], "model": "word-ds4"})
            assert r.status_code == 200, r.text
            assert r.json() == dict(name="soccer", model="word-ds4", layers=[2, 3], words=["soccer"], cut=[],
                                    band=list(J.workspace_band(L)))
            r = client.post("/scope/api/steer/words", json={"name": "pineapple", "words": ["pineapple"], "model": "word-ds4"})
            assert r.json()["cut"] == ["pineapple"]
        assert asked[0] == (Path("x.gguf"), "output.weight", [6])
        vec = vectors.load_vector("word-ds4", "soccer")
        assert vec.meta["hc_count"] == HC
        want = np.array(J.concept_vectors(t.lens.lens, 2, mx.array(rows(None, None, [6])), None)).sum(0)
        np.testing.assert_allclose(vec.vectors[2], vectors.unit(want), atol=1e-6)
    finally:
        MANAGER.tracers.pop(id(t), None)


def test_the_proxy_sends_ds4_a_removal_with_no_vector_to_add(tmp_path):
    """A steered, untraced request that only removes a word: the steering file is written (version 2, no add, a block
    per layer) and named in the scope options ds4-server gets; it is gone once the request ended."""
    import json

    from fastapi import Request
    from fastapi.responses import JSONResponse

    seen = []

    async def chat(request: Request):
        scope = json.loads(await request.body())["scope"]
        seen.append((scope, Path(scope["steer"]).read_bytes()))
        return JSONResponse({"choices": [{"message": {"content": "hi"}}]})

    config.save_settings(steer=True)
    save_word("leak-ds4", "soccer", [1, 2, 3])
    t, mgr, client = _leak_proxy(tmp_path, chat)
    before = set(mgr.requests)
    try:
        with client:
            r = client.post("/v1/chat/completions", json=MSG, headers={"X-Scope-Steer": "remove~soccer/1-2;positions=decode"})
            assert r.status_code == 200, r.text
            assert r.headers["x-scope-steer-accepted"] == "remove~soccer/1-2"            # the label: parts only
        scope, sent = seen[0]
        assert scope["steer_decode_only"] is True and "trace" not in scope
        assert struct.unpack_from("<IIII", sent, 8) == (2, 0, HC * E, 2)
        assert [struct.unpack_from("<II", sent, 24 + i * (8 + 4 * HC * E)) for i in range(2)] == [(1, 1), (2, 1)]
        assert _until_released(mgr, before) == set()
        assert not Path(scope["steer"]).exists()
    finally:
        mgr.tracers.pop(id(t), None)


def test_the_proxy_steers_a_dose_without_layers_at_the_concepts_steer_layer(tmp_path):
    """ds4 takes the steer the shared middleware resolved: NAME@DOSE at the profile's reader layer only, a vector with
    no profile at all its layers."""
    import json

    from fastapi import Request
    from fastapi.responses import JSONResponse

    seen = []

    async def chat(request: Request):
        scope = json.loads(await request.body())["scope"]
        seen.append(Path(scope["steer"]).read_bytes())
        return JSONResponse({"choices": [{"message": {"content": "hi"}}]})

    config.save_settings(steer=True)
    save_word("leak-ds4", "joyful", [1, 2, 3])
    models.profile_path("leak-ds4", "joyful").write_text(json.dumps(dict(concept="joyful", reader=dict(layer=2))))
    save_word("leak-ds4", "soccer", [1, 2, 3])
    write_norms("leak-ds4", [1, 2, 3], streams=[1.0] * HC)
    t, mgr, client = _leak_proxy(tmp_path, chat)
    before = set(mgr.requests)
    try:
        with client:
            for header, accepted in (("joyful@0.4", "joyful@0.4/2"), ("soccer@0.4", "soccer@0.4")):
                r = client.post("/v1/chat/completions", json=MSG, headers={"X-Scope-Steer": header})
                assert r.status_code == 200, r.text
                assert r.headers["x-scope-steer-accepted"] == accepted
        n_add = [struct.unpack_from("<IIII", s, 8)[1] for s in seen]
        first = [struct.unpack_from("<I", s, 24)[0] for s in seen]
        assert (n_add, first) == ([1, 3], [2, 1])
        assert _until_released(mgr, before) == set()
    finally:
        mgr.tracers.pop(id(t), None)


def test_the_proxy_refuses_chat_while_a_gpu_job_measures_the_model(tmp_path):
    """A learn job on ds4 holds the GPU lock and sends its requests to ds4-server directly: meanwhile the proxy
    refuses chat with the job's message, before ds4-server sees it; the model list and tokenizing pass."""
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    import httpx

    from metalscope import gpulock
    from metalscope.ds4.proxy import Served, build_app
    from metalscope.live import MANAGER

    config.save_settings(mode="off")
    seen = []
    fake = FastAPI()

    @fake.get("/v1/models")
    async def fake_models():
        return {"object": "list", "data": [{"id": "x", "context_length": 4096}]}

    @fake.post("/v1/{rest:path}")
    async def post(rest: str, request: Request):
        seen.append(rest)
        return {"tokens": [1, 2]} if rest == "scope/tokenize" else {"choices": [{"message": {"content": "hi"}}]}

    t = tracer("lock-ds4")
    MANAGER.tracers[id(t)] = t
    msg = {"model": "lock-ds4", "messages": [{"role": "user", "content": "hello"}]}
    try:
        app = build_app(Served("lock-ds4", tmp_path / "x.gguf", "http://ds4", t), transport=httpx.ASGITransport(app=fake))
        # From this machine, as Pi asks (the /scope routes are loopback-only).
        with TestClient(app, base_url="http://127.0.0.1:8009", client=("127.0.0.1", 50000)) as client:
            lock = gpulock.GpuLock(gpulock.Holder("learning toasty", stage="selection sweep",
                                                  eta=time.time() + 600)).acquire()
            try:
                for path in ("/v1/chat/completions", "/v1/completions", "/v1/messages"):
                    r = client.post(path, json=msg)
                    assert r.status_code == 409, path
                    assert r.json()["error"]["message"] == ("Model 'lock-ds4' cannot answer now: learning toasty: selection "
                                                            "sweep, about 10 min left; chat works again when it is done")
                assert seen == []                                                  # ds4-server never saw them
                assert client.get("/v1/models").json()["data"][0]["id"] == "lock-ds4"
                assert client.post("/v1/scope/tokenize", json={"text": "hi"}).json() == {"tokens": [1, 2]}
                # metalscope's own routes answer: Pi can follow and cancel the job.
                for path in ("/scope/api/jobs/x-1/seen", "/scope/api/jobs/x-1/cancel"):
                    assert client.post(path).status_code == 200, path
                assert client.get("/scope/api/jobs").json()["gpu"]["what"] == "learning toasty"
            finally:
                lock.release()
            assert client.post("/v1/chat/completions", json=msg).status_code == 200
            assert seen == ["scope/tokenize", "chat/completions"]
    finally:
        MANAGER.tracers.pop(id(t), None)


def test_the_proxy_does_not_refuse_chat_for_a_momentary_look_at_the_lock(tmp_path):
    """Another process's look at the lock (Pi's job poll, oMLX's load guard) in progress does not refuse chat; a run
    holding it does."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import fcntl
    import os
    import httpx

    from metalscope import gpulock
    from metalscope.ds4 import proxy
    from metalscope.live import MANAGER

    config.save_settings(mode="off")
    fake = FastAPI()
    fake.post("/v1/chat/completions")(lambda: {"choices": [{"message": {"content": "hi"}}]})
    t = tracer("look-ds4")
    MANAGER.tracers[id(t)] = t
    msg = {"model": "look-ds4", "messages": [{"role": "user", "content": "hello"}]}
    try:
        app = proxy.build_app(proxy.Served("look-ds4", tmp_path / "x.gguf", "http://ds4", t),
                              transport=httpx.ASGITransport(app=fake))
        with TestClient(app, base_url=PROXY) as client:
            gpulock.GpuLock(gpulock.Holder("learning old")).acquire().release()
            fd = os.open(gpulock.lock_path(), os.O_RDONLY)
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)        # a look in progress, as gpulock.held takes it
            try:
                assert client.post("/v1/chat/completions", json=msg).status_code == 200
            finally:
                os.close(fd)
            lock = gpulock.GpuLock(gpulock.Holder("learning toasty")).acquire()
            try:
                r = client.post("/v1/chat/completions", json=msg)
            finally:
                lock.release()
            assert r.status_code == 409 and "learning toasty" in r.json()["error"]["message"]
    finally:
        MANAGER.tracers.pop(id(t), None)


def _leak_proxy(tmp_path, chat):
    """The proxy in front of a stand-in ds4-server whose chat route is ``chat``; yields (client, MANAGER)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import httpx

    from metalscope.ds4.proxy import Served, build_app
    from metalscope.live import MANAGER

    config.save_settings(mode="off")
    fake = FastAPI()
    fake.post("/v1/chat/completions")(chat)
    t = tracer("leak-ds4")
    MANAGER.tracers[id(t)] = t
    served = Served("leak-ds4", tmp_path / "x.gguf", "http://ds4", t)
    app = build_app(served, transport=httpx.ASGITransport(app=fake))
    return t, MANAGER, TestClient(app, base_url=PROXY, raise_server_exceptions=False)


def _until_released(mgr, before, timeout=15):
    end = time.time() + timeout
    while set(mgr.requests) - before and time.time() < end:
        time.sleep(0.02)
    return set(mgr.requests) - before


MSG = {"model": "leak-ds4", "messages": [{"role": "user", "content": "hello"}]}
TRACE = {"X-Scope-Trace": "on"}


@pytest.mark.parametrize("stream", [True, False])
def test_a_ds4_request_whose_forward_raises_is_not_left_active(tmp_path, stream):
    """ds4-server's exchange fails with something other than an HTTP error (the proxy saw none of its answer):
    the registered request is released at once and its files removed."""
    async def chat():
        raise RuntimeError("ds4-server fell over")

    t, mgr, client = _leak_proxy(tmp_path, chat)
    before = set(mgr.requests)
    try:
        with client:
            r = client.post("/v1/chat/completions", json=dict(MSG, stream=stream), headers=TRACE)
        assert r.status_code == 500
        assert set(mgr.requests) - before == set()
        assert not list((config.home() / "ds4" / "scope" / "runs").glob("*"))
    finally:
        mgr.tracers.pop(id(t), None)


def test_a_ds4_request_whose_setup_raises_is_not_left_active(tmp_path, monkeypatch):
    from metalscope.ds4 import proxy

    async def chat():
        pytest.fail("never forwarded")

    def broken(*a, **k):
        raise OSError("runs dir")
    monkeypatch.setattr(proxy, "Run", broken)
    t, mgr, client = _leak_proxy(tmp_path, chat)
    before = set(mgr.requests)
    try:
        with client:
            assert client.post("/v1/chat/completions", json=MSG, headers=TRACE).status_code == 500
        assert set(mgr.requests) - before == set()
    finally:
        mgr.tracers.pop(id(t), None)


def test_a_ds4_request_whose_wrap_up_raises_is_not_left_active(tmp_path, monkeypatch):
    from fastapi.responses import JSONResponse

    async def chat():
        return JSONResponse({"choices": [{"message": {"content": "hi"}}]})

    def broken(self, served):
        raise RuntimeError("reader")
    monkeypatch.setattr(Run, "finish", broken)
    t, mgr, client = _leak_proxy(tmp_path, chat)
    before = set(mgr.requests)
    try:
        with client:
            assert client.post("/v1/chat/completions", json=MSG, headers=TRACE).status_code == 200
        assert _until_released(mgr, before) == set()
    finally:
        mgr.tracers.pop(id(t), None)


def test_a_clients_scope_options_never_reach_ds4_server(tmp_path):
    """ds4-server writes and reads the files a request's "scope" object names: only the proxy sets it. A client's is
    dropped from every body, traced or not, whatever its spelling; a body that is not a JSON object is refused and
    never sent on."""
    import json

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    import httpx

    from metalscope.ds4.proxy import Served, build_app
    from metalscope.live import MANAGER

    config.save_settings(mode="off")
    seen = []
    fake = FastAPI()

    async def answer(request: Request):
        body = await request.body()
        seen.append(body)
        scope = json.loads(body).get("scope") or {}
        if scope.get("trace"):
            threading.Thread(target=write_request, args=(scope["trace"], [1, 2, 3], [11, 0])).start()
        return JSONResponse({"choices": [{"message": {"content": "hi"}}], "tokens": [1]})
    routes = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses", "/v1/scope/tokenize")
    for route in routes:
        fake.post(route)(answer)
    t = tracer("scope-ds4")
    MANAGER.tracers[id(t)] = t
    evil = {"trace": str(tmp_path / "written"), "jlens": str(tmp_path / "read"), "steer": "/etc/hosts"}
    msg = {"model": "scope-ds4", "messages": [{"role": "user", "content": "hello"}], "scope": evil}
    try:
        app = build_app(Served("scope-ds4", tmp_path / "x.gguf", "http://ds4", t), transport=httpx.ASGITransport(app=fake))
        with TestClient(app, base_url=PROXY) as client:
            for path in routes:
                assert client.post(path, json=msg).status_code == 200
                assert "scope" not in json.loads(seen[-1])
            # The key spelled with an escape, twice.
            raw = b'{"model": "scope-ds4", "prompt": "x", "\\u0073cope": {"trace": "/tmp/a"}, "scope": {"trace": "/tmp/b"}}'
            r = client.post("/v1/completions", content=raw, headers={"Content-Type": "application/json"})
            assert r.status_code == 200 and "scope" not in json.loads(seen[-1]) and b"/tmp/" not in seen[-1]
            # Traced: the proxy's own options, never the client's.
            saved = MANAGER.saved
            r = client.post("/v1/chat/completions", json=msg, headers={"X-Scope-Trace": "on"})
            assert r.status_code == 200
            scope = json.loads(seen[-1])["scope"]
            assert scope["trace"].startswith(str(config.home() / "ds4" / "scope" / "runs")) and "steer" not in scope
            assert str(tmp_path) not in seen[-1].decode()
            wait_saved(MANAGER, saved + 1)
            # Not a JSON object: refused, nothing sent on.
            n = len(seen)
            for body in (b'scope={"trace": "/tmp/c"}', b'[{"scope": {}}]', b"\xff\xfe", b"",
                         b'{"model": "m", "scope\\u0000": {"trace": "/tmp/pwn"}}',       # a C string: "scope"
                         b'{"model": "m", "sco\\u0000pe": {"trace": "/tmp/pwn"}}'):
                r = client.post("/v1/chat/completions", content=body, headers={"Content-Type": "application/json"})
                assert r.status_code == 400 and "not a JSON object" in r.json()["error"]["message"]
            assert len(seen) == n
        assert not (tmp_path / "written").exists()
    finally:
        MANAGER.tracers.pop(id(t), None)


def test_a_body_nested_too_deep_to_parse_is_not_a_json_object():
    from metalscope.ds4.proxy import client_body
    assert client_body(b"[" * 100_000) is None and client_body(b'{"a":' * 100_000) is None


def test_the_proxy_refuses_web_pages_of_other_sites_and_bodies_not_sent_as_json(tmp_path):
    """Any web page the user opens can make the browser post to 127.0.0.1: a text/plain POST needs no CORS preflight,
    and after DNS rebinding the page can read the answers too. Every proxy route refuses a Host or Origin that is not
    this machine and a cross-site fetch, and a POST to /v1 must be JSON; nothing reaches ds4-server and nothing is
    traced. Pi, curl, the OpenAI client and same-origin fetches pass."""
    import json

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    import httpx

    from metalscope.ds4.proxy import Served, build_app, is_json
    from metalscope.live import MANAGER

    config.save_settings(mode="on")           # every request traced: a page must not write into the history
    seen = []
    fake = FastAPI()

    @fake.get("/v1/models")
    async def fake_models():
        seen.append("models")
        return {"object": "list", "data": [{"id": "x", "context_length": 4096}]}

    @fake.post("/v1/chat/completions")
    async def chat(request: Request):
        body = json.loads(await request.body())
        seen.append(body)
        if (body.get("scope") or {}).get("trace"):
            threading.Thread(target=write_request, args=(body["scope"]["trace"], [1, 2, 3], [11, 0])).start()
        return JSONResponse({"choices": [{"message": {"content": "hi"}}]})

    t = tracer("guard-ds4")
    MANAGER.tracers[id(t)] = t
    msg = {"model": "guard-ds4", "messages": [{"role": "user", "content": "hello"}]}
    raw = json.dumps(msg).encode()
    try:
        app = build_app(Served("guard-ds4", tmp_path / "x.gguf", "http://ds4", t), transport=httpx.ASGITransport(app=fake))
        with TestClient(app, base_url=PROXY) as client:
            saved = MANAGER.saved
            refused = [
                ("GET", "/v1/models", dict(host="rebound.evil.example:8009")),                  # DNS rebinding
                ("GET", "/scope/api/ds4", dict(host="rebound.evil.example:8009")),
                ("POST", "/v1/chat/completions", dict(host="rebound.evil.example:8009")),
                ("POST", "/v1/chat/completions", dict(origin="https://evil.example")),           # a page elsewhere
                ("POST", "/v1/chat/completions", dict(origin="null")),                           # a sandboxed or file:// page
                ("POST", "/v1/chat/completions", {"sec-fetch-site": "cross-site"}),
                ("GET", "/v1/models", dict(origin="http://127.0.0.1.evil.example")),
            ]
            for method, path, headers in refused:
                r = client.request(method, path, content=raw if method == "POST" else None,
                                   headers={"content-type": "application/json", **headers})
                assert r.status_code == 403, (path, headers)
                assert r.json()["error"]["message"] == "metalscope answers only this machine's own clients, not another site's requests"
            # A page's "simple" POST (no preflight), a form, curl -d without -H, and no body type at all.
            for ctype in ("text/plain", "text/plain;charset=UTF-8", "application/x-www-form-urlencoded",
                          "multipart/form-data; boundary=x", None):
                r = client.post("/v1/chat/completions", content=raw, headers={"content-type": ctype} if ctype else {})
                assert r.status_code == 415, ctype
                assert "Content-Type: application/json" in r.json()["error"]["message"]
            assert seen == [] and MANAGER.saved == saved and not MANAGER.requests
            # The real clients. Pi (Node fetch through the OpenAI SDK): JSON, no Origin.
            assert client.post("/v1/chat/completions", json=msg).status_code == 200
            # curl -H "Content-Type: application/json", to localhost; the OpenAI Python client (its own headers).
            assert client.post("http://localhost:8009/v1/chat/completions", content=raw,
                               headers={"Content-Type": "application/json", "User-Agent": "curl/8.7.1"}).status_code == 200
            assert client.post("/v1/chat/completions", content=raw,
                               headers={"Content-Type": "application/json; charset=utf-8", "User-Agent": "OpenAI/Python 1.99.0",
                                        "X-Stainless-Lang": "python"}).status_code == 200
            assert client.get("http://[::1]:8009/v1/models").status_code == 200
            # Same-origin fetches name this machine.
            page = {"origin": PROXY, "sec-fetch-site": "same-origin"}
            assert client.get("/scope/api/ds4", headers=page).status_code == 200
            assert client.post("/v1/chat/completions", json=msg, headers=page).status_code == 200
            # A GET carries no body to declare.
            assert client.get("/v1/models").status_code == 200
            wait_saved(MANAGER, saved + 4)            # the four chats, traced under mode on
    finally:
        MANAGER.tracers.pop(id(t), None)
        config.save_settings(mode="off")
    assert is_json("Application/JSON ; charset=utf-8") and not is_json("application/jsonx") and not is_json(None)


def test_a_request_whose_registration_fails_is_refused_if_steered_and_served_untraced_if_not(tmp_path, monkeypatch):
    """register raises (a steer plan that cannot be built): a steered request gets oMLX's 503 with an OpenAI error
    body; a traced one is sent on untraced, its body still without a client's scope."""
    import json

    from fastapi import Request
    from fastapi.responses import JSONResponse

    from metalscope import steer
    from metalscope.live import TraceManager as TM

    seen = []

    async def chat(request: Request):
        seen.append(json.loads(await request.body()))
        return JSONResponse({"choices": [{"message": {"content": "hi"}}]})

    def broken(self, *a, **k):
        raise ValueError("no vector for toasty")
    monkeypatch.setattr(TM, "register", broken)
    monkeypatch.setattr(TM, "steer_enabled", lambda self: True)       # the header accepted as it stands
    monkeypatch.setattr(steer, "parse_header", lambda v: v)
    monkeypatch.setattr(steer, "with_defaults", lambda spec, model: spec)
    monkeypatch.setattr(steer, "check_spec", lambda spec, model: None)
    t, mgr, client = _leak_proxy(tmp_path, chat)
    before = set(mgr.requests)
    try:
        with client:
            r = client.post("/v1/chat/completions", json=dict(MSG, scope={"trace": "/tmp/x"}), headers=TRACE)
            assert r.status_code == 200 and seen and "scope" not in seen[-1]
            n = len(seen)
            r = client.post("/v1/chat/completions", json=MSG, headers={"X-Scope-Steer": "toasty@0.5"})
            assert r.status_code == 503 and len(seen) == n
            assert r.json() == {"error": {"message": "metalscope could not set up steering: no vector for toasty",
                                          "type": "metalscope"}}
        assert set(mgr.requests) - before == set()
    finally:
        mgr.tracers.pop(id(t), None)


def test_the_proxy_lists_no_model_while_ds4_server_does_not_answer(tmp_path):
    from fastapi.responses import JSONResponse

    async def chat():
        return JSONResponse({})

    t, mgr, client = _leak_proxy(tmp_path, chat)        # its stand-in has no /v1/models: a 404
    try:
        with client:
            r = client.get("/v1/models")
        assert r.status_code == 503 and "ds4-server is not answering for leak-ds4" in r.json()["error"]["message"]
    finally:
        mgr.tracers.pop(id(t), None)


def test_a_reader_error_ends_the_wait_and_the_trace_is_saved_as_partial(tmp_path, caplog):
    """The capture cannot be read on (a record cut wrong): Run stops reading at the first error, logged once, does not
    wait out END_WAIT_S for an end record, and what it read is saved marked partial with why."""
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    mgr.register("rid-e", "on", dict(conversation="ce", turn=1))
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [0])
    w.prompt([1, 2, 3])
    w.step(0, [1, 2, 3], out_top=9)
    run = Run("rid-e", t, path, manager=mgr)
    run.pump()
    w.rec(b"STEP", b"\0" * 4)                       # too short for a forward
    run.start()
    t0 = time.time()
    run.finish(True)
    assert time.time() - t0 < 5
    assert len([r for r in caplog.records if r.levelname == "ERROR" and "reading ds4's capture" in r.getMessage()]) == 1
    wait_saved(mgr, 1)
    tr = Trace.load(read_index()[0]["path"])
    (why,) = tr.meta["partial"]
    assert tr.tokens == [1, 2, 3] and why.startswith("reading ds4's capture failed: ")


def test_a_capture_with_no_end_record_is_saved_as_partial(tmp_path, monkeypatch):
    from metalscope.ds4 import backend
    monkeypatch.setattr(backend, "END_WAIT_S", 0.2)
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    mgr.register("rid-n", "on", dict(conversation="cn", turn=1))
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [0])
    w.prompt([1, 2, 3])
    w.step(0, [1, 2, 3], out_top=9)                 # ds4-server stopped here: no END_
    run = Run("rid-n", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    wait_saved(mgr, 1)
    tr = Trace.load(read_index()[0]["path"])
    assert tr.tokens == [1, 2, 3] and tr.meta["partial"] == ["ds4 wrote no end record"]


def test_a_capture_ds4_stopped_partway_is_saved_as_partial_with_why(tmp_path):
    """ds4 stopped recording mid-request (the reply went on): its END_ says why after the finish, and the trace is
    saved partial with that reason, not as complete."""
    config.save_settings(mode="on")
    mgr, t = TraceManager(), tracer()
    mgr.tracers[id(t)] = t
    mgr.register("rid-s", "on", dict(conversation="cs", turn=1))
    path = tmp_path / "r.bin"
    w = ScopeWriter(path, [0])
    w.prompt([1, 2, 3])
    w.step(0, [1, 2, 3], out_top=9)
    w.end([9, 4, 0], finish="done; trace stopped: forward wider than the staging")
    r = ScopeReader(path)
    r.poll()
    assert r.end.finish == "done; trace stopped: forward wider than the staging"
    assert r.end.stopped == "forward wider than the staging"
    run = Run("rid-s", t, path, manager=mgr)
    run.pump()
    run.finish(True)
    wait_saved(mgr, 1)
    tr = Trace.load(read_index()[0]["path"])
    assert tr.meta["partial"] == ["ds4's trace stopped: forward wider than the staging"]


def test_a_step_record_that_does_not_match_its_flags_is_refused(tmp_path):
    """A STEP whose body is longer than its flags account for, or that sets a flag this reader does not know, is an
    error, not a record read wrong."""
    for extra, flags_or in ((b"\0" * 4, 0), (b"", 8)):
        path = tmp_path / f"r{flags_or}.bin"
        w = ScopeWriter(path, [0])
        w.prompt([1, 2])
        rec = []
        w.rec = lambda tag, payload: rec.append((tag, payload))
        w.step(0, [1, 2], out_top=9)
        (tag, body), = rec
        flags = struct.unpack_from("<I", body, 8)[0] | flags_or
        ScopeWriter.rec(w, tag, body[:8] + struct.pack("<I", flags) + body[12:] + extra)
        with pytest.raises(ValueError, match="STEP"):
            ScopeReader(path).poll()


def test_a_request_file_whose_write_fails_is_removed(tmp_path, monkeypatch):
    """Each file is listed for removal before it is written: a steering file cut short by a failed write does not
    stay in the runs folder."""
    from fastapi.responses import JSONResponse

    from metalscope.ds4 import backend, proxy

    monkeypatch.setattr(backend, "END_WAIT_S", 0.2)          # the stand-in writes no capture

    def half(path, *a, **k):
        Path(path).write_bytes(b"DS4STEER")
        raise OSError("disk full")
    monkeypatch.setattr(proxy, "write_steering", half)
    config.save_settings(steer=True)
    save_word("leak-ds4", "soccer", [1, 2, 3])

    async def chat():
        return JSONResponse({"choices": [{"message": {"content": "hi"}}]})

    t, mgr, client = _leak_proxy(tmp_path, chat)
    before = set(mgr.requests)
    try:
        with client:
            r = client.post("/v1/chat/completions", json=MSG, headers=dict(TRACE, **{"X-Scope-Steer": "remove~soccer/1-2"}))
            assert r.status_code == 500
        assert _until_released(mgr, before) == set()
        assert not list((config.home() / "ds4" / "scope" / "runs").glob("*"))
    finally:
        mgr.tracers.pop(id(t), None)


def test_serve_stops_when_ds4_server_exits_and_records_its_pid(tmp_path, monkeypatch):
    """ds4-server crashes while the proxy serves: serve notices, stops the proxy and clears serving.json (status and
    Pi stop listing the model; `metalscope ds4 MODEL` starts it again). While it ran, the record named its pid. Request
    files an earlier serve left are swept as it starts."""
    import json
    import os

    import uvicorn

    from metalscope.ds4 import proxy

    runs = config.home() / "ds4" / "scope" / "runs"
    runs.mkdir(parents=True)
    old, fresh = runs / "old.bin", runs / "fresh.steer"
    old.write_text("x")
    fresh.write_text("x")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    crashed = threading.Event()

    class Proc:
        pid = 31337

        def wait(self, timeout=None):
            crashed.wait()
            return 139

        def poll(self):
            return 139          # never None: serve's stop() must not signal a real process group

    seen = {}

    class Server:
        def __init__(self, cfg):
            self.should_exit = False

        def handle_exit(self, sig, frame):
            self.should_exit = True

        def run(self):
            seen["record"] = json.loads((config.home() / "ds4" / "serving.json").read_text())
            crashed.set()                                  # ds4-server falls over
            end = time.time() + 5
            while not self.should_exit and time.time() < end:
                time.sleep(0.01)
            seen["stopped"] = self.should_exit

    t = tracer("watch-ds4")
    monkeypatch.setattr(proxy, "make_served", lambda gguf, name, up: proxy.Served("watch-ds4", gguf, up, t))
    monkeypatch.setattr(proxy, "start_ds4", lambda *a: Proc())
    monkeypatch.setattr(proxy, "wait_ready", lambda *a: None)
    monkeypatch.setattr(uvicorn, "Server", Server)
    from metalscope import backends
    monkeypatch.setattr(backends, "_serve_pids", lambda: [])          # not the machine's own serve
    monkeypatch.setattr(backends, "free_omlx", lambda log=print: None)
    with pytest.raises(SystemExit, match="ds4-server exited with 139 while serving watch-ds4"):
        proxy.serve(tmp_path / "m.gguf", None, 9999, tmp_path, 9998, [])
    assert seen["stopped"] and seen["record"]["server"] == 31337 and seen["record"]["pid"] == os.getpid()
    assert not (config.home() / "ds4" / "serving.json").exists()
    assert json.loads((config.home() / "ds4" / "models.json").read_text())["watch-ds4"]["args"] == []
    assert not old.exists() and fresh.exists()


def test_ds4_server_opens_scope_files_only_in_metalscopes_scope_dir(monkeypatch, tmp_path):
    """ds4-server's port takes a scope from any request (a page can post one), so it opens scope files only in the
    directory --scope-dir names (metalscope.patch, scope_path_inside): ds4 serve starts it with metalscope's, last so
    no option passed on replaces it, and every file a scope names (the proxy's request files, a measurement's, the
    J-lens) is written there."""
    import subprocess

    from metalscope.ds4 import proxy, scope_dir
    from metalscope.ds4.backend import runs_dir

    (tmp_path / "ds4-server").write_text("")
    seen = {}
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: seen.setdefault("cmd", cmd))
    proxy.start_ds4(tmp_path, tmp_path / "m.gguf", 9998, ["--ctx", "4096", "--scope-dir", "/"], tmp_path / "log")
    d = config.home() / "ds4" / "scope"
    assert scope_dir() == d and d.is_dir()
    assert seen["cmd"][-2:] == ["--scope-dir", str(d.resolve())] and seen["cmd"].count("--ctx") == 1
    assert runs_dir().parent == d          # a measurement's: test_ds4_engine, test_files_a_run_left_behind_are_swept
    assert d.stat().st_mode & 0o777 == 0o700                       # private: traces hold prompts and replies
    d.chmod(0o755)                                                 # made by an older metalscope
    assert scope_dir().stat().st_mode & 0o777 == 0o700


def test_ds4_servers_log_past_the_limit_is_moved_aside_at_its_start(monkeypatch, tmp_path):
    """The log keeps at most LOG_LIMIT bytes from before a start, in one ``.1`` (an older one replaced); under the
    limit it is appended to as before; a missing one is made."""
    import subprocess

    from metalscope.ds4 import proxy

    (tmp_path / "ds4-server").write_text("")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: None)
    monkeypatch.setattr(config, "LOG_LIMIT", 2000)
    log, old = tmp_path / "ds4-server.log", tmp_path / "ds4-server.log.1"
    start = lambda: proxy.start_ds4(tmp_path, tmp_path / "m.gguf", 9998, [], log)
    start()
    assert log.read_text().startswith("\n==== ") and not old.exists()
    start()
    assert log.read_text().count("====") == 2 and not old.exists()                 # 2 lines: under the limit
    old.write_text("older")
    log.write_text("x" * 2001)
    start()
    assert old.read_text() == "x" * 2001 and log.read_text().count("====") == 1


def test_serve_refuses_a_ds4_server_that_takes_no_scope(monkeypatch):
    """Without --scope-dir (or without the patch) ds4-server serves a scoped request as any other: a steered reply
    would come back unsteered with a 200. Its tokenize route answers only when it takes scopes, so serve asks it."""
    import types

    from metalscope.ds4 import proxy

    class Proc:
        returncode = None

        def poll(self):
            return None

    status = {"post": 404}
    monkeypatch.setattr(proxy.httpx, "get", lambda url, **kw: types.SimpleNamespace(status_code=200))
    monkeypatch.setattr(proxy.httpx, "post", lambda url, **kw: types.SimpleNamespace(status_code=status["post"]))
    with pytest.raises(SystemExit, match="takes no scope requests.*metalscope ds4 install"):
        proxy.wait_ready("http://127.0.0.1:1", Proc(), 5.0)
    status["post"] = 200
    proxy.wait_ready("http://127.0.0.1:1", Proc(), 5.0)


# ---------------------------------------------------------------- an edit at one token (edit.py)

class EditTok(Tok):
    """Tok with a thinking template: 3 opens the thinking, 13 closes it; 31 and 32 are the two halves of "é"; 40 a
    tool call's opener."""
    words = {3: "<think>\n", 11: "Hmm", 12: " so", 13: "</think>", 21: " well", 40: "<tool_call>"}
    raw = {31: b"\xc3", 32: b"\xa9"}

    def decode_bytes(self, ids):
        return b"".join(self.raw.get(int(i)) or self.words.get(int(i), f"<{int(i)}>").encode() for i in ids)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", "replace")


def test_chat_text_splits_reasoning_from_content_at_the_end_of_thinking():
    from metalscope.edit import ChatText, thinking_open

    t = ChatText(thinking_open("<|im_start|>assistant\n<think>\n"))
    # a piece that could begin </think> is held back until the next one says
    assert t.feed("Hmm </th") == [("reasoning_content", "Hmm ")]
    assert t.feed("ink>\n\n") == []
    assert t.feed("Done") == [("content", "Done")]
    assert t.feed(" </think> x", final=True) == [("content", " </think> x")]
    assert ChatText(thinking_open("<think>\n\n</think>\n\n")).feed("Hi") == [("content", "Hi")]
    left = ChatText(True)
    assert left.feed("cut </thi") == [("reasoning_content", "cut ")]
    assert left.feed("", final=True) == [("reasoning_content", "</thi")]


def test_an_edit_header_parses_and_a_server_that_cannot_edit_refuses_it():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from metalscope import steer
    from metalscope.middleware import TraceContextMiddleware

    e = steer.parse_edit("3f2a@57 remove~soccer/18-44+rugby@0.7/31", "m")
    assert (e["rid"], e["at"], e["steer"].label) == ("3f2a", 57, "remove~soccer/18-44+rugby@0.7")
    for bad in ("3f2a remove~soccer/1-2", "@3 remove~soccer/1-2", "3f2a@x remove~soccer/1-2", "3f2a@3",
                "3f2a@3 soccer@0.3;positions=decode"):
        with pytest.raises(ValueError):
            steer.parse_edit(bad, "m")
    app = FastAPI()
    app.post("/v1/chat/completions")(lambda: {"ok": True})
    with TestClient(TraceContextMiddleware(app), base_url="http://127.0.0.1:8008") as client:      # oMLX's: no edits
        r = client.post("/v1/chat/completions", json=MSG, headers={"X-Scope-Edit": "3f2a@1 remove~soccer/1-2"})
        assert r.status_code == 400 and "one token" in r.json()["error"]["message"]


def sse_text(pieces, finish="stop", used=3):
    """A completion stream as ds4-server sends it: ``pieces`` (str, or bytes sent as they are), its finish, usage."""
    import json
    for piece in pieces:
        text = piece if isinstance(piece, bytes) else json.dumps(piece).encode()
        yield b'data: {"choices":[{"text":' + text + b',"finish_reason":null}]}\n\n'
    yield f'data: {{"choices":[{{"text":"","finish_reason":"{finish}"}}]}}\n\n'.encode()
    yield f'data: {{"choices":[],"usage":{{"prompt_tokens":5,"completion_tokens":{used}}}}}\n\n'.encode()
    yield b"data: [DONE]\n\n"


class EditRig:
    """The ds4 proxy before a stand-in ds4-server whose replies are traced: chat answers prompt 1 2 3 with
    ``reply``; the edited token (a completion of 1 token) is ``a(body)``: (tokens, finish, text, status); the rest
    (a streamed completion) is ``b(body)``: (tokens, SSE pieces). ``seen``: what reached ds4-server."""

    def __init__(self, tmp_path, reply=(11, 12, 13, 0), a=None, b=None):
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        import httpx

        from metalscope.ds4.proxy import Served, build_app
        from metalscope.live import MANAGER

        config.save_settings(mode="off", steer=True)
        save_word("edit-ds4", "soccer", [1, 2])
        self.seen, self.manager = [], MANAGER
        a = a or (lambda body: ([21], "length", " well", 200))
        b = b or (lambda body: ([22, 23, 0], sse_text([" ok</th", "ink>\n\nDo", "ne"])))
        fake = FastAPI()

        @fake.get("/v1/models")
        async def models():
            return {"data": [{"id": "x", "context_length": 4096, "top_provider": {"max_completion_tokens": 50}}]}

        @fake.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            self.seen.append(("chat", body))
            threading.Thread(target=write_request, args=(body["scope"]["trace"], [1, 2, 3], list(reply))).start()

            async def sse():
                yield b'data: {"choices":[{"delta":{"reasoning_content":"Hmm so"}}]}\n\n'
                yield b"data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        @fake.post("/v1/completions")
        async def complete(request: Request):
            body = await request.json()
            self.seen.append(("complete", body))
            if not body.get("stream"):                       # the edited token
                toks, finish, text, status = a(body)
                threading.Thread(target=write_request, args=(body["scope"]["trace"], body["prompt"], toks)).start()
                if status >= 400:
                    return JSONResponse({"error": {"message": "boom"}}, status_code=status)
                return JSONResponse({"choices": [{"text": text, "finish_reason": finish}]})
            toks, pieces = b(body)
            threading.Thread(target=write_request, args=(body["scope"]["trace"], body["prompt"], toks)).start()

            async def sse():
                for p in pieces:
                    yield p
            return StreamingResponse(sse(), media_type="text/event-stream")

        self.tracer = Ds4Tracer("edit-ds4", EditTok(), "qwen4", L, HC, E, K_EXP, N_EXP)
        MANAGER.tracers[id(self.tracer)] = self.tracer
        self.loop = asyncio.new_event_loop()
        self.q = MANAGER.hub.subscribe(self.loop, "pi-e")
        self.app = build_app(Served("edit-ds4", tmp_path / "x.gguf", "http://ds4", self.tracer),
                             transport=httpx.ASGITransport(app=fake))
        self.head = {"X-Scope-Trace": "on", "X-Scope-Trace-Client": "pi-e"}
        self.msg = dict(MSG, model="edit-ds4", stream=True, max_tokens=100, temperature=0.6,
                        stream_options={"include_usage": True})

    def __enter__(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(self.app, base_url=PROXY).__enter__()
        return self

    def __exit__(self, *exc):
        try:
            self.client.__exit__(*exc)
        finally:
            self.manager.hub.unsubscribe(self.q)
            self.manager.tracers.pop(id(self.tracer), None)
            self.loop.close()

    def chat(self, edit=None, steer=None, **msg):
        head = dict(self.head, **({"X-Scope-Edit": edit} if edit else {}), **({"X-Scope-Steer": steer} if steer else {}))
        return self.client.post("/v1/chat/completions", json=dict(self.msg, **msg), headers=head)

    def rid(self, n=0):
        """The request id of the n-th request that reached ds4-server."""
        return Path(self.seen[n][1]["scope"]["trace"]).stem

    def starts(self):
        self.loop.run_until_complete(asyncio.sleep(0.05))
        out = []
        while not self.q.empty():
            ev = self.q.get_nowait()
            if ev["type"] == "start":
                out.append((ev["rid"], ev.get("edit")))
        return out


def stream_of(r):
    """A chat stream's reasoning, content, finish reason and usage, and its error events."""
    import json
    deltas, finish, usage, errors = [], None, None, []
    for line in r.text.split("\n"):
        if line.startswith("data: {"):
            ev = json.loads(line[6:])
            usage = ev.get("usage") or usage
            if ev.get("error"):
                errors.append(ev["error"])
            for c in ev.get("choices") or []:
                deltas += [(k, v) for k, v in c["delta"].items() if k != "role"]
                finish = c["finish_reason"] or finish
    return ("".join(v for k, v in deltas if k == "reasoning_content"), "".join(v for k, v in deltas if k == "content"),
            finish, usage, errors)


def test_an_edit_keeps_the_reply_before_the_token_edits_that_token_alone_and_continues(tmp_path):
    """A traced reply (prompt 1 2 3, reply 11 12 13 0), then an edit of its token 1 (12): the edited token is one
    greedy completion of the prompt and token 0, the removal at its last row only; the rest a completion from there
    with no steer that stops before a tool call. The client gets one chat stream: the kept text, the edited token,
    the rest, reasoning split from content; each completion is traced and its live start names the reply it
    continues; the ids before the edit are the recorded ones."""
    from metalscope.edit import TOOL_OPENERS

    with EditRig(tmp_path) as rig:
        saved = rig.manager.saved
        assert rig.chat().status_code == 200
        rid = rig.rid()
        r = rig.chat(f"{rid}@1 remove~soccer/1-2")
        assert r.status_code == 200, r.text
        wait_saved(rig.manager, saved + 3)
        # Edits of a reply the server does not hold, and of a token past its end, are refused.
        r2 = rig.chat("nope@1 remove~soccer/1-2")
        assert r2.status_code == 409 and "not kept here" in r2.json()["error"]["message"]
        assert rig.chat(f"{rid}@4 remove~soccer/1-2").status_code == 400
        # A steer on stays on: with the edit at the edited token (decode rows only), on its own for the rest.
        save_word("edit-ds4", "rugby", [1, 2])
        assert rig.chat(f"{rid}@0 remove~soccer/1-2", steer="remove~rugby/1-2").status_code == 200
        (_, a), (_, b) = rig.seen[-2:]
        assert a["prompt"] == [1, 2, 3] and a["scope"]["steer_decode_only"] is True
        assert b["prompt"] == [1, 2, 3, 21] and b["scope"]["steer_decode_only"] is False
        (_, one), (_, rest) = rig.seen[1], rig.seen[2]
        assert one["prompt"] == [1, 2, 3, 11] and one["max_tokens"] == 1 and one["temperature"] == 0
        assert one["scope"]["steer_decode_only"] is True and one["scope"]["trace"]
        assert rest["prompt"] == [1, 2, 3, 11, 21] and rest["stop"] == TOOL_OPENERS and "steer" not in rest["scope"]
        assert rest["max_tokens"] == 98 and rest["temperature"] == 0.6 and rest["stream"] is True
        reasoning, content, finish, usage, _ = stream_of(r)
        assert (reasoning, content, finish) == ("Hmm well ok", "Done", "stop") and r.text.endswith("data: [DONE]\n\n")
        assert usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
        starts = rig.starts()
        (r0, e0), (ra, ea), (rb, eb) = starts[:3]
        assert r0 == rid and e0 is None and len(starts) == 5
        assert ea == dict(of=rid, at=1, steer="remove~soccer/1-2") and eb == dict(of=ra, at=2, steer="remove~soccer/1-2")
        traces = {e["id"][-8:]: Trace.load(e["path"]) for e in read_index()}
        assert traces[ra[:8]].tokens == [1, 2, 3, 11] and traces[ra[:8]].meta["edit"]["of"] == rid
        assert traces[rb[:8]].tokens == [1, 2, 3, 11, 21, 22, 23] and traces[rb[:8]].meta["edit"]["at"] == 2


def test_an_edit_on_ds4_takes_a_word_named_by_its_token_id(tmp_path):
    """A lens word that is no name (初步) goes by its id (tok104495, the dock's tokenName): the header stays ASCII and
    the edited token's steering file is written from it as from any word's."""
    with EditRig(tmp_path) as rig:
        save_word("edit-ds4", "tok104495", [1, 2], seed=3)
        assert rig.chat().status_code == 200
        rid = rig.rid()
        r = rig.chat(f"{rid}@1 remove~tok104495/1-2")
        assert r.status_code == 200, r.text
        (_, one), (_, rest) = rig.seen[1], rig.seen[2]
        assert one["scope"]["steer_decode_only"] is True and one["scope"].get("steer") and "steer" not in rest["scope"]
        assert [e for _, e in rig.starts()][1] == dict(of=rid, at=1, steer="remove~tok104495/1-2")


def test_an_edited_reply_is_edited_at_the_token_the_dock_shows(tmp_path):
    """Reply 11 12 13 0 edited at 1 reads 11 21 22 23 0 in the dock: its token 3 (23) is the rest's, its token 1
    (21) the edited one's. An edit of either names the token as shown; a reply the edited token ended (a stop token
    there) is edited the same way."""
    stop = {"on": False}
    a = lambda body: ([0], "stop", "", 200) if stop["on"] else ([21], "length", " well", 200)
    with EditRig(tmp_path, a=a) as rig:
        saved = rig.manager.saved
        rig.chat()
        assert rig.chat(f"{rig.rid()}@1 remove~soccer/1-2").status_code == 200
        wait_saved(rig.manager, saved + 3)
        rid_b = rig.rid(2)
        r = rig.chat(f"{rid_b}@3 remove~soccer/1-2")
        assert r.status_code == 200, r.text
        assert rig.seen[3][1]["prompt"] == [1, 2, 3, 11, 21, 22]
        wait_saved(rig.manager, saved + 5)
        assert rig.chat(f"{rid_b}@1 remove~soccer/1-2").status_code == 200
        assert rig.seen[5][1]["prompt"] == [1, 2, 3, 11]
        wait_saved(rig.manager, saved + 7)
        # The edited token a stop token: the reply ends there (11 0), no rest; that reply is edited as shown too.
        stop["on"] = True
        n = len(rig.seen)
        r = rig.chat(f"{rig.rid()}@1 remove~soccer/1-2")
        assert r.status_code == 200 and len(rig.seen) == n + 1
        assert stream_of(r)[:3] == ("Hmm", "", "stop")
        wait_saved(rig.manager, saved + 8)
        rid_a = rig.rid(n)
        assert rig.chat(f"{rid_a}@1 remove~soccer/1-2").status_code == 200
        assert rig.chat(f"{rid_a}@2 remove~soccer/1-2").status_code == 400
        assert rig.seen[-1][1]["prompt"] == [1, 2, 3, 11]


def test_an_edited_token_that_ended_the_reply_with_none_recorded_fails_where_the_stop_token_is_kept():
    """An edited token said to end the reply ("stop") with no token recorded: on ds4, which keeps a reply's stop
    token among its tokens, a failure (502); on oMLX, which leaves it out, the reply ends at the edit."""
    from metalscope import edit, steer
    from metalscope.live import MANAGER

    class Answered:
        status_code, headers, rid = 200, {}, None

        async def aread(self):
            return b'{"choices":[{"text":"","finish_reason":"stop"}]}'

        async def aclose(self):
            pass

    class Engine:
        name, model, tokenizer = "engine", "x", EditTok()

        def __init__(self, stop_in_reply):
            self.stop_in_reply = stop_in_reply

        async def complete(self, body, level, spec, meta, base):
            MANAGER.kept.put("edited", base[0], [], base, "x")           # nothing of its own recorded
            return Answered(), "edited", None, None

        async def budget(self):
            return None

        async def kept_bytes(self, kept, rest):
            return self.tokenizer.decode_bytes(kept)

    MANAGER.kept.put("reply", [1, 2, 3], [11, 12, 13], model="x")
    e = dict(rid="reply", at=1, steer=steer.SteerSpec(()))
    for stop_in_reply, status in ((True, 502), (False, 200)):
        r = asyncio.run(edit.answer(e, "/v1/chat/completions", dict(model="x", stream=True), {}, "on", None,
                                    Engine(stop_in_reply)))
        assert r.status_code == status


def test_the_edited_tokens_steer_is_marked_an_edits_own_with_or_without_a_steer_on():
    """SteerSpec.edit (the only steer a word vector takes): set on the edited token's steer, also when a steer is on
    (its parts ride along at that token, still the edit's own one-token steer)."""
    from metalscope import edit, steer
    from metalscope.live import MANAGER

    class Answered:
        status_code, headers, rid = 200, {}, None

        async def aread(self):
            return b'{"choices":[{"text":"","finish_reason":"stop"}]}'

        async def aclose(self):
            pass

    class Engine:
        name, model, tokenizer, stop_in_reply = "engine", "x", EditTok(), False

        def __init__(self):
            self.specs = []

        async def complete(self, body, level, spec, meta, base):
            self.specs.append(spec)
            MANAGER.kept.put("edited", base[0], [], base, "x")           # the edited token ended the reply
            return Answered(), "edited", None, None

        async def budget(self):
            return None

        async def kept_bytes(self, kept, held):
            return self.tokenizer.decode_bytes(kept)

    MANAGER.kept.put("reply", [1, 2, 3], [11, 12, 13], model="x")
    e = dict(rid="reply", at=1, steer=steer.parse_header("remove~soccer/1-2"))
    for on, marked, label in ((None, True, "remove~soccer/1-2"), (steer.parse_header("calm@0.3"), True,
                                                                  "calm@0.3+remove~soccer/1-2")):
        eng = Engine()
        r = asyncio.run(edit.answer(e, "/v1/chat/completions", dict(model="x", stream=True), {}, "on", on, eng))
        assert r.status_code == 200 and len(eng.specs) == 1
        assert (eng.specs[0].edit, eng.specs[0].positions, eng.specs[0].label) == (marked, "decode", label)


def test_chat_text_opens_the_thinking_where_the_text_begins_with_think():
    """Inkling 2's text on oMLX (its parser's): the reply itself opens the thinking with <think>, the prompt does not.
    Whole, or split anywhere in the marker or a piece at a time: the thinking as reasoning, the answer as content, the
    marker dropped. A text that only begins like it, or has no thinking, is content as it was."""
    from metalscope.edit import ChatText

    def read(pieces, thinking=False):
        t, out = ChatText(thinking), []
        for p in pieces:
            out += t.feed(p)
        out += t.feed("", final=True)
        merged = []
        for k, v in out:
            if merged and merged[-1][0] == k:
                merged[-1] = (k, merged[-1][1] + v)
            else:
                merged.append((k, v))
        return merged

    whole = "<think>Hmm so</think>Done"
    want = [("reasoning_content", "Hmm so"), ("content", "Done")]
    assert read([whole]) == read(list(whole)) == read([whole], thinking=True) == want
    for i in range(len(whole) + 1):
        assert read([whole[:i], whole[i:]]) == want, i
    assert read(["", "<think>", "", "Hmm so</think>\n\nDone"]) == want
    assert read(["<", "b>Hi</b>"]) == [("content", "<b>Hi</b>")] and read(["Done"]) == [("content", "Done")]
    assert read(["<thi"]) == [("content", "<thi")]                     # the end: what was held is content
    assert read(["<think>cut </thi"]) == [("reasoning_content", "cut </thi")]


def test_an_edit_after_a_tool_call_is_refused_and_a_failed_edited_token_passes_its_error_on(tmp_path):
    """A token after a tool call's opener is not edited (its markup is ds4-server's to read); ds4-server failing the
    edited token: its error goes back as it came, nothing sent after it and nothing left active."""
    with EditRig(tmp_path, reply=(11, 40, 12, 0), a=lambda body: ([21], "length", "", 500)) as rig:
        rig.chat()
        rid = rig.rid()
        r = rig.chat(f"{rid}@2 remove~soccer/1-2")
        assert r.status_code == 400 and "after a tool call" in r.json()["error"]["message"]
        r = rig.chat(f"{rid}@1 remove~soccer/1-2")
        assert r.status_code == 500 and r.json()["error"]["message"] == "boom"
        assert len(rig.seen) == 2
        end = time.time() + 5
        while rig.manager.requests and time.time() < end:
            time.sleep(0.02)
        assert not rig.manager.requests


def test_an_edit_joins_a_character_split_between_the_kept_tokens_and_the_rest(tmp_path):
    """The edited token is the first byte of "é", the rest's first piece its second byte (ds4-server streams it
    as it is, not UTF-8): the client reads "é", and no event is lost."""
    b = lambda body: ([32, 0], sse_text([b'"\xa9 ok"', "</think>Done"]))
    with EditRig(tmp_path, a=lambda body: ([31], "length", "", 200), b=b) as rig:
        rig.chat()
        r = rig.chat(f"{rig.rid()}@1 remove~soccer/1-2")
        reasoning, content, finish, _, _ = stream_of(r)
        assert (reasoning, content, finish) == ("Hmmé ok", "Done", "stop")


def test_an_error_mid_stream_ends_the_edited_reply_and_saves_it_as_not_served(tmp_path, monkeypatch):
    """ds4-server's error event in the rest: the text held back is sent, then the error, a finish and [DONE]; the
    rest's trace is saved as not served."""
    from metalscope.ds4 import proxy

    served = []
    finish = proxy.Run.finish
    monkeypatch.setattr(proxy.Run, "finish", lambda self, ok: (served.append(ok), finish(self, ok))[1])
    b = lambda body: ([22, 0], [b'data: {"choices":[{"text":" ok</th","finish_reason":null}]}\n\n',
                                b'data: {"error":{"message":"ds4 fell over"}}\n\n'])
    with EditRig(tmp_path, b=b) as rig:
        saved = rig.manager.saved
        rig.chat()
        r = rig.chat(f"{rig.rid()}@1 remove~soccer/1-2")
        reasoning, content, finish_reason, _, errors = stream_of(r)
        assert reasoning == "Hmm well ok</th" and errors == [{"message": "ds4 fell over"}]
        assert finish_reason == "error" and r.text.endswith("data: [DONE]\n\n")
        wait_saved(rig.manager, saved + 3)
        assert served == [True, True, False]


def test_an_edit_without_a_budget_takes_ds4_servers_less_the_kept_tokens(tmp_path):
    with EditRig(tmp_path) as rig:
        rig.chat()
        msg = {k: v for k, v in rig.msg.items() if k != "max_tokens"}
        rig.msg = msg
        assert rig.chat(f"{rig.rid()}@1 remove~soccer/1-2").status_code == 200
        assert rig.seen[-1][1]["max_tokens"] == 48

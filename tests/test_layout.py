"""The expert map's layout (layout.py): built from the newest saved traces, kept per checkpoint, served one layer at a
time as the dock's EXPERTS grid (read only; metalscope experts layout builds it); a saved reply's routing."""

import base64
import json

import numpy as np
from pathlib import Path

from metalscope import layout as LY
from tests.helpers import E, K, make_trace, random_ids


def test_layout_is_built_and_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    rng = np.random.default_rng(0)
    words = [" the", " ice", ",", " 3", " water", " of"]
    for i in range(4):
        texts = [words[j] for j in rng.integers(0, len(words), 40)]
        tr = make_trace(texts, random_ids(rng, 3, 40), prompt_len=10, title=f"t{i}")
        tr.meta["model"] = "Toy-MoE"
        tr.save(tmp_path / "traces" / "d" / f"tr{i}")
    lay = LY.build_layout("Toy-MoE")
    assert LY.layout_path("Toy-MoE").exists() and LY._cached_layout("Toy-MoE") == json.loads(json.dumps(lay))
    assert [l["layer"] for l in lay["layers"]] == [0, 1, 2] and lay["num_experts"] == E
    xy = [p for l in lay["layers"] for p in l["xy"] if p]
    assert xy and max(abs(v) for p in xy for v in p) <= 1.0001


def test_a_layout_pools_only_the_newest_traces_within_its_token_budget(tmp_path, monkeypatch):
    """Flash Next has thousands of saved traces; pooling them all took over ten minutes and 40 GB."""
    import os
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    rng = np.random.default_rng(0)
    for i in range(5):
        tr = make_trace([f" w{i}", " ice"] * 20, random_ids(rng, 3, 40), prompt_len=10, title=f"t{i}")   # distinct: pooling dedupes
        tr.meta["model"] = "Toy-MoE"
        d = tr.save(tmp_path / "traces" / "d" / f"tr{i}")
        os.utime(d / "meta.json", (1e9 + i, 1e9 + i))          # tr4 the newest
    paths = sorted((tmp_path / "traces" / "d").iterdir())
    assert [p.name for p in LY.newest_within(paths, 100)] == ["tr4", "tr3"]       # 40 tokens each
    assert [p.name for p in LY.newest_within(paths, 10)] == ["tr4"]               # at least one
    pooled = lambda budget: LY.build_layout("Toy-MoE", budget=budget)["tokens"]
    assert pooled(100) == 2 * pooled(10)


def test_a_layout_built_with_token_categories_still_reads(tmp_path, monkeypatch):
    """Layouts built before the token categories went hold cat, uses and pref per layer and the category labels:
    the grid reads its xy and nothing else."""
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    xy = [[0.9, 0.0], [-0.9, 0.5], None, [0.1, 0.2]]
    old = dict(version=LY.LAYOUT_VERSION, model="Toy-MoE", num_experts=4, k=2, tokens=40, traces=1,
               categories=["prose word", "function word"], layers=[dict(layer=0, xy=xy, cat=[0, -1, -1, 1], uses=[3, 0, 0, 5],
                                                                     pref=[[0.5, 0.2], None, None, [0.4, 0.1]])])
    LY.layout_path("Toy-MoE").parent.mkdir(parents=True)
    LY.layout_path("Toy-MoE").write_text(json.dumps(old))
    g = LY.expert_grid("Toy-MoE", 0, cols=2)
    assert (g["num_experts"], g["n_layers"], g["pos"]) == (4, 1, LY.grid_positions(xy, 2)[1])


def test_a_trace_saved_with_attention_sources_still_reads(tmp_path, monkeypatch):
    """Traces saved before attention sources went hold attn_src_* records and attn_block_* meta: kept, never read."""
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    rng = np.random.default_rng(0)
    tr = make_trace([" the", " ice", ",", " 3"] * 10, random_ids(rng, 3, 40), prompt_len=10)
    tr.meta.update(model="Toy-MoE", attn_block_size=4, attn_block_topk=2)
    tr.arrays.update({"attn_src_ids/-1": np.full((40, 16), -1, np.int32), "attn_src_w/-1": np.zeros((40, 16), np.float32),
                      "attn_src_votes/-1": np.zeros((40, 16), np.int32)})
    path = tr.save(tmp_path / "traces" / "d" / "old")
    assert LY.build_layout("Toy-MoE")["traces"] == 1
    assert LY.reply_routing(path)["n"] == 30


def test_grid_positions_keep_map_neighbours_together():
    """Columns of ``rows`` experts by x, left to right, each column by y, top to bottom; experts without xy last."""
    xy = [[0.9, 0.0], [-0.9, 0.5], [-0.8, -0.5], [0.1, 0.2], None, [0.0, -0.9]]
    rows, pos = LY.grid_positions(xy, cols=3)
    assert rows == 2 and sorted(pos) == list(range(6))
    # by x: 1 (-0.9), 2 (-0.8) | 5 (0.0), 3 (0.1) | 0 (0.9), 4 (none); each column by y
    assert pos == [2, 3, 0, 4, 5, 1]
    rows, pos = LY.grid_positions([[float(i), 0.0] for i in range(512)])     # 512 experts: 32 columns of 16
    assert rows == 16 and sorted(pos) == list(range(512)) and pos[0] == 0 and pos[16] == 1 and pos[511] == 15 * 32 + 31
    rows, pos = LY.grid_positions([None] * 8)                                   # no map at all: number order, one row
    assert rows == 1 and pos == list(range(8))


def _routed_home(tmp_path, monkeypatch):
    """One saved MoE reply of Toy-MoE and its index entry."""
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    rng = np.random.default_rng(0)
    tr = make_trace([" the", " ice", ",", " 3"] * 10, random_ids(rng, 3, 40), prompt_len=10)
    tr.meta["model"] = "Toy-MoE"
    path = tmp_path / "traces" / "d" / "tr0"
    tr.save(path)
    (tmp_path / "index.jsonl").write_text(json.dumps(dict(id="tr0", path=str(path), model="Toy-MoE", time=1, experts=True)) + "\n")


NO_MAP = "no expert map for Toy-MoE yet: metalscope experts layout builds one"


def test_expert_grid_route_reads_the_map_and_never_builds_one(tmp_path, monkeypatch):
    """GET /scope/api/experts/grid without a map: pos null and the size from a saved reply, nothing built; with one,
    the map's grid."""
    from tests.helpers import scope_client
    _routed_home(tmp_path, monkeypatch)
    c = scope_client()
    build = LY.build_layout
    monkeypatch.setattr(LY, "build_layout", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built a layout")))
    none = c.get("/scope/api/experts/grid", params=dict(model="Toy-MoE", layer=1)).json()
    assert none["pos"] is None and none["num_experts"] == E and none["cols"] == 32 and "top" not in none
    assert none["rows"] == -(-E // 32) and none["note"] == NO_MAP and none["n_layers"] is None
    lay = build("Toy-MoE")
    got = c.get("/scope/api/experts/grid", params=dict(model="Toy-MoE", layer=1)).json()
    rows, pos = LY.grid_positions(lay["layers"][1]["xy"])
    assert got["pos"] == pos and got["rows"] == rows and got["n_layers"] == 3 and got["num_experts"] == E
    assert "top" not in got and "top" not in lay["layers"][1] and "note" not in got
    bad = c.get("/scope/api/experts/grid", params=dict(model="Toy-MoE", layer=3))
    assert bad.status_code == 400 and bad.json()["detail"] == "layer must be 0 to 2"
    assert c.get("/scope/api/experts/grid").status_code == 404                 # no model loaded or named
    neg = c.get("/scope/api/experts/grid", params=dict(model="Toy-MoE", layer=-1))
    assert neg.status_code == 400 and neg.json()["detail"] == "layer must be 0 or more"
    LY.layout_path("Toy-MoE").write_text("{not json")                           # a corrupt map: number order
    broken = c.get("/scope/api/experts/grid", params=dict(model="Toy-MoE", layer=1)).json()
    assert broken["pos"] is None and broken["num_experts"] == E and broken["note"] == NO_MAP
    assert c.get("/scope/api/experts/grid", params=dict(model="Nope", layer=-2)).status_code == 400   # bounds without a map too
    dense = LY.expert_grid("Dense", 0)
    assert dense["pos"] is None and dense["num_experts"] is None


def test_expert_grid_route_says_a_loaded_dense_model_has_no_experts(tmp_path, monkeypatch):
    """A loaded model whose hooks found no MoE layers (capture.routing None): the note says it has none; one with a
    routing rule keeps the no-map note."""
    from types import SimpleNamespace
    from metalscope.live import MANAGER
    from tests.helpers import scope_client
    _routed_home(tmp_path, monkeypatch)
    t = SimpleNamespace(name="Dense", installed=True, capture=SimpleNamespace(routing=None))
    monkeypatch.setitem(MANAGER.tracers, id(t), t)
    c = scope_client()
    dense = c.get("/scope/api/experts/grid", params=dict(model="Dense")).json()
    assert dense["note"] == "Dense has no experts" and dense["num_experts"] is None and dense["pos"] is None
    t.capture.routing = object()
    assert c.get("/scope/api/experts/grid", params=dict(model="Dense")).json()["note"] == "no expert map for Dense yet: metalscope experts layout builds one"


def test_experts_layout_command_builds_where_there_is_none(tmp_path, monkeypatch, capsys):
    from metalscope import cli
    _routed_home(tmp_path, monkeypatch)
    cli.main(["experts", "layout"])
    out = capsys.readouterr().out
    assert out == f"Toy-MoE: 40 tokens from 1 traces, {LY.layout_path('Toy-MoE')}\n" and LY.usable_layout("Toy-MoE")
    cli.main(["experts", "layout", "Toy-MoE"])
    assert capsys.readouterr().out == f"Toy-MoE: built already, {LY.layout_path('Toy-MoE')} (--rebuild builds it again)\n"
    LY.layout_path("Toy-MoE").write_text("{not json")
    cli.main(["experts", "layout", "Toy-MoE", "--rebuild"])
    assert capsys.readouterr().out.startswith("Toy-MoE: 40 tokens") and LY.usable_layout("Toy-MoE")
    cli.main(["experts", "layout", "Dense"])
    assert capsys.readouterr().out == "Dense: no expert map (no traces with routing records)\n"
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "empty"))
    cli.main(["experts", "layout"])
    assert capsys.readouterr().out == "no routed reply saved yet: trace a chat on an MoE model first\n"


def test_expert_grid_reads_the_index_once_per_change(tmp_path, monkeypatch):
    from metalscope import layout
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    (tmp_path / "index.jsonl").write_text("")
    reads = []
    monkeypatch.setattr(layout, "read_index", lambda: reads.append(1) or [])
    for _ in range(3):
        layout.expert_grid("Toy-MoE", 0)
    assert len(reads) == 1
    (tmp_path / "index.jsonl").write_text("\n")                                  # a new reply saved: read again
    layout.expert_grid("Toy-MoE", 0)
    assert len(reads) == 2


def test_expert_grid_names_each_models_checkpoint_once(tmp_path, monkeypatch):
    """Looking for the expert count, each served name in the index is resolved to its checkpoint once."""
    from metalscope import layout
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    entries = [dict(id=f"t{i}", path=str(tmp_path / f"t{i}"), model=["A-4bit", "B-8bit"][i % 2], experts=True) for i in range(10)]
    for e in entries:
        Path(e["path"]).mkdir()                                                    # listed only while saved
    (tmp_path / "index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    calls = []
    monkeypatch.setattr(layout.models, "checkpoint_of", lambda name: calls.append(name) or name.split("-")[0])
    assert layout._num_experts("C-4bit") is None                                 # no reply of C: every entry looked at
    assert sorted(calls) == ["A-4bit", "B-8bit", "C-4bit"]


def test_reply_routing_route_reads_the_router_arrays_alone(tmp_path, monkeypatch):
    """GET /scope/api/traces/ID/routing: reply token i's routing at trace position prompt_len + i, as the live stream
    sends it (ids, each expert's share of its layer's mix 0-255), the last N tokens with ``last``; a dense model's is
    empty."""
    from tests.helpers import scope_client
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path))
    rng = np.random.default_rng(3)
    ids = random_ids(rng, 3, 40)
    probs = rng.random((3, 40, K)).astype(np.float32)
    tr = make_trace([f" w{i}" for i in range(40)], ids, prompt_len=10, probs=probs)
    tr.meta["model"] = "Toy-MoE"
    path = tmp_path / "traces" / "d" / "tr0"
    tr.save(path)
    dense = make_trace([" a", " b", " c", " d"], ids[:, :4], prompt_len=2)
    dense.arrays = {}
    dense.save(tmp_path / "traces" / "d" / "tr1")
    (tmp_path / "index.jsonl").write_text(json.dumps(dict(id="tr0", path=str(path), model="Toy-MoE", time=1, experts=True)) + "\n" +
                                          json.dumps(dict(id="tr1", path=str(tmp_path / "traces" / "d" / "tr1"), model="Dense", time=2)) + "\n")
    c = scope_client()
    got = c.get("/scope/api/traces/tr0/routing").json()
    assert (got["start"], got["n"], got["L"], got["k"]) == (0, 30, 3, K)
    sel = np.frombuffer(base64.b64decode(got["ids"]), "<u2").reshape(30, 3, K)
    w = np.frombuffer(base64.b64decode(got["w"]), np.uint8).reshape(30, 3, K)
    assert (sel[0, 1] == ids[1, 10]).all() and (sel[29, 2] == ids[2, 39]).all()       # reply token 0 is position 10
    share = probs[1, 10] / probs[1, 10].sum()
    assert np.abs(w[0, 1].astype(int) - np.rint(255 * share)).max() <= 1
    tail = c.get("/scope/api/traces/tr0/routing", params=dict(last=5)).json()
    assert (tail["start"], tail["n"]) == (25, 5)
    assert np.frombuffer(base64.b64decode(tail["ids"]), "<u2").reshape(5, 3, K)[0, 0].tolist() == ids[0, 35].tolist()
    assert c.get("/scope/api/traces/tr1/routing").json() == dict(start=0, n=0, L=0, k=0, ids="", w="")
    assert c.get("/scope/api/traces/nope/routing").status_code == 404
    assert c.get("/scope/api/traces/tr0/routing", params=dict(last=0)).status_code == 400
    (path / "arrays.npz").write_bytes(b"not a zip")                                   # corrupt: 404, not 500
    bad = c.get("/scope/api/traces/tr0/routing")
    assert bad.status_code == 404 and "no routing saved" in bad.json()["detail"]
    (path / "arrays.npz").write_bytes(b"PK\x03\x04 truncated")
    assert c.get("/scope/api/traces/tr0/routing").status_code == 404


def test_reply_routing_right_aligned_records_and_last(tmp_path):
    """layout.reply_routing, which takes each layer's reply rows before converting them: records right-aligned to
    the tokens (fewer records than tokens) still line up with the reply's tokens, and ``last`` keeps the newest."""
    rng = np.random.default_rng(4)
    ids = random_ids(rng, 3, 40)
    tr = make_trace([f" w{i}" for i in range(40)], ids, prompt_len=30)
    for l in range(3):                                                             # the first 10 positions unrecorded
        tr.arrays[f"router_ids/{l}"] = tr.arrays[f"router_ids/{l}"][10:]
        tr.arrays[f"router_probs/{l}"] = tr.arrays[f"router_probs/{l}"][10:]
    tr.save(tmp_path / "t")
    got = LY.reply_routing(tmp_path / "t", last=4)
    assert (got["start"], got["n"], got["L"]) == (6, 4, 3)
    sel = np.frombuffer(base64.b64decode(got["ids"]), "<u2").reshape(4, 3, K)
    assert (sel[0, 2] == ids[2, 36]).all() and (sel[3, 0] == ids[0, 39]).all()      # reply token 6 is position 36


def test_router_probs_saved_as_float16_give_what_float32_gave(tmp_path):
    """Traces save router_probs as float16 (older ones float32). The dock's routing shares (0-255) move by a unit at
    most. The tally's top 3 by weight is where half precision could change a pick: only on a near-tie (third and
    fourth within 1e-3), so without one the tallies agree."""
    from metalscope import tally
    rng = np.random.default_rng(7)
    ids = random_ids(rng, 3, 40)
    probs = rng.dirichlet(np.ones(K), (3, 40)).astype(np.float32)
    probs[0, 0] = [0.4, 0.3, 0.15001, 0.15003]                    # a near-tie half precision cannot tell apart: a flip
    top3 = lambda w: np.sort(np.argsort(-w, -1, kind="stable")[..., :3], -1)
    third_fourth = -np.diff(np.sort(probs, -1)[..., ::-1][..., 2:4], axis=-1)[..., 0]
    flips = (top3(probs) != top3(probs.astype(np.float16).astype(np.float32))).any(-1)
    assert flips[0, 0] and (third_fourth[flips] < 1e-3).all()
    steps = np.arange(K, 0, -1, dtype=np.float32) / (K * (K + 1) / 2)                    # 0.1 apart at K = 4
    clear = rng.permuted(np.tile(steps, (3, 40, 1)), axis=-1) + rng.uniform(0, 0.004, (3, 40, K)).astype(np.float32)
    accs = []
    for name, w in (("f32", clear), ("f16", clear.astype(np.float16))):
        make_trace([f" w{i}" for i in range(40)], ids, prompt_len=10, probs=w).save(tmp_path / name)
        accs.append(tally.Acc())
        assert accs[-1].add_trace(tmp_path / name, name)
    assert np.load(tmp_path / "f16" / "arrays.npz")["router_probs/0"].dtype == np.float16
    for li in range(3):
        (u32, c32), (u16, c16) = accs[0].layer(li), accs[1].layer(li)
        assert np.array_equal(u32, u16) and np.array_equal(c32, c16)
    a, b = LY.reply_routing(tmp_path / "f32"), LY.reply_routing(tmp_path / "f16")
    assert a["ids"] == b["ids"] and a["n"] == b["n"] == 30
    wa, wb = (np.frombuffer(base64.b64decode(x["w"]), np.uint8).astype(int) for x in (a, b))
    assert np.abs(wa - wb).max() <= 1


def test_a_trace_saved_now_reads_back_as_one_saved_with_numpy_s_default_and_a_failed_save_leaves_no_file(tmp_path):
    """Traces are written at zlib level 1 through a temp file (store.write_npz). Every reader gets what it got from
    one saved by np.savez_compressed, as traces saved before were, and those still load. A save that raises part way
    leaves no arrays.npz behind."""
    from metalscope import experts, tally
    from metalscope.store import Trace
    rng = np.random.default_rng(5)
    probs = rng.random((3, 40, K)).astype(np.float16)
    tr = make_trace([f" w{i}" for i in range(40)], random_ids(rng, 3, 40), prompt_len=10, probs=probs)
    tr.arrays["resid_norm/0"] = np.where(rng.random(40) < 0.2, np.nan, rng.random(40)).astype(np.float32)
    new = tr.save(tmp_path / "new")
    old = tmp_path / "old"
    old.mkdir()
    (old / "meta.json").write_text((new / "meta.json").read_text())
    np.savez_compressed(old / "arrays.npz", **tr.arrays)
    a, b = Trace.load(new), Trace.load(old)
    assert a.arrays.keys() == b.arrays.keys() == tr.arrays.keys() and (a.tokens, a.meta) == (b.tokens, b.meta)
    for k, v in a.arrays.items():
        assert v.dtype == b.arrays[k].dtype and np.array_equal(v, b.arrays[k], equal_nan=True)
    assert LY.reply_routing(new, last=7) == LY.reply_routing(old, last=7)
    accs = [tally.Acc() for _ in range(2)]
    assert accs[0].add_trace(new, "new") and accs[1].add_trace(old, "old")
    for li in range(3):
        assert all(np.array_equal(x, y) for x, y in zip(accs[0].layer(li), accs[1].layer(li)))
    assert experts._npz_has_router(new / "arrays.npz")

    class Bad:
        def __array__(self, *a, **k):
            raise OSError("disk full")
    tr.arrays["zz/0"] = Bad()
    try:
        tr.save(tmp_path / "cut")
    except OSError:
        pass
    assert not (tmp_path / "cut" / "arrays.npz").exists()

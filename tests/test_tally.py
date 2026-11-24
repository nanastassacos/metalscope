"""The expert tally (tally.py): counting saved chats under the top-3 rule, prompt-prefix dedup, exclusions,
idempotence, two writers, the scores, the routes and the command."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from metalscope import cli, models, tally
from metalscope.store import Trace, append_index

E, L, K = 16, 2, 6     # experts a layer, MoE layers, mixed a token


def routing(tokens, rng, special=None):
    """Router records for ``tokens``: K random experts a position, the first three the strongest; ``special``
    {token: expert}: that expert the strongest wherever the token is."""
    T = len(tokens)
    ids = np.stack([np.stack([rng.choice(E, K, replace=False) for _ in range(T)]) for _ in range(L)])
    probs = np.tile(np.array([0.4, 0.3, 0.2, 0.05, 0.03, 0.02], np.float32), (L, T, 1))
    for (l, p), t in np.ndenumerate(np.array([tokens] * L)):
        x = (special or {}).get(int(t))
        if x is not None and x not in ids[l, p, :1]:
            row = [x] + [i for i in ids[l, p] if i != x][:K - 1]
            ids[l, p] = row
    return ids, probs


def save(home, tid, tokens, prompt_len, rng, client="pi-abc", model="Toy-MoE", steer=None, special=None, read_from=0,
         experts=True):
    ids, probs = routing(tokens, rng, special)
    arrays = {f"router_ids/{l}": ids[l].astype(np.int32) for l in range(L)} if experts else {}
    arrays.update({f"router_probs/{l}": probs[l] for l in range(L)} if experts else {})
    tr = Trace(arrays, list(tokens), [f" t{t}" for t in tokens], prompt_len, dict(num_experts=E, model=model, read_from=read_from))
    path = tr.save(Path(home) / "traces" / "d" / tid)
    entry = dict(id=tid, path=str(path), model=model, time=len(list((Path(home) / "traces" / "d").iterdir())),
                 client=client, steer=steer, experts=experts, prompt_tokens=prompt_len)
    append_index(entry)
    return entry, ids


def top3(ids):
    return ids[..., :3]                                          # routing() puts the strongest first


def test_counts_each_token_once_under_the_top_3_rule_and_the_prompt_prefix_once(home):
    rng = np.random.default_rng(0)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ea, ia = save(home, "A", a, 6, rng)
    b = a + [11, 12, 13, 14]                                     # the next turn: A's tokens again, then new ones
    eb, ib = save(home, "B", b, 12, rng)
    assert tally.add(ea) and tally.add(eb)
    st = tally.Stored(tally.tally_path("Toy-MoE"))
    assert st.N == 10 + 4 and list(st.rows) == [14, 14] and st.ids == {"A", "B"} and st.E == E
    for l in range(L):
        want = {}
        for ids, toks, start in ((ia, a, 0), (ib, b, 10)):
            for p in range(start, len(toks)):
                for x in top3(ids)[l, p]:
                    want[(int(x), toks[p])] = want.get((int(x), toks[p]), 0) + 1
        k, c = st.layer(l)
        e, t = (k >> np.uint64(20)).astype(int), (k & np.uint64(2 ** 20 - 1)).astype(int)
        assert dict(zip(zip(e.tolist(), t.tolist()), c.tolist())) == want
        qk, qc = st.cofire(l)
        assert qc.sum() == 3 * 14                                 # three pairs in each token's three
    assert not tally.add(eb), "counted once"
    # The rows the engine did not read (read_from) are not counted, though their tokens are new.
    ec, _ = save(home, "C", [50, 51, 52, 53, 54, 55], 4, rng, read_from=3)
    for l in range(L):                                           # their records empty, as a reused prefix's are
        z = dict(np.load(Path(ec["path"]) / "arrays.npz"))
        z[f"router_ids/{l}"][:3] = -1
        np.savez_compressed(Path(ec["path"]) / "arrays.npz", **z)
    tally.add(ec)
    assert tally.Stored(tally.tally_path("Toy-MoE")).N == 14 + 3


def test_left_out_automated_runs_steered_replies_and_models_without_routing(home):
    rng = np.random.default_rng(1)
    for tid, kw in (("auto", dict(client="calibrate-x")), ("steered", dict(steer="joyful 0.4")), ("dense", dict(experts=False))):
        e, _ = save(home, tid, [1, 2, 3, 4], 2, rng, **kw)
        assert not tally.add(e), tid
    assert not tally.tally_path("Toy-MoE").exists()
    e, _ = save(home, "chat", [1, 2, 3, 4], 2, rng, client=None)   # no client at all: a chat from any client
    assert tally.add(e)


def test_rebuild_equals_the_live_updates_and_clear_deletes_it(home):
    rng = np.random.default_rng(2)
    entries = [save(home, f"T{i}", list(range(i, i + 30)), 20, rng)[0] for i in range(4)]
    save(home, "run", list(range(30)), 20, rng, client="ladder-1")
    for e in entries:
        tally.add(e)
    path = tally.tally_path("Toy-MoE")
    live = {k: v for k, v in np.load(path).items()}
    assert tally.rebuild("Toy-MoE")["traces"] == 4
    again = {k: v for k, v in np.load(path).items()}
    assert live.keys() == again.keys() and all(np.array_equal(live[k], again[k]) for k in live)
    assert tally.size("Toy-MoE")["tokens"] == 4 * 30 and tally.size("Toy-MoE")["bytes"] > 0
    assert tally.clear("Toy-MoE") and not path.exists() and tally.size("Toy-MoE")["traces"] == 0


def test_the_served_copies_of_one_checkpoint_pool(home):
    models.write_record(dict(name="Toy", served=["Toy-a", "Toy-b"]))
    rng = np.random.default_rng(3)
    for tid, m in (("a", "Toy-a"), ("b", "Toy-b")):
        tally.add(save(home, tid, [7, 8, 9, 10 + len(tid)], 2, rng, model=m)[0])
    assert tally.tally_path("Toy-a") == tally.tally_path("Toy-b") == models.models_dir() / "Toy" / tally.FILE
    assert tally.size("Toy-b")["traces"] == 2


def test_two_writers_at_once_both_count(home):
    rng = np.random.default_rng(4)
    entries = [save(home, f"W{i}", [100 * i + j for j in range(40)], 20, rng)[0] for i in range(4)]
    with ProcessPoolExecutor(2) as pool:                         # two processes, as oMLX and the ds4 proxy
        assert all(pool.map(tally.add, entries))
    st = tally.Stored(tally.tally_path("Toy-MoE"))
    assert st.ids == {"W0", "W1", "W2", "W3"} and st.N == 160


def test_the_updater_counts_after_the_requests_end_and_a_failure_never_raises(home, caplog):
    rng = np.random.default_rng(5)
    busy = [True]
    u = tally.Updater(busy=lambda: busy[0])
    u.IDLE_WAIT = 0.01
    e, _ = save(home, "U", [1, 2, 3, 4, 5], 2, rng)
    u.submit(e)
    u.drain(0.2)
    assert not tally.tally_path("Toy-MoE").exists(), "not while a request runs"
    busy[0] = False
    u.drain()
    assert tally.size("Toy-MoE")["traces"] == 1
    u.submit(dict(e, id="gone", path=str(Path(home) / "nowhere")))
    u.submit(dict(e, id="gone2", path=str(Path(home) / "nowhere")))
    u.drain()
    assert "the expert tally skipped gone" in caplog.text and tally.size("Toy-MoE")["traces"] == 1


def planted(home, n_traces=12, special=None, client="pi-abc", start=0):
    rng = np.random.default_rng(6 + start)
    out = []
    for i in range(n_traces):
        toks = list(rng.integers(1, 60, 200)) + [99] * 20      # 99: a token expert 5 always takes
        rng.shuffle(toks)
        out.append(save(home, f"P{start + i}", [int(t) for t in toks], 100, rng, client=client,
                        special=special or {99: 5})[0])
    return out


def test_a_planted_specialist_ranks_first_and_an_expert_under_the_floor_is_not_ranked(home):
    for e in planted(home):
        tally.add(e)
    view = tally.layer_view("Toy-MoE", 0)
    assert view.order[0] == 5
    assert view.tops(5, 1)[0][0] == 99
    assert tally.floor_of(view.N, E) == max(3 * view.N / E / 4, 10)
    assert all(view.ne[x] >= tally.floor_of(view.N, E) for x in view.order)
    # A tiny expert, all its few tokens one rare token: fewer than the floor (10 here), not ranked.
    keys = np.array([0 << 20 | 1, 0 << 20 | 2, 1 << 20 | 1, 1 << 20 | 2, 2 << 20 | 3], np.uint64)
    counts = np.array([300, 300, 280, 320, 5])
    small = tally._layer("all", 3, 400, 1, (keys, counts), tally._EMPTY, tally._mix, tally.floor_of(400, 3))
    assert 2 not in small.order and np.isnan(small.score[2]) and small.order == [0, 1] or small.order == [1, 0]


def test_session_scores_against_each_experts_usual_mix_or_its_own_mix_without_a_tally(home):
    for e in planted(home, client="pi-old"):                    # the usual: expert 5 takes 99
        tally.add(e)
    mine = planted(home, n_traces=6, special={99: 9}, client="pi-me", start=100)   # this session: 9 takes 99
    for e in mine:
        tally.add(e)
    s = tally.layer_view("Toy-MoE", 0, "pi-me")
    assert s.against == "tally" and s.traces == 6 and s.order[0] == 9, s.order[:3]
    assert s.tops(9, 1)[0][0] == 99
    assert tally.layer_view("Toy-MoE", 0, "pi-nobody") is None
    tally.clear("Toy-MoE")
    tally._SCORED.clear()
    fallback = tally.layer_view("Toy-MoE", 0, "pi-me")
    assert fallback.against == "session" and fallback.order[0] == 9


def test_the_routes(home, monkeypatch):
    from tests.helpers import scope_client
    for e in planted(home):
        tally.add(e)
    c = scope_client()
    got = c.get("/scope/api/experts/specialists", params=dict(model="Toy-MoE", layer=0)).json()
    assert got["source"] == "all" and got["tokens"] == tally.size("Toy-MoE")["tokens"] > 2600
    order = tally.layer_view("Toy-MoE", 0).order
    assert [x["e"] for x in got["experts"]] == sorted(order) and len(order) > 3, "every expert over the floor, by number"
    assert "ranked" not in got
    assert next(x for x in got["experts"] if x["e"] == 5)["top"][0] == "<99>"          # no tokenizer loaded: ids
    one = c.get("/scope/api/experts/expert", params=dict(model="Toy-MoE", layer=0, e=5)).json()
    assert one["rank"] == 1 and one["ranked"] == len(order) and one["n"] > 0 and one["top"][0][0] == "<99>" and one["top"][0][1] > 1
    assert 0 < one["share"] <= 1 and len(one["overlaps"]) <= 8 and all(o["lift"] > 0 for o in one["overlaps"])
    sess = c.get("/scope/api/experts/specialists", params=dict(model="Toy-MoE", layer=1, client="pi-abc")).json()
    assert sess["source"] == "session" and sess["against"] == "tally"
    none = c.get("/scope/api/experts/specialists", params=dict(model="Other", layer=0)).json()
    assert none["experts"] == [] and none["tokens"] == 0
    assert c.get("/scope/api/experts/expert", params=dict(model="Other", layer=0, e=3)).json()["rank"] is None
    assert c.get("/scope/api/experts/specialists", params=dict(model="Toy-MoE", layer=2)).status_code == 400
    assert c.get("/scope/api/experts/specialists", params=dict(layer=0)).status_code == 404   # no model loaded or named


def test_the_command(home, capsys):
    planted(home, n_traces=3)
    cli.main(["experts", "tally"])
    assert "no expert tally yet" in capsys.readouterr().out
    cli.main(["experts", "tally", "--rebuild"])
    out = capsys.readouterr().out
    assert "Toy-MoE: 660 tokens from 3 traces" in out and "--clear deletes it" in out
    cli.main(["experts", "tally", "Toy-MoE"])
    assert "660 tokens from 3 traces" in capsys.readouterr().out
    cli.main(["experts", "tally", "--clear"])
    assert "Toy-MoE: deleted" in capsys.readouterr().out and not tally.tally_path("Toy-MoE").exists()
    with pytest.raises(SystemExit):
        cli.main(["experts", "tally", "--rebuild", "--clear"])


def test_a_served_copys_own_tally_moves_into_its_checkpoint_on_link_and_clear_makes_no_folder(home):
    models.write_record(dict(name="Ckpt", shape=None, served=["Ckpt"]))
    models.write_record(dict(name="Copy-4bit", shape=None, served=["Copy-4bit"]))
    rng = np.random.default_rng(8)
    tally.add(save(home, "own", [1, 2, 3, 4], 2, rng, model="Copy-4bit")[0])
    tally.add(save(home, "ck", [5, 6, 7, 8], 2, rng, model="Ckpt")[0])
    assert tally.size("Copy-4bit")["traces"] == 1                         # its own, under its own name
    models.link("Copy-4bit", "Ckpt")                                       # the tally is not data that blocks a link
    assert tally.size("Copy-4bit")["traces"] == 2 and tally.Stored(tally.tally_path("Ckpt")).ids == {"own", "ck"}
    assert not (models.models_dir() / "Copy-4bit").exists()
    assert not tally.clear("Nobody") and not (models.models_dir() / "Nobody").exists()
    (models.models_dir() / "Ckpt" / ".experts-tally.npz.x.tmp").write_text("")
    assert tally.clear("Ckpt") and sorted(p.name for p in (models.models_dir() / "Ckpt").iterdir()) == ["checkpoint.json",
                                                                                                    "checkpoint.lock"]


@pytest.mark.parametrize("k", [1, 2])
def test_routing_of_fewer_than_three_experts_counts(home, k):
    toks = [1, 2, 3, 4, 5, 6]
    arrays = {f"router_ids/{l}": np.tile(np.arange(k, dtype=np.int32), (6, 1)) + l for l in range(L)}
    arrays.update({f"router_probs/{l}": np.full((6, k), 0.5, np.float32) for l in range(L)})
    path = Trace(arrays, toks, [""] * 6, 3, dict(num_experts=E, model="Toy-MoE")).save(Path(home) / "traces" / "d" / "k")
    assert tally.add(dict(id="k", path=str(path), model="Toy-MoE", client="pi-x", experts=True))
    st = tally.Stored(tally.tally_path("Toy-MoE"))
    assert st.layer(0)[1].sum() == 6 * k and st.cofire(0)[1].sum() == 6 * (k * (k - 1) // 2)


def test_a_session_counts_replies_and_takes_out_only_the_replies_the_tally_holds(home):
    rng = np.random.default_rng(9)
    system = list(range(500, 540))                                         # every chat's system prompt, 510 its odd word
    old = [save(home, f"O{i}", system + [int(t) for t in rng.integers(1, 60, 80)], 80, rng, client="pi-old",
                special={510: 7})[0] for i in range(6)]
    mine = [save(home, f"M{i}", system + [int(t) for t in rng.integers(1, 60, 80)], 80, rng, client="pi-me",
                 special={510: 7})[0] for i in range(3)]
    for e in old + mine[:2]:                                               # the newest reply not tallied yet
        tally.add(e)
    s = tally.layer_view("Toy-MoE", 0, "pi-me")
    assert s.N == 3 * 40, "replies only: the prompt is not the session's"
    assert 510 not in set(s.t.tolist()) and all(t != 510 for x in s.order for t, _, _ in s.tops(x))
    st = tally.stored("Toy-MoE")
    s = tally.session("Toy-MoE", "pi-me", st)
    assert list(s.fresh) == ["M2"] and s.read == {"M0", "M1", "M2"}
    rk, rc, n_ref = tally._reference(s.held, 0, st)
    held = sum(int(replies(home, [e])[e["id"]].layer(0)[1].sum()) for e in mine[:2])
    assert rc.sum() == st.layer(0)[1].sum() - held and n_ref == st.rows[0] - 2 * 40


def replies(home, entries):
    """Each entry's reply counted on its own (trace id -> Acc), as the session kept them before it merged them."""
    out = {}
    for e in entries:
        acc = tally.Acc()
        out[e["id"]] = acc if acc.add_trace(Path(e["path"]), e["id"], reply_only=True) else None
        acc.compact()
    return out


def old_score_session(accs, li, st):
    """score_session as it was before a session's replies were merged: a reply each (trace id -> Acc)."""
    def reference():
        held = [x for i, x in accs.items() if x is not None and li < x.L and i in st.ids]
        rk, rc = tally._merge(st.layer(li), tally._sum([x.layer(li) for x in held]), -1)
        return rk, rc, max(int(st.rows[li]) - sum(x.rows[li] for x in held), 1)
    mine = [a for a in accs.values() if a is not None and li < a.L]
    N, E = sum(a.rows[li] for a in mine), max([a.E for a in mine] + [st.E if st else 0])
    kc, co = tally._sum([a.layer(li) for a in mine]), tally._sum([a.cofire(li) for a in mine])
    a = tally.floor_of(N, E)
    if st is None or li >= st.L:
        return tally._layer("session", E, N, len(accs), kc, co, tally._mix, a, "session")
    rk, rc, n_ref = reference()
    re_, rt = tally._split(rk)
    g = tally.floor_of(n_ref, E)

    def own(e, t, c):
        V = int(max(rt.max() if len(rt) else 0, t.max() if len(t) else 0)) + 1
        tf = np.bincount(rt, weights=rc, minlength=V)
        vocab = len(np.union1d(np.unique(rt), np.unique(t)))
        pbg = (tf + 0.5) / (tf.sum() + 0.5 * vocab)
        nre = np.bincount(re_, weights=rc, minlength=E)
        k = (e.astype(np.uint64) << np.uint64(tally.SHIFT)) | t.astype(np.uint64)
        j = np.clip(np.searchsorted(rk, k), 0, max(len(rk) - 1, 0))
        cref = np.where(rk[j] == k, rc[j], 0.0) if len(rk) else np.zeros(len(k))
        return (cref + g * pbg[t]) / (nre[e] + g)
    return tally._layer("session", E, N, len(accs), kc, co, own, a, "tally")


def same_layer(a, b):
    assert (a.source, a.E, a.N, a.traces, a.against, a.order) == (b.source, b.E, b.N, b.traces, b.against, b.order)
    for f in ("e", "t", "c", "score", "ne", "lift", "co"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)


def test_a_session_keeps_its_replies_merged_and_scores_as_it_did_a_reply_each(home, monkeypatch):
    """The session keeps one merged count of the replies the tally holds, one of older replies it does not, and at
    most SESSION_FRESH apart; its scores are the ones a reply each gave in every state: some replies tallied and
    some not (more than SESSION_FRESH), the tally catching up, the tally cleared and grown again without them, none."""
    monkeypatch.setattr(tally, "SESSION_FRESH", 2)
    for e in planted(home, client="pi-old"):
        tally.add(e)
    mine = planted(home, n_traces=7, special={99: 9}, client="pi-me", start=100)
    for e in mine[:3]:
        tally.add(e)

    def check(entries, st):
        s = tally.session("Toy-MoE", "pi-me", st)
        assert s.held_ids == {e["id"] for e in entries} & (st.ids if st else set()) or st is None
        for li in range(L):
            same_layer(tally.score_session(s, li, st), old_score_session(replies(home, entries), li, st))
        return s
    s = check(mine, tally.stored("Toy-MoE"))
    assert list(s.fresh) == ["P105", "P106"] and s.apart_ids == {"P103", "P104"} and len(s.read) == 7
    for e in mine[3:5]:                                                           # the tally catches up: counted again
        tally.add(e)
    s = check(mine, tally.stored("Toy-MoE"))
    assert s.apart_ids == set() and list(s.fresh) == ["P105", "P106"]
    for e in mine[5:]:
        tally.add(e)
    st = tally.stored("Toy-MoE")
    s = check(mine, st)
    assert s.fresh == {} and s.held.N == sum(a.N for a in replies(home, mine).values())
    for li in range(L):
        same_layer(tally.score_session(s, li, None), old_score_session(replies(home, mine), li, None))
    tally.clear("Toy-MoE")                                                        # `experts tally --clear` elsewhere
    tally._SCORED.clear()
    check(mine, None)
    for e in planted(home, n_traces=3, client="pi-new", start=300):               # it grows again without them
        tally.add(e)
    s = check(mine, tally.stored("Toy-MoE"))
    assert s.held is None and len(s.apart_ids) == 5 and tally.layer_view("Toy-MoE", 0, "pi-me").traces == 7
    st = tally.rebuild("Toy-MoE") and tally.stored("Toy-MoE")                     # all counted again: held again
    s = check(mine, st)
    assert s.held_ids == {e["id"] for e in mine}
    # Some replies not tallied yet, fewer than SESSION_FRESH: the same scores as a reply each.
    more = planted(home, n_traces=2, special={99: 9}, client="pi-me", start=200)
    s = tally.session("Toy-MoE", "pi-me", st)
    assert list(s.fresh) == ["P200", "P201"]
    for li in range(L):
        same_layer(tally.score_session(s, li, st), old_score_session(replies(home, mine + more), li, st))
    assert tally.layer_view("Toy-MoE", 0, "pi-me").traces == 9


def test_a_long_trace_is_compacted_a_layer_at_a_time_with_the_same_counts(home, monkeypatch):
    """Its keys never all held at once: at most COMPACT_ROWS plus one layer's rows wait to be compacted."""
    T, layers = 3000, 8
    rng = np.random.default_rng(11)
    ids = rng.integers(0, E, (layers, T, K)).astype(np.int32)
    arrays = {f"router_ids/{l}": ids[l] for l in range(layers)}
    arrays.update({f"router_probs/{l}": rng.random((T, K)).astype(np.float32) for l in range(layers)})
    path = Trace(arrays, [int(t) for t in rng.integers(1, 500, T)], [""] * T, 100, dict(num_experts=E)).save(Path(home) / "long")
    whole = tally.Acc()
    monkeypatch.setattr(tally, "COMPACT_ROWS", 10 ** 9)
    whole.add_trace(path, "x")
    waiting, compact = [], tally.Acc.compact

    def spy(self):
        waiting.append(sum(len(k) for raw in self._k for k in raw) // tally.K_TOP)
        compact(self)
    monkeypatch.setattr(tally.Acc, "compact", spy)
    monkeypatch.setattr(tally, "COMPACT_ROWS", 1000)
    acc = tally.Acc()
    acc.add_trace(path, "x")
    assert max(waiting) <= 1000 + T < layers * T and len(waiting) >= layers // 2
    for li in range(layers):
        for a, b in ((acc.layer(li), whole.layer(li)), (acc.cofire(li), whole.cofire(li))):
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert acc.rows == whole.rows and acc.N == whole.N and np.array_equal(acc.seen, whole.seen)


def test_catch_up_queues_the_newest_untallied_traces_once(home, monkeypatch):
    rng = np.random.default_rng(10)
    for i in range(4):
        save(home, f"C{i}", [10 * i + j + 1 for j in range(8)], 4, rng)
    u = tally.Updater()
    monkeypatch.setattr(tally, "CATCH_UP", 3)
    tally.catch_up("Toy-MoE", u)
    tally.catch_up("Toy-MoE", u)                                           # once a process
    u.drain()
    assert tally.Stored(tally.tally_path("Toy-MoE")).ids == {"C1", "C2", "C3"}


def test_a_pass_over_the_index_reads_the_checkpoint_records_once_a_name(home, monkeypatch):
    """A rebuild, a catch-up and a session look up each index entry's checkpoint, and a lookup reads every
    checkpoint's record: each name is looked up once a pass, not once an entry."""
    models.write_record(dict(name="Toy", served=["Toy-a", "Toy-b"]))
    rng = np.random.default_rng(11)
    for i in range(30):
        save(home, f"R{i}", [10 * i + j + 1 for j in range(8)], 4, rng, model="Toy-a" if i % 2 else "Toy-b")
    reads, real = [0], models._records

    def counting():
        reads[0] += 1
        return real()
    monkeypatch.setattr(models, "_records", counting)

    class Queue:
        def submit(self, entry):
            pass
    for run in (lambda: tally.rebuild("Toy-a"), lambda: tally.catch_up("Toy-a", Queue()),
                lambda: tally.session("Toy-a", "pi-abc")):
        reads[0] = 0
        run()
        assert reads[0] < 10

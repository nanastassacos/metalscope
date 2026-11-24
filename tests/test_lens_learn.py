"""/jlens learn: the corpus (a directory, a .txt, the default), the fit child on a tiny model, the deadline, the
lens-learn job in the GPU window, which lens a model uses (find_for, /jlens use) and the routes."""
import datetime as dt
import io
import json
import signal
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from metalscope import cli, gpulock, jlens, models
from metalscope.jlens import corpus as LC
from metalscope.learn import command as Cmd, job as J, lens_fit_gpu as F, lens_job as LJ, spawn, state
from tests.helpers import fake_child, scope_client, tiny_llama
from tests.test_lens import FakeAdmin, fake_lens


class Tok:
    """Words as token ids (a tiny model's vocabulary: 100), and a chat template that marks each turn."""
    def encode(self, text, add_special_tokens=False):
        return [3 + (sum(map(ord, w)) % 90) for w in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
        out = []
        for m in msgs:
            out += [1 if m["role"] == "user" else 2] + self.encode(m["content"]) + [99]
        return out


def words(n, seed=0):
    return " ".join(f"w{(i * 7 + seed) % 50}" for i in range(n))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("METALSCOPE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


# ---------------------------------------------------------------- the corpus

def test_a_directory_gives_one_document_per_text_file_and_skips_the_rest(tmp_path, home, monkeypatch):
    src = tmp_path / "notes"
    (src / "sub").mkdir(parents=True)
    (src / "long.md").write_text(words(1200))
    (src / "sub" / "code.py").write_text(words(300, 1))
    (src / "short.txt").write_text(words(60, 2))
    (src / "tiny.txt").write_text("too short")
    (src / ".hidden").mkdir()
    (src / ".hidden" / "x.txt").write_text(words(100))
    (src / "node_modules").mkdir()
    (src / "node_modules" / "y.js").write_text(words(100))
    (src / "image.png").write_bytes(b"\x89PNG" + bytes(100))
    (src / "binary.txt").write_bytes(b"abc\0def" * 20)
    (src / "huge.txt").write_text(words(3000))
    monkeypatch.setattr(LC, "MAX_FILE_BYTES", 8000)
    kind, docs = LC.documents(str(src))
    assert kind == "dir" and [d.name for d in docs] == ["long.md", "short.txt", "sub/code.py", "tiny.txt"]   # huge.txt over the cap
    monkeypatch.setattr(LC, "MAX_FILE_BYTES", 1_000_000)
    kind, docs = LC.documents(str(src))
    assert [d.name for d in docs] == ["huge.txt", "long.md", "short.txt", "sub/code.py", "tiny.txt"]
    assert [d.kind for d in docs] == ["text", "text", "text", "code", "text"]
    c = LC.build(str(src), Tok(), window=32, max_depth=500)
    assert c.kind == "dir" and len(c.held) == 2 and c.description.endswith(": 5 files, windows up to 500 tokens deep")
    wins = c.windows(20)
    train = [i for i in c._pools()[0]]                     # the usable ones: tiny.txt is under 48 tokens
    assert all(c.docs[i].name != "tiny.txt" for i in train)
    first = [wins[i]["doc"] for i in range(len(train))]
    assert sorted(first) == sorted(c.docs[i].name for i in train)     # round 0: every document once
    long = {c.docs[i].name for i in c._pools()[1]}
    assert all(wins[i]["doc"] in long for i in range(len(train), 20))  # then only the long ones, at new depths
    for w in wins:
        assert len(w["ids"]) <= 500 + 32
        if w.get("window"):
            assert w["window"] == 32 and 0 < w["depth"] <= 500 and len(w["ids"]) == w["depth"] + 32
        else:
            assert len(w["ids"]) <= 32 and w["depth"] == 0
    with pytest.raises(ValueError, match="does not exist"):
        LC.documents(str(tmp_path / "nope"))
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="a directory of text files or one .txt"):
        LC.documents(str(tmp_path / "a.pdf"))


def test_one_txt_gives_its_paragraphs_shallow_only(tmp_path, home):
    f = tmp_path / "book.txt"
    f.write_text("\n\n".join(words(80 + i, i) for i in range(30)) + "\n\n  \n\nshort one\n")
    c = LC.build(str(f), Tok(), window=32)
    assert c.kind == "txt" and c.shallow_only and len(c.docs) == 31 and len(c.held) == 3
    assert "31 paragraphs, shallow windows only" in c.description
    wins = c.windows(500)
    fitted = [i for i in c.train if len(c.tokens(i)) >= LC.MIN_TOKENS]       # the short one has too few tokens
    assert len(wins) == len(fitted) >= 27                    # one window per paragraph, no more
    assert all("window" not in w and len(w["ids"]) == 32 for w in wins)
    assert {w["doc"] for w in wins}.isdisjoint(c.docs[i].name for i in c.held)
    assert all("window" not in h for h in c.heldout())


def fake_fetch(calls):
    """The dataset viewer's pages: 250 rows, headings and short lines among them."""
    def fetch(url):
        calls.append(url)
        off = int(url.split("offset=")[1].split("&")[0])
        rows = []
        for i in range(off, min(off + 100, 250)):
            text = f" = Heading {i} = \n" if i % 5 == 0 else (" " + words(160 if i % 3 else 40, i) + " \n")
            rows.append(dict(row_idx=i, row=dict(text=text)))
        return json.dumps(dict(rows=rows)).encode()
    return fetch


def test_the_default_downloads_wikitext_once_snapshots_the_code_and_never_reads_mixed_v1(home):
    (home / "corpus").mkdir(parents=True)
    (home / "corpus" / "mixed_v1.json").write_text(json.dumps(dict(train=dict(chat=[dict(ids=[1, 2], window=1)]),
                                                                    secret="PRIVATE-TRANSCRIPT")))
    calls = []
    kind, docs = LC.documents(None, fetch=fake_fetch(calls))
    paras = json.loads((home / "corpus" / LC.WIKITEXT_FILE).read_text())
    assert kind == "default" and len(calls) == 4 and len(paras) == 133      # over 600 characters, no headings; then no rows
    assert all(len(p) > 600 and not p.startswith("=") for p in paras) and paras[0] == words(160, 1)
    assert (home / "corpus" / LC.CODE_FILE).exists()
    assert any(d.name == "metalscope/jlens/corpus.py" and d.kind == "code" for d in docs)
    assert not any("PRIVATE-TRANSCRIPT" in d.text for d in docs)
    LC.documents(None, fetch=fake_fetch(calls))
    assert len(calls) == 4                                                   # on disk now: no second download
    c = LC.build(None, Tok(), fetch=fake_fetch(calls))
    wiki = sum(d.kind == "wiki" for d in c.docs)
    assert len([i for i in c.held if c.docs[i].kind == "wiki"]) == round(0.1 * wiki)
    wins = c.windows(14)
    kinds = [wins[i]["kind"] for i in range(14)]
    assert kinds == ["shallow"] * 4 + ["chat", "wiki", "chat", "wiki", "code"] * 2      # the default's order
    for w in (wins[i] for i in range(14)):
        if w["kind"] == "chat":
            assert w["ids"][-1] == 99 and 2 in w["ids"][-w["window"] - 200:]          # ends on an assistant turn
            assert w["depth"] > 900
    held_docs = {c.docs[i].name for i in c.held}
    assert all(w["doc"] not in held_docs for w in (wins[i] for i in range(14)))
    assert sorted({h["kind"] for h in c.heldout()}) == ["chat", "code", "shallow", "wiki"]


def test_held_out_documents_and_windows_are_the_same_every_time(tmp_path, home):
    assert LC.held_out(30, 0) == LC.held_out(30, 0) and len(LC.held_out(30, 0)) == 3
    assert len(LC.held_out(5, 0)) == 2 and len(LC.held_out(400, 0)) == 40
    assert LC.held_out(30, 0) != LC.held_out(30, 1)
    src = tmp_path / "d"
    src.mkdir()
    for i in range(12):
        (src / f"f{i}.md").write_text(words(400 + 30 * i, i))
    a, b = LC.build(str(src), Tok(), window=32), LC.build(str(src), Tok(), window=32)
    assert a.id == b.id and a.held == b.held and a.heldout() == b.heldout()
    # Window i is the same however many are asked for: a fit resumes on the same windows.
    assert a.windows(5)[:] == b.windows(40)[:5] and a.windows(40)[30] == b.windows(31)[30]
    assert LC.build(str(src), Tok(), window=32, seed=1).id != a.id
    (src / "f3.md").write_text(words(10, 99) + " changed")
    assert LC.build(str(src), Tok(), window=32).id != a.id                     # new text: a new corpus


def test_a_fit_resumes_for_the_same_model_and_corpus_only(tmp_path, home):
    src = tmp_path / "d"
    src.mkdir()
    for i in range(4):
        (src / f"f{i}.md").write_text(words(300, i))
    plan = spawn.lens_learn_plan("m", None, until="07:30", source=str(src))
    assert plan["kind"] == "dir" and plan["resume"] == 0 and plan["corpus_id"]
    d = F.fit_dir("m", plan["corpus_id"])
    d.mkdir(parents=True)
    (d / "fit_state.json").write_text(json.dumps(dict(done=7)))
    assert spawn.lens_learn_plan("m", None, until="07:30", source=str(src))["resume"] == 7
    assert spawn.lens_learn_plan("other", None, until="07:30", source=str(src))["resume"] == 0
    (src / "f9.md").write_text(words(300, 9))
    assert spawn.lens_learn_plan("m", None, until="07:30", source=str(src))["resume"] == 0
    # The default corpus is never downloaded to plan: no id until its text is on disk.
    p = spawn.lens_learn_plan("m", None, windows=3)
    assert (p["kind"], p["corpus_id"], p["windows_estimate"]) == ("default", None, 3)
    assert not (home / "corpus" / LC.WIKITEXT_FILE).exists()


# ---------------------------------------------------------------- the deadline

def test_the_deadline_is_the_next_hh_mm_and_no_window_starts_that_would_end_after_it():
    import datetime as dt
    now = dt.datetime(2026, 10, 14, 23, 0).timestamp()
    assert F.parse_until("07:30", now) == dt.datetime(2026, 10, 15, 7, 30).timestamp()
    assert F.parse_until("23:30", now) == dt.datetime(2026, 10, 14, 23, 30).timestamp()
    assert F.parse_until("23:00", now) == dt.datetime(2026, 10, 15, 23, 0).timestamp()      # now is past: tomorrow
    for bad in ("7.30", "25:00", "07:5", "seven", ""):
        with pytest.raises(ValueError, match="not a time"):
            F.parse_until(bad, now)
    deadline = now + 3600
    assert not F.should_stop(now, deadline, 3000) and F.should_stop(now, deadline, 3200)    # +15%: 3450 s > 3600? no; 3680 > 3600
    assert not F.should_stop(now, None, 10 ** 9)
    assert F.windows_by(now, deadline, 600) == 5 and F.windows_by(now, None, 600) is None and F.windows_by(now, now - 1, 60) == 0
    # The plan's estimate scales the one measured point (Flash Next, ~35 min a window) by the weights on disk.
    assert F.fit_estimate(None) is None


def test_the_plan_estimates_windows_from_the_config_and_refuses_bad_input(tmp_path, home):
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "model.safetensors").write_bytes(b"\0" * 1000)
    (ck / "config.json").write_text(json.dumps(dict(text_config=dict(hidden_size=2560, num_hidden_layers=48, hc_count=4))))
    import unittest.mock as um
    with um.patch.object(gpulock, "model_bytes", lambda p: int(55.75e9)):         # half Flash Next's 111.5 GB
        # Flash Next's shape: d_model passes a window, each 8e-6 s per layer and d_model (measured ~33 min).
        est = F.fit_estimate(str(ck))
        assert est == dict(d_model=2560, layers=48, min_per_window=45, overhead_min=9.0)
        now = time.mktime((2026, 10, 14, 23, 0, 0, 0, 0, -1))
        p = spawn.lens_learn_plan("m", str(ck), until="07:30", now=now)
        assert p["windows_estimate"] == int((8.5 * 3600 - 9 * 60) / (1.15 * 45 * 60))
        assert p["need_gb"] == F.fit_need_gb(str(ck)) and p["lens"] == "m-learned-20261014"
        assert spawn.lens_learn_plan("m", str(ck), until="07:30", windows=3, now=now)["windows_estimate"] == 3
    assert round(F.fit_state_gb(str(ck)), 2) == 4.59                          # Flash Next's sums
    # A dense 27B (d_model 5120, 64 layers): hours a window (measured 4.9 h), not minutes.
    (ck / "config.json").write_text(json.dumps(dict(text_config=dict(hidden_size=5120, num_hidden_layers=64))))
    assert F.fit_estimate(str(ck))["min_per_window"] == 227
    # 35B-A3B (measured 34 min), its shape at the top level beside a text_config without it.
    (ck / "config.json").write_text(json.dumps(dict(text_config=dict(vocab_size=10), hidden_size=2048, num_hidden_layers=40)))
    assert F.fit_estimate(str(ck))["min_per_window"] == 25
    (ck / "config.json").unlink()
    assert F.fit_estimate(str(ck)) is None                                    # no config: no guess
    with pytest.raises(ValueError, match="say until when"):
        spawn.lens_learn_plan("m", None)
    with pytest.raises(ValueError, match="not a time"):
        spawn.lens_learn_plan("m", None, until="7pm")
    with pytest.raises(ValueError, match="windows: a number"):
        spawn.lens_learn_plan("m", None, until="07:30", windows=0)
    with pytest.raises(ValueError, match="does not exist"):
        spawn.lens_learn_plan("m", None, until="07:30", source=str(tmp_path / "missing"))


# ---------------------------------------------------------------- which lens a model uses

def set_meta(d, **kw):
    """Fields added to a fake lens's lens.json."""
    (d / "lens.json").write_text(json.dumps(dict(json.loads((d / "lens.json").read_text()), **kw)))
    return d


def test_find_for_is_the_checkpoints_choice_else_its_only_lens(home):
    import mlx.core as mx
    a = fake_lens(home, "Flash", checkpoint="Flash")
    models.link("served", "Flash")
    assert jlens.find_for("served") == a == jlens.find_for("Flash")    # no choice recorded: its only lens
    b = fake_lens(home, "Flash-learned", checkpoint="Flash")
    assert jlens.find_for("served") is None                 # two and no choice: the logit lens
    fake_lens(home, "served", checkpoint="Other")           # a lens is its checkpoint folder's, whatever its name
    jlens.use("Flash-learned", "served")
    assert jlens.find_for("served") == b == jlens.find_for("Flash")    # recorded once, for every served name
    assert models.record("served")["lens"] == "Flash-learned"
    with pytest.raises(ValueError, match="a lens names its checkpoint"):
        jlens.save(jlens.Lens({0: mx.eye(2)}), jlens.lenses_dir("Flash") / "x")


def test_a_lens_name_can_repeat_across_checkpoints_and_each_is_its_own(home):
    a, b = fake_lens(home, "same", checkpoint="A"), set_meta(fake_lens(home, "same", checkpoint="B"), fit="fit-b")
    assert a != b and jlens.find_for("A") == a and jlens.find_for("B") == b
    assert jlens.lens_dirs() == [a, b] and jlens.lens_dirs("B") == [b]
    jlens.use("same", "A")
    assert jlens.in_use("same", "A") and not jlens.in_use("same", "B")
    assert F.lens_target("A", "same", "fit-a", today=dt.date(2026, 10, 14)) == "A-learned-20261014"   # chosen on A
    assert F.lens_target("B", "same", "fit-b") == "same"     # B's, not chosen there: its fit keeps writing it
    assert F.fit_dir("B", "abc") == models.model_dir("B") / "fits" / "lens-B-abc"
def test_lens_use_records_the_choice_on_the_checkpoint_and_takes_only_its_lenses(home):
    old = fake_lens(home, "Flash", checkpoint="Flash")
    new = set_meta(fake_lens(home, "served-learned-20261014", checkpoint="Flash"), learned_for="served",
                   heldout=dict(hit10=0.25, logit_hit10=0.15))
    models.link("served", "Flash")
    vec = models.vector_path("served", "calm")
    assert vec == models.vector_path("Flash", "calm")
    before = (old / "lens.json").read_text()
    out = jlens.use("served-learned-20261014", "served")
    assert (out["lens"], out["was"], out["checkpoint"], out["served"]) == ("served-learned-20261014", "logit", "Flash", ["served"])
    assert jlens.find_for("served") == new
    rec = models.record("Flash")
    assert (rec["lens"], rec["jlens"]) == ("served-learned-20261014", "served-learned-20261014")
    assert (old / "lens.json").read_text() == before and not (old / "lens.json.bak").exists()   # no lens is touched
    assert models.vector_path("served", "calm") == vec      # a lens never moves the data
    assert jlens.use("Flash", "served")["was"] == "served-learned-20261014" and jlens.find_for("served") == old
    fake_lens(home, "other-lens", checkpoint="Other")
    with pytest.raises(ValueError, match="no lens named other-lens under .*Flash"):    # another checkpoint's
        jlens.use("other-lens", "served")
    models.check_shape("served", models.shape_of(4, 1, 16))            # the fake lenses are 8 wide
    with pytest.raises(ValueError, match="is 8 wide, Flash is 16"):
        jlens.use("served-learned-20261014", "served")
    with pytest.raises(ValueError, match="no lens named nope"):
        jlens.use("nope", "served")
    with pytest.raises(ValueError, match="not a lens name"):
        jlens.use("../x", "served")
def test_lens_status_lists_every_candidate_ranked_by_held_out_hit10(home):
    fake_lens(home, "Flash", checkpoint="Flash")
    fake_lens(home, "Flash-ropebug", checkpoint="Flash")
    set_meta(fake_lens(home, "served-learned-20261014", checkpoint="Flash"), learned_for="served", n_prompts=11,
             heldout=dict(hit10=0.25, logit_hit10=0.15))
    fake_lens(home, "unrelated", checkpoint="Other")
    models.link("served", "Flash")
    jlens.use("Flash", "served")
    cs = jlens.status("served")["candidates"]
    assert [(c["name"], c["in_use"], c["why"]) for c in cs] == [
        ("served-learned-20261014", False, "learned for it"), ("logit", False, "J = I"), ("Flash", True, "in use"),
        ("Flash-ropebug", False, "its lens")]
    assert cs[0]["hit10"] == 0.25 and cs[1]["hit10"] == 0.15 and cs[2]["hit10"] is None   # the logit lens, on the learned one's windows
    lines = cli.lens_status_lines(jlens.status("served"))
    assert "    - served-learned-20261014: learned for it · hit@10 0.250 vs logit lens 0.150 · 11 windows" in lines
    assert "    * Flash: in use · not scored" in lines and "  checkpoint Flash" in lines


# ---------------------------------------------------------------- the child, on a tiny model

def child_run(monkeypatch, home, src, **kw):
    """lens_fit_gpu.run on the tiny llama (a real fit, real scoring); its stdout lines."""
    monkeypatch.setattr(F, "load_model", lambda path, loader: (tiny_llama(), Tok()))
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = F.run("/ckpt", "m", source=str(src), window=24, max_depth=120, **kw)
    return out, buf.getvalue().splitlines(keepends=True)


@pytest.fixture
def notes(tmp_path):
    src = tmp_path / "notes"
    src.mkdir()
    for i in range(8):
        (src / f"n{i}.md").write_text(words(200 + 40 * i, i))
    return src


def test_the_child_fits_exports_after_every_window_and_resumes(home, notes, monkeypatch):
    out, lines = child_run(monkeypatch, home, notes, windows=3, mem_cap_gb=50)
    evs = [json.loads(l[2:]) for l in lines if l.startswith("@@")]
    stages = [e["stage"] for e in evs if "stage" in e]
    assert stages[:4] == ["loading the model", "building the corpus", "scoring the logit lens", "sizing the batch"]
    assert stages[-1] == "done" and stages.count("fitting") == 4
    wins = [e["window"] for e in evs if "window" in e]
    assert [w["n"] for w in wins] == [1, 2, 3] and all(w["of"] == 3 and not w["approx"] for w in wins)
    assert all(0 <= w["hit10"] <= 1 and 0 <= w["logit_hit10"] <= 1 and w["compared"] is None for w in wins)
    assert out["stopped"] == "windows" and out["windows"] == 3 and out["this_run"] == 3
    d = jlens.lenses_dir("m") / out["lens"]
    assert out["lens"] == f"m-learned-{time.strftime('%Y%m%d')}" and d == Path(out["path"])
    lens = jlens.load(d)
    assert lens.layers == [0, 1] and lens.meta["n_prompts"] == 3 and lens.meta["learned_for"] == "m"
    assert lens.meta["checkpoint"] == "m" and lens.meta["heldout"]["hit10"] == wins[-1]["hit10"]
    assert jlens.find_for("m") == d                          # m's only lens: read until another is chosen
    assert {c["name"] for c in jlens.candidates("m")} == {out["lens"], "logit"}    # ranked by score, either first
    fit = F.fit_dir("m", out["corpus"]["id"])
    assert json.loads((fit / "fit_state.json").read_text())["done"] == 3
    # Again: resumes after 3, into the same lens directory.
    out2, lines2 = child_run(monkeypatch, home, notes, windows=2, mem_cap_gb=50)
    assert any("resuming the fit" in l for l in lines2) and out2["windows"] == 5 and out2["this_run"] == 2
    assert out2["lens"] == out["lens"] and jlens.load(d).meta["n_prompts"] == 5
    # In use now (/jlens use): the next run writes a new directory, never into a lens a model reads.
    jlens.use(out["lens"], "m")
    out3, _ = child_run(monkeypatch, home, notes, windows=1, mem_cap_gb=50)
    assert out3["lens"] == out["lens"] + "-2" and out3["windows"] == 6
    assert out3["heldout"]["compared"]["lens"] == out["lens"]        # the lens in use, on the same held-out windows
    # --fresh starts over.
    out4, _ = child_run(monkeypatch, home, notes, windows=1, fresh=True, mem_cap_gb=50)
    assert out4["windows"] == 1


def test_the_child_stops_at_the_deadline(home, notes, monkeypatch):
    out, lines = child_run(monkeypatch, home, notes, deadline=time.time() + 3, mem_cap_gb=50)
    assert out["stopped"] == "deadline" and 0 < out["this_run"] < 2000          # the tiny model's windows take milliseconds
    assert LJ.LensLearnJob.WEIGHTS.keys() >= {json.loads(l[2:])["stage"] for l in lines if l.startswith('@@{"stage"') and "done" not in l}
    # No window fits (the deadline is past): it says so once the batch is sized and writes nothing, for a new fit
    # (m2) or over an old one (--fresh keeps m's until a window runs).
    fit, lenses = F.fit_dir("m", out["corpus"]["id"]), jlens.lens_dirs()
    done = json.loads((fit / "fit_state.json").read_text())["done"]
    for name, fresh in (("m2", False), ("m", True)):
        buf = io.StringIO()
        with redirect_stdout(buf), pytest.raises(SystemExit, match=r"^about 1 min a window: none fits before the deadline$"):
            F.run("/ckpt", name, source=str(notes), window=24, max_depth=120, deadline=time.time() - 1, mem_cap_gb=50, fresh=fresh)
        evs = [json.loads(l[2:]) for l in buf.getvalue().splitlines() if l.startswith("@@")]
        assert any(e.get("none_fits") is True and "window_s" in e for e in evs)
    assert not F.fit_dir("m2", out["corpus"]["id"]).exists() and jlens.lens_dirs() == lenses
    assert json.loads((fit / "fit_state.json").read_text())["done"] == done


def test_the_batch_is_the_fastest_per_dim_under_the_cap(monkeypatch):
    import mlx.core as mx
    from metalscope.jlens import fit as jfit
    from metalscope.jlens import Lens
    peaks = {1: 2.0, 2: 3.0, 4: 5.0, 8: 9.0, 16: 17.0}
    per_pass = {1: 1.0, 2: 1.2, 4: 1.6, 8: 2.4, 16: 4.0}
    tried = []

    def fit(model, tok, prompts, dim_batch, **k):
        tried.append(dim_batch)
        return Lens({0: mx.zeros((2, 2))}, dict(last=dict(per_pass_s=per_pass[dim_batch], prefix_s=1.0,
                                                          peak_gb=peaks[dim_batch] * gpulock.GB / 1e9, d_out=64)))
    monkeypatch.setattr(jfit, "fit", fit)
    p = F.pick_batch(None, None, dict(ids=[1, 2, 3]), cap_gb=10, max_seq_len=8)
    assert tried == [1, 1, 2, 4, 8] and p["B"] == 8                 # a warm-up, then sizes; 16 is predicted at 17 GB: never tried
    assert p["window_s"] == 64 / 8 * 2.4 + 1.0
    p = F.pick_batch(None, None, dict(ids=[1]), cap_gb=1.0, max_seq_len=8)
    assert p["B"] == 1                                              # nothing under the cap: the smallest still ran
    tried.clear()
    p = F.pick_batch(None, None, dict(ids=[1, 2, 3]), cap_gb=10, max_seq_len=8, window_gb=4.0)
    assert p["B"] == 4 and p["peak_gb"] == 9.0 and tried == [1, 1, 2, 4]   # 5 GB + the window's 4 fits; 8 would need 13


# ---------------------------------------------------------------- the job

@pytest.fixture
def window(home, monkeypatch):
    """The GPU window faked, and the default corpus on disk (a job without a source fetches it first)."""
    (home / "corpus").mkdir(parents=True, exist_ok=True)
    (home / "corpus" / LC.WIKITEXT_FILE).write_text(json.dumps([words(200)]))
    (home / "corpus" / LC.CODE_FILE).write_text(json.dumps(dict(files={"a.py": "x = 1\n"})))
    monkeypatch.setattr(gpulock, "wait_no_loading", lambda admin, **k: True)
    monkeypatch.setattr(gpulock, "wait_available", lambda need, **k: True)
    monkeypatch.setattr(gpulock, "available", lambda: 100 * gpulock.GB)
    killed = []
    monkeypatch.setattr(J.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    return killed


def test_a_lens_learn_job_unloads_fits_until_the_deadline_and_reloads(window, home, notes, monkeypatch):
    out, lines = child_run(monkeypatch, home, notes, windows=2, mem_cap_gb=50)     # the real child's output
    seen = {}
    fake_child(monkeypatch, lines, seen)
    admin = FakeAdmin()
    now = time.mktime((2026, 10, 14, 23, 0, 0, 0, 0, -1))
    j = LJ.LensLearnJob("/ckpt", "m", source=str(notes), until="07:30", loader="mlx_lm", need_gb=7, admin=admin,
                       job="lens-learn-m-1", now=now)
    assert j.deadline == time.mktime((2026, 10, 15, 7, 30, 0, 0, 0, -1)) and j.state["until"] == "07:30"
    st = j.run()
    assert st["outcome"] == "done" and st["kind"] == "lens-learn" and admin.calls == [("unload", "m"), ("load", "m", True)]
    assert seen["holder"].what == "learning a lens for m until 07:30"
    assert seen["cmd"][1:] == ["-m", "metalscope.learn.lens_fit_gpu", "m", "--model", "/ckpt", "--loader", "mlx_lm",
                               "--memory-limit-gb", "94", "--source", str(notes), "--deadline", f"{j.deadline - LJ.RELOAD_S:.0f}"]
    assert st["result"]["lens"] == out["lens"] and st["fit"]["n"] == 2
    steps = {s["key"]: (s["status"], s["summary"]) for s in st["steps"]}
    assert [s["key"] for s in st["steps"]] == ["unload", "load", "corpus", "compare", "batch", "fit", "reload"]
    assert all(v[0] == "done" for v in steps.values())
    assert steps["fit"][1].startswith("2 windows this run, 2 in all · stopped after the 2 asked for · held-out hit@10 ")
    assert steps["compare"][1].startswith("held-out hit@10 in the workspace band: logit lens ")
    assert state.outcome_text(st).startswith("2 windows, hit@10 ")
    assert gpulock.held() is None and not window


def test_a_cancelled_lens_learn_keeps_its_windows_and_reloads(window, home, notes, monkeypatch):
    _, lines = child_run(monkeypatch, home, notes, windows=2, mem_cap_gb=50)
    cut = next(i for i, l in enumerate(lines) if '"window"' in l) + 1      # cancelled after the first window
    seen, admin = {}, FakeAdmin()
    fake_child(monkeypatch, lines[:cut], seen, interrupt=True)
    st = LJ.LensLearnJob("/ckpt", "m", windows=5, loader="mlx_lm", need_gb=7, admin=admin, job="lens-learn-m-2").run()
    assert st["outcome"] == "failed" and st["error"] == "stopped" and st["fit"]["n"] == 1
    assert admin.calls == [("unload", "m"), ("load", "m", True)] and window == [(987654321, signal.SIGTERM)]
    fit = next(s for s in st["steps"] if s["key"] == "fit")
    assert fit["status"] == "failed" and " min a window · hit@10 " in fit["detail"]
    with pytest.raises(ValueError, match="say until when"):
        LJ.LensLearnJob("/ckpt", "m", admin=admin, job="lens-learn-m-3")


def test_the_plan_counts_the_windows_the_child_resumes_from(home, notes, monkeypatch):
    """The plan's corpus id is the child's by default: the same window, depth and seed (jlens.corpus's)."""
    import inspect
    defaults = {k: p.default for k, p in inspect.signature(F.run).parameters.items()}
    assert (defaults["window"], defaults["max_depth"], defaults["seed"]) == (LC.WINDOW, LC.MAX_DEPTH, 0)
    plan = spawn.lens_learn_plan("m", None, windows=1, source=str(notes))
    assert plan["corpus_id"] == LC.build(str(notes), Tok()).id


def test_the_default_corpus_is_downloaded_before_the_gpu_is_taken(window, home, monkeypatch):
    for f in (LC.WIKITEXT_FILE, LC.CODE_FILE):
        (home / "corpus" / f).unlink()
    seen, admin, at = {}, FakeAdmin(), []

    def download(dest, log=None, **kw):
        at.append((gpulock.held(), list(admin.calls)))
        raise RuntimeError("could not download the default corpus (offline): give a directory or a .txt")
    monkeypatch.setattr(LC, "download_wikitext", download)
    fake_child(monkeypatch, [], seen)
    st = LJ.LensLearnJob("/ckpt", "m", windows=1, loader="mlx_lm", need_gb=7, admin=admin, job="lens-learn-m-6").run()
    assert at == [(None, [])] and st["outcome"] == "failed" and st["error"].startswith("could not download the default")
    assert admin.calls == [] and not seen                   # nothing unloaded, no child


def test_a_lens_learn_job_where_no_window_fits_fails_with_why_and_reloads(window, home, monkeypatch, capsys):
    lines = ['@@{"stage": "loading the model"}\n', '@@{"stage": "sizing the batch"}\n',
             '@@{"step": "sizing the batch", "summary": "1 output dims a pass"}\n', "about 4.9 h a window (5120 dims, 1 a pass)\n",
             '@@{"window_s": 17532.0, "none_fits": true}\n', "about 4.9 h a window: none fits before the deadline\n"]
    seen, admin = {}, FakeAdmin()
    fake_child(monkeypatch, lines, seen, rc=1)
    st = LJ.LensLearnJob("/ckpt", "m", until="13:00", loader="mlx_lm", need_gb=7, admin=admin, job="lens-learn-m-4").run()
    assert (st["outcome"], st["error"]) == ("failed", "a window takes about 4.9 h: none fits before 13:00")
    assert st["result"] is None and st["fit"] is None and admin.calls == [("unload", "m"), ("load", "m", True)]
    assert next(s for s in st["steps"] if s["key"] == "batch")["summary"] == "1 output dims a pass"
    assert not jlens.lenses_dir("m").exists()
    capsys.readouterr()
    cli.main(["jobs"])                                          # the reason on the job's lines
    assert "failed (a window takes about 4.9 h: none fits before 13:00)  (lens-learn-m-4)" in capsys.readouterr().out
    # A window that fits: its time is the job's, for `metalscope jobs`.
    j = LJ.LensLearnJob("/ckpt", "m", until="13:00", admin=admin, job="lens-learn-m-5")
    j.child_event(dict(window_s=1800.0, none_fits=False))
    assert j.state["window_s"] == 1800.0


def test_the_fit_needs_the_weights_the_sums_and_a_pass(tmp_path):
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "config.json").write_text(json.dumps(dict(hidden_size=2048, num_hidden_layers=24)))
    assert round(F.fit_state_gb(str(ck)), 2) == 0.36 and F.fit_need_gb(str(ck)) == 6         # the 6 GB floor
    assert F.fit_state_gb(str(tmp_path / "none")) == 0.0
    # The served Flash Next's shape and size on disk: 0.75 x 99 + 2 x 4.6 + 2 = 85 GB (the old rule asked 103).
    (ck / "config.json").write_text(json.dumps(dict(text_config=dict(hidden_size=2560, num_hidden_layers=48, hc_count=4))))
    with open(ck / "model.safetensors", "wb") as f:
        f.truncate(99 * gpulock.GB)                                   # sparse: no disk used
    assert round(F.fit_state_gb(str(ck)), 1) == 4.6 and F.fit_need_gb(str(ck)) == 85
    assert F.fit_sources(str(ck)) is None                             # every layer: its sums fit


def test_deepseek_v4_fits_only_its_scored_layers(tmp_path):
    """V4-Flash's every-layer sums would be 11.3 GB (42 x 4096 x hc_mult 4 x 4096), twice that with a window's: past
    FULL_STATE_GB the fit takes the layers the lens is scored at, and the memory it asks counts only those."""
    from metalscope.learn.lens_fit_gpu import eval_layers
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "config.json").write_text(json.dumps(dict(hidden_size=4096, num_hidden_layers=43, hc_mult=4)))
    sources = F.fit_sources(str(ck))
    assert sources == eval_layers(43)[0] == [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 41]
    assert round(F.fit_state_gb(str(ck)), 1) == 3.0                    # 12 x 4096 x 16384 x 4 bytes


def test_inkling_2_fits_every_visit_on_its_state(tmp_path):
    """Inkling 2 Mini from its config.json: 48 layer visits, each J as wide as the visit's state (1 to 5 sums of 2304),
    164 sources over the 47 fitted visits: 3.48e9 bytes of sums (3.2 GB as gpulock counts), not a 24-layer reading's
    0.45. Every visit fits; the time estimate counts visits."""
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "config.json").write_text(json.dumps(dict(model_type="inkling2_loop", text_config=dict(
        hidden_size=2304, num_hidden_layers=24, n_layers_per_attn_res_block=6,
        loop_config=dict(core_start=6, core_end=18, num_passes=3, gate_rank=64)))))
    assert F.fit_sources(str(ck)) is None
    assert F.fit_state_gb(str(ck)) * gpulock.GB == pytest.approx(164 * 2304 * 2304 * 4)
    assert round(F.fit_state_gb(str(ck)), 2) == 3.24
    assert F.fit_estimate(str(ck))["layers"] == 48


# ---------------------------------------------------------------- routes and the command line

@pytest.fixture
def client(home):
    return scope_client()


def test_the_learn_routes_plan_start_or_say_why_not(client, home, notes, monkeypatch):
    monkeypatch.setattr(Cmd, "model_path_for", lambda name, admin=None: None)
    assert client.get("/scope/api/lens/learn?until=07:30").status_code == 409          # no model loaded or named
    r = client.get("/scope/api/lens/learn", params=dict(model="m", until="07:30", path=str(notes)))
    assert r.status_code == 200 and r.json()["kind"] == "dir" and r.json()["until"] == "07:30"
    assert client.get("/scope/api/lens/learn", params=dict(model="m")).status_code == 400
    assert "not a time" in client.get("/scope/api/lens/learn", params=dict(model="m", until="9")).json()["detail"]
    started = []

    class Proc:
        pid = 4242

        def wait(self):
            return 0
    monkeypatch.setattr(spawn.subprocess, "Popen", lambda cmd, **k: started.append(cmd) or Proc())
    r = client.post("/scope/api/lens/learn", json=dict(model="m", until="07:30", path=str(notes), fresh=True))
    out = r.json()
    assert r.status_code == 200 and out["job"].startswith("lens-learn-m-")
    assert started[0][-11:] == ["jlens", "learn", "m", str(notes.resolve()), "--until", "07:30", "--fresh", "--job", out["job"],
                                "--origin", "pi"]
    st = json.loads((state.job_dir(out["job"]) / "state.json").read_text())
    assert (st["kind"], st["until"], st["outcome"]) == ("lens-learn", "07:30", None)
    assert client.post("/scope/api/lens/learn", json=dict(model="m")).status_code == 409
    lock = gpulock.GpuLock(gpulock.Holder("learning other")).acquire()
    try:
        r = client.post("/scope/api/lens/learn", json=dict(model="m", windows=3))
        assert r.status_code == 409 and "the GPU is in use" in r.json()["detail"]
    finally:
        lock.release()


def test_the_use_route_and_the_jobs_list(client, home):
    fake_lens(home, "m")
    new = set_meta(fake_lens(home, "m-learned-20261014"), learned_for="m")
    assert client.post("/scope/api/lens/use", json=dict(model="m", name="nope")).status_code == 409
    r = client.post("/scope/api/lens/use", json=dict(model="m", name="m-learned-20261014")).json()
    assert r["lens"] == "m-learned-20261014" and r["was"] == "logit" and r["checkpoint"] == "m"   # two lenses, none chosen
    assert r["live"] is None                                 # no tracer installed: read when tracing starts
    assert jlens.find_for("m") == new                        # the choice
    assert client.get("/scope/api/lens?model=m").json()["path"] == str(new)
    d = state.job_dir("lens-learn-m-9")
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps(dict(kind="lens-learn", job="lens-learn-m-9", name="m", outcome="done", pid=1,
                                                  started=1.0, result=dict(windows=11, lens="m-learned-20261014",
                                                                           heldout=dict(hit10=0.21, logit_hit10=0.15)))))
    brief = client.get("/scope/api/jobs").json()["jobs"][-1]
    assert brief["fit"] == dict(windows=11, hit10=0.21, logit_hit10=0.15, lens="m-learned-20261014")


def test_lens_learn_and_use_on_the_command_line(home, capsys):
    with pytest.raises(SystemExit, match="--until HH:MM or --windows N"):
        cli.main(["jlens", "learn", "m"])
    fake_lens(home, "m")
    fake_lens(home, "m-learned-20261014")
    cli.main(["jlens", "use", "m", "m-learned-20261014"])
    out = capsys.readouterr().out
    assert out.startswith("m-learned-20261014 is now the lens for m (was logit)") and "baseline" not in out
    with pytest.raises(SystemExit, match="error: no lens named x"):
        cli.main(["jlens", "use", "m", "x"])


def test_lens_use_logit_turns_the_checkpoints_lens_off_and_a_lens_takes_it_back(home, capsys):
    named = fake_lens(home, "served", checkpoint="served")
    set_meta(fake_lens(home, "served-learned-20261014", checkpoint="served"), learned_for="served",
             heldout=dict(hit10=0.25, logit_hit10=0.15))
    jlens.use("served", "served")
    out = jlens.use(jlens.LOGIT, "served")
    assert (out["lens"], out["path"], out["was"]) == ("logit", None, "served")
    assert jlens.find_for("served") is None and models.record("served")["lens"] == jlens.LOGIT
    cs = jlens.status("served")["candidates"]
    assert [(c["name"], c["in_use"]) for c in cs][:2] == [("served-learned-20261014", False), ("logit", True)]
    assert cs[1]["hit10"] == 0.15 and cs[1]["scored_on"] == "served-learned-20261014" and cs[1]["logit"]
    lines = cli.lens_status_lines(jlens.status("served"))
    assert lines[0] == "served: the logit lens (no J-lens in use: no word vectors)"
    assert "    * logit: in use · hit@10 0.150 on served-learned-20261014's windows" in lines
    assert jlens.use(jlens.LOGIT, "served")["was"] == "logit"                      # already off
    # A lens takes it back, the one named after the model too.
    assert jlens.use("served", "served")["was"] == "logit" and jlens.find_for("served") == named
    assert [c["name"] for c in jlens.status("served")["candidates"] if c["in_use"]] == ["served"]
    cli.main(["jlens", "off", "served"])                                          # what jlens use served logit does
    printed = capsys.readouterr().out
    assert printed.startswith("served now reads through the logit lens (was served)") and "no word vectors" in printed
    assert "metalscope jlens on served switches back" in printed
    assert "the server reads it when it next loads served" in printed            # METALSCOPE_HOME: the files only
    cli.main(["jlens", "on", "served"])                                           # the one it had
    assert capsys.readouterr().out.startswith("served is now the lens for served (was logit)")
def test_a_lens_in_the_models_folder_is_its_own_whatever_lens_json_names_and_only_at_its_width(home):
    import mlx.core as mx
    d = set_meta(fake_lens(home, "converted", checkpoint="Flash"), checkpoint="Other")
    models.link("served", "Flash")
    assert jlens.find_for("served") == d and jlens.use("converted", "served")["lens"] == "converted"
    models.check_shape("served", models.shape_of(4, 1, 16))            # the fake lens is 8 wide
    with pytest.raises(ValueError, match="lens converted is 8 wide, Flash is 16"):
        jlens.setup_for("served", 4)                                   # what the tracer reads at load: not read
    real = jlens.save(jlens.Lens({1: mx.zeros((8, 8))}, dict(checkpoint="Flash")), d)
    set_meta(real, d_model=None)                                       # no width in lens.json: the weights tell
    with pytest.raises(ValueError, match="lens converted is 8 wide, Flash is 16"):
        jlens.setup_for("served", 4)
    jlens.save(jlens.Lens({1: mx.zeros((16, 16))}, dict(checkpoint="Flash")), d)
    assert jlens.setup_for("served", 4).name == "converted"


def test_a_resumed_fit_never_writes_over_the_lens_jlens_on_goes_back_to(home):
    d = set_meta(fake_lens(home, "m-learned-20261014", checkpoint="m"), fit="fit-a")
    jlens.use(d.name, "m")
    jlens.use(jlens.LOGIT, "m")                                        # off: jlens on goes back to it
    assert jlens.in_use(d.name, "m")
    assert F.lens_target("m", d.name, "fit-a", today=dt.date(2026, 10, 15)) == "m-learned-20261015"


def test_a_model_with_no_lens_lists_no_logit_lens(home):
    assert jlens.status("nobody")["candidates"] == []


def test_lens_use_jlens_goes_back_to_the_lens_it_had_else_the_best_scored(home):
    named = fake_lens(home, "served", checkpoint="served")  # a published lens
    set_meta(fake_lens(home, "served-learned-20261014", checkpoint="served"), learned_for="served",
             heldout=dict(hit10=0.25, logit_hit10=0.15))
    assert jlens.pick_jlens("served") == "served-learned-20261014"   # none chosen of two: the better scored
    jlens.use("served", "served")
    assert jlens.pick_jlens("served") == "served"            # in use: that one
    jlens.use(jlens.LOGIT, "served")
    out = jlens.use(jlens.JLENS, "served")                   # the one it had, not the better-scored learned one
    assert (out["lens"], out["was"]) == ("served", "logit") and jlens.find_for("served") == named
    jlens.use("served-learned-20261014", "served")
    jlens.use(jlens.LOGIT, "served")
    assert jlens.use(jlens.JLENS, "served")["lens"] == "served-learned-20261014"
    # A model that never had one: the best by held-out hit@10.
    set_meta(fake_lens(home, "other-learned-a", checkpoint="other"), learned_for="other", heldout=dict(hit10=0.2, logit_hit10=0.1))
    set_meta(fake_lens(home, "other-learned-b", checkpoint="other"), learned_for="other", heldout=dict(hit10=0.3, logit_hit10=0.1))
    assert jlens.find_for("other") is None and jlens.pick_jlens("other") == "other-learned-b"
    assert jlens.use(jlens.JLENS, "other")["lens"] == "other-learned-b"
    with pytest.raises(ValueError, match="no J-lens for nobody"):
        jlens.use(jlens.JLENS, "nobody")

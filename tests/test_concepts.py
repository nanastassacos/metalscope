"""Concept profiles (the concepts package) on tiny models: corpus statistics, the
forced-choice readout, the sweep against a random control, the reader rule,
and the page."""
import json

import numpy as np
import pytest

from metalscope import concepts as C, vectors
from metalscope import models, steer
from metalscope.engines.mlx import MlxEngine, MlxTokenizer
from tests.helpers import fake_profile, generation_rung, tiny_llama, write_norms

D = 64


class Tok:
    """Whitespace-separated integer tokens; letters map to single ids."""
    LETTERS = {"A": 90, "B": 91, "C": 92, "D": 93}

    def encode(self, text, add_special_tokens=False):
        text = text.strip()
        if text in self.LETTERS:
            return [self.LETTERS[text]]
        out = []
        for t in text.replace("\n", " ").split():
            t = t.strip("):")
            if t in self.LETTERS:
                out.append(self.LETTERS[t])
            elif t.isdigit():
                out.append(int(t) % 90)
            else:
                out.append(sum(map(ord, t)) % 90)
        return out

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def write_concept(model, name="toy", n=8, task=True):
    """``name``'s texts in ``model``'s own folder."""
    d = models.concept_dir(model, name)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pos = [" ".join(str(x) for x in rng.integers(1, 20, size=6)) + " 77 77" for _ in range(n)]
    neg = [" ".join(str(x) for x in rng.integers(1, 20, size=6)) + " 33 33" for _ in range(n)]
    (d / "pos.txt").write_text("\n".join(pos))
    (d / "neg.txt").write_text("\n".join(neg))
    (d / "concept.json").write_text(json.dumps(dict(title="Toy", description="t", words=["77"], target_words=["5"])))
    if task:
        items = [dict(prompt=f"pick {i}", options=["5", "6", "7"], target="5") for i in range(3)]
        (d / "task.json").write_text(json.dumps(dict(instruction="letter", items=items)))
    return d


# ---------------------------------------------------------------- loading and statistics

def test_concepts_are_found_on_the_model_and_shipped(home):
    write_concept("tiny")
    names = [c["name"] for c in C.list_concepts("tiny")]
    assert "toy" in names and "fluffy" in names and "duty" in names and "toy" not in [c["name"] for c in C.list_concepts()]
    c = C.load_concept("tiny", "toy")
    assert len(c.pos) == 8 and c.task is not None and c.task.items[0].target_index == 0
    for shipped, pairs in (("fluffy", 16), ("duty", 16), ("doubt", 16), ("stoic", 48), ("flux", 16)):
        s = C.corpus.load_dir(C.shipped_dir(shipped), shipped)
        assert len(s.pos) == len(s.neg) == pairs, shipped
        assert s.task is not None and all(it.target in it.options for it in s.task.items), shipped
        assert s.words and s.target_words, shipped
    assert {"energized", "witty", "calm"} <= set(names)
    with pytest.raises(ValueError, match="no texts of fluffy on tiny"):
        C.load_concept("tiny", "fluffy")                      # shipped, but not on this model: learn copies them in
    d = write_concept("tiny", "lopsided")                     # pos and neg are matched pairs: unequal files are refused
    (d / "neg.txt").write_text("\n".join((d / "neg.txt").read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="matched pairs, line i of each"):
        C.load_concept("tiny", "lopsided")


def test_contrast_stats_separate_real_contrast_and_floor_random_one():
    rng = np.random.default_rng(1)
    P = rng.standard_normal((16, 40)).astype(np.float32)
    N = rng.standard_normal((16, 40)).astype(np.float32)
    P[:, 3] += 4.0                                     # a real difference along one axis
    st = C.contrast_stats(P, N, hc_count=2, permutations=99)
    assert st["dprime"] > st["heldout"]                                                 # in-sample flatters
    assert st["p_value"] == pytest.approx(1 / 100) and st["heldout"] > st["perm_p95"] + 1.0 and st["permutations"] == 99
    assert "shuffle_p95" not in st and "shuffle_mean" not in st
    assert len(st["streams"]) == 2 and st["streams"][0] > st["streams"][1]            # the axis is in the first block
    null = C.contrast_stats(rng.standard_normal((16, 40)), rng.standard_normal((16, 40)), permutations=99)
    assert null["heldout"] < null["perm_p95"] + 1.0 and null["p_value"] > 0.05
    assert C.contrast_stats(P, N, permutations=0)["p_value"] is None


def test_writer_and_carrier_rules():
    stats = {l: dict(heldout=h, p_value=p, perm_p95=0.5)
             for l, (h, p) in enumerate(zip([0.5, 0.8, 3.5, 4.0, 4.2, 4.1], [0.6, 0.2, 0.03, 0.005, 0.005, 0.005]))}
    w = C.writer_layer(stats)
    assert w["first"] == 3 and w["jump"] == 2 and w["rule"] == "p < 0.01" and w["alpha"] == 0.01 and "margin" not in w
    assert C.writer_layer({l: dict(st, p_value=None) for l, st in stats.items()})["first"] is None   # stats taken without permutations
    assert C.floor_of(stats[0]) == 0.5
    cos = {0: 0.2, 1: 0.9, 2: 0.95, 3: 0.4, 4: 0.8}
    band = C.carrier_band(cos)
    assert (band["start"], band["end"]) == (1, 3)
    assert C.carrier_band(cos, start=3) == dict(start=4, end=5, threshold=C.CARRIER_COS, mean_cos=0.8)
    assert C.carrier_band({0: 0.1, 1: 0.2}) is None


# ---------------------------------------------------------------- the task readout

def tiny(name="tiny") -> MlxEngine:
    return MlxEngine(tiny_llama(), Tok(), name)


def test_render_rotates_options_and_scores_target_within_letters():
    tok = MlxTokenizer(Tok())
    task = C.Task("letter", [C.Item("pick", ["5", "6", "7"], "5")])
    ids0, letters, slot0 = C.render_item(tok, task, task.items[0], 0)
    ids1, _, slot1 = C.render_item(tok, task, task.items[0], 1)
    assert letters == [90, 91, 92] and slot0 == 0 and slot1 == 1 and ids0 != ids1
    row = np.zeros(100)
    row[[90, 91, 92]] += [2.0, 1.0, 1.0]
    lp = (row - np.log(np.exp(row).sum()))[letters]                  # what engine.logprobs returns for the letters
    lo, mass, p = C.choice_scores(lp, 0)
    assert lo == pytest.approx(np.log(np.e ** 2 / (2 * np.e)))     # 2 - log(e + e)
    assert mass == pytest.approx(float(np.exp(lp).sum())) and mass < 1
    assert p < mass
    assert C.target_chosen(lp, 0) is True and C.target_chosen(lp, 1) is False
    # Chosen is the argmax over the option letters, not log-odds > 0: with four options a target at 0.4 against
    # three at 0.2 is chosen while its log-odds against the rest are below 0.
    four = np.log([0.4, 0.2, 0.2, 0.2])
    assert C.choice_scores(four, 0)[0] < 0 and C.target_chosen(four, 0)
    prompts = C.prompts_for(tok, task)
    assert len(prompts) == 3 and [p["slot"] for p in prompts] == [0, 1, 2]


def test_sweep_zero_strength_is_identity_and_reader_picks_the_effective_layer(home):
    engine = tiny()
    name = "tiny"
    write_norms(name, [0, 1, 2], value=4.0)
    task = C.Task("letter", [C.Item(f"pick {i}", ["5", "6", "7"], "5") for i in range(2)])
    prompts = C.prompts_for(engine.tok, task)
    rng = np.random.default_rng(0)
    # A direction that pushes the target letter: the unembedding row of letter "A"'s id, mapped back through the final layer? Keep it
    # simple: a random vector, and check the machinery rather than the physics.
    vec = vectors.Vector("v", {l: vectors.unit(rng.standard_normal(D)) for l in (0, 1, 2)}, dict(hc_count=1))
    sw = C.sweep(engine, vec, steer.norm_table(name), prompts, layers=[0, 2], strengths=[0.0, 1.0], seeds=(1,))
    assert sw["determinism"] == 0.0
    np.testing.assert_allclose(sw["concept_logodds"][:, 0], np.broadcast_to(sw["base_logodds"], (2, len(prompts))), atol=1e-5)
    np.testing.assert_allclose(sw["control_logodds"][:, 0], np.broadcast_to(sw["base_logodds"], (2, len(prompts))), atol=1e-5)
    assert np.isfinite(sw["concept_logodds"]).all() and (sw["concept_mass"] <= 1.0 + 1e-6).all()
    np.testing.assert_array_equal(sw["concept_chosen"][:, 0], np.broadcast_to(sw["base_chosen"], (2, len(prompts))))
    assert set(np.unique(sw["concept_chosen"])) <= {0.0, 1.0} and ((sw["control_chosen"] >= 0) & (sw["control_chosen"] <= 1)).all()
    rd = C.reader_layer(sw, mass_floor=0.0)
    assert rd["layer"] in (0, 2) and rd["best"] is not None and set(rd["per_layer"]) == {0, 2}
    assert rd["base_chosen"] == pytest.approx(sw["base_chosen"].mean())
    assert rd["per_layer"][0]["dose"][0]["chosen"] == pytest.approx(rd["base_chosen"]) and "control_chosen" in rd["best"]
    assert len(rd["per_layer"][0]["dose"]) == 2
    # With an impossible floor no layer qualifies.
    assert C.reader_layer(sw, mass_floor=2.0)["layer"] is None
    # No hooks left behind.
    from metalscope.hooks import is_hooked
    assert not any(is_hooked(l) for l in engine.model.model.layers)


def test_generation_runs_three_conditions_and_a_ladder(home):
    engine = tiny()
    name = "tiny"
    write_norms(name, [0, 1, 2], value=4.0)
    write_concept(name, n=4)
    concept = C.load_concept(name, "toy")
    rng = np.random.default_rng(0)
    vec = vectors.Vector("v", {1: vectors.unit(rng.standard_normal(D))}, dict(hc_count=1))
    g = C.generation(engine, concept, vec, steer.norm_table(name), layer=1, strength=0.5, max_tokens=5,
                     ladder=(0.5, 2.0), ladder_items=1)
    assert len(g["items"]) == 3 and set(g["items"][0]["replies"]) == {"base", "concept", "control"}
    r = g["items"][0]["replies"]["concept"]
    assert r["n_tokens"] <= 5 and 0 < r["distinct"] <= 1 and isinstance(r["hits"], list)
    # The unsteered rung is always run first, under the same instruction and budget.
    assert [x["strength"] for x in g["ladder"]] == [0.0, 0.5, 2.0] and g["fluency"]["reference"] is not None
    assert [r["strength"] for r in g["fluency"]["rungs"]] == [0.5, 2.0] and g["fluency"]["reference"]["n"] == 1
    assert set(g["summary"]) == {"base", "concept", "control"} and 0 <= g["summary"]["concept"]["target"] <= 1
    abc = C.option_letters(C.Task("", []), 3)
    assert C.measure._letter_of("B) because it is soft", abc) == "B" and C.measure._letter_of("I would pick B", abc) is None
    assert C.measure._letter_of("D. no", abc) is None                                           # a letter outside the options
    assert C.measure._letter_of("(2) since", C.option_letters(C.Task("", [], ["1", "2", "3"]), 3)) == "2"   # the task's own letters
    assert C.measure._concept_hits("The cat is very Fluffy and soft.", ["fluffy", "soft", "fur"]) == ["fluffy", "soft"]
    assert g["ladder_instruction"] == C.EXPLAIN_LONG and g["ladder_tokens"] == C.LADDER_TOKENS and "early_stop" in g["ladder"][0]
    # Early stop: on its own, in the first half of the budget, not at a sentence end.
    assert C.measure._fluency([1, 2, 3], 100, "I would choose the fluffy")["early_stop"] is True
    assert C.measure._fluency([1, 2, 3], 100, "B) The cat.")["early_stop"] is False
    assert C.measure._fluency(list(range(60)), 100, "half way there")["early_stop"] is False
    assert C.measure._fluency([1], 100, "B")["short"] is True and C.measure._fluency(list(range(C.SHORT_TOKENS)), 100, "x")["short"] is False
    assert C.measure._concept_hits("Substantially changed, I think", ["change", "shift"], exclude="A) Largely the same B) Substantially changed") == []
    from metalscope.hooks import is_hooked
    assert not any(is_hooked(l) for l in engine.model.model.layers)


def _rung(s, n, stopped=True, distinct=0.7, early=False, text="A) a paragraph."):
    return [dict(item=i, strength=s, target_letter="A", text=text, n_tokens=n, stopped=stopped, distinct=distinct, distinct_window=distinct,
                 early_stop=early) for i in range(4)]


def test_fluency_edge_is_measured_against_the_unsteered_rung():
    # A chatty model: unsteered replies already run past the budget. Overrunning is not damage.
    ladder = _rung(0.0, 160, stopped=False) + _rung(0.2, 160, stopped=False) + _rung(0.4, 160, stopped=False, distinct=0.6)
    gen = dict(ladder=ladder)
    fr = C.fluency_rungs(gen)
    assert fr["edge"] is None and not any(r["broken"] for r in fr["rungs"])
    assert fr["reference"] == dict(n=4, overrun=1.0, short=0.0, early=0.0, distinct=0.7)
    assert C.fluency_rungs(dict(ladder=[x for x in ladder if x["strength"]])) is None              # without the unsteered rung: nothing to measure against
    # Bare letters on three of four replies: short rises 0.75 over the reference.
    bare = _rung(0.8, 1, text="B")
    bare[0].update(n_tokens=120, text="A) fine.")
    fr = C.fluency_rungs(dict(ladder=ladder + bare))
    assert fr["edge"] == 0.8 and fr["rungs"][-1]["why"] == ["short"] and fr["rungs"][-1]["short"] == 0.75
    # One more short reply than unsteered (+0.25) is within the margin; two (+0.5) is over it.
    one = _rung(0.8, 120)
    one[0].update(n_tokens=2)
    assert C.fluency_rungs(dict(ladder=_rung(0.0, 120) + one))["edge"] is None
    one[1].update(n_tokens=2)
    assert C.fluency_rungs(dict(ladder=_rung(0.0, 120) + one))["edge"] == 0.8
    # Repetition: distinct falls from 0.7 to 0.5 (more than 0.15).
    fr = C.fluency_rungs(dict(ladder=_rung(0.0, 120) + _rung(0.4, 120, distinct=0.6) + _rung(1.6, 160, stopped=False, distinct=0.5)))
    assert fr["edge"] == 1.6 and fr["rungs"][-1]["why"] == ["distinct"] and not fr["rungs"][0]["broken"]
    # Running past the budget is shown but breaks nothing: a concept that makes replies longer is not damage.
    fr = C.fluency_rungs(dict(ladder=_rung(0.0, 120) + _rung(0.2, 160, stopped=False)))
    assert fr["edge"] is None and fr["rungs"][0]["overrun"] == 1.0 and fr["rungs"][0]["why"] == []
    assert C.fluency_rungs(None) is None and C.fluency_rungs(dict(ladder=[])) is None


def _fluent(n, seed):
    """A fluent-looking reply of ``n`` token ids: function words recur, content words are mostly fresh, so its whole-reply
    distinct ratio falls with length while each window's does not."""
    import random
    rnd = random.Random(seed)
    return [rnd.randrange(10) if rnd.random() < 0.4 else 1000 + 1000 * seed + rnd.randrange(400) for _ in range(n)]


def _reply(s, toks, item=0, text="A) a paragraph that ends."):
    from metalscope.concepts.measure import _fluency
    return dict(item=item, strength=s, target_letter="A", text=text if toks else "", **_fluency(toks, C.LADDER_TOKENS, text if toks else ""))


def _ref(n=30):
    return [_reply(0.0, _fluent(n, i), i) for i in range(4)]


def test_distinct_skips_short_replies_and_does_not_fall_with_length():
    from metalscope.concepts.measure import _rung
    ref = _ref()
    # (i) One empty reply beside three fluent ones: counted by short (one more than unsteered is within the margin), not by
    # distinct; averaged over every reply the ratio would drop by more than DISTINCT_DROP.
    rung = [_reply(0.4, _fluent(30, 10 + i), i) for i in range(3)] + [_reply(0.4, [], 3)]
    assert np.mean([x["distinct"] for x in ref]) - np.mean([x["distinct"] for x in rung]) > C.DISTINCT_DROP
    fr = C.fluency_rungs(dict(ladder=ref + rung))
    assert fr["edge"] is None and fr["rungs"][0]["why"] == [] and fr["rungs"][0]["short"] == 0.25
    # A rung with no reply that is not short takes the unsteered rung's ratio; an unsteered rung of bare letters has no distinct test.
    assert _rung([_reply(0.4, [], 0)], distinct_fallback=0.9)["distinct"] == 0.9 and _rung([_reply(0.0, [], 0)])["distinct"] is None
    bare = C.fluency_rungs(dict(ladder=[_reply(0.0, [90], i, "A") for i in range(4)] + [_reply(0.4, [90, 5, 5, 5, 5, 5, 5, 5, 5, 5], i) for i in range(4)]))
    assert bare["reference"]["distinct"] is None and "distinct" not in bare["rungs"][0]["why"]
    # (ii) Fluent 140-token replies against 30-token unsteered ones: the whole-reply ratio falls by more than DISTINCT_DROP
    # with length alone; the windowed ratio does not.
    long = [_reply(0.8, _fluent(140, 20 + i), i) for i in range(4)]
    assert np.mean([x["distinct"] for x in ref]) - np.mean([x["distinct"] for x in long]) > C.DISTINCT_DROP
    fr = C.fluency_rungs(dict(ladder=ref + long))
    assert fr["edge"] is None and fr["distinct_window"] == C.DISTINCT_WINDOW
    # (iii) Two replies looping a 12-token phrase: broken by distinct (and by nothing else).
    loop = [_reply(1.6, (list(range(2000, 2012)) * 12)[:140], i) for i in range(2)] + [_reply(1.6, _fluent(140, 30 + i), 2 + i) for i in range(2)]
    fr = C.fluency_rungs(dict(ladder=ref + loop))
    assert fr["edge"] == 1.6 and fr["rungs"][0]["why"] == ["distinct"]


def test_fluency_rungs_take_dose_zero_as_reference_wherever_it_sits():
    import random
    ladder = _ref() + [_reply(0.4, _fluent(30, 10 + i), i) for i in range(4)] + [_reply(1.6, [], i) for i in range(4)]
    ladder += [_reply(-0.4, _fluent(30, 40 + i), i) for i in range(4)]
    shuffled = list(ladder)
    random.Random(3).shuffle(shuffled)
    assert shuffled[0]["strength"] != 0.0
    fr = C.fluency_rungs(dict(ladder=shuffled))
    same = C.fluency_rungs(dict(ladder=ladder))      # equal up to the order the replies are averaged in
    assert [(r["strength"], r["broken"], r["why"]) for r in fr["rungs"]] == [(r["strength"], r["broken"], r["why"]) for r in same["rungs"]]
    assert fr["reference"] == pytest.approx(same["reference"]) and fr["edge"] == same["edge"]
    assert fr["reference"]["n"] == 4 and fr["reference"]["short"] == 0.0
    assert [r["strength"] for r in fr["rungs"]] == [-0.4, 0.4, 1.6] and fr["edge"] == 1.6


def test_profile_end_to_end_on_tiny_model_with_identity_lens(home):
    engine = tiny()
    name = "tiny"
    write_norms(name, range(3), value=4.0)
    write_concept(name, n=6)
    concept = C.load_concept(name, "toy")
    prof = C.profile(engine, concept, log_fn=lambda s: None)              # presence, as learn's GPU stage starts
    assert prof["concept"] == "toy" and set(prof["stats"]) == {"0", "1", "2"} and prof["inputs"]["word_tokens"]
    assert "writer" in prof and "adjacent_cosine" in prof and "reader" not in prof and "coarse" not in prof
    # learn's selection sweep (learn/gpu.py measure_push), on two items at two layers.
    from metalscope.concepts.run import sweep_inputs, sweep_json
    vec, table = vectors.load_vector(name, "toy"), steer.norm_table(name)
    prompts = C.prompts_for(engine.tok, concept.task, 2)
    sw = C.sweep(engine, vec, table, prompts, [0, 2], [0.5], (1,))
    prof["coarse"] = sweep_json(sw)
    prof["coarse"]["reader"] = prof["reader"] = C.reader_layer(sw)
    prof["inputs"].update(sweep_inputs(table, [0, 2], (0.5,), (1,), 2))
    L, _ = prof["reader"]["layer"], prof["reader"]["best"]["strength"]
    C.attach_generation(prof, C.generation(engine, concept, vec, table, L, 0.5, max_tokens=4, ladder=(0.5,),
                                           ladder_items=1))
    out = C.write_profile(prof, home / "out")
    assert (out / "profile.json").exists()
    saved = json.loads((out / "profile.json").read_text())
    assert any("generation at" in s for s in C.summary_lines(saved))
    assert "meter_layer" not in prof and not any("scores_pos" in st for st in prof["stats"].values())
    assert any("(against the unsteered rung)" in s for s in C.summary_lines(saved)) and "edge_rule" not in C.summarize(saved)["generation"]
    # The target choice rate at the reader layer and dose, beside the log-odds.
    ch = C.summary.reader_choice(saved)
    assert set(ch) == {"base", "concept", "control"} and ch["base"] == saved["reader"]["base_chosen"] and ch["concept"] == saved["reader"]["best"]["chosen"]
    assert any("as the most likely option letter" in s for s in C.summary_lines(saved))
    assert not any("half the letter mass" in s for s in C.summary_lines(saved)) and "flipped" not in C.summarize(saved)["reader"]
    # No chat calibration: no /steer default, and the line says so.
    assert "  /steer default: not calibrated (concept calibrate sets it)" in C.summary_lines(saved)
    assert vectors.find_vector(name, "toy") is not None          # the vector is stored for the steer header
    assert any("reader" in line for line in C.summary_lines(prof))


# ---------------------------------------------------------------- paired folds, the paired null

def _matched(seed, n=16, width=64, common=1.0, concept=1.5, noise=0.5):
    """A matched corpus: pair i shares a strong common component (its wording) and differs by a weak concept axis."""
    rng = np.random.default_rng(seed)
    v = vectors.unit(rng.standard_normal(width))
    shared = common * rng.standard_normal((n, width))
    step = concept * np.sqrt(width) * 0.15 * v
    return shared + step + noise * rng.standard_normal((n, width)), shared - step + noise * rng.standard_normal((n, width))


def test_paired_folds_on_a_matched_corpus():
    from metalscope.concepts.stats import _heldout_dprime
    for seed in range(4):
        P, N = _matched(seed)
        paired = _heldout_dprime(P, N, 4, np.random.default_rng(0))
        assert paired > 2.0, (seed, paired)
        assert C.contrast_stats(P, N, permutations=49)["heldout"] == pytest.approx(paired)      # same seed, same folds
    P, N = _matched(0)
    with pytest.raises(ValueError, match="paired folds need"):
        C.contrast_stats(P, N[:-1], permutations=9)


def test_paired_null_flips_labels_within_pairs():
    P, N = _matched(1)
    st = C.contrast_stats(P, N, permutations=99)
    assert st["p_value"] == pytest.approx(1 / 100) and st["heldout"] > st["perm_p95"] + 1.0 and abs(st["perm_mean"]) < 0.5
    # No concept axis, only the shared wording: the paired null does not call it present.
    rng = np.random.default_rng(5)
    shared = 3 * rng.standard_normal((16, 64))
    null = C.contrast_stats(shared + 0.5 * rng.standard_normal((16, 64)), shared + 0.5 * rng.standard_normal((16, 64)), permutations=99)
    assert null["p_value"] > 0.05


def _stored_profile(n=16, layers=(0, 1, 2), seed=0):
    """A stored profile with every section."""
    rng = np.random.default_rng(seed)
    stats = {str(l): dict(dprime=3.0, heldout=1.0, natural_norm=1.0, p_value=0.2, perm_p95=0.6, perm_mean=0.0, permutations=200, streams=[],
                          fold_repeats=C.FOLD_REPEATS) for l in layers}
    return dict(concept="toy", title="Toy", model="tiny", n_pos=n, n_neg=n, position="mean",
                inputs=dict(n_pos=n, n_neg=n, position="mean", n_layers=len(layers), pos_example="a", neg_example="b", sink=4,
                            words=["w"], word_tokens=["w"], target_words=["t"], target_tokens=["t"]), stats=stats,
                adjacent_cosine={"0": 0.1, "1": 0.2}, writer=dict(first=None, jump=1, rule="p < 0.01", alpha=0.01), carrier=None,
                reader=dict(layer=1, base_p=0.3, base_chosen=0.25, best=dict(strength=0.5, concept=0.8, control=0.1, effect=0.7, concept_mass_ratio=0.97,
                                                                         control_mass_ratio=0.99, chosen=0.6, control_chosen=0.3)),
                coarse=dict(layers=[0, 2], x=rng.random(3).tolist()), full=dict(layers=[1]), split=None, chain=None,
                lens_words={"1": [["w", 0.5]]}, readability={"1": dict(diff_all=0.1, diff_last=0.2)},
                generation=dict(layer=1, strength=0.5, task_items=[], items=[], ladder=[], reader_strength=0.5, below_edge=True,
                                summary={k: dict(target=0.5, concept_words=0.2) for k in ("base", "concept", "control")}),
                word_ids=[1], target_ids=[2], seconds=12.5)


def test_an_old_profile_with_the_retired_sections_still_reads():
    """A profile from the retired ``concept profile`` (lens readability and words, the chain, the refined sweep, the
    meter's layer and sentence scores) or with the words-or-more split loads and summarizes its other parts; the
    retired sections are not read."""
    prof = dict(_stored_profile(), generation=None, meter_layer=1)          # the fixture's generation is a stub
    for st in prof["stats"].values():
        st.update(scores_pos=[1.0, 2.0], scores_neg=[-1.0, -2.0])
    prof["chain"] = dict(layer=1, strength=0.5, lens_layers=[1, 2], conditions={
        k: dict(logodds=0.0, words={"1": 0.0, "2": 1.0}, target={}) for k in ("base", "concept", "control")})
    prof["full"] = dict(layers=[1], strengths=[0.5], prompts=dict(base_p=[0.2]))
    whole = dict(effect=0.7)
    prof["split"] = dict(layer=1, strength=0.5, rank=6, width=8, energy=0.9, readable_share=0.4, random_share=0.3,
                         conditions=dict(whole=whole, readable=whole, remainder=whole), readable_words=[["w", 0.5]],
                         chain={"readable": prof["chain"], "remainder": prof["chain"]})
    lines = C.summary_lines(prof)
    assert "split" not in C.summarize(prof) and not any("words or more" in l for l in lines)
    assert any(l.strip().startswith("reader: L1") for l in lines) and not any(l.strip().startswith(("lens:", "chain:")) for l in lines)
    assert "lens_words" not in C.summarize(prof) and C.summarize(prof)["reader"]["layer"] == 1
    assert "meter" not in C.summarize(prof)


def _loop_heldout(P, N, folds, rng):
    """The single-split paired estimator as a plain loop (the reference the batched one must match)."""
    n = len(P)
    fold = np.empty(n, dtype=int)
    fold[rng.permutation(n)] = np.arange(n) % folds
    sp, sn = np.full(n, np.nan), np.full(n, np.nan)
    for k in range(folds):
        test, train = fold == k, fold != k
        uk = vectors.unit(P[train].mean(0) - N[train].mean(0))
        sp[test], sn[test] = P[test] @ uk, N[test] @ uk
    return vectors.dprime(sp, sn)


def test_fold_averaging_matches_the_loop_and_reduces_split_noise():
    from metalscope.concepts.stats import _heldout_dprime
    P, N = _matched(3, concept=0.6)
    # One split, batched in pair space, is the plain loop to float precision.
    for seed in range(3):
        assert _heldout_dprime(P, N, 4, np.random.default_rng(seed), repeats=1) == pytest.approx(_loop_heldout(P, N, 4, np.random.default_rng(seed)), abs=1e-5)
    # Same corpus, different fold seeds: averaging five splits shrinks the seed-to-seed spread.
    one = np.array([_heldout_dprime(P, N, 4, np.random.default_rng(s), repeats=1) for s in range(40)])
    five = np.array([_heldout_dprime(P, N, 4, np.random.default_rng(s), repeats=5) for s in range(40)])
    assert one.std() > 0.05 and five.std() < 0.7 * one.std(), (one.std(), five.std())
    assert abs(five.mean() - one.mean()) < 2 * one.std() / np.sqrt(len(one)) + 0.1
    # contrast_stats uses the averaged estimator for the observed value and for every permutation, and says so.
    st = C.contrast_stats(P, N, permutations=49)
    assert st["fold_repeats"] == C.FOLD_REPEATS and st["heldout"] == pytest.approx(_heldout_dprime(P, N, 4, np.random.default_rng(0)))
    assert C.contrast_stats(P, N, permutations=0, repeats=1)["fold_repeats"] == 1


def _steer_profile(reader=(8, 0.8), gen=(8, 0.4), edge=0.8):
    """A profile with a reader, a generation and a ladder whose first broken rung is ``edge`` (None: nothing breaks)."""
    ok = dict(text="A) a fine reason, said plainly and to the end.", stopped=True, distinct=1.0, distinct_window=1.0, n_tokens=30, early_stop=False)
    bad = dict(text="", stopped=True, distinct=0.0, distinct_window=0.0, n_tokens=0, early_stop=False)
    ladder = [dict(item=i, strength=s, **(bad if edge is not None and s >= edge else ok)) for i in range(4) for s in (0.0, 0.2, 0.4, 0.8, 1.6, 3.2)]
    p = dict(reader=dict(layer=reader[0], best=dict(strength=reader[1])), generation=None)
    if gen:
        C.attach_generation(p, dict(layer=gen[0], strength=gen[1], ladder=ladder,
                                    summary={k: dict(target=0.5, concept_words=0.2) for k in ("base", "concept", "control")}))
    return p


def test_generation_dose():
    # learn's test dose, never a /steer default. The generation ran one dose below the reader's (which is the edge):
    # the generation's dose.
    p = _steer_profile()
    assert C.generation_dose(p) == dict(layer=8, strength=0.4)
    # A generation at another layer is not evidence for the reader layer: the reader's dose, below the edge.
    p = _steer_profile(reader=(8, 0.4), gen=(12, 0.4), edge=1.6)
    assert C.generation_dose(p) == dict(layer=8, strength=0.4)
    # The generation itself at the edge: half the edge.
    p = _steer_profile(reader=(8, 0.8), gen=(8, 0.8), edge=0.8)
    assert C.generation_dose(p) == dict(layer=8, strength=0.4)
    # No generation: the reader's dose; nothing breaks: the generation's dose at any size.
    assert C.generation_dose(_steer_profile(gen=None)) == dict(layer=8, strength=0.8)
    assert C.generation_dose(_steer_profile(gen=(8, 1.6), edge=None)) == dict(layer=8, strength=1.6)
    assert C.generation_dose(dict(reader=None)) is None


def test_steer_dose_is_the_chat_calibration_only():
    # Without a calibration there is no default, whatever the profile measured.
    p = _steer_profile(reader=(8, 0.8), gen=(8, 0.8), edge=0.8)
    assert C.steer_dose(p) is None and C.steer_dose(_steer_profile(gen=None)) is None
    # A chat calibration steered at the reader layer sets it; an older (version 3) calibration, one at
    # another layer or one without a dose sets nothing.
    chat = dict(version=4, layer=8, default_dose=0.3)
    assert C.steer_dose(p, chat) == dict(layer=8, strength=0.3, basis="chat")
    assert C.steer_dose(p, dict(chat, default_dose=0.5)) == dict(layer=8, strength=0.5, basis="chat")
    for other in (dict(x=0.9, steer=dict(layer=8, dose=0.3)), dict(chat, layer=9), dict(chat, default_dose=None),
                  dict(chat, version=3), dict(version=3, layer=8, outcome="scored", default_dose=0.3)):
        assert C.steer_dose(p, other) is None
    assert C.steer_dose(dict(reader=None), chat) is None


def test_attach_generation_records_the_dose():
    p = _steer_profile()
    g = p.pop("generation")
    C.attach_generation(p, dict(g, strength=0.4))
    assert p["generation"]["reader_strength"] == 0.8 and p["generation"]["below_edge"] is True
    C.attach_generation(p, dict(g, strength=0.8))
    assert p["generation"]["below_edge"] is False


def test_an_old_status_block_is_read_by_nothing(home):
    d = write_concept("tiny")
    meta = json.loads((d / "concept.json").read_text())
    meta["status"] = dict(decision="drop", date="2026-10-06", reason="Presence fails. More text.")
    (d / "concept.json").write_text(json.dumps(meta))
    row = next(c for c in C.list_concepts("tiny") if c["name"] == "toy")
    prof = _stored_profile()
    assert "status" not in row and "status" not in C.summarize(prof)


def test_summary_has_what_steer_needs(home):
    _, prof = fake_profile(home)
    s = C.summarize(prof)
    assert s["reader"] == dict(layer=1, strength=0.5, concept=0.8, control=0.1) and s["generation"] == dict(edge=2.0)
    assert C.fluency_rungs(None) is None and C.fluency_rungs(dict(ladder=[generation_rung(0.0), generation_rung(0.5)]))["edge"] is None


def test_summary_carries_the_default_steer_dose(home):
    _, prof = fake_profile(home)
    # Not calibrated: no default, whatever the profile measured (the fake ladder ran at the reader's 0.5).
    s = C.summarize(prof)
    assert s["steer"] is None
    shown = dict(prof, n_pos=3, n_neg=3, position="mean")
    assert "  /steer default: not calibrated (concept calibrate sets it)" in C.summary_lines(shown)
    cal = dict(version=models.CALIBRATION_VERSION, default_dose=0.3, layer=1, good=15, turns=18)
    assert C.summarize(prof, cal)["steer"] == dict(layer=1, strength=0.3, basis="chat")
    assert "  /steer default: L1 @ 0.3 (the most good replies in conversation: 15/18)" in C.summary_lines(shown, cal)


def test_summary_says_the_push_both_ways(home):
    _, prof = fake_profile(home)
    shown = dict(prof, n_pos=3, n_neg=3, position="mean")
    assert C.summarize(prof)["push"] is None and not any("push on held-out" in s for s in C.summary_lines(shown))
    row = lambda base, chosen, concept, control=None: dict(base_chosen=base, chosen=chosen, concept=concept, control_chosen=control)
    shown["learn"] = prof["learn"] = dict(push=dict(dose=0.8, at_dose=row(0.21, 0.64, 5.2, 0.32), away=row(0.21, 0.21, 2.65)))
    assert set(C.summarize(prof)["push"]) == {"toward", "away"}
    assert ("  push on held-out items: Steering toward toy: picked more often, 21% to 64% (a random direction: 32%). "
            "Steering away from toy: picked as often, 21% to 21%, but made more likely (expected less often).") in C.summary_lines(shown)

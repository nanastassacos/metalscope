"""The served model writes a concept's corpus: spec, matched pairs, task items.

Through the server's chat API, one request at a time, thinking off. What the
experiment and the reviews taught:

- The contrast must be a minimal neighbour ("same mood, same decency,
  differs only in X"). Asked for "the opposite pole", the model wrote rage
  as stoic's contrast and wrongdoing as duty's.
- A style concept must be written *in* the style (the words someone says or
  writes), not described ("he kept his voice emotionless").
- Several cues per sentence, the same form on both sides (if one quotes
  speech, so does the other).
- Task items are written from the user's own one-line description, not from
  the spec's contrast, so a misread contrast does not carry into the test.
  The model cannot write balanced items; the GPU step measures each one and
  keeps the useful ones, so this step only over-generates.
- No blind judge: it passed 95% of pairs, including wrong-concept ones.

A corpus the served model cannot write (``read_corpus``) comes from files, held to the same checks.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

from ..client import UNTRACED
from .grade import MAX_TOKENS, thinking_fields, thinks_anyway

N_PAIRS, ASK_PAIRS, ASK_ITEMS = 16, 20, 60

SPEC = """We are building a steering vector for a concept in a language model. Write its specification.

Concept name: {name}
What the user means: {desc}

1. Kind: "style" if the concept is a way of speaking or writing (a tone, a register, a manner of talking: sarcastic, robotic, formal), "content" otherwise (a feeling, an attitude, a texture, a topic, a behaviour).
2. Contrast: the minimal neighbour of the concept. It must be the same in every other respect (same mood, same decency, same competence, same topic) and differ only in the concept itself. Never a crude opposite or a vice. Examples: stoic acceptance vs calmly working to change the situation (not vs rage); acting from duty vs acting for the best outcome, both decent (not vs cheating); fluffy textures vs hard, smooth textures; angry vs a plain, neutral telling of the same event.

Reply with JSON only:
{{"title": "...", "kind": "style or content",
 "description": "two sentences: what the concept is and what it is contrasted with",
 "contrast": "one phrase: the minimal neighbour",
 "words": ["10 single words that carry the concept"],
 "target_words": ["4 words a reply leaning toward the concept would contain"]}}"""

FORM = {
    "style": ("Write each sentence IN the style: the actual words a person says or writes, quoted or as a first-person line. "
              "The positive says it in the {name} manner; the negative says the same thing in the contrast's manner. "
              "Do not describe how anyone speaks."),
    "content": "Each sentence is a short narrative line about a person in a situation.",
}

PAIRS = """Write a contrast corpus for finding the direction of a concept inside a language model.

Concept: {name}. {description}
Negative side (the minimal neighbour): {contrast}.

Write {n} matched pairs. Each pair is a POSITIVE sentence that clearly carries the concept and a NEGATIVE sentence that carries the negative side instead.
{form}
Rules:
- Same skeleton in both: same situation, same people, same facts, same tense. Change only what carries the concept.
- Carry the concept with two or three cues spread through the sentence, not by one swapped word.
- The negative must not carry the concept at all, not even faintly or ironically, and must not be a crude opposite: it differs from the positive only in the concept.
- Same form in both: if one sentence quotes speech, so does the other, at the same place. Tell the negative in the same natural voice, not as a flat report.
- Same length in a pair, within 15% in words. Each sentence 14 to 28 words.
- Never use the word "{name}" or its forms.
- Every pair in a different setting: home, work, travel, food, weather, family, school, shopping, health, sport, nature, objects, and so on.

Reply with JSON only: {{"pairs": [{{"pos": "...", "neg": "..."}}]}}"""

TASK = """Write {n} two-option questions that test whether an assistant leans toward "{name}", as the user described it: {desc}

Each item: a short everyday situation and a question (which reply, which advice, which description, which choice fits best), and 2 options. The target option leans toward "{name}"; the other option does not.
Rules:
- Both options are sensible and match each other in mood, politeness, morality and length. The only difference is the lean toward "{name}".
- The target must not simply be the correct, kinder or safer answer.
- Never use the word "{name}".
- Vary the situations widely.

Reply with JSON only: {{"items": [{{"prompt": "...", "options": ["...", "..."], "target": "..."}}]}}"""


class Writer:
    """Chat requests to the served ``model`` through ``server`` (a client.Client)."""

    def __init__(self, server, model: str, log: Callable[[str], None] = print, timeout: float = 1800):
        self.server, self.model, self.log, self.timeout = server, model, log, timeout
        self.calls: list[dict] = []

    def chat(self, prompt: str, max_tokens: int = 4000, temperature: float = 0.7) -> str:
        """Thinking off; a model that thinks anyway (grade.thinks_anyway) gets the thinking budget: Inkling 2 spent
        all 4,000 tokens thinking over the pairs, and wrote them in 11,000."""
        if thinks_anyway(self.model):
            max_tokens = max(max_tokens, MAX_TOKENS[True])
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": temperature, **thinking_fields(self.model, False)}
        t = time.time()
        d = self.server.post("/v1/chat/completions", body, timeout=self.timeout, headers=UNTRACED)
        text = d["choices"][0]["message"].get("content") or ""
        self.calls.append(dict(prompt=prompt, reply=text, seconds=round(time.time() - t, 1)))
        return text

    def ask_json(self, prompt: str, key: str, tries: int = 3, **kw) -> dict:
        """The reply's object holding ``key`` (``parse_json``), asked again while there is none."""
        last = None
        for _ in range(tries):
            try:
                return parse_json(self.chat(prompt, **kw), key, prompt)
            except ValueError as exc:
                last = exc
        raise RuntimeError(f"the model gave no JSON after {tries} tries: {last}")


def _strings(v) -> list[str]:
    """Every string value in a JSON value (not its keys), stripped."""
    if isinstance(v, str):
        return [v.strip()]
    return [s for x in (v.values() if isinstance(v, dict) else v if isinstance(v, list) else ()) for s in _strings(x)]


def parse_json(text: str, key: str, prompt: str = "") -> dict:
    """The first JSON object in a reply that holds ``key`` and is not ``prompt``'s reply format echoed (every string in
    it "..." or one of the format's own example strings), the reply's values searched outermost first
    (``grade._json_values``: code fences, chatter and stray brackets around it cost nothing). ValueError without one
    (a bare list of items, a reply cut off before its object closed), and ``ask_json`` asks again."""
    from .grade import _json_values
    examples = {"...", *(s for v in _json_values(prompt) for s in _strings(v))}

    def found(v):
        strings = _strings(v)
        if isinstance(v, dict) and key in v and not (strings and all(s in examples for s in strings)):
            return v
        inner = v.values() if isinstance(v, dict) else v if isinstance(v, list) else ()
        return next((f for f in map(found, inner) if f is not None), None)
    got = next((f for f in map(found, _json_values(text)) if f is not None), None)
    if got is None:
        raise ValueError(f"no JSON object with {key!r} in the reply")
    return got


def words(s: str) -> list[str]:
    return re.findall(r"[a-z']+", s.lower())


def name_stem(name: str) -> str:
    """What counts as the concept's own name in a sentence: its first four
    letters for names of five letters or more (angry: anger, angrily;
    sarcastic: sarcasm), the whole name for shorter ones (sad)."""
    letters = re.sub(r"[^a-z]", "", name.lower())
    return letters[:4] if len(letters) >= 5 else letters


def pair_problem(name: str, pos: str, neg: str) -> str:
    """Why a pair fails the mechanical checks ("" if it passes): empty, lengths
    more than 15% apart, the concept's own name (``name_stem``) used."""
    wp, wn = words(pos), words(neg)
    if not wp or not wn:
        return "empty"
    if abs(len(wp) - len(wn)) / max(len(wp), len(wn)) > 0.15:
        return "length"
    stem = name_stem(name)
    if stem and any(w.startswith(stem) for w in wp + wn):
        return "name"
    if pos.strip() == neg.strip():
        return "identical"
    return ""


def valid_items(items, name: str) -> list[dict]:
    """Items with a prompt, 2 to 4 distinct options, a target among them, and the name kept out."""
    out, seen = [], set()
    stem = name_stem(name)
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        prompt, opts, target = it.get("prompt"), it.get("options"), it.get("target")
        if not isinstance(prompt, str) or not isinstance(opts, list) or not (2 <= len(opts) <= 4):
            continue
        if not all(isinstance(o, str) and o.strip() for o in opts) or len(set(opts)) != len(opts) or target not in opts:
            continue
        text = " ".join([prompt] + opts)
        if stem and any(w.startswith(stem) for w in words(text)):
            continue
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append(dict(prompt=prompt.strip(), options=[o.strip() for o in opts], target=target.strip()))
    return out


def write(writer: Writer, name: str, desc: str, on_stage: Optional[Callable[[str], None]] = None) -> dict:
    """The concept's files as a dict: ``concept`` (concept.json), ``pos``,
    ``neg`` (lists of lines), ``task`` (task.json), ``writing`` (writing.json:
    the writer, the date, the request and every instruction it was sent, in
    order, without its replies) and ``log`` (counts and every request with its
    reply). Raises RuntimeError when too few pairs or items survive."""
    stage = on_stage or (lambda s: None)
    t0 = time.time()
    stage("writing the spec")
    spec = writer.ask_json(SPEC.format(name=name, desc=desc), key="title", temperature=0.3)
    kind = "style" if str(spec.get("kind", "")).lower().startswith("style") else "content"
    description, contrast = str(spec.get("description", desc)), str(spec.get("contrast", "a plain, neutral telling of the same thing"))
    stage("writing the pairs")
    pairs, problems = [], {}
    for _ in range(2):
        got = writer.ask_json(PAIRS.format(name=name, description=description, contrast=contrast, n=ASK_PAIRS,
                                           form=FORM[kind].format(name=name)), key="pairs")
        for p in got.get("pairs", []) if isinstance(got, dict) else []:
            if not isinstance(p, dict) or not isinstance(p.get("pos"), str) or not isinstance(p.get("neg"), str):
                continue
            why = pair_problem(name, p["pos"], p["neg"])
            if why:
                problems[why] = problems.get(why, 0) + 1
            elif all(p["pos"] != q["pos"] for q in pairs):
                # pos.txt / neg.txt are one sentence per line, and lines starting with # are comments
                clean = lambda s: " ".join(s.split()).lstrip("#").strip()
                pairs.append(dict(pos=clean(p["pos"]), neg=clean(p["neg"])))
        if len(pairs) >= N_PAIRS:
            break
    if len(pairs) < 8:
        raise RuntimeError(f"only {len(pairs)} pairs passed the checks ({problems})")
    pairs = pairs[:N_PAIRS]
    stage("writing the task items")
    items: list[dict] = []
    for _ in range(3):
        got = writer.ask_json(TASK.format(name=name, desc=desc, n=min(30, ASK_ITEMS - len(items))), key="items")
        items += [it for it in valid_items(got.get("items", []) if isinstance(got, dict) else [], name)
                  if all(it["prompt"] != j["prompt"] for j in items)]
        if len(items) >= ASK_ITEMS:
            break
    if len(items) < 8:
        raise RuntimeError(f"only {len(items)} task items were usable")
    learned = dict(writer=writer.model, request=desc, date=time.strftime("%Y-%m-%d"))
    concept = dict(title=str(spec.get("title") or name), description=f"{description} Contrast: {contrast}.",
                   inspiration=f"written by {writer.model} from the request: {desc}",
                   words=[w for w in spec.get("words", []) if isinstance(w, str)][:12],
                   target_words=[w for w in spec.get("target_words", []) if isinstance(w, str)][:6],
                   kind=kind, contrast=contrast, learned=learned)
    log = dict(spec=spec, kind=kind, pairs_kept=len(pairs), pair_problems=problems, items=len(items),
               requests=len(writer.calls), seconds=round(time.time() - t0), calls=writer.calls)
    return dict(concept=concept, pos=[p["pos"] for p in pairs], neg=[p["neg"] for p in pairs],
                task=dict(instruction="Reply with the letter only.", items=items),
                writing=dict(writer=writer.model, date=learned["date"], request=desc,
                             prompts=[c["prompt"] for c in writer.calls]), log=log)


PROBLEM_TEXT = {"empty": "a side is empty", "length": "lengths more than 15% apart in words",
                "name": "uses the concept's own name", "identical": "both sides are the same"}


def _read_dir(d: Path, comments: bool) -> tuple[dict, list[str], list[str], dict]:
    """DIR's concept.json, pos.txt and neg.txt lines and task.json, checked for their structure: the files there and
    readable, the pairs matched line for line, at least 8 of them. ``comments``: lines starting with # are skipped
    (as ``concepts.corpus.load_dir`` reads them); else such a line is an error. ValueError naming what fails."""
    from ..concepts.corpus import TEXTS_NEEDED
    missing = [f for f in TEXTS_NEEDED if not (d / f).is_file()]
    if missing:
        raise ValueError(f"{d}: missing {', '.join(missing)}")

    def load(f):
        try:
            return json.loads((d / f).read_text())
        except ValueError as exc:
            raise ValueError(f"{d / f}: not JSON ({exc})") from None
    concept = load("concept.json")
    if not isinstance(concept, dict):
        raise ValueError(f"{d / 'concept.json'}: not a JSON object")
    sides = {}
    for side in ("pos", "neg"):
        sides[side] = [l.strip() for l in (d / f"{side}.txt").read_text().splitlines()
                       if l.strip() and not (comments and l.startswith("#"))]
        bad = next((i for i, l in enumerate(sides[side], 1) if l.startswith("#")), None)
        if bad:
            raise ValueError(f"{side}.txt sentence {bad} starts with #, which marks a comment: one sentence per line")
    pos, neg = sides["pos"], sides["neg"]
    if len(pos) != len(neg):
        raise ValueError(f"pos.txt has {len(pos)} sentences and neg.txt {len(neg)}: they are matched pairs, line i of each")
    if len(pos) < 8:
        raise ValueError(f"only {len(pos)} pairs: at least 8 are needed")
    task = load("task.json")
    return concept, pos, neg, task if isinstance(task, dict) else {}


def _task(task: dict, items: list[dict]) -> dict:
    return dict(instruction=task.get("instruction") or "Reply with the letter only.", items=items,
                **({"letters": task["letters"]} if task.get("letters") else {}))


def _read_texts(d: Path, name: str) -> tuple[dict, list[str], list[str], dict, list[str]]:
    """Texts that already exist (the model's own, shipped, pulled or supplied with --corpus), one rule for all: their
    structure must hold (``_read_dir``: the files, pairs matched line for line, at least 8; at least 8 task items with a prompt,
    2 to 4 distinct options and the target among them), else ValueError naming what fails; the writer's pair rules
    (``pair_problem``) are only reported, as warnings, since they are for a model's new writing and the hand-written
    sets break them and work (fluffy uses its own name, terse's pairs differ in length). Every pair is used.
    ``(concept, pos, neg, task, warnings)``."""
    concept, pos, neg, task = _read_dir(d, comments=True)
    if not str(concept.get("description") or concept.get("title") or "").strip():
        raise ValueError(f"{d / 'concept.json'}: a description is needed (what the concept means)")
    items = valid_items(task.get("items"), "")
    if len(items) < 8:
        raise ValueError(f"{d / 'task.json'}: only {len(items)} task items are usable (a prompt, 2 to 4 distinct "
                         "options, the target among them): at least 8 are needed")
    warnings = []
    for i, (a, b) in enumerate(zip(pos, neg), 1):
        why = pair_problem(name, a, b)
        if why:
            warnings.append(f"pair {i}: {PROBLEM_TEXT.get(why, why)} ({len(words(a))} / {len(words(b))} words)")
    return concept, pos, neg, _task(task, items), warnings


def read_corpus(path, name: str) -> dict:
    """Texts written elsewhere (a person, another model) in DIR, for ``--corpus``, in ``write``'s shape (its
    ``writing``: DIR's writing.json, None without one; ``concept["learned"]`` DIR's own, with where it came from):
    held to the one rule for existing texts (``_read_texts``:
    structure checked, pair-rule breaks in ``log["warnings"]``). The learn job copies them into the model's folder."""
    d = Path(path).expanduser().resolve()
    concept, pos, neg, task, warnings = _read_texts(d, name)
    learned = dict(concept.get("learned") or {})       # no writer recorded: none made up
    learned.update(corpus=str(d), date=learned.get("date") or time.strftime("%Y-%m-%d"))
    concept = {k: v for k, v in concept.items() if k != "status"}       # the label is this run's
    log = dict(supplied=str(d), pairs_kept=len(pos), items=len(task["items"]), warnings=warnings, requests=0, seconds=0)
    return dict(concept=dict(concept, learned=learned), pos=pos, neg=neg, task=task,
                writing=_json_or_none(d / "writing.json"), log=log)


def read_texts(d: Path, name: str) -> dict:
    """Texts that exist (the model's own, or the shipped ones) in ``d``, for learning, in ``write``'s shape plus
    ``path``: held to the one rule for existing texts (``_read_texts``), and read as the GPU child reads them
    (``concepts.corpus.load_dir``, which uses every task item as it stands), so a text the child would fail on fails
    here, before the model is unloaded. ValueError when they do not hold together."""
    from ..concepts.corpus import load_dir
    concept, pos, neg, task, warnings = _read_texts(d, name)
    load_dir(d, name)
    log = dict(texts=str(d), pairs_kept=len(pos), items=len(task["items"]), warnings=warnings, requests=0, seconds=0)
    return dict(concept=concept, pos=pos, neg=neg, task=task, writing=_json_or_none(d / "writing.json"),
                path=str(d), log=log)


def _json_or_none(p: Path):
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except ValueError:
        return None


def description_of(concept: dict) -> str:
    """What a concept means, from its concept.json: the user's own request when a model wrote it from one (or a
    supplied corpus records one), else its description (hand-written texts: their request was the writer's task)."""
    from ..concepts.corpus import by_hand
    learned = concept.get("learned") or {}
    return (None if by_hand(learned) else learned.get("request")) or concept.get("description") or ""

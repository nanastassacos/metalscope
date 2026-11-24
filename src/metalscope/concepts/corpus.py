"""The concept corpus on disk and the forced-choice task.

A concept's texts are a directory with ``concept.json`` (title, words, target words, ``learned``: who wrote the
texts, from what request, when), ``pos.txt`` / ``neg.txt`` (the matched contrast texts), optionally ``task.json``
(the forced-choice items) and ``writing.json`` (the instructions the writer model was sent). A model's are in its
own concept folder (``models.concept_dir``), beside the direction learned from them: the only texts it learns from
(``load_concept``). Shipped concepts live under ``metalscope/data/concepts/``, read only, and are copied into a
model's folder when it learns one. A user concept's definition (``{name, description}``) is
``~/.metalscope/concepts/<name>/concept.json`` (``define``), so another model can write its own texts from it. The
task half renders an item as the model sees it: options rotated, lettered, chat-templated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import config, models

DATA_DIR = Path(__file__).parent.parent / "data" / "concepts"


# ---------------------------------------------------------------- concepts on disk

@dataclass
class Item:
    prompt: str
    options: list[str]
    target: str

    @property
    def target_index(self) -> int:
        return self.options.index(self.target)


@dataclass
class Task:
    instruction: str
    items: list[Item]
    letters: Optional[list[str]] = None     # override "A", "B", ... ("letters" in task.json); read through option_letters


@dataclass
class Concept:
    name: str
    path: Path
    pos: list[str]
    neg: list[str]
    words: list[str]                 # concept vocabulary the lens is asked about
    target_words: list[str]          # what the task's target should read as, e.g. ["cat"]
    task: Optional[Task]
    meta: dict = field(default_factory=dict)


TEXTS_NEEDED = ("concept.json", "pos.txt", "neg.txt", "task.json")     # a whole set of texts (writing.json optional)
TEXTS = (*TEXTS_NEEDED, "writing.json")                                 # every file of a concept's texts


def texts_dir(model: str, name: str) -> Optional[Path]:
    """The model's own texts of ``name`` (its concept folder, when it holds them), else None."""
    d = models.concept_dir(model, name)
    return d if (d / "concept.json").is_file() and (d / "pos.txt").is_file() else None


def shipped_dir(name: str) -> Optional[Path]:
    """A shipped concept's texts (data/concepts/), else None."""
    d = DATA_DIR / name
    return d if (d / "concept.json").is_file() else None


def definition(name: str) -> Optional[str]:
    """A user concept's stored description (``define``), else None."""
    meta = config.read_json(models.definition_path(name))
    return str(meta.get("description") or "") or None if isinstance(meta, dict) else None


def define(name: str, description: str) -> Optional[Path]:
    """Store ``name``'s definition, once: None when it has one already, is shipped, or no description is given."""
    if shipped_dir(name) is not None or not description:
        return None
    p = models.definition_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, "x") as f:
            f.write(json.dumps(dict(name=name, description=description), indent=1) + "\n")
    except FileExistsError:
        return None
    return p


def list_concepts(model: Optional[str] = None) -> list[dict]:
    """The concept library: the shipped concepts, the defined ones and (with ``model``) the model's own folders that
    hold texts or a profile. Each from the model's own texts when it has them, else the shipped ones, else its
    definition: ``{name, title, description, inspiration, task, path, texts}`` (``texts``: ``texts_line``; none on
    this model when it has a profile there but no texts)."""
    names = {p.parent.name for p in DATA_DIR.glob("*/concept.json")}
    names |= {p.parent.name for p in models.concepts_dir().glob("*/concept.json")}
    if model:
        names |= {p.name for p in (models.model_dir(model) / "concepts").glob("[!.]*")
                  if texts_dir(model, p.name) or (p / "profile.json").is_file()}
    out = []
    for n in sorted(names):
        defined, mine = models.definition_path(n), model and texts_dir(model, n)
        d = mine or shipped_dir(n) or (defined.parent if defined.is_file() else models.concept_dir(model, n))
        meta = config.read_json(d / "concept.json", {})
        meta = meta if isinstance(meta, dict) else {}
        learned = model and not mine and models.profile_path(model, n).is_file()     # the model's state, not others' texts
        texts = texts_line(d) if (d / "concept.json").is_file() and not learned else "texts: none on this model"
        out.append(dict(name=n, title=meta.get("title", n), description=meta.get("description", ""),
                        inspiration=meta.get("inspiration", ""), task=(d / "task.json").exists(), path=str(d), texts=texts))
    return out


BY_HAND = " (by hand)"               # a writer's suffix in ``learned`` for texts written by hand (the shipped ones)


def by_hand(learned: dict) -> bool:
    """Whether ``learned`` (concept.json's) says the texts were written by hand: its ``request`` is then what the
    writer was asked, not the user's own words for the concept."""
    return str((learned or {}).get("writer") or "").endswith(BY_HAND)


def texts_line(d: Path, request: bool = True) -> str:
    """How the concept in ``d`` got its texts, in one line: 'texts: hand-written by Claude Fable 5.1 (2026-10-05)'
    when its writer wrote them by hand (``BY_HAND``), else 'texts: written by Qwen3.6 from "dry humour" (2026-10-27)'
    (``request`` False: without the request), with "(instructions not kept)" when writing.json holds none of the
    writer's instructions. A folder without texts (a definition) says each model writes its own."""
    if not (d / "pos.txt").is_file():
        return "texts: each model writes its own"
    meta, writing = config.read_json(d / "concept.json"), config.read_json(d / "writing.json")
    learned = meta.get("learned") if isinstance(meta, dict) else None
    if not isinstance(learned, dict) or not learned.get("writer"):
        return "texts: no record of who wrote them"
    writer, date = str(learned["writer"]), f" ({learned['date']})" if learned.get("date") else ""
    if writer.endswith(BY_HAND):
        return f"texts: hand-written by {writer.removesuffix(BY_HAND)}{date}"
    asked = f' from "{learned["request"]}"' if request and learned.get("request") else ""
    kept = "" if isinstance(writing, dict) and writing.get("prompts") else " (instructions not kept)"
    return f"texts: written by {writer}{asked}{date}{kept}"


def _lines(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def load_task(path: Path) -> Task:
    """A task.json as the learn job's GPU child reads it: every item used, so one that is not usable is a ValueError
    naming it (``learn.writer.read_texts`` runs this before anything is unloaded)."""
    data = json.loads(path.read_text())
    items = []
    for i, it in enumerate(data.get("items") or [] if isinstance(data, dict) else [], 1):
        try:
            item = Item(it["prompt"], list(it["options"]), it["target"])
        except (KeyError, TypeError):
            raise ValueError(f"{path}: item {i} needs a prompt, options and a target") from None
        if item.target not in item.options:
            raise ValueError(f"{path}: item {i}'s target {item.target!r} is not among its options {item.options}: "
                             "fix or remove that item")
        items.append(item)
    if not items:
        raise ValueError(f"{path}: no items")
    return Task(data.get("instruction", "Reply with the letter only."), items, data.get("letters"))


def load_concept(model: str, name: str) -> Concept:
    """The model's own texts of ``name`` (``texts_dir``), the only ones it learns from."""
    d = texts_dir(model, name)
    if d is None:
        raise ValueError(f"no texts of {name} on {models.checkpoint_of(model)} ({models.concept_dir(model, name)})")
    return load_dir(d, name)


def load_dir(d: Path, name: str) -> Concept:
    meta = json.loads((d / "concept.json").read_text())
    pos, neg = _lines(d / "pos.txt"), _lines(d / "neg.txt")
    if len(pos) < 4 or len(pos) != len(neg):
        raise ValueError(f"concept {name}: pos.txt and neg.txt are matched pairs, line i of each, at least 4 "
                         f"(have {len(pos)} / {len(neg)})")
    task = load_task(d / "task.json") if (d / "task.json").exists() else None
    return Concept(name, d, pos, neg, list(meta.get("words", [])), list(meta.get("target_words", [])), task, meta)


# ---------------------------------------------------------------- the task

def option_letters(task: Task, n: int) -> list[str]:
    """The letters of ``n`` options in slot order: the task's own or A, B, C,
    ... The one source for the prompt, the scored letter tokens, the target
    letter and the letter a reply opens with."""
    letters = task.letters or [chr(ord("A") + i) for i in range(n)]
    if len(letters) < n:
        raise ValueError(f"task has {n} options but only {len(letters)} letters")
    return list(letters[:n])


def letter_ids(tok, task: Task, n: int) -> list[int]:
    ids = []
    for s in option_letters(task, n):
        t = tok.encode(s)
        if len(t) != 1:
            raise ValueError(f"option letter {s!r} is not a single token ({t})")
        ids.append(int(t[0]))
    return ids


def render_item(tok, task: Task, item: Item, shift: int) -> tuple[list[int], list[int], int]:
    """The chat-templated prompt ids for ``item`` (thinking off, so the next
    token is the answer) with its options rotated by ``shift`` (the target
    lands in slot (target_index + shift) mod n), the letter token ids in slot
    order and the target's slot. ``tok`` is an engine's tokenizer (engines)."""
    n = len(item.options)
    return tok.chat(render_body(task, item, shift), thinking=False), letter_ids(tok, task, n), (item.target_index + shift) % n


def render_body(task: Task, item: Item, shift: int) -> str:
    """The user turn as text: prompt, lettered options rotated by ``shift``, instruction."""
    n = len(item.options)
    opts = [item.options[(i - shift) % n] for i in range(n)]
    letters = option_letters(task, n)
    return item.prompt + "\n\n" + "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(opts)) + "\n\n" + task.instruction


def prompts_for(tok, task: Task, n_items: Optional[int] = None) -> list[dict]:
    """Every (item, rotation) as rendered ids with its letters and target slot."""
    out = []
    for i, item in enumerate(task.items[:n_items] if n_items else task.items):
        for shift in range(len(item.options)):
            ids, letters, slot = render_item(tok, task, item, shift)
            out.append(dict(item=i, shift=shift, ids=ids, letters=letters, slot=slot))
    return out

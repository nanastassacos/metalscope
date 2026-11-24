"""The windows a Jacobian lens is fitted on (``/jlens learn``; learn/lens_fit_gpu.py fits them).

A source is a list of documents:

  a directory   every text file under it (``TEXT_SUFFIXES``; hidden paths, node_modules, binaries and files over
                ``MAX_FILE_BYTES`` skipped) is one document;
  one .txt      its paragraphs (split on blank lines), one document each;
  the default   (no path) wikitext paragraphs (``wikitext_paragraphs``: downloaded on first use into
                ``<home>/corpus/``), and the metalscope package's own source as code (``package_code``: a snapshot taken
                on first use, so the corpus stays the same as the code changes). Never the private ``mixed_v1.json``.

About 10% of the documents (at least 2, picked by the seed) are held out: never fitted, scored after every window.

A window is a jfit prompt: ``{"ids", "window": W}`` for a deep one (the prefix runs once without gradients, so the
lens sees positions thousands of tokens in for the price of W), or ``{"ids"}`` alone for a shallow one (the first W
tokens; jfit skips the first 16 positions, the attention sinks). A document shorter than W gives itself whole; a
longer one a window at a random depth up to ``max_depth``, a new depth each round. A .txt gives shallow windows only.
The default mixes four kinds in a fixed order: a few shallow
wikitext windows, then chat, wiki, chat, wiki, code over and over, where

  wiki   consecutive paragraphs read as one text, the window ending a few thousand tokens in;
  chat   the same paragraphs as alternating user and assistant turns through the model's chat template, the window
         ending on an assistant turn a few thousand tokens deep (Pi's replies sit 2-6k tokens into a chat);
  code   a package source file, the window at a random point in it.

Window ``i`` is built from its own seed (``Windows[i]``), so any prefix of the list is the same however long it is
made: a fit that stops at a deadline resumes on the same windows. ``corpus_id`` hashes the documents and the
settings (not the tokenizer: the model is the other half of a fit's identity).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from .. import config
from . import SKIP_FIRST_N_POSITIONS as SKIP     # a shallow window's first positions are not fitted

VERSION = 1                       # bump when the window recipe changes: old fits then start fresh
WINDOW = 128
MAX_DEPTH = 8192                  # prefix tokens: their KV cache is replicated once per output dim of a pass
MIN_TOKENS = SKIP + 32            # a shorter document has too few positions to fit
HELDOUT_SHARE, MIN_HELDOUT, MAX_HELDOUT_WINDOWS = 0.1, 2, 12
MAX_WINDOWS = 2000
CHAT_DEPTHS = (1024, 2048, 3072, 4096, 6144)   # where Pi's replies sit (the lens corpus research)
CODE_SUFFIXES = {".py", ".ts", ".js", ".rs", ".go", ".c", ".h"}
TEXT_SUFFIXES = CODE_SUFFIXES | {".txt", ".md", ".json", ".html", ".tex", ".rst"}
MAX_FILE_BYTES = 1_000_000
MAX_DOCS = 5000
SKIP_DIRS = {"node_modules", "__pycache__"}

# The default text: wikitext-2 (raw) training paragraphs over 600 characters, cut at 2000
# (<home>/corpus/wikitext_paragraphs.json, 406 paragraphs). Hugging Face's dataset viewer serves the
# rows as JSON, 100 at a time, so plain urllib does.
WIKITEXT_FILE = "wikitext_paragraphs.json"
WIKITEXT_ROWS = ("https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext"
                 "&config=wikitext-2-raw-v1&split=train&offset={offset}&length=100")
N_PARAGRAPHS, MIN_CHARS, MAX_CHARS = 406, 600, 2000
CODE_FILE = "lens_code.json"
N_SHALLOW_FIRST = 4
PATTERN = ("chat", "wiki", "chat", "wiki", "code")   # after N_SHALLOW_FIRST shallow ones, over and over


@dataclass
class Doc:
    name: str
    text: str
    kind: str = "text"            # text (a user's file or paragraph), wiki, code


def corpus_dir() -> Path:
    return config.home() / "corpus"


# ---------------------------------------------------------------- reading a source

def _read_text(path: Path) -> Optional[str]:
    """The file's text, None for one over MAX_FILE_BYTES, unreadable, binary or not UTF-8."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def read_dir(root: Path) -> list[Doc]:
    """Every text file under ``root``, one document each, in path order."""
    docs = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part.startswith(".") or part in SKIP_DIRS for part in rel.parts) or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if not p.is_file():
            continue
        text = _read_text(p)
        if text and text.strip():
            docs.append(Doc(str(rel), text, "code" if p.suffix.lower() in CODE_SUFFIXES else "text"))
        if len(docs) >= MAX_DOCS:
            break
    return docs


def read_txt(path: Path) -> list[Doc]:
    """One .txt's paragraphs (blank lines between them), one document each."""
    if path.stat().st_size > 50 * MAX_FILE_BYTES:
        raise ValueError(f"{path} is over {50 * MAX_FILE_BYTES // 1_000_000} MB: split it, or give a directory")
    text = path.read_text(errors="replace")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [Doc(f"{path.name}#{i + 1}", p) for i, p in enumerate(paras)]


def download_wikitext(dest: Path, n: int = N_PARAGRAPHS, fetch: Callable[[str], bytes] = None,
                      log: Callable[[str], None] = lambda _m: None) -> list[str]:
    """The default paragraphs, fetched page by page until ``n`` are kept, written to ``dest``."""
    fetch = fetch or (lambda url: urllib.request.urlopen(url, timeout=60).read())

    def page(url: str) -> bytes:
        for attempt in range(5):              # the viewer answers 502 now and then
            try:
                return fetch(url)
            except OSError as exc:
                if attempt == 4:
                    raise RuntimeError(f"could not download the default corpus ({exc}): give a directory or a .txt") from None
                time.sleep(2 * (attempt + 1))
    out, offset = [], 0
    log(f"downloading wikitext paragraphs into {dest}")
    while len(out) < n:
        rows = json.loads(page(WIKITEXT_ROWS.format(offset=offset))).get("rows") or []
        if not rows:
            break
        for r in rows:
            t = (r.get("row") or {}).get("text", "").strip()
            if len(t) > MIN_CHARS and not t.startswith("="):
                out.append(t[:MAX_CHARS])
        offset += len(rows)
    if not out:
        raise RuntimeError("no wikitext paragraphs downloaded")
    dest.parent.mkdir(parents=True, exist_ok=True)
    config.atomic_write(dest, json.dumps(out[:n]))
    return out[:n]


def wikitext_paragraphs(fetch=None, log=lambda _m: None) -> list[str]:
    """<home>/corpus/wikitext_paragraphs.json, downloaded the first time."""
    p = corpus_dir() / WIKITEXT_FILE
    if p.exists():
        return json.loads(p.read_text())
    return download_wikitext(p, fetch=fetch, log=log)


def package_code() -> list[Doc]:
    """The metalscope package's Python source, as it was when first asked for (<home>/corpus/lens_code.json)."""
    p = corpus_dir() / CODE_FILE
    if p.exists():
        files = json.loads(p.read_text())["files"]
    else:
        pkg = Path(__file__).resolve().parents[1]               # metalscope/, not jlens/
        files = {str(f.relative_to(pkg.parent)): f.read_text() for f in sorted(pkg.rglob("*.py")) if "__pycache__" not in f.parts}
        p.parent.mkdir(parents=True, exist_ok=True)
        config.atomic_write(p, json.dumps(dict(files=files)))
    return [Doc(name, text, "code") for name, text in files.items() if text.strip()]


def documents(source: Optional[str], fetch=None, log=lambda _m: None) -> tuple[str, list[Doc]]:
    """``(kind, documents)``: "default", "dir" or "txt". ValueError for a path that is neither."""
    if not source:
        return "default", [Doc(f"wikitext#{i + 1}", t, "wiki") for i, t in enumerate(wikitext_paragraphs(fetch, log))] + package_code()
    p = Path(source).expanduser()
    if p.is_dir():
        docs = read_dir(p)
        if not docs:
            raise ValueError(f"no text files under {p} ({', '.join(sorted(TEXT_SUFFIXES))})")
        return "dir", docs
    if p.is_file() and p.suffix.lower() == ".txt":
        return "txt", read_txt(p)
    if p.exists():
        raise ValueError(f"{p}: a directory of text files or one .txt, not a {p.suffix or 'file without a suffix'}")
    raise ValueError(f"{p} does not exist")


def corpus_id(kind: str, docs: list[Doc], window: int, max_depth: int, seed: int) -> str:
    h = hashlib.sha256(json.dumps([VERSION, kind, window, max_depth, seed]).encode())
    for d in docs:
        h.update(d.name.encode() + b"\0" + hashlib.sha256(d.text.encode()).digest())
    return h.hexdigest()[:12]


def held_out(n: int, seed: int) -> set[int]:
    """Which of ``n`` documents are held out: about 10%, at least 2, picked by the seed."""
    k = min(n, max(MIN_HELDOUT, round(HELDOUT_SHARE * n)))
    return set(random.Random(f"{seed}:heldout").sample(range(n), k))


# ---------------------------------------------------------------- windows

def _ids(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def _at(ids: list[int], depth: int, window: int, kind: str, name: str) -> dict:
    """The window ``depth`` tokens in: a deep one (the prefix runs without gradients), or at depth 0 a shallow one."""
    if depth <= 0:
        return dict(ids=ids[:window], kind=kind, depth=0, doc=name)
    return dict(ids=ids[:depth + window], window=window, kind=kind, depth=depth, doc=name)


def _deep_or_shallow(ids: list[int], rng: random.Random, window: int, max_depth: int, kind: str, name: str,
                     shallow_only: bool = False) -> dict:
    if shallow_only or len(ids) <= window + SKIP:
        return _at(ids, 0, window, kind, name)
    return _at(ids, rng.randint(0, min(len(ids) - window, max_depth)), window, kind, name)


def render_chat(tok, turns: list[str]) -> Optional[list[int]]:
    """Paragraphs as alternating user and assistant turns, through the tokenizer's chat template (None without one)."""
    msgs = [dict(role="user" if i % 2 == 0 else "assistant", content=t) for i, t in enumerate(turns)]
    try:
        out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    except Exception:                              # no template (a base model), or one that refuses these turns
        return None
    return list(out["input_ids"] if hasattr(out, "keys") else out)


class Windows(Sequence):
    """The fit windows, built on demand: ``self[i]`` from its own seed (see the module docstring)."""

    def __init__(self, corpus: "Corpus", n: int):
        self.c, self.n = corpus, n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self.n))]
        if not 0 <= i < self.n:
            raise IndexError(i)
        return self.c.window(i)


class Corpus:
    """A source's documents split into fit and held-out ones, and the windows over them (``build``)."""

    def __init__(self, kind: str, docs: list[Doc], tok, *, source: Optional[str] = None, seed: int = 0,
                 window: int = WINDOW, max_depth: int = MAX_DEPTH):
        self.kind, self.source, self.tok, self.seed = kind, source, tok, seed
        self.window_len, self.max_depth = window, max_depth
        self.shallow_only = kind == "txt"
        self.id = corpus_id(kind, docs, window, max_depth, seed)
        self._toks: dict[int, list[int]] = {}
        if kind == "default":                      # held out per part: paragraphs and code files each
            wiki = [i for i, d in enumerate(docs) if d.kind == "wiki"]
            code = [i for i, d in enumerate(docs) if d.kind == "code"]
            held = {wiki[j] for j in held_out(len(wiki), seed)} | {code[j] for j in held_out(len(code), seed + 1)}
        else:
            held = held_out(len(docs), seed)
        self.docs = docs
        self.train = [i for i in range(len(docs)) if i not in held]
        self.held = sorted(held)
        if len(self.train) < 1:
            raise ValueError(f"{len(docs)} document(s): too few to hold {len(self.held)} out and fit on the rest")
        self.description = self._describe()

    def _describe(self) -> str:
        n = len(self.docs)
        if self.kind == "default":
            wiki = sum(d.kind == "wiki" for d in self.docs)
            return (f"the default corpus: {wiki} wikitext paragraphs (plain, and as chat turns up to "
                    f"{min(max(CHAT_DEPTHS), self.max_depth)} tokens deep) and {n - wiki} metalscope source files")
        if self.kind == "txt":
            return f"{Path(self.source).name}: {n} paragraphs, shallow windows only (the first {self.window_len} tokens of each)"
        return f"{self.source}: {n} files, windows up to {self.max_depth} tokens deep"

    def tokens(self, i: int) -> list[int]:
        if i not in self._toks:
            self._toks[i] = _ids(self.tok, self.docs[i].text)
        return self._toks[i]

    def _usable(self, idx: list[int]) -> list[int]:
        return [i for i in idx if len(self.tokens(i)) >= MIN_TOKENS]

    def _shallow(self, j: int) -> dict:
        return _at(self.tokens(j), 0, self.window_len, "shallow", self.docs[j].name)

    def _chat_depth(self, rng: random.Random) -> int:
        return rng.choice([d for d in CHAT_DEPTHS if d <= self.max_depth] or [self.max_depth])

    # -- the default's kinds
    def _wiki_run(self, idx: list[int], rng: random.Random, depth: int, chat: bool) -> Optional[dict]:
        """Consecutive paragraphs (in corpus order, from ``idx``) ending at a random one, back to ``depth`` tokens."""
        paras = [i for i in idx if self.docs[i].kind == "wiki"]
        if not paras:
            return None
        end = rng.randrange(len(paras))
        turns, total, j = [], 0, end
        while j >= 0 and total < depth + self.window_len:
            turns.insert(0, self.docs[paras[j]].text)
            total += len(self.tokens(paras[j])) + 8
            j -= 1
        if chat:
            if len(turns) % 2:                     # start on a user turn so the last is the assistant's
                turns = turns[1:]
            ids = render_chat(self.tok, turns) if turns else None
            if ids is None:
                return None
        else:
            ids = _ids(self.tok, "\n\n".join(turns))
        ids = ids[-(min(depth, self.max_depth) + self.window_len):]
        return _at(ids, len(ids) - self.window_len, self.window_len, "chat" if chat else "wiki", self.docs[paras[end]].name)

    def _default_window(self, i: int, idx: list[int], rng: random.Random, shallow_rank: int) -> dict:
        wiki = [j for j in idx if self.docs[j].kind == "wiki"]
        code = self._usable([j for j in idx if self.docs[j].kind == "code"])
        if shallow_rank is not None:
            order = list(wiki)
            random.Random(f"{self.seed}:shallow").shuffle(order)
            return self._shallow(order[shallow_rank % len(order)])
        kind = PATTERN[(i - N_SHALLOW_FIRST) % len(PATTERN)]
        if kind == "code" and code:
            j = rng.choice(code)
            return _deep_or_shallow(self.tokens(j), rng, self.window_len, self.max_depth, "code", self.docs[j].name)
        depth = self._chat_depth(rng)
        w = self._wiki_run(idx, rng, depth, chat=kind == "chat")
        return w or self._wiki_run(idx, rng, depth, chat=False)

    # -- every window
    def _pools(self) -> tuple[list[int], list[int]]:
        """The documents to fit on (48 tokens or more) and the long ones among them (deep windows; none for a .txt)."""
        if not hasattr(self, "_pool"):
            train = self._usable(self.train)
            if not train:
                raise ValueError(f"every document to fit on is under {MIN_TOKENS} tokens")
            long = [] if self.shallow_only else [j for j in train if len(self.tokens(j)) > self.window_len + SKIP]
            self._pool = (train, long)
        return self._pool

    def window(self, i: int) -> dict:
        rng = random.Random(f"{self.seed}:{i}")
        if self.kind == "default":
            return self._default_window(i, self.train, rng, i if i < N_SHALLOW_FIRST else None)
        # Round 0 is every document once (shuffled); later rounds only the long ones, each at a new depth.
        train, long = self._pools()
        if i < len(train):
            r, pos, pool = 0, i, train
        else:
            r, pos, pool = 1 + (i - len(train)) // len(long), (i - len(train)) % len(long), long
        order = list(pool)
        random.Random(f"{self.seed}:round{r}").shuffle(order)
        j = order[pos]
        return _deep_or_shallow(self.tokens(j), rng, self.window_len, self.max_depth, self.docs[j].kind,
                                self.docs[j].name, self.shallow_only)

    def windows(self, n: int = MAX_WINDOWS) -> Windows:
        """The first ``n`` fit windows; with no long document (a .txt, or only short files), one per document."""
        if self.kind != "default":
            train, long = self._pools()
            if not long:
                n = min(n, len(train))
        return Windows(self, n)

    def heldout(self) -> list[dict]:
        """The held-out windows (at most ``MAX_HELDOUT_WINDOWS``), the same every run: one per held-out document; the
        default's: shallow, wiki, chat and code, two of each, from held-out paragraphs and files only."""
        rng = random.Random(f"{self.seed}:heldout-windows")
        if self.kind == "default":
            out = [self._shallow(j) for j in self._usable([j for j in self.held if self.docs[j].kind == "wiki"])[:2]]
            for chat in (False, True):
                for _ in range(2):
                    w = self._wiki_run(self.held, rng, self._chat_depth(rng), chat)
                    if w:
                        out.append(w)
            for j in self._usable([j for j in self.held if self.docs[j].kind == "code"])[:2]:
                out.append(_deep_or_shallow(self.tokens(j), rng, self.window_len, self.max_depth, "code", self.docs[j].name))
            return out[:MAX_HELDOUT_WINDOWS]
        held = self._usable(self.held)
        return [_deep_or_shallow(self.tokens(j), rng, self.window_len, self.max_depth, self.docs[j].kind, self.docs[j].name,
                                 self.shallow_only) for j in held[:MAX_HELDOUT_WINDOWS]]

    def info(self) -> dict:
        """What lens.json and the job record say of the corpus."""
        return dict(id=self.id, kind=self.kind, source=self.source, description=self.description, docs=len(self.docs),
                    heldout_docs=len(self.held), shallow_only=self.shallow_only, window=self.window_len, max_depth=self.max_depth,
                    seed=self.seed)


def build(source: Optional[str], tok, *, seed: int = 0, window: int = WINDOW, max_depth: int = MAX_DEPTH, fetch=None,
          log=lambda _m: None) -> Corpus:
    """The corpus for a source (a directory, a .txt, or None: the default)."""
    kind, docs = documents(source, fetch, log)
    return Corpus(kind, docs, tok, source=str(Path(source).expanduser().resolve()) if source else None, seed=seed,
                  window=window, max_depth=max_depth)

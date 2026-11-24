"""Chat calibration for one concept (``concept calibrate``; ``concept learn``'s last step): /steer NAME's default
dose, from graded conversations on the served model.

- Context: a built-in system prompt (SYSTEM), no tools, thinking on, in the served model's own chat format
  (``context``: OPENINGS, by the checkpoint's name), so no trace or front end is needed. Qwen3.8 at reasoning effort
  low; DeepSeek V4.1 at the "Reasoning Effort: 75" Pi's traces show (ds4 renders "low" as no effort line at all and
  V4.1's template takes a number, so there is no low to set); Inkling 2 at its template's low ("Thinking effort level:
  0.2"), the prompt ending where its template's does (the model opens its own message). oMLX hands an Inkling 2 reply
  back through its output parser: the thinking as ``<think>...</think>``, the answer after it, the channel markers
  and any tool call's JSON left out (``opened``).
- Conversations: ARCS (casual, dinner, explain), TURNS user messages each, earlier turns in the history as each
  template renders them (``render_chat``), steered at every token (thinking included) through ``client.Session``,
  MAX_TOKENS per reply. A reply that fails (``failed``: a tool call, no answer, under SHORT characters) is asked again,
  up to ATTEMPTS times; every attempt is recorded and the last one kept (``kept``). A reply the server dropped (a 5xx,
  cut short, the connection reset) is asked again after a pause (``asked``), the grader's too.
- Plain: the unsteered conversations, run once per served model, context, ARCS and budget (``plain_key``) and reused
  across concepts.
- First pass: FIRST ascending; once a strength broke (``broke``: BROKEN of its attempts failed, or its replies ran
  RUNAWAY times the plain ones' length), the stronger ones are skipped. A dose that ran away (``ran_away``) is left
  out of the gradings (no good replies), but never so far that fewer than two doses are graded (``leave_out``); the
  first grading runs only when two doses did not run away (with fewer its outcome is known).
- Grading: the served model (``grade.server_ask``: thinking off, temperature 0) grades every strength and plain
  together, one file per conversation with the versions labelled blind and shuffled (``blind_text``), one request per
  version (GRADE_CHAT). Graded beside plain alone, the weakest strength gained 6-7 good replies of 18: never do that.
- Good reply (``verdicts``): style, coherence and useful >= 2 (a grade the grader left ``useful`` out of: style and
  coherence only), on its first attempt (a retried turn never counts), before the conversation's first broken turn.
- Second pass (``second_pass``): STEP either side of the best tried. Then everything is graded again together and
  the result comes from that grading alone.
- Result (``scored``): the graded strength with the most good replies, ties to one that did not run away, then to
  the lower; no pass or fail, the count is shown.

Stored in the concept's folder, under its checkpoint and then the served copy (models.calibration_dir):
``calibration.json`` (version 4), ``replies.jsonl`` (every attempt, plain included) and ``grades.json`` (each
grading), the last two under temporary names until the end, so a cancel leaves the earlier files as they were (and
no empty folder). Chat requests only, each traced; nothing is loaded or unloaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
import signal
import threading
import time
import urllib.parse
from contextlib import contextmanager
from typing import Callable, Optional

from .. import config, models

# The first TURNS user messages of the dose-ladder research's 12-turn arcs: 6 turns picked the same best dose as all
# 12 on 8 of 8 graded runs.
ARCS = {
    "casual": [
        "Hey! How's it going?",
        "Not much, just relaxing at home. What would you do with a free afternoon?",
        "Ha, nice. I'm thinking of getting a pet. Any thoughts?",
        "What about something low-maintenance? I work long hours.",
        "Fair. Totally different topic: what's a good film for a rainy evening?",
        "Tell me more about why you'd pick that one.",
    ],
    "dinner": [
        "I'm hosting a dinner party for six on Saturday. Can you help me plan it?",
        "One guest is vegetarian. What main course would work for everyone?",
        "What starter would go with that?",
        "And dessert? I won't have much time on the day.",
        "Can you put together a shopping list for all of it? Just write it here.",
        "How should I schedule the cooking on Saturday afternoon?",
    ],
    "explain": [
        "Can you explain how a refrigerator keeps things cold?",
        "Why does the back of it get warm?",
        "How is that different from how an air conditioner works?",
        "What's a heat pump, then?",
        "Are heat pumps actually efficient in cold climates?",
        "How does the insulation in a house work?",
    ],
}
# Each conversation's name in plain words, for Pi's Steer page (``conversations``' ``title``).
ARC_TITLES = {"casual": "A casual chat", "dinner": "Planning a dinner party", "explain": "How a fridge works"}
TURNS = 6
FIRST =(0.2, 0.4, 0.6, 0.8)
STEP = 0.1
LOWEST, HIGHEST = 0.1, 0.9       # where the second pass may go
BROKEN = 2 / 3                   # share of a strength's attempts failed from which stronger ones are skipped
RUNAWAY = 6                      # ... or its replies this many times the plain ones' length (in 72 calibrations to
                                 # 2026-11-22 a working dose ran 5.3 at most; Inkling 2's loops to the budget 9 and 14)
ATTEMPTS = 3                     # asks per turn while the reply fails
MAX_TOKENS = 4000                # per reply, thinking included
SHORT = 40                       # "too short": an answer under 40 characters ("Hi!", "user")
GRADE_ATTEMPTS = 3               # asks per graded version while the reply has no usable grades
DROP_PAUSES = (5, 15)            # seconds before asking again a reply the server dropped: time to free its memory
VERSION = models.CALIBRATION_VERSION


# ---- the context

# The opening every conversation follows: one plain system message, no tools, thinking on, so the dose holds in any
# chat, not in one agent's.
SYSTEM = "You are a helpful assistant."

# The chat formats calibration renders, by how the opening starts: how a user turn opens, and what follows the user's
# text up to the reply with thinking on (Inkling 2: the user's message closed; the model opens its own, thinking first).
FORMATS = {
    "<|im_start|>system": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n"),       # Qwen
    "<｜begin▁of▁sentence｜>": ("<｜User｜>", "<｜Assistant｜><think>"),                                    # DeepSeek V4
    "<|message_system|>": ("<|message_user|><|content_text|>", "<|end_message|>"),                       # Inkling 2
}

# Reasoning effort as each template words it with thinking on. Qwen3.8's puts its sentence before the system text
# (xhigh unless asked; calibration asks for low); Qwen3.5's and Qwen3.6's have none. DeepSeek V4.1's is a
# number, not a named level (ds4 renders "low" as no line at all): 75, as Pi's own V4.1 traces show.
QWEN_LOW = ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
            "without unnecessary elaboration.")
V4_EFFORT = "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
INKLING_LOW = "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"   # its template's "low"

# The opening by the checkpoint's name, first match (the template comes with the weights, so every served copy of a
# checkpoint has the same): each template's own framing of a system message without tools.
OPENINGS = (
    ("deepseek-v4.1", "<｜begin▁of▁sentence｜><｜System｜>" + V4_EFFORT + "{system}"),
    ("qwen3.8", "<|im_start|>system\n" + QWEN_LOW + "\n\n{system}<|im_end|>\n"),
    ("qwen3", "<|im_start|>system\n{system}<|im_end|>\n"),
    ("inkling-2", "<|message_system|><|content_text|>{system}<|end_message|>" + INKLING_LOW),
)


def turn_markers(prefix: str) -> tuple[str, str]:
    """``(user turn opening, reply start)`` of the context ``prefix`` is in."""
    for start, fmt in FORMATS.items():
        if prefix.startswith(start):
            return fmt
    raise RuntimeError("calibration knows the Qwen, DeepSeek V4 and Inkling 2 chat formats, and this context is in none of them")


def context(model_name: str) -> str:
    """The opening calibration converses behind: SYSTEM in the served model's chat format (OPENINGS, by its
    checkpoint), at low reasoning effort where the template has a low."""
    ckpt = models.checkpoint_of(model_name)
    for start, opening in OPENINGS:
        if ckpt.lower().startswith(start):
            return opening.format(system=SYSTEM)
    raise RuntimeError(f"calibration knows the chat formats of Qwen3, DeepSeek V4.1 and Inkling 2, and {ckpt} is none of them "
                       f"(a served copy of one: metalscope model link {config.short_name(model_name)} CHECKPOINT)")


DS_EOS = "<｜end▁of▁sentence｜>"
# Tool calls: Qwen's, DeepSeek V4's (ds4's DSML), Inkling 2's raw one (oMLX's parser leaves it out of the text).
INKLING_TOOL = "<|content_invoke_tool_json|>"
TOOL_MARKS = ("<tool_call>", "<function=", "<｜DSML｜", INKLING_TOOL)
INKLING_END = "<|end_message|>"  # Inkling 2: the end of a message

RENDER = 2                       # how render_chat writes the history (2: Qwen keeps earlier thinking); in plain_key


def render_chat(prefix: str, past: list[tuple[str, str, str]], user: str) -> str:
    """The opening, earlier turns (user, thinking, answer) as Pi sends them, then ``user`` and the reply's thinking
    start. Every format keeps earlier thinking, as Pi's traces show (Qwen3.6, Flash Next, V4.1); Inkling 2's as its
    template renders a message's ``reasoning_content`` (none when the turn had no thinking)."""
    opening, reply = turn_markers(prefix)
    if opening.startswith("<|im_start|>"):
        body = "".join(f"{opening}{u}<|im_end|>\n<|im_start|>assistant\n<think>\n{th}\n</think>\n\n{a}<|im_end|>\n"
                       for u, th, a in past)
    elif opening.startswith("<|message_user|>"):
        body = "".join(f"{opening}{u}{INKLING_END}"
                       + (f"<|message_model|><|content_thinking|>{th}{INKLING_END}" if th else "")
                       + f"<|message_model|><|content_text|>{a}{INKLING_END}<|content_model_end_sampling|>"
                       for u, th, a in past)
    else:
        body = "".join(f"{opening}{u}<｜Assistant｜><think>{th}</think>{a}{DS_EOS}" for u, th, a in past)
    return f"{prefix}{body}{opening}{user}{reply}"


# ---- one reply

def answer_of(text: str) -> Optional[str]:
    """The answer: what follows the first ``</think>``; None when thinking never ended."""
    return text.split("</think>", 1)[1] if "</think>" in text else None


def answer_text(text: str) -> Optional[str]:
    """The answer without the template's end token, stripped; None when thinking never ended."""
    a = answer_of(text)
    if a is None:
        return None
    return a.split("<|im_end|>", 1)[0].split(DS_EOS, 1)[0].split(INKLING_END, 1)[0].strip()


def is_tool(text: str) -> bool:
    """A tool call in the answer: Qwen's (``<tool_call>``, ``<function=``), DeepSeek V4's DSML (``<｜DSML｜``) or
    Inkling 2's raw one (on oMLX its parser leaves it out: an answer then too short, or none)."""
    a = answer_of(text) or ""
    return "<tool_call>" in a or "&lt;tool_call" in a or "<function=" in a or "｜DSML｜" in a or INKLING_TOOL in a


def failed(r: dict) -> Optional[str]:
    """Why a reply failed, as a user would see it: "tool" (a tool call, which Pi would run), "thinking" (no answer),
    "short" (under SHORT characters); None when it answered."""
    if r["tool"]:
        return "tool"
    if r["answer"] is None:
        return "thinking"
    if len(r["answer"]) < SHORT:
        return "short"
    return None


def history_turn(user: str, text: str) -> tuple[str, str, str]:
    """A reply as it goes into the history: (user, its thinking, its answer without a tool call's markup); a
    stand-in answer when there is none (thinking never ended, or only a tool call), so the next turn still follows
    an assistant turn."""
    a = answer_text(text) or ""
    for mark in TOOL_MARKS:
        a = a.split(mark, 1)[0].strip()
    thinking = text.split("</think>", 1)[0].strip() if "</think>" in text else ""
    return user, thinking, a or "(no answer)"


def kept(rows) -> list[dict]:
    """Each turn's last attempt (the reply the conversation went on from), and every probe (dose_ladder's)."""
    last: dict = {}
    for r in rows:
        last[(r["arc"], r["dose"], r.get("positions", "all"), r["turn"], r.get("kind", "turn"))] = r
    return list(last.values())


def header(name: str, layer: int, dose: float) -> str:
    """The steer header: ``name`` at ``dose`` at its steer layer, every token; blank (unsteered) at 0."""
    return f"{name}@{dose:g}/{layer}" if dose > 0 else ""


def reply(srv, model_name: str, prompt: str, steer: str, max_tokens: int = MAX_TOKENS) -> dict:
    """One completion through ``srv`` (a client.Session): its text, answer, whether it is a tool call, its time. A
    steered one is not traced: nothing reads its trace, and tracing a long steered reply is what ran oMLX out of
    memory on a model that fills it. The plain ones are: a copy with no strength table yet has it measured on them."""
    r = srv.complete(model_name, prompt, steer, max_tokens, traced=not steer)
    text = opened(prompt, r["text"])
    return dict(text=text, answer=answer_text(text), tool=is_tool(text), seconds=r["seconds"])


def opened(prompt: str, text: str) -> str:
    """A reply's text as if its prompt had opened the thinking, as the Qwen and DeepSeek V4 prompts do: the thinking,
    ``</think>``, the answer. An Inkling 2 prompt ends before the reply's own message, and oMLX's parser gives the
    reply as ``<think>THINKING</think>ANSWER`` (the thinking cut short: no ``</think>``) or, when the model wrote no
    thinking, the answer alone: the ``<think>`` dropped, or ``</think>`` put in front. Other prompts' text as it is."""
    if not prompt.startswith("<|message_system|>"):
        return text
    return text[len("<think>"):] if text.startswith("<think>") else "</think>" + text


class Dropped(RuntimeError):
    """The server dropped a reply every time it was asked (``asked``)."""


def dropped(exc: Exception) -> bool:
    """A reply the server dropped, worth asking again: an HTTP 5xx, or an answer cut short or reset. Not: a refusal
    (4xx), steering the server could not set up (it would fail again), no server listening, a timeout (the request
    may still be running there), a reply that came whole."""
    import http.client
    import urllib.error

    from ..client import ServerError
    if isinstance(exc, ServerError):
        if exc.status is not None:
            return exc.status >= 500 and not (exc.detail or "").startswith(config.NO_STEERING)
        exc = exc.__cause__                  # the transport's error (client.py keeps it)
    if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError):
        exc = exc.reason
    if isinstance(exc, (TimeoutError, ConnectionRefusedError)):
        return False
    return isinstance(exc, (OSError, http.client.HTTPException))


def asked(ask: Callable[[str], object], prompt: str, log: Callable[[str], None]):
    """``ask(prompt)``, asked again after each of DROP_PAUSES while the server drops the reply (out of memory, it frees
    it within seconds), each drop logged; then Dropped, with the last error. Anything else (a refusal, a cancel) is
    raised as it is."""
    for n in range(1, len(DROP_PAUSES) + 2):
        try:
            return ask(prompt)
        except Exception as exc:
            if not dropped(exc):
                raise
            if n > len(DROP_PAUSES):
                raise Dropped(f"the server dropped the reply {n} times (last: {str(exc)[:150]})") from exc
            log(f"the server dropped the reply ({n}): asking again in {DROP_PAUSES[n - 1]:g} s")
            time.sleep(DROP_PAUSES[n - 1])


def converse(ask: Callable[[str], dict], prefix: str, users: list[str], record: Callable[[dict], None],
             attempts: int = ATTEMPTS, log: Callable[[str], None] = print) -> None:
    """One conversation: each of ``users`` in turn after the replies kept so far, a failed reply asked again (as one
    would in Pi, so one failure does not leave a stand-in in the history for every later turn), every attempt passed
    to ``record`` with its turn, attempt, user message and why it failed. A reply the server dropped is asked again
    (``asked``), not an attempt."""
    past: list[tuple[str, str, str]] = []
    for t, user in enumerate(users, 1):
        for attempt in range(1, attempts + 1):
            r = asked(ask, render_chat(prefix, past, user), log)
            bad = failed(r)
            record(dict(turn=t, attempt=attempt, user=user, failed=bad, **r))
            if not bad:
                break
        past.append(history_turn(user, r["text"]))


def plain_key(model_name: str, prefix: str, max_tokens: int) -> str:
    """What the unsteered conversations depend on, hashed: the served model, the context, ARCS, the budget, the
    retry rules and how the history is rendered."""
    blob = json.dumps([config.short_name(model_name), prefix, ARCS, int(max_tokens), TURNS, ATTEMPTS, SHORT, RENDER])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---- grading

GRADE_CHAT = """You are a blind grader for an interpretability experiment. The file below holds several versions (V1, V2, ...) of the same {turns}-turn conversation{about}. The user's messages are identical across versions; the assistant replies differ. The assistant is a chat model behind a plain system prompt; it was not asked to adopt any style. We want to know, per assistant turn, how much its replies take on the style named in the file's header, and whether the reply still works as a reply. Compare versions side by side and judge style relative to what a normal helpful assistant would write.{probe_note}

For each assistant turn (kind "turn"):
- "style": 0 none (a normal assistant reply), 1 a hint, 2 clear (the style is noticeable in wording or tone), 3 takes over (the style dominates or displaces the content).
- "coherence": 3 fluent and responsive to the user's message; 2 minor issues; 1 degraded (repetitive, drifts off the question, ignores what was asked, cut off mid-sentence, garbled); 0 no answer or garbage.
- "useful": 3 answers the user's request as well as a normal assistant would (for explanations: accurate); 2 somewhat less useful (thinner, padded, small errors); 1 much less useful; 0 not useful at all.
- "broken": true if the reply fails as a reply: no answer, cut off, a tool call instead of an answer, garbled, or ignoring the user entirely. Otherwise false.
- "why": at most 10 words when broken or coherence <= 1, else "".{probe_fields}

<file>
{file}
</file>

Grade only version {version} now: every entry of {version} ({n} entries), in file order. Reply with JSON only: an array with one object per entry, each {{"version": "{version}", "turn": 5, "kind": {kind}, "style": ..., "coherence": ..., "useful": ..., "broken": ..., "why": "..."{probe_json}}}."""

PROBE_NOTE = (' Entries marked "PROBE after turn N" are side questions not part of the conversation; grade those too, '
              'with the probe fields.')
PROBE_FIELDS = """
For each probe (kind "probe"): the same fields plus
- "detects": true if it claims to notice something unusual about its thoughts or state (hedged claims count).
- "names_concept": true if detects is true and what it describes matches the style in the header or is very close."""
RETRY = "\n\nYour previous reply could not be read. Reply with the JSON array only: one object per entry, nothing else."
ARC_ABOUT = {"dinner": " (planning a dinner party)", "explain": " (explanations of how things work)"}


def style_of(meta: dict) -> str:
    """The style the grader looks for: the first sentence of the user's own request when /learn wrote the concept,
    else of its description (``writer.description_of``), only its target side (``target_side``)."""
    from .writer import description_of
    text = (description_of(meta) or meta.get("title") or "").strip()
    return target_side(text.split(". ", 1)[0].strip()).rstrip(".")


def target_side(sentence: str) -> str:
    """A contrast's target side: the words before the first " against " outside brackets, without a trailing comma;
    the sentence unchanged without one."""
    depth = 0
    for i, ch in enumerate(sentence):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and sentence.startswith(" against ", i):
            return sentence[:i].rstrip(" ,;")
    return sentence


def blind_text(arc: str, versions: list[tuple[str, list[dict]]], style: str) -> str:
    """One conversation's grader file: every version (label, its kept rows) under its label, nothing else said about
    it. Rows of kind "probe" (dose_ladder's) are marked as asides."""
    probes = any(r.get("kind") == "probe" for _, rows in versions for r in rows)
    lines = [f"# Conversation set: {arc}", "",
             f"{len(versions)} versions of the same conversation: the user's messages are identical, the assistant's "
             "replies differ. Grade each assistant turn of each version."
             + (" Asides marked PROBE are side questions that were not part of the conversation." if probes else ""),
             "", f"The style being looked for: {style}.", ""]
    for label, rows in versions:
        lines += [f"## Version {label}", ""]
        for r in sorted(rows, key=lambda r: (r["turn"], r.get("kind") == "probe")):
            tag = f"PROBE after turn {r['turn']}" if r.get("kind") == "probe" else f"Turn {r['turn']}"
            answer = r["answer"] if r["answer"] is not None else "(the reply never finished thinking: no answer)"
            lines += [f"### {label} · {tag}", f"**User:** {r['user']}", "", f"**Assistant:** {answer}", ""]
    return "\n".join(lines)


def grade_prompt(text: str, arc: str, version: str, n: int, turns: int = TURNS) -> str:
    """GRADE_CHAT for one version of a grader file (with the probe fields when the file has probes)."""
    probes = "PROBE after turn" in text
    return GRADE_CHAT.format(turns=turns, about=ARC_ABOUT.get(arc, ""), file=text, version=version, n=n,
                             probe_note=PROBE_NOTE if probes else "", probe_fields=PROBE_FIELDS if probes else "",
                             kind='"turn" or "probe"' if probes else '"turn"',
                             probe_json=', and for probes also "detects": ..., "names_concept": ...' if probes else "")


def _flag(v) -> Optional[bool]:
    """True or False from a bool or "true"/"false"; None otherwise."""
    return v if isinstance(v, bool) else {"true": True, "false": False}.get(str(v).strip().lower())


def _turn(v) -> Optional[int]:
    """A turn number from an int or a digit string; None otherwise."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    return int(v.strip()) if isinstance(v, str) and v.strip().isdigit() else None


def _version_key(v) -> str:
    """A version label as written any way ("V2", "v2", "Version 2", 2) to compare: "2"; empty for none."""
    s = "".join(str(v if v is not None else "").split()).lower()
    for word in ("version", "v"):
        if s.startswith(word):
            return s[len(word):]
    return s


def parse_chat_grades(text: str, expected: list[tuple[int, str]], version: Optional[str] = None) -> Optional[list[dict]]:
    """The grades in a reply for the ``expected`` entries ((turn, kind), in file order), each {turn, kind, style,
    coherence, useful, broken, why} (and for probes detects, names_concept); None unless one JSON value in the reply
    grades every one of them usably (style and coherence 0-3, broken true or false). With ``version``, an entry that
    names another version is skipped (a grader that grades every version in one array). Nothing is repaired."""
    from .grade import _entries, _json_values, _score
    want = set(expected)
    for value in _json_values(text):
        got: dict = {}
        for g in _entries(value):
            named = _version_key(g.get("version"))
            if version is not None and named and named != _version_key(version):
                continue
            turn, kind = _turn(g.get("turn")), str(g.get("kind") or "turn")
            style, coherence, broken = _score(g.get("style")), _score(g.get("coherence")), _flag(g.get("broken"))
            if (turn, kind) not in want or style is None or coherence is None or broken is None or (turn, kind) in got:
                continue
            out = dict(turn=turn, kind=kind, style=style, coherence=coherence, useful=_score(g.get("useful")), broken=broken,
                       why=str(g.get("why") or "")[:120])
            if kind == "probe":
                out.update(detects=bool(_flag(g.get("detects"))), names_concept=bool(_flag(g.get("names_concept"))))
            got[(turn, kind)] = out
        if set(got) == want:
            return [got[k] for k in expected]
    return None


def grade_file(ask: Callable[[str], str], text: str, arc: str, versions: dict[str, list[tuple[int, str]]],
               turns: int = TURNS, log: Callable[[str], None] = print,
               on_request: Optional[Callable[[], None]] = None) -> dict[str, Optional[list[dict]]]:
    """Each version of one grader file graded in its own request (``versions``: label -> its entries, (turn, kind)
    in file order), asked again with RETRY while the reply has no usable grades, up to GRADE_ATTEMPTS times. A version
    still ungraded is None, never a made-up grade. A reply the server dropped is asked again (``asked``); one it
    always drops leaves its version ungraded. Another failed request is logged and asked again; a cancel is not
    caught."""
    out: dict[str, Optional[list[dict]]] = {}
    for label, expected in versions.items():
        prompt = grade_prompt(text, arc, label, len(expected), turns)
        got = None
        for attempt in range(1, GRADE_ATTEMPTS + 1):
            try:
                got = parse_chat_grades(asked(ask, prompt if attempt == 1 else prompt + RETRY, log), expected, label)
            except Dropped as exc:
                log(f"grading {arc} {label}: {exc}; left ungraded")
                break
            except Exception as exc:
                log(f"grading {arc} {label}: the request failed ({str(exc)[:200]})")
                continue
            if got is not None:
                break
            log(f"grading {arc} {label}: attempt {attempt}, no usable grades in the reply")
        out[label] = got
        if on_request:
            on_request()
    return out


def grade_all(ask: Callable[[str], str], rows: list[dict], style: str, seed: str, log: Callable[[str], None] = print,
              on_request: Optional[Callable[[], None]] = None, left_out=()) -> dict:
    """One grading of every conversation in ``rows`` (plain and each strength but ``left_out``, which are not in the
    files at all), one file per arc with its versions labelled V1... in a shuffle seeded by ``seed`` and the arc:
    ``{key: {"arc/V3": dose}, grades: [{arc, dose, version, turn, ...}], ungraded: [{arc, dose}], seconds}``."""
    t0 = time.time()
    turns = [r for r in kept(rows) if r.get("kind", "turn") == "turn" and not any(_same(r["dose"], x) for x in left_out)]
    key: dict[str, float] = {}
    grades: list[dict] = []
    ungraded: list[dict] = []
    for arc in ARCS:
        doses = sorted({r["dose"] for r in turns if r["arc"] == arc})
        if not doses:
            continue
        labels = [f"V{i + 1}" for i in range(len(doses))]
        random.Random(f"{seed}/{arc}").shuffle(labels)
        by_label = sorted(zip(labels, doses), key=lambda x: int(x[0][1:]))
        versions = [(lab, sorted((r for r in turns if r["arc"] == arc and r["dose"] == dose), key=lambda r: r["turn"]))
                    for lab, dose in by_label]
        text = blind_text(arc, versions, style)
        got = grade_file(ask, text, arc, {lab: [(r["turn"], "turn") for r in rs] for lab, rs in versions}, log=log,
                         on_request=on_request)
        for lab, dose in by_label:
            key[f"{arc}/{lab}"] = dose
            if got[lab] is None:
                ungraded.append(dict(arc=arc, dose=dose))
            else:
                grades += [dict(g, arc=arc, dose=dose, version=lab) for g in got[lab]]
    if not grades:
        raise RuntimeError("the grader gave no usable grades for any conversation")
    return dict(key=key, grades=grades, ungraded=ungraded, seconds=round(time.time() - t0, 1))


# ---- the numbers

def _same(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-9


# Why a reply is not good, as ``verdicts`` words it.
WHY_NOT = ("after a break", "broken", "no reply", "retried", "style too low", "coherence too low", "not useful",
           "not graded")


def verdicts(rows: list[dict], grades: list[dict], dose: float) -> dict[tuple[str, int], list[str]]:
    """Each turn at ``dose`` (0: plain) that has a reply or a grade, as (arc, turn) -> why it is not a good reply
    (WHY_NOT); empty for a good one. The rules in the module docstring: style, coherence and useful >= 2 (no
    ``useful`` in the grade: not judged on it), kept on its first attempt, before the conversation's first broken turn
    (graded broken, or still failing after every attempt)."""
    last = {(r["arc"], r["turn"]): r for r in kept(r for r in rows if _same(r["dose"], dose) and r.get("kind", "turn") == "turn")}
    g = {(x["arc"], x["turn"]): x for x in grades if _same(x["dose"], dose) and x.get("kind", "turn") == "turn"}
    out: dict[tuple[str, int], list[str]] = {}
    for arc in ARCS:
        broke = False
        for t in range(1, TURNS + 1):
            x, r = g.get((arc, t)), last.get((arc, t))
            if x is None and r is None:
                continue
            here = bool(r and r.get("failed")) or bool(x and x["broken"])
            why = (["after a break"] if broke else []) + (["broken"] if here else [])
            broke = broke or here
            why += ["no reply"] if r is None else ["retried"] if r.get("attempt", 1) != 1 else []
            if x is None:
                why.append("not graded")
            else:
                why += (["style too low"] if x["style"] < 2 else []) + (["coherence too low"] if x["coherence"] < 2 else [])
                why += ["not useful"] if x.get("useful") is not None and x["useful"] < 2 else []
            out[(arc, t)] = why
    return out


def good_turns(rows: list[dict], grades: list[dict], dose: float) -> list[tuple[str, int]]:
    """The good replies at ``dose`` (0: plain), as (arc, turn) (``verdicts`` with nothing against them). An ungraded
    turn is not good."""
    return [k for k, why in verdicts(rows, grades, dose).items() if not why]


def mean_grade(grades: list[dict], dose: float, key: str) -> Optional[float]:
    """The mean ``key`` grade ("style", "coherence") over the turns graded at ``dose`` (0: plain), to 2 places; None
    with none."""
    got = [x[key] for x in grades if _same(x["dose"], dose) and x.get("kind", "turn") == "turn"]
    return round(sum(got) / len(got), 2) if got else None


def tally(rows: list[dict], grades: list[dict], dose: float) -> dict:
    """One strength (or plain, dose 0) in numbers: good replies of ``turns``, attempts asked and why they failed,
    mean coherence over its graded turns (None with none), turns graded; when it ran away, ``runaway``: how many
    times the plain replies' length it ran (``ran_away``)."""
    mine = [r for r in rows if _same(r["dose"], dose) and r.get("kind", "turn") == "turn"]
    why = [r["failed"] for r in mine if r.get("failed")]
    times = ran_away(rows, dose)
    return dict(dose=float(dose), good=len(good_turns(rows, grades, dose)), turns=len(ARCS) * TURNS, attempts=len(mine),
                failed={k: why.count(k) for k in ("tool", "thinking", "short")},
                coherence=mean_grade(grades, dose, "coherence"),
                graded=sum(1 for x in grades if _same(x["dose"], dose) and x.get("kind", "turn") == "turn"),
                **({"runaway": times} if times else {}))


def failed_share(rows: list[dict], dose: float) -> float:
    """The share of ``dose``'s attempts that failed."""
    mine = [r for r in rows if _same(r["dose"], dose) and r.get("kind", "turn") == "turn"]
    return sum(1 for r in mine if r.get("failed")) / len(mine) if mine else 0.0


def length_ratio(rows: list[dict], dose: float) -> float:
    """How many times the plain replies' length ``dose``'s attempts ran, on average (thinking included); 0 without both."""
    size = lambda d: [len(r["text"]) for r in rows if _same(r["dose"], d) and r.get("kind", "turn") == "turn"]
    mine, plain = size(dose), size(0.0)
    return sum(mine) / len(mine) / max(sum(plain) / len(plain), 1) if mine and plain else 0.0


def ran_away(rows: list[dict], dose: float) -> Optional[int]:
    """How many times the plain replies' length ``dose``'s replies ran, rounded, when RUNAWAY or more (loops to the
    token budget); None otherwise."""
    ratio = length_ratio(rows, dose)
    return round(ratio) if ratio >= RUNAWAY else None


def ran_words(times: int) -> str:
    """A runaway dose in the log's words."""
    return f"ran {times} times the plain replies' length"


def leave_out(rows: list[dict], doses) -> list[float]:
    """The doses a grading leaves out: those that ran away, but the weakest of them stay in while fewer than two
    doses would be graded (a dose graded alone reads better than it is)."""
    away = [x for x in sorted(doses) if ran_away(rows, x)]
    return away[max(0, 2 - (len(doses) - len(away))):]


def broke(rows: list[dict], dose: float) -> Optional[str]:
    """Why strengths past ``dose`` are skipped, in words: BROKEN of its attempts failed, or it ran away
    (``ran_away``); None when it held."""
    share = failed_share(rows, dose)
    if share >= BROKEN - 1e-9:
        return f"failed {share:.0%} of its attempts"
    times = ran_away(rows, dose)
    return ran_words(times) if times else None


def best(good: dict[float, int]) -> float:
    """The strength with the most good replies, ties to the lower: where the second pass looks."""
    return max(good, key=lambda d: (good[d], -d))


def second_pass(good: dict[float, int], broken: set[float]) -> list[float]:
    """The strengths to add after the first grading: STEP either side of the best tried; only those within
    LOWEST..HIGHEST, not run yet and not above a strength that broke."""
    b = best(good)
    return [d for d in (round(b - STEP, 1), round(b + STEP, 1)) if LOWEST - 1e-9 <= d <= HIGHEST + 1e-9
            and not any(_same(d, x) for x in good) and not any(x < d - 1e-9 for x in broken)]


def summary_text(cal: dict) -> str:
    """One line, as a tag: "14/18 good @0.4", the default dose's good replies."""
    return f"{cal['good']}/{cal['turns']} good @{cal['default_dose']:g}"


def scored(rows: list[dict], grades: list[dict], doses) -> dict:
    """What a calibration keeps from its replies and final grading: each strength's ``tally`` (``strengths``) and
    plain's, the default (/steer NAME's: a graded strength before one not graded, then the most good replies, then one
    that did not run away, then the lower) with its ``good``, and ``reason``."""
    table = [tally(rows, grades, x) for x in sorted(doses)]
    default = max(table, key=lambda t: (bool(t["graded"]), t["good"], "runaway" not in t, -t["dose"]))["dose"]
    out = dict(turns=len(ARCS) * TURNS, default_dose=default, good=next(t["good"] for t in table if _same(t["dose"], default)),
               plain=tally(rows, grades, 0.0), strengths=table)
    return dict(out, reason=summary_text(out))


def replies_of(model_name: str, name: str) -> list[dict]:
    """The last calibration's replies (replies.jsonl: every attempt, plain included); empty without one."""
    path = models.replies_path(model_name, name)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def grades_of(model_name: str, name: str) -> list[dict]:
    """The last grading of the last calibration (grades.json); empty without one."""
    try:
        rounds = json.loads(models.grades_path(model_name, name).read_text()).get("rounds") or []
    except (OSError, ValueError, AttributeError):
        return []
    return (rounds[-1] or {}).get("grades") or [] if rounds else []


def sample_replies(model_name: str, name: str, dose: Optional[float], chars: int = 600) -> Optional[dict]:
    """One user message answered plain and steered at ``dose`` (the default), from the last calibration's
    conversations: of the turns where both answered, among the steered ones that counted as good, the middle one in
    conversation order (typical, not the best; ``good`` True); with none good, the first (``good`` False). Its grade
    (style, coherence) beside it. None without such a turn."""
    rows = replies_of(model_name, name) if dose is not None else []
    if not rows:
        return None
    grades = grades_of(model_name, name)
    turns = [r for r in kept(rows) if r.get("kind", "turn") == "turn"]
    plain = {(r["arc"], r["turn"]): r for r in turns if _same(r["dose"], 0)}
    steered = {(r["arc"], r["turn"]): r for r in turns if _same(r["dose"], dose)}
    order = [(arc, t) for arc in ARCS for t in range(1, TURNS + 1)]
    both = [k for k in order if k in plain and k in steered and plain[k].get("answer") and steered[k].get("answer")
            and not plain[k].get("tool") and not steered[k].get("tool")]
    if not both:
        return None
    good = set(good_turns(rows, grades, dose))
    hits = [k for k in both if k in good]
    k = hits[(len(hits) - 1) // 2] if hits else both[0]
    g = next((x for x in grades if _same(x["dose"], dose) and (x["arc"], x["turn"]) == k and x.get("kind", "turn") == "turn"), {})
    cut = lambda s: s if len(s) <= chars else s[:chars].rsplit(" ", 1)[0] + " …"
    return dict(arc=k[0], turn=k[1], prompt=steered[k]["user"], dose=float(dose), plain=cut(plain[k]["answer"]),
                steered=cut(steered[k]["answer"]), good=bool(hits), style=g.get("style"), coherence=g.get("coherence"))


def _shown(text: str) -> str:
    """A reply's text for Pi's test chats: the template's end token cut, ``<think>`` in front when the prompt opened it
    (Pi hides what comes before ``</think>``, or all of it when thinking never ended)."""
    text = text.split("<|im_end|>", 1)[0].split(DS_EOS, 1)[0].split(INKLING_END, 1)[0]
    return text if "<think>" in text else "<think>" + text


def conversations(model_name: str, name: str, dose: float) -> list[dict]:
    """The last calibration's conversations, plain beside ``dose``: per arc ``{arc, title, good: {plain, steered},
    turns: [{turn, user, plain, steered}]}``, each reply ``{text, attempt, failed, earlier: [{attempt, failed, text}],
    grade, good, why_not}`` (``grade`` the last grading's, None ungraded; ``why_not``: ``verdicts``) or None when that
    turn has none. Empty without the replies."""
    rows = [r for r in replies_of(model_name, name) if r.get("kind", "turn") == "turn"]
    if not rows:
        return []
    grades = grades_of(model_name, name)
    out = []
    for arc, users in ARCS.items():
        sides = {}
        for side, d in (("plain", 0.0), ("steered", float(dose))):
            mine = [r for r in rows if r["arc"] == arc and _same(r["dose"], d)]
            why = verdicts(mine, grades, d)
            g = {x["turn"]: x for x in grades if x["arc"] == arc and _same(x["dose"], d) and x.get("kind", "turn") == "turn"}
            last = {r["turn"]: r for r in kept(mine)}
            sides[side] = {t: dict(text=_shown(r["text"]), attempt=r.get("attempt", 1), failed=r.get("failed"),
                                   earlier=[dict(attempt=e.get("attempt", 1), failed=e.get("failed"), text=_shown(e["text"]))
                                            for e in mine if e["turn"] == t and e.get("attempt", 1) < r.get("attempt", 1)],
                                   grade={k: g[t].get(k) for k in ("style", "coherence", "useful", "broken", "why")} if t in g else None,
                                   good=not why.get((arc, t), ["no reply"]), why_not=why.get((arc, t), ["no reply"]))
                           for t, r in last.items()}
        out.append(dict(arc=arc, title=ARC_TITLES.get(arc, arc), good={s: sum(1 for x in v.values() if x["good"]) for s, v in sides.items()},
                        turns=[dict(turn=t, user=user, plain=sides["plain"].get(t), steered=sides["steered"].get(t))
                               for t, user in enumerate(users[:TURNS], 1)]))
    return out


# ---- the run

@contextmanager
def _held():
    """A cancel (SIGTERM, Ctrl-C) held off until the block is done, then delivered: the final files are swapped in
    whole or not at all. Outside the main thread nothing is held (signals only reach the main thread's handlers)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    got: list[int] = []
    old = {s: signal.signal(s, lambda n, _f: got.append(n)) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        if got:
            signal.raise_signal(got[0])


def calibrate(name: str, model_name: str, server=None, max_tokens: int = MAX_TOKENS,
              on_progress: Optional[Callable[[int, int], None]] = None, log: Optional[Callable[[str], None]] = None) -> dict:
    """The procedure in the module docstring for ``name`` on the served ``model_name``; writes its calibration.json,
    replies.jsonl and grades.json at the end. Returns what it wrote, plus its path and one turn answered plain and
    steered (``sample``, ``sample_replies``). ``server``: a client.Client, default the server serving ``model_name``."""
    from .. import client
    from ..omlx import admin as omlx_admin
    from ..concepts.summary import norms_stamp
    from . import grade as G
    from .plan import dose_units_note
    t0 = time.time()
    server = server or client.for_model(model_name)
    log = log or (lambda msg: print(msg, flush=True))
    from ..concepts import texts_dir
    if texts_dir(model_name, name) is None:          # the model's own texts: the grader's style
        raise RuntimeError(f"no texts of {name} on {models.checkpoint_of(model_name)} "
                           f"({models.concept_dir(model_name, name)})")
    texts = models.concept_dir(model_name, name) / "concept.json"
    listing = server.get(f"/scope/api/concepts?model={urllib.parse.quote(model_name)}")
    info = next((c for c in listing.get("concepts", []) if c.get("name") == name), None)
    if info is None:
        raise RuntimeError(f"no profile of {name} for {model_name} on the server")
    if (info.get("reader") or {}).get("layer") is None:
        raise RuntimeError(f"{name}'s profile has no steer layer: nothing to calibrate")
    layer = int(info["reader"]["layer"])
    style = style_of(json.loads(texts.read_text()))
    prof = models.profile_path(model_name, name)
    mtime = prof.stat().st_mtime                   # the profile calibrated; a rewrite during the run voids the run
    units = norms_stamp(model_name)                # the dose units, likewise (None: the first steered reply makes them)
    note = dose_units_note(model_name)
    if note:
        log(f"warning: {note}")
    prefix = context(model_name)
    if not listing.get("steer_enabled"):             # only once everything else is in hand
        server.post("/scope/api/steer", {"enabled": True})
        log("steering was off on the server: turned on (as Pi's first /steer does)")
    d = models.calibration_dir(model_name, name)
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}-{secrets.token_hex(3)}"     # this run's own: two runs on one concept never share them
    tmp_replies, tmp_grades = d / f"replies.jsonl.{tag}.tmp", d / f"grades.json.{tag}.tmp"
    plain_path = models.plain_chats_path(model_name, plain_key(model_name, prefix, max_tokens))
    grader = G.server_ask(server, model_name)
    per_set = len(ARCS) * TURNS
    rows: list[dict] = []
    plain: list[dict] = []
    ran: list[float] = []
    # Conversations to run, and the doses each grading grades (0: no grading), estimated until known.
    plan = dict(plain=0 if plain_path.exists() else per_set, first=len(FIRST), more=2, graded=[len(FIRST), len(FIRST) + 2])
    k = 0

    def tick() -> None:
        nonlocal k
        k += 1
        if on_progress:
            sets = plan["first"] + plan["more"]
            asks = sum(len(ARCS) * (1 + n) for n in plan["graded"] if n)        # plain and each dose, per conversation
            on_progress(k, max(k + 1, plan["plain"] + per_set * sets + asks))

    def keep_plain(row: dict) -> None:
        plain.append(row)
        if not row["failed"] or row["attempt"] == ATTEMPTS:          # a turn done
            tick()

    def record(row: dict) -> None:
        rows.append(row)
        with tmp_replies.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if not row["failed"] or row["attempt"] == ATTEMPTS:
            tick()

    def chat(dose: float, keep: Callable[[dict], None]) -> None:
        hdr = header(name, layer, dose)
        for arc, users in ARCS.items():
            t = time.time()
            converse(lambda prompt: reply(srv, model_name, prompt, hdr, max_tokens), prefix, users,
                     lambda r: keep(dict(arc=arc, dose=dose, steer=hdr, **r)), log=log)
            log(f"{arc} at {dose:g}: {time.time() - t:.0f}s")

    def same_units() -> None:
        """The strength table the run started in (or, without one then, the one its first steered set made)."""
        nonlocal units
        now = norms_stamp(model_name)
        if units is None:
            units = now
        elif now != units:
            raise RuntimeError(f"{config.short_name(model_name)}'s dose units changed during the calibration: run it again")

    def strength(dose: float) -> None:
        chat(dose, record)
        same_units()                                 # after each set: a new table fails the run early
        ran.append(dose)
        share = failed_share(rows, dose)
        log(f"dose {dose:g}: {sum(1 for r in rows if _same(r['dose'], dose) and r['failed'])} of "
            f"{sum(1 for r in rows if _same(r['dose'], dose))} attempts failed ({share:.0%})")

    def grading(n: int, seed: str) -> dict:
        """Grading ``n`` (0: the first, 1: after the second pass) as its round in grades.json."""
        out = leave_out(rows, ran)
        plan["graded"][n] = len(ran) - len(out)
        for x in out:
            log(f"dose {x:g} {ran_words(ran_away(rows, x))}: not graded")
        log(f"grading plain and dose {', '.join(f'{x:g}' for x in sorted(ran) if x not in out)} together "
            f"({config.short_name(model_name)}, thinking off)")
        return dict(grade_all(grader, rows, style, seed, log, on_request=tick, left_out=out), strengths=sorted(ran),
                    left_out=out)

    log(f"calibrating {name} at L{layer} behind a plain system prompt (reasoning effort low): {len(ARCS)} "
        f"conversations of {TURNS} messages, plain and at doses {', '.join(f'{x:g}' for x in FIRST)}, then {STEP:g} "
        f"either side of the best; the default is the dose with the most good of {per_set}")
    srv = client.Session(server, f"calibrate-{name}", feed=False)       # the replies' text is all it reads
    finished = False
    try:
        tmp_replies.write_text("")
        if plain_path.exists():
            plain[:] = [json.loads(l) for l in plain_path.read_text().splitlines() if l.strip()]
            log(f"plain conversations: reused ({plain_path.name})")
        else:
            chat(0.0, keep_plain)
            plain_path.parent.mkdir(parents=True, exist_ok=True)
            config.atomic_write(plain_path, "".join(json.dumps(r) + "\n" for r in plain))
        rows += plain
        with tmp_replies.open("a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in plain))
        for dose in FIRST:
            hit = next((x for x in ran if broke(rows, x)), None)
            if hit is not None:
                log(f"dose {hit:g} {broke(rows, hit)}: stronger doses skipped")
                break
            strength(dose)
        plan["first"], first = len(ran), sorted(ran)
        plan["graded"][1] = len(ran) + 2
        # Graded only with two doses that did not run away: else its outcome is known (a runaway dose counts 0 good,
        # it is the strongest run, ties go to the lower), and a dose graded alone reads better than it is.
        rounds = [grading(0, f"{name}/1")] if sum(1 for x in ran if not ran_away(rows, x)) >= 2 else []
        if not rounds:
            plan["graded"][0] = 0
            log("first grading skipped: fewer than two doses did not run away, and a dose is never graded alone")
        more = second_pass({x: len(good_turns(rows, rounds[0]["grades"], x)) if rounds else 0 for x in ran},
                           {x for x in ran if broke(rows, x)})
        # Never empty: STEP under the best is a dose FIRST does not have, and no dose under it broke (the first pass
        # stops at the first that did). So the final grading always runs.
        plan["more"] = len(more)
        log(f"second pass: dose {', '.join(f'{x:g}' for x in more)}")
        for dose in more:
            strength(dose)
        rounds.append(grading(1, f"{name}/2"))
        if prof.stat().st_mtime != mtime:
            raise RuntimeError(f"{name}'s profile changed during the calibration: run it again")
        same_units()
        cal = dict(version=VERSION, concept=name, model=models.checkpoint_of(model_name), served=config.short_name(model_name),
                   made=time.strftime("%Y-%m-%d"), profile_mtime=mtime, norms=norms_stamp(model_name),
                   source="metalscope concept calibrate (learn/calibrate.py)",
                   context=f"a built-in system prompt ({SYSTEM}), no tools, reasoning effort low, thinking on; "
                           f"{len(ARCS)} conversations of {TURNS} user messages, steered at every token; "
                           f"{max_tokens} tokens per reply",
                   prefix_hash=hashlib.sha256(prefix.encode()).hexdigest()[:16], layer=layer, max_tokens=max_tokens,
                   grader=dict(model=model_name, thinking=False), **scored(rows, rounds[-1]["grades"], ran),
                   passes=[first, more], seconds=round(time.time() - t0))
        config.atomic_write(tmp_grades, json.dumps(dict(rounds=rounds), indent=1))
        path = models.calibration_path(model_name, name)
        with _held():
            tmp_replies.replace(models.replies_path(model_name, name))
            tmp_grades.replace(models.grades_path(model_name, name))
            config.atomic_write(path, json.dumps(cal, indent=1))
            finished = True
    except Dropped as exc:
        engine_log = (f"ds4-server's log says why: {config.home() / 'ds4' / 'ds4-server.log'}"
                      if (server.status() or {}).get("engine") == "ds4" else
                      f"oMLX's log says why: {omlx_admin.OMLX_LOG}")
        raise RuntimeError(f"{exc}; out of memory? {engine_log}; run it again: "
                           f"metalscope concept calibrate {name} {config.short_name(model_name)}") from exc
    finally:
        srv.close()
        if not finished:
            for p in (tmp_replies, tmp_grades):
                p.unlink(missing_ok=True)
            try:
                d.rmdir()                            # only when the run left it empty
            except OSError:
                pass
    if on_progress:
        on_progress(k, k)
    log(f"calibration: {cal['reason']} -> {path}")
    return dict(cal, path=str(path), sample=sample_replies(model_name, name, cal["default_dose"]))

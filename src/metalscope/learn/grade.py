"""Asking the served model to grade, and reading its grades back: what the chat calibration (learn/calibrate.py)
uses. ``server_ask`` asks through the server's chat API (thinking off, greedy); ``_json_values``, ``_entries`` and
``_score`` find the grade objects in a reply and read a 0-3 score, nothing repaired."""
from __future__ import annotations

import json
from typing import Callable, Optional

from .. import models
from ..client import UNTRACED

# Thinking off, and no switch for it: on the 12 dose ladders Opus graded, DeepSeek V4.1 off matched Opus's style
# within 1 on all 288 texts in ~80 s a ladder; on agreed less, took ~7x as long and graded nothing of shouting, every
# thought outrunning 16k tokens.
THINKING = False
# Per request: 24 grades with short notes are about 1,000 tokens; thinking needs room to finish before the JSON.
MAX_TOKENS = {False: 2000, True: 16000}
# Where ``enable_thinking`` switches nothing, by the checkpoint's name, the lowest ``reasoning_effort`` its template
# takes (oMLX reads it as a request field). Inkling 2 still thinks at "none" (about 300 characters on a one-line
# question, 2026-11-22), so its thought needs room too (``server_ask``).
EFFORT_OFF = (("inkling-2", "none"),)


def thinking_fields(model: str, thinking: bool) -> dict:
    """A chat request's thinking switch: ``chat_template_kwargs.enable_thinking``, and with thinking off on a model
    in EFFORT_OFF its lowest ``reasoning_effort``."""
    out: dict = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    ckpt = models.checkpoint_of(model).lower()
    effort = next((e for start, e in EFFORT_OFF if ckpt.startswith(start)), None)
    return dict(out, reasoning_effort=effort) if effort and not thinking else out


def thinks_anyway(model: str) -> bool:
    """Whether ``model`` thinks even with thinking off (EFFORT_OFF: only less)."""
    return "reasoning_effort" in thinking_fields(model, False)


def answer_text(text: str, thinking: bool = False) -> str:
    """The reply after its thinking (the last ``</think>``). Without one: the whole text, or nothing when the reply was
    meant to think first (``thinking``: it ran out of tokens mid-thought, and a draft there is no answer)."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return "" if thinking else text


def _score(v) -> Optional[int]:
    """0-3 from an int, an integral float or a digit string; None otherwise."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str) and v.strip().isdigit():
        v = int(v.strip())
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return v if isinstance(v, int) and 0 <= v <= 3 else None


def _json_values(text: str):
    """Every JSON array or object in ``text`` that parses, outermost first, left to right: a reply's chatter, a
    bracket in a preamble or a remark after the array cost nothing, and a reply cut off mid-array still yields its
    complete objects."""
    dec, i = json.JSONDecoder(), 0
    while True:
        starts = [j for j in (text.find("[", i), text.find("{", i)) if j >= 0]
        if not starts:
            return
        i = min(starts)
        try:
            value, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1                          # not JSON from here (or cut off): look inside it
            continue
        yield value
        i = end


def _entries(value):
    """The grade objects in one JSON value: an array of them, an object holding such an array, an object keyed by id,
    or one grade object."""
    if isinstance(value, list):
        return [g for g in value if isinstance(g, dict)]
    if not isinstance(value, dict):
        return []
    if "style" in value or "coherence" in value:
        return [value]
    lists = [v for v in value.values() if isinstance(v, list)]
    if lists:
        return _entries(lists[0])
    return [dict(g, id=k) for k, g in value.items() if isinstance(g, dict)]


def server_ask(server, model: str, thinking: bool = THINKING) -> Callable[[str], str]:
    """Ask through a server's chat API (``server``: a client.Client), greedy, the reply's answer (not its thinking).
    A server that parses thinking returns it apart and the answer as the content; one that does not leaves the thought
    in the content, and when that was cut at the token budget it is a draft, no answer. A model that thinks anyway
    gets the thinking budget."""
    body = {"model": model, "max_tokens": MAX_TOKENS[bool(thinking) or thinks_anyway(model)], "temperature": 0,
            **thinking_fields(model, thinking)}

    def ask(prompt: str) -> str:
        d = server.post("/v1/chat/completions", dict(body, messages=[{"role": "user", "content": prompt}]), timeout=7200,
                        headers=UNTRACED)
        choice = (d.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        cut = bool(thinking) and choice.get("finish_reason") == "length" and not msg.get("reasoning_content")
        return answer_text(msg.get("content") or "", cut)
    return ask

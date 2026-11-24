"""Engines: what a measurement runs on, so profile, generate and learn do the same thing on MLX and on ds4.

A concept's measurements need four things of a model, and nothing else:

    residuals(prompts_ids, layers, sink, adds)   per prompt and layer, the mean residual from ``sink`` and the last
                                                 one: ``mean [n, L, W]``, ``last [n, L, W]``
    logprobs(ids, token_ids, adds)               the log-softmax at the last position, at ``token_ids`` only
    generate(ids, adds, max_tokens)              a greedy reply: ``(tokens, stopped)``, stopped = it ended on its own
    tok                                          ``encode(text)`` (no special tokens), ``chat(user, thinking=False)``
                                                 (the ids the engine itself would serve), ``decode(ids)``, ``eos``

``adds`` is ``{layer: scaled vector [W]}``, added to what that layer writes at every position, prompt and reply (the
steering header's ``positions=all``); None or empty runs unsteered. ``W`` is what a layer writes: ``hc * d_model``,
the hyper-connection streams flattened on models that have them (``shape``). On Inkling 2 a layer is a layer visit
and W the running sum; a concept's push spans its block (``block``) and the sweep tries its own layers
(``steer_layers``).

``next_logprobs(ids, start, lookup=True)`` reads ``ids`` once, teacher-forced, and gives each token from ``start`` on
its log-probability given the ones before it, and each of those rows' NEXT_TOP likeliest next tokens (``NextLp``);
``lookup=False`` leaves out what the model's n-gram lookup adds to the residual (Flash Next's PLE), at every position
(lookup.py: the dock's engram inspection). Only for a model with a lookup. Each engine can be called off mid-read
(``Stopped``): MLX between its chunks (``run``), ds4 by dropping its request (``stop``).

``MlxEngine.load`` loads a copy of the checkpoint in this process (engines/mlx.py), ``Ds4Engine.open(served)``
measures on the model ds4-server serves (engines/ds4.py). Either way the engine's ``checkpoint`` is the served name's checkpoint (models.py), whose vectors and profiles the
measurements read and write; the strength table a dose multiplies is the served name's own (``served``). Opening one
checks its shape against the checkpoint's (``checked``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np

# The texts that end a turn in the families metalscope serves (Qwen, DeepSeek V4); each engine adds the tokenizer's
# own eos id.
END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>")
SINK = 4                      # attention-sink positions left out of the mean residual
NEXT_TOP = 10                 # next_logprobs' likeliest next tokens a row


class NextLp(NamedTuple):
    """next_logprobs' answer: each row's log-probability of the token after it (``lp [n]``) and its NEXT_TOP likeliest
    next tokens, likeliest first (``ids [n, NEXT_TOP]``, their log-probabilities ``top``)."""
    lp: np.ndarray
    ids: np.ndarray
    top: np.ndarray


@dataclass(frozen=True)
class Shape:
    n_layers: int
    width: int                # W: what a layer writes, hc * d_model
    hc: int                   # hyper-connection streams (1 for a plain residual)
    d_model: int
    vocab: int


def hc_of(out_shape: tuple, d_model: int) -> tuple[int, int]:
    """``(W, hc)`` of a layer output ``[B, T, ...]``: W is the product of the dims after batch and token, so
    ``[B, T, D]``, ``[B, T, hc, D]`` and ``[B, T, hc*D]`` all read right, and hc = W // d_model."""
    width = int(math.prod(out_shape[2:]))
    if width % d_model:
        raise ValueError(f"a layer output {tuple(out_shape)} is not a whole number of {d_model}-wide streams")
    return width, width // d_model


class EngineUnavailable(RuntimeError):
    """The engine cannot be opened as asked (no server, another model served): said in one line, not a traceback."""


class Stopped(Exception):
    """A measurement called off before its end (lookup.py ``Run.stop``): it gives no answer."""


class Tokenizer:
    """The engine's tokenizer: ``encode``, ``chat``, ``decode``, ``eos`` (the module docstring)."""
    eos: set[int]

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def chat(self, user: str, thinking: Optional[bool] = False) -> list[int]:
        """One user turn and the generation prompt; ``thinking=None`` leaves the template's default."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError


class Engine:
    """The interface (module docstring). ``served`` is the served model's name, ``checkpoint`` the model it is a copy
    of (models.checkpoint_of): where the measurements are stored."""
    kind: str
    served: str
    checkpoint: str
    shape: Shape
    tok: Tokenizer
    block: Optional[int] = None                 # a push spans this many layers from it on (Inkling 2); None: one
    steer_layers: Optional[list[int]] = None    # the selection sweep's candidate steer layers; None: every 4th

    def residuals(self, prompts_ids: list[list[int]], layers: list[int], sink: int = SINK,
                  adds: Optional[dict[int, np.ndarray]] = None) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def logprobs(self, ids: list[int], token_ids: list[int], adds: Optional[dict[int, np.ndarray]] = None) -> np.ndarray:
        raise NotImplementedError

    def generate(self, ids: list[int], adds: Optional[dict[int, np.ndarray]], max_tokens: int) -> tuple[list[int], bool]:
        raise NotImplementedError

    def next_logprobs(self, ids: list[int], start: int, lookup: bool = True) -> NextLp:
        raise NotImplementedError(f"the {self.kind} engine reads no next-token log-probabilities")

    def close(self) -> None:
        pass


def checked(engine: Engine) -> Engine:
    """The engine, once its shape is its checkpoint's (models.check_shape: recorded the first time); otherwise closed,
    and EngineUnavailable saying which differs."""
    from .. import models
    s = engine.shape
    try:
        models.check_shape(engine.served, models.shape_of(s.n_layers, s.hc, s.d_model))
    except models.ShapeMismatch as exc:
        engine.close()
        raise EngineUnavailable(str(exc)) from None
    return engine


def samples(engine: Engine, texts: list[str], layers: list[int], position: str = "last", chat: bool = False,
            sink: int = SINK) -> dict[int, np.ndarray]:
    """``[n_texts, W]`` residual samples per layer: each text's last position, or its mean from ``sink``
    (``position="mean"``). ``chat`` wraps each text as one user turn, the template's thinking left as it is."""
    ids = [engine.tok.chat(t, thinking=None) if chat else engine.tok.encode(t) for t in texts]
    mean, last = engine.residuals(ids, list(layers), sink)
    X = last if position == "last" else mean
    return {l: X[:, j] for j, l in enumerate(layers)}

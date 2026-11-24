"""A model's vocabulary from its GGUF, enough to turn ids into text.

metalscope needs each recorded id's text (the sampled tokens, every top-k and
lens alternative) and which ids are special. ds4's GGUFs (Qwen3.8 Flash Next,
DeepSeek V4/V4.1) carry a GPT-2 style byte-level BPE vocabulary, so decoding
is mapping each token's characters back to bytes. The table is read once from
the GGUF header and cached under ~/.metalscope/ds4/.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .. import config

if TYPE_CHECKING:
    import numpy as np

CONTROL, USER_DEFINED = 3, 4      # tokenizer.ggml.token_type values that decode as their own text
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _byte_decoder() -> dict[str, int]:
    """GPT-2's bytes_to_unicode, inverted: printable bytes map to themselves,
    the rest to code points from 256 up."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def read_gguf_meta(path: Path, arrays: tuple[str, ...] = (), tensors: bool = False):
    """A GGUF header's metadata: every scalar and string, and the arrays named
    in ``arrays`` (others are skipped). ``tensors``: also the tensor table,
    ``{name: (dims, ggml type, absolute offset)}``, returned as (meta, table)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        _version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

        def string() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def value(t: int, keep: bool):
            if t == 8:
                return string()
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et == 8:
                    if keep:
                        return [string() for _ in range(n)]
                    for _ in range(n):                  # skipped unread: a vocabulary is ~150k strings
                        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
                    return None
                fmt = _SCALAR[et]
                if not keep:
                    f.seek(struct.calcsize(fmt) * n, 1)
                    return None
                return list(struct.unpack(f"<{n}{fmt[1]}", f.read(struct.calcsize(fmt) * n)))
            fmt = _SCALAR[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        meta = {}
        for _ in range(n_kv):
            key = string()
            (t,) = struct.unpack("<I", f.read(4))
            v = value(t, key in arrays)
            if v is not None:
                meta[key] = v
        if not tensors:
            return meta
        table = {}
        for _ in range(n_tensors):
            name = string()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            typ, off = struct.unpack("<IQ", f.read(12))
            table[name] = (dims, typ, off)
        align = int(meta.get("general.alignment", 32))
        start = -(-f.tell() // align) * align
        return meta, {k: (d, t, start + o) for k, (d, t, o) in table.items()}


GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30


def gguf_rows(path: Path, tensor: str, rows: list[int]) -> "np.ndarray":
    """Rows of a 2-D GGUF tensor (``[n_rows][row width]``, ggml's dims (width, n_rows)), dequantised to float32:
    the output head's rows of a few tokens without loading the model. F32, F16, BF16 and Q8_0."""
    import numpy as np
    _meta, table = read_gguf_meta(path, tensors=True)
    if tensor not in table:
        raise ValueError(f"{Path(path).name} has no tensor {tensor!r}")
    dims, typ, off = table[tensor]
    width = int(dims[0])
    if typ == GGML_Q8_0:
        per = width // 32 * 34                        # blocks of 32: an f16 scale, then 32 int8
    elif typ in (GGML_F16, GGML_BF16):
        per = 2 * width
    elif typ == GGML_F32:
        per = 4 * width
    else:
        raise ValueError(f"{tensor}: ggml type {typ} (metalscope reads F32, F16, BF16 and Q8_0 rows)")
    out = np.empty((len(rows), width), np.float32)
    with open(path, "rb") as f:
        for n, r in enumerate(rows):
            f.seek(off + int(r) * per)
            raw = f.read(per)
            if typ == GGML_Q8_0:
                b = np.frombuffer(raw, np.uint8).reshape(-1, 34)
                d = b[:, :2].copy().view(np.float16).astype(np.float32)          # [blocks, 1]
                q = b[:, 2:].copy().view(np.int8).astype(np.float32)             # [blocks, 32]
                out[n] = (q * d).reshape(-1)
            elif typ == GGML_F16:
                out[n] = np.frombuffer(raw, np.float16).astype(np.float32)
            elif typ == GGML_BF16:
                out[n] = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                out[n] = np.frombuffer(raw, np.float32)
    return out


def read_gguf_vocab(path: Path) -> tuple[list[str], list[int], str, list[str]]:
    """tokenizer.ggml.tokens, .token_type, .model and .merges from a GGUF's header."""
    meta = read_gguf_meta(path, ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"))
    tokens = meta.get("tokenizer.ggml.tokens")
    if tokens is None:
        raise ValueError(f"{path} has no tokenizer.ggml.tokens")
    return (tokens, meta.get("tokenizer.ggml.token_type") or [1] * len(tokens), meta.get("tokenizer.ggml.model", "gpt2"),
            meta.get("tokenizer.ggml.merges") or [])


# The GGUF architectures metalscope's ds4 capture covers, and the family each is (the capture file's names; models.TESTED).
FAMILIES = {"qwen4exp": "qwen4", "deepseek41": "deepseek41"}


def model_shape(path: Path) -> dict:
    """What metalscope needs of a ds4 model's shape, from its GGUF: family
    (FAMILIES), traced layers, streams, width, k, experts."""
    m = read_gguf_meta(path)
    arch = m.get("general.architecture")
    if arch == "qwen4exp":
        return dict(family=FAMILIES[arch], n_layer=int(m["qwen4exp.block_count"]) - int(m.get("qwen4exp.nextn_predict_layers", 0)),
                    n_hc=int(m["qwen4exp.hyper_connection.count"]), n_embd=int(m["qwen4exp.embedding_length"]),
                    k=int(m["qwen4exp.expert_used_count"]), n_expert=int(m["qwen4exp.expert_count"]))
    if arch == "deepseek41":
        return dict(family=FAMILIES[arch], n_layer=int(m["deepseek41.num_hidden_layers"]), n_hc=int(m["deepseek41.hc_mult"]),
                    n_embd=int(m["deepseek41.hidden_size"]), k=int(m["deepseek41.num_experts_per_tok"]),
                    n_expert=int(m["deepseek41.n_routed_experts"]))
    raise ValueError(f"{Path(path).name}: architecture {arch!r}; metalscope's ds4 capture covers Qwen3.8 "
                     f"(qwen4exp) and DeepSeek V4.1 (deepseek41)")


class GgufTokenizer:
    """What metalscope asks of a tokenizer: ``decode(ids)``, ``all_special_ids``
    and, with the GGUF's merges, ``encode(text)`` (a word's first token, for
    a word vector: byte-level BPE with the merges, no special tokens)."""

    def __init__(self, tokens: list[str], types: list[int], merges: list[str] | None = None):
        self._tokens, self._merges, self._bpe = tokens, merges, None
        dec = _byte_decoder()
        self._bytes: list[bytes] = []
        for tok, typ in zip(tokens, types):
            # Special and added tokens are stored as their text; the rest as byte-level characters.
            byte_level = typ not in (CONTROL, USER_DEFINED) and all(c in dec for c in tok)
            self._bytes.append(bytes(dec[c] for c in tok) if byte_level else tok.encode("utf-8"))
        self.all_special_ids = [i for i, typ in enumerate(types) if typ == CONTROL]

    def __len__(self) -> int:
        return len(self._bytes)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", "replace")

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """The tokens' bytes: a character can be split between two tokens (edit.py joins replies so)."""
        n = len(self._bytes)
        return b"".join(self._bytes[i] for i in (int(x) for x in ids) if 0 <= i < n)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not self._merges:
            raise ValueError("this GGUF tokenizer has no merges: it decodes only")
        if self._bpe is None:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers
            vocab = {t: i for i, t in enumerate(self._tokens)}
            bpe = Tokenizer(models.BPE(vocab, [tuple(m.split(" ", 1)) for m in self._merges], fuse_unk=False))
            bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
            bpe.decoder = decoders.ByteLevel()
            self._bpe = bpe
        return list(self._bpe.encode(text, add_special_tokens=False).ids)


def tokenizer_for(gguf: Path) -> GgufTokenizer:
    """The GGUF's tokenizer, cached by file identity (path, size, mtime)."""
    gguf = Path(gguf).resolve()
    st = gguf.stat()
    key = hashlib.sha1(f"{gguf}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:16]
    cache = config.home() / "ds4" / f"vocab-{key}.json"
    try:
        data = json.loads(cache.read_text())
        return GgufTokenizer(data["tokens"], data["types"], data["merges"])
    except (OSError, ValueError, KeyError):
        pass
    tokens, types, model, merges = read_gguf_vocab(gguf)
    if model != "gpt2":
        raise ValueError(f"{gguf.name}: tokenizer model {model!r}; metalscope's ds4 backend decodes byte-level BPE (gpt2)")
    config.atomic_write(cache, json.dumps(dict(gguf=str(gguf), tokens=tokens, types=types, merges=merges)))
    return GgufTokenizer(tokens, types, merges)

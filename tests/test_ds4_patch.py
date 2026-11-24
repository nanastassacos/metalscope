"""The ds4 patch and the Python side agree: the format version, the limits both sides hold, the family numbers, every
record tag the patch writes is one the reader handles (and the reader handles no tag the patch has stopped writing),
and each record's fields in the order the patch writes them. Read-only on the patch: only the lines it adds are
parsed, and only the names pinned here, so other additions to the patch do not matter.

The field order is checked by type (u32, an int32 or float array, raw bytes), and a STEP's only by its head and the
order of its flagged parts, so two fields of one type swapped inside a record, or inside a STEP part, pass; only a
built ds4 writing a file the reader reads would catch that."""

import ast
import re

import pytest

from metalscope import engines
from metalscope.ds4 import backend, build, frames
from metalscope.engines import ds4 as ds4_engine


def added_lines() -> str:
    return "\n".join(ln[1:] for ln in build.PATCH.read_text().splitlines() if ln.startswith("+") and not ln.startswith("+++"))


def defines(pattern: str) -> dict[str, int]:
    """The patch's ``#define NAME 64u`` whose NAME matches ``pattern``, by NAME."""
    return {n: int(v) for n, v in re.findall(rf"^\s*#\s*define\s+({pattern})\s+(\d+)u?\b", added_lines(), re.M)}


def written_tags(c: str) -> set[str]:
    return set(re.findall(r'\bscope_record\s*\([^;"]*?"([A-Z0-9_]{4})"', c))


def tree() -> ast.Module:
    return ast.parse(open(frames.__file__).read())


def function(name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(tree()) if isinstance(n, ast.FunctionDef) and n.name == name)


def tag_branches() -> dict[str, list[ast.stmt]]:
    """ScopeReader's branch for each tag it compares a record's tag with (``tag == b"STEP"``)."""
    return {c.value.decode(): n.body for n in ast.walk(tree()) if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == "tag"
            for c in n.test.comparators if isinstance(c, ast.Constant) and isinstance(c.value, bytes)}


@pytest.mark.parametrize("define, python", [
    pytest.param("DS4_SCOPE_VERSION", lambda: frames.VERSION, id="VERSION"),
    pytest.param("DS4_SCOPE_MAX_LENS", lambda: backend.MAX_LENS, id="MAX_LENS"),
    pytest.param("DS4_SCOPE_MAX_IDS", lambda: ds4_engine.MAX_IDS, id="MAX_IDS"),
    pytest.param("DS4_SCOPE_NX_TOP", lambda: engines.NEXT_TOP, id="NEXT_TOP"),
    pytest.param("DS4_SCOPE_MAX_REMOVE", lambda: frames.MAX_REMOVE, id="MAX_REMOVE"),
])
def test_patch_constant_is_pythons(define, python):
    got = defines(define)
    assert got, f"no #define {define} among {build.PATCH.name}'s added lines"
    assert got[define] == python()


FAMILY_NAMES = {"QWEN4": "qwen4", "DS41": "deepseek41"}   # the patch's DS4_SCOPE_FAMILY_<name>, frames.FAMILIES' name


def test_patch_families_are_the_readers():
    c = defines(r"DS4_SCOPE_FAMILY_\w+")
    assert {v: FAMILY_NAMES.get(n.removeprefix("DS4_SCOPE_FAMILY_"), n) for n, v in c.items()} == frames.FAMILIES


def test_patch_writes_the_tags_the_reader_reads():
    written, read = written_tags(added_lines()), set(tag_branches())
    assert written, f"no scope_record(..., \"TAG\", ...) call among {build.PATCH.name}'s added lines"
    assert written == read, f"written only: {sorted(written - read)}; read only: {sorted(read - written)}"


SIZEOF = {"int32_t": "i4[]", "uint32_t": "u4[]", "float": "f4[]"}
STRUCT = {"I": "u4", "i": "i4", "f": "f4"}


def c_fields(c: str, tag: str) -> list[list[str]]:
    """Each place the patch writes ``tag``: the scope_put calls right after its scope_record, as u4 (a u32),
    i4[] / u4[] / f4[] (an array, by its sizeof) or bytes (no sizeof)."""
    out = []
    for m in re.finditer(rf'\bscope_record\s*\([^;"]*"{tag}"[^;]*;\n', c):
        fields = []
        for ln in c[m.end():].splitlines():
            if not ln.strip().startswith("scope_put"):
                break
            size = re.search(r"sizeof\((\w+)\)", ln)
            fields.append("u4" if ln.strip().startswith("scope_put_u32") else SIZEOF.get(size[1], size[1]) if size else "bytes")
        out.append(fields)
    return out


def py_fields(body: list[ast.stmt]) -> list[str]:
    """What a reader branch reads, in source order, in c_fields' terms."""
    calls = sorted((n for s in body for n in ast.walk(s) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))
    out = []
    for call in calls:
        f = ast.unparse(call.func)
        if f == "struct.unpack_from":
            out += [STRUCT[ch] for k, ch in re.findall(r"(\d*)(\w)", call.args[0].value.lstrip("<")) for _ in range(int(k or 1))]
        elif f == "np.frombuffer":
            out.append(call.args[1].value.lstrip("<") + "[]")
        elif f == "bytes":
            out.append("bytes")
    return out


@pytest.mark.parametrize("tag", sorted(written_tags(added_lines()) - {"STEP"}))
def test_patch_writes_each_record_in_the_readers_order(tag):
    read = py_fields(tag_branches()[tag])
    for written in c_fields(added_lines(), tag):
        assert written == read


@pytest.mark.parametrize("array, reader, names", [
    pytest.param("head", "_header", {"DS4_SCOPE_VERSION": "version", "sc->family": "fam", "sc->n_layer": "L",
                                     "DS4_N_HC": "hc", "DS4_N_EMBD": "E", "DS4_N_EXPERT_USED": "k", "DS4_N_EXPERT": "ne",
                                     "DS4_N_VOCAB": "V", "sc->topk": "K", "sc->n_lens": "nl"}, id="header"),
    pytest.param("hdr", "_step", {"keep": "T"}, id="STEP"),
])
def test_patch_header_fields_are_in_the_readers_order(array, reader, names):
    """The file header and a STEP's head: the patch's ``const uint32_t head[] = {...}``, each entry by the name the
    reader unpacks it to (``names``; the same name where absent)."""
    m = re.search(rf"\bconst uint32_t {array}\[\]\s*=\s*\{{([^}}]*)\}}", added_lines())
    assert m, f"no const uint32_t {array}[] among {build.PATCH.name}'s added lines"
    c = [names.get(x.strip(), x.strip()) for x in m[1].split(",")]
    unpack = next(n for n in ast.walk(function(reader)) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
                  and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "struct.unpack_from")
    assert c == [t.id for t in unpack.targets[0].elts]


def test_patch_step_flags_and_their_order_are_the_readers():
    """A STEP's optional parts: the patch's flag bit for each (``(out ? 1u : 0u) | ...``) and the order its
    ``if (out) {`` blocks write them in, against the reader's ``x = take(..) if flags & OUT else ..`` in order."""
    c = added_lines()
    m = re.search(r"\bconst uint32_t flags\s*=([^;]*);", c)
    assert m, f"no const uint32_t flags among {build.PATCH.name}'s added lines"
    bit = {v: int(b) for v, b in re.findall(r"\((\w+)\s*\?\s*(\d+)u\s*:\s*0u\)", m[1])}
    writer = re.search(r'\bscope_record\s*\([^;"]*"STEP".*?^}', c, re.S | re.M)[0]
    written = [bit[v] for v in re.findall(r"^\s*if \((\w+)\) \{", writer, re.M) if v in bit]
    read = [getattr(frames, n.value.test.right.id) for n in function("_step").body if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.IfExp) and isinstance(n.value.test, ast.BinOp)
            and ast.unparse(n.value.test.left) == "flags"]
    assert sorted(written) == sorted(bit.values()), "a flag the STEP writer has no block for"
    assert written == read


# What each output of the staging kernel (ds4_gpu_scope_rows_sumsq_tensor(x, out, before, wout, ...)) holds, by its
# argument position, under the reader's name for the STEP block it becomes.
SUMSQ_OUTPUTS = {1: "stream", 3: "writes"}     # out: each stream's sum of squares; wout: what the layer changed in it


def test_patch_writes_the_stream_norms_then_the_writes_as_the_reader_takes_them():
    """A STEP's per-stream blocks: the staging each one is filled from (scope_resid's kernel call: its sum of squares,
    then what the layer changed), in the order the writer's ``squares[]`` puts them, every one written and counted in
    the record's size, against the reader's [L, T, hc] takes in order (``stream``, then ``writes``)."""
    c = added_lines()
    call = re.search(r"\bds4_gpu_scope_rows_sumsq_tensor\(R,([^;]*)\)", c)
    assert call, f"no ds4_gpu_scope_rows_sumsq_tensor(R, ...) among {build.PATCH.name}'s added lines"
    args = ["R"] + [a.strip() for a in call[1].split(",")]
    staged = {args[i].removeprefix("sc->"): name for i, name in SUMSQ_OUTPUTS.items()}
    writer = re.search(r'\bscope_record\s*\([^;"]*"STEP".*?^}', c, re.S | re.M)[0]
    m = re.search(r"\bconst float \*squares\[\]\s*=\s*\{([^}]*)\}", writer)
    assert m, "the STEP writer has no const float *squares[] = {...}"
    blocks = [staged.get(x.strip(), x.strip()) for x in m[1].split(",")]
    assert re.search(rf"for \(uint32_t s = 0; s < {len(blocks)}u; s\+\+\)", writer), "not every squares[] block is written"
    assert "squares[s]" in writer
    emit = re.search(r"^static void scope_emit_forward\(.*?^}", c, re.S | re.M)[0]      # the size is counted before the record
    size = re.search(r"uint64_t bytes = [^;]*\+ (\d+)u \* \(uint64_t\)L \* keep \* hc;", emit)
    assert size and int(size[1]) == 4 * len(blocks), "the record's size does not count every [L][T][hc] block"
    read = [n.targets[0].id for n in function("_step").body if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call) and ast.unparse(n.value) == "take('<f4', L, T, hc)"]
    assert blocks == read == list(SUMSQ_OUTPUTS.values())

"""The Pi dock's copies of Python constants (it cannot import them) equal their sources. Each is read from the
TypeScript by its declaration (``const NAME = ...;``) and compared with the Python value it mirrors."""

import re
from pathlib import Path

import pytest

import metalscope
from metalscope import backends, client, config, edit, lookup, models, steer, tally, vectors
from metalscope.learn import decide
from metalscope.learn import job, lens_job, plan, window

PI = Path(metalscope.__file__).parent / "frontends" / "pi"


def ts(file: str, name: str) -> str:
    """The right-hand side of ``const NAME = ...;`` in a dock file (one line or several)."""
    m = re.search(rf"\bconst {name}\s*=\s*(.+?);[ \t]*(?://[^\n]*)?$", (PI / file).read_text(), re.S | re.M)
    assert m, f"no const {name} in {file}"
    return m.group(1)


def strings(s: str) -> list[str]:
    return re.findall(r'"((?:[^"\\]|\\.)*)"', s)


def number(s: str) -> float:
    return float(re.match(r"\s*(-?[\d.]+)", s).group(1))


def gpu_phases() -> set[str]:
    """The phases from a job's GPU window opening (UNLOAD, HOLD) to its closing (RELOAD, FREE), both included."""
    out = set()
    for kind in [*vars(job).values(), *vars(lens_job).values()]:
        for steps in (getattr(kind, "STEPS", None) or (), getattr(kind, "DS4_STEPS", None) or ()):
            opens = [i for i, s in enumerate(steps) if s in (window.UNLOAD, window.HOLD)]
            closes = [i for i, s in enumerate(steps) if s in (window.RELOAD, window.FREE)]
            if opens and closes:
                out |= {p for s in steps[opens[0]:closes[-1] + 1] for p in s[2]}
    return out


def reserved_names() -> set[str]:
    rhs = ts("concept.ts", "RESERVED_NAMES")
    spread = set(strings(ts("concept.ts", "CONCEPT_SUBCOMMANDS"))) if "...CONCEPT_SUBCOMMANDS" in rhs else set()
    return set(strings(rhs)) | spread


def chat_model_skip() -> set[str]:
    m = re.search(r"/([\w|]+)/i\.test\(m\.id\)", (PI / "provider.ts").read_text())
    assert m, "no /a|b/i.test(m.id) model filter in provider.ts"
    return set(m.group(1).split("|"))


@pytest.mark.parametrize("dock, python", [
    pytest.param(lambda: ts("args.ts", "NAME"), lambda: f"/{models.NAME_RE.pattern}/", id="NAME"),
    pytest.param(lambda: ts("steerpage.ts", "MODEL_NAME"), lambda: f"/{models.NAME_RE.pattern}/".replace("\\-", "-"),
                 id="MODEL_NAME"),
    pytest.param(lambda: number(ts("args.ts", "NAME_MAX")), lambda: vectors.NAME_MAX, id="NAME_MAX"),
    pytest.param(lambda: number(ts("args.ts", "MAX_STRENGTH")), lambda: steer.MAX_STRENGTH, id="MAX_STRENGTH"),
    pytest.param(lambda: number(ts("steer.ts", "MAX_PARTS")), lambda: steer.MAX_PARTS, id="MAX_PARTS"),
    pytest.param(reserved_names, lambda: plan.RESERVED_NAMES, id="RESERVED_NAMES"),
    pytest.param(lambda: strings(ts("state.ts", "TOOL_OPENERS")), lambda: edit.TOOL_OPENERS, id="TOOL_OPENERS"),
    pytest.param(lambda: number(ts("lookup.ts", "HELPED")), lambda: lookup.HELPED, id="HELPED"),
    pytest.param(lambda: strings(ts("jobs.ts", "CANT_STEER")), lambda: [decide.CANT_STEER], id="CANT_STEER"),
    pytest.param(lambda: number(ts("dock.ts", "LENS_OFF")), lambda: config.LENS_OFF, id="LENS_OFF"),
    pytest.param(lambda: number(ts("expert.ts", "K_TOP")), lambda: tally.K_TOP, id="K_TOP"),
    pytest.param(lambda: set(strings(ts("jobs.ts", "GPU_PHASES"))), gpu_phases, id="GPU_PHASES"),
    pytest.param(chat_model_skip, lambda: set(backends.CHAT_MODEL_SKIP), id="CHAT_MODEL_SKIP"),
    pytest.param(lambda: strings(ts("steer.ts", "WORD_EDIT_ONLY")), lambda: [steer.WORD_EDIT_ONLY], id="WORD_EDIT_ONLY"),
    pytest.param(lambda: number(ts("concept.ts", "CALIBRATION_VERSION")), lambda: models.CALIBRATION_VERSION,
                 id="CALIBRATION_VERSION"),
    pytest.param(lambda: strings(ts("provider.ts", "DS4_URL")), lambda: [client.DS4_URL], id="DS4_URL"),
])
def test_dock_copy_equals_python(dock, python):
    assert dock() == python()


def test_dock_headers_are_pythons():
    """Every X-Scope-* header the dock sends is one config.py names (headers are case-insensitive)."""
    dock = {h.lower() for f in PI.glob("*.ts") for h in re.findall(r"X-Scope-[\w-]+", f.read_text())}
    assert dock and dock <= {v for k, v in vars(config).items() if k.endswith("_HEADER") or k == "HEADER"}

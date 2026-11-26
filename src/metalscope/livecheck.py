"""``metalscope doctor --live``: a served model end to end, on each backend that serves one.

Short completions of one prompt, traced under the client ``doctor`` with the live feed asked for what the dock asks
for (routing, the lens top 10s, the layer writes): plain, then steered by one of the model's concepts, then the plain
reply edited at one token (``X-Scope-Edit``, a word removed there over the server's ``edit_band``) where the server
lists ``edit_at``, and the plain reply's n-gram lookup measured (the tokens it helped most) where it lists ``lookup``.
They show the trace saved whole (no ``meta.partial``), what the dock draws from it (both lenses' rows, the MAP's
per-token probabilities, the expert routing on the feed and from the saved trace, RESIDUAL STREAM's layer writes),
LENS's steps and the EXPERTS pages' tally routes, every steer applied (not FAILED or PARTIAL), and the edit's one
stream and two traces. These are the parts an engine update breaks first, oMLX's patched internals or ds4's capture,
so this is the check after either moves (docs/backends.md, "Updating the engines"). Each backend ends with a summary
line; two backends add a feature-by-engine table.
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.parse
import zipfile
from pathlib import Path
from typing import Optional

from . import client, config, models
from .store import read_index

PROMPT = "Describe a tiger in the rain in two sentences."
CLIENT = "doctor"
MAX_TOKENS = 24
DOSE = 0.3          # a concept without a chat calibration: a dose that changes most replies
WORDS = ("tiger", "lion", "Asia", "Africa", "America", "jaguar")   # one-token words with vectors here already
EDIT_AT = 8         # the plain reply's token the edit answers again from
FEATURES = ("traced reply", "live feed", "trace saved", "lens rows", "token probabilities", "expert routing",
            "residual stream", "lens layers", "experts pages", "concept steer", "one-token edit", "lookup")


def pick_concept(concepts: dict) -> Optional[tuple[str, int, float]]:
    """(concept, reader layer, dose) to steer with, from /scope/api/concepts: a calibrated concept at its default dose
    first, else any with a reader at DOSE; None when the model has no profiled concept."""
    best = None
    for c in concepts.get("concepts") or []:
        layer = (c.get("reader") or {}).get("layer")
        if layer is None:
            continue
        s = c.get("steer") or {}
        if s.get("basis") == "chat" and s.get("strength"):        # a calibration fitting the reader layer
            return c["name"], int(s.get("layer", layer)), float(s["strength"])
        best = best or (c["name"], int(layer), DOSE)
    return best


def pick_words(have: dict, n: int = 2) -> list[str]:
    """The ``n`` words to edit by (the first; the second a spare), from ``have`` (word: its vector.json, None without a vector, False for a
    concept's name): one-token vectors first, then words with none (made then), in order; a word cut to its first
    token (``cut``) or a concept is passed over."""
    ready = [w for w, m in have.items() if isinstance(m, dict) and not m.get("cut")]
    return (ready + [w for w, m in have.items() if m is None])[:n]


VERDICTS = {"passed": config.MARKS["ok"][1], "failed": config.MARKS["fail"][1], "not checked": config.MARKS["info"][1]}


def verdict(goods: list) -> str:
    """A feature's findings as VERDICTS words it: failed when any failed, passed when one passed, else not checked."""
    return "failed" if False in goods else "passed" if True in goods else "not checked"


def summary(engine: str, found: dict) -> str:
    v = [verdict(found.get(f, [])) for f in FEATURES]
    return " · ".join([engine] + [f"{v.count(w)} {w}" for w in VERDICTS if v.count(w)])


def table(results: list[tuple[str, dict]]) -> list[str]:
    """The features by engine (``results``: each engine's findings, as ``check`` fills them)."""
    w = max(map(len, FEATURES)) + 4
    cell = lambda v: config.paint(v, VERDICTS[v]) + " " * (13 - len(v))
    return [config.paint(f"{'Live':<{w}}" + "".join(f"{e:<13}" for e, _ in results).rstrip(), "1")] + [
        f"  {f:<{w - 2}}" + "".join(cell(verdict(found.get(f, []))) for _, found in results).rstrip() for f in FEATURES]


def _traces_since(t0: float, model: str, client: str = CLIENT) -> list[dict]:
    out = []
    for e in read_index():
        if e.get("time", 0) < t0:
            break
        if e.get("client") == client and config.short_name(e.get("model") or "") == config.short_name(model):
            out.append(e)
    return out


def saved_traces(t0: float, model: str, n: int = 1, wait: float = 20, client: str = CLIENT) -> list[dict]:
    """``client``'s traces saved since ``t0``, newest first, once there are ``n`` (what there is after ``wait`` s)."""
    end = time.time() + wait
    while len(got := _traces_since(t0, model, client)) < n and time.time() < end:
        time.sleep(0.5)
    return got


def trace_meta(entry: dict) -> dict:
    return config.read_json(Path(entry["path"]) / "meta.json", {}) if "path" in entry else {}


def edit_steer(pick: Optional[tuple], words: list, band) -> Optional[str]:
    """The steer the live check and bench edit with: the first word removed over the server's edit band (``band``,
    /scope/api/lens/layers' ``edit_band``, as the dock reads it) where a word vector is, else ``pick``'s concept
    (pick_concept) at its dose; None with neither."""
    if words and band:
        return f"remove~{words[0]}/{band[0]}-{band[1]}"
    return f"{pick[0]}@{pick[2]:g}/{pick[1]}" if pick else None


def restore(set_step, was, log) -> dict:
    """The lens step put back, tried once more if ctrl+c cuts it short; what could not be, said."""
    if was is None:
        return {}
    why = None
    for _ in range(2):
        try:
            set_step(was)
            return dict(lens_step=was)
        except KeyboardInterrupt:
            why = "interrupted"
        except client.ServerError as exc:
            why = str(exc)
            break
    log("  " + config.mark("warn", f"could not put the lens back to {config.lens_name(was)} ({why}): set it with - + in the dock's LENS"))
    return dict(lens_step=f"not put back to {config.lens_name(was)} ({why})")


def _bare(s: str) -> str:
    """A reply's text with no whitespace or thinking markers: a chat stream drops those the raw tokens have."""
    return re.sub(r"\s+|</?think>", "", s)


def check(server: client.Client, log=print, found: Optional[dict] = None) -> bool:
    """The completions on ``server``; one line per finding, each kept under its feature in ``found``. True when
    nothing failed."""
    status = server.status() or {}
    model = config.short_name(server.served()[0])    # oMLX lists a model by its path; requests name its id
    q = urllib.parse.quote(model)
    found = {} if found is None else found
    ok = True

    def say(good: Optional[bool], text: str, feature: Optional[str] = None) -> None:
        nonlocal ok
        ok = ok and good is not False
        if feature:
            found.setdefault(feature, []).append(good)
        log("  " + config.mark("ok" if good else "info" if good is None else "fail", text))

    def steered(spec: str, feature: str, what: str) -> None:
        t = time.time()
        try:
            out = srv.complete(model, PROMPT, spec, MAX_TOKENS, timeout=300, temperature=0.0)
        except client.ServerError as exc:
            return say(False, f"{what} {spec}: refused or failed: {exc}", feature)
        entry = (saved_traces(t, model) or [None])[0]
        label = (entry or {}).get("steer") or ""
        differs = f"differs: {out['text'][:50]!r}" if out["text"] != plain["text"] else "is the same as the plain one"
        say(entry is not None and "FAILED" not in label and "PARTIAL" not in label, f"{what} {spec}: "
            + (f"applied ({label})" if entry else "no trace saved") + f"; the reply {differs}", feature)

    log("")
    log(config.paint(f"Live · {status.get('engine')} · {model}", "1"))
    log("  " + config.mark("info", f"{server.url} · caps: {', '.join(server.caps()) or 'none'}"))
    srv = client.Session(server, CLIENT, experts=True, lens=True, writes=True)
    set_step, off = lambda k: server.post("/scope/api/lens/layers", dict(step=k, model=model)), False
    try:
        try:   # the lenses off on this server: checked at the default step, then off again (restore)
            if server.get(f"/scope/api/lens/layers?model={q}").get("at") == config.LENS_OFF:
                set_step(config.LENS_STEP_DEFAULT)
                off = True
                say(None, f"the lenses are off here: checked at {config.lens_name(config.LENS_STEP_DEFAULT)}, then off again")
        except client.ServerError:
            pass                                      # the lens layers' check below says why
        if not srv.alive:
            say(False, "the live feed (/scope/api/live) did not connect", "live feed")
        t0 = time.time()
        try:
            plain = srv.complete(model, PROMPT, "", MAX_TOKENS, timeout=300, temperature=0.0)
        except client.ServerError as exc:
            say(False, f"a traced completion failed: {exc}", "traced reply")
            return False
        say(bool(plain["text"]), f"traced reply in {plain['seconds']}s: {plain['text'][:60]!r}", "traced reply")
        evs, n = plain["events"], len(plain["tok_text"])
        say(n > 0, f"{n} tokens on the live feed", "live feed")
        entry = (saved_traces(t0, model) or [None])[0]
        if entry is None:
            say(False, "no trace saved for it", "trace saved")
        else:
            partial = trace_meta(entry).get("partial")
            say(not partial, f"trace {entry['id']} saved" + (f", partial: {'; '.join(partial)}" if partial else ""),
                "trace saved")

        # What the dock draws: both lenses' rows and top 10s, the MAP's probabilities, the routing (feed and trace).
        jlens = any(e.get("lens_by") == "jlens" for e in evs)
        if jlens:
            lenses = (("J-lens", "lens", "lens10"), ("logit lens", "lens_logit", "lens10_logit"))
        else:
            say(None, "no J-lens on this model: its rows not checked", "lens rows")
            lenses = (("logit lens", "lens", "lens10"),)
        for name, rows, tops in lenses:
            got = [e for e in evs if e.get(rows)]
            top = sum(1 for e in got if e.get(tops))
            say(n > 0 and len(got) == top == n, f"{name} rows on {len(got)}/{n} tokens "
                f"({len(got[0][rows]) if got else 0} layers), top 10s on {top}", "lens rows")
        p = [x for x in plain["p"] if x is not None]
        say(n > 0 and len(p) == n, f"token probabilities on {len(p)}/{n} tokens (the MAP)", "token probabilities")
        L = _routing(server, evs, entry, say)
        _stream(evs, entry, say)

        try:
            lens = server.get(f"/scope/api/lens/layers?model={q}")
        except client.ServerError as exc:
            lens = say(False, f"lens layers: {exc}", "lens layers") or {}
        steps, band = lens.get("steps") or [], lens.get("edit_band")
        if steps:
            sizes = ", ".join(f"{s.get('name')} {s.get('n')}" for s in steps)
            say(all(s.get("layers") for s in steps), f"lens layers: {sizes}", "lens layers")
        if L:
            _experts(server, q, L, say)
        _lookup(server, entry, say, log)

        try:
            concepts = server.get(f"/scope/api/concepts?model={q}")
        except client.ServerError:
            concepts = {}
        if not concepts.get("steer_enabled"):
            say(None, "steering is off (`metalscope steering on`): steers and the edit not checked")
            return ok
        if (pick := pick_concept(concepts)) is None:
            say(None, f"no concept profiled for {model}: the concept steer not checked", "concept steer")
        else:
            name, layer, dose = pick
            steered(f"{name}@{dose:g}/{layer}", "concept steer", "steered by")
        words = word_vectors(server, model, say, jlens)
        edit = edit_steer(pick, words, band)
        if "edit_at" not in server.caps():
            say(None, "the server lists no edit_at: the one-token edit not checked", "one-token edit")
        elif entry is not None and edit and n > 2:
            _edit(server, model, plain, entry, edit, say, edit.startswith("remove~") and words[0])
    finally:
        srv.close()
        if off:
            restore(set_step, config.LENS_OFF, log)
    return ok


def _routing(server, evs: list, entry: Optional[dict], say) -> int:
    """The routing on the feed and from the saved trace (/scope/api/traces/ID/routing, the dock's after a reply);
    its MoE layers, 0 for a dense model."""
    live = sum(1 for e in evs if e.get("experts"))
    try:
        r = server.get(f"/scope/api/traces/{entry['id']}/routing?last=64") if entry else {}
    except client.ServerError as exc:
        return say(False, f"routing from the saved trace: {exc}", "expert routing") or 0
    L, k, m = r.get("L") or 0, r.get("k") or 0, r.get("n") or 0
    if not L and not live:
        return say(None, "no experts (a dense model): routing and the EXPERTS pages not checked", "expert routing") or 0
    whole = m > 0 and len(base64.b64decode(r.get("ids") or "")) == m * L * k * 2
    # A token the feed sent without routing is not a failure: the dock reads its routing from the trace.
    say(live > 0 and whole, f"routing on {live}/{len(evs)} tokens of the feed; {m} tokens x {L} layers x top "
        f"{k} from the saved trace", "expert routing")
    return L


def _stream(evs: list, entry: Optional[dict], say) -> None:
    """RESIDUAL STREAM's feed: each layer's writes on every token in one shape, the layers' shares beside them on every
    token or on none (a model with reads has none). Not checked where neither the feed nor the saved trace has
    writes."""
    live = [e for e in evs if e.get("writes")]
    try:
        with zipfile.ZipFile(Path(entry["path"]) / "arrays.npz") as z:
            saved = any(n.startswith("resid_write_norm/") for n in z.namelist())
    except (TypeError, KeyError, OSError, zipfile.BadZipFile):
        saved = False
    if not saved and not live:
        return say(None, "no layer writes (a model that records none): RESIDUAL STREAM not checked", "residual stream")
    shapes = {(len(e["writes"]), e.get("write_k") or 0, len(e.get("shares") or ())) for e in live}
    w, k, s = next(iter(shapes)) if shapes else (0, 0, 0)
    L = w // k if k else 0
    shape = f"in {len(shapes)} shapes" if len(shapes) > 1 else \
        f"({L} layers x {k} streams), " + ("shares beside them" if s else "no shares (no layers view)")
    say(len(live) == len(evs) and len(shapes) == 1 and L > 0 and s in (0, L),
        f"layer writes on {len(live)}/{len(evs)} tokens of the feed {shape}", "residual stream")


def _experts(server, q: str, L: int, say) -> None:
    """The EXPERTS pages' tally routes at the middle MoE layer: its specialists, then the first one's page."""
    layer = L // 2
    try:
        sp = server.get(f"/scope/api/experts/specialists?layer={layer}&model={q}")
        if not sp.get("experts"):
            return say(None, f"no expert tally at layer {layer} yet: the EXPERTS pages not checked", "experts pages")
        e = sp["experts"][0]["e"]
        one = server.get(f"/scope/api/experts/expert?layer={layer}&e={e}&model={q}")
    except client.ServerError as exc:
        return say(False, f"the experts routes: {exc}", "experts pages")
    top = [t[0] for t in one.get("top") or []][:3]
    say(bool(one.get("n")) and bool(top), f"experts at layer {layer}: {len(sp['experts'])} specialists over "
        f"{sp.get('tokens')} tokens; expert {e}: {one.get('n')} tokens, top {top}", "experts pages")


def _lookup(server, entry: Optional[dict], say, log) -> None:
    """The plain reply read again without the n-gram lookup (/scope/api/lookup), where the server lists ``lookup``:
    the tokens it helped most, each after the n-gram it looked up."""
    from .lookup import table
    if "lookup" not in server.caps():
        return say(None, "no n-gram lookup on this model: not checked", "lookup")
    if entry is None:
        return say(False, "no saved reply to read again", "lookup")
    t = time.time()
    try:
        got = server.post("/scope/api/lookup", dict(trace=entry["id"]), timeout=900)
    except client.ServerError as exc:
        return say(False, f"the lookup: {exc}", "lookup")
    n = len(got.get("ids") or [])
    tops = len(got.get("top_ids_on") or []) == len(got.get("top_ids_off") or []) == n
    say(n > 0 and len(got.get("lp_on") or []) == len(got.get("lp_off") or []) == n and tops,
        f"lookup read {n} reply tokens again in {time.time() - t:.1f}s" + ("" if tops else ", without the top choices"), "lookup")
    for line in table(got, trace_meta(entry).get("token_text") or []):
        log(config.paint("    " + line, "2"))


def word_vectors(server, model: str, say, jlens: bool) -> list[str]:
    """One-token words with vectors for the one-token edit (WORDS; pick_words), made by /scope/api/steer/words when
    missing, as the dock's edit makes them; none without a J-lens (``jlens``), which a word vector needs."""
    feature = "one-token edit"
    if not jlens:
        say(None, "no J-lens on this model: no word to edit with", feature)
        return []
    have = {w: False if models.profile_path(model, w).exists() else
            config.read_json(models.vector_path(model, w).with_suffix(".json")) for w in WORDS}
    words = pick_words(have)
    for w in [w for w in words if have[w] is None]:
        try:
            made = server.post("/scope/api/steer/words", dict(model=model, name=w, words=[w]))
        except client.ServerError as exc:
            say(False, f"a word vector for {w} could not be made: {exc}", feature)
            words.remove(w)
            continue
        say(None, f"made a word vector for {w}" + (": more than one token, not used" if made.get("cut") else ""))
        if made.get("cut"):
            words.remove(w)
    if not words:
        say(None, "no one-token word vector: no word to edit with", feature)
    return words


def _edit(server, model: str, plain: dict, entry: dict, steer: str, say, word=None) -> None:
    """The plain reply answered again from token ``at`` with ``steer`` there only, as the dock's word edit sends it
    (at the removed ``word`` where the reply says it, else EDIT_AT): one chat stream whose text keeps the reply's up
    to ``at``, and two traces saved naming the edit."""
    toks = plain["tok_text"]
    at = next((i for i, t in enumerate(toks[1:-1], 1) if word and t.strip() == word), min(EDIT_AT, len(toks) - 2))
    orig = trace_meta(entry)
    rid = orig.get("request_id") or plain["rid"]
    body = dict(model=model, messages=[dict(role="user", content=PROMPT)], stream=True, max_tokens=MAX_TOKENS,
                temperature=0.0)
    t = time.time()
    try:
        text = server.request("POST", "/v1/chat/completions", body, timeout=300, as_text=True, headers={
            config.HEADER: "on", config.CLIENT_HEADER: CLIENT, config.EDIT_HEADER: f"{rid}@{at} {steer}"})
    except client.ServerError as exc:
        if exc.status == 400 and "cannot edit this model" in (exc.detail or ""):     # omlx/patches.py NO_EDIT
            return say(None, f"the edit: {exc.detail}", "one-token edit")
        return say(False, f"the edit at token {at} ({steer}): {exc}", "one-token edit")
    evs = [json.loads(x[5:]) for x in text.splitlines() if x.startswith("data:") and x[5:].strip().startswith("{")]
    said = "".join(v for e in evs for c in e.get("choices") or [] for k, v in (c.get("delta") or {}).items()
                   if k in ("reasoning_content", "content"))
    one = text.count("data: [DONE]") == 1 and not any(e.get("error") for e in evs)
    kept = "".join(toks[:at])
    same = _bare(said).startswith(_bare(kept))
    say(one and same, f"edit at token {at} ({steer}): {'one stream' if one else 'not one stream'}, the text before "
        + ("it kept" if same else f"it changed: {said[:60]!r} for {kept!r}"), "one-token edit")
    saved = [e for e in saved_traces(t, model, 2) if trace_meta(e).get("edit")]
    a = next((m for m in map(trace_meta, saved) if (m.get("edit") or {}).get("of") == rid), {})
    ids = a.get("tokens") == (orig.get("tokens") or [])[:orig.get("prompt_len", 0) + at]
    say(len(saved) == 2 and (a.get("edit") or {}).get("at") == at and ids, f"{len(saved)} traces saved naming the "
        f"edit; the reply's ids before it {'kept' if ids else 'differ'}", "one-token edit")
    was, now = toks[at], a.get("last_token_text")
    say(None, f"token {at}: {was!r} " + (f"became {now!r}" if now != was else "unchanged"))


def run(log=print, results: Optional[list] = None) -> bool:
    """``check`` on each backend serving a model, a section each ending with its tally, then the features by engine
    where there are two; each engine's findings appended to ``results``. True when nothing failed."""
    found = client.serving()
    if not found:
        log("")
        log(config.paint("Live", "1"))
        log("  " + config.mark("info", "no backend serves a model: start oMLX with one, or metalscope ds4 serve MODEL.gguf"))
        return True
    ok, results = True, [] if results is None else results
    for server in found:
        res: dict = {}
        ok = check(server, log, res) and ok
        results.append(((server.status() or {}).get("engine") or server.url, res))
        log("  " + summary(*results[-1]))
    if len(results) > 1:
        log("")
        for line in table(results):
            log(line)
    return ok

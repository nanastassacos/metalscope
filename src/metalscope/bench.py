"""``metalscope bench``: how fast the model one backend serves now is, traced and not.

Chat requests as Pi sends them (streamed, under the client ``bench``, each saying whether to trace it: the header
overrides the global mode, which bench leaves alone), one prompt at temperature 0 capped at ``max_tokens``: the
writing speed untraced, then traced with the lenses off and at each lens step (how many layers the lenses read);
the dock's one-token edit, timed to its new token and to its end (``edit_at``); and the engram inspection of a short
reply after conversations of several lengths (``lookup``). Writing speed is the gaps between streamed tokens over
the time from the first piece of text to the last, so reading the prompt is left out. The Mac slows itself under
sustained load, so each measurement starts after a rest (``rest``) and tracing off is measured again after the lens steps.
Best effort: it refuses a server tracing a request when it starts, but other requests meanwhile skew the numbers. It
puts back the lens step, also on an error or ctrl+c.
"""

from __future__ import annotations

import json
import platform
import random
import statistics
import subprocess
import time
import urllib.parse
from typing import Optional

from . import client, config, livecheck

CLIENT = "bench"
PROMPT = ("Write a long, detailed story about a lighthouse keeper who finds a message in a bottle. Describe the "
          "island, the weather, the keeper's routine and every person in the story. At least 1500 words.")
MAX_TOKENS = 300
REPEATS = 3
REST = 60                   # seconds idle before each measurement: the Mac slows itself under sustained load
CONTEXTS = (1000, 4000, 16000)
SHORT_REPLY = 32            # the reply the engram inspection reads after each conversation
BAR = 16                    # the writing speed's bars at the fastest: a row stays under 80 columns
TEXT = ("content", "reasoning_content", "reasoning")
ASK_AGAIN = "ask again"     # oMLX's 409 when it gave back expert slots during the reading (omlx/patches.py): once more
NOUNS = ("river stone lamp window garden letter harbor bridge candle market valley engine orchard tower ladder "
         "ribbon kettle compass meadow forest island anchor lantern violin pebble harvest canyon").split()
VERBS = "carried painted followed counted repaired measured noticed watched gathered folded opened traded".split()
ADJS = "quiet old narrow bright heavy distant careful yellow broken patient silver early wooden".split()


def chat(server: client.Client, model: str, messages: list, max_tokens: int, traced: bool,
         headers: Optional[dict] = None) -> dict:
    """One streamed chat completion: when its pieces of text came (seconds after sending), its token counts."""
    body = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0.0, stream=True,
                stream_options=dict(include_usage=True))
    hdr = {config.HEADER: "on" if traced else "off", config.CLIENT_HEADER: CLIENT, **(headers or {})}
    times, usage, finish, t = [], {}, None, 0.0
    for t, ev in server.stream("/v1/chat/completions", body, timeout=3600, headers=hdr):
        if ev.get("error"):
            err = ev["error"]
            err = err.get("message", err) if isinstance(err, dict) else err
            raise client.ServerError(f"the reply failed: {err}", None, str(err))
        usage = ev.get("usage") or usage
        for c in ev.get("choices") or []:
            if any(v for k, v in (c.get("delta") or {}).items() if k in TEXT):
                times.append(round(t, 4))
            finish = c.get("finish_reason") or finish
    return dict(seconds=round(t, 4), first_s=times[0] if times else None, last_s=times[-1] if times else None,
                pieces=len(times), tokens=usage.get("completion_tokens"), prompt_tokens=usage.get("prompt_tokens"),
                finish=finish)


def tps(run: dict) -> Optional[float]:
    """Writing speed: the gaps between the reply's streamed tokens over the time from its first piece of text to its
    last. A reply cut at max_tokens ("length") streams all its tokens; one that stopped, all but its stop token."""
    n = (run.get("tokens") or run.get("pieces") or 0) - (1 if run.get("finish") == "length" else 2)
    dt = (run.get("last_s") or 0) - (run.get("first_s") or 0)
    return round(n / dt, 1) if n > 0 and dt > 0 else None


def median(xs: list) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def filler(n_chars: int, seed: int = 0) -> str:
    """Deterministic notes to fill a conversation, ``n_chars`` long or a sentence more."""
    rng, out, size, i = random.Random(seed), [], 0, 0
    while size < n_chars:
        i += 1
        s = (f"Note {i}: the {rng.choice(ADJS)} {rng.choice(NOUNS)} {rng.choice(VERBS)} the {rng.choice(ADJS)} "
             f"{rng.choice(NOUNS)} near the {rng.choice(NOUNS)}, and the {rng.choice(NOUNS)} was {rng.choice(ADJS)}.")
        out.append(s)
        size += len(s) + 1
    return " ".join(out)


def conversation(n_chars: int, turn: int = 2000) -> list[dict]:
    """A conversation of ``n_chars`` of notes in turns of ``turn``, then a question with a short answer."""
    text, msgs = filler(n_chars), []
    for i in range(0, len(text), turn):
        msgs += [dict(role="user", content="More notes:\n" + text[i:i + turn]), dict(role="assistant", content="Noted.")]
    return msgs + [dict(role="user", content="Which object did the last note mention first? Answer in a few words.")]


def machine() -> str:
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    mem = sysctl("hw.memsize")
    return ", ".join(x for x in (sysctl("machdep.cpu.brand_string") or platform.machine(),
                                 f"{int(mem) // 2**30} GB" if mem.isdigit() else "",
                                 f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else "") if x)


def pick(backend: Optional[str], model: Optional[str] = None) -> tuple[client.Client, str]:
    """The server to measure and the model on it (``model``, else the first it serves)."""
    found = [c for c in client.serving() if backend in (None, (c.status() or {}).get("engine"))]
    if model:
        found = [c for c in found if config.short_name(model) in map(config.short_name, c.served())]
    if not found:
        raise SystemExit(f"no {backend or 'backend'} serves {model or 'a model'}: start one first (metalscope status)")
    if len(found) > 1:
        raise SystemExit("both backends serve a model: pick one with --backend omlx or --backend ds4")
    return found[0], config.short_name(model or found[0].served()[0])


def run(server: client.Client, res: dict, repeats: int = REPEATS, contexts=CONTEXTS, max_tokens: int = MAX_TOKENS,
        log=print, model: Optional[str] = None, rest: float = REST) -> dict:
    """The measurements into ``res`` (kept as far as they got on an error), each after ``rest`` seconds idle; the
    lens step put back."""
    st = server.status(fresh=True) or {}
    if st.get("active"):
        raise SystemExit(f"{st.get('engine')} at {server.url} is tracing {st['active']} request(s): bench needs it "
                         "idle; try again when they are done")
    pause = lambda: rest and time.sleep(rest)
    if rest:
        log(config.paint(f"  each measurement after {rest:g} s idle", "2"))
    model = config.short_name(model or server.served()[0])
    q = urllib.parse.quote(model)
    lens = server.get(f"/scope/api/lens/layers?model={q}")
    res.update(model=model, engine=st.get("engine"), url=server.url, machine=machine(), caps=server.caps(),
               when=time.strftime("%Y-%m-%d %H:%M"), prompt=PROMPT, max_tokens=max_tokens, repeats=repeats,
               rest=rest, was=dict(lens_step=lens.get("at")), speed=[])
    set_step = lambda k: server.post("/scope/api/lens/layers", dict(step=k, model=model))
    changed = False
    try:
        ask = [dict(role="user", content=PROMPT)]

        def timed(traced: bool) -> dict:
            """One reply; traced, its trace waited for (its save must not overlap the next reply) and read."""
            t = time.time()
            r = chat(server, model, ask, max_tokens, traced)
            if traced:
                entry = (livecheck.saved_traces(t, model, client=CLIENT) or [{}])[0]
                r.update(trace=entry.get("id"), lens_step=entry.get("lens_step"), server_tps=entry.get("decode_tps"))
            return r

        def speed(row: dict) -> None:
            traced = row["traced"]
            how = ("off" + (", again" if row.get("again") else "") if not traced
                   else "on, lenses off" if row["step"] == config.LENS_OFF else f"on, step {row['step']}")
            log(config.paint(f"  writing, tracing {how} ...", "2"))
            res["speed"].append(row)              # kept as far as it got
            row["runs"] = []
            pause()
            row["warmup"] = timed(traced)
            for _ in range(repeats):
                r = timed(traced)
                r["tps"] = tps(r)
                row["runs"].append(r)
            row["tps"] = median([r["tps"] for r in row["runs"]])
            other = {r.get("lens_step") for r in row["runs"] if traced} - {row["step"]}
            if other:                             # a reply read at another step than asked: flagged, not hidden
                row["read_at"] = sorted(other, key=str)

        speed(dict(traced=False, step=None))
        changed = True
        set_step(config.LENS_OFF)
        speed(dict(traced=True, step=config.LENS_OFF, layers=0))
        for k, s in enumerate(lens.get("steps") or []):
            set_step(k)
            speed(dict(traced=True, step=k, layers=s.get("n"), layer_ids=s.get("layers")))
        speed(dict(traced=False, step=None, again=True))      # the drift since the first
        set_step(config.LENS_STEP_DEFAULT)        # the edit and the engram inspection at the default step
        res["edit"] = edit(server, model, ask, max_tokens, lens, log, pause)
        res["lookup"] = lookup(server, model, contexts, log, pause) if contexts else dict(skipped=True)
    finally:
        res["restored"] = livecheck.restore(set_step, res["was"]["lens_step"], log) if changed else {}
    return res


def edit(server: client.Client, model: str, ask: list, max_tokens: int, lens: dict, log, pause=lambda: None) -> dict:
    """The dock's one-token edit of a fresh traced reply, halfway through it, with doctor --live's steer: a word
    removed over the server's edit band (``lens`` /scope/api/lens/layers' edit_band, there with a J-lens) where a word
    vector is or can be made, else the model's concept."""
    if "edit_at" not in server.caps():
        return dict(na="this server cannot change one word (no edit_at)")
    concepts = server.get(f"/scope/api/concepts?model={urllib.parse.quote(model)}")
    if not concepts.get("steer_enabled"):
        return dict(na="steering is off (metalscope steering on)")
    notes: list = []
    words = livecheck.word_vectors(server, model, lambda good, text, feature=None: notes.append(text), "edit_band" in lens)
    steer = livecheck.edit_steer(livecheck.pick_concept(concepts), words, lens.get("edit_band"))
    if steer is None:
        return dict(na="; ".join(notes + ["no word vector or concept to edit with"]))
    log(config.paint("  changing one word ...", "2"))
    pause()
    t = time.time()
    orig = chat(server, model, ask, max_tokens, True)
    entry = (livecheck.saved_traces(t, model, client=CLIENT) or [None])[0]
    rid = livecheck.trace_meta(entry).get("request_id") if entry else None
    if not rid or not orig.get("tokens"):
        return dict(na="the reply to edit was not saved", reply=orig)
    at = orig["tokens"] // 2
    out = dict(at=at, of=orig["tokens"], steer=steer, reply=orig, notes=notes)
    try:
        r = chat(server, model, ask, max_tokens, True, {config.EDIT_HEADER: f"{rid}@{at} {steer}"})
    except client.ServerError as exc:
        return dict(out, na=exc.detail or str(exc))
    return dict(out, first_s=r["first_s"], end_s=r["seconds"], written=(r["tokens"] or 0) - at, run=r)


def lookup(server: client.Client, model: str, contexts, log, pause=lambda: None) -> dict:
    """The engram inspection of a short reply after a conversation of each length in ``contexts`` (tokens): its time,
    the conversation's real length; a length the server refuses is noted, not fatal."""
    if "lookup" not in server.caps():
        return dict(na="no engram on this model here")
    ratio = 4.0       # characters a token, measured where the server tokenizes, then from each real count
    if "token_prompts" in server.caps():
        sample = filler(4000, seed=1)
        ratio = len(sample) / max(1, len(server.tokenize(sample)))
    rows = []
    for n in contexts:
        log(config.paint(f"  engram inspection after {n:,} tokens ...", "2"))
        msgs = conversation(int(n * ratio))
        row = dict(target=n, chars=sum(len(m["content"]) for m in msgs))
        rows.append(row)
        pause()
        t = time.time()
        try:
            r = chat(server, model, msgs, SHORT_REPLY, True)
        except client.ServerError as exc:
            row["na"] = f"refused: {exc.detail or exc}"
            continue
        row.update(reply=r, prompt_tokens=r["prompt_tokens"])
        if r["prompt_tokens"]:
            ratio = row["chars"] / r["prompt_tokens"]
        entry = (livecheck.saved_traces(t, model, client=CLIENT) or [None])[0]
        if entry is None:
            row["na"] = "the reply was not saved"
            continue
        row["prompt_tokens"] = row["prompt_tokens"] or livecheck.trace_meta(entry).get("prompt_len")
        for again in (False, True):
            t = time.time()
            try:
                got = server.post("/scope/api/lookup", dict(trace=entry["id"]), timeout=3600)
            except client.ServerError as exc:
                why = exc.detail or str(exc)
                if again or exc.status != 409 or ASK_AGAIN not in why:
                    row["na"] = why
                    break
                row["retried"] = why
                continue
            row.update(seconds=round(time.time() - t, 2), trace=entry["id"], read=len(got.get("ids") or []))
            break
    return dict(rows=rows)


def _num(x, fmt="{:.1f}") -> str:
    return "?" if x is None else fmt.format(x)


def bar(x: float, top: float) -> str:
    """``x`` as a bar BAR characters long at ``top``, in eighths of a character."""
    n = round(x / top * BAR * 8)
    return "█" * (n // 8) + " ▏▎▍▌▋▊▉"[n % 8].strip()


def table(res: dict) -> list[str]:
    """The results as a plain table for anyone: the writing speeds with a bar each, and the change from the first
    untraced one."""
    head = lambda text, unit: config.paint(f"{text:<46}{unit:>9}", "1")
    out = [config.paint(" · ".join(str(res.get(k)) for k in ("model", "engine", "machine", "when")), "1"), ""] if "model" in res else []
    rows = res.get("speed") or []
    if rows:
        out.append(head(f"Writing speed ({res.get('max_tokens')} tokens, median of {res.get('repeats')})", "tokens/s"))
        top = max((r["tps"] for r in rows if r.get("tps")), default=None)
        off = next((r.get("tps") for r in rows if not r["traced"]), None)
        for r in rows:
            label = ("tracing off, again" if r.get("again") else "tracing off" if not r["traced"]
                     else "tracing on, lenses off" if r["step"] == config.LENS_OFF
                     else f"tracing on, {r.get('layers')} layers read (step {r['step']})")
            note = f"  read at step {', '.join(map(str, r['read_at']))}, not {r['step']}" if r.get("read_at") else ""
            v = r.get("tps")
            b = bar(v, top) if v else ""
            pct = f"{(v / off - 1) * 100:+.0f}%" if (r["traced"] or r.get("again")) and v and off else ""
            out.append(f"  {label:<44}{_num(v):>9}  {config.paint(b, '36')}{' ' * (BAR - len(b) + 6 - len(pct))}"
                       f"{config.paint(pct, '2')}{note}")
        out.append("")
    ed = res.get("edit")
    if ed is not None:
        out.append(head(f"Changing one word (token {ed['at']} of {ed['of']})" if "of" in ed else "Changing one word",
                        "seconds"))
        if ed.get("na"):
            out.append(f"  n/a: {ed['na']}")
        else:
            out.append(f"  {'until the new word appears':<44}{_num(ed.get('first_s'), '{:.2f}'):>9}")
            out.append(f"  {'until the reply is finished':<44}{_num(ed.get('end_s'), '{:.2f}'):>9}")
        out.append("")
    lk = res.get("lookup")
    if lk is not None and not lk.get("skipped"):
        out.append(head("Engram inspection (the conversation read twice)", "seconds"))
        if lk.get("na"):
            out.append(f"  n/a: {lk['na']}")
        for r in lk.get("rows") or []:
            real = f"{r['prompt_tokens']:,}" if r.get("prompt_tokens") else f"~{r['target']:,}"
            label = f"after {real} tokens of conversation"
            out.append(f"  {label:<44}" + (f"  n/a: {r['na']}" if r.get("na") else f"{_num(r.get('seconds')):>9}"))
        out.append("")
    if res.get("error"):
        out.append(f"stopped early: {res['error']}")
    out = [line.rstrip() for line in out]
    return out[:-1] if out and out[-1] == "" else out


def main(backend: Optional[str] = None, repeats: Optional[int] = None, contexts=None, max_tokens: Optional[int] = None,
         json_path: Optional[str] = None, model: Optional[str] = None, rest: Optional[float] = None) -> None:
    """None: the defaults; ``contexts`` empty: no engram inspection."""
    repeats, max_tokens, rest = repeats or REPEATS, max_tokens or MAX_TOKENS, REST if rest is None else rest
    contexts = CONTEXTS if contexts is None else contexts
    server, model = pick(backend, model)
    res: dict = {}
    log = lambda s: print(s, flush=True)
    served = server.served()
    log(f"bench · {model} · {(server.status() or {}).get('engine')} · {server.url}"
        + (f" (of the {len(served)} models it serves; --model picks another)" if len(served) > 1 else ""))
    try:
        run(server, res, repeats, contexts, max_tokens, log, model, rest)
    except KeyboardInterrupt:
        res["error"] = "stopped (ctrl+c)"
    except client.ServerError as exc:
        res["error"] = str(exc)
    finally:
        if json_path and res:
            config.atomic_write(json_path, json.dumps(res, indent=1))
    if isinstance((res.get("restored") or {}).get("lens_step"), int):       # else restore said what was not
        log(config.paint(f"put back: lens step {res['restored']['lens_step']}", "2"))
    print()
    print("\n".join(table(res)))
    if res.get("error"):
        raise SystemExit(1)

"""Trace retention (retention.py): the size budget, what a prune removes and in what order, what it never removes,
the index rewrite against concurrent appends, the running total, and the automatic prune off the request path."""

import json
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from metalscope import cli, config, doctor, retention as R, steer, store
from metalscope.live import TraceManager
from metalscope.store import Trace, append_index, read_index
from tests.helpers import Tok, tiny_llama, wait_saved


def make(i, client=None, model="m", steer_label="s@1", size=1000, t=None, day="2026-10-01"):
    """A trace folder of ``size`` bytes and its index entry (not written)."""
    d = config.traces_dir() / day / f"t{i}"
    d.mkdir(parents=True)
    (d / "arrays.npz").write_bytes(b"\0" * size)
    return dict(id=f"t{i}", path=str(d), time=float(i if t is None else t), model=model, client=client, steer=steer_label,
                bytes=size)


def write_index(entries):
    config.index_path().parent.mkdir(parents=True, exist_ok=True)
    config.index_path().write_text("".join(json.dumps(e) + "\n" for e in entries))


def ids():
    return [json.loads(l)["id"] for l in config.index_path().read_text().splitlines()]


def run(capsys, *argv) -> str:
    cli.main(list(argv))
    return capsys.readouterr().out


def test_the_budget_is_10gb_until_set_and_sizes_read_plainly(home, capsys):
    assert R.budget() == R.DEFAULT_BUDGET == 10 * 10 ** 9
    assert [R.parse_size(s) for s in ("20GB", "20", "500MB", "1.5tb", " 2 gb ")] == \
           [20 * 10 ** 9, 20 * 10 ** 9, 500 * 10 ** 6, 1500 * 10 ** 9, 2 * 10 ** 9]
    for bad in ("", "GB", "-1GB", "0", "20 parsecs", "1e9"):
        with pytest.raises(ValueError):
            R.parse_size(bad)
    out = run(capsys, "traces", "keep", "20GB")
    assert out.splitlines()[0] == "budget: 20.0 GB"
    assert config.load_settings()["trace_budget"] == 20 * 10 ** 9 and R.budget() == 20 * 10 ** 9
    assert config.load_settings()["mode"] == "off"                         # the other settings are kept
    with pytest.raises(SystemExit, match="error: not a size"):
        cli.main(["traces", "keep", "lots"])
    with pytest.raises(ValueError):
        config.save_settings(trace_budget=-5)
    write_index([make(0, size=3000), make(1, size=2000)])
    for argv in (("traces",), ("traces", "status")):
        lines = run(capsys, *argv).splitlines()
        assert lines[:5] == ["traces: 2", "total: 0.0 MB", "budget: 20.0 GB (metalscope traces keep SIZE changes it)",
                             "automatic cleanup: on (metalscope traces auto on|off)", "pinned: 0"]
    config.save_settings(trace_budget=4000)                                  # bytes, to go over with two small traces
    lines = run(capsys, "traces").splitlines()
    assert lines[1] == "total: 0.0 MB" and lines[5].startswith("over the budget by ")


def test_prune_removes_automated_runs_oldest_first_then_chats_oldest_first(home):
    entries = [make(0, client=None), make(1, client="calibrate-x"), make(2, client="pi-abc"),
               make(3, client="ladder-1"), make(4, client=None), make(5, client="calibrate-x"), make(6, client="pi-abc")]
    write_index(entries)
    res = R.prune(limit=7000, dry_run=True)
    assert res.removed == [] and res.before_bytes == 7000
    res = R.prune(limit=3000)
    # Four to go: the three automated runs (oldest first), then the oldest chat.
    assert [e["id"] for e in res.removed] == ["t1", "t3", "t5", "t0"]
    assert (res.auto_count, res.auto_bytes, res.chat_count, res.chat_bytes) == (3, 3000, 1, 1000)
    assert (res.freed, res.after_bytes, res.after_count) == (4000, 3000, 3)
    assert ids() == ["t2", "t4", "t6"]
    for e in entries:
        assert (config.traces_dir() / "2026-10-01" / e["id"]).exists() == (e["id"] in ("t2", "t4", "t6"))
    assert [e["id"] for e in read_index()] == ["t6", "t4", "t2"]


def test_the_empty_day_folder_goes_with_its_last_trace(home):
    write_index([make(0, client="bench", day="2026-09-01"), make(1, day="2026-10-01")])
    R.prune(limit=1000)
    assert not (config.traces_dir() / "2026-09-01").exists() and (config.traces_dir() / "2026-10-01" / "t1").exists()
    # Everything removable goes when nothing is protected; a folder another kept line still names stays.
    shared = make(2, client="pi-x")
    write_index([make(3, client="bench"), dict(shared, id="dup", client="bench", time=0.0), dict(shared, steer=None)])
    R.prune(limit=1)
    assert ids() == ["t2"] and (config.traces_dir() / "2026-10-01" / "t2").exists()
    assert not (config.traces_dir() / "2026-10-01" / "t3").exists()
    write_index([dict(make(4, client="bench"), steer="s@1"), dict(make(5), steer="s@1")])
    R.prune(limit=1)
    assert ids() == [] and not (config.traces_dir() / "2026-10-01" / "t4").exists()


def test_prune_never_removes_pins_the_strength_tables_traces_or_traces_being_written(home, capsys):
    """Budget 1 byte: everything that can go goes. What stays: the pinned trace, the newest NORM_TRACES unsteered
    traces of each served model (what steer.norm_table measures on next), a folder not yet in the index (being
    written), and a folder outside the traces directory."""
    tdir = config.traces_dir() / "2026-10-01"
    entries = []
    for i in range(60):        # model a: unsteered and steered, interleaved; model b (two served copies' paths): fewer
        model = "org/a" if i % 3 else "b"
        label = "s@1" if i % 4 == 0 else None
        arrays = {f"resid_norm/{l}": np.full(5, float(i + l), np.float32) for l in range(2)}
        p = Trace(arrays, [1] * 5, ["x"] * 5, 2, dict(model=model)).save(tdir / f"t{i}")
        entries.append(dict(id=f"t{i}", path=str(p), time=float(i), model=model, steer=label,
                            client="calibrate-x" if i % 2 else None, bytes=100))
    outside = config.home().parent / f"{config.home().name}-elsewhere" / "t99"
    outside.mkdir(parents=True)
    (outside / "arrays.npz").write_bytes(b"\0" * 10)
    entries.append(dict(id="t99", path=str(outside), time=0.0, model="c", steer="s@1", client="bench", bytes=10))
    write_index(entries)
    writing = tdir / "t-writing"
    writing.mkdir()
    (writing / "arrays.npz").write_bytes(b"\0" * 10)
    run(capsys, "traces", "pin", "t1")
    with pytest.raises(SystemExit, match="error: no saved trace nope"):
        cli.main(["traces", "pin", "nope"])
    norms_a = [e["id"] for e in read_index() if not e.get("steer") and config.short_name(e["model"]) == "a"][:steer.NORM_TRACES]
    norms_b = [e["id"] for e in read_index() if not e.get("steer") and e["model"] == "b"][:steer.NORM_TRACES]
    assert len(norms_a) == steer.NORM_TRACES and 0 < len(norms_b) < steer.NORM_TRACES
    res = R.prune(limit=1)
    assert set(ids()) == set(norms_a) | set(norms_b) | {"t1", "t99"}
    assert res.protected == {"pinned": 1, "the strength table's newest unsteered": len(norms_a) + len(norms_b),
                             "outside the traces folder": 1}
    assert writing.exists() and outside.exists()
    # The strength table, measured afresh after the prune, picks exactly the traces it had before.
    assert steer.norm_table("a", refresh=True)["traces"] == norms_a
    assert steer.norm_table("b", refresh=True)["traces"] == norms_b
    lines = R.summary(res)
    assert "still over the budget: the protected traces alone take 0.0 MB" in lines
    # Unpinned, t1 goes: unsteered, but older than model a's newest 20.
    run(capsys, "traces", "unpin", "t1")
    assert R.pinned() == set()
    R.prune(limit=1)
    assert "t1" not in ids()


def test_a_dry_run_changes_nothing_and_says_what_a_prune_would_do(home, capsys):
    write_index([make(i, client="calibrate-x" if i < 3 else None, size=1000 + i) for i in range(6)])
    before = config.index_path().read_bytes()
    config.save_settings(trace_budget=3000)
    out = run(capsys, "traces", "prune", "--dry-run").splitlines()
    assert config.index_path().read_bytes() == before
    assert all((config.traces_dir() / "2026-10-01" / f"t{i}").exists() for i in range(6))
    assert out[:5] == ["before: 6 traces, 0.0 MB", "budget: 0.0 MB", "would remove: 4 traces, 0.0 MB",
                       "  automated runs: 3, 0.0 MB", "  chats: 1, 0.0 MB"]
    assert out[5] == "would keep: 2 traces, 0.0 MB" and out[-1] == "nothing deleted (--dry-run)"
    dry = R.prune(dry_run=True)
    real = R.prune()
    assert [e["id"] for e in dry.removed] == [e["id"] for e in real.removed] == ["t0", "t1", "t2", "t3"]
    assert ids() == ["t4", "t5"]
    out = run(capsys, "traces", "prune").splitlines()
    assert out[2] == "nothing to remove: under the budget"


def test_a_prune_prints_the_totals_before_and_after_and_keeps_the_recorded_sizes(home, capsys):
    write_index([make(i, client=None, size=2_000_000) for i in range(5)])
    sizes = R.Usage()
    assert sizes.refresh().bytes == 10_000_000 and sizes.count == 5
    config.save_settings(trace_budget=5_000_000)
    out = run(capsys, "traces", "prune").splitlines()
    assert out[:4] == ["before: 5 traces, 10.0 MB", "budget: 5.0 MB", "removed: 3 traces, 6.0 MB", "  chats: 3, 6.0 MB"]
    assert out[4] == "kept: 2 traces, 4.0 MB"
    assert [json.loads(l)["bytes"] for l in config.index_path().read_text().splitlines()] == [2_000_000, 2_000_000]


def test_index_lines_for_folders_already_gone_are_dropped(home):
    e = make(0)
    write_index([dict(e, id="gone", path=str(config.traces_dir() / "nowhere")), make(1)])
    res = R.prune(limit=10 ** 9)
    assert res.stale == 1 and res.removed == [] and ids() == ["t1"]


def test_the_rewrite_keeps_every_line_appended_meanwhile(home):
    """Appends through the lock wait for the rewrite."""
    write_index([make(i, client="calibrate-x") for i in range(40)])
    added, stop = [], threading.Event()

    def appender():
        n = 0
        while not stop.is_set() or n < 50:
            e = dict(id=f"new{n}", path=str(config.traces_dir() / "2026-10-01" / "t39"), time=1e9 + n, bytes=0,
                     client="pi-x")
            append_index(e)
            added.append(e["id"])
            n += 1
    th = threading.Thread(target=appender)
    th.start()
    for limit in (30_000, 20_000, 10_000, 5_000):
        R.prune(limit=limit)
    stop.set()
    th.join()
    left = ids()
    assert [i for i in left if i.startswith("new")] == added           # none lost, in order
    assert not any(f"t{i}" in left for i in range(34))                 # pruned
    assert len(left) == len(set(left))


def test_the_running_total_reads_only_what_was_appended(home):
    write_index([make(0, size=100), make(1, size=200)])
    u = R.Usage()
    assert (u.refresh().bytes, u.count) == (300, 2)
    append_index(make(2, size=50))
    with open(config.index_path(), "a") as f:
        f.write('{"id": "half", "path": "/x", "by')                     # a line still being written
    assert (u.refresh().bytes, u.count) == (350, 3)                     # the new line only
    with open(config.index_path(), "a") as f:
        f.write('tes": 7}\n')
    assert (u.refresh().bytes, u.count) == (357, 4)
    append_index(dict(id="unsized", path="/y", time=0.0))
    assert (u.refresh().bytes, u.count) == (357, 5)                     # no bytes recorded: counts as 0
    R.prune(limit=10 ** 9)                                              # a rewrite: read from the start again
    assert (u.refresh().bytes, u.count) == (350, 3)


def test_read_index_is_parsed_again_only_when_the_file_changes(home, monkeypatch):
    write_index([make(0), make(1)])
    first = read_index()
    first[0]["title"] = "changed by a caller"
    checks = []
    real = store.Path.exists
    monkeypatch.setattr(store.Path, "exists", lambda self: checks.append(self) or real(self))
    assert [e["id"] for e in read_index()] == ["t1", "t0"] and "title" not in read_index()[0]
    assert checks == []                                                 # unchanged file: no folder checked
    append_index(make(2))
    assert [e["id"] for e in read_index()] == ["t2", "t1", "t0"] and len(checks) == 3


def _trace_one(mgr, tracer, model, rid):
    mgr.register(rid, "on", dict(conversation=rid, turn=1, last_user="hi"))
    mgr.rows_for(tracer, [rid])
    model(mx.array([[1, 5]]))
    mgr.clear_rows(tracer)
    for tok in (9, 11):
        mgr.rows_for(tracer, [rid])
        model(mx.array([[tok]]))
        mgr.clear_rows(tracer)
    mgr.finish(rid, [1, 5, 9], [11, 12], tracer)


@pytest.mark.parametrize("fails", ["arrays", "index"])
def test_a_save_that_fails_leaves_no_folder_and_prune_removes_one_cut_short_a_day_ago(home, monkeypatch, fails):
    """A save that raises, writing its arrays or adding its index line, leaves no folder that no prune would see. A
    save killed part way leaves its dot folder (store.PART): prune removes one untouched for a day, never a newer one
    (a save under way)."""
    import os
    from metalscope import experts, live
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(*((store, "write_npz") if fails == "arrays" else (live, "append_index")), boom)
    _trace_one(mgr, tracer, model, "req-1")
    mgr.writer.submit(lambda: None).result()
    assert mgr.saved == 0 and not list(config.traces_dir().glob("*/*"))       # nothing in the day's folder
    day = config.traces_dir() / "2026-10-01"
    old, new = day / f".t1{store.PART}", day / f".t2{store.PART}"
    for d in (old, new):
        d.mkdir(parents=True)
        (d / "arrays.npz").write_bytes(b"\0" * 10)
        (d / "meta.json").write_text(json.dumps(dict(model="m", num_experts=4)))
    assert experts.find_traces(config.traces_dir()) == []                     # never read as a trace
    stale = time.time() - R.CUT_SHORT_AGE - 60
    for x in (old / "arrays.npz", old / "meta.json", old):
        os.utime(x, (stale, stale))
    R.prune(dry_run=True)
    assert old.exists()
    R.prune()
    assert not old.exists() and new.exists()


def test_saved_traces_record_their_size_and_the_prune_runs_on_its_own_thread(home, monkeypatch):
    """Each index entry records its folder's bytes. Past the budget, the server prunes from its prune thread: the
    request (finish) and the writer never wait for it, and saves go on while it runs."""
    write_index([make(i, client="calibrate-x", size=5000) for i in range(3)])
    config.save_settings(mode="on", trace_budget=6000)
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    started, release, seen = threading.Event(), threading.Event(), []
    real_prune = R.prune

    def slow_prune(*a, **k):
        seen.append(threading.current_thread().name)
        started.set()
        assert release.wait(10)
        return real_prune(*a, **k)
    monkeypatch.setattr(R, "prune", slow_prune)
    t0 = time.time()
    _trace_one(mgr, tracer, model, "req-1")
    assert time.time() - t0 < 5
    assert started.wait(10)                                   # the prune is running, and blocked
    _trace_one(mgr, tracer, model, "req-2")
    wait_saved(mgr, 2)                                        # the writer saved another meanwhile
    release.set()
    deadline = time.time() + 10
    while time.time() < deadline and any(i in ids() for i in ("t0", "t1", "t2")):
        time.sleep(0.02)
    assert seen and all(n.startswith("metalscope-prune") for n in seen)
    saved = [json.loads(l) for l in config.index_path().read_text().splitlines()]
    assert [e["conversation"] for e in saved] == ["req-1", "req-2"]       # the old automated runs went first
    for e in saved:
        assert e["bytes"] == R.folder_bytes(e["path"]) > 0


def test_under_the_budget_the_check_after_a_save_does_not_prune(home, monkeypatch):
    config.save_settings(mode="on")
    mgr = TraceManager()
    model = tiny_llama()
    tracer = mgr.attach(model, "tiny", Tok())
    monkeypatch.setattr(R, "prune", lambda *a, **k: pytest.fail("pruned under the budget"))
    _trace_one(mgr, tracer, model, "req-1")
    wait_saved(mgr, 1)
    deadline = time.time() + 5
    while mgr.keeper.usage.count < 1 and time.time() < deadline:   # the queued check has run
        time.sleep(0.02)
    assert mgr.keeper.usage.count == 1 and mgr.keeper.last is None


def test_older_than_and_all_remove_whatever_the_budget_and_keep_the_protected(home, capsys):
    now = time.time()
    day = 86400
    entries = [make(0, client="calibrate-x", t=now - 30 * day), make(1, client=None, t=now - 10 * day),
               make(2, client="pi-abc", t=now - 8 * day), make(3, client=None, t=now - 2 * day),
               make(4, client="pi-abc", t=now - day, steer_label=None)]       # the model's newest unsteered: kept
    write_index(entries)
    R.pin("t1")
    # Under the 10 GB budget, so a plain prune removes nothing; a cutoff removes what is older, pins kept.
    assert R.prune(dry_run=True).removed == []
    res = R.prune(dry_run=True, before=now - 7 * day)
    assert [e["id"] for e in res.removed] == ["t0", "t2"]
    assert "removing: every trace saved before " in "\n".join(R.summary(res))
    out = run(capsys, "traces", "prune", "--older-than", "7d")
    assert "removed: 2 traces" in out and ids() == ["t1", "t3", "t4"]
    out = run(capsys, "traces", "prune", "--all", "--dry-run")
    assert "removing: every trace" in out and "would remove: 1 traces" in out and ids() == ["t1", "t3", "t4"]
    run(capsys, "traces", "prune", "--all")
    assert ids() == ["t1", "t4"]                       # the pin and the strength table's trace stay
    with pytest.raises(SystemExit, match="is not an age"):
        cli.main(["traces", "prune", "--older-than", "soon"])
    assert [R.parse_age(a) for a in ("7d", "2w", "12h", "3")] == [7 * day, 14 * day, 12 * 3600, 3 * day]


def test_doctor_says_how_much_space_traces_take_and_how_to_free_it(home, capsys, monkeypatch):
    now = time.time()
    write_index([make(0, client=None, t=now - 9 * 86400, size=4000), make(1, client=None, t=now, size=1000)])
    st = R.status(week=True)
    assert (st["count"], st["bytes"], st["old_count"], st["old_bytes"]) == (2, 5000, 1, 4000)
    def no_process(*a, **k):
        raise OSError("not here")
    monkeypatch.setattr(doctor.subprocess, "run", no_process)
    monkeypatch.setattr(doctor.client.urllib.request, "urlopen", no_process)
    with pytest.raises(SystemExit):
        cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "  · 2 traces · 0.0 MB of 10.0 GB · 0.0 MB older than a week · " in out
    assert "  · free space: metalscope traces prune --older-than 7d, or --all (--dry-run first)\n" in out


def test_an_age_must_be_a_finite_number_and_the_automatic_prune_leaves_slack(home, monkeypatch):
    for bad in ("nan", "inf", "-1d", "0"):
        with pytest.raises(ValueError, match="is not an age"):
            R.parse_age(bad)
    write_index([make(i, client=None, size=1000) for i in range(10)])
    config.save_settings(trace_budget=5000)
    keeper = R.Keeper()
    keeper._check()
    assert keeper.last.budget == int(5000 * R.SLACK) and keeper.last.after_bytes == 4000   # down to 90%: room for the next saves


def test_automatic_cleanup_can_be_turned_off_and_doctor_says_so(home, capsys, monkeypatch):
    write_index([make(i, client=None, size=1000) for i in range(10)])
    config.save_settings(trace_budget=5000)
    assert R.auto() and "automatic cleanup: on" in run(capsys, "traces", "auto")
    assert "nothing is removed until" in run(capsys, "traces", "auto", "off")
    keeper = R.Keeper()
    keeper._check()
    assert keeper.last is None and len(ids()) == 10                  # off: over the budget, nothing removed
    assert "nothing goes until metalscope traces prune" in run(capsys, "traces")
    def no_process(*a, **k):
        raise OSError("not here")
    monkeypatch.setattr(doctor.subprocess, "run", no_process)
    monkeypatch.setattr(doctor.client.urllib.request, "urlopen", no_process)
    with pytest.raises(SystemExit):
        cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "  · cleanup past the budget: off (metalscope traces auto on|off)" in out
    assert "  ! 0.0 MB over the budget and cleanup is off: metalscope traces prune frees it" in out
    run(capsys, "traces", "auto", "on")
    keeper._check()
    assert keeper.last is not None and len(ids()) < 10              # on again: down to the budget's slack
    with pytest.raises(ValueError, match="trace_auto"):
        config.save_settings(trace_auto="no")


def test_planning_a_prune_loads_no_mlx(home):
    """The strength table's pick, which a prune keeps, is store.norm_picks: retention needs no MLX for it."""
    import subprocess
    import sys
    (home / "t").mkdir()
    code = (f"import sys; from metalscope import retention as R; "
            f"R.plan([dict(id='a', path={str(home / 't')!r}, model='m', time=1)], 1, set()); "
            f"assert 'mlx.core' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_prune_deletes_nothing_outside_the_traces_folder_when_a_folder_turns_into_a_link_after_its_plan(home, monkeypatch):
    write_index([make(0, client="bench")])
    day, elsewhere = config.traces_dir() / "2026-10-01", home / "elsewhere"
    real = R.plan

    def plan_then_swap(*a, **k):
        out = real(*a, **k)
        day.rename(elsewhere)
        day.symlink_to(elsewhere)
        return out
    monkeypatch.setattr(R, "plan", plan_then_swap)
    assert [e["id"] for e in R.prune(limit=1).removed] == ["t0"]
    assert (elsewhere / "t0" / "arrays.npz").exists()


def saved(i, meta=None, day="2026-10-01", **kw):
    """A trace folder as a save leaves it (``make``'s, with its meta.json), untouched for longer than CUT_SHORT_AGE,
    and the index line a save would write for it."""
    e = make(i, day=day, **kw)
    d = Path(e["path"])
    meta = dict(started=float(i), model=e["model"], client=e["client"], level="on", prompt_len=2, generated_tokens=3,
                title=f"reply {i}", **({"steer": dict(label=e["steer"])} if e["steer"] else {})) if meta is None else meta
    (d / "meta.json").write_text(json.dumps(meta))
    aged(d)
    return store.index_entry(d, meta) if isinstance(meta, dict) else None


def aged(d: Path) -> None:
    import os
    stale = time.time() - R.CUT_SHORT_AGE - 60
    for x in (*d.iterdir(), d):
        os.utime(x, (stale, stale))


def test_prune_adopts_a_trace_folder_the_index_lost_and_leaves_anything_else_under_traces_alone(home, capsys):
    """A day-old <day>/<trace id> folder with a meta.json that no index line names (a process stopped between the
    save and its index line, a line lost) gets the line a save writes; it is never deleted for it. Anything else
    under traces/ is not a trace: left alone, never listed."""
    import os
    write_index([saved(0)])                                  # indexed and a day old: no second line
    lost = saved(1)
    recent = make(2)                                         # a save under way
    (Path(recent["path"]) / "meta.json").write_text("{}")
    no_meta = make(3)
    aged(Path(no_meta["path"]))
    saved(5, meta=[1, 2])                                    # meta.json not an object
    elsewhere = home / "elsewhere" / "t4"
    elsewhere.mkdir(parents=True)
    (elsewhere / "meta.json").write_text("{}")
    (config.traces_dir() / "2026-10-01" / "t4").symlink_to(elsewhere)
    os.utime(elsewhere / "meta.json", (0, 0))
    mine = config.traces_dir() / "exports" / "for-paper"
    mine.mkdir(parents=True)
    (mine / "meta.json").write_text("{}")
    aged(mine)
    before = config.index_path().read_text()
    assert "would adopt: 1 trace folders the index did not list" in run(capsys, "traces", "prune", "--dry-run")
    assert config.index_path().read_text() == before
    out = run(capsys, "traces", "prune")
    assert "adopted: 1 trace folders the index did not list" in out and "nothing to remove: under the budget" in out
    assert ids() == ["t0", "t1"] and read_index()[0] == dict(lost, bytes=R.folder_bytes(lost["path"]))
    assert lost["bytes"] == R.folder_bytes(lost["path"]) and lost["path"] == str(Path(lost["path"]).resolve())
    assert all(Path(p).exists() for p in (lost["path"], recent["path"], no_meta["path"], elsewhere, mine))
    assert "adopted" not in run(capsys, "traces", "prune") and ids() == ["t0", "t1"]      # once


def test_a_folder_two_prunes_find_at_once_gets_one_line(home):
    """Each prune finds the lost folders outside the index lock; the append reads the index again under it, so a
    folder another prune adopted meanwhile is not adopted twice."""
    write_index([saved(0)])
    lost = saved(1)
    found = R._unindexed(config.traces_dir().resolve(), [])                # both prunes found t1 (and t0's line was read)
    assert [e["id"] for e in found] == ["t0", "t1"]
    assert R.prune().adopted == 1                                        # the first prune
    assert append_index(*found, new_only=True) == []                     # the second's append, after it
    assert ids() == ["t0", "t1"] and Path(lost["path"]).exists()


def test_an_empty_or_damaged_index_loses_no_trace(home):
    a, b = saved(1, steer_label=None), saved(2, client="pi-x")
    config.index_path().write_text("")                       # a reset by hand, a rewrite lost at power-off
    assert R.prune().adopted == 2 and sorted(ids()) == ["t1", "t2"]
    config.index_path().write_text(json.dumps(a)[:-3] + "XX}\n" + json.dumps(b) + "\n")     # one line damaged
    R.prune()
    assert sorted(e["id"] for e in read_index()) == ["t1", "t2"] and all(Path(e["path"]).exists() for e in (a, b))


def test_a_moved_data_home_loses_no_trace(home, monkeypatch, tmp_path):
    import shutil
    write_index([saved(1), saved(2)])
    new = tmp_path.parent / (tmp_path.name + "-moved")
    shutil.move(str(tmp_path), str(new))
    monkeypatch.setenv("METALSCOPE_HOME", str(new))
    tmp_path.mkdir()
    R.prune()                                                # the old paths' lines dropped as stale
    R.prune()
    moved = [new / "traces" / "2026-10-01" / x for x in ("t1", "t2")]
    assert all(p.exists() for p in moved) and sorted(e["path"] for e in read_index()) == [str(p.resolve()) for p in moved]


def test_an_adopted_trace_is_pruned_by_the_budget_in_its_order_and_the_protected_stay(home):
    saved(1, client="calibrate-x")
    saved(2, client="pi-x")
    saved(3, client="pi-x")
    saved(4, client="pi-x", steer_label=None)                # the strength table's newest unsteered
    R.pin("t3")
    config.index_path().write_text("")
    res = R.prune(limit=1)
    assert res.adopted == 4 and [e["id"] for e in res.removed] == ["t1", "t2"] and sorted(ids()) == ["t3", "t4"]
    assert not (config.traces_dir() / "2026-10-01" / "t1").exists() and (config.traces_dir() / "2026-10-01" / "t4").exists()


def test_an_append_after_a_line_cut_short_starts_on_a_line_of_its_own(home):
    append_index(make(0))
    with open(config.index_path(), "a") as f:
        f.write(json.dumps(make(1))[:25])                               # a crash mid-append: no newline
    append_index(make(2))
    assert [e["id"] for e in read_index()] == ["t2", "t0"]


def test_an_index_line_that_is_not_an_entry_is_skipped_by_every_reader(home):
    good = make(0)
    config.index_path().write_text("[1, 2]\n" + json.dumps(dict(path=make(1)["path"], steer="s@1")) + "\n" + json.dumps(good) + "\n")
    assert [e["id"] for e in read_index()] == ["t0"]
    assert R.Usage().refresh().count == 2
    assert R.prune(limit=10 ** 9).removed == []
    R.prune(limit=1)
    assert ids() == [] and not Path(good["path"]).exists()

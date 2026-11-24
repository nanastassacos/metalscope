import assert from "node:assert/strict";
import { test } from "node:test";
import { SPIN_MS, registerJobs } from "../job-runner.ts";
import { type LearnState, STOP_LABEL, trimJob } from "../jobs.ts";
import { visibleWidth } from "@earendil-works/pi-tui";
import { CAL } from "./fixtures/concepts.ts";

/** A fake Pi recording its handlers and entries; one job store every server reads (as the data home is shared), whose
 * GET /api/jobs answers `server.latest` (once `server.hold` resolves, when set) and records each job marked seen; the
 * api asks the selected model's server (live.base), which refuses while it is not `up`; a ctx recording notes. */
function setup() {
	const handlers: Record<string, (event: any, ctx: any) => void> = {};
	const entries: any[] = [];
	const notes: [string, string][] = [];
	const seen: string[] = [], cancels: string[] = [];
	const server = { latest: undefined as LearnState | undefined, gets: 0, hold: undefined as Promise<void> | undefined,
		plan: { name: "x", source: "write", texts: null, description: "", taken: null } as Record<string, unknown>, planned: [] as string[],
		posted: [] as any[] };
	const pi = { on: (name: string, fn: any) => void (handlers[name] = fn), appendEntry: (type: string, data: any) => void entries.push({ type, ...data }) } as any;
	const live = { base: OMLX_BASE };
	const up = new Set([OMLX_BASE, DS4_BASE]);
	const api = async (path: string, init?: any) => {
		if (!up.has(live.base)) throw new Error("ECONNREFUSED");     // as traceApi: on the selected model's server
		if (path === "/api/concept/learn" && init?.method === "POST") return (server.posted.push(JSON.parse(init.body)), { job: "l-9" });
		if (path.startsWith("/api/concept/learn?")) {
			server.planned.push(path);
			return server.plan;
		}
		if (path === "/api/jobs") {
			server.gets++;
			await server.hold;
			return { latest: server.latest, jobs: [] };
		}
		if (path.endsWith("/seen")) return void seen.push(path);
		if (path.endsWith("/cancel")) {
			cancels.push(path);
			return { cancelled: true };
		}
		throw new Error(`unexpected ${path}`);
	};
	// The widget as Pi 1.5.2 keeps it (setExtensionWidget): the one shown disposed when replaced or removed, a factory
	// called with Pi's TUI (renders counted) and theme.
	const widget = { shown: undefined as any, renders: 0 };
	const tui = { requestRender: () => void widget.renders++ };
	const theme = { fg: (_c: string, s: string) => s, bold: (s: string) => s };
	// Raw terminal input as Pi 1.5.2 hands it to extensions (TUI.handleTerminalInput): each listener in turn, before
	// the focused component; one that consumes ends it. `press` says whether the input was consumed.
	const listeners = new Set<(data: string) => any>();
	const press = (data: string) => [...listeners].some((l) => l(data)?.consume);
	const ctx: any = { hasUI: true, ui: { setStatus() {}, notify: (m: string, kind: string) => void notes.push([m, kind]),
		setWidget: (_key: string, content: any) => {
			widget.shown?.dispose?.();
			widget.shown = typeof content === "function" ? content(tui, theme) : content;
		},
		onTerminalInput: (h: (data: string) => any) => (listeners.add(h), () => void listeners.delete(h)) } };
	const raised: any[] = [];
	const jobs = registerJobs({ pi, live, api } as any, (c) => void raised.push(c));
	return { handlers, entries, notes, seen, server, ctx, jobs, live, up, widget, cancels, raised, listeners, press };
}

const OMLX_BASE = "http://127.0.0.1:9", DS4_BASE = "http://127.0.0.1:19";

/** Let every pending promise run. */
const settle = () => new Promise((r) => setImmediate(r));

const running: LearnState = { kind: "calibrate", job: "c-1", name: "calm", phase: "calibrating", origin: "pi", steps: [] };

test("a job watched while it runs: polled every 3 s, one poll at a time, its result shown once", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval"] });
	const { handlers, entries, notes, seen, server, ctx } = setup();
	server.latest = running;
	handlers.session_start({}, ctx);                    // running: its widget back, polled at once
	await settle();
	assert.equal(server.gets, 2);                       // the startup look and the first poll
	server.latest = { ...running, phase: "done", outcome: "done", result: { calibration: { version: 4, default_dose: 0.2, good: 8, turns: 18 } } };
	let release!: () => void;
	server.hold = new Promise((r) => (release = r));
	t.mock.timers.tick(3000);                           // a poll asks, and waits for the answer…
	t.mock.timers.tick(3000);                           // …so the next one does nothing
	release();
	await settle();
	assert.equal(server.gets, 3);
	assert.equal(entries.length, 1);                    // watched while it ran: the full result entry
	assert.equal(entries[0].type, "scope-job");         // the job's state, trimmed: its result drawn when Pi renders it
	assert.deepEqual(entries[0].job, trimJob(server.latest));
	assert.deepEqual(notes, [["calibrate calm: 8/18 good @0.2", "info"]]);   // one colon, after the name
	assert.deepEqual(seen, ["/api/jobs/c-1/seen"]);
	t.mock.timers.tick(3000);                           // finished: no more polls
	await settle();
	assert.equal(server.gets, 3);
	handlers.session_shutdown({}, ctx);
});

test("a job stays in view when /model moves to the other backend, its result arriving though its own server stopped", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval"] });
	const { handlers, entries, seen, server, ctx, live, up } = setup();
	live.base = DS4_BASE;
	server.latest = running;
	handlers.session_start({}, ctx);                    // seen running through ds4's server
	await settle();
	assert.equal(server.gets, 2);
	up.delete(DS4_BASE);                                // a yes to a switch stopped ds4
	t.mock.timers.tick(3000);                           // its server gone: that poll fails quietly
	await settle();
	assert.equal(server.gets, 2);
	live.base = OMLX_BASE;                              // the oMLX model now selected: the same job store
	t.mock.timers.tick(3000);
	await settle();
	assert.equal(server.gets, 3);
	server.latest = { ...running, phase: "done", outcome: "done", result: { calibration: { version: 4, default_dose: 0.4, good: 16, turns: 18 } } };
	t.mock.timers.tick(3000);
	await settle();
	assert.equal(entries.length, 1);                    // the result arrives
	assert.deepEqual(seen, ["/api/jobs/c-1/seen"]);
	handlers.session_shutdown({}, ctx);
});

test("a poll that throws is dropped, never thrown out of setInterval; the next tick polls again", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval"] });
	const { handlers, server, ctx } = setup();
	let broken = true;
	const stale = { ...ctx, ui: { ...ctx.ui, setWidget() {
		if (broken) throw new Error("stale ctx");       // as Pi's ctx after the session it belonged to was replaced
	} } };
	server.latest = running;
	handlers.session_start({}, stale);
	await settle();                                     // the first poll threw inside: caught
	assert.equal(server.gets, 2);
	broken = false;
	t.mock.timers.tick(3000);                           // not stuck "polling": the next tick asks again
	await settle();
	assert.equal(server.gets, 3);
	handlers.session_shutdown({}, ctx);
});

test("two Pi windows: the one that watched the job run shows its result though the other marked it seen first", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval"] });
	const { handlers, entries, notes, seen, server, ctx } = setup();
	server.latest = running;
	handlers.session_start({}, ctx);
	await settle();
	server.latest = { ...running, phase: "done", outcome: "failed", error: "grader timed out", seen: true };   // the other window was first
	t.mock.timers.tick(3000);
	await settle();
	assert.equal(entries.length, 1);
	assert.deepEqual(notes, [["calibrate calm: failed (grader timed out)", "warning"]]);
	assert.deepEqual(seen, []);                         // already marked: not posted again
	handlers.session_shutdown({}, ctx);
});

test("/concept learn asks in the selected model's backend's terms: ds4 keeps serving, oMLX unloads", async () => {
	const { notes, ctx, jobs } = setup();
	const asked: [string, string][] = [];
	const ui = { ...ctx.ui, confirm: async (title: string, text: string) => { asked.push([title, text]); return false; } };
	await jobs.learn({ ...ctx, ui, model: { provider: "metalscope-ds4", id: "V41-ds4" } }, "crunchy", "crisp textures");
	await jobs.learn({ ...ctx, ui, model: { provider: "metalscope-omlx", id: "Flash" } }, "crunchy", "crisp textures");
	assert.equal(asked[0][0], "/concept learn crunchy");
	assert.match(asked[0][1], /^V41-ds4 writes its own texts for "crunchy" from your description: crisp textures\.\nA GPU job measures it on V41-ds4 in ds4\.\nds4 keeps serving V41-ds4\.\n/);
	assert.match(asked[1][1], /Flash is unloaded for about 5 to 35 min: .*\nThen the chat calibration, chat working: about 20 min to 2 h\.\n/);
	assert.deepEqual(notes, [["not started", "info"], ["not started", "info"]]);   // declined: nothing posted
});

test("/concept learn NAME: the server says whose texts it uses, or why it cannot be learned here", async () => {
	const { notes, ctx, jobs, server, handlers } = setup();
	const asked: string[] = [];
	const ui = { ...ctx.ui, confirm: async (_title: string, text: string) => { asked.push(text); return false; } };
	const omlx = { ...ctx, ui, model: { provider: "metalscope-omlx", id: "Qwen3.6" } };
	await jobs.learn(omlx, "crunchy", "");                                         // nothing to write from, no description
	// Refused: `taken` is read first (a plan refused early still says "write").
	server.plan = { name: "fluffy", source: "write", texts: null, description: "", taken: "a profile for 'fluffy' already exists for Qwen3.6" };
	await jobs.learn(omlx, "fluffy", "");
	server.plan = { name: "fluffy", source: "shipped", texts: "texts: hand-written by Claude Fable 5.1 (2026-10-05)", description: "Soft.", taken: null };
	await jobs.learn(omlx, "fluffy", "");
	server.plan = { name: "pirate", source: "own", texts: "texts: written by Qwen3.6 (2026-10-24)", description: "Pirate talk.", taken: null };
	await jobs.learn(omlx, "pirate", "");
	// A saved description, none typed: shown, and learned from it.
	server.plan = { name: "toasty", source: "write", texts: null, description: "Toasty things.", taken: null };
	await jobs.learn(omlx, "toasty", "");
	assert.equal(server.planned[0], "/api/concept/learn?name=crunchy&model=Qwen3.6");
	assert.match(notes[0][0], /^say what crunchy means: \/concept learn crunchy </);
	assert.deepEqual(notes[1], ["/concept learn fluffy: a profile for 'fluffy' already exists for Qwen3.6", "error"]);
	assert.equal(asked.length, 3);
	assert.match(asked[0], /^"fluffy" uses metalscope's texts, copied for Qwen3\.6: none are written\.\ntexts: hand-written by Claude Fable 5\.1/);
	assert.match(asked[1], /^"pirate" uses the texts already on Qwen3\.6, from an earlier run: none are written\.\ntexts: written by Qwen3\.6 \(2026-10-24\)\n/);
	assert.match(asked[2], /^Qwen3\.6 writes its own texts for "toasty" from the saved description: Toasty things\.\n/);
	// Accepted: the body as typed (the server fills in the saved description).
	const yes = { ...omlx, ui: { ...ui, confirm: async () => true } };
	await jobs.learn(yes, "toasty", "");
	assert.deepEqual(server.posted.at(-1), { name: "toasty", description: "", model: "Qwen3.6" });
	handlers.session_shutdown({}, ctx);                 // the job's poll stops
});

test("a job's request that fails: the server down said as everywhere else; a bug here as itself", async () => {
	const { notes, ctx, jobs, up, server } = setup();
	const omlx = { ...ctx, model: { provider: "metalscope-omlx", id: "Qwen3.6" } };
	up.delete(OMLX_BASE);
	const down = "oMLX is not answering at 127.0.0.1:9: send a message (Pi starts the model) or run metalscope omlx Qwen3.6";
	await jobs.learn(omlx, "calm", "calm and slow");
	await jobs.calibrate(omlx, "calm");
	await jobs.lensLearn(omlx, {});
	await jobs.jobs(omlx);
	await jobs.cancel(omlx);
	assert.deepEqual(notes, ["learn", "calibrate", "jlens learn", "jobs", "cancel"].map((w) => [`${w}: ${down}`, "error"]));
	up.add(OMLX_BASE);
	server.plan = { name: "calm", source: "own", texts: null, description: "Calm.", taken: null };
	await jobs.learn({ ...omlx, ui: { ...omlx.ui, confirm: async () => { throw new Error("no dialog here"); } } }, "calm", "");
	assert.deepEqual(notes.at(-1), ["learn: no dialog here", "error"]);
});

test("a result Pi missed: one line at startup, no entry, marked seen; never shown twice", async () => {
	const { handlers, entries, notes, seen, server, ctx } = setup();
	server.latest = { kind: "learn", job: "l-1", name: "crunchy", phase: "done", outcome: "done", origin: "pi", updated: Date.now() / 1000, steps: [],
		result: { steerable: true, reason: null, calibration: CAL } };
	handlers.session_start({}, ctx);
	await settle();
	assert.deepEqual(notes, [["learn crunchy finished while Pi was closed: 16/18 good @0.4 · /scope jobs shows the result", "info"]]);
	assert.equal(entries.length, 0);
	assert.deepEqual(seen, ["/api/jobs/l-1/seen"]);
	handlers.session_start({}, ctx);                    // a new session before the server marks it seen: already shown here
	await settle();
	assert.equal(notes.length, 1);
	assert.equal(seen.length, 1);
	handlers.session_shutdown({}, ctx);
});

test("the widget turns while shown, a render every SPIN_MS; set once a job, it draws each poll's state; its timer stops when the job ends and when the session ends", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval", "Date"] });
	const { handlers, server, ctx, widget, raised } = setup();
	const perSecond = async () => {
		widget.renders = 0;
		t.mock.timers.tick(1000);
		await settle();
		return widget.renders;
	};
	server.latest = { ...running, steps: [{ label: "calibrate", status: "running", n: 2, total: 18, started: Date.now() / 1000 - 65 }] };
	handlers.session_start({}, ctx);
	await settle();
	const frame = () => widget.shown.render(80)[1].match(/^ {3}([⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]) calibrate {3}█*░+ 2\/18 · (1m \d\ds)$/).slice(1);
	const [spin, clock] = frame();
	assert.equal(clock, "1m 05s");
	assert.ok(widget.shown.render(80)[0].endsWith(` · ${STOP_LABEL} stop`));
	t.mock.timers.tick(SPIN_MS);                        // the spinner turns between polls
	assert.notEqual(frame()[0], spin);
	t.mock.timers.tick(1000 - SPIN_MS);                 // the clock ticks between polls
	assert.equal(frame()[1], "1m 06s");
	assert.equal(await perSecond(), 1000 / SPIN_MS);
	const first = widget.shown;
	assert.deepEqual(raised, [ctx], "said once, as the widget appears");
	server.latest = { ...server.latest, steps: [{ ...server.latest.steps![0], n: 3 }] };
	t.mock.timers.tick(2000);                           // the poll at 3 s: the same widget (Pi would move one set again), the newer state, one timer
	await settle();
	assert.equal(widget.shown, first);
	assert.match(widget.shown.render(80)[1], / 3\/18 /);
	assert.deepEqual(raised, [ctx]);
	assert.equal(await perSecond(), 1000 / SPIN_MS);
	server.latest = { ...running, phase: "failed", outcome: "failed", error: "stopped" };
	t.mock.timers.tick(2000);                           // the job ends: the widget removed, its timer stopped
	await settle();
	assert.equal(widget.shown, undefined);
	assert.equal(await perSecond(), 0);
	// The session ends while a job runs: the timer stops though Pi has not disposed the widget.
	const s = setup();
	s.server.latest = running;
	s.handlers.session_start({}, s.ctx);
	await settle();
	s.handlers.session_shutdown({}, s.ctx);
	s.widget.renders = 0;
	t.mock.timers.tick(5000);
	assert.equal(s.widget.renders, 0);
	// The job gone from the store (no latest): the widget removed too, not left with a still spinner.
	const g = setup();
	g.server.latest = running;
	g.handlers.session_start({}, g.ctx);
	await settle();
	g.server.latest = undefined;
	t.mock.timers.tick(3000);
	await settle();
	assert.equal(g.widget.shown, undefined);
	g.handlers.session_shutdown({}, g.ctx);
});

test("the widget's render never throws (Pi calls it from its own timer): a job's first state has no steps; a line builder that throws draws nothing", async () => {
	const { handlers, server, ctx, widget } = setup();
	// learn/spawn.py's stub, the first state of every job, as on disk.
	server.latest = { kind: "calibrate", name: "pirate", description: null, model: "M", job: "pirate-calibrate-1", phase: "starting", progress: 0.0,
		started: Date.now() / 1000, updated: Date.now() / 1000, pid: null, log: [], outcome: null, origin: "pi" } as any;
	handlers.session_start({}, ctx);
	await settle();
	assert.match(widget.shown.render(100)[0], new RegExp(`^ calibrating pirate · ░+ 0% · ${STOP_LABEL.replace("+", "\\+")} stop$`));
	handlers.session_shutdown({}, ctx);
	const b = setup();
	b.server.latest = Object.defineProperty({ ...running }, "eta", { get() { throw new Error("bad state"); }, enumerable: true });
	b.handlers.session_start({}, b.ctx);
	await settle();
	assert.deepEqual(b.widget.shown.render(100), []);
	b.handlers.session_shutdown({}, b.ctx);
});

test("the widget's lines fit the width it is given, whatever the text: one column over and Pi throws", async () => {
	const { handlers, server, ctx, widget } = setup();
	const wide = "漢字の結果🎉✅".repeat(8);
	server.latest = { ...running, name: `pirate-${wide}`, warning: `slow ${wide}`, progress: 0.4, eta: Date.now() / 1000 + 600, steps: [
		{ label: "corpus", status: "done", summary: `16 pairs ${wide}`, started: 0, ended: 3 },
		{ label: "items漢", status: "failed", summary: wide, started: 3, ended: 9 },
		{ label: "calibrate", status: "running", n: 2, total: 18, detail: `grading ${wide}`, started: Date.now() / 1000 - 65 },
		{ label: "reload", status: "pending" }] };
	handlers.session_start({}, ctx);
	await settle();
	for (let w = 1; w <= 160; w++) for (const l of widget.shown.render(w)) assert.ok(visibleWidth(l) <= w, `${w}: ${JSON.stringify(l)}`);
	handlers.session_shutdown({}, ctx);
});

test("the stop key: one question, titled as the widget names the job; yes stops it as /scope jobs cancel does; no job: one line", async () => {
	const { notes, server, ctx, jobs, cancels } = setup();
	const asked: string[][] = [];
	let yes = false;
	const options: string[][] = [];
	const ask = { ...ctx, ui: { ...ctx.ui, select: async (title: string, opts: string[]) => (asked.push(title.split("\n")), options.push(opts), opts[yes ? 1 : 0]) } };
	server.latest = { kind: "learn", job: "l-1", name: "pirate", phase: "presence", origin: "pi", steps: [] };
	await jobs.cancel(ask, true);
	// Under the question, what stopping it means: the command its result entry will give.
	assert.deepEqual(asked, [["Stop learning pirate?", "/concept learn pirate <what it means> starts it again"]]);
	assert.deepEqual([cancels, notes], [[], []]);       // no: nothing posted, nothing said
	assert.deepEqual(options, [["No, keep it running", "Yes, stop it"]]);   // enter takes the first: a ≈ typed in a message, then enter, stops nothing
	await Promise.all([jobs.cancel(ask, true), jobs.cancel(ask, true)]);   // two fast presses: one question
	assert.equal(asked.length, 2);
	asked.pop();
	yes = true;
	await jobs.cancel(ask, true);
	assert.deepEqual(cancels, ["/api/jobs/l-1/cancel"]);
	assert.deepEqual(notes, [["stopping learn pirate: the model is reloaded when it has stopped", "info"]]);
	server.latest = { kind: "lens-learn", job: "f-1", name: "Qwen3.6", phase: "fitting", origin: "pi", steps: [] };
	await jobs.cancel(ask, true);
	assert.deepEqual(asked.at(-1), ["Stop learning a lens for Qwen3.6?", "/jlens learn until HH:MM starts it again"]);
	await jobs.cancel(ctx);                             // /scope jobs cancel: no question
	assert.equal(asked.length, 3);
	assert.equal(cancels.length, 3);
	server.latest = { ...server.latest, outcome: "done" };
	await jobs.cancel(ask, true);
	assert.equal(asked.length, 3);
	assert.deepEqual(notes.at(-1), ["no job is running", "info"]);
});

test("the stop key as raw input (alt+x, or ≈: Option+x on a Mac): while a job runs, one question and the input consumed; no job, nothing", async (t) => {
	const { handlers, server, ctx, press, listeners } = setup();
	t.after(() => handlers.session_shutdown({}, ctx));   // a failed assertion must not leave the poll's timer running
	const asked: string[] = [];
	ctx.ui = { ...ctx.ui, select: async (title: string) => void asked.push(title.split("\n")[0]) } as any;
	handlers.session_start({}, ctx);
	await settle();
	assert.equal(listeners.size, 1);
	assert.equal(press("≈"), false, "no job: ≈ is typed as usual");
	assert.equal(press("\x1bx"), false);
	await settle();
	assert.deepEqual(asked, []);
	server.latest = running;
	handlers.session_start({}, ctx);                    // a reload or a new session: one live handler still
	await settle();
	assert.equal(listeners.size, 1);
	for (const key of ["≈", "\x1bx", "\x1b[8776u"]) {
		assert.equal(press(key), true, JSON.stringify(key));
		await settle();
	}
	assert.deepEqual(asked, ["Stop calibrating calm?", "Stop calibrating calm?", "Stop calibrating calm?"]);
	assert.equal(press("x"), false);
	assert.equal(press("\x1b[8776;1:3u"), false, "a release");
	handlers.session_shutdown({}, ctx);
	assert.equal(listeners.size, 0);
});

test("the stop key pressed again while its question is open: consumed (no ≈ typed into the dialog), nothing more asked", async (t) => {
	const { handlers, server, ctx, press } = setup();
	t.after(() => handlers.session_shutdown({}, ctx));
	const asked: string[] = [];
	let answer!: (choice?: string) => void;
	ctx.ui = { ...ctx.ui, select: (title: string) => (asked.push(title), new Promise<string | undefined>((r) => (answer = r))) } as any;
	server.latest = running;
	handlers.session_start({}, ctx);
	await settle();
	assert.equal(press("≈"), true);
	await settle();
	assert.equal(press("≈"), true);
	assert.equal(press("\x1bx"), true);
	await settle();
	assert.equal(asked.length, 1);
	answer(undefined);
	await settle();
});

test("no terminal (pi -p, RPC, a ui without onTerminalInput or one that throws): no handler, nothing thrown", async (t) => {
	const ends: (() => void)[] = [];
	t.after(() => ends.forEach((end) => end()));
	for (const ui of [undefined, {}, { onTerminalInput: () => { throw new Error("no terminal"); } }]) for (const hasUI of [false, true]) {
		const { handlers, ctx, server } = setup();
		server.latest = running;
		const c = { ...ctx, hasUI, ui: ui && { ...ctx.ui, onTerminalInput: undefined, ...ui } };
		ends.push(() => handlers.session_shutdown({}, c));
		handlers.session_start({}, c);
		await settle();
	}
});

test("the stop key's handler never throws (Pi calls it with nothing around it): a stale ctx's dialog and notes", async (t) => {
	const { handlers, server, ctx, press } = setup();
	t.after(() => handlers.session_shutdown({}, ctx));
	server.latest = running;
	handlers.session_start({}, ctx);
	await settle();
	ctx.ui.select = () => { throw new Error("stale ctx"); };
	ctx.ui.notify = () => { throw new Error("stale ctx"); };
	assert.equal(press("≈"), true);
	await settle();
	let asked = 0;
	ctx.ui.notify = () => {};
	ctx.ui.select = async () => void asked++;
	assert.equal(press("≈"), true, "not left dead");
	await settle();
	assert.equal(asked, 1);
});

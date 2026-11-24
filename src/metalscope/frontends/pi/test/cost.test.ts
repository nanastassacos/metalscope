// What a frame costs: the dock's redraw cap (Dock.render) and the counts kept as a reply grows, each the same as
// working it out whole.
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { Dock, REDRAW_MAX_MS, REDRAW_SHARE, TRAIL_MS } from "../dock.ts";
import { ALONE, UNSURE, offTop, plainWidth, replyCounts, textWidth } from "../draw.ts";
import { type Live, onEvent } from "../state.ts";
import { createUi } from "../widget.ts";

const theme = {
	fg: (c: string, s: string) => `\x1b[38;5;${c.length}m${s}\x1b[39m`,
	bg: (_c: string, s: string) => `\x1b[48;5;236m${s}\x1b[49m`,
	bold: (s: string) => `\x1b[1m${s}\x1b[22m`,
};

test("textWidth: pi-tui's width; ALONE's characters each a grapheme of their own beside ASCII, colour codes and each other", () => {
	const seg = new Intl.Segmenter(undefined, { granularity: "grapheme" }), all: string[] = [];
	for (const [a, b] of ALONE) for (let c = a; c <= b; c++) all.push(String.fromCharCode(c));
	all.forEach((ch, k) => {
		const next = all[(k * 7919 + 13) % all.length], s = `a${ch}\x1b[38;2;1;22;255m${ch}${next}\x1b[0m b${next}`;
		assert.equal([...seg.segment(s.replace(/\x1b\[[0-9;]*m/g, ""))].length, 7, JSON.stringify(s));
		assert.equal(textWidth(s), visibleWidth(s), JSON.stringify(s));
		assert.ok(plainWidth(s) >= 0);
	});
	// Anything else is pi-tui's to measure.
	for (const s of ["é", "中文", "🙂x", "a\tb", "\x1b]8;;u\x07x\x1b]8;;\x07", "\x1b[2Kx", "x‍y", "a­b", "\x1b[3", "\n"]) {
		assert.equal(textWidth(s), visibleWidth(s), JSON.stringify(s));
		assert.equal(plainWidth(s), -1, JSON.stringify(s));
	}
});

test("replyCounts: as a reply grows, what counting it whole gives; an array cut under it counted afresh", () => {
	const toks: { text: string; p: number | null; alts: [string, number][] }[] = [];
	for (let i = 0; i < 300; i++) {
		toks.push({ text: i === 200 ? "ok</think>" : ` w${i}`, p: i % 13 ? (i % 7) / 7 : null, alts: [[" a", i % 5 ? 0.1 : 0.7]] });
		const c = replyCounts(toks), ps = toks.map((t) => t.p ?? 0);
		assert.equal(c.sum, ps.reduce((a, b) => a + b, 0));
		assert.deepEqual(c.unsure, ps.flatMap((p, k) => (p < UNSURE ? [k] : [])));
		assert.equal(c.off, toks.filter((t) => offTop(t.p, t.alts)).length);
		assert.equal(c.think, toks.findIndex((t) => t.text.includes("</think>")));
	}
	toks.length = 50;
	toks.push({ text: "</think>", p: 0.95, alts: [] });
	assert.deepEqual([replyCounts(toks).think, replyCounts(toks).unsure.length], [50, toks.filter((t) => (t.p ?? 0) < UNSURE).length]);
});

/** A one-panel board (MAP in focus: its title counts the tokens) following a reply, on a clock the test moves; a draw
 * takes `took` ms (charged at its first host.flags: the header's). */
function capped(clock: { t: number; took: number }, extra: Record<string, unknown> = {}) {
	const live: Live = { base: "", clientId: "me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r" });
	const pi = { terminal: { rows: 40, columns: 120 }, asked: 0, requestRender() { pi.asked++; } };
	let charged = false;
	const host: any = { live, captions: false, close() {}, unfocus() {}, now: () => ((charged = false), clock.t), panel: "MAP",
		flags: () => (charged ? {} : ((charged = true), (clock.t += clock.took), {})), ...extra };
	const d = new Dock(pi, theme, host);
	// As widget.ts wires it: anything but the stream's progress invalidates.
	live.requestRender = (progress) => {
		if (!progress) d.invalidate();
		pi.requestRender();
	};
	const token = (i: number) => onEvent(live, { type: "token", rid: "r", i, id: i, text: ` w${i}`, p: 0.9, entropy: 0, alts: [], n_layers: null });
	token(0);
	return { d, pi, live, host, token };
}

test("the redraw cap: without a clock every render draws (as the tests before it render)", () => {
	const live: Live = { base: "", clientId: "me", level: "on", connected: true, reply: { rid: "r", tokens: [], done: false, prefill: 0 } };
	const d = new Dock({ terminal: { rows: 40 }, requestRender() {} }, theme, { live, captions: false, close() {}, unfocus() {} });
	assert.notEqual(d.render(80), d.render(80));
});

test("the redraw cap: the lines last drawn while only stream progress is new, for the draw's time over REDRAW_SHARE, at most REDRAW_MAX_MS", () => {
	assert.equal(REDRAW_SHARE, 0.25);
	for (const took of [1, 10, 30, 50]) {
		const clock = { t: 1000, took }, { d, token } = capped(clock);
		const wait = Math.min(REDRAW_MAX_MS, took / REDRAW_SHARE);
		const first = d.render(80);
		token(1);
		clock.t = 1000 + wait - 0.5;
		assert.equal(d.render(80), first, `took ${took}: still waiting`);
		clock.t = 1000 + wait;
		const next = d.render(80);
		assert.notEqual(next, first, `took ${took}: drawn once the wait is over`);
		assert.match(next.join("\n"), / 2 tok /);
	}
});

test("the redraw cap: a key, invalidate, stepIn, a draw the dock asks for, another width or height: drawn at once", async () => {
	let answer: (x: any) => void = () => {};
	const clock = { t: 0, took: 10 }, { d, pi } = capped(clock, { steering: () => new Promise((r) => (answer = r)) });
	const fresh = (what: string, act: () => void | Promise<void>, width = 80) => async () => {
		const before = d.render(80);
		assert.equal(d.render(80), before, `${what}: cached before`);
		await act();
		assert.notEqual(d.render(width), before, what);
	};
	await fresh("a key that does nothing here", () => d.handleInput("z"))();
	await fresh("invalidate", () => d.invalidate())();
	await fresh("stepIn", () => d.stepIn(0))();
	await fresh("the concepts' answer", async () => {
		const asked = pi.asked;
		answer({ model: "", concepts: [] });
		await new Promise((r) => setTimeout(r, 0));
		assert.equal(pi.asked, asked + 1);
	})();
	await fresh("another width", () => {}, 81)();
	d.render(80);
	await fresh("another height", () => void (pi.terminal.rows = 41))();
});

test("the redraw cap: a reply's end, a new reply, the server going: drawn at once; tokens and the prompt's count wait", () => {
	const clock = { t: 0, took: 10 }, { d, live, token } = capped(clock);
	let before = d.render(80);
	token(1);
	onEvent(live, { type: "prefill", rid: "r", read: 5 });
	assert.equal(d.render(80), before);
	onEvent(live, { type: "end", rid: "r", n_tokens: 2, trace_id: "t" });
	before = d.render(80);
	assert.match(before.join("\n"), / 2 tok /);
	onEvent(live, { type: "start", rid: "r2" });
	assert.notEqual(d.render(80), before);
	before = d.render(80);
	live.connected = false;
	live.requestRender?.();
	assert.match(d.render(80).join("\n"), /not answering/);
});

test("the redraw cap: a frame left waiting is drawn TRAIL_MS after the wait (the trailing draw), unless one drew first", (t) => {
	t.mock.timers.enable({ apis: ["setTimeout"] });
	const clock = { t: 0, took: 20 }, { d, pi, token } = capped(clock);
	const first = d.render(80);   // drawn at 0 to 20: the next no sooner than 80
	token(1);
	clock.t = 30;
	const asked = pi.asked;
	assert.equal(d.render(80), first);
	assert.equal(d.render(80), first);   // one trailing draw however many frames waited
	t.mock.timers.tick(50 + TRAIL_MS - 1);
	assert.equal(pi.asked, asked);
	clock.t = 80 + TRAIL_MS;
	t.mock.timers.tick(1);
	assert.equal(pi.asked, asked + 1);   // Pi asked to draw
	const now = d.render(80);
	assert.match(now.join("\n"), / 2 tok /);
	t.mock.timers.tick(1000);
	assert.equal(pi.asked, asked + 1);
	// A frame drawn after the wait (the stream's next) first: no trailing draw.
	token(2);
	const streamed = pi.asked;   // the token's own request
	clock.t += 1;
	assert.equal(d.render(80), now);
	clock.t += 80;
	assert.notEqual(d.render(80), now);
	t.mock.timers.tick(1000);
	assert.equal(pi.asked, streamed);
});

test("widget.ts: the dock on Pi's clock; the stream's progress leaves it waiting, anything else invalidates", () => {
	let dock: any;
	const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {},
		custom: (make: any) => new Promise<void>(() => { dock = make({ requestRender() {}, terminal: { rows: 40, columns: 160 } }, {}, {}, () => {}); }) } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: true } as any;
	const ui = createUi({ live } as any, join(mkdtempSync(join(tmpdir(), "ms-prefs-")), "pi.json"));
	ui.showDock(ctx);
	assert.equal(typeof dock.host.now(), "number");
	dock.render(80);
	live.requestRender(true);
	assert.equal(dock.dirty, false);
	live.requestRender();
	assert.equal(dock.dirty, true);
	dock.render(80);
	ui.updateUi(ctx);   // /steer, /scope on|off, another model: the status line and the dock
	assert.equal(dock.dirty, true);
});

test("the redraw cap: a reply adopted by its first token (no start: a reconnect, an overlapping stream) is drawn at once", () => {
	const clock = { t: 0, took: 10 }, { d, live } = capped(clock);
	const before = d.render(80);
	onEvent(live, { type: "token", rid: "r2", i: 0, id: 0, text: " BRANDNEW", p: 0.9, entropy: 0, alts: [], n_layers: null });
	const now = d.render(80);
	assert.notEqual(now, before);
	assert.match(now.join("\n"), /BRANDNEW/);
});

test("the redraw cap: a render after dispose shows the last lines and arms no timer", (t) => {
	t.mock.timers.enable({ apis: ["setTimeout"] });
	const clock = { t: 0, took: 10 }, { d, pi, token } = capped(clock);
	const last = d.render(80);
	d.dispose();
	token(1);
	const asked = pi.asked;
	clock.t += 1;
	assert.equal(d.render(80), last);
	t.mock.timers.tick(1000);
	assert.equal(pi.asked, asked);
});

test("engram inspection (g): its answer read once (off, never); on, a frame reads the tokens it draws, not the whole reply", async () => {
	/** The board following a finished reply of `n` tokens whose lookup answers at once; `reads.answer` counts the
	 * answer's reads, `reads.level` a frame's reads of its tints once `count` wraps them. */
	const reads = { answer: 0, level: 0 };
	async function board(n: number) {
		const live: Live = { base: "", clientId: "me", level: "on", connected: true };
		onEvent(live, { type: "start", rid: "r" });
		for (let i = 0; i < n; i++) onEvent(live, { type: "token", rid: "r", i, id: i, text: ` w${i % 50}`, p: (i % 9) / 9, entropy: 0, alts: [], n_layers: null });
		onEvent(live, { type: "end", rid: "r", n_tokens: n, trace_id: "T" });
		const ids = Array.from({ length: n - 1 }, (_, i) => i), on = ids.map((i) => (i % 7 ? -1 : -0.2)), off = ids.map(() => -1);
		const top = ids.map((i) => [i]), lp = ids.map(() => [-1]);
		const answer = { trace: "T", ngram: 3, ids, lp_off: off, get lp_on() { reads.answer++; return on; }, top_ids_on: top, top_lp_on: lp, top_ids_off: top, top_lp_off: lp, vocab: {} };
		const host: any = { live, captions: false, close() {}, unfocus() {}, fetchJson: async () => null, caps: () => ["lookup"], lookup: async () => answer };
		const d = new Dock({ terminal: { rows: 67 }, requestRender() {} }, theme, host);
		const count = () => {
			const v = (d as any).engram.views.get(live.reply);
			v.level = new Proxy(v.level, { get: (t, k) => (typeof k === "string" && /^\d+$/.test(k) && reads.level++, Reflect.get(t, k)) });
		};
		return { d, host, count };
	}
	const { d } = await board(10_000);
	d.stepIn(-1);
	for (let k = 0; k < 5; k++) d.render(104);
	assert.equal(reads.answer, 0, "off: never read");
	d.handleInput("g");
	await new Promise((r) => setTimeout(r, 0));
	d.render(104);
	const once = reads.answer;
	assert.ok(once > 0);
	for (let k = 0; k < 20; k++) d.render(104);
	d.handleInput("g");                                                                         // off
	for (let k = 0; k < 20; k++) d.render(104);
	d.handleInput("g");                                                                         // on again: kept
	for (let k = 0; k < 20; k++) d.render(104);
	assert.equal(reads.answer, once, "worked out once");
	// A frame's tint reads: MAP's cells (two rows of tokens a line) and the strip's tokens, about the same at 2,000
	// tokens and 10,000, under half the reply.
	const frame = async (n: number) => {
		const b = await board(n);
		b.d.stepIn(-1);
		b.d.handleInput("g");
		await new Promise((r) => setTimeout(r, 0));
		b.d.render(104);
		b.count();
		reads.level = 0;
		b.d.render(104);
		return reads.level;
	};
	const small = await frame(2000), big = await frame(10_000);
	assert.ok(small > 0 && big <= small * 1.2 && big < 10_000 / 2, `${small} ${big}`);
});

// g in the dock: engram inspection on one finished reply (lookup.ts), on the board, the one-panel board and the mini dock.
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { KEYS_W, MINI_WIDTH, type Panel, boardKeys, expertKeys, layerKeys, mapCells, mapPanel, miniKeys, sessionMap } from "../board.ts";
import { COLOURS, EXPLAIN, LABELS, colourCode, dockTheme, wrapTokens } from "../draw.ts";
import { Dock } from "../dock.ts";
import { AHEAD, FULL, HELPED, type LookupAnswer, type LookupProgress, askAgain, MAP_BLOCK, MOST, STRONG, back, engramOf, isAnswer, lookupBar, lookupLine, lookupMap, lookupPaint, lookupRgb,
	lookupTokens, lookupView, progressAt, BAR_TICK_MS, SWEEP_MS, SWEEP_PASSES, SWEEP_TICK_MS, sweepAt } from "../lookup.ts";
import { mix } from "../lights.ts";
import { type Catalogue, DS4, OMLX, registerCatalogue } from "../backends.ts";
import { DS4_URL, type Server, refreshFlags } from "../provider.ts";
import { type Live, type Reply, mapReply, onEvent } from "../state.ts";
import { createUi } from "../widget.ts";

const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const text = (lines: string[]) => lines.map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd());
const tick = () => new Promise((r) => setTimeout(r, 0));
const piTheme = { fg: (_c: string, s: string) => s, bg: (_c: string, s: string) => s, bold: (s: string) => s };
const TEXTS = [" Trade", " moved", " across", " the", " Mediterranean", " Sea", ",", " the", " Mediterranean", " Sea", "."];
const TITLE = LABELS.engramTitle;
const BLUE = (l: number) => mix(COLOURS.dim, COLOURS.help, [0.5, 0.78, 1][l - 1]);
const WHITE = COLOURS.text.join(), BLOCK = MAP_BLOCK.join();
/** Half-block lines (truecolor) back to their cells, "r,g,b" each: a row of cells a half line, the upper first. */
const halfRows = (lines: string[]): string[][] => lines.flatMap((l) => {
	const up: string[] = [], lo: string[] = [];
	let f = "", b = "";
	for (const m of l.matchAll(/\x1b\[([34])8;2;(\d+);(\d+);(\d+)m|▀/g))
		if (m[0] === "▀") up.push(f), lo.push(b);
		else if (m[1] === "3") f = `${m[2]},${m[3]},${m[4]}`;
		else b = `${m[2]},${m[3]},${m[4]}`;
	return [up, lo];
});
/** MAP's cells on the board, in order: the lines under its title, in the left column. */
function mapCellsIn(raw: string[]): string[] {
	const t = text(raw), at = t.findIndex((l) => /^ ?MAP /.test(l)), ls: string[] = [];
	for (let i = at + 1; at >= 0 && i < raw.length && raw[i].split("│")[0].includes("▀"); i++) ls.push(raw[i].split("│")[0]);
	return halfRows(ls).flat();
}

/** A finished reply of `n` tokens (TEXTS over and over), trace T1. */
function reply(n = TEXTS.length, done = true): Live {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r", model: "Flash" });
	for (let i = 0; i < n; i++)
		onEvent(live, { type: "token", rid: "r", i, id: 100 + i, text: TEXTS[i % TEXTS.length], p: 0.95, entropy: 0, alts: [], n_layers: null });
	if (done) onEvent(live, { type: "end", rid: "r", n_tokens: n, trace_id: "T1" });
	return live;
}
/** The answer for reply `n` tokens long: every fed token (all but the last) at -1 nats with the engram, `gain` more
 * than without it where given; each reading's top three at each row (the written token first, then 7 and 8 with the
 * engram, 7 and 9 without), with their texts. */
function answer(n: number, gain: Record<number, number> = {}): LookupAnswer {
	const ids = Array.from({ length: n - 1 }, (_, i) => 100 + i), lp_on = ids.map(() => -1), lp_off = ids.map((_, i) => -1 - (gain[i] ?? 0));
	return { trace: "T1", ngram: 3, ids, lp_on, lp_off, top_ids_on: ids.map((id) => [id, 7, 8]), top_lp_on: lp_on.map((x) => [x, -2, -3]),
		top_ids_off: ids.map((id) => [7, id, 9]), top_lp_off: lp_off.map((x) => [Math.max(x, -0.5), x, -4]), vocab: { 7: " seven", 8: " eight", 9: " nine" } };
}
/** A dock (the board at 104 by 67, the one-panel board narrower) on `live`, with the cap or not; the engram answered by
 * `ask`; each ask's id (`ids`) and each stop (`stops`: trace and ask) recorded. */
function dock(live: Live, caps = ["lookup"], ask?: (trace: string) => Promise<LookupAnswer>, extra: Record<string, any> = {}) {
	const asked: string[] = [], ids: string[] = [], stops: [string, string][] = [];
	const host: any = { live, captions: false, close() {}, unfocus() {}, fetchJson: async () => null, caps: () => caps,
		lookup: (t: string, a: string) => (asked.push(t), ids.push(a), (ask ?? (async () => answer(live.reply!.tokens.length, { 5: 1.95 })))(t)),
		stopLookup: async (t: string, a: string) => void stops.push([t, a]), ...extra };
	const d = new Dock({ terminal: { rows: 67 }, requestRender() {} }, piTheme, host);
	return { d, host, asked, ids, stops };
}
/** The readings' progress for ask `ask` as the live stream sends it. */
const progress = (ask: string, reading: number, done: number, next: number, elapsed: number, expect: number): LookupProgress =>
	({ type: "lookup", ask, reading, done, next, elapsed, expect });
const titleLine = (lines: string[]) => lines.find((l) => l.startsWith(TITLE) || l.startsWith("REPLY"));

test("lookupView: helped tokens in three strengths (helped, doubled, quadrupled); each token's engram, fewer at the start; the last token and other traces' tokens unread", () => {
	const live = reply(), r = live.reply!, n = r.tokens.length;
	const v = lookupView(r, answer(n, { 0: 0.8, 1: 0.7, 2: 0.2, 5: 1.5, 7: 0.05 }));
	assert.deepEqual([...v.level], [2, 2, 1, 0, 0, 3, 0, 0, 0, 0, 0]);
	assert.deepEqual([HELPED, STRONG, FULL], [0.1, Math.log(2), Math.log(4)]);
	assert.equal(v.helped, 4);
	assert.ok(Number.isNaN(v.on[n - 1]) && Number.isNaN(v.off[n - 1]) && v.level[n - 1] === 0 && v.at[n - 1] === -1);   // the last token: never fed
	assert.deepEqual([...v.at].slice(0, 3), [0, 1, 2]);
	// The cursor's engram: the 3 before it, fewer at the start, helped or not; none on the last token or during a sweep.
	assert.deepEqual([0, 1, 2, 5, 7, n - 1].map((k) => engramOf(v, k)), [[0, 0], [0, 1], [0, 2], [2, 5], [4, 7], [0, 0]]);
	assert.deepEqual(engramOf(v, 5, { pass: 1, at: 3 }), [0, 0]);
	// DeepSeek V4.1's Engram looks up 4-grams (the answer's ngram): the 4 before it.
	const four = lookupView(r, { ...answer(n, { 5: 1.5 }), ngram: 4 });
	assert.deepEqual([2, 5, 7].map((k) => engramOf(four, k)), [[0, 2], [1, 5], [3, 7]]);
	// An id that does not agree is not read; an edit's reply (from 4): only its own trace's tokens, by i - from.
	const odd = answer(n, { 5: 2 });
	odd.ids[3] = 7;
	assert.ok(Number.isNaN(lookupView(r, odd).on[3]));
	const edited: Reply = { ...r, from: 4, tokens: r.tokens.map((t) => ({ ...t, id: t.i >= 4 ? 100 + t.i - 4 : t.id })) };
	const e = lookupView(edited, answer(n - 4, { 1: 1 }));
	assert.deepEqual([...e.on].map((x) => !Number.isNaN(x)), r.tokens.map((t) => t.i >= 4 && t.i < n - 1));
	assert.deepEqual([...e.level].map((l, k) => (l ? k : -1)).filter((k) => k >= 0), [5]);
});

test("engram colours: one muted blue in three strengths, the rest quiet, no orange; the engram on a block, the token it predicted white in MAP, bold and underlined in its own colour in the text", () => {
	const live = reply(), r = live.reply!, n = r.tokens.length, v = lookupView(r, answer(n, { 1: 0.7, 2: 0.2, 5: 2.5 }));
	assert.deepEqual(COLOURS.help, [110, 150, 200]);
	// MAP's cells: by strength; the rest (and the last token) one quiet colour; no white focus, no orange.
	const p = lookupMap(v, 5), run = { n, p: () => 0.95, think: -1, shade: p.shade };
	const cells = mapCells([run], 8, 0, 16, -1);
	assert.deepEqual([cells[1], cells[2], cells[5]], [BLUE(2), BLUE(1), BLUE(3)]);
	assert.deepEqual(cells[0], cells[n - 1]);
	assert.ok(!cells.some((c) => c.join() === COLOURS.accent.join() || c.join() === COLOURS.text.join()));
	assert.deepEqual([p.engram, p.pred, p.count], [[2, 5], 5, "3 helped"]);
	// MAP drawn: the engram's cells in their own clear blue, the predicted one white (the focus's colour, not drawn).
	const dt = dockTheme(), map = mapPanel(dt, r.tokens, 5, 8, 3, false, true, 0, p);
	assert.equal(plain(map[1]), "▀".repeat(8));
	const own = (cells as number[][]).map((c) => c.join());
	assert.deepEqual(halfRows(map.slice(1))[0], [own[0], own[1], BLOCK, BLOCK, BLOCK, WHITE, own[6], own[7]]);
	assert.deepEqual(halfRows(map.slice(1))[1], own.slice(8, 16), "the second row, the first line's lower half");
	// Clearly unlike the quiet cells and the background beside it.
	const lum = (c: number[]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
	assert.ok(lum(MAP_BLOCK) - lum(cells[0]) > 20 && lum(MAP_BLOCK) - lum(COLOURS.bg) > 40, `${MAP_BLOCK} ${cells[0]}`);
	// The text: helped in their blue (foreground), the rest faint; the block behind the engram; the cursor bold underlined.
	const th = { fg: (c: string, s: string) => `<${c}>${s}</>`, bg: (_c: string, s: string) => s, bold: (s: string) => `*${s}*` };
	const t = lookupPaint(th, v, 5);
	assert.equal(t.at(0, " Trade"), "<faint> Trade</>");
	assert.equal(t.at(1, " moved"), colourCode(BLUE(2), "truecolor") + " moved\x1b[39m");
	assert.equal(t.at(2, " across"), " " + colourCode(MAP_BLOCK, "truecolor", true) + colourCode(COLOURS.text, "truecolor") + "across\x1b[39m\x1b[49m", "the block from the first word");
	assert.equal(t.at(3, " the"), colourCode(MAP_BLOCK, "truecolor", true) + colourCode(COLOURS.text, "truecolor") + " the\x1b[39m\x1b[49m");
	assert.equal(t.at(5, " Sea"), `\x1b[1m\x1b[4m${colourCode(BLUE(3), "truecolor")} Sea\x1b[39m\x1b[24m\x1b[22m`);
	assert.ok(t.cursor);
	assert.equal(lookupPaint(th, v, n - 1).cursor, false, "a token not read: wrapTokens draws its cursor (inverse)");
	// Through wrapTokens (the strip passes `from`): the cursor not inverse; no confidence colours.
	const lines = wrapTokens(th, r.tokens.slice(2), 80, 3, -1, undefined, lookupPaint(th, v, 5, 2)).lines.join("");
	assert.ok(!lines.includes("\x1b[7m") && !lines.includes("<success>") && !lines.includes("<accent>"), lines);
	assert.notEqual(wrapTokens(th, r.tokens, 80).lines.join(""), wrapTokens(th, r.tokens, 80, -1, -1, undefined, t).lines.join(""));
	// MAP full size (the session) draws the shown reply's cells and marks as MAP does, behind the replies before it.
	assert.deepEqual(sessionMap(dt, [], r, 5, 8, 3, 0, p).lines.slice(1), map.slice(1));
	assert.match(plain(mapPanel(dt, r.tokens, 5, 30, 3, false, true, 0, p)[0]), /^MAP {2}11 tok · 3 helped$/);
	const a16 = mapReply({ rid: "a", tokens: [...r.tokens, ...r.tokens.slice(0, 5)] });                   // two rows: a line of its own
	assert.deepEqual(sessionMap(dt, [a16], r, 5, 8, 4, 0, p).lines.slice(2, 3), map.slice(1, 2));
});

test("lookupLine: the engram and the token, both probabilities or no change; shorter where narrow; the last token, the reply's first", () => {
	const live = reply(), r = live.reply!, n = r.tokens.length;
	const a = answer(n, {});
	a.lp_off[5] = Math.log(0.12);
	a.lp_on[5] = Math.log(0.85);
	a.lp_off[0] = Math.log(0.3);
	a.lp_on[0] = Math.log(0.6);
	const v = lookupView(r, a);
	assert.equal(lookupLine(v, r.tokens, 5, 80), 'engram " across the Mediterranean" → " Sea"   without 12% → with 85%');
	assert.equal(lookupLine(v, r.tokens, 5, 40), '" across the Mediter… → " Sea" 12% → 85%');
	assert.equal(lookupLine(v, r.tokens, 0, 80), 'engram the prompt → " Trade"   without 30% → with 60%');
	assert.equal(lookupLine(v, r.tokens, 3, 80), 'engram " Trade moved across" → " the"   no change');
	assert.equal(lookupLine(v, r.tokens, n - 1, 80), '"." last token: not read');
	assert.equal(lookupLine(v, r.tokens, -1, 80), "2 tokens helped");
});

test("TOKENS in engram inspection: the top choices without and with, a percentage and a bar each; narrower, the without bar goes, then both; the written token's own numbers", () => {
	const live = reply(), r = live.reply!, n = r.tokens.length, v = lookupView(r, answer(n, { 5: 1.95 }));
	const th = dockTheme(), at = (w: number, a = v) => lookupTokens(th, a, r.tokens, 5, w, false)!.map(plain);
	const wide = at(50);
	assert.deepEqual(wide.map((l) => visibleWidth(l) <= 50), wide.map(() => true));
	assert.match(wide[0], /^TOKENS {2}#5 +without +with$/);
	assert.match(wide[1], /^▸ " Sea" +5\.2% [▏▎▍▌▋▊▉█]+ +37% █+/);                     // written: without e^-2.95, with e^-1
	assert.match(wide[2], /^ {2}" seven" +61% █+[▏▎▍▌▋▊▉]? +14% █/);                        // 7: with e^-2, without e^-0.5
	assert.match(wide[3], /^ {2}" eight" +<1\.8% +5\.0% /);                        // 8: not in the without top three (its least: e^-4)
	assert.equal(wide.length, 4);
	assert.equal(wide[1].indexOf("%", 20), wide[2].indexOf("%", 20), "the without percentages line up");
	// Narrower: both percentages and the with bar; narrower still, the percentages alone.
	const one = at(30), none = at(26);
	assert.ok(one.every((l) => visibleWidth(l) <= 30) && (one[1].match(/█|▏|▎|▍|▌|▋|▊|▉/g) ?? []).length > 0, one.join("\n"));
	assert.match(one[2], /61% +14% [█▏▎▍▌▋▊▉]+ *$/);
	assert.match(none[2], /^ {2}" seven" +61% +14%$/);
	// Too few lines for every choice: the written token's row kept in the last line, as TOKENS keeps it (keepRow).
	const low = answer(n, { 5: 1.95 });
	low.top_ids_on = low.ids.map((id) => [7, 8, 9, id]);
	low.top_lp_on = low.lp_on.map((x) => [-0.5, -2, -3, x]);
	const short = lookupTokens(th, lookupView(r, low), r.tokens, 5, 50, false, 3)!.map(plain);
	assert.equal(short.length, 3);
	assert.match(short[1], /^ {2}" seven"/);
	assert.match(short[2], /^▸ " Sea" +5\.2% .* 37% /, "the written token's row, its without and with");
	// The written token outside the without reading's top 10: its own without (the title's), not a floor.
	const out = answer(n, { 5: 1.95 });
	out.top_ids_off[5] = [7, 9, 8];
	const own = at(50, lookupView(r, out));
	assert.match(own[1], /^▸ " Sea" +5\.2% /);
	assert.match(lookupLine(lookupView(r, out), r.tokens, 5, 80), /without 5\.2% → with 37%$/);
	// An answer without the top choices is not one (the server always sends them).
	const { top_ids_on: _a, ...bare } = out;
	assert.equal(isAnswer(bare, "T1"), false);
	assert.equal(isAnswer({ ...out, ngram: undefined }, "T1"), false);     // an older server's answer: asked again
	assert.equal(isAnswer(out, "T1"), true);
	// A token not read: none (the dock draws TOKENS as ever).
	assert.equal(lookupTokens(th, v, r.tokens, n - 1, 50, false), undefined);
});

test("g on every key line where it works: the board's panels, the one-panel board, the mini dock; g close while on; the last part to go where the line is tight", async () => {
	const G = LABELS.engramKey;
	for (const p of ["LENS", "MAP", "EXPERTS", "TOKENS"] as Panel[]) for (const windowed of [false, true]) {
		const line = boardKeys(p, true, windowed, undefined, false, false, true, true, "", KEYS_W, G);
		assert.ok(line.includes(G) && visibleWidth(line) <= KEYS_W, `${p} ${windowed}: ${line}`);
	}
	// Short only where whole it would go. The tightest: LENS stepped to a token, a steer on, the lens steps shown: the
	// others go first, g short (a dock 100 wide: 97), whole at 101 (a dock 104 wide).
	assert.equal(boardKeys("LENS", true, false, undefined, false, false, true, true, "", KEYS_W, G),
		"tab experts · ←/→ token · ↑/↓ layer · -/+ layers · enter edit · g engram inspection · 1-6 panel");
	const TIGHT = "tab experts · ←/→ token · f live · ↑/↓ layer · -/+ layers · enter edit · x off", TIGHT_97 = TIGHT + " · g engram", TIGHT_101 = TIGHT + " · g engram inspection";
	assert.equal(boardKeys("LENS", true, false, undefined, true, true, true, true, "", KEYS_W, G), TIGHT_97);
	assert.equal(boardKeys("LENS", true, false, undefined, true, true, true, true, "", 101, G), TIGHT_101);
	assert.equal(boardKeys("LENS", true, false, undefined, true, false, false, true, "", KEYS_W, G),
		"tab experts · ←/→ token · f live · ↑/↓ layer · enter edit · g engram inspection · 1-6 panel");
	assert.equal(boardKeys("MAP", true, false, undefined, true, false, false, true, "", KEYS_W, LABELS.engramClose),
		"tab lens · ←/→ token · f live · u/U unsure · g close · 1-6 panel · ? explain · esc back");
	assert.ok(!boardKeys("MAP", true).includes("g "), "no cap: no g");
	// The mini dock: 50 columns; where tighter (33), ctrl+\ closes, [ ] more, then f live go first; g short where that keeps one.
	assert.equal(miniKeys(true, false, G), "m full · [ ] more · g engram · ctrl+\\ closes");
	assert.equal(miniKeys(true, true, G), "f live · m full · [ ] more · g engram inspection");
	assert.equal(miniKeys(true, false, G, 33), "m full · [ ] more · g engram");
	assert.equal(miniKeys(true, true, G, 33), "f live · m full · g engram");
	assert.equal(miniKeys(true, true, LABELS.engramClose, 33), "f live · m full · g close");
	assert.equal(miniKeys(true, false), "m full · [ ] more · ctrl+\\ closes");
	assert.ok([G, LABELS.engramClose].every((g) => [true, false].every((h) => visibleWidth(miniKeys(true, h, g)) <= MINI_WIDTH - 3)));
	// Before stepping in: short, the first to go where tight; none without the cap.
	const OUT = "ctrl+\\ steps in: tab a panel, ? explain, 1-5 open, g engram, 6 steer · /scope help dock";
	assert.equal(boardKeys("MAP", false, false, undefined, false, false, false, true, "", KEYS_W, G), OUT);
	assert.equal(boardKeys("MAP", false, false, undefined, false, false, false, true, "", KEYS_W, LABELS.engramClose), OUT.replace("g engram", "g close"));
	assert.equal(boardKeys("MAP", false), "ctrl+\\ steps in: tab a panel, ? explain, 1-5 open, 6 steer · /scope help dock");
	assert.equal(miniKeys(false, false, G), "ctrl+\\ steps in, then m · [ ] · g");
	assert.equal(miniKeys(false, false, G, 32), "ctrl+\\ steps in, then m · [ ]", "tight: g goes");
	assert.equal(miniKeys(false, false), "ctrl+\\ steps in, then m · [ ]");
	// In the dock: the board's every panel, the one-panel board (its two key lines), the mini dock; g close once on.
	const { d, host } = dock(reply());
	assert.equal(text(d.render(104)).at(-1), OUT.replace("1-5", "1-4"), "the board, not stepped in (no writes: no 5)");
	host.mini = true;
	assert.equal(text(d.render(MINI_WIDTH)).at(-1), "ctrl+\\ steps in, then m · [ ] · g", "the mini dock, not stepped in");
	host.mini = false;
	d.stepIn(5);
	for (const p of ["MAP", "LENS", "TOKENS", "EXPERTS"]) {
		host.panel = p;
		assert.ok(text(d.render(104)).at(-1)!.includes(G), p);
	}
	host.panel = "MAP";
	const one = () => text(d.render(60)).slice(-2).join(" ");
	assert.match(one(), /u\/U unsure (· )?g engram inspection (· )?1-4 6 panel/, "the one-panel board: its numbers");
	d.handleInput("g");
	await tick();
	assert.ok(text(d.render(104)).at(-1)!.includes("· g close ·"));
	assert.match(one(), /u\/U unsure (· )?g close (· )?1-4 6 panel/);
	host.mini = true;
	assert.match(text(d.render(MINI_WIDTH)).at(-1)!, /g close/);
});

test("g: only where the server lists the engram; a reply streaming or not saved: a note, nothing asked", async () => {
	const none = dock(reply(), []);
	none.d.stepIn(-1);
	assert.ok(!text(none.d.render(104)).join("\n").includes(LABELS.engramKey));
	none.d.handleInput("g");
	assert.match(text(none.d.render(104)).at(-1)!, new RegExp(LABELS.lookupNone));
	assert.equal(LABELS.lookupNone, "no engram on this model");
	assert.deepEqual(none.asked, []);
	const no = dock(reply(), ["lookup"], undefined, { lookup: undefined });                     // a host without the route
	no.d.stepIn(-1);
	no.d.handleInput("g");
	assert.match(text(no.d.render(104)).at(-1)!, new RegExp(LABELS.lookupNone));
	const streaming = dock(reply(5, false));
	streaming.d.stepIn(-1);
	streaming.d.handleInput("g");
	assert.match(text(streaming.d.render(60)).join("\n"), new RegExp(LABELS.lookupWait));
	const unsaved = reply();
	unsaved.reply!.traceId = undefined;
	const u = dock(unsaved);
	u.d.stepIn(-1);
	u.d.handleInput("g");
	assert.match(text(u.d.render(104)).at(-1)!, new RegExp(LABELS.lookupUnsaved));
	assert.deepEqual([streaming.asked, u.asked], [[], []]);
});

test("g on the board, no clock: measuring, then the last frame at once (the title, the legend in the rule, MAP and the text, TOKENS without and with); g closes it, again opens it with nothing asked", async () => {
	const live = reply();
	let answerIt: (a: LookupAnswer) => void = () => {};
	const { d, host, asked, ids, stops } = dock(live, ["lookup"], () => new Promise((r) => (answerIt = r)));
	d.stepIn(5);
	const before = text(d.render(104));
	assert.match(before.join("\n"), /MAP +11 tok · \d+ unsure/);
	assert.match(before[1], /^─+$/);
	d.handleInput("g");
	assert.deepEqual(asked, ["T1"]);
	const busy = text(d.render(104));
	assert.equal(titleLine(busy), `${TITLE}  ${LABELS.lookupBusy}`, busy.join("\n"));
	assert.match(busy[1], /^─+$/, "no legend before the answer (no sweep without a clock)");
	d.handleInput("g");                                                                         // stopped while it reads: closed at once
	assert.deepEqual(stops, [["T1", ids[0]]]);
	assert.equal(host.lookupOn, undefined);
	assert.ok(text(d.render(104)).includes("REPLY"));
	const late = answerIt;
	d.handleInput("g");                                                                         // open again: asked again
	assert.deepEqual(asked, ["T1"], "once the server took the stop");
	await tick();
	assert.deepEqual(asked, ["T1", "T1"]);
	assert.notEqual(ids[1], ids[0]);
	late(answer(11, { 5: 0 }));                                                                  // the stopped ask's answer: dropped
	await tick();
	assert.equal(titleLine(text(d.render(104))), `${TITLE}  ${LABELS.lookupBusy}`);
	answerIt(answer(11, { 5: 1.95 }));
	await tick();
	const raw = d.render(104), on = text(raw);
	assert.equal(titleLine(on), `${TITLE}  engram " across the Mediterranean" → " Sea"   without 5.2% → with 37%`, on.join("\n"));
	assert.match(on[1], new RegExp(`^── ${LABELS.engramLegend[0]} ─+$`));
	assert.match(on.join("\n"), /MAP +11 tok · 1 helped/);
	assert.deepEqual(mapCellsIn(raw).slice(2, 7).map((c) => (c === WHITE ? "pred" : c === BLOCK ? "block" : "")), ["block", "block", "block", "pred", ""],
		"MAP: the engram's cells, the predicted token marked");
	assert.ok(on.some((l) => /^ ?TOKENS {2,3}#5 +without +with +│/.test(l)), "TOKENS: without and with");
	assert.ok(!raw.join("").includes(colourCode(COLOURS.accent, "truecolor") + " across"), "no orange in the text");
	assert.ok(raw.some((l) => l.includes(colourCode(MAP_BLOCK, "truecolor", true) + colourCode(COLOURS.text, "truecolor"))), "the engram's block in the text");
	host.captions = true;
	for (const p of ["MAP", "TOKENS"]) {
		host.panel = p;
		assert.ok(text(d.render(104)).join(" ").replace(/\s+/g, " ").includes(EXPLAIN.ENGRAM.slice(0, 40)), p);
	}
	host.captions = false;
	d.handleInput("\x1b[D");                                                                    // ← a token: the readout follows the cursor
	assert.equal(titleLine(text(d.render(104))), `${TITLE}  engram " moved across the" → " Mediterranean"   no change`);
	d.handleInput("g");
	const off = text(d.render(104));
	assert.ok(off.includes("REPLY") && /MAP +11 tok · \d+ unsure/.test(off.join("\n")), off.join("\n"));   // closed: as before
	assert.match(off[1], /^─+$/);
	d.handleInput("g");
	assert.match(text(d.render(104)).join("\n"), /MAP +11 tok · 1 helped/);
	assert.deepEqual(asked, ["T1", "T1"]);
	assert.equal(stops.length, 1, "closed with the answer in: nothing to stop");
	// Remembered for the session (the host's): a new dock shows it at once.
	const again = new Dock({ terminal: { rows: 67 }, requestRender() {} }, piTheme, host);
	assert.match(text(again.render(104)).join("\n"), /MAP +11 tok · 1 helped/);
	// A new reply: drawn by confidence again (the engram was the other reply's).
	onEvent(live, { type: "start", rid: "r2", model: "Flash" });
	onEvent(live, { type: "token", rid: "r2", i: 0, id: 5, text: " x", p: 0.9, entropy: 0, alts: [], n_layers: null });
	assert.match(text(d.render(104)).join("\n"), /MAP +1 tok · 0 unsure/);
});

test("g with a clock: the sweep, without the engram then with it, the blue behind the second; over in two passes with the answer in hand, alternating while it is coming; any key ends it", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval"] });
	const clock = { t: 0 }, live = reply(30);
	let answerIt: (a: LookupAnswer) => void = () => {};
	const { d } = dock(live, ["lookup"], () => new Promise((r) => (answerIt = r)), { now: () => clock.t });
	d.stepIn(20);
	d.render(104);
	d.handleInput("g");
	const frame = (ms: number) => ((clock.t = ms), d.render(104));
	const legend = (ls: string[]) => text(ls)[1];
	// While it is coming the band crosses the text with no legend: the title line says how far the readings are.
	const coming = frame(100);
	assert.match(legend(coming), /^─+$/);
	assert.ok(mapCellsIn(coming).includes(WHITE), "the band in MAP");
	assert.match(legend(frame(SWEEP_MS + 100)), /^─+$/);
	answerIt(answer(30, { 5: 1.95, 21: 1 }));
	await tick();
	frame(2 * SWEEP_MS + 100);                                                                  // drawn as it comes
	// The answer came in the third pass (without): the fourth (with) is the last.
	const mid = frame(3 * SWEEP_MS + SWEEP_MS / 2);
	assert.match(legend(mid), new RegExp(LABELS.engramSweep[1]));
	assert.ok(mapCellsIn(mid).includes(WHITE), "the band bright in MAP");
	assert.ok(!mapCellsIn(mid).includes(BLOCK), "no marks while it sweeps");
	const over = frame(4 * SWEEP_MS + 1), done = text(over);
	assert.match(done[1], new RegExp(`^── ${LABELS.engramLegend[0]}`));
	assert.ok(mapCellsIn(over).includes(BLOCK), "the marks once it is over");
	// Opened again with the answer in hand (g closing, g again ends it closed, g opens): two passes, about 1.3 s.
	clock.t = 10_000;
	d.handleInput("g");
	d.handleInput("g");
	d.handleInput("g");
	assert.match(legend(frame(10_000 + 200)), new RegExp(LABELS.engramSweep[0]));
	assert.match(legend(frame(10_000 + SWEEP_MS + 200)), new RegExp(LABELS.engramSweep[1]));
	assert.match(legend(frame(10_000 + 2 * SWEEP_MS)), new RegExp(LABELS.engramLegend[0]));
	// Any key ends it at once into the last frame.
	clock.t = 20_000;
	d.handleInput("g");
	d.handleInput("g");
	d.handleInput("g");
	assert.match(legend(frame(20_000 + 100)), new RegExp(LABELS.engramSweep[0]));
	d.handleInput("\x1b[D");
	assert.match(legend(frame(20_000 + 150)), new RegExp(LABELS.engramLegend[0]));
	d.dispose();
});

test("g refused (409, 503): the server's detail on the title line, colours as they were; g again asks again; a throw never leaves the key", async () => {
	const live = reply();
	let n = 0;
	const { d, asked } = dock(live, ["lookup"], async () => {
		if (n++ === 0) throw new Error("a reply is being served: ask again when it is done");
		return answer(11, { 5: 1.95 });
	});
	d.stepIn(5);
	d.handleInput("g");
	await tick();
	const refused = text(d.render(104));
	assert.equal(titleLine(refused), `${TITLE}  a reply is being served: ask again when it is done`, refused.join("\n"));
	assert.match(refused.join("\n"), /MAP +11 tok · \d+ unsure/);
	assert.ok(refused.some((l) => l.includes(LABELS.engramKey)) && !refused.some((l) => l.includes(LABELS.engramClose)), "g asks again: its label");
	d.handleInput("g");
	await tick();
	assert.match(text(d.render(104)).join("\n"), /MAP +11 tok · 1 helped/);
	assert.deepEqual(asked, ["T1", "T1"]);
	const sync = dock(reply(), ["lookup"], () => { throw new Error("boom"); });
	sync.d.stepIn(5);
	assert.doesNotThrow(() => sync.d.handleInput("g"));
	await tick();
	assert.equal(titleLine(text(sync.d.render(104))), `${TITLE}  boom`);
	const bad = dock(reply(), ["lookup"], async () => ({ trace: "T1" }) as any);
	bad.d.stepIn(5);
	bad.d.handleInput("g");
	await tick();
	assert.equal(titleLine(text(bad.d.render(104))), `${TITLE}  ${LABELS.lookupBad}`);
});

test("g on the one-panel board and the mini dock: the title, its readout and the legend over it (shorter where narrow); the mini dock's top line and its MAP", async () => {
	const live = reply();
	const { d, host } = dock(live, ["lookup"], undefined, { panel: "LENS" });
	d.stepIn(5);
	d.handleInput("g");
	await tick();
	const page = text(d.render(60)), at = page.findIndex((l) => l.startsWith(TITLE));
	assert.ok(page[at].startsWith(`${TITLE}  " across the Medi… → " Sea" 5.2% → 37%`), page.join("\n"));
	assert.equal(page[at - 1], `── ${LABELS.engramLegend[1]} ${"─".repeat(57 - 4 - LABELS.engramLegend[1].length)}`);
	assert.match(text(d.render(44)).find((l) => l.startsWith("── "))!, new RegExp(`^── ${LABELS.engramLegend[2]}`));
	host.mini = true;
	const mini = text(d.render(MINI_WIDTH));
	assert.match(mini[0], /^" across the Mediterranean" → " Sea" 5.2% → 37%/, "whole at 50 columns");
	assert.ok(mini.some((l) => /^MAP +11 tok · 1 helped/.test(l)), mini.join("\n"));
	d.handleInput("g");
	assert.match(text(d.render(MINI_WIDTH))[0], /^#5 " Sea"/);
});

test("widget: caps from the selected model's server (refreshFlags keeps them); g's POST /api/lookup with the trace, a refusal's detail as it is", async () => {
	const posts: any[] = [];
	let answers: (() => any)[] = [];
	const api = async (path: string, init?: any, base?: string) => {
		if (path === "/api/lookup") {
			posts.push([init.method, JSON.parse(init.body), !!init.signal, base]);
			const next = answers.shift();
			if (next) return next();
			throw Object.assign(new Error("an engram inspection is under way: ask again when it is done"), { status: 409 });
		}
		if (path === "/api/status") return { engine: "omlx", caps: ["lookup"], chat: { "Flash-ds4": { tested: true } } };
		return { model: "Flash-ds4", concepts: [] };
	};
	const servers = new Map([["metalscope-ds4", { base: "http://127.0.0.1:8009", key: "", status: { caps: ["edit_at", "lookup"], chat: {} } }]]);
	let dk: any;
	const ctx = { hasUI: true, model: { provider: "metalscope-ds4", id: "Flash-ds4" },
		ui: { setWidget() {}, setStatus() {}, notify() {}, custom: (make: any) => new Promise<void>(() => { dk = make({ requestRender() {}, terminal: { rows: 44, columns: 160 } }, piTheme, {}, () => {}); }) } };
	const live: any = { base: "http://127.0.0.1:8009", clientId: "pi-x", level: "on", connected: true };
	const ui = createUi({ pi: { sendUserMessage() {} }, live, api, servers } as any, join(mkdtempSync(join(tmpdir(), "ms-look-")), "pi.json"));
	ui.showDock(ctx, true);
	const host = dk.host;
	assert.deepEqual(host.caps(), ["edit_at", "lookup"]);
	assert.ok(host.looked instanceof Map);
	assert.equal(typeof host.now, "function", "the widget's clock: the sweep runs");
	const B = "http://127.0.0.1:8009";
	await assert.rejects(host.lookup("T9", "a1"), /^Error: an engram inspection is under way: ask again when it is done$/);
	assert.deepEqual(posts, [["POST", { trace: "T9", client: "pi-x", ask: "a1" }, true, B]]);
	await host.stopLookup("T9", "a1");                                                         // answered: nothing to stop
	assert.equal(posts.length, 1);
	// Past fetch's idle timeout (undici's code on the error's cause) the same ask goes again, marked `again`; the stop
	// goes to the server asked, even after /model picked another; any other failure is thrown at once.
	const over = (code: string) => () => { throw Object.assign(new TypeError("fetch failed"), { cause: Object.assign(new Error("timeout"), { code }) }); };
	let release = (_: any) => {};
	answers = [over("UND_ERR_HEADERS_TIMEOUT"), over("UND_ERR_BODY_TIMEOUT"), () => new Promise((r) => (release = r))];
	const asked = host.lookup("T9", "a2");
	await tick();
	const body = { trace: "T9", client: "pi-x", ask: "a2" };
	assert.deepEqual(posts.slice(1).map((p) => [p[1], p[3]]), [[body, B], [{ ...body, again: true }, B], [{ ...body, again: true }, B]]);
	live.base = "http://127.0.0.1:8008";
	await host.stopLookup("T9", "a2");
	assert.deepEqual(posts[4], ["POST", { trace: "T9", ask: "a2", stop: true }, false, B]);
	release({ trace: "T9" });
	assert.deepEqual(await asked, { trace: "T9" });
	answers = [over("ECONNREFUSED")];
	await assert.rejects(host.lookup("T9", "a3"), /not answering/);
	assert.equal(posts.length, 6, "a connection error: asked once");
	// Stopped, it is not sent again.
	let fail = (_: any) => {};
	answers = [() => new Promise((_, no) => (fail = no))];
	const stopped = host.lookup("T9", "a4");
	await tick();
	await host.stopLookup("T9", "a4");
	fail(Object.assign(new TypeError("fetch failed"), { cause: { code: "UND_ERR_HEADERS_TIMEOUT" } }));
	await assert.rejects(stopped);
	assert.deepEqual(posts.slice(6).map((p) => p[1]), [{ trace: "T9", client: "pi-x", ask: "a4" }, { trace: "T9", ask: "a4", stop: true }]);
	live.base = B;
	// A reply's start refreshes the caps with the flags (oMLX lists the cap once Flash Next is loaded).
	const s = { base: "http://o", key: "", status: { caps: [] as string[], chat: { Q: { tested: true } } } };
	await refreshFlags(new Map([["metalscope-omlx", s]]), "http://o", "Q", async () => ({ caps: ["lookup"], chat: { Q: { tested: true, edit: true } } }));
	assert.deepEqual(s.status.caps, ["lookup"]);
	// Stale after a model switch (flags known, so refreshFlags asks nothing): g's one fresh read keeps the server's caps.
	servers.get("metalscope-ds4")!.status.caps = ["edit_at"];
	assert.deepEqual(host.caps(), ["edit_at"]);
	assert.deepEqual(await host.freshCaps(), ["lookup"]);
	assert.deepEqual(host.caps(), ["lookup"]);
	// The session gone (/new, /resume): its answers and the trace shown with it; readings under way stopped, not
	// asked again past fetch's limit.
	host.looked.set("T9", { answer: { trace: "T9" } });
	host.lookupOn = "T9";
	let late = (_: any) => {};
	answers = [() => new Promise((_, no) => (late = no))];
	const running = host.lookup("T8", "a5");
	host.looked.set("T8", { ask: "a5" });
	await tick();
	const n = posts.length;
	ui.detach();
	assert.deepEqual([host.looked.size, host.lookupOn], [0, undefined]);
	assert.deepEqual(posts.slice(n).map((p) => p[1]), [{ trace: "T8", ask: "a5", stop: true }]);
	late(Object.assign(new TypeError("fetch failed"), { cause: { code: "UND_ERR_HEADERS_TIMEOUT" } }));
	await assert.rejects(running);
	assert.equal(posts.length, n + 1, "not asked again");
});

test("g keeps the answer of the trace last opened alone (the server keeps each beside its trace): g on another reply drops the first's", async () => {
	const live = reply(30);
	const { d, host, asked } = dock(live, ["lookup"], undefined, { looked: new Map() });
	d.stepIn(20);
	d.handleInput("g");
	await tick();
	assert.ok(host.looked.get("T1")?.answer);
	onEvent(live, { type: "start", rid: "r2", model: "Flash" });
	onEvent(live, { type: "token", rid: "r2", i: 0, id: 5, text: " x", p: 0.9, entropy: 0, alts: [], n_layers: null });
	onEvent(live, { type: "token", rid: "r2", i: 1, id: 6, text: " y", p: 0.9, entropy: 0, alts: [], n_layers: null });
	onEvent(live, { type: "end", rid: "r2", n_tokens: 2, trace_id: "T2" });
	d.handleInput("g");
	await tick();
	assert.deepEqual([...host.looked.keys()], ["T2"]);
	assert.deepEqual(asked, ["T1", "T2"]);
	d.dispose();
});

test("g's label through the real host: servers from `metalscope status` (flags, no caps), the dock's key line asks the server once and shows g", async () => {
	const servers = new Map<string, Server>();
	const cat: Catalogue = { omlx: { models: [] }, ds4: { models: [{ name: "Flash-ds4", ctx: 1000, loaded: true, reasoning: true, thinking_format: null, tested: true, edit: true }] } };
	registerCatalogue({ registerProvider() {} } as any, cat, { base: "http://o", key: "" }, servers, async () => undefined);
	let statuses = 0;
	const live = reply();
	live.base = DS4_URL;
	// traceApi asks live.base: the server followed, ds4's.
	const api = async (path: string) => {
		if (path === "/api/status") return assert.equal(live.base, DS4_URL), statuses++, { engine: "ds4", caps: ["edit_at", "lookup"], chat: { "Flash-ds4": { tested: true, edit: true } } };
		return { model: "Flash-ds4", concepts: [] };
	};
	let dk: any;
	const ctx = { hasUI: true, model: { provider: DS4, id: "Flash-ds4" },
		ui: { setWidget() {}, setStatus() {}, notify() {}, custom: (make: any) => new Promise<void>(() => { dk = make({ requestRender() {}, terminal: { rows: 67, columns: 200 } }, piTheme, {}, () => {}); }) } };
	const ui = createUi({ pi: { sendUserMessage() {} }, live, api, servers } as any, join(mkdtempSync(join(tmpdir(), "ms-look-")), "pi.json"));
	ui.showDock(ctx, true);
	dk.stepIn(5);
	dk.render(104);                                                                             // the caps unknown: asked
	for (let k = 0; k < 3; k++) await tick();
	dk.invalidate();
	const keyLine = text(dk.render(104)).at(-1)!;
	assert.ok(keyLine.includes(LABELS.engramKey), keyLine);
	dk.invalidate();
	dk.render(104);
	assert.equal(statuses, 1, "asked once");
	// A reply's start asks too, the flags known but not the caps (refreshFlags).
	const fresh = new Map<string, Server>();
	registerCatalogue({ registerProvider() {} } as any, cat, { base: "http://o", key: "" }, fresh, async () => undefined);
	await refreshFlags(fresh, DS4_URL, "Flash-ds4", async () => ({ caps: ["lookup"], chat: {} }));
	assert.deepEqual(fresh.get(DS4)!.status.caps, ["lookup"]);
});

test("caps kept before a model loads: a reply of it starting asks once (oMLX lists lookup with Flash Next); an answer after /model, or from another server, kept nowhere; a replaced entry asked again", async () => {
	const O = "http://o", servers = new Map<string, Server>();
	const cat: Catalogue = { omlx: { models: [{ name: "Flash", ctx: 1000, loaded: false, reasoning: true, thinking_format: null, tested: true, edit: true }] }, ds4: { models: [] } };
	registerCatalogue({ registerProvider() {} } as any, cat, { base: O, key: "" }, servers, async () => undefined);
	servers.get(OMLX)!.status.caps = ["edit_at"];                                                // read before Flash Next loaded
	const live = reply();
	live.base = O;
	let answers: (() => void)[] = [], now: any = { caps: ["edit_at", "lookup"], chat: { Flash: { tested: true, edit: true } } };
	const asked: string[] = [];
	const api = (path: string) => new Promise((ok) => {
		if (path !== "/api/status") return ok({ model: "Flash-oq4", concepts: [] });
		asked.push(live.base);
		const st = now;
		answers.push(() => ok(st));
	});
	let dk: any;
	const ctx: any = { hasUI: true, model: { provider: OMLX, id: "Flash" },
		ui: { setWidget() {}, setStatus() {}, notify() {}, custom: (make: any) => new Promise<void>(() => { dk = make({ requestRender() {}, terminal: { rows: 67, columns: 200 } }, piTheme, {}, () => {}); }) } };
	createUi({ pi: { sendUserMessage() {} }, live, api, servers } as any, join(mkdtempSync(join(tmpdir(), "ms-look-")), "pi.json")).showDock(ctx, true);
	dk.stepIn(5);
	const keys = () => (dk.invalidate(), text(dk.render(104)).at(-1)!);
	assert.ok(!keys().includes(LABELS.engramKey), "no lookup cached: no g");
	assert.deepEqual(asked, [], "the caps known: nothing asked on a draw");
	// A reply of Flash starts (index.ts live.onStart): asked once, whatever is kept; the label follows.
	let reads = 0;
	const status = async (base: string) => (assert.equal(base, O), reads++, now);
	await refreshFlags(servers, O, "Flash", status);
	assert.deepEqual(servers.get(OMLX)!.status.caps, ["edit_at", "lookup"]);
	assert.ok(keys().includes(LABELS.engramKey), keys());
	await refreshFlags(servers, O, "Flash", status);
	assert.equal(reads, 1, "the same model again on the same entry: not asked");
	// Replaced (a backend switch or /model's catalogue): asked again on its next start.
	registerCatalogue({ registerProvider() {} } as any, cat, { base: O, key: "" }, servers, async () => undefined);
	servers.get(OMLX)!.status.caps = [];
	await refreshFlags(servers, O, "Flash", status);
	assert.equal(reads, 2);
	assert.deepEqual(servers.get(OMLX)!.status.caps, ["edit_at", "lookup"]);
	// g's fresh read answered after /model went to ds4: the oMLX entry is not ds4's, ds4's untouched.
	const ds4 = { base: DS4_URL, key: "", status: { caps: ["edit_at"], chat: {} } };
	servers.set(DS4, ds4);
	servers.get(OMLX)!.status.caps = ["edit_at"];
	const fresh = dk.host.freshCaps();
	ctx.model = { provider: DS4, id: "Flash-ds4" };
	now = { caps: ["nothing"] };
	answers.shift()!();
	assert.deepEqual(await fresh, ["edit_at", "lookup"], "freshCaps says what the server said");
	assert.deepEqual(servers.get(OMLX)!.status.caps, ["edit_at", "lookup"], "kept in the entry asked for");
	assert.deepEqual(ds4.status.caps, ["edit_at"], "not in the model's now");
	// The entry replaced while asking: kept nowhere.
	ctx.model = { provider: OMLX, id: "Flash" };
	const again = dk.host.freshCaps();
	registerCatalogue({ registerProvider() {} } as any, cat, { base: O, key: "" }, servers, async () => undefined);
	const replaced = servers.get(OMLX)!;
	replaced.status.caps = ["edit_at"];
	answers.shift()!();
	await again;
	assert.deepEqual(replaced.status.caps, ["edit_at"], "a replaced entry: not stored");
	// The server followed is another (live.base): kept nowhere.
	live.base = DS4_URL;
	const other = dk.host.freshCaps();
	answers.shift()!();
	await other;
	assert.deepEqual(replaced.status.caps, ["edit_at"], "another server's answer: not stored");
	assert.deepEqual(asked, [O, O, DS4_URL]);
});

test("g, review fixes: following shows the count; stale caps asked once more; closed with the cap gone; ids that differ; an answer after the dock closed", async () => {
	const f = dock(reply());
	f.d.stepIn(-1);
	f.d.handleInput("g");
	await tick();
	assert.equal(titleLine(text(f.d.render(104))), `${TITLE}  1 tokens helped`);
	f.host.mini = true;
	assert.equal(text(f.d.render(MINI_WIDTH))[0], "1 tokens helped");
	let fresh = ["lookup"], reads = 0;
	const s = dock(reply(), [], undefined, { freshCaps: async () => (reads++, fresh) });
	s.d.stepIn(5);
	s.d.handleInput("g");
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual([reads, s.asked], [1, ["T1"]]);
	assert.match(text(s.d.render(104)).join("\n"), /MAP +11 tok · 1 helped/);
	const no = dock(reply(), [], undefined, { freshCaps: async () => (reads++, (fresh = [])) });
	no.d.stepIn(5);
	no.d.handleInput("g");
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual([reads, no.asked], [2, []]);
	assert.match(text(no.d.render(104)).at(-1)!, new RegExp(LABELS.lookupNone));
	const caps = ["lookup"], h = dock(reply(), caps);
	h.d.stepIn(5);
	h.d.handleInput("g");
	await tick();
	caps.length = 0;
	assert.ok(text(h.d.render(104)).at(-1)!.includes(LABELS.engramClose), "on: g close whatever the caps");
	h.d.handleInput("g");
	assert.equal(h.host.lookupOn, undefined);
	assert.match(text(h.d.render(104)).join("\n"), /MAP +11 tok · \d+ unsure/);
	assert.deepEqual(h.asked, ["T1"]);
	const wrong = answer(11, { 5: 2 });
	wrong.ids = wrong.ids.map((x) => x + 1000);
	const x = dock(reply(), ["lookup"], async () => wrong);
	x.d.stepIn(5);
	x.d.handleInput("g");
	await tick();
	assert.equal(titleLine(text(x.d.render(104))), `${TITLE}  ${LABELS.lookupDiffer}`);
	assert.match(text(x.d.render(104)).join("\n"), /MAP +11 tok · \d+ unsure/);
	const other = dock(reply(), ["lookup"], async () => ({ ...answer(11, { 5: 2 }), trace: "T2" }));
	other.d.stepIn(5);
	other.d.handleInput("g");
	await tick();
	assert.equal(titleLine(text(other.d.render(104))), `${TITLE}  ${LABELS.lookupBad}`);
	const live = reply();
	let answerIt: (a: LookupAnswer) => void = () => {}, drawn = 0;
	live.requestRender = () => void drawn++;
	const first = dock(live, ["lookup"], () => new Promise((r) => (answerIt = r)));
	first.d.stepIn(5);
	first.d.handleInput("g");
	first.d.dispose();
	const next = new Dock({ terminal: { rows: 67 }, requestRender() {} }, piTheme, first.host);
	answerIt(answer(11, { 5: 1.95 }));
	await tick();
	assert.equal(drawn, 1);
	assert.match(text(next.render(104)).join("\n"), /MAP +11 tok · 1 helped/);
	const empty = dock({ base: "", clientId: "me", level: "on", connected: true });
	empty.d.handleInput("g");
	assert.ok(!text(empty.d.render(104)).join("\n").includes(LABELS.lookupUnsaved));
});

test("engram marks by the stream's index: a gap marks nothing across it; the edited token takes the paint", () => {
	const live = reply(), r = live.reply!;
	const gapped: Reply = { ...r, tokens: [...r.tokens.slice(0, 3), ...r.tokens.slice(6)] };   // i 0 1 2, then 6..10
	const v = lookupView(gapped, answer(11, { 6: 1, 2: 1 }));
	assert.deepEqual([engramOf(v, 3), engramOf(v, 2)], [[3, 3], [0, 2]]);                       // none before i 6 (i 3-5 missing)
	assert.equal(lookupLine(v, gapped.tokens, 3, 80), 'engram ? → ","   without 14% → with 37%');
	const th = { fg: (c: string, s: string) => `<${c}>${s}</>`, bg: (_c: string, s: string) => s, bold: (s: string) => s };
	const line = wrapTokens(th, r.tokens, 200, -1, -1, { k: 2, changed: false }, lookupPaint(th, lookupView(r, answer(11)), -1)).lines.join("");
	assert.ok(line.includes("<faint> </><faint>▸</><faint>across</>"), line);
});

test("lookupRgb during a sweep: the band bright, without the engram all quiet, with it the blue behind the head only", () => {
	const r = reply(30).reply!, v = lookupView(r, answer(30, { 2: 2, 20: 2 }));
	const quiet = lookupRgb(v, 0);
	assert.deepEqual([lookupRgb(v, 2, { pass: 0, at: 10 }), lookupRgb(v, 9, { pass: 0, at: 10 }), lookupRgb(v, 20, { pass: 0, at: 10 })], [quiet, COLOURS.text, quiet]);
	assert.deepEqual([lookupRgb(v, 2, { pass: 1, at: 10 }), lookupRgb(v, 20, { pass: 1, at: 10 })], [BLUE(3), quiet]);
	assert.deepEqual(lookupRgb(undefined, 2, { pass: 1, at: 10 }), quiet, "no answer yet: quiet");
});

test("sweepAt: two passes with the answer in hand; alternating while it is coming, held after SWEEP_PASSES; the answer in the hold: one with-pass from then", () => {
	const M = SWEEP_MS, held = SWEEP_PASSES * M;
	assert.deepEqual([sweepAt(100, 0), sweepAt(M + 100, 0), sweepAt(2 * M, 0)], [{ pass: 0, frac: 100 / M }, { pass: 1, frac: 100 / M }, "done"]);
	assert.deepEqual([sweepAt(2 * M + 100), sweepAt(3 * M + 100)], [{ pass: 0, frac: 100 / M }, { pass: 1, frac: 100 / M }]);
	assert.equal(sweepAt(held + 1), "hold");
	assert.deepEqual(sweepAt(held + 500, held + 400), { pass: 1, frac: 100 / M });
	assert.equal(sweepAt(held + 400 + M, held + 400), "done");
	// The answer in a pass without (the third): the fourth, with, is the last.
	assert.deepEqual(sweepAt(3 * M + 1, 2 * M + 100), { pass: 1, frac: 1 / M });
	assert.equal(sweepAt(4 * M, 2 * M + 100), "done");
});

test("g, a token not read (the last): its cursor drawn as ever (inverse in the text, white in MAP), no engram marks", async () => {
	const { d } = dock(reply());
	d.stepIn(10);
	d.handleInput("g");
	await tick();
	const raw = d.render(104);
	assert.match(titleLine(text(raw))!, /last token: not read$/);
	assert.ok(raw.some((l) => l.includes("\x1b[7m")), "the text's cursor");
	assert.equal(mapCellsIn(raw).indexOf(WHITE), 10, "MAP's white cell");
	assert.ok(!mapCellsIn(raw).includes(BLOCK));
	d.stepIn(5);
	const read = mapCellsIn(d.render(104));
	assert.ok(read.includes(BLOCK) && read.indexOf(WHITE) === 5 && read.lastIndexOf(WHITE) === 5, "read: marked instead, the predicted token white");
});

test("g with a clock, the answer slow: held quiet after SWEEP_PASSES (the title line reading), then one pass with the engram from when it comes", async () => {
	const clock = { t: 0 };
	let answerIt: (a: LookupAnswer) => void = () => {};
	const { d } = dock(reply(30), ["lookup"], () => new Promise((r) => (answerIt = r)), { now: () => clock.t });
	d.stepIn(20);
	d.handleInput("g");
	const frame = (ms: number) => ((clock.t = ms), text(d.render(104)));
	const held = frame(SWEEP_PASSES * SWEEP_MS + 100);
	assert.match(held[1], /^─+$/);
	assert.equal(titleLine(held), `${TITLE}  ${LABELS.lookupBusy}`);
	assert.deepEqual(frame(SWEEP_PASSES * SWEEP_MS + 900), held, "nothing moves");
	answerIt(answer(30, { 5: 1.95 }));
	await tick();
	const T = SWEEP_PASSES * SWEEP_MS + 1000;
	assert.match(frame(T)[1], new RegExp(LABELS.engramSweep[1]));
	assert.match(frame(T + SWEEP_MS)[1], new RegExp(`^── ${LABELS.engramLegend[0]}`));
	d.dispose();
});

test("g's readings across docks: one dock asks and goes; another shows them reading (g stop), the answer from the first's callback; there g stops them, g again asks anew", async () => {
	const clock = { t: 0 };
	let answerIt: (a: LookupAnswer) => void = () => {};
	const a = dock(reply(), ["lookup"], () => new Promise((r) => (answerIt = r)), { now: () => clock.t });
	a.d.stepIn(5);
	a.d.handleInput("g");
	a.d.dispose();
	const b = new Dock({ terminal: { rows: 67 }, requestRender() {} }, piTheme, a.host);
	b.stepIn(5);
	const reading = text(b.render(104));
	assert.equal(titleLine(reading), `${TITLE}  ${LABELS.lookupBusy}`);
	assert.ok(reading.some((l) => l.includes(LABELS.engramStop)), reading.slice(-2).join("\n"));
	clock.t = 100;
	answerIt(answer(11, { 5: 1.95 }));
	await tick();
	assert.match(text(b.render(104)).join("\n"), /MAP +11 tok · 1 helped/);
	assert.deepEqual(a.asked, ["T1"]);
	// Asked again in a dock that did not ask: g stops it there too.
	b.handleInput("g");
	clock.t = 5000;
	b.render(104);
	b.handleInput("g");
	assert.deepEqual(a.asked, ["T1"], "the answer kept for the session: not asked");
	a.host.looked.clear();
	b.handleInput("g");                                                                         // closed: nothing there to stop
	b.handleInput("g");                                                                         // asked anew
	b.handleInput("g");                                                                         // stopped
	assert.deepEqual([a.asked, a.stops], [["T1", "T1"], [["T1", a.ids[1]]]]);
	b.dispose();
});

test("the sweep's ticker (mock timers): Pi asked to draw every SWEEP_TICK_MS; a tick that finds the dock not drawn ends it; over, the last frame at once; gone with the dock", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval"] });
	const clock = { t: 0 };
	let asked = 0;
	const host: any = { live: reply(), captions: false, close() {}, unfocus() {}, fetchJson: async () => null, caps: () => ["lookup"],
		lookup: async () => answer(11, { 5: 1.95 }), now: () => clock.t };
	const d = new Dock({ terminal: { rows: 67 }, requestRender: () => void asked++ }, piTheme, host);
	// Not drawn: the second tick ends it; drawn afresh, the last frame.
	d.stepIn(5);
	d.handleInput("g");
	await tick();
	d.render(104);
	asked = 0;
	t.mock.timers.tick(SWEEP_TICK_MS);
	assert.equal(asked, 1);
	t.mock.timers.tick(SWEEP_TICK_MS);
	t.mock.timers.tick(SWEEP_TICK_MS);
	assert.equal(asked, 1, "not drawn since: stopped");
	assert.match(text(d.render(104))[1], new RegExp(`^── ${LABELS.engramLegend[0]}`));
	// Drawn every tick: over after two passes, the last frame asked for at once, then nothing more.
	d.handleInput("g");
	d.handleInput("g");
	d.handleInput("g");
	asked = 0;
	for (let ms = 0; ms < 2 * SWEEP_MS + 3 * SWEEP_TICK_MS; ms += SWEEP_TICK_MS) {
		d.render(104);
		clock.t += SWEEP_TICK_MS;
		t.mock.timers.tick(SWEEP_TICK_MS);
	}
	const n = asked;
	assert.ok(n >= Math.floor((2 * SWEEP_MS) / SWEEP_TICK_MS) && n <= Math.ceil((2 * SWEEP_MS) / SWEEP_TICK_MS) + 1, String(n));
	assert.match(text(d.render(104))[1], new RegExp(`^── ${LABELS.engramLegend[0]}`));
	t.mock.timers.tick(10 * SWEEP_TICK_MS);
	assert.equal(asked, n, "over: no more ticks");
	// Gone with the dock.
	d.handleInput("g");
	d.handleInput("g");
	d.dispose();
	asked = 0;
	t.mock.timers.tick(10 * SWEEP_TICK_MS);
	assert.equal(asked, 0);
});

test("g on the full-size EXPERTS pages' key lines, short", () => {
	assert.equal(layerKeys("this session", 3, LABELS.engramShort), "←→ token  ↑↓ pick  tab other list  enter open #3  [ ] layer  a this session  g engram  esc back");
	assert.equal(expertKeys("all chats", LABELS.engramClose), "←→ token  ↑↓ pick  enter go  [ ] layer  a all chats  g close  b back  esc layer");
	assert.ok(!layerKeys("", 3).includes("g "));
});

test("g to close, with a clock: one pass right to left, the ordinary colours behind it; the title and TOKENS as they were at its end; any key, or another reply, ends it at once", async () => {
	const clock = { t: 0 }, live = reply(30);
	const { d, host } = dock(live, ["lookup"], async () => answer(30, { 5: 1.95, 21: 1 }), { now: () => clock.t });
	d.stepIn(20);
	d.handleInput("g");
	await tick();
	const frame = (ms: number) => ((clock.t = ms), d.render(104));
	frame(10_000);                                                                              // opened: its sweep over
	clock.t = 20_000;
	d.handleInput("g");
	assert.equal(host.lookupOn, "T1", "still on while it closes");
	const mid = frame(20_000 + SWEEP_MS / 2), m = text(mid);
	assert.match(m[1], new RegExp(`^── ${LABELS.engramClosing} ─+$`));
	assert.ok(titleLine(m)!.startsWith(TITLE), "the title until the end");
	assert.ok(m.some((l) => /^ ?TOKENS {2,3}#20 +without +with +│/.test(l)), "TOKENS until the end");
	assert.ok(!mapCellsIn(mid).includes(BLOCK), "the marks go at its start");
	// Behind the head (the tokens to its right) the confidence colours, ahead the engram's.
	const confident = mapCells([{ n: 1, p: () => 0.95, think: -1 }], 1, 0, 1)[0].join();
	assert.ok(mapCellsIn(mid).includes(confident) && mapCellsIn(mid).includes(BLUE(3).join()), "both, either side");
	const done = text(frame(20_000 + SWEEP_MS));
	assert.equal(host.lookupOn, undefined);
	assert.ok(done.includes("REPLY") && /^─+$/.test(done[1]), done.slice(0, 3).join("\n"));
	// Any key ends it at once, closed.
	d.handleInput("g");
	frame(30_000);
	clock.t = 40_000;
	d.handleInput("g");
	d.handleInput("\x1b[D");
	assert.equal(host.lookupOn, undefined);
	// g during the closing pass (a double tap): closed, not opened again; the next g opens it.
	d.handleInput("g");
	frame(45_000);
	clock.t = 46_000;
	d.handleInput("g");
	frame(46_000 + SWEEP_MS / 4);
	d.handleInput("g");
	assert.equal(host.lookupOn, undefined, "a double tap ends closed");
	assert.ok(text(frame(46_000 + SWEEP_MS / 2)).includes("REPLY"), "drawn as before");
	d.handleInput("g");
	assert.equal(host.lookupOn, "T1", "g again opens it");
	d.handleInput("g");
	frame(48_000);
	// Another reply ends it, closed.
	d.handleInput("g");
	frame(50_000);
	clock.t = 60_000;
	d.handleInput("g");
	onEvent(live, { type: "start", rid: "r2", model: "Flash" });
	onEvent(live, { type: "token", rid: "r2", i: 0, id: 5, text: " x", p: 0.9, entropy: 0, alts: [], n_layers: null });
	frame(60_100);
	assert.equal(host.lookupOn, undefined);
	d.dispose();
});

test("g to close, no clock: off at once; the closing pass's ticker (mock timers): over, closed and drawn; unseen or gone with the dock, closed", async (t) => {
	const plainDock = dock(reply());
	plainDock.d.stepIn(5);
	plainDock.d.handleInput("g");
	await tick();
	plainDock.d.handleInput("g");
	assert.equal(plainDock.host.lookupOn, undefined);
	t.mock.timers.enable({ apis: ["setInterval"] });
	const clock = { t: 0 };
	let asked = 0;
	const host: any = { live: reply(), captions: false, close() {}, unfocus() {}, fetchJson: async () => null, caps: () => ["lookup"],
		lookup: async () => answer(11, { 5: 1.95 }), now: () => clock.t };
	const d = new Dock({ terminal: { rows: 67 }, requestRender: () => void asked++ }, piTheme, host);
	d.stepIn(5);
	d.handleInput("g");
	await tick();
	clock.t = 5000;
	d.render(104);                                                                              // its opening sweep over
	const close = () => {
		d.handleInput("g");
		d.render(104);
		assert.equal(host.lookupOn, "T1");
	};
	// Drawn each tick: over after one pass, closed, the last frame asked for.
	close();
	for (let ms = 0; ms < SWEEP_MS + 2 * SWEEP_TICK_MS; ms += SWEEP_TICK_MS) {
		d.render(104);
		clock.t += SWEEP_TICK_MS;
		t.mock.timers.tick(SWEEP_TICK_MS);
	}
	assert.equal(host.lookupOn, undefined);
	// Unseen (not drawn since the last tick): closed.
	d.handleInput("g");
	await tick();
	clock.t += 5000;
	d.render(104);
	close();
	t.mock.timers.tick(SWEEP_TICK_MS);
	t.mock.timers.tick(SWEEP_TICK_MS);
	assert.equal(host.lookupOn, undefined);
	// Gone with the dock: closed.
	d.handleInput("g");
	clock.t += 5000;
	d.render(104);
	close();
	d.dispose();
	assert.equal(host.lookupOn, undefined);
	asked = 0;
	t.mock.timers.tick(10 * SWEEP_TICK_MS);
	assert.equal(asked, 0);
});

test("back: behind a closing sweep (right of its band) a token is drawn as ever; ahead, the engram's colours; the band bright", () => {
	const r = reply(30).reply!, v = lookupView(r, answer(30, { 2: 2, 20: 2 })), sw = { pass: 1 as const, at: 10, close: true };
	assert.deepEqual([back(sw, 9), back(sw, 13), back(sw, 14), back({ pass: 1, at: 10 }, 20)], [false, false, true, false]);
	const p = lookupMap(v, -1, sw);
	assert.deepEqual([p.shade(2), p.shade(11), p.shade(20)], [BLUE(3), COLOURS.text, undefined]);
	const th = { fg: (c: string, s: string) => `<${c}>${s}</>`, bg: (_c: string, s: string) => s, bold: (s: string) => s };
	assert.equal(lookupPaint(th, v, -1, 0, sw).at(20, " x"), null);
});

test("progressAt: the reading's time over its expected, at least what is known done, short of the next checkpoint and of full; never back as time goes; seconds left while the estimate holds", () => {
	// ds4: reading 1 of 2 expected to take 10 s, nothing known inside it.
	const p0 = progress("a", 0, 0, 1, 0, 10);
	assert.deepEqual(progressAt(p0, 0), { share: 0, left: 20 });
	assert.deepEqual(progressAt(p0, 5), { share: 0.25, left: 15 });
	assert.deepEqual(progressAt(p0, 30), { share: AHEAD / 2, left: undefined }, "past the estimate: held short of the reading's end");
	// Reading 2, expected as long as reading 1 took: from half way, never full however long it takes.
	const p1 = progress("a", 1, 0, 1, 0.5, 12);
	assert.equal(progressAt(p1, -1).share, 0.5 + 0.5 / 24, "a clock behind the event's: no earlier than the event");
	assert.equal(progressAt(p1, 1e6).share, (1 + AHEAD) / 2);
	assert.ok(progressAt(progress("a", 1, 1, 1, 9, 9), 5).share === MOST, "all of it known done, the answer still coming: never full");
	// MLX: chunk 2 of 4 done, the clock behind it: what is known; ahead of it, short of the next chunk's end.
	assert.equal(progressAt(progress("a", 0, 0.5, 0.75, 1, 100), 0).share, 0.25);
	assert.equal(progressAt(progress("a", 0, 0.5, 0.75, 1, 2), 100).share, (0.5 + AHEAD * 0.25) / 2);
	// Sampled through a run (events as a slow MLX run sends them): never back, never full.
	const events = [progress("a", 0, 0, 0.25, 0, 4), progress("a", 0, 0.25, 0.5, 3, 12), progress("a", 0, 0.5, 0.75, 6, 12), progress("a", 0, 0.75, 1, 9, 12),
		progress("a", 1, 0, 0.25, 0, 12), progress("a", 1, 0.25, 0.5, 2, 8), progress("a", 1, 0.5, 0.75, 4, 8), progress("a", 1, 0.75, 1, 6, 8)];
	let most = 0;
	for (const p of events)
		for (let s = 0; s < 3; s += 0.5) {
			const got = progressAt(p, s).share;
			most = Math.max(most, got);
			assert.ok(got < 1 && most >= got, `${JSON.stringify(p)} +${s}`);
		}
});

test("askAgain: sent again only on fetch's own limit (undici's codes on the cause), and only so long in all", async () => {
	const over = Object.assign(new TypeError("fetch failed"), { cause: { code: "UND_ERR_HEADERS_TIMEOUT" } });
	let n = 0;
	await assert.rejects(askAgain(async () => (n++, await new Promise((r) => setTimeout(r, 5)), Promise.reject(over)), 60), (e) => e === over);
	assert.ok(n >= 2 && n < 30, `${n} sends`);
	const refused = Object.assign(new TypeError("fetch failed"), { cause: { code: "ECONNREFUSED" } });
	n = 0;
	await assert.rejects(askAgain(async () => (n++, Promise.reject(refused))), (e) => e === refused);
	assert.equal(n, 1);
});

test("lookupBar: which reading and the seconds left (from 5 s; minutes past 90 s), shorter where narrow, the bar in heavy blue on the rule", () => {
	const th = { fg: (c: string, s: string) => (c === "rule" ? `{${s}}` : s), bold: (s: string) => s };
	const bar = (share: number, reading: number, left: number | undefined, w: number) => plain(lookupBar(th, share, reading, left, w));
	assert.equal(bar(0.5, 0, 40, 46), `reading 1 of 2 · 40 s left ${"━".repeat(9)}╸{${"─".repeat(9)}}`);
	assert.equal(bar(0.25, 1, 200, 40), `reading 2 of 2 · 3 min left ${"━".repeat(3)}{${"─".repeat(9)}}`);
	assert.equal(bar(0.5, 1, 3, 30), `reading 2 of 2 ${"━".repeat(7)}╸{${"─".repeat(7)}}`, "under 5 s: no seconds");
	assert.equal(bar(0.5, 0, undefined, 30), `reading 1 of 2 ${"━".repeat(7)}╸{${"─".repeat(7)}}`, "past the estimate: no seconds");
	assert.equal(bar(0, 0, 40, 33), `1 of 2 · 40 s left {${"─".repeat(14)}}`, "narrow: shorter");
	assert.equal(bar(1, 0, 40, 22), `1 of 2 ${"━".repeat(15)}{}`, "narrower: the bar keeps its least");
	assert.equal(visibleWidth(lookupBar(piTheme, 0.3, 0, 40, 104)), 104);
	assert.ok(lookupBar(piTheme, 0.5, 0, 40, 46).includes(colourCode(COLOURS.help, "truecolor") + "━"));
});

/** A dock on a 30-token reply with a clock, g pressed and its answer held back (`answerIt`); `show(ms, w)` draws it
 * then; `feed(...)` sends the readings' progress for the newest ask as the live stream does. */
function reading(extra: Record<string, any> = {}) {
	const clock = { t: 0 }, live = reply(30);
	let answerIt: (a: LookupAnswer) => void = () => {};
	const k = dock(live, ["lookup"], () => new Promise((r) => (answerIt = r)), { now: () => clock.t, ...extra });
	k.d.stepIn(20);
	k.d.handleInput("g");
	const show = (ms: number, w = 104) => ((clock.t = ms), text(k.d.render(w)));
	const feed = (reading: number, done: number, next: number, elapsed: number, expect: number, ask = k.ids[k.ids.length - 1]) => {
		onEvent(live, { type: "lookup", client: "me", ask, reading, done, next, elapsed, expect });
		live.lookup!.at = clock.t;                                                                 // stamped on the dock's clock
	};
	return { ...k, clock, live, show, feed, answer: (a: LookupAnswer) => answerIt(a) };
}

test("the readings' bar: the title line from the first progress (board, one-panel board, mini dock), g stop on the key lines, never back; the answer ends it", async () => {
	const k = reading();
	const busy = k.show(100);
	assert.equal(titleLine(busy), `${TITLE}  ${LABELS.lookupBusy}`, "before the first progress");
	k.feed(0, 0, 1, 0, 20);                                                                    // reading 1 of 2, 20 s expected (ds4)
	assert.match(titleLine(k.show(100))!, /reading 1 of 2 · 40 s left ─+$/, "drawn as it comes");
	const early = k.show(5100);
	assert.match(titleLine(early)!, /^ENGRAM INSPECTION {2}reading 1 of 2 · 35 s left ━+╸?─+$/, early.join("\n"));
	const w = visibleWidth(early[1]);                                                          // the dock's inner width
	assert.equal(visibleWidth(titleLine(early)!), w);
	assert.ok(early.some((l) => l.includes(LABELS.engramStop)) && !early.some((l) => l.includes(LABELS.engramClose)), "g stop on the key line");
	assert.match(early[1], /^─+$/, "no legend while it reads");
	const fill = (ls: string[]) => (titleLine(ls)!.match(/━/g) ?? []).length;
	assert.equal(fill(early), Math.floor(((5 / 20) / 2) * (w - TITLE.length - 2 - "reading 1 of 2 · 35 s left".length - 1)));
	k.clock.t = 25_000;
	k.feed(1, 0, 1, 0, 19);                                                                    // reading 1 took 19 s: reading 2
	const mid = k.show(30_000);
	assert.match(titleLine(mid)!, /reading 2 of 2 · 14 s left/);
	assert.ok(fill(mid) > fill(early));
	// Another dock (the dock shown again, /reload): the same bar, from when the progress came, never back.
	k.live.lookup!.most = 0.9;
	const other = new Dock({ terminal: { rows: 67 }, requestRender() {} }, piTheme, k.host);
	other.stepIn(20);
	const shown = text(other.render(104));
	assert.ok(fill(shown) > fill(mid), "the most a bar showed");
	k.live.lookup!.most = 0;
	assert.equal(fill(text(other.render(104))), fill(mid), "from when it came, not from this dock's first look");
	other.dispose();
	// The one-panel board and the mini dock: the same line, narrower.
	const one = k.show(30_000, 70);
	assert.ok(one.some((l) => /^ENGRAM INSPECTION {2}reading 2 of 2 · 14 s left ━+─+$/.test(l)), one.join("\n"));
	assert.ok(one.some((l) => l.includes(LABELS.engramStop)), "g stop on the one-panel key lines");
	k.host.mini = true;
	const mini = k.show(30_000, MINI_WIDTH);
	assert.match(mini[0], /^reading 2 of 2 · 14 s left ━+╸?─+$/, "whole at the mini dock's 50 columns");
	assert.ok(mini.some((l) => l.includes(LABELS.engramStop)), mini.slice(-2).join("\n"));
	k.host.mini = false;
	k.feed(1, 0, 1, 0, 19);                                                                    // a copy, later: the bar from where it was, not back
	assert.equal(fill(k.show(30_000)), fill(mid));
	const late = k.show(200_000);                                                              // long past the estimate: no seconds, short of full
	assert.match(titleLine(late)!, /reading 2 of 2 ━+╸?─+$/);
	k.answer(answer(30, { 5: 1.95 }));
	await tick();
	const T = 300_000;
	k.show(T);
	assert.match(text(k.d.render(104)).join("\n"), /MAP +30 tok · 1 helped/);
	assert.ok(k.show(T + 2 * SWEEP_MS).some((l) => l.includes(LABELS.engramClose)), "answered: g close");
	k.d.dispose();
});

test("g while it reads: stopped at once with the closing pass (stopped on the title and MAP), the server told; g again asks anew, the old ask's progress ignored", async () => {
	const k = reading();
	k.feed(0, 0, 1, 2, 20);
	k.show(3000);
	k.clock.t = 4000;
	k.d.handleInput("g");
	assert.deepEqual(k.stops, [["T1", k.ids[0]]]);
	assert.equal(k.host.lookupOn, "T1", "on while it closes");
	const closing = k.show(4000 + SWEEP_MS / 2);
	assert.match(closing[1], new RegExp(`^── ${LABELS.engramClosing} ─+$`));
	assert.equal(titleLine(closing), `${TITLE}  ${LABELS.engramStopped}`);
	assert.match(closing.join("\n"), /MAP +30 tok · stopped/);
	const after = k.show(4000 + SWEEP_MS);
	assert.equal(k.host.lookupOn, undefined);
	assert.ok(after.includes("REPLY") && /^─+$/.test(after[1]));
	k.answer(answer(30, { 5: 1.95 }));                                                        // the stopped ask's 409 or answer: dropped
	await tick();
	assert.equal(k.host.looked.get("T1"), undefined);
	k.d.handleInput("g");
	assert.deepEqual(k.asked, ["T1", "T1"]);
	k.feed(1, 0, 1, 0, 19, k.ids[0]);                                                          // the old ask's late progress
	assert.equal(titleLine(k.show(5000)), `${TITLE}  ${LABELS.lookupBusy}`);
	k.feed(0, 0, 1, 0, 20);                                                                    // the new ask's: from its own start
	assert.match(titleLine(k.show(5000))!, /reading 1 of 2 · 40 s left ─+$/);
	k.d.dispose();
});

test("g on another reply while one reads: the first stopped; the bar's ticker (mock timers): every BAR_TICK_MS while drawn, stopped unseen, with the answer, with the dock", async (t) => {
	t.mock.timers.enable({ apis: ["setInterval"] });
	let asked = 0;
	const k = reading();
	const d = new Dock({ terminal: { rows: 67 }, requestRender: () => void asked++ }, piTheme, k.host);
	k.d.dispose();
	d.stepIn(20);
	k.feed(0, 0, 1, 0, 20);
	k.clock.t = SWEEP_PASSES * SWEEP_MS + 100;                                                 // the opening sweep held
	d.render(104);
	asked = 0;
	t.mock.timers.tick(BAR_TICK_MS);
	assert.equal(asked, 1);
	d.render(104);
	t.mock.timers.tick(BAR_TICK_MS);
	assert.equal(asked, 2);
	t.mock.timers.tick(BAR_TICK_MS);
	t.mock.timers.tick(BAR_TICK_MS);
	assert.equal(asked, 2, "not drawn since: stopped");
	d.render(104);                                                                             // drawn again: going again
	t.mock.timers.tick(BAR_TICK_MS);
	assert.equal(asked, 3);
	k.answer(answer(30, { 5: 1.95 }));
	await tick();
	k.clock.t += 10_000;
	d.render(104);
	asked = 0;
	t.mock.timers.tick(5 * BAR_TICK_MS);
	assert.equal(asked, 0, "answered: no bar, no ticks");
	// Another reply's g while this one reads: this one stopped.
	k.host.looked.clear();
	d.handleInput("g");
	d.handleInput("g");
	assert.equal(k.asked.length, 2);
	k.feed(0, 0, 1, 0, 20);
	d.render(104);
	onEvent(k.live, { type: "start", rid: "r2", model: "Flash" });
	onEvent(k.live, { type: "token", rid: "r2", i: 0, id: 5, text: " x", p: 0.9, entropy: 0, alts: [], n_layers: null });
	onEvent(k.live, { type: "end", rid: "r2", n_tokens: 1, trace_id: "T2" });
	const settle: (() => void)[] = [];
	k.host.stopLookup = (t: string, a: string) => (k.stops.push([t, a]), new Promise<void>((r) => settle.push(r)));
	d.handleInput("g");
	assert.deepEqual(k.stops, [["T1", k.ids[1]]]);
	await tick();
	assert.deepEqual(k.asked, ["T1", "T1"], "the new ask waits until the server took the stop");
	// g on a third reply at once: the second (never sent) dropped, the third waits for every stop before it.
	onEvent(k.live, { type: "start", rid: "r3", model: "Flash" });
	onEvent(k.live, { type: "token", rid: "r3", i: 0, id: 5, text: " y", p: 0.9, entropy: 0, alts: [], n_layers: null });
	onEvent(k.live, { type: "end", rid: "r3", n_tokens: 1, trace_id: "T3" });
	d.handleInput("g");
	assert.equal(k.stops.length, 2);
	settle[1]();
	await tick();
	assert.deepEqual(k.asked, ["T1", "T1"], "the first stop not taken yet: nothing sent");
	settle[0]();
	await tick();
	assert.deepEqual(k.asked, ["T1", "T1", "T3"]);
	d.dispose();
	asked = 0;
	t.mock.timers.tick(5 * BAR_TICK_MS);
	assert.equal(asked, 0, "gone with the dock");
});

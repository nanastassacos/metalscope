import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mock, test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { BOARD_MIN_ROWS, BOARD_MIN_WIDTH, KEYS_W, LAYER_WINDOW, LIST_ROWS, MIDDLE_MIN, MINI_WIDTH, type Panel, PANEL_KEY, PANELS, RIGHT_W, TOP_N, arrange, boardFits, boardKeys, columns, layoutKeys, panelNums,
	DOCK_MIN_ROWS, DOCK_MIN_WIDTH, miniAt, steerColumns, steerKeys, tabRow, wrapKeys, editKeys, explainOf, expertsLayers, expertsStack, layerWindow, lensSplit, MAP_LINES, mapCells, type MapPaint, type MapRun, mapPanel, mapStarts, mapWindow, MINI_MAP, type MiniRoom, miniKeys, miniSplit, nameTag, scrollWindow, SETTLE_STOPS, settleLayers, settleLevels, sessionMap, settlingPanel, stackLayers, tokensPanel, topChoices, topsOf } from "../board.ts";
import { COLOURS, EXPLAIN, LABELS, ansiBold, ansiInverse, colourCode, cut, dockTheme, label, tint, wrapTokens } from "../draw.ts";
import { createHash } from "node:crypto";
import { TuiMainScreen, wrapTextWithAnsi } from "@earendil-works/pi-tui";
import * as lights from "../lights.ts";
import { STOPS, colourMode, gridCells, mix, palette, rampAt, reshapeGrid } from "../lights.ts";
import { MIN_LINES, swapFrames } from "../telescope.ts";
import { Dock } from "../dock.ts";
import { fillRouting } from "../expert.ts";
import { registerJobs } from "../job-runner.ts";
import { type Live, type Reply, type TokenEvent, editMark, mapReply, onEvent, switchServer, thinkEnd } from "../state.ts";
import { parseSteering } from "../steerpage.ts";
import { createUi, dockColumns, readPrefs, writePrefs } from "../widget.ts";

/** The mini dock's layers of lights drawn, by their headings (expertsStack's): "L1" for layer 1. */
const miniL = (ls: string[]) => ls.flatMap((l) => /^[▸ ] layer (\d+) of \d+$/.exec(l)?.slice(1).map((n) => `L${n}`) ?? []);
const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const text = (lines: string[]) => lines.map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd());
/** Half-block lines (truecolor) back to their ▀ cells' colours ([]: the background): a row of cells a half line, the
 * upper first. */
const halfRows = (lines: string[]): number[][][] => lines.flatMap((l) => {
	const up: number[][] = [], lo: number[][] = [];
	let f: number[] = [], b: number[] = [];
	for (const m of l.matchAll(/\x1b\[([34])8;2;(\d+);(\d+);(\d+)m|\x1b\[49m|▀/g))
		if (m[0] === "▀") up.push(f), lo.push(b);
		else if (m[0] === "\x1b[49m") b = [];
		else if (m[1] === "3") f = [+m[2], +m[3], +m[4]];
		else b = [+m[2], +m[3], +m[4]];
	return [up, lo];
});
const tick = () => new Promise((r) => setTimeout(r, 0));
const KEY = { up: "\x1b[A", down: "\x1b[B", right: "\x1b[C", left: "\x1b[D", tab: "\t", shiftTab: "\x1b[Z", esc: "\x1b", enter: "\r" };
const piTheme = { fg: (_c: string, s: string) => s, bg: (_c: string, s: string) => s, bold: (s: string) => s };

// 48 layers read at every fourth; the written token settles at L40. Every token routed (4 MoE layers, top 2 of 64).
const LAYERS = Array.from({ length: 12 }, (_, i) => i * 4);
const tok = (i: number, extra: Partial<TokenEvent> & Record<string, unknown> = {}): any => ({
	type: "token", rid: "r", i, id: 100 + i, text: ` w${i}`, p: i % 5 === 0 ? 0.3 : 0.95, entropy: 0.5, alts: [[" other", 0.6], [" else", 0.05]],
	n_layers: 48,
	lens: LAYERS.map((l) => [l, l >= 40 ? 100 + i : 7, l >= 40 ? ` w${i}` : " the"]), lens_by: "jlens",
	lens_logit: LAYERS.map((l) => [l, l >= 44 ? 100 + i : 8, l >= 44 ? ` w${i}` : " a"]),
	experts: [1, 2, 3, 4, 5, 6, 7, 63], expert_w: [200, 55, 128, 127, 255, 0, 100, 155], expert_k: 2, ...extra,
});

const MODEL = "Flash";
const STEERING = { checkpoint: "Flash", concepts: [
	{ name: "joyful", title: "Joyful", description: "Upbeat and warm.", chat: { turns: 18, default_dose: 0.4, layer: 30,
		plain: { good: 0, turns: 18, style: 0.1, coherence: 2.9 },
		strengths: [{ dose: 0.2, good: 9, turns: 18, style: 1.2, coherence: 2.9 }, { dose: 0.4, good: 15, turns: 18, style: 2.2, coherence: 2.7 }] } },
	{ name: "terse", title: "Terse", description: "Short.", chat: null },
] };

/** A dock `rows` terminal lines tall on a done reply of `n` tokens, with the steer page's concepts. */
function board(n = 40, rows = 67, extra: Record<string, any> = {}, tx: Record<string, any> = {}, done = true) {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r", model: MODEL });
	for (let i = 0; i < n; i++) onEvent(live, tok(i, tx));
	if (done) onEvent(live, { type: "end", rid: "r", n_tokens: n, trace_id: "T1" });
	const steered: string[] = [];
	const host: any = { live, captions: false, close() {}, unfocus() {},
		fetchJson: async () => null, steerModel: () => MODEL, steering: async () => parseSteering(STEERING, MODEL),
		steerChats: async () => ({ dose: 0.4, turns: [] }), steer: async (a: string) => { steered.push(a); return [{ text: `steer: ${a}`, level: "info" }]; },
		...extra };
	const d = new Dock({ terminal: { rows }, requestRender() {} }, piTheme, host);
	return { d, host, live, steered };
}

test("wrapKeys: whole keys a line, split at the separator; a key wider than the line wrapped on its own", () => {
	assert.deepEqual(wrapKeys("tab lens · ←/→ token · f live", 30), ["tab lens · ←/→ token · f live"]);
	assert.deepEqual(wrapKeys("tab lens · ←/→ token · f live", 20), ["tab lens · ←/→ token", "f live"]);
	assert.deepEqual(wrapKeys("←→ token  ↑↓ pick  esc back", 18, "  "), ["←→ token  ↑↓ pick", "esc back"]);
	assert.ok(wrapKeys("a · " + "x".repeat(30), 10).every((l) => visibleWidth(l) <= 10));
	// The board's key line at the one-panel board's widths: two lines at most, g kept (short where tight).
	for (const w of [41, 57, 77, 101]) for (const p of PANELS) {
		const line = boardKeys(p, true, false, PANELS, true, false, true, true, "", w, LABELS.engramKey, true);
		const ls = wrapKeys(line, w);
		assert.ok(ls.length <= 2 && ls.every((l) => visibleWidth(l) <= w), `${p} ${w}: ${ls.join(" / ")}`);
		if (w >= 57) assert.match(line, /g engram/, `${p} ${w}: g kept`);   // the first to go where tight
	}
	// Every state at every one-panel width: two lines hold all the parts kept (the dock draws two), the token, layer and
	// steer keys always among them.
	for (let w = 41; w <= 97; w++) for (const p of PANELS) for (const held of [false, true]) for (const steered of [false, true]) for (const layers of [false, true])
		for (const g of ["", LABELS.engramKey, "g close"]) for (const to of ["", "this session"]) {
			const line = boardKeys(p, true, p === "EXPERTS", PANELS, held, steered, layers, true, p === "EXPERTS" ? to : "", w, g, true);
			const what = `${w} ${p} held ${held} steered ${steered} layers ${layers} g ${g} to ${to}: ${line}`;
			assert.ok(wrapKeys(line, w).length <= 2, what);
			if (p === "LENS") assert.ok(line.includes("←/→ token") && line.includes("↑/↓ layer"), what);
			if (steered && p === "LENS") assert.match(line, /x off/, what);
			if (held) assert.match(line, /f live/, what);
		}
	// The tightest: a 44-column dock, LENS, a token held, a steer on, the lens steps known: -/+ layers gives way, then
	// the numbers (1-4 6 panel, four columns wider than 6 steer).
	assert.deepEqual(wrapKeys(boardKeys("LENS", true, false, PANELS, true, true, true, true, "", 41, "", true, "", true, true, ["MAP", "LENS", "TOKENS", "EXPERTS"]), 41),
		["tab tokens · ←/→ token · f live", "↑/↓ layer · enter edit · x off"]);
	assert.deepEqual(wrapKeys(boardKeys("LENS", true, false, PANELS, true, false, true, true, "", 41, "", true, "", true, true, ["MAP", "LENS", "TOKENS", "EXPERTS"]), 41),
		["tab tokens · ←/→ token · f live", "↑/↓ layer · enter edit · 1-4 6 panel"], "no steer: the numbers kept");
	// The word edit's keys on two lines: typing and tab kept from 41 columns.
	for (const w of [41, 51, 61]) assert.match(editKeys(w, 2), /type or ↑\/↓ a word · tab next/, `${w}`);
	assert.equal(editKeys(41), "⌫ remove · enter apply · esc cancel");
});

test("miniAt: the mini dock where chosen, or in a terminal too short for the full one (never for its columns); the least dock; the Steer page stacks below 84 columns", () => {
	assert.deepEqual([miniAt(true), miniAt(false), miniAt(false, 22), miniAt(false, 23), miniAt(undefined, 22), miniAt(undefined)], [true, false, true, false, true, false]);
	assert.equal(MINI_WIDTH, 3 + RIGHT_W, "the right column's width");
	assert.deepEqual([DOCK_MIN_WIDTH, DOCK_MIN_ROWS], [44, 23]);
	assert.deepEqual([steerColumns(83).stacked, steerColumns(84).stacked, steerColumns(101).stacked], [true, false, false]);
	assert.equal(steerColumns(41).across, 1);
	// The tab row: each panel with its number; on the board its number again back where it fits, on the one-panel
	// board nothing to go back to. Every name whole at the widths the dock draws it: the board's full-size panel from 97
	// columns, the one-panel board's (no STREAM) from 41.
	assert.equal(plain(tabRow(piTheme, "LENS", 97, PANELS)).trimEnd(), " 1 MAP   2 LENS   3 TOKENS   4 EXPERTS   5 STREAM   2 back");
	assert.equal(plain(tabRow(piTheme, "STREAM", 56, PANELS)).trimEnd(), " 1 MAP   2 LENS   3 TOKENS   4 EXPERTS   5 STREAM", "back first to go");
	assert.equal(plain(tabRow(piTheme, "LENS", 41, ["MAP", "LENS", "TOKENS", "EXPERTS"], true)).trimEnd(), " 1 MAP   2 LENS   3 TOKENS   4 EXPERTS");
	for (const one of [false, true]) for (let w = one ? 41 : 97; w <= 141; w++) for (const shown of SHOWN.filter((s) => !one || !s.includes("STREAM")))
		assert.ok(shown.every((p) => plain(tabRow(piTheme, "MAP", w, shown, one)).includes(` ${PANEL_KEY[p]} ${p} `)), `${w} ${one}`);
});

test("the one-panel board: tab cycles the four panels, a number picks one, the reply's last lines over the panel; TOKENS with EXPERTS when tall; below 44x23 one line", async () => {
	const { d, host } = board(40, 30);
	await tick();
	d.stepIn(20);
	const order: string[] = [];
	for (let k = 0; k < 4; k++) {
		const lines = text(d.render(80));
		assert.equal(lines.length, 30);
		assert.ok(lines.some((l) => /^ ?1 MAP +2 LENS +3 TOKENS +4 EXPERTS$/.test(l)), "the tab row");
		assert.equal(lines[2], "REPLY");
		assert.equal(lines[6], "─".repeat(77), "three lines of the reply, then the rule");
		order.push(host.panel ?? "LENS");
		d.handleInput("\t");
	}
	assert.deepEqual(order, ["LENS", "TOKENS", "EXPERTS", "MAP"]);
	assert.equal(host.panel, "LENS", "round again");
	d.handleInput("3");
	assert.deepEqual([host.panel, d.full], ["TOKENS", false], "a number: that panel, nothing full size");
	assert.ok(text(d.render(80)).some((l) => /^#20 /.test(l)), "TOKENS drawn");
	d.handleInput("5");
	assert.equal(host.panel, "TOKENS", "5: no STREAM on the one-panel board");
	// A full-size panel kept through the one-panel board: a number there picks the panel, the board opens it again.
	const kept = board(40, 40);
	await tick();
	kept.d.stepIn(20);
	kept.d.render(104);
	kept.d.handleInput("3");
	(kept.d as any).tui.terminal.rows = 30;
	kept.d.render(104);
	kept.d.handleInput("2");
	(kept.d as any).tui.terminal.rows = 40;
	const back = text(kept.d.render(104));
	assert.deepEqual([kept.d.full, kept.host.panel], [true, "LENS"]);
	assert.ok(back.some((l) => /^ ?1 MAP +2 LENS +3 TOKENS +4 EXPERTS +2 back$/.test(l)), "LENS full size on the board");
	// Tall enough: TOKENS with EXPERTS under it.
	const tall = board(40, 67);
	await tick();
	tall.d.stepIn(20);
	tall.host.panel = "TOKENS";
	const lines = text(tall.d.render(80));
	// The tab row names TOKENS: no title of its own under it; EXPERTS under it keeps its.
	assert.ok(lines.some((l) => /^#20 /.test(l)) && !lines.some((l) => /^ ?TOKENS /.test(l)) && lines.some((l) => /^ ?EXPERTS /.test(l)), lines.join("\n"));
	// Too short for the least dock: the mini dock (down to the dock's least 12 rows); too narrow: one line, the frame its size.
	for (const rows of [22, 12]) {
		(d as any).tui.terminal.rows = rows;
		const out = d.render(80);
		assert.equal(out.length, rows);
		assert.match(text(out).join("\n"), /^[▸ ] layer \d+ of \d+$/m, "the mini dock's grids");
	}
	(d as any).tui.terminal.rows = 30;
	const small = d.render(43);
	assert.equal(small.length, 30);
	assert.equal(text(small)[0], "dock too small: needs 44x23");
});

/** The board's least height before it went down to 36 (BOARD_MIN_ROWS): the tests written at it keep it. */
const ROWS_51 = 51;

test("board: drawn from 100 columns and 36 lines (the fixed parts, the middle at its shortest, two lines for ?); narrower or shorter, the one-panel board", async () => {
	assert.equal(BOARD_MIN_WIDTH, 100);
	assert.equal(BOARD_MIN_ROWS, 36);
	assert.ok(boardFits(104, 60) && boardFits(104, 36) && !boardFits(99, 60) && !boardFits(104, 35));
	const { d } = board();
	await tick();
	const one = (lines: string[]) => lines.some((l) => /^ ?1 MAP +2 LENS /.test(l));        // the one-panel board's tab row
	assert.ok(!one(text(d.render(104))));
	assert.ok(one(text(d.render(99))), "99 columns: the one-panel board");
	const short = board(40, BOARD_MIN_ROWS - 1);                           // the dock is the terminal's whole height
	assert.ok(one(text(short.d.render(104))), "35 lines: the one-panel board");
	const tall = board(40, BOARD_MIN_ROWS);
	assert.ok(!one(text(tall.d.render(104))), "36 lines: the board");
});

test("board at 104 columns: every line exactly the dock's width, the columns where the layout puts them", async () => {
	const { d } = board();
	await tick();
	d.sel = 20;
	const raw = d.render(104);
	assert.equal(raw.length, 67);
	for (const l of raw) assert.equal(visibleWidth(l), 104, plain(l));
	const lines = text(raw), w = 101, { lw, rw } = columns(w);
	assert.deepEqual([lw, rw], [48, RIGHT_W]);
	// The header: ◆ metalscope on the left (the counts are MAP's) and, a token stepped to, that the view stays on it;
	// "no steering" ending at the right edge; REPLY alone.
	assert.match(lines[0], /^◆ metalscope {2}paused at #20 {3,}no steering$/);
	assert.equal(lines[2], "REPLY");
	assert.ok(lines.some((l) => /^MAP {2}40 tok · 8 unsure +│/.test(l)), "MAP: the reply's size and unsure count");
	assert.ok(plain(raw[0]).replace(/^[▌▐] /, "").startsWith("◆") && plain(raw[0]).slice(2, 2 + w).endsWith("no steering"));
	// The middle: MAP and LENS (SETTLING in it) on the left, TOKENS / EXPERTS on the right, " │ " between at
	// column lw, down to the key line's rule (STEER is a page of its own).
	const mid = lines.findIndex((l) => /^ ?MAP /.test(l));
	assert.ok(mid > 0 && / │ TOKENS {2}#20 /.test(lines[mid]), lines[mid]);
	assert.ok(lines.some((l) => /│ EXPERTS {2}4 layers · 64 experts · 2 a token$/.test(l)), "every layer: the room is there");
	for (let i = mid; i < lines.length - 2; i++) assert.equal([...plain(raw[i]).slice(2)].slice(lw, lw + 3).join(""), " │ ", `line ${i}: ${lines[i]}`);
	for (const name of ["TOKENS", "EXPERTS"]) assert.ok(lines.some((l) => l.includes(` │ ${name}  `)), name);
	assert.ok(!lines.some((l) => /FEATURES|SAE/.test(l)), "no FEATURES");
	for (const name of ["SETTLING", "MAP"]) assert.ok(lines.some((l) => l.startsWith(`${name}  `)), name);
	assert.ok(lines.findIndex((l) => l.startsWith("MAP")) < lines.findIndex((l) => /^ ?LENS /.test(l)), "MAP over LENS");
	assert.ok(lines.findIndex((l) => /^ ?LENS /.test(l)) < lines.findIndex((l) => l.startsWith("SETTLING")), "SETTLING in LENS, under the telescope");
	assert.ok(!lines.some((l) => /RECENT/.test(l)), "no RECENT");
	// EXPERTS: as many layers as its lines hold, here all four.
	assert.deepEqual(lines.filter((l) => /│ [▸ ] layer \d of 4$/.test(l)).map((l) => l.slice(-14)), ["  layer 1 of 4", "  layer 2 of 4", "▸ layer 3 of 4", "  layer 4 of 4"]);
	assert.ok(lines.some((l) => / │ TOKENS {2}#20 " w20" · 30% · rank 2 · entropy 0\.50$/.test(l)), "off-top left out: it does not fit");
	d.sel = 21;
	assert.ok(text(d.render(104)).some((l) => / │ TOKENS {2}#21 " w21" · 95% · entropy 0\.50$/.test(l)), "not off-top");
	assert.ok(lines.some((l) => /#20 "w20" · settles L40\/48/.test(l)));
	assert.ok(!lines.some((l) => /STEER|by the chat test/.test(l)), "STEER is a page of its own (6)");
	// The key line last; a rule over it.
	assert.match(lines.at(-1)!, /^tab tokens · ←\/→ token · /);
	assert.match(lines.at(-2)!, /^─+$/);
});

/** The panels the board can show: all, or without EXPERTS (a model without experts), STREAM (no writes, no room), both. */
const SHOWN: Panel[][] = [[], ["EXPERTS"], ["STREAM"], ["EXPERTS", "STREAM"]].map((off) => PANELS.filter((p) => !off.includes(p)));

test("board: every key line fits the narrowest board, whatever is shown; tab names the next panel shown", () => {
	for (const order of SHOWN) for (const p of order) for (const windowed of [false, true]) for (const inspecting of [false, true]) {
		const keys = boardKeys(p, inspecting, windowed, order);
		assert.ok(visibleWidth(keys) <= BOARD_MIN_WIDTH - 3, `${p} ${order} ${windowed} ${inspecting}: ${keys}`);
		if (!inspecting) continue;
		const next = order[(order.indexOf(p) + 1) % order.length];
		assert.ok(keys.startsWith(`tab ${next.toLowerCase()} · `), `${p} in ${order}: ${keys}`);
	}
	// Only keys that do something there: s on LENS full size too (SETTLING under both lenses follows it), enter on LENS
	// (x there while a steer is on), n/N only on the Steer page; the numbers (the panels', 6 the Steer page's) on every
	// panel's line, 6 steer before stepping in.
	assert.ok(boardKeys("LENS", true).includes("s other lens"));
	for (const p of PANELS) assert.ok(!/x off|n\/N/.test(boardKeys(p, true)) && boardKeys(p, true).includes("1-6 panel"), p);
	for (const p of PANELS) assert.equal(/enter edit/.test(boardKeys(p, true)), p === "LENS", p);
	for (const held of [false, true]) {
		const keys = boardKeys("LENS", true, false, PANELS, held, true);
		assert.ok(keys.includes("enter edit · x off") && /1-6 panel( · \? explain)?$/.test(keys) && visibleWidth(keys) <= BOARD_MIN_WIDTH - 3, keys);   // the numbers drop last
	}
	assert.equal(editKeys(99), "⌫ remove · type or ↑/↓ a word · tab next · enter apply, redo from here · esc cancel");
	assert.equal(editKeys(70), "⌫ remove · type or ↑/↓ a word · tab next · enter apply · esc cancel");
	assert.equal(editKeys(41), "⌫ remove · enter apply · esc cancel");                // the narrowest pages dock
	assert.match(boardKeys("LENS", false), /6 steer/);
	assert.equal(boardKeys("LENS", false), "ctrl+\\ steps in: tab a panel, ? explain, 1-5 open, 6 steer · /scope help dock", "not stepped in: the numbers");
	assert.match(boardKeys("LENS", true), /^tab tokens · ←\/→ token · ↑\/↓ layer · enter edit · s other lens · 1-6 panel/);
	assert.match(boardKeys("MAP", true), /^tab lens · ←\/→ token · u\/U unsure · 1-6 panel/);
	assert.equal(boardKeys("TOKENS", true, false, ["LENS", "TOKENS"]), "tab lens · ←/→ token · ↑/↓ unsure token · 2-3 6 panel · ? explain · esc back");
	// The numbers: those that do something (the panels a number opens, 6 the Steer page) as one part, in 6 steer's slot.
	const four: Panel[] = ["MAP", "LENS", "TOKENS", "EXPERTS"], dense: Panel[] = ["MAP", "LENS", "TOKENS"];
	assert.equal(boardKeys("STREAM", true, false, PANELS, true, true, false, true, "", KEYS_W, LABELS.engramKey),
		"tab map · ←/→ token · f live · s view · g engram inspection · 1-6 panel · ? explain · esc back");
	assert.equal(boardKeys("LENS", true, false, PANELS, true, true, true, true, "", KEYS_W),
		"tab tokens · ←/→ token · f live · ↑/↓ layer · -/+ layers · enter edit · x off · 1-6 panel", "the tightest: kept last");
	assert.equal(boardKeys("LENS", true, false, dense, false, false, true, true, "", KEYS_W, "", false, "", true, false, dense),
		"tab tokens · ←/→ token · ↑/↓ layer · -/+ layers · enter edit · s other lens · 1-3 6 panel", "a model without experts or writes");
	assert.deepEqual(wrapKeys(boardKeys("MAP", true, false, four, false, false, true, true, "", 57, LABELS.engramKey, true, "", true, true, four), 57),
		["tab lens · ←/→ token · u/U unsure · g engram inspection", "1-4 6 panel · ? explain · esc back"], "the one-panel board: 1-5");
	// The Steer page's: esc back (as on the board), then d delete go first where it does not fit.
	assert.equal(steerKeys(101), "↑/↓ concept · ←/→ dose · enter steer · x off · d delete · n/N turn · ? explain · 6 board · esc back");
	assert.equal(steerKeys(92), "↑/↓ concept · ←/→ dose · enter steer · x off · d delete · n/N turn · ? explain · 6 board");
	assert.equal(steerKeys(85), "↑/↓ concept · ←/→ dose · enter steer · x off · n/N turn · ? explain · 6 board");
	assert.equal(steerKeys(101, 1, true, false, false), "ctrl+\\ steps in: ↑/↓ concept, 6 board · /scope help dock");
});

test("board: ? texts are one or two lines at the narrowest board; the layout's keys say how to get the key line back", () => {
	for (const p of PANELS) for (const windowed of [false, true]) {
		const lines = wrapTextWithAnsi(explainOf(p, windowed), BOARD_MIN_WIDTH - 3);
		assert.ok(lines.length <= 2, `${p} ${windowed}: ${lines.length} lines`);
	}
	assert.notEqual(explainOf("EXPERTS", true), explainOf("EXPERTS"));
	assert.deepEqual([layoutKeys("1-5"), layoutKeys()], ["1-5 open a panel · 6 steer page · m mini · ? keys", "6 steer page · m mini · ? keys"]);
	assert.deepEqual(SHOWN.map((s) => panelNums(s)), ["1-5", "1-3 5", "1-4", "1-3"], "the numbers of the panels that can be opened");
	assert.deepEqual(SHOWN.map((s) => panelNums(s, true)), ["1-6", "1-3 5-6", "1-4 6", "1-3 6"], "with the Steer page's");
	assert.deepEqual(PANEL_KEY, { MAP: "1", LENS: "2", TOKENS: "3", EXPERTS: "4", STREAM: "5" }, "6 the Steer page (STEER_KEY)");
	assert.equal(explainOf("LENS"), EXPLAIN.BOARD_LENS, "LENS's ? covers SETTLING too");
	assert.match(explainOf("LENS"), /SETTLING/);
	for (const inspecting of [false, true]) assert.ok(miniKeys(inspecting).length <= MINI_WIDTH - 3, miniKeys(inspecting));
});

test("board: tab moves the focus down the left column, down the right (TOKENS, EXPERTS) and round; the title in focus is drawn inverse", async () => {
	const { d, host } = board();
	await tick();
	d.sel = 10;
	d.render(104);
	assert.equal(host.panel ?? "LENS", "LENS");
	const order: string[] = [];
	for (let k = 0; k < 4; k++) {
		d.handleInput(KEY.tab);
		order.push(host.panel);
	}
	assert.deepEqual(order, ["TOKENS", "EXPERTS", "MAP", "LENS"], "SETTLING no tab stop: it is in LENS; STEER a page");
	d.handleInput(KEY.shiftTab);
	assert.equal(host.panel, "MAP");
	d.handleInput(KEY.shiftTab);
	assert.equal(host.panel, "EXPERTS");
	const raw = d.render(104);
	assert.ok(raw.some((l) => l.includes("\x1b[7m EXPERTS \x1b[27m")) && !raw.some((l) => l.includes("\x1b[7m LENS \x1b[27m")));
	assert.match(text(raw).at(-1)!, /^tab map · ←\/→ token · f live · \[ \] layers/);
});

test("board: the panel in focus takes the arrows", async () => {
	const { d, host, steered } = board();
	await tick();
	d.sel = 12;
	d.render(104);
	// LENS: ↑/↓ the layer (from the settle layer, L40, a read layer deeper).
	d.handleInput(KEY.up);
	assert.equal(host.lensLayer, 44);
	d.handleInput(KEY.right);
	assert.equal(d.sel, 13);
	// TOKENS: ↓ the previous unsure token, ↑ the next.
	host.panel = "TOKENS";
	d.handleInput(KEY.down);
	assert.equal(d.sel, 10);
	d.handleInput(KEY.up);
	assert.equal(d.sel, 15);
	// EXPERTS: ] the next layer.
	host.panel = "EXPERTS";
	d.handleInput("]");
	assert.equal(host.expertLayer, 3);
	// The Steer page (6): ←/→ the dose, not the token; enter steers with it; 6 the board again.
	d.handleInput("6");
	d.render(104);
	d.handleInput(KEY.left);
	assert.equal(d.sel, 15);
	d.handleInput(KEY.enter);
	await tick();
	assert.deepEqual(steered, ["joyful 0.2"]);
	d.handleInput("x");
	await tick();
	assert.deepEqual(steered, ["joyful 0.2", "off"]);
	d.handleInput("6");
	assert.equal(d.steerPage, false);
});

test("board: a panel's number opens it under the reply, tab switches it, another number that one, the same number goes back", async () => {
	const { d, host } = board();
	await tick();
	d.sel = 20;
	d.render(104);
	d.handleInput("2");
	assert.deepEqual([host.panel, d.full], ["LENS", true]);
	let lines = text(d.render(104));
	const tabs = lines.findIndex((l) => /^ ?1 MAP +2 LENS +3 TOKENS +4 EXPERTS +2 back$/.test(l));
	assert.ok(tabs > 0, "the tab row, no SETTLING in it");
	// SETTLING over both lenses (no steps: no step line), across the width: its sub-heading, its bands each labelled
	// once; a blank line, both lenses side by side, no second LENS title.
	const s = tabs + 1, names = lines.findIndex((l) => /^J-lens +│ logit lens$/.test(l));
	assert.equal(lines[s], "SETTLING  J-lens · old → now");
	assert.ok(names > s + 3 && lines[names - 1] === "", "a blank line, then the lenses");
	assert.ok(lines.filter((l) => /#20 "w20" · settles L4\d\/48/.test(l)).length === 1);
	assert.ok(lines.some((l) => /#20 "w20" · settles L40\/48 +│ #20 "w20" · settles L44\/48/.test(l)));
	const bands = lines.slice(s + 1, names - 1), labels = bands.map((l) => /^L\d+/.exec(l)?.[0]).filter(Boolean);
	assert.ok(bands.every((l) => /^(L\d+)? +▀+$/.test(l)), "every line of SETTLING drawn");
	assert.ok(/^L\d+/.test(bands[0]) && new Set(labels).size === labels.length, `a band labelled once: ${labels}`);
	assert.ok(lines[s + 1].length > 90, "across the width");
	assert.match(lines.at(-3)!, /^both lenses · .* · ◆ the steered layer\s*$/, "the legend last, whole");
	assert.match(lines.at(-1)!, /^tab tokens · ←\/→ token · f live · ↑\/↓ layer · enter edit · s other lens · /);
	d.handleInput(KEY.shiftTab);
	assert.equal(host.panel, "MAP", "MAP before LENS");
	lines = text(d.render(104));
	assert.equal(lines[tabs + 1], "1 reply · 40 tok · 8 unsure");
	d.handleInput(KEY.tab);
	d.handleInput(KEY.tab);
	assert.equal(host.panel, "TOKENS");
	lines = text(d.render(104));
	const top = lines.findIndex((l) => /^#20 " w20" · 30% · rank 2 · entropy 0\.50 · off-top$/.test(l));
	assert.ok(top > 0, "no second TOKENS title; off-top where it fits (60% on another: sampling chose it)");
	assert.match(lines[top + 1], /^ {2}" other" +█+[▏▎▍▌▋▊▉]? +60%$/, "across the width");
	assert.ok(visibleWidth(lines[top + 1]) > 90);
	d.handleInput(KEY.tab);
	assert.equal(host.panel, "EXPERTS");
	assert.equal(d.full, true);
	const raw = d.render(104);
	lines = text(raw);
	assert.ok(lines.some((l) => /^layers 1-4 of 4 · 64 experts · 2 mixed for each token/.test(l)));
	assert.ok(lines.some((l) => /^ {2}layer 1 +layer 2 +▸ layer 3$/.test(l)), "three across, the focus layer marked");
	assert.ok(lines.some((l) => /^ {2}layer 4$/.test(l)), "all four: the fourth under them");
	assert.ok(lines.some((l) => new RegExp(`^${LABELS.forToken('" w20"')} +${LABELS.source.all} +${LABELS.source.session}$`).test(l)), "EXPERTS FOR under the grids");
	assert.ok(lines.some((l) => /^ {2}#5 +█+ +100%/.test(l)), "share bars");
	assert.ok(raw.every((l) => visibleWidth(l) === 104));
	d.handleInput(KEY.tab);
	assert.equal(host.panel, "MAP", "round to MAP");
	d.handleInput("1");
	assert.equal(d.full, false);
	assert.equal(d.sel, 20, "its number left the full-size panel, not the reply");
	d.handleInput("4");
	d.handleInput("3");
	assert.deepEqual([host.panel, d.full], ["TOKENS", true], "another number: straight to that panel");
	d.handleInput("5");
	assert.deepEqual([host.panel, d.full], ["TOKENS", true], "5: no STREAM here (no writes)");
	d.handleInput("3");
	assert.equal(d.full, false);
	const before = text(d.render(104));
	d.handleInput("z");
	assert.deepEqual([text(d.render(104)), d.full], [before, false], "z: nothing");
	for (const l of d.render(104)) assert.equal(visibleWidth(l), 104);
});

test("board: esc on a full-size panel hands the keyboard back, the panel kept and following the reply; stepping in finds it, its number the board; q leaves on the board", async () => {
	let closed = 0;
	const { d, live } = board(40, 67, { close: () => closed++ }, {}, false);   // still streaming
	await tick();
	d.stepIn(-1);
	d.handleInput("4");
	d.handleInput(KEY.esc);
	assert.deepEqual([d.full, d.stepped, d.sel, closed], [true, false, -1, 0], "the keyboard back, the dock open, EXPERTS full size");
	onEvent(live, tok(40));
	let lines = text(d.render(104));
	assert.ok(lines.some((l) => /^ ?1 MAP +2 LENS +3 TOKENS +4 EXPERTS +4 back$/.test(l)), "still full size");
	assert.ok(lines.some((l) => l.startsWith(LABELS.forToken('" w40"'))), "following the newest token");
	assert.match(lines.at(-1)!, /^ctrl\+\\ steps in: tab a panel, \? explain, 1-4 open, .*6 steer · \/scope help dock$/);
	d.stepIn(-1);
	assert.equal(d.full, true, "stepping in finds it");
	d.handleInput("4");
	assert.deepEqual([d.full, d.stepped], [false, true], "4: the board, still stepped in");
	d.handleInput("4");
	d.handleInput("q");
	assert.deepEqual([d.full, d.stepped], [false, false], "q: out at once, on the board");
	// The dock opened for the keys: esc closes it, full size or not.
	d.closeOnEsc = true;
	d.stepIn(-1);
	d.handleInput("4");
	d.handleInput(KEY.esc);
	assert.equal(closed, 1);
});

test("board: from a full-size panel, an edit takes the numbers as letters; the Steer page goes and comes back to it", async () => {
	const { d, host } = board(40, 67, { edit: async () => [] });
	await tick();
	d.sel = 7;
	d.render(104);
	d.handleInput("2");
	d.render(104);
	d.handleInput(KEY.enter);
	d.handleInput("\x7f");
	d.handleInput("3");
	const edit = () => (d as any).edit;
	assert.deepEqual([edit()?.field, host.panel, d.full], ["3", "LENS", true], "3 typed into the word");
	d.handleInput(KEY.esc);
	assert.deepEqual([edit(), d.full], [undefined, true], "esc cancels the edit, LENS still open");
	d.handleInput("6");
	assert.equal(d.steerPage, true);
	d.handleInput("6");
	assert.deepEqual([d.steerPage, d.full, host.panel], [false, true, "LENS"], "back to LENS open");
});

test("board: ? puts what the panel in focus shows where the key line was", async () => {
	const { d, host } = board();
	await tick();
	d.sel = 3;
	d.render(104);
	d.handleInput("?");
	const lines = text(d.render(104));
	assert.ok(lines.some((l) => /^SETTLING: where tokens settled\. Then the token in focus at each layer, deep on top/.test(l)), "LENS's explanation");
	assert.ok(lines.includes(layoutKeys("1-4")));
	assert.ok(!lines.some((l) => /^tab tokens · ←\/→ token/.test(l)));
	assert.equal(host.captions, true);
});

test("board: a steer moves nothing on the board; the Steer page under the header's rule", async () => {
	const { d, live } = board();
	await tick();
	const before = text(d.render(104));
	live.steer = { model: MODEL, parts: [{ name: "joyful", model: MODEL, layer: 30, strength: 0.4, source: "profile" }] };
	for (const sel of [-1, 20]) {
		d.sel = sel;
		const lines = text(d.render(104));
		assert.match(lines[1], /^─+$/, "the rule under the header");
		assert.equal(lines[2], "REPLY");
	}
	d.sel = -1;
	assert.equal(text(d.render(104)).indexOf("REPLY"), before.indexOf("REPLY"), "a steer moves nothing");
	d.handleInput("6");
	const page = text(d.render(104));
	assert.match(page[1], /^─+$/, "the Steer page: the rule under the header");
});

test("board: keys and drawing never throw out of the dock", async () => {
	const { d, live } = board();
	await tick();
	d.sel = 5;
	(live.reply!.tokens[5] as any).text = null;                                         // a malformed token
	assert.doesNotThrow(() => d.render(104));
	assert.match(text(d.render(104)).join("\n"), /the dock: /);
	assert.doesNotThrow(() => d.handleInput("u"));
});

test("board: a token carrying terminal escapes draws them as ·, every line its width, nothing reaching the terminal", async () => {
	const osc = "\x1b]52;c;aGk=\x07\x1b]8;;http://x\x07\x9b2J\r";
	const { d } = board(40, 67, {}, { text: `${osc}w`, alts: [[osc, 0.6]], lens: LAYERS.map((l) => [l, 7, osc]) });
	for (const sel of [-1, 5]) {
		d.sel = sel;
		for (const width of [100, 104, 140]) {
			const out = d.render(width);
			for (const l of out) assert.equal(visibleWidth(l), width, `${width}: ${text([l])[0]}`);
			assert.ok(!out.some((l) => /\x1b\]|[\x07\x80-\x9f\r]/.test(l)), `${width}: an escape reached the terminal`);
		}
	}
});

test("board and mini dock: a key that throws says so on the key line, where the user is", async () => {
	const { d, host } = board();
	await tick();
	d.stepIn(-1);
	(d as any).boardKey = () => { throw new Error("boom"); };
	(d as any).miniKey = () => { throw new Error("bang"); };
	d.handleInput("4");
	assert.ok(text(d.render(104)).some((l) => l.includes("the dock: boom")));
	host.mini = true;
	d.handleInput("4");
	assert.ok(text(d.render(MINI_WIDTH)).some((l) => l.includes("the dock: bang")));
});

test("widget: on the board and the one-panel board the stream asks for routing and lens top words", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	let dock: any;
	const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {},
		custom: (make: any) => new Promise<void>(() => { dock = make({ requestRender() {}, terminal: { rows: 67, columns: 290 } }, {}, {}, () => {}); }) } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false } as any;
	let asks = 0;
	const ui = createUi({ live } as any, join(dir, "pi.json"), () => asks++);
	ui.showDock(ctx);
	assert.deepEqual([live.experts, live.lens, asks], [true, true, 1]);                    // asked for as the dock opens
	dock.render(104);
	dock.render(60);                                                                       // narrow: the one-panel board
	await tick();
	assert.deepEqual([live.experts, live.lens, asks], [true, true, 1]);
});

test("board: an earlier reply traced without routing shows its experts from its saved trace, a line saying so meanwhile", async () => {
	let answer: any;
	const asked: string[] = [];
	const { d, live } = board(40, 67, { fetchJson: async (url: string) => { asked.push(url); return url.includes("/routing") ? answer : null; } });
	for (const t of live.reply!.tokens) { delete t.experts; delete t.expert_w; delete t.expert_k; }
	live.reply!.routed = false;
	// Routing for all 40 reply tokens: 4 layers of top 2, experts 10 + layer, the first one with most of the mix.
	const n = 40, per = 8, ids = Buffer.alloc(n * per * 2), w = Buffer.alloc(n * per);
	for (let t = 0; t < n; t++) for (let j = 0; j < per; j++) { ids.writeUInt16LE(10 + (j >> 1) + (j & 1) * 30, 2 * (t * per + j)); w[t * per + j] = j & 1 ? 55 : 200; }
	answer = { start: 0, n, L: 4, k: 2, ids: ids.toString("base64"), w: w.toString("base64") };
	let lines = text(d.render(104));
	assert.ok(lines.some((l) => / │ EXPERTS$/.test(l)) && lines.some((l) => /│ loading this reply's experts from its saved trace…$/.test(l)), "a line while it loads");
	await tick();
	lines = text(d.render(104));
	assert.ok(lines.some((l) => /│ EXPERTS {2}4 layers · 64 experts · 2 a token$/.test(l)), lines.join("\n"));
	assert.equal(lines.filter((l) => /│ [▸ ] layer \d of 4$/.test(l)).length, 4, "the loaded routing's layers drawn");
	assert.equal(asked.filter((u) => u.includes("/scope/api/traces/T1/routing")).length, 1);
	// Streaming without routing (the stream was not asked): a line says when the experts come.
	const live2 = board(5).live;
	live2.reply!.done = false;
	for (const t of live2.reply!.tokens) delete t.experts;
	const d2 = new Dock({ terminal: { rows: 67 }, requestRender() {} }, piTheme, { live: live2, captions: false, close() {}, unfocus() {}, fetchJson: async () => null });
	assert.ok(text(d2.render(104)).some((l) => /│ this reply streams without routing: its experts/.test(l)));
});

test("an edit's reply takes the routing its own trace saved for its own tokens only, by its trace's index", () => {
	// Reply tokens 0-4: 0-2 kept from the edited reply, 3-4 its own (its trace's rows 0 and 1).
	const r: any = { rid: "b", from: 3, done: true, prefill: 0, tokens: [0, 1, 2, 3, 4].map((i) => ({ i, id: i, text: `t${i}` })) };
	const ids = Buffer.alloc(2 * 2 * 2), w = Buffer.from([200, 55, 201, 56]);
	[7, 8, 9, 10].forEach((e, j) => ids.writeUInt16LE(e, 2 * j));
	fillRouting(r, { start: 0, n: 2, L: 1, k: 2, ids: ids.toString("base64"), w: w.toString("base64") });
	assert.deepEqual(r.tokens.map((t: any) => t.experts && [...t.experts]), [undefined, undefined, undefined, [7, 8], [9, 10]]);
});

test("board: steering on, the header says so and the telescope marks the steered layer", async () => {
	const { d, live } = board();
	await tick();
	d.sel = 7;
	live.steer = { model: MODEL, parts: [{ name: "joyful", model: MODEL, layer: 30, strength: 0.4, source: "profile" }] };
	const raw = d.render(104), lines = text(raw);
	assert.match(lines[0], /^◆ metalscope {2}paused at #7 +STEER joyful@0\.4 L30$/);
	const dt = dockTheme();
	assert.ok(raw[0].includes(dt.fg("accent", ansiBold(ansiInverse(" STEER joyful@0.4 L30 ")))), "a solid chip at the right edge");
	// Cut to half the dock's width, the chip's ends kept.
	live.steer = { model: MODEL, parts: ["joyful", "calm", "terse", "angry"].map((name) => ({ name, model: MODEL, layer: 30, strength: 0.4, source: "profile" as const })) };
	assert.equal(/STEER .*$/.exec(text(d.render(104))[0])![0], "STEER joyful@0.4 L30 + calm@0.4 L30 + terse@0.4…");
	assert.ok(d.render(104)[0].includes(ansiInverse(" STEER joyful@0.4 L30 + calm@0.4 L30 + terse@0.4… ")));
	live.steer = { model: MODEL, parts: [{ name: "joyful", model: MODEL, layer: 30, strength: 0.4, source: "profile" }] };
	const at = lines.find((l) => /◆ steer/.test(l));
	assert.ok(at && /│ L(28|30|31).*◆ steer/.test(at), String(at));                                // in the band holding L30
	assert.equal(lines.filter((l) => /◆ steer/.test(l)).length, 1);
	for (const l of raw) assert.equal(visibleWidth(l), 104);
	live.steer = undefined;
	assert.ok(!text(d.render(104)).some((l) => /◆ steer/.test(l)));
});

test("board: REPLY names the steer the reply ran under: steered LABEL in accent, faint not steered while one is on now; the one-panel board too", async () => {
	const { d, live } = board();
	await tick();
	const dt = dockTheme(), on = { model: MODEL, parts: [{ name: "joyful", model: MODEL, layer: 30, strength: 0.4, source: "profile" as const }] };
	const title = (cols = 104) => text(d.render(cols))[2];
	assert.equal(title(), "REPLY");                                                    // unknown (adopted mid-stream): nothing
	live.steer = on;
	assert.equal(title(), "REPLY");
	live.reply!.steer = "";                                                             // ran unsteered, a steer on now
	assert.equal(title(), "REPLY  not steered");
	assert.ok(d.render(104)[2].includes(dt.fg("faint", "not steered")));
	live.steer = undefined;
	assert.equal(title(), "REPLY");                                                    // none then, none now: nothing
	live.reply!.steer = "joyful@0.4 L30";
	assert.equal(title(), "REPLY  steered joyful@0.4 L30");                            // kept after the steer is off
	assert.ok(d.render(104)[2].includes(dt.fg("accent", "steered joyful@0.4 L30")));
	live.steer = on;
	assert.equal(title(), "REPLY  steered joyful@0.4 L30");
	assert.equal(title(99), "REPLY  steered joyful@0.4 L30");                          // the one-panel board
	live.reply!.steer = "joyful@0.4 L30 + calm@0.4 L30 + terse@0.4 L30";
	assert.equal(title(47), "REPLY  steered joyful@0.4 L30 + calm@0.4 L3…");          // narrow: the label gives way
	for (const l of d.render(104)) assert.equal(visibleWidth(l), 104);
});

test("board: LENS enter edits the eyepiece word in place, its key line; enter answers again from that token with -WORD; one field opened", async () => {
	const edited: any[] = [];
	const { d, host, steered } = board(40, 67, { edit: async (a: string, ids: any, at: any) => {
		edited.push([a, ids, at]);
		return [{ text: `steer: remove ${a.slice(1)}`, level: "info" }];
	} });
	await tick();
	d.sel = 7;
	host.panel = "LENS";
	d.render(104);                                                                     // drawn as the board: its keys
	d.handleInput(KEY.enter);
	let lines = text(d.render(104));
	assert.ok(lines.some((l) => /^│▸w7 /.test(l)), lines.join("\n"));                          // the top word alone: no top 3s sent
	assert.equal(lines.at(-1), "⌫ remove · type or ↑/↓ a word · tab next · enter apply, redo from here · esc cancel");
	d.handleInput("\x7f");
	d.full = true;                                                                     // full size, set: a number is a letter while editing
	lines = text(d.render(104));
	assert.equal(lines.filter((l) => /▸w7 removed/.test(l)).length, 1, "the lens in use only");
	d.full = false;
	d.handleInput(KEY.enter);
	assert.deepEqual(edited, [["-tok107", { tok107: { id: 107, text: "w7" } }, { rid: "r", at: 7, word: "w7" }]]);     // the token shown, by its id
	await tick();
	assert.deepEqual(steered, []);
	assert.equal(text(d.render(104)).at(-1), LABELS.editing);                          // until its reply has the token
});

test("board: LENS edit of a word with control characters: trimmed as the server strips it; one that is only whitespace refused here", async () => {
	for (const [word, sent] of [[";\r\n", ";"], ["\r\n", undefined]] as const) {
		const edited: any[] = [];
		const lens = LAYERS.map((l) => [l, l >= 40 ? 107 : 7, l >= 40 ? word : " the"]);
		const { d, host } = board(40, 67, { edit: async (a: string, ids: any) => (edited.push([a, ids]), []) }, { lens, lens_logit: lens });
		await tick();
		d.sel = 7;
		host.panel = "LENS";
		d.render(104);
		d.handleInput(KEY.enter);
		d.handleInput("\x7f");
		d.handleInput(KEY.enter);
		if (sent) assert.deepEqual(edited, [["-tok107", { tok107: { id: 107, text: sent } }]]);
		else {
			assert.deepEqual(edited, []);
			assert.match(text(d.render(104)).at(-1)!, /is not a word$/);
		}
	}
});

test("board: LENS's keys act on the token it shows: streaming, the newest tokens not read yet, ↑/↓ and enter on the last read one", async () => {
	const { d, host, live } = board(20, 67);
	await tick();
	const r = live.reply!;
	r.done = false;
	for (let i = 20; i < 23; i++) r.tokens.push({ ...r.tokens[19], i, id: 100 + i, text: ` w${i}`, lens: undefined, lens_logit: undefined });
	d.stepIn(-1);                                                                      // following the newest (#22, no lens read)
	host.panel = "LENS";
	assert.ok(text(d.render(104)).some((l) => /^#19 "w19"/.test(l)), "LENS shows #19");
	d.handleInput(KEY.up);
	assert.notEqual(host.lensLayer, undefined, "↑ a layer of #19's");
	d.handleInput(KEY.enter);
	assert.equal(d.sel, 19, "enter: the word edited on #19, the token held");
	assert.equal(text(d.render(104)).at(-1), "⌫ remove · type or ↑/↓ a word · tab next · enter apply, redo from here · esc cancel");
	d.dispose();
});

test("board: LENS enter on a finished reply whose newest tokens have no lens read: the edit opens on the last read one and stays", async () => {
	const { d, host, live } = board(20, 67);
	await tick();
	const r = live.reply!;
	for (let i = 20; i < 23; i++) r.tokens.push({ ...r.tokens[19], i, id: 100 + i, text: ` w${i}`, lens: undefined, lens_logit: undefined });
	d.stepIn(-1);
	host.panel = "LENS";
	d.render(104);
	d.handleInput(KEY.enter);
	const lines = text(d.render(104));
	assert.ok(lines.some((l) => /^│▸w19 /.test(l)), lines.join("\n"));
	assert.equal(lines.at(-1), "⌫ remove · type or ↑/↓ a word · tab next · enter apply, redo from here · esc cancel");
	d.dispose();
});

test("board: a token going in before the one paused on (the old stream's, late): the pause and an open LENS edit stay on their token", async () => {
	const { d, host, live } = board(20, 67, {}, {}, false);
	await tick();
	live.reply!.tokens.splice(2, 1);                                                   // #2 still on the old stream
	d.sel = 5;                                                                         // the token i 6
	host.panel = "LENS";
	d.render(104);
	d.handleInput(KEY.enter);
	assert.ok(text(d.render(104)).some((l) => /^│▸w6 /.test(l)));
	onEvent(live, tok(2));
	const lines = text(d.render(104));
	assert.equal(live.reply!.tokens[d.sel].i, 6);
	assert.ok(lines.some((l) => /^│▸w6 /.test(l)), lines.join("\n"));
	d.dispose();
});

test("board: a LENS edit stays on the token it was opened on (a new reply, a token with no lens: dropped); a refused edit says why", async () => {
	const edited: string[] = [];
	let said: { text: string; level: string }[] = [];
	const { d, host, live } = board(40, 67, { edit: async (a: string) => {
		edited.push(a);
		return said;
	} });
	await tick();
	host.panel = "LENS";
	d.stepIn(-1);                                                                      // following the newest of a finished reply
	d.render(104);
	d.handleInput(KEY.enter);
	assert.ok(text(d.render(104)).some((l) => /^│▸w39 /.test(l)));
	// The next reply starts under the open edit: the eyepiece shows another word, so the edit goes; ⌫ enter edits nothing.
	onEvent(live, { type: "start", rid: "r2", model: MODEL });
	onEvent(live, { ...tok(0), rid: "r2" });
	let lines = text(d.render(104));
	assert.ok(!lines.some((l) => /^│▸w/.test(l)) && /enter edit/.test(lines.at(-1)!), lines.join("\n"));
	d.handleInput("\x7f");
	d.handleInput(KEY.enter);                                                          // a fresh edit, on the word shown now
	d.handleInput(KEY.enter);
	assert.deepEqual(edited, []);
	assert.equal(text(d.render(104)).at(-1), "⌫ removes it, or type a word");
	// Its token loses its lens (no layer in focus): the edit goes, not kept unseen.
	Object.assign(live.reply!.tokens[0], { lens: [], lens_logit: [] });
	d.handleInput("\x7f");
	live.reply!.tokens[0] = { ...tok(0), rid: "r2" };
	assert.ok(!text(d.render(104)).some((l) => /removed/.test(l)));
	// Refused (this server cannot, /steer could not build it): the reason on the key line.
	said = [{ text: LABELS.editHere, level: "error" }];
	d.handleInput(KEY.enter);
	d.handleInput("\x7f");
	d.handleInput(KEY.enter);
	await tick();
	assert.deepEqual([edited, text(d.render(104)).at(-1)], [["-tok100"], LABELS.editHere]);
});

test("board: a saved trace that stopped at the server's cap says where, dim in the header, after the reply ends", async () => {
	const { d, live } = board(40);
	await tick();
	assert.match(text(d.render(104))[0], /^◆ metalscope {3,}no steering$/);
	onEvent(live, { type: "capped", rid: "other", client: "me", tokens: 3 });          // another reply's: nothing
	onEvent(live, { type: "capped", rid: "r", client: "me", tokens: 30 });
	const raw = d.render(104);
	assert.match(text(raw)[0], /^◆ metalscope {2}trace stops at #30 {3,}no steering$/);
	assert.ok(raw[0].includes(dockTheme().fg("dim", "trace stops at #30")), "dim");
	d.sel = 7;
	assert.match(text(d.render(104))[0], /^◆ metalscope {2}paused at #7 {2}trace stops at #30 /);
	onEvent(live, { type: "start", rid: "r2", model: MODEL });                       // the next reply reads its prompt
	onEvent(live, { type: "capped", rid: "r", client: "me", tokens: 31 });          // the last one's, still its own
	assert.equal(live.prev?.capped, 31);
	assert.equal(live.reply?.capped, undefined);
});

test("board: the served model's flags: untested dim in the header; the LENS edit gone where the server refuses it; none known yet change nothing", async () => {
	const edited: string[] = [];
	let flags: Record<string, boolean> | undefined = { tested: false, edit: false };
	const { d, host } = board(40, 67, { flags: () => flags ?? {}, edit: async (a: string) => (edited.push(a), []) });
	await tick();
	d.sel = 7;
	host.panel = "LENS";
	const raw = d.render(104), lines = text(raw);
	assert.match(lines[0], /^◆ metalscope {2}untested {2}paused at #7 {3,}no steering$/);
	assert.ok(raw[0].includes(dockTheme().fg("dim", "untested")), "dim");
	assert.equal(lines.at(-1), "tab tokens · ←/→ token · f live · ↑/↓ layer · s other lens · 1-4 6 panel · ? explain · esc back");   // the dock's own width, past the least
	d.handleInput(KEY.enter);                                                          // not offered: enter edits nothing
	assert.ok(!text(d.render(104)).some((l) => /^│▸w7 /.test(l)) && !/⌫ remove/.test(text(d.render(104)).at(-1)!));
	assert.match(text(d.render(99))[0], /^◆ metalscope {2}untested {2}paused at #7/);  // the one-panel board's header too
	assert.ok(!boardKeys("LENS", true, false, PANELS, false, false, false, false).includes("enter edit"));
	const explained = () => text(d.render(104)).join(" ").replace(/\s+/g, " ");
	host.captions = true;                                                              // ? on LENS: no edit there either
	assert.ok(/SETTLING: where tokens settled\. Then the token in focus .*▸ settled\)\. -\/\+: layers read\./.test(explained()), explained());
	flags = { tested: true, edit: true };
	assert.match(explained(), /▸ settled\)\. enter: remove or swap a word \(experimental\)\. -\/\+/);
	host.captions = false;
	assert.match(text(d.render(104))[0], /^◆ metalscope {2}paused at #7 /);
	assert.match(text(d.render(104)).at(-1)!, /↑\/↓ layer · enter edit · s other lens/);
	flags = undefined;                                                                 // none known yet (no status): as before
	assert.match(text(d.render(104))[0], /^◆ metalscope {2}paused at #7 /);
	d.handleInput(KEY.enter);
	assert.ok(text(d.render(104)).some((l) => /^│▸w7 /.test(l)));
	assert.deepEqual(edited, []);
});

test("board: what came of an edit, once its reply has the token: changed or not, and for a swap whether to the word asked", async () => {
	const { d, host, live } = board(40, 67);
	await tick();
	host.panel = "LENS";
	const r = live.reply!;
	const after = (rid: string, id?: number, said = "", want?: string, end = false) => {
		live.reply = r;
		live.prev = undefined;
		live.edit = { rid: "r", at: 7, steer: "remove~w7/18-44", model: MODEL, word: "w7", sent: true, ...(want ? { want } : {}) };
		onEvent(live, { type: "start", rid, model: MODEL, edit: { of: "r", at: 7 } });
		if (id !== undefined) onEvent(live, { ...tok(0), rid, id, text: said });
		if (end) onEvent(live, { type: "end", rid, n_tokens: id === undefined ? 0 : 1, trace_id: "TA" });
		return text(d.render(104)).at(-1);
	};
	assert.ok(!/^edit:/.test(after("e0")!));                                              // not said before the token comes
	assert.deepEqual(live.reply!.tokens, r.tokens.slice(0, 7));                          // the tokens before it, as they were
	assert.equal(after("e1", 555, " cat", undefined, true), LABELS.edited(label(" w7"), label(" cat"), true));
	assert.equal(live.reply!.tokens[7].i, 7);
	assert.equal(after("e2", 555, " Cat", "cat"), LABELS.edited(label(" w7"), label(" Cat"), true, label("cat"), true));
	assert.equal(after("e3", 107, " w7", "dog"), LABELS.edited(label(" w7"), label(" w7"), false, label("dog"), false));
	assert.equal(after("e4", undefined, "", undefined, true), LABELS.edited(label(" w7"), null, false));   // it ended the reply
	assert.equal(LABELS.edited('"the"', '" your"', true, '"kitchen"', false), 'edit: "the" became " your", not "kitchen"');
});

test("board: an edit's outcome is judged against the top 1 there, says when the rest stopped at a tool call, and an edit no reply took says so", async () => {
	const { d, host, live } = board(40, 67, { edit: async () => [] });
	await tick();
	host.panel = "LENS";
	const r = live.reply!;
	// Token 5 was sampled below its top 1 (" other", 0.6 > 0.3): the edited token being that top 1 is no change.
	live.edit = { rid: "r", at: 5, steer: "remove~w5/18-44", model: MODEL, word: "w5", sent: true };
	onEvent(live, { type: "start", rid: "a", model: MODEL, edit: { of: "r", at: 5 } });
	onEvent(live, { ...tok(0), rid: "a", id: 900, text: " other" });
	assert.equal(text(d.render(104)).at(-1), LABELS.edited(label(" other"), label(" other"), false));
	onEvent(live, { type: "end", rid: "a", n_tokens: 1, trace_id: "TA" });
	// The rest stopped at a tool call: said again at its end.
	onEvent(live, { type: "start", rid: "b", model: MODEL, edit: { of: "a", at: 6 } });
	onEvent(live, { ...tok(0), rid: "b", id: 901, text: " ok" });
	assert.equal(text(d.render(104)).at(-1), LABELS.edited(label(" other"), label(" other"), false));
	onEvent(live, { ...tok(1), rid: "b", id: 902, text: "<tool_call>" });
	onEvent(live, { type: "end", rid: "b", n_tokens: 2, trace_id: "TB" });
	assert.equal(text(d.render(104)).at(-1), LABELS.edited(label(" other"), label(" other"), false, undefined, false, true));
	assert.match(LABELS.edited('" a"', '" b"', true, undefined, false, true), /became " b", stopped at a tool call$/);
	// An edit sent whose run ended with no reply taking it (Pi's request failed): not edited.
	live.reply = r;
	live.prev = undefined;
	d.sel = 7;
	d.render(104);
	d.handleInput(KEY.enter);
	d.handleInput("\x7f");
	live.edit = { rid: "r", at: 7, steer: "remove~w7/18-44", model: MODEL, word: "w7" };
	d.handleInput(KEY.enter);
	await tick();
	assert.equal(text(d.render(104)).at(-1), LABELS.editing);
	live.edit.over = true;
	assert.equal(text(d.render(104)).at(-1), LABELS.editNot);
});

/** An edit of reply `of` at token `at` answered: the edited token (`id`, `said`), then the rest (`n` more tokens). */
function edited(live: Live, of: string, at: number, id: number, said: string, n = 3, rid = `e${of}${at}`) {
	live.edit = { rid: of, at, steer: `remove~w${at}/18-44`, model: MODEL, word: `w${at}`, sent: true };
	onEvent(live, { type: "start", rid: `${rid}a`, model: MODEL, edit: { of, at } });
	onEvent(live, { ...tok(0), rid: `${rid}a`, id, text: said });
	onEvent(live, { type: "end", rid: `${rid}a`, n_tokens: 1, trace_id: `T${rid}a` });
	onEvent(live, { type: "start", rid: `${rid}b`, model: MODEL, edit: { of: `${rid}a`, at: at + 1 } });
	for (let i = 0; i < n; i++) onEvent(live, { ...tok(i), rid: `${rid}b`, id: 800 + i, text: ` new${i}` });
	onEvent(live, { type: "end", rid: `${rid}b`, n_tokens: n, trace_id: `T${rid}b` });
}

/** The board's reply strip: its lines under REPLY down to its rule, as drawn (less tint's text colour after each
 * colour's end) or as text. */
function replyLines(d: Dock, raw = false) {
	const drawn = d.render(104), ls = text(drawn), a = ls.indexOf("REPLY"), b = ls.findIndex((l, i) => i > a && /^─+$/.test(l));
	return (raw ? drawn.map(untint) : ls).slice(a + 1, b);
}
/** A drawn line less what tint adds: the text colour after each colour's end, the panel's after each background's. */
const untint = (l: string) => l.replaceAll(dockTheme().ink, "").replaceAll(colourCode(COLOURS.bg, "truecolor", true), "");
const marks = (d: Dock) => replyLines(d).join("\n").match(/▸\S*/g) ?? [];

test("the reply strip marks an edit's token: the word bold on orange where the key line says it became another, a faint ▸ where unchanged", async () => {
	const th = dockTheme(), block = (s: string) => th.bg("accent", th.fg("customMessageBg", `\x1b[1m${s}\x1b[22m`));
	const strip = (d: Dock) => replyLines(d, true);
	// Changed: " w7" (its top 1) became " cat": dark and bold on orange; token 0's 30% is red, kept on its ▸.
	let { d, live } = board(40, 67);
	await tick();
	edited(live, "r", 7, 555, " cat");
	assert.equal(text(d.render(104)).at(-1), LABELS.edited(label(" w7"), label(" cat"), true));
	assert.ok(strip(d).some((l) => l.includes(th.fg("error", " ") + th.fg("error", "▸") + block("cat"))), strip(d).join("\n"));
	assert.deepEqual(marks(d), ["▸cat"]);
	assert.ok(replyLines(d).some((l) => / w6 ▸cat new0 new1 new2$/.test(l)));        // the kept part ends where ▸ starts
	// The cursor on it: its inverse wins on the word, the ▸ beside it orange; stepping on, the block again.
	d.sel = 7;
	assert.ok(strip(d).some((l) => l.includes(th.fg("accent", "▸") + "\x1b[7m\x1b[1mcat\x1b[22m\x1b[27m")), strip(d).join("\n"));
	assert.ok(!strip(d).some((l) => l.includes(block("cat"))));
	d.sel = 9;
	assert.ok(strip(d).some((l) => l.includes(th.fg("error", "▸") + block("cat"))));
	// The one-panel board draws it too.
	d.sel = -1;
	assert.ok(d.render(80).some((l) => untint(l).includes(block("cat"))) && text(d.render(80)).some((l) => l.includes("▸cat")), "one-panel");
	// Unchanged: the token written there is the top 1 it had (" w7", id 107): a faint ▸, the word as drawn.
	({ d, live } = board(40, 67));
	await tick();
	edited(live, "r", 7, 107, " w7");
	assert.equal(text(d.render(104)).at(-1), LABELS.edited(label(" w7"), label(" w7"), false));
	assert.ok(strip(d).some((l) => l.includes(th.fg("faint", "▸") + th.fg("error", "w7"))), strip(d).join("\n"));
	assert.ok(!strip(d).some((l) => l.includes("\x1b[1mw7")));
	d.sel = 7;
	assert.ok(strip(d).some((l) => l.includes(th.fg("faint", "▸") + "\x1b[7mw7\x1b[27m")));
});

test("the reply strip's edit mark: on a long reply's cut strip, only the newest of several edits, gone with the next reply", async () => {
	// A long reply on the shortest board: the strip's text from n lines' worth of tokens before the token shown (past
	// the start here: the mark's index shifted), its newest lines shown.
	const { d, live } = board(1200, ROWS_51);
	await tick();
	edited(live, "r", 1150, 555, " cat");
	const from = 1153 - Math.max(600, replyLines(d).length * 101);
	assert.ok(from > 0 && editMark(live.reply!, from)!.k === 1150 - from && !editMark(live.reply!, 1151));
	assert.deepEqual(marks(d), ["▸cat"]);                                                 // following: the newest lines
	d.sel = 1150;                                                                          // the cursor on it
	assert.deepEqual(marks(d), ["▸cat"]);
	d.sel = 3;
	assert.deepEqual(marks(d), []);                                                        // scrolled off: no mark
	// The edited reply edited again (here before the first edit): only the newest edit is marked.
	d.sel = -1;
	edited(live, live.reply!.rid, 2, 556, " dog", 3, "x");
	d.sel = 2;
	assert.deepEqual(marks(d), ["▸dog"]);
	// Edited after the first edit's token: the first's token is kept, unmarked; the newest marked.
	const b = board(40, 67);
	await tick();
	edited(b.live, "r", 7, 555, " cat");
	edited(b.live, b.live.reply!.rid, 9, 557, " owl", 3, "y");
	assert.deepEqual(marks(b.d), ["▸owl"]);
	assert.ok(replyLines(b.d).some((l) => / w6 cat new0 ▸owl new0/.test(l)));
	// The next request reading its prompt: the edited reply is still the one shown, marked; the next reply's tokens: none.
	onEvent(b.live, { type: "start", rid: "next", model: MODEL });
	assert.deepEqual(marks(b.d), ["▸owl"]);
	onEvent(b.live, { ...tok(0), rid: "next" });
	assert.deepEqual(marks(b.d), []);
});

test("the reply strip unedited: byte for byte as before the edit mark (main 7b5573c's wrapTokens, no underline)", () => {
	const toks: any[] = [
		{ text: "Hello", p: 0.95 }, { text: " world", p: 0.6 }, { text: ",\n\n", p: 0.3 }, { text: "\tthe", p: null },
		{ text: " supercalifragilistic", p: 0.92 }, { text: "\n", p: 0.99 }, { text: " end", p: 0.4 }, { text: ".", p: 0.97 },
	];
	const th = dockTheme(), runs: unknown[] = [];
	for (const w of [6, 12, 40]) for (const [sel, input] of [[-1, -1], [1, 0], [5, 4], [6, 5], [2, 1]]) {
		const { lines, selLine } = wrapTokens(th, toks, w, sel, input);   // its lines as they were (ends is new)
		runs.push({ lines, selLine });
	}
	assert.equal(createHash("sha256").update(JSON.stringify(runs)).digest("hex"), "f5a426c026f750f5cbc880651aeb0d0157e3b4599b60cb5397092e64d4e2d1c2");
	const { live } = board(40, 67);
	assert.equal(editMark(live.reply!), undefined);
	// A marked token of newlines only: its ▸ on the line after; one cut by the width: ▸ and the word kept together.
	assert.deepEqual(wrapTokens(piTheme, toks, 40, -1, -1, { k: 5, changed: true }).lines.map(plain).slice(-2), [" the supercalifragilistic", "▸ end."]);
	assert.deepEqual(wrapTokens(piTheme, toks, 24, -1, -1, { k: 4, changed: false }).lines.map(plain).slice(2, 4), [" the", " ▸supercalifragilistic"]);
});

test("board: a reply whose trace is not saved yet is asked again (0.5, 1, 2, 4 s), then given up with the reason", async () => {
	const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
	const { ROUTING_RETRY_MS } = await import("../expert.ts");
	const saved = ROUTING_RETRY_MS.splice(0, 4, 5, 5, 5, 5);   // the waits shortened for the test
	try {
		let calls = 0, ready = false;
		const n = 39, per = 8, ids = Buffer.alloc(n * per * 2, 0), w = Buffer.alloc(n * per, 100);
		for (let j = 0; j < n * per; j++) ids.writeUInt16LE(3 + (j % 8), 2 * j);
		const fetchJson = async (url: string) => (url.includes("/routing") ? (calls++, ready ? { start: 0, n, L: 4, k: 2, ids: ids.toString("base64"), w: w.toString("base64") } : null) : null);
		const { d, live } = board(40, 67, { fetchJson });
		for (const t of live.reply!.tokens) delete t.experts;
		d.render(104);
		await tick();
		assert.equal(calls, 1);                                    // 404: the end came before the trace was saved
		assert.equal(live.reply!.routingFrom, "loading");
		assert.ok(text(d.render(104)).some((l) => /│ loading this reply's experts from its saved trace…$/.test(l)));
		ready = true;
		await sleep(10);
		d.render(104);                                             // the timer asked for a render; it asks again
		await tick();
		assert.equal(calls, 2);
		assert.equal(live.reply!.routingFrom, "loaded");
		assert.ok(text(d.render(104)).some((l) => /│ EXPERTS {2}4 layers · 64 experts/.test(l)));
		// Never saved: four more tries, then the reason.
		const never = board(40, 67, { fetchJson: async (url: string) => (url.includes("/routing") ? (calls++, null) : null) });
		for (const t of never.live.reply!.tokens) delete t.experts;
		calls = 0;
		for (let k = 0; k < 8; k++) {
			never.d.render(104);
			await sleep(8);
		}
		assert.equal(calls, 5);
		assert.equal(never.live.reply!.routingFrom, "no routing saved for this reply");
		// A dock closed while it waits: no timer outlives it, and the next dock asks.
		const third = board(40, 67, { fetchJson: async () => null });
		for (const t of third.live.reply!.tokens) delete t.experts;
		third.d.render(104);
		await tick();
		third.d.dispose();
		assert.equal(third.live.reply!.routingFrom, "again");
	} finally {
		ROUTING_RETRY_MS.splice(0, 4, ...saved);
	}
});

test("board: a model known dense is not asked for routing; EXPERTS not drawn (TOKENS has the column), not in the tab cycle, 4 does nothing", async () => {
	const asked: string[] = [];
	const { d, live } = board(40, 67, { fetchJson: async (url: string) => (asked.push(url), null) });
	for (const t of live.reply!.tokens) delete t.experts;
	live.dense = MODEL;
	live.reply!.routingFrom = "no routing saved for this reply";   // even after an earlier answer
	let lines = text(d.render(104));
	await tick();
	assert.ok(!lines.some((l) => /EXPERTS/.test(l)) && lines.some((l) => /│ TOKENS /.test(l)), lines.join("\n"));
	d.stepIn(-1);
	d.handleInput("4");
	lines = text(d.render(104));
	assert.ok(!lines.some((l) => /EXPERTS/.test(l)));
	for (const _ of PANELS) d.handleInput("\t");
	assert.notEqual((d as any).host.panel, "EXPERTS");
	assert.ok(!asked.some((u) => u.includes("/routing")));
});

test("board: ? at the shortest board keeps the middle's least rows", async () => {
	const { d, host } = board(40, BOARD_MIN_ROWS, {}, { alts: Array.from({ length: 9 }, (_, k) => [` c${k}`, 0.05 - k * 0.004]) });
	await tick();
	d.sel = 30;
	host.panel = "TOKENS";
	const before = text(d.render(104));
	d.handleInput("?");
	const after = text(d.render(104));
	const choices = (ls: string[]) => ls.filter((l) => /│ [▸ ] " (w\d+|c\d)"/.test(l)).length;
	assert.ok(choices(after) === TOP_N && choices(before) === TOP_N, `${choices(before)} → ${choices(after)}`);
	assert.ok(after.some((l) => l.startsWith("The model's top 10 choices")), "the explanation");
	assert.equal(after.at(-1), layoutKeys("1-4"), "and the layout's keys under it");
	const middle = after.filter((l) => l.includes(" │")).length;
	assert.ok(middle >= MIDDLE_MIN, `${middle} lines of middle`);
});

test("board: the examples follow the dose under the cursor: its saved chats, the unsteered alone at no steering, a line where none was saved", async () => {
	// Saved at 0.2 and 0.4 (the picked one); joyful's doses on the page: 0 (no steering), 0.2, 0.4; terse's 0.3 none.
	const asked: (number | undefined)[] = [];
	const chatsAt = (dose: number) => ({ dose, turns: [{ conv: "A casual chat", n: 1, of: 6, user: "Hi!", plain: { text: "Hello.", style: 0, sense: 3, good: false },
		steered: { text: `Hello at ${dose}!!`, style: 2, sense: 3, good: dose === 0.4 } }] });
	const steerChats = async (_name: string, dose?: number) => {
		asked.push(dose);
		if (dose === 0.2 || dose === 0.4) return chatsAt(dose);
		throw new Error(`not tried at dose ${dose}`);
	};
	const { d } = board(40, 67, { steerChats: async (n: string, dose?: number) => steerChats(n, dose).catch((e) => ({ error: String(e.message) })) });
	await tick();
	d.sel = 3;
	d.render(104);                                                                    // drawn as the board first, as Pi does
	d.handleInput("6");
	d.render(104);
	await tick();
	const page = () => text(d.render(104));
	assert.ok(page().some((l) => /│ steered at 0\.4 +✓ good/.test(l)), "the cursor on the picked dose: its chats");
	assert.ok(page().some((l) => /│ Hello at 0\.4!!/.test(l)));
	d.handleInput(KEY.left);                                                          // 0.2
	d.render(104);
	await tick();
	assert.ok(page().some((l) => /│ steered at 0\.2 +style 2/.test(l)), "← a dose: the chats saved at it");
	assert.ok(page().some((l) => /│ Hello at 0\.2!!/.test(l)));
	d.handleInput(KEY.left);                                                          // 0: no steering
	d.render(104);
	await tick();
	const none = page();
	assert.ok(none.some((l) => /^no steering +style 0 · sense 3$/.test(l)) && none.some((l) => /^Hello\.$/.test(l)), "no steering: the unsteered reply alone");
	assert.ok(!none.some((l) => /steered at|│/.test(l.slice(0, 104)) && /steered at/.test(l)));
	assert.deepEqual(asked, [0.4, 0.2], "no steering: the picked dose's chats again, nothing asked");
	d.handleInput(KEY.right);
	assert.deepEqual(asked, [0.4, 0.2], "asked once a dose");
	// A dose with no saved chat says so.
	const miss: any = board(40, 67, { steerChats: async () => ({ error: "404: not tried at dose 0.2" }) });
	await tick();
	miss.d.sel = 3;
	miss.d.render(104);
	miss.d.handleInput("6");
	miss.d.render(104);
	miss.d.handleInput(KEY.left);
	miss.d.render(104);
	await tick();
	assert.ok(text(miss.d.render(104)).some((l) => l === "no saved chat at 0.2"));
});

test("board: the Steer page's examples stack the chat's turns from the current one, whole, as many as fit; the rest below a cut line, scrolled by the page keys", async () => {
	const long = (k: number) => Array.from({ length: k }, (_, j) => `Line ${j + 1} of a longer reply.`).join("\n\n");
	const turns = [1, 2, 3, 4, 5, 6].map((n) => ({ conv: "A casual chat", n, of: 6, user: `Question ${n}?`, plain: { text: long(2), style: 0, sense: 3, good: false },
		steered: { text: long(n === 4 ? 30 : 3), style: 2, sense: 3, good: true } }));
	turns.push({ conv: "Planning a dinner party", n: 1, of: 6, user: "Dinner?", plain: { text: "Ok.", style: 0, sense: 3, good: false }, steered: { text: "Yes!", style: 2, sense: 3, good: true } });
	const { d } = board(40, 78, { steerChats: async () => ({ dose: 0.4, turns }) });
	await tick();
	d.sel = 3;
	d.render(104);
	d.handleInput("6");
	d.render(104);
	await tick();
	let lines = text(d.render(104));
	const at = lines.findIndex((l) => l.startsWith("EXAMPLES"));
	assert.match(lines[at], /^EXAMPLES {2}A casual chat · turn 1-\d of 6 · 1\/7 · n next · N back$/);
	const shown = lines.filter((l) => /^turn \d /.test(l)).map((l) => Number(l.split(" ")[1]));
	assert.ok(shown.length >= 3 && shown[0] === 1, `turns ${shown}`);
	shown.forEach((n, k) => assert.equal(n, k + 1, "consecutive"));
	// Each turn whole: every reply line of the turns before the last drawn.
	for (const n of shown.slice(0, -1)) assert.ok(lines.some((l) => /│ Line 3 of a longer reply\./.test(l)), `turn ${n}`);
	assert.match(lines.at(-3)!, /^… \d+ more lines? \(page keys scroll\)$/, "the cut marked");
	assert.ok(!lines.some((l) => /Dinner/.test(l)), "only this chat's turns");
	d.handleInput("n");
	lines = text(d.render(104));
	assert.match(lines.find((l) => l.startsWith("EXAMPLES"))!, /turn 2-\d of 6 · 2\/7/);
	d.handleInput("\x1b[6~");                                                         // page down
	lines = text(d.render(104));
	assert.match(lines.find((l) => l.startsWith("EXAMPLES"))!, /turn [3-6](-\d)? of 6/);
	assert.ok(lines.some((l) => / · 10 above \(page keys scroll\)$/.test(l)));
});


// A token's top 10 as the stream sends it now: the written token and nine others (or ten when it is not among them).
const ten = (written: number | null, others = 9): any => {
	const alts = Array.from({ length: others }, (_, k) => [` c${k}`, +(0.3 / (k + 1)).toFixed(4)]);
	return { i: 5, id: 5, text: " me", p: written, entropy: 1.25, alts, n_layers: 4 };
};

test("TOKENS: the top 10 likeliest first, the written token marked where it ranks", () => {
	const t = ten(0.12);                                               // c0 30%, c1 15%, then " me" 12%
	assert.deepEqual(topChoices(t).rows.map(([w]) => w), [" c0", " c1", " me", " c2", " c3", " c4", " c5", " c6", " c7", " c8"]);
	assert.equal(topChoices(t).rank, 3);
	const lines = text(tokensPanel(piTheme, t, 5, 50, false));
	assert.equal(lines[0], 'TOKENS  #5 " me" · 12% · rank 3 · entropy 1.25');
	assert.equal(lines.length, 1 + TOP_N);
	assert.match(lines[1], /^ {2}" c0" +█+▏? +30%$/);
	assert.match(lines[3], /^▸ " me" +█+.* +12%$/);
	assert.ok(lines.every((l) => visibleWidth(l) <= 50));
	// The top choice: no rank said.
	assert.equal(text(tokensPanel(piTheme, ten(0.5), 5, 50, false))[0], 'TOKENS  #5 " me" · 50% · entropy 1.25');
});

test("TOKENS: a written token outside the top 10 is an eleventh row; fewer than 10 sent, fewer rows and a line saying so", () => {
	const out = text(tokensPanel(piTheme, ten(null, 10), 5, 50, false));
	assert.match(out[0], /^TOKENS {2}#5 " me" · not in the top 10 · entropy/);   // cut at 50 columns, the rank kept
	assert.equal(out.length, 1 + TOP_N + 1);
	assert.match(out[11], /^▸ " me" +<top-10$/);
	assert.ok(!out.slice(1, 11).some((l) => l.startsWith("▸")));
	const low = ten(0.001, 10);                                        // known, but under the ten
	assert.match(text(tokensPanel(piTheme, low, 5, 50, false))[11], /^▸ " me" +▏? *0\.1%$/);
	// Five runners-up sent, not ten: six rows and the line.
	const old = text(tokensPanel(piTheme, ten(0.2, 5), 5, 50, false));
	assert.equal(old.length, 1 + 6 + 1);
	assert.equal(old[7], "  the stream sent 6 choices for this token");
	// Control characters and wide ones stay one cell each way.
	const odd = text(tokensPanel(piTheme, { ...ten(0.2), text: "\x1b[31m\t日本\n" }, 5, 50, false));
	assert.ok(odd.every((l) => visibleWidth(l) <= 50 && !l.includes("\x1b")), odd.join("\n"));
	assert.match(odd[0], /"·\[31m 日本↵"/);
});

test("board: exact widths from 100 to 200 columns, focus on TOKENS, inspecting and full size", async () => {
	const { d, host } = board();
	await tick();
	d.sel = 22;
	host.panel = "TOKENS";
	for (const w of [100, 101, 104, 123, 150, 177, 200]) {
		for (const full of [false, true]) {
			d.full = full;
			const raw = d.render(w);
			assert.equal(raw.length, 67);
			for (const l of raw) assert.equal(visibleWidth(l), w, `${w} ${full}: ${plain(l)}`);
		}
		d.full = false;
		const lines = text(d.render(w));
		for (const p of ["SETTLING", "MAP"]) assert.ok(lines.some((l) => l.startsWith(`${p}  `)), `${w}: ${p}`);   // SETTLING: in LENS
		for (const p of ["EXPERTS", "TOKENS"]) assert.ok(lines.some((l) => new RegExp(`│ {1,2}${p} `).test(l)), `${w}: ${p}`);
	}
});

// 12 MoE layers, top 2 of 64: expert (7j + i) % 64 at slot j, the first of each layer with most of the mix.
const TWELVE = { experts: Array.from({ length: 24 }, (_, j) => (7 * j) % 64), expert_w: Array.from({ length: 24 }, (_, j) => (j % 2 ? 55 : 200)) };
// The same with an expert past 256 at the last layer: 512 experts, grids 32 by 16 (8 lines), five to a window at 51 lines.
const TWELVE_BIG = { ...TWELVE, experts: TWELVE.experts.map((e, j) => (j === 23 ? 500 : e)) };
const TWELVE_HUGE = { ...TWELVE, experts: TWELVE.experts.map((e, j) => (j === 23 ? 1000 : e)) };   // 1,024 experts: 12 lines a layer

test("EXPERTS full size at 51 lines: six layers, two whole rows of three; the band under them, the lists side by side", async () => {
	const { d, host } = board(40, ROWS_51, {}, TWELVE_BIG);
	await tick();
	d.sel = 20;
	host.panel = "EXPERTS";
	d.render(104);
	d.handleInput("4");
	const raw = d.render(104), lines = text(raw);
	assert.ok(raw.every((l) => visibleWidth(l) === 104), "every line the dock's width");
	assert.ok(lines.some((l) => /^layers 7-12 of 12 · 512 experts · 2 mixed for each token +token #20 " w20"$/.test(l)));   // layer 7 in focus
	const r1 = lines.findIndex((l) => /^▸ layer 7 +layer 8 +layer 9$/.test(l));
	assert.ok(r1 > 0, "the first row: three across, the focus layer marked");
	assert.equal(lines[r1].indexOf("layer 8"), 36, "34 columns apart: a grid and a gap");
	assert.equal(lines[r1].indexOf("layer 9"), 70);
	const r2 = lines.findIndex((l) => /^ {2}layer 10 +layer 11 +layer 12$/.test(l));
	assert.equal(r2, r1 + 9, "the second row under the first's grid, whole");
	const b = lines.findIndex((l) => new RegExp(`^${LABELS.forToken('" w20"')} +${LABELS.source.all} +${LABELS.source.session}$`).test(l));
	assert.equal(b, r2 + 10, "the band under the grids, after a blank");
	assert.equal(lines[b - 1], "");
	assert.match(lines[b + 1], /^ {2}#20 +█+▋? +78%$/);             // 200 / 255 of the mix at layer 7's first slot
	assert.match(lines.at(-1)!, /^tab map · ←\/→ token · f live · ↑\/↓ layer · enter open · 1-4 6 panel · \? explain · esc back$/);
});

test("EXPERTS full size, taller: every whole grid row that fits over the band; [ ] a layer, the window following", async () => {
	const { d, host } = board(40, 60, {}, TWELVE_BIG);
	await tick();
	d.sel = 20;
	host.panel = "EXPERTS";
	d.render(104);
	d.handleInput("4");
	let lines = text(d.render(104));
	assert.ok(lines.some((l) => /^layers 1-9 of 12 · /.test(l)), "nine: three whole rows");
	assert.equal(lines.filter((l) => /^(▸| ) layer \d+ +(▸| ) layer \d+ +(▸| ) layer \d+$/.test(l)).length, 3);
	d.handleInput("]");
	assert.equal(host.expertLayer, 7, "on by one");
	host.expertLayer = 11;
	assert.ok(text(d.render(104)).some((l) => /^layers 4-12 of 12 · /.test(l)), "the last window ends at the last layer, full");
	const tall = board(40, 160, {}, TWELVE_BIG);
	await tick();
	tall.host.panel = "EXPERTS";
	tall.d.full = true;
	lines = text(tall.d.render(104));
	assert.ok(lines.some((l) => /^layers 1-12 of 12 · /.test(l)), "all of them");
});

test("EXPERTS full size: the band on the same lines whatever the reply's length or the layer in focus", async () => {
	const at: number[] = [];
	for (const [n, layer] of [[30, 0], [40, 6], [60, 11]]) {
		const { d, host } = board(n, 60, {}, TWELVE_BIG);
		await tick();
		host.panel = "EXPERTS";
		host.expertLayer = layer;
		d.full = true;
		const lines = text(d.render(104));
		at.push(lines.findIndex((l) => l.startsWith("EXPERTS FOR ")), lines.findIndex((l) => /^lit|^bright/.test(l)));
	}
	assert.ok(at[0] > 0);
	assert.deepEqual(at, [at[0], at[1], at[0], at[1], at[0], at[1]], `${at}`);
});

test("EXPERTS full size: the window holds the focus layer (six at 51 lines; the last ends at the last layer); [ ] and ↑/↓ one layer, ←/→ the token", async () => {
	assert.deepEqual(layerWindow(12, 6), [5, 10]);
	assert.deepEqual(layerWindow(12, 11), [7, 12], "the last window ends at the last layer, full");
	assert.deepEqual(layerWindow(48, 47), [43, 48]);
	assert.deepEqual(layerWindow(3, 2), [0, 3]);
	assert.deepEqual(layerWindow(40, 20, 12), [12, 24]);
	assert.deepEqual(layerWindow(40, 39, 12), [28, 40]);
	assert.deepEqual(layerWindow(40, 5, 40), [0, 40]);
	assert.equal(LAYER_WINDOW, 5);
	const { d, host } = board(40, ROWS_51, {}, TWELVE_BIG);
	await tick();
	d.sel = 20;
	host.panel = "EXPERTS";
	d.render(104);
	d.handleInput("]");
	assert.equal(host.expertLayer, 7, "the board: one layer");
	d.handleInput("4");
	d.handleInput("]");
	d.handleInput("]");
	d.handleInput("]");
	assert.equal(host.expertLayer, 10, "full size: one layer too");
	assert.ok(text(d.render(104)).some((l) => /^layers 7-12 of 12 /.test(l)), "the window follows the focus layer");
	d.handleInput("]");
	d.handleInput("]");
	assert.equal(host.expertLayer, 11, "clamped to the last");
	for (let i = 0; i < 6; i++) d.handleInput("[");
	assert.equal(host.expertLayer, 5);
	assert.ok(text(d.render(104)).some((l) => /^▸ layer 6$/.test(l.slice(68))), "the second row's third");
	d.handleInput(KEY.up);
	assert.equal(host.expertLayer, 4, "↑ the layer before");
	d.handleInput(KEY.down);
	assert.equal(host.expertLayer, 5, "↓ the next");
	assert.equal(d.sel, 20, "the layer keys leave the token");
	d.handleInput(KEY.left);
	assert.equal(d.sel, 19, "← the token before");
});

test("EXPERTS full size: one glow a layer shown, one a layer at most; the token keys still move the token", async () => {
	const { d, host } = board(40, ROWS_51, {}, TWELVE_BIG, false);   // streaming: the glow
	await tick();
	host.panel = "EXPERTS";
	d.full = true;
	const dk: any = d;
	d.render(104);                                                        // following: the glows
	assert.equal(dk.experts.glows.size, 6);
	for (const l of [0, 5, 10, 3, 8]) { host.expertLayer = l; d.render(104); }
	assert.ok(dk.experts.glows.size <= 12, `${dk.experts.glows.size} glows kept`);                     // a tall dock draws every layer
});

test("EXPERTS full size, drawn alone: exact widths 40 to 200, never over the rows given, fewer across where narrower", () => {
	const th = { fg: (_c: string, s: string) => s, bold: (s: string) => s };
	const E = 512, layers = [20, 21, 22, 23, 24].map((layer) => ({ layer, cols: 32, rows: 16, cells: Array.from({ length: E }, (_, e) => (e % 7) / 7) }));
	const v: any = { layer: 24, L: 48, E, k: 10, cells: [], cols: 32, rows: 16, top: [], label: "" };
	const mine: [number, number][] = Array.from({ length: 10 }, (_, j) => [j * 40, 0.3 - j * 0.02]);
	for (let w = 40; w <= 200; w++) {
		for (const rows of [6, 12, 20, 40]) {
			const out = expertsLayers(th, v, layers, palette(th), w, mine, 7, "word", true, rows);
			assert.ok(out.length <= rows, `${w}x${rows}: ${out.length} lines`);
			for (const l of out) assert.equal(visibleWidth(l), w, `${w}x${rows}: ${JSON.stringify(plain(l))}`);
		}
	}
	const at = (w: number, rows = 40) => expertsLayers(th, v, layers, palette(th), w, mine, 7, "word", true, rows).map((l) => plain(l).trimEnd());
	assert.ok(at(101).some((l) => /^ {2}layer 21 +layer 22 +layer 23$/.test(l)), "three across at 101");
	assert.ok(at(70).some((l) => /^ {2}layer 21 +layer 22$/.test(l)), "two across at 70");
	const one = at(40, 60);
	assert.ok(one.some((l) => l === "  layer 21") && !one.some((l) => /layer 22 +layer/.test(l)), "one across at 40");
	const w8 = one.findIndex((l) => l.startsWith("EXPERTS FOR "));
	assert.ok(w8 > 0 && /^ {2}#0 /.test(one[w8 + 1]) && one[w8 + 10] !== "", `narrow: EXPERTS FOR across the width, ten rows (${w8})`);
	const short = at(101, 12);
	assert.match(short[0], /^layers 21-23 of 48 /, "too short: one row of grids, the band cut");
	assert.match(short.at(-1)!, /^EXPERTS FOR /);
	const few = expertsLayers(th, v, layers.slice(0, 2), palette(th), 101, mine, 7, "word", true, 40).map((l) => plain(l).trimEnd());
	assert.match(few[0], /^layers 21-22 of 48 /);
	assert.ok(few.some((l) => /^ {2}layer 21 +layer 22$/.test(l)), "two layers: one row");
	const six = [18, 19, 20, 21, 22, 23].map((layer) => ({ ...layers[0], layer }));
	for (const focus of [18, 20, 23]) {
		const rows = expertsLayers(th, { ...v, layer: focus }, six, palette(th), 101, mine.slice(0, focus - 17), focus, "word", true, 40).map((l) => plain(l));
		assert.equal(rows.findIndex((l) => /^EXPERTS FOR /.test(l)), 20, `the band fixed, focus ${focus + 1}`);
		const heads = rows.filter((l) => /^(▸| ) layer/.test(l));
		assert.ok(heads.every((l) => (l.match(/layer \d+/g) ?? []).length === 3), "whole rows of three");
	}
});

test("board: the token that went into the lens's pass is on violet in the reply, beside the inverse selection; LENS's bottom names it", async () => {
	const IN = "\x1b[48;2;60;46;98m", BG = "\x1b[48;2;11;13;16m";
	const { d, host } = board();
	await tick();
	d.sel = 20;
	const raw = d.render(104), lines = text(raw);
	const strip = raw.find((l) => l.includes(IN))!;
	assert.ok(strip, "the input token marked");
	assert.match(plain(strip.slice(strip.indexOf(IN))), /^ w19/);
	assert.ok(strip.includes("\x1b[7m w20\x1b[27m"), "the selection still inverse");
	assert.ok(strip.slice(strip.indexOf(IN)).includes("\x1b[49m" + BG), "the dock's background back after it (tint)");
	assert.ok(lines.some((l) => /╰─+ in: "w19" ─+╯/.test(l)), "the telescope's bottom");
	d.sel = 0;
	assert.ok(text(d.render(104)).some((l) => /in: end of prompt/.test(l)));
	host.panel = "LENS";
	d.sel = 20;
	d.render(104);
	d.handleInput("2");
	const both = text(d.render(104)).filter((l) => /in: "w19"/.test(l));
	assert.equal(both.length, 1);
	assert.match(both[0], /in: "w19".*│.*in: "w19"/, "both lenses full size name it");
});

// -- the layout and the mini dock (m) ---------------------------------------------------------------------------------

test("board: at 100-200 columns and 49-80 lines it fills the dock exactly, every panel titled; a model without experts: no EXPERTS", async () => {
	for (const rows of [49, 51, 67, 80]) for (const dense of [false, true]) {
		const { d, live } = board(40, rows, {}, TWELVE);
		await tick();
		if (dense) live.dense = MODEL;
		for (const sel of [20, -1]) {
			d.sel = sel;
			for (const width of sel < 0 ? [101] : [100, 133, 200]) {
				const raw = d.render(width);
				assert.equal(raw.length, rows, `${width}x${rows}`);
				for (const l of raw) assert.equal(visibleWidth(l), width, `${width}x${rows}: ${JSON.stringify(plain(l))}`);
				const lines = text(raw);
				assert.ok(!lines.some((l) => /^the dock: /.test(l)), String(lines.find((l) => /the dock: /.test(l))));
				const has = (p: string) => lines.some((l) => new RegExp(`(^ ?| │ {1,2})${p}( |$)`).test(l));   // a title (inverse in focus: padded)
				for (const p of ["MAP", "LENS", "TOKENS"]) assert.ok(has(p), `${width}x${rows}: ${p}`);
				assert.equal(has("EXPERTS"), !dense, `${width}x${rows} dense ${dense}: EXPERTS`);
			}
		}
	}
});

test("board: [ ] move EXPERTS' layers by one", async () => {
	const { d, host } = board(40, ROWS_51 + 6, {}, TWELVE_BIG);                   // room for four layers of 512, no more
	await tick();
	d.sel = 20;
	host.panel = "EXPERTS";
	const shown = () => text(d.render(104)).map((l) => /│ [▸ ] (layer \d+ of 12)$/.exec(l)?.[1]).filter(Boolean);
	assert.deepEqual(shown(), ["layer 3 of 12", "layer 5 of 12", "layer 7 of 12", "layer 9 of 12"], "four, a fifth of the model apart");
	// The right column from EXPERTS' title to the key line's rule: four layers, a heading over 6 lines of grid each.
	const right = text(d.render(104)).map((l) => l.slice(columns(101).lw + 3));
	const top = right.findIndex((l) => /^ ?EXPERTS /.test(l)), end = right.findIndex((l, i) => i > top && /^─+$/.test(l));
	const grids = right.slice(top + 1, end);
	while (grids.length && !grids.at(-1)!.trim()) grids.pop();                       // the column's spare lines, under its last panel
	assert.deepEqual(grids.flatMap((l, i) => (/layer \d+ of 12$/.test(l) ? [i] : [])), [0, 7, 14, 21], "each under its heading, no blank lines");
	assert.equal(grids.length, 28, "four grids");
	assert.ok(grids.every((l) => !/%/.test(l)), "the lights only");
	d.handleInput("]");
	assert.equal(host.expertLayer, 7);
	assert.deepEqual(shown(), ["layer 4 of 12", "layer 6 of 12", "layer 8 of 12", "layer 10 of 12"]);
	assert.match(text(d.render(104)).at(-1)!, /· \[ \] layers · /);
});

test("mini dock: lights only, as many layers as fit under EXPERTS' title, each headed as on the board; one key line; [ ] page; m back", async () => {
	const changed: number[] = [];
	const { d, host } = board(40, 60, { mini: true, layoutChanged: () => changed.push(1) }, TWELVE);
	await tick();
	let raw = d.render(MINI_WIDTH), lines = text(raw);
	assert.equal(raw.length, 60);
	assert.ok(raw.every((l) => visibleWidth(l) === MINI_WIDTH));
	const labels = () => miniL(text(d.render(MINI_WIDTH)));
	const all = Array.from({ length: 12 }, (_, l) => `L${l + 1}`);
	// Tall enough: every layer, no page cap.
	assert.deepEqual(miniL(lines), all);
	assert.equal(lines[1], "EXPERTS  12 layers · 64 experts · 2 a token", "EXPERTS' title, as the board's");
	assert.deepEqual(lines.filter((l) => /layer \d+ of 12$/.test(l)).slice(5, 8), ["  layer 6 of 12", "▸ layer 7 of 12", "  layer 8 of 12"], "the board's headings, ▸ its focus layer");
	assert.ok(!lines.some((l) => /REPLY|TOKENS|STEER|metalscope/.test(l)), "no reply, no text panels");
	assert.equal(lines.at(-1), "ctrl+\\ steps in, then m · [ ]");                  // its keys go to Pi's editor until then
	assert.match(lines[0], /^#39 " w39" /, "the token line, following: the newest token");
	d.stepIn(-1);                                                                     // stepped in: the keys reach it
	assert.equal(text(d.render(MINI_WIDTH)).at(-1), "m full · [ ] more · ctrl+\\ closes");
	assert.match(text(d.render(MINI_WIDTH))[0], /^#39 " w39" /, "following: the newest token");
	d.sel = 20;                                                                       // a token stepped to: end follows again
	assert.equal(text(d.render(MINI_WIDTH)).at(-1), "f live · m full · [ ] more");
	assert.match(text(d.render(MINI_WIDTH))[0], /^#20 " w20" 30%$/, "the token whose experts are lit");
	d.handleInput("[");
	assert.deepEqual(labels(), all, "every layer drawn: [ ] keep them");
	// Shorter: a page of the grids that fit, [ ] by that many.
	(d as any).tui.terminal.rows = 30;
	host.expertLayer = undefined;
	const n = labels().length, mid = 6;                                               // EXPERTS' layer: the middle (7)
	assert.ok(n > 1 && n < 12, String(n));
	assert.equal(labels()[0], `L${Math.min(mid, 12 - n) + 1}`, "from the EXPERTS layer, a full page");
	d.handleInput("[");
	assert.deepEqual(labels(), all.slice(0, n), `back by the ${n} drawn`);
	d.handleInput("]");
	assert.deepEqual(labels()[0], `L${12 - n + 1}`, `on by ${n}, the page kept full`);
	d.handleInput("]");
	assert.deepEqual(labels()[0], `L${12 - n + 1}`);
	(d as any).tui.terminal.rows = 60;
	d.handleInput("\x1b[C");
	assert.equal(d.sel, 21, "←/→ a token: its experts lit");
	d.handleInput("m");
	assert.equal(host.mini, false);
	assert.deepEqual(changed, [1]);
	assert.ok(text(d.render(104)).some((l) => l.startsWith("◆ metalscope")));
	d.handleInput("m");                                                               // from the board too
	assert.equal(host.mini, true);
	// Not stepped in, a token key that reaches it (a focused dock after esc): from the newest token, never a throw;
	// forward, it keeps following the newest (-1), back, it steps to a token.
	for (const [k, want] of [["L", -1], ["\x1b[1;3C", -1], ["H", "back"], ["\x1b[C", -1], ["\x1b[D", 38]] as const) {
		const m = board(40, 60, { mini: true }, TWELVE);
		await tick();
		m.d.render(MINI_WIDTH);
		assert.doesNotThrow(() => m.d.handleInput(k));
		if (want === "back") assert.ok(m.d.sel >= 0 && m.d.sel < 39, `${JSON.stringify(k)}: ${m.d.sel}`);
		else assert.equal(m.d.sel, want, JSON.stringify(k));
		assert.ok(!text(m.d.render(MINI_WIDTH)).some((l) => /the dock: /.test(l)));
	}
	// Under 23 rows: m would leave a focused dock that cannot be drawn; refused, said where the key line is.
	const narrow = board(40, 20, { mini: true }, TWELVE);
	await tick();
	narrow.d.sel = 3;
	narrow.d.render(MINI_WIDTH);
	narrow.d.handleInput("m");
	assert.equal(narrow.host.mini, true);
	assert.equal(text(narrow.d.render(MINI_WIDTH)).at(-1), "full dock needs 23 rows, has 20");
	narrow.d.handleInput("\x1b[C");
	assert.match(text(narrow.d.render(MINI_WIDTH)).at(-1)!, /^f live · \[ \] more/, "the note goes with the next key; m full not offered");
	assert.doesNotMatch(text(narrow.d.render(MINI_WIDTH)).at(-1)!, /m full/);
	// Tall 512-expert grids: as many as the height holds.
	const big = board(40, 30, { mini: true });
	await tick();
	assert.ok(text(big.d.render(MINI_WIDTH)).filter((l) => /^[▸ ] layer \d+ of \d+$/.test(l)).length >= 1);
	// No reply yet, a dense model: one line says so.
	const none = board(0, 40, { mini: true });
	assert.match(text(none.d.render(MINI_WIDTH)).join(" "), /lights for the next traced reply/);
});

test("miniSplit: the maps take their want, else less (SETTLING before MAP); one grid kept; every grid that fits; the lines over to SETTLING, then MAP, up to their most, the rest left over", () => {
	const S: MiniRoom = [3, 5, 9];                                                   // SETTLING over 15 layers (settleRoom)
	assert.deepEqual(MINI_MAP, [2, 4, 4], "MAP: its title and a line at the least, 3 lines at the most");
	// Flash Next's 512-expert grids: 9 lines a layer. The body is the dock less its key line.
	assert.deepEqual(miniSplit(77, 9, 48, MINI_MAP, S), { n: 7, map: 4, settle: 9 }, "78 rows: 7 grids, SETTLING its most, a line over");
	assert.deepEqual(miniSplit(39, 9, 48, MINI_MAP, S), { n: 3, map: 4, settle: 8 }, "40 rows: 3 grids, SETTLING the 3 lines left");
	assert.deepEqual(miniSplit(31, 7, 48, MINI_MAP, [3, 5, 7]), { n: 3, map: 4, settle: 6 }, "every grid that fits, SETTLING short of its most, nothing over");
	assert.deepEqual(miniSplit(39, 9, 48, MINI_MAP), { n: 3, map: 4, settle: 0 }, "no SETTLING: 3 grids, 8 lines over");
	assert.deepEqual(miniSplit(30, 9, 48, [2, 2, 4], [3, 3, 5]), { n: 2, map: 4, settle: 5 }, "SETTLING first, then MAP, up to their most; 3 left over");
	// Every grid that fits: under a grid's worth left over, and only with the maps at their most, never past it.
	for (let rows = 20; rows <= 120; rows++) {
		const x = miniSplit(rows, 9, 48, MINI_MAP, S), over = rows - x.n * 9 - x.map - x.settle;
		assert.ok(x.n >= 1 && over >= 0 && over < 9 && x.map <= 4 && x.settle <= 9, `${rows}: ${JSON.stringify(x)}`);
		if (over) assert.deepEqual([x.map, x.settle], [4, 9], `${rows}: lines over only with the maps at their most`);
		const y = miniSplit(rows, 9, 48, MINI_MAP), alone = rows - y.n * 9 - y.map;
		assert.ok(alone >= 0 && alone < 9 && y.map === 4, `${rows}, no SETTLING: under a grid's worth over`);
	}
	assert.deepEqual(miniSplit(19, 9, 48, MINI_MAP, S), { n: 1, map: 4, settle: 6 }, "20 rows");
	assert.deepEqual(miniSplit(15, 9, 48, MINI_MAP, S), { n: 1, map: 2, settle: 4 }, "both at their least");
	assert.deepEqual(miniSplit(13, 9, 48, MINI_MAP, S), { n: 1, map: 4, settle: 0 }, "SETTLING goes first");
	assert.deepEqual(miniSplit(11, 9, 48, MINI_MAP, S), { n: 1, map: 2, settle: 0 });
	assert.deepEqual(miniSplit(10, 9, 48, MINI_MAP, S), { n: 1, map: 0, settle: 0 }, "then MAP: the grid stays");
	assert.deepEqual(miniSplit(5, 9, 48, MINI_MAP, S), { n: 1, map: 0, settle: 0 }, "shorter than a grid");
	assert.deepEqual(miniSplit(39, 5, 2, MINI_MAP, S), { n: 2, map: 4, settle: 9 }, "every layer's grid fits: all drawn, the maps their most");
	assert.deepEqual(miniSplit(30, 5, 0, MINI_MAP, S), { n: 0, map: 21, settle: 9 }, "no grids: MAP the rest past its most");
});

test("mini dock: MAP and SETTLING under the lights, the board's own drawn the right column's width (50); stepped, the token marked in both", async () => {
	const { d, live } = board(40, 38, { mini: true }, TWELVE);
	await tick();
	const th = dockTheme(piTheme), framed = (ls: string[]) => ls.map((l) => tint(th, th.fg("dim", "▌") + " " + l, MINI_WIDTH));   // sel alone: the keyboard in the chat
	for (const sel of [-1, 20]) {
		d.sel = sel;
		const raw = d.render(MINI_WIDTH), lines = text(raw), toks = live.reply!.tokens;
		assert.equal(raw.length, 38);
		assert.ok(raw.every((l) => visibleWidth(l) === MINI_WIDTH));
		const map = lines.findIndex((l) => /^MAP {2}40 tok · /.test(l)), settle = lines.indexOf("SETTLING  J-lens · old → now");
		assert.ok(lines.map((l) => /^[▸ ] layer \d+ of \d+$/.test(l)).lastIndexOf(true) < map && map < settle, lines.join("\n"));
		assert.deepEqual(lines.filter((l) => /^[▸ ] layer \d+ of \d+$/.test(l)).length, 12, "every grid");
		assert.equal(settle + 7, 37, "SETTLING down to the key line: every layer read, two to a line");
		const at = sel < 0 ? -1 : sel, k = sel < 0 ? 39 : sel;
		assert.deepEqual(raw.slice(map, settle), framed(mapPanel(th, toks, at, RIGHT_W, settle - map, false, true, mapWindow(40, RIGHT_W, settle - map - 1, at))), `MAP ${sel}`);
		assert.deepEqual(raw.slice(settle, 37), framed(settlingPanel(th, toks, k, undefined, RIGHT_W, 7, palette(th), sel >= 0)), `SETTLING ${sel}`);
		const focus = halfRows(raw.slice(map + 1, settle)).flat().some((c) => c.join() === COLOURS.text.join());
		assert.equal(focus, sel >= 0, "the token stepped to: white in MAP");
	}
});

test("mini dock: no lens read, no SETTLING; no routing, the maps under the reason", async () => {
	const { d } = board(40, 60, { mini: true }, { ...TWELVE, lens: [], lens_logit: [] });
	await tick();
	let lines = text(d.render(MINI_WIDTH));
	assert.ok(lines.some((l) => /^MAP {2}40 tok/.test(l)) && !lines.some((l) => /^SETTLING/.test(l)), "MAP only");
	// A reply without routing: the reason where the lights would be, the maps under it.
	const dense = board(40, 40, { mini: true }, { experts: undefined, expert_w: undefined, expert_k: undefined });
	await tick();
	lines = text(dense.d.render(MINI_WIDTH));
	assert.ok(!/^(MAP|SETTLING|L\d)/.test(lines[0]) && lines[0].length > 0, lines[0]);
	const mapAt = lines.findIndex((l) => /^MAP /.test(l));
	assert.ok(mapAt > 0 && mapAt <= 4 && lines.slice(0, mapAt - 1).every((l) => l.length > 0) && lines[mapAt - 1] === "",
		"the maps under the reason, a blank line between: " + lines.join("\n"));
	assert.ok(lines.findIndex((l) => /^SETTLING /.test(l)) > mapAt);
});

test("mini dock, short: the maps drop out before the one grid of lights (SETTLING first)", async () => {
	const shown = async (rows: number, stepped = false) => {
		const { d } = board(40, rows, { mini: true }, TWELVE_HUGE);                   // 1,024 experts: 12 lines a layer
		await tick();
		if (stepped) d.stepIn(-1);                                                       // the token's line on top
		const raw = d.render(MINI_WIDTH), lines = text(raw);
		assert.equal(raw.length, rows);
		assert.ok(raw.every((l) => visibleWidth(l) === MINI_WIDTH));
		return [lines.filter((l) => /^[▸ ] layer \d+ of \d+$/.test(l)).length, lines.some((l) => /^MAP /.test(l)), lines.some((l) => /^SETTLING /.test(l))];
	};
	assert.deepEqual(await shown(16), [1, false, false], "the token's line, EXPERTS' title, the grid alone");
	assert.deepEqual(await shown(12), [1, false, false], "the dock's least: the grid cut, still alone");
	assert.deepEqual(await shown(16, true), [1, false, false], "stepped in: the same");
	assert.deepEqual(await shown(17), [1, true, false], "MAP's title and a line");
	assert.deepEqual(await shown(19), [1, true, false], "SETTLING gone first");
	assert.deepEqual(await shown(20), [1, true, true]);
	assert.deepEqual(await shown(60), [4, true, true], "4 grids fit: all 4");
});

test("mini dock while a reply streams: the maps follow the newest token; no end needed", async () => {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r", model: MODEL });
	for (let i = 0; i < 30; i++) onEvent(live, tok(i, TWELVE as any));
	const d = new Dock({ terminal: { rows: 40 }, requestRender() {} }, piTheme, { live, mini: true, close() {}, unfocus() {}, fetchJson: async () => null } as any);
	await tick();
	let lines = text(d.render(MINI_WIDTH));
	assert.ok(lines.some((l) => /^MAP {2}30 tok/.test(l)) && lines.includes("SETTLING  J-lens · old → now"), lines.join("\n"));
	for (let i = 30; i < 600; i++) onEvent(live, tok(i, TWELVE as any));
	lines = text(d.render(MINI_WIDTH));
	assert.ok(lines.some((l) => /^MAP {2}600 tok · .* \d+-600$/.test(l)), String(lines.find((l) => /^MAP/.test(l))));   // the newest lines
	assert.ok(!lines.some((l) => /the dock: /.test(l)));
});

test("mini dock while a reply streams: the grids, the height and the key line hold token to token from the first", async () => {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r", model: MODEL });
	onEvent(live, tok(0, TWELVE_BIG as any));
	const d = new Dock({ terminal: { rows: 52 }, requestRender() {} }, piTheme, { live, mini: true, close() {}, unfocus() {}, fetchJson: async () => null } as any);
	d.render(MINI_WIDTH);
	await tick();
	const grids = new Set<number>(), maps = new Set<number>(), keys = new Set<number>();
	for (let i = 1; i < 400; i++) {
		onEvent(live, tok(i, TWELVE_BIG as any));
		const raw = d.render(MINI_WIDTH), lines = text(raw);
		assert.equal(raw.length, 52);
		grids.add(lines.filter((l) => /^[▸ ] layer \d+ of \d+$/.test(l)).length);
		maps.add(lines.findIndex((l) => /^MAP /.test(l)));
		keys.add(lines.findIndex((l) => /ctrl\+\\ steps in/.test(l)));
	}
	assert.deepEqual([...grids], [5], "5 grids fit, SETTLING at its most: all 5, from the first token");
	assert.equal(maps.size, 1, "MAP where it was");
	assert.equal(keys.size, 1, "the key line where it was");
});

test("dock: keys and drawing never throw, whatever the mode", async () => {
	const { d } = board(40, 67, {}, TWELVE);
	await tick();
	d.sel = 10;
	const keys = ["1", "2", "3", "4", "5", "6", "7", "m", "[", "]", "z", "\t", "\x1b[Z", "\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D", "?", "u", "U", "s", "x", "c", "n", "N", "\r", "p"];
	let seed = 7;
	const rnd = (n: number) => ((seed = (seed * 1103515245 + 12345) % 2 ** 31), seed % n);
	for (let i = 0; i < 600; i++) {
		const k = keys[rnd(keys.length)];
		assert.doesNotThrow(() => d.handleInput(k));
		if (d.sel < 0) d.sel = rnd(40);
		const width = [36, 60, 100, 104, 150, 200][rnd(6)];
		const raw = d.render(width);
		assert.equal(raw.length, 67);
		assert.ok(raw.every((l) => visibleWidth(l) === width), `${width} after ${k}`);
		assert.ok(!text(raw).some((l) => /^the dock: /.test(l)), `${k}: ${text(raw).find((l) => /the dock: /.test(l))}`);
	}
});

test("mini dock, a token before the first lens reading: SETTLING keeps its room (the board's words there), MAP stays put", async () => {
	const { d, live } = board(40, 60, { mini: true }, TWELVE);
	await tick();
	for (const t of live.reply!.tokens.slice(0, 10)) Object.assign(t, { lens: [], lens_logit: [] });
	const places = () => {
		const lines = text(d.render(MINI_WIDTH));
		return { map: lines.findIndex((l) => /^MAP /.test(l)), settle: lines.indexOf("SETTLING  J-lens · old → now"), lines };
	};
	const follow = places();
	d.sel = 3;
	const at3 = places();
	assert.deepEqual([at3.map, at3.settle], [follow.map, follow.settle], at3.lines.join("\n"));
	assert.equal(at3.lines[at3.settle + 1], "no lens read on these tokens", "as the board's SETTLING draws it, its shortest form");
	assert.ok(at3.settle > 0);
});

test("mini dock: [ ] page the lights with the maps shown, the maps where they were; a long reply stepped far back, MAP on that token", async () => {
	const { d } = board(40, 40, { mini: true }, TWELVE_BIG);                          // 512 experts: four grids over the maps
	await tick();
	d.stepIn(-1);
	const view = () => {
		const lines = text(d.render(MINI_WIDTH));
		return { labels: lines.filter((l) => /^[▸ ] layer \d+ of \d+$/.test(l)).map((l) => +/(\d+) of/.exec(l)![1]), map: lines.findIndex((l) => /^MAP /.test(l)) };
	};
	const was = view();
	assert.equal(was.labels.length, 4);
	d.handleInput("[");
	const back = view();
	assert.deepEqual(back.labels, was.labels.map((l) => l - 4), "back by the four drawn");
	assert.equal(back.map, was.map);
	d.handleInput("]");
	assert.deepEqual(view().labels, was.labels);
	// 2,000 tokens, stepped to #100: MAP's window holds it, marked white; the token named on top.
	const long = board(2000, 40, { mini: true }, TWELVE);
	await tick();
	long.d.stepIn(100);
	const raw = long.d.render(MINI_WIDTH), lines = text(raw);
	assert.match(lines[0], /^#100 /);
	const m = lines.findIndex((l) => /^MAP /.test(l)), s = lines.findIndex((l) => /^SETTLING /.test(l));
	const [, a, b] = /(\d+)-(\d+)$/.exec(lines[m])!.map(Number);
	assert.ok(a <= 101 && 101 <= b, lines[m]);
	assert.ok(halfRows(raw.slice(m + 1, s)).flat().some((c) => c.join() === COLOURS.text.join()), "the token stepped to: white in MAP");
});

test("widget: the stream asks for routing and the top 10s while the board or the mini dock is shown; asked again only on a change", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	let dock: any;
	const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {},
		custom: (make: any) => new Promise<void>(() => { dock = make({ requestRender() {}, terminal: { rows: 67, columns: 290 } }, {}, {}, () => {}); }) } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false } as any;
	let asks = 0;
	createUi({ live } as any, join(dir, "pi.json"), () => asks++).showDock(ctx);
	dock.render(104);
	await tick();
	const host = dock.host;
	assert.deepEqual([live.experts, live.lens], [true, true], "the board");
	const n = asks;
	host.layoutChanged();
	assert.equal(asks, n, "nothing changed: not asked again");
	host.mini = true;
	host.layoutChanged();
	assert.deepEqual([live.experts, live.lens], [true, true], "mini: routing, and the top 10s for SETTLING's faint level");
	host.mini = false;
	host.layoutChanged();
	assert.deepEqual([live.experts, live.lens], [true, true]);
});

test("widget: the mini dock is remembered (pi.json), per user; an older pi.json's `hidden` ignored, not written back; a re-created dock keeps the mode; the room follows", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	const path = join(dir, "pi.json");
	writePrefs({ hidden: ["TOKENS"] } as any, path);
	const root = { render: () => ["pi"], invalidate() {} };
	const mk = () => {
		const tui: any = { layoutRoot: root, terminal: { rows: 67, columns: 160 }, setLayoutRoot(r: any) { this.layoutRoot = r; }, requestRender() {} };
		let dock: any, close = () => {}, opts: any;
		const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {},
			custom: (make: any, o: any) => new Promise<void>((done) => { opts = o.overlayOptions; dock = make(tui, {}, {}, done); close = done; }) } };
		const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false,
			reply: { rid: "r", tokens: [{ i: 0, id: 0, text: " w", p: 0.9 }], done: true, prefill: 0 } } as any;
		const ui = createUi({ live } as any, path, () => {});
		ui.showDock(ctx);
		const gap =() => tui.layoutRoot === root ? 0 : tui.layoutRoot.entries[1].basis;
		return { ui, ctx, tui, dock: () => dock, opts: () => opts, close: () => close(), gap };
	};
	let a = mk();
	assert.equal(a.gap(), dockColumns(160));
	a.dock().render(104);
	assert.ok(text(a.dock().render(104)).some((l) => /│ TOKENS /.test(l)), "the old `hidden` ignored");
	a.dock().sel = 0;
	a.dock().handleInput("m");
	assert.deepEqual(readPrefs(path), { hidden: ["TOKENS"], dock: true, mini: true }, "left as it was, never written");
	assert.equal(a.gap(), MINI_WIDTH + 1, "the mini dock's room: chat gets the columns back");
	assert.deepEqual([a.opts().width, a.opts().minWidth, a.opts().visible(100), a.opts().visible(99)], [MINI_WIDTH, MINI_WIDTH, true, false]);
	a.close();
	await tick();
	a.ui.showDock(a.ctx);                                                           // the same session, the dock again
	assert.match(text(a.dock().render(MINI_WIDTH)).at(-1)!, /^ctrl\+\\ steps in, then m/, "still mini");
	a = mk();                                                                       // Pi restarted: from pi.json
	assert.equal(a.dock().host.mini, true);
	assert.equal(a.gap(), MINI_WIDTH + 1);
	a.dock().sel = 0;
	a.dock().handleInput("m");
	assert.equal(a.gap(), dockColumns(160));
	assert.deepEqual([a.opts().width, a.opts().minWidth, a.opts().visible(100), a.opts().visible(99)], ["36%", 44, true, false]);   // under 100: hidden
	assert.deepEqual(readPrefs(path), { hidden: ["TOKENS"], dock: true, mini: false });
});

test("TOKENS: the written token's rank among the top 10", () => {
	const full = ten(0.001);                                                       // nine sent and the written last
	assert.equal(topChoices(full).rank, 10);
	assert.equal(topChoices({ ...full, p: 0.5 }).rank, 1);
	assert.match(text(tokensPanel(piTheme, full, 3, 50, false))[0], /· rank 10 · /);
});

test("board: the violet input only for the token just before in the stream", async () => {
	const IN = "\x1b[48;2;60;46;98m";
	const { d, live } = board();
	await tick();
	live.reply!.tokens.forEach((t, k) => (t.i = k < 20 ? k : k + 1));              // a special token unpublished before #20
	d.sel = 20;
	let raw = d.render(104);
	assert.ok(!raw.some((l) => l.includes(IN)), "not adjacent: no pair");
	assert.ok(text(raw).some((l) => /in: not streamed/.test(l)));
	d.sel = 21;
	assert.ok(d.render(104).some((l) => l.includes(IN)));
});

test("widget: /scope dock full is refused where the terminal is too short for the full dock, the mini one kept, or opened", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	const tui: any = { requestRender() {}, terminal: { rows: 20, columns: 120 } };
	const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {}, custom: (make: any) => new Promise<void>(() => make(tui, {}, {}, () => {})) } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false } as any;
	const ui = createUi({ live } as any, join(dir, "pi.json"), () => {});
	assert.equal(ui.dockMode("mini", ctx), undefined);
	assert.match(ui.dockMode("full", ctx)!, /^the full dock needs 23 rows; this terminal has 20: the mini dock stays$/);
	assert.equal(readPrefs(join(dir, "pi.json")).mini, true);
	tui.terminal.rows = 60;
	assert.equal(ui.dockMode("full", ctx), undefined);
	assert.equal(readPrefs(join(dir, "pi.json")).mini, false);
	// The dock closed, the terminal short: the mini dock opens.
	let close = () => {};
	const ctx2 = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget: (_: string, f: any) => f?.(tui), setStatus() {},
		custom: (make: any) => new Promise<void>((done) => ((close = done), make(tui, {}, {}, done))) } };
	const ui2 = createUi({ live } as any, join(dir, "pi2.json"), () => {});
	ui2.updateUi(ctx2);
	ui2.showDock(ctx2);
	ui2.updateUi(ctx2);                                                              // the resize widget: the terminal known
	ui2.hideDock();
	close();
	await tick();
	assert.equal(ui2.docked(), false);
	for (const [cols, rows, need] of [[120, 20, "23 rows; this terminal has 20"]] as const) {
		Object.assign(tui.terminal, { columns: cols, rows });
		assert.equal(ui2.dockMode("full", ctx2), `the full dock needs ${need}: the mini dock opens`);
		assert.equal(ui2.docked(), true);
		ui2.hideDock();
		await tick();
	}
});

// -- the new panels: SETTLING (in LENS), MAP, CONCEPTS; EXPERTS' three layers; the layout -----------------------------------

test("arrange: the left column MAP, LENS, the right TOKENS, EXPERTS (a model with experts), STREAM (where it is drawn)", () => {
	assert.deepEqual(arrange(true, true), { left: ["MAP", "LENS"], right: ["TOKENS", "EXPERTS", "STREAM"] });
	assert.deepEqual(arrange(true, false), { left: ["MAP", "LENS"], right: ["TOKENS", "EXPERTS"] });
	assert.deepEqual(arrange(false, true), { left: ["MAP", "LENS"], right: ["TOKENS", "STREAM"] });
	assert.deepEqual(arrange(false, false), { left: ["MAP", "LENS"], right: ["TOKENS"] });
	assert.deepEqual(PANELS, ["MAP", "LENS", "TOKENS", "EXPERTS", "STREAM"], "tab order: MAP over LENS, then the right column top down");
});

test("lensSplit: the leasts first, the telescope whole, then SETTLING, a blank line between out of SETTLING's share; too short, SETTLING shrinks, the blank goes, then SETTLING; the telescope keeps its least", () => {
	const scope: [number, number] = [1 + MIN_LINES, 30], settle: [number, number] = [5, 25];
	assert.deepEqual(lensSplit(20, scope, settle), [15, 1, 4], "short of the telescope's most: SETTLING its least (the blank in it), the telescope the rest");
	assert.deepEqual(lensSplit(35, scope, settle), [30, 1, 4], "the telescope whole");
	assert.deepEqual(lensSplit(45, scope, settle), [30, 1, 14], "then SETTLING");
	assert.deepEqual(lensSplit(56, scope, settle), [30, 1, 25], "both whole and the blank");
	assert.deepEqual(lensSplit(80, scope, settle), [30, 1, 25], "the rest blank");
	assert.deepEqual(lensSplit(15, scope, settle), [10, 1, 4], "both at their least");
	assert.deepEqual(lensSplit(13, scope, settle), [10, 1, 2], "shorter: SETTLING shrinks first, to its sub-heading and one line");
	assert.deepEqual(lensSplit(12, scope, settle), [10, 0, 2], "then the blank goes, not SETTLING");
	assert.deepEqual(lensSplit(11, scope, settle), [11, 0, 0], "less than that: SETTLING goes, its line to the telescope");
	assert.deepEqual(lensSplit(10, scope, settle), [10, 0, 0]);
	assert.deepEqual(lensSplit(6, scope, settle), [6, 0, 0], "under its least, the telescope takes what there is");
	assert.deepEqual(lensSplit(11, [10, 10], settle), [10, 0, 0], "a line over a whole telescope: blank, not a lone sub-heading");
	assert.deepEqual(lensSplit(12, [10, 10], settle), [10, 0, 2]);
	assert.deepEqual(lensSplit(13, [10, 10], settle), [10, 1, 2]);
	for (let rows = 0; rows < 60; rows++) {
		const [t, g, s] = lensSplit(rows, scope, settle);
		assert.ok(t + g + s <= rows && (s === 0 || s >= 2) && (g === 0 || (g === 1 && s >= 2)) && (rows < scope[0] || t >= scope[0]), `${rows}: ${t} ${g} ${s}`);
		assert.equal(g, s && rows - t >= 3 ? 1 : 0, `${rows}: the blank whenever SETTLING keeps two lines with it`);
	}
});

test("board: LENS is one panel under MAP, SETTLING over the telescope (a violet sub-heading under LENS's title, a blank line, no rule, no title of its own)", async () => {
	const { d } = board(40, ROWS_51);
	await tick();
	d.sel = 20;
	let raw = d.render(104), lines = text(raw);
	if (process.env.SHOW_DOCK) console.log(lines.join("\n"));
	const lens = lines.findIndex((l) => /^ ?LENS /.test(l)), s = lines.findIndex((l) => l.startsWith("SETTLING  J-lens · old → now"));
	const map = lines.findIndex((l) => l.startsWith("MAP  "));
	const scope = lines.findIndex((l) => /^#20 "w20" · settles L40\/48/.test(l)), rule = lines.findIndex((l, i) => i > s && /^─+/.test(l));
	assert.ok(map > 0 && map < lens && s === lens + 1 && scope > s, `${map} ${lens} ${s} ${scope}`);
	assert.ok(lines.slice(lens + 1, rule).every((l) => !/^─+ │/.test(l)), "no rule between SETTLING and the telescope");
	assert.match(lines[lens - 1], /^─+ │/, "a rule between MAP and LENS");
	assert.equal(lens - map, 1 + 1 + MAP_LINES, "MAP its least at 51 lines: its title and two lines");
	assert.ok(rule - scope >= MIN_LINES, "the telescope keeps its least");
	const bar = lines[s].indexOf(" │");   // the left column's width
	assert.deepEqual(lines.slice(s + 1, s + 7).map((l) => l.slice(0, 4).trimEnd()), ["L44", "L36", "L28", "L20", "L12", "L4"],
		"SETTLING whole at 51 lines (the telescope's 12 layers read): two a line");
	assert.equal(scope, s + 8, "then a blank line, the telescope");
	assert.equal(lines[scope - 1].slice(0, bar).trim(), "", "a blank line between SETTLING and the telescope");
	assert.ok(raw[s].includes("\x1b[1mSETTLING\x1b[22m"), "SETTLING bold");
	// The tab row and the panels: no SETTLING; its focus column grey stepped in (it shares LENS's token).
	d.handleInput("2");
	assert.ok(!text(d.render(104)).some((l) => /^ ?LENS +SETTLING/.test(l)));
	d.handleInput("2");
});

test("stackLayers: three layers a quarter of the model apart around the focus one; near an end the third on the other side", () => {
	assert.deepEqual(stackLayers(48, 24), [12, 24, 36]);
	assert.deepEqual(stackLayers(48, 25), [13, 25, 37]);
	assert.deepEqual(stackLayers(48, 0), [0, 12, 24]);
	assert.deepEqual(stackLayers(48, 47), [23, 35, 47]);
	assert.deepEqual(stackLayers(4, 2), [1, 2, 3]);
	assert.deepEqual(stackLayers(2, 1), [0, 1]);
	assert.deepEqual(stackLayers(1, 0), [0]);
});

test("EXPERTS' grids as full rectangles (the mini dock's `plate`): the columns past the grid and the short last row's cells unlit, an odd row count's last lower half the background; without, the board's grids as they draw them", () => {
	const th = dockTheme(piTheme), pal = palette(th), E = 512, v = { layer: 3, L: 48, E, k: 10 };
	const shape = reshapeGrid(E, { cols: 32, rows: 16, pos: null }, 48), unlit = STOPS[0][1];
	assert.deepEqual([shape.cols, shape.rows], [47, 11], "Flash Next's 512 at 48 columns: 11 rows of 47, the last 42");
	const levels = Array.from({ length: E }, (_, e) => ((e * 37) % 100) / 100);
	const layers = [3, 4].map((layer) => ({ layer, cells: gridCells(levels, shape), cols: shape.cols, rows: shape.rows }));
	// Without a plate: the board's EXPERTS, pinned by the hash of its lines.
	const out: string[][] = [];
	for (const [w, rows, noMap] of [[50, 40, false], [50, 9, true], [48, 20, false], [97, 30, true]] as const) out.push(expertsStack(th, v, layers, pal, w, rows, true, noMap));
	assert.equal(createHash("sha256").update(JSON.stringify(out)).digest("hex").slice(0, 16), "6816b496383cbc31");
	const plated = expertsStack(th, v, layers, pal, 50, 40, false, false, 48), plain0 = expertsStack(th, v, layers, pal, 50, 40, false);
	assert.deepEqual(plated.map((l) => plain(l).trimEnd()).slice(0, 9), [plain(plated[0]), "▸ layer 4 of 48", ...Array(6).fill("▀".repeat(48)), "  layer 5 of 48"],
		"the title and labels as ever, the grid 48 wide");
	assert.ok(plated.slice(1).every((l) => visibleWidth(l) === 50), "each line under the title the width");
	const got = halfRows(plated.slice(2, 8)), was = halfRows(plain0.slice(2, 8));
	for (let r = 0; r < 12; r++)
		for (let c = 0; c < 48; c++) {
			const want = r === 11 ? [] : c >= 47 || (r === 10 && c >= 42) ? unlit : was[r][c];
			assert.deepEqual(got[r][c], want, `row ${r}, column ${c}`);
		}
});

test("EXPERTS on the board: the layers stacked, each a dim heading over its grid, the lights only; as many as the rows hold, at least one", () => {
	const th = { fg: (_c: string, s: string) => s, bold: (s: string) => s };
	const E = 512, v: any = { layer: 24, L: 48, E, k: 10 };
	const layers = [12, 24, 36].map((layer) => ({ layer, cols: 47, rows: 11, cells: Array.from({ length: 47 * 11 }, (_, e) => (e < E ? (e % 9) / 9 : null)) }));
	const out = expertsStack(th, v, layers, palette(th), 50, 22, false).map((l) => plain(l).trimEnd());
	assert.equal(out.length, 22, "the title, three layers of a heading and 6 lines");
	assert.equal(out[0], "EXPERTS  3 layers · 512 experts · 10 a token");
	assert.deepEqual(out.filter((l) => /layer \d+ of 48$/.test(l)), ["  layer 13 of 48", "▸ layer 25 of 48", "  layer 37 of 48"]);
	assert.deepEqual(out.flatMap((l, i) => (/layer \d+ of 48$/.test(l) ? [i] : [])), [1, 8, 15]);
	assert.ok(!out.some((l) => /%|of mix/.test(l)), "no shares, no words");
	assert.equal(visibleWidth(out[2]), 47, "the grid near the column's width");
	for (const l of expertsStack(th, v, layers, palette(th), 50, 22, false).slice(1)) assert.equal(visibleWidth(l), 50);
	assert.match(expertsStack(th, v, layers, palette(th), 50, 21, false).map(plain)[0], /^EXPERTS {2}2 layers/, "a line short of the third: two");
	const short = expertsStack(th, v, layers, palette(th), 50, 6, false).map((l) => plain(l).trimEnd());
	assert.equal(short.length, 6);
	assert.match(short[0], /^EXPERTS {2}1 layer · /);
	assert.equal(short[1], "▸ layer 25 of 48", "one fits: the focus layer, cut");
	const two = expertsStack(th, v, layers, palette(th), 50, 15, false).map((l) => plain(l).trimEnd()).filter((l) => /layer \d+ of 48$/.test(l));
	assert.deepEqual(two, ["  layer 13 of 48", "▸ layer 25 of 48"], "two: the focus and the nearest, in order");
	assert.equal(expertsStack(th, v, layers, palette(th), 50, 5, false).length, 5, "never over the rows given");
	assert.deepEqual(expertsStack(th, { ...v, reason: "a dense model: no experts to show" }, [], palette(th), 50, 9, false).map(plain),
		["EXPERTS", "a dense model: no experts to show"]);
});

test("EXPERTS without an expert map: a dim line saying how to make one, under the layers where a line is free, else in the title's note; the full size window's under its legend", () => {
	const th = { fg: (_c: string, s: string) => s, bold: (s: string) => s };
	const E = 512, v: any = { layer: 24, L: 48, E, k: 10 };
	const layers = [12, 24, 36].map((layer) => ({ layer, cols: 47, rows: 11, cells: Array.from({ length: 47 * 11 }, (_, e) => (e < E ? (e % 9) / 9 : null)) }));
	const free = expertsStack(th, v, layers, palette(th), 50, 23, false, true).map((l) => plain(l).trimEnd());
	assert.equal(free.length, 23);
	assert.equal(free.at(-1), LABELS.noMap);
	assert.equal(free[0], "EXPERTS  3 layers · 512 experts · 10 a token");
	const full = expertsStack(th, v, layers, palette(th), 50, 22, false, true).map((l) => plain(l).trimEnd());
	assert.equal(full.length, 22, "no line added");
	assert.equal(full[0], `EXPERTS  ${LABELS.noMap}`);
	assert.ok(!expertsStack(th, v, layers, palette(th), 50, 23, false).some((l) => l.includes(LABELS.noMap)), "with a map: none");
	assert.equal(cut(LABELS.noMap, 20), "no expert map · met…");
	const grid = { layer: 24, cols: 32, rows: 16, cells: new Array(E).fill(0) };
	const win = expertsLayers(th, v, [grid], palette(th), 101, [], 7, " zebra", true, 40, [undefined], "all", true).map((l) => plain(l).trimEnd());
	assert.equal(win.at(-1), LABELS.noMap);
	assert.equal(expertsLayers(th, v, [grid], palette(th), 101, [], 7, " zebra", true, 40).length, win.length - 1);
});

test("reshapeGrid: the fewest rows the width allows, the map's columns refilled in order, each sorted by its old row", () => {
	const E = 512, rows0 = 16, cols0 = 32;
	// A map: expert e at column e % 32, row (e * 7) % 16 within it (a permutation of the cells).
	const pos = Array.from({ length: E }, (_, e) => ((Math.floor(e / 32) * 7) % rows0) * cols0 + (e % 32));
	assert.equal(new Set(pos).size, E);
	const g = reshapeGrid(E, { cols: cols0, rows: rows0, pos }, 50);
	assert.deepEqual([g.cols, g.rows], [47, 11], "512 at 50 wide: 11 rows, 47 columns");
	assert.equal(new Set(g.pos).size, E, "one cell each");
	assert.ok(g.pos!.every((p) => p >= 0 && p < 47 * 11));
	const oldCol = (e: number) => pos[e] % cols0, newCol = (e: number) => g.pos![e] % 47;
	for (let a = 0; a < E; a++) for (const b of [a + 1, a + 33]) if (b < E && oldCol(a) < oldCol(b)) assert.ok(newCol(a) <= newCol(b), "left to right kept");
	const col0 = Array.from({ length: E }, (_, e) => e).filter((e) => newCol(e) === 0).sort((a, b) => g.pos![a] - g.pos![b]);
	assert.deepEqual(col0.map((e) => Math.floor(pos[e] / cols0)), [...col0.map((e) => Math.floor(pos[e] / cols0))].sort((a, b) => a - b), "top to bottom kept");
	const plainOrder = reshapeGrid(E, { cols: 32, rows: 16, pos: null }, 50);
	assert.deepEqual([plainOrder.cols, plainOrder.rows, plainOrder.pos], [47, 11, null], "number order stays number order");
	assert.equal(reshapeGrid(64, { cols: 32, rows: 2, pos: null }, 50).rows, 2, "64 at 50 wide: two rows of 32");
	const same = { cols: 47, rows: 11, pos };
	assert.equal(reshapeGrid(E, same, 50), same, "already that shape: kept");
});

/** A lens top 10 for read layers `l`: the written token's probability byte at each (255: not in the top 10). */
const top10 = (l: number[], wp: number[]): any => ({ l, id: new Int32Array(l.length * 3).fill(-1), w: new Array(l.length * 3).fill(""), p: new Uint8Array(l.length * 3).fill(255), wp: Uint8Array.from(wp) });

test("settleLevels: bright from where the lens settles on the written word, mid where it lands and leaves, faint in the top 10 only", () => {
	const t: any = { id: 9, text: " x", n_layers: 48, lens_by: "jlens",
		lens: [[0, 1, "a"], [8, 9, "x"], [16, 1, "a"], [24, 1, "a"], [32, 9, "x"], [40, 9, "x"], [47, 9, "x"]],
		top: top10([0, 8, 16, 24, 32, 40, 47], [255, 3, 255, 40, 2, 1, 0]),
		lens_logit: [[0, 2, "b"], [47, 9, "x"]] };
	assert.deepEqual([...settleLevels(t, "jlens")], [[0, 0], [8, 2], [16, 0], [24, 1], [32, 3], [40, 3], [47, 3]]);
	assert.deepEqual([...settleLevels(t, "logit")], [[0, 0], [47, 3]], "the logit lens (s): its own reads, no top 10 sent");
	assert.deepEqual([...settleLevels({ ...t, lens: [[0, 9, "x"], [8, 1, "a"]], top: undefined }, "jlens")], [[0, 2], [8, 0]], "never settles");
});

test("board: one blank line between SETTLING and the telescope, on the board and LENS full size, at 51 and 75 lines; none where SETTLING has two lines only", async () => {
	/** The lines over the telescope's first (the board) or the lenses' names (full size) blank in its column. */
	const gapOver = (lines: string[], under = /^#\d+ "/) => {
		const s = lines.findIndex((l) => under.test(l)), bar = lines[s].indexOf(" │"), left = (l: string) => (bar < 0 ? l : l.slice(0, bar)).trim();
		let n = 0;
		while (!left(lines[s - 1 - n])) n++;
		return n;
	};
	for (const rows of [51, 75]) {
		const { d, host } = board(40, rows);
		await tick();
		d.sel = 20;
		const before = d.render(104);
		assert.equal(gapOver(text(before)), 1, `the board at ${rows}`);
		assert.equal(before.length, rows, "the dock's height as it was");
		host.panel = "LENS";
		d.full = true;
		const opened = text(d.render(104));
		assert.equal(gapOver(opened, /^J-lens +│ logit lens$/), 1, `LENS full size at ${rows}`);
		assert.equal(opened.length, rows);
	}
	// The lens read at two layers: SETTLING's least and most its sub-heading and one line; at 51 lines (the board
	// without STEER) the blank line fits too (lensSplit's own test: tight, it goes rather than SETTLING).
	const { d } = board(40, 51, {}, { lens: [[40, 0, " x"], [44, 0, " x"]], lens_logit: [[40, 0, " x"], [44, 0, " x"]] });
	await tick();
	const lines = text(d.render(104)), s = lines.findIndex((l) => l.startsWith("SETTLING  "));
	assert.ok(s > 0 && /^L44 /.test(lines[s + 1]), "SETTLING shown");
	assert.equal(gapOver(lines), 1, "room for the blank line");
});

// Flash Next's lens steps (tracer.lens_steps: the J-lens's and the logit lens's layers; 4, 8, 15, 24 and 47 J-lens layers).
const upTo = (n: number) => Array.from({ length: n }, (_, l) => l);
const FLASH_STEPS: [number[], number[]][] = [
	[[24, 31, 38, 46], [24, 38, 46, 47]],
	[[4, 16, 24, 28, 31, 38, 40, 46], [4, 16, 24, 28, 38, 40, 46, 47]],
	[[0, 4, 8, 12, 16, 20, 24, 28, 31, 32, 36, 38, 40, 44, 46], [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]],
	[[0, 2, 4, 7, 8, 11, 12, 16, 17, 20, 22, 24, 26, 28, 31, 32, 33, 36, 37, 38, 40, 43, 44, 46],
		[0, 2, 4, 7, 8, 11, 12, 16, 17, 20, 22, 24, 26, 28, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]],
	[upTo(47), upTo(48)],
];
/** A host answering the lens steps route as the server does, at step `at`; `answer` replaces its answers. */
function stepsHost(at = 2) {
	const data = { model: MODEL, at, steps: FLASH_STEPS.map(([j, l]) => ({ n: j.length, layers: [...new Set([...j, ...l])].sort((a, b) => a - b), tps: null, replies: 0 })) };
	const host: any = { data, asked: [] as (number | undefined)[], answer: undefined as any,
		lensLayers: async (step?: number) => {
			host.asked.push(step);
			if (host.answer) return host.answer;
			if (step !== undefined) data.at = step;
			return structuredClone(data);
		} };
	return host;
}
/** A token's lenses as read at step `s` (the written word nowhere: the layers alone matter here). */
const readAt = (s: number) => ({ lens: FLASH_STEPS[s][0].map((l) => [l, 7, " the"]), lens_logit: FLASH_STEPS[s][1].map((l) => [l, 8, " a"]) });
const places = (lines: string[]) => {
	const reply = lines.indexOf("REPLY"), map = lines.findIndex((l) => /^ ?MAP /.test(l)), lens = lines.findIndex((l) => /^ ?LENS /.test(l));
	return { reply, map, lens, end: lines.findIndex((l, i) => i > lens && /^─+$/.test(l)) };
};
const dots = (lines: string[]) => lines.reduce((n, l) => n + (l.match(/│ L\d+(?: \+\d+)? +· +│/g)?.length ?? 0), 0);
const lensRows = (lines: string[]) => lines.reduce((n, l) => n + (l.match(/│[ ▸]L\d+/g)?.length ?? 0), 0);

test("board: LENS sized on the server's steps, whatever the step in use: REPLY, MAP and LENS stay put between replies read at other steps; without the steps, on the reply's", async () => {
	for (const rows of [ROWS_51, 75]) {
		const got: any[] = [];
		for (const s of [0, 2, 4]) {
			const { d } = board(40, rows, stepsHost(2), readAt(s));
			d.sel = 20;
			d.render(104);
			await tick();
			got.push(places(text(d.render(104))));
			d.dispose();
		}
		assert.deepEqual(got[1], got[0], `${rows} lines: replies read at steps 0 and 2`);
		assert.deepEqual(got[2], got[0], `${rows} lines: replies read at steps 4 and 2`);
	}
	// No steps (a server without the route, a failure, not answered): over the layers the reply read.
	const few = board(40, 75, {}, readAt(0)), many = board(40, 75, {}, readAt(4));
	for (const x of [few, many]) x.d.sel = 20;
	const [a, b] = [few, many].map((x) => places(text(x.d.render(104))));
	assert.ok(b.end - b.lens > a.end - a.lens, "a reply that read more: a taller LENS");
});

test("board: a lens-steps answer of another shape is a failure, not a blank dock; a panel that throws says so in its place only", async () => {
	for (const bad of [{}, null, { model: MODEL, at: 0, steps: null }, { model: MODEL, at: 0, steps: [{ n: 4 }] }]) {
		const host = stepsHost(2);
		host.answer = bad;
		const { d } = board(40, 75, host, readAt(2));
		d.sel = 20;
		d.render(104);
		await tick();
		const lines = text(d.render(104));
		assert.ok(!lines.some((l) => /the dock: /.test(l)), `${JSON.stringify(bad)}: ${lines.find((l) => /the dock: /.test(l))}`);
		assert.ok(lines.includes("REPLY") && lines.some((l) => /LENS/.test(l)), JSON.stringify(bad));
		assert.doesNotThrow(() => d.handleInput("+"));
		d.dispose();
	}
	const { d } = board(40, 75, {}, readAt(2));
	d.sel = 20;
	(d as any).tokens = () => { throw new Error("boom\n\x1b]8;;x\x07"); };
	let lines = text(d.render(104));
	const at = lines.findIndex((l) => / │ TOKENS$/.test(l));
	assert.match(lines[at + 1], / │ the dock: boom ·\]8;;x·$/, "under its title, one line, cleaned");
	assert.ok(lines.includes("REPLY") && lines.some((l) => /^ ?MAP /.test(l)) && lines.some((l) => /^ ?LENS /.test(l)), "the other panels drawn");
	// Working out its lines failing: the same, not a panel squeezed in silence.
	const other = board(40, 75, {}, readAt(2));
	other.d.sel = 20;
	const needOf = (other.d as any).needOf.bind(other.d);
	(other.d as any).needOf = (p: string, ...a: any[]) => { if (p === "TOKENS") throw new Error("no size"); return needOf(p, ...a); };
	lines = text(other.d.render(104));
	assert.match(lines[lines.findIndex((l) => / │ TOKENS$/.test(l)) + 1], / │ the dock: no size$/);
});

test("board: LENS's blank lines: the one over the telescope, and those a short tube frees at the bottom; - + move nothing: LENS keeps its lines and the lines above theirs", async () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const host = stepsHost(2), { d } = board(40, 75, host, readAt(2));
		d.sel = 20;
		d.render(104);
		await tick();
		const lines = text(d.render(104)), p = places(lines);
		const bar = lines[p.lens].indexOf(" │");
		// Blank in LENS besides the line over the telescope: how many, all at its bottom (-1: some elsewhere).
		const blanks = (ls: string[]) => {
			const tel = ls.findIndex((l) => /^#20 "/.test(l)), at = ls.map((_, i) => i).filter((i) => i > p.lens && i < p.end && !ls[i].slice(0, bar).trim());
			const foot = at.filter((i) => i !== tel - 1);
			return at.includes(tel - 1) && foot.every((b, i) => b === p.end - foot.length + i) ? foot.length : -1;
		};
		assert.equal(blanks(lines), 0, `blank lines in LENS:\n${lines.join("\n")}`);
		assert.match(lines[p.lens], /^ LENS +J-lens ◉────○ logit +▮▮▮▯▯ 15 layers/, "the step in LENS's title");
		assert.match(lines.at(-1)!, /↑\/↓ layer · -\/\+ layers · enter edit/);
		for (const key of ["+", "+", "-", "-", "-"]) {
			d.handleInput(key);
			await tick();
			mock.timers.tick(70 * 60);
			const now = text(d.render(104));
			assert.deepEqual(places(now), p, `${key}: REPLY, MAP and LENS where they were`);
			assert.ok(blanks(now) >= 0, `${key}: blank lines only at LENS's bottom besides the one over the telescope`);
			if (host.data.at >= 2) assert.equal(blanks(now), 0, `${key}: the tube whole at step ${host.data.at}`);
		}
		assert.deepEqual(host.asked.filter((x: any) => x !== undefined), [3, 4, 3, 2, 1]);
		d.dispose();
	} finally {
		mock.timers.reset();
	}
});

test("board: - + in every view of the lens: its column, full size with both lenses and with one; the tube follows", async () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const press = async (d: any, key: string) => {
			d.handleInput(key);
			await tick();
			mock.timers.tick(70 * 60);
			return text(d.render(104));
		};
		for (const [name, tx, full] of [["its column", readAt(2), false], ["full size, both lenses", readAt(2), true],
			["full size, one lens", { lens: [...new Set(FLASH_STEPS[2].flat())].map((l) => [l, 7, " the"]), lens_logit: undefined }, true]] as [string, any, boolean][]) {
			const host = stepsHost(2), { d } = board(40, 75, host, tx);
			d.sel = 20;
			d.render(104);
			await tick();
			if (full) d.handleInput("2");
			let lines = text(d.render(104));
			assert.equal(dots(lines), 0, `${name}: the step's layers, all read`);
			if (full) assert.match(lines.find((l) => /^▮/.test(l)) ?? "", /^▮▮▮▯▯  reads 15 layers/, `${name}: the step's line`);
			const rows = lensRows(lines);
			lines = await press(d, "+");
			assert.ok(dots(lines) > 0 && lensRows(lines) >= rows, `${name}: + the layers no reply read yet, faint`);
			if (full) assert.match(lines.find((l) => /^▮/.test(l)) ?? "", /^▮▮▮▮▯  reads 24 layers/);
			lines = await press(d, "-");
			lines = await press(d, "-");
			assert.ok(dots(lines) === 0 && lensRows(lines) < rows, `${name}: - fewer, at once`);
			assert.match(lines.at(-1)!, /-\/\+ layers/, `${name}: the key line says so`);
			d.dispose();
		}
	} finally {
		mock.timers.reset();
	}
});

test("board: a server without the route (404): no step and no -/+; one without the model loaded (409) says so where the step has its line", async () => {
	const host = stepsHost(2);
	host.answer = { error: "oMLX at h answered: Not Found", missing: true };
	const { d } = board(40, 75, host, readAt(2));
	d.sel = 20;
	d.render(104);
	await tick();
	let lines = text(d.render(104));
	assert.match(lines[places(lines).lens], /s swaps +│/, "no step in the title");
	assert.ok(!lines.at(-1)!.includes("-/+"));
	d.handleInput("+");
	await tick();
	assert.deepEqual(host.asked, [undefined], "- + do nothing");
	d.handleInput("2");
	lines = text(d.render(104));
	assert.ok(!lines.some((l) => /Not Found|▮/.test(l)), "full size: nothing either");
	d.dispose();
	const other = stepsHost(2);
	other.answer = { error: "model 'Flash' is not loaded" };
	const o = board(40, 75, other, readAt(2));
	o.d.sel = 20;
	o.d.render(104);
	await tick();
	o.d.handleInput("2");
	assert.ok(text(o.d.render(104)).some((l) => l.startsWith("model 'Flash' is not loaded")), "full size: the reason");
	o.d.dispose();
});

test("board: the layout at 100 to 200 columns: the dock's height and width at any step, replies read at any step; nothing moves at one step", async () => {
	for (const width of [100, 120, 140, 200])
		for (const full of [false, true])
			for (const at of [0, 2, 4]) {
				const seen: string[] = [];
				for (const read of [0, 2, 4]) {
					const { d } = board(40, 75, stepsHost(at), readAt(read));
					d.sel = 20;
					d.render(width);
					await tick();
					if (full) d.handleInput("2");
					const raw = d.render(width), why = `${width} columns, full ${full}, step ${at}, read at ${read}`;
					assert.equal(raw.length, 75, why);
					for (const l of raw) assert.ok(visibleWidth(l) <= width, `${why}: ${plain(l)}`);
					seen.push(JSON.stringify(places(text(raw))));
					d.dispose();
				}
				if (!full) assert.equal(new Set(seen).size, 1, `${width} columns, step ${at}: ${seen.join(" | ")}`);
			}
});

test("board: - + do nothing on the EXPERTS pages, the Steer page, on another panel full size, or typed into a word", async () => {
	for (const [name, setup] of [
		["EXPERTS' layer page", (d: any) => { d.handleInput("4"); d.handleInput(KEY.enter); }],
		["the Steer page", (d: any) => d.handleInput("6")],
		["TOKENS full size", (d: any) => d.handleInput("3")],
		["a word edited", (d: any) => d.handleInput(KEY.enter)],
	] as [string, (d: any, h: any) => void][]) {
		const host = stepsHost(2), { d, host: h } = board(40, 75, host, readAt(2));
		d.sel = 20;
		d.render(104);
		await tick();
		setup(d, h);
		d.render(104);
		for (const k of "-+=_") d.handleInput(k);
		await tick();
		assert.deepEqual(host.asked.filter((x: any) => x !== undefined), [], name);
		assert.equal(host.data.at, 2, name);
		d.dispose();
	}
});

test("board: a failure with steps held (409: the model unloaded) says so in LENS's title, why full size; - + and their hint off until the steps are back", async () => {
	const host = stepsHost(2), { d, live } = board(40, 75, host, readAt(2));
	d.sel = 20;
	d.render(104);
	await tick();
	assert.match(text(d.render(104)).at(-1)!, /-\/\+ layers/);
	host.answer = { error: "model 'Flash' is not loaded" };
	live.reply!.traceId = "T2";                                                   // a reply saved: asked again
	d.render(104);
	await tick();
	let lines = text(d.render(104));
	assert.match(lines[places(lines).lens], /logit +layers unknown +│/, "short, where the step was");
	assert.ok(!lines.at(-1)!.includes("-/+"));
	d.full = true;
	assert.ok(text(d.render(104)).includes("model 'Flash' is not loaded"), "the reason: LENS full size's step line");
	d.full = false;
	d.handleInput("+");
	await tick();
	assert.deepEqual(host.asked.filter((x: any) => x !== undefined), [], "no POST");
	assert.equal(host.data.at, 2);
	host.answer = undefined;
	live.reply!.traceId = "T3";
	d.render(104);
	await tick();
	lines = text(d.render(104));
	assert.match(lines[places(lines).lens], /▮▮▮▯▯ 15 layers/, "the steps back");
	d.dispose();
});

test("board: a failed lens-steps request is asked again as the next reply starts, even after a reply that was not saved", async () => {
	const host = stepsHost(2), { d, live } = board(40, 75, host, readAt(2));
	host.answer = { error: "ds4 at 127.0.0.1:8009 answered: Service Unavailable" };
	d.render(104);
	await tick();
	const lens = () => { const ls = text(d.render(104)); return ls[places(ls).lens]; };
	assert.match(lens(), /layers unknown +│/, "short: no address cut in the middle");
	assert.doesNotMatch(lens(), /127\.0/);
	host.answer = undefined;
	onEvent(live, { type: "start", rid: "r2", model: MODEL });                         // ended unsaved: no trace to wait for
	for (let i = 0; i < 3; i++) onEvent(live, { ...tok(i, readAt(2) as any), rid: "r2" });
	onEvent(live, { type: "end", rid: "r2", n_tokens: 3 });
	d.render(104);
	await tick();
	assert.match(lens(), /▮▮▮▯▯ 15 layers/, "asked as it started");
	host.answer = { error: "ds4 at 127.0.0.1:8009 answered: Service Unavailable" };
	onEvent(live, { type: "start", rid: "r3", model: MODEL });
	onEvent(live, { ...tok(0, readAt(2) as any), rid: "r3" });
	d.render(104);
	await tick();
	assert.match(lens(), /layers unknown/);
	host.answer = undefined;
	const n = host.asked.length;
	onEvent(live, { type: "end", rid: "r3", n_tokens: 1 });                           // unsaved again: same key as it streamed
	onEvent(live, { type: "start", rid: "r4", model: MODEL });
	onEvent(live, { ...tok(0, readAt(2) as any), rid: "r4" });
	d.render(104);
	await tick();
	assert.equal(host.asked.length, n + 1, "the next reply asks again");
	assert.match(lens(), /▮▮▮▯▯ 15 layers/);
	d.dispose();
});

test("board: the lens steps of another server or model are never shown; asked again for the new one", async () => {
	let model = MODEL;
	const host = stepsHost(2), { d, live } = board(40, 75, { ...host, steerModel: () => model }, readAt(2));
	host.answer = { error: "ds4 at 127.0.0.1:8009 answered: Service Unavailable" };
	d.render(104);
	await tick();
	const lens = () => { const ls = text(d.render(104)); return ls[places(ls).lens]; };
	assert.match(lens(), /layers unknown/);
	// A request to the old server in flight (the next reply started there) when the model and server switch.
	const answers: ((v: any) => void)[] = [];
	(d as any).host.lensLayers = () => new Promise((res) => answers.push(res));
	onEvent(live, { type: "start", rid: "r2", model });
	onEvent(live, { ...tok(0, readAt(2) as any), rid: "r2" });
	d.render(104);
	assert.equal(answers.length, 1);
	model = "Qwen3.6-35B";
	switchServer(live, "http://other", { restart() {}, running: false });
	answers[0]({ error: "ds4 at 127.0.0.1:8009 answered: Service Unavailable" });     // the old pair's, arriving late
	await tick();
	onEvent(live, { type: "start", rid: "q", model });
	for (let i = 0; i < 3; i++) onEvent(live, { ...tok(i, readAt(2) as any), rid: "q" });
	assert.doesNotMatch(lens(), /layers unknown|ds4|▮/, "neither the old failure nor the late one");
	assert.match(lens(), /s swaps +│/);
	assert.equal(answers.length, 2, "asked for the new pair");
	answers[1]({ ...structuredClone(host.data), model });
	await tick();
	assert.match(lens(), /▮▮▮▯▯ 15 layers/, "the new pair's own answer");
	model = "Other";                                                                  // another model on the same server
	assert.doesNotMatch(lens(), /▮/, "not the last model's steps");
	assert.equal(answers.length, 3);
	d.dispose();
});

test("board: a reply of another model than the steps are for shows no step, takes no - +", async () => {
	const host = stepsHost(2), { d, live } = board(40, 75, host, readAt(2));
	live.reply!.model = "Other";
	d.sel = 20;
	d.render(104);
	await tick();
	let lines = text(d.render(104));
	assert.match(lines[places(lines).lens], /s swaps +│/);
	d.handleInput("+");
	await tick();
	assert.deepEqual(host.asked.filter((x: any) => x !== undefined), []);
	d.handleInput("2");
	lines = text(d.render(104));
	assert.ok(!lines.some((l) => /▮|reads \d+ layers/.test(l)), "full size: no step line either");
	d.dispose();
});

test("boardKeys: every combination fits the key line's width", () => {
	const flags = [false, true];
	for (const p of PANELS) for (const order of [PANELS, [p]] as Panel[][])
		for (const inspecting of flags) for (const windowed of flags)
			for (const held of flags) for (const steered of flags) for (const layers of flags) {
				const k = boardKeys(p, inspecting, windowed, order, held, steered, layers);
				assert.ok(visibleWidth(k) <= KEYS_W, `${p} ${[inspecting, windowed, held, steered, layers]}: ${k}`);
			}
	assert.match(boardKeys("LENS", true, false, PANELS, true, true, true), /-\/\+ layers/, "kept: s other lens goes first");
});

test("board: MAP under the reply strip, over LENS; SETTLING's sub-heading in the lens's violet (the turret's), the word SETTLING bold, its tail dim", async () => {
	const { d, live } = board(40, ROWS_51, stepsHost(2), readAt(2));
	d.render(104);
	await tick();
	d.sel = 20;
	for (const lines of [text(d.render(104)), text(d.render(130))]) {
		const reply = lines.indexOf("REPLY"), map = lines.findIndex((l) => /^ ?MAP /.test(l)), lens = lines.findIndex((l) => /^ ?LENS /.test(l));
		assert.ok(reply > 0 && reply < map && map < lens, `${reply} ${map} ${lens}`);
		assert.equal(map, reply + 1 + 6 + 1, "MAP's title the first line under the reply strip's rule");
	}
	const th = dockTheme(), head = settlingPanel(th, live.reply!.tokens, 39, "jlens", 48, 9, palette(th))[0];
	assert.equal(head, th.fg("lens", th.bold("SETTLING")) + th.fg("lens", "  J-lens") + th.fg("dim", " · old → now"));
	const violet = colourCode(COLOURS.lens, "truecolor");
	assert.ok(head.startsWith(`${violet}\x1b[1mSETTLING\x1b[22m`), "SETTLING bold, in the lens's violet");
	assert.ok(head.includes(`${violet}  J-lens\x1b[39m`), "the lens's name in the violet too, not bold");
	assert.equal(head.split("\x1b[1m").length, 2, "only SETTLING bold");
	assert.ok(!head.includes("\x1b[7m"), "no inverse: a sub-heading, not a panel title");
});

test("SETTLING: a column a token, the newest on the right, deep layers on top, every line filled: bands where the layers are few, binned where the lines are", () => {
	const { live } = board(40);
	const toks = live.reply!.tokens, pal = palette(piTheme);
	const out = settlingPanel(piTheme, toks, 39, "jlens", 48, 9, pal);
	assert.equal(plain(out[0]), "SETTLING  J-lens · old → now");
	assert.deepEqual(out.slice(1).map((l) => plain(l).slice(0, 5).trimEnd()), ["L44", "L40", "L32", "L28", "L20", "L16", "L8", "L4"],
		"12 read layers in 16 half lines: a band of one or two each, labelled on the line it starts (one a line where two start)");
	// Fewer layers than lines: whole-line bands, each labelled once; no line blank.
	const five = toks.map((t) => ({ ...t, lens: t.lens!.filter(([l]) => l >= 28), lens_logit: [] }));
	const bands = settlingPanel(piTheme, five, 39, "jlens", 48, 9, pal).slice(1);
	assert.deepEqual(bands.map((l) => plain(l).slice(0, 5).trimEnd()), ["L44", "", "L40", "", "L36", "L32", "", "L28"], "5 layers in 8 lines");
	assert.ok(bands.every((l) => /▀{40}$/.test(plain(l))), "every line drawn");
	assert.equal(bands[0].slice(5), bands[1].slice(5), "a band: its lines alike");
	for (const l of out) assert.ok(visibleWidth(l) <= 48);
	assert.ok(out.slice(1).every((l) => visibleWidth(l) === 48));
	// The first 3 token columns of 43 are before the reply: blank; the 40 tokens drawn.
	assert.match(plain(out[1]), /^L44 {2}   ▀{40}$/);
	// Three lines: four layers to a half line.
	assert.deepEqual(settlingPanel(piTheme, toks, 39, "jlens", 48, 3, pal).slice(1).map((l) => plain(l).slice(0, 5).trimEnd()), ["L44", "L20"],
		"three layers to a half line");
	assert.equal(plain(settlingPanel(piTheme, toks, 39, "logit", 48, 9, pal)[0]), "SETTLING  logit lens · old → now", "follows s");
	const none = toks.map((t) => ({ ...t, lens: [], lens_logit: [] }));
	assert.deepEqual(settlingPanel(piTheme, none, 39, "jlens", 48, 9, pal).map(plain), ["SETTLING  J-lens · old → now", "no lens read on these tokens"]);
	// The colours: the written word settled (L40, L44) bright, the rest unlit; on the violet SETTLE_STOPS, not the
	// lights' amber (STOPS shares only its unlit colour).
	const grid = settlingPanel(piTheme, toks, 39, "jlens", 48, 9, pal), lit = grid[1];
	const code = (c: number[]) => colourMode(piTheme) === "truecolor" ? c.join(";") : null;
	const bright = code(rampAt(SETTLE_STOPS, 0.75));
	if (bright) {
		assert.ok(lit.includes(bright), "L44 lit, in the lens's violet");
		assert.equal(bright, COLOURS.lens.join(";"), "lit at the lens's own colour");
		const used = new Set(grid.slice(1).flatMap((l) => [...l.matchAll(/\x1b\[[34]8;2;(\d+;\d+;\d+)m/g)].map((m) => m[1])));
		const settle = new Set([0, 0.14, 0.4, 0.75].map((v) => rampAt(SETTLE_STOPS, v).join(";")));
		assert.ok(used.size >= 2 && [...used].every((c) => settle.has(c)), `every cell on SETTLE_STOPS: ${[...used].join(" ")}`);
		const amber = STOPS.slice(1).map(([, c]) => c.join(";")).concat([0.14, 0.4, 0.75].map((v) => rampAt(STOPS, v).join(";")));
		assert.ok(!amber.some((c) => used.has(c)), "no amber from the lights' ramp");
		assert.ok(used.has(STOPS[0][1].join(";")), "the unlit colour the lights'");
	}
	// Stepped in: the token in focus has a grey column, held near the middle while later tokens exist.
	const grey = code(mix(STOPS[0][1], COLOURS.text, 0.45));
	if (grey) {
		assert.ok(!settlingPanel(piTheme, toks, 39, "jlens", 48, 9, pal).join("").includes(grey), "no mark stepped out");
		const marked = settlingPanel(piTheme, toks, 10, "jlens", 48, 9, pal, true);
		assert.ok(marked.slice(1).some((l) => l.includes(grey)), "the focus column is grey where unlit");
		// 43 columns, focus 10, 40 tokens: the window ends at token 39 (10 + 21 < 42), so later tokens stay in view.
		assert.match(plain(marked[1]), /^L44 {2}   ▀{40}$/);
	}
});

test("SETTLING: a row only for the layers the lens shown read, from the reply's own reads, each lens its own", () => {
	const { live } = board(40);
	const toks = live.reply!.tokens, pal = palette(piTheme);
	for (const t of toks) {   // the J-lens read layers 12-44 here, the logit lens every layer
		t.lens_logit = Array.from({ length: 48 }, (_, l) => [l, 8, " a"] as [number, number, string]);
		t.lens = Array.from({ length: 33 }, (_, j) => [12 + j, 8, " a"] as [number, number, string]);
	}
	assert.deepEqual(settleLayers(toks, 39, "jlens").layers, Array.from({ length: 33 }, (_, j) => 44 - j));
	assert.equal(settleLayers(toks, 39, "logit").layers.length, 48);
	const labels = (view: "jlens" | "logit") => settlingPanel(piTheme, toks, 39, view, 48, 40, pal).slice(1).map((l) => plain(l).slice(0, 5).trim()).filter(Boolean).map((l) => Number(l.slice(1)));
	const j = labels("jlens"), g = labels("logit");
	assert.equal(j.length, 33, "33 layers in 39 lines: a band each, labelled once; no line for layers 0-11");
	assert.ok(j.every((l) => l >= 12) && j.at(-1) === 12, `${j}`);
	assert.equal(g.length, 39, "the logit lens: 48 layers in 78 half lines, a band starting on every line");
	assert.equal(g.at(-1), 0);
	for (const t of toks) t.lens = t.lens!.filter(([l]) => l % 2 === 0);   // another reply, another set
	assert.ok(labels("jlens").every((l) => l >= 12 && l % 2 === 0), "the reply's own reads");
});

/** Board tokens read lit at more layers: the J-lens settles at L0 (every layer lit), the logit lens at L8. */
function litBoard(rows = 67) {
	const b = board(40, rows);
	for (const t of b.live.reply!.tokens) {
		t.lens = t.lens!.map(([l]) => [l, t.id, t.text]);
		t.lens_logit = t.lens_logit!.map(([l]) => (l >= 8 ? [l, t.id, t.text] : [l, 8, " a"]));
	}
	return b;
}

test("SETTLING in the lens swap: the frame's lens, fogged toward the unlit colour (2 dimmer than 1), cleared from the deep end", () => {
	const toks = litBoard().live.reply!.tokens, pal = palette(piTheme);
	const at = (view: "jlens" | "logit", swap?: { fog?: number; clear?: number }) => settlingPanel(piTheme, toks, 20, view, 48, 9, pal, true, swap);
	const sharp = at("jlens"), f1 = at("jlens", { fog: 1, clear: 0 }), f2 = at("jlens", { fog: 2, clear: 0 });
	assert.equal(f1.length, sharp.length);
	assert.equal(f1[0], sharp[0], "the title as it is");
	for (let i = 1; i < sharp.length; i++) {
		assert.notEqual(f1[i], sharp[i], `line ${i} fogged`);
		assert.notEqual(f2[i], f1[i], `line ${i}: fog 2 not fog 1`);
		assert.equal(plain(f1[i]), plain(sharp[i]), "the same cells, dimmer");
	}
	const bright = rampAt(SETTLE_STOPS, 0.75).join(";"), dim1 = mix(rampAt(SETTLE_STOPS, 0.75), STOPS[0][1], 0.5).join(";"), dim2 = mix(rampAt(SETTLE_STOPS, 0.75), STOPS[0][1], 0.8).join(";");
	assert.ok(f1[1].includes(dim1) && !f1[1].includes(bright) && f2[1].includes(dim2));
	// Half of the 16 half lines sharp: the deepest four lines as without the swap, the shallow four fogged.
	const half = at("jlens", { fog: 1, clear: 0.5 });
	assert.deepEqual(half.slice(0, 5), sharp.slice(0, 5));
	for (let i = 5; i < 9; i++) assert.equal(half[i], f1[i], `line ${i} still fogged`);
	// The last frame (no fog) is the panel itself; the early frames draw the old lens.
	const fr = swapFrames("jlens", "logit", 21);
	assert.deepEqual(at(fr[8].view!, { fog: fr[8].fog }), at("logit"));
	assert.notDeepEqual(at(fr[0].view!, { fog: fr[0].fog }), at("logit", { fog: fr[0].fog }), "frame 0: the J-lens's reads, not the logit lens's");
	// The focus column's grey stays grey under the fog.
	const grey = mix(STOPS[0][1], COLOURS.text, 0.45).join(";"), orig = board(40).live.reply!.tokens;
	assert.ok(settlingPanel(piTheme, orig, 20, "jlens", 48, 9, pal, true, { fog: 2, clear: 0 }).slice(2).every((l) => l.includes(grey)));
});

test("board: s plays the lens swap on SETTLING too, in step with the telescope, a frame a tick; LENS full size and alone too; never left fogged", () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		// SETTLING's lines on the board (the right column does not move while the swap plays).
		const settling = (d: Dock) => {
			const raw = d.render(104), lines = text(raw), s = lines.findIndex((l) => l.startsWith("SETTLING  "));
			let e = s + 1;
			while (/^L\d+ /.test(lines[e])) e++;
			return { title: lines[s], rows: raw.slice(s + 1, e) };
		};
		const { d } = litBoard();
		d.sel = 20;
		const before = settling(d);
		d.handleInput("s");
		const frames: { title: string; rows: string[] }[] = [];
		for (let i = 0; i < 9; i++) {
			frames.push(settling(d));
			mock.timers.tick(70);
		}
		const ref = litBoard();
		ref.host.lensView = "logit";
		ref.d.sel = 20;
		const after = settling(ref.d);
		assert.deepEqual(frames[8], after, "the swap done: the logit lens's panel, sharp");
		assert.deepEqual(settling(d), after, "and it stays");
		assert.equal(frames[0].rows.length, before.rows.length);
		assert.notDeepEqual(frames[0].rows, before.rows, "frame 0 fogged");
		assert.notDeepEqual(frames[1].rows, frames[0].rows, "fog 2");
		assert.deepEqual(frames[1].rows, frames[2].rows, "the J-lens, fog 2");
		assert.notDeepEqual(frames[2].rows, frames[3].rows, "frame 3: the logit lens");
		assert.deepEqual(frames[3].rows, frames[4].rows);
		for (let i = 0; i < 7; i++) assert.notDeepEqual(frames[i].rows, after.rows, `frame ${i} not yet sharp`);
		assert.notDeepEqual(frames[5].rows, frames[6].rows, "clearing from the deep end");
		assert.equal(frames[0].title, before.title, "the title follows the frame's lens");
		assert.equal(frames[3].title, after.title);
		// LENS full size (2): s plays the swap on SETTLING there too; another panel full size stops it.
		const full = litBoard();
		full.d.sel = 20;
		full.d.render(104);
		full.d.handleInput("2");
		const before2 = full.d.render(104);
		full.d.handleInput("s");
		const f0 = full.d.render(104);
		assert.notDeepEqual(f0, before2, "frame 0 fogged");
		assert.ok(text(f0).some((l) => l === "SETTLING  J-lens · old → now"), "still the J-lens in frame 0");
		mock.timers.tick(70 * 10);
		assert.ok(text(full.d.render(104)).some((l) => l === "SETTLING  logit lens · old → now"), "the swap done");
	} finally {
		mock.timers.reset();
	}
});

test("MAP's cells: a token a cell by confidence, darker while thinking, the token in focus white; each reply from a row's start", () => {
	const toks = Array.from({ length: 30 }, (_, i) => ({ text: i === 9 ? "</think>" : ` t${i}`, p: i < 20 ? 0.95 : i < 25 ? 0.7 : 0.1 }));
	assert.equal(thinkEnd(toks), 9);
	assert.equal(thinkEnd([{ text: "a" }]), -1);
	const run = (ts: typeof toks, model?: string): MapRun => ({ n: ts.length, p: (i) => ts[i].p, think: thinkEnd(ts), model });
	const cells = mapCells([run(toks)], 12, 0, 36, 27) as number[][];
	const sum = (c: any) => c[0] + c[1] + c[2];
	assert.ok(sum(cells[0]) < sum(cells[10]), "the thinking (tokens 0-9) darker than the answer at the same confidence");
	assert.ok(cells[10][1] > cells[10][0], "sure: green");
	assert.ok(cells[22][0] > cells[22][2] && cells[22][1] > cells[22][2], "middling: yellow");
	assert.deepEqual(cells[27], COLOURS.text, "token 27: the focus");
	assert.ok(cells[26][0] > cells[26][1], "unsure: red");
	assert.deepEqual(cells[30], cells[35], "past the reply: unlit");
	// A probability kept as a byte, in 200ths rounded down: 50% and 90% stay on their side of the colours' edges.
	assert.deepEqual([...mapReply({ rid: "b", tokens: [0.9, 0.8999, 0.5, 0.4999, null, 1].map((p) => ({ text: "x", p })) }).p], [180, 179, 100, 99, 0, 200]);
	// Replies, rows 12 cells wide: each from a row's start (the rest of the row before it unlit), one of another model
	// than the last known before it after an amber cell (a model not known: none).
	const r = (_rid: string, n: number, model?: string) => run(toks.slice(0, n), model);
	assert.deepEqual(mapStarts([r("a", 30, "M"), r("b", 5, "M"), r("c", 12, "M"), r("d", 3, "N"), r("e", 2)], 12),
		{ starts: [0, 36, 48, 61, 72], marks: [60], end: 74 });
	assert.deepEqual(mapStarts([r("a", 3, "M"), r("b", 3), r("c", 3, "N"), r("d", 3), r("e", 3, "N")], 12).marks, [24], "M, unknown, N: amber before N");
	assert.deepEqual(mapStarts([r("a", 3, "M"), r("b", 3), r("c", 3, "M")], 12).marks, [], "M, unknown, M: none");
	const two = mapCells([r("a", 30, "M"), r("d", 3, "N")], 12, 0, 48) as number[][];
	assert.deepEqual([two[30], two[35]], [two[47], two[47]], "the rest of a's last row unlit");
	assert.deepEqual(two[36], COLOURS.accent, "another model: amber first");
	assert.deepEqual(two[37], (mapCells([r("d", 3, "N")], 12, 0, 1) as number[][])[0], "then d's first token");
	// From cell `a`: the same cells, shifted.
	assert.deepEqual(mapCells([r("a", 30, "M"), r("d", 3, "N")], 12, 24, 24), two.slice(24));
});

test("MAP on the board: a fixed window, a token half a character, sliding by whole lines (two rows); it follows the token in focus; the title names the tokens shown", () => {
	// 48 columns: two rows of 48 tokens a line, 96; 14 lines hold 1344.
	assert.deepEqual([1344, 1345, 1440, 1441, 3236].map((n) => mapWindow(n, 48, 14, -1)), [0, 1, 1, 2, 20], "live: the newest line last");
	assert.equal(mapWindow(600, 48, 14, -1), 0, "the reply fits: from the top");
	assert.equal(mapWindow(3236, 48, 14, 600, 20), 6, "stepped back to 600 (line 6): the window moves up to it");
	assert.equal(mapWindow(3236, 48, 14, 599, 3), 3, "within the window: it stays");
	assert.equal(mapWindow(3236, 48, 14, 287, 3), 2, "a line above: one line");
	assert.equal(mapWindow(3236, 48, 14, 2000, 2), 7, "below it: just enough to hold it (line 20 the last)");
	assert.equal(mapWindow(3236, 48, 14, 3235, 19), 20, "never past the newest");
	const toks = (n: number) => Array.from({ length: n }, (_, i) => ({ text: ` t${i}`, p: i % 3 === 0 ? 0.1 : 0.95 }));
	const draw = (n: number, focus = -1, prev = 0) => mapPanel(piTheme, toks(n), focus, 48, 15, false, true, mapWindow(n, 48, 14, focus, prev));
	// Fixed height from the first token; the lines past the reply unlit; a token half a character.
	for (const n of [1, 600, 1344, 1345, 3236]) assert.equal(draw(n).length, 15, `${n} tokens`);
	assert.equal(plain(draw(1)[1]), "▀".repeat(48));
	const one = halfRows(draw(1).slice(1)), unlit = STOPS[0][1];
	assert.equal(one.length, 28, "two rows a line");
	assert.notDeepEqual(one[0][0], unlit, "token 0: the first line's upper half");
	one.forEach((row, r) => row.forEach((c, i) => (r || i) && assert.deepEqual(c, unlit, `row ${r} cell ${i}: unlit`)));
	assert.equal(plain(draw(600)[0]), "MAP  600 tok · 200 unsure", "the whole reply: no range");
	assert.equal(plain(draw(3236)[0]), "MAP  3236 tok · 1079 unsure · 1921-3236");
	assert.equal(plain(draw(3236, 600, 20)[0]), "MAP  3236 tok · 1079 unsure · 577-1920");
	// A token more changes one cell, nothing shifts sideways; past the window's end the lines move up by one whole line.
	const a = draw(1400), b = draw(1401);
	assert.deepEqual(a.slice(1).map((l, i) => (l === b[i + 1] ? 0 : 1)), [...Array(13).fill(0), 1], "only the newest line");
	const c = draw(1441);
	assert.deepEqual(c.slice(1, 13), draw(1440).slice(2, 14), "slid by a whole line, two rows of tokens");
	// Narrow: unsure goes first, then the range.
	assert.equal(plain(mapPanel(piTheme, toks(3236), -1, 30, 8, false, true, 0)[0]), "MAP  3236 tok · 1-420");
	assert.equal(plain(mapPanel(piTheme, toks(3236), -1, 16, 8, false, true, 0)[0]), "MAP  3236 tok");
});

test("MAP's half cells: token i on row floor(i / w), an even row a line's upper half; the window two rows a line; the focus and the engram's marks on their half cell", () => {
	const w = 10, toks = Array.from({ length: 95 }, (_, i) => ({ text: ` t${i}`, p: i % 7 === 0 ? 0.1 : 0.95 }));
	const lines = (first: number, focus = -1, paint?: MapPaint) => mapPanel(piTheme, toks, focus, w, 4, false, true, first, paint).slice(1);
	const want = mapCells([{ n: 95, p: (i) => toks[i].p, think: -1 }], w, 0, 100) as number[][];
	const drawn = lines(0);
	assert.equal(halfRows(drawn).length, 6, "3 lines, 6 rows");
	for (let i = 0; i < 60; i++) {
		const r = Math.floor(i / w), line = Math.floor(i / (2 * w));
		assert.deepEqual(halfRows([drawn[line]])[r % 2][i % w], want[i], `token ${i}: line ${line}'s ${r % 2 ? "lower" : "upper"} half`);
	}
	// From line 1: tokens 20 on, the window two rows a line.
	assert.equal(mapWindow(95, w, 3, 25, 0), 0);
	assert.equal(mapWindow(95, w, 3, 60, 0), 1, "token 60, line 3: one line down");
	const from1 = halfRows(lines(1));
	for (let i = 20; i < 80; i++) assert.deepEqual(from1[Math.floor(i / w) - 2][i % w], want[i], `from line 1, token ${i}`);
	assert.equal(plain(mapPanel(piTheme, toks, -1, 30, 3, false, true, 1)[0]), "MAP  95 tok · 61-95", "30 wide: 60 tokens a line, from line 1");
	// The focus: white on its half cell, the upper (token 23, row 2) or the lower (token 37, row 3).
	for (const f of [23, 37]) {
		const got = halfRows(lines(0, f));
		assert.deepEqual(got[Math.floor(f / w)][f % w], COLOURS.text, `focus ${f}`);
		assert.deepEqual(got[Math.floor(f / w) ^ 1][f % w], want[f + ((Math.floor(f / w) & 1) ? -w : w)], `focus ${f}: the other half its own`);
	}
	// Engram inspection: the engram's tokens [12, 16) in the block colour, the token it predicted (16) in the text
	// colour (no focus mark then); the rest as painted.
	const block: [number, number, number] = [40, 60, 90], paint: MapPaint = { shade: () => undefined, engram: [12, 16], pred: 16, block };
	const marked = halfRows(lines(0, 16, paint));
	for (let i = 0; i < 60; i++)
		assert.deepEqual(marked[Math.floor(i / w)][i % w], i === 16 ? COLOURS.text : i >= 12 && i < 16 ? block : want[i], `engram, token ${i}`);
	const lower = halfRows(lines(0, 35, { ...paint, engram: [31, 35], pred: 35 }));
	assert.deepEqual([lower[3][1], lower[3][4], lower[3][5], lower[2][5]], [block, block, COLOURS.text, want[25]], "on a lower half: rows 3, line 1's");
	assert.deepEqual(halfRows(lines(1, -1, { ...paint, engram: [31, 35], pred: 35 }))[1][5], COLOURS.text, "the window from line 1: the same token, a line up");
	// No whole-character cells left.
	assert.equal("tallBlocks" in lights, false);
});

test("board: MAP's window follows the token in focus; full size, the whole reply", async () => {
	const { d, host } = board(1800, 78);
	await tick();
	const head = () => text(d.render(104)).find((l) => /^ ?MAP /.test(l))!.split(" │ ")[0].trim();
	const lines = text(d.render(104)), k = lines.findIndex((l) => /^ ?LENS /.test(l)) - 2 - lines.findIndex((l) => /^ ?MAP /.test(l));
	assert.ok(k >= MAP_LINES && k <= 14, `${k} lines of cells`);
	// 48 columns, two rows a line: 96 tokens a line, 19 lines.
	const range = (first: number) => `${first * 96 + 1}-${Math.min(1800, (first + k) * 96)}`, top = 19 - k;
	assert.equal(head(), `MAP  1800 tok · 360 unsure · ${range(top)}`, "live: the newest");
	d.sel = 0;
	assert.ok(head().endsWith(range(0)), "stepped back to the first token: the window moved up");
	d.handleInput("\x1b[C");                                                            // 1: within, it stays
	assert.ok(head().endsWith(range(0)));
	d.sel = k * 96 + 5;
	assert.ok(head().endsWith(range(1)), "past its end: down by a whole line");
	d.sel = 1799;
	assert.ok(head().endsWith(range(top)));
	host.panel = "MAP";
	d.full = true;
	assert.ok(text(d.render(104)).some((l) => l === "1 reply · 1800 tok · 360 unsure"), "full size: the session's replies");
});

test("MAP on the board and the mini dock: byte for byte, a token half a character", () => {
	// Every window of the board's MAP over replies with and without thinking, at the mini dock's width and the board's,
	// following and stepped; its title too. Each line's halves the cells mapCells gives; the hash is of the lines drawn
	// once a token took half a character.
	const th = dockTheme(piTheme), P = [0.95, 0.9, 0.8999, 0.7, 0.5, 0.49999, 0.1, null, 0.99, 0.3];
	const toks = (n: number, think: number) => Array.from({ length: n }, (_, i) => ({ text: i === think ? "</think>" : ` t${i}`, p: P[(i * 7) % P.length] }));
	const out: string[][] = [];
	for (const n of [1, 37, 300, 900])
		for (const think of [-1, 0, Math.floor(n / 3)])
			for (const [w, rows] of [[33, 2], [33, 4], [48, 8], [47, 3], [101, 8]])
				for (const focus of [-1, 0, Math.floor(n / 2), n - 1])
					for (const prev of [0, 3]) {
						const ts = toks(n, think), first = mapWindow(n, w, rows - 1, focus, prev), ls = mapPanel(th, ts, focus, w, rows, focus === 0, true, first);
						const want = mapCells([{ n, p: (i) => ts[i].p ?? 0, think: thinkEnd(ts) }], w, first * 2 * w, 2 * w * (rows - 1), focus) as number[][];
						assert.deepEqual(halfRows(ls.slice(1)).flat(), want, `${n} tokens, ${w} by ${rows}, focus ${focus}`);
						out.push(ls);
					}
	assert.equal(createHash("sha256").update(JSON.stringify(out)).digest("hex").slice(0, 16), "33d5a374a66fc4f5");
});

test("MAP in 256 colours: only 256-colour codes, each line its width, on the board, full size and in engram inspection", () => {
	const th = { ...piTheme, getColorMode: () => "256color" }, toks = Array.from({ length: 300 }, (_, i) => ({ text: ` t${i}`, p: i % 3 ? 0.95 : 0.2 }));
	const paint: MapPaint = { shade: () => undefined, engram: [10, 14], pred: 14, block: [40, 60, 90] };
	const ls = [mapPanel(th, toks, 17, 40, 5, false, true, 0), mapPanel(th, toks, -1, 40, 5, false, true, 0, paint),
		sessionMap(th, [], sessionReply("a", 300), 17, 40, 5).lines].flatMap((x) => x.slice(1));
	for (const l of ls) {
		assert.ok((l.match(/\x1b\[[0-9;]*m/g) ?? []).every((c) => /^\x1b\[(?:[34]8;5;\d+|0)m$/.test(c)), JSON.stringify(l));
		assert.equal(visibleWidth(l), 40);
		assert.equal(plain(l), "▀".repeat(40));
	}
});

/** A done reply of `n` tokens from `model`: a quarter unsure, the rest sure; its first `think` thinking. */
const sessionReply = (rid: string, n: number, model = "M", think = 0): Reply => ({ rid, model, done: true, prefill: 0,
	tokens: Array.from({ length: n }, (_, i) => ({ i, id: i, text: i === think - 1 ? "</think>" : ` t${i}`, p: i % 4 ? 0.95 : 0.2, entropy: 0, alts: [], n_layers: null })) });

test("MAP full size: every reply of the session a token half a character in a fixed box, filling from the top; full, the newest at the bottom", () => {
	// 20 wide, 8 lines: the title and 7 lines of two rows of 20 cells, 280 cells.
	const draw = (rs: Reply[], sel = -1, prev = 0) => sessionMap(piTheme, rs.slice(0, -1).map(mapReply), rs.at(-1)!, sel, 20, 8, prev);
	const rs = [sessionReply("a", 60, "M", 20), sessionReply("b", 7), sessionReply("c", 33)];
	const one = draw(rs.slice(0, 1)), three = draw(rs);
	const head = (rs: Reply[], w = 40) => plain(sessionMap(piTheme, rs.slice(0, -1).map(mapReply), rs.at(-1)!, -1, w, 8).lines[0]);
	assert.deepEqual([head(rs.slice(0, 1)), head(rs)], ["1 reply · 60 tok · 15 unsure", "3 replies · 100 tok · 26 unsure"]);
	assert.deepEqual([head(rs, 26), head(rs, 18)], ["3 replies · 100 tok", "3 replies"], "narrow: unsure goes, then the tokens");
	// The box: the same lines, each as wide, whatever the replies; the first reply's cells where they were.
	for (const m of [one, three, draw([sessionReply("x", 1)])]) {
		assert.equal(m.lines.length, 8);
		for (const l of m.lines.slice(1)) assert.equal(visibleWidth(l), 20);
	}
	assert.equal(three.lines[1], one.lines[1], "a's first 40 tokens: the same cells with b and c after it");
	assert.equal(one.first, 0);
	// a 0-59 (3 rows), b from row 3 (60-66), c from row 4 (80-112): 113 cells, 6 rows (3 of 7 lines), the rest unlit.
	assert.deepEqual(mapStarts(rs.map((x) => ({ n: x.tokens.length, p: () => 0, think: -1 })), 20), { starts: [0, 60, 80], marks: [], end: 113 });
	assert.equal(three.lines[7], one.lines[7], "past the session: unlit");
	// b starts on a row of its own, line 2's lower half: a's last row over it, the rest of b's row unlit.
	const cells = halfRows(three.lines.slice(1)), alone = halfRows(draw([rs[1]]).lines.slice(1));
	assert.deepEqual(cells[3].slice(0, 7), alone[0].slice(0, 7), "b's tokens from row 3");
	assert.deepEqual(cells[3].slice(7), Array(13).fill(STOPS[0][1]), "the rest of its row unlit");
	assert.deepEqual(cells[2], halfRows(one.lines.slice(1))[2], "a's last row, line 2's upper half");
	// Full: 30 replies of 33 tokens take 2 rows (a line) each, 30 lines: the last 7 lines, the newest the last one.
	const many = Array.from({ length: 30 }, (_, i) => sessionReply(`r${i}`, 33));
	const full = draw(many);
	assert.equal(full.first, 30 - 7);
	assert.equal(head(many), "30 replies · 990 tok · 270 unsure");
	assert.equal(draw([...many.slice(0, -1), sessionReply("r29", 34)]).first, 30 - 7, "the newest growing in its row: nothing moves");
	assert.equal(draw([...many, sessionReply("new", 1)]).first, 31 - 7, "a new reply (row 60, line 30's upper half): up a line");
	assert.equal(draw([...many, sessionReply("new", 1), sessionReply("newer", 1)]).first, 31 - 7, "another (row 61, its lower half): nothing moves");
	// Stepped on the newest reply's first token: in view, the window kept while the reply grows below it (f follows again).
	const long = [...many, sessionReply("long", 400)];
	const at = draw(long, 0, 0);
	assert.equal(at.first, 30 - 7 + 1, "moved down just enough to hold it: its line the last");
	assert.deepEqual(at.shown, [0, 40], "its tokens in view: line 30's two rows");
	assert.equal(draw([...many, sessionReply("long", 600)], 0, at.first).first, at.first, "it grows below: the view stays");
	assert.equal(draw([...many, sessionReply("long", 600)]).first, 30 + 15 - 7, "following: the newest line last");
});

test("MAP full size in the dock: 1 draws the session in the same box as replies end and the newest streams", async () => {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	const say = (rid: string, n: number, end = true) => {
		onEvent(live, { type: "start", rid, model: MODEL });
		for (let i = 0; i < n; i++) onEvent(live, tok(i, { rid }));
		if (end) onEvent(live, { type: "end", rid, n_tokens: n, trace_id: `T${rid}` });
	};
	const d = new Dock({ terminal: { rows: 67 }, requestRender() {} }, piTheme, { live, captions: false, close() {}, unfocus() {},
		fetchJson: async () => null, panel: "MAP" } as any);
	d.full = true;
	const box = () => {
		const lines = text(d.render(104)), head = lines.findIndex((l) => /^\d+ repl/.test(l));
		return { head: lines[head], cells: lines.slice(head + 1).findIndex((l) => /^─+$/.test(l)) };
	};
	say("a", 40);
	const first = box();
	assert.equal(first.head, "1 reply · 40 tok · 8 unsure");
	say("b", 300);
	onEvent(live, { type: "start", rid: "c", model: MODEL });                            // streaming: drawn at the end, counted as it comes
	for (let i = 0; i < 25; i++) {
		onEvent(live, tok(i, { rid: "c" }));
		assert.equal(box().head, `3 replies · ${341 + i} tok · ${68 + Math.floor(i / 5) + 1} unsure`);
	}
	assert.deepEqual(box(), { head: "3 replies · 365 tok · 73 unsure", cells: first.cells });
	assert.ok(first.cells >= 40, `${first.cells} lines of cells`);
	assert.equal(live.map?.map((m) => m.rid).join(), "a,b", "only the ended ones kept");
});

test("board: a STEER name past 11 columns takes the tag's room: the dose dropped, then the turns, then the tag", () => {
	assert.equal(nameTag("joyful", "15/18 @0.4", "15/18"), "joyful       15/18 @0.4");
	assert.equal(nameTag("uncooperative", "15/18 @0.4", "15/18"), "uncooperative     15/18");
	assert.equal(nameTag("explain-reasoning", "15/18 @0.4", "15/18"), "explain-reasoning      ", "the name whole, no room for the turns");
	assert.equal(nameTag("a-very-long-concept-name-indeed", "15/18 @0.4", "15/18"), "a-very-long-concept-…  ");
	for (const n of ["joyful", "uncooperative", "explain-reasoning", "a-very-long-concept-name-indeed"])
		assert.equal(visibleWidth(nameTag(n, "15/18 @0.4", "15/18")), 23);
});


test("board: the right column top down TOKENS, EXPERTS; EXPERTS takes the lines past its four layers a layer at a time, the rest blank under it", async () => {
	const at = async (rows: number) => {
		const { d } = board(40, rows, {}, TWELVE_BIG);
		await tick();
		d.sel = 20;
		const lines = text(d.render(104));
		const right = lines.map((l) => l.slice(columns(101).lw + 3));
		const title = (p: string) => right.findIndex((l) => new RegExp(`^ ?${p}  `).test(l));
		const end = right.findIndex((l, i) => i > title("EXPERTS") && /^─+$/.test(l));      // the key line's rule
		const col = right.slice(title("EXPERTS") + 1, end);
		let last = col.length - 1;
		while (last >= 0 && !col[last].trim()) last--;
		return { order: ["TOKENS", "EXPERTS"].map(title), experts: Number(/^ ?EXPERTS {2}(\d+) layers?/.exec(right[title("EXPERTS")])![1]),
			blank: col.length - 1 - last, block: col.findIndex((l, i) => i > 0 && /layer \d+ of 12$/.test(l)) };
	};
	const a51 = await at(51), a75 = await at(75), a125 = await at(125);
	for (const a of [a51, a75, a125]) {
		assert.ok(a.order[0] > 0 && a.order[0] < a.order[1], `${a.order}`);
		assert.ok(a.blank < a.block, `${a.blank} blank lines under EXPERTS: fewer than a layer's ${a.block}`);
	}
	assert.ok(a51.experts >= 1 && a75.experts > a51.experts && a125.experts > a75.experts, `${a51.experts} ${a75.experts} ${a125.experts}: EXPERTS grows`);
	assert.ok(a125.experts <= 12);
});

test("board: 6 opens the Steer page over the dock (the header and its steer kept), 6 the board again, esc the keyboard back; its list, the cursor's dose curve and test chats", async () => {
	const chats = { dose: 0.4, turns: [{ conv: "A casual chat", n: 1, of: 6, user: "Hi!", plain: { text: "Hello.", style: 0, sense: 3, good: false },
		steered: { text: "Hello!!", style: 2, sense: 3, good: true } }, { conv: "A casual chat", n: 2, of: 6, user: "And you?", plain: { text: "Fine.", style: 0, sense: 3, good: false },
		steered: { text: "Wonderful!", style: 3, sense: 3, good: true } }] };
	const { d, host, live, steered } = board(40, 75, { steerChats: async () => chats });
	await tick();
	live.steer = { model: MODEL, parts: [{ name: "joyful", model: MODEL, layer: 30, strength: 0.4, source: "profile" }] };
	d.sel = 20;
	d.render(104);
	d.handleInput("6");
	assert.equal(d.steerPage, true);
	d.render(104);
	await tick();
	const raw = d.render(104), lines = text(raw);
	assert.equal(raw.length, 75);
	for (const l of raw) assert.equal(visibleWidth(l), 104, plain(l));
	assert.match(lines[0], /^◆ metalscope {2}paused at #20 +STEER joyful@0\.4 L30$/, "the header as on the board");
	assert.ok(!lines.some((l) => /^REPLY|│ TOKENS|^ ?MAP /.test(l)), "the whole dock: no reply strip, no panels");
	const top = lines.findIndex((l) => /^ ?STEER +by the chat test · ★ top five/.test(l));
	assert.ok(top > 0 && /│ Joyful · pushes at L30 of 48  on at 0\.4$/.test(lines[top]), lines[top]);
	assert.match(lines[top + 1], /^▸ joyful {7}15\/18 @0\.4 +│ Upbeat and warm\.$/);
	assert.match(lines[top + 2], /^ {2}terse {10}untested /);
	assert.ok(!lines.some((l) => /│ meter /.test(l)), "no meter beside the dose curve");
	assert.ok(lines.some((l) => /│ dose +0 +0\.2 +0\.4$/.test(l)));
	assert.ok(lines.some((l) => /^EXAMPLES {2}A casual chat · turn 1-2 of 6 · 1\/2 · n next · N back$/.test(l)), "its test chats under it, both turns");
	assert.ok(lines.some((l) => /^no steering .*│ steered at 0\.4/.test(l)));
	assert.match(lines.at(-1)!, /^↑\/↓ concept · ←\/→ dose · enter steer · x off · d delete · n\/N turn · \? explain · 6 board · esc back$/);
	// n the next turn; ← a dose and enter steers with it; the token stays.
	d.handleInput("n");
	assert.ok(text(d.render(104)).some((l) => /turn 2 of 6/.test(l)));
	d.handleInput(KEY.left);
	d.handleInput(KEY.enter);
	await tick();
	assert.deepEqual(steered, ["joyful 0.2"]);
	assert.equal(d.sel, 20);
	// ? says what the page shows; again, the keys.
	d.handleInput("?");
	assert.ok(text(d.render(104)).some((l) => /^Concepts ranked by the chat test/.test(l)));
	d.handleInput("?");
	// 6: the board as it was (the focus and the token kept); 6 again: the page.
	host.panel = "TOKENS";
	d.handleInput("6");
	assert.equal(d.steerPage, false);
	assert.equal(d.sel, 20, "6 on the page goes back to the board, not out of inspect");
	assert.ok(text(d.render(104)).some((l) => / │ {1,2}TOKENS {2,3}#20/.test(l)));
	d.handleInput("6");
	assert.ok(text(d.render(104)).some((l) => /^ ?STEER +by the chat test/.test(l)));
	// esc (? text up or not): the keyboard back to Pi, the page kept, its key line how to step in; stepping in finds it.
	d.handleInput("?");
	d.handleInput(KEY.esc);
	assert.deepEqual([d.steerPage, d.stepped, d.sel], [true, false, -1], "esc: the keyboard back, the page kept");
	const out = text(d.render(104));
	assert.ok(out.some((l) => /^ ?STEER +by the chat test/.test(l)));
	assert.equal(out.at(-1), "ctrl+\\ steps in: ↑/↓ concept, 6 board · /scope help dock");
	d.stepIn(20);
	assert.equal(d.steerPage, true);
	assert.match(text(d.render(104)).at(-1)!, / · 6 board · esc back$/);
	d.handleInput("6");
	assert.ok(!text(d.render(104)).some((l) => /by the chat test/.test(l)));
	// A dock too short for the board: the one-panel board, the page kept.
	d.handleInput("6");
	(d as any).tui.terminal.rows = BOARD_MIN_ROWS - 1;
	assert.ok(text(d.render(104)).some((l) => /^ ?STEER +by the chat test/.test(l)));
	assert.equal(d.steerPage, true);
});

test("board: d on the Steer page asks, y deletes; esc, ctrl+\\ or any other key cancels; the list asked for again", async () => {
	let list = STEERING;
	const deleted: string[] = [];
	const { d, host } = board(40, 75, { steering: async () => parseSteering(list, MODEL),
		deleteConcept: async (name: string) => (deleted.push(name), list = { ...STEERING, concepts: STEERING.concepts.filter((c) => c.name !== name) },
			[{ text: `${name} deleted on Flash: moved to the trash`, level: "info" }]) });
	await tick();
	d.sel = 20;
	d.render(104);
	d.handleInput("6");
	d.render(104);
	await tick();
	d.handleInput("d");
	assert.ok(text(d.render(104)).some((l) => l.endsWith("│ y deletes joyful on Flash")));
	d.handleInput("n");                                                                       // cancels, and turns as ever
	assert.ok(!text(d.render(104)).some((l) => l.includes("y deletes")));
	d.handleInput("d");
	d.handleInput(KEY.esc);                                                                   // cancels, and hands the keyboard back
	assert.deepEqual([d.steerPage, d.sel, (d as any).sp.steer.del], [true, -1, undefined]);
	d.stepIn(20);
	d.handleInput("y");
	d.handleInput("d");
	d.handleInput("\x1c");                                                                    // ctrl+\ too
	assert.equal((d as any).sp.steer.del, undefined);
	if (!d.steerPage) d.handleInput("6");
	d.handleInput("d");
	d.handleInput("d");                                                                       // key repeat: asks again
	await tick();
	assert.deepEqual(deleted, []);
	d.handleInput("y");
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual(deleted, ["joyful"]);
	assert.equal(host.steerAt, "terse");
	const lines = text(d.render(104));
	assert.ok(lines.some((l) => /^▸ terse /.test(l)) && !lines.some((l) => /^. joyful /.test(l)));
	assert.ok(lines.some((l) => l.endsWith("│ ✓ joyful deleted on Flash: moved to the trash")));
});

test("widget: an older pi.json's hidden panels, its features and pins ignored", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	const path = join(dir, "pi.json");
	writePrefs({ hidden: ["FEATURES", "MAP"], features: ["f-comfort"], pins: ["terse"] } as any, path);   // as an older Pi left it
	const mk = () => {
		let dock: any;
		const ctx = { hasUI: true, model: { provider: "metalscope-omlx", id: MODEL }, ui: { setWidget() {}, setStatus() {}, notify() {},
			custom: (make: any) => new Promise<void>(() => { dock = make({ requestRender() {}, terminal: { rows: 67, columns: 290 } }, {}, {}, () => {}); }) } };
		const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false } as any;
		onEvent(live, { type: "start", rid: "r", model: MODEL });
		for (let i = 0; i < 5; i++) onEvent(live, tok(i));
		const api = async (p: string) => (p.startsWith("/api/steering") ? STEERING : { concepts: [] });
		const ui = createUi({ live, api } as any, path, () => {});
		ui.showDock(ctx);
		return { ui, live, dock: () => dock };
	};
	const a = mk();
	await tick();
	assert.equal((a.live as any).pins, undefined);
	const lines = text(a.dock().render(104));
	assert.ok(lines.some((l) => l.startsWith("◆ metalscope")) && !lines.some((l) => /FEATURES|terse/.test(l)), "the board as now");
	assert.ok(lines.some((l) => /^MAP /.test(l)), "MAP drawn, though the old file hid it");
});

test("following the newest while stepped in: only the token keys step to a token, which stays put as tokens arrive; end, or → on the newest, follows again; a new reply is followed", async () => {
	const { d, live } = board(40, 75);
	await tick();
	onEvent(live, { type: "start", rid: "s", model: MODEL });
	const more = (a: number, b: number) => { for (let i = a; i < b; i++) onEvent(live, { ...tok(i), rid: "s" }); };
	more(0, 30);
	const lines = () => text(d.render(104));
	const focusAt = () => Number(/TOKENS {2}#(\d+) /.exec(lines().join("\n"))![1]);
	d.stepIn(-1);
	assert.equal(focusAt(), 29, "stepped in, following: the newest");
	assert.ok(!/paused/.test(lines()[0]) && !/f live/.test(lines().at(-1)!));
	for (const k of [KEY.tab, "3", "3", KEY.tab, KEY.tab, "s", "?", "?"]) {
		d.handleInput(k);
		assert.equal(d.sel, -1, `${JSON.stringify(k)} leaves it following`);
	}
	more(30, 31);
	assert.equal(focusAt(), 30, "following: a token more, the view moves on");
	d.handleInput(KEY.left);
	assert.equal(d.sel, 29, "← steps to the token before the newest");
	assert.match(lines()[0], /paused at #29/);
	assert.match(lines().at(-1)!, /←\/→ token · f live · /);
	more(31, 36);
	assert.equal(focusAt(), 29, "stepped to: it stays as tokens arrive");
	d.handleInput("\x1b[F");                                                           // end
	assert.equal(d.sel, -1);
	assert.equal(focusAt(), 35);
	d.handleInput(KEY.left);
	d.handleInput("f");                                                                // f: the same, on any keyboard
	assert.equal(d.sel, -1, "f follows the newest again");
	d.handleInput(KEY.left);
	d.handleInput(KEY.right);
	assert.equal(d.sel, 35, "→ to the newest: on it");
	d.handleInput(KEY.right);
	assert.equal(d.sel, -1, "→ on the newest: following again");
	d.handleInput(KEY.left);
	onEvent(live, { type: "start", rid: "t", model: MODEL });
	for (let i = 0; i < 5; i++) onEvent(live, { ...tok(i), rid: "t" });
	assert.equal(focusAt(), 4, "a new reply: followed");
	assert.equal(d.sel, -1);
	d.handleInput(KEY.esc);
	assert.ok(/ctrl\+\\ steps in/.test(lines().at(-1)!), "esc: the keyboard back");
});

test("EXPERTS full size: EXPERTS FOR under the grids across the width, its tokens from the tally in one column, no cursor's violet", () => {
	const th = { fg: (_c: string, s: string) => s, bold: (s: string) => s };
	const E = 64, layers = [0, 1, 2, 3, 4].map((layer) => ({ layer, cols: 32, rows: 2, cells: Array.from({ length: E }, () => 0) }));
	const tops = topsOf({ against: null, experts: [{ e: 7, top: ["w7a", "w7b", "w7c"] }] });
	const v: any = { layer: 4, L: 5, E, k: 2 };
	const mine: [number, number][] = [[7, 0.78], [63, 0.22]];
	const band = (t: ReturnType<typeof topsOf>, source: "all" | "session" = "all") => {
		const out = expertsLayers(th, v, layers, palette(th), 101, mine, 7, "word", true, 40, [t], source).map((l) => plain(l).trimEnd());
		const at = out.findIndex((l) => l.startsWith(LABELS.forToken('"word"')));
		return out.slice(at, at + 1 + LIST_ROWS);
	};
	const [head, ...rows] = band(tops);
	assert.match(head, new RegExp(`^${LABELS.forToken('"word"')} +${LABELS.its} · ${LABELS.source.all}$`));
	assert.deepEqual(rows.map((l) => /^ {2}#\d+ /.test(l)), [true, true, ...new Array(LIST_ROWS - 2).fill(false)], "ten rows kept, two filled");
	assert.match(rows[0], /^ {2}#7 +█{12} +78% {2}"w7a" "w7b" "w7c"$/, "the largest share a full bar, its tokens quoted");
	assert.match(rows[1], /^ {2}#63 +█{1,5}[▏▎▍▌▋▊▉]? +22% {2}-$/, "none in the tally: a dash");
	assert.equal(rows[0].indexOf('"w7a"'), head.indexOf(LABELS.its), "under the heading");
	assert.match(band(undefined)[1], /78%$/, "asked for: blank");
	assert.match(band(topsOf(null))[1], /78% {2}-$/, "no tally: a dash");
	assert.match(band(tops, "session")[0], new RegExp(`${LABELS.its} · ${LABELS.source.session}$`));
	const raw = expertsLayers({ ...th, fg: (c: string, s: string) => `<${c}>${s}` }, v, layers, palette(th), 101, mine, 7, "word", true, 40, [tops]);
	assert.ok(!raw.some((l) => l.includes("<lens>")), "no violet cursor");
	assert.ok(raw.some((l) => l.includes("  <faint>-")), "the dash faint");
});

test("scrollWindow: the least move that keeps a row in view, with a margin where the list allows", () => {
	assert.equal(scrollWindow(512, 10, 0, 0, 2), 0);
	assert.equal(scrollWindow(512, 10, 7, 0, 2), 0, "within the margin: it stays");
	assert.equal(scrollWindow(512, 10, 8, 0, 2), 1, "past it: one row");
	assert.equal(scrollWindow(512, 10, 100, 0, 2), 93, "a jump: two rows under it");
	assert.equal(scrollWindow(512, 10, 94, 93, 2), 92, "up: two rows over it");
	assert.equal(scrollWindow(512, 10, 511, 93, 2), 502, "the last: the list's end");
	assert.equal(scrollWindow(5, 10, 4, 3, 2), 0, "a short list: from the top");
	assert.equal(scrollWindow(512, 10, -1, 0), 502, "none: the last rows");
});

/** Pi's TUI with its editor focused (the keys it is given in `got`), the dock an overlay shown as widget.ts shows it, then
 * stepped into; `forKeys`: opened for the keys (alt+i), so leaving closes it (`closed`). */
function piWithDock(extra: object, forKeys: boolean) {
	const tui: any = new TuiMainScreen({ columns: 200, rows: 67, write() {}, hideCursor() {}, showCursor() {} } as any);
	tui.stopped = true;   // nothing drawn
	const got: string[] = [], editor = { render: () => [], invalidate() {}, handleInput: (k: string) => got.push(k) };
	tui.addChild(editor);
	tui.setFocus(editor);
	let handle: any;
	const closed = { n: 0 };
	const { host } = board(40, 67, { ...extra, unfocus: () => handle.unfocus(), close: () => (closed.n++, handle.hide()) });
	const d = new Dock(tui, piTheme, host);
	handle = tui.showOverlay(d, { nonCapturing: !forKeys });
	if (forKeys) d.closeOnEsc = true;
	else handle.focus();
	d.stepIn(3);
	return { tui, d, editor, got, closed };
}

test("ctrl+c in a focused dock: the keyboard back to Pi's editor, and the key on to it (board and mini, Pi's own TUI)", () => {
	for (const extra of [{}, { mini: true }]) for (const forKeys of [false, true]) {
		const what = JSON.stringify({ ...extra, forKeys });
		const { tui, editor, got, closed } = piWithDock(extra, forKeys);
		tui.handleTerminalInput("\x03");
		assert.equal(tui.getFocusedComponent(), editor, what);
		assert.deepEqual(got, ["\x03"], what);
		assert.equal(closed.n, forKeys ? 1 : 0, what);
	}
});

test("the stop key in a focused dock: Pi's raw input listener has it before the dock, so the job's question is asked and the dock keeps the keyboard", async (t) => {
	const settle = () => new Promise((r) => setImmediate(r));
	for (const extra of [{}, { mini: true }]) for (const key of ["\x1bx", "≈"]) {
		const what = JSON.stringify({ ...extra, key });
		const { tui, d, got, closed } = piWithDock(extra, false);
		// The job runner as index.ts registers it, its ctx's onTerminalInput as Pi 1.5.2's interactive mode gives it
		// (addExtensionTerminalInputListener: the TUI's addInputListener).
		const handlers: Record<string, (e: any, ctx: any) => void> = {}, asked: string[] = [];
		const running = { kind: "calibrate", job: "c-1", name: "calm", phase: "calibrating", origin: "pi", steps: [] };
		registerJobs({ pi: { on: (n: string, f: any) => void (handlers[n] = f) }, live: { base: "" },
			api: async (path: string) => (path === "/api/jobs" ? { latest: running, jobs: [] } : undefined) } as any);
		const ctx = { hasUI: true, ui: { setStatus() {}, setWidget() {}, notify() {}, select: async (title: string) => void asked.push(title.split("\n")[0]),
			onTerminalInput: (h: any) => tui.addInputListener(h) } };
		handlers.session_start({}, ctx);
		t.after(() => handlers.session_shutdown({}, ctx));   // a failed assertion must not leave the poll's timer running
		await settle();
		tui.handleTerminalInput(key);
		await settle();
		assert.deepEqual(asked, ["Stop calibrating calm?"], what);
		assert.equal(tui.getFocusedComponent(), d, what);
		assert.deepEqual([got, closed.n], [[], 0], what);
	}
});

/** A LENS edit applied to token `at` of a done reply of `n` tokens, on a dock `width` x `rows` (`full`: LENS full size),
 * with a clock: the frames from the word chosen on, each [ms since enter, label, lines]. The answer: its reply starts at
 * `startMs`, its token (`said`) at `tokenMs`, then the rest (three tokens a step) and its end. */
async function applyEdit(o: { n?: number; at?: number; width?: number; rows?: number; full?: boolean; swap?: string; said?: string; startMs?: number;
	tokenMs?: number; keys?: [number, string][]; fail?: string }) {
	const { n = 300, at = 290, width = 104, rows = 67, said = " new", startMs = 900, tokenMs = 1400 } = o;
	let T = 0;
	const sent: string[] = [];
	const { d, host, live } = board(n, rows, { now: () => T, edit: async (a: string, _ids: unknown, x: any) => {
		sent.push(a);
		if (o.fail) return [{ text: o.fail, level: "error" }];   // the server's 4xx: no answer
		live.edit = { ...x, steer: `remove~w${at}/18-44`, model: MODEL, sent: true };
		return [];
	} });
	await tick();
	host.panel = "LENS";
	d.full = !!o.full;
	d.sel = at;
	const frames: [number, string, string[]][] = [];
	const shot = (label: string, t0: number) => { d.invalidate(); frames.push([T - t0, label, d.render(width)]); };
	d.render(width);
	d.handleInput(KEY.enter);
	if (o.swap) for (const c of o.swap) d.handleInput(c);
	else d.handleInput("\x7f");
	shot("chosen", T);
	d.handleInput(KEY.enter);
	await tick();
	const t0 = T, events: [number, string, () => void][] = o.fail ? [] : [
		[startMs, "start", () => onEvent(live, { type: "start", rid: "a", model: MODEL, edit: { of: "r", at } })],
		[tokenMs, "token", () => {
			onEvent(live, { ...tok(0), rid: "a", id: 900, text: said });
			onEvent(live, { type: "end", rid: "a", n_tokens: 1, trace_id: "TA" });
			onEvent(live, { type: "start", rid: "b", model: MODEL, edit: { of: "a", at: at + 1 } });
		}],
		[tokenMs + 100, "rest", () => { for (let i = 0; i < 3; i++) onEvent(live, { ...tok(i), rid: "b", id: 901 + i, text: ` n${i}` }); }],
		[tokenMs + 200, "rest", () => { for (let i = 3; i < 6; i++) onEvent(live, { ...tok(i), rid: "b", id: 901 + i, text: ` n${i}` }); }],
		[tokenMs + 300, "done", () => onEvent(live, { type: "end", rid: "b", n_tokens: 6, trace_id: "TB" })],
	];
	for (const [t, k] of o.keys ?? []) events.push([t, `key ${JSON.stringify(k)}`, () => d.handleInput(k)]);   // before that frame
	const times = [...new Set([...Array.from({ length: 14 }, (_, k) => k * 70), ...events.map(([t]) => t)])].sort((a, b) => a - b);
	for (const t of times) {
		T = t0 + t;
		const now = events.filter(([e]) => e === t);
		for (const [, , f] of now) f();
		shot(now.map(([, l]) => l).join(",") || "sweep", t0);
	}
	d.dispose();
	return { frames, d, live, sent };
}

/** The reply strip's text in the frame `ms` after enter (applyEdit). */
function editStrip(frames: [number, string, string[]][], ms: number, raw = false): string {
	const f = frames.find(([t]) => t === ms)![2], ls = text(f), a = ls.indexOf("REPLY");
	return (raw ? f : ls).slice(a + 1, ls.findIndex((l, i) => i > a && /^─+$/.test(l))).join("\n");
}

test("a LENS edit applied moves nothing before the edit point: the strip's rows, the header, MAP's window; at the reply's end, mid-reply, one-panel, full size, a fast answer", async () => {
	for (const o of [{}, { at: 150 }, { swap: "lion" }, { width: 60, rows: 40 }, { full: true }, { startMs: 50, tokenMs: 100 }, { n: 120, at: 115, width: 160, rows: 80 }, { n: 3000, at: 2990 }, { at: 0 }, { n: 20, at: 0 }]) {
		const { frames } = await applyEdit(o), at = (o as any).at ?? 290;
		const base = text(frames[0][2]), row = base.findIndex((l, i) => i > 2 && l.includes(` w${at}`)), cut = base[row].indexOf(` w${at}`);
		assert.ok(row > 2 && /paused at #/.test(base[0]), JSON.stringify(o));
		const map = (ls: string[]) => ls.map((l) => /MAP .*· (\d+)-\d+/.exec(l)?.[1]).find(Boolean) ?? "1";   // the whole reply: no range
		for (const [t, label, lines] of frames.slice(1)) {
			const ls = text(lines), why = `${JSON.stringify(o)} at ${t} ms (${label})`;
			// Every row before the edit's, byte for byte; the last of them may take the answer's first word at its end (a
			// shorter word that fits there, as the reply's text is laid out), its text before unchanged.
			const up = !(t < ((o as any).tokenMs ?? 1400)) && row > 3 && ls[row - 1] !== base[row - 1] && ls[row - 1].startsWith(base[row - 1].trimEnd());
			assert.deepEqual(lines.slice(0, up ? row - 1 : row), frames[0][2].slice(0, up ? row - 1 : row), why);
			assert.equal(ls[row].slice(0, cut), base[row].slice(0, cut), why);              // and its line up to the word
			assert.equal(map(ls), map(base), why);                                          // MAP's window: from the same token
			assert.equal(lines.length, frames[0][2].length, why);
		}
	}
});

test("a LENS edit applied: the word struck, burnt out in 210 ms, the tail after it swept away right to left in 650 ms, the new word in its place", async () => {
	const { frames } = await applyEdit({});
	const at = (ms: number) => frames.find(([t]) => t === ms)![2].join("\n"), strip = (ms: number) => editStrip(frames, ms);
	assert.match(at(0), /\x1b\[9mw290/);                                                            // struck
	assert.match(strip(70), /▒▒▒▒/);
	assert.match(strip(140), /░░░░/);
	assert.match(strip(210), /▸ {3}/);
	assert.ok(!/ w290/.test(strip(210)) && / w291/.test(strip(210)) && / w299/.test(strip(210)));   // the burn over, the sweep from the end
	assert.ok(/ w291/.test(strip(490)) && !/ w299/.test(strip(490)));                                // half way: the end gone first
	for (const ms of [910, 900]) assert.ok(!/ w29[1-9]/.test(strip(ms)), `${ms}`);                   // swept, held until the answer
	assert.match(strip(1400), /▸new/);                                                               // the new word in its place
	assert.match(strip(1700), /▸new n0 n1 n2 n3 n4 n5/);
});

test("a LENS edit applied, the new word shorter than the burnt one: it ends the row above where it fits, as the reply's text is laid out; that row's text kept", async () => {
	const { frames } = await applyEdit({ n: 3000, at: 2990 });
	const row = (ms: number) => editStrip(frames, ms).split("\n").find((l) => / w2989/.test(l))!;
	assert.match(row(0), / w2989 *$/);
	assert.match(row(1400), / w2989 ▸new *$/);
});

test("a LENS edit while one is answered: enter sends nothing more, g changes nothing, ↑/↓/←/→ move the cursor and the animation goes on", async () => {
	const plain = await applyEdit({});
	const enter = await applyEdit({ keys: [[140, KEY.enter], [280, KEY.enter], [1400, KEY.enter]] });
	assert.equal(enter.sent.length, 1);
	assert.deepEqual(enter.frames.map(([, , l]) => l), plain.frames.map(([, , l]) => l));   // the same frames: nothing reopened
	const g = await applyEdit({ keys: [[280, "g"]] });
	assert.deepEqual(g.frames.map(([, , l]) => l), plain.frames.map(([, , l]) => l));
	const moved = await applyEdit({ keys: [[280, KEY.left], [420, KEY.right], [420, KEY.right]] });
	const raw = (f: typeof plain.frames, ms: number) => editStrip(f, ms, true), inv = (ms: number) => /\x1b\[7m ?(w\d+)/.exec(raw(moved.frames, ms))?.[1];
	assert.ok(!/\x1b\[7m/.test(raw(plain.frames, 280)));   // on the edited token: the burn drawn, no inverse
	assert.equal(inv(280), "w289");                           // on the token before: its inverse
	assert.equal(inv(420), "w291");                           // on one not swept yet: as ever
	assert.equal(inv(910), undefined);                        // swept: not drawn
	for (const ms of [280, 490, 910]) assert.equal(editStrip(moved.frames, ms), editStrip(plain.frames, ms), `${ms}`);   // the animation goes on
	assert.equal(moved.sent.length, 1);
});

test("a LENS edit the server refuses (4xx): the reply back whole at once, its note on the key line", async () => {
	const { frames, sent } = await applyEdit({ fail: "steer: 400 no such word" });
	assert.equal(sent.length, 1);
	const s = editStrip(frames, 0), last = text(frames[1][2]).filter(Boolean).at(-1)!;
	assert.ok(/ w290/.test(s) && / w299/.test(s) && !/\x1b\[9m|▸|▒|░/.test(s), s);
	assert.match(last, /steer: 400 no such word/);
	assert.deepEqual(frames.at(-1)![2], frames[1][2]);   // nothing after it moves
});

test("a LENS edit applied with an answer faster than the animation: its token takes the word's place at once, the rest of the old tail gone", async () => {
	const { frames } = await applyEdit({ startMs: 50, tokenMs: 100 });
	const s = (ms: number) => editStrip(frames, ms);
	assert.ok(/ w29[1-9]/.test(s(70)) && !/▸new/.test(s(70)));     // the answer's reply started: the burn goes on, the old tail still there
	assert.ok(/▸new/.test(s(100)) && !/ w29[0-9]/.test(s(100)));   // its token: in place, the old tail gone with the sweep
	assert.ok(!/ w29[0-9]/.test(s(210)));
});

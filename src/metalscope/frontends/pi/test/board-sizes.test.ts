import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { DOCK_MIN_ROWS, MAP_CELL_LINES, MINI_WIDTH, RIGHT_W, layoutKeys, MAP_LINES, type Panel, PANELS } from "../board.ts";
import { gridShape, reshapeGrid } from "../lights.ts";
import { Dock, SETTLE_MOST } from "../dock.ts";
import { type Live, onEvent } from "../state.ts";

// The board at every size and layout: each frame exactly the dock's lines, each line exactly its columns, for models of
// other depths and grid shapes. And nothing moves as a reply grows.

const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const text = (lines: string[]) => lines.map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd());
const tick = () => new Promise((r) => setTimeout(r, 0));
const piTheme = { fg: (_c: string, s: string) => s, bg: (_c: string, s: string) => s, bold: (s: string) => s };

/** A frame as every size test wants it: `rows` lines, each `width` columns, and nothing failed drawing it (the whole
 * dock's error, or a panel's in its place: dock.ts guard, which would otherwise keep the frame whole). */
function framed(out: string[], rows: number, width: number, at: string): string[] {
	assert.equal(out.length, rows, `${at}: ${out.length} lines`);
	for (const l of out) assert.equal(visibleWidth(l), width, `${at}: ${plain(l)}`);
	const failed = text(out).find((l) => l.includes("the dock: "));
	assert.equal(failed, undefined, `${at}: ${failed}`);
	return out;
}

interface Model { name: string; L: number; E: number; k: number; layers: number; read?: number[] }
const MODELS: Model[] = [
	{ name: "forty", L: 40, E: 384, k: 8, layers: 40 },     // grids 32 by 12
	{ name: "fortyeight", L: 48, E: 512, k: 10, layers: 48 },   // grids 32 by 16
];

/** Reply token `i` of model `m`: routed at every layer, read by the lens at every layer (or `m.read`). */
function tok(m: Model, i: number): any {
	const experts: number[] = [], w: number[] = [];
	for (let l = 0; l < m.L; l++) for (let j = 0; j < m.k; j++) { experts.push((i * 7 + l * 13 + j * 31) % m.E); w.push(j ? 20 : 120); }
	const lens = (m.read ?? [...Array(m.layers).keys()]).map((l) => [l, l >= m.layers - 6 ? 100 + i : 7, l >= m.layers - 6 ? ` w${i}` : " the"]);
	return { type: "token", rid: "r", i, id: 100 + i, text: ` w${i}`, p: i % 5 ? 0.95 : 0.3, entropy: 0.5,
		alts: Array.from({ length: 9 }, (_, a) => [` a${a}`, 0.04 - a * 0.004]), n_layers: m.layers,
		lens, lens_by: "jlens", experts, expert_w: w, expert_k: m.k };
}

/** 22 concepts by the chat test (two of them not chat-tested: no steering only), and a test chat of a long reply each. */
const STEER = (model: string) => ({ model, concepts: Array.from({ length: 22 }, (_, k) => ({ name: `concept-${k}`, title: `Concept ${k}`, description: "A style. ".repeat(12),
	layer: 20, chat: k < 20, stale: false, good: k < 20 ? 18 - k : 0, turns: 18, pick: k < 20 ? 0.4 : 0, rank: k < 20 ? k + 1 : undefined,
	doses: (k < 20 ? [0, 0.2, 0.4, 0.6] : [0]).map((dose) => ({ dose, good: k < 20 ? 9 : null, turns: k < 20 ? 18 : 0, style: k < 20 ? 2 : null, sense: k < 20 ? 2.5 : null })) })) });
const CHATS = { dose: 0.4, turns: [{ conv: "A casual chat", n: 1, of: 6, user: "Tell me about your day. ".repeat(10), plain: { text: "Fine. ".repeat(200), style: 0, sense: 3, good: false },
	steered: { text: "Wonderful! ".repeat(200), style: 2, sense: 3, good: true } }] };

function dockFor(m: Model, rows: number, n: number, done = true) {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r", model: m.name });
	for (let i = 0; i < n; i++) onEvent(live, tok(m, i));
	if (done) onEvent(live, { type: "end", rid: "r", n_tokens: n, trace_id: "T" });
	const grid = { cols: 32, rows: m.E / 32, pos: Array.from({ length: m.E }, (_, e) => e), num_experts: m.E, n_layers: m.layers };
	const host: any = { live, captions: false, close() {}, unfocus() {},
		fetchJson: async (url: string) => (url.includes("/experts/grid") ? grid : null), steerModel: () => m.name,
		steering: async () => STEER(m.name), steerChats: async () => CHATS, steer: async () => [] };
	const d = new Dock({ terminal: { rows }, requestRender() {} }, piTheme, host);
	return { d, host, live };
}

// The whole matrix takes ~30 s: BOARD_SIZES=all runs it (build3/tall/stress.sh); the suite runs its corners.
const ALL = process.env.BOARD_SIZES === "all";
const HEIGHTS = ALL ? [36, 37, 40, 44, 51, 60, 75, 100, 125, 160] : [36, 40, 51, 125], WIDTHS = ALL ? [100, 104, 130] : [100, 130];

test("board sizes: every height and width, both models, each panel in focus, full size or not: the dock's lines exactly, each its width", async () => {
	for (const m of MODELS) for (const rows of HEIGHTS) {
		const { d, host } = dockFor(m, rows, 50);
		d.render(104);
		await tick();
		d.steerPage = true;                                                          // the Steer page's test chats asked for, and here
		d.render(104);
		await tick();
		assert.ok(text(d.render(104)).some((l) => /^EXAMPLES /.test(l)), "the test chats drawn");
		d.steerPage = false;
		for (const width of WIDTHS) {
			const check = (what: string) => framed(d.render(width), rows, width, `${m.name} ${width}x${rows} ${what}`);
			for (const inspecting of ALL ? [false, true] : [false]) {
				d.sel = inspecting ? 30 : -1;
				for (const p of PANELS) {
					host.panel = p;
					check(`${p} in focus`);
				}
				d.steerPage = true;
				check("the Steer page");
				d.steerPage = false;
				for (const p of PANELS) {
					host.panel = p;
					d.full = true;
					check(`${p} full size`);
					d.full = false;
				}
			}
		}
	}
});

test("the size check fails on a frame with a panel that threw, or whose lines could not be worked out", async () => {
	// The board's columns; TOKENS full size; the one-panel board's LENS; the lines a panel needs.
	for (const [method, rows, width, full] of [["panelLines", 51, 104, false], ["fullLines", 51, 104, true], ["lensPanel", 30, 60, false], ["needOf", 51, 104, false]] as const) {
		const { d, host } = dockFor(MODELS[0], rows, 50);
		host.panel = full ? "TOKENS" : "LENS";
		d.full = full;
		d.render(width);
		await tick();
		framed(d.render(width), rows, width, "as it is");
		const was = (d as any)[method].bind(d);
		(d as any)[method] = (p: Panel, ...a: any[]) => { if (method === "lensPanel" || p === "TOKENS" || p === "LENS") throw new Error("boom"); return was(p, ...a); };
		assert.throws(() => framed(d.render(width), rows, width, method), /the dock: boom/, method);
	}
});

// The one-panel board, below the board's size: down to DOCK_MIN_WIDTH by DOCK_MIN_ROWS; below that one line.
const ONE_WIDTHS = ALL ? [44, 45, 50, 60, 70, 80, 90, 99, 104] : [44, 60, 99], ONE_HEIGHTS = ALL ? [23, 24, 26, 30, 35, 44, 60] : [23, 30, 60];

test("one-panel sizes: each panel in focus, a token held or following, engram inspection on, the EXPERTS pages, the Steer page: the dock's lines exactly, each its width", async () => {
	for (const m of ALL ? MODELS : [MODELS[1]]) for (const rows of ONE_HEIGHTS) {
		const { d, host, live } = dockFor(m, rows, 50);
		const ids = Array.from({ length: 49 }, (_, i) => 100 + i), lp = ids.map((_, i) => (i % 3 ? -1 : -3));
		host.caps = () => ["lookup"];
		host.lookup = async () => ({ trace: "T", ngram: 3, ids, lp_on: ids.map(() => -1), lp_off: lp, top_ids_on: ids.map((id) => [id, 7]), top_lp_on: ids.map(() => [-1, -2]),
			top_ids_off: ids.map((id) => [7, id]), top_lp_off: lp.map((x) => [-0.5, x]), vocab: { 7: " seven" } });
		d.render(60);
		await tick();
		for (const width of ONE_WIDTHS) {
			if (width >= 100 && rows >= 36) continue;                                    // the board
			const check = (what: string) => {
				const at = `${m.name} ${width}x${rows} ${what}`, lines = text(framed(d.render(width), rows, width, at));
				assert.ok(lines.some((l) => /^ ?1 TOKENS +2 MAP /.test(l)) || d.steerPage || (d as any).experts.xp.at !== "grid", `${at}: the tab row`);
				return lines;
			};
			for (const sel of [30, -1]) for (const g of [false, true]) {
				d.sel = sel;
				if (g !== !!host.lookupOn) {
					d.stepIn(30);
					d.handleInput("g");
					await tick();
					d.sel = sel;
				}
				for (const p of PANELS) {
					host.panel = p;
					const lines = check(`${p}${sel < 0 ? " following" : " held"}${g ? " engram" : ""}`);
					assert.equal(lines.some((l) => l.startsWith("ENGRAM INSPECTION")), g, `${width}x${rows} ${p}: the engram's title`);
				}
			}
			// EXPERTS' layer and expert pages (enter, enter), then back.
			d.sel = 30;
			host.panel = "EXPERTS";
			d.handleInput("\r");
			assert.equal((d as any).experts.xp.at, "layer");
			check("EXPERTS' layer page");
			d.handleInput("\r");
			assert.equal((d as any).experts.xp.at, "expert");
			check("EXPERTS' expert page");
			d.handleInput("\x1b");
			d.handleInput("\x1b");
			d.sel = 30;
			d.steerPage = true;
			d.render(width);
			await tick();
			check("the Steer page");
			d.steerPage = false;
		}
		// Below the least: the mini dock (under 23 rows), else one line.
		for (const [width, h, say] of [[43, rows, "44x23"], [60, 22, ""], [44, 22, ""], [44, 12, ""]] as const) {
			(d as any).tui.terminal.rows = h;
			const out = framed(d.render(width), h, width, `${width}x${h}`);
			if (say) assert.equal(text(out)[0], `dock too small: needs ${say}`, `${width}x${h}`);
			else assert.match(text(out).join("\n"), /^[▸ ] layer \d+ of \d+$/m, `${width}x${h}: the mini dock`);
			(d as any).tui.terminal.rows = rows;
		}
		assert.ok(live.reply);
	}
});

/** Where each panel's title is, and EXPERTS' title (its layer count). */
const layout = (lines: string[]) => {
	const at: Record<string, number> = {};
	lines.forEach((l, i) => { for (const p of PANELS) if (new RegExp(`(^| │ )${p}  `).test(l) && at[p] === undefined) at[p] = i; });
	return { at, experts: lines.find((l) => / │ EXPERTS {2}\d+ layers?/.test(l))?.replace(/^.* │ /, "") };
};

/** The reply strip's lines: from its title to the rule under it. */
const replyLines = (lines: string[]) => { const r = lines.indexOf("REPLY"); return lines.findIndex((l, i) => i > r && /^─+$/.test(l)) - r - 1; };

/** Each part's lines: the reply strip's, MAP's (its title to the rule over LENS), and where each panel's title is. */
const heightsOf = (lines: string[]) => {
	const map = lines.findIndex((l) => l.startsWith("MAP  ")), lens = lines.findIndex((l) => /^ ?LENS /.test(l));
	return { reply: replyLines(lines), map: lens - 1 - map, ...layout(lines) };
};

test("board sizes: a reply streaming 960 tokens changes no panel's height: the reply strip, MAP, every title where it was", async () => {
	for (const m of MODELS) for (const rows of [75, 120]) {
		const { d, live } = dockFor(m, rows, 1, false);
		d.render(104);
		await tick();
		const first = heightsOf(text(d.render(104)));
		for (let i = 1; i < 960; i++) {
			onEvent(live, tok(m, i));
			if (i % 7) continue;
			assert.deepEqual(heightsOf(text(d.render(104))), first, `${m.name} at ${rows} rows, token ${i}`);
		}
		assert.ok(first.map >= 3, `MAP ${first.map}`);
	}
});

test("board sizes: MAP's height and the reply strip's from the dock, not the reply: the left column full at 104x78", async () => {
	const m = MODELS[0];
	const at = async (rows: number, n: number) => {
		const { d } = dockFor(m, rows, n);
		d.render(104);
		await tick();
		return heightsOf(text(d.render(104)));
	};
	// 78 lines: the middle 66 under the strip at its least. The left: TOKENS 7 (5 choices and a row for the written
	// token), a rule, MAP, a rule, LENS at its most (36 for 40 read layers: SETTLING 6, a blank, the telescope 28); the
	// right: EXPERTS, a layer of 5 at a time. The left spares 66 - 7 - 1 - 3 - 1 - 36 = 18: the strip 6 + 18, MAP its
	// least 3 and nothing left over on the left.
	for (const n of [10, 50, 900]) assert.deepEqual([(await at(78, n)).reply, (await at(78, n)).map], [24, 1 + MAP_LINES], `${n} tokens`);
	// 120 lines: the strip a third of the dock (40), MAP 1 + 7 (672 tokens), the rest blank under LENS.
	assert.deepEqual([(await at(120, 50)).reply, (await at(120, 50)).map], [40, 1 + MAP_CELL_LINES]);
	assert.deepEqual((await at(51, 900)).reply, 6, "51 lines: nothing spare");
	assert.equal((await at(160, 50)).reply, 53, "160 lines: a third of the dock");
});

test("board sizes: the left column at 63, 50, 45, 40 and 36 lines: MAP its least (2 lines), the rest under TOKENS to LENS; short, SETTLING goes, then TOKENS' last choice, before the telescope's sixth row of tube", async () => {
	const room = async (m: Model, rows: number) => {
		const { d } = dockFor(m, rows, 50);
		d.render(104);
		await tick();
		const left = text(d.render(104)).map((l) => l.split(" │ ")[0].trimEnd());
		const map = left.findIndex((l) => l.startsWith("MAP  ")), lens = left.findIndex((l) => /^ ?LENS /.test(l)), st = left.findIndex((l) => l.startsWith("SETTLING"));
		return { map: lens - 2 - map, settle: st < 0 ? 0 : left.slice(st + 1).findIndex((l) => !/^L\d+/.test(l)), tube: left.filter((l) => /^│[ ▸]L\d+/.test(l)).length,
			fold: Number(/… (\d+) shallow/.exec(left.join("\n"))?.[1] ?? 0), choices: left.filter((l) => /^[▸ ] " (w|a)\d+"/.test(l)).length };
	};
	// [TOKENS' choices, MAP's cell lines, SETTLING's lines of layers, the tube's rows, layers folded]. TOKENS its 5
	// choices over MAP; at 40 SETTLING its sub-heading and a line, at 36 none and TOKENS 4 choices, the tube 6 rows (5 at
	// 36, MAP's second line kept).
	const want: Record<string, number[][]> = {
		forty: [[5, 2, 5, 9, 0], [5, 2, 3, 9, 27], [5, 2, 3, 7, 33], [5, 2, 1, 6, 34], [4, 2, 0, 5, 35]],
		fortyeight: [[5, 2, 5, 11, 0], [5, 2, 3, 11, 37], [5, 2, 3, 7, 41], [5, 2, 1, 6, 42], [4, 2, 0, 5, 43]] };
	for (const m of MODELS)
		for (const [i, rows] of [63, 50, 45, 40, 36].entries()) {
			const r = await room(m, rows);
			assert.deepEqual([r.choices, r.map, r.settle, r.tube, r.fold], want[m.name][i], `${m.name} at ${rows}`);
		}
});

test("TOKENS: its first 5 choices in the board's column, all 10 full size and on the one-panel board", async () => {
	const choices = (ls: string[]) => ls.map((l) => l.split(" │ ")[0]).filter((l) => /^[▸ ] " (w|a)\d+"/.test(l)).length;
	const { d, host } = dockFor(MODELS[0], 63, 50);
	d.render(104);
	await tick();
	assert.equal(choices(text(d.render(104))), 5, "the board's column");
	host.panel = "TOKENS";
	d.full = true;
	assert.equal(choices(text(d.render(104))), 10, "full size");
	const one = dockFor(MODELS[0], 30, 50);
	one.host.panel = "TOKENS";
	one.d.render(104);
	await tick();
	assert.equal(choices(text(one.d.render(104))), 10, "the one-panel board");
});

test("board sizes: MAP two lines at 49 lines (a token half a character), the telescope folding more for them; two lines more unfold four layers (two a row there), MAP still two lines", async () => {
	const m = MODELS[0];
	const at = async (rows: number) => {
		const { d } = dockFor(m, rows, 900);
		d.render(104);
		await tick();
		const lines = text(d.render(104)), map = lines.findIndex((l) => l.startsWith("MAP  ")), lens = lines.findIndex((l) => /^ ?LENS /.test(l));
		const fold = /… (\d+) shallower layers?/.exec(lines.join("\n"));
		return { map: lens - map - 2, fold: fold ? Number(fold[1]) : 0, reply: replyLines(lines), cells: lines[map + 1].split(" │ ")[0] };
	};
	const tight = await at(49), more = await at(51);
	assert.deepEqual([tight.map, more.map, tight.reply, more.reply], [MAP_LINES, MAP_LINES, 6, 6], "MAP its least, the strip its least");
	// The two lines went to the telescope: two rows of its second section, two layers a row there ("L30 +1").
	assert.deepEqual([tight.fold, more.fold], [29, 25], `${tight.fold} then ${more.fold} layers folded`);
	assert.equal(tight.cells, "▀".repeat(48), "a token half a character, two rows of 48 a line");
});

test("board sizes: the height goes to content; EXPERTS more layers, SETTLING more layers, the window more grids", async () => {
	const count = (m: Model, rows: number) => {
		const { d, host } = dockFor(m, rows, 50);
		d.render(104);
		return tick().then(() => {
			const board = text(d.render(104));
			const experts = Number(/ │ EXPERTS {2}(\d+) layers?/.exec(board.find((l) => / │ EXPERTS {2}\d/.test(l))!)![1]);
			const s = board.findIndex((l) => l.startsWith("SETTLING")), settling = board.slice(s + 1).findIndex((l) => !/^L\d+ /.test(l));
			host.panel = "EXPERTS";
			d.full = true;
			const head = text(d.render(104)).find((l) => /^layers? \d+(-\d+)? of /.test(l))!;
			const [, a, b] = /^layers? (\d+)(?:-(\d+))? of/.exec(head)!;
			return { experts, settling, window: Number(b ?? a) - Number(a) + 1 };
		});
	};
	const m = MODELS[0];
	const at51 = await count(m, 51), at75 = await count(m, 75), at120 = await count(m, 120), at160 = await count(m, 160);
	assert.equal(at51.window, 9, "51 lines: three whole rows of three");
	assert.ok(at75.window > at51.window && at120.window > at75.window, `${at51.window} ${at75.window} ${at120.window}`);
	assert.equal(at160.window, 40, "tall enough: every layer");
	assert.ok(at120.experts > at75.experts && at75.experts >= 3, `${at75.experts} ${at120.experts}`);
	assert.ok(at75.settling > at51.settling && at120.settling >= at75.settling, `${at51.settling} ${at75.settling} ${at120.settling}`);
	assert.equal(at160.settling, SETTLE_MOST, "at its most: SETTLE_MOST lines, the 40 layers eight a line");
});

/** Both lenses on every token, reading other layers: the J-lens 12 up, the logit lens every layer but the deepest, its
 * words longer; `depth` false: no layer count sent (the depth from the layers read). */
function twoLenses(live: Live, depth: boolean) {
	for (const t of live.reply!.tokens as any[]) {
		t.lens_logit = t.lens.filter(([l]: number[]) => l < t.n_layers - 1).map(([l, id, w]: [number, number, string]) => (id === t.id && l < t.n_layers - 3 ? [l, 8, " averylongword"] : [l, id, w]));
		t.lens = t.lens.filter(([l]: number[]) => l >= 12);
		if (!depth) delete t.n_layers;
	}
}
/** A frame's geometry: on each line where the boxes and the column rules are, the line of each title (the panels',
 * the telescope's token line and eyepiece), SETTLING's row labels, and the last `footer` lines (the key line). */
const shape = (lines: string[], footer = 1) => ({
	rules: lines.map((l) => [...l].flatMap((c, i) => ("│╭╮╰╯".includes(c) ? [i] : [])).join(",")),
	titles: lines.flatMap((l, i) => (/^(REPLY|MAP |SETTLING |#\d+ ")|^ ?LENS |╭─ out of| │ (TOKENS|EXPERTS) /.test(l) ? [i] : [])),
	settling: lines.filter((l) => /^L\d+ +▀/.test(l)).map((l) => l.slice(0, 5)),
	footer: lines.slice(-footer),
});
const SIZES = [[100, 51], [104, 78], [130, 125]];

test("board sizes: s swaps the lens and nothing moves: every frame of the swap, the board, LENS full size and the one-panel board", async () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		for (const depth of [true, false]) for (const [width, rows] of [...SIZES, [80, 60]]) for (const full of width < 100 ? [false] : [false, true]) {
			const { d, host, live } = dockFor(MODELS[0], rows, 50);
			twoLenses(live, depth);
			d.render(width);
			await tick();
			d.sel = 30;
			host.panel = "LENS";
			d.full = full;
			// SETTLING's rows are the layers the lens shown read (fewer for the J-lens here): the rest of the frame stays.
			const frame = (lines: string[]) => ({ ...shape(lines), settling: undefined });
			const was = frame(text(d.render(width)));
			assert.ok(was.titles.length >= 2, `${width}x${rows}: ${was.titles}`);
			for (const view of ["logit", "jlens"]) {
				d.handleInput("s");
				assert.equal(host.lensView, view);
				for (let f = 0; f < 10; f++, mock.timers.tick(70))
					assert.deepEqual(frame(text(d.render(width))), was, `${width}x${rows}${full ? " full size" : ""}${depth ? "" : " no depth"}: to ${view}, frame ${f}`);
				const read = new Set((view === "logit" ? live.reply!.tokens[30].lens_logit! : live.reply!.tokens[30].lens!).map(([l]) => l));
				const got = shape(text(d.render(width))).settling.map((l) => Number(l.slice(1)));
				assert.ok((got.length || width < 100) && got.every((l) => read.has(l)), `${width}x${rows}: ${view}'s rows only the layers it read: ${got}`);
			}
		}
	} finally {
		mock.timers.reset();
	}
});

test("board sizes: a LENS word edited moves nothing: entering, typing, a word filled in, removed, cancelled, applied", async () => {
	for (const [width, rows] of [...SIZES, [80, 60]]) for (const captions of [false, true]) {
		const { d, host, live } = dockFor(MODELS[0], rows, 50);
		twoLenses(live, true);
		d.render(width);
		await tick();
		d.sel = 30;
		host.panel = "LENS";
		host.captions = captions;
		const footer = (lines: string[]) => [...lines].reverse().findIndex((l) => /^─+$/.test(l));   // the lines under the last rule
		const at = (lines: string[]) => { const s = shape(lines, footer(lines)); return { ...s, footer: s.footer.length }; };
		const was = at(text(d.render(width)));
		for (const keys of ["\r", "x", "\x7f", "\x7f", "averyveryverylongreplacementword", "\x1b[A", "\x1b[B", "\t", "\x1b", "\r", "y", "\r"]) {
			for (const c of keys === "\x1b[A" || keys === "\x1b[B" ? [keys] : keys) d.handleInput(c);
			const lines = text(d.render(width));
			if (keys === "x") assert.ok(lines.some((l) => /│▸w30 → x /.test(l)), `${width}x${rows}: the field drawn`);
			assert.deepEqual(at(lines), was, `${width}x${rows}${captions ? " captions" : ""}: after ${JSON.stringify(keys)}`);
		}
	}
});

/** The oMLX Qwen3.6-35B shape: 40 layers, 256 experts (grids 32 by 8), a J-lens of 13 layers (SETTLING at its most in 8 lines). */
const SHORT_LENS: Model = { name: "thirteen read", L: 40, E: 256, k: 8, layers: 40, read: [0, 4, 8, 12, 16, 20, 24, 28, 31, 32, 36, 38, 39] };

test("mini sizes: the token's line, EXPERTS' title, as many whole grids as fit, the maps under them, the key line; a blank line only over a part, under the key line less than a grid, 12 to 120 lines", async () => {
	for (const m of [...MODELS, SHORT_LENS]) {
		const { d, host } = dockFor(m, 12, 50);
		host.mini = true;
		d.render(MINI_WIDTH);
		await tick();
		const block = 1 + Math.ceil(reshapeGrid(m.E, gridShape(m.E), RIGHT_W).rows / 2);   // a heading and the grid's half-block lines, the right column's
		let most = 0;
		for (let rows = 12; rows <= 120; rows++) {
			(d as any).tui.terminal.rows = rows;
			const what = `${m.name} ${rows} lines`, lines = text(framed(d.render(MINI_WIDTH), rows, MINI_WIDTH, what));
			assert.match(lines[0], /^#49 " w49" /, `${what}: the token's line`);
			const key = lines.indexOf(rows < DOCK_MIN_ROWS ? "ctrl+\\ steps in · [ ]" : "ctrl+\\ steps in, then m · [ ]");   // no m where the full dock does not fit
			assert.ok(key > 0, what);
			const blanks = lines.slice(1, key).flatMap((l, i) => (l === "" ? [i + 1] : []));
			assert.ok(blanks.every((i) => /^(MAP|SETTLING|RESIDUAL STREAM)( |$)/.test(lines[i + 1])), `${what}: a blank line only over a part\n${lines.join("\n")}`);
			assert.ok(lines.slice(key + 1).every((l) => l === "") && rows - 1 - key < block, `${what}: under the key line, less than a grid`);
			const grids = lines.filter((l) => /^[▸ ] layer \d+ of \d+$/.test(l)).length, end = lines.findIndex((l) => /^(MAP|SETTLING)( |$)/.test(l));
			assert.match(lines[1], /^EXPERTS {2}\d+ layers? · /, `${what}: EXPERTS' title`);
			const under = end < 0 ? key : end - +(lines[end - 1] === "");
			assert.equal(under - 2, grids * block, `${what}: whole grids, the maps under them`);
			most = grids;
		}
		assert.ok(most > 8, `${m.name}: ${most} grids at 120 lines, past the old eight`);
	}
});

test("mini dock: the maps the right column's width, with routing or without", async () => {
	for (const routed of [true, false]) {
		const { d, live } = dockFor(SHORT_LENS, 60, 50);
		if (!routed) for (const t of live.reply!.tokens as any[]) delete t.experts, delete t.expert_w, delete t.expert_k;
		live.dense = routed ? undefined : "thirteen read";
		(d as any).host.mini = true;
		d.render(MINI_WIDTH);
		await tick();
		const lines = text(d.render(MINI_WIDTH)), map = lines.findIndex((l) => l.startsWith("MAP ")), settle = lines.findIndex((l) => l.startsWith("SETTLING"));
		assert.ok(map > 0 && settle > map, lines.join("\n"));
		const widest = Math.max(...lines.slice(map + 1, settle).map((l) => visibleWidth(l)), ...lines.slice(settle + 1, -1).map((l) => visibleWidth(l)));
		assert.equal(widest, RIGHT_W, `${routed ? "routed" : "no routing"}: ${widest}`);
	}
});

test("mini dock: the lights always the glow (the history view); stepped to an older token, the glow as it stood there", async () => {
	const grids = (d: Dock) => { const raw = d.render(MINI_WIDTH), lines = text(raw), end = lines.findIndex((l) => /^(MAP|SETTLING)( |$)/.test(l)); return raw.slice(1, end).map((l) => l.replace(/\x1b\[[0-9;]*m[▌▐]/, "")); };   // the edge: where the keyboard is
	for (const done of [true, false]) {
		const { d, host } = dockFor(SHORT_LENS, 60, 50, done);
		host.mini = true;
		d.render(MINI_WIDTH);
		await tick();
		const unfocused = grids(d);
		d.stepIn(-1);
		assert.deepEqual(grids(d), unfocused, `${done ? "done" : "streaming"}: stepped in, following`);
		d.handleInput("\x1b[D");
		assert.equal(d.sel, 48);
		assert.notDeepEqual(grids(d), unfocused, "paused: the glow at #48");
		d.handleInput("f");
		assert.equal(d.sel, -1);
		assert.deepEqual(grids(d), unfocused, "f: the newest glow again");
	}
	// Paused on #20: the lights as a reply 21 tokens in, still streaming, showed them then.
	const { d, host } = dockFor(SHORT_LENS, 60, 50), then = dockFor(SHORT_LENS, 60, 21, false);
	for (const x of [host, then.host]) x.mini = true;
	d.render(MINI_WIDTH);
	then.d.render(MINI_WIDTH);
	await tick();
	d.stepIn(20);
	assert.match(text(d.render(MINI_WIDTH))[0], /^#20 " w20" /, "the token line: the cursor's token");
	assert.deepEqual(grids(d), grids(then.d));
	d.handleInput("\x1b[C");
	d.handleInput("\x1b[D");
	assert.deepEqual(grids(d), grids(then.d), "forward and back: the same");
});

test("mini dock: a followed reply ends, the lights, the token line and the key line as at its last token, stepped in or not (as the board)", async () => {
	for (const [how, sel] of [["not stepped in", undefined], ["stepped in", -1], ["paused on the newest token", 49]] as const) {
		const { d, host, live } = dockFor(SHORT_LENS, 60, 50, false);
		host.mini = true;
		d.render(MINI_WIDTH);
		await tick();
		if (sel === -1) d.stepIn(-1);
		else if (sel !== undefined) d.sel = sel;
		const before = d.render(MINI_WIDTH);
		assert.ok(before.some((l) => l.includes("▀")));
		onEvent(live, { type: "end", rid: "r", n_tokens: 50, trace_id: "T" });
		assert.deepEqual(d.render(MINI_WIDTH), before, how);
		assert.equal(d.sel, sel ?? -1, how);
	}
});

test("the board's EXPERTS, every view: the glow while following, streaming or done, stepped in or not; paused on a token (the newest too), its own experts", async () => {
	const views: [string, number, Record<string, any>, boolean][] = [["the column", 104, {}, false],
		["full size", 104, { panel: "EXPERTS" }, true], ["the one-panel board", 60, { panel: "EXPERTS" }, false]];
	for (const done of [true, false]) for (const [what, w, extra, full] of views) {
		const { d, host } = dockFor(SHORT_LENS, 60, 50, done);
		Object.assign(host, extra);
		d.full = full;
		d.render(w);
		await tick();
		// The lights alone: right of the last column rule (the left column's SETTLING marks the token in focus).
		const grids = () => d.render(w).filter((l) => l.includes("▀")).map((l) => l.replace(/\x1b\[[0-9;]*m[▌▐]/, "")).map((l) => l.slice(Math.max(0, l.lastIndexOf("│"))));   // the edge: where the keyboard is
		const following = grids();
		assert.ok(following.length > 0, what);
		d.stepIn(-1);
		assert.deepEqual(grids(), following, `${what}, ${done ? "done" : "streaming"}: stepped in, as before stepping in`);
		d.sel = 49;                                                                  // paused on the newest: its own experts
		assert.notDeepEqual(grids(), following, `${what}, ${done ? "done" : "streaming"}: paused on the newest, its own experts`);
		d.handleInput("\x1b[D");
		assert.equal(d.sel, 48);
		assert.notDeepEqual(grids(), following, `${what}: paused, the token's own experts`);
		d.handleInput("f");
		assert.deepEqual(grids(), following, `${what}: f, following again`);
	}
});

test("focus changes no geometry: the board and the mini dock, stepped in or not, at several sizes", async () => {
	// Where the boxes, the column rules and the titles are: the same unfocused and stepped in (ctrl+\: the newest,
	// followed), or paused on a token. The words and the marks may change, never where things are. (The pages are laid
	// out for what stepping in shows: not pinned here.)
	const TITLE = /^(REPLY|MAP|LENS|TOKENS|EXPERTS|SETTLING|L\d+)\b/;
	const skel = (line: string) => {
		// Rules: a whole line of them, a column's (beside " │ "), a box's edges; any other ─ is content.
		const l = line.replace(/▸/g, " "), rules = /^─+$/.test(l.trim()) || /[╭╮╰╯]/.test(l);
		const cs = [...(rules ? l : l.replace(/─+(?! │)/g, (s, at) => (l.slice(at - 2, at) === "│ " ? s : " ".repeat(s.length))))];
		let out = "";
		for (let i = 0; i < cs.length; i++) {
			const t = (i === 0 || cs[i - 1] === " ") && (i <= 1 || cs[i - 2] === "│") ? TITLE.exec(cs.slice(i).join("")) : null;
			if (t) { out += t[1]; i += t[1].length - 1; continue; }
			out += "│╭╮╰╯─".includes(cs[i]) ? cs[i] : " ".repeat(visibleWidth(cs[i]));
		}
		return out.replace(/╰─*\s*─*╯/g, (s) => "╰" + " ".repeat(visibleWidth(s) - 2) + "╯").trimEnd();   // the eyepiece's foot names the token that went in
	};
	const first = (live: Live) => Math.max(0, live.reply!.tokens.findIndex((t) => (t.p ?? 0) < 0.5));
	const cases: [string, number, number, Record<string, any>][] = [];
	for (const [w, h] of [[104, 51], [130, 75], [200, 60], [100, 120]]) {
		cases.push(["board", w, h, {}]);
		cases.push(["board, ? left on", w, h, { captions: true }]);
	}
	for (const h of [12, 30, 51, 90]) cases.push(["mini", MINI_WIDTH, h, { mini: true }]);
	for (const m of [...MODELS, SHORT_LENS]) for (const [what, w, h, extra] of cases) for (const at of ["an unsure token", "newest"]) {
		const a = dockFor(m, h, 50), b = dockFor(m, h, 50);
		for (const x of [a, b]) {
			Object.assign(x.host, extra);
			x.d.render(w);
		}
		await tick();
		const unfocused = text(a.d.render(w)).map(skel);
		b.d.render(w);
		b.d.stepIn(at === "newest" ? -1 : first(b.live));
		assert.deepEqual(text(b.d.render(w)).map(skel), unfocused, `${m.name} ${what} ${w}x${h}, stepped in, on ${at === "newest" ? "the newest" : at}`);
	}
});

test("focus changes no geometry on the one-panel board: each panel, following or held, a steer on or off; nothing moves", async () => {
	// The tab row and the rule over the keys: where they are whatever the focus, the token held or followed, a steer on
	// or not (the keys always two lines); a panel's own rules: the same followed or held.
	const rules = (lines: string[]) => lines.map((l, i) => (/^─+$/.test(l) ? i : -1)).filter((i) => i >= 0);
	const geo = (lines: string[]) => [lines.findIndex((l) => /^ ?1 TOKENS +2 MAP /.test(l)), rules(lines).at(-1) ?? -1];
	const sizes = ALL ? [[44, 23], [44, 30], [60, 24], [60, 35], [80, 30], [99, 40], [104, 30], [130, 35]] : [[44, 23], [60, 30], [104, 30]];
	for (const m of [MODELS[0], SHORT_LENS]) for (const [w, h] of sizes) {
		let want: number[] | undefined;
		for (const steer of [false, true]) for (const p of PANELS) {
			const { d, host, live } = dockFor(m, h, 50);
			if (steer) live.steer = { parts: [{ name: "joyful", strength: 0.4, layer: 20, source: "profile" }] } as any;
			host.panel = p;
			d.render(w);
			await tick();
			const what = `${m.name} ${w}x${h} ${p}${steer ? " steered" : ""}`;
			const first = text(d.render(w)), own = rules(first);
			want ??= geo(first);
			assert.deepEqual(geo(first), want, `${what}, unfocused`);
			d.stepIn(-1);
			assert.deepEqual(geo(text(d.render(w))), want, `${what}, following`);
			assert.deepEqual(rules(text(d.render(w))), own, `${what}, following: its rules`);
			d.handleInput("\x1b[D");
			assert.deepEqual(geo(text(d.render(w))), want, `${what}, held`);
			assert.deepEqual(rules(text(d.render(w))), own, `${what}, held: its rules`);
		}
	}
});

test("one-panel keys at 44 columns: every part kept is drawn, each panel and state; ? and an edit, a note: on the keys' lines", async () => {
	const keyArea = (lines: string[]) => lines.slice(lines.length - 1 - [...lines].reverse().findIndex((l) => /^─+$/.test(l)) + 1);
	for (const p of PANELS) for (const held of [false, true]) for (const steer of [false, true]) {
		const { d, host, live } = dockFor(MODELS[0], 23, 50);
		if (steer) live.steer = { parts: [{ name: "joyful", strength: 0.4, layer: 20, source: "profile" }] } as any;
		host.panel = p;
		d.render(44);
		await tick();
		d.stepIn(30);
		if (held) d.handleInput("\x1b[D");
		const keys = keyArea(text(d.render(44))), what = `${p}${held ? " held" : ""}${steer ? " steered" : ""}: ${keys.join(" / ")}`;
		assert.equal(keys.length, 2, what);
		assert.ok(keys.every((l) => !l.endsWith("…")), what);
		const parts = keys.join(" · ").split(" · ");
		assert.match(keys[0], /^tab \w+ · /, what);
		for (const k of [...(held ? ["f live"] : []), ...(p === "LENS" ? ["←/→ token", "↑/↓ layer", ...(steer ? ["x off"] : [])] : [])])
			assert.ok(parts.includes(k), `${k} in ${what}`);
	}
	// ? at 23 rows: the explanation, then the layout's keys whole ("? keys" seen); a word edited: its keys, last.
	const { d, host } = dockFor(MODELS[0], 23, 50);
	host.panel = "LENS";
	d.render(44);
	await tick();
	d.stepIn(30);
	host.captions = true;
	let lines = text(d.render(44));
	assert.deepEqual(lines.slice(-2), ["", layoutKeys()], "on the key lines' two: no numbers (the tab row's)");
	assert.ok(lines.at(-3)!.length > 0, "the explanation over them");
	host.captions = false;
	d.handleInput("\r");                                                              // a word edited
	const edit = text(d.render(44)).slice(-2);
	assert.deepEqual(edit, ["⌫ remove · type or ↑/↓ a word · tab next", "enter apply, redo from here · esc cancel"]);
	host.captions = true;
	lines = text(d.render(44));
	assert.deepEqual(lines.slice(-2), edit, "? keeps the edit's keys");
	assert.ok(lines.at(-3)!.length > 0, "the explanation over them");
	host.captions = false;
	d.handleInput("\x1b");
});

test("one-panel board across the threshold: a full-size panel kept for the board again; an EXPERTS page closed in a column is not opened again; the strip centres the token", async () => {
	const { d, host } = dockFor(MODELS[0], 40, 50);
	const term = (d as any).tui.terminal;
	d.render(104);
	await tick();
	d.stepIn(30);
	d.handleInput("1");                                                                 // TOKENS full size on the board
	assert.equal(d.full, true);
	term.rows = 30;
	d.render(104);                                                                      // the one-panel board: kept
	assert.equal(d.full, true);
	term.rows = 40;
	assert.ok(text(d.render(104)).some((l) => / 1 TOKENS .* 1 back/.test(l)), "full size again");
	d.handleInput("1");
	assert.equal(d.full, false);
	// EXPERTS' layer page on the one-panel board, then the board: EXPERTS in its column, its page not opened again.
	term.rows = 30;
	host.panel = "EXPERTS";
	d.render(104);
	d.handleInput("\r");
	assert.equal((d as any).experts.xp.at, "layer");
	term.rows = 40;
	const back = text(d.render(104));
	assert.ok(!back.some((l) => /^EXPERTS ›/.test(l)), back.join("\n"));
	d.handleInput("4");
	assert.ok(!text(d.render(104)).some((l) => /^EXPERTS ›/.test(l)), "full size later: its grid, not the old page");
	d.handleInput("4");
	// The one-panel strip's three lines: the token on the middle one.
	term.rows = 30;
	d.stepIn(30);
	const lines = text(d.render(60)), at = lines.indexOf("REPLY");
	const strip = lines.slice(at + 1, at + 4);
	assert.ok(strip[1].includes("w30"), strip.join("\n"));
});

test("board sizes: a LENS word edited, at 100 and 130 columns: the field inside the eyepiece, every line its width, the key line whole", async () => {
	for (const m of MODELS) for (const width of [100, 130]) {
		const { d, host } = dockFor(m, 51, 50);
		d.render(width);
		await tick();
		d.sel = 30;
		host.panel = "LENS";
		d.handleInput("\r");
		for (const keys of ["", "\x7f", "x".repeat(70) + "end"]) {
			for (const c of keys) d.handleInput(c);
			const lines = text(framed(d.render(width), 51, width, `${m.name} ${width}`));
			const box = lines.find((l) => /^╭─ out of L/.test(l))!, row = lines.find((l) => /^│▸/.test(l))!;
			assert.equal(row.indexOf("│", 1), box.indexOf("╮"), `${m.name} ${width}: ${row}`);
			assert.equal(lines.at(-1), "⌫ remove · type or ↑/↓ a word · tab next · enter apply, redo from here · esc cancel");
		}
	}
});

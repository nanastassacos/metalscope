import assert from "node:assert/strict";
import { test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { Dock } from "../dock.ts";
import { routedLayers, shownLayer } from "../expert.ts";
import {
	GLOW_DECAY, Glow, Grids, LEVELS, WARM_FLOOR, gridCells, gridShape, guessExperts, halfBlocks,
	STOPS, lit, palette, rampAt, rgbBlocks, tokenExperts,
} from "../lights.ts";
import { type Rgb, colourCode } from "../draw.ts";
import type { Live } from "../state.ts";

const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const COLOURS: Record<string, number> = { success: 2, muted: 8, dim: 7, accent: 6, warning: 3, error: 1 };
const theme = {
	fg: (c: string, s: string) => `\x1b[38;5;${COLOURS[c] ?? 15}m${s}\x1b[39m`,
	bg: (_c: string, s: string) => `\x1b[48;5;236m${s}\x1b[49m`,
	bold: (s: string) => `\x1b[1m${s}\x1b[22m`,
};

test("ramp: the dock's own amber, from a shade off its background to near-white, clamped", () => {
	assert.deepEqual(rampAt(STOPS, 0), [24, 27, 33]);
	assert.deepEqual(rampAt(STOPS, 0.45), [204, 122, 30]);
	assert.deepEqual(rampAt(STOPS, 2), [255, 245, 216]);                       // clamped
	assert.deepEqual(rampAt(STOPS, -1), [24, 27, 33]);
});

test("palette: 24-bit steps of the dock's ramp whatever Pi's theme; 256 colours where the terminal has no more", () => {
	const pi = { colors: { customMessageBg: { kind: "rgb", r: 40, g: 30, b: 60 }, accent: { kind: "rgb", r: 180, g: 120, b: 255 } }, getColorMode: () => "truecolor" };
	const p = palette(pi);
	assert.equal(p.fg.length, LEVELS);
	assert.equal(p.fg[LEVELS - 1], "\x1b[38;2;255;245;216m");                    // Pi's purple is not used
	assert.equal(palette(theme).bg[0], "\x1b[48;2;24;27;33m");
	assert.match(palette({ ...pi, getColorMode: () => "256color" }).fg[5], /^\x1b\[38;5;\d+m$/);
});

test("halfBlocks: two lights a character, upper in the foreground, lower in the background; blanks stay the panel's", () => {
	const pal = { fg: Array.from({ length: LEVELS }, (_, i) => `<f${i}>`), bg: Array.from({ length: LEVELS }, (_, i) => `<b${i}>`) };
	// 3 columns, 3 rows: the last line's lower half is past the rows (blank).
	const cells = [1, 0, null, 0, null, 0.5, 0, 1, null];
	const lines = halfBlocks(cells, 3, 3, pal);
	assert.equal(lines.length, 2);
	// A colour written only where it changes on the line; the background back to the panel's (ESC[49m) where a half or a
	// blank has none.
	const raw = (l: string) => l.replace(/\x1b\[0m/g, "|").replace(/\x1b\[49m/g, "~");
	assert.equal(raw(lines[0]), `<b0><f31>▀~<f0>▀<f15>▄|`.replace("<f15>", `<f${Math.round(0.5 * 31)}>`));
	assert.equal(raw(lines[1]), "<f0>▀<f31>▀ |");
	assert.ok(lines.every((l) => visibleWidth(plain(l).replace(/<[fb]\d+>/g, "")) === 3 && l.endsWith("\x1b[0m")));
});

test("Glow: a token's experts light by share and fade per token; busy experts keep a dim warm floor", () => {
	const g = new Glow(8);
	g.feed([[2, 0.6], [5, 0.03]]);
	assert.equal(g.glow[2], 1);                                               // 0.3 or more of the mix: full
	assert.equal(g.levels()[2], 1);                                           // and drawn so: the glow, not the floor alone
	assert.ok(Math.abs(g.glow[5] - lit(0.03)) < 1e-6 && lit(0.03) > 0.03);   // small shares lifted
	g.feed([]);
	assert.ok(Math.abs(g.glow[2] - GLOW_DECAY) < 1e-6);
	for (let i = 0; i < 200; i++) g.feed([]);
	const rest = g.levels();
	assert.ok(Math.abs(rest[2] - WARM_FLOOR) < 1e-6);                          // the busiest: the floor's top
	assert.ok(rest[5] > 0 && rest[5] < rest[2] && rest[0] === 0);
	const before = Float32Array.from(g.glow);
	g.feed([[9, 1], [-1, 1]]);                                                // out of range: ignored
	assert.ok(g.glow.every((x, e) => Math.abs(x - before[e] * GLOW_DECAY) < 1e-6));
});

test("tokenExperts: one layer's experts of a token's flattened routing, biggest share first", () => {
	const t = { experts: Uint16Array.from([1, 2, 3, 7, 8, 9]), expert_w: Uint8Array.from([51, 153, 51, 0, 200, 55]), expert_k: 3 };
	assert.deepEqual(tokenExperts(t, 0).map(([e, s]) => [e, Math.round(s * 255)]), [[2, 153], [1, 51], [3, 51]]);
	assert.deepEqual(tokenExperts(t, 1).map(([e]) => e), [8, 9]);              // a zero share is left out
	assert.deepEqual(tokenExperts(t, 2), []);
	assert.deepEqual(tokenExperts({}, 0), []);
	assert.equal(routedLayers([t, {}]), 2);
	assert.equal(routedLayers([{}]), 0);
	assert.equal(shownLayer(48), 24);
	assert.equal(shownLayer(48, 60), 47);
});

test("grid: the server's positions when it has them, else number order 32 across; cells no expert sits in are blank", () => {
	assert.deepEqual(gridShape(512), { cols: 32, rows: 16, pos: null });
	const server = { cols: 32, rows: 1, pos: [3, 0, 1, 2], num_experts: 4, n_layers: 2 };
	assert.equal(gridShape(4, server).pos, server.pos);
	assert.equal(gridShape(8, server).pos, null);                              // another size: not this grid
	const cells = gridCells([0.1, 0.2, 0.3, 0.4], { cols: 3, rows: 2, pos: [3, 0, 1, 2] });
	assert.deepEqual(cells, [0.2, 0.3, 0.4, 0.1, null, null]);
	assert.equal(guessExperts(500), 512);
	assert.equal(guessExperts(511), 512);
	assert.equal(guessExperts(2), 8);
});

test("Grids: asks once per layer, answers when loaded, asks again later only without an expert map", async () => {
	const asked: string[] = [];
	let t = 0, loads = 0;
	const answers: Record<string, any> = { 1: { cols: 32, rows: 16, pos: [0], num_experts: 1, n_layers: 2 }, 2: { cols: 32, rows: null, pos: null, num_experts: 512, n_layers: null } };
	const g = new Grids(async (url) => { asked.push(url); return answers[new URL(url).searchParams.get("layer")!]; }, () => loads++, () => t);
	assert.equal(g.at("http://h", "M", 1), undefined);
	assert.equal(g.at("http://h", "M", 1), undefined);                         // pending: not asked twice
	await new Promise((r) => setTimeout(r, 0));
	assert.equal(g.at("http://h", "M", 1)!.pos![0], 0);
	assert.deepEqual(asked, ["http://h/scope/api/experts/grid?layer=1&model=M"]);
	g.at("http://h", undefined, 2);
	await new Promise((r) => setTimeout(r, 0));
	assert.equal(g.at("http://h", undefined, 2)!.pos, null);
	t = Grids.RETRY_MS + 1;
	g.at("http://h", undefined, 2);                                              // no map then: asked again
	g.at("http://h", "M", 1);                                                    // a map: kept
	assert.equal(asked.length, 3);
	assert.equal(asked[1], "http://h/scope/api/experts/grid?layer=2");
	assert.ok(loads >= 2);
});

const text = (lines: string[]) => lines.map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd());

test("dock, EXPERTS: a reply without routing says why in one line, and loads its routing from its saved trace", async () => {
	const tok0 = { i: 0, id: 1, text: "Hi", p: 0.9, entropy: 0.1, alts: [], n_layers: null };
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true,
		reply: { rid: "r", tokens: [{ ...tok0 }, { ...tok0, i: 1, text: " there" }, { ...tok0, i: 2, text: "!" }], done: true, prefill: 0, model: "Moe" } };
	const asked: string[] = [];
	let answer: any = null;
	const fetchJson = async (url: string) => { asked.push(url); return url.includes("/routing") ? answer : null; };
	const d = new Dock({ terminal: { rows: 40 }, requestRender() {} }, theme, { live, captions: false, close() {}, unfocus() {}, panel: "EXPERTS", fetchJson });
	let lines = text(d.render(60));
	assert.ok(lines.includes("this reply was not saved: no experts to load"), lines.join("\n"));
	assert.equal(asked.length, 0);                                            // no trace: nothing asked
	live.reply!.traceId = "T 1";
	// Routing for reply tokens 0..1 (the last token is never fed back): 2 layers of top 2.
	const ids = Buffer.alloc(2 * 4 * 2), w = Buffer.alloc(2 * 4);
	[[5, 9, 7, 3], [8, 1, 2, 6]].forEach((row, t) => row.forEach((e, j) => { ids.writeUInt16LE(e, 2 * (t * 4 + j)); w[t * 4 + j] = j % 2 ? 55 : 200; }));
	answer = { start: 0, n: 2, L: 2, k: 2, ids: ids.toString("base64"), w: w.toString("base64") };
	lines = text(d.render(60));
	assert.ok(lines.includes("loading this reply's experts from its saved trace…"));
	assert.deepEqual(asked, ["http://h/scope/api/traces/T%201/routing?last=4096"]);
	await new Promise((r) => setTimeout(r, 0));
	const t = live.reply!.tokens;
	assert.deepEqual([...t[0].experts!], [5, 9, 7, 3]);
	assert.deepEqual([...t[1].expert_w!], [200, 55, 200, 55]);
	assert.equal(t[2].experts, undefined);
	lines = text(d.render(60));
	assert.ok(lines.some((l) => /^layers? [\d-]+ of 2 · /.test(l)), lines.join("\n"));
	const dk: any = d, E = dk.experts.expertsView(live.reply).E;
	assert.ok(dk.experts.levelsAt(live.reply, 0, E, 2).some((x: number) => x > 0), "following: the glow of the tokens loaded, though the newest has none");
	d.render(60);
	assert.equal(asked.filter((u) => u.includes("/routing")).length, 1);   // once per reply
	// A dense model's answer: said so, and the model taken for dense.
	live.reply = { rid: "r2", tokens: [{ ...tok0 }, { ...tok0, i: 1 }], done: true, prefill: 0, model: "Dense", traceId: "T2" };
	answer = { start: 0, n: 0, L: 0, k: 0, ids: "", w: "" };
	d.render(60);
	await new Promise((r) => setTimeout(r, 0));
	assert.equal(live.dense, "Dense");
	(d as any).host.panel = "EXPERTS";
	assert.ok(!text(d.render(60)).some((l) => /EXPERTS|no experts/.test(l)), "EXPERTS not drawn");
});


/** The cells a line of half blocks draws: each character with its foreground (none for a blank: a space shows none)
 * and background (null: the panel's), the escape sequences played through. */
function cellsOf(line: string): [string, string | null, string | null][] {
	const out: [string, string | null, string | null][] = [];
	let fg: string | null = null, bg: string | null = null;
	for (const m of line.matchAll(/\x1b\[([0-9;]*)m|([^\x1b])/gu)) {
		if (m[2] !== undefined) out.push([m[2], m[2] === " " ? null : fg, bg]);
		else if (m[1] === "0") fg = bg = null;
		else if (m[1] === "39") fg = null;
		else if (m[1] === "49") bg = null;
		else if (m[1].startsWith("38;")) fg = m[1];
		else if (m[1].startsWith("48;")) bg = m[1];
	}
	return out;
}
/** rgbBlocks as it was, every cell's colours written in full: the reference the shorter output must draw the same as. */
function rgbBlocksFull(cells: (Rgb | null)[], cols: number, rows: number): string[] {
	const code = (c: Rgb, bg: boolean) => colourCode(c, "truecolor", bg), at = (r: number, c: number) => (r < rows ? cells[r * cols + c] ?? null : null);
	const out: string[] = [];
	for (let r = 0; r < rows; r += 2) {
		let line = "";
		for (let c = 0; c < cols; c++) {
			const up = at(r, c), lo = at(r + 1, c);
			if (up === null && lo === null) line += "\x1b[0m ";
			else if (up === null) line += "\x1b[0m" + code(lo!, false) + "▄";
			else if (lo === null) line += "\x1b[0m" + code(up, false) + "▀";
			else line += code(up, false) + code(lo, true) + "▀";
		}
		out.push(line + "\x1b[0m");
	}
	return out;
}

test("rgbBlocks: a colour written only where it changes on the line, drawn the same cell for cell; a mostly blank line far shorter", () => {
	let seed = 3;
	const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
	const pick: Rgb[] = [[255, 180, 84], [86, 204, 242], [20, 22, 25]];
	for (const blank of [0.1, 0.5, 0.95]) {
		// Runs of a colour two cells wide (STREAM's keys), blanks among them.
		const cells = Array.from({ length: 16 * 96 }, (_, i) => (i % 2 ? null : rnd() < blank ? null : pick[Math.floor(rnd() * 3)]));
		for (let i = 1; i < cells.length; i += 2) cells[i] = cells[i - 1];
		const now = rgbBlocks(cells, 96, 16, "truecolor"), was = rgbBlocksFull(cells, 96, 16);
		assert.deepEqual(now.map((l) => plain(l)), was.map((l) => plain(l)), "the same characters");
		assert.deepEqual(now.map(cellsOf), was.map(cellsOf), `blank ${blank}: the same colours, cell for cell`);
		assert.ok(now.join("").length < was.join("").length);
	}
	const one = new Array<Rgb | null>(2 * 96).fill(null);
	one[40] = one[41] = [255, 180, 84];
	const [a, b] = [rgbBlocks(one, 96, 2, "truecolor")[0], rgbBlocksFull(one, 96, 2)[0]];
	assert.ok(Buffer.byteLength(a) * 4 < Buffer.byteLength(b), `a mostly blank line: ${Buffer.byteLength(a)} bytes, was ${Buffer.byteLength(b)}`);
});

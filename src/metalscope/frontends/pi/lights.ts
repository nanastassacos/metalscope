/**
 * The dock's expert lights: a layer's experts as a grid of lights (EXPERTS, the mini dock), drawn with half-height
 * blocks, two lights to a character (▀: the upper light in the foreground colour, the lower in the background colour).
 * The colours are a ramp of the dock's own (STOPS) from its background through amber to near-white; the brightness of
 * each expert follows the routing token by token (Glow). dock.ts lays them out.
 */

import { type Rgb, colourCode } from "./draw.ts";
import type { TokenEvent } from "./state.ts";

// -- colour ------------------------------------------------------------------------------------------------------

export type Stops = [number, Rgb][];

/** The lights' ramp, the dock's own (not Pi's theme): an unlit light a shade off the dock's background (the grid's
 * shape shows), through amber, to near-white at the peak. */
export const STOPS: Stops = [[0, [24, 27, 33]], [0.12, [78, 52, 22]], [0.45, [204, 122, 30]], [0.8, [255, 198, 98]], [1, [255, 245, 216]]];

/** `a` moved toward `b` by `t` (0-1). */
export const mix = (a: Rgb, b: Rgb, t: number): Rgb => [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)) as Rgb;

/** The ramp's colour at `v` (0-1). */
export function rampAt(stops: Stops, v: number): Rgb {
	v = Math.max(0, Math.min(1, v));
	for (let i = 1; i < stops.length; i++)
		if (v <= stops[i][0]) {
			const [a, ca] = stops[i - 1], [b, cb] = stops[i];
			return mix(ca, cb, b > a ? (v - a) / (b - a) : 1);
		}
	return stops[stops.length - 1][1];
}

/** Escape sequences per brightness step: `fg[s]` / `bg[s]` for step s of LEVELS (0 = unlit). */
export interface Palette {
	fg: string[];
	bg: string[];
}
export const LEVELS = 32;

/** 24-bit colour unless `theme` draws in 256 colours. */
export function colourMode(theme: any): "truecolor" | "256color" {
	try {
		if (theme?.getColorMode?.() === "256color") return "256color";
	} catch {}
	return "truecolor";
}

/** The STOPS ramp in LEVELS steps: 24-bit colour unless `theme` draws in 256 colours (then pi-tui's nearest). */
export function palette(theme: any): Palette {
	const mode = colourMode(theme);
	const fg: string[] = [], bg: string[] = [];
	for (let s = 0; s < LEVELS; s++) {
		const rgb = rampAt(STOPS, s / (LEVELS - 1));
		fg.push(colourCode(rgb, mode));
		bg.push(colourCode(rgb, mode, true));
	}
	return { fg, bg };
}

const step = (v: number) => Math.max(0, Math.min(LEVELS - 1, Math.round(v * (LEVELS - 1))));

/** Lines of half-height blocks: `cells` row-major (cell = row * cols + column), each a brightness 0-1, or null for
 * no light (the dock's own background, a blank). Two rows to a line; an odd count leaves the last half blank. Each
 * line ends in a full reset, after which the dock re-applies its background (draw.ts tint). */
export function halfBlocks(cells: (number | null)[], cols: number, rows: number, pal: Palette): string[] {
	return halfLines(cells, cols, rows, (v) => pal.fg[step(v)], (v) => pal.bg[step(v)]);
}

/** halfBlocks' and rgbBlocks' lines, each cell's colours as `fg` and `bg` give their escape sequences. A sequence is
 * written only where it changes from the cell before on the line (most of a line is runs of one colour, or blank):
 * a blank is a space on the dock's background (ESC[49m, which tint answers with it, where a background was set), a
 * half on its own the other half's background the same. */
function halfLines<T>(cells: (T | null)[], cols: number, rows: number, fg: (c: T) => string, bg: (c: T) => string): string[] {
	const at = (r: number, c: number) => (r < rows && c < cols ? cells[r * cols + c] ?? null : null);
	const out: string[] = [];
	for (let r = 0; r < rows; r += 2) {
		let line = "", f = "", b = "";   // the colours in force on the line ("": the dock's)
		for (let c = 0; c < cols; c++) {
			const up = at(r, c), lo = at(r + 1, c);
			const nf = up !== null ? fg(up) : lo !== null ? fg(lo) : f, nb = up !== null && lo !== null ? bg(lo) : "";
			if (nb !== b) line += (b = nb) || "\x1b[49m";
			if (nf !== f) line += (f = nf);
			line += up === null && lo === null ? " " : up === null ? "▄" : "▀";
		}
		out.push(line + "\x1b[0m");
	}
	return out;
}

/** An Rgb's escape sequence in `mode`, foreground or background, each worked out once a call of rgbBlocks. */
function codes(mode: "truecolor" | "256color") {
	const got = new Map<number, string>();
	return (c: Rgb, bg: boolean) => {
		const key = ((c[0] * 256 + c[1]) * 256 + c[2]) * 2 + (bg ? 1 : 0);   // the channels are whole 0-255 (mix rounds)
		let s = got.get(key);
		if (s === undefined) got.set(key, (s = colourCode(c, mode, bg)));
		return s;
	};
}

/** halfBlocks with a colour per cell (MAP, SETTLING, RESIDUAL STREAM): `cells` row-major, each an Rgb or null (blank). */
export function rgbBlocks(cells: (Rgb | null)[], cols: number, rows: number, mode: "truecolor" | "256color"): string[] {
	const code = codes(mode);
	return halfLines(cells, cols, rows, (c) => code(c, false), (c) => code(c, true));
}

// -- EXPERTS -----------------------------------------------------------------------------------------------------

// Brightness: a token's experts light up by their share of the mix (a share of SHARE_FULL or more is full
// brightness, GAMMA lifts small shares) and fade by GLOW_DECAY a token (a short afterglow at about 40 tokens a
// second); a slow WARM level (decaying by WARM_DECAY a token) keeps experts busy lately at a
// dim floor, WARM_FLOOR at the busiest.
export const GLOW_DECAY = 0.95;
export const WARM_DECAY = 0.99;
export const WARM_FLOOR = 0.3;
const SHARE_FULL = 0.3, GAMMA = 0.6;
// Starting part way into a reply (another layer, a reopened dock): the warm level forgets in about this many tokens.
export const GLOW_REPLAY = 500;

/** A share's brightness. */
export const lit = (share: number) => Math.min(1, Math.pow(Math.max(0, share) / SHARE_FULL, GAMMA));

/** A token's experts at `layer`: [expert, share 0-1], biggest share first (none without routing there). */
export function tokenExperts(t: Pick<TokenEvent, "experts" | "expert_w" | "expert_k">, layer: number): [number, number][] {
	const k = t.expert_k ?? 0;
	if (!t.experts || !t.expert_w || !k || (layer + 1) * k > t.experts.length) return [];
	const out: [number, number][] = [];
	for (let j = layer * k; j < (layer + 1) * k; j++) if (t.expert_w[j] > 0) out.push([t.experts[j], t.expert_w[j] / 255]);
	return out.sort((a, b) => b[1] - a[1]);
}

/** One layer's lights over a reply: feed() each token in turn. */
export class Glow {
	glow: Float32Array;
	warm: Float32Array;
	constructor(readonly E: number) {
		this.glow = new Float32Array(E);
		this.warm = new Float32Array(E);
	}
	/** One token: every light fades, then this token's experts (tokenExperts) light up by their share. */
	feed(experts: [number, number][]) {
		for (let e = 0; e < this.E; e++) {
			this.glow[e] *= GLOW_DECAY;
			this.warm[e] *= WARM_DECAY;
		}
		for (const [e, s] of experts) {
			if (e < 0 || e >= this.E || s <= 0) continue;
			this.glow[e] = Math.max(this.glow[e], lit(s));
			this.warm[e] += s;
		}
	}
	/** Each expert's brightness: its glow, at least its warm floor; a reply's end changes nothing. */
	levels(): Float32Array {
		let top = 0;
		for (let e = 0; e < this.E; e++) top = Math.max(top, this.warm[e]);
		const out = new Float32Array(this.E);
		for (let e = 0; e < this.E; e++) out[e] = Math.max(this.glow[e], top > 0 ? WARM_FLOOR * Math.sqrt(this.warm[e] / top) : 0);
		return out;
	}
}

/** One layer's grid from the server (/scope/api/experts/grid): `pos[e]` the cell of expert e, null without an expert
 * map. */
export interface Grid {
	cols: number;
	rows: number | null;
	pos: number[] | null;
	num_experts: number | null;
	n_layers: number | null;
}

export const GRID_COLS = 32;

/** The grid to draw for `E` experts: the server's when it has positions for this many, else number order. */
export function gridShape(E: number, grid?: Grid | null): { cols: number; rows: number; pos: number[] | null } {
	if (grid?.pos && grid.pos.length === E && grid.rows) return { cols: grid.cols, rows: grid.rows, pos: grid.pos };
	return { cols: GRID_COLS, rows: Math.max(1, Math.ceil(E / GRID_COLS)), pos: null };
}

/** A grid `shape` for `E` experts redrawn about `w` wide (EXPERTS on the board, filling its column): the fewest rows
 * that width allows, then as few columns as those rows need. The map's sense kept: its columns (left to right on the
 * map) refilled in order, each new column sorted by the old row (top to bottom); number order stays number order. */
export function reshapeGrid<S extends { cols: number; rows: number; pos: number[] | null }>(E: number, shape: S, w: number): S {
	const rows = Math.max(1, Math.ceil(E / Math.max(1, Math.min(w, E)))), cols = Math.max(1, Math.ceil(E / rows));
	if (cols === shape.cols && rows === shape.rows) return shape;
	if (!shape.pos) return { ...shape, cols, rows };
	const old = shape.pos, c0 = shape.cols, r0 = Math.max(1, shape.rows), key = `${E} ${c0} ${r0} ${cols} ${rows}`;
	const done = reshaped.get(old) ?? new Map<string, number[]>();
	reshaped.set(old, done);
	let pos = done.get(key);
	if (pos) return { ...shape, cols, rows, pos };
	const order = Array.from({ length: E }, (_, e) => e).sort((a, b) => (old[a] % c0) - (old[b] % c0) || old[a] - old[b] || a - b);
	pos = new Array<number>(E).fill(0);
	for (let c = 0; c < cols; c++) {
		const col = order.slice(c * rows, (c + 1) * rows).sort((a, b) => Math.floor(old[a] / c0) / r0 - Math.floor(old[b] / c0) / r0 || a - b);
		col.forEach((e, r) => (pos![e] = r * cols + c));
	}
	done.set(key, pos);
	return { ...shape, cols, rows, pos };
}
/** reshapeGrid's positions by the map's own (a grid's, kept while its answer is) and size: worked out once each. */
const reshaped = new WeakMap<number[], Map<string, number[]>>();

/** Experts per layer before the server says: the smallest power of two over the highest id seen (at least 8). */
export const guessExperts = (maxId: number) => Math.max(8, 2 ** Math.ceil(Math.log2(maxId + 1)));

/** The grid's cells for per-expert brightness `levels`: cells no expert sits in are blank. */
export function gridCells(levels: ArrayLike<number>, shape: { cols: number; rows: number; pos: number[] | null }): (number | null)[] {
	const cells: (number | null)[] = new Array(shape.cols * shape.rows).fill(null);
	for (let e = 0; e < levels.length; e++) {
		const at = shape.pos ? shape.pos[e] : e;
		if (at >= 0 && at < cells.length) cells[at] = levels[e];
	}
	return cells;
}

/** The grids fetched from the server, one per (server, model, layer): `at` answers at once, undefined until the
 * answer arrives (then `onLoad`). A failed request or one without an expert map is asked again after RETRY_MS (a map
 * built meanwhile shows). */
export class Grids {
	static RETRY_MS = 60_000;
	private got = new Map<string, { grid: Grid | null; at: number } | "pending">();
	constructor(private get: (url: string) => Promise<any>, private onLoad: () => void, private now = () => Date.now()) {}
	/** Every grid asked for again (a link moved the tally they come from). */
	clear() {
		this.got.clear();
	}
	at(base: string, model: string | undefined, layer: number): Grid | null | undefined {
		const key = `${base}|${model ?? ""}|${layer}`;
		const have = this.got.get(key);
		if (have === "pending") return undefined;
		if (have && (have.grid?.pos || this.now() - have.at < Grids.RETRY_MS)) return have.grid;
		this.got.set(key, "pending");
		const q = new URLSearchParams({ layer: String(layer), ...(model ? { model } : {}) });
		this.get(`${base}/scope/api/experts/grid?${q}`).then(
			(g) => this.got.set(key, { grid: g && typeof g.cols === "number" ? g : null, at: this.now() }),
			() => this.got.set(key, { grid: null, at: this.now() }),
		).finally(() => this.onLoad());
		return have ? have.grid : undefined;
	}
}

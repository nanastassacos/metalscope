/**
 * The dock as one screen (the board). The header (◆ metalscope, the steer on the right), the reply strip, then on the
 * left MAP (the whole reply by confidence) and LENS (the telescope, telescope.ts, and over it SETTLING, how deep each
 * recent token settled); on the right TOKENS (the token's top 10 choices), EXPERTS (layers of lights, lights.ts) and
 * STREAM (what each layer changed in the residual stream, a key a layer), then one key line. tab moves the focus (an
 * orange title) through the panels, and the panel in focus takes the arrows; its number (PANEL_KEY) opens a panel over
 * the dock under the reply strip. 6 is the Steer page (the concepts by the chat test, the cursor's dose curve and test
 * chats, steerpage.ts; the whole dock) and m the mini dock (lights, MAP, SETTLING and STREAM under them, MINI_WIDTH
 * columns). A dock too small for the board (boardFits) is the one-panel board: a short reply strip, the tab row and the
 * panel in focus full size. dock.ts gathers the data and the keys; everything here is pure, each line fitted to its
 * width.
 */

import { colorToRgb, indexedColor, truncateToWidth, visibleWidth, wrapTextWithAnsi } from "@earendil-works/pi-tui";
import { COLOURS, EXPLAIN, LABELS, type Rgb, UNSURE, ansiInverse, bar, bold, centre, colourCode, confColor, cut, fit, label, lensSettle, lpad, offTop, padTo, pct, replyCounts, streamExplain } from "./draw.ts";
import { type Palette, type Grid, type Stops, STOPS, colourMode, halfBlocks, mix, rampAt, rgbBlocks } from "./lights.ts";
import type { Specialists } from "./expert.ts";
import { type MapReply, type Reply, type TokenEvent, WRITE_BEAM, mapPlace, unpackLargest } from "./state.ts";
import { type DoseRow, type ExampleTurn, type Examples, type LinkOffer, type Note, type Side, type SteerList, block, doseLine, fmt, linkLabel, reading,
	starred } from "./steerpage.ts";
import { type LensName, SWAP_FOG, lensOf, onTube } from "./telescope.ts";

export type Panel = "LENS" | "MAP" | "EXPERTS" | "TOKENS" | "STREAM";
/** The panels in tab order: the left column top to bottom (MAP, about the reply, under the reply strip), then the
 * right one. LENS holds SETTLING over the telescope (one panel: one tab stop, one key). STEER is a page of its own (6). */
export const PANELS: Panel[] = ["MAP", "LENS", "TOKENS", "EXPERTS", "STREAM"];
/** The middle's panels, in its two columns; STREAM (RESIDUAL STREAM) last in the right one where the board has room
 * for it (streamLines), never on the one-panel board; on the mini dock a block of its own (dock.ts renderMini). */
const LEFT: Panel[] = ["MAP", "LENS"], RIGHT: Panel[] = ["TOKENS", "EXPERTS", "STREAM"];

// -- the layout: what it needs ----------------------------------------------------------------------------------------

export const REPLY_ROWS = 6;          // the reply strip's lines under its title, at the least (replyRows: more where both columns spare them)
export const STEER_ROWS = 8;          // the Steer page's concept rows a column at the least (two columns, a count under)
export const RIGHT_W = 50;            // the right column: EXPERTS' 32 lights and 16 columns beside them
export const LEFT_MIN = 44;           // the telescope with its eyepiece's words readable and the turret line whole
export const TOP_N = 10;               // TOKENS: the choices shown for the token in focus
// The middle at its shortest: TOKENS (its line, the top 10 and an eleventh row for a written token outside them), a
// rule, EXPERTS (title, one layer's 8 lines of lights).
export const MIDDLE_MIN = (1 + TOP_N + 1) + 1 + (1 + 8);
// The header, a rule, the reply (title and rows), a rule; under the middle a rule and the key line.
const FIXED = 1 + 1 + (1 + REPLY_ROWS) + 1 + 1 + 1;
/** The narrowest dock (the overlay's columns, its edge and pad included) and the shortest (its lines) the board is
 * drawn in: the left column at its narrowest beside the right one; the fixed parts (12), the middle at its shortest
 * (22) and two lines for ? (its explanation and the layout's keys). Below either, the one-panel
 * board. */
export const BOARD_MIN_WIDTH = 3 + LEFT_MIN + 3 + RIGHT_W;   // 100
export const BOARD_MIN_ROWS = FIXED + MIDDLE_MIN + 2;   // 36

export const boardFits = (width: number, height: number) => width >= BOARD_MIN_WIDTH && height >= BOARD_MIN_ROWS;

/** The one-panel board: the reply strip's lines, and the panel's least lines under the tab row (TOKENS whole). */
export const ONE_REPLY_ROWS = 3, ONE_BODY = 1 + TOP_N + 1;
/** The smallest dock drawn (the one-panel board; smaller, one line says so): 44 columns, the full dock's least
 * (widget.ts), where LENS's column (41 text columns) still reads; the header, a rule, the reply strip and a rule, the
 * tab row and the panel, a rule, two key lines. */
export const DOCK_MIN_WIDTH = 44;
export const DOCK_MIN_ROWS = 1 + 1 + (1 + ONE_REPLY_ROWS) + 1 + 1 + ONE_BODY + 1 + 2;   // 23

/** The reply strip's lines on a board `height` lines tall whose columns spare `left` and `right` lines (what their
 * panels do not need): REPLY_ROWS, plus the lines both spare, at most a third of the dock. */
export const replyRows = (height: number, left: number, right: number) =>
	REPLY_ROWS + Math.max(0, Math.min(left, right, Math.floor(height / 3) - REPLY_ROWS));

/** The middle's two columns in `w` text columns: the right one RIGHT_W wide, the left the rest, " │ " between. */
export const columns = (w: number) => ({ lw: w - 3 - RIGHT_W, rw: RIGHT_W });

/** The middle's two columns: MAP and LENS on the left; on the right TOKENS, EXPERTS where the model has experts
 * (`experts`) and STREAM where it is drawn (`stream`). */
export const arrange = (experts: boolean, stream: boolean): { left: Panel[]; right: Panel[] } =>
	({ left: LEFT, right: RIGHT.filter((p) => (p !== "EXPERTS" || experts) && (p !== "STREAM" || stream)) });
/** The Steer page's two halves: the concept list (two columns of 25, a space between) and the dose curve; where the
 * curve would have under STEER_DETAIL columns, `stacked`: the list over it, each the width, the list one column where
 * two do not fit (`across`). */
const STEER_DETAIL = 30;
export const steerColumns = (w: number) => {
	const stacked = w - 51 - 3 < STEER_DETAIL;
	return { sl: stacked ? w : 51, sr: stacked ? w : w - 51 - 3, stacked, across: w >= 51 ? 2 : 1 };
};

// -- lines ------------------------------------------------------------------------------------------------------------

/** Two columns side by side, " │ " between them (in the rule colour), each line exactly its column's width. */
export function side(th: any, a: string[], b: string[], wa: number, wb: number): string[] {
	const out: string[] = [];
	for (let i = 0; i < Math.max(a.length, b.length); i++) out.push(fit(a[i] ?? "", wa) + th.fg("rule", " │ ") + fit(b[i] ?? "", wb));
	return out;
}

/** A panel's title: orange, and drawn inverse when the panel is in focus (tab). */
export function title(th: any, name: string, focused: boolean): string {
	return focused ? th.fg("accent", bold(th, ansiInverse(` ${name} `))) : th.fg("accent", bold(th, name));
}

export const rule = (th: any, w: number) => th.fg("rule", "─".repeat(w));
/** A rule `w` wide with a dim line in it (engram inspection's legend): the first of `texts` that fits whole, else the
 * last cut to fit. */
export function ruleText(th: any, texts: readonly string[], w: number): string {
	const room = Math.max(1, w - 4), t = texts.find((x) => visibleWidth(x) <= room) ?? cut(texts[texts.length - 1], room);
	return th.fg("rule", "── ") + th.fg("dim", t) + th.fg("rule", " " + "─".repeat(Math.max(0, w - 4 - visibleWidth(t))));
}

/** The tab row of a full-size panel: the panels that can be opened (`shown`), each with its number, the one in focus
 * inverse; its number again back to the board where it fits (`one`, the one-panel board: nothing to go back to). */
export function tabRow(th: any, focus: Panel, w: number, shown: readonly Panel[], one = false): string {
	const names = shown.map((p) => {
		const s = ` ${PANEL_KEY[p]} ${p} `;
		return p === focus ? th.fg("accent", bold(th, ansiInverse(s))) : th.fg("dim", s);
	}).join(" ");
	const back = one ? "" : th.fg("faint", `  ${PANEL_KEY[focus]} back`);
	return fit(visibleWidth(names + back) <= w ? names + back : names, w);
}

// -- the Steer page (6) and the mini dock (m) -------------------------------------------------------------------------

export const STEER_KEY = "6";
/** The number that opens each panel over the board (again: the board): PANELS in order from 1, STEER_KEY after them. */
export const PANEL_KEY = Object.fromEntries(PANELS.map((p, i) => [p, String(i + 1)])) as Record<Panel, string>;
/** The numbers of the panels `shown` (PANEL_KEY), with `steer` the Steer page's (STEER_KEY) too, runs of them
 * joined: "1-3 5", "1-6". */
export function panelNums(shown: readonly Panel[], steer = false): string {
	const ns = [...shown.map((p) => +PANEL_KEY[p]), ...(steer ? [+STEER_KEY] : [])].sort((a, b) => a - b), out: string[] = [];
	for (let i = 0, j = 0; i < ns.length; i = ++j) {
		while (ns[j + 1] === ns[j] + 1) j++;
		out.push(j > i ? `${ns[i]}-${ns[j]}` : `${ns[i]}`);
	}
	return out.join(" ");
}
/** What ? adds on the board under the explanation: the layout's keys (`nums`: the panels' numbers, panelNums; none on
 * the one-panel board, whose tab row shows them), and how to get the key line back. */
export const layoutKeys = (nums = "") => `${nums && `${nums} open a panel · `}${STEER_KEY} steer page · m mini · ? keys`;

/** The mini dock: MINI_WIDTH columns (the overlay's: its edge, the right column's width, a pad). */
export const MINI_WIDTH = 3 + RIGHT_W;
/** The mini dock's key line, `w` wide (MINI_WIDTH's text): stepped in, its keys, `g` engram inspection's where offered
 * (boardKeys; short where that keeps another key), ctrl+\ closes, then [ ] more, then f live going first where they do
 * not all fit; else how to reach them (until then they go to Pi's editor), g among them where offered and it fits.
 * `full`: m the full dock (not in a terminal too short for it). */
export const miniKeys = (inspecting: boolean, held = false, g = "", w = MINI_WIDTH - 3, full = true) => !inspecting && !held
	? stepOut((x) => `ctrl+\\ steps in${full ? ", then m" : ""} · [ ]${x}`, " · g", g, w)
	: gCut([held ? "f live" : "", full ? "m full" : "", "[ ] more", g, held ? "" : "ctrl+\\ closes"].filter(Boolean), ["ctrl+\\ closes", "[ ] more", "f live"], w, g, true);
/** The narrowest terminal the dock (full or mini) is shown in (widget.ts). */
export const DOCK_MIN_COLS = 100;
/** The mini dock drawn: chosen (m), or the terminal (`rows` lines; undefined: not known) too short (DOCK_MIN_ROWS) for
 * the full one. */
export const miniAt = (mini: boolean | undefined, rows?: number) => !!mini || (typeof rows === "number" && rows < DOCK_MIN_ROWS);
/** How many grids the mini dock stacks in `rows` lines, each a label and `h` lines of lights: at least one. */
export const miniCount = (rows: number, h: number) => Math.max(1, Math.floor(rows / (1 + h)));
/** A map's lines under the mini dock's lights, its title included: [least, want, most]. */
export type MiniRoom = [number, number, number];
/** The mini dock's `rows` lines shared out: the lights in grids of `block` lines (one at the least, `grids` at the
 * most), then MAP and SETTLING (undefined: not drawn, no lens read). The maps take their want, else less (SETTLING its
 * least, MAP its least, then SETTLING goes, then MAP) where one grid would not be left; the lights every grid the rest
 * holds; the lines over go to SETTLING, then MAP, up to their most, the rest left over (dock.ts renderMini). Without
 * lights (no grids) MAP takes the lines, past its most: SETTLING up to its most, MAP the rest. */
export function miniSplit(rows: number, block: number, grids: number, map: MiniRoom, settle?: MiniRoom): { n: number; map: number; settle: number } {
	const [ml, mw, mm] = map, [sl, sw, sm] = settle ?? [0, 0, 0];
	if (!grids) {   // no lights (a model without experts): SETTLING up to its most, MAP the rest
		const s = settle ? Math.max(0, Math.min(sm, rows - ml)) : 0;
		return { n: 0, map: rows - s, settle: s };
	}
	const [m0, s0] = ([[mw, sw], [mw, sl], [ml, sl], [mw, 0], [ml, 0]] as const).find(([m, s]) => m + s <= rows - block) ?? [0, 0];
	const n = Math.min(Math.max(1, grids), miniCount(rows - m0 - s0, block - 1));
	const left = rows - m0 - s0 - n * block, s = s0 ? s0 + Math.max(0, Math.min(left, sm - s0)) : 0;
	return { n, map: m0 ? m0 + Math.max(0, Math.min(left - (s - s0), mm - m0)) : 0, settle: s };
}

/** The mini dock's line for the token stepped to (one line, `w` wide): its index and text in its confidence colour,
 * then its probability. */
export function miniToken(th: any, t: { text: string; p: number | null }, sel: number, w: number): string {
	return fit(th.fg(confColor(t.p), cut(`#${sel} ${label(t.text)}`, Math.max(1, w - 7))) + th.fg("dim", ` ${pct(t.p)}`), w);
}

/** The widest key line the board draws: its text columns at BOARD_MIN_WIDTH (the edge, a space and a pad off). */
export const KEYS_W = BOARD_MIN_WIDTH - 3;

/** The key line for the panel in focus: only keys that do something there, the next panel tab goes to named from `order`
 * (the panels shown), the numbers that do something (`open`'s and 6 the Steer page's, panelNums) as one "N panel". The
 * parts first in `spare` go first where the line would pass the text width `w`, N panel last. `windowed`: EXPERTS drawn
 * as its layer window (full size). `held`: a token stepped to (not the newest, followed): f live after the token keys.
 * `steered`: a steer on (LENS: x off). `edit`: the LENS word edit offered (not where the server refuses the model one).
 * `to`: EXPERTS' window shows one source, `a` switching to `to` (LABELS.source). `g`: engram inspection's key
 * (LABELS.engramKey or engramClose; "" where the server lists none), on every panel's line before N panel, the last
 * spare part to go; before stepping in, short, the first. `one`: the one-panel board, the keys on up to two lines
 * (wrapKeys) before any goes, -/+ layers spare before N panel so two lines always hold the rest. `shut`: LENS dulled,
 * its reading keys left out: the lenses "off" (+ lenses), or "none" read for the reply. `swap`: the token LENS shows
 * has both lenses (s other lens). `views`: STREAM has more than one (s view). `open`: the panels a number opens; before
 * stepping in their numbers (panelNums) and 6 steer in words, none on the one-panel board. */
export function boardKeys(focus: Panel, inspecting: boolean, windowed = false, order: readonly Panel[] = PANELS, held = false,
	steered = false, layers = false, edit = true, to = "", w = KEYS_W, g = "", one = false, shut: "" | "off" | "none" = "", swap = true, views = true, open: readonly Panel[] = order): string {
	const lines = one ? 2 : 1, step = !layers ? "" : shut === "off" ? LABELS.lensOn : LABELS.layersKey;
	const nums = one || !open.length ? "" : `${panelNums(open)} open, `, pk = `${panelNums(open, true)} panel`;
	if (!inspecting) return stepOut((x) => `ctrl+\\ steps in: tab a panel, ? explain, ${nums}${x}${STEER_KEY} steer · /scope help dock`,
		`${g === LABELS.engramKey ? LABELS.engramShort : g}, `, g, w, lines);
	const next = `tab ${order[(Math.max(0, order.indexOf(focus)) + 1) % order.length].toLowerCase()}`;
	const parts: Record<Panel, string[]> = {
		LENS: [next, "←/→ token", shut ? "" : "↑/↓ layer", step, edit && !shut ? "enter edit" : "", steered ? "x off" : "", shut || !swap ? "" : "s other lens", g, pk, "? explain", "esc back"],
		MAP: [next, "←/→ token", "u/U unsure", g, pk, "? explain", "esc back"],
		EXPERTS: windowed ? [next, "←/→ token", "↑/↓ layer", "enter open", to && `a ${to}`, g, pk, "? explain", "esc back"]
			: [next, "←/→ token", "[ ] layers", g, pk, "? explain", "esc back"],
		TOKENS: [next, "←/→ token", "↑/↓ unsure token", g, pk, "? explain", "esc back"],
		STREAM: [next, "←/→ token", views ? "s view" : "", g, pk, "? explain", "esc back"],
	};
	const keys = parts[focus].filter(Boolean);
	const at = keys.indexOf("←/→ token");
	if (held) keys.splice(at < 0 ? Math.min(2, keys.length) : at + 1, 0, "f live");
	return gCut(keys, ["esc back", "? explain", "s other lens", "s view", ...(one && step ? [step] : []), pk, g], w, g, false, lines);
}

/** Key line `text` in lines `w` wide, broken between its keys (parts `sep` apart; a part wider than `w` wrapped). */
export function wrapKeys(text: string, w: number, sep = " · "): string[] {
	const out: string[] = [];
	let cur = "";
	for (const k of text.split(sep)) {
		const next = cur ? cur + sep + k : k;
		if (visibleWidth(next) <= w) cur = next;
		else {
			if (cur) out.push(cur);
			const ls = visibleWidth(k) <= w ? [k] : wrapTextWithAnsi(k, w);
			out.push(...ls.slice(0, -1));
			cur = ls.at(-1) ?? "";
		}
	}
	return [...out, cur];
}
/** Whether key line `text` fits `lines` lines `w` wide (wrapKeys). */
const fitsIn = (text: string, w: number, lines: number) => (lines === 1 ? visibleWidth(text) <= w : wrapKeys(text, w).length <= lines);

/** A key line before stepping in: `line(x)` with g's part `x` where engram inspection's key `g` is offered and the line
 * fits `lines` lines `w` wide, else without it. */
const stepOut = (line: (x: string) => string, x: string, g: string, w: number, lines = 1) => (g && fitsIn(line(x), w, lines) ? line(x) : line(""));

/** spareCut with engram inspection's key `g` whole, in its short form where whole it would go (`keep`: where whole it
 * costs another key). */
function gCut(keys: string[], spare: string[], w: number, g: string, keep = false, lines = 1): string {
	const as = (x: string) => spareCut(keys.map((k) => (k === g ? x : k)), spare.map((k) => (k === g ? x : k)), w, lines);
	const whole = as(g);
	if (g !== LABELS.engramKey || (!keep && whole.includes(g))) return whole;
	const short = as(LABELS.engramShort);
	return whole.split(" · ").length >= short.split(" · ").length && whole.includes(g) ? whole : short;
}

/** The key lines of the full-size EXPERTS' pages: one layer (enter opens `e`, the pick) and one expert. `a` names the
 * source it switches to (`to`, LABELS.source; none: both shown); `g` engram inspection's key where it works. */
export const layerKeys = (to: string, e?: number, g = "") =>
	`←→ token  ↑↓ pick  tab other list  enter open${e === undefined ? "" : ` #${e}`}  [ ] layer${to ? `  a ${to}` : ""}${g && `  ${g}`}  esc back`;
export const expertKeys = (to: string, g = "") => `←→ token  ↑↓ pick  enter go  [ ] layer  a ${to}${g && `  ${g}`}  b back  esc layer`;

/** `keys` joined with " · ", the ones in `spare` dropped in order while they pass `lines` lines `w` wide. */
function spareCut(keys: string[], spare: string[], w: number, lines = 1): string {
	for (const s of spare) {
		if (fitsIn(keys.join(" · "), w, lines)) break;
		const i = keys.indexOf(s);
		if (i >= 0) keys.splice(i, 1);
	}
	return keys.join(" · ");
}

/** The key line while a LENS word is edited, `w` wide on up to `lines` lines: narrower, enter's "redo from here" goes,
 * then tab, then typing. */
const EDIT_KEYS = ["⌫ remove", "type or ↑/↓ a word", "tab next", "enter apply, redo from here", "esc cancel"];
export const editKeys = (w: number, lines = 1) => fitsIn(EDIT_KEYS.join(" · "), w, lines) ? EDIT_KEYS.join(" · ")
	: spareCut(EDIT_KEYS.map((k) => k.replace(", redo from here", "")), ["tab next", "type or ↑/↓ a word"], w, lines);

/** The Steer page's key line, `w` wide on up to `lines` lines: esc back, then d delete go first where it does not fit.
 * `tested`: the cursor's concept was chat-tested (else no doses or chats to step through, and enter only turns a steer
 * `on` off). Not `inspecting`: how to step in. */
export const steerKeys = (w: number, lines = 1, tested = true, on = false, inspecting = true, link = false) => !inspecting
	? `ctrl+\\ steps in: ${link ? "l link" : "↑/↓ concept"}, ${STEER_KEY} board · /scope help dock`
	: link ? spareCut(["l link", "x off", "? explain", `${STEER_KEY} board`, "esc back"], ["esc back", "? explain"], w, lines)
	: spareCut(["↑/↓ concept", tested ? "←/→ dose" : "", tested ? "enter steer" : on ? "enter off" : "", "x off", "d delete", tested ? "n/N turn" : "",
		"? explain", `${STEER_KEY} board`, "esc back"].filter(Boolean), ["esc back", "d delete", "? explain"], w, lines);

/** What ? says for the panel in focus, where the key line is (draw.ts EXPLAIN): the EXPERTS window's own; STREAM's by
 * what the model has (`s`, streamExplain). */
export function explainOf(focus: Panel, windowed = false, s?: { k: number; rk: number }): string {
	if (focus === "LENS") return EXPLAIN.BOARD_LENS;
	if (focus === "STREAM") return streamExplain(s?.k ?? 0, s?.rk ?? 0);
	if (focus === "EXPERTS") return windowed ? EXPLAIN.EXPERTS_WINDOW : EXPLAIN.EXPERTS;
	return EXPLAIN[focus];
}

// -- the panels -------------------------------------------------------------------------------------------------------


/** EXPERTS for reply token `sel`: what dock.ts worked out (or a reason there is nothing to show). `layer` the layer
 * in focus ([ ] on the board). */
export interface ExpertsView {
	layer: number;
	L: number;              // MoE layers
	E: number;              // experts a layer
	k: number;              // mixed for each token
	reason?: string;           // set: nothing else is drawn but this
}

/** The `n` layers EXPERTS shows on the board (as many as its lines hold, dock.ts heights): evenly spaced, L/(n+1)
 * apart, around the focus layer `m` (three: a quarter of the model apart). Near an end the rest go on the other side.
 * Ascending; all of them when `n` >= L. */
export function stackLayers(L: number, m: number, n = 3): number[] {
	if (n >= L) return Array.from({ length: L }, (_, i) => i);
	const q = Math.max(1, Math.round(L / (n + 1))), out = new Set<number>();
	for (let j = 0; out.size < n && j <= 2 * L; j++) {
		const l = m + (j % 2 ? -1 : 1) * Math.ceil(j / 2) * q;
		if (l >= 0 && l < L) out.add(l);
	}
	for (let l = 0; out.size < n && l < L; l++) out.add(l);   // a spacing too wide to fit n: the gaps filled
	return [...out].sort((a, b) => a - b);
}

/** EXPERTS on the board, `w` by at most `rows`: its title, then the layers stacked (as many as fit, at least one, the
 * focus layer first), each a dim heading over its grid (lights.ts reshapeGrid: the column's width), the focus layer's
 * marked ▸ in accent. Only the lights: full size (4) for the experts themselves. `noMap`: the checkpoint has no expert
 * map, said on a line of its own under the layers where one is free, else in place of the title's note. `plate` (the
 * mini dock's): each grid a rectangle that many columns wide, its cells without an expert unlit rather than blank. */
export function expertsStack(th: any, v: ExpertsView, layers: LayerLights[], pal: Palette, w: number, rows: number, focused: boolean,
	noMap = false, plate = 0): string[] {
	const head = title(th, "EXPERTS", focused);
	if (v.reason) return [head, ...wrapTextWithAnsi(th.fg("dim", v.reason), w).slice(0, Math.max(1, rows - 1))];
	const cols = Math.max(1, ...layers.map((l) => l.cols));
	if (cols > w) return [head, th.fg("dim", `needs ${cols} columns`)];
	const blocks = layers.map((l) => {
		const on = l.layer === v.layer, name = cut(`${on ? "▸ " : "  "}layer ${l.layer + 1} of ${v.L}`, w);
		const cells = plate ? Array.from({ length: plate * l.rows }, (_, i) => (i % plate < l.cols ? l.cells[Math.floor(i / plate) * l.cols + (i % plate)] : 0) ?? 0) : l.cells;
		return [on ? th.fg("accent", bold(th, name)) : th.fg("dim", name), ...halfBlocks(cells, plate || l.cols, l.rows, pal)].map((x) => fit(x, w));
	});
	// As many as fit (the first always, cut where it must be): the focus layer first, then the nearest to it.
	let n = 0, used = 1;
	const pick = blocks.map((_, i) => i).sort((a, b) => Math.abs(layers[a].layer - v.layer) - Math.abs(layers[b].layer - v.layer));
	for (const i of pick) {
		if (n && used + blocks[i].length > rows) break;
		used += blocks[i].length;
		n++;
	}
	const out = [head, ...pick.slice(0, n).sort((a, b) => a - b).flatMap((i) => blocks[i])];
	let note = `  ${n} layer${n === 1 ? "" : "s"} · ${v.E} experts · ${v.k} a token`;
	if (visibleWidth(head) + visibleWidth(note) > w) note = `  ${v.k} of ${v.E} a token`;
	if (noMap && out.length < rows) out.push(th.fg("dim", cut(LABELS.noMap, w)));
	else if (noMap) note = "  " + LABELS.noMap;
	out[0] = head + th.fg("dim", note);
	return out.slice(0, Math.max(1, rows));
}

/** A share as a percentage: a decimal under 10% (the busiest over a window are a few percent each). */
export const sharePct = (s: number) => (s < 0.095 ? `${(s * 100).toFixed(1)}%` : `${Math.round(s * 100)}%`);
/** Where listRow's `said` starts: after the cursor's slot and the number (7), the bar and `num` (`numW` wide) where
 * there is a bar, and two spaces. */
export const saidAt = (bw: number, numW: number) => 9 + (bw ? 1 + bw + numW : 0);
/** A token column's least width (the tally's three tokens: 19-20 characters at the median, whole in 7 of 10 rows at
 * 22): narrower, the lists show one source and `a` switches it. The first of two at most SAID_MAX (whole in 95%). */
export const SAID_MIN = 22, SAID_MAX = 32;
/** The token columns in `room` (from saidAt on): one, or two (all chats, this session) two spaces apart. */
export const saidWidths = (room: number, n: number) => {
	const a = Math.min(SAID_MAX, Math.floor((room - 2) / 2));
	return n > 1 ? [a, room - 2 - a] : [room];
};
/** Where the second of two token columns starts in a list `w` wide (listRow's `bw` and `num` width). */
export const secondAt = (w: number, bw: number, numW: number) => saidAt(bw, numW) + saidWidths(w - saidAt(bw, numW), 2)[0] + 2;
/** EXPERTS FOR's bar, `w` wide. */
const bandBar = (w: number) => Math.max(4, Math.min(12, w - 34));
/** Whether EXPERTS FOR `w` wide shows both sources' tokens, each column SAID_MIN at least. */
export const bothSources = (w: number) => w - saidAt(bandBar(w), 5) >= 2 * SAID_MIN + 2;
/** One expert in the experts' lists (EXPERTS FOR, a layer's experts, MIXED WITH), `w` wide, the same columns in each:
 * a cursor's slot (`on`: ▸ and its number inverse; "mark": ▸ and its number in accent, the other list's cursor), its
 * number, a bar of `bw` at `frac` and a number (`num`) where `bw` is not 0, then `said` (saidAt; two: saidWidths) dim
 * (`on` brighter, "-" faint). `tone` "hit": in accent, the number too; "dim": the number dim. */
export function listRow(th: any, e: number, frac: number, bw: number, num: string, said: string | string[], w: number, on: boolean | "mark" = false,
	tone: "" | "hit" | "dim" = ""): string {
	const n = padTo(`#${e}`, 5), hit = tone === "hit" || on === "mark";
	const head = on ? th.fg("accent", "▸ ") + th.fg("accent", bold(th, on === true ? ansiInverse(n) : n)) : "  " + th.fg(hit ? "accent" : tone || "text", n);
	const mid = bw ? " " + th.fg("accent", padTo(bar(frac, bw), bw)) + th.fg("dim", num) : "";
	const room = w - saidAt(bw, visibleWidth(num)), cols = typeof said === "string" ? [said] : said, ws = saidWidths(room, cols.length);
	const col = (s: string, i: number) => th.fg(s === "-" ? "faint" : hit ? "accent" : on ? "text" : "dim", i < cols.length - 1 ? padTo(cut(s, ws[i]), ws[i] + 2) : cut(s, ws[i]));
	return head + mid + (room > 0 && cols.some(Boolean) ? "  " + cols.map(col).join("") : "");
}
/** `left` with `right` at the end of the line, `w` wide, where both fit; else `left` alone. */
export const rightOf = (left: string, right: string, w: number) =>
	visibleWidth(left) + 2 + visibleWidth(right) <= w ? fit(left, w - visibleWidth(right)) + right : left;
/** The token in focus, for the right of a heading line: `token #212 " not"`. */
export const tokenTag = (th: any, sel: number, text: string) => th.fg("dim", `${LABELS.token(sel)} `) + bold(th, cut(label(text), 24));
/** What the lights show, two short lines: token `sel`'s experts (`held`: paused on it), else the glow of the newest tokens. */
export const gridLegend = (held: boolean, sel: number): string[] => (held ? LABELS.legend.lit(sel) : [...LABELS.legend.glow]);
/** The widest legend line either way: what a grid leaves beside it, so pausing or following never resizes it. */
export const legendWidth = (sel: number) => Math.max(...[...gridLegend(true, sel), ...gridLegend(false, sel)].map((l) => visibleWidth(l)));

/** The tally's tokens for each expert at a layer (the specialists route's `top`, `sp`; experts under the evidence
 * floor absent, every one with no tally, `sp` null): undefined while asked for. */
export type Tops = ReadonlyMap<number, string[]> | undefined;
export const topsOf = (sp: Specialists | null | undefined): Tops =>
	sp === undefined ? undefined : new Map((sp?.experts ?? []).map((x) => [x.e, x.top]));
/** Tokens as the experts' lists show them: quoted, "-" for none. */
export const tokensText = (top: string[] | undefined) => top?.map(label).join(" ") || "-";
/** An expert's tokens from the tally (tokensText), "" while asked for. */
export const saidOf = (tops: Tops, e: number) => (tops ? tokensText(tops.get(e)) : "");

/** EXPERTS FOR, `w` wide, 1 + LIST_ROWS lines: its heading (the token's text, cut to end before the words, then
 * LABELS.its over them and `source`, the source alone where both do not fit; with two sources each one's name over its
 * column) and a row for each of the token's experts at a layer (`mine`, blank where fewer): a bar of its share of the
 * mix against the largest's, the share, and its tokens (saidOf) by each source in `tops` (one, or all chats and this
 * session: bothSources); `on` the cursor's row (-1: none; `mark`: the other list has the cursor). */
export function forTokenBand(th: any, text: string, mine: [number, number][], tops: Tops[], w: number, on = -1, source = "", mark = false): string[] {
	const bw = bandBar(w), top = Math.max(0, ...mine.map(([, x]) => x)), at = saidAt(bw, 5);
	const name = (room: number) => th.fg("accent", bold(th, LABELS.forToken(cut(label(text), Math.max(3, room - LABELS.forToken("").length)))));
	const one = [LABELS.its, source].filter(Boolean).join(" · ");
	const heads = tops.length > 1 ? padTo(LABELS.source.all, secondAt(w, bw, 5) - at) + LABELS.source.session : at + visibleWidth(one) <= w ? one : source || one;
	const head = at + visibleWidth(heads) <= w ? padTo(name(at - 2), at) + th.fg("dim", heads) : name(w);
	const out = mine.slice(0, LIST_ROWS).map(([e, s], i) => listRow(th, e, top > 0 ? s / top : 0, bw, lpad(sharePct(s), 5), tops.map((t) => saidOf(t, e)), w,
		i === on && (mark ? "mark" : true)));
	if (!out.length) out.push(th.fg("dim", "  no routing kept for this token"));
	while (out.length < LIST_ROWS) out.push("");
	return [head, ...out];
}

/** The full-size EXPERTS' window: `n` consecutive MoE layers (windowCount), 1-n, n+1-2n, … the last ending at the
 * model's last layer (so always `n`), the one holding `focus`: [first, past the last]. */
export const LAYER_WINDOW = 5;   // the window at the board's least height
export function layerWindow(L: number, focus: number, n = LAYER_WINDOW): [number, number] {
	n = Math.max(1, Math.min(L, n));
	const a = Math.min(Math.floor(Math.max(0, Math.min(L - 1, focus)) / n) * n, Math.max(0, L - n));
	return [a, Math.min(L, a + n)];
}

/** The rows of the full-size EXPERTS' lists (EXPERTS FOR; the layer page's two). */
export const LIST_ROWS = 10;
/** The full-size EXPERTS' band under the grids, the same lines at every width: a blank, EXPERTS FOR's heading and its
 * LIST_ROWS rows. */
export const BAND_LINES = 2 + LIST_ROWS;
/** Grids a row in EXPERTS full size, `w` wide, grids `cols` wide: three at most. */
export const windowAcross = (w: number, cols: number) => Math.min(3, Math.floor((w + GAP) / (cols + GAP)));
/** The grid rows EXPERTS full size draws in `rows`, grids `gridRows` half-block rows tall: every whole row that fits
 * over the header, the band and the legend; at least one (the band then gives way). */
function gridRowsFit(rows: number, gridRows: number): number {
	const gh = 1 + Math.ceil(gridRows / 2);
	return Math.max(1, Math.floor((rows - 1 - BAND_LINES - 2) / gh));
}
/** How many layers EXPERTS full size shows, `w` by `rows`, for a model of `L` layers: whole rows of grids
 * (windowAcross a row, gridRowsFit rows). At least one; at most L. */
export function windowCount(L: number, w: number, rows: number, cols: number, gridRows: number): number {
	const across = windowAcross(w, cols);
	if (across < 1) return Math.max(1, Math.min(L, LAYER_WINDOW));
	return Math.max(1, Math.min(L, across * gridRowsFit(rows, gridRows)));
}

/** One layer in the full-size EXPERTS: its lights in its own grid's shape (number order until its map arrives). */
export interface LayerLights {
	layer: number;
	cells: (number | null)[];
	cols: number;
	rows: number;
}

const GAP = 2;

/** EXPERTS full size, `w` by at most `rows`: a line naming the layers (the token in focus on its right), the window's
 * layers (`layers`, windowCount of them) as half-block grids in whole rows, three across where they fit (windowAcross),
 * each under its label, the focus layer's (`v.layer`) ▸ in accent; under them, after a blank, the band (BAND_LINES,
 * the same lines whatever the token or the reply): EXPERTS FOR, the token's own experts at the focus layer (`mine`)
 * with their tokens from the tally (`tops`: by `source`, named in the heading, or both). The band gives way to the
 * first row of grids where both do not fit. `noMap`: the checkpoint has no expert map, said on the naming line's place
 * under it where a line is free. */
export function expertsLayers(th: any, v: ExpertsView, layers: LayerLights[], pal: Palette, w: number, mine: [number, number][],
	sel: number, text: string, held: boolean, rows: number, tops: Tops[] = [undefined], source: "all" | "session" = "all", noMap = false): string[] {
	if (v.reason) return wrapTextWithAnsi(th.fg("dim", v.reason), w).slice(0, rows);
	const cw = Math.max(1, ...layers.map((l) => l.cols)), across = windowAcross(w, cw);
	if (across < 1) return [th.fg("dim", fit(`the grids need ${cw} columns`, w))].slice(0, rows);
	const gh = 1 + Math.max(0, ...layers.map((l) => Math.ceil(l.rows / 2)));
	const shown = layers.slice(0, across * gridRowsFit(rows, gh * 2 - 2));
	const first = shown[0]?.layer ?? v.layer, last = shown.at(-1)?.layer ?? v.layer;
	const span = first === last ? `layer ${first + 1}` : `layers ${first + 1}-${last + 1}`;
	const tag = tokenTag(th, sel, text), names = [`${span} of ${v.L} · ${v.E} experts · ${v.k} mixed for each token`, `${span} of ${v.L} · ${v.E} experts · ${v.k} a token`, `${span} of ${v.L}`];
	const name = names.find((x) => visibleWidth(x) + 2 + visibleWidth(tag) <= w) ?? names.find((x) => visibleWidth(x) <= w) ?? cut(names[2], w);
	const out = [rightOf(th.fg("dim", name), tag, w)];
	const block = (l: LayerLights): string[] => {
		const on = l.layer === v.layer, name = cut((on ? "▸ " : "  ") + LABELS.layerName(l.layer + 1), cw);
		return [on ? th.fg("accent", bold(th, name)) : th.fg("dim", name), ...halfBlocks(l.cells, l.cols, l.rows, pal)];
	};
	for (let at = 0; at < shown.length; at += across) {
		const blocks = shown.slice(at, at + across).map(block);
		for (let i = 0; i < gh; i++) out.push(blocks.map((b) => fit(b[i] ?? "", cw)).join(" ".repeat(GAP)));
	}
	out.push("", ...forTokenBand(th, text, mine, tops, w, -1, tops.length > 1 ? "" : LABELS.source[source]));
	const [a, b] = gridLegend(held, sel);
	if (out.length + 2 <= rows) out.push("", th.fg("faint", fit(`${a}${held ? " " : " · "}${b}`, w)));
	if (noMap && out.length < rows) out.push(th.fg("dim", cut(LABELS.noMap, w)));
	return out.slice(0, rows).map((l) => fit(l, w));
}

/** A token's likeliest choices, likeliest first, at most TOP_N: [text, probability, the one written]. The written
 * token is among them when the stream gave its probability and it is in the first TOP_N (`rank`, from 1). */
export function topChoices(t: Pick<TokenEvent, "text" | "p" | "alts">): { rows: [string, number, boolean][]; rank: number | null } {
	const sent = t.alts ?? [];
	const all: [string, number, boolean][] = sent.map(([a, p]): [string, number, boolean] => [a, p, false]);
	if (t.p !== null) all.push([t.text, t.p, true]);
	all.sort((a, b) => b[1] - a[1] || Number(b[2]) - Number(a[2]));
	const rows = all.slice(0, TOP_N), k = rows.findIndex((r) => r[2]);
	return { rows, rank: k < 0 ? null : k + 1 };
}

/** TOKENS, `w` wide: the token in focus (`#i`, its probability, its rank when another was likelier, the entropy,
 * "off-top" when sampling chose it),
 * then its top choices with a bar and the percentage each, ▸ the one written in its confidence colour (an extra row
 * under them when it was not among them; in `lines` too few for it, the last choice gives way). `titled`: false
 * under the full-size panels' tab row. */
export function tokensPanel(th: any, t: TokenEvent, sel: number, w: number, focused: boolean, titled = true, lines = Infinity): string[] {
	const { rows, rank } = topChoices(t);
	const col = confColor(t.p);
	const where = rank === null ? ` · not in the top ${TOP_N}` : rank > 1 ? ` · rank ${rank}` : "";
	const entropy = typeof t.entropy === "number" ? ` · entropy ${t.entropy.toFixed(2)}` : "";
	const head = (titled ? title(th, "TOKENS", focused) + "  " : "") + th.fg("dim", `#${sel} `) + th.fg(col, cut(label(t.text), 20)) +
		th.fg("dim", `${t.p === null ? "" : ` · ${pct(t.p)}`}${where}${entropy}`);   // the rank before the entropy: kept when cut
	// Off-top (written under half while another had over half: sampling chose it), said only where it fits whole.
	const off = offTop(t.p, t.alts ?? []) ? th.fg("warning", " · off-top") : "";
	const out = [visibleWidth(head) + visibleWidth(off) <= w ? head + off : visibleWidth(head) > w ? truncateToWidth(head, w, "") : head];
	const ww = Math.min(18, Math.max(8, Math.floor(w / 3))), bw = Math.max(4, w - ww - 10);
	const row = (text: string, p: number | null, mine: boolean) => {
		const c = mine ? col : "dim";
		return (mine ? th.fg(col, "▸ ") : "  ") + (mine ? th.fg(col, padTo(label(text), ww)) : padTo(label(text), ww)) + " " +
			th.fg(c, padTo(p === null ? "" : bar(p, bw), bw)) + th.fg(mine ? col : "dim", lpad(pct(p), 7));
	};
	const sent = rows.length < TOP_N && rows.length && (t.alts?.length ?? 0) < TOP_N - 1;
	const room = Math.max(0, lines - 1 - (rank === null ? 1 : 0) - (sent ? 1 : 0));
	for (const [text, p, mine] of rows.slice(0, room)) out.push(row(text, p, mine));
	if (rank === null) out.push(row(t.text, t.p, true));
	if (sent) out.push(th.fg("faint", `  the stream sent ${rows.length} choices for this token`));
	return out.slice(0, lines);
}

// -- SETTLING (in LENS): how deep each recent token settled ---------------------------------------------------------

/** LENS's `rows` lines shared between the telescope (its title included) and SETTLING over it, each [least, most]:
 * the leasts first, then the telescope up to its most, then SETTLING up to its; the rest blank. A blank line parts
 * them, out of SETTLING's share: its most one line more, its least not (tight, SETTLING gives the line). Too few for
 * both leasts: SETTLING shrinks, the blank goes where SETTLING would be left fewer than two lines (its sub-heading and
 * one), then SETTLING goes where those two are not left it; the telescope keeps its least. [telescope, blank,
 * settling]. */
export function lensSplit(rows: number, scope: [number, number], settle: [number, number]): [number, number, number] {
	const room = (n: number) => (n >= 2 ? n : 0);
	const s = room(Math.min(settle[0], rows - scope[0]));
	const t = Math.max(0, Math.min(Math.max(scope[0], scope[1]), rows - s));
	const share = room(Math.min(settle[1] + 1, rows - t)), gap = share >= 3 ? 1 : 0;
	return [t, gap, share - gap];
}

/** How far LENS's colours go toward the dock's background with the lenses off (dull). */
export const DULL = 0.6;
/** `line` with every colour, its plain text's too, `t` of the way to the dock's background: each part where it was,
 * quieter (24-bit or 256 colours, foreground and background). */
export function dull(th: any, line: string, t: number): string {
	const mode = colourMode(th), to = (rgb: Rgb, bg = false) => colourCode(mix(rgb, COLOURS.bg, t), mode, bg), ink = to(COLOURS.text);
	const indexed = (n: string): Rgb => { const { r, g, b } = colorToRgb(indexedColor(+n)); return [r, g, b]; };
	return ink + line.replace(/\x1b\[([34])8;(?:2;(\d+);(\d+);(\d+)|5;(\d+))m|\x1b\[39m/g, (_, k, r, g, b, n) =>
		(k ? to(n !== undefined ? indexed(n) : [+r, +g, +b], k === "4") : ink)) + "\x1b[39m";
}
/** The lenses-off sweep (dullLines): one edge over LENS, ROWS_A_TICK of its lines a tick; a taller LENS (full size) a
 * multiple of that, in SWEEP_MAX_TICKS at most. */
export const ROWS_A_TICK = 2, SWEEP_MAX_TICKS = 12;
/** The sweep over `n` lines: its lines a tick and its ticks. */
export function dullPace(n: number): { rows: number; ticks: number } {
	const rows = ROWS_A_TICK * Math.max(1, Math.ceil(n / ROWS_A_TICK / SWEEP_MAX_TICKS));
	return { rows, ticks: Math.ceil(n / rows) };
}
/** LENS's `lines` `step` ticks of dullPace's into going off (dulled, DULL, where they are; back on, the same ticks the
 * other way round): the first (its title or the step's line) kept, one edge up from the bottom. dock.ts lensLines. */
export function dullLines(th: any, lines: string[], step: number): string[] {
	const edge = lines.length - step * dullPace(lines.length - 1).rows;
	return lines.map((l, i) => (i && i >= edge ? dull(th, l, DULL) : l));
}

/** How the lens `view` read token `t`'s written word at each read layer: 3 the layer's top word from where it settles
 * (lensSettle) to the deepest; 2 its top word below that (it left again); 1 only among the layer's top 10 (the
 * stream's lens top 10s, while sent); 0 not there. */
export function settleLevels(t: Pick<TokenEvent, "id" | "text" | "n_layers" | "lens" | "lens_by" | "lens_logit" | "top" | "top_logit">, view: LensName | undefined): Map<number, number> {
	const { top1, top } = lensOf(t, view), settle = lensSettle(top1, t.id), out = new Map<number, number>();
	for (const [l, id] of top1) {
		const r = top ? top.l.indexOf(l) : -1;
		out.set(l, id === t.id ? (settle !== null && l >= settle ? 3 : 2) : r >= 0 && top!.wp[r] < 255 ? 1 : 0);
	}
	return out;
}
/** SETTLING's ramp, in the lens's violet: from the unlit colour up to the lens's own colour and past it. */
export const SETTLE_STOPS: Stops = [[0, STOPS[0][1]], [0.14, [50, 40, 84]], [0.4, [112, 90, 170]], [0.75, COLOURS.lens], [1, [236, 226, 255]]];
/** A settle level's brightness on SETTLE_STOPS. */
const SETTLE_LIT = [0, 0.14, 0.4, 0.75];
/** The unlit cells of the column of the token in focus. */
const SETTLE_MARK: Rgb = mix(STOPS[0][1], COLOURS.text, 0.45);
/** How far a fogged cell goes toward the unlit colour during the lens swap, by fog level (telescope.ts Frame); toward
 * the background during STREAM's view switch (streamFrames). */
const SETTLE_FOG = [0, 0.5, 0.8];

/** SETTLING's rows: the layers lens `view` read, deep first, on token `k` or the nearest before it with a read (a reply's
 * tokens read the same layers, so the rows stay put through it), and which lens that was; of them, those of the step
 * in use (`want`, as the tube's lit lines: onTube). A layer the lens did not read has no row. */
export function settleLayers(toks: TokenEvent[], k: number, view: LensName | undefined, want?: number[]): { by?: LensName; layers: number[] } {
	let ref = Math.min(k, toks.length - 1);
	while (ref > 0 && !lensOf(toks[ref], view).top1.length) ref--;
	if (!toks[ref]) return { layers: [] };
	const { by, top1 } = lensOf(toks[ref], view);
	return { by, layers: onTube([...new Set(top1.map(([l]) => l))].sort((a, b) => b - a), want) };
}

/** SETTLING, the part of LENS over the telescope, `w` by `rows`: a sub-heading (its name bold and the lens after it, in
 * the lens's violet, as the turret's; "· old → now" dim), then a column per token up to `k` (the newest at the right),
 * the layers the lens shown read of the step's (settleLayers, `want`) deep at the top: `fill` (LENS's) every line
 * filled, no more layers than lines a band of whole lines each, else of half lines; not (the mini dock's) two a line,
 * more to a half line evenly where the lines are too few. Each lit by settleLevels on SETTLE_STOPS; a faint label on
 * the line a band starts, its deepest layer. None read on these tokens: the step's layers (`want`), every cell unlit.
 * `inspecting`: `k` is the token in focus; its column is marked (its unlit cells grey, as MAP's focus is white) and
 * held near the middle while later tokens exist. `swap`: during the lens swap, the cells dimmed toward the unlit colour
 * by `fog` (1 or 2) except the deepest `clear` share (0-1) of the half lines, as the telescope's tube clears from the
 * deep end; the focus mark kept. */
export function settlingPanel(th: any, toks: TokenEvent[], k: number, view: LensName | undefined, w: number, rows: number, _pal: Palette,
	inspecting = false, swap: { fog?: number; clear?: number } = {}, want?: number[], fill = true): string[] {
	const { by, layers: read } = settleLayers(toks, k, view, want), none = !read.length, layers = none ? [...(want ?? [])].sort((a, b) => b - a) : read;
	const head = th.fg("lens", bold(th, "SETTLING")) + th.fg("lens", `  ${by === "logit" ? "logit lens" : by === "jlens" ? "J-lens" : "lens"}`) + th.fg("dim", " · old → now");
	if (!layers.length) return [head, th.fg("dim", "no lens read on these tokens")];   // none read and no step's layers: say so
	// A layer a half line where they fit, else shared; `fill`: every half line drawn, a layer a whole line where the
	// lines hold them.
	const n = layers.length, L = Math.max(1, rows - 1), half = fill ? 2 * L : Math.min(n, 2 * L), whole = fill && n <= L, U = whole ? L : half;
	const cols = Math.max(1, w - 5), at = (h: number) => Math.floor(((whole ? h >> 1 : h) * n) / U), group = (h: number) => {
		const a = at(h), b = Math.floor((((whole ? h >> 1 : h) + 1) * n) / U);
		return layers.slice(a, Math.max(a + 1, b));
	};
	const starts = (h: number) => h === 0 || at(h) !== at(h - 1);
	const cells: (Rgb | null)[] = new Array(half * cols).fill(null);
	const end = inspecting ? Math.min(toks.length - 1, Math.max(k + (cols >> 1), cols - 1)) : k;
	for (let c = 0; c < cols; c++) {
		const i = end - cols + 1 + c, t = toks[i], mark = inspecting && i === k;
		if (mark) for (let h = 0; h < half; h++) cells[h * cols + c] = SETTLE_MARK;
		if (t && none) for (let h = 0; h < half; h++) cells[h * cols + c] = mark ? SETTLE_MARK : rampAt(SETTLE_STOPS, 0);   // none read: unlit
		if (!t || !lensOf(t, view).top1.length) continue;
		const lv = settleLevels(t, view);
		for (let h = 0; h < half; h++) {
			let best = 0;
			for (const l of group(h)) best = Math.max(best, lv.get(l) ?? 0);
			cells[h * cols + c] = mark && !best ? SETTLE_MARK : rampAt(SETTLE_STOPS, SETTLE_LIT[best]);
		}
	}
	const dimBy = SETTLE_FOG[swap.fog ?? 0] ?? 0;
	if (dimBy)
		for (let h = Math.round(Math.max(0, Math.min(1, swap.clear ?? 0)) * half); h < half; h++)
			for (let c = 0; c < cols; c++) {
				const x = cells[h * cols + c];
				if (x && x !== SETTLE_MARK) cells[h * cols + c] = mix(x, STOPS[0][1], dimBy);
			}
	const out = [head];
	rgbBlocks(cells, cols, half, colourMode(th)).forEach((g, i) => {
		const h = starts(2 * i) ? 2 * i : starts(2 * i + 1) ? 2 * i + 1 : -1;
		out.push(th.fg("faint", padTo(h < 0 ? "" : `L${group(h)[0]}`, 5)) + g);
	});
	return out.slice(0, Math.max(1, rows));
}

// -- MAP: the whole reply by confidence ---------------------------------------------------------------------------

const BLANK = STOPS[0][1];
/** MAP's colours: sure, middling, unsure, each toned down toward the unlit colour, more while thinking (mixed once). */
const shades = (g: number, m: number, b: number): Rgb[] => [mix(BLANK, COLOURS.good, g), mix(BLANK, COLOURS.middling, m), mix(BLANK, COLOURS.bad, b)];
const MAP_SHADE = { answer: shades(0.45, 0.7, 1), think: shades(0.25, 0.4, 0.6) };
const shadeOf = (p: number, think: boolean): Rgb => {
	const c = confColor(p);
	return MAP_SHADE[think ? "think" : "answer"][c === "success" ? 0 : c === "warning" ? 1 : 2];
};

/** One reply in MAP's cells: its `n` tokens' probabilities (`p`), where its thinking ends (-1: none), its model;
 * `shade` each token's colour instead (engram inspection's, lookup.ts; undefined for a token: its own). */
export interface MapRun {
	n: number;
	p: (i: number) => number;
	think: number;
	model?: string;
	shade?: (i: number) => Rgb | undefined;
}
/** MAP in engram inspection (lookup.ts): each token's colour, the engram's tokens [first, past the last] on its
 * block and the token it predicted (-1 none) marked (paintMarks), the count in the title instead of the unsure one. */
export interface MapPaint {
	shade: (i: number) => Rgb | undefined;
	engram: [number, number];
	pred: number;
	block: Rgb;
	count?: string;
}
/** MAP's `cells` (the window from cell `a`, the reply's token 0 at cell `at0`) with a MapPaint's marks: the engram's
 * tokens in its block colour, the token it predicted in the text colour. */
function paintMarks(cells: Rgb[], p: MapPaint | undefined, at0: number, a: number): Rgb[] {
	const put = (i: number, rgb: Rgb) => { const c = at0 + i - a; if (c >= 0 && c < cells.length) cells[c] = rgb; };
	if (p) for (let i = p.engram[0]; i < p.engram[1]; i++) put(i, p.block);
	if (p && p.pred >= 0) put(p.pred, COLOURS.text);
	return cells;
}
/** A reply's tokens as a MAP run, and its unsure count (draw.ts replyCounts: counted as it grows). */
function tokenRun(toks: { text: string; p: number | null }[], model?: string): { run: MapRun; unsure: number } {
	const c = replyCounts(toks);
	return { run: { n: toks.length, p: (i) => toks[i].p ?? 0, think: c.think, model }, unsure: c.unsure.length };
}
/** A kept reply's unsure count, by its probabilities (a kept reply never changes: counted once). */
const keptUnsure = new WeakMap<object, number>();
/** A kept reply (state.ts MapReply) as a MAP run, and its unsure count. */
function keptRun(m: MapReply): { run: MapRun; unsure: number } {
	const p = (i: number) => m.p[i] / 200;
	let unsure = keptUnsure.get(m.p);
	if (unsure === undefined) {
		unsure = 0;
		for (let i = 0; i < m.p.length; i++) if (p(i) < UNSURE) unsure++;
		keptUnsure.set(m.p, unsure);
	}
	return { run: { n: m.p.length, p, think: m.think, model: m.model }, unsure };
}

/** Where MAP's replies `rs` start in its cells, rows `w` cells wide: each on a row of its own (the rest of the row
 * before it unlit), after an amber cell (`marks`) where its model is not the last one known before it; `end` the cell
 * past the last token. */
export function mapStarts(rs: MapRun[], w: number): { starts: number[]; marks: number[]; end: number } {
	const starts: number[] = [], marks: number[] = [];
	let at = 0, known: string | undefined;
	rs.forEach((r, k) => {
		if (k) {
			at = Math.ceil(at / w) * w;
			if (r.model !== undefined && known !== undefined && r.model !== known) marks.push(at++);
		}
		if (r.model !== undefined) known = r.model;
		starts.push(at);
		at += r.n;
	});
	return { starts, marks, end: at };
}

/** MAP's cells, `n` of them from cell `a`, rows `w` wide: replies `rs` laid out by mapStarts, a token a cell in its
 * confidence colour (its reply's thinking darker; its `shade` where it has one), cell `focus` (-1: none) in the text
 * colour, the rest unlit. Only the cells asked for are worked out. */
export function mapCells(rs: MapRun[], w: number, a: number, n: number, focus = -1): Rgb[] {
	const cells: Rgb[] = new Array(n).fill(BLANK), { starts, marks, end } = mapStarts(rs, w);
	const put = (c: number, rgb: Rgb) => { if (c >= a && c < a + n) cells[c - a] = rgb; };
	for (const c of marks) put(c, COLOURS.accent);
	rs.forEach((r, k) => {
		const at = starts[k];
		for (let i = Math.max(0, a - at), to = Math.min(r.n, a + n - at); i < to; i++) cells[at + i - a] = r.shade?.(i) ?? shadeOf(r.p(i), i <= r.think);
	});
	if (focus < end) put(focus, COLOURS.text);
	return cells;
}

/** MAP's cell lines on the board: at the least (MAP_LINES, its 192 tokens at 104 columns) and at the most (it takes
 * its column's rest up to these): a token half a character, 2w tokens a line, so 7 lines hold 672 tokens at 104 columns. */
export const MAP_LINES = 2, MAP_CELL_LINES = 7;
/** MAP under the mini dock's lights (miniSplit): its title and a line at the least, 3 lines (the newest 300 tokens at
 * 50 columns, the right column's) where the height allows, and no more; without lights (a model without experts) MAP
 * takes the lines. */
export const MINI_MAP: MiniRoom = [2, 4, 4];
/** SETTLING under the mini dock's lights at the least: its sub-heading and two lines (four layers' rows). */
export const MINI_SETTLE_LEAST = 3;

/** The first line MAP shows (a line holds two rows of w cells, a token half a character) of `n` cells in `lines` lines:
 * live (`focus` -1) the newest; stepped in, `prev` moved by whole lines just enough to hold cell `focus`. The window
 * never starts past where the newest line is its last. */
export function mapWindow(n: number, w: number, lines: number, focus: number, prev = 0): number {
	const per = 2 * Math.max(1, w);
	return scrollWindow(Math.ceil(n / per), lines, focus < 0 ? -1 : Math.floor(focus / per), prev);
}
/** The first of `lines` rows shown of `n`, from `prev`, moved the least that keeps row `at` in view with `margin` rows
 * either side where the list allows (`at` < 0: the last rows). MAP's lines; the layer page's list. */
export function scrollWindow(n: number, lines: number, at: number, prev = 0, margin = 0): number {
	const top = Math.max(0, n - Math.max(1, lines)), m = Math.max(0, Math.min(margin, Math.floor((lines - 1) / 2)));
	if (at < 0) return top;
	let first = Math.max(0, Math.min(prev, top));
	if (at - m < first) first = at - m;
	else if (at + m >= first + lines) first = at + m - lines + 1;
	return Math.max(0, Math.min(first, top));
}

/** MAP on the board and the mini dock, `w` by at most `rows`: its title the reply's size and unsure count, then from
 * line `first` (mapWindow) the reply a token half a character (mapCells, rgbBlocks) in `rows` - 1 lines, the lines past it
 * unlit, the title naming the tokens shown when that is not the whole reply; `focus` the token marked (-1 none).
 * `paint`: engram inspection's colours and marks instead, the token it predicted marked in place of the white focus
 * (MapPaint; a token not read keeps it). */
export function mapPanel(th: any, toks: { text: string; p: number | null }[], focus: number, w: number, rows: number, focused: boolean, titled = true,
	first = 0, paint?: MapPaint): string[] {
	const name = titled ? title(th, "MAP", focused) + "  " : "", lines = Math.max(1, rows - 1), n = 2 * w * lines, a = first * 2 * w;
	const { run, unsure } = tokenRun(toks);
	const parts = [`${toks.length} tok`, paint?.count ?? `${unsure} unsure`,...(a > 0 || a + n < toks.length ? [`${a + 1}-${Math.min(toks.length, a + n)}`] : [])];
	// Where the width is short: unsure goes first, then the range.
	while (parts.length > 1 && visibleWidth(name) + visibleWidth(parts.join(" · ")) > w) parts.splice(1, 1);
	const cells = mapCells([paint ? { ...run, shade: paint.shade } : run], w, a, n, paint && paint.pred >= 0 ? -1 : focus);
	return [name + th.fg("dim", parts.join(" · ")), ...rgbBlocks(paintMarks(cells, paint, 0, a), w, 2 * lines, colourMode(th))];
}

/** MAP full size, `w` by `rows`: every reply of this session, oldest first (Live.map's, `r`, the one shown, where
 * state.ts mapPlace puts it), a token half a character (mapCells, rgbBlocks) in the `rows` - 1 lines under its title
 * (the replies, tokens and unsure ones). It fills from the top; once full, from line `prev` moved the least that keeps
 * token `sel` of `r` in view (-1: the newest line last; mapWindow). The lines, the first line shown and `r`'s tokens in
 * view [first, past the last] (none: [0, 0]). A frame's work follows the replies kept and the cells shown, not their
 * tokens. `paint`: `r` in engram inspection (MapPaint; the title kept). */
export function sessionMap(th: any, map: MapReply[] = [], r: Reply, sel: number, w: number, rows: number, prev = 0,
	paint?: MapPaint): { lines: string[]; first: number; shown: [number, number] } {
	const kept = map.map(keptRun), { at, replace } = mapPlace(map, r), runs = kept.map((k) => k.run), lines = Math.max(1, rows - 1);
	let unsure = kept.reduce((s, k) => s + k.unsure, 0);
	if (at >= 0) {
		const shown = tokenRun(r.tokens, r.model);
		if (replace) unsure -= kept[at].unsure;
		runs.splice(at, replace ? 1 : 0, paint ? { ...shown.run, shade: paint.shade } : shown.run);
		unsure += shown.unsure;
	}
	const { starts, end } = mapStarts(runs, w), tok = runs.reduce((s, x) => s + x.n, 0);
	const focus = sel < 0 || at < 0 ? -1 : starts[at] + sel, first = mapWindow(end, w, lines, focus, prev), n = runs.length;
	const parts = [`${n} repl${n === 1 ? "y" : "ies"}`, `${tok} tok`, `${unsure} unsure`];
	while (parts.length > 1 && visibleWidth(parts.join(" · ")) > w) parts.pop();
	const per = 2 * w, shown = at >= 0 ? paint : undefined, cs = mapCells(runs, w, first * per, per * lines, shown && shown.pred >= 0 ? -1 : focus);
	const a = at < 0 ? 0 : Math.max(0, Math.min(r.tokens.length, first * per - starts[at])), b = at < 0 ? 0 : Math.max(a, Math.min(r.tokens.length, (first + lines) * per - starts[at]));
	return { lines: [th.fg("dim", parts.join(" · ")), ...rgbBlocks(paintMarks(cs, shown, starts[at], first * per), w, 2 * lines, colourMode(th))], first, shown: [a, b] };
}

/** The Steer page's list and detail, `w` wide: on the left the concepts by the chat test, two columns of `rows`, ★ the
 * top five, the cursor's ▸; on the right the cursor's concept: its dose curve and what the dose under the cursor means,
 * the last apply's note; narrow, the list over the rest (steerColumns). Only the chat test ranks: the untested come
 * last, counted as not calibrated on this build (each build has its own chat doses). */
export interface SteerView {
	list?: SteerList;
	error?: string;
	model?: string;
	cur: number;
	dose: number;           // an index into the cursor's concept's doses
	on: Map<string, number>;
	nLayers?: number;
	note?: Note;
	focused: boolean;
	offer?: LinkOffer | null;   // l's link offer, shown on an empty list (offerLines)
}

/** A reading or a note `w` wide in colour `col`: its first line on at most two (the second cut, marked, where it runs
 * over), and the same for a line after a newline (the server's note under a link's). */
function twoLines(th: any, s: string, w: number, col: string): string[] {
	const two = (t: string) => { const ls = wrapTextWithAnsi(t, w); return ls.length > 2 ? [ls[0], cut(ls.slice(1).join(" "), w)] : ls; };
	const [head, ...rest] = s.split("\n");
	return [...two(head), ...(rest.length ? two(rest.join(" ")) : [])].map((l) => th.fg(col, l));
}
/** A note's colour where it stands alone on the page: an error or warning as itself, else dim. */
const noteColour = (n: Note) => (n.level === "error" || n.level === "warning" ? n.level : "dim");

/** How many of the link offer's candidates the Steer page lists (the pick list has them all). */
const OFFER_MOST = 5;

/** The Steer page with nothing learned on the served model and same-shape models with data (`o`): the model, what it
 * has, the candidates, then what l does; the note under it, on up to two lines. */
function offerLines(th: any, o: LinkOffer, w: number, titled: boolean, focused: boolean, note?: Note): string[] {
	const one = o.candidates.length === 1, more = o.candidates.length - OFFER_MOST;
	const shape = o.shape ? ` (${o.shape.n_layers} layers, width ${o.shape.width})` : "";
	const dim = (s: string) => wrapTextWithAnsi(s, w).map((l) => th.fg("dim", l));
	return [(titled ? title(th, "STEER", focused) + th.fg("dim", "· ") : "") + o.model, "", ...wrapTextWithAnsi("Nothing learned on this model yet.", w), "",
		...dim(`Same shape as ${one ? "a model" : "models"} you have${shape}:`),
		...o.candidates.slice(0, OFFER_MOST).map((c) => cut(`  ${linkLabel(c)}`, w)), ...(more > 0 ? dim(`  and ${more} more`) : []),
		"", ...dim(`l link to ${one ? "it" : "one"} · /concept learn NAME starts fresh`),
		...(note ? ["", ...twoLines(th, note.text, w, noteColour(note))] : [])];
}

export function steerPanel(th: any, v: SteerView, w: number, titled = true, rows = STEER_ROWS): string[] {
	const { sl, sr, stacked, across } = steerColumns(w);
	const left = [(titled ? title(th, "STEER", v.focused) + "  " : "") + th.fg("dim", "by the chat test · ★ top five")];
	const C = v.list?.concepts;
	if (v.offer && v.list && !C?.length) {
		const out = offerLines(th, v.offer, w, titled, v.focused, v.note);
		while (out.length < rows + 2) out.push("");
		return out.map((l) => fit(l, w));
	}
	if (!C || !C.length) {
		const say = v.error ? th.fg("error", v.error) : !v.list ? th.fg("dim", `asking for ${v.model || "the model"}'s concepts…`)
			: th.fg("dim", `no learned concepts on ${v.list.model || "this model"}: /concept learn NAME teaches one`);
		const out = [left[0], ...wrapTextWithAnsi(say, w).slice(0, 2)];
		if (v.note) out.push(...twoLines(th, v.note.text, w, noteColour(v.note)));
		while (out.length < rows + 2) out.push("");
		return out.map((l) => fit(l, w));
	}
	const page = across * rows, cur = Math.max(0, Math.min(C.length - 1, v.cur)), top = Math.floor(cur / page) * page;
	for (let r = 0; r < rows; r++) {
		let line = "";
		for (let col = 0; col < across; col++) {
			const i = top + col * rows + r, x = C[i];
			if (!x) { line += " ".repeat(26); continue; }
			const me = i === cur, on = v.on.has(x.name);
			const tag = x.chat ? `${x.good}/${x.turns} @${fmt(x.pick)}` : "untested";
			const text = (me ? "▸ " : starred(x) ? "★ " : "  ") + nameTag(x.name, tag, x.chat ? `${x.good}/${x.turns}` : "");
			const paint = me && v.focused ? (s: string) => th.fg("accent", bold(th, ansiInverse(s))) : on ? (s: string) => th.fg("accent", s)
				: me ? (s: string) => bold(th, s) : x.chat ? (s: string) => s : (s: string) => th.fg("faint", s);
			line += paint(text) + " ";
		}
		left.push(line);
	}
	const nText = C.filter((x) => !x.chat).length, end = Math.min(C.length, top + page);
	const range = C.length > page ? `${top + 1}-${end} of ${C.length} · ` : "";
	const count = (nText ? [`${nText} not calibrated on this build`, `${nText} not calibrated here`, `${nText} not calibrated`] : [`${C.length} chat-tested`])
		.map((x) => `  ${range}${x}`);
	left.push(th.fg("faint", cut(count.find((x) => visibleWidth(x) <= sl) ?? count.at(-1)!, sl)));

	const c = C[cur], k = Math.max(0, Math.min(c.doses.length - 1, v.dose)), d = c.doses[k];
	const onDose = v.on.get(c.name);
	const right = [bold(th, cut(c.title, Math.max(8, sr - 24))) + th.fg("dim", c.layer === null ? " · no layer: /steer refuses it"
		: ` · pushes at L${c.layer}${v.nLayers ? ` of ${v.nLayers}` : ""}`) + (onDose !== undefined ? th.fg("accent", `  on at ${fmt(onDose)}`) : "")];
	const desc = c.description ? wrapTextWithAnsi(c.description, sr) : [];
	right.push(th.fg("dim", desc[0] ?? ""), th.fg("dim", desc.length > 2 ? cut(desc[1] + " " + desc[2], sr) : desc[1] ?? ""));
	if (c.chat) {      // the dose curve and the dose under the cursor (a concept never chat-tested has none)
		const n = c.doses.length, cw = Math.max(3, Math.min(6, Math.floor((sr - 7) / Math.max(1, n))));
		const lab = (s: string) => th.fg("dim", s.padEnd(7));
		right.push(lab("dose") + c.doses.map((x, i) => (i === k ? th.fg("accent", bold(th, ansiInverse(centre(fmt(x.dose), cw)))) : centre(fmt(x.dose), cw))).join(""));
		const bars = (col: string, f: (x: DoseRow) => number | null) => c.doses.map((x) => th.fg(col, centre(block(f(x)), cw))).join("");
		right.push(lab("good") + bars("success", (x) => (x.turns ? (x.good ?? 0) / x.turns : null)));
		right.push(lab("style") + bars("accent", (x) => (x.style === null ? null : x.style / 3)));
		right.push(lab("sense") + bars("dim", (x) => (x.sense === null ? null : x.sense / 3)));
		const [said, rest] = doseLine(c, d);
		right.push(bold(th, cut(said, sr)) + th.fg("dim", cut(rest, Math.max(0, sr - visibleWidth(said)))));
	}
	// The reading and the last apply's note, each on up to two lines, the second cut short with … (enter on an untested
	// concept notes its reading: said once, as the reading).
	const [col, text] = reading(c, d, [...v.on.keys()]);
	right.push(...twoLines(th, text, sr, col));
	if (v.note && v.note.text !== text) right.push(...twoLines(th, (v.note.level === "info" ? "✓ " : "") + v.note.text, sr, v.note.level === "info" ? "success" : v.note.level === "busy" ? "dim" : v.note.level));
	return stacked ? [...left, ...right].map((l) => fit(l, w)) : side(th, left, right, sl, sr);
}

/** A STEER list entry after its cursor or star, 23 columns: the name (11), its tag right-aligned (10).
 * A longer name stays whole (up to 21) and takes the tag's room: the tag if it still fits, else `short` (the good
 * turns alone), else none. */
export function nameTag(name: string, tag: string, short: string): string {
	if (visibleWidth(name) <= 11) return padTo(name, 11) + "  " + lpad(tag, 10);
	const n = cut(name, 21), room = 23 - visibleWidth(n) - 2, t = [tag, short, ""].find((s) => visibleWidth(s) <= room) ?? "";
	return n + "  " + lpad(t, room);
}

/** The example chats on the Steer page, `w` wide in `rows` lines: from turn `i` of the chat test, its conversation's
 * turns stacked, each its prompt, then the reply with no steering and the steered one side by side (Markdown, whole;
 * narrow, steerColumns' stacked, one over the other);
 * `plainOnly` (no steering under the cursor): the unsteered reply alone. The stack is scrolled by `scroll` (page keys);
 * where it does not fit, the last line says how much is below. Returns the lines and how far the scroll goes. */
export function examplesPanel(th: any, ex: Examples, i: number, scroll: number, md: (text: string, width: number, colour?: string) => string[],
	w: number, rows: number, plainOnly = false): { lines: string[]; maxScroll: number } {
	i = Math.max(0, Math.min(ex.turns.length - 1, i));
	const t0 = ex.turns[i];
	if (!t0) return { lines: [th.fg("dim", "no test chats saved for this concept")], maxScroll: 0 };
	const run = [t0];
	for (let j = i + 1; j < ex.turns.length && ex.turns[j].conv === t0.conv && ex.turns[j].n > ex.turns[j - 1].n; j++) run.push(ex.turns[j]);
	const { lw, rw } = columns(w), stacked = steerColumns(w).stacked;
	const score = (s: Side) => (s.style === null ? "not graded" : `style ${s.style} · sense ${s.sense ?? "–"}`);
	const plainHead = (t: ExampleTurn) => th.fg("dim", bold(th, "no steering")) + th.fg("faint", "  " + score(t.plain));
	// Every turn whole: the line each starts on (`starts`), so the heading can say which are in view.
	const body: string[] = [], starts: number[] = [];
	for (const t of run) {
		if (body.length) body.push("");
		starts.push(body.length);
		const user = wrapTextWithAnsi(t.user.replace(/\s+/g, " "), w - 8);
		user.forEach((l, k) => body.push(th.fg("warning", k ? "        " : `turn ${t.n}`.padEnd(8)) + bold(th, l)));
		body.push("");
		if (plainOnly) body.push(plainHead(t), ...md(t.plain.text, w, "muted"));
		else {
			const right = th.fg("accent", bold(th, `steered at ${fmt(ex.dose)}`)) +
				th.fg(t.steered.good ? "success" : "faint", "  " + (t.steered.good ? "✓ good · " : "") + score(t.steered));
			if (stacked) body.push(plainHead(t), ...md(t.plain.text, w, "muted"), "", right, ...md(t.steered.text, w));
			else body.push(...side(th, [plainHead(t), ...md(t.plain.text, lw, "muted")], [right, ...md(t.steered.text, rw)], lw, rw));
		}
	}
	const room = Math.max(1, rows - 1), over = body.length > room, cap = over ? Math.max(1, room - 1) : room;
	const maxScroll = Math.max(0, body.length - cap), s = Math.max(0, Math.min(maxScroll, scroll));
	const shown = body.slice(s, s + cap);
	const turnAt = (line: number) => run[Math.max(0, starts.filter((x) => x <= line).length - 1)].n;
	const a = turnAt(s), b = turnAt(Math.min(body.length, s + cap) - 1);
	const turn = `turn ${a}${b > a ? `-${b}` : ""} of ${t0.of} · ${i + 1}/${ex.turns.length}`, keys = " · n next · N back", space = w - 8 - 2 - visibleWidth(keys);
	const what = [`${t0.conv} · ${turn}`, turn].find((x) => visibleWidth(x) <= space) ?? cut(turn, Math.max(1, space));
	const head = th.fg("accent", bold(th, "EXAMPLES")) + th.fg("dim", `  ${what}${keys}`);
	const out = [fit(head, w), ...shown];
	if (over) {
		const below = body.length - s - cap;
		out.push(th.fg("faint", `${below > 0 ? `… ${below} more line${below === 1 ? "" : "s"}` : "end"}${s ? ` · ${s} above` : ""} (page keys scroll)`));
	}
	return { lines: out, maxScroll };
}

// -- STREAM: what each layer changed in the residual stream (RESIDUAL STREAM, a piano roll under EXPERTS) -----------

/** STREAM's views, in the order s steps through them, each where the model's data has it (streamViews). */
export const STREAM_VIEWS = ["spikes", "largest", "reads"] as const;
export type StreamView = (typeof STREAM_VIEWS)[number];
/** A remembered view (pi.json), undefined for anything else. */
export const storedView = (x: unknown): StreamView | undefined => (STREAM_VIEWS.includes(x as StreamView) ? (x as StreamView) : undefined);
/** The views a model's data allows: spikes with writes (into `k` streams), largest with writes into more than one,
 * reads with reads (from `rk` sources). */
export const streamViews = (k: number, rk: number): StreamView[] => STREAM_VIEWS.filter((v) => (v === "spikes" ? k > 0 : v === "largest" ? k > 1 : rk > 0));
/** The view drawn: the one stored where the model has it, else the first it has (spikes where it writes). */
export const viewOf = (stored: StreamView | undefined, views: StreamView[]): StreamView => (stored && views.includes(stored) ? stored : views[0]);
/** The view after the one drawn (s). */
export const nextView = (stored: StreamView | undefined, views: StreamView[]): StreamView => views[(views.indexOf(viewOf(stored, views)) + 1) % views.length];
/** STREAM's view switch from `a` to `b` (s), on the lens swap's schedule (telescope.ts SWAP_FOG): the old view fogs,
 * then the new one clears from the deep end (`clear`: the deepest layers' share); the last frame clear. */
export const streamFrames = (a: StreamView, b: StreamView): { stream: StreamView; fog?: number; clear?: number }[] =>
	SWAP_FOG.map((f) => ({ stream: f.next ? b : a, fog: f.fog, clear: f.clear }));
/** The streams' (or sources') colours, by index (more take them again). */
export const STREAM_RGB: Rgb[] = [[255, 180, 84], [86, 204, 242], [244, 114, 182], [155, 229, 100], [167, 139, 250]];
/** STREAM's look: `largest`'s lights fading with age by the power `age`; a beam's head whitened by `head`, the beam
 * fading with age by the power `beam`, its tail's cells at `tail` less `step` a cell back; KEY_ALT `alt` toward the
 * background. */
export const STREAM_LOOK = { age: 1.2, head: 0.35, beam: 1.4, tail: 0.62, step: 0.08, alt: 0.4 };
/** A key unlit; every other one a shade darker, so keys with no gap between them still read as keys. */
const KEY: Rgb = [37, 45, 58], KEY_ALT = mix(KEY, COLOURS.bg, STREAM_LOOK.alt), WHITE: Rgb = [255, 255, 255];
/** STREAM's history lines (two tokens a line) where the board has room, the fewest it is drawn with, its own lines
 * past the history (its title, keys and depth marks), and those with the rule over it in the right column. */
export const STREAM_LINES = 8, STREAM_MIN = 3, STREAM_FIXED = 3, STREAM_FRAME = 1 + STREAM_FIXED;
/** STREAM full size: the oldest line at 1 - FULL_FADE of the newest's strength (the column's fades all the way), so
 * old lines stay readable. */
export const FULL_FADE = 0.6;
/** STREAM's history lines with `room` lines left under what keeps its least over it (the board's right column: TOKENS
 * and EXPERTS, dock.ts rightLeast; the mini dock: a grid of lights, MAP and SETTLING), its `frame` lines taken: STREAM_LINES,
 * fewer before those fall under their least, none under STREAM_MIN (it gives way). */
export function streamLines(room: number, frame = STREAM_FRAME, most = STREAM_LINES): number {
	const n = Math.min(most, room - frame);
	return n >= STREAM_MIN ? n : 0;
}
/** The history lines STREAM takes on a mini dock `rows` tall before the lights: a quarter of its rows, STREAM_LINES at
 * least, twice that at most (fewer where they would leave under two layers of lights; the lines left over after, up to
 * twice STREAM_LINES in all: dock.ts renderMini). */
export const miniStreamLines = (rows: number) => Math.max(STREAM_LINES, Math.min(2 * STREAM_LINES, Math.floor(rows / 4)));
/** A key's columns for `L` layers in `w`: two, one, or none (more layers than columns: streamPanel spreads them). */
export const keyWidth = (w: number, L: number) => (L < 1 ? 0 : w >= 2 * L ? 2 : w >= L ? 1 : 0);
/** Where the blocks over RESIDUAL STREAM in a column `w` wide end (EXPERTS' grids; the mini dock's MAP and SETTLING
 * too), STREAM drawn over `L` layers (0: not drawn): its keys' edge where they fall short of `w` by 4 columns at most,
 * the grids then full rectangles that wide (expertsStack's plate); else 0, the column's width as ever. */
export const keysEdge = (w: number, L: number) => { const k = L * keyWidth(w, L); return k && w - k <= 4 ? k : 0; };
type StreamLights = Pick<TokenEvent, "strikes" | "largest" | "readStrikes">;
/** `L` layers on `W` columns, more layers than columns: column c's layers from `from(c)` up to `to(c)`, floor(c L / W)
 * to floor((c + 1) L / W), one at least. */
const span = (L: number, W: number) => {
	const from = (c: number) => Math.floor((c * L) / W);
	return { from, to: (c: number) => from(c + 1) };
};
/** Token `t`'s lights of `L` layers spread over `W` columns, fewer than the layers (span), each column the strongest
 * of its layers: a strike (the longest beam), the brightest largest. */
export function spread(t: StreamLights, L: number, W: number): StreamLights {
	const { from, to } = span(L, W);
	const strikes = (x?: Uint16Array) => {
		if (!x) return x;
		const out: number[] = [];
		for (let c = 0; c < W; c++) {
			let best = -1;
			for (let i = 0; i + 2 < x.length; i += 3) if (x[i] >= from(c) && x[i] < to(c) && (best < 0 || x[i + 2] > x[best + 2])) best = i;
			if (best >= 0) out.push(c, x[best + 1], x[best + 2]);
		}
		return Uint16Array.from(out);
	};
	const b = t.largest, top = (c: number) => {
		let m = b![from(c)] ?? 0;
		for (let l = from(c) + 1; l < to(c); l++) if (unpackLargest(b![l] ?? 0)[1] > unpackLargest(m)[1]) m = b![l];
		return m;
	};
	return { strikes: strikes(t.strikes), readStrikes: strikes(t.readStrikes), largest: b && Uint16Array.from({ length: W }, (_, c) => top(c)) };
}

/** The names of `n` read sources (Inkling 2's resid_read_share slots): the embedding, the finished blocks, the
 * running sum. */
const sourceNames = (n: number) => Array.from({ length: n }, (_, j) => (!j ? "embedding" : j === n - 1 ? "running sum" : `block ${j}`));
/** Whether STREAM full size has a legend line in `view` on a model writing `k` streams: a colour a stream, or a source. */
export const hasLegend = (view: StreamView, k: number) => view === "reads" || k > 1;

/** RESIDUAL STREAM's title, `w` wide (inverse `focused`): what the model has (`s`: `k` streams written, `rk` sources
 * read; one stream with reads is a block's sum, Inkling 2's) and what is tracked (the view; ` · s`, faint, with
 * `swap`: s switches it), and `note` (dim: why it is dark) after them; where all do not fit, "tracking" goes first,
 * then the count, then the note in their place. */
function streamTitle(th: any, s: { k: number; rk: number }, view: StreamView, w: number, o: { note?: string; focused?: boolean; swap?: boolean }): string {
	const name = title(th, "RESIDUAL STREAM", !!o.focused), why = o.note ? th.fg("dim", `  ${o.note}`) : "";
	const has = s.k > 1 ? `${s.k} streams` : view === "reads" ? `${s.rk} sources` : s.rk ? "block sum" : "1 stream";
	const all = s.k > 1 ? `hyper-connections: ${s.k} residual streams` : s.rk ? has : "1 residual stream";
	const line = (t: string) => name + th.fg("dim", `  ${t}`) + (o.swap ? th.fg("faint", " · s") : "") + why;
	return [`${all} · tracking ${view}`, `${has} · tracking ${view}`, `${has} · ${view}`, view].map(line).find((l) => visibleWidth(l) <= w)
		?? (o.note ? name + why : line(view));
}
/** RESIDUAL STREAM, `w` wide, `lines` + STREAM_FIXED lines (`full`: its legend's line, where it has one, out of
 * `lines`): its title (streamTitle), then the history of token `k` and those before it in `lines` lines of half blocks
 * (rgbBlocks), two tokens a line, the newest lowest; a key a layer of the model's `s.L`, shallow on the left (keyWidth
 * columns each), lit in the stream's colour while struck; faint depth marks every 8 layers (L0, L8, as the telescope
 * numbers them), more apart where 8 layers take under 8 columns. More layers than `w`: spread over it (spread), a
 * column a key. Each token's lights as packWrites decided them. `spikes` (TokenEvent.strikes) and `reads`
 * (readStrikes, by source): a struck layer's beam trails behind its head, the head whiter, all fading with age; a key
 * glows a few tokens after its strike. `largest`: every layer lit in its largest stream's colour at its brightness,
 * fading with age; the keys as the newest token (STREAM_LOOK). `full` (full size): a swatch and a name for each colour
 * in use (hasLegend); the keys as the right column draws them (RIGHT_W: two columns a layer up to 25, one up to the
 * width, spread past it); the age fading only part way (FULL_FADE). `fog` (1 or 2, the view switch: streamFrames):
 * every layer's lights and key toward the background (SETTLE_FOG) but the deepest `clear` share (0-1) of them. */
export function streamPanel(th: any, toks: StreamLights[], k: number, s: { L: number; k: number; rk: number }, view: StreamView, w: number, lines: number,
	o: { note?: string; focused?: boolean; swap?: boolean; full?: boolean; fog?: number; clear?: number } = {}): string[] {
	const keyRoom = o.full ? Math.min(w, Math.max(RIGHT_W, s.L)) : w;   // the width the keys are fitted to
	// The keys: a layer each, or (`fill`: more layers than columns) a column each over the layers spread.
	const layers = s.L, fill = !keyWidth(keyRoom, layers), nKeys = fill ? w : layers, tok = (i: number) => (fill && toks[i] ? spread(toks[i], layers, w) : toks[i]);
	const legend = !!o.full && hasLegend(view, s.k), look = STREAM_LOOK;
	const kw = fill ? 1 : keyWidth(keyRoom, nKeys), R = 2 * Math.max(1, lines - +legend), cols = nKeys * kw, fade = o.full ? FULL_FADE : 1;
	const cells: (Rgb | null)[] = new Array(R * cols).fill(null), glow = new Float32Array(nKeys), hue = new Int16Array(nKeys);
	// `age` tokens back, alpha `a` of `rgb` over what is there (the dock's background where nothing is).
	const paint = (age: number, key: number, rgb: Rgb, a: number) => {
		if (age < 0 || age >= R || key >= nKeys) return;
		const at = (R - 1 - age) * cols + key * kw, c = mix(cells[at] ?? COLOURS.bg, rgb, a);
		for (let j = 0; j < kw; j++) cells[at + j] = c;
	};
	const rgb = (stream: number) => STREAM_RGB[stream % STREAM_RGB.length];
	if (view === "largest")
		for (let a = 0; a < R; a++) {
			const b = tok(k - a)?.largest;
			if (b) for (let l = 0; l < Math.min(nKeys, b.length); l++) {
				const [st, v] = unpackLargest(b[l]);
				paint(a, l, rgb(st), v * Math.pow(1 - (fade * a) / R, look.age));
				if (a === 0) [glow[l], hue[l]] = [v, st];
			}
		}
	else
		// The oldest first, so newer beams lie over older ones; a beam's head at its token's age, its tail the cells after.
		for (let a = R + WRITE_BEAM - 1; a >= 0; a--) {
			const t = tok(k - a), x = view === "reads" ? t?.readStrikes : t?.strikes;
			for (let i = 0; x && i + 2 < x.length; i += 3) {
				const [l, st, len] = [x[i], x[i + 1], x[i + 2]];
				for (let j = 0; j < len; j++) {
					const r = a - j, near = 1 - (fade * r) / R;
					if (r >= 0 && r < R) paint(r, l, j ? rgb(st) : mix(rgb(st), WHITE, look.head * near), Math.pow(near, look.beam) * (j ? look.tail - look.step * j : 1));
				}
				const f = 1 - a / (len + 2);
				if (f > 0 && l < nKeys && f > glow[l]) [glow[l], hue[l]] = [f, st];
			}
		}
	// The view switch's fog: the keys shallower than the clear share toward the background, their lights too.
	const dimBy = SETTLE_FOG[o.fog ?? 0] ?? 0, fogTo = nKeys - Math.round(nKeys * (o.clear ?? 0));
	const fogged = (c: Rgb, key: number) => (dimBy && key < fogTo ? mix(c, COLOURS.bg, dimBy) : c);
	if (dimBy) cells.forEach((c, i) => c && (cells[i] = fogged(c, Math.floor((i % cols) / kw))));
	const mode = colourMode(th), keyRgb = (l: number) => fogged(glow[l] > 0 ? mix(l % 2 ? KEY_ALT : KEY, rgb(hue[l]), glow[l]) : l % 2 ? KEY_ALT : KEY, l);
	let keys = "", marks = "", every = 8;
	for (let l = 0; l < nKeys; l++) keys += colourCode(keyRgb(l), mode) + "▆".repeat(kw);
	// The marks: every 8 layers, or 16 or 32 where 8 take under 8 columns; each over its layer's first column.
	while ((every * cols) / layers < 8) every *= 2;
	const { to } = span(layers, w), first = (l: number) => { let c = 0; while (to(c) <= l) c++; return c; };
	for (let l = 0; l < layers; l += every) {
		const at = fill ? first(l) : l * kw, m = `L${l}`;
		if (marks.length <= at && at + m.length <= w) marks = marks.padEnd(at) + m;   // whole, or not at all
	}
	const names = view === "reads" ? sourceNames(s.rk) : Array.from({ length: s.k }, (_, j) => `stream ${j + 1}`);
	const named = legend ? [names.map((n, j) => colourCode(rgb(j), mode) + "■ " + th.fg("dim", n)).join("  ")] : [];
	return [streamTitle(th, s, view, w, o), ...named, ...rgbBlocks(cells, cols, R, mode), keys + "\x1b[39m", th.fg("faint", marks)].map((l) => fit(l, w));
}

/** STREAM's playhead while it follows a streaming reply (Dock.playhead): the token it shows trails the newest by about
 * PLAY_LAG_MS and moves on evenly, however bunched the tokens come (several at once, then a wait, as speculative
 * decoding writes them). Its speed is the backlog over the lag, a token a lag at least so it reaches the newest; at a
 * steady rate that is the rate, the lag held. A token waiting over PLAY_JUMP_MS jumps it to the tokens that came the
 * lag ago; once the reply ends it drains at the rate (tokens over the last PLAY_RATE_MS), faster where that would take
 * over the lag. The dock moves it every PLAY_TICK_MS while there is a backlog. */
export const PLAY_LAG_MS = 250, PLAY_TICK_MS = 33, PLAY_JUMP_MS = 2000, PLAY_RATE_MS = 1000;
/** The playhead on reply `rid`: where it is (`pos`, a token and how far on to the next) at `at`, the newest token then,
 * when it began (`start`), the arrival times of the last tokens (`arr`, the newest last: those not shown yet and those
 * within PLAY_RATE_MS), the rate (tokens a ms) and the drain's speed. */
export interface Play { rid: string; pos: number; at: number; newest: number; start: number; arr: number[]; rate: number; drain?: number }
/** The playhead `p` moved on to `now`, the newest token `newest` (`done`: the reply ended). Another reply, or a first
 * look: at the newest. */
export function pace(p: Play | undefined, rid: string, newest: number, done: boolean, now: number): Play {
	if (!p || p.rid !== rid || newest < p.newest) return { rid, pos: newest, at: now, newest, start: now, arr: [now], rate: 0 };
	const arr = [...p.arr, ...Array<number>(newest - p.newest).fill(now)], first = newest - arr.length + 1;   // arr[j]: token first + j's
	const rate = done ? p.rate : arr.filter((t) => t > now - PLAY_RATE_MS).length / Math.max(100, Math.min(PLAY_RATE_MS, now - p.start));
	const dt = p.pos >= p.newest ? 0 : now - p.at;   // caught up till now: the wait does not count
	let pos = p.pos, drain = p.drain;
	const next = Math.floor(pos) + 1 - first;
	if (next < arr.length && now - arr[next] > PLAY_JUMP_MS) pos = first + arr.filter((t) => t <= now - PLAY_LAG_MS).length - 1;
	const b = newest - pos;
	if (done) drain ??= Math.max(rate, b / PLAY_LAG_MS);
	const v = b <= 0 ? 0 : done ? drain! : Math.max(b, 1) / PLAY_LAG_MS;
	pos = Math.min(newest, pos + v * dt);
	const keep = arr.findIndex((t, j) => first + j > pos || t > now - PLAY_RATE_MS);
	return { rid, pos, at: now, newest, start: p.start, arr: keep < 0 ? [] : arr.slice(keep), rate, drain };
}
/** The token the playhead shows. */
export const playShown = (p: Play) => Math.floor(p.pos);

export type { Grid };

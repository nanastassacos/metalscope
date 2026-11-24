/**
 * The full-size EXPERTS' two pages: one layer (enter on the full-size grids) and one expert (enter on the layer page's
 * list). The layer page: the layer's grid, large, lit as the board's (the glow, or the token's own); the token's
 * experts there beside the layer's experts from the server's tally (/scope/api/experts/specialists). The expert
 * page: what the tally says about it (tally.py, /scope/api/experts/expert: its tokens and the experts mixed with it, over every traced chat or this Pi
 * session) and where it is strong in the current reply (IN THIS REPLY, worked out here from the reply's routing). A
 * token counts for an expert when the expert is among its K_TOP strongest. The expert page's lines are fixed
 * (EXPERT_LINES), the layer page's follow only the dock's size: nothing moves between tokens, experts or sources.
 * Experts (at the end) is EXPERTS in the dock: its state, keys and requests, the lights and the layer window; the dock
 * (dock.ts) holds one. The rest here is pure.
 */

import { matchesKey } from "@earendil-works/pi-tui";
import { type DockTheme, LABELS, ansiInverse, bar, bold, fit, label, lpad, padTo, replyCounts, showTabs } from "./draw.ts";
import { type ExpertsView, type LayerLights, LIST_ROWS, bothSources, expertsLayers, forTokenBand, gridLegend, layerWindow, legendWidth, listRow, rightOf, saidOf,
	scrollWindow, secondAt, sharePct, side, stackLayers, tokenTag, tokensText, topsOf, windowCount } from "./board.ts";
import type { DockHost } from "./dock.ts";
import { GLOW_REPLAY, Glow, type Grids, type Palette, gridCells, gridShape, guessExperts, halfBlocks, lit, palette, reshapeGrid, tokenExperts } from "./lights.ts";
import { MAX_ROUTED, type Reply } from "./state.ts";

/** The number of MoE layers in a reply's routing (its newest routed token's), 0 without routing. */
export function routedLayers(toks: { experts?: Uint16Array; expert_k?: number }[]): number {
	for (let i = toks.length - 1; i >= 0; i--) {
		const t = toks[i];
		if (t.experts && t.expert_k) return t.experts.length / t.expert_k;
	}
	return 0;
}

/** The layer EXPERTS shows of `n`: the chosen one, else the middle. */
export const shownLayer = (n: number, chosen?: number) => Math.max(0, Math.min(n - 1, chosen ?? Math.floor(n / 2)));

/** A reply's routing from its saved trace (/scope/api/traces/ID/routing: base64 ids <u2 and shares u8, [n, L, k], for
 * reply tokens start..start+n-1) onto its tokens that have none, as the live stream would have sent it. An edit's
 * reply: its own tokens only (its trace's; the ones it joins came with theirs). */
export function fillRouting(r: Reply, res: { start: number; n: number; L: number; k: number; ids: string; w: string }) {
	const { start, n, L, k } = res, per = L * k;
	const ids = Buffer.from(res.ids, "base64"), w = Buffer.from(res.w, "base64");
	if (ids.length !== n * per * 2 || w.length !== n * per) return;
	let top = r.maxExpert ?? 0;
	for (const t of r.tokens) {
		const row = t.i - (r.from ?? 0) - start;
		if (t.i < (r.from ?? 0)) continue;
		if (t.experts || row < 0 || row >= n) continue;
		const ew = Uint8Array.from(w.subarray(row * per, (row + 1) * per));
		if (!ew.some((x) => x > 0)) continue;                     // nothing recorded for this token
		const ex = new Uint16Array(per);
		for (let j = 0; j < per; j++) ex[j] = ids[2 * (row * per + j)] | (ids[2 * (row * per + j) + 1] << 8);
		t.experts = ex;
		t.expert_w = ew;
		t.expert_k = k;
		for (const e of ex) if (e > top) top = e;
		r.routed = true;
	}
	r.maxExpert = top;
}

/** The waits before asking again for a reply's routing its trace does not have yet (the end event comes before the
 * server saves it); then it is given up. */
export const ROUTING_RETRY_MS = [500, 1000, 2000, 4000];
/** A GET of the dock's (the grid, a reply's routing) that has not answered by then fails, so a stalled server ends in
 * a reason, not "loading" for good. */
export const GET_TIMEOUT_MS = 10_000;
/** How long after its first answer a route counting saved replies (a tally, the lens steps' speeds) is asked again:
 * the server saves a reply after its end event, once idle. */
export const TALLY_LAG_MS = 4000;

export const K_TOP = 3;
export type Source = "all" | "session";

/** /scope/api/experts/specialists, what the grids' EXPERTS FOR and the layer page read: the layer's experts over the
 * tally's evidence floor, each with its top tokens by lift (LABELS.its). */
export interface Specialists {
	against: "tally" | "session" | null;
	experts: { e: number; top: string[] }[];
}
/** /scope/api/experts/expert, what the expert page reads: one expert in the tally (rank null: under the evidence floor). */
export interface ExpertInfo {
	tokens: number;
	against: "tally" | "session" | null;
	ranked: number;
	e: number;
	n: number;
	share: number;
	rank: number | null;
	top: [string, number, number][];
	overlaps: { e: number; lift: number; co: number; top: string[] }[];
}

type Tok = { text: string; experts?: Uint16Array; expert_w?: Uint8Array; expert_k?: number };

/** The reply tokens where expert `e` is among the K_TOP strongest at `layer`: their index and the expert's share. */
export function placesOf(toks: Tok[], layer: number, e: number): { p: number; w: number }[] {
	const out: { p: number; w: number }[] = [];
	toks.forEach((t, p) => {
		const x = tokenExperts(t, layer).slice(0, K_TOP).find(([y]) => y === e);
		if (x) out.push({ p, w: x[1] });
	});
	return out;
}

/** IN THIS REPLY's places: the heaviest, at least 9 tokens apart, at most two of a text, CONTEXT_ROWS of them, in order. */
export function contextPicks(toks: Tok[], places: { p: number; w: number }[]): number[] {
	const picked: number[] = [], seen = new Map<string, number>();
	for (const x of [...places].sort((a, b) => b.w - a.w || a.p - b.p)) {
		const t = toks[x.p].text.trim().toLowerCase();
		if (picked.some((q) => Math.abs(q - x.p) < 9) || (seen.get(t) ?? 0) >= 2) continue;
		picked.push(x.p);
		seen.set(t, (seen.get(t) ?? 0) + 1);
		if (picked.length === CONTEXT_ROWS) break;
	}
	return picked.sort((a, b) => a - b);
}

/** IN THIS REPLY's strip: the reply in `cols` columns, each the share of its tokens in `places` against the busiest
 * column's, 0-8 (-1: a column past the reply). */
export function whereStrip(n: number, places: { p: number }[], cols: number): number[] {
	const size = new Array(cols).fill(0), c = new Array(cols).fill(0), col = (p: number) => Math.min(cols - 1, Math.floor((p * cols) / Math.max(1, n)));
	for (let p = 0; p < n; p++) size[col(p)]++;
	for (const x of places) c[col(x.p)]++;
	const fr = c.map((x, i) => (size[i] ? x / size[i] : 0)), top = Math.max(0, ...fr);
	return fr.map((x, i) => (!size[i] ? -1 : x > 0 ? Math.max(1, Math.min(8, Math.ceil((8 * x) / top))) : 0));
}

/** The layer page's EXPERTS FOR: half the page `w` wide, the layer's list the rest. */
const half = (w: number) => Math.floor((w - 3) / 2);
/** Whether the layer page `w` wide shows both sources' tokens (bothSources for its EXPERTS FOR, the narrower list). */
export const layerBoth = (w: number) => bothSources(half(w));

/** What the layer page draws: the layer, its grid (`grid`: lit for token `sel`, `text`, where `held` on it, else the
 * glow of the newest tokens), the token's experts there
 * (`mine`, the cursor on `row`, dock.ts xpLists: within the list, -1 none), and the layer's experts by number (`n`
 * of them: none without the tally's answer), from #`first`, `at` the one marked (-1 none). `right`: the cursor is in
 * the layer's list (on `at`); else in the token's (`at` the expert it picks). Both lists show each expert's tokens
 * from the tally's answers (`sps`, each undefined while asked for, null when there is none): by `source`, or all
 * chats and this session side by side (layerBoth). */
export interface LayerPage {
	layer: number;
	sel: number;
	text: string;
	grid: LayerLights;
	held: boolean;
	mine: [number, number][];
	sps: (Specialists | null | undefined)[];
	source: Source;
	row: number;
	n: number;
	first: number;
	at: number;
	right: boolean;
}

/** The layer page, `w` by `rows` (the board's: 97 columns at the least): the breadcrumb and the source, the token in
 * focus on its right; the layer's grid, each light s by s cells, as large as fits beside its legend (the wider either
 * way, legendWidth) and over the lists; then EXPERTS FOR (a cursor list) beside the layer's experts by number, LIST_ROWS
 * of them from `first`, the marked one's place in the heading, the token's in accent, those with no tokens dim. One
 * source: named in the breadcrumb; both: each over its column, this session's over the layer's list's second too. Its
 * lines follow only `w`, `rows` and the grid's shape. */
export function layerPage(th: any, v: LayerPage, pal: Palette, w: number, rows: number): string[] {
	const both = v.sps.length > 1;
	const crumb = th.fg("dim", LABELS.crumb) + th.fg("faint", LABELS.sep) + th.fg("accent", bold(th, LABELS.layerName(v.layer + 1))) +
		(both ? "" : th.fg("dim", ` · ${LABELS.source[v.source]}`)) + (v.sps.some((sp) => sp?.against === "session") ? th.fg("faint", ` · ${LABELS.noTally}`) : "");
	const out = [rightOf(crumb, tokenTag(th, v.sel, v.text), w), ""];
	const legend = gridLegend(v.held, v.sel), lg = 3 + legendWidth(v.sel), g = v.grid;
	const room = rows - 4 - LIST_ROWS;   // the breadcrumb, two blanks, the lists' heading and rows
	const s = Math.max(1, Math.min(Math.floor((w - 2 - lg) / Math.max(1, g.cols)), Math.floor((2 * room) / Math.max(1, g.rows))));
	const cols = g.cols * s, cells = Array.from({ length: cols * g.rows * s }, (_, i) => g.cells[Math.floor(Math.floor(i / cols) / s) * g.cols + Math.floor((i % cols) / s)] ?? null);
	halfBlocks(cells, cols, g.rows * s, pal).forEach((l, i) => out.push("  " + l + (legend[i] ? "   " + th.fg("dim", legend[i]) : "")));
	const lw = half(w), rw = w - 3 - lw, tops = v.sps.map(topsOf), mine = new Set(v.mine.map(([e]) => e));
	const left = forTokenBand(th, v.text, v.mine, tops, lw, v.row, "", v.right);
	const shown = Array.from({ length: Math.max(0, Math.min(LIST_ROWS, v.n - v.first)) }, (_, i) => v.first + i);
	const title = th.fg("accent", bold(th, LABELS.layer(v.layer + 1))) + (v.at >= 0 ? th.fg("dim", `  ${LABELS.at(v.at + 1, v.n)}`) : "");
	const second = secondAt(rw, 0, 0);
	const right = [both ? fit(title, second - 2) + "  " + th.fg("dim", LABELS.source.session) : title,
		...(v.n ? shown.map((e) => listRow(th, e, 0, 0, "", tops.map((t) => saidOf(t, e)), rw, e === v.at && (v.right || "mark"),
			mine.has(e) ? "hit" : tops.some((t) => t?.get(e)?.length) ? "" : "dim"))
			: v.sps.every((sp) => sp === undefined) ? [] : ["  " + th.fg("faint", "-")])];
	out.push("", ...side(th, left, right, lw, rw));
	return out.map((l) => fit(l, w));
}

/** What the expert page draws: the expert (`e` at `layer`), the source, the tally's answer (undefined while asked for,
 * null when there is none), the reply's tokens and `think` (the token ending the thinking, -1), the cursor in MIXED
 * WITH (`pick`) and the expert `b` goes back to. */
export interface ExpertPage {
	layer: number;
	e: number;
	source: Source;
	info: ExpertInfo | null | undefined;
	toks: Tok[];
	think: number;
	pick: number;
	back?: number;
}

const OVERLAP_ROWS = 8, TOKEN_ROWS = 10, CONTEXT_ROWS = 6, LEFT_W = 47, AT = 44;
/** The expert page's lines, whatever it shows. */
export const EXPERT_LINES = 25;

/** MIXED WITH's experts (OVERLAP_ROWS at most): where enter goes. */
export const mixedWith = (info: ExpertInfo | null | undefined) => (info?.overlaps ?? []).slice(0, OVERLAP_ROWS).map((x) => x.e);

const many = (x: number) => (x >= 9.95 ? `${Math.round(x)}` : x.toFixed(1));
/** A count in a few characters: 820, 8.2k, 174k. */
export const kNum = (n: number) => (n < 1000 ? `${n}` : n < 9950 ? `${(n / 1000).toFixed(1)}k` : `${Math.round(n / 1000)}k`);
const nth = (n: number) => `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] ?? "th"}`;

/** The expert page, `w` wide (the board's: 97 columns at the least), EXPERT_LINES lines: the breadcrumb; a sentence
 * (what it takes, its rank, the source); one legend line for the ×; ITS TOKENS beside MIXED WITH (a cursor list);
 * IN THIS REPLY, the strip of where along the reply under it. A part with nothing to show: a dim -. */
export function expertPage(th: any, v: ExpertPage, w: number): string[] {
	const dash = th.fg("faint", "-"), head = (name: string, note = "") => th.fg("accent", bold(th, name)) + th.fg("dim", note);
	const info = v.info?.tokens ? v.info : null, src = v.source, out: string[] = [];
	// The breadcrumb, the expert bright; where b goes back to on the right.
	const crumb = th.fg("dim", LABELS.crumb) + th.fg("faint", LABELS.sep) + th.fg("dim", LABELS.layerName(v.layer + 1)) + th.fg("faint", LABELS.sep) + th.fg("accent", bold(th, `#${v.e}`));
	out.push(rightOf(crumb, v.back !== undefined ? th.fg("faint", LABELS.back(v.back)) : "", w));
	const rank = info ? (info.rank ? `${nth(info.rank)} ${LABELS.rank[src]} of ${info.ranked}` : LABELS.unranked) : "";
	out.push(th.fg("dim", info ? `${LABELS.takes(info.n.toLocaleString("en-US"), kNum(info.tokens), sharePct(info.share))} · ${rank} · ${LABELS.source[src]}` : `- · ${LABELS.source[src]}`) +
		(info?.against === "session" ? th.fg("faint", ` · ${LABELS.noTally}`) : ""));
	const top1 = info?.top[0]?.[1];
	out.push(top1 ? th.fg("faint", LABELS.times(many(top1))) : "", "");
	// Its tokens | mixed with.
	const lw = LEFT_W, rw = w - lw - 3;
	const left: string[] = [head(LABELS.tokens)], right: string[] = [head(LABELS.mixed, `  ${LABELS.mixedTag}`)];
	const toks = info?.top.slice(0, TOKEN_ROWS) ?? [], tw = lw - 16 - 1 - 6, tt = Math.max(1e-9, ...toks.map((x) => x[1]));
	for (const [t, lift] of toks) left.push(padTo(label(t), 16) + th.fg("accent", padTo(bar(lift / tt, tw), tw)) + " " + th.fg("dim", lpad(`×${many(lift)}`, 6)));
	if (!toks.length) left.push(dash);
	const over = (v.info?.overlaps ?? []).slice(0, OVERLAP_ROWS), ot = Math.max(0, ...over.map((x) => x.lift));
	over.forEach((x, i) => right.push(listRow(th, x.e, ot > 0 ? x.lift / ot : 0, 8, lpad(`×${many(x.lift)}`, 6), tokensText(x.top), rw, i === v.pick)));
	if (!over.length) right.push("  " + dash);
	for (let i = 0; i < 1 + TOKEN_ROWS; i++) out.push(fit(left[i] ?? "", lw) + th.fg("rule", " │ ") + fit(right[i] ?? "", rw));
	out.push("");
	// IN THIS REPLY: the token at a fixed column, four tokens either side; under them the reply in w columns, the
	// thinking dim.
	const places = placesOf(v.toks, v.layer, v.e), picks = contextPicks(v.toks, places);
	out.push(head(LABELS.reply));
	for (let i = 0; i < CONTEXT_ROWS; i++) {
		const p = picks[i];
		if (p === undefined) { out.push(i || picks.length ? "" : dash); continue; }
		const before = showTabs(v.toks.slice(Math.max(0, p - 4), p).map((t) => t.text).join("")), after = showTabs(v.toks.slice(p + 1, p + 5).map((t) => t.text).join(""));
		const room = AT - 7, bf = [...before].length > room ? "…" + [...before].slice(-(room - 1)).join("") : before;
		const tok = showTabs(v.toks[p].text), lead = tok.length - tok.trimStart().length, word = tok.slice(lead) || "␣";
		out.push(th.fg("dim", padTo(`#${p}`, 6)) + " ".repeat(Math.max(1, AT - 6 - [...bf].length)) + bf + tok.slice(0, lead) +
			th.fg("accent", bold(th, ansiInverse(word))) + after);
	}
	const n = v.toks.length, tb = v.think >= 0 ? Math.floor((v.think * w) / Math.max(1, n)) : -1;
	const strip = places.length ? whereStrip(n, places, w) : [];
	out.push(strip.length ? strip.map((d, i) => (d < 0 ? " " : d ? th.fg(i <= tb ? "dim" : "accent", "▁▂▃▄▅▆▇█"[d - 1]) : th.fg("rule", "·"))).join("") : "");
	out.push(tb >= 0 ? th.fg("faint", fit("thinking", Math.max(9, tb + 1)) + "answer") : "");
	return out.map((l) => fit(l, w));
}

/** What EXPERTS may ask of the Dock: the dock gone, the token in focus. */
export interface ExpertsDock {
	readonly disposed: boolean;
	readonly sel: number;
	cur(n: number): number;
}

/** EXPERTS in the dock: its lights (the glow, the token's own), the layer window, the full-size EXPERTS' pages and their
 * keys, the server's tallies and a reply's routing from its saved trace. The Dock (dock.ts) holds one; `grids`, `get`
 * and `timers` are the Dock's own (its layer grids, its GET, the timers it clears as it goes). */
export class Experts {
	constructor(private host: DockHost, private theme: DockTheme, private tui: { readonly terminal?: any; requestRender(): void }, private grids: Grids,
		private get: (url: string) => Promise<any>, private timers: Set<ReturnType<typeof setTimeout>>, private dock: ExpertsDock) {}
	// EXPERTS' lights, fed token by token: one Glow a layer shown (a tall dock draws every layer), the least recently
	// used dropped past the reply's layer count (another model's left over).
	private glows = new Map<number, { rid: string; E: number; fed: number; g: Glow }>();
	// The mini dock's glow at an older token stepped to, a layer each (miniGlow).
	private pastGlows = new Map<number, { rid: string; E: number; sel: number; g: Glow }>();
	// The full-size EXPERTS' screen (`at`: its grids, the layer page or the expert page), the layer page's cursors (`row`
	// in the token's experts, `pe` the expert in the layer's list, -1 its first; `right` the list tab put the keys in)
	// and its list's first row shown (`first`), the expert page's expert (`e`), its cursor in MIXED WITH (`pick`) and
	// where b goes back to (at most 30). The tally's answers by URL and newest saved reply, and each URL's last (xpGet).
	xp: { at: "grid" | "layer" | "expert"; row: number; right: boolean; pe: number; first: number; e: number; pick: number;
		hist: { layer: number; e: number; pick: number }[] } = { at: "grid", row: 0, right: false, pe: -1, first: 0, e: 0, pick: 0, hist: [] };
	xpData = new Map<string, any>();
	xpLast = new Map<string, any>();
	private xpLate = new Set<string>();   // answers to ask for once more (xpGet)
	xpBoth = false;               // the lists last drawn show both sources' tokens (bothSources, layerBoth): no a
	// EXPERTS' layers as last worked out (expertsData), for this frame only (renderBoard clears it).
	stack?: { r: Reply; sel: number; held: boolean; n: number; w: number; data: { v: ExpertsView; layers: LayerLights[] } };
	/** One layer's lines in EXPERTS on the board (expertsStack's), `w` wide: its heading and its grid's half-block lines
	 * (lights.ts reshapeGrid). */
	expertsBlock(r: Reply, v: ExpertsView, w: number): number {
		return 1 + Math.ceil(reshapeGrid(v.E, gridShape(v.E, this.grids.at(this.host.live.base, r.model, v.layer)), w).rows / 2);
	}
	/** EXPERTS on the board, `w` wide: `n` layers (board.ts stackLayers) around the focus layer, each lit as levelsAt,
	 * its grid the column's width. */
	expertsData(r: Reply, sel: number, n: number, w: number): { v: ExpertsView; layers: LayerLights[] } {
		const c = this.stack, held = this.held();
		if (c?.r === r && c.sel === sel && c.held === held && c.n === n && c.w === w) return c.data;
		const v = this.expertsView(r);
		if (v.reason) return { v, layers: [] };
		const layers = stackLayers(v.L, v.layer, n).map((l): LayerLights => {
			const shape = reshapeGrid(v.E, gridShape(v.E, this.grids.at(this.host.live.base, r.model, l)), w);
			return { layer: l, cells: gridCells(this.levelsAt(r, l, v.E, sel), shape), cols: shape.cols, rows: shape.rows };
		});
		this.stack = { r, sel, held, n, w, data: { v, layers } };
		return { v, layers };
	}
	/** EXPERTS' layer window (board.ts expertsLayers), `w` by at most `rows`: full size, or the one-panel board's; as many
	 * layers as the lines hold (windowCount), EXPERTS FOR under them. */
	expertsWindow(r: Reply, sel: number, w: number, rows: number): string[] {
		const th = this.theme, toks = r.tokens, held = this.held();
		const v = this.expertsView(r);
		if (v.reason) return expertsLayers(th, v, [], palette(th), w, [], sel, toks[sel].text, held, rows);
		const focusShape = gridShape(v.E, this.grids.at(this.host.live.base, r.model, v.layer));
		const [a, b] = layerWindow(v.L, v.layer, windowCount(v.L, w, rows, focusShape.cols, focusShape.rows)), layers: LayerLights[] = [];
		for (let l = a; l < b; l++) layers.push(this.layerLights(r, v, l, sel));
		return expertsLayers(th, v, layers, palette(th), w, tokenExperts(toks[sel], v.layer), sel, toks[sel].text, held, rows,
			this.xpSaid(r, v.layer, bothSources(w)).map(topsOf), this.host.expertSource ?? "all", this.noMap(r, v.layer));
	}
	/** Whether the server answered reply `r`'s layer `l` grid without an expert map (pos null: number order). */
	noMap = (r: Reply, l: number) => this.grids.at(this.host.live.base, r.model, l)?.pos === null;
	/** One layer's lights in its own grid's shape, lit as levelsAt. */
	private layerLights(r: Reply, v: ExpertsView, l: number, sel: number): LayerLights {
		const shape = gridShape(v.E, this.grids.at(this.host.live.base, r.model, l));
		return { layer: l, cells: gridCells(this.levelsAt(r, l, v.E, sel), shape), cols: shape.cols, rows: shape.rows };
	}
	/** A tally route's URL for reply `r`'s model at `layer` by `source` (the session: this Pi's client). */
	private xpUrl(r: Reply, kind: "specialists" | "expert", layer: number, e?: number, source = this.host.expertSource ?? "all"): string {
		const live = this.host.live, q = new URLSearchParams({ layer: String(layer), ...(r.model ? { model: r.model } : {}),
			...(source === "session" ? { client: live.clientId } : {}), ...(e === undefined ? {} : { e: String(e) }) });
		return `${live.base}/scope/api/experts/${kind}?${q}`;
	}
	/** A tally route's answer: undefined while first asked for (drawn when it comes), null when it failed. Asked again
	 * once a newer reply is saved (the newest saved is the key: not as one streams), and once more TALLY_LAG_MS after
	 * the first answer (the server counts a saved reply when idle), the URL's last answer drawn meanwhile. */
	private xpGet<T>(url: string): T | null | undefined {
		const live = this.host.live, key = `${url}|${live.reply?.traceId ?? live.prev?.traceId ?? ""}`;
		const have = this.xpData.get(key), last = this.xpLast.get(url);
		if (have !== undefined && !this.xpLate.delete(key)) return have === "pending" ? last : have;
		if (have === undefined) {
			if (this.xpData.size > 200) this.xpData.clear(), this.xpLast.clear();
			this.xpData.set(key, "pending");
		}
		const got = (x: any) => { this.xpData.set(key, x); this.xpLast.set(url, x); };
		this.get(url).then((x) => got(x ?? null), () => got(null)).finally(() => {
			this.tui.requestRender();
			if (have !== undefined || this.dock.disposed) return;
			const t = setTimeout(() => { this.timers.delete(t); this.xpLate.add(key); this.tui.requestRender(); }, TALLY_LAG_MS);
			t.unref?.();
			this.timers.add(t);
		});
		return have === undefined || have === "pending" ? last : have;
	}
	/** The tally's lists for reply `r`'s model at `layer` (one request a layer and source: xpGet): by the source, or
	 * (`both`) all chats and this session, each drawn as it comes. */
	private xpSaid(r: Reply, layer: number, both: boolean): (Specialists | null | undefined)[] {
		const sp = (source?: Source) => this.xpGet<Specialists>(this.xpUrl(r, "specialists", layer, undefined, source));
		return both ? [sp("all"), sp("session")] : [sp()];
	}
	/** The layer page's two lists for token `sel`: its experts at the focus layer (`mine`, LIST_ROWS at most) with the
	 * cursor's row (-1: an empty list), the layer's `n` experts by number (none until the tally answers) with the
	 * cursor's expert (`pos`: `xp.pe`, else #0), whether the keys are in that list (`right`: tab, never in an empty
	 * one), and the expert enter opens (`e`): one clamp for the page, its key line and enter. */
	xpLists(r: Reply, sel: number) {
		const v = this.expertsView(r), sps = this.xpSaid(r, v.layer, this.xpBoth), n = sps.some((sp) => sp?.experts.length) ? v.E : 0;
		const mine = tokenExperts(r.tokens[sel] ?? {}, v.layer).slice(0, LIST_ROWS);
		const row = Math.min(this.xp.row, mine.length - 1), right = this.xp.right && n > 0;
		const pos = right ? Math.max(0, Math.min(n - 1, this.xp.pe)) : -1;
		return { v, sps, n, mine, row, pos, right, e: right ? pos : mine[row]?.[0] };
	}
	/** The layer page's data for the focus layer and token `sel` (expert.ts): the layer's list scrolled to keep its
	 * cursor in view (tab), else the expert the token's list picks, two rows around it where the list allows. */
	xpLayer(r: Reply, sel: number): LayerPage {
		const xp = this.xp, { v, sps, n, mine, row, pos, right, e } = this.xpLists(r, sel);
		const at = right ? pos : e !== undefined && e < n ? e : -1;
		if (at >= 0) xp.first = scrollWindow(n, LIST_ROWS, at, xp.first, 2);
		return { layer: v.layer, sel, text: r.tokens[sel].text, grid: this.layerLights(r, v, v.layer, sel), held: this.held(), mine,
			sps, source: this.host.expertSource ?? "all", row, n, first: Math.min(xp.first, Math.max(0, n - LIST_ROWS)), at, right };
	}
	/** The expert page's data (expert.ts). */
	xpPage(r: Reply): ExpertPage {
		const v = this.expertsView(r), e = Math.max(0, Math.min(v.E - 1, this.xp.e)), toks = r.tokens;
		return { layer: v.layer, e, source: this.host.expertSource ?? "all", info: this.xpGet<ExpertInfo>(this.xpUrl(r, "expert", v.layer, e)),
			toks, think: replyCounts(toks).think, pick: this.xp.pick, back: this.xp.hist.at(-1)?.e };
	}
	/** The full-size EXPERTS' keys (its window: full size, or the one-panel board's). The grids: ↑/↓ the focus layer, enter
	 * its layer page; tab is the board's (the next panel). The layer page: tab the other list
	 * (EXPERTS FOR or the layer's; not into an empty one), ↑/↓ pick in it, enter its expert's page, [ ] the layer, esc
	 * the grids. The expert page: ↑/↓ pick in MIXED WITH, enter goes there, [ ] the same number a layer over, b back,
	 * esc the layer page (the layer's list, where it has the keys, on the expert last shown); tab is the board's. a the
	 * source on the expert page, and on the grids and the layer page where they show one (xpBoth). ←/→ are left to step
	 * the token everywhere. Whether it took the key. */
	expertKey(data: string, r: Reply): boolean {
		const h = this.host, xp = this.xp, at = xp.at, v = this.expertsView(r);   // at: the screen the key came on
		if (v.reason) return false;
		const up = matchesKey(data, "up"), down = matchesKey(data, "down"), enter = matchesKey(data, "enter"), esc = matchesKey(data, "escape");
		const shift = data === "]" ? 1 : data === "[" ? -1 : 0;
		const cycle = (i: number, n: number) => (n ? (Math.min(i, n - 1) + (down ? 1 : -1) + n) % n : 0);
		if (data === "a" && (at === "expert" || !this.xpBoth)) {
			h.expertSource = h.expertSource === "session" ? "all" : "session";
			xp.pick = 0;   // the layer's list keeps its expert: only the tokens shown change
			return true;
		}
		if (at === "grid") {
			if (up || down) h.expertLayer = shownLayer(v.L, v.layer + (down ? 1 : -1));
			else if (enter) Object.assign(xp, { at: "layer", row: 0, right: false, pe: -1, first: 0 });
			else return false;
			return true;
		}
		if (data === "s" || data === "S") return true;   // the EXPERTS pages: not the board's lens swap, unseen under them
		if (at === "layer") {
			const { n, mine, row, pos, right, e } = this.xpLists(r, this.dock.cur(r.tokens.length));
			if (matchesKey(data, "tab") || matchesKey(data, "shift+tab")) {
				// Into the layer's list (not an empty one): on the expert the token's list picks.
				xp.right = !right && n > 0;
				if (xp.right && mine[row]) xp.pe = mine[row][0];
			} else if (up || down) {
				if (right) xp.pe = cycle(pos, n);
				else xp.row = cycle(row, mine.length);
			} else if (enter) { if (e !== undefined) Object.assign(xp, { at: "expert", e, pick: 0, hist: [] }); }
			else if (shift) h.expertLayer = shownLayer(v.L, v.layer + shift);   // the layer's list stays on the same number
			else if (esc) xp.at = "grid";
			else return false;
		} else {
			const go = (layer: number, to: number) => {
				if (layer < 0 || layer >= v.L || (layer === v.layer && to === xp.e)) return;
				xp.hist.push({ layer: v.layer, e: xp.e, pick: xp.pick });
				if (xp.hist.length > 30) xp.hist.shift();
				Object.assign(xp, { e: to, pick: 0 });
				h.expertLayer = layer;
			};
			const links = mixedWith(this.xpPage(r).info);
			if (up || down) xp.pick = cycle(xp.pick, links.length);
			else if (enter) { const x = links[xp.pick]; if (x !== undefined) go(v.layer, x); }
			else if (shift) go(v.layer + shift, xp.e);
			else if (data === "b") { const x = xp.hist.pop(); if (x) Object.assign(xp, { e: x.e, pick: x.pick }), (h.expertLayer = x.layer); }
			else if (esc) Object.assign(xp, { at: "layer", pe: xp.right ? xp.e : xp.pe });   // the layer's list on the expert last shown
			else return false;
		}
		return true;
	}
	/** The mini dock's lights at `layer`: the glow (bright: just used, warm: busy lately) of the newest tokens while
	 * following; stepped to an older token `sel`, the glow as it stood there (fed from GLOW_REPLAY tokens before it, kept
	 * per layer and moved on a token at a time going forward). */
	miniGlow(r: Reply, layer: number, E: number, sel: number): ArrayLike<number> {
		const toks = r.tokens;
		if (this.dock.sel < 0 || sel >= toks.length - 1) return this.glowAt(r, layer, E).levels();
		let st = this.pastGlows.get(layer);
		if (!st || st.rid !== r.rid || st.E !== E || st.sel > sel) st = { rid: r.rid, E, sel: Math.max(0, sel + 1 - GLOW_REPLAY) - 1, g: new Glow(E) };
		for (let i = st.sel + 1; i <= sel; i++) if (toks[i].experts) st.g.feed(tokenExperts(toks[i], layer));
		st.sel = sel;
		this.pastGlows.set(layer, st);
		return st.g.levels();
	}
	/** EXPERTS' frame for reply `r`: the layer in focus ([ ]), the layers, experts and how many mix; or why there is
	 * nothing to draw. */
	expertsView(r: Reply): ExpertsView {
		this.loadRouting(r);
		const L = routedLayers(r.tokens);
		if (!L) return { layer: 0, L: 0, E: 0, k: 0, reason: this.routingReason(r) };
		const layer = shownLayer(L, this.host.expertLayer);
		const grid = this.grids.at(this.host.live.base, r.model, layer);
		const E = grid?.num_experts ?? guessExperts(r.maxExpert ?? 0);
		let k = 0;
		for (let i = r.tokens.length - 1; i >= 0 && !k; i--) k = r.tokens[i].expert_k ?? 0;
		return { layer, L, E, k };
	}
	/** EXPERTS' grids show token `sel`'s own experts (held) once a token is stepped to; following, streaming or done,
	 * the glow as it stood on the last token. */
	private held = () => this.dock.sel >= 0;
	/** Each expert's brightness at `layer`: held (above), token `sel`'s own experts lit by their share; else the
	 * layer's glow over the newest tokens. */
	private levelsAt(r: Reply, layer: number, E: number, sel: number): ArrayLike<number> {
		if (!this.held()) return this.glowAt(r, layer, E).levels();
		const lv = new Float32Array(E);
		for (const [e, s] of tokenExperts(r.tokens[sel], layer)) if (e < E) lv[e] = lit(s);
		return lv;
	}
	/** Why a reply has no experts to show (one line, never an empty panel). */
	routingReason(r: Reply): string {
		const from = r.routingFrom;
		if (from === "loading" || from === "again") return "loading this reply's experts from its saved trace…";
		if (from && from !== "loaded") return from;
		if (!r.done) return this.host.live.experts ? "waiting for the stream's routing…" : "this reply streams without routing: its experts load from its trace when it ends";
		if (!r.traceId) return "this reply was not saved: no experts to load";
		return "no routing for this reply";
	}
	/** A finished reply whose newest tokens came without routing (traced before the dock asked the stream for it, or
	 * from a stream that did not): its routing from its saved trace (/scope/api/traces/ID/routing), once per reply,
	 * filling the tokens that have none. A dense model's answer marks the model dense; a
	 * model known dense is not asked. The end event names the trace before the server has saved it (live.py), so a
	 * reply not found yet is asked again after ROUTING_RETRY_MS (Reply.routingTries), then given up with the reason. */
	loadRouting(r: Reply) {
		if (!r.done || !r.traceId || (r.routingFrom && r.routingFrom !== "again")) return;
		const live = this.host.live, base = live.base;
		if (live.dense === (r.model ?? "")) return;
		const toks = r.tokens;
		if (!toks.slice(Math.max(0, toks.length - MAX_ROUTED), -1).some((t) => !t.experts)) return;
		r.routingFrom = "loading";
		const done = (why: string) => {
			r.routingFrom = why;
			if (!this.dock.disposed) this.tui.requestRender();
		};
		// Not saved yet: again after a wait (a dock closed meanwhile leaves it to the next one to ask).
		const later = () => {
			const n = (r.routingTries = (r.routingTries ?? 0) + 1);
			if (n > ROUTING_RETRY_MS.length) return done("no routing saved for this reply");
			if (this.dock.disposed) return void (r.routingFrom = "again");
			const timer = setTimeout(() => {
				this.timers.delete(timer);
				if (r.routingFrom !== "loading") return;
				r.routingFrom = "again";
				this.tui.requestRender();
			}, ROUTING_RETRY_MS[n - 1]);
			this.timers.add(timer);
		};
		let asked: Promise<any>;
		try {
			asked = this.get(`${base}/scope/api/traces/${encodeURIComponent(r.traceId)}/routing?last=${MAX_ROUTED}`);
		} catch (e: any) {
			return done(`this reply's experts did not load: ${e?.message ?? e}`);
		}
		asked.then((res: any) => {
			if (!res) return later();   // 404 (or no answer body): not saved and indexed yet
			if (typeof res.n !== "number" || typeof res.ids !== "string" || typeof res.w !== "string") return done("no routing saved for this reply");
			if (!res.L) {   // no router records at all (a reply of no tokens of its own still has its layers: not this)
				if (live.base === base) live.dense = r.model ?? "";
				return done("a dense model: no experts to show");
			}
			if (!res.k || !res.n) return done("no routing saved for this reply");
			fillRouting(r, res);
			this.glows.clear();   // the lights fed afresh, with the routing
			this.pastGlows.clear();
			done("loaded");
		}, (e: any) => done(`this reply's experts did not load: ${e?.name === "TimeoutError" ? `no answer in ${GET_TIMEOUT_MS / 1000} s` : e?.message ?? e}`));
	}
	/** EXPERTS' lights for reply `r` at `layer`, fed up to its newest token (from GLOW_REPLAY tokens back when it
	 * starts afresh: another reply, layer or size). Tokens without routing are skipped. */
	private glowAt(r: Reply, layer: number, E: number): Glow {
		const toks = r.tokens;
		let st = this.glows.get(layer);
		this.glows.delete(layer);   // re-set below: the map's order is the use order
		if (!st || st.rid !== r.rid || st.E !== E || st.fed >= toks.length)
			st = { rid: r.rid, E, fed: Math.max(0, toks.length - GLOW_REPLAY) - 1, g: new Glow(E) };
		this.glows.set(layer, st);
		const keep = routedLayers(toks);
		for (const old of this.glows.keys()) if (this.glows.size > keep) this.glows.delete(old);
		for (let i = st.fed + 1; i < toks.length; i++) if (toks[i].experts) st.g.feed(tokenExperts(toks[i], layer));
		st.fed = toks.length - 1;
		return st.g;
	}
}

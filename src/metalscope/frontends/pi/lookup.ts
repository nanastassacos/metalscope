/**
 * Engram inspection: the engram's help for one finished reply (g in the dock, where the server lists `lookup`): POST
 * /scope/api/lookup reads the reply again with and without the model's engram (its n-gram lookup; lookup.py). The
 * reply text and MAP then show the tokens it helped in one blue, brighter by how much; the cursor's engram (the tokens
 * before it that the engram looked up) on a block of the blue and the cursor's token, the one that predicted,
 * underlined (in MAP white); the title line says what was looked up and both probabilities, and TOKENS the token's top choices
 * without and with the engram. As the mode opens, a sweep reads the reply once without and once with (Sweep); while
 * the server reads it, the title line shows how far (lookupBar), and g again calls it off.
 */

import { type MapPaint, keepRow, ruleText, title } from "./board.ts";
import type { DockHost } from "./dock.ts";
import { COLOURS, type DockTheme, LABELS, type Rgb, ansiBold, bar, colourCode, cut, label, lpad, padTo, pct, qt, textWidth } from "./draw.ts";
import { STOPS, colourMode, mix } from "./lights.ts";
import { type Reply, shownReply } from "./state.ts";
import { fog } from "./telescope.ts";

/** POST /scope/api/lookup's answer (lookup.py measure): for each fed reply token of trace `trace` (all but the last
 * sampled one), from reply token 0, its id and its log-probability with the engram and without; and each reading's top
 * choices at that row ([token][10] ids and log-probabilities, likeliest first) with the ids' texts (`vocab`); the
 * engram's longest n-gram (`ngram`: Flash Next 3, DeepSeek V4.1 4). */
export interface LookupAnswer {
	trace: string;
	ngram: number;
	ids: number[];
	lp_on: number[];
	lp_off: number[];
	top_ids_on: number[][];
	top_lp_on: number[][];
	top_ids_off: number[][];
	top_lp_off: number[][];
	vocab: Record<string, string>;
}
/** A trace's engram inspection for the Pi session (DockHost.looked): refused (`error`, the server's detail) or
 * answered; while it is being measured, the ask's id (`ask`: its progress on the live feed, its stop). */
export interface Looked {
	error?: string;
	answer?: LookupAnswer;
	ask?: string;
}

/** How long the dock waits for an answer in all: two reads of the whole conversation, 46 s at 24,000 tokens on ds4,
 * several times that on oMLX. A request given up on is still measured and kept by the server: g again gets it. */
const LOOKUP_TIMEOUT_MS = 15 * 60_000;
/** Pi's fetch (undici) gives up on a request idle for Pi's idle timeout (30 s to 5 min, or off; 5 min by default),
 * without its headers or more of its body: these codes. */
const OVER = new Set(["UND_ERR_HEADERS_TIMEOUT", "UND_ERR_BODY_TIMEOUT"]);
/** `send(signal, again)`'s answer, sent again (`again` true: the server waits for the run under way, never starts one)
 * while it fails only on fetch's own limit (OVER); any other failure, or LOOKUP_TIMEOUT_MS (`ms`) in all, thrown. */
export async function askAgain<T>(send: (signal: AbortSignal, again: boolean) => Promise<T>, ms = LOOKUP_TIMEOUT_MS): Promise<T> {
	const signal = AbortSignal.timeout(ms);
	for (let again = false; ; again = true) {
		try {
			return await send(signal, again);
		} catch (e: any) {
			if (signal.aborted || !OVER.has(e?.cause?.code)) throw e;
		}
	}
}

/** The readings' progress, as the live feed sends it (lookup.py Run): the reading (0 with the engram, 1 without), the
 * share of it known done and the share at the next checkpoint (an oMLX chunk; a ds4 prefill chunk), its
 * seconds so far and the seconds it is expected to take, for ask `ask`. */
export interface LookupProgress {
	type: "lookup";
	ask: string;
	reading: number;
	done: number;
	next: number;
	elapsed: number;
	expect: number;
}
/** The estimate stops this far toward the next checkpoint; the bar, short of full; seconds left shown from LEFT_FROM. */
export const AHEAD = 0.95, MOST = 0.99, LEFT_FROM = 5;
/** Both readings' share done `since` s after progress `p` came (the reading's time over its expected, at least what is
 * known done, short of the next checkpoint, never full), and the seconds left while the estimate holds (none once the
 * reading runs past it). */
export function progressAt(p: LookupProgress, since: number): { share: number; left?: number } {
	const t = p.elapsed + Math.max(0, since), f = p.expect > 0 ? t / p.expect : 0;
	const at = Math.min(Math.max(f, p.done), p.done + AHEAD * (p.next - p.done));
	return { share: Math.min(MOST, (p.reading + at) / 2), left: f < 1 ? p.expect - t + (p.reading ? 0 : p.expect) : undefined };
}
const BAR_LEAST = 10;   // columns the bar keeps, the readout shortened first
/** The readings' line `w` wide: which reading and the seconds left (from LEFT_FROM), shorter where narrow, then the bar
 * of `share` (heavy blue on the rule). */
export function lookupBar(th: any, share: number, reading: number, left: number | undefined, w: number): string {
	const s = left !== undefined && left >= LEFT_FROM ? LABELS.secs(left) : "", i = reading + 1;
	const texts = [[LABELS.reading(i), s && `${s} left`], [LABELS.of2(i), s && `${s} left`], [LABELS.of2(i), s], [LABELS.of2(i)]].map((x) => x.filter(Boolean).join(" · "));
	const t = cut(texts.find((x) => w - textWidth(x) - 1 >= BAR_LEAST) ?? texts[texts.length - 1], w);
	const bw = Math.max(0, w - textWidth(t) - 1), cells = Math.max(0, Math.min(1, share)) * bw, full = Math.floor(cells), half = cells - full >= 0.5 ? 1 : 0;
	return th.fg("dim", t) + (bw ? " " + colourCode(COLOURS.help, colourMode(th)) + "━".repeat(full) + (half ? "╸" : "") + "\x1b[39m" + th.fg("rule", "─".repeat(bw - full - half)) : "");
}
/** nats: a token the engram helped (lookup.py HELPED), doubled, quadrupled: the three strengths of the blue. */
export const HELPED = 0.1, STRONG = Math.log(2), FULL = Math.log(4);

/** A reply's engram, worked out once per answer, by index in `r.tokens`: both log-probabilities (NaN: none, the last
 * token or one of another trace), the answer's row (`at`, -1 none), the strength (0 none, 1-3), where each token's
 * engram starts (`tri`: the first of the tokens up to `ngram` before it by the stream's index, so a gap in the
 * stream marks nothing across it); how many it helped. `differ`: the trace's ids disagree with the reply's (none
 * read, or one where it should agree). `id` tells one view from another (wrapTokens' cache). */
export interface LookupView {
	id: number;
	answer: LookupAnswer;
	on: Float32Array;
	off: Float32Array;
	at: Int32Array;
	level: Uint8Array;
	tri: Int32Array;
	helped: number;
	differ: boolean;
}
let views = 0;

/** Whether `a` is an answer the dock can draw, for trace `trace`: a row of each kind for every id, and the texts. */
export const isAnswer = (a: any, trace: string): a is LookupAnswer => !!a && a.trace === trace && Array.isArray(a.ids) &&
	Number.isInteger(a.ngram) && a.ngram > 0 &&
	[a.lp_on, a.lp_off, a.top_ids_on, a.top_lp_on, a.top_ids_off, a.top_lp_off].every((x) => Array.isArray(x) && x.length === a.ids.length) &&
	!!a.vocab && typeof a.vocab === "object";

/** Answer `a` laid on reply `r`'s tokens: token t is the trace's reply token t.i (an edit's reply: its own, from
 * `from`), taken only where the ids agree. */
export function lookupView(r: Reply, a: LookupAnswer): LookupView {
	const toks = r.tokens, n = toks.length, from = r.from ?? 0, on = new Float32Array(n).fill(NaN), off = new Float32Array(n).fill(NaN);
	const at = new Int32Array(n).fill(-1), level = new Uint8Array(n), tri = new Int32Array(n);
	let helped = 0, placed = 0, matched = 0;
	toks.forEach((t, k) => {
		let s = k;
		while (s > 0 && toks[s - 1].i >= t.i - a.ngram) s--;
		tri[k] = s;
		const j = t.i - from;
		if (j < 0 || j >= a.ids.length) return;
		placed++;
		if (a.ids[j] !== t.id) return;
		matched++;
		at[k] = j;
		on[k] = a.lp_on[j];
		off[k] = a.lp_off[j];
		const g = on[k] - off[k];
		if (!(g >= HELPED)) return;
		helped++;
		level[k] = g >= FULL ? 3 : g >= STRONG ? 2 : 1;
	});
	return { id: ++views, answer: a, on, off, at, level, tri, helped, differ: !matched || matched < placed };
}

/** The mode's colours: one muted blue (COLOURS.help) in three strengths toward the dim colour; the engram's block,
 * in the text (under the text colour) and in MAP, the blue past half way into the panel's background. */
const TINTS: Rgb[] = [0.5, 0.78, 1].map((t) => mix(COLOURS.dim, COLOURS.help, t));
export const MAP_BLOCK: Rgb = mix(COLOURS.bg, COLOURS.help, 0.55);
/** MAP's quiet cell (a token not helped, or not read): its unlit cell's colour a little toward the dim one. */
const QUIET: Rgb = mix(STOPS[0][1], COLOURS.dim, 0.25);

/** A frame of the sweep as the mode opens (dock.ts sweepNow): reading the reply without the engram (pass 0) or with
 * it (pass 1), its head at token `at` (-Infinity: waiting for the answer, nothing moving); the band of tokens it is on
 * drawn bright; reading with it, the blue behind. `close`: as the mode closes, one pass the other way (right to left),
 * the ordinary colours behind it (back). */
export interface Sweep {
	pass: 0 | 1;
	at: number;
	close?: boolean;
	/** The LENS edit applied (dock.ts editAt) instead: its token `at` struck, then fogged (`burn` 1, 2), then gone (3: a
	 * ▸ in its place, a swap's word `want` dim beside it); then the tokens after it cleared right to left, `frac` of the
	 * way. Nothing moves: every token keeps its place, cleared ones as blanks. */
	edit?: { at: number; burn: number; frac: number; want?: string };
}
const SCAN = 3;   // the band's tokens behind its head
/** The edit's head over the tokens after it, the last shown `hi`: right to left, none (Infinity) while the word burns. */
const editHead = (e: NonNullable<Sweep["edit"]>, hi: number) => (e.burn < 3 ? Infinity : hi - Math.floor(e.frac * (Math.max(hi, e.at) - e.at + SCAN + 1)));
/** A token's state under the edit's sweep: as ever (null), in its band, or cleared. */
const editState = (e: NonNullable<Sweep["edit"]>, head: number, j: number) => (j <= e.at || j < head ? null : j <= head + SCAN ? "band" : "gone");
/** The reply text under the edit's sweep (wrapTokens' paint; `from` where the text starts, `hi` the last token the strip
 * shows; `sel` the cursor): the edited token burnt in place, those after it cleared to blanks of their width (the
 * text before it keeps its rows; the answer's first token is laid out as text: a shorter word can fit at the end of the
 * row above, where the burnt one had wrapped), the band in the text colour. */
export function editPaint(th: any, e: NonNullable<Sweep["edit"]>, from: number, hi: number, sel = -1): { key: string; cursor: boolean; at: (k: number, s: string) => string | null } {
	const mode = colourMode(th), head = editHead(e, hi), ink = (c: Rgb, s: string) => colourCode(c, mode) + s + "\x1b[39m";
	return {
		key: `edit${e.at}:${e.burn}:${head}`,
		cursor: sel === e.at || (sel > e.at && editState(e, head, sel) !== null),   // on the edited token or one cleared: drawn here; else its inverse
		at: (k, s) => {
			const j = k + from, st = editState(e, head, j);
			if (j !== e.at) return st === "gone" ? " ".repeat(textWidth(s)) : st === "band" ? ink(COLOURS.text, s) : null;
			const lead = s.match(/^\s*/)![0], word = s.slice(lead.length), w = textWidth(word);
			if (!w) return null;
			if (e.burn === 0) return lead + ink(COLOURS.accent, `\x1b[9m${word}\x1b[29m`);
			if (e.burn < 3) return lead + ink(e.burn === 1 ? COLOURS.accent : COLOURS.faint, fog(word, e.burn));
			const want = cut(e.want ?? "", w - 1);
			return lead + ink(COLOURS.accent, "▸") + (want ? th.fg("dim", want) : "") + " ".repeat(Math.max(0, w - 1 - textWidth(want)));
		},
	};
}
/** MAP under the edit's sweep (`hi` the last token it shows): the edited token's cell orange, those after it unlit
 * behind the band. */
export function editMap(e: NonNullable<Sweep["edit"]>, hi: number): MapPaint {
	const head = editHead(e, hi), st = (i: number) => editState(e, head, i);
	return { shade: (i) => (i === e.at ? COLOURS.accent : st(i) === "gone" ? STOPS[0][1] : st(i) === "band" ? COLOURS.text : undefined),
		engram: [0, 0], pred: -1, block: STOPS[0][1] };
}
/** Whether token `k` is behind a closing sweep: drawn as ever (no paint). */
export const back = (sw: Sweep | undefined, k: number) => !!sw?.close && k > sw.at + SCAN;
/** Token `k`'s colour: its strength's blue, null for quiet; during a sweep the band's text colour, the blue only
 * behind a sweep with the engram (`v` absent: no answer yet, quiet); a closing one: the blue ahead of it. */
function tone(v: LookupView | undefined, k: number, sw?: Sweep): Rgb | null {
	if (sw?.close) {
		if (k >= sw.at && k <= sw.at + SCAN) return COLOURS.text;
		sw = undefined;
	}
	if (sw && k <= sw.at && k > sw.at - 1 - SCAN) return COLOURS.text;
	if (sw && (!sw.pass || k > sw.at)) return null;
	const l = v?.level[k] ?? 0;
	return l ? TINTS[l - 1] : null;
}
/** A token's MAP colour. */
export const lookupRgb = (v: LookupView | undefined, k: number, sw?: Sweep): Rgb => tone(v, k, sw) ?? QUIET;
/** The engram of cursor `sel` (-1 none): [first, past the last] of the tokens it looked up, where the token was read;
 * none during a sweep. */
export const engramOf = (v: LookupView | undefined, sel: number, sw?: Sweep): [number, number] =>
	predicted(v, sel, sw) >= 0 ? [v!.tri[sel], sel] : [0, 0];
/** The token the engram predicted: the cursor `sel` where it was read, else -1; none during a sweep. */
export const predicted = (v: LookupView | undefined, sel: number, sw?: Sweep): number => (!sw && v && sel >= 0 && v.at[sel] >= 0 ? sel : -1);

/** MAP in the mode (board.ts MapPaint), the cursor `sel`: the colours, the engram's cells in its block colour, the
 * token it predicted marked; the count of tokens helped in the title. */
export const lookupMap = (v: LookupView | undefined, sel: number, sw?: Sweep): MapPaint =>
	({ shade: (i) => (back(sw, i) ? undefined : lookupRgb(v, i, sw)), engram: engramOf(v, sel, sw), pred: predicted(v, sel, sw), block: MAP_BLOCK, count: v ? `${v.helped} helped` : sw?.close ? LABELS.engramStopped : "reading" });

/** The reply text's colours, `sel` the cursor (-1 none), `from` where the wrapped tokens start in the reply: helped
 * tokens in their blue, the rest quiet (faint); the cursor's engram on the block in the text colour, the cursor's token bold and
 * underlined in its own colour (`cursor`: then wrapTokens leaves it to the paint, no inverse; a token not read, or
 * during a sweep, keeps its inverse). `sw`: a sweep's frame. */
export function lookupPaint(th: any, v: LookupView | undefined, sel = -1, from = 0, sw?: Sweep): { key: string; cursor: boolean; at: (k: number, s: string) => string | null } {
	const mode = colourMode(th), [a, b] = engramOf(v, sel, sw), pred = predicted(v, sel, sw);
	const block = colourCode(MAP_BLOCK, mode, true), text = colourCode(COLOURS.text, mode);
	const ink = (j: number, s: string) => {
		const c = tone(v, j, sw);
		return c ? colourCode(c, mode) + s + "\x1b[39m" : th.fg("faint", s);
	};
	return {
		key: `look${v?.id ?? "-"}:${sel}:${sw ? `${sw.close ? "c" : sw.pass}:${sw.at}` : ""}`,
		cursor: pred >= 0,
		at: (k, s) => {
			const j = k + from;
			if (back(sw, j)) return null;   // behind a closing sweep: as ever
			if (j === pred) return ansiBold(`\x1b[4m${ink(j, s)}\x1b[24m`);
			if (j < a || j >= b) return ink(j, s);
			// The block from its first token's word (that token's leading space plain), the text colour on it.
			const lead = j === a ? (s.match(/^\s*/)?.[0] ?? "") : "", w = s.slice(lead.length);
			return lead + (w ? `${block}${text}${w}\x1b[39m\x1b[49m` : "");
		},
	};
}

/** The title line's readout for token `k` (-1: none, following), `w` wide: the engram it looked up, the token, and
 * its probability without and with it (or no change); else why there is nothing. */
export function lookupLine(v: LookupView, toks: { text: string; i: number }[], k: number, w: number): string {
	if (k < 0) return cut(`${v.helped} tokens helped`, w);
	const tok = qt(toks[k].text), on = v.on[k], off = v.off[k];
	if (Number.isNaN(on)) return cut(`${tok} ${k === toks.length - 1 ? "last token: not read" : "not read"}`, w);
	const before = v.tri[k] < k ? qt(toks.slice(v.tri[k], k).map((t) => t.text).join("")) : toks[k].i ? "?" : "the prompt";
	const helped = on - off >= HELPED, change = helped ? `without ${pct(Math.exp(off))} → with ${pct(Math.exp(on))}` : "no change";
	const long = `engram ${before} → ${tok}   ${change}`;
	if (textWidth(long) <= w) return long;
	const tail = ` → ${tok} ${helped ? `${pct(Math.exp(off))} → ${pct(Math.exp(on))}` : "no change"}`;
	return cut(before, Math.max(4, w - textWidth(tail))) + tail;
}

/** TOKENS in the mode, `w` wide in at most `lines`, for token `k` of `toks` (its id and text): its top choices
 * without and with the engram (both fresh readings, likeliest with it first; ▸ and bold the one written, an eleventh
 * row where it is not among them), a percentage and a bar each; too narrow for two bars the without one goes, then
 * both; the written token's own percentages are its row's (the title's). Undefined where the token was not read. */
export function lookupTokens(th: any, v: LookupView, toks: { text: string; id: number }[], k: number, w: number, focused: boolean, lines = Infinity, titled = true): string[] | undefined {
	const j = v.at[k];
	if (j < 0) return undefined;
	const a = v.answer, t = toks[k], mode = colourMode(th), blue = (s: string) => colourCode(COLOURS.help, mode) + s + "\x1b[39m";
	const text = (id: number) => (id === t.id ? t.text : a.vocab[String(id)] ?? `<${id}>`);
	const offIds = a.top_ids_off[j], offLp = a.top_lp_off[j], floor = Math.exp(Math.min(...offLp));
	const mine = { id: t.id, on: Math.exp(v.on[k]), off: Math.exp(v.off[k]) };
	const rows: { id: number; off: number | null; on: number }[] = a.top_ids_on[j].map((id, i) => (id === t.id ? mine
		: { id, on: Math.exp(a.top_lp_on[j][i]), off: offIds.includes(id) ? Math.exp(offLp[offIds.indexOf(id)]) : null }));
	if (!rows.includes(mine)) rows.push(mine);
	// The columns: a name, then without and with, each a percentage and (where the width allows) a bar.
	const ww = Math.max(8, Math.min(18, Math.floor(w * 0.3))), bars = w >= 2 + ww + 2 * (7 + 4) ? 2 : w >= 2 + ww + 6 + 7 + 4 ? 1 : 0;
	const bw = bars === 2 ? Math.floor((w - 2 - ww - 14) / 2) : bars ? w - 2 - ww - 13 : 0;
	const offW = 6 + (bars === 2 ? 1 + bw : 0);
	const col = (p: number | null, show: boolean, bw: number) => lpad(p === null ? `<${pct(floor)}` : pct(p), 6) + (show ? " " + padTo(p === null ? "" : bar(p, bw), bw) : "");
	// The title, the token's number, and each column's name over it.
	const head = (titled ? title(th, "TOKENS", focused) + "  " : "") + th.fg("dim", `#${k}`), at = 2 + ww;
	const heads = padTo(offW >= 8 ? "without" : "w/o", offW) + "with";
	const out = [textWidth(head) + 1 <= at ? padTo(head, at) + th.fg("dim", heads) : head];
	for (const r of keepRow(rows, lines - 1, rows.indexOf(mine))) {
		const mine = r.id === t.id, name = (mine ? "▸ " : "  ") + padTo(label(text(r.id)), ww);
		out.push((mine ? th.fg("text", ansiBold(name)) : th.fg("dim", name)) + th.fg("dim", col(r.off, bars === 2, bw)) + blue(col(r.on, bars > 0, bw)));
	}
	return out.slice(0, lines);
}

/** Engram inspection's sweep (sweepAt): a pass across the reply in SWEEP_MS, without the engram then with it (1.3 s
 * with the answer in hand), at most SWEEP_PASSES while it is coming; Pi asked to draw every SWEEP_TICK_MS meanwhile. */
export const SWEEP_MS = 650, SWEEP_PASSES = 4, SWEEP_TICK_MS = 70;
/** The LENS edit applied (editAt): the word struck, then burnt out a tick at a time (EDIT_BURN_MS), then the tokens
 * after it swept away right to left in SWEEP_MS; the ticker the same. */
export const EDIT_BURN_MS = 3 * SWEEP_TICK_MS;
/** The readings' bar (lookupBar) is drawn again every BAR_TICK_MS while it shows, with a clock: its estimate moves. */
export const BAR_TICK_MS = 500;
let asks = 0;   // g's asks in this Pi: each one's id (with the time) tells its progress and its stop from another's
let stopping: Promise<void> | undefined;   // g's stops in this Pi not yet taken by the server: the next ask waits for them

/** The sweep `t` ms after it began, the answer in hand from `ready` ms (undefined: not yet): passes of SWEEP_MS,
 * without the engram then with it (`frac` of the way across), alternating up to SWEEP_PASSES while the answer is
 * coming, then held (quiet, nothing moving). The last pass is with it: the first such pass to begin with the answer in
 * hand, or, the answer coming while it held, one from then. "done" once that pass is over. */
export function sweepAt(t: number, ready?: number): { pass: 0 | 1; frac: number } | "hold" | "done" {
	const held = SWEEP_PASSES * SWEEP_MS, pass = { pass: (Math.floor(t / SWEEP_MS) % 2) as 0 | 1, frac: (t % SWEEP_MS) / SWEEP_MS };
	if (ready === undefined) return t >= held ? "hold" : pass;
	const k = Math.max(0, Math.ceil(ready / SWEEP_MS)), last = ready < held ? (k % 2 ? k : k + 1) * SWEEP_MS : ready;
	if (t >= last + SWEEP_MS) return "done";
	return t >= last ? { pass: 1, frac: (t - last) / SWEEP_MS } : pass;
}

/** A ticker of the dock's (the idle scene, engram inspection's sweep): `step` every `ms` while Pi draws the dock
 * (`drawn`, set by each draw); a tick that finds it not drawn since the one before ends it through `lost` (Pi draws
 * the dock no more: hidden), and one whose owner let it go (`mine` false) clears it. The lens swap keeps its own. */
export interface Loop {
	drawn: boolean;
	timer: ReturnType<typeof setInterval>;
}
export function loop(ms: number, mine: (l: Loop) => boolean, lost: () => void, step: () => void): Loop {
	const l: Loop = { drawn: true, timer: setInterval(() => {
		if (!mine(l)) return clearInterval(l.timer);
		if (!l.drawn) return lost();
		l.drawn = false;
		step();
	}, ms) };
	return l;
}

/** What engram inspection may ask of the Dock: the dock gone, the token in focus, what MAP and the reply strip drew last
 * (the sweep crosses them), the key line's note. */
export interface EngramDock {
	readonly disposed: boolean;
	readonly sel: number;
	cur(n: number): number;
	readonly mapShown?: [string, number, number];
	readonly stripAt?: { rid: string; last: number };
	boardNote: string;
}

/** Engram inspection in the dock (g): its asks and stops, the view worked out per reply, the title line's readout and
 * the readings' bar, MAP's and the reply strip's paint; and the sweep, the ticker it opens and closes with (the LENS edit
 * and the lenses going off or on ride on it too). The Dock (dock.ts) holds one. */
export class Engram {
	constructor(private host: DockHost, private theme: DockTheme, private tui: { readonly terminal?: any; requestRender(): void }, private piTui: { requestRender(): void },
		private dock: EngramDock) {}
	private views = new WeakMap<Reply, LookupView>();   // g: each reply's engram view, worked out once (look)
	// g: the sweep as engram inspection opens (sweepNow): its trace, when it began and when this dock first saw the
	// answer (`ready`), its ticker (none while it holds); this frame's (`sw`, paint). `close`: the pass as it closes;
	// `edit`: instead, the LENS edit applied (editAt) to reply `old`'s token `at` (TokenEvent.i), until its answer's
	// first token; `lens`: instead, the lenses going off (`off`) or back on, LENS swept in `ms` (dulled).
	sweep?: { trace: string; t0: number; ready?: number; loop?: Loop; close?: boolean; edit?: { old: Reply; at: number; want?: string }; lens?: { off: boolean; ms: number } };
	sw?: Sweep;
	// g: the readings' bar's ticker (lookupNote) while drawn (`barNow`: this render drew it).
	private bar?: Loop;
	barNow = false;
	stopBar() {
		clearInterval(this.bar?.timer);
		this.bar = undefined;
	}
	/** g on the key lines: the server listed the engram (cap `lookup`) when last asked (host.caps; a hint, g asks again). */
	private lookupCap = () => !!this.host.lookup && (this.host.caps?.() ?? []).includes("lookup");
	/** g's label on the key lines for reply `r`: while its engram inspection is on, g stop as the server reads it, else
	 * g close (refused: g asks again); else g engram inspection where offered; "" neither. */
	gKey = (r = shownReply(this.host.live)) => {
		const x = r?.traceId && this.host.lookupOn === r.traceId ? this.host.looked?.get(r.traceId) ?? {} : undefined;
		return x && !x.error ? (x.ask ? LABELS.engramStop : LABELS.engramClose) : this.lookupCap() ? LABELS.engramKey : "";
	};
	/** Trace `id`'s readings called off while the server runs them (g closing, or g on another reply): forgotten, the
	 * server told to stop (its answer, a 409, dropped; `stopping` until it took it). Whether they were running. */
	private stopAsk(id: string): boolean {
		const h = this.host, x = h.looked?.get(id);
		if (!x?.ask) return false;
		h.looked!.delete(id);
		const all: Promise<void> = (stopping = Promise.allSettled([stopping, h.stopLookup?.(id, x.ask)]).then(() => {
			if (stopping === all) stopping = undefined;
		}));
		return true;
	}
	/** g: reply `r`'s engram inspection closed when on (always; called off while it reads: stopAsk), else opened
	 * (host.lookupOn) on a finished, saved reply where the server lists it: the caps as last known, else asked for once
	 * more (host.freshCaps: a model loaded since); else a note. Asked for once while its trace is the last opened
	 * (host.looked), again after a refusal or a stop; opened, the sweep starts (startSweep). No reply shown: nothing. */
	lookupKey(r: Reply | undefined) {
		const h = this.host, ask = h.lookup, id = r?.traceId;
		if (!r) return;
		if (id && h.lookupOn === id && !h.looked?.get(id)?.error) {
			// Closed: a pass back to the ordinary colours where it is drawn and there is a clock (stopSweep closes it at
			// its end); else at once.
			const stopped = this.stopAsk(id);
			if (!(this.lookView(r) || stopped) || !this.startSweep(id, true)) h.lookupOn = undefined;
			return;
		}
		if (!ask) return void (this.dock.boardNote = LABELS.lookupNone);
		const go = () => {
			if (!r.done || !id) return void (this.dock.boardNote = !r.done ? LABELS.lookupWait : LABELS.lookupUnsaved);
			const looked = (h.looked ??= new Map()), got = looked.get(id);
			if (h.lookupOn && h.lookupOn !== id) this.stopAsk(h.lookupOn);
			for (const k of looked.keys()) if (k !== id) looked.delete(k);   // asked again when opened again (the server's copy)
			h.lookupOn = id;
			this.startSweep(id);
			if (got && !got.error) return;
			const mine: Looked = { ask: `${Date.now().toString(36)}-${++asks}` };
			looked.set(id, mine);
			const send = () => (looked.get(id) === mine ? ask(id, mine.ask!) : undefined);
			// Every stop sent so far taken by the server before this ask comes (stopping).
			new Promise<LookupAnswer | undefined>((sent) => sent(stopping ? stopping.then(send) : send()))
				.then((a): Looked => (isAnswer(a, id) ? { answer: a } : { error: LABELS.lookupBad }), (e: any): Looked => ({ error: String(e?.message ?? e) }))
				.then((x) => {
					if (looked.get(id) !== mine) return;   // called off meanwhile
					looked.set(id, x);
					this.redraw();
				}).catch(() => {});
		};
		if (this.lookupCap()) return go();
		if (!h.freshCaps) return void (this.dock.boardNote = LABELS.lookupNone);
		Promise.resolve().then(() => h.freshCaps!()).then((caps) => {
			if (caps.includes("lookup")) go();
			else this.dock.boardNote = LABELS.lookupNone;
		}, () => void (this.dock.boardNote = LABELS.lookupNone)).then(() => this.redraw()).catch(() => {});
	}
	/** Something the dock waited for came: drawn by whichever dock is open now (one closed meanwhile asked for it). */
	private redraw() {
		const live = this.host.live;
		if (live.requestRender) live.requestRender();
		else if (!this.dock.disposed) this.tui.requestRender();
	}
	/** Reply `r`'s engram inspection while it is on (g): its answer or refusal, and its view, worked out once for `r` (kept by
	 * the reply, not for the session); else undefined. */
	look(r: Reply): { looked: Looked; view?: LookupView } | undefined {
		const h = this.host;
		if (!r.traceId || h.lookupOn !== r.traceId) return undefined;
		const looked = h.looked?.get(r.traceId);
		if (!looked?.answer) return looked && { looked };
		let view = this.views.get(r);
		if (view?.answer !== looked.answer) this.views.set(r, (view = lookupView(r, looked.answer)));
		return { looked, view };
	}
	/** The engram's view for reply `r` where it is drawn: shown, answered, its ids agreeing with the reply's. */
	lookView(r: Reply): LookupView | undefined {
		const v = this.look(r)?.view;
		return v && !v.differ ? v : undefined;
	}
	/** Engram inspection's cursor: the token stepped to; following (the newest), none. */
	lookAt = (sel: number) => (sel >= 0 && this.dock.sel >= 0 ? sel : -1);
	/** The title line's readout for reply `r`'s token `sel` (following: the count), `w` wide (lookupLine); measuring, the
	 * readings' bar (once their progress comes), the server's refusal, or ids that disagree; called off, "stopped" while
	 * the closing pass runs; "" while it is not on. */
	lookupNote(r: Reply, sel: number, w: number): string {
		const x = this.look(r), th = this.theme;
		if (!x) return this.sw?.close && this.host.lookupOn === r.traceId ? th.fg("dim", cut(LABELS.engramStopped, w)) : "";
		if (x.view?.differ) return th.fg("warning", cut(LABELS.lookupDiffer, w));
		if (x.view) return th.fg("text", lookupLine(x.view, r.tokens, this.lookAt(sel), w));
		if (x.looked.error) return th.fg("warning", cut(x.looked.error, w));
		const p = this.progress(x.looked.ask);
		return p ? lookupBar(th, p.share, p.reading, p.left, w) : th.fg("dim", cut(LABELS.lookupBusy, w));
	}
	/** The readings' progress for ask `ask` now (progressAt: from when the newest came, never below what a bar showed);
	 * undefined before the first. Drawn with a clock, its ticker runs (BAR_TICK_MS; stopped by paint once not drawn, by
	 * loop once the dock is hidden). */
	private progress(ask?: string): { share: number; reading: number; left?: number } | undefined {
		const p = this.host.live.lookup, now = this.host.now?.();
		if (!ask || p?.ask !== ask) return undefined;
		const got = progressAt(p, now === undefined ? 0 : (now - p.at) / 1000);
		p.most = Math.max(p.most, got.share);
		if (now !== undefined) {
			this.bar ??= loop(BAR_TICK_MS, (l) => this.bar === l, () => this.stopBar(), () => this.tui.requestRender());
			this.bar.drawn = this.barNow = true;
		}
		return { ...got, share: p.most, reading: p.reading };
	}
	/** The legend engram inspection puts in the rule over its title, `w` wide (the sweep's line while it runs, but for
	 * the readings: the title line has their bar); "" while it is not drawn. */
	legend(r: Reply, w: number): string {
		const sw = this.sw?.edit ? undefined : this.sw;
		if (!this.lookView(r) && !sw?.close) return "";
		return ruleText(this.theme, sw ? [sw.close ? LABELS.engramClosing : LABELS.engramSweep[sw.pass]] : LABELS.engramLegend, w);
	}
	/** MAP in engram inspection while it is drawn (answered, or sweeping), the cursor `sel`'s engram marked. */
	mapPaint(r: Reply, sel: number): MapPaint | undefined {
		if (this.sw?.edit) return editMap(this.sw.edit, (this.dock.mapShown?.[0] === r.rid ? this.dock.mapShown[2] : r.tokens.length) - 1);
		const v = this.lookView(r);
		return v || this.sw ? lookupMap(v, this.lookAt(sel), this.sw) : undefined;
	}
	/** The reply text in engram inspection while it is drawn (wrapTokens' paint); `from` where the text starts. */
	textPaint(r: Reply, sel: number, from = 0) {
		if (this.sw?.edit) return editPaint(this.theme, this.sw.edit, from, this.dock.stripAt?.rid === r.rid ? this.dock.stripAt.last : r.tokens.length - 1, sel);
		const v = this.lookView(r);
		return v || this.sw ? lookupPaint(this.theme, v, this.lookAt(sel), from, this.sw) : undefined;
	}
	/** Engram inspection opened on trace `trace`, with a clock (host.now; none: its last frame at once): the sweep
	 * (sweepAt), its ticker (sweepLoop). The same ticker for a word edit (`edit`) and the lenses off or on (`lens`). */
	startSweep(trace: string, close = false, edit?: NonNullable<Engram["sweep"]>["edit"], lens?: NonNullable<Engram["sweep"]>["lens"]): boolean {
		this.stopSweep();
		const t0 = this.host.now?.();
		if (t0 === undefined) return false;
		const s: NonNullable<Engram["sweep"]> = { trace, t0, close, edit, lens };
		this.sweep = s;
		this.sweepPhase(s, t0);
		s.loop = this.sweepLoop(s);
		return true;
	}
	/** Sweep `s`'s ticker: Pi asked to draw every SWEEP_TICK_MS (the redraw cap still decides: render); over, the last
	 * frame drawn at once and the sweep gone; held, the ticker stops (sweepNow starts it again with the answer). Ended
	 * too by any key, by a tick that finds the dock not drawn since the last (loop), or as the dock goes. */
	private sweepLoop(s: NonNullable<Engram["sweep"]>): Loop {
		return loop(SWEEP_TICK_MS, (l) => this.sweep === s && s.loop === l, () => this.stopSweep(), () => {
			const p = this.sweepPhase(s, this.host.now!());
			if (p === "done") this.stopSweep();
			else if (p === "hold") s.loop = void clearInterval(s.loop!.timer);
			if (typeof p === "string") return this.tui.requestRender();   // the last frame, or the held one, at once
			this.piTui.requestRender();
		});
	}
	/** The sweep ends (done, a key, the dock hidden or gone, another reply); a closing one closes the mode then. */
	stopSweep() {
		const s = this.sweep;
		if (!s) return;
		clearInterval(s.loop?.timer);
		this.sweep = undefined;
		this.sw = undefined;
		if (s.close && this.host.lookupOn === s.trace) this.host.lookupOn = undefined;
	}
	/** Sweep `s` at time `now` (sweepAt; a closing one: one pass), the answer taken as in hand from the first time it
	 * is seen here (for the session: whichever dock asked for it). */
	sweepPhase(s: NonNullable<Engram["sweep"]>, now: number): ReturnType<typeof sweepAt> {
		// An edit's: the word's burn, then one pass; then held (swept) until its answer's first token.
		if (s.edit) return now - s.t0 >= EDIT_BURN_MS + SWEEP_MS ? "hold" : { pass: 1, frac: Math.max(0, now - s.t0 - EDIT_BURN_MS) / SWEEP_MS };
		if (s.close || s.lens) {   // a closing pass, or LENS's: once
			const ms = s.lens?.ms ?? SWEEP_MS;
			return now - s.t0 >= ms ? "done" : { pass: 1, frac: (now - s.t0) / ms };
		}
		if (s.ready === undefined && this.host.looked?.get(s.trace)?.answer) s.ready = now;
		return sweepAt(now - s.t0, s.ready === undefined ? undefined : s.ready - s.t0);
	}
	/** This frame of the sweep on reply `r` (paint), undefined once it is over (then stopped): the head crosses the tokens
	 * MAP drew last (else the strip's, around the cursor); held, quiet. A refusal, or ids that disagree, stop it. */
	sweepNow(r: Reply | undefined): Sweep | undefined {
		const s = this.sweep, h = this.host;
		if (!s || s.lens) return undefined;   // the lenses' sweep dulls LENS alone (dulled)
		if (s.edit) {
			// Over once the reply shown is another, or the edit's answer with a token of its own: that token is drawn in the
			// word's place (a fast answer cuts the burn or the sweep short: what was left of the old tail goes with it).
			const e = s.edit, now = h.now!(), p = this.sweepPhase(s, now);
			if (r !== e.old) return void this.stopSweep();
			return { pass: 1, at: -1, edit: { at: e.old.tokens.findIndex((t) => t.i === e.at), burn: Math.min(3, Math.floor((now - s.t0) / SWEEP_TICK_MS)),
				frac: p === "hold" ? 1 : (p as { frac: number }).frac, want: e.want } };
		}
		const looked = h.looked?.get(s.trace);
		if (!r || r.traceId !== s.trace || h.lookupOn !== s.trace || looked?.error || this.look(r)?.view?.differ) return void this.stopSweep();
		const p = this.sweepPhase(s, h.now!());
		if (p === "done") return void this.stopSweep();
		if (p === "hold") return { pass: 0, at: -Infinity };
		s.loop ??= this.sweepLoop(s);   // the answer came while it held: the last pass
		const n = r.tokens.length, anchor = this.lookAt(this.dock.cur(n)) >= 0 ? this.dock.cur(n) : n - 1;
		const [lo, hi] = this.dock.mapShown?.[0] === r.rid ? this.dock.mapShown.slice(1) as [number, number] : [Math.max(0, anchor - 150), Math.min(n, anchor + 30)];
		const step = Math.floor(p.frac * (hi - lo + 4));
		return s.close ? { pass: 1, at: hi - step, close: true } : { pass: p.pass, at: lo - 1 + step };   // closing: right to left
	}
}

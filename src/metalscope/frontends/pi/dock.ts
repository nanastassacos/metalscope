/**
 * Dock: a live side panel with its own background. With a selection it is inspect mode, stepping through the last
 * reply token by token. The board (board.ts): its panels on one screen, or, in a dock too small for that
 * (boardFits), the one-panel board, the panel in focus alone; 6 the Steer page over it (the concepts, the cursor's
 * dose curve, its test chats). m, or a terminal too short for the full dock, the mini dock. Before the
 * session's first token, tracing on, the idle scene (idle.ts).
 */

import { matchesKey, truncateToWidth, visibleWidth, wrapTextWithAnsi } from "@earendil-works/pi-tui";
import { NAME, parseNumber } from "./args.ts";
import { type DockTheme, EDIT_HINT, EXPERT_HELP, EXPLAIN, LABELS, UNSURE, ansiInverse, bold, centreLeft, clean, cut, fit, label, lensSettle, noteLines, padAbove, plainWidth, tint, wrapTokens, dockTheme } from "./draw.ts";
import { Grids, colourMode, gridCells, gridShape, palette, reshapeGrid } from "./lights.ts";
import { IDLE_FRAMES, IDLE_LINES, IDLE_MS, idleLines } from "./idle.ts";
import { DOCK_MIN_ROWS, DOCK_MIN_WIDTH, LEFT_MIN, MIDDLE_MIN, ONE_REPLY_ROWS, type Panel, PANEL_KEY, PANELS, REPLY_ROWS, RIGHT_W, STEER_KEY,
	arrange, BOARD_TOP, boardFits, boardKeys, bothSources, columns, DULL, dull, dullLines, dullPace, editKeys, expertKeys, explainOf, expertsStack, layerKeys, TOP_N,
	type LayerLights, lensSplit, MAP_CELL_LINES, MAP_LINES, mapPanel, mapWindow, MINI_MAP, MINI_SETTLE_LEAST, type MiniRoom, miniAt, miniKeys, miniSplit,
	streamShare, miniToken, replyRows, sessionMap, rule as ruleLine, settleLayers, settlingPanel, side,
	tabRow, title as panelTitle, layoutKeys, panelNums, tokensPanel, wrapKeys, STREAM_FIXED, STREAM_FRAME, STREAM_MIN, type StreamView, nextView, streamPanel, streamFrames, streamLines, streamViews,
	viewOf, PLAY_TICK_MS, type Play, pace, playShown } from "./board.ts";
import { Experts, GET_TIMEOUT_MS, type Source, TALLY_LAG_MS, expertPage, layerBoth, layerPage, routedLayers, shownLayer } from "./expert.ts";
import { Engram, type Loop, type LookupAnswer, type Looked, SWEEP_TICK_MS, loop, lookupTokens } from "./lookup.ts";
import { type ModelFlags, hostOf, scopeAuth } from "./provider.ts";
import { type Edited, type Live, type Reply, TOOL_OPENERS, type TokenEvent, editChanged, editMark, joined, shownReply } from "./state.ts";
import { EDIT_SWAP_DOSE, type WordIds, steerLabel, tokenName } from "./steer.ts";
import { type Examples, LENS_LINK, type LinkOffer, type SteerList } from "./steerpage.ts";
import { SteerPage } from "./steerdock.ts";
import { type Frame, MIN_LINES, MIN_WIDTH, bothLenses, eyepiece, fitCap, focusLayer, lensOf, onTube, readLayers, scopeHeight, stepLayer, stretchFrames, swapFrames, telescope, tubeOf, turret } from "./telescope.ts";

/** What the dock needs from the extension; outlives any one dock. */
export interface DockHost {
	live: Live;
	captions: boolean; // ? in the dock: what the part in focus shows, where the key line is
	close(): void; // close the dock
	unfocus(): void; // hand the keyboard back to the editor
	lensView?: "jlens" | "logit"; // s: which lens LENS shows where a token has both (display only)
	expertLayer?: number; // [ and ]: the MoE layer EXPERTS shows, from 0 (undefined: the middle one)
	expertSource?: Source; // a: EXPERTS' tokens (the grids where they show one, the layer page where it shows one, the expert page) over every traced chat or this session
	lensLayer?: number; // ↑/↓ in LENS: the layer focused (the nearest read one; undefined: the settle layer)
	fetchJson?(url: string): Promise<any>; // the grid's request (tests stand in for fetch)
	now?(): number; // the redraw cap's clock, in ms (widget.ts: performance.now; tests a stand-in); none: every render draws
	// The Steer page (widget.ts; without `steering` it says to use /steer):
	steering?(): Promise<SteerList | { error: string }>; // the served model's learned concepts, ranked (one line on failure)
	steerChats?(name: string, dose?: number): Promise<Examples | { error: string }>; // a concept's test chats at `dose` (its picked one), cached for the session
	// /steer ARGS through /steer's own handler; what it said.
	steer?(args: string): Promise<{ text: string; level: string }[]>;
	// d d: the concept's data on this model to the trash (steering it turned off first); what came of it.
	deleteConcept?(name: string): Promise<{ text: string; level: string }[]>;
	// The served model's link offer, as the server answers it (GET /scope/api/model/link; asked once a served model,
	// widget.ts), and l: Pi's pick list, then the link; what came of it (undefined: kept separate).
	linkOffer?(): Promise<any>;
	link?(offer: LinkOffer): Promise<{ text: string; level: string }[] | undefined>;
	// A LENS word edit (its steer as -OLD or OLD>NEW, `ids` its words by token: name, id, text): reply `rid` answered again
	// from its token `at`, the edit at that token only (widget.ts; the server's edit_at); what /steer said, an error
	// when not sent. `word`, `want`: the words as shown.
	edit?(args: string, ids: WordIds, at: { rid: string; at: number; word: string; want?: string }): Promise<{ text: string; level: string }[]>;
	lensLayers?(step?: number): Promise<LensLayers | { error: string; missing?: boolean }>; // LENS's - +: the server's lens steps (step: set it); missing: a 404
	steerModel?(): string; // the served model now (another one fetches the concepts again)
	flags?(): ModelFlags; // its flags (status `chat`): tested false marks the header untested, edit false hides the LENS edit
	steerAt?: string; // the concept the Steer page's cursor is on, kept across docks
	panel?: Panel; // tab on the board: the panel in focus (undefined: LENS)
	mini?: boolean; // m: the mini dock, lights (MAP, SETTLING and STREAM under them), MINI_WIDTH columns (remembered in pi.json; drawn too in a terminal too short for the full one: miniAt)
	stream?: StreamView; // s on STREAM: the view chosen (undefined: the model's first, board.ts viewOf; remembered in pi.json)
	split?: number; // { }: EXPERTS' layers' lines the board's RESIDUAL STREAM takes (negative: gives; remembered in pi.json)
	layoutChanged?(): void; // after m or STREAM's s: remembered, the stream asked for what is shown, the dock's room refitted (widget.ts)
	roomChanged?(): void; // Dock.streamRoom changed: the stream asks for the writes again (widget.ts)
	// g, engram inspection (lookup.ts): the server's caps (status `caps`; g where they list `lookup`), POST
	// /scope/api/lookup for a trace as ask `ask` (throws the server's detail; its progress on the live feed: Live.lookup)
	// and its stop (settled once the server took it, or failed), the answer by trace (the last opened's alone: the
	// server keeps each beside its trace), the trace shown.
	caps?(): readonly string[];
	freshCaps?(): Promise<readonly string[]>; // the caps asked for now (g where the last known lack `lookup`)
	lookup?(trace: string, ask: string): Promise<LookupAnswer>;
	stopLookup?(trace: string, ask: string): Promise<void>;
	looked?: Map<string, Looked>;
	lookupOn?: string;
}

/** GET /scope/api/lens/layers: the step the next traced reply reads the lenses at, and each step's name (small,
 * medium, large), layer count (`n`), layers (both lenses') and the decode speed of the served model's last replies read
 * at it (`tps`, over `replies`). */
export interface LensLayers {
	model: string;
	at: number;
	steps: { name: string; n: number; layers: number[]; tps: number | null; replies: number }[];
	edit_band?: [number, number]; // where the word edit acts (the removal's band; a swap pushes at pushLayer); none: no J-lens
}
/** What went wrong in the dock, as drawn: one line, cleaned (a message can quote what the server sent). */
const failed = (e: any) => `the dock: ${clean(String(e?.message ?? e)).replace(/[\n\t]/g, " ")}`;
/** LensLayers' `at` with the lenses off: - past the fewest layers (+ the fewest again); replies traced without them. */
export const LENS_OFF = -1;
const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
/** Whether `a` is a lens-steps answer the dock can draw: a step in use among its steps, each step whole. */
export const isLensLayers = (a: any): a is LensLayers => !!a && typeof a.model === "string" && Array.isArray(a.steps) && a.steps.length > 0 &&
	Number.isInteger(a.at) && a.at >= LENS_OFF && a.at < a.steps.length &&
	a.steps.every((s: any) => !!s && typeof s.name === "string" && isNum(s.n) && Array.isArray(s.layers) && s.layers.every(isNum) && (s.tps === null || isNum(s.tps)) && isNum(s.replies)) &&
	(a.edit_band === undefined || (Array.isArray(a.edit_band) && a.edit_band.length === 2 && a.edit_band.every(isNum)));

/** The board's panels in the order they get what their content needs in a column (dock.heights), past their least, each
 * up to the lines given here: TOKENS its eleventh row, LENS the telescope whole and SETTLING its want (lensNeed),
 * EXPERTS EXPERTS_WANT layers, MAP its least. STREAM's lines are set before any of it (its least is its want: board.ts
 * streamLines, streamShare), so EXPERTS gives up layers, down to one, before STREAM's history shrinks. What is left is
 * the column's slack: the reply strip takes what both columns spare (board.ts replyRows); then, in GROW_ORDER, EXPERTS
 * a layer at a time, MAP the rest of its column up to MAP_CELL_LINES (a window that slides, board.ts mapWindow),
 * SETTLING up to its most, the largest step's tube up to a layer a line, and STREAM the rest of its column (it sits on
 * the key line's rule). Lines no panel needs stay blank under the column's last panel. Nothing here follows the reply's
 * length, so no panel changes height as a reply grows. LENS's lines follow no step (lensNeed) but the largest's,
 * which takes MAP's lines over its least (heights); a smaller step leaves the lines under it (lensFit): - + move the
 * left column alone. */
const EXPERTS_WANT = 4;
const WANT_ORDER: [Panel, number?][] = [["TOKENS"], ["LENS"], ["EXPERTS"], ["MAP"]];
const GROW_ORDER: Panel[] = ["EXPERTS", "MAP", "LENS", "STREAM"];
/** A panel's lines in a column: its least, what its content needs, its most, the step it grows by (EXPERTS: a layer),
 * and the lines it gives up to first where the column is short (`floor`: LENS its telescope at SCOPE_EASY, SETTLING
 * gone; TOKENS TOKENS_FLOOR choices). */
interface Need { least: number; want: number; most: number; step?: number; floor?: number }
type AnimFrame = Frame & { stream?: StreamView; clear?: number; large?: number };
/** LENS's lines in its box (Dock.lensFit). */
type LensFit = { gap: number; sn: number; room: number; cap?: number; rows: number };
/** Whether step `at` (off as the fewest) is the largest of `d`'s: LENS then takes MAP's lines too. */
const isLargest = (d: LensLayers, at: number) => Math.max(0, at) === d.steps.length - 1;
/** Too few lines for every least, the panels that give lines up, each down to the lines given: SETTLING first, then
 * TOKENS' last choices, then the telescope down to its least, MAP to its title, the rest. */
const GIVE_ORDER: [Panel, (n: Need) => number][] = [["LENS", (n) => n.floor ?? 1], ["TOKENS", (n) => n.floor ?? 1],
	["LENS", () => 1 + MIN_LINES], ["MAP", () => 1], ["TOKENS", () => 1], ["EXPERTS", () => 1], ["LENS", () => 1]];
/** On the board, the telescope's least past which TOKENS gives its last choices first: its title, the eyepiece and 6
 * rows of tube (MIN_LINES draws one). */
const SCOPE_EASY = 1 + MIN_LINES + 5;
/** TOKENS' choices kept while SETTLING and its last choices give way (GIVE_ORDER). */
const TOKENS_FLOOR = 4;
/** The token that went into the forward pass the lens read for reply token `k`, by the stream's index `i` (which counts
 * special tokens never published, and a reply adopted mid-stream starts part way): the token before when it is the
 * one just before in the stream; null for the reply's first (i 0: the prompt's last token went in); false when that
 * token was not streamed (telescope: "in: not streamed"). */
export function inputOf(toks: { text: string; i: number }[], k: number): string | null | false {
	const t = toks[k];
	if (!t) return false;
	if (t.i === 0) return null;
	return inputAt(toks, k) >= 0 ? toks[k - 1].text : false;
}
/** The index in `toks` of the token that went in for token `k` (inputOf), or -1 when it is not among them. */
export const inputAt = (toks: { i: number }[], k: number): number => (k > 0 && toks[k] && toks[k - 1]?.i === toks[k].i - 1 ? k - 1 : -1);

/** Where a jump lands from token `sel`: `step` moves by a fixed count (clamped);
 * `sentence` moves to the start of the next sentence (dir > 0) or of the
 * current one, or the previous when already at its start (dir < 0). A
 * sentence starts after a token that ends with . ! ? or holds a newline. */
export function jumpTo(toks: { text: string }[], sel: number, dir: 1 | -1, kind: "step" | "sentence", step = 10): number {
	const n = toks.length;
	if (!n) return 0;
	if (kind === "step") return Math.max(0, Math.min(n - 1, sel + dir * step));
	const ends = (i: number) => /[.!?]["')\]]*\s*$/.test(toks[i].text) || toks[i].text.includes("\n");
	// A sentence starts at the first token with any text after an end; bare newline / space tokens are skipped.
	const startAfter = (i: number) => { let j = i + 1; while (j < n - 1 && !toks[j].text.trim()) j++; return j; };
	if (dir > 0) {
		for (let i = sel; i < n - 1; i++) if (ends(i) && startAfter(i) > sel) return startAfter(i);
		return n - 1;
	}
	let start = 0;
	for (let i = sel - 1; i >= 0; i--) if (ends(i) && startAfter(i) <= sel) { start = startAfter(i); break; }
	if (start < sel) return start;                       // to the start of this sentence
	for (let i = start - 2; i >= 0; i--) if (ends(i) && startAfter(i) < start) return startAfter(i);   // already there: the previous one
	return 0;
}

/** The redraw cap (Dock.render): while nothing but the stream's progress is new, the dock draws again no sooner than
 * its last draw's time over REDRAW_SHARE after that began (its share of the time a quarter at most), and at least every
 * REDRAW_MAX_MS; the lines last drawn stand in meanwhile. A frame left waiting is drawn TRAIL_MS after the wait unless
 * one drew first (a stream's next frame usually does: no frame of Pi's added). */
export const REDRAW_SHARE = 0.25, REDRAW_MAX_MS = 125, TRAIL_MS = 32;
/** SETTLING's [least, most] lines over `n` layers, its sub-heading included: four lines of layers at least (fewer
 * where there are fewer), every layer two to a line at most, and never over SETTLE_MOST lines of layers. */
export function settleRoom(n: number): [number, number] {
	const lines = Math.max(1, Math.ceil(n / 2));
	return [1 + Math.min(4, lines), 1 + Math.min(SETTLE_MOST, lines)];
}
/** SETTLING's most lines of layers (ten half lines: the medium step's 16 J-lens layers on Flash Next a half line or two
 * each); the rest goes to the telescope's rows and MAP. */
export const SETTLE_MOST = 5;
/** LENS's `rows` lines in every view of it: `head` (its title, or the step's line), SETTLING (`settle`) on its `sn`
 * lines, `gap` blank lines (the first `note`, where there is one), then the telescope (`scope`), the lines a short
 * tube leaves blank at the foot: - + only lengthen or shorten the tube. */
function stackLens(head: string[], settle: string[], sn: number, gap: number, scope: string[], rows: number, note = ""): string[] {
	const lines = [...head, ...settle.slice(0, sn), ...Array<string>(Math.max(0, sn - settle.length)).fill(""), ...withNote(gap, note), ...scope].slice(0, rows);
	while (lines.length < rows) lines.push("");
	return lines;
}

/** LENS's link pointer (`note`, "" none) on the first of the `gap` lines: none, no pointer (it gives way first). */
function withNote(gap: number, note: string): string[] {
	const lines = Array<string>(gap).fill("");
	if (gap && note) lines[0] = note;
	return lines;
}


export class Dock {
	// The token in focus: >= 0 a token the keys stepped to (the view stays on it as the reply grows: end or → past the
	// newest follows the reply again); -1 the newest, followed. A reply other than the one it was set in follows (draw).
	private _sel = -1;
	private selRid?: string;
	private selI?: number;   // its token's i (reindex)
	get sel(): number {
		return this._sel;
	}
	set sel(v: number) {
		const r = shownReply(this.host.live);
		this._sel = v;
		this.selRid = v >= 0 ? r?.rid : undefined;
		this.selI = v >= 0 ? r?.tokens[v]?.i : undefined;
	}
	/** A token gone in before the one in focus or the one edited (onEvent: the old stream's, late, while the
	 * subscription changes): their indexes in reply `r` follow their tokens (by i). */
	private reindex(r: Reply | undefined) {
		const at = (k: number, i?: number) => {
			const j = !r || i === undefined || r.tokens[k]?.i === i ? -1 : r.tokens.findIndex((t) => t.i === i);
			return j < 0 ? k : j;
		};
		if (this._sel >= 0) this._sel = at(this._sel, this.selI);
		if (this.edit && this.edit.rid === r?.rid) this.edit.at = at(this.edit.at, this.edit.i);
	}
	// The keyboard is in the dock (stepIn; esc hands it back): the keys and the token in focus are drawn, -1 the newest.
	stepped = false;
	/** ctrl+\ or alt+i: the keyboard to the dock, the token in focus `sel` (-1: the newest, followed); the key line, not
	 * a ? text left from before, so stepping in moves nothing on the board. */
	stepIn(sel: number) {
		this.invalidate();
		this.stepped = true;
		this.sel = sel;
		this.host.captions = false;
	}
	/** The mini dock is drawn: m, or a terminal too short for the full dock (board.ts miniAt). */
	private get mini(): boolean {
		return miniAt(this.host.mini, this.tui.terminal?.rows);
	}
	/** The keys and the token in focus are drawn: stepped in, or a token stepped to (tests set `sel` alone). */
	private get inspecting(): boolean {
		return this.stepped || this._sel >= 0;
	}
	/** The token in focus in a reply of `n` tokens: the one stepped to, else the newest. */
	private cur(n: number): number {
		return this._sel >= 0 ? Math.min(this._sel, n - 1) : n - 1;
	}
	closeOnEsc = false;
	private sizeFailed = new Map<Panel, unknown>();   // this frame's panels whose lines could not be worked out (need): guard draws why
	private boardNote = "";    // one key's answer on the board, where the key line is (an edit's, engram inspection's)
	// LENS shut (dulled) or not where the last frame drew it (undefined: not drawn), this frame's (`lensWas` the last one's).
	private lensWas?: boolean;
	private lensNow?: boolean;
	private mapShown?: [string, number, number];   // the reply and tokens [first, past the last] MAP drew last: the sweep crosses them
	// The reply strip as last drawn: the reply, the token in focus, the width, where its text starts, its first line and
	// its last token shown. Kept while the token in focus stays (in this reply or an edit's answer to it): nothing moves.
	private stripAt?: { rid: string; sel: number; w: number; from: number; start: number; last: number };
	private kept = false;   // this frame keeps the strip's window: MAP keeps its own
	private editWait?: Edited; // a word edit sent with /steer again, until a reply takes it or it is dropped (editTold)
	private editSending = false;   // a word edit's request on its way (host.edit not answered yet)
	// LENS: the eyepiece word edited (lensKey), on token `at` of reply `rid`: dropped on any other (editOn); `from`
	// the id of the lens word the field was filled with (↑/↓), until it is typed over.
	private edit?: { rid: string; at: number; i: number; row: number; field: string | null; fill: number; from?: number };
	private miniN = 1;         // the layers the mini dock last drew at once: [ ] page by that many
	private miniFrom = 0;      // the first of them
	// The lens swap (telescope.ts swapFrames), LENS's - + (stretchFrames) or STREAM's view switch (board.ts streamFrames:
	// frames with `stream`): its frames, the one shown, its timer and token (the switch's: the token stepped to, -1
	// following), and the tube's lines the frames' `sharp` counts out of (SETTLING clears the same share: `clear`). A
	// stretch's frames carry `large`: how far LENS's box is from its panel toward the largest size's (heights).
	private anim?: { frames: AnimFrame[]; i: number; timer: ReturnType<typeof setInterval>; sel: number; nBands: number };
	// LENS's - +: the server's lens steps (host.lensLayers), asked for again as each reply starts and once it is
	// saved, and for another server or served model (`key`), the newest request winning (`seq`); `for` the server and
	// model the answer is from (depthAt: shown only while they are the ones now).
	// The lens steps (LensLayers): `missing` the server has no such route (404: the mock);
	// `posting` a step being set, `queued` the one pressed since (sent when it returns), `reget` a GET asked meanwhile.
	private depth: { data?: LensLayers; error?: string; missing?: boolean; for?: string; key?: string; seq: number; posting?: boolean; queued?: number; reget?: boolean } = { seq: 0 };
	// The idle scene (idle.ts) while there is no reply to show: its frame, its ticker; `idleNow`: this render drew it
	// (render stops the ticker otherwise).
	private idle?: { i: number; loop: Loop };
	private idleNow = false;
	private grids: Grids;
	private timers = new Set<ReturnType<typeof setTimeout>>();   // routing retries, cleared when the dock goes
	private disposed = false;
	// Engram inspection and its sweep (lookup.ts).
	private engram: Engram;
	// EXPERTS (expert.ts): its lights, layer window, pages and their keys, tallies and routing.
	private experts: Experts;
	// The Steer page (steerdock.ts), drawn and keyed while `steerPage`.
	private sp: SteerPage;
	full = false;   // a panel's number on the board (PANEL_KEY): the panel in focus over the whole dock under the reply strip
	// Whether the last frame had room for STREAM: the board (not the one-panel board or the Steer page) tall enough for
	// it in the right column, or STREAM full size; the mini dock tall enough. Nothing to show (no reply yet) keeps the
	// last, so the next reply's first tokens are asked for. The stream asks for the writes only while it holds.
	streamRoom = true;
	// STREAM on the last board frame: its number opens it (streamOpens: not the one-panel board, the model's streams
	// known), with room in the right column (streamCol); on the last frame, drawn on the mini dock (miniStream: s its
	// view). The keys between frames go by them.
	private streamOpens = false;
	private streamCol = false;
	private miniStream = false;
	/** STREAM in the tab cycle (shownPanels): open, or in the right column. */
	private get streamOK() {
		return this.streamOpens && (this.full || this.streamCol);
	}
	// STREAM's history lines in the right column this frame (0: not there; streamLines): its lines there (needOf).
	private streamHistory = 0;
	// { }: the shifts of the divider between EXPERTS and STREAM that change this frame's drawing (none: not both drawn, or none would).
	private splitRange?: [number, number];
	/** host.split within splitRange (0 without one): the divider drawn. */
	private splitNow(): number {
		const [lo, hi] = this.splitRange ?? [0, 0];
		return Math.max(lo, Math.min(hi, this.host.split ?? 0));
	}
	// STREAM's playhead (board.ts pace) while it follows a streaming reply, and its ticker: running while there is a
	// backlog and STREAM is drawn (`playNow`, set by each paint that draws it; paint stops the ticker otherwise).
	private play?: Play;
	private playLoop?: Loop;
	private playNow = false;
	private noteRoom(can: boolean) {
		if (can === this.streamRoom) return;
		this.streamRoom = can;
		this.host.roomChanged?.();
	}
	steerPage = false;   // 6 on the board: the Steer page over the whole dock (6: the board again)
	one = false;    // the last frame was the one-panel board (a dock too small for the board: boardFits)
	private mapFirst = 0;       // MAP on the board: its first line (board.ts mapWindow)
	private sessionFirst = 0;   // MAP full size: its first line of the session's (board.ts sessionMap)
	private theme: DockTheme;   // the dock's own colours (draw.ts dockTheme), not Pi's theme
	// The redraw cap (render): Pi's TUI (`piTui`; the dock asks it to draw through `tui`, which invalidates too), the lines
	// last drawn at that size and when the next draw may be (`next`), whether anything but stream progress changed since
	// (`dirty`: invalidate), and the timer that draws once the wait is over (`trail`).
	private tui: { readonly terminal?: any; requestRender(): void };
	private last?: { lines: string[]; width: number; height: number; next: number };
	private dirty = false;
	private trail?: ReturnType<typeof setTimeout>;
	constructor(private piTui: any, piTheme: any, private host: DockHost) {
		this.tui = { get terminal() { return piTui.terminal; }, requestRender: () => { this.invalidate(); piTui.requestRender(); } };
		this.theme = dockTheme(piTheme);
		// A layer's grid is asked for once while this dock is open (Grids: again after a minute when there was no map);
		// a map rebuilt meanwhile shows the next time the dock opens.
		const get = host.fetchJson ?? ((url: string) =>
			fetch(url, { headers: scopeAuth(url), signal: AbortSignal.timeout(GET_TIMEOUT_MS) }).then((res) => (res.ok ? res.json() : null)));
		this.grids = new Grids(get, () => this.tui.requestRender());
		const dock = this;
		this.engram = new Engram(host, this.theme, this.tui, piTui, {
			get disposed() { return dock.disposed; },
			get sel() { return dock.sel; },
			cur: (n) => this.cur(n),
			get mapShown() { return dock.mapShown; },
			get stripAt() { return dock.stripAt; },
			get boardNote() { return dock.boardNote; },
			set boardNote(v) { dock.boardNote = v; },
		});
		this.experts = new Experts(host, this.theme, this.tui, this.grids, get, this.timers, {
			get disposed() { return dock.disposed; },
			get sel() { return dock.sel; },
			cur: (n) => this.cur(n),
		});
		this.sp = new SteerPage(host, this.theme, this.tui, {
			get disposed() { return dock.disposed; },
			get one() { return dock.one; },
			get boardNote() { return dock.boardNote; },
			set boardNote(v) { dock.boardNote = v; },
			get steerPage() { return dock.steerPage; },
			set steerPage(v) { dock.steerPage = v; },
			served: () => this.served(),
			frame: (body, width, height, footer) => this.frame(body, width, height, footer),
			leave: () => this.leave(),
			setMini: (on) => this.setMini(on),
			boardKey: (data) => this.boardKey(data),
			grids: this.grids,
			xpData: this.experts.xpData,
			xpLast: this.experts.xpLast,
			askDepth: () => this.askDepth(),
		});
		// The concepts are asked for as the dock opens.
		if (host.steering) this.sp.loadSteer();
	}
	/** What the dock draws changed beyond the stream's progress: the next render draws afresh. Pi calls it (a theme);
	 * so does every request of the dock's own to draw (`tui`), every key (handleInput), stepIn, and widget.ts for every
	 * live.requestRender but a token's or the prompt's progress and every status line change (updateUi). Anything missed
	 * still shows within one wait: a frame left waiting always gets its trailing draw. */
	invalidate() {
		this.dirty = true;
	}
	/** Pi calls this when the dock goes: no timer outlives it. */
	dispose() {
		clearTimeout(this.trail);
		this.stopSwap();
		this.stopIdle();
		this.stopPlay();
		this.engram.stopSweep();
		this.engram.stopBar();
		this.disposed = true;
		for (const t of this.timers) clearTimeout(t);
		this.timers.clear();
		// A reply waiting for its routing's next try: the next dock asks.
		for (const r of [this.host.live.reply, this.host.live.prev]) if (r?.routingFrom === "loading" && r.routingTries) r.routingFrom = "again";
	}
	/** The animation running, where it is STREAM's view switch (its frames name a view), or the lens's (the rest: the
	 * view switch draws on STREAM alone). */
	private viewSwitch = () => (this.anim?.frames[0].stream ? this.anim : undefined);
	private lensAnim = () => (this.viewSwitch() ? undefined : this.anim);
	private stopSwap() {
		if (!this.anim) return;
		clearInterval(this.anim.timer);
		this.anim = undefined;
	}
	/** The lens swap on token `t` from lens `from` to `to`, drawn over about 70 ms a frame; the last frame is the new
	 * lens, sharp. */
	private startSwap(t: TokenEvent, from: "jlens" | "logit", to: "jlens" | "logit", sel: number) {
		const nBands = tubeOf(t, this.want()).length;
		this.animate(swapFrames(from, to, nBands), sel, nBands);
	}
	/** `frames` drawn about 70 ms apart on token `sel`; the last is where they end, the timer stopped there. */
	private animate(frames: AnimFrame[], sel: number, nBands = 0) {
		this.stopSwap();
		const anim = { frames, i: 0, sel, nBands, timer: setInterval(() => {
			if (this.anim !== anim) return clearInterval(anim.timer);
			if (++anim.i >= frames.length - 1) this.stopSwap();
			this.tui.requestRender();
		}, 70) };
		this.anim = anim;
	}
	private stopIdle() {
		if (!this.idle) return;
		clearInterval(this.idle.loop.timer);
		this.idle = undefined;
	}
	/** The idle scene's loop from frame 0, a frame every IDLE_MS (loop: hidden, it stops; shown again, it starts over). */
	private startIdle() {
		this.stopIdle();
		const idle: { i: number; loop: Loop } = { i: 0, loop: loop(IDLE_MS, (l) => this.idle?.loop === l, () => this.stopIdle(), () => {
			idle.i = (idle.i + 1) % IDLE_FRAMES;
			this.tui.requestRender();
		}) };
		this.idle = idle;
	}
	/** SETTLING (LENS's upper part) with token `k`'s column in focus: during the lens swap, the frame's lens, fogged as
	 * the telescope is with the same share of it cleared from the deep end. In LENS (`lens`, its frame: lensOpts), over
	 * the step's layers and every line filled; the mini dock's two layers a line. Nothing in fewer than two lines. */
	private settling(toks: TokenEvent[], k: number, w: number, rows: number, inspecting: boolean, lens?: Frame): string[] {
		if (rows < 2) return [];
		const a = this.lensAnim(), f: Frame = a ? a.frames[a.i] : {};
		const clear = a && f.sharp !== undefined ? f.sharp / Math.max(1, a.nBands) : undefined;
		return settlingPanel(this.theme, toks, k, f.view ?? this.host.lensView, w, rows, palette(this.theme), inspecting, { fog: f.fog, clear }, lens?.want, !!lens);
	}
	/** SETTLING's [least, most] lines, its sub-heading included: four lines of layers at least (fewer where the lenses
	 * read fewer), every layer read two to a line at most, by the lens that read more (settleLayers): s swapping the
	 * lens moves nothing, and the lens that read fewer leaves the lines under its rows blank. */
	private settleNeed(toks: TokenEvent[], k: number): [number, number] {
		return settleRoom(Math.max(...(["jlens", "logit"] as const).map((v) => settleLayers(toks, k, v).length)));
	}
	/** A key: never throws (a throw out of handleInput would end Pi); what went wrong is a note where the user is: the
	 * Steer page's, else the key line's (board and mini dock). */
	handleInput(data: string) {
		this.invalidate();
		// Any key ends engram inspection's sweep at once (g on again starts another); g during the closing one: closed. An
		// edit's runs on until its answer.
		const closing = this.engram.sweep?.close;
		if (!this.engram.sweep?.edit) this.engram.stopSweep();
		if (closing && data === "g") return void this.tui.requestRender();
		try {
			this.reindex(shownReply(this.host.live));
			if (this.sp.steer.del && data !== "y") this.sp.disarm();     // a delete asked for: any other key calls it off
			if (matchesKey(data, "ctrl+c")) return this.passOut(data);
			if (this.mini) return this.miniKey(data);
			this.boardKey(data);
		} catch (e: any) {
			const text = failed(e);
			if (this.steerPage && !this.mini) this.sp.steer.note = { text, level: "error" };
			else this.boardNote = text;
			this.tui.requestRender();
		}
	}
	/** ctrl+c is Pi's (clear, twice: exit): the keyboard goes back as esc gives it, and the key on to what has it, as Pi's
	 * TUI says (not `tui`, the dock's own). */
	private passOut(data: string) {
		this.leave();
		const to = this.piTui.getFocusedComponent?.();
		if (to && to !== this) to.handleInput?.(data);
		this.tui.requestRender();
	}
	/** esc (q, ctrl+c): following the newest again, the keyboard handed back, the page shown kept; the dock closed when it
	 * opened for the keys (closeOnEsc). Whether it closed. */
	private leave(): boolean {
		const close = this.closeOnEsc;
		this.sel = -1;
		this.stepped = false;
		if (close) this.host.close();
		else this.host.unfocus();
		return close;
	}
	/** The token keys: ←/→ (h/l) a token, shift ten, alt or cmd (super) or H/L a sentence, page keys 25, u/U the next or
	 * previous unsure token, home; from the token in focus (the newest while following), the view then stays on the one
	 * stepped to. end, or → on the newest, follows the newest again (-1). Whether `data` was one. */
	private moveToken(data: string, r: Reply): boolean {
		const n = r.tokens.length, at = this.cur(n);
		// Faster moves: shift+arrows 10 tokens, alt or cmd (super) + arrows a sentence, page keys 25 tokens.
		const fast = (dir: 1 | -1) => {
			if (matchesKey(data, dir > 0 ? "shift+right" : "shift+left")) return jumpTo(r.tokens, at, dir, "step", 10);
			if (matchesKey(data, dir > 0 ? "pageDown" : "pageUp")) return jumpTo(r.tokens, at, dir, "step", 25);
			if (matchesKey(data, dir > 0 ? "alt+right" : "alt+left") || matchesKey(data, dir > 0 ? "super+right" : "super+left") ||
				data === (dir > 0 ? "L" : "H")) return jumpTo(r.tokens, at, dir, "sentence");
			return null;
		};
		const jumped = fast(1) ?? fast(-1);
		if (jumped !== null) {
			if (this._sel >= 0 || jumped < at) this.sel = jumped;   // following: a jump forward keeps following
		} else if (matchesKey(data, "right") || data === "l") this.sel = at >= n - 1 ? -1 : at + 1;
		else if (matchesKey(data, "left") || data === "h") this.sel = Math.max(0, at - 1);
		else if (data === "u" || data === "U") {
			const dir = data === "u" ? 1 : -1;
			for (let k = at + dir; k >= 0 && k < n; k += dir) {
				if ((r.tokens[k].p ?? 0) < UNSURE) {
					this.sel = k;
					break;
				}
			}
		} else if (matchesKey(data, "home")) this.sel = 0;
		else if (data === "f" || matchesKey(data, "end")) this.sel = -1;   // f: follow the newest again (end too)
		else return false;
		return true;
	}
	/** The dock's lines: the board where it fits (boardFits), else the page shown. Never throws: a bug draws one line.
	 * Pi asks every frame; with a clock (host.now) the lines last drawn stand in while only the stream's progress is
	 * new and the redraw cap says wait (REDRAW_SHARE), a draw then due once the wait is over (`trail`, TRAIL_MS). */
	render(width: number): string[] {
		const height = Math.max(12, this.tui.terminal?.rows ?? 40);   // the terminal's whole height: Pi's content, editor included, is narrowed beside the dock (widget.ts reserveColumns)
		const now = this.host.now?.(), last = this.last;
		if (this.engram.sweep?.loop) this.engram.sweep.loop.drawn = true;
		if (this.disposed && last) return last.lines;   // gone: no timer armed
		// A token after STREAM's playhead caught up, STREAM drawn last time: the ticker starts now, the cap or not.
		const p = this.play, r = p && !this.playLoop && this.playNow && now !== undefined ? shownReply(this.host.live) : undefined;
		if (r && r.rid === p!.rid && r.tokens.length - 1 > p!.newest) {
			this.play = pace(p, r.rid, r.tokens.length - 1, r.done, now!);
			this.startPlay();
		}
		if (now !== undefined && last && !this.dirty && last.width === width && last.height === height && now < last.next) {
			this.trail ??= setTimeout(() => {
				this.trail = undefined;
				this.piTui.requestRender();
			}, last.next - now + TRAIL_MS);
			return last.lines;
		}
		clearTimeout(this.trail);
		this.trail = undefined;
		this.dirty = false;
		const lines = this.paint(width, height);
		if (now !== undefined) this.last = { lines, width, height, next: now + Math.min(REDRAW_MAX_MS, (this.host.now!() - now) / REDRAW_SHARE) };
		return lines;
	}
	private paint(width: number, height: number): string[] {
		this.idleNow = this.engram.barNow = this.playNow = false;
		this.lensWas = this.lensNow;
		this.lensNow = undefined;
		this.sizeFailed.clear();
		try {
			this.editTold(shownReply(this.host.live));
			this.engram.sw = this.engram.sweepNow(shownReply(this.host.live));
			return this.draw(width, height);
		} catch (e: any) {
			return this.frame([this.theme.fg("error", cut(failed(e), Math.max(1, width - 3)))], width, height, []);
		} finally {
			if (!this.idleNow) this.stopIdle();   // anything else drawn: the idle scene's timer stops
			if (!this.playNow) this.stopPlay(), (this.play = undefined);   // STREAM not drawn: its playhead's too
			if (!this.engram.barNow) this.engram.stopBar();     // and the bar's, once it is not drawn
		}
	}
	/** The idle scene (idle.ts) under the header and its rule, centred in the room over the key line, `caption` dim
	 * under it (the least dock holds it). Its timer runs while it is drawn. */
	private renderIdle(width: number, w: number, height: number, caption: string, footer: string[]): string[] {
		const th = this.theme, room = height - 2 - footer.length;
		if (!this.idle) this.startIdle();
		const idle = this.idle!;
		idle.loop.drawn = this.idleNow = true;
		const art = idleLines(idle.i, colourMode(th)).map((s) => centreLeft(s, w));
		const above = Array<string>(Math.floor((room - IDLE_LINES - 1) / 2)).fill("");
		return this.frame([th.fg("accent", "◆ trace") + this.untested(), ruleLine(th, w), ...above, ...art, th.fg("dim", centreLeft(cut(caption, w), w))], width, height, footer);
	}
	private draw(width: number, height: number): string[] {
		const th = this.theme;
		const { live } = this.host;
		// Another served model: its concepts asked for whatever is shown.
		if (this.host.steering && this.served() !== this.sp.steer.model) this.sp.loadSteer();
		const w = width - 3; // the edge and a space (frame) + 1 right pad
		// The mini dock where chosen or the full one is too short (it draws from the dock's least 12 rows); too narrow: one line.
		this.miniStream = false;
		if (this.mini) return this.renderMini(width, height);
		if (width < DOCK_MIN_WIDTH) return this.frame([th.fg("dim", cut(`dock too small: needs ${DOCK_MIN_WIDTH}x${DOCK_MIN_ROWS}`, Math.max(1, w)))], width, height, []);
		// Too small for the board: the one-panel board (a full-size panel kept for the board again).
		this.one = !boardFits(width, height);
		const keys = (text: string) => wrapTextWithAnsi(th.fg("dim", text), w);
		// The status is the request being served (live.reply); the body, the reply shown (the last one while the next
		// reads its prompt).
		const now = live.reply, r = shownReply(live);
		const waiting = !now ? "waiting for a reply" : now.prefill ? `reading the prompt · ${now.prefill} tok` : "reading the prompt";
		const status = !live.connected ? th.fg("error", `not answering at ${hostOf(live.base)}`) : !now ? th.fg("muted", `${live.level} · ${waiting}`)
			: !now.tokens.length ? th.fg("muted", waiting + (r !== now ? " · the last reply below" : ""))
			: now.done ? th.fg("muted", `done · ${now.tokens.length} tok`) : th.fg("success", `writing · ${now.tokens.length} tok`);
		if (!r || !r.tokens.length) {
			// Nothing to show yet: the Steer page needs no reply (6, stepped in). Else, the server answering and tracing
			// on: the idle scene, the status under it (not in the header); off, or not answering: the status in the header
			// and what fills the dock. Stepped in, the keys that work without a reply.
			if (this.steerPage) return this.noteRoom(false), this.sp.renderSteerPage([this.head(w, status), ruleLine(th, w)], width, w, height, undefined, this.inspecting);
			const footer = keys(this.inspecting ? `${STEER_KEY} steer · m mini · esc back` : `ctrl+\\ steps in: ${STEER_KEY} steer · /scope help dock`);
			if (live.connected && live.level === "on") return this.renderIdle(width, w, height, waiting, footer);
			return this.frame([`${th.fg("accent", "◆ trace")}${this.untested()}  ${status}`, "", ...keys("Fills in as a traced reply streams.")], width, height, footer);
		}
		const toks = r.tokens;
		if (this._sel >= 0 && this.selRid !== undefined && r.rid !== this.selRid) {
			// A new reply: followed; an edit's answer to the reply stepped in, from the token in focus on: it stays there.
			if (joined(r, this.selRid) && this._sel <= (r.from ?? -1)) this.selRid = r.rid;
			else this.sel = -1;
		}
		this.reindex(r);
		const inspecting = this.inspecting, sel = this.cur(toks.length);
		this.editOn(r, sel);
		if (this.edit && this.lensShut(r)) this.edit = undefined;   // LENS shut (another client's off): no word to edit
		// The board's header is ◆ metalscope alone (its counts are MAP's), unless the server is not answering or the
		// next request is still reading its prompt.
		return this.renderBoard(width, w, height, r, sel, inspecting, live.connected && (now?.tokens.length || now?.of !== undefined) ? "" : status);   // an edit's answer: nothing said
	}
	/** The step in use (LensLayers), for a reply of the model it is for, while the server's answer stands (none yet, a
	 * failure, a server without the route: undefined); the lenses off, the fewest (LENS keeps its room: + opens it there). */
	private stepNow(r?: Reply): LensLayers["steps"][number] | undefined {
		const st = this.depthNow(), d = st.data;
		return d && !st.error && !st.missing && (!r?.model || r.model.split("/").pop() === d.model) ? d.steps[Math.max(0, d.at)] : undefined;
	}
	/** The lenses off (- past the fewest layers: the server's step LENS_OFF), for reply `r`'s model. */
	private lensOff = (r?: Reply) => this.stepNow(r) !== undefined && this.depthNow().data!.at === LENS_OFF;
	/** Why LENS is shut on reply `r`: the lenses "off", or "none" read for it (traced with them off); "" open. */
	private unread = (r: Reply) => !r.tokens.some((t) => t.lens?.length || t.lens_logit?.length);
	private shutWhy = (r: Reply): "" | "off" | "none" => (this.lensOff(r) ? "off" : this.unread(r) ? "none" : "");
	private lensShut = (r: Reply) => this.shutWhy(r) !== "";
	/** LENS dulled on reply `r`: the lenses off, or none read for it and no step's outline to draw (no steps); read with
	 * them off and the lenses on, its outline lit, so + shows. */
	private lensDull = (r: Reply) => this.lensOff(r) || (this.lensShut(r) && !this.want(r));
	/** LENS drawn this frame, shut or not (`shut`), its sweep `ticks` long (board.ts dullPace): a change since the last
	 * frame drawn starts the sweep (a tick each SWEEP_TICK_MS, on the sweep's ticker; not over a word edit's), but not
	 * the lens steps' first answer. The ticks into going off it is now (`ticks`: off). */
	private dulled(shut: boolean, ticks: number): number {
		const was = this.lensWas;
		this.lensNow = !this.host.lensLayers || this.depth.for === this.depthAt() ? shut : undefined;
		if (was !== undefined && was !== shut && !this.engram.sweep?.edit) this.engram.startSweep("", false, undefined, { off: shut, ms: ticks * SWEEP_TICK_MS });
		const s = this.engram.sweep, now = this.host.now?.(), p = s?.lens && now !== undefined ? this.engram.sweepPhase(s, now) : "done";
		if (typeof p === "string") {
			if (s?.lens) this.engram.stopSweep();   // over: LENS as it is
			return shut ? ticks : 0;
		}
		const k = Math.min(ticks, Math.floor(p.frac * ticks) + 1);
		return s!.lens!.off ? k : ticks - k;
	}
	/** LENS's lines, the first its title (or the step's line), drawn as ever; dull (lensDull), the rest dulled where
	 * they are (board.ts dullLines): one edge, going off up from the bottom, back on down from the top. */
	private lensLines(r: Reply, open: () => string[]): string[] {
		const lines = open(), d = this.dulled(this.lensDull(r), dullPace(lines.length - 1).ticks);
		return d ? dullLines(this.theme, lines, d) : lines;
	}
	/** The server and the model served now: the lens steps' answer is for one pair. */
	private depthAt = () => `${this.host.live.base}|${this.served()}`;
	/** The lens steps' answer for the server and model now (depthAt); nothing from another (a switch since it came). */
	private depthNow(): { data?: LensLayers; error?: string; missing?: boolean } {
		return this.depth.for === this.depthAt() ? this.depth : {};
	}
	/** The tube's layers: the step in use's (stepNow), whatever the reply shown read (the layers it did not read faint),
	 * so the tube stays where - + put it while the next prompt is read. */
	private want(r = shownReply(this.host.live)): number[] | undefined {
		return this.stepNow(r)?.layers;
	}
	/** Whether - + and their key-line hint are on: the server's steps in hand for reply `r` (stepNow). */
	private stepsShown(r?: Reply): boolean {
		return !!this.host.lensLayers && this.stepNow(r) !== undefined;
	}
	/** How every view of the lens (the board's LENS, LENS full size) draws token `k` of reply `r`: the
	 * animation's frame (a lens swap, a step's stretch), the step's tube (want), the marks, the word edited; the server's
	 * lens steps asked for. */
	private lensOpts(r: Reply, k: number): Frame {
		if (this.host.lensLayers) this.loadDepth(r);
		const a = this.lensAnim(), frame = a ? a.frames[a.i] : {};
		return { ...frame, view: frame.view ?? this.host.lensView, focus: this.host.lensLayer, ...this.marks(), edit: this.editFrame(r),
			input: inputOf(r.tokens, k), want: frame.want ?? this.want(r) };
	}
	/** The steps LENS is sized on, whatever the step in use: the middle one (the default) and the largest; undefined
	 * without the server's steps for reply `r` (stepNow). */
	private sizing(r?: Reply): { mid: LensLayers["steps"][number]; max: LensLayers["steps"][number] } | undefined {
		const st = this.stepNow(r) && this.depthNow().data!.steps;
		return st ? { mid: st[st.length >> 1], max: st[st.length - 1] } : undefined;
	}
	/** - or + (= and _ too), in any view of the lens: the step one fewer or more, on token `k`'s tube. A word being edited
	 * types them (lensKey first). Whether it took the key. */
	private depthKey(data: string, r: Reply, k: number): boolean {
		if (!this.stepsShown(r) || this.edit || data.length !== 1 || !"+=-_".includes(data) || !r.tokens.length) return false;
		this.stepDepth(data === "+" || data === "=" ? 1 : -1, k);
		return true;
	}
	/** The lens steps asked for again for another server or served model, as a reply starts (a failure retried) and once
	 * it is saved: now, and (a reply done) once more TALLY_LAG_MS later, when the server has its speed. */
	private loadDepth(r: Reply) {
		const key = `${this.depthAt()}|${r.rid}|${r.done ? r.traceId ?? "" : ""}`;
		if (key === this.depth.key) return;
		this.depth.key = key;
		this.askDepth();
		if (!r.done) return;
		const timer = setTimeout(() => {
			this.timers.delete(timer);
			if (this.depth.key === key) this.askDepth();
		}, TALLY_LAG_MS);
		this.timers.add(timer);
	}
	/** The server's lens steps (`step`: set the one in use first), to `done`; a failure, or an answer that is not
	 * LensLayers, as { error }. */
	private callDepth(step: number | undefined, done: (res: LensLayers | { error: string; missing?: boolean }) => void) {
		const fail = (e: any) => done({ error: String(e?.message ?? e) });
		const took = (res: any) => done(typeof res?.error === "string" || isLensLayers(res) ? res : { error: LABELS.layersBad });
		try {
			this.host.lensLayers?.(step).then(took, fail);
		} catch (e: any) {
			fail(e);
		}
	}
	/** An answer for `at` (depthAt when it was asked). */
	private tookDepth(res: LensLayers | { error: string; missing?: boolean }, at: string) {
		const st = this.depth;
		if ("error" in res) Object.assign(st, { error: res.error, missing: !!res.missing, for: at });
		else Object.assign(st, { data: res, error: undefined, missing: false, for: at });
		this.tui.requestRender();
	}
	/** GET the lens steps; the newest answer kept. While a step is being set, asked once it is. */
	private askDepth() {
		const st = this.depth;
		if (st.posting) return void (st.reget = true);
		const seq = ++st.seq, at = this.depthAt();
		this.callDepth(undefined, (res) => {
			if (seq === st.seq && !this.disposed && !st.posting) this.tookDepth(res, at);
		});
	}
	/** Set step `step` on the server: one request at a time, the step pressed last sent when it returns; a failure
	 * asks again what the server holds. */
	private setDepth(step: number) {
		const st = this.depth;
		if (st.posting) return void (st.queued = step);
		st.posting = true;
		++st.seq;                                         // a GET in flight: overtaken
		const at = this.depthAt();
		this.callDepth(step, (res) => {
			const next = st.queued, reget = st.reget;
			Object.assign(st, { posting: false, queued: undefined, reget: false });
			if (this.disposed) return;
			if ("error" in res) return void (this.boardNote = res.error), this.askDepth();   // refused: why, on the key line
			if (next !== undefined && next !== res.at) return this.setDepth(next);
			this.tookDepth(res, at);
			if (reget) this.askDepth();
		});
	}
	/** - or +: the step one fewer or more, past the fewest the lenses off (LENS_OFF), drawn at once (the tube on token
	 * `sel` retracting or extending), then set on the server; to or from off, LENS dulled or lit (lensLines) once the
	 * server has it. */
	private stepDepth(dir: 1 | -1, sel: number) {
		const { data: d, missing } = this.depthNow(), from = this.want(), was = d?.at;
		if (!d || missing) return;
		const at = Math.max(LENS_OFF, Math.min(d.steps.length - 1, d.at + dir));
		if (at === d.at) return;
		if (at === LENS_OFF || was === LENS_OFF) return this.setDepth(at);
		d.at = at;
		// The box with the frames too, between the panel and the largest size's (to or from the largest: MAP's lines).
		const a = +isLargest(d, was ?? 0), b = +isLargest(d, at);
		if (from) this.animate(stretchFrames(from, d.steps[at].layers).map((f, i, fs) => ({ ...f, large: a + ((b - a) * (i + 1)) / fs.length })), sel);
		this.setDepth(at);
	}
	/** The step's line for reply `r` (LENS full size): a pip a step (lit up to the one in use), its name and the layers
	 * it reads, and on the right the decode speed of the model's last replies read at it on this server; until known, or
	 * failed, one dim line; blank for a server without the route or a reply of another model. */
	private depthLine(w: number, r: Reply): string {
		const th = this.theme, { data: d, error, missing } = this.depthNow(), s = this.stepNow(r);
		if (missing) return "";                           // a server without the route (the mock): nothing
		if (error || !d) return th.fg(error ? "warning" : "dim", cut(error ?? LABELS.asking, w));
		if (!s) return "";
		if (d.at === LENS_OFF) return this.offTag(d);
		const left = this.pips(d) + th.fg("dim", `  ${LABELS.reads(s.name, s.n)}`);
		const room = w - visibleWidth(left) - 2;
		if (room < 12) return left;
		const right = cut(s.tps === null ? LABELS.noSpeed : LABELS.speed(Math.round(s.tps), s.replies), room);
		return left + " ".repeat(w - visibleWidth(left) - visibleWidth(right)) + th.fg("dim", right);
	}
	/** The lenses off: the pips unlit and the key that opens LENS. */
	private offTag = (d: LensLayers) => this.pips(d) + this.theme.fg("dim", `  ${LABELS.lensOff}`);
	/** A pip a step, lit up to the one in use. */
	private pips(d: LensLayers): string {
		return d.steps.map((_, i) => this.theme.fg(i <= d.at ? "lens" : "faint", i <= d.at ? "▮" : "▯")).join("");
	}
	/** The step for LENS's title line in `w` columns, for reply `r`: the pips, its name and how many layers it reads,
	 * shorter where narrow ("medium 16"); null where none of those fits (lensHead shortens the switch and asks again,
	 * `short`: the name goes, then the count); a failure, LABELS.layersFailed (its reason is LENS full size's step
	 * line); "" until known, without the route, for a reply of another model or where it does not fit. */
	private depthTag(w: number, r: Reply, short = false): string | null {
		const th = this.theme, { data: d, error, missing } = this.depthNow(), s = this.stepNow(r);
		if (missing) return "";
		if (error) return visibleWidth(LABELS.layersFailed) <= w ? th.fg("warning", LABELS.layersFailed) : "";
		if (!d || !s) return "";
		if (d.at === LENS_OFF) return w < 5 ? "" : visibleWidth(this.offTag(d)) <= w ? this.offTag(d) : this.pips(d);
		const forms = short ? [` ${s.name} ${s.n}`, ` ${s.n}`] : [`  ${LABELS.reads(s.name, s.n)}`, ` ${LABELS.layers(s.name, s.n)}`, ` ${s.name} ${s.n}`];
		return forms.map((x) => this.pips(d) + th.fg("dim", x)).find((x) => visibleWidth(x) <= w) ?? (!short ? null : w < 5 ? "" : this.pips(d));
	}
	private served = () => this.host.steerModel?.() ?? "";
	/** The header's dim "untested" for a model metalscope was not run on here (host.flags); "" otherwise. */
	private untested = () => (this.host.flags?.().tested === false ? "  " + this.theme.fg("dim", "untested") : "");
	/** The LENS word edit offered: not where the server refuses this model one (host.flags edit false). */
	private canEdit = () => this.host.flags?.().edit !== false;
	/** A ? text without its edit hint where the edit is not offered. */
	private editHint = (s: string) => (this.canEdit() ? s : s.replace(EDIT_HINT, ""));
	/** LENS's link pointer for token `t` of reply `r`: where the Steer page offers the link (offerShown), a reply of the
	 * served model read with the logit lens alone, and a candidate has a J-lens; the longest that fits `w`, else "". */
	private lensPointer(r: Reply, t: TokenEvent | undefined, w: number): string {
		const o = this.sp.offerShown();
		if (!t || !o || (r.model && r.model.split("/").pop() !== o.model)) return "";
		if (t.lens_by !== "logit" || t.lens_logit?.length || this.lensShut(r) || !o.candidates.some((c) => c.lenses)) return "";
		const said = LENS_LINK.find((s) => visibleWidth(s) <= w);
		return said ? this.theme.fg("dim", said) : "";
	}
	/** Enter on a LENS edit (`word` removed, or swapped for `want`; `name`, `into` their vector names, `ids` those sent
	 * by token: lensKey) of token `t` of reply `r`: the reply answered again
	 * from that token, the edit there only (host.edit); the newest reply only (the last message is the one answered
	 * again). Its note on the key line; what came of it once the edited reply has the token (editTold). */
	private editAt(r: Reply, t: TokenEvent, word: string, name: string, ids: WordIds, want?: string, into?: string) {
		if (!this.host.edit || this.editBusy()) return;
		if (r.rid !== this.host.live.reply?.rid) return void (this.boardNote = LABELS.editNewest);
		this.boardNote = LABELS.editing;
		// The token stays in focus (paused there: nothing moves as the answer comes) and the word burns out, the tail after
		// it swept away, while the request goes (sweepNow).
		if (this._sel < 0) this.sel = r.tokens.indexOf(t);
		this.engram.startSweep("", false, { old: r, at: t.i, want });
		const args = want ? `${name}>${into}@${EDIT_SWAP_DOSE}` : `-${name}`;
		this.editSending = true;
		const done = (said: { text: string; level: string }[]) => {
			this.editSending = false;
			const bad = said.find((s) => s.level === "error");
			if (!bad) this.editWait = this.host.live.edit;
			else if (this.boardNote === LABELS.editing) this.boardNote = bad.text;
			if (bad && this.engram.sweep?.edit) this.engram.stopSweep();
			this.tui.requestRender();
		};
		this.host.edit(args, ids, { rid: r.rid, at: t.i, word, ...(want ? { want } : {}) })
			.then(done, (e: any) => done([{ text: String(e?.message ?? e), level: "error" }]));
	}
	/** A word edit is under way: sent, or animated until its answer's first token. enter does not open another, g waits. */
	private editBusy = () => this.editSending || !!this.editWait || !!this.engram.sweep?.edit;
	/** What came of the word edit the shown reply answers (Reply.edit), said on the key line once it has the token:
	 * whether the model's top 1 there changed and, for a swap, to the word asked for (LABELS.edited); again at the end
	 * when the reply stopped at a tool call. An edit sent that no reply took (its run over, or dropped): not edited. */
	private editTold(r: Reply | undefined) {
		const w = this.editWait;
		if (w && (w.answered || w.over || this.host.live.edit !== w)) {
			this.editWait = undefined;
			if (!w.answered && this.boardNote === LABELS.editing) this.boardNote = LABELS.editNot;
			if (!w.answered && this.engram.sweep?.edit) this.engram.stopSweep();
		}
		const e = r?.edit;
		if (!r || !e || (e.told && (!r.done || e.toldEnd === r.rid))) return;
		const now = r.tokens.find((t) => t.i === e.at);
		if (!now && !r.done) return;
		const tool = r.done && TOOL_OPENERS.some((o) => r.tokens.filter((t) => t.i > e.at).map((t) => t.raw ?? t.text).join("").includes(o));
		const again = !!e.told;
		e.told = true;
		if (r.done) e.toldEnd = r.rid;
		if (again && !tool) return;
		const was = e.was ? label(e.was.text) : "it";
		const changed = editChanged(e, now);
		this.boardNote = LABELS.edited(was, now ? label(now.text) : null, changed, e.want === undefined ? undefined : label(e.want),
			!!now && e.want !== undefined && now.text.trim().toLowerCase() === e.want.toLowerCase(), tool);
	}
	// -- the board (board.ts): everything on one screen, or the panel in focus alone in a smaller dock ----------------

	/** The board: header, the reply strip (taller where both columns spare lines), then TOKENS / MAP / LENS (SETTLING in
	 * it) beside EXPERTS / STREAM (streamHistory; without experts MAP / LENS beside TOKENS / STREAM: board.ts arrange),
	 * or the panel in focus full size (its number); the key line (or,
	 * with ?, what the panel in focus shows). The one-panel board (`one`, a dock too small for the board): a short
	 * reply strip, the panel in focus (onePanel), the keys on up to two lines. 6: the Steer page instead, under the
	 * header (renderSteerPage). */
	private renderBoard(width: number, w: number, height: number, r: Reply, sel: number, inspecting: boolean, status: string): string[] {
		const th = this.theme, h = this.host, live = h.live, toks = r.tokens;
		this.experts.stack = undefined;
		// STREAM: a model's writes or reads known (streamData), not the one-panel board, and EXPERTS over it left its
		// least (rightLeast; ? takes no more than leaves it its least), or a panel full size. More layers than the
		// width: spread over it (board.ts streamPanel).
		const one = this.one, strip = 1 + (one ? ONE_REPLY_ROWS : REPLY_ROWS) + 1;   // the reply strip at its least: its title, its lines, the rule under it
		const experts = !this.noExperts(r), over = this.rightLeast(r, experts), s = this.streamData(r);
		// The middle's lines under a footer of `foot` lines: the dock less the header and its rule (2), the reply strip at
		// its least, the footer's rule (1) and the footer; STREAM's history lines in it (streamLines, past `over`: a
		// quarter of the middle, streamShare; more where its column leaves lines over, heights).
		const middle = (foot: number) => height - 2 - strip - 1 - foot, history = (foot: number) => streamLines(middle(foot) - over, STREAM_FRAME, streamShare(middle(foot)));
		const colRoom = history(1);   // under the key line alone
		this.streamOpens = !one && !!s?.L;
		this.streamCol = this.streamOpens && colRoom > 0;   // its room kept only where the column draws it
		const shownP = this.shownPanels();
		// The focus drawn (focusNow): EXPERTS' on a model without experts moves for good, STREAM's not drawn this frame
		// only for the frame.
		const focus = this.focusNow(shownP);
		if (h.panel === "EXPERTS" && !experts) h.panel = focus;
		// Full size on a panel not the one drawn: off (STREAM's room asked for as before this frame's reset).
		const streamFull = this.full && h.panel === "STREAM";
		if (this.full && focus !== (h.panel ?? "LENS")) this.full = false;
		// A token stepped to: said, as the view no longer follows the newest (end does again).
		const out = [this.head(w, status, [this._sel >= 0 ? th.fg("warning", `paused at #${sel}`) : "",
			r.capped !== undefined ? th.fg("dim", `trace stops at #${r.capped}`) : ""].filter(Boolean).join("  "))];
		this.splitRange = undefined;   // set again under the reply strip where the board draws EXPERTS and STREAM
		if (this.steerPage) {
			this.noteRoom(false);
			out.push(ruleLine(th, w));
			return this.sp.renderSteerPage(out, width, w, height, r, inspecting);
		}
		// Asked for where STREAM is drawn, or could be (the model's layers not known: one reply settles it).
		this.noteRoom(!one && (!s || colRoom > 0 || streamFull));
		// The rule under the header: engram inspection's legend in it while that is drawn.
		out.push(this.engram.legend(r, w) || ruleLine(th, w));
		// The key line, or (stepped in, after ?) what the panel in focus shows.
		// The explanation takes at most 6 lines, and never the middle's minimum rows (MIDDLE_MIN) under the reply strip at
		// its least: on a short dock it is cut, the cut marked.
		// The EXPERTS window (as many layers as fit) is drawn where EXPERTS is full size (expertsWindow): its keys then
		// page by that many. Known before drawing it: the layout decides.
		const windowed = this.isWindowed(focus);
		const fullNow = this.full || one;
		if (!fullNow) this.experts.xp.at = "grid";   // EXPERTS in its column: its pages closed, not opened again unasked
		const page = focus === "EXPERTS" && fullNow && this.experts.xp.at !== "grid" && !this.experts.expertsView(r).reason ? this.experts.xp.at : undefined;
		const to = LABELS.source[h.expertSource === "session" ? "all" : "session"];
		// Both sources' tokens where the lists are wide enough (the grids: full size, else EXPERTS' column); else a switches.
		this.experts.xpBoth = page ? layerBoth(w) : bothSources(fullNow ? w : columns(w).rw);
		const src = this.experts.xpBoth ? "" : to;
		const g = this.engram.gKey(r), gs = g === LABELS.engramKey ? LABELS.engramShort : g;   // the EXPERTS pages' lines are long: g short
		const k = this.lensToken(toks, sel, this._sel >= 0);   // following: the newest token with a lens read
		// The middle under a footer of `F` lines, all from its height: STREAM's history lines (`base`: last in the right
		// column, under EXPERTS, its least; none where a panel is full size), from what the column has past EXPERTS' least
		// (rightLeast, streamLines), before the reply strip takes any; the columns; the reply strip: REPLY_ROWS lines and
		// the lines both columns spare (what their panels' content does not need), up to a third of the dock (board.ts
		// replyRows), none of it following the reply's length (full size: its least); the middle's `rows`, under the
		// reply's rule (the header, its rule, REPLY, the strip, the rule) over the footer's rule. And { }'s `range`
		// (host.split of EXPERTS' layers, STREAM's history longer; EXPERTS keeps a layer, STREAM STREAM_MIN lines): STREAM
		// takes the column's rest, so only EXPERTS' lines change the drawing, and the range ends where they stop changing
		// (EXPERTS at its most, or at one layer); none where they never do.
		const layout = (F: number) => {
			const base = this.streamCol && !fullNow ? history(F) : 0, cols = arrange(experts, base > 0);
			this.streamHistory = base;
			const replyN = one ? ONE_REPLY_ROWS : fullNow ? REPLY_ROWS : replyRows(height, ...this.spare(cols, w, middle(F), r, k));
			const rows = height - 4 - replyN - 1 - F, ei = cols.right.indexOf("EXPERTS");
			let range: [number, number] | undefined;
			if (experts && base > 0 && !fullNow && ei >= 0) {
				const block = over - 1, rw = columns(w).rw;
				const lo = Math.min(0, Math.ceil((STREAM_MIN - base) / block)), hi = Math.max(0, Math.floor((rows - over - STREAM_FRAME - base) / block));
				const lines = (s: number) => ((this.streamHistory = base + s * block), this.heights(cols.right, rw, rows, r, k).h[ei]);
				let a = lo, b = hi;
				while (a < b && lines(a + 1) === lines(lo)) a++;
				while (b > a && lines(b - 1) === lines(hi)) b--;
				if (a < b) range = [a, b];
				this.streamHistory = base;
			}
			return { base, cols, replyN, rows, range };
		};
		// The key line; the one-panel board's always two lines (wrapKeys; the EXPERTS pages' up to three, they cut none),
		// so nothing moves as keys come and go. A word edited, or a note, in the keys' place on as many lines. { } where a
		// press changes this frame's drawing: the range under a footer of the key line's height.
		const keyRows = one ? 2 : 1, open = this.openable(r), nums = panelNums(open);
		let at = layout(keyRows), split = !!at.range;
		const keyLines = (text: string, sep?: string, most = keyRows) => (one ? wrapKeys(text, w, sep).slice(0, most) : [text]).map((l) => th.fg("dim", fit(l, w)));
		const keys = padAbove(keyRows, this.edit ? keyLines(editKeys(w, keyRows))
			: page ? keyLines(page === "layer" ? layerKeys(src, this.experts.xpLists(r, sel).e, gs) : expertKeys(to, gs), "  ", 3)
			: keyLines(boardKeys(focus, inspecting, windowed, shownP, this._sel >= 0, !!live.steer, this.stepsShown(r), this.canEdit(),
				windowed && !this.experts.expertsView(r).reason ? src : "", w, this.engram.gKey(r), one, this.shutWhy(r), bothLenses(toks[this.lensToken(toks, sel, this._sel >= 0)] ?? {}),
				this.streamViews(r).length > 1, open, split)));
		let footer = this.boardNote ? padAbove(keys.length, noteLines(th, this.boardNote, w, keys.length)) : keys;
		const captions = (split: boolean) => {
			// The panel's explanation, then (room to spare) the layout's keys under it; a word edited, its keys there
			// instead (the explanation's lines kept: nothing above moves). The board's at most 6 lines, never the middle's
			// least; the one-panel board's from the panel's room (its tab row and three lines kept).
			// The one-panel board's two lines either way, so an edit moves nothing.
			const tail = padAbove(keyRows, this.edit ? keys.filter(Boolean) : wrapKeys(layoutKeys(one ? "" : nums), w).map((l) => th.fg("faint", fit(l, w))));
			const roomFor = (least: number, spare = 1) => Math.max(1, Math.min(one ? 6 + tail.length : 6, height - out.length - strip - 1 - (one ? 4 : least) - spare));
			// STREAM in the right column (no panel full size) keeps its least under the column's (exact: streamLines,
			// no line spare past it); where that leaves no line for the layout's keys, it gives way (not in focus).
			const col = this.streamCol && !fullNow, least = over + STREAM_FRAME + STREAM_MIN;
			let room = col && least > MIDDLE_MIN ? roomFor(least, 0) : roomFor(MIDDLE_MIN);
			if (col && focus !== "STREAM" && room <= tail.length) room = roomFor(MIDDLE_MIN);
			const keysToo = room > tail.length, n = keysToo ? room - tail.length : room;
			// The full-size EXPERTS' pages: the counting rule, the expert page what its rank means (the grids: their explanation).
			const help = page === "layer" ? [EXPERT_HELP.rule] : page === "expert" ? [EXPERT_HELP.rule, EXPERT_HELP.rank[h.expertSource ?? "all"]] : undefined;
			const cap = help ? help.map((l) => cut(l, w)) : wrapTextWithAnsi((focus === "MAP" || focus === "TOKENS") && this.engram.look(r) ? EXPLAIN.ENGRAM
				: focus === "LENS" && this.lensShut(r) ? EXPLAIN.LENS_SHUT : this.editHint(explainOf(focus, windowed, s, split, viewOf(h.stream, this.streamViews(r)))), w);
			if (!this.edit || keysToo) footer = [...(cap.length > n ? [...cap.slice(0, n - 1), cut(cap[n - 1] + " …", w)] : cap).map((l) => th.fg("dim", l)),
				...(keysToo ? tail : [])];
		};
		if (h.captions && inspecting) {
			// The explanation's height can differ from the key line's: the range under it; where that changes whether { }
			// works, the explanation says so instead (and the range is the one under that).
			captions(split);
			if (footer.length !== keyRows) at = layout(footer.length);
			if (!!at.range !== split) {
				captions((split = !!at.range));
				at = layout(footer.length);
			}
		}
		const { cols, replyN, rows } = at;
		this.splitRange = at.range;
		this.streamHistory = at.base + this.splitNow() * (over - 1);
		// g: engram inspection's title and readout instead of REPLY. Else after REPLY the steer it ran under (Reply.steer);
		// unsteered, said only while a steer is on now.
		const name = LABELS.engramTitle, note = this.engram.lookupNote(r, inspecting ? sel : -1, w - name.length - 2);
		const ran = note ? "" : r.steer ? th.fg("accent", cut(`steered ${r.steer}`, w - 7)) : r.steer === "" && live.steer ? th.fg("faint", "not steered") : "";
		out.push(fit(th.fg("accent", bold(th, note ? name : "REPLY")) + (note && "  " + note) + (ran && "  " + ran), w));
		// Paused on a token, the strip keeps its window while that token stays in focus, in this reply or an edit's answer
		// to it (stripAt): tokens arriving, an edit answered, nothing before them moves.
		const was = this.stripAt, keep = inspecting && this._sel >= 0 && was?.sel === sel && was.w === w && joined(r, was.rid) ? was : undefined;
		const from = keep?.from ?? Math.max(0, (inspecting ? sel : toks.length - 1) - Math.max(600, replyN * w));   // enough text to fill the strip, not the whole reply
		// The token that went in, on violet (LENS's "in:" says what the violet is): only when streamed, and not under
		// engram inspection's block.
		const paint = this.engram.textPaint(r, inspecting ? sel : -1, from);
		const went = inspecting && (!paint || this.engram.sw?.edit) ? inputAt(toks, sel) : -1;
		const { lines, selLine, ends } = wrapTokens(th, toks.slice(from), w, inspecting ? sel - from : -1, went >= from ? went - from : -1, editMark(r, from), paint);
		const start = keep && selLine >= keep.start && selLine < keep.start + replyN ? keep.start
			: inspecting ? Math.max(0, Math.min(selLine - (one ? 1 : 2), lines.length - replyN)) : Math.max(0, lines.length - replyN);
		this.kept = start === keep?.start;
		this.stripAt = { rid: r.rid, sel, w, from, start, last: from + (ends[Math.min(lines.length, start + replyN) - 1] ?? -1) };
		for (let i = 0; i < replyN; i++) out.push(lines[start + i] ?? "");
		out.push(ruleLine(th, w));
		if (this.anim && !this.animOn(inspecting, k, (this.full || one) && focus !== "LENS")) this.stopSwap();
		let body: string[];
		if (fullNow) {
			// Full size, or the one-panel board: the panel in focus's full-size view.
			body = page ? this.guard("EXPERTS", w, () => (page === "layer" ? layerPage(th, this.experts.xpLayer(r, sel), palette(th), w, rows) : expertPage(th, this.experts.xpPage(r), w)))
				: [tabRow(th, focus, w, shownP, one), ...(one ? this.guard(focus, w, () => this.onePanel(focus, w, rows - 1, r, sel, inspecting, k)) : this.fullPanel(focus, w, rows - 1, r, sel, inspecting, k))];
		} else body = this.middle(cols, w, rows, r, sel, inspecting, k, focus).map((l) => fit(l, w));
		body = body.slice(0, rows);
		while (body.length < rows) body.push("");
		out.push(...body, ruleLine(th, w), ...footer);
		return this.frame(out, width, height, []);
	}
	/** What is known of reply `r`'s model's streams (Live.streams), where it is about that model. */
	private streamData = (r: Reply) => (this.host.live.streams?.model === (r.model ?? "") ? this.host.live.streams : undefined);
	/** s on STREAM: the next view reply `r`'s model has; remembered (host.layoutChanged) at once, the switch drawn as
	 * the lens swap is (streamFrames) where there is a clock (host.now). */
	private nextStream(r: Reply) {
		const views = this.streamViews(r), from = viewOf(this.host.stream, views);
		if (views.length) this.host.stream = nextView(this.host.stream, views);
		if (this.host.now && views.length > 1) this.animate(streamFrames(from, this.host.stream!), this._sel);
		this.host.layoutChanged?.();
	}
	/** Whether the animation goes on this frame (stepped in, `inspecting`): the lens's while on token `k` and LENS is
	 * drawn (not `covered`: another panel full size), STREAM's view switch while the token stepped to is the same
	 * (following: the moving picture). */
	private animOn(inspecting: boolean, k: number, covered = false) {
		const a = this.anim!;
		return inspecting && (this.viewSwitch() ? a.sel === this._sel : !covered && a.sel === k);
	}
	/** The views reply `r`'s model has (board.ts streamViews). */
	private streamViews(r: Reply): StreamView[] {
		const s = this.streamData(r);
		return s ? streamViews(s.k, s.rk, s.shares) : [];
	}
	/** RESIDUAL STREAM for reply `r` at token `sel`, `lines` lines of history (`full`: drawn full size, one of them its
	 * legend's where it has one), `w` wide (streamPanel; ` · s` in its title where `swap` and the model has more than
	 * one view): the view chosen where the model has it (viewOf: the pref kept); "not recorded" where the reply
	 * streamed without. Nothing where the model's streams are not known, or it records none. */
	private stream(r: Reply, sel: number, w: number, lines: number, focused: boolean, swap = false, full = false): string[] {
		const s = this.streamData(r), views = this.streamViews(r);
		if (!s?.L || !views.length) return [];
		this.playNow = true;
		const a = this.viewSwitch(), f = a?.frames[a.i];   // the view switch's frame: its view, fogged
		return streamPanel(this.theme, r.tokens, this.playhead(r, sel), s, f?.stream ?? viewOf(this.host.stream, views), w, lines,
			{ note: r.usual || r.readUsual ? "" : LABELS.notRecorded, focused, swap: swap && views.length > 1, full, fog: f?.fog, clear: f?.clear });
	}
	/** The token STREAM shows of reply `r`: following it, the playhead's (pace: a little behind the newest, moving on
	 * evenly), its ticker started while it is behind; a token stepped to, or no clock (host.now), `sel` itself. */
	private playhead(r: Reply, sel: number): number {
		const now = this.host.now?.();
		if (this._sel >= 0 || now === undefined) {
			this.stopPlay();
			this.play = undefined;
			return sel;
		}
		this.play = pace(this.play, r.rid, sel, r.done, now);
		if (this.playLoop) this.playLoop.drawn = true;
		else if (playShown(this.play) < sel) this.startPlay();
		return playShown(this.play);
	}
	/** The playhead's ticker: every PLAY_TICK_MS it moves on, a draw asked for where the token shown changed; stopped
	 * caught up (the next token's draw starts it again), or where a draw it asked for never came (loop: hidden). */
	private startPlay() {
		const l: Loop = loop(PLAY_TICK_MS, (x) => this.playLoop === x, () => this.stopPlay(), () => {
			const r = shownReply(this.host.live), p = this.play, now = this.host.now?.();
			if (!r || !p || p.rid !== r.rid || now === undefined) return this.stopPlay();
			const newest = r.tokens.length - 1;
			this.play = pace(p, r.rid, newest, r.done, now);
			const shown = playShown(this.play);
			if (shown >= newest) this.stopPlay();
			if (shown !== playShown(p)) this.tui.requestRender();
			else l.drawn = true;   // nothing new to draw: no draw to wait for
		});
		this.playLoop = l;
	}
	private stopPlay() {
		if (!this.playLoop) return;
		clearInterval(this.playLoop.timer);
		this.playLoop = undefined;
	}
	/** The board's header: ◆ metalscope (untested: a model metalscope was not run on here; and the state only where
	 * something is wrong or waiting, `status`; `held` a token stepped to); the steer on the right, a solid chip. */
	private head(w: number, status: string, held = ""): string {
		const th = this.theme, steer = this.host.live.steer;
		const right = steer ? th.fg("accent", bold(th, ansiInverse(` STEER ${cut(steerLabel(steer), Math.floor(w / 2) - 8)} `))) : th.fg("faint", "no steering");
		const left = th.fg("accent", th.bold("◆ metalscope")) + this.untested() + [status, held].filter(Boolean).map((x) => "  " + x).join("");
		return fit(left, w - visibleWidth(right) - 1) + " " + right;
	}
	/** The token LENS shows: the selection, else the newest with a lens read. */
	private lensToken(toks: TokenEvent[], sel: number, inspecting: boolean): number {
		if (inspecting) return sel;
		let k = toks.length - 1;
		while (k > 0 && !(toks[k].lens?.length || toks[k].lens_logit?.length)) k--;
		return k;
	}
	/** LENS on the board, `w` wide in a box of `rows`: the title, the turret and the step, then SETTLING and token `k`
	 * through the lens (telescope.ts), on the lines lensFit gives them (stackLens; `given`: heights' already). Off, dulled
	 * (lensLines). */
	private lensPanel(r: Reply, k: number, w: number, rows: number, focused: boolean, inspecting: boolean, titled = true, board = true, given?: LensFit): string[] {
		const th = this.theme, toks = r.tokens, t = toks[k], opts = this.lensOpts(r, k), fit = given ?? this.lensFit(r, k, w, rows, board);
		return this.lensLines(r, () => stackLens([this.lensHead(r, t, focused, opts, w, titled)], this.settling(toks, k, w, fit.sn, inspecting, opts), fit.sn, fit.gap,
			telescope(th, t, k, w, fit.room, { ...opts, cap: fit.cap }).lines, fit.rows, this.lensPointer(r, t, w)));
	}
	/** LENS in a box of `box` lines sized on the largest step (lensNeed): lensSplit's blank and SETTLING lines over its
	 * `panel` (heights: the same at every step), then the telescope's `room` and its tube's `cap`, and the lines it draws
	 * (`rows`). On the board's column (`board`) with the server's steps, the step's tube a layer a line where the box
	 * holds it, else sharing lines to fit it (fitCap): the largest fills the box, a smaller one leaves the lines under it
	 * to the column (heights). Else the box, a short tube's lines blank at its foot. */
	private lensFit(r: Reply, k: number, w: number, box: number, board = true, panel = box): LensFit {
		const need = this.lensNeed(r, k, w, false, board), [tn, gap, sn] = lensSplit(panel, need.scope, need.settle), t = r.tokens[k];
		const want = this.lensOpts(r, k).want;
		if (!board || !this.sizing(r) || !t || !want || w < MIN_WIDTH || tn - 1 < MIN_LINES) return { gap, sn, room: tn - 1, rows: box };
		const room = box - 1 - sn - gap, cap = fitCap(want, room);
		return { gap, sn, room, cap, rows: 1 + sn + gap + Math.min(room, scopeHeight(tubeOf(t, want, cap))) };
	}
	/** LENS's parts' [least, most] lines in `w` columns, and SETTLING's want: from the server's steps, not the one in use
	 * (sizing), so LENS's box stays through - + (lensFit draws the step in it): the telescope (its title included) over
	 * the largest step's layers (`full`, LENS full size: its tube a layer a line), and SETTLING its least and want over
	 * as many as the middle step says it reads (`n`: the J-lens's where there is one; the logit lens's more share
	 * lines), its most the largest's. The same for every reply (scopeHeight, whatever the reply read: one with no lens
	 * read, LENS shut, as one that has). Without the server's steps, over those reply `r` read. `board`: the board's column, the telescope's least SCOPE_EASY (where
	 * whole), so SETTLING gives its lines before its tube goes under 6 rows. */
	private lensNeed(r: Reply, k: number, w: number, full = false, board = false): { scope: [number, number]; settle: [number, number]; want: number } {
		const toks = r.tokens, t = toks[k], sz = this.sizing(r), cap = full ? 999 : undefined;
		const whole = 1 + (sz ? scopeHeight(tubeOf(t, sz.max.layers, cap)) : telescope(this.theme, t, k, w, 999, { ...this.lensOpts(r, k), cap }).lines.length);
		const [least, want] = sz ? settleRoom(sz.mid.n) : this.settleNeed(toks, k);
		return { scope: [board ? Math.max(1 + MIN_LINES, Math.min(whole, SCOPE_EASY)) : 1 + MIN_LINES, whole], settle: [least, sz ? settleRoom(sz.max.n)[1] : want], want };
	}
	/** LENS's title line: the title (`titled`) and the turret, then (`w` wide) on the right the step (depthTag; its count
	 * kept before the turret's full switch), else "s swaps". */
	private lensHead(r: Reply, t: TokenEvent, focused: boolean, f: Frame, w?: number, titled = true): string {
		// Shut (lensShut, dulled): the title dim, no turret; why where none was read.
		const th = this.theme, shut = this.lensShut(r), name = shut && !focused ? th.fg("dim", th.bold("LENS")) : panelTitle(th, "LENS", focused);
		const headOf = (short: boolean) => (titled ? name + "  " : "") + (!shut ? turret(th, t, f.view, f.turret, short) : this.lensOff(r) ? "" : th.fg("dim", LABELS.notRead));
		let head = headOf(false), step = this.host.lensLayers && w ? this.depthTag(w - visibleWidth(head) - 2, r) : "";
		if (step === null) step = this.depthTag(w! - visibleWidth((head = headOf(true))) - 2, r, true);
		if (step && w) return head + " ".repeat(Math.max(2, w - visibleWidth(head) - visibleWidth(step))) + step;
		return head + (!shut && bothLenses(t) ? th.fg("faint", "   s swaps") : "");
	}
	/** Is EXPERTS (in focus) drawn as its layer window: full size, or the one-panel board's? Its [ ] then page by the
	 * window and ↑/↓ move its focus layer. */
	private isWindowed = (focus: Panel) => focus === "EXPERTS" && (this.full || this.one);
	/** The board's panels, in order: the tab cycle; EXPERTS only on a model with experts (noExperts), STREAM only where
	 * it can be drawn (streamOK). */
	private shownPanels = () => PANELS.filter((p) => (p !== "EXPERTS" || !this.noExperts()) && (p !== "STREAM" || this.streamOK));
	/** The panels a number opens (PANEL_KEY): EXPERTS on a model with experts, STREAM on the board where the model's
	 * streams are known (streamOpens). */
	private openable = (r: Reply | undefined) => PANELS.filter((p) => (p === "STREAM" ? this.streamOpens : p !== "EXPERTS" || !this.noExperts(r)));
	/** The panel in focus: the one chosen (host.panel) where it is shown, else the first shown. */
	private focusNow(shown = this.shownPanels()): Panel {
		const f = this.host.panel ?? "LENS";
		return shown.includes(f) ? f : shown[0];
	}
	/** The shown reply's model routes no experts (Live.dense: a reply of it streamed whole with routing asked for and
	 * none came, or its saved trace had none): EXPERTS not drawn, the mini dock's lights neither. */
	private noExperts(r = shownReply(this.host.live)): boolean {
		return !!r && this.host.live.dense === (r.model ?? "");
	}
	/** The board's middle, `w` by `mid`: its two columns (board.ts arrange), each its panels stacked (column). */
	private middle(cols: { left: Panel[]; right: Panel[] }, w: number, mid: number, r: Reply, sel: number, inspecting: boolean, k: number, focus: Panel): string[] {
		const pad = (ls: string[]) => { const o = ls.slice(0, mid); while (o.length < mid) o.push(""); return o; };
		const { lw, rw } = columns(w);
		return side(this.theme, pad(this.column(cols.left, lw, mid, r, sel, inspecting, k, focus)), pad(this.column(cols.right, rw, mid, r, sel, inspecting, k, focus)), lw, rw);
	}
	/** A column of panels `cw` wide in `rows` lines, a rule between them, each its share of the lines (heights). */
	private column(ps: Panel[], cw: number, rows: number, r: Reply, sel: number, inspecting: boolean, k: number, focus: Panel, titled = true): string[] {
		if (ps.length === 1) return this.panel(ps[0], cw, rows, r, sel, inspecting, k, focus, titled);
		const { h: hs, box, fit } = this.heights(ps, cw, rows, r, k), out: string[] = [];
		ps.forEach((p, i) => {
			if (i) out.push(ruleLine(this.theme, cw));
			const ls = this.panel(p, cw, p === "LENS" ? box : hs[i], r, sel, inspecting, k, focus, titled || p !== focus, fit).slice(0, hs[i]);
			while (ls.length < hs[i]) ls.push("");
			out.push(...ls);
		});
		return out;
	}
	/** Each panel's lines in a column of `rows` (the rules between taken off): first its least, then, by WANT_ORDER,
	 * what its content needs up to the lines given there; what is left then is the column's `slack`; then, by
	 * GROW_ORDER, up to its most (EXPERTS a whole layer at a time, STREAM the rest); anything left stays blank under
	 * them. Too few for every least: they give up lines by GIVE_ORDER. None of it follows the reply's length. Last, LENS
	 * draws the step in use (lensFit) in its `box`: with the server's steps its `panel`, the lines it has with MAP at
	 * its most, the largest step MAP's lines over its least too (while - + stretch to or from it, the frame's `large`
	 * share of them, so MAP moves with the tube). The lines a step leaves go to MAP, then TOKENS, up to
	 * their mosts, the rest blank at the column's foot. */
	private heights(ps: Panel[], cw: number, rows: number, r: Reply, k: number): { h: number[]; slack: number; box: number; fit?: LensFit } {
		const need = ps.map((p) => this.need(p, cw, r, k));
		const h = need.map((n) => n.least);
		let extra = rows - (ps.length - 1) - h.reduce((a, b) => a + b, 0);
		for (const [p, floor] of GIVE_ORDER) {
			const i = ps.indexOf(p);
			if (i < 0 || extra >= 0) continue;
			const give = Math.max(0, Math.min(h[i] - floor(need[i]), -extra));
			h[i] -= give;
			extra += give;
		}
		const grow = (i: number, upTo: number) => {
			const step = need[i].step ?? 1, add = Math.max(0, Math.floor(Math.min(extra, upTo - h[i]) / step) * step);
			h[i] += add;
			extra -= add;
		};
		for (const [p, upTo] of WANT_ORDER) {
			const i = ps.indexOf(p);
			if (i >= 0 && extra > 0) grow(i, Math.min(need[i].want, upTo ?? Infinity));
		}
		const slack = Math.max(0, extra);
		for (const p of GROW_ORDER) {
			const i = ps.indexOf(p);
			if (i >= 0 && extra > 0) grow(i, need[i].most);
		}
		// LENS with the server's steps: its panel the box it has with MAP at its most; the largest step takes MAP's lines
		// too, MAP down to its least.
		const li = ps.indexOf("LENS"), mi = ps.indexOf("MAP"), d = li >= 0 && this.sizing(r) ? this.depthNow().data : undefined;
		let fit: LensFit | undefined;
		const toMost = d && mi >= 0 ? Math.max(0, Math.min(need[mi].most - h[mi], h[li] - need[li].least)) : 0;
		const large = this.lensAnim()?.frames[this.lensAnim()!.i].large ?? +(!!d && isLargest(d, d.at));
		const panel = (h[li] ?? 0) - toMost, box = d && mi >= 0 ? panel + Math.round(large * (h[li] + Math.max(0, h[mi] - need[mi].least) - panel)) : panel;
		if (li >= 0) {
			if (mi >= 0) h[mi] += h[li] - box;
			h[li] = box;
			let free = 0;
			try {
				fit = this.lensFit(r, k, cw, box, true, panel);
				free = Math.max(0, box - fit.rows);
			} catch {}   // LENS says why in its place (guard)
			h[li] -= free;
			for (const i of [ps.indexOf("MAP"), ps.indexOf("TOKENS")]) {
				if (i < 0) continue;
				const add = Math.max(0, Math.min(free, need[i].most - h[i]));
				h[i] += add;
				free -= add;
			}
		}
		return { h, slack, box, fit };
	}
	/** The lines the board's middle, `mid` lines, spares in each of its columns (`cols`): what is left once each panel
	 * has what its content needs (heights' slack). */
	private spare(cols: { left: Panel[]; right: Panel[] }, w: number, mid: number, r: Reply, k: number): [number, number] {
		const { lw, rw } = columns(w);
		return [this.heights(cols.left, lw, mid, r, k).slack, this.heights(cols.right, rw, mid, r, k).slack];
	}
	/** A panel's lines in a column `cw` wide, its title included (Need). TOKENS its most for any token; MAP its title
	 * and MAP_LINES, then the column's rest up to MAP_CELL_LINES; EXPERTS four layers, then up to every layer; STREAM
	 * its frame and the frame's history lines (streamHistory), then the column's rest. */
	private need(p: Panel, cw: number, r: Reply, k: number): Need {
		try {
			return this.needOf(p, cw, r, k);
		} catch (e) {
			this.sizeFailed.set(p, e);
			return { least: 2, want: 2, most: 2 };   // its title and the error (guard)
		}
	}
	private needOf(p: Panel, cw: number, r: Reply, k: number): Need {
		if (p === "LENS") {   // SETTLING and the telescope: their leasts; the telescope whole, SETTLING its want and the blank between; then SETTLING its most (lensSplit), the largest step a layer a line
			const { scope, settle, want } = this.lensNeed(r, k, cw, false, true), sz = this.sizing(r), t = r.tokens[k];
			const tall = sz && t ? Math.max(scope[1], 1 + scopeHeight(tubeOf(t, sz.max.layers, 999))) : scope[1];
			return { least: scope[0] + settle[0], want: scope[1] + want + 1, most: tall + settle[1] + 1, floor: scope[0] };
		}
		if (p === "MAP") return { least: 1 + MAP_LINES, want: 1 + MAP_LINES, most: 1 + MAP_CELL_LINES };
		if (p === "TOKENS") {   // its first choices; a row for a written token outside them where there is room
			const top = this.top();
			return { least: 1 + top, want: 1 + top + 1, most: 1 + top + 1, floor: 1 + Math.min(top, TOKENS_FLOOR) };
		}
		if (p === "STREAM") {   // the rule over it is the column's
			const n = STREAM_FIXED + this.streamHistory;
			return { least: n, want: n, most: Infinity };
		}
		const v = this.experts.expertsView(r);
		if (v.reason) return { least: 2, want: 4, most: 4 };
		const block = this.experts.expertsBlock(r, v, cw);
		return { least: 1 + block, want: 1 + Math.min(v.L, EXPERTS_WANT) * block, most: 1 + v.L * block, step: block };   // the title, then each layer
	}
	/** The right column's least over STREAM: where EXPERTS is drawn (`experts`) EXPERTS' (one layer of this model's
	 * grids as they are over STREAM), else TOKENS' (its ten choices). EXPERTS gives up layers down to that before
	 * STREAM's history loses a line. */
	private rightLeast(r: Reply, experts: boolean): number {
		return this.need(experts ? "EXPERTS" : "TOKENS", RIGHT_W, r, 0).least;
	}
	/** Panel `p`'s lines from `draw`; a throw (a bug, an answer of an unexpected shape), there or working out its
	 * lines (need), is its title and the error in its place, the rest of the dock drawn as ever. */
	private guard(p: Panel, w: number, draw: () => string[]): string[] {
		try {
			if (this.sizeFailed.has(p)) throw this.sizeFailed.get(p);
			return draw();
		} catch (e: any) {
			return [panelTitle(this.theme, p, false), this.theme.fg("error", cut(failed(e), Math.max(1, w)))];
		}
	}
	/** One panel in a column, `w` by `rows` (LENS: its box, `panel` its lines with MAP at its most: heights). */
	private panel(p: Panel, w: number, rows: number, r: Reply, sel: number, inspecting: boolean, k: number, focus: Panel, titled = true, fit?: LensFit): string[] {
		return this.guard(p, w, () => this.panelLines(p, w, rows, r, sel, inspecting, k, focus, titled, fit));
	}
	private panelLines(p: Panel, w: number, rows: number, r: Reply, sel: number, inspecting: boolean, k: number, focus: Panel, titled: boolean, fit?: LensFit): string[] {
		const th = this.theme, toks = r.tokens, f = focus === p;
		if (p === "LENS") return this.lensPanel(r, k, w, rows, f, inspecting, titled, true, fit);
		if (p === "MAP") {
			const per = 2 * Math.max(1, w), line = Math.floor(sel / per), held = this.kept && line >= this.mapFirst && line < this.mapFirst + rows - 1;
			if (!held) this.mapFirst = mapWindow(toks.length, w, rows - 1, inspecting ? sel : -1, this.mapFirst);
			this.mapShown = [r.rid, this.mapFirst * per, Math.min(toks.length, (this.mapFirst + rows - 1) * per)];
			return mapPanel(th, toks, inspecting ? sel : -1, w, rows, f, true, this.mapFirst, this.engram.mapPaint(r, inspecting ? sel : -1));
		}
		if (p === "TOKENS") return this.tokens(r, sel, w, f, rows, titled, this.top());
		if (p === "STREAM") return this.stream(r, sel, w, rows - STREAM_FIXED, f);
		const v = this.experts.expertsView(r);
		const n = v.reason ? 0 : Math.max(1, Math.floor((rows - 1) / this.experts.expertsBlock(r, v, w)));
		const { layers } = this.experts.expertsData(r, sel, n, w);
		return expertsStack(th, v, layers, palette(th), w, rows, f, !v.reason && this.experts.noMap(r, v.layer));
	}
	/** TOKENS for token `sel`, `w` by at most `rows` (`titled`: its title), its first `top` choices; in engram
	 * inspection on a token stepped to, its choices without and with the engram (lookupTokens). */
	/** TOKENS' first choices: the one-panel board's TOP_N, the board's BOARD_TOP. */
	private top = () => (this.one ? TOP_N : BOARD_TOP);
	private tokens(r: Reply, sel: number, w: number, focused: boolean, rows: number, titled = true, top = TOP_N): string[] {
		const v = this.engram.lookView(r), at = this.engram.lookAt(sel);
		return (v && at >= 0 && lookupTokens(this.theme, v, r.tokens, at, w, focused, rows, titled)) || tokensPanel(this.theme, r.tokens[sel], sel, w, focused, titled, rows, top);
	}
	// -- the mini dock (m): lights, MAP, SETTLING and STREAM --------------------------------------------------------

	/** The mini dock, the board's right column wide (RIGHT_W): the token's line (the newest while following), then
	 * EXPERTS as the board's right column draws it (expertsStack: its title, as many layers as the height holds, each
	 * under its heading, each grid a full rectangle; none on a model without experts) from the page holding the EXPERTS
	 * layer; under them the board's MAP (its newest lines, MINI_MAP) and SETTLING where the reply has a lens read, the
	 * lines shared by miniSplit; last, over the one faint key line, RESIDUAL STREAM as the right column's where the model
	 * records writes or reads, its lines taken first: the lights give up layers before its history shrinks. Where its
	 * keys fall short of the width by 4 columns at most, the grids, MAP and SETTLING end where they do. The lights are
	 * always the glow (miniGlow), stepped in or not: as it stood at the token stepped to. Lines that nothing takes go
	 * first to a blank line between each two parts, from the top, then to STREAM's history, else under the key line
	 * (under a grid's worth).
	 * Never the reply or the text panels. */
	private renderMini(width: number, height: number): string[] {
		const th = this.theme, w = Math.max(8, width - 3), inspecting = this.inspecting;
		const keyLine = () => [this.boardNote ? th.fg("warning", fit(this.boardNote, w))
			: th.fg("faint", fit(miniKeys(inspecting, this.sel >= 0, this.engram.gKey(), w, !miniAt(false, this.tui.terminal?.rows)), w))];
		const say = (s: string) => this.frame(wrapTextWithAnsi(th.fg("dim", s), w), width, height, keyLine());
		const r = shownReply(this.host.live);
		if (!r || !r.tokens.length) return say("lights for the next traced reply");
		const toks = r.tokens, sel = this.cur(toks.length);
		if (this.anim && !this.animOn(inspecting, sel)) this.stopSwap();
		// MAP and SETTLING under the lights: [least, want, most] lines, SETTLING's undefined where no lens read.
		const k = this.lensToken(toks, sel, this._sel >= 0);   // as the board's SETTLING: following, the newest with a lens read
		const map = MINI_MAP;
		if (this.host.lensLayers) this.loadDepth(r);   // the lenses off: SETTLING dulled
		// SETTLING's room from the reply (its newest reading, the lens shown: no s here), not the token stepped to: one
		// before the first reading keeps it, drawn as the board draws it there.
		const read = settleLayers(toks, toks.length - 1, this.host.lensView).length, [sl, sm] = settleRoom(read);
		const settle: MiniRoom | undefined = !read ? undefined : [Math.min(MINI_SETTLE_LEAST, sl), sl, sm];
		// EXPERTS: its view (the layer in focus, or why there is nothing); none on a model without experts.
		const v = this.noExperts(r) ? undefined : this.experts.expertsView(r), L = v?.reason ? 0 : (v?.L ?? 0);
		// A line on top names the token the cursor is on (following, the newest): there stepped in or not, so stepping
		// in moves nothing.
		// g: engram inspection's readout instead.
		const note = this.engram.lookupNote(r, inspecting ? sel : -1, w), head = [note ? fit(note, w) : miniToken(th, toks[sel], sel, w)];
		const body = height - 1 - head.length;
		// The lights: EXPERTS' title, then a layer of `block` lines each (expertsBlock) `w` wide; no routing, its title and
		// why, whole.
		const why = v && !L ? expertsStack(th, v, [], palette(th), w, body, false) : [];
		const sd = this.streamData(r), sL = sd?.L && this.streamViews(r).length ? sd.L : 0;
		const block = L ? this.experts.expertsBlock(r, v!, w) : why.length;
		const title = L ? 1 : 0;
		// STREAM's lines first, a layer of lights (or why), MAP and SETTLING left their least (board.ts streamLines; no rule
		// over it); for a model not known yet, asked for (streamRoom) where the room is there. Its lines over STREAM_LINES
		// (streamShare) only while two layers of lights stay, where STREAM_LINES leaves two.
		const free = body - title - block - map[0] - (settle?.[0] ?? 0), least = streamLines(free, STREAM_FIXED);
		const lit = (n: number) => (L ? miniSplit(body - (n ? n + STREAM_FIXED : 0) - title, block, L, map, settle).n : 0);
		let sn = streamLines(free, STREAM_FIXED, streamShare(height));
		while (sn > least && lit(sn) < Math.min(2, lit(least))) sn--;
		this.noteRoom(sn > 0);
		const hist = sL ? sn : 0, streamN = hist && hist + STREAM_FIXED;   // STREAM drawn: its history lines, all its lines
		this.miniStream = hist > 0;   // s switches its view
		const keys = keyLine(), rest = body - streamN;
		// The lights (`share` lines), the maps right under them, STREAM, the key line, a blank line between each two from
		// the lines left, then STREAM's history the rest; under the key line what is still left (no STREAM) when fewer
		// than a layer's `block` (the maps at their most, or no lights), else over it (every layer drawn).
		const draw = (lights: string[], split: { map: number; settle: number }, share: number) => {
			const parts = [[...lights.slice(0, share), ...Array<string>(Math.max(0, share - lights.length)).fill("")],
				split.map ? this.panel("MAP", w, split.map, r, sel, inspecting, k, "EXPERTS").slice(0, split.map) : [],
				split.settle ? this.settling(toks, k, w, split.settle, inspecting).map((l) => (this.lensOff(r) ? dull(th, l, DULL) : l)) : []].filter((p) => p.length);
			let left = Math.max(0, body - streamN - parts.reduce((t, p) => t + p.length, 0));
			const more = hist && Math.max(0, left - parts.length);   // past a blank line over each part
			if (hist) parts.push(this.stream(r, sel, w, hist + more, false, inspecting)), (left -= more);
			const out = [...head];
			for (const [i, p] of parts.entries()) {
				if (i && left) out.push(""), left--;
				out.push(...p);
			}
			const spare = left < block ? left : 0;
			return [...this.frame(out, width, height - spare, keys), ...this.frame([], width, spare, [])];
		};
		if (!L) return draw(why, miniSplit(rest, why.length, v ? 1 : 0, map, settle, hist > 0), why.length);   // a model without experts: MAP and SETTLING have it all, or leave it to STREAM
		// From the EXPERTS layer, a full page where the layers allow (the last page ends at the deepest), the layers in
		// the right column's grids (reshapeGrid at the blocks' width), lit as the glow.
		const split = miniSplit(rest - title, block, L, map, settle), n = split.n, E = v!.E, base = this.host.live.base;
		const a = (this.miniFrom = Math.max(0, Math.min(v!.layer, L - n))), layers: LayerLights[] = [];
		for (let l = a; l < Math.min(L, a + n); l++) {
			const shape = reshapeGrid(E, gridShape(E, this.grids.at(base, r.model, l)), w);
			layers.push({ layer: l, cells: gridCells(this.experts.miniGlow(r, l, E, sel), shape), cols: shape.cols, rows: shape.rows });
		}
		this.miniN = n;   // [ ] step by the layers drawn
		const room = title + n * block;
		return draw(expertsStack(th, v!, layers, palette(th), w, room, false, this.experts.noMap(r, v!.layer)), split, room);
	}
	/** The mini dock's keys: m the full dock, [ ] the next or previous layers, s STREAM's next view (where it is
	 * drawn; remembered, as the board's), ←/→ and the other token keys a token (the glow as it stood there), esc back,
	 * ctrl+\ closes. */
	private miniKey(data: string) {
		const h = this.host, r = shownReply(h.live), render = () => this.tui.requestRender();
		this.boardNote = "";
		if (this.anim) this.stopSwap(), render();   // any key ends an animation, as on the board
		if (matchesKey(data, "ctrl+\\")) return h.close();
		if (matchesKey(data, "escape") || data === "q") return void (this.leave() || render());
		if (data === "m") return this.setMini(false);
		if (data === "g") return void (this.editBusy() || this.engram.lookupKey(r)), render();   // not while an edit is answered
		if (data === "[" || data === "]") {
			const L = r ? routedLayers(r.tokens) : 0, n = this.miniN;
			if (L) h.expertLayer = shownLayer(L, this.miniFrom + (data === "]" ? n : -n));
			return render();
		}
		if (!r?.tokens.length) return;
		if (data === "s" && this.miniStream) return this.nextStream(r), render();
		if (this.moveToken(data, r)) render();
	}
	/** m: the mini dock on or off; remembered, the dock's room and the stream follow (host.layoutChanged). */
	private setMini(on: boolean) {
		const rows = this.tui.terminal?.rows;
		if (!on && miniAt(false, rows)) {   // the full dock would not show: refused, said within the mini width
			this.boardNote = `full dock needs ${DOCK_MIN_ROWS} rows, has ${rows}`;
			return this.tui.requestRender();
		}
		this.host.mini = on;
		this.full = this.steerPage = false;
		this.host.layoutChanged?.();
		this.tui.requestRender();
	}
	/** The steer on, for the telescope: the layers it pushes at (◆). */
	private marks(): Pick<Frame, "steer"> {
		return { steer: (this.host.live.steer?.parts ?? []).map((p) => p.layer) };
	}
	/** The panel in focus over the dock (its number), `rows` lines `w` wide. */
	private fullPanel(focus: Panel, w: number, rows: number, r: Reply, sel: number, inspecting: boolean, k: number): string[] {
		return this.guard(focus, w, () => this.fullLines(focus, w, rows, r, sel, inspecting, k));
	}
	private fullLines(focus: Panel, w: number, rows: number, r: Reply, sel: number, inspecting: boolean, k: number): string[] {
		const th = this.theme;
		if (focus === "LENS") return this.lensLines(r, () => this.lensPage(r, k, w, rows, inspecting));   // shut: dulled (lensLines)
		if (focus === "EXPERTS") return this.experts.expertsWindow(r, sel, w, rows);
		if (focus === "TOKENS") return this.tokens(r, sel, w, false, rows, false);
		if (focus === "STREAM") return this.stream(r, sel, w, Math.max(1, rows - STREAM_FIXED), true, false, true);
		const m = sessionMap(th, this.host.live.map, r, inspecting ? sel : -1, w, rows, this.sessionFirst, this.engram.mapPaint(r, inspecting ? sel : -1));   // the session's replies
		this.sessionFirst = m.first;
		this.mapShown = [r.rid, ...m.shown];
		return m.lines;
	}
	/** LENS full size, `rows` lines `w` wide, as the board's LENS (lensNeed, lensSplit, stackLens): the step on a line
	 * of its own, SETTLING across the width, then both lenses side by side, each under its name; the rule and the legend
	 * last. SETTLING its want first (the telescopes keep their least: a tube of three lines), the telescopes the rest up
	 * to a layer a line for the largest step, SETTLING any left up to its most. The tube box the same at every step: a
	 * layer a line where the largest step's fit, else shared at the shallow end. */
	private lensPage(r: Reply, k: number, w: number, rows: number, inspecting: boolean): string[] {
		const th = this.theme, toks = r.tokens, { lw, rw } = columns(w);
		const t = toks[k], both = bothLenses(t), one = this.host.lensView ?? "jlens";
		const base = this.lensOpts(r, k), step = this.host.lensLayers ? [this.depthLine(w, r)] : [];
		const opts = (view: "jlens" | "logit", cap?: number): Frame => ({ ...base, view, cap,
			edit: lensOf(t, view).by === lensOf(t, this.host.lensView).by ? base.edit : undefined });
		const mixed = !both && toks.some(bothLenses), ow = mixed ? lw : w;   // one lens: the width, unless others have two
		const need = this.lensNeed(r, k, w, true), room = rows - 2 - step.length, box = this.sizing(r)?.max.layers ?? base.want ?? readLayers(t);
		const least = Math.max(need.scope[0], 1 + scopeHeight(tubeOf(t, box, 3)));
		const [tn, gap, sn] = lensSplit(room, [least, need.scope[1]], [need.want + 1, need.settle[1]]), cap = Math.max(1, tn - (need.scope[1] - box.length));   // want + 1: the blank line too
		const name = (s: string) => th.fg("lens", th.bold(s)), scope = (view: "jlens" | "logit", ww: number, head: string) => [head, ...telescope(th, t, k, ww, tn - 1, opts(view, cap)).lines];
		// One lens: its name; none read the token, as the board's LENS says it.
		const single = () => scope(one, ow, !readLayers(t).length ? th.fg("dim", LABELS.notRead) : name(t.lens_by === "logit" ? "logit lens" : "J-lens"));
		const top = both ? side(th, scope("jlens", lw, name("J-lens")), scope("logit", rw, name("logit lens")), lw, rw)
			: mixed ? side(th, single(), [th.fg("dim", "one lens recorded for this token: nothing to set beside it")], lw, rw) : single();
		return [...stackLens(step, this.settling(toks, k, w, sn, inspecting, base), sn, gap, top, rows - 2), ruleLine(th, w),
			th.fg("faint", fit(`${both ? "both lenses · ↑/↓ the layer in both" : "↑/↓ the layer"} · ▸ the written word settles · ◆ the steered layer`, w))];
	}
	/** The one-panel board's panel `p`, `w` by `rows` under the tab row: its full-size view (fullPanel); LENS, where the
	 * two lenses do not fit side by side, its board column; TOKENS, where EXPERTS fits
	 * under it, the two as one column. */
	private onePanel(p: Panel, w: number, rows: number, r: Reply, sel: number, inspecting: boolean, k: number): string[] {
		if (p === "LENS" && columns(w).lw < LEFT_MIN) return this.lensPanel(r, k, w, rows, true, inspecting, false, false);
		const col: Panel[] = ["TOKENS", "EXPERTS"];
		if (p === "TOKENS" && !this.noExperts(r) && rows >= this.need("TOKENS", w, r, k).want + 1 + this.need("EXPERTS", w, r, k).least)
			return this.column(col, w, rows, r, sel, inspecting, k, p, false);
		return this.fullPanel(p, w, rows, r, sel, inspecting, k);
	}
	/** A key on the board: tab the panel in focus, a panel's number opens it (PANEL_KEY), ? explain it, 6 the Steer
	 * page; the panel in focus takes the arrows. */
	private boardKey(data: string) {
		const h = this.host, focus = this.focusNow(), r = shownReply(h.live);
		if (this.anim) {
			this.stopSwap();
			this.tui.requestRender();
		}
		const render = () => this.tui.requestRender();
		if (!this.editBusy()) this.boardNote = "";   // an edit under way: its note stays
		if (matchesKey(data, "ctrl+\\")) return h.close();
		if (this.steerPage) {
			this.sp.steerPageKey(data);
			return render();
		}
		// LENS's keys on the token it shows (lensToken: following, the newest with a lens read).
		const lk = r?.tokens.length ? this.lensToken(r.tokens, this.cur(r.tokens.length), this._sel >= 0) : -1;
		if (focus === "LENS" && r && lk >= 0 && this.lensKey(data, r, lk, this.want(r))) return render();
		if (focus === "EXPERTS" && this.isWindowed(focus) && r?.tokens.length && this.experts.expertKey(data, r)) return render();
		// - + wherever LENS is drawn (in its column, or full size); a word being edited takes them as letters (lensKey).
		const lensShown = !(this.full || this.one) || focus === "LENS";
		if (lensShown && r && lk >= 0 && this.depthKey(data, r, lk)) return render();
		if (matchesKey(data, "escape") || data === "q") {   // esc: a full-size panel stays (its number goes back); q: the board
			if (data === "q") this.full = false;
			return void (this.leave() || render());
		}
		if (data === STEER_KEY) {   // the Steer page: its concepts asked for again, its chats from the top
			this.steerPage = true;
			this.sp.steer.seen = "";
			this.sp.steer.scroll = 0;
			return render();
		}
		if (data === "m") return this.setMini(true);
		if (!r || !r.tokens.length) return;   // no reply yet: the rest change nothing
		if (data === "?") {
			h.captions = !h.captions;
			return render();
		}
		if (matchesKey(data, "tab") || matchesKey(data, "shift+tab")) {
			const order = this.shownPanels();
			const at = Math.max(0, order.indexOf(focus));
			h.panel = order[(at + (matchesKey(data, "tab") ? 1 : order.length - 1)) % order.length];
			return render();
		}
		// A panel's number: it in focus, open full size (the one-panel board: in focus); again, the board. One not there
		// (EXPERTS on a model without; STREAM where it does not fit the dock's width): nothing.
		const p = PANELS.find((x) => PANEL_KEY[x] === data);
		if (p) {
			if (!this.openable(r).includes(p)) return;
			if (!this.one) this.full = !(this.full && focus === p);   // the one-panel board: a kept full-size panel kept
			h.panel = p;
			this.experts.xp.at = "grid";
			return render();
		}
		if (data === "g") return void (this.editBusy() || this.engram.lookupKey(r)), render();   // not while an edit is answered
		if (focus === "STREAM" && data === "s") return this.nextStream(r), render();   // its view, not the lens
		const up = matchesKey(data, "up") || data === "k", down = matchesKey(data, "down") || data === "j";
		if (data === "s") {   // swapped on the token LENS shows; LENS shut, nothing
			if (this.lensShut(r)) return;
			const lt = r.tokens[lk], from = lensOf(lt, h.lensView).by;
			h.lensView = h.lensView === "logit" ? "jlens" : "logit";
			const to = lensOf(lt, h.lensView).by;
			if ((!(this.full || this.one) || focus === "LENS") && bothLenses(lt) && from && to && from !== to) this.startSwap(lt, from, to, lk);
			return render();
		}
		// { }: the divider between EXPERTS and STREAM a layer of EXPERTS up ({, STREAM more) or down (}); remembered.
		// At an end, or the board not drawn as the last frame's range was (full size, the one-panel board, the Steer
		// page): nothing moves, nothing stored.
		if ((data === "{" || data === "}") && this.splitRange && !this.full && !this.one && !this.steerPage) {
			const [lo, hi] = this.splitRange, at = this.splitNow(), to = Math.max(lo, Math.min(hi, at + (data === "{" ? 1 : -1)));
			if (to === at) return;
			h.split = to;
			h.layoutChanged?.();
			return render();
		}
		// EXPERTS on the board: [ ] a layer (its window: expertKey).
		if (focus === "EXPERTS" && (data === "[" || data === "]")) {
			const L = routedLayers(r.tokens);
			if (L) h.expertLayer = shownLayer(L, shownLayer(L, h.expertLayer) + (data === "]" ? 1 : -1));
			return render();
		}
		// TOKENS: ↑ the next (newer) unsure token, ↓ the previous (older).
		if (this.moveToken(focus === "TOKENS" && (up || down) ? (up ? "u" : "U") : data, r)) render();
	}
	/** LENS's keys: ↑/↓ a layer, x steering off, enter edits a word the eyepiece shows
	 * (the one written, else the first). Editing: ⌫ removes it (then a letter), letters type over it, ↑/↓ fill in the
	 * lens's other words, tab the next word, esc cancels; enter answers the reply again from this token with the edit
	 * (-OLD or OLD>NEW) at this token only (editAt). Whether it took the key. */
	private lensKey(data: string, r: Reply, cur: number, want?: number[]): boolean {
		const h = this.host, t = r.tokens[cur], { top1 } = lensOf(t, h.lensView), layers = onTube(top1.map(([l]) => l), want);
		const fl = focusLayer(layers, lensSettle(top1, t.id), h.lensLayer), e = this.editOn(r, fl === null ? -1 : cur);
		if (!e && data === "x" && h.live.steer) return this.sp.apply("off"), true;
		if (fl === null || this.lensShut(r)) return false;   // shut: x alone
		const eye = eyepiece(t, h.lensView, fl), up = matchesKey(data, "up"), down = matchesKey(data, "down");
		if (!e) {
			if (up || down || data === "k" || data === "j") h.lensLayer = stepLayer(layers, fl, up || data === "k" ? 1 : -1);
			else if (matchesKey(data, "enter") && eye.length && this.canEdit()) {
				if (this.editBusy()) return true;   // one edit at a time: taken, nothing opened
				// The token held while its word is edited: streaming, or not the newest (its newest have no lens read).
				if (this._sel < 0 && (!r.done || cur < r.tokens.length - 1)) this.sel = cur;
				this.edit = { rid: r.rid, at: cur, i: t.i, row: Math.max(0, eye.findIndex((x) => x.id === t.id)), field: null, fill: -1 };
			} else return false;
			return true;
		}
		const raw = eye[e.row]?.text ?? "", word = raw.trim();
		// The lens's other words with their ids, eyepiece first (↑/↓).
		const fills = new Map<string, number>();
		for (const [w, id] of [...eye.map((x) => [x.text.trim(), x.id] as const), ...[...top1].reverse().map(([, id, w]) => [w.trim(), id] as const)])
			if (w && w !== word && !fills.has(w)) fills.set(w, id);
		if (matchesKey(data, "escape")) this.edit = undefined;
		else if (matchesKey(data, "tab")) Object.assign(e, { row: (e.row + 1) % Math.max(1, eye.length), field: null, fill: -1, from: undefined });
		else if (matchesKey(data, "backspace")) Object.assign(e, { field: e.field ? [...e.field].slice(0, -1).join("") : "", from: undefined });
		else if (up || down) {
			const list = [...fills.keys()];
			e.fill = Math.max(0, Math.min(list.length - 1, e.fill + (down ? 1 : -1)));
			if (list.length) Object.assign(e, { field: list[e.fill], from: fills.get(list[e.fill]) });
		} else if (matchesKey(data, "enter")) {
			// Every lens word goes by its token id (tokenName: the token shown), shown as itself everywhere; a word
			// typed by hand by its name.
			const f = e.field, id = eye[e.row]?.id ?? -1, name = tokenName(raw, id);
			const into = !f ? undefined : e.from !== undefined ? tokenName(f, e.from) : NAME.test(f) && parseNumber(f) === undefined ? f : null;
			const bad = name === null ? raw : into === null ? f! : undefined;
			if (f === null) this.boardNote = "⌫ removes it, or type a word";
			else if (bad !== undefined) this.boardNote = `${JSON.stringify(bad)} is ${parseNumber(bad) === undefined ? "not a word" : "a number, not a word"}`;
			else {
				this.edit = undefined;
				const ids: WordIds = { [name!]: { id, text: word } };
				if (f && e.from !== undefined) ids[into!] = { id: e.from, text: f };
				this.editAt(r, t, word, name!, ids, f || undefined, into ?? undefined);
			}
		} else if (/^[\w.-]$/.test(data)) Object.assign(e, { field: (e.field ?? "") + data, from: undefined });
		return true;
	}
	/** The LENS edit as the telescope draws it, with where it acts on reply `r`'s model: the lens steps' answer's
	 * `edit_band` (none without a J-lens: no label). */
	private editFrame(r: Reply): Frame["edit"] {
		const e = this.edit, d = this.depth.data;
		return e && { ...e, band: d && (!r.model || r.model.split("/").pop() === d.model) ? d.edit_band : undefined };
	}
	/** The LENS edit, dropped unless token `cur` of reply `r` is the one it was opened on (a new reply, a token arriving;
	 * -1: no layer to read). */
	private editOn(r: Reply, cur: number) {
		if (this.edit && (this.edit.rid !== r.rid || this.edit.at !== cur)) this.edit = undefined;
		return this.edit;
	}
	/** Panel background + left edge, padded to the overlay's size, footer at the bottom. The edge leans toward whoever
	 * has the keyboard: stepped in, ▐ (the dock's side) in orange; else ▌ (the chat's side), dim. */
	private frame(body: string[], width: number, height: number, footer: string[]): string[] {
		const th = this.theme;
		const w = width - 3;
		const rows = body.slice(0, height - footer.length);
		while (rows.length < height - footer.length) rows.push("");
		rows.push(...footer);
		const edge = (this.stepped ? th.fg("accent", "▐") : th.fg("dim", "▌")) + " ", ew = plainWidth(edge);
		return rows.map((l) => {
			const plain = plainWidth(l), n = plain >= 0 ? plain : visibleWidth(l);   // plain: tint need not measure it again
			return n > w ? tint(th, edge + truncateToWidth(l, w), width) : tint(th, edge + l, width, plain >= 0 ? ew + plain : undefined);
		});
	}
}

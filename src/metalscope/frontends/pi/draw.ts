/**
 * Drawing helpers for the dock: confidence colours, bars, token wrapping; lines of parts that give way (the job
 * widget and the transcript entries).
 */

import { backgroundAnsi, foregroundAnsi, rgbColor, truncateToWidth, visibleWidth, wrapTextWithAnsi } from "@earendil-works/pi-tui";

export const UNSURE = 0.5;
const SURE = 0.9;

/** The tally's tokens an expert is most distinctive for (top by lift): one name on every EXPERTS screen. */
const ITS = "its tokens";
/** The words of the full-size EXPERTS, its layer page and its expert page (by source, `a`), in one place. */
export const LABELS = {
	crumb: "EXPERTS", sep: " › ", layerName: (n: number) => `layer ${n}`, back: (e: number) => `b #${e}`, token: (n: number) => `token #${n}`,
	legend: { lit: (sel: number) => ["lit: the experts mixed", `for #${sel}`], glow: ["bright: just used", "warm: busy lately"] },
	forToken: (t: string) => `EXPERTS FOR ${t}`, its: ITS,
	layer: (n: number) => `LAYER ${n}'S EXPERTS`, at: (i: number, n: number) => `${i} of ${n}`,
	rank: { all: "pickiest", session: "most changed" }, unranked: "too few tokens to rank",
	source: { all: "all chats", session: "this session" }, noTally: "no tally",
	takes: (n: string, of: string, share: string) => `takes ${n} of ${of} tokens (${share})`,
	times: (x: string) => `×${x}: ${x} times more often than chance`,
	tokens: ITS.toUpperCase(), mixed: "MIXED WITH", mixedTag: "same tokens", reply: "IN THIS REPLY",
	// LENS's - + line.
	reads: (name: string, n: number) => `${name} · reads ${n} layers`, layers: (name: string, n: number) => `${name} · ${n} layers`, layersKey: "-/+ layers",
	// LENS dulled: the lenses off (- past the fewest layers; + on again), or none read for the reply shown.
	linkNoAnswer: "no answer: the link may have gone through",
	lensOff: "off", lensOn: "+ lenses", notRead: "not read for this reply",
	notRecorded: "not recorded",   // STREAM on a reply streamed without the writes (not asked for then); its title fits the right column
	asking: "asking how many layers the lenses read…", layersFailed: "layers unknown",
	layersBad: "the lens steps' answer did not parse",
	speed: (tps: number, k: number) => `${tps} tok/s · last ${k === 1 ? "reply" : `${k} replies`}`, noSpeed: "no reply at this setting yet",
	// The LENS word edit: the reply answered again from the edited token (ds4), and what came of it there.
	editing: "answering again from this token…", editNot: "not edited", editNewest: "only the newest reply can be edited",
	editHere: "this server cannot edit one token of this model",
	// g, engram inspection (lookup.ts): the key's labels (its short one where a line is tight; while it reads, stop), the
	// title, the legend (shorter where narrow), the sweep's two lines and its closing pass, the readings' bar (lookupBar),
	// and why there is nothing.
	engramKey: "g engram inspection", engramShort: "g engram", engramClose: "g close", engramStop: "g stop", engramTitle: "ENGRAM INSPECTION",
	engramClosing: "back to the reply as written…", engramStopped: "stopped",
	reading: (i: number) => `reading ${i} of 2`, of2: (i: number) => `${i} of 2`,
	secs: (s: number) => (s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`),
	engramLegend: ["blue: helped by the engram · block: what it looked up · underlined/white: what that predicted",
		"blue: helped · block: engram · marked: predicted", "blue: helped by the engram"],
	engramSweep: ["reading the reply without the engram…", "reading the reply with the engram…"],
	// EXPERTS where the checkpoint has no expert map (the grid in number order): how to make one.
	noMap: "no expert map · metalscope experts layout",
	lookupNone: "no engram on this model", lookupWait: "g: once the reply is done", lookupUnsaved: "this reply was not saved",
	lookupBusy: "reading the reply twice…", lookupBad: "the engram's answer did not parse", lookupDiffer: "trace tokens differ: not shown",
	edited: (was: string, now: string | null, changed: boolean, want?: string, landed?: boolean, tool?: boolean) =>
		(now === null ? `edit: the reply now ends before ${was}`
		: `edit: ${changed ? `${was} became ${now}` : `${was} unchanged`}${want === undefined ? "" : landed ? ", as asked" : `, not ${want}`}`)
		+ (tool ? ", stopped at a tool call" : ""),
} as const;
/** What ? says on the full-size EXPERTS' pages (the grids say EXPLAIN.EXPERTS_WINDOW): the counting rule; on the
 * expert page then what its rank means by its source. Each a line cut to fit. */
export const EXPERT_HELP = { rule: "a token counts for its 3 strongest experts",
	rank: { all: "pickiest: tokens unlike the layer's", session: "most changed: unlike its own usual" } } as const;

/** What each part of the dock shows, short: ? puts the one in focus where the key line is (the board's panels:
 * board.ts explainOf), and /scope help dock lists them (help.ts). The screen says the rest. */
/** The LENS word edit in the ? texts, left out where the edit is not offered (dock.ts canEdit). */
export const EDIT_HINT = " enter: remove or swap a word (experimental).";
export const EXPLAIN: Record<string, string> = {
	REPLY: `Each token coloured by its probability: green ${SURE * 100}%+, yellow ${UNSURE * 100}-${SURE * 100}%, red under ${UNSURE * 100}%. ▸: the edited token, on orange where it changed.`,
	EXPERTS: "Layers of experts through the depth, as many as fit. Following, the glow of the newest tokens; paused on a token, its experts, each lit by its share of the mix.",
	SPLIT: " { }: a layer less or more.",
	EXPERTS_WINDOW: "Consecutive layers of experts, ▸ the layer in focus, lit as the board's; under them the token's experts there, each its share of the mix. enter: the layer, then an expert.",
	TELESCOPE: `The token in focus at each layer, deep on top: its top word (green: written; ▸ settled), the written word's probability. in: the input.${EDIT_HINT} -/+: layers read.`,
	BOARD_LENS: `SETTLING: where tokens settled. Then the token in focus at each layer, deep on top: its likeliest word (green: written; ▸ settled).${EDIT_HINT} -/+: layers read.`,
	LENS_SHUT: "The lenses off (+ on), or not read for this reply.",
	STEER: "Concepts ranked by the chat test (★ top five; 15/18 @0.3: good replies of 18 at its dose). The one under the cursor: a column per dose, no steering first.",
	STEER_EXAMPLES: "The chat test's turns from this one at the dose under the cursor: no steering beside steered, graded 0-3.",
	STEER_LINK: "l: use a same-shape model's concepts and J-lens; doses stay this build's own.",
	TOKENS: "The model's top choices for the token in focus (1: all 10), ▸ the one written (green 90%+, yellow 50-90%, red under 50%). off-top: sampling chose it (under 50%, another over 50%).",
	SETTLING: "SETTLING: a column a token, the layers the lens in use read, deep on top. Bright: settled on the written word; dim: it tops a lower layer; faint: only in a top 10. Grey: the token in focus.",
	MAP: "A token a cell (green 90%+, yellow 50-90%, red under 50%), thinking darker, white in focus: the reply, sliding by lines; full size, the session, each reply on a new row (amber: another model).",
	ENGRAM: "g: the reply read again without the engram. Blue: tokens it helped (brighter: more). Block: what the cursor's token looked up; underlined/white: that token. TOKENS: its top choices without and with.",
};
/** What ? says on STREAM, by what the model has (`k` residual streams written, `rk` sources read; one stream with reads:
 * a block's sum, Inkling 2's): a line for the model, and a line a view by its name (board.ts viewName). */
export function streamExplain(k: number, rk: number): [shared: string, views: Record<string, string>] {
	const largest = "How much each layer changed the residual for its size. Brighter is more.";
	if (k > 1) return [`Hyper-connections give each layer ${k} residual streams. The newest token is the lowest row.`, {
		"spikes (hc)": "One stream's change in a layer was far above its usual. Each colour is a stream.",
		"largest (hc)": "Each layer in the colour of the stream it changed most.",
		"spikes (layers)": `A layer's change across all ${k} streams was far above its usual in this reply.`,
		"largest (layers)": largest }];
	if (!rk) return ["The newest token is the lowest row.", { spikes: "A layer changed the residual far more than it usually does in this reply.", largest }];
	return ["A column is a layer visit. Each block of layers keeps its own sum.", {
		"spikes (writes)": "A visit added far more than usual to its block's sum.",
		"spikes (reads)": "A visit drew far more than usual on one source: the embedding, a finished block, or its own sum. Each colour is a source." }];
}

export function confColor(p: number | null): string {
	if (p === null || p < UNSURE) return "error";
	return p < SURE ? "warning" : "success";
}

/** Text from outside (model tokens, a pulled concept's title) made safe to draw: no control character but newline and
 * tab reaches the terminal as an escape (OSC 52 writes the clipboard, OSC 8 links). Those Python's strip() takes for
 * whitespace (VT, FF, CR, 0x1c-0x1f, NEL) become a space, so a word trimmed here is the word the server strips; the
 * rest of C0, DEL and C1 a "·". */
export const clean = (s: string) => s.replace(/[\x0b-\x0d\x1c-\x1f\x85]/g, " ").replace(/[\x00-\x08\x0e-\x1b\x7f-\x84\x86-\x9f]/g, "·");
/** Text drawn on one line (partsLine's parts): newlines and tabs a space, then made safe (clean). */
export const flat = (s: string) => clean(s.replace(/[\n\t]/g, " "));

export const pct = (p: number | null) => (p === null ? "<top-10" : p < 0.001 ? "<0.1%" : `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`);

// -- text helpers (plain text by visible width: colour goes on after) ----------------------------------------------

/** A token's text on one line: newlines ↵, tabs a space. */
function show(text: string): string {
	return text.replace(/\n/g, "↵").replace(/\t/g, " ");
}
/** A lens word: one line, trimmed, "␣" where it was only spaces (so it never shows as nothing). */
export const showWord = (s: string) => show(s).trim() || "␣";
/** Reply text around a token (the expert page): one line, tabs a "·" so they are seen, made safe. */
export const showTabs = (s: string) => clean(s.replace(/\n/g, "↵").replace(/\t/g, "·"));
export const qt = (s: string) => `"${show(s)}"`;
/** A token's text as the dock shows it: quoted, newlines as ↵, tabs and other control characters made visible. */
export const label = (s: string) => qt(clean(s));

/** `s` in at most `w` columns, an ellipsis where it was cut. */
export function cut(s: string, w: number): string {
	if (visibleWidth(s) <= w) return s;
	let out = "", n = 0;
	for (const ch of s) {
		const cw = visibleWidth(ch);
		if (n + cw > w - 1) break;
		out += ch;
		n += cw;
	}
	return out + "…";
}
/** `line` cut or padded to exactly `w` columns (cut without an ellipsis: the panels end where they end). */
export function fit(line: string, w: number): string {
	const n = textWidth(line);
	if (n <= w) return line + " ".repeat(w - n);
	const t = truncateToWidth(line, w, "");
	return t + " ".repeat(Math.max(0, w - textWidth(t)));
}
/** Padded to `w` columns on the right: pad never cuts (tint cuts after), padTo cuts first (cut). */
export const pad = (s: string, w: number) => s + " ".repeat(Math.max(0, w - visibleWidth(s)));
export const padTo = (s: string, w: number) => { s = cut(s, w); return s + " ".repeat(Math.max(0, w - visibleWidth(s))); };
/** Padded on the left: lpad never cuts, lpadTo cuts first. */
export const lpad = (s: string, w: number) => " ".repeat(Math.max(0, w - visibleWidth(s))) + s;
export const lpadTo = (s: string, w: number) => { s = cut(s, w); return " ".repeat(Math.max(0, w - visibleWidth(s))) + s; };
/** `s` cut to `w` columns and centred in exactly `w`. */
export function centre(s: string, w: number): string {
	s = cut(s, w);
	const l = Math.floor((w - visibleWidth(s)) / 2);
	return " ".repeat(l) + s + " ".repeat(Math.max(0, w - visibleWidth(s) - l));
}
/** `s` after half the room `w` leaves it: never cut, nothing after it (the idle screen's art, framed later). */
export const centreLeft = (s: string, w: number) => " ".repeat(Math.floor((w - visibleWidth(s)) / 2)) + s;

/** Characters each a grapheme of its own (no mark, joiner or control among them): `cellsOf[c]` its width, measured
 * once by pi-tui (-1 until then); -2 for every other character below 0x2c00. */
export const ALONE = [[0xa1, 0xac], [0xae, 0xff], [0x370, 0x3ff], [0x2010, 0x2027], [0x2030, 0x205e], [0x2100, 0x27bf], [0x2b00, 0x2bff]];
const cellsOf = new Int8Array(0x2c00).fill(-2);
for (const [a, b] of ALONE) cellsOf.fill(-1, a, b + 1);
/** pi-tui's visibleWidth, without its grapheme segmenter for the dock's own lines (most of a tall board's frame went
 * there): printable ASCII, colour codes (ESC [ … m) and ALONE's characters summed (plainWidth); anything else is pi-tui's. */
export function textWidth(s: string): number {
	const n = plainWidth(s);
	return n >= 0 ? n : visibleWidth(s);
}
/** textWidth's sum, -1 where `s` holds anything else; two such strings joined are as wide as both. */
export function plainWidth(s: string): number {
	let n = 0;
	for (let i = 0; i < s.length; i++) {
		const c = s.charCodeAt(i);
		if (c >= 0x20 && c <= 0x7e) n++;
		else if (c === 0x1b && s.charCodeAt(i + 1) === 0x5b) {
			let j = i + 2, d = s.charCodeAt(j);
			while (d >= 0x30 && d <= 0x3f) d = s.charCodeAt(++j);
			if (d !== 0x6d) return -1;
			i = j;
		} else {
			const k = c < 0x2c00 ? cellsOf[c] : -2;
			if (k === -2) return -1;
			n += k >= 0 ? k : (cellsOf[c] = visibleWidth(String.fromCharCode(c)));
		}
	}
	return n;
}

/** A full-width line on a panel background. Truncation and some helpers emit
 * a full reset (ESC[0m), which would drop the background mid-line, so the
 * background is re-applied after every reset. On the dock's own theme (dockTheme) the text colour is its own too:
 * re-applied after every reset and every return to the default foreground (ESC[39m), the background after every
 * return to the default background (ESC[49m). */
export function tint(theme: any, line: string, width: number, vw = textWidth(line)): string {
	const ink: string = theme.ink ?? "";
	const back = theme.bg("customMessageBg", "\u0000").split("\u0000")[0], on = back + ink;
	// `vw`: the line's width (the caller's where it knows it); truncateToWidth only where needed (most of a tall board's
	// frame otherwise)
	const fitted = vw > width ? pad(truncateToWidth(line, width), width) : line + " ".repeat(width - vw);
	return on + fitted.replace(/\x1b\[(0?|39|49)m/g, (m: string) => m + (m === "\x1b[39m" ? ink : m === "\x1b[49m" ? back : on)) + (ink ? "\x1b[39m" : "") + "\x1b[49m";
}

// -- the dock's own colours -------------------------------------------------------------------------------------------

export type Rgb = [number, number, number];

/** The dock paints with these, not Pi's theme (whose message background and accent made it purple): a near-black
 * panel, orange for titles and focus, green / yellow / red for sure / middling / unsure, violet only for the lens, a muted blue only for the engram's help. */
export const COLOURS: Record<"bg" | "text" | "dim" | "faint" | "rule" | "accent" | "good" | "middling" | "bad" | "lens" | "lensIn" | "help", Rgb> = {
	bg: [11, 13, 16], text: [215, 219, 226], dim: [123, 132, 146], faint: [77, 85, 98], rule: [44, 50, 60],
	accent: [240, 162, 58], good: [127, 201, 143], middling: [227, 195, 92], bad: [233, 138, 128], lens: [183, 156, 242],
	lensIn: [60, 46, 98],   // behind the token that went into the lens's pass (the reply strip; telescope's "in:")
	help: [110, 150, 200],  // a token the engram helped (lookup.ts)
};
type Colour = keyof typeof COLOURS;

/** Pi's colour names, as the dock's code (and the renderers it shares with Pi's theme) asks for them, to the dock's. */
const AS: Record<string, Colour> = { accent: "accent", success: "good", warning: "middling", error: "bad", muted: "dim", dim: "dim",
	faint: "faint", rule: "rule", lens: "lens", lensIn: "lensIn", text: "text", customMessageBg: "bg" };

/** An escape sequence for `rgb` as foreground (or background): 24-bit, or the nearest of 256 colours where the
 * terminal has no more (pi-tui's own conversion). */
export function colourCode(rgb: Rgb, mode: "truecolor" | "256color", bg = false): string {
	if (mode === "truecolor") return `\x1b[${bg ? 48 : 38};2;${rgb[0]};${rgb[1]};${rgb[2]}m`;
	return (bg ? backgroundAnsi : foregroundAnsi)(rgbColor(rgb[0], rgb[1], rgb[2]), mode);
}

/** Raw bold and inverse: drawn the same on any theme (the cursor, the edited word, the title in focus). */
export const ansiBold = (s: string) => `\x1b[1m${s}\x1b[22m`;
export const ansiInverse = (s: string) => `\x1b[7m${s}\x1b[27m`;

/** The dock's theme: Pi's theme interface (fg, bg, bold, …) in COLOURS, whatever Pi's theme is. `pi` says only
 * whether the terminal has 24-bit colour (getColorMode). `ink` is the text colour's escape sequence (tint). */
export function dockTheme(pi?: any) {
	let mode: "truecolor" | "256color" = "truecolor";
	try {
		if (pi?.getColorMode?.() === "256color") mode = "256color";
	} catch {}
	const fg = {} as Record<Colour, string>, bg = {} as Record<Colour, string>;
	for (const k of Object.keys(COLOURS) as Colour[]) {
		fg[k] = colourCode(COLOURS[k], mode);
		bg[k] = colourCode(COLOURS[k], mode, true);
	}
	return {
		dock: true,
		ink: fg.text,
		fg: (c: string, s: string) => `${fg[AS[c] ?? "text"]}${s}\x1b[39m`,
		bg: (c: string, s: string) => `${bg[AS[c] ?? "bg"]}${s}\x1b[49m`,
		bold: ansiBold,
		italic: (s: string) => `\x1b[3m${s}\x1b[23m`,
		underline: (s: string) => `\x1b[4m${s}\x1b[24m`,
		strikethrough: (s: string) => `\x1b[9m${s}\x1b[29m`,
		inverse: ansiInverse,
		getColorMode: () => mode,
	};
}
export type DockTheme = ReturnType<typeof dockTheme>;
/** `s` in the theme's bold; as it is on a theme without one. */
export const bold = (th: Partial<Pick<DockTheme, "bold">>, s: string) => (th.bold ? th.bold(s) : s);

/** A theme as the job lines and the transcript entries draw with it: Pi's, or plain. */
export type Ink = { fg(colour: string, s: string): string; bold(s: string): string };

/** A part of a line: its plain text, its colour (the theme's, or "bold") and, for a part that gives way, its text
 * in at most `w` columns (`fit`). */
export interface Part { t: string; c?: string; fit?: (w: number) => string }
/** A part that is cut to the room it has, dropped under `least` columns. */
export const cutTo = (t: string, least = 2) => (w: number) => (w < least ? "" : cut(t, w));

/** `parts` in `width` columns, then coloured (plain text is cut, colour goes on after). Every part's text is put on
 * one line and made safe first (flat), whoever gave it. The parts in `order` give way, the first as far as it must
 * first: each starts at its least (fit(0)) and, the last in `order` first, takes back what room is left; then the line
 * is cut, should what never gives way be wider still. */
export function partsLine(parts: Part[], order: number[], width: number, th: Ink): string {
	const fit = (i: number, w: number) => flat(parts[i].fit!(w));
	const ts = parts.map((p, i) => (p.fit ? fit(i, 0) : flat(p.t)));
	let room = width - ts.reduce((n, t) => n + visibleWidth(t), 0);
	for (const i of [...order].reverse()) {
		const has = visibleWidth(ts[i]), all = flat(parts[i].t);
		ts[i] = visibleWidth(all) <= room + has ? all : fit(i, room + has);
		room -= visibleWidth(ts[i]) - has;
	}
	let left = width;
	return ts.map((t, i) => {
		const w = visibleWidth(t), s = w <= left ? t : left > 0 ? cut(t, left) : "";
		left -= w;
		const c = parts[i].c;
		return !s || !c ? s : c === "bold" ? th.bold(s) : th.fg(c, s);
	}).join("");
}

/** Parts that give way, each cut to the room it has (cutTo) unless it has a `fit` of its own. */
export const giving = (parts: Part[]): Part[] => parts.map((p) => (p.fit ? p : { ...p, fit: cutTo(flat(p.t)) }));

/** `parts` in `width` columns, the last giving way first (giving). */
export const lineOf = (parts: Part[], width: number, th: Ink) => partsLine(giving(parts), parts.map((_, i) => parts.length - 1 - i), width, th);

/** The colours of a command form and of the words after it, wherever a command is drawn (help, an entry's last line). */
export const COMMAND = { form: "accent", words: "muted" } as const;

/** An entry's last line, indented under its head: `xs` alternate a command form and its words (COMMAND),
 * ["/steer fluffy", " uses dose 0.4"]. */
export const commandLine = (xs: string[], width: number, th: Ink) =>
	lineOf([{ t: "  " }, ...xs.map((t, i) => ({ t, c: i % 2 ? COMMAND.words : COMMAND.form }))], width, th);

/** Lines drawn one column in, as the job widget and the transcript entries are: `render` is given the width less that
 * column, and each line is cut to the width. */
export const oneIn = (width: number, render: (width: number) => string[]) =>
	render(Math.max(1, width - 1)).map((l) => (visibleWidth(l) < width ? ` ${l}` : truncateToWidth(` ${l}`, width, "")));

/** What the dock counts over a whole reply: the probabilities the model gave its own tokens summed (`sum`; a token
 * below the recorded top-k counts as 0), the unsure tokens' indices, the off-top count and the token ending the
 * thinking (state.ts thinkEnd; -1: none yet). Kept by token array and carried on as tokens are appended, so a long reply
 * costs what a short one does; an array cut or changed under it (its `n`th token not `last`) is counted afresh. */
export interface Counts { n: number; last?: object; sum: number; unsure: number[]; off: number; think: number }
const counted = new WeakMap<object, Counts>();
export function replyCounts(toks: readonly { text: string; p?: number | null; alts?: [string, number][] }[]): Counts {
	let t = counted.get(toks);
	if (!t || t.n > toks.length || (t.n > 0 && toks[t.n - 1] !== t.last))
		counted.set(toks, (t = { n: 0, sum: 0, unsure: [], off: 0, think: -1 }));
	for (; t.n < toks.length; t.n++) {
		const x = toks[t.n], p = x.p ?? 0;
		t.sum += p;
		if (p < UNSURE) t.unsure.push(t.n);
		if (x.alts && offTop(p, x.alts)) t.off++;
		if (t.think < 0 && x.text.includes("</think>")) t.think = t.n;
	}
	t.last = toks[t.n - 1];
	return t;
}
/** An off-top pick: chosen under UNSURE while the model put over UNSURE on another token (`alts`, likeliest
 * first), so sampling, not hesitation, chose it. */
export function offTop(p: number | null, alts: [string, number][]): boolean {
	return (p ?? 0) < UNSURE && (alts[0]?.[1] ?? 0) > UNSURE;
}

export function bar(frac: number, w: number): string {
	const cells = Math.max(0, Math.min(1, frac)) * w;
	const full = Math.floor(cells);
	const part = " ▏▎▍▌▋▊▉"[Math.floor((cells - full) * 8)];
	return ("█".repeat(full) + (full < w ? part : "")).trimEnd() || "▏";
}

/** Wrap tokens into lines of width w, each token coloured by confidence.
 * Returns the lines and the line index holding token `sel`. `input`: the token that went into the
 * lens's pass for `sel` (the one before it), on the lens's dark violet. `edit`: the word edit's token (editMark), ▸
 * before its word: changed, the word bold on an orange block, the ▸ in its confidence colour; unchanged, a faint ▸.
 * `paint`: each token's colours another way (engram inspection's, lookup.ts lookupPaint) instead of its confidence
 * (null for a token: that one as ever); `cursor`: it draws the
 * token `sel` too (no inverse). The board passes only the tokens its strip can show (dock.ts). `ends`: each line's last
 * token (-1 before the first). */
export function wrapTokens(theme: any, toks: { text: string; p: number | null }[], w: number, sel = -1, input = -1,
	edit?: { k: number; changed: boolean }, paint?: { key: string; cursor?: boolean; at: (k: number, s: string) => string | null }) {
	const inv = paint?.cursor ? -1 : sel;   // the token drawn inverse: the cursor, unless the paint draws it
	const lines: string[] = [];
	let cur = "", curW = 0, selLine = -1, last = -1;
	const ends: number[] = [];   // each line's last token
	for (let k = 0; k < toks.length; k++) {
		const t = toks[k];
		const pieces = t.text.split("\n");
		// The edited token's ▸: faint where the edit left it unchanged; changed, in its confidence colour beside its block
		// (under the cursor or the lens input's violet: orange, the block's colour).
		let mark = edit?.k === k ? theme.fg(!edit.changed ? "faint" : k === inv || k === input ? "accent" : confColor(t.p), "▸") : "";
		pieces.forEach((piece, pi) => {
			if (pi > 0) {
				lines.push(cur);
				ends.push(last);
				cur = "";
				curW = 0;
			}
			let shown = piece.replace(/\t/g, " ");
			if (!shown && (k === sel || k === input)) shown = "↵";
			while (shown || (mark && pi === pieces.length - 1)) {   // a token of newlines only: its ▸ on the line after
				const mw = mark ? 1 : 0;
				if (curW + mw + textWidth(shown) > w && curW > 0) {
					lines.push(cur);
					ends.push(last);
					cur = "";
					curW = 0;
				}
				const room = w - curW - mw;
				const head = textWidth(shown) > room ? truncateToWidth(shown, room, "") : shown;
				const own = paint ? paint.at(k, head) : null, painted = own !== null;   // null: this token as ever
				const styled = own ?? theme.fg(confColor(t.p), head);
				const lead = head.length - head.trimStart().length; // the word, not its space
				if (edit?.k === k) {
					// The edited token: ▸ between its space (plain) and its word; changed, the word bold, dark on an orange block
					// (the cursor's inverse and the lens input's violet win there); unchanged, the word as below.
					// `paint` colours its space and word as any token's (the edit's block still wins).
					const c = confColor(t.p), s = head.slice(lead), b = edit.changed && s ? ansiBold(s) : s;
					const ink = (x: string) => (painted ? paint!.at(k, x) ?? theme.fg(c, x) : theme.fg(c, x));
					cur += (lead ? ink(head.slice(0, lead)) : "") + mark + (!s ? "" : k === inv ? ansiInverse(b)
						: k === input ? theme.bg("lensIn", ink(b)) : edit.changed ? theme.bg("accent", theme.fg("customMessageBg", b)) : ink(b));
					mark = "";
				} else cur += k === inv ? ansiInverse(head)
					: k === input ? theme.bg("lensIn", styled)   // the token that went in (the lens's pass): not the inverse
					: styled;
				curW += mw + textWidth(head);
				last = k;
				if (k === sel) selLine = lines.length;
				shown = shown.slice(head.length);
			}
		});
	}
	lines.push(cur);
	ends.push(last);
	return { lines, selLine, ends };
}

/** LENS: the earliest read layer from which every deeper read layer's top-1 is
 * the chosen token `id`; null when the last one differs. */
export function lensSettle(lens: [number, number, string][], id: number): number | null {
	let settle: number | null = null;
	for (let k = lens.length - 1; k >= 0 && lens[k][1] === id; k--) settle = lens[k][0];
	return settle;
}

/** `ls` under `n - ls.length` blank lines: a key area's height kept whatever it holds. */
export const padAbove = (n: number, ls: string[]) => [...Array<string>(Math.max(0, n - ls.length)).fill(""), ...ls];
/** A note in the warning colour, wrapped into `n` lines `w` wide (the last cut, marked, where it runs over). */
export function noteLines(th: any, note: string, w: number, n: number): string[] {
	const ls = wrapTextWithAnsi(note, w);
	return (ls.length > n ? [...ls.slice(0, n - 1), cut(ls[n - 1] + " …", w)] : ls).map((l) => th.fg("warning", fit(l, w)));
}

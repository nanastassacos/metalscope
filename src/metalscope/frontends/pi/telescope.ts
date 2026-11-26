/**
 * The dock's LENS: one token looked at through the lens, drawn as a telescope. Under the turret (which lens: J-lens or logit), the eyepiece shows the focused layer's
 * three likeliest words, then the tube runs over the layers read, deep at the top and shallow at the bottom, in three
 * sections that narrow (`tube`). Each line shows its layer's top word and how likely the written token was there.
 *
 * `telescope` builds one frame, pure: the swap between lenses (dock.ts) is a few frames of it with the words fogged
 * (`fog`), the turret part way (`turret`) and the tube cleared from the deep end (`sharp`); LENS's - + a few
 * with the tube's layers (`want`) coming in or going out (`stretchFrames`).
 */

import { visibleWidth } from "@earendil-works/pi-tui";
import { ansiInverse, bold, centre, clean, cut, lensSettle, lpadTo, padTo, showWord } from "./draw.ts";
import { pushLayer } from "./steer.ts";
import { type LensTop, type TokenEvent, lensP } from "./state.ts";

export type LensName = "jlens" | "logit";
type Lens = [number, number, string][];
type LensTok = Pick<TokenEvent, "id" | "text" | "lens" | "lens_by" | "lens_logit" | "top" | "top_logit" | "n_layers">;

/** The lens LENS looks through for `t`: the logit lens where `view` asks for it and the token has one, else its
 * main lens (the J-lens where one was on). `top1` each read layer's top word; `top` the top 3s, where sent. */
export function lensOf(t: LensTok, view: LensName | undefined): { by?: LensName; top1: Lens; top?: LensTop } {
	if (view === "logit" && t.lens_logit?.length) return { by: "logit", top1: t.lens_logit, top: t.top_logit };
	return { by: t.lens_by, top1: t.lens ?? [], top: t.top };
}

/** Whether `t` recorded both lenses (s swaps them). */
export const bothLenses = (t: LensTok) => !!(t.lens?.length && t.lens_logit?.length);
/** Every layer either lens read for `t`, deep first, and the model's depth: what the layout is drawn from, so s
 * swapping the lens changes no line's place. */
export const readLayers = (t: LensTok) => [...new Set([...(t.lens ?? []), ...(t.lens_logit ?? [])].map(([l]) => l))].sort((a, b) => b - a);
export const depthOf = (t: LensTok) => Math.max(t.n_layers ?? 0, ...readLayers(t).map((l) => l + 1));

/** The eyepiece's words for `t` at layer `fl`: the top 3 where sent (p each), else the top word alone. */
export function eyepiece(t: LensTok, view: LensName | undefined, fl: number): { id: number; text: string; p?: number | null }[] {
	const { top1, top } = lensOf(t, view), r = top ? top.l.indexOf(fl) : -1;
	if (r < 0) return top1.filter(([l]) => l === fl).map(([, id, text]) => ({ id, text }));
	return [0, 1, 2].map((k) => ({ id: top!.id[r * 3 + k], text: top!.w[r * 3 + k], p: lensP(top!.p[r * 3 + k]) })).filter((x) => x.id >= 0);
}

/** A line of the tube: its layers (`ls`, deep first) in section `sec` (0 the widest, 2 the narrowest). */
export interface TubeLine {
	ls: number[];
	sec: 0 | 1 | 2;
}

/** The most lines the tube takes for a model of `n` layers: 21 for 48 (a quarter of the depth a layer a line, a
 * quarter two a line, the shallow half in three). */
export function tubeCap(n: number): number {
	const q = Math.round(n / 4), rest = n - 2 * q;
	return q + Math.ceil(q / 2) + (rest > 0 ? Math.ceil(rest / Math.ceil(rest / 3)) : 0);
}

/** The tube's lines for `layers`, deep first, in three narrowing sections at any length: about half the lines wide, a
 * quarter narrower, a quarter narrowest. Up to `cap` lines a layer a line; past it the deep half still a layer a line
 * and the shallower layers sharing lines: the middle section's lines `a` each, the narrowest's about twice that, any
 * remainder on the shallowest lines, so a line never holds fewer layers than one deeper. */
export function tube(layers: number[], cap: number): TubeLine[] {
	const deep = [...new Set(layers)].sort((a, b) => b - a), L = Math.min(deep.length, Math.max(1, cap));
	const s0 = Math.max(1, Math.round(L / 2)), s1 = Math.max(1, Math.round(L / 4)), s2 = Math.max(1, L - s0 - s1);
	const rest = deep.length - s0, a = Math.max(1, Math.floor(rest / (s1 + 2 * s2))), out: TubeLine[] = [];
	let i = 0;
	const take = (k: number, sec: 0 | 1 | 2) => {
		const j = Math.min(deep.length, i + Math.max(0, k));
		out.push({ ls: deep.slice(i, j), sec });
		i = j;
	};
	for (let k = 0; k < s0; k++) take(1, 0);
	for (let k = 0; k < s1; k++) take(a, 1);
	for (let k = 0; k < s2; k++) take(Math.floor((deep.length - i) / (s2 - k)), 2);
	return out.filter((x) => x.ls.length);
}
/** Token `t`'s tube: over `want` (the step's layers) where given, else every layer either lens read; at most `cap`
 * lines (the box it is drawn in), else tubeCap's. */
export const tubeOf = (t: LensTok, want?: number[], cap?: number) => tube(want ?? readLayers(t), cap ?? tubeCap(depthOf(t)));
/** Of a lens's read `layers`, those on the tube over `want` (all where none is): the eyepiece and ↑/↓ focus among them. */
export const onTube = (layers: number[], want?: number[]) =>
	want && layers.some((l) => want.includes(l)) ? layers.filter((l) => want.includes(l)) : layers;
export const INDENT = [0, 4, 8];
/** The telescope's lines over tube `lines`, room enough (as `telescope` draws them; the board's LENS is sized on it):
 * the token's line, the eyepiece (5) and the written word, the tube with a joint where a section narrows, its bottom. */
export const scopeHeight = (lines: TubeLine[]) => 7 + lines.length + lines.filter((b, i) => i && lines[i - 1].sec !== b.sec).length + 1;
/** The most lines the tube over `want` takes for the telescope to fit `room` lines (scopeHeight): a layer a line where
 * that fits, else the shallow layers sharing lines; undefined where not even that fits. */
export function fitCap(want: number[], room: number): number | undefined {
	for (let c = new Set(want).size; c >= 1; c--) if (scopeHeight(tube(want, c)) <= room) return c;
	return undefined;
}

/** The narrowest dock the telescope fits: the shallow section's word column needs a few characters. */
export const MIN_WIDTH = 36;
/** The fewest lines it is drawn in: the token, the eyepiece (5), the written word, one line of the tube and the line
 * saying how many layers are left out. Fewer, and one line says so instead. */
export const MIN_LINES = 9;

/** The focused layer: the read layer nearest `asked` (a deeper one on a tie); without one asked, the settle layer
 * when read, else the deepest. null without read layers. */
export function focusLayer(layers: number[], settle: number | null, asked?: number): number | null {
	if (!layers.length) return null;
	const deep = [...layers].sort((a, b) => b - a);
	if (asked !== undefined) return deep.reduce((a, b) => (Math.abs(b - asked) < Math.abs(a - asked) ? b : a), deep[0]);
	return settle !== null && layers.includes(settle) ? settle : deep[0];
}

/** The read layer one step deeper (dir 1) or shallower (-1) than `at`, clamped. */
export function stepLayer(layers: number[], at: number, dir: 1 | -1): number {
	const up = [...layers].sort((a, b) => a - b), k = up.indexOf(at);
	return up[Math.max(0, Math.min(up.length - 1, (k < 0 ? 0 : k) + dir))];
}

// -- text by graphemes (the rest of the text helpers: draw.ts) -----------------------------------------------------

const GRAPHEMES = new Intl.Segmenter(undefined, { granularity: "grapheme" });
/** The end of `s` that fits `w` columns, whole characters (a field being typed shows its end). */
function tail(s: string, w: number): string {
	let out = "";
	for (const { segment } of [...GRAPHEMES.segment(s)].reverse()) {
		if (visibleWidth(segment + out) > w) break;
		out = segment + out;
	}
	return out;
}

/** Words seen through a fogged lens: each character as wide as it shows (a grapheme: combining marks, ZWJ sequences
 * and modifiers are one) becomes that many ▒ (level 1) or ░ (2), spaces kept; exactly as wide as `s`. */
export function fog(s: string, lvl: number): string {
	if (!lvl) return s;
	const block = lvl === 1 ? "▒" : "░", want = visibleWidth(s);
	let out = "", n = 0;
	for (const { segment } of GRAPHEMES.segment(s)) {
		const cw = visibleWidth(segment);
		out += /^\s+$/.test(segment) ? " ".repeat(cw) : block.repeat(cw);
		n += cw;
	}
	return n === want ? out : n < want ? out + " ".repeat(want - n) : [...out].slice(0, want).join("");
}
/** A probability as LENS shows it: "9%", "<1%", "·" for none (outside the layer's top 10). */
export const pc = (p: number | null) => (p === null ? "·" : p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`);

// -- one frame -------------------------------------------------------------------------------------------------------

/** How one frame is drawn: `view` the lens asked for (s), `focus` the layer asked for (↑/↓), and during a swap the
 * words fogged (`fog` 1 or 2), only the deepest `sharp` tube lines clear, the turret at `turret` (0 J-lens .. 4 logit). */
export interface Frame {
	view?: LensName;
	focus?: number;
	want?: number[];   // the tube's layers (the step's): those no lens read for the token faint, a ·
	cap?: number;   // the most tube lines (the box the tube is drawn in; tubeCap's where unset)
	steer?: number[];   // the layers a steer pushes at: ◆ on the tube there (and " steer" beside it, with room)
	edit?: { row: number; field: string | null; band?: [number, number] };   // the eyepiece word being edited: null untouched, "" removed, else typed over; `band` where it acts
	input?: string | null | false;   // the token that went in (dock.ts inputOf): null the prompt's end; false (or unset) not streamed
	fog?: number;
	sharp?: number;
	turret?: number;
}

/** The turret: "J-lens ◉────○ logit", the lens in use bright, the other faint; `at` 0 (J-lens) .. 4 (logit); `short`
 * "J-lens ◉─○ logit" (LENS's step needs the room). One lens recorded: its name alone. */
export function turret(theme: any, t: LensTok, view: LensName | undefined, at?: number, short = false): string {
	const by = lensOf(t, view).by;
	const bright = (s: string) => theme.fg("lens", bold(theme, s));
	if (!bothLenses(t)) return bright(by === "logit" ? "logit" : by === "jlens" ? "J-lens" : "lens");
	const step = at ?? (by === "logit" ? 4 : 0);
	const rail = short ? "○─○" : "○────○", track = [...rail].map((c, i) => (i === Math.round((step * (rail.length - 1)) / 4) ? "◉" : c)).join("");
	return (step < 2 ? bright("J-lens") : theme.fg("faint", "J-lens")) + " " + theme.fg("lens", track) + " " + (step > 2 ? bright("logit") : theme.fg("faint", "logit"));
}

/** The lines under the turret for token `t` (reply token `sel`), `w` columns wide, at most `room` lines: the token
 * and where it settles, the eyepiece (the focused layer's top 3), the written token, the tube. When the tube does not
 * fit, its shallow end goes first and a dim line says how many layers are left out. `focus`: the layer focused. A
 * token no lens read: the outline over the step's layers (`want`), the eyepiece empty, the tube's lines faint; without
 * them, one line says so. */
export function telescope(theme: any, t: LensTok, sel: number, w: number, room: number, f: Frame = {}): { lines: string[]; focus: number | null } {
	const dim = (s: string) => theme.fg("dim", s), acc = (s: string) => theme.fg("lens", s);   // the tube: the lens colour
	if (w < MIN_WIDTH) return { lines: [dim(`needs ${MIN_WIDTH} columns`)], focus: null };
	if (room < MIN_LINES) return { lines: room > 0 ? [dim(`needs ${MIN_LINES} lines: a taller terminal shows the lens`)] : [], focus: null };
	const { top1, top } = lensOf(t, f.view);
	if (!top1.length && !f.want) return { lines: [dim("no lens read for this token")], focus: null };
	const n = depthOf(t);
	const marks = (f.steer ?? []).filter((l) => l >= 0 && l < n);
	const label = marks.length > 0 && w - 6 >= MIN_WIDTH;   // " steer" beside the ◆: the telescope six columns narrower
	if (label) w -= 6;
	const lvl = f.fog ?? 0, sharp = f.sharp ?? Infinity;
	const lines = tubeOf(t, f.want, f.cap), read = new Set(readLayers(t)), layers = onTube(top1.map(([l]) => l), f.want);
	const byLayer = new Map(top1.map(([l, id, text]) => [l, { id, text }]));
	const settle = lensSettle(top1, t.id);
	const fl = focusLayer(layers, settle, f.focus);
	const wrote = showWord(t.text);
	const out: string[] = [];
	out.push(dim(`#${sel} "${cut(wrote, 16)}"`) + theme.fg("faint", fl === null ? "" : settle !== null ? ` · settles L${settle}/${n}` : " · never settles"));

	// The eyepiece: the focused layer's three likeliest tokens (its top word alone where the top 3 were not sent).
	const title = fl === null ? "" : ` out of L${fl} `, wordW = w - 10;
	out.push(acc("╭─" + title + "─".repeat(Math.max(0, w - 3 - visibleWidth(title))) + "╮"));
	// The word being edited (f.edit): ▸ and the word inverse; removed, struck and "removed"; typed over, struck → the new.
	// Where it acts, after the cursor when it fits: a swap's push layer ("at L26"), a removal's band ("at L15-36").
	const eye = fl === null ? [] : eyepiece(t, f.view, fl), mark = (s: string) => theme.fg("accent", bold(theme, s)), band = f.edit?.band;
	const where = (field: string) => (!band ? "" : field ? ` at L${pushLayer(band)}` : ` at L${band[0]}-${band[1]}`);
	for (let k = 0; k < 3; k++) {
		const e = eye[k], word = e ? showWord(e.text) : "", ps = fog(lpadTo(e?.p === undefined ? "" : pc(e.p), 5), lvl);
		const field = f.edit?.row === k ? f.edit.field : undefined, old = cut(word, wordW >> 1);
		let cells: string;
		if (field === undefined) {
			const shown = fog(cut(word, wordW), lvl);
			cells = " " + (e?.id === t.id ? theme.fg("success", bold(theme, shown)) : k ? dim(shown) : shown);
		} else if (field === null) cells = mark("▸" + ansiInverse(cut(word, wordW)));
		else {
			// The field as the lens words are shown (a fill can hold a newline), its end kept by width.
			const typed = field ? clean(showWord(field)) : "", tag = where(field);
			const room = wordW - visibleWidth(old) - (field ? 4 : 9);
			const at = visibleWidth(typed) + visibleWidth(tag) <= room ? dim(tag) : "";
			cells = mark("▸") + dim(`\x1b[9m${old}\x1b[29m`) + mark(field ? " → " : " removed") +
				bold(theme, tail(typed, wordW - visibleWidth(old) - 4)) + ansiInverse(" ") + at;
		}
		out.push(acc("│") + cells + " ".repeat(Math.max(0, wordW + 1 - visibleWidth(cells))) + dim(ps) + "  " + acc("│"));
	}
	out.push(acc("╰" + "─".repeat(w - 2) + "╯"));
	out.push(theme.fg("success", bold(theme, centre("→ " + wrote, w))));

	// The tube, deep first: [line, how many layers it holds] (joints and the cap hold none). A line spans the layers
	// from its shallowest to the line above's (the last down to 0): a steer's marks go there.
	const tube: [string, number][] = [];
	lines.forEach((b, i) => {
		const ind = INDENT[b.sec], inner = w - 2 * ind - 2, mid = inner - 14;
		if (i && lines[i - 1].sec !== b.sec) {
			const pi = INDENT[lines[i - 1].sec], step = ind - pi;
			tube.push([" ".repeat(pi) + acc("╰" + "─".repeat(step - 1) + "╮" + " ".repeat(inner) + "╭" + "─".repeat(step - 1) + "╯"), 0]);
		}
		const lo = i === lines.length - 1 ? 0 : Math.min(...b.ls), hi = i ? Math.min(...lines[i - 1].ls) - 1 : n - 1;
		const inLine = layers.filter((l) => b.ls.includes(l));
		const l = fl !== null && inLine.includes(fl) ? fl : inLine.length ? Math.max(...inLine) : null;
		const s = l !== null && l === settle;
		const lab = (s ? "▸" : " ") + padTo(`L${l ?? b.ls[0]}` + (b.ls.length > 1 ? ` +${b.ls.length - 1}` : ""), 7);
		let body: string;
		// Not read by this lens (the other's alone, or wanted, not read yet): the label, the row blank.
		if (l === null) body = (b.ls.some((x) => read.has(x)) ? dim : (s: string) => theme.fg("faint", s))(lab + " ".repeat(mid + 6));
		else {
			const fogged = i >= sharp ? lvl : 0, at = byLayer.get(l)!, hit = at.id === t.id;
			const tr = top ? top.l.indexOf(l) : -1;
			const p = tr >= 0 ? pc(lensP(top!.wp[tr])) : "";
			const word = fog(centre(showWord(at.text), mid), fogged), ps = fog(lpadTo(p, 4), fogged);
			body = l === fl ? acc(bold(theme, ansiInverse(`${lab} ${word} ${ps}`)))
				: (s ? theme.fg("success", bold(theme, lab)) : dim(lab)) + " " + (hit ? theme.fg("success", s ? bold(theme, word) : word) : dim(word)) + " " +
					(hit ? theme.fg("success", ps) : theme.fg("faint", ps));
		}
		const steered = marks.some((x) => x >= lo && x <= hi);
		const edge = steered ? theme.fg("accent", bold(theme, "◆") + (label ? " steer" : "")) : acc("│");
		tube.push([" ".repeat(ind) + acc("│") + body + edge, b.ls.length]);
	});
	// The bottom: the token that went in. The lens rows of reply token j are read at the forward pass fed token j - 1
	// (livestream: token j is announced with the dist of the step whose input was j - 1; the first, the prompt's end).
	const ind = INDENT[lines.at(-1)?.sec ?? 2], inner = w - 2 * ind - 2;
	const went = f.input === null ? "in: end of prompt" : typeof f.input !== "string" ? "in: not streamed"
		: `in: "${cut(clean(showWord(f.input)), Math.max(1, inner - 10))}"`;
	const lab = ` ${cut(went, Math.max(1, inner - 4))} `, half = Math.floor((inner - visibleWidth(lab)) / 2);
	tube.push([" ".repeat(ind) + acc("╰" + "─".repeat(Math.max(0, half))) + theme.fg("lens", bold(theme, lab)) +
		acc("─".repeat(Math.max(0, inner - visibleWidth(lab) - half)) + "╯"), 0]);

	// Room for the tube: what is left under the eyepiece (at least two lines: room >= MIN_LINES).
	const avail = room - out.length;
	if (tube.length <= avail) out.push(...tube.map(([x]) => x));
	else {
		const keep = tube.slice(0, avail - 1), left = tube.slice(avail - 1).reduce((a, [, k]) => a + k, 0);
		out.push(...keep.map(([x]) => x), dim(`… ${left} shallower layer${left === 1 ? "" : "s"}`));
	}
	return { lines: out, focus: fl };
}

/** LENS's - + (about 70 ms a frame): the tube from layers `a` to `b`, those coming in a share a frame (deep
 * first), those going out a share a frame (shallow first). The last frame is `b`. */
export function stretchFrames(a: number[], b: number[]): Frame[] {
	const keep = a.filter((l) => b.includes(l)), add = b.filter((l) => !a.includes(l)).sort((x, y) => y - x);
	const drop = a.filter((l) => !b.includes(l)).sort((x, y) => x - y);
	return [1, 2, 3, 4].map((k) => ({ want: [...keep, ...add.slice(0, Math.round((add.length * k) / 4)), ...drop.slice(Math.round((drop.length * k) / 4))] }));
}

/** The swaps' schedule, a frame each (about 70 ms): the old view (`next` false) fogs, two levels, then the new one,
 * fogged, clears from the deep end (`clear`: the deepest share clear); the last frame clear. The lens swap
 * (swapFrames) and STREAM's view switch (board.ts streamFrames). */
export const SWAP_FOG: { next: boolean; fog?: number; clear?: number }[] = [
	{ next: false, fog: 1 }, { next: false, fog: 2 }, { next: false, fog: 2 }, { next: true, fog: 2 }, { next: true, fog: 2 },
	{ next: true, fog: 1, clear: 0.25 }, { next: true, fog: 1, clear: 0.5 }, { next: true, fog: 1, clear: 0.8 }, { next: true }];
/** The lens swap's frames (SWAP_FOG), from lens `a` to lens `b`: the old words fog, the turret's ◉ slides across in the
 * first five, then the new lens's tube clears from the deep end (`sharp`: its `nBands` lines' clear share). The last
 * frame is the new lens, sharp. */
export function swapFrames(a: LensName, b: LensName, nBands: number): Frame[] {
	const dir = b === "logit" ? 1 : -1, t0 = dir > 0 ? 0 : 4;
	return SWAP_FOG.map((f, i) => ({ view: f.next ? b : a, ...(f.fog ? { fog: f.fog, sharp: Math.round(nBands * (f.clear ?? 0)) } : {}),
		turret: t0 + Math.min(i, 4) * dir }));
}

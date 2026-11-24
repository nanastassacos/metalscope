/**
 * The Steer page's data (board.ts draws it): the served model's learned concepts ranked by the chat test, each
 * concept's doses, and the saved test chats of a concept with their thinking hidden and their markdown drawn by Pi's
 * own renderer. Applying (enter, x) is /steer's own handler, run by the dock through its host (widget.ts); nothing here
 * steers.
 *
 * The data is web.py's /api/steering and /api/steering/NAME/chats, shaped by `parseSteering` and `parseExamples`.
 */

import { Markdown, type MarkdownTheme, truncateToWidth } from "@earendil-works/pi-tui";
import { ranAway } from "./concept.ts";
import { ansiBold, clean } from "./draw.ts";

/** One measured dose of a concept: the chat test's good turns (null: not chat-tested) and the grader's mean scores
 * (0-3; null where nothing was graded). */
export interface DoseRow {
	dose: number;
	good: number | null;
	turns: number;
	style: number | null;
	sense: number | null;
	runaway?: number; // its replies ran this many times the plain length (ChatStrength's)
}

export interface SteerConcept {
	name: string;
	title: string;
	description: string;
	layer: number | null; // where /steer pushes it (the concept list's steer layer, else the calibration's); null: none
	chat: boolean; // chat-tested (a calibration that fits its profile)
	stale: boolean; // its chat test predates the profile (so chat is false)
	good: number; // good turns at the picked dose, the most of any dose tried (0 without a chat test)
	turns: number; // turns per dose (18)
	pick: number; // the chat test's dose (0 without one)
	doses: DoseRow[]; // ascending, no steering (0) first
	rank?: number; // 1.. among the chat-tested, by the ranking (rankConcepts)
}

export interface SteerList {
	model: string; // the served model it was asked for
	concepts: SteerConcept[]; // ranked
	checkpoint?: string; // the checkpoint the concepts are learned on (a delete removes them there, for every served copy)
}

const num = (x: any): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);
// The server's strings (a pulled concept's title, its test chats) cleaned as the model's tokens are.
const str = (x: any): string => (typeof x === "string" ? clean(x) : "");

/** /api/steering's answer as the Steer page shows it, ranked (rankConcepts). `layers`: the steer layer of each concept as
 * /steer resolves it (the concept list's reader layer; null where it has none). */
export function parseSteering(json: any, model: string, layers: Record<string, number | null> = {}): SteerList {
	const out: SteerConcept[] = [];
	for (const c of Array.isArray(json?.concepts) ? json.concepts : []) {
		if (typeof c?.name !== "string" || !c.name || c.error) continue;
		const ch = c.chat && typeof c.chat === "object" ? c.chat : null;
		const strengths: any[] = ch && Array.isArray(ch.strengths) ? ch.strengths.filter((s: any) => num(s?.dose) !== null) : [];
		let doses: DoseRow[], good = 0, turns = 0, pick: number;
		if (ch && strengths.length) {
			turns = num(ch.turns) ?? num(strengths[0].turns) ?? 0;
			const row = (s: any): DoseRow => ({ dose: s.dose, good: num(s.good) ?? 0, turns: num(s.turns) ?? turns, style: num(s.style), sense: num(s.coherence),
				...(num(s.runaway) !== null ? { runaway: s.runaway } : {}) });
			const plain = ch.plain && typeof ch.plain === "object" ? [{ ...ch.plain, dose: 0 }] : [];
			doses = [...plain, ...strengths].map(row).sort((a, b) => a.dose - b.dose);
			pick = num(ch.default_dose) ?? strengths.reduce((a, b) => ((num(b.good) ?? 0) > (num(a.good) ?? 0) ? b : a)).dose;
			good = doses.find((d) => d.dose === pick)?.good ?? num(ch.good) ?? 0;
		} else {
			doses = [{ dose: 0, good: null, turns: 0, style: null, sense: null }];   // not chat-tested: no steering only
			pick = 0;
		}
		const chat = !!(ch && strengths.length);
		const layer = c.name in layers ? layers[c.name] : num(ch?.layer);
		out.push({ name: c.name, title: str(c.title) || c.name, description: str(c.description).split(/\s+/).join(" ").trim(), layer,
			chat, stale: !chat && !!c.chat_stale, good, turns, pick, doses });
	}
	return { model, concepts: rankConcepts(out), ...(str(json?.checkpoint) ? { checkpoint: str(json.checkpoint) } : {}) };
}

/** Chat-tested first, by good turns at the picked dose, the lower dose, the name; the others after, by name.
 * The chat-tested are numbered (rank). */
export function rankConcepts(cs: SteerConcept[]): SteerConcept[] {
	const sorted = [...cs].sort((a, b) => Number(b.chat) - Number(a.chat) || b.good - a.good ||
		(a.chat ? a.pick - b.pick : 0) || a.name.localeCompare(b.name));
	let k = 0;
	return sorted.map((c) => ({ ...c, rank: c.chat ? ++k : undefined }));
}

/** The top five of the chat test are starred. */
export const starred = (c: SteerConcept) => (c.rank ?? Infinity) <= 5;

/** The dose a concept's cursor starts on: its picked dose (the nearest measured one). */
export function pickIndex(c: SteerConcept): number {
	let k = 0;
	c.doses.forEach((d, i) => { if (Math.abs(d.dose - c.pick) < Math.abs(c.doses[k].dose - c.pick)) k = i; });
	return k;
}

/** A dose as /steer takes it and the Steer page shows it: "0", "0.3", "1.6". */
export const fmt = (d: number) => String(+d.toFixed(3));

// -- the link offer --------------------------------------------------------------------------------------------------

/** A served name (a build) with nothing of its own that has the shape of models with data (GET
 * /scope/api/model/link): l links it to one of them, sharing their concepts, texts and J-lens. `model`: the served
 * name asked for; `candidates`: most data first, never empty. */
export interface LinkOffer {
	model: string;
	shape: { n_layers: number; width: number } | null;
	candidates: { name: string; concepts: number; lenses: number }[];
}

const count = (x: any) => (Number.isInteger(x) && x > 0 ? (x as number) : 0);
const within = (x: any, most: number) => Number.isInteger(x) && x >= 1 && x <= most;
const isShape = (s: any) => !!s && within(s.n_layers, 10_000) && within(s.width, 1e7);
/** A model's name as the server takes one (models.NAME_RE): anything else is dropped, never drawn. */
const MODEL_NAME = /^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/;

/** The server's answer for `model` (a name) as the offer, or null: another model's, linked already, data of its own,
 * no same-shape model with data, or an answer it cannot read. A shape out of reason is left out, the offer kept. */
export function parseOffer(json: any, model: string): LinkOffer | null {
	if (!model || !json || typeof json !== "object" || json.model !== model || json.linked === true || json.own === true || !Array.isArray(json.candidates))
		return null;
	const candidates = json.candidates.filter((c: any) => typeof c?.name === "string" && MODEL_NAME.test(c.name) && (count(c.concepts) || count(c.lenses)))
		.map((c: any) => ({ name: c.name, concepts: count(c.concepts), lenses: count(c.lenses) }));
	return candidates.length ? { model, shape: isShape(json.shape) ? { n_layers: json.shape.n_layers, width: json.shape.width } : null, candidates } : null;
}

/** Whether the host keeps an answer for the session: one that lasts (linked, data of its own, or same-shape models of
 * a known shape); one with no shape yet (a build not loaded) or nothing yet is asked for again at the next page entry. */
export const keepOffer = (json: any) => !!json && typeof json === "object" && isShape(json.shape) &&
	(json.linked === true || json.own === true || (Array.isArray(json.candidates) && json.candidates.length > 0));
/** The model a refused link was found linked to on reload (the server's answer for `model`), else "". */
export const linkedTo = (json: any, model: string) =>
	json?.linked === true && json.model === model && typeof json.checkpoint === "string" && MODEL_NAME.test(json.checkpoint) ? json.checkpoint : "";

/** "14 concepts · 1 lens" (a count of none left out). */
export const dataTag = (concepts: number, lenses: number) =>
	[concepts ? `${concepts} concept${concepts === 1 ? "" : "s"}` : "", lenses ? `${lenses} lens${lenses === 1 ? "" : "es"}` : ""].filter(Boolean).join(" · ");
/** A candidate as the page and the pick list name it: "Qwen3.8-Flash-Next · 14 concepts · 1 lens". */
export const linkLabel = (c: LinkOffer["candidates"][number]) => `${c.name} · ${dataTag(c.concepts, c.lenses)}`;
/** The pick list (l): its title, then keeping it separate first (enter's answer), then each candidate. */
export const LINK_KEEP = "Keep it separate";
export const linkTitle = (model: string) => `Link ${model} to`;
/** The note after a link (POST /scope/api/model/link's answer; `c` the candidate picked, should it not say), and the
 * server's own note (the J-lens not read again, and why) on a line under it. */
export const linkedNote = (res: any, c: LinkOffer["candidates"][number]) => [
	`linked to ${typeof res?.checkpoint === "string" && MODEL_NAME.test(res.checkpoint) ? res.checkpoint : c.name}`,
	dataTag(Number.isInteger(res?.concepts) ? count(res.concepts) : c.concepts, Number.isInteger(res?.lenses) ? count(res.lenses) : c.lenses),
].filter(Boolean).join(" · ") + (typeof res?.note === "string" && res.note.trim() ? `\n${clean(res.note.replace(/\s+/g, " ").trim())}` : "");
/** A refused link found linked on reload (linkedTo). */
export const alreadyLinked = (to: string) => `already linked to ${to}`;
/** LENS's pointer where the model reads the logit lens alone and a candidate has a J-lens; the shorter where narrow. */
export const LENS_LINK = ["no J-lens here · 5 to link a model that has one", "no J-lens · 5 to link a model that has one"];

// -- the test chats --------------------------------------------------------------------------------------------------

export interface Side {
	text: string; // the answer, thinking taken out (answerText)
	style: number | null;
	sense: number | null;
	good: boolean;
	useless?: boolean; // not good only because it wasn't useful
}
export interface ExampleTurn {
	conv: string; // the conversation's title
	n: number; // the turn in it, from 1
	of: number; // its turns
	user: string;
	plain: Side;
	steered: Side;
}
export interface Examples {
	dose: number; // the dose the steered side ran at (the chat test's pick)
	turns: ExampleTurn[]; // every conversation's turns, in order
}

/** A reply as the Steer page shows it: the thinking taken out (a closed <think>…</think>, an unclosed <think> to the end,
 * and everything before a lone </think>), blank runs cut to one blank line. Nothing left: says so. */
export function answerText(t: string): string {
	let s = (t ?? "").replace(/<think>[\s\S]*?<\/think>/g, "");
	s = s.replace(/<think>[\s\S]*$/, "");
	const close = s.lastIndexOf("</think>");
	if (close >= 0) s = s.slice(close + "</think>".length);
	s = s.split("\n").map((l) => l.replace(/\s+$/, "")).join("\n").replace(/\n{3,}/g, "\n\n").trim();
	return s || "(no answer, only thinking)";
}

/** /api/steering/NAME/chats as the Steer page steps through it. */
export function parseExamples(json: any): Examples {
	const side = (s: any): Side => {
		const g = s?.grade && typeof s.grade === "object" ? s.grade : null;
		const why = Array.isArray(s?.why_not) ? s.why_not : [];
		return { text: answerText(str(s?.text)), style: num(g?.style), sense: num(g?.coherence), good: !!s?.good,
			useless: why.length === 1 && why[0] === "not useful" };
	};
	const turns: ExampleTurn[] = [];
	for (const conv of Array.isArray(json?.conversations) ? json.conversations : []) {
		const ts = Array.isArray(conv?.turns) ? conv.turns : [];
		ts.forEach((t: any, i: number) => turns.push({ conv: str(conv.title) || str(conv.arc), n: num(t?.turn) ?? i + 1, of: ts.length,
			user: str(t?.user).trim(), plain: side(t?.plain), steered: side(t?.steered) }));
	}
	return { dose: num(json?.dose) ?? 0, turns };
}

// -- markdown -----------------------------------------------------------------------------------------------------------

/** Pi's markdown renderer in the dock's colours (`theme`: the dock's, not Pi's global one). */
export function mdTheme(theme: any): MarkdownTheme {
	const bold = (s: string) => (theme.bold ? theme.bold(s) : ansiBold(s));
	const italic = (s: string) => (theme.italic ? theme.italic(s) : `\x1b[3m${s}\x1b[23m`);
	const fg = (c: string) => (s: string) => theme.fg(c, s);
	return { heading: (s) => theme.fg("accent", bold(s)), link: fg("accent"), linkUrl: fg("dim"), code: fg("accent"), codeBlock: fg("accent"),
		codeBlockBorder: fg("dim"), quote: italic, quoteBorder: fg("dim"), hr: fg("dim"), listBullet: fg("accent"), bold, italic,
		strikethrough: (s) => (theme.strikethrough ? theme.strikethrough(s) : `\x1b[9m${s}\x1b[29m`),
		underline: (s) => (theme.underline ? theme.underline(s) : `\x1b[4m${s}\x1b[24m`) };
}

/** `text` as markdown in lines `width` columns wide (visible width; wide characters count double), in `colour` where
 * no markdown colours it: no blank lines at either end, one between blocks, the padding taken off each line and its
 * colours closed at its end. */
export function mdLines(theme: any, text: string, width: number, colour?: string): string[] {
	const style = colour ? { color: (s: string) => theme.fg(colour, s) } : undefined;
	const lines = new Markdown(text, 0, 0, mdTheme(theme), style).render(Math.max(4, width));
	const out: string[] = [];
	const blank = (l: string) => !l.replace(/\x1b\[[0-9;]*m/g, "").trim();
	for (const l of lines) {
		if (blank(l)) {
			if (out.length && out[out.length - 1] !== "") out.push("");
			continue;
		}
		out.push(truncateToWidth(l.replace(/ +$/, ""), width) + "\x1b[0m");
	}
	while (out.length && out[out.length - 1] === "") out.pop();
	return out;
}

// -- the detail ---------------------------------------------------------------------------------------------------------

/** What the last enter or x said: a steer set, what is on (a warning with it: "warning"); else /steer's own words (its
 * error in red). */
export interface Note {
	text: string;
	level: "info" | "warning" | "error" | "busy";
}

/** The note from what came back (in order): the last error (Pi's warning a refusal too), else the first line of the
 * last message; "caution" (a steer set with the server's warning, widget.ts) its warning level. */
export function noteOf(msgs: { text: string; level: string }[]): Note | undefined {
	const err = [...msgs].reverse().find((m) => m.level === "error" || m.level === "warning");
	if (err) return { text: err.text, level: "error" };
	const last = msgs.at(-1);
	return last ? { text: last.text.split("\n")[0], level: last.level === "caution" ? "warning" : "info" } : undefined;
}

const BARS = " ▁▂▃▄▅▆▇█";
/** A share 0..1 as an eighth-block (blank at 0 and for nothing measured). */
export const block = (v: number | null) => (v === null ? " " : BARS[Math.max(0, Math.min(8, Math.round(v * 8)))]);

/** The one-line reading of dose `d` of `c`; `on`: the concepts steering now (dose 0 then turns them off). */
export function reading(c: SteerConcept, d: DoseRow, on: string[] = []): [string, string] {
	if (d.dose === 0 && on.length) return ["warning", `enter turns steering off (${on.join(" + ")} ${on.length === 1 ? "is" : "are"} on)`];
	if (!c.chat) {     // with no layer, neither command takes it (the title says so)
		const what = c.stale ? "chat test out of date" : "not chat-tested";
		return ["warning", c.layer === null ? what : `${what}: /steer ${c.name} DOSE, or /concept calibrate ${c.name}`];
	}
	const away = d.runaway ? ranAway(d.runaway, d.sense !== null) : "";
	if (d.dose === c.pick) return ["success", `the dose the chat test picked (most good)${away ? `; ${away}` : ""}`];
	if (d.dose === 0) return ["muted", "what the model does on its own"];
	if (away) return ["muted", away];
	if ((d.good ?? 0) >= c.good && c.good > 0) return ["muted", "as many good as the pick"];
	if (d.sense !== null && d.sense < 1.5) return ["muted", "sense falls here"];
	if (d.style !== null && d.style < 1) return ["muted", "little style here"];
	return ["muted", "fewer good than the pick"];
}

const one = (x: number | null) => (x === null ? "–" : x.toFixed(1));

/** The dose line: "dose 0.4 · 14/18 good · style 1.8 · sense 2.9", "no steering · style 0.1 · sense 2.9". */
export function doseLine(c: SteerConcept, d: DoseRow): [string, string] {
	const said = d.dose === 0 ? "no steering" : `dose ${fmt(d.dose)}`;
	const good = !c.chat || d.dose === 0 ? "" : ` · ${d.good}/${d.turns} good`;
	return [said, `${good} · style ${one(d.style)} · sense ${one(d.sense)}`];
}

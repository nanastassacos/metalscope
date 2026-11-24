/**
 * /steer's pure logic: parsing its arguments (and /steer replay's), building a steer from a concept profile or a word,
 * the X-Scope-Steer header, and what /steer says. A steer is one or more parts, each a concept's direction at its
 * layer and dose, added to the residual at every token while on (provider.ts sends the header; replay.ts runs /steer
 * replay). No Pi or network code here, so it is testable.
 */

import { DOSE_RANGE, MIN_DOSE, NAME, offer, parseDose, parseNumber, seeHelp, sig3, unquote } from "./args.ts";
import { type ConceptInfo, defaultDose, fittingCalibration, ranAway } from "./concept.ts";

/** One vector of a steer: a concept at its layer and dose, or (the dock's LENS edit only) a lens word vector. */
export interface SteerPart {
	name: string;
	model: string; // the model its vector came from, as the server named it (informational: nothing compares it)
	layer: number;
	strength: number;
	source: "profile" | "words"; // validated by a profile, or a lens word vector built for the dock's edit
	expect?: string; // what to expect, a fact a line, its chat figure apart: the default beside it
	chat?: { good: number; turns: number; line: string }; // its good replies in the chat test at this dose, where it was tried, and their line (doseLines)
	band?: [number, number]; // a removal: the word projected out over these layers (layer is the first, strength 0)
}

/** What is being added to the residual on every request while on: one or more parts, summed, each the full dose at
 * every token, prompt and reply (thinking included). */
export interface SteerState {
	model: string; // the served model selected when it was set (Pi's model id, not the checkpoint): the header is sent only while it is selected
	parts: SteerPart[];
	note?: string; // the server's warning about the dose units when it was set (the concept list's norms_note)
}

const MAX_PARTS = 8; // steer.MAX_PARTS

/** One concept as typed after /steer: its name and a dose if given; `-word` a removal, `old>new` a swap (old removed, new
 * pushed): the dock's LENS edit's forms, which /steer itself refuses (a word changes one token, not a whole reply). */
export interface SteerWant {
	name: string;
	dose?: number;
	op?: "remove" | "swap";
}

export type SteerArgs =
	| { cmd: "off" | "status" | "help" | "again" }
	| { cmd: "replay"; parts?: SteerWant[]; temperature?: number; error?: string }
	| { cmd: "set"; parts?: SteerWant[]; error?: string };

const tooSmall = (s: string) => {
	const d = parseNumber(s);
	return d !== undefined && d > 0 && d < MIN_DOSE;
};

/** /steer's own words: each goes alone, never as a concept or after one. */
const STEER_WORDS = ["off", "status", "help", "again"];
/** Words that are not /steer forms, each with the form it means. */
const NOT_STEER: Record<string, string> = { list: "/steer list is not a form: the concepts are /concept list", none: "/steer none is not a form: /steer off" };

/** `/steer fluffy`, `/steer fluffy 1.5`, `/steer fluffy@1.5`, `/steer fluffy dose:1.5`, `/steer angry+robotic`,
 * `/steer angry@0.4 + robotic 0.8`. Concepts are joined by +; each one's dose follows its name. `/steer list` is an
 * error naming /concept list. `/steer off`, `/steer help`, `/steer again`, each alone; alone or `status`: what is on. */
export function parseSteerArgs(args: string): SteerArgs {
	const text = unquote(args || "").trim();
	const words = text.split(/\s+/).filter(Boolean);
	const err = (msg: string): SteerArgs => ({ cmd: "set", error: `${msg} · ${seeHelp("steer")}` });
	if (!words.length) return { cmd: "status" };
	if (NOT_STEER[words[0]]) return err(NOT_STEER[words[0]]);
	if (words[0] === "replay") return parseReplayWords(words.slice(1));
	if (STEER_WORDS.includes(words[0])) {
		if (words.length > 1) return err(`${words[0]} goes alone: /steer ${words[0]}`);
		return { cmd: words[0] as "off" | "status" | "help" | "again" };
	}
	const segments = text.split("+").map((s) => s.trim());
	const parts: SteerWant[] = [];
	for (const seg of segments) {
		if (!seg) return err(`an empty concept around "+": /steer angry + robotic`);
		let [head, ...rest] = seg.split(/\s+/);
		if (STEER_WORDS.includes(head) || head === "replay") return err(`${head} goes alone, not as a concept: /steer ${head}`);
		if (NOT_STEER[head]) return err(NOT_STEER[head]);
		let op: SteerWant["op"];
		if (head.includes(">")) {
			const [old, into, ...more] = head.split(">");
			if (more.length || !NAME.test(old) || !into) return err("a swap is OLD>NEW");
			if (parts.some((p) => p.name === old)) return err(`${old} appears twice`);
			parts.push({ name: old, op: "remove" });
			[head, op] = [into, "swap"];
		} else if (head.startsWith("-") && parseNumber(head) === undefined) {
			if (rest.length || head.includes("@")) return err(`${head} takes no dose`);
			[head, op] = [head.slice(1), "remove"];
		}
		const at = head.indexOf("@");
		const name = at >= 0 ? head.slice(0, at) : head;
		// A number where a concept goes (/steer 0.5, /steer fluffy +0.5): a dose with no concept before it, not a word.
		if (parseNumber(name) !== undefined) return err(`${name} is a dose, not a concept: a dose follows its concept, /steer fluffy ${name} or /steer fluffy@${name}`);
		if (!NAME.test(name)) return err(`"${name}" is not a concept name (letters, digits, _ . -, not starting with . or -)`);
		if (parts.some((p) => p.name === name)) return err(`${name} appears twice`);
		const want: SteerWant = op ? { name, op } : { name };
		const setDose = (s: string): string | undefined => {
			if (want.dose !== undefined) return `${name}'s dose is given once`;
			if (tooSmall(s)) return `dose ${s} is below ${MIN_DOSE}: too small to change anything`;
			want.dose = parseDose(s);
			return want.dose === undefined ? `dose must be ${DOSE_RANGE}` : undefined;
		};
		if (at >= 0) {
			const e = setDose(head.slice(at + 1));
			if (e) return err(e);
		}
		for (const w of rest) {
			const m = w.match(/^dose:(.*)$/);
			if (m || /^[\d.+-]/.test(w)) {
				const e = setDose(m ? m[1] : w);
				if (e) return err(e);
			} else if (STEER_WORDS.includes(w)) return err(`${w} goes alone: /steer ${w}`);
			else if (NOT_STEER[w]) return err(NOT_STEER[w]);
			else return err(`"${w}" is not a dose`);
		}
		parts.push(want);
	}
	if (parts.length > MAX_PARTS) return err(`at most ${MAX_PARTS} concepts in one steer`);
	return { cmd: "set", parts };
}

export const MAX_TEMPERATURE = 2.0;

/** /steer replay's length cap for the fresh answer, thinking included. */
export const REPLAY_MAX_TOKENS = 2048;

/** What follows `/steer replay`: a steer in any /steer form (`angry@0.6`, `angry + robotic`; none: unsteered) and
 * `temp:T` anywhere (the sampling temperature; `temp:0` is greedy). */
function parseReplayWords(rest: string[]): SteerArgs {
	const isTemp = (w: string) => w.startsWith("temp:");
	const temps = rest.filter(isTemp), others = rest.filter((w) => !isTemp(w));
	const err = (msg: string): SteerArgs => ({ cmd: "replay", error: `${msg} · ${seeHelp("steer")}` });
	if (temps.length > 1) return err("temp:T is given once");
	const t = temps.length ? parseNumber(temps[0].slice(5)) : undefined;
	if (temps.length && (t === undefined || t < 0 || t > MAX_TEMPERATURE))
		return err(`temp:T needs a temperature from 0 to ${MAX_TEMPERATURE} (0 = greedy, the same reply every time)`);
	const temp = t === undefined ? {} : { temperature: t };
	if (!others.length) return { cmd: "replay", ...temp };
	const sub = parseSteerArgs(others.join(" "));
	if (sub.cmd !== "set") return err(`/steer replay takes a steer to answer under (/steer replay angry@0.6) and temp:T, not "${others.join(" ")}"`);
	if (sub.error) return { cmd: "replay", error: sub.error };
	return { cmd: "replay", parts: sub.parts, ...temp };
}

/** A steer state from its parts, for the served model selected when it was set; `note` the server's warning, if any. */
export function steerState(parts: SteerPart[], model: string, note?: string | null): SteerState {
	return note ? { model, parts, note } : { model, parts };
}

/** The X-Scope-Steer header for a state (steer.parse_header grammar), always explicit about layers so the server
 * looks nothing up: each part as `name@dose/L`, joined by +; a removal (the dock's LENS edit's) as `remove~name/a-b`.
 * `control`: the matched random control, `random~` on every part (a random direction of the same size at the same
 * layer). */
export function steerHeader(s: SteerState, control = false): string {
	const r = control ? "random~" : "";
	return s.parts.map((p) => p.band ? `remove~${p.name}/${p.band[0]}-${p.band[1]}` : `${r}${p.name}@${+p.strength.toFixed(3)}/${p.layer}`).join("+");
}

/** "angry@0.4 L24 + fluffy@0.8 L20": each part's dose and layer. Short enough for the line above the editor, the dock
 * and /steer replay. */
export function steerLabel(s: SteerState): string {
	return s.parts.map((p) => `${p.name}@${+p.strength.toFixed(3)} L${p.layer}`).join(" + ");
}

/** What is on: "steer angry@0.4 L24 + fluffy@0.8 L20" (/scope status, /concept list). */
export function steerTag(s: SteerState): string {
	return `steer ${steerLabel(s)}`;
}

/** Each part's lines, its chat figure's first: one part's as they are; several, each under its name, indented ("" when
 * none has any). `all` false: without the chat figure's line, which the line above the editor shows. */
export function steerExpect(s: SteerState, all = true): string {
	const said = s.parts.map((p) => [p, [...(all && p.chat ? [p.chat.line] : []), ...(p.expect ? p.expect.split("\n") : [])]] as const).filter(([, ls]) => ls.length);
	if (s.parts.length === 1) return said[0]?.[1].join("\n") ?? "";
	return said.flatMap(([p, ls]) => [p.name, ...ls.map((l) => `  ${l}`)]).join("\n");
}

/** /steer's reply once set: only what the line above the editor does not say, each part's lines but its chat figure,
 * and a warning if any (the dose units' `note`); "" when nothing. */
export function steerReplyText(s: SteerState): string {
	return [steerExpect(s, false), ...(s.note ? [`warning: ${s.note}`] : [])].filter(Boolean).join("\n");
}

/** "steering angry@0.4 L24": what is on, as the line above the editor says it; `warn`: the Steer page's note, the
 * server's warning after it. */
export const steeringText = (s: SteerState, warn = false) => `steering ${steerLabel(s)}${warn && s.note ? ` · warning: ${s.note}` : ""}`;

/** What is on, for /steer status: the compact label, the header, what it does, where it acts and how to stop it, how to
 * see what it changed (replay.ts), and where a concept's details are (/concept NAME, for the parts that are concepts). */
export function steerStatusText(s: SteerState | undefined): string {
	if (!s) return "steer: off";
	const concepts = s.parts.filter((p) => p.source === "profile").map((p) => p.name);
	const details = concepts.length === 1 ? [`/concept ${concepts[0]} has the details.`] : concepts.length ? ["/concept NAME has each one's details."] : [];
	return [`steer: ${steerLabel(s)}`, `header: ${steerHeader(s)}`, steerExpect(s), "Every token, thinking included. /steer off stops it.",
		"After a reply, /steer replay shows it beside the same message answered unsteered.", ...details].filter(Boolean).join("\n");
}

/** /steer's completions for what is typed so far, from the concepts it can steer (`names`): a concept or one of
 * /steer's words; after +, a concept not typed yet; after replay, a concept or temp:0; after a concept, nothing (the
 * dose is your own; off, status and help go alone). */
export function steerCompletions(prefix: string, names: string[]): { value: string; label: string }[] {
	const segs = prefix.split("+");
	const seg = segs.at(-1)!.replace(/^\s+/, "");
	const words = seg.split(/\s+/);
	const before = prefix.slice(0, prefix.length - words.at(-1)!.length);
	const word = (xs: string[]) => offer(words.at(-1)!, xs, before);
	if (words.length === 1) return segs.length === 1 ? word([...names, "off", "status", "replay", "again", "help"]) : word(names.filter((n) => !segs.slice(0, -1).some((s) => s.trim().split(/[\s@]/)[0] === n)));
	if (segs.length === 1 && words.length === 2 && words[0] === "replay") return word([...names, "temp:0"]);
	return [];
}

/** What a dose of a concept did in chat, a fact a line: its good replies of 18 there (`chat`, with its line, where the
 * dose was tried: the chat calibration's, when it was made at the layer the concept steers at now), and the default
 * beside a dose of your own; without a calibration, how to get one and where replies broke in /concept learn's test. */
function doseLines(info: ConceptInfo, strength: number): { lines: string[]; chat?: SteerPart["chat"] } {
	const cal = fittingCalibration(info);
	const d = sig3(strength);
	if (cal) {
		const tried = cal.strengths.find((x) => x.dose === strength);
		const away = tried?.runaway ? `; ${ranAway(tried.runaway, tried.coherence !== null)}` : "";
		if (strength === cal.dose) return { lines: [], chat: { good: cal.good, turns: cal.turns, line: `In chat: ${cal.good} of ${cal.turns} replies good (the most of the doses tried)${away}` } };
		const def = `Default dose ${sig3(cal.dose)}: ${cal.good} of ${cal.turns} good (the most)`;
		return tried ? { lines: [def], chat: { good: tried.good, turns: tried.turns, line: `In chat at dose ${d}: ${tried.good} of ${tried.turns} replies good${away}` } }
			: { lines: [`Dose ${d} was not tried in chat.`, def] };
	}
	const lines = [info.calibration ? `Its chat calibration is out of date (made at L${info.calibration.layer}): /concept calibrate ${info.name} runs it again.`
		: `Not calibrated for chat yet: /concept calibrate ${info.name} finds its dose.`];
	const edge = info.generation?.edge ?? null;
	if (edge !== null) lines.push(strength >= edge ? `Replies broke at dose ${sig3(edge)} in /concept learn's test: expect damage at ${d}.`
		: `Replies broke at dose ${sig3(edge)} in /concept learn's test; ${d} is below that.`);
	return { lines };
}

/** A steer part from a learned concept, at the default dose (`defaultDose`) or at `dose` (absolute); without a dose, a
 * concept not calibrated on this model is an error (it has no default). Its lines say what the dose did in chat
 * (doseLines); what /concept learn measured is left to /concept NAME. */
export function steerFromProfile(info: ConceptInfo, dose?: number): SteerPart | { error: string } {
	if (info.error) return { error: `${info.name}: its profile could not be read (${info.error})` };
	if (!info.reader) return { error: `${info.name} can't be steered on ${info.model || "this model"}: /concept learn found no layer where it moves the model. /concept ${info.name} has what was measured; /concept lists the others.` };
	const def = defaultDose(info);
	if (dose === undefined && def === null) {
		const cal = info.calibration;
		const why = cal ? `'s calibration was made at L${cal.layer}, it steers at L${info.reader.layer} now` : ` isn't calibrated on this model`;
		return { error: `${info.name}${why}: give a dose (/steer ${info.name} 0.4) or run /concept calibrate ${info.name}.` };
	}
	const strength = (dose ?? def)!;
	const { lines, chat } = doseLines(info, strength);
	return { name: info.name, model: info.model, layer: info.reader.layer, strength, source: "profile", ...(lines.length ? { expect: lines.join("\n") } : {}), ...(chat ? { chat } : {}) };
}

/** Why /steer takes no word: the server's refusal of a whole-reply steer by one (steer.WORD_EDIT_ONLY). */
export const WORD_EDIT_ONLY = "a word changes one token (the dock's LENS edit), not a whole reply";
/** The dock's word edit, which pushes at one token only: the new word written 5 of 12 at 0.7, 2 of 12 at 0.35, every
 * reply fluent (ds4, Flash Next, 2026-11-15); held on every token, 0.7 loops. */
export const EDIT_SWAP_DOSE = 0.7;

/** The dock's words by token: a word vector's name, the token's id and text (/steer sends the text, the name goes in
 * the header). */
export type WordIds = Record<string, { id: number; text: string }>;

/** A lens token's word vector name in the dock's edit: always its id (`tok104495`), so a number, a concept's name or
 * 初步 goes as the token shown, the header and the file on disk stay ASCII, and no word vector of that name is
 * overwritten; null for whitespace. */
export function tokenName(text: string, id: number): string | null {
	return text.trim() ? `tok${id}` : null;
}

/** /api/steer/words's body for the dock's word edit: a word, by its token when `tok` (the server checks the id is that
 * text). */
export const wordBody = (model: string, name: string, tok?: { id: number; text: string }) =>
	({ model, name, words: [tok?.text ?? name], ...(tok ? { ids: [tok.id] } : {}) });

/** Where a swap pushes its new word: the band's middle (L31 of 48). */
export const pushLayer = ([a, b]: [number, number]) => Math.round((a + b) / 2);

/** A word vector (/api/steer/words: `band` the edit band) as the dock's LENS edit takes it: a removal (`-word`, a
 * swap's old word: projected out over the band) or a swap's new word (pushed at pushLayer, EDIT_SWAP_DOSE unless
 * given). */
export function steerFromWordOp(want: SteerWant, made: { name: string; band: [number, number] }, model: string): SteerPart {
	const [a, b] = made.band;
	if (want.op === "swap") return { name: made.name, model, layer: pushLayer(made.band), strength: want.dose ?? EDIT_SWAP_DOSE, source: "words" };
	return { name: made.name, model, layer: a, band: [a, b], strength: 0, source: "words" };
}

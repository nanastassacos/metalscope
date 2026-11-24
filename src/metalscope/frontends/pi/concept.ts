/**
 * /concept's pure logic, and the concept profiles the other commands read.
 *
 * A concept profile (made by /concept learn) gives a steering vector a validated layer and
 * dose. The server lists profiles at /scope/api/concepts: this module
 * parses them (`parseConcepts`, with their chat calibrations), ranks them by their good replies in graded conversations
 * (/concept list), draws one concept in the transcript (/concept NAME) and the row of doses a calibrate result
 * shares, and parses /concept's arguments. steer.ts builds a steer from a profile; jobs.ts shows the learn and
 * calibrate jobs. No Pi or network code here, so it is testable.
 */

import { NAME, NAME_MAX, addSpans, offer, parseNumber, seeHelp, signed, sig3, timeSpan, unquote } from "./args.ts";
import { type Ink, type Part, clean, commandLine, lineOf } from "./draw.ts";

/** One profiled concept as the server summarises it (concepts.summarize). */
export interface ConceptInfo {
	name: string;
	title: string;
	description: string;
	model: string;
	reader: { layer: number; strength: number; concept: number | null; control: number | null } | null; // its task log-odds at the dose; null: no reader
	generation: { edge: number | null } | null; // where /concept learn's ladder broke; null: no ladder
	steer: SteerDose | null; // the default /steer dose (concepts.steer_dose); null when not calibrated on this model
	calibration?: Calibration | null; // the chat calibration in conversation (learn/calibrate.py); null when none
	texts?: string | null; // who wrote its texts on this model, one line ("texts: written by … (date)", without the request); null: none on it
	error?: string;
}

/** One dose of a chat calibration: its good replies of `turns`, the mean coherence the grader gave (0-3; null: none
 * graded). `runaway`: its replies ran this many times the plain length (a runaway dose is graded only when needed:
 * ranAway). */
export interface ChatStrength {
	dose: number;
	good: number;
	turns: number;
	coherence: number | null;
	runaway?: number;
}

/** A concept's chat calibration on the served model Pi talks to (learn/calibrate.py, its calibration.json):
 * conversations behind a plain system prompt at each dose tried, graded blind by the served model. `dose` is the default /steer
 * uses: the one with the most good replies (`good` of `turns` there; the lower on a tie). `layer`: the steer layer it was
 * made at. */
export interface Calibration {
	layer: number;
	dose: number;
	good: number;
	turns: number;
	strengths: ChatStrength[];
}

/** How many of /concept list's best chat performers it marks. */
const TOP_IN_CHAT = 5;

/** The default /steer dose: the chat calibration's. Only a concept calibrated on this model has one. */
export interface SteerDose {
	layer: number;
	strength: number;
	basis: "chat";
}

/** The dose `/steer <concept>` uses when none is given: the server's `steer` (concepts.steer_dose); null when not
 * calibrated on this model. */
export function defaultDose(info: ConceptInfo): number | null {
	return info.steer?.strength ?? null;
}

export type LearnBackend = "omlx" | "ds4";
type Span = readonly [number, number];

/** How long /concept learn and /concept calibrate take, in minutes: the one table Pi's times come from. A row with a
 * `model` is measured (October 2026 runs) and applies to a served model whose name starts
 * with it; the backend's row without one is the range for every other model there.
 * - `learn`: learn before its calibration. On oMLX the time its own copy holds the GPU, the served model unloaded
 *   (Qwen3.6 the low end, Flash Next the high); on ds4 the corpus and the measuring.
 * - `calibrate`: the chat calibration, after learn or alone (Flash Next 5-15 min a dose, 5-7 doses and the grading).
 * - `whole`: learn's whole job, the two together (their sum when not given).
 * ds4's range is V4.1's, the only runs measured there. */
export const JOB_MINUTES: { backend: LearnBackend; model?: string; name?: string; learn: Span; calibrate: Span; whole?: Span }[] = [
	{ backend: "omlx", model: "Qwen3.6-35B-A3B", name: "Qwen3.6", learn: [5, 5], calibrate: [20, 20], whole: [26, 26] },
	{ backend: "ds4", model: "DeepSeek-V4.1-Flash", name: "DeepSeek V4.1", learn: [60, 90], calibrate: [180, 360] },
	{ backend: "omlx", learn: [5, 35], calibrate: [20, 120] },
	{ backend: "ds4", learn: [60, 90], calibrate: [180, 360] },
];

/** The times for `model` on `backend`: its measured row, else the backend's range. `measured`: the model's name when
 * measured. */
export function jobMinutes(backend: LearnBackend, model?: string): { learn: Span; calibrate: Span; whole: Span; measured?: string } {
	const m = (model ?? "").toLowerCase();
	const row = JOB_MINUTES.find((r) => r.backend === backend && (!r.model || (m && m.startsWith(r.model.toLowerCase()))))!;
	return { learn: row.learn, calibrate: row.calibrate, whole: row.whole ?? addSpans(row.learn, row.calibrate), measured: row.name };
}

/** A time as Pi says it: "26 min (measured on Qwen3.6)", or the range and the backend, "25 min to 2.5 h on oMLX". */
function timeOn(span: Span, backend: LearnBackend, measured: string | undefined, sep?: string): string {
	return `${timeSpan(span, sep)}${measured ? ` (measured on ${measured})` : ` on ${backend === "ds4" ? "ds4" : "oMLX"}`}`;
}

/** /concept learn's whole job, learn and its calibration, for `model` on `backend` (jobMinutes, timeOn). */
export function learnTime(backend: LearnBackend, model?: string, sep?: string): string {
	const t = jobMinutes(backend, model);
	return timeOn(t.whole, backend, t.measured, sep);
}

/** A chat calibration alone, for `model` on `backend`. */
export function calibrateTime(backend: LearnBackend, model?: string, sep?: string): string {
	const t = jobMinutes(backend, model);
	return timeOn(t.calibrate, backend, t.measured, sep);
}

/** A chat calibration as a tag, as calibrate.summary_text writes it: "16/18 good @0.4". */
export function chatTag(c: { good: number; turns: number; dose: number }): string {
	return `${c.good}/${c.turns} good @${sig3(c.dose)}`;
}

/** The concept's chat score (its fitting calibration: good replies at its dose), or null without one. */
export function chatScore(c: ConceptInfo): Calibration | null {
	return fittingCalibration(c);
}

/** The concepts with a chat score, best first: by good replies at their dose, then the lower dose, then name.
 * Unreadable profiles are left out. */
function rankInChat(concepts: ConceptInfo[]): ConceptInfo[] {
	return concepts.filter((c) => !c.error && chatScore(c))
		.sort((a, b) => chatScore(b)!.good - chatScore(a)!.good || chatScore(a)!.dose - chatScore(b)!.dose || a.name.localeCompare(b.name));
}

/** Where `name` ranks in chat among `concepts` (1 = best), or null when it has no chat score. */
export function chatRank(concepts: ConceptInfo[], name: string): { rank: number; of: number } | null {
	const ranked = rankInChat(concepts);
	const i = ranked.findIndex((c) => c.name === name);
	return i < 0 ? null : { rank: i + 1, of: ranked.length };
}

/** Why a concept has no chat score, in a few words (/concept list's column): "not calibrated", or "re-run calibrate"
 * (made at another steer layer); "" when it has one. */
function noScoreText(c: ConceptInfo): string {
	if (chatScore(c)) return "";
	return c.calibration ? "re-run calibrate" : "not calibrated";
}

/** The concept's calibration when it was made steering where /steer steers now (`steerLayer`, by default its reader
 * layer). null otherwise: no calibration, or one made at another layer (the profile changed since). */
export function fittingCalibration(info: ConceptInfo, steerLayer = info.reader?.layer): Calibration | null {
	const cal = info.calibration ?? null;
	if (!cal || steerLayer === undefined) return null;
	return cal.layer === steerLayer ? cal : null;
}

/** Where a concept's chat calibration stands, as a tag: " · chat: 16/18 good @0.4", " · calibrated at L15, now L18:
 * re-run …", " · not calibrated for chat"; "" for a concept with no steer layer. */
export function calibrationTag(c: ConceptInfo): string {
	if (!c.reader) return "";
	const cal = c.calibration;
	if (!cal) return " · not calibrated for chat";
	if (!fittingCalibration(c)) return ` · calibrated at L${cal.layer}, now L${c.reader.layer}: re-run /concept calibrate ${c.name}`;
	return ` · chat: ${chatTag(cal)}`;
}

const GOOD_W = 14;   // /concept list's "good in chat" column

/** One row of /concept list's table (without its rank): name, good replies in chat at the dose /steer uses (or why there
 * is no score: noScoreText) and that dose ("-": not calibrated). `w`: the name column's width (at least 12; /concept list
 * passes the longest name's). */
export function steerListLine(c: ConceptInfo, w = 12): string {
	const name = c.name.padEnd(w);
	if (!c.reader) return `${name}  (no steer layer)`;
	const cal = chatScore(c);
	const good = cal ? `${String(cal.good).padStart(2)} of ${cal.turns}` : noScoreText(c);
	const dose = defaultDose(c);
	return `${name}  ${good.padEnd(GOOD_W)}  ${dose === null ? "-" : sig3(dose)}`.trimEnd();
}

/** /concept list's table: a header, the concepts with a chat score ranked (rankInChat), the first TOP_IN_CHAT as the
 * best in chat, then those without one; unreadable profiles are left out. */
export function steerListLines(concepts: ConceptInfo[]): string[] {
	const shown = concepts.filter((c) => !c.error);
	const w = Math.max(12, ...shown.map((c) => c.name.length));
	const ranked = rankInChat(shown);
	const nw = String(Math.max(ranked.length, 1)).length;
	const pad = " ".repeat(nw + 2);
	const row = (c: ConceptInfo, i: number) => ` ${String(i + 1).padStart(nw)} ${steerListLine(c, w)}`;
	const out = [`${pad}${"concept".padEnd(w)}  ${"good in chat".padEnd(GOOD_W)}  dose`];
	if (ranked.length) out.push(`best in chat`, ...ranked.slice(0, TOP_IN_CHAT).map(row));
	if (ranked.length > TOP_IN_CHAT) out.push("the rest", ...ranked.slice(TOP_IN_CHAT).map((c, i) => row(c, i + TOP_IN_CHAT)));
	const rest = shown.filter((c) => !ranked.includes(c));
	if (rest.length) out.push("no chat score", ...rest.map((c) => `${pad}${steerListLine(c, w)}`));
	return out;
}

/** The notes under /concept list's table: what its columns mean. */
export function steerListNotes(concepts: ConceptInfo[]): string[] {
	const shown = concepts.filter((c) => !c.error);
	const turns = shown.map(chatScore).find((c) => c)?.turns ?? 18;
	return [`good in chat: of ${turns} replies in 3 graded conversations, at its best dose.`,
		...(shown.some((c) => c.reader && defaultDose(c) === null) ? [`dose "-": not calibrated; /steer NAME DOSE, or /concept calibrate NAME.`] : []),
		"/concept NAME for details · /steer NAME to use one."];
}

/** A dose whose replies ran `times` the plain length: "ran 12x the plain length", ", not graded" without grades. */
export function ranAway(times: number, graded: boolean): string {
	return `ran ${Math.round(times)}x the plain length${graded ? "" : ", not graded"}`;
}

/** The first sentence of a text (up to ". "), with its full stop. */
function firstSentence(text: string): string {
	const t = text.trim();
	const i = t.indexOf(". ");
	return i < 0 ? t : t.slice(0, i + 1);
}

/** A chat calibration's doses as the parts of a line, ascending: "0.1 0/18 · 0.3 7/18, ran 7x the plain length · 0.4
 * ran 13x the plain length, not graded" (good replies of how many, then whether its replies ran away; a runaway dose
 * not graded: ranAway alone); the default dose in the normal colour, the others dim. A learn or calibrate result's
 * calibrate step (jobs.ts) and /concept NAME's "in chat" row. */
export function doseRow(cal: Calibration): Part[] {
	const said = (s: ChatStrength) => (s.runaway && s.coherence === null ? ranAway(s.runaway, false)
		: `${s.good}/${s.turns}${s.runaway ? `, ${ranAway(s.runaway, true)}` : ""}`);
	return [...cal.strengths].sort((a, b) => a.dose - b.dose).flatMap((s, i) => [...(i ? [{ t: " · ", c: "dim" }] : []),
		{ t: `${sig3(s.dose)} ${said(s)}`, c: s.dose === cal.dose ? undefined : "dim" }]);
}

/** /concept NAME in the transcript (the "scope-concept" entry, entries.ts), in `width` columns, the job widget's look:
 * a head line (its name bold, then its chat score and its `rank` among the concepts, or why it has none, dim), what it
 * is, whose texts, its doses in chat (doseRow; made at another layer, calibrationTag), what /concept learn found, then
 * the commands that go further (`backend`: the selected model's, whose calibration time they give: jobMinutes). */
export function overviewLines(c: ConceptInfo, rank: { rank: number; of: number } | null, backend: LearnBackend, width: number, th: Ink): string[] {
	const head = (said: string) => lineOf([{ t: c.name, c: "bold" }, { t: ` · ${said}`, c: "dim" }], width, th);
	if (c.error) return [head(`its profile could not be read (${c.error})`)];
	const cal = fittingCalibration(c), r = c.reader, dose = defaultDose(c), time = `about ${timeSpan(jobMinutes(backend, c.model).calibrate)}`;
	const row = (label: string, value: Part[]) => lineOf([{ t: `  ${label.padEnd(11)} ` }, ...value], width, th);
	const dim = (t: string): Part[] => [{ t, c: "dim" }];
	const chat = cal ? doseRow(cal) : r && c.calibration ? dim(calibrationTag(c).replace(/^ · /, "")) : null;
	const next = !r ? [`metalscope concept show ${c.name}${c.model ? ` ${c.model}` : ""}`, " has what was measured"]
		: !c.calibration ? [`/concept calibrate ${c.name}`, ` (${time}) tests it in chat and sets the dose `, `/steer ${c.name}`, " uses"]
		: [`/steer ${c.name}${dose === null ? " DOSE" : ""}`, `${dose === null ? "" : ` uses dose ${sig3(dose)}`} · `, `/concept calibrate ${c.name}`, ` (${time})`];
	return [
		head(cal ? `${chatTag(cal)}${rank ? ` · ranked ${rank.rank} of ${rank.of}` : ""}` : !r ? "no steer layer" : !c.calibration ? "not calibrated" : "out of date"),
		row("what", dim(firstSentence(c.description) || "no description")),
		row("texts", dim(c.texts?.replace(/^texts: /, "") ?? "none on this model")),
		...(chat ? [row("in chat", chat)] : []),
		...(r ? [row("learn", dim(`L${r.layer}` + (r.concept !== null && r.control !== null
			? ` · ${signed(r.concept - r.control)} log-odds over control at dose ${sig3(r.strength)}` : "")))] : []),
		commandLine(next, width, th),
	];
}

/** GET /scope/api/concepts for one model, its profiles parsed (parseConcepts): whether the server allows steering,
 * the profiles. */
export interface ConceptList {
	model: string;
	steer_enabled: boolean;
	concepts: ConceptInfo[];
	norms_note?: string | null; // a warning when the dose units came from few or short traces (steer.norms_note)
	library?: LibraryConcept[]; // the library's concepts not learned on this model yet (web.py _library)
}

/** A concept in the library not learned on this model yet: shipped texts, this model's own texts, or a description
 * alone (each model writes its own texts from it): /concept learn NAME needs no description. `description`: its first
 * sentence; `writer`: who wrote the texts ("Claude Fable 5.1 (by hand)"), null without texts or a record; `texts`: the
 * texts' line ("texts: each model writes its own" for a description alone). */
export interface LibraryConcept {
	name: string;
	title: string;
	description: string;
	writer: string | null;
	texts: string | null;
}

/** The server's library list (web.py _library), a name each at least. */
export function parseLibrary(json: any): LibraryConcept[] {
	const list = Array.isArray(json?.library) ? json.library : [];
	const str = (x: any) => (typeof x === "string" && x ? clean(x) : null);
	return list.filter((c: any) => typeof c?.name === "string" && c.name).map((c: any): LibraryConcept => ({
		name: c.name, title: str(c.title) ?? c.name, description: str(c.description) ?? "", writer: str(c.writer), texts: str(c.texts) }));
}

/** The library's concepts as rows under a line per writer ("texts by Claude Fable 5.1 (by hand)"; no writer: their
 * texts line, "texts: each model writes its own"), writers in the order first met, each row the name and its first
 * sentence cut to fit `width` columns. */
export function libraryLines(lib: LibraryConcept[], width = 80): string[] {
	const w = Math.max(12, ...lib.map((c) => c.name.length));
	const room = Math.max(20, width - w - 4);
	const cut = (s: string) => (s.length <= room ? s : s.slice(0, room - 1).replace(/\s+\S*$/, "") + " …");
	const by = new Map<string, LibraryConcept[]>();
	for (const c of lib) {
		const k = c.writer ? `texts by ${c.writer}` : c.texts ?? "texts: no record of who wrote them";
		by.set(k, [...(by.get(k) ?? []), c]);
	}
	return [...by].flatMap(([head, cs]) => [head, ...cs.map((c) => `  ${c.name.padEnd(w)}  ${cut(c.description)}`.trimEnd())]);
}

/** What /concept says of a concept that is not learned on this model. One in the library: how to learn it and how
 * long that takes for the model on `backend` ("joyful is not learned on M yet: /concept learn joyful, about 25 min to
 * 2.5 h on oMLX", learnTime);
 * else that it is neither, with the concepts learned here. */
export function notLearnedText(name: string, list: Pick<ConceptList, "model" | "concepts" | "library">, backend: LearnBackend = "omlx"): string {
	if (list.library?.some((c) => c.name === name)) return `${name} is not learned on ${list.model} yet: /concept learn ${name}, about ${learnTime(backend, list.model)}`;
	const learned = list.concepts.map((c) => c.name);
	return `${name} is not learned on ${list.model} and is not in the library (learned here: ${learned.join(", ") || "none"}) · ${seeHelp("concept")}`;
}

/** /concept's "Ready to learn" section: the library's concepts not learned on this model (libraryLines), how to learn
 * one (no description needed: `/concept learn joyful`, the README's, when it is there) and how long it takes for the
 * model on `backend`. None when the library has nothing left to learn. */
export function readyToLearnLines(list: Pick<ConceptList, "library"> & { model?: string }, backend: LearnBackend): string[] {
	const lib = list.library ?? [];
	if (!lib.length) return [];
	const eg = lib.some((c) => c.name === "joyful") ? "joyful" : lib[0].name;
	return [`Ready to learn: ${lib.length} in the library`, ...libraryLines(lib),
		`Learn one, no description needed: /concept learn ${eg}`,
		`Each is learned then calibrated in chat: about ${learnTime(backend, list.model)}.`];
}

/** What /concept list says: the concepts learned on this model, their table and its notes under a head line
 * (steerListLines, steerListNotes); then the library's not learned yet (readyToLearnLines, for `backend`); then what is
 * on (`on`: the steer's tag, steer.ts steerTag; undefined: nothing). With nothing learned, the library leads. */
export function conceptListText(list: ConceptList, on: string | undefined, backend: LearnBackend = "omlx"): string {
	const now = `On now: ${on ?? "nothing"}.`;
	const ready = readyToLearnLines(list, backend);
	if (!list.concepts.length) {
		const none = ready.length ? `Nothing learned on ${list.model} yet.` : `Nothing learned on ${list.model} yet: /concept learn NAME <what it means> makes a concept.`;
		return [none, ...ready, now].join("\n");
	}
	const head = `Learned on ${list.model},\nranked by good replies in conversation at the dose /steer uses:`;
	return [head, ...steerListLines(list.concepts), ...steerListNotes(list.concepts), ...ready, now].join("\n");
}

/** --corpus DIR, as /concept learn takes it after the rest (`learn NAME [description] [--corpus DIR]`). */
export interface ConceptFlags {
	corpus?: string; // a directory of texts written elsewhere, as typed (job-runner makes it absolute)
}

/** `words` split at the first flag: what comes before, and the flags (a flag's value runs to the next flag, so a
 * path may have spaces); an error for an unknown flag or a missing value. `allowed`: the flags this subcommand takes. */
export function takeFlags(words: string[], allowed: readonly "corpus"[]): { rest: string[]; flags: ConceptFlags } | { error: string } {
	const at = words.findIndex((w) => w.startsWith("--"));
	if (at < 0) return { rest: words, flags: {} };
	const flags: ConceptFlags = {};
	for (let i = at; i < words.length;) {
		const flag = words[i].slice(2);
		let j = i + 1;
		while (j < words.length && !words[j].startsWith("--")) j++;
		const value = words.slice(i + 1, j).join(" ");
		if (flag !== "corpus" || !allowed.includes(flag)) return { error: `unknown option "${words[i]}" (this takes ${allowed.length ? allowed.map((f) => `--${f}`).join(", ") : "none"})` };
		if (!value) return { error: `--${flag} needs a value: --${flag} DIR` };
		flags.corpus = unquote(value);
		i = j;
	}
	return { rest: words.slice(0, at), flags };
}

export type LearnArgs =
	| { kind: "start"; name: string; description: string; corpus?: string }
	| { kind: "help" }
	| { kind: "error"; message: string };

/** /concept's subcommands, in the order completions offer them. */
export const CONCEPT_SUBCOMMANDS = ["list", "learn", "calibrate", "help"];

/** /concept's completions from the last concept list (`list`; none yet: the subcommands alone): a subcommand or a
 * concept learned here; after calibrate one it can steer (a steer layer), after learn one from the library. */
export function conceptCompletions(prefix: string, list?: Pick<ConceptList, "concepts" | "library">) {
	const words = prefix.trimStart().split(/\s+/);
	const concepts = list?.concepts ?? [];
	if (words.length === 1) return offer(words[0], [...CONCEPT_SUBCOMMANDS, ...concepts.map((c) => c.name)]);
	if (words.length !== 2) return [];
	const [sub, word] = words;
	const names = sub === "learn" ? (list?.library ?? []).map((c) => c.name)
		: sub === "calibrate" ? concepts.filter((c) => !!c.reader).map((c) => c.name) : [];
	return offer(word, names, `${sub} `);
}

/** Words a new concept may not be named: a concept called "list" or "off" could not be named to /concept or /steer
 * (each would read it as its own word). The server refuses the same (learn/plan.py RESERVED_NAMES). */
const RESERVED_NAMES = [...new Set([...CONCEPT_SUBCOMMANDS, "show", "status", "jobs", "cancel", "replay", "off", "none"])];

/** What /concept learn NAME says when NAME has no texts and no saved description, and none was given. */
export function sayWhatText(name: string, form = "/concept learn"): string {
	return `say what ${name} means: ${form} ${name} <one line, e.g. "crisp, brittle textures that crackle"> (no description is saved for it)`;
}

/** What follows `/concept learn`: `NAME description… [--corpus DIR]` (quotes around the description
 * optional), or help. The description may be left out: a library concept is learned from its texts or its saved
 * description (the server says which, `LearnPlan`), supplied ones (--corpus) have their own. */
export function parseLearnArgs(args: string): LearnArgs {
	const s = args.trim();
	if (!s || s === "help") return { kind: "help" };
	const [quoted, ...words] = s.split(/\s+/);
	const name = unquote(quoted);
	if (!NAME.test(name) || name.length > NAME_MAX)
		return { kind: "error", message: `"${name}" is not a usable name: letters, digits, _ . - only, at most ${NAME_MAX} characters` };
	// /steer reads a number as a dose (steer.ts), so a concept named 1984 could be learned and never steered.
	if (parseNumber(name) !== undefined) return { kind: "error", message: `"${name}" reads as a number, and /steer takes a number as a dose, so it cannot name a concept: pick a name with a letter in it` };
	if (RESERVED_NAMES.includes(name.toLowerCase())) return { kind: "error", message: `"${name}" is a command word, so it cannot name a concept: pick another name` };
	const f = takeFlags(words, ["corpus"]);
	if ("error" in f) return { kind: "error", message: f.error };
	let description = f.rest.join(" ").trim();
	const wrapped = description.match(/^["“'](.*)["”']$/);
	if (wrapped && !/["“”]/.test(wrapped[1])) description = wrapped[1].trim();     // one pair of quotes around the whole line
	return { kind: "start", name, description, ...f.flags };
}

/** What /concept learn NAME would do, as GET /scope/api/concept/learn answers (learn/plan.py learn_plan): whose
 * texts it uses (`source`): the model's own, written before ("own"), the ones in --corpus ("corpus"), metalscope's
 * library texts copied for the model ("shipped"), else the served model writes its own ("write") from `description`
 * (the one given, else the saved one; "" none). `texts`: who wrote existing ones, one line. `taken`: why it cannot be
 * learned on this model (read first: a plan refused early says "write"). `notes`: warnings that do not stop it (the
 * dose units', say). */
export interface LearnPlan {
	name: string;
	source: "own" | "corpus" | "shipped" | "write";
	texts: string | null;
	description: string;
	taken: string | null;
	notes?: string[];
}

export type ConceptArgs =
	| { cmd: "list" }
	| { cmd: "help"; topic?: string }               // topic: one subcommand's help (/concept calibrate help)
	| { cmd: "overview"; name: string }       // /concept NAME: the concept in the transcript
	| { cmd: "learn"; name: string; description: string; corpus?: string }
	| { cmd: "calibrate"; name: string }
	| { cmd: "error"; error: string };

/** `/concept` (alone: list) · NAME (its overview) · list · learn NAME [<what it
 * means>] [--corpus DIR] · calibrate NAME · help [SUB]. Every subcommand answers `help` as its next word (`/concept calibrate help`), as
 * does `/concept help calibrate`; one that needs a name and has none shows its help too. Its errors point at that
 * help. */
export function parseConceptArgs(args: string): ConceptArgs {
	const [raw, ...words] = (args || "").trim().split(/\s+/).filter(Boolean);
	const cmd = raw?.toLowerCase();
	const rest = cmd === "learn" ? words : words.map(unquote);
	const sub = cmd && CONCEPT_SUBCOMMANDS.includes(cmd) && cmd !== "help" ? cmd : undefined;
	const err = (msg: string): ConceptArgs => ({ cmd: "error", error: `${msg} · ${seeHelp(sub ? `concept ${sub}` : "concept")}` });
	if (sub && rest[0]?.toLowerCase() === "help") return rest.length === 1 ? { cmd: "help", topic: sub } : err(`help goes alone: /concept ${sub} help`);
	const name = (n: string | undefined, form: string): string | ConceptArgs =>
		!n ? { cmd: "help", topic: form.split(" ")[0] }
			: CONCEPT_SUBCOMMANDS.includes(n) ? err(`"${n}" is a subcommand, not a concept: /concept ${n}`)
			: RESERVED_NAMES.includes(n.toLowerCase()) ? err(`"${n}" is a command word, not a concept`)
			: NAME.test(n) ? n : err(`"${n}" is not a concept name (letters, digits, _ . -)`);
	switch (cmd) {
		case undefined:
		case "list":
			if (rest.length && cmd !== undefined) return err(`/concept ${cmd} takes no arguments`);
			return { cmd: cmd ?? "list" };
		case "help": {
			const topic = rest[0]?.toLowerCase();
			if (!topic) return { cmd: "help" };
			if (rest.length > 1 || !CONCEPT_SUBCOMMANDS.includes(topic) || topic === "help")
				return { cmd: "error", error: `/concept help takes one subcommand (${CONCEPT_SUBCOMMANDS.filter((c) => c !== "help").join(", ")}) · ${seeHelp("concept")}` };
			return { cmd: "help", topic };
		}
		case "learn": {
			const a = parseLearnArgs(rest.join(" "));
			if (a.kind === "help") return { cmd: "help", topic: "learn" };
			if (a.kind === "error") return err(a.message);
			return { cmd: "learn", name: a.name, description: a.description, ...(a.corpus !== undefined ? { corpus: a.corpus } : {}) };
		}
		case "calibrate": {
			const f = takeFlags(rest, []);
			if ("error" in f) return err(f.error);
			if (f.rest.length > 1) return err("/concept calibrate takes one concept (it searches the doses itself)");
			const n = name(f.rest[0], "calibrate NAME");
			return typeof n === "string" ? { cmd: "calibrate", name: n } : n;
		}
		default: {
			const bare = unquote(raw);                      // a lone concept name: its overview
			if (!words.length && NAME.test(bare)) return { cmd: "overview", name: bare };
			return err(`unknown subcommand "${raw}"`);
		}
	}
}

/** A number from the server, else null. */
const num = (x: any): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);
/** Whether each of `keys` in `o` is a number. */
const nums = (o: any, keys: readonly string[]) => !!o && keys.every((k) => num(o[k]) !== null);
/** One dose of a calibration from the server, or null when not well formed (jobs.ts trimJob keeps these alone). */
export function parseStrength(s: any): ChatStrength | null {
	if (!nums(s, ["dose", "good", "turns"])) return null;
	return { dose: s.dose, good: s.good, turns: s.turns, coherence: num(s.coherence), ...(num(s.runaway) !== null ? { runaway: s.runaway } : {}) };
}

/** The calibration.json version this dock reads (models.CALIBRATION_VERSION). */
const CALIBRATION_VERSION = 4;

/** A concept's calibration from the server (calibration.json), or null when absent, of another version (ignored) or
 * not well formed. */
export function parseCalibration(c: any): Calibration | null {
	if (!c || c.version !== CALIBRATION_VERSION || !nums(c, ["layer", "default_dose", "good", "turns"]) || !c.turns) return null;
	return { layer: c.layer, dose: c.default_dose, good: c.good, turns: c.turns,
		strengths: Array.isArray(c.strengths) ? c.strengths.map(parseStrength).filter((s: ChatStrength | null): s is ChatStrength => s !== null) : [] };
}

/** The server's concept list, with missing numbers made explicit so the UI never does arithmetic on null. */
export function parseConcepts(json: any): ConceptInfo[] {
	const list = Array.isArray(json?.concepts) ? json.concepts : [];
	return list.map((c: any): ConceptInfo => {
		const r = c.reader && num(c.reader.layer) !== null && num(c.reader.strength) !== null ? {
			layer: c.reader.layer, strength: c.reader.strength, concept: num(c.reader.concept), control: num(c.reader.control),
		} : null;
		return { name: String(c.name ?? "?"), title: clean(String(c.title ?? c.name ?? "?")), description: clean(String(c.description ?? "")), model: String(c.model ?? json?.model ?? ""),
			reader: r,
			generation: c.generation ? { edge: num(c.generation.edge) } : null,
			steer: c.steer && num(c.steer.layer) !== null && num(c.steer.strength) !== null
				? { layer: c.steer.layer, strength: c.steer.strength, basis: "chat" } : null,
			calibration: parseCalibration(c.calibration),
			texts: typeof c.texts === "string" ? clean(c.texts) : null,
			error: c.error ? clean(String(c.error)) : undefined };
	});
}

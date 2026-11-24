/**
 * /jlens's pure logic (commands.ts registers /jlens; job-runner.ts starts /jlens learn as a job): parsing its
 * arguments, the status it shows for the served model's Jacobian lens (GET /scope/api/lens, jlens.status on the server)
 * with every lens that could be its, ranked, the confirm text before a job unloads the model, and what /jlens use
 * answers.
 */

import { homedir } from "node:os";
import { isAbsolute, join } from "node:path";
import { NAME, f3, offer, seeHelp } from "./args.ts";

/** A lens that is or could be the served model's (jlens.candidates): the one in use and its checkpoint's other lenses
 * (the ones in its folder: fitted by /jlens learn, installed for it); and, when there is any, the
 * logit lens (name "logit"). */
export interface LensCandidate {
	name: string;
	in_use: boolean;
	why: string;                       // in use · learned for it · its lens
	n_prompts?: number | null;
	learned_for?: string | null;
	fitted?: string | null;
	hit10?: number | null;             // held-out hit@10 in the workspace band (a learned lens: on its corpus)
	logit_hit10?: number | null;       // the logit lens on the same windows
	compared?: { lens: string; hit10: number } | null;   // the lens in use when it was fitted, on the same windows
	logit?: boolean;                   // the logit lens (J = I): in use when the model reads no J-lens
	scored_on?: string | null;         // the logit lens: whose held-out windows its hit@10 is on
}

/** What /jlens off and /jlens on ask for: the server's names for the plain logit lens and for the model's J-lens
 * (jlens.LOGIT, jlens.JLENS: the one in use, else the one it used last, else the best scored). */
const LOGIT = "logit";
const JLENS = "jlens";

/** The served model's lens as GET /scope/api/lens returns it (src/metalscope/jlens.py status). */
export interface LensStatus {
	model: string;
	installed: boolean;
	path: string | null;
	lenses_dir: string;
	source?: string | null;
	hf_model?: string | null;
	n_prompts?: number | null;
	layers?: number | null;            // how many layers the lens has a Jacobian for
	candidates?: LensCandidate[];      // ranked by held-out hit@10, unscored ones last
}

/** What /jlens learn would do, as GET /scope/api/lens/learn returns it (learn/spawn.py lens_learn_plan). */
export interface LensLearnPlan {
	model: string;
	until?: string | null;
	windows?: number | null;
	estimate?: { d_model: number; layers: number; min_per_window: number; overhead_min: number } | null;
	windows_estimate?: number | null;
	source?: string | null;            // a directory or a .txt; none: the default corpus
	kind: "default" | "dir" | "txt";
	resume?: number;                   // windows an earlier fit of this model on this corpus got through
	lens: string;                      // the directory it writes (lenses/<model>-learned-<date>)
}

/** /jlens's completions: its subcommands; after `use `, the lenses (`names`: the served model's, from /jlens). */
export function lensCompletions(prefix: string, names: string[]) {
	const m = prefix.trimStart().match(/^use\s+(\S*)$/i);
	return m ? offer(m[1], names, "use ") : offer(prefix.trimStart(), LENS_SUBCOMMANDS);
}

/** The lens names /jlens use takes, from /jlens's status: the candidates, the logit lens left out (/jlens off). */
export const lensNames = (st: Pick<LensStatus, "candidates">): string[] => (st.candidates ?? []).filter((c) => !c.logit).map((c) => c.name);

/** /jlens's subcommands, in the order completions offer them. */
export const LENS_SUBCOMMANDS = ["status", "on", "off", "use", "learn", "help"];

export type LensArgs =
	| { cmd: "status" | "help" }
	| { cmd: "learn"; path?: string; until?: string; windows?: number; fresh?: boolean }
	| { cmd: "use"; name: string }
	| { cmd: "error"; error: string };

const HHMM = /^([01]?\d|2[0-3]):[0-5]\d$/;
const LEARN_FORM = "/jlens learn [PATH] until HH:MM (or windows N)";

/** `/jlens` (alone: status) · status · on · off · use NAME · learn [PATH] until HH:MM | windows N [fresh] · help. off turns the served model's J-lens off (the logit lens alone), on back on (the server's "logit" and
 * "jlens"); use NAME picks a J-lens.
 * learn's PATH is everything before its first keyword (until, windows, fresh), so it may hold spaces; it needs a
 * deadline or a count, since the model stays unloaded until then. */
export function parseLensArgs(args: string): LensArgs {
	const [raw, ...rest] = (args || "").trim().split(/\s+/).filter(Boolean);
	const cmd = raw?.toLowerCase();
	const err = (msg: string): LensArgs => ({ cmd: "error", error: `${msg} · ${seeHelp("jlens")}` });
	if (cmd === undefined) return { cmd: "status" };
	if (!LENS_SUBCOMMANDS.includes(cmd)) return err(`unknown subcommand "${raw}"`);
	if (cmd === "learn") return parseLearn(rest, err);
	if (cmd === "on" || cmd === "off") {
		if (rest.length) return err(`/jlens ${cmd} takes no arguments`);
		return { cmd: "use", name: cmd === "on" ? JLENS : LOGIT };
	}
	if (cmd === "use") {
		if (rest.length === 1 && rest[0].toLowerCase() === "help") return { cmd: "help" };   // not a lens named help
		if (rest.length !== 1) return err("/jlens use takes one lens name (/jlens lists them)");
		if (!NAME.test(rest[0])) return err(`"${rest[0]}" is not a lens name`);
		return { cmd: "use", name: rest[0] };
	}
	if (rest.length) return err(`/jlens ${cmd} takes no arguments`);
	return { cmd: cmd as "status" | "help" };
}

function parseLearn(words: string[], err: (msg: string) => LensArgs): LensArgs {
	const keys = new Set(["until", "windows", "fresh"]);
	const first = words.findIndex((w) => keys.has(w.toLowerCase()));
	const pathWords = first < 0 ? words : words.slice(0, first);
	const out: { cmd: "learn"; path?: string; until?: string; windows?: number; fresh?: boolean } = { cmd: "learn" };
	if (pathWords.length) out.path = pathWords.join(" ");
	for (let i = first < 0 ? words.length : first; i < words.length; i++) {
		const k = words[i].toLowerCase();
		if (k === "fresh") out.fresh = true;
		else if (k === "until") {
			const t = words[++i];
			if (!t || !HHMM.test(t)) return err(`until takes a time, HH:MM (24-hour), not "${t ?? ""}"`);
			out.until = t.padStart(5, "0");
		} else if (k === "windows") {
			const n = Number(words[++i]);
			if (!Number.isInteger(n) || n < 1 || n > 10000) return err(`windows takes a count from 1 to 10000, not "${words[i] ?? ""}"`);
			out.windows = n;
		} else return err(`"${words[i]}" after the path: ${LEARN_FORM}`);
	}
	if (!out.until && !out.windows) return err(`say until when: ${LEARN_FORM}; the model is unloaded until then`);
	return out;
}

/** A path as typed in Pi, made absolute for the server (another process, another working directory). */
export function resolveLensPath(path: string, cwd: string, home = homedir()): string {
	if (path === "~" || path.startsWith("~/")) return join(home, path.slice(1));
	return isAbsolute(path) ? path : join(cwd, path);
}


/** One candidate in a line: its rank, name, why it is listed, its held-out hit@10 against the logit lens (and the
 * lens in use when it was fitted, on the same windows), and for a learned one its windows and date. */
function candidateLine(c: LensCandidate, rank: number): string {
	if (c.logit) {
		const on = c.hit10 != null ? `held-out hit@10 ${f3(c.hit10)}${c.scored_on ? ` on ${c.scored_on}'s windows` : ""}` : "not scored";
		return `  ${rank}. ${LOGIT}: ${c.in_use ? "in use" : "the logit lens"} · ${on}`;
	}
	const score = c.hit10 != null
		? `held-out hit@10 ${f3(c.hit10)}${c.logit_hit10 != null ? ` vs logit lens ${f3(c.logit_hit10)}` : ""}` +
			(c.compared && c.compared.lens !== c.name ? `, ${c.compared.lens} ${f3(c.compared.hit10)} on the same windows` : "")
		: "not scored";
	const learned = c.learned_for ? ` · learned${c.fitted ? ` ${c.fitted}` : ""}${c.n_prompts ? `, ${c.n_prompts} windows` : ""}` : "";
	return `  ${rank}. ${c.name}: ${c.why} · ${score}${learned}`;
}

/** The ranked list under /jlens, when there is more than the lens in use: best held-out hit@10 first. */
export function candidateLines(st: LensStatus): string[] {
	const cs = st.candidates ?? [];
	if (!cs.length || (cs.length === 1 && cs[0].in_use)) return [];
	const scored = cs.filter((c) => c.hit10 != null).length;
	return [`lenses for ${st.model}, best held-out hit@10 first${scored > 1 ? " (each on its own held-out windows)" : ""}:`,
		...cs.map((c, i) => candidateLine(c, i + 1)),
		"/jlens off turns the J-lens off (traces then read the logit lens only), /jlens on turns it back on, from the next token, no reload; /jlens use NAME picks one of these."];
}

/** What /jlens shows: whether the served model has a lens and where; where it came from; what the lens powers; every
 * other lens that could be its, ranked. With none: where lenses go,
 * where published ones are, and /jlens learn. */
export function lensStatusLines(st: LensStatus): string[] {
	if (!st.installed && st.candidates?.some((c) => c.logit && c.in_use)) return [
		`${st.model} reads through the logit lens: no J-lens is in use`,
		"LENS reads the logit lens; the LENS edit has no word vectors.",
		...candidateLines(st),
	];
	if (!st.installed) return [
		`no lens for ${st.model} (a lens goes in ${st.lenses_dir}/<name>/)`,
		"Without one, LENS reads the logit lens; the LENS edit has no word vectors.",
		"Published lenses exist for about 40 open models: neuronpedia/jacobian-lens on Hugging Face; scripts/convert_lens.py --checkpoint converts one into that folder.",
		"/jlens learn until 07:30 fits one overnight (the default corpus, or /jlens learn PATH until 07:30 for your own text); it asks first.",
		...candidateLines(st),
	];
	const from = [st.source, st.hf_model, st.n_prompts ? `fitted on ${st.n_prompts} prompts` : "", st.layers ? `${st.layers} layers` : ""].filter(Boolean);
	return [
		`lens for ${st.model}: ${st.path}`,
		...(from.length ? [`  ${from.join(" · ")}`] : []),
		"what it powers",
		"  its read-out: the dock's LENS and SETTLING, the settle layer; the LENS edit",
		...candidateLines(st),
		"/jlens learn fits a new lens overnight.",
	];
}

/** The corpus of a plan in a few words. */
function corpusText(p: LensLearnPlan): string {
	if (p.kind === "txt") return `the paragraphs of ${p.source} (shallow windows only, the first tokens of each: the lens learns nothing about positions deep in a chat)`;
	if (p.kind === "dir") return `the text files in ${p.source} (long files give windows up to 8k tokens deep)`;
	return "the default corpus: wikitext paragraphs, as plain text and as chat turns a few thousand tokens deep, and metalscope's own source as code";
}

/** Minutes as the job says them (learn/state.py span): "40 min", "3.8 h". */
const span = (m: number) => (m >= 90 ? `${(m / 60).toFixed(1)} h` : `${Math.max(1, Math.round(m))} min`);

/** What the confirm dialog says before /jlens learn unloads the model: what is fitted and until when, that chat pauses
 * all that time, roughly how many windows fit (a rough guess from the model's d_model and layers, until the job times a
 * pass), where the lens goes and that nothing uses it until /jlens use. */
export function lensLearnConfirmText(p: LensLearnPlan): string {
	const stop = [p.until ? `until ${p.until}` : "", p.windows ? `${p.until ? "or " : ""}for ${p.windows} window${p.windows === 1 ? "" : "s"}` : ""].filter(Boolean).join(" ");
	const est = p.estimate;
	const howMany = est
		? `Rough guess until the first pass is timed: about ${span(est.min_per_window)} a window (${est.d_model} dims, ${est.layers} layers)` +
			(p.windows_estimate != null ? `, so about ${p.windows_estimate} window${p.windows_estimate === 1 ? "" : "s"}${p.until ? ` by ${p.until}` : ""}` : "") +
			`.${p.until ? " The job stops if none fits." : ""} /scope jobs shows its pace.`
		: "No estimate of how many windows fit: the served checkpoint's config is unreadable.";
	return [
		`A GPU job loads its own copy of ${p.model} and fits a Jacobian lens on ${corpusText(p)}, one window at a time, ${stop}.`,
		`${p.model} is unloaded the whole time: chat pauses ${p.until ? `until ${p.until} (the reload included)` : "until the windows are done"}, replies in progress are aborted, and model loads are refused until it is back.`,
		howMany,
		...(p.resume ? [`It resumes the earlier fit of ${p.model} on this corpus (${p.resume} windows done).`] : []),
		`After every window the lens is written to lenses/${p.lens}/ with its held-out hit@10, against the logit lens and the lens in use. The model reads it when it is back if it is its only lens; else /jlens use ${p.lens}.`,
		"/scope jobs cancel stops it and keeps the windows done.",
	].join("\n");
}

/** What POST /scope/api/lens/use answers (jlens.use, and the loaded model's switch). */
export interface LensUseResult {
	model: string;
	lens: string;                      // "logit": the logit lens
	was: string | null;
	live?: string | null;              // the lens the loaded model now reads; null: its hooks are not installed yet
}

/** When the switch shows: at once (the loaded model reads it from its next forward), when tracing starts, or not
 * (the lens failed to load on the server, which reads another). */
function whenText(r: LensUseResult): string {
	if (r.live == null) return `It is read when tracing next starts on ${r.model} (or it next loads).`;
	if (r.live !== r.lens) return `${r.lens} did not load on the server, so ${r.model} reads through ${r.live}: the server log says why.`;
	return `${r.model} reads it from its next token, no reload.`;
}

/** What /jlens use says it did, and what is left to do. */
export function lensUseText(r: LensUseResult): string {
	const was = r.was && r.was !== r.lens ? ` (was ${r.was})` : "";
	if (r.lens === LOGIT) return [
		`${r.model} now reads through the logit lens only${was}: the J-lens is off.`,
		whenText(r),
		"No J-lens, so no J-lens read-outs and no /steer word vectors; /jlens on turns it back on.",
	].join("\n");
	return [
		`${r.lens} is now the J-lens for ${r.model}${was}, read beside the logit lens in every trace.`,
		whenText(r),
	].join("\n");
}

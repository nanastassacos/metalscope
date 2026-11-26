/**
 * Help, in three levels:
 * - `/scope help`: one screen: how to start, the dock's keys, steering, learning a concept, the commands (`overviewHelp`).
 * - `/<command> help` (/scope's: `/scope help scope`): that command's forms, one short line each, and a few notes
 *   (`COMMANDS`, `commandHelpLines`).
 * - `/scope help <topic>`: what things mean, a topic at a time (`TOPICS`, `guideLines`); a word that names no topic
 *   finds the topic items and forms that mention it.
 * What a part of the dock shows is explained there too (the dock's ?).
 *
 * Lines are drawn by the "scope-guide" entry (entries.ts): "## " a heading; "| form\twhat it does" a row of a
 * table, the forms in a column; "· " a note; anything else a head line, its "\n"-separated body lines indented.
 */

import { MAX_STRENGTH, MIN_DOSE } from "./args.ts";
import { timeSpan } from "./args.ts";
import { JOB_MINUTES, type LearnBackend, calibrateTime, jobMinutes, learnTime } from "./concept.ts";
import { EXPLAIN, streamExplain } from "./draw.ts";
import { STOP_LABEL } from "./jobs.ts";
import { MAX_TEMPERATURE, REPLAY_MAX_TOKENS } from "./steer.ts";

export type Cmd = "scope" | "steer" | "concept" | "jlens";
type Row = [form: string, what: string];
type Command = { cmd: Cmd; what: string; forms: Row[]; notes: string[] };

/** The model Pi has selected, and its backend: the job times are its own when measured (concept.ts JOB_MINUTES). */
export type Served = { backend: LearnBackend; model?: string };

/** A job's time in the help's tables: the served model's (measured, or its backend's range); with none, both
 * backends' ranges. */
const learnRow = (s?: Served) => s ? learnTime(s.backend, s.model, "–") : `${learnTime("omlx", undefined, "–")}, ${learnTime("ds4", undefined, "–")}`;
const calibrateRow = (s?: Served) => s ? calibrateTime(s.backend, s.model, "–") : `${calibrateTime("omlx", undefined, "–")}, ${calibrateTime("ds4", undefined, "–")}`;

const row = ([form, what]: Row) => `| ${form}\t${what}`;
/** The models STREAM's help explains, as streamExplain's [streams written, sources read]: hyper-connections (Flash
 * Next, DeepSeek V4.1: 4 streams), one stream (the rest), then one whose layer visits read 5 sources (Inkling 2). */
export const HELP_STREAMS = [[4, 0], [1, 0], [1, 5]] as const;
/** STREAM's help for one of them: its line (whose, where it does not say), then each view's by name. */
export const streamHelp = (k: number, rk: number) => {
	const [shared, views] = streamExplain(k, rk);
	return [(k > 1 ? "" : rk ? "Inkling 2: " : "One stream: ") + shared, ...Object.entries(views).map(([v, t]) => `${v}: ${t}`)];
};
const note = (s: string) => `· ${s}`;

/** Each command: what it is for (also its autocomplete description), its forms, a few notes. In daily-use order.
 * `served`: the selected model, whose job times the forms give. */
export const commands = (served?: Served): Command[] => [
	{ cmd: "scope", what: "tracing, the dock, GPU jobs", forms: [
		["/scope on", "trace Pi's replies (on from the start)"],
		["/scope off", "stop tracing, hide the dock"],
		["/scope dock", "open the dock; again: step in; again: close (ctrl+\\)"],
		["/scope dock mini", "the mini dock: expert lights, MAP, SETTLING, STREAM (m in the dock)"],
		["/scope dock full", "the whole dock again"],
		["/scope jobs", "the GPU job running, or the last, then the last ten"],
		["/scope jobs cancel", "stop the running job"],
		["/scope · /scope status", "what is on"],
		["/scope help [topic]", "the overview, a topic, or a command's forms (/scope help scope)"],
	], notes: [
		"What it records and costs: /scope help tracing. /scope on traces Pi's own requests; metalscope on (a terminal) every client's.",
	] },
	{ cmd: "steer", what: "push the model toward a concept", forms: [
		["/steer fluffy", "steer at the concept's chat dose (/concept calibrate sets it)"],
		["/steer fluffy 0.5", `at another dose (${MIN_DOSE}–${MAX_STRENGTH}; also fluffy@0.5)`],
		["/steer angry + robotic", "two concepts at once (angry@0.4 + robotic@0.8: a dose each)"],
		["/steer again", "your last message answered again (a reply that ran tools: refused)"],
		["/steer replay", "your last reply beside the same message with no steering"],
		["/steer replay angry@0.6", "… beside it answered under that steer"],
		["/steer replay temp:0", "… at a temperature (0: greedy)"],
		["/steer off", "stop steering"],
		["/steer · /steer status", "what is on"],
	], notes: [
		"Every token, thinking included, until /steer off. In the dock: 6, the Steer page, enter.",
		"Concepts: /concept list. Doses, replay, the server switch: /scope help steering.",
	] },
	{ cmd: "concept", what: "the concept library: browse, make, calibrate", forms: [
		["/concept · /concept list", "learned here, ranked by good replies; then the library to learn"],
		["/concept NAME", "one concept: who wrote it, its good replies at each dose"],
		["/concept learn NAME <meaning>", `make a concept and calibrate it: ${learnRow(served)}`],
		["/concept learn NAME", "a library concept on this model (no description)"],
		["/concept learn NAME --corpus DIR", "from texts written elsewhere: concept.json, pos/neg.txt, task.json"],
		["/concept calibrate NAME", `find its chat dose in graded chats: ${calibrateRow(served)}`],
	], notes: [
		"learn asks first (oMLX: the model unloaded meanwhile; ds4: chat refused while it measures). One job at a time: /scope jobs follows it, /scope jobs cancel stops it.",
		"What learn and calibrate measure: /scope help concepts.",
	] },
	{ cmd: "jlens", what: "the model's Jacobian lens: check, switch, fit", forms: [
		["/jlens · /jlens status", "its J-lens, the other lenses ranked"],
		["/jlens off", "the logit lens alone (a little faster; no LENS edit)"],
		["/jlens on", "the J-lens again (the last used, else the best)"],
		["/jlens use NAME", "install a lens from the list"],
		["/jlens learn until 07:30", "fit one overnight: the model is unloaded until then"],
		["/jlens learn PATH until 07:30", "… on your own text (a folder, or one .txt)"],
		["/jlens learn … windows N", "… stop after N windows (fresh: start over)"],
	], notes: [
		"The logit lens needs nothing: every model has one. Which recorded lens is shown: s in the dock.",
		"learn asks first. More: /scope help lens.",
	] },
];
export const COMMANDS = commands();

type Item = [head: string, body: string[]];
type Topic = { name: string; title: string; items: Item[] };

/** The measured models on a backend (JOB_MINUTES), as " Measured: DeepSeek V4.1 60 to 90 min, then 3 to 6 h (4 to 7.5 h in all)." */
function measuredOn(backend: LearnBackend): string {
	const runs = JOB_MINUTES.filter((r) => r.backend === backend && r.model).map((r) => {
		const t = jobMinutes(backend, r.model);
		return `${r.name} ${timeSpan(t.learn)}, then ${timeSpan(t.calibrate)} (${timeSpan(t.whole)} in all)`;
	});
	return runs.length ? ` Measured: ${runs.join("; ")}.` : "";
}

/** What things mean, a topic at a time (`/scope help <name>`). */
export const TOPICS: Topic[] = [
	{ name: "tracing", title: "Tracing", items: [
		["off (/scope off)", ["Nothing recorded, nothing slower."]],
		["on (/scope on)", ["Per token: confidence and the top 10 next tokens, residual norms, expert routing, and each read layer's top 10 words through the logit lens and, while /jlens on, the J-lens. A prompt reads the lenses on its last 64 tokens only.",
			"On Flash Next a traced reply writes at 47% of untraced speed on oMLX and 58% on ds4 at medium (the default lens size), 77% and 80% at small, 92% on both with the lenses off (- and + on LENS); measured on the sizes before 2026-11-26. metalscope bench times your model.",
			"/scope on traces Pi's own requests, metalscope on every client's."]],
	] },
	{ name: "dock", title: "The dock", items: [
		["opening it", ["ctrl+\\ opens it, again steps in (the keys go to the dock, the newest still followed), again closes; /scope dock the same. alt+i steps straight in (the terminal must send Option as Alt).",
			"Shown in a terminal 100 columns wide or more; under 23 rows, the mini dock."]],
		["moving", ["←/→ or h/l a token · ⇧←/→ ten · ⌥ or ⌘ ←/→, or H/L, a sentence · page keys 25 · home the first · end (or → on the newest) follows the newest again · u/U the next or previous unsure token · esc back, q out at once."]],
		["the board", ["A dock 100 columns wide and 36 lines tall (a terminal 181 wide): the reply; TOKENS, MAP, LENS (SETTLING in it); EXPERTS, STREAM.",
			"tab the next panel · 1-5 open one (again: the board) · ? what it shows · 6 the Steer page · m the mini dock · s on STREAM its view · { } EXPERTS a layer less or more."]],
		["the one-panel board", ["A smaller dock, down to 44 columns and 23 lines: the reply's last lines and the panel in focus. The board's keys, the numbers choosing the panel, no STREAM."]],
		["the mini dock", ["m, or /scope dock mini: 50 wide: EXPERTS' layers, MAP, SETTLING and STREAM under them. [ ] more layers; s STREAM's view; stepped in, ←/→ a token; m the whole dock."]],
		["steering", ["The Steer page (6 on the board; 6 back): ↑/↓ a concept · ←/→ a dose (its test chats at it) · enter steers the next reply (as /steer) · x off · d then y deletes it on the checkpoint, every served copy (to the trash) · n/N a turn of its test chats · l, nothing learned here, links a same-shape model."]],
		["LENS", [EXPLAIN.SETTLING, EXPLAIN.TELESCOPE]],
		["MAP", [EXPLAIN.MAP]],
		["engram inspection", [EXPLAIN.ENGRAM, "On a finished reply, where the model has an engram (Flash Next, DeepSeek V4.1); g again stops or closes it."]],
		["EXPERTS", [EXPLAIN.EXPERTS, `Full size: ${EXPLAIN.EXPERTS_WINDOW.replace(/^C/, "c")}`, "MoE (mixture of experts) models only."]],
		["TOKENS", [EXPLAIN.TOKENS]],
		["STREAM", [...HELP_STREAMS.flatMap(([k, rk]) => streamHelp(k, rk)), "Where the server records what each layer writes and reads."]],
		["STEER", [EXPLAIN.STEER]],
		["test chats", [EXPLAIN.STEER_EXAMPLES]],
		["REPLY", [EXPLAIN.REPLY]],
	] },
	{ name: "lens", title: "The logit lens and the J-lens", items: [
		["the logit lens", ["Each layer read through the model's own output head. Every model has it; nothing to fit. Early layers mostly echo the input token."]],
		["the J-lens", ["A Jacobian lens: a read-out learned per model that reads the middle layers better. Fitted (/jlens learn) or installed (/jlens use); it also powers the LENS edit."]],
		["recorded and shown", ["Every traced reply records the logit lens, and the J-lens beside it while the J-lens is on (/jlens off, /jlens on).",
			"Which recorded lens is shown: s in the dock. It does not change what is recorded."]],
		["fitting one", ["/jlens learn holds out a tenth of the text and writes the lens after every window, with its held-out hit@10 against the logit lens and the lens in use, so stopping early leaves a usable lens. Flash Next took about 35 min a window. The same command another night resumes it.",
			"The default text is wikitext, plain and as deep chat turns, plus metalscope's source. Your own: a folder (each file a document; long ones read deep) or one .txt (each paragraph a document)."]],
		["none yet", ["/jlens learn fits one; published ones exist for about 40 open models (neuronpedia/jacobian-lens on Hugging Face), converted with scripts/convert_lens.py."]],
		["a read-out only", ["A lens never shapes a steer: concept directions come from residual means, the steer layer and dose from steering itself."]],
	] },
	{ name: "steering", title: "Steering", items: [
		["dose", [`A fraction of the layer's typical residual length (${MIN_DOSE}–${MAX_STRENGTH}). A concept's default is its chat dose from /concept calibrate (the most good replies). An uncalibrated concept has none: give a dose, or calibrate it.`]],
		["where and when", ["At the concept's reader layer, the full dose on every token, prompt and reply, thinking included, until /steer off. Changing model turns it off; the line above the editor shows what is on, /steer status all of it."]],
		["/steer replay", ["Your last reply beside the same message answered fresh, unsteered or under the steer given. It sees what your reply saw (the system prompt, every turn) at the session's thinking level, without tools; the model never sees the pair.",
			`One fresh sample at Pi's temperature (temp:T from 0 to ${MAX_TEMPERATURE}; 0 is greedy), at most ${REPLAY_MAX_TOKENS.toLocaleString("en")} tokens, not traced. So the steer and sampling are what differ.`,
			"Nothing is reused from your reply: steered requests skip the prefix cache, so earlier steered turns reach it as text."]],
		["the server's permission", ["The server refuses steering until it is allowed; /steer allows it the first time. metalscope steering off refuses it again."]],
	] },
	{ name: "concepts", title: "Concepts: learn, calibrate, profiles", items: [
		["/concept learn", [`The served model writes a corpus for the concept; a GPU job measures how cleanly the direction separates it, picks the layer to steer at, how far steering moves the model's choices against a random push, both ways, and where replies break. Then it is calibrated in conversation. It stops early only when it can't be steered: no item had room to move, or no layer to steer at.`,
			`On oMLX: ${timeSpan(jobMinutes("omlx").learn)} on its own copy, the served model unloaded; then the chat calibration (${timeSpan(jobMinutes("omlx").calibrate)}), chat working.${measuredOn("omlx")}`,
			`On ds4: ${timeSpan(jobMinutes("ds4").learn)} for the corpus and the measuring, chat refused while it measures the served model; then the chat calibration (${timeSpan(jobMinutes("ds4").calibrate)}), chat working.${measuredOn("ds4")}`]],
		["/concept calibrate", [`3 conversations of 6 messages behind a plain system prompt (reasoning effort low), steered at every token at doses 0.2, 0.4, 0.6, 0.8, then 0.1 either side of the best; the model grades every dose and the unsteered conversations side by side, blind. ${timeSpan(jobMinutes("omlx").calibrate)} of chat requests on oMLX, ${timeSpan(jobMinutes("ds4").calibrate)} on ds4; chat works meanwhile.`,
			"Good reply: style, coherence and usefulness 2 or more (of 3), on its first try, before the conversation breaks. A failed reply (tool call, no answer, too short) is asked again and never counts.",
			"The dose with the most good replies (the lower on a tie) is /steer's default; its good replies rank /concept list."]],
		["profile terms", ["presence: how well the direction separates concept from contrast sentences per layer (held-out d′, against 200 scrambled labellings).",
			"reader: the layer where adding the direction moves a forced choice most over a random control; where /steer injects.",
			"generation: every task item unsteered, steered at the test dose and with a control; the dose where fluency breaks. The test dose is not /steer's default: the chat calibration sets that."]],
	] },
	{ name: "jobs", title: "GPU jobs", items: [
		["one at a time", ["/jlens learn unloads the model while it runs (oMLX); /concept learn does on oMLX, and on ds4 refuses chat instead; /concept calibrate is chat requests. Each asks first. The widget above the editor shows the steps and the time left.",
			`/scope jobs: the running one, or the last, then the last ten. /scope jobs cancel (or ${STOP_LABEL}) stops it; a model it unloaded is loaded again.`]],
	] },
];

const commandOf = (cmd: Cmd, served?: Served) => commands(served).find((c) => c.cmd === cmd)!;

/** `/scope help`: one screen: start, the dock's keys, steering, learning a concept, the commands. `served`: the
 * selected model, whose learn time it gives. */
export function overviewHelp(served?: Served): string[] {
	return [
		"## Start",
		row(["/model", "pick a model marked (metalscope); Pi starts it"]),
		row(["ctrl+\\", "the dock; again: step in; again: close"]),
		note("Replies are traced from the start (/scope off stops it); chat a little first: doses are sized from them."),
		note("Not set up? In a terminal: metalscope omlx install (or ds4 install), metalscope pi install, then metalscope doctor."),
		"## The dock, stepped in",
		row(["←/→ · u/U", "a token · the next or previous unsure one"]),
		row(["tab · 1-5 · ?", "the next panel · open one · what it shows"]),
		row(["6 · m", "the Steer page · the mini dock"]),
		row(["s", "the other lens · on STREAM, its view"]),
		row(["enter · x", "LENS: remove or swap a word (experimental), redo from there · steering off"]),
		row(["esc", "back"]),
		"## Steer",
		row(["6 in the dock", "the Steer page: ↑/↓ a concept · ←/→ a dose · enter steers · x off"]),
		row(["/steer joyful", "the same by name · /steer off"]),
		row(["/steer replay", "your last reply beside the same message with no steering"]),
		"## Learn a concept",
		row(["/concept", "learned here, ranked, then the library"]),
		row(["/concept learn joyful", `learn one from the library: ${served ? learnTime(served.backend, served.model, "–") : learnTime("omlx", undefined, "–")}`]),
		"## Commands",
		...COMMANDS.map((c) => row([`/${c.cmd}`, c.what])),
		row(["/<command> help", "every form of one (/scope's: /scope help scope)"]),
		row(["/scope help <topic>", TOPICS.map((t) => t.name).join(" · ")]),
	];
}

const itemLine = ([head, body]: Item) => [head, ...body.map((b) => `    ${b}`)].join("\n");

/** `/scope help [topic]`: the overview; a topic by name (or its start: "steer" is steering); a command's forms by its
 * name where no topic has it (/scope help scope); else every topic item and command form that mentions the word. */
export function guideLines(topic?: string, served?: Served): string[] {
	const t = (topic || "").trim().toLowerCase();
	if (!t) return overviewHelp(served);
	const named = topicOf(t);
	if (named) return [`## ${named.title}`, ...named.items.map(itemLine)];
	const cmd = COMMANDS.find((c) => c.cmd === t.replace(/^\//, ""));
	if (cmd) return commandHelpLines(cmd.cmd, undefined, served);
	const out: string[] = [];
	for (const c of commands(served)) {
		const rows = c.forms.filter(([f, w]) => `${f} ${w}`.toLowerCase().includes(t));
		if (rows.length) out.push(`## /${c.cmd}`, ...rows.map(row));
	}
	for (const x of TOPICS) {
		const items = x.items.filter(([h, b]) => h.toLowerCase().includes(t) || b.some((l) => l.toLowerCase().includes(t)));
		if (items.length) out.push(`## ${x.title}`, ...items.map(itemLine));
	}
	return out.length ? out : [`Nothing in the help mentions "${topic}". Topics: ${TOPICS.map((x) => x.name).join(" · ")}.`];
}

/** `/<cmd> help`: its forms and notes. With `sub` (`/concept calibrate help`): the forms naming that subcommand. */
export function commandHelpLines(cmd: Cmd, sub?: string, served?: Served): string[] {
	const c = commandOf(cmd, served);
	if (!sub) return [`## /${cmd}: ${c.what}`, ...c.forms.map(row), ...c.notes.map(note)];
	const forms = c.forms.filter(([f]) => f.split(" · ").some((x) => x.startsWith(`/${cmd} ${sub}`)));
	return [`## /${cmd} ${sub}`, ...forms.map(row), note(`/${cmd} help: every form of /${cmd}.`)];
}

/** The topic a word names: by name, else by its start ("steer" is steering). */
const topicOf = (t: string) => TOPICS.find((x) => x.name === t) || TOPICS.find((x) => x.name.startsWith(t) || t.startsWith(x.name));

/** What /scope help completes: the topics, then the commands no topic takes the name of (scope, jlens). */
export const helpTopics = () => [...TOPICS.map((t) => t.name), ...COMMANDS.map((c) => c.cmd).filter((c) => !topicOf(c))];

/** A command's autocomplete description. */
export const describe = (cmd: Cmd) => `metalscope: ${commandOf(cmd).what} (/${cmd} help)`;

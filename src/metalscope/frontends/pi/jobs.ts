/**
 * The GPU jobs' display, every kind of job alike (job-runner.ts starts, polls and stops them): the confirm text before
 * a learn job, the progress lines the widget above the editor shows while a job runs, the result entry when it is
 * done (its steps frozen), and /scope jobs. A job is the server's `concept learn`, `concept calibrate` or `jlens learn`
 * (src/metalscope/learn/), and its state comes from GET /scope/api/jobs.
 */

import { decodeKittyPrintable, isKeyRelease, matchesKey, visibleWidth } from "@earendil-works/pi-tui";
import { f3, sig3, timeSpan } from "./args.ts";
import { type Ink, type Part, commandLine, cut, cutTo, giving, partsLine } from "./draw.ts";
import { type LearnBackend, type LearnPlan, chatTag, doseRow, jobMinutes, parseCalibration, parseStrength } from "./concept.ts";

/** A learn job that stopped early, before any calibration (learn/decide.py CANT_STEER). */
const CANT_STEER = "can't be steered";

/** A job's state as /scope/api/jobs returns it (src/metalscope/learn/job.py), the fields Pi reads. */
export interface LearnState {
	kind: "learn" | "calibrate" | "lens-learn";
	job: string;
	name: string;
	phase: string;
	progress?: number;
	started?: number;
	eta?: number | null;
	warning?: string | null;                // a phase far past its usual time (learn/job.py Job._pace), until the next phase
	outcome?: "done" | "failed" | "interrupted" | null;
	error?: string | null;
	seen?: boolean;
	origin: "pi" | "cli";                   // who started it: Pi announces only its own
	updated?: number;
	served_was_loaded?: boolean | null;
	window?: "omlx" | "ds4";                // a learn job on ds4 says so (learn/window.py Ds4Window); absent: oMLX
	// a learn job's texts: where they came from (absent from an older server: `stored`, the model's own or shipped)
	corpus?: { stored?: boolean; source?: LearnPlan["source"] };
	description?: string;                   // a learn job's description: typed, else the saved one, else the texts'
	supplied?: string;                      // a learn job's --corpus folder
	fit?: { lens: string } | null;          // a lens-learn job's last finished window (learn/lens_fit_gpu.py): the lens it wrote
	source?: string | null;                 // a lens-learn job's own text (/jlens learn PATH); none: the default corpus
	steps?: LearnStep[];                    // none in a job's first state (learn/spawn.py's stub)
	result?: LearnResult | CalibrateResult | LensLearnResult | null;
}

/** A lens-learn job's result (lens_fit_gpu.run): the lens, its windows in all, its held-out hit@10 in the workspace
 * band against the logit lens. */
export interface LensLearnResult {
	lens: string;
	windows: number;
	heldout?: { hit10: number; logit_hit10: number } | null;
}

/** A calibrate job's result: the calibration alone. */
export interface CalibrateResult {
	calibration: LearnCalibration;
}

type AnyResult = LearnResult | CalibrateResult | LensLearnResult;
const isLearnResult = (r: AnyResult): r is LearnResult => "steerable" in r;
const isLensLearnResult = (r: AnyResult): r is LensLearnResult => "windows" in r && "lens" in r;

/** Each kind of job as /scope jobs and the widget name it, what it is doing while it runs and what it did once done. */
const KINDS: Record<string, { label: string; doing: string; done: string }> = {
	learn: { label: "learn", doing: "learning", done: "learned" },
	calibrate: { label: "calibrate", doing: "calibrating", done: "calibrated" },
	"lens-learn": { label: "jlens learn", doing: "learning a lens for", done: "learned a lens for" },
};
/** A job's kind as a word ("learn"); a kind this Pi does not know by its own name. */
const kindLabel = (kind: string) => KINDS[kind]?.label ?? kind;
/** A job as the widget's head line and the stop key's question name it: "learning crunchy". */
export const doing = (st: Pick<LearnState, "kind" | "name">) => `${KINDS[st.kind]?.doing ?? st.kind} ${st.name}`;
/** A job as a note names it, before its own colon: "learn crunchy" ("learn crunchy: failed (…)"). */
export const jobName = (st: Pick<LearnState, "kind" | "name">) => `${kindLabel(st.kind)} ${st.name}`;

/** One line of the widget (src/metalscope/learn/job.py STEPS): its status, its one-line result once done,
 * the current sub-phase and progress while it runs. */
export interface LearnStep {
	label: string;
	status: "pending" | "running" | "done" | "failed" | "skipped";
	summary?: string | null;
	detail?: string | null;
	n?: number | null;
	total?: number | null;
	started?: number | null;
	ended?: number | null;
}

/** A learn job's result (learn/job.py LearnJob.measured): `steerable` false when it stopped early, `reason` saying why
 * ("can't be steered: ..."). */
export interface LearnResult {
	steerable: boolean;
	reason: string | null;
	calibration?: LearnCalibration; // the chat calibration in conversation (learn/calibrate.py); absent when it can't be steered
	steer?: { layer: number; strength: number; basis: "chat" } | null; // the dose /steer NAME now uses (its calibration's); null when not calibrated
}

/** The chat calibration's result (learn/calibrate.py), the fields the job list reads: the default dose and its good
 * replies of how many; or `error` when it failed. parseCalibration reads the rest (the doses, for the result entry)
 * and checks every field it reads itself. */
export interface LearnCalibration {
	version?: number;
	default_dose?: number | null;
	good?: number;
	turns?: number;
	error?: string;
	[key: string]: unknown;
}

/** Where a learn plan's texts come from as the confirm says it: "write" (the served model writes them) without a plan. */
const WRITE: Pick<LearnPlan, "source" | "texts" | "description" | "notes"> = { source: "write", texts: null, description: "" };
/** How many of the plan's notes the confirm says. */
const NOTES_MOST = 3;
/** A server's line as a sentence: its first letter up, a full stop. */
const sentence = (s: string) => { const t = s.trim(); return t && t[0].toUpperCase() + t.slice(1) + (/[.!?]$/.test(t) ? "" : "."); };

/** What the confirm dialog says before a learn job, one fact a line: whose texts it uses (`plan`, the server's
 * LearnPlan): the texts already on the model from an earlier run, metalscope's copied for it, or the ones in `corpus`
 * (--corpus) copied for it (none written; a description `typed` is ignored), else the served model writes its own
 * from the description typed or saved (`plan.description`); the plan's notes. Then what the job does to the model: on
 * oMLX it unloads it and loads it in a process of its own; on ds4 it measures the served model and chat is refused
 * meanwhile. The times are the model's when measured, else the backend's range (jobMinutes); a measured model's
 * whole job is said once more, in all. */
export function confirmText(name: string, typed: string, model: string, backend: LearnBackend = "omlx", plan = WRITE, corpus?: string): string {
	const result = "The widget above the editor shows progress; the result arrives here: its steps and the dose /steer uses.";
	const t = jobMinutes(backend, model);
	const inAll = t.measured ? [`About ${timeSpan(t.whole)} in all, measured on ${t.measured}.`] : [];
	const supplied = plan.source !== "write";
	const texts = [plan.texts || "texts: no record of who wrote them", ...(typed ? ["The description given is ignored: the texts have their own."] : [])];
	const said = typed || plan.description, stop = /[.!?]$/.test(said) ? "" : ".";
	const head = [
		...(plan.source === "own" ? [`"${name}" uses the texts already on ${model}, from an earlier run: none are written.`, ...texts]
			: plan.source === "shipped" ? [`"${name}" uses metalscope's texts, copied for ${model}: none are written.`, ...texts]
			: plan.source === "corpus" ? [`"${name}" uses the texts in ${corpus}, copied for ${model}: none are written.`]
			: [`${model} writes its own texts for "${name}" from ${typed ? "your" : "the saved"} description: ${said}${stop}`]),
		...(plan.notes ?? []).filter((n) => typeof n === "string" && n.trim()).slice(0, NOTES_MOST).map(sentence),
		backend === "ds4" ? `A GPU job measures it on ${model} in ds4.` : `A GPU job loads ${model} in a process of its own to measure it.`,
	];
	if (backend === "ds4") {
		return [
			...head,
			`ds4 keeps serving ${model}.`,
			supplied ? `The measuring takes up to about ${timeSpan(t.learn)}.` : `The corpus and the measuring take about ${timeSpan(t.learn)}.`,
			`Then the chat calibration: about ${timeSpan(t.calibrate)}.`,
			...inAll,
			"While it measures the model, chat requests are refused with the job's message.",
			"A reply in progress finishes first.",
			supplied ? "Chat works while it calibrates the chat dose." : "Chat works while it writes the corpus and while it calibrates the chat dose.",
			"The next message after it re-reads the conversation (slower once).",
			result,
		].join("\n");
	}
	return [
		...head,
		`${model} is unloaded for about ${timeSpan(t.learn)}: chat pauses, replies in progress are aborted, and model loads are refused until it is back.`,
		`Then the chat calibration, chat working: about ${timeSpan(t.calibrate)}.`,
		...inAll,
		result,
	].join("\n");
}

const GPU_PHASES = new Set(["locking", "unloading", "waiting for memory", "loading the model", "connecting", "presence", "measuring the items",
	"selection sweep", "fluency ladder", "test sweep", "building the corpus", "scoring the logit lens", "sizing the batch", "fitting",
	"releasing", "reloading"]);

/** A text progress bar of `width` cells. */
export function progressBar(frac: number, width = 20): string {
	const f = Math.max(0, Math.min(1, frac || 0));
	const full = Math.round(f * width);
	return "█".repeat(full) + "░".repeat(width - full);
}

function minutesLeft(eta: number | null | undefined, now: number): string {
	if (!eta) return "";
	const m = Math.max(1, Math.round((eta - now) / 60));
	return `about ${m} min left`;
}

/** A duration as "45s", "3m 12s" or (an overnight lens fit) "6h 05m". */
export function duration(seconds: number): string {
	const s = Math.max(0, Math.round(seconds));
	if (s >= 3600) return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
	return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

const MARK: Record<LearnStep["status"], string> = { done: "✓", running: "▸", pending: "·", failed: "✗", skipped: "–" };

/** The key that stops the running job (job-runner.ts listens for it) and its label: option+x on a Mac, alt+x elsewhere. */
const STOP_KEY = "alt+x";
export const keyLabel = (platform: string) => (platform === "darwin" ? "option+x" : "alt+x");
export const STOP_LABEL = keyLabel(process.platform);
/** Option+x where the terminal does not send Option as Alt (Terminal.app and iTerm2 by default): ≈ on US, UK and most
 * Latin layouts. */
const OPTION_X = "≈";
/** Raw terminal input that is the stop key: alt+x as Pi parses it, or ≈ (plain or Kitty's); never a key release. */
export const isStopKey = (data: string) =>
	!isKeyRelease(data) && (matchesKey(data, STOP_KEY) || data === OPTION_X || decodeKittyPrintable(data) === OPTION_X);
/** The stop key's question: its answers, the first (enter's) keeps the job. */
export const STOP_ASK = ["No, keep it running", "Yes, stop it"];
/** The braille spinner the widget turns in place of the running step's ▸. */
export const SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";

/** How the job lines are drawn: in the widget, Pi's theme, `width` columns, the spinner's frame and the stop key's label
 * (job-runner.ts); /scope jobs' note, plain at WIDTH with ▸ and no stop key. */
export interface JobLook {
	width: number;
	th: Ink;
	spin?: string;
	key?: string;
}
const WIDTH = 116;
const PLAIN: JobLook = { width: WIDTH, th: { fg: (_c, s) => s, bold: (s) => s } };

const drop = () => "";

/** One step as a widget line: mark, label, then its result and time (done, dim; failed, ✗ in the error colour), or
 * its progress: the detail gives way first, then the bar, then n/total, the clock kept (running); a step not reached,
 * dim. `row`: parts drawn in place of a finished step's result, the last giving way first (the result entry's doses). */
function stepLine(s: LearnStep, now: number, look: JobLook, row?: Part[]): string {
	const { th, width } = look, label = s.label;
	if (s.status === "pending" || s.status === "skipped") return partsLine([{ t: `  ${MARK[s.status]} ${label}`, c: "dim" }], [], width, th);
	const spin = s.status === "running" && look.spin;
	const mark: Part[] = [{ t: "  " }, { t: spin || MARK[s.status], c: spin ? "accent" : s.status === "done" ? "success" : s.status === "failed" ? "error" : undefined },
		{ t: ` ${label.padEnd(11)}` }];
	if (s.status === "running") {
		const detail = s.detail ?? "", frac = (s.n ?? 0) / (s.total || 1);
		const clock: Part[] = [{ t: s.started != null ? ` · ${duration(now - s.started)}` : "" }];
		if (!s.total) return partsLine([...mark, { t: ` ${detail || "…"}`, fit: (w) => (w < 3 ? "" : ` ${cut(detail || "…", w - 1)}`) }, ...clock], [3], width, th);
		return partsLine([...mark, { t: ` ${progressBar(frac, 16)}`, fit: (w) => (w < 5 ? "" : ` ${progressBar(frac, w - 1)}`) },
			{ t: ` ${s.n ?? 0}/${s.total}`, fit: drop }, { t: detail && ` · ${detail}`, fit: (w) => (w < 6 ? "" : ` · ${cut(detail, w - 3)}`) }, ...clock],
		[5, 3, 4], width, th);
	}
	// A step done in under a second: no time.
	const res = s.summary ?? "", took = s.started != null && s.ended != null && s.ended - s.started >= 1 ? duration(s.ended - s.started) : "";
	const shown = row ?? (res ? [{ t: res, c: s.status === "failed" ? undefined : "dim" }] : []);
	return partsLine([...mark, { t: " " }, ...giving(shown), { t: took && (shown.length ? ` · ${took}` : took), c: "dim" }],
		shown.map((_, i) => mark.length + shown.length - i), width, th).trimEnd();
}

/** The widget's lines while a job runs: a head line (the name bold, the overall bar and percent, the time left,
 * whether chat is paused, then the stop key `look.key` as " · option+x stop", the key in the normal colour: the bar
 * gives way first, then the key, then the time left, then the name), the job's warning when a phase runs far past its
 * usual time,
 * then every step (stepLine). With a spinner frame (`look.spin`: the widget) the running step turns; without, /scope
 * jobs' plain lines, ▸ and no key. A job's first state has no steps yet. */
export function learnWidgetLines(st: LearnState, now = Date.now() / 1000, look: JobLook = PLAIN): string[] {
	const { th, width } = look, steps = st.steps ?? [], p = st.progress ?? 0;
	// Chat pauses while oMLX's model is unloaded (until it is back), or while ds4 holds chat for the job (the lock).
	const holds = st.window === "ds4" || (st.served_was_loaded && st.phase !== "reloading");
	const paused = holds && GPU_PHASES.has(st.phase) ? " · chat paused" : "";
	const left = minutesLeft(st.eta, now);
	const hint: Part[] = look.key ? [{ t: " · ", c: "dim" }, { t: look.key }, { t: " stop", c: "dim" }] : [];
	const parts: Part[] = [{ t: doing(st), c: "bold", fit: cutTo(doing(st)) },
		{ t: ` · ${progressBar(p)} `, fit: (w) => ` · ${w < 5 ? "" : `${progressBar(p, w - 4)} `}` }, { t: `${Math.round(100 * p)}%` },
		{ t: left && ` · ${left}`, fit: drop }, { t: paused }];
	// The key only where the rest fits whole beside a bar of four.
	const whole = [...parts, ...hint].reduce((n, q, i) => n + visibleWidth(i === 1 ? q.fit!(8) : q.t), 0) <= width;
	const head = partsLine(whole ? [...parts, ...hint] : parts, [1, 3, 0], width, th);
	const warn = st.warning ? [partsLine([{ t: `  ⚠ ${st.warning}` }], [], width, th)] : [];
	return [head, ...warn, ...steps.map((s) => stepLine(s, now, look))];
}

const DAY = 24 * 3600;

/** What Pi does at startup about the latest job, so a fresh session stays clean:
 * - "watch": it is running, so show its progress widget;
 * - "notify": it finished while Pi was closed, was started from Pi, is unseen and under a day old: one line, no entry;
 * - "none": anything else (a command-line job, an old or already shown result). */
export function startupAction(st: LearnState | undefined, now = Date.now() / 1000): "watch" | "notify" | "none" {
	if (!st) return "none";
	if (st.outcome == null) return "watch";
	if (st.seen || st.origin === "cli") return "none";
	return now - (st.updated ?? st.started ?? 0) < DAY ? "notify" : "none";
}

/** The one line shown at startup for a result Pi did not see arrive. */
export function startupNotice(st: LearnState): string {
	return `${kindLabel(st.kind)} ${st.name} finished while Pi was closed: ${jobOutcome(st)} · /scope jobs shows the result`;
}

/** A finished job in a few words: a learn or calibrate job's chat finding (a learn job that stopped early: "can't be
 * steered"), else how it ended. */
export function jobOutcome(st: LearnState): string {
	const r = st.result;
	if (!r || st.outcome !== "done") return `${st.outcome ?? "not finished"}${st.error ? ` (${st.error})` : ""}`;
	if (isLearnResult(r)) return r.steerable ? calibrationOutcome(r.calibration) : CANT_STEER;
	if (isLensLearnResult(r)) return fitOutcome(r.windows, r.heldout?.hit10, r.heldout?.logit_hit10);
	return "calibration" in r ? calibrationOutcome(r.calibration) : st.outcome;
}

/** A lens fit in a few words: "11 windows · hit@10 0.214 vs logit lens 0.148". */
function fitOutcome(windows: number | null | undefined, hit10?: number | null, logit?: number | null): string {
	return `${windows ?? 0} window${windows === 1 ? "" : "s"}` + (hit10 != null && logit != null ? ` · hit@10 ${f3(hit10)} vs logit lens ${f3(logit)}` : "");
}

/** A chat calibration in a few words, as a tag (chatTag): "16/18 good @0.4", "not calibrated". */
export function calibrationOutcome(c: Pick<LearnCalibration, "default_dose" | "good" | "turns" | "error"> | null | undefined): string {
	if (!c || c.error || c.default_dose == null || c.good == null || c.turns == null) return "not calibrated";
	return chatTag({ good: c.good, turns: c.turns, dose: c.default_dose });
}

/** One job of /scope jobs' list as GET /scope/api/jobs gives it (`jobs`, the last ten, oldest first). */
export interface JobBrief {
	job: string;
	kind: string;
	name: string;
	phase?: string;
	outcome?: LearnState["outcome"];
	started?: number;
	steerable?: boolean | null;     // a learn job: false when it stopped early
	calibration?: Pick<LearnCalibration, "default_dose" | "good" | "turns"> | null;
	fit?: { windows?: number | null; hit10?: number | null; logit_hit10?: number | null; lens?: string | null } | null;   // a lens fit
}

/** A local date and time, "2026-10-07 14:02". */
function when(t: number): string {
	const d = new Date(t * 1000);
	const p = (n: number) => String(n).padStart(2, "0");
	return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

/** One job in a line: its kind and name, then its chat score (or "can't be steered"), how it ended, or (running) its
 * phase. */
export function jobBriefLine(j: JobBrief): string {
	const what = j.kind === "learn" && j.outcome === "done" && j.steerable === false ? CANT_STEER
		: (j.kind === "calibrate" || j.kind === "learn") && j.outcome === "done" ? calibrationOutcome(j.calibration)
		: j.kind === "lens-learn" && j.fit?.windows ? `${j.outcome ?? `running (${j.phase ?? "starting"})`}, ${fitOutcome(j.fit.windows, j.fit.hit10, j.fit.logit_hit10)}`
		: j.outcome ?? `running (${j.phase ?? "starting"})`;
	return `${kindLabel(j.kind)} ${j.name}: ${what}${j.started ? ` · ${when(j.started)}` : ""}`;
}

/** What /scope jobs shows: with no job, a line saying what starts one; with one running, its widget lines (and that
 * /scope jobs cancel stops it), then the last ten newest first, as a note; otherwise the last job's result entry (what the
 * widget's finish adds to the transcript, its state trimmed: trimJob) with the last ten under it (`recent`, its head
 * line first; none without a list). */
export function jobsView(st: LearnState | undefined, list: JobBrief[], now = Date.now() / 1000):
	| { kind: "none" | "running"; text: string }
	| { kind: "done"; job: LearnState; recent: string[] } {
	if (!st) return { kind: "none", text: "no jobs yet: /concept learn, /concept calibrate and /jlens learn start one" };
	const jobs = [...list].reverse().map(jobBriefLine);
	const recent = jobs.length ? [`last ${jobs.length === 1 ? "job" : `${jobs.length} jobs`}, newest first`, ...jobs] : [];
	if (st.outcome == null) return { kind: "running", text: [...learnWidgetLines(st, now), "/scope jobs cancel stops it.",
		...(recent.length ? ["", `${recent[0]}:`, ...jobs.map((l) => `  ${l}`)] : [])].join("\n") };
	return { kind: "done", job: trimJob(st), recent };
}

/** The status-line text while a job runs. */
export function learnStatus(st: LearnState): string {
	return `${kindLabel(st.kind)}: ${st.name} ${Math.round(100 * (st.progress ?? 0))}%`;
}

/** The steps a result entry leaves out: they say nothing of the result (the model unloaded, loaded, reloaded; ds4
 * held and released; a lens fit's batch sized). */
const SILENT = new Set(["unload", "load model", "reload", "hold ds4", "release", "batch"]);
/** A step the result entry shows: one that ran and says something of the result (a failed one only with a summary). */
const shown = (s: LearnStep) => !SILENT.has(s.label) && (s.status === "done" || (s.status === "failed" && !!s.summary));

/** What a job's result entry keeps of its state (the "scope-job" entry, entries.ts): the fields jobEntryLines draws,
 * never the whole result with its samples and curves; a calibration's doses only where well formed (parseStrength). */
export function trimJob(st: LearnState): LearnState {
	const r = st.result, c = r && "calibration" in r ? r.calibration : undefined;
	const doses = Array.isArray(c?.strengths) ? c.strengths.map(parseStrength).filter((s) => s !== null) : undefined;
	const cal = c && { version: c.version, layer: c.layer, default_dose: c.default_dose, good: c.good, turns: c.turns, error: c.error, strengths: doses };
	const result = !r ? r : isLensLearnResult(r) ? { lens: r.lens, windows: r.windows, heldout: r.heldout && { hit10: r.heldout.hit10, logit_hit10: r.heldout.logit_hit10 } }
		: isLearnResult(r) ? { steerable: r.steerable, reason: r.reason, steer: r.steer, calibration: cal } : cal ? { calibration: cal } : null;
	return { kind: st.kind, job: st.job, name: st.name, phase: st.phase, origin: st.origin, outcome: st.outcome, error: st.error, started: st.started,
		updated: st.updated, description: st.description, supplied: st.supplied, corpus: st.corpus && { source: st.corpus.source, stored: st.corpus.stored },
		source: st.source, fit: st.fit && { lens: st.fit.lens }, result,
		steps: st.steps?.filter(shown).map(({ label, status, summary, started, ended }) => ({ label, status, summary, started, ended })) };
}

/** What /concept learn NAME needs after it to run a failed learn job again: nothing for texts on the model (its own,
 * or shipped), --corpus for supplied ones, else its description (typed or saved: said either way). */
function againArgs(st: LearnState): string {
	const src = st.corpus?.source;
	if (src === "own" || src === "shipped" || (!src && st.corpus?.stored)) return "";
	if (st.supplied) return ` --corpus ${st.supplied}`;
	const d = st.description?.trim();
	return !d ? " <what it means>" : d.includes('"') ? ` ${d}` : ` "${d}"`;
}

/** The one command that goes further from a finished job, as command forms and their words (commandLine): the dose
 * /steer now uses, or (a learn job not calibrated) /concept calibrate; the lens to use; a job that did not finish, how
 * to run it again (a lens fit: /jlens learn in a form it takes, on the job's own text, after the lens it kept).
 * Nothing for a learn job that can't be steered. */
export function nextStep(st: LearnState): string[] {
	const r = st.result;
	const fit = `/jlens learn ${st.source ? `${st.source} ` : ""}until HH:MM`;
	if (st.outcome !== "done") return st.kind === "calibrate" ? [`/concept calibrate ${st.name}`, " runs it again"]
		: st.kind === "lens-learn" ? (st.fit ? [`/jlens use ${st.fit.lens}`, " · ", fit, " resumes the fit"] : [fit, " starts it again"])
		: st.kind === "learn" ? [`/concept learn ${st.name}${againArgs(st)}`, " starts it again"] : [];
	if (!r) return [];
	if (isLensLearnResult(r)) return [`/jlens use ${r.lens}`];
	if (isLearnResult(r)) return !r.steerable ? [] : r.steer ? [`/steer ${st.name}`, ` uses dose ${sig3(r.steer.strength)}`] : [`/concept calibrate ${st.name}`, " runs it again"];
	const dose = !r.calibration?.error ? r.calibration?.default_dose : null;
	return dose == null ? [] : [`/steer ${st.name}`, ` uses dose ${sig3(dose)}`];
}

/** A finished job as its transcript entry draws it, the widget's look frozen: a head line (✓ and what it did, bold,
 * then its outcome (jobOutcome) and its whole time, dim; a job that did not finish or a learn job that can't be steered:
 * ✗, and why), the steps that say something of the result (stepLine; a calibration's calibrate step, its doses: doseRow;
 * a failed step that says the head's error again, left out), then the one command that goes further (nextStep). */
export function jobEntryLines(st: LearnState, look: JobLook): string[] {
	const { th, width } = look, r = st.outcome === "done" ? st.result : null, error = st.error ?? "";
	const cant = !!r && isLearnResult(r) && !r.steerable, done = st.outcome === "done" && !cant;
	const what = done ? `${KINDS[st.kind]?.done ?? kindLabel(st.kind)} ${st.name}` : cant ? jobName(st) : `${jobName(st)} ${st.outcome ?? "not finished"}`;
	const said = done ? jobOutcome(st) : cant ? (r as LearnResult).reason || CANT_STEER : error;
	const took = st.started != null && st.updated != null ? ` · ${duration(st.updated - st.started)}` : "";
	const head = partsLine([{ t: done ? "✓" : "✗", c: done ? "success" : cant ? "warning" : "error" }, { t: " " }, { t: what, c: "bold", fit: cutTo(what) },
		{ t: said && ` · ${said}`, c: "dim", fit: (w) => (w < 5 ? "" : ` · ${cut(said, w - 3)}`) }, { t: took, c: "dim" }], [3, 2], width, th);
	const again = (s: LearnStep) => st.outcome !== "done" && s.status === "failed" && !!error && !!s.summary && (s.summary.includes(error) || error.includes(s.summary));
	// A calibration that ran: its doses in place of its calibrate step's summary.
	const c = r && "calibration" in r && r.calibration && !r.calibration.error ? parseCalibration(r.calibration) : null;
	const steps = (st.steps ?? []).filter((s) => shown(s) && !again(s)).map((s) => stepLine(s, 0, look, c && s.label === "calibrate" ? doseRow(c) : undefined));
	const next = nextStep(st);
	return [head, ...steps, ...(next.length ? [commandLine(next, width, th)] : [])];
}

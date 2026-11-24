import assert from "node:assert/strict";
import { test } from "node:test";
import { type JobBrief, type JobLook, type LearnState, calibrationOutcome, confirmText, duration, jobBriefLine, jobEntryLines, jobOutcome, jobsView,
	learnStatus, learnWidgetLines, nextStep, progressBar, startupAction, startupNotice, isStopKey, keyLabel, STOP_LABEL, trimJob } from "../jobs.ts";
import { visibleWidth } from "@earendil-works/pi-tui";
import { CAL, LOW } from "./fixtures/concepts.ts";

/** A theme whose colours are codes the test can find; the plain text without them. */
const TH = { fg: (c: string, s: string) => `\x1b[3${{ success: 2, accent: 6, dim: 7, error: 1, warning: 3, muted: 5 }[c] ?? 9}m${s}\x1b[39m`, bold: (s: string) => `\x1b[1m${s}\x1b[22m` };
const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");


test("confirmText says the model is unloaded and chat pauses", () => {
	const t = confirmText("crunchy", "crisp textures", "Flash");
	assert.match(t, /Flash is unloaded for about 5 to 35 min: chat pauses/);
	assert.match(t, /\nThen the chat calibration, chat working: about 20 min to 2 h\.\n/);
	assert.equal(t, confirmText("crunchy", "crisp textures", "Flash", "omlx"));
});

test("confirmText: a model with measured times gets them, and the whole job once; others the backend's range", () => {
	const qwen = confirmText("joyful", "", "Qwen3.6-35B-A3B-MLX-4bit", "omlx").split("\n");
	assert.ok(qwen.some((l) => l.startsWith("Qwen3.6-35B-A3B-MLX-4bit is unloaded for about 5 min: chat pauses")), qwen.join("\n"));
	assert.ok(qwen.includes("Then the chat calibration, chat working: about 20 min."));
	assert.ok(qwen.includes("About 26 min in all, measured on Qwen3.6."));
	const v41 = confirmText("joyful", "", "DeepSeek-V4.1-Flash-ds4-q2", "ds4").split("\n");
	assert.ok(v41.includes("The corpus and the measuring take about 60 to 90 min."));
	assert.ok(v41.includes("Then the chat calibration: about 3 to 6 h."));
	assert.ok(v41.includes("About 4 to 7.5 h in all, measured on DeepSeek V4.1."));
	assert.doesNotMatch(confirmText("joyful", "", "Flash", "omlx"), /in all|measured/);
});

test("confirmText on ds4: the model keeps being served, chat is refused, one fact per line", () => {
	const lines = confirmText("crunchy", "crisp textures", "V41", "ds4").split("\n");
	assert.deepEqual(lines.slice(0, 9), [
		'V41 writes its own texts for "crunchy" from your description: crisp textures.',
		"A GPU job measures it on V41 in ds4.",
		"ds4 keeps serving V41.",
		"The corpus and the measuring take about 60 to 90 min.",
		"Then the chat calibration: about 3 to 6 h.",
		"While it measures the model, chat requests are refused with the job's message.",
		"A reply in progress finishes first.",
		"Chat works while it writes the corpus and while it calibrates the chat dose.",
		"The next message after it re-reads the conversation (slower once).",
	]);
	assert.doesNotMatch(lines.join("\n"), /unload/);
});

test("confirmText: each case says whose texts are used, one fact per line", () => {
	const fable = "texts: hand-written by Claude Fable 5.1 (2026-10-05)";
	// The texts already on the model, from an earlier run: none written; on ds4 the measuring alone said.
	const own = confirmText("pirate", "", "V41", "ds4", { source: "own", texts: "texts: written by V41 (2026-10-24)", description: "Talking like a pirate." }).split("\n");
	assert.deepEqual(own.slice(0, 5), ['"pirate" uses the texts already on V41, from an earlier run: none are written.', "texts: written by V41 (2026-10-24)",
		"A GPU job measures it on V41 in ds4.", "ds4 keeps serving V41.", "The measuring takes up to about 60 to 90 min."]);
	assert.ok(own.includes("Chat works while it calibrates the chat dose."));
	assert.doesNotMatch(own.join("\n"), /corpus and the measuring/);
	// metalscope's texts, copied for this model: none written; on oMLX the job loads the model itself.
	const shipped = confirmText("fluffy", "", "Qwen3.6", "omlx", { source: "shipped", texts: fable, description: "Soft, fluffy." });
	assert.match(shipped, /^"fluffy" uses metalscope's texts, copied for Qwen3\.6: none are written\.\ntexts: hand-written by Claude Fable 5\.1 \(2026-10-05\)\nA GPU job loads Qwen3\.6 in a process of its own to measure it\.\nQwen3\.6 is unloaded/);
	assert.doesNotMatch(shipped, /writes its own|ignored/);
	// A description typed for existing texts: said to be ignored, under whose texts.
	const typed = confirmText("fluffy", "soft things", "V41", "ds4", { source: "shipped", texts: fable, description: "Soft, fluffy." }).split("\n");
	assert.equal(typed[2], "The description given is ignored: the texts have their own.");
	// The served model writes its own: from the description typed, else the saved one (shown).
	assert.equal(confirmText("toasty", "", "Qwen3.6", "omlx", { source: "write", texts: null, description: "Toasty things." }).split("\n")[0],
		'Qwen3.6 writes its own texts for "toasty" from the saved description: Toasty things.');
	assert.equal(confirmText("toasty", "warm and crisp", "Qwen3.6", "omlx", { source: "write", texts: null, description: "Toasty things." }).split("\n")[0],
		'Qwen3.6 writes its own texts for "toasty" from your description: warm and crisp.');
	// No texts line said: no record.
	assert.equal(confirmText("old", "", "Q", "omlx", { source: "own", texts: null, description: "" }).split("\n")[1], "texts: no record of who wrote them");
	// Texts supplied from a directory (--corpus).
	const sup = confirmText("kept", "", "V41", "ds4", { source: "corpus", texts: null, description: "Kept." }, "/u/texts/kept").split("\n");
	assert.deepEqual(sup.slice(0, 3), ['"kept" uses the texts in /u/texts/kept, copied for V41: none are written.', "A GPU job measures it on V41 in ds4.", "ds4 keeps serving V41."]);
	assert.ok(sup.includes("Chat works while it calibrates the chat dose."));
	assert.doesNotMatch([...own, shipped, ...typed, ...sup].join("\n"), /stored|own texts, written/);
	// The plan's notes (the dose units' warning among them), a line each after the texts, three at most.
	const noted = confirmText("fluffy", "", "Q", "omlx", { source: "shipped", texts: fable, description: "", notes: ["the dose units came from 2 short traces: doses may be off", "two.", "three", "four"] }).split("\n");
	assert.deepEqual(noted.slice(1, 6), [fable, "The dose units came from 2 short traces: doses may be off.", "Two.", "Three.", "A GPU job loads Q in a process of its own to measure it."]);
	assert.equal(confirmText("w", "x", "Q", "omlx", { source: "write", texts: null, description: "", notes: ["note"] }).split("\n")[1], "Note.");
});

test("a ds4 learn job: chat paused while it holds the lock", () => {
	const st: LearnState = { kind: "learn", job: "j", name: "crunchy", phase: "connecting", progress: 0.02, window: "ds4", origin: "pi",
		steps: [{ label: "hold ds4", status: "running", detail: "connecting", started: 990 }] };
	for (const phase of ["locking", "connecting", "presence", "selection sweep", "test sweep", "releasing"])
		assert.match(learnWidgetLines({ ...st, phase }, 1000)[0], / · chat paused$/, phase);
	for (const phase of ["writing", "calibrating"])
		assert.doesNotMatch(learnWidgetLines({ ...st, phase }, 1000)[0], /chat paused/, phase);
	// oMLX with nothing unloaded: chat runs whatever the phase.
	assert.doesNotMatch(learnWidgetLines({ ...st, window: undefined, served_was_loaded: false, phase: "presence" }, 1000)[0], /chat paused/);
});

test("a failed learn job's command: the name alone for texts on the model, --corpus for supplied ones, else its description", () => {
	const st: LearnState = { kind: "learn", job: "j", name: "crunchy", phase: "failed", outcome: "failed", error: "x", origin: "pi", steps: [] };
	const again = (more: object) => nextStep({ ...st, ...more } as LearnState).join("");
	assert.equal(again({ corpus: { source: "shipped", stored: true } }), "/concept learn crunchy starts it again");
	assert.equal(again({ corpus: { source: "own", stored: true } }), "/concept learn crunchy starts it again");
	assert.equal(again({ corpus: { stored: true } }), "/concept learn crunchy starts it again");          // an older server: stored
	assert.equal(again({ corpus: { source: "write", stored: false } }), "/concept learn crunchy <what it means> starts it again");
	assert.equal(again({ supplied: "/u/texts/crunchy" }), "/concept learn crunchy --corpus /u/texts/crunchy starts it again");
	assert.equal(again({ description: "crisp textures that snap" }), '/concept learn crunchy "crisp textures that snap" starts it again');
	assert.equal(again({ description: 'the "snap" of it' }), '/concept learn crunchy the "snap" of it starts it again');
	assert.equal(again({}), "/concept learn crunchy <what it means> starts it again");
	assert.deepEqual(nextStep({ ...st, outcome: "interrupted" }), ["/concept learn crunchy <what it means>", " starts it again"]);
});

test("a slow phase's warning shows under the head line; no item with room to move: can't be steered, and why", () => {
	const warning = "selection sweep: estimated 8.0 h, expected about 76 min; something is off, consider stopping it";
	const st: LearnState = { kind: "learn", job: "j", name: "candid", phase: "selection sweep", progress: 0.3, window: "ds4", origin: "cli",
		steps: [{ label: "selection", status: "running", n: 400, total: 7200, started: 0 }] };
	assert.equal(learnWidgetLines(st, 1000).length, 2);
	const lines = learnWidgetLines({ ...st, warning }, 1000);
	assert.equal(lines[1], `  ⚠ ${warning}`);
	assert.match(lines[2], /selection/);
	const why = "V41 already picks the candid option on 59 of 60 items unsteered; rephrase the contrast so the " +
		"model doesn't already do it (or the other side), then learn it again under a new name";
	// Finished, but nothing to steer with: ✗ in the warning colour, why, its steps; no command.
	const done: LearnState = { ...st, phase: "done", outcome: "done", started: 0, updated: 1000, result: { steerable: false, reason: `can't be steered: ${why}` },
		steps: [{ label: "presence", status: "done", summary: "held-out d′ 4.10 at L30 (p 0.005)", started: 0, ended: 40 },
			{ label: "items", status: "done", summary: "0 usable of 60 · push not testable", started: 40, ended: 100 },
			{ label: "selection", status: "skipped" }, { label: "calibrate", status: "skipped" }] };
	assert.equal(jobOutcome(done), "can't be steered");
	const look: JobLook = { width: 400, th: TH };
	assert.deepEqual(jobEntryLines(done, look).map(plain), [`✗ learn candid · can't be steered: ${why} · 16m 40s`,
		"  ✓ presence    held-out d′ 4.10 at L30 (p 0.005) · 40s", "  ✓ items       0 usable of 60 · push not testable · 1m 00s"]);
	assert.match(jobEntryLines(done, look)[0], /^\x1b\[33m✗\x1b\[39m \x1b\[1mlearn candid\x1b\[22m\x1b\[37m · can't be steered: /);
	assert.deepEqual(nextStep(done), []);
});

test("progressBar fills proportionally", () => {
	assert.equal(progressBar(0.5, 10), "█████░░░░░");
	assert.equal(progressBar(2, 4), "████");
	assert.equal(progressBar(NaN, 4), "░░░░");
});

test("learnWidgetLines with steps: one line per step, results once done, progress while running", () => {
	const st: LearnState = { kind: "learn", job: "j", name: "righteous", phase: "selection sweep", progress: 0.56, eta: 1000 + 6 * 60, served_was_loaded: true, origin: "pi",
		steps: [
			{ label: "corpus", status: "done", summary: "16 pairs, 60 items · content · contrast: secular moral pragmatism", started: 0, ended: 97 },
			{ label: "items", status: "done", summary: "13 usable of 60 (9 against, 4 favour) · 38 already chosen, 2 never", started: 97, ended: 100 },
			{ label: "selection", status: "running", n: 965, total: 1440, started: 400 },
			{ label: "unload", status: "running", detail: "waiting for memory", started: 990 },
			{ label: "fluency", status: "pending" },
			{ label: "test", status: "skipped" },
		] };
	const lines = learnWidgetLines(st, 1000);
	assert.equal(lines[0], "learning righteous · ███████████░░░░░░░░░ 56% · about 6 min left · chat paused");
	assert.equal(lines[1], "  ✓ corpus      16 pairs, 60 items · content · contrast: secular moral pragmatism · 1m 37s");
	assert.equal(lines[2], "  ✓ items       13 usable of 60 (9 against, 4 favour) · 38 already chosen, 2 never · 3s");
	assert.equal(lines[3], "  ▸ selection   ███████████░░░░░ 965/1440 · 10m 00s");
	assert.equal(lines[4], "  ▸ unload      waiting for memory · 10s");
	assert.equal(lines[5], "  · fluency");
	assert.equal(lines[6], "  – test");
	assert.ok(lines.every((l) => l.length <= 116));
	assert.equal(learnStatus(st), "learn: righteous 56%");
	assert.doesNotMatch(learnWidgetLines({ ...st, phase: "writing" }, 1000)[0], /chat paused/);   // not a GPU phase: chat runs
});


test("the stop key's label: option+x on a Mac, alt+x elsewhere", () => {
	assert.equal(keyLabel("darwin"), "option+x");
	assert.equal(keyLabel("linux"), "alt+x");
	assert.equal(STOP_LABEL, keyLabel(process.platform));
});

test("the stop key: alt+x as Pi parses it, or ≈ (Option+x on a Mac terminal that does not send Option as Alt), plain or Kitty's; never a release", () => {
	for (const k of ["\x1bx", "\x1b[120;3u", "≈", "\x1b[8776u", "\x1b[8776;1u", "\x1b[8776;1:2u"]) assert.ok(isStopKey(k), JSON.stringify(k));
	for (const k of ["x", "\x1b[120;3:3u", "\x1b[8776;1:3u", "≈≈", "\x1b[200~≈\x1b[201~", "\x03", "\x1bi", ""]) assert.ok(!isStopKey(k), JSON.stringify(k));
});

test("the widget's lines: the theme's colours, the spinner on the running step, the stop key on the head line, every step, cut never wrapped", () => {
	const steps = ["corpus", "unload", "load model", "presence", "items", "selection", "fluency", "test", "calibrate"].map((label, i) =>
		({ label, status: i < 5 ? "done" : i === 5 ? "running" : "pending", summary: i === 2 ? null : `${label} result`, started: 100 * i, ended: i < 5 ? 100 * i + 2 : null,
			...(i === 5 ? { n: 7, total: 18, detail: "selection sweep" } : {}) } as const));
	const st: LearnState = { kind: "learn", job: "j", name: "pirate", phase: "selection sweep", progress: 0.56, eta: 1000 + 21 * 60, served_was_loaded: true,
		origin: "pi", steps: [...steps, { label: "reload", status: "pending" }, { label: "extra", status: "skipped" }] };
	const look = { width: 98, th: TH, spin: "⠹", key: "option+x" };
	const lines = learnWidgetLines(st, 1000, look);
	assert.equal(lines.length, 12);                                                              // every step: no 10-line cut
	assert.deepEqual(lines.map(plain).slice(0, 4), [
		"learning pirate · ███████████░░░░░░░░░ 56% · about 21 min left · chat paused · option+x stop",
		"  ✓ corpus      corpus result · 2s",
		"  ✓ unload      unload result · 2s",
		"  ✓ load model  2s"]);                                                                    // no stray " · " without a summary
	assert.equal(plain(lines[6]), "  ⠹ selection   ██████░░░░░░░░░░ 7/18 · selection sweep · 8m 20s");
	assert.equal(plain(lines[7]), "  · fluency");
	assert.match(lines[0], /^\x1b\[1mlearning pirate\x1b\[22m · /);                            // the name bold
	assert.match(lines[0], /\x1b\[37m · \x1b\[39moption\+x\x1b\[37m stop\x1b\[39m$/);           // the key in the normal colour
	assert.match(lines[1], /^ {2}\x1b\[32m✓\x1b\[39m corpus {6}\x1b\[37mcorpus result/);         // ✓ success, the result dim
	assert.match(lines[6], /^ {2}\x1b\[36m⠹\x1b\[39m selection /);
	assert.match(lines[7], /^\x1b\[37m {2}· fluency\x1b\[39m$/);
	assert.match(lines[11], /^\x1b\[37m {2}– extra\x1b\[39m$/);
	// Narrow. The head line: the bar shrinks first (to nothing), then the key goes (it shows only beside the time left
	// and a bar of four), then the time left, then the name is cut; chat paused and the percent stay. The running step:
	// the detail goes, then the bar shrinks (to nothing), then n/total; the clock stays.
	const at = (width: number, s = st, key = "option+x") => learnWidgetLines(s, 1000, { ...look, width, key }).map(plain);
	const NARROW: [number, string, string][] = [
		[30, "learning … · 56% · chat paused", "  ⠹ selection   7/18 · 8m 20s"],
		[40, "learning pirate · ██░░ 56% · chat paused", "  ⠹ selection   ████░░░░░░ 7/18 · 8m 20s"],
		[50, "learning pirate · ████████░░░░░░ 56% · chat paused", "  ⠹ selection   ██████░░░░░░░░░░ 7/18 · 8m 20s"],
		[60, "learning pirate · ██░░ 56% · about 21 min left · chat paused", "  ⠹ selection   ██████░░░░░░░░░░ 7/18 · selection … · 8m 20s"],
		[75, "learning pirate · ███████████░░░░░░░░ 56% · about 21 min left · chat paused", plain(lines[6])],
		[76, "learning pirate · ██░░ 56% · about 21 min left · chat paused · option+x stop", plain(lines[6])],
		[80, "learning pirate · ████░░░░ 56% · about 21 min left · chat paused · option+x stop", plain(lines[6])],
		[99, lines.map(plain)[0], plain(lines[6])],
	];
	for (const [width, head, step] of NARROW) assert.deepEqual([at(width)[0], at(width)[6]], [head, step], String(width));
	assert.equal(at(73, st, "alt+x")[0], "learning pirate · ██░░ 56% · about 21 min left · chat paused · alt+x stop");   // a shorter label, sooner
	assert.equal(at(28)[6], "  ⠹ selection   · 8m 20s");
	const lens = { ...st, name: "Qwen3.6-35B-A3B-MLX-4bit-and-then-some" };
	assert.equal(at(56, lens)[0], "learning Qwen3.6-35B-A3B-MLX-4bit-a… · 56% · chat paused");
	assert.equal(at(96, lens)[0], "learning Qwen3.6-35B-A3B-MLX-4bit-and-then-some · ████░░░░ 56% · about 21 min left · chat paused");
	assert.equal(at(110, lens)[0], "learning Qwen3.6-35B-A3B-MLX-4bit-and-then-some · ███░░░ 56% · about 21 min left · chat paused · option+x stop");
	const long: LearnState = { ...st, warning: "x".repeat(120),
		steps: st.steps!.map((s) => (s.status === "running" ? { ...s, detail: "35 min a window · hit@10 0.214 vs logit 0.148, in use 0.207" } : s)) };
	for (const key of ["option+x", "alt+x"]) for (let width = 1; width <= 160; width++) {
		const cut = learnWidgetLines(long, 1000, { ...look, width, key });
		assert.ok(cut.every((l) => visibleWidth(l) <= width), `${width}\n${cut.map(plain).join("\n")}`);
		// The key from where it fits beside the time left and a bar of four; narrower, the line without it.
		const keyed = width >= (key === "option+x" ? 76 : 73);
		if (width >= 40) assert.match(plain(cut[0]), keyed ? new RegExp(` 56% · about 21 min left · chat paused · ${key.replace("+", "\\+")} stop$`) : / 56%( · about 21 min left)? · chat paused$/, String(width));
		if (width >= 30) assert.match(plain(cut[7]), / · 8m 20s$/);
	}
	// A failed step: ✗ in the error colour, its summary not dim. Text from the job is made safe: no newline or control
	// character reaches the terminal.
	const failed = learnWidgetLines({ ...st, warning: "slow\nvery\x07", steps: [{ label: "items", status: "failed", summary: "no item\nmoved\x1b[2J", started: 0, ended: 3 }] }, 1000, look);
	assert.match(failed[2], /^ {2}\x1b\[31m✗\x1b\[39m items {7}no item moved·\[2J\x1b\[37m · 3s\x1b\[39m$/);
	assert.equal(plain(failed[1]), "  ⚠ slow very·");
	// A job just launched: its first state has no steps yet.
	assert.deepEqual(learnWidgetLines({ kind: "calibrate", job: "c", name: "pirate", phase: "starting", progress: 0, origin: "pi", outcome: null }, 1000, look).map(plain),
		["calibrating pirate · ░░░░░░░░░░░░░░░░░░░░ 0% · option+x stop"]);
	// A step running or none: the key ends the head line, and only it.
	const between = learnWidgetLines({ ...st, steps: st.steps!.map((s) => (s.status === "running" ? { ...s, status: "pending" } : s)) }, 1000, look);
	for (const ls of [lines, between]) {
		assert.match(plain(ls[0]), / · chat paused · option\+x stop$/);
		assert.ok(!ls.slice(1).some((l) => /option\+x/.test(l)));
	}
	// /scope jobs: the same lines, plain, ▸ and no key.
	assert.deepEqual(learnWidgetLines(st, 1000).slice(1, 4), lines.slice(1, 4).map(plain));
	assert.equal(learnWidgetLines(st, 1000)[0], "learning pirate · ███████████░░░░░░░░░ 56% · about 21 min left · chat paused");
	assert.equal(learnWidgetLines(st, 1000)[6], "  ▸ selection   ██████░░░░░░░░░░ 7/18 · selection sweep · 8m 20s");
	// /scope jobs' note is plain text, cut by its width: no escape byte (a reset would end the note's own colour).
	const view = jobsView({ ...st, steps: st.steps!.map((s) => ({ ...s, summary: s.summary && s.summary.repeat(20) })) }, [], 1000);
	assert.ok(view.kind === "running" && !view.text.includes("\x1b") && view.text.includes("…"), JSON.stringify(view));
});

test("startupAction keeps a fresh session clean", () => {
	const base: LearnState = { kind: "learn", job: "j", name: "x", phase: "done", outcome: "done", updated: 1000, seen: false, origin: "pi", steps: [] };
	assert.equal(startupAction(undefined, 1000), "none");
	assert.equal(startupAction({ ...base, outcome: null, phase: "selection sweep" }, 1000), "watch");
	assert.equal(startupAction(base, 1000 + 60), "notify");                       // started from Pi, missed
	assert.equal(startupAction({ ...base, origin: "cli" }, 1000 + 60), "none");   // a command-line job is not announced
	assert.equal(startupAction({ ...base, seen: true }, 1000 + 60), "none");
	assert.equal(startupAction(base, 1000 + 2 * 24 * 3600), "none");               // over a day old
	assert.match(startupNotice({ ...base, result: { steerable: true, reason: null, calibration: LOW } }),
		/^learn x finished while Pi was closed: 8\/18 good @0\.2 · \/scope jobs shows the result$/);
	assert.match(startupNotice({ ...base, result: { steerable: true, reason: null } }),
		/^learn x finished while Pi was closed: not calibrated · /);
});

/** A learn job as the server leaves it, done: its steps (the model unloaded, loaded and reloaded among them), its
 * result with a calibration (a dose that ran away beside CAL's) and what the entry never draws (samples, a curve). */
const LEARNED = {
	kind: "learn", job: "pirate-1", name: "pirate", model: "M", phase: "done", outcome: "done", origin: "pi", started: 0, updated: 4788,
	description: "talking like a pirate", corpus: { pairs: 16, items: 60, kind: "style", contrast: "plain English", stored: false, source: "write" },
	steps: [
		{ label: "corpus", status: "done", summary: "16 pairs, 60 items · style · contrast: plain English", detail: "writing the task items", started: 0, ended: 303 },
		{ label: "unload", status: "done", summary: "M unloaded · 98 GB free", started: 303, ended: 304 },
		{ label: "load model", status: "done", summary: null, started: 304, ended: 309 },
		{ label: "presence", status: "done", summary: "held-out d′ 3.58 at L2 (p 0.005)", started: 309, ended: 312 },
		{ label: "test", status: "done", summary: "+2.18 log-odds over control · picked more often, 25% to 44% (a random direction: 28%)", started: 312, ended: 362 },
		{ label: "reload", status: "done", summary: "M is back", started: 362, ended: 364 },
		{ label: "calibrate", status: "done", summary: "16/18 good @0.4", started: 364, ended: 4615 },
	],
	result: { steerable: true, reason: null, layer: 0, evidence: ["presence: held-out d′ 3.58 at L2, p 0.005"], curve: [{ strength: 0.1, concept: 2, control: 0 }],
		samples: [{ prompt: "p", base: "b", concept: "c", control: "x" }], steer: { layer: 20, strength: 0.4, basis: "chat" },
		calibration: { ...CAL, strengths: [...CAL.strengths, { dose: 1, good: 0, turns: 18, attempts: 30, failed: { tool: 0, thinking: 12, short: 0 }, coherence: null, runaway: 12.6 }],
			sample: { arc: "dinner", turn: 2, prompt: "How are you?", dose: 0.4, plain: "Fine.", steered: "Arr.", good: true } } },
} as unknown as LearnState;
const look = (width = 140): JobLook => ({ width, th: TH });

test("a learn job's result entry: the head line, the steps that say something of the result, its doses for calibrate, the dose /steer uses", () => {
	const lines = jobEntryLines(LEARNED, look());
	assert.deepEqual(lines.map(plain), [
		"✓ learned pirate · 16/18 good @0.4 · 1h 19m",
		"  ✓ corpus      16 pairs, 60 items · style · contrast: plain English · 5m 03s",
		"  ✓ presence    held-out d′ 3.58 at L2 (p 0.005) · 3s",
		"  ✓ test        +2.18 log-odds over control · picked more often, 25% to 44% (a random direction: 28%) · 50s",
		"  ✓ calibrate   0.2 2/18 · 0.3 11/18 · 0.4 16/18 · 0.6 15/18 · 0.8 4/18 · 1 ran 13x the plain length, not graded · 1h 10m",
		"  /steer pirate uses dose 0.4",
	]);
	// ✓ in the success colour, what finished bold, the rest dim; the default dose in the normal colour; the command in the
	// accent colour, its words dim.
	assert.equal(lines[0], "\x1b[32m✓\x1b[39m \x1b[1mlearned pirate\x1b[22m\x1b[37m · 16/18 good @0.4\x1b[39m\x1b[37m · 1h 19m\x1b[39m");
	assert.match(lines[4], /\x1b\[37m0\.3 11\/18\x1b\[39m\x1b\[37m · \x1b\[39m0\.4 16\/18\x1b\[37m · \x1b\[39m\x1b\[37m0\.6 15\/18\x1b\[39m/);
	assert.equal(lines[5], "  \x1b[36m/steer pirate\x1b[39m\x1b[35m uses dose 0.4\x1b[39m");
	// Not calibrated (no dose to suggest): its outcome, the step's own summary, no command. A calibration that failed, or
	// of a version this Pi does not read: the step's summary too.
	const cal = (calibration: any, steer: any = null) => ({ ...LEARNED, result: { ...(LEARNED.result as any), calibration, steer } }) as LearnState;
	const failed = jobEntryLines(cal({ error: "HTTP 500: busy" }), look()).map(plain);
	assert.equal(failed[0], "✓ learned pirate · not calibrated · 1h 19m");
	assert.deepEqual(failed.slice(-2), ["  ✓ calibrate   16/18 good @0.4 · 1h 10m", "  /concept calibrate pirate runs it again"]);   // no dose: calibrate again
	assert.deepEqual(nextStep(cal(undefined)), ["/concept calibrate pirate", " runs it again"]);
	assert.equal(jobEntryLines(cal({ ...CAL, version: 3 }, { layer: 20, strength: 0.4, basis: "chat" }), look()).map(plain)[4], "  ✓ calibrate   16/18 good @0.4 · 1h 10m");
	// A step that never ran is left out.
	const skipped = { ...LEARNED, steps: [...LEARNED.steps!.slice(0, 4), { label: "test", status: "skipped" }, { label: "calibrate", status: "pending" }] } as LearnState;
	assert.deepEqual(jobEntryLines(skipped, look()).map(plain).map((l) => l.slice(0, 13)), ["✓ learned pir", "  ✓ corpus   ", "  ✓ presence ", "  /steer pira"]);
});

test("a result entry's state is trimmed: the fields it draws, never the samples and curves; drawn the same", () => {
	const t = trimJob(LEARNED);
	assert.deepEqual(jobEntryLines(t, look()), jobEntryLines(LEARNED, look()));
	const json = JSON.stringify(t);
	for (const gone of ["samples", "curve", "evidence", "sample", "detail", "attempts", "unloaded", "M is back"]) assert.ok(!json.includes(gone), gone);
	assert.deepEqual(t.steps!.map((s) => s.label), ["corpus", "presence", "test", "calibrate"]);
	assert.deepEqual(t.result, { steerable: true, reason: null, steer: { layer: 20, strength: 0.4, basis: "chat" }, calibration: { version: 4, layer: 20, default_dose: 0.4,
		good: 16, turns: 18, error: undefined, strengths: [...CAL.strengths.map(({ dose, good, turns, coherence }) => ({ dose, good, turns, coherence })), { dose: 1, good: 0, turns: 18, coherence: null, runaway: 12.6 }] } });
	// A dose that is not well formed is left out, never thrown on.
	const odd = { ...LEARNED, result: { ...(LEARNED.result as any), calibration: { ...CAL, strengths: [null, 3, "x", { dose: "0.2" }, CAL.strengths[2]] } } } as LearnState;
	assert.deepEqual((trimJob(odd).result as any).calibration.strengths, [{ dose: 0.4, good: 16, turns: 18, coherence: 2.9 }]);
	assert.deepEqual(t.corpus, { source: "write", stored: false });
	assert.ok(json.length < 1500, String(json.length));
});

test("a result entry at any width: never wider; the outcome gives way before what finished, a step's result before its time", () => {
	for (const st of [LEARNED, { ...LEARNED, outcome: "failed", error: "x".repeat(200), result: null } as LearnState])
		for (let width = 1; width <= 160; width++) {
			const ls = jobEntryLines(st, look(width));
			assert.ok(ls.every((l) => visibleWidth(l) <= width), `${width}\n${ls.map(plain).join("\n")}`);
		}
	assert.deepEqual(jobEntryLines(LEARNED, look(40)).map(plain), [
		"✓ learned pirate · 16/18 good … · 1h 19m",
		"  ✓ corpus      16 pairs, 60 i… · 5m 03s",
		"  ✓ presence    held-out d′ 3.58 a… · 3s",
		"  ✓ test        +2.18 log-odds ov… · 50s",
		"  ✓ calibrate   0.2 2/18 · 0.3… · 1h 10m",
		"  /steer pirate uses dose 0.4",
	]);
	assert.equal(plain(jobEntryLines(LEARNED, look(20))[0]), "✓ learned … · 1h 19m");             // the outcome gone first, then the name cut
});

test("a failed or stopped job's entry: ✗, how it ended and why; a failed step only with a summary; how to run it again", () => {
	const failed = { ...LEARNED, phase: "failed", outcome: "failed", error: "the model gave no JSON after 3 tries", updated: 217, result: null,
		steps: [{ label: "corpus", status: "failed", summary: null, started: 0, ended: 217 }, { label: "unload", status: "pending" }] } as LearnState;
	const lines = jobEntryLines(failed, look());
	assert.deepEqual(lines.map(plain), ["✗ learn pirate failed · the model gave no JSON after 3 tries · 3m 37s", '  /concept learn pirate "talking like a pirate" starts it again']);
	assert.match(lines[0], /^\x1b\[31m✗\x1b\[39m \x1b\[1mlearn pirate failed\x1b\[22m\x1b\[37m · /);
	// A failed step with a summary is shown, ✗ in the error colour; the steps done before it too.
	const later = { ...failed, steps: [LEARNED.steps![0], { label: "presence", status: "failed", summary: "no layer told the texts apart", started: 303, ended: 400 }] } as LearnState;
	assert.deepEqual(jobEntryLines(later, look()).map(plain).slice(1, 3), ["  ✓ corpus      16 pairs, 60 items · style · contrast: plain English · 5m 03s",
		"  ✗ presence    no layer told the texts apart · 1m 37s"]);
	// Stopped, no time known: how it ended alone.
	assert.equal(plain(jobEntryLines({ ...failed, outcome: "interrupted", error: null, updated: undefined }, look())[0]), "✗ learn pirate interrupted");
});

test("a result entry: a failed step that says the head's error again is left out; a step done in under a second shows no time", () => {
	const failed = { ...LEARNED, outcome: "failed", error: "stopped", result: null,
		steps: [{ ...LEARNED.steps![0], ended: 0.4 }, { label: "presence", status: "failed", summary: "stopped", started: 1, ended: 3 }] } as LearnState;
	assert.deepEqual(jobEntryLines(failed, look()).map(plain).slice(0, 2), ["✗ learn pirate failed · stopped · 1h 19m",
		"  ✓ corpus      16 pairs, 60 items · style · contrast: plain English"]);
	assert.equal(jobEntryLines(failed, look()).length, 3);                              // the head, corpus, the command
	const said = { ...failed, steps: [{ label: "calibrate", status: "failed", summary: "not calibrated: HTTP 500: busy", started: 1, ended: 3 }], error: "HTTP 500: busy" } as LearnState;
	assert.equal(jobEntryLines(said, look()).length, 2);
	const other = { ...said, error: "the GPU run exited with code 1" } as LearnState;
	assert.equal(plain(jobEntryLines(other, look())[1]), "  ✗ calibrate   not calibrated: HTTP 500: busy · 2s");
});

test("a result entry: text from the job on one line, made safe, the command's arguments too", () => {
	const odd = { ...LEARNED, name: "pi\trate", outcome: "failed", error: "no\nJSON\x1b[2J", result: null, steps: [],
		description: "talking\nlike a\tpirate \x1b]8;;x\x07" } as LearnState;
	const lines = jobEntryLines(odd, look());
	assert.ok(lines.every((l) => !/[\n\t\x07]/.test(l) && !/\x1b(?!\[[0-9;]*m)/.test(l)), JSON.stringify(lines));
	assert.deepEqual(lines.map(plain), ["✗ learn pi rate failed · no JSON·[2J · 1h 19m", '  /concept learn pi rate "talking like a pirate ·]8;;x·" starts it again']);
	const lens = { kind: "lens-learn", job: "l", name: "M", phase: "failed", origin: "pi", outcome: "failed", error: "x", fit: { lens: "M\nlearned" }, source: "/u/a\tb" } as LearnState;
	assert.equal(plain(jobEntryLines(lens, look()).at(-1)!), "  /jlens use M learned · /jlens learn /u/a b until HH:MM resumes the fit");
	assert.equal(plain(jobEntryLines({ ...odd, supplied: "/u/x\ny" } as LearnState, look()).at(-1)!), "  /concept learn pi rate --corpus /u/x y starts it again");
});

test("a calibrate job: its widget, status line, outcome and result entry", () => {
	const running: LearnState = { kind: "calibrate", job: "a-calibrate-1", name: "anime-nerd", phase: "calibrating", progress: 0.5, eta: 1000 + 4 * 60, origin: "pi",
		started: 900, steps: [{ label: "calibrate", status: "running", n: 25, total: 51, started: 900 }] };
	const w = learnWidgetLines(running, 1000);
	assert.match(w[0], /^calibrating anime-nerd · █{10}░{10} 50% · about 4 min left$/);   // never "chat paused": chat requests only
	assert.match(w[1], /▸ calibrate .* 25\/51 · 1m 40s$/);
	assert.equal(learnStatus(running), "calibrate: anime-nerd 50%");
	const done: LearnState = { ...running, outcome: "done", phase: "done", updated: 2770, result: { calibration: CAL },
		steps: [{ label: "calibrate", status: "done", summary: "16/18 good @0.4", started: 900, ended: 2770 }] };
	assert.equal(jobOutcome(done), "16/18 good @0.4");
	assert.equal(jobOutcome({ ...done, result: { calibration: LOW } }), "8/18 good @0.2");
	assert.equal(calibrationOutcome(null), "not calibrated");                          // a job brief without a calibration
	assert.equal(calibrationOutcome({ error: "stopped" }), "not calibrated");
	assert.equal(startupNotice(done), "calibrate anime-nerd finished while Pi was closed: 16/18 good @0.4 · /scope jobs shows the result");
	assert.deepEqual(jobEntryLines(done, look()).map(plain), ["✓ calibrated anime-nerd · 16/18 good @0.4 · 31m 10s",
		"  ✓ calibrate   0.2 2/18 · 0.3 11/18 · 0.4 16/18 · 0.6 15/18 · 0.8 4/18 · 31m 10s", "  /steer anime-nerd uses dose 0.4"]);
	assert.equal(plain(jobEntryLines({ ...done, result: { calibration: LOW } }, look()).at(-1)!), "  /steer anime-nerd uses dose 0.2");
	const failed: LearnState = { ...running, outcome: "failed", updated: 1000, error: "HTTP 409: the model is unloaded",
		steps: [{ label: "calibrate", status: "failed", summary: "not calibrated: HTTP 409", started: 900, ended: 1000 }] };
	assert.deepEqual(jobEntryLines(failed, look()).map(plain), ["✗ calibrate anime-nerd failed · HTTP 409: the model is unloaded · 1m 40s",
		"  ✗ calibrate   not calibrated: HTTP 409 · 1m 40s", "  /concept calibrate anime-nerd runs it again"]);
});

test("/scope jobs: the running or last job, then the last ten newest first", () => {
	const at = (s: string) => s.replace(/\d{4}-\d\d-\d\d \d\d:\d\d/g, "DATE");
	const list: JobBrief[] = [
		{ job: "a-1", kind: "learn", name: "crunchy", outcome: "done", steerable: true, started: 1000, calibration: { default_dose: 0.6, good: 15, turns: 18 } },
		{ job: "b-1", kind: "calibrate", name: "anime-nerd", outcome: "done", started: 2000, calibration: { default_dose: 0.4, good: 16, turns: 18 } },
		{ job: "c-1", kind: "learn", name: "otaku", outcome: "failed", started: 3000 },
		{ job: "d-1", kind: "learn", name: "stoic", phase: "selection sweep", outcome: null, started: 4000 },
	];
	assert.equal(at(jobBriefLine(list[0])), "learn crunchy: 15/18 good @0.6 · DATE");
	assert.equal(at(jobBriefLine({ ...list[0], steerable: false, calibration: null })), "learn crunchy: can't be steered · DATE");
	assert.equal(at(jobBriefLine({ ...list[0], calibration: null })), "learn crunchy: not calibrated · DATE");
	assert.equal(at(jobBriefLine(list[1])), "calibrate anime-nerd: 16/18 good @0.4 · DATE");
	assert.equal(at(jobBriefLine(list[2])), "learn otaku: failed · DATE");
	assert.equal(jobBriefLine({ ...list[3], started: undefined }), "learn stoic: running (selection sweep)");
	assert.deepEqual(jobsView(undefined, []), { kind: "none", text: "no jobs yet: /concept learn, /concept calibrate and /jlens learn start one" });
	// Running: its widget lines, how to stop it, then the list, newest first.
	const running: LearnState = { kind: "learn", job: "d-1", name: "stoic", phase: "selection sweep", progress: 0.4, eta: 1000 + 9 * 60, origin: "pi",
		served_was_loaded: true, steps: [{ label: "selection", status: "running", n: 10, total: 100, started: 900 }] };
	const r = jobsView(running, list, 1000);
	assert.equal(r.kind, "running");
	if (r.kind === "running") assert.deepEqual(at(r.text).split("\n"), ["learning stoic · ████████░░░░░░░░░░░░ 40% · about 9 min left · chat paused",
		"  ▸ selection   ██░░░░░░░░░░░░░░ 10/100 · 1m 40s", "/scope jobs cancel stops it.", "", "last 4 jobs, newest first:",
		"  learn stoic: running (selection sweep) · DATE", "  learn otaku: failed · DATE", "  calibrate anime-nerd: 16/18 good @0.4 · DATE",
		"  learn crunchy: 15/18 good @0.6 · DATE"]);
	// Finished: the result entry the widget's finish adds (the state trimmed), the list under it, its head line first.
	const failed: LearnState = { kind: "learn", job: "c-1", name: "otaku", phase: "failed", outcome: "failed", error: "only 3 pairs passed the checks", origin: "pi", steps: [] };
	const d = jobsView(failed, list.slice(0, 3));
	assert.equal(d.kind, "done");
	if (d.kind === "done") {
		assert.deepEqual(d.job, trimJob(failed));
		assert.deepEqual(d.recent.map(at), ["last 3 jobs, newest first", "learn otaku: failed · DATE", "calibrate anime-nerd: 16/18 good @0.4 · DATE",
			"learn crunchy: 15/18 good @0.6 · DATE"]);
	}
	const one = jobsView(failed, list.slice(2, 3));
	assert.equal(one.kind === "done" && one.recent[0], "last job, newest first");
	// No list: no "last 0 jobs" line, finished or running.
	assert.deepEqual(jobsView(failed, []), { kind: "done", job: trimJob(failed), recent: [] });
	const bare = jobsView(running, [], 1000);
	assert.ok(bare.kind === "running" && bare.text.endsWith("/scope jobs cancel stops it.") && !bare.text.includes("last"), JSON.stringify(bare));
});

const SERVED = "Qwen3.8-Flash-Next-oQ4e-mtp";

test("/scope jobs during a lens-learn: the fit line shows the window and its held-out scores; the result, a cancel, the list", () => {
	const fit = { n: 3, of: 12, approx: true, minutes: 34.6, hit10: 0.214, logit_hit10: 0.148, compared: "Qwen3.8-Flash-Next-4bit", compared_hit10: 0.207,
		kind: "chat", depth: 3011, lens: `${SERVED}-learned-20261014`, path: `/l/${SERVED}-learned-20261014` };
	const running: LearnState = { kind: "lens-learn", job: "lens-learn-x-1", name: SERVED, origin: "pi", phase: "fitting", progress: 0.3,
		served_was_loaded: true, fit, eta: 1000 + 5 * 3600, started: 0, steps: [
			{ label: "unload", status: "done", summary: `${SERVED} unloaded · 112 GB free`, started: 0, ended: 20 },
			{ label: "load model", status: "done", started: 20, ended: 80 },
			{ label: "corpus", status: "done", summary: "the default corpus: 406 wikitext paragraphs … · 8 held-out windows", started: 80, ended: 82 },
			{ label: "compare", status: "done", summary: "held-out hit@10 in the workspace band: logit lens 0.148, Qwen3.8-Flash-Next-4bit (in use) 0.207", started: 82, ended: 200 },
			{ label: "batch", status: "done", summary: "2 output dims a pass · 1.54s a pass · peak 91 GB (cap 107) · about 33.1 min for a window 6144 tokens deep", started: 200, ended: 400 },
			{ label: "fit", status: "running", n: 3, total: 12, started: 400,
				detail: "35 min a window · hit@10 0.214 vs logit 0.148, in use 0.207" },
			{ label: "reload", status: "pending" }] } as LearnState;
	const list: JobBrief[] = [{ job: "other-m-1", kind: "other", name: SERVED, outcome: "done", started: 1 },
		{ job: "lens-learn-x-1", kind: "lens-learn", name: SERVED, phase: "fitting", started: 2, fit: { windows: 3, hit10: 0.214, logit_hit10: 0.148 } }];
	const view = jobsView(running, list, 1000);
	assert.equal(view.kind, "running");
	const text = (view as { text: string }).text.split("\n");
	assert.equal(text[0], `learning a lens for ${SERVED} · ██████░░░░░░░░░░░░░░ 30% · about 300 min left · chat paused`);
	assert.equal(text[6], "  ▸ fit         ████░░░░░░░░░░░░ 3/12 · 35 min a window · hit@10 0.214 vs logit 0.148, in use 0.207 · 10m 00s");
	assert.equal(duration(6 * 3600 + 5 * 60 + 30), "6h 05m");
	assert.ok(text.includes("/scope jobs cancel stops it."));
	assert.match(text.at(-2)!, /^ {2}jlens learn Qwen3\.8-Flash-Next-oQ4e-mtp: running \(fitting\), 3 windows · hit@10 0\.214 vs logit lens 0\.148 · /);
	assert.equal(learnStatus(running), `jlens learn: ${SERVED} 30%`);
	// Done: what it learned, its steps but the model's and the batch's, the lens to use.
	const steps = running.steps!.map((s) => (s.label === "fit" ? { ...s, status: "done", ended: 5990,
		summary: "11 windows this run, 11 in all · stopped at the deadline · held-out hit@10 0.219 vs logit lens 0.148" } : { ...s, status: "done" }));
	const done = { ...running, phase: "done", outcome: "done", updated: 6000, steps, result: { lens: fit.lens, path: fit.path, windows: 11, this_run: 11, stopped: "deadline",
		heldout: { hit10: 0.219, logit_hit10: 0.148, windows: 8, kinds: { chat: 0.197, code: 0.121 }, compared: { lens: "Qwen3.8-Flash-Next-4bit", hit10: 0.207 } } } } as LearnState;
	assert.equal(jobOutcome(done), "11 windows · hit@10 0.219 vs logit lens 0.148");
	assert.deepEqual(jobEntryLines(done, look()).map(plain), [
		`✓ learned a lens for ${SERVED} · 11 windows · hit@10 0.219 vs logit lens 0.148 · 1h 40m`,
		"  ✓ corpus      the default corpus: 406 wikitext paragraphs … · 8 held-out windows · 2s",
		"  ✓ compare     held-out hit@10 in the workspace band: logit lens 0.148, Qwen3.8-Flash-Next-4bit (in use) 0.207 · 1m 58s",
		"  ✓ fit         11 windows this run, 11 in all · stopped at the deadline · held-out hit@10 0.219 vs logit lens 0.148 · 1h 33m",
		`  /jlens use ${SERVED}-learned-20261014`,
	]);
	assert.deepEqual(trimJob(done).result, { lens: fit.lens, windows: 11, heldout: { hit10: 0.219, logit_hit10: 0.148 } });
	assert.deepEqual(trimJob(done).fit, { lens: fit.lens });
	// Stopped after a window: the lens kept, and that /jlens learn resumes; no window finished: how to start it again.
	const cancelled = jobEntryLines({ ...running, outcome: "failed", error: "stopped", updated: 1000 }, look()).map(plain);
	assert.equal(cancelled[0], `✗ jlens learn ${SERVED} failed · stopped · 16m 40s`);
	assert.equal(cancelled.at(-1), `  /jlens use ${SERVED}-learned-20261014 · /jlens learn until HH:MM resumes the fit`);
	assert.equal(cancelled.length, 4);                                            // the running fit step is not shown
	assert.deepEqual(nextStep({ ...running, fit: null, outcome: "failed", error: "x" }), ["/jlens learn until HH:MM", " starts it again"]);
	// The job's own text (/jlens learn PATH) in the form, kept by trimJob.
	assert.deepEqual(nextStep(trimJob({ ...running, source: "/u/my notes", outcome: "failed", error: "x" })),
		[`/jlens use ${SERVED}-learned-20261014`, " · ", "/jlens learn /u/my notes until HH:MM", " resumes the fit"]);
	assert.equal(jobBriefLine({ job: "j", kind: "lens-learn", name: "m", outcome: "done", fit: { windows: 1, hit10: 0.5, logit_hit10: 0.4 } }),
		"jlens learn m: done, 1 window · hit@10 0.500 vs logit lens 0.400");
});

test("a job whose result has no calibration shows its outcome, and its line in /scope jobs", () => {
	const done = { kind: "other", job: "other-m-1", name: "m", origin: "pi", phase: "done", outcome: "done", steps: [],
		result: { path: "/x" } } as any as LearnState;
	assert.equal(jobOutcome(done), "done");
	assert.deepEqual(jobEntryLines(done, look()).map(plain), ["✓ other m · done"]);
	assert.deepEqual(jobEntryLines(trimJob(done), look()), jobEntryLines(done, look()));
	assert.equal(jobBriefLine({ job: "other-m-1", kind: "other", name: "m", outcome: "done" }), "other m: done");
});

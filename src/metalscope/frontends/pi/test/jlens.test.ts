import assert from "node:assert/strict";
import { test } from "node:test";
import { type LensLearnPlan, type LensStatus, LENS_SUBCOMMANDS, candidateLines, lensLearnConfirmText,
	lensStatusLines, lensUseText, parseLensArgs, resolveLensPath } from "../jlens.ts";

test("parseLensArgs: status, help alone; learn [PATH] until HH:MM | windows N [fresh]; use NAME; errors point at /jlens help", () => {
	assert.deepEqual(parseLensArgs(""), { cmd: "status" });
	for (const c of ["status", "help"] as const) assert.deepEqual(parseLensArgs(c), { cmd: c });
	assert.deepEqual(parseLensArgs("STATUS"), { cmd: "status" });
	assert.deepEqual(parseLensArgs("learn until 07:30"), { cmd: "learn", until: "07:30" });
	assert.deepEqual(parseLensArgs("learn until 7:05"), { cmd: "learn", until: "07:05" });
	assert.deepEqual(parseLensArgs("learn ~/my notes/pi until 23:59 fresh"), { cmd: "learn", path: "~/my notes/pi", until: "23:59", fresh: true });
	assert.deepEqual(parseLensArgs("learn book.txt windows 3"), { cmd: "learn", path: "book.txt", windows: 3 });
	assert.deepEqual(parseLensArgs("learn windows 2 until 06:00"), { cmd: "learn", windows: 2, until: "06:00" });
	const err = (e: string) => ({ cmd: "error", error: `${e} · see /jlens help` });
	assert.deepEqual(parseLensArgs("learn"), err("say until when: /jlens learn [PATH] until HH:MM (or windows N); the model is unloaded until then"));
	assert.deepEqual(parseLensArgs("learn ~/notes"), err("say until when: /jlens learn [PATH] until HH:MM (or windows N); the model is unloaded until then"));
	assert.deepEqual(parseLensArgs("learn until 7pm"), err('until takes a time, HH:MM (24-hour), not "7pm"'));
	assert.deepEqual(parseLensArgs("learn until 24:00"), err('until takes a time, HH:MM (24-hour), not "24:00"'));
	assert.deepEqual(parseLensArgs("learn windows 0"), err('windows takes a count from 1 to 10000, not "0"'));
	assert.deepEqual(parseLensArgs("learn until 07:30 ~/notes"), err('"~/notes" after the path: /jlens learn [PATH] until HH:MM (or windows N)'));
	assert.deepEqual(parseLensArgs("use m-learned-20261015"), { cmd: "use", name: "m-learned-20261015" });
	assert.deepEqual(parseLensArgs("use a b"), err("/jlens use takes one lens name (/jlens lists them)"));
	assert.deepEqual(parseLensArgs("use help"), { cmd: "help" });                       // help, not a lens named help
	assert.deepEqual(parseLensArgs("use ../x"), err('"../x" is not a lens name'));
	assert.deepEqual(parseLensArgs("status now"), err("/jlens status takes no arguments"));
	assert.deepEqual(parseLensArgs("fit"), err('unknown subcommand "fit"'));
	assert.deepEqual(parseLensArgs("baseline"), err('unknown subcommand "baseline"'));
	assert.deepEqual(LENS_SUBCOMMANDS, ["status", "on", "off", "use", "learn", "help"]);
	assert.equal(resolveLensPath("~/notes", "/work", "/Users/me"), "/Users/me/notes");
	assert.equal(resolveLensPath("docs", "/work", "/Users/me"), "/work/docs");
	assert.equal(resolveLensPath("/abs/book.txt", "/work", "/Users/me"), "/abs/book.txt");
});

const LENS: LensStatus = { model: "Qwen3.5-2B-4bit", installed: true, path: "/Users/me/.metalscope/models/Qwen3.5-2B/lenses/neuronpedia", lenses_dir: "/Users/me/.metalscope/models/Qwen3.5-2B/lenses",
	source: "neuronpedia/jacobian-lens qwen3.5-2b (Salesforce-wikitext, 1000 prompts)", hf_model: "Qwen/Qwen3.5-2B", n_prompts: 283, layers: 23 };

test("lensStatusLines: where the lens is and came from, what it powers", () => {
	assert.deepEqual(lensStatusLines(LENS), [
		"lens for Qwen3.5-2B-4bit: /Users/me/.metalscope/models/Qwen3.5-2B/lenses/neuronpedia",
		"  neuronpedia/jacobian-lens qwen3.5-2b (Salesforce-wikitext, 1000 prompts) · Qwen/Qwen3.5-2B · fitted on 283 prompts · 23 layers",
		"what it powers",
		"  its read-out: the dock's LENS and SETTLING, the settle layer; the LENS edit",
		"/jlens learn fits a new lens overnight.",
	]);
	// Only what lens.json has: no source line at all when it says nothing.
	const bare = lensStatusLines({ model: "m", installed: true, path: "/l/m", lenses_dir: "/l" });
	assert.deepEqual(bare.slice(0, 2), ["lens for m: /l/m", "what it powers"]);
	// Which checkpoint the served model is a copy of, and its other served names, are the command line's: not shown.
	const copy = { ...LENS, checkpoint: "Qwen3.5-2B", served: ["Qwen3.5-2B-4bit", "Qwen3.5-2B"] } as LensStatus;
	assert.deepEqual(lensStatusLines(copy), lensStatusLines(LENS));
	assert.doesNotMatch(lensStatusLines({ ...copy, installed: false }).join("\n"), /checkpoint Qwen3\.5-2B|served as/);
});

test("lensStatusLines without a lens: where one goes, what is missing, where published ones are", () => {
	assert.deepEqual(lensStatusLines({ model: "Llama-3-8B", installed: false, path: null, lenses_dir: "/Users/me/.metalscope/models/Llama-3-8B/lenses" }), [
		"no lens for Llama-3-8B (a lens goes in /Users/me/.metalscope/models/Llama-3-8B/lenses/<name>/)",
		"Without one, LENS reads the logit lens; the LENS edit has no word vectors.",
		"Published lenses exist for about 40 open models: neuronpedia/jacobian-lens on Hugging Face; scripts/convert_lens.py --checkpoint converts one into that folder.",
		"/jlens learn until 07:30 fits one overnight (the default corpus, or /jlens learn PATH until 07:30 for your own text); it asks first.",
	]);
});

const SERVED = "Qwen3.8-Flash-Next-oQ4e-mtp";
const TWO: LensStatus = { model: SERVED, installed: true, path: "/l/Qwen3.8-Flash-Next-4bit", lenses_dir: "/l",
	candidates: [
		{ name: `${SERVED}-learned-20261015`, in_use: false, why: "learned for it", learned_for: SERVED, fitted: "2026-10-15", n_prompts: 14,
			hit10: 0.231, logit_hit10: 0.148, compared: { lens: "Qwen3.8-Flash-Next-4bit", hit10: 0.207 } },
		{ name: "Qwen3.8-Flash-Next-4bit", in_use: true, why: "in use" },
		{ name: "Qwen3.8-Flash-Next-4bit-ropebug", in_use: false, why: "its lens" }] };

test("/jlens with two candidate lenses: ranked by held-out hit@10, the one in use marked, how to switch", () => {
	const lines = lensStatusLines(TWO);
	assert.deepEqual(candidateLines(TWO), [
		`lenses for ${SERVED}, best held-out hit@10 first:`,
		`  1. ${SERVED}-learned-20261015: learned for it · held-out hit@10 0.231 vs logit lens 0.148, Qwen3.8-Flash-Next-4bit 0.207 on the same windows · learned 2026-10-15, 14 windows`,
		"  2. Qwen3.8-Flash-Next-4bit: in use · not scored",
		"  3. Qwen3.8-Flash-Next-4bit-ropebug: its lens · not scored",
		"/jlens off turns the J-lens off (traces then read the logit lens only), /jlens on turns it back on, from the next token, no reload; /jlens use NAME picks one of these.",
	]);
	assert.ok(lines.join("\n").includes(candidateLines(TWO).join("\n")));
	assert.deepEqual(candidateLines({ ...TWO, candidates: [TWO.candidates![1]] }), []);       // only the lens in use: nothing to rank
	// No lens in use, a learned one waiting: listed under the "no lens" lines.
	const none = lensStatusLines({ model: SERVED, installed: false, path: null, lenses_dir: "/l", candidates: [TWO.candidates![0]] });
	assert.equal(none.at(-1), "/jlens off turns the J-lens off (traces then read the logit lens only), /jlens on turns it back on, from the next token, no reload; /jlens use NAME picks one of these.");
	// What /jlens use says names the served model only: which checkpoint it is a copy of is the command line's business.
	assert.equal(lensUseText({ model: SERVED, lens: `${SERVED}-learned-20261015`, was: "Qwen3.8-Flash-Next-4bit" }), [
		`${SERVED}-learned-20261015 is now the J-lens for ${SERVED} (was Qwen3.8-Flash-Next-4bit), read beside the logit lens in every trace.`,
		`It is read when tracing next starts on ${SERVED} (or it next loads).`,
	].join("\n"));
});

test("/jlens off and back on: the logit lens ranked with the others, in use when the model reads no J-lens; the switch is live", () => {
	const logit = { name: "logit", in_use: false, why: "J = I", hit10: 0.148, logit: true, scored_on: `${SERVED}-learned-20261015` };
	const three: LensStatus = { ...TWO, candidates: [TWO.candidates![0], logit, ...TWO.candidates!.slice(1)] };
	assert.equal(candidateLines(three)[2], `  2. logit: the logit lens · held-out hit@10 0.148 on ${SERVED}-learned-20261015's windows`);
	assert.deepEqual(parseLensArgs("off"), { cmd: "use", name: "logit" });               // the server's name for the logit lens
	assert.deepEqual(parseLensArgs("ON"), { cmd: "use", name: "jlens" });                // the server picks which J-lens
	assert.deepEqual(parseLensArgs("use logit"), { cmd: "use", name: "logit" });          // the same request as /jlens off
	assert.deepEqual(parseLensArgs("on x"), { cmd: "error", error: "/jlens on takes no arguments · see /jlens help" });
	for (const old of ["logit", "jlens"]) assert.deepEqual(parseLensArgs(old), { cmd: "error", error: `unknown subcommand "${old}" · see /jlens help` });
	const using: LensStatus = { model: SERVED, installed: false, path: null, lenses_dir: "/l",
		candidates: [TWO.candidates![0], { ...logit, in_use: true, why: "in use" }] };
	const lines = lensStatusLines(using);
	assert.equal(lines[0], `${SERVED} reads through the logit lens: no J-lens is in use`);
	assert.equal(lines[1], "LENS reads the logit lens; the LENS edit has no word vectors.");   // the layers read follow the lens step
	assert.ok(lines.includes(`  2. logit: in use · held-out hit@10 0.148 on ${SERVED}-learned-20261015's windows`));
	const base = { model: SERVED };
	assert.equal(lensUseText({ ...base, lens: "logit", was: "Qwen3.8-Flash-Next-4bit", live: "logit" }), [
		`${SERVED} now reads through the logit lens only (was Qwen3.8-Flash-Next-4bit): the J-lens is off.`,
		`${SERVED} reads it from its next token, no reload.`,
		"No J-lens, so no J-lens read-outs and no /steer word vectors; /jlens on turns it back on.",
	].join("\n"));
	const back = lensUseText({ ...base, lens: "Qwen3.8-Flash-Next-4bit", was: "logit", live: "Qwen3.8-Flash-Next-4bit" });
	assert.equal(back, [`Qwen3.8-Flash-Next-4bit is now the J-lens for ${SERVED} (was logit), read beside the logit lens in every trace.`, `${SERVED} reads it from its next token, no reload.`].join("\n"));
	const failed = lensUseText({ ...base, lens: "Qwen3.8-Flash-Next-4bit", was: "logit", live: "logit" });
	assert.ok(failed.includes(`Qwen3.8-Flash-Next-4bit did not load on the server, so ${SERVED} reads through logit: the server log says why.`));
});

const PLAN: LensLearnPlan = { model: SERVED, until: "07:30", windows: null,
	estimate: { d_model: 2560, layers: 48, min_per_window: 45, overhead_min: 14.4 }, windows_estimate: 9, source: null, kind: "default",
	resume: 0, lens: `${SERVED}-learned-20261014` };

test("lensLearnConfirmText: what is fitted until when, chat paused all that time, about how many windows, where the lens goes", () => {
	assert.equal(lensLearnConfirmText(PLAN), [
		`A GPU job loads its own copy of ${SERVED} and fits a Jacobian lens on the default corpus: wikitext paragraphs, as plain text and as chat turns a few thousand tokens deep, and metalscope's own source as code, one window at a time, until 07:30.`,
		`${SERVED} is unloaded the whole time: chat pauses until 07:30 (the reload included), replies in progress are aborted, and model loads are refused until it is back.`,
		"Rough guess until the first pass is timed: about 45 min a window (2560 dims, 48 layers), so about 9 windows by 07:30. The job stops if none fits. /scope jobs shows its pace.",
		`After every window the lens is written to lenses/${SERVED}-learned-20261014/ with its held-out hit@10, against the logit lens and the lens in use. The model reads it when it is back if it is its only lens; else /jlens use ${SERVED}-learned-20261014.`,
		"/scope jobs cancel stops it and keeps the windows done.",
	].join("\n"));
	assert.match(lensLearnConfirmText({ ...PLAN, estimate: { d_model: 5120, layers: 64, min_per_window: 227, overhead_min: 4.9 }, windows_estimate: 1 }),
		/about 3\.8 h a window \(5120 dims, 64 layers\), so about 1 window by 07:30\. The job stops if none fits\./);
	const txt = lensLearnConfirmText({ ...PLAN, kind: "txt", source: "/me/book.txt", until: null, windows: 3, windows_estimate: 3, resume: 5, estimate: null });
	assert.match(txt, /on the paragraphs of \/me\/book\.txt \(shallow windows only, .*\), one window at a time, for 3 windows\./);
	assert.match(txt, /chat pauses until the windows are done,/);
	assert.match(txt, /No estimate of how many windows fit/);
	assert.match(txt, /It resumes the earlier fit of .* on this corpus \(5 windows done\)\./);
	assert.match(lensLearnConfirmText({ ...PLAN, kind: "dir", source: "/me/notes" }), /the text files in \/me\/notes \(long files give windows up to 8k tokens deep\)/);
});


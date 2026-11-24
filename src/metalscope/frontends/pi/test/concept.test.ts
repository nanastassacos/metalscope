import assert from "node:assert/strict";
import { test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { JOB_MINUTES, calibrateTime, calibrationTag, chatRank, conceptListText, doseRow, fittingCalibration, jobMinutes, learnTime, notLearnedText, parseLibrary,
	overviewLines, parseConceptArgs, parseConcepts, parseLearnArgs, sayWhatText, steerListLine, steerListLines, steerListNotes } from "../concept.ts";
import { CAL, LOW, calConcept, chatConcept, listRow, terseJson } from "./fixtures/concepts.ts";
import { parseExamples, parseSteering } from "../steerpage.ts";
import { addSpans, timeSpan } from "../args.ts";

/** A plain theme; one whose colours are codes a test can find, and the text without them. */
const PLAIN = { fg: (_c: string, s: string) => s, bold: (s: string) => s };
const TH = { fg: (c: string, s: string) => `\x1b[3${{ success: 2, accent: 6, dim: 7, error: 1, warning: 3, muted: 5 }[c] ?? 9}m${s}\x1b[39m`, bold: (s: string) => `\x1b[1m${s}\x1b[22m` };
const strip = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");

test("timeSpan: minutes up to 90, hours past it, to the half hour", () => {
	assert.equal(timeSpan([5, 35]), "5 to 35 min");
	assert.equal(timeSpan([60, 90]), "60 to 90 min");
	assert.equal(timeSpan([20, 120]), "20 min to 2 h");
	assert.equal(timeSpan([180, 360]), "3 to 6 h");
	assert.equal(timeSpan(addSpans([60, 90], [180, 360]), "–"), "4–7.5 h");   // a ds4 learn and its calibration
	assert.equal(timeSpan([26, 26]), "26 min");                                 // one measured figure
	assert.equal(timeSpan([90, 90]), "90 min");
	assert.equal(timeSpan([180, 180]), "3 h");
});

test("job times: the served model's measured figure from the one table, the backend's range for any other model", () => {
	assert.deepEqual(jobMinutes("omlx", "Qwen3.6-35B-A3B-MLX-4bit"), { learn: [5, 5], calibrate: [20, 20], whole: [26, 26], measured: "Qwen3.6" });
	assert.equal(learnTime("omlx", "Qwen3.6-35B-A3B-MLX-4bit"), "26 min (measured on Qwen3.6)");
	assert.equal(calibrateTime("omlx", "Qwen3.6-35B-A3B"), "20 min (measured on Qwen3.6)");     // a checkpoint's name too
	assert.equal(learnTime("ds4", "DeepSeek-V4.1-Flash-ds4-q2", "–"), "4–7.5 h (measured on DeepSeek V4.1)");
	assert.equal(calibrateTime("ds4", "DeepSeek-V4.1-Flash-ds4-q2"), "3 to 6 h (measured on DeepSeek V4.1)");
	// Others: the range, and where.
	assert.equal(learnTime("omlx", "Qwen3.8-Flash-Next-oQ4e-mtp"), "25 min to 2.5 h on oMLX");
	assert.equal(learnTime("omlx"), "25 min to 2.5 h on oMLX");
	assert.equal(learnTime("ds4", "Qwen3.8-Flash-Next-ds4-q4k", "–"), "4–7.5 h on ds4");
	assert.equal(calibrateTime("omlx", "Qwen3.5-2B-4bit"), "20 min to 2 h on oMLX");
	// A measured row belongs to its backend: Qwen3.6 on ds4 is not the oMLX run.
	assert.equal(jobMinutes("ds4", "Qwen3.6-35B-A3B").measured, undefined);
	// Every backend has a range row (a model of its own never falls through).
	for (const b of ["omlx", "ds4"] as const) assert.ok(JOB_MINUTES.some((r) => r.backend === b && !r.model), b);
});

const HEAD = (w = 12) => `   ${"concept".padEnd(w)}  ${"good in chat".padEnd(14)}  dose`;

test("uncalibrated concepts: \"not calibrated\", an old concept.json status ignored", () => {
	const raw = (status: any) => ({ concepts: [{ name: "terse", title: "Terse", reader: { layer: 8, strength: 0.8 }, steer: null, status }] });
	const [k] = parseConcepts(raw({ decision: "drop", reason: "Presence fails the bar.", date: "2026-10-06" }));
	assert.equal(steerListLine(k), listRow("terse", "not calibrated", "-"));
	assert.ok(!("status" in k));                                                       // read by nothing
	assert.doesNotMatch(overviewLines(k, null, "omlx", 116, PLAIN).join("\n"), /label|keep|drop/);
	// No chat scores: a header, then one group in the order given, one aligned name column.
	const [s1] = parseConcepts({ concepts: [{ ...raw(null).concepts[0], name: "uncooperative" }] });
	assert.deepEqual(steerListLines([k, s1, ...parseConcepts({ concepts: [{ ...raw(null).concepts[0], name: "fluffy" }] })]), [
		HEAD(13),
		"no chat score",
		`   ${listRow("terse", "not calibrated", "-", 13)}`,
		`   ${listRow("uncooperative", "not calibrated", "-", 13)}`,
		`   ${listRow("fluffy", "not calibrated", "-", 13)}`,
	]);
});

test("a pulled concept's strings (title, description, who wrote its texts) reach Pi with their control characters as ·", () => {
	const osc = "\x1b]8;;http://x\x07";
	const [c] = parseConcepts({ concepts: [{ name: "calm", title: `${osc}Calm`, description: `Slow.${osc}`, texts: `texts: ${osc}`, error: osc }] });
	assert.deepEqual([c.title, c.description, c.texts, c.error], ["·]8;;http://x·Calm", "Slow.·]8;;http://x·", "texts: ·]8;;http://x·", "·]8;;http://x·"]);
	const [l] = parseLibrary({ library: [{ name: "calm", title: osc, description: osc, writer: osc, texts: osc }] });
	assert.deepEqual([l.title, l.description, l.writer, l.texts], Array(4).fill("·]8;;http://x·"));
	const [s] = parseSteering({ concepts: [{ name: "calm", title: `${osc}Calm`, description: osc }] }, "m").concepts;
	assert.deepEqual([s.title, s.description], ["·]8;;http://x·Calm", "·]8;;http://x·"]);
	const ex = parseExamples({ conversations: [{ title: osc, turns: [{ user: osc, plain: { text: osc }, steered: { text: osc } }] }] });
	assert.deepEqual([ex.turns[0].conv, ex.turns[0].user, ex.turns[0].plain.text, ex.turns[0].steered.text], Array(4).fill("·]8;;http://x·"));
});

test("the fluency edge: parsed, not on /concept list", () => {
	const parsed = parseConcepts({ concepts: [
		{ name: "fluffy", reader: { layer: 3, strength: 0.4 }, generation: { edge: 1.6 } },
		{ name: "nogen", reader: { layer: 3, strength: 0.4 } },
	] });
	assert.equal(parsed[0].generation!.edge, 1.6);
	assert.equal(parsed[1].generation, null);
	assert.equal(parsed[0].texts, null);                                           // who wrote its texts: none said
	assert.equal(parseConcepts({ concepts: [{ name: "fluffy", texts: "texts: hand-written by Claude Fable 5.1 (2026-10-05)" }] })[0].texts,
		"texts: hand-written by Claude Fable 5.1 (2026-10-05)");
	// /concept list shows chat findings, not the ladder's edge
	assert.equal(steerListLine(parsed[0]), listRow("fluffy", "not calibrated", "-"));
	assert.equal(steerListLine({ ...parsed[0], reader: null }), `${"fluffy".padEnd(12)}  (no steer layer)`);
});

test("parseConcepts: a chat calibration (version 4) is kept when well formed; else null", () => {
	const c = calConcept(CAL).calibration!;
	assert.deepEqual({ ...c, strengths: c.strengths.length }, { layer: 20, dose: 0.4, good: 16, turns: 18, strengths: 5 });
	assert.deepEqual(c.strengths[4], { dose: 0.8, good: 4, turns: 18, coherence: 1.6 });
	assert.deepEqual(calConcept({ ...CAL, strengths: [{ ...CAL.strengths[4], coherence: null, runaway: 12.3 }] }).calibration!.strengths,
		[{ dose: 0.8, good: 4, turns: 18, coherence: null, runaway: 12.3 }]);
	// Not well formed: none, an older version 3, a layer that isn't a number, no turns, no good count.
	for (const bad of [null, { ...CAL, version: 3 }, { ...CAL, layer: "20" }, { ...CAL, turns: 0 }, { ...CAL, good: undefined }])
		assert.equal(calConcept(bad).calibration, null, JSON.stringify(bad));
	const partial = { ...CAL, strengths: [CAL.strengths[0], { ...CAL.strengths[1], good: undefined }] };
	assert.equal(calConcept(partial).calibration!.strengths.length, 1);         // a dose missing a count is left out
});

test("fittingCalibration: kept only when made steering at the layer /steer steers now", () => {
	const c = calConcept(CAL);
	assert.equal(fittingCalibration(c), c.calibration);                                     // reader L20, made at L20
	assert.equal(fittingCalibration(calConcept(CAL, 22)), null);                            // the reader moved to L22
	assert.deepEqual(fittingCalibration(calConcept(CAL, 22), 20), calConcept(CAL, 22).calibration);   // steered at L20 it fits again
	assert.equal(fittingCalibration(calConcept(null)), null);
	assert.equal(fittingCalibration({ ...c, reader: null }), null);                          // no steer layer
});

test("steerListLine: good replies at the dose /steer uses and that dose; or why there is no score", () => {
	assert.equal(steerListLine(chatConcept()), listRow("fluffy", "16 of 18", "0.4"));
	assert.equal(steerListLine(chatConcept(LOW)), listRow("fluffy", " 8 of 18", "0.2"));
	// Not calibrated (or made at another steer layer): no default, so no dose.
	assert.equal(steerListLine(calConcept(null)), listRow("fluffy", "not calibrated", "-"));
	assert.equal(steerListLine(calConcept(CAL, 22)), listRow("fluffy", "re-run calibrate", "-"));
	assert.equal(steerListLine(parseConcepts(terseJson(null))[0]), listRow("terse", "not calibrated", "-"));
	assert.equal(parseConcepts(terseJson(null))[0].steer, null);
	// As tags (/concept NAME's out-of-date line).
	assert.equal(calibrationTag(chatConcept()), " · chat: 16/18 good @0.4");
	assert.equal(calibrationTag(chatConcept(LOW)), " · chat: 8/18 good @0.2");
	assert.equal(calibrationTag(calConcept(CAL, 22)), " · calibrated at L20, now L22: re-run /concept calibrate fluffy");
	assert.equal(calibrationTag(calConcept(null)), " · not calibrated for chat");
	assert.equal(calibrationTag({ ...calConcept(null), reader: null }), "");   // nothing to steer: no tag
});

test("steerListLines: concepts ranked by good replies (then the lower dose), the top five first, then those without one", () => {
	// The server's default is a fitting calibration's dose (chatConcept); none for g (made at L18) and h (not calibrated).
	const named = (name: string, cal: any) => ({ ...(cal?.layer === 20 ? chatConcept(cal) : calConcept(cal)), name });
	const scored = (good: number, dose = 0.4) => ({ ...CAL, good, default_dose: dose });
	const list = [named("a", scored(12)), named("b", scored(17)), named("c", scored(12, 0.3)), named("d", scored(16)), named("e", scored(15)),
		named("f", scored(14)), named("g", { ...CAL, layer: 18 }), named("h", null)];
	const row = (rank: string, name: string, good: string, dose = "0.4") => `${rank} ${listRow(name, good, dose)}`;
	assert.deepEqual(steerListLines(list), [
		HEAD(),
		"best in chat",
		row(" 1", "b", "17 of 18"), row(" 2", "d", "16 of 18"), row(" 3", "e", "15 of 18"),
		row(" 4", "f", "14 of 18"), row(" 5", "c", "12 of 18", "0.3"),       // c ties a on good replies, at a lower dose
		"the rest",
		row(" 6", "a", "12 of 18"),
		"no chat score",
		row("  ", "g", "re-run calibrate", "-"),
		row("  ", "h", "not calibrated", "-"),
	]);
	assert.deepEqual(steerListNotes(list), [
		"good in chat: of 18 replies in 3 graded conversations, at its best dose.",
		'dose "-": not calibrated; /steer NAME DOSE, or /concept calibrate NAME.',
		"/concept NAME for details · /steer NAME to use one.",
	]);
	assert.equal(steerListNotes(list.slice(0, 6)).length, 2);                 // every concept calibrated: no "-" note
	assert.deepEqual([chatRank(list, "c"), chatRank(list, "g")], [{ rank: 5, of: 6 }, null]);
});

test("overviewLines: /concept NAME in a few lines: its score and rank, what it is, whose texts, its doses in chat, what learn found, the commands", () => {
	const angry = { ...chatConcept(LOW), name: "angry", title: "Angry", description: "Feeling angry. Notes on the corpus follow." };
	const sad = { ...chatConcept(CAL), name: "sad" };
	const rank = chatRank([angry, sad], "angry");
	assert.deepEqual(overviewLines(angry, rank, "omlx", 116, PLAIN), [
		"angry · 8/18 good @0.2 · ranked 2 of 2",
		"  what        Feeling angry.",
		"  texts       none on this model",
		"  in chat     0.1 3/18 · 0.2 8/18 · 0.3 8/18 · 0.4 5/18",
		"  learn       L20 · +0.40 log-odds over control at dose 0.8",
		"  /steer angry uses dose 0.2 · /concept calibrate angry (about 20 min to 2 h)",
	]);
	// The selected backend's time; who wrote its texts; a concept without a rank.
	assert.equal(overviewLines(angry, rank, "ds4", 116, PLAIN).at(-1), "  /steer angry uses dose 0.2 · /concept calibrate angry (about 3 to 6 h)");
	assert.equal(overviewLines({ ...angry, texts: "texts: hand-written by X (2026-10-06)" }, rank, "omlx", 116, PLAIN)[2], "  texts       hand-written by X (2026-10-06)");
	assert.equal(overviewLines(sad, null, "omlx", 116, PLAIN)[0], "sad · 16/18 good @0.4");
	// The name bold, the rest of the head dim; labels in the normal colour, values dim, the default dose's not; commands
	// in the accent colour, their words dim.
	const lit = overviewLines(angry, rank, "omlx", 116, TH);
	assert.equal(lit[0], "\x1b[1mangry\x1b[22m\x1b[37m · 8/18 good @0.2 · ranked 2 of 2\x1b[39m");
	assert.equal(lit[1], "  what        \x1b[37mFeeling angry.\x1b[39m");
	assert.equal(lit[3], "  in chat     \x1b[37m0.1 3/18\x1b[39m\x1b[37m · \x1b[39m0.2 8/18\x1b[37m · \x1b[39m\x1b[37m0.3 8/18\x1b[39m\x1b[37m · \x1b[39m\x1b[37m0.4 5/18\x1b[39m");
	assert.equal(lit[5], "  \x1b[36m/steer angry\x1b[39m\x1b[35m uses dose 0.2 · \x1b[39m\x1b[36m/concept calibrate angry\x1b[39m\x1b[35m (about 20 min to 2 h)\x1b[39m");
	// Every case one line: not calibrated (the command says what calibrate does), out of date (its tag), no steer layer,
	// a profile that could not be read.
	assert.deepEqual(overviewLines(calConcept(null), null, "omlx", 140, PLAIN), [
		"fluffy · not calibrated",
		"  what        no description",
		"  texts       none on this model",
		"  learn       L20 · +0.40 log-odds over control at dose 0.8",
		"  /concept calibrate fluffy (about 20 min to 2 h) tests it in chat and sets the dose /steer fluffy uses",
	]);
	const old = overviewLines(calConcept(CAL, 22), null, "omlx", 116, PLAIN);
	assert.equal(old[0], "fluffy · out of date");
	assert.equal(old[3], "  in chat     calibrated at L20, now L22: re-run /concept calibrate fluffy");
	assert.equal(old.at(-1), "  /steer fluffy DOSE · /concept calibrate fluffy (about 20 min to 2 h)");       // no default: a dose is needed
	assert.deepEqual(overviewLines({ ...calConcept(null), reader: null }, null, "omlx", 116, PLAIN), [
		"fluffy · no steer layer", "  what        no description", "  texts       none on this model", "  metalscope concept show fluffy m has what was measured"]);
	assert.deepEqual(overviewLines({ ...calConcept(null), error: "bad json" }, null, "omlx", 116, PLAIN), ["fluffy · its profile could not be read (bad json)"]);
	// Narrow: each line is cut from its end, never wider than the width.
	assert.deepEqual(overviewLines(angry, rank, "omlx", 40, PLAIN).slice(0, 4), [
		"angry · 8/18 good @0.2 · ranked 2 of 2", "  what        Feeling angry.", "  texts       none on this model", "  in chat     0.1 3/18 · 0.2 8/18 · 0.3…"]);
	for (const c of [angry, calConcept(null), calConcept(CAL, 22), { ...calConcept(null), reader: null }])
		for (let width = 1; width <= 120; width++) {
			const ls = overviewLines(c, rank, "omlx", width, TH);
			assert.ok(ls.every((l) => visibleWidth(l) <= width), `${width}\n${ls.map(strip).join("\n")}`);
		}
});

test("overviewLines: text from the server on one line, made safe: newlines and tabs a space, an escape a ·", () => {
	const odd = { ...calConcept(null), name: "odd\tname", model: "m\nx", description: "Line one\twith a tab\nand \x1b[2J an escape. Then more.",
		texts: "texts: by\nsomeone\x1b]8;;x\x07" };
	const lines = overviewLines(odd, null, "omlx", 160, TH);
	assert.ok(lines.every((l) => !/[\n\t\x07]/.test(l) && !/\x1b(?!\[[0-9;]*m)/.test(l)), JSON.stringify(lines));
	assert.deepEqual(lines.map(strip).slice(0, 3), ["odd name · not calibrated", "  what        Line one with a tab and ·[2J an escape.", "  texts       by someone·]8;;x·"]);
	assert.equal(strip(overviewLines({ ...odd, reader: null }, null, "omlx", 160, TH).at(-1)!), "  metalscope concept show odd name m x has what was measured");
	assert.deepEqual(overviewLines({ ...odd, error: "bad\njson\x1b[2J" }, null, "omlx", 160, PLAIN), ["odd name · its profile could not be read (bad json·[2J)"]);
});

test("doseRow: every dose tried, ascending, good replies of how many; a runaway dose says so; the default dose not dim", () => {
	const cal = calConcept({ ...CAL, strengths: [CAL.strengths[3], { ...CAL.strengths[4], good: 0, coherence: null, runaway: 12.6 }, CAL.strengths[2],
		{ ...CAL.strengths[1], runaway: 7 }] }).calibration!;
	const row = doseRow(cal);
	assert.equal(row.map((p) => p.t).join(""), "0.3 11/18, ran 7x the plain length · 0.4 16/18 · 0.6 15/18 · 0.8 ran 13x the plain length, not graded");
	assert.deepEqual(row.filter((p) => !p.c).map((p) => p.t), ["0.4 16/18"]);
});

/** Four concepts of the library as the server sends them (web.py _library), two writers and a definition alone. */
const LIBRARY = { library: [
	{ name: "angry", title: "Angry", description: "Feeling angry: hostility at people and events: glaring, slamming, snapping, the jaw set hard.", writer: "Claude Fable 5.1 (by hand)", texts: "texts: hand-written by Claude Fable 5.1 (2026-10-06)" },
	{ name: "joyful", title: "Joyful", description: "Feeling joyful: delight.", writer: "Claude Fable 5.1 (by hand)", texts: "texts: hand-written by Claude Fable 5.1 (2026-10-06)" },
	{ name: "witty", title: "Witty", description: "Dry humour.", writer: "Qwen3.6-35B-A3B-MLX-4bit", texts: 'texts: written by Qwen3.6-35B-A3B-MLX-4bit from "dry humour" (2026-10-27)' },
	{ name: "half", title: "half", description: "Half a thing.", writer: null, texts: "texts: each model writes its own" },
	{ title: "no name" }] };

test("conceptListText: the learned table under a head line, its notes, the library ready to learn, then what is on", () => {
	const list = { model: "m", steer_enabled: true, concepts: [chatConcept()] };
	const text = conceptListText(list, "steer fluffy@0.4 L20").split("\n");
	assert.deepEqual(text.slice(0, 3), ["Learned on m,", "ranked by good replies in conversation at the dose /steer uses:", steerListLines(list.concepts)[0]]);
	assert.equal(text.at(-1), "On now: steer fluffy@0.4 L20.");
	assert.equal(conceptListText(list, undefined).split("\n").at(-1), "On now: nothing.");
	assert.equal(conceptListText({ ...list, concepts: [] }, undefined),
		"Nothing learned on m yet: /concept learn NAME <what it means> makes a concept.\nOn now: nothing.");
	// The library under the learned ones: a line per writer, a row per concept, how to learn one and how long it takes.
	const library = parseLibrary(LIBRARY);
	assert.deepEqual(library.map((c) => c.name), ["angry", "joyful", "witty", "half"]);       // one without a name is dropped
	assert.deepEqual([library[3].writer, library[3].texts], [null, "texts: each model writes its own"]);   // a definition alone
	const withLib = conceptListText({ ...list, library }, undefined).split("\n");
	const at = withLib.indexOf("Ready to learn: 4 in the library");
	assert.ok(at > 3 && withLib.slice(0, at).includes(steerListLines(list.concepts)[1]), "the learned table first, as it was");
	assert.deepEqual(withLib.slice(at + 1, at + 8), ["texts by Claude Fable 5.1 (by hand)",
		"  angry         Feeling angry: hostility at people and events: glaring, …",
		"  joyful        Feeling joyful: delight.",
		"texts by Qwen3.6-35B-A3B-MLX-4bit",
		"  witty         Dry humour.",
		"texts: each model writes its own",
		"  half          Half a thing."]);
	assert.deepEqual(withLib.slice(at + 8), ["Learn one, no description needed: /concept learn joyful",
		"Each is learned then calibrated in chat: about 25 min to 2.5 h on oMLX.", "On now: nothing."]);
	assert.match(conceptListText({ ...list, library }, undefined, "ds4"), /\nEach is learned then calibrated in chat: about 4 to 7\.5 h on ds4\.\n/);
	assert.ok(withLib.every((l) => l.length <= 80), "every line fits 80 columns");
});

test("/concept on a fresh model leads with the library, ready to learn", () => {
	const fresh = conceptListText({ model: "Qwen3.6-35B-A3B-MLX-4bit", steer_enabled: false, concepts: [], library: parseLibrary(LIBRARY) }, undefined);
	if (process.env.SHOW_REPLY) console.log(fresh);
	assert.deepEqual(fresh.split("\n").slice(0, 3), ["Nothing learned on Qwen3.6-35B-A3B-MLX-4bit yet.", "Ready to learn: 4 in the library", "texts by Claude Fable 5.1 (by hand)"]);
	assert.match(fresh, /\nLearn one, no description needed: \/concept learn joyful\n/);
	assert.doesNotMatch(fresh, /<what it means>|profile/);
});

test("notLearnedText: a library concept says how to learn it and how long; another name, what is learned here", () => {
	const list = { model: "m", steer_enabled: true, concepts: [chatConcept()], library: parseLibrary(LIBRARY) };
	assert.equal(notLearnedText("joyful", list), "joyful is not learned on m yet: /concept learn joyful, about 25 min to 2.5 h on oMLX");
	assert.equal(notLearnedText("joyful", list, "ds4"), "joyful is not learned on m yet: /concept learn joyful, about 4 to 7.5 h on ds4");
	// A model with a measured time: that figure, not the backend's range.
	assert.equal(notLearnedText("joyful", { ...list, model: "DeepSeek-V4.1-Flash-ds4-q2" }, "ds4"),
		"joyful is not learned on DeepSeek-V4.1-Flash-ds4-q2 yet: /concept learn joyful, about 4 to 7.5 h (measured on DeepSeek V4.1)");
	assert.equal(notLearnedText("nope", list), "nope is not learned on m and is not in the library (learned here: fluffy) · see /concept help");
	assert.equal(notLearnedText("nope", { ...list, concepts: [] }), "nope is not learned on m and is not in the library (learned here: none) · see /concept help");
});

test("parseLearnArgs: a name and a description", () => {
	assert.deepEqual(parseLearnArgs('crunchy "crisp, brittle textures"'), { kind: "start", name: "crunchy", description: "crisp, brittle textures" });
	assert.deepEqual(parseLearnArgs("crunchy crisp textures that snap"), { kind: "start", name: "crunchy", description: "crisp textures that snap" });
	assert.deepEqual(parseLearnArgs(""), { kind: "help" });
	assert.equal(parseLearnArgs(`${"a".repeat(96)} x`).kind, "start");
	assert.deepEqual(parseLearnArgs(`${"a".repeat(97)} x`),
		{ kind: "error", message: `"${"a".repeat(97)}" is not a usable name: letters, digits, _ . - only, at most 96 characters` });
	// No description: a library concept, learned from its texts or its saved description (the server says which).
	assert.deepEqual(parseLearnArgs("fluffy"), { kind: "start", name: "fluffy", description: "" });
	assert.match(sayWhatText("crunchy"), /^say what crunchy means: \/concept learn crunchy <one line, .*> \(no description is saved for it\)$/);
	assert.equal(parseLearnArgs("../x something").kind, "error");
	// A name that reads as a number is refused: /steer would take it as a dose, so it could never be steered.
	for (const n of ["1984", "1e3", "0.5"])
		assert.deepEqual(parseLearnArgs(`${n} a year`), { kind: "error", message: `"${n}" reads as a number, and /steer takes a number as a dose, so it cannot name a concept: pick a name with a letter in it` }, n);
	assert.equal(parseLearnArgs("r2d2 robot voice").kind, "start");             // digits beside letters are fine
});

test("parseLearnArgs: --corpus DIR after the rest; --pass is gone", () => {
	assert.deepEqual(parseLearnArgs("kept --corpus ~/my texts/kept"), { kind: "start", name: "kept", description: "", corpus: "~/my texts/kept" });
	for (const [args, why] of [["kept --corpus", "needs a value"], ["kept --pass 9", "unknown option"], ["kept --dose 3", "unknown option"]])
		assert.match((parseLearnArgs(args) as { message: string }).message, new RegExp(why), args);
	assert.match((parseConceptArgs("calibrate anime-nerd --pass 12") as { error: string }).error, /unknown option "--pass" \(this takes none\)/);
});

test("parseConceptArgs: the library, its jobs, and errors that point at /concept help", () => {
	assert.deepEqual(parseConceptArgs(""), { cmd: "list" });
	for (const c of ["list", "help"] as const) assert.deepEqual(parseConceptArgs(c), { cmd: c });
	// The job words are /scope's: here they are names, never a subcommand.
	assert.deepEqual(parseConceptArgs("jobs"), { cmd: "overview", name: "jobs" });
	assert.deepEqual(parseConceptArgs("show"), { cmd: "overview", name: "show" });       // not a subcommand: a name, refused by learn
	assert.match((parseConceptArgs("show fluffy") as any).error, /unknown subcommand "show"/);
	assert.deepEqual(parseConceptArgs("example"), { cmd: "overview", name: "example" });     // no subcommand now: a name
	assert.match((parseConceptArgs("example scaly") as any).error, /unknown subcommand "example"/);
	assert.deepEqual(parseConceptArgs('learn otaku "dramatic anime speech"'), { cmd: "learn", name: "otaku", description: "dramatic anime speech" });
	assert.deepEqual(parseConceptArgs("calibrate anime-nerd"), { cmd: "calibrate", name: "anime-nerd" });
	// help: alone, for one subcommand either way round, or a subcommand missing its name.
	assert.deepEqual(parseConceptArgs("help"), { cmd: "help" });
	for (const a of ["calibrate help", "help calibrate", "calibrate", "CALIBRATE Help"]) assert.deepEqual(parseConceptArgs(a), { cmd: "help", topic: "calibrate" }, a);
	assert.deepEqual(parseConceptArgs("learn"), { cmd: "help", topic: "learn" });
	assert.deepEqual(parseConceptArgs("learn fluffy"), { cmd: "learn", name: "fluffy", description: "" });     // a library concept
	assert.deepEqual(parseConceptArgs("learn kind speech that will help you"), { cmd: "learn", name: "kind", description: "speech that will help you" });
	assert.deepEqual(parseConceptArgs("angry"), { cmd: "overview", name: "angry" });      // a bare name: its overview
	for (const [bad, why] of [["adopt x", /unknown subcommand "adopt"/], ["list all", /\/concept list takes no arguments/], ["help adopt", /\/concept help takes one subcommand/],
		["calibrate help now", /help goes alone: \/concept calibrate help/],
		["calibrate a/b", /not a concept name/], ["calibrate list", /"list" is a subcommand, not a concept: \/concept list/],
		["calibrate jobs", /"jobs" is a command word, not a concept · see/], ["calibrate list", /"list" is a subcommand/], ["learn list", /"list" is a command word/],
		["calibrate anime-nerd 0.4", /\/concept calibrate takes one concept \(it searches the doses itself\)/]] as [string, RegExp][]) {
		const r = parseConceptArgs(bad);
		assert.equal(r.cmd, "error", bad);
		if (r.cmd !== "error") continue;
		assert.match(r.error, why, bad);
		assert.match(r.error, /see \/concept( \w+)? help$/, bad);
	}
});

test("command words cannot name a concept; descriptions keep inner quotes; subcommands in any case", () => {
	for (const n of ["list", "status", "help", "show", "off", "none", "jobs", "cancel", "replay"]) {
		const a = parseLearnArgs(`${n} something it means`);
		assert.equal(a.kind, "error", n);
		if (a.kind === "error") assert.match(a.message, new RegExp(`^"${n}" is a command word, so it cannot name a concept`), n);
	}
	for (const n of ["hold", "meter", "dose", "until"]) assert.equal(parseLearnArgs(`${n} something it means`).kind, "start", n);   // not /steer words
	// /concept learn status: a reserved name.
	const r = parseConceptArgs("learn status means x");
	assert.equal(r.cmd, "error");
	if (r.cmd === "error") assert.match(r.error, /^"status" is a command word/);
	assert.deepEqual(parseLearnArgs('x "a" "b"'), { kind: "start", name: "x", description: '"a" "b"' });
	assert.deepEqual(parseLearnArgs('x "dramatic, high intensity"'), { kind: "start", name: "x", description: "dramatic, high intensity" });
	assert.deepEqual(parseConceptArgs("LIST"), { cmd: "list" });
	assert.deepEqual(parseConceptArgs('calibrate "fluffy"'), { cmd: "calibrate", name: "fluffy" });
	assert.deepEqual(parseConceptArgs('"fluffy"'), { cmd: "overview", name: "fluffy" });
	const u = parseConceptArgs("fluffy now");
	assert.equal(u.cmd, "error");
	if (u.cmd === "error") assert.equal(u.error, 'unknown subcommand "fluffy" · see /concept help');
});

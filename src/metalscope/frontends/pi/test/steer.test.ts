import assert from "node:assert/strict";
import { test } from "node:test";
import { parseConcepts } from "../concept.ts";
import { type SteerPart, parseSteerArgs, steerCompletions, steerExpect, steerFromProfile, steerFromWordOp, steerHeader,
	steerLabel, steerReplyText, steerState, steerStatusText, steerTag, steeringText } from "../steer.ts";
import { CAL, LOW, calConcept, chatConcept, info, terseJson } from "./fixtures/concepts.ts";

const errOf = (a: ReturnType<typeof parseSteerArgs>) => ("error" in a ? a.error : undefined);

test("parseSteerArgs: forms and errors", () => {
	assert.deepEqual(parseSteerArgs(""), { cmd: "status" });
	assert.deepEqual(parseSteerArgs("off"), { cmd: "off" });
	assert.deepEqual(parseSteerArgs("again"), { cmd: "again" });
	assert.match(errOf(parseSteerArgs("again fluffy"))!, /again goes alone/);
	assert.match(errOf(parseSteerArgs("fluffy + again"))!, /again goes alone/);
	assert.deepEqual(parseSteerArgs("fluffy"), { cmd: "set", parts: [{ name: "fluffy" }] });
	assert.deepEqual(parseSteerArgs("fluffy 1.5"), { cmd: "set", parts: [{ name: "fluffy", dose: 1.5 }] });
	assert.deepEqual(parseSteerArgs("fluffy@0.4"), { cmd: "set", parts: [{ name: "fluffy", dose: 0.4 }] });
	assert.deepEqual(parseSteerArgs("fluffy@0.001"), { cmd: "set", parts: [{ name: "fluffy", dose: 0.001 }] });
	assert.match(errOf(parseSteerArgs("fluffy 9"))!, /dose/);
	assert.match(errOf(parseSteerArgs("flu ffy"))!, /dose/);             // two words: the second is read as a dose (or a mode)
	assert.match(errOf(parseSteerArgs("fl/uffy"))!, /not a concept name/);
	// A number where a concept goes is a dose with no concept: refused, never a word vector named "0.5".
	for (const [bad, head] of [["0.5", "0.5"], ["fluffy +0.5", "0.5"], ["fluffy + 0.5", "0.5"], ["0.5@0.4", "0.5"], ["fluffy 0.4 + -1", "-1"]])
		assert.match(errOf(parseSteerArgs(bad))!, new RegExp(`^${head.replace(".", "\\.")} is a dose, not a concept: `), bad);
	assert.equal(errOf(parseSteerArgs("fluffy +0.5")), "0.5 is a dose, not a concept: a dose follows its concept, /steer fluffy 0.5 or /steer fluffy@0.5 · see /steer help");
	assert.deepEqual(parseSteerArgs("r2d2 0.5"), { cmd: "set", parts: [{ name: "r2d2", dose: 0.5 }] });   // digits in a name are fine
	// A dose below 0.001 changes nothing: refused, with a pointer to the help.
	for (const tiny of ["fluffy@0.0001", "fluffy 0.0005", "a@0.4+b@0.0009"]) assert.match(errOf(parseSteerArgs(tiny))!, /below 0\.001: too small.*see \/steer help$/, tiny);
	assert.match(errOf(parseSteerArgs("fluffy@0"))!, /dose must be a number from 0\.001 to 4/);
	assert.deepEqual(parseSteerArgs("status"), { cmd: "status" });
	assert.deepEqual(parseSteerArgs("list"), { cmd: "set", error: "/steer list is not a form: the concepts are /concept list · see /steer help" });
	assert.deepEqual(parseSteerArgs("help"), { cmd: "help" });
	for (const bad of ["fluffy 9", "fl/uffy", "fluffy 0.5 extra"]) assert.match(errOf(parseSteerArgs(bad))!, /see \/steer help$/);
});

test("parseSteerArgs: combinations and per-concept dose; misplaced words are errors", () => {
	assert.deepEqual(parseSteerArgs("angry+robotic"), { cmd: "set", parts: [{ name: "angry" }, { name: "robotic" }] });
	assert.deepEqual(parseSteerArgs("angry@0.4+robotic@0.8"), { cmd: "set", parts: [{ name: "angry", dose: 0.4 }, { name: "robotic", dose: 0.8 }] });
	assert.deepEqual(parseSteerArgs("angry + robotic@0.8"), { cmd: "set", parts: [{ name: "angry" }, { name: "robotic", dose: 0.8 }] });
	assert.deepEqual(parseSteerArgs("angry 0.4 + robotic dose:0.8"), { cmd: "set", parts: [{ name: "angry", dose: 0.4 }, { name: "robotic", dose: 0.8 }] });
	// The server's name rule: letters, digits, _ . -, not starting with . or -.
	assert.deepEqual(parseSteerArgs("doubt-48"), { cmd: "set", parts: [{ name: "doubt-48" }] });
	assert.deepEqual(parseSteerArgs("ice-cream"), { cmd: "set", parts: [{ name: "ice-cream" }] });   // a word the server may cut (commands.test.ts)
	// Two words, quoted or not, are no word steer: the second is read as a dose.
	for (const bad of ['"ice cream"', "ice cream", "“ice cream”"]) assert.equal(errOf(parseSteerArgs(bad)), `"cream" is not a dose · see /steer help`, bad);
	assert.deepEqual(parseSteerArgs("_x.y"), { cmd: "set", parts: [{ name: "_x.y" }] });
	for (const bad of [".hidden", "-.x", "-"]) assert.match(errOf(parseSteerArgs(bad))!, /not a concept name/, bad);
	// A word after a concept that is not a dose is an error, whatever it is: steering is constant, the dose its only setting.
	for (const [bad, word] of [["fluffy hold:0.6", "hold:0.6"], ["fluffy@0.4 hold", "hold"], ["fluffy meter:9", "meter:9"], ["robotic + fluffy fade", "fade"],
		["fluffy constant", "constant"], ["hold fluffy", "fluffy"]] as [string, string][])
		assert.equal(errOf(parseSteerArgs(bad)), `"${word}" is not a dose · see /steer help`, bad);
	// /steer's own words go alone.
	for (const [bad, why] of [["off junk", /off goes alone: \/steer off/], ["list all", /\/steer list is not a form: the concepts are \/concept list/], ["fluffy + off", /off goes alone, not as a concept/],
		["fluffy help", /help goes alone/]] as [string, RegExp][])
		assert.match(errOf(parseSteerArgs(bad))!, why, bad);
	for (const [bad, why] of [
		["angry+", /empty concept/], ["angry+angry", /appears twice/], ["angry pulse", /^"pulse" is not a dose · /],
		["angry@0.4 0.5", /dose is given once/], ["angry 0.4 dose:0.5", /dose is given once/], ["a+b+c+d+e+f+g+h+i", /at most 8/],
		["fluffy dose:9", /dose must be a number from 0\.001 to 4/], ["fluffy dose:0.0001", /below 0\.001/], ["fl/uffy hold:1", /not a concept name/],
	] as [string, RegExp][]) {
		const e = errOf(parseSteerArgs(bad));
		assert.ok(e, bad);
		assert.match(e!, why, bad);
		assert.match(e!, /see \/steer help$/, bad);
	}
});

test("steerHeader and labels: constant parts, the random control; what /steer prints", () => {
	const part = (name: string, layer: number, strength: number): SteerPart => ({ name, model: "m", layer, strength, source: "profile" });
	const one = steerState([part("angry", 24, 0.4)], "m");
	assert.equal(steerHeader(one), "angry@0.4/24");
	assert.equal(steerHeader(one, true), "random~angry@0.4/24");
	assert.equal(steerLabel(one), "angry@0.4 L24");
	assert.equal(steerTag(one), "steer angry@0.4 L24");
	const two = steerState([part("angry", 24, 0.4), part("robotic", 20, 0.8)], "m");
	assert.equal(steerHeader(two), "angry@0.4/24+robotic@0.8/20");
	assert.equal(steerHeader(two, true), "random~angry@0.4/24+random~robotic@0.8/20");
	assert.equal(steerLabel(two), "angry@0.4 L24 + robotic@0.8 L20");
	// Three parts in the order typed; never a per-part setting.
	const three = steerState([part("robotic", 24, 0.8), part("angry", 24, 0.4), part("fluffy", 20, 0.8)], "m");
	assert.equal(steerHeader(three), "robotic@0.8/24+angry@0.4/24+fluffy@0.8/20");
	assert.equal(steerHeader(three, true), "random~robotic@0.8/24+random~angry@0.4/24+random~fluffy@0.8/20");
	assert.doesNotMatch(steerHeader(three), /:meter=/);
	assert.equal(steerLabel(three), "robotic@0.8 L24 + angry@0.4 L24 + fluffy@0.8 L20");
	// What /steer prints: one part's lines as they are; several, each under its name (no pacing line: the thermostat it
	// described was cut).
	const withExpect = steerState([{ ...part("a", 3, 0.2), expect: "A in chat." }, { ...part("b", 3, 0.2), expect: "B in chat.\nReplies broke at 1.6." }], "m");
	assert.deepEqual(steerExpect(withExpect).split("\n"), ["a", "  A in chat.", "b", "  B in chat.", "  Replies broke at 1.6."]);
	assert.equal(steerExpect(steerState([{ ...part("a", 3, 0.2), expect: "A in chat.\nmore" }], "m")), "A in chat.\nmore");
	assert.equal(steerExpect(one), "");                                     // no expectations: nothing
	// The reply: only what the line above the editor does not say (the dose's facts, a warning); nothing to say, "".
	assert.equal(steerReplyText(one), "");
	assert.equal(steerReplyText(steerState(two.parts, "m", "few traces")), "warning: few traces");   // the norms note, kept on the state
	assert.equal(steeringText(two), "steering angry@0.4 L24 + robotic@0.8 L20");
	assert.equal(steeringText(steerState(two.parts, "m", "few traces"), true), "steering angry@0.4 L24 + robotic@0.8 L20 · warning: few traces");
	assert.equal(steeringText(two, true), steeringText(two));
	assert.equal(steerReplyText(withExpect), steerExpect(withExpect));
	// /steer status: the compact label, the header, what it does, where it acts, how to see it, where the details are.
	const tail = "Every token, thinking included. /steer off stops it.\nAfter a reply, /steer replay shows it beside the same message answered unsteered.";
	assert.equal(steerStatusText(undefined), "steer: off");
	assert.equal(steerStatusText(one), `steer: angry@0.4 L24\nheader: angry@0.4/24\n${tail}\n/concept angry has the details.`);
	assert.equal(steerStatusText(withExpect), `steer: a@0.2 L3 + b@0.2 L3\nheader: a@0.2/3+b@0.2/3\n${steerExpect(withExpect)}\n${tail}\n/concept NAME has each one's details.`);
});

test("the dock's LENS edit: a word removed or swapped, its forms and header", () => {
	assert.deepEqual(parseSteerArgs("-soccer"), { cmd: "set", parts: [{ name: "soccer", op: "remove" }] });
	assert.deepEqual(parseSteerArgs("soccer>rugby"), { cmd: "set", parts: [{ name: "soccer", op: "remove" }, { name: "rugby", op: "swap" }] });
	assert.deepEqual(parseSteerArgs("soccer>rugby 0.5"), { cmd: "set", parts: [{ name: "soccer", op: "remove" }, { name: "rugby", op: "swap", dose: 0.5 }] });
	assert.deepEqual(parseSteerArgs("-soccer + calm@0.3"), { cmd: "set", parts: [{ name: "soccer", op: "remove" }, { name: "calm", dose: 0.3 }] });
	for (const [bad, why] of [["-soccer 0.3", /takes no dose/], ["-soccer@0.3", /takes no dose/], ["soccer>", /a swap is OLD>NEW/], [">rugby", /a swap is OLD>NEW/],
		["a>b>c", /a swap is OLD>NEW/], ["soccer>soccer", /soccer appears twice/], ["-soccer + soccer", /appears twice/], ["a>b+c>d+e>f+g>h+i", /at most 8/]] as [string, RegExp][])
		assert.match(errOf(parseSteerArgs(bad))!, why, bad);
	const made = { name: "soccer", band: [18, 44] as [number, number] };
	const off = steerFromWordOp({ name: "soccer", op: "remove" }, made, "m");
	const on = steerFromWordOp({ name: "rugby", op: "swap" }, { ...made, name: "rugby" }, "m");
	assert.deepEqual([off.layer, off.band, off.strength, on.layer, on.strength, on.band], [18, [18, 44], 0, 31, 0.7, undefined]);
	const swap = steerState([off, on], "m");
	assert.equal(steerHeader(swap), "remove~soccer/18-44+rugby@0.7/31");
	assert.equal(off.expect, undefined);
	assert.equal(steerFromWordOp({ name: "rugby", op: "swap", dose: 0.5 }, { ...made, name: "rugby" }, "m").strength, 0.5);
});

test("/steer replay: temp:T, a steer to answer under, errors that point at help", () => {
	assert.deepEqual(parseSteerArgs("replay"), { cmd: "replay" });
	assert.deepEqual(parseSteerArgs("replay temp:0"), { cmd: "replay", temperature: 0 });
	for (const bad of ["temp:3", "temp:x", "temp:"]) assert.match((parseSteerArgs(`replay ${bad}`) as any).error, /0 to 2.*see \/steer help$/, bad);
	assert.match((parseSteerArgs("replay temp:0 temp:1") as any).error, /^temp:T is given once/);
	assert.deepEqual(parseSteerArgs("replay angry@0.6 + robotic temp:0.5"), { cmd: "replay", parts: [{ name: "angry", dose: 0.6 }, { name: "robotic" }], temperature: 0.5 });
	assert.deepEqual(parseSteerArgs("replay temp:0 angry"), { cmd: "replay", parts: [{ name: "angry" }], temperature: 0 });
	assert.match((parseSteerArgs("replay off") as any).error, /^\/steer replay takes a steer to answer under/);
	assert.match((parseSteerArgs("replay angry hold:1") as any).error, /^"hold:1" is not a dose/);
});

test("steerFromProfile: the reader's layer and dose; uncalibrated, how to calibrate and where replies broke", () => {
	const s = steerFromProfile(info);
	assert.ok(!("error" in s));
	if ("error" in s) return;
	assert.equal(s.layer, 20);
	assert.equal(s.strength, 0.8);
	assert.equal(steerHeader(steerState([s], "m")), "fluffy@0.8/20");
	// No chat calibration here: how to get one, and the test's edge in plain words; none of learn's log-odds.
	assert.deepEqual(s.expect!.split("\n"), ["Not calibrated for chat yet: /concept calibrate fluffy finds its dose.",
		"Replies broke at dose 1.6 in /concept learn's test; 0.8 is below that."]);
	assert.doesNotMatch(s.expect!, /log-odds|chosen|L20/);
	const hot = steerFromProfile(info, 2);
	if (!("error" in hot)) assert.match(hot.expect!, /Replies broke at dose 1\.6 in \/concept learn's test: expect damage at 2\./);
	// No steer layer: it can't be steered here, and learning it again would be refused (its profile exists).
	const noReader = steerFromProfile({ ...info, reader: null });
	assert.deepEqual(noReader, { error: "fluffy can't be steered on m: /concept learn found no layer where it moves the model. /concept fluffy has what was measured; /concept lists the others." });
	assert.ok("error" in steerFromProfile({ ...info, error: "boom" }));
	// Server JSON with holes never reaches the arithmetic.
	const parsed = parseConcepts({ model: "m", concepts: [{ name: "x", reader: { layer: 3, strength: 0.4, concept: null }, steer: { layer: 3, strength: 0.4, basis: "chat" } }, { name: "bad", error: "KeyError" }, { name: "noreader", reader: { layer: null } }] });
	assert.equal(parsed.length, 3);
	assert.deepEqual(parsed[0].reader, { layer: 3, strength: 0.4, concept: null, control: null });
	const holed = parseConcepts({ concepts: [{ name: "z", reader: { layer: 3, strength: null } }] });
	assert.equal(holed[0].reader, null);                                     // a reader without its dose is not well formed
	const sparse = steerFromProfile(parsed[0]);
	assert.ok(!("error" in sparse) && !/NaN|log-odds/.test(sparse.expect!));
	assert.equal(parsed[1].error, "KeyError");
	assert.equal(parsed[2].reader, null);
});

test("steer default dose: parsed (nullable), used by /steer; not calibrated, a dose given steers", () => {
	const [t] = parseConcepts(terseJson({ layer: 8, strength: 0.4, basis: "chat" }));
	assert.deepEqual(t.steer, { layer: 8, strength: 0.4, basis: "chat" });
	for (const bad of [undefined, null, { layer: 8, strength: null, basis: "chat" }])
		assert.equal(parseConcepts(terseJson(bad))[0].steer, null);
	assert.deepEqual(parseConcepts(terseJson({ layer: 8, strength: 0.5, basis: "chat" }))[0].steer, { layer: 8, strength: 0.5, basis: "chat" });
	const s = steerFromProfile(t);
	assert.ok(!("error" in s));
	if ("error" in s) return;
	assert.equal(s.strength, 0.4);
	assert.equal(steerHeader(steerState([s], "m")), "terse@0.4/8");
	assert.match(s.expect!, /Replies broke at dose 0\.8 in \/concept learn's test; 0\.4 is below that\./);
	const asked = steerFromProfile(t, 0.8);
	assert.ok(!("error" in asked) && /expect damage at 0\.8\./.test(asked.expect!));
	// Not calibrated: no default. Without a dose, no steer and one line saying what to do; with one, the steer.
	const [plain] = parseConcepts(terseJson(null, null));
	assert.deepEqual(steerFromProfile(plain), { error: "terse isn't calibrated on this model: give a dose (/steer terse 0.4) or run /concept calibrate terse." });
	const ps = steerFromProfile(plain, 0.8);
	assert.ok(!("error" in ps) && ps.strength === 0.8 && ps.expect === "Not calibrated for chat yet: /concept calibrate terse finds its dose.");
});

test("/steer's reply: what the dose did in chat, N of 18, in /steer status; set, only what the line above the editor does not say", () => {
	const s = steerFromProfile(chatConcept());
	assert.ok(!("error" in s));
	if ("error" in s) return;
	// The figure (the line above the editor's) apart from the other facts, its own line with it.
	assert.deepEqual(s.chat, { good: 16, turns: 18, line: "In chat: 16 of 18 replies good (the most of the doses tried)" });
	assert.equal(s.expect, undefined);
	assert.equal(steerReplyText(steerState([s], "m")), "");                  // the default dose: nothing to say
	assert.equal(steerReplyText(steerState([s, { ...s, name: "calm" }], "m")), "");
	assert.equal(steerReplyText(steerState([s], "m", "few traces")), "warning: few traces");
	assert.match(steerStatusText(steerState([s], "m")), /\nIn chat: 16 of 18 replies good \(the most of the doses tried\)\n/);
	const held = steerFromProfile(chatConcept(LOW));
	assert.ok(!("error" in held) && held.chat?.line === "In chat: 8 of 18 replies good (the most of the doses tried)");
	// A dose of your own: what it did when it was tried, the default beside it; one never tried, said.
	const tried = steerFromProfile(chatConcept(), 0.6);
	assert.ok(!("error" in tried) && tried.expect === "Default dose 0.4: 16 of 18 good (the most)");
	assert.deepEqual("error" in tried ? 0 : tried.chat, { good: 15, turns: 18, line: "In chat at dose 0.6: 15 of 18 replies good" });
	assert.equal(steerReplyText(steerState([tried as SteerPart], "m")), "Default dose 0.4: 16 of 18 good (the most)");   // its figure is the line's
	assert.equal(steerStatusText(steerState([tried as SteerPart], "m")).split("\n").slice(2, 4).join("\n"),
		"In chat at dose 0.6: 15 of 18 replies good\nDefault dose 0.4: 16 of 18 good (the most)");                   // status: all of it
	const low = steerFromProfile(chatConcept(), 0.3);
	assert.ok(!("error" in low) && low.chat?.line === "In chat at dose 0.3: 11 of 18 replies good");
	// Doses that ran away: the default (graded) says so after its figure; one not graded says that.
	const ranCal = { ...CAL, strengths: CAL.strengths.map((s) => (s.dose === 0.4 ? { ...s, runaway: 7 } : s.dose === 0.8 ? { ...s, good: 0, coherence: null, runaway: 12 } : s)) };
	const ranDefault = steerFromProfile(chatConcept(ranCal));
	assert.ok(!("error" in ranDefault) && ranDefault.chat?.line === "In chat: 16 of 18 replies good (the most of the doses tried); ran 7x the plain length");
	const ranUngraded = steerFromProfile(chatConcept(ranCal), 0.8);
	assert.ok(!("error" in ranUngraded) && ranUngraded.chat?.line === "In chat at dose 0.8: 0 of 18 replies good; ran 12x the plain length, not graded");
	const untried = steerFromProfile(chatConcept(), 0.5);
	assert.ok(!("error" in untried) && untried.expect === "Dose 0.5 was not tried in chat.\nDefault dose 0.4: 16 of 18 good (the most)");
	assert.ok(!("error" in untried) && untried.chat === undefined);
	assert.equal(steerReplyText(steerState([untried as SteerPart], "m")), (untried as SteerPart).expect);
	if (process.env.SHOW_REPLY) for (const x of [s, untried]) console.log(steerReplyText(steerState([x as SteerPart], "m")) || "(nothing)");
	assert.ok(![s, tried, untried].some((x) => "error" in x || /Replies broke|log-odds/.test(x.expect ?? "")));
	// Not calibrated: a dose given steers, with the test's edge in place of a chat finding.
	assert.ok("error" in steerFromProfile(calConcept(null)));
	const uncal = steerFromProfile(calConcept(null), 0.8);
	assert.ok(!("error" in uncal) && uncal.expect === "Not calibrated for chat yet: /concept calibrate fluffy finds its dose.\nReplies broke at dose 1.6 in /concept learn's test; 0.8 is below that.");
	assert.ok(!("error" in uncal) && uncal.chat === undefined);
	assert.equal(steerReplyText(steerState([uncal as SteerPart], "m")), (uncal as SteerPart).expect);
	if (process.env.SHOW_REPLY) console.log(steerReplyText(steerState([uncal as SteerPart], "m")));
	// Calibrated at another layer: out of date.
	const old = steerFromProfile(calConcept(CAL, 18), 0.4);
	assert.ok(!("error" in old) && /^Its chat calibration is out of date \(made at L20\): \/concept calibrate fluffy runs it again\./.test(old.expect!));
});

test("/steer: words that go alone, words that are not forms, quotes", () => {
	assert.equal(errOf(parseSteerArgs("none")), "/steer none is not a form: /steer off · see /steer help");
	assert.match(errOf(parseSteerArgs("fluffy list"))!, /^\/steer list is not a form: the concepts are \/concept list/);
	assert.match(errOf(parseSteerArgs("fluffy + list"))!, /^\/steer list is not a form: the concepts are \/concept list/);
	assert.match(errOf(parseSteerArgs("fluffy + none"))!, /^\/steer none is not a form: \/steer off/);
	assert.deepEqual(parseSteerArgs('"fluffy" 0.4'), { cmd: "set", parts: [{ name: "fluffy", dose: 0.4 }] });
});

test("steerCompletions: a concept or a word of /steer's; after +, another concept; after a concept, nothing", () => {
	const names = ["angry", "fluffy"];
	const values = (prefix: string) => steerCompletions(prefix, names).map((c) => c.value);
	assert.deepEqual(values("ang"), ["angry"]);
	assert.deepEqual(values("rep"), ["replay"]);
	assert.deepEqual(values("ag"), ["again"]);
	assert.deepEqual(values("replay t"), ["replay temp:0"]);
	assert.deepEqual(values("replay ang"), ["replay angry"]);
	for (const after of ["angry ", "angry 0.4 ", "off ", "angry + fluffy@0.4 h"]) assert.deepEqual(values(after), [], after);
	assert.deepEqual(values("angry + fl"), ["angry + fluffy"]);
	assert.deepEqual(values("angry@0.4 + fl"), ["angry@0.4 + fluffy"]);
	assert.deepEqual(values("angry + an"), []);                                           // angry is already typed
});

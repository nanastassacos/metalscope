import assert from "node:assert/strict";
import { test } from "node:test";
import { registerCommands } from "../commands.ts";
import { parseConcepts } from "../concept.ts";
import { steerHeader } from "../steer.ts";

/** The jobs, recorded instead of run (job-runner.ts runs them). */
function fakeJobs() {
	const calls: any[][] = [];
	const rec = (name: string) => async (_ctx: any, ...a: any[]) => void calls.push([name, ...a]);
	return { calls, jobs: { learn: rec("learn"), calibrate: rec("calibrate"), lensLearn: rec("lensLearn"),
		jobs: rec("jobs"), cancel: rec("cancel") } };
}

/** A fake Pi recording commands and transcript entries; a ctx recording notes; a stub UI. */
function setup() {
	const commands = new Map<string, any>();
	const entries: [string, any][] = [];
	const notes: [string, string][] = [];
	const pi = {
		registerCommand: (name: string, cmd: any) => commands.set(name, cmd),
		appendEntry: (type: string, data: any) => entries.push([type, data]),
		on: () => {},
	} as any;
	const live = { base: "http://127.0.0.1:9", clientId: "pi-test", level: "on", connected: true } as any;
	const modes: string[] = [];                                                  // /scope dock mini | full, recorded
	const ui = { updateUi() {}, showDock() {}, hideDock() {}, docked: () => false, steerStatus: () => "",
		dockMode: (m: string) => (modes.push(m), m === "full" && modes.length > 2 ? "the full dock needs 23 rows; this terminal has 20: the mini dock stays" : undefined) } as any;
	const api = async () => {
		throw new Error("offline");
	};
	const fj = fakeJobs();
	const replays: any[] = [];                                                  // /steer replay's runs, recorded (replay.ts runs them)
	registerCommands({ pi, live, api } as any, ui, fj.jobs, async (_ctx: any, o: any) => void replays.push(o));
	const ctx = { model: { provider: "metalscope-omlx", id: "m" }, hasUI: false, ui: { notify: (m: string, kind: string) => notes.push([m, kind]) } };
	return { commands, entries, notes, live, ctx, calls: fj.calls, replays, modes };
}

test("/scope dock mini | full: the dock's mode, said; anything else refused", async () => {
	const { commands, notes, ctx, modes } = setup();
	await commands.get("scope").handler("dock mini", ctx);
	await commands.get("scope").handler("dock full", ctx);
	await commands.get("scope").handler("dock small", ctx);
	await commands.get("scope").handler("dock full", ctx);                         // refused (too short): said why
	assert.deepEqual(modes, ["mini", "full", "full"]);
	assert.deepEqual(notes.map(([m]) => m), ["dock: mini (m in the dock or /scope dock full for the whole dock)", "dock: full",
		'/scope dock takes nothing, mini or full, not "small" · see /scope help scope', "the full dock needs 23 rows; this terminal has 20: the mini dock stays"]);
	assert.equal(notes.at(-1)![1], "error");
});

test("/scope traces and guides; four commands", async () => {
	const { commands, entries, notes, live, ctx } = setup();
	assert.deepEqual([...commands.keys()], ["steer", "scope", "concept", "jlens"]);
	await commands.get("scope").handler("off", ctx);
	assert.equal(live.level, "off");
	await commands.get("scope").handler("on", ctx);
	assert.equal(live.level, "on");
	await commands.get("scope").handler("off", ctx);
	assert.equal(live.level, "off");
	await commands.get("scope").handler("help", ctx);
	const [type, data] = entries.pop()!;
	assert.equal(type, "scope-guide");
	const text = data.lines.join("\n");
	assert.equal(data.title, "metalscope help");                               // the overview: how to start, the four commands
	assert.doesNotMatch(text, /\/scope open|web page/);
	for (const cmd of ["/model\t", "/concept learn joyful\t", "/steer joyful\t", "/scope\t", "/steer\t", "/concept\t", "/jlens\t"])
		assert.ok(text.includes(cmd), cmd);
	assert.doesNotMatch(text, /\/replay|\/learn |\/scope inspect|\/scope cancel|\/trace|\/interpret|\/meter/);
	const offered = commands.get("scope").getArgumentCompletions("").map((c: any) => c.value);
	assert.deepEqual(offered, ["status", "on", "off", "dock", "dock mini", "dock full", "jobs", "jobs cancel", "help"]);
	for (const [old, now] of [["example fluffy", 'unknown subcommand "example" · see /scope help scope'],
		["cancel", 'unknown subcommand "cancel" · see /scope help scope'], ["jobs now", '/scope jobs takes nothing or cancel, not "now" · see /scope help scope']]) {
		await commands.get("scope").handler(old, ctx);
		assert.deepEqual(notes.at(-1), [now, "error"]);
	}
	// Where those errors point: /scope's own forms.
	await commands.get("scope").handler("help scope", ctx);
	const forms = entries.pop()![1].lines;
	assert.equal(forms[0], "## /scope: tracing, the dock, GPU jobs");
	for (const f of ["/scope dock mini\t", "/scope dock full\t", "/scope jobs cancel\t"]) assert.ok(forms.some((l: string) => l.startsWith(`| ${f}`)), f);
	// /scope help completes its topics, and the commands no topic is named for.
	assert.deepEqual(commands.get("scope").getArgumentCompletions("help ").map((c: any) => c.value),
		["help tracing", "help dock", "help lens", "help steering", "help concepts", "help jobs", "help scope", "help jlens"]);
	assert.deepEqual(commands.get("scope").getArgumentCompletions("help st").map((c: any) => c.value), ["help steering"]);
});

test("/concept: the library's subcommands, its jobs handed to the job runner, and its completions", async () => {
	const { commands, entries, notes, ctx, calls } = setup();
	const concept = commands.get("concept");
	await concept.handler("learn otaku high-intensity anime speech", ctx);
	await concept.handler("calibrate anime-nerd", ctx);
	for (const sub of ["jobs", "jobs cancel"]) await commands.get("scope").handler(sub, ctx);
	assert.deepEqual(calls, [["learn", "otaku", "high-intensity anime speech", { corpus: undefined }],
		["calibrate", "anime-nerd"], ["jobs"], ["cancel"]]);
	calls.length = 0;
	await concept.handler("learn kept --corpus ~/my texts/kept", ctx);
	await concept.handler("calibrate anime-nerd", ctx);
	assert.deepEqual(calls, [["learn", "kept", "", { corpus: "~/my texts/kept" }], ["calibrate", "anime-nerd"]]);
	await concept.handler("help", ctx);
	const [, data] = entries.pop()!;
	assert.equal(data.title, "concept: help");
	assert.match(data.lines[0], /^## \/concept: /);
	assert.ok(data.lines.some((l: string) => l.startsWith("| /concept calibrate NAME\t")));
	await concept.handler("calibrate anime-nerd 0.4", ctx);
	assert.match(notes.at(-1)![0], /^\/concept calibrate takes one concept \(it searches the doses itself\) · see \/concept calibrate help$/);
	// Every subcommand answers help: as its next word, after /concept help, or when its name is missing.
	for (const [args, title] of [["calibrate help", "concept calibrate: help"], ["help calibrate", "concept calibrate: help"],
		["learn", "concept learn: help"]]) {
		await concept.handler(args, ctx);
		const [, d] = entries.pop()!;
		assert.equal(d.title, title, args);
		assert.equal(d.lines.at(-1), "· /concept help: every form of /concept.", args);
	}
	await concept.handler("calibrate help", ctx);
	const cal = entries.pop()![1].lines;
	assert.equal(cal[0], "## /concept calibrate");
	assert.ok(cal.some((l: string) => l.startsWith("| /concept calibrate NAME\tfind its chat dose")));
	assert.ok(!cal.some((l: string) => l.includes("/concept learn")));
	assert.deepEqual(concept.getArgumentCompletions("ca").map((c: any) => c.value), ["calibrate"]);
	await concept.handler("learn jobs something", ctx);
	assert.match(notes.at(-1)![0], /"jobs" is a command word, so it cannot name a concept/);
	await concept.handler("calibrate anime-nerd --pass 12", ctx);
	assert.match(notes.at(-1)![0], /^unknown option "--pass" \(this takes none\) · see \/concept calibrate help$/);
	await concept.handler("learn kept --corpus", ctx);
	assert.match(notes.at(-1)![0], /^--corpus needs a value: --corpus DIR · see \/concept learn help$/);
	await concept.handler("calibrate anime-nerd --corpus x", ctx);
	assert.match(notes.at(-1)![0], /^unknown option "--corpus" \(this takes none\)/);
	assert.equal(calls.length, 2);                                              // the errors started nothing
});

test("/jlens: status from the server, learn handed to the jobs, use, help, and errors that point at /jlens help", async () => {
	const { commands, entries, notes, ctx, calls } = setup();
	const lens = commands.get("jlens");
	await lens.handler("", ctx);
	assert.deepEqual(notes.at(-1), ["oMLX is not answering at 127.0.0.1:9: send a message (Pi starts the model) or run metalscope omlx m", "error"]);       // the status asks the server
	await lens.handler("learn ~/notes until 7:30", ctx);
	assert.deepEqual(calls, [["lensLearn", { cmd: "learn", path: "~/notes", until: "07:30" }]]);
	await lens.handler("use Qwen3.8-Flash-Next-oQ4e-mtp-learned-20261015", ctx);
	assert.deepEqual(notes.at(-1), ["/jlens use: oMLX is not answering at 127.0.0.1:9: send a message (Pi starts the model) or run metalscope omlx m", "error"]);             // one request, no job
	await lens.handler("help", ctx);
	const [, data] = entries.pop()!;
	assert.equal(data.title, "jlens: help");
	assert.equal(data.lines[0], "## /jlens: the model's Jacobian lens: check, switch, fit");
	assert.equal(data.lines[1], "| /jlens · /jlens status\tits J-lens, the other lenses ranked");
	for (const [args, why] of [["learn", "say until when: /jlens learn [PATH] until HH:MM (or windows N); the model is unloaded until then · see /jlens help"],
		["baseline", 'unknown subcommand "baseline" · see /jlens help'], ["fit", 'unknown subcommand "fit" · see /jlens help'],
		["use", "/jlens use takes one lens name (/jlens lists them) · see /jlens help"]]) {
		await lens.handler(args, ctx);
		assert.deepEqual(notes.at(-1), [why, "error"], args);
	}
	assert.equal(calls.length, 1);                                              // the errors started nothing
	await lens.handler("use help", ctx);                                          // help, not a lens named help
	assert.equal(entries.pop()![1].title, "jlens: help");
	assert.deepEqual(lens.getArgumentCompletions("").map((c: any) => c.value), ["status", "on", "off", "use", "learn", "help"]);
	await lens.handler("status", { ...ctx, model: { provider: "openai", id: "gpt" } });
	assert.deepEqual(notes.at(-1), ["Switch to a metalscope model with /model first", "warning"]);
	await lens.handler("off", ctx);
	assert.deepEqual(notes.at(-1), ["/jlens use: oMLX is not answering at 127.0.0.1:9: send a message (Pi starts the model) or run metalscope omlx m", "error"]);             // off and on: the same request as use
	for (const word of ["logit", "jlens"]) {                                       // not subcommands: off and on are
		await lens.handler(word, ctx);
		assert.deepEqual(notes.at(-1), [`unknown subcommand "${word}" · see /jlens help`, "error"]);
	}
});

test("/steer help, /concept help append their forms; bad arguments point at help; /steer replay runs the replay", async () => {
	const { commands, entries, notes, ctx, replays } = setup();
	for (const cmd of ["concept", "steer", "jlens"]) {
		await commands.get(cmd).handler("help", ctx);
		const [type, data] = entries.pop()!;
		assert.equal(type, "scope-guide");
		assert.equal(data.title, `${cmd}: help`);
		assert.ok(data.lines[0].startsWith(`## /${cmd}: `), cmd);
		assert.ok(data.lines.slice(1).every((l: string) => l.startsWith("| ") || l.startsWith("· ")), cmd);   // forms, then notes
	}
	await commands.get("steer").handler("fluffy 99", ctx);
	await commands.get("steer").handler("fluffy hold:9", ctx);
	await commands.get("steer").handler("replay fluffy 99", ctx);
	await commands.get("steer").handler("replay temp:9", ctx);
	await commands.get("scope").handler("on loud", ctx);
	await commands.get("concept").handler("adopt fluffy", ctx);
	assert.deepEqual(notes.map(([m]) => m.match(/see \/(\w+) help( scope)?$/)?.[1]), ["steer", "steer", "steer", "steer", "scope", "concept"]);
	assert.match(notes[2][0], /^dose must be /);                              // a replay's steer is parsed as /steer's
	await commands.get("steer").handler("replay", ctx);
	await commands.get("steer").handler("replay temp:0", ctx);
	assert.deepEqual(replays.map((o) => [o.steer, o.temperature]), [[undefined, undefined], [undefined, 0]]);   // the bad forms ran nothing
	await commands.get("steer").handler("fluffy + replay", ctx);
	assert.deepEqual(notes.at(-1), ["replay goes alone, not as a concept: /steer replay · see /steer help", "error"]);
	await commands.get("steer").handler("replay help", ctx);
	assert.deepEqual(notes.at(-1), ['/steer replay takes a steer to answer under (/steer replay angry@0.6) and temp:T, not "help" · see /steer help', "error"]);
	await commands.get("steer").handler("replay temp:0 temp:1", ctx);
	assert.deepEqual(notes.at(-1), ["temp:T is given once · see /steer help", "error"]);
	assert.equal(replays.length, 2);
});

test("/steer: a combination, defaults per concept, no warnings; constant on every part; a word not a dose refused", async () => {
	const concept = (name: string, layer: number, strength: number) => ({
		name, title: name, model: "m", reader: { layer, strength, concept: 0.5, control: 0.1 },
		steer: { layer, strength, basis: "chat" },
	});
	const list = { model: "m", steer_enabled: true, concepts: [concept("angry", 24, 0.4), concept("calm", 20, 0.2), concept("joyful", 22, 0.3),
		concept("old", 10, 0.5), concept("robotic", 24, 0.8), concept("fluffy", 20, 0.8), concept("early", 10, 0.3)] };
	const commands = new Map<string, any>();
	const notes: [string, string][] = [];
	const asked: string[] = [];
	const on: Record<string, (event: any, ctx: any) => void> = {};
	const pi = { registerCommand: (name: string, cmd: any) => commands.set(name, cmd), appendEntry: () => {}, on: (e: string, fn: any) => void (on[e] = fn) } as any;
	const live = { base: "http://127.0.0.1:9", clientId: "pi-test", level: "on", connected: true } as any;
	const ui = { updateUi() {}, showDock() {}, hideDock() {}, docked: () => false, steerStatus: () => "" } as any;
	const api = async (path: string) => {
		asked.push(path);
		if (path.startsWith("/api/concepts")) return list;
		throw new Error(`unexpected ${path}`);
	};
	const replays: any[] = [];
	registerCommands({ pi, live, api } as any, ui, fakeJobs().jobs, async (_c: any, o: any) => void replays.push(o));
	const ctx = { model: { provider: "metalscope-omlx", id: "m" }, hasUI: false, ui: { notify: (m: string, kind: string) => notes.push([m, kind]) } };
	on.session_start({}, ctx);
	const steer = commands.get("steer");
	await steer.handler("angry+calm@0.6+joyful", ctx);
	assert.deepEqual(live.steer.parts.map((p: any) => [p.name, p.layer, p.strength]), [["angry", 24, 0.4], ["calm", 20, 0.6], ["joyful", 22, 0.3]]);
	assert.ok(live.steer.parts.every((p: any) => !("hold" in p) && !("calibration" in p)));
	assert.deepEqual(notes.filter(([, k]) => k === "warning"), []);
	let reply = notes.at(-1)![0];
	// Only what the line above the editor does not say: each part's dose facts under its name.
	assert.match(reply, /^angry\n {2}Not calibrated for chat yet: \/concept calibrate angry finds its dose\.\ncalm\n/);
	assert.doesNotMatch(reply, /^steer:|Every token|\/steer replay|has each one's details/m);
	assert.doesNotMatch(reply, /experimental|thermostat|\bhold\b|hold:|pacing/);   // the cut thermostat leaves no line behind
	// A word after a concept that is not a dose is refused, and the steer stays in place.
	for (const [args, word] of [["angry fade:1", "fade:1"], ["angry hold:0.8", "hold:0.8"], ["robotic@0.5 hold:1", "hold:1"], ["fluffy meter:0.6", "meter:0.6"]]) {
		await steer.handler(args, ctx);
		assert.deepEqual(notes.at(-1), [`"${word}" is not a dose · see /steer help`, "error"], args);
		assert.equal(live.steer.parts.length, 3, args);
	}
	await steer.handler("", ctx);
	assert.ok(notes.at(-1)![0].startsWith("steer: angry@0.4 L24 + calm@0.6 L20 + joyful@0.3 L22\nheader: angry@0.4/24+calm@0.6/20+joyful@0.3/22\n"));
	assert.doesNotMatch(notes.at(-1)![0], /pacing/);
	await commands.get("concept").handler("list", ctx);
	assert.match(notes.at(-1)![0], /On now: steer angry@0\.4 L24 \+ calm@0\.6 L20 \+ joyful@0\.3 L22\.$/);
	await steer.handler("list", ctx);
	assert.deepEqual(notes.at(-1), ["/steer list is not a form: the concepts are /concept list · see /steer help", "error"]);
	// One concept at a dose of its own replaces the steer; the header has one part.
	await steer.handler("robotic@0.5", ctx);
	assert.deepEqual(live.steer.parts.map((p: any) => [p.name, p.layer, p.strength]), [["robotic", 24, 0.5]]);
	assert.deepEqual(notes.at(-1), ["Not calibrated for chat yet: /concept calibrate robotic finds its dose.", "info"]);
	await steer.handler("status", ctx);
	assert.match(notes.at(-1)![0], /\nheader: robotic@0\.5\/24\n/);
	assert.match(notes.at(-1)![0], /\nEvery token, thinking included\. \/steer off stops it\.\n.*\/steer replay.*\n\/concept robotic has the details\.$/);
	// /steer replay STEER: the parts resolved as /steer's, handed to the replay; the steer that is on stays.
	await steer.handler("replay angry@0.6 + fluffy temp:0", ctx);
	assert.deepEqual(replays.at(-1).steer.parts.map((p: any) => [p.name, p.layer, p.strength]), [["angry", 24, 0.6], ["fluffy", 20, 0.8]]);
	assert.equal(replays.at(-1).temperature, 0);
	assert.deepEqual(live.steer.parts.map((p: any) => [p.name, p.strength]), [["robotic", 0.5]]);
	// Parts at different layers, each its own.
	await steer.handler("angry + fluffy + early + old", ctx);
	assert.deepEqual(live.steer.parts.map((p: any) => [p.name, p.layer, p.strength]), [["angry", 24, 0.4], ["fluffy", 20, 0.8], ["early", 10, 0.3], ["old", 10, 0.5]]);
	await steer.handler("", ctx);
	assert.match(notes.at(-1)![0], /\nheader: angry@0\.4\/24\+fluffy@0\.8\/20\+early@0\.3\/10\+old@0\.5\/10\n/);
	assert.ok(!asked.some((p) => p.includes("overlap")));                        // no cross-talk to ask about
	// A word that is not a dose is refused before anything is built (no lens word vector is made for velvet).
	const before = asked.length;
	await steer.handler("hold fluffy", ctx);
	assert.deepEqual(notes.at(-1), ['"fluffy" is not a dose · see /steer help', "error"]);
	await steer.handler("velvet hold:1", ctx);
	assert.deepEqual(notes.at(-1), ['"hold:1" is not a dose · see /steer help', "error"]);
	assert.ok(!asked.slice(before).some((p) => p.includes("/api/steer/words")));
	assert.equal(live.steer.parts[0].name, "angry");
	// /steer off stops everything.
	await steer.handler("off", ctx);
	assert.equal(live.steer, undefined);
	assert.equal(notes.at(-1)![0], "steer: off");
	notes.length = 0;
	await steer.handler("calm", ctx);                                          // already warned about calm this session
	assert.equal(notes.filter(([m, k]) => k === "warning" && m.startsWith("calm")).length, 0);
	await steer.handler("off", ctx);
	// Completions: a concept; after it nothing (a dose is your own); after +, another concept (not one already typed).
	await commands.get("concept").handler("list", ctx);                         // fills the concept cache
	assert.deepEqual(steer.getArgumentCompletions("ang").map((c: any) => c.value), ["angry"]);
	assert.deepEqual(steer.getArgumentCompletions("rep").map((c: any) => c.value), ["replay"]);
	assert.deepEqual(steer.getArgumentCompletions("replay t").map((c: any) => c.value), ["replay temp:0"]);
	assert.deepEqual(steer.getArgumentCompletions("replay ang").map((c: any) => c.value), ["replay angry"]);
	for (const after of ["angry ", "angry me", "angry 0.4 ", "angry + fluffy@0.4 h"]) assert.deepEqual(steer.getArgumentCompletions(after), [], after);
	assert.deepEqual(steer.getArgumentCompletions("angry + fl").map((c: any) => c.value), ["angry + fluffy"]);
	assert.deepEqual(steer.getArgumentCompletions("angry@0.4 + fl").map((c: any) => c.value), ["angry@0.4 + fluffy"]);
	assert.deepEqual(steer.getArgumentCompletions("angry + an").map((c: any) => c.value), []);                // angry is already typed
	// After /model the cached list is another model's: nothing offered until the new model's list is fetched.
	on.model_select({}, { ...ctx, model: { provider: "metalscope-omlx", id: "other" } });
	assert.deepEqual(steer.getArgumentCompletions("ang"), []);
});

test("/steer with a chat calibration: the chat dose by default, a dose of your own beside it", async () => {
	// A chat calibration as the server sends it (version 4): `good` of 18 replies at `dose`, made steering at `layer`.
	const v4 = (dose: number, good: number, layer = 20) => ({ version: 4, layer, default_dose: dose, good, turns: 18,
		strengths: [{ dose, good, turns: 18, attempts: 18, failed: { tool: 0, thinking: 0, short: 0 }, coherence: 2.8 }], grader: { model: "m", thinking: false } });
	const concept = (name: string, calibration: any, steer: any = null) => ({
		name, title: name, model: "m", reader: { layer: 20, strength: 0.8, concept: 0.5, control: 0.1 },
		steer, calibration });
	const list = { model: "m", steer_enabled: true, concepts: [
		concept("fluffy", v4(0.4, 16), { layer: 20, strength: 0.4, basis: "chat" }),
		concept("anime", v4(0.2, 8), { layer: 20, strength: 0.2, basis: "chat" }),
		concept("old", v4(0.4, 16, 18)),                                       // made steering at another layer
		concept("raw", null)] };
	const commands = new Map<string, any>();
	const notes: [string, string][] = [];
	const entries: any[] = [];
	const pi = { registerCommand: (name: string, cmd: any) => commands.set(name, cmd), appendEntry: (_: string, e: any) => entries.push(e), on: () => {} } as any;
	const live = { base: "http://127.0.0.1:9", clientId: "pi-test", level: "on", connected: true } as any;
	const ui = { updateUi() {}, showDock() {}, hideDock() {}, docked: () => false, steerStatus: () => "" } as any;
	const api = async (path: string) => {
		if (path.startsWith("/api/concepts/")) throw new Error("HTTP 404");
		if (path.startsWith("/api/concepts")) return list;
		throw new Error(`unexpected ${path}`);
	};
	registerCommands({ pi, live, api } as any, ui, fakeJobs().jobs, async () => {});
	const ctx = { model: { provider: "metalscope-omlx", id: "m" }, hasUI: false, ui: { notify: (m: string, kind: string) => notes.push([m, kind]) } };
	const steer = commands.get("steer");
	// The chat dose is the default: nothing said (the line above the editor has its figure); /steer status has it all.
	const before = notes.length;
	await steer.handler("fluffy", ctx);
	assert.equal(live.steer.parts[0].strength, 0.4);
	assert.equal(notes.length, before);
	await steer.handler("anime", ctx);
	assert.equal(live.steer.parts[0].strength, 0.2);
	assert.equal(notes.length, before);
	await steer.handler("status", ctx);
	assert.match(notes.at(-1)![0], /\nIn chat: 8 of 18 replies good \(the most of the doses tried\)\n/);
	// A dose given: the chat default is named beside it; the header carries the dose, constant.
	await steer.handler("fluffy 0.5", ctx);
	assert.deepEqual(notes.at(-1), ["Dose 0.5 was not tried in chat.\nDefault dose 0.4: 16 of 18 good (the most)", "info"]);
	if (process.env.SHOW_REPLY) console.log(notes.at(-1)![0]);
	await steer.handler("", ctx);
	assert.match(notes.at(-1)![0], /\nheader: fluffy@0\.5\/20\n/);
	// A word that is not a dose is refused, and the steer on stays as it was.
	for (const [bad, word] of [["fluffy hold:0.8", "hold:0.8"], ["fluffy hold", "hold"], ["fluffy meter", "meter"]]) {
		await steer.handler(bad, ctx);
		assert.deepEqual(notes.at(-1), [`"${word}" is not a dose · see /steer help`, "error"], bad);
	}
	assert.deepEqual(live.steer.parts.map((p: any) => [p.name, p.strength]), [["fluffy", 0.5]]);
	assert.ok(!("hold" in live.steer.parts[0]) && !("calibration" in live.steer.parts[0]));
	// Not calibrated on this model: no default, so no dose is no steer; one line says what to do. A dose given steers.
	await steer.handler("raw", ctx);
	assert.deepEqual(notes.at(-1), ["raw isn't calibrated on this model: give a dose (/steer raw 0.4) or run /concept calibrate raw.", "error"]);
	await steer.handler("old + fluffy", ctx);
	assert.deepEqual(notes.at(-1), ["old's calibration was made at L18, it steers at L20 now: give a dose (/steer old 0.4) or run /concept calibrate old.", "error"]);
	await steer.handler("replay raw", ctx);
	assert.equal(notes.at(-1)![0], "raw isn't calibrated on this model: give a dose (/steer raw 0.4) or run /concept calibrate raw.");
	assert.deepEqual(live.steer.parts.map((p: any) => [p.name, p.strength]), [["fluffy", 0.5]]);   // the steer on stays as it was
	await steer.handler("raw 0.3", ctx);
	assert.deepEqual(live.steer.parts.map((p: any) => [p.name, p.strength]), [["raw", 0.3]]);
	assert.deepEqual(notes.at(-1), ["Not calibrated for chat yet: /concept calibrate raw finds its dose.", "info"]);
	await steer.handler("fluffy 0.5", ctx);
	// /concept list: ranked by good replies in chat, then those without a score.
	await commands.get("concept").handler("list", ctx);
	const listed = notes.at(-1)![0];
	assert.match(listed, /^Learned on m,\nranked by good replies in conversation at the dose \/steer uses:\n +concept +good in chat +dose\n/);
	assert.match(listed, /\nbest in chat\n 1 fluffy +16 of 18 +0\.4\n 2 anime + 8 of 18 +0\.2\nno chat score\n/);
	assert.match(listed, /\n +old +re-run calibrate +-\n/);
	assert.match(listed, /\n +raw +not calibrated +-\n/);
	assert.match(listed, /\ngood in chat: [^\n]*at its best dose\.\ndose "-": not calibrated[^\n]*\n\/concept NAME for details[^\n]*\nOn now: /);
	assert.doesNotMatch(listed, /hold/);
	assert.ok(listed.split("\n").every((l) => l.length <= 80), "every line of /concept list fits 80 columns");
	// /concept NAME: the concept in the transcript, its rank among the concepts learned here and the backend, drawn when
	// Pi renders it; no other request than the list.
	await commands.get("concept").handler("anime", ctx);
	assert.deepEqual(entries.at(-1), { v: 1, concept: parseConcepts(list).find((c) => c.name === "anime"), rank: { rank: 2, of: 2 }, backend: "omlx" });
	await commands.get("concept").handler("fluffy", ctx);
	assert.deepEqual(entries.at(-1).rank, { rank: 1, of: 2 });
	await commands.get("concept").handler("raw", ctx);                          // not calibrated: no rank
	assert.equal(entries.at(-1).rank, null);
	await commands.get("concept").handler("nope", ctx);
	assert.equal(notes.at(-1)![0], "nope is not learned on m and is not in the library (learned here: fluffy, anime, old, raw) · see /concept help");
});

test("completions stop after a word that goes alone; learn and calibrate need a metalscope model", async () => {
	const { commands, notes, calls } = setup();
	for (const w of ["off", "status", "help"]) assert.deepEqual(commands.get("steer").getArgumentCompletions(`${w} `), [], w);
	const other = { model: { provider: "anthropic", id: "claude" }, hasUI: false, ui: { notify: (m: string, kind: string) => notes.push([m, kind]) } };
	await commands.get("concept").handler("learn otaku dramatic anime speech", other);
	assert.deepEqual(notes.at(-1), ["Switch to a metalscope model with /model first: it writes the corpus", "warning"]);
	await commands.get("concept").handler("calibrate fluffy", other);
	assert.deepEqual(notes.at(-1), ["Switch to a metalscope model with /model first: it holds the conversations", "warning"]);
	assert.equal(calls.length, 0);
	// /steer off and /steer status answer on any model.
	await commands.get("steer").handler("status", other);
	assert.deepEqual(notes.at(-1), ["steer: off", "info"]);
});

test("a fresh model: /concept leads with the library; /steer of a library concept says how to learn it; a word is no concept", async () => {
	const lib = [{ name: "joyful", title: "Joyful", description: "Feeling joyful: delight.", writer: "Claude Fable 5.1 (by hand)", texts: "texts: hand-written by Claude Fable 5.1 (2026-10-06)" },
		{ name: "calm", title: "Calm", description: "Feeling calm.", writer: null, texts: "texts: each model writes its own" }];
	const list = { model: "Qwen3.6-35B-A3B-MLX-4bit", steer_enabled: true, concepts: [], library: lib };
	const commands = new Map<string, any>();
	const notes: [string, string][] = [];
	const asked: string[] = [];
	const pi = { registerCommand: (name: string, cmd: any) => commands.set(name, cmd), appendEntry: () => {}, on: () => {} } as any;
	const live = { base: "http://127.0.0.1:8008", clientId: "pi-test", level: "on", connected: true } as any;
	const ui = { updateUi() {}, showDock() {}, hideDock() {}, docked: () => false, steerStatus: () => "" } as any;
	const api = async (path: string) => {
		asked.push(path);
		if (path.startsWith("/api/concepts")) return list;
		throw new Error(`unexpected ${path}`);
	};
	registerCommands({ pi, live, api } as any, ui, fakeJobs().jobs, async () => {});
	const ctx = { model: { provider: "metalscope-omlx", id: "Qwen3.6-35B-A3B-MLX-4bit" }, hasUI: false, ui: { notify: (m: string, kind: string) => notes.push([m, kind]) } };
	await commands.get("concept").handler("", ctx);
	const shown = notes.at(-1)![0];
	if (process.env.SHOW_REPLY) console.log(shown);
	assert.deepEqual(shown.split("\n"), ["Nothing learned on Qwen3.6-35B-A3B-MLX-4bit yet.", "Ready to learn: 2 in the library", "texts by Claude Fable 5.1 (by hand)",
		"  joyful        Feeling joyful: delight.", "texts: each model writes its own", "  calm          Feeling calm.",
		"Learn one, no description needed: /concept learn joyful",
		"Each is learned then calibrated in chat: about 26 min (measured on Qwen3.6).", "On now: nothing."]);
	// /steer joyful before it is learned: how to learn it, no word steer of its name; nothing changes.
	await commands.get("steer").handler("joyful", ctx);
	assert.deepEqual(notes.at(-1), ["joyful is not learned on Qwen3.6-35B-A3B-MLX-4bit yet: /concept learn joyful, about 26 min (measured on Qwen3.6)", "error"]);
	assert.ok(!asked.includes("/api/steer/words") && live.steer === undefined);
	await commands.get("steer").handler("joyful", { ...ctx, model: { provider: "metalscope-ds4", id: "Qwen3.6-35B-A3B-MLX-4bit" } });
	assert.match(notes.at(-1)![0], /, about 4 to 7\.5 h on ds4$/);
	// /concept joyful before it is learned: how to learn it.
	await commands.get("concept").handler("joyful", ctx);
	assert.equal(notes.at(-1)![0], "joyful is not learned on Qwen3.6-35B-A3B-MLX-4bit yet: /concept learn joyful, about 26 min (measured on Qwen3.6)");
	// A word that is no concept: said so, no word vector made (a word changes one token: the dock's edit).
	await commands.get("steer").handler("velvet", ctx);
	assert.deepEqual(notes.at(-1), ["velvet is not a concept here: /concept lists them · see /steer help", "error"]);
	assert.ok(!asked.includes("/api/steer/words") && live.steer === undefined);
});

test("a word is the dock's LENS edit's alone: its vectors over the server's band, on either engine; /steer refuses it", async () => {
	const list = { model: "Flash", steer_enabled: true, concepts: [], library: [] };
	const commands = new Map<string, any>();
	const notes: [string, string][] = [];
	const asked: string[] = [], bodies: any[] = [];
	const pi = { registerCommand: (name: string, cmd: any) => commands.set(name, cmd), appendEntry: () => {}, on: () => {} } as any;
	const live = { base: "http://127.0.0.1:8008", clientId: "pi-test", level: "on", connected: true } as any;
	const ui = { updateUi() {}, showDock() {}, hideDock() {}, docked: () => false, steerStatus: () => "" } as any;
	const api = async (path: string, init?: any) => {
		asked.push(path);
		if (path.startsWith("/api/concepts")) return list;
		const name = JSON.parse(init.body).name;
		bodies.push(JSON.parse(init.body));
		if (path === "/api/steer/words" && name === "calm") throw Object.assign(new Error("'calm' is a concept: learn it (/concept learn calm ...) and steer it by name"), { status: 409 });
		if (path === "/api/steer/words") return { name, model: "Flash", band: [18, 44], cut: ["pineapple", "ice-cream"].includes(bodies.at(-1).words[0]) && !bodies.at(-1).ids ? bodies.at(-1).words : [] };
		throw new Error(`unexpected ${path}`);
	};
	registerCommands({ pi, live, api } as any, ui, fakeJobs().jobs, async () => {});
	const ctx = { model: { provider: "metalscope-omlx", id: "Flash" }, hasUI: false, ui: { notify: (m: string, kind: string) => notes.push([m, kind]) } };
	// Typed, a word removed or swapped is refused: nothing asked of the server, nothing turned on.
	for (const form of ["-soccer", "soccer>rugby", "calm + -soccer", "replay -soccer"]) {
		await commands.get("steer").handler(form, ctx);
		assert.deepEqual(notes.at(-1), ["a word changes one token (the dock's LENS edit), not a whole reply · see /steer help", "error"], form);
	}
	assert.ok(!asked.includes("/api/steer/words") && live.steer === undefined);
	// The dock's LENS edit (`use`): the removed word by the id of the token shown, the typed word as a word.
	let built: any;
	const use = (s: any) => (built = s);
	await commands.get("steer").handler("soccer>rugby@0.7", ctx, { soccer: { id: 4321, text: "soccer" } }, use);
	assert.deepEqual(bodies, [{ model: "Flash", name: "soccer", words: ["soccer"], ids: [4321] }, { model: "Flash", name: "rugby", words: ["rugby"] }]);
	assert.equal(steerHeader(built), "remove~soccer/18-44+rugby@0.7/31");
	await commands.get("steer").handler("-soccer", { ...ctx, model: { provider: "metalscope-ds4", id: "Flash" } }, undefined, use);
	assert.equal(steerHeader(built), "remove~soccer/18-44");                                      // ds4 as oMLX
	// By its token's id the server cuts nothing; the word typed there is a word: more than one token is refused.
	await commands.get("steer").handler("tok7>rugby@0.7", ctx, { tok7: { id: 7, text: "pineapple" } }, use);
	assert.equal(steerHeader(built), "remove~tok7/18-44+rugby@0.7/31");
	built = undefined;
	for (const form of ["tok7>pineapple@0.7", "-pineapple"]) {
		await commands.get("steer").handler(form, ctx, { tok7: { id: 7, text: "pineapple" } }, use);
		assert.deepEqual([notes.at(-1), built], [["pineapple is more than one token: the edit takes a one-token word", "error"], undefined], form);
	}
	// A word typed in the dock that is a concept's name: said for the key line.
	await commands.get("steer").handler("tok42>calm@0.7", ctx, { tok42: { id: 42, text: "reader" } }, use);
	assert.deepEqual([notes.at(-1), built], [['"calm" is a concept here, not a word: /steer calm steers it', "error"], undefined]);
	assert.ok(bodies.every((b: any) => b.edit === undefined) && live.steer === undefined);
	await commands.get("steer").handler("again", { ...ctx, isIdle: () => true, sessionManager: { getBranch: () => [] } });   // answerAgain (replay.test.ts)
	assert.deepEqual(notes.at(-1), ["no message to answer again", "error"]);
});

test("the server down names the engine and the next step; an error it answered is its own", async () => {
	const notes: [string, string][] = [];
	const commands = new Map<string, any>();
	const pi = { registerCommand: (name: string, cmd: any) => commands.set(name, cmd), appendEntry: () => {}, on: () => {} } as any;
	const live = { base: "http://127.0.0.1:8009", clientId: "pi-test", level: "on", connected: true } as any;
	let fail: any = new TypeError("fetch failed");
	registerCommands({ pi, live, api: async () => { throw fail; } } as any, {} as any, fakeJobs().jobs, async () => {});
	const ctx = { model: { provider: "metalscope-ds4", id: "deepseek-v4-flash" }, hasUI: false, ui: { notify: (m: string, kind: string) => notes.push([m, kind]) } };
	await commands.get("concept").handler("", ctx);
	assert.deepEqual(notes.at(-1), ["ds4 is not answering at 127.0.0.1:8009: send a message (Pi starts the model) or run metalscope ds4 deepseek-v4-flash", "error"]);
	await commands.get("jlens").handler("use x", ctx);
	assert.deepEqual(notes.at(-1), ["/jlens use: ds4 is not answering at 127.0.0.1:8009: send a message (Pi starts the model) or run metalscope ds4 deepseek-v4-flash", "error"]);
	fail = Object.assign(new Error("model 'x' is not loaded"), { status: 404 });
	await commands.get("concept").handler("", ctx);
	assert.deepEqual(notes.at(-1), ["ds4 at 127.0.0.1:8009 answered: model 'x' is not loaded", "error"]);
});

test("a request that failed is said in the same words everywhere; a bug reading an answer is not the server down", async () => {
	const notes: [string, string][] = [];
	const commands = new Map<string, any>();
	const pi = { registerCommand: (name: string, cmd: any) => commands.set(name, cmd), appendEntry: () => {}, on: () => {} } as any;
	const live = { base: "http://127.0.0.1:8008", clientId: "pi-test", level: "on", connected: true } as any;
	const down = new TypeError("fetch failed");
	const api = async (path: string, init?: any) => {
		if (path.startsWith("/api/concepts?")) return { model: "Q", steer_enabled: false, concepts: [{ name: "calm", title: "Calm" }] };
		if (path === "/api/steer/words") {
			const { name } = JSON.parse(init.body);
			return name === "velvet" ? { name, model: "Q", band: [18, 44] } : null;   // null: a malformed answer
		}
		throw down;
	};
	const ui = { updateUi() {}, steerStatus: () => "" } as any;
	registerCommands({ pi, live, api } as any, ui, fakeJobs().jobs, async () => {});
	const ctx = { model: { provider: "metalscope-omlx", id: "Q" }, hasUI: false, ui: { notify: (m: string, kind: string) => notes.push([m, kind]) } };
	const unreachable = "oMLX is not answering at 127.0.0.1:8008: send a message (Pi starts the model) or run metalscope omlx Q";
	await commands.get("concept").handler("calm", ctx);                         // /concept NAME asks for the list alone
	assert.deepEqual(notes, []);
	await commands.get("steer").handler("-velvet", ctx, undefined, () => {});   // steering off on the server: turning it on fails
	assert.deepEqual(notes.at(-1), [`Could not enable steering on the server: ${unreachable}`, "error"]);
	const n = notes.length;
	await assert.rejects(commands.get("steer").handler("-sunny", ctx, undefined, () => {}), TypeError);
	assert.equal(notes.length, n, "not said as the server down");
});

test("completions with a model selected: /concept a learned name, learn a library one, calibrate one it can steer; /jlens use a lens", async () => {
	const handlers = new Map<string, any>();
	const commands = new Map<string, any>();
	const asked: string[] = [];
	const pi = { registerCommand: (name: string, cmd: any) => commands.set(name, cmd), appendEntry: () => {}, on: (e: string, f: any) => handlers.set(e, f) } as any;
	const live = { base: "http://127.0.0.1:8008", clientId: "pi-test", level: "on", connected: true } as any;
	const ui = { updateUi() {}, showDock() {}, hideDock() {}, docked: () => false, steerStatus: () => "" } as any;
	const concepts = { model: "Q", steer_enabled: true, library: [{ name: "joyful" }, { name: "calm" }],
		concepts: [{ name: "fluffy", reader: { layer: 20, strength: 0.4 }, title: "Fluffy" }, { name: "terse", title: "Terse" }] };
	const lens = { model: "Q", installed: true, candidates: [{ name: "Q-learned-1", in_use: true, why: "in use" }, { name: "logit", logit: true, in_use: false, why: "" },
		{ name: "Q-learned-2", in_use: false, why: "learned for it" }] };
	const api = async (path: string) => {
		asked.push(path);
		if (path.startsWith("/api/concepts")) return concepts;
		if (path.startsWith("/api/lens")) return lens;
		throw new Error(`unexpected ${path}`);
	};
	registerCommands({ pi, live, api } as any, ui, fakeJobs().jobs, async () => {});
	const values = async (r: any) => (await r).map((c: any) => c.value);
	const concept = commands.get("concept"), jlens = commands.get("jlens");
	// No model selected yet: the subcommands alone, nothing asked.
	assert.deepEqual(await values(concept.getArgumentCompletions("le")), ["learn"]);
	assert.deepEqual(asked, []);
	handlers.get("session_start")({}, { model: { provider: "metalscope-omlx", id: "Q" } });
	assert.deepEqual(await values(concept.getArgumentCompletions("learn ")), ["learn joyful", "learn calm"]);   // asked for once
	assert.deepEqual(asked, ["/api/concepts?model=Q"]);
	assert.deepEqual(await values(concept.getArgumentCompletions("f")), ["fluffy"]);                            // bare: a learned one
	assert.deepEqual(await values(concept.getArgumentCompletions("")), ["list", "learn", "calibrate", "help", "fluffy", "terse"]);
	assert.deepEqual(await values(concept.getArgumentCompletions("calibrate ")), ["calibrate fluffy"]);
	assert.equal((asked as string[]).length, 1, "the list kept for the model");
	assert.deepEqual(await values(jlens.getArgumentCompletions("use Q")), ["use Q-learned-1", "use Q-learned-2"]);   // the logit lens is /jlens off
	assert.deepEqual(await values(jlens.getArgumentCompletions("use Q-learned-2")), ["use Q-learned-2"]);
	assert.equal((asked as string[]).filter((a) => a.startsWith("/api/lens")).length, 1);
	assert.deepEqual(await values(jlens.getArgumentCompletions("us")), ["use"]);
	// A server that does not answer: no names, no throw.
	const quiet = new Map<string, any>();
	const on = new Map<string, any>();
	registerCommands({ pi: { registerCommand: (n: string, c: any) => quiet.set(n, c), appendEntry: () => {}, on: (e: string, f: any) => on.set(e, f) }, live,
		api: async () => { throw new Error("offline"); } } as any, ui, fakeJobs().jobs, async () => {});
	on.get("session_start")({}, { model: { provider: "metalscope-omlx", id: "Q" } });
	assert.deepEqual(await values(quiet.get("concept").getArgumentCompletions("learn ")), []);
	assert.deepEqual(await values(quiet.get("jlens").getArgumentCompletions("use ")), []);
});

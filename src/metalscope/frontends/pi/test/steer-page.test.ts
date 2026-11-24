import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { registerCommands } from "../commands.ts";
import { Dock } from "../dock.ts";
import { type Live, type TokenEvent, onEvent } from "../state.ts";
import { applyLiveHeaders, editStarts } from "../provider.ts";
import { LABELS, dockTheme } from "../draw.ts";
const dt = dockTheme();   // the dock's own colours
import {
	answerText, doseLine, fmt, mdLines, noteOf, parseExamples, parseSteering, pickIndex, reading, starred,
} from "../steerpage.ts";
import { createUi, dockStep, registerShortcuts } from "../widget.ts";

const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const COLOURS: Record<string, number> = { success: 2, muted: 8, dim: 7, accent: 6, warning: 3, error: 1 };
const theme = {
	fg: (c: string, s: string) => `\x1b[38;5;${COLOURS[c] ?? 15}m${s}\x1b[39m`,
	bg: (_c: string, s: string) => `\x1b[48;5;236m${s}\x1b[49m`,
	bold: (s: string) => `\x1b[1m${s}\x1b[22m`,
};

// -- the Steer page's data, shaped as /api/steering sends it (web.py steering) --------------------------------------

const row = (dose: number, good: number, style: number | null, coherence: number) => ({ dose, good, turns: 18, style, coherence, graded: 18 });
function chatConcept(name: string, layer: number, pick: number, goods: [number, number][]) {
	const strengths = goods.map(([dose, good]) => row(dose, good, Math.min(3, dose * 6), Math.max(0, 3 - dose * 3)));
	return { name, title: `${name[0].toUpperCase()}${name.slice(1)} register`, description: `Sounding ${name}: a long description of the register that runs over two lines of the dock and on past them so the second line is cut.`,
		chat: { default_dose: pick, good: goods.find(([d]) => d === pick)?.[1] ?? 0, turns: 18, layer,
			plain: { dose: 0, good: 0, turns: 18, attempts: 18, failed: 0, coherence: 3, graded: 18, style: 0.1 }, strengths }, chat_stale: false };
}
const STEERING = {
	model: "Flash", checkpoint: "Flash", served: "Flash-ds4", concepts: [
		{ name: "doubt", title: "Doubt", description: "Hedging.", chat: null, chat_stale: false },
		chatConcept("calm", 8, 0.6, [[0.2, 3], [0.4, 7], [0.6, 9], [0.8, 4]]),
		chatConcept("robotic", 24, 0.3, [[0.1, 2], [0.2, 9], [0.3, 15], [0.4, 12], [0.6, 3]]),
		chatConcept("joyful", 16, 0.3, [[0.2, 9], [0.3, 14], [0.4, 16], [0.6, 11]]),
		chatConcept("melancholic", 4, 0.6, [[0.2, 3], [0.4, 10], [0.6, 15], [0.8, 12]]),
		chatConcept("tender", 24, 0.4, [[0.2, 6], [0.3, 11], [0.4, 13], [0.6, 9]]),
		chatConcept("terse", 8, 0.4, [[0.2, 6], [0.4, 13], [0.6, 10]]),
		chatConcept("anxious", 20, 0.3, [[0.2, 4], [0.3, 7], [0.4, 5]]),
		{ name: "broken", error: "bad profile" },
	],
};

test("parseSteering: chat-tested first, by good turns at the picked dose (then the lower dose); ranked, the top five starred", () => {
	const l = parseSteering(STEERING, "Flash-ds4");
	assert.deepEqual(l.concepts.map((c) => c.name), ["robotic", "melancholic", "joyful", "tender", "terse", "calm", "anxious", "doubt"]);
	assert.deepEqual(l.concepts.map((c) => c.rank), [1, 2, 3, 4, 5, 6, 7, undefined]);
	assert.deepEqual(l.concepts.filter(starred).map((c) => c.name), ["robotic", "melancholic", "joyful", "tender", "terse"]);
	const robotic = l.concepts[0];
	assert.deepEqual([robotic.good, robotic.pick, robotic.layer], [15, 0.3, 24]);
	assert.deepEqual(robotic.doses.map((d) => d.dose), [0, 0.1, 0.2, 0.3, 0.4, 0.6]);          // no steering first
	assert.equal(l.concepts[2].good, 14);                                                     // joyful: good at its pick, not its best (16)
	const calm = l.concepts.find((c) => c.name === "calm")!;
	assert.equal(calm.good, 9);
	assert.ok(!("holds" in calm) && !("best" in calm));
	const doubt = l.concepts.at(-1)!;
	assert.deepEqual([doubt.chat, doubt.pick, doubt.doses.map((d) => d.dose)], [false, 0, [0]]);            // not chat-tested: no steering only
	// The layer /steer uses wins over the tab's: the concept list's reader layer (null: none).
	const moved = parseSteering(STEERING, "Flash-ds4", { robotic: 20, doubt: null });
	assert.deepEqual([moved.concepts[0].layer, moved.concepts.at(-1)!.layer], [20, null]);
	assert.deepEqual(parseSteering({}, "m").concepts, []);
	assert.deepEqual(parseSteering(null, "m").concepts, []);
});

test("the dose line and its reading", () => {
	const l = parseSteering(STEERING, "Flash-ds4"), robotic = l.concepts[0], doubt = l.concepts.at(-1)!;
	assert.equal(pickIndex(robotic), 3);
	assert.equal(pickIndex(doubt), 0);
	assert.deepEqual(doseLine(robotic, robotic.doses[3]), ["dose 0.3", " · 15/18 good · style 1.8 · sense 2.1"]);
	assert.deepEqual(doseLine(robotic, robotic.doses[0]), ["no steering", " · style 0.1 · sense 3.0"]);
	assert.deepEqual(doseLine(doubt, doubt.doses[0]), ["no steering", " · style – · sense –"]);
	assert.deepEqual(reading(robotic, robotic.doses[3]), ["success", "the dose the chat test picked (most good)"]);
	assert.deepEqual(reading(robotic, robotic.doses[0]), ["muted", "what the model does on its own"]);
	assert.deepEqual(reading(robotic, robotic.doses[1]), ["muted", "little style here"]);
	assert.deepEqual(reading(robotic, robotic.doses[4]), ["muted", "fewer good than the pick"]);
	assert.deepEqual(reading(robotic, robotic.doses[5]), ["muted", "sense falls here"]);
	// Doses that ran away, from the server's payload: the pick (graded) says so after the pick; one not graded says that.
	const wild: any = chatConcept("wild", 8, 0.3, [[0.2, 9], [0.3, 12], [0.4, 0]]);
	wild.chat.strengths[1] = { ...wild.chat.strengths[1], runaway: 7 };
	wild.chat.strengths[2] = { ...wild.chat.strengths[2], runaway: 12, style: null, coherence: null, graded: 0 };
	const w = parseSteering({ concepts: [wild] }, "Flash").concepts[0];
	assert.deepEqual(w.doses.map((d) => d.runaway), [undefined, undefined, 7, 12]);
	assert.deepEqual(reading(w, w.doses[2]), ["success", "the dose the chat test picked (most good); ran 7x the plain length"]);
	assert.deepEqual(reading(w, w.doses[3]), ["muted", "ran 12x the plain length, not graded"]);
	const joyful = l.concepts[2];
	assert.deepEqual(reading(joyful, joyful.doses[3]), ["muted", "as many good as the pick"]);
	assert.deepEqual(reading(doubt, doubt.doses[0]), ["warning", "not chat-tested"]);                  // no layer: neither command takes it
	assert.deepEqual(reading({ ...doubt, layer: 8 }, doubt.doses[0]), ["warning", "not chat-tested: /steer doubt DOSE, or /concept calibrate doubt"]);
	assert.deepEqual(reading({ ...doubt, layer: 8, stale: true }, doubt.doses[0]), ["warning", "chat test out of date: /steer doubt DOSE, or /concept calibrate doubt"]);
	assert.deepEqual(reading(doubt, doubt.doses[0], ["calm"]), ["warning", "enter turns steering off (calm is on)"]);
	assert.equal(fmt(0), "0");
	assert.equal(fmt(0.30000000000000004), "0.3");
});

test("answerText: thinking hidden, closed, unclosed or before a lone </think>", () => {
	assert.equal(answerText("<think>plan\nmore</think>\n\nHey!\n\n\n\nThere."), "Hey!\n\nThere.");
	assert.equal(answerText("Start. <think>never closed\nso on"), "Start.");
	assert.equal(answerText("thinking without its tag\n</think>\nThe answer."), "The answer.");
	assert.equal(answerText("<think>only thinking</think>"), "(no answer, only thinking)");
	assert.equal(answerText(""), "(no answer, only thinking)");
});

const CHATS = {
	name: "robotic", served: "Flash-ds4", dose: 0.3, default_dose: 0.3, conversations: [
		{ arc: "casual", title: "A casual chat", turns: [
			{ turn: 1, user: "Hey! How's it going?", plain: { text: "<think>hi</think>\n\nGoing great, thanks for asking!", grade: { style: 0, coherence: 3 }, good: false },
				steered: { text: "<think>status</think>\n\nStatus: nominal.", grade: { style: 2, coherence: 3 }, good: true } },
			{ turn: 2, user: "Any pets?", plain: { text: Array.from({ length: 30 }, (_, i) => `- point ${i}`).join("\n"), grade: { style: 0, coherence: 3 }, good: false },
				steered: { text: Array.from({ length: 12 }, (_, i) => `Line ${i}.`).join("\n\n"), grade: null, good: false } },
		] },
		{ arc: "dinner", title: "Planning a dinner party", turns: [
			{ turn: 1, user: "Starter ideas?", plain: { text: "<think>unclosed", grade: { style: 0, coherence: 1 } }, steered: { text: "Radicchio.", grade: { style: 3, coherence: 2 }, good: true } },
		] },
	],
};

test("parseExamples: every conversation's turns in order, thinking hidden, the grades", () => {
	const ex = parseExamples(CHATS);
	assert.equal(ex.dose, 0.3);
	assert.deepEqual(ex.turns.map((t) => [t.conv, t.n, t.of]), [["A casual chat", 1, 2], ["A casual chat", 2, 2], ["Planning a dinner party", 1, 1]]);
	assert.deepEqual(ex.turns[0].steered, { text: "Status: nominal.", style: 2, sense: 3, good: true, useless: false });
	assert.deepEqual(ex.turns[1].steered.style, null);
	assert.equal(ex.turns[2].plain.text, "(no answer, only thinking)");
	assert.deepEqual(parseExamples({}).turns, []);
});

// -- markdown -------------------------------------------------------------------------------------------------------------

const MD = "Pet = **long-running** process, 5-15 years `uptime`. Evaluate *before* spawn.\n\n## Decision matrix:\n- Daily active time < 1h → fish / hamster / gecko and more words to wrap here\n  - nested bullet 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者 读者\n1. first\n2. second item that wraps over the line for sure ok\n> a quote\n\n```js\nconst x = 1; // a long code line that goes on and on and on\n```\n---\n| a | b |\n|---|---|\n| 1 | 2 |\n👨‍💻 done";

test("mdLines: Pi's markdown at the dock's width; bold, code, headings, nested bullets with hanging indents, fences, wide characters", () => {
	for (const width of [24, 30, 41, 60]) {
		const lines = mdLines(theme, MD, width);
		for (const l of lines) assert.ok(visibleWidth(l) <= width, `${width}: ${JSON.stringify(plain(l))} is ${visibleWidth(l)}`);
		const text = lines.map(plain);
		assert.notEqual(text[0], "");                                                         // no blank line at either end
		assert.notEqual(text.at(-1), "");
		assert.ok(!text.some((l, i) => l === "" && text[i + 1] === ""));                      // one blank between blocks at most
		assert.ok(!text.join("\n").includes("**"));                                           // bold drawn, not shown
		assert.ok(lines.join("").includes("\x1b[1m"));
		assert.ok(text.some((l) => l.includes("Decision matrix")));
		assert.ok(text.some((l) => l.includes("const x = 1;")));                             // the fence's code
		assert.ok(text.some((l) => l.includes("👨‍💻 done")));
		const nested = text.findIndex((l) => l.includes("nested"));
		assert.ok(/^\s{2,}- nested/.test(text[nested]), text[nested]);                       // nested under its parent
		const cont = text[nested + 1];
		assert.ok(cont.startsWith(" ".repeat(text[nested].indexOf("nested"))) && cont.includes("读者"), `hanging indent: ${JSON.stringify(cont)}`);
	}
	assert.ok(mdLines(theme, "plain words", 20, "muted")[0].includes("\x1b[38;5;8m"));     // the colour where nothing else colours it
});

test("noteOf: /steer's last error, else the first line of what it said last", () => {
	assert.deepEqual(noteOf([{ text: "Steering enabled", level: "info" }, { text: "steer: robotic at dose 0.3\nEvery token", level: "info" }]), { text: "steer: robotic at dose 0.3", level: "info" });
	assert.deepEqual(noteOf([{ text: "doubt isn't calibrated: give a dose", level: "error" }]), { text: "doubt isn't calibrated: give a dose", level: "error" });
	assert.equal(noteOf([{ text: "Switch to a metalscope model with /model first", level: "warning" }])?.level, "error");
	assert.equal(noteOf([]), undefined);
	assert.deepEqual(noteOf([{ text: "steering a@0.3 L2 · warning: few traces", level: "caution" }]), { text: "steering a@0.3 L2 · warning: few traces", level: "warning" });
});

// -- the dock ----------------------------------------------------------------------------------------------------------------

const tok = (i: number): TokenEvent => ({ i, id: i, text: " w", p: 0.9, entropy: 0.1, alts: [], n_layers: 48 } as any);
const tick = () => new Promise((r) => setTimeout(r, 0));

/** The Steer page (5) on a dock `rows` tall; `text` draws it 49 wide (the one-panel board's: list over detail). */
function steerDock(over: Record<string, any> = {}, rows = 44) {
	const live: Live = { base: "http://x", clientId: "me", level: "on", connected: true, dense: "", reply: { rid: "r", tokens: [tok(0), tok(1), tok(2)], done: true, prefill: 1 } } as any;
	const applied: string[] = [], asked: string[] = [];
	const host: any = { live, captions: false, close() {}, unfocus() {},
		steerModel: () => "Flash-ds4",
		steering: async () => (asked.push("steering"), parseSteering(STEERING, "Flash-ds4")),
		steerChats: async (name: string) => (asked.push(`chats ${name}`), parseExamples(CHATS)),
		steer: async (args: string) => (applied.push(args), [{ text: `steer: ${args}\nEvery token, thinking included.`, level: "info" }]),
		...over };
	const d = new Dock({ terminal: { rows }, requestRender() {} }, theme, host);
	d.steerPage = true;
	return { d, host, live, applied, asked };
}
const text = (d: Dock, width = 49) => d.render(width).map((l) => plain(l).replace(/^[▌▐] /, "").replace(/\s+$/, ""));
/** The cursor's concept: the title line of the detail under the list. */
const titleOf = (lines: string[]) => lines.find((l) => / · pushes at L\d+|no layer: \/steer refuses it/.test(l));
/** The list's rows. */
const listRows = (lines: string[]) => lines.filter((l) => /^[▸★ ] \S+ +(\d+\/\d+ @[\d.]+|untested)$/.test(l));
const KEY = { up: "\x1b[A", down: "\x1b[B", right: "\x1b[C", left: "\x1b[D", enter: "\r", esc: "\x1b" };

test("dock, steer page: ↑/↓ a concept, ←/→ a dose from the picked one, enter applies through /steer, x off; token keys do nothing", async () => {
	const { d, host, applied } = steerDock();
	await tick();
	d.sel = 1;
	let lines = text(d);
	assert.match(titleOf(lines)!, /^Robotic register · pushes at L24 of 48$/);   // off: no "on at"
	d.handleInput(KEY.enter);                                                                   // the picked dose
	await tick();
	assert.deepEqual(applied, ["robotic 0.3"]);
	assert.ok(text(d).includes("✓ steer: robotic 0.3"));                                        // what /steer said, its first line
	d.handleInput(KEY.right);
	d.handleInput(KEY.right);
	d.handleInput(KEY.right);                                                                   // past the last dose: stays
	assert.ok(text(d).includes("dose 0.6 · 3/18 good · style 3.0 · sense 1.2"));
	d.handleInput(KEY.enter);
	await tick();
	assert.equal(applied.at(-1), "robotic 0.6");
	for (let k = 0; k < 6; k++) d.handleInput(KEY.left);
	assert.ok(text(d).includes("no steering · style 0.1 · sense 3.0"));
	d.handleInput(KEY.enter);                                                                   // dose 0: off
	await tick();
	assert.equal(applied.at(-1), "off");
	d.handleInput(KEY.down);
	d.handleInput("j");
	assert.equal(host.steerAt, "joyful");
	assert.match(titleOf(text(d))!, /Joyful register/);
	d.handleInput("x");
	await tick();
	assert.equal(applied.at(-1), "off");
	for (const k of ["h", "l", "u", "U", "H", "L", "\x1b[1;2C", "[", "]", "e", "s"]) d.handleInput(k);
	assert.equal(d.sel, 1);                                                                      // the token keys do nothing here
	assert.equal(host.lensView, undefined);
	for (let k = 0; k < 20; k++) d.handleInput(KEY.down);
	assert.equal(host.steerAt, "doubt");                                                      // the last, and no further
	// Not chat-tested (and no layer), nothing on: enter sends nothing and says so, once, in the reading's colour; the
	// key line offers no enter.
	assert.ok(!text(d, 104).at(-1)!.includes("enter") && !text(d, 104).at(-1)!.includes("←/→ dose"));
	d.handleInput(KEY.enter);
	await tick();
	assert.equal(applied.length, 4);
	assert.equal(text(d, 104).join("\n").split("not chat-tested").length - 1, 1, text(d, 104).join("\n"));
	assert.ok(d.render(104).some((l) => l.includes(dt.fg("warning", "not chat-tested"))));
	// With a steer on, enter turns it off, as no steering does everywhere; the key line says so.
	host.live.steer = { model: "Flash-ds4", parts: [{ name: "calm", layer: 8, strength: 0.6, source: "profile" }] };
	assert.match(text(d, 104).at(-1)!, /· enter off ·/);
	d.handleInput(KEY.enter);
	await tick();
	assert.equal(applied.at(-1), "off");
	assert.equal(applied.length, 5);
});

test("dock, steer page: /steer's refusal shows on the page in red; one apply at a time", async () => {
	let release: () => void = () => {};
	const { d, applied } = steerDock({
		steer: (args: string) => (applied.push(args), new Promise((r) => (release = () => r([{ text: "robotic can't be steered on Flash: no layer", level: "error" }])))),
	});
	await tick();
	d.sel = 0;
	d.handleInput(KEY.enter);
	assert.deepEqual(applied, ["robotic 0.3"]);
	assert.ok(text(d).includes("steering with robotic 0.3…"));
	release();
	await tick();
	const raw = d.render(49);
	assert.ok(raw.some((l) => l.includes(dt.fg("error", "robotic can't be steered on Flash: no layer"))));
});

test("dock, steer page: x during an apply runs after it, the last asked for wins", async () => {
	const releases: (() => void)[] = [];
	const { d, applied } = steerDock({
		steer: (args: string) => (applied.push(args), new Promise((r) => releases.push(() => r([{ text: `steer: ${args}`, level: "info" }])))),
	});
	await tick();
	d.sel = 0;
	d.handleInput(KEY.enter);                                                                   // robotic 0.3, in flight
	d.handleInput(KEY.down);
	d.handleInput(KEY.enter);                                                                   // melancholic: queued
	d.handleInput("x");                                                                         // then off: replaces it
	assert.deepEqual(applied, ["robotic 0.3"]);
	assert.ok(text(d).includes("then steering off…"));
	releases.shift()!();
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual(applied, ["robotic 0.3", "off"]);                                          // off ran after, last
	releases.shift()!();
	for (let k = 0; k < 3; k++) await tick();
	assert.ok(text(d).includes("✓ steer: off"));
	assert.equal(releases.length, 0);
});

test("dock, steer page: dose 0 with a steer on says enter turns it off", async () => {
	const { d, live } = steerDock();
	await tick();
	d.sel = 0;
	for (let k = 0; k < 6; k++) d.handleInput(KEY.left);
	assert.ok(text(d).includes("what the model does on its own"));
	live.steer = { model: "Flash-ds4", parts: [{ name: "joyful", model: "m", layer: 16, strength: 0.3, source: "profile" }] };
	assert.ok(text(d).includes("enter turns steering off (joyful is on)"));
	const raw = d.render(49);
	assert.ok(raw.some((l) => l.includes(dt.fg("warning", "enter turns steering off (joyful is on)"))));
});

/** A dock whose served model can change (`serve`), model B's concepts held back until `releaseB`. */
function twoModels(over: Record<string, any> = {}) {
	let model = "A", releaseB: () => void = () => {};
	const asks: string[] = [];
	const A = STEERING;
	const B = { concepts: [chatConcept("calm", 6, 0.2, [[0.2, 15], [0.4, 10]]), chatConcept("robotic", 30, 0.4, [[0.4, 13]])] };
	const m = steerDock({
		steerModel: () => model,
		steering: () => (asks.push(`steering ${model}`), model === "A" ? Promise.resolve(parseSteering(A, "A"))
			: new Promise((r) => (releaseB = () => r(parseSteering(B, "B"))))),
		steerChats: async (name: string) => (asks.push(`chats ${model} ${name}`), parseExamples({ ...CHATS, dose: model === "A" ? 0.3 : 0.4 })),
		...over,
	});
	return { ...m, asks, serve: (x: string) => void (model = x), releaseB: () => releaseB() };
}

test("dock, steer page: another served model clears the page; nothing applies while its concepts load, x excepted; no stale dose index", async () => {
	const m = twoModels();
	await tick();
	m.d.sel = 0;
	m.d.render(49);
	for (let k = 0; k < 6; k++) m.d.handleInput(KEY.right);                                 // robotic at 0.6, the last of six doses
	m.serve("B");
	m.d.handleInput(KEY.enter);                                                               // the first key after the switch: asks, sends nothing
	assert.deepEqual(m.applied, []);
	let lines = text(m.d);
	assert.ok(lines.includes("asking for B's concepts…"), lines.join("\n"));
	assert.equal(listRows(lines).length, 0);
	m.d.handleInput(KEY.enter);
	m.d.handleInput(KEY.right);
	assert.deepEqual(m.applied, []);
	m.d.handleInput("x");                                                                     // off still goes
	await tick();
	assert.deepEqual(m.applied, ["off"]);
	m.releaseB();
	await tick();
	lines = text(m.d);
	assert.match(titleOf(lines)!, /Calm register/);
	m.d.handleInput(KEY.down);                                                                // robotic on B: two doses, no index 5
	assert.doesNotThrow(() => m.d.handleInput(KEY.enter));
	await tick();
	assert.deepEqual(m.applied, ["off", "robotic 0.4"]);                                     // B's picked dose
	assert.ok(text(m.d).includes("dose 0.4 · 13/18 good · style 2.4 · sense 1.8"));
});

test("dock, steer page: an apply queued under one model is not sent under another", async () => {
	const releases: (() => void)[] = [];
	const m = twoModels({ steer: (args: string) => new Promise((r) => releases.push(() => (m.applied.push(args), r([{ text: args, level: "info" }])))) });
	await tick();
	m.d.sel = 0;
	m.d.render(49);
	m.d.handleInput(KEY.enter);                                                               // in flight
	m.d.handleInput(KEY.down);
	m.d.handleInput(KEY.enter);                                                               // melancholic, queued
	m.serve("B");
	m.d.render(49);
	releases.shift()!();
	for (let k = 0; k < 4; k++) await tick();
	assert.deepEqual(m.applied, ["robotic 0.3"]);
	assert.equal(releases.length, 0);                                                         // melancholic never sent to B
	assert.ok(text(m.d, 90).some((l) => l.includes("melancholic is not among B's concepts shown: nothing sent")));
});

test("dock, steer page: the test chats are the served model's", async () => {
	const m = twoModels();
	await tick();
	m.d.sel = 0;
	m.d.handleInput(KEY.down);                                                                // robotic is second on B; keep the name
	m.d.handleInput(KEY.up);
	text(m.d);
	await tick();
	assert.ok(text(m.d).some((l) => /^EXAMPLES /.test(l)));
	m.serve("B");
	m.d.render(49);
	m.releaseB();
	await tick();
	text(m.d);
	await tick();
	assert.match(titleOf(text(m.d))!, /^Robotic register/);
	m.d.handleInput(KEY.enter);
	await tick();
	assert.deepEqual(m.applied, ["robotic 0.4"]);
	assert.deepEqual(m.asks.filter((a) => a.startsWith("chats")), ["chats A robotic", "chats B robotic"]);
});

test("dock, steer page: asked for again on entering it (6, stepping in), not every frame; the cursor keeps its concept", async () => {
	let n = 0;
	const later = { concepts: [...STEERING.concepts, chatConcept("aaa-new", 10, 0.3, [[0.3, 18]])] };
	const { d, host } = steerDock({ steering: async () => parseSteering(n++ ? later : STEERING, "Flash-ds4") });
	d.steerPage = false;
	d.sel = 0;
	d.render(49);
	await tick();
	const opened = n;
	d.handleInput("6");
	assert.equal(d.steerPage, true);
	d.render(49);
	await tick();
	assert.equal(n, opened + 1);                                                               // entered from the board
	d.handleInput(KEY.down);
	d.handleInput(KEY.down);
	assert.equal(host.steerAt, "joyful");
	for (let k = 0; k < 5; k++) d.render(49);
	await tick();
	assert.equal(n, opened + 1);                                                               // frames ask nothing
	d.sel = -1;
	d.render(49);
	d.sel = 0;
	d.render(49);                                                                              // stepped in again
	await tick();
	assert.equal(n, opened + 2);
	const lines = text(d);
	assert.match(titleOf(lines)!, /Joyful register/);               // same concept, though the order moved
	assert.ok(lines.some((l) => /^▸ joyful/.test(l)));
	assert.ok(lines.some((l) => /aaa-new +18\/18 @0\.3/.test(l)));
});

test("dock, steer page: a key handler that throws is a note on the page, never out of handleInput", async () => {
	let boom = false;
	const { d } = steerDock({ steerModel: () => { if (boom) { boom = false; throw new Error("no model"); } return "Flash-ds4"; } });
	await tick();
	d.sel = 0;
	boom = true;
	assert.doesNotThrow(() => d.handleInput("j"));
	assert.ok(text(d).includes("the dock: no model"));
});

test("dock, steer page: d asks, y deletes through the host; the concept leaves at once, the list asked for again, the cursor on the next one", async () => {
	let list = STEERING;
	const deleted: string[] = [];
	let asks = 0;
	const { d, host } = steerDock({ steering: async () => (asks++, parseSteering(list, "Flash-ds4")),
		deleteConcept: async (name: string) => {
			deleted.push(name);
			list = { ...STEERING, concepts: STEERING.concepts.filter((c) => c.name !== name) };
			return [{ text: `${name} deleted on Flash: moved to the trash`, level: "info" }];
		} });
	await tick();
	d.sel = 0;
	assert.match(text(d).join(" ").replace(/\s+/g, " "), /x off (· )?d delete/);
	d.handleInput(KEY.down);                                                                    // melancholic
	d.handleInput("d");
	assert.ok(text(d).includes("y deletes melancholic on Flash"));                            // the checkpoint, every copy's
	d.handleInput(KEY.down);                                                                    // any other key cancels
	assert.ok(!text(d).some((l) => l.includes("y deletes")));
	d.handleInput("y");                                                                         // y alone does nothing
	d.handleInput(KEY.up);
	for (let k = 0; k < 5; k++) d.handleInput("d");                                             // key repeat: asks, never deletes
	await tick();
	assert.deepEqual(deleted, []);
	assert.ok(text(d).includes("y deletes melancholic on Flash"));
	const n = asks;
	d.handleInput("y");
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual(deleted, ["melancholic"]);
	assert.equal(asks, n + 1);                                                                  // the list asked for again
	assert.equal(host.steerAt, "joyful");                                                       // the next one
	const lines = text(d);
	assert.ok(lines.some((l) => l.startsWith("✓ melancholic deleted on Flash: moved to")));
	assert.ok(!lines.some((l) => /^[▸★ ] melancholic /.test(l)));
	assert.equal(listRows(lines).length, 7);
	// The last row: the cursor goes to the one before it.
	for (let k = 0; k < 20; k++) d.handleInput(KEY.down);
	d.handleInput("d");
	d.handleInput("y");
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual(deleted, ["melancholic", "doubt"]);
	assert.equal(host.steerAt, "anxious");
	assert.match(titleOf(text(d))!, /Anxious register/);
});

test("dock, steer page: every key but y calls a delete off, the page's early ones too", async () => {
	for (const k of [KEY.esc, "q", "?", "m", "p", "\t", "\x1c", "o", "x", KEY.enter, "c", "n", KEY.left]) {
		const deleted: string[] = [];
		const { d } = steerDock({ deleteConcept: async (name: string) => (deleted.push(name), [{ text: "gone", level: "info" }]) });
		await tick();
		d.sel = 0;
		d.handleInput("d");
		assert.equal((d as any).sp.steer.del, "robotic", JSON.stringify(k));
		d.handleInput(k);
		assert.equal((d as any).sp.steer.del, undefined, JSON.stringify(k));
		assert.ok(!text(d).some((l) => l.includes("y deletes")), JSON.stringify(k));
		d.handleInput("y");
		await tick();
		assert.deepEqual(deleted, [], JSON.stringify(k));
	}
});

test("dock, steer page: d refused while steering changes; enter and x wait for a delete; a failed delete in red; an emptied list never throws", async () => {
	let release: () => void = () => {};
	let gone: () => void = () => {};
	let list = { ...STEERING, concepts: STEERING.concepts.filter((c) => c.name === "robotic") };
	const { d, host, applied } = steerDock({ steering: async () => parseSteering(list, "Flash-ds4"),
		steer: (args: string) => (applied.push(args), new Promise((r) => (release = () => r([{ text: "steer: off", level: "info" }])))),
		deleteConcept: (name: string) => new Promise((r) => (gone = () => {
			list = { ...list, concepts: [] };
			r([{ text: `${name} deleted on Flash: moved to the trash`, level: "info" }]);
		})) });
	await tick();
	d.sel = 0;
	d.handleInput("x");                                                                         // in flight
	d.handleInput("d");
	assert.ok(text(d).includes("steering is changing: d after it"));
	release();
	for (let k = 0; k < 3; k++) await tick();
	d.handleInput("d");
	d.handleInput("y");                                                                         // the delete in flight
	d.handleInput(KEY.enter);                                                                   // not queued against the old list
	d.handleInput("x");
	assert.deepEqual(applied, ["off"]);
	assert.ok(text(d).join(" ").replace(/\s+/g, " ").includes("deleting robotic: again after it"));
	gone();
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual(applied, ["off"]);
	assert.equal(host.steerAt, undefined);
	assert.ok(text(d).some((l) => l.includes("no learned concepts on Flash-ds4")));
	for (const k of ["d", "y", "d", KEY.up, KEY.down, KEY.enter, "x", "c"]) assert.doesNotThrow(() => d.handleInput(k));
	const failing = steerDock({ deleteConcept: async () => [{ text: "a job is running: calibrating robotic (/scope jobs)", level: "error" }] });
	await tick();
	failing.d.sel = 0;
	failing.d.handleInput("d");
	failing.d.handleInput("y");
	for (let k = 0; k < 3; k++) await tick();
	assert.ok(failing.d.render(80).some((l) => l.includes(dt.fg("error", "a job is running: calibrating robotic (/scope jobs)"))));
	assert.equal(failing.host.steerAt, undefined);                                              // nothing moved
	assert.ok(text(failing.d).some((l) => /^▸ robotic /.test(l)));                             // still listed
	const none = steerDock();                                                                   // a host that cannot delete: d does nothing
	await tick();
	none.d.sel = 0;
	none.d.handleInput("d");
	assert.ok(!text(none.d).some((l) => l.includes("y deletes")));
});

test("dock, steer page: a concept without test chats says so where the chats go; one with them, its chats", async () => {
	const { d, asked } = steerDock();
	await tick();
	d.sel = 0;
	for (let k = 0; k < 20; k++) d.handleInput(KEY.down);                                     // doubt: not chat-tested
	assert.match(titleOf(text(d))!, /^Doubt · /);
	assert.ok(text(d).includes("no test chats"));
	assert.ok(!asked.includes("chats doubt"), "none asked for");
	d.handleInput(KEY.up);
	assert.ok(!text(d).includes("no test chats"));                                          // drawn: its chats asked for
	await tick();
	assert.ok(text(d).some((l) => l.startsWith("EXAMPLES")));
});

test("dock, steer page: ? explains it where the keys are; again, the keys", async () => {
	const { d } = steerDock();
	await tick();
	d.sel = 0;
	d.handleInput("?");
	assert.match(text(d).join(" "), /Concepts ranked by the chat test.*\? the keys again/);
	assert.ok(!text(d).join(" ").includes("enter steer"));
	d.handleInput("?");
	assert.ok(text(d).join(" ").includes("enter steer"));
});

test("dock, steer page: a failed request is one line, never a throw", async () => {
	const down = steerDock({ steering: async () => ({ error: "ds4 is not answering at 127.0.0.1:8009: send a message" }) });
	await tick();
	down.d.sel = 0;
	assert.ok(text(down.d, 80).includes("ds4 is not answering at 127.0.0.1:8009: send a message"));
	down.d.handleInput(KEY.enter);
	down.d.handleInput(KEY.down);                                                              // nothing to move or apply
	assert.deepEqual(down.applied, []);
	const thrown = steerDock({ steering: async () => { throw new Error("boom"); } });
	await tick();
	thrown.d.sel = 0;
	assert.ok(text(thrown.d).includes("boom"));
	const noChats = steerDock({ steerChats: async () => ({ error: "ds4 at 127.0.0.1:8009 answered: Not Found" }) });
	await tick();
	noChats.d.sel = 0;
	text(noChats.d);
	await tick();
	assert.ok(text(noChats.d).includes("no saved chat at 0.3"));
	assert.equal(listRows(text(noChats.d)).length, 8);
	// The smallest dock: nothing wider than it, the keys at the bottom.
	const small = steerDock({}, 23);
	await tick();
	small.d.sel = 0;
	const lines = small.d.render(44);
	assert.equal(lines.length, 23);
	for (const l of lines) assert.equal(visibleWidth(l), 44);
	assert.match(plain(lines.slice(-2).join(" ")).replace(/[▌▐]/g, " ").replace(/\s+/g, " "), /6 board\s*$/);
});

// -- the widget's host: the routes, and /steer's own handler ---------------------------------------------------------------

/** createUi, the commands and the shortcuts over one fake server; the dock opened stepped in with a ctx recording what
 * is said. `fresh`: a session before its first reply, the dock not opened. */
function wired(api: (path: string, init?: any) => Promise<any>, servers?: Map<string, any>, fresh = false) {
	const live: any = { base: "http://127.0.0.1:8009", clientId: "pi-x", level: "on", connected: true,
		reply: fresh ? undefined : { rid: "r", tokens: [tok(0)], done: true, prefill: 0 } };
	const notes: [string, string][] = [], statuses: (string | undefined)[] = [], widgets = new Map<string, any>();
	let dock: any;
	const ctx = { hasUI: true, model: { provider: "metalscope-ds4", id: "Flash-ds4" },
		ui: { setWidget: (k: string, f: any) => void (f ? widgets.set(k, f) : widgets.delete(k)), setStatus: (_k: string, s?: string) => statuses.push(s), notify: (m: string, k: string) => notes.push([m, k]),
			custom: (make: any) => new Promise<void>((done) => { dock = make({ requestRender() {}, terminal: { rows: 44, columns: 160 } }, theme, {}, done); }) } };
	const sent: any[][] = [], keys: Record<string, (ctx: any) => Promise<void>> = {};
	const pi = { registerCommand() {}, appendEntry() {}, on() {}, sendUserMessage: (...a: any[]) => sent.push(a),
		registerShortcut: (k: string, o: any) => (keys[k] = o.handler) } as any;
	const ui = createUi({ pi, live, api, servers } as any, join(mkdtempSync(join(tmpdir(), "ms-steer-")), "pi.json"));
	registerCommands({ pi, live, api } as any, ui, {} as any, async () => {});
	registerShortcuts({ pi, live } as any, ui);
	if (!fresh) ui.showDock(ctx, true);
	return { live, notes, statuses, widgets, dock: () => dock as Dock, ui, sent, ctx, key: (k: string) => keys[k](ctx) };
}
// doubt: learned, but /concept learn found no layer where it moves the model (reader null): /steer refuses it.
const CONCEPTS = { model: "Flash-ds4", steer_enabled: true, concepts: ["robotic", "joyful", "melancholic", "tender", "terse", "calm", "anxious", "doubt"].map((name) => ({
	name, title: name, model: "Flash-ds4", reader: name === "doubt" ? null : { layer: name === "robotic" ? 22 : 10, strength: 0.4, concept: 0.5, control: 0.1 },
	split: null, generation: null, steer: { layer: 10, strength: 0.4 }, calibration: null })) };

test("widget: the steer page applies through /steer's own handler: live.steer, the line above the editor and the page's note, as /steer does", async () => {
	const asked: string[] = [];
	const w = wired(async (path: string, init?: any) => {
		asked.push(`${init?.method ?? "GET"} ${path}`);
		if (path.startsWith("/api/steering?")) return STEERING;
		if (path.startsWith("/api/concepts")) return CONCEPTS;
		throw Object.assign(new Error("Not Found"), { status: 404 });
	});
	await tick();
	// The link offer with the list, once a served model (404 here: none).
	assert.deepEqual(asked, ["GET /api/model/link?model=Flash-ds4", "GET /api/steering?model=Flash-ds4&served=Flash-ds4", "GET /api/concepts?model=Flash-ds4"]);
	const d = w.dock();
	d.steerPage = true;
	const lines = text(d);
	await tick();                                                                               // entering the page asked again, and the cursor's test chats
	assert.deepEqual(asked.slice(3).map((a) => a.split("?")[0]), ["GET /api/model/link", "GET /api/steering", "GET /api/concepts", "GET /api/steering/robotic/chats"]);
	assert.equal(listRows(lines).length, 8);
	assert.match(titleOf(lines)!, / · pushes at L22 of 48$/, lines.join("\n"));               // the layer /steer will use
	d.handleInput(KEY.enter);
	for (let k = 0; k < 5; k++) await tick();
	assert.deepEqual(w.live.steer.parts.map((p: any) => [p.name, p.layer, p.strength, p.source]), [["robotic", 22, 0.3, "profile"]]);
	const line = () => w.widgets.get("metalscope-steer")?.({}, theme).render(99).map((l: string) => plain(l));
	assert.deepEqual(line(), [" ◆ steering robotic@0.3 L22 · /steer off"]);                    // the line above the editor, as after /steer
	assert.ok(!/steer/.test(w.statuses.at(-1)!), "the footer: no steer");
	assert.deepEqual(w.notes.at(-1), ["Not calibrated for chat yet: /concept calibrate robotic finds its dose.", "info"]);   // /steer's own words
	assert.deepEqual(d["sp"].steer.note, { text: "steering robotic@0.3 L22", level: "info" });   // the page's note: what is on, not the chat's words
	assert.ok(text(d).some((l) => l.startsWith("✓ steering robotic@0.3 L22")));
	assert.ok(asked.every((a) => a.startsWith("GET ")));                                       // nothing posted: steering was on already
	d.handleInput("x");
	for (let k = 0; k < 3; k++) await tick();
	assert.equal(w.live.steer, undefined);
	assert.deepEqual(w.notes.at(-1), ["steer: off", "info"]);
	assert.deepEqual(d["sp"].steer.note, { text: "steer: off", level: "info" });                  // x: as before
	assert.equal(line(), undefined, "the line gone");
	// x on LENS: the same.
	d.handleInput(KEY.enter);
	for (let k = 0; k < 5; k++) await tick();
	assert.ok(w.live.steer);
	d.handleInput("6");                                                                          // the board, LENS in focus
	assert.equal(d.steerPage, false);
	d.handleInput("x");
	for (let k = 0; k < 3; k++) await tick();
	assert.equal(w.live.steer, undefined);
	assert.deepEqual(w.notes.at(-1), ["steer: off", "info"]);
	assert.deepEqual(d["sp"].steer.note, { text: "steer: off", level: "info" });
});

test("widget: the Steer page's note after enter is what is on, steering LABEL, at any dose; a norms warning after it; never /steer's dose facts", async () => {
	// robotic chat-calibrated at L22: the page's dose 0.3 its default, or (default 0.4) a dose not tried.
	const cal = (dose: number) => ({ version: 4, layer: 22, default_dose: dose, good: 15, turns: 18, grader: { model: "m", thinking: false },
		strengths: [{ dose, good: 15, turns: 18, attempts: 18, failed: { tool: 0, thinking: 0, short: 0 }, coherence: 2.8 }] });
	const listWith = (dose: number, norms_note?: string) => ({ ...CONCEPTS, ...(norms_note ? { norms_note } : {}), concepts: CONCEPTS.concepts.map((c) =>
		(c.name === "robotic" ? { ...c, calibration: cal(dose), steer: { layer: 22, strength: dose, basis: "chat" } } : c)) });
	for (const [dose, norms, chat, note, level] of [
		[0.3, undefined, undefined, "steering robotic@0.3 L22", "info"],                                          // the default dose: nothing in the chat
		[0.4, undefined, "Dose 0.3 was not tried in chat.\nDefault dose 0.4: 15 of 18 good (the most)", "steering robotic@0.3 L22", "info"],
		[0.3, "few traces", "warning: few traces", "steering robotic@0.3 L22 · warning: few traces", "warning"],   // a warning: its colour, no ✓
	] as [number, string | undefined, string | undefined, string, string][]) {
		const w = wired(async (path: string) => {
			if (path.startsWith("/api/steering?")) return STEERING;
			if (path.startsWith("/api/concepts")) return listWith(dose, norms);
			if (path.startsWith("/api/steering/")) return CHATS;
			throw Object.assign(new Error("Not Found"), { status: 404 });
		});
		await tick();
		const d = w.dock();
		d.steerPage = true;
		text(d);
		await tick();
		d.handleInput(KEY.enter);
		for (let k = 0; k < 5; k++) await tick();
		assert.deepEqual(w.live.steer.parts.map((p: any) => [p.name, p.strength]), [["robotic", 0.3]]);
		assert.deepEqual(w.notes.map(([m]) => m), chat ? [chat] : [], `${dose} ${norms}`);
		assert.deepEqual(d["sp"].steer.note, { text: note, level }, `${dose} ${norms}`);
		const drawn = d.render(160).find((l) => plain(l).includes("steering robotic"))!;
		if (level === "info") assert.ok(drawn.includes(dt.fg("success", "✓ steering robotic@0.3 L22")), drawn);
		else assert.ok(drawn.includes(dt.fg("warning", "steering robotic@0.3 L22 · warning: few traces")) && !plain(drawn).includes("✓"), drawn);
	}
});

test("widget: a steer on, the Steer page's enter failing (the concept list's 409, steering not enabled): its error is the note, never steering OLD", async () => {
	let fail = "";
	const w = wired(async (path: string, init?: any) => {
		if (path.startsWith("/api/steering?")) return STEERING;
		if (path.startsWith("/api/steering/")) return CHATS;
		if (path.startsWith("/api/concepts")) {
			if (fail === "409") throw Object.assign(new Error("model 'Flash-ds4' is not loaded"), { status: 409 });
			return { ...CONCEPTS, steer_enabled: fail !== "enable" };
		}
		if (path === "/api/steer" && init?.method === "POST") throw Object.assign(new Error("HTTP 500"), { status: 500 });
		throw Object.assign(new Error("Not Found"), { status: 404 });
	});
	await tick();
	const d = w.dock();
	d.steerPage = true;
	text(d);
	await tick();
	d.handleInput(KEY.enter);
	for (let k = 0; k < 5; k++) await tick();
	const on = w.live.steer;
	assert.deepEqual(d["sp"].steer.note, { text: "steering robotic@0.3 L22", level: "info" });
	for (const [mode, said] of [["409", /model 'Flash-ds4' is not loaded/], ["enable", /^Could not enable steering on the server: /]] as [string, RegExp][]) {
		fail = mode;
		d.handleInput(KEY.enter);
		for (let k = 0; k < 5; k++) await tick();
		assert.equal(w.live.steer, on, mode);                                                      // the steer on stays
		assert.equal(d["sp"].steer.note.level, "error", mode);
		assert.match(d["sp"].steer.note.text, said, mode);
	}
	// An error said wins over a steer that changed meanwhile.
	const host = d["host"], real = w.ui.steer!;
	w.ui.steer = async (_args: string, ctx: any) => {
		w.live.steer = { ...on };
		ctx.ui.notify("refused", "error");
	};
	assert.deepEqual(await host.steer!("robotic 0.3"), [{ text: "refused", level: "error" }]);
	w.ui.steer = real;
});

const steerApi = async (path: string) => {
	if (path.startsWith("/api/steering?")) return STEERING;
	if (path.startsWith("/api/concepts")) return CONCEPTS;
	if (path.startsWith("/api/steering/")) return CHATS;
	throw Object.assign(new Error("Not Found"), { status: 404 });
};

test("widget: before the first reply, ctrl+\\ twice steps in; 6 the Steer page, enter steers, esc back to Pi (the page kept), 6 the board; the token keys do nothing", async () => {
	const w = wired(steerApi, undefined, true);
	await w.key("ctrl+\\");                                                                      // the dock, as the setting opens it
	assert.equal(w.dock().stepped, false);
	await w.key("ctrl+\\");                                                                      // stepped in, no reply needed
	const d = w.dock();
	assert.ok(w.ui.docked() && d.stepped);
	await tick();
	assert.equal(text(d, 104).at(-1), "6 steer · m mini · esc back");
	for (const k of ["h", "l", KEY.left, KEY.right, KEY.up, KEY.down, "u", "U", "H", "L", "f", "\x1b[H", "\x1b[F", "\x1b[1;2C", "\x1b[5~", "s", "[", "]", "g", "-", "+", "x", KEY.enter, "?", "z", "\t", "1"]) {
		d.handleInput(k);
		assert.equal(d["sp"].steer.note, undefined, `${JSON.stringify(k)} threw`);
	}
	assert.deepEqual([d.stepped, d.sel, d.steerPage, d.full], [true, -1, false, false]);
	const host = d["host"];
	assert.deepEqual([host.captions, host.panel], [false, undefined]);                         // nothing explained or moved
	assert.equal(text(d, 104).at(-1), "6 steer · m mini · esc back");
	assert.equal(w.live.steer, undefined);
	d.handleInput("6");
	assert.match(text(d, 104)[0], /◆ metalscope {2}on · waiting for a reply +no steering$/);
	const lines = text(d);
	await tick();
	assert.equal(listRows(lines).length, 8);
	assert.match(titleOf(lines)!, / · pushes at L22$/, lines.join("\n"));                       // no reply: the model's depth not known
	assert.ok(text(d, 104).some((l) => l.includes("good")), "the test chats, from the server");
	d.handleInput(KEY.enter);
	for (let k = 0; k < 5; k++) await tick();
	assert.deepEqual(w.live.steer.parts.map((p: any) => [p.name, p.layer, p.strength]), [["robotic", 22, 0.3]]);
	assert.deepEqual(d["sp"].steer.note, { text: "steering robotic@0.3 L22", level: "info" });
	d.handleInput(KEY.esc);                                                                      // back to Pi, the page kept
	assert.deepEqual([d.steerPage, d.stepped, w.ui.docked()], [true, false, true]);
	assert.ok(text(d, 104).some((l) => l.includes("good")), "the page as it was");
	assert.equal(text(d, 104).at(-1), "ctrl+\\ steps in: ↑/↓ concept, 6 board · /scope help dock");
	await w.key("ctrl+\\");                                                                      // stepped in again: the same page
	assert.deepEqual([d.steerPage, d.stepped], [true, true]);
	d.handleInput("6");                                                                          // the board
	assert.deepEqual([d.steerPage, d.stepped], [false, true]);
	assert.match(text(d, 104)[0], /◆ trace/);
	d.handleInput(KEY.esc);                                                                      // back to Pi, the dock open
	assert.deepEqual([d.stepped, w.ui.docked()], [false, true]);
	assert.equal(text(d, 104).at(-1), "ctrl+\\ steps in: 6 steer · /scope help dock");
});

test("widget: before the first reply, alt+i and /scope dock step in too, alt+i with no warning", async () => {
	const a = wired(steerApi, undefined, true);
	await a.key("alt+i");
	assert.ok(a.dock().stepped);
	assert.deepEqual(a.notes, []);
	a.dock().handleInput(KEY.esc);                                                               // opened stepped in: esc closes it
	const b = wired(steerApi, undefined, true);
	dockStep(b.ui, b.ctx);
	dockStep(b.ui, b.ctx);
	assert.ok(b.dock().stepped);
	b.dock().handleInput("6");
	assert.ok(b.dock().steerPage);
});

test("widget: /steer's own refusal (no layer to steer at) is the page's note, in red", async () => {
	const noLayer = { ...CONCEPTS, concepts: CONCEPTS.concepts.map((c) => (c.name === "robotic" ? { ...c, reader: null, steer: null } : c)) };
	const w = wired(async (path: string) => {
		if (path.startsWith("/api/steering?")) return STEERING;
		if (path.startsWith("/api/concepts")) return noLayer;
		throw Object.assign(new Error("Not Found"), { status: 404 });
	});
	await tick();
	const d = w.dock();
	d.steerPage = true;
	text(d);
	for (let k = 0; k < 3; k++) await tick();
	d.handleInput(KEY.enter);                                                                   // robotic, chat-tested: sent
	for (let k = 0; k < 5; k++) await tick();
	assert.equal(w.live.steer, undefined);
	const [said, level] = w.notes.at(-1)!;
	assert.equal(level, "error");
	const raw = d.render(104);
	const red = dt.fg("error", "\0").split("\0")[0];
	assert.ok(raw.some((l) => l.includes(red + said.slice(0, 40))), said);                    // its start, in red
});

test("widget: d, y turns off a steer that uses the concept through /steer, posts the delete for the served model", async () => {
	const asked: string[] = [];
	let gone = "";
	const w = wired(async (path: string, init?: any) => {
		asked.push(`${init?.method ?? "GET"} ${path}${init?.body ? ` ${init.body}` : ""}`);
		if (path === "/api/concept/delete") return (gone = JSON.parse(init.body).name, { trash: "/t", checkpoint: "Flash" });
		if (path.startsWith("/api/steering?")) return { ...STEERING, concepts: STEERING.concepts.filter((c) => c.name !== gone) };
		if (path.startsWith("/api/concepts")) return CONCEPTS;
		throw Object.assign(new Error("Not Found"), { status: 404 });
	});
	await tick();
	const d = w.dock();
	d.steerPage = true;
	text(d);
	await tick();
	d.handleInput(KEY.enter);                                                                   // robotic on
	for (let k = 0; k < 5; k++) await tick();
	assert.equal(w.live.steer.parts[0].name, "robotic");
	d.handleInput("d");
	assert.ok(text(d).includes("y deletes robotic on Flash"));
	d.handleInput("y");
	for (let k = 0; k < 6; k++) await tick();
	assert.equal(w.live.steer, undefined);                                                      // turned off first, as /steer off does
	assert.deepEqual(w.notes.slice(-1), [["steer: off", "info"]]);
	assert.ok(asked.includes('POST /api/concept/delete {"name":"robotic","model":"Flash-ds4"}'));
	assert.ok(text(d).some((l) => l.startsWith("✓ robotic deleted on Flash: moved to")));
	assert.ok(!text(d).some((l) => /^[▸★ ] robotic /.test(l)));
	// Not steering with it: no /steer at all; the server's refusal is the note.
	const before = w.notes.length;
	const r = await (d as any).host.deleteConcept("calm");
	assert.deepEqual(r, [{ text: "calm deleted on Flash: moved to the trash", level: "info" }]);
	assert.equal(w.notes.length, before);
});

test("widget: a delete that took a description no model has, and a 404's detail, said as the server says them", async () => {
	let answer: any = { definition: true, checkpoint: "Flash" };
	const w = wired(async (path: string) => {
		if (path === "/api/concept/delete") {
			if (answer instanceof Error) throw answer;
			return answer;
		}
		if (path.startsWith("/api/steering?")) return STEERING;
		if (path.startsWith("/api/concepts")) return CONCEPTS;
		throw Object.assign(new Error("Not Found"), { status: 404 });
	});
	await tick();
	const host = w.dock()["host"];
	assert.deepEqual(await host.deleteConcept!("half"), [{ text: "half's description deleted", level: "info" }]);
	answer = Object.assign(new Error("half has nothing on Flash (it is on A, B)"), { status: 404 });
	assert.deepEqual(await host.deleteConcept!("half"), [{ text: "half has nothing on Flash (it is on A, B)", level: "error" }]);
});

test("widget: a LENS edit, where the server can (edit_at): its steer built as /steer builds it, not turned on, then /steer again as a command; elsewhere a line says why", async () => {
	let caps = ["token_prompts", "edit_at"];
	const words: any[] = [];
	const w = wired(async (path: string, init?: any) => {
		if (path === "/api/status") return { engine: "ds4", caps };
		if (path === "/api/steer/words") {
			words.push(JSON.parse(init.body));
			return { name: "cat", model: "Flash-ds4", band: [18, 44] };
		}
		return CONCEPTS;
	});
	await tick();
	const host = w.dock()["host"];
	// A steer on: its line above the editor stays as it is through the edit (the edit's steer never goes through live.steer).
	const on = w.live.steer = { model: "Flash-ds4", parts: [{ name: "calm", model: "Flash-ds4", layer: 10, strength: 0.4, source: "profile" }] };
	w.ui.updateUi(w.ctx);
	const line = w.widgets.get("metalscope-steer");
	assert.ok(line);
	const said = await host.edit!("-cat", { cat: { id: 107, text: "cat" } }, { rid: "r", at: 3, word: "cat" });
	assert.ok(!said.some((s) => s.level === "error"), JSON.stringify(said));
	assert.deepEqual(words.map((b) => b.ids), [[107]]);                                         // the token shown, by its id
	assert.deepEqual(w.live.edit, { rid: "r", at: 3, word: "cat", steer: "remove~cat/18-44", model: "Flash-ds4" });
	assert.equal(w.live.steer, on);
	assert.equal(w.widgets.get("metalscope-steer"), line, "the line not set again");
	assert.deepEqual(w.sent, [["/steer again", { expandPromptTemplates: true }]]);
	w.live.edit = undefined;
	caps = ["token_prompts"];                                                                    // a server without edit_at
	assert.deepEqual(await host.edit!("-cat", { cat: { id: 107, text: "cat" } }, { rid: "r", at: 3, word: "cat" }), [{ text: LABELS.editHere, level: "error" }]);
	assert.deepEqual([w.live.edit, w.sent.length], [undefined, 1]);
	// No steer on: the edit turns none on and sets no line.
	w.live.steer = undefined;
	w.ui.updateUi(w.ctx);
	assert.ok(!w.widgets.has("metalscope-steer"));
	caps = ["token_prompts", "edit_at"];
	const again = await host.edit!("-cat", { cat: { id: 107, text: "cat" } }, { rid: "r", at: 3, word: "cat" });
	assert.ok(!again.some((s) => s.level === "error"), JSON.stringify(again));
	assert.deepEqual([w.live.edit?.steer, w.live.steer, w.sent.length], ["remove~cat/18-44", undefined, 2]);
	assert.ok(!w.widgets.has("metalscope-steer"));
});

test("widget, end to end: a LENS edit of a lens word that is no name (初步) goes by its id, its headers ASCII; said as itself", async () => {
	const words: any[] = [], posts: string[] = [];
	const w = wired(async (path: string, init?: any) => {
		if (init?.method === "POST") posts.push(path);
		if (path === "/api/status") return { engine: "omlx", caps: ["edit_at"] };
		if (path === "/api/steer/words") {
			const b = JSON.parse(init.body);
			words.push(b);
			return { name: b.name, model: "Flash-ds4", band: [15, 36], cut: [] };
		}
		// The lens steps, asked for the reply: where the edit acts (the band the words route gives too).
		if (path.startsWith("/api/lens/layers")) return { model: "Flash-ds4", at: 0, steps: [{ n: 1, layers: [31], tps: null, replies: 0 }], edit_band: [15, 36] };
		return CONCEPTS;
	});
	await tick();
	// The reply's token 0 is 初步 (one token, id 104495), the lens's top word at L31 too.
	w.live.reply = { rid: "r", done: true, prefill: 0, tokens: [{ ...tok(0), id: 104495, text: "初步", lens: [[31, 104495, "初步"]], lens_by: "jlens" }] };
	const d = w.dock();
	d["host"].panel = "LENS";
	d.sel = 0;
	text(d, 60);
	for (let k = 0; k < 3; k++) await tick();
	d.handleInput("\r");
	for (let k = 0; k < 3; k++) await tick();
	for (const c of "vague") d.handleInput(c);
	assert.ok(text(d, 60).some((l) => /▸初步 → vague {2}at L26 /.test(l)), text(d, 60).join("\n"));
	assert.deepEqual(posts, []);                                                                // nothing written until enter
	d.handleInput("\r");
	for (let k = 0; k < 5; k++) await tick();
	assert.deepEqual(words, [{ model: "Flash-ds4", name: "tok104495", words: ["初步"], ids: [104495] }, { model: "Flash-ds4", name: "vague", words: ["vague"] }]);
	assert.deepEqual(w.live.edit, { rid: "r", at: 0, word: "初步", want: "vague", steer: "remove~tok104495/15-36+vague@0.7/26", model: "Flash-ds4" });
	assert.deepEqual(w.sent, [["/steer again", { expandPromptTemplates: true }]]);
	// /steer again's request: the edit in an ASCII header.
	w.live.edit.prompt = "hi";
	editStarts(w.live, "hi");
	const h = applyLiveHeaders({} as Record<string, string | undefined>, w.live, "Flash-ds4");
	assert.equal(h["X-Scope-Edit"], "r@0 remove~tok104495/15-36+vague@0.7/26");
	assert.ok(Object.values(h).every((v) => /^[\x20-\x7e]*$/.test(v ?? "")), JSON.stringify(h));
	// What came of it, in the real text.
	onEvent(w.live, { type: "start", rid: "e1", model: "Flash-ds4", edit: { of: "r", at: 0 } });
	onEvent(w.live, { type: "token", rid: "e1", ...tok(0), id: 900, text: " vague" });
	assert.ok(text(d, 60).some((l) => l.includes('edit: "初步" became " vague", as asked')), text(d, 60).join("\n"));
});

test("widget: the dock's flags are the selected model's on its server; a LENS edit follows the model's edit (the server's now), else the server's caps", async () => {
	const status: any = { engine: "omlx", caps: ["edit_at"], chat: { "Flash-ds4": { reasoning: true, tested: false, edit: false } } };
	const servers = new Map([["metalscope-ds4", { base: "http://127.0.0.1:8009", key: "", status: { chat: { "Flash-ds4": { reasoning: true, tested: false, edit: true } } } }]]);
	const w = wired(async (path: string) => (path === "/api/status" ? status : CONCEPTS), servers);
	await tick();
	const host = w.dock()["host"];
	assert.deepEqual(host.flags!(), { reasoning: true, tested: false, edit: true });              // as /model listed it
	const edit = () => host.edit!("-cat", { cat: { id: 107, text: "cat" } }, { rid: "r", at: 3, word: "cat" });
	assert.deepEqual(await edit(), [{ text: LABELS.editHere, level: "error" }]);                 // loaded since: oMLX refuses it
	assert.equal(host.flags!().edit, false);                                                     // and the key goes
	status.chat = {};
	status.caps = [];                                                                            // the model's edit not known: the caps
	assert.deepEqual(await edit(), [{ text: LABELS.editHere, level: "error" }]);
	assert.deepEqual([w.live.edit, w.sent.length], [undefined, 0]);
	assert.deepEqual(wired(async () => CONCEPTS).dock()["host"].flags!(), {});                   // no server entry: no flags
});

test("widget: the steer page's requests, failed: one line each", async () => {
	const w = wired(async () => { throw new TypeError("fetch failed"); });
	await tick();
	const d = w.dock();
	d.steerPage = true;
	const lines = text(d);
	assert.ok(lines.some((l) => /^ds4 is not answering at 127\.0\.0\.1:8009/.test(l)), lines.join("\n"));
	const nf = wired(async (path: string) => {
		if (path.startsWith("/api/steering?")) return STEERING;
		if (path.startsWith("/api/concepts")) throw new TypeError("fetch failed");               // the layers fall back to the tab's
		throw Object.assign(new Error("Not Found"), { status: 404 });
	});
	await tick();
	const d2 = nf.dock();
	d2.steerPage = true;
	assert.match(titleOf(text(d2))!, / · pushes at L24 of 48$/);
	for (let k = 0; k < 3; k++) await tick();
	assert.ok(text(d2).includes("no saved chat at 0.3"));                                       // the test chats: Not Found
});

test("dock, steer page: the count line under the list, every concept chat-tested or not, at any width", async () => {
	const tested = { ...STEERING, concepts: STEERING.concepts.filter((c: any) => c.name !== "doubt") };
	const all = steerDock({ steering: async () => parseSteering(tested, "Flash-ds4") });
	const some = steerDock();
	await tick();
	for (const w of [44, 49, 70, 104]) {
		assert.ok(text(all.d, w).some((l) => /^ {2}7 chat-tested\b/.test(l)) && !text(all.d, w).some((l) => /untested/.test(l)), `${w}`);
		assert.ok(text(some.d, w).some((l) => /^ {2}1 not calibrated\b/.test(l)) && !text(some.d, w).some((l) => /experimental/.test(l)), `${w}`);
	}
});

test("dock, steer page: a reading or a note past two lines ends its second with …, never dropped silently", async () => {
	const long = "robotic can't be steered on Flash-ds4: /concept learn found no layer where it moves the model. /concept robotic has what was measured; /concept lists the others.";
	const { d } = steerDock({ steer: async () => [{ text: long, level: "error" }] });
	await tick();
	d.sel = 0;
	d.handleInput(KEY.enter);
	for (let k = 0; k < 3; k++) await tick();
	const right = text(d, 104).map((l) => l.split("│")[1]?.trim() ?? "");
	const at = right.findIndex((l) => l.startsWith("robotic can't be steered"));
	assert.ok(at >= 0 && right[at + 1].endsWith("…") && !right.join(" ").includes("lists the others"), right.join("\n"));
	// A stale concept's reading (with a layer, the commands) at a width where it takes three lines.
	const stale = { ...STEERING, concepts: STEERING.concepts.map((c: any) => (c.name === "doubt" ? { ...c, name: "uncooperative-doubt", chat_stale: true } : c)) };
	const s = steerDock({ steering: async () => parseSteering(stale, "Flash-ds4", { "uncooperative-doubt": 8 }) });
	await tick();
	for (let k = 0; k < 20; k++) s.d.handleInput(KEY.down);
	const lines = text(s.d, 88).map((l) => l.split("│")[1]?.trim() ?? "");
	const i = lines.findIndex((l) => l.startsWith("chat test out of date"));
	assert.ok(i >= 0 && lines[i + 1].endsWith("…"), lines.join("\n"));
});

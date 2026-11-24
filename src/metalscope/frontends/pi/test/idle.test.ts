import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { Dock } from "../dock.ts";
import { COLOURS } from "../draw.ts";
import { IDLE_FRAMES, IDLE_H, IDLE_LINES, IDLE_MS, IDLE_PAL, IDLE_W, idleFrame, idleLines } from "../idle.ts";
import type { Live, TokenEvent } from "../state.ts";
// The approved scene's frames as the Python reference drew them (w, h, pal, frames: 84 frames of 28 rows of letters).
import ref from "./fixtures/idle-seq.json" with { type: "json" };

const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const piTheme = { fg: (_c: string, s: string) => s, bg: (_c: string, s: string) => s, bold: (s: string) => s, getColorMode: () => "truecolor" };

test("idle: every frame equals the approved reference pixel for pixel; its palette, the lens violet and the dock's background", () => {
	assert.equal(ref.w, IDLE_W);
	assert.equal(ref.h, IDLE_H);
	assert.equal(ref.frames.length, IDLE_FRAMES);
	for (let i = 0; i < IDLE_FRAMES; i++) assert.deepEqual(idleFrame(i), ref.frames[i], `frame ${i}`);
	assert.deepEqual(idleFrame(IDLE_FRAMES), ref.frames[0], "it loops");
	const pal = ref.pal as Record<string, number[]>;
	assert.deepEqual(Object.keys(IDLE_PAL).sort(), Object.keys(pal).sort());
	for (const [ch, rgb] of Object.entries(pal)) assert.deepEqual(IDLE_PAL[ch] ?? COLOURS.bg, rgb, ch);
	assert.equal(IDLE_PAL["."], null, "the background is the dock's own (a blank)");
	assert.equal(IDLE_PAL.v, COLOURS.lens);
});

test("idle: half blocks, 11 lines of 36, in 24-bit colour or the 256-colour fallback", () => {
	for (const mode of ["truecolor", "256color"] as const) {
		const lines = idleLines(34, mode);
		assert.equal(lines.length, IDLE_LINES);
		for (const l of lines) assert.equal(visibleWidth(l), IDLE_W);
		assert.ok(lines.some((l) => l.includes("▀")));
		assert.equal(lines.some((l) => l.includes("38;2;")), mode === "truecolor");
	}
});

const text = (lines: string[]) => lines.map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd());
/** Where frame `i` is drawn whole in a dock `width` wide, centred: the line its first row is on, else -1. */
function sceneAt(lines: string[], i: number, width: number): number {
	if (width - 3 < IDLE_W) return -1;
	const t = text(lines), lead = " ".repeat(Math.floor((width - 3 - IDLE_W) / 2));
	const want = JSON.stringify(idleLines(i, "truecolor").map((l) => (lead + plain(l)).trimEnd()));
	for (let k = 0; k + IDLE_LINES <= t.length; k++) if (JSON.stringify(t.slice(k, k + IDLE_LINES)) === want) return k;
	return -1;
}
/** Whether any frame is drawn. */
const scene = (lines: string[], width = 104) => Array.from({ length: IDLE_FRAMES }, (_, i) => i).some((i) => sceneAt(lines, i, width) >= 0);
const ev = (i: number): TokenEvent => ({ i, id: 1, text: " Hi", p: 0.9, entropy: 0.1, alts: [], n_layers: 24 });

function idleDock(rows: number, live: Partial<Live> = {}) {
	const tui = { terminal: { rows }, renders: 0, requestRender() { tui.renders++; } };
	const host = { live: { base: "http://h", clientId: "me", level: "on", connected: true, ...live } as Live, captions: false, close() {}, unfocus() {} };
	return { d: new Dock(tui, piTheme, host), tui, live: host.live };
}

test("idle: the scene at 104x75 and 104x51, centred under the header and its rule, the status under it, the dock exactly its size", () => {
	for (const rows of [75, 51]) {
		const { d } = idleDock(rows);
		const out = d.render(104);
		assert.equal(out.length, rows);
		for (const l of out) assert.equal(visibleWidth(l), 104, `${rows}: ${plain(l)}`);
		const t = text(out);
		assert.equal(t[0], "◆ trace", "the header alone: the status is under the scene");
		assert.match(t[1], /^─{101}$/);
		// Frame 0, centred across; the caption right under its 11 lines; as many lines above (under the rule) as below
		// the caption (over the key line), give or take one.
		const at = sceneAt(out, 0, 104);
		assert.ok(at > 1, "frame 0 drawn, centred");
		const capAt = t.findIndex((l) => l.trim() === "waiting for a reply");
		assert.equal(capAt, at + IDLE_LINES, "the caption under the scene");
		const top = at - 2, bottom = rows - 1 - (capAt + 1);
		assert.ok(Math.abs(top - bottom) <= 1, `${rows}: ${top} above, ${bottom} below`);
		assert.ok(t.slice(2, capAt - IDLE_LINES).every((l) => l === ""), "blank above the scene");
		assert.equal(t.filter((l) => /waiting for a reply/.test(l)).length, 1, "the status once");
		assert.match(t[rows - 1], /ctrl\+\\ steps in: 6 steer · \/scope help dock/);
		// Horizontally: the caption's centre and the scene's within a column of the middle.
		const lead = (l: string) => l.length - l.trimStart().length;
		assert.ok(Math.abs(lead(t[capAt]) * 2 + "waiting for a reply".length - 101) <= 1);
		d.dispose();
	}
});

test("idle: while the prompt is read (no token yet) the scene stays, saying so; the first token, the reply instead", () => {
	const { d, live } = idleDock(75, { reply: { rid: "r", tokens: [], done: false, prefill: 812 } });
	assert.ok(scene(d.render(104)));
	assert.ok(text(d.render(104)).some((l) => l.trim() === "reading the prompt · 812 tok"));
	live.reply!.tokens.push(ev(0));
	const out = d.render(104);
	assert.ok(!scene(out), "the board at once");
	assert.equal(d["idle"], undefined, "its timer stopped");
	// A reply shown before: the next one's prompt keeps it drawn, no scene.
	live.prev = live.reply;
	live.reply = { rid: "r2", tokens: [], done: false, prefill: 0 };
	assert.ok(!scene(d.render(104)));
	assert.ok(!scene(d.render(80), 80), "on the pages too");
	d.dispose();
});

test("idle: not when the dock is too small (its one line, or the mini dock, instead), the server not answering or tracing off: what it says today", () => {
	const small = /dock too small: needs/;
	for (const [rows, width, shown, line] of [[23, 44, true, false], [22, 44, false, false], [23, 43, false, true]] as const) {
		const { d } = idleDock(rows);
		const out = d.render(width);
		assert.equal(scene(out, width), shown, `${width}x${rows}`);
		assert.equal(small.test(text(out).join(" ")), line);
		assert.equal(out.length, rows);
		for (const l of out) assert.equal(visibleWidth(l), width);
		d.dispose();
	}
	const gone = idleDock(75, { connected: false });
	const out = text(gone.d.render(104)).join(" ");
	assert.ok(!scene(gone.d.render(104)));
	assert.match(out, /not answering at h/);
	assert.equal(gone.d["idle"], undefined);
	const off = idleDock(75, { level: "off" });
	assert.ok(!scene(off.d.render(104)));
	assert.match(text(off.d.render(104))[0], /◆ trace {2}off · waiting for a reply/);
	assert.equal(off.d["idle"], undefined);
	// The mini dock keeps its own line.
	const mini = idleDock(75);
	(mini.d as any).host.mini = true;
	assert.ok(!scene(mini.d.render(40), 40));
});

test("idle: a frame every IDLE_MS from frame 0, looping; stopped by the first token, by not being drawn (hidden) and by dispose", () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const { d, tui, live } = idleDock(75);
		const shows = (i: number) => sceneAt(d.render(104), i, 104) >= 0;
		assert.ok(shows(0));
		for (let k = 1; k <= IDLE_FRAMES + 2; k++) {
			mock.timers.tick(IDLE_MS);
			assert.equal(tui.renders, k, "a render asked for each tick");
			assert.equal(d["idle"]!.i, k % IDLE_FRAMES);
			assert.ok(shows(k % IDLE_FRAMES), `frame ${k % IDLE_FRAMES}`);
		}
		// Not drawn for a tick (Pi hides the dock): the next tick stops it; drawn again, from frame 0.
		mock.timers.tick(IDLE_MS);
		const asked = tui.renders;
		mock.timers.tick(IDLE_MS * 3);
		assert.equal(d["idle"], undefined);
		assert.equal(tui.renders, asked, "no renders asked once stopped");
		assert.ok(shows(0));
		assert.equal(d["idle"]!.i, 0);
		mock.timers.tick(IDLE_MS * 5);
		d.render(104);
		// The first token: stopped, no more renders asked.
		live.reply = { rid: "r", tokens: [ev(0)], done: false, prefill: 0 };
		d.render(104);
		assert.equal(d["idle"], undefined);
		const n = tui.renders;
		mock.timers.tick(IDLE_MS * 10);
		assert.equal(tui.renders, n);
		// Dispose while it runs: stopped.
		const other = idleDock(51);
		other.d.render(104);
		assert.ok(other.d["idle"]);
		other.d.dispose();
		assert.equal(other.d["idle"], undefined);
		mock.timers.tick(IDLE_MS * 4);
		assert.equal(other.tui.renders, 0);
	} finally {
		mock.timers.reset();
	}
});

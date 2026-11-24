import assert from "node:assert/strict";
import { test } from "node:test";
import { Dock, jumpTo } from "../dock.ts";
import { dockTheme, lensSettle, offTop } from "../draw.ts";
import type { Live, TokenEvent } from "../state.ts";

const toks = ["The", " cat", " sat", ".", " It", " purred", " softly", ".", "\n", "Then", " it", " slept", "!", " Done"].map((text) => ({ text }));
// sentence starts: 0 ("The"), 4 (" It"), 9 ("Then", after the newline token 8), 13 (" Done", after "!")

test("jumpTo: fixed steps clamp to the ends", () => {
	assert.equal(jumpTo(toks, 0, 1, "step", 10), 10);
	assert.equal(jumpTo(toks, 10, 1, "step", 10), 13);
	assert.equal(jumpTo(toks, 3, -1, "step", 10), 0);
	assert.equal(jumpTo([], 0, 1, "step"), 0);
});

test("jumpTo: sentences forward and back", () => {
	assert.equal(jumpTo(toks, 0, 1, "sentence"), 4);
	assert.equal(jumpTo(toks, 4, 1, "sentence"), 9);
	assert.equal(jumpTo(toks, 9, 1, "sentence"), 13);
	assert.equal(jumpTo(toks, 13, 1, "sentence"), 13);            // the last sentence: stays on the last token
	assert.equal(jumpTo(toks, 6, -1, "sentence"), 4);             // mid-sentence: to its start
	assert.equal(jumpTo(toks, 4, -1, "sentence"), 0);             // at a start: to the previous start
	assert.equal(jumpTo(toks, 13, -1, "sentence"), 9);
	assert.equal(jumpTo(toks, 0, -1, "sentence"), 0);
});

// A theme that marks colours; ANSI-shaped, so widths stay right.
const COLOURS: Record<string, number> = { success: 2, muted: 8, dim: 7, accent: 6, warning: 3, error: 1 };
const theme = {
	fg: (c: string, s: string) => `\x1b[38;5;${COLOURS[c] ?? 15}m${s}\x1b[39m`,
	bg: (_c: string, s: string) => `\x1b[48;5;236m${s}\x1b[49m`,
	bold: (s: string) => `\x1b[1m${s}\x1b[22m`,
};
const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");

// The token being chosen is " Paris" (id 42). The lens has it at L8, loses it at L12 and holds it from L16 on: settle L16.
const lens: [number, number, string][] = [[0, 7, " the"], [4, 9, " a"], [8, 42, " Paris"], [12, 13, " France"], [16, 42, " Paris"], [20, 42, " Paris"]];
const ev = (i: number, extra: Partial<TokenEvent> = {}): TokenEvent =>
	({ i, id: 42, text: " Paris", p: 0.8, entropy: 0.6, alts: [[" Lyon", 0.1]], n_layers: 24, ...extra });

test("lensSettle: the earliest layer from which every deeper read layer has the token", () => {
	assert.equal(lensSettle(lens, 42), 16);
	assert.equal(lensSettle(lens.map(([l, , s]) => [l, 42, s] as [number, number, string]), 42), 0);   // all agree: the first layer
	assert.equal(lensSettle([...lens, [24, 5, " Rome"]], 42), null);                                     // the last read layer differs
});

/** A dock on a fake TUI following a reply of `toks`. */
function dock(toks: TokenEvent[], done = false) {
	const live: Live = { base: "", clientId: "me", level: "on", connected: true, reply: { rid: "r", tokens: toks, done, prefill: 3 } };
	const host = { live, captions: false, close() {}, unfocus() {} };
	return new Dock({ terminal: { rows: 40 }, requestRender() {} }, theme, host);
}

test("dock: while the next request reads its prompt, the last reply stays drawn and the status says so", () => {
	const d = dock([ev(0, { text: "Purring" }), ev(1, { text: " cats" })], true);
	const live = d["host"].live;
	live.prev = live.reply;
	live.reply = { rid: "next", tokens: [], done: false, prefill: 812 };
	const text = d.render(104).map((l) => plain(l).replace(/^[▌▐] /, "").trim()).join(" ");
	assert.match(text, /reading the prompt · 812 tok · the last reply below/);
	assert.match(text, /Purring cats/);
	assert.doesNotMatch(text, /Fills in as a traced reply streams/);
});

test("dock: an empty dock says what fills it and how to step in", () => {
	const d = dock([]);
	d["host"].live.reply = undefined;
	d["host"].live.level = "off";   // tracing on, the idle scene instead (test/idle.test.ts)
	const empty = d.render(60).map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd()).join(" ");
	assert.match(empty, /Fills in as a traced reply streams/);
	assert.match(empty, /ctrl\+\\ steps in: 6 steer · \/scope help dock/);
	d.stepIn(-1);                                                                    // stepped in: the keys that need no reply
	assert.equal(plain(d.render(60).at(-1)!).replace(/^[▌▐] /, "").trimEnd(), "6 steer · m mini · esc back");
});

test("offTop: under half chosen while another token had over half", () => {
	assert.equal(offTop(0.068, [[" far", 0.83]]), true);
	assert.equal(offTop(0.4, [[" a", 0.35]]), false);                           // hesitant: nothing had half
	assert.equal(offTop(0.6, [[" x", 0.4]]), false);                            // not unsure at all
	assert.equal(offTop(null, []), false);
});

test("dock colours: its own palette whatever Pi's theme, 256 colours where the terminal has no more", () => {
	const purple = { ...theme, getColorMode: () => "truecolor" };
	const lines = dock([ev(0, { lens })], true).render(60);
	assert.ok(lines.every((l) => l.startsWith("\x1b[48;2;11;13;16m\x1b[38;2;215;219;226m")), "background and text colour");
	assert.ok(lines.every((l) => !l.includes("\x1b[48;5;236m")), "not Pi's message background");
	assert.ok(lines.every((l) => l.includes("\x1b[38;2;123;132;146m▌")), "the keyboard in the chat: a dim edge on the chat's side");
	assert.equal(dockTheme(purple).fg("success", "x"), "\x1b[38;2;127;201;143mx\x1b[39m");
	assert.equal(dockTheme(purple).fg("lens", "x"), "\x1b[38;2;183;156;242mx\x1b[39m");
	const t256 = dockTheme({ getColorMode: () => "256color" });
	assert.match(t256.fg("accent", "x"), /^\x1b\[38;5;\d+mx\x1b\[39m$/);
	assert.match(t256.bg("customMessageBg", "x"), /^\x1b\[48;5;\d+mx\x1b\[49m$/);
});

test("dock edge: it leans toward whoever has the keyboard, the content's columns unchanged", () => {
	const d = dock([ev(0, { lens })], true), out = d.render(104);
	d.stepIn(-1);
	const inn = d.render(104);
	assert.ok(inn.every((l) => l.includes("\x1b[38;2;240;162;58m▐")), "stepped in: an orange edge on the dock's side");
	assert.ok(inn.every((l) => !l.includes("▌")), "no chat-side edge left");
	const body = (ls: string[]) => ls.map((l) => plain(l).slice(2));
	assert.deepEqual(inn.map((l) => plain(l).length), out.map((l) => plain(l).length), "the same widths");
	assert.equal(body(inn)[1], body(out)[1], "the content in the same columns");
	d["leave"]();
	assert.ok(d.render(104).every((l) => l.includes("\x1b[38;2;123;132;146m▌")), "esc: back to the chat's side");
});

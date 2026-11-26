import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { Dock, inputAt, inputOf } from "../dock.ts";
import { COLOURS as DOCK, LABELS, centre, colourCode, cut } from "../draw.ts";
import { type Live, type TokenEvent, lensP, onEvent, packLens } from "../state.ts";
import { focusLayer, fog, pc, scopeHeight, stepLayer, stretchFrames, swapFrames, telescope, tube, tubeCap, tubeOf, turret } from "../telescope.ts";

const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const COLOURS: Record<string, number> = { success: 2, muted: 8, dim: 7, accent: 6, warning: 3, error: 1 };
/** A tube's row of a layer wanted, not read: its label in `faint` (the colour code it opens with), the row blank. */
const esc = (s: string) => s.replace(/[[\]\\^$.|?*+()]/g, "\\$&");
const unreadIn = (faint: string) => new RegExp(`${esc(faint)}[ ▸]L\\d+(?: \\+\\d+)? +\x1b`);
/** The dock's (dockTheme: truecolor) unread rows in its raw lines. */
const unread = (raw: string[]) => raw.filter((l) => unreadIn(colourCode(DOCK.faint, "truecolor")).test(l)).length;
const theme = {
	fg: (c: string, s: string) => `\x1b[38;5;${COLOURS[c] ?? 15}m${s}\x1b[39m`,
	bg: (_c: string, s: string) => `\x1b[48;5;236m${s}\x1b[49m`,
	bold: (s: string) => `\x1b[1m${s}\x1b[22m`,
};
const byte = (p: number) => Math.round(-Math.log(p) * 20);

/** A lens10 field as the stream sends it: per layer, the top 10 as ids (with `words`) and probabilities. */
function lens10(rows: [number, [number, number][]][], words: Record<number, string>) {
	const i: number[] = [], c: number[] = [], p: number[] = [];
	for (const [, top] of rows)
		for (let r = 0; r < 10; r++) {
			const [id, pr] = top[r] ?? [900 + r, 0.001];
			if (!i.includes(id)) i.push(id);
			c.push(i.indexOf(id));
			p.push(byte(pr));
		}
	return { l: rows.map(([l]) => l), k: 10, i, w: i.map((x) => words[x] ?? ` w${x}`), c, p };
}
const WORDS: Record<number, string> = { 42: " reader", 7: " audience", 8: " 读者", 9: " tone", 5: " vibe" };

// " reader" (id 42) in a 48-layer model. The J-lens reads 36, 40, 44, 46 (settles at 44); the logit lens 24, 44, 47.
const J: [number, [number, number][]][] = [[36, [[5, 0.3]]], [40, [[7, 0.2], [42, 0.05]]], [44, [[42, 0.09], [7, 0.08], [8, 0.06]]], [46, [[42, 0.2]]]];
const LG: [number, [number, number][]][] = [[24, [[9, 0.4]]], [44, [[7, 0.3], [42, 0.1]]], [47, [[42, 0.5]]]];
function token(extra: Partial<TokenEvent> & Record<string, unknown> = {}): TokenEvent {
	const t: any = {
		i: 1056, id: 42, text: " reader", p: 0.4, entropy: 1, alts: [], n_layers: 48,
		lens: J.map(([l, top]) => [l, top[0][0], WORDS[top[0][0]]]), lens_by: "jlens",
		lens_logit: LG.map(([l, top]) => [l, top[0][0], WORDS[top[0][0]]]),
		lens10: lens10(J, WORDS), lens10_logit: lens10(LG, WORDS), ...extra,
	};
	packLens(t);
	return t;
}
/** A token whose J-lens read every layer, 0-46 (no logit lens, no top 3s): " reader" from L44, " vibe" at L34. */
const deepToken = () => token({ lens: Array.from({ length: 47 }, (_, l) => (l >= 44 ? [l, 42, " reader"] : l === 34 ? [l, 5, " vibe"] : [l, 9, " tone"])) as [number, number, string][],
	lens_logit: undefined, lens10: undefined, lens10_logit: undefined, top: undefined });

test("packLens: each read layer's top 3 and the written token's probability, from the stream's lens10s", () => {
	const t = token();
	assert.deepEqual(t.top!.l, [36, 40, 44, 46]);
	assert.deepEqual([...t.top!.id.slice(6, 9)], [42, 7, 8]);                   // L44's top 3
	assert.deepEqual(t.top!.w.slice(6, 9), [" reader", " audience", " 读者"]);
	assert.ok(Math.abs(lensP(t.top!.p[6])! - 0.09) < 0.005);
	assert.equal(lensP(t.top!.wp[0]), null);                                     // L36: " reader" not in its top 10
	assert.ok(Math.abs(lensP(t.top!.wp[1])! - 0.05) < 0.003);                    // L40: second
	assert.deepEqual(t.top_logit!.l, [24, 44, 47]);
	assert.equal((t as any).lens10, undefined);                                  // the raw fields are not kept
	const bad: any = { id: 1, lens10: { l: [1], k: 10, i: [1], w: ["a"], c: [0], p: [0] } };   // too few cells
	packLens(bad);
	assert.equal(bad.top, undefined);
});

test("tube: over the layers read, three narrowing sections at any length; past the cap the shallow layers share lines", () => {
	const secs = (ls: number[], cap = 21) => [0, 1, 2].map((s) => tube(ls, cap).filter((x) => x.sec === s).length);
	const upTo = (n: number) => Array.from({ length: n }, (_, k) => k);
	assert.equal(tubeCap(48), 21);                                                // the old depth-wide tube's lines
	assert.deepEqual(secs([24, 31, 38, 46]), [2, 1, 1]);                         // 4 layers: a short telescope, same shape
	assert.deepEqual(tube([24, 46, 31, 38], 21), [{ ls: [46], sec: 0 }, { ls: [38], sec: 0 }, { ls: [31], sec: 1 }, { ls: [24], sec: 2 }]);
	assert.deepEqual(secs(upTo(15)), [8, 4, 3]);
	const all = tube(upTo(47), 21);                                              // every layer: the cap, a gentler taper
	assert.equal(all.length, 21);
	assert.deepEqual(secs(upTo(47)), [11, 5, 5]);
	assert.ok(all.filter((x) => x.sec === 0).every((x) => x.ls.length === 1));    // the deep half a layer a line
	assert.deepEqual(all.flatMap((x) => x.ls), upTo(47).reverse());             // every layer once, deep first
	assert.deepEqual(all.map((x) => x.ls.length), [...Array(11).fill(1), 2, 2, 2, 2, 2, 5, 5, 5, 5, 6]);
	assert.deepEqual(tube(upTo(22), 21).map((x) => x.ls.length), [...Array(16).fill(1), 1, 1, 1, 1, 2]);
	assert.deepEqual(tube([5], 21), [{ ls: [5], sec: 0 }]);
	assert.deepEqual(tube([], 21), []);
	// Any length at any depth: every layer once, deep first; at most the cap; sections in order; a line never holds
	// fewer layers than one deeper.
	for (const depth of [28, 40, 48, 64])
		for (let k = 1; k <= depth; k++) {
			const ls = Array.from({ length: k }, (_, j) => Math.floor((j * depth) / k)), why = `depth ${depth}, ${k} layers`;
			const t = tube(ls, tubeCap(depth));
			assert.deepEqual(t.flatMap((x) => x.ls), [...ls].reverse(), why);
			assert.ok(t.length <= tubeCap(depth), why);
			assert.ok(t.every((x, j) => !j || (x.sec >= t[j - 1].sec && x.ls.length >= t[j - 1].ls.length)), `${why}: ${t.map((x) => `${x.sec}:${x.ls.length}`)}`);
		}
});

test("stretch frames: the layers coming in deep first, going out shallow first; the last frame the new set", () => {
	const fr = stretchFrames([24, 31, 38, 46], [6, 12, 24, 31, 38, 43, 46]);
	assert.equal(fr.length, 4);
	assert.deepEqual(fr.map((f) => f.want!.length), [5, 6, 6, 7]);
	assert.deepEqual(fr[0].want, [24, 31, 38, 46, 43]);
	assert.deepEqual(new Set(fr[3].want), new Set([6, 12, 24, 31, 38, 43, 46]));
	const back = stretchFrames([6, 12, 24, 31, 38, 43, 46], [24, 31, 38, 46]);
	assert.deepEqual(back.map((f) => f.want!.length), [6, 5, 5, 4]);
	assert.ok(!back[0].want!.includes(6) && back[0].want!.includes(43));
});

test("focus and steps: the asked layer's nearest read layer, else the settle layer, else the deepest", () => {
	const ls = [36, 40, 44, 46];
	assert.equal(focusLayer(ls, 44), 44);
	assert.equal(focusLayer(ls, null), 46);
	assert.equal(focusLayer(ls, 44, 41), 40);
	assert.equal(focusLayer(ls, 44, 42), 44);                                   // a tie: the deeper
	assert.equal(focusLayer([], null), null);
	assert.equal(stepLayer(ls, 44, 1), 46);
	assert.equal(stepLayer(ls, 46, 1), 46);
	assert.equal(stepLayer(ls, 36, -1), 36);
	assert.equal(stepLayer(ls, 40, -1), 36);
});

test("text helpers: fog keeps widths and spaces; centre and cut by display width; probabilities as the page shows them", () => {
	assert.equal(fog("ab c", 1), "▒▒ ▒");
	assert.equal(fog("读者", 2), "░░░░");
	assert.equal(fog("x", 0), "x");
	assert.equal(centre("ab", 6), "  ab  ");
	assert.equal(visibleWidth(centre("读者", 7)), 7);
	assert.equal(cut("interpretability", 6), "inter…");
	assert.deepEqual([pc(0.091), pc(0.001), pc(null)], ["9%", "<1%", "·"]);
});

test("telescope: the token, the eyepiece, the written word and the tube, 44 wide", () => {
	const { lines, focus } = telescope(theme, token(), 1056, 44, 60);
	const p = lines.map(plain);
	if (process.env.SHOW_DOCK) console.log(p.join("\n"));
	assert.equal(focus, 44);
	assert.ok(p.every((l) => visibleWidth(l) <= 44), String(p.find((l) => visibleWidth(l) > 44)));
	assert.ok(p.slice(1, 10).every((l) => visibleWidth(l) === 44));             // the eyepiece and the widest section fill it
	assert.equal(p[0], '#1056 "reader" · settles L44/48');
	assert.match(p[1], /^╭─ out of L44 ─+╮$/);
	assert.match(p[2], /^│ reader +9%  │$/);
	assert.match(p[3], /^│ audience +8%  │$/);
	assert.match(p[4], /^│ 读者 +6%  │$/);
	assert.ok(lines[2].includes(theme.fg("success", theme.bold("reader"))));   // the written token: green, bold
	assert.match(p[5], /^╰─+╯$/);
	assert.equal(p[6].trim(), "→ reader");
	// The tube: the six layers either lens read (J 36 40 44 46, logit 24 44 47), three sections of 3, 2 and 1 lines.
	assert.match(p[7], /^│ L47 +│$/);                                           // read by the logit lens alone: no word
	assert.match(p[8], /^│ L46 +reader +20%│$/);
	assert.match(p[9], /^│▸L44 +reader +9%│$/);
	assert.ok(lines[9].includes("\x1b[7m"));                                    // the focused row inverted
	assert.match(p[10], /^╰───╮ +╭───╯$/);                                      // a joint
	assert.match(p[11], /^ {4}│ L40 +audience +5%│$/);
	assert.match(p[12], /^ {4}│ L36 +vibe +·│$/);                               // the written word outside the top 10
	assert.match(p[13], /^ {4}╰───╮ +╭───╯$/);
	assert.match(p[14], /^ {8}│ L24 +│$/);
	assert.match(p[15], /^ {8}╰─+ in: not streamed ─+╯$/);                     // no input given: not claimed
	assert.equal(p.length, 16);                                                 // 17 with the turret above
	assert.ok(!lines.some((l) => unreadIn(theme.fg("faint", "").slice(0, -"\x1b[39m".length)).test(l)), "no faint row: every layer here was read (L24 and L47 the other lens's: dim)");
	assert.ok(lines[7].includes("\x1b[38;5;7m L47"));                          // the other lens's line: dim
	assert.ok(lines[8].includes(theme.fg("success", centre("reader", 44 - 2 - 14))));
});

test("telescope: several read layers on a line show the focused or the deepest, with +n; the logit lens; no top 3s", () => {
	// The J-lens read every layer (47 past the 21-line cap): the shallower ones share lines.
	const t = deepToken();
	const p = telescope(theme, t, 7, 44, 60).lines.map(plain);
	assert.ok(p.some((l) => /│ L35 \+1 +tone +│/.test(l)), p.join("\n"));      // L35 34 on a line: the deepest
	assert.equal(p.filter((l) => /│[ ▸]L\d+/.test(l)).length, 21);
	assert.match(p[2], /^│ reader +│$/);                                       // the top word alone, no probability
	assert.match(p[3], /^│ +│$/);
	const f = plain(telescope(theme, t, 7, 44, 60, { focus: 34 }).lines.find((l) => /L34 \+1/.test(plain(l)))!);
	assert.match(f, /L34 \+1 +vibe/);                                          // the focused layer shown on its line
	const lg = telescope(theme, token(), 1056, 44, 60, { view: "logit" }).lines.map(plain);
	assert.match(lg[1], /out of L47/);                                         // the logit lens settles at L47
	assert.match(lg[2], /^│ reader +50%  │$/);
	assert.ok(lg.some((l) => /│▸L47 +reader +50%│/.test(l)));
	const lg44 = telescope(theme, token(), 1056, 44, 60, { view: "logit", focus: 44 }).lines.map(plain);
	assert.match(lg44[2], /^│ audience +30%  │$/);
	assert.match(lg44[3], /^│ reader +10%  │$/);
	assert.equal(lg[0], '#1056 "reader" · settles L47/48');
	const never = telescope(theme, token({ lens: [[44, 7, " audience"]] as [number, number, string][], lens_logit: undefined }), 3, 44, 60).lines.map(plain);
	assert.equal(never[0], '#3 "reader" · never settles');
});

test("telescope: a short dock drops the shallow end first and says how many layers; a narrow one says what it needs", () => {
	const p = telescope(theme, deepToken(), 1056, 44, 20).lines.map(plain);
	assert.equal(p.length, 20);
	assert.equal(p[19], "… 36 shallower layers");                             // L46..L36 kept, a line each, and a joint
	assert.match(p[17], /│ L36 +tone/);
	assert.deepEqual(telescope(theme, token(), 1, 30, 40).lines.map(plain), ["needs 36 columns"]);
	assert.ok(telescope(theme, token(), 1, 41, 60).lines.every((l) => visibleWidth(plain(l)) <= 41));
});

test("turret and swap frames: the lens in use bright; the old lens fogs, the ◉ slides, the new one clears from the deep end", () => {
	assert.equal(plain(turret(theme, token(), "jlens")), "J-lens ◉────○ logit");
	assert.equal(plain(turret(theme, token(), "logit")), "J-lens ○────◉ logit");
	assert.equal(plain(turret(theme, token(), "jlens", 2)), "J-lens ○──◉─○ logit".replace("○──◉─○", [..."○────○"].map((c, i) => (i === 3 ? "◉" : c)).join("")));
	assert.ok(turret(theme, token(), "jlens").includes(theme.fg("faint", "logit")));
	assert.equal(plain(turret(theme, token({ lens_logit: undefined }), "logit")), "J-lens");   // one lens: its name
	const fr = swapFrames("jlens", "logit", 21);
	assert.equal(fr.length, 9);
	assert.deepEqual(fr.map((f) => f.turret), [0, 1, 2, 3, 4, 4, 4, 4, 4]);
	assert.deepEqual(fr.map((f) => f.view), ["jlens", "jlens", "jlens", "logit", "logit", "logit", "logit", "logit", "logit"]);
	assert.deepEqual(fr.map((f) => f.fog ?? 0), [1, 2, 2, 2, 2, 1, 1, 1, 0]);
	assert.equal(fr[8].sharp, undefined);
	const foggy = telescope(theme, token(), 1056, 44, 60, fr[1]).lines.map(plain);
	assert.match(foggy[2], /^│ ░░░░░░ +░░  │$/);
	assert.ok(foggy.some((l) => /│ L46 +░░░░░░ +░░░│/.test(l)));
	const clearing = telescope(theme, token(), 1056, 44, 60, fr[5]).lines.map(plain);
	assert.ok(clearing.some((l) => /│▸L47 +reader +50%│/.test(l)));            // the deep end sharp first
	assert.ok(clearing.some((l) => /│ L24 \+1? *▒+/.test(l) || /L24.*▒/.test(l)), clearing.join("\n"));
	assert.deepEqual(swapFrames("logit", "jlens", 21).map((f) => f.turret), [4, 3, 2, 1, 0, 0, 0, 0, 0]);
});

/** A dock (the one-panel board, LENS in focus, below 100 columns) following one reply of tokens like `token()`. */
function lensDock(words = WORDS) {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r", model: "Flash" });
	for (let i = 0; i < 6; i++) {
		const t: any = token();
		onEvent(live, { type: "token", rid: "r", ...t, i, text: i ? " reader" : "The", top: undefined, top_logit: undefined,
			lens10: lens10(J, words), lens10_logit: lens10(LG, words) });
	}
	onEvent(live, { type: "end", rid: "r", n_tokens: 6, trace_id: "T" });
	const renders: number[] = [];
	const host: any = { live, captions: false, close() {}, unfocus() {}, panel: "LENS" };
	const d = new Dock({ terminal: { rows: 50 }, requestRender: () => renders.push(1) }, theme, host);
	return { d, host, live, renders };
}
const text = (lines: string[]) => lines.map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd());
/** An edit sent is over (its answer in; no server here): the dock takes the next one. */
const answered = async (d: Dock) => (await tick(), (d as any).engram.stopSweep());
/** LENS's head line (its lens switch) on the one-panel board: under the tab row, which names it. */
const lensAt = (lines: string[]) => lines.findIndex((l) => /^ ?1 TOKENS +2 MAP +3 LENS/.test(l)) + 1;

test("dock, LENS on the one-panel board: inspecting, the telescope; ↑/↓ step the focused layer and it is kept across tokens", () => {
	const { d, host } = lensDock();
	d.sel = 3;
	let lines = text(d.render(48));
	if (process.env.SHOW_DOCK) console.log(lines.join("\n"));
	const at = lensAt(lines);
	assert.ok(at > 0);
	assert.match(lines[at], /^J-lens ◉────○ logit/);
	assert.match(lines[at + 1], /^SETTLING\b/, "SETTLING under LENS's head line, as on the board");
	const tel = lines.indexOf('#3 "reader" · settles L44/48');
	assert.ok(tel > at + 2 && !lines[tel - 1], "a blank line, then the telescope");
	assert.match(lines[tel + 1], /out of L44/);
	assert.match(lines.join(" ").replace(/\s+/g, " "), /↑\/↓ layer (· )?enter edit (· )?s other lens/);
	d.handleInput("\x1b[A");                                                      // ↑: deeper
	assert.equal(host.lensLayer, 46);
	assert.match(text(d.render(48))[tel + 1], /out of L46/);
	d.handleInput("\x1b[C");                                                      // → the next token: the layer kept
	assert.equal(d.sel, 4);
	assert.match(text(d.render(48))[tel + 1], /out of L46/);
	d.handleInput("j");                                                           // ↓ (j): shallower
	d.handleInput("j");
	assert.equal(host.lensLayer, 40);
	lines = text(d.render(48));
	assert.ok(lines.every((l) => visibleWidth(l) <= 48));
	assert.ok(lines.length <= 50);
});

test("dock, LENS on the one-panel board: s swaps the lens with a short animation; a key finishes it; it never leaves a fogged frame", () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const { d, host, renders } = lensDock();
		d.sel = 2;
		d.render(48);
		d.handleInput("s");
		assert.equal(host.lensView, "logit");
		const first = text(d.render(48)).join("\n");
		assert.match(first, /▒/);                                                 // the old lens fogging
		mock.timers.tick(70 * 3);
		assert.match(text(d.render(48)).join("\n"), /J-lens ○──◉─○ logit|J-lens ○───◉○ logit/);
		mock.timers.tick(70 * 10);
		const done = text(d.render(48)).join("\n");
		assert.doesNotMatch(done, /[▒░]/);
		assert.match(done, /J-lens ○────◉ logit/);
		assert.ok(renders.length >= 8);
		const n = renders.length;
		mock.timers.tick(70 * 10);
		assert.equal(renders.length, n);                                            // the timer stopped
		d.handleInput("s");                                                        // back, and a key half way
		mock.timers.tick(70);
		d.handleInput("\x1b[C");
		const after = text(d.render(48)).join("\n");
		assert.doesNotMatch(after, /[▒░]/);
		assert.match(after, /J-lens ◉────○ logit/);
		d.handleInput("s");
		d.dispose();                                                               // the dock gone: the timer with it
		const m = renders.length;
		mock.timers.tick(70 * 10);
		assert.equal(renders.length, m);
	} finally {
		mock.timers.reset();
	}
});

test("fog: exactly as wide as the word, whatever its graphemes (combining marks, ZWJ sequences, modifiers)", () => {
	for (const s of ["नमस्ते", "👨‍💻", "👍🏽 ok", "éte", "读者", " reader "])
		for (const lvl of [1, 2]) {
			const f = fog(s, lvl);
			assert.equal(visibleWidth(f), visibleWidth(s), `${s} ${lvl}`);
			assert.match(f, lvl === 1 ? /^[▒ ]*$/ : /^[░ ]*$/);
		}
	assert.equal(fog("👍🏽 ok", 2), "░░ ░░");
});

test("telescope: a short page keeps the eyepiece and the deep end with the note; shorter still, one line says so", () => {
	for (let room = 0; room < 40; room++) {
		const p = telescope(theme, token(), 1056, 44, room).lines.map(plain);
		assert.ok(p.length <= room, `room ${room}: ${p.length} lines`);
		if (room === 0) assert.deepEqual(p, []);
		else if (room < 9) assert.deepEqual(p, ["needs 9 lines: a taller terminal shows the lens"]);
		else {
			assert.match(p[1], /out of L44/);                                      // the eyepiece always whole
			assert.ok(/shallower layers?$/.test(p[p.length - 1]) || /in: /.test(p[p.length - 1]), `room ${room}: ${p[p.length - 1]}`);
		}
	}
	const nine = telescope(theme, token(), 1056, 44, 9).lines.map(plain);
	assert.match(nine[7], /^│ L47 +│$/);                                       // one tube line, the deepest
	assert.equal(nine[8], "… 5 shallower layers");
});

test("dock, LENS on the one-panel board: at small docks nothing is cut without saying so, and the key line stays", () => {
	for (const rows of [23, 24, 28, 32, 40]) {
		const { d } = lensDock();
		(d as any).tui.terminal.rows = rows;
		d.sel = 3;
		const lines = text(d.render(48));
		const flat = lines.join("\n");
		assert.match(flat, /tab experts · ←\/→ token/, `rows ${rows}`);              // the key line
		assert.ok(/needs 9 lines|shallower layers?|in: /.test(flat), `rows ${rows}:\n${flat}`);
		assert.ok(lines.length === Math.max(12, rows), `rows ${rows}`);
	}
});

test("telescope: the bottom names the token that went in (the one before), the first token the prompt's end; cut to fit", () => {
	const bottom = (input: string | null | undefined, w = 44) => {
		const lines = telescope(theme, token(), 1056, w, 60, { input }).lines;
		return { raw: lines.at(-1)!, p: plain(lines.at(-1)!), w: visibleWidth(plain(lines.at(-1)!)) };
	};
	let b = bottom(" the");
	assert.match(b.p, /^ {8}╰─+ in: "the" ─+╯$/);
	assert.equal(b.w, 36, "as wide as the shallow section above it (indented 8)");
	assert.ok(b.raw.includes(theme.bold(' in: "the" ')), "the label bold in the lens colour");
	b = bottom(null);
	assert.match(b.p, /^ {8}╰─+ in: end of prompt ─+╯$/);
	assert.equal(b.w, 36);
	b = bottom(" 读者");                                                           // wide characters: two columns each
	assert.match(b.p, /^ {8}╰─+ in: "读者" ─+╯$/);
	assert.equal(b.w, 36);
	b = bottom("\n\n");
	assert.match(b.p, /in: "↵↵"/);
	b = bottom("\x07bell");
	assert.match(b.p, /in: "·bell"/, "control characters made visible");
	b = bottom(" interpretability-research-programme");                        // longer than the bottom: cut with …
	assert.match(b.p, /^ {8}╰─* in: "interpret[a-z-]*…" ─*╯$/);
	assert.equal(b.w, 36);
	for (const width of [36, 40, 50, 60]) {
		for (const input of [null, " x", " 读者读者读者读者读者读者读者读者", " a-very-long-token-indeed-it-is"]) {
			const x = bottom(input, width);
			assert.equal(x.w, width - 8, `${width} ${JSON.stringify(input)}: ${x.p}`);
			assert.match(x.p, /╰─+ in: .*─+╯$/, `at least a dash each side: ${x.p}`);
		}
	}
});

test("dock, LENS on the one-panel board: the token that went in is on the lens's violet in the reply, the selection inverse; the telescope says it", () => {
	const { d, live } = lensDock();
	["The", " old", " lamp", " reader", " here", " now"].forEach((x, i) => (live.reply!.tokens[i].text = x));
	d.sel = 3;
	const raw = d.render(48), lines = text(raw);
	const IN = "\x1b[48;2;60;46;98m";                                         // the dock's own lensIn (draw.ts COLOURS)
	assert.ok(raw.some((l) => l.includes(IN) && plain(l.slice(l.indexOf(IN))).startsWith(" lamp")), "the token before on a background");
	assert.ok(raw.some((l) => l.includes("\x1b[7m reader\x1b[27m")), "the selection inverse");
	assert.ok(lines.some((l) => /╰─+ in: "lamp" ─+╯/.test(l)));
	d.sel = 0;
	const first = d.render(48);
	assert.ok(!first.some((l) => l.includes(IN)), "the first token: nothing before it in the reply");
	assert.ok(text(first).some((l) => /in: end of prompt/.test(l)));
	d.sel = -1;
	assert.ok(!d.render(48).some((l) => l.includes(IN)), "not inspecting: no pair");
});

test("telescope: the input is the token before only where the stream's index says so (specials unpublished, a reply adopted mid-stream)", () => {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	for (const i of [57, 58, 60]) onEvent(live, { type: "token", rid: "r", ...(token() as any), i, text: ` t${i}`, top: undefined, top_logit: undefined } as any);
	const toks = live.reply!.tokens;
	assert.deepEqual(toks.map((_t, k) => inputOf(toks, k)), [false, " t57", false]);
	assert.deepEqual(toks.map((_t, k) => inputAt(toks, k)), [-1, 0, -1]);
	const bottom = (input: string | null | false) => plain(telescope(theme, token(), 1, 44, 60, { input }).lines.at(-1)!);
	assert.match(bottom(false), /in: not streamed/);
	assert.match(bottom(null), /in: end of prompt/);
	assert.equal(inputOf([{ i: 0, text: "x" }], 0), null, "the stream's first token: the prompt's end went in");
	assert.equal(inputOf([{ i: 1, text: "x" }], 0), false, "after an unpublished special token");
	// LENS marks the token before only when it went in.
	const d = lensDock();
	const tk = d.live.reply!.tokens;
	tk.forEach((t, k) => (t.i = k < 3 ? k : k + 1));                                 // a special between #2 and #3
	d.d.sel = 3;
	const IN = "\x1b[48;2;60;46;98m";
	let raw = d.d.render(48);
	assert.ok(!raw.some((l) => l.includes(IN)) && text(raw).some((l) => /in: not streamed/.test(l)));
	d.d.sel = 2;
	raw = d.d.render(48);
	assert.ok(raw.some((l) => l.includes(IN)));
});

const tick = () => new Promise((r) => setTimeout(r, 0));
const KEY = { up: "\x1b[A", down: "\x1b[B", tab: "\t", esc: "\x1b", enter: "\r", bs: "\x7f" };

test("telescope: the word edited is the field: ▸ inverse; ⌫ struck and removed; typed over, struck → the new; never wider", () => {
	const row = (edit: { row: number; field: string | null }, t = token(), w = 44) => telescope(theme, t, 3, w, 60, { edit }).lines[2 + edit.row];
	assert.match(row({ row: 0, field: null }), /▸\x1b\[7mreader\x1b\[27m/);
	assert.match(plain(row({ row: 0, field: "" })), /^│▸reader removed  +9% {2}│$/);
	assert.match(row({ row: 0, field: "" }), /\x1b\[9mreader\x1b\[29m/);
	assert.match(plain(row({ row: 1, field: "rugby" })), /^│▸audience → rugby  +8% {2}│$/);
	assert.match(plain(row({ row: 1, field: "x" })), /^│▸audience → x /);
	assert.match(plain(telescope(theme, token(), 3, 44, 60, { edit: { row: 1, field: "x" } }).lines[2]), /^│ reader/, "only the row edited");
	// A long word and a long field: the old word cut to half the column, the field to its end, every line w wide at most.
	const long = token({ lens10: lens10(J, { ...WORDS, 42: " extraordinarilylongwordthatgoesonandon" }) } as any);
	for (const w of [36, 44, 60, 97]) for (const field of [null, "", "x".repeat(80) + "end"]) {
		const lines = telescope(theme, long, 3, w, 60, { edit: { row: 0, field } }).lines.map(plain);
		assert.ok(lines.every((l) => visibleWidth(l) <= w), `${w} ${field}: ${lines.join("\n")}`);
		assert.equal(visibleWidth(lines[2]), w);
		if (field) assert.match(lines[2], /xend +9% {2}│$/);
	}
});

test("telescope: a push marked ◆ on the line holding its layer", () => {
	// A line spans from its shallowest layer to the line above's: marks land on the line holding their layer.
	const lines = telescope(theme, deepToken(), 3, 44, 60, { steer: [31] }).lines.map(plain);
	const at = (layers: string) => lines.find((l) => new RegExp(`^ *│[ ▸]${layers} `).test(l)) ?? "";
	assert.equal(lines.filter((l) => /◆/.test(l)).length, 1);
	assert.match(at("L31 \\+1"), /◆ steer$/);                                   // L31 30
	assert.match(at("L44"), /│$/);
	// Few layers read: the line above a gap holds it (token(): L24's line spans 0-35).
	const few = telescope(theme, token(), 3, 44, 60, { steer: [31] }).lines.map(plain);
	assert.match(few.find((l) => /│ L24 /.test(l))!, /◆ steer$/);
});

test("dock, LENS on the one-panel board: enter edits the written word (else the first shown); ⌫, letters, ↑/↓ and tab; enter answers again from that token", async () => {
	const { d, host, live } = lensDock();
	const edited: any[] = [], steered: string[] = [];
	let answer: (said: { text: string; level: string }[]) => void = () => {};
	host.edit = (a: string, ids: any, at: any) => {
		edited.push([a, ids, at]);
		return new Promise((done) => (answer = done));
	};
	host.steer = async (a: string) => {
		steered.push(a);
		return [{ text: `steer: ${a}`, level: "info" }];
	};
	d.sel = 3;
	const eyeRows = () => text(d.render(48)).filter((l) => /^│[ ▸]/.test(l)).slice(0, 3);
	const keyLine = () => text(d.render(48)).at(-1)!;
	// The starting word: the one written where the layer shows it (L44 row 1, L40 row 2), else the first (L36).
	for (const [layer, row, word] of [[44, 0, "reader"], [40, 1, "reader"], [36, 0, "vibe"]] as [number, number, string][]) {
		host.lensLayer = layer;
		d.handleInput(KEY.enter);
		assert.match(eyeRows()[row], new RegExp(`^│▸${word} `), `L${layer}`);
		d.handleInput(KEY.esc);
		assert.ok(!eyeRows().some((l) => l.startsWith("│▸")), "esc cancels");
	}
	host.lensLayer = 44;
	d.handleInput(KEY.enter);
	assert.deepEqual(text(d.render(48)).slice(-2), ["⌫ remove · type or ↑/↓ a word · tab next", "enter apply, redo from here · esc cancel"]);   // two lines, whole
	d.handleInput(KEY.enter);                                                        // untouched: nothing sent
	assert.equal(keyLine(), "⌫ removes it, or type a word");
	d.handleInput(KEY.bs);
	assert.match(eyeRows()[0], /^│▸reader removed/);
	for (const c of "rugby") d.handleInput(c);
	assert.match(eyeRows()[0], /^│▸reader → rugby/);
	d.handleInput(KEY.bs);
	assert.match(eyeRows()[0], /^│▸reader → rugb /);
	// ↑/↓: the eyepiece's other words, then the tube's (deep first): audience, 读者 (a word, by its id), vibe.
	d.handleInput(KEY.down);
	assert.match(eyeRows()[0], /→ audience/);
	d.handleInput(KEY.down);
	assert.match(eyeRows()[0], /→ 读者/);
	d.handleInput(KEY.down);
	assert.match(eyeRows()[0], /→ vibe/);
	d.handleInput(KEY.up);
	assert.match(eyeRows()[0], /→ 读者/);
	d.handleInput(KEY.bs);                                                           // a character at a time
	assert.match(eyeRows()[0], /→ 读 /);
	d.handleInput(KEY.tab);                                                          // the next word, untouched
	assert.match(eyeRows()[1], /^│▸audience /);
	d.handleInput(KEY.tab);
	assert.match(eyeRows()[2], /^│▸读者 /);
	d.handleInput(KEY.tab);                                                          // round to reader
	for (const c of "x-ray") d.handleInput(c);                                       // x and the others are letters while editing
	d.handleInput(KEY.enter);
	// The old word by the id of the token shown, the new one pushed at the edit's dose; this token of this reply.
	assert.deepEqual(edited, [["tok42>x-ray@0.7", { tok42: { id: 42, text: "reader" } }, { rid: "r", at: 3, word: "reader", want: "x-ray" }]]);
	assert.equal(keyLine(), LABELS.editing);
	answer([{ text: "steer: built", level: "info" }]);
	await tick();
	assert.ok(!eyeRows().some((l) => l.startsWith("│▸")) && keyLine() === LABELS.editing);   // until its reply has the token
	assert.deepEqual([steered, live.steer], [[], undefined]);                       // nothing turned on
	// Refused: its line on the key line, until the next key.
	d.handleInput(KEY.enter);
	d.handleInput(KEY.bs);
	d.handleInput(KEY.enter);
	answer([{ text: "Soccer is more than one token: the edit takes a one-token word", level: "error" }]);
	await tick();
	assert.equal(text(d.render(48)).slice(-2).join(" "), "Soccer is more than one token: the edit takes a one-token word");   // the command's line, wrapped into the two key lines
	d.handleInput(KEY.up);                                                           // the next key clears it
	assert.match(text(d.render(48)).join(" "), /enter edit/);
	host.flags = () => ({ edit: false });                                            // the server refuses this model one
	assert.doesNotMatch(text(d.render(48)).join(" "), /enter edit/);
	d.handleInput(KEY.enter);
	assert.ok(!eyeRows().some((l) => l.startsWith("│▸")), "not offered");
	host.flags = undefined;
	// Only the newest reply (the last message is the one answered again): not the one shown while the next reads its prompt.
	onEvent(live, { type: "start", rid: "r2", model: "Flash" });
	d.handleInput(KEY.enter);
	d.handleInput(KEY.bs);
	d.handleInput(KEY.enter);
	assert.deepEqual([keyLine(), edited.length], [LABELS.editNewest, 2]);
});

test("telescope: the edit says where it acts, after the cursor where it fits: a swap's push layer, a removal's band", () => {
	const row = (field: string | null, w = 44, band: [number, number] | null = [18, 44]) =>
		plain(telescope(theme, token(), 3, w, 60, { edit: { row: 1, field, band: band ?? undefined } }).lines[3]);
	assert.match(row("rugby"), /^│▸audience → rugby  at L31 +8% {2}│$/);                   // the band's middle: pushLayer
	assert.match(row(""), /^│▸audience removed  at L18-44 +8% {2}│$/);
	assert.match(row(null), /^│▸audience +8% {2}│$/);                                       // untouched: nothing yet
	assert.match(row("rugby", 44, null), /^│▸audience → rugby  +8% {2}│$/);           // no band known: nothing
	assert.match(row("rugby", 44, [15, 36]), /→ rugby  at L26 /);                            // the 35B's band
	assert.match(row("x".repeat(20)), /x  +8%/);                                             // no room: the field first
	for (const w of [36, 44, 60]) for (const field of [null, "", "rugby", "x".repeat(40)])
		assert.equal(visibleWidth(row(field, w)), w);
});

test("dock, LENS on the one-panel board: a word that is no name (读者) is edited by its token id, shown as itself; whitespace stays refused", async () => {
	const { d, host } = lensDock({ ...WORDS, 9: " " });   // the logit lens's L24: " "
	const edited: any[] = [], asked: (number | undefined)[] = [];
	host.edit = async (a: string, ids: any, at: any) => (edited.push([a, ids, at]), []);
	// The lens steps' answer, as the dock asks it anyway: where the edit acts (the 35B's band).
	host.lensLayers = async (step?: number) => (asked.push(step), { model: "Flash", at: 0, steps: [{ name: "small", n: 4, layers: [24, 36, 40, 44, 46, 47], tps: null, replies: 0 }], edit_band: [15, 36] });
	d.sel = 3;
	host.lensLayer = 44;
	const eyeRows = () => text(d.render(48)).filter((l) => /^│[ ▸]/.test(l)).slice(0, 3);
	eyeRows();
	await tick();
	d.handleInput(KEY.enter);
	await tick();
	assert.deepEqual([asked, edited], [[undefined], []]);                                    // opening asks nothing
	d.handleInput(KEY.tab);
	d.handleInput(KEY.tab);
	d.handleInput(KEY.bs);
	assert.match(eyeRows()[2], /^│▸读者 removed  at L15-36 /);
	d.handleInput(KEY.enter);
	assert.deepEqual(edited.at(-1), ["-tok8", { tok8: { id: 8, text: "读者" } }, { rid: "r", at: 3, word: "读者" }]);
	await answered(d);
	d.handleInput(KEY.enter);
	d.handleInput(KEY.tab);
	d.handleInput(KEY.tab);
	for (const c of "vague") d.handleInput(c);
	assert.match(eyeRows()[2], /^│▸读者 → vague  at L26 /);
	d.handleInput(KEY.enter);
	assert.deepEqual(edited.at(-1), ["tok8>vague@0.7", { tok8: { id: 8, text: "读者" } }, { rid: "r", at: 3, word: "读者", want: "vague" }]);
	// A swap into a lens word that is no name: it goes by its id too.
	await answered(d);
	d.handleInput(KEY.enter);
	d.handleInput(KEY.down);
	d.handleInput(KEY.down);
	assert.match(eyeRows()[0], /^│▸reader → 读者/);
	d.handleInput(KEY.enter);
	assert.deepEqual(edited.at(-1), ["tok42>tok8@0.7", { tok42: { id: 42, text: "reader" }, tok8: { id: 8, text: "读者" } },
		{ rid: "r", at: 3, word: "reader", want: "读者" }]);
	assert.deepEqual(asked, [undefined]);
	// Whitespace alone is no word: refused as before, nothing sent.
	await answered(d);
	host.lensView = "logit";
	host.lensLayer = 24;
	d.handleInput(KEY.enter);
	d.handleInput(KEY.bs);
	d.handleInput(KEY.enter);
	assert.equal(text(d.render(48)).at(-1), '" " is not a word');
	assert.equal(edited.length, 3);
});

test("dock, LENS on the one-panel board: lens steps without a J-lens (no edit_band), or another model's: no label, the edit as before", async () => {
	for (const data of [{ model: "Flash", at: 0, steps: [{ name: "small", n: 4, layers: [24, 36, 40, 44, 46, 47], tps: null, replies: 0 }] },
		{ model: "Other", at: 0, steps: [], edit_band: [15, 36] }]) {
		const { d, host } = lensDock();
		const edited: any[] = [];
		host.edit = async (a: string, ids: any, at: any) => (edited.push([a, ids, at]), []);
		host.lensLayers = async () => data;
		d.sel = 3;
		host.lensLayer = 44;
		d.render(48);
		await tick();
		d.handleInput(KEY.enter);
		for (const c of "rugby") d.handleInput(c);
		const row = text(d.render(48)).find((l) => /^│▸reader → rugby/.test(l))!;
		assert.ok(row && !/ at L/.test(row), row);
		d.handleInput(KEY.enter);
		assert.deepEqual(edited, [["tok42>rugby@0.7", { tok42: { id: 42, text: "reader" } }, { rid: "r", at: 3, word: "reader", want: "rugby" }]]);
	}
});

test("telescope: the field is drawn as the lens words are (a newline shown ↵) and cut by width, wide characters too", () => {
	const row = (field: string, w: number) => plain(telescope(theme, token(), 3, w, 60, { edit: { row: 1, field, band: [18, 44] } }).lines[3]);
	for (const w of [36, 44]) {
		const nl = row(".\n\n##", w), wide = row("读者".repeat(12), w);
		assert.ok(!nl.includes("\n") && nl.includes(".↵↵##"), nl);
		assert.equal(visibleWidth(nl), w);
		assert.equal(visibleWidth(wide), w, wide);
		assert.match(wide, /→ [读者]+ +8%/);                                        // its end, whole characters
	}
});

test("dock, LENS on the one-panel board: every lens word goes by its id (a number, a concept's name, a plain word filled in); typed: by name, a number refused", async () => {
	const { d, host } = lensDock({ ...WORDS, 7: " 2025", 8: " calm" });
	const edited: any[] = [];
	host.edit = async (a: string, ids: any, at: any) => (edited.push([a, ids, at]), []);
	d.sel = 3;
	host.lensLayer = 44;
	d.render(48);
	d.handleInput(KEY.enter);
	d.handleInput(KEY.tab);                                                          // 2025: a number, by its id
	d.handleInput(KEY.bs);
	d.handleInput(KEY.enter);
	assert.deepEqual(edited.at(-1), ["-tok7", { tok7: { id: 7, text: "2025" } }, { rid: "r", at: 3, word: "2025" }]);
	await answered(d);
	d.handleInput(KEY.enter);
	d.handleInput(KEY.down);
	d.handleInput(KEY.down);                                                         // calm: a concept's name here, filled in
	d.handleInput(KEY.enter);
	assert.deepEqual(edited.at(-1), ["tok42>tok8@0.7", { tok42: { id: 42, text: "reader" }, tok8: { id: 8, text: "calm" } },
		{ rid: "r", at: 3, word: "reader", want: "calm" }]);
	await answered(d);
	d.handleInput(KEY.enter);
	for (const c of "2025") d.handleInput(c);                                        // typed: no id
	d.handleInput(KEY.enter);
	assert.equal(text(d.render(48)).at(-1), '"2025" is a number, not a word');
	d.handleInput(KEY.down);
	d.handleInput(KEY.down);
	d.handleInput(KEY.bs);                                                           // a fill typed over: by its name
	d.handleInput(KEY.enter);
	assert.deepEqual(edited.at(-1), ["tok42>cal@0.7", { tok42: { id: 42, text: "reader" } }, { rid: "r", at: 3, word: "reader", want: "cal" }]);
	assert.equal(edited.length, 3);
});

test("dock, LENS on the one-panel board: editing while a reply streams holds its token", () => {
	const { d, live } = lensDock();
	live.reply!.done = false;
	d.stepIn(-1);
	d.handleInput(KEY.enter);
	assert.equal(d.sel, 5);
});

/** The server's lens sizes as GET /scope/api/lens/layers answers (the reply was read at medium: J 36 40 44 46, logit
 * 24 44 47), and a host that answers it, `step` setting the one in use. LENS full size on the board (104 wide): the
 * size's line under its title. */
function depthDock() {
	const x = lensDock(), asked: (number | undefined)[] = [];
	x.d.full = true;
	const data = { model: "Flash", at: 1, steps: [
		{ name: "small", n: 4, layers: [24, 44, 46, 47], tps: 52.4, replies: 1 }, { name: "medium", n: 6, layers: [24, 36, 40, 44, 46, 47], tps: 41.2, replies: 5 },
		{ name: "large", n: 48, layers: Array.from({ length: 48 }, (_, l) => l), tps: null, replies: 0 }] };
	x.host.lensLayers = async (step?: number) => {
		asked.push(step);
		if (step !== undefined) data.at = step;
		return structuredClone(data);
	};
	return { ...x, asked };
}
/** The full size board's tab row: the step's line under it. */
const tabAt = (lines: string[]) => lines.findIndex((l) => /^ ?1 TOKENS +2 MAP /.test(l));
const tubeRows = (lines: string[]) => lines.filter((l) => /^ *│[ ▸]L\d+/.test(l)).map((l) => /L\d+/.exec(l)![0]);

test("dock, LENS full size on the board: - + step the layers read: pips, the count and this setting's measured speed; the tube follows", async () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const { d, asked } = depthDock();
		d.sel = 3;
		d.render(104);
		await tick();
		let lines = text(d.render(104));
		if (process.env.SHOW_DOCK) console.log(lines.join("\n"));
		const at = tabAt(lines);
		assert.match(lines[at + 1], /^▮▮▯  medium · reads 6 layers +41 tok\/s · last 5 replies$/);
		assert.equal(visibleWidth(lines[at + 1]), 101, "the speed at the right edge");
		assert.deepEqual(tubeRows(lines), ["L47", "L46", "L44", "L40", "L36", "L24"]);
		assert.match(lines.slice(-3).join(" "), /↑\/↓ layer · -\/\+ layers · enter/);
		const footerAt = lines.length;
		// -: fewer layers, at once and set on the server; the tube retracts over a few frames.
		d.handleInput("-");
		assert.equal(tubeRows(text(d.render(104))).length, 5, "the first frame a quarter of the way: one of the two going gone");
		mock.timers.tick(70 * 6);
		await tick();
		lines = text(d.render(104));
		assert.deepEqual(asked, [undefined, 0]);
		assert.match(lines[at + 1], /^▮▯▯  small · reads 4 layers +52 tok\/s · last reply$/);
		assert.deepEqual(tubeRows(lines), ["L47", "L46", "L44", "L24"], "L40 and L36 hidden: going down hides at once");
		assert.equal(lines.length, footerAt, "the page keeps its height");
		// + twice: layers no reply read yet come in as faint blank lines, no speed yet.
		d.handleInput("+");
		d.handleInput("=");
		mock.timers.tick(70 * 60);
		await tick();
		const big = d.render(104);
		lines = text(big);
		assert.match(lines[at + 1], /^▮▮▮  large · reads 48 layers +no reply at this setting yet$/);
		assert.ok(["L47", "L46", "L44", "L40", "L36"].every((l) => tubeRows(lines).includes(l)), "the layers read");
		assert.ok(unread(big) > 5, "the layers wanted, not read yet: faint");
		d.handleInput("+");                                                         // past the last: nothing
		mock.timers.tick(70 * 6);
		await tick();
		assert.equal(asked.at(-1), 2);
		assert.ok(text(d.render(104)).every((l) => visibleWidth(l) <= 101));
		// While a word is edited, - is a letter of it.
		d.handleInput(KEY.enter);
		d.handleInput("-");
		assert.equal(asked.at(-1), 2);
		assert.match(text(d.render(104)).join("\n"), /→ -/);
		d.dispose();
	} finally {
		mock.timers.reset();
	}
});

test("dock, LENS full size on the board: the lens steps asked again once a reply is saved; a failure said in their line", async () => {
	const { d, host, asked } = depthDock();
	d.sel = 3;
	d.render(104);
	await tick();
	assert.deepEqual(asked, [undefined]);
	d.render(104);
	assert.deepEqual(asked, [undefined], "asked once a reply");
	host.live.reply!.traceId = "T2";
	d.render(104);
	assert.deepEqual(asked, [undefined, undefined]);
	host.lensLayers = async () => ({ error: "oMLX at h answered: Not Found" });
	host.live.reply!.traceId = "T3";
	d.render(104);
	await tick();
	const at = tabAt(text(d.render(104)));
	assert.equal(text(d.render(104))[at + 1], "oMLX at h answered: Not Found");
	d.handleInput("-");                                                           // no steps known: nothing happens
	d.dispose();
});

test("dock, LENS full size on the board: one step set at a time, the last pressed sent when it returns; a failure shows what the server holds", async () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const { d, host } = depthDock();
		const sent: (number | undefined)[] = [], answers: ((v: any) => void)[] = [];
		let server = 0;
		const steps = (at: number) => ({ model: "Flash", at, steps: ["small", "medium", "large"].map((name, i) => ({ name, n: 4 + i, layers: [44, 46, 47].slice(0, 1 + i), tps: null, replies: 0 })) });
		host.lensLayers = (step?: number) => {
			sent.push(step);
			return new Promise((res) => answers.push((v: any) => {
				if (v) return res(v);
				if (step !== undefined) server = step;
				res(steps(server));
			}));
		};
		d.sel = 3;
		d.render(104);
		answers.shift()!(undefined);
		await tick();
		const at = tabAt(text(d.render(104))), pips = () => text(d.render(104))[at + 1].slice(0, 3);
		d.handleInput("+");
		d.handleInput("+");
		d.handleInput("-");
		d.handleInput("+");
		assert.deepEqual(sent, [undefined, 1], "one in flight; the presses since wait");
		assert.equal(pips(), "▮▮▮", "drawn at once");
		answers.shift()!(undefined);
		await tick();
		assert.deepEqual(sent, [undefined, 1, 2], "the last pressed, sent when the first returned");
		answers.shift()!(undefined);
		await tick();
		assert.equal(server, 2);
		assert.equal(pips(), "▮▮▮");
		// A failure: asked again, and what the server holds shown.
		d.handleInput("-");
		assert.equal(pips(), "▮▮▯");
		answers.shift()!({ error: "oMLX at h answered: busy" });
		await tick();
		assert.deepEqual(sent.slice(-2), [1, undefined]);
		answers.shift()!(undefined);
		await tick();
		assert.equal(pips(), "▮▮▮", "the server still at large");
		d.dispose();
	} finally {
		mock.timers.reset();
	}
});

test("dock, LENS full size on the board: a server without the route (404) leaves the step line blank and no -/+ hint; other failures said", async () => {
	const { d, host } = depthDock();
	host.lensLayers = async () => ({ error: "oMLX at h answered: Not Found", missing: true });
	d.sel = 3;
	d.render(104);
	await tick();
	const lines = text(d.render(104)), at = tabAt(lines);
	assert.equal(lines[at + 1], "", "the line kept, blank");
	assert.ok(lines.slice(at + 2).some((l) => /reader/.test(l)), "the telescope under it");
	assert.ok(!lines.slice(-3).join(" ").includes("-/+ layers"));
	d.handleInput("-");
	assert.equal(text(d.render(104))[at + 1], "");
	d.dispose();
});

test("dock, LENS full size on the board: the step shapes the tube of the reply shown, kept where it is while the next prompt is read", async () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const { d, live, asked } = depthDock();
		d.sel = 3;
		d.render(104);
		await tick();
		d.handleInput("+");
		await tick();
		mock.timers.tick(70 * 60);
		await tick();
		const newestRaw = d.render(104), newest = text(newestRaw);
		assert.equal(asked.at(-1), 2);
		assert.ok(unread(newestRaw) > 5, "the newest: the step's layers (a short page: shared lines), those not read faint");
		// The next request reading its prompt: the last reply shown, its tube still the step's.
		onEvent(live, { type: "start", rid: "r2", model: "Flash" });
		assert.deepEqual(tubeRows(text(d.render(104))), tubeRows(newest), "not back over the layers it read");
		d.dispose();
	} finally {
		mock.timers.reset();
	}
});

test("scopeHeight: the telescope's lines drawn in full, whatever the token read (LENS sized on it: a reply read without the lenses the same height)", () => {
	for (const t of [token(), deepToken()])
		for (const want of [undefined, [24, 31, 38, 46, 47], [8, 16, 24, 31, 38, 46, 47], [...Array(48).keys()]]) {
			const drawn = telescope(theme, t, 0, 60, 999, { want }).lines.length;
			assert.equal(drawn, scopeHeight(tubeOf(t, want)), `${t.lens!.length} layers read, want ${want?.length}`);
		}
});

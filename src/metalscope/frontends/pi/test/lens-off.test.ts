import assert from "node:assert/strict";
import { test } from "node:test";
import { DULL, MAP_CELL_LINES, MINI_WIDTH, ROWS_A_TICK, SWEEP_MAX_TICKS, boardKeys, dull, dullLines, dullPace, PANELS } from "../board.ts";
import { Dock, LENS_OFF } from "../dock.ts";
import { SWEEP_TICK_MS } from "../lookup.ts";
import { COLOURS, LABELS, colourCode, dockTheme } from "../draw.ts";
import { mix } from "../lights.ts";
import { colorToRgb, indexedColor } from "@earendil-works/pi-tui";
import { type Live, onEvent } from "../state.ts";

// The lenses off: - past the fewest layers (LENS_OFF), + the fewest again. LENS stays where it is, dulled; a reply read
// with them off is dulled too, saying so; the dulling on the sweep's ticker, up from LENS's bottom; nothing else moves.

const plain = (s: string) => s.replace(/\x1b\[[0-9;:]*m/g, "");
const text = (lines: string[]) => lines.map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd());
const tick = () => new Promise((r) => setTimeout(r, 0));
const LW = 100 - 3 - 3 - 50;   // the left column at 100 columns (board.ts columns)
const left = (l: string) => l.slice(0, LW).trimEnd();
/** The dulled text colour's escape: a line of LENS's left column that has it is dulled. */
const INK = dull(dockTheme(), "", DULL).slice(0, -"\x1b[39m".length);
const SEP = dockTheme().fg("rule", " │ "), leftRaw = (l: string) => l.slice(0, Math.max(0, l.indexOf(SEP)));
const dulledRows = (raw: string[], top: number) => raw.map((l, i) => i > top && i < raw.length - 2 && leftRaw(l).includes(INK));

/** Reply token `i` of a 48-layer model, read by the J-lens at every layer (`lens` false: read with the lenses off). */
function tok(i: number, lens = true): any {
	const experts: number[] = [], w: number[] = [];
	for (let l = 0; l < 48; l++) for (let j = 0; j < 8; j++) { experts.push((i * 7 + l * 13 + j * 31) % 384); w.push(j ? 20 : 120); }
	return { type: "token", rid: "r", i, id: 100 + i, text: ` w${i}`, p: i % 5 ? 0.95 : 0.3, entropy: 0.5,
		alts: Array.from({ length: 9 }, (_, a) => [` a${a}`, 0.04 - a * 0.004]), n_layers: 48, experts, expert_w: w, expert_k: 8,
		...(lens ? { lens: [...Array(48).keys()].map((l) => [l, l >= 42 ? 100 + i : 7, l >= 42 ? ` w${i}` : " the"]), lens_by: "jlens" } : {}) };
}
const STEPS = [[24, 31, 38, 46, 47], [8, 16, 24, 31, 38, 46, 47], [0, 8, 16, 24, 31, 36, 40, 44, 46, 47], [...Array(24).keys()].map((l) => 2 * l), [...Array(48).keys()]];
/** The server's five steps on a 48-layer model: 5, 9, 22, 29 and 48 layers, the J-lens reading `n` of them (REAL_N). */
const REAL = [[24, 31, 38, 46, 47], [4, 16, 24, 28, 31, 38, 40, 46, 47], [0, 4, 8, 12, 16, 20, 24, 28, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
	[0, 2, 4, 7, 8, 11, 12, 16, 17, 20, 22, 24, 26, 28, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], [...Array(48).keys()]], REAL_N = [4, 8, 15, 24, 47];

/** The board 100 wide, `rows` tall, a token held, LENS in focus, the server's lens steps at `at` (POSTs in `sent`). */
async function dockAt(rows: number, at: number, o: { lens?: boolean; read?: number[]; clock?: { t: number }; steer?: boolean; panel?: string; steps?: number[][]; ns?: number[]; full?: boolean } = {}) {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r", model: "M" });
	for (let i = 0; i < 40; i++) {
		const t = tok(i, o.lens ?? true);
		if (o.read && t.lens) t.lens = t.lens.filter(([l]: [number]) => o.read!.includes(l));   // read at those layers only
		onEvent(live, t);
	}
	onEvent(live, { type: "end", rid: "r", n_tokens: 40, trace_id: "T" });
	if (o.steer) live.steer = { parts: [{ name: "calm", layer: 20, strength: 0.4 }] } as any;
	const data = { model: "M", at, steps: (o.steps ?? STEPS).map((layers, i) => ({ n: o.ns?.[i] ?? layers.length, layers, tps: null, replies: 0 })) }, sent: number[] = [], said: string[] = [];
	const grid = { cols: 32, rows: 12, pos: Array.from({ length: 384 }, (_, e) => e), num_experts: 384, n_layers: 48 };
	const host: any = { live, captions: false, close() {}, unfocus() {}, panel: o.panel ?? "LENS", steerModel: () => "M",
		fetchJson: async (url: string) => (url.includes("/experts/grid") ? grid : null), now: o.clock && (() => o.clock!.t),
		steer: async (a: string) => (said.push(a), []), steering: async () => ({ model: "M", concepts: [] }),
		lensLayers: async (step?: number) => (step !== undefined && (sent.push(step), (data.at = step)), structuredClone(data)) };
	const d = new Dock({ terminal: { rows, columns: 200 }, requestRender() {} }, { getColorMode: () => "truecolor" }, host);
	d.stepIn(30);
	if (o.full) d.handleInput("2");
	d.render(100);
	await tick();
	d.render(100);
	return { d, host, sent, said, data };
}
const lensRow = (lines: string[]) => lines.findIndex((l) => /^ ?LENS /.test(l));
/** Every line but LENS's rows in the left column (its title to the footer's rule) and the key line: as `a`. */
function still(a: string[], b: string[], what: string) {
	const top = lensRow(text(a));
	assert.ok(top > 0, what);
	for (let i = 0; i < a.length - 1; i++) {
		if (i < top || i >= a.length - 2) assert.equal(b[i], a[i], `${what}: row ${i}`);
		else assert.equal(b[i].slice(b[i].indexOf(SEP)), a[i].slice(a[i].indexOf(SEP)), `${what}: row ${i}, the right column`);
	}
}

test("lenses off: - past the fewest layers dulls LENS where it is, its text as before, nothing else moved; + brings it back", async () => {
	const { d, sent } = await dockAt(52, 0);
	const open = d.render(100), was = text(open), top = lensRow(was);
	assert.match(left(was[top]), /^ LENS +J-lens +▮▯▯▯▯  reads 5 layers$/);
	assert.match(was.at(-1)!, /↑\/↓ layer · -\/\+ layers · enter edit/);
	assert.ok(!dulledRows(open, top).some(Boolean));
	d.handleInput("-");
	await tick();                                                                             // dulled once the server has it
	const off = d.render(100), lines = text(off);
	assert.deepEqual(sent, [LENS_OFF], "set on the server");
	assert.match(left(lines[top]), /^ LENS {10,}▯▯▯▯▯  off$/, "the pips unlit, off");
	for (let i = top + 1; i < lines.length - 2; i++) assert.equal(left(lines[i]), left(was[i]), `row ${i}: the same text`);
	assert.ok(dulledRows(off, top).filter(Boolean).length > 10, "dulled");
	assert.match(lines.at(-1)!, /^tab tokens · ←\/→ token · f live · \+ lenses · /);
	assert.ok(!/↑\/↓ layer|enter edit|s other lens|-\/\+/.test(lines.at(-1)!));
	still(open, off, "off");
	d.handleInput("-");                                                                       // past off: nothing
	assert.deepEqual(sent, [LENS_OFF]);
	// Off, LENS's reading keys do nothing; + brings back the fewest layers.
	for (const k of ["s", "\x1b[A", "\r"]) d.handleInput(k);
	assert.deepEqual(text(d.render(100)), lines);
	d.handleInput("+");
	await tick();
	assert.deepEqual(sent, [LENS_OFF, 0]);
	assert.deepEqual(text(d.render(100)), was, "as before");
	d.dispose();
});

test("- + move nothing outside LENS: at 52, 76 and 100 rows every step and off draw the same lines outside LENS's rows; LENS's title, SETTLING under it and the telescope's first line fixed, the lines a short tube frees blank at the bottom", async () => {
	const settling = (lines: string[]) => lines.findIndex((l) => /^ ?SETTLING /.test(left(l)));
	const scope = (lines: string[]) => lines.findIndex((l) => /^#\d+ "/.test(left(l)));
	const last = (ls: string[], sub: number) => sub + ls.slice(sub + 1).findIndex((l) => !/▀/.test(left(l)));
	for (const rows of [52, 76, 100]) {
		const frames: string[][] = [];
		for (const at of [LENS_OFF, 0, 1, 2, 3, 4]) {
			const { d } = await dockAt(rows, at, { steps: REAL, ns: REAL_N });
			frames.push(d.render(100));
			d.dispose();
		}
		const ref = text(frames[3]), top = lensRow(ref), sub = settling(ref), tel = scope(ref), end = ref.length - 3;
		assert.ok(top > 0 && sub === top + 1 && tel > sub, `${rows}: ${top} ${sub} ${tel}`);
		// SETTLING no more than its want (the middle step's: its sub-heading and 8 lines) while MAP can still grow; LENS
		// grows past it only after (GROW_ORDER).
		const mapLines = top - 1 - ref.findIndex((l) => /^ ?MAP /.test(left(l))), sl = last(ref, sub) - sub + 1;
		assert.ok(mapLines === 1 + MAP_CELL_LINES || sl <= 9, `${rows}: MAP ${mapLines} lines, SETTLING ${sl}`);
		if (rows >= 76) assert.ok(sl >= 9, `${rows}: SETTLING its want, ${sl} lines`);
		if (rows === 100) assert.ok(mapLines === 1 + MAP_CELL_LINES && sl > 9, `${rows}: MAP its most, then SETTLING grows: ${mapLines}, ${sl}`);
		// SETTLING over the step's layers: at the fewest, the five it reads, each a band labelled once.
		const labels = (ls: string[]) => ls.slice(sub + 1, last(ls, sub) + 1).map((l) => /^L\d+/.exec(left(l))?.[0]).filter(Boolean);
		if (rows >= 76) assert.deepEqual(labels(text(frames[1])), ["L47", "L46", "L38", "L31", "L24"], `${rows}: step 0's layers`);
		for (const [k, f] of frames.entries()) {
			const what = `${rows} rows, ${k ? `step ${k - 1}` : "off"}`, ls = text(f);
			still(frames[3], f, what);                                 // the key line aside: off has its own keys
			assert.equal(lensRow(ls), top, `${what}: LENS's title`);
			assert.equal(settling(ls), sub, `${what}: SETTLING's sub-heading`);
			assert.equal(last(ls, sub), last(ref, sub), `${what}: SETTLING's last line`);
			assert.equal(scope(ls), tel, `${what}: the telescope's first line`);
			// Blank in LENS: the line between SETTLING and the telescope, then a run at the bottom (none with the tube whole).
			const blank = ls.slice(top + 1, end + 1).map((l, i) => (!left(l) ? top + 1 + i : -1)).filter((i) => i >= 0);
			const foot = blank.filter((b) => b !== tel - 1);
			assert.ok(blank.includes(tel - 1) && foot.every((b, i) => b === end - foot.length + 1 + i), `${what}: blank ${blank}`);
			if (k >= 3) assert.equal(foot.length, 0, `${what}: from the default step, nothing blank at the bottom`);
		}
	}
});

test("- + move nothing on LENS full size (2): at 36, 40, 52, 76 and 100 rows every step and off keep the step's line, SETTLING's lines and the lenses' first line where they are; only the tubes lengthen, the lines a short tube frees blank at the foot; SETTLING its want first, the tubes' box the rest, shared at the shallow end, never folded", async () => {
	const at = (ls: string[], re: RegExp) => ls.findIndex((l) => re.test(l));
	const parts = (ls: string[]) => {
		const step = at(ls, /^[▮▯]+ +(reads|off)/), sub = at(ls, /^SETTLING /), names = at(ls, /^J-lens\b/), rule = ls.length - 4;
		return { step, sub, last: sub + ls.slice(sub + 1).findIndex((l) => !/▀/.test(l)), names, rule };
	};
	for (const rows of [36, 40, 52, 76, 100]) {
		const frames: string[][] = [];
		for (const s of [LENS_OFF, 0, 1, 2, 3, 4]) {
			const { d } = await dockAt(rows, s, { steps: REAL, ns: REAL_N, full: true });
			frames.push(text(d.render(100)));
			d.dispose();
		}
		const ref = frames[3], p = parts(ref);
		assert.ok(p.step > 0 && p.sub === p.step + 1 && p.last > p.sub + 2 && p.names === p.last + 2 && /^─+$/.test(ref[p.rule]), `${rows}: ${JSON.stringify(p)}`);
		// SETTLING its want (the middle step's: its sub-heading and 8 lines) where the page holds it.
		if (rows >= 76) assert.ok(p.last - p.sub >= 8, `${rows}: SETTLING ${p.last - p.sub + 1} lines`);
		for (const [k, ls] of frames.entries()) {
			const what = `${rows} rows, ${k ? `step ${k - 1}` : "off"}`;
			assert.deepEqual(parts(ls), p, `${what}: the step's line, SETTLING, the lenses and the rule where they were`);
			assert.deepEqual(ls.slice(0, p.step), ref.slice(0, p.step), `${what}: the lines over LENS`);
			assert.deepEqual(ls.slice(p.rule, -1), ref.slice(p.rule, -1), `${what}: the rule and the legend (the key line aside)`);
			// Blank: the line over the lenses, then one run at the foot; none with the largest step.
			const blank = ls.map((l, i) => (i > p.step && i < p.rule && !l ? i : -1)).filter((i) => i >= 0), foot = blank.filter((b) => b !== p.names - 1);
			assert.ok(blank.includes(p.names - 1) && foot.every((b, i) => b === p.rule - foot.length + i), `${what}: blank ${blank}`);
			if (k === 5) assert.equal(foot.length, 0, `${what}: the largest step's tubes fill the page`);
			assert.ok(!ls.some((l) => /shallower layer/.test(l)), `${what}: shared lines, no fold`);
		}
		const tube = (ls: string[]) => ls.slice(p.names, p.rule).filter((l) => /│[ ▸]L\d+/.test(l)), shared = (ls: string[]) => tube(ls).filter((l) => /^ *│[ ▸]L\d+ \+\d/.test(l)).length;
		assert.deepEqual(frames.slice(1).map((f) => tube(f).length).map((n, i, a) => !i || n >= a[i - 1]), [true, true, true, true, true], `${rows}: the tubes longer step by step`);
		assert.equal(shared(frames[5]) === 0, rows === 100, `${rows}: the largest step a layer a line only where the page holds it`);
	}
});

test("a reply read with the lenses off, LENS full size (2): off, the fewest step's outline dulled under SETTLING; + sends step 0 and lights it in place, not read", async () => {
	const { d, sent } = await dockAt(76, LENS_OFF, { lens: false, full: true });
	const faint = (ls: string[]) => ls.filter((l) => /^ *│ L\d+ +· +│/.test(l)).map((l) => /L\d+/.exec(l)![0]);
	let raw = d.render(100), lines = text(raw);
	const step = lines.findIndex((l) => /^▯+ +off$/.test(l)), names = lines.findIndex((l) => /^not read for this reply$/.test(l));   // titled as the board titles the outline
	assert.ok(step > 0 && lines[step + 1].startsWith("SETTLING ") && names > step + 3 && !lines[names - 1], "SETTLING under the step's line, a blank line, the lens");
	assert.ok(lines.slice(step + 2, names - 1).every((l) => (l.match(/▀/g) ?? []).length >= 40), "a cell a token on every line: unlit, not blank");
	assert.deepEqual(faint(lines), ["L47", "L46", "L38", "L31", "L24"], "the fewest step's outline, every row faint");
	assert.ok(raw.slice(step + 1, -4).filter((l) => l.includes(INK)).length > 20, "dulled");
	d.handleInput("+");
	await tick();
	assert.deepEqual(sent, [0], "+: the fewest step");
	raw = d.render(100);
	const on = text(raw);
	assert.match(on[step], /^▮▯▯▯▯ /);
	assert.ok(on[step + 1].startsWith("SETTLING ") && on[names] === "not read for this reply", "nothing moved");
	assert.deepEqual(faint(on), faint(lines), "the same outline");
	assert.ok(!raw.slice(step + 1, -4).some((l) => l.includes(INK)), "lit");
	d.dispose();
});

// `under`: the column's lines left blank under LENS (one step's LENS at its most, MAP at its 7).
for (const [what, rows, full, steps, under] of [["the board at 52 rows, one step", 52, false, STEPS.slice(0, 1), 4], ["the board at 100 rows", 100, false, STEPS, 0], ["LENS full size at 76 rows", 76, true, STEPS, 0]] as const)
	test(`lenses off, ${what}: the sweep on the sweep's ticker, one edge up from LENS's bottom, ROWS_A_TICK rows a tick (a multiple of it tall), SWEEP_MAX_TICKS at most; back on, down from the top; a key ends it`, async () => {
		const clock = { t: 1000 }, { d } = await dockAt(rows, 0, { clock, steps: [...steps], full });
		const open = d.render(100), lines = text(open);
		const top = full ? lines.findIndex((l) => /^[▮▯]+ +reads/.test(l)) : lensRow(lines);   // LENS's first line kept
		const part = (l: string) => (full ? l : leftRaw(l));
		assert.ok(top > 0 && lines[top + 1].startsWith("SETTLING "), "SETTLING under LENS's first line");
		d.handleInput("-");
		await tick();
		const frames: string[][] = [d.render(100)], ticks = (d as any).engram.sweep.lens.ms / SWEEP_TICK_MS;
		for (let k = 1; k <= ticks; k++) (clock.t += SWEEP_TICK_MS), frames.push(d.render(100));
		assert.equal((d as any).engram.sweep, undefined, "over: the ticker gone");
		// LENS's last line: the last dulled once off (its blank lines too; on the board, not the column's under it).
		const off = frames.at(-1)!;
		let end = off.length - 1;
		while (end > top && !part(off[end]).includes(INK)) end--;
		const n = end - top, pace = dullPace(n);
		assert.equal(end, lines.length - 3 - under, `LENS down to the column's bottom, less ${under} blank lines`);
		assert.ok(lines.slice(end + 1, end + 1 + under).every((l) => !l.split("│")[0].trim()), "the lines under it blank");
		const at = (f: string[]) => f.map((l, i) => i > top && i <= end && part(l).includes(INK)).slice(top + 1, end + 1);
		const fromBottom = (f: string[]) => { const x = at(f), c = x.filter(Boolean).length; return x.every((b, i) => b === i >= n - c) ? c : -1; };
		assert.equal(ticks, pace.ticks);
		assert.ok(ticks <= SWEEP_MAX_TICKS && pace.rows % ROWS_A_TICK === 0 && (n > ROWS_A_TICK * SWEEP_MAX_TICKS) === (pace.rows > ROWS_A_TICK), `${n} lines: ${JSON.stringify(pace)}`);
		for (const [k, f] of frames.entries()) {
			if (!full) still(open, f, `frame ${k}`);
			assert.deepEqual(text(f).slice(0, top), lines.slice(0, top), `frame ${k}: the lines over LENS`);
			assert.deepEqual(text(f).slice(top + 1, end + 1).map(full ? (l) => l : left), lines.slice(top + 1, end + 1).map(full ? (l) => l : left), `frame ${k}: LENS's text kept`);
			assert.match(full ? text(f)[top] : left(text(f)[top]), /▯+ +off$/, "the step off from the first frame");
			assert.equal(fromBottom(f), Math.min(n, (k + 1) * pace.rows), `frame ${k}: one edge up from the bottom`);
		}
		// +: from the top down; any key ends it at once.
		d.handleInput("+");
		await tick();
		const back = [d.render(100)];
		for (let k = 1; k < ticks; k++) (clock.t += SWEEP_TICK_MS), back.push(d.render(100));
		const left0 = back.map(fromBottom);
		assert.ok(left0.every((c, k) => c >= 0 && (!k || c <= left0[k - 1])) && left0[0] < n && left0.at(-1)! <= pace.rows, `the top lit first, down: ${left0}`);
		d.handleInput("y");                                                                       // a key that does nothing else
		assert.deepEqual(d.render(100), open);
		d.dispose();
	});

test("lenses off: a server that refuses off (400) leaves LENS as it was, no sweep, and says why", async () => {
	const clock = { t: 1000 }, { d, host } = await dockAt(52, 0, { clock });
	const open = d.render(100), get = host.lensLayers;
	host.lensLayers = async (step?: number) => (step === undefined ? get() : Promise.reject(new Error("HTTP 400: lens_step must be 0 to 4")));
	d.handleInput("-");
	await tick();
	const now = d.render(100);
	assert.equal((d as any).engram.sweep, undefined, "no sweep");
	assert.deepEqual(now.slice(0, -1), open.slice(0, -1), "LENS as it was");
	assert.match(text(now).at(-1)!, /HTTP 400: lens_step must be 0 to 4/);
	d.handleInput("y");
	assert.deepEqual(d.render(100), open, "the note gone with the next key");
	d.dispose();
});

test("lenses off: a reply traced while off stays dulled, saying off, with no sweep", async () => {
	const clock = { t: 1000 }, { d, host } = await dockAt(52, LENS_OFF, { clock });
	d.render(100);
	const live = host.live;
	onEvent(live, { type: "start", rid: "r2", model: "M" });
	for (let i = 0; i < 5; i++) onEvent(live, { ...tok(i, false), rid: "r2" });
	onEvent(live, { type: "end", rid: "r2", n_tokens: 5, trace_id: "T2" });
	const raw = d.render(100), lines = text(raw), top = lensRow(lines);
	assert.match(left(lines[top]), /▯▯▯▯▯  off$/, "off, not 'not read'");
	assert.equal((d as any).engram.sweep, undefined, "no sweep: LENS was dulled already");
	assert.ok(dulledRows(raw, top).some(Boolean));
	d.dispose();
});

test("lenses off: the lens steps' first answer dulls LENS without the sweep; a reply read with them off keeps the layout and says why", async () => {
	const clock = { t: 0 }, on = (await dockAt(52, 0, { clock })).d, off = (await dockAt(52, LENS_OFF, { clock })).d;
	assert.equal((off as any).engram.sweep, undefined, "no sweep for the first answer");
	const raw = text(off.render(100));
	assert.match(left(raw[lensRow(raw)]), /off$/);
	const { d, sent } = await dockAt(52, 0, { lens: false, clock });
	const lines = text(d.render(100)), was = on.render(100), top = lensRow(text(was));
	assert.equal(lensRow(lines), top, "LENS's room as a reply read with them");
	assert.match(left(lines[top]), /^ LENS +not read for this reply +▮▯▯▯▯$/);
	assert.ok(!dulledRows(d.render(100), top).some(Boolean), "the lenses on: not dulled, the step's outline lit");
	assert.match(lines.at(-1)!, /· -\/\+ layers ·/);
	assert.ok(!/↑\/↓ layer|enter edit|s other lens/.test(lines.at(-1)!));
	still(was, d.render(100), "a reply read without the lenses");
	d.handleInput("+");
	await tick();
	assert.deepEqual(sent, [1], "+ still adds layers for the next reply");
	for (const x of [on, off, d]) x.dispose();
});

test("a reply read with the lenses off: off, the fewest step's outline dulled (SETTLING unlit, an empty eyepiece, faint rows); + sends step 0 and lights it, not read", async () => {
	const { d, sent } = await dockAt(76, LENS_OFF, { lens: false });
	const faint = (ls: string[]) => ls.filter((l) => /^ *│ L\d+ +· +│$/.test(left(l))).map((l) => /L\d+/.exec(l)![0]);
	let raw = d.render(100), lines = text(raw);
	const top = lensRow(lines), sub = lines.findIndex((l) => /^ ?SETTLING /.test(left(l)));
	assert.match(left(lines[top]), /▯▯▯▯▯  off$/);
	assert.deepEqual(faint(lines), ["L47", "L46", "L38", "L31", "L24"], "the fewest step's outline, every row faint");
	const scope = lines.findIndex((l) => /^#\d+ "/.test(left(l)));
	assert.match(left(lines[scope + 1]), /^╭─+╮$/, "the eyepiece empty");
	assert.ok(sub === top + 1 && /▀/.test(left(lines[scope - 2])) && !left(lines[scope - 1]), "SETTLING's cells under the title, a blank line, the telescope");
	assert.ok(lines.slice(sub + 1, scope - 1).every((l) => (left(l).match(/▀/g) ?? []).length >= 30), "a cell a token on every line: unlit, not blank");
	assert.ok(dulledRows(raw, top).filter(Boolean).length > 20, "dulled");
	d.handleInput("+");
	await tick();
	assert.deepEqual(sent, [0], "+: the fewest step");
	raw = d.render(100);
	const on = text(raw);
	assert.match(left(on[top]), /^ LENS +not read for this reply +▮▯▯▯▯$/);
	assert.deepEqual(faint(on), faint(lines), "the same outline");
	assert.ok(!dulledRows(raw, top).some(Boolean), "lit");
	for (const k of ["s", "\x1b[A", "\r"]) d.handleInput(k);
	assert.deepEqual(text(d.render(100)), on, "↑/↓, s and enter inert");
	d.handleInput("+");
	await tick();
	assert.ok(faint(text(d.render(100))).length > 5, "+ again: a longer outline");
	d.dispose();
});

test("a reply read with the lenses off, the lenses on: the next prompt sent leaves its outline lit where it is, no sweep", async () => {
	const clock = { t: 1000 }, { d, host } = await dockAt(76, 0, { lens: false, clock });
	const faint = (ls: string[]) => ls.filter((l) => /^ *│ L\d+ +· +│$/.test(left(l))).length;
	const before = d.render(100), top = lensRow(text(before));
	assert.ok(faint(text(before)) === 5 && !dulledRows(before, top).some(Boolean), "lit, the fewest step's outline");
	onEvent(host.live, { type: "start", rid: "r2", model: "M" });                                // the next prompt: no token yet
	const now = d.render(100);
	assert.equal((d as any).engram.sweep, undefined, "no sweep");
	assert.ok(!dulledRows(now, top).some(Boolean), "still lit");
	assert.equal(faint(text(now)), 5, "the outline kept");
	d.dispose();
});

test("lenses off: LENS full size (2) and the one-panel board dulled under the step's line; x still turns steering off; ? says why", async () => {
	for (const rows of [52, 34]) {
		const { d, said } = await dockAt(rows, LENS_OFF, { steer: true });
		if (rows === 52) d.handleInput("2");
		const raw = d.render(100), lines = text(raw), tab = lines.findIndex((l) => /^ ?1 MAP +2 LENS /.test(l));
		assert.equal(lines[tab + 1], "▯▯▯▯▯  off", `${rows}`);
		assert.ok(!raw[tab + 1].includes(INK) && raw[tab + 3].includes(INK), `${rows}: the step's line as it was, the lens dulled`);
		assert.ok(lines.slice(-2).join(" ").includes("+ lenses") && lines.slice(-2).join(" ").includes("x off"), lines.slice(-2).join("\n"));
		d.handleInput("x");
		assert.deepEqual(said, ["off"]);
		d.handleInput("?");
		assert.match(text(d.render(100)).join(" "), /The lenses off \(\+ on\), or not read for this reply\./);
		d.dispose();
	}
});

test("the mini dock: SETTLING over the layers the reply read from the deepest, sized on them, whatever the step: a reply read at step 2 (22 layers) in the lines the lights leave (8 at 76 rows), dulled with the lenses off; one read at step 1 (9) every other layer a line", async () => {
	for (const [read, lines] of [[REAL[2], 8], [REAL[1], 5]] as const) {
		let first: (string | undefined)[] | undefined;
		for (const at of [LENS_OFF, 0, 2, 4]) {
			const { d, host } = await dockAt(76, at, { read: [...read], steps: REAL, ns: REAL_N });
			host.mini = true;
			const raw = d.render(MINI_WIDTH), ls = text(raw), s = ls.findIndex((l) => /^SETTLING/.test(l));
			const rows = ls.slice(s + 1).findIndex((l) => !/▀/.test(l)), labels = ls.slice(s + 1, s + 1 + rows).map((l) => /^L\d+/.exec(l)?.[0]);
			const what = `${read.length} layers read, step ${at}`;
			assert.equal(rows, lines, `${what}: its lines`);
			// settlingPanel's rule (not filled): the layers deep first, shared evenly over min(n, 2 lines) half lines,
			// a line labelled with its upper half's deepest layer; 9 layers in 5 lines every other one.
			const deep = [...read].sort((a, b) => b - a), half = Math.min(deep.length, 2 * lines);
			assert.deepEqual(labels, deep.slice(0, lines).map((_, i) => `L${deep[Math.floor((2 * i * deep.length) / half)]}`), `${what}: the labels`);
			assert.deepEqual(labels, (first ??= labels), `${what}: as at every step`);
			assert.equal(raw[s + 1].includes(INK), at === LENS_OFF, `${what}: dulled only with the lenses off`);
			d.dispose();
		}
	}
});

test("lenses off: the mini dock keeps SETTLING where it is, dulled", async () => {
	const { d, host } = await dockAt(52, 0);
	host.mini = true;
	const on = text(d.render(MINI_WIDTH));
	d.dispose();
	const off = await dockAt(52, LENS_OFF);
	off.host.mini = true;
	const raw = off.d.render(MINI_WIDTH), at = text(raw).findIndex((l) => /^SETTLING/.test(l));
	assert.ok(at > 0 && at === on.findIndex((l) => /^SETTLING/.test(l)), "where it was");
	assert.ok(raw[at + 1].includes(INK) && !raw[at - 1].includes(INK), "SETTLING dulled, MAP not");
	off.d.dispose();
});

test("dull, dullLines, dullPace: every colour toward the background, the text kept; the title kept, one edge up from the bottom; boardKeys: LENS dulled leaves its reading keys out", () => {
	const th = dockTheme(), line = th.fg("success", "word") + " plain";
	assert.equal(plain(dull(th, line, DULL)), "word plain");
	assert.ok(dull(th, line, DULL).startsWith(INK) && !dull(th, line, DULL).includes(th.fg("success", "").slice(0, -5)), "no colour left as it was");
	// 256 colours (Terminal.app): a half-block cell's foreground and background both dulled, nothing else changed.
	const t256 = { getColorMode: () => "256color" }, cell = "\x1b[38;5;196m\x1b[48;5;21m▀\x1b[39m\x1b[49m";
	const rgb = (n: number): [number, number, number] => { const { r, g, b } = colorToRgb(indexedColor(n)); return [r, g, b]; };
	const want = (n: number, bg: boolean) => colourCode(mix(rgb(n), COLOURS.bg, DULL), "256color", bg);
	const ink256 = colourCode(mix(COLOURS.text, COLOURS.bg, DULL), "256color");
	assert.equal(dull(t256, cell, DULL), ink256 + want(196, false) + want(21, true) + "▀" + ink256 + "\x1b[49m\x1b[39m");
	assert.notEqual(want(21, true), "\x1b[48;5;21m");
	// The title and 12 lines: two a tick from the bottom; the title kept.
	const lines = ["T", ...Array.from({ length: 12 }, (_, i) => th.fg("success", `line ${i + 1}`))];
	const dulled = (ls: string[]) => ls.map((l) => (l.startsWith(INK) ? 1 : 0)).join("");
	const at = (step: number) => dullLines(th, lines, step);
	assert.deepEqual(dullPace(12), { rows: 2, ticks: 6 });
	assert.deepEqual(at(0), lines);
	assert.deepEqual(at(6).map(plain), lines.map(plain), "the text kept");
	assert.deepEqual([1, 2, 3, 4, 5, 6].map((k) => dulled(at(k))),
		["0000000000011", "0000000001111", "0000000111111", "0000011111111", "0001111111111", "0111111111111"]);
	assert.deepEqual(dullPace(11), { rows: 2, ticks: 6 }, "an odd count: the last tick one line");
	// A tall LENS: more a tick (a multiple of ROWS_A_TICK), the sweep SWEEP_MAX_TICKS at most.
	assert.deepEqual(dullPace(24), { rows: 2, ticks: 12 });
	assert.deepEqual(dullPace(25), { rows: 4, ticks: 7 });
	assert.deepEqual(dullPace(73), { rows: 8, ticks: 10 });
	const keys = (shut: "" | "off" | "none") => boardKeys("LENS", true, false, PANELS, false, true, true, true, "", 200, "", false, shut);
	assert.match(keys(""), /↑\/↓ layer · -\/\+ layers · enter edit · x off · s other lens/);
	assert.equal(keys("off"), "tab tokens · ←/→ token · + lenses · x off · 1-6 panel · ? explain · esc back");
	assert.equal(keys("none"), "tab tokens · ←/→ token · -/+ layers · x off · 1-6 panel · ? explain · esc back");
	assert.equal(LABELS.lensOn, "+ lenses");
});

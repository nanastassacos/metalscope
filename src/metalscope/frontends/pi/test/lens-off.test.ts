import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { DULL, MAP_CELL_LINES, MAP_LINES, MINI_WIDTH, ROWS_A_TICK, SWEEP_MAX_TICKS, boardKeys, dull, dullLines, dullPace, PANELS } from "../board.ts";
import { Dock, LENS_OFF, SETTLE_MOST } from "../dock.ts";
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
/** The tube's rows of layers wanted, not read, in the dock's raw lines: the label faint (dulled with the lenses off),
 * the row blank; their labels. */
const esc = (s: string) => s.replace(/[[\]\\^$.|?*+()]/g, "\\$&");
const UNREAD = new RegExp(`(${[COLOURS.faint, mix(COLOURS.faint, COLOURS.bg, DULL)].map((c) => esc(colourCode(c, "truecolor"))).join("|")})[ ▸]L\\d+ +\x1b`);
const unread = (raw: string[]) => raw.filter((l) => UNREAD.test(l)).map((l) => /L\d+/.exec(plain(l))![0]);
const SEP = dockTheme().fg("rule", " │ "), leftRaw = (l: string) => l.slice(0, Math.max(0, l.indexOf(SEP)));
const dulledRows = (raw: string[], top: number) => raw.map((l, i) => i > top && i < raw.length - 2 && leftRaw(l).includes(INK));

/** Reply token `i` of a 48-layer model, read by the J-lens at every layer (`lens` false: read with the lenses off);
 * `L` 64: a dense model's, read by the logit lens, no experts. */
function tok(i: number, lens = true, L = 48): any {
	const experts: number[] = [], w: number[] = [];
	if (L === 48) for (let l = 0; l < 48; l++) for (let j = 0; j < 8; j++) { experts.push((i * 7 + l * 13 + j * 31) % 384); w.push(j ? 20 : 120); }
	return { type: "token", rid: "r", i, id: 100 + i, text: ` w${i}`, p: i % 5 ? 0.95 : 0.3, entropy: 0.5,
		alts: Array.from({ length: 9 }, (_, a) => [` a${a}`, 0.04 - a * 0.004]), n_layers: L, ...(L === 48 && { experts, expert_w: w, expert_k: 8 }),
		...(lens ? { lens: [...Array(L).keys()].map((l) => [l, l >= L - 6 ? 100 + i : 7, l >= L - 6 ? ` w${i}` : " the"]), lens_by: L === 48 ? "jlens" : "logit" } : {}) };
}
const SIZES = ["small", "medium", "large"];
const STEPS = [[24, 31, 38, 46, 47], [0, 8, 16, 24, 31, 36, 40, 44, 46, 47], [...Array(48).keys()]];
/** Layer sets a reply may have been read at on a 48-layer model: 5, 9, 22, 29 and 48 layers. */
const READS = [[24, 31, 38, 46, 47], [4, 16, 24, 28, 31, 38, 40, 46, 47], [0, 4, 8, 12, 16, 20, 24, 28, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
	[0, 2, 4, 7, 8, 11, 12, 16, 17, 20, 22, 24, 26, 28, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], [...Array(48).keys()]];
/** The server's three sizes on a 48-layer model: 5, 17 and 25 layers (the J-lens's 4, 16 and 24 of its 47: REAL_N; the
 * logit lens's the same and the last). */
const REAL = [[10, 22, 34, 46, 47], [...Array.from({ length: 16 }, (_, i) => 3 * i + 1), 47], [...Array.from({ length: 24 }, (_, i) => 2 * i), 47]], REAL_N = [4, 16, 24];
/** A 64-layer model without a J-lens (tracer.lens_steps(None, 64)): 4, 16 and 24 layers. */
const DENSE = [[15, 31, 47, 63], Array.from({ length: 16 }, (_, i) => 4 * i + 3), Array.from({ length: 24 }, (_, i) => Math.floor(((i + 1) * 64) / 24) - 1)];

/** The board 100 wide, `rows` tall, a token held, LENS in focus, the server's lens steps at `at` (POSTs in `sent`). */
async function dockAt(rows: number, at: number, o: { lens?: boolean; read?: number[]; clock?: { t: number }; steer?: boolean; panel?: string; steps?: number[][]; ns?: number[]; full?: boolean; dense?: boolean } = {}) {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r", model: "M" });
	for (let i = 0; i < 40; i++) {
		const t = tok(i, o.lens ?? true, o.dense ? 64 : 48);
		if (o.read && t.lens) t.lens = t.lens.filter(([l]: [number]) => o.read!.includes(l));   // read at those layers only
		onEvent(live, t);
	}
	onEvent(live, { type: "end", rid: "r", n_tokens: 40, trace_id: "T" });
	if (o.dense) live.dense = "M";
	if (o.steer) live.steer = { parts: [{ name: "calm", layer: 20, strength: 0.4 }] } as any;
	const data = { model: "M", at, steps: (o.steps ?? STEPS).map((layers, i) => ({ name: SIZES[i], n: o.ns?.[i] ?? layers.length, layers, tps: null, replies: 0 })) }, sent: number[] = [], said: string[] = [];
	const grid = { cols: 32, rows: 12, pos: Array.from({ length: 384 }, (_, e) => e), num_experts: 384, n_layers: 48 };
	const host: any = { live, captions: false, close() {}, unfocus() {}, panel: o.panel ?? "LENS", steerModel: () => "M",
		fetchJson: async (url: string) => (url.includes("/experts/grid") ? grid : null), now: o.clock && (() => o.clock!.t),
		steer: async (a: string) => (said.push(a), []), steering: async () => ({ model: "M", concepts: [] }),
		lensLayers: async (step?: number) => (step !== undefined && (sent.push(step), (data.at = step)), structuredClone(data)) };
	const d = new Dock({ terminal: { rows, columns: 200 }, requestRender() {} }, { getColorMode: () => "truecolor" }, host);
	d.stepIn(30);
	if (o.full) d.handleInput("3");
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
	assert.match(left(was[top]), /^ LENS +J-lens +▮▯▯  small · reads 5 layers$/);
	assert.match(was.at(-1)!, /↑\/↓ layer · -\/\+ layers · enter edit/);
	assert.ok(!dulledRows(open, top).some(Boolean));
	d.handleInput("-");
	await tick();                                                                             // dulled once the server has it
	const off = d.render(100), lines = text(off);
	assert.deepEqual(sent, [LENS_OFF], "set on the server");
	assert.match(left(lines[top]), /^ LENS {10,}▯▯▯  off$/, "the pips unlit, off");
	for (let i = top + 1; i < lines.length - 2; i++) assert.equal(left(lines[i]), left(was[i]), `row ${i}: the same text`);
	assert.ok(dulledRows(off, top).filter(Boolean).length > 10, "dulled");
	assert.match(lines.at(-1)!, /^tab experts · ←\/→ token · f live · \+ lenses · /);
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

test("- + at 52, 76 and 100 rows: the right column, REPLY, TOKENS and MAP's title stay; under LENS's title SETTLING and the telescope's first line where they were; the largest step to the column's foot, a smaller one leaving MAP its lines, the rest blank at the bottom; off as the fewest", async () => {
	const settling = (lines: string[]) => lines.findIndex((l) => /^ ?SETTLING\b/.test(left(l)));
	const scope = (lines: string[]) => lines.findIndex((l) => /^#\d+ "/.test(left(l)));
	const last = (ls: string[], sub: number) => sub + ls.slice(sub + 1).findIndex((l) => !/▀/.test(left(l)));
	const mapRow = (ls: string[]) => ls.findIndex((l) => /^ ?MAP /.test(left(l)));
	for (const rows of [52, 76, 100]) {
		const frames: string[][] = [];
		for (const at of [LENS_OFF, 0, 1, 2]) {
			const { d } = await dockAt(rows, at, { steps: REAL, ns: REAL_N });
			frames.push(d.render(100));
			d.dispose();
		}
		const ref = text(frames[3]), top = lensRow(ref), sub = settling(ref), tel = scope(ref), end = ref.length - 3, map = mapRow(ref);
		assert.ok(top > 0 && sub === top + 1 && tel > sub, `${rows}: ${top} ${sub} ${tel}`);
		// SETTLING (its sub-heading and lines) at the largest step: its most at 100 rows; at 52 and 76 its least, the
		// panel's lines going to the telescope first (lensSplit). MAP its most, but at the largest step under 100 rows its
		// least (LENS takes its lines); at 100 the largest's 25 layers fit without them.
		const sl = last(ref, sub) - sub + 1, mapLines = (ls: string[]) => lensRow(ls) - 1 - map;
		assert.equal(sl, rows === 100 ? 1 + SETTLE_MOST : 4, `${rows}: SETTLING ${sl} lines`);
		assert.deepEqual(frames.map((f) => mapLines(text(f))), [1, 1, 1, 0].map((most) => (most || rows === 100 ? 1 + MAP_CELL_LINES : 1 + MAP_LINES)), `${rows}: MAP`);
		// SETTLING over the step's layers: at the fewest, the five it reads, each a band labelled once at 100 rows; in its
		// three lines at 52 and 76, binned (a label where a band starts).
		const labels = (ls: string[], s: number) => ls.slice(s + 1, last(ls, s) + 1).map((l) => /^L\d+/.exec(left(l))?.[0]).filter(Boolean);
		assert.deepEqual(labels(text(frames[1]), settling(text(frames[1]))), rows === 100 ? ["L47", "L46", "L34", "L22", "L10"] : ["L47", "L46", "L22"], `${rows}: step 0's layers`);
		const drawn: number[] = [];
		for (const [k, f] of frames.entries()) {
			const what = `${rows} rows, ${k ? `step ${k - 1}` : "off"}`, ls = text(f), t = lensRow(ls), dy = t - top;
			for (let i = 0; i < f.length - 1; i++)                     // the key line aside: off has its own keys
				if (i <= map || i >= f.length - 2) assert.equal(f[i], frames[3][i], `${what}: row ${i}`);
				else assert.equal(f[i].slice(f[i].indexOf(SEP)), frames[3][i].slice(frames[3][i].indexOf(SEP)), `${what}: row ${i}, the right column`);
			assert.ok(dy >= 0 && dy === mapLines(ls) - mapLines(ref), `${what}: LENS under MAP's lines`);
			assert.deepEqual([settling(ls), last(ls, settling(ls)), scope(ls)], [sub + dy, last(ref, sub) + dy, tel + dy], `${what}: SETTLING and the telescope under LENS's title`);
			// Blank in LENS: the line between SETTLING and the telescope, then a run at the bottom: none at the largest
			// step, none at another while MAP is short of its most.
			const blank = ls.slice(t + 1, end + 1).map((l, i) => (!left(l) ? t + 1 + i : -1)).filter((i) => i >= 0);
			const foot = blank.filter((b) => b !== tel + dy - 1);
			assert.ok(blank.includes(tel + dy - 1) && foot.every((b, i) => b === end - foot.length + 1 + i), `${what}: blank ${blank}`);
			assert.ok(foot.length === 0 || mapLines(ls) === 1 + MAP_CELL_LINES, `${what}: ${foot.length} blank, MAP ${mapLines(ls)} lines`);
			// The largest step to the column's foot, but at 100 rows: its 25 layers a line, MAP at its most, one left.
			if (k === 3) assert.equal(foot.length, rows === 100 ? 1 : 0, `${what}: the largest step to the column's foot`);
			assert.ok(!ls.some((l) => /shallower layer/.test(l)), `${what}: every layer on the tube`);
			drawn.push(end - t - foot.length);
		}
		assert.equal(drawn[0], drawn[1], `${rows}: off as the fewest`);
		assert.ok(drawn[1] < drawn[2] && drawn[2] <= drawn[3], `${rows}: LENS ${drawn} lines`);
	}
});

test("the board at 63 and 40 rows, the three sizes: small compact, the rest blank under it; medium LENS's panel with MAP at its most; large MAP's lines too, as many as it needs; Flash Next and a 64-layer model without experts", async () => {
	const got: Record<string, number[][]> = {};
	for (const rows of [63, 40])
		for (const [model, o] of [["flash", { steps: REAL, ns: REAL_N }], ["dense", { steps: DENSE, dense: true }]] as const)
			for (const at of [0, 1, 2]) {
				const { d } = await dockAt(rows, at, o), name = `${model} ${rows}`;
				const ls = text(d.render(100)), top = lensRow(ls), end = ls.length - 3, map = ls.findIndex((l) => /^ ?MAP /.test(left(l)));
				let last = end;
				while (!left(ls[last])) last--;
				assert.ok(!ls.some((l) => /shallower layer/.test(l)), `${name}, ${SIZES[at]}: every layer on the tube`);
				const tube = ls.filter((l) => /│[ ▸]L\d+/.test(left(l))), shared = tube.filter((l) => /L\d+ \+\d/.test(l)).length;
				(got[name] ??= []).push([top - 1 - map, last - top + 1, tube.length, shared, end - last]);
				d.dispose();
			}
	// [MAP's lines, LENS's lines, the tube's rows, rows sharing layers, blank lines under LENS]. Small its layers a
	// line, MAP its most, the rest blank; medium the panel MAP at its most leaves (Flash Next's 17 layers a line, one
	// line blank; the dense model's 16, five); large MAP's lines over its least too, as many as its tube takes (Flash
	// Next's 25 layers in 23 rows, MAP at its least; the dense model's 24 a line, MAP two lines over its least).
	// 40 rows: MAP already at its least with Flash Next, so the three the same height; without experts MAP has lines to
	// give, medium short of LENS's least to give MAP its most.
	assert.deepEqual(got, {
		"flash 63": [[1 + MAP_CELL_LINES, 21, 5, 0, 13], [1 + MAP_CELL_LINES, 33, 17, 0, 1], [1 + MAP_LINES, 39, 23, 2, 0]],
		"dense 63": [[1 + MAP_CELL_LINES, 20, 4, 0, 17], [1 + MAP_CELL_LINES, 32, 16, 0, 5], [3 + MAP_LINES, 40, 24, 0, 0]],
		"flash 40": [[1 + MAP_LINES, 17, 4, 1, 0], [1 + MAP_LINES, 17, 4, 2, 0], [1 + MAP_LINES, 17, 4, 2, 0]],
		"dense 40": [[MAP_CELL_LINES, 20, 4, 0, 0], [MAP_CELL_LINES, 20, 4, 2, 0], [1 + MAP_LINES, 24, 8, 4, 0]] });
});

test("- + move nothing on LENS full size (2): at 36, 40, 52, 76 and 100 rows every step and off keep the step's line, SETTLING's lines and the lenses' first line where they are; only the tubes lengthen, the lines a short tube frees blank at the foot; SETTLING its want first, the tubes' box the rest, shared at the shallow end, never folded", async () => {
	const at = (ls: string[], re: RegExp) => ls.findIndex((l) => re.test(l));
	const parts = (ls: string[]) => {
		const step = at(ls, /^[▮▯]+ +(\w+ · |off)/), sub = at(ls, /^SETTLING\b/), names = at(ls, /^J-lens\b/), rule = ls.length - 4;
		return { step, sub, last: sub + ls.slice(sub + 1).findIndex((l) => !/▀/.test(l)), names, rule };
	};
	for (const rows of [36, 40, 52, 76, 100]) {
		const frames: string[][] = [];
		for (const s of [LENS_OFF, 0, 1, 2]) {
			const { d } = await dockAt(rows, s, { steps: REAL, ns: REAL_N, full: true });
			frames.push(text(d.render(100)));
			d.dispose();
		}
		const ref = frames[2], p = parts(ref);
		const tube = (ls: string[]) => ls.slice(p.names, p.rule).filter((l) => /│[ ▸]L\d+/.test(l)), shared = (ls: string[]) => tube(ls).filter((l) => /^ *│[ ▸]L\d+ \+\d/.test(l)).length;
		assert.ok(p.step > 0 && p.sub === p.step + 1 && p.last > p.sub + 2 && p.names === p.last + 2 && /^─+$/.test(ref[p.rule]), `${rows}: ${JSON.stringify(p)}`);
		// SETTLING its most (its sub-heading and SETTLE_MOST lines) where the page holds it.
		if (rows >= 76) assert.equal(p.last - p.sub, SETTLE_MOST, `${rows}: SETTLING ${p.last - p.sub + 1} lines`);
		for (const [k, ls] of frames.entries()) {
			const what = `${rows} rows, ${k ? `step ${k - 1}` : "off"}`;
			assert.deepEqual(parts(ls), p, `${what}: the step's line, SETTLING, the lenses and the rule where they were`);
			assert.deepEqual(ls.slice(0, p.step), ref.slice(0, p.step), `${what}: the lines over LENS`);
			assert.deepEqual(ls.slice(p.rule, -1), ref.slice(p.rule, -1), `${what}: the rule and the legend (the key line aside)`);
			// Blank: the line over the lenses, then one run at the foot; none with the largest step.
			const blank = ls.map((l, i) => (i > p.step && i < p.rule && !l ? i : -1)).filter((i) => i >= 0), foot = blank.filter((b) => b !== p.names - 1);
			assert.ok(blank.includes(p.names - 1) && foot.every((b, i) => b === p.rule - foot.length + i), `${what}: blank ${blank}`);
			if (k === 3) assert.ok(!foot.length || !shared(ls), `${what}: the large size's tubes fill the page, or are whole (SETTLING at its most)`);
			assert.ok(!ls.some((l) => /shallower layer/.test(l)), `${what}: shared lines, no fold`);
		}
		assert.deepEqual(frames.slice(1).map((f) => tube(f).length).map((n, i, a) => !i || n >= a[i - 1]), [true, true, true], `${rows}: the tubes longer size by size`);
		assert.equal(shared(frames[3]) === 0, rows >= 76, `${rows}: the largest step a layer a line only where the page holds it`);
	}
});

test("a reply read with the lenses off, LENS full size (2): off, the fewest step's outline dulled under SETTLING; + sends step 0 and lights it in place, not read", async () => {
	const { d, sent } = await dockAt(76, LENS_OFF, { lens: false, full: true });
	let raw = d.render(100), lines = text(raw);
	const step = lines.findIndex((l) => /^▯+ +off$/.test(l)), names = lines.findIndex((l) => /^not read for this reply$/.test(l));   // titled as the board titles the outline
	assert.ok(step > 0 && lines[step + 1].startsWith("SETTLING") && names > step + 3 && !lines[names - 1], "SETTLING under the step's line, a blank line, the lens");
	assert.ok(lines.slice(step + 2, names - 1).every((l) => (l.match(/▀/g) ?? []).length >= 40), "a cell a token on every line: unlit, not blank");
	const outline = unread(raw);
	assert.deepEqual(outline, ["L47", "L46", "L38", "L31", "L24"], "the fewest step's outline, every row faint");
	assert.ok(raw.slice(step + 1, -4).filter((l) => l.includes(INK)).length > 20, "dulled");
	d.handleInput("+");
	await tick();
	assert.deepEqual(sent, [0], "+: the fewest step");
	raw = d.render(100);
	const on = text(raw);
	assert.match(on[step], /^▮▯▯ /);
	assert.ok(on[step + 1].startsWith("SETTLING") && on[names] === "not read for this reply", "nothing moved");
	assert.deepEqual(unread(raw), outline, "the same outline");
	assert.ok(!raw.slice(step + 1, -4).some((l) => l.includes(INK)), "lit");
	d.dispose();
});

// `under`: the column's lines left blank under LENS (none where the step is the largest; the fewest of three at 100 rows
// leaves 21, MAP at its most).
for (const [what, rows, full, steps, under] of [["the board at 52 rows, one step", 52, false, STEPS.slice(0, 1), 0], ["the board at 100 rows", 100, false, STEPS, 21], ["LENS full size at 76 rows", 76, true, STEPS, 0]] as const)
	test(`lenses off, ${what}: the sweep on the sweep's ticker, one edge up from LENS's bottom, ROWS_A_TICK rows a tick (a multiple of it tall), SWEEP_MAX_TICKS at most; back on, down from the top; a key ends it`, async () => {
		const clock = { t: 1000 }, { d } = await dockAt(rows, 0, { clock, steps: [...steps], full });
		const open = d.render(100), lines = text(open);
		const top = full ? lines.findIndex((l) => /^[▮▯]+ +\w+ · /.test(l)) : lensRow(lines);   // LENS's first line kept
		const part = (l: string) => (full ? l : leftRaw(l));
		assert.ok(top > 0 && lines[top + 1].startsWith("SETTLING"), "SETTLING under LENS's first line");
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
	host.lensLayers = async (step?: number) => (step === undefined ? get() : Promise.reject(new Error("HTTP 400: step must be 0 to 2, or -1 (off)")));
	d.handleInput("-");
	await tick();
	const now = d.render(100);
	assert.equal((d as any).engram.sweep, undefined, "no sweep");
	assert.deepEqual(now.slice(0, -1), open.slice(0, -1), "LENS as it was");
	assert.match(text(now).at(-1)!, /HTTP 400: step must be 0 to 2, or -1 \(off\)/);
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
	assert.match(left(lines[top]), /▯▯▯  off$/, "off, not 'not read'");
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
	assert.match(left(lines[top]), /^ LENS +not read for this reply +▮▯▯ small \d+$/);
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
	let raw = d.render(100), lines = text(raw);
	const top = lensRow(lines), sub = lines.findIndex((l) => /^ ?SETTLING\b/.test(left(l)));
	assert.match(left(lines[top]), /▯▯▯  off$/);
	const outline = unread(raw);
	assert.deepEqual(outline, ["L47", "L46", "L38", "L31", "L24"], "the fewest step's outline, every row faint");
	const scope = lines.findIndex((l) => /^#\d+ "/.test(left(l)));
	assert.match(left(lines[scope + 1]), /^╭─+╮$/, "the eyepiece empty");
	assert.ok(sub === top + 1 && /▀/.test(left(lines[scope - 2])) && !left(lines[scope - 1]), "SETTLING's cells under the title, a blank line, the telescope");
	assert.ok(lines.slice(sub + 1, scope - 1).every((l) => (left(l).match(/▀/g) ?? []).length >= 30), "a cell a token on every line: unlit, not blank");
	// LENS's lines under its title dulled: 20 at 76 rows with the fewest step's outline.
	assert.equal(dulledRows(raw, top).filter(Boolean).length, 20, "dulled");
	d.handleInput("+");
	await tick();
	assert.deepEqual(sent, [0], "+: the fewest step");
	raw = d.render(100);
	const on = text(raw);
	assert.match(left(on[top]), /^ LENS +not read for this reply +▮▯▯ small \d+$/);
	assert.deepEqual(unread(raw), outline, "the same outline");
	assert.ok(!dulledRows(raw, top).some(Boolean), "lit");
	for (const k of ["s", "\x1b[A", "\r"]) d.handleInput(k);
	assert.deepEqual(text(d.render(100)), on, "↑/↓, s and enter inert");
	d.handleInput("+");
	await tick();
	assert.ok(unread(d.render(100)).length > 5, "+ again: a longer outline");
	d.dispose();
});

test("a reply read with the lenses off, the lenses on: the next prompt sent leaves its outline lit where it is, no sweep", async () => {
	const clock = { t: 1000 }, { d, host } = await dockAt(76, 0, { lens: false, clock });
	const before = d.render(100), top = lensRow(text(before));
	assert.ok(unread(before).length === 5 && !dulledRows(before, top).some(Boolean), "lit, the fewest step's outline");
	onEvent(host.live, { type: "start", rid: "r2", model: "M" });                                // the next prompt: no token yet
	const now = d.render(100);
	assert.equal((d as any).engram.sweep, undefined, "no sweep");
	assert.ok(!dulledRows(now, top).some(Boolean), "still lit");
	assert.equal(unread(now).length, 5, "the outline kept");
	d.dispose();
});

test("lenses off: LENS full size (2) and the one-panel board dulled under the step's line; x still turns steering off; ? says why", async () => {
	for (const rows of [52, 34]) {
		const { d, said } = await dockAt(rows, LENS_OFF, { steer: true });
		if (rows === 52) d.handleInput("3");
		const raw = d.render(100), lines = text(raw), tab = lines.findIndex((l) => /^ ?1 TOKENS +2 MAP /.test(l));
		assert.equal(lines[tab + 1], "▯▯▯  off", `${rows}`);
		assert.ok(!raw[tab + 1].includes(INK) && raw[tab + 3].includes(INK), `${rows}: the step's line as it was, the lens dulled`);
		assert.ok(lines.slice(-2).join(" ").includes("+ lenses") && lines.slice(-2).join(" ").includes("x off"), lines.slice(-2).join("\n"));
		d.handleInput("x");
		assert.deepEqual(said, ["off"]);
		d.handleInput("?");
		assert.match(text(d.render(100)).join(" "), /The lenses off \(\+ on\), or not read for this reply\./);
		d.dispose();
	}
});

test("the mini dock: SETTLING over the layers the reply read from the deepest, sized on them, whatever the size: a reply read at 22 layers in SETTLE_MOST lines (its most), dulled with the lenses off; one read at 9 every other layer a line", async () => {
	for (const [read, lines] of [[READS[2], SETTLE_MOST], [READS[1], 5]] as const) {
		let first: (string | undefined)[] | undefined;
		for (const at of [LENS_OFF, 0, 1, 2]) {
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
	assert.equal(keys("off"), "tab experts · ←/→ token · + lenses · x off · 1-6 panel · ? explain · esc back");
	assert.equal(keys("none"), "tab experts · ←/→ token · -/+ layers · x off · 1-6 panel · ? explain · esc back");
	assert.equal(LABELS.lensOn, "+ lenses");
});

test("- + between medium and large animate MAP and LENS: the frames' heights between the two, a step at a time, ending as the size drawn at rest; the right column and the reply strip still; small and medium too, both ways; Flash Next and a 64-layer model at 63 rows", async () => {
	const shape = (raw: string[]) => {
		const ls = text(raw), top = lensRow(ls), map = ls.findIndex((l) => /^ ?MAP /.test(left(l)));
		let last = ls.length - 3;
		while (!left(ls[last])) last--;
		return [top - 1 - map, last - top + 1];
	};
	const right = (raw: string[], map: number) => raw.map((l, i) => (i <= map ? l : l.slice(l.indexOf(SEP))));
	for (const [model, o] of [["flash", { steps: REAL, ns: REAL_N }], ["dense", { steps: DENSE, dense: true }]] as const)
		for (const [from, key, to] of [[1, "+", 2], [2, "-", 1], [0, "+", 1], [1, "-", 0]] as const) {
			const rest = await dockAt(63, to, o), end = rest.d.render(100), want = shape(end), what = `${model} ${SIZES[from]} ${key}`;
			rest.d.dispose();
			mock.timers.enable({ apis: ["setInterval"] });
			try {
				const { d } = await dockAt(63, from, o), start = d.render(100), map = text(start).findIndex((l) => /^ ?MAP /.test(left(l)));
				const seen = [shape(start)];
				d.handleInput(key);
				for (let i = 0; i < 6; i++, mock.timers.tick(70)) {
					const raw = d.render(100);
					seen.push(shape(raw));
					assert.deepEqual(right(raw, map), right(start, map), `${what} frame ${i}: the right column and all over MAP still`);
				}
				await tick();
				assert.deepEqual(text(d.render(100)), text(end), `${what}: at rest, as the size drawn alone`);
				// Each frame between the last and the next, a step at a time; MAP moves only to or from the largest.
				const maps = seen.map(([m]) => m), lens = seen.map(([, l]) => l), dir = Math.sign(want[1] - seen[0][1]);
				assert.deepEqual(seen.at(-1), want, what);
				assert.ok(lens.every((l, i) => !i || (l - lens[i - 1]) * dir >= 0) && maps.every((m, i) => !i || (maps[i - 1] - m) * dir >= 0), `${what}: ${JSON.stringify(seen)}`);
				assert.ok(new Set(lens).size >= 4, `${what}: ${JSON.stringify(seen)} at least two heights between`);
				const moved = Math.abs(maps[0] - maps.at(-1)!);                // MAP's lines the large size takes: 5 on Flash Next, 3 dense
				if (from + to === 3) assert.ok(moved >= 3 && new Set(maps).size >= Math.min(4, moved), `${what}: MAP ${maps} heights between`);
				d.dispose();
			} finally {
				mock.timers.reset();
			}
		}
});

test("LENS's title keeps the step's count: shorter forms, then a shorter switch, then the name goes; the count last", async () => {
	const { d, host } = await dockAt(52, 1, { steps: REAL, ns: REAL_N });
	const r = host.live.reply, both = { ...r.tokens[30], lens_logit: r.tokens[30].lens }, head = (t: any, w: number) => plain((d as any).lensHead(r, t, false, {}, w));
	const tail = (t: any, w: number) => head(t, w).replace(/^LENS {2}/, "").replace(/ {2,}/, " | ");
	assert.deepEqual([56, 49, 44, 40, 37, 30, 29].map((w) => tail(both, w)), [
		"J-lens ◉────○ logit | ▮▮▯  medium · reads 16 layers", "J-lens ◉────○ logit | ▮▮▯ medium · 16 layers",
		"J-lens ◉────○ logit | ▮▮▯ medium 16", "J-lens ◉────○ logit | ▮▮▯ medium 16", "J-lens ◉─○ logit | ▮▮▯ medium 16",
		"J-lens ◉─○ logit | ▮▮▯ 16", "J-lens ◉─○ logit | ▮▮▯"]);
	for (const w of [56, 44, 37, 30]) assert.ok(head(both, w).length <= w, `${w}`);
	// One lens: no switch to shorten, the name goes.
	const one = r.tokens[30];
	assert.deepEqual([38, 28, 24].map((w) => tail(one, w)), ["J-lens | ▮▮▯ medium · 16 layers", "J-lens | ▮▮▯ medium 16", "J-lens | ▮▮▯ 16"]);
	d.dispose();
});

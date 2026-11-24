import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mock, test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { BOARD_MIN_WIDTH, FULL_FADE, MINI_WIDTH, PLAY_JUMP_MS, PLAY_LAG_MS, PLAY_TICK_MS, RIGHT_W, STREAM_LINES, STREAM_MIN, type Play, type StreamView, columns, hasLegend, keyWidth, layoutKeys, miniStreamLines, pace, playShown,
	nextView, spread, storedView, STREAM_LOOK, STREAM_RGB, streamPanel, streamViews, viewOf } from "../board.ts";
import { Dock } from "../dock.ts";
import { COLOURS, LABELS, colourCode, dockTheme, streamExplain } from "../draw.ts";
import { mix } from "../lights.ts";
import { type Live, type Reply, WRITE_FLOOR, WRITE_WARM, liveUrl, onEvent, packLargest, packWrites, unpackLargest, writeSd } from "../state.ts";
import { createUi, readPrefs } from "../widget.ts";

// STREAM (RESIDUAL STREAM): a key a layer, last in the board's right column, lit by what each layer changed in the
// residual streams or read from its sources. The rule is decided per token as it comes (packWrites), the panel drawn
// from that (streamPanel); tab reaches it, s switches its view, 5 opens it full size.

const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
/** A model writing four streams (Flash Next's), `L` its layers. */
const S4 = { L: 48, k: 4, rk: 0 };
const text = (lines: string[]) => lines.map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd());
const tick = () => new Promise((r) => setTimeout(r, 0));

/** `L` layers' writes into 4 streams, every value `v` but those in `at` ([layer, stream] -> value). */
const flat = (L: number, v: (l: number, s: number) => number, at: Record<string, number> = {}) =>
	Array.from({ length: L * 4 }, (_, j) => at[`${Math.floor(j / 4)},${j % 4}`] ?? v(Math.floor(j / 4), j % 4));
/** A token of a reply with writes `w` (k 4). */
const decide = (r: Reply, w: number[]) => {
	const t: any = { writes: w, write_k: 4 };
	packWrites(r, t);
	return t;
};
const strikes = (t: any) => (t.strikes ? [...t.strikes] : undefined);
// The usual: 1.0 and 1.2 by turns, so after 12 tokens its mean is 1.1 and its spread SD (writeSd: the sample deviation
// widened for the mean's error); a value `z` spreads over it is usual + z * SD.
const usual = (i: number) => (i % 2 ? 1.2 : 1.0);
const SD = writeSd(12 * 0.01, 12), over = (z: number) => 1.1 + z * SD;
/** A reply that has seen WRITE_WARM tokens of the usual at `L` layers. */
const warmed = (L: number) => {
	const r = { rid: "r", tokens: [], done: false, prefill: 0 } as Reply;
	for (let i = 0; i < WRITE_WARM; i++) assert.deepEqual(strikes(decide(r, flat(L, () => usual(i)))), [], `token ${i}: nothing to strike`);
	return r;
};

test("packWrites: a layer is struck where a stream is past 2 deviations over its usual, its beam by how far", () => {
	assert.ok(Math.abs(SD - Math.sqrt((0.12 / 11) * (13 / 12))) < 1e-12, "var = m2 / (n - 1) * (1 + 1 / n)");
	const r = warmed(2);
	// z 3.5 at layer 1, stream 2: struck, a beam of 1 + floor(1.5 / 0.6) = 3; layer 0 at z 1.5 is not.
	assert.deepEqual(strikes(decide(r, flat(2, () => 1.1, { "1,2": over(3.5), "0,0": over(1.5) }))), [1, 2, 3]);
	assert.equal(r.usual!.n, WRITE_WARM + 1, "the usual takes the token in after deciding it");
	// Two streams past 2 at one layer: the higher z shows (z 2.5 and 4: stream 3, beam 1 + floor(2 / 0.6) = 4).
	assert.deepEqual(strikes(decide(warmed(1), flat(1, () => 1.1, { "0,1": over(2.5), "0,3": over(4) }))), [0, 3, 4]);
	// The beam stops at 6 cells.
	assert.deepEqual(strikes(decide(warmed(1), flat(1, () => 1.1, { "0,0": 9 }))), [0, 0, 6]);
});

test("packWrites: the first 12 tokens strike nothing, however far out; the 13th can", () => {
	assert.equal(WRITE_WARM, 12);
	const r = { rid: "r", tokens: [], done: false, prefill: 0 } as Reply;
	for (let i = 0; i < 10; i++) decide(r, flat(1, () => usual(i)));
	assert.deepEqual(strikes(decide(r, flat(1, () => 1.1, { "0,0": 50 }))), [], "the 11th token: 10 before it");
	assert.deepEqual(strikes(decide(r, flat(1, () => 1.1))), [], "the 12th: 11 before it");
	assert.equal(strikes(decide(r, flat(1, () => 1.1, { "0,1": 50 })))!.length, 3, "the 13th: 12 before it");
});

test("packWrites: a stream under 1% of its layer's write never strikes (the recording does not resolve it), yet counts in the usual", () => {
	assert.equal(WRITE_FLOOR, 0.01);
	// Streams 0-2 write 100 each; stream 3 its usual 0.5 and 0.7 by turns.
	const small = (v: number) => flat(1, (_, s) => (s < 3 ? 100 : v));
	const warm = () => {
		const r = { rid: "r", tokens: [], done: false, prefill: 0 } as Reply;
		for (let i = 0; i < WRITE_WARM; i++) decide(r, small(i % 2 ? 0.7 : 0.5));
		return r;
	};
	const r = warm();
	assert.deepEqual(strikes(decide(r, small(2))), [], "2 of 302: under 1%, however far over its usual");
	assert.equal(r.usual!.n, WRITE_WARM + 1, "counted all the same");
	assert.deepEqual(strikes(decide(warm(), small(4))), [0, 3, 6], "4 of 304: struck");
});

test("packWrites: largest, each layer's largest stream as bright as its whole change against its usual; the raw list dropped", () => {
	const r = { rid: "r", tokens: [], done: false, prefill: 0 } as Reply;
	const lit = (t: any, l: number) => unpackLargest(t.largest[l]);
	const first = decide(r, flat(3, (l, s) => (s === l + 1 ? 2 : 1)));   // layer 0's largest stream 1, layer 1's stream 2
	assert.deepEqual([lit(first, 0)[0], lit(first, 1)[0]], [1, 2]);
	assert.equal(lit(first, 0)[1], Math.round(0.55 * 31) / 31, "no usual yet: 0.55");
	assert.ok(!("writes" in first) && !("write_k" in first), "the raw list dropped");
	// Its usual total is 5 a layer: 5 is 0.1 + 0.9 * 1/2, 10 or more full, 0 the floor.
	const t = decide(r, flat(3, (l, s) => (l === 0 ? [1, 1, 2, 1][s] : l === 1 ? [1, 1, 1, 7][s] : 0)));
	assert.deepEqual(lit(t, 0), [2, Math.round(0.55 * 31) / 31]);
	assert.deepEqual(lit(t, 1), [3, 1]);
	assert.deepEqual(lit(t, 2), [0, Math.round(0.1 * 31) / 31]);
	// Malformed: nothing kept, nothing added to the usual.
	for (const bad of [{ writes: [1, 2, 3], write_k: 4 }, { writes: [1, -1, 0, 0], write_k: 4 }, { writes: [1, 2], write_k: 0 }, { writes: "x", write_k: 4 }]) {
		const n = r.usual!.n, x: any = { ...bad };
		assert.equal(packWrites(r, x), false);
		assert.ok(!x.strikes && !x.largest && !("writes" in x) && r.usual!.n === n, JSON.stringify(bad));
	}
});

test("liveUrl: the writes only while asked for", () => {
	const live: Live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: true };
	assert.equal(liveUrl(live), "http://o:1/scope/api/live?client=pi-x");
	live.writes = true;
	assert.equal(liveUrl(live), "http://o:1/scope/api/live?client=pi-x&writes=true");
});

/** Reply token `i` of a model of `L` layers (48 where L is 0 or null): writes of `L` layers into `k` streams (a fixed
 * pseudo-random spread, stream 1 the largest where there are several; none: L 0 or null), reads from `rk` sources
 * where `rk` is given, and with `moe` routing at every layer, 10 of `moe` experts a token (true: 512, Flash Next's). */
/** Token `i`'s lens readings (both lenses, every fourth of 48 layers), for a board whose LENS is not shut. */
const LENS_READ = (i: number) => ({ lens: Array.from({ length: 12 }, (_, j) => [j * 4, 100 + i, ` w${i}`]), lens_by: "jlens",
	lens_logit: Array.from({ length: 12 }, (_, j) => [j * 4, 100 + i, ` w${i}`]) });
const tok = (i: number, L: number | null = 48, rid = "r", k = 4, rk = 0, moe: boolean | number = false): any => ({ type: "token", rid, i, id: 100 + i, text: ` w${i}`, p: 0.9, entropy: 0.4,
	alts: [[" a", 0.05]], n_layers: L || 48,
	...(moe ? { experts: Array.from({ length: 10 * (L || 48) }, (_, j) => (i * 37 + j * 53) % (moe === true ? 512 : moe)), expert_w: Array.from({ length: 10 * (L || 48) }, (_, j) => (j % 10 ? 20 : 120)), expert_k: 10 } : {}),
	...(L ? { writes: Array.from({ length: L * k }, (_, j) => (j % k === 1 ? 3 : 1) + (((i * 31 + j * 7) % 13) / 13) ** 4 * 6), write_k: k } : {}),
	...(L && rk ? { reads: Array.from({ length: L * rk }, (_, j) => (1 + (((i * 17 + j * 5) % 11) / 11) ** 4 * 4) / rk), read_k: rk } : {}) });

test("onEvent: writes and reads packed as they come; a model's streams known from its first token with them, none from one asked for that came without", () => {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true, writes: true };
	onEvent(live, { type: "start", rid: "r", model: "Flash" });
	onEvent(live, tok(0));
	const t = live.reply!.tokens[0] as any;
	assert.ok(t.largest?.length === 48 && t.strikes && !("writes" in t), "packed, the raw list dropped");
	assert.deepEqual(live.streams, { model: "Flash", L: 48, k: 4, rk: 0 });
	onEvent(live, tok(1, null));
	assert.deepEqual(live.streams, { model: "Flash", L: 48, k: 4, rk: 0 }, "one missing once the reply had some: a stray");
	onEvent(live, { type: "end", rid: "r", n_tokens: 2 });
	onEvent(live, { type: "start", rid: "i", model: "Inkling" });
	onEvent(live, tok(0, 48, "i", 1, 5));
	const ti = live.reply!.tokens[0] as any;
	assert.ok(ti.readStrikes && !("reads" in ti) && !("read_k" in ti), "the reads packed too");
	assert.deepEqual(live.streams, { model: "Inkling", L: 48, k: 1, rk: 5 });
	onEvent(live, { type: "end", rid: "i", n_tokens: 1 });
	onEvent(live, { type: "start", rid: "s", model: "Other" });
	onEvent(live, tok(0, null, "s"));
	assert.deepEqual(live.streams, { model: "Other", L: 0, k: 0, rk: 0 }, "asked for and none came: none recorded");
	live.writes = false;   // not asked for: no word on the model
	onEvent(live, { type: "start", rid: "u", model: "Another" });
	onEvent(live, tok(0, null, "u"));
	assert.equal(live.streams!.model, "Other");
});

test("packWrites: reads strike as writes do, by source: z over 2 against the reply so far, the warm-up, the 1% floor on the share", () => {
	// Five sources, two layers: layer 0 reads source 0 only (the others do not exist yet: 0); layer 1 0.1 from sources 0-2,
	// source 4 0.2 and 0.4 by turns (mean 0.3, the usual's spread), source 3 the rest.
	const shares = (i: number, at: Record<string, number> = {}) => Array.from({ length: 10 }, (_, j) => {
		const l = Math.floor(j / 5), s = j % 5, v = usual(i) - 0.8;
		return at[`${l},${s}`] ?? (l === 0 ? (s === 0 ? 1 : 0) : s === 4 ? v : s === 3 ? 0.7 - v : 0.1);
	});
	const read = (r: Reply, x: number[]) => {
		const t: any = { reads: x, read_k: 5 };
		packWrites(r, t);
		return t.readStrikes ? [...t.readStrikes] : undefined;
	};
	const r = { rid: "r", tokens: [], done: false, prefill: 0 } as Reply;
	for (let i = 0; i < WRITE_WARM; i++) assert.deepEqual(read(r, shares(i)), [], `token ${i}: the warm-up`);
	assert.equal(r.readUsual!.n, WRITE_WARM);
	assert.equal(r.usual, undefined, "reads alone: no writes' usual");
	// Source 4 of layer 1 3.5 spreads over its usual: struck, a beam of 3; layer 0's lone source never (no spread).
	assert.deepEqual(read(r, shares(0, { "1,4": 0.3 + 3.5 * SD })), [1, 4, 3]);
	// Under 1% of the layer: not struck, however far over its usual.
	const r2 = { rid: "r2", tokens: [], done: false, prefill: 0 } as Reply;
	const small = (v: number) => Array.from({ length: 5 }, (_, s) => (s === 2 ? v : 1));
	for (let i = 0; i < WRITE_WARM; i++) read(r2, small(i % 2 ? 0.003 : 0.001));
	assert.deepEqual(read(r2, small(0.03)), [], "0.03 of 4.03: under 1%");
	assert.deepEqual(read(r2, small(0.08)), [0, 2, 6], "0.08 of 4.08: struck");
});

test("streamPanel: lines + 3 lines (title, history, keys, depth marks); two columns a key, one, none; the reads view by source", () => {
	const th = dockTheme();
	const r = { rid: "r", tokens: [], done: false, prefill: 0 } as Reply;
	const toks = Array.from({ length: 30 }, (_, i) => decide(r, tok(i).writes));
	assert.deepEqual([keyWidth(97, 48), keyWidth(60, 48), keyWidth(40, 48), keyWidth(97, 0)], [2, 1, 0, 0]);
	for (const [w, kw] of [[97, 2], [60, 1]]) for (const view of ["spikes", "largest"] as const) {
		const lines = streamPanel(th, toks, 29, S4, view, w, 8);
		assert.equal(lines.length, 8 + 3, `${w} ${view}`);
		assert.ok(lines.every((l: string) => plain(l).length === w), `${w} ${view}: every line its width`);
		const t = lines.map(plain);
		assert.equal(t[0].trimEnd(), `RESIDUAL STREAM  ${w === 97 ? "hyper-connections: 4 residual streams" : "4 streams"} · tracking ${view}`);
		assert.equal(t[9].trimEnd(), "▆".repeat(48 * kw), "a key a layer");
		assert.match(t[10], kw === 2 ? /^L0 {14}L8 {14}L16 {13}L24/ : /^L0 {6}L8 {6}L16 {5}L24/, "depth marks every 8 layers");
	}
	assert.match(plain(streamPanel(th, toks, 29, S4, "spikes", 97, 8, { focused: true })[0]), /^ RESIDUAL STREAM  {2}hyper-connections/, "in focus: inverse, padded");
	// Full size: the same count (the legend one of its lines) and width, every view, four streams and one with sources.
	for (const [shape, views] of [[S4, ["spikes", "largest"]], [{ L: 48, k: 1, rk: 5 }, ["spikes", "reads"]]] as const) for (const view of views) {
		const full = streamPanel(th, toks, 29, shape, view, 97, 20, { full: true });
		assert.equal(full.length, 20 + 3, `full size ${shape.k} ${view}`);
		assert.ok(full.every((l: string) => plain(l).length === 97), `full size ${shape.k} ${view}: every line its width`);
	}
	// largest: every cell of the history lit (the newest 16 tokens); spikes: only where struck.
	const big = streamPanel(th, toks, 29, S4, "largest", 97, 8).slice(1, 9).map(plain);
	assert.ok(big.every((l: string) => /^[▀▄]{96}/.test(l)), "largest: a full sheet");
	const odd = streamPanel(th, toks, 29, S4, "spikes", 97, 8).slice(1, 9).map(plain).join("");
	assert.ok(/[▀▄]/.test(odd) && / /.test(odd), "spikes: some struck, most not");
	// A struck key glows in its stream's colour on the key line (stream 3's pink); the reads view its source's (the fifth: violet).
	const one = { rid: "x", tokens: [], done: false, prefill: 0 } as Reply;
	const quiet = Array.from({ length: WRITE_WARM }, (_, i) => decide(one, flat(2, () => usual(i))));
	quiet.push(decide(one, flat(2, () => 1.1, { "1,2": 9 })));
	const lines = streamPanel(th, quiet, WRITE_WARM, { L: 2, k: 4, rk: 0 }, "spikes", 97, 3);
	assert.ok(lines[4].includes("\x1b[38;2;244;114;182m▆▆"), "the key struck now: full pink");
	assert.match(plain(lines[3]), /^ {2}[▄█▀]{2} *$/, "its beam's head right over the key, the newest token lowest");
	const reads = [...quiet.map(() => ({})), { readStrikes: Uint16Array.from([0, 4, 1]) }];
	assert.ok(streamPanel(th, reads, reads.length - 1, { L: 2, k: 1, rk: 5 }, "reads", 97, 3)[4].includes("\x1b[38;2;167;139;250m▆▆"), "source 5: violet");
	assert.ok(!streamPanel(th, reads, reads.length - 1, { L: 2, k: 1, rk: 5 }, "spikes", 97, 3)[4].includes("\x1b[38;2;167;139;250m"), "spikes draws the writes' strikes only");
});

test("views: each where the model's data has it; a stored one it lacks drawn as its first, the pref kept", () => {
	assert.deepEqual(streamViews(4, 0), ["spikes", "largest"], "writes into four streams");
	assert.deepEqual(streamViews(1, 5), ["spikes", "reads"], "one stream, reads from five sources: no largest");
	assert.deepEqual(streamViews(1, 0), ["spikes"]);
	assert.deepEqual(streamViews(0, 5), ["reads"]);
	assert.deepEqual([viewOf(undefined, ["spikes", "largest"]), viewOf("reads", ["spikes", "largest"]), viewOf("reads", ["spikes", "reads"])], ["spikes", "spikes", "reads"]);
	assert.deepEqual([nextView(undefined, ["spikes", "largest"]), nextView("largest", ["spikes", "largest"]), nextView("reads", ["spikes", "largest"]), nextView(undefined, ["spikes"])],
		["largest", "spikes", "largest", "spikes"]);
	assert.deepEqual([storedView("reads"), storedView("largest"), storedView("off"), storedView(3)], ["reads", "largest", undefined, undefined]);
});

test("STREAM full size: its full title, the legend line under it (none for one stream's spikes), the history a line shorter for it", () => {
	const open = (o: { k?: number; rk?: number }, view?: StreamView) => {
		const { d, view: v } = boardAt(63, { ...o, host: { stream: view } });
		v();
		d.stepIn(-1);
		d.handleInput("5");
		const ls = v(), { at, lines } = streamIn(ls);
		return { title: ls[at].trim().replace(/^RESIDUAL STREAM +/, ""), next: ls[at + 1].trim(), lines };
	};
	const four = open({}), reads = open({ k: 1, rk: 5 }, "reads"), one = open({ k: 1 });
	assert.deepEqual([four.title, four.next], ["hyper-connections: 4 residual streams · tracking spikes", "■ stream 1  ■ stream 2  ■ stream 3  ■ stream 4"]);
	assert.deepEqual([reads.title, reads.next], ["5 sources · tracking reads", "■ embedding  ■ block 1  ■ block 2  ■ block 3  ■ running sum"]);
	assert.equal(one.title, "1 residual stream · tracking spikes");
	assert.doesNotMatch(one.next, /■/, "one stream's spikes: no legend");
	assert.equal(four.lines, one.lines, "the legend's line from the history");
});

test("? on STREAM by what the model has: four streams, one stream with reads, one without; at the narrowest board, three lines at most; full size whole from 39 rows, in the column from 42 (STREAM at its least under it)", () => {
	const seen: string[] = [];
	for (const [o, want] of [[{}, streamExplain(4, 0)], [{ k: 1, rk: 5 }, streamExplain(1, 5)], [{ k: 1 }, streamExplain(1, 0)]] as const)
		for (const rows of [39, 40, 41, 42, 43, 44, 45, 46, 63]) for (const open of [false, true]) {
			const { d, view } = boardAt(rows, { ...o, moe: true, host: { panel: "STREAM" } });
			view();
			d.stepIn(-1);
			if (open) d.handleInput("5");
			d.handleInput("?");
			const ls = view(), first = ls.findIndex((l) => l.startsWith(want.slice(0, 30))), what = `${want.slice(0, 12)} ${rows}${open ? " full size" : ""}`;
			assert.ok(first > 0, what);
			const end = ls.findIndex((l, i) => i > first && (l === "" || /^1-5 open/.test(l) || i === ls.length - 1));
			const got = ls.slice(first, end < 0 ? ls.length : end).join(" ").replace(/\s+/g, " ");
			if (got !== want) seen.push(what);
			else assert.ok(end - first === (o.k === 1 && !o.rk ? 2 : 3), what);
			assert.ok(end - first <= 3, `${what}: ${end - first} lines`);
		}
	assert.deepEqual(seen, ["Each colour  39", "Each colour  40", "Each colour  41", "A column is  39", "A column is  40", "A column is  41", "A column is  39", "A column is  40"],
		"cut (the cut marked) only in the column at 39 to 41 rows, the keys under it");
	assert.match(streamExplain(4, 0), /hyper-connections/);
	assert.match(streamExplain(1, 5), /5 sources/);
	assert.doesNotMatch(streamExplain(1, 0), /Reads|colour/);
});

/** A board `rows` lines tall on a 30-token reply of model "Flash" with `L` layers' writes into `k` streams (none: 0),
 * reads from `rk` sources, routing where `moe` (tok); `dense`: the model known to have no experts. */
function boardAt(rows: number, o: { width?: number; L?: number; k?: number; rk?: number; moe?: boolean | number; dense?: boolean; host?: Record<string, any>; done?: boolean; lens?: boolean } = {}) {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true, writes: true };
	onEvent(live, { type: "start", rid: "r", model: "Flash" });
	for (let i = 0; i < 30; i++) onEvent(live, { ...tok(i, o.L ?? 48, "r", o.k ?? 4, o.rk ?? 0, o.moe), ...(o.lens && LENS_READ(i)) });
	if (o.done !== false) onEvent(live, { type: "end", rid: "r", n_tokens: 30, trace_id: "T" });
	if (o.dense) live.dense = "Flash";
	const host: any = { live, captions: false, close() {}, unfocus() {}, panel: "LENS", fetchJson: async () => null, layoutChanged() {}, ...o.host };
	const d = new Dock({ terminal: { rows, columns: 255 }, requestRender() {} }, { getColorMode: () => "truecolor" }, host);
	return { d, host, live, view: () => text(d.render(o.width ?? BOARD_MIN_WIDTH)) };
}
/** A board line's right column (after the middle's " │ "), or the whole line where it is not split (full size). */
const LW = columns(BOARD_MIN_WIDTH - 3).lw;
const right = (l: string) => (/^ │( |$)/.test(l.slice(LW, LW + 3)) ? l.slice(LW + 3) : l);
/** STREAM on a board's lines: where its title is (-1 none) and its history lines. */
const streamIn = (ls: string[]) => {
	const rs = ls.map(right), at = rs.findIndex((l) => /^ ?RESIDUAL STREAM {1,2} ?(hyper-connections: )?\d+ (residual )?(streams?|sources)/.test(l));
	return { at, lines: at < 0 ? 0 : rs.slice(at + 1).findIndex((l) => /^▆/.test(l)) };
};
/** EXPERTS' layers drawn in the right column. */
const layersIn = (ls: string[]) => ls.map(right).filter((l) => /^[▸ ] layer \d+ of \d+$/.test(l)).length;

test("board: STREAM last in the right column, under EXPERTS and a rule, on the key line's rule; a key a column; 8 lines of history at 63 rows; the left column the middle's height", () => {
	const ls = boardAt(63, { moe: true }).view(), rs = ls.map(right), { at, lines } = streamIn(ls);
	assert.equal(lines, STREAM_LINES);
	assert.ok(rs.slice(0, at).some((l) => /^EXPERTS /.test(l)), "under EXPERTS");
	assert.equal(rs[at - 1], "─".repeat(RIGHT_W), "the column's rule over it");
	assert.equal(rs[at + lines + 1], "▆".repeat(48), "a key a layer, a column each, left-aligned");
	assert.match(rs[at + lines + 2], /^L0 {6}L8 {6}L16 {5}L24 {5}L32 {5}L40$/, "depth marks every 8 layers");
	assert.match(ls[at + lines + 3], /^─{97}$/, "then the key line's rule");
	assert.equal(at + lines + 5, ls.length, "and the key line last");
	assert.equal(ls.length, 63);
	assert.ok(ls.slice(at, at + lines + 3).every((l) => l.slice(LW, LW + 2) === " │"), "the left column down to the key line's rule");
	assert.equal(layersIn(ls), 3, "EXPERTS: three layers over it");
});

test("board: at 63, 50, 45, 41, 40 and 36 rows every line is there and its width, ? on or off; EXPERTS gives up layers, down to one of this model's, before STREAM's history shrinks, at least 3; 512 and 128 experts", () => {
	const seen: Record<string, [number, number, number, number][]> = {};
	for (const [moe, L] of [[512, 48], [128, 24]]) {
		const shape = `${moe} experts, ${L} layers`, at = (rows: number, o: Parameters<typeof boardAt>[1] = {}) => boardAt(rows, { moe, L, ...o });
		seen[shape] = [];
		for (const rows of [63, 50, 45, 41, 40, 36]) {
			const raw = at(rows).d.render(BOARD_MIN_WIDTH), ls = text(raw), without = at(rows, { L: 0 }).view(), { lines } = streamIn(ls);
			assert.equal(raw.length, rows);
			assert.ok(raw.every((l) => visibleWidth(l) === BOARD_MIN_WIDTH), `${shape}, ${rows} rows: every line the dock's width`);
			if (lines && lines < STREAM_LINES) assert.equal(layersIn(ls), 1, `${shape}, ${rows} rows: the history shrinks only with one layer left`);
			// No line of the right column left blank over STREAM: EXPERTS' layers, the history, as many as fit.
			const rs = ls.map(right), st = streamIn(ls).at;
			if (st > 0) assert.notEqual(rs[st - 2].trim(), "", `${shape}, ${rows} rows: no blank line over STREAM`);
			seen[shape].push([rows, lines, layersIn(ls), layersIn(without)]);
			// ? on, each panel in focus: the explanation and the layout's keys where the key line was, the lines still the rows.
			const b = at(rows);
			b.view();
			b.d.stepIn(-1);
			b.host.captions = true;
			for (const p of ["MAP", "LENS", "TOKENS", "EXPERTS", "STREAM"]) {
				b.host.panel = p;
				const q = b.d.render(BOARD_MIN_WIDTH);
				assert.equal(q.length, rows, `${shape}, ${rows} rows, ? on ${p}`);
				assert.ok(q.every((l) => visibleWidth(l) === BOARD_MIN_WIDTH), `${shape}, ${rows} rows, ? on ${p}`);
				if (p !== "STREAM") assert.ok(text(q).includes(layoutKeys("1-5")), `${shape}, ${rows} rows, ? on ${p}: the layout's keys`);   // STREAM in focus keeps its least instead
			}
		}
	}
	assert.deepEqual(seen, {
		"512 experts, 48 layers": [[63, 8, 3, 4], [50, 8, 1, 3], [45, 8, 1, 2], [41, 5, 1, 2], [40, 4, 1, 2], [36, 0, 1, 1]],
		"128 experts, 24 layers": [[63, 8, 4, 7], [50, 8, 4, 4], [45, 8, 2, 4], [41, 8, 1, 4], [40, 8, 1, 4], [36, 4, 1, 3]],
	}, "[rows, history, EXPERTS' layers, without STREAM]");
});

test("board: a model that records no writes: TOKENS and EXPERTS on the right, nothing under them", () => {
	for (const rows of [63, 45]) {
		const ls = boardAt(rows, { moe: true, L: 0 }).view(), rs = ls.map(right);
		assert.equal(streamIn(ls).at, -1);
		const ex = rs.findIndex((l) => /^EXPERTS /.test(l)), end = ls.findIndex((l, i) => i > ex && /^─{97}$/.test(l));
		assert.ok(ex > 0 && !rs.slice(ex, end).some((l) => /^─+$/.test(l)), `${rows}: EXPERTS the column's last panel`);
	}
});

test("board: a model without experts that records writes: STREAM under TOKENS, its history the rest of the column", () => {
	const ls = boardAt(63, { dense: true }).view(), rs = ls.map(right), { at, lines } = streamIn(ls);
	assert.ok(!ls.some((l) => /EXPERTS/.test(l)));
	assert.match(rs[at - 1], /^─+$/);
	assert.ok(rs.slice(0, at - 1).some((l) => /^TOKENS /.test(l)), "under TOKENS");
	assert.ok(lines > STREAM_LINES, `the rest of the column (${lines} lines)`);
	assert.match(ls[at + lines + 3], /^─{97}$/, "down to the key line's rule");
	assert.equal(ls.length, 63);
	for (const rows of [36, 40]) assert.ok(streamIn(boardAt(rows, { dense: true }).view()).lines >= STREAM_MIN, `${rows} rows: under TOKENS' least alone`);
});

test("board: STREAM not drawn on a model that records none, on the one-panel board or the Steer page; past 50 layers the column's 50 keys share them", () => {
	assert.equal(streamIn(boardAt(63, { L: 0 }).view()).at, -1, "the reply came without: none recorded");
	for (const L of [50, 51, 128]) assert.match(boardAt(63, { L }).view().map(right).find((l) => /^▆/.test(l))!, /^▆{50}$/, `${L} layers: 50 keys`);
	assert.equal(streamIn(boardAt(30).view()).at, -1, "the one-panel board");
	const { d, view } = boardAt(63);
	d.stepIn(-1);
	d.handleInput("6");
	assert.equal(streamIn(view()).at, -1, "the Steer page");
});

test("5 opens STREAM full size, exact where the column shares its keys (64 layers: 50 keys in the column, 64 full size); keys before a frame go by it; 5 or 4 do nothing where the panel cannot be", () => {
	const { d, host, view } = boardAt(63, { L: 64 });
	view();
	d.stepIn(-1);
	assert.equal(view().map(right).find((l) => /^▆/.test(l)), "▆".repeat(50), "the column: the layers spread over it");
	d.handleInput("5");
	const ls = view();
	assert.deepEqual([d.full, host.panel, d.streamRoom], [true, "STREAM", true], "open, the writes asked for");
	assert.equal(ls.find((l) => /^▆/.test(l)), "▆".repeat(64), "full size: a column a key");
	assert.ok(ls.some((l) => /^ ?1 MAP +2 LENS +3 TOKENS +4 EXPERTS +5 STREAM +5 back$/.test(l)), "in the tab row");
	d.handleInput("5");
	view();
	assert.deepEqual([d.full, d.streamRoom], [false, true], "the board: in the column, asked");
	// Keys with no frame between them go by what the number left: 5 5 the board again; 5 then s its next view.
	d.handleInput("5");
	d.handleInput("5");
	assert.equal(d.full, false, "5 5: the board");
	d.handleInput("5");
	d.handleInput("s");
	assert.deepEqual([d.full, host.stream], [true, "largest"], "5 s: the view switched");
	// ? on another panel full size at 41 rows: no room kept for STREAM (the column not drawn), the explanation whole.
	const q = boardAt(41, { L: 64, moe: true });
	q.view();
	q.d.stepIn(-1);
	q.d.handleInput("?");
	for (const k of ["1", "2", "3", "4"]) {
		q.d.handleInput(k);
		assert.ok(!q.view().some((l) => l.endsWith(" …")), `${k} then ?: not cut`);
		q.d.handleInput(k);
	}
	// A board too short for STREAM in the column (100x37, 512 experts): the writes asked for while the model's streams
	// are not known (another model's reply, or none seen yet), so 5 can open it after one reply.
	const s = boardAt(37, { moe: true });
	s.view();
	assert.equal(s.d.streamRoom, false, "known, no room: not asked");
	s.live.writes = false;   // as the widget then has it: the next replies start without
	onEvent(s.live, { type: "start", rid: "o", model: "Other" });
	onEvent(s.live, tok(0, null, "o", 4, 0, true));
	s.view();
	assert.equal(s.d.streamRoom, true, "another model, not known: asked");
	s.live.streams = undefined;
	onEvent(s.live, { type: "start", rid: "p", model: "Flash" });
	onEvent(s.live, tok(0, null, "p", 4, 0, true));
	s.view();
	assert.equal(s.d.streamRoom, true, "none seen yet: asked");
	for (const [o, key] of [[{ L: 0 }, "5"], [{ dense: true }, "4"]] as const) {
		const b = boardAt(63, o);
		b.view();
		b.d.stepIn(-1);
		b.d.handleInput(key);
		assert.deepEqual([b.d.full, b.host.panel], [false, "LENS"], `${key} on ${JSON.stringify(o)}: nothing`);
	}
});

test("STREAM a panel: tab reaches it after EXPERTS, its title inverse in focus; ? explains it; 5 opens it over the panels, a column a key as in the column", () => {
	const { d, host, view } = boardAt(63, { host: { panel: "EXPERTS" } });
	view();
	d.stepIn(-1);
	d.handleInput("\t");
	assert.equal(host.panel, "STREAM");
	let ls = view();
	assert.ok(ls.map(right).some((l) => /^ RESIDUAL STREAM {3}4 streams · tracking spikes/.test(l)), "in focus: inverse");
	assert.match(ls.at(-1)!, /^tab map · ←\/→ token · s view · 1-6 panel · \? explain · esc back$/);
	d.handleInput("\t");
	assert.equal(host.panel, "MAP", "round to MAP");
	d.handleInput("\x1b[Z");
	assert.equal(host.panel, "STREAM", "shift+tab back");
	d.handleInput("?");
	ls = view();
	assert.ok(ls.join(" ").includes(streamExplain(4, 0).slice(0, 40)), "? explains it");
	assert.ok(ls.includes(layoutKeys("1-5")), "the layout's keys under it");
	d.handleInput("?");
	d.handleInput("5");
	ls = view();
	const { at, lines } = streamIn(ls);
	assert.match(ls[at - 1], /^ ?1 MAP +2 LENS +3 TOKENS +4 EXPERTS +5 STREAM +5 back$/, "the tab row over it");
	assert.ok(lines > 20, `full size: the history fills the panels' room (${lines} lines)`);
	assert.equal(ls.find((l) => /^▆/.test(l)), "▆".repeat(48), "a column a key (48 layers)");
	assert.equal(ls.filter((l) => /RESIDUAL STREAM/.test(l)).length, 1, "once, not in the column too");
	d.handleInput("2");
	assert.equal(streamIn(view()).at, -1, "another panel full size: over the column, STREAM too");
	d.handleInput("2");
	assert.ok(streamIn(view()).at > 0 && !d.full, "2 again: the board");
});

test("STREAM's s switches the view the model's data has: four streams spikes and largest, one stream with reads spikes and reads, without only spikes; enter nothing; s elsewhere the lens", () => {
	for (const [k, rk, want] of [[4, 0, ["largest", "spikes"]], [1, 5, ["reads", "spikes"]], [1, 0, ["spikes", "spikes"]]] as const) {
		const { d, host, view } = boardAt(63, { k, rk, lens: true, host: { panel: "STREAM" } });
		view();
		d.stepIn(-1);
		d.handleInput("\r");
		assert.equal(host.stream, undefined, "enter: nothing");
		const seen: string[] = [];
		for (let i = 0; i < 2; i++) {
			d.handleInput("s");
			assert.equal(host.lensView, undefined, "not the lens");
			seen.push(/RESIDUAL STREAM .* (\w+)$/.exec(view().map(right).find((l) => /RESIDUAL STREAM/.test(l))!)![1]);
		}
		assert.deepEqual(seen, [...want], `k ${k}, rk ${rk}`);
		assert.equal(host.stream, want[1], "remembered as chosen");
	}
	for (const panel of ["LENS", "MAP"]) {   // the lens read on both: s swaps it
		const lens = boardAt(63, { lens: true, host: { panel } });
		lens.view();
		lens.d.stepIn(-1);
		lens.d.handleInput("s");
		assert.deepEqual([lens.host.lensView, lens.host.stream], ["logit", undefined], `${panel} in focus: s the lens, not the view`);
	}
	// A view the model lacks: drawn as its first, the choice kept.
	const { host, view } = boardAt(63, { host: { stream: "reads" } });
	assert.ok(view().map(right).some((l) => /^RESIDUAL STREAM {2}4 streams · tracking spikes/.test(l)));
	assert.equal(host.stream, "reads");
});

test("board: STREAM follows the token in focus; stepped back it shows what it showed live", () => {
	const b = boardAt(63, { done: false }), live: string[][] = [];
	const band = (ls: string[]) => ls.slice(streamIn(ls).at, streamIn(ls).at + STREAM_LINES + 3);
	for (let i = 30; i < 40; i++) {
		onEvent(b.live, tok(i));
		live.push(band(b.d.render(BOARD_MIN_WIDTH)));
	}
	b.d.sel = 34;
	assert.deepEqual(band(b.d.render(BOARD_MIN_WIDTH)), live[4], "token 34 as it was");
	b.d.sel = 39;
	assert.deepEqual(band(b.d.render(BOARD_MIN_WIDTH)), live[9]);
});

test("board: STREAM stays while the next reply reads its prompt and through its tokens; a model that records none loses it at its first", () => {
	const { live, view } = boardAt(63);
	onEvent(live, { type: "start", rid: "s", model: "Flash" });
	assert.equal(streamIn(view()).lines, STREAM_LINES, "the last reply shown, its band too");
	onEvent(live, tok(0, 48, "s"));
	assert.equal(streamIn(view()).lines, STREAM_LINES);
	onEvent(live, { type: "end", rid: "s", n_tokens: 1 });
	onEvent(live, { type: "start", rid: "u", model: "Other" });
	onEvent(live, tok(0, null, "u"));
	assert.equal(streamIn(view()).at, -1, "asked for, none came");
});

test("the view remembered (pi.json `stream`); writes asked for while the board has room for STREAM", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-")), path = join(dir, "pi.json");
	const mk = () => {
		let dock: any;
		const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {},
			custom: (make: any) => new Promise<void>(() => { dock = make({ requestRender() {}, terminal: { rows: 63, columns: 255 } }, {}, {}, () => {}); }) } };
		const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: true } as any;
		createUi({ live } as any, path, () => {}).showDock(ctx);
		onEvent(live, { type: "start", rid: "r", model: "Flash" });
		for (let i = 0; i < 20; i++) onEvent(live, tok(i));
		dock.render(BOARD_MIN_WIDTH);
		dock.stepIn(-1);
		return { dock, live };
	};
	let a = mk();
	assert.deepEqual([a.dock.host.stream, a.live.writes], [undefined, true], "none remembered; shown: asked for");
	a.dock.host.panel = "STREAM";
	a.dock.handleInput("s");
	assert.equal(readPrefs(path).stream, "largest");
	a = mk();
	await tick();
	assert.deepEqual([a.dock.host.stream, a.live.writes], ["largest", true], "Pi restarted: from pi.json");
	a.dock.handleInput("m");
	a.dock.render(MINI_WIDTH);
	await tick();
	assert.equal(a.live.writes, true, "the mini dock: its STREAM");
});

test("an edit's reply carries its base's usuals on (copies): the tokens after the edit strike with no warm-up; oMLX's second reply too", () => {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true, writes: true };
	const ev = (rid: string, i: number, w: number[]) => ({ type: "token", rid, i, id: i, text: ` w${i}`, p: 0.9, entropy: 0.4, alts: [], writes: w, write_k: 4,
		reads: [0.5, 0.5], read_k: 2 });
	onEvent(live, { type: "start", rid: "r", model: "Flash" });
	for (let i = 0; i < 20; i++) onEvent(live, ev("r", i, flat(2, () => usual(i))));
	onEvent(live, { type: "end", rid: "r", n_tokens: 20 });
	const base = live.reply!;
	onEvent(live, { type: "start", rid: "e", model: "Flash", edit: { of: "r", at: 5 } });
	const e = live.reply!;
	assert.equal(e.of, "r");
	assert.ok(e.usual!.n === 20 && e.usual!.mean !== base.usual!.mean && e.usual!.m2 !== base.usual!.m2, "the base's, a copy");
	assert.ok(e.readUsual!.n === 20 && e.readUsual!.mean !== base.readUsual!.mean, "the reads' too");
	onEvent(live, ev("e", 0, flat(2, () => 1.1, { "0,2": 9 })));
	assert.deepEqual([...e.tokens[5].strikes!], [0, 2, 6], "the edited token strikes at once");
	assert.notEqual(unpackLargest(e.tokens[5].largest![0])[1], Math.round(0.55 * 31) / 31, "and is lit against the usual, not 0.55");
	assert.equal(base.usual!.n, 20, "the base's own usual untouched");
	onEvent(live, { type: "end", rid: "e", n_tokens: 1 });
	onEvent(live, { type: "start", rid: "f", model: "Flash", edit: { of: "e", at: 6 } });   // oMLX: the rest, a reply of its own
	assert.equal(live.reply!.usual!.n, 21, "carried on again, the edited token in it");
	onEvent(live, ev("f", 0, flat(2, () => 1.1, { "1,1": 9 })));
	assert.deepEqual([...live.reply!.tokens[6].strikes!], [1, 1, 6]);
});

test("board: a reply streamed without the writes on a model that records them says so in STREAM's title", () => {
	const { live, view } = boardAt(63);
	live.writes = false;   // not asked for while it streamed
	onEvent(live, { type: "start", rid: "s", model: "Flash" });
	for (let i = 0; i < 5; i++) onEvent(live, tok(i, null, "s"));
	assert.ok(view().map(right).includes(`RESIDUAL STREAM  4 streams · spikes  ${LABELS.notRecorded}`), "in the right column, tracking given way");
	assert.ok(boardAt(63).view().map(right).includes("RESIDUAL STREAM  4 streams · tracking spikes"), "recorded: nothing said");
});

test("writes asked for only where STREAM can be drawn: not on the one-panel board, the Steer page, a board too short", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-")), path = join(dir, "pi.json");
	let dock: any;
	const tui = { requestRender() {}, terminal: { rows: 63, columns: 255 } };
	const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {},
		custom: (make: any) => new Promise<void>(() => { dock = make(tui, {}, {}, () => {}); }) } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: true } as any;
	createUi({ live } as any, path, () => {}).showDock(ctx);
	onEvent(live, { type: "start", rid: "r", model: "Flash" });
	for (let i = 0; i < 20; i++) onEvent(live, tok(i, 48, "r", 4, 0, true));
	const at = async (rows: number, width = BOARD_MIN_WIDTH) => {
		tui.terminal.rows = rows;
		dock.invalidate();
		dock.render(width);
		await tick();
		return live.writes;
	};
	assert.equal(await at(63), true, "the board");
	assert.equal(await at(36), false, "a board too short for it");
	assert.equal(await at(30), false, "the one-panel board");
	dock.stepIn(-1);
	assert.ok(!text(dock.render(BOARD_MIN_WIDTH)).join(" ").includes("STREAM"), "not in the one-panel board's tab row");
	assert.equal(await at(63), true, "the board again");
	dock.handleInput("6");
	assert.equal(await at(63), false, "the Steer page");
	dock.handleInput("6");
	assert.equal(await at(63), true);
	dock.handleInput("m");
	assert.equal(await at(40, MINI_WIDTH), true, "the mini dock");
	assert.equal(await at(17, MINI_WIDTH), false, "a mini dock too short for it");
	assert.equal(await at(40, MINI_WIDTH), true);
});

/** The playhead ticked every PLAY_TICK_MS over arrivals at `times` (ms; the reply ends at the last): at each tick the
 * token shown and the newest, and when the shown token changed (`steps`: [ms, token]). */
function play(times: number[]) {
	let p: Play | undefined, t = times[0], i = 0;
	const ticks: [number, number, number][] = [], steps: [number, number][] = [];
	while (t < times.at(-1)! + 3000) {
		while (i < times.length && times[i] <= t) i++;
		const newest = i - 1, was = p && playShown(p);
		p = pace(p, "r", newest, i === times.length, t);
		ticks.push([t, playShown(p), newest]);
		if (playShown(p) !== was) steps.push([t, playShown(p)]);
		if (i === times.length && playShown(p) === newest) break;
		t += PLAY_TICK_MS;
	}
	return { ticks, steps, gaps: steps.slice(2).map(([t], k) => t - steps[k + 1][0]) };
}

test("pace: tokens in pairs a millisecond apart, a 400 ms wait among them: the playhead steps evenly, never ahead of the newest, caught up within the lag of the end", () => {
	// Pairs every 120 ms (about 17 tokens a second), a wait of 400 ms after the 20th pair, then pairs again.
	const times: number[] = [];
	for (let k = 0, t = 0; k < 40; k++, t += k === 20 ? 400 : 120) times.push(t, t + 1);
	const { ticks, steps, gaps } = play(times);
	assert.ok(ticks.every(([, shown, newest]) => shown <= newest), "never ahead of the newest");
	assert.ok(steps.every(([, k], j) => !j || k === steps[j - 1][1] + 1), "a token a step");
	const flow = gaps.filter((_, j) => steps[j + 2][0] <= times.at(-1)!);
	assert.ok(Math.max(...flow) <= 200, `no step more than 200 ms after the one before while tokens come (most ${Math.max(...flow)})`);
	const raw = Math.max(...times.slice(1).map((t, j) => t - times[j]));
	assert.ok(raw >= 399 && flow.filter((g) => g <= 2 * PLAY_TICK_MS).length > flow.length * 0.8, "most steps a tick or two apart, the arrivals 1 ms or 120-400 ms");
	assert.ok(steps.at(-1)![0] - times.at(-1)! <= PLAY_LAG_MS + PLAY_TICK_MS, "the last token shown within the lag of the end");
});

test("pace: about the lag behind at a steady rate; a backlog over PLAY_JUMP_MS jumps to the lag; another reply starts at its newest", () => {
	const times = Array.from({ length: 200 }, (_, i) => i * 50);   // 20 tokens a second
	const { ticks } = play(times), mid = ticks.filter(([t]) => t > 3000 && t < 9000);
	const lag = mid.map(([t, shown]) => t - times[shown + 1]).filter((x) => x > -Infinity);
	assert.ok(lag.every((x) => x > 0 && x <= PLAY_LAG_MS + 2 * PLAY_TICK_MS), `the token after the one shown came within the lag (${Math.max(...lag)})`);
	// A token waiting over PLAY_JUMP_MS (the playhead held back): to the tokens that came the lag ago.
	assert.equal(PLAY_JUMP_MS, 2000);
	// Token k came at k * 50 ms; at 3050 the playhead held at token `pos`, the tokens after it to 60 waiting.
	const held = (pos: number): Play => ({ rid: "r", pos, at: 3000, newest: 60, start: 0, rate: 0.02,
		arr: Array.from({ length: 60 - pos }, (_, j) => (pos + 1 + j) * 50) });
	const p = pace(held(2), "r", 61, false, 3050);
	assert.ok(p.pos >= 56 && p.pos < 58, `token 3 waiting 2.9 s: to token 56, the one that came 250 ms before 3050 (${p.pos})`);
	assert.ok(Math.abs(pace(held(30), "r", 61, false, 3050).pos - (30 + (31 / PLAY_LAG_MS) * 50)) < 1e-9, "token 31 waiting 1.5 s: no jump, the backlog over the lag a ms");
	assert.equal(pace(p, "s", 5, false, 1100).pos, 5, "another reply: its newest");
	assert.equal(pace(undefined, "r", 7, true, 0).pos, 7, "a finished reply first seen: its newest");
});

/** A board with a clock (host.now) on a reply still streaming, ticked with the mock timers; `band` its STREAM lines. */
function clocked(o: { host?: Record<string, any> } = {}) {
	const clock = { t: 0 };
	const b = boardAt(63, { done: false, host: { now: () => clock.t, ...o.host } });
	const band = () => {
		b.d.invalidate();
		const ls = text(b.d.render(BOARD_MIN_WIDTH)), at = streamIn(ls).at;
		return at < 0 ? [] : ls.slice(at, at + STREAM_LINES + 3);
	};
	const tick = (n = 1) => {
		for (let k = 0; k < n; k++) {
			clock.t += PLAY_TICK_MS;
			mock.timers.tick(PLAY_TICK_MS);
		}
	};
	return { ...b, clock, band, tick, dk: b.d as any };
}

test("the dock's playhead: STREAM a little behind the newest while it streams, the ticker stopped caught up; stepped back, the token itself at once", () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const { d, live, clock, band, tick, dk } = clocked();
		const at29 = band();
		assert.equal(dk.playLoop, undefined, "caught up: no ticker");
		onEvent(live, tok(30));
		onEvent(live, tok(31));
		assert.deepEqual(band(), at29, "two tokens at once: still token 29");
		assert.ok(dk.playLoop, "the ticker on");
		const shown: number[] = [];
		for (let k = 0; k < 20 && dk.playLoop; k++) {
			tick();
			band();
			shown.push(playShown(dk.play));
		}
		assert.deepEqual([...new Set(shown)], [29, 30, 31], "a token at a time");
		assert.ok(clock.t <= 2 * PLAY_LAG_MS, `caught up within twice the lag (${clock.t} ms)`);
		assert.equal(dk.playLoop, undefined, "caught up: the ticker stopped");
		// Stepped back while it is behind: the token itself, the ticker off.
		onEvent(live, tok(32));
		onEvent(live, tok(33));
		band();
		assert.ok(dk.playLoop);
		d.sel = 33;
		const exact = band();
		assert.equal(dk.playLoop, undefined);
		d.sel = -1;
		assert.deepEqual(band(), exact, "followed again: from the newest");
		// Not drawn (the Steer page), the ticker stops; asked to draw and no draw came (the dock hidden by Pi): stops too.
		onEvent(live, tok(34));
		onEvent(live, tok(35));
		band();
		assert.ok(dk.playLoop);
		dk.steerPage = true;
		band();
		assert.equal(dk.playLoop, undefined, "the Steer page");
		dk.steerPage = false;
		band();
		onEvent(live, tok(36));
		onEvent(live, tok(37));
		band();
		assert.ok(dk.playLoop);
		tick(12);
		assert.equal(dk.playLoop, undefined, "no draw came");
		onEvent(live, tok(38));
		onEvent(live, tok(39));
		band();
		assert.ok(dk.playLoop);
		d.dispose();
		assert.equal(dk.playLoop, undefined, "the dock closed");
	} finally {
		mock.timers.reset();
	}
});

test("the dock's playhead: none on a finished reply, without a clock; the other panels show the newest", () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const { live, band, dk, d } = clocked();
		band();
		onEvent(live, tok(30));
		onEvent(live, tok(31));
		band();
		assert.ok(dk.playLoop);
		d.invalidate();
		assert.ok(text(d.render(BOARD_MIN_WIDTH)).some((l) => / w31 *$/.test(l)), "REPLY: the newest");
		onEvent(live, { type: "end", rid: "r", n_tokens: 32, trace_id: "T" });
		const none = boardAt(63, { done: false });
		onEvent(none.live, tok(30));
		none.view();
		assert.equal((none.d as any).playLoop, undefined, "no clock: none");
		const done = clocked();
		onEvent(done.live, { type: "end", rid: "r", n_tokens: 30, trace_id: "T" });
		done.band();
		assert.equal(done.dk.playLoop, undefined, "finished: none");
	} finally {
		mock.timers.reset();
	}
});

test("STREAM not drawn for a frame (a token without writes, a short board): the focus kept for when it is back; full size off where the panel is not drawn", () => {
	const { d, host, live, view } = boardAt(63, { done: false, host: { panel: "STREAM" } });
	view();
	d.stepIn(-1);
	onEvent(live, { type: "end", rid: "r", n_tokens: 30 });
	onEvent(live, { type: "start", rid: "s", model: "Flash" });
	onEvent(live, tok(0, null, "s"));   // the first token came without: none known for a frame
	assert.equal(streamIn(view()).at, -1);
	assert.equal(host.panel, "STREAM", "the focus kept");
	assert.match(view().at(-1)!, /^tab lens · /, "drawn on MAP for the frame");
	onEvent(live, tok(1, 48, "s"));
	assert.ok(view().map(right).some((l) => /^ RESIDUAL STREAM/.test(l)), "back, in focus");
	// STREAM full size: kept at the board's least height; the one-panel board closes it.
	d.handleInput("5");
	const tui = (d as any).tui.terminal;
	tui.rows = 36;
	assert.ok(streamIn(view()).lines > 10 && d.full, "36 rows: STREAM full size still");
	tui.rows = 30;
	view();
	tui.rows = 63;
	const ls = view();
	assert.deepEqual([d.full, host.panel], [false, "STREAM"], "closed; the focus kept");
	assert.equal(streamIn(ls).lines, STREAM_LINES, "the board, STREAM in the right column at STREAM_LINES");
});

test("s view only where STREAM has more than one view", () => {
	for (const [k, rk, has] of [[4, 0, true], [1, 5, true], [1, 0, false]] as const) {
		const { d, view } = boardAt(63, { k, rk, host: { panel: "STREAM" } });
		view();
		d.stepIn(-1);
		assert.equal(view().at(-1)!.includes("s view"), has, `k ${k}, rk ${rk}`);
	}
});

test("? on a short board with STREAM shown: the layout's keys stay, STREAM giving way where both do not fit; in focus, STREAM stays", () => {
	const seen: string[] = [];
	for (let rows = 36; rows <= 46; rows++) {
		const { d, host, view } = boardAt(rows, { moe: true });
		view();
		d.stepIn(-1);
		const shown = streamIn(view()).at >= 0;
		host.captions = true;
		let ls = view();
		assert.ok(ls.includes(layoutKeys("1-5")), `${rows}: the layout's keys under the explanation`);
		assert.equal(ls.length, rows);
		host.panel = "STREAM";
		const focused = streamIn(view()).at >= 0;
		assert.equal(focused, shown, `${rows}: STREAM in focus stays`);
		seen.push(`${rows}:${+shown}${+(streamIn(ls).at >= 0)}`);
	}
	assert.equal(seen.join(" "), "36:00 37:00 38:00 39:10 40:11 41:11 42:11 43:11 44:11 45:11 46:11", "rows:[drawn][drawn under ?]");
});

test("spread: column c the layers floor(c L / W) up to floor((c + 1) L / W); two in a column, the stronger wins", () => {
	// 48 layers on 32 columns: column 0 layer 0, column 1 layers 1 and 2. A strike on each: the longer beam wins.
	const two = spread({ strikes: Uint16Array.of(1, 1, 2, 2, 3, 4), readStrikes: Uint16Array.of(2, 0, 1, 1, 4, 3) }, 48, 32);
	assert.deepEqual([...two.strikes!], [1, 3, 4]);
	assert.deepEqual([...two.readStrikes!], [1, 4, 3]);
	// largest: the brighter (its low 5 bits), its stream with it.
	const b = new Uint16Array(48);
	b[1] = packLargest(1, 10 / 31), b[2] = packLargest(2, 20 / 31), b[47] = packLargest(3, 1);
	const big = spread({ largest: b }, 48, 32).largest!;
	assert.equal(big.length, 32);
	assert.deepEqual([big[1], big[31]], [b[2], b[47]], "the brighter; the last column the last layer");
});

test("streamPanel past its width (51 or more layers in the right column's 50): the layers spread over it, a key a column; at or under it exact, one or two columns a key", () => {
	const th = dockTheme();
	const marks: Record<number, string> = { 24: "L0              L8              L16", 42: "L0      L8      L16     L24     L32     L40", 48: "L0      L8      L16     L24     L32     L40",
		64: "L0           L16         L32          L48", 97: "L0      L16      L32     L48     L64     L80", 128: "L0          L32          L64         L96" };
	for (const L of [24, 42, 48, 64, 97, 128]) {   // 42: L40 past the last key, inside the line
		const r = { rid: "r", tokens: [], done: false, prefill: 0 } as Reply;
		const toks = Array.from({ length: 30 }, (_, i) => decide(r, tok(i, L).writes));
		for (const view of ["spikes", "largest"] as const) {
			const t = streamPanel(th, toks, 29, { L, k: 4, rk: 0 }, view, RIGHT_W, 8).map(plain), what = `${L} layers, ${view}`, cols = L > RIGHT_W ? RIGHT_W : L * keyWidth(RIGHT_W, L);
			assert.equal(t.length, 8 + 3, what);
			assert.ok(t.every((l) => l.length === RIGHT_W), `${what}: every line its width`);
			assert.equal(t[9].trimEnd(), "▆".repeat(cols), `${what}: ${cols} columns of keys`);
			assert.equal(t[10].trimEnd(), marks[L], what);
			if (view === "largest") assert.ok(t.slice(1, 9).every((l) => l.trimEnd() === "▀".repeat(cols)), `${what}: a full sheet, end to end`);
		}
	}
});

test("STREAM's title: what the model has, then what is tracked; tracking goes first where it does not fit, then the count, not recorded keeping its place", () => {
	const th = dockTheme(), title = (s: { L: number; k: number; rk: number }, v: StreamView, w: number, o = {}) => plain(streamPanel(th, [], 0, s, v, w, 3, o)[0]).trimEnd();
	const S1 = { L: 48, k: 1, rk: 0 }, R5 = { L: 48, k: 1, rk: 5 };
	assert.deepEqual([title(S4, "spikes", 97), title(S4, "largest", 97), title(S1, "spikes", 97), title(R5, "reads", 97)],
		["RESIDUAL STREAM  hyper-connections: 4 residual streams · tracking spikes", "RESIDUAL STREAM  hyper-connections: 4 residual streams · tracking largest",
			"RESIDUAL STREAM  1 residual stream · tracking spikes", "RESIDUAL STREAM  5 sources · tracking reads"], "full size");
	assert.deepEqual([title(S4, "spikes", 50), title(S1, "spikes", 50), title(R5, "reads", 50)],
		["RESIDUAL STREAM  4 streams · tracking spikes", "RESIDUAL STREAM  1 stream · tracking spikes", "RESIDUAL STREAM  5 sources · tracking reads"], "the right column");
	assert.deepEqual([title(R5, "spikes", 97), title(R5, "spikes", 50), title(R5, "spikes", 36)],
		["RESIDUAL STREAM  block sum · tracking spikes", "RESIDUAL STREAM  block sum · tracking spikes", "RESIDUAL STREAM  block sum · spikes"], "one stream with reads (Inkling 2): a block's sum, full size too");
	assert.deepEqual([title(S4, "largest", 50, { swap: true }), title(R5, "spikes", 50, { swap: true }), title(S4, "spikes", 50, { note: LABELS.notRecorded }), title(S4, "spikes", 50, { swap: true, note: LABELS.notRecorded }), title(S4, "spikes", 30, { note: LABELS.notRecorded })],
		["RESIDUAL STREAM  4 streams · tracking largest · s", "RESIDUAL STREAM  block sum · tracking spikes · s", `RESIDUAL STREAM  4 streams · spikes  ${LABELS.notRecorded}`, `RESIDUAL STREAM  spikes · s  ${LABELS.notRecorded}`, `RESIDUAL STREAM  ${LABELS.notRecorded}`]);
});

test("STREAM's legend (full size): a swatch and a name a colour in use; none for one stream's spikes", () => {
	const th = dockTheme(), at = (s: { L: number; k: number; rk: number }, v: StreamView) => streamPanel(th, [], 0, s, v, 97, 3, { full: true });
	const four = at(S4, "largest");
	assert.equal(four.length, 3 + 3, "the legend's line out of the history's");
	assert.equal(plain(four[1]).trimEnd(), "■ stream 1  ■ stream 2  ■ stream 3  ■ stream 4");
	assert.ok(four[1].includes("\x1b[38;2;255;180;84m■") && four[1].includes("\x1b[38;2;155;229;100m■"), "each in its stream's colour");
	assert.equal(plain(at({ L: 48, k: 1, rk: 5 }, "reads")[1]).trimEnd(), "■ embedding  ■ block 1  ■ block 2  ■ block 3  ■ running sum");
	assert.equal(at({ L: 48, k: 1, rk: 5 }, "spikes").length, 3 + 3, "one stream's spikes: none");
	assert.deepEqual([hasLegend("spikes", 4), hasLegend("spikes", 1), hasLegend("reads", 1)], [true, false, true]);
});

test("STREAM full size: the band as the right column draws its keys (two columns a layer up to 25, one up to the width, spread past it); nothing beside it", () => {
	const th = dockTheme();
	for (const [L, cols] of [[24, 48], [48, 48], [64, 64], [90, 90], [128, 97]] as const) {
		const { live } = boardAt(63, { L }), toks = live.reply!.tokens;
		const t = streamPanel(th, toks, 29, live.streams!, "spikes", 97, 12, { full: true }).map(plain), what = `${L} layers`;
		assert.equal(t.length, 1 + 12 + 2, `${what}: the legend's line out of the 12`);
		assert.ok(t.every((l) => l.length === 97), `${what}: every line its width`);
		assert.equal(t.at(-2)!.trimEnd(), "▆".repeat(cols), `${what}: ${cols} columns of keys`);
		assert.ok(t.slice(2, 13).every((l) => l.slice(cols).trim() === ""), `${what}: no text beside the history`);
		if (L === 128) assert.equal(t.at(-1), plain(streamPanel(th, toks, 29, live.streams!, "spikes", 97, 12).at(-1)!), "spread as the width's column: its marks");
	}
});

test("STREAM full size: the column's lines, the newest tokens k - 1 and k, whatever k", () => {
	const th = dockTheme(), { live } = boardAt(63), toks = live.reply!.tokens, s = live.streams!;
	for (const k of [28, 29]) for (const [t, half] of [[k, "▄"], [k - 1, "▀"]] as const) {
		const one = toks.map((_, i) => ({ strikes: i === t ? Uint16Array.of(0, 0, 1) : undefined }));
		const ls = streamPanel(th, one, k, s, "spikes", 97, 9, { full: true }).slice(2, 10).map(plain);
		assert.equal(ls.findIndex((l) => l[0] !== " "), 7, `#${t} at k ${k}: the newest line`);
		assert.equal(ls[7][0], half, `#${t} at k ${k}`);
	}
});

test("STREAM full size fades old lines part way (FULL_FADE); the right column and the mini dock all the way, no legend", () => {
	const th = dockTheme(), { live } = boardAt(63), s = live.streams!, R = 2 * 8;
	const flat = live.reply!.tokens.map(() => ({ largest: Uint16Array.from({ length: 48 }, () => 31) }));   // stream 0 at full on every layer
	const top = (full: boolean) => streamPanel(th, flat, 29, s, "largest", 97, full ? 9 : 8, { full })[full ? 2 : 1];
	const at = (a: number) => colourCode(mix(COLOURS.bg, STREAM_RGB[0], a), "truecolor");
	assert.ok(top(true).includes(at(Math.pow(1 - (FULL_FADE * (R - 1)) / R, STREAM_LOOK.age))), "full size: the oldest row faded by FULL_FADE of its age");
	assert.ok(top(false).includes(at(Math.pow(1 - (R - 1) / R, STREAM_LOOK.age))), "the column: all the way");
	for (const ls of [boardAt(63).view().map(right), miniAt(40).view()]) {
		const at = ls.findIndex((l) => /^RESIDUAL STREAM/.test(l));
		assert.ok(at > 0 && !/■/.test(ls[at + 1]), "not full size: no legend");
	}
});

test("STREAM full size, every view and both shapes: each frame exactly its rows and width, 36 to 63 rows, on an even and an odd token", () => {
	for (const o of [{}, { k: 1, rk: 5 }] as const) for (const view of ["spikes", "largest", "reads"] as const) for (let rows = 36; rows <= 63; rows += 3) for (const sel of [20, 21]) {
		const { d } = boardAt(rows, { ...o, host: { panel: "STREAM", stream: view } });
		d.render(BOARD_MIN_WIDTH);
		d.stepIn(sel);
		d.handleInput("5");
		const raw = d.render(BOARD_MIN_WIDTH), what = `${JSON.stringify(o)} ${view} ${rows} #${sel}`;
		assert.ok(d.full, what);
		assert.equal(raw.length, rows, what);
		assert.ok(raw.every((l) => visibleWidth(l) === BOARD_MIN_WIDTH), what);
	}
});

test("s on STREAM with a clock: the switch drawn as the lens swap is (streamFrames), the pref set at once; the last frame the new view sharp, as without a clock; s again starts over; a key or dispose stops it", () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const at = (o: { full?: boolean; mini?: boolean; clock?: boolean } = {}) => {
			const b = o.mini ? miniAt(40, { host: { now: o.clock ? () => 0 : undefined } }) : boardAt(63, { host: { panel: "STREAM", now: o.clock ? () => 0 : undefined } });
			const w = o.mini ? MINI_WIDTH : BOARD_MIN_WIDTH, draw = () => b.d.render(w);
			draw();
			b.d.stepIn(-1);
			if (o.full) b.d.handleInput("5");
			return { ...b, draw, title: () => text(draw()).map(right).find((l) => /RESIDUAL STREAM/.test(l))! };
		};
		for (const o of [{}, { full: true }, { mini: true }]) {
			const ref = at(o), sharp = { spikes: ref.draw() } as Record<string, string[]>;
			ref.d.handleInput("s");
			sharp.largest = ref.draw();
			assert.equal((ref.d as any).anim, undefined, `${JSON.stringify(o)}: no clock, no animation`);
			const b = at({ ...o, clock: true }), seen: string[] = [], frames: string[][] = [];
			b.d.handleInput("s");
			assert.equal(b.host.stream, "largest", "the pref at once");
			for (let i = 0; i < 9; i++, mock.timers.tick(70)) {
				const raw = b.draw();
				frames.push(raw);
				seen.push(/tracking (\w+)/.exec(b.title())?.[1] ?? /(\w+)( · s)?$/.exec(b.title())![1]);
				assert.equal(raw.length, o.mini ? 40 : 63, `${JSON.stringify(o)} frame ${i}: its rows`);
				assert.ok(raw.every((l) => visibleWidth(l) === (o.mini ? MINI_WIDTH : BOARD_MIN_WIDTH)), `${JSON.stringify(o)} frame ${i}: its width`);
			}
			assert.deepEqual(seen, ["spikes", "spikes", "spikes", "largest", "largest", "largest", "largest", "largest", "largest"], `${JSON.stringify(o)}: the view changes mid-way`);
			assert.notDeepEqual(frames[0], sharp.spikes, "the old view fogged");
			assert.notDeepEqual(frames[1], frames[0], "two levels");
			assert.notDeepEqual(frames[5], frames[6], "clearing from the deep end");
			assert.deepEqual(frames[8], sharp.largest, "the last frame: the new view sharp, as without a clock");
			assert.equal((b.d as any).anim, undefined, "stopped there");
		}
		// s again mid-way: as the lens's, from the view the pref now has, back; a key stops it; dispose leaves nothing.
		const b = at({ clock: true });
		b.d.handleInput("s");
		mock.timers.tick(140);
		b.d.handleInput("s");
		assert.equal(b.host.stream, "spikes");
		assert.match(b.title(), /tracking largest/, "starting over from the view the pref had");
		mock.timers.tick(70 * 8);
		assert.match(b.title(), /tracking spikes/);
		b.d.handleInput("s");
		b.d.handleInput("\x1b[D");
		assert.equal((b.d as any).anim, undefined, "a key: stopped");
		b.d.handleInput("s");
		b.d.dispose();
		assert.equal((b.d as any).anim, undefined, "dispose: nothing left running");
	} finally {
		mock.timers.reset();
	}
});

test("board: EXPERTS' grids in the right column end where STREAM's keys do (48 layers: 48 columns, full rectangles); at the column's width where STREAM is not drawn or its keys are not 4 columns short (64 layers spread, 12)", () => {
	const grids = (ls: string[]) => ls.map(right).filter((l) => /^▀+$/.test(l));
	for (const rows of [63, 50, 40]) {
		const ls = boardAt(rows, { moe: true }).view(), g = grids(ls);
		assert.equal(ls.map(right).find((l) => /^▆+$/.test(l)), "▆".repeat(48), `${rows}: STREAM's keys`);
		assert.ok(g.length >= 6 && g.every((l) => l === "▀".repeat(48)), `${rows}: the grids 48 wide\n${g.join("\n")}`);
		assert.ok(ls.map(right).some((l) => /^TOKENS {2}#/.test(l)) && ls.some((l) => l.endsWith(" │ " + "─".repeat(RIGHT_W))), `${rows}: TOKENS and the rules the column's width`);
	}
	// 512 experts at 50 columns: 11 rows of 47, the last 42, the cells past them blank.
	for (const [rows, L] of [[36, 48], [63, 0], [63, 64], [63, 12]]) {
		const ls = boardAt(rows, { moe: true, L }).view(), g = grids(ls);
		assert.equal(streamIn(ls).at >= 0, rows > 36 && L > 0, `${rows} rows, ${L} layers: STREAM drawn`);
		assert.ok(g.length >= 6 && g.every((l, i) => l === "▀".repeat((i + 1) % 6 ? 47 : 42)), `${rows} rows, ${L} layers: the grids at the column's width\n${g.join("\n")}`);
	}
});

/** The mini dock (MINI_WIDTH columns) on boardAt's reply, routed (512 experts) unless `o.moe` is false. */
const miniAt = (rows: number, o: Parameters<typeof boardAt>[1] = {}) => boardAt(rows, { moe: true, width: MINI_WIDTH, ...o, host: { mini: true, ...o.host } });
/** The mini dock's parts: its layers of lights, MAP's lines, STREAM's title and history lines, the key line. */
function miniParts(ls: string[]) {
	const st = ls.findIndex((l) => /^RESIDUAL STREAM/.test(l)), marks = ls.findIndex((l, i) => st >= 0 && i > st && /^L0 /.test(l));
	const map = ls.findIndex((l) => /^MAP /.test(l));
	return { grids: ls.filter((l) => /^[▸ ] layer \d+ of \d+$/.test(l)).length, map: map < 0 ? 0 : (st < 0 ? ls.length - 1 : st) - map, at: st, history: st < 0 ? 0 : marks - st - 2 };
}

test("mini dock: STREAM the last block, over the key line, as the board's right column draws it: its title, a quarter of the rows in lines of two tokens (8 at least, 16 at most) and the lines nothing else takes (16 in all at most), a key a column, the depth marks; at 63, 40, 30 and 24 rows every line its width, nothing under the key line", () => {
	assert.deepEqual([24, 30, 40, 50, 63, 90].map(miniStreamLines), [8, 8, 10, 12, 15, 16]);
	for (const rows of [63, 40, 30, 24]) {
		const { d } = miniAt(rows), raw = d.render(MINI_WIDTH), ls = text(raw), p = miniParts(ls);
		assert.equal(raw.length, rows);
		assert.ok(raw.every((l) => visibleWidth(l) === MINI_WIDTH), `${rows}: every line the dock's width`);
		const h = { 63: 16, 40: 14, 30: 11, 24: 8 }[rows]!;   // a quarter of the rows (15, 10, 8, 8), then the lines left: 1, 4, 3, 0
		assert.ok(h >= miniStreamLines(rows) && h <= 2 * STREAM_LINES);
		assert.equal(p.history, h, `${rows} rows`);
		assert.equal(ls[p.at], "RESIDUAL STREAM  4 streams · tracking spikes");
		assert.equal(ls[p.at + h + 1], "▆".repeat(48), `${rows}: a key a layer`);
		assert.equal(ls[p.at + h + 2], "L0      L8      L16     L24     L32     L40");
		assert.match(ls[p.at + h + 3], /^ctrl\+\\ steps in/, `${rows}: then the key line`);
		assert.equal(p.at + h + 3, rows - 1, `${rows}: the key line last, nothing empty under it`);
		assert.ok(p.grids >= 1 && p.map >= 2, `${rows}: a grid of lights and MAP's least over it`);
	}
});

test("mini dock: the lights give up grids, down to one, before STREAM's history goes under 8; over 8 only while two grids stay; MAP keeps its least; the history down to 3, then not drawn", () => {
	const seen: string[] = [];
	for (let rows = 12; rows <= 66; rows++) {
		const with_ = miniParts(miniAt(rows).view()), without = miniParts(miniAt(rows, { L: 0 }).view());
		assert.ok(with_.history === 0 || with_.history >= STREAM_MIN, `${rows}: ${with_.history} lines`);
		if (with_.history && with_.history < STREAM_LINES) assert.equal(with_.grids, 1, `${rows}: the history under ${STREAM_LINES} only with one grid left`);
		if (with_.history && with_.history < miniStreamLines(rows) && with_.grids > 2) assert.fail(`${rows}: ${with_.history} lines with ${with_.grids} grids`);
		if (with_.history) assert.ok(with_.map >= 2, `${rows}: MAP its least`);
		if (!with_.history) assert.deepEqual(with_, { ...without, at: -1, history: 0 }, `${rows}: no room, the mini dock as without`);
		seen.push(`${rows}:${with_.grids}/${without.grids}:${with_.history}`);
	}
	assert.ok(seen.includes("23:1/2:8") && seen.includes("20:1/1:5") && seen.includes("18:1/1:3") && seen.includes("17:1/1:0"), seen.join(" "));
	assert.ok(seen.includes("36:2/4:10") && seen.includes("40:2/4:14") && seen.includes("63:5/8:16"), seen.join(" "));
	// Over 8 lines only while two grids stay, where 8 leaves two (1,024 experts, 12 lines a layer): 41 rows has one grid
	// either way (its history 10, and the 6 lines left over), 42 keeps its second with 8, 43 has a line more.
	const big = [41, 42, 43].map((rows) => miniParts(miniAt(rows, { moe: 1024 }).view())).map((p) => `${p.grids}:${p.history}`);
	assert.deepEqual(big, ["1:16", "2:8", "2:9"]);
});

test("mini dock: one width: the grids (full rectangles), MAP and SETTLING end where STREAM's keys do where those are 4 columns short of the dock's at most (48 layers: 48), else at the dock's (64 layers spread, 12); every line the dock's width, `rows` lines, at 63, 50, 40, 30 and 24 rows", () => {
	for (const [L, end, keys] of [[48, 48, 48], [64, 50, 50], [12, 50, 24]])
		for (const lens of [false, true])
			for (const rows of [63, 50, 40, 30, 24]) {
				const what = `${L} layers${lens ? ", SETTLING" : ""}, ${rows} rows`, { d } = miniAt(rows, { L, lens }), raw = d.render(MINI_WIDTH), ls = text(raw);
				assert.equal(raw.length, rows, what);
				assert.ok(raw.every((l) => visibleWidth(l) === MINI_WIDTH), `${what}: every line the dock's width`);
				const map = ls.findIndex((l) => /^MAP /.test(l)), settle = ls.findIndex((l) => /^SETTLING /.test(l)), st = ls.findIndex((l) => /^RESIDUAL STREAM/.test(l));
				const grid = ls.slice(2, map).filter((l) => l.includes("▀")), cells = ls.slice(map + 1).filter((_, i, a) => a.slice(0, i + 1).every((l) => /^▀+$/.test(l)));
				assert.ok(grid.length >= 6 && grid.every((l) => l === "▀".repeat(end)), `${what}: the grids ${end} wide\n${ls.join("\n")}`);
				assert.ok(cells.length >= 1 && cells.every((l) => l === "▀".repeat(end)), `${what}: MAP ${end} wide`);
				if (settle > 0) assert.ok(ls.slice(settle + 1, st - 1).filter((l) => /▀/.test(l)).every((l) => l.length === end), `${what}: SETTLING ${end} wide`);
				assert.equal(lens, settle > 0, `${what}: SETTLING where a lens read`);
				assert.equal(ls.find((l) => /^▆+$/.test(l)), "▆".repeat(keys), `${what}: STREAM's keys`);
			}
	// STREAM not drawn (no room at 16 rows): the grids and MAP the dock's width again.
	const ls = miniAt(16).view(), map = ls.findIndex((l) => /^MAP /.test(l));
	assert.ok(!ls.some((l) => /RESIDUAL STREAM/.test(l)) && map > 0, ls.join("\n"));
	assert.ok(ls.filter((l) => l.includes("▀")).every((l) => l === "▀".repeat(50)), `the grids and MAP 50 wide\n${ls.join("\n")}`);
});

test("mini dock: a blank line between the parts, then STREAM's history, only from lines left over: never a layer of lights, a line of MAP, SETTLING or STREAM's history fewer; fewer than the gaps, from the top", () => {
	const block = 7;   // 512 experts, 48 wide: 11 rows of 47, a heading and 6 lines
	const gapsSeen = new Set<number>();
	for (const lens of [false, true])
		for (let rows = 12; rows <= 90; rows++) {
			const ls = miniAt(rows, { lens }).view(), what = `${lens ? "SETTLING, " : ""}${rows} rows`;
			const key = ls.findIndex((l) => /^ctrl\+\\ steps in/.test(l)), p = miniParts(ls);
			const titles = ls.flatMap((l, i) => (/^(MAP|SETTLING|RESIDUAL STREAM) /.test(l) ? [i] : []));
			// Blank lines over STREAM (its history has lines with nothing lit) and under the key line.
			const blanks = ls.slice(1, p.at < 0 ? key : p.at).flatMap((l, i) => (l === "" ? [i + 1] : [])), under = ls.length - 1 - key;
			assert.ok(blanks.every((i) => titles.includes(i + 1)), `${what}: blank lines only over a part`);
			assert.deepEqual(blanks.map((i) => titles.indexOf(i + 1)), titles.slice(0, blanks.length).map((_, j) => j), `${what}: the gaps from the top`);
			gapsSeen.add(blanks.length);
			if (!blanks.length) continue;
			const extra = p.history ? p.history - miniStreamLines(rows) : 0;
			assert.ok(extra >= 0 && blanks.length + extra + under < block, `${what}: a layer of lights would not fit in them`);
			if (p.history) assert.ok(!under || p.history === 2 * STREAM_LINES, `${what}: under the key line only past STREAM's most`);
			const map = ls.findIndex((l) => /^MAP /.test(l));
			assert.ok(ls.slice(map + 1, map + 4).every((l) => /^▀+$/.test(l)), `${what}: MAP its most, 3 lines`);
			const settle = ls.findIndex((l) => /^SETTLING /.test(l));
			if (settle > 0) assert.equal(ls.slice(settle + 1).findIndex((l) => !/^L\d+ /.test(l)), 6, `${what}: SETTLING its most (12 layers read, two a line)`);
		}
	assert.ok(gapsSeen.has(0) && gapsSeen.has(1) && gapsSeen.has(2) && gapsSeen.has(3), [...gapsSeen].join());
	// At 63 rows on Flash Next: 5 layers, MAP its title and 3 lines, 16 lines of history, a blank line between each two.
	const ls = miniAt(63).view(), p = miniParts(ls), map = ls.findIndex((l) => /^MAP /.test(l));
	assert.deepEqual([p.grids, p.history], [5, 16], "15 lines of history and the one left");
	assert.deepEqual([ls[map - 1], ls[map + 4], ls[map + 5]], ["", "", "RESIDUAL STREAM  4 streams · tracking spikes"]);
});

test("mini dock: STREAM not drawn on a model that records no writes; past 50 layers spread over the 50 columns, at or under exact", () => {
	assert.equal(miniParts(miniAt(40, { L: 0 }).view()).at, -1, "no writes");
	for (const [L, cols] of [[24, 48], [40, 40], [50, 50], [64, 50], [128, 50]]) assert.ok(miniAt(40, { L }).view().includes("▆".repeat(cols)), `${L} layers: ${cols} columns`);
});

test("mini dock: s switches STREAM's view (remembered, as the board's), named in its title stepped in where there is more than one; enter nothing", () => {
	const { d, host, view } = miniAt(40);
	assert.ok(view().includes("RESIDUAL STREAM  4 streams · tracking spikes"), "not stepped in: no s");
	d.stepIn(-1);
	assert.ok(view().includes("RESIDUAL STREAM  4 streams · tracking spikes · s"), "tracking kept beside s at 50 columns");
	for (const k of ["2", "5", "\r"]) d.handleInput(k);
	assert.deepEqual([d.full, host.panel, host.stream], [false, "LENS", undefined], "the numbers and enter do nothing on the mini dock");
	d.handleInput("s");
	assert.equal(host.stream, "largest");
	assert.ok(view().includes("RESIDUAL STREAM  4 streams · tracking largest · s"));
	d.handleInput("s");
	assert.equal(host.stream, "spikes");
	assert.ok(!view().at(-1)!.includes(" s "), "the key line without it");
	const single = miniAt(40, { k: 1 });
	single.view();
	single.d.stepIn(-1);
	assert.ok(single.view().includes("RESIDUAL STREAM  1 stream · tracking spikes"), "one view: no s");
	const inkling = miniAt(40, { k: 1, rk: 5 });
	inkling.view();
	inkling.d.stepIn(-1);
	assert.ok(inkling.view().includes("RESIDUAL STREAM  block sum · tracking spikes · s"), "one stream with reads: a block's sum");
	const none = miniAt(40, { L: 0 });
	none.view();
	none.d.stepIn(-1);
	none.d.handleInput("s");
	assert.equal(none.host.stream, undefined, "no STREAM drawn: s does nothing");
	// The board drawn with STREAM, the terminal then too short for it: s before the mini dock's first frame does nothing.
	const b = boardAt(63);
	b.view();
	b.d.stepIn(-1);
	(b.d as any).tui.terminal.rows = 20;
	b.d.handleInput("s");
	assert.equal(b.host.stream, undefined, "s before the mini dock is drawn: nothing");
});

test("mini dock: STREAM's playhead a little behind the newest while it streams, as on the board", () => {
	mock.timers.enable({ apis: ["setInterval"] });
	try {
		const clock = { t: 0 }, b = miniAt(40, { done: false, host: { now: () => clock.t } }), dk = b.d as any;
		b.view();
		onEvent(b.live, tok(30, 48, "r", 4, 0, true));
		onEvent(b.live, tok(31, 48, "r", 4, 0, true));
		b.d.invalidate();
		b.view();
		assert.ok(dk.playLoop, "the ticker on");
		assert.equal(playShown(dk.play), 29, "still token 29");
		for (let k = 0; k < 20 && dk.playLoop; k++) {
			clock.t += PLAY_TICK_MS;
			mock.timers.tick(PLAY_TICK_MS);
			b.d.invalidate();
			b.view();
		}
		assert.equal(dk.playLoop, undefined, "caught up");
		assert.equal(playShown(dk.play), 31);
	} finally {
		mock.timers.reset();
	}
});

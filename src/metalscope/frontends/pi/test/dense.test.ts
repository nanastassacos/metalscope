import assert from "node:assert/strict";
import { test } from "node:test";
import { MINI_MAP, MINI_WIDTH, miniSplit } from "../board.ts";
import { Dock } from "../dock.ts";
import { type Live, onEvent } from "../state.ts";

// A model without experts: EXPERTS not drawn (TOKENS has the column, tab and the keys skip it, the mini dock's MAP and
// SETTLING have the lights' room). One lens: drawn across LENS's page, no s other lens.

const plain = (s: string) => s.replace(/\x1b\[[0-9;:]*m/g, "");
const text = (lines: string[]) => lines.map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd());
const tick = () => new Promise((r) => setTimeout(r, 0));

/** Reply token `i` of a 24-layer model: the J-lens at every layer, the logit lens beside it (`two`), routing (`moe`). */
function tok(i: number, o: { moe?: boolean; two?: boolean } = {}): any {
	const lens = (w: string) => [...Array(24).keys()].map((l) => [l, l >= 20 ? 100 + i : 7, l >= 20 ? ` w${i}` : w]);
	const experts: number[] = [], w: number[] = [];
	for (let l = 0; l < 24; l++) for (let j = 0; j < 4; j++) { experts.push((i * 7 + l * 13 + j * 31) % 64); w.push(j ? 40 : 135); }
	return { type: "token", rid: "r", i, id: 100 + i, text: ` w${i}`, p: i % 5 ? 0.95 : 0.3, entropy: 0.5, n_layers: 24,
		alts: Array.from({ length: 9 }, (_, a) => [` a${a}`, 0.04 - a * 0.004]), lens: lens(" the"), lens_by: "jlens",
		...(o.two ? { lens_logit: lens(" a") } : {}), ...(o.moe ? { experts, expert_w: w, expert_k: 4 } : {}) };
}

/** A dock `rows` tall on a 30-token reply of model `model`; `dense`: known to have no experts (Live.dense). */
function dockAt(rows: number, o: { moe?: boolean; two?: boolean; dense?: boolean; mini?: boolean; panel?: string;
	model?: string; mixed?: boolean; asked?: string[] } = {}) {
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true, experts: true };
	const model = o.model ?? "M";
	onEvent(live, { type: "start", rid: "r", model });
	for (let i = 0; i < 30; i++) onEvent(live, tok(i, { moe: o.moe, two: o.two || (o.mixed && i % 2 === 0) }));
	onEvent(live, { type: "end", rid: "r", n_tokens: 30, trace_id: "T" });
	if (!o.dense) live.dense = undefined;
	const host: any = { live, captions: false, close() {}, unfocus() {}, mini: !!o.mini, panel: o.panel ?? "LENS",
		fetchJson: async (u: string) => (o.asked?.push(u), null), layoutChanged() {} };
	const d = new Dock({ terminal: { rows, columns: 200 }, requestRender() {} }, { getColorMode: () => "truecolor" }, host);
	return { d, host, live };
}
const view = (d: Dock, w = 100) => text(d.render(w));

test("no experts: EXPERTS not drawn on the board, the one-panel board or the mini dock", async () => {
	for (const [rows, mini] of [[52, false], [34, false], [52, true]] as const) {
		const dense = dockAt(rows, { dense: true, mini }), lines = view(dense.d, mini ? MINI_WIDTH : 100);
		assert.equal(dense.live.dense, "M", "a whole reply streamed with routing asked for and none came");
		assert.ok(!lines.some((l) => /EXPERTS|no experts|^L\d+$/.test(l)), `${rows} rows${mini ? ", mini" : ""}:\n${lines.join("\n")}`);
	}
	const { d, host } = dockAt(52, { dense: true, panel: "TOKENS" });
	d.render(100);
	d.handleInput("\t");
	assert.equal(host.panel, "MAP", "tab from TOKENS: MAP");
	const board = view(d);
	assert.ok(!board.some((l) => /EXPERTS|no experts/.test(l)) && board.some((l) => /│ TOKENS /.test(l)));
});

test("no experts: the mini dock gives the lights' room to MAP and SETTLING, no blank lines", () => {
	const lines = view(dockAt(52, { dense: true, mini: true }).d, MINI_WIDTH);
	const map = lines.findIndex((l) => /^MAP /.test(l)), settle = lines.findIndex((l) => /^SETTLING\b/.test(l));
	assert.ok(map === 1 && settle > map + MINI_MAP[2], `MAP right under the token, past its most:\n${lines.join("\n")}`);
	assert.ok(!lines.slice(settle, -1).some((l) => l === ""), "SETTLING to the key line");
	assert.ok(view(dockAt(52, { moe: true, mini: true }).d, MINI_WIDTH).some((l) => /^[▸ ] layer \d+ of \d+$/.test(l)), "experts: the lights");
	// Without lights MAP takes the lines, past its most, at any height.
	const tall = view(dockAt(120, { dense: true, mini: true }).d, MINI_WIDTH), m = tall.findIndex((l) => /^MAP /.test(l)), s = tall.findIndex((l) => /^SETTLING\b/.test(l));
	assert.ok(tall.length === 120 && m === 1 && s - m - 1 > 60 && tall.slice(m + 1, s).every((l) => /^▀+$/.test(l)), `120 rows: MAP ${s - m - 1} lines`);
	assert.deepEqual(miniSplit(40, 0, 0, MINI_MAP, [3, 6, 13]), { n: 0, map: 27, settle: 13 });
	assert.deepEqual(miniSplit(40, 0, 0, MINI_MAP), { n: 0, map: 40, settle: 0 });
});

test("no experts, model by model: 4 does nothing; another model without experts too; an MoE model's reply never loses it", async () => {
	const { d, live } = dockAt(52, { dense: true });
	d.render(100);
	d.stepIn(-1);
	d.handleInput("4");
	assert.ok(!view(d).some((l) => /EXPERTS/.test(l)));
	// Another model without experts.
	onEvent(live, { type: "start", rid: "r2", model: "N" });
	for (let i = 0; i < 3; i++) onEvent(live, { ...tok(i), rid: "r2" });
	onEvent(live, { type: "end", rid: "r2", n_tokens: 3, trace_id: "T2" });
	assert.equal(live.dense, "N");
	assert.ok(!view(d).some((l) => /EXPERTS/.test(l)));
	// An MoE model: shown from its first token, while it streams and after.
	onEvent(live, { type: "start", rid: "r3", model: "Moe" });
	onEvent(live, { ...tok(0, { moe: true }), rid: "r3" });
	assert.ok(view(d).some((l) => /│ EXPERTS /.test(l)), "first token");
	onEvent(live, { type: "end", rid: "r3", n_tokens: 1, trace_id: "T3" });
	assert.ok(view(d).some((l) => /│ EXPERTS /.test(l)));
	// A model without experts, its first reply: shown while it streams (not known yet), hidden from its end.
	onEvent(live, { type: "start", rid: "r4", model: "D2" });
	for (let i = 0; i < 3; i++) onEvent(live, { ...tok(i), rid: "r4" });
	assert.ok(view(d).some((l) => /│ EXPERTS/.test(l)), "not known yet");
	onEvent(live, { type: "end", rid: "r4", n_tokens: 3, trace_id: "T4" });
	assert.ok(!view(d).some((l) => /EXPERTS/.test(l)), "known at its end");
	await tick();
	d.dispose();
});

test("no experts only from 2 own tokens or more, none routed: an edit's one-token request, a one-token reply, a cancelled edit leave an MoE model alone", () => {
	const { d, live } = dockAt(52, { moe: true });
	const shown = () => view(d).some((l) => /│ EXPERTS /.test(l));
	// A word edit: its one-token request (that token never fed back: no routing), then the rest from there.
	onEvent(live, { type: "start", rid: "e1", model: "M", edit: { of: "r", at: 10 } });
	onEvent(live, { ...tok(0), rid: "e1" });
	onEvent(live, { type: "end", rid: "e1", n_tokens: 1, trace_id: "E1" });
	assert.equal(live.dense, undefined, "the edit's request");
	onEvent(live, { type: "start", rid: "e2", model: "M", edit: { of: "e1", at: 11 } });
	onEvent(live, { ...tok(0), rid: "e2" });
	assert.ok(shown(), "while the rest streams");
	onEvent(live, { ...tok(1, { moe: true }), rid: "e2" });
	onEvent(live, { type: "end", rid: "e2", n_tokens: 2, trace_id: "E2" });
	assert.ok(shown() && live.dense === undefined);
	// A reply of one token; an edit cancelled before its first token.
	onEvent(live, { type: "start", rid: "o", model: "M" });
	onEvent(live, { ...tok(0), rid: "o" });
	onEvent(live, { type: "end", rid: "o", n_tokens: 1, trace_id: "O" });
	assert.equal(live.dense, undefined, "one token");
	onEvent(live, { type: "start", rid: "c", model: "M", edit: { of: "e2", at: 5 } });
	onEvent(live, { type: "end", rid: "c", n_tokens: 0, trace_id: "C" });
	assert.ok(live.dense === undefined && shown(), "cancelled");
	// Two own tokens, none routed: no experts.
	onEvent(live, { type: "start", rid: "t", model: "M" });
	for (let i = 0; i < 2; i++) onEvent(live, { ...tok(i), rid: "t" });
	onEvent(live, { type: "end", rid: "t", n_tokens: 2, trace_id: "T2" });
	assert.equal(live.dense, "M");
	d.dispose();
});

test("mini dock, a model without experts: no routing asked", async () => {
	const asked: string[] = [];
	const { d } = dockAt(52, { mini: true, dense: true, asked });   // a reply without routing, saved
	d.render(MINI_WIDTH);
	await tick();
	assert.ok(!asked.some((u) => u.includes("/routing")));
	const moe = dockAt(52, { mini: true, asked });
	moe.d.render(MINI_WIDTH);
	assert.ok(asked.some((u) => u.includes("/routing")), "not known to have none: asked");
	d.dispose();
	moe.d.dispose();
});

test("one lens: LENS's page draws it across the width under its name; the sentence only where the reply has tokens with two; s other lens only with two", () => {
	const page = (o: any) => {
		const { d } = dockAt(52, { ...o, panel: "LENS" });
		d.stepIn(15);
		d.render(100);
		d.handleInput("3");
		return view(d);
	};
	const one = page({}), mixed = page({ mixed: true }), two = page({ two: true });
	assert.ok(!one.some((l) => /one lens recorded/.test(l)) && one.some((l) => /^J-lens$/.test(l)), "its name alone on its line");
	const box = (ls: string[]) => ls.find((l) => /╭─ out of L/.test(l))!;
	assert.ok(box(one).length > 90 && /╮ │ ╭─/.test(box(two)), "the width; two side by side");
	assert.ok(mixed.some((l) => /│ one lens recorded for this token/.test(l)), "mixed: said");
	assert.ok(one.some((l) => /^↑\/↓ the layer · ▸ the written word settles/.test(l)), "its legend");
	assert.ok(two.some((l) => /^both lenses · ↑\/↓ the layer in both/.test(l)));
	const keys = (o: any) => {
		const { d } = dockAt(34, { ...o, panel: "LENS" });
		d.stepIn(15);
		return view(d).slice(-2).join(" ");
	};
	assert.ok(!/s other lens/.test(keys({})) && /s other lens/.test(keys({ two: true })), keys({ two: true }));
});

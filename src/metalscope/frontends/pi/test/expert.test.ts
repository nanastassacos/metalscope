import assert from "node:assert/strict";
import { test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { expertKeys, expertsLayers, layerKeys, topsOf } from "../board.ts";
import { EXPERT_HELP, EXPLAIN, LABELS } from "../draw.ts";
import { EXPERT_LINES, contextPicks, layerPage, placesOf, whereStrip } from "../expert.ts";
import { palette } from "../lights.ts";
import { Dock } from "../dock.ts";
import { type Live, onEvent } from "../state.ts";

// The full-size EXPERTS' three screens: the grids with EXPERTS FOR under them, one layer, one expert. A 48-layer model
// of 512 experts mixing 10, the tally's answers from a fixture (fetchJson), the reply's routing for IN THIS REPLY.

const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const text = (lines: string[]) => lines.map((l) => plain(l).replace(/^[▌▐] /, "").trimEnd());
const tick = () => new Promise((r) => setTimeout(r, 0));
const KEY = { up: "\x1b[A", down: "\x1b[B", right: "\x1b[C", left: "\x1b[D", esc: "\x1b", enter: "\r" };
const piTheme = { fg: (_c: string, s: string) => s, bg: (_c: string, s: string) => s, bold: (s: string) => s };
const L = 48, E = 512, K = 10, LAYER = 38;

/** Reply token `i`: routed at every layer; at layer 39 (index 38) #385 the strongest on every fifth token and #12 on
 * every seventh, and at layer 40 #77 with #385. Thinking until token 30. */
function tok(i: number): any {
	const experts: number[] = [], w: number[] = [];
	for (let l = 0; l < L; l++) for (let j = 0; j < K; j++) {
		let e = (i * 7 + l * 13 + j * 31) % E;
		if (j === 0 && l === LAYER && i % 5 === 0) e = 385;
		else if (j === 0 && l === LAYER && i % 7 === 0) e = 12;
		else if (j === 0 && l === LAYER + 1 && i % 5 === 0) e = 77;
		else if (e === 385 || e === 12 || e === 77) e = (e + 1) % E;
		experts.push(e);
		w.push(j ? 20 + j : 120);
	}
	const text = i === 30 ? "</think>" : i % 5 === 0 ? " zebra" : ` w${i}`;
	return { type: "token", rid: "r", i, id: 100 + i, text, p: 0.95, entropy: 0.5, alts: [], experts, expert_w: w, expert_k: K };
}

const SPEC = (source: string, against: string | null = source === "session" ? "tally" : null) => ({ model: "M", layer: LAYER, source, tokens: source === "all" ? 215_000 : 8_200,
	against, experts: [3, 9, 12, 77, 385].map((e) => ({ e, top: source === "session" ? [` s${e}a`, ` s${e}b`] : [` t${e}a`, ` t${e}b`, "\n"] })) });
const INFO = (e: number, source: string) => ({ model: "M", layer: LAYER, source, tokens: source === "all" ? 215_000 : 8_200,
	against: source === "session" ? "tally" : null, ranked: 336, e, n: 1204, share: 1204 / 215_000, rank: e === 385 ? 7 : null,
	top: Array.from({ length: 12 }, (_, j) => [` tok${j}`, 30 - j * 2, 40 - j]),
	overlaps: [12, 3, 9, 41, 42, 43, 44, 45, 46].map((b, j) => ({ e: b, lift: 9 - j, co: 50, top: [` o${b}`] })) });
const NONE = { model: "M", layer: LAYER, source: "all", tokens: 0, against: null, experts: [] };

function dock(rows: number, opts: { tally?: boolean; dense?: boolean; held?: (source: string) => Promise<void> | undefined; fail?: "all" | "session"; streaming?: boolean } = {}) {
	const live: Live = { base: "http://h", clientId: "pi-me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r", model: "M" });
	for (let i = 0; i < 120; i++) onEvent(live, opts.dense ? { ...tok(i), experts: undefined, expert_w: undefined, expert_k: undefined } : tok(i));
	if (!opts.streaming) onEvent(live, { type: "end", rid: "r", n_tokens: 120, trace_id: "T" });
	if (opts.dense) live.dense = "M";
	const urls: string[] = [];
	const grid = { cols: 32, rows: 16, pos: Array.from({ length: E }, (_, e) => e), num_experts: E, n_layers: L };
	const host: any = { live, captions: false, close() {}, unfocus() {}, panel: "EXPERTS", expertLayer: LAYER,
		fetchJson: async (url: string) => {
			urls.push(url);
			const q = new URL(url).searchParams, source = q.get("client") ? "session" : "all";
			if (url.includes("/experts/grid")) return grid;
			if (source === opts.fail) throw new Error("no answer");
			if (opts.tally === false) return url.includes("/specialists") ? NONE : { ...NONE, e: Number(q.get("e")), n: 0, share: 0, rank: null, top: [], overlaps: [] };
			if (url.includes("/experts/specialists")) return (await opts.held?.(source), SPEC(source));
			if (url.includes("/experts/expert")) return INFO(Number(q.get("e")), source);
			return null;
		} };
	const d = new Dock({ terminal: { rows }, requestRender() {} }, piTheme, host);
	d.sel = 50;
	return { d, host, live, urls };
}
const settle = async (d: Dock, width: number) => { d.render(width); await tick(); d.render(width); await tick(); return text(d.render(width)); };

test("the reply's maths: places in the top 3, IN THIS REPLY's places apart, its strip", () => {
	const toks = Array.from({ length: 120 }, (_, i) => tok(i));
	const places = placesOf(toks, LAYER, 385);
	assert.deepEqual(places.map((x) => x.p), Array.from({ length: 24 }, (_, j) => j * 5));
	assert.deepEqual(contextPicks(toks, places), [0, 10, 30], "at most two of a text (all zebra but #30), at least 9 apart");
	const strip = whereStrip(120, places, 40);
	assert.equal(strip.length, 40);
	assert.ok(strip.every((d) => d >= 0 && d <= 8) && Math.max(...strip) === 8);
	assert.deepEqual(whereStrip(3, [{ p: 1 }], 6).filter((d) => d < 0).length, 3, "columns past a short reply blank");
});

test("the grids: ←/→ the token, ↑/↓ the focus layer, EXPERTS FOR under them with its tokens, no cursor and no ranked list; enter the layer page", async () => {
	const { d, host, urls } = dock(78);
	d.full = true;
	let lines = await settle(d, 104);
	assert.match(lines.find((l) => /^layers \d+-\d+ of 48 · /.test(l))!, /^layers \d+-\d+ of 48 · 512 experts · 10 mixed for each token +token #50 " zebra"$/);
	assert.ok(lines.some((l) => /▸ layer 39( |$)/.test(l)) && !lines.some((l) => /layer 39 +#\d/.test(l)), "the focus layer, no cursor's number");
	const band = lines.findIndex((l) => l.startsWith(LABELS.forToken('" zebra"')));
	assert.match(lines[band], new RegExp(`^${LABELS.forToken('" zebra"')} +${LABELS.source.all} +${LABELS.source.session}$`), "both sources: wide enough");
	assert.match(lines[band + 1], /^ {2}#385 +█+ +\d+% {2}" t385a" " t385b" "↵" +" s385a" " s385b"$/, "its share and its tokens (the tally's), by each source");
	assert.match(lines[band + 2], /^ {2}#\d+ +█+[▏▎▍▌▋▊▉]? +\d+% {2}- +-$/, "an expert the tally has nothing for: a dash, not the map's words");
	assert.equal(lines.slice(band + 1, band + 11).filter((l) => /^ {2}#\d+ /.test(l)).length, 10, "the token's ten experts");
	assert.ok(!lines.some((l) => l.includes(LABELS.layer(39))), "the layer's list is on the layer page");
	const asked = () => urls.filter((u) => u.includes("/specialists"));
	assert.deepEqual(asked(), ["http://h/scope/api/experts/specialists?layer=38&model=M", "http://h/scope/api/experts/specialists?layer=38&model=M&client=pi-me"],
		"one request for the focus layer and each source");
	assert.match(lines.at(-1)!, /←\/→ token · f live · ↑\/↓ layer · enter open/);
	d.handleInput(KEY.right);
	assert.equal(d.sel, 51, "→ the next token");
	assert.ok((await settle(d, 104)).some((l) => l.startsWith(LABELS.forToken('" w51"'))));
	d.handleInput(KEY.left);
	assert.equal(d.sel, 50);
	await settle(d, 104);
	assert.equal(asked().length, 2, "none for another token");
	d.handleInput(KEY.down);
	assert.equal(host.expertLayer, LAYER + 1, "↓ the next layer");
	await settle(d, 104);
	assert.ok(asked()[2].includes("layer=39") && asked().length === 4, "one for the new layer and each source");
	d.handleInput(KEY.up);
	d.handleInput(KEY.up);
	assert.equal(host.expertLayer, LAYER - 1, "↑ the one before");
	d.handleInput(KEY.down);
	for (const k of ["s", "S", "a"]) d.handleInput(k);
	assert.equal(host.expertSource, undefined, "a does nothing here: both shown");
	d.handleInput(KEY.enter);
	lines = await settle(d, 104);
	assert.match(lines.find((l) => l.startsWith("EXPERTS › "))!, /^EXPERTS › layer 39 · all chats +token #50 " zebra"$/);
	d.handleInput(KEY.esc);
	assert.ok(text(d.render(104)).some((l) => l.startsWith(`${LABELS.forToken('" zebra"')}`) && l.endsWith(LABELS.source.session)), "esc: the grids");
	assert.equal(d.full, true);
	d.handleInput(KEY.esc);
	assert.deepEqual([d.full, d.sel], [true, -1], "esc again, on the grids: the keyboard back, EXPERTS still full size");
});

test("both sources side by side where wide enough: the grids at the board's width, the layer page from 149 columns; no a there", async () => {
	const { d, host } = dock(78);
	d.full = true;
	let lines = await settle(d, 104);
	const band = lines.findIndex((l) => l.startsWith(LABELS.forToken('" zebra"')));
	const col = lines[band].indexOf(LABELS.source.session);
	assert.ok(col > lines[band].indexOf(LABELS.source.all), "all chats, then this session");
	assert.equal(lines[band + 1].indexOf('" s385a"'), col, "each source's tokens under its name");
	assert.ok(!/ a (all chats|this session)/.test(lines.at(-1)!), "no a in the key line");
	d.handleInput(KEY.enter);
	lines = await settle(d, 152);
	assert.match(lines.find((l) => l.startsWith("EXPERTS › "))!, /^EXPERTS › layer 39 +token #50 " zebra"$/, "no source in the breadcrumb: both shown");
	const head = lines.findIndex((l) => l.startsWith(LABELS.forToken('" zebra"')));
	assert.match(lines[head], new RegExp(`^${LABELS.forToken('" zebra"')} +${LABELS.source.all} +${LABELS.source.session} +│ ${LABELS.layer(39)}  386 of 512 +${LABELS.source.session}$`));
	const right = (l: string) => l.slice(l.indexOf("│") + 2);
	assert.match(right(lines[head + 8]), /^▸ #385 +" t385a" " t385b" "↵" +" s385a" " s385b"$/, "the layer's list: all chats, then this session");
	assert.equal(right(lines[head + 8]).indexOf('" s385a"'), right(lines[head]).indexOf(LABELS.source.session), "this session's under its name");
	assert.match(right(lines[head + 1]), /^ {2}#378 +- +-$/, "none in either: two dashes");
	assert.equal(lines.at(-1), layerKeys("", 385), "no a");
	d.handleInput("a");
	assert.equal(host.expertSource, undefined, "a does nothing with both shown");
	d.handleInput(KEY.enter);
	lines = await settle(d, 152);
	assert.equal(lines.at(-1), expertKeys(LABELS.source.session), "the expert page keeps a");
	d.handleInput(KEY.esc);
	lines = await settle(d, 148);
	assert.match(lines.find((l) => l.startsWith("EXPERTS › "))!, /^EXPERTS › layer 39 · all chats /, "148: one source, named in the breadcrumb");
	assert.match(lines[head], new RegExp(`^${LABELS.forToken('" zebra"')} +${LABELS.its} +│ ${LABELS.layer(39)}  386 of 512$`));
	assert.equal(lines.at(-1), layerKeys(LABELS.source.session, 385), "a back");
});

test("EXPERTS on the one-panel board: one source named in the heading, a switches it and the key line says so", async () => {
	const { d, host, urls } = dock(78);
	let lines = await settle(d, 60);
	const heading = () => lines.find((l) => l.includes(LABELS.forToken('" zebra"')))!;
	assert.ok(heading().endsWith(`${LABELS.its} · ${LABELS.source.all}`));
	assert.match(lines.slice(-2).join(" · "), / · enter open · a this session · /, "on the one-panel board's two key lines");
	assert.ok(!urls.some((u) => u.includes("client=")), "one source: one request");
	d.handleInput("a");
	assert.equal(host.expertSource, "session");
	lines = await settle(d, 60);
	assert.ok(heading().endsWith(`${LABELS.its} · ${LABELS.source.session}`));
	assert.ok(lines.some((l) => /#385 +█+ +\d+% {2}" s385a" " s385b"$/.test(l)), "the session's tokens");
	assert.ok(urls.some((u) => u.includes("/specialists") && u.includes("client=pi-me")));
	assert.match(lines.slice(-2).join(" · "), / · a all chats · /);
});

test("one source's answer missing or still asked for: the other's column and the layer's list stay", async () => {
	const ALL = '" t385a" " t385b" "↵"', SESSION = '" s385a" " s385b"';
	for (const [how, source] of [["failed", "session"], ["failed", "all"], ["pending", "session"], ["pending", "all"]] as const) {
		const never = new Promise<void>(() => {});
		const { d } = dock(78, how === "failed" ? { fail: source } : { held: (s) => (s === source ? never : undefined) });
		const gone = how === "failed" ? "-" : "";   // failed: a dash; asked for: blank
		const want = (lead: string) => new RegExp(`${lead}${source === "session" ? `${ALL} *${gone}` : `${gone} +${SESSION}`}$`);
		d.full = true;
		let lines = await settle(d, 104);
		const band = lines.findIndex((l) => l.startsWith(LABELS.forToken('" zebra"')));
		assert.match(lines[band + 1], want("% {2}"), `${source} ${how}: the grids`);
		d.handleInput(KEY.enter);
		lines = await settle(d, 152);
		const head = lines.findIndex((l) => l.startsWith(LABELS.forToken('" zebra"')));
		assert.match(lines[head], /  386 of 512 +this session$/, `${source} ${how}: the layer's list kept`);
		assert.match(lines[head + 8], want("│ ▸ #385 +"), `${source} ${how}: the layer's list`);
	}
});

test("one source or both: the same lines, the rows the same up to the tokens", () => {
	const th = { fg: (_c: string, s: string) => s, bold: (s: string) => s };
	const sp: any = SPEC("all"), mine: [number, number][] = [[385, 0.5], [12, 0.3], [600, 0.1]];
	const grid = { layer: LAYER, cols: 32, rows: 16, cells: new Array(E).fill(0) }, v: any = { layer: LAYER, L, E, k: K };
	const at = (lines: string[]) => lines.map((l) => plain(l).replace(/(EXPERTS FOR \S+|#\d+ +[█▏▎▍▌▋▊▉]* *[\d.]*%?|│ +(▸ )?#\d+|│ LAYER).*/g, "$1"));
	for (const w of [101, 149, 167]) {
		const one = expertsLayers(th, v, [grid], palette(th), w, mine, 7, " zebra", true, 40, [topsOf(sp)]);
		const two = expertsLayers(th, v, [grid], palette(th), w, mine, 7, " zebra", true, 40, [topsOf(sp), topsOf(sp)]);
		assert.equal(one.length, two.length);
		assert.deepEqual(at(one), at(two), `the grids ${w}`);
		const page = (sps: any[]) => layerPage(th, { layer: LAYER, sel: 7, text: " zebra", grid, held: true, mine, sps, source: "all", row: 0,
			n: E, first: 0, at: 0, right: false }, palette(th), w, 60);
		const p1 = page([sp]), p2 = page([sp, sp]);
		assert.equal(p1.length, p2.length);
		assert.deepEqual(at(p1).slice(1), at(p2).slice(1), `the layer page ${w}`);
		assert.ok([...p1, ...p2, ...one, ...two].every((l) => visibleWidth(l) === w));
	}
});

test("the grids' tally request: none while a reply streams, one once it is saved", async () => {
	const { d, live, urls } = dock(78);
	d.full = true;
	d.sel = -1;   // following the newest
	await settle(d, 104);
	const asked = () => urls.filter((u) => u.includes("/specialists")).length;
	assert.equal(asked(), 2, "one a source");
	onEvent(live, { type: "start", rid: "r2", model: "M" });
	for (let i = 0; i < 30; i++) {
		onEvent(live, { ...tok(i), rid: "r2" });
		await settle(d, 104);
	}
	assert.equal(asked(), 2, "the same tally while it streams");
	onEvent(live, { type: "end", rid: "r2", n_tokens: 30, trace_id: "T2" });
	await settle(d, 104);
	assert.equal(asked(), 4, "a newer reply saved: asked again");
});

test("following, EXPERTS keeps the glow as the reply ends; a token stepped to its own experts; f the glow again", async () => {
	const { d, live } = dock(78, { streaming: true }), dk: any = d;
	d.stepIn(-1);
	d.full = true;
	// The full-size grids, from the naming line to the legend, coloured; the board's stack (its cells, layer by layer).
	const grids = async () => {
		const ls = await settle(d, 104), a = ls.findIndex((l) => /^layers \d+-\d+ of 48 · /.test(l));
		return d.render(104).slice(a, ls.findIndex((l, i) => i > a && /bright: just used|lit: the experts mixed/.test(l)) + 1);
	};
	const stack = () => {
		dk.experts.stack = undefined;
		const r = live.reply!;
		return dk.experts.expertsData(r, d.sel >= 0 ? d.sel : r.tokens.length - 1, 3, 50).layers.map((l: any) => ({ layer: l.layer, cells: [...l.cells] }));
	};
	stack();
	await tick();   // the stack's grids
	const before = await grids(), board = stack();
	assert.match(plain(before.at(-1)!), /bright: just used · warm: busy lately/);
	onEvent(live, { type: "end", rid: "r", n_tokens: 120, trace_id: "T" });
	assert.deepEqual(await grids(), before, "the same lights, the same legend, at the end");
	assert.deepEqual(stack(), board, "the board's too");
	d.handleInput(KEY.left);
	assert.equal(d.sel, 118);
	const lines = await grids();
	assert.match(plain(lines.at(-1)!), /lit: the experts mixed for #118/);
	const focus = stack().find((l: any) => l.layer === LAYER)!;
	assert.equal(focus.cells.filter((x: number | null) => x! > 0).length, K, "the token's own experts");
	d.handleInput("f");
	assert.deepEqual(await grids(), before, "f: the glow again");
	assert.deepEqual(stack(), board);
});

test("the layer page: its grid large, EXPERTS FOR (a cursor list) beside the layer's experts; ←/→ the token, ↑/↓ pick, [ ] the layer, a the source", async () => {
	const { d, host, urls } = dock(78);
	d.full = true;
	await settle(d, 104);
	d.handleInput(KEY.enter);
	let lines = await settle(d, 104);
	const top = lines.findIndex((l) => l.startsWith("EXPERTS › layer 39"));
	assert.equal(lines[top + 1], "");
	assert.match(lines[top + 2], /^ {2}▀{64} {3}lit: the experts mixed$/, "the grid two cells a light, the legend beside it");
	assert.match(lines[top + 3], /^ {2}▀{64} {3}for #50$/);
	const head = lines.findIndex((l) => l.startsWith(LABELS.forToken('" zebra"')));
	assert.equal(head, top + 2 + 16 + 1, "the grid's 16 lines, a blank, the lists");
	assert.match(lines[head], new RegExp(`^${LABELS.forToken('" zebra"')} +${LABELS.its} +│ ${LABELS.layer(39)}  386 of 512$`), "by number: #385 the 386th");
	assert.match(lines[head + 1], /^▸ #385 +█+ +\d+% {2}" t385a" " t385b" "↵" +│ {3}#378 +-$/, "the window from #378, two rows under #385");
	assert.match(lines[head + 2], /^ {2}#\d+ .*- +│ {3}#379 /);
	assert.match(lines[head + 8], /│ ▸ #385 +" t385a" " t385b" "↵"$/, "the pick marked on the right too");
	assert.ok(urls.some((u) => u.includes("/experts/specialists?layer=38&model=M")));
	assert.equal(lines.at(-1), layerKeys(LABELS.source.session, 385));
	d.handleInput(KEY.down);
	lines = text(d.render(104));
	const second = /^▸ #(\d+) /.exec(lines[head + 2])![1];
	assert.equal(lines.at(-1), layerKeys(LABELS.source.session, Number(second)), "↓ the next; enter names it");
	d.handleInput(KEY.up);
	d.handleInput(KEY.up);
	assert.match(text(d.render(104))[head + 10], /^▸ #/, "↑ from the first: wraps to the last");
	d.handleInput(KEY.right);
	assert.equal(d.sel, 51, "→ the next token, still here");
	lines = text(d.render(104));
	assert.match(lines[top], /^EXPERTS › layer 39 · all chats +token #51 " w51"$/);
	assert.match(lines[head + 10], /^▸ #/, "the pick kept on its row");
	d.handleInput(KEY.left);
	d.handleInput("]");
	assert.equal(host.expertLayer, LAYER + 1);
	assert.ok(text(d.render(104)).some((l) => /^EXPERTS › layer 40 /.test(l)));
	d.handleInput("[");
	d.handleInput("a");
	lines = await settle(d, 104);
	assert.equal(host.expertSource, "session");
	assert.ok(urls.some((u) => u.includes("client=pi-me")), "the session: this Pi's client");
	assert.ok(lines.some((l) => new RegExp(`│ ${LABELS.layer(39)}  \\d+ of 512$`).test(l)), "no order named, the place");
	assert.ok(lines.some((l) => l.startsWith(`EXPERTS › layer 39 · ${LABELS.source.session} `)), "the source in the breadcrumb");
	assert.match(lines.at(-1)!, new RegExp(`  a ${LABELS.source.all}  esc back$`), "a names where it goes");
	assert.equal(d.full, true);
});

test("the layer page: an expert of the layer's that is also the token's in accent", () => {
	const th = { fg: (c: string, s: string) => `<${c}>${s}`, bold: (s: string) => s };
	const grid = { layer: LAYER, cols: 4, rows: 2, cells: new Array(8).fill(0) };
	const sp: any = SPEC("all");
	const out = layerPage(th, { layer: LAYER, sel: 5, text: " x", grid, held: true, mine: [[12, 0.5], [500, 0.3]], sps: [sp], source: "all", row: 0,
		n: E, first: 3, at: 12, right: false }, palette(th), 101, 40);
	const rows = out.filter((l) => l.includes("│"));
	assert.deepEqual(rows.slice(1).map((l) => Number(/│ .*#(\d+) /.exec(l)![1])), [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "by number from #3");
	assert.ok(rows.some((l) => /│ .*<accent>▸ <accent>#12 /.test(l)), "#12, the token's pick, marked in accent on the right");
	assert.ok(rows.some((l) => /│ .*<text>#9 /.test(l)), "#9, not the token's, plain");
	assert.ok(rows.some((l) => /│ .*<dim>#4 /.test(l)), "#4, no tokens in the tally, dim");
	assert.ok(out.every((l) => visibleWidth(l) === 101));
});

test("the layer page's whole list: tab into it, ↑/↓ scroll every expert, enter opens one; the token's pick followed, its experts in accent", async () => {
	const { d, host } = dock(78);
	d.full = true;
	await settle(d, 104);
	d.handleInput(KEY.enter);
	let lines = await settle(d, 104);
	const head = lines.findIndex((l) => l.startsWith("EXPERTS FOR "));
	const right = (l: string) => l.slice(l.indexOf("│") + 2);
	const rows = () => lines.slice(head + 1, head + 11).map(right);
	// Following the token's list: its first pick (#385) two rows over the window's foot; its second (lower) two rows
	// under the top; marked; the place in the heading.
	assert.equal(rows().findIndex((l) => l.startsWith("▸ #385 ")), 7, "#385 in view, two rows under it");
	d.handleInput(KEY.down);
	lines = text(d.render(104));
	const second = Number(/^▸ #(\d+) /.exec(lines[head + 2])![1]), i = second;
	assert.ok(i > 20 && i < 375, `#${second}: past the first window, before #385's`);
	assert.match(right(lines[head]), new RegExp(`  ${i + 1} of 512$`));
	assert.equal(rows().filter((l) => l.startsWith(`▸ #${second} `)).length, 1, "marked on the right");
	assert.equal(rows().findIndex((l) => l.startsWith("▸ ")), 2, "two rows over it");
	// tab: the keys in the layer's list, from that place; ↑/↓ move through every expert; enter opens it.
	d.handleInput("\t");
	lines = text(d.render(104));
	assert.equal(lines.at(-1), layerKeys(LABELS.source.session, second), "enter names the right list's expert");
	assert.equal(host.panel, "EXPERTS", "tab stayed on the page");
	for (let k = 0; k < 8; k++) d.handleInput(KEY.down);
	lines = text(d.render(104));
	assert.match(right(lines[head]), new RegExp(`  ${i + 9} of 512$`));
	assert.equal(rows().findIndex((l) => l.startsWith("▸ ")), 7, "the window scrolls with it");
	assert.match(lines[head + 2], /^▸ #/, "the token's list keeps its pick, marked");
	d.handleInput(KEY.enter);
	assert.ok(text(d.render(104)).includes(`EXPERTS › layer 39 › #${i + 8}`), "enter: an expert the token does not use");
	d.handleInput(KEY.esc);
	for (let k = 0; k < i + 8; k++) d.handleInput(KEY.up);
	lines = text(d.render(104));
	assert.match(right(lines[head]), /  1 of 512$/);
	assert.match(rows()[0], /^▸ #0 +-$/, "back at the top: #0");
	assert.ok(rows().slice(1).some((l) => /^ {2}#3 +" t3a"/.test(l)));
	d.handleInput(KEY.up);
	assert.match(right(text(d.render(104))[head]), /  512 of 512$/, "↑ from the first: the last");
	d.handleInput("\t");
	lines = text(d.render(104));
	assert.match(right(lines[head]), new RegExp(`  ${i + 1} of 512$`), "tab back: following the token's pick again");
	d.handleInput("\t");
	d.handleInput("a");
	lines = await settle(d, 104);
	assert.match(right(lines[head]), new RegExp(`^${LABELS.layer(39)}  ${i + 1} of 512$`), "a: the session's tokens, the same place");
	assert.equal(lines.at(-1), layerKeys(LABELS.source.all, i), "the same expert");
});

test("the layer page with no tally: tab does not go into the empty list, the token's list keeps the keys", async () => {
	const { d, host } = dock(78, { tally: false });
	d.full = true;
	await settle(d, 104);
	d.handleInput(KEY.enter);
	let lines = await settle(d, 104);
	const head = lines.findIndex((l) => l.startsWith("EXPERTS FOR "));
	d.handleInput("\t");
	d.handleInput(KEY.down);
	lines = text(d.render(104));
	assert.equal(host.panel, "EXPERTS", "tab stayed on the page");
	assert.match(lines[head + 2], /^▸ #/, "↓ moved the token's list");
	assert.equal(lines.at(-1), layerKeys(LABELS.source.session, Number(/^▸ #(\d+) /.exec(lines[head + 2])![1])), "enter names its pick");
	assert.ok(!lines.slice(head, head + 11).some((l) => /│ ▸/.test(l)), "no cursor in the empty list");
});

test("a reply saved while on the layer page: the list stays drawn while asked again, the cursor on its expert", async () => {
	let hold: Promise<void> | undefined, release = () => {};
	const { d, live } = dock(78, { held: () => hold });
	d.full = true;
	await settle(d, 104);
	d.handleInput(KEY.enter);
	let lines = await settle(d, 104);
	const head = lines.findIndex((l) => l.startsWith("EXPERTS FOR "));
	d.handleInput("\t");
	for (let k = 0; k < 100; k++) d.handleInput(KEY.down);   // from #385, the token's pick: #485
	lines = text(d.render(104));
	assert.match(lines[head], /  486 of 512$/);
	const tokens = lines[head + 1].slice(0, lines[head + 1].indexOf("│"));
	hold = new Promise((r) => (release = r));
	onEvent(live, { type: "start", rid: "r2", model: "M" });
	for (let i = 0; i < 120; i++) onEvent(live, { ...tok(i), rid: "r2" });
	onEvent(live, { type: "end", rid: "r2", n_tokens: 120, trace_id: "T2" });
	d.sel = 50;
	lines = await settle(d, 104);
	assert.match(lines[head], /  486 of 512$/, "asked again: the last answer drawn");
	assert.equal(lines[head + 1].slice(0, lines[head + 1].indexOf("│")), tokens, "its tokens too");
	d.handleInput(KEY.down);
	release();
	lines = await settle(d, 104);
	assert.match(lines[head], /  487 of 512$/, "a key meanwhile moved on from there");
});

test("back from the expert page the layer's list is on the expert last shown (MIXED WITH, ] and b)", async () => {
	const { d } = dock(78);
	d.full = true;
	await settle(d, 104);
	d.handleInput(KEY.enter);
	await settle(d, 104);
	d.handleInput("\t");                            // on #385, the token's pick
	d.handleInput(KEY.enter);
	await settle(d, 104);
	d.handleInput(KEY.down);
	d.handleInput(KEY.enter);                       // MIXED WITH's second: #3
	await settle(d, 104);
	d.handleInput(KEY.esc);
	let lines = await settle(d, 104);
	const head = lines.findIndex((l) => l.startsWith("EXPERTS FOR "));
	assert.match(lines[head], /  4 of 512$/, "#3");
	assert.equal(lines.at(-1), layerKeys(LABELS.source.session, 3));
	d.handleInput(KEY.enter);
	await settle(d, 104);
	d.handleInput("]");
	await settle(d, 104);
	d.handleInput(KEY.down);
	d.handleInput(KEY.down);
	d.handleInput(KEY.enter);                       // at layer 40, MIXED WITH's third: #9
	await settle(d, 104);
	d.handleInput("b");                             // back to #3
	d.handleInput(KEY.esc);
	lines = await settle(d, 104);
	assert.ok(lines.some((l) => l.startsWith("EXPERTS › layer 40 ")));
	assert.equal(lines.at(-1), layerKeys(LABELS.source.session, 3), "#3 at layer 40");
});

test("its tokens: the heading over the words column, on the grids and the layer page, at every width", () => {
	const th = { fg: (_c: string, s: string) => s, bold: (s: string) => s };
	const sp: any = SPEC("all"), tops = topsOf(sp), mine: [number, number][] = [[385, 0.5], [12, 0.3], [600, 0.1]];
	const layers = [{ layer: LAYER, cols: 32, rows: 16, cells: new Array(E).fill(0) }];
	const v: any = { layer: LAYER, L, E, k: K };
	for (const word of [" zebra", " an exceptionally long token indeed", "\n"]) {
		const check = (lines: string[], cw: number, what: string) => {
			const t = lines.map(plain), h = t.findIndex((l) => l.startsWith("EXPERTS FOR "));
			const col = t[h].indexOf(LABELS.its);
			assert.ok(col > 0, `${what}: a heading`);
			for (const r of t.slice(h + 1, h + 4)) {
				const left = r.slice(0, cw);
				assert.equal(left.search(/" t|-/), col, `${what}: ${JSON.stringify(t[h])} over ${JSON.stringify(left)}`);
			}
		};
		for (const w of [50, ...Array.from({ length: 104 }, (_, j) => 97 + j)]) {
			check(expertsLayers(th, v, layers, palette(th), w, mine, 7, word, true, 40, [tops]), w, `the grids ${w}`);
			if (w < 97) continue;
			const page = layerPage(th, { layer: LAYER, sel: 7, text: word, grid: layers[0], held: true, mine, sps: [sp], source: "all", row: 0,
				n: E, first: 0, at: 0, right: false }, palette(th), w, 60);
			check(page, Math.floor((w - 3) / 2), `the layer page ${w}`);
		}
	}
});

test("the expert page: a sentence, the legend, ITS TOKENS beside MIXED WITH, IN THIS REPLY with its strip under it", async () => {
	const { d } = dock(78);
	d.full = true;
	await settle(d, 104);
	d.handleInput(KEY.enter);
	await settle(d, 104);
	d.handleInput(KEY.enter);
	const lines = await settle(d, 104);
	const top = lines.indexOf("EXPERTS › layer 39 › #385");
	assert.ok(top > 0);
	assert.equal(lines[top + 1], `takes 1,204 of 215k tokens (0.6%) · 7th ${LABELS.rank.all} of 336 · ${LABELS.source.all}`);
	assert.equal(lines[top + 2], LABELS.times("30"));
	assert.equal(lines[top + 3], "");
	assert.match(lines[top + 4], new RegExp(`^${LABELS.tokens} +│ ${LABELS.mixed} {2}${LABELS.mixedTag}$`));
	assert.match(lines[top + 5], /^" tok0" +█+ +×30 │ ▸ #12 +█+ +×9\.0 {2}" o12"$/);
	assert.equal(lines.slice(top + 5, top + 15).filter((l) => /^" tok/.test(l)).length, 10, "ten tokens");
	assert.equal(lines.slice(top + 5, top + 15).filter((l) => /│ (▸| ) #\d+/.test(l)).length, 8, "eight mixed with");
	assert.equal(lines[top + 16], LABELS.reply);
	assert.match(lines[top + 17], /^#\d+ +.*zebra/);
	assert.equal([...lines[top + 23]].length, 101, "the strip the dock's width, under the places");
	assert.match(lines[top + 24], /^thinking +answer$/);
	assert.ok(!lines.some((l) => /^group|NEXT LAYER|WHERE/.test(l)), "no group, no next layer, no WHERE heading");
	assert.equal(lines.at(-1), expertKeys(LABELS.source.session));
});

test("the expert page's keys: ↑↓ pick, enter goes, b back, [ ] the same number a layer over, a the source, ←/→ the token, esc the layer page", async () => {
	const { d, host, urls } = dock(78);
	d.full = true;
	await settle(d, 104);
	d.handleInput(KEY.enter);
	d.handleInput(KEY.enter);
	await settle(d, 104);
	d.handleInput(KEY.down);
	d.handleInput(KEY.enter);
	let lines = await settle(d, 104);
	assert.ok(lines.some((l) => /^EXPERTS › layer 39 › #3 +b #385$/.test(l)), "enter: the second mixed with; b says where back goes");
	d.handleInput("b");
	assert.ok(text(d.render(104)).some((l) => /^EXPERTS › layer 39 › #385$/.test(l)));
	d.handleInput("]");
	assert.equal(host.expertLayer, LAYER + 1);
	assert.ok(text(d.render(104)).some((l) => /^EXPERTS › layer 40 › #385 +b #385$/.test(l)), "] the same number a layer deeper");
	d.handleInput("b");
	assert.equal(host.expertLayer, LAYER);
	d.handleInput(KEY.right);
	assert.equal(d.sel, 51, "→ the token");
	assert.ok(text(d.render(104)).some((l) => /^EXPERTS › layer 39 › #385$/.test(l)), "the page stays");
	d.handleInput("a");
	lines = await settle(d, 104);
	assert.equal(host.expertSource, "session");
	assert.ok(urls.some((u) => u.includes("/experts/expert") && u.includes("client=pi-me")));
	assert.ok(lines.some((l) => l === `takes 1,204 of 8.2k tokens (0.6%) · 7th ${LABELS.rank.session} of 336 · ${LABELS.source.session}`));
	d.handleInput(KEY.esc);
	assert.ok(text(d.render(104)).some((l) => /^EXPERTS › layer 39 · .* +token #51 /.test(l)), "esc: the layer page");
	d.handleInput(KEY.esc);
	assert.ok(text(d.render(104)).some((l) => l.startsWith(LABELS.forToken('" w51"')) && l.endsWith(LABELS.source.session)), "esc again: the grids");
});

test("the ? text: the grids their explanation; the layer page the rule and both ranks; the expert page the rule and its source's", async () => {
	const { d, host } = dock(78);
	host.captions = true;
	d.full = true;
	await settle(d, 104);
	const footer = () => text(d.render(104)).slice(-4).join(" ");
	assert.ok(footer().includes(EXPLAIN.EXPERTS_WINDOW.slice(0, 40)) && !footer().includes(EXPERT_HELP.rank.all), "the grids: no ranked list, no rank help");
	d.handleInput(KEY.enter);
	await settle(d, 104);
	let lines = text(d.render(104));
	assert.ok(lines.includes(EXPERT_HELP.rule) && !lines.some((l) => l.includes(EXPERT_HELP.rank.all)), "the layer page: the rule, no ranks");
	d.handleInput(KEY.enter);
	await settle(d, 104);
	lines = text(d.render(104));
	assert.ok(lines.includes(EXPERT_HELP.rule) && lines.includes(EXPERT_HELP.rank.all) && !lines.some((l) => l.includes(EXPERT_HELP.rank.session)), "the expert page");
	host.expertSource = "session";
	assert.ok(text(d.render(104)).includes(EXPERT_HELP.rank.session), "its source's");
	for (const l of [EXPERT_HELP.rule, EXPERT_HELP.rank.all, EXPERT_HELP.rank.session]) assert.ok(!l.includes("—") && l.length <= 97);
});

test("the layer page's grid the same size following or paused: room kept for the wider legend", () => {
	const th = { fg: (_c: string, s: string) => s, bold: (s: string) => s };
	const grid = { layer: LAYER, cols: 32, rows: 16, cells: new Array(512).fill(0) };
	for (let w = 97; w <= 160; w++) {
		const at = (held: boolean) => layerPage(th, { layer: LAYER, sel: 5, text: " x", grid, held, mine: [[12, 0.5]], sps: [undefined],
			source: "all", row: 0, n: 0, first: 0, at: -1, right: false }, palette(th), w, 50).findIndex((l) => plain(l).startsWith("EXPERTS FOR"));
		assert.equal(at(true), at(false), `w=${w}`);
	}
});

test("the layer page's cursor past a shorter list: the cursor, the key line and enter agree", async () => {
	const { d, live } = dock(78);
	const t: any = live.reply!.tokens[51];
	for (let j = 3; j < K; j++) t.expert_w[LAYER * K + j] = 0;   // token 51: three experts at the layer
	d.full = true;
	await settle(d, 104);
	d.handleInput(KEY.enter);
	await settle(d, 104);
	for (let i = 0; i < 9; i++) d.handleInput(KEY.down);          // the tenth row on token 50
	d.handleInput(KEY.right);
	const lines = text(d.render(104)), head = lines.findIndex((l) => l.startsWith("EXPERTS FOR "));
	const e = Number(/^▸ #(\d+) /.exec(lines[head + 3])![1]);
	assert.equal(lines.slice(head + 1).filter((l) => /^▸ #/.test(l)).length, 1, "on the last of three");
	assert.equal(lines.at(-1), layerKeys(LABELS.source.session, e), "the key line names it");
	d.handleInput(KEY.enter);
	assert.ok(text(d.render(104)).some((l) => l === `EXPERTS › layer 39 › #${e}`), "enter opens it");
	d.handleInput(KEY.esc);
	d.handleInput(KEY.up);
	assert.match(text(d.render(104))[head + 2], /^▸ #/, "↑ from the last of three: the second");
});

test("s and S on the layer and expert pages: nothing (the lens under them stays)", async () => {
	const { d, host } = dock(78);
	d.full = true;
	await settle(d, 104);
	for (let page = 0; page < 2; page++) {
		d.handleInput(KEY.enter);
		await settle(d, 104);
		const was = host.lensView;
		d.handleInput("s");
		d.handleInput("S");
		assert.equal(host.lensView, was, `page ${page + 2}`);
	}
});

test("the pages' geometry: the same lines at every size, for every token, expert, layer and source; with no tally a dash", async () => {
	const where = (lines: string[]) => {
		const top = lines.findIndex((l) => /^EXPERTS › layer \d+ › #\d+/.test(l));
		const at = (re: RegExp) => lines.findIndex((l) => re.test(l)) - top;
		return [top, at(new RegExp(`^${LABELS.tokens} `)), at(new RegExp(`│ ${LABELS.mixed}`)), at(new RegExp(`^${LABELS.reply}$`)), at(/^thinking/)];
	};
	const layerAt = (lines: string[]) => [lines.findIndex((l) => /^EXPERTS › layer \d+ /.test(l)), lines.findIndex((l) => l.startsWith("EXPERTS FOR "))];
	let first: number[] | undefined;
	for (const [width, rows] of [[100, 51], [104, 78], [130, 125]]) {
		let here: number[] | undefined, lp: number[] | undefined;
		for (const tally of [true, false]) for (const source of ["all", "session"]) for (const move of [0, 1]) {
			const { d, host } = dock(rows, { tally });
			host.expertSource = source;
			d.full = true;
			await settle(d, width);
			d.handleInput(KEY.enter);
			for (let i = 0; i < move; i++) d.handleInput(KEY.right), d.handleInput("]");
			let raw = d.render(width);
			await tick();
			raw = d.render(width);
			assert.equal(raw.length, rows, `${width}x${rows}`);
			for (const l of raw) assert.equal(visibleWidth(l), width, `${width}x${rows}: ${plain(l)}`);
			const l2 = layerAt(text(raw));
			lp ??= l2;
			assert.deepEqual(l2, lp, `the layer page ${width}x${rows} ${tally} ${source} ${move}`);
			d.handleInput(KEY.enter);
			for (let i = 0; i < move; i++) d.handleInput(KEY.right), d.handleInput(KEY.down);
			d.render(width);
			await tick();
			raw = d.render(width);
			const lines = text(raw);
			assert.equal(raw.length, rows, `${width}x${rows}`);
			for (const l of raw) assert.equal(visibleWidth(l), width, `${width}x${rows}: ${plain(l)}`);
			const w = where(lines);
			here ??= w;
			assert.deepEqual(w, here, `${width}x${rows} ${tally} ${source} ${move}: ${w} vs ${here}`);
			if (!tally) {
				assert.equal(lines[w[0] + 1], `- · ${LABELS.source[source as "all" | "session"]}`);
				assert.equal(lines[w[0] + 5].slice(0, 1), "-");
			}
		}
		const rel = here!.slice(1);
		first ??= rel;
		assert.deepEqual(rel, first, "the page's own lines the same at every size");
		assert.ok(here![0] + EXPERT_LINES <= rows);
	}
});

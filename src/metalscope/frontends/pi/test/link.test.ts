// The link offer: a served name (a build) with nothing of its own, linked from the Steer page (l) to a same-shape model
// with data; LENS's pointer to it. The server's shapes as web.py answers GET and POST /scope/api/model/link.
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { STEER_ROWS, steerKeys, steerPanel } from "../board.ts";
import { registerCommands } from "../commands.ts";
import { Dock } from "../dock.ts";
import { type Live, onEvent } from "../state.ts";
import { LENS_LINK, LINK_KEEP, dataTag, keepOffer, linkLabel, linkTitle, linkedNote, parseOffer, parseSteering } from "../steerpage.ts";
import { createUi } from "../widget.ts";

const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const theme = { fg: (_c: string, s: string) => s, bg: (_c: string, s: string) => s, bold: (s: string) => s };
const tick = () => new Promise((r) => setTimeout(r, 0));
const BUILD = "Qwen3.8-Flash-Next-oQ4e-mtp", MODEL = "Qwen3.8-Flash-Next";
const OFFER = { model: BUILD, checkpoint: BUILD, linked: false, own: false, shape: { n_layers: 48, width: 10240 },
	candidates: [{ name: MODEL, concepts: 14, lenses: 1 }] };
const THREE = { ...OFFER, candidates: [{ name: MODEL, concepts: 14, lenses: 1 }, { name: "Qwen3.8-Flash-Next-ds4-q4k", concepts: 3, lenses: 0 },
	{ name: "Qwen3.8-Flash-Next-bf16", concepts: 0, lenses: 2 }] };
const LINKED = { model: BUILD, checkpoint: MODEL, concepts: 14, lenses: 1, lens: "neuronpedia" };
/** The concepts a linked build lists: the model's, not calibrated on this build. */
const SHARED = { concepts: ["calm", "robotic"].map((name) => ({ name, title: name, description: `Sounding ${name}.`, chat: null, chat_stale: false })) };

test("parseOffer: the candidates the server sends, most data first; none when linked, its own, or unreadable", () => {
	assert.deepEqual(parseOffer(OFFER, BUILD), { model: BUILD, shape: { n_layers: 48, width: 10240 }, candidates: [{ name: MODEL, concepts: 14, lenses: 1 }] });
	for (const bad of [{ ...OFFER, linked: true }, { ...OFFER, own: true }, { ...OFFER, candidates: [] }, null, undefined, 1, "x", [], {}, { candidates: "x" },
		{ candidates: [null, 1, { name: 5, concepts: 3 }, { name: "", concepts: 2 }, { name: "a" }, { name: "b", concepts: -1, lenses: 0.5 }, { name: "c", concepts: "14" }] }])
		assert.equal(parseOffer(bad, BUILD), null, JSON.stringify(bad));
	// A candidate with something to share is kept; the rest dropped; a shape it cannot read: none said.
	assert.deepEqual(parseOffer({ model: BUILD, shape: { n_layers: "48" }, candidates: [{ name: "a", concepts: 2, lenses: "1" }, { name: 3 }] }, BUILD),
		{ model: BUILD, shape: null, candidates: [{ name: "a", concepts: 2, lenses: 0 }] });
	// Names only as the server names models (anything else would reach the terminal, or be refused): the rest dropped.
	const names = ["a\x1b]8;;http://x\x07b", "two\nlines", ".hidden", "a b", "tab\tx", "ok_1.2-x"].map((name) => ({ name, concepts: 1 }));
	assert.deepEqual(parseOffer({ model: BUILD, candidates: names }, BUILD)!.candidates.map((c) => c.name), ["ok_1.2-x"]);
	// For the model asked about, and a model at all (nothing is offered, so nothing posted, for "").
	assert.equal(parseOffer({ ...OFFER, model: "Other" }, BUILD), null);
	assert.equal(parseOffer({ ...OFFER, model: undefined }, BUILD), null);
	assert.equal(parseOffer({ ...OFFER, model: "" }, ""), null);
	// A shape out of reason: the offer without it.
	for (const shape of [{ n_layers: 0, width: 10 }, { n_layers: -5, width: 10 }, { n_layers: 1e21, width: 10 }, { n_layers: 10001, width: 10 }, { n_layers: 48, width: 1e8 },
		{ n_layers: 48, width: 0 }, { n_layers: 4.5, width: 10 }])
		assert.equal(parseOffer({ ...OFFER, shape }, BUILD)!.shape, null, JSON.stringify(shape));
	assert.deepEqual(parseOffer({ ...OFFER, shape: { n_layers: 1, width: 1e7 } }, BUILD)!.shape, { n_layers: 1, width: 1e7 });
	// Kept by the host for the session only once it says something lasting: linked, its own, or candidates of a known shape.
	assert.deepEqual([OFFER, { ...OFFER, linked: true, candidates: [] }, { ...OFFER, own: true, candidates: [] }, { ...OFFER, shape: null }, { ...OFFER, candidates: [] },
		{ ...OFFER, linked: true, shape: null }, null, "x"].map(keepOffer), [true, true, true, false, false, false, false, false]);
	assert.deepEqual(THREE.candidates.map(linkLabel), [`${MODEL} · 14 concepts · 1 lens`, "Qwen3.8-Flash-Next-ds4-q4k · 3 concepts", "Qwen3.8-Flash-Next-bf16 · 2 lenses"]);
	assert.equal(dataTag(1, 0), "1 concept");
	assert.equal(linkTitle(BUILD), `Link ${BUILD} to`);
	assert.equal(LINK_KEEP, "Keep it separate");
	assert.equal(linkedNote(LINKED, OFFER.candidates[0]), `linked to ${MODEL} · 14 concepts · 1 lens`);
	assert.equal(linkedNote({}, OFFER.candidates[0]), `linked to ${MODEL} · 14 concepts · 1 lens`);           // the candidate's, should it not say
	assert.equal(linkedNote({ checkpoint: MODEL, concepts: 0, lenses: 0 }, OFFER.candidates[0]), `linked to ${MODEL}`);
});

/** The Steer page (5) of a dock `rows` tall on `BUILD`, stepped in, its list `list` (none learned by default), the
 * offer as the host's linkOffer answers it (`offer`; an Error: it fails), l's pick through `link`. */
function offerDock(o: { list?: any; offer?: any; rows?: number; link?: (offer: any) => Promise<any> } = {}) {
	const live: Live = { base: "http://x", clientId: "me", level: "on", connected: true, dense: "",
		reply: { rid: "r", tokens: [0, 1, 2].map((i) => ({ i, id: i, text: " w", p: 0.9, entropy: 0.1, alts: [], n_layers: 48 })), done: true, prefill: 1 } } as any;
	const asked: string[] = [], linked: any[] = [];
	let model = BUILD;
	const host: any = { live, captions: false, close() {}, unfocus() {}, steerModel: () => model,
		steering: async () => (asked.push(`steering ${model}`), parseSteering(o.list ?? { concepts: [] }, model)),
		steerChats: async () => ({ error: "none" }),
		steer: async () => [],
		linkOffer: async () => {
			asked.push(`offer ${model}`);
			if (o.offer instanceof Error) throw o.offer;
			return "offer" in o ? o.offer : OFFER;
		},
		link: async (offer: any) => (linked.push(offer), o.link ? o.link(offer) : undefined),
		lensLayers: async () => (asked.push("lens layers"), { error: "none" }), fetchJson: async (url: string) => (asked.push(url), null) };
	const d = new Dock({ terminal: { rows: o.rows ?? 52, columns: 200 }, requestRender() {} }, theme, host);
	d.steerPage = true;
	d.stepIn(-1);
	return { d, host, asked, linked, serve: (m: string) => void (model = m) };
}
const text = (d: Dock, width = 100) => d.render(width).map((l) => plain(l).replace(/^[▌▐] /, "").replace(/\s+$/, ""));
/** The page under the header and its rule, to the first blank run past the offer. */
const page = (lines: string[]) => lines.slice(2, lines.findIndex((l, i) => i > 9 && !l && !lines[i + 1]));

test("the Steer page offers the link only when nothing is learned here and a same-shape model has data", async () => {
	const one = offerDock();
	await tick();
	let lines = text(one.d);
	assert.deepEqual(page(lines), [` STEER · ${BUILD}`, "", "Nothing learned on this model yet.", "", "Same shape as a model you have (48 layers, width 10240):",
		`  ${MODEL} · 14 concepts · 1 lens`, "", "l link to it · /concept learn NAME starts fresh"]);
	assert.equal(lines.at(-1), "l link · x off · ? explain · 6 board · esc back");
	const three = offerDock({ offer: THREE });
	await tick();
	lines = text(three.d);
	assert.deepEqual(page(lines).slice(4), ["Same shape as models you have (48 layers, width 10240):", `  ${MODEL} · 14 concepts · 1 lens`,
		"  Qwen3.8-Flash-Next-ds4-q4k · 3 concepts", "  Qwen3.8-Flash-Next-bf16 · 2 lenses", "", "l link to one · /concept learn NAME starts fresh"]);
	// Past five: the rest counted (the pick list has them all).
	const many = offerDock({ offer: { ...OFFER, candidates: Array.from({ length: 7 }, (_, i) => ({ name: `m${i}`, concepts: 7 - i, lenses: 0 })) } });
	await tick();
	lines = text(many.d);
	assert.ok(lines.includes("  m4 · 3 concepts") && !lines.includes("  m5 · 2 concepts") && lines.includes("  and 2 more"), lines.join("\n"));
	// ? on the page: what l does.
	one.d.handleInput("?");
	assert.match(text(one.d).slice(-3).join(" "), /l: use a same-shape model's concepts and J-lens; doses stay this build's own\./);
	// Keyboard in the chat: what stepping in gives.
	one.d.handleInput("?");
	one.d.handleInput("\x1b");
	assert.equal(text(one.d).at(-1), "ctrl+\\ steps in: l link, 6 board · /scope help dock");
});

test("no offer: linked, its own data, no candidates, a failed or unreadable answer, or concepts learned here; l does nothing then", async () => {
	const cases: [string, any][] = [["linked", { ...OFFER, linked: true }], ["own", { ...OFFER, own: true }], ["none", { ...OFFER, candidates: [] }],
		["failed", new Error("boom")], ["404", null], ["garbage", "<html>"]];
	for (const [why, offer] of cases) {
		const m = offerDock({ offer });
		await tick();
		const lines = text(m.d);
		assert.ok(lines.includes(`no learned concepts on ${BUILD}: /concept learn NAME teaches one`), `${why}\n${lines.join("\n")}`);
		assert.ok(!lines.some((l) => /l link|Same shape/.test(l)), why);
		m.d.handleInput("l");
		await tick();
		assert.deepEqual([m.linked, m.d["sp"].steer.note], [[], undefined], why);
	}
	// Concepts learned here: the list, never the offer, whatever the server said.
	const learned = offerDock({ list: SHARED });
	await tick();
	const lines = text(learned.d);
	assert.ok(lines.some((l) => l.includes("calm")) && !lines.some((l) => /Same shape|l link/.test(l)), lines.join("\n"));
	learned.d.handleInput("l");
	assert.deepEqual(learned.linked, []);
	// A linkOffer that throws as it is called: nothing shown, the page as ever.
	const thrown = offerDock();
	thrown.host.linkOffer = () => { throw new Error("sync"); };
	thrown.serve("other");
	text(thrown.d);
	await tick();
	assert.ok(text(thrown.d).includes("no learned concepts on other: /concept learn NAME teaches one"));
	// The mini dock: no offer; l is its token key.
	const mini = offerDock();
	await tick();
	mini.host.mini = true;
	mini.d.render(40);
	mini.d.handleInput("l");
	await tick();
	assert.deepEqual(mini.linked, []);
});

test("l: the host's pick list; kept separate (or esc) changes nothing; linked, the concepts and the lens steps asked for again, the note", async () => {
	const keep = offerDock();
	await tick();
	keep.d.handleInput("l");
	await tick();
	assert.deepEqual(keep.linked, [parseOffer(OFFER, BUILD)]);
	assert.equal(keep.d["sp"].steer.note, undefined);
	assert.ok(text(keep.d).includes("l link to it · /concept learn NAME starts fresh"));
	let served = OFFER as any;
	const m = offerDock({ link: async () => {
		served = { ...OFFER, linked: true, candidates: [] };
		return [{ text: `linked to ${MODEL} · 14 concepts · 1 lens`, level: "info" }];
	} });
	m.host.linkOffer = async () => (m.asked.push("offer"), served);
	await tick();
	m.asked.length = 0;
	let list: any = { concepts: [] };
	m.host.steering = async () => (m.asked.push("steering"), parseSteering(list, BUILD, { calm: 20, robotic: 24 }));
	list = SHARED;
	m.d.handleInput("l");
	await tick();
	await tick();
	assert.deepEqual(m.asked.sort(), ["lens layers", "offer", "steering"]);
	const lines = text(m.d);
	// The model's concepts, not calibrated on this build: untested, how to calibrate one.
	assert.ok(lines.some((l) => /^▸ calm +untested/.test(l)), lines.join("\n"));
	assert.ok(lines.some((l) => l.includes("2 not calibrated on this build")) && !lines.some((l) => l.includes("experimental")), lines.join("\n"));
	const right = lines.map((l) => (l.includes("│") ? l.split("│").at(-1)!.trim() : "")).join(" ");
	assert.match(right, /not chat-tested: \/steer calm DOSE, or \/concept calibrate calm ✓ linked to Qwen3\.8-Flash-Next · 14 concepts · 1 lens/, lines.join("\n"));
	assert.ok(!lines.some((l) => /Same shape|l link/.test(l)));
	// While its answer is out, "asking for" and the note.
	const slow = offerDock({ link: async () => [{ text: "linked to x", level: "info" }] });
	await tick();
	slow.host.steering = () => new Promise(() => {});
	slow.d.handleInput("l");
	await tick();
	const wait = text(slow.d);
	assert.ok(wait.includes(`asking for ${BUILD}'s concepts…`) && wait.includes("linked to x"), wait.join("\n"));
});

test("l refused: the server's reason on the page's note line, the offer kept", async () => {
	const m = offerDock({ link: async () => [{ text: `${BUILD} has data of its own: nothing linked`, level: "error" }] });
	await tick();
	m.d.handleInput("l");
	await tick();
	const lines = text(m.d);
	assert.ok(lines.includes(`${BUILD} has data of its own: nothing linked`), lines.join("\n"));
	assert.ok(lines.includes("l link to it · /concept learn NAME starts fresh"));
	// A link that throws: its message as the note.
	const thrown = offerDock({ link: async () => { throw new Error("no dialog here"); } });
	await tick();
	thrown.d.handleInput("l");
	await tick();
	assert.deepEqual(thrown.d["sp"].steer.note, { text: "no dialog here", level: "error" });
});

test("a refusal (any error) reloads the page: the link may have gone through; found linked, it says so", async () => {
	let served: any = OFFER;
	const m = offerDock({ link: async () => {
		served = { ...OFFER, checkpoint: MODEL, linked: true, candidates: [] };          // it went through after all
		return [{ text: `${MODEL} is not a model of the same shape with data`, level: "error" }];
	} });
	m.host.linkOffer = async () => (m.asked.push("offer"), served);
	await tick();
	m.asked.length = 0;
	m.d.handleInput("l");
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual(m.asked, ["offer", "steering Qwen3.8-Flash-Next-oQ4e-mtp"]);
	assert.deepEqual(m.d["sp"].steer.note, { text: `already linked to ${MODEL}`, level: "info" });
	const lines = text(m.d);
	assert.ok(lines.includes(`already linked to ${MODEL}`) && !lines.some((l) => /not a model of the same shape|l link/.test(l)), lines.join("\n"));
	// Not linked after all: the refusal stays through that reload, the offer again.
	const r = offerDock({ link: async () => [{ text: "refused", level: "error" }] });
	await tick();
	r.d.handleInput("l");
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual(r.asked.filter((a) => a.startsWith("offer")).length, 2);
	assert.ok(text(r.d).includes("refused") && text(r.d).includes("l link to it · /concept learn NAME starts fresh"));
});

test("a refusal clears on the next l, on leaving the page, and when the offer is asked for again", async () => {
	let refuse = true;
	const m = offerDock({ link: async () => (refuse ? (refuse = false, [{ text: "refused", level: "error" }]) : undefined) });
	const refused = () => m.d["sp"].steer.note?.text === "refused";
	const press = async (k: string) => { m.d.handleInput(k); for (let i = 0; i < 3; i++) await tick(); };
	await tick();
	await press("l");
	assert.ok(refused());
	await press("l");                                                                 // kept separate this time
	assert.ok(!refused() && !text(m.d).includes("refused"));
	refuse = true;
	await press("l");
	await press("6");                                                                 // the board
	assert.ok(!refused());
	await press("6");
	refuse = true;
	await press("l");
	assert.ok(refused());
	m.d.handleInput("\x1b");                                                          // the keyboard back to Pi, then in again: the list and offer asked for again
	m.d.stepIn(-1);
	text(m.d);
	for (let i = 0; i < 3; i++) await tick();
	assert.ok(!refused());
});

test("a link's answer after the model changed, or after the dock closed, changes nothing", async () => {
	let done: (x: any) => void = () => {};
	const m = offerDock({ link: () => new Promise((r) => (done = r)) });
	await tick();
	m.d.handleInput("l");
	m.serve("Other");
	text(m.d);
	await tick();
	const asked = m.asked.length;
	done([{ text: `linked to ${MODEL}`, level: "info" }]);
	for (let k = 0; k < 3; k++) await tick();
	assert.equal(m.d["sp"].steer.note, undefined);
	assert.deepEqual(m.asked.slice(asked), []);
	const c = offerDock({ link: () => new Promise((r) => (done = r)) });
	await tick();
	c.d.handleInput("l");
	c.d.dispose();
	const before = c.asked.length;
	done([{ text: `linked to ${MODEL}`, level: "info" }]);
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual(c.asked.slice(before), [], "nothing asked for by a closed dock");
});

test("a link drops the dock's EXPERTS grids and tallies (they move to the model linked to)", async () => {
	const m = offerDock({ link: async () => [{ text: `linked to ${MODEL}`, level: "info" }] });
	await tick();
	m.d["grids"].at("http://x", BUILD, 3);
	m.d["experts"].xpData.set("k", 1);
	m.d["experts"].xpLast.set("u", 1);
	await tick();
	assert.equal(m.d["grids"]["got"].size, 1);
	m.d.handleInput("l");
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual([m.d["grids"]["got"].size, m.d["experts"].xpData.size, m.d["experts"].xpLast.size], [0, 0, 0]);
});

test("the offer is asked for as the list is (never per render), and dropped for another served model", async () => {
	const m = offerDock();
	await tick();
	text(m.d);                                                                         // stepping in on the page asks for its list again
	await tick();
	const asks = m.asked.filter((a) => a.startsWith("offer")).length;
	for (let k = 0; k < 10; k++) text(m.d);
	await tick();
	assert.equal(m.asked.filter((a) => a.startsWith("offer")).length, asks);
	assert.deepEqual(m.asked.filter((a) => a.startsWith("steering")).length, asks, "the offer asked for as the list is (the host keeps it: one request)");
	const was = m.host.linkOffer;
	assert.ok(text(m.d).includes("l link to it · /concept learn NAME starts fresh"));
	// Another model whose offer has not come yet, its list in and empty: the last model's offer is not shown meanwhile.
	m.host.linkOffer = () => new Promise(() => {});
	m.serve("Third-build");
	text(m.d);
	await tick();
	assert.ok(!text(m.d).some((l) => /Same shape|l link/.test(l)), "no offer of another model's");
	m.host.linkOffer = async () => (await was(), { ...OFFER, own: true, candidates: [] });   // Other-build has data of its own
	m.serve("Other-build");
	let lines = text(m.d);
	assert.ok(!lines.some((l) => /Same shape|l link/.test(l)), "another model: the last one's offer is gone at once");
	await tick();
	lines = text(m.d);
	assert.equal(m.asked.filter((a) => a.startsWith("offer")).at(-1), "offer Other-build");
	assert.ok(!lines.some((l) => /Same shape/.test(l)), lines.join("\n"));
	// Back to BUILD: its offer again (the host's answer for it).
	m.host.linkOffer = was;
	m.serve(BUILD);
	text(m.d);
	await tick();
	assert.ok(text(m.d).includes("l link to it · /concept learn NAME starts fresh"));
});

test("every offer line within the width, 30 to 120 columns; a malformed offer never throws", async () => {
	const long = { ...THREE, candidates: [...THREE.candidates, { name: "A-very-long-model-name-that-runs-past-any-narrow-dock-width-x", concepts: 123, lenses: 45 }] };
	for (let w = 30; w <= 120; w++) {
		const lines = steerPanel(theme, { list: { model: BUILD, concepts: [] }, model: BUILD, cur: 0, dose: 0, on: new Map(), focused: true,
			offer: parseOffer({ ...long, model: `${BUILD}-and-a-long-tail-${"x".repeat(40)}` }, `${BUILD}-and-a-long-tail-${"x".repeat(40)}`),
			note: { text: `${"a refusal that runs on ".repeat(8)}\n${"and a second line that runs on ".repeat(5)}`, level: w % 2 ? "error" : "info" } }, w);
		for (const l of lines) assert.ok(visibleWidth(l) <= w, `${w}: ${JSON.stringify(plain(l))}`);
		assert.ok(lines.length >= STEER_ROWS + 2);
	}
	for (const offer of [null, 1, "x", [], { candidates: [null] }, { candidates: [{ name: "a", concepts: 1e309 }] }, { shape: { n_layers: 1e400, width: "w" },
		candidates: [{ name: "\x1b]8;;evil\x07x".repeat(20), concepts: 2 }] }, { candidates: [{ name: "a", concepts: 2 }], shape: "48x" }]) {
		const m = offerDock({ offer });
		await tick();
		for (let width = 44; width <= 123; width += 13)
			for (const l of m.d.render(width)) assert.ok(visibleWidth(l) <= width, `${JSON.stringify(offer)} at ${width}: ${JSON.stringify(plain(l))}`);
		m.d.handleInput("l");
		await tick();
	}
	assert.equal(steerKeys(60, 1, true, false, true, true), "l link · x off · ? explain · 6 board · esc back");
	assert.equal(steerKeys(30, 1, true, false, true, true), "l link · x off · 6 board");
	assert.equal(steerKeys(101), "↑/↓ concept · ←/→ dose · enter steer · x off · d delete · n/N turn · ? explain · 6 board · esc back");   // no offer: as ever
});

// -- LENS's pointer ------------------------------------------------------------------------------------------------------

const LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47];
/** A reply of 40 tokens of a 48-layer model, read with the logit lens alone (`jlens`: with the J-lens and the logit
 * lens beside it), as the stream sends them. */
function lensLive(jlens = false): Live {
	const live: Live = { base: "http://x", clientId: "me", level: "on", connected: true, dense: "" } as any;
	onEvent(live, { type: "start", rid: "r", model: BUILD });
	for (let i = 0; i < 40; i++) {
		const rows = LAYERS.map((l) => [l, l >= 30 ? 100 + i : 7, l >= 30 ? ` w${i}` : " the"]);
		onEvent(live, { type: "token", rid: "r", i, id: 100 + i, text: ` w${i}`, p: 0.9, entropy: 0.2, alts: [[" a", 0.05]], n_layers: 48,
			lens: rows, lens_by: jlens ? "jlens" : "logit", ...(jlens ? { lens_logit: rows } : {}) } as any);
	}
	onEvent(live, { type: "end", rid: "r", n_tokens: 40, trace_id: "T" } as any);
	return live;
}
function lensBoard(o: { offer?: any; jlens?: boolean; rows?: number; list?: any } = {}) {
	const live = lensLive(o.jlens);
	const host: any = { live, captions: false, close() {}, unfocus() {}, panel: "LENS", steerModel: () => BUILD,
		steering: async () => parseSteering(o.list ?? { concepts: [] }, BUILD), linkOffer: async () => ("offer" in o ? o.offer : OFFER),
		lensLayers: async () => ({ model: BUILD, at: 1, steps: [{ name: "small", n: 4, layers: [24, 36, 44, 47], tps: null, replies: 0 }, { name: "medium", n: 13, layers: LAYERS, tps: null, replies: 0 }] }) };
	const d = new Dock({ terminal: { rows: o.rows ?? 52, columns: 200 }, requestRender() {} }, theme, host);
	return { d, host };
}
const lensText = async (b: ReturnType<typeof lensBoard>, width = 100) => {
	b.d.render(width);
	await tick();
	await tick();
	return text(b.d, width);
};

test("LENS points at the link where the model reads the logit lens alone and a same-shape model has a J-lens", async () => {
	for (const rows of [52, 76]) {
		const lines = await lensText(lensBoard({ rows }));
		const at = lines.findIndex((l) => l.startsWith(`${LENS_LINK[1]} `) && l.includes(" │"));     // LENS's column is 44 wide here: the shorter one
		assert.ok(at > 0, `${rows}\n${lines.join("\n")}`);
		assert.match(lines[at - 1], /^L\d+ /, "under SETTLING, on the line before the telescope");
		assert.equal(lines.filter((l) => l.includes("no J-lens")).length, 1);
	}
	// Wide enough: the whole pointer.
	const wide = await lensText(lensBoard(), 140);
	assert.ok(wide.some((l) => l.startsWith(`${LENS_LINK[0]} `)), wide.join("\n"));
	// No pointer: a J-lens read here, no candidate with a lens, no offer, another model's reply.
	// ... nor where the page would not offer it: concepts learned here (an offer kept from before them).
	const none = [lensBoard({ jlens: true }), lensBoard({ offer: { ...OFFER, candidates: [{ name: MODEL, concepts: 14, lenses: 0 }] } }), lensBoard({ offer: null }),
		lensBoard({ offer: { ...OFFER, own: true } }), lensBoard({ list: SHARED })];
	for (const b of none) assert.ok(!(await lensText(b)).some((l) => l.includes("no J-lens")));
	const other = lensBoard();
	other.host.steerModel = () => "Other";
	assert.ok(!(await lensText(other)).some((l) => l.includes("no J-lens")));
	// No room for it (a short dock: no blank line under SETTLING): it gives way, the rest as without it.
	for (const rows of [30, 36, 40]) {
		const b = lensBoard({ rows }), lines = await lensText(b);
		const bare = lensBoard({ rows, offer: null }), without = await lensText(bare);
		const pointer = lines.findIndex((l) => l.includes("no J-lens"));
		if (pointer < 0) assert.deepEqual(lines, without, `${rows}`);
		else assert.equal(without[pointer].split("│")[0].trim(), "", `${rows}: only on a blank line`);
	}
});

// -- through Pi: the host's offer, its pick list and the link ------------------------------------------------------------

/** The extension wired as Pi runs it, on `BUILD` (`select`: Pi's pick list's answer), the server's routes faked
 * (`answers` by path; a function's thrown error as the server's refusal). */
function wired(select: (title: string, options: string[]) => Promise<string | undefined>, answers: Record<string, any> = {}) {
	const asked: string[] = [], picks: [string, string[]][] = [], posted: any[] = [];
	const routes: Record<string, any> = { "/api/steering?": { concepts: [] }, "/api/concepts?": { model: BUILD, steer_enabled: true, concepts: [], library: [] },
		"/api/model/link?": OFFER, "/api/lens?": { model: BUILD, installed: false, lenses_dir: "/l" }, ...answers };
	const api = async (path: string, init?: any) => {
		asked.push(`${init?.method ?? "GET"} ${path}`);
		if (init?.method === "POST") posted.push([path, JSON.parse(init.body)]);
		const key = Object.keys(routes).find((k) => (init?.method === "POST" ? path === k : path.startsWith(k)));
		const a = key === undefined ? undefined : routes[key];
		if (a === undefined) throw Object.assign(new Error("Not Found"), { status: 404 });
		return typeof a === "function" ? a(path, init) : a;
	};
	const live: any = { base: "http://127.0.0.1:8009", clientId: "pi-x", level: "on", connected: true, reply: { rid: "r", tokens: [{ i: 0, id: 0, text: " w", p: 0.9, entropy: 0.1, alts: [] }], done: true, prefill: 0 } };
	let dock: any;
	const commands = new Map<string, any>();
	const ctx: any = { hasUI: true, model: { provider: "metalscope-ds4", id: BUILD },
		ui: { setWidget() {}, setStatus() {}, notify() {}, select: async (title: string, options: string[]) => (picks.push([title, options]), select(title, options)),
			custom: (make: any) => new Promise<void>((done) => { dock = make({ requestRender() {}, terminal: { rows: 52, columns: 200 } }, theme, {}, done); }) } };
	const pi = { registerCommand: (name: string, c: any) => void commands.set(name, c), appendEntry() {}, on: (e: string, f: any) => e === "session_start" && f({}, ctx),
		sendUserMessage() {}, registerShortcut() {} } as any;
	const ui = createUi({ pi, live, api } as any, join(mkdtempSync(join(tmpdir(), "ms-link-")), "pi.json"));
	registerCommands({ pi, live, api } as any, ui, {} as any, async () => {});
	ui.showDock(ctx, true);
	return { asked, picks, posted, ctx, commands, dock: () => dock as Dock };
}
const gets = (asked: string[], route: string) => asked.filter((a) => a.startsWith(`GET ${route}`)).length;

test("through Pi: one offer request a served model; the pick list keep first; keep or esc posts nothing; a pick posts {model, checkpoint}", async () => {
	let answer: string | undefined = LINK_KEEP;
	const w = wired(async () => answer, { "/api/model/link": () => LINKED });
	await tick();
	const d = w.dock();
	d.steerPage = true;
	for (let k = 0; k < 5; k++) text(d);
	d.handleInput("6");
	d.handleInput("6");                                                                // the page entered again: its list asked for again, the offer not
	await tick();
	assert.equal(gets(w.asked, "/api/model/link?"), 1, w.asked.join("\n"));
	assert.ok(text(d).includes("l link to it · /concept learn NAME starts fresh"));
	d.handleInput("l");
	await tick();
	assert.deepEqual(w.picks, [[`Link ${BUILD} to`, [LINK_KEEP, `${MODEL} · 14 concepts · 1 lens`]]]);
	answer = undefined;                                                                // esc
	d.handleInput("l");
	await tick();
	assert.deepEqual(w.posted, []);
	assert.equal(d["sp"].steer.note, undefined);
	// The completions' lists, asked for once.
	const complete = (p: string) => w.commands.get("concept").getArgumentCompletions(p);
	await complete("learn ");
	await complete("learn ");
	assert.equal(gets(w.asked, "/api/concepts?"), 1 + gets(w.asked, "/api/steering?"));
	const before = gets(w.asked, "/api/concepts?");
	answer = `${MODEL} · 14 concepts · 1 lens`;
	d.handleInput("l");
	for (let k = 0; k < 4; k++) await tick();
	assert.deepEqual(w.posted, [["/api/model/link", { model: BUILD, checkpoint: MODEL }]]);
	assert.equal(gets(w.asked, "/api/model/link?"), 2, "the offer asked for again after the link");
	assert.ok(gets(w.asked, "/api/lens/layers"), "LENS's steps asked for again");
	assert.deepEqual(d["sp"].steer.note, { text: `linked to ${MODEL} · 14 concepts · 1 lens`, level: "info" });
	const after = gets(w.asked, "/api/concepts?");
	await complete("learn ");
	assert.equal(gets(w.asked, "/api/concepts?"), after + 1, `the completions ask again (${before} → ${after})`);
	// Another served model: its own offer; back again: asked again (the last one dropped).
	w.ctx.model.id = "Other";
	text(d);
	await tick();
	w.ctx.model.id = BUILD;
	text(d);
	await tick();
	assert.deepEqual(w.asked.filter((a) => a.startsWith("GET /api/model/link?")).slice(-2), ["GET /api/model/link?model=Other", `GET /api/model/link?model=${BUILD}`]);
});

test("through Pi: a 409 shows the server's detail; a failed offer request is asked again next time, showing nothing", async () => {
	const w = wired(async (_t, options) => options[1], { "/api/model/link": () => { throw Object.assign(new Error(`${BUILD} has data of its own`), { status: 409 }); } });
	await tick();
	const d = w.dock();
	d.steerPage = true;
	text(d);
	await tick();
	d.handleInput("l");
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual(d["sp"].steer.note, { text: `${BUILD} has data of its own`, level: "error" });
	assert.ok(text(d).includes(`${BUILD} has data of its own`));
	// Any other answer the server refuses with: its detail too (500: stopped part way).
	const part = wired(async (_t, options) => options[1], { "/api/model/link": () => { throw Object.assign(new Error("the link stopped part way: disk full; run it again to finish it"), { status: 500 }); } });
	await tick();
	const pd = part.dock();
	pd.steerPage = true;
	text(pd);
	await tick();
	pd.handleInput("l");
	for (let k = 0; k < 3; k++) await tick();
	assert.deepEqual(pd["sp"].steer.note, { text: "the link stopped part way: disk full; run it again to finish it", level: "error" });
	let fails = 1;
	const f = wired(async () => undefined, { "/api/model/link?": () => (fails-- > 0 ? Promise.reject(new Error("ECONNREFUSED")) : OFFER) });
	await tick();
	await tick();
	assert.equal(gets(f.asked, "/api/model/link?"), 1);                               // as the dock opened: failed
	const fd = f.dock();
	fd.steerPage = true;
	text(fd);                                                                          // the page entered: asked again
	await tick();
	await tick();
	assert.equal(gets(f.asked, "/api/model/link?"), 2);
	assert.ok(text(fd).includes("l link to it · /concept learn NAME starts fresh"));
});

test("through Pi: the candidate picked is the one posted, for the model the offer was for", async () => {
	const w = wired(async (_t, options) => options[3], { "/api/model/link?": THREE, "/api/model/link": () => LINKED });
	await tick();
	const d = w.dock();
	d.steerPage = true;
	text(d);
	await tick();
	d.handleInput("l");
	for (let k = 0; k < 4; k++) await tick();
	assert.deepEqual(w.picks[0][1], [LINK_KEEP, `${MODEL} · 14 concepts · 1 lens`, "Qwen3.8-Flash-Next-ds4-q4k · 3 concepts", "Qwen3.8-Flash-Next-bf16 · 2 lenses"]);
	assert.deepEqual(w.posted, [["/api/model/link", { model: BUILD, checkpoint: "Qwen3.8-Flash-Next-bf16" }]]);
	// Pi's model switched while the pick list was up: the link is still for the model the offer named.
	const s = wired(async (_t, options) => ((s.ctx.model.id = "Other"), options[2]), { "/api/model/link?": THREE, "/api/model/link": () => LINKED });
	await tick();
	const sd = s.dock();
	sd.steerPage = true;
	text(sd);
	await tick();
	sd.handleInput("l");
	for (let k = 0; k < 4; k++) await tick();
	assert.deepEqual(s.posted, [["/api/model/link", { model: BUILD, checkpoint: "Qwen3.8-Flash-Next-ds4-q4k" }]]);
});

test("through Pi: an answer that may change (no shape yet, or nothing yet) is asked again each page entry; a lasting one once", async () => {
	for (const [why, answer, times] of [["no shape", { ...OFFER, shape: null, candidates: [] }, 3], ["nothing yet", { ...OFFER, candidates: [] }, 3],
		["linked", { ...OFFER, linked: true, candidates: [] }, 1], ["own", { ...OFFER, own: true, candidates: [] }, 1], ["offer", OFFER, 1]] as [string, any, number][]) {
		const w = wired(async () => undefined, { "/api/model/link?": answer });
		await tick();
		const d = w.dock();
		d.steerPage = true;
		text(d);
		await tick();
		d.handleInput("6");
		d.handleInput("6");
		text(d);
		await tick();
		for (let k = 0; k < 5; k++) text(d);
		assert.equal(gets(w.asked, "/api/model/link?"), times, why);
	}
});

test("through Pi: the link's own long wait; no answer may still have linked; the server's note under the link's line", async () => {
	let fail = true, inits: any[] = [];
	const w = wired(async (_t, options) => options[1], {
		"/api/model/link": (_p: string, init: any) => {
			inits.push(init);
			if (fail) throw Object.assign(new Error("The operation was aborted due to timeout"), { name: "TimeoutError" });
			return { ...LINKED, note: "the J-lens is read when the model next loads: none is loaded now" };
		} });
	await tick();
	const d = w.dock();
	d.steerPage = true;
	text(d);
	await tick();
	const offers = gets(w.asked, "/api/model/link?");
	const timeout = AbortSignal.timeout, waits: number[] = [];
	AbortSignal.timeout = (ms: number) => (waits.push(ms), timeout.call(AbortSignal, ms));
	try {
		d.handleInput("l");
		for (let k = 0; k < 4; k++) await tick();
	} finally {
		AbortSignal.timeout = timeout;
	}
	assert.ok(inits[0].signal instanceof AbortSignal && waits.includes(120_000), "a wait of its own (120 s), not the helper's 20 s");
	assert.deepEqual(d["sp"].steer.note, { text: "no answer: the link may have gone through", level: "error" });
	assert.equal(gets(w.asked, "/api/model/link?"), offers + 1, "the offer dropped and asked for again");
	fail = false;
	d.handleInput("l");
	for (let k = 0; k < 4; k++) await tick();
	const note = `linked to ${MODEL} · 14 concepts · 1 lens\nthe J-lens is read when the model next loads: none is loaded now`;
	assert.deepEqual(d["sp"].steer.note, { text: note, level: "info" });
	// Drawn: the link's line, then the server's note on the next, cut to the width.
	for (const width of [44, 60, 100]) {
		const lines = text(d, width);
		const at = lines.findIndex((l) => l.startsWith("linked to")), under = lines.findIndex((l) => l.startsWith("the J-lens is read"));
		assert.ok(at >= 0 && under > at && under <= at + 2, `${width}\n${lines.join("\n")}`);   // under the link's line (on up to two)
		for (const l of d.render(width)) assert.ok(visibleWidth(l) <= width);
	}
	// The server's note whole, on up to two lines, wherever the page puts it: the detail column at its narrowest (30, a
	// page 84 wide) up, stacked from the dock's least width. web.py's text (test_web_concepts pins it there).
	const server = "J-lens loads at the next model load (the server log says why)";
	const list = parseSteering(SHARED, BUILD, { calm: 20, robotic: 24 });
	for (let pw = 30; pw <= 140; pw++) {
		const lines = steerPanel(theme, { list, model: BUILD, cur: 0, dose: 0, on: new Map(), focused: true,
			note: { text: linkedNote({ ...LINKED, lens: null, note: server }, OFFER.candidates[0]), level: "info" } }, pw).map(plain);
		const right = lines.map((l) => (l.includes("│") ? l.split("│").at(-1)! : l).trim()), at = right.findIndex((l) => l.startsWith("J-lens"));
		assert.ok(at > 0 && (right[at] === server || right.slice(at, at + 2).join(" ") === server), `${pw}\n${lines.join("\n")}`);
	}
	// A 409 whose reload finds the build linked (another Pi linked it): said so, not the refusal.
	let linked = false;
	const o = wired(async (_t, options) => options[1], {
		"/api/model/link?": () => (linked ? { ...OFFER, checkpoint: MODEL, linked: true, candidates: [] } : OFFER),
		"/api/model/link": () => { linked = true; throw Object.assign(new Error(`${MODEL} is not a model of the same shape with data`), { status: 409 }); } });
	await tick();
	const od = o.dock();
	od.steerPage = true;
	text(od);
	await tick();
	od.handleInput("l");
	for (let k = 0; k < 4; k++) await tick();
	assert.deepEqual(od["sp"].steer.note, { text: `already linked to ${MODEL}`, level: "info" });
});

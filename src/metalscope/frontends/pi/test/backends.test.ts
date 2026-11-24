import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Catalogue, DS4, OMLX, cliCommand, confirmSwitch, ds4Serving, holdRequest, isServed, modelsOf, readCatalogue, registerCatalogue, runningOn, runSwitch, servedCache, switchArgs, updateServers,
	whatSwitchStops } from "../backends.ts";
import { DS4_URL, type Server } from "../provider.ts";

function fakePi() {
	const calls: [string, any][] = [];
	return { calls, pi: { registerProvider: (name: string, cfg: any) => calls.push([name, cfg]) } as any };
}

async function withFetch(routes: Record<string, any>, fn: () => Promise<void>) {
	const saved = globalThis.fetch;
	globalThis.fetch = (async (url: string) => {
		if (!(url in routes)) throw new Error("ECONNREFUSED");
		return { ok: true, json: async () => routes[url] };
	}) as any;
	try {
		await fn();
	} finally {
		globalThis.fetch = saved;
	}
}

const omlx = { base: "http://o:1", key: "k" };
const catalogue = (over: Partial<Catalogue["ds4"]> = {}): Catalogue => ({
	omlx: { models: [
		{ name: "Qwen3.8-Flash-Next-oQ4e-mtp", ctx: 524288, loaded: true, reasoning: true, thinking_format: "qwen-chat-template" },
		{ name: "Qwen3.5-2B-4bit", ctx: 262144, loaded: false, reasoning: false, thinking_format: null, tested: false, edit: false },
	] },
	ds4: { models: [
		{ name: "Qwen3.8-Flash-Next-ds4-q4k", ctx: 262144, loaded: false, reasoning: true, thinking_format: "qwen-chat-template" },
		{ name: "DeepSeek-V4.1-Flash-ds4-q2", ctx: 1048576, loaded: false, reasoning: true, thinking_format: "qwen-chat-template" },
	] , ...over },
});

test("cliCommand: the Python `metalscope pi install` recorded, else metalscope on PATH", () => {
	const home = mkdtempSync(join(tmpdir(), "ms-"));
	assert.deepEqual(cliCommand({ METALSCOPE_HOME: home }), ["metalscope"]);
	writeFileSync(join(home, "cli.json"), JSON.stringify({ python: "/v/bin/python" }));
	assert.deepEqual(cliCommand({ METALSCOPE_HOME: home }), ["/v/bin/python", "-m", "metalscope.cli"]);
	writeFileSync(join(home, "cli.json"), "{not json");
	assert.deepEqual(cliCommand({ METALSCOPE_HOME: home }), ["metalscope"]);
});

test("registerCatalogue: both backends' models whether their server runs or not, each at its context", async () => {
	const { calls, pi } = fakePi();
	const servers = new Map<string, Server>();
	assert.deepEqual(registerCatalogue(pi, catalogue(), omlx, servers, async () => undefined), [OMLX, DS4]);
	assert.deepEqual(calls.map(([name, cfg]) => [name, cfg.baseUrl, cfg.apiKey, cfg.models.map((m: any) => [m.id, m.contextWindow])]), [
		[OMLX, "http://o:1/v1", "k", [["Qwen3.8-Flash-Next-oQ4e-mtp", 524288], ["Qwen3.5-2B-4bit", 262144]]],
		[DS4, `${DS4_URL}/v1`, "metalscope", [["Qwen3.8-Flash-Next-ds4-q4k", 262144], ["DeepSeek-V4.1-Flash-ds4-q2", 1048576]]],
	]);
	// What is loaded (the default server to follow) and each model's thinking controls, as the servers would say.
	assert.deepEqual(servers.get(OMLX)!.status.models, [{ name: "Qwen3.8-Flash-Next-oQ4e-mtp" }]);
	assert.deepEqual(servers.get(DS4)!.status.models, []);
	assert.equal(servers.get(DS4)!.status.chat!["DeepSeek-V4.1-Flash-ds4-q2"].thinking_format, "qwen-chat-template");
	assert.equal(calls[0][1].models[1].reasoning, false);
	// Each model's flags as `metalscope status` gave them (the dock's untested and its edit); none listed, none kept.
	assert.deepEqual([servers.get(OMLX)!.status.chat!["Qwen3.5-2B-4bit"].tested, servers.get(OMLX)!.status.chat!["Qwen3.5-2B-4bit"].edit], [false, false]);
	assert.equal(servers.get(OMLX)!.status.chat!["Qwen3.8-Flash-Next-oQ4e-mtp"].tested, undefined);
	// /model asks again when it opens: the catalogue then, and the servers with it.
	const now = catalogue();
	now.ds4.models[0].loaded = true;
	const { calls: c2, pi: pi2 } = fakePi();
	registerCatalogue(pi2, catalogue(), omlx, servers, async () => now);
	assert.deepEqual((await c2[1][1].refreshModels()).map((m: any) => m.id), ["Qwen3.8-Flash-Next-ds4-q4k", "DeepSeek-V4.1-Flash-ds4-q2"]);
	assert.deepEqual(servers.get(DS4)!.status.models, [{ name: "Qwen3.8-Flash-Next-ds4-q4k" }]);
});

test("registerCatalogue: a backend with no models yet is registered all the same (/model fills it); updateServers follows a switch", async () => {
	const { calls, pi } = fakePi();
	const servers = new Map<string, Server>();
	const cat = catalogue({ models: [] });
	assert.deepEqual(registerCatalogue(pi, cat, omlx, servers, async () => catalogue()), [OMLX, DS4]);
	assert.deepEqual(calls[1][1].models, []);
	assert.equal((await calls[1][1].refreshModels()).length, 2);   // a GGUF served since
	const after = catalogue();
	after.omlx.models[0].loaded = false;
	updateServers(after, omlx, servers);
	assert.deepEqual(servers.get(OMLX)!.status.models, []);
});

test("readCatalogue: stopped by its signal (session shutdown), nothing", async () => {
	const stop = new AbortController(), t0 = Date.now();
	const got = readCatalogue(["node", "-e", "setTimeout(() => console.log('{}'), 5000)", "--"], 10_000, stop.signal);
	setTimeout(() => stop.abort(), 50);
	assert.equal(await got, undefined);
	assert.ok(Date.now() - t0 < 2000);
});

test("modelsOf: no usable context reported: Pi's default window", () => {
	assert.deepEqual(modelsOf([{ name: "x", ctx: null }]).map((m) => [m.contextWindow, m.maxTokens]), [[131072, 32768]]);
});

test("switchArgs: the command that makes a backend's model the one served; none for other providers", () => {
	assert.deepEqual(switchArgs(DS4, "Qwen3.8-Flash-Next-ds4-q4k"), ["ds4", "Qwen3.8-Flash-Next-ds4-q4k"]);
	assert.deepEqual(switchArgs(OMLX, "Qwen3.5-2B-4bit"), ["omlx", "Qwen3.5-2B-4bit"]);
	assert.equal(switchArgs("anthropic", "claude"), undefined);
});

test("isServed: ds4's model when its proxy lists it; oMLX's when oMLX lists it and ds4 is not running", async () => {
	const ds4Has = { [`${DS4_URL}/v1/models`]: { data: [{ id: "Q-ds4" }] } };
	const omlxHas = { "http://o:1/v1/models": { data: [{ id: "Q-omlx" }] } };
	const none = () => undefined, loading = () => ({ pid: 1, name: "Q-ds4" });
	await withFetch(ds4Has, async () => {
		assert.equal(await isServed(DS4, "Q-ds4", omlx, loading), true);
		assert.equal(await isServed(DS4, "Q-ds4", omlx, none), true);           // a ds4 started without metalscope: its proxy says
		assert.equal(await isServed(DS4, "Other-ds4", omlx, none), false);      // ds4 serves another: switch
		assert.equal(await isServed(DS4, "Other-ds4", omlx, () => ({ pid: 1, name: "Other-ds4" })), false);   // still loading
	});
	await withFetch({}, async () => assert.equal(await isServed(DS4, "Q-ds4", omlx, none), false));   // ds4 down
	await withFetch({ ...ds4Has, ...omlxHas }, async () => assert.equal(await isServed(OMLX, "Q-omlx", omlx, none), false));   // ds4 holds memory
	await withFetch(omlxHas, async () => {
		assert.equal(await isServed(OMLX, "Q-omlx", omlx, none), true);
		assert.equal(await isServed(OMLX, "Q-omlx", omlx, loading), false);   // ds4 loading: its ports closed, its memory filling
		assert.equal(await isServed("anthropic", "claude", omlx), true);
	});
	await withFetch({}, async () => assert.equal(await isServed(OMLX, "Q-omlx", omlx, none), false));   // oMLX down: start it
});

test("servedCache: ds4's record read before every request; only a model listing's yes kept (until forget), for ds4 by its process", async () => {
	const ds4Has = (id: string) => ({ [`${DS4_URL}/v1/models`]: { data: [{ id }] } });
	const omlxHas = { "http://o:1/v1/models": { data: [{ id: "A" }] } };
	let rec: { pid: number; name: string } | undefined, asked = 0;
	const counting = async (routes: Record<string, any>, fn: () => Promise<void>) =>
		withFetch(new Proxy(routes, { has: (o, k) => (asked++, k in o) }), fn);
	const s = servedCache(() => rec);
	// oMLX's A, then a terminal's `metalscope ds4 V4.1`: the next request is not sent to oMLX (it would load A beside it).
	await counting(omlxHas, async () => {
		assert.equal(await s.is(OMLX, "A", omlx), true);
		const n = asked;
		assert.equal(await s.is(OMLX, "A", omlx), true);
		assert.equal(asked, n + 1, "oMLX's listing kept; ds4's port asked again");
		rec = { pid: 7, name: "V4.1-ds4" };
		assert.equal(await s.is(OMLX, "A", omlx), false);
	});
	// ds4 serving X, then another client switches it to Y: X is not served.
	rec = { pid: 7, name: "X" };
	await counting(ds4Has("X"), async () => {
		assert.equal(await s.is(DS4, "X", omlx), true);
		const n = asked;
		assert.equal(await s.is(DS4, "X", omlx), true);
		assert.equal(asked, n, "its listing kept while the same process serves X");
		rec = { pid: 7, name: "Y" };
		assert.equal(await s.is(DS4, "X", omlx), false);
		rec = { pid: 8, name: "X" };                      // X again, a new process: asked again
		assert.equal(await s.is(DS4, "X", omlx), true);
		assert.equal(asked, n + 1);
	});
	s.forget();                                          // a failed request, /model
	await counting({}, async () => assert.equal(await s.is(DS4, "X", omlx), false));
});

test("runSwitch: each line as it comes; a failure rejects with its last lines", async () => {
	const lines: string[] = [];
	await runSwitch(["ds4", "X"], (l) => lines.push(l), ["node", "-e", "console.log('unloaded A\\nds4 loading X'); console.error('ds4 serves X')"]);
	assert.deepEqual(lines.sort(), ["ds4 loading X", "ds4 serves X", "unloaded A"]);
	// A line written in pieces is one line; the last, with no newline, is not lost.
	const split: string[] = [];
	await runSwitch(["ds4", "X"], (l) => split.push(l), ["node", "-e",
		"const w = (s) => new Promise((r) => process.stdout.write(s, () => setTimeout(r, 30))); (async () => { await w('loading 4'); await w('0%\\nloa'); await w('ded'); })()"]);
	assert.deepEqual(split, ["loading 40%", "loaded"]);
	await assert.rejects(runSwitch(["ds4", "X"], () => {}, ["node", "-e", "console.error('no ds4 model'); process.exit(1)"]), /no ds4 model/);
	await assert.rejects(runSwitch(["ds4", "X"], () => {}, ["/nonexistent/metalscope"]), /could not run/);
});

test("runningOn: the unfinished job it reports and its replies in progress; nothing when it does not answer", async () => {
	const at = "http://o:1/scope";
	await withFetch({ [`${at}/api/status`]: { active: 2 }, [`${at}/api/jobs`]: { latest: { kind: "calibrate", job: "c-1", name: "fluffy", outcome: null }, jobs: [] } },
		async () => assert.deepEqual(await runningOn("http://o:1"), { job: { id: "c-1", name: "calibrate fluffy" }, replies: 2 }));
	await withFetch({ [`${at}/api/status`]: { active: 0 }, [`${at}/api/jobs`]: { latest: { kind: "learn", job: "l-1", name: "x", outcome: "done" }, jobs: [] } },
		async () => assert.deepEqual(await runningOn("http://o:1"), { job: undefined, replies: 0 }));   // a finished job stops nothing
	await withFetch({}, async () => assert.equal(await runningOn("http://o:1"), undefined));
});

test("confirmSwitch: nothing running goes on; a reply asks first, no yes refuses; a job (once, from the shared store) refuses without asking", async () => {
	const servers = new Map<string, Server>();
	registerCatalogue(fakePi().pi, catalogue(), omlx, servers, async () => undefined);
	const quiet = async () => ({ replies: 0 });
	const replying = async (base: string) => (base === omlx.base ? { replies: 0 } : { replies: 1 });
	const asked: [string, string][] = [];
	const ctx = (answer: boolean) => ({ hasUI: true, ui: { confirm: async (title: string, text: string) => (asked.push([title, text]), answer) } });
	const none = () => undefined;
	await confirmSwitch(ctx(false), "Q-ds4", "ds4", servers.values(), quiet, none);
	assert.equal(asked.length, 0);                                               // nothing runs: no question
	await confirmSwitch(ctx(true), "Q-ds4", "ds4", servers.values(), replying, none);   // yes: the switch goes on
	assert.deepEqual(asked[0], ["Start Q-ds4 on ds4?",
		"ds4 has 1 reply in progress.\nStarting Q-ds4 now stops it.\nNo: nothing changes, and no message goes to Q-ds4 meanwhile."]);
	const refused = /^Q-ds4 not started on ds4: no message goes to it meanwhile\.\nds4 has 1 reply in progress\.\nSend your message again once it is done: Q-ds4 starts then\.$/;
	await assert.rejects(confirmSwitch(ctx(false), "Q-ds4", "ds4", servers.values(), replying, none), (e: any) => e.refused && refused.test(e.message));
	await assert.rejects(confirmSwitch({ hasUI: false }, "Q-ds4", "ds4", servers.values(), replying, none), (e: any) => e.refused);   // no UI to ask: not started
	assert.equal(asked.length, 2);
	// One job store under both servers: each reports the same job. metalscope refuses every switch while it runs, so
	// there is nothing to ask: the way out is to stop it.
	const job = { id: "c-1", name: "calibrate fluffy" };
	const busy = async (base: string) => (base === omlx.base ? { job, replies: 0 } : { job, replies: 1 });
	assert.deepEqual(await whatSwitchStops(servers.values(), busy, none), ["a job is running: calibrate fluffy", "ds4 has 1 reply in progress"]);
	await assert.rejects(confirmSwitch(ctx(true), "Q-ds4", "ds4", servers.values(), busy, none), (e: any) => e.refused &&
		e.message === "Q-ds4 not started on ds4: a job is running: calibrate fluffy. /scope jobs cancel stops it; then send again.");
	assert.equal(asked.length, 2);
});

test("confirmSwitch: ds4 loading (its record live, its port not answering yet) asks first, unless the switch is to that model on ds4", async () => {
	const servers = new Map<string, Server>();
	registerCatalogue(fakePi().pi, catalogue(), omlx, servers, async () => undefined);
	const loading = () => ({ pid: 1, name: "V4.1-ds4" });
	const omlxOnly = async (base: string) => (base === omlx.base ? { replies: 0 } : undefined);
	const asked: string[] = [];
	const ctx = { hasUI: true, ui: { confirm: async (_: string, text: string) => (asked.push(text), true) } };
	await confirmSwitch(ctx, "Q-omlx", "oMLX", servers.values(), omlxOnly, loading);
	assert.match(asked[0], /^ds4 is loading V4\.1-ds4\.\nStarting Q-omlx now stops it\./);
	await confirmSwitch(ctx, "Q-ds4", "ds4", servers.values(), omlxOnly, loading);        // another ds4 model: asked
	await confirmSwitch(ctx, "V4.1-ds4", "ds4", servers.values(), omlxOnly, loading);     // the one loading: waited for
	await confirmSwitch(ctx, "Q-omlx", "oMLX", servers.values(), async () => ({ replies: 0 }), loading);   // its port answers: loaded
	assert.equal(asked.length, 2);
});

test("holdRequest: a refused or failed switch aborts the request (Pi sends it whatever the hook throws); print mode says why on stderr", async () => {
	const fake = (hasUI: boolean) => {
		const c = new AbortController();
		const said: string[] = [];
		return { said, ctx: { hasUI, signal: c.signal, aborted: 0, abort() { this.aborted++; c.abort(); } } };
	};
	const refusal = Object.assign(new Error("Q-ds4 not started on ds4: no message goes to it meanwhile."), { refused: true });
	const ui = fake(true);
	await holdRequest(Promise.reject(refusal), ui.ctx, (s) => void ui.said.push(s));
	assert.equal(ui.ctx.aborted, 1);
	assert.deepEqual(ui.said, []);                                                // with a UI, the switch's own note says it
	const print = fake(false);
	await holdRequest(Promise.reject(refusal), print.ctx, (s) => void print.said.push(s));
	assert.equal(print.ctx.aborted, 1);
	assert.deepEqual(print.said, ["Q-ds4 not started on ds4: no message goes to it meanwhile.\n"]);
	const done = fake(true);                                                      // done: the request goes on
	await holdRequest(Promise.resolve(), done.ctx, (s) => void done.said.push(s));
	assert.equal(done.ctx.aborted, 0);
	// Failed (metalscope refused it, or the backend did not start): the request never goes to a model not started.
	const failure = new Error("cannot switch backends now: a job is running: calibrating fluffy (/scope jobs); wait for it or stop it: metalscope jobs cancel c-1");
	const failed = fake(false);
	await holdRequest(Promise.reject(failure), failed.ctx, (s) => void failed.said.push(s));
	assert.equal(failed.ctx.aborted, 1);
	assert.deepEqual(failed.said, [`${failure.message}\n`]);
});

test("ds4Serving: its serving record's process and model while that process lives", () => {
	const home = mkdtempSync(join(tmpdir(), "ms-"));
	assert.equal(ds4Serving({ METALSCOPE_HOME: home }), undefined);
	mkdirSync(join(home, "ds4"));
	writeFileSync(join(home, "ds4", "serving.json"), JSON.stringify({ pid: process.pid, name: "Q-ds4" }));
	assert.deepEqual(ds4Serving({ METALSCOPE_HOME: home }), { pid: process.pid, name: "Q-ds4" });
	writeFileSync(join(home, "ds4", "serving.json"), JSON.stringify({ pid: 2 ** 22 + 12345, name: "Q-ds4" }));
	assert.equal(ds4Serving({ METALSCOPE_HOME: home }), undefined);   // a stale record
});

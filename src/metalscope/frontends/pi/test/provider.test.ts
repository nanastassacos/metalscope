import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runningOn } from "../backends.ts";
import { DEFAULT_CONTEXT, DS4_URL, PROVIDER, type Server, contextWindow, defaultServer, keptFirst, keptText, nextBase, omlxServer, onMetalscope, refreshFlags, registerUrl,
	requestFailed, scopeAuth, serverFor, staleNotices, traceApi } from "../provider.ts";
import { type Live, follow, onEvent } from "../state.ts";

/** A fake Pi that records provider registrations. */
function fakePi() {
	const calls: [string, any][] = [];
	return { calls, pi: { registerProvider: (name: string, cfg: any) => calls.push([name, cfg]) } as any };
}

/** fetch answering from `routes` (URL -> JSON body); any other URL is refused, as a server that is down. */
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

test("omlxServer: oMLX's address and key from its settings; 127.0.0.1:8008 without them", () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-"));
	const settings = join(dir, "settings.json");
	writeFileSync(settings, JSON.stringify({ server: { host: "0.0.0.0", port: 8100 }, auth: { api_key: "k" } }));
	assert.deepEqual(omlxServer(settings), { base: "http://127.0.0.1:8100", key: "k" });
	assert.deepEqual(omlxServer(join(dir, "none.json")), { base: "http://127.0.0.1:8008", key: "" });
});

test("scopeAuth: oMLX's key to oMLX's own address only", () => {
	const omlx = { base: "http://127.0.0.1:8100", key: "k" };
	assert.deepEqual(scopeAuth("http://127.0.0.1:8100/scope/api/live?client=pi-x", omlx), { Authorization: "Bearer k" });
	for (const url of [DS4_URL, "http://127.0.0.1:8101", "http://127.0.0.1:8100.evil.example", "not a url"]) assert.deepEqual(scopeAuth(url, omlx), {}, url);
	assert.deepEqual(scopeAuth(omlx.base, { ...omlx, key: "" }), {});                    // no key set: none sent
});

test("every request to metalscope's routes sends oMLX's key to oMLX, from its settings, and to no other server", async () => {
	const home = mkdtempSync(join(tmpdir(), "ms-home-"));
	mkdirSync(join(home, ".omlx"));
	writeFileSync(join(home, ".omlx", "settings.json"), JSON.stringify({ server: { port: 8100 }, auth: { api_key: "k" } }));
	const omlx = "http://127.0.0.1:8100", sent: [string, string | undefined][] = [], abort = new AbortController();
	const savedHome = process.env.HOME, savedFetch = globalThis.fetch;
	process.env.HOME = home;                                   // Pi reads oMLX's settings under the home
	globalThis.fetch = (async (url: string, init: any) => {
		sent.push([url, init?.headers?.Authorization]);
		if (url.includes("/api/live")) abort.abort();
		return { ok: true, json: async () => ({}) };
	}) as any;
	try {
		const api = traceApi({ base: omlx });
		await api("/api/jobs", { method: "POST", body: "{}", headers: { "x-other": "1" } });
		await api("/api/jobs", undefined, DS4_URL);
		await staleNotices([server(omlx, "omlx"), server(DS4_URL, "ds4")]);       // the status
		await runningOn(omlx);
		await follow({ ...newLive(), base: omlx } as any, abort.signal);
	} finally {
		process.env.HOME = savedHome;
		globalThis.fetch = savedFetch;
	}
	assert.deepEqual(sent.map(([u, a]) => [u.replace(/\?.*/, ""), a]), [
		[`${omlx}/scope/api/jobs`, "Bearer k"], [`${DS4_URL}/scope/api/jobs`, undefined],
		[`${omlx}/scope/api/status`, "Bearer k"], [`${DS4_URL}/scope/api/status`, undefined],
		[`${omlx}/scope/api/status`, "Bearer k"], [`${omlx}/scope/api/jobs`, "Bearer k"],
		[`${omlx}/scope/api/live`, "Bearer k"]]);
});

const newLive = (): Live => ({ base: "http://o:1", clientId: "pi-x", level: "on", connected: false });

const server = (base: string, engine: string, models: string[] = []): Server => ({ base, key: "", status: { engine, models: models.map((name) => ({ name })) } });

test("contextWindow: the server's max_model_len; nothing usable: the default", () => {
	assert.equal(contextWindow(1048576), 1048576);
	assert.equal(contextWindow(32768), 32768);
	for (const bad of [undefined, null, 0, 512, -1, 1.5, "1048576", NaN]) assert.equal(contextWindow(bad), DEFAULT_CONTEXT, String(bad));
});

test("onMetalscope: every metalscope-<engine> provider, nothing else", () => {
	assert.ok(onMetalscope({ model: { provider: "metalscope-omlx" } }));
	assert.ok(onMetalscope({ model: { provider: "metalscope-ds4" } }));
	assert.ok(!onMetalscope({ model: { provider: PROVIDER } }));
	assert.ok(!onMetalscope({ model: { provider: "omlx" } }));
	assert.ok(!onMetalscope({}));
});

test("serverFor / defaultServer: the selected model's server; before one is picked, the first serving a model", () => {
	const omlx = server("http://o:1", "omlx"), ds4 = server("http://d:2", "ds4", ["V4"]);
	const servers = new Map([["metalscope-omlx", omlx], ["metalscope-ds4", ds4]]);
	assert.equal(serverFor(servers, { model: { provider: "metalscope-ds4" } }), ds4);
	assert.equal(serverFor(servers, { model: { provider: "metalscope-omlx" } }), omlx);
	assert.equal(serverFor(servers, { model: { provider: "anthropic" } }), undefined);
	assert.equal(defaultServer(servers.values()), ds4);                                  // oMLX up, nothing loaded
	assert.equal(defaultServer([server("http://o:1", "omlx", ["Q"]), ds4])?.base, "http://o:1");  // both: oMLX's
	assert.equal(defaultServer([omlx]), omlx);                                            // none serving: the first
	assert.equal(defaultServer([]), undefined);
});

test("registerUrl: down, or no chat model, registers nothing", async () => {
	const routes = { "http://e:1/v1/models": { data: [{ id: "bge-embed" }] }, "http://e:1/scope/api/status": { engine: "omlx" } };
	await withFetch(routes, async () => {
		const { calls, pi } = fakePi();
		const servers = new Map<string, Server>();
		assert.equal(await registerUrl(pi, "http://127.0.0.1:9", servers), undefined);
		assert.equal(await registerUrl(pi, "http://e:1", servers), undefined);
		assert.equal(calls.length, 0);
		assert.equal(servers.size, 0);
	});
});

test("nextBase: the selected model's server; no metalscope model: stay on a registered server, else the default", () => {
	const omlx = server("http://o:1", "omlx"), ds4 = server("http://d:2", "ds4", ["V4"]);
	const servers = new Map([["metalscope-omlx", omlx], ["metalscope-ds4", ds4]]);
	assert.equal(nextBase(servers, { model: { provider: "metalscope-ds4" } }, "http://o:1"), "http://d:2");
	assert.equal(nextBase(servers, { model: { provider: "metalscope-omlx" } }, "http://d:2"), "http://o:1");
	assert.equal(nextBase(servers, { model: { provider: "anthropic" } }, "http://o:1"), "http://o:1");   // stays
	assert.equal(nextBase(servers, undefined, "http://dead:9"), "http://d:2");                          // the default: serving
	assert.equal(nextBase(new Map(), undefined, "http://dead:9"), "http://dead:9");                     // nothing registered
});

test("registerUrl: its chat models (not embedders) as provider metalscope-<engine>, context and thinking as the server reports them", async () => {
	const routes = {
		"http://h:1/v1/models": { data: [{ id: "Qwen3-8B", max_model_len: 262144 }, { id: "bge-embed" }, { id: "Plain-7B", max_model_len: 8192 }, { id: "R1" }] },
		"http://h:1/scope/api/status": { engine: "mock", models: [], chat: { "Qwen3-8B": { reasoning: true, thinking_format: "qwen-chat-template" },
			"Plain-7B": { reasoning: false, thinking_format: null }, R1: { reasoning: true, thinking_format: null } } },
	};
	await withFetch(routes, async () => {
		const { calls, pi } = fakePi();
		const servers = new Map<string, Server>();
		assert.equal(await registerUrl(pi, "http://h:1/", servers), `${PROVIDER}-mock`);
		assert.equal(calls.length, 1);
		assert.equal(calls[0][0], `${PROVIDER}-mock`);
		assert.equal(calls[0][1].baseUrl, "http://h:1/v1");
		assert.equal(servers.get(`${PROVIDER}-mock`)!.base, "http://h:1");
		const models = calls[0][1].models;
		assert.deepEqual(models.map((m: any) => m.id), ["Qwen3-8B", "Plain-7B", "R1"]);
		assert.deepEqual(models.map((m: any) => [m.reasoning, m.compat.thinkingFormat]),
			[[true, "qwen-chat-template"], [false, undefined], [true, undefined]]);
		// Each model's context as its server reports it: when Pi compacts. A window under 32768 caps the output too.
		assert.deepEqual(models.map((m: any) => [m.contextWindow, m.maxTokens]), [[262144, 32768], [8192, 8192], [DEFAULT_CONTEXT, 32768]]);
		assert.ok(models.every((m: any) => !/oMLX|ds4/.test(m.name)));
	});
});

test("refreshFlags: a reply's start asks its server once a model (oMLX's edit and caps, known once loaded); then only while one is unknown", async () => {
	const servers = new Map<string, Server>([["metalscope-omlx", { base: "http://o", key: "", status: { chat: {
		Q: { reasoning: true, tested: true }, D: { tested: false, edit: false } } } }]]);
	const asked: string[] = [];
	const status = async (base: string) => (asked.push(base), { chat: { Q: { reasoning: true, tested: true, edit: false } } });
	await refreshFlags(servers, "http://o", "Q", status);
	assert.deepEqual(servers.get("metalscope-omlx")!.status.chat!.Q, { reasoning: true, tested: true, edit: false });
	await refreshFlags(servers, "http://o", "Q", status);                            // known now: nothing asked
	await refreshFlags(servers, "http://o", "D", status);                            // D's first start: asked, all known or not
	await refreshFlags(servers, "http://o", "D", status);
	await refreshFlags(servers, "http://elsewhere", "Q", status);                    // not a registered server
	assert.deepEqual(asked, ["http://o", "http://o"]);
	// The live stream's start of a reply is what calls it (index.ts sets Live.onStart), once per reply.
	const started: (string | undefined)[] = [];
	const live: Live = { base: "http://o", clientId: "c", level: "on", connected: true, onStart: (m) => started.push(m) };
	onEvent(live, { type: "start", rid: "r1", model: "Q" });
	onEvent(live, { type: "start", rid: "r1", model: "Q" });                          // the same start from a second stream
	onEvent(live, { type: "token", rid: "r1", i: 0, id: 1, text: "a", p: 0.9, entropy: 0, alts: [] });
	assert.deepEqual(started, ["Q"]);
});

test("staleNotices: a warning for each server whose status says it runs older code than is installed; none without `stale`", async () => {
	const server = (base: string): Server => ({ base, key: "", status: {} });
	const servers = [server("http://o"), server("http://d"), server("http://o"), server("http://x"), server("http://down")];
	const asked: string[] = [];
	const said: Record<string, any> = { "http://o": { engine: "omlx", stale: "metalscope omlx restart" },
		"http://d": { engine: "ds4", stale: "metalscope ds4 stop, then metalscope ds4" }, "http://x": { engine: "omlx" } };
	const status = async (base: string) => (asked.push(base), said[base]);
	assert.deepEqual(await staleNotices(servers, status), ["server code changed: metalscope omlx restart", "server code changed: metalscope ds4 stop, then metalscope ds4"]);
	assert.deepEqual(asked, ["http://o", "http://d", "http://x", "http://down"]);   // each address once
	said["http://o"].stale = null;
	said["http://d"] = { stale: "restart it" };                                      // the server's text, as given
	assert.deepEqual(await staleNotices(servers, status), ["server code changed: restart it"]);
});

test("keptFirst: an edit's answer starts with the reply up to the edited token at once; the server's copy of it is not shown twice; other requests are Pi's stream as is", async () => {
	const { createAssistantMessageEventStream } = await import("@earendil-works/pi-ai/compat");
	const toks = ["Let", " me", " think", "</think>", "\n\n", "The", " answer", " is", " forty"].map((text, i) => ({ type: "token", rid: "r", i, id: i, text, p: 0.9, entropy: 0, alts: [] }));
	const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r", model: "m" });
	for (const t of toks) onEvent(live, t);
	// The server's answer, after a wait: the kept part (thinking, then the text up to the edited token), then the rest.
	let release!: () => void;
	const go = new Promise<void>((r) => (release = r));
	const inner = () => {
		const s = createAssistantMessageEventStream(), m: any = { role: "assistant", content: [], usage: {}, stopReason: "stop" };
		void go.then(() => {
			s.push({ type: "start", partial: m });
			m.content.push({ type: "thinking", thinking: "" });
			s.push({ type: "thinking_start", contentIndex: 0, partial: m });
			for (const d of ["Let me", " think"]) { m.content[0].thinking += d; s.push({ type: "thinking_delta", contentIndex: 0, delta: d, partial: m }); }
			s.push({ type: "thinking_end", contentIndex: 0, content: "Let me think", partial: m });
			m.content.push({ type: "text", text: "" });
			s.push({ type: "text_start", contentIndex: 1, partial: m });
			for (const d of ["The answer is", " forty-two", "."]) { m.content[1].text += d; s.push({ type: "text_delta", contentIndex: 1, delta: d, partial: m }); }
			s.push({ type: "text_end", contentIndex: 1, content: m.content[1].text, partial: m });
			s.push({ type: "done", reason: "stop", message: m });
			s.end(m);
		});
		return s;
	};
	const model: any = { api: "openai-completions", provider: "metalscope-ds4", id: "m", reasoning: true };
	const stream = keptFirst(live, inner as any);
	const same: any = {}, asIs = keptFirst(live, (() => same) as any);
	assert.equal(asIs(model, {} as any, {}), same);   // no edit: Pi's stream as is
	live.edit = { rid: "r", at: 8, steer: "remove~forty/18-44", model: "m", word: "forty", running: true, sent: true };
	const out = stream(model, {} as any, { reasoning: "medium" } as any), seen: string[] = [];
	const done = (async () => { for await (const ev of out as any) seen.push(ev.type + (ev.delta !== undefined ? `:${ev.delta}` : "")); })();
	await new Promise((r) => setTimeout(r, 5));
	assert.deepEqual(seen, ["start", "thinking_start", "thinking_delta:Let me think", "text_start", "text_delta:The answer is"]);   // before the server says anything
	release();
	await done;
	assert.deepEqual(seen.slice(5), ["thinking_end", "text_delta: forty-two", "text_delta:.", "text_end", "done"]);   // its copy of the kept part not shown again
	const msg: any = await (out as any).result();
	assert.deepEqual(msg.content.map((c: any) => c.thinking ?? c.text), ["Let me think", "The answer is forty-two."]);
	live.edit.over = true;
	assert.equal(asIs(model, {} as any, {}), same);   // its answer over: as is
});

test("keptFirst: when the server's copy may differ, Pi's stream from the start; when it does differ, the server's from there, never taking text back", async () => {
	const { createAssistantMessageEventStream } = await import("@earendil-works/pi-ai/compat");
	const run = async (texts: string[], reasoning: boolean, server: [string, string[]][]) => {
		const live: Live = { base: "http://h", clientId: "me", level: "on", connected: true };
		onEvent(live, { type: "start", rid: "r", model: "m" });
		texts.forEach((text, i) => onEvent(live, { type: "token", rid: "r", i, id: i, text, p: 0.9, entropy: 0, alts: [] }));
		live.edit = { rid: "r", at: texts.length - 1, steer: "remove~x/18-44", model: "m", word: "x", running: true, sent: true };
		const m: any = { role: "assistant", content: [], usage: {}, stopReason: "stop" }, src = createAssistantMessageEventStream();
		const out = keptFirst(live, (() => src) as any)({ api: "openai-completions", provider: "p", id: "m", reasoning: true } as any, {} as any, (reasoning ? { reasoning: "medium" } : {}) as any);
		const sizes: number[] = [], size = (p: any) => p.content.reduce((n: number, c: any) => n + (c.text ?? c.thinking ?? "").length, 0);
		const done = (async () => { for await (const ev of out as any) if (ev.partial) sizes.push(size(ev.partial)); })();
		src.push({ type: "start", partial: m });
		for (const [kind, deltas] of server) {
			const at = m.content.push(kind === "text" ? { type: "text", text: "" } : { type: "thinking", thinking: "" }) - 1, f = kind as "text" | "thinking";
			src.push({ type: `${kind}_start`, contentIndex: at, partial: m } as any);
			for (const d of deltas) { m.content[at][f] += d; src.push({ type: `${kind}_delta`, contentIndex: at, delta: d, partial: m } as any); }
			src.push({ type: `${kind}_end`, contentIndex: at, content: m.content[at][f], partial: m } as any);
		}
		src.push({ type: "done", reason: "stop", message: m });
		src.end(m);
		await done;
		const msg: any = await (out as any).result();
		return { asIs: (out as any) === src, sizes, shown: msg.content.map((c: any) => c.thinking ?? c.text) };
	};
	const thinks = ["Hm", "</think>", "\n\n", "The", " end", " x"];
	for (const [texts, reasoning] of [[["A", "�", " x"], false], [["<think>", "Hm", "</think>", "A", " x"], true], [thinks, false], [["A", " b", " x"], true]] as const)
		assert.equal((await run([...texts], reasoning, [["text", ["A b y"]]])).asIs, true, texts.join(""));   // broken byte, <think> token, Pi's flag unlike the reply
	const okay = await run(thinks, true, [["thinking", ["Hm"]], ["text", ["The end", " y"]]]);
	assert.equal(okay.asIs, false);
	assert.deepEqual(okay.shown, ["Hm", "The end y"]);
	for (const server of [[["thinking", ["Hm, well"]], ["text", ["The end y"]]], [["thinking", ["Hm"]], ["text", ["A", "nother end, y"]]], [["thinking", ["Hm"]], ["thinking", ["more"]], ["text", ["The end"]]]] as [string, string[]][][]) {
		const r = await run(thinks, true, server);
		assert.ok(r.sizes.every((n, i) => !i || n >= r.sizes[i - 1]), JSON.stringify(r.sizes));   // nothing taken back
		assert.deepEqual(r.shown, server.map(([, d]) => d.join("")));   // the server's message in the end
	}
	const late = await run(["A", " b", " x"], false, [["text", ["A b"]], ["thinking", ["late"]]]);   // a thinking block after the kept text
	assert.deepEqual(late.shown, ["A b", "late"]);
	assert.ok(late.sizes.every((n, i) => !i || n >= late.sizes[i - 1]));
});

test("keptText: an edit's kept part as the server has it, though the dock draws its control characters cleaned", () => {
	const live: Live = { base: "http://h", clientId: "c", level: "on", connected: true };
	onEvent(live, { type: "start", rid: "r" });
	["Hi\r", "\x1b[1m there", " x"].forEach((text, i) => onEvent(live, { type: "token", rid: "r", i, id: i, text, p: 0.9, entropy: 0, alts: [], n_layers: null }));
	assert.equal(live.reply!.tokens[0].text, "Hi ");
	live.edit = { rid: "r", at: 2, steer: "-x", model: "m", word: "x", running: true };
	assert.deepEqual(keptText(live, false), { thinking: "", text: "Hi\r\x1b[1m there" });
});

test("requestFailed: no answer in time says the server is up and may still be working, not that it is down", () => {
	const ctx = { model: { provider: "metalscope-omlx", id: "Q" } };
	const timeout = new DOMException("The operation was aborted due to timeout", "TimeoutError");
	assert.equal(requestFailed(timeout, "http://127.0.0.1:8008", ctx), "oMLX at 127.0.0.1:8008: no answer in 20 s (it may still be working: try again in a moment)");
	assert.match(requestFailed(new TypeError("fetch failed"), "http://127.0.0.1:8008", ctx), /^oMLX is not answering at 127\.0\.0\.1:8008: /);
});

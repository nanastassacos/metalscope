import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { clean } from "../draw.ts";
import { type FollowOpts, type Live, type TokenEvent, MAP_KEPT, MAX_LENS_TOP, MAX_ROUTED, OVERLAP_MS, STALL_MS, follow, liveUrl, makeFollower, mapPlace, onEvent, shownReply, switchServer } from "../state.ts";

const tok = (i: number, text: string): TokenEvent => ({ i, id: i, text, p: 0.9, entropy: 0.1, alts: [], n_layers: null });

test("onEvent: follows a reply from start to end, trusting the end's count", () => {
	const live: Live = { base: "", clientId: "c", level: "on", connected: true };
	let renders = 0;
	live.requestRender = () => renders++;
	onEvent(live, { type: "start", rid: "a", level: "on" });
	onEvent(live, { type: "prefill", rid: "a", read: 7 });
	onEvent(live, { type: "token", rid: "a", ...tok(0, "Hi") });
	onEvent(live, { type: "token", rid: "a", ...tok(1, "!") });
	assert.equal(live.reply!.prefill, 7);
	assert.equal(live.reply!.tokens.length, 2);
	onEvent(live, { type: "token", rid: "b", ...tok(0, "x") });
	assert.equal(live.reply!.rid, "b"); // a token of another reply adopts it (late reconnect)
	assert.equal(live.reply!.tokens.length, 1);
	onEvent(live, { type: "start", rid: "a2", level: "on" });
	assert.equal(live.reply!.rid, "a2");
	onEvent(live, { type: "token", rid: "a2", ...tok(0, "One") });
	onEvent(live, { type: "token", rid: "a2", ...tok(1, " two") });
	onEvent(live, { type: "end", rid: "a2", n_tokens: 1, trace_id: "t1" });
	assert.equal(live.reply!.done, true);
	assert.equal(live.reply!.traceId, "t1");
	assert.deepEqual(live.reply!.tokens.map((t) => t.text), ["One"]);
	assert.equal(renders, 9);
});

const newLive = (): Live => ({ base: "http://o:1", clientId: "pi-x", level: "on", connected: true });

test("onEvent: an edit's replies continue the one edited: its tokens before the edit as they were, the edited token, the rest after it", () => {
	const live = newLive();
	onEvent(live, { type: "start", rid: "r" });
	["A", " b", " c", " d", " e"].forEach((text, i) => onEvent(live, { type: "token", rid: "r", ...tok(i, text) }));
	onEvent(live, { type: "end", rid: "r", n_tokens: 6, trace_id: "T0" });                // a stop token past the last
	const r = live.reply!;
	live.edit = { rid: "r", at: 2, steer: "remove~c/18-44", model: "m", word: "c", sent: true };
	// The edited token (its own token 0 is the reply's 2), then the rest from 3.
	onEvent(live, { type: "start", rid: "a", edit: { of: "r", at: 2, steer: "remove~c/18-44" } });
	assert.deepEqual(live.reply!.tokens, r.tokens.slice(0, 2));                             // the very tokens, unchanged
	assert.equal(live.edit.answered, true);                                                // kept: a retry's reply takes it too
	assert.deepEqual(live.reply!.edit!.was, { id: 2, text: " c" });
	onEvent(live, { type: "token", rid: "a", ...tok(0, " C") });
	onEvent(live, { type: "end", rid: "a", n_tokens: 1, trace_id: "TA" });
	const a = live.reply!;
	onEvent(live, { type: "start", rid: "b", edit: { of: "a", at: 3 } });
	onEvent(live, { type: "token", rid: "b", ...tok(0, " x") });
	onEvent(live, { type: "token", rid: "b", ...tok(0, " x") });                            // the other stream's copy
	onEvent(live, { type: "token", rid: "b", ...tok(1, " y") });
	onEvent(live, { type: "token", rid: "b", ...tok(2, " past the stop") });
	onEvent(live, { type: "end", rid: "b", n_tokens: 2, trace_id: "TB" });
	const b = live.reply!;
	assert.deepEqual(b.tokens.map((t) => [t.i, t.text]), [[0, "A"], [1, " b"], [2, " C"], [3, " x"], [4, " y"]]);
	assert.equal(b.edit, a.edit);                                                          // said once, on whichever is shown
	assert.deepEqual([b.traceId, live.prev, r.tokens.length], ["TB", a, 5]);               // the edited reply left as it was
	// The replies its tokens came from: the edited reply's, then A's.
	assert.deepEqual(b.parts, [{ from: 0, rid: "r" }, { from: 2, rid: "a" }]);
	// An edit of that joined reply at its token 4 (B's 1): kept tokens from all three, then its own.
	onEvent(live, { type: "start", rid: "c2", edit: { of: "b", at: 4 } });
	onEvent(live, { type: "token", rid: "c2", ...tok(0, " Y") });
	assert.deepEqual(live.reply!.parts, [{ from: 0, rid: "r" }, { from: 2, rid: "a" }, { from: 3, rid: "b" }]);
	onEvent(live, { type: "end", rid: "c2", n_tokens: 1, trace_id: "TC" });
	// An edit of a reply this Pi does not hold: a reply of its own, nothing kept.
	onEvent(live, { type: "start", rid: "c", edit: { of: "gone", at: 4 } });
	onEvent(live, { type: "token", rid: "c", ...tok(0, "z") });
	assert.deepEqual(live.reply!.tokens.map((t) => t.i), [0]);
});

test("onEvent: Pi's retry of an edit whose rest failed continues the edited reply again, though two replies came since", () => {
	const live = newLive();
	onEvent(live, { type: "start", rid: "r" });
	["A", " b", " c"].forEach((text, i) => onEvent(live, { type: "token", rid: "r", ...tok(i, text) }));
	onEvent(live, { type: "end", rid: "r", n_tokens: 3, trace_id: "T0" });
	live.edit = { rid: "r", at: 2, steer: "remove~c/18-44", model: "m", word: "c", sent: true };
	onEvent(live, { type: "start", rid: "a1", edit: { of: "r", at: 2 } });
	onEvent(live, { type: "token", rid: "a1", ...tok(0, " C") });
	onEvent(live, { type: "end", rid: "a1", n_tokens: 1, trace_id: "TA1" });
	onEvent(live, { type: "start", rid: "b1", edit: { of: "a1", at: 3 } });
	onEvent(live, { type: "token", rid: "b1", ...tok(0, " x") });
	onEvent(live, { type: "end", rid: "b1", n_tokens: 1, trace_id: "TB1" });           // its stream failed; Pi retries
	onEvent(live, { type: "start", rid: "a2", edit: { of: "r", at: 2 } });
	assert.deepEqual(live.reply!.tokens.map((t) => t.text), ["A", " b"]);
	assert.deepEqual([live.reply!.edit?.rid, live.reply!.edit?.was], ["r", { id: 2, text: " c" }]);
	assert.deepEqual(live.reply!.parts, [{ from: 0, rid: "r" }]);
});

test("Live.map: each reply as it ends, a byte a token, in order; an edit's reply in the edited one's place, however often retried", () => {
	const live = newLive(), rids = () => live.map?.map((m) => m.rid + (m.replaced ? `<${m.replaced.join("<")}` : "")).join(" ");
	const say = (rid: string, texts: string[], end = texts.length, edit?: { of: string; at: number }, model = "m") => {
		onEvent(live, { type: "start", rid, model, ...(edit ? { edit } : {}) });
		texts.forEach((text, i) => onEvent(live, { type: "token", rid, ...tok(i, text), p: i ? 0.95 : 0.3 }));
		onEvent(live, { type: "end", rid, n_tokens: end, trace_id: `T${rid}` });
	};
	say("r", ["<think>", " hm", "</think>", "A", " b", " c"], 7);                       // a stop token past the last
	assert.deepEqual([rids(), [...live.map![0].p], live.map![0].think, live.map![0].model], ["r", [60, 190, 190, 190, 190, 190], 2, "m"]);
	say("s", [], 0);                                                                       // no tokens: nothing to draw
	onEvent(live, { type: "end", rid: "r", n_tokens: 6, trace_id: "Tr" });                  // a late copy of r's end: once
	assert.equal(rids(), "r");
	// Edited at token 4: drawn nowhere before a token of its own (r stays), then in r's place (mapPlace), and kept there.
	live.edit = { rid: "r", at: 4, steer: "remove~b/18-44", model: "m", word: "b", sent: true };
	onEvent(live, { type: "start", rid: "a", model: "m", edit: { of: "r", at: 4 } });
	assert.deepEqual(mapPlace(live.map!, live.reply), { at: -1, replace: false });
	onEvent(live, { type: "token", rid: "a", ...tok(0, " B") });
	assert.deepEqual(mapPlace(live.map!, live.reply), { at: 0, replace: true });
	onEvent(live, { type: "end", rid: "a", n_tokens: 1, trace_id: "Ta" });
	say("b", [" x"], 1, { of: "a", at: 5 });                                              // the rest of the edit
	assert.equal(rids(), "b<r<a");
	say("a2", [" B"], 1, { of: "r", at: 4 });                                             // Pi retried the edit: still one
	assert.equal(rids(), "a2<r<a<b");
	onEvent(live, { type: "end", rid: "b", n_tokens: 1, trace_id: "Tb" });                  // a late copy of an end replaced since
	assert.equal(rids(), "a2<r<a<b");
	// An edit that ended before a token of its own (esc at once, or its request failed): the reply it edits kept whole.
	say("z", [], 0, { of: "a2", at: 2 });
	assert.deepEqual([rids(), live.map![0].p.length], ["a2<r<a<b", 5]);
	// A new reply after it, then another model's: kept after it in order.
	say("n", ["ok"]);
	say("o", ["hi"], 1, undefined, "other");
	assert.equal(rids(), "a2<r<a<b n o");
	switchServer(live, "http://d:2", { running: false, restart() {} });                   // a model on the other server: kept
	assert.equal(live.map?.length, 3);
});

test("Live.map: ds4's edit (the edited token's reply, then the rest's) kept once whichever end comes first", () => {
	for (const order of ["a's end first", "b's end first"]) {
		const live = newLive(), rids = () => live.map?.map((m) => m.rid + (m.replaced ? `<${m.replaced.join("<")}` : "")).join(" ");
		onEvent(live, { type: "start", rid: "r", model: "m" });
		["A", " b", " c", " d", " e"].forEach((text, i) => onEvent(live, { type: "token", rid: "r", ...tok(i, text) }));
		onEvent(live, { type: "end", rid: "r", n_tokens: 5, trace_id: "Tr" });
		live.edit = { rid: "r", at: 4, steer: "remove~e/18-44", model: "m", word: "e", sent: true };
		onEvent(live, { type: "start", rid: "a", model: "m", edit: { of: "r", at: 4 } });
		onEvent(live, { type: "token", rid: "a", ...tok(0, " E") });
		onEvent(live, { type: "start", rid: "b", model: "m", edit: { of: "a", at: 5 } });   // the rest starts before a's end
		onEvent(live, { type: "token", rid: "b", ...tok(0, " f") });
		assert.deepEqual(mapPlace(live.map!, live.reply), { at: 0, replace: true }, `${order}: drawn in r's place`);
		const ends = [() => onEvent(live, { type: "end", rid: "a", n_tokens: 1, trace_id: "Ta" }), () => onEvent(live, { type: "end", rid: "b", n_tokens: 1, trace_id: "Tb" })];
		if (order === "b's end first") ends.reverse();
		for (const end of ends) end();
		assert.deepEqual([rids(), live.map![0].p.length], ["b<r<a", 6], order);
		ends[0]();                                                                             // either end again: nothing changes
		assert.equal(rids(), "b<r<a", order);
	}
});

test("Live.map: at most MAP_KEPT tokens, the oldest replies dropped whole (the newest always kept)", () => {
	const live = newLive();
	const old = (rid: string, n: number) => ({ rid, p: new Uint8Array(n), think: -1 });
	const say = (rid: string) => {
		onEvent(live, { type: "start", rid });
		for (let i = 0; i < 5; i++) onEvent(live, { type: "token", rid, ...tok(i, "x") });
		onEvent(live, { type: "end", rid, n_tokens: 5, trace_id: `T${rid}` });
	};
	live.map = [old("a", 3), old("b", MAP_KEPT - 8)];
	say("c");
	assert.equal(live.map.map((m) => m.rid).join(), "a,b,c", "MAP_KEPT exactly: all kept");
	say("d");
	assert.equal(live.map.map((m) => m.rid).join(), "c,d", "past it: a, then b");
	say("e");
	assert.equal(live.map.map((m) => m.rid).join(), "c,d,e");
	live.map = [old("huge", MAP_KEPT + 1)];
	say("f");
	assert.equal(live.map.map((m) => m.rid).join(), "f");
});

test("shownReply: the last reply stays shown while the next reads its prompt, until the next one's first token", () => {
	const live = newLive();
	const token = (rid: string, i: number, text: string) => onEvent(live, { type: "token", rid, i, id: i, text, p: 0.9, entropy: 0.1, alts: [] });
	assert.equal(shownReply(live), undefined);
	onEvent(live, { type: "start", rid: "a" });
	assert.equal(shownReply(live)?.rid, "a");                                 // nothing before it: the new one, empty
	token("a", 0, "Hi");
	onEvent(live, { type: "end", rid: "a", n_tokens: 1, trace_id: "ta" });
	onEvent(live, { type: "start", rid: "b" });                               // a tool call's result: the next request
	onEvent(live, { type: "prefill", rid: "b", read: 900 });
	assert.equal(live.reply?.rid, "b");
	assert.equal(shownReply(live)?.rid, "a");
	assert.equal(shownReply(live)?.traceId, "ta");
	token("b", 0, "Done");
	assert.equal(shownReply(live)?.rid, "b");
	onEvent(live, { type: "end", rid: "b", n_tokens: 1 });
	onEvent(live, { type: "start", rid: "c" });
	onEvent(live, { type: "end", rid: "c", n_tokens: 0 });                    // wrote nothing: b stays
	onEvent(live, { type: "start", rid: "d" });
	assert.equal(shownReply(live)?.rid, "b");
	// The shown reply's end arriving late, after the next one started: it completes the shown one.
	onEvent(live, { type: "start", rid: "e" });
	token("e", 0, "More");
	onEvent(live, { type: "start", rid: "f" });                               // e not ended yet
	onEvent(live, { type: "end", rid: "e", n_tokens: 1, trace_id: "te" });
	assert.deepEqual([shownReply(live)?.rid, shownReply(live)?.done, shownReply(live)?.traceId, live.reply?.rid], ["e", true, "te", "f"]);
});

test("makeFollower: one follower at a time; restart aborts the running one, stop ends it", () => {
	const live = newLive();
	const signals: AbortSignal[] = [];
	const f = makeFollower(live, async (_l, signal) => void signals.push(signal));
	assert.equal(f.running, false);
	f.restart();
	assert.equal(live.connected, false);
	f.restart();
	assert.deepEqual(signals.map((s) => s.aborted), [true, false]);
	assert.equal(f.running, true);
	f.stop();
	assert.deepEqual(signals.map((s) => s.aborted), [true, true]);
	assert.equal(f.running, false);
});

test("switchServer: moves the stream and drops the old server's reply and steer labels; same server: nothing", () => {
	const live = newLive();
	let restarts = 0, renders = 0;
	live.requestRender = () => void renders++;
	const follower = { running: true, restart: () => void restarts++ };
	live.reply = { rid: "r", tokens: [], done: false, prefill: 0, traceId: "t-old" };
	live.prev = { rid: "q", tokens: [], done: true, prefill: 0 };
	live.sentSteer = "angry@0.3";
	live.requestSteer = "angry@0.3";
	assert.equal(switchServer(live, "http://o:1", follower), false);
	assert.equal(live.reply?.traceId, "t-old");
	assert.equal(restarts, 0);
	assert.equal(switchServer(live, "http://d:2", follower), true);
	assert.deepEqual([live.base, live.reply, live.prev, live.sentSteer, live.requestSteer, restarts, renders],
		["http://d:2", undefined, undefined, undefined, undefined, 1, 1]);
	assert.equal(switchServer(live, "http://o:1", { running: false, restart: () => void restarts++ }), true);   // not following yet
	assert.equal(restarts, 1);
});

test("follow: an aborted follower leaves live alone", async () => {
	const live = newLive();
	const saved = globalThis.fetch;
	const abort = new AbortController();
	globalThis.fetch = (async () => {
		abort.abort();                                    // replaced while connecting
		throw new Error("aborted");
	}) as any;
	try {
		await follow(live, abort.signal);
		assert.equal(live.connected, true);               // the new follower's to set, not this one's
	} finally {
		globalThis.fetch = saved;
	}
});

test("follow: the server's pings keep a quiet stream open; silent after them for STALL_MS, it is opened again", async () => {
	const live = newLive();
	const saved = globalThis.fetch;
	const abort = new AbortController();
	const flush = () => new Promise((r) => setImmediate(r));
	let calls = 0, send: ((s: string) => void) | undefined;
	globalThis.fetch = (async (_url: string, init: { signal: AbortSignal }) => {
		if (++calls > 1) {
			abort.abort();
			throw new Error("aborted");
		}
		const reader = { read: () => new Promise((ok, fail) => {
			send = (s) => ok({ value: new TextEncoder().encode(s), done: false });
			init.signal.addEventListener("abort", () => fail(new Error("aborted")));
		}) };
		return { ok: true, status: 200, body: { getReader: () => reader } };
	}) as any;
	mock.timers.enable({ apis: ["setTimeout"] });
	try {
		const done = follow(live, abort.signal);
		await flush();
		for (let ping = 0; ping < 8; ping++) {                // two minutes of pings, no tokens
			mock.timers.tick(15_000);
			send!(": ping\n\n");
			await flush();
		}
		assert.deepEqual([calls, live.connected], [1, true]);
		mock.timers.tick(STALL_MS);
		await flush();
		assert.equal(live.connected, false);
		mock.timers.tick(1000);
		await done;
		assert.equal(calls, 2);
	} finally {
		mock.timers.reset();
		globalThis.fetch = saved;
	}
});

test("follow: a stream silent for STALL_MS (the server pings every 15 s) is dropped and opened again", async () => {
	const live = newLive();
	const saved = globalThis.fetch;
	const abort = new AbortController();
	const flush = () => new Promise((r) => setImmediate(r));
	let calls = 0;
	globalThis.fetch = (async (_url: string, init: { signal: AbortSignal }) => {
		if (++calls > 1) {
			abort.abort();
			throw new Error("aborted");
		}
		let sent = false;
		const reader = { read: () => sent
			? new Promise((_, fail) => init.signal.addEventListener("abort", () => fail(new Error("aborted"))))   // a half-open socket
			: (sent = true, Promise.resolve({ value: new TextEncoder().encode(": connected\n\n"), done: false })) };
		return { ok: true, status: 200, body: { getReader: () => reader } };
	}) as any;
	mock.timers.enable({ apis: ["setTimeout"] });
	try {
		const done = follow(live, abort.signal);
		await flush();
		assert.deepEqual([calls, live.connected], [1, true]);
		mock.timers.tick(STALL_MS - 1);
		await flush();
		assert.deepEqual([calls, live.connected], [1, true]);
		mock.timers.tick(1);
		await flush();
		assert.equal(live.connected, false);                  // silent too long: down, reconnecting
		mock.timers.tick(1000);
		await done;
		assert.equal(calls, 2);
	} finally {
		mock.timers.reset();
		globalThis.fetch = saved;
	}
});

test("clean: what Python strips as whitespace a space, every other control character but newline and tab ·; the model's text cleaned once as it comes, `raw` kept where that changed it", () => {
	assert.equal(clean("a\x00\x07\x1b]52;c;aGk=\x07\x7f\x80\x9b2J\x9fz\n\t é 读 🙂"), "a···]52;c;aGk=····2J·z\n\t é 读 🙂");
	assert.equal(clean(";\x0b\x0c\r\x1c\x1d\x1e\x1f\x85;"), ";        ;");
	assert.equal(clean(";\r\n").trim(), ";", "trimmed as the server strips it");
	const live = newLive();
	const osc = "\x1b]8;;http://x\x07";
	onEvent(live, { type: "start", rid: "a" });
	onEvent(live, { type: "token", rid: "a", ...tok(0, `${osc}hi`), alts: [[`\x9b2J${osc}`, 0.2]], lens: [[3, 7, osc]], lens_logit: [[3, 8, "\x1b[2J"]],
		lens10: { l: [3], k: 1, i: [7], w: [osc], c: [0], p: [10] } } as any);
	const t = live.reply!.tokens[0];
	assert.deepEqual([t.text, t.alts[0][0], t.lens![0][2], t.lens_logit![0][2], t.top!.w[0]],
		["·]8;;http://x·hi", "·2J·]8;;http://x·", "·]8;;http://x·", "·[2J", "·]8;;http://x·"]);
	assert.equal(t.raw, `${osc}hi`);
	onEvent(live, { type: "token", rid: "a", ...tok(1, " plain") } as any);
	assert.ok(!("raw" in live.reply!.tokens[1]) && !("lens" in live.reply!.tokens[1]), "nothing added where nothing changed");
});

test("liveUrl: expert routing only while asked for", () => {
	const live = newLive();
	assert.equal(liveUrl(live), "http://o:1/scope/api/live?client=pi-x");
	live.experts = true;
	assert.equal(liveUrl(live), "http://o:1/scope/api/live?client=pi-x&experts=true");
	live.experts = false;
	live.lens = true;
	assert.equal(liveUrl(live), "http://o:1/scope/api/live?client=pi-x&lens=true");
});

test("onEvent: the lens top 3s are kept on the newest MAX_LENS_TOP tokens only", () => {
	const live = newLive();
	onEvent(live, { type: "start", rid: "r" });
	const lens10 = { l: [4], k: 3, i: [1, 2, 3], w: ["a", "b", "c"], c: [0, 1, 2], p: [10, 20, 30] };
	for (let i = 0; i < MAX_LENS_TOP + 1; i++) onEvent(live, { type: "token", rid: "r", i, id: 2, text: "x", p: 1, entropy: 0, alts: [], lens10 });
	const toks = live.reply!.tokens;
	assert.equal(toks[0].top, undefined);
	assert.deepEqual([...toks[1].top!.id], [1, 2, 3]);
	assert.equal(toks[1].top!.wp[0], 20);                                        // the written token (2) was second
	assert.equal(toks.filter((t) => t.top).length, MAX_LENS_TOP);
});

test("onEvent: a reply that streamed with routing asked for and none came marks its model as without experts", () => {
	const live = newLive();
	live.experts = true;
	onEvent(live, { type: "start", rid: "r", model: "Dense" });
	for (const i of [0, 1]) onEvent(live, { type: "token", rid: "r", i, id: 0, text: "x", p: 1, entropy: 0, alts: [] });
	onEvent(live, { type: "end", rid: "r", n_tokens: 2 });
	assert.equal(live.dense, "Dense");
	onEvent(live, { type: "start", rid: "s", model: "Dense" });
	onEvent(live, { type: "token", rid: "s", i: 0, id: 0, text: "x", p: 1, entropy: 0, alts: [], experts: [1, 2], expert_w: [128, 127], expert_k: 2 });
	onEvent(live, { type: "end", rid: "s", n_tokens: 1 });
	assert.equal(live.dense, undefined);                                         // it routes after all
});

test("makeFollower: resubscribe opens the new stream first, in use once it answers (the old one kept a moment); one waits at a time", () => {
	const live = newLive();
	const runs: { signal: AbortSignal; opts?: FollowOpts }[] = [];
	const f = makeFollower(live, async (_l, signal, opts) => void runs.push({ signal, opts }));
	f.resubscribe();                                     // not following: nothing
	assert.equal(runs.length, 0);
	f.restart();
	live.connected = true;
	f.resubscribe();
	assert.deepEqual(runs.map((r) => r.signal.aborted), [false, false]);   // both open meanwhile
	assert.equal(runs[1].opts!.pending!(), true);
	runs[1].opts!.onOpen!();
	assert.deepEqual(runs.map((r) => r.signal.aborted), [false, false]);   // the old one kept until both deliver one event
	assert.equal(runs[1].opts!.pending!(), false);       // now the one in use
	f.resubscribe();
	f.resubscribe();                                     // a newer one replaces the one still waiting
	assert.deepEqual(runs.map((r) => r.signal.aborted), [false, false, true, false]);
	runs[2].opts!.onOpen!();                             // the replaced one answering late changes nothing
	assert.deepEqual(runs.map((r) => r.signal.aborted), [false, false, true, false]);
	runs[3].opts!.onOpen!();                             // the one it replaces kept too, each until it caught up
	assert.deepEqual(runs.map((r) => r.signal.aborted), [false, false, true, false]);
	for (let k = 0; k < 3; k++) f.resubscribe();         // server down, three flips: one waiting at most (and the ones kept)
	assert.equal(runs.filter((r) => !r.signal.aborted).length, 4);
	f.stop();                                            // stop closes the one in use and the one waiting
	assert.ok(runs.every((r) => r.signal.aborted));
});

test("makeFollower: a token the old stream still holds as the new one opens is not lost (the new one starts past it); the old closes once both delivered one event", () => {
	const live = newLive();
	const runs: { signal: AbortSignal; opts?: FollowOpts }[] = [];
	const f = makeFollower(live, async (_l, signal, opts) => void runs.push({ signal, opts }));
	const deliver = (s: number, i: number) => {
		const ev = { type: "token", rid: "r", i, id: i, text: i === 3 ? "–" : ` t${i}`, p: 0.9, entropy: 0, alts: [] };
		if (runs[s].signal.aborted) return;
		runs[s].opts?.seen?.(ev);
		onEvent(live, ev);
	};
	f.restart();
	onEvent(live, { type: "start", rid: "r" });
	for (let i = 0; i < 3; i++) deliver(0, i);
	f.resubscribe();                                     // the dock stepped into: lens and routing asked for
	runs[1].opts!.onOpen!();                             // token 3 published before it subscribed: on the old stream only
	deliver(1, 4);
	deliver(0, 3);                                       // the old stream's token, after the new one's next
	assert.equal(runs[0].signal.aborted, false, "not both at one event yet");
	deliver(0, 4);                                       // both delivered 4: the old has given all it held
	assert.equal(runs[0].signal.aborted, true);
	deliver(1, 5);
	assert.deepEqual(live.reply!.tokens.map((t) => t.i), [0, 1, 2, 3, 4, 5]);
	f.stop();
});

test("makeFollower: the old stream closes after OVERLAP_MS when no event comes on both; restart and stop close it", (t) => {
	t.mock.timers.enable({ apis: ["setTimeout"] });
	const live = newLive();
	const runs: { signal: AbortSignal; opts?: FollowOpts }[] = [];
	const f = makeFollower(live, async (_l, signal, opts) => void runs.push({ signal, opts }));
	f.restart();
	f.resubscribe();
	runs[1].opts!.onOpen!();
	assert.equal(runs[1].opts!.pending!(), false);
	assert.equal(runs[0].opts!.pending!(), true, "the old one failing says nothing of the server");
	t.mock.timers.tick(OVERLAP_MS);
	assert.deepEqual(runs.map((r) => r.signal.aborted), [true, false]);
	f.resubscribe();
	runs[2].opts!.onOpen!();
	f.restart();
	assert.deepEqual(runs.map((r) => r.signal.aborted), [true, true, true, false]);
	f.stop();
});

test("onEvent: a token missing between kept ones (from the old stream, late) goes in its place, also once the reply ended; a copy changes nothing", () => {
	const live = newLive();
	const tokenOf = (i: number) => ({ type: "token", rid: "r", i, id: i, text: ` t${i}`, p: 0.9, entropy: 0, alts: [] });
	onEvent(live, { type: "start", rid: "r" });
	for (const i of [0, 1, 3, 4]) onEvent(live, tokenOf(i));
	onEvent(live, tokenOf(5));                                                  // past the stop (pipelined decode)
	onEvent(live, { type: "end", rid: "r", n_tokens: 5, trace_id: "t" });
	assert.deepEqual(live.reply!.tokens.map((t) => t.i), [0, 1, 3, 4]);
	onEvent(live, tokenOf(2));
	onEvent(live, tokenOf(2));
	onEvent(live, tokenOf(5));
	assert.deepEqual(live.reply!.tokens.map((t) => t.i), [0, 1, 2, 3, 4]);
	assert.equal(live.map!.at(-1)!.p.length, 5, "MAP's copy too");
});

test("onEvent: tokens still queued on the old stream after the new one brought the end are kept up to its count; a late token of the reply before goes there", () => {
	const live = newLive();
	const tokenOf = (rid: string, i: number) => ({ type: "token", rid, i, id: i, text: ` t${i}`, p: 0.9, entropy: 0, alts: [] });
	onEvent(live, { type: "start", rid: "r" });
	for (const i of [0, 1, 2]) onEvent(live, tokenOf("r", i));
	onEvent(live, { type: "end", rid: "r", n_tokens: 5, trace_id: "t" });
	for (const i of [3, 4, 5]) onEvent(live, tokenOf("r", i));                 // 5: past the stop
	assert.deepEqual(live.reply!.tokens.map((t) => t.i), [0, 1, 2, 3, 4]);
	assert.equal(live.map!.at(-1)!.p.length, 5);
	// The next reply started; a token of this one still on the old stream goes into it.
	live.reply!.tokens.splice(1, 1);
	onEvent(live, { type: "start", rid: "s" });
	onEvent(live, tokenOf("s", 0));
	onEvent(live, tokenOf("r", 1));
	onEvent(live, tokenOf("r", 1));
	assert.deepEqual([live.prev!.rid, live.prev!.tokens.map((t) => t.i), live.reply!.tokens.map((t) => t.i)], ["r", [0, 1, 2, 3, 4], [0]]);
});

test("onEvent: a short reply whose tokens all come late (the new stream brought its start, its end and the next start first) is kept", () => {
	const live = newLive();
	const tokenOf = (rid: string, i: number) => ({ type: "token", rid, i, id: i, text: ` t${i}`, p: 0.9, entropy: 0, alts: [] });
	onEvent(live, { type: "start", rid: "R1" });
	onEvent(live, tokenOf("R1", 0));
	onEvent(live, { type: "start", rid: "R2" });
	onEvent(live, { type: "end", rid: "R2", n_tokens: 2, trace_id: "t2" });
	onEvent(live, { type: "start", rid: "R3" });
	for (const i of [0, 1]) onEvent(live, tokenOf("R2", i));
	assert.deepEqual([live.prev?.rid, live.prev?.tokens.map((t) => t.i), live.map?.map((m) => m.rid)], ["R2", [0, 1], ["R2"]]);
	// One that ended with none (a reply with no tokens of its own) does not take the place of the one before.
	onEvent(live, { type: "end", rid: "R3", n_tokens: 0, trace_id: "t3" });
	onEvent(live, { type: "start", rid: "R4" });
	assert.equal(live.prev?.rid, "R2");
});

test("makeFollower: resubscribed twice within the overlap, each replaced stream kept until it caught up with a newer one", () => {
	const live = newLive();
	const runs: { signal: AbortSignal; opts?: FollowOpts }[] = [];
	const f = makeFollower(live, async (_l, signal, opts) => void runs.push({ signal, opts }));
	const deliver = (s: number, i: number) => {
		const ev = { type: "token", rid: "r", i, id: i, text: ` t${i}`, p: 0.9, entropy: 0, alts: [] };
		if (runs[s].signal.aborted) return;
		runs[s].opts?.seen?.(ev);
		onEvent(live, ev);
	};
	f.restart();
	onEvent(live, { type: "start", rid: "r" });
	deliver(0, 0);
	f.resubscribe();
	runs[1].opts!.onOpen!();                             // stream 0 still holds 1
	f.resubscribe();                                     // 1-4 pressed quickly
	runs[2].opts!.onOpen!();                             // stream 1 holds 2, stream 0 still 1
	assert.deepEqual(runs.map((r) => r.signal.aborted), [false, false, false]);
	deliver(2, 3);
	deliver(1, 2);
	deliver(1, 3);                                       // stream 1 caught up with stream 2
	assert.deepEqual(runs.map((r) => r.signal.aborted), [false, true, false]);
	deliver(0, 1);
	deliver(0, 2);                                       // stream 0 caught up with stream 1's
	assert.deepEqual(runs.map((r) => r.signal.aborted), [true, true, false]);
	assert.deepEqual(live.reply!.tokens.map((t) => t.i), [0, 1, 2, 3]);
	f.stop();
});

test("follow: a stream still waiting to replace the one in use does not mark the server down when it fails", async () => {
	const live = newLive();
	const saved = globalThis.fetch;
	const abort = new AbortController();
	let calls = 0;
	globalThis.fetch = (async () => {
		if (++calls > 1) abort.abort();
		throw new Error("down");
	}) as any;
	try {
		await follow(live, abort.signal, { pending: () => true });
		assert.equal(live.connected, true);
	} finally {
		globalThis.fetch = saved;
	}
});

test("onEvent: two streams overlapping across a reply's end: the ended reply is not adopted again", () => {
	const live = newLive();
	const tokenOf = (rid: string, i: number) => ({ type: "token", rid, i, id: i, text: ` t${i}`, p: 0.9, entropy: 0.1, alts: [] });
	onEvent(live, { type: "start", rid: "A" });
	for (let i = 0; i < 50; i++) onEvent(live, tokenOf("A", i));
	// The old stream delivers A's last token, its end, B's start and a prompt chunk; the new stream then repeats them.
	const tail = [tokenOf("A", 50), { type: "end", rid: "A", n_tokens: 51, trace_id: "tA" }, { type: "start", rid: "B" },
		{ type: "prefill", rid: "B", read: 512 }];
	for (const ev of [...tail, ...tail]) onEvent(live, structuredClone(ev));
	assert.equal(shownReply(live)?.rid, "A");
	assert.equal(shownReply(live)?.tokens.length, 51);
	assert.equal(live.prev?.rid, "A");
	assert.deepEqual([live.reply?.rid, live.reply?.prefill], ["B", 512]);   // the repeated chunk not counted twice
	onEvent(live, { type: "prefill", rid: "B", read: 612 });
	onEvent(live, { type: "prefill", rid: "B", read: 600 });                    // a late copy: the most read stands
	assert.equal(live.reply?.prefill, 612);
	onEvent(live, tokenOf("B", 0));
	assert.deepEqual([shownReply(live)?.rid, live.prev?.rid], ["B", "A"]);
	onEvent(live, tokenOf("A", 50));                                           // a late copy of the old reply: ignored
	assert.equal(live.reply?.rid, "B");
});

test("onEvent: a repeated start of the reply being followed keeps the previous reply", () => {
	const live = newLive();
	onEvent(live, { type: "start", rid: "A" });
	onEvent(live, { type: "token", rid: "A", i: 0, id: 0, text: "a", p: 1, entropy: 0, alts: [] });
	onEvent(live, { type: "end", rid: "A", n_tokens: 1 });
	onEvent(live, { type: "start", rid: "B" });
	onEvent(live, { type: "token", rid: "B", i: 0, id: 0, text: "b", p: 1, entropy: 0, alts: [] });
	onEvent(live, { type: "start", rid: "B" });                                // the other stream's copy
	assert.deepEqual([live.prev?.rid, live.reply?.rid, live.reply?.tokens.length], ["A", "B", 1]);
});

test("packRouting: routing with an id or weight out of range is dropped, the token kept", () => {
	const live = newLive();
	onEvent(live, { type: "start", rid: "r" });
	const tok = (i: number, experts: any[], w: any[]) => onEvent(live, { type: "token", rid: "r", i, id: i, text: "x", p: 1, entropy: 0, alts: [], experts, expert_w: w, expert_k: 2 });
	tok(0, [-1, 3], [100, 155]);
	tok(1, [70000, 3], [100, 155]);
	tok(2, [1.5, 3], [100, 155]);
	tok(3, ["7", 3], [100, 155]);
	tok(4, [1, 3], [300, 155]);
	tok(5, [1, 3], [-1, 155]);
	tok(6, [1, 3], [100, 155]);
	assert.equal(live.reply!.tokens.length, 7);
	assert.deepEqual(live.reply!.tokens.map((t) => !!t.experts), [false, false, false, false, false, false, true]);
	assert.equal(live.reply!.maxExpert, 3);
});

test("onEvent: routing kept as typed arrays on the newest MAX_ROUTED tokens; the other stream's copies dropped", () => {
	const live = newLive();
	onEvent(live, { type: "start", rid: "a", model: "Flash" });
	onEvent(live, { type: "start", rid: "a", model: "Flash" });                     // the same start twice: one reply
	const token = (i: number, extra: object = {}) => onEvent(live, { type: "token", rid: "a", i, id: i, text: "x", p: 0.9, entropy: 0.1, alts: [], ...extra });
	token(0, { experts: [3, 700, 5, 6], expert_w: [200, 55, 255, 0], expert_k: 2 });
	const t0 = live.reply!.tokens[0];
	assert.ok(t0.experts instanceof Uint16Array && t0.expert_w instanceof Uint8Array && t0.expert_k === 2);
	assert.deepEqual([...t0.experts!], [3, 700, 5, 6]);
	assert.equal(live.reply!.model, "Flash");
	assert.equal(live.reply!.routed, true);
	assert.equal(live.reply!.maxExpert, 700);
	token(0, { text: "dup" });                                                     // already here
	assert.equal(live.reply!.tokens.length, 1);
	token(1, { experts: [1, 2, 3], expert_w: [1, 2], expert_k: 2 });               // malformed routing: dropped, token kept
	assert.equal(live.reply!.tokens[1].experts, undefined);
	for (let i = 2; i < MAX_ROUTED + 2; i++) token(i, { experts: [1, 2], expert_w: [128, 127], expert_k: 2 });
	assert.equal(live.reply!.tokens[0].experts, undefined);                        // past the bound: let go
	assert.ok(live.reply!.tokens[2].experts);
	assert.equal(live.reply!.tokens.filter((t) => t.experts).length, MAX_ROUTED);
});

test("onEvent: a late copy of a token past the stop does not grow a reply that ended", () => {
	const live = newLive();
	const tk = (i: number) => ({ type: "token", rid: "r", i, id: i, text: "x", p: 1, entropy: 0, alts: [] });
	onEvent(live, { type: "start", rid: "r" });
	for (let i = 0; i < 4; i++) onEvent(live, tk(i));
	onEvent(live, { type: "end", rid: "r", n_tokens: 3 });                      // the stop: token 3 was past it
	onEvent(live, tk(3));                                                        // the other stream's copy
	onEvent(live, tk(4));
	assert.equal(live.reply!.tokens.length, 3);
});

test("onEvent: a reply whose tokens came through a stream without routing says nothing of the model", () => {
	const live = newLive();
	live.experts = true;
	const asked = { experts: true }, not = { experts: false };
	onEvent(live, { type: "start", rid: "r", model: "Moe" }, asked);
	onEvent(live, { type: "token", rid: "r", i: 0, id: 0, text: "x", p: 1, entropy: 0, alts: [] }, not);   // the old stream
	onEvent(live, { type: "end", rid: "r", n_tokens: 1 }, asked);
	assert.equal(live.dense, undefined);
	onEvent(live, { type: "start", rid: "s", model: "Moe" }, not);              // started before routing was asked for
	onEvent(live, { type: "token", rid: "s", i: 0, id: 0, text: "x", p: 1, entropy: 0, alts: [] }, asked);
	onEvent(live, { type: "end", rid: "s", n_tokens: 1 }, asked);
	assert.equal(live.dense, undefined);
	onEvent(live, { type: "start", rid: "t", model: "Dense" }, asked);           // every event through the asking stream
	for (const i of [0, 1]) onEvent(live, { type: "token", rid: "t", i, id: 0, text: "x", p: 1, entropy: 0, alts: [] }, asked);
	onEvent(live, { type: "end", rid: "t", n_tokens: 2 }, asked);
	assert.equal(live.dense, "Dense");
});

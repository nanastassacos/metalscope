import assert from "node:assert/strict";
import { test } from "node:test";
import { applyLiveHeaders as applyTo, editOver, editStarts } from "../provider.ts";
import { onEvent } from "../state.ts";
import { type SteerPart, steerHeader, steerLabel, steerState } from "../steer.ts";

/** The live headers on headers of any names, as Pi passes them (not the literal type of `{}`). */
const applyLiveHeaders = (h: Record<string, string | undefined>, l: Parameters<typeof applyTo>[1], m: string) => applyTo(h, l, m);

const part: SteerPart = { name: "fluffy", model: "m", layer: 20, strength: 0.5, source: "profile" };
const steer = { model: "m", parts: [part] } as any;
const live = { steer, level: "on", clientId: "pi-test" };

test("a plain request gets the live steer and trace headers", () => {
	const h = applyLiveHeaders({}, live, "m");
	assert.equal(h["X-Scope-Steer"], steerHeader(steer));
	assert.equal(h["X-Scope-Trace"], "on");
	assert.equal(h["X-Scope-Trace-Client"], "pi-test");
});

test("headers a call sets are kept: /replay's unsteered column stays unsteered and untraced", () => {
	// The bug this guards against: every /replay column went out with the live steer and was traced,
	// so the "unsteered" reply was a steered one.
	const h = applyLiveHeaders({ "X-Scope-Steer": "", "X-Scope-Trace": "off" }, live, "m");
	assert.equal(h["X-Scope-Steer"], undefined);
	assert.equal(h["X-Scope-Trace"], "off");
	assert.equal(h["X-Scope-Trace-Client"], undefined);
	const c = applyLiveHeaders({ "X-Scope-Steer": `random~${steerHeader(steer)}`, "X-Scope-Trace": "off" }, live, "m");
	assert.equal(c["X-Scope-Steer"], `random~${steerHeader(steer)}`);
});

test("the dock's word edit goes out as X-Scope-Edit on the requests of /steer again's run only, a retry's too, beside the steer on", () => {
	const edit = { rid: "r9", at: 57, steer: "remove~soccer/18-44+rugby@0.7/31", model: "m", word: "soccer", want: "rugby" };
	const l: any = { ...live, edit: { ...edit } };
	assert.equal(applyLiveHeaders({}, l, "m")["X-Scope-Edit"], undefined);                // not armed: no request carries it
	l.edit.prompt = "Name a sport.";                                                      // armed by /steer again (replay.ts)
	editStarts(l, "Name a sport.");
	const h = applyLiveHeaders({}, l, "m");
	assert.equal(h["X-Scope-Edit"], "r9@57 remove~soccer/18-44+rugby@0.7/31");
	assert.equal(h["X-Scope-Steer"], steerHeader(steer));
	assert.equal(l.edit.sent, true);
	assert.equal(applyLiveHeaders({}, l, "m")["X-Scope-Edit"], "r9@57 remove~soccer/18-44+rugby@0.7/31");   // Pi's retry
	l.edit.over = true;                                                                   // the run settled
	assert.equal(applyLiveHeaders({}, l, "m")["X-Scope-Edit"], undefined);
	editStarts(l, "Name a sport.");                                                      // the next run: dropped
	assert.equal(l.edit, undefined);
	// Armed, but Pi never started that run (its send failed): the next run, another message, does not carry it.
	const f: any = { ...live, edit: { ...edit, prompt: "Name a sport." } };
	editStarts(f, "something else");
	assert.equal(f.edit, undefined);
	assert.equal(applyLiveHeaders({}, f, "m")["X-Scope-Edit"], undefined);
});

test("the edit goes on its run's requests until the answer: a retry after an error keeps it; a queued message's request or a compaction's does not", () => {
	const edit = { rid: "r9", at: 57, steer: "remove~soccer/18-44", model: "m", word: "soccer", prompt: "Name a sport." };
	const header = (l: any) => applyLiveHeaders({}, l, "m")["X-Scope-Edit"];
	const l: any = { ...live, edit: { ...edit } };
	editStarts(l, "Name a sport.");
	assert.ok(header(l));
	editOver(l, { type: "message_end", message: { role: "user" } });
	editOver(l, { type: "message_end", message: { role: "assistant", stopReason: "error" } });   // 502: Pi retries
	assert.ok(header(l), "the retry carries it");
	editOver(l, { type: "message_end", message: { role: "assistant", stopReason: "stop" } });    // the answer
	assert.equal(header(l), undefined, "a message queued meanwhile, answered in the same run");
	// Compaction inside the run (after an overflow error, no answer yet): its summary request carries no edit.
	const c: any = { ...live, edit: { ...edit } };
	editStarts(c, "Name a sport.");
	editOver(c, { type: "message_end", message: { role: "assistant", stopReason: "error" } });
	editOver(c, { type: "session_before_compact" });
	assert.equal(header(c), undefined);
});

test("an edit made while a reply streams: the aborted run settling does not end it, nor one settling once its own started", () => {
	const edit = { rid: "r9", at: 57, steer: "remove~soccer/18-44", model: "m", word: "soccer" };
	const header = (l: any) => applyLiveHeaders({}, l, "m")["X-Scope-Edit"];
	// The widget sets live.edit; /steer again aborts the streaming run, which settles; then it arms and sends.
	const l: any = { ...live, edit: { ...edit } };
	editOver(l, { type: "agent_settled" }, true);
	l.edit.prompt = "Name a sport.";
	editStarts(l, "Name a sport.");
	assert.deepEqual([l.edit.over, l.edit.running], [undefined, true]);
	assert.ok(header(l));
	// /steer again landing while agent_settled is being emitted: the old run's settle arrives after the edit's run
	// started (the session not idle), and leaves it.
	const m: any = { ...live, edit: { ...edit, prompt: "Name a sport." } };
	editStarts(m, "Name a sport.");
	editOver(m, { type: "agent_settled" }, false);
	assert.ok(header(m));
	editOver(m, { type: "agent_settled" }, true);                                         // its own run settling
	assert.equal(header(m), undefined);
});

test("the steer header is only added for the model it belongs to; tracing off says off", () => {
	assert.equal(applyLiveHeaders({}, live, "other")["X-Scope-Steer"], undefined);
	const h = applyLiveHeaders({}, { ...live, level: "off" }, "m");
	assert.equal(h["X-Scope-Steer"], steerHeader(steer));
	// Said, not left out: the server's global mode (metalscope on) would trace a request with no trace header.
	assert.equal(h["X-Scope-Trace"], "off");
	assert.equal(h["X-Scope-Trace-Client"], undefined);
	// header names are matched without regard to case
	const l = applyLiveHeaders({ "x-scope-steer": "" }, live, "m");
	assert.deepEqual(Object.keys(l).filter((k) => k.toLowerCase() === "x-scope-steer"), []);
});


test("a steer is sent for the served model it was set on, though its vectors name the checkpoint", () => {
	// The bug this guards against: the steer kept the profile's model (the checkpoint, Qwen3.8-Flash-Next) and the
	// header went out only when that equalled Pi's model id (Qwen3.8-Flash-Next-ds4-q4k), so never.
	const p: SteerPart = { name: "fluffy", model: "Qwen3.8-Flash-Next", layer: 20, strength: 0.8, source: "profile" };
	const s = steerState([p], "Qwen3.8-Flash-Next-ds4-q4k");
	const l = { steer: s, level: "on", clientId: "pi-test" };
	assert.equal(applyLiveHeaders({}, l, "Qwen3.8-Flash-Next-ds4-q4k")["X-Scope-Steer"], "fluffy@0.8/20");
	assert.equal(applyLiveHeaders({}, l, "Qwen3.8-Flash-Next-oQ4e-mtp")["X-Scope-Steer"], undefined);   // another copy: its own doses
});


test("each traced reply knows the steer it ran under (Reply.steer, from the request's headers)", () => {
	const l: any = { steer, level: "on", clientId: "pi-test", connected: true };
	applyLiveHeaders({}, l, "m");
	assert.equal(l.sentSteer, steerLabel(steer));
	onEvent(l, { type: "start", rid: "a", level: "on" });
	assert.equal(l.reply.steer, steerLabel(steer));
	applyLiveHeaders({}, l, "other");                                        // the steer belongs to another model: none sent
	onEvent(l, { type: "start", rid: "b", level: "on" });
	assert.equal(l.reply.steer, "");
	applyLiveHeaders({ "X-Scope-Trace": "off", "X-Scope-Steer": "x@1" }, l, "m");   // an untraced call (/replay) never reaches the stream
	assert.equal(l.sentSteer, "");
	onEvent(l, { type: "token", rid: "c", i: 0, id: 1, text: "x" });          // adopted mid-stream: unknown
	assert.equal(l.reply.rid, "c");
	assert.equal(l.reply.steer, undefined);
});

test("every request records the steer it went out with, traced or not (/steer replay's label)", () => {
	const l: any = { steer, level: "off", clientId: "pi-test", connected: true };
	assert.equal(l.requestSteer, undefined);                                   // none sent yet: unknown
	applyLiveHeaders({}, l, "m");                                              // tracing off: steered all the same
	assert.equal(l.requestSteer, steerLabel(steer));
	assert.equal(l.sentSteer, undefined);                                      // the live stream's card: traced requests only
	applyLiveHeaders({}, l, "other");
	assert.equal(l.requestSteer, "");
	applyLiveHeaders({ "X-Scope-Steer": "x@1" }, l, "m");                       // the caller's own steer: not ours to name
	assert.equal(l.requestSteer, undefined);
});

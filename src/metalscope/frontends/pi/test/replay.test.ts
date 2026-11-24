import assert from "node:assert/strict";
import { test } from "node:test";
import { answerAgain, replayText, replyAfterLastUser, yourReplyLabel } from "../replay.ts";
import { REPLAY_MAX_TOKENS } from "../steer.ts";

const msg = (role: string, content: any) => ({ type: "message", message: { role, content } });

test("your reply is every assistant message after your last one; tool calls are noted", () => {
	assert.equal(replyAfterLastUser([msg("user", "hi")]), undefined);                    // not answered yet
	assert.equal(replyAfterLastUser([]), undefined);
	const plain = replyAfterLastUser([msg("user", "old"), msg("assistant", [{ type: "text", text: "old reply" }]), msg("user", "Howdy!"),
		msg("assistant", [{ type: "thinking", thinking: "hmm" }, { type: "text", text: "Hello!" }])]);
	assert.deepEqual(plain, { text: "Hello!", tools: false });
	const tooled = replyAfterLastUser([msg("user", "list files"), msg("assistant", [{ type: "toolCall", name: "ls" }]),
		{ type: "message", message: { role: "toolResult", content: "a b" } }, msg("assistant", [{ type: "text", text: "a and b" }])]);
	assert.deepEqual(tooled, { text: "a and b", tools: true });
});

test("the labels say which steer your reply ran under", () => {
	assert.equal(yourReplyLabel("angry@0.3 L24", false), "your reply · steer angry@0.3 L24");
	assert.equal(yourReplyLabel("", false), "your reply · unsteered");
	assert.equal(yourReplyLabel(undefined, true), "your reply · its steer unknown (sent before Pi last started) · with tool calls (the replay offers none)");
});

test("the fresh answer is capped at 2,048 tokens and says so when cut", () => {
	assert.equal(REPLAY_MAX_TOKENS, 2048);
	assert.equal(replayText({ content: [{ type: "text", text: "Hi." }], stopReason: "stop" }), "Hi.");
	assert.equal(replayText({ content: [{ type: "text", text: "On and on" }], stopReason: "length" }), "On and on\n[cut off at 2,048 tokens]");
	assert.equal(replayText({ content: [{ type: "thinking", thinking: "..." }], stopReason: "length" }), "(cut off at 2,048 tokens while still thinking)");
	assert.equal(replayText({ content: [], stopReason: "error", errorMessage: "boom" }), "error: boom");
	assert.equal(replayText({ content: [], stopReason: "stop" }), "(empty)");
});

/** A command's ctx on `branch`, its calls recorded; `busy`: a reply running; `editor`: the editor's text. */
function commandCtx(branch: any[], { busy = false, editor = "", cancelled = false } = {}) {
	const calls: string[] = [];
	let text = editor;
	const ctx = {
		isIdle: () => !busy,
		abort: () => calls.push("abort"),
		waitForIdle: async () => { calls.push("waitForIdle"); busy = false; },
		navigateTree: async (id: string) => { calls.push(`navigateTree ${id}`); if (!text.trim()) text = "restored"; return { cancelled }; },
		sessionManager: { getBranch: () => branch },
		ui: { notify: (m: string, kind: string) => calls.push(`notify ${kind}: ${m}`), getEditorText: () => text,
			setEditorText: (t: string) => { calls.push(`setEditorText ${JSON.stringify(t)}`); text = t; } },
	};
	const pi = { sendUserMessage: (c: any) => calls.push(`send ${JSON.stringify(c)}`) } as any;
	return { ctx, pi, calls, editor: () => text };
}
const entry = (id: string, role: string, content: any) => ({ id, ...msg(role, content) });

test("/steer again: the reply running stopped, back to before your last message, the editor as it was, the message sent again", async () => {
	const branch = [entry("u1", "user", "old"), entry("a1", "assistant", [{ type: "text", text: "x" }]), entry("u2", "user", [{ type: "text", text: "Name a sport." }]),
		entry("a2", "assistant", [{ type: "text", text: "Soccer" }])];
	const a = commandCtx(branch, { busy: true });
	await answerAgain(a.pi, a.ctx);
	assert.deepEqual(a.calls, ["abort", "waitForIdle", "navigateTree u2", 'setEditorText ""', 'send [{"type":"text","text":"Name a sport."}]']);
	const b = commandCtx(branch, { editor: "half a thought" });                       // idle; the editor's own text is kept
	await answerAgain(b.pi, b.ctx);
	assert.deepEqual(b.calls, ["navigateTree u2", 'send [{"type":"text","text":"Name a sport."}]']);
	assert.equal(b.editor(), "half a thought");
	const c = commandCtx(branch, { cancelled: true });                                // an extension cancelled the navigation
	await answerAgain(c.pi, c.ctx);
	assert.deepEqual(c.calls, ["navigateTree u2", "notify error: not answered again: cancelled"]);
});

test("/steer again: one at a time (asked for meanwhile, the one under way answers); Pi refusing to navigate is one line", async () => {
	const branch = [entry("u1", "user", "Name a sport."), entry("a1", "assistant", [{ type: "text", text: "Soccer" }])];
	const a = commandCtx(branch, { busy: true });
	const first = answerAgain(a.pi, a.ctx);                                           // waiting for the reply to stop
	await answerAgain(a.pi, a.ctx);                                                   // a second edit applied meanwhile
	await first;
	assert.deepEqual(a.calls, ["abort", "waitForIdle", "navigateTree u1", 'setEditorText ""', 'send "Name a sport."']);
	const b = commandCtx(branch);
	b.ctx.navigateTree = async () => { throw new Error("Wait for the current response to finish before navigating the session tree."); };
	await answerAgain(b.pi, b.ctx);
	assert.deepEqual(b.calls, ["notify error: not answered again: Wait for the current response to finish before navigating the session tree."]);
	await answerAgain(a.pi, a.ctx);                                                   // not left under way: the next one runs
	assert.equal(a.calls.filter((c) => c.startsWith("send")).length, 2);
});

test("/steer again: refused after a turn that ran tools, or with no message; one line each", async () => {
	const tooled = [entry("u1", "user", "list files"), entry("a1", "assistant", [{ type: "toolCall", name: "ls" }])];
	for (const busy of [false, true]) {
		const a = commandCtx(tooled, { busy });
		await answerAgain(a.pi, a.ctx);
		assert.deepEqual(a.calls, ["notify error: your last reply ran tools: not answered again (their changes stay)"], `busy ${busy}`);
	}
	const none = commandCtx([]);
	await answerAgain(none.pi, none.ctx);
	assert.deepEqual(none.calls, ["notify error: no message to answer again"]);
});

test("/steer again: the dock's word edit is armed with the message sent, dropped when none is or the send fails, refused for another model", async () => {
	const edit = { rid: "r", at: 3, steer: "remove~cat/18-44", model: "m", word: "cat" };
	const on = (c: ReturnType<typeof commandCtx>, id = "m") => Object.assign(c.ctx, { model: { id, provider: "metalscope-ds4" } });
	const tooled = commandCtx([entry("u1", "user", "list files"), entry("a1", "assistant", [{ type: "toolCall", name: "ls" }])]);
	on(tooled);
	const refused = { edit: { ...edit } };
	await answerAgain(tooled.pi, tooled.ctx, refused);
	assert.equal(refused.edit, undefined);                                         // not left for the next message
	const branch = [entry("u1", "user", [{ type: "text", text: "Name a sport." }]), entry("a1", "assistant", [{ type: "text", text: "Soccer" }])];
	const plain = commandCtx(branch);
	on(plain);
	const sent = { edit: { ...edit } };
	await answerAgain(plain.pi, plain.ctx, sent);
	assert.deepEqual(sent.edit, { ...edit, prompt: "Name a sport." });            // its run's requests carry it (provider.ts)
	const failing = commandCtx(branch);
	on(failing);
	failing.pi.sendUserMessage = () => { throw new Error("busy"); };
	const lost = { edit: { ...edit } };
	await answerAgain(failing.pi, failing.ctx, lost);
	assert.deepEqual([lost.edit, failing.calls.at(-1)], [undefined, "notify error: not answered again: busy"]);
	const other = commandCtx(branch);
	on(other, "another");
	const mismatch = { edit: { ...edit } };
	await answerAgain(other.pi, other.ctx, mismatch);
	assert.deepEqual([mismatch.edit, other.calls], [undefined, ["notify error: not edited: that reply is another model's"]]);
});

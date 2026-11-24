/**
 * /steer replay: what your last message would have got the other way. Your reply stays as it is (labelled with the
 * steer it ran under); one fresh answer to the same message runs beside it: unsteered (`/steer replay`), or under
 * the steer given (`/steer replay angry@0.6`, any /steer form) without changing the steer that is on. At most
 * REPLAY_MAX_TOKENS, thinking included; a cut answer says so. The pair lands as an "scope-replay" entry (entries.ts),
 * outside the transcript the model sees. commands.ts parses /steer and calls `Replay`.
 */

import { type Ext, modelShort, notOnMetalscope, onMetalscope } from "./provider.ts";
import type { Edited } from "./state.ts";
import { REPLAY_MAX_TOKENS, type SteerState, steerHeader, steerTag } from "./steer.ts";

/** A message's text: the string, or its text parts joined by `sep`. */
const textOf = (content: any, sep = "\n"): string =>
	typeof content === "string" ? content : Array.isArray(content) ? content.filter((c: any) => c.type === "text").map((c: any) => c.text).join(sep) : "";

/** Your reply to the last user message, from the session branch: the text of every assistant message after it
 * (a reply with tool calls is several), and whether it called tools; undefined when there is none yet. */
export function replyAfterLastUser(branch: any[]): { text: string; tools: boolean } | undefined {
	const msgs = branch.filter((e: any) => e.type === "message" && e.message).map((e: any) => e.message);
	const lastUser = msgs.map((m: any) => m.role).lastIndexOf("user");
	if (lastUser < 0) return undefined;
	const after = msgs.slice(lastUser + 1).filter((m: any) => m.role === "assistant");
	if (!after.length) return undefined;
	const text = after.map((m: any) => textOf(m.content)).filter(Boolean).join("\n\n");
	const tools = after.some((m: any) => Array.isArray(m.content) && m.content.some((c: any) => c.type === "toolCall"));
	return { text, tools };
}

/** The last user message on the session branch (its entry), or undefined. */
const lastUserEntry = (branch: any[]): any => [...branch].reverse().find((e: any) => e.type === "message" && e.message?.role === "user");

let answering = false;   // answerAgain under way: its message is not sent yet

/** `/steer again`: your last message answered again under the steer on now, the reply running stopped first. The old
 * reply leaves the transcript and stays in the session tree (/tree). A turn that ran tools is refused: their changes
 * stay. Needs a command's ctx (navigateTree); the dock sends `/steer again` as a command. One at a time: asked for
 * while one is under way, that one's answer is the one (its request reads the steer on when it goes out). The dock's
 * word edit (`live.edit`) goes with this answer only (armed with the message sent: provider.ts editStarts); dropped
 * when none is sent, and refused for another model's reply (it would go out as a plain answer again). */
export async function answerAgain(pi: Ext["pi"], ctx: any, live?: { edit?: Edited }) {
	if (answering) return;
	const tools = () => !!replyAfterLastUser(ctx.sessionManager.getBranch())?.tools;
	const refuse = "your last reply ran tools: not answered again (their changes stay)";
	let sent = false;
	answering = true;
	try {
		if (live?.edit && (!onMetalscope(ctx) || live.edit.model !== modelShort(ctx)))
			return ctx.ui.notify("not edited: that reply is another model's", "error");
		if (tools()) return ctx.ui.notify(refuse, "error");
		if (!ctx.isIdle()) {
			ctx.abort();
			await ctx.waitForIdle();
		}
		const user = lastUserEntry(ctx.sessionManager.getBranch());
		if (!user) return ctx.ui.notify("no message to answer again", "error");
		if (tools()) return ctx.ui.notify(refuse, "error");
		const empty = !ctx.ui.getEditorText().trim();   // navigateTree puts the message in an empty editor
		if ((await ctx.navigateTree(user.id)).cancelled) return ctx.ui.notify("not answered again: cancelled", "error");
		if (empty) ctx.ui.setEditorText("");
		if (live?.edit) live.edit.prompt = textOf(user.message.content);
		pi.sendUserMessage(user.message.content);
		sent = true;
	} catch (e: any) {
		ctx.ui.notify(`not answered again: ${e?.message ?? e}`, "error");   // Pi refuses to navigate while a reply starts
	} finally {
		answering = false;
		if (!sent && live) live.edit = undefined;
	}
}

/** The label over your reply: the steer it ran under (live.reply.steer: "" none, undefined unknown). */
export function yourReplyLabel(steer: string | undefined, tools: boolean): string {
	const how = steer ? `steer ${steer}` : steer === "" ? "unsteered" : "its steer unknown (sent before Pi last started)";
	return `your reply · ${how}${tools ? " · with tool calls (the replay offers none)" : ""}`;
}

/** The fresh answer's text as shown: its text, or why there is none, and a note when the cap cut it. */
export function replayText(msg: { content?: any[]; stopReason?: string; errorMessage?: string }, max = REPLAY_MAX_TOKENS): string {
	const text = textOf(msg.content, "");
	const cut = msg.stopReason === "length";
	if (!text) return msg.errorMessage ? `error: ${msg.errorMessage}` : cut ? `(cut off at ${max.toLocaleString("en")} tokens while still thinking)` : "(empty)";
	return text + (cut ? `\n[cut off at ${max.toLocaleString("en")} tokens]` : msg.stopReason && msg.stopReason !== "stop" ? ` [${msg.stopReason}]` : "");
}

/** `/steer replay [STEER] [temp:T]`: `steer` the steer to answer under (undefined: unsteered); `temperature` as
 * parsed (undefined: Pi's default). */
export type Replay = (ctx: any, opts: { steer?: SteerState; temperature?: number }) => Promise<void>;

export function registerReplay({ pi, live }: Ext): Replay {
	// The transcript as the model last saw it (system messages included), kept for /steer replay.
	let lastContext: any[] | undefined;
	pi.on("context_with_system", (event) => {
		lastContext = event.messages;
	});

	return async (ctx, { steer, temperature }) => {
		if (notOnMetalscope(ctx)) return;
		const yours = replyAfterLastUser(ctx.sessionManager.getBranch());
		const yourSteer = live.requestSteer;   // the latest request is your reply's (or its last step's), traced or not
		if (!steer && yours && yourSteer === "")
			return ctx.ui.notify("Your reply was unsteered already: /steer replay CONCEPT [dose] shows what a steer would have made of it", "info");
		// What the real reply saw: the system prompt and every turn up to and including your last
		// message (anything after it, the reply or its tool calls, is dropped). Fallback when no
		// request has been made yet in this session: the last user message from the branch alone.
		const { convertToLlm } = await import("@earendil-works/pi-coding-agent");
		let msgs: any[] = lastContext ? convertToLlm(lastContext as any) : [];
		if (!msgs.length) {
			const last = lastUserEntry(ctx.sessionManager.getBranch());
			if (last) msgs = [{ role: "user", content: last.message.content, timestamp: Date.now() }];
		}
		const lastUser = msgs.map((m) => m.role).lastIndexOf("user");
		if (lastUser < 0) return ctx.ui.notify("No user message to replay", "warning");
		msgs = msgs.slice(0, lastUser + 1);
		const systemPrompt = msgs.filter((m) => m.role === "system").map((m) => textOf(m.content)).join("\n\n") || undefined;
		const context = { systemPrompt, messages: msgs.filter((m) => m.role !== "system") } as any;
		const prompt = textOf(msgs[lastUser].content);
		const model = ctx.model;
		if (!model) return ctx.ui.notify("No model selected", "warning");
		const level = pi.getThinkingLevel();
		const reasoning = level === "off" ? undefined : level;   // the request option has no "off": omit it instead
		const tempNote = temperature === undefined ? "" : temperature === 0 ? ", temp 0 (greedy)" : `, temp ${temperature}`;
		// An empty X-Scope-Steer means "no steer", to the server and to the header hook (provider.ts), so the live
		// steer is never added to the unsteered answer, whether or not Pi runs the hook here.
		const label = steer ? steerTag(steer) : "unsteered";
		const headers = { "X-Scope-Trace": "off", "X-Scope-Steer": steer ? steerHeader(steer) : "" };
		ctx.ui.notify(`replaying ${label} (${context.messages.length} messages, thinking ${level}${tempNote}, at most ${REPLAY_MAX_TOKENS.toLocaleString("en")} tokens)…`, "info");
		let fresh: string;
		try {
			const msg = await ctx.modelRegistry.streamSimple(model, context, { headers, reasoning, maxTokens: REPLAY_MAX_TOKENS,
				...(temperature === undefined ? {} : { temperature }) }).result();
			fresh = replayText(msg);
		} catch (e: any) {
			fresh = `error: ${e.message}`;
		}
		const replies: [string, string][] = [];
		if (yours) replies.push([yourReplyLabel(yourSteer, yours.tools), yours.text || "(no text)"]);
		replies.push([label, fresh]);
		pi.appendEntry("scope-replay", { v: 1, prompt, replies, ...(temperature === undefined ? {} : { temperature }) });
	};
}

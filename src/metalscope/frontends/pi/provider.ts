/**
 * The metalscope servers Pi talks to, each as a Pi provider named after its engine ("metalscope-omlx",
 * "metalscope-ds4"), and what the other modules share about them.
 *
 * - The servers come from the metalscope command line (backends.ts registers both backends' models); omlxServer is
 *   where oMLX's answers (~/.omlx/settings.json). Picking a model with /model picks its server: the live stream, the
 *   dock and the commands follow it (index.ts).
 * - registerUrl: with METALSCOPE_URL set, that one server instead (one pointed at by hand, or
 *   scripts/pi_mock_omlx.py): its chat models from /v1/models, each with the thinking controls the server reports
 *   for it (status `chat`).
 * - registerHeaders: tags Pi's requests to the server (X-Scope-Trace, X-Scope-Trace-Client, X-Scope-Steer) so they
 *   are traced and steered.
 * - Ext: Pi, the live state (its `base` is the selected model's server) and the metalscope API on it, handed to every
 *   feature module.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { createAssistantMessageEventStream, streamSimpleOpenAICompletions } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ProviderConfig } from "@earendil-works/pi-coding-agent";
import type { Edited, Live } from "./state.ts";
import { steerHeader, steerLabel } from "./steer.ts";

/** Every provider the extension registers is named PROVIDER-<engine>. */
export const PROVIDER = "metalscope";
/** Where `metalscope ds4 serve` puts its proxy by default. */
export const DS4_URL = "http://127.0.0.1:8009";

/** What every feature module gets from the extension entry. */
export interface Ext {
	pi: ExtensionAPI;
	live: Live; // live.base: the selected model's metalscope server, e.g. http://127.0.0.1:8008
	/** JSON from metalscope's routes on the server (`${base ?? live.base}/scope${path}`); throws the server's detail. */
	api: (path: string, init?: RequestInit, base?: string) => Promise<any>;
	servers?: Map<string, Server>; // the registered servers by provider (index.ts): each model's flags (status `chat`)
}

/** The key a request to metalscope's routes at `url` carries: oMLX's, to oMLX's own address only (its /scope routes
 * ask for the key its API does, while one is set); none to any other server. */
export function scopeAuth(url: string, omlx: Candidate = omlxServer()): Record<string, string> {
	try {
		return omlx.key && new URL(url).origin === new URL(omlx.base).origin ? { Authorization: `Bearer ${omlx.key}` } : {};
	} catch {
		return {};
	}
}

export function traceApi(live: { base: string }) {
	return async (path: string, init?: RequestInit, base = live.base) => {
		const res = await fetch(`${base}/scope${path}`, { ...init, headers: { "content-type": "application/json", ...scopeAuth(base), ...(init?.headers ?? {}) },
			signal: init?.signal ?? AbortSignal.timeout(API_TIMEOUT_MS) });
		if (!res.ok) throw Object.assign(new Error((await res.json().catch(() => ({})))?.detail ?? `HTTP ${res.status}`), { status: res.status });
		return res.json();
	};
}

export const modelShort = (ctx: any) => (ctx.model?.id ?? "").split("/").pop();

/** Whether the selected model is a metalscope server's (only their requests are traced and steered). */
export const onMetalscope = (ctx: { model?: { provider?: string } }) => (ctx.model?.provider ?? "").startsWith(`${PROVIDER}-`);

/** Whether the selected model is not a metalscope server's, saying so with a warning (`why`: what needs one). */
export function notOnMetalscope(ctx: any, why?: string): boolean {
	if (onMetalscope(ctx)) return false;
	ctx.ui.notify(`Switch to a metalscope model with /model first${why ? `: ${why}` : ""}`, "warning");
	return true;
}

/** The selected model's engine: "omlx" or "ds4" for the backends' providers; undefined for another (a server set
 * with METALSCOPE_URL, or a model that is not metalscope's). */
export function engineOf(ctx: { model?: { provider?: string } } | undefined): "omlx" | "ds4" | undefined {
	const p = ctx?.model?.provider;
	return p === `${PROVIDER}-omlx` ? "omlx" : p === `${PROVIDER}-ds4` ? "ds4" : undefined;
}

/** A server's address without its scheme: "127.0.0.1:8008". */
export const hostOf = (base: string) => base.replace(/^https?:\/\//, "");

/** What every place says when the server did not answer: the engine, where, and the next step ("oMLX is not answering
 * at 127.0.0.1:8008: send a message (Pi starts the model) or run metalscope omlx MODEL"; `engine` undefined: no
 * backend's model is selected). */
export function unreachable(base: string, engine?: "omlx" | "ds4", model?: string): string {
	if (engine) return `${engine === "omlx" ? "oMLX" : "ds4"} is not answering at ${hostOf(base)}: send a message (Pi starts the model) or run metalscope ${engine} ${model || "MODEL"}`;
	return `no metalscope server is answering at ${hostOf(base)}: pick a model marked (metalscope) with /model`;
}

/** How long traceApi waits for an answer. */
export const API_TIMEOUT_MS = 20000;

/** A failed request to the server, in words: what it answered (an error carrying its HTTP `status`, traceApi's); no
 * answer in API_TIMEOUT_MS (it is up, maybe still working: /jlens use loading a lens); else that it did not answer
 * (unreachable). */
export function requestFailed(e: any, base: string, ctx: any): string {
	const engine = engineOf(ctx), who = `${engine === "ds4" ? "ds4" : engine === "omlx" ? "oMLX" : "The server"} at ${hostOf(base)}`;
	if (typeof e?.status === "number") return `${who} answered: ${e.message}`;
	if (e?.name === "TimeoutError") return `${who}: no answer in ${API_TIMEOUT_MS / 1000} s (it may still be working: try again in a moment)`;
	return unreachable(base, engine, modelShort(ctx));
}

export interface Candidate {
	base: string;
	key: string;
}

/** Where oMLX's server answers, and its key, from its settings (127.0.0.1:8008 without them). */
export function omlxServer(settingsPath = join(homedir(), ".omlx", "settings.json")): Candidate {
	try {
		const s = JSON.parse(readFileSync(settingsPath, "utf8"));
		const host = ["0.0.0.0", "::"].includes(s.server?.host) ? "127.0.0.1" : s.server?.host ?? "127.0.0.1";
		return { base: `http://${host}:${s.server?.port ?? 8008}`, key: s.auth?.api_key ?? "" };
	} catch {
		return { base: "http://127.0.0.1:8008", key: "" };
	}
}

/** What a server says of itself at /scope/api/status (the parts Pi reads). */
export interface ServerStatus {
	engine?: string; // "omlx", "ds4"
	caps?: string[]; // what it can do now (web.py Backend; `lookup` while a served model has an engram)
	models?: { name: string }[]; // the models it serves now (loaded)
	chat?: Record<string, ModelFlags>; // each served model's thinking controls and flags
	stale?: string | null; // the installed metalscope code changed since it started: the restart that loads it
}

/** A served model as its server reports it (status `chat`): its thinking controls; `tested`: metalscope was run on its
 * family on this engine; `edit`: the server takes the dock's one-token edit for it (left out until oMLX knows). */
export interface ModelFlags {
	reasoning?: boolean;
	thinking_format?: string | null;
	tested?: boolean;
	edit?: boolean;
}

export interface Server extends Candidate {
	status: ServerStatus;
}

async function statusOf(base: string): Promise<ServerStatus | undefined> {
	try {
		const res = await fetch(`${base}/scope/api/status`, { headers: scopeAuth(base), signal: AbortSignal.timeout(3000) });
		return res.ok ? ((await res.json()) as ServerStatus) : undefined;
	} catch {
		return undefined;
	}
}

/** A line for each server (one per address) whose status says it runs older code than is installed. */
export async function staleNotices(servers: Iterable<Server>, status = statusOf): Promise<string[]> {
	const sts = await Promise.all([...new Set([...servers].map((s) => s.base))].map((b) => status(b)));
	return sts.flatMap((st) => (st?.stale ? [`server code changed: ${st.stale}`] : []));
}

/** A server's caps as its status gives them; none without a status. */
export const capsOf = (st: { caps?: unknown } | undefined): string[] => (Array.isArray(st?.caps) ? st.caps : []);

/** The models seen starting on each server entry (its status; a replaced entry starts afresh). */
const started = new WeakMap<ServerStatus, Set<string>>();

/** A reply of `model` started on the server at `base`: the model is loaded now, so the flags its server could not know
 * before (oMLX's edit) and its caps (oMLX lists `lookup` once Flash Next is loaded) are asked once a model an entry,
 * whatever is kept, and kept in that server's entry; after that only while one is unknown (`metalscope status` gives
 * the flags, never the caps). */
export async function refreshFlags(servers: Map<string, Server>, base: string, model: string | undefined, status = statusOf): Promise<void> {
	const s = [...servers.values()].find((x) => x.base === base), known = model ? s?.status.chat?.[model] : undefined;
	if (!s || !model) return;
	const seen = started.get(s.status) ?? new Set<string>();
	started.set(s.status, seen);
	if (seen.has(model) && known?.tested !== undefined && known?.edit !== undefined && s.status.caps !== undefined) return;
	seen.add(model);
	const st = await status(base), now = st?.chat?.[model];
	if (now) s.status.chat = { ...s.status.chat, [model]: { ...known, ...now } };
	if (st) s.status.caps = capsOf(st);
}

/** The selected model's server: its provider's, when that is one the extension registered. */
export const serverFor = (servers: Map<string, Server>, ctx: { model?: { provider?: string } }) => servers.get(ctx.model?.provider ?? "");

/** Before a metalscope model is picked, the server to follow: the first registered serving a model, else the first
 * registered (oMLX's first). */
export function defaultServer(servers: Iterable<Server>): Server | undefined {
	const all = [...servers];
	return all.find((s) => (s.status.models ?? []).length > 0) ?? all[0];
}

/** A model's context window as its server reports it in /v1/models (`max_model_len`: oMLX's model setting, ds4-server's
 * --ctx), which is when Pi compacts the conversation; DEFAULT_CONTEXT when the server says nothing usable. */
export const DEFAULT_CONTEXT = 131072, MAX_OUTPUT = 32768;
export function contextWindow(maxModelLen: unknown): number {
	return typeof maxModelLen === "number" && Number.isInteger(maxModelLen) && maxModelLen >= 1024 ? maxModelLen : DEFAULT_CONTEXT;
}

/** The models a server lists at /v1/models (asked with its key, within `ms`); throws when it does not answer. */
export async function listModels(c: Candidate, ms: number): Promise<{ id: string; max_model_len?: number | null }[]> {
	const res = await fetch(`${c.base}/v1/models`, { headers: c.key ? { Authorization: `Bearer ${c.key}` } : {}, signal: AbortSignal.timeout(ms) });
	if (!res.ok) throw new Error(`HTTP ${res.status}`);
	return ((await res.json()) as { data: { id: string; max_model_len?: number | null }[] }).data;
}

async function chatModels(server: Candidate, chat?: ServerStatus["chat"]) {
	const all = await listModels(server, 3000);
	// The thinking controls as the server reports them now (a model loaded since startup included).
	chat ??= (await statusOf(server.base))?.chat ?? {};
	return all.filter((m) => !/markitdown|embed|rerank|whisper|tts/i.test(m.id)).map((m) => piModel(m.id, m.max_model_len, chat[m.id]));
}

/** A served model as Pi registers it: its context window (contextWindow), output cap and thinking controls. */
export function piModel(id: string, maxModelLen: unknown, thinking: { reasoning?: boolean; thinking_format?: string | null } = {}) {
	const context = contextWindow(maxModelLen);
	return {
		id,
		name: `${id} (metalscope)`,
		reasoning: !!thinking.reasoning,
		input: ["text"] as ("text" | "image")[],
		contextWindow: context,
		maxTokens: Math.min(MAX_OUTPUT, context),
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		compat: {
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsUsageInStreaming: true,
			maxTokensField: "max_tokens" as const,
			...(thinking.reasoning && thinking.thinking_format ? { thinkingFormat: thinking.thinking_format as "qwen-chat-template" } : {}),
		},
	};
}

export type PiModel = ReturnType<typeof piModel>;

/** A server as Pi's provider: named after its engine, its OpenAI-compatible API and key, its models, and how /model
 * asks for them again; `streamSimple` Pi's own for that API, or keptFirst's around it. */
export const providerConfig = (engine: string | undefined, c: Candidate, models: PiModel[], refreshModels: () => Promise<PiModel[]>,
	streamSimple?: Stream) => ({
	name: `metalscope on ${engine ?? "a server"} (local)`,
	baseUrl: `${c.base}/v1`,
	apiKey: c.key || "metalscope",
	api: "openai-completions" as const,
	models,
	refreshModels,
	...(streamSimple ? { streamSimple } : {}),
});

/** An edited reply's kept part (the dock's word edit, live.edit, on the request now): its tokens before the edited one
 * as the server sends them first (edit.py ChatText), the thinking up to </think> (`thinking`: Pi asks for it), the
 * text after it, its leading newlines dropped. Undefined without an edit or its reply, or when the server's copy may
 * differ: a broken character (U+FFFD), a <think> token, the reply thinking or not unlike Pi's request. */
export function keptText(live: Pick<Live, "edit" | "reply" | "prev">, thinking: boolean): { thinking: string; text: string } | undefined {
	const e = live.edit, r = e?.running && !e.over ? [live.reply, live.prev, e.base].find((x) => x?.rid === e.rid) : undefined;
	if (!e || !r) return undefined;
	const s = r.tokens.filter((t) => t.i < e.at).map((t) => t.raw ?? t.text).join(""), end = s.indexOf("</think>");   // as the server has it
	if (s.includes("\uFFFD") || s.includes("<think>") || r.tokens.some((t) => t.text.includes("</think>")) !== thinking) return undefined;
	if (!thinking) return { thinking: "", text: s.replace(/^\n+/, "") };
	return end < 0 ? { thinking: s, text: "" } : { thinking: s.slice(0, end), text: s.slice(end + 8).replace(/^\n+/, "") };
}

/** Pi's stream for the servers (`inner`, openai-completions) that shows an edit's kept part at once (keptText): an
 * edited reply's answer starts with the reply up to the edited token, which the server sends only after it has
 * written that token, so the transcript would sit empty meanwhile. Pi's message starts with it; the server's copy,
 * when it comes, is taken in without showing it twice (a block shows the kept text while the server's agrees with it,
 * the server's from there). The server's differing (a block of another kind, or other text): its own events from
 * there, held until its message is as long as the one shown, so nothing is taken back. Any other request: `inner`. */
export function keptFirst(live: Pick<Live, "edit" | "reply" | "prev">, inner: Stream = streamSimpleOpenAICompletions as any): Stream {
	return (model, context, options) => {
		const kept = keptText(live, !!(model.reasoning && options?.reasoning)), src = inner(model, context, options);
		if (!kept || !(kept.thinking || kept.text)) return src;
		const out = createAssistantMessageEventStream(), zero = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0 };
		const msg: any = { role: "assistant", content: [], api: model.api, provider: model.provider, model: model.id,
			usage: { ...zero, cost: { ...zero, total: 0 } } as any, stopReason: "stop", timestamp: Date.now() };
		type Kind = "text" | "thinking";
		const blocks: { kind: Kind; at: number; want: string; got: string; taken: boolean }[] = [], map = new Map<number, number>();
		const field = (kind: Kind) => (kind === "text" ? "text" : "thinking");
		const size = (m: any) => m.content.reduce((n: number, c: any) => n + (c.text ?? c.thinking ?? "").length, 0);
		let raw: "" | "held" | "on" = "";   // the server differs: its events as they are, once its message is as long as the one shown
		out.push({ type: "start", partial: msg });
		for (const kind of ["thinking", "text"] as const) {
			const s = kept[kind];
			if (!s) continue;
			const at = msg.content.push(kind === "text" ? { type: "text", text: s } : { type: "thinking", thinking: s }) - 1;
			blocks.push({ kind, at, want: s, got: "", taken: false });
			out.push({ type: `${kind}_start`, contentIndex: at, partial: msg } as any);
			out.push({ type: `${kind}_delta`, contentIndex: at, delta: s, partial: msg } as any);
		}
		void (async () => {
			for await (const ev of src as AsyncIterable<any>) {
				if (ev.type === "start") continue;
				if (ev.type === "done" || ev.type === "error") {
					out.push(ev);
					return out.end(ev.type === "done" ? ev.message : ev.error);
				}
				if (raw) {
					if (raw === "held" && size(ev.partial) >= size(msg)) raw = "on";
					if (raw === "on") out.push(ev);
					continue;
				}
				const kind = ev.type.split("_")[0] as Kind | "toolcall";
				if (ev.type.endsWith("_start")) {
					const next = blocks.find((x) => !x.taken);
					if (next && next.kind !== kind) {   // not the kept block that comes next (a thinking one after the text)
						raw = "held";
						continue;
					}
					if (next) {
						next.taken = true;
						map.set(ev.contentIndex, next.at);
						continue;
					}
					const at = msg.content.push(kind === "toolcall" ? ev.partial.content[ev.contentIndex] : kind === "text" ? { type: "text", text: "" } : { type: "thinking", thinking: "" }) - 1;
					map.set(ev.contentIndex, at);
					if (kind !== "toolcall") blocks.push({ kind, at, want: "", got: "", taken: true });
					out.push({ ...ev, contentIndex: at, partial: msg } as any);
					continue;
				}
				const at = map.get(ev.contentIndex) ?? ev.contentIndex, b = blocks.find((x) => x.at === at), c: any = msg.content[at];
				if (b && ev.type.endsWith("_delta")) {
					b.got += (ev as any).delta;
					if (!b.want.startsWith(b.got) && !b.got.startsWith(b.want)) {   // other text than the kept
						raw = size(ev.partial) >= size(msg) ? "on" : "held";
						if (raw === "on") out.push(ev);
						continue;
					}
					const shown: string = c[field(b.kind)], now = b.want.startsWith(b.got) ? b.want : b.got;
					c[field(b.kind)] = now;
					if (now === shown) continue;
					out.push({ ...ev, contentIndex: at, delta: now.slice(shown.length), partial: msg } as any);
					continue;
				}
				if (b && ev.type.endsWith("_end")) c[field(b.kind)] = (ev as any).content;
				else if (kind === "toolcall") msg.content[at] = ev.partial.content[ev.contentIndex];
				out.push({ ...ev, contentIndex: at, partial: msg } as any);
			}
			out.end();
		})().catch((e: any) => {
			const error = { ...msg, stopReason: "error" as const, errorMessage: String(e?.message ?? e) };
			out.push({ type: "error", reason: "error", error });
			out.end(error);
		});
		return out as any;
	};
}
/** Pi's stream function for a provider (ProviderConfig's). */
type Stream = NonNullable<ProviderConfig["streamSimple"]>;

/** The server at `url` as provider PROVIDER-<engine> ("metalscope-mock" for the mock), added to `servers`; its name,
 * or undefined when it does not answer, serves no chat model, or Pi refuses it (a stale extension after /reload). */
export async function registerUrl(pi: ExtensionAPI, url: string, servers: Map<string, Server>, stream?: Stream): Promise<string | undefined> {
	const c: Candidate = { base: url.replace(/\/$/, ""), key: "" };
	try {
		const status = await statusOf(c.base);
		const models = status ? await chatModels(c, status.chat ?? {}) : [];
		if (!status || !models.length) return undefined;
		const name = `${PROVIDER}-${(status.engine ?? "server").toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
		pi.registerProvider(name, providerConfig(status.engine, c, models, () => chatModels(c), stream));
		servers.set(name, { ...c, status });
		return name;
	} catch {
		return undefined;
	}
}

/** Where the live stream belongs: the selected model's server; with no metalscope model selected, where it is while
 * that is a registered server, else the default server. */
export function nextBase(servers: Map<string, Server>, ctx: { model?: { provider?: string } } | undefined, current: string): string {
	const selected = serverFor(servers, ctx ?? {});
	if (selected) return selected.base;
	if ([...servers.values()].some((s) => s.base === current)) return current;
	return defaultServer(servers.values())?.base ?? current;
}

/** Tag requests to the server: trace level and client (tracing off: `X-Scope-Trace: off` alone), and
 * the active steer when it belongs to the selected model.
 *
 * Where this hook runs on a call's own headers, a header the caller set is
 * kept, and an empty `X-Scope-Steer` ("no steer") is dropped. Pi runs it on
 * the agent's own requests and its compaction's summary requests, though not
 * on `ctx.modelRegistry.streamSimple`, so /steer replay's columns go out with
 * exactly the headers it sets, and the server reads an empty `X-Scope-Steer` as
 * no steer too. */
export function registerHeaders({ pi, live }: Ext) {
	pi.on("before_provider_headers", (event, ctx) => {
		if (!onMetalscope(ctx)) return;
		applyLiveHeaders(event.headers as Record<string, string | undefined>, live, modelShort(ctx));
	});
	pi.on("before_agent_start", (event) => editStarts(live, (event as { prompt?: string }).prompt));
	pi.on("message_end", (event) => editOver(live, event as any));
	pi.on("session_before_compact", (event) => editOver(live, event as any));
	pi.on("agent_settled", (event, ctx) => editOver(live, event as any, ctx.isIdle()));
}

/** The edit's run is past its answer (`message_end` of an assistant message that is not an error: Pi's retry follows
 * an error, so it still carries the edit), compacting (the summary request is not the answer), or settled: no later
 * request carries it (a message queued meanwhile is answered in the same run, with no edit). Only the edit's own run
 * settling: one settling while the next has started (`idle` false: /steer again landing during the settle) leaves
 * it. */
export function editOver(live: { edit?: Edited }, ev: { type: string; message?: { role?: string; stopReason?: string } }, idle = true) {
	const e = live.edit;
	if (!e?.running || e.over) return;
	if (ev.type === "message_end" && (ev.message?.role !== "assistant" || ev.message.stopReason === "error")) return;
	if (ev.type === "agent_settled" && !idle) return;
	e.over = true;
}

/** A run starts: the dock's word edit (live.edit) is its if /steer again armed it with this message (replay.ts
 * answerAgain sets `prompt`), and is dropped otherwise: it never reaches another request. */
export function editStarts(live: { edit?: Edited }, prompt: string | undefined) {
	const e = live.edit;
	if (e && e.prompt !== undefined && !e.running && prompt === e.prompt) e.running = true;
	else live.edit = undefined;
}

/** The header rule above, on a plain headers object (exported for tests). The dock's word edit (live.edit) goes out
 * as X-Scope-Edit (the server answers it: edit.py) on its run's requests until its answer (editOver): Pi's
 * retry of a failed answer carries it too, so the answer is never a silent full regeneration. */
export function applyLiveHeaders<H extends Record<string, string | undefined>>(headers: H, live: { steer?: any; level: string; clientId: string; sentSteer?: string; requestSteer?: string; edit?: Edited }, model: string): H {
	const h = headers as Record<string, string | undefined>;
	const has = (name: string) => Object.keys(h).some((k) => k.toLowerCase() === name.toLowerCase());
	const key = (name: string) => Object.keys(h).find((k) => k.toLowerCase() === name.toLowerCase()) ?? name;
	if (live.edit?.running && !live.edit.over) {
		h["X-Scope-Edit"] = `${live.edit.rid}@${live.edit.at} ${live.edit.steer}`;
		live.edit.sent = true;
	}
	const callerSteer = has("X-Scope-Steer") && h[key("X-Scope-Steer")] !== "";
	let ours = false;
	if (has("X-Scope-Steer")) {
		if (h[key("X-Scope-Steer")] === "") delete h[key("X-Scope-Steer")];   // explicit "no steer"
	} else if (live.steer && live.steer.model === model) {
		h["X-Scope-Steer"] = steerHeader(live.steer);
		ours = true;
	}
	live.requestSteer = ours ? steerLabel(live.steer) : callerSteer ? undefined : "";   // traced or not: /steer replay's label
	if (has("X-Scope-Trace")) return headers;                                 // the caller chose the trace level
	if (live.level === "off") {
		h["X-Scope-Trace"] = "off";                                            // said, so the server's global mode (metalscope on) does not trace it
		return headers;
	}
	h["X-Scope-Trace"] = live.level;
	h["X-Scope-Trace-Client"] = live.clientId;
	live.sentSteer = ours ? steerLabel(live.steer) : "";                     // the live stream's next reply ran under this (its card says so)
	return headers;
}

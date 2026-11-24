/**
 * The four commands, registered with Pi: /steer, /scope, /concept and /jlens (their forms: help.ts COMMANDS).
 * They share the concept list cache, the once-per-session warnings and the server's steering switch. Their pure logic
 * is in steer.ts, scope.ts, concept.ts and jlens.ts; the jobs run in job-runner.ts and
 * /steer replay in replay.ts.
 */

import { seeHelp } from "./args.ts";
import { backendOf } from "./backends.ts";
import { type ConceptInfo, type ConceptList, chatRank, conceptCompletions, conceptListText, notLearnedText, parseConceptArgs, parseConcepts, parseLibrary } from "./concept.ts";
import { type Cmd, type Served, commandHelpLines, describe, guideLines, helpTopics } from "./help.ts";
import { lensCompletions, lensNames, lensStatusLines, lensUseText, parseLensArgs } from "./jlens.ts";
import type { Jobs } from "./job-runner.ts";
import { type Ext, engineOf, modelShort, notOnMetalscope, onMetalscope, requestFailed, unreachable } from "./provider.ts";
import { type Replay, answerAgain } from "./replay.ts";
import { parseScopeArgs, scopeCompletions, scopeStatusText } from "./scope.ts";
import { type SteerPart, type SteerState, type WordIds, parseSteerArgs, steerCompletions, steerFromProfile, steerFromWordOp, steerReplyText, steerState, steerStatusText, steerTag, WORD_EDIT_ONLY, wordBody } from "./steer.ts";
import { type Ui, dockStep } from "./widget.ts";

/** The selected model and its backend, whose job times the help gives. */
const servedOf = (ctx: any): Served => ({ backend: backendOf(ctx), model: modelShort(ctx) || undefined });

/** One command's forms as a guide entry in the transcript (never sent to the model); the job times `ctx`'s model's. */
function appendHelp(pi: Ext["pi"], ctx: any, cmd: Cmd, sub?: string) {
	const lines = commandHelpLines(cmd, sub, servedOf(ctx));
	pi.appendEntry("scope-guide", { v: 1, title: `${cmd}${sub ? ` ${sub}` : ""}: help`, lines });
}

export function registerCommands({ pi, live, api }: Ext, ui: Ui, jobs: Jobs, replay: Replay) {
	// The last concept list fetched, for the completions (every command fetches afresh). Pi gives completions no ctx,
	// so the selected model is followed here: after /model they offer nothing until the new model's list is fetched.
	let conceptCache: { model: string; data: ConceptList } | undefined;
	let selected = "";
	pi.on("session_start", (_event, ctx) => void (selected = modelShort(ctx)));
	pi.on("model_select", (_event, ctx) => void (selected = modelShort(ctx)));
	const cached = () => (conceptCache?.model === selected ? conceptCache.data : undefined);
	const conceptsUrl = (model: string) => `/api/concepts?model=${encodeURIComponent(model)}`;
	function listed(model: string, raw: any): ConceptList {
		const data = { ...raw, concepts: parseConcepts(raw), library: parseLibrary(raw) };
		conceptCache = { model, data };
		return data;
	}
	const conceptListOf = async (model: string) => listed(model, await api(conceptsUrl(model)));
	const conceptList = (ctx: any) => conceptListOf(modelShort(ctx));
	// A completion that needs names not fetched yet asks for them (the selected model's), quietly: none on failure.
	const quietly = <T>(p: Promise<T>): Promise<T | undefined> => p.catch(() => undefined);
	// The selected model's lens names (/jlens use), from its last /jlens status.
	let lensCache: { model: string; names: string[] } | undefined;
	// A link (the dock's l) gives the model another's concepts and lenses: the lists are asked for again.
	ui.forgetLists = () => void (conceptCache = lensCache = undefined);
	async function lensStatus(model: string) {
		const st = await api(`/api/lens?model=${encodeURIComponent(model)}`);
		lensCache = { model, names: lensNames(st) };
		return st;
	}
	/** The concept list, or undefined after saying the request failed (only the request: a bug reading the answer is not
	 * the server down). */
	async function listOrSay(ctx: any): Promise<ConceptList | undefined> {
		const model = modelShort(ctx);
		let raw;
		try {
			raw = await api(conceptsUrl(model));
		} catch (e: any) {
			return void ctx.ui.notify(requestFailed(e, live.base, ctx), "error");
		}
		return listed(model, raw);
	}

	/** Steering must be enabled on the server (off by default); turn it on from here, once, with a note. */
	async function ensureSteerEnabled(ctx: any, list: { steer_enabled: boolean }) {
		if (list.steer_enabled) return;
		await api("/api/steer", { method: "POST", body: JSON.stringify({ enabled: true }) });
		ctx.ui.notify("Steering enabled on the metalscope server for this machine (metalscope steering off reverses it)", "info");
	}

	/** /steer's handler; the dock's steer page runs it too (ui.steer), so its enter and x steer exactly as /steer does.
	 * `ids` (the dock's LENS edit): a word's own token (id and text, by its name), for its word vector; `use` (the same): the steer built is
	 * handed to it rather than turned on. A word (`-word`, `old>new`) is the edit's alone: a word changes one token, not a whole reply. */
	async function steer(args: string, ctx: any, ids?: WordIds, use?: (s: SteerState) => void) {
		const p = parseSteerArgs(args);
		if (p.cmd === "help") return appendHelp(pi, ctx, "steer");
		if ("error" in p && p.error) return ctx.ui.notify(p.error, "error");
		if (p.cmd === "off") {
			live.steer = undefined;
			ui.updateUi(ctx);
			return ctx.ui.notify("steer: off", "info");
		}
		if (p.cmd === "status") return ctx.ui.notify(steerStatusText(live.steer), "info");
		if (p.cmd === "again") return answerAgain(pi, ctx, live);
		// /steer replay alone: your reply against the same message unsteered.
		if (p.cmd === "replay" && !p.parts) return replay(ctx, { temperature: p.temperature });
		if (notOnMetalscope(ctx)) return;
		const list = await listOrSay(ctx);
		if (!list || (p.cmd !== "set" && p.cmd !== "replay")) return;
		const wants = p.parts ?? [];
		if (!use && wants.some((w) => w.op)) return ctx.ui.notify(`${WORD_EDIT_ONLY} · ${seeHelp("steer")}`, "error");
		const parts: SteerPart[] = [];
		// Every learned part is resolved before anything is built (a lens word vector is saved on the server) or changed.
		const built = new Map<number, [ConceptInfo, SteerPart]>();
		for (const [i, want] of wants.entries()) {
			if (want.op) continue;                                  // a removal or a swap: always a word (the dock's edit)
			const info = list.concepts.find((c) => c.name === want.name);
			// A library concept not learned here: say how to learn it. A word is the dock's edit's alone.
			if (!info && list.library?.some((c) => c.name === want.name)) return ctx.ui.notify(notLearnedText(want.name, list, backendOf(ctx)), "error");
			if (!info) return ctx.ui.notify(`${want.name} is not a concept here: /concept lists them · ${seeHelp("steer")}`, "error");
			const s = steerFromProfile(info, want.dose);
			if ("error" in s) return ctx.ui.notify(s.error, "error");
			built.set(i, [info, s]);
		}
		for (const [i, want] of wants.entries()) {
			const b = built.get(i);
			if (b) {
				parts.push(b[1]);
			} else {
				const tok = ids?.[want.name];
				let made: any;
				try {
					made = await api("/api/steer/words", { method: "POST", body: JSON.stringify(wordBody(modelShort(ctx), want.name, tok)) });
				} catch (e: any) {
					// 409: the server's own reason (a concept folder with no texts, a vector by that name) says it all.
					// The dock's edit (`use`): a word typed there that is a concept's name, said for the dock's key line.
					if (e?.status === 409) return ctx.ui.notify(use && !tok ? `"${want.name}" is a concept here, not a word: /steer ${want.name} steers it`
						: `${e.message} · ${seeHelp("steer")}`, "error");
					return ctx.ui.notify(`a word vector for ${tok?.text ?? want.name} could not be made: ${typeof e?.status === "number" ? e.message : requestFailed(e, live.base, ctx)} · ${seeHelp("steer")}`, "error");
				}
				// `cut`: words the server cut to their first token; every word form takes one token (the only ones tested).
				if (made.cut?.length) return ctx.ui.notify(`${want.name} is more than one token: the edit takes a one-token word`, "error");
				parts.push(steerFromWordOp(want, made, made.model ?? modelShort(ctx)));
			}
		}
		if (!parts.length) return ctx.ui.notify(`no concept given · ${seeHelp("steer")}`, "error");
		try {
			await ensureSteerEnabled(ctx, list);
		} catch (e: any) {
			return ctx.ui.notify(`Could not enable steering on the server: ${requestFailed(e, live.base, ctx)}`, "error");
		}
		const state = steerState(parts, modelShort(ctx), list.norms_note);
		// /steer replay STEER: the same message answered under it, the steer that is on left as it is.
		if (p.cmd === "replay") return replay(ctx, { steer: state, temperature: p.temperature });
		if (use) return use(state);
		live.steer = state;
		ui.updateUi(ctx);   // the line above the editor says what is on
		const said = steerReplyText(state);
		if (said) ctx.ui.notify(said, "info");
	}
	ui.steer = steer;
	pi.registerCommand("steer", {
		description: describe("steer"),
		getArgumentCompletions: (prefix: string) => steerCompletions(prefix, (cached()?.concepts ?? []).map((c) => c.name)),
		handler: steer,
	});

	// -- /scope: tracing, the dock, the guide -------------------------

	async function scope(args: string, ctx: any) {
		const a = parseScopeArgs(args);
		switch (a.cmd) {
			case "error":
				return ctx.ui.notify(a.error, "error");
			case "on":
				live.level = "on";
				return scopeStatus(ctx);
			case "off":
				live.level = "off";
				ui.hideDock();
				return scopeStatus(ctx);
			case "dock":
				if (!("mode" in a)) return dockStep(ui, ctx);
				const refused = ui.dockMode(a.mode, ctx);
				if (refused) return ctx.ui.notify(refused, "error");
				return ctx.ui.notify(a.mode === "mini" ? "dock: mini (m in the dock or /scope dock full for the whole dock)" : "dock: full", "info");
			case "jobs":
				return jobs.jobs(ctx);
			case "cancel":
				return jobs.cancel(ctx);
			case "help":
				pi.appendEntry("scope-guide", { v: 1, title: a.topic ? `scope help: ${a.topic}` : "metalscope help", lines: guideLines(a.topic, servedOf(ctx)) });
				return;
			case "status":
				return scopeStatus(ctx);
		}
	}

	/** Trace level and connection at once; the server's concepts and steering when it answers (scopeStatusText). */
	async function scopeStatus(ctx: any) {
		ui.updateUi(ctx);
		const head = { level: live.level, unreachable: live.connected ? undefined : unreachable(live.base, engineOf(ctx), modelShort(ctx)), metalscope: onMetalscope(ctx),
			steer: ui.steerStatus() };
		if (!head.metalscope) return ctx.ui.notify(scopeStatusText(head), "info");
		try {
			ctx.ui.notify(scopeStatusText(head, await conceptList(ctx)), "info");
		} catch {
			ctx.ui.notify(scopeStatusText(head), "info");
		}
	}

	pi.registerCommand("scope", {
		description: describe("scope"),
		getArgumentCompletions: (prefix: string) => scopeCompletions(prefix, helpTopics()),
		handler: (args, ctx) => scope(args, ctx),
	});

	// -- /concept: the concept library and its jobs ------------------------------------
	async function concept(args: string, ctx: any) {
		const a = parseConceptArgs(args);
		switch (a.cmd) {
			case "error":
				return ctx.ui.notify(a.error, "error");
			case "help":
				return appendHelp(pi, ctx, "concept", a.topic);
			case "learn":
				return notOnMetalscope(ctx, "it writes the corpus") ? undefined : jobs.learn(ctx, a.name, a.description, { corpus: a.corpus });
			case "calibrate":
				return notOnMetalscope(ctx, "it holds the conversations") ? undefined : jobs.calibrate(ctx, a.name);
		}
		if (notOnMetalscope(ctx)) return;
		const list = await listOrSay(ctx);
		if (!list) return;
		if (a.cmd === "list") return ctx.ui.notify(conceptListText(list, live.steer ? steerTag(live.steer) : undefined, backendOf(ctx)), "info");
		if (a.cmd === "overview") {
			const c = list.concepts.find((x) => x.name === a.name);
			if (!c) return ctx.ui.notify(notLearnedText(a.name, list, backendOf(ctx)), "error");
			pi.appendEntry("scope-concept", { v: 1, concept: c, rank: chatRank(list.concepts, c.name), backend: backendOf(ctx) });
			return;
		}
	}

	pi.registerCommand("concept", {
		description: describe("concept"),
		// A subcommand or a learned concept; after calibrate a learned one, after learn the library's.
		// With no list fetched for the selected model yet, it is asked for once a name could follow.
		getArgumentCompletions: (prefix: string) => {
			const have = cached();
			if (have || !selected || !prefix.trim()) return conceptCompletions(prefix, have);
			return quietly(conceptListOf(selected)).then((l) => conceptCompletions(prefix, l));
		},
		handler: (args, ctx) => concept(args, ctx),
	});

	// -- /jlens: the served model's Jacobian lens -----------------------------------------
	async function lens(args: string, ctx: any) {
		const a = parseLensArgs(args);
		if (a.cmd === "error") return ctx.ui.notify(a.error, "error");
		if (a.cmd === "help") return appendHelp(pi, ctx, "jlens");
		if (notOnMetalscope(ctx)) return;
		if (a.cmd === "learn") return jobs.lensLearn(ctx, a);
		if (a.cmd === "use") {
			let r;
			try {
				r = await api("/api/lens/use", { method: "POST", body: JSON.stringify({ model: modelShort(ctx), name: a.name }) });
			} catch (e: any) {
				return ctx.ui.notify(`/jlens use: ${requestFailed(e, live.base, ctx)}`, "error");
			}
			return ctx.ui.notify(lensUseText(r), "info");
		}
		let st;
		try {
			st = await lensStatus(modelShort(ctx));
		} catch (e: any) {
			return ctx.ui.notify(requestFailed(e, live.base, ctx), "error");
		}
		ctx.ui.notify(lensStatusLines(st).join("\n"), "info");
	}

	pi.registerCommand("jlens", {
		description: describe("jlens"),
		// The subcommands; after use, the selected model's lenses (asked for when /jlens has not fetched them).
		getArgumentCompletions: (prefix: string) => {
			if (!/^\s*use\s/i.test(prefix) || !selected) return lensCompletions(prefix, []);
			if (lensCache?.model === selected) return lensCompletions(prefix, lensCache.names);
			return quietly(lensStatus(selected)).then(() => lensCompletions(prefix, lensCache?.model === selected ? lensCache.names : []));
		},
		handler: (args, ctx) => lens(args, ctx),
	});
}

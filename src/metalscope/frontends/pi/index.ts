/**
 * metalscope for Pi: watch a local model's internals while it answers.
 *
 * - Registers the two backends as providers, "metalscope-omlx" and "metalscope-ds4", with every model of each
 *   whether its server runs or not (backends.ts: `metalscope status --json`). Picking one makes it the model served:
 *   the other backend's model is freed and this one's backend started (`metalscope ds4 NAME`, `metalscope omlx NAME`),
 *   begun on /model's pick and awaited before the request goes. With METALSCOPE_URL set, that one server instead
 *   (provider.ts registerUrl: a server pointed at by hand, or the mock).
 * - Tags Pi's requests so the server traces them, and follows the live token stream of the selected model's server
 *   (picking a model of the other server with /model moves the stream, the dock and the commands there).
 * - One place to look: the dock beside the chat (live; on or off).
 * - Four commands: /scope looks (and lists the jobs), /concept keeps the concept library, /steer changes what the model
 *   does, /jlens is the lens. Their forms are in help.ts (COMMANDS), which /scope help
 *   and /<command> help show.
 *
 * This file is Pi's entry point; it wires the modules into Pi. Each file's header says what it holds; the pure logic
 * is tested under test/.
 *
 * Install: `metalscope pi install` (`pi install`s this folder as a Pi package); a backend: `metalscope omlx install` or `metalscope ds4 install`.
 */

import { randomUUID } from "node:crypto";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { DS4, confirmSwitch, holdRequest, readCatalogue, registerCatalogue, runSwitch, servedCache, switchArgs, updateServers } from "./backends.ts";
import { registerCommands } from "./commands.ts";
import { registerEntries } from "./entries.ts";
import { type Ext, type Server, defaultServer, keptFirst, modelShort, nextBase, omlxServer, refreshFlags, registerHeaders, registerUrl, staleNotices, traceApi } from "./provider.ts";
import { registerJobs } from "./job-runner.ts";
import { registerReplay } from "./replay.ts";
import { type Live, makeFollower, switchServer } from "./state.ts";
import { createUi, registerShortcuts } from "./widget.ts";

export default async function metalscope(pi: ExtensionAPI) {
	const omlx = omlxServer(), url = process.env.METALSCOPE_URL;
	// The selected model's server (nextBase); before one is picked, the default server, else oMLX's (or the URL).
	const live: Live = { base: url?.replace(/\/$/, "") ?? omlx.base, clientId: `pi-${randomUUID().slice(0, 8)}`, level: "on", connected: false };
	const servers = new Map<string, Server>();
	const ext: Ext = { pi, live, api: traceApi(live), servers };
	// A reply started: its model is loaded, so the flags its server only knows then (oMLX's edit) are asked once.
	live.onStart = (model) => void refreshFlags(servers, live.base, model).then(() => live.requestRender?.(), () => {});
	// The live stream, on live.base: started at session start, moved when the selected model's server changes.
	const follower = makeFollower(live);

	// Both backends' models from metalscope's command line (every model, its server up or not); with METALSCOPE_URL,
	// that server's. Nothing registered is said once, at the first session start. Too slow at startup (oMLX loading a
	// large model): asked again then in the background, with longer to answer (lateCatalogue).
	let catalogue = url ? undefined : await readCatalogue();
	let unregistered: string | undefined;
	// Every server's replies through keptFirst: an edit's answer shows the reply up to the edit at once.
	const stream = keptFirst(live);
	if (catalogue) registerCatalogue(pi, catalogue, omlx, servers, readCatalogue, stream);
	else if (url) {
		if (!(await registerUrl(pi, url, servers, stream))) unregistered = `Start the server at METALSCOPE_URL (${url}), then /reload: no metalscope server with a chat model answers there`;
	}
	let late = !url && !catalogue, lateStop: AbortController | undefined;   // stopped at session shutdown: the next asks
	const lateCatalogue = async (ctx: any) => {
		late = false;
		const stop = (lateStop = new AbortController());
		const cat = await readCatalogue(undefined, 120_000, stop.signal);
		if (stop.signal.aborted) return;
		lateStop = undefined;
		try {
			if (!cat) {
				if (ctx.hasUI) ctx.ui.notify("In a terminal: metalscope pi install, then /reload. No metalscope model is registered: metalscope status --json did not answer (pi install records which metalscope Pi runs).", "warning");
				return;
			}
			registerCatalogue(pi, (catalogue = cat), omlx, servers, readCatalogue, stream);
			switchServer(live, nextBase(servers, ctx, live.base), follower);
			ui.updateUi(ctx);
			warnStale(ctx);
		} catch {}   // the session gone meanwhile
	};
	// A server running older code than is installed (status `stale`): a warning each, at session start.
	const warnStale = (ctx: any) => {
		if (ctx.hasUI) void staleNotices(servers.values()).then((lines) => lines.forEach((l) => ctx.ui.notify(l, "warning"))).catch(() => {});
	};
	live.base = defaultServer(servers.values())?.base ?? live.base;
	registerHeaders(ext);
	const ui = createUi(ext, undefined, () => follower.resubscribe());   // routing or lens top 10s while the dock shows EXPERTS or LENS

	// -- the backend switch ------------------------------------------------------------
	// The selected model made the one served: started when /model picks it (not when a session restores it, or a
	// cycle passes it), awaited before each request. One switch at a time, in order (a later pick waits for the
	// earlier one, whose command it then undoes). Its progress in the status line; a failure noted, and the request
	// stopped (holdRequest). While a job runs it is refused with the way out; while a reply runs on either server, it
	// asks first (confirmSwitch); a no leaves everything as it is and stops the request, and the next one asks again.
	let switching: { key: string; done: Promise<void> } | undefined;
	const served = servedCache();   // ds4's record read every request; a listing's yes kept until /model or a failed request
	const say = (ctx: any, text?: string) => {
		try {
			if (!ctx?.hasUI) return;
			ctx.ui.setStatus("metalscope-backend", text);
			ctx.ui.setWorkingMessage(text);
		} catch {}
	};
	const makeServed = (ctx: any, model = ctx?.model): Promise<void> | undefined => {
		const provider: string = model?.provider ?? "", id: string = model?.id ?? "";
		const args = catalogue ? switchArgs(provider, id) : undefined;
		if (!args) return undefined;
		const key = `${provider}/${id}`;
		if (switching?.key === key) return switching.done;
		const where = provider === DS4 ? "ds4" : "oMLX";
		const done = (switching?.done.catch(() => {}) ?? Promise.resolve()).then(async () => {
			if (await served.is(provider, id, omlx)) return;
			// A job or a reply in progress (another window's, the command line's) is cut short only on a yes.
			await confirmSwitch(ctx, id, where, servers.values());
			say(ctx, `starting ${id} on ${where}…`);
			try {
				await runSwitch(args, (line) => say(ctx, `${id} on ${where}: ${line.trim()}`));
			} finally {
				say(ctx, undefined);
			}
			const now = await readCatalogue();
			if (now) updateServers((catalogue = now), omlx, servers);
			if (!switchServer(live, nextBase(servers, ctx, live.base), follower) && follower.running) follower.restart();
			ui.updateUi(ctx);
		});
		switching = { key, done };
		const clear = () => {
			if (switching?.done === done) switching = undefined;
		};
		done.then(clear, (e) => {
			clear();
			try {
				if (ctx?.hasUI) ctx.ui.notify(e?.refused ? e.message : `${id} could not start on ${where}: ${e?.message ?? e}`, e?.refused ? "info" : "error");
			} catch {}
		});
		return done;
	};
	pi.on("before_provider_request", async (_event, ctx) => {
		const done = makeServed(ctx);
		if (done) await holdRequest(done, ctx);   // a refused switch stops the request
		return undefined;
	});
	// A failed request (the model stopped behind Pi's back): the next one checks again.
	pi.on("message_end", (event) => {
		const m = event.message as { role?: string; stopReason?: string };
		if (m.role === "assistant" && m.stopReason === "error") served.forget();
	});

	// -- lifecycle -------------------------------------------------------------------
	pi.on("session_start", (_event, ctx) => {
		switchServer(live, nextBase(servers, ctx, live.base), follower);   // the restored model's server
		follower.restart();                   // one follower per session
		ui.updateUi(ctx);
		ui.restoreDock(ctx);                  // open when it was open as Pi last closed
		ui.firstRunHint(ctx);                 // where to look, once per install
		if (unregistered && ctx.hasUI) ctx.ui.notify(unregistered, "warning");
		unregistered = undefined;
		warnStale(ctx);
		if (late) void lateCatalogue(ctx);
	});

	pi.on("session_shutdown", () => {
		if (lateStop) {
			lateStop.abort();
			lateStop = undefined;
			late = true;
		}
		ui.detach();                          // Pi hides the dock itself; its reserved columns go back
		follower.stop();
	});

	pi.on("model_select", (event, ctx) => {
		served.forget();
		if (event.source === "set") void makeServed(ctx, event.model)?.catch(() => {});   // loading while the prompt is typed
		// A steering vector belongs to one model: drop the steer when the model changes.
		if (live.steer && live.steer.model !== modelShort(ctx)) {
			ctx.ui.notify(`steer off: ${live.steer.parts.map((p) => p.name).join("+")} belongs to ${live.steer.model}`, "info");
			live.steer = undefined;
		}
		switchServer(live, nextBase(servers, ctx, live.base), follower);
		ui.updateUi(ctx);
	});

	// -- commands and entries, registered in the order Pi lists the commands -----------
	const jobs = registerJobs(ext, ui.steerLast);
	registerCommands(ext, ui, jobs, registerReplay(ext));
	registerShortcuts(ext, ui);
	registerEntries(pi);
}

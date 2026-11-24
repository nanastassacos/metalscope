/**
 * The UI outside the transcript: the status line, the steer's line above the editor (steerLine), the dock (a live side panel, the only live view; dock.ts draws it),
 * the keyboard shortcuts that open it (ctrl+\ dock, then stepped in, then closed; alt+i stepped in). Whether the dock is open is remembered across Pi restarts (prefsPath): open when Pi closed, it opens
 * again at session start; so are the mini dock (m) and STREAM's view. The first session after install says once where to look (FIRST_RUN_HINT).
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { HStack, visibleWidth } from "@earendil-works/pi-tui";
import { metalscopeHome } from "./backends.ts";
import { BOARD_MIN_WIDTH, DOCK_MIN_COLS, DOCK_MIN_ROWS, DOCK_MIN_WIDTH, MINI_WIDTH, type StreamView, miniAt, storedView } from "./board.ts";
import { Dock, type DockHost, type LensLayers } from "./dock.ts";
import { parseConcepts } from "./concept.ts";
import { LABELS, clean, cut } from "./draw.ts";
import { askAgain } from "./lookup.ts";
import { type Ext, type ModelFlags, capsOf, modelShort, onMetalscope, requestFailed } from "./provider.ts";
import type { Edited, Live } from "./state.ts";
import { type SteerState, type WordIds, steerHeader, steerTag, steeringText } from "./steer.ts";
import { type Examples, LINK_KEEP, type LinkOffer, type SteerList, keepOffer, linkLabel, linkTitle, linkedNote, parseExamples, parseSteering } from "./steerpage.ts";

/** What the commands need from the widget and the dock. */
export interface Ui {
	/** The status line (and the dock's resize watch), for the current level, model and dock; the steer's line (steerLine). */
	updateUi(ctx: any): void;
	/** Open the dock; `focus`: stepped in, still following the newest token (not where the dock cannot show: said why). */
	showDock(ctx: any, focus?: boolean): void;
	hideDock(): void;
	/** Is a dock open? */
	docked(): boolean;
	/** Is a dock open and drawn (the terminal is wide enough for it)? */
	dockVisible(): boolean;
	/** " · steer name@dose Lnn [+ …]" for /scope status, or "". */
	steerStatus(): string;
	/** The steer's line set again: Pi puts a widget set again last, nearest the editor (the job widget just appeared). */
	steerLast(ctx: any): void;
	/** At session start: open the dock when it was open the last time Pi closed. */
	restoreDock(ctx: any): void;
	/** At session start: the first-run hint, once per install (FIRST_RUN_HINT). */
	firstRunHint(ctx: any): void;
	/** At session shutdown: let go of the dock Pi is hiding (its reserved columns), remembering it as open. */
	detach(): void;
	/** /scope dock mini | full: the dock as lights and maps (m in the dock) or whole again; opened if it was closed. Refused
	 * (the reason returned) in a terminal under DOCK_MIN_COLS columns, and the full dock where it is too short. */
	dockMode(mode: "mini" | "full", ctx: any): string | undefined;
	/** /steer's handler (commands.ts sets it): the dock's Steer page steers through it, as /steer ARGS; the LENS word
	 * edit builds its steer through it (`ids`, `use`). */
	steer?(args: string, ctx: any, ids?: WordIds, use?: (s: SteerState) => void): Promise<void>;
	/** The concept and lens lists the completions hold (commands.ts sets it), dropped: a link changed them. */
	forgetLists?(): void;
}

/** What the extension remembers across Pi restarts: ~/.metalscope/pi.json (METALSCOPE_HOME's, as the server's data). */
export const prefsPath = (env: Record<string, string | undefined> = process.env, home = homedir()) => join(metalscopeHome(env, home), "pi.json");

export interface Prefs {
	dock?: boolean; // the dock was open when last left
	hinted?: boolean; // the first-run hint was shown
	mini?: boolean; // the mini dock (m, /scope dock mini)
	stream?: StreamView; // STREAM's view (s on it)
}

/** Said once, at the first session start with a UI after install: where the tracing shows. */
export const FIRST_RUN_HINT = "metalscope: replies are traced. ctrl+\\ opens the dock, /scope help the rest.";

/** The first-run hint, once: shown when `path` does not record it, then recorded. No UI (pi -p): nothing, nothing
 * recorded. A write that fails shows it again next time, which is harmless. */
export function firstRunHint(ctx: any, path = prefsPath()): void {
	if (!ctx?.hasUI || readPrefs(path).hinted) return;
	writePrefs({ hinted: true }, path);
	ctx.ui.notify(FIRST_RUN_HINT, "info");
}

export function readPrefs(path = prefsPath()): Prefs {
	try {
		const p = JSON.parse(readFileSync(path, "utf8"));
		return p && typeof p === "object" && !Array.isArray(p) ? p : {};
	} catch {
		return {};
	}
}

/** Merge `changes` into the file; a write that fails is not worth a message (it only costs the dock's reopening). */
export function writePrefs(changes: Prefs, path = prefsPath()): void {
	try {
		mkdirSync(dirname(path), { recursive: true });
		writeFileSync(path, JSON.stringify({ ...readPrefs(path), ...changes }));
	} catch {}
}

/** The dock's size: Pi draws it as an overlay `DOCK_WIDTH` of the terminal wide (at least board.ts DOCK_MIN_WIDTH
 * columns; the board's BOARD_MIN_WIDTH where the terminal spares it and leaves Pi PI_MIN_COLS), from `DOCK_MIN_COLS`
 * columns. The mini dock (board.ts MINI_WIDTH, lights, MAP, SETTLING and STREAM) is that many columns, chosen (m) or in
 * its place in a terminal too short for the full one (miniAt: `mini` below takes the rows in), from `DOCK_MIN_COLS` too. */
export { DOCK_MIN_COLS };
const DOCK_WIDTH = 0.36, PI_MIN_COLS = 80;

/** The full dock's least width at a terminal `cols` wide: the board's from BOARD_MIN_WIDTH + its gap + PI_MIN_COLS
 * (181 columns), else DOCK_MIN_WIDTH. */
const fullMin = (cols: number) => (cols >= BOARD_MIN_WIDTH + 1 + PI_MIN_COLS ? BOARD_MIN_WIDTH : DOCK_MIN_WIDTH);

/** The columns the dock takes at a terminal `cols` wide (0 when it is hidden there): its width the way Pi resolves
 * the overlay's (floor of the percentage, then the minimum; the mini dock's MINI_WIDTH), plus one column of gap. */
export function dockColumns(cols: number, mini = false): number {
	if (cols < DOCK_MIN_COLS) return 0;
	return (mini ? MINI_WIDTH : Math.max(Math.floor(cols * DOCK_WIDTH), fullMin(cols))) + 1;
}

/** The dock overlay's options for Pi, read at every frame (Pi keeps this object and resolves its width each render):
 * the full dock's or the mini one's, as `mini()` says, at the terminal's columns now (`cols`; 0 not known; as
 * dockColumns). */
export function dockOverlay(mini: () => boolean, focus: boolean, cols: () => number = () => 0) {
	return {
		anchor: "top-right" as const,
		get width() { return mini() ? MINI_WIDTH : `${DOCK_WIDTH * 100}%`; },
		get minWidth() { return mini() ? MINI_WIDTH : fullMin(cols()); },
		visible: (cols: number) => cols >= DOCK_MIN_COLS,
		nonCapturing: !focus,
	};
}

/** Room for the dock beside Pi's own content (transcript, editor, footer), so the reply reflows next to the dock
 * instead of being drawn under it; `fit(cols)` follows a resize, `undo()` gives the room back.
 *
 * Pi draws the dock as an overlay on top of its content. In fullscreen mode (Pi's default) the content is a layout
 * tree under a root set with `setLayoutRoot`: the room is that root in an HStack beside an empty column the dock's
 * width. In regular mode the TUI renders its children with `render(width)` and composites overlays afterwards: an
 * instance `render` that passes a smaller width. A TUI with neither keeps the plain overlay. */
export interface Room {
	fit(cols: number): void;
	undo(): void;
}

// What reserveColumns leaves on the TUI is tagged with this, so a later one (another extension instance after /reload,
// /new or /resume, whose dock Pi hid without its close running) takes the earlier room off instead of adding a second.
const ROOM = Symbol.for("metalscope.dockRoom");

const NO_ROOM: Room = { fit() {}, undo() {} };

export function reserveColumns(tui: any, mini: () => boolean = () => false): Room {
	if (!tui) return NO_ROOM;
	let root = tui.layoutRoot;
	if (typeof tui.setLayoutRoot === "function" && root) {
		if (root[ROOM]) root = root[ROOM];                                      // an earlier room's: unwrap it
		const gap = { render: () => [] as string[], invalidate() {} };
		let at = -1;
		const fit = (cols: number) => {
			const want = dockColumns(cols, mini());
			if (want === at) return;
			at = want;
			const stack: any = want ? new HStack([{ component: root, grow: 1, shrink: 1 }, { component: gap, basis: want, grow: 0, shrink: 0 }]) : root;
			if (want) stack[ROOM] = root;
			tui.setLayoutRoot(stack);
			tui.requestRender?.(true);
		};
		fit(tui.terminal?.columns ?? 0);
		return { fit, undo: () => { tui.setLayoutRoot(root); tui.requestRender?.(true); } };
	}
	if (typeof tui.render !== "function") return NO_ROOM;
	if (Object.prototype.hasOwnProperty.call(tui, "render")) {
		if (!tui.render[ROOM]) return NO_ROOM;                                     // someone else's override: leave it
		delete tui.render;                                                     // an earlier room's
	}
	const render = tui.render;
	tui.render = function (width: number) {
		return render.call(this, width - dockColumns(width, mini()));
	};
	tui.render[ROOM] = true;
	tui.requestRender?.(true);
	return {
		fit() {},
		undo: () => {
			delete tui.render;
			tui.requestRender?.(true);
		},
	};
}

/** The line above the editor while a steer is on: " ◆ steering pirate@0.4 L16 · 12 of 18 good in chat · /steer off",
 * the chat figure only for one concept tried at its dose (SteerPart.chat). Cut to `width`, never wrapped: narrow, the
 * figure goes first, then /steer off. */
export function steerLine(th: { fg(c: string, s: string): string }, s: SteerState, width: number): string {
	const chat = s.parts.length === 1 ? s.parts[0].chat : undefined;
	const head = steeringText(s), room = width - 3;
	if (room < 1) return "";
	let tail = [...(chat ? [`${chat.good} of ${chat.turns} good in chat`] : []), "/steer off"].map((t) => ` · ${t}`);
	while (tail.length && visibleWidth(head + tail.join("")) > room) tail = tail.slice(1);
	return ` ${th.fg("accent", "◆")} ${cut(head, room)}${tail.length ? th.fg("dim", tail.join("")) : ""}`;
}

/** `resubscribe`: the live stream asks again for what Live says (state.ts liveUrl), called when expert routing, the
 * lens top 10s or the writes are wanted or no longer (askLive). */
export function createUi({ pi, live, api, servers }: Ext, prefs = prefsPath(), resubscribe: () => void = () => {}): Ui {
	const opened: { dockCtx?: any } = {};   // the ctx the dock was opened with (ctx.model follows /model): the Steer page's model and /steer's
	const ui: Ui = { updateUi: (ctx: any) => life.updateUi(ctx), showDock: (ctx: any, focus?: boolean) => life.showDock(ctx, focus), hideDock: () => life.hideDock(),
		docked: () => life.dock !== undefined, dockVisible: () => life.dock !== undefined && life.dockShown, steerStatus: () => life.steerStatus(),
		steerLast: (ctx: any) => life.steerLast(ctx), restoreDock: (ctx: any) => life.restoreDock(ctx), firstRunHint: (ctx: any) => firstRunHint(ctx, prefs),
		detach: () => life.detach(), dockMode: (mode: "mini" | "full", ctx: any) => life.dockMode(mode, ctx) };
	const calls = modelCalls(live, api, servers, opened, ui), { served, flags, caps, statusNow } = calls;
	const steer = steerCalls(live, api, opened, ui, calls), trace = traceCalls(pi, live, api, opened, calls);
	const dockHost: DockHost = {
		live,
		captions: false,
		close: () => life.hideDock(),
		unfocus: () => life.dockHandle?.unfocus(),
		mini: readPrefs(prefs).mini === true,
		stream: storedView(readPrefs(prefs).stream),   // unset until chosen
		layoutChanged: () => life.layoutChanged(),
		roomChanged: () => queueMicrotask(() => life.askLive()),   // not while rendering
		steerModel: served,
		now: () => performance.now(),   // the dock's redraw cap
		...steer.host,
		flags,
		caps,
		freshCaps: async () => capsOf(await statusNow()),
		...trace.host,
		looked: new Map(),
	};
	const life = new DockLife(live, dockHost, prefs, resubscribe, opened, steer.chats, trace.asked);
	return ui;
}

type ModelCalls = ReturnType<typeof modelCalls>;

/** The selected model's server as the dock's calls reach it (`opened.dockCtx`: the ctx the dock was opened with): the
 * served model, its flags and caps, its status asked for now, a failed request in one line, and /steer run for the dock
 * through `ui.steer` (commands.ts sets it). */
function modelCalls(live: Live, api: Ext["api"], servers: Ext["servers"], opened: { dockCtx?: any }, ui: Ui) {
	// The served model (a ctx gone stale with its session answers nothing); a failed request in one line.
	const served = () => {
		try {
			return opened.dockCtx ? modelShort(opened.dockCtx) : "";
		} catch {
			return "";
		}
	};
	// The selected model's flags as its server reports them (status `chat`: tested, edit); none for a model it has not loaded.
	const flags = (): ModelFlags => {
		try {
			const m = opened.dockCtx?.model;
			return (m && servers?.get(m.provider)?.status.chat?.[m.id]) || {};
		} catch {
			return {};
		}
	};
	// The selected model's server's caps as it reported them (provider.ts refreshFlags keeps them current). Unknown (an
	// entry from `metalscope status`, which lists none, or one replaced after a switch): asked for once an entry
	// (statusNow), the dock drawn again with them.
	let capsAsked: object | undefined;
	const caps = (): readonly string[] => {
		try {
			const m = opened.dockCtx?.model, s = m && servers?.get(m.provider);
			if (s && s.status.caps === undefined && capsAsked !== s.status) {
				capsAsked = s.status;
				void statusNow().then(() => live.requestRender?.(), () => {});
			}
			return s?.status.caps ?? [];
		} catch {
			return [];
		}
	};
	// GET /scope/api/status now (the LENS edit, g): the selected model's flags and its server's caps kept as it says them,
	// so the keys follow (a model loaded or switched since they were last read). Kept in the entry asked for, only while
	// it is still that provider's and the server asked is its own (an answer after /model changes nothing).
	const statusNow = async (): Promise<any> => {
		const m = opened.dockCtx?.model, s = m && servers?.get(m.provider), base = live.base;
		const st = await api("/api/status");
		if (s && st && servers?.get(m.provider) === s && s.base === base) {
			const mine: ModelFlags | undefined = st.chat?.[m.id], had = s.status.chat?.[m.id];
			if (mine && had) Object.assign(had, mine);
			s.status.caps = capsOf(st);
		}
		return st;
	};
	const failed = (e: any) => {
		let model: any;
		try {
			model = opened.dockCtx?.model;
		} catch {}
		return requestFailed(e, live.base, { model: model && { id: model.id, provider: model.provider } }).split("\n")[0];
	};
	// /steer ARGS as the command runs it, with what it says recorded for the dock (and shown as it always is).
	async function viaSteer(args: string, ids?: WordIds, use?: (s: SteerState) => void) {
		const said: { text: string; level: string }[] = [];
		if (!ui.steer || !opened.dockCtx) return [{ text: "steering is not ready: use /steer", level: "error" }];
		const real = opened.dockCtx, notify = (text: string, level = "info") => {
			said.push({ text, level });
			real.ui.notify(text, level);
		};
		const uiProxy = new Proxy(real.ui, { get: (t, k) => (k === "notify" ? notify : typeof t[k] === "function" ? t[k].bind(t) : t[k]) });
		const ctx = new Proxy(real, { get: (t, k) => (k === "ui" ? uiProxy : typeof t[k] === "function" ? t[k].bind(t) : t[k]) });
		await ui.steer(args, ctx, ids, use);
		return said;
	}
	return { served, flags, caps, statusNow, failed, viaSteer };
}

/** The Steer page's calls to the server (the dock's host: its concepts, test chats, steer, delete, the link offer and
 * the link), and the test chats asked for this session (`chats`: detach drops them). */
function steerCalls(live: Live, api: Ext["api"], opened: { dockCtx?: any }, ui: Ui, { served, failed, viaSteer }: ModelCalls) {
	const chats = new Map<string, Promise<Examples | { error: string }>>();   // the Steer page's test chats, for the session
	let offer: { model: string; got: Promise<any> } | undefined;   // the served model's link offer, asked once (dropped by another model, a link, a failure)
	const LINK_TIMEOUT_MS = 120_000;   // a link reads the J-lens again (2.5 GB on Flash Next, tens of seconds under memory pressure)
	const host: Pick<DockHost, "steering" | "steerChats" | "steer" | "deleteConcept" | "linkOffer" | "link"> = {
		// The Steer page's concepts, with the layer /steer pushes each at (the concept list's, as /steer resolves it).
		steering: async (): Promise<SteerList | { error: string }> => {
			const model = served();
			const q = encodeURIComponent(model);
			try {
				const [steering, concepts] = await Promise.all([api(`/api/steering?model=${q}&served=${q}`), api(`/api/concepts?model=${q}`).catch(() => null)]);
				// From the concept list, as /steer reads it: each one's steer layer.
				const layers: Record<string, number | null> = {};
				if (concepts) for (const c of parseConcepts(concepts)) layers[c.name] = c.reader?.layer ?? null;
				return parseSteering(steering, model, layers);
			} catch (e: any) {
				return { error: failed(e) };
			}
		},
		// A concept's test chats at `dose` (the server's: its picked dose), once a session: 404 where that dose has none.
		steerChats: (name: string, dose?: number) => {
			const model = served();
			const key = `${model}\u0000${name}\u0000${dose ?? ""}`;
			let got = chats.get(key);
			if (!got) {
				const at = dose === undefined ? "" : `&dose=${dose}`;
				got = api(`/api/steering/${encodeURIComponent(name)}/chats?model=${encodeURIComponent(model)}${at}`).then(parseExamples, (e: any) => {
					chats.delete(key);   // asked again next time
					return { error: failed(e) };
				});
				chats.set(key, got);
			}
			return got;
		},
		// The Steer page's note: a steer set, what is on (the state /steer built; with the server's warning, at its level),
		// never the chat's words; an error said wins whatever is on, and anything else (steer: off) is as /steer said it.
		steer: async (args: string) => {
			const was = live.steer, said = await viaSteer(args), now = live.steer;
			if (said.some((s) => s.level === "error" || s.level === "warning") || !now || now === was) return said;
			return [{ text: steeringText(now, true), level: now.note ? "caution" : "info" }];
		},
		// d, y on the Steer page: steering turned off first where it uses the concept (/steer off), then its data on the
		// checkpoint moved to the trash (POST /scope/api/concept/delete); its test chats asked for again.
		deleteConcept: async (name: string) => {
			if (live.steer?.parts.some((p) => p.name === name)) {
				const off = await viaSteer("off");
				if (off.some((s) => s.level === "error")) return off;
			}
			let res: any;
			try {
				res = await api("/api/concept/delete", { method: "POST", body: JSON.stringify({ name, model: served() }) });
			} catch (e: any) {
				return [{ text: typeof e?.status === "number" ? clean(String(e.message)) : failed(e), level: "error" }];   // a refusal's detail as it is
			}
			for (const k of [...chats.keys()]) if (k.split("\u0000")[1] === name) chats.delete(k);
			// A description no model has: that alone went.
			return [{ text: res?.definition === true ? `${name}'s description deleted` : `${name} deleted on ${res?.checkpoint ?? served()}: moved to the trash`, level: "info" }];
		},
		// The served model's link offer, kept for the session once it lasts (keepOffer); another model's dropped; one that
		// may change (no shape yet: not loaded; nothing yet) or a failure asked for again next time.
		linkOffer: () => {
			const model = served(), had = offer;
			if (had && had.model === model) return had.got;
			const drop = () => void (offer?.got === got && (offer = undefined));
			const got: Promise<any> = api(`/api/model/link?model=${encodeURIComponent(model)}`).then((json) => (keepOffer(json) || drop(), json), () => (drop(), null));
			offer = { model, got };
			return got;
		},
		// l on the Steer page: Pi's pick list, kept separate first (enter's answer, esc too); a candidate picked, the model
		// the offer named linked to it (POST /scope/api/model/link, with LINK_TIMEOUT_MS: it reads the J-lens again). Any
		// answer drops the offer (an error too: the link may have gone through); a refusal's detail as it is (400, 409,
		// 500), no answer said as such; linked, the completions' lists dropped.
		link: async (o: LinkOffer) => {
			if (!o.model) return undefined;
			const options = [LINK_KEEP, ...o.candidates.map(linkLabel)];
			const c = o.candidates[options.indexOf(await opened.dockCtx?.ui.select(linkTitle(o.model), options)) - 1];
			if (!c) return undefined;
			let res: any;
			try {
				res = await api("/api/model/link", { method: "POST", body: JSON.stringify({ model: o.model, checkpoint: c.name }), signal: AbortSignal.timeout(LINK_TIMEOUT_MS) });
			} catch (e: any) {
				offer = undefined;
				return [{ text: typeof e?.status === "number" ? clean(String(e.message)) : LABELS.linkNoAnswer, level: "error" }];
			}
			offer = undefined;
			ui.forgetLists?.();
			return [{ text: linkedNote(res, c), level: "info" }];
		},
	};
	return { chats, host };
}

/** The dock's calls on a reply's trace (the dock's host): engram inspection and its stop (`asked`: each ask's server,
 * until it ends; detach forgets them), the LENS word edit, LENS's - +. */
function traceCalls(pi: Ext["pi"], live: Live, api: Ext["api"], opened: { dockCtx?: any }, { served, failed, statusNow, viaSteer }: ModelCalls) {
	const asked = new Map<string, string>();   // g's readings under way: the server each ask went to, by ask (its stop's)
	const host: Pick<DockHost, "lookup" | "stopLookup" | "edit" | "lensLayers"> = {
		// g: engram inspection on a saved reply (two reads of the whole conversation: up to minutes on a long one, asked
		// again past fetch's limit: askAgain, never once stopped), its progress to this Pi's live stream; a refusal's
		// detail as it is. Its stop, to the server asked (`asked`, by ask while it runs): its answer not read.
		lookup: (trace: string, ask: string) => {
			const base = live.base;
			asked.set(ask, base);
			return askAgain(async (signal, again) => {
				if (again && !asked.has(ask)) throw new Error(LABELS.engramStopped);
				return api("/api/lookup", { method: "POST", body: JSON.stringify({ trace, client: live.clientId, ask, ...(again && { again }) }), signal }, base);
			}).catch((e: any) => { throw new Error(typeof e?.status === "number" ? e.message : failed(e)); })
				.finally(() => asked.delete(ask));
		},
		stopLookup: async (trace: string, ask: string) => {
			const base = asked.get(ask);
			asked.delete(ask);
			if (base !== undefined) await api("/api/lookup", { method: "POST", body: JSON.stringify({ trace, ask, stop: true }) }, base).catch(() => {});
		},
		// The LENS word edit, where the server answers a reply again from one token (status `chat`: the model's edit;
		// not known yet: the server's edit_at cap): the edit's steer built as /steer builds it, sent with the last message
		// answered again (/steer again: a command's ctx can navigate the session; provider.ts adds it to that request).
		edit: async (args: string, ids: WordIds, at: { rid: string; at: number; word: string; want?: string }) => {
			let st: any;
			try {
				st = await statusNow();   // the server's now (a model loaded since /model listed it): the key follows
			} catch (e: any) {
				return [{ text: failed(e), level: "error" }];
			}
			const mine: ModelFlags | undefined = st?.chat?.[opened.dockCtx?.model?.id];
			if (!(mine?.edit ?? capsOf(st).includes("edit_at"))) return [{ text: LABELS.editHere, level: "error" }];
			let edit: Edited | undefined;
			const said = await viaSteer(args, ids, (s) => void (edit = { ...at, steer: steerHeader(s), model: served() }));
			if (!edit) return said.some((s) => s.level === "error") ? said : [...said, { text: LABELS.editNot, level: "error" }];
			live.edit = edit;
			pi.sendUserMessage("/steer again", { expandPromptTemplates: true });
			return said;
		},
		// LENS's - +: the server's lens steps for the served model; `step` sets the one in use.
		lensLayers: async (step?: number): Promise<LensLayers | { error: string; missing?: boolean }> => {
			const q = `?model=${encodeURIComponent(served())}`;
			try {
				return await (step === undefined ? api(`/api/lens/layers${q}`)
					: api("/api/lens/layers", { method: "POST", body: JSON.stringify({ model: served(), step }) }));
			} catch (e: any) {
				return { error: failed(e), missing: e?.status === 404 };   // 404: a server without the route (the mock)
			}
		},
	};
	return { asked, host };
}

/** The dock's lifecycle and the status line: the dock opened, closed and detached, its room beside Pi's content and
 * the resize watch, the stream asked for what it shows, the status line and the steer's line above the editor. */
class DockLife {
	// Whether the open dock fits the terminal (it is hidden below DOCK_MIN_COLS columns), and whether it is the mini
	// dock (miniAt); the status line says which.
	dockShown = true;
	private dockMini = false;
	private watchTui: any;
	private statusCtx: any;
	// The terminal's size: Pi's TUI once a dock has had one, before that (a session's first dock) Pi's own stdout.
	private termCols = (): number | undefined => (this.watchTui ?? this.dockTui)?.terminal?.columns ?? process.stdout.columns;
	private termRows = (): number | undefined => (this.watchTui ?? this.dockTui)?.terminal?.rows ?? process.stdout.rows;
	/** The mini dock chosen, or the terminal too short for the full one (dockColumns, dockOverlay). */
	private miniNow = () => miniAt(this.dockHost.mini, this.termRows());
	private noteCols(cols: number) {
		const shown = cols >= DOCK_MIN_COLS, mini = this.miniNow();
		if (shown === this.dockShown && mini === this.dockMini) return;
		this.dockShown = shown;
		this.dockMini = mini;
		const ctx = this.statusCtx;
		if (ctx) queueMicrotask(() => this.updateUi(ctx));          // not while rendering
		queueMicrotask(() => this.askLive());                   // a hidden dock draws nothing: no routing for it
	}
	/** Why the dock cannot show (the terminal's columns known and under DOCK_MIN_COLS), else undefined. */
	private narrow = (): string | undefined => {
		const cols = this.termCols();
		return cols !== undefined && cols < DOCK_MIN_COLS ? `dock needs ${DOCK_MIN_COLS} columns, has ${cols}` : undefined;
	};
	private dockStatus = () => (this.dockShown ? ` · docked${this.miniNow() ? " (mini)" : ""}` : ` · dock hidden (needs ${DOCK_MIN_COLS} columns)`);

	// The dock is the only live view. An empty component above the editor, there while the dock is open, is how a
	// resize reaches it: Pi renders it every frame, so the dock's reserved column follows the terminal's width and the
	// status line says when the dock is hidden for want of columns. It draws nothing.
	private watchResize(ctx: any) {
		ctx.ui.setWidget("metalscope", (tui: any) => {
			this.watchTui = tui;
			return {
				render: (width: number) => {
					const cols = this.watchTui?.terminal?.columns ?? width;
					this.noteCols(cols);
					this.dockRoom.fit(cols);
					return [];
				},
				invalidate() {},
			};
		});
	}

	steerStatus = () => (this.live.steer ? ` · ${steerTag(this.live.steer)}` : "");
	private steerShown = false;   // the steer's line set (Pi keeps it)

	updateUi(ctx: any) {
		this.dock?.invalidate();   // the level, model, steer or dock changed: the dock draws them too
		if (!ctx.hasUI) return;
		this.statusCtx = ctx;
		if (this.dockOn) this.watchResize(ctx);
		else ctx.ui.setWidget("metalscope", undefined);
		// The steer's line, there while one is on whoever set it (each path that changes live.steer comes here), drawn from
		// live.steer at every frame; set only as it comes or goes (no flicker, no re-creation). Pi draws it from a timer with
		// nothing around it: a throw there would end Pi, so a steer it cannot draw is no line.
		if (!!this.live.steer !== this.steerShown) {
			this.steerShown = !!this.live.steer;
			ctx.ui.setWidget("metalscope-steer", !this.steerShown ? undefined : (_tui: any, theme: any) => ({
				render: (w: number) => {
					try {
						return this.live.steer ? [steerLine(theme, this.live.steer, w)] : [];
					} catch {
						return [];
					}
				},
				invalidate() {},
			}));
		}
		ctx.ui.setStatus("metalscope", !onMetalscope(ctx) ? undefined : `metalscope: ${this.live.level}${this.dockOn ? this.dockStatus() : ""}`);
	}

	// -- dock: a live side panel with its own background ----------------------------
	private dockOn = false;
	dockHandle: any;
	dock: Dock | undefined;
	private dockClose: (() => void) | undefined;
	private dockRoom = NO_ROOM;
	private dockTui: any;   // the open dock's TUI (its terminal's columns when the resize watch has not drawn yet)
	private dockSeq = 0;   // the open dock's number: a close that arrives for an earlier one (detached) changes nothing
	constructor(private live: Live, private dockHost: DockHost, private prefs: string, private resubscribe: () => void, private opened: { dockCtx?: any },
		private chats: { clear(): void }, private asked: { clear(): void }) {}

	// m and STREAM's view: remembered; the stream asks for what is shown; the room beside Pi's content follows the width.
	layoutChanged() {
		writePrefs({ mini: !!this.dockHost.mini, ...(this.dockHost.stream && { stream: this.dockHost.stream }) }, this.prefs);
		this.askLive();
		const cols = this.termCols();
		if (typeof cols === "number") {
			this.noteCols(cols);
			this.dockRoom.fit(cols);
		}
		if (this.statusCtx) this.updateUi(this.statusCtx);
		this.live.requestRender?.();
	}

	// What the stream adds only while it is drawn, the dock open and shown (DOCK_MIN_COLS): expert routing, each read
	// layer's top 10 (Flash Next, measured 2026-11-11: a token event 1.2 KB, 4.5 KB with routing, 7.5 KB with the top
	// 10s, 10.7 KB with both; Pi takes it in 36 µs). Routing and the top 10s while the board, one-panel too, or the mini
	// dock is shown (EXPERTS' lights, LENS's telescope and SETTLING), on a model without experts too (dock.ts
	// noExperts), so a model that has experts after all is seen. The writes while the board or the mini dock has room
	// for STREAM (Dock.streamRoom: not the one-panel board or the Steer page, tall enough), on a model that
	// recorded none too, for the same reason.
	askLive() {
		const shown = this.dockOn && this.dockShown;
		const writes = shown && (this.dock?.streamRoom ?? true);
		if (!!this.live.experts === shown && !!this.live.lens === shown && !!this.live.writes === writes) return;
		this.live.experts = this.live.lens = shown;
		this.live.writes = writes;
		this.resubscribe();
	}

	showDock(ctx: any, focus = false) {
		if (!ctx.hasUI) return;
		// Stepping in where the dock cannot show: Pi gives a hidden overlay no keys. Said why instead.
		const why = focus && this.narrow();
		if (why) return void ctx.ui.notify(why, "warning");
		this.opened.dockCtx = ctx;
		if (this.dock) {
			if (focus) {
				if (!this.dock.stepped) this.dock.stepIn(-1);   // already in: a token paused on stays
				this.dockHandle?.focus();
			}
			return;
		}
		this.dockOn = true;
		this.live.dense = undefined;   // a model taken for one without experts is tried again
		this.askLive();
		const mine = ++this.dockSeq;
		writePrefs({ dock: true }, this.prefs);
		void ctx.ui.custom(
			(tui: any, theme: any, _kb: any, done: () => void) => {
				this.dock = new Dock(tui, theme, this.dockHost);
				this.dockClose = () => done();
				this.dockRoom = reserveColumns(tui, this.miniNow);
				this.dockTui = tui;
				this.live.requestRender = (progress?: boolean) => {
					if (!progress) this.dock?.invalidate();   // anything but the stream's progress: drawn at once
					tui.requestRender();
				};
				if (focus) {
					this.dock.closeOnEsc = true;
					this.dock.stepIn(-1);
				}
				return this.dock;
			},
			{
				overlay: true,
				overlayOptions: dockOverlay(this.miniNow, focus, () => this.termCols() ?? 0),
				onHandle: (h: any) => (this.dockHandle = h),
			},
		).then(() => {
			if (mine !== this.dockSeq) return;
			this.forget();
			this.updateUi(ctx);
		});
		this.updateUi(ctx);
	}

	// Closed by the user (ctrl+\, Esc while inspecting, /scope dock): remembered. Pi tearing the UI down at exit does
	// not come through here, so a dock open then opens again next time.
	hideDock() {
		if (this.dock) writePrefs({ dock: false }, this.prefs);
		this.dockClose?.();
	}

	restoreDock(ctx: any) {
		if (ctx.hasUI && !this.dock && readPrefs(this.prefs).dock) this.showDock(ctx);
	}

	// The dock gone (closed, or detached): its reserved columns given back.
	private forget() {
		this.dock?.dispose();
		this.dockRoom.undo();
		this.dockRoom = NO_ROOM;
		this.dock = this.dockHandle = this.dockClose = this.dockTui = undefined;
		this.live.requestRender = undefined;   // nothing drawn: tokens no longer redraw Pi
		this.dockOn = false;
		this.askLive();
	}

	// At session shutdown (/new, /resume, /reload, exit): Pi hides the dock's overlay itself without running its
	// close, so forget the dock here, the preference untouched (it reopens).
	detach() {
		this.dockSeq++;
		this.forget();
		this.opened.dockCtx = undefined;   // stale with its session
		this.chats.clear();
		for (const [trace, x] of this.dockHost.looked!) if (x.ask) void this.dockHost.stopLookup!(trace, x.ask);   // the server's one slot freed
		this.asked.clear();   // not asked again past fetch's limit (askAgain)
		this.dockHost.looked!.clear();
		this.dockHost.lookupOn = undefined;
	}

	// /scope dock mini | full: the mode set (and remembered), the dock opened if it was closed.
	dockMode(mode: "mini" | "full", ctx: any): string | undefined {
		const rows = this.termRows(), why = this.narrow();
		if (why) return why;   // refused: nothing would show
		if (mode === "full" && miniAt(false, rows)) {   // refused; a closed dock opens as the mini one there
			const open = !!this.dock;
			if (!open) this.showDock(ctx);
			return `the full dock needs ${DOCK_MIN_ROWS} rows; this terminal has ${rows}: the mini dock ${open ? "stays" : "opens"}`;
		}
		this.dockHost.mini = mode === "mini";
		this.dockHost.layoutChanged?.();
		if (!this.dock) this.showDock(ctx);
	}
	steerLast = (ctx: any) => {
		if (!this.steerShown) return;
		this.steerShown = false;
		this.updateUi(ctx);
	};
}

/** One step of ctrl+\ and /scope dock: open the dock; open, step in (no reply yet too: the Steer page is there); else
 * close it. A dock too narrow to show: close. */
export function dockStep(ui: Ui, ctx: any) {
	if (!ui.docked()) return ui.showDock(ctx);
	if (ui.dockVisible()) return ui.showDock(ctx, true);
	ui.hideDock();
}

/** ctrl+\ steps through dock, stepped in, closed (as /scope dock does); alt+i steps straight in. (The job's stop key is
 * raw input: job-runner.ts.) */
export function registerShortcuts({ pi }: Ext, ui: Ui) {
	// One key for both: on a Mac, Option+i types ˆ unless the terminal sends Option as Alt, and every other
	// control key a legacy terminal can send is Pi's. Pi runs this only while its editor has the keyboard; once
	// stepped in, the dock gets ctrl+\ itself and closes.
	pi.registerShortcut("ctrl+\\", {
		description: "metalscope: open the trace dock; again to step in; again to close (/scope dock)",
		handler: async (ctx) => dockStep(ui, ctx),
	});
	// alt+i: step into the dock straight away. ctrl+] would be Pi's editor jump-forward.
	pi.registerShortcut("alt+i", {
		description: "metalscope: step into the dock",
		handler: async (ctx) => ui.showDock(ctx, true),
	});
}

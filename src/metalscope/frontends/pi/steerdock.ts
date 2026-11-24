/**
 * The Steer page (6 on the board): the served model's concepts, the cursor's dose curve, its test chats, the link offer
 * and the page's note; its keys, its requests through the host (widget.ts) and its drawing. The Dock (dock.ts) holds one
 * and hands it the keys and the room while the page is shown; the data and its shapes are steerpage.ts'.
 */

import { matchesKey, wrapTextWithAnsi } from "@earendil-works/pi-tui";
import { STEER_KEY, STEER_ROWS, examplesPanel, rule as ruleLine, steerColumns, steerKeys, steerPanel, wrapKeys } from "./board.ts";
import type { DockHost } from "./dock.ts";
import { type DockTheme, EXPLAIN, cut, fit, noteLines, padAbove } from "./draw.ts";
import type { Grids } from "./lights.ts";
import type { Reply } from "./state.ts";
import { type Examples, type LinkOffer, type Note, type SteerConcept, type SteerList, alreadyLinked, fmt, linkedTo, mdLines, noteOf, parseOffer, pickIndex,
	reading } from "./steerpage.ts";

/** What the Steer page may ask of the Dock: its state that the page reads or sets (the board's note, the page shown,
 * the one-panel board, the dock gone), the frame and the keys it hands back, and what a link drops (EXPERTS' grids and
 * tallies, the lens steps asked for again). */
export interface SteerDock {
	readonly disposed: boolean;
	readonly one: boolean;
	boardNote: string;
	steerPage: boolean;
	served(): string;
	frame(body: string[], width: number, height: number, footer: string[]): string[];
	leave(): boolean;
	setMini(on: boolean): void;
	boardKey(data: string): void;
	grids: Pick<Grids, "clear">;
	xpData: Map<string, any>;
	xpLast: Map<string, any>;
	askDepth(): void;
}

export class SteerPage {
	constructor(private host: DockHost, private theme: DockTheme, private tui: { readonly terminal?: any; requestRender(): void }, private dock: SteerDock) {}
	// The Steer page, for the served model `model` (another model clears it all: resetSteer): its concepts (asked for
	// when the dock opens, when the page is entered and for another model), the cursor's concept, each concept's dose
	// under the cursor (the dose itself, so a refetch with other doses cannot leave an index past them), the test
	// chats' turn per concept and scroll; the last apply's note, one apply at a time and the one queued after it. The
	// model's link offer (undefined: not answered; null: none), a link under way, and a refused link's note (`refused`:
	// "reload" until the reload it asked for answers, then "shown": cleared by the next l, leaving the page or the offer
	// asked for again).
	steer: { model?: string; list?: SteerList; error?: string; loading?: boolean; cur: number; dose: Record<string, number>;
		turn: Record<string, number>; scroll: number; maxScroll: number; note?: Note; busy?: boolean;
		pending?: { args: string; name?: string }; seen: "" | "out" | "in"; del?: string; deleting?: string; offer?: LinkOffer | null; linking?: boolean;
		refused?: "reload" | "shown" } =
		{ cur: 0, dose: {}, turn: {}, scroll: 0, maxScroll: 0, seen: "" };
	private steerSeq = 0;   // the newest concept request: an older answer is dropped
	private chats = new Map<string, Examples | { error: string } | "loading">();   // by served model and name
	private mdCache = new Map<string, string[]>();
	/** Another served model: everything the page held for the last one goes (its concepts, cursor, doses, examples). */
	private resetSteer(model: string) {
		const st = this.steer;
		Object.assign(st, { model, list: undefined, error: undefined, cur: 0, dose: {}, turn: {}, scroll: 0, maxScroll: 0, del: undefined, offer: undefined, refused: undefined });
		if (!st.busy) st.note = undefined;
		this.chats.clear();
	}
	/** Ask for the served model's concepts (another model's state cleared first), and its link offer (loadOffer); an
	 * answer overtaken by a newer request, or for a model no longer served, is dropped. The cursor stays on its concept
	 * when it is still there. */
	loadSteer() {
		const h = this.host, st = this.steer, model = this.dock.served();
		if (!h.steering) return;
		if (model !== st.model) this.resetSteer(model);
		this.loadOffer(model);
		st.loading = true;
		const seq = ++this.steerSeq;
		const done = (res: SteerList | { error: string }) => {
			if (seq !== this.steerSeq || st.model !== model) return;
			st.loading = false;
			if ("error" in res) {
				st.error = res.error;
				st.list = undefined;
			} else {
				const at = h.steerAt ?? st.list?.concepts[st.cur]?.name;
				st.list = res;
				st.error = undefined;
				st.cur = Math.max(0, res.concepts.findIndex((c) => c.name === at));
			}
			this.tui.requestRender();
		};
		try {
			h.steering().then(done, (e: any) => done({ error: String(e?.message ?? e) }));
		} catch (e: any) {
			done({ error: String(e?.message ?? e) });
		}
	}
	/** The served model's link offer (host.linkOffer: the host keeps a lasting answer for the session), for the model
	 * still served; a failure or an answer it cannot read: none (the offer is optional). After a refused link, its
	 * reload: found linked, said so; the next answer clears the refusal. */
	private loadOffer(model: string) {
		const st = this.steer, took = (json: any) => {
			if (st.model !== model || this.dock.disposed) return;
			st.offer = parseOffer(json, model);
			const to = st.refused === "reload" ? linkedTo(json, model) : "";
			if (to) st.note = { text: alreadyLinked(to), level: "info" };
			else if (st.refused === "shown") st.note = undefined;
			st.refused = st.refused === "reload" && !to ? "shown" : undefined;
			this.tui.requestRender();
		};
		try {
			this.host.linkOffer?.().then(took, () => took(null));
		} catch {
			took(null);
		}
	}
	/** The link offer where the Steer page shows it: the served model's list in and empty (another model's offer goes
	 * with its list: resetSteer). */
	offerShown(): LinkOffer | undefined {
		const st = this.steer, o = st.offer;
		return o && st.list && !st.error && !st.list.concepts.length ? o : undefined;
	}
	/** A refused link's note dropped (the next l, leaving the page). */
	private dropRefusal() {
		const st = this.steer;
		if (st.refused) st.note = st.refused = undefined;
	}
	/** l on the Steer page, with the offer shown: Pi's pick list (host.link). Linked: the model's concepts, the offer and
	 * the lens steps (LENS) asked for again, the EXPERTS grids and tallies dropped (they move to the model linked to).
	 * Refused, or no answer: the page asked for again all the same (the link may have gone through; loadOffer). What
	 * came of it is the page's note; nothing once the model changed or the dock closed. One at a time. */
	private async linkKey(o: LinkOffer) {
		const st = this.steer, model = st.model;
		if (!this.host.link || st.linking) return;
		st.linking = true;
		this.dropRefusal();
		try {
			const said = await this.host.link(o);
			if (this.dock.disposed || !said || st.model !== model) return;
			const err = said.find((x) => x.level === "error" || x.level === "warning");
			st.note = err ? { text: err.text, level: "error" } : { text: said.map((x) => x.text).join("\n"), level: "info" };
			if (err) st.refused = "reload";
			else {
				st.list = st.offer = undefined;               // "asking for its concepts…" until they come
				this.dock.grids.clear();
				this.dock.xpData.clear();
				this.dock.xpLast.clear();
				this.dock.askDepth();
			}
			this.loadSteer();
		} catch (e: any) {
			if (!this.dock.disposed) st.note = { text: String(e?.message ?? e), level: "error" };
		} finally {
			st.linking = false;
			if (!this.dock.disposed) this.tui.requestRender();
		}
	}
	/** The index of the dose under the cursor for `c`: the one chosen, if it is still measured, else the picked one. */
	private doseIndex(c: SteerConcept): number {
		const v = this.steer.dose[c.name], k = v === undefined ? -1 : c.doses.findIndex((d) => d.dose === v);
		return k >= 0 ? k : Math.max(0, Math.min(c.doses.length - 1, pickIndex(c)));
	}
	private chatKey = (name: string, dose?: number) => `${this.steer.model ?? ""}\u0000${name}\u0000${dose === undefined ? "" : fmt(dose)}`;
	/** The dose the Steer page shows `c`'s test chats at: the one under the cursor (no steering: the picked one's
	 * chats, their unsteered side alone). */
	private chatDose(c: SteerConcept): number {
		const d = c.doses[this.doseIndex(c)]?.dose ?? c.pick;
		return d === 0 ? c.pick : d;
	}
	/** The markdown of an answer, drawn once per width and colour. */
	private md = (text: string, width: number, colour?: string): string[] => {
		const key = `${width}\u0000${colour ?? ""}\u0000${text}`;
		let lines = this.mdCache.get(key);
		if (!lines) {
			if (this.mdCache.size > 200) this.mdCache.clear();
			this.mdCache.set(key, (lines = mdLines(this.theme, text, width, colour)));
		}
		return lines;
	};
	/** n/N on the Steer page's test chats: the next or previous turn, round. */
	private stepTurn(c: SteerConcept, dir: 1 | -1) {
		const st = this.steer, ex = this.chats.get(this.chatKey(c.name, this.chatDose(c)));
		const turns = ex && ex !== "loading" && !("error" in ex) ? ex.turns.length : 0;
		if (turns) st.turn[c.name] = ((st.turn[c.name] ?? 0) + dir + turns) % turns;
		st.scroll = 0;
		st.maxScroll = Infinity;
	}
	/** Ask for `c`'s test chats at `dose` (undefined: the picked one) unless they are here or on their way (`retry`:
	 * again after a failure). */
	private loadChats(c: SteerConcept, retry: boolean, dose?: number) {
		const key = this.chatKey(c.name, dose), have = this.chats.get(key);
		if (!c.chat || !this.host.steerChats || !(!have || (retry && have !== "loading" && "error" in have))) return;
		this.chats.set(key, "loading");
		const settle = (res: Examples | { error: string }) => {
			if (this.chats.get(key) === "loading") this.chats.set(key, res);   // not after another model cleared it
			this.tui.requestRender();
		};
		this.host.steerChats(c.name, dose).then(settle, (e: any) => settle({ error: String(e?.message ?? e) }));
	}
	/** The delete asked for (d) is called off: any key but y does it (handleInput, before the key does its own). */
	disarm() {
		const st = this.steer;
		st.del = undefined;
		if (st.note?.level === "busy" && !st.busy) st.note = undefined;
	}
	/** d on the Steer page asks; y then deletes the concept on the checkpoint
	 * (host.deleteConcept): it leaves the list at once, the cursor on a neighbour, and the list is asked for again.
	 * A second d only asks again (key repeat never deletes). Other keys: false. */
	private deleteKey(data: string): boolean {
		const st = this.steer;
		const cs = st.list?.concepts ?? [], i = Math.max(0, Math.min(cs.length - 1, st.cur)), c = cs[i];
		if ((data !== "d" && data !== "y") || !c || !this.host.deleteConcept || this.dock.served() !== st.model) return false;
		if (data === "y") {
			if (st.del !== c.name) return false;
			st.del = undefined;
			void this.remove(c.name, cs[i + 1]?.name ?? cs[i - 1]?.name);
			return true;
		}
		if (st.busy) {
			st.note = { text: "steering is changing: d after it", level: "error" };
			return true;
		}
		st.del = c.name;
		st.note = { text: `y deletes ${c.name} on ${st.list?.checkpoint || st.model}`, level: "busy" };
		return true;
	}
	private async remove(name: string, next?: string) {
		const st = this.steer;
		st.busy = true;
		st.deleting = name;              // enter and x wait for it (apply): nothing runs against the list it changes
		st.note = { text: `deleting ${name}…`, level: "busy" };
		this.tui.requestRender();
		try {
			st.note = noteOf(await this.host.deleteConcept!(name));
		} catch (e: any) {
			st.note = { text: String(e?.message ?? e), level: "error" };
		}
		st.busy = false;
		st.deleting = undefined;
		if (st.note?.level !== "error") {
			if (this.host.steerAt === name || this.host.steerAt === undefined) this.host.steerAt = next;
			if (st.list) {               // gone at once; the server's list follows (loadSteer)
				const left = st.list.concepts.filter((c) => c.name !== name);
				st.list = { ...st.list, concepts: left };
				st.cur = Math.max(0, left.findIndex((c) => c.name === this.host.steerAt));
			}
			for (const k of [...this.chats.keys()]) if (k.split("\u0000")[1] === name) this.chats.delete(k);
			delete st.dose[name];
			this.loadSteer();
		}
		this.tui.requestRender();
	}
	/** Whether `name` is a concept of the list shown, fetched for the model served now. */
	private sendable(name: string): boolean {
		const st = this.steer, model = this.dock.served();
		return !!st.list && st.model === model && st.list.model === model && st.list.concepts.some((c) => c.name === name);
	}
	/** /steer ARGS through the host (/steer's own handler): a steer set, what is on (steeringText) is the page's note; else
	 * its last error, or the first line of what it said. One at a time: one asked for meanwhile runs after it (the last asked for wins). `name`: the concept
	 * it steers with, sent only while it is in the list shown for the model served then. */
	apply(args: string, name?: string) {
		const st = this.steer;
		if (!this.host.steer) return;
		if (st.deleting) return void (st.note = { text: `deleting ${st.deleting}: again after it`, level: "error" });
		if (st.busy) {
			st.pending = { args, name };
			st.note = { text: `then ${args === "off" ? "steering off" : args}…`, level: "busy" };
			return;
		}
		void this.run({ args, name });
	}
	private async run(first: { args: string; name?: string }) {
		const st = this.steer;
		st.busy = true;
		let next: typeof first | undefined = first;
		while (next) {
			const { args, name } = next;
			st.pending = undefined;
			if (name && !this.sendable(name)) st.note = { text: `${name} is not among ${this.dock.served() || "this model"}'s concepts shown: nothing sent`, level: "error" };
			else {
				st.note = { text: args === "off" ? "turning steering off…" : `steering with ${args}…`, level: "busy" };
				this.tui.requestRender();
				try {
					st.note = noteOf(await this.host.steer!(args));
				} catch (e: any) {
					st.note = { text: String(e?.message ?? e), level: "error" };
				}
			}
			next = st.pending;
		}
		st.busy = false;
		this.tui.requestRender();
	}
	/** The Steer page (6 on the board), under the header and its rule (`out`): the concepts by the chat test in two
	 * columns (as many rows as the concepts need, 8 to 16, while the test chats keep their room), the cursor's concept
	 * with its dose curve, then its test chats (the prompt, no steering beside steered); the key line,
	 * or (?) what it shows. No reply yet (`r`): the same, the layer count unknown. */
	renderSteerPage(out: string[], width: number, w: number, height: number, r: Reply | undefined, inspecting: boolean): string[] {
		const th = this.theme, h = this.host, st = this.steer, offer = !!this.offerShown();
		const footer = h.captions && inspecting ? [...wrapTextWithAnsi(th.fg("dim", offer ? EXPLAIN.STEER_LINK : `${EXPLAIN.STEER} ${EXPLAIN.STEER_EXAMPLES}`), w).slice(0, 4),
			th.fg("faint", fit("? the keys again", w))]
			: padAbove(this.dock.one ? 2 : 1, this.dock.boardNote ? noteLines(th, this.dock.boardNote, w, this.dock.one ? 2 : 1)
				: wrapKeys(steerKeys(w, this.dock.one ? 2 : 1, st.list?.concepts[st.cur]?.chat ?? true, !!h.live.steer?.parts.length, inspecting, offer), w).map((l) => th.fg("dim", fit(l, w))));
		const rows = height - out.length - 1 - footer.length;
		const view = this.steerView(r, inspecting, true);
		const n = view.list?.concepts.length ?? 0, { stacked, across } = steerColumns(w);
		// Stacked (narrow): the list over the detail (about 12 lines), its rows leaving the test chats some.
		const listRows = stacked ? Math.max(3, Math.min(STEER_ROWS, Math.ceil(n / across), rows - 20)) : Math.max(STEER_ROWS, Math.min(16, Math.ceil(n / 2), rows - 30));
		const body = steerPanel(th, view, w, true, listRows).map((l) => fit(l, w));
		const c = st.list?.concepts[st.cur];
		if (c) {
			body.push(ruleLine(th, w));
			if (!c.chat) body.push(th.fg("dim", "no test chats"));
			else {
				// The chats at the dose under the cursor (no steering: the unsteered replies alone).
				const at = this.chatDose(c), plainOnly = c.doses[this.doseIndex(c)]?.dose === 0;
				this.loadChats(c, false, at);
				const ex = this.chats.get(this.chatKey(c.name, at));
				if (!ex || ex === "loading") body.push(th.fg("dim", `asking for the test chats at ${fmt(at)}…`));
				else if ("error" in ex) body.push(th.fg("dim", cut(`no saved chat at ${fmt(at)}`, w)));
				else {
					const turn = Math.max(0, Math.min(ex.turns.length - 1, st.turn[c.name] ?? 0));
					const e = examplesPanel(th, ex, turn, st.scroll, this.md, w, rows - body.length, plainOnly);
					st.maxScroll = e.maxScroll;
					st.scroll = Math.min(st.scroll, e.maxScroll);
					body.push(...e.lines);
				}
			}
		}
		const lines = body.slice(0, rows).map((l) => fit(l, w));
		while (lines.length < rows) lines.push(fit("", w));
		return this.dock.frame([...out, ...lines, ruleLine(th, w), ...footer], width, height, []);
	}
	/** The Steer page's data: the served model's concepts (asked for again on stepping in, or for another model), the cursor's
	 * concept and dose, the steer on. */
	private steerView(r: Reply | undefined, inspecting: boolean, focused: boolean) {
		const h = this.host, st = this.steer, live = h.live, seen = inspecting ? "in" : "out";
		if (h.steering && this.dock.served() !== st.model) this.loadSteer();
		else if (h.steering && !st.loading && seen === "in" && st.seen !== "in") this.loadSteer();
		st.seen = seen;
		const cs = st.list?.concepts;
		if (cs?.length) st.cur = Math.max(0, Math.min(cs.length - 1, st.cur));
		let nLayers: number | undefined;
		if (r && (!r.model || !st.model || r.model.split("/").pop() === st.model))
			for (let i = r.tokens.length - 1; i >= 0 && !nLayers; i--) nLayers = r.tokens[i].n_layers ?? undefined;
		return { list: st.list, error: st.error ?? (h.steering ? undefined : "steering is not ready: use /steer"), model: st.model, cur: st.cur,
			dose: cs?.length ? this.doseIndex(cs[st.cur]) : 0, on: new Map((live.steer?.parts ?? []).map((p) => [p.name, p.strength])), nLayers,
			note: st.note, focused, offer: this.offerShown() };
	}
	/** The Steer page's keys: 6 the board, esc the keyboard back (the page kept), q leaves at once, ↑/↓ a concept, ←/→ a
	 * dose (the test chats at it), enter steers with it, x off, n/N a turn of the test chats, the page keys scroll them, ?
	 * what the page shows; l links the model where the link is offered (offerShown). */
	steerPageKey(data: string) {
		const st = this.steer;
		if (this.deleteKey(data)) return;
		if (matchesKey(data, "escape")) return void this.dock.leave();
		if (data === STEER_KEY || data === "q") {
			this.dropRefusal();
			this.dock.steerPage = false;
			st.seen = "";
			if (data === "q") this.dock.boardKey(data);
			return;
		}
		if (data === "?") return void (this.host.captions = !this.host.captions);
		if (data === "m") {
			this.dropRefusal();
			this.dock.steerPage = false;
			return this.dock.setMini(true);
		}
		if (data === "x") return this.apply("off");
		if (this.host.steering && this.dock.served() !== st.model) return this.loadSteer();
		const offer = this.offerShown();
		if (data === "l" && offer) return void this.linkKey(offer);
		const list = st.list?.concepts;
		if (!list?.length) return;
		st.cur = Math.max(0, Math.min(list.length - 1, st.cur));
		const c = list[st.cur];
		if (data === "n" || data === "N") return this.stepTurn(c, data === "n" ? 1 : -1);
		if (matchesKey(data, "pageUp") || matchesKey(data, "pageDown")) {
			st.scroll = Math.max(0, Math.min(st.maxScroll, st.scroll + (matchesKey(data, "pageUp") ? -10 : 10)));
			return;
		}
		const up = matchesKey(data, "up") || data === "k", down = matchesKey(data, "down") || data === "j";
		if (up || down) {
			st.cur = Math.max(0, Math.min(list.length - 1, st.cur + (down ? 1 : -1)));
			this.host.steerAt = list[st.cur].name;
			st.scroll = 0;
			if (!st.busy) st.note = undefined;
		} else if (matchesKey(data, "left") || matchesKey(data, "right")) {
			const k = Math.max(0, Math.min(c.doses.length - 1, this.doseIndex(c) + (matchesKey(data, "right") ? 1 : -1)));
			if (c.doses[k]) st.dose[c.name] = c.doses[k].dose;
			if (!st.busy) st.note = undefined;
		} else if (matchesKey(data, "enter")) {
			const d = c.doses[this.doseIndex(c)];
			// Never chat-tested and nothing on: enter has nothing to do but say how to use it.
			if (d && !c.chat && !this.host.live.steer?.parts.length) st.note = { text: reading(c, d)[1], level: "info" };   // drawn as the reading
			else if (d) this.apply(d.dose === 0 ? "off" : `${c.name} ${fmt(d.dose)}`, d.dose === 0 ? undefined : c.name);
		}
	}
}

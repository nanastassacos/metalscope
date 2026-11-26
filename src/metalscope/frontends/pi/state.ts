/**
 * State: the live token stream from the metalscope server and the reply being followed.
 * `Live` is the small context shared by the stream, the widget, the dock
 * and the commands; everything else takes what it needs as arguments.
 */

import { clean } from "./draw.ts";
import { scopeAuth } from "./provider.ts";

/** Tracing: off or on. */
export type Level = "off" | "on";

export interface TokenEvent {
	i: number;
	id: number;
	text: string; // as drawn (draw.ts clean)
	raw?: string; // as the model wrote it, where cleaning changed it: the reply's text is rebuilt from `raw ?? text`
	p: number | null; // probability of the emitted token (null: below the recorded top-k)
	entropy: number;
	alts: [string, number][]; // runners-up, likeliest first (the output top 10 less the written token: livestream.N_ALTS)
	n_layers: number | null;
	// The lens's top-1 at every layer read, shallow to deep: [layer, token id, its text].
	// The J-lens where one is on, else the logit lens; settle is computed from the same list.
	lens?: [number, number, string][];
	lens_by?: "logit" | "jlens"; // which lens read `lens` (/jlens off turns the J-lens off, /jlens on turns it on)
	lens_logit?: [number, number, string][]; // with a J-lens: the logit lens's, read beside it (the dock's s shows it)
	// MoE routing, sent only while the dock shows experts (Live.experts): the top-k experts of the forward this
	// token was fed into, every MoE layer in order, flattened [L * k]; each one's share of its layer's mix, 0-255; k.
	// Kept as typed arrays (onEvent), and only on the newest MAX_ROUTED tokens of a reply.
	experts?: Uint16Array;
	expert_w?: Uint8Array;
	expert_k?: number;
	// Each read layer's top 3 with probabilities and the written token's probability there, sent only while the dock
	// shows LENS (Live.lens: the stream's lens10 and lens10_logit, packed by packLens): `top` the lens of
	// `lens`, `top_logit` the logit lens beside it. Kept on the newest MAX_LENS_TOP tokens of a reply.
	top?: LensTop;
	top_logit?: LensTop;
	// What each layer changed in each residual stream while this token was chosen (the stream's `writes`), and where each
	// layer read from (`reads`), sent only while the dock shows STREAM (Live.writes); decided as they came against the
	// reply's usual (packWrites) and kept in brief: `strikes` spikes (hc)'s, [layer, stream, beam length] a layer
	// struck; `wholeStrikes` spikes (layers)'s, as `strikes` from each layer's whole write (stream 0; one stream: the
	// same); `largest` largest (hc)'s, a layer its stream and brightness (packLargest); `layers` largest (layers)'s, as
	// `largest`, each layer by the writes' `shares` (shareLayers); `readStrikes` spikes (reads)'s, as `strikes` with a source for
	// the stream.
	strikes?: Uint16Array;
	wholeStrikes?: Uint16Array;
	largest?: Uint16Array;
	layers?: Uint16Array;
	readStrikes?: Uint16Array;
}

/** A token's lens read-out in brief (packLens): per read layer `l[r]`, its three likeliest tokens (`id`, `w`, `p` at
 * r * 3 + rank; id -1 none) and the written token's probability (`wp[r]`); probabilities as the stream's byte,
 * round(-ln p * 20), 255 = none (outside the layer's top 10). */
export interface LensTop {
	l: number[];
	id: Int32Array;
	w: string[];
	p: Uint8Array;
	wp: Uint8Array;
}

/** A LensTop probability byte as a probability, null for none. */
export const lensP = (b: number) => (b >= 255 ? null : Math.exp(-b / 20));

// What a kept token costs on the heap (measured 2026-11-19, two lenses of 24 layers): about 4.8 KB with no cap, 3.8 KB
// of it the lens top 1s (`lens`, `lens_logit`); up to three replies are held (Live.reply, Live.prev, an edit's base).
// The lens top 3s add about 6 KB a token, kept on this many of a reply's newest tokens (12 MB).
export const MAX_LENS_TOP = 2048;

// Routing is kept on this many of a reply's newest tokens (about 1.5 KB each on a 48-layer, top-10 model: 6 MB).
export const MAX_ROUTED = 4096;

export interface Reply {
	rid: string;
	tokens: TokenEvent[];
	done: boolean;
	end?: number; // once done: past its last token's i (its end's count; addToken drops a token from there)
	traceId?: string;
	capped?: number; // its saved trace stops before this token (the server's trace cap, event "capped")
	prefill: number;
	steer?: string; // the steer it ran under, as steerLabel shows it ("" = none); undefined when unknown (adopted mid-stream)
	model?: string; // the served model's name, from "start" (undefined when adopted mid-stream)
	routed?: boolean; // some token carried expert routing
	maxExpert?: number; // the highest expert id seen (the dock's grid size before the server says it)
	askedExperts?: boolean; // its start and every token came through a stream that asked for routing (onEvent `from`)
	askedWrites?: boolean;  // the same for the writes (Live.writes)
	usual?: Usual;          // its writes so far, the usual each token's are weighed against (packWrites)
	wholeUsual?: Usual;     // the same of each layer's whole write, on a model writing several streams
	readUsual?: Usual;      // and its reads
	// Its routing from its saved trace (the dock's loadRouting, for a reply traced before the stream was asked for
	// routing): "loading", "loaded", or why there is none.
	routingFrom?: string;   // also "again": not saved yet when last asked, to be asked again
	routingTries?: number;  // how many times its routing was not found (the dock gives up after ROUTING_RETRY_MS)
	// An edit's reply (the dock's word edit, "start"'s `edit`): it continues reply `of`, whose tokens before `at` it
	// holds; its own tokens come `at` on (`from`: added to each one's i). `parts`: the replies its earlier tokens
	// came from, oldest first.
	of?: string;
	from?: number;
	parts?: Part[];
	edit?: Edited;
}

/** Where a joined reply's tokens from `from` on came from: reply `rid`. */
export interface Part {
	from: number;
	rid: string;
}

/** A word edit in the dock: the reply `rid`, its token `at` (TokenEvent.i), the edit's steer header, for the served
 * model `model`; `word` the lens word removed, `want` a swap's new word. It goes with /steer again's answer only
 * (provider.ts): `prompt` the message sent again, `running` once that run started, `over` once it settled, `sent`
 * once a request carried it, `answered` once a reply took it (onEvent). The edited reply's copy has `was`: the
 * token the edited reply's model liked best there (its top 1, the id when it was the token written). */
export interface Edited {
	rid: string;
	at: number;
	steer: string;
	model: string;
	word: string;
	want?: string;
	prompt?: string;
	running?: boolean;
	over?: boolean;
	sent?: boolean;
	answered?: boolean;
	base?: Reply; // the reply edited, once a reply took the edit: a retry's reply continues it again (onEvent)
	was?: { id?: number; text: string };
	told?: boolean; // the dock said what came of it (draw.ts LABELS.edited)
	toldEnd?: string; // the reply whose end it looked at for a tool call (an edit's two replies share this Edited)
}

// Tool call openers (edit.py TOOL_OPENERS): ds4-server's, then Inkling 2's; an edit's reply stops at one.
export const TOOL_OPENERS = ["<tool_call>", "<tool_calls>", "<｜DSML｜ calls>", "<｜DSML｜tool_calls>", "<DSML｜tool_calls>", "<|content_invoke_tool_json|>"];

/** What the stream, widget, dock and commands share. */
export interface Live {
	base: string; // the metalscope server's URL (provider.ts picks it at startup)
	clientId: string; // tags this Pi's requests and its live-stream subscription
	level: Level;
	reply?: Reply; // the reply being followed (the last one, once done)
	prev?: Reply; // the last reply with tokens before it: shown while the next reads its prompt (shownReply)
	connected: boolean;
	requestRender?: (progress?: boolean) => void; // progress: only a token or the prompt's count is new (the dock may wait: Dock.render)
	onStart?: (model?: string) => void; // a new reply started (index.ts: its model's flags, provider.ts refreshFlags)
	steer?: import("./steer.ts").SteerState; // what /steer adds, while on
	sentSteer?: string; // the steer the latest traced request went out with ("" = none): the next reply's Reply.steer
	requestSteer?: string; // the steer the latest request went out with, traced or not ("" = none; unset: none sent yet)
	experts?: boolean; // ask the stream for each token's expert routing (while the dock shows experts: widget.ts askLive)
	lens?: boolean; // ask the stream for each read layer's top 10 (while the dock shows LENS)
	writes?: boolean; // ask the stream for each token's writes and reads (while the dock shows STREAM)
	// The last word on a model's writes and reads: its layers (`L`) and how many streams it writes into (`k`) and sources
	// it reads from (`rk`; 0: none sent), and whether the writes came with their `shares`, from a token with either, L 0
	// once a token of a reply asked for them came without and none of that reply's had any (a model that records none:
	// STREAM not drawn). Kept until the next.
	streams?: { model: string; L: number; k: number; rk: number; shares: boolean };
	dense?: string; // a model whose reply (2 own tokens or more) streamed with routing asked for and none came: EXPERTS not drawn, routing not asked of its traces
	ended?: string[]; // the replies whose end arrived, newest last (ENDED_KEPT): a second stream's copies of their events are dropped
	edit?: Edited; // the dock's word edit, until the next run after its own starts (provider.ts editStarts)
	map?: MapReply[]; // this session's ended replies, oldest first, as the full-size MAP draws them (keepMap)
	// g's readings: the newest progress the stream sent, when it came (performance.now) and the most of both readings
	// the bar showed for its ask (the dock's bar: never back, in any dock)
	lookup?: import("./lookup.ts").LookupProgress & { at: number; most: number };
}

/** The index of the token that ends the thinking (its text holds "</think>"), -1 without one. */
export function thinkEnd(toks: { text: string }[]): number {
	return toks.findIndex((t) => t.text.includes("</think>"));
}

/** A reply as the full-size MAP keeps it (Live.map): each token's probability in 200ths rounded down, a byte (null 0;
 * 50% and 90% stay on their side of the colours' edges), where its thinking ends (thinkEnd), its model; `replaced`
 * the replies whose place it took (edited ones, an edit's earlier tries), oldest first. */
export interface MapReply {
	rid: string;
	replaced?: string[];
	model?: string;
	p: Uint8Array;
	think: number;
}
/** The tokens Live.map keeps (a byte each, 1 MB): past it the oldest replies go, from the picture and its counts. */
export const MAP_KEPT = 1 << 20;

/** Reply `r` as Live.map keeps it. */
export const mapReply = (r: { rid: string; model?: string; tokens: { text: string; p: number | null }[] }): MapReply =>
	({ rid: r.rid, model: r.model, p: Uint8Array.from(r.tokens, (t) => Math.floor((t.p ?? 0) * 200)), think: thinkEnd(r.tokens) });

/** The replies an edit's reply joins (the one it edits, those its earlier tokens came from). */
const joins = (r: Reply) => [...(r.of === undefined ? [] : [r.of]), ...(r.parts ?? []).map((p) => p.rid)];
/** Whether reply `r` is reply `rid` or continues it (an edit's reply: joins). */
export const joined = (r: Reply, rid: string) => r.rid === rid || joins(r).includes(rid);

/** Whether reply `r` has tokens of its own (an edit's reply: past the ones it continues). */
const ownTokens = (r?: Reply) => !!r?.tokens.length && (r.from === undefined || r.tokens[r.tokens.length - 1].i >= r.from);

/** Where reply `r` (the one shown, ended or not) goes among the session's replies: over its own copy, or one it joins
 * or one that took that one's place (an edit retried; `replace`), else last. Nowhere (-1) without tokens of its own (an
 * edit's reply before its first token, or ended without one) or once a later reply took its place. */
export function mapPlace(map: MapReply[], r?: Reply): { at: number; replace: boolean } {
	if (!r || !ownTokens(r) || map.some((x) => x.replaced?.includes(r.rid))) return { at: -1, replace: false };
	const ids = joins(r), at = map.findIndex((x) => x.rid === r.rid || ids.some((id) => x.rid === id || !!x.replaced?.includes(id)));
	return at < 0 ? { at: map.length, replace: false } : { at, replace: true };
}

/** Reply `r` ended: into Live.map where mapPlace puts it (taking over what the one it replaces had replaced), the
 * oldest dropped while it holds more than MAP_KEPT tokens. */
function keepMap(live: Live, r: Reply) {
	const map = [...(live.map ?? [])], { at, replace } = mapPlace(map, r);
	if (at < 0) return;
	const m = mapReply(r);
	if (replace) {
		const x = map[at], replaced = [...new Set([...(x.replaced ?? []), ...(x.rid === r.rid ? [] : [x.rid]), ...joins(r)])];
		map[at] = replaced.length ? { ...m, replaced } : m;
	} else map.push(m);
	let n = map.reduce((s, x) => s + x.p.length, 0);
	while (map.length > 1 && n > MAP_KEPT) n -= map.shift()!.p.length;
	live.map = map;
}

// The ended replies remembered (Live.ended): more than a stream switch's overlap can hold.
const ENDED_KEPT = 16;

/** The live stream's URL for `live`: this Pi's replies, with expert routing, the lens top 10s and the writes while
 * each is asked for. */
export function liveUrl(live: Pick<Live, "base" | "clientId" | "experts" | "lens" | "writes">): string {
	return `${live.base}/scope/api/live?client=${live.clientId}${live.experts ? "&experts=true" : ""}${live.lens ? "&lens=true" : ""}${live.writes ? "&writes=true" : ""}`;
}

/** A reply's writes (or reads) so far (Reply.usual, readUsual), `n` tokens of `L` layers into `k` streams (from `k`
 * sources): each layer and stream's running mean and sum of squared deviations (Welford's). */
export interface Usual {
	n: number;
	L: number;
	k: number;
	mean: Float64Array;
	m2: Float64Array;
}
/** The spikes views (and reads): a layer is struck where it changed a stream (read a source) more than WRITE_Z
 * standard deviations over its usual in the reply before this token (writeSd), from the reply's WRITE_WARM + 1th token
 * with them on; its beam a cell longer for every WRITE_PER past that, up to WRITE_BEAM cells. */
export const WRITE_Z = 2, WRITE_WARM = 12, WRITE_PER = 0.6, WRITE_BEAM = 6;
/** A stream that took under this share of its layer's write for the token never strikes: the recording does not resolve
 * it (the server subtracts two 16-bit streams; such a cell records a third of its size or 0). It still counts in the usual. */
export const WRITE_FLOOR = 0.01;
/** The spread a new value is weighed against, from `n` earlier ones (their squared deviations summed `m2`): the sample
 * deviation widened for the mean's own error, var = m2 / (n - 1) * (1 + 1 / n), so a short usual strikes no more than a
 * long one. */
export const writeSd = (m2: number, n: number) => (n > 1 ? Math.sqrt((m2 / (n - 1)) * (1 + 1 / n)) : 0);
export const beamOf = (z: number) => 1 + Math.min(WRITE_BEAM - 1, Math.floor((z - WRITE_Z) / WRITE_PER));

/** A layer's `largest` entry (TokenEvent.largest): its largest stream and its brightness (0-1) in 32 steps. */
export const packLargest = (stream: number, v: number) => (stream << 5) | Math.round(v * 31);
export const unpackLargest = (x: number): [stream: number, v: number] => [x >> 5, (x & 31) / 31];

/** The faintest a layer with a share is drawn (largest (layers)), so it stays apart from an unlit one. */
export const SHARE_FLOOR = 0.1;
/** A token's `shares` (livestream._writes: each layer's write over the residual it read, layer 0 0) as its `layers`
 * (packLargest, stream 0): every layer as bright as its share over the token's largest (half the share, half as
 * bright), SHARE_FLOOR at least; a share of 0 dark. Undefined when malformed. */
export function shareLayers(x: unknown): Uint16Array | undefined {
	if (!Array.isArray(x) || !x.length || x.length > 2047 || !x.every((v) => typeof v === "number" && Number.isFinite(v) && v >= 0)) return undefined;
	const top = Math.max(...x);
	return Uint16Array.from(x, (v) => (v > 0 ? packLargest(0, Math.max(SHARE_FLOOR, v / top)) : 0));
}

/** One flow of token event `t` (`list` [L * k] layer-major, shallow to deep, `k` its streams or sources), dropped from
 * the event, decided against usual `u` before it and then added to it (a new usual where `u` is of another shape):
 * `strikes` (a layer whose stream z is over WRITE_Z and whose share of the layer is WRITE_FLOOR or more, the highest
 * z's stream where several are) and `largest` (each layer's largest stream, as bright as the layer's whole change
 * against its usual: 0.1 + 0.9 * total / usual / 2, at most 1; no usual yet 0.55; the writes' only, `big`), by decide.
 * Undefined when malformed. */
function decideFlow(u: Usual | undefined, t: any, list: "writes" | "reads", kOf: "write_k" | "read_k", big: boolean) {
	const x = t[list], k = t[kOf];
	delete t[list];
	delete t[kOf];
	if (!Array.isArray(x) || !Number.isInteger(k) || k < 1 || k > 2047 || !x.length || x.length % k
		|| !x.every((v: any) => typeof v === "number" && Number.isFinite(v) && v >= 0)) return undefined;
	return { ...decide(u, x, k, big), x, k };
}
function decide(u: Usual | undefined, x: number[], k: number, big: boolean) {
	const L = x.length / k;
	if (!u || u.L !== L || u.k !== k) u = { n: 0, L, k, mean: new Float64Array(L * k), m2: new Float64Array(L * k) };
	const strikes: number[] = [], largest = big ? new Uint16Array(L) : undefined;
	for (let l = 0; l < L; l++) {
		let top = 0, total = 0, usual = 0, hit = -1, z = WRITE_Z;
		for (let s = 0; s < k; s++) total += x[l * k + s];
		for (let s = 0; s < k; s++) {
			const j = l * k + s, v = x[j], sd = writeSd(u.m2[j], u.n);
			usual += u.mean[j];
			if (v > x[l * k + top]) top = s;
			if (u.n >= WRITE_WARM && sd > 0 && v >= WRITE_FLOOR * total && (v - u.mean[j]) / sd > z) [hit, z] = [s, (v - u.mean[j]) / sd];
		}
		if (hit >= 0) strikes.push(l, hit, beamOf(z));
		if (largest) largest[l] = packLargest(top, !u.n ? 0.55 : 0.1 + 0.9 * Math.min(1, usual > 0 ? total / usual / 2 : 0.5));
	}
	u.n++;
	for (let j = 0; j < x.length; j++) {
		const d = x[j] - u.mean[j];
		u.mean[j] += d / u.n;
		u.m2[j] += d * (x[j] - u.mean[j]);
	}
	return { u, strikes: Uint16Array.from(strikes), largest };
}

/** Token event `t`'s writes (`writes`, `write_k`, `shares`) and reads (`reads`, `read_k`: each layer's shares of its
 * sources) decided in place against reply `r`'s usuals (decideFlow): `strikes` and `largest` from the writes,
 * `wholeStrikes` the same way from each layer's streams as one (one stream: `strikes`), `layers`
 * from their shares (shareLayers, its layers the writes'), `readStrikes` from the reads; the raw lists dropped, each whole
 * when malformed. Whether it had either. */
export function packWrites(r: Reply, t: any): boolean {
	const lit = shareLayers(t.shares);
	delete t.shares;
	const w = decideFlow(r.usual, t, "writes", "write_k", true), rd = decideFlow(r.readUsual, t, "reads", "read_k", false);
	if (w) [r.usual, t.strikes, t.largest, t.wholeStrikes] = [w.u, w.strikes, w.largest, w.strikes];
	if (w && w.k > 1) {   // the streams' writes as one: the root of their squares' sum, a layer's whole write
		const whole = decide(r.wholeUsual, Array.from({ length: w.u.L }, (_, l) => Math.hypot(...w.x.slice(l * w.k, (l + 1) * w.k))), 1, false);
		[r.wholeUsual, t.wholeStrikes] = [whole.u, whole.strikes];
	}
	if (w && lit?.length === w.u.L) t.layers = lit;
	if (rd) [r.readUsual, t.readStrikes] = [rd.u, rd.strikes];
	return !!(w || rd);
}

/** A token event's lens10 / lens10_logit (livestream._lens10: per read layer the top k as indices `c` into distinct
 * ids `i` and words `w`, probability bytes `p`) packed in place as top / top_logit; dropped when malformed. */
export function packLens(t: any): void {
	for (const [from, to] of [["lens10", "top"], ["lens10_logit", "top_logit"]] as const) {
		const x = t[from];
		delete t[from];
		if (!x || !Array.isArray(x.l) || !Array.isArray(x.i) || !Array.isArray(x.w) || !Array.isArray(x.c) || !Array.isArray(x.p)) continue;
		const L = x.l.length, k = x.k;
		if (!Number.isInteger(k) || k < 1 || x.c.length !== L * k || x.p.length !== L * k || x.i.length !== x.w.length) continue;
		const id = new Int32Array(L * 3).fill(-1), w: string[] = new Array(L * 3).fill(""), p = new Uint8Array(L * 3).fill(255);
		const wp = new Uint8Array(L).fill(255);
		let ok = true;
		for (let r = 0; r < L && ok; r++)
			for (let j = 0; j < k; j++) {
				const c = x.c[r * k + j], b = x.p[r * k + j];
				if (!Number.isInteger(c) || c < 0 || c >= x.i.length || !Number.isInteger(b) || b < 0 || b > 255) { ok = false; break; }
				if (j < 3) {
					id[r * 3 + j] = x.i[c];
					w[r * 3 + j] = clean(String(x.w[c]));
					p[r * 3 + j] = b;
				}
				if (x.i[c] === t.id && wp[r] === 255) wp[r] = b;
			}
		if (ok) t[to] = { l: x.l.map(Number), id, w, p, wp } as LensTop;
	}
}

/** A token event's routing as typed arrays (in place); dropped when malformed. */
export function packRouting(t: any): void {
	const ids = t.experts, w = t.expert_w, k = t.expert_k;
	delete t.experts;
	delete t.expert_w;
	delete t.expert_k;
	if (!Array.isArray(ids) || !Array.isArray(w) || ids.length !== w.length || !Number.isInteger(k) || k < 1 || !ids.length || ids.length % k) return;
	// Values too: an id or weight that is not a whole number in range (a stray -1) drops the routing, not the token.
	if (!ids.every((e: any) => Number.isInteger(e) && e >= 0 && e <= 0xffff) || !w.every((x: any) => Number.isInteger(x) && x >= 0 && x <= 255)) return;
	t.experts = Uint16Array.from(ids);
	t.expert_w = Uint8Array.from(w);
	t.expert_k = k;
}

/** The reply the dock shows: the one being followed once it has a token of its own (an edit's answer: its first new
 * one, or its end), else the last one that had (between the requests of one answer, a tool call's result being read,
 * an edit being answered, the dock keeps what it showed). */
export function shownReply(live: Pick<Live, "reply" | "prev">): Reply | undefined {
	return ownTokens(live.reply) || (live.reply?.done && live.reply.tokens.length) ? live.reply : live.prev ?? live.reply;
}

/** Reply `r`'s end says it had `n` tokens of its own (an edit's reply keeps the ones it continues too): `end` past the
 * last, by index (pipelined decode can announce a token past the stop: dropped, now and when it comes). */
function upTo(r: Reply, n: number) {
	const end = (r.end = (r.from ?? 0) + n);
	r.tokens = r.tokens.filter((t) => t.i < end);
}

/** The text at `j` of each row (an alternative's [text, p], a lens read's [layer, id, text]) cleaned in place (the
 * event is freshly parsed); anything else left as it is. */
function cleanAt(rows: unknown, j: number) {
	if (Array.isArray(rows)) for (const row of rows) if (Array.isArray(row) && typeof row[j] === "string") row[j] = clean(row[j]);
}

/** Token event `ev` into reply `r` in its place by `i` (while the subscription changes, makeFollower, the old stream
 * can bring one after the new one's later ones, or after the end); the other stream's copy, or one past the end,
 * dropped. Whether it was taken. */
function addToken(live: Live, r: Reply, ev: any, from: EventSource): boolean {
	if (typeof ev.text !== "string" || typeof ev.i !== "number") return false;   // a malformed frame must not disturb the word state
	ev.i += r.from ?? 0;
	if (!from.experts) r.askedExperts = false;   // a token from a stream without routing: no verdict on the model
	if (!from.writes) r.askedWrites = false;     // nor, without writes, on its streams
	const toks = r.tokens;
	let at = toks.length;
	while (at > 0 && toks[at - 1].i > ev.i) at--;
	if ((at > 0 && toks[at - 1].i === ev.i) || (r.end !== undefined && ev.i >= r.end)) return false;
	// The model's text is drawn as it comes: cleaned once here (the lens top 3s in packLens); `raw` kept where that
	// changed it, for the reply's text as the server has it.
	const text = clean(ev.text);
	if (text !== ev.text) ev.raw = ev.text;
	ev.text = text;
	cleanAt(ev.alts, 0);
	cleanAt(ev.lens, 2);
	cleanAt(ev.lens_logit, 2);
	packRouting(ev);
	packLens(ev);
	// The writes: a token with them says the model has streams; one asked for that came without, before any of its
	// reply's had them, says it has none (one missing later is a stray, not a verdict).
	const had = !!(r.usual || r.readUsual);
	if (packWrites(r, ev)) live.streams = { model: r.model ?? "", L: (r.usual ?? r.readUsual)!.L, k: r.usual?.k ?? 0, rk: r.readUsual?.k ?? 0, shares: !!ev.layers };
	else if (r.askedWrites && !had) live.streams = { model: r.model ?? "", L: 0, k: 0, rk: 0, shares: false };
	// Kept without what every token repeats (a fresh object: V8 keeps one with deleted fields as a slower, larger dictionary).
	const { type: _type, rid: _rid, client: _client, ...t } = ev;
	toks.splice(at, 0, t as TokenEvent);
	if (r.done) keepMap(live, r);
	if (t.experts) {
		r.routed = true;
		let top = r.maxExpert ?? 0;
		for (const e of t.experts) if (e > top) top = e;
		r.maxExpert = top;
	}
	const old = toks[toks.length - 1 - MAX_ROUTED];
	if (old?.experts) {
		delete old.experts;
		delete old.expert_w;
	}
	const older = toks[toks.length - 1 - MAX_LENS_TOP];
	if (older?.top || older?.top_logit) {
		delete older.top;
		delete older.top_logit;
	}
	return true;
}

/** A token's top 1: the token written, unless a runner-up was likelier (sampled, not greedy). */
const topOf = (t: TokenEvent) => (t.alts?.[0] && (t.p === null || t.alts[0][1] > t.p) ? { text: t.alts[0][0] } : { id: t.id, text: t.text });

/** Reply `r` continues `base` from its token `at` (an edit's reply, "start"'s `edit`): base's tokens before it, its
 * own from there. The dock's word edit (live.edit), once sent and answered by this reply, becomes its Edited with
 * the top 1 there before (`was`); the reply after the edited token hands that on. */
function continues(live: Live, r: Reply, base: Reply, at: unknown) {
	if (typeof at !== "number") return;
	r.of = base.rid;
	r.from = at;
	r.tokens = base.tokens.filter((t) => t.i < at);
	// The usuals its writes and reads are weighed against: the base's, copies (the raw lists are not kept, so not as they
	// stood at `at`), so the tokens after the edit strike and light as they would have, no warm-up again.
	const copy = (u?: Usual) => u && { ...u, mean: u.mean.slice(), m2: u.m2.slice() };
	r.usual = copy(base.usual);
	r.wholeUsual = copy(base.wholeUsual);
	r.readUsual = copy(base.readUsual);
	const own = base.from ?? 0;
	r.parts = [...(base.parts ?? []).filter((p) => p.from < at), ...(own < at ? [{ from: own, rid: base.rid }] : [])];
	const mine = live.edit;
	if (mine?.sent && mine.rid === base.rid && mine.at === at) {
		const was = base.tokens.find((t) => t.i === at);
		mine.answered = true;
		mine.base = base;
		const { base: _, ...told } = mine;
		r.edit = { ...told, ...(was ? { was: topOf(was) } : {}) };
	} else if (base.edit) r.edit = base.edit;
}

/** Whether edit `e` changed its token: `now`, the token written there, is not the top 1 there before (`was`). The key
 * line's outcome and the reply strip's mark (editMark) both ask this. */
export const editChanged = (e: Edited, now?: TokenEvent) => !!now && (e.was?.id !== undefined ? now.id !== e.was.id : now.text !== e.was?.text);

/** The reply strip's mark for reply `r`'s word edit (Reply.edit, the newest): its token's index in `r.tokens` less
 * `from` (where the strip's tokens start), and whether the edit changed it; none before `from` or before the token came. */
export function editMark(r: Reply, from = 0): { k: number; changed: boolean } | undefined {
	const e = r.edit, k = e ? r.tokens.findIndex((t) => t.i === e.at) : -1;
	return e && k >= from ? { k: k - from, changed: editChanged(e, r.tokens[k]) } : undefined;
}

/** Where an event came from: whether its stream asked for routing (follow passes it; tests and callers without it:
 * whatever Live.experts says now). */
export interface EventSource {
	experts?: boolean;
	writes?: boolean;
}

/** One event from the live stream; g's readings' progress (type "lookup") is kept as it comes, with when. While the
 * subscription changes (makeFollower.resubscribe) two streams can deliver the same events: a reply's tokens are taken
 * once each, in their place (by `i`), a start for the reply being followed is not a new reply, prefill counts are cumulative (`read`), and
 * events of a reply that already ended or was already left behind (Live.ended, live.prev) are not adopted again. */
export function onEvent(live: Live, ev: any, from: EventSource = { experts: live.experts, writes: live.writes }) {
	if (ev.type === "lookup") {
		live.lookup = { ...ev, at: performance.now(), most: live.lookup && live.lookup.ask === ev.ask ? live.lookup.most : 0 };
		live.requestRender?.();
		return;
	}
	// A saved trace that stopped at the server's cap (sent after its end): `tokens` of its own tokens kept.
	if (ev.type === "capped") {
		const r = [live.reply, live.prev].find((x) => x?.rid === ev.rid);
		if (r && typeof ev.tokens === "number") {
			r.capped = (r.from ?? 0) + ev.tokens;
			live.requestRender?.();
		}
		return;
	}
	// A token of the last reply after the next one started (the old stream's, while the subscription changes): its own.
	if (ev.type === "token" && live.prev && ev.rid === live.prev.rid && ev.rid !== live.reply?.rid) {
		if (addToken(live, live.prev, ev, from)) live.requestRender?.();
		return;
	}
	if (ev.type === "end" && typeof ev.rid === "string" && !live.ended?.includes(ev.rid))
		live.ended = [...(live.ended ?? []), ev.rid].slice(-ENDED_KEPT);
	else if (ev.type !== "end" && ev.rid !== live.reply?.rid && (live.ended?.includes(ev.rid) || ev.rid === live.prev?.rid)) return;
	// The last reply's end, arriving after the next one started (it is still the one shown): it is done, with its trace.
	if (ev.type === "end" && live.prev && ev.rid === live.prev.rid && ev.rid !== live.reply?.rid) {
		upTo(live.prev, ev.n_tokens);
		live.prev.done = true;
		live.prev.traceId = ev.trace_id;
		keepMap(live, live.prev);
		live.requestRender?.();
		return;
	}
	// Adopt a reply on its first event, so a late (re)connect still follows it.
	let adopted = false;   // a new reply: drawn at once, not as the stream's progress
	if (ev.type === "start" || ((!live.reply || ev.rid !== live.reply.rid) && ev.type !== "end")) {
		// The steer goes with "start" only: a reply adopted mid-stream (a reconnect) has no known steer.
		// The same start twice (two streams overlap for a moment while the subscription changes, makeFollower): kept.
		if (!(ev.type === "start" && live.reply?.rid === ev.rid)) {
			const base = ev.type === "start" && ev.edit ? [live.reply, live.prev, live.edit?.base].find((r) => r?.rid === ev.edit.of) : undefined;
			// The last reply, if it has tokens or its end says it had some (all still on the old stream: they go there).
			const last = live.reply;
			if (last?.tokens.length || (last?.end ?? 0) > (last?.from ?? 0)) live.prev = last;
			adopted = true;
			live.reply = { rid: ev.rid, tokens: [], done: false, prefill: 0, ...(ev.type === "start" && from.experts ? { askedExperts: true } : {}),
				...(ev.type === "start" && from.writes ? { askedWrites: true } : {}),
				...(ev.type === "start" && live.sentSteer !== undefined ? { steer: live.sentSteer } : {}),
				...(ev.type === "start" && typeof ev.model === "string" ? { model: ev.model } : {}) };
			if (base) continues(live, live.reply, base, ev.edit.at);
			if (ev.type === "start") live.onStart?.(live.reply.model);
		}
	}
	const reply = live.reply;
	if (!reply || ev.rid !== reply.rid || ev.type === "start") {
		live.requestRender?.();
		return;
	} else if (ev.type === "prefill") {
		// `read`: the prompt tokens read so far (a copy from the other stream changes nothing).
		if (typeof ev.read === "number") reply.prefill = Math.max(reply.prefill, ev.read);
	} else if (ev.type === "token") {
		if (!addToken(live, reply, ev, from)) return;
	} else if (ev.type === "end") {
		// A reply that streamed whole with routing asked for and none came, of 2 tokens of its own or more (its last is
		// never fed back, so has none; an edit's copied tokens are not its own): a model without experts.
		const own = reply.tokens.filter((t) => t.i >= (reply.from ?? 0)).length;
		if (reply.askedExperts && !reply.routed && own > 1) live.dense = reply.model ?? "";
		else if (reply.routed && live.dense === (reply.model ?? "")) live.dense = undefined;
		upTo(reply, ev.n_tokens);
		reply.done = true;
		reply.traceId = ev.trace_id;
		keepMap(live, reply);
	}
	live.requestRender?.(!adopted && (ev.type === "token" || ev.type === "prefill"));
}

/** How makeFollower runs a stream: `onOpen` is called once the first connection answers; while `pending()` says so
 * the stream is not yet the one in use, so its failures do not mark the server down. */
export interface FollowOpts {
	onOpen?: () => void;
	pending?: () => boolean;
	seen?: (ev: any) => void;   // each event as it comes, before onEvent (makeFollower: when the old stream may close)
}

/** A live stream silent this long is taken for dead (a half-open socket after sleep or a network change) and opened
 * again: the server sends a ping every 15 s. */
export const STALL_MS = 45_000;

/** Follow the server's live stream until aborted, reconnecting with backoff (to live.base as it is then, liveUrl). */
export async function follow(live: Live, session: AbortSignal, opts: FollowOpts = {}) {
	let backoff = 1000;
	let onOpen = opts.onOpen;
	while (!session.aborted) {
		const stalled = new AbortController(), signal = AbortSignal.any([session, stalled.signal]);
		let quiet: ReturnType<typeof setTimeout> | undefined;
		const heard = () => {
			clearTimeout(quiet);
			quiet = setTimeout(() => stalled.abort(), STALL_MS);
		};
		try {
			heard();
			const from: EventSource = { experts: !!live.experts, writes: !!live.writes };   // what this connection asked for
			const res = await fetch(liveUrl(live), { headers: scopeAuth(live.base), signal });
			if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
			if (session.aborted) break;
			onOpen?.();
			onOpen = undefined;
			live.connected = true;
			backoff = 1000;
			live.requestRender?.();
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buf = "";
			for (;;) {
				const { value, done } = await reader.read();
				if (done) break;
				heard();
				buf += decoder.decode(value, { stream: true });
				let cut;
				while ((cut = buf.indexOf("\n\n")) >= 0) {
					const frame = buf.slice(0, cut);
					buf = buf.slice(cut + 2);
					for (const line of frame.split("\n")) {
						if (!line.startsWith("data: ")) continue;
						try {
							const ev = JSON.parse(line.slice(6));
							opts.seen?.(ev);
							onEvent(live, ev, from);
						} catch {
							// one bad frame must not end the stream
						}
					}
				}
			}
		} catch {
			// server down or restarting, or silent too long
		} finally {
			clearTimeout(quiet);
		}
		if (session.aborted) break;          // replaced (another server) or the session ended: the new follower owns live
		if (!opts.pending?.()) {              // a stream not yet in use failing says nothing of the one in use
			live.connected = false;
			live.requestRender?.();
		}
		await new Promise((r) => setTimeout(r, backoff));
		backoff = Math.min(backoff * 2, 15000);
	}
}

// The longest the replaced stream is kept beside the new one (makeFollower).
export const OVERLAP_MS = 3000;

/** One live-stream follower at a time: restart() aborts the running one and starts another on live.base; stop()
 * ends it. resubscribe() changes what is asked for (liveUrl: expert routing, lens top 10s, writes) without a gap: a new stream
 * is opened beside the one in use, which is kept once the new one answers until it delivered an event a newer stream
 * delivered too (a stream keeps its order, so it has then given every event queued for it before the newer one
 * existed), at most OVERLAP_MS; onEvent takes what several deliver once. At most one stream waits to replace the one
 * in use: a newer resubscribe replaces it. `run` is follow (a stand-in in tests). */
export function makeFollower(live: Live, run: (live: Live, signal: AbortSignal, opts?: FollowOpts) => Promise<void> = follow) {
	type Stream = { c: AbortController; n: number };   // `n`: the order the streams were opened in
	let current: Stream | undefined;   // the stream in use
	let waiting: Stream | undefined;   // the one replacing it, until it answers
	let opened = 0;
	// The streams replaced, each kept until it caught up (OVERLAP_MS at most); while any is: each event's streams by n.
	let old: (Stream & { timer: ReturnType<typeof setTimeout> })[] = [];
	const by = new Map<string, number[]>();
	const drop = (o: (typeof old)[number]) => {
		clearTimeout(o.timer);
		o.c.abort();
		old = old.filter((x) => x !== o);
		if (!old.length) by.clear();
	};
	const dropAll = () => old.forEach(drop);
	const start = (onOpen?: (s: Stream) => void): Stream => {
		const s = { c: new AbortController(), n: ++opened };
		void run(live, s.c.signal, {
			onOpen: onOpen && (() => onOpen(s)),
			pending: () => s !== current,   // waiting, or replaced: its failures say nothing of the server
			seen: (ev) => {
				if (!old.length || typeof ev?.rid !== "string") return;
				const k = `${ev.type}:${ev.rid}:${ev.i ?? ev.read ?? ""}`, ns = by.get(k) ?? [];
				if (!ns.includes(s.n)) by.set(k, [...ns, s.n]);
				const now = by.get(k)!;
				for (const o of [...old]) if (now.includes(o.n) && now.some((m) => m > o.n)) drop(o);
			},
		});
		return s;
	};
	return {
		restart() {
			dropAll();
			waiting?.c.abort();
			waiting = undefined;
			current?.c.abort();
			live.connected = false;
			current = start();
		},
		resubscribe() {
			if (!current) return;
			waiting?.c.abort();
			waiting = start((s) => {
				if (waiting !== s) return;
				if (current) {
					const o = { ...current, timer: setTimeout(() => drop(o), OVERLAP_MS) };
					old = [...old, o];
				}
				current = s;
				waiting = undefined;
			});
		},
		stop() {
			dropAll();
			waiting?.c.abort();
			current?.c.abort();
			waiting = current = undefined;
		},
		get running() {
			return !!current;
		},
	};
}

/** Move the live stream to `base` (the selected model's server). The reply being followed and the steer labels
 * belong to the old server's request, whose end that server's stream would bring: dropped, so the dock does not wait
 * on it. False when already there. */
export function switchServer(live: Live, base: string, follower: { restart(): void; running: boolean }): boolean {
	if (base === live.base) return false;
	live.base = base;
	live.reply = undefined;
	live.prev = undefined;
	live.sentSteer = undefined;
	live.requestSteer = undefined;
	live.dense = undefined;
	if (follower.running) follower.restart();
	live.requestRender?.();
	return true;
}

/**
 * Both backends' models in /model whether their server runs or not, and the switch that makes a picked model the
 * one served. The metalscope command line knows both backends (`metalscope status --json`: oMLX's models, ds4's
 * GGUFs, each with the context it runs at) and starts them (`metalscope ds4 NAME`: oMLX's model unloaded first,
 * ds4 serving NAME in the background; `metalscope omlx NAME`: ds4 stopped, NAME loaded). One model fits at a time
 * and neither server knows the other's memory (oMLX refuses a load while ds4 holds its model), so the extension
 * makes the switch before a request goes out (index.ts: on /model's pick, and before each request).
 *
 * - cliCommand: how to run metalscope (the Python `metalscope pi install` recorded, else `metalscope` on PATH).
 * - readCatalogue: `status --json`; undefined when metalscope can't be run (no metalscope model is registered then).
 * - registerCatalogue: the providers metalscope-omlx and metalscope-ds4 from it, and their servers.
 * - isServed / servedCache / switchArgs / runSwitch: whether a model is served now, and the command that makes it so.
 * - confirmSwitch: before a switch, what it would stop (a job, replies in progress), and a yes first when anything runs.
 */

import { execFile, spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { LearnBackend } from "./concept.ts";
import { jobName } from "./jobs.ts";
import { type Candidate, DS4_URL, type ModelFlags, PROVIDER, type PiModel, type Server, listModels, piModel, providerConfig, scopeAuth } from "./provider.ts";

export interface CatalogueModel extends ModelFlags {
	name: string;
	ctx: number | null;
	loaded?: boolean;
}
export interface Catalogue {
	omlx: { models: CatalogueModel[] };
	ds4: { models: CatalogueModel[] };
}
export const OMLX = `${PROVIDER}-omlx`, DS4 = `${PROVIDER}-ds4`;

/** Each backend: its provider, its engine, where it answers (oMLX: from its settings, passed in; ds4: its proxy) and
 * its models in the catalogue. */
const BACKENDS: [string, string, (omlx: Candidate) => Candidate, (c: Catalogue) => CatalogueModel[]][] = [
	[OMLX, "omlx", (omlx) => ({ base: omlx.base, key: omlx.key }), (c) => c.omlx.models],
	[DS4, "ds4", () => ({ base: DS4_URL, key: "" }), (c) => c.ds4.models],
];

/** metalscope's data directory: METALSCOPE_HOME (empty counts as unset), else ~/.metalscope. */
export const metalscopeHome = (env: Record<string, string | undefined> = process.env, home = homedir()) => env.METALSCOPE_HOME || join(home, ".metalscope");

/** The metalscope command: `metalscope pi install` records its Python in ~/.metalscope/cli.json, for Pi alone (Pi's
 * PATH may not hold metalscope; the server's jobs use the Python `omlx install` recorded); else `metalscope` on PATH. */
export function cliCommand(env: Record<string, string | undefined> = process.env): string[] {
	try {
		const c = JSON.parse(readFileSync(join(metalscopeHome(env), "cli.json"), "utf8"));
		if (typeof c.python === "string" && c.python) return [c.python, "-m", "metalscope.cli"];
	} catch {}
	return ["metalscope"];
}

export function readCatalogue(cmd = cliCommand(), timeoutMs = 8000, signal?: AbortSignal): Promise<Catalogue | undefined> {
	return new Promise((resolve) => {
		execFile(cmd[0], [...cmd.slice(1), "status", "--json"], { timeout: timeoutMs, maxBuffer: 4 << 20, signal }, (err, stdout) => {
			if (err) return resolve(undefined);
			try {
				const c = JSON.parse(stdout);
				resolve(c?.omlx && c?.ds4 ? (c as Catalogue) : undefined);
			} catch {
				resolve(undefined);
			}
		});
	});
}

export const modelsOf = (models: CatalogueModel[]): PiModel[] => models.map((m) => piModel(m.name, m.ctx, m));

/** A backend as the extension's servers map holds it: its base, key, what it has loaded, each model's thinking
 * controls and flags. */
function serverOf(engine: string, c: Candidate, models: CatalogueModel[]): Server {
	return {
		...c,
		status: {
			engine,
			models: models.filter((m) => m.loaded).map((m) => ({ name: m.name })),
			chat: Object.fromEntries(models.map((m) => [m.name, { reasoning: !!m.reasoning, thinking_format: m.thinking_format ?? null, tested: m.tested, edit: m.edit }])),
		},
	};
}

/** Each backend as its provider (metalscope-omlx at oMLX's address and key, metalscope-ds4 at the ds4 proxy's), with
 * its models whether its server runs or not; `servers` updated. /model asks again each time it opens (refreshModels).
 * The names registered. */
export function registerCatalogue(pi: ExtensionAPI, cat: Catalogue, omlx: Candidate, servers: Map<string, Server>, refresh: () => Promise<Catalogue | undefined>,
	stream?: Parameters<typeof providerConfig>[4]): string[] {
	const added: string[] = [];
	for (const [name, engine, at, pick] of BACKENDS) {
		const cand = at(omlx);
		const models = pick(cat); // none yet (oMLX never seen, no GGUF): registered all the same, /model fills it later
		try {
			pi.registerProvider(name, providerConfig(engine, cand, modelsOf(models), async () => {
				const now = await refresh();
				const ms = now ? pick(now) : models;
				if (now) servers.set(name, serverOf(engine, cand, ms));
				return modelsOf(ms);
			}, stream));
		} catch {
			continue; // Pi refused it (a stale extension after /reload): the other still registers
		}
		servers.set(name, serverOf(engine, cand, models));
		added.push(name);
	}
	return added;
}

/** The registered backends' servers as the catalogue now says (what is loaded, after a switch). */
export function updateServers(cat: Catalogue, omlx: Candidate, servers: Map<string, Server>): void {
	for (const [name, engine, at, pick] of BACKENDS) if (servers.has(name)) servers.set(name, serverOf(engine, at(omlx), pick(cat)));
}

async function listed(base: string, id: string, key = ""): Promise<boolean | undefined> {
	try {
		return ((await listModels({ base, key }, 2000)) ?? []).some((m) => m.id === id);
	} catch {
		return undefined; // not answering
	}
}

export type Ds4Serving = { pid: number; name?: string } | undefined;

/** What `ds4 serve` runs, loading or serving: its record's process and model while that process lives (its ports
 * stay closed while it loads, minutes for V4.1). */
export function ds4Serving(env: Record<string, string | undefined> = process.env): Ds4Serving {
	try {
		const rec = JSON.parse(readFileSync(join(metalscopeHome(env), "ds4", "serving.json"), "utf8"));
		if (!Number.isInteger(rec.pid)) return undefined;
		try {
			process.kill(rec.pid, 0);
		} catch (e: any) {
			if (e?.code !== "EPERM") return undefined;   // EPERM: alive, another user's
		}
		return { pid: rec.pid, name: typeof rec.name === "string" ? rec.name : undefined };
	} catch {
		return undefined;
	}
}

/** Whether a request to this model can go now: ds4 serves it (its record names it, or there is none, and its proxy
 * lists it), or, for oMLX's, ds4 is not running, not even loading (its model would leave oMLX no room), and oMLX
 * lists it (oMLX loads its own on the request). The record and ds4's port are read on every call (another client
 * can switch); only a model listing's yes is kept (`is`) until `forget` (/model, a failed request): ds4's for the
 * process that serves it. */
export function servedCache(serving: () => Ds4Serving = ds4Serving) {
	let known: string | undefined;
	const kept = async (key: string | undefined, check: () => Promise<boolean | undefined>) => {
		if (key && known === key) return true;
		const ok = (await check()) === true;
		known = ok ? key : undefined;
		return ok;
	};
	return {
		async is(provider: string, id: string, omlx: Candidate): Promise<boolean> {
			const ds4 = serving();
			if (provider === DS4) return (!ds4 || ds4.name === id) && kept(ds4 && `${DS4}/${ds4.pid}/${id}`, () => listed(DS4_URL, id));
			if (provider === OMLX) return !ds4 && (await listed(DS4_URL, id)) === undefined && kept(`${OMLX}/${id}`, () => listed(omlx.base, id, omlx.key));
			return true;
		},
		forget: () => void (known = undefined),
	};
}

/** servedCache's check, nothing kept. */
export const isServed = (provider: string, id: string, omlx: Candidate, serving: () => Ds4Serving = ds4Serving) =>
	servedCache(serving).is(provider, id, omlx);

/** What a switch would stop on one server: the job running (GET /scope/api/jobs: `latest`, unfinished; every server
 * reads the same job store, so each reports it) and the replies in progress there (/scope/api/status: `active`);
 * undefined when it answers neither (nothing there to stop). */
export async function runningOn(base: string, ms = 2000): Promise<{ job?: { id: string; name: string }; replies: number } | undefined> {
	const get = async (path: string) => {
		try {
			const res = await fetch(`${base}/scope${path}`, { headers: scopeAuth(base), signal: AbortSignal.timeout(ms) });
			return res.ok ? await res.json() : undefined;
		} catch {
			return undefined;
		}
	};
	const [status, jobs] = await Promise.all([get("/api/status"), get("/api/jobs")]);
	if (!status && !jobs) return undefined;
	const latest = jobs?.latest;
	const job = latest && latest.outcome == null && typeof latest.name === "string" ? { id: String(latest.job), name: jobName(latest) } : undefined;
	return { job, replies: Number.isInteger(status?.active) ? status.active : 0 };
}

/** A backend as the user calls it: "oMLX", "ds4". */
export const engineLabel = (engine?: string) => (engine === "omlx" ? "oMLX" : engine ?? "the server");

/** What a switch would stop on the registered servers, a line each ("a job is running: calibrate fluffy", once however
 * many servers report it; "ds4 is loading X": its record live, its port not answering yet, unless the switch is to
 * X on ds4 (`keep`), which waits for it; "ds4 has 1 reply in progress"); none when nothing runs. Every server
 * counts: `metalscope ds4` unloads oMLX's model and restarts ds4 on another GGUF, `metalscope omlx` stops ds4. */
export async function whatSwitchStops(servers: Iterable<Server>, on = runningOn, serving: () => Ds4Serving = ds4Serving, keep?: string): Promise<string[]> {
	const all = [...new Map([...servers].map((s) => [s.base, s])).values()];
	const found = await Promise.all(all.map((s) => on(s.base)));
	const jobs = new Map(found.flatMap((r) => (r?.job ? [[r.job.id, r.job.name] as const] : [])));
	const ds4 = serving(), loading = ds4 && ds4.name !== keep && !found[all.findIndex((s) => s.base === DS4_URL)];
	return [...[...jobs.values()].map((name) => `a job is running: ${name}`), ...(loading ? [`ds4 is loading ${ds4.name ?? "a model"}`] : []),
		...all.flatMap((s, i) => {
			const n = found[i]?.replies ?? 0;
			return n > 0 ? [`${engineLabel(s.status.engine)} has ${n} ${n === 1 ? "reply" : "replies"} in progress`] : [];
		})];
}

/** Before a switch: go on when nothing runs; a job running: reject with `refused` and the way out, without asking
 * (metalscope refuses every switch while one runs); else ask (Pi's confirm), and with no yes (no UI, or no) reject
 * with `refused` and what to do instead (holdRequest then stops the request). Another client's reply is never cut
 * short without a yes. */
export async function confirmSwitch(ctx: any, id: string, where: string, servers: Iterable<Server>, on = runningOn,
	serving: () => Ds4Serving = ds4Serving): Promise<void> {
	const stops = await whatSwitchStops(servers, on, serving, where === "ds4" ? id : undefined);
	if (!stops.length) return;
	const jobs = stops.filter((l) => l.startsWith("a job is running"));
	if (jobs.length)
		throw Object.assign(new Error(`${id} not started on ${where}: ${jobs.join("; ")}. /scope jobs cancel stops it; then send again.`), { refused: true });
	const facts = stops.map((l) => `${l}.`);
	const ok = ctx?.hasUI ? await ctx.ui.confirm(`Start ${id} on ${where}?`,
		[...facts, `Starting ${id} now stops ${stops.length === 1 ? "it" : "them"}.`, `No: nothing changes, and no message goes to ${id} meanwhile.`].join("\n")) : false;
	if (ok) return;
	throw Object.assign(new Error([`${id} not started on ${where}: no message goes to it meanwhile.`, ...facts,
		`Send your message again once ${stops.length === 1 ? "it is" : "they are"} done: ${id} starts then.`].join("\n")), { refused: true });
}

/** Before a request: wait for the switch (Esc stops waiting; the switch goes on). A refused or failed switch stops the
 * request, since Pi sends it whatever this hook throws, and it would go to a model the switch did not start (oMLX
 * would load it beside what runs): ctx.abort() aborts the agent's signal, which the request checks before it goes
 * out. With no UI (print mode) the reason goes to stderr; with one, the switch's own note says it. */
export async function holdRequest(served: Promise<void>, ctx: any, write = (s: string) => void process.stderr.write(s)): Promise<void> {
	let failed: any;
	const abort: AbortSignal | undefined = ctx.signal;
	await Promise.race([served.catch((e) => void (failed = e ?? new Error("the switch failed"))),
		new Promise<void>((r) => abort?.addEventListener("abort", () => r(), { once: true }))]);
	if (!failed) return;
	if (!ctx.hasUI) write(`${failed.message ?? failed}\n`);
	ctx.abort();
}

/** The selected model's backend, as the job times are given (concept.ts LEARN_MINUTES): "ds4" for ds4's, else "omlx". */
export const backendOf = (ctx: { model?: { provider?: string } } | undefined): LearnBackend => (ctx?.model?.provider === DS4 ? "ds4" : "omlx");

/** The metalscope words that make this model the one served; undefined for a model that is not a backend's. */
export function switchArgs(provider: string, id: string): string[] | undefined {
	return provider === DS4 ? ["ds4", id] : provider === OMLX ? ["omlx", id] : undefined;
}

/** Run a switch, each line it prints to onLine; resolves when it is done, rejects with its last lines. */
export function runSwitch(args: string[], onLine: (line: string) => void, cmd = cliCommand()): Promise<void> {
	return new Promise((resolve, reject) => {
		let child;
		try {
			child = spawn(cmd[0], [...cmd.slice(1), ...args], { stdio: ["ignore", "pipe", "pipe"] });
		} catch (e) {
			return reject(e);
		}
		const tail: string[] = [];
		const take = (line: string) => {
			if (!(line = line.trimEnd())) return;
			tail.push(line);
			if (tail.length > 6) tail.shift();
			onLine(line);
		};
		// Whole lines only: a chunk may end mid-line (or mid-character), its rest kept for the next.
		for (const s of [child.stdout, child.stderr]) {
			let rest = "";
			s.setEncoding("utf8");
			s.on("data", (text: string) => {
				const ls = (rest + text).split("\n");
				rest = ls.pop()!;
				ls.forEach(take);
			});
			s.on("end", () => take(rest));
		}
		child.on("error", (e) => reject(new Error(`metalscope could not run (${e.message}): metalscope pi install records it`)));
		child.on("close", (code) => (code === 0 ? resolve() : reject(new Error(tail.join("\n") || `metalscope ${args.join(" ")} exited ${code}`))));
	});
}

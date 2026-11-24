/**
 * The GPU jobs: starting /concept learn, /concept calibrate and /jlens learn, /scope jobs and /scope
 * jobs cancel, the stop key too (jobs.ts and jlens.ts hold the pure logic; commands.ts registers /concept, /scope and /jlens and calls
 * `Jobs`).
 *
 * Starting a learn job or a lens fit asks first: the server's `concept learn` and `jlens learn` unload the served
 * model for their GPU part (a lens fit until its deadline), so chat pauses; on
 * ds4 `concept learn` measures the served model instead, and the ds4 proxy refuses chat meanwhile. A calibrate job
 * only sends chat requests and starts at once.
 * While a job runs, a widget above the editor shows its steps, a progress bar and the time left, polled from GET
 * /scope/api/jobs every 3 seconds on the selected model's server (every server reads the same job store, so picking
 * another backend's model keeps it in view);
 * metalscope's routes keep answering while the model is unloaded. When it finishes, the result lands in the
 * transcript as a job entry (entries.ts "scope-job"), never sent to the model, in the session that watched it run. A fresh Pi session
 * stays clean: a running job gets its widget back, and a result Pi missed (a job started from Pi that finished
 * while it was closed) gets one line pointing at /scope jobs; jobs started from the command line are not
 * announced.
 */

import { backendOf } from "./backends.ts";
import { type ConceptFlags, calibrateTime, type LearnPlan, sayWhatText } from "./concept.ts";
import { flat, oneIn } from "./draw.ts";
import { type LensLearnPlan, lensLearnConfirmText, resolveLensPath } from "./jlens.ts";
import { type JobLook, type LearnState, SPIN, STOP_ASK, STOP_LABEL, confirmText, doing, isStopKey, jobName, jobOutcome, jobsView, learnStatus,
	learnWidgetLines, nextStep, startupAction, startupNotice, trimJob } from "./jobs.ts";
import { type Ext, modelShort, requestFailed } from "./provider.ts";

const POLL_MS = 3000;
/** The widget's spinner: a frame, and a render of Pi asked for, this often while a job runs. */
export const SPIN_MS = 100;
const KEY = "metalscope-jobs";
const JOBS = "/api/jobs";

/** What starts, shows and stops the jobs (/concept learn and calibrate, /jlens learn, /scope jobs and
 * jobs cancel); registerJobs returns it. */
export interface Jobs {
	learn(ctx: any, name: string, description: string, flags?: ConceptFlags): Promise<void>;
	calibrate(ctx: any, name: string): Promise<void>;
	lensLearn(ctx: any, a: { path?: string; until?: string; windows?: number; fresh?: boolean }): Promise<void>;
	jobs(ctx: any): Promise<void>;
	/** /scope jobs cancel; `confirm` (the stop key): asked first. */
	cancel(ctx: any, confirm?: boolean): Promise<void>;
}

/** `shown(ctx)`: the job widget just appeared (index.ts sets the steer's line again, so it stays nearest the editor). */
export function registerJobs({ pi, live, api }: Ext, shown?: (ctx: any) => void): Jobs {
	let timer: ReturnType<typeof setInterval> | undefined;
	let ctxRef: any;
	let polling = false;                      // one poll at a time: a slow answer must not let two polls both finish a job
	const finished = new Set<string>();       // jobs whose result this session has already shown
	const sawRunning = new Set<string>();     // jobs this session watched while they ran: only these get the full result entry
	const warned = new Set<string>();         // warnings already notified (the widget keeps showing them)
	let asking = false;                       // the stop key's question asked (or its job being looked up)
	let unlisten: (() => void) | undefined;   // the stop key's raw input handler (listen)

	// The widget drawn by the extension (Pi cuts a widget of plain lines at 10 and wraps them): its lines built at each
	// render from the job as the last poll had it (`job`) and the time then, so the spinner turns and the clock ticks;
	// a render asked for every SPIN_MS until Pi disposes it (removed), the poll stops or the session ends (`unspin`).
	// Pi renders it from its own timer, outside any try: a render never throws (the last lines drawn at that width,
	// else none).
	let unspin: (() => void) | undefined;
	let job: LearnState | undefined;
	const widget = (tui: any, th: JobLook["th"]) => {
		const spin = setInterval(() => tui.requestRender(), SPIN_MS);
		const dispose = (unspin = () => clearInterval(spin));
		let last = { width: -1, lines: [] as string[] };
		return {
			render: (width: number) => {
				try {
					const ms = Date.now(), frame = SPIN[Math.floor(ms / SPIN_MS) % SPIN.length];
					last = { width, lines: oneIn(width, (w) => learnWidgetLines(job!, ms / 1000, { width: w, th, spin: frame, key: STOP_LABEL })) };
				} catch {}
				return last.width === width ? last.lines : [];
			},
			invalidate() {},
			dispose,
		};
	};
	// Set only as a job comes or goes: Pi puts a widget set again last, nearest the editor, which is the steer line's
	// place (`shown`).
	const show = (st: LearnState | undefined) => {
		if (!ctxRef?.hasUI) return;
		const flip = !st !== !job;
		if (flip) ctxRef.ui.setWidget(KEY, st ? widget : undefined);
		job = st;
		if (flip && st) shown?.(ctxRef);
		ctxRef.ui.setStatus(KEY, st ? learnStatus(st) : undefined);
	};
	const stop = () => {
		if (timer) clearInterval(timer);
		timer = undefined;
		unspin?.();
	};
	const latest = async (get = api): Promise<LearnState | undefined> => (await get(JOBS))?.latest ?? undefined;

	async function finish(st: LearnState) {
		stop();
		show(undefined);
		// Another Pi window may have marked it seen first: the window that watched it run shows its result all the same.
		const watched = sawRunning.has(st.job);
		if ((st.seen && !watched) || finished.has(st.job) || !ctxRef) return;    // no ctx: the session ended while the poll was out
		finished.add(st.job);
		if (watched) {
			pi.appendEntry("scope-job", { v: 1, job: trimJob(st) });
			ctxRef?.ui?.notify(`${jobName(st)}: ${jobOutcome(st)}`, st.outcome === "done" ? "info" : "warning");
		} else {
			ctxRef?.ui?.notify(startupNotice(st), st.outcome === "done" ? "info" : "warning");
		}
		if (!st.seen) await api(`${JOBS}/${encodeURIComponent(st.job)}/seen`, { method: "POST" }).catch(() => undefined);
	}

	async function poll() {
		if (polling) return;
		polling = true;
		try {
			let st: LearnState | undefined;
			try {
				st = await latest();
			} catch {
				return; // the server is restarting or busy: try again next tick
			}
			if (!ctxRef || !st) {
				stop();
				return show(undefined);       // no job left in the store: no widget either
			}
			if (st.outcome == null) {
				sawRunning.add(st.job);
				show(st);
				if (st.warning && !warned.has(`${st.job} ${st.warning}`)) {
					warned.add(`${st.job} ${st.warning}`);
					ctxRef?.ui?.notify(`${jobName(st)}: ${st.warning}`, "warning");
				}
			}
			else await finish(st);
		} catch {
			// Anything else (a stale ctx, a malformed answer) is dropped, never thrown: poll runs from setInterval, where a
			// rejection is unhandled and ends Pi. The next tick polls again.
		} finally {
			polling = false;
		}
	}

	const watch = () => {
		if (timer) return;
		timer = setInterval(() => void poll(), POLL_MS);
		void poll();
	};

	// The stop key, from the terminal's raw input: Pi gives it to its listeners before the focused component (pi-tui
	// handleTerminalInput), so it works from the editor, the dock and a dialog alike, and Option+x's ≈ is caught where the
	// terminal does not send Option as Alt. Only while a job is shown, consumed then; the question once (cancel's
	// `asking`). Pi calls it with nothing around it: it never throws. One per session; no terminal (pi -p, RPC), none.
	const listen = (ctx: any) => {
		try {
			unlisten?.();
			unlisten = ctx?.ui?.onTerminalInput?.((data: string) => {
				if (!job || !isStopKey(data)) return undefined;
				jobs.cancel(ctx, true).catch(() => {});
				return { consume: true };
			});
		} catch {}
	};

	pi.on("session_start", (_event, ctx) => {
		ctxRef = ctx;
		job = undefined;                        // a new session has no widgets yet
		listen(ctx);
		void (async () => {
			try {
				// A fresh session stays clean: a running job gets its widget, a result Pi missed gets one line (startupAction).
				const st = await latest();
				const action = startupAction(st);
				if (action === "watch") watch();
				else if (action === "notify" && st) await finish(st);
			} catch {}
		})();
	});
	pi.on("session_shutdown", () => {
		stop();
		listen(undefined);
		ctxRef = undefined;
	});

	/** Run `fn`, reporting what failed as an error: a request made through `ask` in requestFailed's words (a 409 says
	 * why a job cannot start), anything else (a bug here) as itself, never as the server not answering. */
	const guarded = async (ctx: any, what: string, fn: (ask: Ext["api"]) => Promise<void>) => {
		ctxRef = ctx;
		const ask: Ext["api"] = (...a) => api(...a).catch((e: any) => {
			throw new Error(requestFailed(e, live.base, ctx));
		});
		try {
			await fn(ask);
		} catch (err) {
			ctx.ui.notify(`${what}: ${(err as Error).message}`, "error");
		}
	};

	const jobs: Jobs = {
		learn: (ctx, name, description, flags = {}) => guarded(ctx, "learn", async (ask) => {
			const model = modelShort(ctx);
			// The selected model's server runs the job: on ds4 it measures the served model, on oMLX its own copy.
			const backend = backendOf(ctx);
			const corpus = flags.corpus ? resolveLensPath(flags.corpus, ctx.cwd ?? process.cwd()) : undefined;
			// The server says whose texts it uses (LearnPlan source), or why not here (`taken`, read first: a plan refused
			// early still says "write"). Written ones need a description: typed, else the saved one.
			const plan: LearnPlan = await ask(`/api/concept/learn?name=${encodeURIComponent(name)}&model=${encodeURIComponent(model)}` +
				(corpus ? `&corpus=${encodeURIComponent(corpus)}` : ""));
			if (plan.taken) return ctx.ui.notify(`/concept learn ${name}: ${plan.taken}`, "error");
			if (plan.source === "write" && !description && !plan.description) return ctx.ui.notify(sayWhatText(name), "error");
			const ok = await ctx.ui.confirm(`/concept learn ${name}`, confirmText(name, description, model || "the served model", backend, plan, corpus));
			if (!ok) return ctx.ui.notify("not started", "info");
			// As typed: the server fills in the saved description.
			const res = await ask("/api/concept/learn", { method: "POST", body: JSON.stringify({ name, description, model, corpus }) });
			ctx.ui.notify(`learning ${name} (job ${res.job}): the widget above the editor shows progress; the result arrives here`, "info");
			watch();
		}),
		calibrate: (ctx, name) => guarded(ctx, "calibrate", async (ask) => {
			const res = await ask("/api/concept/calibrate", { method: "POST", body: JSON.stringify({ name, model: modelShort(ctx) }) });
			ctx.ui.notify(`calibrating ${name} (job ${res.job}): conversations graded by the model, chat requests only, about ${calibrateTime(backendOf(ctx), modelShort(ctx))}; ` +
				"chat works meanwhile. The widget above the editor shows progress; the result arrives here", "info");
			watch();
		}),
		lensLearn: (ctx, a) => guarded(ctx, "jlens learn", async (ask) => {
			const model = modelShort(ctx);
			const path = a.path ? resolveLensPath(a.path, ctx.cwd ?? process.cwd()) : undefined;
			const q = new URLSearchParams({ model });
			if (a.until) q.set("until", a.until);
			if (a.windows) q.set("windows", String(a.windows));
			if (path) q.set("path", path);
			const plan: LensLearnPlan = await ask(`/api/lens/learn?${q}`);
			const ok = await ctx.ui.confirm("/jlens learn", lensLearnConfirmText(plan));
			if (!ok) return ctx.ui.notify("not started", "info");
			const res = await ask("/api/lens/learn", { method: "POST", body: JSON.stringify({ model, path, until: a.until, windows: a.windows, fresh: a.fresh }) });
			ctx.ui.notify(`learning a lens for ${model} (job ${res.job})${a.until ? ` until ${a.until}` : ""}: the widget above the editor shows each window's held-out score; the result arrives here`, "info");
			watch();
		}),
		// The running or last job, then the last ten: a running one as a note (and its widget), a finished one's result
		// as a transcript entry (jobsView).
		jobs: (ctx) => guarded(ctx, "jobs", async (ask) => {
			const res = await ask(JOBS);
			const view = jobsView(res?.latest ?? undefined, res?.jobs ?? []);
			if (view.kind === "running") watch();
			if (view.kind === "done") return void pi.appendEntry("scope-job", { v: 1, job: view.job, recent: view.recent });
			ctx.ui.notify(view.text, "info");
		}),
		// The stop key (`confirm`): asked first, under the question the command its result entry will give once stopped; "no" is
		// the choice enter takes (a ≈ typed in a message opens the question too); a press while one is being asked does
		// nothing.
		cancel: async (ctx, confirm = false) => {
			if (confirm && asking) return;
			asking ||= confirm;
			try {
				await guarded(ctx, "cancel", async (ask) => {
					const st = await latest(ask);
					if (!st || st.outcome != null) return ctx.ui.notify("no job is running", "info");
					if (confirm && (await ctx.ui.select(`Stop ${flat(doing(st))}?\n${flat(nextStep({ ...st, outcome: "interrupted" }).join(""))}`, STOP_ASK)) !== STOP_ASK[1]) return;
					const res = await ask(`${JOBS}/${encodeURIComponent(st.job)}/cancel`, { method: "POST" });
					const after = st.kind === "calibrate" ? "" : `: the model is reloaded when it has stopped${st.kind === "lens-learn" ? "; the windows done are kept" : ""}`;
					ctx.ui.notify(res?.cancelled ? `stopping ${jobName(st)}${after}` : "it had already stopped", "info");
				});
			} finally {
				if (confirm) asking = false;       // a thrown notify (a stale ctx) must not leave the stop key dead
			}
		},
	};
	return jobs;
}

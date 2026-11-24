import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { visibleWidth } from "@earendil-works/pi-tui";
import { onEvent } from "../state.ts";
import { steerState } from "../steer.ts";
import { DOCK_MIN_COLS, FIRST_RUN_HINT, createUi, dockColumns, dockOverlay, firstRunHint, prefsPath, readPrefs, reserveColumns, steerLine, writePrefs } from "../widget.ts";

test("firstRunHint: said once per install, at a session with a UI, and recorded in pi.json beside the dock", () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	const path = join(dir, "pi.json");
	const notes: [string, string][] = [];
	const ctx = { hasUI: true, ui: { notify: (m: string, kind: string) => notes.push([m, kind]) } };
	firstRunHint({ ...ctx, hasUI: false }, path);                            // pi -p: nothing said, nothing recorded
	assert.deepEqual(notes, []);
	assert.deepEqual(readPrefs(path), {});
	writePrefs({ dock: false }, path);
	createUi({ live: {} } as any, path).firstRunHint(ctx);                    // the first session start
	assert.deepEqual(notes, [[FIRST_RUN_HINT, "info"]]);
	assert.deepEqual(readPrefs(path), { dock: false, hinted: true });       // the dock's preference kept
	firstRunHint(ctx, path);                                                  // every later one: nothing
	createUi({ live: {} } as any, path).firstRunHint(ctx);
	assert.equal(notes.length, 1);
	assert.equal(FIRST_RUN_HINT, "metalscope: replies are traced. ctrl+\\ opens the dock, /scope help the rest.");
});

test("prefsPath: METALSCOPE_HOME's pi.json, else ~/.metalscope's", () => {
	assert.equal(prefsPath({ METALSCOPE_HOME: "/x/h" }, "/home/u"), "/x/h/pi.json");
	assert.equal(prefsPath({}, "/home/u"), "/home/u/.metalscope/pi.json");
	assert.equal(prefsPath({ METALSCOPE_HOME: "" }, "/home/u"), "/home/u/.metalscope/pi.json");   // empty is unset, as cliCommand reads it
});

test("readPrefs / writePrefs: merged into the file; missing or broken reads as nothing remembered", () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	const path = join(dir, "sub", "pi.json");
	assert.deepEqual(readPrefs(path), {});
	writePrefs({ dock: true }, path);
	assert.deepEqual(readPrefs(path), { dock: true });
	writeFileSync(path, JSON.stringify({ dock: true, other: 1 }));
	writePrefs({ dock: false }, path);
	assert.deepEqual(JSON.parse(readFileSync(path, "utf8")), { dock: false, other: 1 });   // other keys kept
	writeFileSync(path, "{not json");
	assert.deepEqual(readPrefs(path), {});
	writeFileSync(path, "[1]");
	assert.deepEqual(readPrefs(path), {});
});

test("restoreDock: opens the dock only when it was left open, and records opening it", () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	const path = join(dir, "pi.json");
	let opened = 0;
	const ctx = { hasUI: true, model: { provider: "metalscope-omlx" },
		ui: { custom: () => (opened++, new Promise(() => {})), setWidget() {}, setStatus() {} } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false } as any;
	createUi({ live } as any, path).restoreDock(ctx);
	assert.equal(opened, 0);                                                  // nothing remembered: stays closed
	writePrefs({ dock: false }, path);
	createUi({ live } as any, path).restoreDock(ctx);
	assert.equal(opened, 0);
	writePrefs({ dock: true }, path);
	createUi({ live } as any, path).restoreDock({ ...ctx, hasUI: false });
	assert.equal(opened, 0);                                                  // no UI (pi -p): nothing to open
	createUi({ live } as any, path).restoreDock(ctx);
	assert.equal(opened, 1);
	assert.deepEqual(readPrefs(path), { dock: true });
});

test("dockColumns: the dock's width the way Pi resolves it, plus a gap; nothing where it is hidden", () => {
	assert.equal(dockColumns(160), Math.floor(160 * 0.36) + 1);
	assert.equal(dockColumns(110), 44 + 1);                                   // the minimum width
	assert.equal(dockColumns(DOCK_MIN_COLS - 1), 0);                          // narrower: hidden
	// The board's width (100) where the terminal spares it and leaves Pi 80 columns: from 181; 36% past it from 278.
	assert.deepEqual([180, 181, 277, 278, 300].map((c) => dockColumns(c)), [64 + 1, 100 + 1, 100 + 1, 100 + 1, 108 + 1]);
	assert.ok([181, 200, 277].every((c) => c - dockColumns(c) >= 80), "Pi keeps 80 columns");
	// The overlay's least width follows the terminal's columns, in step with dockColumns.
	let cols = 180;
	const o = dockOverlay(() => false, false, () => cols);
	assert.equal(o.minWidth, 44);
	cols = 181;
	assert.equal(o.minWidth, 100);
});

test("reserveColumns, regular mode: Pi's content renders narrower beside the dock, and full width again after", () => {
	class FakeTui {
		renders = 0;
		render(width: number) { return [`${width}`]; }
		requestRender() { this.renders++; }
	}
	const tui = new FakeTui();
	const room = reserveColumns(tui);
	assert.deepEqual(tui.render(160), [`${160 - dockColumns(160)}`]);
	assert.deepEqual(tui.render(99), ["99"]);                                 // too narrow for the dock: nothing reserved
	room.undo();
	assert.deepEqual(tui.render(160), ["160"]);
	assert.equal(tui.renders, 2);                                             // a full redraw on open and on close
	// A TUI whose render is already its own (another extension's): left alone.
	const own: any = { render: (w: number) => [`${w}`] };
	reserveColumns(own).undo();
	assert.deepEqual(own.render(160), ["160"]);
});

test("reserveColumns: a room left behind (Pi hid the dock without its close: /new, /reload) is replaced, not stacked", () => {
	const root = { render: () => ["pi"], invalidate() {} };
	const tui: any = { layoutRoot: root, terminal: { columns: 160 }, setLayoutRoot(r: any) { this.layoutRoot = r; }, requestRender() {} };
	reserveColumns(tui);                                                      // the old dock's, never undone
	const room = reserveColumns(tui);                                         // the restored dock's
	assert.deepEqual(tui.layoutRoot.entries.map((e: any) => e.component === root ? "pi" : "gap"), ["pi", "gap"]);
	room.undo();
	assert.equal(tui.layoutRoot, root);                                       // Pi's own root back
	class FakeTui {
		render(width: number) { return [`${width}`]; }
		requestRender() {}
	}
	const reg: any = new FakeTui();
	reserveColumns(reg);
	const again = reserveColumns(reg);
	assert.deepEqual(reg.render(160), [`${160 - dockColumns(160)}`]);        // one dock's width, not two
	again.undo();
	assert.deepEqual(reg.render(160), ["160"]);
});

test("detach: the dock Pi hid gives its columns back; the dock stays remembered as open", () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	const path = join(dir, "pi.json");
	const root = { render: () => ["pi"], invalidate() {} };
	const tui: any = { layoutRoot: root, terminal: { columns: 160 }, setLayoutRoot(r: any) { this.layoutRoot = r; }, requestRender() {} };
	let done: (() => void) | undefined;
	const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {},
		custom: (factory: any) => new Promise<void>((resolve) => { done = resolve; factory(tui, { fg: (_c: string, s: string) => s, bg: (_c: string, s: string) => s }, {}, resolve); }) } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false } as any;
	const ui = createUi({ live } as any, path);
	ui.showDock(ctx);
	assert.ok(ui.docked());
	assert.notEqual(tui.layoutRoot, root);
	ui.detach();
	assert.equal(tui.layoutRoot, root);
	assert.ok(!ui.docked());
	assert.deepEqual(readPrefs(path), { dock: true });                        // reopens at the next session start
	ui.showDock(ctx);                                                         // the next session's dock
	const room = tui.layoutRoot;
	done?.();                                                                 // a late close of the first: ignored
	assert.equal(tui.layoutRoot, room);
	assert.ok(ui.docked());
});

test("reserveColumns, fullscreen: Pi's layout root goes beside an empty column the dock's width, which follows a resize", () => {
	const root = { render: () => ["pi"], invalidate() {} };
	const tui: any = { layoutRoot: root, terminal: { columns: 160 }, set: [] as any[], setLayoutRoot(r: any) { this.layoutRoot = r; this.set.push(r); }, requestRender() {} };
	const room = reserveColumns(tui);
	const row = tui.layoutRoot;
	assert.notEqual(row, root);
	const entries = row.entries.map((e: any) => [e.component === root ? "pi" : "gap", e.basis, e.grow]);
	assert.deepEqual(entries, [["pi", undefined, 1], ["gap", dockColumns(160), 0]]);
	room.fit(160);
	assert.equal(tui.set.length, 1);                                          // same width: nothing to do
	room.fit(120);
	assert.equal(tui.layoutRoot.entries[1].basis, dockColumns(120));
	room.fit(99);
	assert.equal(tui.layoutRoot, root);                                       // the dock is hidden there: no gap
	room.undo();
	assert.equal(tui.layoutRoot, root);
});

test("expert routing and the lens top 10s are asked for while the board or the mini dock is shown", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	let close: () => void = () => {};
	let dock: any;
	const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {},
		custom: (make: any) => new Promise<void>((done) => { dock = make({ requestRender() {}, terminal: { rows: 40, columns: 160 } }, {}, {}, done); close = done; }) } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false,
		reply: { rid: "r", tokens: [{ i: 0, id: 0, text: " w", p: 0.9 }], done: true, prefill: 0 } } as any;
	let asks = 0;
	const ui = createUi({ live } as any, join(dir, "pi.json"), () => asks++);
	ui.showDock(ctx);
	assert.deepEqual([live.experts, live.lens, asks], [true, true, 1]);           // the board: both
	assert.equal(typeof live.requestRender, "function");
	close();
	await new Promise((r) => setTimeout(r, 0));
	assert.deepEqual([live.experts, live.lens, asks], [false, false, 2]);         // closed: off again
	assert.equal(live.requestRender, undefined, "closed: tokens no longer redraw Pi");
	live.dense = "Moe";
	ui.showDock(ctx);
	assert.deepEqual([live.experts, live.lens, asks], [true, true, 3]);
	assert.equal(live.dense, undefined);                                         // a model taken for dense is tried again
	dock.handleInput("m");                                                        // the mini dock: its lights, SETTLING and STREAM too
	assert.deepEqual([live.experts, live.lens, live.writes, asks], [true, true, true, 3], "nothing new to ask");
});

test("stepping in (ctrl+\\ again, alt+i, /scope dock) keeps following the newest token: nothing moves until a token key", () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: true } as any;
	onEvent(live, { type: "start", rid: "r", model: "m" });
	for (let i = 0; i < 5; i++) onEvent(live, { type: "token", rid: "r", i, id: i, text: ` w${i}`, p: i === 1 ? 0.1 : 0.9 } as any);   // #1 unsure, still streaming
	const theme = { fg: (_c: string, s: string) => s, bg: (_c: string, s: string) => s, bold: (s: string) => s };
	for (const open of [true, false]) {
		let dock: any;
		const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {},
			custom: (make: any) => new Promise<void>((done) => { dock = make({ requestRender() {}, terminal: { rows: 40, columns: 160 } }, theme, {}, done); }) } };
		const ui = createUi({ live } as any, join(dir, "pi.json"));
		if (open) ui.showDock(ctx);                                                 // open, then stepped in
		ui.showDock(ctx, true);                                                     // (closed: opened stepped in)
		assert.deepEqual([dock.stepped, dock.sel], [true, -1], open ? "stepped in" : "opened stepped in");
		const i = live.reply.tokens.length;
		onEvent(live, { type: "token", rid: "r", i, id: i, text: ` w${i}`, p: 0.9 } as any);   // the next token streams
		const shown = dock.render(104).map((l: string) => l.replace(/\x1b\[[0-9;]*m/g, ""));
		assert.ok(shown.some((l: string) => l.includes(`TOKENS  #${i} " w${i}"`)), `the board follows it to #${i}`);
		dock.handleInput("\x1b[D");                                                 // ←: paused on the token before
		ui.showDock(ctx, true);                                                     // ctrl+\ again while in: the pause stays
		assert.equal(dock.sel, i - 1);
		ui.hideDock();
	}
});

test("the mini dock's columns: MINI_WIDTH and a gap from 100 columns, as the full dock; chosen, never for a narrow terminal", () => {
	assert.equal(dockColumns(160, true), 54);
	assert.equal(dockColumns(100, true), 54);
	assert.equal(dockColumns(DOCK_MIN_COLS - 1, true), 0, "under 100: hidden");
	let mini = false, cols = 0;
	const o = dockOverlay(() => mini, false, () => cols);
	assert.deepEqual([o.width, o.minWidth, o.visible(99), o.visible(100), o.nonCapturing], ["36%", 44, false, true, true]);
	mini = true;
	assert.deepEqual([o.width, o.minWidth], [53, 53]);
	mini = false;
	cols = DOCK_MIN_COLS;
	assert.deepEqual([o.width, o.minWidth], ["36%", 44]);
	// Regular mode: Pi's content narrower by the mode's columns at every frame.
	const tui: any = { render: (w: number) => [`${w}`], requestRender() {} };
	Object.setPrototypeOf(tui, { render: (w: number) => [`${w}`] });
	delete tui.render;
	reserveColumns(tui, () => mini);
	assert.deepEqual([tui.render(160), tui.render(90)], [[`${160 - dockColumns(160)}`], ["90"]]);
	mini = true;
	assert.deepEqual([tui.render(160), tui.render(90)], [["106"], ["90"]]);
	// Fullscreen: the gap's basis refitted at the next fit (a resize, or layoutChanged).
	const root = { render: () => ["pi"], invalidate() {} };
	const fs: any = { layoutRoot: root, terminal: { columns: 120 }, setLayoutRoot(r: any) { this.layoutRoot = r; }, requestRender() {} };
	const room = reserveColumns(fs, () => mini);
	assert.equal(fs.layoutRoot.entries[1].basis, 54, "120 columns, mini chosen: its room");
	room.fit(99);
	assert.equal(fs.layoutRoot, root, "99 columns: no dock, no room");
	room.fit(120);
	assert.equal(fs.layoutRoot.entries[1].basis, 54);
	mini = false;
	room.fit(160);
	assert.equal(fs.layoutRoot.entries[1].basis, dockColumns(160));
});

test("widget: LENS's lens steps: GET for the served model, POST its step; a 404 said missing", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	let dock: any, fail: any;
	const calls: [string, any][] = [];
	const api = async (path: string, init?: any) => {
		calls.push([path, init && { method: init.method, body: JSON.parse(init.body) }]);
		if (fail) throw fail;
		return { model: "Flash Next", at: 2, steps: [] };
	};
	const ctx = { hasUI: true, model: { id: "org/Flash Next", provider: "metalscope-omlx" }, ui: { setWidget() {}, setStatus() {},
		custom: (make: any) => new Promise<void>(() => { dock = make({ requestRender() {}, terminal: { rows: 67, columns: 290 } }, {}, {}, () => {}); }) } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false } as any;
	createUi({ live, api } as any, join(dir, "pi.json"), () => {}).showDock(ctx);
	const host = dock.host;
	assert.deepEqual(await host.lensLayers(), { model: "Flash Next", at: 2, steps: [] });
	await host.lensLayers(3);
	assert.deepEqual(calls.filter(([p]) => p.startsWith("/api/lens/")), [["/api/lens/layers?model=Flash%20Next", undefined],
		["/api/lens/layers", { method: "POST", body: { model: "Flash Next", step: 3 } }]]);
	fail = Object.assign(new Error("Not Found"), { status: 404 });
	assert.equal((await host.lensLayers()).missing, true);
	fail = Object.assign(new Error("model 'Flash Next' is not loaded"), { status: 409 });
	const idle = await host.lensLayers();
	assert.ok(!idle.missing && /is not loaded/.test(idle.error), "409: the reason, said");
	fail = Object.assign(new Error("step must be 0 to 4"), { status: 400 });
	const bad = await host.lensLayers(9);
	assert.ok(!bad.missing && /step must be 0 to 4/.test(bad.error));
});

test("the steer line above the editor: ◆ steering LABEL, one concept's chat figure at its dose, /steer off; cut, never wrapped", () => {
	const th = { fg: (_c: string, s: string) => s };
	const part = (name: string, chat?: { good: number; turns: number }) => ({ name, model: "m", layer: 16, strength: 0.4, source: "profile" as const,
		...(chat ? { chat: { ...chat, line: `In chat: ${chat.good} of ${chat.turns} replies good (the most of the doses tried)` } } : {}) });
	const one = steerState([part("pirate", { good: 12, turns: 18 })], "m");
	assert.equal(steerLine(th, one, 99), " ◆ steering pirate@0.4 L16 · 12 of 18 good in chat · /steer off");
	assert.equal(steerLine(th, one, 60), " ◆ steering pirate@0.4 L16 · /steer off");          // narrow: the figure goes first
	assert.equal(steerLine(th, one, 30), " ◆ steering pirate@0.4 L16");                       // then /steer off
	assert.equal(steerLine(th, one, 20), " ◆ steering pirate@…");
	assert.equal(steerLine(th, steerState([part("pirate")], "m"), 99), " ◆ steering pirate@0.4 L16 · /steer off");   // no chat figure
	const two = steerState([part("pirate", { good: 12, turns: 18 }), part("calm", { good: 9, turns: 18 })], "m");
	assert.equal(steerLine(th, two, 99), " ◆ steering pirate@0.4 L16 + calm@0.4 L16 · /steer off");                // several: none
	for (const s of [one, two]) for (let w = 0; w <= 100; w++) assert.ok(visibleWidth(steerLine(th, s, w)) <= w, `${w}`);
	// ◆ in the accent colour, steering and its label plain, the rest dim.
	const tag = { fg: (c: string, s: string) => `<${c}>${s}</>` };
	assert.equal(steerLine(tag, one, 99), " <accent>◆</> steering pirate@0.4 L16<dim> · 12 of 18 good in chat · /steer off</>");
});

test("the steer line comes and goes with live.steer, the dock open or not; the footer has no steer tag", () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	const widgets = new Map<string, any>(), statuses: (string | undefined)[] = [];
	const ctx = { hasUI: true, model: { provider: "metalscope-omlx", id: "m" },
		ui: { setWidget: (k: string, f: any) => void (f ? widgets.set(k, f) : widgets.delete(k)), setStatus: (_k: string, s?: string) => statuses.push(s) } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: true } as any;
	const ui = createUi({ live } as any, join(dir, "pi.json"));
	ui.updateUi(ctx);
	assert.ok(!widgets.has("metalscope-steer"));
	live.steer = steerState([{ name: "pirate", model: "m", layer: 16, strength: 0.4, source: "profile", chat: { good: 12, turns: 18, line: "In chat: 12 of 18 replies good (the most of the doses tried)" } }], "m");
	ui.updateUi(ctx);
	const line = widgets.get("metalscope-steer")({}, { fg: (_c: string, s: string) => s });
	assert.deepEqual(line.render(99), [" ◆ steering pirate@0.4 L16 · 12 of 18 good in chat · /steer off"]);
	assert.equal(statuses.at(-1), "metalscope: on");                                            // no steer tag in the footer
	assert.equal(ui.steerStatus(), " · steer pirate@0.4 L16");                                 // /scope status keeps it
	// Another steer: the same line, not set again (it keeps its place among the widgets), drawn with the new one.
	const set = widgets.get("metalscope-steer");
	live.steer = steerState([{ name: "calm", model: "m", layer: 20, strength: 0.3, source: "profile" }], "m");
	ui.updateUi(ctx);
	assert.equal(widgets.get("metalscope-steer"), set);
	assert.deepEqual(line.render(99), [" ◆ steering calm@0.3 L20 · /steer off"]);
	// The job widget appears: the line is set again (Pi puts a widget set again last, nearest the editor); no steer, nothing.
	ui.steerLast(ctx);
	assert.notEqual(widgets.get("metalscope-steer"), set);
	// Pi draws it from a timer, nothing around it: a steer it cannot draw is no line, never a throw.
	live.steer = { model: "m", parts: null };
	assert.deepEqual(line.render(99), []);
	live.steer = undefined;
	assert.deepEqual(line.render(99), []);                                                      // drawn from live.steer each frame
	ui.updateUi(ctx);
	assert.ok(!widgets.has("metalscope-steer"));
	ui.steerLast(ctx);
	assert.ok(!widgets.has("metalscope-steer"));
});

test("widget across 100 columns: under it the dock hidden (said on the status line), nothing extra asked, its room given back, no stepping in and no /scope dock mode (said why); from 100 all back", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-"));
	const root = { render: () => ["pi"], invalidate() {} };
	const tui: any = { layoutRoot: root, terminal: { rows: 60, columns: 120 }, setLayoutRoot(r: any) { this.layoutRoot = r; }, requestRender() {} };
	let dock: any, widget: any, status = "";
	const notes: string[] = [];
	const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setStatus: (_: string, s: string) => (status = s), notify: (m: string) => notes.push(m),
		setWidget: (k: string, f: any) => k === "metalscope" && (widget = f?.(tui)), custom: (make: any) => new Promise<void>(() => (dock = make(tui, {}, {}, () => {}))) } };
	const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false } as any;
	const ui = createUi({ live } as any, join(dir, "pi.json"), () => {});
	const at = async (cols: number) => {
		tui.terminal.columns = cols;
		widget.render(cols);
		await new Promise((r) => setTimeout(r, 0));
		return { status, asked: [!!live.experts, !!live.lens, !!live.writes], gap: tui.layoutRoot === root ? 0 : tui.layoutRoot.entries[1].basis };
	};
	ui.showDock(ctx);
	assert.deepEqual(await at(120), { status: "metalscope: on · docked", asked: [true, true, true], gap: dockColumns(120) });
	for (const cols of [90, 99]) {
		assert.deepEqual(await at(cols), { status: `metalscope: on · dock hidden (needs ${DOCK_MIN_COLS} columns)`, asked: [false, false, false], gap: 0 }, `${cols}`);
		ui.showDock(ctx, true);                                                        // alt+i
		assert.equal(notes.at(-1), `dock needs 100 columns, has ${cols}`);
		assert.equal(dock.stepped, false, "not stepped in: Pi would give a hidden dock no keys");
		assert.deepEqual([ui.dockMode("mini", ctx), ui.dockMode("full", ctx)], [`dock needs 100 columns, has ${cols}`, `dock needs 100 columns, has ${cols}`]);
		assert.equal(readPrefs(join(dir, "pi.json")).mini, undefined, "the mode unchanged");
	}
	assert.deepEqual(await at(100), { status: "metalscope: on · docked", asked: [true, true, true], gap: dockColumns(100) });
	await at(120);
	assert.equal(dock.stepped, false, "widened: still not stepped in");
	ui.showDock(ctx, true);
	assert.equal(dock.stepped, true, "now it shows: stepped in");
	assert.equal(ui.dockMode("mini", ctx), undefined);
});

test("widget, a session's first dock under 100 columns (no TUI seen yet: Pi's stdout's columns): alt+i and /scope dock mini | full refused, said why, nothing changed", async () => {
	const dir = mkdtempSync(join(tmpdir(), "ms-prefs-")), path = join(dir, "pi.json");
	const was = Object.getOwnPropertyDescriptor(process.stdout, "columns");
	Object.defineProperty(process.stdout, "columns", { value: 90, configurable: true });
	try {
		let made = 0;
		const notes: string[] = [];
		const ctx = { hasUI: true, model: { provider: "metalscope-omlx" }, ui: { setStatus() {}, setWidget() {}, notify: (m: string) => notes.push(m), custom: () => (made++, new Promise<void>(() => {})) } };
		const live = { base: "http://o:1", clientId: "pi-x", level: "on", connected: false } as any;
		const ui = createUi({ live } as any, path, () => {});
		ui.showDock(ctx, true);                                                        // alt+i
		assert.deepEqual(notes, ["dock needs 100 columns, has 90"]);
		assert.deepEqual([ui.dockMode("mini", ctx), ui.dockMode("full", ctx)], ["dock needs 100 columns, has 90", "dock needs 100 columns, has 90"]);
		assert.deepEqual([made, ui.docked(), readPrefs(path), !!live.experts, !!live.lens, !!live.writes], [0, false, {}, false, false, false], "no dock made, nothing remembered or asked for");
	} finally {
		if (was) Object.defineProperty(process.stdout, "columns", was);
		else delete (process.stdout as any).columns;
	}
});

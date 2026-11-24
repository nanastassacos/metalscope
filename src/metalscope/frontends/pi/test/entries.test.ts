import assert from "node:assert/strict";
import { test } from "node:test";
import { visibleWidth } from "@earendil-works/pi-tui";
import { chatRank } from "../concept.ts";
import { registerEntries } from "../entries.ts";
import { type LearnState, jobEntryLines, trimJob } from "../jobs.ts";
import { CAL, chatConcept } from "./fixtures/concepts.ts";

/** The renderers registerEntries gives Pi, by entry type; Pi's theme as codes a test can find (a background among them). */
const renderers = new Map<string, any>();
registerEntries({ registerEntryRenderer: (type: string, fn: any) => renderers.set(type, fn) } as any);
const TH = { fg: (c: string, s: string) => `\x1b[3${{ success: 2, accent: 6, dim: 7, error: 1, warning: 3, muted: 5 }[c] ?? 9}m${s}\x1b[39m`,
	bold: (s: string) => `\x1b[1m${s}\x1b[22m`, bg: (_c: string, s: string) => `\x1b[44m${s}\x1b[49m` };
const plain = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
const drawn = (type: string, data: any, width: number): string[] => renderers.get(type)({ data }, {}, TH).render(width);

const JOB = { kind: "calibrate", job: "c-1", name: "calm", phase: "done", outcome: "done", origin: "pi", started: 0, updated: 1870,
	steps: [{ label: "calibrate", status: "done", summary: "16/18 good @0.4", started: 0, ended: 1870 }], result: { calibration: CAL } } as LearnState;

test("every entry is drawn on the terminal's own background, one column in, never wider than the width", () => {
	const cases: [string, any][] = [
		["scope-guide", { v: 1, title: "concept: help", lines: ["## /concept: the library", "| /concept NAME\tone concept", "· a note that runs on ".repeat(8)] }],
		["scope-job", { v: 1, job: trimJob(JOB), recent: ["last 2 jobs, newest first", "calibrate calm: 16/18 good @0.4 · 2026-11-23 02:15", "learn calm: failed"] }],
		["scope-concept", { v: 1, concept: chatConcept(), rank: { rank: 1, of: 1 }, backend: "omlx" }],
		["scope-replay", { v: 1, prompt: "pancakes?", replies: [["your reply", "Flour, eggs, milk."], ["unsteered", "Whisk and fry."]], temperature: 0 }],
	];
	for (const [type, data] of cases) for (let width = 1; width <= 120; width++) {
		const ls = drawn(type, data, width);
		assert.ok(ls.every((l) => visibleWidth(l) <= width), `${type} ${width}\n${ls.join("\n")}`);
		assert.ok(ls.every((l) => !l.includes("\x1b[44m")), `${type}: no background`);
		if (width >= 20) assert.ok(ls.every((l) => l === "" || l.startsWith(" ")), `${type} ${width}`);
	}
	assert.equal(renderers.get("scope-job")({ data: { v: 2 } }, {}, TH), undefined);   // another version: not drawn
});

test("a job entry: the job's lines one column in, then /scope jobs' list under it, dim", () => {
	const lines = drawn("scope-job", { v: 1, job: trimJob(JOB), recent: ["last 2 jobs, newest first", "calibrate calm: 16/18 good @0.4", "learn calm: failed"] }, 80);
	assert.deepEqual(lines, [...jobEntryLines(JOB, { width: 79, th: TH }).map((l) => ` ${l}`),
		" \x1b[37mlast 2 jobs, newest first\x1b[39m", " \x1b[37m  calibrate calm: 16/18 good @0.4\x1b[39m", " \x1b[37m  learn calm: failed\x1b[39m"]);
	assert.deepEqual(drawn("scope-job", { v: 1, job: trimJob(JOB) }, 90).map(plain), [" ✓ calibrated calm · 16/18 good @0.4 · 31m 10s",
		"   ✓ calibrate   0.2 2/18 · 0.3 11/18 · 0.4 16/18 · 0.6 15/18 · 0.8 4/18 · 31m 10s", "   /steer calm uses dose 0.4"]);
});

test("a concept entry: /concept NAME's lines one column in", () => {
	const c = chatConcept();
	assert.deepEqual(drawn("scope-concept", { v: 1, concept: c, rank: chatRank([c], c.name), backend: "omlx" }, 100).map(plain), [
		" fluffy · 16/18 good @0.4 · ranked 1 of 1", "   what        no description", "   texts       none on this model",
		"   in chat     0.2 2/18 · 0.3 11/18 · 0.4 16/18 · 0.6 15/18 · 0.8 4/18", "   learn       L20 · +0.40 log-odds over control at dose 0.8",
		"   /steer fluffy uses dose 0.4 · /concept calibrate fluffy (about 20 min to 2 h)"]);
});

test("help and replay entries: the title bold, no ◆, the body as before", () => {
	const help = drawn("scope-guide", { v: 1, title: "concept: help", lines: ["## /concept: the library", "| /concept NAME\tone concept"] }, 80);
	assert.equal(help[0], " \x1b[1mconcept: help\x1b[22m");
	assert.deepEqual(help.slice(1).map(plain), [" ", " /concept: the library", "   /concept NAME  one concept"]);
	const replay = drawn("scope-replay", { v: 1, prompt: "pancakes?\nplease", replies: [["unsteered", "Whisk and fry."]], temperature: 0 }, 80);
	assert.deepEqual(replay.map(plain), [" replay · temp 0 (greedy) · pancakes? please", " ", " unsteered", "   Whisk and fry."]);
	assert.match(replay[0], /^ \x1b\[1mreplay\x1b\[22m\x1b\[37m · temp 0/);
	assert.ok(!help.concat(replay).some((l) => l.includes("◆")));
});

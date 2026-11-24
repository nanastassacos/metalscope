/**
 * The transcript entries the extension leaves (session entries, never sent to the model), each drawn on the terminal's
 * own background one column in, in the look of the job widget above the editor (jobs.ts):
 * - "scope-guide": the guide and a command's help (/scope help, /steer help).
 * - "scope-job": a finished job's result, its steps frozen (jobs.ts jobEntryLines), when it finishes and in /scope jobs
 *   (with the last jobs under it).
 * - "scope-concept": one concept (/concept NAME: concept.ts overviewLines).
 * - "scope-replay": /steer replay's pair, your reply and the same message answered unsteered or under another steer.
 * Each is drawn at render time from its data, with the theme and the width of the moment.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { visibleWidth, wrapTextWithAnsi } from "@earendil-works/pi-tui";
import { type ConceptInfo, type LearnBackend, overviewLines } from "./concept.ts";
import { COMMAND, bold, lineOf, oneIn } from "./draw.ts";
import { type LearnState, jobEntryLines } from "./jobs.ts";

// Word wrap for the guide and replay entries (never narrower than 20 columns; what overflows is cut: margin).
const wrap = (s: string, w: number): string[] => (s ? wrapTextWithAnsi(s, Math.max(20, w)) : []);

/** An entry's lines one column in, as the widget's are (oneIn). */
const margin = (render: (width: number) => string[]) => ({ render: (width: number) => oneIn(width, render), invalidate() {} });

/** One body line of the guide or a command's help: a line that starts with a command form ("/steer fluffy 0.5: the
 * same at another dose") shows the form in the command colours (COMMAND), so the forms stand out from what they do. */
export function helpLine(theme: any, line: string): string {
	const m = line.match(/^(\/[^:]{1,80}?):\s(.*)$/);
	return m ? theme.fg(COMMAND.form, m[1]) + theme.fg(COMMAND.words, ": " + m[2]) : theme.fg(COMMAND.words, line);
}

/** The entries' renderers, registered with Pi. A finished job's entry (`job`, trimmed: jobs.ts trimJob) has /scope jobs'
 * list under it (`recent`: its head line, then a job a line), dim. */
export function registerEntries(pi: ExtensionAPI) {
	pi.registerEntryRenderer<{ v: 1; lines: string[]; title: string }>("scope-guide", (entry, _opts, theme) => {
		const g = entry.data;
		if (!g || g.v !== 1) return undefined;
		return margin((width) => [bold(theme, g.title), ...helpBody(theme, g.lines, width)]);
	});
	pi.registerEntryRenderer<{ v: 1; job: LearnState; recent?: string[] }>("scope-job", (entry, _opts, theme) => {
		const g = entry.data;
		if (!g || g.v !== 1 || !g.job) return undefined;
		return margin((width) => {
			const [head, ...jobs] = g.recent ?? [];
			const list = head ? [head, ...jobs.map((l) => `  ${l}`)].map((t) => lineOf([{ t, c: "dim" }], width, theme)) : [];
			return [...jobEntryLines(g.job, { width, th: theme }), ...list];
		});
	});
	pi.registerEntryRenderer<{ v: 1; concept: ConceptInfo; rank: { rank: number; of: number } | null; backend: LearnBackend }>("scope-concept", (entry, _opts, theme) => {
		const g = entry.data;
		if (!g || g.v !== 1 || !g.concept) return undefined;
		return margin((width) => overviewLines(g.concept, g.rank, g.backend, width, theme));
	});
	// /steer replay: your reply and the same message answered again, one under another, outside the transcript the model sees.
	pi.registerEntryRenderer<{ v: 1; prompt: string; replies: [string, string][]; temperature?: number }>("scope-replay", (entry, _opts, theme) => {
		const g = entry.data;
		if (!g || g.v !== 1) return undefined;
		return margin((width) => {
			const temp = g.temperature === undefined ? "" : g.temperature === 0 ? " · temp 0 (greedy)" : ` · temp ${g.temperature}`;
			const out: string[] = [lineOf([{ t: "replay", c: "bold" }, { t: `${temp} · ${g.prompt.replace(/\s+/g, " ")}`, c: "dim" }], width, theme)];
			for (const [label, text] of g.replies) {
				out.push("", theme.fg("warning", label));
				for (const para of text.trim().split("\n")) out.push(...wrap(para, width - 4).map((x) => "  " + x));
			}
			return out;
		});
	});
}

const FORM_MAX = 32; // a form column wider than this (or than 45% of the width) puts that row's text on the next line

/** The lines of a guide entry (help.ts): "## " a heading; "| form\twhat" rows, each run of them a table with the forms
 * in one column; "· " a note; anything else a head line with its "\n"-separated body lines indented under it. */
export function helpBody(theme: any, lines: string[], width: number): string[] {
	const out: string[] = [];
	for (let i = 0; i < lines.length; i++) {
		const l = lines[i];
		if (l.startsWith("## ")) out.push("", theme.fg("accent", bold(theme, l.slice(3))));
		else if (l.startsWith("| ")) {
			let j = i;
			while (j < lines.length && lines[j].startsWith("| ")) j++;
			const rows = lines.slice(i, j).map((r) => { const [form, what = ""] = r.slice(2).split("\t"); return { form, what }; });
			const cap = Math.min(FORM_MAX, Math.floor(0.45 * width));
			const col = Math.min(cap, Math.max(...rows.map((r) => visibleWidth(r.form))));
			for (const r of rows) {
				const fw = visibleWidth(r.form), text = wrap(r.what, width - 6 - col - 2);
				if (fw > col) {
					out.push("  " + theme.fg("accent", r.form));
					for (const x of text) out.push(" ".repeat(col + 4) + theme.fg("muted", x));
				} else {
					text.forEach((x, k) => out.push("  " + (k ? " ".repeat(col) : theme.fg("accent", r.form) + " ".repeat(col - fw)) + "  " + theme.fg("muted", x)));
					if (!text.length) out.push("  " + theme.fg("accent", r.form));
				}
			}
			i = j - 1;
		} else if (l.startsWith("· ")) out.push(...wrap(l.slice(2), width - 6).map((x, k) => (k ? "    " : "  · ") + theme.fg("muted", x)));
		else {
			const [head, ...rest] = l.split("\n");
			out.push(theme.fg("warning", head));
			for (const r of rest) out.push(...wrap(helpLine(theme, r.trim()), width - 6).map((x) => "    " + x));
		}
	}
	return out;
}

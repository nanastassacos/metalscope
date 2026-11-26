import assert from "node:assert/strict";
import { test } from "node:test";
import { CONCEPT_SUBCOMMANDS } from "../concept.ts";
import { EXPLAIN } from "../draw.ts";
import { helpBody, helpLine } from "../entries.ts";
import { COMMANDS, HELP_STREAMS, TOPICS, streamHelp, commandHelpLines, describe, guideLines, helpTopics, overviewHelp } from "../help.ts";
import { SCOPE_SUBCOMMANDS } from "../scope.ts";

const plain = { fg: (_: string, s: string) => s, bold: (s: string) => s };
const drawn = (lines: string[], width = 90) => helpBody(plain, lines, width);

test("/scope help: one screen: start, the dock's keys, steer, learn a concept, the commands and the topics", () => {
	const out = drawn(overviewHelp());
	assert.ok(out.length <= 36, `${out.length} lines`);
	const heads = ["Start", "The dock, stepped in", "Steer", "Learn a concept", "Commands"];
	assert.deepEqual(heads.map((h) => out.indexOf(h)), [...heads.map((h) => out.indexOf(h))].sort((a, b) => a - b), "in this order");
	for (const h of heads) assert.ok(out.includes(h), h);
	for (const c of ["/scope", "/steer", "/concept", "/jlens"]) assert.ok(out.some((l) => l.trimStart().startsWith(c + " ")), c);
	// The path a newcomer takes: pick a model, the dock; then steer (in the dock, by name), then learn.
	const shown = out.join("\n");
	const path = [/\/model +pick a model marked \(metalscope\)/, /ctrl\+\\ +the dock; again: step in; again: close/,
		/6 in the dock +the Steer page: ↑\/↓ a concept · ←\/→ a dose · enter steers · x off/, /\/steer joyful +the same by name · \/steer off/,
		/\/steer replay +your last reply beside the same message with no steering/, /\/concept learn joyful +learn one from the library: 25 min–2\.5 h on oMLX/];
	for (const re of path) assert.match(shown, re);
	const at = path.map((re) => shown.search(re));
	assert.deepEqual(at, [...at].sort((a, b) => a - b), "in the order a newcomer takes them");
	assert.doesNotMatch(shown, /\/scope on|fluffy|captions|tab explain|full view/);
	// The dock's keys, each said once.
	for (const k of ["←/→ · u/U", "tab · 1-5 · ?", "6 · m", "s ", "esc "]) assert.ok(out.some((l) => l.trimStart().startsWith(k)), k);
	assert.doesNotMatch(shown, /\bp\b.*page/);
	assert.match(shown, /\/<command> help +every form of one \(\/scope's: \/scope help scope\)/);
	// The served model's own learn time when it was measured; the range for any other.
	const qwen = { backend: "omlx", model: "Qwen3.6-35B-A3B-MLX-4bit" } as const;
	assert.match(overviewHelp(qwen).join("\n"), /\/concept learn joyful\tlearn one from the library: 26 min \(measured on Qwen3\.6\)/);
	assert.match(guideLines("", { backend: "ds4", model: "Qwen3.8-Flash-Next-ds4-q4k" }).join("\n"), /from the library: 4–7\.5 h on ds4/);
	const forms = commandHelpLines("concept", undefined, { backend: "ds4", model: "DeepSeek-V4.1-Flash-ds4-q2" }).join("\n");
	assert.match(forms, /\/concept learn NAME <meaning>\tmake a concept and calibrate it: 4–7\.5 h \(measured on DeepSeek V4\.1\)/);
	assert.match(forms, /\/concept calibrate NAME\tfind its chat dose in graded chats: 3–6 h \(measured on DeepSeek V4\.1\)/);
	assert.match(commandHelpLines("concept").join("\n"), /make a concept and calibrate it: 25 min–2\.5 h on oMLX, 4–7\.5 h on ds4/);   // no model: both ranges
	// The concepts topic: each backend's range, then its measured runs, from the same table.
	const concepts = guideLines("concepts").join("\n");
	assert.match(concepts, /chat working\. Measured: Qwen3\.6 5 min, then 20 min \(26 min in all\)\./);
	assert.match(concepts, /chat working\. Measured: DeepSeek V4\.1 60 to 90 min, then 3 to 6 h \(4 to 7\.5 h in all\)\./);
	// No web page anywhere in the help.
	const all = [...overviewHelp(), ...TOPICS.flatMap((t) => guideLines(t.name)), ...COMMANDS.flatMap((c) => commandHelpLines(c.cmd))].join("\n");
	assert.doesNotMatch(all, /web page|browser|\/scope open|Layer by layer|Experts tab|Steering tab/);
	if (process.env.SHOW_REPLY) console.log(out.join("\n"));
	for (const t of TOPICS) assert.ok(out.join(" ").includes(t.name), t.name);
	assert.deepEqual(guideLines(), overviewHelp());
	// /scope's own forms by its name; a word a topic starts with stays the topic.
	assert.deepEqual(guideLines("scope"), commandHelpLines("scope"));
	assert.deepEqual(guideLines("jlens"), commandHelpLines("jlens"));
	assert.equal(guideLines("steer")[0], "## Steering");
	assert.deepEqual(helpTopics(), [...TOPICS.map((t) => t.name), "scope", "jlens"]);
	assert.ok(!out.join("\n").includes("/lens"));                      // the J-lens command is /jlens
});

test("each command's help lists its forms, short; every subcommand has a form", () => {
	for (const c of COMMANDS) {
		const lines = commandHelpLines(c.cmd);
		assert.equal(lines[0], `## /${c.cmd}: ${c.what}`);
		assert.ok(c.what.length <= 60, c.what);
		assert.ok(describe(c.cmd).length <= 80, describe(c.cmd));
		for (const [form, what] of c.forms) {
			assert.ok(form.startsWith(`/${c.cmd}`), form);
			assert.ok(what.length <= 70, `${form}: ${what}`);
		}
		assert.ok(c.notes.length <= 2, c.cmd);
	}
	const forms = (cmd: any) => commandHelpLines(cmd).join("\n");
	for (const s of SCOPE_SUBCOMMANDS) assert.ok(forms("scope").includes(`/scope ${s}`), s);
	for (const s of CONCEPT_SUBCOMMANDS.filter((s) => s !== "help")) assert.ok(forms("concept").includes(`/concept ${s}`), s);
	for (const s of ["status", "on", "off", "use", "learn"]) assert.ok(forms("jlens").includes(`/jlens ${s}`), s);
	for (const s of ["replay", "off", "status"]) assert.ok(forms("steer").includes(`/steer ${s}`), s);
	// One subcommand's forms alone, with where the rest is.
	const cal = commandHelpLines("concept", "calibrate");
	assert.equal(cal[0], "## /concept calibrate");
	assert.ok(cal.some((l) => l.includes("/concept calibrate NAME\t")) && !cal.some((l) => l.includes("/concept learn")));
	assert.match(cal[cal.length - 1], /\/concept help/);
});

test("/scope help <topic>: by name or its start; else what mentions the word", () => {
	assert.equal(guideLines("tracing")[0], "## Tracing");
	const tracing = guideLines("tracing").join("\n");
	assert.match(tracing, /^off \(\/scope off\)/m);
	assert.match(tracing, /a traced reply writes at 47% of untraced speed on oMLX and 58% on ds4 at medium \(the default lens size\), 77% and 80% at small/);
	assert.match(tracing, /92% on both with the lenses off/);
	assert.doesNotMatch(drawn(overviewHelp()).join("\n") + commandHelpLines("scope").join("\n") + TOPICS.map((t) => guideLines(t.name).join("\n")).join("\n"),
		/\/scope on lens|lens level/);                                    // one trace level everywhere
	assert.equal(guideLines("steer")[0], "## Steering");                  // the start of a topic's name
	assert.doesNotMatch(drawn(overviewHelp()).join("\n") + TOPICS.map((t) => guideLines(t.name).join("\n")).join("\n"), /meter/i);
	const replay = guideLines("replay").join("\n");                         // no topic: the forms and items naming it
	assert.match(replay, /## \/steer/);
	assert.match(replay, /\/steer replay temp:0/);
	assert.match(replay, /## Steering/);
	assert.match(guideLines("zzz")[0], /Nothing in the help mentions "zzz"/);
});

test("the measured caveats stay, a line each", () => {
	assert.match(guideLines("lens").join("\n"), /The logit lens needs nothing|Every model has it/);
	assert.match(guideLines("jlens").join("\n"), /\/jlens off/);
	assert.match(commandHelpLines("steer").join("\n"), /In the dock: 6, the Steer page/);
});

test("the help entry draws a run of rows as a table, forms in one column", () => {
	const out = drawn(["| /a\tone", "| /bbbb\ttwo", "· a note"], 60);
	assert.deepEqual(out, ["  /a     one", "  /bbbb  two", "  · a note"]);
	// A row's text wraps under its column; a form wider than the column puts the text on the next line.
	const wrapped = drawn(["| /x\t" + "word ".repeat(20).trim()], 40);
	assert.ok(wrapped.length > 1 && wrapped.slice(1).every((l) => l.startsWith("      ")), wrapped.join("|"));
	const wide = drawn(["| /a\tone", "| /" + "b".repeat(40) + "\ttwo"], 60);
	assert.equal(wide[1].trim(), "/" + "b".repeat(40));
	assert.equal(wide[2].trim(), "two");
});

test("helpLine: a command form stands out from what it does", () => {
	const theme = { fg: (c: string, s: string) => `<${c}>${s}</${c}>` };
	assert.equal(helpLine(theme, "/steer fluffy 0.5: the same at another dose"), "<accent>/steer fluffy 0.5</accent><muted>: the same at another dose</muted>");
	assert.equal(helpLine(theme, "Steering is constant."), "<muted>Steering is constant.</muted>");
});

test("/scope help dock: the board, the one-panel board, the mini dock, steering from it; each part as ? explains it", () => {
	const dock = guideLines("dock").join("\n");
	for (const head of ["opening it", "moving", "the board", "the one-panel board", "the mini dock", "steering"]) assert.match(dock, new RegExp(`\n${head}\n`), head);
	assert.match(dock, /down to 44 columns and 23 lines/);
	assert.ok(!/the pages|Words page|\bp\b/.test(dock), "no pages");
	// The board's panels as its ? says them.
	for (const [head, key] of [["LENS", "SETTLING"], ["MAP", "MAP"], ["TOKENS", "TOKENS"], ["STEER", "STEER"], ["test chats", "STEER_EXAMPLES"]])
		assert.ok(dock.includes(`\n${head}\n    ${EXPLAIN[key]}`), head);
	assert.ok(dock.includes(`\nEXPERTS\n    ${EXPLAIN.EXPERTS}`));
	assert.ok(dock.includes(`\nLENS\n    ${EXPLAIN.SETTLING}\n    ${EXPLAIN.TELESCOPE}`), "SETTLING in LENS's entry, first, not one of its own");
	assert.ok(!/\nSETTLING\n/.test(dock));
	assert.ok(dock.includes(`\nREPLY\n    ${EXPLAIN.REPLY}`));
	assert.deepEqual(HELP_STREAMS, [[4, 0], [1, 0], [1, 5]], "four streams, one, then one with reads");
	assert.ok(dock.includes(`\nSTREAM\n    ${HELP_STREAMS.flatMap(([k, rk]) => streamHelp(k, rk)).join("\n    ")}`), "STREAM: each in order");
	// "100 columns" says whose: the terminal's, or the dock's (a terminal 181 wide).
	assert.match(dock, /in a terminal 100 columns wide/);
	assert.match(dock, /A dock 100 columns wide and 36 lines tall \(a terminal 181 wide\)/);
	assert.doesNotMatch(dock, /captions|tab explain|full view|swap lens|—/);
	assert.doesNotMatch(EXPLAIN.REPLY, /nderline/);
});

test("the dock's ? texts: short, no em-dashes, the wording the screen uses", () => {
	for (const [k, v] of Object.entries(EXPLAIN)) {
		assert.ok(v.length <= 200, `${k}: ${v.length}`);
		assert.doesNotMatch(v, /—|plain|unsteered|\bkeys?\b.*←/, k);
	}
});

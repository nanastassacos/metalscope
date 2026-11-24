import assert from "node:assert/strict";
import { test } from "node:test";
import { SCOPE_SUBCOMMANDS, parseScopeArgs, scopeStatusText } from "../scope.ts";

test("parseScopeArgs: every subcommand, the /concept words, errors", () => {
	assert.deepEqual(parseScopeArgs(""), { cmd: "status" });
	assert.deepEqual(parseScopeArgs("status"), { cmd: "status" });
	assert.deepEqual(parseScopeArgs("on"), { cmd: "on" });
	assert.deepEqual(parseScopeArgs("on lens"), { cmd: "error", error: "/scope on takes no arguments · see /scope help scope" });
	assert.deepEqual(parseScopeArgs("on full"), { cmd: "error", error: "/scope on takes no arguments · see /scope help scope" });
	assert.deepEqual(parseScopeArgs("off"), { cmd: "off" });
	assert.deepEqual(parseScopeArgs("dock"), { cmd: "dock" });
	assert.deepEqual(parseScopeArgs("dock mini"), { cmd: "dock", mode: "mini" });
	assert.deepEqual(parseScopeArgs("dock Full"), { cmd: "dock", mode: "full" });
	assert.deepEqual(parseScopeArgs("dock tiny"), { cmd: "error", error: '/scope dock takes nothing, mini or full, not "tiny" · see /scope help scope' });
	assert.deepEqual(parseScopeArgs("dock mini full"), { cmd: "error", error: '/scope dock takes nothing, mini or full, not "mini full" · see /scope help scope' });
	assert.deepEqual(parseScopeArgs("open"), { cmd: "error", error: 'unknown subcommand "open" · see /scope help scope' });
	assert.deepEqual(parseScopeArgs("ON"), { cmd: "on" });                                        // subcommands in any case
	assert.deepEqual(parseScopeArgs("help"), { cmd: "help", topic: "" });
	assert.deepEqual(parseScopeArgs("help unsaid words"), { cmd: "help", topic: "unsaid words" });
	assert.deepEqual(parseScopeArgs("example scaly"), { cmd: "error", error: 'unknown subcommand "example" · see /scope help scope' });
	assert.deepEqual(parseScopeArgs("fluffy"), { cmd: "error", error: 'unknown subcommand "fluffy" · see /scope help scope' });
	// The library's own words go to /concept as they are, never as a concept's page.
	for (const bad of ["on loud", "on lens lens", "on light full", "dock now", "open a b", "fluffy page", "fl/uffy", "inspect", "cancel", "trace"]) {
		const r = parseScopeArgs(bad);
		assert.equal(r.cmd, "error", bad);
		if (r.cmd === "error") assert.match(r.error, /see \/scope help scope$/);
	}
	// Completions offer every subcommand.
	assert.deepEqual(parseScopeArgs("jobs"), { cmd: "jobs" });
	assert.deepEqual(parseScopeArgs("jobs cancel"), { cmd: "cancel" });
	assert.deepEqual(parseScopeArgs("jobs all"), { cmd: "error", error: '/scope jobs takes nothing or cancel, not "all" · see /scope help scope' });
	assert.deepEqual(parseScopeArgs("jobs cancel now"), { cmd: "error", error: '/scope jobs takes nothing or cancel, not "cancel now" · see /scope help scope' });
	assert.deepEqual(SCOPE_SUBCOMMANDS, ["status", "on", "off", "dock", "dock mini", "dock full", "jobs", "jobs cancel", "help"]);
});

test("scopeStatusText: the trace level and connection at once, then the server's concepts and steering", () => {
	const head = { level: "on", metalscope: true, steer: "" };
	assert.equal(scopeStatusText(head), "metalscope: on");
	assert.equal(scopeStatusText({ ...head, unreachable: "metalscope server not reachable at http://x", metalscope: false, steer: " · steer angry@0.4 L24" }),
		"metalscope: on · metalscope server not reachable at http://x (switch to a metalscope model with /model to trace) · steer angry@0.4 L24");
	const concepts = [{ name: "angry" }, { name: "calm" }] as any;
	assert.equal(scopeStatusText(head, { model: "m", steer_enabled: false, concepts }),
		"metalscope: on\nm: learned angry, calm · steering on server: off");
	assert.equal(scopeStatusText(head, { model: "m", steer_enabled: true, concepts: [] }).split("\n")[1],
		"m: learned nothing yet · steering on server: on");
});

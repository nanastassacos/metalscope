/**
 * /scope's pure logic: parsing its arguments, and what /scope status says. /scope looks at what the model does:
 * tracing, the dock, the GPU jobs and the help (commands.ts registers it). No Pi or network code here,
 * so it is testable.
 */

import { offer } from "./args.ts";
import type { ConceptList } from "./concept.ts";

/** Where a /scope usage error points: its forms (/scope help alone is the overview). */
export const SEE_HELP = "see /scope help scope";

/** `/scope` subcommands, in the order completions offer them. */
export const SCOPE_SUBCOMMANDS = ["status", "on", "off", "dock", "dock mini", "dock full", "jobs", "jobs cancel", "help"];

export type ScopeArgs =
	| { cmd: "status" | "on" | "off" | "dock" | "jobs" | "cancel" }
	| { cmd: "dock"; mode: "mini" | "full" }
	| { cmd: "help"; topic: string }
	| { cmd: "error"; error: string };

/** `/scope`: tracing, the dock, the GPU jobs (`jobs cancel` stops the running one) and the guide;
 * alone = status. */
export function parseScopeArgs(args: string): ScopeArgs {
	const [raw, ...rest] = (args || "").trim().split(/\s+/).filter(Boolean);
	const cmd = raw?.toLowerCase();
	const err = (msg: string): ScopeArgs => ({ cmd: "error", error: `${msg} · ${SEE_HELP}` });
	const none = (c: "status" | "on" | "off" | "dock"): ScopeArgs => (rest.length ? err(`/scope ${c} takes no arguments`) : { cmd: c });
	switch (cmd) {
		case undefined:
		case "status":
		case "on":
		case "off":
			return none(cmd ?? "status");
		case "dock":
			if (!rest.length) return { cmd: "dock" };
			if (rest.length > 1 || !["mini", "full"].includes(rest[0].toLowerCase())) return err(`/scope dock takes nothing, mini or full, not "${rest.join(" ")}"`);
			return { cmd: "dock", mode: rest[0].toLowerCase() as "mini" | "full" };
		case "jobs":
			if (rest.length > 1 || (rest[0] && rest[0] !== "cancel")) return err(`/scope jobs takes nothing or cancel, not "${rest.join(" ")}"`);
			return { cmd: rest[0] === "cancel" ? "cancel" : "jobs" };
		case "help":
			return { cmd: "help", topic: rest.join(" ") };
		default:
			return err(`unknown subcommand "${raw}"`);
	}
}

/** /scope's completions: its subcommands; after `help `, the help's topics and commands (`topics`). */
export function scopeCompletions(prefix: string, topics: string[]) {
	const m = prefix.match(/^help\s+(\S*)$/i);
	return m ? offer(m[1], topics, "help ") : offer(prefix, SCOPE_SUBCOMMANDS);
}

/** Where /scope status starts: the trace level, `unreachable` (provider.ts's words for a server that does not answer;
 * undefined when it does), whether the model is a metalscope one (it can be traced) and `steer` (" · steer …", or
 * ""). */
export interface ScopeHead {
	level: string;
	unreachable?: string;
	metalscope: boolean;
	steer: string;
}

/** What /scope status says: trace level and connection at once, then, when the server answered with its concept list,
 * the concepts learned on the model and whether the server allows steering. */
export function scopeStatusText(h: ScopeHead, list?: ConceptList): string {
	const where = h.metalscope ? "" : " (switch to a metalscope model with /model to trace)";
	const head = `metalscope: ${h.level}${h.unreachable ? ` · ${h.unreachable}` : ""}${where}${h.steer}`;
	if (!list) return head;
	return `${head}\n${list.model}: learned ${list.concepts.map((c) => c.name).join(", ") || "nothing yet"} · ` +
		`steering on server: ${list.steer_enabled ? "on" : "off"}`;
}

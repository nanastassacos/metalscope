/** Concept profiles and chat calibrations as the server sends them, shared by the concept and steer tests. */

import { type ConceptInfo, parseConcepts } from "../../concept.ts";

/** One /concept list row as steerListLine draws it: name, good in chat, dose. */
export const listRow = (name: string, good: string, dose: string, w = 12) => `${name.padEnd(w)}  ${good.padEnd(14)}  ${dose}`;

export const info: ConceptInfo = {
	name: "fluffy", title: "Fluffy", description: "soft things", model: "m",
	reader: { layer: 20, strength: 0.8, concept: 0.84, control: 0.14 },
	generation: { edge: 1.6 },
	steer: { layer: 20, strength: 0.8, basis: "chat" },
};

/** A concept with a reader at L8 and a ladder whose edge is its reader's dose, whose default dose is `steer` (null:
 * not calibrated). */
export const terseJson = (steer: any, gen: any = { edge: 0.8 }) =>
	({ concepts: [{ name: "terse", title: "Terse", reader: { layer: 8, strength: 0.8, concept: 0.94, control: -0.48 }, generation: gen, steer }] });

/** One dose of a calibration as the server stores it. */
const strength = (dose: number, good: number, attempts = 18, failed = { tool: 0, thinking: 0, short: 0 }, coherence: number | null = 2.9) =>
	({ dose, good, turns: 18, attempts, failed, coherence });

// A chat calibration from the server (learn/calibrate.py, version 4): the most good replies at 0.4 (16 of 18).
export const CAL = { version: 4, concept: "fluffy", layer: 20, default_dose: 0.4, good: 16, turns: 18,
	grader: { model: "m", thinking: false }, plain: strength(0, 0),
	strengths: [strength(0.2, 2), strength(0.3, 11), strength(0.4, 16, 19, { tool: 1, thinking: 0, short: 0 }), strength(0.6, 15),
		strength(0.8, 4, 40, { tool: 2, thinking: 20, short: 0 }, 1.6)],
	passes: [[0.2, 0.4, 0.6, 0.8], [0.3]], reason: "16/18 good @0.4" };
// Few good replies: at most 8 of 18, at 0.2 (Qwen robotic; 0.3 ties it, the lower wins).
export const LOW = { ...CAL, default_dose: 0.2, good: 8,
	strengths: [strength(0.1, 3), strength(0.2, 8, 22, { tool: 3, thinking: 1, short: 0 }), strength(0.3, 8), strength(0.4, 5, 43, { tool: 3, thinking: 33, short: 0 }, 1.2)],
	passes: [[0.2, 0.4], [0.1, 0.3]], reason: "8/18 good @0.2" };
/** A concept steered at `layer` (its reader) with `calibration` (made at L20 for CAL and LOW); `steer`: the
 * server's default (none unless given). */
export const calConcept = (calibration: any, layer = 20, steer: any = null) => parseConcepts({ model: "m",
	concepts: [{ name: "fluffy", title: "Fluffy", model: "m", reader: { layer, strength: 0.8, concept: 0.5, control: 0.1 },
		steer, generation: { edge: 1.6 }, calibration }] })[0];
/** A concept whose calibration fits (the server's default is then its dose, basis chat). */
export const chatConcept = (cal: any = CAL) => calConcept(cal, 20, { layer: 20, strength: cal.default_dose, basis: "chat" });

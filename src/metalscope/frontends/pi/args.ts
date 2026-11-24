/**
 * The rules every command shares: what a name may be, quotes around it, the dose range, where a usage error points,
 * the completions, and how numbers are written. No imports, so any module can use them.
 */

/** A concept, vector or lens name as the server takes it (steer._NAME, jlens.use, learn/plan.py): a file name and a
 * header token, letters, digits, _ . -, not starting with . or -, at most 128. */
export const NAME = /^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}$/;
/** The longest concept name (vectors.NAME_MAX): its job ids must stay names too. */
export const NAME_MAX = 96;

/** A name as typed, without the quotes around it (`/steer "fluffy"`). */
export const unquote = (s: string) => s.replace(/["“”]/g, "");

export const MAX_STRENGTH = 4.0;
export const MIN_DOSE = 0.001; // below this a dose changes nothing measurable
export const DOSE_RANGE = `a number from ${MIN_DOSE} to ${MAX_STRENGTH}`;

/** A number as typed, else undefined; a dose (DOSE_RANGE) as typed, else undefined. */
export const parseNumber = (s: string): number | undefined => {
	const d = Number(s);
	return s.trim() !== "" && Number.isFinite(d) ? d : undefined;
};
export const parseDose = (s: string): number | undefined => {
	const d = parseNumber(s);
	return d !== undefined && d >= MIN_DOSE && d <= MAX_STRENGTH ? d : undefined;
};

/** What ends every usage error of a command: where its forms are listed. */
export const seeHelp = (cmd: string) => `see /${cmd} help`;

/** The completions among `xs` that start with `prefix`, each after `before` (what is typed ahead of the word). */
export const offer = (prefix: string, xs: string[], before = "") =>
	xs.filter((c) => c.startsWith(prefix)).map((c) => ({ value: before + c, label: before + c }));

/** How the commands, entries and the Steer page write numbers: a dose as the sweep writes it (three significant figures); log-odds,
 * signed to two decimals; a share as a whole percentage; three decimals (a hit@10); a span of minutes with its unit, hours past 90 minutes ("20 to 35 min", "20 min to 2 h", "1.5 to 7.5 h", "26 min";
 * `sep` "–" as the help's tables write it). */
export const sig3 = (x: number) => +x.toPrecision(3);
export const signed = (x: number) => `${x >= 0 ? "+" : ""}${x.toFixed(2)}`;
export const percent = (x: number) => `${Math.round(100 * x)}%`;
export const f3 = (x: number) => x.toFixed(3);
export const timeSpan = ([from, to]: readonly [number, number], sep = " to ") => {
	const h = (m: number) => `${Math.round(m / 30) / 2}`;     // to the half hour
	if (from === to) return to <= 90 ? `${to} min` : `${h(to)} h`;   // one measured figure
	if (to <= 90) return `${from}${sep}${to} min`;
	return from >= 60 ? `${h(from)}${sep}${h(to)} h` : `${from} min${sep}${h(to)} h`;
};
/** Two spans one after the other (a learn job, then its calibration). */
export const addSpans = (a: readonly [number, number], b: readonly [number, number]): [number, number] => [a[0] + b[0], a[1] + b[1]];

/**
 * The dock's idle scene, "The lens comes by" (the user's, approved as drawn): a box-headed robot on a floor tile seen
 * from above. One loop of IDLE_FRAMES frames, IDLE_MS each (about 14 s): it waves alone; the lens floats in from the
 * upper left and it leans in; it takes the lens and looks back through it (an eye in the glass glances and blinks);
 * the lens floats off to the right and it waves goodbye. IDLE_W by IDLE_H pixels, two to a character (lights.ts
 * rgbBlocks). A port of the Python that drew it (idleFrame equals its frames pixel for pixel: test/idle.test.ts).
 * dock.ts shows it while the dock has no reply to show.
 */

import { COLOURS, type Rgb } from "./draw.ts";
import { rgbBlocks } from "./lights.ts";

export const IDLE_W = 36, IDLE_H = 22, IDLE_FRAMES = 84, IDLE_MS = 170;
/** The scene's lines in the dock (two pixels to a character). */
export const IDLE_LINES = IDLE_H / 2;

/** A pixel's colour by its letter; "." the dock's own background (null: a blank, so the art sits flush). */
export const IDLE_PAL: Record<string, Rgb | null> = {
	".": null, k: [22, 25, 30], q: [30, 34, 41], r: [46, 52, 62], f: [78, 85, 97], g: [126, 134, 147],
	t: [196, 201, 209], w: [246, 246, 250], v: COLOURS.lens, h: [150, 98, 40],
};

/** Python's round(): half to even (the reference drew with it). */
function pyRound(x: number): number {
	const f = Math.floor(x), d = x - f;
	return d > 0.5 ? f + 1 : d < 0.5 ? f : f % 2 === 0 ? f : f + 1;
}

class Canvas {
	g: string[][];
	constructor(base: string[]) {
		this.g = base.map((r) => [...r]);
	}
	px(x: number, y: number, c: string) {
		x = pyRound(x);
		y = pyRound(y);
		if (x >= 0 && x < IDLE_W && y >= 0 && y < IDLE_H) this.g[y][x] = c;
	}
	rect(x0: number, y0: number, x1: number, y1: number, c: string) {
		for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) this.px(x, y, c);
	}
	line(x0: number, y0: number, x1: number, y1: number, c: string) {
		const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)) || 1;
		for (let k = 0; k <= n; k++) this.px(x0 + ((x1 - x0) * k) / n, y0 + ((y1 - y0) * k) / n, c);
	}
	rows(): string[] {
		return this.g.map((r) => r.join(""));
	}
}

/** The floor: a rhombus slab seen from above (lit rim, four tiles, a short front face). The same every frame. */
let floorRows: string[] | undefined;
function floor(): string[] {
	if (floorRows) return floorRows;
	const c = new Canvas(Array.from({ length: IDLE_H }, () => ".".repeat(IDLE_W)));
	const FCX = 17.5, FCY = 14.0, FHW = 18.0, FHH = 5.5;
	const top = new Set<number>();
	for (let y = 0; y < IDLE_H; y++)
		for (let x = 0; x < IDLE_W; x++) {
			const u = (x - FCX) / FHW + (y - FCY) / FHH, v = (x - FCX) / FHW - (y - FCY) / FHH;
			if (Math.abs(u) <= 1 && Math.abs(v) <= 1) {
				top.add(y * IDLE_W + x);
				c.px(x, y, Math.min(Math.abs(u), Math.abs(v)) < 0.07 ? "q" : "r");
			}
		}
	for (let x = 0; x < IDLE_W; x++) {
		const col: number[] = [];
		for (let y = 0; y < IDLE_H; y++) if (top.has(y * IDLE_W + x)) col.push(y);
		if (!col.length) continue;
		const last = col[col.length - 1];
		c.px(x, col[0], "f");
		c.px(x, last, "f");
		c.px(x, last + 1, "q");
		c.px(x, last + 2, "q");
	}
	return (floorRows = c.rows());
}

const RX = 14, BY = 11;   // the robot's x; its body's top row
const HOVER: [number, number] = [7, 7], EYE: [number, number] = [RX + 1, 7];   // where the lens hovers; held up to an eye
const ease = (t: number) => t * t * (3 - 2 * t);

function robot(c: Canvas, i: number, st: { wave: boolean; lean: number; blink: boolean }) {
	const bob = st.wave && [2, 3, 6, 7, 10, 11].includes(i % 12) ? 1 : 0;
	const x = RX, y = 4 + bob, hx0 = x - st.lean;
	c.rect(x, BY + 5, x + 7, BY + 5, "k");   // its shadow
	c.rect(x + 1, BY + 6, x + 6, BY + 6, "k");
	// antenna, its tip blinking
	c.rect(hx0 + 3, y - 2, hx0 + 3, y - 1, "g");
	c.px(hx0 + 3, y - 3, Math.floor(i / 3) % 2 === 0 ? "w" : "g");
	// head: a box from above, bright top face, then the front with a dark visor and two eyes
	c.rect(hx0, y, hx0 + 6, y, "w");
	c.rect(hx0, y + 1, hx0 + 6, y + 5, "t");
	c.rect(hx0 + 6, y + 1, hx0 + 6, y + 5, "g");
	c.rect(hx0 + 1, y + 2, hx0 + 5, y + 4, "k");
	if (st.blink) {
		c.px(hx0 + 1, y + 3, "g");
		c.px(hx0 + 4, y + 3, "g");
	} else {
		c.rect(hx0 + 1, y + 2, hx0 + 1, y + 3, "w");
		c.rect(hx0 + 4, y + 2, hx0 + 4, y + 3, "w");
	}
	c.rect(x + 2, y + 6, x + 4, y + 6, "f");   // neck
	// body: top face, front, three chest lights that tick, legs, feet
	c.rect(x + 1, BY, x + 5, BY, "t");
	c.rect(x + 1, BY + 1, x + 5, BY + 3, "g");
	c.rect(x + 5, BY + 1, x + 5, BY + 3, "f");
	for (let k = 0; k < 3; k++) c.px(x + 2 + k, BY + 2, Math.floor(i / 4) % 3 === k ? "w" : "r");
	c.px(x + 2, BY + 4, "f");
	c.px(x + 4, BY + 4, "f");
	c.rect(x + 1, BY + 5, x + 2, BY + 5, "t");
	c.rect(x + 4, BY + 5, x + 5, BY + 5, "t");
	c.rect(x, BY + 1, x, BY + 2, "f");   // near arm, at rest
	// far arm: waving, or at rest
	if (st.wave) {
		const [ex, ey] = Math.floor(i / 2) % 2 === 0 ? [x + 8, BY - 2] : [x + 9, BY];
		c.line(x + 6, BY + 1, ex, ey, "g");
		c.rect(ex, ey - 1, ex + 1, ey, "w");
	} else {
		c.rect(x + 6, BY + 1, x + 6, BY + 2, "f");
		c.px(x + 6, BY + 3, "t");
	}
}

/** The lens: a dark glass in a violet ring; in it the eye looking back (its pupil in column 0 or 1, or shut), else a
 * glint sliding across (0-4), else the glass's resting shine. */
function ring(c: Canvas, cx: number, cy: number, glint: number | null, eye: number | "blink" | null) {
	for (let dy = -3; dy <= 3; dy++)
		for (let dx = -3; dx <= 3; dx++) {
			const r = dx * dx + dy * dy;
			if (r <= 5) c.px(cx + dx, cy + dy, "k");
			else if (r <= 10) c.px(cx + dx, cy + dy, "v");
		}
	if (eye !== null) {
		if (eye === "blink") c.rect(cx - 2, cy, cx + 1, cy, "g");
		else {
			c.rect(cx - 1, cy - 2, cx + 1, cy - 2, "w");
			c.rect(cx - 2, cy - 1, cx + 1, cy + 1, "w");
			c.rect(cx - 1 + eye, cy - 1, cx - 1 + eye, cy, "k");
		}
	} else if (glint !== null && glint >= 0 && glint < 5) {
		const [gx, gy] = ([[-2, 0], [-1, -1], [0, -2], [1, -1], [2, 0]] as const)[glint];
		c.px(cx + gx, cy + gy, "w");
	} else c.px(cx - 1, cy - 1, "f");
}

/** The lens's handle, brown, trailing down to the left of the ring wherever the lens goes. */
function handle(c: Canvas, cx: number, cy: number) {
	c.line(cx - 2, cy + 3, cx - 4, cy + 5, "h");
	c.line(cx - 3, cy + 3, cx - 5, cy + 5, "h");
}

/** Frame `i` (mod IDLE_FRAMES): IDLE_H rows of IDLE_W palette letters (IDLE_PAL). */
export function idleFrame(i: number): string[] {
	i = ((i % IDLE_FRAMES) + IDLE_FRAMES) % IDLE_FRAMES;
	const c = new Canvas(floor());
	const st = { wave: false, lean: 0, blink: false };
	let lens: [number, number] | null = null, eye: number | "blink" | null = null, glint: number | null = null, held = false;
	if (i < 18) {                                   // alone: a wave, then a look around
		st.wave = i < 12;
		st.blink = i === 15 || i === 16;
	} else if (i < 30) {                            // the lens floats in from the upper left
		const t = ease((i - 18 + 1) / 12);
		lens = [-5 + (HOVER[0] + 5) * t, 2 + (HOVER[1] - 2) * t + Math.sin(t * Math.PI) * -2];
		st.lean = i >= 26 ? 1 : 0;
	} else if (i < 42) {                            // it hovers; the robot leans in to look
		lens = [HOVER[0], HOVER[1] + [0, 0, -1, -1, -1, 0, 0, 0, 1, 1, 0, 0][i - 30]];
		glint = i - 33;
		st.lean = 1;
		st.blink = i === 38 || i === 39;
	} else if (i < 46) {                            // the robot takes it
		const t = (i - 42 + 1) / 4;
		lens = [HOVER[0] + (EYE[0] - HOVER[0]) * t, HOVER[1] + (EYE[1] - HOVER[1]) * t];
		held = true;
	} else if (i < 66) {                            // looking back at you through it
		lens = EYE;
		held = true;
		eye = i === 60 || i === 61 ? "blink" : i >= 51 && i <= 56 ? 1 : 0;
	} else if (i < 78) {                            // it lets go: the lens floats off to the right
		const t = ease((i - 66 + 1) / 12);
		lens = [EYE[0] + (IDLE_W + 6 - EYE[0]) * t, EYE[1] + (1 - EYE[1]) * t + Math.sin(t * Math.PI) * -2];
		st.wave = i >= 69;
	} else {
		st.wave = i < 81;
		st.blink = i === 82 || i === 83;
	}
	robot(c, i, st);
	if (lens) {
		const cx = pyRound(lens[0]), cy = pyRound(lens[1]);
		if (held) {                                 // the arm up to the handle, the hand on it
			c.line(RX - 1, BY + 1, cx - 2, cy + 3, "h");
			c.rect(RX - 2, BY + 1, RX - 1, BY + 2, "w");
		} else handle(c, cx, cy);
		ring(c, cx, cy, glint, eye);
	}
	return c.rows();
}

const drawn = new Map<string, string[]>();
/** Frame `i` as IDLE_LINES lines of half blocks, IDLE_W wide, each ending in a full reset (the dock re-applies its
 * background: draw.ts tint). Drawn once a frame and colour mode. */
export function idleLines(i: number, mode: "truecolor" | "256color"): string[] {
	const key = `${mode}:${((i % IDLE_FRAMES) + IDLE_FRAMES) % IDLE_FRAMES}`;
	let lines = drawn.get(key);
	if (!lines) {
		const cells = idleFrame(i).flatMap((r) => [...r].map((ch) => IDLE_PAL[ch] ?? null));
		drawn.set(key, (lines = rgbBlocks(cells, IDLE_W, IDLE_H, mode)));
	}
	return lines;
}

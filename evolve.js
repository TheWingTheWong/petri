// Petri — automatic discovery of life-like interaction matrices.
//
// "Life-like" is scored from three measurable ingredients, each in [0, 1]:
//   clump      — the density field is lumpier than random scatter (structure)
//   coherence  — particles in the same neighbourhood move together (bodies,
//                not gas); measured as local velocity alignment above chance
//   motion     — things are actually moving (not frozen crystals)
// score = clump^0.6 · (0.25 + coherence) · motion
//
// The search is a (μ+λ) evolution strategy with an archive that keeps the
// best *distinct* specimens, so the gallery doesn't fill with near-copies.

import { createWorld, step, randomMatrix, mulberry32, snap } from './sim.js';

export function score(w) {
  const B = Math.max(6, Math.round(2 / w.rMax));
  const cells = B * B;
  const count = new Float64Array(cells), sx = new Float64Array(cells), sy = new Float64Array(cells), sm = new Float64Array(cells);
  let speed = 0;
  for (let i = 0; i < w.n; i++) {
    const c = Math.min(B - 1, Math.floor(w.y[i] * B)) * B + Math.min(B - 1, Math.floor(w.x[i] * B));
    const vx = w.vx[i], vy = w.vy[i], m = Math.hypot(vx, vy);
    count[c]++; sx[c] += vx; sy[c] += vy; sm[c] += m; speed += m;
  }
  const mean = w.n / cells;
  let v = 0;
  for (let c = 0; c < cells; c++) v += (count[c] - mean) ** 2;
  const cv = Math.sqrt(v / cells) / mean, cv0 = 1 / Math.sqrt(mean);
  const clump = Math.max(0, 1 - Math.exp(-Math.max(0, cv / cv0 - 1) / 4));

  let cohSum = 0, cohW = 0;
  for (let c = 0; c < cells; c++) {
    if (count[c] < 4 || sm[c] === 0) continue;
    const align = Math.hypot(sx[c], sy[c]) / sm[c];
    cohSum += Math.max(0, align - 1 / Math.sqrt(count[c])) * count[c];
    cohW += count[c];
  }
  const coherence = cohW ? cohSum / cohW : 0;
  const s = speed / w.n / w.rMax; // mean speed in "reach lengths" per unit time
  const motion = 1 - Math.exp(-s / 0.25);
  return { score: Math.pow(clump, 0.6) * (0.25 + coherence) * motion, clump, coherence, motion, speed: s };
}

/** Run a fresh world with `matrix` and score its settled behaviour. */
export function evaluate(matrix, { k, seed = 1, n = 700, rMax = 0.2, friction = 0.04, steps = 200 } = {}) {
  const w = createWorld({ n, k, seed, rMax, friction, matrix });
  const samples = [];
  for (let t = 1; t <= steps; t++) {
    step(w);
    if (t > steps * 0.6 && t % Math.round(steps / 8) === 0) samples.push(score(w));
  }
  const avg = (key) => samples.reduce((a, s) => a + s[key], 0) / samples.length;
  return { score: avg('score'), clump: avg('clump'), coherence: avg('coherence'), motion: avg('motion'), world: w };
}

export function mutate(m, rng, strength = 1) {
  const out = Float32Array.from(m);
  const edits = 1 + Math.floor(rng() * 3 * strength);
  for (let e = 0; e < edits; e++) {
    const i = Math.floor(rng() * out.length);
    out[i] = rng() < 0.15 ? snap(rng() * 2 - 1) : snap(out[i] + (rng() < 0.5 ? -1 : 1) * (0.1 + 0.1 * Math.floor(rng() * 3)));
  }
  return out;
}

export const distance = (a, b) => { let d = 0; for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]); return d / a.length; };

/**
 * Evolution strategy. Call next() repeatedly; each call evaluates one
 * generation and returns a progress report. Deterministic for a given seed.
 */
export function createEvolution({ k, seed = 1, popSize = 8, elite = 3, archiveSize = 6, evalOpts = {}, start = [] }) {
  const rng = mulberry32(seed);
  let parents = [];
  const archive = [];
  let gen = 0, tested = 0;
  const ev = (m) => { tested++; const r = evaluate(m, { k, seed: seed + 7, ...evalOpts }); return { matrix: m, ...r }; };

  function remember(c) {
    const twin = archive.findIndex((a) => distance(a.matrix, c.matrix) < 0.12);
    if (twin >= 0) { if (c.score > archive[twin].score) archive[twin] = c; else return; }
    else archive.push(c);
    archive.sort((a, b) => b.score - a.score);
    archive.length = Math.min(archive.length, archiveSize);
  }

  return {
    next() {
      gen++;
      const kids = [];
      if (!parents.length) {
        for (const m of start) kids.push(ev(Float32Array.from(m)));
        while (kids.length < popSize) kids.push(ev(randomMatrix(k, rng)));
      } else {
        for (let i = 0; i < popSize - 1; i++) kids.push(ev(mutate(parents[i % parents.length].matrix, rng, 1 + (i % 3) * 0.5)));
        kids.push(ev(randomMatrix(k, rng))); // fresh blood
      }
      const pool = [...parents, ...kids].sort((a, b) => b.score - a.score);
      parents = pool.slice(0, elite);
      kids.forEach(remember);
      return { gen, tested, best: parents[0], archive: archive.slice() };
    },
  };
}

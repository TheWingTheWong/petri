// Petri — particle-life simulation core. Pure, deterministic, dependency-free.
//
// N particles of K species live on a unit torus. Every pair closer than rMax
// interacts through a piecewise-linear force: universal repulsion at very
// short range (r < beta), then attraction or repulsion set by an asymmetric
// K×K matrix. Because a[i][j] != a[j][i], momentum is not conserved and
// chasing, orbiting and cell-like structures emerge on their own.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Force profile; r is distance / rMax in [0, 1). */
export function force(r, a, beta) {
  if (r < beta) return r / beta - 1;
  if (r < 1) return a * (1 - Math.abs(2 * r - 1 - beta) / (1 - beta));
  return 0;
}

/** Shortest signed offset on the unit torus. */
export const wrapDelta = (d) => (d > 0.5 ? d - 1 : d < -0.5 ? d + 1 : d);

export const snap = (v) => Math.max(-1, Math.min(1, Math.round(v * 10) / 10)) || 0; // no -0

export function randomMatrix(k, rng) {
  return Float32Array.from({ length: k * k }, () => snap(rng() * 2 - 1));
}

export const PRESETS = {
  snakes: (k) => build(k, (i, j) => (i === j ? 1 : j === (i + 1) % k ? 0.2 : 0)),
  chase: (k) => build(k, (i, j) => (i === j ? 0.6 : j === (i + 1) % k ? 0.8 : i === (j + 1) % k ? -0.6 : 0)),
  clusters: (k) => build(k, (i, j) => (i === j ? 0.9 : -0.3)),
  cells: (k) => build(k, (i, j) => (i === j ? (i === 0 ? 1 : 0.2) : j === 0 ? 0.6 : i === 0 ? -0.2 : -0.1)),
};
function build(k, f) {
  const m = new Float32Array(k * k);
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) m[i * k + j] = snap(f(i, j));
  return m;
}

export const DEFAULTS = { n: 2400, k: 6, rMax: 0.08, beta: 0.3, friction: 0.04, forceFactor: 10, dt: 0.02 };

export function createWorld(opts = {}) {
  const o = { ...DEFAULTS, seed: 1, ...opts };
  if (o.rMax > 1 / 3) throw new Error('rMax must be <= 1/3');
  const rng = mulberry32(o.seed);
  const matrix = o.matrix ? Float32Array.from(o.matrix) : randomMatrix(o.k, rng);
  if (matrix.length !== o.k * o.k) throw new Error('matrix size must be k*k');
  const w = {
    ...o, matrix,
    x: new Float32Array(o.n), y: new Float32Array(o.n),
    vx: new Float32Array(o.n), vy: new Float32Array(o.n),
    type: new Uint8Array(o.n),
    fx: new Float32Array(o.n), fy: new Float32Array(o.n),
    steps: 0,
  };
  const prng = mulberry32(o.seed ^ 0x9e3779b9);
  for (let i = 0; i < o.n; i++) { w.x[i] = prng(); w.y[i] = prng(); w.type[i] = Math.floor(prng() * o.k); }
  return w;
}

/** Bucket particles into a G×G grid (cell side >= rMax) via counting sort. */
function buildGrid(w) {
  const G = Math.max(3, Math.floor(1 / w.rMax));
  const cells = G * G;
  if (!w.grid || w.grid.G !== G || w.grid.order.length !== w.n) {
    w.grid = { G, start: new Int32Array(cells + 1), order: new Int32Array(w.n), cellOf: new Int32Array(w.n) };
  }
  const { start, order, cellOf } = w.grid;
  start.fill(0);
  for (let i = 0; i < w.n; i++) {
    const c = (Math.min(G - 1, Math.floor(w.y[i] * G)) * G) + Math.min(G - 1, Math.floor(w.x[i] * G));
    cellOf[i] = c; start[c + 1]++;
  }
  for (let c = 0; c < cells; c++) start[c + 1] += start[c];
  const fill = start.slice(0, cells);
  for (let i = 0; i < w.n; i++) order[fill[cellOf[i]]++] = i;
  return w.grid;
}

/** Accumulate inter-particle forces into w.fx / w.fy. */
export function computeForces(w, brute = false) {
  const { n, k, x, y, type, matrix, rMax, beta, fx, fy } = w;
  fx.fill(0); fy.fill(0);
  const r2max = rMax * rMax;
  const pair = (i, j) => {
    const dx = wrapDelta(x[j] - x[i]), dy = wrapDelta(y[j] - y[i]);
    const d2 = dx * dx + dy * dy;
    if (d2 === 0 || d2 >= r2max) return;
    const d = Math.sqrt(d2);
    const f = force(d / rMax, matrix[type[i] * k + type[j]], beta);
    fx[i] += (dx / d) * f; fy[i] += (dy / d) * f;
  };
  if (brute) {
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) pair(i, j);
  } else {
    const { G, start, order, cellOf } = buildGrid(w);
    for (let i = 0; i < n; i++) {
      const cx = cellOf[i] % G, cy = (cellOf[i] / G) | 0;
      for (let oy = -1; oy <= 1; oy++) {
        const row = ((cy + oy + G) % G) * G;
        for (let ox = -1; ox <= 1; ox++) {
          const c = row + ((cx + ox + G) % G);
          for (let s = start[c], e = start[c + 1]; s < e; s++) { const j = order[s]; if (j !== i) pair(i, j); }
        }
      }
    }
  }
  const scale = rMax * w.forceFactor;
  for (let i = 0; i < n; i++) { fx[i] *= scale; fy[i] *= scale; }
}

/**
 * Advance one tick. Optional `stir` = {x, y, radius, strength}: a mouse
 * "pipette" that pulls (strength > 0) or pushes (< 0) nearby particles.
 */
export function step(w, stir = null) {
  computeForces(w);
  const { n, x, y, vx, vy, fx, fy, dt } = w;
  const damp = Math.pow(0.5, dt / w.friction);
  for (let i = 0; i < n; i++) {
    let ax = fx[i], ay = fy[i];
    if (stir) {
      const dx = wrapDelta(stir.x - x[i]), dy = wrapDelta(stir.y - y[i]);
      const d = Math.hypot(dx, dy);
      if (d < stir.radius && d > 1e-6) { const s = stir.strength * (1 - d / stir.radius); ax += (dx / d) * s; ay += (dy / d) * s; }
    }
    vx[i] = vx[i] * damp + ax * dt;
    vy[i] = vy[i] * damp + ay * dt;
    x[i] = (x[i] + vx[i] * dt + 1) % 1;
    y[i] = (y[i] + vy[i] * dt + 1) % 1;
  }
  w.steps++;
}

export function kineticEnergy(w) {
  let e = 0;
  for (let i = 0; i < w.n; i++) e += w.vx[i] * w.vx[i] + w.vy[i] * w.vy[i];
  return e / w.n;
}

// ------------------------------------------------- shareable URL state ----
// Matrix values are on a 0.1 grid in [-1, 1]; each maps to one letter a..u.
const ALPHA = 'abcdefghijklmnopqrstu';
export function encodeMatrix(m) {
  return Array.from(m, (v) => ALPHA[Math.round(snap(v) * 10) + 10]).join('');
}
export function decodeMatrix(s) {
  return Float32Array.from(s, (c) => {
    const i = ALPHA.indexOf(c);
    if (i < 0) throw new Error(`bad matrix char ${c}`);
    return (i - 10) / 10;
  });
}

export function encodeState({ seed, k, n, matrix, rMax, friction }) {
  const p = new URLSearchParams({ s: seed, k, n, m: encodeMatrix(matrix), r: rMax, f: friction });
  return p.toString();
}
export function decodeState(str) {
  const p = new URLSearchParams(str.replace(/^#/, ''));
  const out = {};
  if (p.has('s')) out.seed = Number(p.get('s')) >>> 0;
  if (p.has('k')) out.k = clampInt(p.get('k'), 2, 8);
  if (p.has('n')) out.n = clampInt(p.get('n'), 100, 20000);
  if (p.has('r')) out.rMax = Math.min(1 / 3, Math.max(0.02, Number(p.get('r')) || DEFAULTS.rMax));
  if (p.has('f')) out.friction = Math.min(1, Math.max(0.005, Number(p.get('f')) || DEFAULTS.friction));
  if (p.has('m')) {
    const m = decodeMatrix(p.get('m'));
    const k = Math.round(Math.sqrt(m.length));
    if (k * k === m.length && k >= 2 && k <= 8) { out.matrix = m; out.k = k; }
  }
  return out;
}
const clampInt = (v, lo, hi) => Math.min(hi, Math.max(lo, Math.round(Number(v)) || lo));

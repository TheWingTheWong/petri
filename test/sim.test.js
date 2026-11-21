import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  force, wrapDelta, createWorld, computeForces, step, encodeState, decodeState,
  encodeMatrix, decodeMatrix, PRESETS, randomMatrix, mulberry32, kineticEnergy, snap,
} from '../sim.js';

test('force profile: repulsive core, matrix-scaled hump, zero beyond rMax', () => {
  assert.equal(force(0, 1, 0.3), -1);
  assert.ok(force(0.15, 1, 0.3) < 0);
  assert.equal(force(0.3, 1, 0.3), 0);
  assert.ok(Math.abs(force(0.65, 1, 0.3) - 1) < 1e-9, 'peak at (1+beta)/2');
  assert.ok(Math.abs(force(0.65, -0.5, 0.3) + 0.5) < 1e-9);
  assert.equal(force(1, 1, 0.3), 0);
  assert.equal(force(2, 1, 0.3), 0);
});

test('torus wrap picks the shortest offset', () => {
  assert.ok(Math.abs(wrapDelta(0.9) + 0.1) < 1e-12);
  assert.ok(Math.abs(wrapDelta(-0.8) - 0.2) < 1e-12);
  assert.equal(wrapDelta(0.2), 0.2);
});

test('spatial grid gives the same forces as brute force', () => {
  for (const rMax of [0.08, 0.15, 0.3]) {
    const a = createWorld({ n: 600, k: 5, seed: 9, rMax });
    const b = createWorld({ n: 600, k: 5, seed: 9, rMax });
    computeForces(a); computeForces(b, true);
    for (let i = 0; i < a.n; i++) {
      assert.ok(Math.abs(a.fx[i] - b.fx[i]) < 1e-4 && Math.abs(a.fy[i] - b.fy[i]) < 1e-4, `rMax ${rMax} particle ${i}`);
    }
  }
});

test('forces across the wrap seam are counted', () => {
  const w = createWorld({ n: 2, k: 1, matrix: [1], seed: 1 });
  w.x[0] = 0.005; w.y[0] = 0.5; w.x[1] = 0.995; w.y[1] = 0.5; // 0.01 apart via the seam
  computeForces(w);
  assert.ok(w.fx[0] > 0, 'pushed away from neighbour on the left (repulsive core)');
  assert.ok(w.fx[1] < 0);
});

test('simulation is deterministic and stays on the torus', () => {
  const a = createWorld({ n: 800, seed: 42 }), b = createWorld({ n: 800, seed: 42 });
  for (let i = 0; i < 40; i++) { step(a); step(b); }
  assert.deepEqual(a.x, b.x);
  assert.deepEqual(a.vy, b.vy);
  for (let i = 0; i < a.n; i++) assert.ok(a.x[i] >= 0 && a.x[i] < 1 && a.y[i] >= 0 && a.y[i] < 1);
  assert.ok(Number.isFinite(kineticEnergy(a)));
});

test('self-attracting species clump together', () => {
  const w = createWorld({ n: 400, k: 1, matrix: [1], seed: 3 });
  const nearCount = () => { computeForces(w); let c = 0; for (let i = 0; i < 200; i++) for (let j = 0; j < w.n; j++) { const dx = wrapDelta(w.x[j] - w.x[i]), dy = wrapDelta(w.y[j] - w.y[i]); if (i !== j && dx * dx + dy * dy < 0.0025) c++; } return c; };
  const before = nearCount();
  for (let i = 0; i < 150; i++) step(w);
  assert.ok(nearCount() > before * 2, 'neighbour density rises as clusters form');
});

test('stir pulls particles toward the pipette', () => {
  const w = createWorld({ n: 300, k: 2, matrix: [0, 0, 0, 0], seed: 5 });
  const d = () => { let s = 0; for (let i = 0; i < w.n; i++) s += Math.hypot(wrapDelta(w.x[i] - 0.5), wrapDelta(w.y[i] - 0.5)); return s; };
  const before = d();
  for (let i = 0; i < 30; i++) step(w, { x: 0.5, y: 0.5, radius: 0.4, strength: 4 });
  assert.ok(d() < before);
});

test('presets and random matrices are k*k on the 0.1 grid', () => {
  for (const [name, f] of Object.entries(PRESETS)) for (const k of [2, 5, 8]) {
    const m = f(k);
    assert.equal(m.length, k * k, name);
    m.forEach((v) => assert.ok(Math.abs(v * 10 - Math.round(v * 10)) < 1e-5 && v >= -1 && v <= 1));
  }
  const m = randomMatrix(6, mulberry32(1));
  assert.deepEqual(decodeMatrix(encodeMatrix(m)), m);
  assert.equal(snap(1.7), 1);
});

test('URL state round-trips and rejects junk safely', () => {
  const m = PRESETS.chase(4);
  const s = encodeState({ seed: 77, k: 4, n: 1500, matrix: m, rMax: 0.1, friction: 0.05 });
  const d = decodeState('#' + s);
  assert.deepEqual(d.matrix, m);
  assert.equal(d.k, 4); assert.equal(d.n, 1500); assert.equal(d.seed, 77); assert.equal(d.rMax, 0.1);
  assert.equal(decodeState('n=999999').n, 20000); // GPU_MAX: links made on the GPU engine at full size still reproduce
  assert.equal(decodeState('r=5').rMax, 1 / 3);
  assert.throws(() => decodeMatrix('zz'));
  assert.deepEqual(decodeState('m=abc'), {}, 'non-square matrix ignored');
});

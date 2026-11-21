import { test } from 'node:test';
import assert from 'node:assert/strict';
import { score, evaluate, mutate, distance, createEvolution } from '../evolve.js';
import { createWorld, PRESETS, mulberry32 } from '../sim.js';

const k = 5, fast = { n: 400, steps: 120, rMax: 0.2 };

test('random scatter with no motion scores ~0', () => {
  const w = createWorld({ n: 1500, k: 3, seed: 2 });
  const s = score(w);
  assert.ok(s.score < 0.02 && s.motion < 0.01, JSON.stringify(s));
});

test('life-likeness ranks moving creatures above gas and frozen clumps', () => {
  const gas = evaluate(new Float32Array(k * k), { k, ...fast }).score;
  const frozen = evaluate(PRESETS.clusters(k), { k, ...fast }).score;
  const alive = Math.max(evaluate(PRESETS.snakes(k), { k, ...fast }).score, evaluate(PRESETS.cells(k), { k, ...fast }).score);
  assert.ok(gas < 0.05, `gas ${gas}`);
  assert.ok(alive > frozen && alive > gas + 0.3, `alive ${alive} frozen ${frozen} gas ${gas}`);
});

test('mutation stays on the 0.1 grid within [-1, 1] and changes something', () => {
  const rng = mulberry32(4);
  const m = PRESETS.chase(k);
  for (let i = 0; i < 50; i++) {
    const c = mutate(m, rng, 2);
    assert.ok(distance(c, m) > 0 || i > 0);
    c.forEach((v) => assert.ok(v >= -1 && v <= 1 && Math.abs(v * 10 - Math.round(v * 10)) < 1e-5));
  }
});

test('evolution is deterministic, elitist and keeps distinct finds', () => {
  const run = () => { const e = createEvolution({ k: 3, seed: 11, popSize: 4, evalOpts: fast }); return [e.next(), e.next(), e.next()]; };
  const a = run(), b = run();
  assert.deepEqual(a.map((r) => r.best.score), b.map((r) => r.best.score), 'same seed, same search');
  for (let g = 1; g < a.length; g++) assert.ok(a[g].best.score >= a[g - 1].best.score, 'best never regresses');
  const arch = a[2].archive;
  for (let i = 0; i < arch.length; i++) for (let j = i + 1; j < arch.length; j++) assert.ok(distance(arch[i].matrix, arch[j].matrix) >= 0.12);
  for (let i = 1; i < arch.length; i++) assert.ok(arch[i - 1].score >= arch[i].score, 'archive sorted');
  assert.equal(a[2].tested, 12);
});

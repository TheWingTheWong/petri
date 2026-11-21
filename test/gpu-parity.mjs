// Runs in headless Chromium: the WebGL2 engine must reproduce the CPU engine.
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';
import { startServer } from '../serve.mjs';

const { chromium } = createRequire(import.meta.url)('playwright');
const server = await startServer(0);
const browser = await chromium.launch();
let failed = false;
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/test/blank.html`);
  const res = await page.evaluate(async () => {
    const { createWorld, step } = await import('/sim.js');
    const { GpuSim } = await import('/gpu.js');
    if (!GpuSim.supported()) return { skipped: true };
    const out = {};
    for (const [label, opts, steps, stir] of [
      ['1 step', { n: 700, k: 5, seed: 3, rMax: 0.12 }, 1, null],
      ['3 steps + stir', { n: 1000, k: 6, seed: 8, rMax: 0.08 }, 3, { x: 0.5, y: 0.5, radius: 0.2, strength: 5 }],
      ['non-square n', { n: 333, k: 3, seed: 1, rMax: 0.2 }, 2, null],
    ]) {
      const cpu = createWorld(opts), gw = createWorld(opts);
      const gpu = new GpuSim(document.createElement('canvas'), ['#ffffff']);
      gpu.load(gw);
      for (let s = 0; s < steps; s++) { step(cpu, stir); gpu.step(stir); }
      gpu.readback();
      let maxDx = 0, maxDv = 0;
      for (let i = 0; i < cpu.n; i++) {
        const dx = Math.abs(cpu.x[i] - gw.x[i]), dy = Math.abs(cpu.y[i] - gw.y[i]);
        maxDx = Math.max(maxDx, Math.min(dx, 1 - dx), Math.min(dy, 1 - dy));
        maxDv = Math.max(maxDv, Math.abs(cpu.vx[i] - gw.vx[i]), Math.abs(cpu.vy[i] - gw.vy[i]));
      }
      out[label] = { maxDx, maxDv, moved: cpu.vx.some((v) => v !== 0) };
    }
    return out;
  });
  if (res.skipped) console.log('- WebGL2 float targets unavailable: GPU parity skipped');
  else for (const [label, r] of Object.entries(res)) {
    assert.ok(r.moved, `${label}: particles moved`);
    assert.ok(r.maxDx < 1e-5 && r.maxDv < 1e-3, `${label}: max |dx| ${r.maxDx}, max |dv| ${r.maxDv}`);
    console.log(`✓ GPU matches CPU (${label}): max position error ${r.maxDx.toExponential(1)}`);
  }
  assert.deepEqual(errors, []);
} catch (e) { failed = true; console.error(e); }
finally { await browser.close(); server.close(); }
process.exit(failed ? 1 : 0);

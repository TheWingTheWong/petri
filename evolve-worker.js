// Runs the evolution strategy off the main thread so the microscope stays smooth.
import { createEvolution } from './evolve.js';

let evo = null, token = 0;

const pack = (a) => ({
  matrix: Array.from(a.matrix), score: a.score, clump: a.clump, coherence: a.coherence, motion: a.motion,
  thumb: { x: Array.from(a.world.x), y: Array.from(a.world.y), type: Array.from(a.world.type) },
});

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'stop') { token++; return; }
  if (msg.type !== 'start') return;
  const mine = ++token;
  evo = createEvolution({ k: msg.k, seed: msg.seed, evalOpts: msg.evalOpts, start: msg.start || [] });
  while (mine === token) {
    const r = evo.next();
    if (mine !== token) break;
    self.postMessage({ type: 'progress', gen: r.gen, tested: r.tested, k: msg.k, archive: r.archive.map(pack) });
    await new Promise((res) => setTimeout(res, 0)); // let a pending "stop" arrive
  }
};

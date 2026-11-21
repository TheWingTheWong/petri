import {
  createWorld, step, randomMatrix, mulberry32, PRESETS, snap, encodeState, decodeState, DEFAULTS, kineticEnergy,
} from './sim.js';
import { GpuSim } from './gpu.js';

const $ = (id) => document.getElementById(id);
const cv = $('dish');
const ctx = cv.getContext('2d');
const SPECIES = ['#ff5c7a', '#ffb020', '#b8f35a', '#35e0a1', '#3cc8ff', '#9b7bff', '#ff8a4c', '#efe9d8'];
const NAMES = 'ABCDEFGH';
const MONO = '"Plex Mono", ui-monospace, monospace';

let world, byType, paused = false, trails = true, fps = 60, stir = null, pointer = null;
let state = { seed: 1, ...DEFAULTS };
const glCanvas = $('gl');
const CPU_MAX = 6000, GPU_MAX = 20000;
let gpu = null, engine = 'cpu', frameNo = 0;
let worker = null, evolving = false, finds = [], selected = -1;
const FINDS_KEY = 'petri.finds.v1';
const gpuOk = GpuSim.supported() && new URLSearchParams(location.search).get('engine') !== 'cpu';
function startGpu() {
  try { gpu = gpu || new GpuSim(glCanvas, SPECIES); return true; }
  catch (err) { console.warn('GPU engine unavailable, using CPU:', err.message); gpu = null; return false; }
}

// --------------------------------------------------------------- world ----
function rebuild({ keepMatrix = true, newSeed = false } = {}) {
  if (newSeed) state.seed = (Math.random() * 2 ** 31) >>> 0;
  let matrix = keepMatrix && state.matrix && state.matrix.length === state.k * state.k ? state.matrix : null;
  if (!matrix) matrix = randomMatrix(state.k, mulberry32(state.seed));
  state.matrix = matrix;
  world = createWorld({ ...state, matrix });
  if (engine === 'gpu') gpu.load(world);
  byType = Array.from({ length: state.k }, () => []);
  for (let i = 0; i < world.n; i++) byType[world.type[i]].push(i);
  clearDish = true;
  syncUI();
}

function setMatrix(m) { state.matrix = m; world.matrix = m; syncUI(); }

function syncUI() {
  $('specId').textContent = state.seed.toString(16).toUpperCase().padStart(8, '0').slice(-6);
  $('specK').textContent = state.k;
  $('specN').textContent = state.n.toLocaleString('en');
  $('specE').textContent = engine === 'gpu' ? 'GPU · WebGL2' : 'CPU';
  $('engineBtn').firstChild.textContent = engine === 'gpu' ? 'Engine: GPU ' : 'Engine: CPU ';
  $('nIn').max = engine === 'gpu' ? GPU_MAX : CPU_MAX;
  for (const [id, key, fmt] of [['k', 'k', (v) => v], ['n', 'n', (v) => v], ['r', 'rMax', (v) => v.toFixed(3)], ['f', 'friction', (v) => v.toFixed(2)]]) {
    $(id + 'In').value = state[key]; $(id + 'Out').textContent = fmt(state[key]);
  }
  renderMatrix();
  history.replaceState(null, '', '#' + encodeState(state));
}

// -------------------------------------------------------------- matrix ----
function renderMatrix() {
  const k = state.k, m = state.matrix, el = $('matrix');
  el.style.gridTemplateColumns = `22px repeat(${k}, 1fr)`;
  let html = '<div></div>';
  for (let j = 0; j < k; j++) html += `<div class="mh"><i style="background:${SPECIES[j]}"></i>${NAMES[j]}</div>`;
  for (let i = 0; i < k; i++) {
    html += `<div class="mh"><i style="background:${SPECIES[i]};margin:0"></i></div>`;
    for (let j = 0; j < k; j++) {
      const v = m[i * k + j];
      const col = v >= 0 ? '53, 224, 161' : '255, 92, 122';
      const verb = v > 0 ? 'is attracted to' : v < 0 ? 'is repelled by' : 'ignores';
      html += `<button class="mc" data-i="${i * k + j}" title="${NAMES[i]} ${verb} ${NAMES[j]} (${v.toFixed(1)})" ` +
        `style="background:linear-gradient(rgba(${col},${Math.abs(v) * 0.85}),rgba(${col},${Math.abs(v) * 0.85})),#1c1e23">` +
        `<span>${v === 0 ? '·' : (v > 0 ? '+' : '−') + Math.abs(v).toFixed(1).replace(/^0/, '')}</span></button>`;
    }
  }
  el.innerHTML = html;
}
const nudge = (idx, d) => { const m = Float32Array.from(state.matrix); m[idx] = snap(m[idx] + d); setMatrix(m); };
$('matrix').addEventListener('click', (e) => {
  const c = e.target.closest('.mc'); if (!c) return;
  if (e.altKey) { const m = Float32Array.from(state.matrix); m[+c.dataset.i] = 0; setMatrix(m); } else nudge(+c.dataset.i, 0.2);
});
$('matrix').addEventListener('contextmenu', (e) => { const c = e.target.closest('.mc'); if (!c) return; e.preventDefault(); nudge(+c.dataset.i, -0.2); });
$('matrix').addEventListener('wheel', (e) => { const c = e.target.closest('.mc'); if (!c) return; e.preventDefault(); nudge(+c.dataset.i, e.deltaY < 0 ? 0.1 : -0.1); }, { passive: false });

// ------------------------------------------------------------ controls ----
const randomize = () => { state.seed = (Math.random() * 2 ** 31) >>> 0; state.matrix = null; rebuild({ keepMatrix: false }); toast('New specimen'); };
const mutate = () => {
  const rng = mulberry32((Math.random() * 2 ** 31) >>> 0);
  const m = Float32Array.from(state.matrix);
  for (let t = 0; t < Math.max(2, state.k); t++) { const i = Math.floor(rng() * m.length); m[i] = snap(m[i] + (rng() < 0.5 ? -0.2 : 0.2)); }
  setMatrix(m); toast('Mutated');
};
$('randBtn').onclick = randomize;
$('mutBtn').onclick = mutate;
$('reseedBtn').onclick = () => rebuild({ newSeed: true });
$('presets').innerHTML = Object.keys(PRESETS).slice(0, 3).map((p) => `<button class="b" data-p="${p}">${p}</button>`).join('');
$('presets').insertAdjacentHTML('beforeend', '');
const presetRow = document.createElement('div'); presetRow.className = 'row';
presetRow.innerHTML = Object.keys(PRESETS).slice(3).map((p) => `<button class="b" data-p="${p}">${p}</button>`).join('');
$('presets').after(presetRow);
for (const b of document.querySelectorAll('[data-p]')) b.onclick = () => { setMatrix(PRESETS[b.dataset.p](state.k)); rebuild(); toast(`Preset: ${b.dataset.p}`); };

for (const [id, key, live] of [['kIn', 'k', false], ['nIn', 'n', false], ['rIn', 'rMax', true], ['fIn', 'friction', true]]) {
  $(id).addEventListener('input', () => {
    state[key] = Number($(id).value);
    if (live) { world[key] = state[key]; syncUI(); }
    else if (key === 'k') { state.matrix = randomMatrix(state.k, mulberry32(state.seed)); rebuild(); }
    else rebuild();
  });
}
function setEngine(next, { quiet = false } = {}) {
  if (next === 'gpu' && !startGpu()) { if (!quiet) toast('GPU engine not available on this device'); next = 'cpu'; }
  if (engine === 'gpu' && next === 'cpu' && gpu) gpu.readback();
  engine = next;
  glCanvas.style.display = engine === 'gpu' ? 'block' : 'none';
  if (engine === 'cpu' && state.n > CPU_MAX) { // keep the same density: fewer particles, longer reach
    state.rMax = Math.min(1 / 3, state.rMax * Math.sqrt(state.n / CPU_MAX)); state.n = CPU_MAX;
    if (world) rebuild();
  } else if (world) {
    if (engine === 'gpu') gpu.load(world);
    clearDish = true; syncUI();
  }
}
$('engineBtn').onclick = () => { setEngine(engine === 'gpu' ? 'cpu' : 'gpu'); toast(`Engine: ${engine.toUpperCase()}`); };
const setPaused = (p) => { paused = p; $('pauseBtn').classList.toggle('on', p); $('pauseBtn').firstChild.textContent = p ? 'Play ' : 'Pause '; };
const setTrails = (t) => { trails = t; $('trailBtn').classList.toggle('on', t); };
$('pauseBtn').onclick = () => setPaused(!paused);
$('trailBtn').onclick = () => setTrails(!trails);
const stepOnce = () => { setPaused(true); engine === 'gpu' ? gpu.step() : step(world); };
$('stepBtn').onclick = stepOnce;
$('shareBtn').onclick = async () => {
  try { await navigator.clipboard.writeText(location.href); toast('Link copied. It opens this exact specimen.'); }
  catch { toast('Copy failed. The address bar holds the link.'); }
};
$('pngBtn').onclick = () => {
  const a = document.createElement('a');
  const out = document.createElement('canvas'); out.width = cv.width; out.height = cv.height;
  const o = out.getContext('2d');
  if (engine === 'gpu') o.drawImage(glCanvas, 0, 0);
  o.drawImage(cv, 0, 0);
  a.download = `petri-${$('specId').textContent}.png`; a.href = out.toDataURL('image/png'); a.click();
};
addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === ' ') { e.preventDefault(); setPaused(!paused); }
  else if (k === 'r') randomize();
  else if (k === 'm') mutate();
  else if (k === 's') rebuild({ newSeed: true });
  else if (k === 't') setTrails(!trails);
  else if (k === '.') stepOnce();
  else if (k === 'g') $('engineBtn').click();
  else if (k === 'e') $('evoBtn').click();
  else if (k === 'l') $('shareBtn').click();
  else if (k === 'p') $('pngBtn').click();
});

let toastTimer;
function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 1600); }

// ------------------------------------------------------------- pointer ----
function toWorld(e) {
  const r = cv.getBoundingClientRect();
  const { cx, cy, R } = dishGeom();
  const px = (e.clientX - r.left) * (cv.width / r.width), py = (e.clientY - r.top) * (cv.height / r.height);
  return { px, py, x: (px - (cx - R)) / (2 * R), y: (py - (cy - R)) / (2 * R), inside: Math.hypot(px - cx, py - cy) < R };
}
cv.addEventListener('contextmenu', (e) => e.preventDefault());
cv.addEventListener('pointerdown', (e) => {
  cv.setPointerCapture(e.pointerId);
  const p = toWorld(e); pointer = p;
  stir = { x: p.x, y: p.y, radius: 0.12, strength: e.button === 2 || e.shiftKey ? -6 : 6 };
});
cv.addEventListener('pointermove', (e) => { const p = toWorld(e); pointer = p; if (stir) { stir.x = p.x; stir.y = p.y; } });
addEventListener('pointerup', () => { stir = null; });
cv.addEventListener('pointerleave', () => { if (!stir) pointer = null; });

// ---------------------------------------------------------------- draw ----
let clearDish = true;
function dishGeom() { const s = cv.width; return { cx: s / 2, cy: s / 2, R: s * 0.44 }; }

function resize() {
  const dpr = Math.min(2, devicePixelRatio || 1);
  const size = Math.round(cv.getBoundingClientRect().width * dpr);
  if (size > 0 && size !== cv.width) { cv.width = cv.height = glCanvas.width = glCanvas.height = size; clearDish = true; }
}
new ResizeObserver(resize).observe(cv);

function draw() {
  const { cx, cy, R } = dishGeom();
  const s = cv.width, dpr = Math.min(2, devicePixelRatio || 1);
  const u = Math.max(s / 900, dpr * 0.9); // unit for line widths / text (readable on phones)
  const small = s < 900; // small dishes saturate faster: tone the glow down
  const ox = cx - R, scale = 2 * R;

  if (engine === 'gpu') {
    ctx.clearRect(0, 0, s, s); // the GL canvas underneath shows through the field of view
    gpu.draw({ cx, cy, R }, { trails, clear: clearDish, size: Math.max(3, 7 * (s / 900)) * (state.n > 12000 ? 0.8 : 1), glow: state.n > 8000 ? 0.1 : 0.16 });
    clearDish = false;
  } else {
    // field of view: fade previous frame for trails, or clear
    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.clip();
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = trails && !clearDish ? 'rgba(4, 5, 7, 0.32)' : '#040507';
    ctx.fillRect(cx - R, cy - R, 2 * R, 2 * R);
    clearDish = false;
    // faint dark-field vignette glow
    ctx.globalCompositeOperation = 'lighter';
    const dot = Math.max(1.4, 2.1 * (s / 900)), glow = dot * (small ? 2.4 : 3.2);
    for (let t = 0; t < byType.length; t++) {
      const idx = byType[t];
      ctx.fillStyle = SPECIES[t] + (small ? '12' : '1c');
      for (const i of idx) ctx.fillRect(ox + world.x[i] * scale - glow / 2, ox + world.y[i] * scale - glow / 2, glow, glow);
      ctx.fillStyle = SPECIES[t];
      for (const i of idx) ctx.fillRect(ox + world.x[i] * scale - dot / 2, ox + world.y[i] * scale - dot / 2, dot, dot);
    }
    ctx.restore();
  }

  // instrument body: everything outside the field of view
  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, s, s); ctx.arc(cx, cy, R, 0, Math.PI * 2, true);
  ctx.fillStyle = '#0b0c0e'; ctx.fill('evenodd');
  ctx.restore();
  // rim, degree ticks, reticle
  ctx.strokeStyle = '#3a3e46'; ctx.lineWidth = 1.5 * u;
  ctx.beginPath(); ctx.arc(cx, cy, R + 1, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, R + 14 * u, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath();
  for (let a = 0; a < 360; a += 5) {
    const r0 = R + 14 * u, r1 = r0 + (a % 30 === 0 ? 12 : a % 10 === 0 ? 7 : 4) * u, t = (a * Math.PI) / 180;
    ctx.moveTo(cx + Math.cos(t) * r0, cy + Math.sin(t) * r0); ctx.lineTo(cx + Math.cos(t) * r1, cy + Math.sin(t) * r1);
  }
  ctx.stroke();
  ctx.fillStyle = '#5c5a54'; ctx.font = `${10 * u}px ${MONO}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let a = 0; a < 360; a += 30) { const t = ((a - 90) * Math.PI) / 180, r = R + 38 * u; ctx.fillText(String(a).padStart(3, '0'), cx + Math.cos(t) * r, cy + Math.sin(t) * r); }
  ctx.strokeStyle = '#e9e4d622'; ctx.lineWidth = u; ctx.beginPath();
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.moveTo(cx + dx * R * 0.9, cy + dy * R * 0.9); ctx.lineTo(cx + dx * R, cy + dy * R); }
  ctx.moveTo(cx - 8 * u, cy); ctx.lineTo(cx + 8 * u, cy); ctx.moveTo(cx, cy - 8 * u); ctx.lineTo(cx, cy + 8 * u);
  ctx.stroke();

  // corner readouts
  const pad = 18 * u, line = 16 * u;
  ctx.font = `600 ${11 * u}px ${MONO}`; ctx.fillStyle = '#8d8a80'; ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillText(`T ${String(world.steps).padStart(6, '0')}`, pad, pad);
  ctx.fillText(paused ? '‖ PAUSED' : '▶ LIVE', pad, pad + line);
  ctx.textAlign = 'right';
  ctx.fillText(`${Math.round(fps)} FPS`, s - pad, pad);
  ctx.fillText(`E ${(kineticEnergy(world) * 1e2).toFixed(2)}`, s - pad, pad + line);
  ctx.textBaseline = 'bottom'; ctx.textAlign = 'left';
  // scale bar = reach (rMax) in screen units
  const bar = world.rMax * scale;
  ctx.fillStyle = '#e9e4d6'; ctx.fillRect(pad, s - pad - 6 * u, bar, 2 * u);
  ctx.fillRect(pad, s - pad - 10 * u, 1.5 * u, 10 * u); ctx.fillRect(pad + bar - 1.5 * u, s - pad - 10 * u, 1.5 * u, 10 * u);
  ctx.fillStyle = '#8d8a80'; ctx.fillText(`REACH ${world.rMax.toFixed(2)}`, pad, s - pad - 14 * u);
  ctx.textAlign = 'right';
  ctx.fillText(`N ${world.n}  ·  K ${world.k}`, s - pad, s - pad);

  // pipette cursor
  if (pointer && pointer.inside) {
    const r = (stir ? stir.radius : 0.12) * scale;
    ctx.strokeStyle = stir ? (stir.strength > 0 ? '#35e0a1' : '#ff5c7a') : '#e9e4d666';
    ctx.lineWidth = 1.5 * u; ctx.setLineDash([4 * u, 5 * u]);
    ctx.beginPath(); ctx.arc(pointer.px, pointer.py, r, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(pointer.px, pointer.py, 2.5 * u, 0, Math.PI * 2); ctx.fillStyle = ctx.strokeStyle; ctx.fill();
  }
}

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = now - last; last = now;
  if (dt > 0 && dt < 1000) fps = fps * 0.92 + (1000 / dt) * 0.08;
  if (!paused) engine === 'gpu' ? gpu.step(stir) : step(world, stir);
  if (engine === 'gpu' && ++frameNo % 15 === 0) gpu.readback(); // keeps HUD energy + evolution seeds current
  draw();
}

// ---------------------------------------------------------------- boot ----
const fromHash = decodeState(location.hash);
Object.assign(state, fromHash);
if (gpuOk) setEngine('gpu', { quiet: true });
if (!location.hash) {
  state.matrix = PRESETS.chase(state.k); state.seed = 20261120;
  if (engine === 'gpu') { state.n = 10000; state.rMax = 0.045; }
}
if (engine === 'cpu') setEngine('cpu', { quiet: true }); // clamps oversized shared worlds
setTrails(true);
resize();
rebuild();
initEvolution();
requestAnimationFrame(frame);

window.__petri = {
  get world() { return world; }, get fps() { return fps; }, get state() { return state; }, get engine() { return engine; },
  get finds() { return finds; }, get evolving() { return evolving; },
};

// ----------------------------------------------------------- evolution ----

function initEvolution() {
  try { finds = JSON.parse(localStorage.getItem(FINDS_KEY) || '[]'); } catch { finds = []; }
  renderGallery();
  $('evoBtn').onclick = () => (evolving ? stopEvolution() : startEvolution());
}

function startEvolution() {
  if (!worker) {
    worker = new Worker(new URL('./evolve-worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type !== 'progress' || !evolving) return;
      finds = m.archive.map((a) => ({ ...a, k: m.k }));
      try { localStorage.setItem(FINDS_KEY, JSON.stringify(finds.map(({ thumb, ...rest }) => ({ ...rest, thumb: shrinkThumb(thumb) })))); } catch {}
      $('evoStat').textContent = `gen ${m.gen} · ${m.tested} tested · best ${finds[0] ? finds[0].score.toFixed(2) : '–'}`;
      renderGallery();
    };
    worker.onerror = (err) => { toast('Evolution worker failed'); console.error(err.message); stopEvolution(); };
  }
  evolving = true;
  // evaluate on a small world with the same neighbour density as the dish
  const n = 700, rMax = Math.min(1 / 3, state.rMax * Math.sqrt(state.n / n));
  worker.postMessage({ type: 'start', k: state.k, seed: (Math.random() * 2 ** 31) >>> 0, start: [Array.from(state.matrix)],
    evalOpts: { n, rMax, friction: state.friction } });
  $('evoBtn').classList.add('on'); $('evoBtn').firstChild.textContent = 'Stop evolving ';
  $('evoStat').textContent = 'breeding generation 1…';
}

function stopEvolution() {
  evolving = false;
  worker?.postMessage({ type: 'stop' });
  $('evoBtn').classList.remove('on'); $('evoBtn').firstChild.textContent = 'Start evolving ';
  if (!finds.length) $('evoStat').textContent = 'idle';
}

const shrinkThumb = (t) => { // keep localStorage small: 350 points is plenty for a 120px thumbnail
  const step = Math.max(1, Math.floor(t.x.length / 350)), o = { x: [], y: [], type: [] };
  for (let i = 0; i < t.x.length; i += step) { o.x.push(+t.x[i].toFixed(3)); o.y.push(+t.y[i].toFixed(3)); o.type.push(t.type[i]); }
  return o;
};

function renderGallery() {
  const g = $('gallery');
  if (!finds.length) { g.innerHTML = '<div class="empty">No finds yet. Start evolving and the best distinct specimens appear here.</div>'; return; }
  g.innerHTML = finds.map((f, i) =>
    `<button class="find ${i === selected ? 'sel' : ''}" data-f="${i}" title="score ${f.score.toFixed(2)} · lumpiness ${f.clump.toFixed(2)} · coordination ${f.coherence.toFixed(2)} · motion ${f.motion.toFixed(2)}">` +
    `<canvas width="120" height="120"></canvas><span>${String(i + 1).padStart(2, '0')} · ${f.score.toFixed(2)}</span></button>`).join('');
  g.querySelectorAll('.find').forEach((b) => {
    const f = finds[+b.dataset.f], c = b.querySelector('canvas').getContext('2d');
    c.fillStyle = '#040507'; c.fillRect(0, 0, 120, 120);
    c.globalCompositeOperation = 'lighter';
    for (let i = 0; i < f.thumb.x.length; i++) { c.fillStyle = SPECIES[f.thumb.type[i]]; c.fillRect(f.thumb.x[i] * 120 - 1, f.thumb.y[i] * 120 - 1, 2, 2); }
    b.onclick = () => loadFind(+b.dataset.f);
  });
}

function loadFind(i) {
  const f = finds[i];
  selected = i;
  if (f.k !== state.k) { state.k = f.k; state.matrix = Float32Array.from(f.matrix); rebuild(); }
  else { setMatrix(Float32Array.from(f.matrix)); rebuild(); }
  renderGallery();
  toast(`Find ${String(i + 1).padStart(2, '0')} under the microscope`);
}

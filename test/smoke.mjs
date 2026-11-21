// Headless end-to-end check for Petri: boots, animates, edits, shares.
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';
import { startServer } from '../serve.mjs';

const { chromium } = createRequire(import.meta.url)('playwright');
const shotDir = process.env.SHOT_DIR;
const server = await startServer(0);
const root = `http://127.0.0.1:${server.address().port}/`;
const base = root + '?engine=cpu';
const browser = await chromium.launch();
let failed = false;
const canvasHash = (page) => page.evaluate(() => {
  const c = document.getElementById('dish'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let h = 0; for (let i = 0; i < d.length; i += 97) h = (h * 31 + d[i]) | 0; return h;
});
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));

  await page.goto(base);
  await page.waitForFunction(() => window.__petri?.world?.steps > 30);
  const h1 = await canvasHash(page);
  await page.waitForTimeout(400);
  assert.notEqual(await canvasHash(page), h1, 'dish animates');
  const fps = await page.evaluate(() => window.__petri.fps);
  console.log(`✓ boots and animates (${fps.toFixed(0)} fps headless, ${await page.evaluate(() => window.__petri.world.n)} particles)`);
  assert.match(page.url(), /#s=\d+&k=6&n=2400&m=[a-u]{36}/);
  console.log('✓ URL hash encodes the specimen');

  // matrix editing: left-click +0.2, right-click -0.2
  const v0 = await page.evaluate(() => window.__petri.state.matrix[0]);
  await page.click('.mc[data-i="0"]');
  const v1 = await page.evaluate(() => window.__petri.state.matrix[0]);
  assert.ok(Math.abs(v1 - Math.min(1, v0 + 0.2)) < 1e-6);
  await page.click('.mc[data-i="1"]', { button: 'right' });
  assert.ok(page.url().includes('m='));
  console.log('✓ matrix cells edit and sync to the URL');

  // sharing: opening the URL reproduces the exact matrix + seed
  const shared = page.url();
  const want = await page.evaluate(() => [Array.from(window.__petri.state.matrix), window.__petri.state.seed]);
  const p2 = await browser.newPage();
  await p2.goto(shared);
  await p2.waitForFunction(() => window.__petri?.world);
  assert.deepEqual(await p2.evaluate(() => [Array.from(window.__petri.state.matrix), window.__petri.state.seed]), want);
  await p2.close();
  console.log('✓ share link reproduces the specimen');

  // controls
  await page.click('[data-p="snakes"]');
  assert.equal(await page.evaluate(() => window.__petri.state.matrix[0]), 1);
  await page.fill('#kIn', '4'); await page.dispatchEvent('#kIn', 'input');
  assert.equal(await page.evaluate(() => window.__petri.world.k), 4);
  await page.keyboard.press('Space');
  const s1 = await page.evaluate(() => window.__petri.world.steps);
  await page.waitForTimeout(200);
  assert.equal(await page.evaluate(() => window.__petri.world.steps), s1, 'paused');
  await page.keyboard.press('Space');
  await page.keyboard.press('r');
  console.log('✓ presets, species slider, pause, randomize');

  // pipette drag
  const box = await page.locator('#dish').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down(); await page.waitForTimeout(300); await page.mouse.up();
  console.log('✓ pipette drag');

  // evolution: breed in a worker, gallery fills, a find loads into the dish, finds persist
  await page.goto(base); await page.waitForFunction(() => window.__petri?.world);
  await page.click('#evoBtn');
  await page.waitForFunction(() => window.__petri.finds.length >= 3, null, { timeout: 60000 });
  assert.match(await page.textContent('#evoStat'), /gen \d+ · \d+ tested · best \d\.\d\d/);
  assert.ok(await page.locator('.find canvas').count() >= 3);
  await page.click('.find[data-f="0"]');
  const same = await page.evaluate(() => Array.from(window.__petri.state.matrix).every((v, i) => Math.abs(v - window.__petri.finds[0].matrix[i]) < 1e-6));
  assert.ok(same, 'clicking a find loads its matrix');
  await page.click('#evoBtn');
  assert.equal(await page.evaluate(() => window.__petri.evolving), false);
  await page.reload(); await page.waitForFunction(() => window.__petri?.world);
  assert.ok(await page.locator('.find').count() >= 3, 'finds persist across reloads');
  console.log('✓ evolution: worker breeds, gallery fills, finds load and persist');

  // GPU engine (WebGL2): boots by default when supported, renders, switches engines
  const gpuUrl = root + '#s=5&k=4&n=1500&m=uqkkkuqkkkuqqkku&r=0.08&f=0.04';
  await page.goto(gpuUrl); await page.waitForFunction(() => window.__petri?.world?.steps > 5, null, { timeout: 60000 });
  if (await page.evaluate(() => window.__petri.engine) === 'gpu') {
    const lit = await page.evaluate(() => {
      const g = document.getElementById('gl'), t = document.createElement('canvas'); t.width = g.width; t.height = g.height;
      const c = t.getContext('2d'); c.drawImage(g, 0, 0);
      const d = c.getImageData(0, 0, t.width, t.height).data; let n = 0;
      for (let i = 0; i < d.length; i += 4) if (d[i] + d[i + 1] + d[i + 2] > 120) n++; return n;
    });
    assert.ok(lit > 500, `GPU canvas draws particles (${lit} lit pixels)`);
    const dl = page.waitForEvent('download'); await page.keyboard.press('p');
    assert.match((await dl).suggestedFilename(), /^petri-.+\.png$/);
    await page.keyboard.press('g');
    assert.equal(await page.evaluate(() => window.__petri.engine), 'cpu');
    const s0 = await page.evaluate(() => window.__petri.world.steps);
    await page.waitForFunction((s) => window.__petri.world.steps > s + 5, s0);
    await page.keyboard.press('g');
    assert.equal(await page.evaluate(() => window.__petri.engine), 'gpu');
    console.log(`✓ GPU engine renders (${lit} lit px), photo exports, engine hot-swaps both ways`);
  } else console.log('- GPU engine unavailable here; skipped GPU UI checks');

  if (shotDir) {
    await page.setViewportSize({ width: 1280, height: 1000 });
    await page.goto(root + '#s=20261120&k=6&n=4000&m=qokkkerqokkkkerqokkkkerqokkkkerqoakkkeq&r=0.06&f=0.04');
    await page.waitForTimeout(9000);
    await page.screenshot({ path: `${shotDir}/petri-gpu.png` });
  }
  assert.deepEqual(errors, []);
  console.log('✓ no console errors');
} catch (e) { failed = true; console.error(e); }
finally { await browser.close(); server.close(); }
process.exit(failed ? 1 : 0);

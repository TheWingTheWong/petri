// Takes the README screenshot (docs/screenshot.png) with Playwright: `node scripts/screenshot.mjs`
import { createRequire } from "node:module";
import { startServer } from "../serve.mjs";

const { chromium } = createRequire(import.meta.url)("playwright");
const server = await startServer(0);
const base = `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch();
const out = new URL("../docs/screenshot.png", import.meta.url).pathname;
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 2 });
  await page.goto(base + "#s=7&k=6&n=6000");
  await page.waitForFunction(() => window.__petri?.world?.steps > 5, null, { timeout: 60000 });
  await page.waitForTimeout(9000); // let structures form
  await page.screenshot({ path: out });
  console.log("wrote", out);
} finally { await browser.close(); server.close(); }

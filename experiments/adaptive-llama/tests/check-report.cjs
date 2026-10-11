// Optional UI verification; reuse an installed Playwright rather than installing a second copy.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const fs = require('node:fs');
(async () => {
  const report = path.resolve(process.argv[2] || 'experiments/adaptive-llama/examples/seed-42.html');
  const browser = await chromium.launch({ executablePath: process.env.CHROME_BIN || '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1180, height: 900 } });
    const errors = [], network = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => { if (/^https?:/.test(r.url())) network.push(r.url()); });
    await page.goto(pathToFileURL(report).href);
    const data = await page.locator('#report-data').textContent().then(JSON.parse);
    const assert = (ok, message) => { if (!ok) throw new Error(message); };
    assert(await page.locator('[data-phase]').count() === 4, 'Four phase controls required');
    if (data.growth_proposed ?? true) {
      await page.locator('[data-phase=growth]').click();
      assert(await page.locator('#graph .new').textContent() === 'Réteg 5', 'Growth must display a fifth layer');
    } else {
      assert(await page.locator('[data-phase=growth]').isDisabled(), 'Skipped growth must not expose an invented candidate');
    }
    await page.locator('[data-phase=control]').click();
    assert(await page.locator('#graph .new').count() === 0, 'Control must retain four layers');
    await page.locator('[data-phase=selected]').click();
    assert(await page.locator('#graph .new').count() === Number(data.growth_accepted), 'Selected graph differs from measured decision');
    assert(await page.locator('#examples tr').count() === data.examples.length, 'Measured examples missing');
    assert(await page.locator('#accuracy').textContent() === (100 * data.test.exact_accuracy).toFixed(1) + '%', 'Metric differs from run data');
    if (data.validation_history) {
      await page.locator('#history-step').fill('0');
      await page.locator('#history-step').dispatchEvent('input');
      assert((await page.locator('#history-value').textContent()).startsWith('0 tanítási lépés'), 'History slider must show recorded initial evidence');
    }
    await page.setViewportSize({ width: 600, height: 850 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Narrow viewport overflow');
    assert(errors.length === 0, errors.join('\n'));
    assert(network.length === 0, 'Offline report contacted a server');
    console.log('PASS:', path.basename(report), 'phase graphs, decision, measured metrics, mobile layout, no errors/network');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });

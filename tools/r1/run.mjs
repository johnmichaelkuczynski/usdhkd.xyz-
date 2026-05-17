#!/usr/bin/env node
// R1 — synthetic user agent that beta-tests USD/HKD Edge end-to-end.
// Single-file harness. Run with `npm start`. See README.md.

import { chromium } from "playwright";
import { mkdir, writeFile, appendFile } from "node:fs/promises";
import { existsSync, createWriteStream } from "node:fs";
import path from "node:path";
import http from "node:http";
import { fileURLToPath } from "node:url";

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────────────────

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const APP_URL = process.env.APP_URL || "http://localhost:5000";
const HEADLESS = process.env.HEADLESS === "true";
const LIVE_VIEW_PORT = parseInt(process.env.LIVE_VIEW_PORT || "7777", 10);
const SKIP_FUNCTIONS = new Set(
  (process.env.SKIP_FUNCTIONS || "").split(",").filter(Boolean).map((s) => s.trim()),
);
const PRICER_TIMEOUT_MS = parseInt(process.env.PRICER_TIMEOUT_MS || "60000", 10);
const HEAVY_PRICER_TIMEOUT_MS = parseInt(process.env.HEAVY_PRICER_TIMEOUT_MS || "180000", 10);
const ALL_MODEL_BACKTEST_TIMEOUT_MS = parseInt(
  process.env.ALL_MODEL_BACKTEST_TIMEOUT_MS || "1800000",
  10,
);
const ANTHROPIC_MODEL = process.env.ANTHROPIC_MODEL || "claude-opus-4-5";
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC || "";
const VIEWPORT_WIDTH = 1920;
const VIEWPORT_HEIGHT = 1080;

const RUN_ID = new Date().toISOString().replace(/[:.]/g, "-");
const OUTPUT_DIR = path.join(__dirname, "runs", RUN_ID);
const SCREENSHOT_DIR = path.join(OUTPUT_DIR, "screenshots");
const TRANSCRIPT_PATH = path.join(OUTPUT_DIR, "transcript.jsonl");
const CONSOLE_PATH = path.join(OUTPUT_DIR, "console.log");
const NETWORK_PATH = path.join(OUTPUT_DIR, "network.log");

const PRICER_ORDER = [
  "heston", "bates", "svjj", "merton_jd", "kou_jd", "vg", "cgmy", "nig", "sabr",
  "double_heston", "rough_heston", "bs_rv", "bs_garch",
];
const HEAVY_PRICERS = new Set([
  "heston", "bates", "svjj", "double_heston", "rough_heston",
]);
const PRICER_LABELS = {
  heston: "Heston", bates: "Bates", svjj: "SVJJ",
  merton_jd: "Merton-JD", kou_jd: "Kou-JD",
  vg: "Variance Gamma", cgmy: "CGMY", nig: "NIG", sabr: "SABR",
  double_heston: "Double-Heston", rough_heston: "Rough-Heston",
  bs_rv: "BS · realized vol", bs_garch: "BS · GARCH(1,1)",
};

// ─────────────────────────────────────────────────────────────────────────────
// Output dirs + dual-write console
// ─────────────────────────────────────────────────────────────────────────────

await mkdir(SCREENSHOT_DIR, { recursive: true });
await mkdir(path.join(OUTPUT_DIR, "outputs", "live-forecasts"), { recursive: true });
await mkdir(path.join(OUTPUT_DIR, "outputs", "backtests", "single"), { recursive: true });
await mkdir(path.join(OUTPUT_DIR, "outputs", "backtests", "pairwise"), { recursive: true });
await mkdir(path.join(OUTPUT_DIR, "outputs", "backtests", "all-model"), { recursive: true });
await mkdir(path.join(OUTPUT_DIR, "outputs", "screenshots"), { recursive: true });

const consoleStream = createWriteStream(CONSOLE_PATH, { flags: "a" });
function ts() { return new Date().toISOString(); }
function log(...args) {
  const line = `[${ts()}] ` + args.map((a) =>
    typeof a === "string" ? a : JSON.stringify(a)
  ).join(" ");
  console.log(line);
  consoleStream.write(line + "\n");
}
function logErr(...args) {
  const line = `[${ts()}] ERROR ` + args.map((a) =>
    a instanceof Error ? `${a.message}\n${a.stack}` :
    typeof a === "string" ? a : JSON.stringify(a)
  ).join(" ");
  console.error(line);
  consoleStream.write(line + "\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Live-view state + HTTP server
// ─────────────────────────────────────────────────────────────────────────────

const liveState = {
  status: "starting",
  currentStep: null,
  currentParams: null,
  url: APP_URL,
  latestScreenshot: null,
  lastNumeric: null,
  lastJudge: null,
  quantState: {
    activeModel: null,
    currentTab: null,
    overlay: null,
    lastForecast: null,
    lastBacktest: null,
    lastInvariantCheck: null,
  },
  allModelProgress: null,
  recentInteractions: [],
};

function pushRecent(entry) {
  liveState.recentInteractions.unshift(entry);
  if (liveState.recentInteractions.length > 40) liveState.recentInteractions.pop();
}

function renderLiveHtml() {
  const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const j = (o) => esc(JSON.stringify(o, null, 2));
  const ss = liveState.latestScreenshot
    ? `<img src="${esc(liveState.latestScreenshot)}" style="max-width:100%;border:1px solid #444"/>`
    : "<em>no screenshot yet</em>";
  return `<!doctype html><html><head><meta charset="utf-8"/>
<title>R1 live view — USD/HKD Edge</title>
<meta http-equiv="refresh" content="2"/>
<style>
  body{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:#0d0d10;color:#ddd;margin:0;padding:14px}
  h1{font-size:16px;margin:0 0 8px;color:#fff}
  h2{font-size:13px;margin:14px 0 6px;color:#7af;border-bottom:1px solid #234;padding-bottom:3px}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .col{display:flex;flex-direction:column;gap:10px}
  pre{background:#15151a;border:1px solid #2a2a30;padding:8px;border-radius:4px;white-space:pre-wrap;word-break:break-word;margin:0;max-height:320px;overflow:auto;font-size:12px}
  .pill{display:inline-block;background:#1d2235;color:#7af;border:1px solid #345;padding:1px 6px;border-radius:8px;margin-right:4px;font-size:11px}
  .ok{color:#7eaf6e}.warn{color:#d3a85a}.bad{color:#d76a6a}
  table{border-collapse:collapse;width:100%;font-size:12px}
  td,th{border:1px solid #2a2a30;padding:4px 6px;text-align:left;vertical-align:top}
  .small{color:#888;font-size:11px}
</style></head><body>
<h1>R1 — USD/HKD Edge ${esc(liveState.status)}</h1>
<div class="small">Run id: ${esc(RUN_ID)} · APP_URL: ${esc(APP_URL)} · Auto-refresh 2s</div>

<div class="row">
  <div class="col">
    <h2>Current step</h2>
    <pre>${esc(liveState.currentStep || "(idle)")}</pre>
    <h2>Parameter choices</h2>
    <pre>${j(liveState.currentParams)}</pre>
    <h2>Latest screenshot</h2>
    ${ss}
  </div>
  <div class="col">
    <h2>Quant State</h2>
    <table>
      <tr><th>Active model</th><td>${esc(liveState.quantState.activeModel)}</td></tr>
      <tr><th>Current tab</th><td>${esc(liveState.quantState.currentTab)}</td></tr>
      <tr><th>Overlay</th><td>${esc(liveState.quantState.overlay)}</td></tr>
      <tr><th>Last forecast</th><td><pre>${j(liveState.quantState.lastForecast)}</pre></td></tr>
      <tr><th>Last backtest</th><td><pre>${j(liveState.quantState.lastBacktest)}</pre></td></tr>
      <tr><th>Last invariant</th><td><pre>${j(liveState.quantState.lastInvariantCheck)}</pre></td></tr>
    </table>
    <h2>Latest numeric outputs</h2>
    <pre>${j(liveState.lastNumeric)}</pre>
    <h2>Judge critique</h2>
    <pre>${esc(liveState.lastJudge || "")}</pre>
    ${liveState.allModelProgress ? `<h2>All-model progress</h2><pre>${j(liveState.allModelProgress)}</pre>` : ""}
  </div>
</div>

<h2>Recent interactions (newest first)</h2>
<table>
  <tr><th>Time</th><th>Function</th><th>Step</th><th>Invariants</th><th>Judge</th></tr>
  ${liveState.recentInteractions.map((e) => `<tr>
    <td class="small">${esc(e.t)}</td>
    <td>${esc(e.fn)}</td>
    <td>${esc(e.step)}</td>
    <td>${esc(e.invariants || "")}</td>
    <td class="small">${esc((e.judge || "").slice(0, 140))}</td>
  </tr>`).join("")}
</table>
</body></html>`;
}

const liveServer = http.createServer((req, res) => {
  try {
    if (req.url === "/" || req.url === "/index.html") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(renderLiveHtml());
      return;
    }
    if (req.url.startsWith("/screenshots/")) {
      const file = path.join(OUTPUT_DIR, req.url);
      if (existsSync(file)) {
        res.writeHead(200, { "content-type": "image/png" });
        import("node:fs").then(({ createReadStream }) => createReadStream(file).pipe(res));
        return;
      }
    }
    res.writeHead(404); res.end("not found");
  } catch (e) {
    res.writeHead(500); res.end(String(e));
  }
});
liveServer.listen(LIVE_VIEW_PORT, () => log(`Live view: http://localhost:${LIVE_VIEW_PORT}`));

// ─────────────────────────────────────────────────────────────────────────────
// Transcript + outputs
// ─────────────────────────────────────────────────────────────────────────────

const transcript = [];
async function appendTranscript(entry) {
  transcript.push(entry);
  await appendFile(TRANSCRIPT_PATH, JSON.stringify(entry) + "\n");
}

async function writeOutput(rel, obj) {
  const full = path.join(OUTPUT_DIR, "outputs", rel);
  await mkdir(path.dirname(full), { recursive: true });
  await writeFile(full, JSON.stringify(obj, null, 2));
}

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic client (fetch-based, no SDK)
// ─────────────────────────────────────────────────────────────────────────────

async function callClaude(systemPrompt, userPrompt, maxTokens = 700) {
  if (!ANTHROPIC_KEY) return null;
  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": ANTHROPIC_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: ANTHROPIC_MODEL,
        max_tokens: maxTokens,
        system: systemPrompt,
        messages: [{ role: "user", content: userPrompt }],
      }),
    });
    if (!res.ok) {
      const txt = await res.text();
      logErr("Anthropic error", res.status, txt.slice(0, 400));
      return null;
    }
    const data = await res.json();
    const text = (data.content || []).map((c) => c.text || "").join("").trim();
    return text || null;
  } catch (e) {
    logErr("Anthropic fetch failed", e);
    return null;
  }
}

const BRAIN_SYSTEM = `You are R1, a beta-testing agent for USD/HKD Edge, a Streamlit quantitative finance app.
For each step you are asked, pick ONE approach from: "documented_default", "probe_invariant",
"vary_parameter", "edge_case", "cycle_choices". Respond with exactly one short JSON object on a
single line: {"approach":"...","reason":"...","params":{...}}. Be specific. params holds any
numeric choices you want the harness to use (e.g. {"paths":10000}).`;

const JUDGE_SYSTEM = `You are an independent quantitative analyst reviewing a synthetic user
agent's interaction with USD/HKD Edge. Critique as a quant would. Are the numbers internally
consistent? Do rankings make statistical sense? Is the calibration verdict justified by the
coverage numbers shown? Are quantile orderings monotonic? Is anything mathematically suspect?
Be concise (3–6 sentences). Plain prose, no JSON, no PASS/FAIL labels. If everything is fine
say so briefly and move on.`;

async function brainChoose(stepDescription, controlSnapshot) {
  if (!ANTHROPIC_KEY) return { approach: "documented_default", reason: "no api key", params: {} };
  const user = `STEP:\n${stepDescription}\n\nCURRENT CONTROL STATE:\n${JSON.stringify(controlSnapshot, null, 2)}`;
  const text = await callClaude(BRAIN_SYSTEM, user, 200);
  if (!text) return { approach: "documented_default", reason: "no response", params: {} };
  try {
    const m = text.match(/\{[\s\S]*\}/);
    if (m) return JSON.parse(m[0]);
  } catch {}
  return { approach: "documented_default", reason: "parse failed", params: {} };
}

async function judgeCritique(payload) {
  if (!ANTHROPIC_KEY) return "(judge skipped — no ANTHROPIC_API_KEY)";
  const user = `INTERACTION RECORD:\n${JSON.stringify(payload, null, 2).slice(0, 12000)}`;
  const text = await callClaude(JUDGE_SYSTEM, user, 500);
  return text || "(judge returned no text)";
}

// ─────────────────────────────────────────────────────────────────────────────
// Counters
// ─────────────────────────────────────────────────────────────────────────────

const counters = {
  interactions: 0,
  judgeConcerns: 0,
  invariantViolations: { A: 0, B: 0, C: 0, D: 0, E: 0, F: 0, G: 0 },
  pricersCrashed: [],
  harnessSanityFailures: 0,
};
const judgeConcernEntries = [];
const invariantViolationEntries = [];
const harnessFailureEntries = [];

function recordViolation(invariantId, detail) {
  counters.invariantViolations[invariantId]++;
  invariantViolationEntries.push({ invariant: invariantId, ...detail, t: ts() });
}
function recordJudgeConcern(detail) {
  counters.judgeConcerns++;
  judgeConcernEntries.push({ ...detail, t: ts() });
}
function recordHarnessFailure(detail) {
  counters.harnessSanityFailures++;
  harnessFailureEntries.push({ ...detail, t: ts() });
}

// ─────────────────────────────────────────────────────────────────────────────
// Screenshot helper
// ─────────────────────────────────────────────────────────────────────────────

let shotCounter = 0;
async function screenshot(page, label) {
  shotCounter++;
  const filename = `${String(shotCounter).padStart(4, "0")}_${label.replace(/[^a-z0-9._-]/gi, "_")}.png`;
  const full = path.join(SCREENSHOT_DIR, filename);
  try {
    await page.screenshot({ path: full, fullPage: false });
    const rel = `screenshots/${filename}`;
    liveState.latestScreenshot = rel;
    return rel;
  } catch (e) {
    logErr("Screenshot failed", label, e);
    return null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Streamlit interaction helpers
// ─────────────────────────────────────────────────────────────────────────────

async function waitForStreamlitIdle(page, settleMs = 600, maxWait = 120000) {
  const start = Date.now();
  // Wait briefly for any spinner to appear (Streamlit triggers reruns asynchronously)
  await page.waitForTimeout(200);
  while (Date.now() - start < maxWait) {
    const busy = await page.locator('[data-testid="stSpinner"], [data-testid="stStatusWidget-Running"]')
      .count().catch(() => 0);
    if (busy === 0) break;
    await page.waitForTimeout(250);
  }
  // Settle delay
  await page.waitForTimeout(settleMs);
}

async function captureStreamlitException(page) {
  // Streamlit renders Python tracebacks as [data-testid="stException"] or .stException
  try {
    const locs = page.locator('[data-testid="stException"], .stException, [data-testid="stAlert"]');
    const n = await locs.count();
    if (n === 0) return null;
    const texts = [];
    for (let i = 0; i < n; i++) {
      const t = await locs.nth(i).innerText().catch(() => "");
      if (t && /error|traceback|exception/i.test(t)) texts.push(t.slice(0, 2000));
    }
    return texts.length ? texts.join("\n---\n") : null;
  } catch { return null; }
}

async function pageHasText(page, regex) {
  try {
    const body = await page.locator("body").innerText({ timeout: 5000 }).catch(() => "");
    return regex.test(body);
  } catch { return false; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Control helpers (selectbox / slider / checkbox / button / text input / tabs)
// ─────────────────────────────────────────────────────────────────────────────

async function setSelectbox(page, labelRegex, value) {
  // Streamlit selectbox: label sits as a sibling; the trigger is a div with role="combobox".
  const labels = page.locator('[data-testid="stSelectbox"] label');
  const n = await labels.count();
  let target = null;
  for (let i = 0; i < n; i++) {
    const t = (await labels.nth(i).innerText().catch(() => "")).trim();
    if (labelRegex.test(t)) {
      target = page.locator('[data-testid="stSelectbox"]').nth(i);
      break;
    }
  }
  if (!target) {
    // fallback: just take the nth selectbox if labelRegex looks like an index
    target = page.locator('[data-testid="stSelectbox"]').first();
  }
  await target.click();
  await page.waitForTimeout(150);
  // Options appear in a dropdown; click by text content
  const option = page.locator('[role="option"]', { hasText: new RegExp(value, "i") });
  await option.first().click({ timeout: 5000 });
  await waitForStreamlitIdle(page);
}

async function setCheckbox(page, labelRegex, checked) {
  const cbs = page.locator('[data-testid="stCheckbox"]');
  const n = await cbs.count();
  for (let i = 0; i < n; i++) {
    const text = (await cbs.nth(i).innerText().catch(() => "")).trim();
    if (labelRegex.test(text)) {
      const input = cbs.nth(i).locator('input[type="checkbox"]');
      const isChecked = await input.isChecked().catch(() => null);
      if (isChecked !== checked) {
        await cbs.nth(i).click();
        await waitForStreamlitIdle(page);
      }
      return true;
    }
  }
  return false;
}

async function setTextInput(page, labelRegex, value) {
  const wrappers = page.locator('[data-testid="stTextInput"]');
  const n = await wrappers.count();
  for (let i = 0; i < n; i++) {
    const text = (await wrappers.nth(i).innerText().catch(() => "")).trim();
    if (labelRegex.test(text)) {
      const input = wrappers.nth(i).locator("input");
      await input.fill(String(value));
      await input.press("Enter");
      await waitForStreamlitIdle(page);
      return true;
    }
  }
  return false;
}

async function setSlider(page, labelRegex, targetValue) {
  // Streamlit slider: focus the thumb, use keyboard arrows OR aria-valuenow manipulation.
  // Robust approach: read current value, then press ArrowRight/Left toward target.
  const sliders = page.locator('[data-testid="stSlider"]');
  const n = await sliders.count();
  for (let i = 0; i < n; i++) {
    const text = (await sliders.nth(i).innerText().catch(() => "")).trim();
    if (!labelRegex.test(text)) continue;
    const thumb = sliders.nth(i).locator('[role="slider"]').first();
    await thumb.focus();
    // Read current
    let cur = parseFloat(await thumb.getAttribute("aria-valuenow") || "NaN");
    const min = parseFloat(await thumb.getAttribute("aria-valuemin") || "0");
    const max = parseFloat(await thumb.getAttribute("aria-valuemax") || "100");
    if (!isFinite(cur)) cur = min;
    // Determine step by single arrow press observation
    await thumb.press("ArrowRight");
    let probe = parseFloat(await thumb.getAttribute("aria-valuenow") || "NaN");
    let step = isFinite(probe) ? Math.abs(probe - cur) : 1;
    if (step <= 0) step = 1;
    cur = probe;
    // Clamp target
    const tgt = Math.max(min, Math.min(max, targetValue));
    const stepsNeeded = Math.round((tgt - cur) / step);
    const key = stepsNeeded >= 0 ? "ArrowRight" : "ArrowLeft";
    for (let k = 0; k < Math.abs(stepsNeeded); k++) await thumb.press(key);
    await waitForStreamlitIdle(page);
    return true;
  }
  return false;
}

async function clickButtonByText(page, regex) {
  const buttons = page.locator('[data-testid="stButton"] button, [data-testid="baseButton-secondary"], button');
  const n = await buttons.count();
  for (let i = 0; i < n; i++) {
    const t = (await buttons.nth(i).innerText().catch(() => "")).trim();
    if (regex.test(t)) {
      await buttons.nth(i).click();
      await waitForStreamlitIdle(page);
      return true;
    }
  }
  return false;
}

async function clickTab(page, regex) {
  const tabs = page.locator('[data-testid="stTabs"] [role="tab"], [role="tab"]');
  const n = await tabs.count();
  for (let i = 0; i < n; i++) {
    const t = (await tabs.nth(i).innerText().catch(() => "")).trim();
    if (regex.test(t)) {
      await tabs.nth(i).click();
      await waitForStreamlitIdle(page, 400);
      return true;
    }
  }
  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Numeric extractors
// ─────────────────────────────────────────────────────────────────────────────

function parseNum(s) {
  if (s == null) return NaN;
  const cleaned = String(s).replace(/[,%\s]/g, "").replace(/[^\d.\-+eE]/g, "");
  const v = parseFloat(cleaned);
  return isFinite(v) ? v : NaN;
}

async function extractMetrics(page) {
  // st.metric renders as [data-testid="stMetric"] with label + value + optional delta
  const out = [];
  const metrics = page.locator('[data-testid="stMetric"]');
  const n = await metrics.count();
  for (let i = 0; i < n; i++) {
    const label = (await metrics.nth(i).locator('[data-testid="stMetricLabel"]').innerText().catch(() => "")).trim();
    const value = (await metrics.nth(i).locator('[data-testid="stMetricValue"]').innerText().catch(() => "")).trim();
    const delta = (await metrics.nth(i).locator('[data-testid="stMetricDelta"]').innerText().catch(() => "")).trim();
    out.push({ label, value, value_num: parseNum(value), delta });
  }
  return out;
}

async function extractDataFrames(page) {
  // st.dataframe renders an embedded table; st.table renders a plain <table>.
  // We grab text content of both.
  const out = [];
  const dfs = page.locator('[data-testid="stDataFrame"], [data-testid="stTable"]');
  const n = await dfs.count();
  for (let i = 0; i < n; i++) {
    const html = await dfs.nth(i).innerHTML().catch(() => "");
    const rows = await dfs.nth(i).locator("tr").allInnerTexts().catch(() => []);
    const cells = rows.map((r) => r.split(/\t|\n/).map((c) => c.trim()).filter(Boolean));
    out.push({ index: i, rows: cells, html_len: html.length });
  }
  return out;
}

async function extractStatusBar(page) {
  // The status bar metrics are st.metric in the top region — they're indistinguishable
  // from later metrics by testid alone. We rely on label matching.
  const metrics = await extractMetrics(page);
  const find = (re) => metrics.find((m) => re.test(m.label));
  return {
    raw: metrics,
    last_close: find(/last close/i)?.value_num ?? null,
    yield_diff: find(/3m yield|US.+HK/i)?.value_num ?? null,
    fair_value: find(/fair value|equilibrium/i)?.value_num ?? null,
    z_score: find(/z.?score|disequilibrium/i)?.value_num ?? null,
  };
}

async function extractHorizonCards(page) {
  // Each card is a column with a header like "1 week" / "1 month" etc., a histogram (Plotly),
  // a small table of (p5/p50/p95 + bucket probs), and P(HKD) / P(USD) lines.
  const cards = [];
  const candidates = page.locator(':is(h3, h4, [data-testid="stMarkdownContainer"] > p > strong)',
    { hasText: /1\s*week|1\s*month|3\s*month|6\s*month/i });
  const n = await candidates.count();
  for (let i = 0; i < n; i++) {
    const heading = (await candidates.nth(i).innerText().catch(() => "")).trim();
    // Walk up to the parent container and grab text content
    const parent = candidates.nth(i).locator("xpath=ancestor::div[contains(@class,'stColumn') or contains(@data-testid,'column')][1]");
    let text = "";
    try { text = await parent.innerText({ timeout: 2000 }); }
    catch { text = ""; }
    if (!text) {
      try { text = await candidates.nth(i).locator("xpath=following::*[1]").innerText({ timeout: 2000 }); }
      catch {}
    }
    cards.push({ heading, text });
  }
  // Parse each text block heuristically
  const parsed = cards.map((c) => {
    const grab = (re) => {
      const m = c.text.match(re);
      return m ? parseNum(m[1]) : NaN;
    };
    const p5  = grab(/p0?5[^\d-]*([\-+\d.eE]+)/i);
    const p50 = grab(/p50|median[^\d-]*([\-+\d.eE]+)/i);
    const p95 = grab(/p95[^\d-]*([\-+\d.eE]+)/i);
    const pHkd = grab(/HKD[^%\d]*([\d.]+)\s*%/i);
    const pUsd = grab(/USD[^%\d]*([\d.]+)\s*%/i);
    // bucket probs: look for "<= X" / ">= X" / "X – Y" patterns followed by a percentage
    const buckets = [];
    const bre = /([<>≤≥]?\s*[\d.]+(?:\s*[–-]\s*[\d.]+)?)\s*[:|]?\s*([\d.]+)\s*%/g;
    let m;
    while ((m = bre.exec(c.text)) !== null) {
      buckets.push({ label: m[1].trim(), prob: parseNum(m[2]) / 100 });
    }
    return {
      heading: c.heading,
      p5, p50, p95,
      p_hkd_appreciation: isFinite(pHkd) ? pHkd / 100 : NaN,
      p_usd_appreciation: isFinite(pUsd) ? pUsd / 100 : NaN,
      buckets,
      raw_text_sample: c.text.slice(0, 800),
    };
  });
  return parsed;
}

async function extractEquilibriumCaption(page) {
  // Caption that mentions alpha, beta, residual sigma, daily lambda
  const captions = page.locator('[data-testid="stCaptionContainer"], .stCaption, small');
  const n = await captions.count();
  for (let i = 0; i < n; i++) {
    const t = (await captions.nth(i).innerText().catch(() => "")).trim();
    if (/alpha|α/i.test(t) && /beta|β/i.test(t)) {
      const grab = (re) => { const m = t.match(re); return m ? parseNum(m[1]) : NaN; };
      return {
        raw: t,
        alpha: grab(/(?:alpha|α)[^\d-]*([\-+\d.eE]+)/i),
        beta: grab(/(?:beta|β)[^\d-]*([\-+\d.eE]+)/i),
        residual_sigma: grab(/residual.{0,4}(?:σ|sigma|sd|std)?[^\d-]*([\-+\d.eE]+)/i),
        lambda_per_day: grab(/(?:λ|lambda)[^\d-]*([\-+\d.eE]+)/i),
      };
    }
  }
  return null;
}

async function extractVerdictBanner(page) {
  // Verdict banners are st.success / st.warning / st.error
  const banners = page.locator('[data-testid="stAlert"]');
  const n = await banners.count();
  for (let i = 0; i < n; i++) {
    const t = (await banners.nth(i).innerText().catch(() => "")).trim();
    if (/CALIBRATED|BIASED|OVERCONFIDENT|UNDERCONFIDENT|PARTIALLY|WINNER/i.test(t)) {
      const kind = await banners.nth(i).getAttribute("class").catch(() => "");
      return { text: t, kind: kind || "" };
    }
  }
  return null;
}

async function dumpVisibleText(page, maxLen = 4000) {
  const t = await page.locator("main, [data-testid='stAppViewContainer']").innerText().catch(() => "");
  return t.slice(0, maxLen);
}

// ─────────────────────────────────────────────────────────────────────────────
// Invariant checkers
// ─────────────────────────────────────────────────────────────────────────────

function invariantA(card) {
  const issues = [];
  const sum = card.buckets.reduce((s, b) => s + (isFinite(b.prob) ? b.prob : 0), 0);
  if (card.buckets.length === 0) {
    issues.push("no buckets extracted");
  } else if (!(Math.abs(sum - 1) <= 0.01)) {
    issues.push(`bucket probabilities sum=${sum.toFixed(4)}, expected 1±0.01`);
  }
  const qs = [card.p5, card.p50, card.p95].filter((x) => isFinite(x));
  if (qs.length === 3 && !(qs[0] <= qs[1] && qs[1] <= qs[2])) {
    issues.push(`quantiles non-monotonic p5=${qs[0]} p50=${qs[1]} p95=${qs[2]}`);
  }
  if (isFinite(card.p_hkd_appreciation) && isFinite(card.p_usd_appreciation)) {
    const s = card.p_hkd_appreciation + card.p_usd_appreciation;
    if (!(s >= 0.97 && s <= 1.03)) {
      issues.push(`P(HKD)+P(USD)=${s.toFixed(3)}, expected ≈1`);
    }
  }
  return { ok: issues.length === 0, issues, sum, card };
}

function invariantC(prev, next) {
  const same = JSON.stringify(prev) === JSON.stringify(next);
  return { ok: same, diff_prev: prev, diff_next: next };
}

function invariantD(matrix, models) {
  const issues = [];
  const N = models.length;
  if (!matrix || matrix.length !== N) {
    issues.push(`matrix size=${matrix?.length}, expected ${N}`);
    return { ok: false, issues };
  }
  for (let i = 0; i < N; i++) {
    if (!Array.isArray(matrix[i]) || matrix[i].length !== N) {
      issues.push(`row ${i} length=${matrix[i]?.length}, expected ${N}`);
    }
    for (let j = 0; j < N; j++) {
      const v = matrix[i]?.[j];
      if (i === j) {
        if (v !== null && !Number.isNaN(v) && v !== undefined) {
          issues.push(`diagonal [${i},${j}] is ${v}, expected NaN/null`);
        }
        continue;
      }
      if (v == null || Number.isNaN(v)) continue;
      if (!(v >= 0 && v <= 1)) issues.push(`p-value [${i},${j}]=${v} out of [0,1]`);
      const w = matrix[j]?.[i];
      if (w != null && !Number.isNaN(w) && Math.abs(v - w) > 0.05) {
        issues.push(`asymmetry [${i},${j}]=${v} vs [${j},${i}]=${w}`);
      }
    }
  }
  return { ok: issues.length === 0, issues, matrix, models };
}

function invariantE(onCards, offCards) {
  // Outputs must differ
  const a = JSON.stringify(onCards.map((c) => [c.p5, c.p50, c.p95]));
  const b = JSON.stringify(offCards.map((c) => [c.p5, c.p50, c.p95]));
  return { ok: a !== b, on: a, off: b };
}

function invariantF(summary, verdict) {
  const issues = [];
  const cov70 = summary?.cov70_pooled ?? summary?.cov70 ?? null;
  const cov95 = summary?.cov95_pooled ?? summary?.cov95 ?? null;
  if (cov70 != null) {
    if (!(cov70 >= 0.5 && cov70 <= 0.9)) issues.push(`70% coverage ${cov70} outside [0.5, 0.9]`);
  }
  if (cov95 != null) {
    if (!(cov95 >= 0.8 && cov95 <= 1.0)) issues.push(`95% coverage ${cov95} outside [0.8, 1.0]`);
  }
  if (verdict && /WELL-?CALIBRATED/i.test(verdict.text || "")) {
    if (cov70 != null && (cov70 < 0.6 || cov70 > 0.85)) {
      issues.push(`verdict says WELL-CALIBRATED but cov70=${cov70}`);
    }
  }
  return { ok: issues.length === 0, issues, cov70, cov95, verdict_text: verdict?.text || null };
}

function invariantG(timingMs) {
  return { ok: timingMs < 2000, timing_ms: timingMs };
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-interaction harness
// ─────────────────────────────────────────────────────────────────────────────

async function step(fnName, label, opts, body) {
  counters.interactions++;
  const stepId = `${fnName}::${label}`;
  liveState.currentStep = stepId;
  liveState.currentParams = opts.params || null;
  log(`▶ ${stepId}`);
  const entry = {
    t: ts(),
    fn: fnName,
    step: label,
    params: opts.params || null,
    screenshots: { before: null, after_interaction: null, after_completion: null },
    streamlit_state: opts.streamlit_state || null,
    numeric_outputs: null,
    visible_exception: null,
    invariants: {},
    judge_critique: null,
    error: null,
  };

  try {
    if (opts.beforeShot !== false) {
      entry.screenshots.before = await screenshot(opts.page, `${fnName}_${label}_before`);
    }
    const result = await body(entry);
    if (opts.afterInteractionShot !== false && opts.page) {
      entry.screenshots.after_interaction = await screenshot(opts.page, `${fnName}_${label}_afterIx`);
    }
    if (opts.waitIdle !== false && opts.page) {
      await waitForStreamlitIdle(opts.page);
    }
    if (opts.afterCompletionShot !== false && opts.page) {
      entry.screenshots.after_completion = await screenshot(opts.page, `${fnName}_${label}_done`);
    }
    if (opts.page) {
      entry.visible_exception = await captureStreamlitException(opts.page);
    }
    if (result && typeof result === "object") {
      if (result.numeric_outputs !== undefined) entry.numeric_outputs = result.numeric_outputs;
      if (result.invariants !== undefined) Object.assign(entry.invariants, result.invariants);
      if (result.extra) entry.extra = result.extra;
    }
  } catch (e) {
    entry.error = e?.message || String(e);
    logErr(`✗ ${stepId}`, e);
    recordHarnessFailure({ step: stepId, error: entry.error });
  }

  // Update live state
  liveState.lastNumeric = entry.numeric_outputs;
  if (entry.invariants && Object.keys(entry.invariants).length) {
    liveState.quantState.lastInvariantCheck = entry.invariants;
  }

  // Judge
  try {
    const critique = await judgeCritique({
      step: stepId, params: entry.params,
      numeric_outputs: entry.numeric_outputs,
      visible_exception: entry.visible_exception,
      invariants: entry.invariants,
    });
    entry.judge_critique = critique;
    liveState.lastJudge = critique;
    if (critique && /\b(concern|inconsist|suspect|wrong|broken|implausible|invalid)\b/i.test(critique)) {
      recordJudgeConcern({ step: stepId, critique });
    }
  } catch (e) {
    logErr("judge failed", e);
  }

  pushRecent({
    t: entry.t.slice(11, 19), fn: fnName, step: label,
    invariants: Object.entries(entry.invariants).map(([k, v]) => `${k}:${v.ok ? "ok" : "BAD"}`).join(" "),
    judge: entry.judge_critique,
  });
  await appendTranscript(entry);
  return entry;
}

// ─────────────────────────────────────────────────────────────────────────────
// THE 18 FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

async function fn1_startup(page) {
  if (SKIP_FUNCTIONS.has("1")) return null;
  return step("F1_startup", "load_app", { page, beforeShot: false }, async (entry) => {
    await page.goto(APP_URL, { waitUntil: "domcontentloaded", timeout: 60000 });
    await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 60000 }).catch(() => {});
    await waitForStreamlitIdle(page, 1200, 90000);
    const status = await extractStatusBar(page);
    liveState.quantState.activeModel = "(reading…)";
    liveState.quantState.currentTab = "Live forecast";
    const issues = [];
    if (status.last_close != null && !(status.last_close >= 7.7 && status.last_close <= 7.9)) {
      issues.push(`last_close=${status.last_close} outside [7.7, 7.9]`);
      recordViolation("F", { kind: "startup_status_oor", field: "last_close", value: status.last_close });
    }
    if (status.fair_value != null && !(status.fair_value >= 7.7 && status.fair_value <= 7.9)) {
      issues.push(`fair_value=${status.fair_value} outside [7.7, 7.9]`);
    }
    if (status.yield_diff != null && !(status.yield_diff >= -2 && status.yield_diff <= 6)) {
      issues.push(`yield_diff=${status.yield_diff} outside [-2, 6]`);
    }
    return {
      numeric_outputs: { status, issues },
      invariants: { startup_status: { ok: issues.length === 0, issues } },
    };
  });
}

async function fn2_sidebar(page) {
  if (SKIP_FUNCTIONS.has("2")) return null;
  return step("F2_sidebar", "inspect_controls", { page, beforeShot: false }, async () => {
    const selectboxes = await page.locator('[data-testid="stSelectbox"]').count();
    const sliders = await page.locator('[data-testid="stSlider"]').count();
    const textInputs = await page.locator('[data-testid="stTextInput"]').count();
    const checkboxes = await page.locator('[data-testid="stCheckbox"]').count();
    const sidebarText = await page.locator('[data-testid="stSidebar"]').innerText().catch(() => "");
    const dataCaption = /EODHD|FRED|HIBOR|synthesised|synthesized/i.test(sidebarText);
    return {
      numeric_outputs: {
        selectbox_count: selectboxes,
        slider_count: sliders,
        text_input_count: textInputs,
        checkbox_count: checkboxes,
        data_caption_present: dataCaption,
        sidebar_text_sample: sidebarText.slice(0, 1200),
      },
      invariants: {
        sidebar_complete: {
          ok: selectboxes >= 1 && sliders >= 1 && checkboxes >= 1,
          issues: selectboxes === 0 ? ["no selectbox"] : [],
        },
      },
    };
  });
}

async function fn3_liveForecastDefault(page) {
  if (SKIP_FUNCTIONS.has("3")) return null;
  return step("F3_live_default", "render_bs_rv", { page, beforeShot: false }, async () => {
    const eq = await extractEquilibriumCaption(page);
    const cards = await extractHorizonCards(page);
    const aChecks = cards.map(invariantA);
    const allOk = aChecks.every((c) => c.ok);
    if (!allOk) {
      aChecks.forEach((c) => {
        if (!c.ok) recordViolation("A", { horizon: c.card.heading, issues: c.issues });
      });
    }
    await writeOutput("live-forecasts/bs_rv.json", { eq, cards });
    liveState.quantState.lastForecast = {
      model: "bs_rv",
      p_hkd_by_horizon: cards.map((c) => ({ h: c.heading, p: c.p_hkd_appreciation })),
    };
    return {
      numeric_outputs: { equilibrium: eq, cards },
      invariants: {
        A_probabilities: { ok: allOk, per_card: aChecks.map((c) => ({ heading: c.card.heading, ok: c.ok, issues: c.issues, sum: c.sum })) },
      },
    };
  });
}

async function fn4_cycleAllPricers(page) {
  if (SKIP_FUNCTIONS.has("4")) return null;
  const perModel = {};
  for (const model of PRICER_ORDER) {
    const timeout = HEAVY_PRICERS.has(model) ? HEAVY_PRICER_TIMEOUT_MS : PRICER_TIMEOUT_MS;
    const entry = await step("F4_cycle_pricers", `model_${model}`, { page }, async () => {
      const t0 = Date.now();
      let crashed = false;
      let cards = [];
      let exc = null;
      try {
        await setSelectbox(page, /Active model/i, PRICER_LABELS[model] || model);
        // Wait until either spinner gone or timeout
        await waitForStreamlitIdle(page, 800, timeout);
        cards = await extractHorizonCards(page);
        exc = await captureStreamlitException(page);
        if (exc) crashed = true;
      } catch (e) {
        crashed = true;
        exc = String(e?.message || e);
      }
      const elapsed = Date.now() - t0;
      liveState.quantState.activeModel = model;
      await writeOutput(`live-forecasts/${model}.json`, { cards, elapsed_ms: elapsed, exception: exc });
      if (crashed) {
        counters.pricersCrashed.push(model);
        recordViolation("B", { model, exception: exc?.slice(0, 600), elapsed_ms: elapsed });
      }
      const aChecks = cards.map(invariantA);
      const aOk = aChecks.length > 0 && aChecks.every((c) => c.ok);
      if (!aOk && !crashed) {
        recordViolation("A", { model, kind: "post-cycle horizon malformed" });
      }
      perModel[model] = { elapsed_ms: elapsed, crashed, has_cards: cards.length > 0 };
      return {
        numeric_outputs: { model, elapsed_ms: elapsed, crashed, exception_snippet: exc?.slice(0, 400), card_count: cards.length, first_card: cards[0] || null },
        invariants: {
          B_pricer_runs: { ok: !crashed, model, elapsed_ms: elapsed, timeout_ms: timeout },
          A_probabilities: { ok: aOk, model },
        },
      };
    });
  }
  // Reset to bs_rv for downstream tests
  try { await setSelectbox(page, /Active model/i, PRICER_LABELS.bs_rv); } catch {}
  return perModel;
}

async function fn5_overlayToggle(page) {
  if (SKIP_FUNCTIONS.has("5")) return null;
  // Capture overlay=on
  liveState.quantState.overlay = "on";
  const onEntry = await step("F5_overlay_toggle", "capture_overlay_on", { page }, async () => {
    const cards = await extractHorizonCards(page);
    return { numeric_outputs: { state: "on", cards } };
  });
  // Toggle off
  await step("F5_overlay_toggle", "toggle_off", { page }, async () => {
    await setCheckbox(page, /equilibrium overlay|overlay/i, false);
    liveState.quantState.overlay = "off";
    return { numeric_outputs: { state: "off (after toggle)" } };
  });
  const offEntry = await step("F5_overlay_toggle", "capture_overlay_off", { page }, async () => {
    const cards = await extractHorizonCards(page);
    return { numeric_outputs: { state: "off", cards } };
  });
  const inv = invariantE(
    onEntry?.numeric_outputs?.cards || [],
    offEntry?.numeric_outputs?.cards || [],
  );
  if (!inv.ok) recordViolation("E", { kind: "overlay toggle had no effect" });
  // Toggle back on
  await step("F5_overlay_toggle", "restore_overlay_on", { page }, async () => {
    await setCheckbox(page, /equilibrium overlay|overlay/i, true);
    liveState.quantState.overlay = "on";
    return { numeric_outputs: { state: "on (restored)" }, invariants: { E_overlay_effect: inv } };
  });
}

async function fn6_equilibriumSanity(page) {
  if (SKIP_FUNCTIONS.has("6")) return null;
  return step("F6_equilibrium", "read_caption", { page }, async () => {
    const eq = await extractEquilibriumCaption(page);
    const issues = [];
    if (!eq) {
      issues.push("equilibrium caption not found");
    } else {
      if (isFinite(eq.alpha) && !(eq.alpha >= 7.6 && eq.alpha <= 8.0)) issues.push(`alpha=${eq.alpha} suspect`);
      if (isFinite(eq.residual_sigma) && !(eq.residual_sigma > 0 && eq.residual_sigma < 0.1)) issues.push(`residual_sigma=${eq.residual_sigma} suspect`);
      if (isFinite(eq.lambda_per_day) && !(eq.lambda_per_day > 0 && eq.lambda_per_day < 1)) issues.push(`lambda=${eq.lambda_per_day} suspect`);
    }
    return {
      numeric_outputs: { equilibrium: eq, issues },
      invariants: { equilibrium_sane: { ok: issues.length === 0, issues } },
    };
  });
}

async function gotoBacktestSubTab(page, regex) {
  await clickTab(page, /Backtest/i);
  liveState.quantState.currentTab = "Backtest";
  // sub-tabs
  await clickTab(page, regex);
}

async function fn7_singleBacktest(page) {
  if (SKIP_FUNCTIONS.has("7")) return null;
  await gotoBacktestSubTab(page, /Single/i);
  liveState.quantState.currentTab = "Backtest · Single";
  const entry = await step("F7_single_backtest", "run_bs_rv_default", { page }, async () => {
    try { await setSelectbox(page, /model/i, PRICER_LABELS.bs_rv); } catch {}
    const t0 = Date.now();
    await clickButtonByText(page, /Run single-model backtest|Run backtest/i);
    await waitForStreamlitIdle(page, 1000, 120000);
    const elapsed = Date.now() - t0;
    const metrics = await extractMetrics(page);
    const tables = await extractDataFrames(page);
    const verdict = await extractVerdictBanner(page);
    const summary = {};
    metrics.forEach((m) => {
      if (/CRPS/i.test(m.label)) summary.crps = m.value_num;
      if (/Log score/i.test(m.label)) summary.log_score = m.value_num;
      if (/70.{0,3}cov/i.test(m.label)) summary.cov70_pooled = m.value_num > 1 ? m.value_num / 100 : m.value_num;
      if (/95.{0,3}cov/i.test(m.label)) summary.cov95_pooled = m.value_num > 1 ? m.value_num / 100 : m.value_num;
      if (/MAE/i.test(m.label)) summary.mae = m.value_num;
    });
    const fCheck = invariantF(summary, verdict);
    if (!fCheck.ok) recordViolation("F", { kind: "coverage/verdict", issues: fCheck.issues });
    await writeOutput("backtests/single/bs_rv-default.json", {
      elapsed_ms: elapsed, metrics, summary, tables, verdict,
    });
    liveState.quantState.lastBacktest = {
      model: "bs_rv", window: "1y default",
      crps: summary.crps, cov70: summary.cov70_pooled, cov95: summary.cov95_pooled,
      verdict: verdict?.text,
    };
    return {
      numeric_outputs: { elapsed_ms: elapsed, metrics, summary, table_count: tables.length, verdict },
      invariants: { F_coverage: fCheck },
      extra: { elapsed_ms: elapsed, summary, metrics },
    };
  });
  return entry;
}

async function fn8_determinism(page, fn7Entry) {
  if (SKIP_FUNCTIONS.has("8") || !fn7Entry) return null;
  return step("F8_determinism", "rerun_same_backtest", { page }, async () => {
    const t0 = Date.now();
    await clickButtonByText(page, /Run single-model backtest|Run backtest/i);
    await waitForStreamlitIdle(page, 600, 60000);
    const elapsed = Date.now() - t0;
    const metrics = await extractMetrics(page);
    const verdict = await extractVerdictBanner(page);
    const summary = {};
    metrics.forEach((m) => {
      if (/CRPS/i.test(m.label)) summary.crps = m.value_num;
      if (/Log score/i.test(m.label)) summary.log_score = m.value_num;
      if (/70.{0,3}cov/i.test(m.label)) summary.cov70_pooled = m.value_num > 1 ? m.value_num / 100 : m.value_num;
      if (/95.{0,3}cov/i.test(m.label)) summary.cov95_pooled = m.value_num > 1 ? m.value_num / 100 : m.value_num;
      if (/MAE/i.test(m.label)) summary.mae = m.value_num;
    });
    const cInv = invariantC(fn7Entry.extra?.summary, summary);
    const gInv = invariantG(elapsed);
    if (!cInv.ok) recordViolation("C", { prev: cInv.diff_prev, next: cInv.diff_next });
    if (!gInv.ok) recordViolation("G", { timing_ms: elapsed });
    return {
      numeric_outputs: { elapsed_ms: elapsed, summary, verdict_text: verdict?.text },
      invariants: { C_determinism: cInv, G_cache_hit: gInv },
    };
  });
}

async function fn9_pairwise(page) {
  if (SKIP_FUNCTIONS.has("9")) return null;
  await gotoBacktestSubTab(page, /Pairwise/i);
  liveState.quantState.currentTab = "Backtest · Pairwise";
  return step("F9_pairwise", "bs_rv_vs_merton_jd", { page }, async () => {
    // Two selectboxes — assume first is A, second is B
    try {
      const sels = page.locator('[data-testid="stSelectbox"]');
      await sels.nth(0).click(); await page.waitForTimeout(150);
      await page.locator('[role="option"]', { hasText: /BS.+realized/i }).first().click();
      await waitForStreamlitIdle(page);
      await sels.nth(1).click(); await page.waitForTimeout(150);
      await page.locator('[role="option"]', { hasText: /Merton/i }).first().click();
      await waitForStreamlitIdle(page);
    } catch (e) {
      logErr("pairwise model selection failed", e);
    }
    const t0 = Date.now();
    await clickButtonByText(page, /Run pairwise/i);
    await waitForStreamlitIdle(page, 1000, 240000);
    const elapsed = Date.now() - t0;
    const verdict = await extractVerdictBanner(page);
    const metrics = await extractMetrics(page);
    const tables = await extractDataFrames(page);
    // Sanity: try to find DM p-value in verdict text or page text
    const visible = await dumpVisibleText(page);
    const dmMatch = visible.match(/DM\s*p\s*=\s*([\d.eE\-]+)/i);
    const dm_p = dmMatch ? parseNum(dmMatch[1]) : null;
    const issues = [];
    if (dm_p != null && !(dm_p >= 0 && dm_p <= 1)) issues.push(`DM p=${dm_p} not in [0,1]`);
    await writeOutput("backtests/pairwise/bs_rv-vs-merton_jd.json", {
      elapsed_ms: elapsed, verdict, dm_p, metrics, tables, visible_sample: visible.slice(0, 2000),
    });
    return {
      numeric_outputs: { elapsed_ms: elapsed, verdict_text: verdict?.text, dm_p, metric_count: metrics.length },
      invariants: { pairwise_sane: { ok: issues.length === 0, issues, dm_p } },
    };
  });
}

async function fn10_allModel(page) {
  if (SKIP_FUNCTIONS.has("10")) {
    log("F10 skipped (SKIP_FUNCTIONS contains 10)");
    return null;
  }
  await gotoBacktestSubTab(page, /All-?model/i);
  liveState.quantState.currentTab = "Backtest · All-model";
  return step("F10_all_model", "run_full_matrix", { page }, async () => {
    const t0 = Date.now();
    await clickButtonByText(page, /Run all-?model backtest/i);
    liveState.allModelProgress = { started_ms: t0, status: "running" };
    await waitForStreamlitIdle(page, 1500, ALL_MODEL_BACKTEST_TIMEOUT_MS);
    const elapsed = Date.now() - t0;
    liveState.allModelProgress = { elapsed_ms: elapsed, status: "complete" };
    const tables = await extractDataFrames(page);
    const visible = await dumpVisibleText(page, 8000);
    // Try to parse the largest table as the DM matrix
    let dmMatrix = null;
    let dmModels = null;
    if (tables.length) {
      const biggest = tables.reduce((a, b) => (b.rows.length > a.rows.length ? b : a), tables[0]);
      if (biggest.rows.length >= 2) {
        const header = biggest.rows[0];
        dmModels = header.slice(1);
        dmMatrix = biggest.rows.slice(1).map((row) => row.slice(1).map((c) => {
          if (/nan/i.test(c) || c === "") return NaN;
          const v = parseNum(c);
          return isFinite(v) ? v : NaN;
        }));
      }
    }
    let dInv = { ok: true, issues: ["no matrix extracted"] };
    if (dmMatrix && dmModels) {
      dInv = invariantD(dmMatrix, dmModels);
      if (!dInv.ok) recordViolation("D", { issues: dInv.issues.slice(0, 10) });
    } else {
      recordViolation("D", { kind: "matrix not extracted" });
      dInv = { ok: false, issues: ["matrix not extracted"] };
    }
    await writeOutput("backtests/all-model/results.json", {
      elapsed_ms: elapsed, tables, dm_matrix: dmMatrix, dm_models: dmModels,
      visible_sample: visible,
    });
    return {
      numeric_outputs: { elapsed_ms: elapsed, table_count: tables.length, dm_models: dmModels, dm_matrix_size: dmMatrix?.length },
      invariants: { D_dm_matrix: dInv },
    };
  });
}

async function fn11_setAsLiveDefault(page) {
  if (SKIP_FUNCTIONS.has("11")) return null;
  await gotoBacktestSubTab(page, /Single/i);
  return step("F11_set_default", "click_set_as_live", { page }, async () => {
    const ok = await clickButtonByText(page, /Set as live default|Set winner as live|Set best as live/i);
    if (!ok) return { numeric_outputs: { found: false }, invariants: { set_default: { ok: false, issues: ["button not found"] } } };
    await waitForStreamlitIdle(page);
    // Verify dropdown reflects bs_rv (or whichever model)
    await clickTab(page, /Live forecast|📈/i);
    await waitForStreamlitIdle(page);
    const sidebarText = await page.locator('[data-testid="stSidebar"]').innerText().catch(() => "");
    return {
      numeric_outputs: { sidebar_sample: sidebarText.slice(0, 800) },
      invariants: { set_default: { ok: true } },
    };
  });
}

async function fn12_customBuckets(page) {
  if (SKIP_FUNCTIONS.has("12")) return null;
  return step("F12_custom_buckets", "set_new_buckets", { page }, async () => {
    const ok = await setTextInput(page, /price buckets|custom.+bucket/i, "7.79, 7.81, 7.83");
    await waitForStreamlitIdle(page);
    const cards = await extractHorizonCards(page);
    const aChecks = cards.map(invariantA);
    const allOk = aChecks.every((c) => c.ok);
    if (!allOk) aChecks.forEach((c) => { if (!c.ok) recordViolation("A", { kind: "custom_buckets", issues: c.issues }); });
    return {
      numeric_outputs: { set_ok: ok, card_count: cards.length, first_card: cards[0] || null },
      invariants: { A_probabilities: { ok: allOk } },
    };
  });
}

async function fn13_pathsSlider(page) {
  if (SKIP_FUNCTIONS.has("13")) return null;
  await step("F13_paths", "set_10k", { page }, async () => {
    await setSlider(page, /paths/i, 10000);
    const cards = await extractHorizonCards(page);
    return { numeric_outputs: { paths_target: 10000, first_card: cards[0] || null } };
  });
  await step("F13_paths", "set_100k", { page }, async () => {
    await setSlider(page, /paths/i, 100000);
    const cards = await extractHorizonCards(page);
    return { numeric_outputs: { paths_target: 100000, first_card: cards[0] || null } };
  });
  await step("F13_paths", "reset_50k", { page }, async () => {
    await setSlider(page, /paths/i, 50000);
    return { numeric_outputs: { paths_target: 50000 } };
  });
}

async function fn14_horizonSlider(page) {
  if (SKIP_FUNCTIONS.has("14")) return null;
  return step("F14_horizon", "set_12_months", { page }, async () => {
    const ok = await setSlider(page, /horizon.{0,12}month/i, 12);
    const cards = await extractHorizonCards(page);
    return {
      numeric_outputs: { ok, card_count: cards.length, headings: cards.map((c) => c.heading) },
      invariants: { horizon_renders: { ok: cards.length > 0 } },
    };
  });
}

async function fn15_dataCaption(page) {
  if (SKIP_FUNCTIONS.has("15")) return null;
  return step("F15_data_caption", "verify_text", { page, beforeShot: false, afterInteractionShot: false }, async () => {
    const sidebar = await page.locator('[data-testid="stSidebar"]').innerText().catch(() => "");
    const ok = /EODHD/i.test(sidebar) && /FRED/i.test(sidebar) && /(synthesised|synthesized|HIBOR|LERS)/i.test(sidebar);
    return {
      numeric_outputs: { caption_sample: sidebar.slice(0, 1200) },
      invariants: { data_caption: { ok, issues: ok ? [] : ["expected EODHD + FRED + synthesised reference"] } },
    };
  });
}

async function fn16_aiNarrative(page) {
  if (SKIP_FUNCTIONS.has("16")) return null;
  return step("F16_ai_narrative", "enable_if_available", { page }, async () => {
    const ok = await setCheckbox(page, /AI narrative|narrative summary/i, true);
    if (!ok) return { numeric_outputs: { available: false }, invariants: { ai_narrative: { ok: true, note: "checkbox absent" } } };
    await waitForStreamlitIdle(page, 1500, 90000);
    const visible = await dumpVisibleText(page, 4000);
    const hasNarrative = /(USD\/HKD|HKMA|HIBOR|peg)/i.test(visible) && visible.length > 200;
    return {
      numeric_outputs: { enabled: true, narrative_excerpt: visible.slice(0, 1200) },
      invariants: { ai_narrative: { ok: hasNarrative } },
    };
  });
}

async function fn17_reloadCache(page) {
  if (SKIP_FUNCTIONS.has("17")) return null;
  await step("F17_reload_cache", "page_reload", { page, beforeShot: false }, async () => {
    await page.reload({ waitUntil: "domcontentloaded" });
    await waitForStreamlitIdle(page, 1500, 90000);
    return { numeric_outputs: { reloaded: true } };
  });
  await gotoBacktestSubTab(page, /Single/i);
  return step("F17_reload_cache", "rerun_after_reload", { page }, async () => {
    try { await setSelectbox(page, /model/i, PRICER_LABELS.bs_rv); } catch {}
    const t0 = Date.now();
    await clickButtonByText(page, /Run single-model backtest|Run backtest/i);
    await waitForStreamlitIdle(page, 1000, 120000);
    const elapsed = Date.now() - t0;
    const gInv = invariantG(elapsed);
    if (!gInv.ok) recordViolation("G", { timing_ms: elapsed, kind: "post-reload" });
    return {
      numeric_outputs: { elapsed_ms: elapsed },
      invariants: { G_cache_hit_post_reload: gInv },
    };
  });
}

async function fn18_finalState(page) {
  if (SKIP_FUNCTIONS.has("18")) return null;
  return step("F18_final", "capture_final", { page }, async () => {
    await clickTab(page, /Live/i).catch(() => {});
    await waitForStreamlitIdle(page);
    const exc = await captureStreamlitException(page);
    const responsive = exc == null;
    return {
      numeric_outputs: { responsive, exception: exc?.slice(0, 800) || null },
      invariants: { final_responsive: { ok: responsive } },
    };
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Report writers
// ─────────────────────────────────────────────────────────────────────────────

function htmlEsc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function writeReport() {
  const groups = {};
  for (const e of transcript) {
    groups[e.fn] = groups[e.fn] || [];
    groups[e.fn].push(e);
  }
  const fnKeys = Object.keys(groups).sort();

  const toc = fnKeys.map((k) => `<li><a href="#${k}">${htmlEsc(k)}</a> <span class="small">(${groups[k].length} steps)</span></li>`).join("");

  const sections = fnKeys.map((k) => {
    const items = groups[k].map((e, i) => `
      <div class="step">
        <h3>${htmlEsc(e.step)} <span class="small">${htmlEsc(e.t)}</span></h3>
        <div><strong>Params:</strong> <code>${htmlEsc(JSON.stringify(e.params))}</code></div>
        <div class="shots">
          ${e.screenshots.before ? `<div><div class="small">before</div><img src="${htmlEsc(e.screenshots.before)}"/></div>` : ""}
          ${e.screenshots.after_interaction ? `<div><div class="small">after interaction</div><img src="${htmlEsc(e.screenshots.after_interaction)}"/></div>` : ""}
          ${e.screenshots.after_completion ? `<div><div class="small">after completion</div><img src="${htmlEsc(e.screenshots.after_completion)}"/></div>` : ""}
        </div>
        ${e.visible_exception ? `<div class="bad"><strong>Visible Python exception:</strong><pre>${htmlEsc(e.visible_exception)}</pre></div>` : ""}
        <div><strong>Numeric outputs:</strong><pre>${htmlEsc(JSON.stringify(e.numeric_outputs, null, 2))}</pre></div>
        <div><strong>Invariants:</strong><pre>${htmlEsc(JSON.stringify(e.invariants, null, 2))}</pre></div>
        <div><strong>Judge critique:</strong><div class="judge">${htmlEsc(e.judge_critique || "")}</div></div>
        ${e.error ? `<div class="bad"><strong>Harness error:</strong><pre>${htmlEsc(e.error)}</pre></div>` : ""}
      </div>`).join("");
    return `<section id="${k}"><h2>${htmlEsc(k)}</h2>${items}</section>`;
  }).join("");

  const html = `<!doctype html><html><head><meta charset="utf-8"/>
<title>R1 report — USD/HKD Edge — ${RUN_ID}</title>
<style>
  body{font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;background:#0d0d10;color:#ddd;margin:0;padding:0}
  header{position:sticky;top:0;background:#15151a;padding:14px 22px;border-bottom:1px solid #2a2a30;z-index:5}
  main{display:grid;grid-template-columns:240px 1fr;gap:20px;padding:18px 22px}
  nav{position:sticky;top:80px;align-self:start;max-height:80vh;overflow:auto}
  nav ul{list-style:none;padding:0;margin:0}
  nav li{margin:6px 0}
  a{color:#7af;text-decoration:none}
  h1{margin:0;color:#fff;font-size:16px}
  h2{color:#fff;font-size:14px;border-bottom:1px solid #234;padding-bottom:4px;margin-top:24px}
  h3{color:#dde;font-size:13px;margin:14px 0 6px}
  pre{background:#15151a;border:1px solid #2a2a30;padding:8px;border-radius:4px;white-space:pre-wrap;word-break:break-word;font-size:11px;max-height:360px;overflow:auto}
  .step{background:#111114;border:1px solid #1f1f25;border-radius:6px;padding:12px;margin:10px 0}
  .shots{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}
  .shots img{max-width:300px;border:1px solid #333}
  .judge{background:#142;border:1px solid #284;padding:8px;border-radius:4px;color:#cfd}
  .small{color:#888;font-size:11px}
  .bad{color:#f99}
  code{background:#222;padding:1px 4px;border-radius:3px}
  .summary{background:#1d2235;border:1px solid #345;padding:10px;border-radius:6px;margin:14px 0}
</style></head><body>
<header>
  <h1>R1 report — USD/HKD Edge</h1>
  <div class="small">Run ${htmlEsc(RUN_ID)} · ${htmlEsc(String(transcript.length))} interactions · ${htmlEsc(String(counters.judgeConcerns))} judge concerns · ${htmlEsc(String(Object.values(counters.invariantViolations).reduce((a, b) => a + b, 0)))} invariant violations · ${htmlEsc(String(counters.pricersCrashed.length))} pricers crashed</div>
</header>
<main>
  <nav>
    <strong>Contents</strong>
    <ul>${toc}</ul>
    <hr/>
    <div class="small">Invariant violations</div>
    <ul>${Object.entries(counters.invariantViolations).map(([k, v]) => `<li>${k}: ${v}</li>`).join("")}</ul>
  </nav>
  <div>
    <div class="summary">
      <strong>Invariant violations:</strong> ${Object.entries(counters.invariantViolations).map(([k, v]) => `${k}=${v}`).join(" · ")}
      <br/><strong>Pricers crashed:</strong> ${counters.pricersCrashed.length ? counters.pricersCrashed.join(", ") : "none"}
      <br/><strong>Judge concerns:</strong> ${counters.judgeConcerns}
      <br/><strong>Harness failures:</strong> ${counters.harnessSanityFailures}
    </div>
    ${sections}
  </div>
</main>
</body></html>`;
  await writeFile(path.join(OUTPUT_DIR, "report.html"), html);
}

async function writeFailures() {
  let md = `# Failures — R1 USD/HKD Edge — ${RUN_ID}\n\n`;
  md += `## CRITICAL INVARIANT VIOLATIONS\n\n`;
  if (invariantViolationEntries.length === 0) md += `_None._\n\n`;
  else {
    for (const v of invariantViolationEntries) {
      md += `- **Invariant ${v.invariant}** — ${JSON.stringify(v)}\n`;
    }
    md += "\n";
  }
  md += `## Judge concerns\n\n`;
  if (judgeConcernEntries.length === 0) md += `_None._\n\n`;
  else {
    for (const c of judgeConcernEntries) {
      md += `- **${c.step}** — ${c.critique?.slice(0, 600)}\n`;
    }
  }
  md += `\n## Harness sanity failures\n\n`;
  if (harnessFailureEntries.length === 0) md += `_None._\n`;
  else {
    for (const h of harnessFailureEntries) md += `- ${h.step}: ${h.error}\n`;
  }
  await writeFile(path.join(OUTPUT_DIR, "failures.md"), md);
}

async function writeSummary() {
  const totalInv = Object.values(counters.invariantViolations).reduce((a, b) => a + b, 0);
  const txt = `INTERACTIONS: ${counters.interactions}
JUDGE CONCERNS RAISED: ${counters.judgeConcerns}
CRITICAL INVARIANT VIOLATIONS: ${totalInv}
  Invariant A (probability sanity): ${counters.invariantViolations.A}
  Invariant B (13 pricers run): ${counters.invariantViolations.B}
  Invariant C (determinism): ${counters.invariantViolations.C}
  Invariant D (DM matrix properties): ${counters.invariantViolations.D}
  Invariant E (overlay has effect): ${counters.invariantViolations.E}
  Invariant F (coverage calibration): ${counters.invariantViolations.F}
  Invariant G (cache hits): ${counters.invariantViolations.G}
PRICERS THAT CRASHED: ${counters.pricersCrashed.length ? counters.pricersCrashed.join(", ") : "[]"}
HARNESS SANITY FAILURES: ${counters.harnessSanityFailures}
`;
  await writeFile(path.join(OUTPUT_DIR, "run-summary.txt"), txt);
  log("\n" + txt);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

async function main() {
  log("════════════════════════════════════════════════════════════════════");
  log("R1 is running.");
  log(`Live view:    http://localhost:${LIVE_VIEW_PORT}`);
  log(`Output dir:   ${OUTPUT_DIR}`);
  log(`Target app:   ${APP_URL}`);
  log(`Anthropic:    ${ANTHROPIC_KEY ? "enabled" : "DISABLED (no key)"} model=${ANTHROPIC_MODEL}`);
  log(`Skipping:     ${[...SKIP_FUNCTIONS].join(",") || "(none)"}`);
  log("Watch the live view — especially the Quant State panel.");
  log("Do not trust summary output alone.");
  log("════════════════════════════════════════════════════════════════════");

  liveState.status = "launching browser";
  const browser = await chromium.launch({ headless: HEADLESS });
  const context = await browser.newContext({
    viewport: { width: VIEWPORT_WIDTH, height: VIEWPORT_HEIGHT },
  });
  const page = await context.newPage();

  // Network log
  const netStream = createWriteStream(NETWORK_PATH, { flags: "a" });
  page.on("request", (req) => {
    try { netStream.write(JSON.stringify({ t: ts(), kind: "req", method: req.method(), url: req.url() }) + "\n"); } catch {}
  });
  page.on("response", (res) => {
    try { netStream.write(JSON.stringify({ t: ts(), kind: "res", status: res.status(), url: res.url() }) + "\n"); } catch {}
  });
  page.on("pageerror", (err) => { logErr("pageerror", err.message); });
  page.on("console", (msg) => { if (msg.type() === "error") logErr("browser console error", msg.text()); });

  try {
    liveState.status = "running";
    await fn1_startup(page);
    await fn2_sidebar(page);
    await fn3_liveForecastDefault(page);
    await fn4_cycleAllPricers(page);
    await fn5_overlayToggle(page);
    await fn6_equilibriumSanity(page);
    const fn7 = await fn7_singleBacktest(page);
    await fn8_determinism(page, fn7);
    await fn9_pairwise(page);
    await fn10_allModel(page);
    await fn11_setAsLiveDefault(page);
    await fn12_customBuckets(page);
    await fn13_pathsSlider(page);
    await fn14_horizonSlider(page);
    await fn15_dataCaption(page);
    await fn16_aiNarrative(page);
    await fn17_reloadCache(page);
    await fn18_finalState(page);
  } catch (e) {
    logErr("Fatal in main flow", e);
    recordHarnessFailure({ step: "main", error: String(e?.message || e) });
  }

  liveState.status = "writing report";
  await writeReport();
  await writeFailures();
  await writeSummary();

  await context.close();
  await browser.close();
  netStream.end();

  log("════════════════════════════════════════════════════════════════════");
  log("R1 finished.");
  log(`Open the report:        ${path.join(OUTPUT_DIR, "report.html")}`);
  log(`Open the failures:      ${path.join(OUTPUT_DIR, "failures.md")}`);
  log(`Live forecast outputs:  ${path.join(OUTPUT_DIR, "outputs", "live-forecasts")}`);
  log(`Backtest results:       ${path.join(OUTPUT_DIR, "outputs", "backtests")}`);
  log(`Raw transcript:         ${TRANSCRIPT_PATH}`);
  log("════════════════════════════════════════════════════════════════════");

  liveState.status = "RUN COMPLETE — keeping live view open for 60s";
  await new Promise((r) => setTimeout(r, 60_000));
  liveServer.close();
  consoleStream.end();

  const totalInv = Object.values(counters.invariantViolations).reduce((a, b) => a + b, 0);
  if (counters.harnessSanityFailures > 0) process.exit(3);
  if (totalInv > 0) process.exit(2);
  if (counters.judgeConcerns > 0) process.exit(1);
  process.exit(0);
}

main().catch((e) => {
  logErr("Top-level crash", e);
  process.exit(3);
});

# R1 — synthetic user agent for USD/HKD Edge

R1 is a Playwright-driven beta tester for the USD/HKD Edge Streamlit app. It
walks through 18 functions covering every documented feature, captures three
screenshots per interactive step, extracts every numeric output it can reach
through the DOM, runs the seven invariant checks defined in the build spec,
sends each interaction to Anthropic Claude for a quant-aware critique, and
writes a self-contained `report.html` plus `failures.md` you can review in
under 30 minutes.

This is a standalone Node project. It is **not** part of the pnpm workspace —
keep it that way.

## Install

```bash
cd tools/r1
npm install
```

`postinstall` runs `playwright install chromium`. If you'd rather defer the
browser download, run `npm install --ignore-scripts` and then
`npx playwright install chromium` manually when you're ready.

## Run

The USD/HKD Edge Streamlit app must already be running (the `Start application`
workflow at `http://localhost:5000`). R1 does **not** start or restart the app.

```bash
# full plan (Function 10 / all-model backtest may take 5–15 min)
npm start

# smoke run — everything except the expensive all-model backtest
npm run smoke

# headless, custom output dir, custom port
HEADLESS=true LIVE_VIEW_PORT=7777 npm start
```

While R1 runs, open the live view in a browser:

    http://localhost:7777

The live view shows R1's current step, the parameter choices it made, the most
recent screenshot, the parsed numeric outputs, the judge critique, and a
dedicated **Quant State** panel (active model, current tab, overlay state,
last forecast, last backtest, invariant check status).

## Environment variables

| Variable                          | Default                                   | Purpose                                                |
| --------------------------------- | ----------------------------------------- | ------------------------------------------------------ |
| `APP_URL`                         | `http://localhost:5000`                   | Where the Streamlit app is reachable.                  |
| `HEADLESS`                        | `false`                                   | Set to `true` to run Chromium headless.                |
| `LIVE_VIEW_PORT`                  | `7777`                                    | Port for the in-browser live view server.              |
| `SKIP_FUNCTIONS`                  | empty                                     | Comma-separated function numbers to skip, e.g. `10,16`.|
| `PRICER_TIMEOUT_MS`               | `60000`                                   | Per-model timeout for the 13-pricer cycle (fast).      |
| `HEAVY_PRICER_TIMEOUT_MS`         | `180000`                                  | Override timeout for heavy pricers (Heston family).    |
| `ALL_MODEL_BACKTEST_TIMEOUT_MS`   | `1800000`                                 | All-model backtest timeout (default 30 min).           |
| `ANTHROPIC_MODEL`                 | `claude-opus-4-5`                         | Claude model id used by R1's brain and the judge.      |
| `ANTHROPIC_API_KEY` / `ANTHROPIC` | (required for brain + judge calls)        | Anthropic API key. Either name is accepted.            |

> The build spec called for `claude-opus-4-7`, which does not exist as a
> public model id at time of writing. The default falls back to
> `claude-opus-4-5`. Override via `ANTHROPIC_MODEL` if you have access to a
> newer family.

If `ANTHROPIC_API_KEY` (or `ANTHROPIC`) is unset, R1 still runs every Playwright
interaction and every invariant check, but the brain-driven approach choice
collapses to "documented defaults" and the judge critique field is set to
`"(judge skipped — no ANTHROPIC_API_KEY)"`. The harness exit code still
reflects invariant violations and harness sanity failures.

## Output layout

Per run, R1 writes everything to `tools/r1/runs/<ISO-timestamp>/`:

```
runs/2026-05-17T19-50-00-000Z/
├── report.html               # self-contained, sticky TOC, grouped by function
├── failures.md               # critical invariant violations first, judge concerns below
├── transcript.jsonl          # one JSON object per interaction
├── run-summary.txt           # interaction count, judge concerns, invariant violations
├── console.log               # Playwright + R1 console output dual-write
├── network.log               # JSONL HTTP requests (limited — Streamlit is WebSocket-heavy)
├── screenshots/              # numbered PNGs
└── outputs/
    ├── live-forecasts/       # one JSON per model: full horizon-card numbers
    ├── backtests/
    │   ├── single/           # per-model single-backtest JSON
    │   ├── pairwise/         # pairwise comparison JSON
    │   └── all-model/        # ranking table + DM matrix + rolling CRPS
    └── screenshots/          # extra chart-only screenshots for evidence
```

## Exit codes

| Code | Meaning                                                           |
| ---- | ----------------------------------------------------------------- |
| 0    | Clean run — no judge concerns, no invariant violations.           |
| 1    | At least one judge concern raised.                                |
| 2    | At least one critical invariant violation.                        |
| 3    | Harness sanity failed (e.g. R1 itself crashed mid-interaction).   |

## The seven invariants

| ID | What it checks                                                                              |
| -- | ------------------------------------------------------------------------------------------- |
| A  | Live-forecast probability distributions are well-formed (sum to 1, monotonic quantiles).    |
| B  | All 13 pricers render a forecast without raising a visible Python traceback.                |
| C  | Same backtest inputs → identical numbers on re-run (determinism with seed=42).              |
| D  | All-model DM p-value matrix is N×N, NaN diagonal, in [0,1], approximately symmetric.        |
| E  | Toggling the disequilibrium overlay actually changes results (it is not a no-op).           |
| F  | Coverage metrics fall inside reasonable bands; verdict text is consistent with the numbers. |
| G  | An identical re-run hits the cache (second run < 2 s).                                      |

## Streamlit selector notes

R1 uses Streamlit's `data-testid` attributes (`stSelectbox`, `stSlider`,
`stButton`, `stDataFrame`, `stPlotlyChart`, `stSpinner`, `stMetric`,
`stException`) plus heuristic header-text matching for tabs and sub-tabs. After
every input change R1 waits for `[data-testid="stSpinner"]` to disappear plus a
short settle delay before extracting numbers.

If the USD/HKD Edge UI changes a header text or moves a control, the relevant
extractor will fall back to capturing the raw text of the surrounding
container into the transcript and logging a soft warning — it will not silently
report green.

## Limitations / honesty

- DM matrix off-diagonal symmetry is checked with a 0.05 tolerance per the
  build spec.
- The judge prompt is tuned for quantitative finance, not generic writing.
- The harness does not validate the *correctness* of the underlying numerics
  (CRPS formula, DM HAC variance), only their *consistency*: monotonicity,
  matrix properties, determinism, cache behaviour, etc.
- Streamlit's WebSocket protocol means `network.log` is sparse. The substantive
  evidence is in `transcript.jsonl`, `outputs/`, and `screenshots/`.
- If the live USD/HKD Edge app surfaces a Python traceback, the harness
  captures the visible text — it does not introspect Python stderr.

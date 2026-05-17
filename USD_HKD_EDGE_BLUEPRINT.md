USD/HKD EDGE — COMPLETE APPLICATION BLUEPRINT

================================================================================
PART 1: APPLICATION OVERVIEW

USD/HKD Edge is a Streamlit-based probability-forecasting and statistical-backtesting platform for the USD/HKD spot exchange rate. It calibrates one of thirteen pricing models to live USD/HKD history, projects the full distribution of future prices across multiple horizons via Monte Carlo, optionally tilts the drift with a rate-differential disequilibrium overlay, and rigorously evaluates model quality with a walk-forward backtest module (single-model, pairwise, and all-model comparison sub-tabs).

The app is designed around the principle of statistically defensible output: every forecast is a probability distribution rather than a point estimate; every model is held accountable via walk-forward CRPS / log score / coverage diagnostics; every model comparison reports a Diebold-Mariano p-value with horizon-aware HAC variance pooling; and the app explicitly acknowledges the structural realities of the HKD peg (HKMA convertibility band 7.75–7.85) rather than pretending USD/HKD is a free-floating currency.

Stack:
  Frontend / UI       — Streamlit (single-page app, sidebar + tabs + sub-tabs)
  Plotting            — Plotly (graph_objects, subplots)
  Numerics            — NumPy, SciPy (optimize, stats, special functions)
  Data                — pandas (parquet caching, time-series alignment)
  Optional AI         — Anthropic Claude (narrative summary; gracefully disabled if no key)
  Caching             — Local parquet cache in cache/ (12h TTL) + in-process st.cache_data
  Runtime             — Python 3.11

Authentication: None. Single-user local Streamlit app. No session secrets or user table.

Workflow / start command:
  streamlit run main.py --server.port 5000 --server.address 0.0.0.0
  (Registered as the `Start application` workflow.)

Environment secrets:
  EODHD            — required; USD/HKD daily price history. Code also accepts EODHD_API_KEY.
  ANTHROPIC        — optional; only used when "AI narrative summary" is enabled. Code also accepts ANTHROPIC_API_KEY.
  POLYGON          — reserved for a future fallback data source; not currently consumed.
  SESSION_SECRET   — present but unused by the Python app (Streamlit manages its own session state).

================================================================================
PART 2: THE CORE FEATURES

──────────────────────────────────────────────────────────────────────────────
FEATURE 1 — MODEL SELECTOR (13 PRICERS, SINGLE INTERFACE)
──────────────────────────────────────────────────────────────────────────────
Location: Sidebar dropdown "Active model (drives live forecasts)" + parameter panel at the bottom of the Live tab.
Purpose: Choose which stochastic pricing model drives both live forecasts and (by default) backtests, with calibrated parameters displayed for transparency.

The 13 registered models, in `PRICER_ORDER`:
  heston            Heston (stochastic volatility)
  bates             Bates (Heston + Merton-style jumps in price)
  svjj              SVJJ (price + variance jumps)
  merton_jd         Merton jump-diffusion
  kou_jd            Kou double-exponential jump-diffusion
  vg                Variance Gamma
  cgmy              CGMY (tempered stable; Cornish-Fisher cumulant calibration)
  nig               Normal Inverse Gaussian (cumulant calibration)
  sabr              SABR (β=1, lognormal forward)
  double_heston     Double-Heston (two variance factors)
  rough_heston      Rough Heston (Hurst H<0.5; simplified calibration)
  bs_rv             BS · realized vol  (default live model — fast, peg-appropriate)
  bs_garch          BS · GARCH(1,1) forecast vol

Components / files:
  engine/pricers.py
    - `Pricer` abstract base + `ParamSpec` dataclass
    - 13 subclasses (one per model above)
    - `PRICERS` dict[str, Pricer], `PRICER_ORDER` list[str], `get_pricer(name)`, `pricer_choices() -> list[(key,label)]`
  main.py
    - Sidebar selectbox bound to `st.session_state["selected_model"]`
    - `calibrate_pricer(model_key, returns_key, annual_drift)` — cached wrapper
    - Parameter panel renders `pricer.display_params(params)` at the bottom of the Live tab

Pricer interface (must be honoured by every model):
  class Pricer:
      name: str
      label: str
      params_spec: list[ParamSpec]
      def calibrate(self, log_returns: np.ndarray, dt: float, annual_drift: float) -> dict[str, float]
      def simulate_paths(self, params, s0, n_steps, n_paths, dt, annual_drift,
                         extra_drift_fn=None, seed=42) -> np.ndarray  # shape (n_steps+1, n_paths)
      def display_params(self, params) -> list[(name, formatted_value, description)]

──────────────────────────────────────────────────────────────────────────────
FEATURE 2 — LIVE PROBABILITY FORECAST
──────────────────────────────────────────────────────────────────────────────
Location: Top-level tab "📈 Live forecast"
Purpose: Render the full forward probability distribution of USD/HKD for the active model across 1w / 1m / 3m / 6m horizons, with a fan chart, terminal histograms, and custom bucket probabilities.

Sub-sections (top → bottom):
  - Probability Fan chart (Plotly)             — quantile bands (2.5/5/15/25/50/75/85/95/97.5%) over time
  - Horizon distribution cards (×4)            — 1 week / 1 month / 3 months / 6 months
      Each card: terminal histogram, p5/p50/p95, bucket probability table, top-bucket caption,
                 P(HKD strengthens) + P(USD appreciates)
  - Equilibrium Tracker chart (Plotly secondary-y) — actual USD/HKD vs model equilibrium, z-score line
  - Equilibrium caption                         — alpha, beta, residual σ, daily λ, overlay on/off
  - Model parameter panel                       — calibrated parameter values from `display_params`
  - Optional AI narrative summary               — collapsed, opt-in checkbox

Simulation config (sidebar):
  - Number of paths        (default 50,000)
  - Horizon in months      (default 6 — sliced internally for 1w / 1m / 3m / 6m views)
  - Custom price buckets   (default "7.78, 7.80, 7.82, 7.84")
  - Apply rate-differential equilibrium overlay (default on)

Key code paths:
  main.py
    - `load_fx_history()` (`st.cache_data`)
    - `load_rate_differential()` (`st.cache_data`)
    - `calibrate_pricer(model_key, returns_key, annual_drift)` (`st.cache_data`)
    - `make_extra_drift_fn(eq, use_overlay)` — closure over `disequilibrium_drift_per_step`
    - `simulate_live_paths(...)` — calls `pricer.simulate_paths(...)`
    - `summarize_terminals(...)` — slices the path matrix per horizon
    - `render_horizon_card(label, days, stats)` — draws histogram + bucket table
    - Fan chart built directly with `go.Scatter` traces using QS quantiles
  engine/monte_carlo.py
    - `calendar_to_trading_steps(calendar_days) -> int`
    - `summarize_terminals(horizon_days, terminals, s0, buckets) -> HorizonStats`
    - `HorizonStats` dataclass (mean, p05, p25, p75, p95, bucket_probs, p_hkd_appreciation, p_usd_appreciation)

──────────────────────────────────────────────────────────────────────────────
FEATURE 3 — RATE-DIFFERENTIAL DISEQUILIBRIUM OVERLAY
──────────────────────────────────────────────────────────────────────────────
Location: Sidebar checkbox "Apply rate-differential equilibrium overlay" (default on); affects both Live and Backtest paths.
Purpose: Tilt the simulated drift toward the rate-implied fair value so paths mean-revert when USD/HKD is dislocated, while keeping the same dynamics in live and backtest.

Equilibrium model (OLS on aligned daily history):
    USDHKD_t = α + β · (US_3m_t − HK_3m_t) + residual_t
    residual_std (rolling, 252d, min 60) computed at each date
    z_t = residual_t / residual_std_t
    λ (per-day, fraction of σ to revert per day) estimated by regressing
       next-day log-return on z_t, multiplied by -1.

Per-step drift contribution applied inside every Monte Carlo path:
    extra_log_drift_per_day = −λ · (s_t − fair_value_at_t) / s_t · dt
                                  + σ_resid_daily · dW            (path-dependent)

Files:
  engine/disequilibrium_fx.py
    - `EquilibriumModel` dataclass: alpha, beta, lambda_, residual_std, z_score,
      equilibrium, fitted (DataFrame with usdhkd, diff, equilibrium, residual, z, resid_std)
    - `fit_equilibrium(fx_close, rate_diff, rolling_window=252) -> EquilibriumModel`
    - `disequilibrium_drift_per_step(s_t, eq, sigma_resid, lambda_per_day) -> ndarray`
    - `EquilibriumModel.zscore_at(when)` — as-of lookup used by the backtest

──────────────────────────────────────────────────────────────────────────────
FEATURE 4 — WALK-FORWARD BACKTEST ENGINE
──────────────────────────────────────────────────────────────────────────────
Location: Top-level tab "🧪 Backtest"; engine in engine/backtest.py.
Purpose: For every backtest date D in the chosen window, calibrate the model on history through D-1, simulate paths for each requested horizon, and score the forecast distribution against the realised rate at D + horizon (in trading days).

Defaults (interactive UX tuned):
  Window               1 year ending at the latest available date
  Paths                1,000
  Step                 10 trading days between forecast dates
  Horizons             {1w (5td), 1m (21td), 3m (63td)}
  Recalibration        monthly (re-uses parameters between recalibrations)
  Overlay              on
  Cache key            SHA1(returns_bytes + rate_diff_bytes + last_index_date) +
                       model + date_range + horizons + paths + step + overlay

Per-forecast metrics computed (per `ForecastRecord`):
  s0, realised, median, mean, p05, p25, p50, p75, p95
  crps                 — sample CRPS via Hersbach decomposition
  log_score            — Gaussian-KDE log score (Silverman bandwidth)
  cov70, cov95         — coverage indicators
  band70_lo/hi, band95_lo/hi
  abs_err              — |realised − median|

Walk-forward driver:
  engine/backtest.py
    - `HORIZONS_TD` = {"1w":5, "2w":10, "1m":21, "3m":63, "6m":126}
    - `crps_sample(samples, y)` (Hersbach closed form for sorted samples)
    - `log_score_kde(samples, y)` (Silverman bandwidth)
    - `coverage_indicator(samples, y, level)`
    - `diebold_mariano(loss_a, loss_b, h)` — legacy single-h DM
    - `_dm_pooled_by_horizon(records_a, records_b)` — horizon-aware HAC pooling (preferred)
    - `run_single_model_backtest(model_key, fx, start, end, horizons, n_paths, step_days,
                                  equilibrium, extra_drift_lambda, use_eq_overlay, seed)`
        returns `BacktestResult` (forecasts DataFrame + summary metrics + calibration table)
    - `calibration_verdict(summary) -> tuple[level, headline_text]`
        levels: "WELL-CALIBRATED" / "PARTIALLY" / "BIASED" / "OVERCONFIDENT" / "UNDERCONFIDENT"
    - `pairwise_winner(result_a, result_b) -> dict`
        keys: winner, crps_a, crps_b, rel_improvement, dm_stat, dm_p, n
    - `all_model_pvalue_matrix(results: dict[str, BacktestResult]) -> pd.DataFrame`
    - `rolling_crps(forecasts: pd.DataFrame, window_forecasts=10) -> pd.DataFrame`

Diebold-Mariano implementation notes:
  - `_dm_pooled_by_horizon` groups per-horizon losses and applies a Newey-West HAC variance
    with bandwidth bw = min(h_td, max(NW_rule, n // 4)) where NW_rule = floor(4·(n/100)^(2/9)).
  - If the sample HAC sum is suspiciously small (var_d < 0.1 · γ₀) it falls back to γ₀ (iid).
  - Pooled stat across horizons is the inverse-variance-weighted mean d_bar / sqrt(1/Σwᵢ).
  - n < 10 in any horizon: that horizon is skipped, not clamped.

──────────────────────────────────────────────────────────────────────────────
FEATURE 5 — BACKTEST: SINGLE-MODEL SUB-TAB
──────────────────────────────────────────────────────────────────────────────
Location: "Backtest" → "Single model" sub-tab.
Purpose: Evaluate one model in isolation against history, with a verdict banner and full diagnostics.

UI elements (top → bottom):
  - Model dropdown (defaults to the current live model)
  - "Run single-model backtest" button (runs via `run_backtest_cached`)
  - Verdict banner (colour-coded — green / yellow / red)
  - 5 metric cards: CRPS, Log score, 70% coverage, 95% coverage, MAE (median)
  - Per-horizon summary table
  - Scatter plot:        median forecast vs realised, with 45° line
  - Calibration plot:    nominal vs empirical coverage at 50/70/95
  - Time-series chart:   realised vs median + 70/95 bands
  - "Set as live default" button — queues `pending_default_model` for next rerun
  - Runtime caption:     "X.X s · N forecasts evaluated"

──────────────────────────────────────────────────────────────────────────────
FEATURE 6 — BACKTEST: PAIRWISE COMPARISON SUB-TAB
──────────────────────────────────────────────────────────────────────────────
Location: "Backtest" → "Pairwise comparison" sub-tab.
Purpose: Head-to-head model A vs B, with the Diebold-Mariano p-value as the headline test of forecast-skill difference.

UI elements:
  - Two model dropdowns (Model A / Model B)
  - "Run pairwise comparison" button
  - Winner banner: "WINNER: <label> — CRPS A=… B=… — DM p=…"
  - Side-by-side metrics table with a "Better" column
  - Two scatter plots side by side (one per model)
  - Combined time-series: realised + median A + median B
  - "Set winner as live default" button

Code: main.py uses `pairwise_winner(res_a, res_b)` and renders the result.

──────────────────────────────────────────────────────────────────────────────
FEATURE 7 — BACKTEST: ALL-MODEL RANKING SUB-TAB
──────────────────────────────────────────────────────────────────────────────
Location: "Backtest" → "All-model ranking" sub-tab.
Purpose: Rank every selected model on the same window, with a full DM p-value matrix and rolling-CRPS lines.

UI elements:
  - Multiselect "Models to include" (all 13 by default)
  - "Run all-model backtest" button (heavy — only triggered on click)
  - Ranking table sorted by CRPS ascending
  - DM p-value matrix (DataFrame, NaN on diagonal)
  - Rolling-CRPS chart (Plotly, one line per model)
  - "Set best as live default" button

──────────────────────────────────────────────────────────────────────────────
FEATURE 8 — DATA PIPELINE
──────────────────────────────────────────────────────────────────────────────
Purpose: Pull, cache, and align all market data required by the live and backtest paths.

USD/HKD daily history (data/eodhd_fx.py):
  - `fetch_usdhkd_history(years=5, symbol="USDHKD.FOREX") -> pd.DataFrame`
      columns: open, high, low, close, adjusted_close, volume
  - Cache: cache/eodhd_USDHKD_FOREX.parquet (12h TTL)
  - On API failure with a cached file present: silently serves stale cache.
  - `latest_close(df) -> float`
  - `daily_log_returns(df) -> pd.Series`

US 3-month T-bill yield (data/rates.py):
  - `fetch_us_3m_yield() -> pd.Series`        — FRED DTB3 via public CSV (no key)
  - Cache: cache/fred_DTB3.parquet (12h TTL)

HKD 3-month rate (synthetic) (data/rates.py):
  - `fetch_hk_3m_yield(us_yield=None) -> pd.Series`
      Synthesises HIBOR under the HKMA Linked Exchange Rate System:
          HIBOR = US + HK_SPREAD_PCT + HK_SPREAD_BETA · (US − mean(US))
          HK_SPREAD_PCT = -0.30      (long-run US − HK spread, pp)
          HK_SPREAD_BETA = 0.05      (short-end elasticity to US level)
      Justification: FRED no longer publishes a free HK 3m series under a stable ID; the LERS
      peg makes HIBOR closely track the US rate, so a deterministic synthesis is defensible.
  - Documented in the sidebar caption and replit.md.

Joined rate differential:
  - `build_rate_differential() -> pd.DataFrame[us_yield, hk_yield, diff]`  (diff = us − hk)

──────────────────────────────────────────────────────────────────────────────
FEATURE 9 — HESTON BUILDING BLOCK
──────────────────────────────────────────────────────────────────────────────
Location: engine/heston.py (used directly by `HestonPricer` and as a substrate for `BatesPricer`,
`SVJJPricer`, `DoubleHestonPricer`).
Purpose: Provide a robust Heston SDE simulator and a particle-filter MLE calibration.

Key exports:
  - `HestonParams` dataclass (kappa, theta, sigma_v, rho, v0, mu)
  - `calibrate_heston_pf(log_returns, dt, n_particles=500, bounds=...) -> HestonParams`
      Particle-bootstrap filter, bounded optimisation, graceful fallback to a historical-vol
      moment-match if MLE fails or returns degenerate values.
  - `simulate_heston_paths(params, s0, n_steps, n_paths, dt, drift_per_day,
                            extra_drift_fn=None, seed=42) -> ndarray`
      Full-truncation Euler scheme; variance is clamped to ≥0 before each step.

──────────────────────────────────────────────────────────────────────────────
FEATURE 10 — OPTIONAL AI NARRATIVE SUMMARY
──────────────────────────────────────────────────────────────────────────────
Location: Bottom of the Live tab, behind a checkbox (opt-in).
Purpose: Short (≤200 word) Anthropic Claude narrative that explains the active model, the
current disequilibrium signal, and the resulting horizon probabilities in plain language.

Behaviour:
  - If `ANTHROPIC` / `ANTHROPIC_API_KEY` is not set, the checkbox is disabled and a caption
    explains why.
  - The quantitative output is the source of truth; the narrative is never substituted for
    the numbers.

================================================================================
PART 3: COMPLETE FILE TREE

/
├── USD_HKD_EDGE_BLUEPRINT.md        # This document
├── README.md                        # Public-facing overview
├── replit.md                        # Project documentation and preferences
├── main.py                          # Streamlit app — single page, tabs + sidebar (≈950 lines)
├── pyproject.toml / requirements    # Python dependency manifest
│
├── engine/
│   ├── __init__.py
│   ├── pricers.py                   # Pricer base + 13 model classes (≈960 lines)
│   ├── backtest.py                  # Walk-forward backtest + metrics + DM tests (≈490 lines)
│   ├── disequilibrium_fx.py         # OLS equilibrium + per-step drift (≈110 lines)
│   ├── heston.py                    # Heston SDE + particle-filter MLE (≈240 lines)
│   └── monte_carlo.py               # Horizon-stat helpers, bucket probs (≈140 lines)
│
├── data/
│   ├── __init__.py
│   ├── eodhd_fx.py                  # USD/HKD daily history loader + parquet cache
│   └── rates.py                     # US 3m FRED + synthetic HK 3m rate + rate-diff builder
│
├── cache/                           # Local on-disk parquet cache (12h TTL)
│   ├── eodhd_USDHKD_FOREX.parquet
│   └── fred_DTB3.parquet
│
├── .streamlit/
│   └── config.toml                  # port 5000, headless, 0.0.0.0
│
└── attached_assets/                 # User-uploaded images / reference docs

================================================================================
PART 4: STATE & DATA FLOW (STREAMLIT)

Streamlit session state keys (managed in main.py):
  selected_model             str        — current active model key (sidebar dropdown)
  use_overlay                bool       — disequilibrium overlay on/off
  pending_default_model      str        — set by "Set as live default" buttons; consumed at the
                                          TOP of the next rerun before the selectbox is created
                                          (avoids StreamlitAPIException from mutating widget state)
  bt_single_*, bt_pairwise_*, bt_all_*  — backtest configuration knobs in the Backtest tab

Cached computations (st.cache_data):
  - load_fx_history()                       6h TTL
  - load_rate_differential()                6h TTL
  - calibrate_pricer(model, returns_key,    no TTL; key includes hash of returns
                     annual_drift)
  - run_backtest_cached(model, date_range,  keyed by SHA1 of returns + rate-diff + last_date,
                        horizons, paths,    plus all settings
                        step, overlay)

Data flow for a Live forecast render:
  fx = load_fx_history()
  rates = load_rate_differential()
  joined = concat(fx_close, rates['diff'])
  eq = fit_equilibrium(joined['usdhkd'], joined['diff'], 252)
  cal = calibrate_pricer(selected_model, returns_key, annual_drift_estimate)
  extra_drift_fn = make_extra_drift_fn(eq, use_overlay)
  s_paths = pricer.simulate_paths(cal['params'], s0=current_spot, n_steps=..., n_paths=...,
                                   extra_drift_fn=extra_drift_fn)
  → fan chart + per-horizon HorizonStats

Data flow for a Backtest run:
  result = run_single_model_backtest(model_key, fx, start, end, horizons,
                                      n_paths=1000, step_days=10,
                                      equilibrium=eq, extra_drift_lambda=eq.lambda_,
                                      use_eq_overlay=use_overlay)
  verdict = calibration_verdict(result.summary)
  → banner + metric cards + per-horizon table + scatter + calibration + time-series

================================================================================
PART 5: KEY INTERFACES (PYTHON)

── Pricer (engine/pricers.py) ─────────────────────────────────────────────────
class Pricer:
    name: str
    label: str
    params_spec: list[ParamSpec]
    def calibrate(self, log_returns: np.ndarray,
                  dt: float = 1.0/252,
                  annual_drift: float = 0.0) -> dict[str, float]
    def simulate_paths(self, params: dict, s0: float, n_steps: int, n_paths: int,
                       dt: float = 1.0/252, annual_drift: float = 0.0,
                       extra_drift_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                       seed: Optional[int] = 42) -> np.ndarray   # (n_steps+1, n_paths)
    def display_params(self, params: dict) -> list[tuple[str, str, str]]
PRICERS: dict[str, Pricer]
PRICER_ORDER: list[str]                  # display order
get_pricer(name: str) -> Pricer
pricer_choices() -> list[tuple[str, str]]   # [(key, label), ...]

── EquilibriumModel (engine/disequilibrium_fx.py) ─────────────────────────────
@dataclass
class EquilibriumModel:
    alpha: float
    beta: float
    residual_std: float
    lambda_: float
    z_score: float
    equilibrium: float
    fitted: pd.DataFrame                     # usdhkd, diff, equilibrium, residual, z, resid_std
    def zscore_at(self, when: pd.Timestamp) -> float

fit_equilibrium(fx_close: pd.Series, rate_diff: pd.Series,
                rolling_window: int = 252) -> EquilibriumModel

disequilibrium_drift_per_step(s_t: np.ndarray,
                              eq: EquilibriumModel,
                              sigma_resid: float,
                              lambda_per_day: float) -> np.ndarray

── HorizonStats (engine/monte_carlo.py) ───────────────────────────────────────
@dataclass
class HorizonStats:
    horizon_days: int
    mean: float; p05: float; p25: float; p50: float; p75: float; p95: float
    bucket_probs: dict[str, float]
    p_hkd_appreciation: float
    p_usd_appreciation: float

calendar_to_trading_steps(calendar_days: int) -> int
summarize_terminals(horizon_days, terminals, s0, buckets) -> HorizonStats

── Backtest (engine/backtest.py) ──────────────────────────────────────────────
HORIZONS_TD: dict[str, int] = {"1w":5, "2w":10, "1m":21, "3m":63, "6m":126}

@dataclass
class ForecastRecord:
    date, horizon, horizon_td, target_date, s0, realised
    median, mean, p05, p25, p50, p75, p95
    crps, log_score, cov70, cov95
    band70_lo, band70_hi, band95_lo, band95_hi
    abs_err

@dataclass
class BacktestResult:
    model: str
    forecasts: pd.DataFrame          # one row per (date, horizon)
    summary: dict                    # per-horizon + pooled metrics
    calibration: pd.DataFrame        # nominal vs empirical coverage
    runtime_s: float
    n_forecasts: int

crps_sample(samples, y) -> float
log_score_kde(samples, y) -> float
coverage_indicator(samples, y, level) -> int
diebold_mariano(loss_a, loss_b, h=1) -> (dm_stat, p)          # legacy single-h
_dm_pooled_by_horizon(records_a, records_b) -> (dm_stat, p, n) # preferred; HAC pooling
calibration_verdict(summary) -> (level, headline)
run_single_model_backtest(...) -> BacktestResult
pairwise_winner(a: BacktestResult, b: BacktestResult) -> dict
all_model_pvalue_matrix(results: dict[str, BacktestResult]) -> pd.DataFrame
rolling_crps(forecasts: pd.DataFrame, window_forecasts: int = 10) -> pd.DataFrame

── Data loaders ───────────────────────────────────────────────────────────────
data/eodhd_fx.py:
    fetch_usdhkd_history(years=5, symbol="USDHKD.FOREX") -> pd.DataFrame
    latest_close(df) -> float
    daily_log_returns(df) -> pd.Series

data/rates.py:
    US_SERIES = "DTB3"
    HK_SERIES = "IR3TIB01HKM156N"   # name kept for cache compatibility; series is synthesised
    HK_SPREAD_PCT = -0.30
    HK_SPREAD_BETA = 0.05
    fetch_us_3m_yield() -> pd.Series
    fetch_hk_3m_yield(us_yield: pd.Series | None = None) -> pd.Series
    build_rate_differential() -> pd.DataFrame[us_yield, hk_yield, diff]

================================================================================
PART 6: NAVIGATION MAP (STREAMLIT)

Single Streamlit page with sidebar + two top-level tabs:

  Sidebar (always visible):
    - "USD/HKD Edge" title + tagline
    - Active model dropdown (13 options)
    - Number of paths slider (live)
    - Horizon (months) slider
    - Custom price buckets text input
    - Apply rate-differential equilibrium overlay checkbox
    - AI narrative summary checkbox (disabled if no ANTHROPIC key)
    - Data caption: "EODHD (USD/HKD) · FRED (DTB3 US 3m). HKD 3m rate is synthesised…"

  Status bar (above the tabs):
    - USD/HKD (last close)
    - US – HK 3m yield
    - Equilibrium fair value
    - Disequilibrium z-score (colour-coded)
    - Active model caption

  Tab "📈 Live forecast":
    - Probability Fan chart
    - 4 horizon distribution cards (1w / 1m / 3m / 6m)
    - Equilibrium Tracker chart + caption
    - Model parameter panel
    - Optional AI narrative summary

  Tab "🧪 Backtest":
    Sub-tab "Single model"
      - Configuration (start/end date, paths, step, horizons, overlay)
      - Run button → verdict banner + 5 metric cards + per-horizon table
      - Scatter + calibration plots + time-series with bands
      - "Set as live default" button
    Sub-tab "Pairwise comparison"
      - Model A / Model B dropdowns
      - Run button → winner banner with DM p-value
      - Side-by-side metrics table + two scatter plots + combined time-series
      - "Set winner as live default" button
    Sub-tab "All-model ranking"
      - "Models to include" multiselect (all 13 by default)
      - Run button → ranking table sorted by CRPS
      - Full DM p-value matrix
      - Rolling-CRPS chart
      - "Set best as live default" button

================================================================================
PART 7: KNOWN COMPLEXITY AREAS

engine/pricers.py (≈960 lines) — Largest module. Highest complexity:
  - Heston-family pricers (HestonPricer, BatesPricer, SVJJPricer, DoubleHestonPricer)
    share calibration code via engine/heston.py but each adds its own jump or second-factor
    overlay in simulate_paths.
  - CGMY and NIG use Cornish-Fisher cumulant matching rather than full MLE — adequate for
    relative model comparison but not for production option pricing.
  - Rough-Heston uses a simplified two-state Volterra approximation, not full hybrid scheme.
  - SVJJ correlates price and variance jumps; calibration uses cumulant matching on the
    Heston-filtered residuals.

main.py (≈950 lines) — All UI in one file. Hottest sections:
  - Sidebar + session-state plumbing (lines ~200–270), including the `pending_default_model`
    handoff that MUST run before the selectbox is instantiated to avoid StreamlitAPIException.
  - Cache-key construction for the backtest (SHA1 of returns + rate-diff + last_index_date).
  - Per-horizon card rendering with Plotly histograms and bucket tables.
  - The three backtest sub-tabs each have their own run/render block.

engine/backtest.py (≈490 lines) — DM machinery is the most subtle:
  - `_dm_pooled_by_horizon` is the preferred entry point; `diebold_mariano` is kept for
    legacy/simple single-horizon calls.
  - HAC bandwidth selection: bw = min(h_td, max(NW_rule, n // 4)). If the resulting var_d
    is unphysically small (< 0.1 · γ₀), it falls back to γ₀ (iid SE) — this avoids spurious
    near-infinite t-stats at small n.
  - The walk-forward loop recalibrates monthly; between recalibrations it reuses parameters
    but still re-simulates paths per forecast date.

engine/disequilibrium_fx.py — Equilibrium is fit ONCE on the full sample and reused as the
anchor across all backtest dates. The drift dynamics inside the simulator are path-dependent
(correct), but α / β / λ themselves contain in-sample lookahead. Documented as a known scope
limitation in replit.md and README.md. The fix is to refit on an expanding window keyed to
each backtest date.

Cache invalidation — Stale parquet files in cache/ (e.g. leftover from a prior FX symbol)
will be silently served if their mtime is within 12h. When changing currency / FRED series,
delete the relevant cache file or wait out the TTL.

================================================================================
PART 8: EXTERNAL API DEPENDENCY MAP

Provider     Secret              Used in                                 Required?
─────────────────────────────────────────────────────────────────────────────────
EODHD        EODHD               USD/HKD daily price history             yes
FRED         (none — public CSV) US 3-month T-bill yield (DTB3)          yes
HKMA         (n/a — synthesised) HKD 3-month rate (LERS synthesis)       no
Anthropic    ANTHROPIC           Optional AI narrative summary           no
Polygon      POLYGON             Reserved for future fallback data       no (unused)

Failure modes & fallbacks:
  - EODHD HTTP error AND stale cache present → silently serves the cached parquet.
  - EODHD HTTP error AND no cache           → raises RuntimeError, surfaced as a Streamlit error.
  - FRED CSV HTTP error AND stale cache     → silently serves the cached parquet.
  - FRED CSV HTTP error AND no cache        → raises through `requests.raise_for_status`.
  - Anthropic missing/invalid               → AI narrative checkbox is disabled, with a caption.
  - HK 3m series                            → always synthesised; no external dependency.

================================================================================
PART 9: PERFORMANCE & RUNTIME CHARACTERISTICS

Live forecast:
  - BS-RV / BS-GARCH                ~0.5 s end-to-end (calibration + 50k path simulation)
  - Merton-JD / Kou-JD              ~1–2 s
  - Heston / Bates / SVJJ           ~3–6 s (particle filter MLE + Euler simulation)
  - Double-Heston / Rough-Heston    ~4–8 s
  - VG / CGMY / NIG                 ~1–3 s (closed-form-ish increments)
  - SABR                            ~1–2 s

Backtest at defaults (1y / 1000 paths / step=10td / 3 horizons):
  - BS-RV / BS-GARCH                ~5–10 s
  - Merton-JD                       ~10–20 s
  - Heston-family                   ~30–60 s (heavy; not the live default for that reason)
  - All-model run                   minutes; only run on explicit user click

Caching behaviour:
  - Identical backtest inputs hit the SHA1-keyed cache and return effectively instantly.
  - Live calibration is cached on (model, returns_hash, annual_drift) so cycling models is fast.

================================================================================
PART 10: SCOPE LIMITATIONS (DOCUMENTED IN replit.md)

1. Equilibrium model (α, β, λ, σ_resid) is fit ONCE on full data and reused as the anchor
   across all backtest dates. Drift inside the simulator is path-dependent (correct), but the
   anchor itself contains in-sample lookahead. Future improvement: refit on an expanding window
   keyed to each backtest date for a fully out-of-sample backtest.

2. Several heavier pricers (CGMY, NIG, Rough-Heston, SVJJ) use Cornish-Fisher / cumulant
   matching or simplified calibration rather than full MLE — adequate for relative model
   comparison but not for production option pricing.

3. HKD 3-month rate is synthesised from the US rate under the HKMA Linked Exchange Rate System
   because FRED no longer publishes a free HK short-rate series under a stable ID. The synthesis
   uses fixed coefficients (HK_SPREAD_PCT = -0.30, HK_SPREAD_BETA = 0.05) calibrated to the
   long-run LERS relationship rather than fitted dynamically.

4. USD/HKD is a pegged regime (HKMA convertibility band 7.75–7.85). Stochastic-vol and jump
   models will produce structurally tight distributions; the app surfaces this caveat in the
   header caption and treats results as relative-model comparisons rather than free-float
   forecasts.

5. Single-user local app. No authentication, no per-user persistence, no analytics, no
   multi-tenant concerns. All state is either in Streamlit session state (lost on rerun
   without explicit handoff) or in the on-disk parquet cache.

================================================================================
PART 11: FINE-TUNING ENTRY POINTS (FOR CLAUDE)

Adding a new pricer:
  1. Subclass `Pricer` in engine/pricers.py with `name`, `label`, `params_spec`,
     `calibrate(...)`, `simulate_paths(...)`.
  2. Register in `PRICERS[name] = MyPricer()` and append to `PRICER_ORDER`.
  3. No UI changes needed; the sidebar dropdown and backtest multiselect both read from the
     registry automatically.

Changing the default live model:
  - main.py line ~211: `st.session_state["selected_model"] = "bs_rv"`.

Tuning backtest defaults:
  - main.py: the "Single model" sub-tab block — `n_paths`, `step_days`, `horizons`, window.
  - engine/backtest.py: `HORIZONS_TD` for adding/removing horizon labels.

Tuning the disequilibrium overlay:
  - engine/disequilibrium_fx.py: `rolling_window` for residual std, regression length for λ.
  - The per-step formula lives in `disequilibrium_drift_per_step` — modify carefully because
    it is used by both live and backtest paths.

Adjusting the HKD rate synthesis:
  - data/rates.py: `HK_SPREAD_PCT`, `HK_SPREAD_BETA`, `_synthetic_hk_yield`.

Improving DM small-sample behaviour:
  - engine/backtest.py: `_dm_pooled_by_horizon` — bandwidth rule and fallback threshold.

Adding a new horizon label (e.g. 9m):
  - engine/backtest.py: add to `HORIZONS_TD`.
  - main.py: add to the multiselect default list in the backtest sub-tabs and the live horizon
    list `LIVE_HORIZONS` if it should also drive a live card.

Switching to a fully out-of-sample equilibrium fit:
  - engine/backtest.py: inside `run_single_model_backtest`, refit `fit_equilibrium` on data
    truncated to each forecast date, and pass the as-of `EquilibriumModel` into
    `disequilibrium_drift_per_step`. Expect 2–5× backtest slowdown.

Adding a new data source (e.g. real HIBOR via paid API):
  - data/rates.py: replace `_synthetic_hk_yield` with a real fetcher; keep the same return
    signature (`pd.Series` with daily DatetimeIndex) so `build_rate_differential` is unchanged.

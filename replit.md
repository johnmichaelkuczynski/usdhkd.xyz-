# USD/JPY Edge

Streamlit web app that forecasts the future probability distribution of USD/JPY
using a user-selectable pricing model (13 supported) plus a rate-differential
disequilibrium overlay, and benchmarks model quality with a built-in backtest
module.

## Stack

- **Language**: Python 3.11
- **UI**: Streamlit + Plotly
- **Numerics**: NumPy, SciPy, pandas
- **Data**: EODHD (USD/JPY daily history) + FRED (DTB3 US 3m yield, IRSTCB01JPM156N JP short rate)
- **AI**: Anthropic Claude (optional narrative summary)
- **Caching**: Local parquet cache in `cache/` (12h TTL) + in-process `st.cache_data` for backtests

## Secrets required

- `EODHD` — EODHD API key (USD/JPY history). The code also accepts `EODHD_API_KEY`.
- `ANTHROPIC` — optional, only used when "AI narrative summary" is enabled. Also accepts `ANTHROPIC_API_KEY`.
- `POLYGON` — reserved for a future fallback data source.

## Run

The `Start application` workflow runs:

```bash
streamlit run main.py --server.port 5000 --server.address 0.0.0.0
```

## File layout

- `main.py` — Streamlit app: status bar, sidebar model selector (drives live forecasts),
  Live / Backtest tabs, and three backtest sub-tabs (Single / Pairwise / All-model).
- `engine/pricers.py` — Modular `Pricer` interface + 13 implementations
  (BS-RV, BS-GARCH, Merton-JD, Kou-JD, Heston, Bates, SVJJ, Double-Heston, Rough-Heston,
  VG, CGMY, NIG, SABR). Each declares `param_spec`, `calibrate(log_returns, dt, drift)`,
  and `simulate_paths(...)` accepting an `extra_drift_fn` for the disequilibrium overlay.
  Registered in `PRICERS` / `PRICER_ORDER` / `pricer_choices()`.
- `engine/backtest.py` — Walk-forward backtest with monthly recalibration. Computes
  CRPS, log score, coverage (50/70/95), MAE, calibration table, Diebold-Mariano test
  with horizon-aware HAC variance pooling, pairwise winner, all-model p-value matrix,
  rolling CRPS, and a `calibration_verdict()` summary banner.
- `engine/heston.py` — Heston SDE, particle-filter MLE, full-truncation Euler simulator
  (used by both the Heston pricer and as a building block for SVJJ / Bates / Double-Heston).
- `engine/disequilibrium_fx.py` — OLS equilibrium model `usdjpy = α + β·(US3m − JP3m)`,
  residual std, lambda estimate, and `disequilibrium_drift_per_step(s_t, ...)` used by
  both live and backtest paths so the overlay is path-dependent.
- `engine/monte_carlo.py` — Quantile + horizon-stat helpers shared by all pricers.
- `data/eodhd_fx.py` — USD/JPY daily history loader with parquet cache.
- `data/rates.py` — US/JP yields from FRED with parquet cache.
- `cache/` — local on-disk cache (parquet).
- `.streamlit/config.toml` — Streamlit server config (port 5000, headless, 0.0.0.0).

## Live forecast

- Active model is chosen from the sidebar dropdown.
- Default sim is 50,000 paths over 6 months, sliced for 1w / 1m / 3m / 6m horizons.
- Custom price buckets are user-editable in the sidebar, e.g. `145, 150, 155, 160`.
- The disequilibrium overlay adds `−λ·(s_t − fair(t)) · dt + σ_resid·dW` per step so
  the live and backtest paths share identical mean-reversion dynamics.

## Backtest module

- **Defaults** tuned for interactive UX: 1-year window, 1000 paths, step = 10 trading days,
  horizons {1w, 1m, 3m}. Heavier configurations are available in the sidebar but slower.
- **Caching** keyed by SHA1 of (returns bytes + rate-diff bytes + last index date), model,
  date range, horizons, paths, step, overlay flag — re-runs serve instantly.
- **Single-model** tab shows verdict banner, 5 metric cards, per-horizon table,
  scatter + calibration plots, time-series with bands, and a "Set as live default" button.
- **Pairwise** tab shows winner banner with Diebold-Mariano p-value, side-by-side metrics
  table, two scatter plots, combined time-series, and "Set winner as live default".
- **All-model** tab runs all selected models, ranks them by CRPS, shows full DM p-value
  matrix and a rolling-CRPS chart.
- **DM test** uses `_dm_pooled_by_horizon`: per-horizon HAC variance with a Newey-West
  rule-of-thumb bandwidth and a fallback to the iid variance when the HAC sum is
  suspiciously small (avoids spurious near-infinite t-stats at small n).
- **"Set as live default"** writes a `pending_default_model` key consumed at the top of
  the next rerun before the dropdown widget is instantiated (avoids
  `StreamlitAPIException` from mutating widget-keyed state).

## Known scope limitations

- The equilibrium model (α, β, λ, σ_resid) is fit once on the full data and reused as the
  anchor across all backtest dates. The drift dynamics inside the simulator are
  path-dependent (correct), but the anchor itself contains in-sample lookahead. A future
  improvement is to refit the equilibrium on an expanding window keyed to each backtest
  date for a fully out-of-sample backtest.
- Several heavier pricers (CGMY, NIG, Rough-Heston, SVJJ) use Cornish-Fisher / cumulant
  matching or simplified calibration rather than full MLE — adequate for relative model
  comparison but not for production option pricing.

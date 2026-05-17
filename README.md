# 🧠 USD/HKD EDGE

**Multi-Model Probability Forecaster and Statistical Backtesting Suite for the USD/HKD Exchange Rate**

---

## 🧩 Overview

USD/HKD Edge is a statistically defensible probability-forecasting platform for the USD/HKD spot rate. It calibrates one of thirteen pricing models — ranging from classical Black-Scholes through stochastic-volatility, jump-diffusion, and pure-jump Lévy processes — to live USD/HKD history, then projects the full distribution of future prices across multiple horizons.

Unlike point-forecast trading tools that produce a single "expected" number, USD/HKD Edge is built around a strict operating principle: every forecast is a distribution, every model is held accountable, and every metric is reported. The Backtest module runs walk-forward, monthly-recalibrated evaluations against actual history, then ranks the models head-to-head using Diebold-Mariano hypothesis tests, CRPS, log scores, and full coverage diagnostics. No padding, no hedging, no rhetorical "the model thinks…" — just the numbers.

---

## 👥 Who It's For

- **Quantitative researchers** -- need to benchmark stochastic-volatility, jump, and pure-jump models against each other on a single, well-defined FX series before committing to one in production
- **FX desk strategists** -- need calibrated probability distributions, not point forecasts, to size USD/HKD options trades and assess tail risk inside the HKMA convertibility band
- **Risk managers** -- need rigorous coverage diagnostics (50/70/95) and calibration tables to verify that a model's stated probabilities match realised frequencies
- **Academics and graduate students** -- need a clean, reproducible reference implementation of thirteen pricing models with a shared calibration and simulation interface
- **Anyone forecasting a pegged or band-constrained currency** -- who needs to understand how classical stochastic models behave when the underlying is structurally mean-reverting

---

## ⚙️ Core Capabilities

- **Thirteen Pricing Models** -- BS Realized-Vol, BS-GARCH(1,1), Merton Jump-Diffusion, Kou Jump-Diffusion, Heston, Bates, SVJJ, Double-Heston, Rough-Heston, Variance Gamma, CGMY, Normal Inverse Gaussian, and SABR. Each implements a single `Pricer` interface — `calibrate(log_returns, dt, drift)` plus `simulate_paths(...)` — so any model can be swapped in for live or backtest use without changing surrounding code.

- **Live Probability Fan** -- One-click 50,000-path Monte Carlo simulation over six months, sliced for 1-week, 1-month, 3-month, and 6-month horizons. Renders quantile-banded fan charts, terminal-distribution histograms, custom user-defined price bucket probabilities, and per-horizon directional statistics.

- **Rate-Differential Disequilibrium Overlay** -- An optional drift adjustment fits an OLS equilibrium model `USDHKD = α + β·(US₃ₘ − HK₃ₘ)`, computes the current z-score, and tilts simulated paths toward fair value with strength `λ` per step. The same path-dependent overlay is applied identically in both live forecasts and backtests.

- **Walk-Forward Backtest Engine** -- Monthly recalibration, configurable date range, paths, step size, and horizons. Produces per-date forecast distributions with CRPS, log score, coverage rates (50/70/95), MAE, calibration tables, and time-series of forecasts versus realised outcomes. Results are SHA1-cached on (returns, rates, model, settings) so re-runs return instantly.

- **Single-Model Backtest Tab** -- A colour-coded calibration verdict banner (WELL-CALIBRATED, BIASED, OVERCONFIDENT, UNDERCONFIDENT, PARTIALLY), five headline metric cards, a per-horizon summary table, scatter and calibration plots, a forecast time-series with bands, and a one-click "Set as live default" button.

- **Pairwise Comparison Tab** -- Direct head-to-head between any two models. Reports the CRPS winner and Diebold-Mariano p-value with horizon-aware HAC variance pooling, plus a side-by-side metrics table with a "Better" column, twin scatter plots, and a combined time-series chart.

- **All-Model Ranking Tab** -- Runs every selected model in one pass, ranks them by CRPS, renders a full pairwise Diebold-Mariano p-value matrix, and plots rolling CRPS for each model so users can see when one regime favoured one model over another.

- **Diebold-Mariano with Horizon-Aware HAC** -- The DM test uses per-horizon Newey-West HAC variance pooling with a rule-of-thumb bandwidth and a defensible fallback to the iid variance when the sample HAC sum is suspiciously small. Avoids the spurious near-infinite t-statistics that small-n DM tests routinely produce.

- **Optional AI Narrative Summary** -- A short (≤200 word) Anthropic-generated narrative that interprets the active model, the current disequilibrium signal, and the resulting horizon probabilities in plain language. Strictly optional; the quantitative output is the source of truth.

- **Data Pipeline** -- USD/HKD daily history from EODHD, US 3-month T-bill yield from FRED, and a synthetic HKD 3-month rate derived from the US rate under the HKMA Linked Exchange Rate System (HIBOR ≈ US + α + β·(US − mean(US))). All sources are parquet-cached locally for twelve hours.

---

## 🚀 What Makes It Different

- **It is honest about uncertainty** -- The output is never a single number. Every horizon produces a full distribution, quantile bands, bucket probabilities, and directional probabilities — because that is what a pricing model actually produces, and anything less is editorial.

- **It is honest about the peg** -- USD/HKD trades inside the HKMA convertibility band of 7.75–7.85. The app says so up front and treats results as relative-model comparisons rather than free-float forecasts. Stochastic-vol and jump models will produce structurally tight distributions, and the app does not pretend otherwise.

- **Every model is held accountable** -- A model is only as good as its calibration on real history. The Backtest module runs a walk-forward evaluation against actual outcomes for every horizon, so the ranking shown in the All-Model tab reflects measured performance, not theoretical elegance.

- **Statistically defensible hypothesis testing** -- The Diebold-Mariano implementation accounts for horizon overlap with HAC variance pooling and explicitly handles the small-sample edge cases that cause naive DM implementations to report meaningless p-values. When the test cannot be trusted at the sample size, the app says so instead of returning a confident lie.

- **Live and backtest dynamics are identical** -- The disequilibrium overlay uses the same path-dependent drift function `−λ·(s_t − fair(t))·dt + σ_resid·dW` in both live simulation and backtest replay. There is no calibration-time advantage given to live forecasts that the backtest doesn't also receive.

- **One interface, thirteen models** -- Every pricer declares its `param_spec`, calibrates from the same `(log_returns, dt, drift)` signature, and simulates paths with the same `extra_drift_fn` plug-in for the overlay. Adding a fourteenth model is a single new class and one line in the registry.

- **Reproducible and cached** -- Backtest results are SHA1-keyed on the actual byte content of the input series plus all configuration. Identical inputs always return identical outputs, and a re-run with the same configuration is served instantly from cache.

- **No silent fallbacks** -- When a data source fails, the app raises an explicit error rather than substituting placeholder values. When a synthetic HKD rate is used in place of an unavailable FRED series, the sidebar caption says so. The user always knows what they are looking at.

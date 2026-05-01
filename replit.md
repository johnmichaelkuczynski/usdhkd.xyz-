# USD/JPY Edge

Streamlit web app that forecasts the future probability distribution of USD/JPY
using a Heston stochastic-volatility model with a rate-differential
disequilibrium overlay.

## Stack

- **Language**: Python 3.11
- **UI**: Streamlit + Plotly
- **Numerics**: NumPy, SciPy, pandas
- **Data**: EODHD (USD/JPY daily history) + FRED (DTB3 US 3m yield, IRSTCB01JPM156N JP short rate)
- **AI**: Anthropic Claude (optional narrative summary)
- **Caching**: Local parquet cache in `cache/` (12h TTL)

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

- `main.py` — Streamlit app (status bar, fan chart, horizon cards, equilibrium tracker, parameter panel, AI narrative)
- `engine/heston.py` — Heston SDE, particle-filter MLE calibration, full-truncation Euler simulator
- `engine/disequilibrium_fx.py` — OLS equilibrium model: `usdjpy = α + β·(US3m − JP3m)`, residual std, lambda estimate
- `engine/monte_carlo.py` — Disequilibrium-adjusted Monte Carlo, fan-chart quantiles, horizon stats + bucket probabilities
- `data/eodhd_fx.py` — USD/JPY daily history loader with parquet cache
- `data/rates.py` — US/JP yields from FRED with parquet cache
- `cache/` — local on-disk cache (parquet)
- `.streamlit/config.toml` — Streamlit server config (port 5000, headless, 0.0.0.0)

## Notes

- The app simulates 50,000 paths by default (configurable in the sidebar) over 6 months and slices that single MC run for the 1w / 1m / 3m horizons.
- Custom price buckets are user-editable in the sidebar, e.g. `145, 150, 155, 160`.
- Heston MLE uses a small-particle bootstrap filter; calibration is bounded and falls back to historical-vol parameters if it fails.

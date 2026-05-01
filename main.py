"""USD/JPY Edge — Streamlit app.

Forecasts the future probability distribution of USD/JPY using a
disequilibrium-adjusted Heston stochastic-volatility model.
"""
from __future__ import annotations

import os
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.eodhd_fx import fetch_usdjpy_history, latest_close, daily_log_returns
from data.rates import build_rate_differential
from engine.disequilibrium_fx import fit_equilibrium
from engine.heston import HestonParams, calibrate_heston, TRADING_DAYS
from engine.monte_carlo import (
    calendar_to_trading_steps,
    fan_chart_quantiles,
    simulate_disequilibrium_paths,
    summarize_terminals,
)

st.set_page_config(page_title="USD/JPY Edge", layout="wide")

HORIZONS = [("1 week", 7), ("1 month", 30), ("3 months", 90), ("6 months", 180)]
DEFAULT_BUCKETS = "145, 150, 155, 160"


# ---------------------------------------------------------------------------
# Cached data + model
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_fx_history() -> pd.DataFrame:
    return fetch_usdjpy_history(years=5)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_rate_differential() -> pd.DataFrame:
    return build_rate_differential()


@st.cache_resource(show_spinner=False)
def calibrate_model(close_series_values: tuple, annual_drift: float) -> dict:
    """Calibrate Heston on log returns. Cached on the values tuple."""
    closes = np.asarray(close_series_values, dtype=np.float64)
    log_rets = np.diff(np.log(closes))
    params, ok = calibrate_heston(log_rets, annual_drift=annual_drift, n_particles=250)
    return {"params": params, "converged": ok}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_buckets(text: str) -> List[float]:
    out: List[float] = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return sorted(set(out))


def z_color(z: float) -> str:
    az = abs(z)
    if az < 1.0:
        return "#10b981"  # green
    if az < 2.0:
        return "#f59e0b"  # yellow
    return "#ef4444"      # red


def z_label(z: float) -> str:
    az = abs(z)
    if az < 1.0:
        return "Near equilibrium"
    if az < 2.0:
        return "Stretched"
    return "Very stretched"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("USD/JPY Edge")
st.sidebar.caption("Heston SV + rate-differential equilibrium overlay")

bucket_text = st.sidebar.text_input(
    "Custom price buckets (comma-separated)",
    value=DEFAULT_BUCKETS,
    help="Cut-points used for bucket probabilities, e.g. '145, 150, 155, 160'.",
)
n_paths = st.sidebar.select_slider(
    "Monte Carlo paths",
    options=[5_000, 10_000, 25_000, 50_000, 100_000],
    value=50_000,
)
recalibrate = st.sidebar.button("Recalibrate model", help="Re-runs Heston MLE on the latest history")
if recalibrate:
    calibrate_model.clear()
    st.cache_data.clear()
    st.rerun()

show_narrative = st.sidebar.checkbox(
    "Generate AI narrative summary (Anthropic)",
    value=False,
    help="Uses ANTHROPIC_API_KEY to produce a written interpretation of the forecast.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: EODHD (USD/JPY daily) · FRED (DTB3, IRSTCB01JPM156N). "
    "Cached locally for 12 hours."
)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

st.title("USD/JPY Edge — Probability Forecasts")

try:
    fx = load_fx_history()
except Exception as e:
    st.error(f"Could not load USD/JPY history from EODHD: {e}")
    st.stop()

try:
    rates = load_rate_differential()
except Exception as e:
    st.warning(f"Could not load fresh rate data, using last known: {e}")
    rates = pd.DataFrame()

if rates.empty:
    st.error("Rate differential data is unavailable and no cache exists. Cannot fit equilibrium model.")
    st.stop()

# Align FX close and rate diff on common dates
fx_close = fx["close"].copy()
fx_close.index = pd.to_datetime(fx_close.index).tz_localize(None)
rates.index = pd.to_datetime(rates.index).tz_localize(None)
joined = pd.concat(
    [fx_close.rename("usdjpy"), rates["diff"].rename("diff")],
    axis=1,
).dropna()
if len(joined) < 250:
    st.error("Not enough overlapping FX + rate data to fit the equilibrium model.")
    st.stop()

# Fit equilibrium
eq = fit_equilibrium(joined["usdjpy"], joined["diff"], rolling_window=252)

# Calibrate Heston
log_rets = daily_log_returns(fx).to_numpy()
annual_drift_estimate = float(np.mean(log_rets) * TRADING_DAYS)
with st.spinner("Calibrating Heston model (particle-filter MLE)…"):
    cal = calibrate_model(tuple(fx["close"].to_numpy().tolist()), annual_drift_estimate)
params: HestonParams = cal["params"]
converged: bool = cal["converged"]

if not converged:
    st.warning(
        "Heston MLE did not fully converge — using fallback parameters derived from realised volatility."
    )

current_spot = latest_close(fx)
us_yield = float(rates["us_yield"].iloc[-1])
jp_yield = float(rates["jp_yield"].iloc[-1])
rate_diff = float(rates["diff"].iloc[-1])


# ---------------------------------------------------------------------------
# Top status bar
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.3, 1.3])
c1.metric("USD/JPY (last close)", f"{current_spot:,.3f}")
c2.metric("US – JP 3m yield", f"{rate_diff:+.2f}%", help=f"US: {us_yield:.2f}%   JP: {jp_yield:.2f}%")
with c3:
    st.markdown(
        f"<div style='font-size:0.85rem;color:#666'>Equilibrium fair value</div>"
        f"<div style='font-size:1.6rem;font-weight:600'>{eq.equilibrium:,.2f}</div>"
        f"<div style='font-size:0.8rem;color:#888'>α={eq.alpha:.2f}, β={eq.beta:.2f}</div>",
        unsafe_allow_html=True,
    )
with c4:
    color = z_color(eq.z_score)
    st.markdown(
        f"<div style='font-size:0.85rem;color:#666'>Disequilibrium z-score</div>"
        f"<div style='font-size:1.6rem;font-weight:600;color:{color}'>{eq.z_score:+.2f}σ</div>"
        f"<div style='font-size:0.8rem;color:{color}'>{z_label(eq.z_score)}</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# Run Monte Carlo for the longest horizon (use sub-paths for shorter horizons)
# ---------------------------------------------------------------------------

LONGEST_DAYS = 180
with st.spinner(f"Simulating {n_paths:,} Monte Carlo paths over 180 days…"):
    s_paths, _v_paths = simulate_disequilibrium_paths(
        params=params,
        s0=current_spot,
        horizon_calendar_days=LONGEST_DAYS,
        n_paths=int(n_paths),
        equilibrium=eq.equilibrium,
        residual_std=eq.residual_std,
        lambda_daily=eq.lambda_,
        annual_drift=annual_drift_estimate,
        seed=42,
    )

n_steps_total = s_paths.shape[0] - 1  # trading-day steps to 6 months
# Build calendar-day axis (linearly spaced from today to today + 180 calendar days)
last_date = fx.index[-1]
future_dates = pd.date_range(
    start=last_date,
    periods=n_steps_total + 1,
    freq=pd.tseries.offsets.BDay(),
)


# ---------------------------------------------------------------------------
# Probability Fan Chart
# ---------------------------------------------------------------------------

st.subheader("Probability Fan — USD/JPY forecast distribution")

QS = (0.025, 0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95, 0.975)
bands = fan_chart_quantiles(s_paths, QS)
median_path = bands[QS.index(0.5)]

# Historical context: last 365 calendar days
hist = fx["close"].last("365D")

fig_fan = go.Figure()

# Confidence bands (drawn from widest to narrowest)
band_specs = [
    (0.025, 0.975, "rgba(99,102,241,0.10)", "95% CI"),
    (0.05, 0.95, "rgba(99,102,241,0.15)", "90% CI"),
    (0.15, 0.85, "rgba(99,102,241,0.22)", "70% CI"),
    (0.25, 0.75, "rgba(99,102,241,0.30)", "50% CI"),
]
for lo, hi, color, name in band_specs:
    lo_arr = bands[QS.index(lo)]
    hi_arr = bands[QS.index(hi)]
    fig_fan.add_trace(
        go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(hi_arr) + list(lo_arr[::-1]),
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            name=name,
            hoverinfo="skip",
            showlegend=True,
        )
    )

fig_fan.add_trace(
    go.Scatter(
        x=future_dates,
        y=median_path,
        mode="lines",
        line=dict(color="#4f46e5", width=2),
        name="Forecast median",
    )
)
fig_fan.add_trace(
    go.Scatter(
        x=hist.index,
        y=hist.values,
        mode="lines",
        line=dict(color="#111827", width=1.6),
        name="Historical close",
    )
)
fig_fan.add_hline(
    y=eq.equilibrium,
    line=dict(color="#10b981", width=1, dash="dash"),
    annotation_text=f"Equilibrium {eq.equilibrium:.2f}",
    annotation_position="top left",
)
fig_fan.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="USD/JPY",
    xaxis_title=None,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_fan, use_container_width=True)


# ---------------------------------------------------------------------------
# Horizon distribution cards (2x2)
# ---------------------------------------------------------------------------

st.subheader("Horizon distributions")

buckets = parse_buckets(bucket_text) or parse_buckets(DEFAULT_BUCKETS)

# For each horizon, slice s_paths at the appropriate trading-day index
horizon_stats = []
for label, days in HORIZONS:
    idx = min(calendar_to_trading_steps(days), s_paths.shape[0] - 1)
    terminals = s_paths[idx]
    stats = summarize_terminals(
        horizon_days=days,
        n_steps=idx,
        terminals=terminals,
        s0=current_spot,
        buckets=buckets,
    )
    horizon_stats.append((label, stats))


def render_horizon_card(label: str, stats) -> None:
    st.markdown(f"#### {label}  \n<span style='color:#666;font-size:0.85rem'>{stats.horizon_days} calendar days</span>",
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=stats.terminals,
        nbinsx=60,
        marker=dict(color="#6366f1", line=dict(color="#4338ca", width=0.3)),
        opacity=0.85,
        name="Terminal rate",
    ))
    for q, color, name in [
        (stats.p05, "#ef4444", "5%"),
        (stats.median, "#111827", "median"),
        (stats.p95, "#ef4444", "95%"),
    ]:
        fig.add_vline(x=q, line=dict(color=color, width=1, dash="dot"),
                      annotation_text=f"{name} {q:.2f}", annotation_position="top")
    fig.add_vline(x=current_spot, line=dict(color="#10b981", width=1.2),
                  annotation_text=f"spot {current_spot:.2f}", annotation_position="bottom")
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title="USD/JPY (terminal)",
        yaxis_title=None,
        bargap=0.02,
    )
    st.plotly_chart(fig, use_container_width=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Mean", f"{stats.mean:.2f}")
    sc2.metric("Median", f"{stats.median:.2f}")
    sc3.metric("5th pct", f"{stats.p05:.2f}")
    sc4.metric("95th pct", f"{stats.p95:.2f}")

    # Bucket probability table
    if stats.bucket_probs:
        bdf = pd.DataFrame(
            [{"Bucket": k, "Probability": f"{v * 100:.1f}%"} for k, v in stats.bucket_probs.items()]
        )
        st.dataframe(bdf, use_container_width=True, hide_index=True)

    # "Most likely move" summary line — pick the highest-probability bucket
    if stats.bucket_probs:
        top_bucket = max(stats.bucket_probs.items(), key=lambda kv: kv[1])
        tail_above = next(
            (kv for kv in stats.bucket_probs.items() if kv[0].startswith("≥")),
            None,
        )
        tail_below = next(
            (kv for kv in stats.bucket_probs.items() if kv[0].startswith("<")),
            None,
        )
        bits = [f"**{top_bucket[1] * 100:.0f}%** probability USD/JPY {top_bucket[0]} in {stats.horizon_days} days"]
        if tail_above:
            bits.append(f"{tail_above[1] * 100:.0f}% {tail_above[0]}")
        if tail_below:
            bits.append(f"{tail_below[1] * 100:.0f}% {tail_below[0]}")
        st.caption(" · ".join(bits))

    direction_a, direction_b = st.columns(2)
    direction_a.caption(f"P(JPY appreciates) = {stats.p_jpy_appreciation * 100:.1f}%")
    direction_b.caption(f"P(USD appreciates) = {stats.p_usd_appreciation * 100:.1f}%")


row1 = st.columns(2)
with row1[0]:
    render_horizon_card(*horizon_stats[0])
with row1[1]:
    render_horizon_card(*horizon_stats[1])
row2 = st.columns(2)
with row2[0]:
    render_horizon_card(*horizon_stats[2])
with row2[1]:
    render_horizon_card(*horizon_stats[3])


# ---------------------------------------------------------------------------
# Equilibrium tracker
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Equilibrium Tracker")

et = eq.fitted.last("3Y").copy()

fig_eq = make_subplots(specs=[[{"secondary_y": True}]])
fig_eq.add_trace(
    go.Scatter(x=et.index, y=et["usdjpy"], name="Actual USD/JPY",
               line=dict(color="#111827", width=1.6)),
    secondary_y=False,
)
fig_eq.add_trace(
    go.Scatter(x=et.index, y=et["equilibrium"], name="Model equilibrium",
               line=dict(color="#10b981", width=1.4, dash="dash")),
    secondary_y=False,
)
fig_eq.add_trace(
    go.Scatter(x=et.index, y=et["z"], name="z-score",
               line=dict(color="#ef4444", width=1.0)),
    secondary_y=True,
)
fig_eq.add_hline(y=0, line=dict(color="#bbb", width=0.6), secondary_y=True)
fig_eq.update_yaxes(title_text="USD/JPY", secondary_y=False)
fig_eq.update_yaxes(title_text="z-score", secondary_y=True)
fig_eq.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=10, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_eq, use_container_width=True)


# ---------------------------------------------------------------------------
# Parameter display
# ---------------------------------------------------------------------------

st.subheader("Model parameters")
pc = st.columns(6)
pc[0].metric("κ (mean-rev speed)", f"{params.kappa:.3f}")
pc[1].metric("θ (long-run vol)", f"{params.long_run_vol_annual * 100:.2f}%")
pc[2].metric("ξ (vol-of-vol)", f"{params.xi:.3f}")
pc[3].metric("ρ (corr)", f"{params.rho:+.3f}")
pc[4].metric("v₀ (current vol)", f"{params.current_vol_annual * 100:.2f}%")
pc[5].metric("λ (daily reversion)", f"{eq.lambda_ * 1e4:.2f} bp/σ")

st.caption(
    f"Equilibrium model: USDJPY = {eq.alpha:.2f} + {eq.beta:.2f} × (US₃ₘ − JP₃ₘ).  "
    f"Residual σ = {eq.residual_std:.2f}.  "
    f"Heston MLE {'converged' if converged else 'used fallback parameters'}."
)


# ---------------------------------------------------------------------------
# Optional Anthropic narrative
# ---------------------------------------------------------------------------

if show_narrative:
    st.markdown("---")
    st.subheader("AI narrative summary")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC")
    if not api_key:
        st.info("Set ANTHROPIC_API_KEY (or ANTHROPIC) to enable the AI narrative.")
    else:
        try:
            import anthropic

            facts = {
                "spot": current_spot,
                "us_yield": us_yield,
                "jp_yield": jp_yield,
                "rate_diff": rate_diff,
                "equilibrium": eq.equilibrium,
                "z_score": eq.z_score,
                "alpha": eq.alpha,
                "beta": eq.beta,
                "lambda_daily": eq.lambda_,
                "heston": params.as_dict(),
                "horizons": [
                    {
                        "label": label,
                        "days": s.horizon_days,
                        "median": s.median,
                        "p05": s.p05,
                        "p95": s.p95,
                        "p_jpy_appreciation": s.p_jpy_appreciation,
                        "buckets": s.bucket_probs,
                    }
                    for label, s in horizon_stats
                ],
            }
            client = anthropic.Anthropic(api_key=api_key)
            with st.spinner("Asking Claude for a narrative…"):
                msg = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=600,
                    messages=[{
                        "role": "user",
                        "content": (
                            "You are an FX strategist. Write a concise, neutral interpretation "
                            "(max 200 words) of this USD/JPY forecast. Do not give trading advice. "
                            "Reference the equilibrium z-score and the multi-horizon distribution. "
                            "Facts (JSON): " + str(facts)
                        ),
                    }],
                )
                text = "".join(getattr(b, "text", "") for b in msg.content)
                st.write(text)
        except Exception as e:
            st.warning(f"Narrative generation failed: {e}")


st.caption(
    "This app calibrates a stochastic-volatility model to historical FX returns. "
    "It is not investment advice. Probabilities reflect the model's assumptions, "
    "not certainties about the future."
)

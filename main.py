"""USD/JPY Edge — Streamlit app.

Forecasts the future probability distribution of USD/JPY using one of 13
selectable pricing models, optionally adjusted by a rate-differential
equilibrium overlay. Includes a Backtest module with single-model,
pairwise, and all-model evaluation tabs.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.eodhd_fx import fetch_usdjpy_history, latest_close, daily_log_returns
from data.rates import build_rate_differential
from engine.disequilibrium_fx import (
    EquilibriumModel,
    fit_equilibrium,
    disequilibrium_drift_per_step,
)
from engine.pricers import (
    PRICERS,
    PRICER_ORDER,
    TRADING_DAYS,
    get_pricer,
    pricer_choices,
)
from engine.backtest import (
    HORIZONS_TD,
    ModelBacktest,
    all_model_pvalue_matrix,
    calibration_verdict,
    pairwise_winner,
    rolling_crps,
    run_single_model_backtest,
)
from engine.monte_carlo import calendar_to_trading_steps, summarize_terminals

st.set_page_config(page_title="USD/JPY Edge", layout="wide")

LIVE_HORIZONS: List[Tuple[str, int]] = [("1 week", 7), ("1 month", 30), ("3 months", 90), ("6 months", 180)]
BACKTEST_HORIZON_LABELS = {"1w": "1 week", "2w": "2 weeks", "1m": "1 month",
                           "3m": "3 months", "6m": "6 months"}
DEFAULT_BUCKETS = "145, 150, 155, 160"
QS = (0.025, 0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95, 0.975)


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
def calibrate_pricer(model_name: str, returns_key: tuple, annual_drift: float) -> dict:
    """Cache pricer calibration on the closes-tuple key."""
    pricer = get_pricer(model_name)
    closes = np.asarray(returns_key, dtype=np.float64)
    log_rets = np.diff(np.log(closes))
    params = pricer.calibrate(log_rets, dt=1.0 / TRADING_DAYS, annual_drift=annual_drift)
    return {"params": params}


@st.cache_data(show_spinner=False)
def cached_backtest(
    model_name: str,
    returns_hash: str,
    start_iso: str,
    end_iso: str,
    horizons_tuple: Tuple[str, ...],
    n_paths: int,
    step_days: int,
    recal_every_days: int,
    use_overlay: bool,
    extra_lambda: float,
    eq_signature: str,
) -> dict:
    """Wrapper so Streamlit caches a backtest result. We pass a returns hash and an
    equilibrium signature so cache invalidates when inputs change."""
    fx = load_fx_history()
    rates = load_rate_differential()
    eq = fit_equilibrium(fx["close"], rates["diff"], rolling_window=252)
    res = run_single_model_backtest(
        model=model_name,
        fx=fx,
        start=pd.Timestamp(start_iso),
        end=pd.Timestamp(end_iso),
        horizons=list(horizons_tuple),
        n_paths=n_paths,
        step_days=step_days,
        recal_every_days=recal_every_days,
        equilibrium=eq if use_overlay else None,
        extra_drift_lambda=extra_lambda,
        use_eq_overlay=use_overlay,
    )
    return {
        "model": res.model,
        "forecasts": res.forecasts,
        "summary": res.summary,
        "overall": res.overall,
        "calibration": res.calibration,
        "params_history": res.params_history,
        "runtime_seconds": res.runtime_seconds,
    }


def restore_backtest(d: dict) -> ModelBacktest:
    return ModelBacktest(
        model=d["model"],
        forecasts=d["forecasts"],
        summary=d["summary"],
        overall=d["overall"],
        calibration=d["calibration"],
        params_history=d["params_history"],
        runtime_seconds=d["runtime_seconds"],
    )


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
        return "#10b981"
    if az < 2.0:
        return "#f59e0b"
    return "#ef4444"


def z_label(z: float) -> str:
    az = abs(z)
    if az < 1.0:
        return "Near equilibrium"
    if az < 2.0:
        return "Stretched"
    return "Very stretched"


def make_extra_drift_fn(eq: EquilibriumModel, use_overlay: bool):
    if not use_overlay or eq.lambda_ == 0 or eq.residual_std <= 0:
        return None

    def fn(s_at_t: np.ndarray) -> np.ndarray:
        return disequilibrium_drift_per_step(
            s_at_t, eq.equilibrium, eq.residual_std, eq.lambda_,
        )
    return fn


def simulate_live_paths(
    model_name: str,
    params: Dict[str, float],
    s0: float,
    n_steps: int,
    n_paths: int,
    annual_drift: float,
    eq: EquilibriumModel,
    use_overlay: bool,
) -> np.ndarray:
    pricer = get_pricer(model_name)
    extra_fn = make_extra_drift_fn(eq, use_overlay)
    return pricer.simulate_paths(
        params, s0, n_steps, n_paths,
        dt=1.0 / TRADING_DAYS,
        annual_drift=annual_drift,
        extra_drift_fn=extra_fn,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("USD/JPY Edge")
st.sidebar.caption("13-model probability forecaster + backtest module")

# initialise session state defaults
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "heston"
if "use_overlay" not in st.session_state:
    st.session_state["use_overlay"] = True

# A button-driven request to change the active model is queued via this key
# (we cannot directly write to a widget-bound key after the widget is created).
if "pending_default_model" in st.session_state:
    pending = st.session_state.pop("pending_default_model")
    st.session_state["selected_model"] = pending

choices = pricer_choices()
choice_labels = {name: label for name, label in choices}
choice_keys = [name for name, _ in choices]

selected_model = st.sidebar.selectbox(
    "Active model (drives live forecasts)",
    options=choice_keys,
    format_func=lambda k: choice_labels.get(k, k),
    key="selected_model",
)

use_overlay = st.sidebar.checkbox(
    "Apply rate-differential equilibrium overlay",
    key="use_overlay",
    help="Tilts the drift toward fair value when USD/JPY is dislocated.",
)

bucket_text = st.sidebar.text_input(
    "Custom price buckets (comma-separated)",
    value=DEFAULT_BUCKETS,
    help="Cut-points used for bucket probabilities, e.g. '145, 150, 155, 160'.",
)
n_paths = st.sidebar.select_slider(
    "Live Monte Carlo paths",
    options=[5_000, 10_000, 25_000, 50_000, 100_000],
    value=25_000,
)

if st.sidebar.button("Recalibrate (clear caches)"):
    calibrate_pricer.clear()
    st.cache_data.clear()
    st.rerun()

show_narrative = st.sidebar.checkbox(
    "Generate AI narrative summary (Anthropic)",
    value=False,
    help="Uses ANTHROPIC_API_KEY to produce a written interpretation.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: EODHD (USD/JPY) · FRED (DTB3, IRSTCB01JPM156N). Cached locally for 12 hours."
)


# ---------------------------------------------------------------------------
# Load data + fit equilibrium (shared across all tabs)
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
    st.error("Rate differential data is unavailable. Cannot fit equilibrium model.")
    st.stop()

fx_close = fx["close"].copy()
fx_close.index = pd.to_datetime(fx_close.index).tz_localize(None)
rates.index = pd.to_datetime(rates.index).tz_localize(None)
joined = pd.concat([fx_close.rename("usdjpy"), rates["diff"].rename("diff")], axis=1).dropna()
if len(joined) < 250:
    st.error("Not enough overlapping FX + rate data to fit the equilibrium model.")
    st.stop()

eq = fit_equilibrium(joined["usdjpy"], joined["diff"], rolling_window=252)

log_rets = daily_log_returns(fx).to_numpy()
annual_drift_estimate = float(np.mean(log_rets) * TRADING_DAYS)
returns_key = tuple(fx["close"].to_numpy().tolist())
import hashlib as _hashlib
_h = _hashlib.sha1()
_h.update(np.asarray(returns_key, dtype=np.float64).tobytes())
_h.update(np.asarray(rates["diff"].to_numpy(), dtype=np.float64).tobytes())
_h.update(str(fx.index[-1]).encode())
returns_hash = _h.hexdigest()[:16]

with st.spinner(f"Calibrating {choice_labels[selected_model]} on USD/JPY history…"):
    cal = calibrate_pricer(selected_model, returns_key, annual_drift_estimate)
params = cal["params"]

current_spot = latest_close(fx)
us_yield = float(rates["us_yield"].iloc[-1])
jp_yield = float(rates["jp_yield"].iloc[-1])
rate_diff = float(rates["diff"].iloc[-1])


# ---------------------------------------------------------------------------
# Status bar
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

st.caption(f"Active model: **{choice_labels[selected_model]}** · Overlay: "
           f"{'on' if use_overlay else 'off'} · λ = {eq.lambda_*1e4:.2f} bp/σ/day")
st.markdown("---")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

live_tab, backtest_tab = st.tabs(["📈 Live forecast", "🧪 Backtest"])


# =========================================================================
# LIVE TAB
# =========================================================================
with live_tab:
    LONGEST_DAYS = 180
    n_steps_total = calendar_to_trading_steps(LONGEST_DAYS)
    with st.spinner(f"Simulating {n_paths:,} paths over 180 days with {choice_labels[selected_model]}…"):
        s_paths = simulate_live_paths(
            selected_model, params, current_spot, n_steps_total, int(n_paths),
            annual_drift_estimate, eq, use_overlay,
        )

    last_date = fx.index[-1]
    future_dates = pd.date_range(start=last_date, periods=n_steps_total + 1,
                                  freq=pd.tseries.offsets.BDay())

    # ---------- Fan chart ----------
    st.subheader("Probability Fan — USD/JPY forecast distribution")
    bands = np.quantile(s_paths, QS, axis=1)
    median_path = bands[QS.index(0.5)]

    _hist_cutoff = fx.index[-1] - pd.Timedelta(days=365)
    hist = fx["close"].loc[fx["close"].index >= _hist_cutoff]

    fig_fan = go.Figure()
    band_specs = [
        (0.025, 0.975, "rgba(99,102,241,0.10)", "95% CI"),
        (0.05, 0.95, "rgba(99,102,241,0.15)", "90% CI"),
        (0.15, 0.85, "rgba(99,102,241,0.22)", "70% CI"),
        (0.25, 0.75, "rgba(99,102,241,0.30)", "50% CI"),
    ]
    for lo, hi, color, name in band_specs:
        lo_arr = bands[QS.index(lo)]
        hi_arr = bands[QS.index(hi)]
        fig_fan.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(hi_arr) + list(lo_arr[::-1]),
            fill="toself", fillcolor=color, line=dict(width=0),
            name=name, hoverinfo="skip", showlegend=True,
        ))
    fig_fan.add_trace(go.Scatter(x=future_dates, y=median_path, mode="lines",
                                  line=dict(color="#4f46e5", width=2), name="Forecast median"))
    fig_fan.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines",
                                  line=dict(color="#111827", width=1.6), name="Historical close"))
    fig_fan.add_hline(y=eq.equilibrium, line=dict(color="#10b981", width=1, dash="dash"),
                      annotation_text=f"Equilibrium {eq.equilibrium:.2f}", annotation_position="top left")
    fig_fan.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="USD/JPY", hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_fan, use_container_width=True)

    # ---------- Horizon cards ----------
    st.subheader("Horizon distributions")
    buckets = parse_buckets(bucket_text) or parse_buckets(DEFAULT_BUCKETS)
    horizon_stats = []
    for label, days in LIVE_HORIZONS:
        idx = min(calendar_to_trading_steps(days), s_paths.shape[0] - 1)
        terminals = s_paths[idx]
        stats = summarize_terminals(horizon_days=days, n_steps=idx,
                                     terminals=terminals, s0=current_spot, buckets=buckets)
        horizon_stats.append((label, stats))

    def render_horizon_card(label, stats):
        st.markdown(
            f"#### {label}  \n<span style='color:#666;font-size:0.85rem'>{stats.horizon_days} calendar days</span>",
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=stats.terminals, nbinsx=60,
                                    marker=dict(color="#6366f1", line=dict(color="#4338ca", width=0.3)),
                                    opacity=0.85, name="Terminal rate"))
        for q, color, name in [(stats.p05, "#ef4444", "5%"), (stats.median, "#111827", "median"),
                                (stats.p95, "#ef4444", "95%")]:
            fig.add_vline(x=q, line=dict(color=color, width=1, dash="dot"),
                          annotation_text=f"{name} {q:.2f}", annotation_position="top")
        fig.add_vline(x=current_spot, line=dict(color="#10b981", width=1.2),
                      annotation_text=f"spot {current_spot:.2f}", annotation_position="bottom")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                          xaxis_title="USD/JPY (terminal)", bargap=0.02)
        st.plotly_chart(fig, use_container_width=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Mean", f"{stats.mean:.2f}")
        sc2.metric("Median", f"{stats.median:.2f}")
        sc3.metric("5th pct", f"{stats.p05:.2f}")
        sc4.metric("95th pct", f"{stats.p95:.2f}")
        if stats.bucket_probs:
            bdf = pd.DataFrame([{"Bucket": k, "Probability": f"{v * 100:.1f}%"} for k, v in stats.bucket_probs.items()])
            st.dataframe(bdf, use_container_width=True, hide_index=True)
            top = max(stats.bucket_probs.items(), key=lambda kv: kv[1])
            st.caption(f"**{top[1]*100:.0f}%** probability USD/JPY {top[0]} in {stats.horizon_days} days")
        d1, d2 = st.columns(2)
        d1.caption(f"P(JPY appreciates) = {stats.p_jpy_appreciation*100:.1f}%")
        d2.caption(f"P(USD appreciates) = {stats.p_usd_appreciation*100:.1f}%")

    row = st.columns(2)
    with row[0]: render_horizon_card(*horizon_stats[0])
    with row[1]: render_horizon_card(*horizon_stats[1])
    row = st.columns(2)
    with row[0]: render_horizon_card(*horizon_stats[2])
    with row[1]: render_horizon_card(*horizon_stats[3])

    # ---------- Equilibrium tracker ----------
    st.markdown("---")
    st.subheader("Equilibrium Tracker")
    _eq_cutoff = eq.fitted.index[-1] - pd.Timedelta(days=3 * 365)
    et = eq.fitted.loc[eq.fitted.index >= _eq_cutoff].copy()
    fig_eq = make_subplots(specs=[[{"secondary_y": True}]])
    fig_eq.add_trace(go.Scatter(x=et.index, y=et["usdjpy"], name="Actual USD/JPY",
                                 line=dict(color="#111827", width=1.6)), secondary_y=False)
    fig_eq.add_trace(go.Scatter(x=et.index, y=et["equilibrium"], name="Model equilibrium",
                                 line=dict(color="#10b981", width=1.4, dash="dash")), secondary_y=False)
    fig_eq.add_trace(go.Scatter(x=et.index, y=et["z"], name="z-score",
                                 line=dict(color="#ef4444", width=1.0)), secondary_y=True)
    fig_eq.add_hline(y=0, line=dict(color="#bbb", width=0.6), secondary_y=True)
    fig_eq.update_yaxes(title_text="USD/JPY", secondary_y=False)
    fig_eq.update_yaxes(title_text="z-score", secondary_y=True)
    fig_eq.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                          hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_eq, use_container_width=True)

    # ---------- Active-model parameter panel ----------
    st.subheader(f"Model parameters — {choice_labels[selected_model]}")
    pricer = get_pricer(selected_model)
    rows = pricer.display_params(params)
    if rows:
        ncols = min(5, len(rows))
        cols = st.columns(ncols)
        for i, (name, val, desc) in enumerate(rows):
            with cols[i % ncols]:
                st.metric(name, val, help=desc)

    st.caption(
        f"Equilibrium model: USDJPY = {eq.alpha:.2f} + {eq.beta:.2f} × (US₃ₘ − JP₃ₘ).  "
        f"Residual σ = {eq.residual_std:.2f}.  "
        f"Daily mean-reversion λ = {eq.lambda_*1e4:.2f} bp/σ.  "
        f"Overlay {'on' if use_overlay else 'off'}."
    )

    # ---------- AI narrative ----------
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
                    "model": choice_labels[selected_model],
                    "spot": current_spot, "us_yield": us_yield, "jp_yield": jp_yield,
                    "rate_diff": rate_diff, "equilibrium": eq.equilibrium,
                    "z_score": eq.z_score, "lambda_daily": eq.lambda_,
                    "params": params,
                    "horizons": [
                        {"label": label, "days": s.horizon_days, "median": s.median,
                         "p05": s.p05, "p95": s.p95,
                         "p_jpy_appreciation": s.p_jpy_appreciation, "buckets": s.bucket_probs}
                        for label, s in horizon_stats
                    ],
                }
                client = anthropic.Anthropic(api_key=api_key)
                with st.spinner("Asking Claude for a narrative…"):
                    msg = client.messages.create(
                        model="claude-3-5-sonnet-latest", max_tokens=600,
                        messages=[{"role": "user", "content": (
                            "You are an FX strategist. Write a concise, neutral interpretation "
                            "(max 200 words) of this USD/JPY forecast. Reference the active model, "
                            "the equilibrium z-score and the multi-horizon distribution. "
                            "Do not give trading advice. Facts (JSON): " + str(facts)
                        )}],
                    )
                    text = "".join(getattr(b, "text", "") for b in msg.content)
                    st.write(text)
            except Exception as e:
                st.warning(f"Narrative generation failed: {e}")


# =========================================================================
# BACKTEST TAB
# =========================================================================
with backtest_tab:
    st.subheader("Backtest module")
    st.caption(
        "Walk-forward evaluation with monthly recalibration. Each forecast distribution "
        "is scored against the realised rate at the corresponding horizon. CRPS and log "
        "score quantify probabilistic accuracy; coverage rates measure calibration."
    )

    bt_default_end = fx.index[-1] - pd.Timedelta(days=30)
    bt_default_start = bt_default_end - pd.Timedelta(days=365)

    cfgA, cfgB, cfgC = st.columns([1.2, 1.2, 1.6])
    with cfgA:
        bt_start = st.date_input("Backtest start", value=bt_default_start.date(),
                                  min_value=fx.index[0].date(), max_value=fx.index[-1].date())
        bt_end = st.date_input("Backtest end", value=bt_default_end.date(),
                                min_value=fx.index[0].date(), max_value=fx.index[-1].date())
    with cfgB:
        bt_paths = st.select_slider("Paths per forecast",
            options=[500, 1_000, 2_500, 5_000, 10_000], value=1_000,
            help="Smaller values = much faster backtest. 1000 paths is sufficient for CRPS to within ~3%.")
        bt_step = st.select_slider("Forecast every N trading days",
            options=[5, 10, 21], value=10,
            help="Larger values = fewer forecast dates = faster backtest.")
    with cfgC:
        bt_horizons = st.multiselect(
            "Horizons to score",
            options=list(HORIZONS_TD.keys()),
            format_func=lambda x: BACKTEST_HORIZON_LABELS[x],
            default=["1w", "1m", "3m"],
        )
        bt_use_overlay = st.checkbox("Apply equilibrium overlay during backtest",
                                      value=True, key="bt_overlay")

    if not bt_horizons:
        st.warning("Select at least one horizon.")
        st.stop()

    sub_single, sub_pair, sub_all = st.tabs([
        "Single model", "Pairwise comparison", "All-model ranking"
    ])

    # ---------------- helper to render forecast vs actual scatter + time series
    def scatter_actual_vs_predicted(forecasts: pd.DataFrame, title: str) -> go.Figure:
        fig = go.Figure()
        for h in forecasts["horizon"].unique():
            sub = forecasts[forecasts["horizon"] == h]
            fig.add_trace(go.Scatter(x=sub["realised"], y=sub["median"],
                                      mode="markers", name=BACKTEST_HORIZON_LABELS.get(h, h),
                                      marker=dict(size=6, opacity=0.7)))
        lo = float(min(forecasts["realised"].min(), forecasts["median"].min()))
        hi = float(max(forecasts["realised"].max(), forecasts["median"].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                                  line=dict(color="#888", dash="dash")))
        fig.update_layout(title=title, xaxis_title="Realised USD/JPY",
                          yaxis_title="Predicted median",
                          height=380, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    def time_series_with_bands(forecasts: pd.DataFrame, model_label: str,
                                color: str = "#4f46e5") -> go.Figure:
        # use first horizon present for the time series chart
        h = sorted(forecasts["horizon"].unique(),
                   key=lambda x: HORIZONS_TD[x])[0]
        sub = forecasts[forecasts["horizon"] == h].sort_values("target_date")
        fig = go.Figure()
        # actual
        actual = fx["close"].loc[(fx.index >= sub["target_date"].min()) &
                                  (fx.index <= sub["target_date"].max())]
        fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="lines",
                                  line=dict(color="#111827", width=1.4), name="Actual"))
        # 70% band
        fig.add_trace(go.Scatter(
            x=list(sub["target_date"]) + list(sub["target_date"][::-1]),
            y=list(sub["band70_hi"]) + list(sub["band70_lo"][::-1]),
            fill="toself", fillcolor=color.replace(")", ",0.18)").replace("rgb", "rgba")
                if color.startswith("rgb") else "rgba(99,102,241,0.18)",
            line=dict(width=0), name=f"{model_label} 70%", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(x=sub["target_date"], y=sub["median"], mode="lines",
                                  line=dict(color=color, width=1.8),
                                  name=f"{model_label} median ({BACKTEST_HORIZON_LABELS[h]})"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=None, yaxis_title="USD/JPY",
                          hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        return fig

    def calibration_chart(cal: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal["nominal"], y=cal["realised"], mode="markers+lines",
                                  name="Observed", line=dict(color="#4f46e5", width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                  line=dict(color="#888", dash="dash")))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Stated confidence",
                          yaxis_title="Realised coverage",
                          xaxis=dict(range=[0, 1.05]), yaxis=dict(range=[0, 1.05]))
        return fig

    # ---------------- SUB A: SINGLE MODEL ----------------
    with sub_single:
        cA, cB = st.columns([1.5, 1])
        with cA:
            single_model = st.selectbox(
                "Model to backtest",
                options=choice_keys, format_func=lambda k: choice_labels[k],
                index=choice_keys.index(selected_model),
                key="single_bt_model",
            )
        with cB:
            run_single = st.button("Run single-model backtest", type="primary", use_container_width=True)

        if run_single:
            with st.spinner(f"Running {choice_labels[single_model]} walk-forward backtest…"):
                d = cached_backtest(
                    single_model, returns_hash, bt_start.isoformat(), bt_end.isoformat(),
                    tuple(bt_horizons), bt_paths, bt_step, 21,
                    bt_use_overlay, eq.lambda_, f"a{eq.alpha:.4f}b{eq.beta:.4f}l{eq.lambda_:.5f}",
                )
            st.session_state["last_single_bt"] = d

        d = st.session_state.get("last_single_bt")
        if d:
            res = restore_backtest(d)
            if res.forecasts.empty:
                st.warning("No forecasts produced — try a wider window or different horizons.")
            else:
                v = calibration_verdict(res.overall)
                color = ("#10b981" if v.startswith("WELL") else
                         "#f59e0b" if v.startswith("PARTIAL") or v.startswith("UNDER") else
                         "#ef4444")
                st.markdown(
                    f"<div style='padding:10px 14px;border-left:4px solid {color};"
                    f"background:#f9fafb;border-radius:4px'>"
                    f"<b>{choice_labels[res.model]}:</b> {v}</div>",
                    unsafe_allow_html=True,
                )
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("CRPS", f"{res.overall['crps']:.3f}")
                m2.metric("Log score", f"{res.overall['log_score']:.3f}")
                m3.metric("70% coverage", f"{res.overall['cov70']*100:.1f}%")
                m4.metric("95% coverage", f"{res.overall['cov95']*100:.1f}%")
                m5.metric("MAE (median)", f"{res.overall['mae_median']:.3f}")

                st.dataframe(
                    res.summary.assign(
                        cov70=lambda d: (d["cov70"] * 100).map("{:.1f}%".format),
                        cov95=lambda d: (d["cov95"] * 100).map("{:.1f}%".format),
                        crps=lambda d: d["crps"].map("{:.3f}".format),
                        log_score=lambda d: d["log_score"].map("{:.3f}".format),
                        mae_median=lambda d: d["mae_median"].map("{:.3f}".format),
                        mae_mean=lambda d: d["mae_mean"].map("{:.3f}".format),
                        bias=lambda d: d["bias"].map("{:+.3f}".format),
                        calibration_err=lambda d: d["calibration_err"].map("{:.3f}".format),
                    ),
                    use_container_width=True, hide_index=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(scatter_actual_vs_predicted(res.forecasts,
                                                                  "Predicted median vs realised"),
                                    use_container_width=True)
                with col2:
                    st.plotly_chart(calibration_chart(res.calibration),
                                    use_container_width=True)
                st.plotly_chart(time_series_with_bands(res.forecasts, choice_labels[res.model]),
                                use_container_width=True)

                if st.button("Set as live default", key="set_default_single"):
                    st.session_state["pending_default_model"] = res.model
                    st.success(f"Live default set to {choice_labels[res.model]}.")
                    st.rerun()

                st.caption(f"Backtest runtime: {res.runtime_seconds:.1f}s · "
                           f"{int(res.overall['n'])} forecasts evaluated")

    # ---------------- SUB B: PAIRWISE ----------------
    with sub_pair:
        ca, cb, cc = st.columns([1, 1, 1])
        with ca:
            pair_a = st.selectbox("Model A", options=choice_keys,
                                   format_func=lambda k: choice_labels[k],
                                   index=0, key="pair_a")
        with cb:
            pair_b = st.selectbox("Model B", options=choice_keys,
                                   format_func=lambda k: choice_labels[k],
                                   index=min(1, len(choice_keys) - 1), key="pair_b")
        with cc:
            run_pair = st.button("Run pairwise comparison", type="primary",
                                  use_container_width=True, disabled=(pair_a == pair_b))

        if run_pair and pair_a != pair_b:
            sig = f"a{eq.alpha:.4f}b{eq.beta:.4f}l{eq.lambda_:.5f}"
            with st.spinner(f"Backtesting {choice_labels[pair_a]} and {choice_labels[pair_b]}…"):
                da = cached_backtest(pair_a, returns_hash, bt_start.isoformat(), bt_end.isoformat(),
                                      tuple(bt_horizons), bt_paths, bt_step, 21,
                                      bt_use_overlay, eq.lambda_, sig)
                db = cached_backtest(pair_b, returns_hash, bt_start.isoformat(), bt_end.isoformat(),
                                      tuple(bt_horizons), bt_paths, bt_step, 21,
                                      bt_use_overlay, eq.lambda_, sig)
            st.session_state["last_pair_bt"] = (da, db)

        pair = st.session_state.get("last_pair_bt")
        if pair:
            ra = restore_backtest(pair[0]); rb = restore_backtest(pair[1])
            if ra.forecasts.empty or rb.forecasts.empty:
                st.warning("Backtest produced no forecasts.")
            else:
                pw = pairwise_winner(ra, rb)
                winner_label = choice_labels[pw["winner"]] if pw["winner"] else "—"
                color = "#10b981"
                p_str = f"p={pw['dm_p']:.3f}" if pw["dm_p"] == pw["dm_p"] else "p=n/a"
                sig_marker = " ★ statistically significant" if pw["dm_p"] < 0.05 else ""
                st.markdown(
                    f"<div style='padding:14px;border-left:4px solid {color};"
                    f"background:#f9fafb;border-radius:4px;font-size:1.05rem'>"
                    f"<b>WINNER: {winner_label}</b> · "
                    f"{choice_labels[ra.model]} CRPS {pw['crps_a']:.3f} vs "
                    f"{choice_labels[rb.model]} CRPS {pw['crps_b']:.3f} "
                    f"({pw['rel_improvement']*100:+.1f}% better) · "
                    f"DM-test {p_str}{sig_marker}</div>",
                    unsafe_allow_html=True,
                )

                # side-by-side metrics table
                def overall_row(res: ModelBacktest) -> dict:
                    o = res.overall
                    return {
                        "Model": choice_labels[res.model],
                        "CRPS (↓)": f"{o['crps']:.3f}",
                        "Log score (↑)": f"{o['log_score']:.3f}",
                        "70% coverage": f"{o['cov70']*100:.1f}%",
                        "95% coverage": f"{o['cov95']*100:.1f}%",
                        "MAE median (↓)": f"{o['mae_median']:.3f}",
                        "MAE mean (↓)": f"{o['mae_mean']:.3f}",
                        "Calibration err (↓)": f"{o['calibration_err']:.3f}",
                        "Bias": f"{o['bias']:+.3f}",
                    }
                better_col = []
                row_a = overall_row(ra); row_b = overall_row(rb)
                better = {}
                for key in ["CRPS (↓)", "MAE median (↓)", "MAE mean (↓)", "Calibration err (↓)"]:
                    better[key] = "A" if float(row_a[key]) < float(row_b[key]) else "B"
                for key in ["Log score (↑)"]:
                    better[key] = "A" if float(row_a[key]) > float(row_b[key]) else "B"
                for key in ["70% coverage", "95% coverage"]:
                    target = 0.70 if key.startswith("70") else 0.95
                    a_err = abs(float(row_a[key].rstrip("%")) / 100 - target)
                    b_err = abs(float(row_b[key].rstrip("%")) / 100 - target)
                    better[key] = "A" if a_err < b_err else "B"

                metrics_df = pd.DataFrame([
                    {"Metric": k, choice_labels[ra.model]: row_a[k],
                     choice_labels[rb.model]: row_b[k],
                     "Better": better.get(k, "—")}
                    for k in ["CRPS (↓)", "Log score (↑)", "70% coverage", "95% coverage",
                              "MAE median (↓)", "MAE mean (↓)", "Calibration err (↓)", "Bias"]
                ])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                # side-by-side scatter
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(scatter_actual_vs_predicted(ra.forecasts, choice_labels[ra.model]),
                                    use_container_width=True)
                with col2:
                    st.plotly_chart(scatter_actual_vs_predicted(rb.forecasts, choice_labels[rb.model]),
                                    use_container_width=True)

                # combined time series
                st.markdown("**Median forecasts vs realised (first horizon)**")
                h0 = sorted(set(ra.forecasts["horizon"]) & set(rb.forecasts["horizon"]),
                             key=lambda x: HORIZONS_TD[x])[0]
                sub_a = ra.forecasts[ra.forecasts["horizon"] == h0].sort_values("target_date")
                sub_b = rb.forecasts[rb.forecasts["horizon"] == h0].sort_values("target_date")
                actual = fx["close"].loc[
                    (fx.index >= min(sub_a["target_date"].min(), sub_b["target_date"].min())) &
                    (fx.index <= max(sub_a["target_date"].max(), sub_b["target_date"].max()))]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="lines",
                                          line=dict(color="#111827", width=1.4), name="Actual"))
                for sub, label, color in [(sub_a, choice_labels[ra.model], "#4f46e5"),
                                           (sub_b, choice_labels[rb.model], "#ef4444")]:
                    fig.add_trace(go.Scatter(x=sub["target_date"], y=sub["median"], mode="lines",
                                              line=dict(color=color, width=1.6), name=f"{label} median"))
                fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                                   yaxis_title="USD/JPY", hovermode="x unified",
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig, use_container_width=True)

                if st.button("Set winner as live default", key="set_default_pair"):
                    st.session_state["pending_default_model"] = pw["winner"]
                    st.success(f"Live default set to {choice_labels[pw['winner']]}.")
                    st.rerun()

    # ---------------- SUB C: ALL-MODEL ----------------
    with sub_all:
        models_to_run = st.multiselect(
            "Models to include",
            options=choice_keys, default=choice_keys,
            format_func=lambda k: choice_labels[k],
            key="allmodel_select",
        )
        run_all = st.button("Run all-model backtest", type="primary",
                             disabled=(len(models_to_run) < 2))
        st.caption(
            "First-time runs may take several minutes per model. Subsequent identical runs "
            "are served from cache."
        )

        if run_all:
            sig = f"a{eq.alpha:.4f}b{eq.beta:.4f}l{eq.lambda_:.5f}"
            results: Dict[str, ModelBacktest] = {}
            prog = st.progress(0.0, text="Starting…")
            for i, m in enumerate(models_to_run):
                prog.progress(i / len(models_to_run),
                               text=f"Backtesting {choice_labels[m]} ({i+1}/{len(models_to_run)})…")
                d = cached_backtest(m, returns_hash, bt_start.isoformat(), bt_end.isoformat(),
                                     tuple(bt_horizons), bt_paths, bt_step, 21,
                                     bt_use_overlay, eq.lambda_, sig)
                results[m] = restore_backtest(d)
            prog.progress(1.0, text="Done.")
            st.session_state["last_all_bt"] = results

        results = st.session_state.get("last_all_bt")
        if results:
            valid = {n: r for n, r in results.items() if not r.forecasts.empty}
            if not valid:
                st.warning("All backtests returned empty results.")
            else:
                rows = []
                for n, r in valid.items():
                    o = r.overall
                    rows.append({
                        "Model": choice_labels[n],
                        "CRPS": o["crps"],
                        "Log score": o["log_score"],
                        "70% cov": o["cov70"],
                        "95% cov": o["cov95"],
                        "MAE median": o["mae_median"],
                        "Calibration err": o["calibration_err"],
                        "Bias": o["bias"],
                        "_key": n,
                    })
                df = pd.DataFrame(rows).sort_values("CRPS").reset_index(drop=True)
                df_disp = df.drop(columns=["_key"]).copy()
                df_disp["CRPS"] = df_disp["CRPS"].map("{:.3f}".format)
                df_disp["Log score"] = df_disp["Log score"].map("{:.3f}".format)
                df_disp["70% cov"] = (df_disp["70% cov"] * 100).map("{:.1f}%".format)
                df_disp["95% cov"] = (df_disp["95% cov"] * 100).map("{:.1f}%".format)
                df_disp["MAE median"] = df_disp["MAE median"].map("{:.3f}".format)
                df_disp["Calibration err"] = df_disp["Calibration err"].map("{:.3f}".format)
                df_disp["Bias"] = df_disp["Bias"].map("{:+.3f}".format)
                st.dataframe(df_disp, use_container_width=True, hide_index=True)

                best_key = df.iloc[0]["_key"]
                st.success(f"Best model by CRPS: **{choice_labels[best_key]}** "
                           f"(CRPS = {df.iloc[0]['CRPS']:.3f})")

                # Diebold-Mariano triangular matrix
                st.markdown("**Diebold–Mariano p-value matrix** (off-diagonal entries; "
                            "values < 0.05 indicate statistically significant difference in CRPS)")
                M = all_model_pvalue_matrix(valid)
                M.index = [choice_labels[n] for n in M.index]
                M.columns = [choice_labels[n] for n in M.columns]
                st.dataframe(M.round(3).fillna(""), use_container_width=True)

                # rolling CRPS chart
                roll = rolling_crps(valid, window_days=90)
                if not roll.empty:
                    fig = go.Figure()
                    palette = ["#4f46e5", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
                               "#ec4899", "#0ea5e9", "#84cc16", "#f97316", "#06b6d4",
                               "#a855f7", "#22c55e", "#eab308"]
                    for i, col in enumerate(roll.columns):
                        s = roll[col].dropna()
                        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                                  name=choice_labels[col],
                                                  line=dict(color=palette[i % len(palette)], width=1.4)))
                    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                                       yaxis_title="Trailing 90-day CRPS",
                                       hovermode="x unified",
                                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig, use_container_width=True)

                # current recommendation panel
                rec_recent = None
                if not roll.empty:
                    last_row = roll.iloc[-1].dropna()
                    if not last_row.empty:
                        rec_recent = last_row.idxmin()
                rec_long = best_key
                std_dev = roll.std().dropna() if not roll.empty else pd.Series()
                most_consistent = std_dev.idxmin() if not std_dev.empty else best_key
                colA, colB, colC = st.columns(3)
                colA.metric("Best (recent 90d)",
                             choice_labels[rec_recent] if rec_recent else "—",
                             help="Lowest trailing-90d CRPS at the end of the backtest window.")
                colB.metric("Best (overall)", choice_labels[rec_long])
                colC.metric("Most consistent", choice_labels[most_consistent],
                             help="Lowest std-dev of trailing CRPS — most stable performer.")

                if st.button("Set best (overall) as live default", key="set_default_all"):
                    st.session_state["pending_default_model"] = best_key
                    st.success(f"Live default set to {choice_labels[best_key]}.")
                    st.rerun()


st.caption(
    "This app calibrates parametric models to historical FX returns. It is not investment "
    "advice. Probabilities reflect the model's assumptions, not certainties about the future."
)

"""Walk-forward backtest engine for FX forecast distributions.

For every backtest date D in the chosen range we:
  1. Calibrate the requested model on log-returns through D-1 (no lookahead).
  2. Generate Monte Carlo paths for each requested horizon (in trading days).
  3. Score the resulting forecast distribution against the realised rate at D + horizon.

Recalibration is monthly by default to keep runtime sane; between recalibrations
we re-use the latest fitted parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .pricers import PRICERS, TRADING_DAYS, get_pricer
from .disequilibrium_fx import EquilibriumModel, disequilibrium_drift_per_step

# ---------------------------------------------------------------------------
# Helpers / metrics
# ---------------------------------------------------------------------------

HORIZONS_TD = {  # canonical horizons in trading days
    "1w": 5,
    "2w": 10,
    "1m": 21,
    "3m": 63,
    "6m": 126,
}


def crps_sample(samples: np.ndarray, y: float) -> float:
    """Sample-based CRPS estimator (Hersbach decomposition).

    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|, X, X' ~ F i.i.d.
    Lower is better.
    """
    s = np.sort(samples)
    n = len(s)
    if n == 0:
        return float("nan")
    term1 = float(np.mean(np.abs(s - y)))
    # closed form for sorted samples: 2 * sum_i (i - 0.5*(n-1)) * s_i / n²
    idx = np.arange(n)
    term2 = float(2.0 * np.sum((idx - (n - 1) / 2.0) * s) / (n * n))
    return term1 - 0.5 * term2


def log_score_kde(samples: np.ndarray, y: float) -> float:
    """Log score of realised value y under a Gaussian-KDE estimate of the forecast density."""
    n = len(samples)
    if n < 5:
        return float("nan")
    sigma = float(np.std(samples, ddof=1))
    if sigma <= 0:
        return float("nan")
    h = 1.06 * sigma * (n ** (-1.0 / 5.0))  # Silverman
    z = (y - samples) / h
    log_kernels = -0.5 * (z * z) - 0.5 * np.log(2 * np.pi) - np.log(h)
    m = float(np.max(log_kernels))
    return m + float(np.log(np.mean(np.exp(log_kernels - m))))


def coverage_indicator(samples: np.ndarray, y: float, level: float) -> int:
    lo = float(np.quantile(samples, (1 - level) / 2.0))
    hi = float(np.quantile(samples, 1.0 - (1 - level) / 2.0))
    return int(lo <= y <= hi)


# Diebold–Mariano (loss differential test, autocorrelation-aware)
def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> Tuple[float, float]:
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    d_bar = float(np.mean(d))
    # Newey–West HAC variance with bandwidth = h - 1
    gamma0 = float(np.var(d, ddof=0))
    var_d = gamma0
    for k in range(1, h):
        cov = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var_d += 2.0 * (1.0 - k / h) * cov
    var_d = max(var_d, 1e-16)
    dm_stat = d_bar / np.sqrt(var_d / n)
    # two-sided p-value under N(0,1)
    from scipy.stats import norm
    p = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p)


# ---------------------------------------------------------------------------
# Backtest result containers
# ---------------------------------------------------------------------------

@dataclass
class ForecastRecord:
    date: pd.Timestamp
    horizon: str
    horizon_td: int
    target_date: pd.Timestamp
    s0: float
    realised: float
    median: float
    mean: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    crps: float
    log_score: float
    cov70: int
    cov95: int
    band70_lo: float
    band70_hi: float
    band95_lo: float
    band95_hi: float


@dataclass
class ModelBacktest:
    model: str
    forecasts: pd.DataFrame                 # one row per (date, horizon)
    summary: pd.DataFrame                   # per-horizon metrics
    overall: Dict[str, float]               # one-line summary across all horizons
    calibration: pd.DataFrame               # nominal vs realised coverage table
    params_history: pd.DataFrame            # one row per recalibration
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------

def _select_backtest_dates(fx: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                           step_days: int) -> List[pd.Timestamp]:
    idx = fx.index
    start = max(start, idx[0])
    end = min(end, idx[-1])
    keep = idx[(idx >= start) & (idx <= end)]
    return list(keep[::step_days])


def run_single_model_backtest(
    model: str,
    fx: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizons: List[str],
    n_paths: int = 5000,
    step_days: int = 5,                    # forecast every N trading days
    recal_every_days: int = 21,            # recalibrate monthly
    history_window_years: int = 3,         # rolling calibration window
    equilibrium: Optional[EquilibriumModel] = None,
    extra_drift_lambda: float = 0.0,       # daily mean-reversion strength
    use_eq_overlay: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    seed: int = 42,
) -> ModelBacktest:
    import time
    t0 = time.time()
    pricer = get_pricer(model)
    horizons = [h for h in horizons if h in HORIZONS_TD]
    horizon_steps = {h: HORIZONS_TD[h] for h in horizons}
    max_h = max(horizon_steps.values())

    fx = fx.sort_index()
    log_returns_full = np.log(fx["close"]).diff().dropna()
    dates = _select_backtest_dates(fx, start, end - pd.Timedelta(days=int(max_h * 1.6)), step_days)

    records: List[ForecastRecord] = []
    params_history: List[Dict] = []
    last_params: Optional[Dict] = None
    last_calibration_date: Optional[pd.Timestamp] = None
    n_total = len(dates)

    for i, d in enumerate(dates):
        # 1) calibrate (monthly cadence)
        need_recal = (last_params is None or
                      last_calibration_date is None or
                      (d - last_calibration_date).days >= recal_every_days)
        cutoff = d - pd.Timedelta(days=1)
        history_start = d - pd.Timedelta(days=int(history_window_years * 365.25))
        rets_window = log_returns_full.loc[(log_returns_full.index >= history_start) &
                                           (log_returns_full.index <= cutoff)]
        if len(rets_window) < 60:
            continue
        annual_drift = float(rets_window.mean() * TRADING_DAYS)
        if need_recal:
            try:
                last_params = pricer.calibrate(rets_window.to_numpy(), dt=1.0 / TRADING_DAYS,
                                               annual_drift=annual_drift)
                last_calibration_date = d
                params_history.append({"date": d, **last_params})
            except Exception as exc:  # pragma: no cover
                if progress_cb:
                    progress_cb(i / max(n_total, 1), f"calibration failed at {d.date()}: {exc}")
                continue

        # 2) build extra-drift function for the disequilibrium overlay.
        #    Path-dependent: at every step, drift = -λ · (s_t - eq) / σ_resid
        #    so backtest matches live simulation exactly.
        extra_drift_fn = None
        if (use_eq_overlay and equilibrium is not None
                and extra_drift_lambda > 0 and equilibrium.residual_std > 0):
            eq_val = float(equilibrium.equilibrium)
            res_std = float(equilibrium.residual_std)
            lam = float(extra_drift_lambda)

            def _fn(s_at_t, _eq=eq_val, _rs=res_std, _lam=lam):
                return disequilibrium_drift_per_step(s_at_t, _eq, _rs, _lam)
            extra_drift_fn = _fn

        s0 = float(fx.loc[:d, "close"].iloc[-1])

        # 3) simulate to the maximum horizon (we read off intermediate horizons too)
        try:
            S = pricer.simulate_paths(
                last_params, s0, max_h, n_paths,
                dt=1.0 / TRADING_DAYS,
                annual_drift=annual_drift,
                extra_drift_fn=extra_drift_fn,
                seed=seed + i,
            )
        except Exception as exc:  # pragma: no cover
            if progress_cb:
                progress_cb(i / max(n_total, 1), f"simulation failed at {d.date()}: {exc}")
            continue

        # 4) score each horizon
        for h_name, h_td in horizon_steps.items():
            target_date = _next_trading_day(fx.index, d, h_td)
            if target_date is None:
                continue
            samples = S[h_td]
            if not np.all(np.isfinite(samples)):
                samples = samples[np.isfinite(samples)]
            if len(samples) < 100:
                continue
            realised = float(fx.loc[target_date, "close"])
            crps = crps_sample(samples, realised)
            ll = log_score_kde(samples, realised)
            cov70 = coverage_indicator(samples, realised, 0.70)
            cov95 = coverage_indicator(samples, realised, 0.95)
            qs = np.quantile(samples, [0.025, 0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95, 0.975])
            records.append(ForecastRecord(
                date=d, horizon=h_name, horizon_td=h_td, target_date=target_date,
                s0=s0, realised=realised,
                median=float(qs[4]), mean=float(np.mean(samples)),
                p05=float(qs[1]), p25=float(qs[3]), p50=float(qs[4]),
                p75=float(qs[5]), p95=float(qs[7]),
                crps=crps, log_score=ll, cov70=cov70, cov95=cov95,
                band70_lo=float(qs[2]), band70_hi=float(qs[6]),
                band95_lo=float(qs[0]), band95_hi=float(qs[8]),
            ))

        if progress_cb and (i % 5 == 0 or i == n_total - 1):
            progress_cb((i + 1) / max(n_total, 1), f"{model}: {d.date()} ({i+1}/{n_total})")

    # ---- aggregate ----
    if not records:
        empty = pd.DataFrame()
        return ModelBacktest(model, empty, empty, {}, empty, pd.DataFrame(params_history),
                              runtime_seconds=time.time() - t0)

    fc = pd.DataFrame([r.__dict__ for r in records])
    fc["abs_med_err"] = (fc["median"] - fc["realised"]).abs()
    fc["abs_mean_err"] = (fc["mean"] - fc["realised"]).abs()

    # per-horizon summary
    rows = []
    for h_name in horizons:
        sub = fc[fc["horizon"] == h_name]
        if sub.empty:
            continue
        rows.append({
            "horizon": h_name,
            "n": len(sub),
            "crps": float(sub["crps"].mean()),
            "log_score": float(sub["log_score"].mean()),
            "cov70": float(sub["cov70"].mean()),
            "cov95": float(sub["cov95"].mean()),
            "mae_median": float(sub["abs_med_err"].mean()),
            "mae_mean": float(sub["abs_mean_err"].mean()),
            "calibration_err": abs(float(sub["cov70"].mean()) - 0.70)
                + abs(float(sub["cov95"].mean()) - 0.95),
            "bias": float((sub["median"] - sub["realised"]).mean()),
        })
    summary = pd.DataFrame(rows)

    overall = {
        "n": int(len(fc)),
        "crps": float(fc["crps"].mean()),
        "log_score": float(fc["log_score"].mean()),
        "cov70": float(fc["cov70"].mean()),
        "cov95": float(fc["cov95"].mean()),
        "mae_median": float(fc["abs_med_err"].mean()),
        "mae_mean": float(fc["abs_mean_err"].mean()),
        "calibration_err": float(summary["calibration_err"].mean()) if not summary.empty else float("nan"),
        "bias": float((fc["median"] - fc["realised"]).mean()),
    }

    # calibration table: nominal vs realised coverage at multiple levels
    cal_rows = []
    for level in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
        in_band = []
        for _, row in fc.iterrows():
            in_band.append(int(row["band95_lo"] <= row["realised"] <= row["band95_hi"]) if level == 0.95
                          else int(row["band70_lo"] <= row["realised"] <= row["band70_hi"]) if level == 0.70
                          else _coverage_from_pcts(row, level))
        cal_rows.append({"nominal": level, "realised": float(np.mean(in_band))})
    cal_df = pd.DataFrame(cal_rows)

    return ModelBacktest(
        model=model,
        forecasts=fc.sort_values(["horizon", "date"]).reset_index(drop=True),
        summary=summary,
        overall=overall,
        calibration=cal_df,
        params_history=pd.DataFrame(params_history),
        runtime_seconds=time.time() - t0,
    )


def _coverage_from_pcts(row: pd.Series, level: float) -> int:
    # interpolate from p05/p25/p75/p95 to approximate other levels
    if level <= 0.50:
        lo, hi = float(row["p25"]), float(row["p75"])  # 50%
    elif level <= 0.70:
        lo, hi = float(row["band70_lo"]), float(row["band70_hi"])
    elif level <= 0.90:
        lo = 0.5 * (float(row["p05"]) + float(row["band70_lo"]))
        hi = 0.5 * (float(row["p95"]) + float(row["band70_hi"]))
    elif level <= 0.95:
        lo, hi = float(row["band95_lo"]), float(row["band95_hi"])
    else:
        lo = float(row["band95_lo"]) - (float(row["band70_lo"]) - float(row["band95_lo"]))
        hi = float(row["band95_hi"]) + (float(row["band95_hi"]) - float(row["band70_hi"]))
    return int(lo <= float(row["realised"]) <= hi)


def _next_trading_day(idx: pd.DatetimeIndex, d: pd.Timestamp, n: int) -> Optional[pd.Timestamp]:
    if d not in idx:
        idx = idx[idx >= d]
        if len(idx) == 0:
            return None
        d = idx[0]
    pos = idx.get_loc(d)
    target = pos + n
    if target >= len(idx):
        return None
    return idx[target]


# ---------------------------------------------------------------------------
# Verdict utilities
# ---------------------------------------------------------------------------

def calibration_verdict(overall: Dict[str, float]) -> str:
    cov70 = overall.get("cov70", float("nan"))
    cov95 = overall.get("cov95", float("nan"))
    bias = overall.get("bias", 0.0)
    cov70_err = abs(cov70 - 0.70) if cov70 == cov70 else float("nan")
    cov95_err = abs(cov95 - 0.95) if cov95 == cov95 else float("nan")

    # bias check first — large persistent miss in one direction
    bias_threshold = max(0.5, 0.005 * abs(overall.get("mae_median", 0.5) * 50))
    if abs(bias) > bias_threshold:
        direction = "high" if bias > 0 else "low"
        return f"BIASED — median forecasts run systematically {direction} (mean bias {bias:+.2f})"

    if cov70_err <= 0.05 and cov95_err <= 0.05:
        return "WELL-CALIBRATED — observed coverage matches stated confidence levels within 5%"
    if cov70 < 0.65 or cov95 < 0.90:
        return "OVERCONFIDENT — realised coverage below stated levels (forecast bands too narrow)"
    if cov70 > 0.75 or cov95 > 0.99:
        return "UNDERCONFIDENT — realised coverage exceeds stated levels (forecast bands too wide)"
    return "PARTIALLY CALIBRATED — coverage close to target but not within 5% on every level"


def _dm_pooled_by_horizon(merged: pd.DataFrame) -> Tuple[float, float]:
    """Pool CRPS-loss differentials across horizons with horizon-aware HAC bandwidths.

    Combines per-horizon DM statistics into a single pooled stat using inverse-variance
    weighting. For each horizon group we use HAC bandwidth = horizon_td so that
    overlapping multi-step forecasts don't deflate the variance.
    """
    if merged.empty:
        return float("nan"), float("nan")
    weights, stats, ns = [], [], 0
    for h_name, sub in merged.groupby("horizon", sort=False):
        sub = sub.sort_values("date")
        d = sub["crps_a"].to_numpy(float) - sub["crps_b"].to_numpy(float)
        d = d[np.isfinite(d)]
        n = len(d)
        if n < 10:
            continue
        h_td = HORIZONS_TD.get(str(h_name), 1)
        d_bar = float(np.mean(d))
        gamma0 = float(np.var(d, ddof=0))
        # HAC bandwidth: smaller of (a) horizon length (so overlap is captured),
        # (b) Newey-West rule-of-thumb floor(4·(n/100)^(2/9)) at minimum,
        # (c) n // 3 to avoid sample-autocovariance noise dominating.
        nw_rule = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
        bw = max(1, min(h_td, max(nw_rule, n // 4)))
        var_d = gamma0
        for k in range(1, bw):
            cov = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
            var_d += 2.0 * (1.0 - k / bw) * cov
        # Sample HAC can come out slightly negative for small n; clamp to gamma0/n
        # (i.e. fall back to the iid SE) which is a defensible lower bound.
        if var_d < gamma0 * 0.1:
            var_d = gamma0
        se = float(np.sqrt(var_d / n))
        if se <= 0 or not np.isfinite(se):
            continue
        weights.append(1.0 / (se * se))
        stats.append(d_bar)
        ns += n
    if not weights:
        return float("nan"), float("nan")
    w = np.asarray(weights); s = np.asarray(stats)
    pooled_d = float(np.sum(w * s) / np.sum(w))
    pooled_se = float(np.sqrt(1.0 / np.sum(w)))
    dm_stat = pooled_d / max(pooled_se, 1e-16)
    from scipy.stats import norm
    p = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p)


def pairwise_winner(a: ModelBacktest, b: ModelBacktest) -> Dict[str, object]:
    """Compare two model backtests on aligned (date, horizon) records and pick a winner."""
    fa, fb = a.forecasts, b.forecasts
    if fa.empty or fb.empty:
        return {"winner": None, "reason": "no overlapping forecasts"}
    merged = fa.merge(fb, on=["date", "horizon"], suffixes=("_a", "_b"))
    if merged.empty:
        return {"winner": None, "reason": "no aligned dates"}
    crps_a = merged["crps_a"].to_numpy()
    crps_b = merged["crps_b"].to_numpy()
    dm_stat, dm_p = _dm_pooled_by_horizon(merged)
    mean_a = float(np.nanmean(crps_a))
    mean_b = float(np.nanmean(crps_b))
    winner = a.model if mean_a < mean_b else b.model
    rel = (mean_b - mean_a) / max(mean_b, 1e-12) if winner == a.model else (mean_a - mean_b) / max(mean_a, 1e-12)
    return {
        "winner": winner,
        "crps_a": mean_a, "crps_b": mean_b,
        "rel_improvement": rel,
        "dm_stat": dm_stat, "dm_p": dm_p,
        "n": int(len(merged)),
    }


def all_model_pvalue_matrix(results: Dict[str, ModelBacktest]) -> pd.DataFrame:
    """Triangular Diebold–Mariano p-value matrix over CRPS losses (horizon-aware HAC)."""
    names = list(results.keys())
    M = pd.DataFrame(np.full((len(names), len(names)), np.nan), index=names, columns=names)
    for i, a in enumerate(names):
        fa = results[a].forecasts
        for j, b in enumerate(names):
            if j <= i:
                continue
            fb = results[b].forecasts
            merged = fa.merge(fb, on=["date", "horizon"], suffixes=("_a", "_b"))
            if merged.empty:
                continue
            _, p = _dm_pooled_by_horizon(merged)
            M.iloc[i, j] = p
    return M


def rolling_crps(results: Dict[str, ModelBacktest], window_days: int = 90) -> pd.DataFrame:
    """Trailing CRPS per model resampled to a daily index."""
    frames = []
    for name, res in results.items():
        if res.forecasts.empty:
            continue
        sub = res.forecasts.sort_values("date")
        sub = sub.set_index("date")["crps"].rolling(f"{window_days}D").mean()
        sub.name = name
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)

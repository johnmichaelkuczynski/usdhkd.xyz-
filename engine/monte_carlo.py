"""Monte Carlo simulation of disequilibrium-adjusted Heston paths and distribution stats."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .heston import HestonParams, TRADING_DAYS

CALENDAR_TO_TRADING = TRADING_DAYS / 365.25


@dataclass
class HorizonStats:
    horizon_days: int
    n_steps: int
    terminals: np.ndarray
    mean: float
    median: float
    p05: float
    p25: float
    p75: float
    p95: float
    bucket_probs: Dict[str, float]
    p_hkd_appreciation: float
    p_usd_appreciation: float


def calendar_to_trading_steps(calendar_days: int) -> int:
    return max(1, int(round(calendar_days * CALENDAR_TO_TRADING)))


def simulate_disequilibrium_paths(
    params: HestonParams,
    s0: float,
    horizon_calendar_days: int,
    n_paths: int,
    equilibrium: float,
    residual_std: float,
    lambda_daily: float,
    annual_drift: float = 0.0,
    seed: Optional[int] = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate paths with a per-step, per-path disequilibrium drift overlay.

    Returns (S, V) where S has shape (n_steps + 1, n_paths) and V same shape.
    Time is measured in trading days; lambda_daily is applied per trading day.
    """
    n_steps = calendar_to_trading_steps(horizon_calendar_days)
    dt = 1.0 / TRADING_DAYS
    rng = np.random.default_rng(seed)

    s = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    v = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    s[0] = s0
    v[0] = max(params.v0, 1e-10)

    rho = params.rho
    sqrt_one_minus_rho2 = float(np.sqrt(max(1.0 - rho * rho, 1e-8)))
    sqrt_dt = float(np.sqrt(dt))

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w_s = z1
        w_v = rho * z1 + sqrt_one_minus_rho2 * z2

        v_pos = np.maximum(v[t], 0.0)
        sqrt_v = np.sqrt(v_pos)

        v[t + 1] = v[t] + params.kappa * (params.theta - v_pos) * dt + params.xi * sqrt_v * sqrt_dt * w_v
        v[t + 1] = np.maximum(v[t + 1], 0.0)

        # disequilibrium overlay: -lambda * z applied each trading day
        if residual_std > 0 and lambda_daily != 0.0:
            z_path = (s[t] - equilibrium) / residual_std
            diseq_drift = -lambda_daily * z_path  # per-day
        else:
            diseq_drift = 0.0

        mu_step = annual_drift * dt + diseq_drift  # per-day total drift
        log_inc = (mu_step - 0.5 * v_pos * dt) + sqrt_v * sqrt_dt * w_s
        s[t + 1] = s[t] * np.exp(log_inc)

    return s, v


def summarize_terminals(
    horizon_days: int,
    n_steps: int,
    terminals: np.ndarray,
    s0: float,
    buckets: List[float],
) -> HorizonStats:
    """Compute summary statistics + bucket probabilities for terminal rates.

    `buckets` is a sorted list of cut-points; produces probabilities for
      < buckets[0], [b0, b1), ..., >= buckets[-1].
    """
    terminals = np.asarray(terminals, dtype=np.float64)

    bucket_probs: Dict[str, float] = {}
    if buckets:
        sb = sorted(buckets)
        # < first
        key = f"< {sb[0]:g}"
        bucket_probs[key] = float(np.mean(terminals < sb[0]))
        for i in range(len(sb) - 1):
            lo, hi = sb[i], sb[i + 1]
            key = f"{lo:g} – {hi:g}"
            bucket_probs[key] = float(np.mean((terminals >= lo) & (terminals < hi)))
        key = f"≥ {sb[-1]:g}"
        bucket_probs[key] = float(np.mean(terminals >= sb[-1]))

    return HorizonStats(
        horizon_days=horizon_days,
        n_steps=n_steps,
        terminals=terminals,
        mean=float(np.mean(terminals)),
        median=float(np.median(terminals)),
        p05=float(np.percentile(terminals, 5)),
        p25=float(np.percentile(terminals, 25)),
        p75=float(np.percentile(terminals, 75)),
        p95=float(np.percentile(terminals, 95)),
        bucket_probs=bucket_probs,
        p_hkd_appreciation=float(np.mean(terminals < s0)),
        p_usd_appreciation=float(np.mean(terminals > s0)),
    )


def fan_chart_quantiles(
    s_paths: np.ndarray,
    quantiles: Tuple[float, ...] = (0.025, 0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95, 0.975),
) -> np.ndarray:
    """Return array of shape (len(quantiles), n_steps + 1) giving quantile bands across time."""
    return np.quantile(s_paths, list(quantiles), axis=1)

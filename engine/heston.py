"""Heston stochastic-volatility model.

Maximum-likelihood calibration of (kappa, theta, xi, rho, v0) from a return series
using the Euler-discretised SDE and a per-step bivariate-normal likelihood with
latent variance integrated out via a particle filter (bootstrap, small N for speed).

We also expose a fast Monte-Carlo path generator using full-truncation Euler.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

TRADING_DAYS = 252


@dataclass
class HestonParams:
    kappa: float   # mean-reversion speed of variance (per year)
    theta: float   # long-run variance (per year, in variance units)
    xi: float      # vol-of-vol
    rho: float     # correlation between asset and variance shocks
    v0: float      # initial variance

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def long_run_vol_annual(self) -> float:
        return float(np.sqrt(max(self.theta, 0.0)))

    @property
    def current_vol_annual(self) -> float:
        return float(np.sqrt(max(self.v0, 0.0)))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _neg_log_lik_pf(
    params_vec: np.ndarray,
    log_returns: np.ndarray,
    drift: float,
    dt: float,
    n_particles: int = 400,
    seed: int = 7,
) -> float:
    """Bootstrap particle-filter negative log-likelihood for the Heston model.

    State: variance v_t. Observation: log return r_t.
    Transition (full truncation Euler):
      v_{t+1} = max(v_t + kappa*(theta - v_t)*dt + xi*sqrt(max(v_t,0))*sqrt(dt)*z_v, 1e-10)
    Observation given v_t (and z_v):
      r_t ~ Normal(mean = (drift - 0.5*v_t)*dt + rho*sqrt(v_t*dt)*z_v,
                   var  = (1-rho^2)*v_t*dt)
    We marginalise z_v by sampling — bootstrap PF.
    """
    kappa, theta, xi, rho, v0 = params_vec
    # Bounds enforcement (soft penalty handled by `minimize` bounds, but be defensive)
    if not (0.05 < kappa < 25.0):
        return 1e8
    if not (1e-6 < theta < 1.0):
        return 1e8
    if not (1e-4 < xi < 5.0):
        return 1e8
    if not (-0.999 < rho < 0.999):
        return 1e8
    if not (1e-6 < v0 < 1.0):
        return 1e8

    rng = np.random.default_rng(seed)
    N = n_particles
    v = np.full(N, v0, dtype=np.float64)
    log_lik = 0.0

    one_minus_rho2 = max(1.0 - rho * rho, 1e-8)
    sqrt_dt = np.sqrt(dt)

    for r in log_returns:
        v_pos = np.maximum(v, 1e-10)
        sqrt_v_dt = np.sqrt(v_pos * dt)

        # sample z_v for each particle
        z_v = rng.standard_normal(N)

        mean_r = (drift - 0.5 * v_pos) * dt + rho * sqrt_v_dt * z_v
        var_r = one_minus_rho2 * v_pos * dt
        var_r = np.maximum(var_r, 1e-12)

        # per-particle likelihood weight
        diff = r - mean_r
        log_w = -0.5 * (np.log(2 * np.pi * var_r) + (diff * diff) / var_r)

        # log-sum-exp
        m = np.max(log_w)
        lw = m + np.log(np.mean(np.exp(log_w - m)))
        if not np.isfinite(lw):
            return 1e8
        log_lik += lw

        # advance variance with the same z_v draw used for the obs likelihood
        v = v_pos + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * z_v
        v = np.maximum(v, 1e-10)

        # resample (systematic) using normalised weights
        w = np.exp(log_w - m)
        w_sum = w.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            return 1e8
        w /= w_sum
        # systematic resampling
        positions = (rng.random() + np.arange(N)) / N
        cumw = np.cumsum(w)
        cumw[-1] = 1.0
        idx = np.searchsorted(cumw, positions)
        v = v[idx]

    return -log_lik


def calibrate_heston(
    log_returns: np.ndarray,
    annual_drift: float = 0.0,
    dt: float = 1.0 / TRADING_DAYS,
    initial: Optional[HestonParams] = None,
    n_particles: int = 300,
) -> Tuple[HestonParams, bool]:
    """Calibrate Heston via particle-filter MLE.

    Returns (params, converged_flag). Falls back to a sensible historical-vol-based
    parameter set if the optimiser fails.
    """
    log_returns = np.asarray(log_returns, dtype=np.float64)
    log_returns = log_returns[np.isfinite(log_returns)]

    realized_var = float(np.var(log_returns) * (1.0 / dt))  # annualised variance
    realized_var = max(realized_var, 1e-5)

    if initial is None:
        initial = HestonParams(
            kappa=2.0,
            theta=realized_var,
            xi=0.4,
            rho=-0.4,
            v0=realized_var,
        )

    x0 = np.array(
        [initial.kappa, initial.theta, initial.xi, initial.rho, initial.v0],
        dtype=np.float64,
    )

    bounds = [
        (0.1, 20.0),       # kappa
        (1e-5, 0.5),       # theta (annualised variance, ~ vol up to ~70%)
        (0.01, 3.0),       # xi
        (-0.95, 0.95),     # rho
        (1e-5, 0.5),       # v0
    ]

    fallback = HestonParams(
        kappa=initial.kappa,
        theta=realized_var,
        xi=0.3,
        rho=-0.3,
        v0=realized_var,
    )

    try:
        res = minimize(
            _neg_log_lik_pf,
            x0,
            args=(log_returns, annual_drift, dt, n_particles),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 30, "ftol": 1e-4},
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            return fallback, False
        kappa, theta, xi, rho, v0 = res.x
        return HestonParams(kappa, theta, xi, rho, v0), True
    except Exception:
        return fallback, False


# ---------------------------------------------------------------------------
# Path simulation (full-truncation Euler)
# ---------------------------------------------------------------------------

def simulate_paths(
    params: HestonParams,
    s0: float,
    n_steps: int,
    n_paths: int,
    dt: float = 1.0 / TRADING_DAYS,
    drift_per_step: Optional[np.ndarray] = None,
    annual_drift: float = 0.0,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Simulate Heston price paths. Returns array of shape (n_steps + 1, n_paths)."""
    rng = np.random.default_rng(seed)
    n_paths = int(n_paths)
    n_steps = int(n_steps)

    s = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    v = np.empty((n_steps + 1, n_paths), dtype=np.float64)
    s[0] = s0
    v[0] = max(params.v0, 1e-10)

    rho = params.rho
    sqrt_one_minus_rho2 = np.sqrt(max(1.0 - rho * rho, 1e-8))
    sqrt_dt = np.sqrt(dt)

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        # correlated Brownian increments
        w_s = z1
        w_v = rho * z1 + sqrt_one_minus_rho2 * z2

        v_pos = np.maximum(v[t], 0.0)
        sqrt_v = np.sqrt(v_pos)

        # variance update (full truncation)
        v[t + 1] = v[t] + params.kappa * (params.theta - v_pos) * dt + params.xi * sqrt_v * sqrt_dt * w_v
        v[t + 1] = np.maximum(v[t + 1], 0.0)

        # asset drift for this step
        if drift_per_step is not None:
            mu_step = float(drift_per_step[t])
        else:
            mu_step = annual_drift * dt

        log_inc = (mu_step - 0.5 * v_pos * dt) + sqrt_v * sqrt_dt * w_s
        s[t + 1] = s[t] * np.exp(log_inc)

    return s

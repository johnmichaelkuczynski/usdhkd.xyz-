"""Rate-differential equilibrium model for USD/HKD.

equilibrium_rate = baseline + beta * (US_3m_yield - HK_3m_yield)

Fits baseline (alpha) and beta via OLS on aligned daily history. Computes the
rolling residual standard deviation, the current z-score, and an estimate of
lambda (mean-reversion strength) from a regression of next-day log returns on z.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class EquilibriumModel:
    alpha: float           # baseline
    beta: float            # slope on rate differential
    residual_std: float    # rolling residual std at the latest date
    lambda_: float         # daily mean-reversion strength on z (per-day fraction)
    z_score: float         # current z-score
    equilibrium: float     # current equilibrium fair value
    fitted: pd.DataFrame   # date-indexed DataFrame with usdhkd, diff, equilibrium, residual, z

    def zscore_at(self, when: pd.Timestamp) -> float:
        """Return the band z-score at (or just before) `when`."""
        z = self.fitted["z"].dropna()
        z = z.loc[z.index <= when]
        if len(z) == 0:
            return 0.0
        return float(z.iloc[-1])


def fit_equilibrium(
    fx_close: pd.Series,
    rate_diff: pd.Series,
    rolling_window: int = 252,
) -> EquilibriumModel:
    """Fit alpha, beta on the joined sample. Compute z-score and lambda."""
    df = pd.concat([fx_close.rename("usdhkd"), rate_diff.rename("diff")], axis=1).dropna()
    if len(df) < 60:
        raise RuntimeError("Not enough overlapping data to fit equilibrium model.")

    x = df["diff"].to_numpy()
    y = df["usdhkd"].to_numpy()

    # OLS y = alpha + beta * x
    X = np.column_stack([np.ones_like(x), x])
    beta_vec, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(beta_vec[0])
    beta = float(beta_vec[1])

    df["equilibrium"] = alpha + beta * df["diff"]
    df["residual"] = df["usdhkd"] - df["equilibrium"]

    # rolling residual std
    rstd = df["residual"].rolling(rolling_window, min_periods=60).std()
    df["resid_std"] = rstd
    df["z"] = df["residual"] / df["resid_std"]

    last_resid_std = float(df["resid_std"].dropna().iloc[-1])
    last_z = float(df["z"].dropna().iloc[-1])
    last_eq = float(df["equilibrium"].iloc[-1])

    # Estimate lambda from regression of next-day log return on z
    log_ret = np.log(df["usdhkd"]).diff().shift(-1)  # next-day return
    reg = pd.concat([df["z"], log_ret.rename("ret_next")], axis=1).dropna()
    if len(reg) >= 100:
        zx = reg["z"].to_numpy()
        ry = reg["ret_next"].to_numpy()
        Z = np.column_stack([np.ones_like(zx), zx])
        coef, *_ = np.linalg.lstsq(Z, ry, rcond=None)
        # ret = a + b * z; mean-reversion implies b < 0; lambda_ = -b (per-day pull)
        lam_daily = float(-coef[1])
    else:
        lam_daily = 0.0

    # Clip lambda to a sensible range so the drift overlay can't blow up
    lam_daily = float(np.clip(lam_daily, -0.005, 0.02))

    return EquilibriumModel(
        alpha=alpha,
        beta=beta,
        residual_std=last_resid_std,
        lambda_=lam_daily,
        z_score=last_z,
        equilibrium=last_eq,
        fitted=df,
    )


def disequilibrium_drift_per_step(
    s_paths_at_t: np.ndarray,
    equilibrium: float,
    residual_std: float,
    lambda_daily: float,
) -> np.ndarray:
    """Vectorised drift adjustment for one Monte Carlo step.

    Returns an array (n_paths,) with the *additional* per-day log-return drift
    contributed by the disequilibrium overlay: -lambda * z, where
    z = (s - equilibrium) / residual_std for each path's current spot.
    """
    if residual_std <= 0:
        return np.zeros_like(s_paths_at_t)
    z = (s_paths_at_t - equilibrium) / residual_std
    return -lambda_daily * z

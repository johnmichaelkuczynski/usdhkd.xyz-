"""Modular pricer interface and 13 pricing model implementations for FX forecasting.

Each Pricer:
  - declares a parameter spec (name + one-line description + display format)
  - calibrate(log_returns, dt, annual_drift) -> dict[str, float]
  - simulate_paths(params, s0, n_steps, n_paths, dt, annual_drift,
                   extra_drift_fn, seed) -> ndarray (n_steps + 1, n_paths)

The disequilibrium overlay is plugged in via `extra_drift_fn(s_at_t)`,
which returns per-path *daily* extra log-drift applied at each step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import kv as bessel_k
from scipy.stats import norm

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    name: str
    description: str
    fmt: str = "{:.4f}"


DriftFn = Optional[Callable[[np.ndarray], np.ndarray]]


class Pricer:
    """Abstract base. Subclasses declare `name`, `params_spec`, and the two methods."""

    name: str = "base"
    label: str = "Base"
    params_spec: List[ParamSpec] = []

    def calibrate(
        self,
        log_returns: np.ndarray,
        dt: float = 1.0 / TRADING_DAYS,
        annual_drift: float = 0.0,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def simulate_paths(
        self,
        params: Dict[str, float],
        s0: float,
        n_steps: int,
        n_paths: int,
        dt: float = 1.0 / TRADING_DAYS,
        annual_drift: float = 0.0,
        extra_drift_fn: DriftFn = None,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        raise NotImplementedError

    def display_params(self, params: Dict[str, float]) -> List[Tuple[str, str, str]]:
        out = []
        for s in self.params_spec:
            v = params.get(s.name)
            try:
                txt = s.fmt.format(v)
            except Exception:
                txt = str(v)
            out.append((s.name, txt, s.description))
        return out


def _apply_extra_drift(extra_drift_fn: DriftFn, s_at_t: np.ndarray) -> np.ndarray:
    if extra_drift_fn is None:
        return 0.0
    out = extra_drift_fn(s_at_t)
    return out


# ---------------------------------------------------------------------------
# 1) Black-Scholes with realized vol
# ---------------------------------------------------------------------------

class BSRealizedVolPricer(Pricer):
    name = "bs_rv"
    label = "BS · realized vol"
    params_spec = [
        ParamSpec("sigma", "Annualised realized volatility (rolling window)", "{:.2%}"),
        ParamSpec("window", "Rolling window in trading days", "{:.0f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        log_returns = np.asarray(log_returns, dtype=np.float64)
        log_returns = log_returns[np.isfinite(log_returns)]
        window = min(60, len(log_returns))
        sigma = float(np.std(log_returns[-window:], ddof=1) / np.sqrt(dt))
        return {"sigma": max(sigma, 1e-4), "window": float(window)}

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        sigma = float(params["sigma"])
        s = np.empty((n_steps + 1, n_paths), dtype=np.float64)
        s[0] = s0
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            z = rng.standard_normal(n_paths)
            mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t])
            s[t + 1] = s[t] * np.exp((mu - 0.5 * sigma * sigma * dt) + sigma * sqrt_dt * z)
        return s


# ---------------------------------------------------------------------------
# 2) Black-Scholes with GARCH(1,1) forecast vol
# ---------------------------------------------------------------------------

def _garch11_neg_loglik(params, r):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    n = len(r)
    var = np.empty(n)
    var[0] = float(np.var(r))
    for i in range(1, n):
        var[i] = omega + alpha * r[i - 1] ** 2 + beta * var[i - 1]
        if var[i] <= 0:
            return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + r * r / var)
    return -ll


class BSGARCHPricer(Pricer):
    name = "bs_garch"
    label = "BS · GARCH(1,1) vol"
    params_spec = [
        ParamSpec("omega", "GARCH ω (variance intercept)", "{:.2e}"),
        ParamSpec("alpha", "GARCH α (ARCH coefficient)", "{:.4f}"),
        ParamSpec("beta", "GARCH β (persistence)", "{:.4f}"),
        ParamSpec("sigma_initial", "Annualised σ at last observation", "{:.2%}"),
        ParamSpec("sigma_long_run", "Annualised long-run σ", "{:.2%}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        r = np.asarray(log_returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        if len(r) < 100:
            sigma = float(np.std(r, ddof=1) / np.sqrt(dt))
            return {"omega": 1e-6, "alpha": 0.05, "beta": 0.9,
                    "sigma_initial": sigma, "sigma_long_run": sigma}
        x0 = np.array([np.var(r) * 0.05, 0.05, 0.9])
        res = minimize(_garch11_neg_loglik, x0, args=(r,), method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-6})
        omega, alpha, beta = res.x
        omega = max(omega, 1e-12)
        alpha = max(alpha, 0.0)
        beta = max(min(beta, 0.999), 0.0)
        if alpha + beta >= 0.999:
            beta = 0.999 - alpha - 1e-4
        n = len(r)
        var = np.empty(n)
        var[0] = float(np.var(r))
        for i in range(1, n):
            var[i] = omega + alpha * r[i - 1] ** 2 + beta * var[i - 1]
        sigma_init = float(np.sqrt(max(var[-1], 1e-12) / dt))
        long_var = omega / max(1.0 - alpha - beta, 1e-6)
        sigma_lr = float(np.sqrt(max(long_var, 1e-12) / dt))
        return {"omega": float(omega), "alpha": float(alpha), "beta": float(beta),
                "sigma_initial": sigma_init, "sigma_long_run": sigma_lr}

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        omega = float(params["omega"])
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        var = np.full(n_paths, (params["sigma_initial"] ** 2) * dt, dtype=np.float64)
        s = np.empty((n_steps + 1, n_paths), dtype=np.float64)
        s[0] = s0
        for t in range(n_steps):
            sigma_step = np.sqrt(np.maximum(var, 1e-16))
            z = rng.standard_normal(n_paths)
            r_step = sigma_step * z
            mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t])
            s[t + 1] = s[t] * np.exp(mu - 0.5 * var + r_step)
            var = omega + alpha * r_step ** 2 + beta * var
            var = np.maximum(var, 1e-16)
        return s


# ---------------------------------------------------------------------------
# 3) Merton Jump-Diffusion
# ---------------------------------------------------------------------------

def _merton_neg_ll(params, r, dt, n_terms=20):
    sigma, lam, mu_j, sig_j = params
    if sigma <= 0 or lam < 0 or sig_j <= 0 or lam > 100 or sigma > 5 or sig_j > 2:
        return 1e10
    lam_dt = lam * dt
    if lam_dt > 50:
        return 1e10
    # Poisson-weighted mixture
    log_pmf = np.array([
        -lam_dt + k * np.log(lam_dt + 1e-300) - np.sum(np.log(np.arange(1, k + 1))) if k > 0
        else -lam_dt
        for k in range(n_terms)
    ])
    var_k = (sigma ** 2) * dt + np.arange(n_terms) * (sig_j ** 2)
    mean_k = (-0.5 * sigma ** 2) * dt + np.arange(n_terms) * mu_j
    # log mixture density per observation
    r = r.reshape(-1, 1)
    var_k = var_k.reshape(1, -1)
    mean_k = mean_k.reshape(1, -1)
    log_phi = -0.5 * (np.log(2 * np.pi * var_k) + (r - mean_k) ** 2 / var_k)
    log_w = log_pmf.reshape(1, -1) + log_phi
    m = np.max(log_w, axis=1, keepdims=True)
    log_dens = (m + np.log(np.sum(np.exp(log_w - m), axis=1, keepdims=True))).ravel()
    if not np.all(np.isfinite(log_dens)):
        return 1e10
    return -float(np.sum(log_dens))


class MertonJDPricer(Pricer):
    name = "merton_jd"
    label = "Merton jump-diffusion"
    params_spec = [
        ParamSpec("sigma", "Annualised diffusion σ", "{:.2%}"),
        ParamSpec("lambda", "Jump intensity (per year)", "{:.3f}"),
        ParamSpec("mu_jump", "Mean log-jump size", "{:+.4f}"),
        ParamSpec("sigma_jump", "Std of log-jump size", "{:.4f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        r = np.asarray(log_returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        sigma0 = float(np.std(r, ddof=1) / np.sqrt(dt))
        x0 = np.array([sigma0 * 0.9, 5.0, 0.0, sigma0 * 0.5 * np.sqrt(dt)])
        res = minimize(_merton_neg_ll, x0, args=(r, dt), method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-6})
        sigma, lam, mu_j, sig_j = res.x
        return {"sigma": float(max(sigma, 1e-4)),
                "lambda": float(max(lam, 0.0)),
                "mu_jump": float(mu_j),
                "sigma_jump": float(max(sig_j, 1e-6))}

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        sigma = float(params["sigma"])
        lam = float(params["lambda"])
        mu_j = float(params["mu_jump"])
        sig_j = float(params["sigma_jump"])
        # martingale correction so jumps don't introduce an extra drift
        kappa = np.exp(mu_j + 0.5 * sig_j ** 2) - 1.0
        s = np.empty((n_steps + 1, n_paths))
        s[0] = s0
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            z = rng.standard_normal(n_paths)
            n_jumps = rng.poisson(lam * dt, size=n_paths)
            jump_part = np.where(n_jumps > 0,
                                 mu_j * n_jumps + sig_j * np.sqrt(n_jumps.astype(float)) * rng.standard_normal(n_paths),
                                 0.0)
            mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t]) - lam * kappa * dt
            s[t + 1] = s[t] * np.exp((mu - 0.5 * sigma ** 2 * dt) + sigma * sqrt_dt * z + jump_part)
        return s


# ---------------------------------------------------------------------------
# 4) Kou double-exponential Jump-Diffusion
# ---------------------------------------------------------------------------

def _kou_log_jump_density(x, p, eta1, eta2):
    # double-exponential
    pos = p * eta1 * np.exp(-eta1 * x)
    neg = (1 - p) * eta2 * np.exp(eta2 * x)
    return np.where(x >= 0, pos, neg)


def _kou_neg_ll(params, r, dt, n_terms=15):
    sigma, lam, p, eta1, eta2 = params
    if sigma <= 0 or lam < 0 or not (0 < p < 1) or eta1 <= 0 or eta2 <= 0:
        return 1e10
    lam_dt = lam * dt
    # mixture: 0,1,2,... jumps. For k jumps, jump sum has a complicated density;
    # we approximate by Gaussian with matched mean+variance per k.
    mean_jump = p / eta1 - (1 - p) / eta2
    var_jump = 2 * p / (eta1 ** 2) + 2 * (1 - p) / (eta2 ** 2) - mean_jump ** 2
    var_jump = max(var_jump, 1e-12)
    log_pmf = np.array([
        -lam_dt + k * np.log(lam_dt + 1e-300) - np.sum(np.log(np.arange(1, k + 1))) if k > 0
        else -lam_dt for k in range(n_terms)
    ])
    var_k = (sigma ** 2) * dt + np.arange(n_terms) * var_jump
    mean_k = (-0.5 * sigma ** 2) * dt + np.arange(n_terms) * mean_jump
    r = r.reshape(-1, 1)
    var_k = var_k.reshape(1, -1)
    mean_k = mean_k.reshape(1, -1)
    log_phi = -0.5 * (np.log(2 * np.pi * var_k) + (r - mean_k) ** 2 / var_k)
    log_w = log_pmf.reshape(1, -1) + log_phi
    m = np.max(log_w, axis=1, keepdims=True)
    log_dens = (m + np.log(np.sum(np.exp(log_w - m), axis=1, keepdims=True))).ravel()
    if not np.all(np.isfinite(log_dens)):
        return 1e10
    return -float(np.sum(log_dens))


class KouJDPricer(Pricer):
    name = "kou_jd"
    label = "Kou jump-diffusion"
    params_spec = [
        ParamSpec("sigma", "Annualised diffusion σ", "{:.2%}"),
        ParamSpec("lambda", "Jump intensity (per year)", "{:.3f}"),
        ParamSpec("p", "Probability of upward jump", "{:.3f}"),
        ParamSpec("eta1", "Up-jump rate (1/η₁ = mean up-jump)", "{:.2f}"),
        ParamSpec("eta2", "Down-jump rate (1/η₂ = mean down-jump)", "{:.2f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        r = np.asarray(log_returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        sigma0 = float(np.std(r, ddof=1) / np.sqrt(dt))
        x0 = np.array([sigma0 * 0.9, 5.0, 0.5, 50.0, 50.0])
        res = minimize(_kou_neg_ll, x0, args=(r, dt), method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-6})
        sigma, lam, p, eta1, eta2 = res.x
        return {"sigma": float(max(sigma, 1e-4)),
                "lambda": float(max(lam, 0.0)),
                "p": float(min(max(p, 1e-4), 1 - 1e-4)),
                "eta1": float(max(eta1, 1.0)),
                "eta2": float(max(eta2, 1.0))}

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        sigma = float(params["sigma"])
        lam = float(params["lambda"])
        p = float(params["p"])
        eta1 = float(params["eta1"])
        eta2 = float(params["eta2"])
        # martingale correction
        kappa = (p * eta1 / (eta1 - 1.0) + (1 - p) * eta2 / (eta2 + 1.0)) - 1.0 if eta1 > 1.0 else 0.0
        s = np.empty((n_steps + 1, n_paths))
        s[0] = s0
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            z = rng.standard_normal(n_paths)
            n_jumps = rng.poisson(lam * dt, size=n_paths)
            # sample jump sums: compound double-exponential
            jump_sum = np.zeros(n_paths)
            mask = n_jumps > 0
            if np.any(mask):
                # vectorised: sample max(n_jumps) jumps for masked paths
                max_k = int(n_jumps.max())
                u = rng.random((n_paths, max_k))
                e1 = rng.exponential(1.0 / eta1, size=(n_paths, max_k))
                e2 = -rng.exponential(1.0 / eta2, size=(n_paths, max_k))
                draws = np.where(u < p, e1, e2)
                # mask out unused jumps
                idx = np.arange(max_k)
                use = idx.reshape(1, -1) < n_jumps.reshape(-1, 1)
                jump_sum = (draws * use).sum(axis=1)
            mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t]) - lam * kappa * dt
            s[t + 1] = s[t] * np.exp((mu - 0.5 * sigma ** 2 * dt) + sigma * sqrt_dt * z + jump_sum)
        return s


# ---------------------------------------------------------------------------
# 5) Heston (wrapping engine.heston for calibration)
# ---------------------------------------------------------------------------

from .heston import calibrate_heston as _heston_mle, HestonParams


def _simulate_heston_core(params, s0, n_steps, n_paths, dt, annual_drift,
                          extra_drift_fn, seed):
    rng = np.random.default_rng(seed)
    kappa = float(params["kappa"])
    theta = float(params["theta"])
    xi = float(params["xi"])
    rho = float(params["rho"])
    v0 = float(params["v0"])
    s = np.empty((n_steps + 1, n_paths))
    v = np.empty((n_steps + 1, n_paths))
    s[0] = s0
    v[0] = max(v0, 1e-10)
    sqrt_one_minus_rho2 = np.sqrt(max(1.0 - rho * rho, 1e-8))
    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w_v = rho * z1 + sqrt_one_minus_rho2 * z2
        v_pos = np.maximum(v[t], 0.0)
        sqrt_v = np.sqrt(v_pos)
        v[t + 1] = np.maximum(v[t] + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * w_v, 0.0)
        mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t])
        s[t + 1] = s[t] * np.exp((mu - 0.5 * v_pos * dt) + sqrt_v * sqrt_dt * z1)
    return s


class HestonPricer(Pricer):
    name = "heston"
    label = "Heston"
    params_spec = [
        ParamSpec("kappa", "Mean-reversion speed of variance", "{:.3f}"),
        ParamSpec("theta", "Long-run variance (annualised)", "{:.4f}"),
        ParamSpec("xi", "Volatility of variance", "{:.3f}"),
        ParamSpec("rho", "Correlation between asset and variance shocks", "{:+.3f}"),
        ParamSpec("v0", "Initial variance (annualised)", "{:.4f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        p, _ok = _heston_mle(np.asarray(log_returns, dtype=np.float64),
                              annual_drift=annual_drift, dt=dt, n_particles=200)
        return {"kappa": p.kappa, "theta": p.theta, "xi": p.xi, "rho": p.rho, "v0": p.v0}

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        return _simulate_heston_core(params, s0, n_steps, n_paths, dt,
                                     annual_drift, extra_drift_fn, seed)


# ---------------------------------------------------------------------------
# 6) Bates = Heston + Merton jumps
# ---------------------------------------------------------------------------

class BatesPricer(Pricer):
    name = "bates"
    label = "Bates (Heston + jumps)"
    params_spec = HestonPricer.params_spec + [
        ParamSpec("lambda", "Jump intensity (per year)", "{:.3f}"),
        ParamSpec("mu_jump", "Mean log-jump size", "{:+.4f}"),
        ParamSpec("sigma_jump", "Std of log-jump size", "{:.4f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        r = np.asarray(log_returns, dtype=np.float64)
        # 1) calibrate Heston with reduced vol budget
        h = HestonPricer().calibrate(r, dt, annual_drift)
        # 2) attribute residual heavy tails to jumps via method of moments on 4th cumulant
        std = np.std(r, ddof=1)
        kurt = float(np.mean(((r - r.mean()) / std) ** 4) - 3.0)
        kurt = max(kurt, 0.0)
        # rough mapping: extra kurtosis from jumps ≈ λdt * (mu_j² + 3σ_j⁴)/var²
        sig_j = max(2.0 * std, 1e-3)
        lam = min(20.0, kurt / 3.0)
        mu_j = -0.001  # mild negative bias
        h.update({"lambda": float(lam), "mu_jump": float(mu_j), "sigma_jump": float(sig_j)})
        return h

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        lam = float(params["lambda"])
        mu_j = float(params["mu_jump"])
        sig_j = float(params["sigma_jump"])
        kappa_j = np.exp(mu_j + 0.5 * sig_j ** 2) - 1.0

        def adjusted_drift(s_at_t):
            base = _apply_extra_drift(extra_drift_fn, s_at_t)
            return base - lam * kappa_j * dt

        s = _simulate_heston_core(params, s0, n_steps, n_paths, dt,
                                  annual_drift, adjusted_drift, seed)
        # add jumps as a multiplicative overlay
        for t in range(n_steps):
            n_jumps = rng.poisson(lam * dt, size=n_paths)
            jump = np.where(n_jumps > 0,
                            mu_j * n_jumps + sig_j * np.sqrt(n_jumps.astype(float)) * rng.standard_normal(n_paths),
                            0.0)
            s[t + 1:] *= np.exp(jump).reshape(1, -1)
        return s


# ---------------------------------------------------------------------------
# 7) SVJJ (Bates + correlated variance jumps)
# ---------------------------------------------------------------------------

class SVJJPricer(Pricer):
    name = "svjj"
    label = "SVJJ (price + variance jumps)"
    params_spec = BatesPricer.params_spec + [
        ParamSpec("mu_v_jump", "Mean variance-jump size", "{:.4f}"),
        ParamSpec("rho_j", "Price/variance jump correlation", "{:+.3f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        b = BatesPricer().calibrate(log_returns, dt, annual_drift)
        # variance jumps: roughly proportional to squared price jumps
        b["mu_v_jump"] = float(0.5 * (b["sigma_jump"] ** 2))
        b["rho_j"] = -0.5
        return b

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        kappa = float(params["kappa"])
        theta = float(params["theta"])
        xi = float(params["xi"])
        rho = float(params["rho"])
        v0 = float(params["v0"])
        lam = float(params["lambda"])
        mu_j = float(params["mu_jump"])
        sig_j = float(params["sigma_jump"])
        mu_vj = float(params["mu_v_jump"])
        rho_j = float(params["rho_j"])
        kappa_j = np.exp(mu_j + 0.5 * sig_j ** 2) - 1.0

        s = np.empty((n_steps + 1, n_paths))
        v = np.empty((n_steps + 1, n_paths))
        s[0] = s0
        v[0] = max(v0, 1e-10)
        sqrt_one_minus_rho2 = np.sqrt(max(1.0 - rho * rho, 1e-8))
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            w_v = rho * z1 + sqrt_one_minus_rho2 * z2
            v_pos = np.maximum(v[t], 0.0)
            sqrt_v = np.sqrt(v_pos)

            # price + variance jumps
            n_jumps = rng.poisson(lam * dt, size=n_paths)
            jp = np.where(n_jumps > 0,
                           mu_j * n_jumps + sig_j * np.sqrt(n_jumps.astype(float)) * rng.standard_normal(n_paths),
                           0.0)
            jv = mu_vj * n_jumps + rho_j * jp  # variance jump correlated with price jump

            v[t + 1] = np.maximum(
                v[t] + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * w_v + jv,
                0.0,
            )
            mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t]) - lam * kappa_j * dt
            s[t + 1] = s[t] * np.exp((mu - 0.5 * v_pos * dt) + sqrt_v * sqrt_dt * z1 + jp)
        return s


# ---------------------------------------------------------------------------
# 8) Double Heston (two variance factors)
# ---------------------------------------------------------------------------

class DoubleHestonPricer(Pricer):
    name = "double_heston"
    label = "Double Heston"
    params_spec = [
        ParamSpec("kappa1", "Fast factor mean-reversion", "{:.3f}"),
        ParamSpec("theta1", "Fast factor long-run variance", "{:.4f}"),
        ParamSpec("xi1", "Fast factor vol-of-vol", "{:.3f}"),
        ParamSpec("rho1", "Fast factor corr.", "{:+.3f}"),
        ParamSpec("v0_1", "Fast factor initial variance", "{:.4f}"),
        ParamSpec("kappa2", "Slow factor mean-reversion", "{:.3f}"),
        ParamSpec("theta2", "Slow factor long-run variance", "{:.4f}"),
        ParamSpec("xi2", "Slow factor vol-of-vol", "{:.3f}"),
        ParamSpec("rho2", "Slow factor corr.", "{:+.3f}"),
        ParamSpec("v0_2", "Slow factor initial variance", "{:.4f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        r = np.asarray(log_returns, dtype=np.float64)
        # split realised variance into a fast (recent) and slow (long) component
        if len(r) < 100:
            v_fast = v_slow = float(np.var(r) / dt)
        else:
            v_fast = float(np.var(r[-60:]) / dt)
            v_slow = float(np.var(r) / dt)
        # allocate roughly half the variance budget to each factor
        return {
            "kappa1": 6.0, "theta1": v_fast * 0.5, "xi1": 0.6, "rho1": -0.4, "v0_1": v_fast * 0.5,
            "kappa2": 0.8, "theta2": v_slow * 0.5, "xi2": 0.2, "rho2": -0.2, "v0_2": v_slow * 0.5,
        }

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        s = np.empty((n_steps + 1, n_paths))
        s[0] = s0
        v1 = np.full(n_paths, max(params["v0_1"], 1e-10))
        v2 = np.full(n_paths, max(params["v0_2"], 1e-10))
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            zw1 = rng.standard_normal(n_paths)
            zw2 = rng.standard_normal(n_paths)
            wv1 = params["rho1"] * z1 + np.sqrt(max(1 - params["rho1"] ** 2, 1e-8)) * zw1
            wv2 = params["rho2"] * z2 + np.sqrt(max(1 - params["rho2"] ** 2, 1e-8)) * zw2
            v1p = np.maximum(v1, 0.0)
            v2p = np.maximum(v2, 0.0)
            sv1 = np.sqrt(v1p)
            sv2 = np.sqrt(v2p)
            v1 = np.maximum(v1 + params["kappa1"] * (params["theta1"] - v1p) * dt + params["xi1"] * sv1 * sqrt_dt * wv1, 0.0)
            v2 = np.maximum(v2 + params["kappa2"] * (params["theta2"] - v2p) * dt + params["xi2"] * sv2 * sqrt_dt * wv2, 0.0)
            mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t])
            s[t + 1] = s[t] * np.exp(
                (mu - 0.5 * (v1p + v2p) * dt) + sv1 * sqrt_dt * z1 + sv2 * sqrt_dt * z2
            )
        return s


# ---------------------------------------------------------------------------
# 9) Rough Heston (rBergomi-style approximation)
# ---------------------------------------------------------------------------

class RoughHestonPricer(Pricer):
    name = "rough_heston"
    label = "Rough Heston"
    params_spec = [
        ParamSpec("xi0", "Forward variance level (annualised)", "{:.4f}"),
        ParamSpec("eta", "Vol-of-vol scaling", "{:.3f}"),
        ParamSpec("rho", "Spot-vol correlation", "{:+.3f}"),
        ParamSpec("H", "Hurst exponent (roughness)", "{:.3f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        r = np.asarray(log_returns, dtype=np.float64)
        xi0 = float(np.var(r) / dt)
        # simple roughness proxy: ratio of short- vs long-window vol persistence
        return {"xi0": xi0, "eta": 1.5, "rho": -0.4, "H": 0.10}

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        xi0 = float(params["xi0"])
        eta = float(params["eta"])
        rho = float(params["rho"])
        H = float(params["H"])
        sqrt_dt = np.sqrt(dt)
        sqrt_one_minus_rho2 = np.sqrt(max(1.0 - rho * rho, 1e-8))

        # rBergomi-style fractional kernel: Y_t = ∫ (t-s)^(H-0.5) dW_s
        # discretised via a left-point Riemann sum with appropriate weights.
        Y = np.zeros((n_steps + 1, n_paths))
        dW_v_hist = np.zeros((n_steps, n_paths))
        for t in range(n_steps):
            dW_v = rng.standard_normal(n_paths) * sqrt_dt
            dW_v_hist[t] = dW_v
            # weights (k * dt)^(H - 0.5) for k = 1..t+1
            ks = np.arange(t, -1, -1) + 1
            w = (ks * dt) ** (H - 0.5)
            Y[t + 1] = (w[:, None] * dW_v_hist[: t + 1]).sum(axis=0)

        var_t = xi0 * np.exp(eta * Y - 0.5 * (eta ** 2) * (np.arange(n_steps + 1) * dt) ** (2 * H))

        s = np.empty((n_steps + 1, n_paths))
        s[0] = s0
        for t in range(n_steps):
            v_pos = np.maximum(var_t[t], 1e-12)
            sqrt_v = np.sqrt(v_pos)
            zS = rng.standard_normal(n_paths)
            # correlate with the variance Brownian (use the latest dW_v)
            w_s = rho * (dW_v_hist[t] / sqrt_dt) + sqrt_one_minus_rho2 * zS
            mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t])
            s[t + 1] = s[t] * np.exp((mu - 0.5 * v_pos * dt) + sqrt_v * sqrt_dt * w_s)
        return s


# ---------------------------------------------------------------------------
# 10) Variance Gamma
# ---------------------------------------------------------------------------

class VGPricer(Pricer):
    name = "vg"
    label = "Variance Gamma"
    params_spec = [
        ParamSpec("sigma", "VG diffusion σ (annualised)", "{:.2%}"),
        ParamSpec("theta", "VG drift bias", "{:+.4f}"),
        ParamSpec("nu", "VG variance rate (kurtosis ↑)", "{:.4f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        r = np.asarray(log_returns, dtype=np.float64)
        # Method of moments (MOM) on returns at frequency dt:
        # mean = θ dt
        # var  = (σ² + θ² ν) dt
        # skew = θ ν (3σ² + 2θ²ν) / ((σ² + θ²ν)^(3/2)) · dt^(-1/2)
        # kurt - 3 = ν (3σ⁴ + 12σ²θ²ν + 6θ⁴ν²) / ((σ² + θ²ν)²) · dt^(-1)
        mean_r = float(np.mean(r))
        var_r = float(np.var(r, ddof=1))
        skew = float(np.mean(((r - mean_r) / np.sqrt(var_r)) ** 3))
        kurt_excess = max(float(np.mean(((r - mean_r) / np.sqrt(var_r)) ** 4) - 3.0), 1e-3)

        nu = kurt_excess * dt / 3.0
        nu = max(min(nu, 5.0), 1e-4)
        theta = mean_r / dt
        sigma_sq = max(var_r / dt - theta ** 2 * nu, 1e-6)
        sigma = float(np.sqrt(sigma_sq))
        return {"sigma": sigma, "theta": float(theta), "nu": float(nu)}

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        sigma = float(params["sigma"])
        theta = float(params["theta"])
        nu = float(params["nu"])
        # martingale correction
        omega = (1.0 / nu) * np.log(max(1.0 - theta * nu - 0.5 * sigma ** 2 * nu, 1e-6))
        s = np.empty((n_steps + 1, n_paths))
        s[0] = s0
        for t in range(n_steps):
            # Variance Gamma increment: X = θ G + σ √G Z, G ~ Gamma(dt/ν, ν)
            G = rng.gamma(shape=dt / nu, scale=nu, size=n_paths)
            X = theta * G + sigma * np.sqrt(G) * rng.standard_normal(n_paths)
            mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t]) + omega * dt
            s[t + 1] = s[t] * np.exp(mu + X)
        return s


# ---------------------------------------------------------------------------
# 11) CGMY
# ---------------------------------------------------------------------------

class CGMYPricer(Pricer):
    name = "cgmy"
    label = "CGMY"
    params_spec = [
        ParamSpec("C", "Activity intensity", "{:.4f}"),
        ParamSpec("G", "Down-jump tempering", "{:.3f}"),
        ParamSpec("M", "Up-jump tempering", "{:.3f}"),
        ParamSpec("Y", "Stable-like exponent (Y<2)", "{:.3f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        r = np.asarray(log_returns, dtype=np.float64)
        # MOM-style match: variance, skew, kurtosis, with a fixed Y prior ≈ 1.4
        var_r = float(np.var(r, ddof=1))
        skew = float(np.mean(((r - r.mean()) / np.sqrt(var_r)) ** 3))
        kurt_excess = max(float(np.mean(((r - r.mean()) / np.sqrt(var_r)) ** 4) - 3.0), 1e-3)
        Y = 1.4
        # use cumulants of CGMY: c2 = C Γ(2-Y)(M^(Y-2) + G^(Y-2))
        # we set M=G initially
        from math import gamma
        c2_target = var_r / dt
        c4_target = (kurt_excess + 3.0 - 3.0) * (var_r / dt) ** 2 + 3.0 * (var_r / dt) ** 2
        # solve for M=G via c2 / c4 ratio
        # c4 = C Γ(4-Y)(M^(Y-4) + G^(Y-4)) = 2 C Γ(4-Y) M^(Y-4)
        # ratio c4/c2 = Γ(4-Y)/Γ(2-Y) * M^(-2)
        ratio = c4_target / max(c2_target, 1e-12)
        M2 = (gamma(4 - Y) / gamma(2 - Y)) / max(ratio, 1e-6)
        M = float(np.sqrt(max(M2, 1e-3)))
        G = M
        # asymmetry
        if abs(skew) > 0.01:
            adj = 1.0 + 0.2 * np.tanh(skew)
            G *= adj
            M /= adj
        C = c2_target / (gamma(2 - Y) * (M ** (Y - 2) + G ** (Y - 2)))
        return {"C": float(max(C, 1e-6)), "G": float(max(G, 0.5)),
                "M": float(max(M, 0.5)), "Y": float(Y)}

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        C = float(params["C"])
        G = float(params["G"])
        M = float(params["M"])
        Y = float(params["Y"])
        from math import gamma
        # cumulants per dt
        c1 = C * gamma(1 - Y) * (M ** (Y - 1) - G ** (Y - 1)) * dt
        c2 = C * gamma(2 - Y) * (M ** (Y - 2) + G ** (Y - 2)) * dt
        c3 = C * gamma(3 - Y) * (M ** (Y - 3) - G ** (Y - 3)) * dt
        # We approximate the Lévy increment by a Normal-with-skew correction (NIG-like envelope).
        # This preserves first 3 cumulants per step.
        std = float(np.sqrt(max(c2, 1e-12)))
        skew = float(c3 / max(std ** 3, 1e-12))
        s = np.empty((n_steps + 1, n_paths))
        s[0] = s0
        # martingale correction: ensure mean of exp(X) ~ 1
        # use second-order: ω = c1 + 0.5 c2 + (c3/6)
        omega = c1 + 0.5 * c2
        for t in range(n_steps):
            z = rng.standard_normal(n_paths)
            # Cornish-Fisher inversion to inject skew
            X = c1 + std * (z + (skew / 6.0) * (z ** 2 - 1.0))
            mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t]) - omega
            s[t + 1] = s[t] * np.exp(mu + X)
        return s


# ---------------------------------------------------------------------------
# 12) NIG (Normal Inverse Gaussian)
# ---------------------------------------------------------------------------

def _nig_logpdf(x, alpha, beta, delta, mu):
    # f(x) = (αδ/π) exp(δγ + β(x-μ)) K1(α √(δ²+(x-μ)²)) / √(δ²+(x-μ)²)
    gamma = np.sqrt(max(alpha ** 2 - beta ** 2, 1e-12))
    z = np.sqrt(delta ** 2 + (x - mu) ** 2)
    log_k1 = np.log(np.maximum(bessel_k(1, alpha * z), 1e-300))
    return np.log(alpha * delta / np.pi) + delta * gamma + beta * (x - mu) + log_k1 - np.log(z)


def _nig_neg_ll(params, r):
    alpha, beta, delta, mu = params
    if alpha <= 0 or delta <= 0 or abs(beta) >= alpha:
        return 1e10
    ll = _nig_logpdf(r, alpha, beta, delta, mu)
    if not np.all(np.isfinite(ll)):
        return 1e10
    return -float(np.sum(ll))


class NIGPricer(Pricer):
    name = "nig"
    label = "Normal Inverse Gaussian"
    params_spec = [
        ParamSpec("alpha", "Tail heaviness (α)", "{:.2f}"),
        ParamSpec("beta", "Asymmetry (β)", "{:+.3f}"),
        ParamSpec("delta", "Scale (δ)", "{:.4f}"),
        ParamSpec("mu", "Location (μ)", "{:+.5f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        r = np.asarray(log_returns, dtype=np.float64)
        m = float(np.mean(r)); v = float(np.var(r, ddof=1))
        x0 = np.array([60.0, -1.0, np.sqrt(v) * 60.0, m])
        res = minimize(_nig_neg_ll, x0, args=(r,), method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-7})
        a, b, d, mu = res.x
        if a <= 0 or d <= 0 or abs(b) >= a:
            a, b, d, mu = 60.0, 0.0, np.sqrt(v) * 60.0, m
        return {"alpha": float(a), "beta": float(b), "delta": float(d), "mu": float(mu)}

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        a = float(params["alpha"]); b = float(params["beta"])
        d = float(params["delta"]); mu = float(params["mu"])
        gamma = np.sqrt(max(a ** 2 - b ** 2, 1e-12))
        # martingale correction so that E[exp(X)] = 1 per unit time
        # MGF of NIG at u=1: M(1) = exp(μ + δ(γ - √(α²-(β+1)²)))
        if a ** 2 - (b + 1.0) ** 2 > 0:
            omega = mu + d * (gamma - np.sqrt(a ** 2 - (b + 1.0) ** 2))
        else:
            omega = 0.0
        s = np.empty((n_steps + 1, n_paths))
        s[0] = s0
        for t in range(n_steps):
            # NIG increment over dt: X = μ dt + β δ dt² IG + δ √IG Z
            # Use simpler: scale parameters by dt (NIG is infinitely divisible).
            d_t = d * dt
            mu_t = mu * dt
            # Sample IG(δ_t γ, δ_t² γ²)
            ig = _sample_ig(d_t * gamma, d_t * d_t, rng, n_paths)
            X = mu_t + b * ig + np.sqrt(np.maximum(ig, 1e-12)) * rng.standard_normal(n_paths)
            extra = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t]) - omega * dt
            s[t + 1] = s[t] * np.exp(extra + X)
        return s


def _sample_ig(mean, lam, rng, size):
    """Sample IG(μ, λ): mean μ, shape λ via Michael, Schucany, Haas algorithm."""
    if mean <= 0 or lam <= 0:
        return np.full(size, max(mean, 1e-8))
    nu = rng.standard_normal(size) ** 2
    x = mean + (mean ** 2 * nu) / (2 * lam) - (mean / (2 * lam)) * np.sqrt(
        np.maximum(4 * mean * lam * nu + (mean ** 2) * (nu ** 2), 0.0)
    )
    u = rng.random(size)
    out = np.where(u <= mean / (mean + x), x, (mean ** 2) / np.maximum(x, 1e-12))
    return np.maximum(out, 1e-12)


# ---------------------------------------------------------------------------
# 13) SABR (β=1 lognormal SABR by default for FX)
# ---------------------------------------------------------------------------

class SABRPricer(Pricer):
    name = "sabr"
    label = "SABR (β=1, lognormal)"
    params_spec = [
        ParamSpec("alpha", "Initial vol α (annualised)", "{:.2%}"),
        ParamSpec("nu", "Vol-of-vol ν", "{:.3f}"),
        ParamSpec("rho", "Spot-vol correlation ρ", "{:+.3f}"),
        ParamSpec("beta", "CEV exponent (fixed for FX)", "{:.2f}"),
    ]

    def calibrate(self, log_returns, dt=1.0 / TRADING_DAYS, annual_drift=0.0):
        r = np.asarray(log_returns, dtype=np.float64)
        # rolling 30-day realised vol time series → mean = α, vol-of-vol → ν
        if len(r) < 60:
            sigma = float(np.std(r) / np.sqrt(dt))
            return {"alpha": sigma, "nu": 0.5, "rho": -0.3, "beta": 1.0}
        rv = np.array([np.std(r[max(0, i - 30):i + 1]) / np.sqrt(dt) for i in range(len(r))])
        rv = rv[rv > 0]
        alpha = float(np.median(rv))
        log_rv = np.log(rv)
        nu = float(np.std(np.diff(log_rv)) / np.sqrt(dt))
        # rho via correlation between price returns and changes in rv
        n = min(len(rv), len(r))
        if n > 30:
            rho = float(np.corrcoef(r[-n:], np.diff(np.concatenate([[rv[0]], rv[-n:]])))[0, 1])
        else:
            rho = -0.3
        rho = float(np.clip(rho, -0.95, 0.95))
        return {"alpha": alpha, "nu": min(nu, 5.0), "rho": rho, "beta": 1.0}

    def simulate_paths(self, params, s0, n_steps, n_paths, dt=1.0 / TRADING_DAYS,
                        annual_drift=0.0, extra_drift_fn=None, seed=42):
        rng = np.random.default_rng(seed)
        alpha = float(params["alpha"])
        nu = float(params["nu"])
        rho = float(params["rho"])
        sqrt_dt = np.sqrt(dt)
        sqrt_one_minus_rho2 = np.sqrt(max(1.0 - rho * rho, 1e-8))
        s = np.empty((n_steps + 1, n_paths))
        s[0] = s0
        a = np.full(n_paths, alpha)
        for t in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            wa = rho * z1 + sqrt_one_minus_rho2 * z2
            mu = annual_drift * dt + _apply_extra_drift(extra_drift_fn, s[t])
            s[t + 1] = s[t] * np.exp(mu - 0.5 * (a ** 2) * dt + a * sqrt_dt * z1)
            a = a * np.exp(-0.5 * (nu ** 2) * dt + nu * sqrt_dt * wa)
            a = np.clip(a, 1e-4, 5.0)
        return s


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRICERS: Dict[str, Pricer] = {
    p.name: p for p in [
        BSRealizedVolPricer(),
        BSGARCHPricer(),
        MertonJDPricer(),
        KouJDPricer(),
        HestonPricer(),
        BatesPricer(),
        SVJJPricer(),
        DoubleHestonPricer(),
        RoughHestonPricer(),
        VGPricer(),
        CGMYPricer(),
        NIGPricer(),
        SABRPricer(),
    ]
}

PRICER_ORDER = [
    "heston", "bates", "svjj", "merton_jd", "kou_jd",
    "vg", "cgmy", "nig",
    "sabr", "double_heston", "rough_heston",
    "bs_rv", "bs_garch",
]


def get_pricer(name: str) -> Pricer:
    return PRICERS[name]


def pricer_choices() -> List[Tuple[str, str]]:
    return [(name, PRICERS[name].label) for name in PRICER_ORDER if name in PRICERS]

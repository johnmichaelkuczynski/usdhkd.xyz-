"""Self-check / diagnostic for USD/HKD Edge.

Runs a system + functional self-check and returns a structured JSON-ready
dict shaped like:

    {
      "ok": bool, "runAt": ISO timestamp,
      "totals": {"pass": N, "fail": N, "skip": N},
      "checks": [
        {"name": ..., "group": "system"|"functional",
         "status": "pass"|"fail"|"skip",
         "ms": int, "info": str,
         "evidence": [{"kind": ..., "label": ..., "value": ...}, ...]},
        ...
      ]
    }

Every check is wrapped in try/except so a broken check never crashes the run.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _redact_len(v: str | None) -> dict:
    return {"present": bool(v), "length": len(v) if v else 0}


class _Reporter:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def run(
        self,
        name: str,
        group: str,
        fn: Callable[[], tuple[str, str, list[dict]]],
        *,
        skip_if: Callable[[], tuple[bool, str]] | None = None,
    ) -> None:
        if skip_if is not None:
            try:
                should_skip, reason = skip_if()
            except Exception as e:  # noqa: BLE001
                should_skip, reason = True, f"skip predicate failed: {e}"
            if should_skip:
                self.checks.append({
                    "name": name, "group": group, "status": "skip",
                    "ms": 0, "info": reason, "evidence": [],
                })
                return

        t0 = time.perf_counter()
        try:
            status, info, evidence = fn()
        except Exception as e:  # noqa: BLE001
            self.checks.append({
                "name": name, "group": group, "status": "fail",
                "ms": int((time.perf_counter() - t0) * 1000),
                "info": str(e),
                "evidence": [{"kind": "error", "label": "exception", "value": str(e)}],
            })
            return
        self.checks.append({
            "name": name, "group": group, "status": status,
            "ms": int((time.perf_counter() - t0) * 1000),
            "info": info, "evidence": evidence,
        })


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

def _check_env_required(var: str) -> tuple[str, str, list[dict]]:
    v = os.environ.get(var)
    ev = [{"kind": "assertion", "label": f"os.environ['{var}'] is non-empty",
           "value": _redact_len(v)}]
    if v:
        return "pass", f"length={len(v)} chars (value redacted)", ev
    return "fail", "not set", ev


def _check_env_optional(var: str) -> tuple[str, str, list[dict]]:
    v = os.environ.get(var)
    ev = [{"kind": "assertion", "label": f"os.environ['{var}'] is non-empty",
           "value": _redact_len(v)}]
    if v:
        return "pass", f"length={len(v)} chars (value redacted)", ev
    return "skip", "not set (optional)", ev


def _check_cache_dir() -> tuple[str, str, list[dict]]:
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    probe = cache_dir / ".diagnostic_probe"
    probe.write_text("ok")
    probe.unlink()
    return "pass", f"writable at {cache_dir.resolve()}", [
        {"kind": "output", "label": "cache dir", "value": str(cache_dir.resolve())},
    ]


def _check_fx_history() -> tuple[str, str, list[dict]]:
    from data.eodhd_fx import fetch_usdhkd_history, latest_close
    fx = fetch_usdhkd_history(years=5)
    n = int(len(fx))
    last_close = float(latest_close(fx))
    last_date = str(fx.index[-1])
    in_band = 7.7 <= last_close <= 7.9
    ev = [
        {"kind": "output", "label": "rows", "value": n},
        {"kind": "output", "label": "last_date", "value": last_date},
        {"kind": "output", "label": "last_close", "value": last_close},
        {"kind": "assertion", "label": "last_close in [7.70, 7.90]",
         "value": {"ok": in_band}},
    ]
    if n < 250 or not in_band:
        return "fail", f"n={n} last_close={last_close:.4f} ({last_date})", ev
    return "pass", f"{n} rows, last close {last_close:.4f} on {last_date[:10]}", ev


def _check_rate_differential() -> tuple[str, str, list[dict]]:
    from data.rates import build_rate_differential
    rates = build_rate_differential()
    n = int(len(rates))
    last_diff = float(rates["diff"].iloc[-1])
    last_date = str(rates.index[-1])
    ev = [
        {"kind": "output", "label": "rows", "value": n},
        {"kind": "output", "label": "last_date", "value": last_date},
        {"kind": "output", "label": "last_diff_pct", "value": last_diff},
        {"kind": "assertion", "label": "diff finite", "value": {"ok": np.isfinite(last_diff)}},
    ]
    if n < 250 or not np.isfinite(last_diff):
        return "fail", f"n={n}, last_diff={last_diff}", ev
    return "pass", f"{n} rows, last diff {last_diff:+.2f}% on {last_date[:10]}", ev


def _check_equilibrium() -> tuple[str, str, list[dict]]:
    import pandas as pd
    from data.eodhd_fx import fetch_usdhkd_history
    from data.rates import build_rate_differential
    from engine.disequilibrium_fx import fit_equilibrium

    fx = fetch_usdhkd_history(years=5)
    rates = build_rate_differential()
    fx_close = fx["close"].copy()
    fx_close.index = pd.to_datetime(fx_close.index).tz_localize(None)
    rates.index = pd.to_datetime(rates.index).tz_localize(None)
    joined = pd.concat(
        [fx_close.rename("usdhkd"), rates["diff"].rename("diff")], axis=1
    ).dropna()
    eq = fit_equilibrium(joined["usdhkd"], joined["diff"], rolling_window=252)
    fields = {
        "alpha": float(eq.alpha), "beta": float(eq.beta),
        "residual_sigma": float(eq.residual_sigma),
        "lambda_per_day": float(eq.lambda_),
        "z_score": float(eq.z_score),
        "equilibrium": float(eq.equilibrium),
    }
    finite = all(np.isfinite(v) for v in fields.values())
    plausible = (7.6 <= fields["alpha"] <= 8.0) and fields["residual_sigma"] > 0
    ev = [
        {"kind": "output", "label": "fit", "value": fields},
        {"kind": "assertion", "label": "all finite", "value": {"ok": finite}},
        {"kind": "assertion", "label": "alpha plausible + sigma > 0",
         "value": {"ok": plausible}},
    ]
    if not (finite and plausible):
        return "fail", f"finite={finite} plausible={plausible}", ev
    return "pass", (f"α={fields['alpha']:.3f} β={fields['beta']:.3f} "
                    f"σ={fields['residual_sigma']:.4f}"), ev


def _check_pricer(model_name: str) -> tuple[str, str, list[dict]]:
    from engine.pricers import get_pricer, TRADING_DAYS
    from data.eodhd_fx import fetch_usdhkd_history, daily_log_returns

    fx = fetch_usdhkd_history(years=2)
    log_rets = daily_log_returns(fx).to_numpy()
    pricer = get_pricer(model_name)
    dt = 1.0 / TRADING_DAYS
    annual_drift = float(np.mean(log_rets) * TRADING_DAYS)
    t_cal0 = time.perf_counter()
    params = pricer.calibrate(log_rets, dt=dt, annual_drift=annual_drift)
    t_cal = time.perf_counter() - t_cal0

    t_sim0 = time.perf_counter()
    paths = pricer.simulate_paths(
        s0=float(fx["close"].iloc[-1]),
        n_paths=200,
        n_steps=10,
        dt=dt,
        params=params,
        seed=42,
    )
    t_sim = time.perf_counter() - t_sim0

    # pricers return shape (n_steps+1, n_paths) — terminals are the last row.
    if paths.ndim == 2:
        terminals = paths[-1, :]
    else:
        terminals = paths
    fin = float(np.mean(np.isfinite(terminals)))
    med = float(np.median(terminals[np.isfinite(terminals)])) if fin > 0 else float("nan")
    in_band = np.isfinite(med) and 7.4 <= med <= 8.2 and fin > 0.999
    ev = [
        {"kind": "output", "label": "calibration ms", "value": int(t_cal * 1000)},
        {"kind": "output", "label": "simulation ms", "value": int(t_sim * 1000)},
        {"kind": "output", "label": "paths shape (steps+1, paths)", "value": list(paths.shape)},
        {"kind": "output", "label": "median terminal", "value": med},
        {"kind": "output", "label": "fraction finite terminals", "value": fin},
        {"kind": "output", "label": "param keys", "value": list(params.keys())},
    ]
    if not in_band:
        return "fail", f"median={med:.4f} finite={fin:.3f}", ev
    return "pass", f"calibrated in {int(t_cal * 1000)}ms, sim {int(t_sim * 1000)}ms, median {med:.4f}", ev


def _check_backtest_smoke() -> tuple[str, str, list[dict]]:
    import pandas as pd
    from data.eodhd_fx import fetch_usdhkd_history
    from data.rates import build_rate_differential
    from engine.disequilibrium_fx import fit_equilibrium
    from engine.backtest import run_single_model_backtest

    fx = fetch_usdhkd_history(years=2)
    rates = build_rate_differential()
    fx_close = fx["close"].copy()
    fx_close.index = pd.to_datetime(fx_close.index).tz_localize(None)
    rates.index = pd.to_datetime(rates.index).tz_localize(None)
    fx_aligned = fx.copy()
    fx_aligned.index = pd.to_datetime(fx_aligned.index).tz_localize(None)
    joined = pd.concat(
        [fx_close.rename("usdhkd"), rates["diff"].rename("diff")], axis=1
    ).dropna()
    eq = fit_equilibrium(joined["usdhkd"], joined["diff"], rolling_window=252)

    end = fx_aligned.index[-1]
    start = end - pd.Timedelta(days=365)

    t0 = time.perf_counter()
    bt = run_single_model_backtest(
        model="bs_rv",
        fx=fx_aligned,
        start=start,
        end=end,
        horizons=["1w", "1m"],
        n_paths=200,
        step_days=20,
        recal_every_days=21,
        history_window_years=2,
        equilibrium=eq,
        use_eq_overlay=True,
        seed=42,
    )
    elapsed = time.perf_counter() - t0
    n_forecasts = int(len(bt.forecasts)) if bt.forecasts is not None else 0
    crps = float(bt.overall.get("crps", float("nan")))
    ev = [
        {"kind": "output", "label": "elapsed s", "value": round(elapsed, 2)},
        {"kind": "output", "label": "n forecasts evaluated", "value": n_forecasts},
        {"kind": "output", "label": "pooled CRPS", "value": crps},
        {"kind": "output", "label": "overall keys", "value": list(bt.overall.keys())[:12]},
    ]
    if n_forecasts == 0 or not np.isfinite(crps):
        return "fail", f"n={n_forecasts} crps={crps}", ev
    return "pass", f"{n_forecasts} forecasts in {elapsed:.1f}s, CRPS={crps:.5f}", ev


def _check_anthropic_ping() -> tuple[str, str, list[dict]]:
    import urllib.request
    import urllib.error
    import json as _json

    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC") or ""
    if not key:
        return "skip", "no ANTHROPIC key set", []

    body = _json.dumps({
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5"),
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with exactly: DIAGNOSTIC_OK"}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = _json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")[:400]
        return "fail", f"HTTP {e.code}: {text}", [
            {"kind": "error", "label": "HTTPError", "value": f"{e.code} {text}"},
        ]
    rt = int((time.perf_counter() - t0) * 1000)
    parts = data.get("content", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    ev = [
        {"kind": "output", "label": "round-trip ms", "value": rt},
        {"kind": "output", "label": "response", "value": text[:200]},
        {"kind": "output", "label": "model echoed", "value": data.get("model")},
    ]
    if "DIAGNOSTIC_OK" not in text:
        return "fail", f"unexpected response: {text[:80]}", ev
    return "pass", f"{rt}ms · response=DIAGNOSTIC_OK", ev


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_diagnostic() -> dict:
    """Run the full self-check. Pure function — returns a JSON-serialisable dict."""
    from engine.pricers import PRICER_ORDER

    r = _Reporter()

    # System group
    r.run("Environment: EODHD present", "system",
          lambda: _check_env_required("EODHD"))
    r.run("Environment: ANTHROPIC present", "system",
          lambda: _check_env_optional("ANTHROPIC"))
    r.run("Environment: POLYGON present", "system",
          lambda: _check_env_optional("POLYGON"))
    r.run("Storage: cache directory writable", "system", _check_cache_dir)
    r.run("Data: USD/HKD history fetch (EODHD)", "system", _check_fx_history)
    r.run("Data: US/HK rate differential (FRED + LERS synthetic)", "system",
          _check_rate_differential)
    r.run("Model: disequilibrium equilibrium fit", "system", _check_equilibrium)

    # Functional group — 13 pricers
    for model_name in PRICER_ORDER:
        r.run(f"Pricer: {model_name} calibrate + simulate", "functional",
              lambda m=model_name: _check_pricer(m))

    r.run("Backtest: smoke (bs_rv, 1y, 200 paths)", "functional",
          _check_backtest_smoke)

    r.run("AI: Anthropic Claude live ping", "functional",
          _check_anthropic_ping,
          skip_if=lambda: (
              not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC")),
              "no ANTHROPIC key set",
          ))

    totals = {"pass": 0, "fail": 0, "skip": 0}
    for c in r.checks:
        totals[c["status"]] = totals.get(c["status"], 0) + 1

    return {
        "ok": totals["fail"] == 0,
        "runAt": _now_iso(),
        "totals": totals,
        "checks": r.checks,
    }

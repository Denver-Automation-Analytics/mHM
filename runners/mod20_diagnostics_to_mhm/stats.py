"""Timeseries and field statistics for mod20 diagnostics.

Complements metrics.py (water-balance verdicts) with descriptive statistics for
the hydrograph (per-gauge sim-vs-obs skill), terrain (elevation/slope/aspect),
and precipitation fields. All functions return plain JSON-serialisable dicts.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _paired(sim: np.ndarray, obs: np.ndarray):
    """Return (sim, obs) restricted to timesteps where both are finite."""
    m = np.isfinite(sim) & np.isfinite(obs)
    return sim[m], obs[m]


def _kge(sim: np.ndarray, obs: np.ndarray):
    """Kling-Gupta efficiency and its r/alpha/beta components."""
    if sim.size < 2 or obs.std() == 0:
        return None, None, None, None
    r = float(np.corrcoef(sim, obs)[0, 1])
    alpha = float(sim.std() / obs.std())
    beta = float(sim.mean() / obs.mean()) if obs.mean() != 0 else np.nan
    kge = 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return kge, r, alpha, beta


def _nse(sim: np.ndarray, obs: np.ndarray):
    denom = float(((obs - obs.mean()) ** 2).sum())
    if denom == 0:
        return None
    return 1.0 - float(((obs - sim) ** 2).sum()) / denom


def _lognse(sim: np.ndarray, obs: np.ndarray, eps: float = 1e-3):
    ls, lo = np.log(np.clip(sim, eps, None)), np.log(np.clip(obs, eps, None))
    return _nse(ls, lo)


def discharge_stats(disch: Dict) -> List[Dict]:
    """Per-gauge sim-vs-obs skill (KGE, NSE, logNSE, PBIAS, r, RMSE, means)."""
    rows: List[Dict] = []
    for g in disch["gauges"]:
        sim, obs = _paired(g["qsim"], g["qobs"])
        if sim.size == 0:
            rows.append({"site_no": g["site_no"], "name": g["name"],
                         "n_obs": 0, "note": "no overlapping observations"})
            continue
        kge, r, alpha, beta = _kge(sim, obs)
        pbias = float(100.0 * (sim - obs).sum() / obs.sum()) if obs.sum() != 0 else None
        rmse = float(np.sqrt(((sim - obs) ** 2).mean()))
        rows.append({
            "site_no": g["site_no"],
            "name": g["name"],
            "n_obs": int(sim.size),
            "kge": kge, "kge_r": r, "kge_alpha": alpha, "kge_beta": beta,
            "nse": _nse(sim, obs),
            "lognse": _lognse(sim, obs),
            "pbias_pct": pbias,
            "rmse_m3s": rmse,
            "mean_sim_m3s": float(sim.mean()),
            "mean_obs_m3s": float(obs.mean()),
        })
    return rows


def _field_summary(field: np.ndarray) -> Dict:
    v = field[np.isfinite(field)]
    if v.size == 0:
        return {"n_cells": 0}
    return {
        "n_cells": int(v.size),
        "min": float(v.min()), "max": float(v.max()),
        "mean": float(v.mean()), "std": float(v.std()),
        "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
    }


def terrain_stats(terr: Dict) -> Dict:
    """Descriptive statistics of the domain terrain fields."""
    out: Dict = {}
    for var, fld in terr["fields"].items():
        s = _field_summary(fld)
        if var == "dem":
            s["range_m"] = (s.get("max", 0) - s.get("min", 0)) if s["n_cells"] else None
        out[var] = s
    return out


def precip_stats(inp: Dict, window) -> Dict:
    """Precipitation statistics over the diagnostics window.

    Reports the domain-mean annual depth, the spatial spread of per-cell annual
    totals, and simple wet-day metrics from the domain-mean daily series.
    """
    time = inp["time"]
    years = max((time[-1] - time[0]).days / 365.25, 1e-9)
    daily = pd.Series(inp["series"]["pre"], index=time)

    cell_total = inp["fields"]["pre"]          # per-cell sum over window [mm]
    cell_annual = cell_total / years
    spatial = _field_summary(cell_annual)

    wet = daily[daily >= 1.0]
    return {
        "window": [window[0], window[1]],
        "years": float(years),
        "annual_domain_mm": float(daily.sum() / years),
        "annual_per_cell_mm": {k: spatial[k] for k in spatial},
        "max_daily_domain_mm": float(daily.max()),
        "wet_day_fraction": float((daily >= 1.0).mean()),
        "mean_wet_day_mm": float(wet.mean()) if len(wet) else 0.0,
    }


def parameter_stats(params: List[Dict]) -> Dict:
    """Summarise calibrated-parameter rail-pinning."""
    free = [p for p in params if not p["fixed"]]
    railed = [p["name"] for p in params if p["railed"]]
    return {
        "n_total": len(params),
        "n_free": len(free),
        "n_fixed": len(params) - len(free),
        "n_railed": len(railed),
        "railed": railed,
        "parameters": params,
    }


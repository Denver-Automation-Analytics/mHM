"""Readers for mod22 TRITON performance diagnostics (--perf).

TRITON writes ``performance.txt`` (final per-rank summary) and, in
``performance/``, one ``performanceN.txt`` per spatial-output print step N
containing the same columns as *cumulative* wall-clock seconds up to that
step. Diffing consecutive steps gives the incremental time spent per output
interval, which is what surfaces load imbalance or solver slowdowns.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_STEP_RE = re.compile(r"^performance(\d+)\.txt$")
DELTA_COLS = ("Compute", "MPI", "IO", "Resize", "Other")


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip().lstrip("%") for c in df.columns]
    return df


def read_summary(path: Path) -> pd.DataFrame:
    """Return the final performance.txt table, one row per rank + Average."""
    return _read_csv(path)


def read_series(perf_dir: Path) -> pd.DataFrame:
    """Return cumulative per-step, per-rank timings from performance/*.txt."""
    rows = []
    for p in Path(perf_dir).glob("performance*.txt"):
        m = _STEP_RE.match(p.name)
        if not m:
            continue
        df = _read_csv(p)
        df = df[df["Rank"] != "Average"].copy()
        df["Rank"] = df["Rank"].astype(int)
        df["step"] = int(m.group(1))
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No performanceN.txt files found in {perf_dir}")
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["Rank", "step"])
        .reset_index(drop=True)
    )


def step_deltas(series: pd.DataFrame, cols=DELTA_COLS) -> pd.DataFrame:
    """Per-step incremental wall time per rank (diff of the cumulative series)."""
    out = series.sort_values(["Rank", "step"]).copy()
    for c in cols:
        out[f"d_{c}"] = out.groupby("Rank")[c].diff()
    return out


def wet_stats(nc_path: Path, var: str, nodata: float) -> pd.DataFrame:
    """Per-step wet-cell count and depth-sum volume proxy from a mod22 netCDF cube."""
    import netCDF4

    ds = netCDF4.Dataset(nc_path, "r")
    try:
        v = ds.variables[var]
        v.set_auto_maskandscale(False)
        tv = ds.variables["time"]
        times = netCDF4.num2date(tv[:], tv.units)
        n = v.shape[0]
        n_wet = np.empty(n, dtype=np.int64)
        volume = np.empty(n, dtype=np.float64)
        for i in range(n):
            arr = np.asarray(v[i], dtype=np.float32)
            valid = arr[arr != np.float32(nodata)]
            n_wet[i] = valid.size
            volume[i] = float(valid.sum())
    finally:
        ds.close()
    return pd.DataFrame(
        {"time": times, "step": np.arange(1, n + 1), "n_wet": n_wet, "volume": volume}
    )


def read_roff(roff_path: Path, l1_cellsize_m: float) -> pd.DataFrame:
    """Domain applied-runoff time series from TRITON's ``.roff`` input.

    Each row is ``time_hr,val_zone1,val_zone2,...`` in mm/hr, one equal-area L1
    zone per column (mod21 ``writers.write_roff``); returns the domain-mean
    intensity and domain-total volume rate actually driving the simulation.
    """
    times, means = [], []
    n_zones = 0
    with open(roff_path) as f:
        next(f)  # header comment line
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            vals = np.asarray(parts[1:], dtype=float)
            times.append(float(parts[0]))
            means.append(float(vals.mean()))
            n_zones = vals.size
    mean_mm_hr = np.asarray(means)
    total_m3_hr = mean_mm_hr * 1e-3 * n_zones * (l1_cellsize_m**2)
    return pd.DataFrame(
        {
            "step": np.arange(1, len(times) + 1),
            "time_hr": times,
            "mean_mm_hr": mean_mm_hr,
            "total_m3_hr": total_m3_hr,
        }
    )

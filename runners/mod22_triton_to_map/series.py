"""Stage-hydrograph plots for mod22 (TRITON observation-point time series).

TRITON writes one ``<name>_at_Xsec.txt`` per run under ``series/`` when
``time_series_flag=1``: a CSV whose first column is the simulation time in
seconds and whose remaining columns are the water depth (stage) sampled at each
observation point (``H_at_Point_<n>``, in the order of the ``.obs`` file). This
module turns each such file into a stage-vs-time line plot so the hydrograph at
the monitoring points can be read directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def discover_series(series_dir: Path) -> List[Path]:
    """Return the TRITON stage time-series ``.txt`` files in *series_dir*."""
    return sorted(p for p in Path(series_dir).glob("*.txt") if p.is_file())


def read_series(path: Path, start_date: str) -> pd.DataFrame:
    """Read a TRITON series file into a DataFrame with a datetime index.

    The ``Time(s)`` column (seconds since the simulation start) is converted to
    wall-clock time anchored at *start_date*; the remaining columns (one per
    observation point) are coerced to float so blow-up tokens (``-nan``) become
    NaN rather than object strings.
    """
    df = pd.read_csv(path)
    tcol = df.columns[0]
    df["time"] = pd.Timestamp(start_date) + pd.to_timedelta(df[tcol], unit="s")
    df = df.drop(columns=[tcol]).set_index("time")
    return df.apply(pd.to_numeric, errors="coerce")


def plot_stage_hydrographs(path: Path, out_png: Path, start_date: str) -> int:
    """Plot every observation point's stage over time from *path*. Returns #points.

    The x-axis is pinned to the full simulation window (the file's own time span)
    so a series that goes NaN mid-run reads as a line that stops inside the
    window rather than a plot that merely ends early; the first NaN after valid
    data is flagged per point and the diverged tail is shaded.
    """
    df = read_series(path, start_date)
    cols = list(df.columns)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    onsets = []
    for col in cols:
        s = df[col]
        line, = ax.plot(s.index, s.to_numpy(), lw=1.2, label=col)
        finite = s.notna().to_numpy()
        if finite.any() and not finite.all():
            last = np.where(finite)[0].max()
            nan_after = np.where(~finite)[0]
            nan_after = nan_after[nan_after > last]
            if nan_after.size:
                onsets.append(s.index[nan_after.min()])
                ax.plot(s.index[last], s.iloc[last], "x", color=line.get_color(), ms=8, mew=2)

    if onsets:
        first = min(onsets)
        ax.axvspan(first, df.index.max(), color="red", alpha=0.08)
        ax.axvline(first, color="red", ls="--", lw=1)
        ax.annotate("solution diverged (NaN)", xy=(first, ax.get_ylim()[1]),
                    xytext=(4, -10), textcoords="offset points", color="red",
                    fontsize=8, va="top")

    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_ylabel("stage / water depth [m]")
    ax.set_xlabel("time")
    ax.set_title(f"TRITON stage hydrograph at monitoring points ({path.stem})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=min(len(cols), 4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return len(cols)

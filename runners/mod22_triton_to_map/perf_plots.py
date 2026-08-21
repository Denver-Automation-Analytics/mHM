"""Diagnostic figures for mod22 TRITON performance data (--perf).

plot_load_balance : per-rank Compute/MPI/IO/Resize/Other breakdown (table + bar)
plot_timeseries   : per-step incremental compute/MPI-wait time, optionally next
                    to the domain wet-cell/volume series to spot correlation
                    between solver slowdowns and physical instability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perf import DELTA_COLS

_COLORS = ["#4c72b0", "#c44e52", "#55a868", "#8172b3", "#ccb974"]


def plot_load_balance(summary: pd.DataFrame, out_png: Path) -> None:
    """Stacked bar + table of the final per-rank Compute/MPI/IO/Resize/Other split."""
    ranks = summary[summary["Rank"] != "Average"].copy()
    ranks["Rank"] = ranks["Rank"].astype(int)
    ranks = ranks.sort_values("Rank")

    fig, (ax_bar, ax_tbl) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})
    bottom = np.zeros(len(ranks))
    for col, color in zip(DELTA_COLS, _COLORS):
        vals = ranks[col].to_numpy(dtype=float)
        ax_bar.bar(ranks["Rank"].astype(str), vals, bottom=bottom, label=col, color=color)
        bottom += vals
    ax_bar.set_xlabel("rank")
    ax_bar.set_ylabel("wall time [s]")
    ax_bar.set_title("Final time breakdown per rank")
    ax_bar.legend(fontsize=8)

    ax_tbl.axis("off")
    cols = ["Rank", *DELTA_COLS, "Total"]
    cell_text = ranks[cols].round(1).astype(str).to_numpy()
    tbl = ax_tbl.table(cellText=cell_text, colLabels=cols, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    ax_tbl.set_title("Performance summary [s]")

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_timeseries(deltas: pd.DataFrame, wet: Optional[pd.DataFrame], out_png: Path,
                    roff: Optional[pd.DataFrame] = None, start_date: str = None,
                    interval_s: int = 1800) -> None:
    """Per-step compute/MPI-wait time per rank, next to wet-cell/volume and applied-runoff series.

    All panels share a datetime x-axis anchored at *start_date*: the timing and
    wet-extent panels follow the TRITON output cadence (*interval_s*, 1-based
    step ``k`` -> ``start_date + k*interval_s``), while the runoff panel uses its
    own hourly forcing timestamps (``time_hr``).
    """
    base = pd.Timestamp(start_date)
    n_axes = 1 + (wet is not None) + (roff is not None)
    fig, axes = plt.subplots(n_axes, 1, figsize=(11, 3.5 * n_axes), sharex=True, squeeze=False)
    ax1 = axes[0, 0]
    for rank, g in deltas.groupby("Rank"):
        g = g.sort_values("step")
        t = base + pd.to_timedelta(g["step"] * interval_s, unit="s")
        ax1.plot(t, g["d_Compute"], lw=1, label=f"rank {rank} compute")
        ax1.plot(t, g["d_MPI"], lw=1, ls="--", label=f"rank {rank} MPI wait")
    ax1.set_ylabel("time per output step [s]")
    ax1.set_title("Per-step compute / MPI-wait time (load imbalance & solver slowdowns)")
    ax1.legend(fontsize=7, ncol=4)

    row = 1
    if wet is not None:
        ax2 = axes[row, 0]
        ax2b = ax2.twinx()
        t = base + pd.to_timedelta(wet["step"] * interval_s, unit="s")
        ax2.plot(t, wet["n_wet"], color="#4c72b0")
        ax2b.plot(t, wet["volume"], color="#c44e52")
        ax2.set_ylabel("wet cell count", color="#4c72b0")
        ax2b.set_ylabel("depth-sum proxy [m]", color="#c44e52")
        ax2.set_title("Domain wet extent / volume proxy over the run")
        row += 1

    if roff is not None:
        ax3 = axes[row, 0]
        ax3b = ax3.twinx()
        t = base + pd.to_timedelta(roff["time_hr"], unit="h")
        ax3.plot(t, roff["mean_mm_hr"], color="#55a868")
        ax3b.plot(t, roff["total_m3_hr"], color="#8172b3")
        ax3.set_ylabel("domain-mean runoff [mm/hr]", color="#55a868")
        ax3b.set_ylabel("domain-total runoff [m3/hr]", color="#8172b3")
        ax3.set_title("Excess runoff driving TRITON (input forcing)")

    axes[-1, 0].set_xlabel("time")
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

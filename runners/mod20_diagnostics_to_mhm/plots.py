"""Diagnostic plots for mod20: Budyko, monthly fluxes, spatial maps, partition."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import Metric
from readers import monthly_sum


def _imshow(ax, field: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, cmap: str,
            vmin=None, vmax=None):
    extent = [x.min(), x.max(), y.min(), y.max()]
    # Robust color limits (2nd–98th pct) unless explicit bounds are given,
    # so a few outlier cells cannot flatten the whole map.
    if vmin is None or vmax is None:
        finite = field[np.isfinite(field)]
        if finite.size:
            lo, hi = np.nanpercentile(finite, [2, 98])
            vmin = lo if vmin is None else vmin
            vmax = (hi if hi > lo else lo + 1e-9) if vmax is None else vmax
    im = ax.imshow(field, extent=extent, origin="upper", cmap=cmap, aspect="equal",
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_budyko(flux: Dict, inp: Dict, out_png: Path) -> None:
    """Budyko diagram: point (PET/P, aET/P) against energy/water limits."""
    P = inp["totals"]["pre"]
    pet = inp["totals"]["pet"]
    aet = flux["totals"]["aET"]
    ai = pet / P if P else np.nan
    ei = aet / P if P else np.nan

    fig, ax = plt.subplots(figsize=(5, 5))
    xs = np.linspace(0, max(3.0, ai * 1.2 if np.isfinite(ai) else 3.0), 200)
    ax.plot(xs, np.minimum(xs, 1.0), "k--", lw=1, label="energy / water limits")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    budyko = np.sqrt(xs * np.tanh(1.0 / np.where(xs == 0, np.nan, xs)) * (1 - np.exp(-xs)))
    ax.plot(xs, budyko, "b-", lw=1, label="Budyko curve")
    ax.plot(ai, ei, "ro", ms=9, label=f"basin ({ai:.2f}, {ei:.2f})")
    ax.set_xlabel("Aridity index  PET / P")
    ax.set_ylabel("Evaporative index  aET / P")
    ax.set_title("Budyko diagnostic")
    ax.set_xlim(0, xs.max())
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_monthly(flux: Dict, inp: Dict, out_png: Path) -> None:
    """Monthly domain-mean precipitation, actual ET, and runoff."""
    p = monthly_sum(inp["time"], inp["series"]["pre"])
    aet = monthly_sum(flux["time"], flux["series"]["aET"])
    q = monthly_sum(flux["time"], flux["series"]["Q"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(p.index, p.values, width=20, color="#4c72b0", alpha=0.6, label="P (input)")
    ax.plot(aet.index, aet.values, color="#c44e52", lw=1.5, label="aET")
    ax.plot(q.index, q.values, color="#55a868", lw=1.5, label="Q (total runoff)")
    ax.set_ylabel("mm / month")
    ax.set_title("Monthly water-balance fluxes (domain mean)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_maps(flux: Dict, out_png: Path) -> None:
    """Per-cell totals: aET, recharge, runoff coefficient, baseflow fraction."""
    x, y = flux["easting"], flux["northing"]
    aet = flux["fields"]["aET"]
    rech = flux["fields"]["recharge"]
    q = flux["fields"]["Q"]
    qb = flux["fields"]["QB"]
    preff = flux["fields"]["preEffect"]

    with np.errstate(invalid="ignore", divide="ignore"):
        # Ignore cells with negligible effective precip to avoid Q/preEffect blow-ups.
        rc = np.where(preff > 1.0, q / preff, np.nan)
        rc = np.clip(rc, 0.0, 1.0)
        bfi = np.where(q > 0, qb / q, np.nan)
        bfi = np.clip(bfi, 0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    _imshow(axes[0, 0], aet, x, y, "Actual ET total [mm]", "YlGnBu")
    _imshow(axes[0, 1], rech, x, y, "Recharge (L1_percol) total [mm]", "PuBuGn")
    _imshow(axes[1, 0], rc, x, y, "Runoff coeff  Q/preEffect [-]", "viridis",
            vmin=0.0, vmax=1.0)
    _imshow(axes[1, 1], bfi, x, y, "Baseflow fraction  QB/Q [-]", "magma",
            vmin=0.0, vmax=1.0)
    fig.suptitle("Per-cell diagnostic maps (evaluation window)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_partition(flux: Dict, inp: Dict, out_png: Path) -> None:
    """Stacked partition of gross precipitation into aET, Q, recharge and ΔS."""
    P = inp["totals"]["pre"]
    aet = flux["totals"]["aET"]
    q = flux["totals"]["Q"]
    dS = flux["delta_storage"] or 0.0
    resid = P - aet - q - dS

    labels = ["aET", "Q", "ΔS", "residual"]
    vals = [aet, q, dS, resid]
    colors = ["#c44e52", "#55a868", "#8172b3", "#cccccc"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = 0.0
    for lab, val, col in zip(labels, vals, colors):
        ax.bar("P partition", val, bottom=bottom, color=col, label=f"{lab} ({val:.0f})")
        bottom += val
    ax.axhline(P, color="k", ls="--", lw=1, label=f"P gross ({P:.0f})")
    ax.set_ylabel("mm over evaluation window")
    ax.set_title("Precipitation partitioning")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def write_all(flux: Dict, inp: Dict, metrics: List[Metric], out_dir: Path) -> List[Path]:
    """Render every diagnostic plot into out_dir and return the file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in (
        ("budyko.png", lambda p: plot_budyko(flux, inp, p)),
        ("monthly_fluxes.png", lambda p: plot_monthly(flux, inp, p)),
        ("spatial_maps.png", lambda p: plot_maps(flux, p)),
        ("precip_partition.png", lambda p: plot_partition(flux, inp, p)),
    ):
        p = out_dir / name
        fn(p)
        paths.append(p)
    return paths

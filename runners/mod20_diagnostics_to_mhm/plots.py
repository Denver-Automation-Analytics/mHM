"""Diagnostic plots for mod20: Budyko, monthly fluxes, spatial maps, partition."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import Metric
from readers import monthly_sum, RAIL_TOL


def _imshow(
    ax,
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    cmap: str,
    vmin=None,
    vmax=None,
):
    extent = [x.min(), x.max(), y.min(), y.max()]
    # Match row order to the y-axis direction so maps are not vertically flipped.
    origin = "lower" if y[0] < y[-1] else "upper"
    # Robust color limits (2nd–98th pct) unless explicit bounds are given,
    # so a few outlier cells cannot flatten the whole map.
    if vmin is None or vmax is None:
        finite = field[np.isfinite(field)]
        if finite.size:
            lo, hi = np.nanpercentile(finite, [2, 98])
            vmin = lo if vmin is None else vmin
            vmax = (hi if hi > lo else lo + 1e-9) if vmax is None else vmax
    im = ax.imshow(
        field,
        extent=extent,
        origin=origin,
        cmap=cmap,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
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
    budyko = np.sqrt(
        xs * np.tanh(1.0 / np.where(xs == 0, np.nan, xs)) * (1 - np.exp(-xs))
    )
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
    _imshow(
        axes[1, 0],
        rc,
        x,
        y,
        "Runoff coeff  Q/preEffect [-]",
        "viridis",
        vmin=0.0,
        vmax=1.0,
    )
    _imshow(
        axes[1, 1],
        bfi,
        x,
        y,
        "Baseflow fraction  QB/Q [-]",
        "magma",
        vmin=0.0,
        vmax=1.0,
    )
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


def plot_hydrographs(disch: Dict, dstats: List[Dict], inp: Dict, out_png: Path) -> None:
    """Simulated vs. observed discharge per gauge, with the domain-total daily
    precipitation volume drawn as inverted bars (hyetograph) on a secondary axis."""
    gauges = disch["gauges"]
    stat_by_site = {s["site_no"]: s for s in (dstats or [])}
    # Domain-total daily precip volume = areal-mean depth x domain area (Mm3/day).
    dx = abs(float(inp["x"][1] - inp["x"][0]))
    dy = abs(float(inp["y"][1] - inp["y"][0]))
    area_m2 = int(np.asarray(inp["mask"]).sum()) * dx * dy
    p_t = inp["time"]
    p_v = np.asarray(inp["series"]["pre"], float) * 1e-3 * area_m2 / 1e6
    p_max = float(np.nanmax(p_v)) if len(p_v) else 1.0
    n = len(gauges)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.8 * n), sharex=True, squeeze=False)
    for ax, g in zip(axes[:, 0], gauges):
        # Precip volume as bars hanging from the top on an inverted twin axis.
        axp = ax.twinx()
        axp.bar(p_t, p_v, width=1.0, color="#6baed6", alpha=0.6, linewidth=0, zorder=1)
        axp.set_ylim(
            p_max * 2.6, 0.0
        )  # invert: 0 at top so bars occupy the upper third
        axp.set_ylabel("domain precip [Mm³/day]", fontsize=8, color="#3d6d99")
        axp.tick_params(labelsize=7, colors="#3d6d99")
        # Draw discharge lines above the bars (transparent primary axis on top).
        ax.set_zorder(axp.get_zorder() + 1)
        ax.patch.set_visible(False)
        t = g["time"]
        ax.plot(t, g["qobs"], color="k", lw=1.0, label="observed", zorder=3)
        ax.plot(
            t,
            g["qsim"],
            color="#c44e52",
            lw=1.0,
            alpha=0.9,
            label="simulated",
            zorder=3,
        )
        st = stat_by_site.get(g["site_no"], {})
        skill = ""
        if st.get("kge") is not None:
            skill = f"  KGE={st['kge']:.2f}  NSE={st.get('nse', float('nan')):.2f}  PBIAS={st.get('pbias_pct', float('nan')):.0f}%"
        ax.set_title(f"{g['site_no']} — {g['name']}{skill}", fontsize=9)
        ax.set_ylabel("Q [m³/s]", fontsize=8)
        ax.set_ylim(bottom=0.0)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    axes[-1, 0].set_xlabel("date")
    fig.suptitle("Hydrographs with domain-total precipitation forcing", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_flow_duration(disch: Dict, out_png: Path) -> None:
    """Flow-duration curves (log-y) of sim vs. obs, one panel per gauge."""
    gauges = disch["gauges"]
    n = len(gauges)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, g in zip(axes.flat, gauges):
        ax.set_visible(True)
        for arr, col, lab in ((g["qobs"], "k", "obs"), (g["qsim"], "#4c72b0", "sim")):
            v = np.sort(arr[np.isfinite(arr)])[::-1]
            if v.size == 0:
                continue
            exc = np.arange(1, v.size + 1) / (v.size + 1) * 100.0
            ax.plot(exc, np.clip(v, 1e-3, None), color=col, lw=1.2, label=lab)
        ax.set_yscale("log")
        ax.set_title(f"{g['site_no']}", fontsize=9)
        ax.set_xlabel("exceedance [%]", fontsize=8)
        ax.set_ylabel("Q [m³/s]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    fig.suptitle("Flow-duration curves", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_terrain(terr: Dict, tstats: Dict, out_png: Path) -> None:
    """Terrain maps (DEM, slope, aspect) plus an elevation histogram."""
    x, y = terr["x"], terr["y"]
    fields = terr["fields"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    if "dem" in fields:
        _imshow(axes[0, 0], fields["dem"], x, y, "Elevation [m]", "terrain")
    if "slope" in fields:
        _imshow(axes[0, 1], fields["slope"], x, y, "Slope [deg]", "YlOrBr")
    if "aspect" in fields:
        _imshow(
            axes[1, 0],
            fields["aspect"],
            x,
            y,
            "Aspect [deg]",
            "twilight",
            vmin=0.0,
            vmax=360.0,
        )
    ax = axes[1, 1]
    if "dem" in fields:
        v = fields["dem"][np.isfinite(fields["dem"])]
        ax.hist(v, bins=40, color="#8c7b6b")
        s = tstats.get("dem", {})
        ax.set_title(
            f"Elevation dist.  mean={s.get('mean', float('nan')):.0f} m  "
            f"range={s.get('range_m', float('nan')):.0f} m",
            fontsize=9,
        )
        ax.set_xlabel("elevation [m]", fontsize=8)
        ax.set_ylabel("cells", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Terrain (L0 morphology)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_precip_maps(inp: Dict, pstats: Dict, out_png: Path) -> None:
    """Mean-annual precipitation map and the per-cell annual-total distribution."""
    x, y = inp["x"], inp["y"]
    years = max(pstats.get("years", 1.0), 1e-9)
    cell_annual = inp["fields"]["pre"] / years  # mm/yr per cell

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _imshow(axes[0], cell_annual, x, y, "Mean annual precipitation [mm/yr]", "YlGnBu")
    v = cell_annual[np.isfinite(cell_annual)]
    axes[1].hist(v, bins=40, color="#4c72b0")
    axes[1].axvline(
        pstats.get("annual_domain_mm", np.nan),
        color="k",
        ls="--",
        lw=1,
        label=f"domain mean {pstats.get('annual_domain_mm', float('nan')):.0f} mm/yr",
    )
    axes[1].set_title("Per-cell annual precipitation", fontsize=9)
    axes[1].set_xlabel("mm/yr", fontsize=8)
    axes[1].set_ylabel("cells", fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[1].legend(fontsize=8)
    fig.suptitle("Precipitation input", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_parameters(params: List[Dict], out_png: Path) -> None:
    """Bullet chart: each calibrated parameter's final value within its bounds.

    The value is plotted at its normalised position in [lower, upper]; markers in
    the shaded rail zones (red) indicate a bound was effectively reached.
    """
    n = len(params)
    ys = np.arange(n)[::-1]  # keep namelist order, top-down
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.26 * n)))
    ax.axvspan(0.0, RAIL_TOL, color="#f2b6b6", alpha=0.6, zorder=0)
    ax.axvspan(1.0 - RAIL_TOL, 1.0, color="#f2b6b6", alpha=0.6, zorder=0)
    for y, p in zip(ys, params):
        ax.hlines(y, 0.0, 1.0, color="#dddddd", lw=2.0, zorder=1)
        if p["fixed"]:
            ax.plot(0.5, y, "s", color="#999999", ms=5, zorder=3)
        else:
            col = "#d62728" if p["railed"] else "#2c7fb8"
            ax.plot(p["pos"], y, "o", color=col, ms=6, zorder=3)
        ax.text(1.035, y, f"{p['value']:.3g}", fontsize=5, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([p["name"] for p in params], fontsize=6)
    ax.set_ylim(-1, n)
    ax.set_xlim(-0.02, 1.12)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["lower\nbound", "mid", "upper\nbound"], fontsize=8)
    n_rail = sum(p["railed"] for p in params)
    n_free = sum(not p["fixed"] for p in params)
    ax.set_title(
        f"DDS calibrated parameters vs. bounds — "
        f"{n_rail}/{n_free} rail-pinned (red), grey = fixed",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def write_all(
    flux: Dict,
    inp: Dict,
    metrics: List[Metric],
    out_dir: Path,
    disch: Dict = None,
    dstats: List[Dict] = None,
    terr: Dict = None,
    tstats: Dict = None,
    pstats: Dict = None,
    params: List[Dict] = None,
) -> List[Path]:
    """Render every diagnostic plot into out_dir and return the file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    renderers = [
        ("budyko.png", lambda p: plot_budyko(flux, inp, p)),
        ("monthly_fluxes.png", lambda p: plot_monthly(flux, inp, p)),
        ("spatial_maps.png", lambda p: plot_maps(flux, p)),
        ("precip_partition.png", lambda p: plot_partition(flux, inp, p)),
    ]
    if disch is not None:
        renderers.append(
            ("hydrographs.png", lambda p: plot_hydrographs(disch, dstats, inp, p))
        )
        renderers.append(("flow_duration.png", lambda p: plot_flow_duration(disch, p)))
    if terr is not None:
        renderers.append(
            ("terrain_maps.png", lambda p: plot_terrain(terr, tstats or {}, p))
        )
    if pstats is not None:
        renderers.append(
            ("precip_maps.png", lambda p: plot_precip_maps(inp, pstats, p))
        )
    if params is not None:
        renderers.append(("parameter_rails.png", lambda p: plot_parameters(params, p)))

    paths = []
    for name, fn in renderers:
        p = out_dir / name
        fn(p)
        paths.append(p)
    return paths

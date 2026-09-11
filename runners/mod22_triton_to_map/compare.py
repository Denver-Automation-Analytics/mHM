"""Stage-vs-gauge comparison for mod22 (TRITON --compare).

For each user-specified lat/lon point, samples the TRITON water-depth hydrograph
from the consolidated ``H.nc`` cube (nearest grid cell) and overlays the observed
USGS gauge. The observed gauge height (param 00065, feet) is converted into water
depth *at the point* so the two curves are directly comparable::

    depth[m] = (gauge_altitude_ft + gauge_stage_ft) * 0.3048 - bed_elevation[m]

where ``bed_elevation`` is sampled from the mod21 DEM at the same lat/lon and the
user-supplied altitude accuracy sets a shaded uncertainty band on the observed
depth. Time axes are aligned explicitly: naive ``H.nc`` timestamps are localized
to ``TRITON_COMPARE_TZ`` and converted to UTC, matching the tz-aware UTC index of
the USGS series, so both are plotted on a common UTC axis.

A lat/lon is used rather than the gauge's own coordinate because the published
gauge location is not precise enough to pick the intended channel cell.
"""

from __future__ import annotations

import os
import sys
import io
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import pyproj
import requests
from osgeo import gdal

gdal.UseExceptions()

log = logging.getLogger("triton_to_map")

_FT_TO_M = 0.3048
_NWIS_IV = "https://waterservices.usgs.gov/nwis/iv/"
# USGS RDB tz_cd abbreviation -> UTC offset [h]; used to convert each observation
# from its reported local zone to UTC (DST already encoded in the abbreviation).
_TZ_OFFSET_H = {
    "UTC": 0,
    "GMT": 0,
    "EST": -5,
    "EDT": -4,
    "CST": -6,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
    "AKST": -9,
    "AKDT": -8,
    "HST": -10,
}

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    END_DATE,
    L1_CELL_SIZE_M,
    NODATA,
    TRITON_COMPARE_OUT_DIR,
    TRITON_COMPARE_POINTS,
    TRITON_COMPARE_TZ,
    TRITON_GIF_CMAP,
    TRITON_GIF_FPS,
    TRITON_GIF_MAX_FRAMES,
    TRITON_GIF_VARS,
    TRITON_MAP_CFG,
    TRITON_MAP_CLIP,
    TRITON_MAP_DEM_TIF,
    TRITON_MAP_GTIFF_DIR,
    TRITON_MAP_HMIN,
    TRITON_MAP_MIN_DEPTH,
    TRITON_MAP_OUT_DIR,
    TRITON_MAP_SERIES_DIR,
    TRITON_PERF_DIR,
    TRITON_PERF_ROFF,
    TRITON_PERF_SUMMARY,
    TRITON_PERF_WET_VAR,
    TRITON_START_DATE,
    TRITON_START_FILE,
)

def _cube_geometry(ds: netCDF4.Dataset) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return the cube's x, y cell-centre axes and its CRS (EPSG string or WKT)."""
    x = np.asarray(ds.variables["x"][:], dtype=np.float64)
    y = np.asarray(ds.variables["y"][:], dtype=np.float64)
    crs = ds.variables.get("crs")
    ref = None
    if crs is not None:
        ref = (
            getattr(crs, "epsg_code", None)
            or getattr(crs, "crs_wkt", None)
            or getattr(crs, "spatial_ref", None)
        )
    if not ref:
        raise ValueError(
            "H.nc carries no CRS (crs variable missing epsg_code/crs_wkt)."
        )
    return x, y, ref


def _nearest_index(axis: np.ndarray, value: float) -> int:
    """Index of the cell-centre in *axis* nearest *value* (axis may be descending)."""
    return int(np.abs(axis - value).argmin())


def sample_h_series(
    nc_path: Path, lat: float, lon: float, tz: str, nodata: float
) -> Tuple[pd.Series, Tuple[int, int]]:
    """Sample the TRITON depth hydrograph from ``H.nc`` at *lat*/*lon* (nearest cell).

    Returns a depth [m] series on a UTC index (naive cube times localized to *tz*
    then converted to UTC) and the sampled ``(row, col)``. NODATA cells are read as
    0 m so the hydrograph stays continuous when the cell is dry.
    """
    ds = netCDF4.Dataset(nc_path, "r")
    try:
        x, y, ref = _cube_geometry(ds)
        tf = pyproj.Transformer.from_crs("EPSG:4326", ref, always_xy=True)
        px, py = tf.transform(lon, lat)
        col = _nearest_index(x, px)
        row = _nearest_index(y, py)

        var = ds.variables["H"]
        var.set_auto_maskandscale(False)
        depth = np.asarray(var[:, row, col], dtype=np.float64)
        depth = np.where(depth == np.float64(nodata), 0.0, depth)
        depth[~np.isfinite(depth)] = 0.0

        tvar = ds.variables["time"]
        times = netCDF4.num2date(
            tvar[:],
            tvar.units,
            getattr(tvar, "calendar", "standard"),
            only_use_cftime_datetimes=False,
        )
    finally:
        ds.close()

    idx = pd.DatetimeIndex(pd.to_datetime([t.isoformat() for t in times]))
    idx = idx.tz_localize(tz).tz_convert("UTC")
    return pd.Series(depth, index=idx, name="TRITON depth"), (row, col)


def sample_dem_point(dem_tif: Path, lat: float, lon: float) -> float:
    """Return the DEM bed elevation [m] at *lat*/*lon* (nearest pixel)."""
    ds = gdal.Open(str(dem_tif))
    if ds is None:
        raise FileNotFoundError(f"Cannot open DEM: {dem_tif}")
    try:
        tf = pyproj.Transformer.from_crs(
            "EPSG:4326", ds.GetProjection(), always_xy=True
        )
        px, py = tf.transform(lon, lat)
        inv = gdal.InvGeoTransform(ds.GetGeoTransform())
        if inv is None:
            raise ValueError(f"DEM geotransform is not invertible: {dem_tif}")
        col = int(inv[0] + inv[1] * px + inv[2] * py)
        row = int(inv[3] + inv[4] * px + inv[5] * py)
        if not (0 <= col < ds.RasterXSize and 0 <= row < ds.RasterYSize):
            raise ValueError(f"lat/lon {lat},{lon} falls outside the DEM extent.")
        band = ds.GetRasterBand(1)
        val = float(band.ReadAsArray(col, row, 1, 1)[0, 0])
        nodata = band.GetNoDataValue()
    finally:
        ds = None
    if nodata is not None and val == nodata:
        raise ValueError(f"DEM has NODATA at lat/lon {lat},{lon}.")
    return val


def fetch_observed_depth(
    gauge_id: str, start: str, end: str, altitude_ft: float, bed_m: float
) -> pd.Series:
    """Fetch USGS gauge height (00065) and convert it to water depth [m] at the point.

    Returns a depth series on a tz-aware UTC index, or an empty series when the
    gauge has no stage record over the window.
    """
    stage_ft = _fetch_stage_ft(gauge_id, start, end)
    if stage_ft.empty:
        return pd.Series(dtype=float)
    depth_m = (altitude_ft + stage_ft) * _FT_TO_M - bed_m
    depth_m.name = "observed depth"
    return depth_m


def _fetch_stage_ft(gauge_id: str, start: str, end: str) -> pd.Series:
    """Return the USGS instantaneous gauge height (00065) [ft] on a UTC index.

    Reads the legacy NWIS RDB instantaneous-values service and converts each
    timestamp from its reported ``tz_cd`` zone to UTC so the series aligns with
    the (localized) TRITON cube times.
    """
    params = {
        "format": "rdb",
        "sites": gauge_id,
        "startDT": start,
        "endDT": end,
        "parameterCd": "00065",
        "siteStatus": "all",
    }
    try:
        r = requests.get(_NWIS_IV, params=params, timeout=60)
    except requests.RequestException as exc:
        log.warning("USGS request for gauge %s failed: %s", gauge_id, exc)
        return pd.Series(dtype=float)
    if r.status_code != 200:
        log.warning("USGS returned HTTP %d for gauge %s.", r.status_code, gauge_id)
        return pd.Series(dtype=float)

    df = pd.read_csv(io.StringIO(r.text), sep="\t", comment="#", dtype=str)
    if df.empty or "datetime" not in df.columns or "tz_cd" not in df.columns:
        return pd.Series(dtype=float)
    df = df.iloc[1:]  # drop the RDB format-spec row (e.g. '5s\t15s\t20d\t...')
    val_cols = [c for c in df.columns if c.endswith("_00065")]
    if not val_cols:
        return pd.Series(dtype=float)

    naive = pd.to_datetime(df["datetime"], errors="coerce")
    offset_h = df["tz_cd"].map(_TZ_OFFSET_H)
    utc = (naive - pd.to_timedelta(offset_h, unit="h")).dt.tz_localize("UTC")
    stage = pd.to_numeric(df[val_cols[0]], errors="coerce")
    out = pd.Series(stage.to_numpy(), index=utc, name="stage_ft")
    out = out[out.index.notna()].dropna().sort_index()
    return out[~out.index.duplicated(keep="first")]


def _peak_stats(triton: pd.Series, observed: pd.Series) -> str:
    """One-line peak-stage/timing summary for the plot title (UTC times)."""
    pk_t, t_t = float(triton.max()), triton.idxmax()
    txt = f"peak TRITON {pk_t:.2f} m @ {t_t:%m-%d %H:%M}Z"
    if observed.empty:
        return txt + "  |  no observed stage"
    pk_o, t_o = float(observed.max()), observed.idxmax()
    dh = pk_t - pk_o
    dt_h = (t_t - t_o).total_seconds() / 3600.0
    return (
        txt + f"  |  obs {pk_o:.2f} m @ {t_o:%m-%d %H:%M}Z"
        f"  |  Δpeak {dh:+.2f} m, Δt {dt_h:+.1f} h (TRITON−obs)"
    )


def plot_compare(
    point: Dict,
    triton: pd.Series,
    observed: pd.Series,
    accuracy_m: float,
    bed_m: float,
    out_png: Path,
) -> None:
    """Plot the TRITON vs observed depth hydrographs with the accuracy band."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        triton.index, triton.to_numpy(), lw=1.4, color="tab:blue", label="TRITON depth"
    )
    if not observed.empty:
        ax.plot(
            observed.index,
            observed.to_numpy(),
            lw=1.4,
            color="black",
            label="observed depth (gauge)",
        )
        if accuracy_m > 0:
            ax.fill_between(
                observed.index,
                observed.to_numpy() - accuracy_m,
                observed.to_numpy() + accuracy_m,
                color="black",
                alpha=0.15,
                label=f"altitude accuracy ±{accuracy_m:.2f} m",
            )

    ax.set_ylabel("water depth [m]")
    ax.set_xlabel("time (UTC)")
    ax.set_title(
        f"TRITON vs gauge {point['gauge_id']} at {point['name']} "
        f"(lat {point['lat']}, lon {point['lon']}; bed {bed_m:.1f} m)\n"
        f"{_peak_stats(triton, observed)}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run(
    points: Tuple[Dict, ...],
    h_nc: Path,
    dem_tif: Path,
    tz: str,
    start: str,
    end: str,
    nodata: float,
    out_dir: Path,
) -> List[Path]:
    """Build one comparison plot per point; returns the written PNG paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for point in points:
        name = point["name"]
        lat, lon = float(point["lat"]), float(point["lon"])
        try:
            triton, (row, col) = sample_h_series(h_nc, lat, lon, tz, nodata)
            bed_m = sample_dem_point(dem_tif, lat, lon)
        except (ValueError, FileNotFoundError) as exc:
            log.warning("Skipping compare point %s: %s", name, exc)
            continue
        log.info(
            "%s: TRITON cell (row=%d, col=%d), bed elevation %.2f m",
            name,
            row,
            col,
            bed_m,
        )

        accuracy_m = float(point["altitude_accuracy_ft"]) * _FT_TO_M
        observed = fetch_observed_depth(
            point["gauge_id"], start, end, float(point["gauge_altitude_ft"]), bed_m
        )
        if observed.empty:
            log.warning(
                "No 00065 stage for gauge %s over %s..%s; plotting TRITON only.",
                point["gauge_id"],
                start,
                end,
            )

        out_png = out_dir / f"compare_{point['gauge_id']}_{name}.png"
        plot_compare(point, triton, observed, accuracy_m, bed_m, out_png)
        log.info("Wrote %s", out_png.name)
        written.append(out_png)
    return written


if __name__ == "__main__":

    map_output = Path(TRITON_MAP_OUT_DIR)
    h_nc = map_output / "H.nc"
    compare_output = Path(TRITON_COMPARE_OUT_DIR)

    run(
        points=TRITON_COMPARE_POINTS,
        h_nc=h_nc,
        dem_tif=TRITON_MAP_DEM_TIF,
        tz=TRITON_COMPARE_TZ,
        start=TRITON_START_DATE,
        end=END_DATE,
        nodata=NODATA,
        out_dir=compare_output,
    )
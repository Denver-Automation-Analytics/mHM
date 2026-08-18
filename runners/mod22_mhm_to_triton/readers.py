"""Readers for mod22 (mHM/mRM -> TRITON).

Loads the mHM gridded runoff cube, the flow-accumulation grid used to locate the
domain outlet, and the gauge coordinates used for TRITON observation points. All
spatial data is returned in the projected TRITON CRS (config.OUTPUT_CRS).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import OUTPUT_CRS, TIMESTEP

# mHM total runoff generated per cell; the TRITON gridded-runoff forcing.
RUNOFF_VAR = "Q"
# Hours represented by one output step for each supported model cadence.
STEP_HOURS = {"daily": 24, "hourly": 1}


def step_hours(timestep: str = TIMESTEP) -> int:
    """Return the number of hours spanned by one mHM output step."""
    try:
        return STEP_HOURS[timestep]
    except KeyError:
        raise ValueError(f"Unsupported TIMESTEP {timestep!r}; expected one of {list(STEP_HOURS)}.")


def read_runoff(flux_nc: Path, start=None, end=None) -> Dict:
    """Read the mHM runoff cube and its L1 grid definition.

    Returns the lazily-opened DataArray plus the grid geometry needed to map the
    finer TRITON cells onto L1 runoff zones. easting is ascending, northing is
    descending (north-up), matching how mHM writes the file. When *start*/*end*
    ('YYYY-MM-DD') are given the cube is restricted to that inclusive event window.
    """
    ds = xr.open_dataset(flux_nc, decode_times=True, chunks={"time": 365})
    if RUNOFF_VAR not in ds:
        raise KeyError(
            f"Runoff variable {RUNOFF_VAR!r} missing from {flux_nc}. Enable the "
            "total-runoff output (outputFlxState Q) and rerun mHM.")
    da = ds[RUNOFF_VAR]
    if start is not None or end is not None:
        da = da.sel(time=slice(start, end))
        if da.sizes.get("time", 0) == 0:
            raise ValueError(
                f"No runoff steps in window {start}..{end} (record spans "
                f"{str(ds['time'].values[0])[:10]}..{str(ds['time'].values[-1])[:10]}).")
    east = np.asarray(ds["easting"].values, dtype=float)
    north = np.asarray(ds["northing"].values, dtype=float)
    cs = float(abs(east[1] - east[0]))
    return {
        "da": da,
        "time": pd.DatetimeIndex(da["time"].values),
        "easting": east,
        "northing": north,
        "cellsize": cs,
        # outer edges of the L1 grid (cell centres -> edges)
        "left": float(east[0]) - cs / 2.0,
        "top": float(north[0]) + cs / 2.0,
        "units": ds[RUNOFF_VAR].attrs.get("units", ""),
    }


def find_outlet(facc_nc: Path) -> Tuple[float, float]:
    """Return the (x, y) [m, OUTPUT_CRS] of the maximum flow-accumulation cell."""
    with xr.open_dataset(facc_nc) as ds:
        facc = np.asarray(ds["facc"].values, dtype=float)
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
    facc = np.where(facc <= -9990.0, np.nan, facc)
    if not np.isfinite(facc).any():
        raise ValueError(f"No valid flow-accumulation values in {facc_nc}.")
    iy, ix = np.unravel_index(np.nanargmax(facc), facc.shape)
    return float(x[ix]), float(y[iy])


def read_gauges(id_map_csv: Path) -> List[Dict]:
    """Read gauge metadata and reproject lon/lat (WGS84) to OUTPUT_CRS.

    id_map.csv carries local_id, site_no, name, lat, lon (see mod15). The
    returned x/y are projected coordinates for the TRITON observation file.
    """
    df = pd.read_csv(id_map_csv)
    missing = {"lat", "lon"} - set(df.columns)
    if missing:
        raise KeyError(f"{id_map_csv} is missing column(s) {sorted(missing)}.")
    tf = pyproj.Transformer.from_crs("EPSG:4326", OUTPUT_CRS, always_xy=True)
    xs, ys = tf.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    gauges: List[Dict] = []
    for i, row in df.reset_index(drop=True).iterrows():
        gauges.append({
            "site_no": str(row.get("site_no", "")),
            "name": str(row.get("name", "")),
            "x": float(xs[i]),
            "y": float(ys[i]),
        })
    if not gauges:
        raise ValueError(f"No gauges found in {id_map_csv}.")
    return gauges


def read_manning_lookup(tif: Path, value_col: str = "VALUE",
                        n_col: str = "MANNINGS_N") -> Tuple[Dict[int, float], float]:
    """Return ({land-cover code: Manning n}, raster nodata) from the RAT of *tif*.

    The source land-cover raster carries a raster attribute table whose n_col
    column holds the roughness for each class value.
    """
    ds = gdal.Open(str(tif))
    if ds is None:
        raise FileNotFoundError(f"Cannot open Manning source raster: {tif}")
    band = ds.GetRasterBand(1)
    rat = band.GetDefaultRAT()
    if rat is None:
        raise ValueError(f"{tif} has no raster attribute table (need {n_col!r}).")
    cols = {rat.GetNameOfCol(c): c for c in range(rat.GetColumnCount())}
    if value_col not in cols or n_col not in cols:
        raise KeyError(f"{tif} RAT is missing {value_col!r}/{n_col!r}; has {list(cols)}.")
    vc, nc = cols[value_col], cols[n_col]
    lut = {int(rat.GetValueAsInt(r, vc)): float(rat.GetValueAsDouble(r, nc))
           for r in range(rat.GetRowCount())}
    nodata = band.GetNoDataValue()
    ds = None  # keep alive until RAT fully read, then release
    return lut, nodata


def read_baseflow_preevent(flux_nc: Path, start_date=None,
                           var: str = "QB") -> Dict:
    """Return the baseflow rate field [m/s] at the step just before the event.

    Picks the last mHM output step strictly earlier than *start_date* (or the
    first step if none precede it) so the initial condition reflects pre-event
    baseflow. The field is aligned to the L1 runoff grid.
    """
    ds = xr.open_dataset(flux_nc, decode_times=True)
    if var not in ds:
        raise KeyError(f"Baseflow variable {var!r} missing from {flux_nc}.")
    t = pd.DatetimeIndex(ds[var]["time"].values)
    if start_date is not None:
        before = t[t < pd.Timestamp(start_date)]
        sel = before[-1] if len(before) else t[0]
    else:
        sel = t[0]
    field = np.asarray(ds[var].sel(time=sel).values, dtype=float)  # mm h-1
    ds.close()
    field = np.where(np.isfinite(field), field, 0.0)
    return {"rate": field * 1e-3 / 3600.0, "time": sel}  # -> m s-1

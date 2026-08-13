"""Readers for mod20 diagnostics.

Loads mHM gridded fluxes/states and the meteo inputs, clips each grid to the
domain polygon independently (grids may differ in resolution/extent), and
returns domain-mean daily series plus per-cell totals over the evaluation
window. All grids are equal-area (EPSG:5070), so a plain cell mean equals the
area-weighted basin mean.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.vectorized
import xarray as xr
from shapely.ops import unary_union

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DOMAIN_FILE, EVAL_START_DATE, END_DATE, OUTPUT_CRS

# Flux/state variable names as written by mHM (src/mHM/mo_write_fluxes_states.f90).
FLUX_VARS = ["aET", "PET", "Q", "QB", "recharge", "preEffect"]
# Instantaneous storage states summed for the water-balance change term ΔS.
STORAGE_VARS = [
    "interception", "snowpack",
    "SWC_L01", "SWC_L02", "SWC_L03", "SWC_L04", "SWC_L05", "SWC_L06",
    "sealedSTW", "unsatSTW", "satSTW",
]


def _domain_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a (ny, nx) bool mask, True where a cell center lies in the domain."""
    gdf = gpd.read_file(DOMAIN_FILE).to_crs(OUTPUT_CRS)
    poly = unary_union(gdf.geometry.values)
    xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return shapely.vectorized.contains(poly, xx, yy)


def resolve_window(flux_nc: Path, spinup_years: int = 1):
    """Return (start, end) 'YYYY-MM-DD' covering full water years of the flux record.

    Water-balance ratios assume ΔS≈0, which only holds over whole water years.
    Starting from the model's first output, a spin-up is dropped and the start is
    snapped forward to the next 1-Oct water-year boundary; the end is the last
    available step. This is independent of the calibration EVAL_START_DATE.
    """
    with xr.open_dataset(flux_nc, decode_times=True) as ds:
        t = pd.DatetimeIndex(ds["time"].values)
    start = t[0] + pd.DateOffset(years=max(0, spinup_years))
    wy = pd.Timestamp(year=start.year, month=10, day=1)
    if wy < start:
        wy = pd.Timestamp(year=start.year + 1, month=10, day=1)
    if wy > t[-1]:
        wy = t[0]  # record shorter than one spin-up + year: use everything
    return wy.strftime("%Y-%m-%d"), t[-1].strftime("%Y-%m-%d")


def _mask_da(ds: xr.Dataset, ydim: str, xdim: str) -> xr.DataArray:
    """Domain-cell boolean mask as a DataArray aligned to (ydim, xdim)."""
    mask = _domain_mask(ds[xdim].values, ds[ydim].values)
    return xr.DataArray(mask, dims=(ydim, xdim),
                        coords={ydim: ds[ydim], xdim: ds[xdim]})


def _reduce(da: xr.DataArray, mask_da: xr.DataArray, ydim: str, xdim: str):
    """Return (daily domain-mean series, total mm, per-cell total field).

    Operates lazily on dask-backed arrays so the full 3-D volume is never held
    in memory at once; only the reduced series/field are materialised.
    """
    masked = da.where(mask_da)
    series = masked.mean(dim=(ydim, xdim), skipna=True).compute().values
    field = masked.sum(dim="time", skipna=True).compute().values
    total = float(np.nansum(series))
    return series, total, field


def read_fluxes(flux_nc: Path, window=None) -> Dict:
    """Read mHM flux/state file, restricted to *window* (start, end) dates."""
    start, end = window if window is not None else (EVAL_START_DATE, END_DATE)
    ds = xr.open_dataset(flux_nc, decode_times=True, chunks={"time": 365}).sel(
        time=slice(start, end))
    if ds.sizes.get("time", 0) == 0:
        raise ValueError(f"{flux_nc} has no time steps in {start}..{end}.")
    ydim, xdim = "northing", "easting"
    mask_da = _mask_da(ds, ydim, xdim)

    totals: Dict[str, float] = {}
    series: Dict[str, np.ndarray] = {}
    fields: Dict[str, np.ndarray] = {}
    for var in FLUX_VARS:
        if var not in ds:
            raise KeyError(
                f"Expected flux variable {var!r} missing from {flux_nc}. "
                "Enable the matching outputFlxState in mhm_outputs.nml.")
        s, tot, fld = _reduce(ds[var], mask_da, ydim, xdim)
        series[var], totals[var], fields[var] = s, tot, fld

    delta_storage = _delta_storage(ds, mask_da, ydim, xdim)

    return {
        "time": pd.DatetimeIndex(ds["time"].values),
        "totals": totals,
        "series": series,
        "fields": fields,
        "delta_storage": delta_storage,
        "mask": mask_da.values,
        "easting": np.asarray(ds["easting"].values, dtype=float),
        "northing": np.asarray(ds["northing"].values, dtype=float),
    }


def _delta_storage(ds: xr.Dataset, mask_da: xr.DataArray, ydim: str, xdim: str):
    """Sum end-minus-start domain-mean change of every storage state present [mm].

    Returns None when no storage state was written (minimal output), so the
    water-balance closure can be reported as not-available rather than wrong.
    Only the first and last time slices are read, keeping memory tiny.
    """
    total = 0.0
    found = False
    for var in STORAGE_VARS:
        if var not in ds:
            continue
        first = float(ds[var].isel(time=0).where(mask_da).mean(skipna=True).compute())
        last = float(ds[var].isel(time=-1).where(mask_da).mean(skipna=True).compute())
        total += last - first
        found = True
    return total if found else None


def read_inputs(pre_nc: Path, pet_nc: Path, window=None) -> Dict:
    """Read gross precipitation and PET inputs, clipped to the domain polygon."""
    start, end = window if window is not None else (EVAL_START_DATE, END_DATE)
    out: Dict = {"totals": {}, "series": {}, "fields": {}}
    for key, path, var in (("pre", pre_nc, "pre"), ("pet", pet_nc, "pet")):
        ds = xr.open_dataset(path, decode_times=True, chunks={"time": 730}).sel(
            time=slice(start, end))
        if ds.sizes.get("time", 0) == 0:
            raise ValueError(f"{path} has no time steps in {start}..{end}.")
        mask_da = _mask_da(ds, "y", "x")
        s, tot, fld = _reduce(ds[var], mask_da, "y", "x")
        out["series"][key], out["totals"][key], out["fields"][key] = s, tot, fld
        out.setdefault("time", pd.DatetimeIndex(ds["time"].values))
        out["x"], out["y"], out["mask"] = (
            np.asarray(ds["x"].values, dtype=float),
            np.asarray(ds["y"].values, dtype=float),
            mask_da.values,
        )
    return out


def monthly_sum(time: pd.DatetimeIndex, daily: np.ndarray) -> pd.Series:
    """Aggregate a daily domain-mean series to monthly totals [mm]."""
    return pd.Series(daily, index=time).resample("MS").sum()

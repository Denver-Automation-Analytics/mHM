"""Readers for mod20 diagnostics.

Loads mHM gridded fluxes/states and the meteo inputs, clips each grid to the
domain polygon independently (grids may differ in resolution/extent), and
returns domain-mean daily series plus per-cell totals over the evaluation
window. All grids are equal-area (EPSG:5070), so a plain cell mean equals the
area-weighted basin mean.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.vectorized
import xarray as xr
from pyproj import Transformer
from shapely.ops import unary_union

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DOMAIN_FILE, EVAL_START_DATE, END_DATE, OUTPUT_CRS

# Flux/state variable names as written by mHM (src/mHM/mo_write_fluxes_states.f90).
FLUX_VARS = ["aET", "PET", "Q", "QB", "recharge", "preEffect"]
# Instantaneous storage states summed for the water-balance change term ΔS.
STORAGE_VARS = [
    "interception",
    "snowpack",
    "SWC_L01",
    "SWC_L02",
    "SWC_L03",
    "SWC_L04",
    "SWC_L05",
    "SWC_L06",
    "sealedSTW",
    "unsatSTW",
    "satSTW",
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
    return xr.DataArray(
        mask, dims=(ydim, xdim), coords={ydim: ds[ydim], xdim: ds[xdim]}
    )


def _reduce(da: xr.DataArray, mask_da: xr.DataArray, ydim: str, xdim: str):
    """Return (daily domain-mean series, total mm, per-cell total field).

    Operates lazily on dask-backed arrays so the full 3-D volume is never held
    in memory at once; only the reduced series/field are materialised.
    """
    masked = da.where(mask_da)
    series = masked.mean(dim=(ydim, xdim), skipna=True).compute().values
    # Re-apply the mask: sum(skipna) turns all-NaN (out-of-domain) cells into 0,
    # which would pollute per-cell field stats/maps — keep them NaN instead.
    field = masked.sum(dim="time", skipna=True).where(mask_da).compute().values
    total = float(np.nansum(series))
    return series, total, field


def read_fluxes(flux_nc: Path, window=None) -> Dict:
    """Read mHM flux/state file, restricted to *window* (start, end) dates."""
    start, end = window if window is not None else (EVAL_START_DATE, END_DATE)
    ds = xr.open_dataset(flux_nc, decode_times=True, chunks={"time": 365}).sel(
        time=slice(start, end)
    )
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
                "Enable the matching outputFlxState in mhm_outputs.nml."
            )
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
            time=slice(start, end)
        )
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


# Terrain (L0 morphology) fields to map and summarise.
TERRAIN_VARS = ["dem", "slope", "aspect"]

# A parameter is "rail-pinned" when its final value sits within this fraction of
# a bound (relative to the bound span).
RAIL_TOL = 0.02
# mhm_parameter.nml / FinalParam.nml line: name = lower, upper, value, flag, flag
_PARAM_LINE = re.compile(
    r"^\s*([A-Za-z_0-9]+)\s*=\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)"
)


def read_parameters(param_nml: Path) -> List[Dict]:
    """Parse calibrated parameters (name, lower, upper, value) and flag rails.

    ``pos`` is the value's fractional position in [lower, upper]; a parameter is
    ``railed`` when pos is within RAIL_TOL of either bound, and ``fixed`` when the
    bounds are equal (not calibrated).
    """
    text = param_nml.read_bytes().decode("latin-1")
    params: List[Dict] = []
    for line in text.splitlines():
        m = _PARAM_LINE.match(line)
        if not m:
            continue
        name, lo, hi, val = (
            m.group(1),
            float(m.group(2)),
            float(m.group(3)),
            float(m.group(4)),
        )
        rng = hi - lo
        fixed = rng <= 0.0
        pos = 0.5 if fixed else (val - lo) / rng
        railed = (not fixed) and (pos <= RAIL_TOL or pos >= 1.0 - RAIL_TOL)
        params.append(
            {
                "name": name,
                "lower": lo,
                "upper": hi,
                "value": val,
                "pos": pos,
                "fixed": fixed,
                "railed": railed,
            }
        )
    if not params:
        raise ValueError(f"No parameter lines parsed from {param_nml}.")
    return params


def read_discharge(discharge_nc: Path, gauge_dir: Path, window=None) -> Dict:
    """Read routed simulated/observed discharge per gauge [m3 s-1].

    mRM writes Qsim_/Qobs_<10-digit local id> in discharge.nc; observed gaps are
    stored as the nodata sentinel and returned as NaN. Gauge metadata (USGS site
    number, name) is taken from mhm_input/gauge/id_map.csv.
    """
    start, end = window if window is not None else (EVAL_START_DATE, END_DATE)
    ds = xr.open_dataset(discharge_nc, decode_times=True).sel(time=slice(start, end))
    if ds.sizes.get("time", 0) == 0:
        raise ValueError(f"{discharge_nc} has no time steps in {start}..{end}.")
    time = pd.DatetimeIndex(ds["time"].values)

    meta = pd.read_csv(gauge_dir / "id_map.csv").set_index("local_id")
    gauges: List[Dict] = []
    for lid in meta.index:
        vsim, vobs = f"Qsim_{int(lid):010d}", f"Qobs_{int(lid):010d}"
        if vsim not in ds:
            continue
        qsim = np.asarray(ds[vsim].values, dtype=float)
        qobs = (
            np.asarray(ds[vobs].values, dtype=float)
            if vobs in ds
            else np.full_like(qsim, np.nan)
        )
        qobs[qobs <= -9990.0] = np.nan
        qsim[qsim <= -9990.0] = np.nan
        gauges.append(
            {
                "local_id": int(lid),
                "site_no": str(meta.at[lid, "site_no"]),
                "name": str(meta.at[lid, "name"]),
                "time": time,
                "qsim": qsim,
                "qobs": qobs,
            }
        )
    if not gauges:
        raise ValueError(f"No Qsim_* variables found in {discharge_nc}.")
    return {"gauges": gauges, "time": time}


def _read_gauge_obs(gauge_txt: Path, time_index: pd.DatetimeIndex) -> np.ndarray:
    """Read an mHM gauge file's observed discharge aligned to *time_index* [m3 s-1]."""
    if not gauge_txt.exists():
        return np.full(len(time_index), np.nan)
    recs: Dict[pd.Timestamp, float] = {}
    for ln in gauge_txt.read_text().splitlines():
        f = ln.split()
        if len(f) == 6 and f[0].isdigit() and len(f[0]) == 4:
            recs[
                pd.Timestamp(int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]))
            ] = float(f[5])
    vals = (
        pd.Series(recs)
        .sort_index()
        .reindex(time_index)
        .to_numpy(dtype=float, copy=True)
    )
    vals[vals <= -9990.0] = np.nan
    return vals


def read_discharge_from_qrouted(
    mrm_flux_nc: Path, gauge_dir: Path, window=None
) -> Dict:
    """Fallback hydrograph from the mRM routed-flow grid (Qrouted) [m3 s-1].

    Used when mRM's discharge.nc/subdaily_discharge.nc are missing or corrupt
    (mHM's at-exit heap-corruption crash can truncate them). mRM_Fluxes_States.nc
    is written before the crash, so the routed flow survives. Simulated gauge
    discharge = Qrouted at the gauge's outlet cell (highest mean routed flow in a
    small window around the projected gauge); observed is read from the mHM gauge
    file and aligned to the Qrouted time axis. Resolution follows the mRM output
    cadence (hourly when timeStep_model_outputs_mrm=1).
    """
    start, end = window if window is not None else (EVAL_START_DATE, END_DATE)
    ds = xr.open_dataset(mrm_flux_nc, decode_times=True).sel(time=slice(start, end))
    if "Qrouted" not in ds or ds.sizes.get("time", 0) == 0:
        raise ValueError(f"{mrm_flux_nc} has no Qrouted in {start}..{end}.")
    q = ds["Qrouted"]
    time = pd.DatetimeIndex(ds["time"].values)
    east = np.asarray(ds["easting"].values, dtype=float)
    north = np.asarray(ds["northing"].values, dtype=float)
    qmean = q.mean("time").values

    tf = Transformer.from_crs("EPSG:4326", OUTPUT_CRS, always_xy=True)
    meta = pd.read_csv(gauge_dir / "id_map.csv").set_index("local_id")
    gauges: List[Dict] = []
    for lid in meta.index:
        gx, gy = tf.transform(float(meta.at[lid, "lon"]), float(meta.at[lid, "lat"]))
        ci = int(np.argmin(np.abs(east - gx)))
        ri = int(np.argmin(np.abs(north - gy)))
        r0, r1 = max(0, ri - 2), min(len(north), ri + 3)
        c0, c1 = max(0, ci - 2), min(len(east), ci + 3)
        lr, lc = np.unravel_index(
            int(np.nanargmax(qmean[r0:r1, c0:c1])), (r1 - r0, c1 - c0)
        )
        qsim = np.asarray(q.isel(northing=r0 + lr, easting=c0 + lc).values, dtype=float)
        qobs = _read_gauge_obs(gauge_dir / f"{int(lid)}.txt", time)
        gauges.append(
            {
                "local_id": int(lid),
                "site_no": str(meta.at[lid, "site_no"]),
                "name": str(meta.at[lid, "name"]),
                "time": time,
                "qsim": qsim,
                "qobs": qobs,
            }
        )
    if not gauges:
        raise ValueError(f"No gauges in id_map.csv for {mrm_flux_nc} fallback.")
    return {"gauges": gauges, "time": time}


def read_terrain(morph_dir: Path) -> Dict:
    """Read L0 terrain fields (dem, slope, aspect), clipped to the domain polygon."""
    out: Dict = {"fields": {}}
    for var in TERRAIN_VARS:
        path = morph_dir / f"{var}.nc"
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            x = np.asarray(ds["x"].values, dtype=float)
            y = np.asarray(ds["y"].values, dtype=float)
            fld = np.asarray(ds[var].values, dtype=float)
        fld = np.where(fld <= -9990.0, np.nan, fld)
        mask = _domain_mask(x, y)
        out["fields"][var] = np.where(mask, fld, np.nan)
        out.setdefault("x", x)
        out.setdefault("y", y)
    if not out["fields"]:
        raise FileNotFoundError(
            f"No terrain fields (dem/slope/aspect) under {morph_dir}."
        )
    return out

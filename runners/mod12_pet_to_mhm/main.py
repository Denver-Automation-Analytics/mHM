"""
HRRR tavg → mHM PET preparation.

Reads a gridded temperature NetCDF, derives latitude from the companion
latlon.nc, computes PET via the chosen method, and writes mHM-ready
``pet/pet.nc`` and ``pet/header.txt`` output files.

mHM configuration expected:
    process(5)      = 0   (pre-computed PET)
    dir_referenceet = "<domain>/input/meteo/pet/"
    iFlag_cordinate_sys = 0   (projected LCC metres; same as mod11)
"""

from __future__ import annotations
import logging
import sys
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from pet import METHODS_REQUIRING_TAVG, METHODS_REQUIRING_TMAX_TMIN, _sat_vapor_pressure_kpa, pet_calculator, validate_tmin_tmax
from lat_reader import read_latitude
from utils import detect_time_freq, load_header, setup_logging
from writers import write_header_txt, write_pet

# ---------------------------------------------------------------------------
# USER INPUTS — edit these paths and settings to reconfigure
# ---------------------------------------------------------------------------
TAVG_FILE   = "/workspace/test_domain_3/input/meteo/tavg/tavg.nc"
LATLON_FILE = "/workspace/test_domain_3/input/latlon/latlon.nc"
HEADER_FILE = "/workspace/test_domain_3/input/latlon/header.txt"   # shared with mod11
PET_OUT_DIR = "/workspace/test_domain_3/input/meteo/pet"
METHOD      = "penman_monteith"     # one of: "hargreaves_samani", "oudin", "priestley_taylor", "penman_monteith"
MAX_WORKERS = 1           # set >1 for multicore parallelism
NODATA      = -9999.0

# Optional: set both to enable methods that require tmin/tmax
# (e.g. METHOD = "hargreaves_samani")
TMAX_FILE: str | None = None
TMIN_FILE: str | None = None

# Optional: set all four to enable penman_monteith
SSRD_FILE:      str | None = "/workspace/test_domain_3/input/meteo/ssrd/ssrd.nc"
STRD_FILE:      str | None = "/workspace/test_domain_3/input/meteo/strd/strd.nc"
WINDSPEED_FILE: str | None = "/workspace/test_domain_3/input/meteo/windspeed/windspeed.nc"
RHAVG_FILE:     str | None = "/workspace/test_domain_3/input/meteo/rhavg/rhavg.nc"
DEM_FILE:       str | None = "/workspace/test_domain_3/input/dem/dem_m.tif"
DEM_FEET_TO_METERS: float = 1.0   # elevation already in meters (converted by mod10)
ELEVATION_M:    float = 0.0   # fallback mean elevation (m a.s.l.) when DEM_FILE is None
# ---------------------------------------------------------------------------

log = logging.getLogger("pet_to_mhm")


def _detect_tavg_var(ds: xr.Dataset) -> str:
    for candidate in ("tavg", "tas"):
        if candidate in ds:
            return candidate
    # fall back to the single non-CRS data variable
    scalars = {v for v in ds.data_vars if ds[v].ndim == 0}
    data_vars = [v for v in ds.data_vars if v not in scalars]
    if len(data_vars) == 1:
        return data_vars[0]
    raise ValueError(
        f"Could not identify a temperature variable in {TAVG_FILE}. "
        f"Found: {list(ds.data_vars)}"
    )


def _mean_elevation(dem_path: str) -> float:
    import rioxarray as rxr  # noqa: PLC0415
    da = rxr.open_rasterio(dem_path, masked=True).squeeze()
    return float(da.mean().item())


def _compute_pm_inputs(tavg_t, rhavg_t, ssrd_t, strd_t, windspeed_t, dt_s, elevation_m):
    # FAO-56 Penman-Monteith pre-processing; all arrays shaped (1, ny, nx)
    scale = 86400.0 / (dt_s * 1e6)           # J m-2 per timestep → MJ m-2 day-1
    es    = _sat_vapor_pressure_kpa(tavg_t)
    ea    = (rhavg_t / 100.0) * es
    delta = 4098.0 * es / (tavg_t + 237.3) ** 2
    P     = 101.325 * (1.0 - 2.2577e-5 * elevation_m) ** 5.2568
    Rns   = 0.77 * ssrd_t * scale
    # Stefan-Boltzmann: 5.67e-8 W m-2 K-4 = 4.903e-9 MJ m-2 day-1 K-4
    Rnl   = strd_t * scale - 0.98 * 4.903e-9 * (tavg_t + 273.15) ** 4
    return dict(
        delta=delta,
        rn=Rns + Rnl,
        g=0.0,
        gamma=0.000665 * P,
        u2=windspeed_t * 4.87 / np.log(67.8 * 10.0 - 5.42),  # FAO-56 eq. 47, 10 m → 2 m
        es=es,
        ea=ea,
    )


def main() -> None:
    setup_logging()

    # 1. Load temperature forcing
    log.info("Opening %s", TAVG_FILE)
    ds = xr.open_dataset(TAVG_FILE)
    tavg_var = _detect_tavg_var(ds)
    tavg = ds[tavg_var]
    stat_freq, ref_time = detect_time_freq(ds)
    log.info("Detected frequency: %s  |  ref_time: %s", stat_freq, ref_time)

    # 2. Validate method requirements
    if METHOD in METHODS_REQUIRING_TAVG and TAVG_FILE is None:
        sys.exit(f"Method '{METHOD}' requires TAVG_FILE.")
    if METHOD in METHODS_REQUIRING_TMAX_TMIN and (TMAX_FILE is None or TMIN_FILE is None):
        sys.exit(f"Method '{METHOD}' requires both TMAX_FILE and TMIN_FILE.")
    if METHOD in {"penman_monteith", "penman-monteith"}:
        missing_pm = [n for n, f in (("SSRD_FILE", SSRD_FILE), ("STRD_FILE", STRD_FILE),
                                      ("WINDSPEED_FILE", WINDSPEED_FILE), ("RHAVG_FILE", RHAVG_FILE)) if f is None]
        if missing_pm:
            sys.exit(f"Method '{METHOD}' requires: {', '.join(missing_pm)}")

    # 3. Optional tmin/tmax
    tmin = tmax = None
    if TMAX_FILE is not None and TMIN_FILE is not None:
        ds_tmin = xr.open_dataset(TMIN_FILE)
        ds_tmax = xr.open_dataset(TMAX_FILE)
        tmin_var = next((v for v in ("tmin", "tasmin") if v in ds_tmin), list(ds_tmin.data_vars)[0])
        tmax_var = next((v for v in ("tmax", "tasmax") if v in ds_tmax), list(ds_tmax.data_vars)[0])
        tmin = ds_tmin[tmin_var]
        tmax = ds_tmax[tmax_var]
        if METHOD in METHODS_REQUIRING_TMAX_TMIN:
            validate_tmin_tmax(tmin=tmin.values, tmax=tmax.values)

    # PM inputs (ssrd, strd, windspeed, rhavg)
    ssrd = strd = windspeed = rhavg = None
    dt_s = 3600.0
    elevation_m = ELEVATION_M
    if DEM_FILE is not None:
        elevation_m = _mean_elevation(DEM_FILE) * DEM_FEET_TO_METERS
        log.info("Mean elevation from DEM: %.1f m", elevation_m)
    if METHOD in {"penman_monteith", "penman-monteith"}:
        log.info("Loading PM inputs from %s / %s / %s / %s",
                 SSRD_FILE, STRD_FILE, WINDSPEED_FILE, RHAVG_FILE)
        ds_ssrd      = xr.open_dataset(SSRD_FILE)
        ds_strd      = xr.open_dataset(STRD_FILE)
        ds_windspeed = xr.open_dataset(WINDSPEED_FILE)
        ds_rhavg     = xr.open_dataset(RHAVG_FILE)
        ssrd      = ds_ssrd[list(ds_ssrd.data_vars)[0]]
        strd      = ds_strd[list(ds_strd.data_vars)[0]]
        windspeed = ds_windspeed[list(ds_windspeed.data_vars)[0]]
        rhavg     = ds_rhavg[list(ds_rhavg.data_vars)[0]]
        ssrd_times = pd.DatetimeIndex(ds_ssrd["time"].values)
        dt_s = float((ssrd_times[1] - ssrd_times[0]).total_seconds()) if len(ssrd_times) >= 2 else 3600.0
        log.info("Radiation timestep dt_s = %.0f s", dt_s)

    # 4. Latitude — read from latlon.nc; reshape to (1, nrows, ncols) for broadcasting
    log.info("Reading latitude from %s", LATLON_FILE)
    lat_deg = read_latitude(Path(LATLON_FILE))   # (nrows, ncols)
    lat3d = np.radians(lat_deg)[np.newaxis, :, :]

    # 5. Build per-timestep task list
    log.info("Building %d PET tasks", len(ds.time))
    tasks = []
    for idx in range(len(ds.time)):
        t = pd.Timestamp(ds.time.values[idx])
        current_time = t.to_pydatetime().replace(tzinfo=timezone.utc)
        task = {
            "lat":       lat3d,
            "time":      current_time,
            "stat_freq": stat_freq,
            "method":    METHOD,
            "tavg":      tavg.isel(time=idx).values[np.newaxis, :, :],
            "tmin":      tmin.isel(time=idx).values[np.newaxis, :, :] if tmin is not None else None,
            "tmax":      tmax.isel(time=idx).values[np.newaxis, :, :] if tmax is not None else None,
        }
        if ssrd is not None:
            task.update(_compute_pm_inputs(
                task["tavg"],
                rhavg.isel(time=idx).values[np.newaxis, :, :],
                ssrd.isel(time=idx).values[np.newaxis, :, :],
                strd.isel(time=idx).values[np.newaxis, :, :],
                windspeed.isel(time=idx).values[np.newaxis, :, :],
                dt_s,
                elevation_m,
            ))
        tasks.append(task)

    # 6. Compute PET in parallel
    log.info("Computing PET on %d worker(s)", MAX_WORKERS)
    results = Parallel(n_jobs=MAX_WORKERS, backend="loky")(
        delayed(pet_calculator)(**task) for task in tasks
    )

    # 7. Stack and cast
    pet_data = np.vstack(results).astype(np.float32)
    log.info("PET array shape: %s  range: [%.3f, %.3f]",
             pet_data.shape, float(np.nanmin(pet_data)), float(np.nanmax(pet_data)))

    # 8. Write outputs
    out_dir = Path(PET_OUT_DIR)
    write_pet(pet_data, ds, tavg_var, out_dir / "pet.nc", ref_time, NODATA, stat_freq)
    write_header_txt(load_header(Path(HEADER_FILE)), out_dir / "header.txt")

    ds.close()
    if tmin is not None:
        ds_tmin.close()
        ds_tmax.close()
    if ssrd is not None:
        ds_ssrd.close()
        ds_strd.close()
        ds_windspeed.close()
        ds_rhavg.close()
    log.info("Done. Output: %s", out_dir)


if __name__ == "__main__":
    main()

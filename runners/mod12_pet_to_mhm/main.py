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
import os
import sys
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_CRS, NODATA, WORKING_DIR, PET_METHOD  # noqa: E402

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from pet import METHODS_REQUIRING_TAVG, METHODS_REQUIRING_TMAX_TMIN, _sat_vapor_pressure_kpa, pet_calculator, validate_tmin_tmax
from lat_reader import compute_latitude_from_header
from utils import detect_time_freq, load_header, setup_logging
from writers import create_pet_nc, write_header_txt, write_pet_chunk

# ---------------------------------------------------------------------------
# USER INPUTS — edit these paths and settings to reconfigure
# ---------------------------------------------------------------------------
TAVG_FILE   = os.path.join(WORKING_DIR, "input/meteo/tavg/tavg.nc")
HEADER_FILE = os.path.join(WORKING_DIR, "input/latlon/header.txt")   # shared with mod11
PET_OUT_DIR = os.path.join(WORKING_DIR, "input/meteo/pet")
MAX_WORKERS = 10           # set >1 for multicore parallelism
CHUNK_SIZE  = 256           # timesteps per batch; lower = less peak RAM

# Optional: set both to enable methods that require tmin/tmax
# (e.g. METHOD = "hargreaves_samani")
TMAX_FILE: str | None = None
TMIN_FILE: str | None = None

# Optional: set all four to enable penman_monteith
SSRD_FILE = os.path.join(WORKING_DIR, "input/meteo/ssrd/ssrd.nc")
STRD_FILE = os.path.join(WORKING_DIR, "input/meteo/strd/strd.nc")
WINDSPEED_FILE = os.path.join(WORKING_DIR, "input/meteo/windspeed/windspeed.nc")
RHAVG_FILE = os.path.join(WORKING_DIR, "input/meteo/rhavg/rhavg.nc")
DEM_FILE = os.path.join(WORKING_DIR, "input/dem/dem_corrected.tif")
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
    # block-windowed read to avoid loading the full DEM into RAM
    import rasterio  # noqa: PLC0415
    with rasterio.open(dem_path) as src:
        total, count = 0.0, 0
        for _, window in src.block_windows(1):
            valid = src.read(1, window=window, masked=True).compressed().astype(np.float64)
            valid = valid[np.isfinite(valid)]
            total += valid.sum()
            count += valid.size
    return total / count if count > 0 else 0.0


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
    if PET_METHOD in METHODS_REQUIRING_TAVG and TAVG_FILE is None:
        sys.exit(f"Method '{PET_METHOD}' requires TAVG_FILE.")
    if PET_METHOD in METHODS_REQUIRING_TMAX_TMIN and (TMAX_FILE is None or TMIN_FILE is None):
        sys.exit(f"Method '{PET_METHOD}' requires both TMAX_FILE and TMIN_FILE.")
    if PET_METHOD in {"penman_monteith", "penman-monteith"}:
        missing_pm = [n for n, f in (("SSRD_FILE", SSRD_FILE), ("STRD_FILE", STRD_FILE),
                                      ("WINDSPEED_FILE", WINDSPEED_FILE), ("RHAVG_FILE", RHAVG_FILE)) if f is None]
        if missing_pm:
            sys.exit(f"Method '{PET_METHOD}' requires: {', '.join(missing_pm)}")

    # 3. Optional tmin/tmax
    tmin = tmax = None
    if TMAX_FILE is not None and TMIN_FILE is not None:
        ds_tmin = xr.open_dataset(TMIN_FILE)
        ds_tmax = xr.open_dataset(TMAX_FILE)
        tmin_var = next((v for v in ("tmin", "tasmin") if v in ds_tmin), list(ds_tmin.data_vars)[0])
        tmax_var = next((v for v in ("tmax", "tasmax") if v in ds_tmax), list(ds_tmax.data_vars)[0])
        tmin = ds_tmin[tmin_var]
        tmax = ds_tmax[tmax_var]
        if PET_METHOD in METHODS_REQUIRING_TMAX_TMIN:
            validate_tmin_tmax(tmin=tmin.values, tmax=tmax.values)

    # PM inputs (ssrd, strd, windspeed, rhavg)
    ssrd = strd = windspeed = rhavg = None
    dt_s = 3600.0
    if DEM_FILE is not None:
        elevation_m = _mean_elevation(DEM_FILE)
        log.info("Mean elevation from DEM: %.1f m", elevation_m)
    if PET_METHOD in {"penman_monteith", "penman-monteith"}:
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

    # 4. Latitude — derived from header.txt + projected CRS; reshape to (1, nrows, ncols) for broadcasting
    log.info("Deriving latitude from %s (CRS: %s)", HEADER_FILE, OUTPUT_CRS)
    lat_deg = compute_latitude_from_header(Path(HEADER_FILE), OUTPUT_CRS)   # (nrows, ncols)
    lat3d = np.radians(lat_deg)[np.newaxis, :, :]

    # 5. Pre-compute time offsets (int32 hours-since-ref — negligible memory)
    n_times = len(ds.time)
    all_times = pd.DatetimeIndex(ds.time.values)
    time_offsets = ((all_times - pd.Timestamp(ref_time)) / pd.Timedelta("1h")).to_numpy(dtype="int32")

    # 6. Create output NetCDF structure once (unlimited time; filled chunk-by-chunk)
    out_dir = Path(PET_OUT_DIR)
    out_path = out_dir / "pet.nc"
    create_pet_nc(out_path, ds, tavg_var, ref_time, NODATA, stat_freq, nc_chunk_t=CHUNK_SIZE)

    # 7. Process and write CHUNK_SIZE timesteps at a time
    log.info("Computing PET: %d steps, chunk=%d, workers=%d, backend=threading",
             n_times, CHUNK_SIZE, MAX_WORKERS)
    pet_min, pet_max = np.inf, -np.inf
    for chunk_start in range(0, n_times, CHUNK_SIZE):
        chunk_slice = slice(chunk_start, min(chunk_start + CHUNK_SIZE, n_times))

        tavg_chunk = tavg.isel(time=chunk_slice).values          # (cs, ny, nx)
        tmin_chunk = tmin.isel(time=chunk_slice).values if tmin is not None else None
        tmax_chunk = tmax.isel(time=chunk_slice).values if tmax is not None else None
        if ssrd is not None:
            ssrd_chunk      = ssrd.isel(time=chunk_slice).values
            strd_chunk      = strd.isel(time=chunk_slice).values
            windspeed_chunk = windspeed.isel(time=chunk_slice).values
            rhavg_chunk     = rhavg.isel(time=chunk_slice).values

        tasks = []
        for i, abs_idx in enumerate(range(chunk_start, chunk_start + tavg_chunk.shape[0])):
            t = pd.Timestamp(ds.time.values[abs_idx])
            task = {
                "lat":       lat3d,
                "time":      t.to_pydatetime().replace(tzinfo=timezone.utc),
                "stat_freq": stat_freq,
                "method":    PET_METHOD,
                "tavg":      tavg_chunk[i : i + 1],
                "tmin":      tmin_chunk[i : i + 1] if tmin_chunk is not None else None,
                "tmax":      tmax_chunk[i : i + 1] if tmax_chunk is not None else None,
            }
            if ssrd is not None:
                task.update(_compute_pm_inputs(
                    task["tavg"],
                    rhavg_chunk[i : i + 1],
                    ssrd_chunk[i : i + 1],
                    strd_chunk[i : i + 1],
                    windspeed_chunk[i : i + 1],
                    dt_s,
                    elevation_m,
                ))
            tasks.append(task)

        results = Parallel(n_jobs=MAX_WORKERS, backend="threading")(
            delayed(pet_calculator)(**task) for task in tasks
        )

        chunk_pet = np.vstack(results).astype(np.float32)
        write_pet_chunk(out_path, chunk_pet, time_offsets[chunk_slice], chunk_start)

        cmin, cmax = float(np.nanmin(chunk_pet)), float(np.nanmax(chunk_pet))
        pet_min = min(pet_min, cmin)
        pet_max = max(pet_max, cmax)
        del tavg_chunk, tmin_chunk, tmax_chunk, tasks, results, chunk_pet
        if ssrd is not None:
            del ssrd_chunk, strd_chunk, windspeed_chunk, rhavg_chunk

    log.info("PET written: %d steps, range [%.3f, %.3f]", n_times, pet_min, pet_max)

    # 8. Write header
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

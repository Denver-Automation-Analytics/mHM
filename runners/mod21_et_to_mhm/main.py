"""Monthly actual-ET observations → mHM calibration constraint.

Fetches monthly actual evapotranspiration (TerraClimate ``aet``) from the
Microsoft Planetary Computer, samples it onto the mHM L1 grid, and writes
``et.nc`` (+ ``header.txt``) into input/optional_data so mod19 can calibrate
against observed ET (opti_function kge_q_et, timeStep_et_input=-2).

Run order: mod10 → … → mod18 → mod21 (this) → mod19 (calibration) → mod20.

mHM's combined discharge+ET objectives require monthly (not yearly) ET, and the
per-cell KGE(ET) term penalises the actual-ET magnitude bias directly. TerraClimate
ends in 2021, so later months are gap-filled with the per-cell monthly climatology
to cover the whole modelling period.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import shapely
import xarray as xr
from shapely.ops import unary_union

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import END_DATE, OUTPUT_CRS, START_DATE, WORKING_DIR

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "terraclimate"
NODATA = -9999.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("et_to_mhm")

WATERSHED_FILE = os.path.join(WORKING_DIR, "input", "domain", "watershed.geojson")
if not os.path.exists(WATERSHED_FILE):
    raise FileNotFoundError(f"Expected {WATERSHED_FILE} to exist; run mod10 first.")

def _l1_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return (easting, northing) cell centres of the mHM L1 grid [EPSG:5070]."""
    flux = Path(WORKING_DIR) / "output" / "mHM_Fluxes_States.nc"
    if flux.exists():
        with xr.open_dataset(flux) as ds:
            return (np.asarray(ds["easting"].values, float),
                    np.asarray(ds["northing"].values, float))
    with xr.open_dataset(Path(WORKING_DIR) / "input" / "latlon" / "latlon.nc") as ds:
        return (np.asarray(ds["xc"].values, float),
                np.asarray(ds["yc"].values, float))


def _l1_lonlat() -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D (lon, lat) of L1 cell centres [degrees] for sampling."""
    with xr.open_dataset(Path(WORKING_DIR) / "input" / "latlon" / "latlon.nc") as ds:
        return np.asarray(ds["lon"].values, float), np.asarray(ds["lat"].values, float)


def _domain_mask(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Boolean (ny, nx) mask of L1 cells inside the domain polygon."""
    gdf = gpd.read_file(WATERSHED_FILE).to_crs(OUTPUT_CRS)
    poly = unary_union(gdf.geometry.values)
    xx, yy = np.meshgrid(east, north)
    return shapely.contains_xy(poly, xx, yy)


def fetch_monthly_et(lon2d: np.ndarray, lat2d: np.ndarray) -> xr.DataArray:
    """Sample TerraClimate monthly AET onto the L1 grid for START..END.

    Returns (time, y, x) monthly ET [mm]; months beyond the TerraClimate record
    are gap-filled with the per-cell monthly climatology.
    """
    cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    asset = cat.get_collection(COLLECTION).assets["zarr-abfs"]
    ds = xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])

    minx, miny, maxx, maxy = gpd.read_file(WATERSHED_FILE).to_crs(4326).total_bounds
    aet = ds["aet"].sel(lon=slice(minx - 0.1, maxx + 0.1),
                        lat=slice(maxy + 0.1, miny - 0.1)).load()
    ds.close()  # release the Azure blob store now that data is in memory

    months = pd.date_range(f"{START_DATE[:4]}-01-01", f"{END_DATE[:4]}-12-01", freq="MS")
    lonx = xr.DataArray(lon2d, dims=("y", "x"))
    laty = xr.DataArray(lat2d, dims=("y", "x"))

    real = aet.interp(lon=lonx, lat=laty, method="linear")   # (time, y, x)
    real = real.reindex(time=months)                         # NaN where TerraClimate absent
    clim = real.groupby("time.month").mean("time", skipna=True)

    filled = real.copy()
    empty = real.isnull().all(dim=("y", "x")).values
    missing = pd.DatetimeIndex(months[empty])
    for ts in missing:
        filled.loc[{"time": ts}] = clim.sel(month=ts.month)
    if len(missing):
        log.info("Gap-filled %d months (%s..%s) with monthly climatology.",
                 len(missing), missing[0].strftime("%Y-%m"), missing[-1].strftime("%Y-%m"))
    # mHM expects monthly timesteps stamped at month end.
    filled = filled.assign_coords(time=months.to_period("M").to_timestamp("M"))
    return filled.rename("et")


def write_et_nc(et: xr.DataArray, mask: np.ndarray, east: np.ndarray,
                north: np.ndarray, out_dir: Path) -> Path:
    """Write et.nc (var 'et', monthly) and header.txt on the L1 grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vals = np.asarray(et.values, float)
    vals[:, ~mask] = NODATA
    vals = np.where(np.isfinite(vals), vals, NODATA)

    da = xr.DataArray(vals, dims=("time", "y", "x"),
                      coords={"time": et["time"].values, "y": north, "x": east}, name="et")
    da.attrs = {"units": "mm", "long_name": "actual evapotranspiration"}
    da["x"].attrs = {"units": "m", "standard_name": "projection_x_coordinate", "axis": "X"}
    da["y"].attrs = {"units": "m", "standard_name": "projection_y_coordinate", "axis": "Y"}
    path = out_dir / "et.nc"
    ref = pd.Timestamp(et["time"].values[0]).strftime("%Y-%m-%d")
    da.to_dataset().to_netcdf(
        path, encoding={"et": {"_FillValue": NODATA, "missing_value": NODATA},
                        "time": {"dtype": "i4", "units": f"days since {ref}",
                                 "calendar": "standard"}})

    res = float(abs(east[1] - east[0]))
    (out_dir / "header.txt").write_text(
        f"ncols        {len(east)}\n"
        f"nrows        {len(north)}\n"
        f"xllcorner    {east.min() - res / 2:.1f}\n"
        f"yllcorner    {north.min() - res / 2:.1f}\n"
        f"cellsize     {res:.1f}\n"
        f"NODATA_value {NODATA:.0f}\n")
    return path


def main() -> None:
    east, north = _l1_grid()
    lon2d, lat2d = _l1_lonlat()
    log.info("L1 grid: %d x %d @ %d m", len(east), len(north), int(abs(east[1] - east[0])))

    et = fetch_monthly_et(lon2d, lat2d)
    mask = _domain_mask(east, north)

    basin = np.where(mask[None], et.values, np.nan)
    ann = pd.Series(np.nanmean(basin, axis=(1, 2)),
                    index=pd.DatetimeIndex(et["time"].values)).resample("YS").sum()
    for yr, v in ann.items():
        log.info("  %d basin-mean ET = %.0f mm/yr", yr.year, v)

    path = write_et_nc(et, mask, east, north, Path(WORKING_DIR) / "input" / "optional_data")
    log.info("Wrote %s (+ header.txt), %d monthly steps %s..%s.", path, et.sizes["time"],
             pd.Timestamp(et["time"].values[0]).strftime("%Y-%m"),
             pd.Timestamp(et["time"].values[-1]).strftime("%Y-%m"))


if __name__ == "__main__":
    main()

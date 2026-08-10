"""
Regrid existing mHM meteo NetCDF files to the exact L2 extent that mHM derives
from the L0 DEM at runtime (via calculate_grid_properties).

Run ONCE after mod11/mod12 if the meteo files were written with the wrong grid:

    python runners/fix_meteo_grid.py

Edits files in-place (writes a .bak backup first).
"""

from __future__ import annotations
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "mod11_meteo_to_mhm"))
from latlon_grid import mhm_l2_from_l0  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fix_meteo_grid")

# ---- USER INPUTS -------------------------------------------------------
L0_DEM   = "/workspace/test_domain_3/input/morph/dem.nc"
METEO_DIR = "/workspace/test_domain_3/input/meteo"
L2_CELLSIZE = 3000          # metres — must match config.py L2_CELL_SIZE_M
OUTPUT_CRS  = "EPSG:5070"
NODATA      = -9999.0
# ------------------------------------------------------------------------


def _write_header_txt(path: Path, h: dict) -> None:
    path.write_text(
        f"ncols        {h['ncols']}\n"
        f"nrows        {h['nrows']}\n"
        f"xllcorner    {h['xllcorner']}\n"
        f"yllcorner    {h['yllcorner']}\n"
        f"cellsize     {int(h['cellsize'])}\n"
        f"NODATA_value {int(h['NODATA_value'])}\n"
    )


def regrid_nc(src: Path, dst: Path, l2: dict) -> None:
    """Regrid one meteo NC file to the target L2 grid."""
    try:
        import xarray as xr
        import rioxarray  # noqa: F401
        from affine import Affine
        from rasterio.enums import Resampling
    except ImportError as e:
        raise ImportError("rioxarray, xarray, affine and rasterio are required") from e

    ncols = l2["ncols"]   # x/E-W
    nrows = l2["nrows"]   # y/N-S
    xll   = l2["xllcorner"]
    yll   = l2["yllcorner"]
    cs    = l2["cellsize"]
    yur   = yll + nrows * cs

    target_transform = Affine(cs, 0.0, xll, 0.0, -cs, yur)
    target_shape     = (nrows, ncols)   # (height, width) = (y, x)

    ds = xr.open_dataset(src)
    # ensure spatial dims are declared
    if "x" in ds.dims and "y" in ds.dims:
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    if ds.rio.crs is None:
        ds = ds.rio.write_crs(OUTPUT_CRS)

    ds_out = ds.rio.reproject(
        OUTPUT_CRS,
        shape=target_shape,
        transform=target_transform,
        resampling=Resampling.bilinear,
        nodata=NODATA,
    )
    # Only preserve _FillValue; drop chunksizes/compression (may exceed new dims)
    enc = {
        v: {"_FillValue": ds[v].encoding.get("_FillValue", NODATA)}
        for v in ds.data_vars if v in ds_out
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(dst, encoding=enc)
    ds.close()
    ds_out.close()


def main() -> None:
    l2 = mhm_l2_from_l0(L0_DEM, L2_CELLSIZE)
    log.info(
        "Target L2 grid: ncols=%d (x/E-W)  nrows=%d (y/N-S)  "
        "xll=%.1f  yll=%.1f  cellsize=%d",
        l2["ncols"], l2["nrows"], l2["xllcorner"], l2["yllcorner"], l2["cellsize"],
    )

    meteo_root = Path(METEO_DIR)
    nc_files   = sorted(meteo_root.rglob("*.nc"))
    hdr_files  = sorted(meteo_root.rglob("header.txt"))

    if not nc_files:
        log.error("No .nc files found under %s", meteo_root)
        sys.exit(1)

    for src in nc_files:
        bak = src.with_suffix(".nc.bak")
        if not bak.exists():
            shutil.copy2(src, bak)
            log.info("Backup: %s", bak)
        log.info("Regriding %s …", src.relative_to(meteo_root))
        tmp = src.with_suffix(".nc.tmp")
        regrid_nc(src, tmp, l2)
        tmp.replace(src)
        log.info("  → written (%d×%d x,y)", l2["ncols"], l2["nrows"])

    for hdr in hdr_files:
        _write_header_txt(hdr, l2)
        log.info("Updated header: %s", hdr.relative_to(meteo_root))

    log.info("Done.")


if __name__ == "__main__":
    main()

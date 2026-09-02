"""Shared utilities: logging, DEM-grid derivation, and ESRI ASCII output."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_dem_grid(dem_nc: str | Path, nodata: int = -9999) -> dict:
    """Derive the L0 grid definition and domain mask from mod10's dem.nc.

    Returns keys: ncols, nrows, cellsize, xllcorner, yllcorner, xtop, ytop,
    valid_mask (nrows x ncols bool, row 0 = northernmost, matching ESRI ASCII).
    """
    import netCDF4 as nc4  # noqa: PLC0415

    dem_nc = Path(dem_nc)
    if not dem_nc.is_file():
        raise FileNotFoundError(f"DEM grid not found: {dem_nc}")

    with nc4.Dataset(dem_nc) as ds:
        x = np.asarray(ds.variables["x"][:], dtype=float)
        y = np.asarray(ds.variables["y"][:], dtype=float)
        dem_var = ds.variables["dem"]
        fill = float(getattr(dem_var, "_FillValue", nodata))
        dem = dem_var[:]

    valid_mask = ~np.ma.getmaskarray(dem)
    valid_mask &= np.asarray(dem) != fill

    nrows, ncols = dem.shape
    cellsize = float(x[1] - x[0])
    xll = float(x[0]) - cellsize / 2.0
    yll = float(y[-1]) - cellsize / 2.0
    ytop = yll + nrows * cellsize

    return {
        "ncols": int(ncols),
        "nrows": int(nrows),
        "cellsize": cellsize,
        "xllcorner": xll,
        "yllcorner": yll,
        "xtop": xll,
        "ytop": ytop,
        "valid_mask": np.asarray(valid_mask, dtype=bool),
    }


def write_ascii_grid(
    path: str | Path, grid: np.ndarray, grid_def: dict, nodata: int = -9999
) -> None:
    """Write an integer grid as a 6-line-header ESRI ASCII file (north-up)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"ncols         {grid_def['ncols']}\n")
        fh.write(f"nrows         {grid_def['nrows']}\n")
        fh.write(f"xllcorner     {grid_def['xllcorner']:.1f}\n")
        fh.write(f"yllcorner     {grid_def['yllcorner']:.1f}\n")
        fh.write(f"cellsize      {int(grid_def['cellsize'])}\n")
        fh.write(f"NODATA_value  {nodata}\n")
        np.savetxt(fh, grid, fmt="%d")

"""Readers for mod22 (TRITON gtiff -> map).

Discovers the per-timestep TRITON GeoTIFF series, reads the shared grid geometry
and CRS straight from the ``.vrt`` (which merges the partition tiles TRITON writes
in parallel runs), parses the output cadence from the run's ``.cfg``, and streams
one timestep slice at a time so the full space-time cube is never held in memory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()

# TRITON spatial output variables, in the order they are consolidated/derived.
BASE_VARS = ("H", "MH", "QX", "QY")
# <VAR>_<NN>.vrt (partition-merged) with a zero-padded, 1-based timestep index.
_VRT_RE = re.compile(r"^(?P<var>[A-Za-z]+)_(?P<idx>\d+)\.vrt$")


def discover_timesteps(gtiff_dir: Path, var: str) -> List[Path]:
    """Return the *var* ``.vrt`` files sorted ascending by timestep index."""
    hits = []
    for p in gtiff_dir.glob(f"{var}_*.vrt"):
        m = _VRT_RE.match(p.name)
        if m and m.group("var") == var:
            hits.append((int(m.group("idx")), p))
    return [p for _, p in sorted(hits, key=lambda t: t[0])]


def available_vars(gtiff_dir: Path) -> Dict[str, List[Path]]:
    """Map each TRITON variable present in *gtiff_dir* to its timestep vrt list."""
    return {v: paths for v in BASE_VARS if (paths := discover_timesteps(gtiff_dir, v))}


def read_grid(vrt: Path) -> Dict:
    """Read grid geometry and CRS from a single timestep *vrt*.

    Returns the raster size, GDAL geotransform, projection WKT, an EPSG string
    when resolvable, and the 1-D cell-centre coordinate axes (y descending,
    x ascending) used for the netCDF spatial coordinates.
    """
    ds = gdal.Open(str(vrt))
    if ds is None:
        raise FileNotFoundError(f"Cannot open TRITON vrt: {vrt}")
    nx, ny = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    ds = None

    x0, dx, _, y0, _, dy = gt
    x = x0 + (np.arange(nx) + 0.5) * dx
    y = y0 + (np.arange(ny) + 0.5) * dy

    epsg = None
    if wkt:
        from osgeo import osr

        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        code = srs.GetAuthorityCode(None)
        if code:
            epsg = f"{srs.GetAuthorityName(None)}:{code}"
    return {"nx": nx, "ny": ny, "gt": gt, "wkt": wkt, "epsg": epsg, "x": x, "y": y}


def read_slice(vrt: Path, nx: int, ny: int) -> np.ndarray:
    """Read a single timestep vrt as a (ny, nx) float32 array (partitions merged)."""
    ds = gdal.Open(str(vrt))
    if ds is None:
        raise FileNotFoundError(f"Cannot open TRITON vrt: {vrt}")
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    if arr.shape != (ny, nx):
        raise ValueError(f"{vrt.name} is {arr.shape}, expected {(ny, nx)}.")
    return arr.astype(np.float32, copy=False)


def parse_print_interval(cfg: Path) -> int:
    """Return the TRITON spatial-output cadence [s] parsed from *cfg*."""
    txt = Path(cfg).read_text()
    m = re.search(r"^\s*print_interval\s*=\s*([0-9.]+)", txt, re.MULTILINE)
    if not m:
        raise ValueError(f"print_interval not found in {cfg}.")
    return int(float(m.group(1)))


def build_time_axis(n_steps: int, start_date: str, interval_s: int) -> pd.DatetimeIndex:
    """Datetimes for each output step: index-based, step k (1-based) at k*interval."""
    base = pd.Timestamp(start_date)
    return pd.DatetimeIndex(
        [base + pd.Timedelta(seconds=(k + 1) * interval_s) for k in range(n_steps)]
    )


def build_watershed_mask(watershed_file: Path, grid: Dict) -> np.ndarray:
    """Rasterize *watershed_file* onto *grid* -> boolean (ny, nx), True inside.

    The polygon is reprojected to the grid CRS before burning; a cell counts as
    inside when the boundary touches it (allTouched), so edge cells are kept.
    """
    import tempfile

    import geopandas as gpd

    dom = gpd.read_file(watershed_file).to_crs(grid["epsg"] or grid["wkt"])
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        dom.to_file(tmp_path, driver="GPKG")
        mem = gdal.GetDriverByName("MEM").Create(
            "", grid["nx"], grid["ny"], 1, gdal.GDT_Byte
        )
        mem.SetGeoTransform(grid["gt"])
        if grid.get("wkt"):
            mem.SetProjection(grid["wkt"])
        gdal.Rasterize(mem, tmp_path, burnValues=[1], allTouched=True)
        mask = mem.GetRasterBand(1).ReadAsArray().astype(bool)
        mem = None
    finally:
        os.remove(tmp_path)
    return mask


def read_hillshade(dem_tif: Path, grid: Dict) -> np.ndarray:
    """Return a hillshade (ny, nx) aligned to *grid*, computed from *dem_tif*.

    The DEM is hillshaded natively then warped onto the target grid's transform,
    size and CRS so it lines up pixel-for-pixel even if the source extent differs.
    """
    hs = gdal.DEMProcessing("", str(dem_tif), "hillshade", format="MEM")
    ny, nx = grid["ny"], grid["nx"]
    warped = gdal.Warp(
        "",
        hs,
        format="MEM",
        width=nx,
        height=ny,
        outputBounds=_grid_bounds(grid),
        dstSRS=grid.get("wkt") or None,
        resampleAlg="bilinear",
    )
    hs = None
    arr = warped.GetRasterBand(1).ReadAsArray().astype(np.float32)
    warped = None
    return arr


def _grid_bounds(grid: Dict) -> tuple:
    """Return (minx, miny, maxx, maxy) of *grid* from its geotransform + size."""
    x0, dx, _, y0, _, dy = grid["gt"]
    nx, ny = grid["nx"], grid["ny"]
    x1, y1 = x0 + nx * dx, y0 + ny * dy
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def read_boundary_line(clip_path: Path, grid: Dict):
    """Return the *clip_path* polygon boundary (line geometries) in *grid*'s CRS."""
    import geopandas as gpd

    dom = gpd.read_file(clip_path).to_crs(grid["epsg"] or grid["wkt"])
    return dom.boundary


def max_from_netcdf(nc_path: Path, var: str, nodata: float) -> np.ndarray:
    """Return the pixel-maximum (ny, nx) over time from a cached netCDF cube.

    Streams one timestep at a time so the whole cube is never materialised; the
    NODATA fill is treated as missing and ignored in the maximum.
    """
    import netCDF4

    ds = netCDF4.Dataset(nc_path, "r")
    try:
        v = ds.variables[var]
        v.set_auto_maskandscale(False)
        mx = np.full(v.shape[1:], np.nan, dtype=np.float32)
        for t in range(v.shape[0]):
            arr = np.asarray(v[t], dtype=np.float32)
            arr = np.where(arr == np.float32(nodata), np.float32(np.nan), arr)
            np.fmax(mx, arr, out=mx)
    finally:
        ds.close()
    return mx

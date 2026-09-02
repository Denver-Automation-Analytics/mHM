"""Clip GHL to a watershed and reproject/regrid to the mHM L0 grid with mode resampling."""

from __future__ import annotations
import logging

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  registers .rio accessor
import xarray as xr
from affine import Affine
from pyproj import CRS as ProjCRS
from rasterio.enums import Resampling

log = logging.getLogger(__name__)


def _affine_from_header(header: dict) -> Affine:
    """Build a north-up affine transform from an mHM header dict."""
    cs = header["cellsize"]
    xll = header["xllcorner"]
    yll = header["yllcorner"]
    nrows = header["nrows"]
    y_top = yll + nrows * cs
    return Affine(cs, 0.0, xll, 0.0, -cs, y_top)


def clip_and_reproject_to_grid(
    scene: xr.DataArray,
    watershed_path: str,
    target_crs_wkt: str,
    header: dict,
    src_nodata: int,
    l0_cellsize_m: int,
) -> xr.DataArray:
    """
    1. Ensure the source has a CRS attached.
    2. Clip to the buffered watershed bbox in the source CRS.
    3. Reproject to target CRS if CRS differs; regrid to L0 resolution if resolution differs.

    Grid alignment is enforced by passing an explicit affine transform +
    (nrows, ncols) shape, so the output byte-aligns with `header`.
    """
    # (1) Attach CRS if the source didn't carry one over
    if scene.rio.crs is None:
        # GHL is delivered in EPSG:4326 unless stated otherwise; adjust if needed.
        scene = scene.rio.write_crs("EPSG:4326")
        log.info("Source CRS missing; assumed EPSG:4326.")

    scene = scene.rio.write_nodata(src_nodata)

    # (2) Watershed bbox in source CRS, buffered
    ws = gpd.read_file(watershed_path)
    if ws.crs is None:
        raise ValueError(f"Watershed {watershed_path} has no CRS.")
    ws_src = ws.to_crs(scene.rio.crs)
    minx, miny, maxx, maxy = ws_src.total_bounds
    log.info("Clip bbox in source CRS: %s", (minx, miny, maxx, maxy))

    clipped = scene.rio.clip_box(minx, miny, maxx, maxy).compute()
    log.info("Clipped GHL shape (y, x): %s", clipped.shape)

    # (3) Check CRS first; if it matches, resolution() will be in metres already.
    src_crs = clipped.rio.crs
    needs_reproject = not ProjCRS.from_user_input(target_crs_wkt).equals(src_crs)

    if needs_reproject:
        log.info("CRS differs (%s → target); reprojecting to L0 grid.", src_crs)
        needs_regrid = False  # explicit transform+shape handles resolution
    else:
        x_res, _ = clipped.rio.resolution()
        native_res_m = int(round(abs(x_res)))
        needs_regrid = native_res_m != l0_cellsize_m
        if needs_regrid:
            log.info(
                "Resolution differs (%d m → %d m); regriding to L0 grid.",
                native_res_m,
                l0_cellsize_m,
            )

    if needs_reproject or needs_regrid:
        transform = _affine_from_header(header)
        reprojected = clipped.rio.reproject(
            dst_crs=target_crs_wkt,
            transform=transform,
            shape=(header["nrows"], header["ncols"]),
            resampling=Resampling.mode,
            nodata=src_nodata,
        )
    else:
        log.info("CRS and resolution already match L0 target; skipping reproject.")
        reprojected = clipped

    # Some sources keep a singleton non-spatial dim (e.g., band=1).
    # Squeeze those so downstream writers receive a strict (y, x) grid.
    reprojected = reprojected.squeeze(drop=True)
    if set(reprojected.dims) != {"y", "x"}:
        raise ValueError(
            "Expected reprojected land-cover array with dims {'y','x'}, "
            f"got dims={reprojected.dims}."
        )

    # Ensure conventional order expected by ASCII writer and QA NetCDF.
    reprojected = reprojected.transpose("y", "x")
    log.info("Reprojected to L0 grid shape (y, x): %s", reprojected.shape)
    return reprojected


def _buffer_in_src_units(gdf: gpd.GeoDataFrame, buffer_km: float) -> float:
    """
    Convert buffer_km to source CRS units. For geographic CRS (degrees), use a
    conservative degree approximation at the centroid latitude; for projected
    CRS (meters), a direct meters conversion.
    """
    if gdf.crs.is_geographic:
        # 1 degree latitude ~= 111 km; use latitude scaling for longitude
        lat0 = float(gdf.geometry.centroid.y.mean())
        deg_per_km_lat = 1.0 / 111.0
        # buffer in degrees, taking the larger of lat/lon requirement
        return buffer_km * deg_per_km_lat / np.cos(np.deg2rad(lat0))
    return buffer_km * 1_000.0

#!/usr/bin/env python
"""
acquire_modis_lai_planetary.py
==============================

Acquire a single, representative Leaf Area Index (LAI) snapshot for an
arbitrary boundary from the MODIS/Terra+Aqua Combined LAI/FPAR 4-day 500 m
product on the **Microsoft Planetary Computer** (STAC collection
``modis-15A3H-061``), then clip, scale, quality-mask, and optionally export it
to a local GeoTIFF.

This is a Planetary Computer (PC) equivalent of the Earth Engine version. It
follows the PC MODIS 15A3H-061 example workflow: search the STAC API with
``pystac-client``, sign the Azure Blob asset hrefs with
``planetary_computer.sign_inplace``, then load the Cloud-Optimized GeoTIFF
(COG) assets into an ``xarray`` cube with ``odc.stac.load``.

Reference:
    https://planetarycomputer.microsoft.com/dataset/modis-15A3H-061#Example-Notebook

Product notes (MCD15A3H v6.1)
-----------------------------
* Cadence            : 4-day composite
* Native resolution  : 500 m (MODIS sinusoidal grid, ESRI:54008-like)
* Temporal extent    : 2002-07-04 -> present
* Relevant assets (PC STAC keys -> COG GeoTIFF):
    - ``Lai_500m``        : LAI * 10  (apply scale 0.1 -> m2/m2), valid 0-100
    - ``LaiStdDev_500m``  : StdDev * 10 (apply scale 0.1)
    - ``FparLai_QC``      : bit-packed QC. Bits 0-2 = SCF_QC; 0 = best (main RT
                            algorithm, no saturation), 1 = good (main RT with
                            saturation). Values >1 use the empirical back-up
                            algorithm and are typically discarded.

The workflow collapses every valid 4-day composite that falls inside the
user-supplied time window into ONE representative LAI image via a per-pixel
temporal reducer (default: median, robust to residual cloud/QC outliers).

Dependencies
------------
    pystac-client  planetary-computer  odc-stac  xarray  rioxarray
    geopandas  rasterio  (optional: dask for chunked loads)

Authentication
--------------
No account is required to *read* PC data, but asset hrefs must be signed.
Signing is handled automatically via ``planetary_computer.sign_inplace``.
Set a PC subscription key in the ``PC_SDK_SUBSCRIPTION_KEY`` env var to raise
rate limits (optional).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import odc.stac

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("modis_lai_pc")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_ID = "modis-15A3H-061"
LAI_ASSET = "Lai_500m"
LAI_STD_ASSET = "LaiStdDev_500m"
QC_ASSET = "FparLai_QC"
LAI_SCALE = 0.1          # multiply DN by this to get m2/m2
LAI_VALID_MAX = 100      # DN; >100 are fill/water/etc.
NATIVE_SCALE_M = 500     # native pixel size in metres
# MODIS sinusoidal projection (native grid of the COGs).
MODIS_SINU_WKT = (
    'PROJCS["MODIS Sinusoidal",'
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Sinusoidal"],'
    'PARAMETER["false_easting",0.0],'
    'PARAMETER["false_northing",0.0],'
    'PARAMETER["central_meridian",0.0],'
    'PARAMETER["semi_major",6371007.181],'
    'PARAMETER["semi_minor",6371007.181],'
    'UNIT["m",1.0]]'
)


# --------------------------------------------------------------------------- #
# STAC session
# --------------------------------------------------------------------------- #
def open_catalog() -> pystac_client.Client:
    """Open the PC STAC API with automatic Azure Blob asset signing."""
    catalog = pystac_client.Client.open(
        STAC_API,
        modifier=planetary_computer.sign_inplace,
    )
    log.info("Opened Planetary Computer STAC API.")
    return catalog


# --------------------------------------------------------------------------- #
# Geometry handling
# --------------------------------------------------------------------------- #
def dissolve_boundary(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Dissolve a boundary GeoDataFrame into a single-row frame in EPSG:4326.

    The STAC search and the lon/lat clip both expect geographic coordinates,
    so the frame is reprojected if it carries any other CRS. All features are
    unioned so the query/clip use the full extent regardless of row count.
    """
    if gdf.empty:
        raise ValueError("Input GeoDataFrame is empty.")
    if gdf.crs is None:
        raise ValueError(
            "Input GeoDataFrame has no CRS. Set gdf.crs before running "
            "(e.g. gdf.set_crs('EPSG:4326'))."
        )
    gdf_ll = gdf.to_crs("EPSG:4326")
    dissolved = gdf_ll.union_all() if hasattr(gdf_ll, "union_all") else gdf_ll.unary_union
    return gpd.GeoDataFrame(geometry=[dissolved], crs="EPSG:4326")


# --------------------------------------------------------------------------- #
# Search + load
# --------------------------------------------------------------------------- #
def search_items(
    catalog: pystac_client.Client,
    boundary_ll: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
):
    """Search the collection over the boundary + window. Returns an ItemCollection."""
    # STAC datetime ranges are inclusive on both ends; the caller passes an
    # exclusive end_date to mirror the EE version, so nudge it back one day.
    time_range = f"{start_date}/{end_date}"
    aoi_geojson = boundary_ll.geometry.iloc[0].__geo_interface__

    search = catalog.search(
        collections=[COLLECTION_ID],
        intersects=aoi_geojson,
        datetime=time_range,
    )
    items = search.item_collection()
    if len(items) == 0:
        raise RuntimeError(
            f"No {COLLECTION_ID} items between {start_date} and {end_date} over "
            "the supplied boundary. Widen the date range or check the geometry."
        )
    log.info("STAC search returned %d items.", len(items))
    return items


def load_cube(
    items,
    boundary_ll: gpd.GeoDataFrame,
    scale_m: int = NATIVE_SCALE_M,
    output_crs: str = "EPSG:4326",
    chunks: Optional[dict] = None,
):
    """
    Load the Lai / QC assets into an xarray.Dataset with odc.stac.load.

    Loading is clipped to the boundary bbox and reprojected to ``output_crs``
    at ``scale_m`` resolution. ``chunks`` (e.g. {"x": 2048, "y": 2048}) triggers
    a lazy Dask-backed load for larger regions.
    """
    bbox = tuple(boundary_ll.total_bounds)  # (minx, miny, maxx, maxy) in EPSG:4326

    # Resolution units follow the target CRS. For EPSG:4326 convert metres to
    # degrees (~111.32 km per degree); for a projected CRS pass metres directly.
    if output_crs.upper() in ("EPSG:4326", "4326"):
        resolution = scale_m / 111_320.0
    else:
        resolution = scale_m

    ds = odc.stac.load(
        items,
        bands=[LAI_ASSET, LAI_STD_ASSET, QC_ASSET],
        crs=output_crs,
        resolution=resolution,
        bbox=bbox,
        chunks=chunks or {},
        groupby="time",
        resampling="nearest",   # categorical QC must not be interpolated
    )
    log.info(
        "Loaded cube: %d time steps, grid %s.",
        ds.sizes.get("time", 0),
        {k: v for k, v in ds.sizes.items() if k in ("y", "x")},
    )
    return ds


# --------------------------------------------------------------------------- #
# Masking + scaling + temporal reduction
# --------------------------------------------------------------------------- #
def build_lai_snapshot(ds, reducer: str = "median", max_scf_qc: int = 1):
    """
    Collapse the loaded cube into ONE representative LAI snapshot (m2/m2).

    Applies the SCF_QC bitmask (bits 0-2) and the valid-range mask, scales the
    DN by 0.1, then reduces per-pixel across time with ``reducer``. Also returns
    an ``n_obs`` band counting valid observations behind each pixel.

    Parameters
    ----------
    reducer     : "median" (default, robust), "mean", "max", or "min".
    max_scf_qc  : highest acceptable SCF_QC (0 = best only, 1 = best+good).
    """
    import xarray as xr

    lai_dn = ds[LAI_ASSET]
    qc = ds[QC_ASSET]

    scf_qc = qc.astype("uint16") & 0b111          # bits 0-2
    quality_ok = scf_qc <= max_scf_qc
    range_ok = lai_dn <= LAI_VALID_MAX
    mask = quality_ok & range_ok

    lai = (lai_dn.where(mask) * LAI_SCALE).astype("float32")

    reducers = {"mean", "median", "max", "min"}
    if reducer not in reducers:
        raise ValueError(f"reducer must be one of {sorted(reducers)}")

    log.info(
        "Collapsing %d composites into one snapshot (reducer='%s').",
        ds.sizes.get("time", 0), reducer,
    )

    snapshot = getattr(lai, reducer)(dim="time", skipna=True).rename("Lai")
    n_obs = mask.sum(dim="time").rename("n_obs").astype("int16")

    out = xr.merge([snapshot, n_obs])
    out["Lai"].attrs.update(
        long_name="Leaf Area Index (representative snapshot)",
        units="m2/m2",
        source=COLLECTION_ID,
        reducer=reducer,
        scale_factor_applied=LAI_SCALE,
        max_scf_qc=max_scf_qc,
    )
    return out


def clip_to_boundary(snapshot_ds, boundary_ll: gpd.GeoDataFrame):
    """Mask the snapshot to the exact boundary polygon (not just its bbox)."""
    import rioxarray  # noqa: F401  (registers the .rio accessor)

    crs = snapshot_ds["Lai"].rio.crs or snapshot_ds.rio.crs
    geom = boundary_ll.to_crs(crs) if crs else boundary_ll
    clipped = snapshot_ds.rio.clip(
        geom.geometry.values, geom.crs, drop=True, all_touched=True
    )
    return clipped


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_geotiff(snapshot_ds, out_path: str) -> None:
    """Write the single-band Lai snapshot to a local Cloud-Optimized GeoTIFF."""
    import rioxarray  # noqa: F401

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    snapshot_ds["Lai"].rio.to_raster(out_path, driver="COG", compress="deflate")
    log.info("Saved GeoTIFF -> %s", out_path)


# --------------------------------------------------------------------------- #
# Top-level convenience API
# --------------------------------------------------------------------------- #
def acquire_lai_map(
    boundary: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    out_tif: Optional[str] = None,
    reducer: str = "median",
    max_scf_qc: int = 1,
    scale_m: int = NATIVE_SCALE_M,
    output_crs: str = "EPSG:4326",
    chunks: Optional[dict] = None,
):
    """
    End-to-end: GeoDataFrame boundary -> single representative LAI snapshot,
    sourced from the Microsoft Planetary Computer.

    Returns an ``xarray.Dataset`` with a single-band ``Lai`` snapshot (m2/m2)
    plus an ``n_obs`` support band, clipped/scaled/QC-masked and reduced over
    the window. If ``out_tif`` is given, the snapshot is also written locally.
    Import this function to use the workflow inside a notebook or pipeline.
    """
    catalog = open_catalog()
    boundary_ll = dissolve_boundary(boundary)
    items = search_items(catalog, boundary_ll, start_date, end_date)
    cube = load_cube(items, boundary_ll, scale_m=scale_m,
                     output_crs=output_crs, chunks=chunks)
    snapshot = build_lai_snapshot(cube, reducer=reducer, max_scf_qc=max_scf_qc)
    snapshot = clip_to_boundary(snapshot, boundary_ll)

    # Realize lazily-loaded data before writing/summarizing.
    if chunks:
        snapshot = snapshot.compute()

    if out_tif:
        export_geotiff(snapshot, out_tif)
    return snapshot


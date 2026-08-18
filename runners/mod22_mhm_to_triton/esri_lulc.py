"""
Automated acquisition of ESRI / Impact Observatory 10m Land Cover.

Reads a domain GeoJSON, queries Microsoft Planetary Computer's STAC API for
IO 10m LULC, subsets to the domain, reprojects to a target CRS, and writes
a GeoTIFF to a local folder.
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import rioxarray  # noqa: F401 (registers .rio accessor)
import xarray as xr
from pystac_client import Client
from shapely.geometry import box, mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("io_lulc")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# Annual, updated collection (recommended). Alternatives: "io-lulc-9-class"
COLLECTION = "io-lulc-annual-v02"

# Land cover class metadata (io-lulc-annual-v02)
LULC_CLASSES = {
    0: "No Data",
    1: "Water",
    2: "Trees",
    4: "Flooded vegetation",
    5: "Crops",
    7: "Built area",
    8: "Bare ground",
    9: "Snow/ice",
    10: "Clouds",
    11: "Rangeland",
}


def load_domain(geojson_path: Path, target_crs: str):
    """Load domain, return (gdf_in_wgs84, gdf_in_target_crs, bbox_wgs84)."""
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        log.warning("Domain has no CRS; assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    # Dissolve to a single geometry for clipping
    gdf = gdf.dissolve()

    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    gdf_target = gdf.to_crs(target_crs)
    bbox_wgs84 = tuple(gdf_wgs84.total_bounds)  # (minx, miny, maxx, maxy)

    log.info("Domain bbox (WGS84): %s", bbox_wgs84)
    return gdf_wgs84, gdf_target, bbox_wgs84


def search_items(bbox_wgs84, year: str | None):
    """Search STAC for IO LULC items intersecting the bbox."""
    catalog = Client.open(STAC_URL, modifier=pc.sign_inplace)

    search_kwargs = {
        "collections": [COLLECTION],
        "bbox": bbox_wgs84,
    }
    if year:
        # datetime filter — annual mosaics are stamped at year start
        search_kwargs["datetime"] = f"{year}-01-01/{year}-12-31"

    search = catalog.search(**search_kwargs)
    items = list(search.items())
    log.info("Found %d STAC item(s).", len(items))

    if not items:
        raise RuntimeError(
            "No items found. Check bbox, year, and collection name."
        )
    return items


def build_mosaic(items, bbox_wgs84, target_crs, resolution=10):
    """Load and mosaic items with odc.stac, clipped to bbox and reprojected."""
    from odc.stac import stac_load

    minx, miny, maxx, maxy = bbox_wgs84

    ds = stac_load(
        items,
        bands=["data"],
        bbox=bbox_wgs84,
        crs=target_crs,
        resolution=resolution,
        chunks={"x": 2048, "y": 2048},
        dtype="uint8",
        nodata=0,
    )

    # If multiple time slices exist, take the most recent
    if "time" in ds.dims and ds.sizes.get("time", 1) > 1:
        log.info("Multiple time slices found; selecting most recent.")
        ds = ds.isel(time=-1)
    elif "time" in ds.dims:
        ds = ds.squeeze("time", drop=True)

    da = ds["data"]
    da = da.rio.write_crs(target_crs)
    return da


def clip_to_domain(da: xr.DataArray, gdf_target: gpd.GeoDataFrame):
    """Clip raster to the domain polygon (in target CRS)."""
    geom = [mapping(g) for g in gdf_target.geometry]
    clipped = da.rio.clip(geom, gdf_target.crs, drop=True, all_touched=True)
    return clipped


def summarize(da: xr.DataArray):
    """Log class distribution."""
    values, counts = np.unique(da.values, return_counts=True)
    total = counts.sum()
    log.info("Land cover class distribution:")
    for v, c in zip(values, counts):
        name = LULC_CLASSES.get(int(v), f"Class {int(v)}")
        log.info("  %2d %-20s %8d px (%.2f%%)", int(v), name, c, 100 * c / total)


def save_output(da: xr.DataArray, out_path: Path):
    """Write a compressed, tiled Cloud-Optimized-style GeoTIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da = da.astype("uint8")
    da.rio.write_nodata(0, inplace=True)
    da.rio.to_raster(
        out_path,
        driver="GTiff",
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    log.info("Saved: %s", out_path)


def acquire(geojson_path: Path, target_crs: str, out_path: Path,
            year: str | None = None, resolution: float = 10.0) -> Path:
    """Acquire IO 10 m LULC for a domain and write a GeoTIFF to *out_path*.

    Importable entry point used by the mod22 pipeline; returns *out_path*.
    """
    geojson_path, out_path = Path(geojson_path), Path(out_path)
    _, gdf_target, bbox_wgs84 = load_domain(geojson_path, target_crs)
    items = search_items(bbox_wgs84, year)
    da = build_mosaic(items, bbox_wgs84, target_crs, resolution)
    da = clip_to_domain(da, gdf_target)
    log.info("Computing...")
    da = da.compute()
    summarize(da)
    save_output(da, out_path)
    log.info("Done.")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Acquire ESRI/Impact Observatory 10m Land Cover."
    )
    parser.add_argument("geojson", type=Path, help="Domain GeoJSON file.")
    parser.add_argument(
        "--crs", required=True, help="Target CRS, e.g. 'EPSG:32610'."
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("./output"),
        help="Output folder.",
    )
    parser.add_argument(
        "--year", default=None,
        help="Year to acquire (e.g. 2023). Default: latest available.",
    )
    parser.add_argument(
        "--resolution", type=float, default=10.0,
        help="Output resolution in target CRS units (default 10).",
    )
    parser.add_argument(
        "--name", default="io_lulc",
        help="Output filename prefix.",
    )
    args = parser.parse_args()

    year_tag = args.year if args.year else "latest"
    out_path = args.outdir / f"{args.name}_{year_tag}_{args.crs.replace(':', '')}.tif"
    acquire(args.geojson, args.crs, out_path, year=args.year, resolution=args.resolution)


if __name__ == "__main__":
    main()
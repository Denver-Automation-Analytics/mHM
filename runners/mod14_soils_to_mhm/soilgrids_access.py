"""Download SoilGrids bulk density, clay, and sand via OWSLib WCS 1.0.0."""

from __future__ import annotations
import logging
import time
from pathlib import Path

import geopandas as gpd
from owslib.wcs import WebCoverageService

log = logging.getLogger(__name__)

# EPSG:152160 is an informal surrogate for IGH; the WCS accepts it as a label
# but PROJ databases typically do not. Use the PROJ4 definition for local transforms.
_SG_CRS = "urn:ogc:def:crs:EPSG::152160"  # for WCS getCoverage calls
_SG_PROJ4 = "+proj=igh +datum=WGS84 +no_defs"  # for local pyproj/geopandas
_SG_RES = 250  # native SoilGrids resolution (metres)

# Six GlobalSoilMap standard depth intervals, labelled 01-06 to match Fortran
DEPTH_INTERVALS: list[tuple[int, str]] = [
    (1, "0-5cm"),
    (2, "5-15cm"),
    (3, "15-30cm"),
    (4, "30-60cm"),
    (5, "60-100cm"),
    (6, "100-200cm"),
]

# Maps property key -> (WCS map URL, short name used in Fortran filenames)
_PROPERTIES: dict[str, tuple[str, str]] = {
    "bdod": ("http://maps.isric.org/mapserv?map=/map/bdod.map", "bd"),
    "clay": ("http://maps.isric.org/mapserv?map=/map/clay.map", "cl"),
    "sand": ("http://maps.isric.org/mapserv?map=/map/sand.map", "sn"),
}


def _watershed_bbox_homolosine(
    watershed_path: str,
) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in Homolosine (EPSG:152160)."""
    gdf = gpd.read_file(watershed_path)
    if gdf.crs is None:
        raise ValueError(f"Watershed file has no CRS: {watershed_path}")
    gdf_hom = gdf.to_crs(_SG_PROJ4)
    gdf_hom = gdf_hom.copy()
    minx, miny, maxx, maxy = gdf_hom.total_bounds
    log.info(
        "Watershed bbox in Homolosine (m): %.0f %.0f %.0f %.0f",
        minx,
        miny,
        maxx,
        maxy,
    )
    return float(minx), float(miny), float(maxx), float(maxy)


def _find_coverage_id(wcs: WebCoverageService, prop: str, depth: str, stat: str) -> str:
    """
    Resolve the coverage identifier for a given property/depth/stat.
    Tries '{prop}_{depth}_{stat}' first, then falls back to common aliases.
    """
    candidates = [
        f"{prop}_{depth}_{stat}",
        f"{prop}_{depth}_mean",
        f"{prop}_{depth}_median",
    ]
    for cid in candidates:
        if cid in wcs.contents:
            if cid != candidates[0]:
                log.warning("Stat '%s' not found; using '%s' instead.", stat, cid)
            return cid
    available = [k for k in wcs.contents if k.startswith(f"{prop}_{depth}")]
    raise KeyError(
        f"No coverage found for {prop} {depth} (tried {candidates}). "
        f"Available: {available}"
    )


def _download_coverage(
    wcs: WebCoverageService,
    cov_id: str,
    bbox: tuple[float, float, float, float],
    out_path: Path,
    max_retries: int = 4,
    backoff: float = 2.0,
) -> None:
    """Fetch a single WCS coverage and write to disk; skip if already cached."""
    if out_path.exists():
        log.info("Cache hit: %s", out_path.name)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            response = wcs.getCoverage(
                identifier=cov_id,
                crs=_SG_CRS,
                bbox=bbox,
                resx=_SG_RES,
                resy=_SG_RES,
                format="GEOTIFF_INT16",
            )
            out_path.write_bytes(response.read())
            log.info("Downloaded %-30s -> %s", cov_id, out_path.name)
            return
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"WCS download failed for {cov_id} after {max_retries} attempts"
                ) from exc
            wait = backoff**attempt
            log.warning(
                "Attempt %d for %s failed (%s); retrying in %.1f s",
                attempt,
                cov_id,
                exc,
                wait,
            )
            time.sleep(wait)


def download_soilgrids(
    watershed_path: str,
    cache_dir: Path,
    stat: str = "Q0.5",
) -> dict[str, dict[int, Path]]:
    """
    Download 18 SoilGrids GeoTIFFs (3 properties × 6 depths) to cache_dir.

    Returns paths[short_name][layer_number] -> Path, where short_name is one
    of "bd", "cl", "sn" matching the Fortran input file prefix convention.
    """
    bbox = _watershed_bbox_homolosine(watershed_path)
    paths: dict[str, dict[int, Path]] = {}

    for prop, (url, short) in _PROPERTIES.items():
        wcs = WebCoverageService(url, version="1.0.0")
        log.info("Connected to SoilGrids WCS for %s", prop)
        paths[short] = {}

        for layer_num, depth in DEPTH_INTERVALS:
            cov_id = _find_coverage_id(wcs, prop, depth, stat)
            out_file = cache_dir / f"{short}{layer_num:02d}.tif"
            _download_coverage(wcs, cov_id, bbox, out_file)
            paths[short][layer_num] = out_file

    return paths

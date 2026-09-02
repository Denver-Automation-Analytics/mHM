"""
karst_access.py
===============
Automated acquisition of USGS geology data from an ArcGIS Feature Service,
clipped to a user-provided boundary and written to a user-provided local path.

This script iterates over a LIST of ArcGIS feature services -- one per karst /
pseudokarst class -- auto-discovers the layer(s) in each, and downloads every
feature that intersects the user's boundary. Each service URL may be a
FeatureServer *root* (all layers) or a single-layer endpoint (ending in /<id>).

Default dataset
---------------
USGS "Karst in the United States: A Digital Map Compilation and Database"
  Report  : Weary, D.J., and Doctor, D.H., 2014, USGS Open-File Report 2014-1156,
            https://doi.org/10.3133/ofr20141156
  Classes : Carbonate Karst, Evaporite Karst, Sandstone Karst,
            Piping Pseudokarst, Volcanic Pseudokarst, Gypsum Extent,
            Evaporite Basins  (each published as a SEPARATE service)
  Scale   : national/State/regional guidance only -- NOT for site-specific or
            legal delineation (per the USGS use constraints).

Tile layers vs. feature layers (READ THIS)
------------------------------------------
The class URLs usually shared for this dataset are TILED MapServer layers on
tiles.arcgis.com (e.g., .../Carbonate_Karst/MapServer). Those are pre-rendered
RASTER tiles and CANNOT be queried for vector features. This script instead
uses each class's companion queryable FeatureServer on services.arcgis.com.
If you pass a tile URL anyway, ``tile_to_featureserver()`` auto-redirects it.

Two service names differ from their tile names:
    tile "Gypsum_Karst"     -> FeatureServer "Gypsum"
    tile "Evaporite_Basins" -> FeatureServer "evaporitebasins"

Unresolved services are skipped gracefully so one bad endpoint never aborts the
run. If any class is truly only available as a tile layer, get its vector data
from the USGS OFR 2014-1156 downloads directory
(https://pubs.usgs.gov/of/2014/1156/).

Requirements
------------
    pip install geopandas requests shapely pyproj

Notes
-----
* No argparse. Edit the inputs directly at the bottom of ``main()``.
* Services cap responses (max ~1000-2000 records/request); the script paginates
  and honors ``exceededTransferLimit``.
* The server-side filter uses the boundary's bounding envelope (fast, robust);
  an exact clip to the boundary geometry is then applied client-side per layer.
* Output: a GeoPackage (.gpkg) with one layer per source layer is recommended
  because different geometry types (polygon/line/point) cannot share one file
  in single-geometry formats. For non-GPKG paths, each layer is written to a
  separate file with the layer name appended to the stem.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Service configuration
# ---------------------------------------------------------------------------
# The USGS "Karst in the United States" compilation (Open-File Report 2014-1156,
# Weary & Doctor) publishes its karst/pseudokarst classes as SEPARATE hosted
# services rather than one multi-layer FeatureServer. This script therefore
# iterates over a LIST of feature services, auto-discovering the layer(s) in
# each and downloading everything that intersects the boundary.
#
# IMPORTANT -- tile layers vs. feature layers
# -------------------------------------------
# The class URLs commonly shared for this dataset point at TILED MapServer
# layers on tiles.arcgis.com, e.g.:
#   https://tiles.arcgis.com/tiles/hoKRg7d6zCP8hwp2/arcgis/rest/services/
#       Carbonate_Karst/MapServer
# Those are pre-rendered RASTER tiles and CANNOT be queried for vector features.
#
# Each class, however, has a companion *queryable* FeatureServer on
# services.arcgis.com (same org, hoKRg7d6zCP8hwp2), which is what this script
# uses. Note two service names differ from their tile-layer names:
#     tile "Gypsum_Karst"     -> FeatureServer "Gypsum"
#     tile "Evaporite_Basins" -> FeatureServer "evaporitebasins"
#
# Each entry may be a FeatureServer ROOT (all layers) or a single-layer
# endpoint ending in /<id>. Both are handled automatically.
FEATURE_HOST = "https://services.arcgis.com/hoKRg7d6zCP8hwp2/ArcGIS/rest/services"

KARST_SERVICES = [
    # class label            # queryable FeatureServer URL
    ("Carbonate_Karst", f"{FEATURE_HOST}/Carbonate_Karst/FeatureServer"),
    ("Evaporite_Karst", f"{FEATURE_HOST}/Evaporite_Karst/FeatureServer"),
    ("Sandstone_Karst", f"{FEATURE_HOST}/Sandstone_Karst/FeatureServer"),
    ("Piping_Pseudokarst", f"{FEATURE_HOST}/Piping_Pseudokarst/FeatureServer"),
    ("Volcanic_Pseudokarst", f"{FEATURE_HOST}/Volcanic_Pseudokarst/FeatureServer"),
    (
        "Gypsum_Extent",
        f"{FEATURE_HOST}/Gypsum/FeatureServer",
    ),  # tile name: Gypsum_Karst
    (
        "Evaporite_Basins",
        f"{FEATURE_HOST}/evaporitebasins/FeatureServer",
    ),  # tile name: Evaporite_Basins
]

# Known tile-name -> FeatureServer-name overrides for the converter below.
_TILE_NAME_OVERRIDES = {
    "gypsum_karst": "Gypsum",
    "evaporite_basins": "evaporitebasins",
}

QUERY_SR = 4326  # SR of the envelope we send (boundary reprojected to this)
OUTPUT_SR = 4326  # SR requested from the server for returned geometry
FALLBACK_PAGE_SIZE = 1000


def tile_to_featureserver(url: str) -> str:
    """
    Convert a tiled MapServer URL (tiles.arcgis.com/.../<Name>/MapServer) into
    its queryable FeatureServer twin on services.arcgis.com.

    Handles the two known name mismatches (Gypsum_Karst -> Gypsum,
    Evaporite_Basins -> evaporitebasins). If ``url`` is already a
    FeatureServer/MapServer on services.arcgis.com it is returned unchanged
    except for the MapServer->FeatureServer swap.
    """
    m = re.search(r"/services/([^/]+)/(MapServer|FeatureServer)", url)
    if not m:
        return url  # unrecognized shape; hand back as-is
    name, _ = m.group(1), m.group(2)
    fs_name = _TILE_NAME_OVERRIDES.get(name.lower(), name)
    return f"{FEATURE_HOST}/{fs_name}/FeatureServer"


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _build_session(retries: int = 4, backoff: float = 0.6) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "geology-acquisition/1.0"})
    return session


def _get_json(session: requests.Session, url: str, params: dict) -> dict:
    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Service error at {url}: {data['error']}")
    return data


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------
def _discover_layers(session: requests.Session, service_url: str) -> list[dict]:
    """
    Return a list of {id, name, geometryType, maxRecordCount} for every feature
    layer to download.

    * If ``service_url`` ends in a numeric layer id, only that layer is returned.
    * Otherwise the FeatureServer root is read and all layers (not tables) are
      returned. Tables (no geometry) are skipped.
    """
    service_url = service_url.rstrip("/")
    # Single-layer endpoint?
    if re.search(r"/\d+$", service_url):
        meta = _get_json(session, service_url, {"f": "json"})
        return [
            {
                "id": meta.get("id"),
                "url": service_url,
                "name": meta.get("name", "layer"),
                "geometryType": meta.get("geometryType"),
                "maxRecordCount": int(meta.get("maxRecordCount", FALLBACK_PAGE_SIZE))
                or FALLBACK_PAGE_SIZE,
            }
        ]

    # FeatureServer root -> enumerate layers.
    root = _get_json(session, service_url, {"f": "json"})
    layers = []
    for lyr in root.get("layers", []):
        # Skip non-spatial tables (they appear under "tables", but guard anyway).
        gtype = lyr.get("geometryType")
        if gtype is None:
            continue
        lid = lyr["id"]
        layers.append(
            {
                "id": lid,
                "url": f"{service_url}/{lid}",
                "name": lyr.get("name", f"layer_{lid}"),
                "geometryType": gtype,
                # per-layer maxRecordCount may not be in the root; fetch lazily later
                "maxRecordCount": int(lyr.get("maxRecordCount", 0)) or None,
            }
        )
    return layers


# ---------------------------------------------------------------------------
# Geometry / query helpers
# ---------------------------------------------------------------------------
def _boundary_envelope(boundary_gdf: gpd.GeoDataFrame, sr: int) -> dict:
    if boundary_gdf.empty:
        raise ValueError("The provided boundary GeoDataFrame is empty.")
    if boundary_gdf.crs is None:
        raise ValueError(
            "The boundary GeoDataFrame has no CRS. Set one, e.g. "
            "boundary_gdf.set_crs('EPSG:4326')."
        )
    projected = boundary_gdf.to_crs(epsg=sr)
    minx, miny, maxx, maxy = projected.total_bounds
    return {
        "xmin": float(minx),
        "ymin": float(miny),
        "xmax": float(maxx),
        "ymax": float(maxy),
        "spatialReference": {"wkid": sr},
    }


def _query_page(session, layer_url, envelope, query_sr, out_sr, offset, page_size):
    params = {
        "where": "1=1",
        "geometry": json.dumps(envelope),
        "geometryType": "esriGeometryEnvelope",
        "inSR": query_sr,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": out_sr,
        "resultOffset": offset,
        "resultRecordCount": page_size,
        "f": "geojson",
    }
    resp = session.post(layer_url + "/query", data=params, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(
            f"Query error ({layer_url}, offset {offset}): {payload['error']}"
        )
    return payload


def _fetch_layer(session, layer, envelope, query_sr, out_sr) -> gpd.GeoDataFrame:
    """Paginate one layer and return all features intersecting the envelope."""
    layer_url = layer["url"]
    page_size = layer.get("maxRecordCount")
    if not page_size:
        # fetch the layer's true maxRecordCount if we didn't get it at discovery
        try:
            meta = _get_json(session, layer_url, {"f": "json"})
            page_size = int(meta.get("maxRecordCount", FALLBACK_PAGE_SIZE))
        except Exception:
            page_size = FALLBACK_PAGE_SIZE
    page_size = page_size or FALLBACK_PAGE_SIZE

    frames, offset = [], 0
    while True:
        page = _query_page(
            session, layer_url, envelope, query_sr, out_sr, offset, page_size
        )
        features = page.get("features", [])
        if features:
            frames.append(
                gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{out_sr}")
            )
        exceeded = page.get("exceededTransferLimit") or page.get("properties", {}).get(
            "exceededTransferLimit"
        )
        if not exceeded or not features:
            break
        offset += page_size
        time.sleep(0.2)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{out_sr}")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=f"EPSG:{out_sr}")


def _clip_to_boundary(gdf, boundary_gdf) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    boundary = boundary_gdf.to_crs(gdf.crs)
    clipped = gpd.clip(gdf, boundary)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()]
    return clipped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def _safe_layer_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "layer"


def _write_layers(results: dict[str, gpd.GeoDataFrame], output_path) -> list[Path]:
    """
    Write a dict of {layer_name: GeoDataFrame} to disk.

    * .gpkg  -> a single GeoPackage with one layer per entry (recommended).
    * others -> one file per layer, with the layer name appended to the stem.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower()
    written: list[Path] = []

    non_empty = {n: g for n, g in results.items() if not g.empty}
    if not non_empty:
        print("No features intersect the boundary in any layer; nothing written.")
        return written

    if ext == ".gpkg":
        for name, gdf in non_empty.items():
            gdf.to_file(out, layer=_safe_layer_name(name), driver="GPKG")
        written.append(out)
        return written

    driver_map = {
        ".shp": "ESRI Shapefile",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
        ".fgb": "FlatGeobuf",
    }
    for name, gdf in non_empty.items():
        stem = f"{out.stem}_{_safe_layer_name(name)}"
        target = out.with_name(stem + out.suffix)
        if ext == ".parquet":
            gdf.to_parquet(target)
        elif ext in driver_map:
            gdf.to_file(target, driver=driver_map[ext])
        else:
            raise ValueError(
                f"Unsupported output extension '{ext}'. Use one of: "
                ".gpkg (recommended), .shp, .geojson, .parquet, .fgb"
            )
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def preflight_check(services: list[tuple[str, str]] = KARST_SERVICES) -> None:
    """
    Ping each service and report whether it resolves to a queryable layer.
    Run this once to confirm all endpoints before a full acquisition.
    """
    session = _build_session()
    print("Preflight endpoint check:")
    for label, url in services:
        if "tiles.arcgis.com" in url or "/MapServer" in url:
            url = tile_to_featureserver(url)
        try:
            layers = _discover_layers(session, url)
            gtypes = ", ".join(sorted({l["geometryType"] or "?" for l in layers}))
            print(f"  [OK]   {label:<22} {len(layers)} layer(s) [{gtypes}] -> {url}")
        except Exception as exc:
            print(f"  [FAIL] {label:<22} {exc} -> {url}")


def acquire_karst(
    boundary_gdf: gpd.GeoDataFrame,
    output_path,
    services: list[tuple[str, str]] = KARST_SERVICES,
    query_sr: int = QUERY_SR,
    out_sr: int = OUTPUT_SR,
    clip: bool = True,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Download every feature layer from each service in ``services`` that
    intersects ``boundary_gdf`` and write the combined results to
    ``output_path``.

    Parameters
    ----------
    services : list[tuple[str, str]]
        A list of (class_label, feature_service_url) pairs. Each URL may be a
        FeatureServer ROOT (all layers) or a single-layer endpoint (/<id>).

    Returns
    -------
    dict[str, geopandas.GeoDataFrame]
        Mapping of "<class_label>__<layer_name>" -> GeoDataFrame (EPSG:out_sr).
        Services that fail to resolve are skipped with a warning, so one bad
        endpoint never aborts the whole run.
    """
    session = _build_session()
    envelope = _boundary_envelope(boundary_gdf, query_sr)

    results: dict[str, gpd.GeoDataFrame] = {}
    skipped: list[str] = []

    for label, service_url in services:
        print(f"\n=== {label} ===")
        # Self-correct: if a tiled MapServer URL was supplied, redirect to its
        # queryable FeatureServer twin (tile caches cannot be queried).
        if "tiles.arcgis.com" in service_url or "/MapServer" in service_url:
            fixed = tile_to_featureserver(service_url)
            if fixed != service_url:
                print(f"  (redirecting tile layer -> FeatureServer: {fixed})")
                service_url = fixed
        try:
            layers = _discover_layers(session, service_url)
        except Exception as exc:
            print(f"  ! could not resolve service ({exc}) -- skipping.")
            skipped.append(label)
            continue

        if not layers:
            print("  ! no feature layers found -- skipping.")
            skipped.append(label)
            continue

        for layer in layers:
            print(f"  -> layer '{layer['name']}' [{layer['geometryType']}]")
            try:
                gdf = _fetch_layer(session, layer, envelope, query_sr, out_sr)
            except Exception as exc:
                print(f"     ! query failed ({exc}) -- skipping layer.")
                continue
            print(f"     envelope query returned {len(gdf)} feature(s)")
            if clip and not gdf.empty:
                gdf = _clip_to_boundary(gdf, boundary_gdf)
                print(f"     after clip: {len(gdf)} feature(s)")
            # Tag the source class so provenance is preserved in the output.
            if not gdf.empty:
                gdf.insert(0, "karst_class", label)
            key = f"{label}__{layer['name']}"
            results[key] = gdf

    written = _write_layers(results, output_path)
    total = sum(len(g) for g in results.values())
    n_nonempty = sum(1 for g in results.values() if not g.empty)
    print(
        f"\nDone. {total} total feature(s) across {n_nonempty} non-empty "
        f"layer(s) from {len(services) - len(skipped)} service(s)."
    )
    if skipped:
        print(f"Skipped (unresolved) services: {', '.join(skipped)}")
        print(
            "  -> If these are only published as tiled MapServer layers, get "
            "the vector data from https://pubs.usgs.gov/of/2014/1156/"
        )
    for p in written:
        print(f"  wrote -> {p.resolve()}")
    return results


# ---------------------------------------------------------------------------
# Entry point -- edit the inputs below directly (no argparse by design)
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- USER-DEFINED INPUTS -------------------------------------------------
    # 1) Boundary GeoDataFrame. Replace this with your own AOI. Examples:
    WATERSHED_FILE = "/workspace/test_domain_3/mhm_input/domain/watershed.geojson"
    boundary_gdf = gpd.read_file(WATERSHED_FILE)

    # 2) Local output file path. Extension sets the format.
    OUTPUT_FILE = "/workspace/test_domain_3/mhm_input/geology/karst.gpkg"
    output_path = OUTPUT_FILE
    # -------------------------------------------------------------------------

    # Optional: confirm all endpoints resolve before the full run.
    # preflight_check(KARST_SERVICES)

    acquire_karst(
        boundary_gdf=boundary_gdf,
        output_path=output_path,
        services=KARST_SERVICES,
        clip=True,
    )


if __name__ == "__main__":
    main()

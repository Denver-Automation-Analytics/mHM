"""Automated acquisition of NHDPlus HR known waterbodies (ESRI FeatureServer).

Reads a domain GeoJSON, queries an ArcGIS FeatureServer polygon layer (default:
"Waterbodies and Areas") for features intersecting the domain, reprojects to the
target CRS, clips to the domain, and writes a GeoPackage of waterbody polygons.

Importable entry point used by the mod21 pipeline is ``acquire``.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nhd_waterbodies")

FALLBACK_PAGE_SIZE = 1000

# NHD FType (3-digit) -> class name (USGS NHD data dictionary). Used to filter classes.
NHD_FTYPE = {
    336: "CanalDitch", 343: "DamWeir", 361: "Playa", 378: "Ice Mass",
    390: "LakePond", 403: "Inundation Area", 431: "Rapids", 436: "Reservoir",
    455: "Spillway", 460: "StreamRiver", 466: "SwampMarsh", 493: "Estuary",
}

# NHD FCode (5-digit) -> description for LakePond; the stage text drives the LakePond filter.
NHD_FCODE_LAKEPOND = {
    39000: "LakePond",
    39001: "LakePond: Hydrographic Category = Intermittent",
    39004: "LakePond: Hydrographic Category = Perennial",
    39005: "LakePond: Hydrographic Category = Intermittent; Stage = High Water Elevation",
    39006: "LakePond: Hydrographic Category = Intermittent; Stage = Date of Photography",
    39009: "LakePond: Hydrographic Category = Perennial; Stage = High Water Elevation",
    39010: "LakePond: Hydrographic Category = Perennial; Stage = Normal Pool",
    39011: "LakePond: Hydrographic Category = Perennial; Stage = Date of Photography",
    39012: "LakePond: Hydrographic Category = Perennial; Stage = Spillway Elevation",
}
LAKEPOND_FTYPE = 390


def _col(gdf, name: str):
    """Return the column matching *name* case-insensitively, or None."""
    for c in gdf.columns:
        if c.lower() == name:
            return c
    return None


def filter_features(gdf, exclude_ftypes=(), lakepond_stage=None):
    """Drop unwanted NHD classes and keep only normal-pool LakePonds.

    * Any feature whose FType name (or LakePond FCode description) contains one of
      *exclude_ftypes* (case-insensitive) is removed.
    * When *lakepond_stage* is set, LakePond features are kept only if their FCode
      description contains that text (e.g. "Stage = Normal Pool"); other LakePonds
      are removed. Non-LakePond classes are unaffected by this rule.
    Returns the filtered GeoDataFrame.
    """
    if gdf.empty:
        return gdf
    ftc, fcc = _col(gdf, "ftype"), _col(gdf, "fcode")
    if ftc is None:
        log.warning("No 'ftype' column; skipping waterbody class filtering.")
        return gdf
    ftype = gdf[ftc].fillna(-1).astype(int)
    fcode = gdf[fcc].fillna(-1).astype(int) if fcc is not None else ftype * 0 - 1
    fname = ftype.map(NHD_FTYPE).fillna("")
    fdesc = fcode.map(NHD_FCODE_LAKEPOND).fillna("")

    keep = gdf.index.to_series().map(lambda _: True)
    tokens = [t.strip().lower() for t in exclude_ftypes if t.strip()]
    if tokens:
        text = (fname + " " + fdesc).str.lower()
        excl = text.apply(lambda s: any(tok in s for tok in tokens))
        log.info("Excluding %d feature(s) by FType %s", int(excl.sum()), list(exclude_ftypes))
        keep &= ~excl
    if lakepond_stage:
        want = str(lakepond_stage).lower()
        is_lp = ftype == LAKEPOND_FTYPE
        lp_ok = fdesc.str.lower().str.contains(want, regex=False)
        drop_lp = is_lp & ~lp_ok
        log.info("Dropping %d LakePond(s) without %r (kept %d normal-pool)",
                 int(drop_lp.sum()), lakepond_stage, int((is_lp & lp_ok).sum()))
        keep &= ~drop_lp
    return gdf[keep.to_numpy()].reset_index(drop=True)



def _build_session(retries: int = 4, backoff: float = 0.6) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries, connect=retries, read=retries, status=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "mhm-triton-waterbodies/1.0"})
    return session


def _get_json(session: requests.Session, url: str, params: dict) -> dict:
    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Service error at {url}: {data['error']}")
    return data


def _layer_url(service_url: str, layer_id: int) -> str:
    return f"{service_url.rstrip('/')}/{int(layer_id)}"


def _envelope(gdf: gpd.GeoDataFrame, sr: int) -> dict:
    """Bounding box of *gdf* reprojected to EPSG:*sr*, as an ArcGIS envelope."""
    minx, miny, maxx, maxy = gdf.to_crs(epsg=sr).total_bounds
    return {
        "xmin": float(minx), "ymin": float(miny),
        "xmax": float(maxx), "ymax": float(maxy),
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
        raise RuntimeError(f"Query error ({layer_url}, offset {offset}): {payload['error']}")
    return payload


def _fetch_layer(session, layer_url, envelope, query_sr, out_sr, page_size) -> gpd.GeoDataFrame:
    """Paginate the layer and return every feature intersecting the envelope."""
    frames, offset = [], 0
    while True:
        page = _query_page(session, layer_url, envelope, query_sr, out_sr, offset, page_size)
        features = page.get("features", [])
        if features:
            frames.append(gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{out_sr}"))
        exceeded = page.get("exceededTransferLimit") or \
            page.get("properties", {}).get("exceededTransferLimit")
        log.info("  fetched %d feature(s) at offset %d%s",
                 len(features), offset, " (more pending)" if exceeded else "")
        if not exceeded or not features:
            break
        offset += page_size
        time.sleep(0.2)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{out_sr}")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=f"EPSG:{out_sr}")


def acquire(geojson_path: Path, target_crs: str, out_path: Path, service_url: str,
            layer_id: int, query_sr: int = 4326, exclude_ftypes=(),
            lakepond_stage=None) -> Path:
    """Acquire waterbody polygons for a domain and write them to *out_path* (GPKG).

    Features are requested in *target_crs* (via outSR), clipped to the domain
    polygon, filtered by NHD class (see :func:`filter_features`), and saved.
    Returns *out_path*.
    """
    geojson_path, out_path = Path(geojson_path), Path(out_path)
    out_sr = int(str(target_crs).split(":")[-1])

    dom = gpd.read_file(geojson_path)
    if dom.crs is None:
        log.warning("Domain has no CRS; assuming EPSG:4326.")
        dom = dom.set_crs("EPSG:4326")
    dom = dom.dissolve()

    session = _build_session()
    layer_url = _layer_url(service_url, layer_id)
    try:
        meta = _get_json(session, layer_url, {"f": "json"})
        page_size = int(meta.get("maxRecordCount", FALLBACK_PAGE_SIZE)) or FALLBACK_PAGE_SIZE
        log.info("Layer %s: %r (maxRecordCount=%d)", layer_id, meta.get("name", "?"), page_size)
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort
        log.warning("Could not read layer metadata (%s); using page size %d.", exc, FALLBACK_PAGE_SIZE)
        page_size = FALLBACK_PAGE_SIZE

    envelope = _envelope(dom, query_sr)
    log.info("Querying %s for waterbodies intersecting the domain bbox ...", layer_url)
    gdf = _fetch_layer(session, layer_url, envelope, query_sr, out_sr, page_size)
    log.info("Retrieved %d waterbody feature(s) before clipping.", len(gdf))

    if not gdf.empty:
        gdf = gpd.clip(gdf, dom.to_crs(gdf.crs))
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
    log.info("Kept %d waterbody feature(s) inside the domain.", len(gdf))

    if not gdf.empty and (exclude_ftypes or lakepond_stage):
        gdf = filter_features(gdf, exclude_ftypes, lakepond_stage)
        log.info("Kept %d waterbody feature(s) after class filtering.", len(gdf))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        # write an empty polygon layer so downstream steps have a valid source
        gdf = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{out_sr}")
    gdf.to_file(out_path, driver="GPKG")
    log.info("Saved: %s", out_path)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Acquire NHDPlus HR waterbodies for a domain.")
    p.add_argument("geojson", type=Path, help="Domain GeoJSON file.")
    p.add_argument("--crs", required=True, help="Target CRS, e.g. 'EPSG:5070'.")
    p.add_argument("--service-url", required=True, help="ESRI FeatureServer root URL.")
    p.add_argument("--layer-id", type=int, default=1, help="Waterbody polygon layer id.")
    p.add_argument("--out", type=Path, default=Path("./waterbodies.gpkg"), help="Output GeoPackage.")
    p.add_argument("--exclude-ftypes", nargs="*", default=["SwampMarsh", "Wetland"],
                   help="NHD FType names to drop.")
    p.add_argument("--lakepond-stage", default="Stage = Normal Pool",
                   help="Keep LakePonds only if their FCode description contains this text ('' disables).")
    args = p.parse_args()
    acquire(args.geojson, args.crs, args.out, args.service_url, args.layer_id,
            exclude_ftypes=args.exclude_ftypes, lakepond_stage=args.lakepond_stage or None)


if __name__ == "__main__":
    main()

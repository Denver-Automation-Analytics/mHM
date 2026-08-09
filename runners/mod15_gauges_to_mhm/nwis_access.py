"""USGS NWIS access helpers (simplified, test.py-style implementation)."""

from __future__ import annotations

import logging
from typing import Iterable

import geopandas as gpd
import pandas as pd
from dataretrieval import waterdata

log = logging.getLogger(__name__)

_ALLOWED_DATA_TYPES = {"daily", "continuous"}


def _normalize_data_type(data_type: str | None = None, cadence: str | None = None) -> str:
    """Normalize user input to one of {'daily', 'continuous'}.

    Backward compatibility:
    - cadence='hourly' is mapped to data_type='continuous'.
    """
    value = (data_type or cadence or "daily").strip().lower()
    if value == "hourly":
        value = "continuous"
    if value not in _ALLOWED_DATA_TYPES:
        raise ValueError("data_type must be either 'daily' or 'continuous'.")
    return value


def _watershed_wgs84(watershed_path: str) -> gpd.GeoDataFrame:
    ws = gpd.read_file(watershed_path)
    if ws.crs is None:
        raise ValueError(f"Watershed {watershed_path} has no CRS.")
    return ws.to_crs("EPSG:4326")


def discover_gauges(
    data_type: str = "daily",
    watershed_path: str = "",
    site_type_code: str = "ST",
    parameter_code: str = "00060",
) -> pd.DataFrame:
    """Return gauges strictly inside watershed polygon.

    Parameters
    ----------
    data_type : {'daily', 'continuous'}
        Primary user selector. Included for API symmetry with time-series fetch.
    """
    _ = parameter_code  # reserved for future filtering; API currently ignores it here
    _normalize_data_type(data_type=data_type)
    if not watershed_path:
        raise ValueError("watershed_path is required.")

    watershed = _watershed_wgs84(watershed_path)
    bounds = watershed.total_bounds.tolist()  # [min_lon, min_lat, max_lon, max_lat]

    sites_gdf, _ = waterdata.get_monitoring_locations(
        bbox=bounds,
        site_type_code=site_type_code,
    )
    if sites_gdf is None or sites_gdf.empty:
        return pd.DataFrame(columns=["site_no", "monitoring_location_name", "dec_lat_va", "dec_long_va"])

    # test.py-style geometry -> lon/lat extraction
    sites_gdf["lat"] = sites_gdf["geometry"].y
    sites_gdf["lon"] = sites_gdf["geometry"].x

    points = gpd.GeoDataFrame(
        sites_gdf,
        geometry=gpd.points_from_xy(sites_gdf["lon"], sites_gdf["lat"]),
        crs="EPSG:4326",
    )
    inside = gpd.sjoin(points, watershed, how="inner", predicate="within")

    return _normalize_columns(inside).reset_index(drop=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to legacy columns expected by downstream code."""
    out = df.copy()

    if "site_no" not in out.columns and "monitoring_location_id" in out.columns:
        out["site_no"] = out["monitoring_location_id"].astype(str).str.replace(r"^USGS-", "", regex=True)
    elif "site_no" in out.columns:
        out["site_no"] = out["site_no"].astype(str).str.replace(r"^USGS-", "", regex=True)

    if "monitoring_location_name" not in out.columns and "name" in out.columns:
        out["monitoring_location_name"] = out["name"]

    if "dec_lat_va" not in out.columns:
        if "lat" in out.columns:
            out["dec_lat_va"] = out["lat"]
        elif "geometry" in out.columns:
            out["dec_lat_va"] = out["geometry"].y

    if "dec_long_va" not in out.columns:
        if "lon" in out.columns:
            out["dec_long_va"] = out["lon"]
        elif "geometry" in out.columns:
            out["dec_long_va"] = out["geometry"].x

    keep = ["site_no", "monitoring_location_name", "dec_lat_va", "dec_long_va"]
    return out[keep].dropna(subset=["site_no", "dec_lat_va", "dec_long_va"])


def filter_inside_watershed(gauges: pd.DataFrame, watershed_path: str) -> pd.DataFrame:
    """Retained for compatibility; applies strict inside filter if needed."""
    if gauges.empty:
        return gauges

    ws = _watershed_wgs84(watershed_path)
    ws_union = ws.geometry.unary_union

    pts = gpd.GeoDataFrame(
        gauges,
        geometry=gpd.points_from_xy(gauges["dec_long_va"], gauges["dec_lat_va"]),
        crs="EPSG:4326",
    )
    mask = pts.geometry.within(ws_union)
    return pts.loc[mask].drop(columns="geometry").reset_index(drop=True)


def fetch_series(
    site_no: str,
    data_type: str = "daily",
    parameter_code: str = "00060",
    start_date: str | None = None,
    end_date: str | None = None,
    cadence: str | None = None,
    **_: object,
) -> pd.DataFrame:
    """Fetch one-site USGS series with daily/continuous selector.

    `data_type` is the primary parameter. `cadence` is accepted for backward
    compatibility with existing callers.
    """
    dtype = _normalize_data_type(data_type=data_type, cadence=cadence)
    monitoring_id = site_no if str(site_no).startswith("USGS-") else f"USGS-{site_no}"

    time_arg = None
    if start_date and end_date:
        time_arg = f"{start_date}/{end_date}"

    if dtype == "daily":
        df, _meta = waterdata.get_daily(
            monitoring_location_id=monitoring_id,
            parameter_code=parameter_code,
            time=time_arg,
        )
        return _standardize_series(df, data_type="daily")

    df, _meta = waterdata.get_continuous(
        monitoring_location_id=monitoring_id,
        parameter_code=parameter_code,
        time=time_arg,
    )
    return _standardize_series(df, data_type="continuous")


def fetch_watershed_data(
    data_type: str = "daily",
    watershed_path: str = "",
    parameter_code: str = "00060",
    site_type_code: str = "ST",
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """High-level helper: discover sites in watershed and fetch one dataframe."""
    dtype = _normalize_data_type(data_type=data_type)
    if not watershed_path:
        raise ValueError("watershed_path is required.")
    gauges = discover_gauges(
        watershed_path=watershed_path,
        data_type=dtype,
        site_type_code=site_type_code,
        parameter_code=parameter_code,
    )
    if gauges.empty:
        return gauges, pd.DataFrame()

    monitoring_ids: Iterable[str] = [f"USGS-{sid}" for sid in gauges["site_no"].astype(str)]
    time_arg = f"{start}/{end}" if start and end else None

    if dtype == "daily":
        df, _meta = waterdata.get_daily(
            monitoring_location_id=list(monitoring_ids),
            parameter_code=parameter_code,
            time=time_arg,
        )
    else:
        df, _meta = waterdata.get_continuous(
            monitoring_location_id=list(monitoring_ids),
            parameter_code=parameter_code,
            time=time_arg,
        )

    return gauges, df if df is not None else pd.DataFrame()


def _standardize_series(df: pd.DataFrame | None, data_type: str) -> pd.DataFrame:
    """Normalize to timestamp index with 'value' and 'approval_status'."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["value", "approval_status"])

    ts_col = next((c for c in ("time", "datetime", "dateTime") if c in df.columns), None)
    if ts_col is None:
        return pd.DataFrame(columns=["value", "approval_status"])

    val_col = next((c for c in ("value", "00060_Mean", "00060") if c in df.columns), None)
    if val_col is None:
        return pd.DataFrame(columns=["value", "approval_status"])

    qc_col = next((c for c in ("approval_status", "qualifiers", "00060_Mean_cd") if c in df.columns), None)

    out = pd.DataFrame(
        {
            "value": pd.to_numeric(df[val_col], errors="coerce"),
            "approval_status": df[qc_col].astype(str) if qc_col else "",
        },
        index=pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
    ).dropna(axis=0, subset=["value"])

    # Keep behavior compatible with previous pipeline contract.
    if data_type == "continuous":
        out = out.resample("1h").agg({
            "value": "mean",
            "approval_status": lambda s: _dominant_status(s),
        })
    else:
        out = out.resample("1D").agg({
            "value": "mean",
            "approval_status": lambda s: _dominant_status(s),
        })
    return out


def _dominant_status(series: pd.Series) -> str:
    vals = set(v for v in series.astype(str) if v and v != "nan")
    if not vals:
        return ""
    if any("Approved" in v for v in vals):
        return "Approved"
    if any("Provisional" in v for v in vals):
        return "Provisional"
    return next(iter(vals))

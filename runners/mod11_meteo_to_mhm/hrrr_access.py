"""Access utilities for the NOAA HRRR 48-hour forecast via Arraylake.

Replaces the previous `dynamical_catalog.open(...)` path with the Arraylake
subscription-mirror pattern used elsewhere in this project.
"""

from __future__ import annotations
import logging

import pandas as pd
import xarray as xr
from arraylake import Client
import zarr

log = logging.getLogger(__name__)


VARS = [
    "temperature_2m",
    "precipitation_surface",
    "wind_u_10m",
    "wind_v_10m",
    "relative_humidity_2m",
    "downward_long_wave_radiation_flux_surface",
    "downward_short_wave_radiation_flux_surface",
]
VALID_INIT_HOURS = (0, 6, 12, 18)   # only these carry the full 48-h forecast


def _has_datetime_coord(ds: xr.Dataset, name: str) -> bool:
    """Return True if coord exists and can be parsed as datetimes."""
    if name not in ds.coords:
        return False
    try:
        parsed = pd.to_datetime(ds[name].values, errors="coerce")
    except Exception:
        return False
    return not pd.isna(parsed).all()


def _find_init_coord(ds: xr.Dataset) -> str:
    """Find the coordinate representing initialization time."""
    preferred = (
        "init_time",
        "forecast_reference_time",
        "reference_time",
        "run_time",
        "time",
    )
    for name in preferred:
        if _has_datetime_coord(ds, name):
            return name

    datetime_coords = [name for name in ds.coords if _has_datetime_coord(ds, name)]
    if len(datetime_coords) == 1:
        return datetime_coords[0]

    raise KeyError(
        "Could not determine initialization-time coordinate. "
        f"Coordinates: {list(ds.coords)}"
    )


def _to_utc_naive(values) -> pd.DatetimeIndex:
    """Convert datetime-like values to tz-naive UTC DatetimeIndex."""
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def open_hrrr(repo_name: str, ref: str = "main") -> xr.Dataset:
    """
    Open the HRRR forecast Icechunk repo (subscription mirror) via Arraylake.

    Access pattern (matches the recommendation returned when subscribing):

        client  = Client()
        repo    = client.get_repo(repo_name)
        session = repo.readonly_session(branch=ref)
        root    = zarr.open_group(session.store, zarr_format=3, mode="r")
        ds      = xr.open_zarr(session.store, zarr_format=3, consolidated=False)

    Authentication is handled by the Arraylake client's built-in auth cache
    (`arraylake auth login`). No token is passed here.

    Parameters
    ----------
    repo_name : str
        Fully-qualified Arraylake repo, e.g.
        "mbi-daa/noaa-hrrr-forecast-48-hour-subscription".
        Typically read from the HRRR_REPO env var in main.py.
    ref : str, optional
        Branch, tag, or snapshot id. Default "main". Pin a snapshot id for
        reproducible production runs.
    """

    client  = Client()
    repo    = client.get_repo(repo_name)
    session = repo.readonly_session(branch=ref)
    store   = session.store

    # Probe with zarr first so we surface any v3 metadata issues clearly and
    # can log the group hierarchy before xarray reads it.
    root = zarr.open_group(store, zarr_format=3, mode="r")
    log.info("Opened %s @ %s; top-level arrays: %s",
             repo_name, ref, list(root.array_keys()))
    log.info("Top-level groups: %s", list(root.group_keys()))

    ds = xr.open_zarr(store, zarr_format=3, consolidated=False)
    log.info("Dataset dims: %s | vars: %s",
             dict(ds.sizes), list(ds.data_vars))

    # Restrict to the meteo variables we actually need
    missing = [v for v in VARS if v not in ds.data_vars]
    if missing:
        raise KeyError(
            f"HRRR subscription is missing expected variables: {missing}. "
            f"Available: {list(ds.data_vars)}"
        )
    return ds[VARS]


def resolve_init_time(ds: xr.Dataset, requested: str) -> pd.Timestamp:
    """Resolve requested start time using either init_time or time-style coords."""
    init_coord = _find_init_coord(ds)
    available = _to_utc_naive(ds[init_coord].values).sort_values().unique()

    if requested != "latest":
        ts = pd.Timestamp(requested)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if ts not in available:
            raise ValueError(
                f"Requested {init_coord} {ts} not in dataset. "
                f"Available range: {available.min()} .. {available.max()}"
            )
        return ts

    # Forecast-style data: prefer synoptic 6-hour init cycles.
    if "lead_time" in ds.coords or "lead_time" in ds.dims:
        now = pd.Timestamp.utcnow().tz_localize(None)
        candidate = now.floor("6h")   # snap to 6-hour cadence
        # Walk back until we find one the dataset actually has (handles ingest lag).
        for _ in range(8):            # up to 48 h back
            if candidate in available and candidate.hour in VALID_INIT_HOURS:
                return candidate
            candidate -= pd.Timedelta("6h")
        raise RuntimeError("Could not find a recent full-length HRRR init_time.")

    # Analysis/reanalysis-style data: use latest available timestamp directly.
    return pd.Timestamp(available.max())


def select_window(ds: xr.Dataset,
                  init_time: pd.Timestamp,
                  forecast_hours: int) -> xr.Dataset:
    """Slice one init/time stamp and optionally limit lead_time to 0..N hours."""
    init_coord = _find_init_coord(ds)
    ds_init = ds.sel({init_coord: init_time})

    # If there is no lead_time axis, this is already a ready-to-format window.
    if "lead_time" not in ds_init.coords and "lead_time" not in ds_init.dims:
        return ds_init

    # HRRR lead_time can be stored either as numeric seconds or as timedeltas
    # depending on backend/version. Support both representations.
    lead_index = ds_init.indexes.get("lead_time")
    # Forecast hours must be 1..48 for the HRRR Forecast subscription
    if 1 <= forecast_hours <= 48:

        if isinstance(lead_index, pd.TimedeltaIndex):
            lead_start = pd.to_timedelta(0, unit="s")
            lead_stop = pd.to_timedelta(forecast_hours, unit="h")
        else:
            lead_start = 0
            lead_stop = forecast_hours * 3600

        return ds_init.sel(lead_time=slice(lead_start, lead_stop))
    else:
        # Historical Reanalysis from 2000 to 2024
        return ds_init
    
"""Access utilities for the dynamical.org NOAA HRRR 48-hour forecast Zarr."""

from __future__ import annotations
import logging
import pandas as pd
import xarray as xr
import dynamical_catalog

log = logging.getLogger(__name__)

VARS = ["temperature_2m", "precipitation_surface"]
VALID_INIT_HOURS = (0, 6, 12, 18)   # only these carry the full 48-h forecast


def open_hrrr() -> xr.Dataset:
    """Open the HRRR forecast dataset with lazy chunks."""
    ds = dynamical_catalog.open("noaa-hrrr-forecast-48-hour", chunks=None)
    return ds[VARS]


def resolve_init_time(ds: xr.Dataset, requested: str) -> pd.Timestamp:
    """Resolve 'latest' to the most recent available 00/06/12/18Z init_time."""
    available = pd.DatetimeIndex(ds.init_time.values)

    if requested != "latest":
        ts = pd.Timestamp(requested)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if ts not in available:
            raise ValueError(f"Requested init_time {ts} not in dataset.")
        return ts

    now = pd.Timestamp.utcnow().tz_localize(None)
    # Floor to nearest 6-hour boundary
    candidate = now.floor("6h")
    # Walk back until we find one the dataset actually has (handles ingest lag)
    for _ in range(8):  # up to 48 h back
        if candidate in available and candidate.hour in VALID_INIT_HOURS:
            return candidate
        candidate -= pd.Timedelta("6h")
    raise RuntimeError("Could not find a recent full-length HRRR init_time.")


def select_window(ds: xr.Dataset,
                  init_time: pd.Timestamp,
                  forecast_hours: int) -> xr.Dataset:
    """Slice one init_time and the requested lead_time range (0..N hours)."""
    if not 1 <= forecast_hours <= 48:
        raise ValueError("FORECAST_LENGTH must be between 1 and 48 hours.")
    lead_seconds = forecast_hours * 3600
    return (
        ds.sel(init_time=init_time)
          .sel(lead_time=slice(0, lead_seconds))
    )
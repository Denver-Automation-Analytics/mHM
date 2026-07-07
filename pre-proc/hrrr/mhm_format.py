"""Transform the clipped HRRR subset into an mHM-compliant xarray Dataset."""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def format_for_mhm(ds_clip: xr.Dataset,
                   init_time: pd.Timestamp,
                   nodata: float) -> tuple[xr.Dataset, pd.Timestamp]:
    """
    Rename variables to mHM's hard-coded names, ensure DOUBLE, build a single
    time dimension (init_time + lead_time), and set fill values / attrs.
    Returns (ds_mhm, reference_time).
    """
    # Rename to mHM's required variable names
    ds = ds_clip.rename({
        "temperature_2m":        "tavg",
        "precipitation_surface": "pre",
    })

    # Build a proper 1-D time coordinate from init_time + lead_time
    valid_time = pd.to_datetime(init_time) + pd.to_timedelta(
        ds["lead_time"].values, unit="s"
    )
    ds = ds.assign_coords(time=("lead_time", valid_time))
    ds = ds.swap_dims({"lead_time": "time"}).drop_vars("lead_time", errors="ignore")

    # Cast data variables to DOUBLE and fill NaN with nodata
    for v in ("pre", "tavg"):
        ds[v] = ds[v].astype("float64").fillna(nodata)

    # CF-style attributes
    ds["pre"].attrs.update({
        "units":         "mm",
        "long_name":     "precipitation",
        "standard_name": "precipitation_amount",
        "missing_value": nodata,
    })
    ds["tavg"].attrs.update({
        "units":         "degC",
        "long_name":     "average air temperature",
        "standard_name": "air_temperature",
        "missing_value": nodata,
    })

    ref_time = pd.Timestamp(valid_time[0])
    return ds, ref_time
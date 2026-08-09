"""Shared utilities for the mod15 PET preprocessing module."""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def detect_time_freq(ds: xr.Dataset) -> tuple[str, pd.Timestamp]:
    """Return ('daily'|'hourly', ref_time) inferred from the time coordinate."""
    times = pd.to_datetime(ds["time"].values)
    if len(times) < 2:
        raise ValueError("Dataset has fewer than 2 time steps.")
    median_hours = (
        pd.Series(np.diff(times.values)).median() / np.timedelta64(1, "h")
    )
    if 20.0 <= median_hours <= 28.0:
        stat_freq = "daily"
    elif 0.9 <= median_hours <= 1.1:
        stat_freq = "hourly"
    else:
        raise ValueError(
            f"Unsupported time step: {median_hours:.1f} h (expected ~1 h or ~24 h)."
        )
    return stat_freq, times[0]


def load_header(header_path: Path) -> dict:
    """Parse an mHM header.txt into a typed dict."""
    parsed: dict[str, str] = {}
    for line in Path(header_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        key, val = line.split(maxsplit=1)
        parsed[key] = val
    return {
        "ncols":        int(parsed["ncols"]),
        "nrows":        int(parsed["nrows"]),
        "xllcorner":    float(parsed["xllcorner"]),
        "yllcorner":    float(parsed["yllcorner"]),
        "cellsize":     float(parsed["cellsize"]),
        "NODATA_value": float(parsed["NODATA_value"]),
    }

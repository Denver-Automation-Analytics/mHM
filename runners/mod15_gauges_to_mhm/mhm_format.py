"""Unit conversion, QC filtering, and mHM gauge-file writing."""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FT3_TO_M3 = 0.028316846592  # conversion factor for ft^3/s -> m^3/s


def to_m3s(df: pd.DataFrame, nodata: int) -> pd.DataFrame:
    """Convert `value` column from ft^3/s to m^3/s; preserve NoData rows."""
    out = df.copy()
    mask = out["value"].notna() & (out["value"] != nodata)
    out.loc[mask, "value"] = out.loc[mask, "value"] * FT3_TO_M3
    out.loc[~mask, "value"] = nodata
    return out


def filter_by_qualifiers(df: pd.DataFrame, policy: str, nodata: int) -> pd.DataFrame:
    """
    Apply the qualifier policy. Rows failing the policy have their value set to
    `nodata` so the time index stays regular (required by mHM).
    """
    out = df.copy()
    status = out["approval_status"].astype(str)

    if policy == "keep_all":
        return out

    if policy == "keep_approved_only":
        keep = status.str.contains("Approved", case=False, na=False)
    elif policy == "keep_approved_provisional":
        keep = status.str.contains("Approved|Provisional", case=False, na=False)
    else:
        raise ValueError(f"Unknown QUALIFIER_POLICY: {policy}")

    dropped = int((~keep).sum())
    if dropped:
        log.info("Qualifier policy '%s' dropped %d row(s)", policy, dropped)
    out.loc[~keep, "value"] = nodata
    return out


def write_gauge_file(
    path: Path,
    local_id: int,
    site_no: str,
    name: str,
    series: pd.DataFrame,
    cadence: str,
    nodata: int,
) -> None:
    """
    Write an mHM gauge file:

        <local_id>: <name>/USGS-<site_no>
        nodata  <nodata>
        n_measurements_per_day  <1 for daily, 24 for hourly>
        start  <YYYY MM DD HH MM> (YYYY MM DD HH MM)
        end    <YYYY MM DD HH MM> (YYYY MM DD HH MM)
        <YYYY MM DD HH MM  value>
        ...
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    n_per_day = 24 if cadence == "hourly" else 1

    # mHM derives the expected row count from whole days in the header, so the
    # series must span midnight-to-midnight even when the source data does not.
    freq = "1h" if cadence == "hourly" else "1D"
    start_day = series.index.min().normalize()
    end_day = series.index.max().normalize()
    end = end_day + pd.Timedelta(hours=23) if cadence == "hourly" else end_day
    idx = pd.date_range(start_day, end, freq=freq, tz="UTC")

    # Fill nodata and padded gaps with time-interpolated values; edges use the
    # nearest observation since interpolation cannot extrapolate beyond them.
    filled = series["value"].reindex(idx)
    filled = filled.mask(filled == nodata)
    filled = filled.interpolate(method="time", limit_direction="both")
    filled = filled.ffill().bfill().fillna(nodata)

    start = filled.index.min()
    end = filled.index.max()

    header_lines = [
        f"{local_id}: {name} / USGS-{site_no}",
        f"nodata  {nodata}",
        f"n_measurements_per_day  {n_per_day}",
        f"start  {start.strftime('%Y %m %d %H %M')} (YYYY MM DD HH MM)",
        f"end    {end.strftime('%Y %m %d %H %M')} (YYYY MM DD HH MM)",
    ]

    with open(path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        for ts, val in filled.items():
            val_str = f"{val:12.4f}" if val != nodata else f"{int(nodata):12d}"
            f.write(f"{ts.strftime('%Y %m %d %H %M')} {val_str}\n")

    log.info("Wrote %s (%d records)", path, len(filled))

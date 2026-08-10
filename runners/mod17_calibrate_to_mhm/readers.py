"""Metadata extraction from mod10–16 outputs used to populate mhm.nml."""

from __future__ import annotations

import csv
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import xarray as xr


def read_meteo_dates(pre_nc: Path) -> Tuple[date, date]:
    """Return (first_date, last_date) from the pre.nc time coordinate."""
    with xr.open_dataset(pre_nc, decode_times=True) as ds:
        import pandas as pd
        t = pd.DatetimeIndex(ds["time"].values)
    return t[0].date(), t[-1].date()


def read_lcover_scenes(luse_dir: Path) -> List[Tuple[int, str]]:
    """Return sorted [(year, filename)] by scanning lc_YYYY.asc files."""
    pattern = re.compile(r"^lc_(\d{4})\.asc$")
    scenes = [
        (int(m.group(1)), f.name)
        for f in sorted(luse_dir.glob("lc_*.asc"))
        if (m := pattern.match(f.name))
    ]
    if not scenes:
        raise FileNotFoundError(f"No lc_*.asc files found in {luse_dir}")
    return sorted(scenes, key=lambda x: x[0])


def read_gauge_info(id_map_csv: Path) -> List[Dict]:
    """Return [{"local_id": int, "filename": str}] rows from id_map.csv."""
    with open(id_map_csv) as f:
        rows = list(csv.DictReader(f))
    gauges = [
        {"local_id": int(r["local_id"]), "filename": f"{int(r['local_id'])}.txt"}
        for r in rows
    ]
    return sorted(gauges, key=lambda g: g["local_id"])


def read_l1_resolution(soil_class_asc: Path) -> int:
    """Return L1 cellsize in metres from soil_class.asc ASCII header."""
    with open(soil_class_asc) as f:
        for line in f:
            key, _, val = line.strip().partition(" ")
            if key.lower() == "cellsize":
                return int(float(val.strip()))
    raise ValueError(f"cellsize not found in {soil_class_asc}")


def read_soil_info(soil_classdefinition_txt: Path, default_horizons: int = 2) -> Tuple[int, List[int]]:
    """
    Return (n_horizons, [soil_Depth(1)..soil_Depth(n-1)]) from LUT.

    soil_Depth values are the LD[mm] column for each horizon except the last,
    whose depth mHM reads from the LUT directly (iFlag_soilDB=0 convention).
    """
    with open(soil_classdefinition_txt) as f:
        lines = f.readlines()

    # rows start after the two header lines
    first_type: int | None = None
    max_horizon = 0
    ld_by_horizon: Dict[int, int] = {}

    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            soil_type = int(parts[0])
            horizon   = int(parts[1])
            ld        = int(parts[3])
        except ValueError:
            continue
        max_horizon = max(max_horizon, horizon)
        if first_type is None:
            first_type = soil_type
        if soil_type == first_type:
            ld_by_horizon[horizon] = ld

    n = max_horizon if max_horizon > 0 else default_horizons
    # depths for layers 1..n-1 (last layer depth comes from LUT per iFlag_soilDB=0)
    depths = [ld_by_horizon[i] for i in range(1, n) if i in ld_by_horizon]
    if len(depths) < n - 1:
        # fallback: evenly spaced at 200mm increments
        depths = [200 * i for i in range(1, n)]
    return n, depths


def derive_eval_period(
    first_meteo: date,
    last_meteo: date,
    warming_days: int,
) -> Tuple[date, date]:
    """
    Return (eval_start, eval_end) accounting for warming days.

    Raises ValueError if the available meteo period is too short.
    """
    eval_start = first_meteo + timedelta(days=warming_days)
    eval_end   = last_meteo
    if eval_start > eval_end:
        total = (last_meteo - first_meteo).days
        raise ValueError(
            f"warming_days={warming_days} exceeds available meteo period "
            f"({total} days: {first_meteo} – {last_meteo}). "
            f"Reduce WARMING_DAYS to at most {max(0, total - 1)}."
        )
    return eval_start, eval_end

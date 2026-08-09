"""Read latitude from an mHM-style latlon.nc."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import xarray as xr


def read_latitude(latlon_nc_path: Path) -> np.ndarray:
    """Return the 2-D latitude array (degrees) matching the L1 meteo grid.

    latlon.nc produced by mod11 stores ``lat(yc, xc)``; row 0 is northernmost.
    """
    with xr.open_dataset(latlon_nc_path) as ds:
        # mod11 latlon_grid.py writes the L1 field as 'lat'
        if "lat" in ds:
            return ds["lat"].values.copy()
        # fallback: any variable with standard_name="latitude"
        for name in list(ds.data_vars) + list(ds.coords):
            var = ds[name]
            if str(var.attrs.get("standard_name", "")).lower() == "latitude":
                return var.values.copy()
    raise KeyError(f"No latitude variable found in {latlon_nc_path}")

"""Read or derive the 2-D latitude array for the L1 meteo grid."""

from __future__ import annotations
from pathlib import Path

import importlib.util

import numpy as np
import xarray as xr

# import header_to_latlon from mod11 by path to avoid sys.path conflicts
_latlon_grid_path = (
    Path(__file__).parent.parent / "mod11_meteo_to_mhm" / "latlon_grid.py"
)
_spec = importlib.util.spec_from_file_location("latlon_grid", _latlon_grid_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
header_to_latlon = _mod.header_to_latlon


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


def compute_latitude_from_header(header_path: Path, coord_sys: str) -> np.ndarray:
    """Return 2-D latitude array (degrees) derived from a header.txt + projected CRS.

    Row 0 is northernmost — same convention as latlon.nc produced by mod16.
    """
    _lons, lats, _xx, _yy, _miss = header_to_latlon(header_path, coord_sys)
    return lats

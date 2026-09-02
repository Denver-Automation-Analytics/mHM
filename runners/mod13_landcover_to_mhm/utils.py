"""Header IO and L0 grid construction."""

from __future__ import annotations
import logging
import math
from pathlib import Path

import geopandas as gpd

log = logging.getLogger(__name__)


# --- header parsing / writing --------------------------------------
def load_header(path: str | Path) -> dict:
    """Parse an mHM-style header.txt into a normalized dict."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Header file not found: {path}")

    parsed = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        key, val = line.split()
        parsed[key] = val

    return {
        "ncols": int(parsed["ncols"]),
        "nrows": int(parsed["nrows"]),
        "xllcorner": float(parsed["xllcorner"]),
        "yllcorner": float(parsed["yllcorner"]),
        "cellsize": float(parsed["cellsize"]),
        "NODATA_value": float(parsed["NODATA_value"]),
    }


def write_header_txt(header: dict, out_path: Path) -> None:
    """Write six-line mHM header.txt (identical format to meteo module)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"ncols        {header['ncols']}",
        f"nrows        {header['nrows']}",
        f"xllcorner    {float(header['xllcorner'])}",
        f"yllcorner    {float(header['yllcorner'])}",
        f"cellsize     {int(header['cellsize'])}",
        f"NODATA_value {int(header['NODATA_value'])}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    log.info("Wrote %s", out_path)


def load_header_from_nc(path: str | Path) -> dict:
    """Derive an mHM-style header dict from a morph NetCDF file written by write_nc."""
    import netCDF4 as nc4

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Morph NetCDF not found: {path}")
    with nc4.Dataset(path) as ds:
        x = ds["x"][:].data
        y = ds["y"][:].data
        nodata = -9999.0
        for vname, var in ds.variables.items():
            if vname in ("x", "y"):
                continue
            if var.ndim == 2:
                nodata = float(getattr(var, "_FillValue", -9999.0))
                break
    cellsize = float(x[1] - x[0])
    xllcorner = float(x.min() - 0.5 * cellsize)
    yllcorner = float(y.min() - 0.5 * cellsize)
    return {
        "ncols": len(x),
        "nrows": len(y),
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "cellsize": cellsize,
        "NODATA_value": nodata,
    }


# --- L0 header derivation ------------------------------------------
def build_l0_header_from_watershed(
    watershed_path: str, l0_cellsize_m: int, crs: str
) -> dict:
    """Derive L0 grid by projecting the watershed bbox to crs and snapping to l0_cellsize_m."""
    ws = gpd.read_file(watershed_path).to_crs(crs)
    raw_xmin, raw_ymin, raw_xmax, raw_ymax = ws.total_bounds
    # math.floor/ceil return int; multiply by int cellsize stays int — cast to float for header formatting
    xll = float(math.floor(raw_xmin / l0_cellsize_m) * l0_cellsize_m)
    yll = float(math.floor(raw_ymin / l0_cellsize_m) * l0_cellsize_m)
    xur = float(math.ceil(raw_xmax / l0_cellsize_m) * l0_cellsize_m)
    yur = float(math.ceil(raw_ymax / l0_cellsize_m) * l0_cellsize_m)
    ncols = round((xur - xll) / l0_cellsize_m)
    nrows = round((yur - yll) / l0_cellsize_m)
    return {
        "ncols": ncols,
        "nrows": nrows,
        "xllcorner": xll,
        "yllcorner": yll,
        "cellsize": l0_cellsize_m,
        "NODATA_value": -9999,
    }

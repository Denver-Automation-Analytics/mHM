"""Header IO helpers and projection extraction for the hydro module."""

from __future__ import annotations
import logging
from pathlib import Path

log = logging.getLogger(__name__)


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
    """Write six-line mHM header.txt (identical format to other modules)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"ncols        {header['ncols']}",
        f"nrows        {header['nrows']}",
        f"xllcorner    {header['xllcorner']}",
        f"yllcorner    {header['yllcorner']}",
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


def read_projection_wkt(latlon_path: str | Path) -> str:
    """Extract the projection WKT from the meteo latlon.nc global attrs."""
    import netCDF4 as nc

    with nc.Dataset(latlon_path) as fh:
        proj = getattr(fh, "projection", None)
    if not proj or proj == "geographic":
        raise ValueError(
            f"Could not read a projected CRS from {latlon_path}. "
            "Set TARGET_CRS_WKT explicitly in main.py."
        )
    return proj

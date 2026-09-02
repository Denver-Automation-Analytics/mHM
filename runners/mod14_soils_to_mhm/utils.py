"""Shared utilities: logging, header I/O, and grid derivation."""

from __future__ import annotations
import logging
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_header(path: str | Path) -> dict:
    """Parse a six-line mHM-style header.txt into a normalised dict."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Header file not found: {path}")
    parsed: dict[str, str] = {}
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


def read_lcc_crs_from_latlon(latlon_path: Path) -> str:
    """Read the LCC projection WKT stored in a meteo latlon.nc global attribute."""
    import netCDF4 as nc  # noqa: PLC0415

    with nc.Dataset(latlon_path) as fh:
        proj = getattr(fh, "projection", None)
    if not proj or proj == "geographic":
        raise ValueError(
            f"Could not read a projected CRS from {latlon_path}. "
            "Set TARGET_CRS_WKT explicitly in main.py."
        )
    return proj

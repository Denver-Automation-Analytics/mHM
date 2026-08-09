"""Header IO, L0 grid construction, and cross-grid alignment checks."""

from __future__ import annotations
import logging
from pathlib import Path

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
        "ncols":        int(parsed["ncols"]),
        "nrows":        int(parsed["nrows"]),
        "xllcorner":    float(parsed["xllcorner"]),
        "yllcorner":    float(parsed["yllcorner"]),
        "cellsize":     float(parsed["cellsize"]),
        "NODATA_value": float(parsed["NODATA_value"]),
    }


def write_header_txt(header: dict, out_path: Path) -> None:
    """Write six-line mHM header.txt (identical format to meteo module)."""
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


# --- L0 header derivation ------------------------------------------
def build_l0_header_from_meteo(meteo_header: dict, refinement: int) -> dict:
    """
    Build an L0 header that:
      * Uses cellsize_L0 = cellsize_L2 / refinement (must divide evenly).
      * Shares the exact extent as the L2 (meteo) grid so
        xll + ncols*cs and yll + nrows*cs match byte-for-byte.
    """
    if refinement < 1 or int(meteo_header["cellsize"]) % refinement != 0:
        raise ValueError(
            f"L0_REFINEMENT_FACTOR={refinement} must be a positive integer "
            f"divisor of L2 cellsize ({int(meteo_header['cellsize'])})."
        )

    cs_l0 = int(meteo_header["cellsize"]) // refinement
    ncols = meteo_header["ncols"] * refinement
    nrows = meteo_header["nrows"] * refinement

    return {
        "ncols":        ncols,
        "nrows":        nrows,
        "xllcorner":    meteo_header["xllcorner"],
        "yllcorner":    meteo_header["yllcorner"],
        "cellsize":     cs_l0,
        "NODATA_value": -9999,
    }


# --- alignment checks ----------------------------------------------
def assert_grid_alignment(meteo_header: dict, l0_header: dict) -> None:
    """Verify L0 and L2 share the same extent and satisfy the multiple rule."""
    # multiple-of check
    if int(meteo_header["cellsize"]) % int(l0_header["cellsize"]) != 0:
        raise AssertionError(
            f"L2 cellsize ({meteo_header['cellsize']}) is not an integer "
            f"multiple of L0 cellsize ({l0_header['cellsize']})."
        )

    # extent check: xur, yur must match to sub-meter tolerance
    def _extent(h):
        xur = h["xllcorner"] + h["ncols"] * h["cellsize"]
        yur = h["yllcorner"] + h["nrows"] * h["cellsize"]
        return h["xllcorner"], h["yllcorner"], xur, yur

    xll_m, yll_m, xur_m, yur_m = _extent(meteo_header)
    xll_0, yll_0, xur_0, yur_0 = _extent(l0_header)

    for a, b, name in [
        (xll_m, xll_0, "xllcorner"),
        (yll_m, yll_0, "yllcorner"),
        (xur_m, xur_0, "xurcorner"),
        (yur_m, yur_0, "yurcorner"),
    ]:
        if abs(a - b) > 1e-6:
            raise AssertionError(
                f"L0 and L2 grids disagree on {name}: meteo={a}, L0={b}."
            )
    log.info("Grid alignment check OK: L0 exactly covers L2 extent.")
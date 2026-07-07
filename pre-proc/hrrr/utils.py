"""Shared logging and consistency-check utilities."""

from __future__ import annotations
import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def assert_grid_consistency(ds_clip, header: dict, cell_size_m: int) -> None:
    """Assert clipped grid dims/extents match the header we're writing."""
    ny, nx = ds_clip.sizes["y"], ds_clip.sizes["x"]
    if (nx, ny) != (header["ncols"], header["nrows"]):
        raise AssertionError(
            f"Header ncols/nrows ({header['ncols']}, {header['nrows']}) "
            f"does not match clipped grid ({nx}, {ny})."
        )
    # xllcorner + ncols * cellsize should match the upper-right edge
    xur = header["xllcorner"] + header["ncols"] * cell_size_m
    yur = header["yllcorner"] + header["nrows"] * cell_size_m
    logging.getLogger(__name__).info(
        "Grid check OK: %d x %d cells, extents [%d, %d] to [%d, %d]",
        nx, ny, header["xllcorner"], header["yllcorner"], xur, yur,
    )
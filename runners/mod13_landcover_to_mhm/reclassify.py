"""GHL 10-class -> mHM 3-class reclassification with class-stat logging."""

from __future__ import annotations
import logging
from typing import Dict

import numpy as np

log = logging.getLogger(__name__)

MHM_CLASS_NAMES = {1: "Forest", 2: "Impervious", 3: "Pervious"}


def to_mhm_classes(
    arr: np.ndarray,
    class_map: Dict[int, int],
    nodata_src: int,
    nodata_mhm: int,
) -> np.ndarray:
    """
    Remap `arr` (GHL codes) to mHM's {1, 2, 3} classes with `nodata_mhm` fill.

    Any source code absent from `class_map` becomes NoData (safer than a silent
    default). GHL's native `nodata_src` also becomes NoData.
    """
    out = np.full(arr.shape, nodata_mhm, dtype=np.int32)

    # Vectorized remap
    for src_code, mhm_class in class_map.items():
        out[arr == src_code] = mhm_class

    # Ensure GHL nodata -> mHM nodata
    out[arr == nodata_src] = nodata_mhm

    # Report any unmapped codes for the user to notice
    unmapped = np.setdiff1d(np.unique(arr), list(class_map.keys()) + [nodata_src])
    if unmapped.size > 0:
        log.warning(
            "Unmapped GHL codes present in scene: %s (set to NoData=%d)",
            unmapped.tolist(),
            nodata_mhm,
        )

    return out


def log_class_stats(grid: np.ndarray, nodata: int, year: int) -> None:
    """Log per-class pixel counts and percentages."""
    total = grid.size
    valid = int((grid != nodata).sum())
    log.info(
        "[%s] Grid: %d cells (%d valid, %d NoData)", year, total, valid, total - valid
    )
    for code, name in MHM_CLASS_NAMES.items():
        n = int((grid == code).sum())
        pct = (n / valid * 100.0) if valid else 0.0
        log.info(
            "[%s]   class %d (%-10s): %8d cells (%5.1f%%)", year, code, name, n, pct
        )

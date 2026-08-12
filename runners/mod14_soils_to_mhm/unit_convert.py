"""
Scale raw SoilGrids INT16 values to the integer units expected by lut.py.

SoilGrids ISRIC d_factor table (applied to raw INT16 from GEOTIFF_INT16 WCS):
  bdod: raw × 0.01 = g/cm³  →  store as mg/cm³ (×10) so lut.py /1000 → g/cm³
  clay: raw ÷ 10   = %       →  store as integer % (raw × 0.1)
  sand: raw ÷ 10   = %       →  same as clay
"""

from __future__ import annotations
import numpy as np

_NODATA = -9999

# Per-property multiplier: raw INT16 → integer unit stored in ASCII grid
_SCALE: dict[str, float] = {
    "bd": 10.0,   # raw cg/cm³ → mg/cm³ (lut.py divides by 1000 → g/cm³)
    "cl": 0.1,    # raw g/kg → integer % (÷10)
    "sn": 0.1,    # raw g/kg → integer % (÷10)
}


def convert(arr: np.ndarray, prop: str, nodata: int = _NODATA) -> np.ndarray:
    """
    Apply the per-property scale factor and return a rounded int32 array.
    Pixels equal to `nodata` are passed through unchanged.
    """
    if prop not in _SCALE:
        raise ValueError(f"Unknown property '{prop}'; expected one of {list(_SCALE)}")
    mask = arr == nodata
    scaled = np.round(arr.astype(np.float64) * _SCALE[prop]).astype(np.int32)
    scaled[mask] = nodata
    return scaled

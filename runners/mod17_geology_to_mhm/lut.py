"""Compose the geology class grid (first-wins) and write the mHM LUT + ASC."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from utils import write_ascii_grid

log = logging.getLogger("geology_to_mhm")

MAX_GEO_UNITS = 25   # mHM parameter maxGeoUnit (mo_mpr_constants.f90)
NONKARST_ID = 1
NONKARST_DESC = "Non_karst"


def build_geology(class_masks: list[tuple[str, np.ndarray]], grid_def: dict,
                  out_dir: str | Path, nodata: int = -9999) -> dict:
    """Assign geology ClassUnits and write geology_class.asc + classdefinition.

    class_masks are (label, bool L0 mask) in priority order (first-wins on
    overlap). Unit 1 is the non-karst background over the DEM-valid domain;
    each karst class with >=1 uncontested cell becomes its own karstic unit.
    Returns a mapping of label -> assigned ClassUnit id (present classes only).
    """
    out_dir = Path(out_dir)
    valid = grid_def["valid_mask"]
    shape = valid.shape

    # First-wins ownership: 0 = non-karst background, i = ith karst class.
    owner = np.zeros(shape, dtype=np.int32)
    for i, (_, mask) in enumerate(class_masks, start=1):
        take = mask & valid & (owner == 0)
        owner[take] = i

    # Keep only karst classes that actually claimed cells; assign contiguous ids.
    present: list[tuple[str, int]] = []
    next_id = NONKARST_ID + 1
    id_of: dict[int, int] = {}
    for i, (label, _) in enumerate(class_masks, start=1):
        if np.any(owner == i):
            present.append((label, next_id))
            id_of[i] = next_id
            next_id += 1

    n_units = 1 + len(present)
    if n_units > MAX_GEO_UNITS:
        raise ValueError(
            f"{n_units} geology units exceed mHM maxGeoUnit={MAX_GEO_UNITS}."
        )

    class_grid = np.where(valid, NONKARST_ID, nodata).astype(np.int32)
    for i, gid in id_of.items():
        class_grid[owner == i] = gid

    asc_path = out_dir / "geology_class.asc"
    txt_path = out_dir / "geology_classdefinition.txt"
    write_ascii_grid(asc_path, class_grid, grid_def, nodata=nodata)
    _write_classdefinition(txt_path, present)

    log.info("Wrote %s (%d karst unit(s) + non-karst)", asc_path, len(present))
    log.info("Wrote %s (nGeo_Formations=%d)", txt_path, n_units)
    return {label: gid for label, gid in present}


def _write_classdefinition(path: Path, present: list[tuple[str, int]]) -> None:
    rows = [(NONKARST_ID, NONKARST_ID, 0, NONKARST_DESC)]
    for label, gid in present:
        rows.append((gid, gid, 1, label))

    lines = [
        f"nGeo_Formations  {len(rows)}",
        f"{'GeoParam(i)':<12}{'ClassUnit':<12}{'Karstic':<10}Description",
    ]
    for geoparam, classunit, karstic, desc in rows:
        lines.append(
            f"{geoparam:<12}{classunit:<12}{karstic:<10}{desc}"
        )
    lines.append("!<-END")
    path.write_text("\n".join(lines) + "\n")

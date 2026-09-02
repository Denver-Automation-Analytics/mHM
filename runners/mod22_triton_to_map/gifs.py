"""GIF animations for mod22 (TRITON H/V netCDF cubes -> map).

Renders each variable's cached netCDF cube as an animated GIF: a static DEM
hillshade background and watershed boundary outline, with the variable overlaid
as a semi-transparent color raster (NODATA/dry cells left transparent so the
hillshade shows through). Timesteps are evenly-strided down to a frame cap so
render time and file size stay bounded on long TRITON runs.

The color scale spans the true [0, max] of the frames actually shown (no
percentile clipping, which silently saturated most of the flood extent to the
same top color and made the colorbar misrepresent the real depth range) using a
PowerNorm so shallow, common depths stay visible alongside rare deep pockets.

Frames are quantized to the GIF palette without dithering (Pillow's default
Floyd-Steinberg dithering turns smooth hillshade gradients into a speckled,
tile-like pattern that reads as a decomposed/patchwork raster instead of a
continuous one).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.colors import PowerNorm
from PIL import Image

from writers import VAR_META


def _frame_indices(n: int, max_frames: int) -> np.ndarray:
    """Evenly-spaced, de-duplicated timestep indices spanning the full run."""
    if max_frames is None or n <= max_frames:
        return np.arange(n)
    else:
        return np.unique(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int))


def make_gif(
    nc_path: Path,
    var: str,
    grid: Dict,
    hillshade: np.ndarray,
    boundary,
    cmap: str,
    fps: int,
    nodata: float,
    out_gif: Path,
    max_frames: int,
    gamma: float = 0.4,
    dpi: int = 150,
) -> int:
    """Render *nc_path*'s ``var`` cube to *out_gif*. Returns the frame count used."""
    ds = netCDF4.Dataset(nc_path, "r")
    try:
        v = ds.variables[var]
        v.set_auto_maskandscale(False)
        times = netCDF4.num2date(ds.variables["time"][:], ds.variables["time"].units)
        idx = _frame_indices(v.shape[0], max_frames)

        def read(t: int) -> np.ma.MaskedArray:
            arr = np.asarray(v[t], dtype=np.float32)
            arr = np.where(arr == np.float32(nodata), np.nan, arr)
            return np.ma.masked_invalid(arr)

        # True max over the shown frames (not a percentile clip): keeps the
        # colorbar honest about the real depth range instead of saturating it.
        vmax = 1.0
        for t in idx:
            arr = read(t)
            if arr.count():
                vmax = max(vmax, float(arr.max()))

        var_cmap = copy.copy(matplotlib.colormaps[cmap])
        var_cmap.set_bad(alpha=0.0)  # NODATA/dry cells: hillshade shows through
        norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)

        x, y = grid["x"], grid["y"]
        extent = [x.min(), x.max(), y.min(), y.max()]
        origin = "lower" if y[0] < y[-1] else "upper"
        aspect = (y.max() - y.min()) / (x.max() - x.min())

        fig, ax = plt.subplots(figsize=(8, 8 * aspect))
        ax.imshow(hillshade, extent=extent, origin=origin, cmap="gray", aspect="equal")
        im = ax.imshow(
            read(idx[0]),
            extent=extent,
            origin=origin,
            cmap=var_cmap,
            norm=norm,
            alpha=0.85,
            aspect="equal",
        )
        if boundary is not None:
            boundary.plot(ax=ax, color="black", linewidth=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        meta = VAR_META.get(var, {})
        fig.colorbar(im, ax=ax, shrink=0.8, label=meta.get("units", ""))
        title = ax.set_title(
            f"{meta.get('long_name', var)}  {times[idx[0]]:%Y-%m-%d %H:%M}"
        )
        fig.tight_layout()

        # Render each frame to RGB, then quantize to a shared palette (no dither)
        # so GIF encoding doesn't speckle the smooth hillshade into fake tiling.
        canvas = fig.canvas
        base_palette = None
        pil_frames = []
        for t in idx:
            im.set_array(read(t))
            title.set_text(f"{meta.get('long_name', var)}  {times[t]:%Y-%m-%d %H:%M}")
            canvas.draw()
            rgb = np.asarray(canvas.buffer_rgba())[:, :, :3]
            frame = Image.fromarray(rgb, mode="RGB")
            if base_palette is None:
                base_palette = frame.quantize(colors=256, dither=Image.Dither.NONE)
                pil_frames.append(base_palette)
            else:
                pil_frames.append(
                    frame.quantize(palette=base_palette, dither=Image.Dither.NONE)
                )
        plt.close(fig)

        pil_frames[0].save(
            out_gif,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=False,
        )
    finally:
        ds.close()
    return len(idx)

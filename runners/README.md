# Implementation Checklist

Modules run in numeric order (mod10 → mod22) on a user-supplied watershed boundary
(`data.geojson`) and the shared `config.py` settings.

| Module | Input / artifact | Data source | Preprocess | Final mHM role |
| --- | --- | --- | --- | --- |
| mod10 – Terrain | DEM, slope, aspect, flow direction/accumulation | USGS 3DEP | Breach/fill conditioning, derive slope/aspect + D8 fdir/facc, resample to Level-0, mask | dem/slope/aspect/facc/fdir; terrain predictors + routing network. |
| mod11 – Meteorology | pre, tavg, windspeed, rel. humidity, radiation | NOAA HRRR (dynamical.org Icechunk Zarr) | Clip to watershed on native 3 km LCC grid, unit-convert, write headers | Precipitation/temperature drivers (+ PET inputs). |
| mod12 – PET | pet.nc | Derived from mod11 temperature | Compute PET (Hargreaves–Samani / Oudin / Priestley–Taylor / Penman–Monteith) from tavg + latlon | Potential-evapotranspiration driver. |
| mod13 – Land cover | Forest / Impervious / Pervious fractions | Space Intelligence GHL (30 m, Earthmover/Arraylake) | Clip, reproject to L0, majority resample, reclassify 10 → 3 classes | Vegetation and impervious predictors. |
| mod14 – Soils | bulk density, clay, sand (6 depths) | ISRIC SoilGrids (WCS) | Clip on native Homolosine CRS, reproject to L0 at 250 m, map horizons | MPR soil predictors. |
| mod15 – Gauges | idgauges.asc + `<id>.txt` discharge | USGS NWIS | Discover gauges inside polygon, download daily/hourly Q, ft³/s → m³/s, QC | Routing outlet and calibration observations. |
| mod16 – LAI | lai.nc (monthly climatology) | MODIS MCD15A3H v6.1 (Planetary Computer) | Clip, scale, QA-mask, regrid to L0, build 12-month climatology | Vegetation seasonality. |
| mod17 – Geology | geology_class.asc | USGS "Karst in the United States" | Rasterize karst ClassUnit ids onto the L0 grid | Baseflow/geology MPR predictor. |
| mod18 – latlon | latlon.nc | Generated (mod10/11/14 grid headers) | Combine L0/L1/L2 grid headers into one file | WGS84 grid coordinates. |
| mod19 – Calibration | mhm.nml + companion namelists; calibrated run | mHM | Write `optimize=.TRUE.` namelists, launch mHM, produce FinalParam.nml | Parameter estimation → hydrographs/states. |
| mod20 – Diagnostics | forward run + QA plots/metrics | mHM outputs | Forward run with FinalParam, check runoff coeff / BFI / aET / recharge, water balance | Detect data/preprocess errors; final states/fluxes. |
| mod21 – TRITON inputs | `.dem` / `.rmap` / `.roff` / `.mann` / `.obs` / `.cfg` | mHM/mRM outputs (+ ESRI IO 10 m LULC, NHDPlus HR waterbodies) | Reproject/clip DEM, map L1 runoff zones, burn Manning's n, seed waterbodies | Drives the TRITON 2D hydraulic model. |
| mod22 – Flood maps | H / MH / V netCDF + pixel-max GeoTIFFs | TRITON outputs | Consolidate per-timestep GeoTIFFs, clip, animate (GIF) | Flood depth/velocity maps and envelopes. |
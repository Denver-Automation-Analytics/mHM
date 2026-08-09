# Implementation Checklist

| Input / artifact | Status | Recommended U.S. source | Preprocess | Final mHM role |
| --- | --- | --- | --- | --- |
| Watershed boundary/outlet | Starting point | User boundary + USGS NWIS gauge/site metadata | Snap outlet to DEM/NHDPlus-consistent flow path | Defines domain/gauge IDs. |
| DEM | Required | USGS 3DEP | Fill sinks, resample to Level-0, mask | dem.asc, terrain predictor. |
| Slope/aspect | Required | Derived from DEM | GDAL/GRASS/ArcGIS derivative, same grid | slope.asc, aspect.asc. |
| Flow direction/accumulation | Required | Derived from filled DEM; cross-check with NHDPlus HR | Use D8/compatible convention, same grid | fdir.asc, facc.asc, basin/routing extraction. |
| Gauge map + discharge | Required for routed calibration | USGS NWIS | Create idgauges.asc; format [gauge-id].txt | Routing outlet and calibration observations. |
| Soil class/table | Required | gSSURGO; SoilGrids fallback | Map soil units, horizons, clay/sand/bulk density | MPR soil predictors. |
| Geology/hydrogeology | Required in standard docs; recommended | USGS principal aquifers, NGMDB, GLHYMPS | Classify units, set karst flag/parameter link | Baseflow/geology MPR predictor. |
| Land cover | Required | Annual NLCD | Reclassify to forest / impervious / pervious | Vegetation and impervious predictors. |
| LAI | Required/configurable | MODIS MCD15A3H | Build monthly class lookup or daily gridded Level-0 LAI | Vegetation seasonality. |
| Meteorology | Required | AORC, Daymet, NLDAS, gridMET | Clip, rename, unit-convert, header, _FillValue, align grid | pre, tavg, pet/PET drivers. |
| latlon.nc | Required | Generated | Use mhm-tools setup-creation latlon or official script | WGS84 grid coordinates. |
| Namelists | Required | mHM template/test domain | Update paths, periods, resolutions, switches | Runtime/control files. |
| Calibration settings | Required for calibration | mHM native | Set optimize, method, objective, iterations, flags | Parameter estimation. |
| QA outputs | Recommended | mHM outputs | Enable monthly flux/state outputs and water-balance checks | Detect data/preprocess errors. |
| Final calibrated run | Required deliverable | mHM | Use FinalParams.nml, turn optimization off, rerun | Final hydrographs and states/fluxes. |
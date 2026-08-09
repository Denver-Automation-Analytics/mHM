# -*- coding: utf-8 -*-

# Imports #####################################################################
import io
from typing import Optional
import numpy as np
import requests
import pandas as pd
import geopandas as gpd
from pygeohydro import NWIS

# Functions ###################################################################


def get_usgs_stations(
    model_perimeter: gpd.GeoDataFrame, variable_type: str, dates: Optional[tuple] = None
):
    """
    Get the USGS gage stations within the model perimeter

    Parameters
    ----------
    model_perimeter : gpd.GeoDataFrame
        The perimeter of the model
    variable_type : str
        The type of variable to retrieve (e.g., "flow", "stage")
    dates : tuple
        The start and end dates for the gage station data retrieval

    Returns
    -------
    stations: list
        The USGS gage stations within the model perimeter
    """
    # Create an instance of the NWIS class
    nwis = NWIS()

    # Get the bounding box of the model perimeter
    bbox = tuple(model_perimeter.bounds.values[0])  # (minx, miny, maxx, maxy)
    if variable_type == "flow":
        parameter_cd = "00060" # discharge in cubic feet per second
    elif variable_type == "stage":
        parameter_cd = "00065" # gage height in feet
    # Query gage stations with daily values
    query_dv = {
        "bBox": ",".join(f"{b:.06f}" for b in bbox),
        "hasDataTypeCd": "dv",
        "outputDataTypeCd": "dv",
        "parameterCd": parameter_cd,
    }
    info_box_dv = nwis.get_info(query_dv)

    # Query gage stations with instantaneous values
    query_iv = {
        "bBox": ",".join(f"{b:.06f}" for b in bbox),
        "hasDataTypeCd": "iv",
        "outputDataTypeCd": "iv",
        "parameterCd": parameter_cd,
    }
    info_box_iv = nwis.get_info(query_iv)

    if dates is None:
        # Don't filter the gage stations by date
        df_gages_usgs = pd.concat([info_box_dv, info_box_iv]).drop_duplicates(
            subset=["site_no"]
        )
    else:
        # Filter the gage stations by date
        dv_gages = info_box_dv[
            (info_box_dv.begin_date <= dates[0]) & (info_box_dv.end_date >= dates[1])
        ]
        iv_gages = info_box_iv[
            (info_box_iv.begin_date <= dates[0]) & (info_box_iv.end_date >= dates[1])
        ]
        # Combine the gage stations with daily and instantaneous values
        df_gages_usgs = pd.concat([dv_gages, iv_gages]).drop_duplicates(
            subset=["site_no"]
        )

    # Convert df_gages_usgs to a geodataframe
    df_gages_usgs = gpd.GeoDataFrame(
        df_gages_usgs,
        geometry=gpd.points_from_xy(
            df_gages_usgs.dec_long_va, df_gages_usgs.dec_lat_va
        ),
        crs="EPSG:4326",
    )
    # Filter the gages to only include those within the model perimeter
    # This is necesarry since the NWIS() class only support query by rectangular bbox
    df_gages_usgs = df_gages_usgs[
        df_gages_usgs.within(model_perimeter.geometry.iloc[0])
    ]
    return df_gages_usgs.reset_index(drop=True)


def _map_qualifier(code: str) -> str:
    """Map a USGS RDB qualifier code to a human-readable approval status string."""
    s = str(code).strip()
    if "A" in s:
        return "Approved"
    if "P" in s:
        return "Provisional"
    return s


def get_nwis(site: str, 
             parameter: str,
             frequency: str,
             start_date: str,
             end_date: str,
             output_format: Optional[str] = "rdb",
             ):
    '''retrieve instantaneous data for a usgs site, write to a file (optional, and return as a dataframe

    Note: currently limits to USGS sites only, all sites (regardless of active status), and stream discharge only
    Note: api call built from USGS api builder: https://waterservices.usgs.gov/rest/IV-Test-Tool.html

    Args
        site (str): gage id 
        parameter (str): one of 'Flow', 'Stage', or 'Precipitation'
        frequency (str): one of 'iv' or 'dv'
        start_date (str): formatted to 'YYYY-MM-DD'
        end_date (str): formatted to 'YYYY-MM-DD'
        write_file (boolean): default to False
        output_format (str): one of [txt, waterML-2.0, json].
    Return
        df (pd.DataFrame): formatted dataframe of peak data
    '''

    if parameter == 'Flow':
        
        param_id = '00060'
        try:
            # build url and make call only for USGS funded sites
            url = f"https://waterservices.usgs.gov/nwis/{frequency}/?format={output_format}&sites={site}&startDT={start_date}&endDT={end_date}&parameterCd={param_id}&siteType=ST&agencyCd=usgs&siteStatus=all"
            r = requests.get(url)

            # check that api call worked
            if r.status_code!=200:
                print(f"Server response {r.status_code}: Returning None")
                return None

            # decode results
            if output_format=='rdb':
                df = pd.read_table(io.StringIO(r.content.decode('utf-8')), 
                                comment='#',
                                skip_blank_lines=True)
                df = df.iloc[1:].copy()

            if frequency == 'iv':
                data_col_idx = 4
                qual_col_idx = 5
            elif frequency == 'dv':
                data_col_idx = 3
                qual_col_idx = 4
            timestep_idx = 2

            if len(df.dropna()) == 0:
                print('No data available for the time period specified')
                return None
            elif df[df.columns[data_col_idx]].values[0] == 'ZFL':
                print('Zero flow condition: Return None')
                return None
            elif df[df.columns[data_col_idx]].values[0] == '***':
                print('Data temporarily unavailable for the time period specified')
                return None
            elif set(df['site_no'].isnull()) == {True}:
                return None

            qual_col = df.columns[qual_col_idx] if qual_col_idx < len(df.columns) else None
            timestamps = pd.to_datetime(df[df.columns[timestep_idx]].values)
            values = pd.to_numeric(df[df.columns[data_col_idx]], errors="coerce")
            statuses = (
                df[qual_col].map(_map_qualifier)
                if qual_col is not None
                else pd.Series("", index=df.index)
            )
            final_df = pd.DataFrame(
                {"value": values.values, "approval_status": statuses.values},
                index=timestamps,
            )
            return final_df
    
        # when an incorrect gage number is sent to the api, we end up at usgs url (second failure mechanism)
        except:
            return None
    
    elif parameter == 'Precipitation':
        param_id = '00045'
        try:
            # build url and make call for all available rain gage sites for CONUS
            url = f"https://waterservices.usgs.gov/nwis/{frequency}/?format={output_format}&sites={site}&startDT={start_date}&endDT={end_date}&parameterCd={param_id}&siteStatus=all"
            r = requests.get(url)

            # check that api call worked
            if r.status_code!=200:
                print(f"Server response {r.status_code}: Returning None")
                return None

            # decode results
            if output_format=='rdb':
                df = pd.read_table(io.StringIO(r.content.decode('utf-8')), 
                                comment='#',
                                skip_blank_lines=True)
                df = df.iloc[1:].copy()

            if frequency == 'iv':
                data_col_idx = 4
            elif frequency == 'dv':
                data_col_idx = 3
            timestep_idx = 2

            if len(df.dropna()) == 0:
                print('No data available for the time period specified')
                return None
            elif df[df.columns[data_col_idx]].values[0] == '***':
                print('Data temporarily unavailable for the time period specified')
                return None
            elif set(df['site_no'].isnull()) == {True}:
                return None

            # format the final dataframe to represent the observed rainfall following a datetime index
            final_df = pd.DataFrame(df[df.columns[data_col_idx]].astype('float'))
            final_df.columns = [site]
            final_df.index = pd.to_datetime(df[df.columns[timestep_idx]].values)

            return final_df
    
        # when an incorrect gage number is sent to the api, we end up at usgs url (second failure mechanism)
        except:
            return None
        
    elif parameter == 'Stage':
        
        param_id = '00065'
        try:
            # build url and make call only for USGS funded sites
            url = f"https://waterservices.usgs.gov/nwis/{frequency}/?format={output_format}&sites={site}&startDT={start_date}&endDT={end_date}&parameterCd={param_id}&siteType=ST&agencyCd=usgs&siteStatus=all"
            r = requests.get(url)

            # check that api call worked
            if r.status_code!=200:
                print(f"Server response {r.status_code}: Returning None")
                return None

            # decode results
            if output_format=='rdb':
                df = pd.read_table(io.StringIO(r.content.decode('utf-8')), 
                                comment='#',
                                skip_blank_lines=True)
                df = df.iloc[1:].copy()
                

            if frequency == 'iv':
                # Columns: ['agency_cd', 'site_no', 'datetime', 'value', 'qualifiers', 'remark']
                data_col_idx = 4
            elif frequency == 'dv':
                # Columns: ['agency_cd', 'site_no', 'datetime', 'value', 'qualifiers']
                data_col_idx = 3
            timestep_idx = 2

            if len(df.dropna()) == 0:
                print('No data available for the time period specified')
                return None
            elif df[df.columns[data_col_idx]].values[0] == 'ZFL':
                print('Zero flow condition: Return None')
                return None
            elif df[df.columns[data_col_idx]].values[0] == '***':
                print('Data temporarily unavailable for the time period specified')
                return None
            elif set(df['site_no'].isnull()) == {True}:
                return None
            
            # format the final dataframe to represent the observed rainfall following a datetime index
            final_df = pd.DataFrame(df[df.columns[data_col_idx]].astype('float'))
            final_df.columns = [site]
            final_df.index = pd.to_datetime(df[df.columns[timestep_idx]].values)

            return final_df
    
        # when an incorrect gage number is sent to the api, we end up at usgs url (second failure mechanism)
        except:
            return None

    else:
        print('Only gaged Streamflow and Precipitation are available parameters for analysis at this point in time')
        return


# ---------------------------------------------------------------------------
# iv post-processing helpers
# ---------------------------------------------------------------------------

def _dominant_hourly_status(series: pd.Series) -> str:
    """Return dominant approval status across an aggregation window."""
    vals = {str(v) for v in series if str(v) not in ("nan", "")}
    if any("Approved" in v for v in vals):
        return "Approved"
    if any("Provisional" in v for v in vals):
        return "Provisional"
    return next(iter(vals), "")


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample iv DataFrame to 1-hour bins using a time-weighted mean.

    Each sub-hourly reading is weighted by the duration it represents
    (time interval to the next reading). NaN values are excluded from
    both numerator and denominator so they do not bias the hour average.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["value", "approval_status"])

    df = df.sort_index().copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Duration each reading represents: interval to the next record.
    dt = df.index.to_series().diff().shift(-1).dt.total_seconds()
    # Fill the trailing row with the median interval (avoids zero-weight edge).
    dt.iloc[-1] = float(dt.median()) if len(dt) > 1 else 900.0
    df["_w"] = dt.clip(lower=0)

    # Build weighted-value numerator; NaN where the reading itself is NaN.
    valid = df["value"].notna()
    df["_vw"] = np.where(valid, df["value"] * df["_w"], np.nan)
    df["_wv"] = np.where(valid, df["_w"], 0.0)  # weight only for valid rows

    hourly_vw  = df["_vw"].resample("1h").sum(min_count=1)
    hourly_w   = df["_wv"].resample("1h").sum()
    hourly_val = hourly_vw / hourly_w  # NaN for hours with no valid readings

    hourly_status = df["approval_status"].resample("1h").apply(_dominant_hourly_status)

    return pd.DataFrame({"value": hourly_val, "approval_status": hourly_status})


def interpolate_gaps(
    df: pd.DataFrame,
    nodata: int,
    max_gap_hours: Optional[int] = None,
) -> pd.DataFrame:
    """Fill nodata-sentinel gaps with time-weighted linear interpolation.

    Gaps longer than *max_gap_hours* (if set) remain as *nodata*.
    """
    out = df.copy()
    mask = out["value"] == nodata
    out.loc[mask, "value"] = np.nan
    out["value"] = out["value"].interpolate(method="time", limit=max_gap_hours)
    out["value"] = out["value"].fillna(nodata)
    return out
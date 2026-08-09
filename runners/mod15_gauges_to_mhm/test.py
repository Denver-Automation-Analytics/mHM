import os
from dotenv import load_dotenv
from dataretrieval import waterdata
import geopandas as gpd

load_dotenv(override=True)

WATERSHED_PATH = "/workspace/test_domain_3/watershed/allegheny_huc.geojson"
SITES_PATH = "/workspace/test_domain_3/hydro/gauges/sites.geojson"

if not os.path.exists(SITES_PATH):
    watershed = gpd.read_file(WATERSHED_PATH)
    watershed_bounds = watershed.total_bounds  # [minx, miny, maxx, maxy]

    # Get monitoring location information
    sites_gdf, metadata = waterdata.get_monitoring_locations(
        # state='Maryland',  # full name, postal code ('MD'), or FIPS ('24')
        bbox=watershed_bounds.tolist(),  # [min_lon, min_lat, max_lon, max_lat]
        site_type_code='ST'  # Stream sites
    )
    sites_gdf['lat'] = sites_gdf['geometry'].y
    sites_gdf['lon'] = sites_gdf['geometry'].x

    # subset to only those inside the watershed polygon
    sites_gdf = gpd.GeoDataFrame(sites_gdf, geometry=gpd.points_from_xy(sites_gdf.lon, sites_gdf.lat), crs="EPSG:4326")
    gdf_inside = gpd.sjoin(sites_gdf, watershed, how="inner", predicate="within")
    print(f"Found {len(gdf_inside)} monitoring locations inside the watershed polygon")

    print(gdf_inside)
    gdf_inside.to_file(SITES_PATH, driver="GeoJSON")
else:
    gdf_inside = gpd.read_file(SITES_PATH)
    print(f"Loaded {len(gdf_inside)} monitoring locations from cached GeoJSON")

# Get daily streamflow data (returns DataFrame and metadata)
sites = gdf_inside['monitoring_location_id'].tolist()
print(f"Fetching daily streamflow data for {len(sites)} sites: {sites}")
df, metadata = waterdata.get_daily(
    monitoring_location_id=sites,
    parameter_code='00060',  # Discharge
    time='2024-10-01/..'
)


if df.empty:
    print("No data retrieved. Check the site IDs and date range.")
else:
    print(f"Retrieved {len(df)} records")
    print(f"Site: {df['monitoring_location_id'].iloc[0]}")
    print(f"Mean discharge: {df['value'].mean():.2f} {df['unit_of_measure'].iloc[0]}")

    # save to CSV
    df.to_csv("/workspace/test_domain_3/hydro/gauges/streamflow_data.csv", index=False)
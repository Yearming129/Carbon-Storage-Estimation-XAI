import os
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

# Input file path
shapefile_path = r'E:\梁老师scs\XAI dataset (100m finished)\table\pixel.shp'
csv_folder = r'E:\梁老师scs\XAI dataset (100m finished)\table_1\SHAP\initial csv'
output_folder = r'D:\XAI dataset (100m)\table\results_figures\geoshap'

# Create output folder (if it does not exist)
os.makedirs(output_folder, exist_ok=True)

# Read vector file
gdf_original = gpd.read_file(shapefile_path)

# Check if the vector file contains the 'ID_1' column
if 'ID_1' not in gdf_original.columns:
    raise ValueError("The vector file is missing the 'ID_1' column!")

# Get the list of CSV files
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

if not csv_files:
    raise ValueError("No CSV files found!")

# Define raster resolution (100m x 100m)
resolution = 100  # Unit is meters

# Boundaries of the vector file
minx, miny, maxx, maxy = gdf_original.total_bounds

# Calculate the number of rows and columns for the raster
width = int((maxx - minx) / resolution)
height = int((maxy - miny) / resolution)

# Define the affine transformation for the raster
transform = from_origin(minx, maxy, resolution, resolution)

# Iterate through each quarter's CSV file
for csv_file in csv_files:
    # Extract year and quarter (assuming filename format is 'Shap_YYYY_Q_modified.csv')
    parts = csv_file.split('_')
    if len(parts) < 4:
        print(f"Filename format does not meet requirements, skipping: {csv_file}")
        continue

    year = parts[1]
    quarter = parts[2]

    # Read CSV file
    csv_path = os.path.join(csv_folder, csv_file)
    df = pd.read_csv(csv_path)

    # Check if CSV file contains 'ID_1' and columns with 'SHAP_' prefix
    if 'ID_1' not in df.columns:
        print(f"CSV file is missing the 'ID_1' column, skipping: {csv_file}")
        continue

    shap_columns = [col for col in df.columns if col.startswith('SHAP_')]
    if not shap_columns:
        print(f"No columns with 'SHAP_' prefix found in CSV file, skipping: {csv_file}")
        continue

    # Check for 'SHAP_lat' and 'SHAP_long' columns
    if 'SHAP_lat' not in df.columns or 'SHAP_long' not in df.columns:
        print(f"CSV file is missing 'SHAP_lat' or 'SHAP_long' columns, skipping: {csv_file}")
        continue

    # Add Geo_SHAP column
    df['Geo_SHAP'] = df['SHAP_lat'] + df['SHAP_long']

    # Copy the original vector file to prevent modifying the original
    gdf = gdf_original.copy()

    # Merge CSV and vector data (based on 'ID_1')
    gdf = gdf.merge(df, on='ID_1', how='left')

    # Iterate through each SHAP_ column and generate raster (including Geo_SHAP)
    for col in ['Geo_SHAP']:
        # Define output filename
        output_raster = os.path.join(output_folder, f"{col}_{year}_{quarter}.tif")

        # Convert point field values to raster
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[col]))
        raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=-999,  # Background value
            dtype='float32'
        )

        # Save as GeoTIFF raster file
        with rasterio.open(
                output_raster,
                'w',
                driver='GTiff',
                height=raster.shape[0],
                width=raster.shape[1],
                count=1,
                dtype='float32',
                crs=gdf.crs,  # Use the coordinate reference system from the vector file
                transform=transform,
                compress='LZW',
                nodata=-999
        ) as dst:
            dst.write(raster, 1)

        print(f"Raster file has been saved to: {output_raster}")

    # Remove merged SHAP_ columns and Geo_SHAP to restore the original vector data structure
    gdf.drop(columns=shap_columns + ['Geo_SHAP'], inplace=True)

print("All raster files have been generated!")
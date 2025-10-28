import os
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask

# Define input and output paths
carbon_storage_path = r'E:\梁老师scs\Carbon Storage (Quarter)'
lulc_path = r'E:\梁老师scs\XAI dataset (100m)\lulc\Resam_clcd2018.tif'
output_path = r'E:\梁老师scs\XAI dataset (100m)\carbon storage_quarter'

# Create output folder (if it does not exist)
os.makedirs(output_path, exist_ok=True)

# Read LULC raster metadata
with rasterio.open(lulc_path) as lulc:
    lulc_data = lulc.read(1)  # Read LULC data
    lulc_transform = lulc.transform
    lulc_crs = lulc.crs
    lulc_width = lulc.width
    lulc_height = lulc.height
    lulc_nodata = lulc.nodatavals[0]  # Get LULC NoData value

# Iterate through all .tif files in the carbon storage folder
carbon_files = glob.glob(os.path.join(carbon_storage_path, '*.tif'))

for carbon_file in carbon_files:
    # Extract year and quarter
    filename = os.path.basename(carbon_file)
    year, quarter = filename[2:6], filename[7]

    # Open carbon storage raster
    with rasterio.open(carbon_file) as src:
        # Create a blank array matching LULC
        aggregated_data = np.full((lulc_height, lulc_width), -999, dtype=np.float32)

        # Get carbon storage raster resolution, data, and metadata
        carbon_data = src.read(1)
        carbon_transform = src.transform
        carbon_nodata = src.nodatavals[0]
        carbon_res = src.res[0]  # Assuming square pixels
        scale_factor = int(100 / carbon_res)  # Scale from 10m resolution to 100m resolution

        # Iterate through each LULC grid and calculate the sum of carbon storage values within the coverage
        for row in range(lulc_height):
            for col in range(lulc_width):
                # Skip NoData values in LULC
                if lulc_data[row, col] == lulc_nodata:
                    continue

                # Calculate the pixel range corresponding to the current LULC grid
                x_min = lulc_transform[2] + col * 100
                y_max = lulc_transform[5] + row * -100
                x_max = x_min + 100
                y_min = y_max - 100

                # Locate the window on the carbon storage raster
                col_start = int((x_min - carbon_transform[2]) / carbon_res)
                col_end = int((x_max - carbon_transform[2]) / carbon_res)
                row_start = int((y_max - carbon_transform[5]) / -carbon_res)
                row_end = int((y_min - carbon_transform[5]) / -carbon_res)

                # Extract the corresponding window and calculate the sum of values
                window_data = carbon_data[row_start:row_end, col_start:col_end]
                window_data = window_data[window_data != carbon_nodata]  # Remove NoData values
                if window_data.size > 0:
                    aggregated_data[row, col] = window_data.sum()

    # Save results to output path
    output_filename = f'CS_{year}_{"first" if quarter == "1" else "second" if quarter == "2" else "third" if quarter == "3" else "fourth"}.tif'
    output_filepath = os.path.join(output_path, output_filename)

    # Set output metadata
    out_meta = {
        'driver': 'GTiff',
        'height': lulc_height,
        'width': lulc_width,
        'count': 1,
        'dtype': 'float32',
        'crs': lulc_crs,
        'transform': lulc_transform,
        'nodata': -999,
        'compress': 'LZW'  # Set LZW compression
    }

    # Write to output file
    with rasterio.open(output_filepath, 'w', **out_meta) as dst:
        dst.write(aggregated_data, 1)

    print(f"Processing completed: {output_filename}")

print("All files processed successfully!")
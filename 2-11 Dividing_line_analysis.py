import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from collections import defaultdict

# Function: Read raster data
def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1).astype('float32')  # Read the first band and convert to float
        nodata_value = src.nodatavals[0]  # Get NoData value
        if nodata_value is not None:  # If a NoData value exists
            data[data == nodata_value] = np.nan  # Set NoData values to NaN
        meta = src.meta  # Save metadata
    return data, meta


# Function: Reproject and resample raster data
def reproject_raster(source_array, source_meta, target_meta):
    """
    Reproject and resample raster data to target metadata
    """
    target_height, target_width = target_meta['height'], target_meta['width']
    target_transform = target_meta['transform']

    # Create an empty array to hold the resampled data
    reprojected_array = np.empty((target_height, target_width), dtype='float32')

    # Perform reprojecting and resampling
    with rasterio.Env():
        reproject(
            source_array,
            reprojected_array,
            src_transform=source_meta['transform'],
            src_crs=source_meta['crs'],
            dst_transform=target_transform,
            dst_crs=target_meta['crs'],
            resampling=Resampling.bilinear
        )
    return reprojected_array


# Function: Find the most significant boundary based on SHAP values
def find_most_significant_boundary(shap_values, real_values):
    """
    Find the most significant boundary based on SHAP values and return SHAP values and real value arrays
    :param shap_values: SHAP value array
    :param real_values: Real value array
    :return: Corresponding real value at the boundary, filtered SHAP values, and real values
    """
    valid_idx = ~np.isnan(shap_values) & ~np.isnan(real_values)
    shap_values = shap_values[valid_idx]
    real_values = real_values[valid_idx]

    # Sort real values based on SHAP values
    sorted_idx = np.argsort(real_values)
    shap_sorted = shap_values[sorted_idx]
    real_sorted = real_values[sorted_idx]

    # Calculate the rate of change (gradient) of SHAP values
    shap_diff = np.abs(np.diff(shap_sorted))

    # Find the point with the maximum rate of change
    max_diff_idx = np.argmax(shap_diff)
    boundary_value = real_sorted[max_diff_idx]

    return boundary_value, real_sorted, shap_sorted


# Function: Plot and annotate the minimum and maximum boundaries
def plot_with_boundaries(prefix, quarter, min_boundary, min_year, max_boundary, max_year, output_path, real_sorted, shap_sorted):
    """
    Plot the image and display the minimum and maximum boundaries while adjusting the aspect ratio, increasing font size,
    modifying legend labels, and offsetting latitude and longitude values.
    """
    # Create a square plot
    plt.figure(figsize=(10, 10))

    # Offset latitude and longitude values
    if prefix == 'lat':
        real_sorted = real_sorted - 0.16  # Subtract 0.16 from latitude
        min_boundary = min_boundary - 0.16
        max_boundary = max_boundary - 0.16
    elif prefix == 'long':
        real_sorted = real_sorted - 0.24  # Subtract 0.24 from longitude
        min_boundary = min_boundary - 0.24
        max_boundary = max_boundary - 0.24

    # Plot data points
    plt.scatter(real_sorted, shap_sorted, color='green', s=5, alpha=0.6, label='Data Points')

    # Plot minimum and maximum boundary lines
    if prefix == 'lat':
        plt.axvline(min_boundary, color='blue', linestyle='--', linewidth=5, label=f'Southern Boundary: {min_boundary:.2f} ({min_year})')
        plt.axvline(max_boundary, color='red', linestyle='--', linewidth=5, label=f'Northern Boundary: {max_boundary:.2f} ({max_year})')
    elif prefix == 'long':
        plt.axvline(min_boundary, color='blue', linestyle='--', linewidth=5, label=f'Western Boundary: {min_boundary:.2f} ({min_year})')
        plt.axvline(max_boundary, color='red', linestyle='--', linewidth=5, label=f'Eastern Boundary: {max_boundary:.2f} ({max_year})')

    # # Set title and axis labels
    # plt.title(f'{prefix.upper()} - Quarter {quarter} (2018-2024)', fontsize=14)  # Title font
    # plt.xlabel('Real Values', fontsize=14)  # X-axis label font
    # plt.ylabel('SHAP Values', fontsize=14)  # Y-axis label font

    # Adjust tick label font size
    plt.tick_params(axis='both', which='major', labelsize=22)

    # Add grid and legend
    plt.grid()
    plt.legend(fontsize=16)  # Legend font size

    # Save the image to the output path
    plt.savefig(output_path)
    plt.close()


# Input paths
shap_folder = r'D:\XAI dataset (100m)\table\results_figures'
output_folder = r'D:\XAI dataset (100m)\table\divide_plots_1'
os.makedirs(output_folder, exist_ok=True)

# Real value file paths
real_files = {
    'SHAP_DEM': r'D:\XAI dataset (100m)\topography\Resam_DEM.tif',
    'SHAP_lat': r'D:\XAI dataset (100m)\coordination\latitude_yanan.tif',
    'SHAP_long': r'D:\XAI dataset (100m)\coordination\longitude_yanan.tif',
}


# Used to store quarterly summarized data
quarterly_data = defaultdict(list)

# Iterate through the SHAP folder and process each matching file
for file_name in os.listdir(shap_folder):
    if file_name.endswith('.tif'):
        # Remove the file extension
        file_name_no_ext = os.path.splitext(file_name)[0]
        parts = file_name_no_ext.split('_')

        # Check the standard SHAP file format (SHAP_long_2022_4 or SHAP_lat_2022_2)
        if len(parts) == 4 and parts[0] == 'SHAP' and (parts[1] == 'long' or parts[1] == 'lat'):
            prefix = parts[1]  # Extract prefix (long or lat)
            year = parts[2]  # Extract year
            try:
                quarter = int(parts[3])  # Extract quarter
            except ValueError:
                print(f"Warning: Invalid quarter format in file {file_name}")
                continue

            # Get the corresponding real value file path
            real_file_path = real_files.get(f"SHAP_{prefix}", None)

            if not real_file_path or not os.path.exists(real_file_path):
                print(f"Warning: Real value file not found or invalid for {file_name}")
                continue

            shap_file_path = os.path.join(shap_folder, file_name)

            # Read SHAP and real value rasters
            shap_data, shap_meta = read_raster(shap_file_path)
            real_data, real_meta = read_raster(real_file_path)

            # Resample SHAP raster to match real value raster
            shap_resampled = reproject_raster(shap_data, shap_meta, real_meta)

            # Find the boundary
            boundary, real_sorted, shap_sorted = find_most_significant_boundary(shap_resampled, real_data)

            # Store boundary, year, and point data by quarter
            quarterly_data[(prefix, quarter)].append((boundary, year, real_sorted, shap_sorted))

# Plot images by quarter
for (prefix, quarter), boundaries in quarterly_data.items():
    # Extract the boundary and year for each quarter
    min_boundary = min(boundaries, key=lambda x: x[0])  # Minimum boundary
    max_boundary = max(boundaries, key=lambda x: x[0])  # Maximum boundary

    # Get values and corresponding years for min and max boundaries
    min_value, min_year, min_real_sorted, min_shap_sorted = min_boundary
    max_value, max_year, max_real_sorted, max_shap_sorted = max_boundary

    # Output file path
    output_file = os.path.join(output_folder, f"{prefix}_Q{quarter}_summary_plot.png")

    # Plot the image
    plot_with_boundaries(
        prefix, quarter,
        min_value, min_year,
        max_value, max_year,
        output_file,
        min_real_sorted, min_shap_sorted
    )

    print(f"Saved plot for {prefix} Q{quarter} to {output_file}")
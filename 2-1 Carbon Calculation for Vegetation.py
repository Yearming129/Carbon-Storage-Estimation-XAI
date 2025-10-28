import os
import numpy as np
from osgeo import gdal

# Define input and output paths
input_folder = r'F:\梁老师scs\MaskRS'
output_folder = r'F:\梁老师scs\Carbon Storage (Quarter)'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define carbon density dictionary, unit is kg/m2
carbon_density = {
    0: 0.135057,
    1: 0.211357,
    2: 0.69,
    3: 0.284,
    4: 5.846,
    5: 1.303,
    6: 4.547,
    7: 0.4317,
    8: 0.2968,
    9: 0.284,
    10: 0.07149,
    11: 0.1565,
    12: 0
}

# Define years and quarters
years = range(2018, 2025)
quarters = [1, 2, 3, 4]

# Iterate through years and quarters
for year in years:
    for quarter in quarters:
        # Define input file path and output file path
        input_file = os.path.join(input_folder, f'{year}_{quarter}_mosaicMask.tif')
        output_file = os.path.join(output_folder, f'CV{year}_{quarter}.tif')

        # Open input raster file
        dataset = gdal.Open(input_file)
        if not dataset:
            print(f"Cannot open file: {input_file}")
            continue

        # Read raster data
        band = dataset.GetRasterBand(1)
        mask_array = band.ReadAsArray()
        no_data = band.GetNoDataValue()

        # If NoData value is not defined, set to -999
        if no_data is None:
            no_data = -999

        # Create carbon storage raster array (unit kg)
        carbon_storage_array = np.full_like(mask_array, no_data, dtype=np.float32)  # Initialize with NoData value

        # Calculate carbon storage
        for code, density in carbon_density.items():
            carbon_storage_array[mask_array == code] = density * 10 * 10 / 1000000  # 10m x 10m, unit: thousand tons

        # Get projection and geo-transform
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()

        # Create output raster file
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(
            output_file,
            dataset.RasterXSize,
            dataset.RasterYSize,
            1,  # Single band
            gdal.GDT_Float32,  # Data type is 32-bit float
            ['COMPRESS=LZW']  # LZW compression
        )
        if not out_dataset:
            print(f"Cannot create file: {output_file}")
            continue

        # Set projection and geo-transform
        out_dataset.SetProjection(projection)
        out_dataset.SetGeoTransform(geotransform)

        # Write data
        out_band = out_dataset.GetRasterBand(1)
        out_band.WriteArray(carbon_storage_array)
        out_band.SetNoDataValue(no_data)  # Set NoData value
        out_band.FlushCache()

        # Release memory
        dataset = None
        out_dataset = None
        print(f"Carbon storage raster has been saved: {output_file}")

print("All raster calculations are complete!")
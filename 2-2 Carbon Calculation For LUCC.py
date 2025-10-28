import os
import numpy as np
from osgeo import gdal, osr

# Define input and output folder paths
input_folder = r'E:\梁老师scs\SCS数据【机革D盘】\MosaicCLCD'
output_folder = r'E:\梁老师scs\SCS数据【机革D盘】\Carbon Storage'


# Define the range of years
years = range(2018, 2025)

# Define carbon density dictionary, unit is kg/m2
carbon_density = {
    1: 0.071,  # Water
    2: 3.756,  # Forest
    4: 0.136,  # Flooded Vegetation
    5: 1.847,  # Cultivated land
    7: 0.058,  # Construction Area
    8: 0.096,  # Bare land
    9: 0.071,  # Snow
    10: 0,     # Cloud
    11: 2.571  # Grassland
}

# Iterate through each year's raster files
for year in years:
    input_file = os.path.join(input_folder, f'clcd{year}yanan.tif')
    output_file = os.path.join(output_folder, f'CwithoutS{year}.tif')

    # Open input raster
    dataset = gdal.Open(input_file)
    if not dataset:
        print(f"Cannot open: {input_file}")
        continue

    # Read raster data
    band = dataset.GetRasterBand(1)
    land_cover_array = band.ReadAsArray()

    # Create carbon storage raster array, unit kg
    carbon_storage_array = np.zeros_like(land_cover_array, dtype=np.float32)
    for land_type, density in carbon_density.items():
        carbon_storage_array[land_cover_array == land_type] = density * 10 * 10 / 1000000  # 10m x 10m, unit: thousand tons

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
        ['COMPRESS=LZW']  # Use LZW compression
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
    out_band.SetNoDataValue(-999)  # Set no data value
    out_band.FlushCache()

    # Clean up memory
    dataset = None
    out_dataset = None
    print(f"Carbon storage raster has been saved: {output_file}")
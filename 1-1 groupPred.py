import rasterio
import numpy as np
import joblib
import os
from rasterio.merge import merge

#============================== Part 1: Batch process 20 GeoTIFF files ==============================

# Load the model and class names
model = joblib.load("xgboost_model.pkl")
class_names = joblib.load("class_names.pkl")

# Define the input folder path
input_folder = r"xxx"

# Define the band names used by the training data
expected_band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12',
                       'Elevation', 'Slope', 'Aspect']

# Iterate through 20 files
for i in range(1, 21):
    # Dynamically generate input file paths
    input_tif_path = os.path.join(input_folder, f"{i:02d}.tif")  # 02d means two digits, e.g., 01, 02, ..., 20
    output_tif_path = os.path.join(input_folder, f"{i:02d}_predicted.tif")  # Add _predicted suffix to the output filename

    print(f"Processing file: {input_tif_path}")

    try:
        with rasterio.open(input_tif_path) as src:
            # Get the band names of the input file
            actual_band_names = src.descriptions

            # Check if the input file contains all expected bands
            missing_bands = [band for band in expected_band_names if band not in actual_band_names]
            if missing_bands:
                print(f"File {input_tif_path} is missing the following bands: {missing_bands}")
                continue  # Skip this file

            # Dynamically select the expected bands
            bands = []
            for band_name in expected_band_names:
                band_index = actual_band_names.index(band_name) + 1  # Band index starts from 1
                bands.append(src.read(band_index))

            # Stack band data into a feature matrix
            features = np.stack(bands, axis=-1)  # Shape is (height, width, num_features)
            height, width, num_features = features.shape

            # Flatten the feature matrix into a 2D array
            features_flat = features.reshape(-1, num_features)

            # Identify invalid values (NaN)
            valid_mask = ~np.any(np.isnan(features_flat), axis=1)  # Check if any band is NaN

            # Create a full prediction result array and initialize it to class 13 (Non_Vegetation)
            non_veg_class_index = 12  # Index of Non_Vegetation class (starts from 0, so 12 represents the 13th class)
            y_pred_full = np.full(height * width, fill_value=non_veg_class_index, dtype="uint8")

            # Predict on valid values
            if np.any(valid_mask):  # If there are valid values
                valid_features = features_flat[valid_mask]
                y_pred_valid = model.predict(valid_features)  # Predict only on valid pixels
                y_pred_full[valid_mask] = y_pred_valid  # Fill the valid prediction results into the corresponding positions

            # Reshape the prediction results to the original image shape
            y_pred_indices = y_pred_full.reshape(height, width)

            # Create a new GeoTIFF file to save the prediction results
            meta = src.meta.copy()
            meta.update({
                "count": 1,  # Only one band
                "dtype": "uint8",  # Use uint8 to store class indices
                "nodata": 255,  # Set a nodata value (optional)
            })

            with rasterio.open(output_tif_path, "w", **meta) as dst:
                # Write the prediction results to the label band
                dst.write(y_pred_indices, 1)
                dst.set_band_description(1, "label")  # Set the band description to "label"

        print(f"Prediction results saved to: {output_tif_path}")

    except Exception as e:
        print(f"Error processing file {input_tif_path}: {e}")
        continue  # Skip this file and continue with the next one

# Print the mapping of class indices to vegetation names
print("Mapping of class index to vegetation name:")
for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")
print(f"12: Non_Vegetation")  # Add the 13th class, Non_Vegetation

# ============================== Part 2: Merge the 20 prediction result files ==============================

# Get all predicted file paths
predicted_files = [os.path.join(input_folder, f"{i:02d}_predicted.tif") for i in range(1, 21)]

# Check if files exist
predicted_files = [f for f in predicted_files if os.path.exists(f)]
if not predicted_files:
    raise FileNotFoundError("No prediction result files found!")

# Open all prediction result files
src_files_to_mosaic = [rasterio.open(f) for f in predicted_files]

# Merge the files
mosaic, out_trans = merge(src_files_to_mosaic)

# Get the metadata for the output file
out_meta = src_files_to_mosaic[0].meta.copy()

# Update the metadata
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "nodata": 255,  # Set nodata value
})

# Define the output file path
output_mosaic_path = os.path.join(input_folder, "2018_1_mosaic_predicted.tif")

# Save the merged file
with rasterio.open(output_mosaic_path, "w", **out_meta) as dest:
    dest.write(mosaic)

print(f"Merging complete, result saved to: {output_mosaic_path}")



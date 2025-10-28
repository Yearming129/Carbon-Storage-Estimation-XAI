import os
import numpy as np
import pandas as pd
from osgeo import gdal

# Define input and output paths
input_folder = r'E:\梁老师scs\MaskRS'
output_csv = r'E:\梁老师scs\carbon_storage_uncertainty_results.csv'

# Define carbon density dictionary and its uncertainty information
carbon_density_info = {
    0: {"mean": 0.135057, "type": "range", "min": 0.009756, "max": 0.135057},
    1: {"mean": 0.211357, "type": "range", "min": 0.013573, "max": 0.211357},
    2: {"mean": 0.69, "type": "std", "std": 0.196},
    3: {"mean": 0.284, "type": "std", "std": 0.056},
    4: {"mean": 5.846, "type": "std", "std": 0.316},
    5: {"mean": 1.303, "type": "std", "std": 0.223},
    6: {"mean": 4.547, "type": "std", "std": 0.909},  # Populus tomentosa, using 20% coefficient of variation
    7: {"mean": 0.4317, "type": "range", "min": 0.4182, "max": 0.9652},
    8: {"mean": 0.2968, "type": "range", "min": 0.0908, "max": 0.3899},
    9: {"mean": 0.284, "type": "std", "std": 0.075},
    10: {"mean": 0.07149, "type": "range", "min": 0.04538, "max": 0.07146},
    11: {"mean": 0.1565, "type": "std", "std": 0.0133},
    12: {"mean": 0, "type": "fixed"}  # Non-vegetation, fixed at 0
}

# Monte Carlo simulation parameters
n_iterations = 1000  # Number of simulations
confidence_level = 0.95  # Confidence level

# Define years and quarters
years = range(2018, 2025)
quarters = [1, 2, 3, 4]

# List to store results
results = []

# Iterate through years and quarters
for year in years:
    for quarter in quarters:
        # Define input file path
        input_file = os.path.join(input_folder, f'{year}_{quarter}_mosaicMask.tif')

        # Check if the file exists
        if not os.path.exists(input_file):
            print(f"File does not exist: {input_file}")
            continue

        # Open the input raster file
        dataset = gdal.Open(input_file)
        if not dataset:
            print(f"Cannot open file: {input_file}")
            continue

        # Read raster data
        band = dataset.GetRasterBand(1)
        mask_array = band.ReadAsArray()

        # Count pixels for each vegetation type
        pixel_counts = {}
        total_pixels = 0

        for code in carbon_density_info.keys():
            count = np.sum(mask_array == code)
            pixel_counts[code] = count
            total_pixels += count

        # Perform Monte Carlo simulation
        carbon_storage_iterations = []

        for i in range(n_iterations):
            total_carbon_storage = 0

            # Generate random carbon density values for each vegetation type
            for code, info in carbon_density_info.items():
                count = pixel_counts[code]

                if info["type"] == "fixed":
                    # Fixed value
                    density = info["mean"]
                elif info["type"] == "std":
                    # With standard deviation, use normal distribution
                    density = np.random.normal(info["mean"], info["std"])
                    # Ensure carbon density is not negative
                    density = max(0, density)
                elif info["type"] == "range":
                    # With extreme values, use uniform distribution
                    density = np.random.uniform(info["min"], info["max"])

                # Calculate the carbon storage for this vegetation type (unit: thousand tons)
                # Each pixel area = 10m × 10m = 100 square meters
                # Carbon storage = pixel count × carbon density × pixel area / 1000000 (kg to thousand tons)
                carbon_storage = count * density * 100 / 1000000
                total_carbon_storage += carbon_storage

            carbon_storage_iterations.append(total_carbon_storage)

        # Calculate statistical metrics
        carbon_array = np.array(carbon_storage_iterations)
        mean_carbon = np.mean(carbon_array)
        std_carbon = np.std(carbon_array)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_bound = np.percentile(carbon_array, alpha / 2 * 100)
        upper_bound = np.percentile(carbon_array, (1 - alpha / 2) * 100)

        # Calculate coefficient of variation
        cv = std_carbon / mean_carbon if mean_carbon > 0 else 0

        # Add to results list
        results.append({
            "Year": year,
            "Quarter": quarter,
            "Mean_Carbon_Storage": mean_carbon,
            "Std_Carbon_Storage": std_carbon,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Coefficient_of_Variation": cv,
            "Total_Pixels": total_pixels
        })

        print(f"Completed uncertainty analysis for Year {year} Quarter {quarter}")

        # Free memory
        dataset = None

# Create results DataFrame
results_df = pd.DataFrame(results)

# Sort by year and quarter
results_df = results_df.sort_values(["Year", "Quarter"]).reset_index(drop=True)

# Save results to CSV file
results_df.to_csv(output_csv, index=False)
print(f"Uncertainty analysis results saved to: {output_csv}")

# Print summary statistics
print("\nCarbon storage uncertainty analysis summary:")
print(f"Number of simulations: {n_iterations}")
print(f"Confidence level: {confidence_level * 100}%")
print(f"Mean carbon storage: {results_df['Mean_Carbon_Storage'].mean():.2f} thousand tons")
print(f"Mean coefficient of variation: {results_df['Coefficient_of_Variation'].mean():.4f}")
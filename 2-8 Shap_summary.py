import os
import glob
import shap
import pandas as pd
import matplotlib.pyplot as plt


# Automatically generate a list of quarters
def generate_quarters(start_year, start_quarter, end_year, end_quarter):
    quarters = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            if (year == start_year and quarter < start_quarter) or (year == end_year and quarter > end_quarter):
                continue
            quarters.append((year, quarter))
    return quarters


# Map SHAP feature names to custom display names
def map_feature_names(shap_columns):
    feature_name_mapping = {
        "SHAP_lat": "Latitude",
        "SHAP_long": "Longitude",
        "SHAP_slope": "Slope",
        "SHAP_relief": "Relief",
        "SHAP_aspect": "Aspect",
        "SHAP_DEM": "DEM",
        "SHAP_road": "Distance from road",
        "SHAP_soil": "Soil type",
    }

    # Handle prefix mapping
    prefix_mapping = {
        "SHAP_ndvi_": "NDVI",
        "SHAP_clcd_": "LULC",
        "SHAP_tem_": "Temperature",
        "SHAP_pre_": "Precipitation",
        "SHAP_Dist_": "Landscape Disturb",
        "SHAP_ntl_": "Night light",
    }

    mapped_names = []
    for col in shap_columns:
        # First check if in fixed mapping
        if col in feature_name_mapping:
            mapped_names.append(feature_name_mapping[col])
        else:
            # Check for prefix matches
            mapped = False
            for prefix, new_name in prefix_mapping.items():
                if col.startswith(prefix):
                    mapped_names.append(new_name)
                    mapped = True
                    break
            if not mapped:
                mapped_names.append(col)  # Keep original name
    return mapped_names


# Plot and save SHAP figures
def plot_shap_figures(shap_file, output_folder, year, quarter, tick_fontsize=12, label_fontsize=14):
    try:
        # Load SHAP data
        data_with_shap = pd.read_csv(shap_file, encoding='utf-8')

        # Automatically extract SHAP feature columns
        shap_features = [col for col in data_with_shap.columns if col.startswith('SHAP_')]

        if not shap_features:
            print(f"No SHAP columns found in file {shap_file}, skipping...")
            return

        # Feature set and SHAP values set
        X = data_with_shap[shap_features]
        shap_values = data_with_shap[shap_features]  # SHAP value columns are the same as feature columns

        # Standardize feature names
        mapped_feature_names = map_feature_names(shap_features)

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Calculate the percentage contribution of each feature to the total absolute mean
        mean_abs_shap = shap_values.abs().mean(axis=0)
        total_mean_abs_shap = mean_abs_shap.sum()
        percentage_contributions = (mean_abs_shap / total_mean_abs_shap * 100).round(2)

        # Print percentage information
        print(f"=== {year} Q{quarter} Feature Contribution Percentages ===")
        for feature, percentage in zip(mapped_feature_names, percentage_contributions):
            print(f"{feature}: {percentage}%")

        # # 1. Plot local summary plot
        # plt.figure(figsize=(10, 16))
        # shap.summary_plot(
        #     shap_values.values,
        #     X,
        #     feature_names=mapped_feature_names,
        #     max_display=14,
        #     show=False
        # )
        # plt.title(f'SHAP Summary Plot (Local) - {year} Q{quarter}', fontsize=label_fontsize)
        # plt.xticks(fontsize=tick_fontsize)
        # plt.yticks(fontsize=tick_fontsize)
        # plt.tight_layout()
        # summary_output_file = os.path.join(output_folder, f'shap_summary_local_{year}_Q{quarter}.png')
        # plt.savefig(summary_output_file, dpi=300)
        # plt.close()
        # print(f"Local SHAP Summary plot saved to: {summary_output_file}")
        #
        # # 2. Plot global bar chart
        # plt.figure(figsize=(10, 16))
        # shap.summary_plot(
        #     shap_values.values,
        #     X,
        #     feature_names=mapped_feature_names,
        #     plot_type="bar",
        #     max_display=14,
        #     show=False
        # )
        # plt.title(f'SHAP Summary Plot (Global) - {year} Q{quarter}', fontsize=label_fontsize)
        # plt.xticks(fontsize=tick_fontsize)
        # plt.yticks(fontsize=tick_fontsize)
        # plt.tight_layout()
        # bar_output_file = os.path.join(output_folder, f'shap_summary_global_{year}_Q{quarter}.png')
        # plt.savefig(bar_output_file, dpi=300)
        # plt.close()
        # print(f"Global SHAP bar chart saved to: {bar_output_file}")

    except Exception as e:
        print(f"Error processing file {shap_file}: {e}")


# Main program
def main():
    # Input and output folder paths
    input_folder = r'D:\XAI dataset (100m)\table_1\SHAP\initial csv'  # SHAP results folder
    output_folder = os.path.join(input_folder, 'Figure')  # Chart output folder

    # Set start and end quarters
    start_year, start_quarter = 2018, 4
    end_year, end_quarter = 2024, 4

    # Automatically generate a list of quarters
    quarters = generate_quarters(start_year, start_quarter, end_year, end_quarter)

    # Iterate through each quarter
    for year, quarter in quarters:
        # Construct SHAP file path
        shap_file = os.path.join(input_folder, f'Shap_{year}_{quarter}_modified.csv')

        if os.path.exists(shap_file):
            print(f"Processing {shap_file}...")
            plot_shap_figures(
                shap_file,
                output_folder,
                year,
                quarter,
                tick_fontsize=16,  # Font size for ticks
                label_fontsize=14  # Font size for title
            )
        else:
            print(f"File not found: {shap_file}")


# Run main() if this is the main program
if __name__ == "__main__":
    main()
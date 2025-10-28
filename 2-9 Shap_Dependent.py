import os
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Prevent GUI pop-ups
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


# Plot and save SHAP dependence plots
def plot_shap_dependence(real_file, shap_file, output_folder, year, quarter):
    try:
        # Load feature values table
        X = pd.read_csv(real_file, encoding='utf-8')

        # Load SHAP values table
        shap_values_df = pd.read_csv(shap_file, encoding='utf-8')

        # Get feature names
        features = X.columns.tolist()
        shap_features = shap_values_df.columns.tolist()

        # Print feature names from both tables
        print(f"\nProcessing Year {year} Q{quarter}:")
        print("Feature names from feature values table (X):")
        print(features)
        print("\nFeature names from SHAP values table:")
        print(shap_features)

        # Convert to numpy array (SHAP library usually requires this format)
        shap_values = shap_values_df.values

        # Create a specific output folder for the quarter
        quarter_output_folder = os.path.join(output_folder, f"{year}_Q{quarter}")
        os.makedirs(quarter_output_folder, exist_ok=True)

        # Set the prefixes of features to plot *********
        selected_prefixes = ['DEM', 'ndvi_', 'lat', 'long']
        selected_features = [feature for feature in features if any(feature.startswith(prefix) for prefix in selected_prefixes)]
        print(f"Selected features: {selected_features}")

        # Set the aspect ratio of the figure (closer to square)
        figure_size = (8, 7)

        # Plot dependence plots
        for i, feature in enumerate(selected_features):
            try:
                # Create figure and axes, with dimensions set to square
                fig, ax = plt.subplots(figsize=figure_size)

                # Use feature name or index i and pass the complete SHAP values matrix
                shap.dependence_plot(
                    feature,  # Feature name
                    shap_values,  # Complete SHAP values matrix
                    X,  # Original feature data
                    display_features=X,
                    interaction_index=f'cs_{year}_{quarter}',
                    ax=ax,  # Specify the axes to plot on
                    show=False  # Do not display the figure immediately to avoid saving conflicts
                )

                # Set title, axes labels, and tick font size
                # ax.set_title(f"Feature '{feature}' ({year} Q{quarter})", fontsize=16)
                # ax.set_xlabel(feature, fontsize=24)
                # ax.set_ylabel("SHAP Value", fontsize=24)
                ax.tick_params(axis='both', which='major', labelsize=24)

                # Save the figure
                save_path = os.path.join(quarter_output_folder, f"dependence_plot_{feature}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)  # Explicitly close the figure
                print(f"SHAP dependence plot saved to: {save_path}")
            except Exception as e:
                print(f"Error generating plot for feature '{feature}': {e}")

    except Exception as e:
        print(f"Error processing Year {year} Q{quarter}: {e}")


# Main program
def main():
    # Input folder paths
    real_value_folder = r'D:\XAI dataset (100m)\table_1\SHAP\real value'  # True value folder
    shap_value_folder = r'D:\XAI dataset (100m)\table_1\SHAP\shap value'  # SHAP value folder
    output_folder = r'D:\XAI dataset (100m)\table_1\SHAP\Dependence_Plots'  # Chart output folder

    # Set start and end quarters
    start_year, start_quarter = 2018, 4
    end_year, end_quarter = 2024, 4

    # Automatically generate a list of quarters
    quarters = generate_quarters(start_year, start_quarter, end_year, end_quarter)

    # Iterate through each quarter
    for year, quarter in quarters:
        # Construct true value and SHAP file paths
        real_file = os.path.join(real_value_folder, f'Shap_{year}_{quarter}_modified.csv')
        shap_file = os.path.join(shap_value_folder, f'Shap_{year}_{quarter}_modified_reordered.csv')

        # Check if files exist
        if os.path.exists(real_file) and os.path.exists(shap_file):
            print(f"\nProcessing Year {year} Q{quarter}...")
            plot_shap_dependence(real_file, shap_file, output_folder, year, quarter)
        else:
            print(f"Files not found: {real_file} or {shap_file}")


# Run main() if this is the main program
if __name__ == "__main__":
    main()
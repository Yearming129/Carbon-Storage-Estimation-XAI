import dask
import dask.dataframe as dd
from dask.distributed import Client
import xgboost as xgb
import shap
import pandas as pd
import os
import psutil  # For memory monitoring
from tqdm import tqdm  # For progress bar


# Automatically generate a list of quarters
def generate_quarters(start_year, start_quarter, end_year, end_quarter):
    quarters = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            if (year == start_year and quarter < start_quarter) or (year == end_year and quarter > end_quarter):
                continue
            quarters.append((year, quarter))
    return quarters


# SHAP value calculation logic
def compute_shap_values(file_path, model_path, output_dir, year, quarter):
    try:
        print(f"Processing data for Year {year}, Quarter {quarter}...")

        # 1. Load data
        data = pd.read_csv(file_path, encoding='utf-8')

        # 2. Data preprocessing
        X = data[['aspect', 'DEM', 'relief', 'slope', 'long', 'lat', 'soil',
                  f'clcd_{year}', f'road_{year}', f'Dist_{year}', f'ntl_{year}',
                  f'tem_{year}_{quarter}', f'pre_{year}_{quarter}', f'ndvi_{year}_{quarter}']]
        y = data[f'cs_{year}_{quarter}']

        # Retain original OID number
        if 'ID_1' in data.columns:
            OID = data['ID_1']
        else:
            raise ValueError("The original data does not contain the 'ID_1' column!")

        # 3. Load model
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(model_path)
        print("Model loaded successfully!")

        # 5. SHAP interpretability analysis
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create Dask DataFrame
        X_dask = dd.from_pandas(X, npartitions=8)

        # Define a function to compute SHAP values in parallel
        def compute_shap(chunk, model):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(chunk, approximate=True)
                return shap_values, explainer.expected_value
            except Exception as e:
                print(f"Error computing SHAP values: {e}")
                return None, None

        # Use Dask to compute SHAP values in parallel
        shap_values_delayed = X_dask.map_partitions(lambda chunk: compute_shap(chunk, xgb_model)).to_delayed()

        # Calculate SHAP values and monitor memory usage
        with tqdm(total=len(shap_values_delayed)) as pbar:
            def callback(future):
                pbar.update(1)

            with dask.config.set({"distributed.scheduler.work-stealing": False}):
                results = dask.compute(*shap_values_delayed, _concurrent=True, _callback=callback)

        mem_info = psutil.virtual_memory()
        print(f"Current memory usage: {mem_info.percent}%")

        # Merge SHAP values and expected_value
        shap_values_list = [res[0] for res in results]
        expected_values = [res[1] for res in results]

        # Calculate final SHAP values
        shap_values_full = pd.concat([pd.DataFrame(shap_values, columns=X.columns) for shap_values in shap_values_list],
                                     axis=0)

        # Add prefix 'SHAP_' to SHAP value column names
        shap_values_full.columns = [f"SHAP_{col}" for col in shap_values_full.columns]

        # Save SHAP values to CSV file
        shap_values_df = pd.DataFrame(shap_values_full)
        shap_values_df['ID_1'] = OID  # Retain original OID number

        # Ensure index consistency
        data = data.reset_index(drop=True)
        shap_values_df = shap_values_df.reset_index(drop=True)

        # Merge SHAP values with original data
        output_csv_path = os.path.join(output_dir, f'Shap_{year}_{quarter}_modified.csv')
        merged_data = pd.concat([data, shap_values_df], axis=1)
        merged_data.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"SHAP values saved to: {output_csv_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# Main program
def main():
    # Set start and end quarters
    start_year, start_quarter = 2018, 4
    end_year, end_quarter = 2024, 4

    # Automatically generate a list of quarters
    quarters = generate_quarters(start_year, start_quarter, end_year, end_quarter)

    # Output directory
    shap_output_dir = r'D:\XAI dataset (100m)\table_1\SHAP'

    for year, quarter in quarters:
        # Construct file path and model path
        file_path = rf'D:\XAI dataset (100m)\table_1\modified\Q_{year}_{quarter}_modified.csv'
        model_path = rf'D:\XAI dataset (100m)\table_1\training\Q_{year}_{quarter}_modified.json'

        # Check if files exist
        if os.path.exists(file_path) and os.path.exists(model_path):
            compute_shap_values(file_path, model_path, shap_output_dir, year, quarter)
        else:
            print(f"File or model missing: {file_path}, {model_path}")


# Run main() if this is the main program
if __name__ == '__main__':
    # Initialize Dask client
    client = Client(n_workers=8, threads_per_worker=1, dashboard_address=":0")  # Adjust parameters based on hardware
    print(client)

    main()

    # Close Dask client
    client.close()
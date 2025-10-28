# 1. Import required libraries
import rasterio
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

# 2. Define global variables (unchanged)
file_names = [
    "Sbu_S2_10m_processed.tif", "Sbr_S2_10m_processed.tif", "Rx_S2_10m_processed.tif",
    "Rp_S2_10m_processed.tif", "Ql_S2_10m_processed.tif", "Pt_poplar_S2_10m_processed.tif",
    "Pt_oil_S2_10m_processed.tif", "Pd_S2_10m_processed.tif", "Lb_S2_10m_processed.tif",
    "Ck_S2_10m_processed.tif", "Bi_S2_10m_processed.tif", "As_S2_10m_processed.tif"
]
feature_descriptions = {
    "B1": "B1", "B2": "B2", "B3": "B3", "B4": "B4", "B5": "B5", "B6": "B6",
    "B7": "B7", "B8": "B8", "B8A": "B8A", "B9": "B9", "B11": "B11", "B12": "B12",
    "Elevation": "Elevation", "Slope": "Slope", "Aspect": "Aspect"
}


# 3. Helper functions (unchanged)
def find_band_index(src, target_description):
    """Finds the band index corresponding to a description."""
    for idx, description in enumerate(src.descriptions, start=1):
        if description == target_description:
            return idx
    return None


def get_vegetation_class(file_name):
    """Extracts the vegetation class name from the filename."""
    if file_name.startswith("Pt_"):
        if "oil" in file_name:
            return "Pt_oil"
        elif "poplar" in file_name:
            return "Pt_poplar"
    return file_name.split("_")[0]


# 4. Data processing functions for Spatial CV (unchanged)
def extract_and_split_samples_spatially(file_path, num_total_samples, k_folds):
    """
    Extracts samples from a single file and spatially splits them into k folds
    based on their row index (latitude), creating horizontal bands.
    """
    with rasterio.open(file_path) as src:
        label_index = find_band_index(src, "label")
        if label_index is None: return pd.DataFrame()

        label_data = src.read(label_index)
        valid_indices = np.where(label_data == 1)

        if len(valid_indices[0]) < num_total_samples:
            print(f"  Warning: Not enough samples in {file_path}. Using all {len(valid_indices[0])} available samples.")
            num_total_samples = len(valid_indices[0])

        random_choice = np.random.choice(len(valid_indices[0]), num_total_samples, replace=False)
        rows, cols = valid_indices[0][random_choice], valid_indices[1][random_choice]

        features_list = []
        for desc in feature_descriptions.values():
            band_index = find_band_index(src, desc)
            if band_index is None: return pd.DataFrame()
            band_data = src.read(band_index)
            features_list.append(band_data[rows, cols])

        feature_array = np.column_stack(features_list)

        nan_mask = ~np.isnan(feature_array).any(axis=1)
        feature_array = feature_array[nan_mask]
        rows = rows[nan_mask]

        min_row, max_row = np.min(rows), np.max(rows)
        fold_bins = np.linspace(min_row, max_row, k_folds + 1)

        fold_ids = np.digitize(rows, bins=fold_bins, right=True) - 1
        fold_ids = np.clip(fold_ids, 0, k_folds - 1)

        df = pd.DataFrame(feature_array, columns=feature_descriptions.values())
        df['fold'] = fold_ids
        df['vegetation'] = get_vegetation_class(file_path)

        return df


def create_spatial_folds(file_names, num_samples_per_class=5000, k_folds=5):
    """
    Creates k spatially distinct cross-validation folds by combining
    corresponding spatial bands from each class file.
    """
    print(f"Creating {k_folds} spatial cross-validation folds...")
    all_data = []
    for file_name in file_names:
        print(f"  Processing file: {file_name}")
        class_df = extract_and_split_samples_spatially(file_name, num_samples_per_class, k_folds)
        if not class_df.empty:
            all_data.append(class_df)

    if not all_data:
        print("Error: Could not extract data from any files.")
        return []

    full_dataset = pd.concat(all_data, ignore_index=True)

    list_of_fold_dfs = [
        full_dataset[full_dataset['fold'] == i].drop(columns=['fold'])
        for i in range(k_folds)
    ]

    print("Spatial folds created successfully.")
    return list_of_fold_dfs


# 5. --- Reworked model training and evaluation function ---
def run_model_comparison_spatial_cv(list_of_fold_dfs):
    """
    Performs spatial cross-validation for a dictionary of models and compares their performance.
    """
    k_folds = len(list_of_fold_dfs)

    # Get class information from the first fold
    temp_y = list_of_fold_dfs[0]['vegetation'].astype('category')
    class_names = temp_y.cat.categories.tolist()
    all_labels = list(range(len(class_names)))

    print("Class names and their corresponding integer encodings:")
    for idx, name in enumerate(class_names):
        print(f"{idx}: {name}")

    # --- Define all models to be compared ---
    models = {
        # 'Logistic Regression': Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('logreg', LogisticRegression(solver='saga', max_iter=1000, random_state=42, n_jobs=-1))
        # ]),
        # 'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        # 'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(
            objective='multi:softmax', num_class=len(class_names), max_depth=5,
            learning_rate=0.1, subsample=0.8, colsample_bytree=1.0, gamma=0,
            seed=42, n_estimators=500
        )
    }

    # Dictionary to store the final results for comparison
    comparison_results = {}

    # --- Main evaluation loop: iterate through each model ---
    for model_name, model in models.items():
        print(f"\n{'=' * 20} Evaluating Model: {model_name} {'=' * 20}")

        fold_metrics = []
        all_y_true = []
        all_y_pred = []

        # Inner loop: iterate through each spatial fold
        for i in range(k_folds):
            print(f"\n--- Fold {i + 1}/{k_folds} ---")

            # Prepare training and validation sets for the current fold
            val_df = list_of_fold_dfs[i]
            train_df = pd.concat([df for j, df in enumerate(list_of_fold_dfs) if i != j])

            X_train = train_df.drop(columns=['vegetation'])
            y_train_str = train_df['vegetation']

            X_val = val_df.drop(columns=['vegetation'])
            y_val_str = val_df['vegetation']

            # Encode string labels to integers
            y_train = y_train_str.astype('category').cat.set_categories(class_names).cat.codes
            y_val = y_val_str.astype('category').cat.set_categories(class_names).cat.codes

            # Train and predict
            model.fit(X_train, y_train)
            y_pred_fold = model.predict(X_val)

            # Calculate and print the four metrics for the fold
            acc = accuracy_score(y_val, y_pred_fold)
            f1 = f1_score(y_val, y_pred_fold, average='weighted', labels=all_labels)
            prec = precision_score(y_val, y_pred_fold, average='weighted', labels=all_labels, zero_division=0)
            rec = recall_score(y_val, y_pred_fold, average='weighted', labels=all_labels, zero_division=0)

            print(f"  Accuracy:  {acc:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")

            fold_metrics.append({'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec})
            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred_fold)

        # --- Summarize performance for the current model ---
        avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
        avg_f1 = np.mean([m['f1'] for m in fold_metrics])
        avg_prec = np.mean([m['precision'] for m in fold_metrics])
        avg_rec = np.mean([m['recall'] for m in fold_metrics])

        comparison_results[model_name] = {
            'Avg Accuracy': avg_acc,
            'Avg F1-Score': avg_f1,
            'Avg Precision': avg_prec,
            'Avg Recall': avg_rec
        }

        # --- Plot confusion matrix for the current model ---
        cm = confusion_matrix(all_y_true, all_y_pred, labels=all_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title(f"Overall Confusion Matrix for {model_name} (from Spatial CV)")
        plt.show()

    # --- Final comparison of all models ---
    print(f"\n{'=' * 25} Final Model Comparison {'=' * 25}")
    results_df = pd.DataFrame.from_dict(comparison_results, orient='index')
    print(results_df.round(4))

    # --- Identify and save the best model based on F1-score ---
    best_model_name = results_df['Avg F1-Score'].idxmax()
    print(f"\nBest performing model based on F1-Score: {best_model_name}")

    # Train the best model on the entire dataset
    print(f"Training the final '{best_model_name}' model on the entire dataset...")
    best_model = models[best_model_name]
    full_dataset_for_final_training = pd.concat(list_of_fold_dfs)
    X_final = full_dataset_for_final_training.drop(columns=['vegetation'])
    y_final_str = full_dataset_for_final_training['vegetation']
    y_final = y_final_str.astype('category').cat.set_categories(class_names).cat.codes

    best_model.fit(X_final, y_final)

    print("Saving the best model...")
    joblib.dump(best_model, "best_model_spatial_cv.pkl")
    joblib.dump(class_names, "class_names.pkl")
    print(f"Best model ('{best_model_name}') saved successfully as 'best_model_spatial_cv.pkl'")


# 6. Main execution block
if __name__ == "__main__":
    K_FOLDS = 5
    SAMPLES_PER_CLASS = 5000

    # 1. Create the spatial folds
    list_of_fold_dfs = create_spatial_folds(
        file_names,
        num_samples_per_class=SAMPLES_PER_CLASS,
        k_folds=K_FOLDS
    )

    # 2. Check if data was created successfully
    if list_of_fold_dfs and all(not df.empty for df in list_of_fold_dfs):
        # 3. Run the model comparison
        run_model_comparison_spatial_cv(list_of_fold_dfs)
    else:
        print("Dataset creation failed or resulted in empty folds. Halting execution.")
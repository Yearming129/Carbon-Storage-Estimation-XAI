import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import os
import logging
from datetime import datetime

# Configure logging
log_dir = r'D:\XAI dataset (100m)\table_1\training'
os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# Automatically generate quarterly data
def generate_quarters(start_year, start_quarter, end_year, end_quarter):
    quarters = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            if (year == start_year and quarter < start_quarter) or (year == end_year and quarter > end_quarter):
                continue
            quarters.append((year, quarter))
    return quarters


# XGBoost model training, evaluation, and cross-validation
def train_and_evaluate(file_path, year, quarter):
    logging.info(f"Processing {file_path}...")

    # Load data
    data = pd.read_csv(file_path)

    # Data preprocessing
    feature_prefix = f"{year}_{quarter}"
    X = data[['aspect', 'DEM', 'relief', 'slope', 'long', 'lat', 'soil',
              f'clcd_{year}', f'road_{year}', f'Dist_{year}', f'ntl_{year}',
              f'tem_{feature_prefix}', f'pre_{feature_prefix}', f'ndvi_{feature_prefix}']]
    y = data[f'cs_{feature_prefix}']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                 learning_rate=0.02,
                                 n_estimators=1800,
                                 max_depth=10,
                                 reg_lambda=22,
                                 reg_alpha=0,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 random_state=42)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Model evaluation
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    logging.info(f"XGBoost Regression - {year} Q{quarter}")
    logging.info(f"Training Set MSE: {mse_train:.4f}, R²: {r2_train:.4f}")
    logging.info(f"Test Set MSE: {mse_test:.4f}, R²: {r2_test:.4f}")

    # Feature importance analysis
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_model.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    logging.info("\nFeature Importance:")
    logging.info(importance_df.to_string(index=False))

    # Cross-validation
    logging.info("Performing cross-validation...")
    cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    logging.info(f"Cross-Validation MSE: {cv_mse:.4f}")

    # Save model
    table_name = os.path.splitext(os.path.basename(file_path))[0]
    model_path = os.path.join(log_dir, f"{table_name}.json")
    xgb_model.save_model(model_path)
    logging.info(f"Model saved as: {model_path}")


# Main program
if __name__ == "__main__":
    # Define start and end quarters
    start_year, start_quarter = 2024, 1
    end_year, end_quarter = 2024, 4

    # Automatically generate the list of quarters
    quarters = generate_quarters(start_year, start_quarter, end_year, end_quarter)

    for year, quarter in quarters:
        # Build file path
        file_path = rf'D:\XAI dataset (100m)\table_1\modified\Q_{year}_{quarter}_modified.csv'
        if os.path.exists(file_path):  # Check if the file exists
            try:
                train_and_evaluate(file_path, year, quarter)
            except Exception as e:
                logging.error(f"Error while processing {file_path}: {e}")
        else:
            logging.warning(f"File does not exist: {file_path}")
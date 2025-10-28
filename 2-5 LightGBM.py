import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load data
file_path = r'D:\XAI dataset (100m)\table_1\modified\Q_2019_1_modified.csv'
data = pd.read_csv(file_path)

# Extract table name (without extension)
table_name = os.path.basename(file_path).replace('.csv', '')

# 2. Data preprocessing
X = data[['aspect', 'DEM', 'relief', 'slope', 'long', 'lat', 'soil', 'clcd_2019', 'road_2019', 'Dist_2019', 'ntl_2019',
          'tem_2019_1', 'pre_2019_1', 'ndvi_2019_1'
          ]]
y = data['cs_2019_1']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build LightGBM model using lgb.train()
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 256,  # Adjust to a smaller value to avoid overfitting
    'max_depth': 10,
    'min_child_samples': 10,  # Reduce sample restriction to allow more leaf nodes
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'seed': 42
}

# Train the model
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=1800,
    valid_sets=[lgb_train, lgb_test],
    valid_names=["train", "test"],
    # early_stopping_rounds=50,  # Enable early stopping
    # verbose_eval=100  # Print log every 100 iterations
)

# 4. Model evaluation
y_pred_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("LightGBM Regression")
print(f"Training set MSE: {mse_train:.4f}, R²: {r2_train:.4f}")
print(f"Test set MSE: {mse_test:.4f}, R²: {r2_test:.4f}")

# 5. Feature importance analysis
importance = gbm.feature_importance(importance_type='gain')  # Use 'gain' type importance
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()  # Normalize to 0-1
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature importance proportions:")
print(importance_df)

# 6. Save the model
# Save folder path
model_folder = r'D:\XAI dataset (100m)\table_1\training'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Dynamically generate model filename
model_file_path = os.path.join(model_folder, f'lightgbm_{table_name}.json')

# Save model to JSON format
gbm.save_model(model_file_path, num_iteration=gbm.best_iteration)

print(f"\nModel has been saved to: {model_file_path}")
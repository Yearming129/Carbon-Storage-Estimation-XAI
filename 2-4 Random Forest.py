import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
file_path = r'D:\XAI dataset (100m)\table_1\modified\Q_2019_1_modified.csv'
data = pd.read_csv(file_path)

# 2. Data preprocessing
X = data[['aspect', 'DEM', 'relief', 'slope', 'long', 'lat', 'soil', 'clcd_2019', 'road_2019', 'Dist_2019', 'ntl_2019',
          'tem_2019_1', 'pre_2019_1', 'ndvi_2019_1'
          ]]
y = data['cs_2019_1']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build Random Forest model
rf = RandomForestRegressor(n_estimators=1800, max_depth=10, max_features='sqrt', min_samples_split=10, min_samples_leaf=5, bootstrap=True, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 4. Model evaluation
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Random Forest Regression")
print(f"Training set MSE: {mse_train:.4f}, R²: {r2_train:.4f}")
print(f"Test set MSE: {mse_test:.4f}, R²: {r2_test:.4f}")

# 5. Feature importance analysis
feature_importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature importance proportions:")
print(importance_df)
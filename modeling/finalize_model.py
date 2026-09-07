"""Train the final XGBoost model (tuned hyperparameters from tune_xgboost.py)
on the full train set and save it for reuse. Run from the repo root:
`python modeling/finalize_model.py`
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "data/processed"
MODELS_DIR = "models"

train = pd.read_csv(f"{DATA_DIR}/bike-sharing-train.csv", parse_dates=["date_time"])
test = pd.read_csv(f"{DATA_DIR}/bike-sharing-test.csv", parse_dates=["date_time"])

exog_cols = ["holiday", "workingday", "is_weekend", "weather_mist", "weather_rain",
             "atemp", "hum", "windspeed", "month_sin", "month_cos", "hour_sin", "hour_cos"]
lag_cols = ["users_lag_1", "users_lag_24", "users_lag_168"]
feature_cols = exog_cols + lag_cols

with open(f"{MODELS_DIR}/xgboost_best_params.json") as f:
    best_params = json.load(f)

print("Final hyperparameters (from RandomizedSearchCV, 5-fold time-series CV):")
for k, v in best_params.items():
    print(f"  {k}: {v}")

model = xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist", **best_params)
model.fit(train[feature_cols], train["users"])

pred = np.clip(model.predict(test[feature_cols]), 0, None)
mae = mean_absolute_error(test["users"], pred)
rmse = np.sqrt(mean_squared_error(test["users"], pred))
mape = np.mean(np.abs((test["users"] - pred) / test["users"])) * 100
r2 = r2_score(test["users"], pred)
print(f"\nHeld-out test  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%  R2={r2:.4f}")

model.save_model(f"{MODELS_DIR}/bike_xgboost_final.json")
with open(f"{MODELS_DIR}/bike_xgboost_final_features.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"\nSaved -> {MODELS_DIR}/bike_xgboost_final.json (model), "
      f"{MODELS_DIR}/bike_xgboost_final_features.json (feature order)")
print("\nTo reuse:")
print("  model = xgb.XGBRegressor()")
print(f"  model.load_model('{MODELS_DIR}/bike_xgboost_final.json')")
print(f"  model.predict(df[feature_cols])  # feature_cols from {MODELS_DIR}/bike_xgboost_final_features.json, same order")

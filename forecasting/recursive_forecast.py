"""Pure recursive multi-step forecast: from a single forecast origin (the end
of train), predict the entire 3,509h test horizon by feeding each prediction
back in as users_lag_1 for the next step - no re-anchoring to real data.
Quantifies how fast error compounds. Run from the repo root:
`python forecasting/recursive_forecast.py` (needs finalize_model.py to have
run first).
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "forecasting/results"

train = pd.read_csv(f"{DATA_DIR}/bike-sharing-train.csv", parse_dates=["date_time"])
test = pd.read_csv(f"{DATA_DIR}/bike-sharing-test.csv", parse_dates=["date_time"])

exog_cols = ["holiday", "workingday", "is_weekend", "weather_mist", "weather_rain",
             "atemp", "hum", "windspeed", "month_sin", "month_cos", "hour_sin", "hour_cos"]
LAGS = [1, 24, 168]
feature_cols = exog_cols + [f"users_lag_{l}" for l in LAGS]

with open(f"{MODELS_DIR}/bike_xgboost_final_features.json") as f:
    saved_order = json.load(f)
assert saved_order == feature_cols, "feature order mismatch with the finalized model"

model = xgb.XGBRegressor()
model.load_model(f"{MODELS_DIR}/bike_xgboost_final.json")

# ---------- seed history with the TRUE tail of train (need 168h back) ----------
history = list(train["users"].values[-max(LAGS):])  # last 168 true values, chronological
assert len(history) == max(LAGS)

recursive_preds = np.empty(len(test), dtype="float64")

for i in range(len(test)):
    row = test.iloc[i]
    feat = {c: row[c] for c in exog_cols}
    feat["users_lag_1"] = history[-1]
    feat["users_lag_24"] = history[-24]
    feat["users_lag_168"] = history[-168]
    X = pd.DataFrame([feat])[feature_cols]

    yhat = max(0.0, float(model.predict(X)[0]))
    recursive_preds[i] = yhat
    history.append(yhat)  # feed the PREDICTION back in, not the true value

y_true = test["users"].values

# ---------- one-step-ahead predictions for comparison (true lags, from before) ----------
X_test_true = test[feature_cols]
onestep_preds = np.clip(model.predict(X_test_true), 0, None)


def evaluate(name, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{name:>28}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:6.2f}%  R2={r2:.4f}")
    return mae, rmse, mape, r2


print("=== Full test horizon (3,509h = ~146 days ahead, single forecast origin) ===")
evaluate("One-step-ahead (true lags)", onestep_preds)
evaluate("Recursive (predicted lags)", recursive_preds)

# ---------- error growth by forecast horizon (days since origin) ----------
horizon_days = np.arange(len(test)) // 24
print("\n=== RMSE by forecast horizon (days since forecast origin) ===")
rows = []
for d in range(0, horizon_days.max() + 1, 7):  # weekly buckets
    mask = (horizon_days >= d) & (horizon_days < d + 7)
    if mask.sum() == 0:
        continue
    rmse_1s = np.sqrt(mean_squared_error(y_true[mask], onestep_preds[mask]))
    rmse_rec = np.sqrt(mean_squared_error(y_true[mask], recursive_preds[mask]))
    rows.append({"week_range": f"day {d}-{d+6}", "n_hours": mask.sum(),
                 "RMSE_onestep": rmse_1s, "RMSE_recursive": rmse_rec})
horizon_df = pd.DataFrame(rows)
print(horizon_df.round(1).to_string(index=False))

horizon_df.to_csv(f"{RESULTS_DIR}/recursive_horizon_degradation.csv", index=False)
np.savez(f"{RESULTS_DIR}/recursive_forecast_predictions.npz",
         ts_test=test["date_time"].values, y_test=y_true,
         onestep=onestep_preds, recursive=recursive_preds)
print(f"\nSaved -> {RESULTS_DIR}/recursive_horizon_degradation.csv, "
      f"{RESULTS_DIR}/recursive_forecast_predictions.npz")

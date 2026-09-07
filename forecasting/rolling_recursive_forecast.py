"""Rolling recursive forecast: re-anchor to real data every HORIZON hours
instead of recursing blindly for the whole test set (see
recursive_forecast.py). Simulates "forecast the next day, then get today's
real numbers and forecast again." Run from the repo root:
`python forecasting/rolling_recursive_forecast.py` (needs
recursive_forecast.py to have run first, for the long-recursive comparison).
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HORIZON = 24  # re-anchor to true data every 24h - "forecast tomorrow, then get today's real numbers"

DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "forecasting/results"

train = pd.read_csv(f"{DATA_DIR}/bike-sharing-train.csv", parse_dates=["date_time"])
test = pd.read_csv(f"{DATA_DIR}/bike-sharing-test.csv", parse_dates=["date_time"])

exog_cols = ["holiday", "workingday", "is_weekend", "weather_mist", "weather_rain",
             "atemp", "hum", "windspeed", "month_sin", "month_cos", "hour_sin", "hour_cos"]
LAGS = [1, 24, 168]
feature_cols = exog_cols + [f"users_lag_{l}" for l in LAGS]

model = xgb.XGBRegressor()
model.load_model(f"{MODELS_DIR}/bike_xgboost_final.json")

full_true = np.concatenate([train["users"].values, test["users"].values])
n_train = len(train)

preds = np.empty(len(test), dtype="float64")
step_within_block = np.empty(len(test), dtype="int64")

for origin in range(0, len(test), HORIZON):
    block_len = min(HORIZON, len(test) - origin)
    # history known TRUE up to just before this block (re-anchored every HORIZON hours)
    hist = list(full_true[: n_train + origin])
    block_preds = []
    for step in range(block_len):
        combined_tail = hist if not block_preds else hist + block_preds
        lag1 = combined_tail[-1]
        lag24 = combined_tail[-24]
        lag168 = combined_tail[-168]
        row = test.iloc[origin + step]
        feat = {c: row[c] for c in exog_cols}
        feat["users_lag_1"], feat["users_lag_24"], feat["users_lag_168"] = lag1, lag24, lag168
        X = pd.DataFrame([feat])[feature_cols]
        yhat = max(0.0, float(model.predict(X)[0]))
        block_preds.append(yhat)
    preds[origin:origin + block_len] = block_preds
    step_within_block[origin:origin + block_len] = np.arange(1, block_len + 1)

y_true = test["users"].values
onestep_preds = np.clip(model.predict(test[feature_cols]), 0, None)
long_recursive = np.load(f"{RESULTS_DIR}/recursive_forecast_predictions.npz")["recursive"]


def evaluate(name, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{name:>38}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:6.2f}%  R2={r2:.4f}")


print(f"=== Rolling recursive, re-anchored every {HORIZON}h, vs. the two extremes ===")
evaluate("One-step-ahead (true lags always)", onestep_preds)
evaluate(f"Rolling recursive (re-anchor {HORIZON}h)", preds)
evaluate("Pure recursive (1 origin, 146 days)", long_recursive)

# ---------- error growth WITHIN each 24h block (step 1 vs step 24) ----------
print(f"\n=== RMSE by step-within-block (1..{HORIZON}h since last re-anchor) ===")
rows = []
for s in range(1, HORIZON + 1):
    mask = step_within_block == s
    rmse = np.sqrt(mean_squared_error(y_true[mask], preds[mask]))
    mae = mean_absolute_error(y_true[mask], preds[mask])
    rows.append({"step_ahead_h": s, "RMSE": rmse, "MAE": mae, "n": mask.sum()})
step_df = pd.DataFrame(rows)
print(step_df.round(2).to_string(index=False))

step_df.to_csv(f"{RESULTS_DIR}/rolling_recursive_step_degradation.csv", index=False)
np.savez(f"{RESULTS_DIR}/rolling_recursive_predictions.npz", ts_test=test["date_time"].values,
         y_test=y_true, onestep=onestep_preds, rolling_recursive=preds,
         long_recursive=long_recursive, step_within_block=step_within_block)
print(f"\nSaved -> {RESULTS_DIR}/rolling_recursive_step_degradation.csv, "
      f"{RESULTS_DIR}/rolling_recursive_predictions.npz")

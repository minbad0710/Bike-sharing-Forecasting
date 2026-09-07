"""Sweep the rolling-recursive re-anchor horizon from 1h to 168h to trace out
the accuracy/re-anchor-frequency trade-off curve. Run from the repo root:
`python forecasting/sweep_reanchor_horizon.py`
"""
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

model = xgb.XGBRegressor()
model.load_model(f"{MODELS_DIR}/bike_xgboost_final.json")

full_true = np.concatenate([train["users"].values, test["users"].values])
n_train = len(train)
y_true = test["users"].values


def rolling_recursive(horizon):
    preds = np.empty(len(test), dtype="float64")
    for origin in range(0, len(test), horizon):
        block_len = min(horizon, len(test) - origin)
        hist = list(full_true[: n_train + origin])
        block_preds = []
        for step in range(block_len):
            tail = hist if not block_preds else hist + block_preds
            feat = {c: test.iloc[origin + step][c] for c in exog_cols}
            feat["users_lag_1"], feat["users_lag_24"], feat["users_lag_168"] = tail[-1], tail[-24], tail[-168]
            X = pd.DataFrame([feat])[feature_cols]
            yhat = max(0.0, float(model.predict(X)[0]))
            block_preds.append(yhat)
        preds[origin:origin + block_len] = block_preds
    return preds


results = []
for horizon in [1, 3, 6, 12, 24, 48, 168]:
    preds = rolling_recursive(horizon)
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
    r2 = r2_score(y_true, preds)
    results.append({"reanchor_every_h": horizon, "MAE": mae, "RMSE": rmse, "MAPE%": mape, "R2": r2})
    print(f"reanchor={horizon:>4}h  MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:6.2f}%  R2={r2:.4f}")

df = pd.DataFrame(results)
df.to_csv(f"{RESULTS_DIR}/reanchor_horizon_sweep.csv", index=False)
print(f"\nSaved -> {RESULTS_DIR}/reanchor_horizon_sweep.csv")

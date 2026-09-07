"""Direct multi-step forecast: train one model per horizon h=1..24, with every
feature defined relative to the TARGET so it is always known data (no
recursion at all). Compares against rolling_recursive_forecast.py on the same
24h-block origins. Run from the repo root:
`python forecasting/direct_multistep_forecast.py` (needs
rolling_recursive_forecast.py to have run first).
"""
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HORIZON = 24  # same block length as the rolling-recursive-24h experiment, for a fair comparison

DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "forecasting/results"

train = pd.read_csv(f"{DATA_DIR}/bike-sharing-train.csv", parse_dates=["date_time"])
test = pd.read_csv(f"{DATA_DIR}/bike-sharing-test.csv", parse_dates=["date_time"])

exog_cols = ["holiday", "workingday", "is_weekend", "weather_mist", "weather_rain",
             "atemp", "hum", "windspeed", "month_sin", "month_cos", "hour_sin", "hour_cos"]

combined = pd.concat([train[["users"] + exog_cols], test[["users"] + exog_cols]], ignore_index=True)
users_arr = combined["users"].values
exog_arr = combined[exog_cols].values
n_train = len(train)

with open(f"{MODELS_DIR}/xgboost_best_params.json") as f:
    best_params = json.load(f)

# ---------- 1. build one training set per horizon h, train one model per h ----------
# Features are defined relative to the TARGET (origin+h), mirroring exactly what
# the recursive model's lag_1/24/168 mean - "immediately before", "same hour
# yesterday", "same hour last week" - but computed directly from known history
# (valid as long as h <= 24, so target-24 and target-168 never look past the
# origin). No recursion, no leakage, and features stay seasonally aligned
# regardless of h (unlike origin-relative lags, which drift out of alignment
# as h grows).
#   lag_h   = users[i]            (origin value = h steps before target)
#   lag24   = users[i + h - 24]   (same hour yesterday, relative to TARGET)
#   lag168  = users[i + h - 168]  (same hour last week, relative to TARGET)
#   exog_h  = exog[i + h]         (assumed known in advance)
#   target  = users[i + h]
assert HORIZON <= 24, "target-24/target-168 features require h <= 24 to stay non-recursive"
models = {}
t0 = time.time()
for h in range(1, HORIZON + 1):
    origins = np.arange(167, n_train - h)  # stay fully inside TRAIN, never touch test
    X = np.column_stack([
        users_arr[origins], users_arr[origins + h - 24], users_arr[origins + h - 168],
        exog_arr[origins + h],
    ])
    y = users_arr[origins + h]
    m = xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist", **best_params)
    m.fit(X, y)
    models[h] = m
print(f"Trained {HORIZON} direct models in {time.time()-t0:.1f}s")

# ---------- 2. evaluate on test, same 24h-block origins as the rolling-recursive run ----------
preds = np.empty(len(test), dtype="float64")
step_within_block = np.empty(len(test), dtype="int64")

for origin_local in range(0, len(test), HORIZON):
    block_len = min(HORIZON, len(test) - origin_local)
    i = n_train + origin_local - 1  # combined-array index of the LAST KNOWN true point
    for h in range(1, block_len + 1):
        feat = np.array([[users_arr[i], users_arr[i + h - 24], users_arr[i + h - 168], *exog_arr[i + h]]])
        yhat = max(0.0, float(models[h].predict(feat)[0]))
        preds[origin_local + h - 1] = yhat
        step_within_block[origin_local + h - 1] = h

y_true = test["users"].values


def evaluate(name, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{name:>42}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:6.2f}%  R2={r2:.4f}")


rolling = np.load(f"{RESULTS_DIR}/rolling_recursive_predictions.npz")
print("\n=== Comparison, all evaluated on the SAME 24h-block origins ===")
evaluate("One-step-ahead (true lags, ceiling)", rolling["onestep"])
evaluate("Rolling RECURSIVE (re-anchor 24h)", rolling["rolling_recursive"])
evaluate("DIRECT multi-step (re-anchor 24h)", preds)

print(f"\n=== RMSE by step-ahead (1..{HORIZON}h from origin) - recursive vs direct ===")
rows = []
for s in range(1, HORIZON + 1):
    mask = step_within_block == s
    rmse_rec = np.sqrt(mean_squared_error(y_true[mask], rolling["rolling_recursive"][mask]))
    rmse_dir = np.sqrt(mean_squared_error(y_true[mask], preds[mask]))
    rows.append({"step_ahead_h": s, "RMSE_recursive": rmse_rec, "RMSE_direct": rmse_dir})
step_df = pd.DataFrame(rows)
print(step_df.round(2).to_string(index=False))

step_df.to_csv(f"{RESULTS_DIR}/direct_vs_recursive_step.csv", index=False)
np.savez(f"{RESULTS_DIR}/direct_multistep_predictions.npz", ts_test=test["date_time"].values,
         y_test=y_true, direct=preds, step_within_block=step_within_block)
print(f"\nSaved -> {RESULTS_DIR}/direct_vs_recursive_step.csv, "
      f"{RESULTS_DIR}/direct_multistep_predictions.npz")

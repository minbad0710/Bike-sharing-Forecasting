"""Tune XGBoost hyperparameters with rolling-origin (expanding-window) time-
series cross-validation on the train set only, then evaluate the tuned model
on the held-out test set. Run from the repo root:
`python modeling/tune_xgboost.py`
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "modeling/results"

train = pd.read_csv(f"{DATA_DIR}/bike-sharing-train.csv", parse_dates=["date_time"])
test = pd.read_csv(f"{DATA_DIR}/bike-sharing-test.csv", parse_dates=["date_time"])

exog_cols = ["holiday", "workingday", "is_weekend", "weather_mist", "weather_rain",
             "atemp", "hum", "windspeed", "month_sin", "month_cos", "hour_sin", "hour_cos"]
lag_cols = ["users_lag_1", "users_lag_24", "users_lag_168"]
feature_cols = exog_cols + lag_cols

X_train, y_train = train[feature_cols], train["users"]
X_test, y_test = test[feature_cols], test["users"]

# ---------- rolling-origin CV inside TRAIN only (test set stays untouched) ----------
# 5 expanding-window folds: fold i trains on everything before it and
# validates on the next chunk - never lets a fold "see the future",
# unlike a random k-fold shuffle which would leak lag_1/24/168 info
# across the split.
tscv = TimeSeriesSplit(n_splits=5)
for i, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
    print(f"fold {i}: train rows 0..{tr_idx[-1]}  ({len(tr_idx)}) | "
          f"val rows {va_idx[0]}..{va_idx[-1]} ({len(va_idx)})")

param_dist = {
    "n_estimators": randint(200, 800),
    "max_depth": randint(3, 10),
    "learning_rate": loguniform(0.01, 0.3),
    "subsample": uniform(0.6, 0.4),          # [0.6, 1.0]
    "colsample_bytree": uniform(0.6, 0.4),   # [0.6, 1.0]
    "min_child_weight": randint(1, 10),
    "reg_alpha": loguniform(1e-3, 10),
    "reg_lambda": loguniform(1e-3, 10),
}

base_model = xgb.XGBRegressor(random_state=42, n_jobs=1, tree_method="hist")

search = RandomizedSearchCV(
    base_model, param_distributions=param_dist, n_iter=100,
    scoring="neg_root_mean_squared_error", cv=tscv,
    random_state=42, n_jobs=-1, verbose=1, refit=True,
)

t0 = time.time()
search.fit(X_train, y_train)
search_time = time.time() - t0
print(f"\nRandomizedSearchCV done in {search_time:.1f}s")
print("Best CV RMSE:", -search.best_score_)
print("Best params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

with open(f"{MODELS_DIR}/xgboost_best_params.json", "w") as f:
    json.dump(search.best_params_, f, indent=2)

# ---------- refit best params on full train, evaluate on held-out test ----------
tuned = xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist", **search.best_params_)
t0 = time.time()
tuned.fit(X_train, y_train)
fit_time = time.time() - t0
pred_tuned = np.clip(tuned.predict(X_test), 0, None)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{name:>28}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:6.2f}%  R2={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "R2": r2}

print("\n=== TEST SET (held-out) ===")
default_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  n_jobs=-1).fit(X_train, y_train)
pred_default = np.clip(default_model.predict(X_test), 0, None)
res_default = evaluate("XGBoost (default, previous)", y_test, pred_default)
res_tuned = evaluate("XGBoost (tuned)", y_test, pred_tuned)

comparison = pd.DataFrame({"default": res_default, "tuned": res_tuned}).T
comparison["fit_time_s"] = [None, fit_time]
comparison.to_csv(f"{RESULTS_DIR}/xgboost_tuning_comparison.csv")

np.savez(f"{RESULTS_DIR}/xgboost_tuned_predictions.npz", y_test=y_test.values,
         ts_test=test["date_time"].values, pred_tuned=pred_tuned, pred_default=pred_default)

print(f"\nSaved -> {MODELS_DIR}/xgboost_best_params.json, "
      f"{RESULTS_DIR}/xgboost_tuning_comparison.csv, {RESULTS_DIR}/xgboost_tuned_predictions.npz")

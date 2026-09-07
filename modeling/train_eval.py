"""Compare 6 forecasting models (2 naive baselines, Linear Regression, Random
Forest, XGBoost, SARIMAX) on the same one-step-ahead task. Run from the repo
root: `python modeling/train_eval.py`
"""
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_DIR = "data/processed"
RESULTS_DIR = "modeling/results"

train = pd.read_csv(f"{DATA_DIR}/bike-sharing-train.csv", parse_dates=["date_time"])
test = pd.read_csv(f"{DATA_DIR}/bike-sharing-test.csv", parse_dates=["date_time"])

exog_cols = ["holiday", "workingday", "is_weekend", "weather_mist", "weather_rain",
             "atemp", "hum", "windspeed", "month_sin", "month_cos", "hour_sin", "hour_cos"]
lag_cols = ["users_lag_1", "users_lag_24", "users_lag_168"]
feature_cols = exog_cols + lag_cols

X_train, y_train = train[feature_cols], train["users"]
X_test, y_test = test[feature_cols], test["users"]

results = {}
timings = {}
preds = {}


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    results[name] = {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "R2": r2}
    preds[name] = np.asarray(y_pred)
    print(f"{name:>28}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:6.2f}%  R2={r2:.4f}")


# ---------- 1. Naive baselines (no fitting) ----------
for name, lag_col in [("Naive (lag-24, seasonal day)", "users_lag_24"),
                       ("Naive (lag-168, seasonal week)", "users_lag_168")]:
    t0 = time.time()
    evaluate(name, y_test, test[lag_col])
    timings[name] = time.time() - t0

# ---------- 2. Linear Regression ----------
t0 = time.time()
lr = LinearRegression().fit(X_train, y_train)
timings["Linear Regression"] = time.time() - t0
evaluate("Linear Regression", y_test, np.clip(lr.predict(X_test), 0, None))

# ---------- 3. Random Forest ----------
t0 = time.time()
rf = RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=3,
                            n_jobs=-1, random_state=42).fit(X_train, y_train)
timings["Random Forest"] = time.time() - t0
evaluate("Random Forest", y_test, np.clip(rf.predict(X_test), 0, None))

# ---------- 4. XGBoost ----------
t0 = time.time()
xgbr = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         n_jobs=-1)
xgbr.fit(X_train, y_train)
timings["XGBoost"] = time.time() - t0
evaluate("XGBoost", y_test, np.clip(xgbr.predict(X_test), 0, None))

# ---------- 5. SARIMAX (exogenous, no autoregressive lag columns - the model
#    itself supplies the AR/seasonal structure) ----------
# NOTE on evaluation: a naive fit.forecast(steps=len(test)) forecasts the
# entire 3509h test horizon blind (dynamic multi-step), so the AR/seasonal
# terms decay to the exogenous mean within a few dozen hours and the score
# collapses (R2 ~ 0.25 - worse than the naive baseline). That is not a fair
# comparison against the ML models above, which get to see the *true*
# lag_1/24/168 values at every test row. To compare like-for-like, we do a
# rolling one-step-ahead forecast instead: re-apply the fitted params to
# train+test with statsmodels' `apply`, then predict with dynamic=False so
# each step conditions on the true preceding observation, exactly like the
# lag features do for the other models.
t0 = time.time()
sarimax_exog_cols = exog_cols  # weather + calendar only, not the lag_* columns
model = SARIMAX(y_train.values, exog=X_train[sarimax_exog_cols].values,
                 order=(2, 0, 1), seasonal_order=(1, 0, 0, 24),
                 enforce_stationarity=False, enforce_invertibility=False)
fit = model.fit(disp=False, maxiter=60, method="lbfgs")
combined_y = np.concatenate([y_train.values, y_test.values])
combined_exog = np.concatenate([X_train[sarimax_exog_cols].values, X_test[sarimax_exog_cols].values])
extended = fit.apply(combined_y, exog=combined_exog)
sarimax_pred = extended.get_prediction(start=len(y_train), end=len(combined_y) - 1, dynamic=False).predicted_mean
timings["SARIMAX(2,0,1)(1,0,0,24) 1-step"] = time.time() - t0
evaluate("SARIMAX(2,0,1)(1,0,0,24) 1-step", y_test, np.clip(sarimax_pred, 0, None))

# ---------- summary table ----------
summary = pd.DataFrame(results).T
summary["fit+predict time (s)"] = pd.Series(timings)
summary = summary.sort_values("RMSE")
print("\n=== SUMMARY (sorted by RMSE) ===")
print(summary.round(3))

summary.to_csv(f"{RESULTS_DIR}/model_comparison.csv")
np.savez(f"{RESULTS_DIR}/model_predictions.npz", y_test=y_test.values,
         ts_test=test["date_time"].values, **preds)
print(f"\nSaved -> {RESULTS_DIR}/model_comparison.csv, {RESULTS_DIR}/model_predictions.npz")

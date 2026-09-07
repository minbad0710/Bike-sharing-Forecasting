"""Diagnostic plots for the model comparison in train_eval.py: actual-vs-
predicted, residual distributions, and RF/XGBoost feature importance. Run
from the repo root: `python modeling/plot_model_comparison.py`
(requires train_eval.py to have run first).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

ACCENT = "#0f7d8c"
ACCENT2 = "#e0793a"
plt.rcParams.update({
    "axes.edgecolor": "#c7d0d7", "axes.labelcolor": "#1c2530", "text.color": "#1c2530",
    "xtick.color": "#5b6b7a", "ytick.color": "#5b6b7a", "grid.color": "#e5eaee",
})

DATA_DIR = "data/processed"
RESULTS_DIR = "modeling/results"
PLOTS_DIR = "modeling/plots"

data = np.load(f"{RESULTS_DIR}/model_predictions.npz", allow_pickle=True)
ts = pd.to_datetime(data["ts_test"])
y_true = data["y_test"]

model_names = ["Naive (lag-24, seasonal day)", "Naive (lag-168, seasonal week)",
               "Linear Regression", "Random Forest", "XGBoost",
               "SARIMAX(2,0,1)(1,0,0,24) 1-step"]
colors = {"Naive (lag-24, seasonal day)": "#9db3bd", "Naive (lag-168, seasonal week)": "#c7d0d7",
          "Linear Regression": "#6ba3ab", "Random Forest": "#e0793a",
          "XGBoost": "#0f7d8c", "SARIMAX(2,0,1)(1,0,0,24) 1-step": "#c96a2e"}

# ---------- 1. zoomed-in actual vs predicted (best 2 models + naive) for 1 week ----------
mask = (ts >= "2012-10-01") & (ts < "2012-10-15")
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(ts[mask], y_true[mask], color="#1c2530", linewidth=1.6, label="Actual")
for name in ["XGBoost", "Random Forest", "Naive (lag-168, seasonal week)"]:
    ax.plot(ts[mask], data[name][mask], linewidth=1.2, alpha=0.85, color=colors[name], label=name)
ax.set_title("Actual vs predicted - first 2 weeks of October 2012 (test set)")
ax.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/model_actual_vs_pred_zoom.png", dpi=130)
plt.close()

# ---------- 2. scatter actual vs predicted for top models ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, name in zip(axes, ["Linear Regression", "Random Forest", "XGBoost"]):
    ax.scatter(y_true, data[name], s=4, alpha=0.25, color=colors[name])
    lims = [0, max(y_true.max(), data[name].max())]
    ax.plot(lims, lims, color=ACCENT2, linewidth=1, linestyle="--")
    ax.set_title(name)
    ax.set_xlabel("Actual users")
axes[0].set_ylabel("Predicted users")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/model_scatter_actual_pred.png", dpi=130)
plt.close()

# ---------- 3. residual distribution per model ----------
fig, ax = plt.subplots(figsize=(10, 5))
for name in model_names:
    resid = data[name] - y_true
    sns.kdeplot(resid, ax=ax, label=name, color=colors[name], linewidth=1.8)
ax.axvline(0, color="#1c2530", linewidth=0.8)
ax.set_title("Residual distribution (predicted - actual) by model")
ax.set_xlim(-300, 300)
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/model_residual_dist.png", dpi=130)
plt.close()

# ---------- 4. feature importance (RF & XGBoost) ----------
train = pd.read_csv(f"{DATA_DIR}/bike-sharing-train.csv", parse_dates=["date_time"])
exog_cols = ["holiday", "workingday", "is_weekend", "weather_mist", "weather_rain",
             "atemp", "hum", "windspeed", "month_sin", "month_cos", "hour_sin", "hour_cos"]
lag_cols = ["users_lag_1", "users_lag_24", "users_lag_168"]
feature_cols = exog_cols + lag_cols
X_train, y_train = train[feature_cols], train["users"]

rf = RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=42).fit(X_train, y_train)
xgbr = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1).fit(X_train, y_train)

imp = pd.DataFrame({"feature": feature_cols, "Random Forest": rf.feature_importances_, "XGBoost": xgbr.feature_importances_})
imp = imp.sort_values("XGBoost", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
y_pos = np.arange(len(imp))
ax.barh(y_pos - 0.18, imp["Random Forest"], height=0.36, color="#e0793a", label="Random Forest")
ax.barh(y_pos + 0.18, imp["XGBoost"], height=0.36, color="#0f7d8c", label="XGBoost")
ax.set_yticks(y_pos)
ax.set_yticklabels(imp["feature"])
ax.set_title("Feature importance: Random Forest vs XGBoost")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/model_feature_importance.png", dpi=130)
plt.close()

print(imp.sort_values("XGBoost", ascending=False))
print(f"\nSaved plots to {PLOTS_DIR}/")

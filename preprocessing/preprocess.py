"""Build model-ready features from the raw dataset: fix the calendar flags,
one-hot the weather, drop redundant temp, scale the weather measurements,
encode month/hour as cyclical, then export both a tabular (lag-feature) and
a windowed (sequence) train/test split. Run from the repo root:
`python preprocessing/preprocess.py`
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

WINDOW = 168          # 1 week of hourly history per sample (chosen from the ACF analysis)
TEST_FRACTION = 0.20  # chronological, last 20% of hours held out as test

RAW_PATH = "data/raw/bike-sharing-dataset.csv"
OUT_DIR = "data/processed"

df = pd.read_csv(RAW_PATH, parse_dates=["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

# ---------- 1. weekday -> is_weekend, keep holiday/workingday mutually exclusive ----------
# holiday takes priority so the 3 calendar flags never overlap (see EDA: 2 raw rows
# had holiday==1 and workingday==1 at the same time)
is_weekend_raw = df["weekday"].isin([5, 6])  # Sat, Sun
df["is_weekend"] = (is_weekend_raw & (df["holiday"] == 0)).astype(int)
df["workingday"] = ((~is_weekend_raw) & (df["holiday"] == 0)).astype(int)
df["holiday"] = df["holiday"].astype(int)
df = df.drop(columns=["weekday"])

# ---------- 2. weather -> one-hot ----------
# drop_first=True drops "clear" (the majority baseline, 65% of rows) to avoid the
# dummy-variable trap for linear/SARIMAX-style models; tree-based models would be
# fine keeping all 3, but dropping one is harmless there too.
weather_dummies = pd.get_dummies(df["weather"], prefix="weather", drop_first=True)
df = pd.concat([df.drop(columns=["weather"]), weather_dummies], axis=1)
weather_cols = list(weather_dummies.columns)
df[weather_cols] = df[weather_cols].astype(int)

# ---------- 3. drop temp (redundant with atemp, corr(temp, atemp) ~ 0.99) ----------
df = df.drop(columns=["temp"])

# ---------- 4. month, hour -> cyclical (sin/cos) ----------
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df = df.drop(columns=["month", "hour"])

# ---------- 5. chronological train/test split (no shuffle) ----------
split_idx = int(len(df) * (1 - TEST_FRACTION))
split_date = df["date_time"].iloc[split_idx]
print(f"Split point: row {split_idx}/{len(df)}  ({split_date})")

# ---------- 6. scale atemp/hum/windspeed - fit on TRAIN rows only, apply to all ----------
scale_cols = ["atemp", "hum", "windspeed"]
scaler = StandardScaler()
scaler.fit(df.loc[:split_idx - 1, scale_cols])
df[scale_cols] = scaler.transform(df[scale_cols])

print("\n=== StandardScaler fit stats (train only) ===")
for col, mean, scale in zip(scale_cols, scaler.mean_, scaler.scale_):
    print(f"{col:>10}: mean={mean:.3f}  std={scale:.3f}")

# =====================================================================
# A) TABULAR export - one row per hour, with scalar lag features
#    (SARIMAX / gradient boosting / any non-sequence model)
# =====================================================================
LAGS = [1, 24, 168]
tab = df.copy()
for lag in LAGS:
    tab[f"users_lag_{lag}"] = tab["users"].shift(lag)

n_before = len(tab)
tab = tab.dropna(subset=[f"users_lag_{lag}" for lag in LAGS]).reset_index(drop=True)
print(f"\n[tabular] dropped {n_before - len(tab)} leading rows without full lag(168) history")

lag_cols = [f"users_lag_{lag}" for lag in LAGS]
ordered_cols = (
    ["date_time", "users"]
    + ["holiday", "workingday", "is_weekend"]
    + weather_cols
    + scale_cols
    + ["month_sin", "month_cos", "hour_sin", "hour_cos"]
    + lag_cols
)
tab = tab[ordered_cols]

tab_train = tab[tab["date_time"] < split_date]
tab_test = tab[tab["date_time"] >= split_date]
tab_train.to_csv(f"{OUT_DIR}/bike-sharing-train.csv", index=False)
tab_test.to_csv(f"{OUT_DIR}/bike-sharing-test.csv", index=False)
print(f"[tabular] train {tab_train.shape}  ({tab_train['date_time'].min()} -> {tab_train['date_time'].max()})")
print(f"[tabular] test  {tab_test.shape}  ({tab_test['date_time'].min()} -> {tab_test['date_time'].max()})")

# =====================================================================
# B) WINDOWED export - sliding sequences of WINDOW=168h
#    (LSTM / GRU / 1D-CNN / Transformer style models)
#    Each sample: X = features for hours [t-168, t), y = users at hour t.
#    Test windows are allowed to look back into the tail of train (still
#    historical, already-observed data - not leakage) so no test rows
#    are wasted on warm-up, unlike the tabular export above.
# =====================================================================
feature_cols = (
    ["users", "holiday", "workingday", "is_weekend"]
    + weather_cols
    + scale_cols
    + ["month_sin", "month_cos", "hour_sin", "hour_cos"]
)
target_idx = feature_cols.index("users")

values = df[feature_cols].to_numpy(dtype="float32")
timestamps = df["date_time"].to_numpy()

X_list, y_list, ts_list = [], [], []
for i in range(WINDOW, len(df)):
    X_list.append(values[i - WINDOW:i])
    y_list.append(values[i, target_idx])
    ts_list.append(timestamps[i])

X = np.stack(X_list)                 # (n_samples, 168, n_features)
y = np.array(y_list, dtype="float32")
ts = np.array(ts_list)

is_test = ts >= np.datetime64(split_date)
X_train_w, y_train_w, ts_train_w = X[~is_test], y[~is_test], ts[~is_test]
X_test_w, y_test_w, ts_test_w = X[is_test], y[is_test], ts[is_test]

print(f"\n[windowed] X_train {X_train_w.shape}  X_test {X_test_w.shape}  (window={WINDOW}h, {len(feature_cols)} features/step)")
print(f"[windowed] feature order: {feature_cols}")

np.savez_compressed(
    f"{OUT_DIR}/bike-sharing-windows.npz",
    X_train=X_train_w, y_train=y_train_w, ts_train=ts_train_w,
    X_test=X_test_w, y_test=y_test_w, ts_test=ts_test_w,
    feature_names=np.array(feature_cols),
)
print(f"Saved -> {OUT_DIR}/bike-sharing-train.csv, {OUT_DIR}/bike-sharing-test.csv, "
      f"{OUT_DIR}/bike-sharing-windows.npz")

"""Pick a lag/window size from the ACF, and check whether the sharp rise-then-
drop swings in `users` are noise or genuine weather-driven signal (via an STL
residual decomposition). Run from the repo root:
`python eda/window_outlier_analysis.py`
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

ACCENT = "#0f7d8c"
ACCENT2 = "#e0793a"
plt.rcParams.update({
    "axes.edgecolor": "#c7d0d7", "axes.labelcolor": "#1c2530", "text.color": "#1c2530",
    "xtick.color": "#5b6b7a", "ytick.color": "#5b6b7a", "grid.color": "#e5eaee",
})

PLOTS_DIR = "eda/plots"

df = pd.read_csv("data/raw/bike-sharing-dataset.csv", parse_dates=["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)
s = df.set_index("date_time")["users"].asfreq("h")
print("Missing timestamps after asfreq('h'):", s.isna().sum())

# ---------- 1. ACF / PACF on hourly series to pick window size ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
plot_acf(s, lags=200, ax=axes[0], color=ACCENT)
axes[0].set_title("ACF - hourly users (up to 200 lags)")
plot_pacf(s, lags=72, ax=axes[1], color=ACCENT2, method="ywm")
axes[1].set_title("PACF - hourly users (up to 72 lags)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/acf_pacf_users.png", dpi=130)
plt.close()

acf_vals = pd.Series(s.autocorr(lag=l) for l in [1, 2, 3, 6, 12, 23, 24, 25, 47, 48, 72, 96, 120, 144, 167, 168, 169])
acf_vals.index = [1, 2, 3, 6, 12, 23, 24, 25, 47, 48, 72, 96, 120, 144, 167, 168, 169]
print("\n=== Autocorrelation at key lags (hours) ===")
print(acf_vals.round(3))

# ---------- 2. STL decomposition (period=24h) to separate trend/seasonal/residual ----------
stl = STL(s, period=24, robust=True).fit()
resid = stl.resid

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
axes[0].plot(s.index, s.values, color=ACCENT, linewidth=0.6); axes[0].set_title("Observed")
axes[1].plot(s.index, stl.trend, color=ACCENT, linewidth=0.8); axes[1].set_title("Trend")
axes[2].plot(s.index, stl.seasonal, color=ACCENT, linewidth=0.4); axes[2].set_title("Seasonal (24h)")
axes[3].plot(s.index, resid, color=ACCENT2, linewidth=0.4); axes[3].set_title("Residual")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/stl_decomposition.png", dpi=130)
plt.close()

# ---------- 3. Outlier detection on RESIDUAL (not raw values) ----------
mu, sigma = resid.mean(), resid.std()
z = (resid - mu) / sigma
outlier_mask = z.abs() > 3
print(f"\nResidual outliers (|z|>3): {outlier_mask.sum()} / {len(s)} rows ({outlier_mask.mean()*100:.2f}%)")

out_df = df.set_index("date_time").loc[outlier_mask.index[outlier_mask]]
print("\n=== Weather composition of residual-outlier hours ===")
print(out_df["weather"].value_counts())
print("\n=== Sample of residual-outlier rows ===")
print(out_df[["users", "weather", "windspeed", "holiday"]].head(15))

# how many raw-value outliers (naive IQR on users itself) would we have flagged instead?
q1, q3 = df["users"].quantile([0.25, 0.75])
iqr = q3 - q1
naive_mask = (df["users"] < q1 - 1.5 * iqr) | (df["users"] > q3 + 1.5 * iqr)
print(f"\nNaive IQR outliers on raw users: {naive_mask.sum()} / {len(df)} rows ({naive_mask.mean()*100:.2f}%)")

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(s.index, s.values, color="#c7d0d7", linewidth=0.6, label="users")
ax.scatter(out_df.index, out_df["users"], color=ACCENT2, s=14, zorder=5, label="residual outlier (|z|>3)")
ax.set_title("Hourly users with STL-residual outliers highlighted")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/outliers_highlighted.png", dpi=130)
plt.close()

print("\nDone.")

"""Exploratory data analysis: which exogenous variables explain hourly bike
rental demand (`users`)? Run from the repo root: `python eda/eda.py`
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
ACCENT = "#0f7d8c"
ACCENT2 = "#e0793a"
INK = "#1c2530"
plt.rcParams.update({
    "axes.edgecolor": "#c7d0d7",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#5b6b7a",
    "ytick.color": "#5b6b7a",
    "grid.color": "#e5eaee",
    "font.family": "DejaVu Sans",
})

DATA_PATH = "data/raw/bike-sharing-dataset.csv"
PLOTS_DIR = "eda/plots"
DAILY_OUT = "data/processed/daily_aggregated.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

print("=== SHAPE / DTYPES ===")
print(df.shape)
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isna().sum())

print("\n=== DESCRIBE (numeric) ===")
print(df.describe())

print("\n=== WEATHER CATEGORIES ===")
print(df["weather"].value_counts())

# ---------- derived time fields ----------
df["date"] = df["date_time"].dt.date
DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
df["weekday_name"] = df["weekday"].map(DAY_NAMES)

# holiday takes priority over weekend/workingday so the 3 flags are mutually
# exclusive and collectively exhaustive (the raw data has 2 rows where
# holiday==1 and workingday==1 both fired - see the day_type analysis below)
is_weekend_raw = df["weekday"].isin([5, 6])
df["is_weekend"] = (is_weekend_raw & (df["holiday"] == 0)).astype(int)
df["workingday"] = ((~is_weekend_raw) & (df["holiday"] == 0)).astype(int)

assert (df["holiday"] + df["is_weekend"] + df["workingday"] == 1).all(), \
    "holiday / is_weekend / workingday must partition every row"

# ============================================================
# 1. Correlation of numeric features with users
# ============================================================
num_cols = ["holiday", "workingday", "temp", "atemp", "hum", "windspeed", "month", "hour", "weekday", "users"]
print("\n=== PEARSON CORRELATION WITH users ===")
print(df[num_cols].corr(numeric_only=True)["users"].sort_values(ascending=False))

print("\n=== SPEARMAN CORRELATION WITH users ===")
print(df[num_cols].corr(method="spearman", numeric_only=True)["users"].sort_values(ascending=False))

plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(numeric_only=True), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, linecolor="white")
plt.title("Correlation heatmap (numeric features vs users)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/corr_heatmap.png", dpi=130)
plt.close()

# ============================================================
# 2. Time-of-day / weekday / month patterns
# ============================================================
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="hour", y="users", hue="workingday", estimator="mean", errorbar=None,
             palette=[ACCENT2, ACCENT], linewidth=2.5)
plt.title("Avg users by hour of day (working vs non-working day)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/users_by_hour_workingday.png", dpi=130)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="weekday_name", y="users", order=[DAY_NAMES[i] for i in range(7)],
            estimator="mean", errorbar=None, color=ACCENT)
plt.title("Avg users by weekday")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/users_by_weekday.png", dpi=130)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="month", y="users", estimator="mean", errorbar=None, color=ACCENT2)
plt.title("Avg users by month (seasonality)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/users_by_month.png", dpi=130)
plt.close()

# ============================================================
# 3. Weather: condition (categorical) and numeric measurements
# ============================================================
plt.figure(figsize=(8, 5))
weather_order = df.groupby("weather")["users"].mean().sort_values(ascending=False).index
sns.boxplot(data=df, x="weather", y="users", order=weather_order,
            palette=[ACCENT, "#6ba3ab", ACCENT2], hue="weather", legend=False)
plt.title("Users distribution by weather condition")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/users_by_weather.png", dpi=130)
plt.close()

print("\n=== MEAN users BY WEATHER ===")
print(df.groupby("weather")["users"].mean().sort_values(ascending=False))

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, col, title in zip(
    axes.flat, ["temp", "atemp", "hum", "windspeed"],
    ["temp vs users", "atemp (feels-like) vs users", "humidity vs users", "windspeed vs users"],
):
    sample = df.sample(min(3000, len(df)), random_state=1)
    sns.regplot(data=sample, x=col, y="users", ax=ax,
                scatter_kws={"alpha": 0.25, "s": 10, "color": "#6ba3ab"}, line_kws={"color": ACCENT2})
    ax.set_title(title)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/users_vs_weather_numeric.png", dpi=130)
plt.close()

# ============================================================
# 4. Calendar day type: holiday / weekend / workingday (mutually exclusive)
# ============================================================
df["day_type"] = np.select(
    [df["holiday"] == 1, df["is_weekend"] == 1, df["workingday"] == 1],
    ["holiday", "weekend", "workingday"],
    default="unknown",
)

print("\n=== MEAN users BY day_type ===")
print(df.groupby("day_type")["users"].mean().sort_values(ascending=False))
print("\n=== day_type counts (rows) ===")
print(df["day_type"].value_counts())

day_type_order = ["workingday", "weekend", "holiday"]
day_type_palette = {"workingday": ACCENT, "weekend": "#6ba3ab", "holiday": ACCENT2}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df, x="day_type", y="users", order=day_type_order, palette=day_type_palette,
            hue="day_type", legend=False, ax=axes[0])
axes[0].set_title("Users distribution by day type")
sns.lineplot(data=df, x="hour", y="users", hue="day_type", hue_order=day_type_order,
             palette=day_type_palette, estimator="mean", errorbar=None, linewidth=2.2, ax=axes[1])
axes[1].set_title("Avg users by hour, per day type")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/users_by_daytype.png", dpi=130)
plt.close()

# ============================================================
# 5. Daily aggregated trend + significance tests
# ============================================================
daily = df.groupby("date").agg(
    users=("users", "sum"), temp=("temp", "mean"), hum=("hum", "mean"),
    windspeed=("windspeed", "mean"), workingday=("workingday", "max"), holiday=("holiday", "max"),
).reset_index()
daily["date"] = pd.to_datetime(daily["date"])

plt.figure(figsize=(14, 5))
plt.plot(daily["date"], daily["users"], color=ACCENT, linewidth=1.3)
plt.title("Daily total users over time (trend + seasonality)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/daily_users_timeseries.png", dpi=130)
plt.close()

daily.to_csv(DAILY_OUT, index=False)

weather_groups = [g["users"].values for _, g in df.groupby("weather")]
f_stat, p_val = stats.f_oneway(*weather_groups)
print(f"\n=== ANOVA weather -> users: F={f_stat:.2f}, p={p_val:.2e} ===")

day_type_groups = [g["users"].values for _, g in df.groupby("day_type")]
f_stat, p_val = stats.f_oneway(*day_type_groups)
print(f"=== ANOVA day_type -> users: F={f_stat:.2f}, p={p_val:.2e} ===")

print(f"\nDone. Plots saved to {PLOTS_DIR}/, daily aggregates saved to {DAILY_OUT}")

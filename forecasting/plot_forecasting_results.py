"""Diagnostic plots for the recursive / rolling-recursive / direct multi-step
forecasting experiments. Run from the repo root:
`python forecasting/plot_forecasting_results.py`
(needs recursive_forecast.py, rolling_recursive_forecast.py,
sweep_reanchor_horizon.py and direct_multistep_forecast.py to have run first).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT = "#0f7d8c"
ACCENT2 = "#e0793a"
plt.rcParams.update({
    "axes.edgecolor": "#c7d0d7", "axes.labelcolor": "#1c2530", "text.color": "#1c2530",
    "xtick.color": "#5b6b7a", "ytick.color": "#5b6b7a", "grid.color": "#e5eaee",
})

RESULTS_DIR = "forecasting/results"
PLOTS_DIR = "forecasting/plots"

# ============================================================
# 1-2. Pure recursive (single forecast origin) vs one-step-ahead
# ============================================================
recur = np.load(f"{RESULTS_DIR}/recursive_forecast_predictions.npz", allow_pickle=True)
ts = pd.to_datetime(recur["ts_test"])
y_true, onestep, recursive = recur["y_test"], recur["onestep"], recur["recursive"]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(ts, y_true, color="#1c2530", linewidth=0.5, alpha=0.6, label="Actual")
ax.plot(ts, recursive, color=ACCENT2, linewidth=0.6, label="Recursive (1 forecast origin)")
ax.plot(ts, onestep, color=ACCENT, linewidth=0.4, alpha=0.7, label="One-step-ahead (true lags)")
ax.set_title("Recursive vs one-step-ahead - full 3,509h test horizon")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/recursive_full_horizon.png", dpi=130)
plt.close()

mask = np.arange(len(ts)) < 24 * 14
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(ts[mask], y_true[mask], color="#1c2530", linewidth=1.4, label="Actual")
ax.plot(ts[mask], recursive[mask], color=ACCENT2, linewidth=1.4, label="Recursive")
ax.plot(ts[mask], onestep[mask], color=ACCENT, linewidth=1.2, alpha=0.8, label="One-step-ahead")
ax.set_title("First 2 weeks of the horizon - recursive starts drifting")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/recursive_zoom_early.png", dpi=130)
plt.close()

hz = pd.read_csv(f"{RESULTS_DIR}/recursive_horizon_degradation.csv")
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(hz))
ax.plot(x, hz["RMSE_onestep"], marker="o", color=ACCENT, linewidth=1.8, label="One-step-ahead")
ax.plot(x, hz["RMSE_recursive"], marker="o", color=ACCENT2, linewidth=1.8, label="Recursive")
ax.set_xticks(x)
ax.set_xticklabels(hz["week_range"], rotation=60, ha="right", fontsize=8)
ax.set_ylabel("RMSE")
ax.set_title("RMSE by forecast horizon (1 point = 1 week ahead, from forecast origin)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/recursive_horizon_degradation.png", dpi=130)
plt.close()

# ============================================================
# 3. Rolling recursive: error by hour-within-block (re-anchor every 24h)
# ============================================================
step_df = pd.read_csv(f"{RESULTS_DIR}/rolling_recursive_step_degradation.csv")
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(step_df["step_ahead_h"], step_df["RMSE"], marker="o", color=ACCENT2, linewidth=2)
ax.set_xlabel("Hours since the last re-anchor (block starts at 00:00)")
ax.set_ylabel("RMSE")
ax.set_title("RMSE by hour-within-block (rolling recursive, re-anchor every 24h)")
ax.set_xticks(step_df["step_ahead_h"])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/rolling_recursive_step_rmse.png", dpi=130)
plt.close()

# ============================================================
# 4. Re-anchor horizon sweep: accuracy/frequency trade-off
# ============================================================
sweep = pd.read_csv(f"{RESULTS_DIR}/reanchor_horizon_sweep.csv")
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(sweep["reanchor_every_h"], sweep["RMSE"], marker="o", color=ACCENT, linewidth=2.2)
ax.set_xscale("log")
ax.set_xticks(sweep["reanchor_every_h"])
ax.set_xticklabels(sweep["reanchor_every_h"])
ax.set_xlabel("Re-anchor interval (hours, log scale)")
ax.set_ylabel("RMSE")
ax.set_title("Trade-off: shorter re-anchor interval -> lower error, with diminishing returns")
for x, y in zip(sweep["reanchor_every_h"], sweep["RMSE"]):
    ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=9, color="#1c2530")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/reanchor_horizon_tradeoff.png", dpi=130)
plt.close()

# ============================================================
# 5. Direct multi-step vs rolling recursive, by step-ahead
# ============================================================
direct_df = pd.read_csv(f"{RESULTS_DIR}/direct_vs_recursive_step.csv")
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(direct_df["step_ahead_h"], direct_df["RMSE_recursive"], marker="o", color=ACCENT2,
        linewidth=2, label="Recursive (self-predicted lag_1)")
ax.plot(direct_df["step_ahead_h"], direct_df["RMSE_direct"], marker="o", color=ACCENT,
        linewidth=2, label="Direct (24 separate models, no recursion)")
ax.set_xticks(direct_df["step_ahead_h"])
ax.set_xlabel("Hours since the last re-anchor")
ax.set_ylabel("RMSE")
ax.set_title("Recursive vs Direct multi-step, re-anchor every 24h - roughly tied")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/direct_vs_recursive_step.png", dpi=130)
plt.close()

print(f"Saved 6 plots to {PLOTS_DIR}/")

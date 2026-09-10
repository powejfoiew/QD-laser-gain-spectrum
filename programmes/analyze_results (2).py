import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------
# BASELINE PARAMETER DEFINITIONS
# ---------------------------------------------------------------
BASELINE = {
    # 11 Time Constants
    "tau_aug_GS": 660E-12,
    "tau_aug_ES1": 275E-12,
    "tau_aug_ES2": 110E-12,
    "tau_spon_GS": 2.8E-10, # 0.1 * 2.8e-9
    "tau_spon_ES1": 2.8E-10,
    "tau_spon_ES2": 2.8E-10,
    "tau_c_e_GS": 2E-12,
    "tau_c_e_ES1": 3E-12,
    "tau_c_e_ES2": 3E-12,
    "tau_c_e_W": 1.2,
    "tau_r_W": 100,
    # 6 Rho Shape Multipliers
    "rho_xm_GS": 1.0,
    "rho_xm_ES1": 1.0,
    "rho_xm_ES2": 1.0,
    "rho_n_GS": 1.0,
    "rho_n_ES1": 1.0,
    "rho_n_ES2": 1.0
}

PARAM_LABELS = {
    "tau_aug_GS": "Auger GS (s)",
    "tau_aug_ES1": "Auger ES1 (s)",
    "tau_aug_ES2": "Auger ES2 (s)",
    "tau_spon_GS": "Spon GS (s)",
    "tau_spon_ES1": "Spon ES1 (s)",
    "tau_spon_ES2": "Spon ES2 (s)",
    "tau_c_e_GS": "Capture GS (s)",
    "tau_c_e_ES1": "Capture ES1 (s)",
    "tau_c_e_ES2": "Capture ES2 (s)",
    "tau_c_e_W": "Capture WL (ps)",
    "tau_r_W": "Rec WL (ps)",
    "rho_xm_GS": "Rho Ceiling GS",
    "rho_xm_ES1": "Rho Ceiling ES1",
    "rho_xm_ES2": "Rho Ceiling ES2",
    "rho_n_GS": "Rho Exponent GS",
    "rho_n_ES1": "Rho Exponent ES1",
    "rho_n_ES2": "Rho Exponent ES2"
}

def analyze_and_plot(csv_path="optimisation_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    valid_df = df[df["crashed"] == False].copy()

    if valid_df.empty:
        print("No valid simulation runs to analyze.")
        return

    available_keys = [k for k in BASELINE.keys() if k in valid_df.columns]

    # Compute parameter multipliers relative to baseline
    for k in available_keys:
        valid_df[k + "_mult"] = valid_df[k] / BASELINE[k]

    # Calculate Pearson correlations with final GS occupation
    correlations = {}
    for k in available_keys:
        corr = valid_df[k + "_mult"].corr(valid_df["final_gs"])
        correlations[k] = corr

    corr_series = pd.Series(correlations).sort_values()

    # ---------------------------------------------------------------
    # FIGURE 1: CORRELATION BAR CHART (Positive vs Negative Impact)
    # ---------------------------------------------------------------
    plt.figure(figsize=(13, 8))
    colors = ['#d9534f' if val < 0 else '#5cb85c' for val in corr_series.values]
    bars = plt.barh([PARAM_LABELS.get(k, k) for k in corr_series.index], corr_series.values, color=colors, edgecolor='black', alpha=0.85)

    plt.axvline(0, color='black', linewidth=1.2, linestyle='--')
    plt.title("Impact of Time Constants & Rho Shape Parameters on Final GS Occupation\n(GREEN = Increasing Parameter Helps GS Occ | RED = Decreasing Parameter Helps GS Occ)", fontsize=11, fontweight='bold', pad=15)
    plt.xlabel("Pearson Correlation Coefficient with Final GS Occupation", fontsize=11, fontweight='bold')
    plt.grid(axis='x', linestyle=':', alpha=0.7)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        x_pos = width + (0.02 if width >= 0 else -0.07)
        plt.text(x_pos, bar.get_y() + bar.get_height()/2, f"{width:+.3f}", 
                 va='center', ha='left' if width >= 0 else 'right', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("parameter_correlations_bar.png", dpi=300)
    print("Saved parameter_correlations_bar.png")

    # ---------------------------------------------------------------
    # FIGURE 2: SUBPLOT GRID SHOWING TRENDS FOR EACH PARAMETER
    # ---------------------------------------------------------------
    n_params = len(available_keys)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, k in enumerate(available_keys):
        ax = axes[idx]
        x_vals = valid_df[k + "_mult"]
        y_vals = valid_df["final_gs"]

        # Scatter plot
        ax.scatter(x_vals, y_vals, color='#337ab7', alpha=0.75, edgecolors='k', s=40)

        # Polynomial fit line (trend line)
        if len(x_vals) > 2 and x_vals.nunique() > 1:
            p = np.polyfit(x_vals, y_vals, 1)
            x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_trend = np.polyval(p, x_trend)
            trend_color = '#d9534f' if p[0] < 0 else '#5cb85c'
            ax.plot(x_trend, y_trend, color=trend_color, linewidth=2.5, linestyle='-', label=f"Trend (r={correlations[k]:+.2f})")

        ax.set_title(PARAM_LABELS.get(k, k), fontsize=10, fontweight='bold')
        ax.set_xlabel("Multiplier relative to baseline", fontsize=8)
        ax.set_ylabel("Final GS Occupation", fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for j in range(n_params, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Scatter & Trend Plots: Parameter Multipliers vs Final GS Occupation", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("parameter_trends_grid.png", dpi=300)
    print("Saved parameter_trends_grid.png")

    # ---------------------------------------------------------------
    # PRINT SUMMARY TABLE TO CONSOLE
    # ---------------------------------------------------------------
    print("\n=======================================================")
    print("PARAMETER CORRELATION & TREND SUMMARY")
    print("=======================================================")
    print(f"{'Parameter':<22} | {'Correlation':<12} | {'Effect on GS Occupation':<25}")
    print("-" * 66)
    for k, corr in corr_series.items():
        effect = "INCREASES GS Occ if Parameter UP (+)" if corr > 0 else "INCREASES GS Occ if Parameter DOWN (-)"
        print(f"{PARAM_LABELS.get(k, k):<22} | {corr:+.3f}        | {effect:<25}")

if __name__ == "__main__":
    analyze_and_plot()

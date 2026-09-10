#!/usr/bin/env python3
"""
plot_publication_beeswarm.py
============================
Generates a publication-ready dual SHAP beeswarm plot with:
  • Figure width exactly 6.3 inches
  • Identical tau parameter rows aligned across both subplots
  • Ordered by maximum SHAP importance across both targets
  • Publication serif typography, math LaTeX labels, and clean colorbars
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
import shap

# ── Configuration ────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Correlations\water_87ck.csv"
OUT_PNG  = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Ultra-stable-lasers-and-photonic-integrated-circuits-for-secure-communications\content\Figures\pgf_figures\shap_beeswarm_publication.pdf"

TAU_BASELINE = {
    "tau_aug_GS":   660e-12,
    "tau_aug_ES1":  275e-12,
    "tau_aug_ES2":  110e-12,
    "tau_spon_GS":  0.1 * 2.8e-9,
    "tau_spon_ES1": 0.1 * 2.8e-9,
    "tau_spon_ES2": 0.1 * 2.8e-9,
    "tau_c_e_GS":   2e-12,
    "tau_c_e_ES1":  3e-12,
    "tau_c_e_ES2":  3e-12,
    "tau_c_e_W":    1.2,
    "tau_r_W":      100,
}
MAX_TIME_LOG2 = np.log2(10.0)

TAU_LATEX = {
    "tau_aug_GS":   r"$\tau_{\mathrm{aug}}^{\mathrm{GS}}$",
    "tau_aug_ES1":  r"$\tau_{\mathrm{aug}}^{\mathrm{ES1}}$",
    "tau_aug_ES2":  r"$\tau_{\mathrm{aug}}^{\mathrm{ES2}}$",
    "tau_spon_GS":  r"$\tau_{\mathrm{spon}}^{\mathrm{GS}}$",
    "tau_spon_ES1": r"$\tau_{\mathrm{spon}}^{\mathrm{ES1}}$",
    "tau_spon_ES2": r"$\tau_{\mathrm{spon}}^{\mathrm{ES2}}$",
    "tau_c_e_GS":   r"$\tau_{\mathrm{c,e}}^{\mathrm{GS}}$",
    "tau_c_e_ES1":  r"$\tau_{\mathrm{c,e}}^{\mathrm{ES1}}$",
    "tau_c_e_ES2":  r"$\tau_{\mathrm{c,e}}^{\mathrm{ES2}}$",
    "tau_c_e_W":    r"$\tau_{\mathrm{c,e}}^{\mathrm{W}}$",
    "tau_r_W":      r"$\tau_{\mathrm{r}}^{\mathrm{W}}$",
}

OCC_TARGETS = {"GS": 0.20, "ES1": 0.40, "ES2": 0.50}
OCC_COLS    = {"GS": "occupation_GS_50mA", "ES1": "occupation_ES1_50mA", "ES2": "occupation_ES2_50mA"}

# ── Publication rcParams ──────────────────────────────────────────────────────
maj_tick_width = 0.8
min_tick_width = 0.8

plot_params = {
    "figure.dpi": 200,
    "axes.labelsize": 14,
    "axes.linewidth": 1.5,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.title_fontsize": 11,
    "legend.fontsize": 11,
    "xtick.major.size": 3.5,
    "xtick.major.width": maj_tick_width,
    "xtick.minor.size": 2.5,
    "xtick.minor.width": min_tick_width,
    "ytick.major.size": 3.5,
    "ytick.major.width": maj_tick_width,
    "ytick.minor.size": 2.5,
    "ytick.minor.width": min_tick_width,
    "font.family": "serif",
    "text.usetex": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pgf.texsystem": "pdflatex",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
}

plt.rcParams.update(plot_params)

def to_log2_ratio(series, baseline):
    ratio = series.astype(float) / baseline
    ratio = ratio.clip(lower=1e-12)
    return np.clip(np.log2(ratio), -MAX_TIME_LOG2, MAX_TIME_LOG2)


def main():
    # 1. Load data
    df = pd.read_csv(CSV_PATH)
    df["occupation_GS_50mA"]  = df["occupation_GS_50mA"].fillna(df.get("final_gs",  np.nan))
    df["occupation_ES1_50mA"] = df["occupation_ES1_50mA"].fillna(df.get("final_es1", np.nan))
    df["occupation_ES2_50mA"] = df["occupation_ES2_50mA"].fillna(df.get("final_es2", np.nan))

    valid = df[df["invalid"] == False].copy()
    tau_keys = list(TAU_BASELINE.keys())

    for k in tau_keys:
        valid[f"log2_{k}"] = to_log2_ratio(valid[k], TAU_BASELINE[k])

    X_df = pd.DataFrame({k: valid[f"log2_{k}"].values for k in tau_keys})

    # Targets
    y_score = valid["score"].values
    sq_devs = sum((valid[OCC_COLS[s]] - OCC_TARGETS[s]) ** 2 for s in OCC_TARGETS)
    y_occ   = -np.sqrt(sq_devs / 3.0).values

    # 2. Train RF surrogates
    rf_score = RandomForestRegressor(n_estimators=300, min_samples_split=5, random_state=42, n_jobs=-1)
    rf_score.fit(X_df, y_score)
    expl_score = shap.TreeExplainer(rf_score)
    shap_score = expl_score.shap_values(X_df)

    rf_occ = RandomForestRegressor(n_estimators=300, min_samples_split=5, random_state=42, n_jobs=-1)
    rf_occ.fit(X_df, y_occ)
    expl_occ = shap.TreeExplainer(rf_occ)
    shap_occ = expl_occ.shap_values(X_df)

    # 3. Compute combined importance to rank rows
    imp_score = np.abs(shap_score).mean(axis=0)
    imp_score_pct = imp_score / imp_score.sum()

    imp_occ = np.abs(shap_occ).mean(axis=0)
    imp_occ_pct = imp_occ / imp_occ.sum()

    # Rank by maximum percentage contribution across the two models
    combined_score = np.maximum(imp_score_pct, imp_occ_pct)
    rank_indices = np.argsort(combined_score)[::-1]  # descending: most important first

    # Ordered keys & feature names
    ordered_keys  = [tau_keys[i] for i in rank_indices]
    ordered_latex = [TAU_LATEX[k] for k in ordered_keys]

    # Reorder X and shap matrices to match ordered_keys exactly
    X_ordered = X_df[ordered_keys].copy()
    X_ordered.columns = ordered_latex

    shap_score_ordered = shap_score[:, rank_indices]
    shap_occ_ordered   = shap_occ[:, rank_indices]

    # 4. Create publication figure (6.3 in wide)
    FIG_WIDTH  = 6.3
    FIG_HEIGHT = 3.6

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15, left=0.12, right=0.92, top=0.88, bottom=0.13)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Plot (a) L-I curve score
    plt.sca(ax1)
    shap.summary_plot(
        shap_score_ordered,
        X_ordered,
        max_display=11,
        sort=False,
        show=False,
        plot_size=None,
        color_bar=False,
    )
    ax1.set_title(r"(a)")
    ax1.set_xlabel("SHAP value (L-I)")

    # Plot (b) Occupation score
    plt.sca(ax2)
    shap.summary_plot(
        shap_occ_ordered,
        X_ordered,
        max_display=11,
        sort=False,
        show=False,
        plot_size=None,
        color_bar=False,
    )
    ax2.set_title(r"(b)")
    ax2.set_xlabel("SHAP value (occupation)")
    ax2.set_yticklabels([])  # hide duplicate y labels on right plot for clean look

    # Add shared colorbar on the far right
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    sm = cm.ScalarMappable(cmap=shap.plots.colors.red_blue, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    cbar_ax = fig.add_axes([0.935, 0.22, 0.015, 0.58])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Low", "High"], fontsize=7.5)
    cbar.set_label(r"$\log_2(\tau / \tau_0)$", fontsize=8, labelpad=-2)

    path = OUT_PNG
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Publication beeswarm plot saved to: {path}")


if __name__ == "__main__":
    main()

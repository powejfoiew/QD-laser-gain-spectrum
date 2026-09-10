#!/usr/bin/env python3
"""
shap_dual_analysis.py
=====================
Two separate SHAP analyses on the LHS tau-parameter data:

  Model A:  log2(tau/baseline)  →  score (= -RMSE of L-I curve fit)
  Model B:  log2(tau/baseline)  →  occupation_cost (RMS deviation from
            GS=0.20 / ES1=0.40 / ES2=0.50 targets at 50 mA)

Each model is a Random Forest trained on the 1636 valid LHS runs.
SHAP TreeExplainer is used to compute Shapley values, producing:

  1. Side-by-side beeswarm summary plots  (Fig 1)
  2. Side-by-side bar importance plots     (Fig 2)
  3. Dependence plots for top features     (Fig 3)
  4. SHAP interaction heatmaps             (Fig 4)

All figures are saved at 300 DPI in the Correlations folder.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
import shap

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Correlations\water_87ck.csv"
OUT_DIR  = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Correlations"

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

# LaTeX labels for axes (using mathrm for upright subscripts)
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
OCC_COLS    = {"GS": "occupation_GS_50mA", "ES1": "occupation_ES1_50mA",
               "ES2": "occupation_ES2_50mA"}

RANDOM_STATE = 42
DPI = 300

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
    "pgf.rcfonts": False,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
}

plt.rcParams.update(plot_params)


# =============================================================================
# HELPERS
# =============================================================================

def to_log2_ratio(series, baseline):
    ratio = series.astype(float) / baseline
    ratio = ratio.clip(lower=1e-12)
    return np.clip(np.log2(ratio), -MAX_TIME_LOG2, MAX_TIME_LOG2)


def train_rf_and_shap(X, y, target_name):
    """Train a Random Forest, evaluate it, and compute SHAP values."""
    print(f"\n{'-'*60}")
    print(f"  Training RF surrogate for: {target_name}")
    print(f"  Samples: {len(X)}  |  Features: {X.shape[1]}")
    print(f"{'-'*60}")

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y)

    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    train_r2  = rf.score(X, y)
    print(f"  Train R²:           {train_r2:.4f}")
    print(f"  5-fold CV R²:       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # SHAP
    print(f"  Computing SHAP values ...")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # Importance ranking
    mean_abs = np.abs(shap_values).mean(axis=0)
    total    = mean_abs.sum()
    imp_df   = pd.DataFrame({
        "parameter":   X.columns,
        "mean_abs_shap": mean_abs,
        "pct":         mean_abs / total * 100,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n  SHAP importance ranking ({target_name}):")
    for i, row in imp_df.iterrows():
        bar = "#" * int(row["pct"] / 2)
        print(f"    {i+1:>2d}. {TAU_LATEX.get(row['parameter'], row['parameter']):<30s} "
              f"{row['pct']:>5.1f}%  {bar}")

    return rf, shap_values, imp_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  DUAL SHAP ANALYSIS: Score vs. Occupation Cost")
    print("=" * 60)

    # -- Load & prepare data --------------------------------------------------
    df = pd.read_csv(CSV_PATH)
    df["occupation_GS_50mA"]  = df["occupation_GS_50mA"].fillna(df.get("final_gs",  np.nan))
    df["occupation_ES1_50mA"] = df["occupation_ES1_50mA"].fillna(df.get("final_es1", np.nan))
    df["occupation_ES2_50mA"] = df["occupation_ES2_50mA"].fillna(df.get("final_es2", np.nan))

    valid = df[df["invalid"] == False].copy()
    print(f"\nTotal rows: {len(df)}  |  Valid rows: {len(valid)}")

    tau_keys = list(TAU_BASELINE.keys())

    # Build feature matrix in log2-ratio space
    for k in tau_keys:
        valid[f"log2_{k}"] = to_log2_ratio(valid[k], TAU_BASELINE[k])
    feature_cols = [f"log2_{k}" for k in tau_keys]
    X = valid[feature_cols].copy()
    # Rename columns to use LaTeX labels for SHAP plots
    latex_names = {f"log2_{k}": TAU_LATEX[k] for k in tau_keys}
    X_latex = X.rename(columns=latex_names)

    # Target A: score (= -RMSE)
    y_score = valid["score"].values

    # Target B: occupation score (= -RMS)
    sq_devs = sum((valid[OCC_COLS[s]] - OCC_TARGETS[s]) ** 2 for s in OCC_TARGETS)
    y_occ   = -np.sqrt(sq_devs / 3.0).values

    # -- Train models ---------------------------------------------------------
    rf_score, shap_score, imp_score = train_rf_and_shap(
        X_latex, y_score, "score (= −RMSE)")
    rf_occ, shap_occ, imp_occ = train_rf_and_shap(
        X_latex, y_occ, "occupation score (= −RMS)")

    # ==========================================================================
    # FIGURE 1: Side-by-side SHAP beeswarm summary plots
    # ==========================================================================
    print("\n  Generating Figure 1: SHAP beeswarm summary ...")
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(11, 5.5))

    plt.sca(ax1a)
    shap.summary_plot(shap_score, X_latex, max_display=11, show=False,
                      plot_size=None)
    ax1a.set_title("(a)", fontweight="bold")
    ax1a.set_xlabel("SHAP value (impact on L-I score)")

    plt.sca(ax1b)
    shap.summary_plot(shap_occ, X_latex, max_display=11, show=False,
                      plot_size=None)
    ax1b.set_title("(b)", fontweight="bold")
    ax1b.set_xlabel("SHAP value (impact on occupation score)")

    fig1.tight_layout(w_pad=3)
    path1 = f"{OUT_DIR}\\shap_beeswarm_dual.png"
    fig1.savefig(path1, dpi=DPI, bbox_inches="tight")
    plt.close(fig1)
    print(f"  ✓ Saved: {path1}")

    # ==========================================================================
    # FIGURE 2: Side-by-side SHAP bar importance plots
    # ==========================================================================
    print("  Generating Figure 2: SHAP bar importance ...")
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 5))

    plt.sca(ax2a)
    shap.summary_plot(shap_score, X_latex, plot_type="bar", max_display=11,
                      show=False, plot_size=None)
    ax2a.set_title("(a) L-I curve score ($= -$RMSE)", fontsize=10, fontweight="bold")
    ax2a.set_xlabel("mean |SHAP value|")

    plt.sca(ax2b)
    shap.summary_plot(shap_occ, X_latex, plot_type="bar", max_display=11,
                      show=False, plot_size=None)
    ax2b.set_title("(b) Occupation score ($= -$RMS)", fontsize=10, fontweight="bold")
    ax2b.set_xlabel("mean |SHAP value|")

    fig2.tight_layout(w_pad=3)
    path2 = f"{OUT_DIR}\\shap_bar_dual.png"
    fig2.savefig(path2, dpi=DPI, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Saved: {path2}")

    # ==========================================================================
    # FIGURE 3: SHAP dependence plots for top 3 features per target
    # ==========================================================================
    print("  Generating Figure 3: SHAP dependence plots ...")
    n_top = 3
    fig3, axes3 = plt.subplots(2, n_top, figsize=(4 * n_top, 7))

    for col_idx in range(n_top):
        # Score row
        param_name_score = imp_score.iloc[col_idx]["parameter"]
        feat_idx_score   = list(X_latex.columns).index(param_name_score)
        ax = axes3[0, col_idx]
        plt.sca(ax)
        shap.dependence_plot(feat_idx_score, shap_score, X_latex,
                             ax=ax, show=False)
        ax.set_title(f"L-I score: {param_name_score}", fontsize=9, fontweight="bold")
        ax.set_ylabel("SHAP value" if col_idx == 0 else "")

        # Occupation score row
        param_name_occ = imp_occ.iloc[col_idx]["parameter"]
        feat_idx_occ   = list(X_latex.columns).index(param_name_occ)
        ax = axes3[1, col_idx]
        plt.sca(ax)
        shap.dependence_plot(feat_idx_occ, shap_occ, X_latex,
                             ax=ax, show=False)
        ax.set_title(f"Occ. score: {param_name_occ}", fontsize=9, fontweight="bold")
        ax.set_ylabel("SHAP value" if col_idx == 0 else "")

    fig3.tight_layout(h_pad=2, w_pad=1.5)
    path3 = f"{OUT_DIR}\\shap_dependence_dual.png"
    fig3.savefig(path3, dpi=DPI, bbox_inches="tight")
    plt.close(fig3)
    print(f"  ✓ Saved: {path3}")

    # ==========================================================================
    # FIGURE 4: Combined importance comparison (grouped horizontal bars)
    # ==========================================================================
    print("  Generating Figure 4: Combined importance comparison ...")

    # Merge importance tables on parameter name, sort by score importance
    merged = imp_score[["parameter", "pct"]].rename(columns={"pct": "pct_score"})
    merged = merged.merge(
        imp_occ[["parameter", "pct"]].rename(columns={"pct": "pct_occ"}),
        on="parameter", how="outer"
    ).fillna(0)
    merged = merged.sort_values("pct_score", ascending=True)

    n = len(merged)
    y = np.arange(n)
    bh = 0.35

    COLOR_SCORE = "#2471a3"
    COLOR_OCC   = "#d4681b"

    fig4, ax4 = plt.subplots(figsize=(5.5, 5))
    ax4.barh(y + bh/2, merged["pct_score"], height=bh, color=COLOR_SCORE,
             label="L-I score ($= -$RMSE)", zorder=3)
    ax4.barh(y - bh/2, merged["pct_occ"],   height=bh, color=COLOR_OCC,
             label="Occupation score ($= -$RMS)", zorder=3)

    ax4.set_yticks(y)
    ax4.set_yticklabels([TAU_LATEX.get(p, p) for p in merged["parameter"]])
    ax4.set_xlabel("SHAP importance (%)")
    ax4.grid(True, axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax4.legend(loc="lower right", framealpha=0.9)

    fig4.tight_layout()
    path4 = f"{OUT_DIR}\\shap_importance_comparison.png"
    fig4.savefig(path4, dpi=DPI, bbox_inches="tight")
    plt.close(fig4)
    print(f"  ✓ Saved: {path4}")

    # ==========================================================================
    # Print summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n  Figures saved:")
    print(f"    1. {path1}")
    print(f"    2. {path2}")
    print(f"    3. {path3}")
    print(f"    4. {path4}")

    # Quick comparison
    print(f"\n  Top 3 drivers of L-I SCORE (L-I fit quality):")
    for i in range(3):
        r = imp_score.iloc[i]
        print(f"    {i+1}. {r['parameter']:<30s} {r['pct']:>5.1f}%")

    print(f"\n  Top 3 drivers of OCCUPATION SCORE (population targets):")
    for i in range(3):
        r = imp_occ.iloc[i]
        print(f"    {i+1}. {r['parameter']:<30s} {r['pct']:>5.1f}%")

    print()


if __name__ == "__main__":
    main()

"""
plot_combined_pearson.py
========================
Plots the two Pearson-r correlation analyses side-by-side in a single
horizontal-bar chart.

  • Blue  bars  ── tau vs. score (= -RMSE)       [valid runs only]
  • Orange bars ── tau vs. occupation cost        [valid runs only]

Same tau parameters appear on the same row so the two metrics can be
compared directly.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Correlations\water_87ck.csv"
OUT_PNG  = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Correlations\combined_pearson_correlation.png"

# ── Parameter metadata ───────────────────────────────────────────────────────
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

TARGETS  = {"GS": 0.20, "ES1": 0.40, "ES2": 0.50}
OCC_COLS = {
    "GS":  "occupation_GS_50mA",
    "ES1": "occupation_ES1_50mA",
    "ES2": "occupation_ES2_50mA",
}

# ── rcParams (publication style) ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top":   True,
    "ytick.right": True,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})


def to_log2_ratio(series, baseline):
    ratio = series.astype(float) / baseline
    ratio = ratio.clip(lower=1e-12)
    return np.clip(np.log2(ratio), -MAX_TIME_LOG2, MAX_TIME_LOG2)


def main():
    df = pd.read_csv(CSV_PATH)

    # coalesce old/new column names for occupation state
    df["occupation_GS_50mA"]  = df["occupation_GS_50mA"].fillna(df.get("final_gs",  np.nan))
    df["occupation_ES1_50mA"] = df["occupation_ES1_50mA"].fillna(df.get("final_es1", np.nan))
    df["occupation_ES2_50mA"] = df["occupation_ES2_50mA"].fillna(df.get("final_es2", np.nan))

    tau_keys = list(TAU_BASELINE.keys())
    for k in tau_keys:
        df[f"log2_{k}"] = to_log2_ratio(df[k], TAU_BASELINE[k])

    valid = df[df["invalid"] == False].copy()
    print(f"Total rows: {len(df)}  |  Valid rows: {len(valid)}")

    # ── 1. Pearson r: tau vs score (= -RMSE) ─────────────────────────────────
    score_r  = {}
    score_p  = {}
    for k in tau_keys:
        r, p = pearsonr(valid[f"log2_{k}"].values, valid["score"].values)
        score_r[k] = r
        score_p[k] = p

    # ── 2. Pearson r: tau vs occupation score (= -RMS) ───────────────────────
    sq_devs = sum((valid[OCC_COLS[s]] - TARGETS[s]) ** 2 for s in TARGETS)
    valid["occupation_score"] = -np.sqrt(sq_devs / 3.0)

    occ_r = {}
    occ_p = {}
    for k in tau_keys:
        r, p = pearsonr(valid[f"log2_{k}"].values, valid["occupation_score"].values)
        occ_r[k] = r
        occ_p[k] = p

    # ── Sort rows by |score_r| (descending) ─────────────────────────────────
    sorted_keys = sorted(tau_keys, key=lambda k: abs(score_r[k]), reverse=True)
    # Reverse so that the strongest correlation ends up at the top of the chart
    sorted_keys = sorted_keys[::-1]

    labels   = [TAU_LATEX[k] for k in sorted_keys]
    r_score  = [score_r[k]   for k in sorted_keys]
    r_occ    = [occ_r[k]     for k in sorted_keys]

    n   = len(sorted_keys)
    y   = np.arange(n)
    bh  = 0.35          # bar half-height

    # ── Colours ──────────────────────────────────────────────────────────────
    # Steel-blue family for L-I score; copper/orange family for occupation score
    COLOR_SCORE = "#2471a3"   # steel blue
    COLOR_OCC   = "#d4681b"   # burnt orange

    # ── Figure ───────────────────────────────────────────────────────────────
    GOLDEN = 1.618
    fig_w = 5.5
    fig_h = fig_w / GOLDEN * 1.55     # slightly taller to fit 11 rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bars_score = ax.barh(y + bh/2, r_score, height=bh,
                         color=COLOR_SCORE, label=r"$r$: $\tau$ vs. L-I score ($-$RMSE)",
                         zorder=3)
    bars_occ   = ax.barh(y - bh/2, r_occ,   height=bh,
                         color=COLOR_OCC,   label=r"$r$: $\tau$ vs. occupation score ($-$RMS)",
                         zorder=3)

    # significance markers (p < 0.05)
    sig_kw = dict(va="center", fontsize=7, zorder=4)
    for i, k in enumerate(sorted_keys):
        # score bar
        xs = r_score[i]
        if score_p[k] < 0.05:
            xpos = xs + (0.012 if xs >= 0 else -0.012)
            ha   = "left" if xs >= 0 else "right"
            ax.text(xpos, y[i] + bh/2, "*", color=COLOR_SCORE, ha=ha, **sig_kw)
        # occ bar
        xo = r_occ[i]
        if occ_p[k] < 0.05:
            xpos = xo + (0.012 if xo >= 0 else -0.012)
            ha   = "left" if xo >= 0 else "right"
            ax.text(xpos, y[i] - bh/2, "*", color=COLOR_OCC, ha=ha, **sig_kw)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, zorder=5)
    ax.set_xlabel(r"Pearson $r$")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.7")

    # ── p-value footnote ────────────────────────────────────────────────────
    ax.text(0.01, -0.10, "* $p < 0.05$", transform=ax.transAxes,
            fontsize=7, color="0.4", va="top")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()

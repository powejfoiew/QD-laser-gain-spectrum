"""Analyse the best-scoring run and compare parameter values to correlation trends."""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

df = pd.read_csv(r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Correlations\water_87ck.csv")
valid = df[df["invalid"] == False].copy()

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

def to_log2_ratio(series, baseline):
    ratio = series.astype(float) / baseline
    ratio = ratio.clip(lower=1e-12)
    return np.clip(np.log2(ratio), -MAX_TIME_LOG2, MAX_TIME_LOG2)

# Build log2 columns
tau_keys = list(TAU_BASELINE.keys())
for k in tau_keys:
    valid[f"log2_{k}"] = to_log2_ratio(valid[k], TAU_BASELINE[k])

# Compute correlations
corr_data = {}
for k in tau_keys:
    r, p = pearsonr(valid[f"log2_{k}"].values, valid["score"].values)
    corr_data[k] = {"pearson_r": r, "pearson_p": p}

# Find best scoring row
best_idx = valid["score"].idxmax()
best = valid.loc[best_idx]

print("=" * 80)
print("BEST SCORING RUN")
print("=" * 80)
print(f"Iteration: {best['iteration']:.0f}")
print(f"Score (= -RMSE): {best['score']:.6f}")
print(f"RMSE:            {best['rmse']:.6f}")
print()

hdr = f"{'Parameter':<16} {'Baseline':>14} {'Optimum':>14} {'Ratio':>8} {'log2_ratio':>12} {'Pearson_r':>10} {'Consistent?':>12}"
print(hdr)
print("-" * 90)
for k, base in TAU_BASELINE.items():
    val = best[k]
    ratio = val / base
    lr = np.log2(ratio) if ratio > 0 else float("nan")
    pr = corr_data[k]["pearson_r"]
    # Check consistency: if pearson_r > 0 and score is maximized,
    # we expect the optimum to have log2_ratio > 0 (higher tau = better score)
    # If pearson_r < 0, we expect log2_ratio < 0 (lower tau = better score)
    if abs(pr) < 0.05:
        consistent = "weak corr"
    elif (pr > 0 and lr > 0) or (pr < 0 and lr < 0):
        consistent = "YES"
    else:
        consistent = "**NO**"
    print(f"{k:<16} {base:>14.4e} {val:>14.4e} {ratio:>8.2f} {lr:>12.3f} {pr:>10.4f} {consistent:>12}")

print()
print("=" * 80)
print("Occupation state at 50 mA for best run")
print("=" * 80)
for col in ["occupation_GS_50mA", "occupation_ES1_50mA", "occupation_ES2_50mA"]:
    if col in best.index and not pd.isna(best[col]):
        print(f"  {col}: {best[col]:.4f}")

# Also check the top 10 runs to see if there's a consistent trend
print()
print("=" * 80)
print("TOP 10 SCORING RUNS: key parameter ratios (value / baseline)")
print("=" * 80)
top10 = valid.nlargest(10, "score")
print(f"{'Iter':>6} {'Score':>10} {'tau_aug_GS':>12} {'tau_r_W':>10} {'tau_spon_GS':>12} {'tau_c_e_ES1':>12} {'tau_c_e_GS':>12}")
for _, row in top10.iterrows():
    ratios = {k: row[k]/v for k, v in TAU_BASELINE.items()}
    print(f"{row['iteration']:>6.0f} {row['score']:>10.4f} {ratios['tau_aug_GS']:>12.2f} {ratios['tau_r_W']:>10.2f} {ratios['tau_spon_GS']:>12.2f} {ratios['tau_c_e_ES1']:>12.2f} {ratios['tau_c_e_GS']:>12.2f}")

# Check: are the top runs clustered in a specific region?
print()
print("=" * 80)
print("DISTRIBUTION OF log2(ratio) IN TOP-50 vs ALL VALID RUNS")
print("=" * 80)
top50 = valid.nlargest(50, "score")
print(f"{'Parameter':<16} {'All mean':>10} {'All std':>10} {'Top50 mean':>10} {'Top50 std':>10} {'Shift':>8}")
for k in tau_keys:
    col = f"log2_{k}"
    all_mean = valid[col].mean()
    all_std  = valid[col].std()
    t50_mean = top50[col].mean()
    t50_std  = top50[col].std()
    shift = (t50_mean - all_mean) / all_std if all_std > 0 else 0
    print(f"{k:<16} {all_mean:>10.3f} {all_std:>10.3f} {t50_mean:>10.3f} {t50_std:>10.3f} {shift:>8.2f}sigma")

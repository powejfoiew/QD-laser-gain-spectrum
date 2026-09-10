import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# ── Figure sizing (golden ratio, single-column width) ──────────────────────
GOLDEN_RATIO = 1.618
FIG_WIDTH_IN = 5                      # comfortably under the 6.3 in max
FIG_HEIGHT_IN = FIG_WIDTH_IN / GOLDEN_RATIO

# ── Publication-style rcParams ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,   # embed fonts properly (editable text in PDF)
    "ps.fonttype": 42,
})

# ── Load and Parse Data ───────────────────────────────────────────────────
folder_path = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
csv_files = glob.glob(os.path.join(folder_path, "alpha_freq_data_*mA.csv"))

data = []
for file_path in csv_files:
    filename = os.path.basename(file_path)
    match = re.search(r"alpha_freq_data_([\d.]+)mA\.csv", filename)
    if match:
        current = float(match.group(1))
        df = pd.read_csv(file_path)
        # Extract the final alpha value (last row in 'Alpha' column)
        final_alpha = df['Alpha'].iloc[-1]
        data.append({'current': current, 'final_alpha': final_alpha})

df_plot = pd.DataFrame(data).sort_values('current')

# ── Plotting ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

ax.plot(df_plot['current'], df_plot['final_alpha'], color="tab:blue", marker="o", markersize=4, linewidth=1.2, label=r"$\alpha$ factor")

ax.set_xlabel("Input Current (mA)")
ax.set_ylabel(r"Linewidth Enhancement Factor ($\alpha$)")
ax.grid(True, linewidth=0.4, alpha=0.4)
ax.set_yscale('log')

fig.tight_layout()

# Save output files
pdf_path = os.path.join(folder_path, "LEF_vs_current.pdf")
png_path = os.path.join(folder_path, "LEF_vs_current.png")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=600, bbox_inches="tight")
plt.show()
print(f"Plot saved as {pdf_path} and {png_path}")

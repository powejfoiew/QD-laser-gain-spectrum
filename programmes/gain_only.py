import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Gain_spectrum")
GAIN_FILE = BASE_DIR / "simulated_gain_spectra.csv"
WPD_FILE = BASE_DIR / "wpd_datasets.csv"
OUT_FIG = BASE_DIR / "gain_spectrum.pgf"

# ── Physical Constants & Scaling ────────────────────────────────────────────
q = 1.602176634e-19       # elementary charge (C)
hbar = 1.054571817e-34    # reduced Planck constant (J·s)
omega_0 = (0.959 * q) / hbar  # rad/s (GS transition of central group)
eta_ref = 3.3445          # refractive index
conversion_factor = omega_0 / (3.0e8 * eta_ref)
scaling_factor = conversion_factor * 2.756

# ── Load Data ───────────────────────────────────────────────────────────────
df_gain = pd.read_csv(GAIN_FILE)
wavelength_sim = df_gain["Wavelength_nm"].to_numpy()

raw_wpd = pd.read_csv(WPD_FILE, header=None)
n_datasets = len(raw_wpd.iloc[0, ::2].tolist())
datasets_wpd = []
for i in range(n_datasets):
    x_col = i * 2
    y_col = i * 2 + 1
    df_temp = raw_wpd.iloc[2:, [x_col, y_col]].copy()
    df_temp.columns = ["x", "y"]
    df_temp = df_temp.dropna().astype(float).sort_values("x")
    datasets_wpd.append(df_temp)

current_labels = ["2 mA", "4 mA", "6 mA", "8 mA", "10 mA"]

# ── Custom rcParams Configuration ───────────────────────────────────────────
maj_tick_width = 0.8
min_tick_width = 0.8

plot_params = {
    "figure.dpi": 200,
    "axes.labelsize": 14,
    "axes.linewidth": 1.5,
    "axes.titlesize": 20,
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

# ── Plot Generation ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 3))

sim_colors = plt.cm.Reds(np.linspace(0.4, 1.0, 5))
exp_colors = plt.cm.Blues(np.linspace(0.4, 1.0, 5))

sim_cols = ["Gain_Sim_2.00mA", "Gain_Sim_4.00mA", "Gain_Sim_6.00mA", "Gain_Sim_8.00mA", "Gain_Sim_10.00mA"]

sim_handles, sim_labels = [], []
exp_handles, exp_labels = [], []

# Plot Simulation Data (Columns 1-5 represent 2, 4, 6, 8, 10 mA)
for i, col in enumerate(sim_cols):
    if col in df_gain.columns:
        line, = ax.plot(
            wavelength_sim,
            df_gain[col].to_numpy() * scaling_factor,
            color=sim_colors[i],
            linewidth=1.5
        )
        sim_handles.append(line)
        sim_labels.append(f"{2*(i+1)} mA")

# Plot Experimental Data
for i, (df, label) in enumerate(zip(datasets_wpd, current_labels)):
    line, = ax.plot(
        df["x"], df["y"],
        ls="--", marker="o", markersize=3.0, linewidth=1.2,
        color=exp_colors[i]
    )
    exp_handles.append(line)
    exp_labels.append(label)

# Structured Legend (Two Columns: G&R vs RRE) — matches panel (a) of
# plot_gain_and_emission.py
header_sim_handle = Patch(color="none")
header_exp_handle = Patch(color="none")
header_sim_label = "G\&R"
header_exp_label = "This work"

final_handles = [header_sim_handle] + sim_handles + [header_exp_handle] + exp_handles
final_labels = [header_sim_label] + sim_labels + [header_exp_label] + exp_labels

ax.legend(
    final_handles,
    final_labels,
    ncol=2,
    fontsize=7.5,
    loc="upper left"
)

# Formatting
ax.set_xlim(1150, 1350)
ax.set_ylim(bottom=0)
ax.set_xlabel(r"Wavelength [nm]")
ax.set_ylabel(r"Gain [$\mathrm{cm}^{-1}$]")
#ax.set_title("Material Gain Spectra", fontsize=9, fontweight="bold", loc="left")

plt.tight_layout()

# Save & Show Output
os.makedirs(OUT_FIG.parent, exist_ok=True)
fig.savefig(r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Ultra-stable-lasers-and-photonic-integrated-circuits-for-secure-communications\content\Figures\pgf_figures\gain_spectrum.pgf", bbox_inches="tight")

if __name__ == "__main__":
    plt.show()
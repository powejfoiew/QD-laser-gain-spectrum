#!/usr/bin/env python3
"""
plot_combined_fn.py

Plots the calculated PICWave frequency noise spectrum along with the experimental Bowers data.
Also estimates the Lorentzian linewidth from the white-noise floor of each curve (PICWave +
both experimental datasets) using the Di Domenico et al. (2010) relation: Delta_nu = pi * S_nu(white).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

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

# Colors from Bowers/plot_bowers_data.py
colors = np.array([
    [255, 247, 236], [247, 252, 240], [224, 243, 219], [204, 235, 197],
    [168, 221, 181], [123, 204, 196], [78, 179, 211], [43, 140, 190],
    [183, 79, 43], [8, 88, 158], [11, 37, 58], [182, 160, 157],
    [83, 157, 179], [65, 105, 225]
]) / 256.0


def load_picwave_fn(path, nperseg=2**24):
    print(f"Loading PICWave raw data from '{path}'...")
    with open(path, 'r') as f:
        line1 = f.readline().strip()
        npoints = int(line1.split('//')[0].strip())

    print(f"  Expected points: {npoints:,}")

    # Load using pandas for speed and memory efficiency
    df = pd.read_csv(
        path,
        sep=r'\s+',
        skiprows=2,
        header=None,
        usecols=[0, 2],
        names=["t_ns", "phase_rad"],
        dtype={"t_ns": np.float64, "phase_rad": np.float64},
        nrows=npoints,
        engine="c"
    )

    t_ns = df["t_ns"].to_numpy()
    phase_rad = df["phase_rad"].to_numpy()

    print("  Data loaded. Computing sampling rate...")
    dt_ns = np.mean(np.diff(t_ns[:1000]))
    dt_s = 0.1441E-12  # match hardcoded dt_s from frequency_interpreter.py
    fs = 1.0 / dt_s

    print("  Unwrapping phase...")
    phase_unwrapped = np.unwrap(phase_rad)

    print("  Computing instantaneous frequency...")
    dphase_dt = np.gradient(phase_unwrapped, dt_s)
    inst_freq = dphase_dt / (2.0 * np.pi)

    freq_fluctuations = inst_freq - np.mean(inst_freq)

    # Discard initial 10% transient
    start_index = len(freq_fluctuations) // 10
    freq_fluctuations = freq_fluctuations[start_index:]

    print(f"  Computing PSD with nperseg={nperseg}...")
    f_offset, S_nu = welch(
        freq_fluctuations,
        fs=fs,
        nperseg=nperseg,
        noverlap=None,
        window='hann',
        scaling='density'
    )

    return f_offset, S_nu


def stitching_post_process(freq, psdi):
    freq = np.array(freq)
    psdi = np.array(psdi)
    min_log = int(np.floor(np.log10(np.min(freq))))
    max_log = int(np.floor(np.log10(np.max(freq))))
    n_array = np.arange(min_log, max_log + 1)

    psd0 = psdi.copy()

    i_end = {}
    i_start = {}

    for n in n_array:
        Aa = np.where(freq == 10**n)[0]
        if len(Aa) > 0:
            i_end[n] = Aa[0]
            i_start[n + 1] = Aa[-1]

    i_start[min_log] = 0

    adjust = {}
    for n in range(max_log, min_log - 1, -1):
        if (n + 1) in i_start and n in i_end:
            adjust[n] = 1.0 * psd0[i_start[n + 1]] / psdi[i_end[n]]
            psd0[i_start[n]:i_end[n] + 1] = psdi[i_start[n]:i_end[n] + 1] * adjust[n]

    return freq, psd0


def estimate_linewidth_from_white_noise(f_offset, S_nu, f_low=1e7, f_high=1e8, label=""):
    """
    Estimate the Lorentzian (Schawlow-Townes-like) linewidth from the white
    frequency-noise floor using the Di Domenico et al. (2010) relation:

        Delta_nu = pi * S_nu(white)

    This is only valid where the FN PSD is genuinely flat (white) -- if the
    spectrum still has residual 1/f or feedback-induced structure in this
    window, the estimate will be biased.
    """
    mask = (f_offset >= f_low) & (f_offset <= f_high)
    if not np.any(mask):
        print(f"  [{label}] Warning: no PSD points found in white-noise window "
              f"[{f_low:.2e}, {f_high:.2e}] Hz -- skipping")
        return None, None

    S_white = np.median(S_nu[mask])
    linewidth_hz = np.pi * S_white

    print(f"  [{label}] White noise level (median) over [{f_low:.2e}, {f_high:.2e}] Hz: "
          f"{S_white:.4e} Hz^2/Hz")
    print(f"  [{label}] Estimated Lorentzian linewidth (pi * S_nu): {linewidth_hz:.4e} Hz "
          f"({linewidth_hz/1e3:.2f} kHz)")

    return S_white, linewidth_hz


def main():
    parser = argparse.ArgumentParser(description="Plot combined frequency noise.")
    parser.add_argument("--path", default=r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Frequency Noise\frequency_noise_spectrum.ascii", help="Path to PICWave frequency noise spectrum")
    parser.add_argument("--fmin", type=float, default=1e3, help="Min offset frequency [Hz]")
    parser.add_argument("--fmax", type=float, default=1e8, help="Max offset frequency [Hz]")
    parser.add_argument("--lw-fmin", type=float, default=1e7, help="Min freq for white-noise linewidth fit [Hz]")
    parser.add_argument("--lw-fmax", type=float, default=1e8, help="Max freq for white-noise linewidth fit [Hz]")
    parser.add_argument("--force-recalc", action="store_true", help="Force recalculation of PICWave FN even if cache exists")
    args = parser.parse_args()

    cache_file = "picwave_fn_cache.npz"

    # 1. Load or compute PICWave frequency noise
    if os.path.exists(cache_file) and not args.force_recalc:
        print(f"Loading cached PICWave frequency noise from {cache_file}...")
        cache = np.load(cache_file)
        f_offset = cache["f_offset"]
        S_nu = cache["S_nu"]
    else:
        if not os.path.exists(args.path):
            # Try to look for it one directory up or down if not found locally
            print(f"Error: {args.path} not found.")
            return

        f_offset_full, S_nu_full = load_picwave_fn(args.path)

        # Filter for the plotting range to keep cache file extremely small
        mask = (f_offset_full >= 1e2) & (f_offset_full <= 5e8)
        f_offset = f_offset_full[mask]
        S_nu = S_nu_full[mask]

        np.savez(cache_file, f_offset=f_offset, S_nu=S_nu)
        print(f"Saved frequency noise data to cache '{cache_file}'")

    # 2. Load Bowers data
    files = [r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Frequency Noise\Bowers\210119DFB_Die4Bar4D20_100mA_20C", r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Frequency Noise\Bowers\210119DFB_Die1Bar2D02_160mA_20C"]

    pn = []
    for fn in files:
        filepath = fn
        df = pd.read_csv(filepath, sep=r'\s+', skiprows=12, header=None)
        x = df[0].to_numpy()
        y = df[3].to_numpy()
        x_proc, y_proc = stitching_post_process(x, y)
        pn.append({'x': x_proc, 'y': y_proc})

    # Beta separation line
    # Beta separation line: S_nu = 8*ln(2)*f / pi^2
    beta_line = 8.0 * np.log(2) * f_offset / (np.pi**2)

    # 2.5 Estimate linewidth from the white-noise floor of all three curves
    print("Estimating linewidth from white-noise region for all curves...")
    datasets = [
        {"label": "1000 $\mu$m (Experimental)", "x": pn[0]['x'], "y": pn[0]['y']},
        {"label": "1500 $\mu$m (Experimental)",  "x": pn[1]['x'], "y": pn[1]['y']},
        {"label": "PICWave (Simulated)",         "x": f_offset,  "y": S_nu},
    ]

    for ds in datasets:
        S_white, linewidth_hz = estimate_linewidth_from_white_noise(
            ds["x"], ds["y"], f_low=args.lw_fmin, f_high=args.lw_fmax, label=ds["label"]
        )
        ds["S_white"] = S_white
        ds["linewidth_hz"] = linewidth_hz

    # 3. Create the combined plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Fill background regions matching Bowers/plot_bowers_data.py
    #ax.fill_between([args.fmin, args.fmax], [1e3, 1e3], [1e4, 1e4], color=colors[0], edgecolor=colors[0], label='_nolegend_')
    #ax.fill_between([args.fmin, args.fmax], [1e4, 1e4], [1e5, 1e5], color=colors[2], edgecolor=colors[2], label='_nolegend_')

    # Plot experimental data
    line0, = ax.loglog(datasets[0]["x"], datasets[0]["y"], alpha=0.6, linewidth=1.2, label=datasets[0]["label"])
    line1, = ax.loglog(datasets[1]["x"], datasets[1]["y"], alpha=0.6, linewidth=1.2, label=datasets[1]["label"])

    # Plot PICWave Simulated frequency noise (downsampled for cleaner plot and less density if needed)
    # We can plot it directly with low alpha or thin line since it has high frequency resolution
    mask_picwave = (f_offset >= args.fmin) & (f_offset <= args.fmax)
    line2, = ax.loglog(f_offset[mask_picwave], S_nu[mask_picwave], linewidth=2, alpha=1, label=datasets[2]["label"], zorder=6)

    # Plot beta separation line
    #ax.loglog(f_offset[mask_picwave], beta_line[mask_picwave], color='black', linestyle='--', linewidth=1.4, label=r"$\beta$ separation line",zorder=6)

    # Show the white-noise fit window and resulting linewidth estimate for each curve,
    # colored to match its corresponding data line
    plotted_lines = [line0, line1, line2]
    for ds, line in zip(datasets, plotted_lines):
        if ds["S_white"] is not None:
            ax.hlines(ds["S_white"], args.lw_fmin, args.lw_fmax,
                       colors=line.get_color(), linestyles='--', linewidth=1.4, zorder=7,
                       label=fr"{ds['label']}: $\Delta\nu \approx$ {ds['linewidth_hz']/1e3:.1f} kHz")

    # Styling
    ax.set_xlim(args.fmin, args.fmax)
    ax.set_ylim(1e3, 1e12)
    ax.set_xlabel('Offset Frequency (Hz)')
    ax.set_ylabel(r'Frequency Noise ($\mathrm{Hz^2/Hz}$)')

    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_yticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])

    ax.tick_params(axis='both', which='both', direction='out')
    ax.legend(loc='lower left', frameon=True)
    #ax.grid(True, which='both', ls=':', alpha=0.3)

    plt.tight_layout()
    output_png = "combined_frequency_noise.png"
    plt.savefig(r"combined_frequency_noise.pdf", dpi=300)
    print(f"Successfully generated combined frequency noise plot and saved to '{output_png}'")
    plt.show()


if __name__ == "__main__":
    main()
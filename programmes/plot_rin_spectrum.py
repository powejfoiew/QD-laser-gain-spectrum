#!/usr/bin/env python3
"""
plot_qd_rin_spectra.py

Calculate and plot RIN spectra for QD DFB laser data at different input currents.
Files are named "<I>_mA.ascii" where I=[10,15,25,30,35,40,45,55]
Data format: First line has npoints, then columns: t[ns], real(A), imag(A)

NOTE: The first ~20 ns is current ramp-up transient.
      Only data from t >= 50 ns is used for RIN calculation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path


# Time cutoff: discard initial transient (current ramp-up)
T_START_NS = 50.0  # Only use data from 50 ns onwards


def load_qd_field(file_path):
    """
    Load QD DFB laser field data from .ascii file.
    Returns: (t_ns, field_amplitude)
    """
    print(f"Loading QD field data from '{file_path}'...")

    # Read the first line to get npoints
    with open(file_path, "r") as f:
        first_line = f.readline().strip()

    try:
        npoints_expected = int(first_line.split()[0])
    except (IndexError, ValueError):
        npoints_expected = None
        print(f"Warning: Could not parse npoints from first line of {file_path}")

    # Load data (skip first 2 lines: npoints and comment line)
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=2,
        header=None,
        names=["t_ns", "real_A", "imag_A"],
        dtype={"t_ns": np.float64, "real_A": np.float64, "imag_A": np.float64},
        engine="python"
    )

    if npoints_expected is not None and len(df) != npoints_expected:
        print(f"Warning: {file_path} declared {npoints_expected:,} points, "
              f"read {len(df):,}. Proceeding with what was read.")

    # Calculate field amplitude from real and imaginary parts
    field_amp = np.sqrt(df["real_A"].to_numpy()**2 + df["imag_A"].to_numpy()**2)

    return df["t_ns"].to_numpy(), field_amp


def truncate_transient(t_ns, amp, t_start_ns=T_START_NS):
    """
    Remove initial transient data. Only keep data where t >= t_start_ns.
    """
    mask = t_ns >= t_start_ns
    t_truncated = t_ns[mask]
    amp_truncated = amp[mask]
    print(f"Truncated transient: keeping data from {t_start_ns} ns onwards "
          f"({len(amp_truncated):,} of {len(amp):,} points)")
    return t_truncated, amp_truncated


def resample_uniform(t_ns, amp):
    """
    Resample data onto a uniform time grid.
    """
    print("Resampling onto a uniform time grid...")
    dt_ns = np.median(np.diff(t_ns))
    n = len(t_ns)

    # Create uniform time grid
    t_uniform_ns = t_ns[0] + dt_ns * np.arange(n)

    # Interpolate amplitude onto uniform grid
    amp_uniform = np.interp(t_uniform_ns, t_ns, amp).astype(np.float64)

    return t_uniform_ns, amp_uniform, dt_ns


def compute_rin_psd(amp_uniform, dt_ns, nperseg_pow2=22, noverlap_frac=0.5):
    """
    Compute RIN Power Spectral Density using Welch method.
    """
    print("Computing RIN PSD (Welch method)...")

    # Sampling frequency in Hz (convert ns to seconds)
    fs = 1.0 / (dt_ns * 1e-9)

    # Calculate optical power (P ∝ |E|^2)
    P = amp_uniform ** 2
    P_avg = np.mean(P)

    # Power fluctuations
    dP = P - P_avg

    # Set segment size (power of 2 for FFT efficiency)
    nperseg = min(2 ** nperseg_pow2, len(dP))
    noverlap = int(nperseg * noverlap_frac)

    # Compute PSD using Welch method
    freq, Pxx = welch(
        dP,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend=False,
        scaling="density",
        return_onesided=True,
    )

    # RIN = PSD of power fluctuations / (average power)^2
    rin = Pxx / P_avg ** 2  # Units: 1/Hz

    return freq, rin


def log_bin_average(freq, rin, f_min=1e7, f_max=1e11, points_per_decade=100):
    """
    Average RIN data into logarithmic bins for smoother display.
    """
    print("Applying log-binning for display...")

    # Filter frequencies in desired range
    mask = (freq >= f_min) & (freq <= f_max) & (freq > 0)
    freq_masked = freq[mask]
    rin_masked = rin[mask]

    # Create logarithmic bins
    n_decades = np.log10(f_max / f_min)
    n_bins = max(int(n_decades * points_per_decade), 10)
    bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)

    # Bin the data
    idx = np.digitize(freq_masked, bin_edges)

    freq_binned, rin_binned = [], []
    for i in range(1, n_bins + 1):
        sel = idx == i
        if np.any(sel):
            # Use geometric mean for frequency bin center
            freq_binned.append(np.sqrt(bin_edges[i - 1] * bin_edges[i]))
            # Average RIN in linear scale
            rin_binned.append(np.mean(rin_masked[sel]))

    return np.array(freq_binned), np.array(rin_binned)


def main():
    # Define currents to process
    currents = [10, 15, 25, 30, 35, 40, 45, 55]

    # Define frequency range for plotting
    f_min = 1e7   # 10 MHz
    f_max = 1e11  # 100 GHz

    # Create output directory
    output_dir = "rin_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Colors for different currents
    colors = plt.cm.viridis(np.linspace(0, 1, len(currents)))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Store results
    results = {}

    # Process each current file
    for i, current in enumerate(currents):
        file_name = f"{current}_mA.ascii"
        file_path = os.path.join(r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Damping factor\asciis", file_name)

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {current} mA data...")
        print(f"{'='*60}")

        try:
            # Load data
            t_ns, field_amp = load_qd_field(file_path)

            # TRUNCATE TRANSIENT: remove first 50 ns (ramp-up period)
            t_ns, field_amp = truncate_transient(t_ns, field_amp, t_start_ns=T_START_NS)

            # Resample to uniform time grid
            t_uniform_ns, amp_uniform, dt_ns = resample_uniform(t_ns, field_amp)

            # Compute RIN PSD
            freq, rin = compute_rin_psd(amp_uniform, dt_ns, nperseg_pow2=22)

            # Log-bin average for display
            freq_binned, rin_binned = log_bin_average(freq, rin, f_min, f_max)

            # Convert RIN to dB
            rin_db = 10 * np.log10(rin_binned)

            # Store results
            results[current] = {
                'freq': freq_binned,
                'rin_db': rin_db,
                'color': colors[i]
            }

            # Plot RIN spectrum
            ax.semilogx(freq_binned, rin_db,
                        color=colors[i],
                        linewidth=1.5,
                        label=f"{current} mA")

            print(f"Successfully processed {current} mA")

        except Exception as e:
            print(f"Error processing {current} mA: {e}")

    # Plot styling
    ax.set_xlim(f_min, f_max)
    ax.set_ylim(-170, -120)

    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("RIN (dB/Hz)", fontsize=14)
    ax.set_title("QD DFB Laser RIN Spectra at Different Currents", fontsize=16)

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10, title="Current")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, "qd_rin_spectra_all_currents.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")

    plt.show()

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(results)} out of {len(currents)} current files")
    print(f"Transient cutoff: data before {T_START_NS} ns discarded")
    print(f"Frequency range: {f_min:.0e} Hz to {f_max:.0e} Hz")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
rin_from_picwave.py

Computes the Relative Intensity Noise (RIN) spectrum from a PICWave
raw time-domain field-monitor export (t[ns], abs(A), arg(A)[rad]),
and plots RIN [dB Hz^-1] vs offset frequency [Hz].

RIN(f) = S_deltaP(f) / <P>^2   [Hz^-1],   P(t) = |A(t)|^2

Only the amplitude column is used (phase is skipped -- it's not needed
for RIN, only for frequency-noise/FM analysis).

Usage:
    python rin_from_picwave.py raw_phase_data.ascii [--fmin 1e7] [--fmax 1e10]

Expected file format:

    27758560 // npoints
    //      t[ns]        abs(A)   arg(A)[rad]
    7.204984e-05    0.03178746   -0.02365847
    ...

Notes on approach
------------------
1. PICWave's exported time axis is not guaranteed to be perfectly
   uniformly spaced (the TDTW engine's internal step can vary slightly,
   and/or ASCII rounding introduces jitter). Welch's method (used below)
   assumes uniform sampling, so we interpolate the amplitude trace onto
   a strictly uniform grid first. Because the nominal timestep here
   (tens of fs) is many orders of magnitude finer than the period
   corresponding to the highest frequency we care about (10 GHz -> 100 ps),
   this interpolation introduces negligible error in the frequency range
   of interest.

2. With ~tens of millions of samples, a single periodogram would have
   far more frequency resolution than needed and would be extremely
   noisy point-to-point. Instead we use Welch's method (segment + average)
   to get a reasonably averaged linear-frequency PSD, then further reduce
   noise for the log-log plot by averaging within logarithmically spaced
   frequency bins (same idea as increasing numSlices in PICWave's own
   Spectra panel).
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


def load_picwave_field(path):
    """
    Reads a PICWave ASCII field-monitor export.
    Returns t_ns (float64) and amp (float32) as 1D numpy arrays.
    The phase column is deliberately skipped (not needed for RIN),
    which roughly halves memory/parsing time.
    """
    with open(path, "r") as f:
        first_line = f.readline()
    try:
        npoints_expected = int(first_line.split()[0])
    except (IndexError, ValueError):
        npoints_expected = None
        print("Warning: could not parse npoints from first line; continuing anyway.")

    df = pd.read_csv(
        path,
        delim_whitespace=True,   # fast C-engine whitespace parsing
        skiprows=2,               # skip the npoints line + column-header comment line
        header=None,
        usecols=[0, 1],           # t[ns], abs(A) -- skip arg(A)
        names=["t_ns", "amp"],
        dtype={"t_ns": np.float64, "amp": np.float32},
        engine="c",
    )

    if npoints_expected is not None and len(df) != npoints_expected:
        print(f"Warning: header declared {npoints_expected:,} points, "
              f"read {len(df):,}. Proceeding with what was read.")

    return df["t_ns"].to_numpy(), df["amp"].to_numpy()


def resample_uniform(t_ns, amp):
    """
    Interpolates amp(t) onto a uniform time grid using the median
    timestep found in the data. Returns (t_uniform_ns, amp_uniform, dt_ns).
    """
    dt_ns = np.median(np.diff(t_ns))
    n = len(t_ns)
    t_uniform_ns = t_ns[0] + dt_ns * np.arange(n)

    amp_uniform = np.interp(t_uniform_ns, t_ns, amp).astype(np.float32)

    return t_uniform_ns, amp_uniform, dt_ns


def compute_rin_psd(amp_uniform, dt_ns, nperseg_pow2=22, noverlap_frac=0.5):
    """
    Computes the one-sided RIN PSD via Welch's method.

    Returns freq [Hz], rin [1/Hz] (linear, not dB).
    """
    fs = 1.0 / (dt_ns * 1e-9)  # Hz

    P = amp_uniform.astype(np.float64) ** 2
    P_avg = np.mean(P)
    dP = P - P_avg

    nperseg = min(2 ** nperseg_pow2, len(dP))
    noverlap = int(nperseg * noverlap_frac)

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

    rin = Pxx / P_avg ** 2  # 1/Hz
    n_segments_approx = 1 + (len(dP) - nperseg) // (nperseg - noverlap)

    print(f"  fs = {fs:.4e} Hz, Nyquist = {fs/2:.4e} Hz")
    print(f"  nperseg = {nperseg:,} -> freq resolution = {fs/nperseg:.3e} Hz")
    print(f"  ~{n_segments_approx} Welch segments averaged")

    return freq, rin


def log_bin_average(freq, rin, f_min, f_max, points_per_decade=150):
    """
    Averages rin (linear) within log-spaced frequency bins, for a
    smoother log-log plot. Returns bin-center freq and rin (linear).
    """
    mask = (freq >= f_min) & (freq <= f_max) & (freq > 0)
    freq = freq[mask]
    rin = rin[mask]

    n_decades = np.log10(f_max / f_min)
    n_bins = max(int(n_decades * points_per_decade), 10)
    bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)

    idx = np.digitize(freq, bin_edges)

    freq_binned, rin_binned = [], []
    for i in range(1, n_bins + 1):
        sel = idx == i
        if np.any(sel):
            freq_binned.append(np.sqrt(bin_edges[i - 1] * bin_edges[i]))  # geometric mean
            rin_binned.append(np.mean(rin[sel]))  # average in linear power

    return np.array(freq_binned), np.array(rin_binned)


def main():
    parser = argparse.ArgumentParser(description="Compute RIN spectrum from a PICWave field export.")
    parser.add_argument("path", nargs="?", default=r"C:\Users\josep\Documents\Prompt AWS\27.07.2026\raw_phase_data.ascii",
                         help="Path to the PICWave ASCII field export")
    parser.add_argument("--fmin", type=float, default=1e7, help="Plot lower frequency bound [Hz]")
    parser.add_argument("--fmax", type=float, default=5e10, help="Plot upper frequency bound [Hz]")
    parser.add_argument("--nperseg-pow2", type=int, default=22,
                         help="log2(Welch segment length); larger = better low-f resolution, fewer averages")
    parser.add_argument("--out", default="rin_spectrum.png", help="Output plot filename")
    args = parser.parse_args()

    print(f"Loading PICWave field data from '{args.path}'...")
    t_ns, amp = load_picwave_field(args.path)
    print(f"  {len(t_ns):,} points loaded, spanning {t_ns[-1] - t_ns[0]:.3f} ns")

    print("Resampling onto a uniform time grid...")
    t_ns_u, amp_u, dt_ns = resample_uniform(t_ns, amp)
    print(f"  median dt = {dt_ns * 1e6:.4f} fs")

    print("Computing RIN PSD (Welch)...")
    freq, rin = compute_rin_psd(amp_u, dt_ns, nperseg_pow2=args.nperseg_pow2)

    print("Log-binning for display...")
    freq_b, rin_b = log_bin_average(freq, rin, args.fmin, args.fmax)
    rin_db = 10 * np.log10(rin_b)

    print("Plotting...")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.semilogx(freq_b, rin_db, color="#c1440e", lw=1.1, label="Simulated")
    ax.axhline(-150, color="k", ls="--", lw=1)
    ax.set_xlim(args.fmin, args.fmax)
    ax.set_ylim(-160, -120)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"RIN (dB Hz$^{-1}$)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
# =============================================================================
# RIN spectrum from raw PICWave time-domain data
# =============================================================================
import os
import argparse
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import welch


# -----------------------------------------------------------------------------
# 0. ARGUMENT PARSER
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot RIN spectrum from PICWave time-domain field data."
    )
    parser.add_argument(
        "--start-time", type=float, default=50.0,
        help="Discard data before this time in ns (default: 50 ns)"
    )
    parser.add_argument(
        "--fmin", type=float, default=1e7,
        help="Lower frequency bound for plot in Hz (default: 1e7)"
    )
    parser.add_argument(
        "--fmax", type=float, default=1e11,
        help="Upper frequency bound for plot in Hz (default: 1e11)"
    )
    parser.add_argument(
        "--nperseg-pow2", type=int, default=22,
        help="Welch segment length = 2**this (default: 22)"
    )
    parser.add_argument(
        "--noverlap-frac", type=float, default=0.5,
        help="Welch overlap fraction (default: 0.5)"
    )
    parser.add_argument(
        "--points-per-decade", type=int, default=150,
        help="Log-bin resolution for display (default: 150)"
    )
    parser.add_argument(
        "--no-experimental", action="store_true",
        help="Skip experimental data overlay"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: auto-generated based on start-time)"
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 1. MODEL & EXPERIMENTAL DATA PATHS — edit these
# -----------------------------------------------------------------------------
MODELS = [
    {
        "label": "QD sim. (single level)",
        "path": r"C:\Users\josep\Documents\MRes mini-project 2\Figures\RIN\single_level_phases.ascii",
        "color": "#ff7f0e",
    },
    # Add a second dict here for the multilevel model, e.g.:
    # {
    #     "label": "QD sim. (multilevel)",
    #     "path": r"C:\Users\josep\Documents\MRes mini-project 2\Figures\RIN\multilevel_phases.ascii",
    #     "color": "#2ca02c",
    # },
]

QW_MAT = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\RIN\Fig2b\230215_QW_RIN.mat"
QD_MAT = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\RIN\Fig2b\230301_QD_D3B3_D08_RIN.mat"


# -----------------------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------------------
def load_picwave_field(path):
    """
    Load a PICWave .ascii field file.

    Expected format:
        Line 1: npoints // comment
        Line 2: // column headers
        Line 3+: t[ns]  abs(A)  arg(A)[rad]
    """
    print(f"Loading PICWave field data from '{path}'...")

    with open(path, "r") as f:
        first_line = f.readline()
    try:
        npoints_expected = int(first_line.split()[0])
    except (IndexError, ValueError):
        npoints_expected = None
        print("  Warning: could not parse npoints from first line; continuing anyway.")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=2,
        header=None,
        usecols=[0, 1],
        names=["t_ns", "amp"],
        dtype={"t_ns": np.float64, "amp": np.float64},
        engine="c",
    )

    if npoints_expected is not None and len(df) != npoints_expected:
        print(f"  Warning: header declared {npoints_expected:,} points, "
              f"read {len(df):,}. Proceeding with what was read.")

    t_ns = df["t_ns"].to_numpy()
    amp = df["amp"].to_numpy()

    print(f"  Loaded {len(t_ns):,} samples, t = [{t_ns[0]:.6g}, {t_ns[-1]:.6g}] ns")
    return t_ns, amp


def apply_start_time_cutoff(t_ns, amp, start_time_ns):
    """Discard all samples with t_ns < start_time_ns."""
    mask = t_ns >= start_time_ns
    n_dropped = len(t_ns) - np.count_nonzero(mask)
    print(f"  Start-time cutoff at {start_time_ns:g} ns: dropping {n_dropped:,} / {len(t_ns):,} samples "
          f"({n_dropped / len(t_ns) * 100:.1f}%).")
    if not np.any(mask):
        raise ValueError(
            f"start_time_ns={start_time_ns} is beyond the last sample "
            f"(t_ns max = {t_ns.max():g}). No data left."
        )
    return t_ns[mask], amp[mask]


def check_uniformity_and_get_dt(t_ns, rtol=1e-4):
    """
    Check whether the time array is already uniformly spaced.
    Returns (is_uniform, dt_ns).
    """
    diffs = np.diff(t_ns)

    # Filter out any zero or negative diffs (shouldn't happen with clean data)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        raise ValueError("All time differences are zero or negative — cannot determine dt.")

    dt_ns = np.median(positive_diffs)

    # Check if all diffs are close to the median
    is_uniform = np.allclose(diffs, dt_ns, rtol=rtol)
    return is_uniform, dt_ns


def resample_uniform(t_ns, amp):
    """
    Ensure the data is on a uniform time grid.
    If already uniform, skip interpolation. Otherwise, interpolate.
    """
    print("  Checking time-grid uniformity...")
    is_uniform, dt_ns = check_uniformity_and_get_dt(t_ns)

    if dt_ns <= 0:
        raise ValueError("Could not determine a positive time step from the data.")

    fs = 1.0 / (dt_ns * 1e-9)
    print(f"  dt = {dt_ns:.6g} ns  →  fs = {fs:.4e} Hz")

    if is_uniform:
        print("  Data is already uniformly spaced. No interpolation needed.")
        return t_ns, amp, dt_ns
    else:
        print("  Data is NOT uniformly spaced. Interpolating onto uniform grid...")
        n = len(t_ns)
        t_uniform_ns = t_ns[0] + dt_ns * np.arange(n)
        amp_uniform = np.interp(t_uniform_ns, t_ns, amp)
        return t_uniform_ns, amp_uniform, dt_ns


def compute_rin_psd(amp_uniform, dt_ns, nperseg_pow2=22, noverlap_frac=0.5):
    """
    Compute the RIN power spectral density using Welch's method.
    Returns (freq_hz, rin_per_hz).
    """
    print("  Computing RIN PSD (Welch)...")

    if dt_ns <= 0:
        raise ValueError(f"Invalid dt_ns={dt_ns}. Cannot compute PSD.")

    fs = 1.0 / (dt_ns * 1e-9)  # Hz

    # Optical power = |amplitude|^2
    P = amp_uniform.astype(np.float64) ** 2
    P_avg = np.mean(P)

    if P_avg <= 0:
        raise ValueError("Average optical power is zero or negative — cannot normalise RIN.")

    print(f"  <P> = {P_avg:.6e} (arb. units)")

    # Fluctuations about the mean
    dP = P - P_avg

    # Welch parameters
    nperseg = min(2 ** nperseg_pow2, len(dP))
    actual_pow2 = int(np.log2(nperseg))
    if actual_pow2 < nperseg_pow2:
        print(f"  Warning: requested nperseg=2^{nperseg_pow2} but only {len(dP):,} samples "
              f"available. Using nperseg=2^{actual_pow2} = {nperseg:,}.")
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

    # RIN = S_dP(f) / <P>^2   [units: 1/Hz]
    rin = Pxx / (P_avg ** 2)

    print(f"  PSD computed: {len(freq):,} frequency bins, "
          f"f = [{freq[1]:.3e}, {freq[-1]:.3e}] Hz")

    return freq, rin


def log_bin_average(freq, rin, f_min, f_max, points_per_decade=150):
    """Log-bin the RIN spectrum for cleaner display."""
    print("  Log-binning for display...")
    mask = (freq >= f_min) & (freq <= f_max) & (freq > 0)
    freq = freq[mask]
    rin = rin[mask]

    if len(freq) == 0:
        print("  Warning: no data points in the requested frequency range.")
        return np.array([]), np.array([])

    n_decades = np.log10(f_max / f_min)
    n_bins = max(int(n_decades * points_per_decade), 10)
    bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)

    idx = np.digitize(freq, bin_edges)

    freq_binned, rin_binned = [], []
    for i in range(1, n_bins + 1):
        sel = idx == i
        if np.any(sel):
            freq_binned.append(np.sqrt(bin_edges[i - 1] * bin_edges[i]))  # geometric mean
            rin_binned.append(np.mean(rin[sel]))

    return np.array(freq_binned), np.array(rin_binned)


def smooth_data(y, span=20):
    """Simple rolling-mean smoother for experimental traces."""
    return pd.Series(y).rolling(window=span, min_periods=1, center=True).mean().to_numpy()


# -----------------------------------------------------------------------------
# 3. MAIN
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    START_TIME_NS = args.start_time
    FMIN = args.fmin
    FMAX = args.fmax
    NPERSEG_POW2 = args.nperseg_pow2
    NOVERLAP_FRAC = args.noverlap_frac
    POINTS_PER_DECADE = args.points_per_decade
    INCLUDE_EXPERIMENTAL = not args.no_experimental

    if args.output is not None:
        OUT_PATH = args.output
    else:
        OUT_PATH = (
            r"C:\Users\josep\Documents\MRes mini-project 2\Figures\RIN"
            f"\\rin_spectrum_start{START_TIME_NS:.0f}ns.png"
        )

    # -------------------------------------------------------------------------
    # 3a. Compute RIN for each simulation model
    # -------------------------------------------------------------------------
    results = []
    for m in MODELS:
        if not os.path.exists(m["path"]):
            print(f"Error: file not found: {m['path']}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {m['label']}")
        print(f"{'='*60}")

        # Load
        t_ns, amp = load_picwave_field(m["path"])

        # Apply start-time cutoff
        t_ns, amp = apply_start_time_cutoff(t_ns, amp, START_TIME_NS)

        # Ensure uniform grid
        t_ns_u, amp_u, dt_ns = resample_uniform(t_ns, amp)

        # Compute RIN PSD
        freq, rin = compute_rin_psd(
            amp_u, dt_ns,
            nperseg_pow2=NPERSEG_POW2,
            noverlap_frac=NOVERLAP_FRAC,
        )

        # Log-bin (extend slightly beyond FMAX for edge safety, then clip when plotting)
        fmax_nyquist = 0.5 / (dt_ns * 1e-9)
        bin_fmax = min(FMAX * 2, fmax_nyquist)
        freq_binned, rin_binned = log_bin_average(
            freq, rin, FMIN, bin_fmax,
            points_per_decade=POINTS_PER_DECADE,
        )

        if len(freq_binned) == 0:
            print(f"  Warning: no binned data for '{m['label']}'. Skipping.")
            continue

        # Convert to dB/Hz — floor at 1e-30 to avoid log10(0)
        rin_db = 10 * np.log10(np.maximum(rin_binned, 1e-30))

        results.append({
            "label": m["label"],
            "color": m["color"],
            "freq": freq_binned,
            "rin_db": rin_db,
        })

        # Optional: save cache
        cache_path = os.path.join(
            os.path.dirname(m["path"]),
            f"picwave_rin_cachew_{m['label'].replace(' ', '_')}.npz"
        )
        np.savez(cache_path, freq=freq_binned, rin=rin_binned, rin_db=rin_db)
        print(f"  Saved RIN cache to '{cache_path}'")

    # -------------------------------------------------------------------------
    # 3b. Optional experimental overlay (QW / QD)
    # -------------------------------------------------------------------------
    exp_traces = []
    if INCLUDE_EXPERIMENTAL:
        if os.path.exists(QW_MAT) and os.path.exists(QD_MAT):
            print("\nLoading experimental data...")
            qw_mat = scipy.io.loadmat(QW_MAT)
            qd_mat = scipy.io.loadmat(QD_MAT)
            qw_fr = qw_mat["b"].squeeze()
            qw_rin = qw_mat["bb"].squeeze()
            qd_fr = qd_mat["b"].squeeze()
            qd_rin = qd_mat["bb"].squeeze()
            exp_traces.append((
                "QW exp. (Dong et al. 2024)",
                qw_fr,
                smooth_data(qw_rin, 20),
                (8 / 256, 88 / 256, 158 / 256),
            ))
            exp_traces.append((
                "QD exp. (Dong et al. 2024)",
                qd_fr,
                smooth_data(qd_rin, 20),
                (183 / 256, 79 / 256, 43 / 256),
            ))
        else:
            print("Warning: experimental .mat files not found; skipping overlay.")

    # -------------------------------------------------------------------------
    # 3c. Plot
    # -------------------------------------------------------------------------
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, fr, rin_smooth, color in exp_traces:
        ax.semilogx(fr, rin_smooth, color=color, lw=1.5, label=label)

    for r in results:
        ax.semilogx(r["freq"], r["rin_db"], color=r["color"], lw=1.5, label=r["label"])

    ax.set_xlim(FMIN, FMAX)
    ax.set_ylim(-160, -120)
    ax.set_xlabel("Frequency (Hz)", fontsize=16)
    ax.set_ylabel("RIN (dB/Hz)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_title(f"RIN spectrum (data from t ≥ {START_TIME_NS:g} ns)", fontsize=13)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300)
    plt.show()
    print(f"\nSaved to '{OUT_PATH}'")


if __name__ == "__main__":
    main()
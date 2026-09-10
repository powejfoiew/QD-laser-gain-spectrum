#!/usr/bin/env python3
"""
Fit the relaxation-oscillation (RO) resonance in laser RIN spectra.

Model (Eq. 6 of the reference text)
------------------------------------
    RIN(omega) = a + b * omega^2 / [ (omega^2 - omega_RO^2)^2 + omega^2 * gamma^2 ]

where omega = 2*pi*f is the angular offset frequency, omega_RO = 2*pi*f_RO is
the RO angular frequency, and gamma is the damping factor (rad/s).

For each input .cht file this script:
  1. Parses the (frequency [MHz], RIN [dB/Hz]) trace.
  2. Restricts to a fit window (default 1-45 GHz, see NOTE below) and
     denoises it (median filter to reject spike outliers + Savitzky-Golay
     smoothing) before fitting -- these RIN traces are single noisy sweeps,
     not averaged spectra, so fitting the raw data directly is unstable.
  3. Fits Eq. (6) with scipy.optimize.curve_fit, working in dB space with
     log-parametrised (a, b, omega_RO, gamma) to keep the many-decades-wide
     parameters well conditioned for the optimizer.
  4. Reports f_RO, gamma, and derived quantities; flags fits where no
     resolvable resonance was found (fully damped / overdamped case).
  5. Performs the K-factor regression gamma = K*f_RO^2 + gamma0 (Eq. 7)
     across all currents with a resolvable resonance. IMPORTANT: gamma here
     is the *angular* damping rate exactly as it appears in the resonance
     denominator (omega^2-omega_RO^2)^2 + omega^2*gamma^2 -- the standard
     convention pairs this angular gamma directly with the *linear* ROF,
     f_RO in GHz (not gamma/2pi). With gamma in Grad/s (1e9 rad/s) and f_RO
     in GHz, K comes out directly in ns, matching standard usage (e.g. K~0.2
     to 1 ns is typical for a good QW laser).
  6. Plots the RIN spectra with the fitted curves overlaid, plus a second
     panel showing gamma vs f_RO^2 with the K-factor line (as in Fig. S5 of
     the reference text).

NOTE on the fit window
-----------------------
The user-requested analysis window is 1e9-8e11 Hz. The data only extends to
~1e11 Hz, and inspecting the traces shows a broad, current-independent
feature (a dip around ~50 GHz followed by a rise toward the edge of the
data) that is common to every curve regardless of bias -- i.e. it is a
measurement/system artifact (e.g. photodetector or cable response), not
part of the laser's relaxation-oscillation resonance. Including it would
bias every fit in the same direction, so by default the upper fit bound is
capped at 45 GHz (well clear of the artifact) even though the full spectrum
is still shown in the plot for context. Override with --fit-fmax if you
want to include it or restrict further.

Usage
-----
    python fit_relaxation_oscillation.py *.cht
    python fit_relaxation_oscillation.py --dir /path/to/folder
    python fit_relaxation_oscillation.py *.cht --fit-fmin 1e9 --fit-fmax 4.5e10
"""

import argparse
import csv
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import medfilt, savgol_filter


# --------------------------------------------------------------------------
# File parsing
# --------------------------------------------------------------------------
def parse_cht(filepath):
    """Parse a .cht file and return (freq_hz, rin_dbhz) as numpy arrays."""
    freqs = []
    rin = []
    df = pd.read_csv(filepath, skiprows=2,names=['frequency','RIN_dB'])
    freqs = 1E6*df['frequency']
    rin   = df['RIN_dB']

    return np.array(freqs), np.array(rin)


def extract_label(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    match = re.search(r"(\d+(?:\.\d+)?)\s*_?\s*mA", stem, re.IGNORECASE)
    if match:
        return f"{match.group(1)} mA"
    return stem


def extract_current(filepath):
    """Numeric current in mA if present in the filename, else NaN."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*_?\s*mA", os.path.basename(filepath), re.IGNORECASE)
    return float(match.group(1)) if match else float("nan")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def rin_model_linear(omega, a, b, omega_ro, gamma):
    return a + b * omega ** 2 / ((omega ** 2 - omega_ro ** 2) ** 2 + (omega * gamma) ** 2)


def rin_model_db_logparams(omega, log_a, log_b, log_omega_ro, log_gamma):
    """Same model, parametrised by log10 of each (positive) parameter, and
    evaluated in dB. Fitting in log-parameter space keeps the optimizer
    well-behaved given the huge dynamic range of a, b, omega_ro, gamma."""
    a, b, omega_ro, gamma = 10 ** log_a, 10 ** log_b, 10 ** log_omega_ro, 10 ** log_gamma
    lin = rin_model_linear(omega, a, b, omega_ro, gamma)
    return 10 * np.log10(np.clip(lin, 1e-300, None))


# --------------------------------------------------------------------------
# Smoothing + fitting for one curve
# --------------------------------------------------------------------------
def smooth_dbdata(freq_hz, rin_dbhz, medfilt_kernel=51, savgol_window=201, savgol_poly=3):
    n = freq_hz.size
    mk = min(medfilt_kernel, n - (1 - n % 2))
    if mk < 3:
        mk = 3
    if mk % 2 == 0:
        mk += 1
    med = medfilt(rin_dbhz, kernel_size=mk)

    sw = min(savgol_window, n - (1 - n % 2))
    if sw < 5:
        sw = 5 if n >= 5 else n if n % 2 == 1 else n - 1
    if sw % 2 == 0:
        sw += 1
    sw = max(sw, savgol_poly + 2 + (savgol_poly + 2) % 2 + 1)  # ensure > polyorder
    return savgol_filter(med, window_length=sw, polyorder=savgol_poly, mode="interp")


def fit_relaxation_oscillation(freq_hz, rin_dbhz, fit_fmin, fit_fmax,
                                min_peak_db=1.0, edge_tol=0.97):
    """Fit Eq. (6) to one RIN trace restricted to [fit_fmin, fit_fmax].

    Returns a dict with the fit parameters and diagnostics, or None if there
    are not enough points in range to attempt a fit.
    """
    fmax_eff = min(fit_fmax, freq_hz.max())
    mask = (freq_hz >= fit_fmin) & (freq_hz <= fmax_eff)
    f = freq_hz[mask]
    y_db = rin_dbhz[mask]
    if f.size < 20:
        return None

    y_smooth = smooth_dbdata(f, y_db)
    omega = 2 * np.pi * f

    # Initial guesses from the smoothed curve
    peak_i = np.argmax(y_smooth)
    f_ro0 = f[peak_i]
    omega_ro0 = 2 * np.pi * f_ro0
    gamma0 = 0.3 * omega_ro0
    a0 = 10 ** (np.percentile(y_smooth, 5) / 10.0)
    peak_lin0 = 10 ** (y_smooth[peak_i] / 10.0)
    b0 = max(peak_lin0 - a0, 1e-30) * gamma0 ** 2
    p0 = [np.log10(a0), np.log10(max(b0, 1e-300)), np.log10(omega_ro0), np.log10(gamma0)]

    try:
        popt, pcov = curve_fit(rin_model_db_logparams, omega, y_smooth, p0=p0, maxfev=50000)
    except RuntimeError:
        return {"success": False, "freq": f, "rin_db": y_db, "rin_smooth": y_smooth}

    a, b, omega_ro, gamma = (10 ** popt[0], 10 ** popt[1], 10 ** popt[2], 10 ** popt[3])
    f_RO = omega_ro / (2 * np.pi)

    peak_lin = a + b / gamma ** 2
    peak_above_floor_db = 10 * np.log10(peak_lin) - 10 * np.log10(a)

    fit_curve_db = rin_model_db_logparams(omega, *popt)
    ss_res = np.sum((y_smooth - fit_curve_db) ** 2)
    ss_tot = np.sum((y_smooth - y_smooth.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    reliable = (peak_above_floor_db >= min_peak_db) and (f_RO < edge_tol * fmax_eff)

    return {
        "success": True,
        "reliable": reliable,
        "a": a, "b": b, "omega_ro": omega_ro, "gamma": gamma,
        "f_RO": f_RO, "gamma_over_2pi": gamma / (2 * np.pi),
        "peak_above_floor_db": peak_above_floor_db,
        "r2": r2, "popt": popt, "pcov": pcov,
        "freq": f, "rin_db": y_db, "rin_smooth": y_smooth,
        "fit_fmin": fit_fmin, "fit_fmax": fmax_eff,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fit RO resonance in RIN spectra (.cht files).")
    parser.add_argument("files", nargs="*", help="One or more .cht files")
    parser.add_argument("--dir", default=r'C:\Users\josep\Documents\MRes mini-project 2\Figures\Damping factor\RIN_charts_reloaded', help="Directory to search for *.cht files")
    parser.add_argument("--fit-fmin", type=float, default=1e9,
                         help="Lower bound of the fit window, Hz (default 1e9)")
    parser.add_argument("--fit-fmax", type=float, default=4.5e10,
                         help="Upper bound of the fit window, Hz (default 4.5e10; "
                              "see NOTE in module docstring about the system artifact above ~50 GHz)")
    parser.add_argument("--display-fmin", type=float, default=None,
                         help="Lower bound for the x-axis of the spectrum plot (default: data min)")
    parser.add_argument("--display-fmax", type=float, default=8e10,
                         help="Upper bound for the x-axis of the spectrum plot (default: data max)")
    parser.add_argument("-o", "--output", default="rin_fit.png", help="Output image path")
    parser.add_argument("--csv", default="rin_fit_results.csv", help="Output CSV path for fit results")
    args = parser.parse_args()

    files = list(args.files)
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not files:
        files = sorted(glob.glob("*.csv"))
    if not files:
        sys.exit("No .cht files found. Pass filenames, use --dir, or run from a folder containing .cht files.")

    files = sorted(files, key=lambda fp: (extract_current(fp), fp))

    results = []
    for fp in files:
        freq_hz, rin_dbhz = parse_cht(fp)
        if freq_hz.size == 0:
            print(f"Warning: no data parsed from {fp}", file=sys.stderr)
            continue
        label = extract_label(fp)
        current_mA = extract_current(fp)
        fit = fit_relaxation_oscillation(freq_hz, rin_dbhz, args.fit_fmin, args.fit_fmax)
        results.append({"file": fp, "label": label, "current_mA": current_mA,
                         "freq_full": freq_hz, "rin_full": rin_dbhz, "fit": fit})

    # ---------------- console report ----------------
    print("\n=== Relaxation-oscillation fit results ===")
    print(f"Fit window: {args.fit_fmin:.3g} - {min(args.fit_fmax, max(r['freq_full'].max() for r in results)):.3g} Hz\n")
    header = f"{'Label':>10} | {'f_RO (GHz)':>10} | {'gamma (Grad/s)':>15} | {'gamma/2pi (GHz)':>16} | {'peak (dB)':>9} | {'R2':>6} | note"
    print(header)
    print("-" * len(header))

    fit_rows = []
    for r in results:
        fit = r["fit"]
        if fit is None:
            print(f"{r['label']:>10} | {'--':>10} | {'--':>15} | {'--':>16} | {'--':>9} | {'--':>6} | not enough points in fit window")
            continue
        if not fit["success"]:
            print(f"{r['label']:>10} | {'--':>10} | {'--':>15} | {'--':>16} | {'--':>9} | {'--':>6} | fit did not converge")
            continue
        note = "" if fit["reliable"] else "peak not clearly resolved (overdamped / at edge)"
        print(f"{r['label']:>10} | {fit['f_RO']/1e9:10.3f} | {fit['gamma']/1e9:15.3f} | "
              f"{fit['gamma_over_2pi']/1e9:16.3f} | {fit['peak_above_floor_db']:9.2f} | {fit['r2']:6.3f} | {note}")
        fit_rows.append({
            "label": r["label"], "current_mA": r["current_mA"],
            "a": fit["a"], "b": fit["b"],
            "f_RO_GHz": fit["f_RO"] / 1e9, "gamma_Grad_s": fit["gamma"] / 1e9,
            "gamma_over_2pi_GHz": fit["gamma_over_2pi"] / 1e9,
            "peak_above_floor_dB": fit["peak_above_floor_db"],
            "R2": fit["r2"], "reliable": fit["reliable"],
        })

    # ---------------- K-factor regression ----------------
    # gamma_Grad_s is the *angular* damping rate (1e9 rad/s), paired directly
    # with the *linear* f_RO_GHz -- this is the standard convention, and it's
    # what makes K come out in nanoseconds. Do NOT use gamma/2pi here.
    reliable_rows = [row for row in fit_rows if row["reliable"]]
    K = gamma0 = None
    if len(reliable_rows) >= 2:
        x = np.array([row["f_RO_GHz"] ** 2 for row in reliable_rows])
        y = np.array([row["gamma_Grad_s"] for row in reliable_rows])
        K, gamma0 = np.polyfit(x, y, 1)
        print(f"\nK-factor fit (gamma = K * f_RO^2 + gamma0), using {len(reliable_rows)} "
              f"reliable point(s): {[row['label'] for row in reliable_rows]}")
        print(f"  K      = {K:.4f} ns")
        print(f"  gamma0 = {gamma0:.4f} Grad/s  (= {gamma0/(2*np.pi):.4f} GHz as a linear-frequency damping offset)")
    else:
        print("\nNot enough reliable fits (need >= 2) to perform the K-factor regression.")

    # ---------------- CSV export ----------------
    if fit_rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fit_rows)
        print(f"\nSaved fit results to {args.csv}")

    # ---------------- plotting ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, r in enumerate(results):
        color = colors[i % len(colors)]
        ax1.plot(r["freq_full"], r["rin_full"], linewidth=0.6, alpha=0.35, color=color)
        fit = r["fit"]
        if fit is not None and fit.get("success"):
            ax1.plot(fit["freq"], fit["rin_smooth"], linewidth=1.3, color=color, label=r["label"])
            fit_curve_db = rin_model_db_logparams(2 * np.pi * fit["freq"], *fit["popt"])
            style = "--" if fit["reliable"] else ":"
            ax1.plot(fit["freq"], fit_curve_db, style, linewidth=2.0, color="black",
                      alpha=0.9 if fit["reliable"] else 0.5)
        else:
            ax1.plot([], [], color=color, label=r["label"])  # keep legend entry

    ax1.set_xscale("log")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("RIN (dB/Hz)")
    ax1.set_title("RIN spectra with relaxation-oscillation fits\n(dashed = fit; dotted = peak not clearly resolved)")
    if args.display_fmin is not None or args.display_fmax is not None:
        ax1.set_xlim(args.display_fmin, args.display_fmax)
    ax1.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax1.legend(loc="best", fontsize=8, title="Bias current")

    if reliable_rows:
        xg = np.array([row["f_RO_GHz"] ** 2 for row in reliable_rows])
        yg = np.array([row["gamma_Grad_s"] for row in reliable_rows])
        ax2.scatter(xg, yg, color="C0", zorder=3)
        for row in reliable_rows:
            ax2.annotate(row["label"], (row["f_RO_GHz"] ** 2, row["gamma_Grad_s"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
        if K is not None:
            xx = np.linspace(0, xg.max() * 1.15, 50)
            ax2.plot(xx, K * xx + gamma0, "k--", linewidth=1.5,
                      label=f"K = {K:.3f} ns\n$\\gamma_0$ = {gamma0:.2f} Grad/s")
        ax2.set_xlabel(r"$f_{RO}^2$ (GHz$^2$)")
        ax2.set_ylabel(r"$\gamma$ (Grad/s)")
        ax2.set_title("Damping factor vs. squared ROF (K-factor plot)")
        ax2.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        ax2.legend(loc="best", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Not enough reliable fits\nfor K-factor regression",
                  ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xticks([]); ax2.set_yticks([])

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
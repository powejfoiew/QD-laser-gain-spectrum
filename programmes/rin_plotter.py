#!/usr/bin/env python3
"""
Plot RIN (Relative Intensity Noise) spectra from .cht data files.

Each .cht file is a simple text export (SGCHART format) containing a short
header followed by two-column CSV data:
    Xvals  -> offset frequency, in MHz
    Yvals1 -> RIN, in dB/Hz

Usage
-----
    python plot_rin.py file1.cht file2.cht ...
    python plot_rin.py *.cht
    python plot_rin.py --dir /path/to/folder

If no files/--dir are given, the script looks for *.cht files in the
current directory.

The current (in mA) is auto-extracted from each filename if present
(e.g. "..._25_mA.cht" -> "25 mA"); otherwise the filename itself is
used as the legend label.
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_cht(filepath):
    """Parse a .cht file and return (freq_hz, rin_dbhz) as numpy arrays."""
    freqs = []
    rin = []
    df = pd.read_csv(filepath, skiprows=2,names=['frequency','RIN_dB'])
    freqs = df['frequency']
    rin   = df['RIN_dB']

    return np.array(freqs), np.array(rin)


def extract_label(filepath):
    """Pull a '<value> mA' style label out of the filename, else use the stem."""
    stem = os.path.splitext(os.path.basename(filepath))[0]
    match = re.search(r"(\d+(?:\.\d+)?)\s*_?\s*mA", stem, re.IGNORECASE)
    if match:
        return f"{match.group(1)} mA"
    return stem


def main():
    parser = argparse.ArgumentParser(description="Plot RIN spectra from .cht files.")
    parser.add_argument("files", nargs="*", help="One or more .cht files")
    parser.add_argument("--dir", default=r'C:\Users\josep\Documents\MRes mini-project 2\Figures\Damping factor\RIN_charts_reloaded', help="Directory to search for *.cht files")
    parser.add_argument("--fmin", type=float, default=None, help="Minimum frequency (Hz) to display")
    parser.add_argument("--fmax", type=float, default=8e4, help="Maximum frequency (Hz) to display")
    parser.add_argument("--tmin-note", default=None,
                         help="Optional note for the title, e.g. 'data from t >= 50 ns'")
    parser.add_argument("-o", "--output", default="rin_spectrum.png", help="Output image path")
    args = parser.parse_args()

    files = list(args.files)
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not files:
        files = sorted(glob.glob("*.csv"))
    if not files:
        sys.exit("No .csv files found. Pass filenames, use --dir, or run from a folder containing .cht files.")

    # Sort by the numeric current value when possible, for a tidy legend order.
    def sort_key(fp):
        m = re.search(r"(\d+(?:\.\d+)?)\s*_?\s*mA", os.path.basename(fp), re.IGNORECASE)
        return float(m.group(1)) if m else float("inf")

    files = sorted(files, key=sort_key)

    fig, ax = plt.subplots(figsize=(9, 6.2))

    for fp in files:
        freq_hz, rin_dbhz = parse_cht(fp)
        if freq_hz.size == 0:
            print(f"Warning: no data parsed from {fp}", file=sys.stderr)
            continue
        label = extract_label(fp)
        ax.plot(freq_hz, rin_dbhz, linewidth=1.0, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("RIN (dB/Hz)")

    title = "RIN spectrum"
    if args.tmin_note:
        title += f" ({args.tmin_note})"
    ax.set_title(title)

    if args.fmin is not None or args.fmax is not None:
        ax.set_xlim(args.fmin, args.fmax)

    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(r'C:\Users\josep\Documents\MRes mini-project 2\Figures\Damping factor\RIN_charts_reloaded\rin_spectrum.png', dpi=150)
    plt.show()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
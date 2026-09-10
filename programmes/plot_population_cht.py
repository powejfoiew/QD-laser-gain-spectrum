#!/usr/bin/env python3
"""
plot_population_and_cht.py

Creates a single figure that overlays:
  • The time‑resolved carrier‑density data from all *.cht files in the
    “Population‑Time” folder.
  • The simulation results stored in `population_history.csv`.

Data from PICWave and Python simulations are color-coded by state, with 
dashed lines for PICWave and solid lines for Python. Also calculates 
and prints a comparison table of final populations (last 5 ns average) 
and values at 12.5 ns.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Absolute base directory – adjust only if the project folder is moved.
# --------------------------------------------------------------------------- #
BASE_DIR = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Population-Time"


def read_cht(filepath):
    """Return a DataFrame with the two numeric columns in a .cht file."""
    start_line = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                start_line = i - 1
                break

    df = pd.read_csv(
        filepath,
        sep=r"[\s,]+",
        header=None,
        skiprows=start_line,
        usecols=[0, 1],
        names=["time_ns", "value"],
        engine="python",
        dtype=str,
    )
    df["time_ns"] = pd.to_numeric(df["time_ns"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(inplace=True)
    return df


def get_state_key(filename_or_col):
    """Extracts a normalized state key (e.g., 'GS', 'ES1', 'SCH') to sync colors."""
    text = filename_or_col.upper()
    if "GS" in text:
        return "GS"
    if "ES1" in text:
        return "ES1"
    if "ES2" in text:
        return "ES2"
    if "SCH" in text:
        return "SCH"
    if "W" in text:
        return "W"
    return text


def interpolate_at_time(time_arr, val_arr, target_t):
    """Helper to safely interpolate value at a specific time point."""
    if len(time_arr) == 0:
        return np.nan
    # Ensure sorted by time
    sort_idx = np.argsort(time_arr)
    t_sorted = time_arr[sort_idx]
    v_sorted = val_arr[sort_idx]
    if target_t < t_sorted[0] or target_t > t_sorted[-1]:
        return np.nan
    return np.interp(target_t, t_sorted, v_sorted)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay .cht carrier‑density data with population_history CSV."
    )
    parser.add_argument(
        "--out",
        default="combined_population_cht.png",
        help="Filename for the saved PNG.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------- #
    # 1. Prepare Plot & Color Map
    # ------------------------------------------------------------------- #
    plt.figure(figsize=(9, 6), dpi=300)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    states = ["GS", "ES1", "ES2", "SCH", "W"]
    color_map = {state: color_cycle[i % len(color_cycle)] for i, state in enumerate(states)}

    # Dictionaries to store extracted metrics for table printing
    metrics_data = {state: {"PICWave_12.5ns": np.nan, "PICWave_last5ns": np.nan,
                            "Python_12.5ns": np.nan, "Python_last5ns": np.nan} for state in states}

    # ------------------------------------------------------------------- #
    # 2. Load and plot .cht files (PICWave -> Dashed Lines)
    # ------------------------------------------------------------------- #
    cht_folder = BASE_DIR
    cht_files = [f for f in os.listdir(cht_folder) if f.lower().endswith(".cht")]

    for fn in cht_files:
        fp = os.path.join(cht_folder, fn)
        df = read_cht(fp)

        state_key = get_state_key(fn)
        color = color_map.get(state_key, None)

        # Scale data consistent with plotting
        t_vals = df["time_ns"].to_numpy()
        v_vals = df["value"].to_numpy() * 1e18

        plt.plot(
            t_vals,
            v_vals,
            linestyle="--",
            linewidth=1.8,
            color=color,
            label=f"{state_key} (PICWave)",
        )

        # Compute metrics for PICWave
        if state_key in metrics_data and len(t_vals) > 0:
            # Value at 12.5 ns
            metrics_data[state_key]["PICWave_12.5ns"] = interpolate_at_time(t_vals, v_vals, 12.5)
            
            # Average over the last 5 ns
            max_t = t_vals[-1]
            mask = t_vals >= (max_t - 5.0)
            if np.any(mask):
                metrics_data[state_key]["PICWave_last5ns"] = np.mean(v_vals[mask])

    # ------------------------------------------------------------------- #
    # 3. Load and plot Python CSV data (Python -> Solid Lines)
    # ------------------------------------------------------------------- #
    export_path = os.path.join(BASE_DIR, "population_history.csv")
    if not os.path.isfile(export_path):
        raise FileNotFoundError(f"CSV not found at {export_path}")

    pop_df = pd.read_csv(export_path)
    t_hist_ns = pop_df["t"].to_numpy() * 1e9

    py_columns = [
        ("n_e_gs", r"$N_e^{\mathrm{GS}}$", "GS"),
        ("n_e_es1", r"$N_e^{\mathrm{ES1}}$", "ES1"),
        ("n_e_es2", r"$N_e^{\mathrm{ES2}}$", "ES2"),
        ("n_e_sch", r"$N_e^{\mathrm{SCH}}$", "SCH"),
        ("n_e_w", r"$N_e^{\mathrm{W}}$", "W"),
    ]

    for col_name, latex_label, state_key in py_columns:
        if col_name in pop_df.columns:
            data = pop_df[col_name].to_numpy() / 40e-3
            color = color_map.get(state_key, None)

            plt.plot(
                t_hist_ns,
                data,
                linestyle="-",
                linewidth=1.8,
                color=color,
                label=f"{latex_label} (Python)",
            )

            # Compute metrics for Python
            if state_key in metrics_data and len(t_hist_ns) > 0:
                metrics_data[state_key]["Python_12.5ns"] = interpolate_at_time(t_hist_ns, data, 12.5)
                
                max_t = t_hist_ns[-1]
                mask = t_hist_ns >= (max_t - 5.0)
                if np.any(mask):
                    metrics_data[state_key]["Python_last5ns"] = np.mean(data[mask])

    # ------------------------------------------------------------------- #
    # 4. Print Comparison Table
    # ------------------------------------------------------------------- #
    print("\n" + "=" * 85)
    print(f"{'State':<8} | {'Dataset':<10} | {'Value at 12.5 ns':<18} | {'Avg (Last 5 ns)':<18}")
    print("-" * 85)
    for state in states:
        m = metrics_data[state]
        print(f"{state:<8} | {'PICWave':<10} | {m['PICWave_12.5ns']:<18.4e} | {m['PICWave_last5ns']:<18.4e}")
        print(f"{state:<8} | {'Python':<10} | {m['Python_12.5ns']:<18.4e} | {m['Python_last5ns']:<18.4e}")
        print("-" * 85)

    # ------------------------------------------------------------------- #
    # 5. Styling and Formatting
    # ------------------------------------------------------------------- #
    plt.xlabel("Time (ns)", fontsize=12)
    plt.ylabel("Carrier density (a.u.)", fontsize=12)
    plt.title("Population Dynamics: Python Simulation vs. PICWave", fontsize=13)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, frameon=True)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, args.out)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nCombined figure saved to '{out_path}'")


if __name__ == "__main__":
    main()
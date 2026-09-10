"""
Fits each QD level's Pauli-blocking occupation probability as a function of
a *neighbouring* level's population, since PICWave cannot make one
transition's rate depend on a different level's population directly (see
pauli_blocking/__init__.py and the thesis's Figure 4.1/5.2 discussion).

Produces, for each of the three inter-level transitions, a PICWave-ready
expression plus a plot, written to blocking_factors/<TRANSITION>/:

    blocking_factors/
        GS_ES1/     GS_vs_N_ES1.{png,json}   ES1_vs_N_GS.{png,json}
        ES1_ES2/    ES1_vs_N_ES2.{png,json}  ES2_vs_N_ES1.{png,json}
        ES2_WL/     ES2_vs_N_W.{png,json}

Run from the project root, after generate_gain_table.py has populated
gain_table/ (this script only reads that cache, it does not run the
rate-equation solver itself).
"""

import json
import os

import numpy as np

from pauli_blocking.curve_fitting import (
    apply_publication_style, fit_and_plot, fit_piecewise, get_picwave_expression,
)
from pauli_blocking.data import (
    NORMALIZATION_VOLUME, apply_hole_imbalance_correction, load_carrier_sweep,
)

OUTPUT_DIR = 'blocking_factors'


def fit_single_curve(x, y, title, out_dir, name):
    """Fit x/y with the single-curve model registry, save the plot and a
    JSON summary (model, R^2, fitted params, PICWave expression) to
    out_dir/name.{png,json}, and return the best-fit record."""
    os.makedirs(out_dir, exist_ok=True)
    ranked, _ = fit_and_plot(x, y, title=title,
                              save_path=os.path.join(out_dir, f"{name}.png"))
    best_name, best = ranked[0]
    record = dict(
        model=best_name, r2=best['r2'], params=best['params'],
        picwave_expr=get_picwave_expression(best_name, best['params']),
    )
    with open(os.path.join(out_dir, f"{name}.json"), 'w') as f:
        json.dump(record, f, indent=2, default=str)
    return record


def fit_and_save_piecewise(x, y, title, out_dir, name, Nb_override=None):
    """Fit x/y piecewise, save the plot and a JSON summary to
    out_dir/name.{png,json}, and return the fit result."""
    os.makedirs(out_dir, exist_ok=True)
    result = fit_piecewise(x, y, title=title, Nb_override=Nb_override,
                            save_path=os.path.join(out_dir, f"{name}.png"))
    record = {k: v for k, v in result.items() if k != 'fig'}
    with open(os.path.join(out_dir, f"{name}.json"), 'w') as f:
        json.dump(record, f, indent=2, default=str)
    return result


if __name__ == '__main__':
    apply_publication_style()

    sweep = load_carrier_sweep(output_dir='gain_table')
    rho_e_gs_transformed = apply_hole_imbalance_correction(
        sweep['current_array'], sweep['rho_e_gs_arr'], sweep['rho_h_gs_arr'])

    V = NORMALIZATION_VOLUME

    # --- ES2 <-> WL ------------------------------------------------------
    es2_wl_dir = os.path.join(OUTPUT_DIR, 'ES2_WL')
    x1 = sweep['N_e_w'] / V
    y1 = 1 - np.mean(sweep['rho_e_es2_arr'], axis=1)
    res1 = fit_single_curve(x1, y1, 'ES2 vs N_e_w', es2_wl_dir, 'ES2_vs_N_W')

    # --- ES1 <-> ES2 -------------------------------------------------------
    es1_es2_dir = os.path.join(OUTPUT_DIR, 'ES1_ES2')

    x2 = np.sum(sweep['N_e_es2'], axis=1) / V
    y2 = 1 - np.mean(sweep['rho_e_es1_arr'], axis=1)
    res2 = fit_single_curve(x2, y2, 'ES1 vs sum(N_e_es2)', es1_es2_dir, 'ES1_vs_N_ES2')

    x4 = np.sum(sweep['N_e_es1'], axis=1) / V
    y4 = 1 - np.mean(sweep['rho_e_es2_arr'], axis=1)
    res4 = fit_and_save_piecewise(x4, y4, 'ES2 vs sum(N_e_es1)', es1_es2_dir, 'ES2_vs_N_ES1')

    # --- GS <-> ES1 ----------------------------------------------------
    gs_es1_dir = os.path.join(OUTPUT_DIR, 'GS_ES1')

    x3 = np.sum(sweep['N_e_es1'], axis=1) / V
    y3 = 1 - np.mean(rho_e_gs_transformed, axis=1)
    res3 = fit_and_save_piecewise(x3, y3, 'GS vs sum(N_e_es1)', gs_es1_dir, 'GS_vs_N_ES1',
                                   Nb_override=2.85e17)

    x5 = np.sum(sweep['N_e_gs'], axis=1) / V
    y5 = 1 - np.mean(sweep['rho_e_es1_arr'], axis=1)
    res5 = fit_and_save_piecewise(x5, y5, 'ES1 vs sum(N_e_gs)', gs_es1_dir, 'ES1_vs_N_GS',
                                   Nb_override=2.0e17)

    # --- Summary -----------------------------------------------------------
    print("\n\n=== Summary (single-curve fits) ===")
    for label, res in [("ES2 vs N_e_w", res1), ("ES1 vs sum(N_e_es2)", res2)]:
        print(f"{label:<22} -> {res['model']:<17} R^2={res['r2']:.4f}")
        print(f"    PICWave = {res['picwave_expr']}")

    print("\n=== Summary (piecewise fits) ===")
    for label, res in [("GS vs sum(N_e_es1)", res3),
                        ("ES2 vs sum(N_e_es1)", res4),
                        ("ES1 vs sum(N_e_gs)", res5)]:
        print(f"{label:<22} -> seg1={res['seg1_name']} (R^2={res['seg1_r2']:.4f})"
              f"  seg2={res['seg2_name']} (R^2={res['seg2_r2']:.4f})"
              f"  whole-curve R^2={res['r2_combined']:.4f}")
        print(f"    N_b     = {res['Nb']:.6e}")
        print(f"    PICWave = {res['picwave_expr']}")

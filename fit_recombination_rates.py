"""
Fits the total Auger and spontaneous-emission recombination rates against
total electron carrier density, for use in PICWave's NRADRATE/RADRATE-style
material expressions.

For each rate, both a single-curve fit and a piecewise (knee-anchored) fit
are computed and compared — the piecewise fit is only worth using if it
meaningfully improves on the single-curve R². Produces, per rate, a plot
and a JSON summary for each fit variant plus the comparison, written to
recombination_rate_fits/<RATE>/:

    recombination_rate_fits/
        Auger/         single.{png,json}  piecewise.{png,json}
        Spontaneous/   single.{png,json}  piecewise.{png,json}

Run from the project root, after generate_gain_table.py has populated
gain_table/ (this script only reads that cache, it does not run the
rate-equation solver itself).
"""

import json
import os

from pauli_blocking.curve_fitting import apply_publication_style
from recombination_rates.curve_fitting_rise import fit_and_plot_rise, fit_piecewise_rise
from recombination_rates.data import load_recombination_sweep

OUTPUT_DIR = 'recombination_rate_fits'

# A piecewise fit is only reported as "worth using" if it improves R² over
# the single-curve fit by at least this much.
PIECEWISE_IMPROVEMENT_THRESHOLD = 1e-4


def fit_rate(x, y, title, out_dir):
    """Fit x/y with both a single-curve and a piecewise rise fit, save
    plots + JSON summaries to out_dir/{single,piecewise}.{png,json}, and
    return (single_record, piecewise_result)."""
    os.makedirs(out_dir, exist_ok=True)

    ranked, _ = fit_and_plot_rise(x, y, title=f"{title} (single fit)",
                                   save_path=os.path.join(out_dir, 'single.png'))
    best_name, best = ranked[0]
    single_record = dict(model=best_name, r2=best['r2'], params=best['params'])
    with open(os.path.join(out_dir, 'single.json'), 'w') as f:
        json.dump(single_record, f, indent=2, default=str)

    piecewise = fit_piecewise_rise(x, y, title=title,
                                    save_path=os.path.join(out_dir, 'piecewise.png'))
    piecewise_record = {k: v for k, v in piecewise.items() if k != 'fig'}
    piecewise_record['improves_on_single'] = (
        piecewise['r2_combined'] > best['r2'] + PIECEWISE_IMPROVEMENT_THRESHOLD
    )
    with open(os.path.join(out_dir, 'piecewise.json'), 'w') as f:
        json.dump(piecewise_record, f, indent=2, default=str)

    return single_record, piecewise


if __name__ == '__main__':
    apply_publication_style()

    sweep = load_recombination_sweep(output_dir='gain_table')
    N_tot_qdw = sweep['N_tot_qdw']

    auger_dir = os.path.join(OUTPUT_DIR, 'Auger')
    single_auger, piecewise_auger = fit_rate(
        N_tot_qdw, sweep['R_aug_tot'] + sweep['R_recomb'],
        'Auger rate vs N_tot_qdw', auger_dir)

    spontaneous_dir = os.path.join(OUTPUT_DIR, 'Spontaneous')
    single_spon, piecewise_spon = fit_rate(
        N_tot_qdw, sweep['R_spon_tot'],
        'Spontaneous rate vs N_tot_qdw', spontaneous_dir)

    print("\n\n=== Summary: single-curve fit vs. piecewise fit ===")
    for label, single, piecewise in [
        ('Auger rate', single_auger, piecewise_auger),
        ('Spontaneous rate', single_spon, piecewise_spon),
    ]:
        print(f"\n{label}:")
        print(f"  single fit : {single['model']:<16} R^2={single['r2']:.5f}")
        print(f"  piecewise  : seg1={piecewise['seg1_name']} (R^2={piecewise['seg1_r2']:.4f})"
              f"  seg2={piecewise['seg2_name']} (R^2={piecewise['seg2_r2']:.4f})"
              f"  whole-curve R^2={piecewise['r2_combined']:.5f}")
        print(f"  N_b        = {piecewise['Nb']:.6e}")
        print(f"  PICWave    = {piecewise['picwave_expr']}")
        if piecewise['r2_combined'] > single['r2'] + PIECEWISE_IMPROVEMENT_THRESHOLD:
            print("  -> piecewise fit noticeably improves on the single-curve fit.")
        else:
            print("  -> single-curve fit is about as good; piecewise may be unnecessary here.")

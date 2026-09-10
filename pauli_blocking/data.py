"""
Loads the cached carrier-population sweep from gain_table/*.npz and derives
the per-current occupation probabilities and populations needed for
Pauli-blocking-factor fitting.
"""

import os

import numpy as np

from qd_gain.gain_model import compute_occupation_probabilities
from qd_gain.parameters import build_params

# Ridge dimensions used to convert current density (A/m^2) to bias current
# (mA) for the diagnostic/plot x-axis. Matches the value used elsewhere in
# the original current-density-to-current conversions (distinct from
# parameters.py's W=2.2e-6 ridge width — this mismatch predates this
# refactor and is left as-is; see qd_gain/parameters.py for other examples
# of this kind of inconsistency in the source material).
DEVICE_WIDTH_M = 2.4e-6
DEVICE_LENGTH_M = 400e-6

# Unexplained normalization constant carried over from the original fitting
# notebook (all populations/densities in this package, and in
# recombination_rates/, are divided by it before fitting) -- its physical
# meaning (a reference volume/area?) was not documented in the source
# material.
NORMALIZATION_VOLUME = 40e-3


def load_carrier_sweep(output_dir='gain_table', params=None, m_h_w=0.45):
    """
    Load every cached gain_table/J_<current>.npz carrier state (skipping the
    zero-current entry) and compute, per current:
      - N_e_w:    wetting-layer electron population (scalar)
      - N_e_gs/N_e_es1/N_e_es2: per-QD-group electron populations, shape (N_groups,)
      - rho_e_gs_arr/rho_e_es1_arr/rho_e_es2_arr: per-QD-group electron
        occupation probabilities, shape (n_currents, N_groups)
      - rho_h_gs_arr: hole occupation probability for the GS hole sub-band
        (group-independent — the hole reservoir is lumped, not split across
        QD groups — so this is one value per current, shape (n_currents,))
      - current_array: bias current in mA, shape (n_currents,)

    Returns a dict of these arrays plus the sorted current densities J_trimmed.
    """
    if params is None:
        params = build_params()

    J_detected = []
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith('J_') and filename.endswith('.npz'):
                val_str = filename[2:-4]
                try:
                    val_float = float(val_str)
                    if val_float != 0.0:
                        J_detected.append(val_float)
                except ValueError:
                    pass

    J_trimmed = np.array(sorted(J_detected))
    if len(J_trimmed) == 0:
        raise FileNotFoundError(f"No valid 'J_*.npz' files found in the directory: '{output_dir}'")

    n = len(J_trimmed)
    N_groups = params['N']

    N_e_w = np.zeros(n)
    N_e_gs = np.zeros((n, N_groups))
    N_e_es1 = np.zeros((n, N_groups))
    N_e_es2 = np.zeros((n, N_groups))

    rho_e_gs_arr = np.zeros((n, N_groups))
    rho_e_es1_arr = np.zeros((n, N_groups))
    rho_e_es2_arr = np.zeros((n, N_groups))
    rho_h_gs_arr = np.zeros(n)

    for index, J_val in enumerate(J_trimmed):
        filepath = os.path.join(output_dir, f"J_{J_val}.npz")
        data = np.load(filepath)
        final_state = {
            'n_e_sch': float(data['n_e_sch']),
            'n_e_w':   float(data['n_e_w']),
            'n_e_im':  data['n_e_im'],          # shape (3, N_groups)
            'n_h_sch': float(data['n_h_sch']),
            'n_hq_qd': float(data['n_hq_qd']),
        }

        occ = compute_occupation_probabilities(final_state, params=params, m_h_w=m_h_w)

        N_e_w[index] = final_state['n_e_w']
        N_e_gs[index] = final_state['n_e_im'][0]
        N_e_es1[index] = final_state['n_e_im'][1]
        N_e_es2[index] = final_state['n_e_im'][2]

        rho_e_gs_arr[index] = occ['rho_e_im'][0]
        rho_e_es1_arr[index] = occ['rho_e_im'][1]
        rho_e_es2_arr[index] = occ['rho_e_im'][2]
        rho_h_gs_arr[index] = occ['rho_h_gs']

    current_array = J_trimmed * DEVICE_WIDTH_M * DEVICE_LENGTH_M * 1e3

    return {
        'J_trimmed': J_trimmed,
        'current_array': current_array,
        'N_e_w': N_e_w,
        'N_e_gs': N_e_gs,
        'N_e_es1': N_e_es1,
        'N_e_es2': N_e_es2,
        'rho_e_gs_arr': rho_e_gs_arr,
        'rho_e_es1_arr': rho_e_es1_arr,
        'rho_e_es2_arr': rho_e_es2_arr,
        'rho_h_gs_arr': rho_h_gs_arr,
    }


# Bias current (mA) at which the electron/hole-imbalance correction below
# switches on, and the scaling applied to the hole occupation term in it.
# Both are hand-picked to keep the GS blocking-factor curve consistent with
# quasi-charge-neutrality beyond this current; see the thesis discussion of
# the GS Pauli-blocking fit (Figure 4.1(a)) for why this correction exists.
CORRECTION_CURRENT_MA = 4.641
HOLE_SCALE_FACTOR = 0.7


def apply_hole_imbalance_correction(current_array, rho_e_gs_arr, rho_h_gs_arr):
    """
    Beyond CORRECTION_CURRENT_MA, replace the raw GS electron occupation
    probability with a value consistent with quasi-charge-neutrality
    (1 + target - HOLE_SCALE_FACTOR*rho_h_gs), where `target` is pinned to
    the raw electron/hole occupation values at CORRECTION_CURRENT_MA. Below
    that current, the raw rho_e_gs_arr values are kept unchanged.

    Returns rho_e_gs_transformed, the same shape as rho_e_gs_arr.
    """
    idx = np.abs(current_array - CORRECTION_CURRENT_MA).argmin()

    rho_e_gs_thresh = np.mean(rho_e_gs_arr, axis=1)[idx]
    rho_h_gs_thresh = HOLE_SCALE_FACTOR * rho_h_gs_arr[idx]
    target = rho_e_gs_thresh + rho_h_gs_thresh - 1

    rho_e_gs_transformed = np.copy(rho_e_gs_arr)
    rho_e_gs_transformed[idx:] = (
        1 + target - HOLE_SCALE_FACTOR * rho_h_gs_arr[idx:, None]
    )
    return rho_e_gs_transformed

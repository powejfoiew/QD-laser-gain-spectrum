"""
Loads the cached carrier-population sweep from gain_table/*.npz and derives
the total electron density and total Auger/spontaneous recombination rates
needed for recombination-rate fitting.
"""

import os

import numpy as np

from qd_gain.gain_model import compute_recombination_rates
from qd_gain.parameters import build_params
from pauli_blocking.data import DEVICE_LENGTH_M, DEVICE_WIDTH_M, NORMALIZATION_VOLUME

# Wetting-layer spontaneous-recombination lifetime used for R_recomb below.
# Matches params['tau_r_e_W'] exactly; kept as an explicit local constant
# since R_recomb is a diagnostic add-on to the Auger rate, not itself part
# of the rate-equation solver.
WL_RECOMBINATION_LIFETIME_S = 100e-12


def load_recombination_sweep(output_dir='gain_table', params=None, m_h_w=0.45):
    """
    Load every cached gain_table/J_<current>.npz carrier state (skipping the
    zero-current entry) and compute, per current:
      - N_tot_qdw: total electron density (QD levels + wetting layer),
        normalized by NORMALIZATION_VOLUME, shape (n_currents,)
      - R_aug_tot: total Auger recombination rate (summed over QD levels
        and groups), shape (n_currents,)
      - R_recomb:  wetting-layer spontaneous recombination rate, shape (n_currents,)
      - R_spon_tot: total QD spontaneous-emission rate (summed over levels
        and groups), shape (n_currents,)
      - current_array: bias current in mA, shape (n_currents,)
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

    N_tot_qdw = np.zeros(n)
    R_aug_tot = np.zeros(n)
    R_recomb = np.zeros(n)
    R_spon_tot = np.zeros(n)

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

        rates = compute_recombination_rates(final_state, params=params, m_h_w=m_h_w)

        N_tot_qdw[index] = (
            np.sum(final_state['n_e_im']) + final_state['n_e_w']
        ) / NORMALIZATION_VOLUME
        R_aug_tot[index] = rates['R_aug_im'].sum()
        R_recomb[index] = final_state['n_e_w'] / WL_RECOMBINATION_LIFETIME_S
        R_spon_tot[index] = rates['R_sp_im'].sum()

    current_array = J_trimmed * DEVICE_WIDTH_M * DEVICE_LENGTH_M * 1e3

    return {
        'J_trimmed': J_trimmed,
        'current_array': current_array,
        'N_tot_qdw': N_tot_qdw,
        'R_aug_tot': R_aug_tot,
        'R_recomb': R_recomb,
        'R_spon_tot': R_spon_tot,
    }

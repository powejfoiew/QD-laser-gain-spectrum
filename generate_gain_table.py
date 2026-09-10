"""
End-to-end reproduction of the qd_gain/gain_table/ cache:

1. Build a non-uniform current-density sweep (`generate_custom_samples`),
   concentrated near the near-threshold region.
2. Run the carrier rate-equation simulation (`run_qd_simulation`) to
   convergence at each current and cache the resulting carrier state to
   gain_table/J_<J>.npz (skipped if a cache file already matches).
3. Reload the cache and compute the material gain spectrum at each current
   (`compute_gain_spectrum`), then plot gain vs. wavelength across the
   sweep.

Run from the project root so the cache lands in ./gain_table/, matching the
included results folder. A full 500-point sweep takes on the order of tens
of minutes; re-running is fast afterwards since already-cached currents are
skipped.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from qd_gain.carrier_dynamics import run_qd_simulation
from qd_gain.gain_model import compute_gain_spectrum
from qd_gain.parameters import build_params

OUTPUT_DIR = 'gain_table'
LEN_OMEGA  = 100   # wavelength/frequency grid points per gain spectrum

# Simulation settings used to generate every cached state.
SIM_KWARGS = dict(
    t_end=30e-9,
    tstep=60e-15,
    threshold=0.05,
    m_h_w=0.4,
    m_e_w=0.026,
    delta_E_sch_w=75,
    delta_E_h_sch_w=140,
)

INITIAL_VALUES = {
    'n_e_sch': 0.0,
    'n_e_w':   0.0,
    'n_e_im':  np.zeros((3, 51)),
    'n_h_sch': 0.0,
    'n_hq_qd': 1e6,   # initial hole population in QD reservoir
}


def generate_custom_samples(total_points=500, high_density_ratio=0.6):
    """
    Generate a sample array for J from 0 to 18E8 with a higher density
    of points concentrated between 4.5E6 and 18E6 (the near-threshold
    region, where the gain-vs-current curve changes fastest).

    Parameters:
    - total_points (int): Total number of samples in the final array.
    - high_density_ratio (float): Fraction of total points allocated to the target region.
    """
    n_high = int(total_points * high_density_ratio)
    n_remaining = total_points - n_high
    n_low = int(n_remaining * 0.25)   # 0 to 4.5E6 is a narrow outer band
    n_upper = total_points - n_high - n_low

    seg1 = np.linspace(0, 4.5e6, n_low, endpoint=False)
    seg2 = np.linspace(4.5e6, 18e6, n_high, endpoint=False)
    seg3 = np.linspace(18e6, 18e8, n_upper)

    return np.concatenate([seg1, seg2, seg3])


def build_gain_table(J_arr, output_dir=OUTPUT_DIR, params=None):
    """Run the rate-equation solver for every current in J_arr and cache
    the converged carrier state to output_dir/J_<J>.npz. Currents that
    already have a cache file are skipped, so re-running only computes
    whatever is missing (e.g. after extending J_arr)."""
    os.makedirs(output_dir, exist_ok=True)

    for J_val in J_arr:
        filepath = os.path.join(output_dir, f"J_{J_val}.npz")
        if os.path.exists(filepath):
            print(f'J= {J_val}  (cached, skipping)')
            continue

        print(f'J= {J_val}')
        t_final, final_state, last_step = run_qd_simulation(
            J=J_val,
            initial_values=INITIAL_VALUES,
            params=params,
            verbose=True,
            **SIM_KWARGS,
        )

        np.savez_compressed(
            filepath,
            t_final=np.array([t_final]),
            last_step=np.array([last_step]),
            **final_state,      # n_e_sch, n_e_w, n_e_im, n_h_sch, n_hq_qd
        )
        print(f"  Saved -> {filepath}")


def load_gain_spectra(J_arr, output_dir=OUTPUT_DIR, len_omega=LEN_OMEGA, params=None):
    """Load every cached carrier state and compute its gain spectrum."""
    gain_array  = np.zeros((len(J_arr), len_omega))
    N_sweep_arr = np.zeros(len(J_arr))
    wavelength  = None

    for index, J_val in enumerate(J_arr):
        filepath = os.path.join(output_dir, f"J_{J_val}.npz")
        data = np.load(filepath)
        final_state = {
            'n_e_sch': float(data['n_e_sch']),
            'n_e_w':   float(data['n_e_w']),
            'n_e_im':  data['n_e_im'],          # shape (3, 51)
            'n_h_sch': float(data['n_h_sch']),
            'n_hq_qd': float(data['n_hq_qd']),
        }

        result_output = compute_gain_spectrum(final_state, len_omega=len_omega, params=params)

        gain_array[index]  = result_output['gain_cm_minus_1']
        N_sweep_arr[index] = result_output['total_electrons']

        if wavelength is None:
            wavelength = result_output['wavelength_nm']

    return gain_array, N_sweep_arr, wavelength


if __name__ == '__main__':
    params = build_params()

    J_arr = generate_custom_samples(total_points=500, high_density_ratio=0.6)
    build_gain_table(J_arr, params=params)

    # J=0 has no converged carrier state to speak of; skip it as the
    # original notebook did (J_trimmed = J_arr[1:]).
    J_trimmed = J_arr[1:]
    gain_array, N_sweep_arr, wavelength = load_gain_spectra(J_trimmed, params=params)

    # Empirically fitted dg/dN scaling used to convert this model's raw
    # gain output into material gain (cm^-1); see thesis section 4.1 —
    # not derived in closed form from the parameters above.
    CONVERSION_FACTOR = 4019376.8605
    GAMMA_XY_FITTED   = 0.08303645356032915

    for index in range(len(J_trimmed)):
        plt.plot(wavelength, CONVERSION_FACTOR * gain_array[index],
                  label=f'{N_sweep_arr[index]:.4f}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Modal gain (cm$^{-1}$)')
    plt.show()

    material_gain = CONVERSION_FACTOR * gain_array / GAMMA_XY_FITTED

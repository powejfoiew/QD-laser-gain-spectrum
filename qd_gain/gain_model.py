"""
Optical gain and refractive-index-change calculations built on a converged
carrier state from `qd_gain.carrier_dynamics.run_qd_simulation`.

Each QD transition (GS, ES1, and the lumped ES2 approximation) is modelled
as a Lorentzian centred on its per-group transition frequency, homogeneously
broadened by `Gamma` and weighted by the inhomogeneous broadening
distribution `G_i`; gain is the imaginary part, refractive-index change the
real part, of the resulting complex susceptibility contribution.
"""

import numpy as np

from .parameters import build_params
from .carrier_dynamics import fermi, solve_hole_quasi_fermi_level


def gain(omega, rho_e_im, rho_hmat, params):
    """Complex per-frequency gain contribution summed over GS and ES1."""
    G_i = params['G_i_by_level']
    D_m = params['D_m']
    A_m = params['A_m']
    omegahbar = params['omegahbar']
    return (params['Gamma_xy']/params['h_w']*params['N_D']
            * np.sum(G_i[:2]*D_m[:2]*A_m*1j/params['hbar']/np.pi
                     / (params['Gamma']+1j*(omega-omegahbar))
                     * (rho_e_im[:2]+rho_hmat[:2]-1)))


def gain_index(omega, rho_e_im, rho_hmat, params, index):
    """
    Complex gain contribution from a single transition level: index 0 = GS,
    1 = ES1 (using their own dipole matrix element and transition energy),
    2 = ES2 (approximated using the ES1 dipole matrix element/degeneracy
    weighting, per the lumped-ES2 treatment used when generating the table).
    """
    G_i = params['G_i_by_level']
    D_m = params['D_m']
    A_m = params['A_m']
    omegahbar = params['omegahbar']

    if index == 0 or index == 1:
        return (params['Gamma_xy']/params['h_w']*params['N_D']
                * np.sum(G_i[index]*D_m[index]*A_m[index]*1j/params['hbar']/np.pi
                         / (params['Gamma']+1j*(omega-omegahbar[index]))
                         * (rho_e_im[index]+rho_hmat[index]-1)))
    if index == 2:
        return (params['Gamma_xy']/params['h_w']*params['N_D']
                * np.sum(G_i[0]*6*A_m[1]*1j/params['hbar']/np.pi
                         / (params['Gamma']+1j*(params['E_ES2_i']/params['hbar']))
                         * (rho_e_im[1]+rho_hmat[1]-1)))


def compute_occupation_probabilities(final_state, params=None, m_h_w=0.45):
    """
    Compute the per-QD-group electron occupation probabilities (GS/ES1/ES2)
    and the (group-independent, lumped-reservoir) hole occupation
    probabilities for a converged carrier state.

    This is the same rho_e/rho_h calculation performed inside
    compute_gain_spectrum, factored out here since some post-processing
    (e.g. Pauli-blocking-factor fitting) only needs the occupation
    probabilities, not a full gain spectrum swept over wavelength.
    """
    if params is None:
        params = build_params()
    params = dict(params)

    G_i = params['G_i_by_level']
    D_m = params['D_m']
    E_h_m = params['E_h_m']

    n_e_im_final = final_state['n_e_im']       # shape (n_m, N_groups)
    n_hq_qd_final = final_state['n_hq_qd']

    rho_e_im = n_e_im_final / (params['N_l'] * params['N_D'] * G_i * D_m)

    EF_h_final = solve_hole_quasi_fermi_level(
        n_hq_qd_final, params, m_h_w, kBT_window=15, ev_to_j=params['q'])
    rho_h_m = fermi(E_h_m, EF_h_final, params['kBT'])

    return {
        'rho_e_im':  rho_e_im,      # shape (3, N_groups): GS/ES1/ES2 per QD group
        'rho_h_gs':  rho_h_m[0],
        'rho_h_es1': rho_h_m[1],
        'rho_h_es2': rho_h_m[2],
    }


def compute_recombination_rates(final_state, params=None, m_h_w=0.45):
    """
    Compute the per-QD-group Auger and spontaneous-emission recombination
    rates (GS/ES1/ES2) for a converged carrier state, using the same
    occupation probabilities as compute_occupation_probabilities().
    """
    if params is None:
        params = build_params()
    params = dict(params)

    n_e_im_final = final_state['n_e_im']       # shape (n_m, N_groups)
    occ = compute_occupation_probabilities(final_state, params=params, m_h_w=m_h_w)
    rho_e_im = occ['rho_e_im']

    rho_h_mat = np.zeros((params['n_m'], params['N']))
    rho_h_mat[0, :] = occ['rho_h_gs']
    rho_h_mat[1, :] = occ['rho_h_es1']
    rho_h_mat[2, :] = occ['rho_h_es2']

    R_aug_im = n_e_im_final * rho_e_im * rho_h_mat / params['auger_lifetimes']
    R_sp_im = n_e_im_final * rho_h_mat / params['sp_lifetimes']

    return {
        'R_aug_im': R_aug_im,   # shape (3, N_groups): GS/ES1/ES2 per QD group
        'R_sp_im':  R_sp_im,
    }


def compute_gain_spectrum(final_state, len_omega=100, params=None,
                           wavelength_range_nm=None, m_h_w=0.45):
    """
    Compute the material gain spectrum (and total carrier counts) for a
    converged carrier state, over a wavelength/frequency grid.

    (Renamed from the original `plot_gain_and_carriers` — despite the name,
    it never plotted anything; it only computes and returns a results dict.)
    """
    if params is None:
        params = build_params()
    params = dict(params)

    G_i = params['G_i_by_level']
    D_m = params['D_m']
    E_h_m  = params['E_h_m']

    if wavelength_range_nm is not None:
        wl_min_m = wavelength_range_nm[0] * 1e-9
        wl_max_m = wavelength_range_nm[1] * 1e-9
        omega_max = (2 * np.pi * params['c0']) / wl_min_m
        omega_min = (2 * np.pi * params['c0']) / wl_max_m
    else:
        omega_min = 1.35e15
        omega_max = 1.65e15

    omega = np.linspace(omega_min, omega_max, len_omega)

    n_e_im_final = final_state['n_e_im']       # shape (n_m, N_groups)
    n_hq_qd_final = final_state['n_hq_qd']

    rho_e_im_final = n_e_im_final / (params['N_l'] * params['N_D'] * G_i * D_m)
    rho_e_gs_avg   = np.sum(G_i[0, :] * rho_e_im_final[0, :])
    rho_e_es1_avg  = np.sum(G_i[1, :] * rho_e_im_final[1, :])

    try:
        EF_h_final = solve_hole_quasi_fermi_level(
            n_hq_qd_final, params, m_h_w, kBT_window=15, ev_to_j=params['q'])
    except ValueError as e:
        print(f"Root finding failed! n_hq_qd = {n_hq_qd_final:.4e}")
        raise e

    rho_h_m   = fermi(E_h_m, EF_h_final, params['kBT'])
    rho_h_mat = np.zeros((2, params['N']))
    rho_h_mat[0, :] = rho_h_m[0]
    rho_h_mat[1, :] = rho_h_m[1]

    gain_spectrum = np.array([np.imag(gain(w, rho_e_im_final, rho_h_mat, params))
                               for w in omega])
    gain_cm       = gain_spectrum / 100
    wavelength_nm = (2 * np.pi * params['c0'] / omega) * 1e9

    print(f"  ρ_e(GS)={rho_e_gs_avg:.4f}  ρ_e(ES1)={rho_e_es1_avg:.4f}")
    print(f"  ρ_h(GS)={rho_h_m[0]:.4f}  ρ_h(ES1)={rho_h_m[1]:.4f}")
    print(f"  Gain factor GS:  {rho_e_gs_avg + rho_h_m[0] - 1:.4f}")
    print(f"  Gain factor ES1: {rho_e_es1_avg + rho_h_m[1] - 1:.4f}")

    # Carrier totals from final state only
    n_e_qd = (n_e_im_final[0, :].sum()
              + n_e_im_final[1, :].sum()
              + n_e_im_final[2, :].sum())

    total_electrons    = final_state['n_e_sch'] + final_state['n_e_w'] + n_e_qd
    total_holes        = final_state['n_h_sch'] + final_state['n_hq_qd']
    carrier_difference = total_electrons - total_holes

    return {
        'wavelength_nm':      wavelength_nm,
        'gain_cm_minus_1':    gain_cm,
        'total_electrons':    total_electrons,
        'total_holes':        total_holes,
        'carrier_difference': carrier_difference,
    }


def refractive_index_change(n_e_sch, n_e_w, n_e_im, n_h_sch, n_h_qd,
                             params=None,
                             m_e=None, m_h=None,
                             gamma_xy=None, gamma_xy_sch=None,
                             h_w=None, h_sch=None, dot=True,
                             S_re_e=None, S_re_h=None):
    """
    Drude-model refractive-index perturbation from the free-carrier (WL/SCH)
    and confined-QD populations, with the Uskov correction (`S_re_e`,
    `S_re_h`) scaling how strongly the confined-state populations
    contribute relative to a free 2-D plasma.

    m_e/m_h/gamma_xy/gamma_xy_sch/h_w/h_sch/S_re_e/S_re_h default to
    parameters.build_params()'s m_e_drude, m_h_drude, Gamma_xy_index,
    Gamma_xy_SCH, h_w, h_SCH, S_re_e_default, S_re_h_default.
    """
    if params is None:
        params = build_params()

    if m_e is None:
        m_e = params['m_e_drude']
    if m_h is None:
        m_h = params['m_h_drude']
    if gamma_xy is None:
        gamma_xy = params['Gamma_xy_index']
    if gamma_xy_sch is None:
        gamma_xy_sch = params['Gamma_xy_SCH']
    if h_w is None:
        h_w = params['h_w']
    if h_sch is None:
        h_sch = params['h_SCH']
    if S_re_e is None:
        S_re_e = params['S_re_e_default']
    if S_re_h is None:
        S_re_h = params['S_re_h_default']

    m0 = params['m0']
    epsilon_0 = params['eps0']
    omega_0 = params['omega_0']

    # Leading minus for the Drude coefficient
    coeff = -params['q']**2 / (2 * params['eta'] * epsilon_0 * omega_0**2)

    if dot:
        # THE USKOV CORRECTION:
        # n_e_w (Wetting layer) remains a free 2D plasma (Drude factor = 1)
        # n_e_im (Confined QD states) is scaled by S_re_e
        effective_electrons = n_e_w + (S_re_e * np.sum(n_e_im))

        # Note: the original hardcoded masses (0.026, 0.4) were left intact
        # here rather than the m_e/m_h keyword arguments above, matching the
        # thesis code as run; replace with m_e_kg/m_h_kg below if you want
        # these to actually follow the function's mass arguments.
        m_e_kg = m_e * m0
        m_h_kg = m_h * m0
        electron_term = (gamma_xy / (0.064 * m0 * h_w * params['N_l'])) * effective_electrons

        # In the TDTW paper, n_h_qd includes BOTH Wetting Layer and confined holes.
        # If you can't separate them in your solver, you have to apply an effective S_re_h to the whole lumped term.
        hole_term = (gamma_xy / (0.5 * m0 * h_w * params['N_l'])) * (S_re_h * n_h_qd)

        electron_term_sch = gamma_xy_sch * n_e_sch / (0.064 * m0 * h_sch)
        hole_term_sch = gamma_xy_sch * n_h_sch / (0.5 * m0 * h_sch)

    return coeff * (electron_term + hole_term + electron_term_sch + hole_term_sch)

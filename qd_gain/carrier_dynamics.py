"""
Electron/hole carrier-population rate equations for the QD-DFB model
(Gioannini & Rossetti 2011). Implements the SCH -> WL -> ES2 -> ES1 -> GS
electron cascade and a lumped hole reservoir, integrated with a first-order
Euler scheme until the relative step-to-step change in every population
falls below `threshold`. Stimulated emission is deliberately not included:
the resulting gain-vs-carrier-density table is the *unclamped* material
gain, which is what PICWave's gain-table importer expects.
"""

import numpy as np
from scipy.optimize import root_scalar

from .parameters import build_params


def fun(EF, N_h_qd, params, m_h_w):
    """Charge-neutrality equation for holes (energies in Joules)."""
    kBT = params['kBT']
    wl_term = (m_h_w * params['m0'] * params['N_l'] * kBT
               / (np.pi * params['hbar']**2)
               ) * np.logaddexp(0.0, (params['E_h_WL'] - EF) / kBT)
    exponent      = (EF - params['E_h_m']) / kBT
    confined_term = np.sum(params['N_l'] * params['N_D'] * params['D_m_h']
                            / (1.0 + np.exp(exponent)))
    return confined_term + wl_term - N_h_qd


def fun_eV(EF_eV, N_h_qd, params, m_h_w):
    """Wrapper for fun() that accepts EF in electronvolts."""
    return fun(EF_eV * params['q'], N_h_qd, params, m_h_w)


def fermi(E, EF, kBT):
    """Fermi-Dirac occupation probability."""
    return 1.0 / (1.0 + np.exp((EF - E) / kBT))


def solve_hole_quasi_fermi_level(N_h_qd, params, m_h_w, kBT_window=100,
                                  ev_to_j=None):
    """
    Solve for the hole quasi-Fermi level (Joules) satisfying charge
    neutrality for a given confined hole population `N_h_qd`.

    `kBT_window` sets how many kBT the root-finding bracket extends above
    and below the hole confined-state range; `ev_to_j` is the eV->J
    conversion used for the bracket and returned root, defaulting to
    params['q'] (the exact elementary charge).
    """
    if ev_to_j is None:
        ev_to_j = params['q']
    kBT = params['kBT']
    EF_lo_eV = (params['E_h_m'][0] - kBT_window * kBT) / ev_to_j
    EF_hi_eV = (params['E_h_WL']   + kBT_window * kBT) / ev_to_j
    sol = root_scalar(fun_eV, args=(N_h_qd, params, m_h_w),
                       bracket=[EF_lo_eV, EF_hi_eV], method='brentq')
    return sol.root * ev_to_j


def run_qd_simulation(t_end, initial_values, tstep, J=1E6, verbose=True,
                       threshold=0.01, params=None, delta_E_sch_w=None,
                       delta_E_h_sch_w=None, m_e_sch=None, m_e_w=None,
                       m_h_sch=None, m_h_w=None):
    """
    Integrate the electron/hole carrier-population rate equations at fixed
    injection current density J (A/m^2) until convergence.

    Returns (t_final, final_state, last_step), where final_state holds the
    converged n_e_sch, n_e_w, n_e_im (shape (3, N_groups)), n_h_sch, n_hq_qd.

    delta_E_sch_w/delta_E_h_sch_w/m_e_sch/m_e_w/m_h_sch/m_h_w default to
    the values in parameters.build_params() (delta_E_sch_w, delta_E_h_sch_w,
    m_e_sch_default, m_e_w_default, m_h_sch_default, m_h_w_default).
    """
    if params is None:
        params = build_params()
    params = dict(params)

    if delta_E_sch_w is None:
        delta_E_sch_w = params['delta_E_sch_w']
    if delta_E_h_sch_w is None:
        delta_E_h_sch_w = params['delta_E_h_sch_w']
    if m_e_sch is None:
        m_e_sch = params['m_e_sch_default']
    if m_e_w is None:
        m_e_w = params['m_e_w_default']
    if m_h_sch is None:
        m_h_sch = params['m_h_sch_default']
    if m_h_w is None:
        m_h_w = params['m_h_w_default']

    rt_ev = params['rt_ev']
    delta_e_w_es2   = 286 - delta_E_sch_w - delta_E_h_sch_w
    delta_e_es2_es1 = params['delta_e_es2_es1']
    delta_e_es1_gs  = params['delta_e_es1_gs']

    G_i = params['G_i_by_level']
    D_m = params['D_m']
    auger_lifetimes = params['auger_lifetimes']
    sp_lifetimes    = params['sp_lifetimes']
    E_h_m  = params['E_h_m']
    E_h_WL = params['E_h_WL']

    m0 = params['m0']
    h_planck = params['h_planck']
    kBT = params['kBT']
    DOS_sch   = 2*((2*np.pi*m_e_sch*m0*kBT)/h_planck**2)**(3/2)
    DOS_w     = ((m_e_w*m0*kBT)/(np.pi*params['hbar']**2))
    DOS_sch_h = 2*((2*np.pi*m_h_sch*m0*kBT)/h_planck**2)**(3/2)
    DOS_w_h   = ((m_h_w*m0*kBT)/(np.pi*params['hbar']**2))

    tau_e_e_W   = DOS_w*params['N_l']/DOS_sch/params['h_SCH']*np.exp(delta_E_sch_w/rt_ev)*params['tau_c_e_W']
    tau_e_h_qd  = 1/params['h_SCH']*DOS_w_h*params['N_l']/DOS_sch_h*np.exp(delta_E_h_sch_w/rt_ev)*params['tau_c_h_W']
    tau_e_e_ES1 = 4/6*np.exp(delta_e_es2_es1/rt_ev)*params['tau_c_e']['ES1']
    tau_e_e_ES2 = 6*params['N_D']/DOS_w*np.exp(delta_e_w_es2/rt_ev)*params['tau_c_e']['ES2']
    tau_e_e_GS  = 2/4*np.exp(delta_e_es1_gs/rt_ev)*params['tau_c_e']['GS']

    # Unpack initial values
    n_e_sch = float(initial_values['n_e_sch'])
    n_e_w   = float(initial_values['n_e_w'])
    n_e_im  = initial_values['n_e_im'].copy().astype(float)
    n_h_sch = float(initial_values['n_h_sch'])
    n_hq_qd = float(initial_values['n_hq_qd'])

    n_steps   = int(np.round(t_end / tstep))
    milestone = max(1, n_steps // 10)

    for step in range(n_steps):
        if verbose and step % milestone == 0:
            pct = 100 * step / n_steps
            print(f"  {pct:5.1f}%  t = {step * tstep * 1e9:.3f} ns")

        rho_e_im = n_e_im / (params['N_l'] * params['N_D'] * G_i * D_m)

        root_J = solve_hole_quasi_fermi_level(n_hq_qd, params, m_h_w)

        rho_h_m   = fermi(E_h_m, root_J, params['kBT'])
        rho_h_mat = np.zeros((n_e_im.shape[0], n_e_im.shape[1]))
        rho_h_mat[0, :] = rho_h_m[0]
        rho_h_mat[1, :] = rho_h_m[1]
        rho_h_mat[2, :] = rho_h_m[2]

        R_aug_im = n_e_im * rho_e_im * rho_h_mat / auger_lifetimes
        R_sp_im  = n_e_im * rho_h_mat / sp_lifetimes

        dn_e_sch = (params['eta_i']*J/params['q']
                    - n_e_sch/params['tau_c_e_W']
                    + n_e_w/tau_e_e_W)

        dn_e_w = (n_e_sch/params['tau_c_e_W']
                  - n_e_w/tau_e_e_W
                  - n_e_w/params['tau_r_e_W']
                  - np.sum(G_i[0,:]/params['tau_c_e']['ES2']*n_e_w*(1-rho_e_im[2,:]))
                  + np.sum(n_e_im[2,:]/tau_e_e_ES2))

        dn_i_e_es2 = (G_i[0,:]/params['tau_c_e']['ES2']*n_e_w*(1-rho_e_im[2,:])
                      - n_e_im[2,:]/tau_e_e_ES2
                      - R_sp_im[2,:] - R_aug_im[2,:]
                      - n_e_im[2,:]/params['tau_c_e']['ES1']*(1-rho_e_im[1,:])
                      + n_e_im[1,:]/tau_e_e_ES1*(1-rho_e_im[2,:]))

        dn_i_e_es1 = (n_e_im[2,:]/params['tau_c_e']['ES1']*(1-rho_e_im[1,:])
                      - n_e_im[1,:]/tau_e_e_ES1*(1-rho_e_im[2,:])
                      - R_aug_im[1,:]
                      - n_e_im[1,:]/params['tau_c_e']['GS']*(1-rho_e_im[0,:])
                      + n_e_im[0,:]/tau_e_e_GS*(1-rho_e_im[1,:])
                      - R_sp_im[1,:])

        dn_i_e_gs = (n_e_im[1,:]/params['tau_c_e']['GS']*(1-rho_e_im[0,:])
                     - n_e_im[0,:]/tau_e_e_GS*(1-rho_e_im[1,:])
                     - R_aug_im[0,:] - R_sp_im[0,:])

        dn_h_sch = (params['eta_i']*J/params['q']
                    - n_h_sch/params['tau_c_h_W']
                    + n_hq_qd/tau_e_h_qd)

        dn_h_qd  = (n_h_sch/params['tau_c_h_W']
                    - n_hq_qd/tau_e_h_qd
                    - np.sum(R_aug_im) - np.sum(R_sp_im)
                    - n_e_w / params['tau_r_e_W'])

        rel_changes = [
            abs(dn_e_sch) / (n_e_sch + 1e-20),
            abs(dn_e_w)   / (n_e_w   + 1e-20),
            abs(dn_h_sch) / (n_h_sch + 1e-20),
            abs(dn_h_qd)  / (n_hq_qd + 1e-20),
            np.max(abs(dn_i_e_gs)  / (n_e_im[0, :] + 1e-20)),
            np.max(abs(dn_i_e_es1) / (n_e_im[1, :] + 1e-20)),
            np.max(abs(dn_i_e_es2) / (n_e_im[2, :] + 1e-20)),
        ]

        n_e_sch      += dn_e_sch    * tstep
        n_e_w        += dn_e_w      * tstep
        n_e_im[0, :] += dn_i_e_gs  * tstep
        n_e_im[1, :] += dn_i_e_es1 * tstep
        n_e_im[2, :] += dn_i_e_es2 * tstep
        n_h_sch      += dn_h_sch    * tstep
        n_hq_qd      += dn_h_qd    * tstep

        if max(rel_changes) < threshold:
            if verbose:
                print(f"  Converged at step {step}  (t = {step * tstep * 1e9:.3f} ns)")
            break

    if verbose:
        print("  Done.")

    # Return only the final scalar/array state — no history
    final_state = {
        'n_e_sch': n_e_sch,
        'n_e_w':   n_e_w,
        'n_e_im':  n_e_im,        # shape (n_m, N_groups)
        'n_h_sch': n_h_sch,
        'n_hq_qd': n_hq_qd,
    }
    t_final = step * tstep
    return t_final, final_state, step

# %%
# ============================================================
# Cell 0 — Parameters
# ============================================================
"""
Parameters from Table I of:
Gioannini & Rossetti, "Time-Domain Traveling Wave Model of Quantum Dot DFB Lasers"
IEEE J. Sel. Top. Quantum Electron., Vol. 17, No. 5, 2011

All values are in SI units unless noted.
"""

import numpy as np
import os

# ── Physical constants ────────────────────────────────────────────────────────
q        = 1.602176634e-19   # elementary charge (C)
hbar     = 1.054571817e-34   # reduced Planck constant (J·s)
kB       = 1.380649e-23      # Boltzmann constant (J/K)
c0       = 2.997924580e8     # speed of light in vacuum (m/s)
eps0     = 8.854187817e-12   # vacuum permittivity (F/m)
m0       = 9.1093837015e-31  # free electron mass (kg)
T        = 300.0             # room temperature (K)

# ── Material parameters ───────────────────────────────────────────────────────
h_w      = 5e-9              # QD layer height (m)
eta      = 3.3445            # effective refractive index (dimensionless)
N_l      = 8                 # number of QD layers
N_D      = 5.9e14            # QD surface density (m⁻²)  [5.9×10¹⁰ cm⁻² → ×10⁴]
N_groups = 51                # number of QD groups for inhomogeneous broadening

# State degeneracies: D_m for m = ES2, ES1, GS
D = {
    'ES2': 6,
    'ES1': 4,
    'GS':  2,
}

hbar_Gamma = 7e-3 * q       # homogeneous linewidth (J)  [7 meV → ×q]
Gamma      = hbar_Gamma / hbar  # dephasing rate (rad/s)

Delta_E    = 38e-3 * q      # FWHM of inhomogeneous broadening (J)  [38 meV]

# Dipole matrix elements A_m  (cm³·eV → m³·J)
# 1 cm³·eV = 1e-6 m³ × 1.602e-19 J  →  multiply by 1e-6 * q
A = {
    'ES1': 1.26e-20 * 1e-6 * q,   # m³·J
    'GS':  2.03e-20 * 1e-6 * q,   # m³·J
}

# SCH diffusion/transport times (s)
tau_c_e_W  = 1.2e-12        # electron transport time across SCH (s)
tau_c_h_W  = 23.2e-12       # hole transport time across SCH (s)

tau_e_e_W  = 0.0492e-12

# Electron relaxation times τ_c^{e,m} for m = ES2, ES1, GS (s)
tau_c_e = {
    'ES2': 3e-12,
    'ES1': 2e-12,
    'GS':  2e-12,
}

tau_e_e_im = {
    'ES2': 6.15e-12,
    'ES1': 7.99e-12,
    'GS':  10.63e-12,
}

# Interband recombination time in the WL (s)
tau_r_e_W  = 100e-12

tau_e_h_qd = 1.46E-12

# Spontaneous emission recombination times τ_Sp^m for m = ES2, ES1, GS (s)
tau_Sp = {
    'ES2': 2.8e-9,
    'ES1': 2.8e-9,
    'GS':  2.8e-9,
}

# Auger recombination times τ_Au^m for m = ES2, ES1, GS (s)
tau_Au = {
    'ES2': 110e-12,
    'ES1': 275e-12,
    'GS':  660e-12,
}

# Interband transition energies hbar*omega_im for i=(N+1)/2 (central group)
# m = ES2, ES1, GS  (eV → J)
hbar_omega = {
    'ES2': 1.098 * q,   # J
    'ES1': 1.042 * q,
    'GS':  0.959 * q,   # J
}

# Reference (Bragg) angular frequency — set to GS transition of central group
omega_0 = hbar_omega['GS'] / hbar   # rad/s

# Intrinsic waveguide losses (m⁻¹)  [1.5 cm⁻¹ → ×100]
alpha_i  = 1.5e2

# ── Device parameters ─────────────────────────────────────────────────────────
W        = 2.2e-6            # ridge equivalent width (m)
h_SCH    = 430e-9            # SCH height (m)
r0_sq    = 0.90              # power reflectivity at z=0  (HR facet)
rL_sq    = 0.00              # power reflectivity at z=L  (AR facet)
L        = 400e-6            # total cavity length (m)
k_DFB    = 40e2              # grating coupling coefficient (m⁻¹)  [40 cm⁻¹]

# ── Derived / assumed parameters ──────────────────────────────────────────────
# Internal quantum efficiency (50% per [7] in the paper)
eta_i    = 0.50

# Field confinement factor in QD layers (lateral × transverse)
# Not given in table; typical value for this ridge geometry
Gamma_xy = 0.03             # (dimensionless) — approximate; tune to match gain

# Optical confinement factor of the SCH
Gamma_xy_SCH = 0.3          # (dimensionless) — approximate

# ── QD group energies (Gaussian inhomogeneous distribution) ──────────────────
# Groups indexed i = 0 ... N_groups-1; central group i_c = (N_groups-1)//2
sigma    = Delta_E / (2 * np.sqrt(2 * np.log(2)))   # Gaussian sigma (J)
i_c      = (N_groups - 1) // 2                       # index of central group
i_arr    = np.arange(N_groups)                        # 0 … 50

# Energy offset of each group from central group (evenly spaced, ±3σ span)
# Total span set to 4×FWHM to capture >99% of distribution
E_span   = 4 * Delta_E                               # total span (J)
dE       = E_span / (N_groups - 1)                   # spacing between groups

# GS transition energy for each group
E_GS_i   = hbar_omega['GS'] + (i_arr - i_c) * dE    # (J), shape (N_groups,)

# ES1 and ES2 energies follow same offset as GS (rigid shift of whole spectrum)
E_ES1_i  = hbar_omega['ES1'] + (i_arr - i_c) * dE
E_ES2_i  = hbar_omega['ES2'] + (i_arr - i_c) * dE

# Gaussian weighting G_i (normalised so sum = 1)
G_i      = np.exp(-0.5 * ((i_arr - i_c) * dE / sigma) ** 2)
G_i     /= G_i.sum()

# ── Valence-band confined-state energies (hole states) ───────────────────────
# Five hole confined states (GS, ES1…ES4) equally separated by 12 meV
# (stated in Section III-A of the paper)
delta_h  = 12e-3 * q        # hole level separation (J)
# Hole state energies relative to WL (negative = below WL continuum)
# Labelled GS=0, ES1=1, ES2=2, ES3=3, ES4=4
n_hole_states = 5
E_hole   = np.array([-k * delta_h for k in range(n_hole_states)])  # (J)
D_hole   = np.array([2, 4, 6, 8, 10])  # degeneracy (assumed 2*(k+1))

# ── Collect everything into a single dictionary ───────────────────────────────
params = {
    # Physical constants
    'q':            q,
    'hbar':         hbar,
    'kB':           kB,
    'c0':           c0,
    'eps0':         eps0,
    'm0':           m0,
    'T':            T,

    # Material
    'h_w':          h_w,
    'eta':          eta,
    'N_l':          N_l,
    'N_D':          N_D,
    'N_groups':     N_groups,
    'D':            D,
    'hbar_Gamma':   hbar_Gamma,
    'Gamma':        Gamma,
    'Delta_E':      Delta_E,
    'A':            A,
    'tau_c_e_W':    tau_c_e_W,
    'tau_e_e_W':    tau_e_e_W,
    'tau_c_h_W':    tau_c_h_W,
    'tau_e_h_qd':   tau_e_h_qd,
    'tau_c_e':      tau_c_e,
    'tau_e_e_im':   tau_e_e_im,
    'tau_r_e_W':    tau_r_e_W,
    'tau_Sp':       tau_Sp,
    'tau_Au':       tau_Au,
    'hbar_omega':   hbar_omega,
    'omega_0':      omega_0,
    'alpha_i':      alpha_i,

    # Device
    'W':            W,
    'h_SCH':        h_SCH,
    'r0_sq':        r0_sq,
    'rL_sq':        rL_sq,
    'L':            L,
    'k_DFB':        k_DFB,

    # Derived / assumed
    'eta_i':        eta_i,

    'Gamma_xy':     Gamma_xy,
    'Gamma_xy_SCH': Gamma_xy_SCH,

    # QD group arrays (length N_groups)
    'i_arr':        i_arr,
    'G_i':          G_i,
    'E_GS_i':       E_GS_i,
    'E_ES1_i':      E_ES1_i,
    'E_ES2_i':      E_ES2_i,
    'dE':           dE,

    # Hole confined states
    'n_hole_states': n_hole_states,
    'E_hole':       E_hole,
    'D_hole':       D_hole,
    'delta_h':      delta_h,
}


if __name__ == '__main__':
    print("=== QD-DFB Laser Parameters (SI units) ===\n")
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:20s}: array shape {v.shape}, "
                  f"range [{v.min():.4g}, {v.max():.4g}]")
        elif isinstance(v, dict):
            print(f"  {k:20s}: {v}")
        else:
            print(f"  {k:20s}: {v:.6g}" if isinstance(v, float) else
                  f"  {k:20s}: {v}")

# %%
# ============================================================
# Cell 1 — Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# %%

# ============================================================
# Cell 2 — Fixed arrays and constants
#           Assumes `params`, `i_arr`, `i_c`, `dE`, `sigma`
#           are already defined in your environment.
# ============================================================

N   = 51
n_m = 3   # [0] GS  [1] ES1  [2] ES2

# Inhomogeneous broadening distribution — shape (n_m, N)
G_i  = np.exp(-0.5 * ((i_arr - i_c) * dE / sigma) ** 2)
G_i /= G_i.sum()
G_i  = np.stack([G_i, G_i, G_i], axis=0)

# Degeneracy factors — shape (n_m, N)
D_m = np.array([2, 4, 6])[:, None] * np.ones((n_m, N))
A_m = np.array([params['A']['GS'], params['A']['ES1']])[:, None] * np.ones((2, N))

# Hole sub-band degeneracies and energies
D_m_h   = np.array([2, 4, 6, 8, 10], dtype=int)
delta_E = 12e-3 * 1.6e-19                           # [J]
E_h_m   = np.array([5, 4, 3, 2, 1]) * delta_E
E_h_WL  = 0.0

kBT = params['kB'] * params['T']                    # thermal energy [J]

# Auger and spontaneous-emission lifetimes — shape (n_m, N)
auger_lifetimes = np.zeros((n_m, N))
auger_lifetimes[0, :] = params['tau_Au']['GS']
auger_lifetimes[1, :] = params['tau_Au']['ES1']
auger_lifetimes[2, :] = params['tau_Au']['ES2']

sp_lifetimes = np.zeros((n_m, N))
sp_lifetimes[0, :] = params['tau_Sp']['GS']
sp_lifetimes[1, :] = params['tau_Sp']['ES1']
sp_lifetimes[2, :] = params['tau_Sp']['ES2']

omegahbar = np.array([params['E_GS_i']/params['hbar'],params['E_ES1_i']/params['hbar']])

delta_E_sch_w   = 95.1 # UNCERTAIN - CLAUDE suggests 120 y
delta_E_h_sch_w = 154 # UNCERTAIN - CLAUDE suggests 150  z
delta_e_w_es2   = 36.9 # UNCERTAIN - CLAUDE suggests 52  x
delta_e_es2_es1 = 44 #CERTAIN
delta_e_es1_gs  = 71 #CERTAOM 


# %%
# ============================================================
# Cell 3 — Helper functions
#           These rely on kBT, E_h_m, E_h_WL, D_m_h, params
#           defined in Cell 2.
# ============================================================

def fun(EF, N_h_qd, params, m_h_w):
    """Charge-neutrality equation for holes (energies in Joules)."""
    wl_term = (m_h_w* 9.11E-31 * params['N_l'] * kBT
               / (np.pi * params['hbar']**2)
               ) * np.logaddexp(0.0, (E_h_WL - EF) / kBT)
    exponent      = (EF - E_h_m) / kBT
    confined_term = np.sum(params['N_l'] * params['N_D'] * D_m_h / (1.0 + np.exp(exponent)))
    return confined_term + wl_term - N_h_qd


def fun_eV(EF_eV, N_h_qd, params, m_h_w):
    """Wrapper for fun() that accepts EF in electronvolts."""
    return fun(EF_eV * 1.6e-19, N_h_qd, params, m_h_w)


def fermi(E, EF):
    """Fermi–Dirac occupation probability."""
    return 1.0 / (1.0 + np.exp((EF - E) / kBT))

def gain(omega, rho_e_im, rho_hmat, params):
    gain = (params['Gamma_xy']/params['h_w']*params['N_D']
            * np.sum(G_i[:2]*D_m[:2]*A_m*1j/params['hbar']/np.pi
                     / (params['Gamma']+1j*(omega-omegahbar))
                     * (rho_e_im[:2]+rho_hmat[:2]-1)))
    return gain

# %%
def run_qd_simulation(t_end, initial_values, tstep, J=1E6, verbose=True,
                       threshold=0.01, params=None, delta_E_sch_w=95.1,
                       delta_E_h_sch_w=154, m_e_sch=0.063, m_e_w=0.03,
                       m_h_sch=0.5, m_h_w=0.45):
    if params is None:
        params = globals()['params']
    params = dict(params)

    delta_e_w_es2 = 286 - delta_E_sch_w - delta_E_h_sch_w

    DOS_sch   = 2*((2*np.pi*m_e_sch*9.11E-31*4.11E-21)/(6.626E-34)**2)**(3/2)
    DOS_w     = ((m_e_w*9.11E-31*4.11E-21)/(np.pi*params['hbar']**2))
    DOS_sch_h = 2*((2*np.pi*m_h_sch*9.11E-31*4.11E-21)/(6.626E-34)**2)**(3/2)
    DOS_w_h   = ((m_h_w*9.11E-31*4.11E-21)/(np.pi*params['hbar']**2))

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

        EF_lo_eV = (E_h_m[0] - 100 * kBT) / 1.6e-19
        EF_hi_eV = (E_h_WL  + 100 * kBT)  / 1.6e-19
        sol      = root_scalar(fun_eV, args=(n_hq_qd, params, m_h_w),
                               bracket=[EF_lo_eV, EF_hi_eV], method='brentq')
        root_J   = sol.root * 1.6e-19

        rho_h_m   = fermi(E_h_m, root_J)
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
        'n_e_im':  n_e_im,        # shape (n_m_loc, N_loc)
        'n_h_sch': n_h_sch,
        'n_hq_qd': n_hq_qd,
    }
    t_final = step * tstep
    return t_final, final_state, step

# %%
def plot_gain_and_carriers(final_state, len_omega=100, params=None,
                           wavelength_range_nm=None, m_h_w=0.45):
    if params is None:
        params = globals()['params']
    params = dict(params)

    if wavelength_range_nm is not None:
        wl_min_m = wavelength_range_nm[0] * 1e-9
        wl_max_m = wavelength_range_nm[1] * 1e-9
        omega_max = (2 * np.pi * params['c0']) / wl_min_m
        omega_min = (2 * np.pi * params['c0']) / wl_max_m
    else:
        omega_min = 1.35e15
        omega_max = 1.65e15

    omega = np.linspace(omega_min, omega_max, len_omega)

    n_e_im_final = final_state['n_e_im']       # shape (n_m_loc, N_loc)
    n_hq_qd_final = final_state['n_hq_qd']

    rho_e_im_final = n_e_im_final / (params['N_l'] * params['N_D'] * G_i * D_m)
    rho_e_gs_avg   = np.sum(G_i[0, :] * rho_e_im_final[0, :])
    rho_e_es1_avg  = np.sum(G_i[1, :] * rho_e_im_final[1, :])

    q_exact  = 1.602176634e-19
    EF_lo_eV = (E_h_m[0] - 15 * kBT) / q_exact
    EF_hi_eV = (E_h_WL   + 15 * kBT) / q_exact

    try:
        sol = root_scalar(fun_eV, args=(n_hq_qd_final, params, m_h_w),
                          bracket=[EF_lo_eV, EF_hi_eV], method='brentq')
        EF_h_final = sol.root * q_exact
    except ValueError as e:
        print(f"Root finding failed! n_hq_qd = {n_hq_qd_final:.4e}")
        raise e

    rho_h_m   = fermi(E_h_m, EF_h_final)
    rho_h_mat = np.zeros((2, 51))
    rho_h_mat[0, :] = rho_h_m[0]
    rho_h_mat[1, :] = rho_h_m[1]

    gain_spectrum = gain_spectrum = np.array([np.imag(gain(w, rho_e_im_final, rho_h_mat, params))
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

    total_electrons   = final_state['n_e_sch'] + final_state['n_e_w'] + n_e_qd
    total_holes       = final_state['n_h_sch'] + final_state['n_hq_qd']
    carrier_difference = total_electrons - total_holes

    return {
        'wavelength_nm':    wavelength_nm,
        'gain_cm_minus_1':  gain_cm,
        'total_electrons':  total_electrons,
        'total_holes':      total_holes,
        'carrier_difference': carrier_difference,
    }

# %%
# ============================================================
# Cell 5 — Define initial values and run
# ============================================================


def generate_custom_samples(total_points=500, high_density_ratio=0.6):
    """
    Generates a sample array for J from 0 to 18E8 with a higher density 
    of points concentrated between 4.5E6 and 18E6.
    
    Parameters:
    - total_points (int): Total number of samples in the final array.
    - high_density_ratio (float): Fraction of total points allocated to the target region.
    """
    # 1. Calculate how many points go into each of the 3 segments
    n_high = int(total_points * high_density_ratio)
    # Split the remaining points between the lower and upper outer zones
    n_remaining = total_points - n_high
    n_low = int(n_remaining * 0.25) # 0 to 4.5E6 is a narrow outer band
    n_upper = total_points - n_high - n_low
    
    # 2. Generate the segments
    # Use endpoint=False to prevent duplicate values at the boundary connections
    seg1 = np.linspace(0, 4.5e6, n_low, endpoint=False)
    seg2 = np.linspace(4.5e6, 18e6, n_high, endpoint=False)
    seg3 = np.linspace(18e6, 18e8, n_upper)
    
    # 3. Concatenate them together
    J = np.concatenate([seg1, seg2, seg3])
    return J


initial_values = {
    'n_e_sch' : 0.0,
    'n_e_w'   : 0.0,
    'n_e_im'  : np.zeros((n_m, N)),   # shape (n_m, N)
    'n_h_sch' : 0.0,
    'n_hq_qd' : 1e6,                 # initial hole population in QD reservoir
}

t_end = 20e-9    # [s]  — total simulation time  ← adjust as needed
tstep = 30e-15  # [s]  — timestep               ← adjust as needed
import numpy as np

# Example usage:
J_arr = generate_custom_samples(total_points=500, high_density_ratio=0.6)

N_arr = len(J_arr)
len_omega = 100
gain_array = np.zeros((N_arr,len_omega,))
N_array = np.zeros(N_arr)
rt_ev = 25.9

# %%
output_dir = 'gain_table'
os.makedirs(output_dir, exist_ok=True)

len_omega = 100

for J_val in J_arr:
    print(f'J= {J_val}')

    t_final, final_state, last_step = run_qd_simulation(
        t_end=30e-9,
        initial_values=initial_values,
        tstep=60e-15,
        threshold=0.05,
        J=J_val,
        verbose=True,
        params=None,
        m_h_w=0.4,
        m_e_w=0.026,
        delta_E_sch_w=75,
        delta_E_h_sch_w=140,
    )

    filename = f"J_{J_val}.npz"
    filepath = os.path.join(output_dir, filename)

    np.savez_compressed(
        filepath,
        t_final=np.array([t_final]),
        last_step=np.array([last_step]),
        **final_state,      # n_e_sch, n_e_w, n_e_im, n_h_sch, n_hq_qd
    )
    print(f"  Saved → {filepath}")

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

output_dir = 'gain_table'
len_omega  = 100  # must match what was used during simulation

J_trimmed = J_arr[1:]
gain_array = np.zeros((len(J_trimmed), len_omega))
N_sweep_arr = np.zeros(len(J_trimmed))

wavelength = None
for index, J_val in enumerate(J_trimmed):
    filename = f"J_{J_val}.npz"
    filepath = os.path.join(output_dir, filename)
    data = np.load(filepath)
    final_state = {
        'n_e_sch': float(data['n_e_sch']),
        'n_e_w':   float(data['n_e_w']),
        'n_e_im':  data['n_e_im'],          # shape (3, 51)
        'n_h_sch': float(data['n_h_sch']),
        'n_hq_qd': float(data['n_hq_qd']),
    }

    result_output = plot_gain_and_carriers(
        final_state,
        len_omega=len_omega,
    )

    gain_array[index] = result_output['gain_cm_minus_1']
    N_sweep_arr[index] = result_output['total_electrons'] 


    if wavelength is None:
        wavelength = result_output['wavelength_nm']

print(result_output['total_electrons'])    

# %%
conversion_factor = 4019376.8605
for index, J_val in enumerate(J_trimmed):
    plt.plot(wavelength, 4019376.8605*gain_array[index],label = f'{N_sweep_arr[index]:.4f}')
plt.show()
gamma_xy = 0.08303645356032915
material_gain = 4019376.8605*gain_array[index]/gamma_xy

# %%
# ── Write PICWAVE-compatible wide-band gain table ─────────────────────────────
#
# Assumes the following variables are already defined in the notebook:
#   gain_array      ndarray (n_J, len_omega)  raw gain output from plot_gain_and_carriers
#   N_sweep_arr     ndarray (n_J,)            carrier density [cm^-3] (already volumetric)
#   wavelength      ndarray (len_omega,)      wavelength [nm]
#   conversion_factor  float                  params['omega_0'] / (3e8 * 3.3445)
#   gamma_xy        float                     optical confinement factor
#
# Output file: qd_gain_TE.txt
# Reference in your .mat file with:
#   IMPORT_GAIN_SPECTRA_TE qd_gain_TE.txt

import numpy as np

OUTPUT_FILE   = 'qd_gain_TE.txt'
TEMPERATURE_C = 25.0
height = 430E-7+40E-7 

# ── Convert to material gain [cm^-1] ─────────────────────────────────────────
# modal gain   = gain_array * conversion_factor          [cm^-1]
# material gain = modal gain / gamma_xy                  [cm^-1]

material_gain = gain_array * conversion_factor / gamma_xy   # shape (n_J, len_omega)

# ── Sort by carrier density (ascending) ──────────────────────────────────────
sort_idx      = np.argsort(N_sweep_arr)
N_sorted      = N_sweep_arr[sort_idx]*1E-4/height                       # [cm^-3]
gain_sorted   = material_gain[sort_idx]                     # (n_J, len_omega)

n_N           = len(N_sorted)
n_lambda      = len(wavelength)
lambda_min_um = wavelength.min() / 1000.0                   # nm → um
lambda_max_um = wavelength.max() / 1000.0

print(f"N range  : {N_sorted[0]:.4e} – {N_sorted[-1]:.4e} cm^-3  ({n_N} points)")
print(f"λ range  : {lambda_min_um:.4f} – {lambda_max_um:.4f} um  ({n_lambda} points)")
print(f"Gain range: {gain_sorted.min():.1f} – {gain_sorted.max():.1f} cm^-1")

# ── Write file ────────────────────────────────────────────────────────────────
#
# Format (negainspectrum):
#
#   begin <negainspectrum(1,0)>
#   nLambda  lambdaMin[um]  lambdaMax[um]  nT  minT[C]  maxT[C]  poln
#
#   //T=25 [C]
#   nN
#   N1  N2  ...  Nn                          (one line, cm^-3)
#   g(N1,lam1)  g(N2,lam1)  ...  g(Nn,lam1) (one row per wavelength)
#   g(N1,lam2)  g(N2,lam2)  ...
#   ...
#   end

with open(OUTPUT_FILE, 'w') as f:

    f.write('begin <negainspectrum(1,0)>\n')
    f.write(
        f'{n_lambda} '
        f'{lambda_min_um:.6f} '
        f'{lambda_max_um:.6f} '
        f'1 '
        f'{TEMPERATURE_C:.1f} '
        f'{TEMPERATURE_C:.1f} '
        f'1\n'                   # poln = 1 (TE)
    )

    f.write(f'\n//T={TEMPERATURE_C:.0f} [C]\n')
    f.write(f'{n_N} //nN\n')

    # Carrier densities — one line, space-separated [cm^-3]
    f.write(' '.join(f'{n:.6e}' for n in N_sorted) + '\n')

    # Gain block — one row per wavelength, columns = carrier densities
    # gain_sorted has shape (n_N, n_lambda); PICWAVE wants (n_lambda, n_N)
    for lam_idx in range(n_lambda):
        row = gain_sorted[:, lam_idx]
        f.write(' '.join(f'{g:.6e}' for g in row) + '\n')

    f.write('end\n')

print(f"\nGain table written → {OUTPUT_FILE}")
print(f"Add to your .mat file:  IMPORT_GAIN_SPECTRA_TE {OUTPUT_FILE}")

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

output_dir = 'gain_table'
len_omega  = 100  # must match what was used during simulation

m_h_w_arr           = [0.4]#], 0.375]#, 0.41]
m_e_w_arr           = [0.026]#, 0.0315, 0.04]
delta_E_sch_w_arr   = [75,80,85,90]
delta_E_h_sch_w_arr = [125,130,140,150]


keys = [
    (mh, me, E_sch, E_h, J_val)
    for mh    in m_h_w_arr
    for me    in m_e_w_arr
    for E_sch in delta_E_sch_w_arr
    for E_h   in delta_E_h_sch_w_arr
    for J_val in J_array
]

n_combos   = len(keys)
gain_array = np.zeros((n_combos, len_omega))
wavelength = None
for index, (mh_factor, me_factor, E_sch, E_h, J_val) in enumerate(keys):
    filename = f"J_{J_val}.npz"
    filepath = os.path.join(output_dir, filename)
    data = np.load(filepath)
    final_state = {
        'n_e_sch': float(data['n_e_sch']),
        'n_e_w':   float(data['n_e_w']),
        'n_e_im':  data['n_e_im'],          # shape (3, 51)
        'n_h_sch': float(data['n_h_sch']),
        'n_hq_qd': float(data['n_hq_qd']),
    }

    result_output = plot_gain_and_carriers(
        final_state,
        len_omega=len_omega,
        m_h_w=mh_factor,
    )

    gain_array[index] = result_output['gain_cm_minus_1']

    if wavelength is None:
        wavelength = result_output['wavelength_nm']

print(result_output['total_electrons'])    



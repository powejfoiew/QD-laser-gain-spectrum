"""
Parameters from Table I of:
Gioannini & Rossetti, "Time-Domain Traveling Wave Model of Quantum Dot DFB Lasers"
IEEE J. Sel. Top. Quantum Electron., Vol. 17, No. 5, 2011

All values are in SI units unless noted.

`delta_E_sch_w`, `delta_E_h_sch_w` and `delta_e_w_es2` (see build_params()) are
not from the paper: G&R do not give the transport-energy offsets needed to
reproduce their reported gain curve. They were hand-fit to keep the carrier
rate equations numerically stable and lasing near the intended threshold
(thesis section 4.2.1 / Appendix 7.3), rather than derived from theory.

The original code mixed the exact elementary charge (`q` above) with an
approximate literal (1.6e-19) at various points that convert between eV and
Joules. All such conversions here use the exact `q`; this introduces a
~0.01% shift relative to the original notebook's approximate-conversion
code paths, accepted as negligible relative to the model's other
uncertainties (see above).
"""

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
q        = 1.602176634e-19   # elementary charge (C)
hbar     = 1.054571817e-34   # reduced Planck constant (J·s)
h_planck = 6.62607015e-34    # Planck constant (J·s) — non-reduced, for DOS formulas
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

# Field confinement factor in QD layers (lateral × transverse), used in the
# gain calculation (gain_model.gain/gain_index). Not given in table; typical
# value for this ridge geometry.
Gamma_xy = 0.03             # (dimensionless) — approximate; tune to match gain

# Optical confinement factor of the SCH, used in the refractive-index-change
# calculation (gain_model.refractive_index_change). The original code carried
# an unrelated, unused placeholder of 0.3 here while refractive_index_change
# hardcoded its own working value of 0.06985092 — since nothing else ever
# referenced the placeholder, this now holds the value actually used.
Gamma_xy_SCH = 0.06985092   # (dimensionless) — approximate

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


def build_params():
    """
    Assemble the full parameter/state dictionary consumed by
    `qd_gain.carrier_dynamics` and `qd_gain.gain_model`.

    Includes both the raw Table-I parameters above and the derived
    per-level arrays (broadening weights, degeneracies, dipole matrix
    elements, lifetimes, hole confined-state energies) that the rate
    equations and gain calculation are evaluated against. These were
    previously recomputed as bare module-level globals in whichever
    notebook cell happened to run last; collecting them here means every
    downstream function receives them explicitly instead of relying on
    notebook execution order.
    """
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

    # ── Per-transition-level arrays used by carrier_dynamics/gain_model ──────
    # n_m indexes the three electron levels tracked per QD group: [0] GS,
    # [1] ES1, [2] ES2.
    n_m = 3
    N   = N_groups

    kBT = params['kB'] * params['T']   # thermal energy (J)

    # Inhomogeneous broadening distribution, broadcast to (n_m, N) — same
    # per-group Gaussian weights for every level, since the QD size
    # distribution is shared across transitions.
    G_i_by_level = np.stack([G_i, G_i, G_i], axis=0)

    # Degeneracy factors D_m and dipole matrix elements A_m, broadcast to
    # (n_m, N) / (2, N) respectively (only GS and ES1 have measured dipole
    # matrix elements — ES2 gain is evaluated separately, see gain_index()).
    D_m = np.array([2, 4, 6])[:, None] * np.ones((n_m, N))
    A_m = np.array([A['GS'], A['ES1']])[:, None] * np.ones((2, N))

    # Auger and spontaneous-emission lifetimes, broadcast to (n_m, N).
    auger_lifetimes = np.zeros((n_m, N))
    auger_lifetimes[0, :] = tau_Au['GS']
    auger_lifetimes[1, :] = tau_Au['ES1']
    auger_lifetimes[2, :] = tau_Au['ES2']

    sp_lifetimes = np.zeros((n_m, N))
    sp_lifetimes[0, :] = tau_Sp['GS']
    sp_lifetimes[1, :] = tau_Sp['ES1']
    sp_lifetimes[2, :] = tau_Sp['ES2']

    # Transition angular frequencies for GS/ES1, per QD group — (2, N).
    omegahbar = np.array([E_GS_i / hbar, E_ES1_i / hbar])

    # Hole confined-state degeneracies/energies as used by fun()/fermi()
    # (distinct convention from E_hole/D_hole above: energies here are
    # positive offsets below the wetting-layer continuum, ordered
    # ES4..GS, matching Section III-A of the paper). Same 12 meV level
    # spacing as delta_h above — now computed from the same exact `q`.
    D_m_h  = np.array([2, 4, 6, 8, 10], dtype=int)
    E_h_m  = np.array([5, 4, 3, 2, 1]) * delta_h
    E_h_WL = 0.0

    # Transport-energy offsets (meV) between confined states, used in the
    # thermionic-emission relaxation-time expressions in carrier_dynamics.py.
    # See module docstring: not given in G&R, hand-fit for numerical stability.
    delta_E_sch_w   = 95.1  # UNCERTAIN - CLAUDE suggests 120 y
    delta_E_h_sch_w = 154   # UNCERTAIN - CLAUDE suggests 150  z
    delta_e_w_es2   = 36.9  # UNCERTAIN - CLAUDE suggests 52  x
    delta_e_es2_es1 = 44    #CERTAIN
    delta_e_es1_gs  = 71    #CERTAOM

    rt_ev = 25.9   # thermal voltage k_B*T expressed in meV, used in exp(dE/rt_ev)

    # ── run_qd_simulation() default effective masses (relative to m0) ───────
    # These reproduce the function's original keyword defaults; every call
    # in generate_gain_table.py overrides delta_E_sch_w/delta_E_h_sch_w/
    # m_e_w/m_h_w explicitly, but m_e_sch/m_h_sch are left at these defaults.
    m_e_sch_default = 0.063
    m_e_w_default   = 0.03
    m_h_sch_default = 0.5
    m_h_w_default   = 0.45

    # ── refractive_index_change() defaults ──────────────────────────────────
    # Effective masses (relative to m0) for the free-carrier Drude term —
    # distinct from the simulation masses above (this is a separate,
    # standalone calculation, not part of the rate-equation solver).
    m_e_drude = 0.0465
    m_h_drude = 0.455
    # QD-layer confinement factor as used in the refractive-index
    # calculation — kept distinct from Gamma_xy (gain calculation) above,
    # since the two were independently fit to different quantities and are
    # not necessarily the same physical value despite the similar name.
    Gamma_xy_index = 0.088
    # Uskov correction factors (fraction of confined-state population that
    # contributes to the free-carrier plasma term, vs. remaining "dot-like").
    S_re_e_default = 0.8
    S_re_h_default = 0.8

    params.update({
        'n_m':               n_m,
        'N':                 N,
        'kBT':               kBT,
        'h_planck':          h_planck,
        'G_i_by_level':      G_i_by_level,
        'D_m':               D_m,
        'A_m':               A_m,
        'auger_lifetimes':   auger_lifetimes,
        'sp_lifetimes':      sp_lifetimes,
        'omegahbar':         omegahbar,
        'D_m_h':             D_m_h,
        'E_h_m':             E_h_m,
        'E_h_WL':            E_h_WL,
        'delta_E_sch_w':     delta_E_sch_w,
        'delta_E_h_sch_w':   delta_E_h_sch_w,
        'delta_e_w_es2':     delta_e_w_es2,
        'delta_e_es2_es1':   delta_e_es2_es1,
        'delta_e_es1_gs':    delta_e_es1_gs,
        'rt_ev':             rt_ev,
        'm_e_sch_default':   m_e_sch_default,
        'm_e_w_default':     m_e_w_default,
        'm_h_sch_default':   m_h_sch_default,
        'm_h_w_default':     m_h_w_default,
        'm_e_drude':         m_e_drude,
        'm_h_drude':         m_h_drude,
        'Gamma_xy_index':    Gamma_xy_index,
        'S_re_e_default':    S_re_e_default,
        'S_re_h_default':    S_re_h_default,
    })

    return params

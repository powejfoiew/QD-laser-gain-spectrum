import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# System Parameters (from Marie & Alléaume 2017, Fig. 4)
# =============================================================================

beta = 0.95          # Reconciliation efficiency
eta = 0.7            # Detector quantum efficiency
v_elec = 0.01        # Electronic noise variance (SNU)
xi_tech = 0.01       # Technical excess noise (SNU)
f = 100e6            # Repetition rate (Hz)
V_A = 4.0            # Modulation variance (SNU)
E_R_squared = 100 * V_A  # Reference pulse intensity (photon number)
d_dB = 40            # Amplitude modulator dynamics (dB)
delta_det = 2        # Heterodyne detection (delta_det = 2)
L_fibre = 1         # Fibre length (km)
alpha_fibre = 0.2    # Fibre loss (dB/km)

# =============================================================================
# Derived Channel Parameters
# =============================================================================

# Channel transmission (fibre loss only)
T_channel = 10**(-alpha_fibre * L_fibre / 10)

# Total transmission seen by the signal (channel + detector)
G = T_channel * eta

V = V_A + 1  # Total signal variance


def secret_key_rate(delta_nu_total, f, V_A, E_R_squared, d_dB,
                    G, eta, beta, v_elec, xi_tech, delta_det=2):
    """
    Calculate secret key rate as a function of total linewidth.
    
    Uses the security model from Marie & Alléaume (2017), Appendix Eq. A12-A17,
    under individual attacks with Eve controlling Bob's detection noise.
    """
    V = V_A + 1

    # --- Step 1: Phase drift variance (Eq. 18) ---
    V_drift = 2 * np.pi * delta_nu_total / f

    # --- Step 2: Detection noise for phase estimation (Eq. 11) ---
    # chi in Eq. 11 refers to the noise on the reference pulse measurement.
    # For heterodyne detection: shot noise (delta_det=2 contributes 2 SNU)
    # plus electronic noise referred to input
    # The "chi + 1" term in Eq. 11 represents the total noise variance
    # on the heterodyne measurement of the reference pulse, normalised to SNU.
    # For heterodyne: shot noise contributes 2 (both quadratures),
    # so chi + 1 = delta_det + v_elec/eta = 2 + v_elec/eta
    # But referred to the channel input, we need to account for losses.
    # 
    # In the paper's convention, Eq. 11 gives:
    # V_error = (chi + 1) / E_R^2
    # where chi is defined by Eq. A2: chi = (delta_det - G)/G + xi
    # and E_R^2 is the reference photon number AT BOB (after channel loss)
    #
    # Simpler interpretation: V_error = noise_on_measurement / signal_power
    # For a reference pulse with E_R photons at Alice, Bob receives G*E_R^2 photons.
    # Heterodyne measurement noise = delta_det + 2*v_elec (both quadratures)
    # V_error = (delta_det + 2*v_elec) / (G * E_R_squared)
    
    # Following the paper's approach more carefully:
    # E_R is defined at Alice's output, so after channel: E_R_bob^2 = G * E_R^2
    # The phase estimation error is shot-noise limited:
    # V_error = 1 / (E_R_bob^2) for ideal case
    # With added noise: V_error = (1 + noise_floor) / E_R_bob^2
    
    # Using Eq. 11 directly with chi from Eq. A2 (detection noise only):
    chi_det = (delta_det - G) / G + v_elec * delta_det / G
    V_error = (chi_det + 1) / E_R_squared
    
    # --- Step 3: Total residual phase variance (Eq. 12, V_channel ~ 0) ---
    V_est = V_drift + V_error

    # --- Step 4: Phase-induced excess noise (Eq. 5) ---
    xi_phase = 2 * V_A * (1 - np.exp(-V_est / 2))

    # --- Step 5: AM leakage noise (Eq. 17) ---
    xi_AM = E_R_squared * 10**(-d_dB / 10)

    # --- Step 6: Total excess noise ---
    xi_total = xi_phase + xi_AM + xi_tech

    # --- Step 7: Total channel noise referred to input (Eq. A2) ---
    # Include electronic noise in the total
    xi_with_elec = xi_total + v_elec * delta_det / G
    chi = (delta_det - G) / G + xi_with_elec

    # --- Step 8: Mutual information I_AB (Eq. A13) ---
    I_AB = 0.5 * np.log2((V + chi) / (1 + chi))

    # --- Step 9: Eve's information I_BE (Eqs. A14, A17) ---
    # Eve controls all excess noise (pessimistic model)
    # chi_E from Eq. A17
    xi_eve = xi_total + v_elec * delta_det / G  # Eve controls detection too
    
    # Simplified approach: use the standard formula for individual attacks
    # From Fossier (2009) / Garcia-Patron (2008):
    # I_BE = G(V-1)(1 + xi_eve) / (G(V + chi))  ... but let's use the log form
    
    # Eq. A17: chi_E
    with np.errstate(invalid='ignore', divide='ignore'):
        sqrt_term1 = np.sqrt(np.maximum(2 - 2*G + G*xi_eve, 0))
        sqrt_term2 = np.sqrt(np.maximum(xi_eve, 0))
        denominator = (sqrt_term1 + sqrt_term2)**2 + 1
        chi_E = G * (2 - xi_eve)**2 / denominator
    
    # Eq. A14: I_BE
    with np.errstate(invalid='ignore', divide='ignore'):
        arg = G * (V + chi) * (V + chi_E) / ((chi_E + 1) * (V + 1))
        I_BE = np.where(arg > 0, 0.5 * np.log2(np.maximum(arg, 1e-30)), 0)

    # --- Step 10: Secret key rate (Eq. A12) ---
    k = beta * I_AB - I_BE

    # Key rate cannot be negative
    k = np.maximum(k, 0.0)

    return k


# =============================================================================
# Debug: Check values at zero linewidth
# =============================================================================

print("="*60)
print("DIAGNOSTIC: Values at zero linewidth")
print("="*60)
print(f"  G (total transmission) = {G:.6f} ({10*np.log10(G):.2f} dB)")
print(f"  V = V_A + 1 = {V}")

# At zero linewidth
V_drift_0 = 0
chi_det = (delta_det - G) / G + v_elec * delta_det / G
V_error_0 = (chi_det + 1) / E_R_squared
V_est_0 = V_drift_0 + V_error_0
xi_phase_0 = 2 * V_A * (1 - np.exp(-V_est_0 / 2))
xi_AM_0 = E_R_squared * 10**(-d_dB / 10)
xi_total_0 = xi_phase_0 + xi_AM_0 + xi_tech
xi_with_elec_0 = xi_total_0 + v_elec * delta_det / G
chi_0 = (delta_det - G) / G + xi_with_elec_0

print(f"  chi_det = {chi_det:.6f}")
print(f"  V_error = {V_error_0:.6f}")
print(f"  V_est (at dv=0) = {V_est_0:.6f}")
print(f"  xi_phase (at dv=0) = {xi_phase_0:.6f}")
print(f"  xi_AM = {xi_AM_0:.6f}")
print(f"  xi_total (at dv=0) = {xi_total_0:.6f}")
print(f"  chi (at dv=0) = {chi_0:.6f}")
print(f"  V + chi = {V + chi_0:.6f}")
print(f"  1 + chi = {1 + chi_0:.6f}")

I_AB_0 = 0.5 * np.log2((V + chi_0) / (1 + chi_0))
print(f"  I_AB (at dv=0) = {I_AB_0:.6f}")

k_test = secret_key_rate(0, f, V_A, E_R_squared, d_dB, G, eta, beta, v_elec, xi_tech)
print(f"  Key rate at dv=0: {float(k_test):.6f} bits/pulse")
print("="*60)

# =============================================================================
# Generate Plot
# =============================================================================

linewidths = np.linspace(0, 500e3, 10000)  # 0 to 500 kHz

key_rates = secret_key_rate(linewidths, f, V_A, E_R_squared, d_dB,
                            G, eta, beta, v_elec, xi_tech)

# Find cutoff
nonzero_mask = key_rates > 0
if np.any(nonzero_mask):
    cutoff_idx = np.where(nonzero_mask)[0][-1]
    cutoff_linewidth = linewidths[cutoff_idx]
    print(f"\nCutoff linewidth: {cutoff_linewidth/1e3:.1f} kHz")
else:
    cutoff_linewidth = None
    print("\nNo positive key rate found — further debugging needed.")
    print("Try reducing L_fibre or increasing V_A or E_R_squared.")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: Linear
ax1.plot(linewidths / 1e3, key_rates, 'b-', linewidth=2)
if cutoff_linewidth:
    ax1.axvline(x=cutoff_linewidth / 1e3, color='r', linestyle='--',
                linewidth=1.5, label=f'Cutoff: {cutoff_linewidth/1e3:.0f} kHz')
    ax1.legend(fontsize=11)
ax1.set_xlabel(r'Combined linewidth $\Delta\nu_A + \Delta\nu_B$ (kHz)', fontsize=12)
ax1.set_ylabel('Secret key rate (bits/pulse)', fontsize=12)
ax1.set_title(f'LLO-sequential, {L_fibre} km, f = {f/1e6:.0f} MHz', fontsize=11)
ax1.set_xlim([0, 500])
ax1.grid(True, alpha=0.3)

# Right panel: Log scale
key_rates_log = np.where(key_rates > 0, key_rates, np.nan)
ax2.semilogy(linewidths / 1e3, key_rates_log, 'b-', linewidth=2)
if cutoff_linewidth:
    ax2.axvline(x=cutoff_linewidth / 1e3, color='r', linestyle='--',
                linewidth=1.5, label=f'Cutoff: {cutoff_linewidth/1e3:.0f} kHz')
    ax2.legend(fontsize=11)
ax2.set_xlabel(r'Combined linewidth $\Delta\nu_A + \Delta\nu_B$ (kHz)', fontsize=12)
ax2.set_ylabel('Secret key rate (bits/pulse)', fontsize=12)
ax2.set_title('Log scale', fontsize=11)
ax2.set_xlim([0, 500])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SKR_vs_linewidth.pdf', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Print table
# =============================================================================
print("\nKEY RATES AT TYPICAL LINEWIDTHS:")
print("-"*50)
for lw_kHz in [0, 1, 5, 10, 50, 100, 150, 200, 300, 500]:
    kr = secret_key_rate(lw_kHz * 1e3, f, V_A, E_R_squared, d_dB,
                         G, eta, beta, v_elec, xi_tech)
    print(f"  Delta_nu = {lw_kHz:>4d} kHz  -->  k = {float(kr):.6f} bits/pulse")
print("-"*50)
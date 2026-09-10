import numpy as np

def calculate_q(n_eff, L_ext, loss_db_m, R_dbr, R_interface, f_0):
    # Speed of light
    c = 299792458
    
    # Round-trip time (s)
    tau_rt = 2 * n_eff * L_ext / c
    
    # Waveguide propagation loss (one-way, fraction)
    # loss in dB = -10 * log10(T) -> T = 10^(-loss/10)
    T_prop_one_way = 10**(-(loss_db_m * L_ext) / 10)
    
    # 1. Open-cavity / delay-line model (single round-trip lifetime)
    # Energy decays by propagation loss and DBR reflection, then leaks out entirely at the interface
    # Round-trip power transmission fraction: T_rt = T_prop_one_way^2 * R_dbr
    T_rt_open = (T_prop_one_way**2) * R_dbr
    # Photon lifetime (s): energy decays to 1/e in tau_p = -tau_rt / ln(T_rt)
    tau_p_open = -tau_rt / np.log(T_rt_open)
    Q_open = 2 * np.pi * f_0 * tau_p_open
    
    # 2. Resonant Fabry-Perot Cavity model
    # Cavity is formed between the DFB/waveguide interface (reflectivity R_interface) and DBR (reflectivity R_dbr)
    # Round-trip power transmission: T_rt = T_prop_one_way^2 * R_dbr * R_interface
    if R_interface > 0:
        T_rt_res = (T_prop_one_way**2) * R_dbr * R_interface
        tau_p_res = -tau_rt / np.log(T_rt_res)
        Q_res = 2 * np.pi * f_0 * tau_p_res
    else:
        Q_res = None
        tau_p_res = None
        
    return {
        'tau_rt_ns': tau_rt * 1e9,
        'prop_loss_rt_db': 2 * loss_db_m * L_ext,
        'T_rt_open': T_rt_open,
        'tau_p_open_ns': tau_p_open * 1e9,
        'Q_open': Q_open,
        'tau_p_res_ns': tau_p_res * 1e9 if tau_p_res else None,
        'Q_res': Q_res
    }

def main():
    # Parameters from the paper
    L_ext = 4.0          # external cavity length in m
    loss_db_m = 0.17     # waveguide loss in dB/m
    
    # DBR grating coupling strength: kappa * L = 1
    # Reflectivity: R_dbr = tanh^2(1)
    R_dbr = np.tanh(1.0)**2
    
    # Wavelength and frequency
    lambda_0 = 1558e-9   # m
    c = 299792458
    f_0 = c / lambda_0   # Hz
    
    print("--- ESTIMATING EXTERNAL CAVITY Q-FACTOR ---")
    print(f"External Cavity Length: {L_ext} m")
    print(f"Waveguide Loss: {loss_db_m} dB/m")
    print(f"DBR Power Reflectivity: {R_dbr*100:.2f}% (R = tanh^2(1))")
    print(f"Center Wavelength: {lambda_0*1e9:.1f} nm (Frequency: {f_0*1e-12:.4f} THz)")
    
    # Let's test a couple of effective refractive indices
    # 1. n_eff = 1.30 (first-order Bragg matching with 600nm period at 1558nm)
    # 2. n_eff = 1.50 (typical for loaded silicon nitride waveguides)
    for n_eff in [1.30, 1.50]:
        print(f"\n--- For Waveguide Effective Index n_eff = {n_eff:.2f} ---")
        
        # We check different interface reflectivities
        # R_interface = 0.0 (no reflection back into cavity, i.e., open delay line)
        # R_interface = 0.01 (1% residual reflection, typical for AR coatings)
        # R_interface = 0.10 (10% reflection)
        # R_interface = 0.30 (30% reflection, typical for cleaved silicon facet)
        
        for R_int in [0.0, 0.01, 0.10, 0.30]:
            res = calculate_q(n_eff, L_ext, loss_db_m, R_dbr, R_int, f_0)
            if R_int == 0.0:
                print(f"Open Cavity (Delay Line, R_interface = 0%):")
                print(f"  Round-trip time: {res['tau_rt_ns']:.2f} ns")
                print(f"  Photon Lifetime: {res['tau_p_open_ns']:.2f} ns")
                print(f"  Q-factor: {res['Q_open']/1e6:.2f} Million")
            else:
                print(f"Resonant Cavity (R_interface = {R_int*100:.0f}%):")
                print(f"  Photon Lifetime: {res['tau_p_res_ns']:.2f} ns")
                print(f"  Q-factor: {res['Q_res']/1e6:.2f} Million")

if __name__ == '__main__':
    main()

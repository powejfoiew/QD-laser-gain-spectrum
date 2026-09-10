import numpy as np
import matplotlib.pyplot as plt

# Refined evaluation functions ensuring smooth asymptotic approach to 1.0 without artificial cutoffs below 1

def rho_weibull(N, N_sat_base, k_base, n_base, mult_xm=1.0, mult_n=1.0):
    N_sat = N_sat_base * mult_xm
    n_val = n_base * mult_n
    
    u = np.clip(N / N_sat, 0.0, 1.0 - 1e-7)
    ratio = u / np.maximum(1e-7, 1.0 - u)
    out = 1.0 - np.exp(-k_base * (ratio ** n_val))
    return np.clip(out, 0.0, 1.0)

def rho_aug_GS_smooth(N, mult_xm=1.0, mult_n=1.0):
    # Weibull saturation model for GS Auger: smooth approach to 1.0 at N_sat = 2.35e17
    return rho_weibull(N, N_sat_base=2.355e17, k_base=0.105, n_base=0.85, mult_xm=mult_xm, mult_n=mult_n)

def rho_spon_GS_smooth(N, mult_xm=1.0, mult_n=1.0):
    return rho_weibull(N, N_sat_base=2.33523e17, k_base=0.0972077, n_base=0.425362, mult_xm=mult_xm, mult_n=mult_n)

def rho_aug_ES1_smooth(N, mult_xm=1.0, mult_n=1.0):
    return rho_weibull(N, N_sat_base=4.686e17, k_base=0.140, n_base=0.75, mult_xm=mult_xm, mult_n=mult_n)

def rho_spon_ES1_smooth(N, mult_xm=1.0, mult_n=1.0):
    return rho_weibull(N, N_sat_base=4.82866e17, k_base=0.146837, n_base=0.510774, mult_xm=mult_xm, mult_n=mult_n)

def rho_aug_ES2_smooth(N, mult_xm=1.0, mult_n=1.0):
    return rho_weibull(N, N_sat_base=7.069054e17, k_base=0.07177555, n_base=0.6097, mult_xm=mult_xm, mult_n=mult_n)

def rho_spon_ES2_smooth(N, mult_xm=1.0, mult_n=1.0):
    return rho_weibull(N, N_sat_base=7.37623e17, k_base=0.204794, n_base=0.567465, mult_xm=mult_xm, mult_n=mult_n)

def main():
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Carrier density grids for each level
    N_gs = np.linspace(0, 3.0e17, 500)
    N_es1 = np.linspace(0, 5.5e17, 500)
    N_es2 = np.linspace(0, 8.5e17, 500)
    
    levels_data = [
        ("GS Auger Recombination", N_gs, rho_aug_GS_smooth, axes[0, 0]),
        ("GS Spontaneous Emission", N_gs, rho_spon_GS_smooth, axes[0, 1]),
        ("ES1 Auger Recombination", N_es1, rho_aug_ES1_smooth, axes[1, 0]),
        ("ES1 Spontaneous Emission", N_es1, rho_spon_ES1_smooth, axes[1, 1]),
        ("ES2 Auger Recombination", N_es2, rho_aug_ES2_smooth, axes[2, 0]),
        ("ES2 Spontaneous Emission", N_es2, rho_spon_ES2_smooth, axes[2, 1]),
    ]
    
    for title, N_grid, func, ax in levels_data:
        # Baseline
        ax.plot(N_grid / 1e17, func(N_grid, 1.0, 1.0), 'k-', linewidth=2.5, label='Baseline (1.0x)')
        
        # Scaling xm (ceiling / threshold shift)
        ax.plot(N_grid / 1e17, func(N_grid, 0.5, 1.0), 'b--', alpha=0.8, label='xm = 0.5x (earlier saturation)')
        ax.plot(N_grid / 1e17, func(N_grid, 2.0, 1.0), 'b:', alpha=0.8, label='xm = 2.0x (later saturation)')
        
        # Scaling exponent (steepness change)
        ax.plot(N_grid / 1e17, func(N_grid, 1.0, 0.5), 'r--', alpha=0.8, label='exponent = 0.5x (gentler rise)')
        ax.plot(N_grid / 1e17, func(N_grid, 1.0, 2.0), 'r:', alpha=0.8, label='exponent = 2.0x (steeper rise)')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(r"Carrier Density $N$ ($10^{17}$ cm$^{-3}$)")
        ax.set_ylabel(r"Pauli Blocking Factor $\rho$")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=8, loc='best')

    plt.suptitle("Pauli Blocking Factor (rho) Smooth Weibull Saturation Models Across [0.5x, 2.0x] Range", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("rho_variations_comparison.png", dpi=300)
    print("Saved refined rho_variations_comparison.png")

if __name__ == "__main__":
    main()

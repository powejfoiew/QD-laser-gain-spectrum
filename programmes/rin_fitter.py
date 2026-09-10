#!/usr/bin/env python3
"""
fit_rin_extract_damping.py - Corrected version

Fits the RIN transfer function to extract f_R and gamma for a QD DFB laser,
then plots gamma vs f_R^2 to extract the K-factor.

Uses a simplified RIN model:
    RIN(f) = A * (f_R^4 + f^2/tau_n^2) / [(f^2 - f_R^2)^2 + (gamma*f/(2*pi))^2] + noise_floor

This is equivalent to the standard form from rate equations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


# =============================================================================
# Data Loading - Fixed for the actual file format
# =============================================================================

def load_rin_csv(filepath):
    """
    Load RIN data from the custom CSV format where data alternates:
    Frequency_Hz: <val>\n RIN_dB_Hz: <val> Frequency_Hz: <val>\n ...
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    parts = content.split()
    
    freq_list = []
    rin_list = []
    
    i = 0
    while i < len(parts):
        if parts[i] == "Frequency_Hz:":
            freq_list.append(float(parts[i + 1]))
            i += 2
        elif parts[i] == "RIN_dB_Hz:":
            rin_list.append(float(parts[i + 1]))
            i += 2
        else:
            i += 1
    
    # Ensure equal lengths
    min_len = min(len(freq_list), len(rin_list))
    return np.array(freq_list[:min_len]), np.array(rin_list[:min_len])


# =============================================================================
# RIN Model Function - Simplified standard form
# =============================================================================

def rin_model_dB(f, f_R, gamma, A_dB, noise_floor_dB):
    """
    Simplified RIN model in dB/Hz.
    
    RIN(f) = A * f_R^4 / [(f^2 - f_R^2)^2 + (gamma/(2*pi))^2 * f^2] + noise_floor
    
    This is the standard second-order transfer function for RIN.
    f_R in Hz, gamma in Hz (angular damping / 2pi effectively).
    A_dB: amplitude scaling in dB
    noise_floor_dB: shot noise floor in dB/Hz
    """
    A = 10.0**(A_dB / 10.0)
    noise_floor = 10.0**(noise_floor_dB / 10.0)
    
    # Standard RIN transfer function (simplified, ignoring carrier noise term for robustness)
    numerator = f_R**4
    denominator = (f**2 - f_R**2)**2 + (gamma / (2.0 * np.pi))**2 * f**2
    
    rin_linear = A * numerator / denominator + noise_floor
    
    return 10.0 * np.log10(np.maximum(rin_linear, 1e-30))


def rin_model_dB_full(f, f_R, gamma, A_dB, tau_n, noise_floor_dB):
    """
    Full RIN model with carrier noise term.
    
    RIN(f) = A * [f_R^4 + (f/(2*pi*tau_n))^2] / [(f^2 - f_R^2)^2 + (gamma/(2*pi))^2 * f^2] + noise_floor
    """
    A = 10.0**(A_dB / 10.0)
    noise_floor = 10.0**(noise_floor_dB / 10.0)
    
    numerator = f_R**4 + (f / (2.0 * np.pi * tau_n))**2
    denominator = (f**2 - f_R**2)**2 + (gamma / (2.0 * np.pi))**2 * f**2
    
    rin_linear = A * numerator / denominator + noise_floor
    
    return 10.0 * np.log10(np.maximum(rin_linear, 1e-30))


# =============================================================================
# Main Script
# =============================================================================

def main():
    # Define currents and file paths
    currents = [10, 15, 25, 30, 35, 40, 45, 55]
    csv_dir = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Damping factor\rin_plots"
    
    # Frequency range for fitting
    f_fit_min = 5e7    # 50 MHz (avoid very low freq noise)
    f_fit_max = 1.5e10  # 15 GHz
    
    # Storage for extracted parameters
    fit_results = {}
    
    # Color map
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(currents)))
    
    # =========================================================================
    # FIGURE 1: RIN spectra with fits
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    for idx, current in enumerate(currents):
        csv_file = os.path.join(csv_dir, f"rin_{current}mA.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Skipping {current} mA.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Fitting RIN for {current} mA...")
        print(f"{'='*60}")
        
        # Load data using the CORRECT parser
        freq, rin_dB = load_rin_csv(csv_file)
        
        # Sort by frequency
        sort_idx = np.argsort(freq)
        freq = freq[sort_idx]
        rin_dB = rin_dB[sort_idx]
        
        # Filter to fitting range
        mask = (freq >= f_fit_min) & (freq <= f_fit_max)
        freq_fit = freq[mask]
        rin_dB_fit = rin_dB[mask]
        
        if len(freq_fit) < 10:
            print(f"  Insufficient data points in fitting range. Skipping.")
            continue
        
        # =====================================================================
        # Estimate initial parameters from data
        # =====================================================================
        
        # Noise floor: use the median of high-frequency points
        high_freq_mask = freq_fit > 5e9
        if np.sum(high_freq_mask) > 5:
            noise_floor_guess_dB = np.median(rin_dB_fit[high_freq_mask])
        else:
            noise_floor_guess_dB = np.min(rin_dB_fit)
        
        # Look for relaxation oscillation peak
        # Smooth the data first to find the peak
        from scipy.ndimage import uniform_filter1d
        rin_smooth = uniform_filter1d(rin_dB_fit, size=min(20, len(rin_dB_fit)//5))
        
        # The peak should be above the noise floor
        peak_idx = np.argmax(rin_smooth)
        f_R_guess = freq_fit[peak_idx]
        peak_height_dB = rin_smooth[peak_idx]
        
        # If no clear peak (overdamped), use a lower frequency guess
        if peak_height_dB < noise_floor_guess_dB + 3:
            # Overdamped case - f_R might be below our measurement range
            # or the peak is very broad
            f_R_guess = 0.5e9  # Start with 500 MHz
        
        # Gamma guess: for QD lasers, damping is typically large
        # A rough estimate: if the peak is broad, gamma > 2*pi*f_R
        gamma_guess = 2.0 * np.pi * f_R_guess * 2  # Start overdamped
        
        # Amplitude guess from peak height
        A_guess_dB = peak_height_dB + 10  # Rough starting point
        
        print(f"  Initial guesses: f_R={f_R_guess/1e9:.2f} GHz, "
              f"gamma={gamma_guess/(2*np.pi*1e9):.2f} GHz (angular), "
              f"floor={noise_floor_guess_dB:.1f} dB/Hz")
        
        # =====================================================================
        # Fit using the simplified model (in dB space)
        # =====================================================================
        try:
            # [f_R, gamma, A_dB, noise_floor_dB]
            p0 = [f_R_guess, gamma_guess, A_guess_dB, noise_floor_guess_dB]
            
            # Bounds
            bounds_lower = [1e7, 1e8, -200, -190]
            bounds_upper = [1.5e10, 1e12, -50, -130]
            
            popt, pcov = curve_fit(
                rin_model_dB,
                freq_fit,
                rin_dB_fit,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=100000,
                method='trf',
                loss='soft_l1',  # Robust loss to handle noisy data
                f_scale=3.0      # Scale for robust loss
            )
            
            f_R_fit, gamma_fit, A_fit_dB, noise_floor_fit_dB = popt
            perr = np.sqrt(np.diag(pcov))
            
            # Convert to GHz for display
            f_R_GHz = f_R_fit / 1e9
            gamma_GHz = gamma_fit / (2.0 * np.pi * 1e9)  # Convert angular to frequency
            
            print(f"  f_R = {f_R_GHz:.3f} GHz")
            print(f"  gamma/(2pi) = {gamma_GHz:.3f} GHz")
            print(f"  A = {A_fit_dB:.1f} dB")
            print(f"  Noise floor = {noise_floor_fit_dB:.1f} dB/Hz")
            
            # Store results (gamma in GHz as gamma/(2*pi))
            fit_results[current] = {
                'f_R': f_R_fit,
                'gamma': gamma_fit,
                'f_R_GHz': f_R_GHz,
                'gamma_GHz': gamma_GHz,
                'popt': popt,
                'perr': perr
            }
            
            # Generate fit curve
            f_plot = np.logspace(np.log10(f_fit_min), np.log10(f_fit_max), 500)
            rin_fit_dB_plot = rin_model_dB(f_plot, *popt)
            
            # Plot data
            ax1.semilogx(freq, rin_dB, '.', color=colors[idx], 
                        markersize=2, alpha=0.3)
            # Plot fit
            ax1.semilogx(f_plot, rin_fit_dB_plot, '-', color=colors[idx],
                        linewidth=2, label=f"{current} mA ($f_R$={f_R_GHz:.2f} GHz)")
            
        except Exception as e:
            print(f"  Fitting FAILED for {current} mA: {e}")
            ax1.semilogx(freq, rin_dB, '.', color=colors[idx],
                        markersize=3, alpha=0.5, label=f"{current} mA (no fit)")
    
    # Style Figure 1
    ax1.set_xlim(5e7, 2e10)
    ax1.set_ylim(-185, -125)
    ax1.set_xlabel("Frequency (Hz)", fontsize=14)
    ax1.set_ylabel("RIN (dB/Hz)", fontsize=14)
    ax1.set_title("QD DFB Laser RIN Spectra with Fitted Curves", fontsize=16)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.grid(True, which="both", linestyle=":", alpha=0.3)
    plt.tight_layout()
    fig1.savefig("rin_spectra_with_fits.png", dpi=300, bbox_inches="tight")
    
    # =========================================================================
    # FIGURE 2: gamma vs f_R^2 (K-factor extraction)
    # =========================================================================
    if len(fit_results) >= 2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        f_R_squared = []
        gamma_values = []
        current_labels = []
        
        for current in sorted(fit_results.keys()):
            res = fit_results[current]
            f_R_sq_GHz2 = res['f_R_GHz'] ** 2
            gamma_GHz = res['gamma_GHz']
            
            f_R_squared.append(f_R_sq_GHz2)
            gamma_values.append(gamma_GHz)
            current_labels.append(current)
        
        f_R_squared = np.array(f_R_squared)
        gamma_values = np.array(gamma_values)
        
        # Plot data points
        ax2.plot(f_R_squared, gamma_values, 'ko', markersize=8, zorder=5)
        
        # Annotate
        for i, current in enumerate(current_labels):
            ax2.annotate(f"{current} mA", 
                        (f_R_squared[i], gamma_values[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        
        # Linear fit: gamma = K * f_R^2 + gamma_0
        coeffs = np.polyfit(f_R_squared, gamma_values, 1)
        K_factor = coeffs[0]  # ns
        gamma_0 = coeffs[1]   # GHz
        
        f_R_sq_range = np.linspace(0, max(f_R_squared) * 1.2, 100)
        gamma_fit_line = np.polyval(coeffs, f_R_sq_range)
        
        ax2.plot(f_R_sq_range, gamma_fit_line, 'r-', linewidth=2,
                label=f"K = {K_factor:.2f} ns")
        
        print(f"\n{'='*60}")
        print(f"K-factor = {K_factor:.4f} ns")
        print(f"gamma_0 = {gamma_0:.4f} GHz")
        print(f"{'='*60}")
        
        ax2.set_xlabel(r"$f_{RO}^2$ (GHz$^2$)", fontsize=14)
        ax2.set_ylabel(r"$\gamma / 2\pi$ (GHz)", fontsize=14)
        ax2.set_title(r"Damping Factor vs $f_{RO}^2$ — K-factor Extraction", fontsize=16)
        ax2.legend(loc="upper left", fontsize=12)
        ax2.grid(True, linestyle=":", alpha=0.3)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        plt.tight_layout()
        fig2.savefig("gamma_vs_fR_squared.png", dpi=300, bbox_inches="tight")
    
    plt.show()


if __name__ == "__main__":
    main()
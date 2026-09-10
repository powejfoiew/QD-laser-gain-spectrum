import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def main():
    # Constants
    c = 299792458
    lambda_0 = 1.294e-6
    f_0 = c / lambda_0
    f_0_ghz = f_0 * 1e-9
    
    filename = "raw_phase_qfactor.ascii"
    print("Loading data...")
    data = np.loadtxt(filename, skiprows=2)
    t = data[:, 0]       # time in ns
    abs_A = data[:, 1]   # amplitude
    arg_A = data[:, 2]   # phase in radians
    
    # Reconstruct complex field
    A = abs_A * np.exp(1j * arg_A)
    
    dt = t[1] - t[0]
    N = len(t)
    freqs = np.fft.fftfreq(N, dt)  # GHz (since dt is in ns)
    A_fft = np.fft.fft(A)
    
    # Only keep positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    spectrum = np.abs(A_fft[pos_mask])
    
    # Let's find the main resonance peaks in the spectrum
    # We look in the range 10 to 30 GHz where the spectrum is strong
    range_mask = (freqs_pos >= 10) & (freqs_pos <= 30)
    freqs_range = freqs_pos[range_mask]
    spectrum_range = spectrum[range_mask]
    
    # Detect peaks in the spectrum
    peaks_in_range, _ = find_peaks(spectrum_range, distance=15, prominence=0.05 * np.max(spectrum_range))
    peak_freqs_ghz = freqs_range[peaks_in_range]
    peak_mags = spectrum_range[peaks_in_range]
    
    print(f"\nDetected {len(peak_freqs_ghz)} modes in the spectrum between 10 and 30 GHz:")
    for idx, (f_mode_offset, mag) in enumerate(zip(peak_freqs_ghz, peak_mags), 1):
        f_mode_absolute_thz = (f_0_ghz + f_mode_offset) * 1e-3
        print(f"  Mode {idx}: offset = {f_mode_offset:.4f} GHz, abs freq = {f_mode_absolute_thz:.6f} THz, mag = {mag:.3e}")
        
    # We will pick the strongest mode
    strongest_idx = np.argmax(peak_mags)
    f_mode_offset = peak_freqs_ghz[strongest_idx]
    print(f"\nStrongest mode is Mode {strongest_idx + 1} at offset = {f_mode_offset:.4f} GHz")
    
    # Define a Gaussian filter in frequency domain
    # H(f) = exp(-(f - f_mode)^2 / (2 * sigma^2))
    # Let's choose sigma such that we isolate this single mode.
    # The mode spacing is ~1.15 GHz. A sigma of 0.3 GHz will decay to exp(-1.15^2 / (2 * 0.3^2)) = exp(-7.3) = 6e-4 at the adjacent modes.
    sigma = 0.3  # GHz
    
    # Filter the FFT
    # Note: We must apply it to both positive and negative frequencies, but since the signal is complex,
    # the negative frequencies do not have the same modes (A(t) is complex, so FFT(A) is not symmetric).
    # We just apply the filter centered at f_mode_offset on the entire FFT.
    H = np.exp(-(freqs - f_mode_offset)**2 / (2 * sigma**2))
    A_fft_filtered = A_fft * H
    
    # IFFT to get the filtered time-domain signal
    A_filtered = np.fft.ifft(A_fft_filtered)
    abs_A_filtered = np.abs(A_filtered)
    
    # Let's plot the original and filtered spectrum, and the filtered time domain
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Spectrum plot
    axes[0].plot(freqs_pos, spectrum, label='Original Spectrum')
    axes[0].plot(freqs, np.abs(A_fft_filtered), label='Filtered Spectrum', color='red', alpha=0.7)
    axes[0].set_xlabel('Frequency (GHz)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_xlim(f_mode_offset - 5, f_mode_offset + 5)
    axes[0].set_title('Gaussian Filter on Spectrum')
    axes[0].grid(True)
    axes[0].legend()
    
    # Time domain plot (Linear)
    axes[1].plot(t, abs_A, label='Original |A(t)|', alpha=0.5)
    axes[1].plot(t, abs_A_filtered, label='Filtered |A(t)|', color='red')
    axes[1].set_xlabel('Time (ns)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Filtered Time Domain (Linear Scale)')
    axes[1].grid(True)
    axes[1].legend()
    
    # Time domain plot (Log)
    axes[2].semilogy(t, abs_A, label='Original |A(t)|', alpha=0.3)
    axes[2].semilogy(t, abs_A_filtered, label='Filtered |A(t)|', color='red')
    axes[2].set_xlabel('Time (ns)')
    axes[2].set_ylabel('Amplitude (log10)')
    axes[2].set_title('Filtered Time Domain (Log Scale)')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('gaussian_filter_test.png', dpi=150)
    print("Plots saved to gaussian_filter_test.png")
    
    # Let's fit the decay slope of the filtered signal
    # We fit in the 40% to 60% range of the simulation time (t from 8 ns to 12 ns)
    t_start, t_end = 8.0, 12.0
    fit_mask = (t >= t_start) & (t <= t_end)
    t_fit = t[fit_mask]
    log10_abs_A_fit = np.log10(abs_A_filtered[fit_mask])
    
    m, c_fit = np.polyfit(t_fit, log10_abs_A_fit, 1)
    
    # Resonant frequency for this mode
    f_R_ghz = f_0_ghz + f_mode_offset
    log10_e = np.log10(np.e)
    Q = -np.pi * f_R_ghz * log10_e / m
    print(f"\nFiltered Mode Fit (t from {t_start} to {t_end} ns):")
    print(f"  Slope (m): {m:.6f} ns^-1")
    print(f"  Resonant Frequency f_R: {f_R_ghz:.6f} GHz ({f_R_ghz * 1e-3:.6f} THz)")
    print(f"  Calculated Q-factor: {Q:.2f}")

if __name__ == '__main__':
    main()

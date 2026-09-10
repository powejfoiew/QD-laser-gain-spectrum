import numpy as np
import matplotlib.pyplot as plt

def main():
    filename = "raw_phase_qfactor.ascii"
    print("Loading data...")
    
    # Skip the first 2 lines (header lines)
    # Line 1: 461823 // npoints
    # Line 2: //      t[ns]        abs(A)   arg(A)[rad]
    data = np.loadtxt(filename, skiprows=2)
    t = data[:, 0]       # time in ns
    abs_A = data[:, 1]   # amplitude
    arg_A = data[:, 2]   # phase in radians
    
    print(f"Loaded {len(t)} points.")
    print(f"Time range: {t[0]} ns to {t[-1]} ns")
    print(f"Max abs(A): {np.max(abs_A)}")
    
    # Reconstruct the complex field A(t)
    A = abs_A * np.exp(1j * arg_A)
    
    # Calculate sample spacing dt
    dt = t[1] - t[0]
    print(f"Sample spacing dt: {dt} ns")
    fs = 1.0 / dt  # sampling frequency in GHz
    print(f"Sampling frequency fs: {fs} GHz (or {fs*1e9} Hz)")
    
    # FFT
    N = len(t)
    freqs = np.fft.fftfreq(N, dt)  # in GHz
    A_fft = np.fft.fft(A)
    
    # Only keep positive frequencies for plotting
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    spectrum = np.abs(A_fft[pos_mask])
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time domain plot
    axes[0].plot(t, abs_A, label='|A(t)|')
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain Signal')
    axes[0].grid(True)
    axes[0].legend()
    
    # Frequency domain plot
    axes[1].plot(freqs_pos, spectrum, label='Spectrum')
    axes[1].set_xlabel('Frequency (GHz)')
    axes[1].set_ylabel('Magnitude (a.u.)')
    axes[1].set_title('Frequency Domain Spectrum')
    axes[1].grid(True)
    axes[1].legend()
    
    # Find peak frequency
    peak_idx = np.argmax(spectrum)
    peak_freq_ghz = freqs_pos[peak_idx]
    print(f"Peak frequency: {peak_freq_ghz} GHz")
    print(f"Peak frequency in Hz: {peak_freq_ghz * 1e9} Hz")
    print(f"Peak frequency in THz: {peak_freq_ghz * 1e-3} THz")
    print(f"Peak wavelength (assuming c = 299792458 m/s): {299792458 / (peak_freq_ghz * 1e9) * 1e9:.2f} nm")
    
    plt.tight_layout()
    plt.savefig('inspection_plots.png', dpi=150)
    print("Plots saved to inspection_plots.png")

if __name__ == '__main__':
    main()

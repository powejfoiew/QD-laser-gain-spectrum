import numpy as np
import matplotlib.pyplot as plt

def main():
    filename = "raw_phase_qfactor.ascii"
    data = np.loadtxt(filename, skiprows=2)
    t = data[:, 0]       # time in ns
    abs_A = data[:, 1]   # amplitude
    
    # We only take parts where amplitude is non-zero to avoid log10(0)
    valid_mask = abs_A > 1e-12
    t_v = t[valid_mask]
    abs_A_v = abs_A[valid_mask]
    
    # Find the peak of the signal
    peak_idx = np.argmax(abs_A_v)
    t_peak = t_v[peak_idx]
    val_peak = abs_A_v[peak_idx]
    print(f"Peak amplitude {val_peak} occurs at t = {t_peak} ns (index {peak_idx} of valid points)")
    
    # Let's inspect the decay after the peak
    t_decay = t_v[peak_idx:]
    abs_A_decay = abs_A_v[peak_idx:]
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Linear scale zoom-in
    axes[0].plot(t_v, abs_A_v, label='|A(t)|')
    axes[0].axvline(t_peak, color='r', linestyle='--', label=f'Peak: {t_peak:.3f} ns')
    axes[0].set_xlabel('Time (ns)')
    axes[0].set_ylabel('Amplitude (linear)')
    axes[0].set_title('Time Domain - Linear Scale')
    axes[0].set_xlim(0, 5)  # Zoom in on first 5 ns
    axes[0].grid(True)
    axes[0].legend()
    
    # Log scale zoom-in
    axes[1].semilogy(t_v, abs_A_v, label='|A(t)|')
    axes[1].axvline(t_peak, color='r', linestyle='--', label=f'Peak: {t_peak:.3f} ns')
    axes[1].set_xlabel('Time (ns)')
    axes[1].set_ylabel('Amplitude (log10)')
    axes[1].set_title('Time Domain - Logarithmic Scale')
    axes[1].grid(True)
    axes[1].legend()
    
    # FFT Zoom-in
    dt = t[1] - t[0]
    freqs = np.fft.fftfreq(len(t), dt)
    A = abs_A * np.exp(1j * data[:, 2])
    A_fft = np.fft.fft(A)
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    spectrum = np.abs(A_fft[pos_mask])
    
    axes[2].plot(freqs_pos, spectrum)
    axes[2].set_xlabel('Frequency (GHz)')
    axes[2].set_ylabel('Magnitude (a.u.)')
    axes[2].set_title('Frequency Domain Spectrum (Zoomed)')
    axes[2].set_xlim(0, 100)  # Zoom in on lower frequencies where peak is
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('zoom_plots.png', dpi=150)
    print("Plots saved to zoom_plots.png")

if __name__ == '__main__':
    main()

import numpy as np

def main():
    filename = "raw_phase_qfactor.ascii"
    print("Loading data...")
    data = np.loadtxt(filename, skiprows=2)
    t = data[:, 0]       # time in ns
    abs_A = data[:, 1]   # amplitude
    arg_A = data[:, 2]   # phase in radians
    
    # Let's find the peak region (e.g. within the first pulse where amplitude is high)
    # peak is at t around 0.866 ns
    mask = (t > 0.85) & (t < 0.88)
    t_pulse = t[mask]
    abs_A_pulse = abs_A[mask]
    arg_A_pulse = arg_A[mask]
    
    print(f"Number of points in first pulse: {len(t_pulse)}")
    
    # Unwrap the phase
    unwrapped_phase = np.unwrap(arg_A)
    unwrapped_phase_pulse = unwrapped_phase[mask]
    
    # Calculate frequency f = 1/(2*pi) * d(phase)/dt
    dt = t[1] - t[0]  # in ns
    # d(phase)/dt in rad/ns
    dphase_dt = np.diff(unwrapped_phase) / dt
    freqs_ghz = dphase_dt / (2 * np.pi)
    
    # Look at the frequency during the pulses where SNR is high
    print("\nAnalyzing frequency during the pulses:")
    for i in range(1, 6):
        t_center = 0.866 + (i - 1) * 0.866
        pulse_mask = (t > t_center - 0.01) & (t < t_center + 0.01)
        mean_freq = np.mean(freqs_ghz[pulse_mask[:-1]])
        std_freq = np.std(freqs_ghz[pulse_mask[:-1]])
        print(f"Pulse {i} (around t={t_center:.3f} ns): Mean Freq = {mean_freq:.6f} GHz, Std = {std_freq:.6e} GHz")

if __name__ == '__main__':
    main()

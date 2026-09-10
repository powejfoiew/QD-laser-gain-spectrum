import numpy as np

def main():
    filename = "raw_phase_qfactor.ascii"
    data = np.loadtxt(filename, skiprows=2)
    t = data[:, 0]       # time in ns
    abs_A = data[:, 1]   # amplitude
    arg_A = data[:, 2]   # phase in radians
    
    # Unwrap phase
    unwrapped_phase = np.unwrap(arg_A)
    
    # Let's check the phase values at the peaks of the pulses
    # We find the peak indices in the data
    # The peaks occur roughly at intervals of 0.866 ns
    # Let's find local maxima of abs_A
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(abs_A, distance=10000, prominence=1e-11)
    
    print("Detected pulse peaks:")
    for idx in peaks:
        print(f"t = {t[idx]:.6f} ns, abs(A) = {abs_A[idx]:.3e}, unwrapped phase = {unwrapped_phase[idx]:.6f} rad")
    
    # Let's fit a line to the phase of these peaks
    if len(peaks) > 1:
        t_peaks = t[peaks]
        phase_peaks = unwrapped_phase[peaks]
        slope, intercept = np.polyfit(t_peaks, phase_peaks, 1)
        # slope is in rad/ns
        # freq = slope / (2 * pi) in GHz
        freq_ghz = slope / (2 * np.pi)
        print(f"\nLinear fit of phase at peak locations:")
        print(f"  Slope: {slope:.6f} rad/ns")
        print(f"  Carrier Frequency: {freq_ghz:.6f} GHz (or {freq_ghz * 1e9:.6e} Hz)")
        
        # Let's also fit a line to all points where the amplitude is high
        high_mask = abs_A > 1e-7
        slope_all, intercept_all = np.polyfit(t[high_mask], unwrapped_phase[high_mask], 1)
        freq_all_ghz = slope_all / (2 * np.pi)
        print(f"\nLinear fit of phase for all points with |A| > 1e-7:")
        print(f"  Slope: {slope_all:.6f} rad/ns")
        print(f"  Carrier Frequency: {freq_all_ghz:.6f} GHz (or {freq_all_ghz * 1e9:.6e} Hz)")

if __name__ == '__main__':
    main()

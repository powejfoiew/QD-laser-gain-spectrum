import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def main():
    # Constants
    c = 299792458  # speed of light in m/s
    lambda_0 = 1.294e-6  # central wavelength in m
    f_0 = c / lambda_0  # central frequency in Hz
    f_0_ghz = f_0 * 1e-9  # central frequency in GHz
    
    print(f"Central Wavelength: {lambda_0 * 1e6:.4f} um")
    print(f"Central Frequency (f_0): {f_0:.6e} Hz ({f_0_ghz:.6f} GHz)")
    
    filename = "raw_phase_qfactor.ascii"
    print(f"Loading data from {filename}...")
    data = np.loadtxt(filename, skiprows=2)
    t = data[:, 0]       # time in ns
    abs_A = data[:, 1]   # amplitude
    arg_A = data[:, 2]   # phase in radians
    
    # 1. Detect the peaks of the pulses
    # We use find_peaks on abs_A with a minimum distance and prominence
    # distance=10000 points corresponds to ~0.43 ns which is half the roundtrip
    peaks, _ = find_peaks(abs_A, distance=10000, prominence=1e-11)
    
    t_peaks = t[peaks]
    abs_A_peaks = abs_A[peaks]
    
    print(f"\nDetected {len(peaks)} peaks:")
    for idx, (tp, ap) in enumerate(zip(t_peaks, abs_A_peaks), 1):
        print(f"  Peak {idx}: t = {tp:.6f} ns, amplitude = {ap:.3e}")
        
    # 2. Fit decay of peak amplitudes: log10(abs_A) = m * t + c
    log10_abs_A_peaks = np.log10(abs_A_peaks)
    
    # Fit line using numpy.polyfit
    m, c_fit = np.polyfit(t_peaks, log10_abs_A_peaks, 1)
    
    # Calculate R-squared to check the fit quality
    residuals = log10_abs_A_peaks - (m * t_peaks + c_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log10_abs_A_peaks - np.mean(log10_abs_A_peaks))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nAmplitude decay fit (log10(A) vs t in ns):")
    print(f"  Slope (m): {m:.6f} ns^-1")
    print(f"  Intercept: {c_fit:.6f}")
    print(f"  R^2: {r_squared:.8f}")
    
    # 3. Determine the carrier frequency of the envelope A(t)
    unwrapped_phase = np.unwrap(arg_A)
    phase_peaks = unwrapped_phase[peaks]
    
    # Fit phase at peak locations to get carrier frequency offset
    m_phase, c_phase = np.polyfit(t_peaks, phase_peaks, 1)
    # slope is d(phase)/dt in rad/ns
    # f_envelope = slope / (2 * pi) in GHz
    f_envelope_ghz = m_phase / (2 * np.pi)
    f_envelope_hz = f_envelope_ghz * 1e9
    
    print(f"\nPhase slope at peak locations:")
    print(f"  Slope: {m_phase:.6f} rad/ns")
    print(f"  Envelope Carrier Frequency (f_envelope): {f_envelope_ghz:.6f} GHz ({f_envelope_hz:.6e} Hz)")
    
    # 4. Calculate resonant frequency f_R
    f_R_ghz = f_0_ghz + f_envelope_ghz
    f_R_hz = f_0 + f_envelope_hz
    print(f"\nResonant Frequency (f_R = f_0 + f_envelope):")
    print(f"  f_R = {f_R_hz:.6e} Hz ({f_R_ghz:.6f} GHz)")
    
    # 5. Calculate Q-factor using Lumerical formula
    # Q = -pi * f_R * log10(e) / m
    # where f_R is in Hz, and m is in s^-1
    # Note: f_R in GHz and m in ns^-1 cancels out the 10^9 factor:
    # Q = -pi * (f_R_ghz * 1e9) * log10(e) / (m * 1e9) = -pi * f_R_ghz * log10(e) / m
    log10_e = np.log10(np.e)
    Q = -np.pi * f_R_ghz * log10_e / m
    
    # Also calculate Q using f_0 just for comparison
    Q_f0 = -np.pi * f_0_ghz * log10_e / m
    
    print(f"\nQuality Factor (Q) Calculation Results:")
    print(f"  Using f_R (including phase carrier): Q = {Q:.2f}")
    print(f"  Using f_0 (central frequency only): Q = {Q_f0:.2f}")
    print(f"  Relative difference: {abs(Q - Q_f0) / Q * 100:.6f} %")
    
    # Let's calculate the decay time constant tau = Q / (2 * pi * f_R)
    tau_ns = Q / (2 * np.pi * f_R_ghz)
    print(f"  Decay time constant (tau): {tau_ns:.6f} ns")
    
    # 6. Plotting the fit
    plt.figure(figsize=(8, 6))
    plt.plot(t_peaks, log10_abs_A_peaks, 'o', label='Detected Peaks', color='blue')
    t_fit = np.linspace(t_peaks[0] - 0.2, t_peaks[-1] + 0.2, 100)
    plt.plot(t_fit, m * t_fit + c_fit, '-', label=f'Linear Fit (R² = {r_squared:.6f})', color='red')
    plt.xlabel('Time (ns)')
    plt.ylabel('log10(Amplitude)')
    plt.title('DFB Cavity Decay Rate Fit')
    plt.grid(True)
    plt.legend()
    
    # Text box with results
    textstr = '\n'.join((
        f'Resonant freq f_R: {f_R_hz*1e-12:.6f} THz',
        f'Decay slope m: {m:.6f} ns⁻¹',
        f'Calculated Q: {Q:.1f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.5, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.savefig('qfactor_decay_fit.png', dpi=150)
    print("\nSaved fit plot to qfactor_decay_fit.png")

if __name__ == '__main__':
    main()

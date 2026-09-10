import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline

# ─── Constants ────────────────────────────────────────────────────────────────
c       = 299792458        # m/s
lam0    = 1.294e-6         # m  (central wavelength)
f0      = c / lam0         # Hz
f0_ghz  = f0 * 1e-9       # GHz
log10_e = np.log10(np.e)

# ─── Load data ────────────────────────────────────────────────────────────────
filename = "laser_dfb_pahses.ascii"
print("Loading data...")
data   = np.loadtxt(filename, skiprows=2)
t      = data[:, 0]        # ns
abs_A  = data[:, 1]        # amplitude
arg_A  = data[:, 2]        # phase (rad)

N  = len(t)
dt = t[1] - t[0]           # ns
print(f"Points : {N}")
print(f"dt     : {dt:.6e} ns")
print(f"t range: {t[0]:.6f} ns  to  {t[-1]:.2f} ns")
print(f"max|A| : {np.max(abs_A):.4e}")

# ─── Reconstruct complex field ────────────────────────────────────────────────
A = abs_A * np.exp(1j * arg_A)

# ─── FFT ──────────────────────────────────────────────────────────────────────
freqs   = np.fft.fftfreq(N, dt)   # GHz
A_fft   = np.fft.fft(A)
spectrum = np.abs(A_fft)

pos_mask    = freqs > 0
freqs_pos   = freqs[pos_mask]
spectrum_pos= spectrum[pos_mask]

# ─── Time-domain quick look ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(t, abs_A)
axes[0].set_xlabel('Time (ns)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Time Domain – Linear Scale')
axes[0].grid(True)

valid = abs_A > 1e-20
if valid.any():
    axes[1].semilogy(t[valid], abs_A[valid])
    axes[1].set_xlabel('Time (ns)')
    axes[1].set_ylabel('Amplitude (log)')
    axes[1].set_title('Time Domain – Logarithmic Scale')
    axes[1].grid(True)

plt.tight_layout()
plt.savefig('laser_timedomain.png', dpi=150)
plt.close()
print("Saved laser_timedomain.png")

# ─── Frequency spectrum quick look ────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(freqs_pos, spectrum_pos)
axes[0].set_xlabel('Frequency Offset (GHz)')
axes[0].set_ylabel('Magnitude')
axes[0].set_title('Full Spectrum')
axes[0].grid(True)

axes[1].plot(freqs_pos, spectrum_pos)
axes[1].set_xlim(0, 100)
axes[1].set_xlabel('Frequency Offset (GHz)')
axes[1].set_ylabel('Magnitude')
axes[1].set_title('Spectrum – Zoomed 0–100 GHz')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('laser_spectrum.png', dpi=150)
plt.close()
print("Saved laser_spectrum.png")

# ─── Detect pulse peaks in time domain ───────────────────────────────────────
# min_distance in samples ~ half the expected roundtrip
min_dist = max(1, int(0.3 / dt))   # 0.3 ns guard
peaks_t, _ = find_peaks(abs_A, distance=min_dist, prominence=1e-12)

print(f"\nDetected {len(peaks_t)} pulse peaks in time domain:")
for i, idx in enumerate(peaks_t[:15]):
    print(f"  Peak {i+1}: t={t[idx]:.6f} ns, |A|={abs_A[idx]:.4e}")

# ─── Detect modes in frequency domain ────────────────────────────────────────
# search 0–100 GHz
fmask  = (freqs_pos >= 0) & (freqs_pos <= 100)
f_rng  = freqs_pos[fmask]
s_rng  = spectrum_pos[fmask]
peaks_f, _ = find_peaks(s_rng, distance=5, prominence=0.01*np.max(s_rng))
print(f"\nDetected {len(peaks_f)} spectral modes in 0–100 GHz range:")
for i, idx in enumerate(peaks_f[:20]):
    print(f"  Mode {i+1}: offset={f_rng[idx]:.4f} GHz, abs={f0_ghz+f_rng[idx]:.4f} GHz, mag={s_rng[idx]:.4e}")

plt.savefig('laser_spectrum.png', dpi=150)
plt.close()

print("\nDone with inspection phase.")

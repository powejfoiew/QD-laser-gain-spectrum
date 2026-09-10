import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline

# ── Constants ──────────────────────────────────────────────────────────────────
c        = 299792458
lam0     = 1.294e-6
f0       = c / lam0
f0_ghz   = f0 * 1e-9
log10_e  = np.log10(np.e)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
d_imp = np.loadtxt('power_out_dfb_impulse.ascii',     skiprows=2)
d_cal = np.loadtxt('power_out_dfb_calibration.ascii', skiprows=2)

t    = d_imp[:, 0]
dt   = t[1] - t[0]
N    = len(t)

A_imp  = d_imp[:, 1] + 1j * d_imp[:, 2]
A_cal  = d_cal[:, 1] + 1j * d_cal[:, 2]
A_pert = A_imp - A_cal
abs_p  = np.abs(A_pert)

print(f"Points : {N},  dt={dt:.4e} ns,  t_end={t[-1]:.2f} ns")
print(f"max|A_imp|  : {np.max(np.abs(A_imp)):.4e}")
print(f"max|A_cal|  : {np.max(np.abs(A_cal)):.4e}")
print(f"max|A_pert| : {np.max(abs_p):.4e}")

# ── Time-domain overview ───────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

axes[0].plot(t, np.abs(A_imp), label='Impulse |A|', alpha=0.7)
axes[0].plot(t, np.abs(A_cal), label='Calibration |A|', alpha=0.7)
axes[0].set_xlabel('Time (ns)'); axes[0].set_ylabel('Amplitude')
axes[0].set_title('Raw Fields'); axes[0].grid(True); axes[0].legend()

axes[1].plot(t, abs_p)
axes[1].set_xlabel('Time (ns)'); axes[1].set_ylabel('Amplitude')
axes[1].set_title('Perturbation |A_imp - A_cal| (Linear)'); axes[1].grid(True)

valid = abs_p > 1e-25
if valid.any():
    axes[2].semilogy(t[valid], abs_p[valid])
axes[2].set_xlabel('Time (ns)'); axes[2].set_ylabel('Amplitude (log)')
axes[2].set_title('Perturbation (Log Scale)'); axes[2].grid(True)

plt.tight_layout()
plt.savefig('laser_perturbation_timedomain.png', dpi=150)
plt.close()
print("Saved laser_perturbation_timedomain.png")

# ── FFT ────────────────────────────────────────────────────────────────────────
freqs   = np.fft.fftfreq(N, dt)
A_fft_p = np.fft.fft(A_pert)
spec_p  = np.abs(A_fft_p)

A_fft_i = np.fft.fft(A_imp)
spec_i  = np.abs(A_fft_i)

pos = freqs > 0

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(freqs[pos], spec_p[pos])
axes[0].set_xlabel('Frequency Offset (GHz)'); axes[0].set_ylabel('Magnitude')
axes[0].set_title('Perturbation Spectrum (Full)'); axes[0].grid(True)

axes[1].plot(freqs[pos], spec_p[pos])
axes[1].set_xlim(0, 100)
axes[1].set_xlabel('Frequency Offset (GHz)'); axes[1].set_ylabel('Magnitude')
axes[1].set_title('Perturbation Spectrum (0-100 GHz)'); axes[1].grid(True)

plt.tight_layout()
plt.savefig('laser_perturbation_spectrum.png', dpi=150)
plt.close()
print("Saved laser_perturbation_spectrum.png")

# ── Detect pulse peaks in perturbation time-domain ────────────────────────────
min_dist = max(1, int(0.3 / dt))
peaks_t, _ = find_peaks(abs_p, distance=min_dist, prominence=1e-15)
print(f"\nDetected {len(peaks_t)} perturbation peaks:")
for i, idx in enumerate(peaks_t[:15]):
    print(f"  Peak {i+1:2d}: t={t[idx]:.6f} ns  |A|={abs_p[idx]:.4e}")

# ── Pulse-peak decay Q-factor ──────────────────────────────────────────────────
if len(peaks_t) >= 3:
    t_pk   = t[peaks_t]
    a_pk   = abs_p[peaks_t]
    lg_pk  = np.log10(a_pk)

    m, c_fit = np.polyfit(t_pk, lg_pk, 1)
    res  = lg_pk - (m * t_pk + c_fit)
    r2   = 1 - np.sum(res**2) / np.sum((lg_pk - np.mean(lg_pk))**2)

    # Carrier offset from phase
    unwrapped   = np.unwrap(np.angle(A_pert))
    ph_pk       = unwrapped[peaks_t]
    m_ph, _     = np.polyfit(t_pk, ph_pk, 1)
    f_env_ghz   = m_ph / (2 * np.pi)
    f_R_ghz     = f0_ghz + f_env_ghz
    Q_pk        = -np.pi * f_R_ghz * log10_e / m

    print(f"\n== Pulse-Peak Decay Q-Factor ==")
    print(f"  Decay slope m   : {m:.6f} ns^-1")
    print(f"  R^2             : {r2:.8f}")
    print(f"  Carrier offset  : {f_env_ghz:.4f} GHz")
    print(f"  f_R             : {f_R_ghz:.4f} GHz  ({f_R_ghz*1e-3:.6f} THz)")
    print(f"  Q (pulse peaks) : {Q_pk:.2f}")

    plt.figure(figsize=(8, 6))
    plt.plot(t_pk, lg_pk, 'o', color='blue', label='Perturbation Peaks')
    t_fit = np.linspace(t_pk[0]-0.2, t_pk[-1]+0.2, 200)
    plt.plot(t_fit, m*t_fit+c_fit, '-', color='red',
             label=f'Linear Fit  R^2={r2:.6f}')
    plt.xlabel('Time (ns)'); plt.ylabel('log10(Amplitude)')
    plt.title('Lasing DFB Cavity Decay Fit (Perturbation)')
    plt.grid(True); plt.legend()
    txt = f'f_R = {f_R_ghz*1e-3:.6f} THz\nm = {m:.6f} ns^-1\nQ = {Q_pk:.1f}'
    plt.gca().text(0.6, 0.95, txt, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.savefig('laser_qfactor_decay_fit.png', dpi=150)
    plt.close()
    print("Saved laser_qfactor_decay_fit.png")

# ── Gaussian single-mode filter Q-factor ──────────────────────────────────────
# Find spectral modes in perturbation spectrum
range_mask = (freqs > 5) & (freqs < 80)
f_rng  = freqs[range_mask]
s_rng  = spec_p[range_mask]
peaks_f, _ = find_peaks(s_rng, distance=15,
                         prominence=0.05 * np.max(s_rng))

print(f"\nSpectral modes in perturbation (5-80 GHz): {len(peaks_f)}")
for i, idx in enumerate(peaks_f[:20]):
    print(f"  Mode {i+1:2d}: offset={f_rng[idx]:.4f} GHz  mag={s_rng[idx]:.4e}")

if len(peaks_f) >= 3:
    # Gaussian filter on strongest mode
    sigma     = 0.3   # GHz
    t_s, t_e  = 4.0, 20.0
    fit_mask  = (t >= t_s) & (t <= t_e)
    t_fit     = t[fit_mask]

    strongest = peaks_f[np.argmax(s_rng[peaks_f])]
    f_mode    = f_rng[strongest]

    H          = np.exp(-(freqs - f_mode)**2 / (2*sigma**2))
    A_filt     = np.fft.ifft(A_fft_p * H)
    abs_filt   = np.abs(A_filt)

    lg_fit = np.log10(abs_filt[fit_mask])
    m2, c2 = np.polyfit(t_fit, lg_fit, 1)
    res2   = lg_fit - (m2*t_fit + c2)
    r2_2   = 1 - np.sum(res2**2) / np.sum((lg_fit - np.mean(lg_fit))**2)

    f_R2_ghz = f0_ghz + f_mode
    Q_filt   = -np.pi * f_R2_ghz * log10_e / m2

    print(f"\n== Gaussian Filter Q-Factor (mode at {f_mode:.4f} GHz offset) ==")
    print(f"  Decay slope m   : {m2:.6f} ns^-1")
    print(f"  R^2             : {r2_2:.8f}")
    print(f"  f_R             : {f_R2_ghz:.4f} GHz  ({f_R2_ghz*1e-3:.6f} THz)")
    print(f"  Q (filtered)    : {Q_filt:.2f}")

    plt.figure(figsize=(8, 6))
    plt.semilogy(t, abs_filt, label='Filtered perturbation', color='red', alpha=0.7)
    plt.semilogy(t[valid], abs_p[valid], label='Full perturbation', alpha=0.3)
    plt.axvspan(t_s, t_e, alpha=0.1, color='green', label='Fit window')
    plt.xlabel('Time (ns)'); plt.ylabel('Amplitude (log)')
    plt.title(f'Gaussian-Filtered Mode Decay  (offset={f_mode:.2f} GHz)')
    plt.grid(True); plt.legend()
    plt.savefig('laser_gaussian_decay.png', dpi=150)
    plt.close()
    print("Saved laser_gaussian_decay.png")

# ── DBR envelope from spectrum ─────────────────────────────────────────────────
print("\n== DBR Grating Envelope (from impulse spectrum) ==")
env_mask = (freqs >= -150) & (freqs <= 150)
f_env    = freqs[env_mask]
s_env    = spec_i[env_mask]
si       = np.argsort(f_env)
f_env, s_env = f_env[si], s_env[si]

pk_env, _ = find_peaks(s_env, distance=15, prominence=0.01*np.max(s_env))
if len(pk_env) > 4:
    pf  = f_env[pk_env]
    pp  = s_env[pk_env]**2
    mid = np.argmax(pp)
    f_pk = pf[mid];  mx = pp[mid];  half = mx/2
    try:
        spl   = UnivariateSpline(pf, pp - half, s=0)
        roots = spl.roots()
        lft   = roots[roots < f_pk]
        rgt   = roots[roots > f_pk]
        if len(lft) and len(rgt):
            fl, fr  = lft[-1], rgt[0]
            fwhm    = fr - fl
            Q_dbr   = (f0_ghz + f_pk) / fwhm
            print(f"  DBR peak offset : {f_pk:.4f} GHz")
            print(f"  Left  3dB       : {fl:.4f} GHz")
            print(f"  Right 3dB       : {fr:.4f} GHz")
            print(f"  FWHM            : {fwhm:.4f} GHz")
            print(f"  Q_DBR           : {Q_dbr:.2f}")
    except Exception as e:
        print(f"  FWHM calculation failed: {e}")

print("\nDone.")

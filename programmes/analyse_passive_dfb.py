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

# ── DFB cavity parameters ──────────────────────────────────────────────────────
L_dfb    = 400e-6       # m
n_eff    = 3.5          # semiconductor effective index (estimate)
tau_rt   = 2 * L_dfb * n_eff / c  # roundtrip time (s)
print(f"DFB cavity length  : {L_dfb*1e6:.0f} um")
print(f"Estimated n_eff    : {n_eff}")
print(f"Roundtrip time     : {tau_rt*1e12:.3f} ps")

# ── Load ───────────────────────────────────────────────────────────────────────
print("\nLoading passive_dfb.ascii ...")
d    = np.loadtxt('passive_dfb.ascii', skiprows=2)
t    = d[:,0]; dt = t[1]-t[0]; N = len(t)
A    = d[:,1] + 1j*d[:,2]
absA = np.abs(A)

print(f"Points: {N},  dt={dt:.4e} ns  ({dt*1e3:.4f} ps)")
print(f"max|A|={np.max(absA):.4e}  at t={t[np.argmax(absA)]:.6f} ns")

# ── Peak detection with correct roundtrip separation ─────────────────────────
tau_rt_ns  = tau_rt * 1e9   # ps -> ns
min_dist   = max(1, int(0.5 * tau_rt_ns / dt))  # half roundtrip in samples
print(f"Peak min_distance  : {min_dist} samples ({min_dist*dt*1e3:.3f} ps)")

peaks_t, _ = find_peaks(absA, distance=min_dist, prominence=1e-20)
print(f"\nDetected {len(peaks_t)} peaks:")
for i, idx in enumerate(peaks_t[:30]):
    print(f"  Peak {i+1:2d}: t={t[idx]:.7f} ns  |A|={absA[idx]:.4e}")

if len(peaks_t) >= 2:
    spacings = np.diff(t[peaks_t])
    rt_measured = np.median(spacings)
    n_eff_meas  = rt_measured * 1e-9 * c / (2 * L_dfb)
    print(f"\nMeasured roundtrip : {rt_measured*1e3:.4f} ps")
    print(f"Implied n_eff      : {n_eff_meas:.4f}")

# ── Time-domain log plot zoomed to decay region ───────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

valid = absA > 1e-30
axes[0].semilogy(t[valid], absA[valid])
if len(peaks_t):
    axes[0].semilogy(t[peaks_t], absA[peaks_t], 'ro', ms=5, label='Detected peaks')
axes[0].set_xlabel('Time (ns)'); axes[0].set_ylabel('Amplitude (log)')
axes[0].set_title('Passive DFB -- Full Log Scale'); axes[0].grid(True); axes[0].legend()

# Zoom to decay region
t_start = t[peaks_t[0]] - 0.005 if len(peaks_t) else 0.49
t_end   = t[peaks_t[-1]] + 0.005 if len(peaks_t) else 0.62
zoom    = (t >= t_start) & (t <= t_end)
axes[1].semilogy(t[zoom & valid], absA[zoom & valid])
if len(peaks_t):
    zp = peaks_t[(t[peaks_t] >= t_start) & (t[peaks_t] <= t_end)]
    axes[1].semilogy(t[zp], absA[zp], 'ro', ms=6, label='Detected peaks')
axes[1].set_xlabel('Time (ns)'); axes[1].set_ylabel('Amplitude (log)')
axes[1].set_title('Passive DFB -- Decay Region (Zoomed)'); axes[1].grid(True); axes[1].legend()

plt.tight_layout()
plt.savefig('passive_dfb_timedomain2.png', dpi=150)
plt.close()
print("\nSaved passive_dfb_timedomain2.png")

# ── Q-factor from decay fit ───────────────────────────────────────────────────
if len(peaks_t) >= 3:
    t_pk  = t[peaks_t]
    a_pk  = absA[peaks_t]
    lg_pk = np.log10(a_pk)

    m, c_fit = np.polyfit(t_pk, lg_pk, 1)
    res = lg_pk - (m*t_pk + c_fit)
    r2  = 1 - np.sum(res**2)/np.sum((lg_pk - np.mean(lg_pk))**2)

    # Carrier offset from phase
    unwrapped = np.unwrap(np.angle(A))
    ph_pk     = unwrapped[peaks_t]
    m_ph, _   = np.polyfit(t_pk, ph_pk, 1)
    f_env_ghz = m_ph / (2*np.pi)
    f_R_ghz   = f0_ghz + f_env_ghz
    Q_pk      = -np.pi * f_R_ghz * log10_e / m
    tau_p_ps  = Q_pk / (2 * np.pi * f_R_ghz * 1e9) * 1e12  # photon lifetime in ps

    print(f"\n== Pulse-Peak Decay Q-Factor ==")
    print(f"  Decay slope m       : {m:.4f} ns^-1")
    print(f"  R^2                 : {r2:.8f}")
    print(f"  Carrier offset      : {f_env_ghz:.2f} GHz")
    print(f"  f_R                 : {f_R_ghz:.4f} GHz  ({f_R_ghz*1e-3:.6f} THz)")
    print(f"  Photon lifetime tau : {tau_p_ps:.3f} ps")
    print(f"  Q (pulse peaks)     : {Q_pk:.2f}")

    # Decay fit plot
    plt.figure(figsize=(8, 6))
    plt.plot(t_pk*1e3, lg_pk, 'o', color='blue', label='Detected Peaks', ms=6)
    t_fit = np.linspace(t_pk[0]-0.001, t_pk[-1]+0.001, 300)
    plt.plot(t_fit*1e3, m*t_fit+c_fit, '-', color='red',
             label=f'Linear Fit  R^2={r2:.6f}')
    plt.xlabel('Time (ps)'); plt.ylabel('log10(Amplitude)')
    plt.title('Passive DFB Cavity Q-Factor -- Decay Fit')
    plt.grid(True); plt.legend()
    txt = (f'f_R = {f_R_ghz*1e-3:.6f} THz\n'
           f'm = {m:.4f} ns^-1\n'
           f'tau_p = {tau_p_ps:.2f} ps\n'
           f'Q = {Q_pk:.1f}')
    plt.gca().text(0.6, 0.95, txt, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.savefig('passive_dfb_decay_fit.png', dpi=150)
    plt.close()
    print("Saved passive_dfb_decay_fit.png")

else:
    print(f"\nWARNING: Only {len(peaks_t)} peaks -- cannot fit decay.")
    print("Possible fixes:")
    print("  1. Increase simulation time so more roundtrips are captured")
    print("  2. Reduce absorption in the passive active region (set to transparency)")

print("\nDone.")

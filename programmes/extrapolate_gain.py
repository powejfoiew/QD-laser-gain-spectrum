"""
Extrapolate QD gain tables to higher carrier densities and export
in PICWave-compatible format.

Run this in the gain_table.ipynb notebook after cells 0-8 have executed,
so that gain_0_arr, gain_1_arr, N_e_es1, N_e_es2, and wavelength
are all available in the namespace.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
gamma_xy           = 0.08303645356032915
conversion_factor  = 4019376.8605
height2            = 40E-7          # QD layer height [cm]  (= 40 nm)
TEMPERATURE_C      = 25.0
MAX_N_ENTRIES      = 499            # PICWave hard limit
N_TARGET_MAX       = 1.0E18         # target maximum carrier density [cm^-3]
N_EXTRAP_POINTS    = 150            # how many slots to reserve for extrapolation
N_KEEP_ORIGINAL    = MAX_N_ENTRIES - N_EXTRAP_POINTS   # 349

# ── Logarithmic gain model: g(N) = a·ln(N) + b ──────────────────────────────
#    This naturally captures gain saturation at high carrier densities,
#    which is the expected physical behaviour for QD states.
def log_model(N, a, b):
    return a * np.log(N) + b


# ══════════════════════════════════════════════════════════════════════════════
#  1.  Build sorted carrier-density and material-gain arrays
# ══════════════════════════════════════════════════════════════════════════════

# Carrier densities  [cm^-3]
N_es1_raw = np.sum(N_e_es1, axis=1) * 1E-4 / height2
N_es2_raw = np.sum(N_e_es2, axis=1) * 1E-4 / height2

# Material gain  [cm^-1]
mat_gain_0 = gain_0_arr * conversion_factor / gamma_xy   # GS
mat_gain_1 = gain_1_arr * conversion_factor / gamma_xy   # ES1

# Sort indices — ascending carrier density & ascending wavelength
sort_idx_0   = np.argsort(N_es1_raw)
sort_idx_1   = np.argsort(N_es2_raw)
lam_sort_idx = np.argsort(wavelength)

# Apply sorting
N0_sorted   = N_es1_raw[sort_idx_0]
N1_sorted   = N_es2_raw[sort_idx_1]
mg0_sorted  = mat_gain_0[sort_idx_0][:, lam_sort_idx]
mg1_sorted  = mat_gain_1[sort_idx_1][:, lam_sort_idx]
wl_sorted   = wavelength[lam_sort_idx]                   # [nm], ascending

n_lambda      = len(wl_sorted)
lambda_min_um = wl_sorted.min() / 1000.0
lambda_max_um = wl_sorted.max() / 1000.0


# ── Remove duplicate / zero-valued N entries ─────────────────────────────────
def deduplicate(N, gain):
    """Keep only unique, positive N values (ascending)."""
    mask       = N > 0
    N, gain    = N[mask], gain[mask]
    _, idx     = np.unique(N, return_index=True)
    return N[idx], gain[idx]

N0_sorted, mg0_sorted = deduplicate(N0_sorted, mg0_sorted)
N1_sorted, mg1_sorted = deduplicate(N1_sorted, mg1_sorted)


# ══════════════════════════════════════════════════════════════════════════════
#  2.  Downsample original data to make room for extrapolated points
# ══════════════════════════════════════════════════════════════════════════════

def downsample_to_n(N_arr, gain_arr, n_keep):
    """
    Thin N_arr/gain_arr from their current length down to n_keep points.

    Strategy: always keep the first and last points, then select the
    remaining n_keep-2 points with roughly uniform spacing in log(N).
    This preserves good coverage across the full dynamic range while
    thinning out densely-sampled regions.
    """
    n = len(N_arr)
    if n <= n_keep:
        return N_arr, gain_arr          # nothing to do

    # Uniformly-spaced indices in log(N)
    log_N     = np.log(N_arr)
    target    = np.linspace(log_N[0], log_N[-1], n_keep)
    chosen    = set()
    chosen.add(0)
    chosen.add(n - 1)

    for t in target:
        idx = int(np.argmin(np.abs(log_N - t)))
        chosen.add(idx)

    # If we picked too many (due to rounding), trim from the middle
    chosen = sorted(chosen)
    while len(chosen) > n_keep:
        # Remove the point closest to its neighbours (least needed)
        min_gap  = np.inf
        drop_idx = 1                    # never drop first or last
        for i in range(1, len(chosen) - 1):
            gap = N_arr[chosen[i+1]] - N_arr[chosen[i-1]]
            if gap < min_gap:
                min_gap  = gap
                drop_idx = i
        chosen.pop(drop_idx)

    chosen = np.array(chosen)
    return N_arr[chosen], gain_arr[chosen]


# ══════════════════════════════════════════════════════════════════════════════
#  3.  Extrapolation
# ══════════════════════════════════════════════════════════════════════════════

def extrapolate_gain(N_existing, gain_existing, N_max_target,
                     max_total, n_keep_original):
    """
    Downsample existing data, then extend to higher N using a log-model fit.

    Returns
    -------
    N_combined, gain_combined, n_original
    """
    n_lam     = gain_existing.shape[1]
    N_max_cur = N_existing[-1]

    # ── Downsample originals ──────────────────────────────────────────────
    N_ds, g_ds = downsample_to_n(N_existing, gain_existing, n_keep_original)
    n_ds       = len(N_ds)
    n_extrap   = max_total - n_ds

    print(f"  Original : {len(N_existing)} points,  "
          f"N ∈ [{N_existing[0]:.3e}, {N_max_cur:.3e}]")
    print(f"  Kept     : {n_ds} points after downsampling")
    print(f"  Adding   : {n_extrap} extrapolated points up to "
          f"N = {N_max_target:.3e}")

    if N_max_cur >= N_max_target or n_extrap <= 0:
        print("  ⚠ No room to extrapolate; returning downsampled original.")
        return N_ds[:max_total], g_ds[:max_total], min(n_ds, max_total)

    # ── Log-spaced new N values ───────────────────────────────────────────
    N_extrap = np.logspace(np.log10(N_max_cur * 1.01),
                           np.log10(N_max_target),
                           n_extrap)

    # ── Fit on upper half of FULL original data (before downsampling)
    #    to get the best possible fit ──────────────────────────────────────
    n_fit       = max(len(N_existing) // 2, 20)
    gain_extrap = np.zeros((n_extrap, n_lam))

    for li in range(n_lam):
        N_fit = N_existing[-n_fit:]
        g_fit = gain_existing[-n_fit:, li]
        try:
            popt, _ = curve_fit(log_model, N_fit, g_fit,
                                p0=[100.0, -3000.0], maxfev=10000)
            g_ext = log_model(N_extrap, *popt)
            
            # Force C0 continuity: perfectly attach to the last original point
            offset = g_fit[-1] - g_ext[0]
            gain_extrap[:, li] = g_ext + offset
        except (RuntimeError, ValueError):
            # Fallback: linear extrapolation from last two points
            slope = (g_fit[-1] - g_fit[-2]) / (N_fit[-1] - N_fit[-2])
            gain_extrap[:, li] = g_fit[-1] + slope * (N_extrap - N_fit[-1])

    N_comb = np.concatenate([N_ds, N_extrap])
    g_comb = np.concatenate([g_ds, gain_extrap], axis=0)
    return N_comb, g_comb, n_ds


print("=" * 60)
print("  Gain 0  (GS transition,  indexed by ES1 carrier density)")
print("=" * 60)
N0_ext, g0_ext, n_orig_0 = extrapolate_gain(
    N0_sorted, mg0_sorted, N_TARGET_MAX,
    MAX_N_ENTRIES, N_KEEP_ORIGINAL)

print()
print("=" * 60)
print("  Gain 1  (ES1 transition, indexed by ES2 carrier density)")
print("=" * 60)
N1_ext, g1_ext, n_orig_1 = extrapolate_gain(
    N1_sorted, mg1_sorted, N_TARGET_MAX,
    MAX_N_ENTRIES, N_KEEP_ORIGINAL)


# ══════════════════════════════════════════════════════════════════════════════
#  4.  Plotting
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ── Top-left: Gain 0 spectra ─────────────────────────────────────────────────
ax = axes[0, 0]
# Plot ALL original data
for i in range(n_orig_0):
    ax.plot(wl_sorted, g0_ext[i], color='C0', alpha=0.4)
# Plot extrapolated data with a step
step_extrap = max(1, (len(N0_ext) - n_orig_0) // 15)
for i in range(n_orig_0, len(N0_ext), step_extrap):
    ax.plot(wl_sorted, g0_ext[i], color='C3', alpha=0.6)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Material gain (cm⁻¹)')
ax.set_title('Gain 0 (GS) — blue=original, red=extrapolated')
ax.axhline(0, color='k', lw=0.5)

# ── Top-right: Gain 1 spectra ────────────────────────────────────────────────
ax = axes[0, 1]
# Plot ALL original data
for i in range(n_orig_1):
    ax.plot(wl_sorted, g1_ext[i], color='C0', alpha=0.4)
# Plot extrapolated data with a step
step_extrap = max(1, (len(N1_ext) - n_orig_1) // 15)
for i in range(n_orig_1, len(N1_ext), step_extrap):
    ax.plot(wl_sorted, g1_ext[i], color='C3', alpha=0.6)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Material gain (cm⁻¹)')
ax.set_title('Gain 1 (ES1) — blue=original, red=extrapolated')
ax.axhline(0, color='k', lw=0.5)

# ── Bottom-left: Gain 0 peak gain vs N ───────────────────────────────────────
ax = axes[1, 0]
peak_g0 = np.max(g0_ext, axis=1)
ax.plot(N0_ext[:n_orig_0], peak_g0[:n_orig_0],
        'b.-', lw=1.5, ms=3, label=f'Original ({n_orig_0} pts)')
ax.plot(N0_ext[n_orig_0:], peak_g0[n_orig_0:],
        'r.--', lw=1.5, ms=3, label=f'Extrapolated ({len(N0_ext)-n_orig_0} pts)')
ax.set_xlabel('Carrier density N (cm⁻³)')
ax.set_ylabel('Peak material gain (cm⁻¹)')
ax.set_title('Gain 0 (GS) — Peak gain vs N')
ax.legend()
ax.axhline(0, color='k', lw=0.5)

# ── Bottom-right: Gain 1 peak gain vs N ──────────────────────────────────────
ax = axes[1, 1]
peak_g1 = np.max(g1_ext, axis=1)
ax.plot(N1_ext[:n_orig_1], peak_g1[:n_orig_1],
        'b.-', lw=1.5, ms=3, label=f'Original ({n_orig_1} pts)')
ax.plot(N1_ext[n_orig_1:], peak_g1[n_orig_1:],
        'r.--', lw=1.5, ms=3, label=f'Extrapolated ({len(N1_ext)-n_orig_1} pts)')
ax.set_xlabel('Carrier density N (cm⁻³)')
ax.set_ylabel('Peak material gain (cm⁻¹)')
ax.set_title('Gain 1 (ES1) — Peak gain vs N')
ax.legend()
ax.axhline(0, color='k', lw=0.5)

plt.tight_layout()
plt.savefig('gain_extrapolated.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved → gain_extrapolated.png")


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Write PICWave-compatible files
# ══════════════════════════════════════════════════════════════════════════════

def write_gain_file(filename, N_arr, gain_arr, n_lam,
                    lam_min_um, lam_max_um, temp_c=25.0):
    n_N = len(N_arr)
    assert n_N <= 499, f"n_N = {n_N} exceeds PICWave limit of 499!"
    with open(filename, 'w') as f:
        f.write('begin <negainspectrum(1,0)>\n')
        f.write(
            f'{n_lam} '
            f'{lam_min_um:.6f} '
            f'{lam_max_um:.6f} '
            f'1 '
            f'{temp_c:.1f} '
            f'{temp_c:.1f} '
            f'1\n'
        )
        f.write(f'\n//T={temp_c:.0f} [C]\n')
        f.write(f'{n_N} //nN\n')
        f.write(' '.join(f'{n:.6e}' for n in N_arr) + '\n')
        for lam_idx in range(n_lam):
            row = gain_arr[:, lam_idx]
            f.write(' '.join(f'{g:.6e}' for g in row) + '\n')
        f.write('end\n')
    print(f"  → {filename}  ({n_N} N × {n_lam} λ)")


print("\n=== Writing PICWave gain files ===")
write_gain_file('qd_gain0_red_TE.txt',
                N0_ext, g0_ext,
                n_lambda, lambda_min_um, lambda_max_um, TEMPERATURE_C)
write_gain_file('qd_gain1_red_TE.txt',
                N1_ext, g1_ext,
                n_lambda, lambda_min_um, lambda_max_um, TEMPERATURE_C)


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\nExport complete.")
print(f"  Gain 0: N ∈ [{N0_ext[0]:.3e}, {N0_ext[-1]:.3e}],  "
      f"{len(N0_ext)} total  ({n_orig_0} kept + "
      f"{len(N0_ext) - n_orig_0} extrapolated)")
print(f"  Gain 1: N ∈ [{N1_ext[0]:.3e}, {N1_ext[-1]:.3e}],  "
      f"{len(N1_ext)} total  ({n_orig_1} kept + "
      f"{len(N1_ext) - n_orig_1} extrapolated)")
print(f"\nAdd to your .mat file:")
print(f"  IMPORT_GAIN_SPECTRA_TE qd_gain0_red_TE.txt")
print(f"  IMPORT_GAIN_SPECTRA_TE qd_gain1_red_TE.txt")

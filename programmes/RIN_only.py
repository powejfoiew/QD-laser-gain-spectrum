import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import medfilt, savgol_filter, find_peaks, peak_widths

# ── Formatting parameters matching plot_bowers_data.py ───────────────────────
maj_tick_width = 0.8
min_tick_width = 0.8

plot_params = {
    "figure.dpi": 300,
    "axes.labelsize": 12,
    "axes.linewidth": 1.5,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.title_fontsize": 10,
    "legend.fontsize": 9,
    "xtick.major.size": 3.5,
    "xtick.major.width": maj_tick_width,
    "xtick.minor.size": 2.5,
    "xtick.minor.width": min_tick_width,
    "ytick.major.size": 3.5,
    "ytick.major.width": maj_tick_width,
    "ytick.minor.size": 2.5,
    "ytick.minor.width": min_tick_width,
    "font.family": "serif",
    "text.usetex": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
}
plt.rcParams.update(plot_params)

# ── Paul Tol Palette Colors ──────────────────────────────────────────────────
# Qualitative palette: bright blue for data, rose/magenta/red/grey for noise regions
color_data = '#4477AA'       # Paul Tol Bright Blue
color_pink_shade = '#FFAABB'  # Paul Tol Light Pink
color_pink_fit = '#AA3377'    # Paul Tol Purple/Magenta
color_red_shade = '#EE6677'   # Paul Tol Light Red (Rose)
color_red_fit = '#CC3311'     # Paul Tol Red
color_white_shade = '#DDDDDD' # Paul Tol Light Grey (representing White floor)
color_white_fit = '#555555'   # Paul Tol Dark Grey

# ── Helper functions for Frequency Noise processing ─────────────────────────
def stitching_post_process(freq, psdi):
    freq = np.array(freq)
    psdi = np.array(psdi)
    min_log = int(np.floor(np.log10(np.min(freq))))
    max_log = int(np.floor(np.log10(np.max(freq))))
    n_array = np.arange(min_log, max_log + 1)
    
    psd0 = psdi.copy()
    
    i_end = {}
    i_start = {}
    
    for n in n_array:
        Aa = np.where(freq == 10**n)[0]
        if len(Aa) > 0:
            i_end[n] = Aa[0]
            i_start[n + 1] = Aa[-1]
            
    i_start[min_log] = 0
    
    adjust = {}
    for n in range(max_log, min_log - 1, -1):
        if (n + 1) in i_start and n in i_end:
            adjust[n] = 1.0 * psd0[i_start[n + 1]] / psdi[i_end[n]]
            psd0[i_start[n]:i_end[n] + 1] = psdi[i_start[n]:i_end[n] + 1] * adjust[n]
            
    return freq, psd0

def smooth_moving(y, n):
    smoothed = np.empty_like(y)
    for i in range(len(y)):
        start = max(0, i - (n - 1) // 2)
        end = min(len(y), i + n // 2 + 1)
        smoothed[i] = np.mean(y[start:end])
    return smoothed

def Lorentzian_linewidth(freq, PSD, average, fmin, fmax):
    yy = smooth_moving(PSD, average)
    xmin = np.argmin(np.abs(freq - fmin))
    xmax = np.argmin(np.abs(freq - fmax))
    
    idx_start = min(xmin, xmax)
    idx_end = max(xmin, xmax)
    
    val_min = np.min(yy[idx_start:idx_end + 1])
    return np.pi * val_min

# ── Helper functions for RIN fitting ─────────────────────────────────────────
def rin_model_linear(omega, a, b, omega_ro, gamma):
    return a + b * omega ** 2 / ((omega ** 2 - omega_ro ** 2) ** 2 + (omega * gamma) ** 2)

def rin_model_db_logparams(omega, log_a, log_b, log_omega_ro, log_gamma):
    a, b, omega_ro, gamma = 10 ** log_a, 10 ** log_b, 10 ** log_omega_ro, 10 ** log_gamma
    lin = rin_model_linear(omega, a, b, omega_ro, gamma)
    return 10 * np.log10(np.clip(lin, 1e-300, None))

def smooth_dbdata(freq_hz, rin_dbhz, medfilt_kernel=51, savgol_window=201, savgol_poly=3):
    n = freq_hz.size
    mk = min(medfilt_kernel, n - (1 - n % 2))
    if mk < 3:
        mk = 3
    if mk % 2 == 0:
        mk += 1
    med = medfilt(rin_dbhz, kernel_size=mk)
    sw = min(savgol_window, n - (1 - n % 2))
    if sw < 5:
        sw = 5 if n >= 5 else n if n % 2 == 1 else n - 1
    if sw % 2 == 0:
        sw += 1
    sw = max(sw, savgol_poly + 2 + (savgol_poly + 2) % 2 + 1)
    return savgol_filter(med, window_length=sw, polyorder=savgol_poly, mode="interp")

def light_average_smooth(y, frac=0.0001, min_window=1, max_window=2, fixed_window=None):
    """A gentle moving-average smoothing pass for display purposes only (does not
    feed the fit). window=1 means no smoothing at all (identity pass-through).
    window=0 is invalid — it makes the averaging slice empty and produces NaNs
    (breaks/gaps in the line), so min_window is floored at 1.

    Pass fixed_window=N to bypass the frac-based sizing entirely and force an
    exact window size in samples — more predictable than frac, since the frac
    calculation depends on len(y), which may be larger than you expect."""
    n = len(y)
    win = fixed_window if fixed_window is not None else int(round(n * frac))
    win = max(1, min_window, min(max_window, win))
    win = min(win, n)
    print(f"[light_average_smooth] n={n}, window={win}"
          + (" (no smoothing — identity)" if win == 1 else ""))
    return smooth_moving(y, win)

def fit_relaxation_oscillation(freq_hz, rin_dbhz, fit_fmin, fit_fmax):
    # 1. Smooth the data inside the main fit window first to find the peak
    mask_search = (freq_hz >= fit_fmin) & (freq_hz <= fit_fmax)
    f_s = freq_hz[mask_search]
    y_s = rin_dbhz[mask_search]
    if f_s.size < 30:
        return None
    y_smooth_search = smooth_dbdata(f_s, y_s)
    # 2. Peak search
    peaks, props = find_peaks(y_smooth_search, prominence=0.3)
    if len(peaks) > 0:
        peak_i = peaks[np.argmax(props["prominences"])]
        f_ro_guess = f_s[peak_i]
        
        # Determine peak-centered window
        widths, _, left_ips, right_ips = peak_widths(y_smooth_search, [peak_i], rel_height=0.6)
        half_width_left = peak_i - left_ips[0]
        half_width_right = right_ips[0] - peak_i
        
        # Set fit window bounds around peak (narrower on the left to avoid flat noise underfitting)
        idx_min = max(0, int(np.floor(peak_i - 1.3 * half_width_left)))
        idx_max = min(f_s.size - 1, int(np.ceil(peak_i + 3.0 * half_width_right)))
        
        f_min_fit = f_s[idx_min]
        f_max_fit = f_s[idx_max]
    else:
        f_ro_guess = f_s[np.argmax(y_smooth_search)]
        f_min_fit = fit_fmin
        f_max_fit = fit_fmax
    # 3. Restrict fit data to this local window
    mask_fit = (freq_hz >= f_min_fit) & (freq_hz <= f_max_fit)
    f = freq_hz[mask_fit]
    y_db = rin_dbhz[mask_fit]
    if f.size < 15:
        return None
    
    y_smooth = smooth_dbdata(f, y_db)
    omega = 2 * np.pi * f
    
    # 4. Initial guess from local window
    omega_ro0 = 2 * np.pi * f_ro_guess
    gamma0 = 0.3 * omega_ro0
    a0 = 10 ** (np.percentile(y_smooth, 5) / 10.0)
    
    peak_idx_local = np.argmin(np.abs(f - f_ro_guess))
    peak_lin0 = 10 ** (y_smooth[peak_idx_local] / 10.0)
    b0 = max(peak_lin0 - a0, 1e-30) * gamma0 ** 2
    
    p0 = [np.log10(a0), np.log10(max(b0, 1e-300)), np.log10(omega_ro0), np.log10(gamma0)]
    # Parameter bounds to keep optimizer well-behaved
    omega_fmin, omega_fmax = 2 * np.pi * f[0], 2 * np.pi * f[-1]
    log_wro_lo, log_wro_hi = np.log10(0.3 * omega_fmin), np.log10(3.0 * omega_fmax)
    log_gamma_lo, log_gamma_hi = np.log10(0.02 * omega_fmin), np.log10(10.0 * omega_fmax)
    lower = [-30.0, -50.0, log_wro_lo, log_gamma_lo]
    upper = [-5.0, 50.0, log_wro_hi, log_gamma_hi]
    
    try:
        popt, pcov = curve_fit(rin_model_db_logparams, omega, y_smooth, p0=p0, bounds=(lower, upper), maxfev=100000)
    except (RuntimeError, ValueError) as e:
        print("Fit failed:", e)
        return None
    
    a, b, omega_ro, gamma = 10 ** popt[0], 10 ** popt[1], 10 ** popt[2], 10 ** popt[3]
    f_RO = omega_ro / (2 * np.pi)
    return {
        "success": True,
        "a": a, "b": b, "omega_ro": omega_ro, "gamma": gamma,
        "f_RO": f_RO, "popt": popt,
        "freq": f, "rin_db": y_db, "rin_smooth": y_smooth
    }

# ── Load and Process Data ────────────────────────────────────────────────────
folder = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
bowers_folder = os.path.join(folder, "Bowers")

# 1. Frequency Noise Data
dfb = pd.read_csv(os.path.join(bowers_folder, "210119DFB_Die1Bar2D02_160mA_20C"), sep=r'\s+', skiprows=12, header=None)
fn_x_raw = dfb[0].to_numpy()
fn_y_raw = dfb[3].to_numpy()
fn_x, fn_y = stitching_post_process(fn_x_raw, fn_y_raw)

# Fit Frequency Noise regimes
white_noise_mask = fn_x > 1E7
pink_noise_mask = (fn_x >= 1.25E5) & (fn_x <= 1E7)
blue_noise_mask = fn_x < 1.25E5

blue_xs  = np.geomspace(1E3, 1.25E5, 100)
pink_xs  = np.geomspace(1.25E5, 1E7, 100)
white_xs = np.geomspace(1E7, 1E8, 100)

log_x = np.log10(fn_x)
log_y = np.log10(fn_y)

c_white = np.mean(log_y[white_noise_mask])
y_white_fit = 10**c_white * np.ones_like(white_xs)

c_pink = np.mean(log_y[pink_noise_mask] - (-2.0 * log_x[pink_noise_mask]))
y_pink_fit = 10**c_pink * (pink_xs**(-2.0))

c_blue = np.mean(log_y[blue_noise_mask] - (-1.0 * log_x[blue_noise_mask]))
y_blue_fit = 10**c_blue * (blue_xs**(-1.0))

# 2. QW RIN Data
mat_data = sio.loadmat(os.path.join(bowers_folder, "230215_QW_RIN.mat"))
rin_x = mat_data['b'].flatten()
rin_y = mat_data['bb'].flatten()

# Light display-only smoothing to knock down scatter without over-smoothing.
# The fit itself still uses the raw (unsmoothed) rin_y via fit_relaxation_oscillation,
# which does its own internal smoothing just for peak-finding/fitting purposes.
# Use fixed_window directly (in samples) rather than frac — it's the most direct
# knob: 1 = no smoothing, 2 = adjacent-pair average, 3 = 3-point average, etc.
rin_y_smooth = light_average_smooth(rin_y, fixed_window=3)

# Fit QW RIN
fit_res = fit_relaxation_oscillation(rin_x, rin_y, 1e9, 2e10)

# ── Create 2-Panel Figure ─────────────────────────────────────────────────────
fig, (ax2) = plt.subplots(figsize=(4, 3))
ax2.plot(rin_x, rin_y_smooth, color=color_data, linewidth=0.8, alpha=0.85, label='QW RIN Data')

# Resonance Fit
if fit_res:
    fit_omega = 2 * np.pi * fit_res['freq']
    fit_curve = rin_model_db_logparams(fit_omega, *fit_res['popt'])
    f_ro_ghz = fit_res['f_RO'] / 1e9
    gamma_grads = fit_res['gamma'] / 1e9
    fit_label = f'RO Fit ($f_{{\\mathrm{{RO}}}}={f_ro_ghz:.2f}$~GHz, $\\gamma={gamma_grads:.2f}$~Grad/s)'
    ax2.plot(fit_res['freq'], fit_curve, color='black', linestyle='--', linewidth=1.5, label=fit_label)

# Axes setup
ax2.set_xscale('log')  # Explicitly log scale on x-axis
ax2.set_xlim(1e7, 8e9)
ax2.set_ylim(-170, -120)
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('RIN [dB/Hz]')
#ax2.set_title("(b)")
#ax2.legend(loc='lower left', frameon=True, fontsize=8, handlelength=1.2, handletextpad=0.4)

# ── Finish up and save ────────────────────────────────────────────────────────
fig.tight_layout()

# Save locally in the regimes folder
fig.savefig(os.path.join(folder, "QD_Frequency_noise_and_RIN.png"), dpi=300)
fig.savefig(os.path.join(folder, "QD_Frequency_noise_and_RIN.pdf"))

# Save in the article's pgf folder
pgf_dir = r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Ultra-stable-lasers-and-photonic-integrated-circuits-for-secure-communications\content\Figures\pgf_figures"
if os.path.exists(pgf_dir):
    fig.savefig(os.path.join(pgf_dir, "RIN_only.pgf"))
    fig.savefig("RIN_only.pdf")
    print("PGF and copy figures successfully saved to pgf_figures directory.")

plt.show()
print("Plots generated and saved successfully.")
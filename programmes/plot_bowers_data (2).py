import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

maj_tick_width = 0.8
min_tick_width = 0.8

plot_params = { "figure.dpi": "200", "axes.labelsize": 20, "axes.linewidth": 1.5, "axes.titlesize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.title_fontsize": 16, "legend.fontsize": 16, "xtick.major.size": 3.5, "xtick.major.width": maj_tick_width, "xtick.minor.size": 2.5, "xtick.minor.width": min_tick_width, "ytick.major.size": 3.5, "ytick.major.width": maj_tick_width, "ytick.minor.size": 2.5, "ytick.minor.width": min_tick_width, "font.family": "serif", "text.usetex": True, "xtick.direction": "in","ytick.direction": "in", "pgf.texsystem": "pdflatex","font.family": "serif", "text.usetex": True, "pgf.rcfonts": False,}
plt.rcParams.update(plot_params)

# Professional colors from MATLAB script (divided by 256)
colors = np.array([
    [255, 247, 236], [247, 252, 240], [224, 243, 219], [204, 235, 197],
    [168, 221, 181], [123, 204, 196], [78, 179, 211], [43, 140, 190],
    [183, 79, 43], [8, 88, 158], [11, 37, 58], [182, 160, 157],
    [83, 157, 179], [65, 105, 225]
]) / 256.0

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

# Load data
folder = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
files = ["210119DFB_Die4Bar4D20_100mA_20C", "210119DFB_Die1Bar2D02_160mA_20C"]

pn = []
for fn in files:
    filepath = os.path.join(folder, fn)
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=12, header=None)
    x = df[0].to_numpy()
    y = df[3].to_numpy()
    x_proc, y_proc = stitching_post_process(x, y)
    pn.append({'x': x_proc, 'y': y_proc})

# Linewidth and white noise floor calculations
linewidths = []
fn_levels = []
for i in range(len(files)):
    lw = Lorentzian_linewidth(pn[i]['x'], pn[i]['y'], 250, 1e7, 1e8)
    linewidths.append(lw)
    fn_levels.append(lw / np.pi)

# Beta separation line
beta = (8 * np.log(2) / np.pi**2) * pn[0]['x']

# ── Plot 1: Full Range ──────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(4, 3), dpi=300)

ax1.plot(pn[1]['x'], pn[1]['y'], color=colors[9], linewidth=1, label="1500 $\mu$m")

x = pn[1]['x']
y = pn[1]['y']

white_noise_mask = pn[1]['x']>1E7
pink_noise_mask = (pn[1]['x'] >= 1.25E5) & (pn[1]['x'] <= 1E7)
blue_noise_mask = pn[1]['x']< 1.25E5

# Define logarithmic spacing for smooth curves on a log scale
blue_xs  = np.geomspace(1E3, 1.25E5, 100)
pink_xs  = np.geomspace(1.25E5, 1E7, 100)
white_xs = np.geomspace(1E7, 1E8, 100)

log_x = np.log10(x)
log_y = np.log10(y)

c_white = np.mean(log_y[white_noise_mask])
y_white_fit = 10**c_white * np.ones_like(white_xs)

c_pink = np.mean(log_y[pink_noise_mask] - (-2.0 * log_x[pink_noise_mask]))
y_pink_fit = 10**c_pink * (pink_xs**(-2.0))

c_blue = np.mean(log_y[blue_noise_mask] - (-1.0 * log_x[blue_noise_mask]))
y_blue_fit = 10**c_blue * (blue_xs**(-1.0))

ax1.text(1e4,1e11,r'$f^{-1}$')
ax1.text(1e6,1e11,r'$f^{-2}$')
ax1.text(3e7,1e11,r'$f^{0}$')

# Shading horizontal regions along the x-axis
ax1.axvspan(1e3,    1.25e5, color='#EE3377', alpha=0.2, label='Low Freq (f^-1)')
ax1.axvspan(1.25e5, 1e7,    color='#CC3311',  alpha=0.3, label='Mid Freq (f^-2)')
ax1.axvspan(1e7,    1e8,    color='#BBBBBB', alpha=0.2, label='White Floor (f^0)')

# Plotting on log-log axis
ax1.plot(white_xs, y_white_fit, color='#000000', label='White noise (Constant)',linestyle='dashed')
ax1.plot(pink_xs, y_pink_fit,   color='#CC3311', label='Pink noise (~1/f)',linestyle='dashed')
ax1.plot(blue_xs, y_blue_fit,   color='#EE3377', label='Blue noise (~1/f^2)',linestyle='dashed')

# Print fitted parameters
print(f"White noise constant baseline: {10**c_white:.3e} [Hz^2/Hz]")
print(f"Pink noise coefficient (10^c): {10**c_pink:.3e}")
print(f"Blue noise coefficient (10^c): {10**c_blue:.3e}")

# Plot beta separation line
#ax1.plot(pn[0]['x'], beta, color='black', linestyle='--', linewidth=1, label=r"$\beta$ separation line")

res = stats.linregress(x, y)

ax1.set_xlim(1e3, 1e8)
ax1.set_ylim(1e3, 1e12)

# Scales & Labels
ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.set_xlabel('Offset Frequency [Hz]', fontsize=12)
ax1.set_ylabel(r'Frequency Noise [$\mathrm{Hz^2/Hz}$]', fontsize=12)
ax1.tick_params(axis='both', which='both', direction='out', labelsize=10)
#ax1.legend(loc='lower left', frameon=True, fontsize=9)

# Adjust yticks and xticks to match MATLAB
#ax1.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
#ax1.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])

fig1.tight_layout()
plt.show()

fig1.savefig((r"C:\Users\josep\Documents\MRes mini-project 2\Figures\Ultra-stable-lasers-and-photonic-integrated-circuits-for-secure-communications\content\Figures\pgf_figures\QD_Frequency_noise.pgf"), dpi=300)

"""# ── Plot 2: Zoomed Region ───────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(4.5, 2.0), dpi=300)

# Fill background regions in zoom
ax2.fill_between([1e3, 1e8], [1e3, 1e3], [1e4, 1e4], color=colors[0], edgecolor=colors[0], label='_nolegend_')
ax2.fill_between([1e3, 1e8], [1e4, 1e4], [1e5, 1e5], color=colors[2], edgecolor=colors[2], label='_nolegend_')

# Plot only the 1500 um curve (index 1)
ax2.plot(pn[1]['x'], pn[1]['y'], color=colors[9], linewidth=1, label="1500 $\mu$m")

# Plot white noise floor of 1500 um
ax2.plot([1e6, 1e8], [fn_levels[1], fn_levels[1]], color='black', linestyle='--', linewidth=1)

# Annotate white noise level text
ax2.text(1.5e6, 6e3, "White noise level", fontsize=10, color='black')

# Scales & Labels
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1e6, 1e8)
ax2.set_ylim(1e3, 1e6)
ax2.tick_params(axis='both', which='both', direction='out', labelsize=10)

ax2.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
ax2.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])

#fig2.tight_layout()
#fig2.savefig(os.path.join(folder, "QD_Frequency_noise_zoom.png"), dpi=300)
#fig2.savefig(os.path.join(folder, "QD_Frequency_noise_zoom.pdf"))"""

print("Bowers data plots successfully generated.")
print(f"1000 um (100mA) Linewidth: {linewidths[0]:.2f} Hz, FN Level: {fn_levels[0]:.2f} Hz^2/Hz")
print(f"1500 um (160mA) Linewidth: {linewidths[1]:.2f} Hz, FN Level: {fn_levels[1]:.2f} Hz^2/Hz")

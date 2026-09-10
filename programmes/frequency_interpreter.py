import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================
filename = r"C:\Users\josep\Documents\Prompt AWS\29.07.2026\frequency_noise_spectrum.ascii"
nperseg = 2**24          # FFT segment length for Welch's method (tune for resolution vs. smoothing)
noverlap = None          # defaults to nperseg//2
f_min = 1e3             # minimum offset frequency to plot [Hz]
f_max = 1e8             # maximum offset frequency to plot [Hz]

# =============================================================================
# Load data
# =============================================================================
print("Reading header...")
with open(filename, 'r') as f:
    line1 = f.readline().strip()
    npoints = int(line1.split('//')[0].strip())
    f.readline()  # skip comment line

print(f"Number of points: {npoints}")
print("Loading data (this may take a while for >1 GB files)...")

# Load only columns 0 (time) and 2 (phase)
# Use memory-efficient loading
data = np.loadtxt(filename, skiprows=2, usecols=(0, 2), 
                  max_rows=npoints)

t_ns = data[:, 0]       # time in nanoseconds
phase_rad = data[:, 1]  # phase in radians

print(f"Data loaded. Time span: {t_ns[0]:.6f} to {t_ns[-1]:.6f} ns")

# =============================================================================
# Compute sampling rate
# =============================================================================
dt_ns = np.mean(np.diff(t_ns[:1000]))  # average timestep from first 1000 points
dt_s = 0.1441E-12 #dt_ns * 1e-9                     # convert to seconds
fs = 1.0 / dt_s                         # sampling frequency in Hz

print(f"Timestep: {dt_ns:.6e} ns")
print(f"Sampling frequency: {fs:.3e} Hz")

# =============================================================================
# Unwrap phase
# =============================================================================
print("Unwrapping phase...")
phase_unwrapped = np.unwrap(phase_rad)

# =============================================================================
# Compute instantaneous frequency deviation
# =============================================================================
print("Computing instantaneous frequency...")
# Instantaneous frequency: nu(t) = (1/(2*pi)) * d(phase)/dt
# Use central differences for better accuracy
dphase_dt = np.gradient(phase_unwrapped, dt_s)  # rad/s
inst_freq = dphase_dt / (2.0 * np.pi)           # Hz

# Remove mean frequency (we want frequency fluctuations around the carrier)
mean_freq = np.mean(inst_freq)
freq_fluctuations = inst_freq - mean_freq

print(f"Mean frequency offset from carrier: {mean_freq:.3e} Hz")
print(f"RMS frequency fluctuation: {np.std(freq_fluctuations):.3e} Hz")

# =============================================================================
# Optional: discard initial transient
# =============================================================================
# If your simulation includes start-up transients, discard them here
# For example, discard first 10% of data:
start_index = len(freq_fluctuations) // 10
freq_fluctuations = freq_fluctuations[start_index:]
print(f"Using {len(freq_fluctuations)} points after discarding initial transient")

# =============================================================================
# Compute frequency noise PSD using Welch's method
# =============================================================================
print(f"Computing PSD with nperseg={nperseg}...")
f_offset, S_nu = welch(freq_fluctuations, fs=fs, nperseg=nperseg,
                       noverlap=noverlap, window='hann',
                       scaling='density')  # units: Hz^2/Hz

# =============================================================================
# Compute beta separation line and Lorentzian linewidth
# =============================================================================
# Beta separation line: S_nu = 8*ln(2)*f / pi^2
# Points above this line contribute to the Gaussian linewidth
# The Lorentzian linewidth = pi * S_nu(white noise floor)
beta_line = 8.0 * np.log(2) * f_offset / (np.pi**2)

# Estimate white noise floor from high-frequency region (e.g., 10-100 MHz)
mask_white = (f_offset > 1e7) & (f_offset < 1e8)
if np.any(mask_white):
    white_noise_floor = np.median(S_nu[mask_white])
    lorentzian_linewidth = np.pi * white_noise_floor
    print(f"\nWhite noise floor: {white_noise_floor:.2e} Hz^2/Hz")
    print(f"Lorentzian linewidth: {lorentzian_linewidth:.2e} Hz "
          f"= {lorentzian_linewidth/1e3:.2f} kHz")
else:
    white_noise_floor = None
    print("Warning: insufficient bandwidth to estimate white noise floor")

# =============================================================================
# Plot
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Frequency noise
mask_plot = (f_offset > 0) & (f_offset >= f_min) & (f_offset <= f_max)
ax.loglog(f_offset[mask_plot], S_nu[mask_plot], 'b-', linewidth=0.5,
          label='Frequency noise $S_\\nu(f)$')

# Beta separation line
ax.loglog(f_offset[mask_plot], beta_line[mask_plot], 'k--', linewidth=1.5,
          label='$\\beta$ separation line')

# White noise floor
if white_noise_floor is not None:
    ax.axhline(white_noise_floor, color='r', linestyle=':', linewidth=1.5,
               label=f'White noise floor = {white_noise_floor:.2e} Hz²/Hz\n'
                     f'Lorentzian LW = {lorentzian_linewidth/1e3:.1f} kHz')

ax.set_xlabel('Offset frequency (Hz)', fontsize=12)
ax.set_ylabel('Frequency noise (Hz²/Hz)', fontsize=12)
ax.set_title('Single-sideband frequency noise spectrum', fontsize=13)
ax.set_xlim([f_min, f_max])
ax.set_ylim([1e2, 1e12])
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('frequency_noise_spectrum.png', dpi=150)
plt.show()

print("\nDone.")
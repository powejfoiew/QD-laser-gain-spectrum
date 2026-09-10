# ============================================================
# plot_heatmaps.py
# Plots QW and QD feedback ESP maps side by side.
# ============================================================
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ── Plotting parameters ────────────────────────────────────────────────────────
maj_tick_width = 0.8
min_tick_width = 0.8
plot_params = {
    "figure.dpi": "200",
    "axes.labelsize": 14,
    "axes.linewidth": 1.5,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.title_fontsize": 11,
    "legend.fontsize": 11,
    "xtick.major.size": 3.5,
    "xtick.major.width": maj_tick_width,
    "xtick.minor.size": 2.5,
    "xtick.minor.width": min_tick_width,
    "ytick.major.size": 3.5,
    "ytick.major.width": maj_tick_width,
    "ytick.minor.size": 2.5,
    "ytick.minor.width": min_tick_width,
    "font.family": "Times New Roman",
    "text.usetex": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
}
plt.rcParams.update(plot_params)

# ── File paths ─────────────────────────────────────────────────────────────────
QW_PATH = r'C:\Users\josep\Documents\MRes mini-project 2\Figures\experimental_heatmap\QW_feedback_map.mat'
QD_PATH = r'C:\Users\josep\Documents\MRes mini-project 2\Figures\experimental_heatmap\QD_feedback_map.mat'

# ── Load data ──────────────────────────────────────────────────────────────────
mat_qw = sio.loadmat(QW_PATH)
mat_qd = sio.loadmat(QD_PATH)

# NOTE: if unsure of keys, inspect with: print(mat_qd.keys())
ESP_map_QW = mat_qw['ESP_map_QW']
fb_ext_qw  = mat_qw['fb_ext'].ravel()

ESP_map_QD = mat_qd['ESP_map_QD']   # ← adjust key name if different
fb_ext_qd  = mat_qd['fb_ext'].ravel()

xspan_esp      = 800   # kHz, half-range of frequency-offset axis
colormap_type  = 'hot'

intensity_qw = ESP_map_QW[:, 1:]
offset_qw    = ESP_map_QW[:, 0]

intensity_qd = ESP_map_QD[:, 1:]
offset_qd    = ESP_map_QD[:, 0]

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.3, 3.5))
gs  = GridSpec(1, 2, wspace=0.50,
               left=0.07, right=0.96, top=0.70, bottom=0.18)

ax_qw = fig.add_subplot(gs[0])
ax_qd = fig.add_subplot(gs[1])

# ── Helper: draw one heatmap panel ────────────────────────────────────────────
def draw_heatmap(ax, fig, fb_ext, offset, intensity, tag, title,
                  region_labels=('I', 'II', 'IV'),
                  boundaries=(-69, -37)):
    X, Y = np.meshgrid(fb_ext, offset)
    cf = ax.contourf(X, Y, intensity, 32, cmap=colormap_type,
                     vmin=0, vmax=1.1, antialiased=True)

    # Reversed x-axis: strong → weak feedback left → right
    ax.set_xlim(fb_ext.max(), fb_ext.min())
    ax.set_ylim(-xspan_esp, xspan_esp)

    # Ticks — x start rounded to nearest 10 dB below data minimum
    x_start = int(np.floor(fb_ext.min() / 10) * 10)
    ax.set_xticks(np.arange(x_start, fb_ext.max() + 1, 10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yticks(np.arange(-1000, 1001, 500))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Axis labels — x on top to match original
    ax.set_xlabel('Feedback Strength (dB)')
    ax.set_ylabel('Frequency Offset (kHz)')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='major', direction='out', length=6, labelsize=11)
    ax.tick_params(axis='both', which='minor', direction='out', length=3)
    ax.tick_params(axis='y', which='both', right=False)

    # Spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

    # Material label above the panel
    #ax.set_title(title, pad=30, fontsize=12, fontweight='bold')

    # Inset colorbar — white styling, labelled "P"
    cax = inset_axes(ax, width='38%', height='4%', loc='upper center',
                     bbox_to_anchor=(0, -0.06, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
    cbar.set_ticks([])
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(1)
    cbar.ax.text(1.08, 0.5, 'P', transform=cbar.ax.transAxes,
                 ha='left', va='center', fontsize=12, color='white', fontweight='bold')

    # Tkach–Chraplyvy regime boundary lines
    for xb in boundaries:
        ax.axvline(xb, color='white', linestyle=':', linewidth=1.4)

    # Regime labels (adjust y_label or x positions if data range differs)
    y_label = 620
    x_lo, x_hi = fb_ext.min(), fb_ext.max()
    b1, b2 = boundaries
    x_centers = [(x_lo + b1) / 2, (b1 + b2) / 2, (b2 + x_hi) / 2]
    for xc, lbl in zip(x_centers, region_labels):
        ax.text(xc, y_label, lbl, color='white', fontsize=11,
                ha='center', va='center', fontweight='bold')

    # Panel tag
    ax.text(-0.15, 1.22, tag, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom')

# ── Draw panels ────────────────────────────────────────────────────────────────
# QW panel keeps the original three-regime labelling.
draw_heatmap(ax_qw, fig, fb_ext_qw, offset_qw, intensity_qw, '(a)', 'QW')

# QD panel (RH plot): regime labels corrected — old 'II' -> 'I', old 'IV' -> 'II'.
# Second boundary line also moved from -37 to -35.
draw_heatmap(ax_qd, fig, fb_ext_qd, offset_qd, intensity_qd, '(b)', 'QD',
             region_labels=('I', 'I', 'II'),
             boundaries=(-69, -35))

# ── Save ───────────────────────────────────────────────────────────────────────
OUT_PATH = r'C:\Users\josep\Documents\MRes mini-project 2\Figures\Ultra-stable-lasers-and-photonic-integrated-circuits-for-secure-communications\content\Figures\pgf_figures\Fig_QW_QD_heatmaps.pdf'
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
plt.show()
print('done')
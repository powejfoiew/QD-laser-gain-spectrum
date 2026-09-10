import numpy as np

# RIN caches
rin_c1 = np.load(r'c:\Users\josep\Documents\MRes mini-project 2\Figures\RIN\picwave_rin_cache.npz')
rin_c2 = np.load(r'c:\Users\josep\Documents\MRes mini-project 2\Figures\RIN\picwave_rin_cache2.npz')

rin1_db = 10 * np.log10(rin_c1['rin'])
rin2_db = 10 * np.log10(rin_c2['rin'])

print("=== Average RIN ===")
print(f"Multilevel model   - Mean RIN: {np.mean(rin1_db):.2f} dB/Hz")
print(f"Single level model - Mean RIN: {np.mean(rin2_db):.2f} dB/Hz")

# FN caches - average in 1e7 to 1e8 Hz region
fn_c1 = np.load(r'c:\Users\josep\Documents\MRes mini-project 2\Figures\Figures\picwave_fn_cache.npz')
fn_c2 = np.load(r'c:\Users\josep\Documents\MRes mini-project 2\Figures\Figures\picwave_fn_cache2.npz')

mask1 = (fn_c1['f_offset'] >= 1e7) & (fn_c1['f_offset'] <= 1e8)
mask2 = (fn_c2['f_offset'] >= 1e7) & (fn_c2['f_offset'] <= 1e8)

print()
print("=== Average Frequency Noise (10^7 - 10^8 Hz) ===")
print(f"Multilevel model   - Mean FN: {np.mean(fn_c1['S_nu'][mask1]):.4e} Hz^2/Hz")
print(f"Single level model - Mean FN: {np.mean(fn_c2['S_nu'][mask2]):.4e} Hz^2/Hz")

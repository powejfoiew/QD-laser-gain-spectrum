"""
Auger and spontaneous-emission recombination-rate fitting for the QD-DFB
gain model.

PICWave's NRADRATE/RADRATE-style material expressions need each
recombination rate written as a function of the total electron carrier
density alone. This package fits the total Auger rate (plus wetting-layer
recombination) and the total spontaneous-emission rate against total
electron density, from the converged carrier states in `gain_table/`,
producing PICWave-importable expressions.

Reuses `pauli_blocking.curve_fitting`'s registry-agnostic helpers (spectral
plot styling, polynomial fallback fit, filename slugifying) since these
curves are fit with the same general approach (rank several candidate
models by AIC, optionally split into an anchored piecewise fit) as the
Pauli-blocking-factor fits — just with a mirror-image "rising saturating
curve" model family and a different (knee/elbow) breakpoint detector,
implemented in `curve_fitting_rise.py`.
"""

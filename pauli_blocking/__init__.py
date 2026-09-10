"""
Pauli-blocking-factor fitting for the QD-DFB gain model.

PICWave cannot make one transition's capture/escape rate depend on a
*different* level's population directly, but the underlying rate-equation
model's inter-level transitions genuinely depend on the destination level's
occupation probability (Pauli blocking). This package fits each level's
steady-state occupation probability as a function of a *neighbouring*
level's population instead, from the converged carrier states in
`gain_table/`, producing PICWave-importable expressions for GS<->ES1,
ES1<->ES2 and ES2<->WL.

Modules:
    data            loads the cached gain_table/*.npz sweep and derives the
                     per-current occupation probabilities and populations
    curve_fitting   general single-curve and piecewise curve-fitting
                     utilities, reusable for other saturating quantities
                     (e.g. a future gain-ceiling or recombination-rate fit)
"""

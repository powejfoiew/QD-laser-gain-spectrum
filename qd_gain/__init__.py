"""
Quantum-dot DFB laser gain-spectrum model.

Implements the multi-level electron/hole rate-equation model of

    Gioannini & Rossetti, "Time-Domain Traveling Wave Model of Quantum Dot
    DFB Lasers," IEEE J. Sel. Top. Quantum Electron., Vol. 17, No. 5, 2011

used to pre-compute a material-gain-vs-carrier-density-vs-wavelength table
for import into PICWave (PICWave itself cannot solve the underlying
multi-level QD rate equations).

Module layout:
    parameters        physical constants and QD-DFB device/material parameters
    carrier_dynamics  charge-neutrality/Fermi-occupation helpers and the
                       time-stepped carrier-population rate-equation solver
    gain_model        optical gain and refractive-index-change calculations
                       built on a converged carrier state
"""

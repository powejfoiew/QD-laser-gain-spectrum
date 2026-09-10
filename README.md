# QD-DFB gain-spectrum model

Computes the material gain spectrum of a quantum-dot (QD) distributed-feedback
(DFB) laser as a function of injection current, by numerically integrating
the multi-level electron/hole rate-equation model of

> Gioannini & Rossetti, "Time-Domain Traveling Wave Model of Quantum Dot DFB
> Lasers," *IEEE J. Sel. Top. Quantum Electron.*, Vol. 17, No. 5, 2011.

The resulting gain-vs-carrier-density-vs-wavelength table is what gets fed
into PICWave's gain-table importer, since PICWave itself cannot solve the
underlying multi-level QD rate equations. This repository covers only that
gain-table generation step — the PICWave-facing import/export code and the
downstream result-plotting scripts live elsewhere.

## Layout

```
qd_gain/
    parameters.py
    carrier_dynamics.py
    gain_model.py
generate_gain_table.py
gain_table/
```

| Path | Contents |
| --- | --- |
| `qd_gain/parameters.py` | Physical constants, material/device parameters, and every default/fitted numeric value used below |
| `qd_gain/carrier_dynamics.py` | Charge-neutrality/Fermi-occupation helpers and `run_qd_simulation()`, the rate-equation integrator |
| `qd_gain/gain_model.py` | `gain()`, `gain_index()`, `compute_gain_spectrum()`, and `refractive_index_change()` |
| `generate_gain_table.py` | Entry point: builds the current-density sweep, runs the simulation, caches results, plots spectra |
| `gain_table/` | Cached simulation output (see [Caching](#caching)) |

Adapting this to a different QD/QW device means editing `parameters.py`
(and the `SIM_KWARGS` block in `generate_gain_table.py`, which holds the
per-run overrides used to generate the included `gain_table/` cache) —
`carrier_dynamics.py` and `gain_model.py` implement the general rate
equations and shouldn't need to change.

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`

```
pip install numpy scipy matplotlib
```

## Running it

From the project root:

```
python generate_gain_table.py
```

This will:

1. Build a 500-point, non-uniformly spaced current-density sweep
   (`generate_custom_samples`, denser near threshold).
2. For each current, integrate the rate equations to convergence
   (`run_qd_simulation`) and cache the converged carrier state to
   `gain_table/J_<current>.npz`.
3. Reload the cache and compute the material gain spectrum at each current
   (`compute_gain_spectrum`), then plot gain vs. wavelength across the
   sweep.

A full run takes on the order of tens of minutes (each of the 500 currents
requires ~190,000 integration steps, tstep = 60 fs, to converge). See
**Caching** below for how to avoid repeating this.

## Caching

`gain_table/` already holds the cached output of a full run (500 `.npz`
files, ~2 MB total) — you don't need to run anything to inspect or reuse
these results. Each file stores the converged carrier populations for one
current density (`n_e_sch`, `n_e_w`, `n_e_im`, `n_h_sch`, `n_hq_qd`, plus the
convergence time/step count), not the gain spectrum itself; the gain
spectrum is cheap to recompute from these at any time via
`compute_gain_spectrum` (step 3 above), which is why only the carrier state
is cached, not the derived spectrum.

`build_gain_table()` (in `generate_gain_table.py`) checks for an existing
`gain_table/J_<current>.npz` before running the simulation for that current,
and skips it if found. So:

- Re-running `python generate_gain_table.py` with the included `gain_table/`
  cache present will skip straight to step 3 (loading + plotting) — the slow
  step 2 only runs for currents that aren't already cached.
- To force a full recompute (e.g. after changing `parameters.py` or
  `SIM_KWARGS`), delete `gain_table/` first — otherwise stale cached states
  from the old parameters will silently be reused.
- To extend the sweep (e.g. more points, a wider current range), just call
  `build_gain_table()` with a different `J_arr`; existing cache entries are
  reused and only the new currents are computed.

## Notes on the physics parameters

`parameters.py`'s module docstring documents where its values come from and
which ones are hand-fit rather than taken directly from the reference paper
(the G&R paper doesn't give the transport-energy offsets needed to reproduce
its own reported gain curve — see thesis section 4.2.1 / Appendix 7.3).
`generate_gain_table.py`'s `CONVERSION_FACTOR` and `GAMMA_XY_FITTED`
constants are likewise empirically fitted (thesis section 4.1), not derived
in closed form.

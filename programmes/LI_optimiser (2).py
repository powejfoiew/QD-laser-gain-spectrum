import picwavelib as picw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import minimize
import os

# ---------------------------------------------------------------
# 1. PARAMETERS & BASELINE DEFINITIONS
# ---------------------------------------------------------------

# 11 Time Constant Parameters (same as simulation_looper)
TIME_PARAM_KEYS = [
    "tau_aug_GS", "tau_aug_ES1", "tau_aug_ES2",
    "tau_spon_GS", "tau_spon_ES1", "tau_spon_ES2",
    "tau_c_e_GS", "tau_c_e_ES1", "tau_c_e_ES2",
    "tau_c_e_W", "tau_r_W"
]

# 6 Rho Shape Parameters
RHO_PARAM_KEYS = [
    "rho_xm_GS", "rho_xm_ES1", "rho_xm_ES2",
    "rho_n_GS", "rho_n_ES1", "rho_n_ES2"
]

# sponbetana is the 18th free parameter (set via PICWave API, not template)
TEMPLATE_PARAM_KEYS = TIME_PARAM_KEYS + RHO_PARAM_KEYS
ALL_PARAM_KEYS = TEMPLATE_PARAM_KEYS + ["sponbetana"]

BASELINE = {
    # Time Constants
    "tau_aug_GS": 660E-12,
    "tau_aug_ES1": 275E-12,
    "tau_aug_ES2": 110E-12,
    "tau_spon_GS": 0.1 * 2.8E-9,
    "tau_spon_ES1": 0.1 * 2.8E-9,
    "tau_spon_ES2": 0.1 * 2.8E-9,
    "tau_c_e_GS": 2E-12,
    "tau_c_e_ES1": 3E-12,
    "tau_c_e_ES2": 3E-12,
    "tau_c_e_W": 1.2,
    "tau_r_W": 100,
    # Rho Shape Multipliers
    "rho_xm_GS": 1.0,
    "rho_xm_ES1": 1.0,
    "rho_xm_ES2": 1.0,
    "rho_n_GS": 1.0,
    "rho_n_ES1": 1.0,
    "rho_n_ES2": 1.0,
    # sponbetana (spontaneous emission coupling)
    "sponbetana": 0.0105
}

# L-I Curve Target (from known experimental data)
TARGET_SLOPE = 0.39766708784062943
TARGET_INTERCEPT = -1.8947947784783157

# Population targets from optimisation_results (1).csv best run
# (soft constraint to avoid straying too far from physical populations)
TARGET_GS  = 0.20
TARGET_ES1 = 0.40
TARGET_ES2 = 0.50

# Bounds
MAX_TIME_MULT = 10.0
MAX_TIME_LOG2 = np.log2(MAX_TIME_MULT)  # ~3.322

MAX_RHO_MULT = 10.0
MAX_RHO_LOG2 = np.log2(MAX_RHO_MULT)   # ~3.322

# sponbetana range: [0.001, 0.1] (baseline 0.0105)
SPONBETANA_MIN = 0.001
SPONBETANA_MAX = 0.1
SPONBETANA_LOG2_RANGE = np.log2(SPONBETANA_MAX / BASELINE["sponbetana"])  # ~3.25

# ---------------------------------------------------------------
# Score weighting: how much to prioritise L-I fit vs population
# ---------------------------------------------------------------
W_LI_SLOPE = 40.0       # Weight for slope error
W_LI_INTERCEPT = 10.0   # Weight for intercept error
W_POP = 5.0             # Weight for population deviation penalty


def get_full_params(log2_factors):
    """
    Given an 18-element vector of log2 scaling factors,
    computes the 18 free parameters, 4 derived escape times,
    and all formatted template rho placeholder variables.

    Returns (template_params, sponbetana_value).
    """
    params = {}
    for i, key in enumerate(TEMPLATE_PARAM_KEYS):
        log2_val = log2_factors[i]
        if key in TIME_PARAM_KEYS:
            mult = 2.0 ** np.clip(log2_val, -MAX_TIME_LOG2, MAX_TIME_LOG2)
        else:
            mult = 2.0 ** np.clip(log2_val, -MAX_RHO_LOG2, MAX_RHO_LOG2)
        params[key] = BASELINE[key] * mult

    # sponbetana is the last element (index 17)
    spon_log2 = log2_factors[17] if len(log2_factors) > 17 else 0.0
    sponbetana = BASELINE["sponbetana"] * 2.0 ** np.clip(
        spon_log2, -SPONBETANA_LOG2_RANGE, SPONBETANA_LOG2_RANGE
    )
    sponbetana = np.clip(sponbetana, SPONBETANA_MIN, SPONBETANA_MAX)

    # Fixed ratios derived from tau_c_e_* parameters
    params["tau_e_e_GS"]  = 7.7537 * params["tau_c_e_GS"]
    params["tau_e_e_ES1"] = 2.4390 * params["tau_c_e_ES1"]
    params["tau_e_e_ES2"] = 19.670e12 * params["tau_c_e_ES2"]
    params["tau_e_e_W"]   = 2.3915 * params["tau_c_e_W"]

    # Rho placeholder computation
    xm_gs  = params["rho_xm_GS"];  n_gs  = params["rho_n_GS"]
    xm_es1 = params["rho_xm_ES1"]; n_es1 = params["rho_n_ES1"]
    xm_es2 = params["rho_xm_ES2"]; n_es2 = params["rho_n_ES2"]

    # GS
    params["N_sat_aug_GS"]  = 2.355e17 * xm_gs
    params["exp_aug_GS"]    = 0.85 * n_gs
    params["N_sat_sp_GS"]   = 2.33523e17 * xm_gs
    params["exp_sp_GS"]     = 0.425362 * n_gs
    params["N_brk1_c_GS"]   = 1.994170e17 * xm_gs
    params["N_scl1_c_GS"]   = 2.127919e17 * xm_gs
    params["N_tw1_c_GS"]    = 3.750565e17 * xm_gs
    params["N_brk2_c_GS"]   = 2.010410e17 * xm_gs
    params["N_02_c_GS"]     = 8.450412e16 * xm_gs
    params["N_scl2_c_GS"]   = 1.165369e17 * xm_gs
    params["N_tw2_c_GS"]    = 3.296260e16 * xm_gs

    # ES1
    params["N_sat_aug_ES1"] = 4.686e17 * xm_es1
    params["exp_aug_ES1"]   = 0.75 * n_es1
    params["N_sat_sp_ES1"]  = 4.82866e17 * xm_es1
    params["exp_sp_ES1"]    = 0.510774 * n_es1
    params["N_scl1_c_ES1"]  = 1.725671e17 * xm_es1
    params["N_brk2_c_ES1"]  = 4.088225e17 * xm_es1
    params["N_02_c_ES1"]    = 1.640052e16 * xm_es1
    params["N_scl2_c_ES1"]  = 3.924220e17 * xm_es1
    params["N_tw2_c_ES1"]   = 5.994082e16 * xm_es1

    # ES2
    params["N_sat_aug_ES2"] = 7.069054e17 * xm_es2
    params["exp_aug_ES2"]   = 0.6097 * n_es2
    params["N_sat_sp_ES2"]  = 7.37623e17 * xm_es2
    params["exp_sp_ES2"]    = 0.567465 * n_es2
    params["N_scl1_c_ES2"]  = 4.076835e16 * xm_es2

    return params, sponbetana


def write_custom_refbase(
    template_file="custom_refbase_template.txt",
    output_file="custom_refbase.mat",
    **params
):
    with open(template_file, "r", encoding="utf-8") as f:
        text = f.read()
    for name in sorted(params.keys(), key=len, reverse=True):
        text = text.replace(name, f"{params[name]:.4e}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------
# 2. SIMULATION RUNNER & COMBINED L-I + POPULATION OBJECTIVE
# ---------------------------------------------------------------
def run_simulation(app, template_params, sponbetana):
    """
    Writes template, sets sponbetana, runs PICWave, extracts L-I data
    and carrier populations, and scores the result.
    """
    write_custom_refbase(**template_params)

    circuit = app.getsubnode("subnodes[1].subnodes[2]")
    circuit.setmaterbase("[PrjDir]\\custom_refbase.mat")

    # Set sponbetana via PICWave API
    app.subnodes[1].subnodes[2].mcdevice.objects[1].startchange()
    app.subnodes[1].subnodes[2].mcdevice.objects[1].sponbetana = sponbetana
    app.subnodes[1].subnodes[2].mcdevice.objects[1].finishchange()

    try:
        circuit.tdcalculator.run()
        output_power        = circuit.tdcalculator.rundata.instrumentlist[2].data
        current_data        = circuit.tdcalculator.rundata.instrumentlist[1].data
        gs_carrier_density  = circuit.tdcalculator.rundata.instrumentlist[4].data
        es1_carrier_density = circuit.tdcalculator.rundata.instrumentlist[8].data
        es2_carrier_density = circuit.tdcalculator.rundata.instrumentlist[7].data

        density_data = gs_carrier_density[1][1:]
        if len(density_data) > 0 and density_data[-1] == 0:
            crashed = True
        else:
            crashed = False

    except Exception as e:
        print(f"Simulation execution error: {e}")
        crashed = True

    if crashed:
        return {
            "crashed": True,
            "final_gs": np.nan, "final_es1": np.nan, "final_es2": np.nan,
            "max_power": np.nan, "mean_power": np.nan,
            "fitted_slope": np.nan, "fitted_intercept": np.nan, "li_r2": np.nan,
            "slope_err": np.nan, "intercept_err": np.nan,
            "score": -1e6,
            "time_series": None
        }

    # Extract arrays
    current_vals = np.array(current_data[2][1:], dtype=float)
    power_vals   = np.array(output_power[2][1:], dtype=float)
    gs_vals      = np.array(gs_carrier_density[2][1:], dtype=float)
    es1_vals     = np.array(es1_carrier_density[2][1:], dtype=float)
    es2_vals     = np.array(es2_carrier_density[2][1:], dtype=float)

    # Sort and deduplicate data points to maintain alignment across all variables
    unique_indices = np.unique(current_vals, return_index=True)[1]
    current_vals = current_vals[unique_indices]
    power_vals   = power_vals[unique_indices]
    gs_vals      = gs_vals[unique_indices]
    es1_vals     = es1_vals[unique_indices]
    es2_vals     = es2_vals[unique_indices]

    # Re-sort to maintain strictly increasing current order
    sort_order = np.argsort(current_vals)
    current_vals = current_vals[sort_order]
    power_vals   = power_vals[sort_order]
    gs_vals      = gs_vals[sort_order]
    es1_vals     = es1_vals[sort_order]
    es2_vals     = es2_vals[sort_order]

    # Find the index closest to 50mA
    idx_50 = np.argmin(np.abs(current_vals - 50.0))

    # Normalise carrier densities evaluated at 50mA (in units of 10^18 cm^-3)
    final_gs  = gs_vals[idx_50]  if gs_vals[idx_50]  < 1e10 else gs_vals[idx_50]  / 1e18
    final_es1 = es1_vals[idx_50] if es1_vals[idx_50] < 1e10 else es1_vals[idx_50] / 1e18
    final_es2 = es2_vals[idx_50] if es2_vals[idx_50] < 1e10 else es2_vals[idx_50] / 1e18

    max_power  = np.max(power_vals)
    mean_power = np.mean(power_vals)

    # -----------------------------------------------------------
    # L-I LINEAR FIT (between x-intercept and 60 mA)
    # -----------------------------------------------------------
    end_idx = np.argmin(np.abs(current_vals - 60.0))
    if end_idx == 0:
        end_idx = len(current_vals)

    # Initial guess for start index: power > 5% of max power
    above_thresh = np.where(power_vals > 0.05 * max_power)[0]
    if len(above_thresh) > 0:
        start_idx = above_thresh[0]
    else:
        start_idx = 0

    if start_idx >= end_idx:
        start_idx = 0

    fitted_slope = 0.0
    fitted_intercept = 0.0
    li_r2 = 0.0

    # Iteratively solve for the x-intercept and adjust the start index
    for iteration in range(5):
        if start_idx >= end_idx or (end_idx - start_idx) < 3:
            break

        fit_current = current_vals[start_idx:end_idx]
        fit_power   = power_vals[start_idx:end_idx]

        res = linregress(fit_current, fit_power)

        if res.slope > 0.01:
            x_intercept = -res.intercept / res.slope
            new_start_indices = np.where(current_vals >= x_intercept)[0]
            if len(new_start_indices) > 0:
                new_start_idx = new_start_indices[0]
                if new_start_idx < end_idx - 2:
                    if new_start_idx == start_idx:
                        # Converged!
                        fitted_slope = res.slope
                        fitted_intercept = res.intercept
                        li_r2 = res.rvalue ** 2
                        break
                    start_idx = new_start_idx
                    fitted_slope = res.slope
                    fitted_intercept = res.intercept
                    li_r2 = res.rvalue ** 2
                    continue

        fitted_slope = res.slope
        fitted_intercept = res.intercept
        li_r2 = res.rvalue ** 2
        break

    slope_err = abs(fitted_slope - TARGET_SLOPE)
    intercept_err = abs(fitted_intercept - TARGET_INTERCEPT)

    # -----------------------------------------------------------
    # COMBINED COST FUNCTION
    # -----------------------------------------------------------
    li_score = -(W_LI_SLOPE * slope_err + W_LI_INTERCEPT * intercept_err)

    pop_err = (abs(final_gs - TARGET_GS) +
               abs(final_es1 - TARGET_ES1) +
               abs(final_es2 - TARGET_ES2))
    pop_penalty = -W_POP * pop_err

    score = li_score + pop_penalty

    time_series = {
        "current_data": current_data,
        "output_power": output_power,
        "gs_carrier_density": gs_carrier_density,
        "es1_carrier_density": es1_carrier_density,
        "es2_carrier_density": es2_carrier_density,
    }

    return {
        "crashed": False,
        "final_gs": final_gs, "final_es1": final_es1, "final_es2": final_es2,
        "max_power": max_power, "mean_power": mean_power,
        "fitted_slope": fitted_slope, "fitted_intercept": fitted_intercept,
        "li_r2": li_r2,
        "slope_err": slope_err, "intercept_err": intercept_err,
        "score": score,
        "time_series": time_series
    }


# ---------------------------------------------------------------
# 3. OPTIMIZATION CONTROLLER
# ---------------------------------------------------------------
def latin_hypercube_sampling(n_samples, n_params, seed=42):
    np.random.seed(seed)
    result = np.empty((n_samples, n_params))
    for j in range(n_params):
        if j < len(TEMPLATE_PARAM_KEYS):
            key = TEMPLATE_PARAM_KEYS[j]
            max_b = MAX_TIME_LOG2 if key in TIME_PARAM_KEYS else MAX_RHO_LOG2
        else:
            max_b = SPONBETANA_LOG2_RANGE  # sponbetana
        perm = np.random.permutation(n_samples)
        u = (perm + np.random.uniform(size=n_samples)) / n_samples
        result[:, j] = -max_b + 2.0 * max_b * u
    return result


def load_previous_best(csv_paths=["LI_optimisation_results.csv"]):
    """Load best from previous L-I optimisation run."""
    csv_path = next((p for p in csv_paths if os.path.exists(p)), None)
    if csv_path is None:
        return None, None

    try:
        prev_df = pd.read_csv(csv_path)
        valid_df = prev_df[prev_df["crashed"] == False]
        if valid_df.empty:
            return prev_df, None

        best_row = valid_df.sort_values(by="score", ascending=False).iloc[0]
        start_log2_vec = []
        for k in ALL_PARAM_KEYS:
            val = best_row[k] if k in best_row else BASELINE[k]
            ratio = val / BASELINE[k]
            if k in TIME_PARAM_KEYS:
                max_b = MAX_TIME_LOG2
            elif k == "sponbetana":
                max_b = SPONBETANA_LOG2_RANGE
            else:
                max_b = MAX_RHO_LOG2
            start_log2_vec.append(np.clip(np.log2(max(ratio, 1e-12)), -max_b, max_b))

        return prev_df, np.array(start_log2_vec)
    except Exception as e:
        print(f"Warning: Could not read L-I results CSV ({e}). Starting fresh.")
        return None, None


def load_population_seed(csv_path="optimisation_results (1).csv"):
    """
    Load best parameter set from previous population optimisation
    as a good starting point (seed) for L-I optimisation.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        valid = df[df["crashed"] == False]
        if valid.empty:
            return None
        best = valid.sort_values("score", ascending=False).iloc[0]
        seed_vec = []
        for k in ALL_PARAM_KEYS:
            val = best[k] if k in best else BASELINE[k]
            ratio = val / BASELINE[k]
            if k in TIME_PARAM_KEYS:
                max_b = MAX_TIME_LOG2
            elif k == "sponbetana":
                max_b = SPONBETANA_LOG2_RANGE
            else:
                max_b = MAX_RHO_LOG2
            seed_vec.append(np.clip(np.log2(max(ratio, 1e-12)), -max_b, max_b))
        return np.array(seed_vec)
    except Exception:
        return None


def main():
    MAX_NEW_ITERATIONS = 200
    LHS_SAMPLES = 60
    CSV_OUT = "LI_optimisation_results.csv"

    prev_df, start_vec = load_previous_best([CSV_OUT])
    pop_seed = load_population_seed()

    if prev_df is not None:
        start_iter_offset = int(prev_df["iteration"].max())
        print(f"Found previous L-I optimisation with {start_iter_offset} iterations.")
        history = prev_df.to_dict("records")
    else:
        start_iter_offset = 0
        history = []

    print("Connecting to PICWave...")
    app = picw.connect_to_picwave()
    print("Connected successfully.")

    best_score = -np.inf
    best_result = None
    best_time_series = None

    for h in history:
        if not h.get("crashed", True) and not np.isnan(h.get("score", np.nan)):
            if h["score"] > best_score:
                best_score = h["score"]

    new_iteration_count = 0
    total_budget = start_iter_offset + MAX_NEW_ITERATIONS

    def evaluate_vector(log2_vec):
        nonlocal new_iteration_count, best_score, best_result, best_time_series
        if new_iteration_count >= MAX_NEW_ITERATIONS:
            return 1e6

        new_iteration_count += 1
        current_iter = start_iter_offset + new_iteration_count
        template_params, sponbetana = get_full_params(log2_vec)

        print(f"\n--- Iteration {current_iter}/{total_budget} ---")
        res = run_simulation(app, template_params, sponbetana)

        if res["crashed"]:
            print("Status: CRASHED")
        else:
            print(f"Status: SUCCESS | Slope: {res['fitted_slope']:.4f} (T:{TARGET_SLOPE:.4f}) | "
                  f"Intercept: {res['fitted_intercept']:.4f} (T:{TARGET_INTERCEPT:.4f}) | "
                  f"R2: {res['li_r2']:.4f} | "
                  f"GS: {res['final_gs']:.3f} ES1: {res['final_es1']:.3f} ES2: {res['final_es2']:.3f} | "
                  f"sponbetana: {sponbetana:.4f} | Score: {res['score']:.4f}")

        row = {
            "iteration": current_iter,
            "crashed": res["crashed"],
            "score": res["score"] if not res["crashed"] else np.nan,
            "fitted_slope": res["fitted_slope"],
            "fitted_intercept": res["fitted_intercept"],
            "li_r2": res["li_r2"],
            "slope_err": res["slope_err"],
            "intercept_err": res["intercept_err"],
            "final_gs": res["final_gs"],
            "final_es1": res["final_es1"],
            "final_es2": res["final_es2"],
            "max_power": res["max_power"],
            "mean_power": res["mean_power"],
        }
        # Log all 17 template parameters
        for k in TEMPLATE_PARAM_KEYS:
            row[k] = template_params[k]
        row["sponbetana"] = sponbetana

        history.append(row)

        if not res["crashed"] and res["score"] > best_score:
            best_score = res["score"]
            best_result = (current_iter, template_params, sponbetana, res)
            best_time_series = res["time_series"]

        pd.DataFrame(history).to_csv(CSV_OUT, index=False)
        return -res["score"]

    # -----------------------------------------------------------
    # PHASE 1: EXPLORATION
    # -----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 1: Exploration ({LHS_SAMPLES} Iterations)")
    print(f"{'='*60}")

    queue = []
    # Seed from previous L-I run
    if start_vec is not None:
        queue.append(start_vec)
    # Seed from population optimiser results (best known physical params)
    if pop_seed is not None:
        queue.append(pop_seed)
        print("Loaded population-optimised seed as starting point.")
    # Baseline
    queue.append(np.zeros(len(ALL_PARAM_KEYS)))

    # LHS exploration
    needed = max(0, LHS_SAMPLES - len(queue))
    if needed > 0:
        lhs = latin_hypercube_sampling(needed, len(ALL_PARAM_KEYS), seed=555)
        queue.extend(lhs)

    for sample in queue[:LHS_SAMPLES]:
        if new_iteration_count >= MAX_NEW_ITERATIONS:
            break
        evaluate_vector(sample)

    # -----------------------------------------------------------
    # PHASE 2: NELDER-MEAD REFINEMENT
    # -----------------------------------------------------------
    remaining = MAX_NEW_ITERATIONS - new_iteration_count
    if remaining > 0:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Nelder-Mead Refinement ({remaining} Max Iterations)")
        print(f"{'='*60}")

        valid_hist = [h for h in history
                      if not h.get("crashed", True)
                      and not np.isnan(h.get("score", np.nan))]
        if valid_hist:
            best_entry = max(valid_hist, key=lambda x: x["score"])
            p2_vec = []
            for k in ALL_PARAM_KEYS:
                val = best_entry.get(k, BASELINE[k])
                ratio = val / BASELINE[k]
                if k in TIME_PARAM_KEYS:
                    max_b = MAX_TIME_LOG2
                elif k == "sponbetana":
                    max_b = SPONBETANA_LOG2_RANGE
                else:
                    max_b = MAX_RHO_LOG2
                p2_vec.append(np.clip(np.log2(max(ratio, 1e-12)), -max_b, max_b))
            p2_vec = np.array(p2_vec)
        else:
            p2_vec = np.zeros(len(ALL_PARAM_KEYS))

        bounds = []
        for k in ALL_PARAM_KEYS:
            if k in TIME_PARAM_KEYS:
                bounds.append((-MAX_TIME_LOG2, MAX_TIME_LOG2))
            elif k == "sponbetana":
                bounds.append((-SPONBETANA_LOG2_RANGE, SPONBETANA_LOG2_RANGE))
            else:
                bounds.append((-MAX_RHO_LOG2, MAX_RHO_LOG2))

        try:
            minimize(evaluate_vector, p2_vec, method="Nelder-Mead",
                     bounds=bounds,
                     options={"maxiter": remaining, "adaptive": True})
        except Exception as e:
            print(f"Optimization finished or interrupted: {e}")

    # -----------------------------------------------------------
    # DISCONNECT & SUMMARY
    # -----------------------------------------------------------
    print("\nDisconnecting PICWave...")
    picw.disconnect_picwave()

    df = pd.DataFrame(history)
    df.to_csv(CSV_OUT, index=False)
    print(f"Results saved to {CSV_OUT}")

    valid_df = df[df["crashed"] == False]
    if not valid_df.empty:
        best_row = valid_df.sort_values("score", ascending=False).iloc[0]
        best_iter = int(best_row["iteration"])

        print(f"\n{'='*60}")
        print(f"BEST RESULT (Iteration {best_iter})")
        print(f"{'='*60}")
        print(f"  Fitted Slope     : {best_row['fitted_slope']:.6f}  (Target: {TARGET_SLOPE:.6f})")
        print(f"  Fitted Intercept : {best_row['fitted_intercept']:.6f}  (Target: {TARGET_INTERCEPT:.6f})")
        print(f"  L-I R^2          : {best_row['li_r2']:.6f}")
        print(f"  Slope Error      : {best_row['slope_err']:.6f}")
        print(f"  Intercept Error  : {best_row['intercept_err']:.6f}")
        print(f"  GS: {best_row['final_gs']:.4f}  ES1: {best_row['final_es1']:.4f}  ES2: {best_row['final_es2']:.4f}")
        print(f"  sponbetana       : {best_row['sponbetana']:.6f}")
        print(f"  Score            : {best_row['score']:.6f}")

    # -----------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------
    if not valid_df.empty:
        # 1. Score convergence
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(valid_df["iteration"], valid_df["score"], "bo-", markersize=3)
        ax.set_title("L-I Optimisation: Score vs Iteration")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Score")
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout(); fig.savefig("LI_optimisation_convergence.png", dpi=200)

        # 2. Slope & intercept convergence
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(valid_df["iteration"], valid_df["fitted_slope"], "b-", label="Fitted Slope")
        ax1.axhline(TARGET_SLOPE, color="r", linestyle="--", label=f"Target ({TARGET_SLOPE:.4f})")
        ax1.set_ylabel("Slope"); ax1.legend(); ax1.grid(True, linestyle=":", alpha=0.6)
        ax2.plot(valid_df["iteration"], valid_df["fitted_intercept"], "b-", label="Fitted Intercept")
        ax2.axhline(TARGET_INTERCEPT, color="r", linestyle="--", label=f"Target ({TARGET_INTERCEPT:.4f})")
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("Intercept"); ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)
        fig.suptitle("L-I Fit Parameters vs Iteration", fontweight="bold")
        fig.tight_layout(); fig.savefig("LI_fit_convergence.png", dpi=200)

        # 3. Best run L-I curve
        if best_time_series is not None:
            current_arr = np.array(best_time_series["current_data"][2][1:], dtype=float)
            power_arr   = np.array(best_time_series["output_power"][2][1:], dtype=float)

            # Sort and deduplicate
            unique_indices = np.unique(current_arr, return_index=True)[1]
            current_arr = current_arr[unique_indices]
            power_arr   = power_arr[unique_indices]
            sort_order = np.argsort(current_arr)
            current_arr = current_arr[sort_order]
            power_arr   = power_arr[sort_order]

            # Calculate fit range
            f_slope = best_row["fitted_slope"]
            f_intercept = best_row["fitted_intercept"]
            x_int = -f_intercept / f_slope if f_slope > 0.01 else current_arr[0]
            
            end_val = 60.0
            fit_mask = (current_arr >= x_int) & (current_arr <= end_val)
            
            fig, ax = plt.subplots(figsize=(9, 6))
            # Plot all data in light blue
            ax.plot(current_arr, power_arr, "o", color="#b0c4de", markersize=4, label="Simulation (All Data)", alpha=0.5)
            # Highlight fitted range in dark blue
            if np.any(fit_mask):
                ax.plot(current_arr[fit_mask], power_arr[fit_mask], "bo", markersize=4, label="Simulation (Fitted Range)")
                
                # Plot lines over the fit range
                fit_x = current_arr[fit_mask]
                ax.plot(fit_x, f_slope * fit_x + f_intercept, "r-", linewidth=2,
                        label=f"Fitted (slope={f_slope:.4f}, int={f_intercept:.4f})")
                ax.plot(fit_x, TARGET_SLOPE * fit_x + TARGET_INTERCEPT, "g--", linewidth=2,
                        label=f"Target (slope={TARGET_SLOPE:.4f}, int={TARGET_INTERCEPT:.4f})")
            
            ax.set_title(f"Best L-I Curve (Iteration {best_iter})", fontweight="bold")
            ax.set_xlabel("Current (mA)")
            ax.set_ylabel("Output Power (mW)")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            fig.tight_layout()
            fig.savefig("LI_curve_fit.png", dpi=300)

    plt.show()

if __name__ == "__main__":
    main()
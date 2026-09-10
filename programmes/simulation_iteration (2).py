import picwavelib as picw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os

# ---------------------------------------------------------------
# 1. PARAMETERS & BASELINE DEFINITIONS
# ---------------------------------------------------------------

# 11 Time Constant Parameters
TIME_PARAM_KEYS = [
    "tau_aug_GS", "tau_aug_ES1", "tau_aug_ES2",
    "tau_spon_GS", "tau_spon_ES1", "tau_spon_ES2",
    "tau_c_e_GS", "tau_c_e_ES1", "tau_c_e_ES2",
    "tau_c_e_W", "tau_r_W"
]

# 6 Rho Shape Parameters (Ceiling & Exponent Multipliers)
RHO_PARAM_KEYS = [
    "rho_xm_GS", "rho_xm_ES1", "rho_xm_ES2",
    "rho_n_GS", "rho_n_ES1", "rho_n_ES2"
]

PARAM_KEYS = TIME_PARAM_KEYS + RHO_PARAM_KEYS

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
    # Rho Shape Multipliers (Default 1.0)
    "rho_xm_GS": 1.0,
    "rho_xm_ES1": 1.0,
    "rho_xm_ES2": 1.0,
    "rho_n_GS": 1.0,
    "rho_n_ES1": 1.0,
    "rho_n_ES2": 1.0
}

# Target physical carrier densities (converted to 10^18 cm^-3 units)
# Desired ratio: GS : ES1 : ES2 = 1 : 2 : 3
TARGET_GS  = 2.3545338654262518e17 / 1e18 # ~0.23545
TARGET_ES1 = 4.6879006649710240e17 / 1e18 # ~0.46879
TARGET_ES2 = 7.0683475708220340e17 / 1e18 # ~0.70683

# Bounds:
# Time constants: Up to 10x range (multiplier range [0.1x, 10.0x])
MAX_TIME_MULT = 10.0
MAX_TIME_LOG2 = np.log2(MAX_TIME_MULT) # ~3.322

# Rho shape parameters: Controlled range [0.5x, 2.0x] to preserve physical curve continuity
MAX_RHO_MULT = 2.0
MAX_RHO_LOG2 = np.log2(MAX_RHO_MULT) # 1.0

def get_full_params(log2_factors):
    """
    Given a 17-element vector of log2 scaling factors,
    computes the 17 free parameters, 4 derived fixed-ratio escape times,
    and all formatted template rho placeholder variables.
    """
    params = {}
    for i, key in enumerate(PARAM_KEYS):
        log2_val = log2_factors[i]
        if key in TIME_PARAM_KEYS:
            mult = 2.0 ** np.clip(log2_val, -MAX_TIME_LOG2, MAX_TIME_LOG2)
        else:
            mult = 2.0 ** np.clip(log2_val, -MAX_RHO_LOG2, MAX_RHO_LOG2)
        params[key] = BASELINE[key] * mult

    # Fixed ratios derived from tau_c_e_* parameters
    params["tau_e_e_GS"]  = 7.7537 * params["tau_c_e_GS"]
    params["tau_e_e_ES1"] = 2.4390 * params["tau_c_e_ES1"]
    params["tau_e_e_ES2"] = 19.670e12 * params["tau_c_e_ES2"]
    params["tau_e_e_W"]   = 2.3915 * params["tau_c_e_W"]

    # Compute template placeholders for Rho Expressions (proportional carrier density scaling)
    xm_gs = params["rho_xm_GS"]
    n_gs  = params["rho_n_GS"]
    
    xm_es1 = params["rho_xm_ES1"]
    n_es1  = params["rho_n_ES1"]
    
    xm_es2 = params["rho_xm_ES2"]
    n_es2  = params["rho_n_ES2"]

    # GS Placeholders (Smooth Weibull Saturation Models)
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

    # ES1 Placeholders
    params["N_sat_aug_ES1"] = 4.686e17 * xm_es1
    params["exp_aug_ES1"]   = 0.75 * n_es1
    params["N_sat_sp_ES1"]  = 4.82866e17 * xm_es1
    params["exp_sp_ES1"]    = 0.510774 * n_es1

    params["N_scl1_c_ES1"]  = 1.725671e17 * xm_es1

    params["N_brk2_c_ES1"]  = 4.088225e17 * xm_es1
    params["N_02_c_ES1"]    = 1.640052e16 * xm_es1
    params["N_scl2_c_ES1"]  = 3.924220e17 * xm_es1
    params["N_tw2_c_ES1"]   = 5.994082e16 * xm_es1

    # ES2 Placeholders
    params["N_sat_aug_ES2"] = 7.069054e17 * xm_es2
    params["exp_aug_ES2"]   = 0.6097 * n_es2
    params["N_sat_sp_ES2"]  = 7.37623e17 * xm_es2
    params["exp_sp_ES2"]    = 0.567465 * n_es2

    params["N_scl1_c_ES2"]  = 4.076835e16 * xm_es2

    return params

def write_custom_refbase(
    template_file="custom_refbase_template.txt",
    output_file="custom_refbase.mat",
    **params
):
    with open(template_file, "r", encoding="utf-8") as f:
        text = f.read()
    # Replace longest names first to prevent partial replacements
    for name in sorted(params.keys(), key=len, reverse=True):
        text = text.replace(name, f"{params[name]:.4e}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

# ---------------------------------------------------------------
# 2. SIMULATION RUNNER & RATIO-BASED OBJECTIVE FUNCTION (1:2:3)
# ---------------------------------------------------------------
def run_simulation(app, params):
    """
    Writes custom_refbase.mat, runs PICWave simulation, and evaluates results.
    Returns dict with metrics and raw time series.
    """
    write_custom_refbase(**params)
    
    circuit = app.getsubnode("subnodes[1].subnodes[2]")
    circuit.setmaterbase("[PrjDir]\\custom_refbase.mat")
    
    try:
        circuit.tdcalculator.run()
        output_power        = circuit.tdcalculator.rundata.instrumentlist[2].data
        gs_carrier_density  = circuit.tdcalculator.rundata.instrumentlist[4].data
        sch_carrier_density = circuit.tdcalculator.rundata.instrumentlist[5].data
        wl_carrier_density  = circuit.tdcalculator.rundata.instrumentlist[6].data
        es2_carrier_density = circuit.tdcalculator.rundata.instrumentlist[7].data
        es1_carrier_density = circuit.tdcalculator.rundata.instrumentlist[8].data

        density_data = gs_carrier_density[1][1:]

        # Check if solver crashed (dropped to 0 before finishing)
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
            "final_gs": np.nan,
            "final_es1": np.nan,
            "final_es2": np.nan,
            "max_power": np.nan,
            "mean_power": np.nan,
            "lasing_duration": np.nan,
            "score": -1e6, # penalty score for optimizer
            "time_series": None
        }

    power_time = np.array(output_power[1][1:])
    power_vals = np.array(output_power[2][1:])
    gs_vals    = np.array(gs_carrier_density[2][1:])
    es1_vals   = np.array(es1_carrier_density[2][1:])
    es2_vals   = np.array(es2_carrier_density[2][1:])
    
    # Normalize carrier density to 10^18 cm^-3 if raw values (>1e10) are returned
    gs_series  = gs_vals  if gs_vals[-1]  < 1e10 else gs_vals  / 1e18
    es1_series = es1_vals if es1_vals[-1] < 1e10 else es1_vals / 1e18
    es2_series = es2_vals if es2_vals[-1] < 1e10 else es2_vals / 1e18

    final_gs  = gs_series[-1]
    final_es1 = es1_series[-1]
    final_es2 = es2_series[-1]

    max_power = np.max(power_vals)
    mean_power = np.mean(power_vals)

    # Calculate duration where power >= 10 (lasing condition)
    lasing_mask = power_vals >= 10.0
    if np.any(lasing_mask):
        lasing_time_points = power_time[lasing_mask]
        lasing_duration = lasing_time_points[-1] - lasing_time_points[0]
    else:
        lasing_duration = 0.0

    # -----------------------------------------------------------
    # SMOOTHNESS & OVERSHOOT PENALTY
    # -----------------------------------------------------------
    # 1. Transient Overshoot: Peak density exceeding final steady-state density
    gs_overshoot  = max(0.0, np.max(gs_series) - final_gs)
    es1_overshoot = max(0.0, np.max(es1_series) - final_es1)
    es2_overshoot = max(0.0, np.max(es2_series) - final_es2)

    # 2. Total Variation (TV) Excess: Penalizes oscillations / non-monotonicity
    gs_excess_tv  = np.sum(np.abs(np.diff(gs_series))) - abs(final_gs - gs_series[0])
    es1_excess_tv = np.sum(np.abs(np.diff(es1_series))) - abs(final_es1 - es1_series[0])
    es2_excess_tv = np.sum(np.abs(np.diff(es2_series))) - abs(final_es2 - es2_series[0])

    W_OVERSHOOT = 5.0
    W_SMOOTH    = 2.0

    smoothness_penalty = W_OVERSHOOT * (2.0 * gs_overshoot + es1_overshoot + es2_overshoot) + \
                         W_SMOOTH * (2.0 * max(0.0, gs_excess_tv) + max(0.0, es1_excess_tv) + max(0.0, es2_excess_tv))

    # -----------------------------------------------------------
    # REFINED OBJECTIVE FUNCTION: 1 : 2 : 3 STATE RATIO MATCHING + SMOOTHNESS
    # -----------------------------------------------------------
    safe_gs = max(final_gs, 1e-5)

    ratio_es1_gs = final_es1 / safe_gs
    ratio_es2_gs = final_es2 / safe_gs

    # Ratio errors relative to target ratios 2.0 and 3.0
    ratio_err_es1 = abs(ratio_es1_gs - 2.0)
    ratio_err_es2 = abs(ratio_es2_gs - 3.0)

    # Anchor overall density scale to physical regime (~0.2355 for GS)
    scale_err = abs(final_gs - TARGET_GS)

    # Primary score focused strictly on 1:2:3 state population ratios, physical scale, and smoothness
    score = -50.0 * (ratio_err_es1 + ratio_err_es2) - 20.0 * scale_err - smoothness_penalty

    time_series = {
        "gs_carrier_density": gs_carrier_density,
        "sch_carrier_density": sch_carrier_density,
        "wl_carrier_density": wl_carrier_density,
        "es2_carrier_density": es2_carrier_density,
        "es1_carrier_density": es1_carrier_density,
        "output_power": output_power
    }

    return {
        "crashed": False,
        "final_gs": final_gs,
        "final_es1": final_es1,
        "final_es2": final_es2,
        "max_power": max_power,
        "mean_power": mean_power,
        "lasing_duration": lasing_duration,
        "score": score,
        "time_series": time_series
    }

# ---------------------------------------------------------------
# 3. OPTIMIZATION CONTROLLER (300 ITERATIONS RUN)
# ---------------------------------------------------------------
def latin_hypercube_sampling(n_samples, n_params, seed=42):
    """Generates Latin Hypercube samples scaled to appropriate parameter bounds."""
    np.random.seed(seed)
    result = np.empty((n_samples, n_params))
    for j in range(n_params):
        perm = np.random.permutation(n_samples)
        u = (perm + np.random.uniform(size=n_samples)) / n_samples
        key = PARAM_KEYS[j]
        max_b = MAX_TIME_LOG2 if key in TIME_PARAM_KEYS else MAX_RHO_LOG2
        result[:, j] = -max_b + 2.0 * max_b * u
    return result

def load_previous_best(csv_paths=["optimisation_results.csv", "optimisation_results (1).csv", "optimisation_results_new.csv"]):
    """
    Loads previous optimization CSV if available, and extracts the best performing
    parameter vector in log2-space as a starting point.
    """
    csv_path = None
    for p in csv_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if csv_path is None:
        return None, None

    try:
        prev_df = pd.read_csv(csv_path)
        valid_df = prev_df[prev_df["crashed"] == False]
        if valid_df.empty:
            return prev_df, None
        
        # Sort by score to pick best starting point
        best_row = valid_df.sort_values(by=["score", "final_gs"], ascending=[False, False]).iloc[0]
        
        start_log2_vec = []
        for k in PARAM_KEYS:
            if k in best_row:
                val = best_row[k]
                ratio = val / BASELINE[k]
                log2_ratio = np.log2(ratio)
            else:
                log2_ratio = 0.0 # Default baseline if key was not in previous run
                
            max_b = MAX_TIME_LOG2 if k in TIME_PARAM_KEYS else MAX_RHO_LOG2
            start_log2_vec.append(np.clip(log2_ratio, -max_b, max_b))
            
        return prev_df, np.array(start_log2_vec)
    except Exception as e:
        print(f"Warning: Could not read previous results CSV ({e}). Starting fresh.")
        return None, None

def main():
    MAX_NEW_ITERATIONS = 300 # Set for 300-iteration run
    LHS_SAMPLES = 40

    prev_df, start_vec = load_previous_best()
    
    if prev_df is not None:
        start_iter_offset = int(prev_df["iteration"].max())
        print(f"Found previous optimization run with {start_iter_offset} iterations.")
        if start_vec is not None:
            print("Loaded best parameter set from previous run as starting point.")
        history = prev_df.to_dict("records")
    else:
        start_iter_offset = 0
        history = []
        
    print("Connecting to PICWave...")
    app = picw.connect_to_picwave()
    print("Connected successfully.")

    best_result = None
    best_score = -np.inf
    best_time_series = None

    # Determine best score from existing history
    for h in history:
        if not h.get("crashed", True) and not np.isnan(h.get("score", np.nan)):
            if h["score"] > best_score:
                best_score = h["score"]

    new_iteration_count = 0
    total_iteration_budget = start_iter_offset + MAX_NEW_ITERATIONS

    def evaluate_vector(log2_vec):
        nonlocal new_iteration_count, best_score, best_result, best_time_series
        if new_iteration_count >= MAX_NEW_ITERATIONS:
            return 1e6 # stop optimizer if budget reached

        new_iteration_count += 1
        current_total_iter = start_iter_offset + new_iteration_count
        params = get_full_params(log2_vec)
        
        print(f"\n--- Iteration {current_total_iter}/{total_iteration_budget} (Run Iter {new_iteration_count}/{MAX_NEW_ITERATIONS}) ---")
        res = run_simulation(app, params)

        if res["crashed"]:
            print("Status: CRASHED (Ground state carrier density dropped to 0 or simulation error)")
        else:
            r1 = res['final_es1'] / max(res['final_gs'], 1e-5)
            r2 = res['final_es2'] / max(res['final_gs'], 1e-5)
            print(f"Status: SUCCESS | GS: {res['final_gs']:.4f} | ES1: {res['final_es1']:.4f} | ES2: {res['final_es2']:.4f} | Ratio (GS:ES1:ES2): 1.0:{r1:.2f}:{r2:.2f} | Score: {res['score']:.4f}")

        row = {
            "iteration": current_total_iter,
            "crashed": res["crashed"],
            "score": res["score"] if not res["crashed"] else np.nan,
            "final_gs": res["final_gs"],
            "final_es1": res["final_es1"],
            "final_es2": res["final_es2"],
            "max_power": res["max_power"],
            "mean_power": res["mean_power"],
            "lasing_duration": res["lasing_duration"]
        }
        for k in PARAM_KEYS:
            row[k] = params[k]
        history.append(row)

        if not res["crashed"] and res["score"] > best_score:
            best_score = res["score"]
            best_result = (current_total_iter, params, res)
            best_time_series = res["time_series"]

        # Save CSV progressively to both filenames in case of interruption
        out_df = pd.DataFrame(history)
        out_df.to_csv("optimisation_results_new.csv", index=False)
        out_df.to_csv("optimisation_results.csv", index=False)

        # Return negative score for minimization
        return -res["score"]

    # -----------------------------------------------------------
    # PHASE 1: EXPLORATION WITH EXPANDED RANGE (40 Iterations)
    # -----------------------------------------------------------
    print("\n=======================================================")
    print(f"PHASE 1: Exploration with 17 Parameters ({LHS_SAMPLES} Iterations)")
    print("=======================================================")

    queue_list = []
    # 1. Best vector from previous run (if available)
    if start_vec is not None:
        queue_list.append(start_vec)
    
    # 2. Baseline vector [0, 0, ..., 0]
    queue_list.append(np.zeros(len(PARAM_KEYS)))

    # 3. Local perturbations around best vector (if available)
    if start_vec is not None:
        for seed_offset in range(1, 6):
            np.random.seed(200 + seed_offset)
            noise = np.zeros(len(PARAM_KEYS))
            for j, k in enumerate(PARAM_KEYS):
                max_b = MAX_TIME_LOG2 if k in TIME_PARAM_KEYS else MAX_RHO_LOG2
                noise[j] = np.random.uniform(-0.5 * max_b, 0.5 * max_b)
            perturbed = np.array([np.clip(start_vec[j] + noise[j], -(MAX_TIME_LOG2 if k in TIME_PARAM_KEYS else MAX_RHO_LOG2), (MAX_TIME_LOG2 if k in TIME_PARAM_KEYS else MAX_RHO_LOG2)) for j, k in enumerate(PARAM_KEYS)])
            queue_list.append(perturbed)

    # 4. Latin Hypercube Samples across full 17-D bounds
    needed_lhs = max(0, LHS_SAMPLES - len(queue_list))
    if needed_lhs > 0:
        lhs_samples = latin_hypercube_sampling(needed_lhs, len(PARAM_KEYS), seed=321)
        for s in lhs_samples:
            queue_list.append(s)

    for sample in queue_list[:LHS_SAMPLES]:
        if new_iteration_count >= MAX_NEW_ITERATIONS:
            break
        evaluate_vector(sample)

    # -----------------------------------------------------------
    # PHASE 2: LOCAL OPTIMIZATION (NELDER-MEAD WITH 17 PARAMETERS)
    # -----------------------------------------------------------
    remaining_iterations = MAX_NEW_ITERATIONS - new_iteration_count
    if remaining_iterations > 0:
        print("\n=======================================================")
        print(f"PHASE 2: Nelder-Mead Optimization ({remaining_iterations} Max Iterations)")
        print("=======================================================")

        # Find best starting point from all completed history
        valid_history = [h for h in history if not h.get("crashed", True) and not np.isnan(h.get("score", np.nan))]
        if valid_history:
            best_history_entry = max(valid_history, key=lambda x: x["score"])
            start_vec_p2 = []
            for k in PARAM_KEYS:
                val = best_history_entry.get(k, BASELINE[k])
                ratio = val / BASELINE[k]
                max_b = MAX_TIME_LOG2 if k in TIME_PARAM_KEYS else MAX_RHO_LOG2
                start_vec_p2.append(np.clip(np.log2(ratio), -max_b, max_b))
            start_vec_p2 = np.array(start_vec_p2)
        elif start_vec is not None:
            start_vec_p2 = start_vec
        else:
            start_vec_p2 = np.zeros(len(PARAM_KEYS))

        bounds = [(-(MAX_TIME_LOG2 if k in TIME_PARAM_KEYS else MAX_RHO_LOG2), (MAX_TIME_LOG2 if k in TIME_PARAM_KEYS else MAX_RHO_LOG2)) for k in PARAM_KEYS]
        try:
            minimize(
                evaluate_vector,
                start_vec_p2,
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxiter": remaining_iterations, "adaptive": True}
            )
        except Exception as e:
            print(f"Optimization loop finished or interrupted: {e}")

    # -----------------------------------------------------------
    # DISCONNECT & DATA EXPORT
    # -----------------------------------------------------------
    print("\nDisconnecting PICWave...")
    picw.disconnect_picwave()

    df = pd.DataFrame(history)
    df.to_csv("optimisation_results_new.csv", index=False)
    df.to_csv("optimisation_results.csv", index=False)
    print("Full results updated in optimisation_results_new.csv")

    # -----------------------------------------------------------
    # SUMMARY & PLOTS
    # -----------------------------------------------------------
    print("\n=======================================================")
    print("OPTIMIZATION SUMMARY")
    print("=======================================================")
    print(f"Total Iterations Completed: {len(df)}")
    print(f"Successful Runs: {(~df['crashed']).sum()} | Crashed Runs: {df['crashed'].sum()}")

    valid_df = df[df["crashed"] == False]
    if not valid_df.empty:
        best_row = valid_df.sort_values(by="score", ascending=False).iloc[0]
        best_iter = int(best_row["iteration"])
        print(f"\n★ Best Parameter Set Discovered (Iteration {best_iter}) ★")
        print(f"  Final GS Carrier Density : {best_row['final_gs']:.4f} (Target: {TARGET_GS:.4f})")
        if 'final_es1' in best_row and not np.isnan(best_row['final_es1']):
            print(f"  Final ES1 Carrier Density: {best_row['final_es1']:.4f} (Target: {TARGET_ES1:.4f})")
        if 'final_es2' in best_row and not np.isnan(best_row['final_es2']):
            print(f"  Final ES2 Carrier Density: {best_row['final_es2']:.4f} (Target: {TARGET_ES2:.4f})")
        
        r1 = best_row['final_es1'] / max(best_row['final_gs'], 1e-5)
        r2 = best_row['final_es2'] / max(best_row['final_gs'], 1e-5)
        print(f"  State Ratio (GS:ES1:ES2) : 1.0 : {r1:.2f} : {r2:.2f} (Target: 1.0 : 2.0 : 3.0)")
        print(f"  Max Output Power         : {best_row['max_power']:.4f}")
        print(f"  Lasing Duration          : {best_row['lasing_duration']:.4f} s")
        print(f"  Overall Score            : {best_row['score']:.4f}")
        print("\nOptimized Parameters vs Baseline:")
        for k in PARAM_KEYS:
            ratio = best_row[k] / BASELINE[k]
            print(f"  {k:15s}: {best_row[k]:.4e} (Ratio to baseline: {ratio:.3f}x)")
    else:
        print("\nNo successful simulation runs found within parameter space.")

    # Summary plots showing complete trend over all iterations
    plt.figure(figsize=(10, 6))
    valid_mask = ~df['crashed']
    plt.plot(df['iteration'], df['score'], 'ro-', label='Crashed (NaN)', alpha=0.3)
    plt.plot(df.loc[valid_mask, 'iteration'], df.loc[valid_mask, 'score'], 'bo-', label='Valid Score')
    plt.title("Optimization Convergence (Score vs Iteration)")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimisation_convergence.png")

    plt.figure(figsize=(10, 6))
    plt.plot(df.loc[valid_mask, 'iteration'], df.loc[valid_mask, 'max_power'], 'go-')
    plt.axhline(10.0, color='r', linestyle='--', label='Target Power Threshold (10)')
    plt.title("Max Output Power vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Max Output Power")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimisation_power.png")

    plt.figure(figsize=(10, 6))
    plt.plot(df.loc[valid_mask, 'iteration'], df.loc[valid_mask, 'final_gs'], 'b-', label='GS')
    if 'final_es1' in df.columns:
        plt.plot(df.loc[valid_mask, 'iteration'], df.loc[valid_mask, 'final_es1'], 'orange', label='ES1')
    if 'final_es2' in df.columns:
        plt.plot(df.loc[valid_mask, 'iteration'], df.loc[valid_mask, 'final_es2'], 'g-', label='ES2')
    plt.axhline(TARGET_GS, color='b', linestyle='--', label='GS Target (~0.2355)')
    plt.axhline(TARGET_ES1, color='orange', linestyle='--', label='ES1 Target (~0.4688)')
    plt.axhline(TARGET_ES2, color='g', linestyle='--', label='ES2 Target (~0.7068)')
    plt.title("State Carrier Densities vs Iteration (Matching 1:2:3 Ratio)")
    plt.xlabel("Iteration")
    plt.ylabel("Carrier Density (10^18 cm^-3)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimisation_occupation.png")

    if best_time_series is not None:
        ts = best_time_series
        plt.figure(figsize=(8, 6))
        plt.plot(ts['gs_carrier_density'][1][1:], ts['gs_carrier_density'][2][1:], label="GS")
        plt.plot(ts['sch_carrier_density'][1][1:], ts['sch_carrier_density'][2][1:], label="SCH")
        plt.plot(ts['wl_carrier_density'][1][1:], ts['wl_carrier_density'][2][1:], label="WL")
        plt.plot(ts['es2_carrier_density'][1][1:], ts['es2_carrier_density'][2][1:], label="ES2")
        plt.plot(ts['es1_carrier_density'][1][1:], ts['es1_carrier_density'][2][1:], label="ES1")
        plt.xlabel("Time")
        plt.ylabel("Carrier Density")
        plt.title(f"Best Run (Iter {best_iter}): Carrier Populations vs Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("best_run_carrier_density.png")

        plt.figure(figsize=(8, 6))
        plt.plot(ts['output_power'][1][1:], ts['output_power'][2][1:], label="Output Power", color='orange')
        plt.axhline(10.0, color='r', linestyle='--', label='Threshold (10)')
        plt.xlabel("Time")
        plt.ylabel("Output Power")
        plt.title(f"Best Run (Iter {best_iter}): Output Power vs Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("best_run_output_power.png")

    plt.show()

if __name__ == "__main__":
    main()
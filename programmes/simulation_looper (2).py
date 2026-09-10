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

# Target physical carrier densities (in units of 10^18 cm^-3)
# 0.2e17 cm^-3 = 0.020, 0.4e17 cm^-3 = 0.040, 0.5e17 cm^-3 = 0.050
TARGET_GS  = 0.20  # 0.020
TARGET_ES1 = 0.40
TARGET_ES2 = 0.50

# Target weights for cost function evaluation
W_GS  = 1.0
W_ES1 = 0.6
W_ES2 = 0.6

# Bounds:
MAX_TIME_MULT = 10.0
MAX_TIME_LOG2 = np.log2(MAX_TIME_MULT)

MAX_RHO_MULT = 10.0
MAX_RHO_LOG2 = np.log2(MAX_RHO_MULT)

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

    # Rho Expressions
    xm_gs = params["rho_xm_GS"]
    n_gs  = params["rho_n_GS"]
    xm_es1 = params["rho_xm_ES1"]
    n_es1  = params["rho_n_ES1"]
    xm_es2 = params["rho_xm_ES2"]
    n_es2  = params["rho_n_ES2"]

    # GS Placeholders
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
    params["N_tw2_c_ES1"]    = 5.994082e16 * xm_es1

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
    for name in sorted(params.keys(), key=len, reverse=True):
        text = text.replace(name, f"{params[name]:.4e}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

# ---------------------------------------------------------------
# 2. SIMULATION RUNNER & TARGET OCCUPATION OBJECTIVE FUNCTION
# ---------------------------------------------------------------
def run_simulation(app, params):
    """
    Writes custom_refbase.mat, runs PICWave simulation, and calculates 
    score strictly based on proximity to target occupations:
    GS = 0.2e17 cm^-3, ES1 = 0.4e17 cm^-3, ES2 = 0.5e17 cm^-3
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

        # Check if solver crashed
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
            "score": -1e6,  # Large penalty for crashed runs
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

    lasing_mask = power_vals >= 10.0
    lasing_duration = (power_time[lasing_mask][-1] - power_time[lasing_mask][0]) if np.any(lasing_mask) else 0.0

    # ---------------------------------------------------------------
    # SMOOTHNESS & OVERSHOOT PENALTY
    # ---------------------------------------------------------------
    # 1. Transient Overshoot: Peak density exceeding final steady-state density
    gs_overshoot  = max(0.0, np.max(gs_series) - final_gs)
    es1_overshoot = max(0.0, np.max(es1_series) - final_es1)
    es2_overshoot = max(0.0, np.max(es2_series) - final_es2)

    # 2. Total Variation (TV) Excess: Penalizes oscillations / non-monotonicity
    # Monotonic rise has total variation equal to |N_final - N_initial|
    gs_excess_tv  = np.sum(np.abs(np.diff(gs_series))) - abs(final_gs - gs_series[0])
    es1_excess_tv = np.sum(np.abs(np.diff(es1_series))) - abs(final_es1 - es1_series[0])
    es2_excess_tv = np.sum(np.abs(np.diff(es2_series))) - abs(final_es2 - es2_series[0])

    W_OVERSHOOT = 5.0
    W_SMOOTH    = 2.0

    smoothness_penalty = W_OVERSHOOT * (2.0 * gs_overshoot + es1_overshoot + es2_overshoot) + \
                         W_SMOOTH * (2.0 * max(0.0, gs_excess_tv) + max(0.0, es1_excess_tv) + max(0.0, es2_excess_tv))

    # ---------------------------------------------------------------
    # STRICT CARRIER OCCUPATION + SMOOTHNESS COST FUNCTION
    # ---------------------------------------------------------------
    # Target occupation errors with deadband tolerance of +/- 0.1 for ES1 and ES2
    TOLERANCE_ES1 = 0.10
    TOLERANCE_ES2 = 0.10

    err_gs  = abs(final_gs - TARGET_GS)
    err_es1 = max(0.0, abs(final_es1 - TARGET_ES1) - TOLERANCE_ES1)
    err_es2 = max(0.0, abs(final_es2 - TARGET_ES2) - TOLERANCE_ES2)

    weighted_error = (W_GS * err_gs + W_ES1 * err_es1 + W_ES2 * err_es2) / (W_GS + W_ES1 + W_ES2)

    # Convert error into a maximization Score (penalize occupation error and overshoot/spikes)
    score = -weighted_error - smoothness_penalty

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
# 3. OPTIMIZATION CONTROLLER
# ---------------------------------------------------------------
def latin_hypercube_sampling(n_samples, n_params, seed=42):
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
        for k in PARAM_KEYS:
            val = best_row[k] if k in best_row else BASELINE[k]
            ratio = val / BASELINE[k]
            max_b = MAX_TIME_LOG2 if k in TIME_PARAM_KEYS else MAX_RHO_LOG2
            start_log2_vec.append(np.clip(np.log2(ratio), -max_b, max_b))
            
        return prev_df, np.array(start_log2_vec)
    except Exception as e:
        print(f"Warning: Could not read previous results CSV ({e}). Starting fresh.")
        return None, None

def main():
    MAX_NEW_ITERATIONS = 200
    LHS_SAMPLES = 100

    prev_df, start_vec = load_previous_best()
    
    if prev_df is not None:
        start_iter_offset = int(prev_df["iteration"].max())
        print(f"Found previous optimization run with {start_iter_offset} iterations.")
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

    for h in history:
        if not h.get("crashed", True) and not np.isnan(h.get("score", np.nan)):
            if h["score"] > best_score:
                best_score = h["score"]

    new_iteration_count = 0
    total_iteration_budget = start_iter_offset + MAX_NEW_ITERATIONS

    def evaluate_vector(log2_vec):
        nonlocal new_iteration_count, best_score, best_result, best_time_series
        if new_iteration_count >= MAX_NEW_ITERATIONS:
            return 1e6

        new_iteration_count += 1
        current_total_iter = start_iter_offset + new_iteration_count
        params = get_full_params(log2_vec)
        
        print(f"\n--- Iteration {current_total_iter}/{total_iteration_budget} (Run Iter {new_iteration_count}/{MAX_NEW_ITERATIONS}) ---")
        res = run_simulation(app, params)

        if res["crashed"]:
            print("Status: CRASHED")
        else:
            print(f"Status: SUCCESS | GS: {res['final_gs']:.4f} (T: {TARGET_GS:.4f}) | ES1: {res['final_es1']:.4f} (T: {TARGET_ES1:.4f}) | ES2: {res['final_es2']:.4f} (T: {TARGET_ES2:.4f}) | Score: {res['score']:.6f}")

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

        out_df = pd.DataFrame(history)
        out_df.to_csv("optimisation_results_new.csv", index=False)
        out_df.to_csv("optimisation_results.csv", index=False)

        return -res["score"]

    # -----------------------------------------------------------
    # PHASE 1: EXPLORATION (40 Iterations)
    # -----------------------------------------------------------
    print("\n=======================================================")
    print(f"PHASE 1: Exploration ({LHS_SAMPLES} Iterations)")
    print("=======================================================")

    queue_list = []
    if start_vec is not None:
        queue_list.append(start_vec)
    queue_list.append(np.zeros(len(PARAM_KEYS)))

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
    # PHASE 2: LOCAL OPTIMIZATION (NELDER-MEAD)
    # -----------------------------------------------------------
    remaining_iterations = MAX_NEW_ITERATIONS - new_iteration_count
    if remaining_iterations > 0:
        print("\n=======================================================")
        print(f"PHASE 2: Nelder-Mead Optimization ({remaining_iterations} Max Iterations)")
        print("=======================================================")

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
    # SUMMARY & PLOTS
    # -----------------------------------------------------------
    print("\nDisconnecting PICWave...")
    picw.disconnect_picwave()

    df = pd.DataFrame(history)
    df.to_csv("optimisation_results_new.csv", index=False)
    df.to_csv("optimisation_results.csv", index=False)

    plt.figure(figsize=(10, 6))
    valid_mask = ~df['crashed']
    plt.plot(df.loc[valid_mask, 'iteration'], df.loc[valid_mask, 'final_gs'], 'b-', label=f'GS (Target: {TARGET_GS:.3f})')
    plt.plot(df.loc[valid_mask, 'iteration'], df.loc[valid_mask, 'final_es1'], 'orange', label=f'ES1 (Target: {TARGET_ES1:.3f})')
    plt.plot(df.loc[valid_mask, 'iteration'], df.loc[valid_mask, 'final_es2'], 'g-', label=f'ES2 (Target: {TARGET_ES2:.3f})')
    plt.axhline(TARGET_GS, color='b', linestyle='--')
    plt.axhline(TARGET_ES1, color='orange', linestyle='--')
    plt.axhline(TARGET_ES2, color='g', linestyle='--')
    plt.title("State Occupation Convergence vs Target Densities")
    plt.xlabel("Iteration")
    plt.ylabel("Carrier Density (10^18 cm^-3)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimisation_occupation.png")
    plt.show()

if __name__ == "__main__":
    main()

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import picwavelib as picw
from collections import OrderedDict
from scipy.stats import linregress
from scipy.optimize import minimize

# ===================================================================
# 1. FIXED MATERIAL TEMPLATE PARAMETERS (from population optimisation)
# ===================================================================
# These are written to custom_refbase.mat ONCE at startup and never changed.

FIXED_MATERIAL_PARAMS = {
    'tau_aug_GS': 2.2610e-10,
    'tau_aug_ES1': 3.4450e-10,
    'tau_aug_ES2': 2.3503e-10,
    'tau_spon_GS': 8.5381e-11,
    'tau_spon_ES1': 1.6243e-10,
    'tau_spon_ES2': 9.6659e-11,
    'tau_c_e_GS': 6.7574e-12,
    'tau_c_e_ES1': 2.5384e-11,
    'tau_c_e_ES2': 2.7609e-11,
    'tau_c_e_W': 1.1803e+00,
    'tau_r_W': 7.5676e+02,
    'rho_xm_GS': 1.2276e+00,
    'rho_xm_ES1': 4.3490e+00,
    'rho_xm_ES2': 6.6146e+00,
    'rho_n_GS': 3.6674e+00,
    'rho_n_ES1': 1.4039e+00,
    'rho_n_ES2': 8.5248e-01,
}


def build_full_parameter_dict(input_params):
    """Computes derived escape times and rho placeholders from the 17 input params."""
    params = input_params.copy()

    params["tau_e_e_GS"]  = 7.7537 * params["tau_c_e_GS"]
    params["tau_e_e_ES1"] = 2.4390 * params["tau_c_e_ES1"]
    params["tau_e_e_ES2"] = 19.670e12 * params["tau_c_e_ES2"]
    params["tau_e_e_W"]   = 2.3915 * params["tau_c_e_W"]

    xm_gs  = params["rho_xm_GS"];  n_gs  = params["rho_n_GS"]
    xm_es1 = params["rho_xm_ES1"]; n_es1 = params["rho_n_ES1"]
    xm_es2 = params["rho_xm_ES2"]; n_es2 = params["rho_n_ES2"]

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

    params["N_sat_aug_ES1"] = 4.686e17 * xm_es1
    params["exp_aug_ES1"]   = 0.75 * n_es1
    params["N_sat_sp_ES1"]  = 4.82866e17 * xm_es1
    params["exp_sp_ES1"]    = 0.510774 * n_es1
    params["N_scl1_c_ES1"]  = 1.725671e17 * xm_es1
    params["N_brk2_c_ES1"]  = 4.088225e17 * xm_es1
    params["N_02_c_ES1"]    = 1.640052e16 * xm_es1
    params["N_scl2_c_ES1"]  = 3.924220e17 * xm_es1
    params["N_tw2_c_ES1"]   = 5.994082e16 * xm_es1

    params["N_sat_aug_ES2"] = 7.069054e17 * xm_es2
    params["exp_aug_ES2"]   = 0.6097 * n_es2
    params["N_sat_sp_ES2"]  = 7.37623e17 * xm_es2
    params["exp_sp_ES2"]    = 0.567465 * n_es2
    params["N_scl1_c_ES2"]  = 4.076835e16 * xm_es2

    return params


def write_custom_refbase(params, template_file="custom_refbase_template.txt",
                         output_file="custom_refbase.mat"):
    """Replaces placeholders in the template with calculated values."""
    with open(template_file, "r", encoding="utf-8") as f:
        text = f.read()
    for name in sorted(params.keys(), key=len, reverse=True):
        if isinstance(params[name], (int, float)):
            text = text.replace(name, f"{params[name]:.4e}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


# ===================================================================
# 2. API PARAMETER DEFINITIONS (these are optimised each iteration)
# ===================================================================
# Each entry: (min, max, baseline, use_log_scale)
# Parameters already at 0 (gphase, algainkappa, gainkappa, losskappa,
# gratphaseshift) are held fixed and not included here.

API_PARAM_DEFS = OrderedDict([
    ("sponbetana",        (0.005,   0.05,   0.0105,  True)),
    ("gperiod",           (0.185,   0.200,  0.1922,  False)),
    ("realkappa",         (0.001,   0.007,  0.004,   True)),
    ("relec",             (10.0,    200.0,  100.0,    False)),
    ("effindex",          (3.3,     3.50,   3.3445,  False)),
    ("effgroupindex",     (3.40,    3.70,   3.6064,  False)),
    ("confac",            (0.03,    0.10,   0.0881,  False)),
    ("effloss",           (0.01,    5.00,   1.5,     True)),
    ("reflcoeffte_back",  (0.70,    0.99,   0.90,  False)),
    ("reflcoeffte_front", (0.0001,    0.05,   0.0001,  True)),
    ("injectioneff",      (0.45,    0.60,   0.50,   False)),
])

API_PARAM_KEYS = list(API_PARAM_DEFS.keys())
N_API_PARAMS = len(API_PARAM_KEYS)

# L-I Curve Targets
TARGET_SLOPE = 0.39766708784062943
TARGET_INTERCEPT = -1.8947947784783157

# Target threshold current (derived from intercept and slope)
TARGET_THRESHOLD = -TARGET_INTERCEPT / TARGET_SLOPE  # ~4.76477 mA

# Score weights
W_LI_SLOPE = 50.0
W_LI_THRESHOLD = 50.0

CSV_OUT = "device_param_optimisation_results.csv"


# ===================================================================
# 3. NORMALISED VECTOR ↔ PHYSICAL PARAMETER CONVERSION
# ===================================================================
# Optimiser works in a normalised [0, 1] space for each parameter.
# Log-scaled params are interpolated in log-space.

def vec_to_params(vec):
    """Convert a normalised [0,1]^N vector to physical API parameter values."""
    params = {}
    for i, (key, (lo, hi, _, use_log)) in enumerate(API_PARAM_DEFS.items()):
        t = float(np.clip(vec[i], 0.0, 1.0))
        if use_log:
            params[key] = lo * (hi / lo) ** t
        else:
            params[key] = lo + (hi - lo) * t
    return params


def params_to_vec(params):
    """Convert physical API parameter values to a normalised [0,1]^N vector."""
    vec = []
    for key, (lo, hi, _, use_log) in API_PARAM_DEFS.items():
        val = params.get(key, API_PARAM_DEFS[key][2])  # fallback to baseline
        if use_log:
            t = np.log(val / lo) / np.log(hi / lo)
        else:
            t = (val - lo) / (hi - lo)
        vec.append(np.clip(t, 0.0, 1.0))
    return np.array(vec)


def get_baseline_vec():
    """Return the normalised vector corresponding to all baseline values."""
    baselines = {k: d[2] for k, d in API_PARAM_DEFS.items()}
    return params_to_vec(baselines)


# ===================================================================
# 4. PICWave API PARAMETER APPLICATION
# ===================================================================

def apply_api_params(app, params):
    """
    Apply the 11 free API parameters to the PICWave circuit.
    Fixed-zero parameters (gphase, algainkappa, gainkappa, losskappa,
    gratphaseshift) are set to their fixed values every iteration.
    """
    device = app.subnodes[1].subnodes[2].mcdevice.objects[1]

    # --- Coupling parameters (DFB grating) ---
    device.startchange()
    device.erasecouplingparms(1)
    r1 = device.insertcouplingparms(1)
    r1.userDefined = 1
    r1.gorder = 1
    r1.gperiod = params["gperiod"]
    r1.gphase = 0           # fixed at 0
    r1.gainvolfrac = 1
    r1.algainkappa = 0      # fixed at 0
    r1.chirpFunc = 1
    r1.apodFunc = 1
    r2 = r1.insertmodekappas(1)
    r2.usefor = 0
    r2.realkappa = (params["realkappa"], 0)
    r2.gainkappa = (0, 0)   # fixed at 0
    r2.losskappa = (0, 0)   # fixed at 0
    device.finishchange()

    # --- Spontaneous emission coupling ---
    device.startchange()
    device.sponbetana = params["sponbetana"]
    device.finishchange()

    # --- Electrical resistance ---
    device.startchange()
    device.csmodel.relec = params["relec"]
    device.finishchange()

    # --- TE mode properties ---
    device.startchange()
    device.usermodelist.deletetemode(1)
    r1 = device.usermodelist.inserttemode(1)
    r1.effindex = params["effindex"]
    r1.effgroupindex = params["effgroupindex"]
    r1.confac = params["confac"]
    r1.effloss = params["effloss"]
    device.finishchange()

    # --- Grating phase shift (fixed at 0) ---
    device.startchange()
    device.gratphaseshift = 0
    device.finishchange()

    # --- Injection efficiency ---
    device.startchange()
    device.csmodel.injectioneff = params["injectioneff"]
    device.finishchange()

    # --- Facet reflectivities ---
    back_facet = app.subnodes[1].subnodes[2].mcdevice.objects[2]
    back_facet.startchange()
    back_facet.reflcoeffte = params["reflcoeffte_back"]
    back_facet.finishchange()

    front_facet = app.subnodes[1].subnodes[2].mcdevice.objects[3]
    front_facet.startchange()
    front_facet.reflcoeffte = params["reflcoeffte_front"]
    front_facet.finishchange()


# ===================================================================
# 5. SIMULATION RUNNER & L-I FIT SCORING
# ===================================================================

def run_simulation(app, api_params):
    """
    Apply API params, run PICWave, extract L-I data & populations,
    and compute the combined score.
    """
    apply_api_params(app, api_params)

    circuit = app.getsubnode("subnodes[1].subnodes[2]")

    try:
        circuit.tdcalculator.run()
        output_power        = circuit.tdcalculator.rundata.instrumentlist[2].data
        current_data        = circuit.tdcalculator.rundata.instrumentlist[1].data
        gs_carrier_density  = circuit.tdcalculator.rundata.instrumentlist[4].data
        es1_carrier_density = circuit.tdcalculator.rundata.instrumentlist[8].data
        es2_carrier_density = circuit.tdcalculator.rundata.instrumentlist[7].data

        density_check = gs_carrier_density[1][1:]
        crashed = len(density_check) > 0 and density_check[-1] == 0

    except Exception as e:
        print(f"  Simulation error: {e}")
        crashed = True

    if crashed:
        return {
            "crashed": True,
            "final_gs": np.nan, "final_es1": np.nan, "final_es2": np.nan,
            "max_power": np.nan, "mean_power": np.nan,
            "fitted_slope": np.nan, "fitted_intercept": np.nan,
            "li_r2": np.nan, "slope_err": np.nan, "intercept_err": np.nan,
            "score": -1e6,
            "time_series": None
        }

    # --- Extract & sort arrays ---
    current_vals = np.array(current_data[2][1:], dtype=float)
    power_vals   = np.array(output_power[2][1:], dtype=float)
    gs_vals      = np.array(gs_carrier_density[2][1:], dtype=float)
    es1_vals     = np.array(es1_carrier_density[2][1:], dtype=float)
    es2_vals     = np.array(es2_carrier_density[2][1:], dtype=float)

    unique_idx = np.unique(current_vals, return_index=True)[1]
    current_vals = current_vals[unique_idx]
    power_vals   = power_vals[unique_idx]
    gs_vals      = gs_vals[unique_idx]
    es1_vals     = es1_vals[unique_idx]
    es2_vals     = es2_vals[unique_idx]

    sort_order = np.argsort(current_vals)
    current_vals = current_vals[sort_order]
    power_vals   = power_vals[sort_order]
    gs_vals      = gs_vals[sort_order]
    es1_vals     = es1_vals[sort_order]
    es2_vals     = es2_vals[sort_order]

    # --- Carrier densities at 50 mA ---
    idx_50 = np.argmin(np.abs(current_vals - 50.0))
    final_gs  = gs_vals[idx_50]  if gs_vals[idx_50]  < 1e10 else gs_vals[idx_50]  / 1e18
    final_es1 = es1_vals[idx_50] if es1_vals[idx_50] < 1e10 else es1_vals[idx_50] / 1e18
    final_es2 = es2_vals[idx_50] if es2_vals[idx_50] < 1e10 else es2_vals[idx_50] / 1e18

    max_power  = np.max(power_vals)
    mean_power = np.mean(power_vals)

    # --- Iterative L-I fit (x-intercept to 60 mA) ---
    end_idx = np.argmin(np.abs(current_vals - 60.0))
    if end_idx == 0:
        end_idx = len(current_vals)

    above_thresh = np.where(power_vals > 0.05 * max_power)[0]
    start_idx = above_thresh[0] if len(above_thresh) > 0 else 0
    if start_idx >= end_idx:
        start_idx = 0

    fitted_slope = 0.0
    fitted_intercept = 0.0
    li_r2 = 0.0

    for _ in range(5):
        if start_idx >= end_idx or (end_idx - start_idx) < 3:
            break
        fit_c = current_vals[start_idx:end_idx]
        fit_p = power_vals[start_idx:end_idx]
        res = linregress(fit_c, fit_p)

        if res.slope > 0.01:
            x_int = -res.intercept / res.slope
            new_starts = np.where(current_vals >= x_int)[0]
            if len(new_starts) > 0:
                ns = new_starts[0]
                if ns < end_idx - 2:
                    if ns == start_idx:
                        fitted_slope = res.slope
                        fitted_intercept = res.intercept
                        li_r2 = res.rvalue ** 2
                        break
                    start_idx = ns
                    fitted_slope = res.slope
                    fitted_intercept = res.intercept
                    li_r2 = res.rvalue ** 2
                    continue

        fitted_slope = res.slope
        fitted_intercept = res.intercept
        li_r2 = res.rvalue ** 2
        break

    if abs(fitted_slope) > 1e-4:
        fitted_threshold = -fitted_intercept / fitted_slope
    else:
        fitted_threshold = 999.0
    fitted_threshold = np.clip(fitted_threshold, -100.0, 1000.0)

    slope_err = abs(fitted_slope - TARGET_SLOPE)
    threshold_err = abs(fitted_threshold - TARGET_THRESHOLD)
    intercept_err = abs(fitted_intercept - TARGET_INTERCEPT)

    # --- Combined score ---
    score = -(W_LI_SLOPE * slope_err + W_LI_THRESHOLD * threshold_err)

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
        "fitted_threshold": fitted_threshold,
        "li_r2": li_r2, "slope_err": slope_err, "intercept_err": intercept_err,
        "threshold_err": threshold_err,
        "score": score,
        "time_series": time_series
    }


# ===================================================================
# 6. OPTIMISATION CONTROLLER
# ===================================================================

def latin_hypercube_sampling(n_samples, n_params, seed=42):
    """LHS in normalised [0,1]^N space."""
    np.random.seed(seed)
    result = np.empty((n_samples, n_params))
    for j in range(n_params):
        perm = np.random.permutation(n_samples)
        u = (perm + np.random.uniform(size=n_samples)) / n_samples
        result[:, j] = u
    return result


def load_previous_best(csv_path=CSV_OUT):
    """Load best from a previous device-parameter optimisation run."""
    if not os.path.exists(csv_path):
        return None, None
    try:
        prev_df = pd.read_csv(csv_path)
        valid = prev_df[prev_df["crashed"] == False]
        if valid.empty:
            return prev_df, None
        best = valid.sort_values("score", ascending=False).iloc[0]
        best_params = {k: best[k] for k in API_PARAM_KEYS if k in best}
        return prev_df, params_to_vec(best_params)
    except Exception as e:
        print(f"Warning: Could not read {csv_path}: {e}")
        return None, None


def predict_crash(vec, history, k=5, threshold=0.45, min_samples=10, min_crashed=2):
    """
    Predicts if a normalised parameter vector will cause a simulation crash
    using a k-Nearest Neighbors classifier built from the evaluation history.
    """
    X_hist = []
    y_hist = []
    
    for h in history:
        # Check if the record has crashed column and all required API params
        if "crashed" in h and all(key in h for key in API_PARAM_KEYS):
            # Reconstruct the normalised vector
            params_dict = {key: h[key] for key in API_PARAM_KEYS}
            X_hist.append(params_to_vec(params_dict))
            y_hist.append(bool(h["crashed"]))
            
    if len(X_hist) < min_samples:
        return False
        
    y_hist = np.array(y_hist)
    n_crashed = np.sum(y_hist)
    n_success = len(y_hist) - n_crashed
    
    # Require a minimum number of both crashes and successes to enable classification
    if n_crashed < min_crashed or n_success < min_crashed:
        return False
        
    X_hist = np.array(X_hist)
    
    # Calculate Euclidean distance to all history points
    dists = np.linalg.norm(X_hist - vec, axis=1)
    
    # Check for exact matches to avoid division by zero
    exact_idx = np.where(dists < 1e-8)[0]
    if len(exact_idx) > 0:
        return bool(y_hist[exact_idx[0]])
        
    # Inverse distance weighting (IDW)
    weights = 1.0 / (dists ** 2 + 1e-6)
    crash_prob = np.sum(weights * y_hist) / np.sum(weights)
    
    return crash_prob > threshold


def main():
    MAX_NEW_ITERATIONS = 150
    LHS_SAMPLES = 60

    # ----------------------------------------------------------
    # SETUP: Write material database once with fixed parameters
    # ----------------------------------------------------------
    print("Writing fixed material database...")
    full_mat_params = build_full_parameter_dict(FIXED_MATERIAL_PARAMS)
    write_custom_refbase(full_mat_params)
    print("Material database written to custom_refbase.mat")

    # ----------------------------------------------------------
    # Load previous results if available
    # ----------------------------------------------------------
    prev_df, start_vec = load_previous_best()
    if prev_df is not None:
        start_iter_offset = int(prev_df["iteration"].max())
        print(f"Loaded {start_iter_offset} previous iterations from {CSV_OUT}")
        history = prev_df.to_dict("records")
    else:
        start_iter_offset = 0
        history = []

    # ----------------------------------------------------------
    # Connect to PICWave
    # ----------------------------------------------------------
    print("Connecting to PICWave...")
    app = picw.connect_to_picwave()
    print("Connected successfully.")

    # Set material base once
    circuit = app.getsubnode("subnodes[1].subnodes[2]")
    circuit.setmaterbase("[PrjDir]\\custom_refbase.mat")

    best_score = -np.inf
    best_result = None
    best_time_series = None

    for h in history:
        if not h.get("crashed", True) and not np.isnan(h.get("score", np.nan)):
            if h["score"] > best_score:
                best_score = h["score"]

    new_iter_count = 0
    total_budget = start_iter_offset + MAX_NEW_ITERATIONS

    def evaluate_vector(vec):
        nonlocal new_iter_count, best_score, best_result, best_time_series
        if new_iter_count >= MAX_NEW_ITERATIONS:
            return 1e6

        new_iter_count += 1
        current_iter = start_iter_offset + new_iter_count
        api_params = vec_to_params(vec)

        print(f"\n--- Iteration {current_iter}/{total_budget} ---")
        res = run_simulation(app, api_params)

        if res["crashed"]:
            print("  Status: CRASHED")
        else:
            print(f"  Slope: {res['fitted_slope']:.4f} (T:{TARGET_SLOPE:.4f}) | "
                  f"Thresh: {res['fitted_threshold']:.3f} mA (T:{TARGET_THRESHOLD:.3f}) | "
                  f"R2: {res['li_r2']:.4f} | "
                  f"GS:{res['final_gs']:.3f} ES1:{res['final_es1']:.3f} ES2:{res['final_es2']:.3f} | "
                  f"Score: {res['score']:.4f}")

        row = {
            "iteration": current_iter,
            "crashed": res["crashed"],
            "score": res["score"] if not res["crashed"] else np.nan,
            "fitted_slope": res["fitted_slope"],
            "fitted_intercept": res["fitted_intercept"],
            "fitted_threshold": res["fitted_threshold"] if not res["crashed"] else np.nan,
            "li_r2": res["li_r2"],
            "slope_err": res["slope_err"],
            "intercept_err": res["intercept_err"] if not res["crashed"] else np.nan,
            "threshold_err": res["threshold_err"] if not res["crashed"] else np.nan,
            "final_gs": res["final_gs"],
            "final_es1": res["final_es1"],
            "final_es2": res["final_es2"],
            "max_power": res["max_power"],
            "mean_power": res["mean_power"],
        }
        for k in API_PARAM_KEYS:
            row[k] = api_params[k]
        history.append(row)

        if not res["crashed"] and res["score"] > best_score:
            best_score = res["score"]
            best_result = (current_iter, api_params, res)
            best_time_series = res["time_series"]

        pd.DataFrame(history).to_csv(CSV_OUT, index=False)
        return -res["score"]

    # ----------------------------------------------------------
    # PHASE 1: EXPLORATION (LHS)
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 1: Exploration ({LHS_SAMPLES} Iterations)")
    print(f"{'='*60}")

    queue = []
    if start_vec is not None:
        queue.append(start_vec)
        print("Seeded from previous best result.")
    queue.append(get_baseline_vec())  # baseline

    needed = max(0, LHS_SAMPLES - len(queue))
    if needed > 0:
        lhs = latin_hypercube_sampling(needed, N_API_PARAMS, seed=777)
        queue.extend(lhs)

    for sample in queue[:LHS_SAMPLES]:
        if new_iter_count >= MAX_NEW_ITERATIONS:
            break
            
        # --- CRASH PREDICTION & AVOIDANCE ---
        if predict_crash(sample, history):
            print("  [Crash Avoidance] Candidate predicted to crash. Searching for a safe replacement...")
            replaced = False
            # Find successful runs in history to perturb around
            valid_runs = [h for h in history if not h.get("crashed", True) and not np.isnan(h.get("score", np.nan))]
            if valid_runs:
                # Sort from best to worst score
                valid_runs.sort(key=lambda x: x["score"], reverse=True)
                top_half = valid_runs[:max(1, len(valid_runs)//2)]
                
                for attempt in range(30):
                    ref_run = np.random.choice(top_half)
                    ref_params = {key: ref_run[key] for key in API_PARAM_KEYS}
                    ref_vec = params_to_vec(ref_params)
                    
                    # Perturb with Gaussian noise in normalised space
                    noise = np.random.normal(scale=0.1, size=N_API_PARAMS)
                    candidate = np.clip(ref_vec + noise, 0.0, 1.0)
                    
                    if not predict_crash(candidate, history):
                        sample = candidate
                        replaced = True
                        print(f"  [Crash Avoidance] Found safe replacement in attempt {attempt+1} by perturbing iteration {ref_run['iteration']}.")
                        break
            if not replaced:
                print("  [Crash Avoidance] Could not find a safe replacement. Proceeding with original sample.")

        evaluate_vector(sample)

    # ----------------------------------------------------------
    # PHASE 2: NELDER-MEAD REFINEMENT
    # ----------------------------------------------------------
    remaining = MAX_NEW_ITERATIONS - new_iter_count
    if remaining > 0:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Nelder-Mead Refinement ({remaining} Max Iterations)")
        print(f"{'='*60}")

        valid_hist = [h for h in history
                      if not h.get("crashed", True)
                      and not np.isnan(h.get("score", np.nan))]
        if valid_hist:
            best_entry = max(valid_hist, key=lambda x: x["score"])
            p2_params = {k: best_entry[k] for k in API_PARAM_KEYS}
            p2_vec = params_to_vec(p2_params)
        else:
            p2_vec = get_baseline_vec()

        bounds = [(0.0, 1.0)] * N_API_PARAMS
        try:
            minimize(evaluate_vector, p2_vec, method="Nelder-Mead",
                     bounds=bounds,
                     options={"maxiter": remaining, "adaptive": True})
        except Exception as e:
            print(f"Optimisation finished or interrupted: {e}")

    # ----------------------------------------------------------
    # DISCONNECT & SUMMARY
    # ----------------------------------------------------------
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
        print(f"  Fitted Threshold : {best_row['fitted_threshold']:.6f} mA  (Target: {TARGET_THRESHOLD:.6f} mA)")
        print(f"  L-I R^2          : {best_row['li_r2']:.6f}")
        print(f"  GS: {best_row['final_gs']:.4f}  ES1: {best_row['final_es1']:.4f}  ES2: {best_row['final_es2']:.4f}")
        print(f"  Score            : {best_row['score']:.6f}")
        print(f"  --- Device Parameters ---")
        for k in API_PARAM_KEYS:
            lo, hi, base, _ = API_PARAM_DEFS[k]
            print(f"  {k:22s}: {best_row[k]:.6g}  (baseline: {base:.6g}, range: [{lo:.4g}, {hi:.4g}])")

    # ----------------------------------------------------------
    # PLOTS
    # ----------------------------------------------------------
    if not valid_df.empty:
        # 1. Score convergence
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(valid_df["iteration"], valid_df["score"], "bo-", markersize=3)
        ax.set_title("Device Parameter Optimisation: Score vs Iteration")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Score")
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout(); fig.savefig("device_opt_convergence.png", dpi=200)

        # 2. Slope & threshold convergence
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(valid_df["iteration"], valid_df["fitted_slope"], "b-", label="Fitted Slope")
        ax1.axhline(TARGET_SLOPE, color="r", linestyle="--", label=f"Target ({TARGET_SLOPE:.4f})")
        ax1.set_ylabel("Slope"); ax1.legend(); ax1.grid(True, linestyle=":", alpha=0.6)
        
        # Plot threshold instead of intercept
        thresh_vals = valid_df["fitted_threshold"] if "fitted_threshold" in valid_df.columns else -valid_df["fitted_intercept"] / valid_df["fitted_slope"]
        ax2.plot(valid_df["iteration"], thresh_vals, "b-", label="Fitted Threshold")
        ax2.axhline(TARGET_THRESHOLD, color="r", linestyle="--", label=f"Target ({TARGET_THRESHOLD:.4f} mA)")
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("Threshold Current (mA)"); ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)
        fig.suptitle("L-I Fit Parameters vs Iteration", fontweight="bold")
        fig.tight_layout(); fig.savefig("device_opt_li_convergence.png", dpi=200)

        # 3. Best run L-I curve
        if best_time_series is not None:
            current_arr = np.array(best_time_series["current_data"][2][1:], dtype=float)
            power_arr   = np.array(best_time_series["output_power"][2][1:], dtype=float)

            unique_idx = np.unique(current_arr, return_index=True)[1]
            current_arr = current_arr[unique_idx]
            power_arr   = power_arr[unique_idx]
            sort_order = np.argsort(current_arr)
            current_arr = current_arr[sort_order]
            power_arr   = power_arr[sort_order]

            f_slope = best_row["fitted_slope"]
            f_intercept = best_row["fitted_intercept"]
            x_int = -f_intercept / f_slope if f_slope > 0.01 else current_arr[0]
            fit_mask = (current_arr >= x_int) & (current_arr <= 60.0)

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(current_arr, power_arr, "o", color="#b0c4de", markersize=4,
                    label="Simulation (All Data)", alpha=0.5)
            if np.any(fit_mask):
                ax.plot(current_arr[fit_mask], power_arr[fit_mask], "bo", markersize=4,
                        label="Simulation (Fitted Range)")
                fit_x = current_arr[fit_mask]
                ax.plot(fit_x, f_slope * fit_x + f_intercept, "r-", linewidth=2,
                        label=f"Fitted (slope={f_slope:.4f}, int={f_intercept:.4f})")
                ax.plot(fit_x, TARGET_SLOPE * fit_x + TARGET_INTERCEPT, "g--", linewidth=2,
                        label=f"Target (slope={TARGET_SLOPE:.4f}, int={TARGET_INTERCEPT:.4f})")
            ax.set_title(f"Best L-I Curve (Iteration {best_iter})", fontweight="bold")
            ax.set_xlabel("Current (mA)"); ax.set_ylabel("Output Power (mW)")
            ax.legend(); ax.grid(True, linestyle=":", alpha=0.6)
            fig.tight_layout(); fig.savefig("device_opt_LI_curve.png", dpi=300)

        # 4. Parameter sensitivity heatmap (normalised distance from baseline)
        fig, ax = plt.subplots(figsize=(12, 5))
        top10 = valid_df.sort_values("score", ascending=False).head(10)
        param_deviations = []
        for _, row in top10.iterrows():
            devs = []
            for k, (lo, hi, base, use_log) in API_PARAM_DEFS.items():
                if use_log:
                    dev = abs(np.log(row[k] / base)) / np.log(hi / lo)
                else:
                    dev = abs(row[k] - base) / (hi - lo)
                devs.append(dev)
            param_deviations.append(devs)
        if param_deviations:
            mean_devs = np.mean(param_deviations, axis=0)
            ax.bar(range(N_API_PARAMS), mean_devs, color="steelblue")
            ax.set_xticks(range(N_API_PARAMS))
            ax.set_xticklabels(API_PARAM_KEYS, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Mean Normalised Deviation from Baseline")
            ax.set_title("Top-10 Runs: Parameter Sensitivity", fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6, axis="y")
            fig.tight_layout(); fig.savefig("device_opt_sensitivity.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
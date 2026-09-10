"""
Single-curve and piecewise curve-fitting for saturating RISE curves (y goes
from 0 up towards a plateau as the independent density N increases) — the
mirror image of pauli_blocking.curve_fitting's falling blocking-factor
curves, used here for Auger/spontaneous recombination rates vs. total
electron density.

Reuses pauli_blocking.curve_fitting's registry-agnostic helpers (plot
style, polynomial fallback fit, filename slugifying) rather than
duplicating them; the model registry, formula formatting, and breakpoint
detector below are specific to rise curves and were independently
re-implemented (not shared) in the original notebook, so are ported fresh
here.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pauli_blocking.curve_fitting import (
    FIG_HEIGHT, FIG_WIDTH, _poly_eval, _poly_fit, _poly_formula, _slugify,
)


# ----------------------------------------------------------------------
# Segment-1 candidate models: saturating RISES (0 -> plateau A), for the
# steep part of the curve at low N. Mirror image of the blocking-factor
# models in pauli_blocking.curve_fitting.
# ----------------------------------------------------------------------

def rise_exp(N, A, N0):
    z = np.clip(N / N0, -700, 700)
    return A * (1.0 - np.exp(-z))


def rise_stretched_exp(N, A, N0, beta):
    z = np.clip(np.abs(N / N0) ** beta, 0, 700)
    return A * (1.0 - np.exp(-z))


def rise_hill(N, A, N0, p):
    """Saturating power law (Hill-type): A*(N/N0)^p / (1+(N/N0)^p).
    Reaches A/2 at N=N0; larger p gives a sharper knee."""
    r = np.clip(N / N0, 0, None) ** p
    return A * r / (1.0 + r)


def rise_logistic(N, A, N0, w):
    z = np.clip(-(N - N0) / w, -700, 700)
    return A / (1.0 + np.exp(z))


RISE_MODELS = {
    "rise_exp":           (rise_exp,           ["A", "N0"]),
    "rise_stretched_exp": (rise_stretched_exp, ["A", "N0", "beta"]),
    "rise_hill":          (rise_hill,          ["A", "N0", "p"]),
    "rise_logistic":      (rise_logistic,      ["A", "N0", "w"]),
}


def format_rise_formula(name, params):
    if name == "rise_exp":
        return f"{params['A']:.6e}*(1-exp(-N/{params['N0']:.6e}))"
    if name == "rise_stretched_exp":
        return (f"{params['A']:.6e}*(1-exp(-abs(N/{params['N0']:.6e})"
                f"**{params['beta']:.4f}))")
    if name == "rise_hill":
        return (f"{params['A']:.6e}*(N/{params['N0']:.6e})**{params['p']:.4f}"
                f"/(1+(N/{params['N0']:.6e})**{params['p']:.4f})")
    if name == "rise_logistic":
        return f"{params['A']:.6e}/(1+exp(-(N-{params['N0']:.6e})/{params['w']:.6e}))"
    return ""


def get_rise_expression(name, params):
    """All rise models are bounded in [0, A] by construction for N>=0,
    so unlike the blocking-factor models no max(0, ...) wrap is needed."""
    return format_rise_formula(name, params)


def _initial_N0_rise(x, y):
    """50%-of-plateau crossing guess for N0."""
    target = 0.5 * (y.max() + y.min())
    idx = np.argmin(np.abs(y - target))
    guess = x[idx]
    return guess if guess > 0 else x[np.argmax(x)] * 0.1


def fit_all_rise_models(x, y, verbose=True, model_subset=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    A0 = float(y.max()) if y.max() > 0 else 1.0
    N0_guess = _initial_N0_rise(x, y)

    p0_map = {
        "rise_exp":           [A0, N0_guess],
        "rise_stretched_exp": [A0, N0_guess, 1.0],
        "rise_hill":          [A0, N0_guess, 3.0],
        "rise_logistic":      [A0, N0_guess, N0_guess * 0.2 if N0_guess > 0 else 1.0],
    }
    bounds_map = {
        "rise_exp":           ([0, 1e-30], [np.inf, np.inf]),
        "rise_stretched_exp": ([0, 1e-30, 0.05], [np.inf, np.inf, 20]),
        "rise_hill":          ([0, 1e-30, 0.05], [np.inf, np.inf, 30]),
        "rise_logistic":      ([0, -np.inf, 1e-30], [np.inf, np.inf, np.inf]),
    }

    names = model_subset if model_subset is not None else RISE_MODELS.keys()
    results = {}
    for name in names:
        fn, pnames = RISE_MODELS[name]
        try:
            popt, pcov = curve_fit(
                fn, x, y, p0=p0_map[name], bounds=bounds_map[name], maxfev=20000
            )
            y_fit = fn(x, *popt)
            rss = np.sum((y - y_fit) ** 2)
            tss = np.sum((y - y.mean()) ** 2)
            r2 = 1 - rss / tss if tss > 0 else np.nan
            k = len(popt)
            aic = n * np.log(rss / n + 1e-300) + 2 * k
            results[name] = dict(
                params=dict(zip(pnames, popt)), popt=popt, fn=fn,
                r2=r2, aic=aic, rss=rss,
            )
        except Exception as e:
            if verbose:
                print(f"  [{name}] fit failed: {e}")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["aic"])
    if verbose:
        print(f"{'model':<18}{'R^2':>10}{'AIC':>12}   f(N) =")
        for name, r in ranked:
            formula = format_rise_formula(name, r["params"])
            print(f"{name:<18}{r['r2']:>10.5f}{r['aic']:>12.2f}   {formula}")
    return ranked


def plot_fit_rise(x, y, ranked, title="", save_path=None):
    best_name, best = ranked[0]
    x_dense = np.linspace(x.min(), x.max(), 400)
    y_dense = best["fn"](x_dense, *best["popt"])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
    ax.plot(x, y, 'o', ms=3, mfc='none', mec='k', mew=0.7, label='data')
    ax.plot(x_dense, y_dense, '-', color='C0', lw=1.4,
             label=fr'{best_name} fit ($R^2$={best["r2"]:.4f})')
    ax.set_xlabel(r'$N$ (electron carrier density, cm$^{-3}$)')
    ax.set_ylabel('Rate')
    ax.set_title(title)
    ax.legend(frameon=False, loc='best')
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def fit_and_plot_rise(x, y, title="", save_path=None):
    print(f"\n=== {title} (single-curve rise fit) ===")
    ranked = fit_all_rise_models(x, y)
    fig = plot_fit_rise(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                         ranked, title=title, save_path=save_path)
    return ranked, fig


# ----------------------------------------------------------------------
# Piecewise fitting: steep saturating rise (segment 1) + gentle continued
# rise (segment 2), anchored together for continuity at the knee N_b.
# ----------------------------------------------------------------------

SEG1_RISE_MODEL_NAMES = ["rise_exp", "rise_stretched_exp", "rise_hill", "rise_logistic"]


def fit_segment1_rise(x, y, verbose=True):
    ranked = fit_all_rise_models(x, y, verbose=verbose, model_subset=SEG1_RISE_MODEL_NAMES)
    candidates = []
    for name, r in ranked:
        candidates.append(dict(
            kind="model", name=name, r2=r["r2"], aic=r["aic"],
            formula=get_rise_expression(name, r["params"]),
            eval_fn=lambda N, fn=r["fn"], popt=r["popt"]: fn(N, *popt),
        ))
    for deg in (2, 3):
        try:
            pf = _poly_fit(x, y, deg)
            candidates.append(dict(
                kind="poly", name=f"poly{deg}", r2=pf["r2"], aic=pf["aic"],
                formula=_poly_formula(pf["xmin"], pf["scale"], pf["coeffs"]),
                eval_fn=lambda N, pf=pf: _poly_eval(N, pf["xmin"], pf["scale"], pf["coeffs"]),
            ))
        except Exception as e:
            if verbose:
                print(f"  [poly{deg}] fit failed: {e}")

    candidates.sort(key=lambda c: c["aic"])
    if verbose:
        print("  -- segment 1 candidates (incl. poly) --")
        for c in candidates:
            print(f"  {c['name']:<18}{c['r2']:>10.5f}{c['aic']:>12.2f}")
    return candidates[0]


# --- Segment-2 anchor models: gentle continued rise past the knee -----

def anchored_linear(N, Nb, y_break, a):
    return y_break + a * (N - Nb)


def anchored_power_rise(N, Nb, y_break, a, p):
    """y_break + a*(N-Nb)^p. p<1 gives a tapering (concave) continued
    rise, p=1 is linear, p>1 lets it re-accelerate."""
    dN = np.clip(N - Nb, 0, None)
    return y_break + a * dN ** p


def anchored_quadratic(N, Nb, y_break, a, b):
    dN = N - Nb
    return y_break + a * dN + b * dN ** 2


def anchored_cubic(N, Nb, y_break, a, b, c):
    dN = N - Nb
    return y_break + a * dN + b * dN ** 2 + c * dN ** 3


def anchored_log(N, Nb, y_break, a, w):
    """Logarithmic-style continued rise - never truly flattens but grows
    ever more slowly, good for a tail that is still climbing at the
    largest N in the data."""
    dN = np.clip(N - Nb, 0, None)
    return y_break + a * np.log1p(dN / w)


def anchored_saturating(N, Nb, y_break, a, w):
    """A second, gentler saturating rise on top of y_break - use this if
    the tail looks like it is heading toward a second, higher plateau
    rather than climbing indefinitely."""
    dN = np.clip(N - Nb, 0, None)
    z = np.clip(dN / w, -700, 700)
    return y_break + a * w * (1.0 - np.exp(-z))


SEG2_RISE_MODELS = {
    "anchored_linear":     (anchored_linear,     ["a"]),
    "anchored_power_rise": (anchored_power_rise, ["a", "p"]),
    "anchored_quadratic":  (anchored_quadratic,  ["a", "b"]),
    "anchored_cubic":      (anchored_cubic,      ["a", "b", "c"]),
    "anchored_log":        (anchored_log,        ["a", "w"]),
    "anchored_saturating": (anchored_saturating, ["a", "w"]),
}


def _seg2_p0_polynomial(x, Nb, y, y_break, deg):
    dN = x - Nb
    A = np.column_stack([dN ** i for i in range(1, deg + 1)])
    dy = y - y_break
    coeffs, *_ = np.linalg.lstsq(A, dy, rcond=None)
    return list(coeffs)


def _seg2_slope0(x, Nb, y, y_break):
    dN = x.max() - Nb
    if dN <= 0:
        return 0.0
    return (y[-1] - y_break) / dN


SEG2_RISE_P0 = {
    "anchored_linear":     lambda x, Nb, y, y_break: [_seg2_slope0(x, Nb, y, y_break)],
    "anchored_power_rise": lambda x, Nb, y, y_break: [_seg2_slope0(x, Nb, y, y_break), 1.0],
    "anchored_quadratic":  lambda x, Nb, y, y_break: _seg2_p0_polynomial(x, Nb, y, y_break, 2),
    "anchored_cubic":      lambda x, Nb, y, y_break: _seg2_p0_polynomial(x, Nb, y, y_break, 3),
    "anchored_log":        lambda x, Nb, y, y_break: [
        max(y[-1] - y_break, 1e-6), (x.max() - Nb) * 0.3 if x.max() > Nb else 1.0],
    "anchored_saturating": lambda x, Nb, y, y_break: [
        _seg2_slope0(x, Nb, y, y_break), (x.max() - Nb) * 0.3 if x.max() > Nb else 1.0],
}

SEG2_RISE_BOUNDS = {
    "anchored_linear":     ([-np.inf], [np.inf]),
    "anchored_power_rise": ([0, 0.05], [np.inf, 10]),
    "anchored_quadratic":  ([-np.inf, -np.inf], [np.inf, np.inf]),
    "anchored_cubic":      ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
    "anchored_log":        ([0, 1e-10], [np.inf, np.inf]),
    "anchored_saturating": ([0, 1e-10], [np.inf, np.inf]),
}


def _seg2_rise_formula(name, Nb, y_break, popt):
    if name == "anchored_linear":
        (a,) = popt
        return f"{y_break:.6e}+{a:.6e}*(N-{Nb:.6e})"
    if name == "anchored_power_rise":
        a, p = popt
        return f"{y_break:.6e}+{a:.6e}*max(0,N-{Nb:.6e})**{p:.4f}"
    if name == "anchored_quadratic":
        a, b = popt
        return f"{y_break:.6e}+{a:.6e}*(N-{Nb:.6e})+{b:.6e}*(N-{Nb:.6e})**2"
    if name == "anchored_cubic":
        a, b, c = popt
        return (f"{y_break:.6e}+{a:.6e}*(N-{Nb:.6e})+{b:.6e}*(N-{Nb:.6e})**2"
                f"+{c:.6e}*(N-{Nb:.6e})**3")
    if name == "anchored_log":
        a, w = popt
        return f"{y_break:.6e}+{a:.6e}*log(1+max(0,N-{Nb:.6e})/{w:.6e})"
    if name == "anchored_saturating":
        a, w = popt
        return (f"{y_break:.6e}+{a:.6e}*{w:.6e}*"
                f"(1-exp(-max(0,N-{Nb:.6e})/{w:.6e}))")
    return ""


def find_breakpoint_knee(x, y):
    """Elbow/knee detector: the breakpoint is the data point with the
    largest perpendicular distance from the straight line joining the
    first and last point. This suits a smooth bend (steep rise -> gentle
    rise) far better than a max-jump detector, which is built for actual
    discontinuities/crashes (pauli_blocking.curve_fitting.find_breakpoint)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xn = (xs - xs.min()) / (xs.max() - xs.min())
    yspan = (ys.max() - ys.min())
    yn = (ys - ys.min()) / (yspan if yspan > 0 else 1.0)
    x0, y0 = xn[0], yn[0]
    x1, y1 = xn[-1], yn[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = np.hypot(dx, dy)
    dist = np.abs(dy * (xn - x0) - dx * (yn - y0)) / (norm if norm > 0 else 1.0)
    i = int(np.argmax(dist))
    return order, i, xs, ys, dist


def fit_piecewise_rise(x, y, title="", Nb_override=None, verbose=True, save_path=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    print(f"\n=== {title} (piecewise rise fit) ===")

    order, i, xs, ys, dist = find_breakpoint_knee(x, y)
    if Nb_override is not None:
        i = int(np.argmin(np.abs(xs - Nb_override)))
    Nb = xs[i]
    y_data_at_break = ys[i]
    if verbose:
        print(f"breakpoint (knee) auto-detected at N_b = {Nb:.5e}  "
              f"(data y there = {y_data_at_break:.5e})")

    seg1_x, seg1_y = xs[:i + 1], ys[:i + 1]
    seg2_x, seg2_y = xs[i:], ys[i:]  # include Nb in both, for anchoring

    if len(seg1_x) < 3:
        raise ValueError("Not enough points before the breakpoint to fit "
                          "segment 1 - try a smaller Nb_override.")
    if len(seg2_x) < 3:
        raise ValueError("Not enough points after the breakpoint to fit "
                          "segment 2 - try a larger Nb_override.")

    seg1 = fit_segment1_rise(seg1_x, seg1_y, verbose=verbose)
    y_break = float(np.atleast_1d(seg1["eval_fn"](np.array([Nb])))[0])

    if verbose:
        print(f"segment 1 best: {seg1['name']}  R^2={seg1['r2']:.5f}  "
              f"model value at N_b = {y_break:.5e} "
              f"(data at N_b = {y_data_at_break:.5e}, "
              f"mismatch = {abs(y_break - y_data_at_break):.3e})")
        if y_data_at_break != 0 and abs(y_break - y_data_at_break) / abs(y_data_at_break) > 0.05:
            print("  NOTE: segment-1 fit is not tightly anchored to the "
                  "data at N_b - consider a different Nb_override or "
                  "inspecting this curve by eye.")

    seg2_results = []
    n_seg2 = len(seg2_x) - 1
    for name, (fn, pnames) in SEG2_RISE_MODELS.items():
        try:
            def wrapped(N, *p, fn=fn):
                return fn(N, Nb, y_break, *p)
            p0 = SEG2_RISE_P0[name](seg2_x, Nb, seg2_y, y_break)
            popt, _ = curve_fit(wrapped, seg2_x[1:], seg2_y[1:], p0=p0,
                                 bounds=SEG2_RISE_BOUNDS[name], maxfev=20000)
            yfit = wrapped(seg2_x[1:], *popt)
            rss = np.sum((seg2_y[1:] - yfit) ** 2)
            tss = np.sum((seg2_y[1:] - seg2_y[1:].mean()) ** 2)
            r2 = 1 - rss / tss if tss > 0 else np.nan
            k = len(popt)
            aic = n_seg2 * np.log(rss / n_seg2 + 1e-300) + 2 * k
            seg2_results.append((name, r2, aic, popt, wrapped))
        except Exception as e:
            if verbose:
                print(f"  seg2 [{name}] fit failed: {e}")

    seg2_results.sort(key=lambda t: t[2])
    if verbose:
        print("  -- segment 2 (anchored) candidates --")
        for name, r2, aic, _, _ in seg2_results:
            print(f"  {name:<20}R^2={r2:>8.5f}  AIC={aic:>10.2f}")

    name2, r2_2, aic_2, popt2, fn2 = seg2_results[0]
    if verbose:
        print(f"segment 2 best: {name2}  R^2={r2_2:.5f}  params={popt2}")

    yfit_all = np.concatenate([seg1["eval_fn"](seg1_x), fn2(seg2_x[1:], *popt2)])
    y_all = np.concatenate([seg1_y, seg2_y[1:]])
    rss = np.sum((y_all - yfit_all) ** 2)
    tss = np.sum((y_all - y_all.mean()) ** 2)
    r2_combined = 1 - rss / tss if tss > 0 else np.nan
    if verbose:
        print(f"whole-curve piecewise R^2 = {r2_combined:.5f}")

    seg1_formula = seg1["formula"]
    seg2_formula = _seg2_rise_formula(name2, Nb, y_break, popt2)
    expr = f"if(N<{Nb:.6e}, {seg1_formula}, {seg2_formula})"

    seg1_at_Nb = float(np.atleast_1d(seg1["eval_fn"](np.array([Nb])))[0])
    seg2_at_Nb = float(fn2(np.array([Nb]), *popt2)[0])
    if verbose:
        print(f"continuity check at N_b: segment1={seg1_at_Nb:.6e}  "
              f"segment2={seg2_at_Nb:.6e}  "
              f"(should match to numerical precision: "
              f"{abs(seg1_at_Nb - seg2_at_Nb):.2e})")

    fig = _plot_piecewise_rise(seg1_x, seg1_y, seg2_x[1:], seg2_y[1:], Nb,
                                seg1["eval_fn"], fn2, popt2, r2_combined, title,
                                save_path=save_path)

    return dict(
        Nb=Nb, y_break=y_break,
        seg1_name=seg1["name"], seg1_r2=seg1["r2"], seg1_formula=seg1_formula,
        seg2_name=name2, seg2_r2=r2_2, seg2_formula=seg2_formula, seg2_popt=list(popt2),
        r2_combined=r2_combined, picwave_expr=expr, fig=fig,
    )


def _plot_piecewise_rise(seg1_x, seg1_y, seg2_x, seg2_y, Nb, seg1_eval, seg2_fn, seg2_popt,
                          r2_combined, title, save_path=None):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
    ax.axvline(Nb, color='0.6', lw=0.7, ls='--')

    ax.plot(seg1_x, seg1_y, 'o', ms=3, mfc='none', mec='k', mew=0.7, label='data')
    ax.plot(seg2_x, seg2_y, 'o', ms=3, mfc='none', mec='k', mew=0.7)

    x1d = np.linspace(seg1_x.min(), Nb, 200)
    x2d = np.linspace(Nb, seg2_x.max(), 200)
    ax.plot(x1d, seg1_eval(x1d), '-', color='C0', lw=1.4, label='segment 1 (rise)')
    ax.plot(x2d, seg2_fn(x2d, *seg2_popt), '-', color='C1', lw=1.4,
             label=fr'segment 2, $R^2_{{whole}}$={r2_combined:.4f}')

    ax.set_xlabel(r'$N$ (electron carrier density, cm$^{-3}$)')
    ax.set_ylabel('Rate')
    ax.set_title(title)
    ax.legend(frameon=False, loc='best', fontsize=7)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig

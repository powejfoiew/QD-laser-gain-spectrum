"""
General single-curve and piecewise curve-fitting utilities for saturating
blocking-factor curves (y falls from ~1 towards 0 or a plateau as the
independent population N increases), with PICWave-expression output.

Ported from the notebook cell that produced the thesis's Figure 4.1/5.2
blocking-factor fits, split out as a reusable module rather than three
near-identical copies (this notebook had independent, non-shared versions
of this same machinery for three different quantities — blocking factors,
a gain-ceiling fit, and a recombination-rate fit). Only the blocking-factor
case is wired up for now; the model registry and fitting/breakpoint logic
here are written generally enough that the other two could reuse this
module later instead of re-implementing it a third time.
"""

import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

FIG_WIDTH = 3.4   # inches -- single-column-sized panel
FIG_HEIGHT = 2.7  # inches


def apply_publication_style():
    """Apply the serif/STIX publication plot style used for all figures in
    this module. Call explicitly before plotting — not applied automatically
    on import, since mutating global matplotlib rcParams as an import
    side-effect would surprise anything else in the same process."""
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.4,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'savefig.dpi': 300,
        'figure.dpi': 150,
    })


# ----------------------------------------------------------------------
# Single-curve candidate models (used as-is for well-behaved curves, and as
# the segment-1 candidate pool for piecewise curves)
# ----------------------------------------------------------------------

def model_exp(N, N0):
    z = np.clip(N / N0, -700, 700)
    return np.exp(-z)


def model_fermi2d(N, N0, dE):
    z = np.clip(N / N0, -700, 700)
    dE = np.clip(dE, -700, 700)
    return 1.0 / (1.0 + (np.exp(z) - 1.0) * np.exp(-dE))


def model_power(N, N0, p):
    return 1.0 / (1.0 + np.abs(N / N0) ** p)


def model_stretched_exp(N, N0, beta):
    return np.exp(-np.abs(N / N0) ** beta)


def model_sigmoid(N, N0, w):
    z = np.clip((N - N0) / w, -700, 700)
    return 1.0 / (1.0 + np.exp(z))


def model_shifted_sigmoid(N, N0, w, L):
    """Logistic curve with a free (possibly negative) lower asymptote L,
    instead of the fixed lower asymptote of 0 used by model_sigmoid.
    Reduces to model_sigmoid when L = 0."""
    z = np.clip((N - N0) / w, -700, 700)
    return L + (1.0 - L) / (1.0 + np.exp(z))


def model_power_threshold(N, N0, p):
    """y = 1 - (N/N0)**p. Equals 1 at N=0, crosses zero at N=N0, and keeps
    falling (going negative) for N > N0. The base N/N0 is always >= 0, so
    this is well-defined for any real p - no clamping needed inside the
    pow()."""
    return 1.0 - (N / N0) ** p


def model_power_saturation(N, N0, n):
    """y = max(eps, 1 - N/N0)**n. The same functional family already used
    for NRADRATE_U_EXPRESSION in the material file (e.g.
    pow(max(1e-14, 1-N/Nmax), n)), applied directly to the blocking
    factor itself. For 0 < n < 1 it gives a genuine vertical tangent
    (infinite slope) at N = N0 - a much faster falloff than any sigmoid
    can produce - while still declining gently for N << N0 (since
    (1-x)**n ~ 1 - n*x for small x). Bounded in [0, 1] by construction,
    so unlike shifted_sigmoid/power_threshold it needs no extra clipping
    when used in PICWave."""
    u = np.clip(1.0 - N / N0, 1e-12, None)
    return u ** n


def model_gen_richards(N, N0, w, A, L, nu):
    """Generalized (Richards') logistic: adds an independent upper
    asymptote A (in case the true plateau isn't exactly 1) and an
    asymmetry parameter nu on top of shifted_sigmoid's free lower
    asymptote L. nu=1, A=1 reduces exactly to shifted_sigmoid; nu != 1
    lets the curve fall away from A at a different rate than it
    approaches L - something a symmetric sigmoid cannot do."""
    z = np.clip((N - N0) / w, -700, 700)
    return L + (A - L) / (1.0 + nu * np.exp(z)) ** (1.0 / nu)


MODELS = {
    "exp":              (model_exp,              ["N0"]),
    "fermi2d":          (model_fermi2d,          ["N0", "dE"]),
    "power":            (model_power,            ["N0", "p"]),
    "stretched_exp":    (model_stretched_exp,    ["N0", "beta"]),
    "sigmoid":          (model_sigmoid,          ["N0", "w"]),
    "shifted_sigmoid":  (model_shifted_sigmoid,  ["N0", "w", "L"]),
    "power_threshold":  (model_power_threshold,  ["N0", "p"]),
    "power_saturation": (model_power_saturation, ["N0", "n"]),
    "gen_richards":     (model_gen_richards,     ["N0", "w", "A", "L", "nu"]),
}


def format_formula(name, params):
    """Return the fitted equation as a plain-text string with numbers
    plugged in, e.g. 'exp(-N/2.903000e+17)'."""
    if name == "exp":
        return f"exp(-N/{params['N0']:.6e})"
    if name == "fermi2d":
        return (f"1/(1 + (exp(N/{params['N0']:.6e}) - 1)"
                f"*exp({-params['dE']:.4f}))")
    if name == "power":
        return f"1/(1 + (N/{params['N0']:.6e})**{params['p']:.4f})"
    if name == "stretched_exp":
        return f"exp(-(N/{params['N0']:.6e})**{params['beta']:.4f})"
    if name == "sigmoid":
        return f"1/(1 + exp((N-{params['N0']:.6e})/{params['w']:.6e}))"
    if name == "shifted_sigmoid":
        L = params['L']
        return (f"{L:.6f}+({1 - L:.6f})"
                f"/(1+exp((N-{params['N0']:.6e})/{params['w']:.6e}))")
    if name == "power_threshold":
        return f"1-(N/{params['N0']:.6e})**{params['p']:.4f}"
    if name == "power_saturation":
        return f"max(1e-12,1-N/{params['N0']:.6e})**{params['n']:.4f}"
    if name == "gen_richards":
        A, L, nu = params['A'], params['L'], params['nu']
        return (f"{L:.6f}+({A - L:.6f})"
                f"/((1+{nu:.6f}*exp((N-{params['N0']:.6e})/{params['w']:.6e}))"
                f"**{1.0 / nu:.6f})")
    return ""


def get_picwave_expression(name, params):
    """Same formula as format_formula(), wrapped in max(0, ...) only for
    models that can actually return a negative value: power_threshold
    always can; the sigmoid-family models only can if their fitted lower
    asymptote L came out negative. power_saturation (and the other
    originally-bounded models) never need the wrap."""
    formula = format_formula(name, params)
    needs_clip = (name == "power_threshold") or (params.get("L", 0.0) < 0.0)
    return f"max(0,{formula})" if needs_clip else formula


def _initial_N0(x, y):
    """50%-crossing guess, used for the models whose transition is
    centred on N0 (the sigmoid family)."""
    target = 0.5 * (y.max() + y.min())
    idx = np.argmin(np.abs(y - target))
    guess = x[idx]
    return guess if guess > 0 else x[np.argmax(x)] * 0.1


def _initial_zero_cross(x, y):
    """Zero-crossing guess, used for power_threshold's N0 (the density at
    which the curve is fit to actually hit zero). Falls back to just past
    the last data point if the data never reaches zero."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    below = np.where(ys <= 0)[0]
    if below.size:
        return xs[below[0]]
    return xs.max() * 1.05


def fit_all_models(x, y, verbose=True, model_subset=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    N0_guess = _initial_N0(x, y)
    N0_thresh_guess = _initial_zero_cross(x, y)

    p0_map = {
        "exp":              [N0_guess],
        "fermi2d":          [N0_guess, 0.0],
        "power":            [N0_guess, 1.0],
        "stretched_exp":    [N0_guess, 1.0],
        "sigmoid":          [N0_guess, N0_guess * 0.2 if N0_guess > 0 else 1.0],
        "shifted_sigmoid":  [N0_guess,
                             N0_guess * 0.1 if N0_guess > 0 else 1.0,
                             -0.5],
        "power_threshold":  [N0_thresh_guess, 5.0],
        "power_saturation": [N0_thresh_guess, 0.1],
        "gen_richards":     [N0_guess,
                             N0_guess * 0.1 if N0_guess > 0 else 1.0,
                             1.0, -0.5, 1.0],
    }
    bounds_map = {
        "exp":              ([1e-30], [np.inf]),
        "fermi2d":          ([1e-30, -50], [np.inf, 50]),
        "power":            ([1e-30, 0.05], [np.inf, 20]),
        "stretched_exp":    ([1e-30, 0.05], [np.inf, 20]),
        "sigmoid":          ([-np.inf, 1e-30], [np.inf, np.inf]),
        "shifted_sigmoid":  ([-np.inf, 1e-30, -50.0], [np.inf, np.inf, 0.999]),
        "power_threshold":  ([1e-30, 0.05], [np.inf, 50]),
        "power_saturation": ([1e-30, 0.01], [np.inf, 5.0]),
        "gen_richards":     ([-np.inf, 1e-30, 0.8, -50.0, 1e-3],
                             [np.inf, np.inf, 1.05, 0.999, 50.0]),
    }

    names = model_subset if model_subset is not None else MODELS.keys()
    results = {}
    for name in names:
        fn, pnames = MODELS[name]
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
                params=dict(zip(pnames, popt)),
                popt=popt, fn=fn, r2=r2, aic=aic, rss=rss,
            )
        except Exception as e:
            if verbose:
                print(f"  [{name}] fit failed: {e}")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["aic"])
    if verbose:
        print(f"{'model':<17}{'R^2':>10}{'AIC':>12}   f(N) =")
        for name, r in ranked:
            formula = format_formula(name, r["params"])
            print(f"{name:<17}{r['r2']:>10.5f}{r['aic']:>12.2f}   {formula}")
    return ranked


def _slugify(title):
    slug = re.sub(r"[^\w\-]+", "_", title.strip())
    return slug.strip("_").lower() or "fit"


def plot_fit(x, y, ranked, title="", save_path=None):
    best_name, best = ranked[0]
    x_dense = np.linspace(x.min(), x.max(), 400)
    y_dense = best["fn"](x_dense, *best["popt"])
    y_dense_clipped = np.maximum(y_dense, 0.0)
    goes_negative = np.any(y_dense < 0)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                           constrained_layout=True)

    ax.axhline(0, color='0.6', lw=0.7, ls=':')
    ax.plot(x, y, 'o', ms=3, mfc='none', mec='k', mew=0.7, label='data')

    if goes_negative:
        ax.plot(x_dense, y_dense, '--', color='C1', lw=1.0, alpha=0.7,
                 label=f'{best_name} raw fit')
        ax.plot(x_dense, y_dense_clipped, '-', color='C0', lw=1.4,
                 label=fr'clipped fit ($R^2$={best["r2"]:.4f})')
    else:
        ax.plot(x_dense, y_dense, '-', color='C0', lw=1.4,
                 label=fr'{best_name} fit ($R^2$={best["r2"]:.4f})')

    ax.set_xlabel(r'$N$ (density)')
    ax.set_ylabel(r'$1 - \langle \rho \rangle$')
    ax.set_title(title)
    ax.legend(frameon=False, loc='best')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def fit_and_plot(x, y, title="", save_path=None):
    print(f"\n=== {title} ===")
    ranked = fit_all_models(x, y)
    fig = plot_fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                    ranked, title=title, save_path=save_path)
    return ranked, fig


# ----------------------------------------------------------------------
# Piecewise fitting for curves that decline gently then either crash to
# zero, or bottom out and rise rapidly to a wall.
# ----------------------------------------------------------------------

SEG1_MODEL_NAMES = ["exp", "fermi2d", "power", "stretched_exp",
                     "sigmoid", "shifted_sigmoid"]


def _poly_fit(x, y, deg):
    xmin, xmax = x.min(), x.max()
    scale = xmax - xmin
    xn = (x - xmin) / scale
    c = np.polyfit(xn, y, deg)[::-1]  # increasing order: c0, c1, c2...
    y_fit = np.polyval(c[::-1], xn)
    rss = np.sum((y - y_fit) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    n = len(x)
    k = deg + 1
    aic = n * np.log(rss / n + 1e-300) + 2 * k
    return dict(xmin=xmin, scale=scale, coeffs=c, r2=r2, aic=aic)


def _poly_formula(xmin, scale, coeffs):
    terms = ", ".join(f"{c:.6e}" for c in coeffs)
    return f"poly1d((N-{xmin:.6e})/{scale:.6e}, {terms})"


def _poly_eval(N, xmin, scale, coeffs):
    xn = (np.asarray(N, dtype=float) - xmin) / scale
    return np.polyval(coeffs[::-1], xn)


def fit_segment1(x, y, verbose=True):
    ranked = fit_all_models(x, y, verbose=verbose, model_subset=SEG1_MODEL_NAMES)
    candidates = []
    for name, r in ranked:
        candidates.append(dict(
            kind="model", name=name, r2=r["r2"], aic=r["aic"],
            formula=get_picwave_expression(name, r["params"]),
            eval_fn=lambda N, fn=r["fn"], popt=r["popt"]: fn(N, *popt),
        ))
    for deg in (3, 4):
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
        print("  -- segment 1 candidates (incl. poly1d) --")
        for c in candidates:
            print(f"  {c['name']:<16}{c['r2']:>10.5f}{c['aic']:>12.2f}")
    return candidates[0]


# --- Segment-2 anchor models (continue from the segment-1/segment-2 breakpoint) ---

def anchored_power_saturation(N, Nb, y_break, W, n):
    u = np.clip(1.0 - (N - Nb) / W, 1e-12, None)
    return y_break * u ** n

def anchored_power_threshold(N, Nb, y_break, W, p):
    base = np.clip(1.0 - (N - Nb) / W, 0.0, None)
    return y_break * base ** p

def anchored_stretched_exp(N, Nb, y_break, W, beta):
    return y_break * np.exp(-np.abs((N - Nb) / W) ** beta)

def anchored_exp(N, Nb, y_break, W):
    return y_break * np.exp(-(N - Nb) / W)

def anchored_quadratic(N, Nb, y_break, a, b):
    dN = N - Nb
    return y_break + a * dN + b * dN ** 2

def anchored_cubic(N, Nb, y_break, a, b, c):
    """Extension of quadratic; highly flexible for dips followed by sharp walls."""
    dN = N - Nb
    return y_break + a * dN + b * dN ** 2 + c * dN ** 3

def anchored_exp_rise(N, Nb, y_break, a, b, W):
    """y = y_break + a*(N-Nb) + b*(exp((N-Nb)/W) - 1).
    Matches gentle linear bottoms but allows an explosive exponential blowup at the end."""
    dN = N - Nb
    return y_break + a * dN + b * (np.exp(np.clip(dN / W, -700, 700)) - 1.0)

def anchored_power_rise(N, Nb, y_break, a, b, p):
    """Similar to exp_rise, uses an arbitrary power p to model the sharp turn upward."""
    dN = N - Nb
    return y_break + a * dN + b * np.abs(dN) ** p


SEG2_MODELS = {
    "anchored_power_saturation": (anchored_power_saturation, ["W", "n"]),
    "anchored_power_threshold":  (anchored_power_threshold,  ["W", "p"]),
    "anchored_stretched_exp":    (anchored_stretched_exp,    ["W", "beta"]),
    "anchored_exp":              (anchored_exp,              ["W"]),
    "anchored_quadratic":        (anchored_quadratic,        ["a", "b"]),
    "anchored_cubic":            (anchored_cubic,            ["a", "b", "c"]),
    "anchored_exp_rise":         (anchored_exp_rise,         ["a", "b", "W"]),
    "anchored_power_rise":       (anchored_power_rise,       ["a", "b", "p"]),
}


def _seg2_p0_polynomial(x, Nb, y, y_break, deg):
    dN = x - Nb
    A = np.column_stack([dN ** i for i in range(1, deg + 1)])
    dy = y - y_break
    coeffs, *_ = np.linalg.lstsq(A, dy, rcond=None)
    return list(coeffs)

SEG2_P0 = {
    "anchored_power_saturation": lambda x, Nb, y, y_break: [(x.max() - Nb) * 1.05, 0.3],
    "anchored_power_threshold":  lambda x, Nb, y, y_break: [(x.max() - Nb) * 1.05, 3.0],
    "anchored_stretched_exp":    lambda x, Nb, y, y_break: [(x.max() - Nb) * 0.3, 1.0],
    "anchored_exp":              lambda x, Nb, y, y_break: [(x.max() - Nb) * 0.3],
    "anchored_quadratic":        lambda x, Nb, y, y_break: _seg2_p0_polynomial(x, Nb, y, y_break, 2),
    "anchored_cubic":            lambda x, Nb, y, y_break: _seg2_p0_polynomial(x, Nb, y, y_break, 3),
    "anchored_exp_rise":         lambda x, Nb, y, y_break: [0.0, 1e-4, (x.max() - Nb) * 0.1],
    "anchored_power_rise":       lambda x, Nb, y, y_break: [0.0, 1e-10, 4.0],
}

SEG2_BOUNDS = {
    "anchored_power_saturation": ([1e-10, 0.02], [np.inf, 5.0]),
    "anchored_power_threshold":  ([1e-10, 0.05], [np.inf, 50]),
    "anchored_stretched_exp":    ([1e-10, 0.05], [np.inf, 20]),
    "anchored_exp":              ([1e-10], [np.inf]),
    "anchored_quadratic":        ([-np.inf, -np.inf], [np.inf, np.inf]),
    "anchored_cubic":            ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
    "anchored_exp_rise":         ([-np.inf, -np.inf, 1e-10], [np.inf, np.inf, np.inf]),
    "anchored_power_rise":       ([-np.inf, -np.inf, 1.0], [np.inf, np.inf, 30.0]),
}


def _seg2_formula(name, Nb, y_break, popt):
    if name == "anchored_power_saturation":
        W, n = popt
        return f"{y_break:.6f}*max(1e-12,1-(N-{Nb:.6e})/{W:.6e})**{n:.4f}"
    if name == "anchored_power_threshold":
        W, p = popt
        return f"{y_break:.6f}*max(0,1-(N-{Nb:.6e})/{W:.6e})**{p:.4f}"
    if name == "anchored_stretched_exp":
        W, beta = popt
        return f"{y_break:.6f}*exp(-abs((N-{Nb:.6e})/{W:.6e})**{beta:.4f})"
    if name == "anchored_exp":
        (W,) = popt
        return f"{y_break:.6f}*exp(-(N-{Nb:.6e})/{W:.6e})"
    if name == "anchored_quadratic":
        a, b = popt
        return f"{y_break:.6f}+{a:.6e}*(N-{Nb:.6e})+{b:.6e}*(N-{Nb:.6e})**2"
    if name == "anchored_cubic":
        a, b, c = popt
        return f"{y_break:.6f}+{a:.6e}*(N-{Nb:.6e})+{b:.6e}*(N-{Nb:.6e})**2+{c:.6e}*(N-{Nb:.6e})**3"
    if name == "anchored_exp_rise":
        a, b, W = popt
        return f"{y_break:.6f}+{a:.6e}*(N-{Nb:.6e})+{b:.6e}*(exp((N-{Nb:.6e})/{W:.6e})-1)"
    if name == "anchored_power_rise":
        a, b, p = popt
        return f"{y_break:.6f}+{a:.6e}*(N-{Nb:.6e})+{b:.6e}*abs(N-{Nb:.6e})**{p:.4f}"
    return ""


def find_breakpoint(x, y):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    dy = np.diff(ys)
    i = int(np.argmax(np.abs(dy)))
    return order, i, xs, ys


def fit_piecewise(x, y, title="", Nb_override=None, verbose=True, save_path=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    print(f"\n=== {title} (piecewise) ===")

    order, i, xs, ys = find_breakpoint(x, y)
    if Nb_override is not None:
        i = int(np.argmin(np.abs(xs - Nb_override)))
    Nb = xs[i]
    y_data_at_break = ys[i]
    if verbose:
        print(f"breakpoint auto-detected at N_b = {Nb:.5e}  "
              f"(data y there = {y_data_at_break:.5f}; "
              f"next point jumps to y = {ys[i + 1]:.5f})")

    seg1_x, seg1_y = xs[:i + 1], ys[:i + 1]
    seg2_x, seg2_y = xs[i:], ys[i:]  # include Nb in both, for anchoring

    seg1 = fit_segment1(seg1_x, seg1_y, verbose=verbose)
    y_break = float(np.atleast_1d(seg1["eval_fn"](np.array([Nb])))[0])

    if verbose:
        print(f"segment 1 best: {seg1['name']}  R^2={seg1['r2']:.5f}  "
              f"model value at N_b = {y_break:.5f} "
              f"(data at N_b = {y_data_at_break:.5f}, "
              f"mismatch = {abs(y_break - y_data_at_break):.4f})")
        if abs(y_break - y_data_at_break) > 0.05:
            print("  NOTE: segment-1 fit is not tightly anchored to the "
                  "last data point before the crash - consider trying a "
                  "different breakpoint or inspecting this curve by eye.")

    seg2_results = []

    n_seg2 = len(seg2_x) - 1
    for name, (fn, pnames) in SEG2_MODELS.items():
        try:
            def wrapped(N, *p, fn=fn):
                return fn(N, Nb, y_break, *p)
            p0 = SEG2_P0[name](seg2_x, Nb, seg2_y, y_break)
            popt, _ = curve_fit(wrapped, seg2_x[1:], seg2_y[1:], p0=p0,
                                 bounds=SEG2_BOUNDS[name], maxfev=20000)
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

    seg2_results.sort(key=lambda t: t[2])  # sort by AIC

    if verbose:
        print("  -- segment 2 (anchored) candidates --")
        for name, r2, aic, _, _ in seg2_results:
            print(f"  {name:<28}R^2={r2:>8.5f}  AIC={aic:>10.2f}")

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
    seg2_formula = _seg2_formula(name2, Nb, y_break, popt2)
    picwave_expr = f"if(N<{Nb:.6e}, {seg1_formula}, {seg2_formula})"

    seg1_at_Nb = float(np.atleast_1d(seg1["eval_fn"](np.array([Nb])))[0])
    seg2_at_Nb = float(fn2(np.array([Nb]), *popt2)[0])
    if verbose:
        print(f"continuity check at N_b: segment1={seg1_at_Nb:.6f}  "
              f"segment2={seg2_at_Nb:.6f}  "
              f"(should match to numerical precision: "
              f"{abs(seg1_at_Nb - seg2_at_Nb):.2e})")

    fig = _plot_piecewise(seg1_x, seg1_y, seg2_x[1:], seg2_y[1:], Nb,
                           seg1["eval_fn"], fn2, popt2, r2_combined, title,
                           save_path=save_path)

    return dict(
        Nb=Nb, y_break=y_break,
        seg1_name=seg1["name"], seg1_r2=seg1["r2"], seg1_formula=seg1_formula,
        seg2_name=name2, seg2_r2=r2_2, seg2_formula=seg2_formula, seg2_popt=list(popt2),
        r2_combined=r2_combined, picwave_expr=picwave_expr, fig=fig,
    )


def _plot_piecewise(seg1_x, seg1_y, seg2_x, seg2_y, Nb, seg1_eval, seg2_fn, seg2_popt,
                     r2_combined, title, save_path=None):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
    ax.axhline(0, color='0.6', lw=0.7, ls=':')
    ax.axvline(Nb, color='0.6', lw=0.7, ls='--')

    ax.plot(seg1_x, seg1_y, 'o', ms=3, mfc='none', mec='k', mew=0.7, label='data')
    ax.plot(seg2_x, seg2_y, 'o', ms=3, mfc='none', mec='k', mew=0.7)

    x1d = np.linspace(seg1_x.min(), Nb, 200)
    x2d = np.linspace(Nb, seg2_x.max(), 200)
    ax.plot(x1d, seg1_eval(x1d), '-', color='C0', lw=1.4, label='segment 1 (gentle)')
    ax.plot(x2d, seg2_fn(x2d, *seg2_popt), '-', color='C1', lw=1.4,
             label=fr'segment 2, $R^2_{{whole}}$={r2_combined:.4f}')

    ax.set_xlabel(r'$N$ (density)')
    ax.set_ylabel(r'$1 - \langle \rho \rangle$')
    ax.set_title(title)
    ax.legend(frameon=False, loc='best', fontsize=7)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    return fig

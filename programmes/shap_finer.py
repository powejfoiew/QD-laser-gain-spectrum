#!/usr/bin/env python3
"""
QD DFB Laser Rate Equation Optimization: SHAP Explainability & Dashboard
=========================================================================

This script analyzes optimizer iteration data from PICWave QD DFB laser simulations.
It identifies which time constant parameters (τ) most strongly influence the 
optimization cost function using statistical correlations, Random Forest / XGBoost
surrogate modeling, and SHAP (SHapley Additive exPlanations) analysis.

Author: Expert Data Scientist / Semiconductor Laser Physicist
License: MIT
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the optimizer CSV file
CSV_FILENAME = r"C:\Users\josep\Documents\Prompt AWS\31.07.2026\tau_nelder_mead_history.csv"

# Output dashboard filename
OUTPUT_PNG = "qd_dfb_shap_dashboard.png"

# Model selection: 'random_forest' or 'xgboost'
MODEL_TYPE = "random_forest"  # Change to 'xgboost' if xgboost is installed

# SHAP analysis settings
SHAP_MAX_DISPLAY = 11  # Max features to display in SHAP plots

# Figure DPI for publication quality
FIGURE_DPI = 300

# Random state for reproducibility
RANDOM_STATE = 42

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import shap

# Try importing XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.1)


# =============================================================================
# MODULE 1: DATA LOADING & CLEANING
# =============================================================================

def load_and_clean_data(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the optimizer CSV and perform data cleaning.
    
    Steps:
        1. Load CSV with proper type inference.
        2. Filter out dead lasers and incomplete simulations.
        3. Identify and drop zero-variance (fixed) parameters.
        4. Return cleaned DataFrame and list of active τ parameter names.
    
    Parameters
    ----------
    csv_path : str
        Path to the optimizer iterations CSV file.
    
    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame with only valid runs and variable parameters.
    tau_columns : list[str]
        List of time constant column names that have non-zero variance.
    """
    print("=" * 70)
    print("MODULE 1: DATA LOADING & CLEANING")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n✓ Loaded '{csv_path}': {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Identify tau parameter columns (start with 'tau_')
    all_tau_columns = [col for col in df.columns if col.startswith('tau_')]
    print(f"✓ Detected {len(all_tau_columns)} τ parameter columns: {all_tau_columns}")
    
    # Filter invalid runs
    n_before = len(df)
    
    # Handle boolean columns that might be stored as strings
    for col in ['dead_laser', 'incomplete_sim']:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})
            df[col] = df[col].astype(bool)
    
    mask_valid = ~df.get('dead_laser', pd.Series(False, index=df.index)).fillna(False)
    mask_valid &= ~df.get('incomplete_sim', pd.Series(False, index=df.index)).fillna(False)
    
    df_clean = df[mask_valid].copy().reset_index(drop=True)
    n_after = len(df_clean)
    n_removed = n_before - n_after
    
    print(f"✓ Filtered invalid runs: {n_removed} removed "
          f"({n_removed/n_before*100:.1f}%), {n_after} valid runs remain")
    
    if n_after < 5:
        print("⚠ WARNING: Very few valid samples. Results may be unreliable.")
    
    # Drop zero-variance parameters
    zero_var_cols = []
    for col in all_tau_columns:
        if df_clean[col].nunique() <= 1 or df_clean[col].var() == 0:
            zero_var_cols.append(col)
    
    if zero_var_cols:
        print(f"✓ Dropping {len(zero_var_cols)} zero-variance (fixed) parameters: {zero_var_cols}")
        df_clean = df_clean.drop(columns=zero_var_cols)
    
    tau_columns = [col for col in all_tau_columns if col not in zero_var_cols]
    print(f"✓ Active (variable) τ parameters: {len(tau_columns)}")
    
    # Summary statistics
    print(f"\n  Cost range: [{df_clean['cost'].min():.4f}, {df_clean['cost'].max():.4f}]")
    print(f"  Cost mean ± std: {df_clean['cost'].mean():.4f} ± {df_clean['cost'].std():.4f}")
    if 'I_th_sim' in df_clean.columns:
        print(f"  I_th_sim range: [{df_clean['I_th_sim'].min():.2f}, {df_clean['I_th_sim'].max():.2f}] mA")
    
    return df_clean, tau_columns


# =============================================================================
# MODULE 2: STATISTICAL ANALYSIS
# =============================================================================

def compute_correlations(df: pd.DataFrame, tau_columns: list[str]) -> pd.DataFrame:
    """
    Compute Pearson (linear) and Spearman (non-linear/rank) correlations
    between each τ parameter and the cost function, plus I_th_sim if available.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    tau_columns : list[str]
        List of active τ parameter column names.
    
    Returns
    -------
    corr_df : pd.DataFrame
        DataFrame with Pearson and Spearman correlations and p-values.
    """
    print("\n" + "=" * 70)
    print("MODULE 2: STATISTICAL CORRELATION ANALYSIS")
    print("=" * 70)
    
    targets = ['cost']
    if 'I_th_sim' in df.columns:
        targets.append('I_th_sim')
    
    results = []
    
    for tau in tau_columns:
        row = {'parameter': tau}
        for target in targets:
            # Pearson correlation
            r_pearson, p_pearson = stats.pearsonr(df[tau], df[target])
            row[f'pearson_r_{target}'] = r_pearson
            row[f'pearson_p_{target}'] = p_pearson
            
            # Spearman rank correlation
            r_spearman, p_spearman = stats.spearmanr(df[tau], df[target])
            row[f'spearman_rho_{target}'] = r_spearman
            row[f'spearman_p_{target}'] = p_spearman
        
        results.append(row)
    
    corr_df = pd.DataFrame(results)
    
    # Print summary
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  Parameter Correlations with Cost Function                       │")
    print("├──────────────────┬──────────────┬──────────────┬────────────────┤")
    print("│  Parameter       │  Pearson r   │  Spearman ρ  │  Significance  │")
    print("├──────────────────┼──────────────┼──────────────┼────────────────┤")
    
    for _, row in corr_df.iterrows():
        sig = "***" if row['spearman_p_cost'] < 0.001 else \
              "**" if row['spearman_p_cost'] < 0.01 else \
              "*" if row['spearman_p_cost'] < 0.05 else "ns"
        print(f"│  {row['parameter']:<16s}│  {row['pearson_r_cost']:>+.4f}    │  "
              f"{row['spearman_rho_cost']:>+.4f}    │  p={row['spearman_p_cost']:.2e} {sig} │")
    
    print("└──────────────────┴──────────────┴──────────────┴────────────────┘")
    print("  Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    return corr_df


# =============================================================================
# MODULE 3: SURROGATE MODEL & SHAP ANALYSIS
# =============================================================================

def fit_surrogate_model(df: pd.DataFrame, tau_columns: list[str], 
                        model_type: str = "random_forest"):
    """
    Fit a tree-based surrogate model (Random Forest or XGBoost) to predict
    cost from τ parameters, then compute SHAP values for explainability.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    tau_columns : list[str]
        List of active τ parameter column names.
    model_type : str
        Either 'random_forest' or 'xgboost'.
    
    Returns
    -------
    model : fitted sklearn/xgboost model
        The trained surrogate model.
    shap_values : np.ndarray
        SHAP values array (n_samples × n_features).
    X : pd.DataFrame
        Feature matrix used for modeling.
    feature_importance_df : pd.DataFrame
        DataFrame ranking features by mean |SHAP| value.
    """
    print("\n" + "=" * 70)
    print("MODULE 3: SURROGATE MODEL & SHAP EXPLAINABILITY")
    print("=" * 70)
    
    X = df[tau_columns].copy()
    y = df['cost'].copy()
    
    print(f"\n✓ Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"✓ Target: cost (continuous)")
    
    # Select and fit model
    if model_type == "xgboost" and XGBOOST_AVAILABLE:
        print("✓ Model: XGBoost Regressor")
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0
        )
    else:
        if model_type == "xgboost" and not XGBOOST_AVAILABLE:
            print("⚠ XGBoost not installed. Falling back to Random Forest.")
        print("✓ Model: Random Forest Regressor")
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=max(2, len(X) // 20),
            min_samples_leaf=max(1, len(X) // 40),
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
    model.fit(X, y)
    
    # Cross-validation score (if enough data)
    if len(X) >= 10:
        cv_folds = min(5, len(X) // 3)
        if cv_folds >= 2:
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
            print(f"✓ Cross-validation R² (k={cv_folds}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train R²
    train_r2 = model.score(X, y)
    print(f"✓ Training R²: {train_r2:.4f}")
    
    # SHAP Analysis
    print("\n✓ Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Compute global feature importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'parameter': tau_columns,
        'mean_abs_shap': mean_abs_shap,
        'std_shap': np.abs(shap_values).std(axis=0),
        'max_abs_shap': np.abs(shap_values).max(axis=0)
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    
    # Normalize to percentage contribution
    total_shap = feature_importance_df['mean_abs_shap'].sum()
    feature_importance_df['importance_pct'] = (
        feature_importance_df['mean_abs_shap'] / total_shap * 100
    )
    
    # Print SHAP ranking
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  SHAP Feature Importance Ranking (Global)                        │")
    print("├────┬──────────────────┬───────────────┬──────────────────────────┤")
    print("│ #  │  Parameter       │  Mean |SHAP|  │  Contribution (%)        │")
    print("├────┼──────────────────┼───────────────┼──────────────────────────┤")
    
    for i, row in feature_importance_df.iterrows():
        bar = "█" * int(row['importance_pct'] / 3) + "░" * (20 - int(row['importance_pct'] / 3))
        print(f"│ {i+1:<2d} │  {row['parameter']:<16s}│  {row['mean_abs_shap']:.4f}       │  "
              f"{row['importance_pct']:>5.1f}% {bar[:15]} │")
    
    print("└────┴──────────────────┴───────────────┴──────────────────────────┘")
    
    # Directional impact analysis
    print("\n✓ Directional Impact Analysis (High parameter value → cost change):")
    for i, param in enumerate(tau_columns):
        param_values = X[param].values
        param_shap = shap_values[:, i]
        
        # Split into high/low halves
        median_val = np.median(param_values)
        high_mask = param_values >= median_val
        low_mask = param_values < median_val
        
        if high_mask.sum() > 0 and low_mask.sum() > 0:
            high_effect = param_shap[high_mask].mean()
            low_effect = param_shap[low_mask].mean()
            direction = "↑ increases cost" if high_effect > 0 else "↓ decreases cost"
            print(f"  {param:<16s}: High values {direction} "
                  f"(Δcost ≈ {high_effect:+.3f})")
    
    return model, shap_values, X, feature_importance_df


# =============================================================================
# MODULE 4: DASHBOARD VISUALIZATION
# =============================================================================

def create_dashboard(df: pd.DataFrame, tau_columns: list[str], 
                     corr_df: pd.DataFrame, model, shap_values: np.ndarray,
                     X: pd.DataFrame, feature_importance_df: pd.DataFrame,
                     output_path: str):
    """
    Generate a publication-quality 2×2 multi-panel dashboard.
    
    Panel A: Spearman Correlation Heatmap (τ params vs cost & I_th_sim)
    Panel B: SHAP Summary Beeswarm Plot
    Panel C: SHAP Dependence Plot for the top driver
    Panel D: Optimization Trajectory (top parameters vs iteration)
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    tau_columns : list[str]
        Active τ parameter names.
    corr_df : pd.DataFrame
        Correlation results DataFrame.
    model : fitted model
        Surrogate model.
    shap_values : np.ndarray
        SHAP values.
    X : pd.DataFrame
        Feature matrix.
    feature_importance_df : pd.DataFrame
        SHAP importance ranking.
    output_path : str
        File path for saving the PNG.
    """
    print("\n" + "=" * 70)
    print("MODULE 4: DASHBOARD VISUALIZATION")
    print("=" * 70)
    
    fig = plt.figure(figsize=(18, 14), dpi=FIGURE_DPI)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)
    
    # =========================================================================
    # PLOT A: Spearman Correlation Heatmap
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Build correlation matrix for heatmap
    targets_for_heatmap = ['cost']
    if 'I_th_sim' in df.columns:
        targets_for_heatmap.append('I_th_sim')
    
    # Create short parameter names for readability
    short_names = {col: col.replace('tau_', 'τ_') for col in tau_columns}
    
    heatmap_data = pd.DataFrame(index=[short_names[c] for c in tau_columns])
    
    for target in targets_for_heatmap:
        spearman_col = f'spearman_rho_{target}'
        if spearman_col in corr_df.columns:
            heatmap_data[target] = corr_df[spearman_col].values
    
    # Also add inter-parameter correlations with cost as a reference
    sns.heatmap(
        heatmap_data.astype(float),
        annot=True, fmt='.3f',
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Spearman ρ', 'shrink': 0.8},
        ax=ax_a
    )
    ax_a.set_title('(A) Spearman Rank Correlations\nτ Parameters vs. Targets', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_a.set_ylabel('')
    ax_a.tick_params(axis='y', rotation=0)
    ax_a.tick_params(axis='x', rotation=45)
    
    # =========================================================================
    # PLOT B: SHAP Beeswarm Summary Plot
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Use SHAP's built-in summary plot rendered onto our axis
    plt.sca(ax_b)
    shap.summary_plot(
        shap_values, X,
        feature_names=[short_names.get(c, c) for c in tau_columns],
        max_display=SHAP_MAX_DISPLAY,
        show=False,
        plot_size=None
    )
    ax_b.set_title('(B) SHAP Beeswarm: Feature Impact on Cost', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_b.set_xlabel('SHAP value (impact on cost prediction)')
    
    # =========================================================================
    # PLOT C: SHAP Dependence Plot for Top Driver
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    
    top_param = feature_importance_df.iloc[0]['parameter']
    top_param_idx = tau_columns.index(top_param)
    
    # Find best interaction feature (highest SHAP interaction)
    if len(tau_columns) > 1:
        # Use the second most important feature as interaction color
        second_param = feature_importance_df.iloc[1]['parameter']
        second_param_idx = tau_columns.index(second_param)
        interaction_col = X[second_param].values
        color_label = short_names.get(second_param, second_param)
    else:
        interaction_col = None
        color_label = None
    
    scatter = ax_c.scatter(
        X[top_param].values,
        shap_values[:, top_param_idx],
        c=interaction_col if interaction_col is not None else 'steelblue',
        cmap='viridis' if interaction_col is not None else None,
        alpha=0.7, edgecolors='k', linewidths=0.3, s=50
    )
    
    if interaction_col is not None:
        cbar = plt.colorbar(scatter, ax=ax_c, shrink=0.8)
        cbar.set_label(color_label, fontsize=9)
    
    ax_c.axhline(y=0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_c.set_xlabel(f'{short_names.get(top_param, top_param)} value', fontsize=10)
    ax_c.set_ylabel(f'SHAP value for {short_names.get(top_param, top_param)}', fontsize=10)
    ax_c.set_title(f'(C) SHAP Dependence: Top Driver\n({short_names.get(top_param, top_param)})',
                   fontsize=12, fontweight='bold', pad=10)
    
    # Add trend line
    z = np.polyfit(X[top_param].values, shap_values[:, top_param_idx], deg=2)
    x_sorted = np.sort(X[top_param].values)
    ax_c.plot(x_sorted, np.polyval(z, x_sorted), 'r-', linewidth=2, 
              alpha=0.7, label='Quadratic fit')
    ax_c.legend(loc='best', fontsize=9)
    
    # =========================================================================
    # PLOT D: Optimization Trajectory
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Plot cost trajectory
    iterations = df['iteration'].values if 'iteration' in df.columns else np.arange(1, len(df) + 1)
    
    color_cost = '#E74C3C'
    ax_d.plot(iterations, df['cost'].values, 'o-', color=color_cost, 
              linewidth=1.5, markersize=4, alpha=0.8, label='Cost')
    ax_d.set_xlabel('Iteration', fontsize=10)
    ax_d.set_ylabel('Cost', fontsize=10, color=color_cost)
    ax_d.tick_params(axis='y', labelcolor=color_cost)
    
    # Secondary y-axis for top parameters (normalized)
    ax_d2 = ax_d.twinx()
    
    # Plot top 2-3 parameters (normalized to [0,1] for visual comparison)
    n_top_params = min(3, len(feature_importance_df))
    colors_params = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i in range(n_top_params):
        param = feature_importance_df.iloc[i]['parameter']
        param_vals = df[param].values
        
        # Normalize to [0, 1]
        p_min, p_max = param_vals.min(), param_vals.max()
        if p_max > p_min:
            param_norm = (param_vals - p_min) / (p_max - p_min)
        else:
            param_norm = np.zeros_like(param_vals)
        
        ax_d2.plot(iterations, param_norm, 's--', color=colors_params[i],
                   linewidth=1.2, markersize=3, alpha=0.7,
                   label=f'{short_names.get(param, param)} (norm)')
    
    ax_d2.set_ylabel('Normalized τ Parameter Value', fontsize=10, color='#2E86AB')
    ax_d2.tick_params(axis='y', labelcolor='#2E86AB')
    ax_d2.set_ylim(-0.05, 1.15)
    
    # Combined legend
    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
                fontsize=8, framealpha=0.9)
    
    ax_d.set_title('(D) Optimization Trajectory\nCost & Top Parameters vs. Iteration',
                   fontsize=12, fontweight='bold', pad=10)
    
    # Highlight best iteration
    best_idx = df['cost'].idxmin()
    best_iter = iterations[best_idx]
    best_cost = df['cost'].iloc[best_idx]
    ax_d.annotate(f'Best: {best_cost:.3f}', xy=(best_iter, best_cost),
                  xytext=(best_iter + 0.5, best_cost + (df['cost'].max() - df['cost'].min()) * 0.1),
                  fontsize=8, color=color_cost,
                  arrowprops=dict(arrowstyle='->', color=color_cost, lw=1.2))
    
    # =========================================================================
    # Final layout adjustments and save
    # =========================================================================
    fig.suptitle('QD DFB Laser Rate Equation Optimization: SHAP Explainability Dashboard',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"\n✓ Dashboard saved: '{output_path}' ({FIGURE_DPI} DPI)")


# =============================================================================
# MODULE 5: EXECUTIVE SUMMARY
# =============================================================================

def print_executive_summary(df: pd.DataFrame, tau_columns: list[str],
                            corr_df: pd.DataFrame, 
                            feature_importance_df: pd.DataFrame):
    """
    Print a concise executive summary of key findings.
    """
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY: KEY OPTIMIZATION INSIGHTS")
    print("=" * 70)
    
    top_3 = feature_importance_df.head(3)
    
    print(f"\n📊 Dataset: {len(df)} valid iterations analyzed")
    print(f"📉 Best cost achieved: {df['cost'].min():.4f} (iteration "
          f"{df.loc[df['cost'].idxmin(), 'iteration'] if 'iteration' in df.columns else df['cost'].idxmin() + 1})")
    if 'I_th_sim' in df.columns:
        print(f"⚡ I_th range: {df['I_th_sim'].min():.2f} – {df['I_th_sim'].max():.2f} mA")
    
    print(f"\n🏆 TOP 3 MOST INFLUENTIAL PARAMETERS (by SHAP):")
    for i, (_, row) in enumerate(top_3.iterrows()):
        print(f"   {i+1}. {row['parameter']:<16s} — {row['importance_pct']:.1f}% contribution "
              f"(mean |SHAP| = {row['mean_abs_shap']:.4f})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Focus tuning efforts on: {top_3.iloc[0]['parameter']}")
    
    # Check if top parameter has strong monotonic relationship
    top_param = top_3.iloc[0]['parameter']
    top_corr_row = corr_df[corr_df['parameter'] == top_param].iloc[0]
    rho = top_corr_row['spearman_rho_cost']
    
    if abs(rho) > 0.5:
        direction = "DECREASE" if rho > 0 else "INCREASE"
        print(f"   • {direction} {top_param} to reduce cost (Spearman ρ = {rho:+.3f})")
    else:
        print(f"   • {top_param} shows non-linear effects — check SHAP dependence plot")
    
    # Low-importance parameters
    low_importance = feature_importance_df[feature_importance_df['importance_pct'] < 5]
    if len(low_importance) > 0:
        fixed_candidates = low_importance['parameter'].tolist()
        print(f"   • Consider fixing (low impact): {', '.join(fixed_candidates)}")
    
    print("\n" + "=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main pipeline orchestrating all analysis modules.
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  QD DFB LASER τ-PARAMETER OPTIMIZATION ANALYSIS                    ║")
    print("║  SHAP Explainability & Multi-Panel Dashboard                        ║")
    print("╚" + "═" * 68 + "╝\n")
    
    # Step 1: Load and clean data
    df_clean, tau_columns = load_and_clean_data(CSV_FILENAME)
    
    if len(df_clean) < 3:
        print("\n❌ ERROR: Insufficient valid data points for analysis (need ≥ 3).")
        return
    
    if len(tau_columns) < 1:
        print("\n❌ ERROR: No variable τ parameters found.")
        return
    
    # Step 2: Statistical correlations
    corr_df = compute_correlations(df_clean, tau_columns)
    
    # Step 3: Surrogate model + SHAP
    model, shap_values, X, feature_importance_df = fit_surrogate_model(
        df_clean, tau_columns, model_type=MODEL_TYPE
    )
    
    # Step 4: Dashboard visualization
    create_dashboard(
        df_clean, tau_columns, corr_df, model, 
        shap_values, X, feature_importance_df, OUTPUT_PNG
    )
    
    # Step 5: Executive summary
    print_executive_summary(df_clean, tau_columns, corr_df, feature_importance_df)
    
    print(f"\n✅ Analysis complete. Dashboard saved to: {OUTPUT_PNG}\n")


if __name__ == "__main__":
    main()
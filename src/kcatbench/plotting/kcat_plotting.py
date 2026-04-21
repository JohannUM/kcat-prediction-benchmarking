import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
import warnings
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from upsetplot import from_contents, UpSet
from kcatbench.util import RESULT_DIR

logger = logging.getLogger(__name__)


def _empty_metrics() -> dict[str, float]:
    """Return a metrics dict initialized with NaN values."""
    return {
        'pearson_r': np.nan,
        'srcc': np.nan,
        'r2': np.nan,
        'rmse': np.nan,
        'mae': np.nan
    }


def _format_metric(value: float) -> str:
    """Format numeric metrics for plot annotation and handle undefined values."""
    return f"{value:.2f}" if np.isfinite(value) else "N/A"


def _get_ellipse_inlier_mask(x: pd.Series, y: pd.Series, n_std: float = 3.0) -> np.ndarray:
    """Return a boolean mask for points inside an n-std covariance ellipse."""
    if len(x) == 0:
        return np.array([], dtype=bool)

    xy = np.column_stack((np.asarray(x), np.asarray(y)))
    if xy.shape[0] < 2:
        return np.zeros(xy.shape[0], dtype=bool)

    cov = np.cov(xy, rowvar=False)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        logger.warning("Unable to compute stable covariance for ellipse inlier mask.")
        return np.zeros(xy.shape[0], dtype=bool)

    inv_cov = np.linalg.pinv(cov)
    centered = xy - np.mean(xy, axis=0)
    dist_sq = np.einsum('ij,jk,ik->i', centered, inv_cov, centered)
    return np.isfinite(dist_sq) & (dist_sq <= (n_std ** 2))


def _compute_comparison_metrics(
    x: pd.Series,
    y: pd.Series,
    include_ground_truth_metrics: bool = True
) -> dict[str, float]:
    """Compute comparison metrics, treating y as ground truth when requested."""
    metrics = _empty_metrics()

    if len(x) == 0:
        return metrics

    if include_ground_truth_metrics:
        if len(x) >= 2:
            metrics['r2'] = r2_score(y, x)
        metrics['rmse'] = np.sqrt(mean_squared_error(y, x))
        metrics['mae'] = mean_absolute_error(y, x)

    if len(x) < 2 or np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        logger.warning("Correlation metrics undefined due to low sample count or zero variance.")
        return metrics

    metrics['pearson_r'], _ = pearsonr(x, y)
    metrics['srcc'], _ = spearmanr(x, y)

    return metrics


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Source: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def plot_model_comparison(
    df:pd.DataFrame, 
    model_x:str, 
    model_y:str, 
    model_x_name:str, 
    model_y_name:str, 
    log_scale=True, 
    gridsize=50, 
    vmax=None, 
    save=False, 
    show=True,
    model_vs_model: bool = False,
    show_ellipse_stats: bool = False,
    ellipse_stats_position: str = 'upper_right'
):
    """
    Generate a square-aspect hexbin plot comparing two sets of kcat values.
    
    This function cleans input data (extracting scalars from lists/arrays), calculates 
    agreement statistics and visualizes density with a 3-std confidence ellipse
    and marginal framing ticks.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the model prediction and/or experimental columns.
    model_x : str
        The column name in `df` to be plotted on the x-axis.
    model_y : str
        The column name in `df` to be plotted on the y-axis.
    model_x_name : str
        The display label for the x-axis (e.g., 'DLKcat').
    model_y_name : str
        The display label for the y-axis (e.g., 'Experimental').
    log_scale : bool, default True
        If True, applies log10 transformation and uses 10^x axis formatting.
    gridsize : int, default 50
        The number of hexagons in the x-direction. Controls plot resolution.
    vmax : int or float, optional
        The maximum value for the colorbar scale. Useful for normalizing 
        density colors across multiple plots.
    save : bool, default False
        If True, saves the figure to the project's results directory.
    show : bool, default True
        If False, does not show the figure.
    model_vs_model : bool, default False
        If True, omits ground-truth-dependent metrics (R², RMSE, MAE) from the
        annotation for model-vs-model comparisons.
        If False, y-axis values are treated as ground truth.
    show_ellipse_stats : bool, default False
        If True, adds an optional secondary panel with metrics computed using
        only points inside the confidence ellipse.
    ellipse_stats_position : str, default 'upper_right'
        Position of the secondary panel. Must be one of: 'upper_right',
        'upper_left', 'lower_right', 'lower_left'.

    Returns
    -------
    None
        The function renders the plot using plt.show() and optionally saves it.

    Notes
    -----
        - Input data is automatically cleaned: if a cell contains a list, tuple, or
            numpy array, the first element is extracted.
        - Values <= 0 are filtered out when `log_scale` is True.
        - Ground-truth-dependent metrics assume the y-axis (`model_y`) contains
            experimental values.
        - The optional ellipse panel is a robustness summary and does not replace
            full-data metrics.
        - Red minor ticks on axes indicate the 3rd-standard-deviation boundaries
            of the covariance ellipse.
    """

    if not model_vs_model and model_x == 'experimental_kcat' and model_y != 'experimental_kcat':
        logger.info("Swapping axes so experimental_kcat is on the y-axis for metrics.")
        model_x, model_y = model_y, model_x
        model_x_name, model_y_name = model_y_name, model_x_name

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.set_style("ticks")

    plot_data = df[[model_x, model_y]].dropna().copy()

    for col in [model_x, model_y]:
        plot_data[col] = plot_data[col].apply(
            lambda x: x[0] if type(x) in [list, np.ndarray, tuple] else x
        )
        
        plot_data[col] = pd.to_numeric(plot_data[col], errors='coerce')
    
    plot_data = plot_data[[model_x, model_y]].dropna()
    plot_data = plot_data[~plot_data.isin([np.inf, -np.inf]).any(axis=1)]

    if log_scale:
        plot_data = plot_data[(plot_data[model_x] > 0) & (plot_data[model_y] > 0)]
    
    if len(plot_data) == 0:
        print("No valid overlapping data points found.")
        return
    
    x_vals = plot_data[model_x]
    y_vals = plot_data[model_y]

    if not model_vs_model and model_y != 'experimental_kcat':
        logger.warning(
            "Ground-truth-dependent metrics assume y-axis is experimental data. "
            "Received y column '%s'.",
            model_y
        )

    data_min = min(x_vals.min(), y_vals.min())
    data_max = max(x_vals.max(), y_vals.max())
    
    if log_scale:
        plot_x = np.log10(x_vals)
        plot_y = np.log10(y_vals)
        
        pad_factor = 2.0
        safe_min = data_min if data_min > 1e-10 else 1e-4
        lower_limit = np.log10(safe_min / pad_factor)
        upper_limit = np.log10(data_max * pad_factor)
    else:
        plot_x = x_vals
        plot_y = y_vals
                      
        pad = (data_max - data_min) * 0.05
        lower_limit = data_min - pad
        upper_limit = data_max + pad

    n_std = 3.0

    metrics = _compute_comparison_metrics(
        plot_x,
        plot_y,
        include_ground_truth_metrics=not model_vs_model
    )
    stats_lines = [
        f"$N = {len(plot_data)}$",
        f"Pearson $r = {_format_metric(metrics['pearson_r'])}$",
        f"SRCC = {_format_metric(metrics['srcc'])}"
    ]
    if not model_vs_model:
        stats_lines.extend([
            f"$R^2 = {_format_metric(metrics['r2'])}$",
            f"RMSE = {_format_metric(metrics['rmse'])}",
            f"MAE = {_format_metric(metrics['mae'])}"
        ])
    stats_text = "\n".join(stats_lines)

    ellipse_stats_text = None
    if show_ellipse_stats:
        if model_vs_model:
            logger.info(
                "Skipping ellipse secondary stats panel because model_vs_model=True."
            )
        else:
            inlier_mask = _get_ellipse_inlier_mask(plot_x, plot_y, n_std=n_std)
            inside_count = int(np.sum(inlier_mask))
            total_count = len(plot_x)
            inside_pct = (inside_count / total_count) * 100 if total_count > 0 else np.nan

            min_inside_points = 5
            if inside_count < min_inside_points:
                inside_metrics = _empty_metrics()
            else:
                inside_metrics = _compute_comparison_metrics(
                    plot_x[inlier_mask],
                    plot_y[inlier_mask],
                    include_ground_truth_metrics=True
                )

            ellipse_lines = [
                "Inside Ellipse",
                f"$N_{{in}} = {inside_count}$ ({inside_pct:.1f}%)",
                f"Pearson $r = {_format_metric(inside_metrics['pearson_r'])}$",
                f"SRCC = {_format_metric(inside_metrics['srcc'])}",
                f"$R^2 = {_format_metric(inside_metrics['r2'])}$",
                f"RMSE = {_format_metric(inside_metrics['rmse'])}",
                f"MAE = {_format_metric(inside_metrics['mae'])}"
            ]
            if inside_count < min_inside_points:
                ellipse_lines.append(f"(metrics require >= {min_inside_points} points)")
            ellipse_stats_text = "\n".join(ellipse_lines)

    hb = ax.hexbin(
        plot_x, 
        plot_y, 
        gridsize=gridsize, 
        cmap='inferno_r',
        mincnt=1,     
        edgecolors='none',
        extent=[lower_limit, upper_limit, lower_limit, upper_limit],
        vmax=vmax
    )

    confidence_ellipse(plot_x, plot_y, ax, n_std=n_std, edgecolor='red', linestyle='--', linewidth=1)
    
    cov_matrix = np.cov(plot_x, plot_y)
    std_x = np.sqrt(cov_matrix[0, 0])
    std_y = np.sqrt(cov_matrix[1, 1])
    mean_x = np.mean(plot_x)
    mean_y = np.mean(plot_y)
    ellipse_x_bounds = [mean_x - (n_std * std_x), mean_x + (n_std * std_x)]
    ellipse_y_bounds = [mean_y - (n_std * std_y), mean_y + (n_std * std_y)]

    cb = plt.colorbar(
        hb, 
        label='Count', 
        shrink=0.5,     
        aspect=20,      
        pad=0.05      
    )
    
    plt.plot([lower_limit, upper_limit], [lower_limit, upper_limit], 
             color='black',     
             linestyle='--',      
             alpha=0.6,        
             linewidth=1.0, 
             label='Perfect Agreement')

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )

    if ellipse_stats_text is not None:
        panel_positions = {
            'upper_right': (0.95, 0.95, 'right', 'top'),
            'upper_left': (0.05, 0.95, 'left', 'top'),
            'lower_right': (0.95, 0.05, 'right', 'bottom'),
            'lower_left': (0.05, 0.05, 'left', 'bottom')
        }
        if ellipse_stats_position not in panel_positions:
            logger.warning(
                "Unknown ellipse_stats_position '%s'. Falling back to 'upper_right'.",
                ellipse_stats_position
            )
            ellipse_stats_position = 'upper_right'

        x_text, y_text, ha, va = panel_positions[ellipse_stats_position]
        ax.text(
            x_text,
            y_text,
            ellipse_stats_text,
            transform=ax.transAxes,
            fontsize=10,
            horizontalalignment=ha,
            verticalalignment=va,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
        )

    if log_scale:
        plt.xlabel(f"{model_x_name} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=14)
        plt.ylabel(f"{model_y_name} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=14)

        log_formatter = FuncFormatter(lambda x, pos: f"$10^{{{x:g}}}$")
        
        ax.xaxis.set_major_formatter(log_formatter)
        ax.yaxis.set_major_formatter(log_formatter)
    else:
        plt.xlabel(f"{model_x_name} $k_{{cat}}$ ($s^{{-1}}$)", fontsize=12)
        plt.ylabel(f"{model_y_name} $k_{{cat}}$ ($s^{{-1}}$)", fontsize=12)
        
    plt.xlim(lower_limit, upper_limit)
    plt.ylim(lower_limit, upper_limit)
    
    plt.gca().set_box_aspect(1)
    sns.despine()

    ax.set_xticks(ellipse_x_bounds, minor=True)
    ax.set_yticks(ellipse_y_bounds, minor=True)
    ax.tick_params(which='minor', color='red', length=8, width=2, direction='in')

    cb.ax.minorticks_off()
    
    plt.title(f"{model_x_name} vs {model_y_name}", fontsize=16, fontweight='bold')
    
    plt.tight_layout()

    if save: 
        save_dir = RESULT_DIR / "plots" / "comparison_plots"
        save_dir.mkdir(exist_ok=True)
        plt.savefig(
            str(save_dir / f"{model_x_name}_vs_{model_y_name}.png"), 
            dpi=300,             
            bbox_inches='tight', 
            transparent=False   
        )
    if show:
        plt.show()
    else:
        plt.close(fig)



def get_performance_subsets(
    df:pd.DataFrame, 
    models:list[str], 
    percentage=10, 
    subset_type='best'
) -> dict[str, set]:
    r"""
    Extract subsets of Reaction IDs based on model prediction error thresholds.
    
    This function calculates the absolute log10 error for each model, filters out 
    invalid data (non-positive or NaN), and returns the IDs of the best or 
    worst performing reactions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing an 'ID' column, an 'experimental_kcat' column, 
        and model columns named '{model}_kcat'.
    models : list[str]
        List of model identifiers used to locate the relevant columns in `df`.
    percentage : int, default 10
        The percentage of data to include in the subset (e.g., top 10%).
    subset_type : str, default 'best'
        The type of performance subset to extract. Must be 'best' (lowest error) 
        or 'worst' (highest error).

    Returns
    -------
    dict[str, set]
        A dictionary where keys are model names and values are sets of 
        Reaction IDs belonging to the performance subset.

    Notes
    -----
    - Error is calculated as: $|\log_{10}(k_{cat, pred}) - \log_{10}(k_{cat, exp})|$
    - If a model's prediction is stored as a list or array, the first element 
      is automatically extracted.
    - Non-positive kcat values are excluded from the calculation.
    """
    
    if subset_type not in ['best', 'worst']:
        raise ValueError("subset_type must be either 'best' or 'worst'")
    
    model_sets = {}
    
    exp_mask = (df['experimental_kcat'] > 0) & df['experimental_kcat'].notna()

    for model in models:
        mod_col = model

        if mod_col not in df.columns:
            logger.error(f"Column {mod_col} not found in DataFrame. Skipping model {model}.")
            continue

        clean_df = df.loc[exp_mask, ['ID', 'experimental_kcat', mod_col]].copy()

        clean_df[mod_col] = clean_df[mod_col].apply(
            lambda x: x[0] if isinstance(x, (list, np.ndarray, tuple)) else x
        )
        
        clean_df[mod_col] = pd.to_numeric(clean_df[mod_col], errors='coerce')
        
        model_valid_mask = (clean_df[mod_col] > 0) & clean_df[mod_col].notna()
        clean_df = clean_df[model_valid_mask]
        
        if clean_df.empty:
            logger.warning(f"No valid overlapping data found for model: {model}")
            model_sets[model] = set()
            continue
            
        errors = np.abs(np.log10(clean_df[mod_col]) - np.log10(clean_df['experimental_kcat']))
        
        if subset_type == 'best':
            threshold = np.percentile(errors, percentage)
            subset_mask = errors <= threshold
        else:
            threshold = np.percentile(errors, 100 - percentage)
            subset_mask = errors >= threshold
            
        model_sets[model] = set(clean_df.loc[subset_mask, 'ID'])
        
    return model_sets



def plot_model_intersection_sets(
    df: pd.DataFrame,
    model_names: dict[str, str],
    percentage: int = 10,
    subset_type: str = 'best',
    save: bool = False,
    show: bool = True
) -> None:
    """
    Generate an UpSet plot visualizing the intersection of model performance subsets.

    This function identifies the top or bottom percentage of predictions for each 
    model based on log-error, maps internal model IDs to display names, and 
    renders an UpSet plot to show overlapping consensus across the models.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing 'ID', 'experimental_kcat', and model 
        columns formatted as '{model_id}_kcat'.
    model_names : dict[str, str]
        A mapping where keys are internal model IDs (matching columns in df) 
        and values are the "pretty" names for plot labels.
    percentage : int, default 10
        The percentile threshold (0-100) used to define the 'best' or 'worst' 
        performance subsets.
    subset_type : str, default 'best'
        The performance category to analyze. Must be either 'best' (lowest error) 
        or 'worst' (highest error).
    save : bool, default False
        If True, automatically saves the plot as a high-resolution PNG.
    show : bool, default True
        If True, calls plt.show() to display the plot immediately.

    Returns
    -------
    None
        The function renders/saves the plot and returns nothing.

    Notes
    -----
    - This function depends on `get_performance_subsets` to calculate error 
      and extract Reaction IDs.
    """

    model_sets = get_performance_subsets(df, list(model_names.keys()), percentage=percentage, subset_type=subset_type)

    if not model_sets or all(len(s) == 0 for s in model_sets.values()):
        logger.error("Error: No data to plot.")
        return

    plot_ready_sets = {}
    for old_name, reaction_set in model_sets.items():
        new_name = model_names.get(old_name, old_name) 
        plot_ready_sets[new_name] = reaction_set
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*downcasting.*", category=FutureWarning)
        upset_data = from_contents(plot_ready_sets)
    
    upset = UpSet(
        upset_data, 
        subset_size='count', 
        show_counts=False, 
        sort_by='cardinality',
        sort_categories_by='cardinality',
        facecolor="gray",
        element_size=40
    )

    num_models = len(model_names)
    cmap = LinearSegmentedColormap.from_list("muted_gradient", ["#316FD3", "#D2151C"])
    colors = [cmap(i) for i in np.linspace(0, 1, num_models)]

    for degree in range(1, num_models + 1):
        upset.style_subsets(min_degree=degree, max_degree=degree, facecolor=colors[degree - 1])
    
    fig = plt.figure(figsize=(10, 6))

    axes_dict = upset.plot(fig=fig)

    axes_dict['intersections'].tick_params(axis='y', labelsize=13)
    axes_dict['intersections'].set_ylabel('Intersection size', fontsize=15)

    axes_dict['totals'].tick_params(axis='x', labelsize=13)

    axes_dict['matrix'].tick_params(axis='y', labelsize=15) 

    plt.suptitle(f"Intersection of {percentage}% {subset_type} predictions", fontsize=24, fontweight='bold')
    
    if save:
        save_dir = RESULT_DIR / "plots" / "intersection_plots"
        save_dir.mkdir(exist_ok=True)
        plt.savefig(
            str(save_dir / f"{subset_type}_{percentage}_intersections.png"), 
            dpi=300,             
            bbox_inches='tight', 
            transparent=False   
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
        
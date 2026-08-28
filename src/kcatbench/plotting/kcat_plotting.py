import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
import warnings
import re
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker
from matplotlib.patches import Ellipse, Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, Colormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from typing import Optional, Union
from upsetplot import from_contents, UpSet
from kcatbench.util import RESULT_DIR
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

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


def _extract_scalar(value):
    """Extract scalar from list-like values used in prediction columns."""
    if isinstance(value, (list, np.ndarray, tuple)):
        if len(value) == 0:
            return np.nan
        return value[0]
    return value


def _resolve_model_column(model_name: str) -> str:
    """Resolve model identifier to dataframe column name."""
    if model_name.endswith('_kcat'):
        return model_name
    return f"{model_name}_kcat"


def _resolve_save_target(
    save: bool,
    save_path: Optional[Union[str, Path]],
    default_dir: Path,
    default_filename: str
) -> Optional[Path]:
    """Resolve a save target path and ensure the parent directory exists."""
    if not save:
        return None

    if save_path is None:
        target_path = Path(default_dir) / default_filename
    else:
        candidate = Path(save_path).expanduser()
        if candidate.suffix:
            target_path = candidate
        else:
            target_path = candidate / default_filename

    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


def _parse_sequence_similarity_label(label: str) -> Optional[tuple[str, str]]:
    """Parse a sequence-similarity label into dataset and bin identifiers."""
    label_text = str(label)
    dataset_name = 'EnzyExtract' if label_text.startswith('enzy_') else 'BRENDA'
    core_label = label_text[5:] if label_text.startswith('enzy_') else label_text

    if core_label == 'mean_seq':
        return dataset_name, 'mean_seq'

    match = re.search(r'alignment_(\d+)(?:_(\d+))?$', core_label)
    if match is None:
        return None

    lower = int(match.group(1))
    upper = match.group(2)

    if upper is None:
        if lower != 100:
            return None
        bin_label = '100%'
    else:
        bin_label = f'{lower}%-<{upper}%'

    return dataset_name, bin_label


def _format_sequence_similarity_value(value: float, value_mode: str) -> str:
    """Format a sequence-similarity bin value for in-bar annotation."""
    if value_mode == 'percent':
        return f'{value:.1f}%'

    if np.isclose(value, round(value)):
        return f'{int(round(value)):,}'

    return f'{value:.1f}'


def plot_standalone_colorbar(
    vmin: float,
    vmax: float,
    cmap: Union[str, Colormap],
    label: str = "",
    orientation: str = 'vertical',
    figsize: Optional[tuple[float, float]] = None,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Plot a standalone colorbar for a specific range and colormap.
    
    This is highly useful for generating unified legends for multi-panel 
    publication figures.

    Parameters
    ----------
    vmin : float
        The minimum value of the data range.
    vmax : float
        The maximum value of the data range.
    cmap_name : str
        The name of the Matplotlib/Seaborn colormap (e.g., 'rocket', 'vlag').
    label : str, default ""
        The text label to display alongside the colorbar.
    orientation : str, default 'vertical'
        Must be 'vertical' or 'horizontal'.
    figsize : tuple[float, float], optional
        The dimensions of the figure. If None, defaults are chosen 
        intelligently based on orientation.
    save : bool, default False
        If True, saves the figure.
    save_path : str or Path, optional
        Target path or directory for saving the figure.
    show : bool, default True
        If True, displays the figure; otherwise closes it.
    """
    
    if orientation not in ['vertical', 'horizontal']:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    if figsize is None:
        figsize = (1.5, 6.0) if orientation == 'vertical' else (6.0, 1.5)

    fig, ax = plt.subplots(figsize=figsize)
    
    norm = Normalize(vmin=vmin, vmax=vmax)

    if isinstance(cmap, str):
        resolved_cmap = plt.get_cmap(cmap)
    else:
        resolved_cmap = cmap
    
    sm = ScalarMappable(cmap=resolved_cmap, norm=norm)
    sm.set_array([]) 
    
    cbar = fig.colorbar(sm, cax=ax, orientation=orientation)
    
    if label:
        cbar.set_label(label, fontsize=12, labelpad=10)
        
    cbar.ax.tick_params(labelsize=11)
    
    plt.tight_layout()

    if save:
        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / "plots" / "colorbars",
            f"colorbar_{orientation}.png"
        )
        if save_target is not None:
            plt.savefig(
                str(save_target),
                dpi=300,
                bbox_inches='tight'
            )

    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_dual_dataset_correlation_heatmap(
    dataset_1: pd.DataFrame,
    dataset_name_1: str,
    dataset_2: pd.DataFrame,
    dataset_name_2: str,
    model_names: dict[str, str],
    metric: str = 'pearson',
    log_scale: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    y_axis_right: bool = False,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    cmap_colors: Optional[list[str]] = None,
    model_colors: Optional[dict[str, str]] = None
) -> None:
    """
    Plot a dual-dataset correlation heatmap for model predictions.

    This function calculates intra-model prediction correlations for two distinct
    datasets using Pearson or Spearman metrics. It packs both correlation matrices 
    into a single square grid split along the counter-diagonal: the upper-left 
    triangle represents dataset 1, and the lower-right triangle represents dataset 2.

    Parameters
    ----------
    dataset_1 : pd.DataFrame
        The first input dataset containing model prediction columns.
    dataset_name_1 : str
        Display name for the first dataset, rendered at the top of the plot.
    dataset_2 : pd.DataFrame
        The second input dataset containing model prediction columns.
    dataset_name_2 : str
        Display name for the second dataset, rendered on the right of the plot.
    model_names : dict[str, str]
        Mapping from model identifier keys to their human-readable display names.
    metric : str, default 'pearson'
        Correlation metric to compute. Options are 'pearson', 'srcc', or 'spearman'.
    log_scale : bool, default True
        If True, transforms data to log10 space before calculating correlations.
    vmin : float, default 0.0
        Minimum value bound anchor for the colormap data scale.
    vmax : float, default 1.0
        Maximum value bound anchor for the colormap data scale.
    y_axis_right : bool, default False
        If True, shifts the Y-axis tick labels and spine to the right side.
    save : bool, default False
        If True, saves the generated figure to disk.
    save_path : str or Path, optional
        Custom directory path or complete file path string to override default save tracking.
    show : bool, default True
        If True, displays the plot interactively using plt.show().
    cmap_colors : list[str], optional
        Custom color list for color mapping. If 2 colors are provided, a dual-gradient layout
        is applied (white-to-color1 for dataset 1, white-to-color2 for dataset 2). If 3 colors
        are provided, a unified global linear colormap gradient is generated across both triangles.
    model_colors : dict[str, str], optional
        Mapping from model identifier keys to valid color strings. If provided, colors the 
        respective model label strings on both the X and Y axes.

    Returns
    -------
    None
        Renders or saves the plot configuration directly.

    Raises
    ------
    ValueError
        If an invalid correlation metric or fewer than two models are provided.
    """
    if metric.lower() not in ['pearson', 'srcc', 'spearman']:
        raise ValueError("metric must be 'pearson' or 'srcc'")
        
    pandas_corr_method = 'spearman' if metric.lower() in ['srcc', 'spearman'] else 'pearson'

    if len(model_names) < 2:
        raise ValueError("model_names must contain at least two models to correlate.")

    resolved_columns = {model_key: _resolve_model_column(model_key) for model_key in model_names}
    all_model_keys = list(model_names.keys())
    cols_to_extract = [resolved_columns[k] for k in all_model_keys]
    
    def _prep_dataset(df: pd.DataFrame) -> np.ndarray:
        df_models = pd.DataFrame(index=df.index)
        for col in cols_to_extract:
            if col in df.columns:
                df_models[col] = df[col].apply(_extract_scalar)
            else:
                df_models[col] = np.nan

        for col in cols_to_extract:
            df_models[col] = pd.to_numeric(df_models[col], errors='coerce')
            df_models.loc[np.isinf(df_models[col]), col] = np.nan

        if log_scale:
            for col in cols_to_extract:
                df_models.loc[df_models[col] <= 0, col] = np.nan
            df_models = np.log10(df_models)

        return df_models.corr(method=pandas_corr_method).to_numpy()

    raw_matrix_1 = _prep_dataset(dataset_1)
    raw_matrix_2 = _prep_dataset(dataset_2)

    n_models = len(all_model_keys)
    combined_matrix = np.full((n_models, n_models), np.nan)
    
    for i in range(n_models):
        for j in range(n_models):
            orig_j = n_models - 1 - j
            if i + j < n_models - 1:
                combined_matrix[i, j] = raw_matrix_1[i, orig_j]
            elif i + j > n_models - 1:
                combined_matrix[i, j] = raw_matrix_2[i, orig_j]

    ordered_display_names = [model_names[k] for k in all_model_keys]
    reversed_display_names = ordered_display_names[::-1]

    fig_size = max(6, 0.6 * n_models + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.set_style("ticks")

    base_mask = np.isnan(combined_matrix)

    if y_axis_right:
        sns.despine(left=True, right=False, top=True, bottom=False)
        ax.yaxis.tick_right()                  
        ax.yaxis.set_label_position("right")
        ax.tick_params(
            axis='y',        
            right=True,     
            left=False
        )
    else:
        sns.despine()

    from matplotlib.colors import LinearSegmentedColormap

    if cmap_colors and len(cmap_colors) == 2:
        grid_i, grid_j = np.indices((n_models, n_models))
        is_ds1 = (grid_i + grid_j) < (n_models - 1)
        is_ds2 = (grid_i + grid_j) > (n_models - 1)

        cmap1 = LinearSegmentedColormap.from_list("ds1_white_color", ["#FFFFFF", cmap_colors[0]])
        cmap1.set_bad("gray")
        mask1 = base_mask | ~is_ds1
        sns.heatmap(
            combined_matrix,
            ax=ax,
            cmap=cmap1,
            mask=mask1,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            linewidths=0.5,
            linecolor='white'
        )

        cmap2 = LinearSegmentedColormap.from_list("ds2_white_color", ["#FFFFFF", cmap_colors[1]])
        cmap2.set_bad(color='none') 
        mask2 = base_mask | ~is_ds2
        sns.heatmap(
            combined_matrix,
            ax=ax,
            cmap=cmap2,
            mask=mask2,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            linewidths=0.5,
            linecolor='white'
        )
    else:
        if cmap_colors and len(cmap_colors) == 3:
            cmap = LinearSegmentedColormap.from_list("blue_white_red", cmap_colors)
        else:
            cmap = sns.color_palette("rocket", as_cmap=True)
        cmap.set_bad("gray")
        
        sns.heatmap(
            combined_matrix,
            ax=ax,
            cmap=cmap,
            mask=base_mask,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            linewidths=0.5,
            linecolor='white'
        )

    ax.plot([0, n_models], [n_models, 0], color='white', linewidth=1, zorder=5)

    xtick_objs = ax.set_xticklabels(reversed_display_names, rotation=45, ha='right', fontsize=14)
    ytick_objs = ax.set_yticklabels(ordered_display_names, rotation=0, fontsize=14)

    if model_colors:
        for tick_label, model_key in zip(ytick_objs, all_model_keys):
            color = model_colors.get(model_key)
            if color:
                tick_label.set_color(color)
                
        for tick_label, model_key in zip(xtick_objs, all_model_keys[::-1]):
            color = model_colors.get(model_key)
            if color:
                tick_label.set_color(color)

    for row_idx in range(n_models):
        for col_idx in range(n_models):
            raw_value = combined_matrix[row_idx, col_idx]
            if not np.isfinite(raw_value):
                continue
                
            if cmap_colors:
                text_color = 'black'
            else:
                text_color = 'white' if abs(raw_value) < 0.5 else 'black'
                
            formatted_val = f"{raw_value:.2f}" 
            
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                formatted_val,
                ha='center',
                va='center',
                fontsize=12,
                color=text_color
            )

    ax.text(
        0.5,
        1.02,
        dataset_name_1,
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=14,
        fontweight='bold'
    )

    x_offset_ds2 = 1.25 if y_axis_right else 1.01
    ax.text(
        x_offset_ds2,
        0.5,
        dataset_name_2,
        transform=ax.transAxes,
        ha='left',
        va='center',
        rotation=-90,
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    if save:
        metric_space_label = "log10" if log_scale else "linear"
        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / "plots" / "model_heatmap_plots",
            f"dual_correlation_{metric}_{metric_space_label}_{dataset_name_1}_vs_{dataset_name_2}.png"
        )
        if save_target is not None:
            plt.savefig(
                str(save_target),
                dpi=300,
                bbox_inches='tight',
                transparent=False
            )

    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_r2_comparison_across_datasets(
    datasets: list[pd.DataFrame],
    dataset_names: list[str],
    model_names: dict[str, str],
    log_scale: bool = True,
    clamp_r2_to_unit_interval: bool = False,
    plot_type: str = 'dot',
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Plot model-wise R² values across multiple datasets.

    The function computes R² between each model prediction column and
    `experimental_kcat` for every dataset, then renders a grouped dot plot
    with models on the x-axis and R² on the y-axis.

    Parameters
    ----------
    datasets : list[pd.DataFrame]
        List of input datasets. Each dataframe must include
        `experimental_kcat` and model prediction columns with pattern
        `{model_name}_kcat`.
    dataset_names : list[str]
        Display names for datasets shown in legend. Must match `datasets` length.
    model_names : dict[str, str]
        Mapping from model identifier to display name. Model identifiers are
        resolved to columns using `{model_name}_kcat`, unless they already end
        with `_kcat`.
    log_scale : bool, default True
        If True, computes R² in log10 space after filtering positive values.
        If False, computes R² on raw values.
    clamp_r2_to_unit_interval : bool, default False
        If True, clips plotted R² values to [0, 1].
        If False, plots true R² values (including negatives).
    plot_type : str, default 'dot'
        Plot style for R² values. Must be one of: 'dot', 'bar'.
        In bar mode, values are rendered as thin grouped bars per model.
    save : bool, default False
        If True, saves figure under results/plots/r2_comparison_plots.
    save_path : str or Path, optional
        If provided, overrides the save directory or full file path. If a
        directory is provided, the default filename is used. If a filename
        is provided, it is used directly. Ignored when save is False.
    show : bool, default True
        If True, displays the figure; otherwise closes it.

    Returns
    -------
    None
        The function renders the plot and optionally saves it.

    Raises
    ------
    ValueError
        If experimental_kcat is missing, overlap is insufficient for any
        model-dataset pair, or input arguments are invalid. Missing model
        columns are allowed and are skipped.
    """
    if len(datasets) == 0:
        raise ValueError("datasets must contain at least one dataframe.")

    if len(dataset_names) != len(datasets):
        raise ValueError(
            "dataset_names must have the same length as datasets. "
            f"Received {len(dataset_names)} names for {len(datasets)} datasets."
        )

    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("dataset_names must be unique for an unambiguous legend.")

    if len(model_names) == 0:
        raise ValueError("model_names must contain at least one model mapping.")

    if not isinstance(plot_type, str):
        raise ValueError(
            f"plot_type must be a string ('dot' or 'bar'), got {type(plot_type).__name__}."
        )
    plot_type = plot_type.strip().lower()
    if plot_type not in {'dot', 'bar'}:
        raise ValueError(
            f"Unsupported plot_type '{plot_type}'. Expected one of: 'dot', 'bar'."
        )

    for dataset_idx, dataset in enumerate(datasets):
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError(
                "All entries in datasets must be pandas DataFrames. "
                f"Entry at index {dataset_idx} is {type(dataset).__name__}."
            )

    resolved_columns = {model_key: _resolve_model_column(model_key) for model_key in model_names}
    first_dataset_name = dataset_names[0]
    r2_by_dataset: dict[str, dict[str, float]] = {}

    for dataset_name, dataset in zip(dataset_names, datasets):
        if 'experimental_kcat' not in dataset.columns:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing required column 'experimental_kcat'."
            )

        r2_by_dataset[dataset_name] = {}

        for model_key, model_col in resolved_columns.items():
            if model_col not in dataset.columns:
                logger.warning(
                    "Dataset '%s' is missing model column '%s'. Skipping.",
                    dataset_name,
                    model_col
                )
                r2_by_dataset[dataset_name][model_key] = np.nan
                continue

            pair_df = dataset[['experimental_kcat', model_col]].copy()
            pair_df['experimental_kcat'] = pair_df['experimental_kcat'].apply(_extract_scalar)
            pair_df[model_col] = pair_df[model_col].apply(_extract_scalar)

            pair_df['experimental_kcat'] = pd.to_numeric(pair_df['experimental_kcat'], errors='coerce')
            pair_df[model_col] = pd.to_numeric(pair_df[model_col], errors='coerce')
            pair_df = pair_df.dropna()
            pair_df = pair_df[~pair_df.isin([np.inf, -np.inf]).any(axis=1)]

            if log_scale:
                pair_df = pair_df[
                    (pair_df['experimental_kcat'] > 0)
                    & (pair_df[model_col] > 0)
                ]

            if len(pair_df) < 2:
                raise ValueError(
                    "Insufficient overlapping data for R² computation: "
                    f"dataset='{dataset_name}', model='{model_col}', "
                    f"valid_points={len(pair_df)}. At least 2 points are required."
                )

            y_true = pair_df['experimental_kcat'].to_numpy(dtype=float)
            y_pred = pair_df[model_col].to_numpy(dtype=float)
            if log_scale:
                y_true = np.log10(y_true)
                y_pred = np.log10(y_pred)

            r2_value = float(r2_score(y_true, y_pred))
            r2_by_dataset[dataset_name][model_key] = r2_value

    r2_first_dataset = r2_by_dataset[first_dataset_name]
    ordered_model_keys = sorted(
        model_names.keys(),
        key=lambda model_key: (
            not np.isfinite(r2_first_dataset.get(model_key, np.nan)),
            -r2_first_dataset.get(model_key, np.nan)
            if np.isfinite(r2_first_dataset.get(model_key, np.nan))
            else 0.0,
            model_names[model_key]
        )
    )

    n_models = len(ordered_model_keys)
    n_datasets = len(dataset_names)
    fig_width = max(10, 1.3 * n_models + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    sns.set_style("ticks")

    inter_group_spacing = 0.15
    if plot_type == 'bar':
        intra_group_total_width = min(0.2, 0.02 * max(n_datasets, 1))
    else:
        intra_group_total_width = min(0.2, 0.03 * max(n_datasets - 1, 1))

    if n_datasets == 1:
        dataset_offsets = np.array([0.0])
    else:
        dataset_offsets = np.linspace(
            -intra_group_total_width / 2,
            intra_group_total_width / 2,
            n_datasets
        )
    model_centers = np.arange(n_models, dtype=float) * inter_group_spacing

    palette = sns.color_palette("deep", n_colors=n_datasets)

    all_r2_values = []
    for dataset_idx, dataset_name in enumerate(dataset_names):
        x_positions = model_centers + dataset_offsets[dataset_idx]
        y_values = np.array(
            [r2_by_dataset[dataset_name][model_key] for model_key in ordered_model_keys],
            dtype=float
        )
        if clamp_r2_to_unit_interval:
            y_values = np.clip(y_values, 0.0, 1.0)

        finite_mask = np.isfinite(y_values)
        if not np.any(finite_mask):
            continue

        all_r2_values.extend(y_values[finite_mask].tolist())
        if plot_type == 'dot':
            ax.scatter(
                x_positions[finite_mask],
                y_values[finite_mask],
                s=65,
                color=palette[dataset_idx],
                edgecolor='black',
                linewidth=0.3,
                label=dataset_name,
                zorder=3
            )
        else:
            bar_width = min(
                0.15,
                max(0.03, (intra_group_total_width / max(n_datasets, 1)) * 0.9)
            )
            ax.bar(
                x_positions[finite_mask],
                y_values[finite_mask],
                width=bar_width,
                bottom=0.0,
                color=palette[dataset_idx],
                edgecolor='black',
                linewidth=0.3,
                label=dataset_name,
                zorder=3
            )

    if len(all_r2_values) == 0:
        raise ValueError("No R² values were computed for plotting.")

    y_min = min(all_r2_values)
    y_max = max(all_r2_values)

    if plot_type == 'bar':
        has_negative_r2 = y_min < 0.0
        if has_negative_r2:
            y_span = y_max - y_min
            y_pad = 0.05 if np.isclose(y_span, 0.0) else y_span * 0.02
            lower_limit = y_min - y_pad
            upper_limit = max(y_max + y_pad, 0.05)
        else:
            y_pad = 0.05 if np.isclose(y_max, 0.0) else abs(y_max) * 0.02
            lower_limit = 0.0
            upper_limit = y_max + y_pad

        if np.isclose(lower_limit, upper_limit):
            upper_limit = lower_limit + 0.1
        ax.set_ylim(lower_limit, upper_limit)

        if has_negative_r2:
            ax.axhline(0.0, linestyle='--', color='gray', linewidth=1.0, alpha=0.8, zorder=1)
    else:
        if np.isclose(y_min, y_max):
            y_pad = 0.05 if np.isclose(y_max, 0.0) else abs(y_max) * 0.05
        else:
            y_pad = (y_max - y_min) * 0.02
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.axhline(1.0, linestyle='--', color='gray', linewidth=1.0, alpha=0.7, zorder=1)
    ax.set_xticks(model_centers)
    ax.set_xticklabels([model_names[model_key] for model_key in ordered_model_keys], rotation=0, ha='center')
    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel(r"$R^2$", fontsize=13)

    metric_space_label = "log10" if log_scale else "linear"
    title_text = f"Model R² across datasets ({metric_space_label} space)"
    ax.set_title(title_text, fontsize=16, fontweight='bold')

    legend = ax.legend(
        title="Dataset",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=11,
        title_fontsize=11
    )
    legend.get_frame().set_edgecolor('0.8')

    sns.despine()
    plt.tight_layout()

    if save:
        file_suffix = "" if plot_type == 'dot' else f"_{plot_type}"
        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / "plots" / "r2_comparison_plots",
            f"r2_model_dataset_comparison{file_suffix}.png"
        )
        if save_target is not None:
            plt.savefig(
                str(save_target),
                dpi=300,
                bbox_inches='tight',
                transparent=False
            )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_metric_heatmap_across_datasets(
    datasets: list[pd.DataFrame],
    dataset_names: list[str],
    model_names: dict[str, str],
    log_scale: bool = True,
    y_axis_right: bool = False,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    show_dataset_labels: bool = True,
    cmap_colors: Optional[list[str]] = None,
    model_colors: Optional[dict[str, str]] = None
) -> None:
    """
    Plot a metric heatmap across datasets and models with customized styling.

    The function computes R2, Pearson r, SRCC, RMSE, and MAE between each
    model prediction column and experimental_kcat for every dataset, then
    renders a heatmap with metric columns grouped by dataset. Missing model
    columns or NaN metrics are shown as black cells with no annotation.

    Parameters
    ----------
    datasets : list[pd.DataFrame]
        List of input datasets. Each dataframe must include
        experimental_kcat and model prediction columns with pattern
        {model_name}_kcat.
    dataset_names : list[str]
        Display names for datasets shown above metric groups. Must match
        datasets length.
    model_names : dict[str, str]
        Mapping from model identifier to display name. Model identifiers are
        resolved to columns using {model_name}_kcat, unless they already end
        with _kcat.
    log_scale : bool, default True
        If True, computes metrics in log10 space after filtering positive values.
        If False, computes metrics on raw values.
    y_axis_right : bool, default False
        If True, moves the model labels to the right side of the plot.
    save : bool, default False
        If True, saves figure under results/plots/metric_heatmap_plots.
    save_path : str or Path, optional
        If provided, overrides the save directory or full file path.
    show : bool, default True
        If True, displays the figure; otherwise closes it.
    show_dataset_labels : bool, default True
        If True, renders the dataset names above their respective column groupings.
    cmap_colors : list[str], optional
        A list containing exactly two color strings (e.g., ['blue', 'red']) to 
        generate a custom linear gradient. If None, defaults to 'rocket'.
    model_colors : dict[str, str], optional
        Mapping from model identifier to a color string. If provided, the y-axis
        label for that model will be colored accordingly.

    Returns
    -------
    None
        The function renders the plot and optionally saves it.

    Raises
    ------
    ValueError
        If experimental_kcat is missing or input arguments are invalid.
    """
    if len(datasets) == 0:
        raise ValueError("datasets must contain at least one dataframe.")

    if len(dataset_names) != len(datasets):
        raise ValueError(
            "dataset_names must have the same length as datasets. "
            f"Received {len(dataset_names)} names for {len(datasets)} datasets."
        )

    if len(set(dataset_names)) != len(dataset_names):
        raise ValueError("dataset_names must be unique for an unambiguous legend.")

    if len(model_names) == 0:
        raise ValueError("model_names must contain at least one model mapping.")

    for dataset_idx, dataset in enumerate(datasets):
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError(
                "All entries in datasets must be pandas DataFrames. "
                f"Entry at index {dataset_idx} is {type(dataset).__name__}."
            )

    resolved_columns = {model_key: _resolve_model_column(model_key) for model_key in model_names}
    metrics_by_dataset: dict[str, dict[str, dict[str, float]]] = {}

    for dataset_name, dataset in zip(dataset_names, datasets):
        if 'experimental_kcat' not in dataset.columns:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing required column 'experimental_kcat'."
            )

        metrics_by_dataset[dataset_name] = {}

        for model_key, model_col in resolved_columns.items():
            if model_col not in dataset.columns:
                logger.warning(
                    "Dataset '%s' is missing model column '%s'. Skipping.",
                    dataset_name,
                    model_col
                )
                metrics_by_dataset[dataset_name][model_key] = _empty_metrics()
                continue

            pair_df = dataset[['experimental_kcat', model_col]].copy()
            pair_df['experimental_kcat'] = pair_df['experimental_kcat'].apply(_extract_scalar)
            pair_df[model_col] = pair_df[model_col].apply(_extract_scalar)

            pair_df['experimental_kcat'] = pd.to_numeric(pair_df['experimental_kcat'], errors='coerce')
            pair_df[model_col] = pd.to_numeric(pair_df[model_col], errors='coerce')
            pair_df = pair_df.dropna()
            pair_df = pair_df[~pair_df.isin([np.inf, -np.inf]).any(axis=1)]

            if log_scale:
                pair_df = pair_df[
                    (pair_df['experimental_kcat'] > 0)
                    & (pair_df[model_col] > 0)
                ]

            if len(pair_df) == 0:
                logger.warning(
                    "No valid overlap for metrics: dataset='%s', model='%s'.",
                    dataset_name,
                    model_col
                )
                metrics_by_dataset[dataset_name][model_key] = _empty_metrics()
                continue

            if len(pair_df) < 2:
                logger.warning(
                    "Low sample count for metrics: dataset='%s', model='%s', valid_points=%d.",
                    dataset_name,
                    model_col,
                    len(pair_df)
                )

            y_true = pair_df['experimental_kcat'].to_numpy(dtype=float)
            y_pred = pair_df[model_col].to_numpy(dtype=float)
            if log_scale:
                y_true = np.log10(y_true)
                y_pred = np.log10(y_pred)

            metrics_by_dataset[dataset_name][model_key] = _compute_comparison_metrics(
                y_pred,
                y_true,
                include_ground_truth_metrics=True
            )

    first_dataset_name = dataset_names[0]
    ordered_model_keys = sorted(
        model_names.keys(),
        key=lambda model_key: (
            not np.isfinite(metrics_by_dataset[first_dataset_name][model_key].get('r2', np.nan)),
            -metrics_by_dataset[first_dataset_name][model_key].get('r2', np.nan)
            if np.isfinite(metrics_by_dataset[first_dataset_name][model_key].get('r2', np.nan))
            else 0.0,
            model_names[model_key]
        )
    )

    metric_keys = ['r2', 'pearson_r', 'srcc', 'rmse', 'mae']
    metric_labels = [r"$R^2$", "Pearson r", "SRCC", "RMSE", "MAE"]
    n_metrics = len(metric_keys)
    n_datasets = len(dataset_names)
    n_models = len(ordered_model_keys)
    n_cols = n_metrics * n_datasets

    raw_matrix = np.full((n_models, n_cols), np.nan, dtype=float)

    for dataset_idx, dataset_name in enumerate(dataset_names):
        for model_idx, model_key in enumerate(ordered_model_keys):
            metrics = metrics_by_dataset[dataset_name][model_key]
            for metric_idx, metric_key in enumerate(metric_keys):
                col_idx = dataset_idx * n_metrics + metric_idx
                raw_matrix[model_idx, col_idx] = metrics.get(metric_key, np.nan)

    norm_matrix = np.full_like(raw_matrix, np.nan, dtype=float)
    invert_metrics = {'rmse', 'mae'}
    for metric_idx, metric_key in enumerate(metric_keys):
        values = raw_matrix[:, metric_idx::n_metrics]
        finite_vals = values[np.isfinite(values)]

        if finite_vals.size == 0:
            normalized = np.full(values.shape, 0.5, dtype=float)
        else:
            min_val = np.min(finite_vals)
            max_val = np.max(finite_vals)
            if np.isclose(min_val, max_val):
                normalized = np.full(values.shape, 0.5, dtype=float)
            else:
                normalized = (values - min_val) / (max_val - min_val)
            if metric_key in invert_metrics:
                normalized = 1.0 - normalized

        normalized = np.where(np.isfinite(values), normalized, np.nan)
        norm_matrix[:, metric_idx::n_metrics] = normalized

    best_rows_per_col = {}
    for col_idx in range(n_cols):
        metric_key = metric_keys[col_idx % n_metrics]
        col_vals = raw_matrix[:, col_idx]
        valid_mask = np.isfinite(col_vals)
        if np.any(valid_mask):
            if metric_key in invert_metrics:
                best_val = np.nanmin(col_vals[valid_mask])
            else:
                best_val = np.nanmax(col_vals[valid_mask])
            best_rows_per_col[col_idx] = np.where(np.isclose(col_vals, best_val))[0]
        else:
            best_rows_per_col[col_idx] = []

    fig_width = max(6, 0.6 * n_cols + 2)
    fig_height = max(5, 0.35 * n_models + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.set_style("ticks")

    if cmap_colors and len(cmap_colors) == 2:
        from matplotlib.colors import LinearSegmentedColormap
        color_list = [cmap_colors[0], '#FFFFFF', cmap_colors[1]]
        cmap = LinearSegmentedColormap.from_list("blue_white_red", color_list)
    else:
        cmap = sns.color_palette("rocket", as_cmap=True)
    cmap.set_bad("gray")
    
    mask = np.isnan(norm_matrix)
    sns.heatmap(
        norm_matrix,
        ax=ax,
        cmap=cmap,
        mask=mask,
        cbar=False,
        linewidths=0.5,
        linecolor='white'
    )

    if y_axis_right:
        sns.despine(left=True, right=False, top=True, bottom=False)
        ax.yaxis.tick_right()                  
        ax.yaxis.set_label_position("right")
        ax.tick_params(
            axis='y',        
            right=True,     
            left=False
        )
    else:
        sns.despine()

    xtick_labels = []
    for _ in dataset_names:
        xtick_labels.extend(metric_labels)

    ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=14)
    
    ytick_objs = ax.set_yticklabels(
        [model_names[model_key] for model_key in ordered_model_keys],
        rotation=0,
        fontsize=14
    )

    if model_colors:
        for tick_label, model_key in zip(ytick_objs, ordered_model_keys):
            color = model_colors.get(model_key)
            if color:
                tick_label.set_color(color)

    for boundary in range(1, n_datasets):
        ax.axvline(boundary * n_metrics, color='black', linewidth=1.5)

    if show_dataset_labels:
        for dataset_idx, dataset_name in enumerate(dataset_names):
            center = (dataset_idx * n_metrics + n_metrics / 2) / n_cols
            ax.text(
                center,
                1.02,
                dataset_name,
                transform=ax.transAxes,
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold'
            )

    for row_idx in range(n_models):
        for col_idx in range(n_cols):
            raw_value = raw_matrix[row_idx, col_idx]
            if not np.isfinite(raw_value):
                continue

            if cmap_colors and len(cmap_colors) == 2:
                text_color = 'black'
            else: 
                color_value = norm_matrix[row_idx, col_idx]
                text_color = 'white' if np.isfinite(color_value) and color_value < 0.5 else 'black'
            
            is_best = row_idx in best_rows_per_col.get(col_idx, [])
            font_weight = 'bold' if is_best else 'normal'
            
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                _format_metric(raw_value),
                ha='center',
                va='center',
                fontsize=12,
                color=text_color,
                fontweight=font_weight
            )

    metric_space_label = "log10" if log_scale else "linear"

    plt.tight_layout()
    fig.subplots_adjust(top=0.86)

    if save:
        dataset_names_str = "_".join(dataset_names)
        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / "plots" / "metric_heatmap_plots",
            f"metric_heatmap_{metric_space_label}_{dataset_names_str}.png"
        )
        if save_target is not None:
            plt.savefig(
                str(save_target),
                dpi=300,
                bbox_inches='tight',
                transparent=False
            )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_metric_vs_delta_growth(
    dataset: pd.DataFrame,
    model_names: dict[str, str],
    delta_growth: dict[str, float],
    log_scale: bool = True,
    show_fit_r2: bool = True,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    model_colors: Optional[dict[str, str]] = None
) -> None:
    """
    Plot the relationship between prediction metrics and delta growth across models.

    This function evaluates model performance metrics (R2, Pearson r, SRCC, RMSE, MAE) 
    against a given delta growth parameter. It generates a multi-panel scatter plot matrix 
    showing the correlation between each metric and the growth differential, along 
    with a linear fit and optional R2 of the trendline.

    Parameters
    ----------
    dataset : pd.DataFrame
        The input dataset containing experimental ground truth and model predictions.
    model_names : dict[str, str]
        Mapping from model identifier keys to their human-readable display names.
    delta_growth : dict[str, float]
        Dictionary mapping model identifiers or columns to their corresponding delta 
        growth values.
    log_scale : bool, default True
        If True, transforms values to log10 space before calculating metrics.
    show_fit_r2 : bool, default True
        If True, computes and displays the R2 coefficient of the linear regression line 
        within each subplot panel.
    save : bool, default False
        If True, saves the generated figure to disk.
    save_path : str or Path, optional
        Custom directory path or complete file path string to override default save tracking.
    show : bool, default True
        If True, displays the plot interactively using plt.show().
    model_colors : dict[str, str], optional
        Mapping from model identifier keys to valid color strings. If provided, 
        dictates the color of scatter points and legend representation for each model.
        Falls back to the 'husl' palette if a key is missing or if None.

    Returns
    -------
    None
        Renders or saves the plot configuration directly.

    Raises
    ------
    ValueError
        If the dataset is not a pandas DataFrame, missing the required 'experimental_kcat' 
        column, or if model_names is empty.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("dataset must be a pandas DataFrame.")
        
    if 'experimental_kcat' not in dataset.columns:
        raise ValueError("Dataset is missing required column 'experimental_kcat'.")

    if len(model_names) == 0:
        raise ValueError("model_names must contain at least one model mapping.")

    resolved_columns = {model_key: _resolve_model_column(model_key) for model_key in model_names}
    
    model_metrics = {}
    
    for model_key, model_col in resolved_columns.items():
        if model_col not in dataset.columns:
            logger.warning("Dataset is missing model column '%s'. Skipping.", model_col)
            continue

        pair_df = dataset[['experimental_kcat', model_col]].copy()
        pair_df['experimental_kcat'] = pair_df['experimental_kcat'].apply(_extract_scalar)
        pair_df[model_col] = pair_df[model_col].apply(_extract_scalar)

        pair_df['experimental_kcat'] = pd.to_numeric(pair_df['experimental_kcat'], errors='coerce')
        pair_df[model_col] = pd.to_numeric(pair_df[model_col], errors='coerce')
        pair_df = pair_df.dropna()
        pair_df = pair_df[~pair_df.isin([np.inf, -np.inf]).any(axis=1)]

        if log_scale:
            pair_df = pair_df[
                (pair_df['experimental_kcat'] > 0)
                & (pair_df[model_col] > 0)
            ]

        if len(pair_df) < 2:
            logger.warning("Not enough valid points for metrics: model='%s'.", model_col)
            continue

        y_true = pair_df['experimental_kcat'].to_numpy(dtype=float)
        y_pred = pair_df[model_col].to_numpy(dtype=float)
        
        if log_scale:
            y_true = np.log10(y_true)
            y_pred = np.log10(y_pred)

        model_metrics[model_key] = _compute_comparison_metrics(
            y_pred,
            y_true,
            include_ground_truth_metrics=True
        )

    metric_keys = ['r2', 'pearson_r', 'srcc', 'rmse', 'mae']
    metric_labels = [r"$R^2$", "Pearson r", "SRCC", "RMSE", "MAE"]
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6.5), sharey=True)
    axes = axes.flatten()
    sns.set_style("ticks")
    
    fallback_colors = sns.color_palette("husl", n_colors=len(model_names))
    color_map = {}
    for idx, key in enumerate(model_names.keys()):
        if model_colors and key in model_colors:
            color_map[key] = model_colors[key]
        else:
            color_map[key] = fallback_colors[idx]
            
    corr_min = float('inf')
    corr_max = float('-inf')
    
    for m_key in ['r2', 'pearson_r', 'srcc']:
        for model_key in model_metrics:
            val = model_metrics[model_key].get(m_key, np.nan)
            if np.isfinite(val):
                corr_min = min(corr_min, val)
                corr_max = max(corr_max, val)
                
    if corr_min != float('inf') and corr_max != float('-inf'):
        padding = (corr_max - corr_min) * 0.05
        corr_xlim = (corr_min - padding, corr_max + padding)
    else:
        corr_xlim = None

    err_min = float('inf')
    err_max = float('-inf')
    
    for m_key in ['rmse', 'mae']:
        for model_key in model_metrics:
            val = model_metrics[model_key].get(m_key, np.nan)
            if np.isfinite(val):
                err_min = min(err_min, val)
                err_max = max(err_max, val)
                
    if err_min != float('inf') and err_max != float('-inf'):
        padding = (err_max - err_min) * 0.05
        if padding == 0:
            padding = 0.1
        err_xlim = (err_min - padding, err_max + padding)
    else:
        err_xlim = None
    
    for i, (m_key, m_label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[i]
        ax.set_box_aspect(1)
        x_data = []
        y_data = []
        
        for model_key, model_col in resolved_columns.items():
            if model_key not in model_metrics:
                continue
                
            d_growth = delta_growth.get(model_col, delta_growth.get(model_key))
            metric_val = model_metrics[model_key].get(m_key, np.nan)
            
            if d_growth is not None and np.isfinite(metric_val) and np.isfinite(d_growth):
                x_data.append(metric_val)
                y_data.append(d_growth)
                
                ax.scatter(
                    metric_val, 
                    d_growth, 
                    color=color_map[model_key], 
                    s=100, 
                    zorder=3,
                    edgecolor='white',
                    linewidth=0.5
                )
                
        if len(x_data) >= 2:
            x_arr = np.array(x_data)
            y_arr = np.array(y_data)
            m, b = np.polyfit(x_arr, y_arr, 1)
            
            if m_key in ['r2', 'pearson_r', 'srcc'] and corr_xlim is not None:
                x_line = np.array(corr_xlim)
            elif m_key in ['rmse', 'mae'] and err_xlim is not None:
                x_line = np.array(err_xlim)
            else:
                x_line = np.array([np.min(x_arr), np.max(x_arr)])
                
            y_line = m * x_line + b
            ax.plot(x_line, y_line, color='black', linewidth=1.5, linestyle='-', zorder=2)
            
            if show_fit_r2:
                y_pred_fit = m * x_arr + b
                ss_res = np.sum((y_arr - y_pred_fit) ** 2)
                ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
                r2_fit = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
                
                if np.isfinite(r2_fit):
                    x_pos = 0.05 if m >= 0 else 0.95
                    h_align = 'left' if m >= 0 else 'right'
                    
                    ax.text(
                        x_pos, 0.95,
                        f"Fit $R^2$: {r2_fit:.2f}",
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        horizontalalignment=h_align,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
                        zorder=4
                    )
            
        ax.set_xlabel(m_label, fontsize=11)
        
        if m_key in ['r2', 'pearson_r', 'srcc'] and corr_xlim is not None:
            ax.set_xlim(corr_xlim)
        elif m_key in ['rmse', 'mae'] and err_xlim is not None:
            ax.set_xlim(err_xlim)
        
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        
        if i % 3 != 0:
            ax.tick_params(left=False, labelleft=False)

    fig.supylabel(r"$\Delta$ Growth", fontsize=14)

    legend_ax = axes[5]
    legend_ax.axis('off')
    
    handles = [
        plt.Line2D(
            [0], [0], 
            marker='o', 
            color='w', 
            markerfacecolor=color_map[key], 
            markersize=10, 
            label=name
        ) 
        for key, name in model_names.items()
    ]
    
    legend_ax.legend(
        handles=handles, 
        loc='center', 
        title="Model", 
        title_fontsize=13,
        fontsize=11,
        frameon=False,
        ncol=1
    )

    sns.despine()
    plt.tight_layout()

    if save:
        metric_space_label = "log10" if log_scale else "linear"
        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / "plots" / "delta_growth_scatter_plots",
            f"delta_growth_vs_metrics_{metric_space_label}.png"
        )
        if save_target is not None:
            plt.savefig(
                str(save_target),
                dpi=300,
                bbox_inches='tight',
                transparent=False
            )

    if show:
        plt.show()
    else:
        plt.close(fig)




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

    eigenvalues = np.linalg.eigvals(cov)
    
    major_axis_length = np.sqrt(np.max(eigenvalues))
    minor_axis_length = np.sqrt(np.min(eigenvalues))
    axis_ratio = minor_axis_length / major_axis_length

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    ellipse_x_bounds = [mean_x - scale_x, mean_x + scale_x]
    ellipse_y_bounds = [mean_y - scale_y, mean_y + scale_y]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    return axis_ratio, ellipse_x_bounds, ellipse_y_bounds



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
    save_path: Optional[Union[str, Path]] = None,
    show=True,
    model_vs_model: bool = False,
    show_stats: bool = True,
    show_ellipse_stats: bool = False,
    show_c_bar: bool = True,
    ellipse_stats_position: str = 'upper_right',
    show_title: bool = True
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
    save_path : str or Path, optional
        If provided, overrides the save directory or full file path. If a
        directory is provided, the default filename is used. If a filename
        is provided, it is used directly. Ignored when save is False.
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

    axis_ratio, ellipse_x_bounds, ellipse_y_bounds = confidence_ellipse(plot_x, plot_y, ax, n_std=n_std, edgecolor='red', linestyle='--', linewidth=1)
    
    if show_c_bar:
        cb = plt.colorbar(
            hb, 
            label='Count', 
            shrink=0.5,     
            aspect=20,      
            pad=0.05      
        )

        cb.ax.minorticks_off()
    
    plt.plot([lower_limit, upper_limit], [lower_limit, upper_limit], 
             color='black',     
             linestyle='--',      
             alpha=0.6,        
             linewidth=1.0, 
             label='Perfect Agreement')
    
    median_x = np.mean(plot_x)
    median_y = np.mean(plot_y)

    ax.scatter(
        median_x, median_y, 
        marker='X',         
        s=150,           
        facecolor='white', 
        edgecolor='black',   
        linewidth=1.5,
        zorder=5,       
        label='Median'
    )
    if show_stats:
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=13,
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
            fontsize=16,
            horizontalalignment=ha,
            verticalalignment=va,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
        )

    if log_scale:
        plt.xlabel(f"{model_x_name} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=16)
        plt.ylabel(f"{model_y_name} $k_{{cat}}$ ($s^{{-1}}$) [log scale]", fontsize=16)

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

    ax.tick_params(which='major', labelsize=14, length=6, width=2)

    if show_title:
        plt.title(f"{model_x_name} vs {model_y_name}", fontsize=16, fontweight='bold')
    
    plt.tight_layout()

    if save:
        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / "plots" / "comparison_plots",
            f"{model_x_name}_vs_{model_y_name}.png"
        )
        if save_target is not None:
            plt.savefig(
                str(save_target),
                dpi=300,
                bbox_inches='tight',
                transparent=False
            )
    if show:
        plt.show()
    else:
        plt.close(fig)

    return axis_ratio, ellipse_x_bounds, ellipse_y_bounds



def plot_log10_error_distribution(
    df: pd.DataFrame,
    model_names: dict[str, str],
    model_colors: Optional[dict[str, str]] = None,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    indicators: Optional[list[float]] = None,
    absolute_errors: bool = True
) -> None:
    """
    Plot the distribution of absolute log10 errors across models.

    This function calculates the absolute difference between the log10-transformed
    predicted and experimental values. By evaluating the absolute magnitude of 
    the log10 error, it visualizes the overall scale of model deviation (fold-change 
    distance from the true value) without distinguishing between overprediction 
    and underprediction.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing experimental ground truth and model predictions.
    model_names : dict[str, str]
        Mapping from model identifier keys to their human-readable display names.
    model_colors : dict[str, str], optional
        Mapping from model identifier keys to valid color strings. If provided, 
        dictates the color of the distribution curve for each model. Falls back 
        to the 'husl' palette if a key is missing or if None.
    save : bool, default False
        If True, saves the generated figure to disk.
    save_path : str or Path, optional
        Custom directory path or complete file path string to override default save tracking.
    show : bool, default True
        If True, displays the plot interactively using plt.show().
    indicators : list[float], optional
        A list of numeric values. For each value, a vertical dashed line is drawn 
        on the plot if the value falls within the calculated x-axis limits.

    Returns
    -------
    None
        Renders or saves the plot configuration directly.

    Raises
    ------
    ValueError
        If the dataset is missing the required 'experimental_kcat' column or if 
        model_names is empty.
    """
    if 'experimental_kcat' not in df.columns:
        raise ValueError("Dataset is missing required column 'experimental_kcat'.")
        
    if not model_names:
        raise ValueError("model_names must contain at least one model mapping.")

    resolved_columns = {model_key: _resolve_model_column(model_key) for model_key in model_names}
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("ticks")
    
    fallback_colors = sns.color_palette("husl", n_colors=len(model_names))
    
    for idx, (model_key, model_col) in enumerate(resolved_columns.items()):
        if model_col not in df.columns:
            continue
            
        pair_df = df[['experimental_kcat', model_col]].copy()
        
        pair_df['experimental_kcat'] = pair_df['experimental_kcat'].apply(_extract_scalar)
        pair_df[model_col] = pair_df[model_col].apply(_extract_scalar)
        
        pair_df['experimental_kcat'] = pd.to_numeric(pair_df['experimental_kcat'], errors='coerce')
        pair_df[model_col] = pd.to_numeric(pair_df[model_col], errors='coerce')
        
        pair_df = pair_df.dropna()
        pair_df = pair_df[~pair_df.isin([np.inf, -np.inf]).any(axis=1)]
        pair_df = pair_df[(pair_df['experimental_kcat'] > 0) & (pair_df[model_col] > 0)]
        
        if pair_df.empty:
            continue
            
        log10_true = np.log10(pair_df['experimental_kcat'].to_numpy(dtype=float))
        log10_pred = np.log10(pair_df[model_col].to_numpy(dtype=float))
        
        if absolute_errors:
            log10_error = np.abs(log10_pred - log10_true)
            clip = (0.0, None)
        else:
            log10_error = log10_pred - log10_true
            clip = (None, None)
        
        if model_colors and model_key in model_colors:
            curve_color = model_colors[model_key]
        else:
            curve_color = fallback_colors[idx]
            
        display_name = model_names.get(model_key, model_key)
        
        sns.kdeplot(
            x=log10_error,
            ax=ax,
            color=curve_color,
            label=display_name,
            linewidth=2,
            fill=True,
            alpha=0.1,
            clip=clip
        )

    if absolute_errors:
        ax.set_xlabel(r"Absolute $\log_{10}$ Error", fontsize=12)
    else:
        ax.set_xlabel(r"$\log_{10}$ Error", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    x_min, x_max = ax.get_xlim()
    if absolute_errors:
        ax.set_xlim(left=0.0)
    
    if indicators:
        for ind in indicators:
            if x_min <= ind <= x_max:
                ax.axvline(x=ind, color='black', linestyle='--', linewidth=1.5, zorder=1)

    ax.legend(title="Model", frameon=False, loc='best')
    sns.despine()
    plt.tight_layout()

    if save:
        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / "plots" / "error_distributions",
            "absolute_log10_error_distribution.png"
        )
        if save_target is not None:
            plt.savefig(str(save_target), dpi=300, bbox_inches='tight', transparent=False)

    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_log10_values_distribution(
    df: pd.DataFrame,
    model_names: dict[str, str],
    model_colors: Optional[dict[str, str]] = None,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    indicators: Optional[list[float]] = None
) -> None:
    """
    Plot and compare the log10 distributions of experimental values and model predictions.

    This function visualizes the overall profile of ground truth experimental kcat 
    values alongside the predicted values from each specified model. By transforming 
    the data into log10 space and rendering smooth Kernel Density Estimate (KDE) curves, 
    it provides a direct visual assessment of whether models accurately capture the 
    true spread, dynamic range, and modality of the experimental data.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing experimental ground truth and model predictions.
    model_names : dict[str, str]
        Mapping from model identifier keys to their human-readable display names.
    model_colors : dict[str, str], optional
        Mapping from model identifier keys to valid color strings. If provided, 
        dictates the color of the distribution curve for each model. Falls back 
        to the 'husl' palette if a key is missing or if None.
    save : bool, default False
        If True, saves the generated figure to disk.
    save_path : str or Path, optional
        Custom directory path or complete file path string to override default save tracking.
    show : bool, default True
        If True, displays the plot interactively using plt.show().
    indicators : list[float], optional
        A list of numeric values. For each value, a vertical dashed line is drawn 
        on the plot if the value falls within the calculated x-axis limits.

    Returns
    -------
    None
        Renders or saves the plot configuration directly.

    Raises
    ------
    ValueError
        If the dataset is missing the required 'experimental_kcat' column or if 
        model_names is empty.
    """
    if 'experimental_kcat' not in df.columns:
        raise ValueError("Dataset is missing required column 'experimental_kcat'.")
        
    if not model_names:
        raise ValueError("model_names must contain at least one model mapping.")

    resolved_columns = {model_key: _resolve_model_column(model_key) for model_key in model_names}
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("ticks")
    
    fallback_colors = sns.color_palette("husl", n_colors=len(model_names))
    
    exp_series = df['experimental_kcat'].apply(_extract_scalar)
    exp_series = pd.to_numeric(exp_series, errors='coerce').dropna()
    exp_series = exp_series[~exp_series.isin([np.inf, -np.inf])]
    exp_series = exp_series[exp_series > 0]
    
    if not exp_series.empty:
        log10_exp = np.log10(exp_series.to_numpy(dtype=float))
        sns.kdeplot(
            x=log10_exp,
            ax=ax,
            color="#7F7F7F",
            label='Experimental',
            linewidth=2.5,
            linestyle='-',
            fill=True,
            alpha=0.05,
            zorder=2
        )

    for idx, (model_key, model_col) in enumerate(resolved_columns.items()):
        if model_col not in df.columns:
            continue
            
        model_series = df[model_col].apply(_extract_scalar)
        model_series = pd.to_numeric(model_series, errors='coerce').dropna()
        model_series = model_series[~model_series.isin([np.inf, -np.inf])]
        model_series = model_series[model_series > 0]
        
        if model_series.empty:
            continue
            
        log10_pred = np.log10(model_series.to_numpy(dtype=float))
        
        if model_colors and model_key in model_colors:
            curve_color = model_colors[model_key]
        else:
            curve_color = fallback_colors[idx]
            
        display_name = model_names.get(model_key, model_key)
        
        sns.kdeplot(
            x=log10_pred,
            ax=ax,
            color=curve_color,
            label=display_name,
            linewidth=2,
            fill=True,
            alpha=0.1,
            zorder=3
        )

    ax.set_xlabel(r"$\log_{10}(k_{\text{cat}} \text{ s}^{-1})$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
   
    x_min, x_max = ax.get_xlim()
    
    if indicators:
        for ind in indicators:
            if x_min <= ind <= x_max:
                ax.axvline(x=ind, color='black', linestyle='--', linewidth=1.5, zorder=1)

    ax.legend(title="Data Source", frameon=False, loc='best')
    sns.despine()
    plt.tight_layout()

    if save:
        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / "plots" / "value_distributions",
            "log10_kcat_value_distribution.png"
        )
        if save_target is not None:
            plt.savefig(str(save_target), dpi=300, bbox_inches='tight', transparent=False)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_sequence_similarity_results(
    df: pd.DataFrame,
    model_names: dict[str, str],
    combined_model_keys: Optional[tuple[str, str]] = None,
    value_mode: str = 'count',
    gradient_colors: tuple[str, str] = ("#FFFFFF", "#990F0F"),
    model_colors: Optional[dict[str, str]] = None,
    bar_height: float = 0.42,
    enzy_bar_height_ratio: float = 0.8,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Plot sequence-similarity bin results as paired stacked horizontal bars.

    The input dataframe is expected to contain a leading `label` column that
    identifies the dataset and bin for each row. Rows whose label starts with
    `enzy_` are treated as EnzyExtract results; all other rows are treated as
    BRENDA results. Each model is plotted as a pair of horizontal stacked bars
    with BRENDA on top and EnzyExtract below it.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a `label` column and one or more model columns.
    model_names : dict[str, str]
        Mapping from dataframe column name to the clean display name shown in the plot.
    combined_model_keys : tuple[str, str], optional
        If provided, combine two model keys into one left label of the form
        `model1 / model2` while plotting only the first model's data.
    value_mode : str, default 'count'
        Whether the sequence-similarity values are stored as `count` or `percent`.
    gradient_colors : tuple[str, str], default ('#E6EEF7', '#0F5B99')
        Two endpoint colors used to generate the five-bin color gradient.
    model_colors : dict[str, str], optional
        Mapping from model column name to label color.
    bar_height : float, default 0.42
        Height of the BRENDA bar. The EnzyExtract bar uses this multiplied by
        `enzy_bar_height_ratio`.
    enzy_bar_height_ratio : float, default 0.8
        Relative height of the EnzyExtract bar compared to the BRENDA bar.
        Values above 1 make EnzyExtract thicker than BRENDA.
    save : bool, default False
        If True, saves the figure to disk.
    save_path : str or Path, optional
        Save directory or full filename override.
    show : bool, default True
        If True, displays the plot; otherwise closes the figure.

    Returns
    -------
    None
        Renders or saves the plot configuration directly.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df must be a pandas DataFrame.')

    if 'label' not in df.columns:
        raise ValueError("DataFrame is missing required column 'label'.")

    if not model_names:
        raise ValueError('model_names must contain at least one model mapping.')

    if combined_model_keys is not None:
        if not isinstance(combined_model_keys, (tuple, list)) or len(combined_model_keys) != 2:
            raise ValueError('combined_model_keys must contain exactly two model keys.')
        combined_model_keys = (combined_model_keys[0], combined_model_keys[1])
        missing_keys = [key for key in combined_model_keys if key not in model_names]
        if missing_keys:
            raise ValueError(
                'combined_model_keys must reference keys present in model_names. '
                f"Missing: {', '.join(missing_keys)}"
            )

    if not isinstance(value_mode, str):
        raise ValueError(
            f"value_mode must be a string ('count' or 'percent'), got {type(value_mode).__name__}."
        )

    value_mode = value_mode.strip().lower()
    if value_mode not in {'count', 'percent'}:
        raise ValueError("value_mode must be 'count' or 'percent'.")

    if len(gradient_colors) != 2:
        raise ValueError('gradient_colors must contain exactly two colors.')

    if bar_height <= 0:
        raise ValueError('bar_height must be greater than zero.')

    if enzy_bar_height_ratio <= 0:
        raise ValueError('enzy_bar_height_ratio must be greater than zero.')

    sequence_df = df.copy()
    parsed_labels = sequence_df['label'].apply(_parse_sequence_similarity_label)
    sequence_df['__dataset'] = parsed_labels.apply(lambda item: item[0] if item else np.nan)
    sequence_df['__bin'] = parsed_labels.apply(lambda item: item[1] if item else np.nan)

    ignored_labels = sequence_df.loc[sequence_df['__dataset'].isna() | sequence_df['__bin'].isna(), 'label']
    if not ignored_labels.empty:
        logger.warning(
            'Skipping %d unrecognized sequence similarity labels.',
            len(ignored_labels)
        )

    sequence_df = sequence_df.dropna(subset=['__dataset', '__bin']).copy()
    if sequence_df.empty:
        raise ValueError('No recognizable sequence similarity rows were found.')

    bin_order = ['0%-<50%', '50%-<70%', '70%-<90%', '90%-<100%', '100%']
    dataset_order = ['BRENDA', 'EnzyExtract']
    mean_order = ['BRENDA', 'EnzyExtract']

    resolved_model_keys: list[str] = []
    for model_key in model_names:
        if model_key not in sequence_df.columns:
            logger.warning(
                "Sequence similarity dataframe is missing model column '%s'. Skipping.",
                model_key
            )
            continue
        resolved_model_keys.append(model_key)

    combined_secondary_key: Optional[str] = None
    if combined_model_keys is not None:
        combined_primary_key, combined_secondary_key = combined_model_keys
        if combined_primary_key not in resolved_model_keys:
            raise ValueError(
                f"combined_model_keys primary key '{combined_primary_key}' is not available in the dataframe."
            )
        if combined_secondary_key not in resolved_model_keys:
            raise ValueError(
                f"combined_model_keys secondary key '{combined_secondary_key}' is not available in the dataframe."
            )

    plot_model_keys = [
        model_key for model_key in resolved_model_keys
        if model_key != combined_secondary_key
    ]

    display_model_keys = list(reversed(plot_model_keys))

    if not display_model_keys:
        raise ValueError('No valid model columns were found in the sequence similarity dataframe.')

    for model_key in resolved_model_keys:
        sequence_df[model_key] = sequence_df[model_key].apply(_extract_scalar)
        sequence_df[model_key] = pd.to_numeric(sequence_df[model_key], errors='coerce')

    agg_func = 'sum' if value_mode == 'count' else 'mean'
    grouped = (
        sequence_df
        .groupby(['__dataset', '__bin'], as_index=True)[resolved_model_keys]
        .agg(agg_func)
    )

    mean_sequence_df = sequence_df.loc[sequence_df['__bin'] == 'mean_seq'].copy()
    if not mean_sequence_df.empty:
        mean_grouped = (
            mean_sequence_df
            .groupby(['__dataset'], as_index=True)[resolved_model_keys]
            .agg('mean')
        )
    else:
        mean_grouped = pd.DataFrame(columns=resolved_model_keys)

    full_index = pd.MultiIndex.from_product(
        [dataset_order, bin_order],
        names=['__dataset', '__bin']
    )
    grouped = grouped.reindex(full_index)

    plot_values: dict[str, dict[str, dict[str, float]]] = {
        dataset_name: {
            model_key: {bin_label: 0.0 for bin_label in bin_order}
            for model_key in resolved_model_keys
        }
        for dataset_name in dataset_order
    }

    mean_values: dict[str, dict[str, float]] = {
        dataset_name: {model_key: np.nan for model_key in resolved_model_keys}
        for dataset_name in mean_order
    }

    for (dataset_name, bin_label), row in grouped.iterrows():
        if dataset_name not in plot_values or bin_label not in bin_order:
            continue
        for model_key in resolved_model_keys:
            value = row.get(model_key, np.nan)
            if pd.notna(value):
                plot_values[dataset_name][model_key][bin_label] = float(value)

    for dataset_name, row in mean_grouped.iterrows():
        if dataset_name not in mean_values:
            continue
        for model_key in resolved_model_keys:
            value = row.get(model_key, np.nan)
            if pd.notna(value):
                mean_values[dataset_name][model_key] = float(value)

    gradient = LinearSegmentedColormap.from_list(
        'sequence_similarity_gradient',
        [gradient_colors[0], gradient_colors[1]],
        N=len(bin_order)
    )
    bin_colors = [gradient(i) for i in np.linspace(0.0, 1.0, len(bin_order))]

    fallback_colors = sns.color_palette('husl', n_colors=len(resolved_model_keys))
    fallback_color_map = {
        model_key: fallback_colors[idx]
        for idx, model_key in enumerate(resolved_model_keys)
    }

    brenda_bar_height = float(bar_height)
    enzy_bar_height = float(bar_height) * float(enzy_bar_height_ratio)
    pair_edge_gap = 0.38
    divider_edge_gap = 0.78
    pair_gap = (brenda_bar_height + enzy_bar_height) / 2.0 + pair_edge_gap
    group_gap = pair_gap + (brenda_bar_height + enzy_bar_height) / 2.0 + (2.0 * divider_edge_gap)
    bar_total_width = 320.0
    bar_canvas_width = 328.0

    fig_height = max(4.5, 0.9 * len(display_model_keys) + 2.0)
    fig = plt.figure(figsize=(17.6, fig_height))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[18, 1.35],
        width_ratios=[5.9, 0.95],
        hspace=0.02,
        wspace=0.02
    )
    ax = fig.add_subplot(grid[0, 0])
    mean_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    legend_ax = fig.add_subplot(grid[1, 0])
    spacer_ax = fig.add_subplot(grid[1, 1])
    legend_ax.axis('off')
    spacer_ax.axis('off')
    sns.set_style('ticks')

    max_label_length = max((len(model_names.get(model_key, model_key)) for model_key in plot_model_keys), default=0)
    label_space = max(10.0, max_label_length * 0.45)
    label_x = -label_space * 0.32

    def _normalize_bin_widths(values_by_bin: dict[str, float]) -> dict[str, float]:
        raw_values = np.array([max(float(values_by_bin.get(bin_label, 0.0)), 0.0) for bin_label in bin_order], dtype=float)
        total_value = float(np.nansum(raw_values))
        if not np.isfinite(total_value) or total_value <= 0:
            return {bin_label: 0.0 for bin_label in bin_order}

        scaled = raw_values / total_value * bar_total_width
        positive_mask = scaled > 0
        min_segment_width = 10.0

        if np.count_nonzero(positive_mask) and np.any(scaled[positive_mask] < min_segment_width):
            adjusted = scaled.copy()
            fixed_mask = positive_mask & (adjusted < min_segment_width)
            fixed_total = float(np.sum(np.where(fixed_mask, min_segment_width, 0.0)))
            remaining_width = bar_total_width - fixed_total

            if remaining_width > 0:
                flexible_mask = positive_mask & ~fixed_mask
                flexible_total = float(np.sum(adjusted[flexible_mask]))

                if flexible_total > 0:
                    adjusted[fixed_mask] = min_segment_width
                    adjusted[flexible_mask] = adjusted[flexible_mask] / flexible_total * remaining_width
                else:
                    adjusted[positive_mask] = bar_total_width / np.count_nonzero(positive_mask)
            else:
                adjusted[positive_mask] = bar_total_width / np.count_nonzero(positive_mask)

            scaled = adjusted

        return {bin_label: float(width) for bin_label, width in zip(bin_order, scaled)}

    def _text_color_for_fill(fill_color) -> str:
        rgb = to_rgb(fill_color)
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return 'white' if luminance < 0.55 else 'black'

    plotted_totals: list[float] = []
    bar_extent_mins: list[float] = []
    bar_extent_maxs: list[float] = []
    mean_dataset_colors: dict[str, str] = {
        'BRENDA': gradient_colors[0],
        'EnzyExtract': gradient_colors[1]
    }

    def _model_row_y_positions(model_idx: int) -> tuple[float, float, float]:
        base_y = model_idx * group_gap
        brenda_y = base_y
        enzy_y = base_y + pair_gap
        label_y = base_y + pair_gap / 2.0
        return brenda_y, enzy_y, label_y

    def _row_separator_y(model_idx: int) -> float:
        _, enzy_y, _ = _model_row_y_positions(model_idx)
        return enzy_y + (enzy_bar_height / 2.0) + divider_edge_gap

    def _add_model_label(y_pos: float, model_key: str) -> None:
        if combined_model_keys is not None and model_key == combined_primary_key:
            part1 = model_names[combined_primary_key]
            part2 = model_names[combined_secondary_key]  # type: ignore[index]
            color1 = model_colors.get(combined_primary_key, fallback_color_map[combined_primary_key]) if model_colors else fallback_color_map[combined_primary_key]
            color2 = model_colors.get(combined_secondary_key, fallback_color_map[combined_secondary_key]) if model_colors else fallback_color_map[combined_secondary_key]

            label_box = HPacker(
                children=[
                    TextArea(part1, textprops={'color': color1, 'fontsize': 13, 'fontweight': 'bold'}),
                    TextArea(' / ', textprops={'color': '0.25', 'fontsize': 13, 'fontweight': 'bold'}),
                    TextArea(part2, textprops={'color': color2, 'fontsize': 13, 'fontweight': 'bold'})
                ],
                align='center',
                pad=0,
                sep=0
            )
            ax.add_artist(
                AnnotationBbox(
                    label_box,
                    (label_x, y_pos),
                    xycoords='data',
                    frameon=False,
                    box_alignment=(1.0, 0.5),
                    annotation_clip=False,
                    pad=0
                )
            )
            return

        display_name = model_names.get(model_key, model_key)
        label_color = (
            model_colors.get(model_key)
            if model_colors and model_key in model_colors
            else fallback_color_map[model_key]
        )
        ax.text(
            label_x,
            y_pos,
            display_name,
            ha='right',
            va='center',
            fontsize=13,
            fontweight='bold',
            color=label_color,
            clip_on=False
        )

    for model_idx, model_key in enumerate(display_model_keys):
        brenda_y, enzy_y, model_center_y = _model_row_y_positions(model_idx)
        _add_model_label(model_center_y, model_key)

        for dataset_name, y_pos, bar_thickness in (
            ('BRENDA', enzy_y, brenda_bar_height),
            ('EnzyExtract', brenda_y, enzy_bar_height),
        ):
            width_map = _normalize_bin_widths(plot_values[dataset_name][model_key])
            cumulative_left = 0.0
            nonzero_total = sum(width_map.values())
            plotted_totals.append(nonzero_total)

            for bin_idx, bin_label in enumerate(bin_order):
                width_value = width_map[bin_label]
                raw_value = plot_values[dataset_name][model_key].get(bin_label, 0.0)
                if width_value <= 0 or not np.isfinite(raw_value):
                    continue

                segment_color = bin_colors[bin_idx]
                ax.barh(
                    y=y_pos,
                    width=width_value,
                    left=cumulative_left,
                    height=bar_thickness,
                    color=segment_color,
                    edgecolor='0.82',
                    linewidth=0.45,
                    align='center',
                    zorder=3
                )

                half_height = bar_thickness / 2.0
                bar_extent_mins.append(y_pos - half_height)
                bar_extent_maxs.append(y_pos + half_height)

                segment_label = _format_sequence_similarity_value(raw_value, value_mode)
                ax.text(
                    cumulative_left + width_value / 2.0,
                    y_pos,
                    segment_label,
                    ha='center',
                    va='center',
                    fontsize=12,
                    color=_text_color_for_fill(segment_color),
                    clip_on=True,
                    zorder=4
                )

                cumulative_left += width_value

        if model_idx < len(display_model_keys) - 1:
            divider_y = _row_separator_y(model_idx)
            ax.hlines(
                divider_y,
                xmin=0.0,
                xmax=bar_total_width,
                color='0.85',
                linewidth=1.3,
                zorder=1
            )
            mean_ax.hlines(
                divider_y,
                xmin=0.0,
                xmax=100.0,
                color='0.75',
                linewidth=1.1,
                zorder=1
            )

    mean_x_min = np.inf
    mean_x_max = -np.inf
    mean_markers: dict[str, dict[str, tuple[float, float]]] = {}
    for dataset_name in mean_order:
        for model_idx, model_key in enumerate(display_model_keys):
            brenda_y, enzy_y, _ = _model_row_y_positions(model_idx)
            y_pos = enzy_y if dataset_name == 'BRENDA' else brenda_y
            mean_value = mean_values.get(dataset_name, {}).get(model_key, np.nan)
            if not np.isfinite(mean_value):
                continue
            mean_value_pct = mean_value * 100.0
            mean_markers.setdefault(model_key, {})
            mean_markers[model_key][dataset_name] = (mean_value_pct, y_pos)
            mean_x_min = min(mean_x_min, mean_value_pct)
            mean_x_max = max(mean_x_max, mean_value_pct)

    if np.isfinite(mean_x_min) and np.isfinite(mean_x_max):
        mean_padding = max(2.0, (mean_x_max - mean_x_min) * 0.08)
        mean_ax.set_xlim(mean_x_min - mean_padding, mean_x_max + mean_padding)
    else:
        mean_ax.set_xlim(0, 100)

    mean_xlim = mean_ax.get_xlim()
    mean_x_span = mean_xlim[1] - mean_xlim[0]
    mean_tick_values = np.linspace(mean_xlim[0], mean_xlim[1], 4)
    mean_ax.set_xticks(mean_tick_values)
    mean_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:.0f}%'))
    mean_ax.tick_params(axis='x', labelsize=11, length=4, width=1.0, colors='0.25', pad=4)

    for model_key in display_model_keys:
        entries = mean_markers.get(model_key, {})
        if 'BRENDA' in entries and 'EnzyExtract' in entries:
            (x1, y1) = entries['BRENDA']
            (x2, y2) = entries['EnzyExtract']
            mean_ax.plot([x1, x2], [y1, y2], color='0.35', linewidth=1.8, zorder=2)
            mean_ax.scatter(
                [x1, x2],
                [y1, y2],
                s=95,
                c=[mean_dataset_colors['BRENDA'], mean_dataset_colors['EnzyExtract']],
                edgecolors='black',
                linewidths=1.6,
                zorder=4
            )

            for dataset_name, (mean_value, y_pos) in entries.items():
                offset = 8 if mean_value <= (mean_xlim[0] + mean_x_span * 0.62) else -8
                text_alignment = 'left' if offset > 0 else 'right'
                mean_ax.annotate(
                    _format_sequence_similarity_value(mean_value, 'percent'),
                    xy=(mean_value, y_pos),
                    xytext=(offset, 0),
                    textcoords='offset points',
                    ha=text_alignment,
                    va='center',
                    fontsize=10.5,
                    fontweight='bold',
                    color='0.2',
                    clip_on=False,
                    zorder=5
                )

    mean_ax.set_xlabel('Mean sequence similarity (%)', fontsize=13, labelpad=10)
    mean_ax.set_yticks([])
    mean_ax.spines['left'].set_visible(False)
    mean_ax.spines['right'].set_visible(False)
    mean_ax.spines['top'].set_visible(False)
    mean_ax.spines['bottom'].set_visible(True)
    mean_ax.spines['bottom'].set_linewidth(1.1)
    mean_ax.spines['bottom'].set_color('0.25')

    ax.set_xlim(label_x - label_space * 0.35, bar_canvas_width)

    content_min_y = min(bar_extent_mins) if bar_extent_mins else 0.0
    content_max_y = max(bar_extent_maxs) if bar_extent_maxs else bar_total_width
    y_padding = max(brenda_bar_height, enzy_bar_height) * 0.75

    legend_handles = [
        Patch(facecolor=bin_colors[idx], edgecolor='none', label=bin_label)
        for idx, bin_label in enumerate(bin_order)
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc='center',
        ncol=len(bin_order),
        frameon=False,
        title='Best sequence identity to training set',
        fontsize=12,
        title_fontsize=13
    )

    ax.set_ylim(content_min_y - y_padding, content_max_y + y_padding)
    mean_ax.set_ylim(content_min_y - y_padding, content_max_y + y_padding)
    ax.invert_yaxis()
    mean_ax.invert_yaxis()
    ax.set_axis_off()
    mean_ax.tick_params(axis='y', left=False, labelleft=False)

    fig.subplots_adjust(left=0.04, right=0.985, top=0.985, bottom=0.065)

    fig.canvas.draw()

    if save:
        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / 'plots' / 'sequence_similarity_plots',
            'sequence_similarity_results.png'
        )
        if save_target is not None:
            plt.savefig(str(save_target), dpi=300, bbox_inches='tight', transparent=False)

    if show:
        plt.show()
    else:
        plt.close(fig)



def get_performance_subsets(
    df: pd.DataFrame, 
    models: list[str], 
    threshold_value: float = 10.0, 
    subset_type: str = 'best',
    threshold_mode: str = 'percentile'
) -> dict[str, set]:
    r"""
    Extract subsets of Reaction IDs based on model prediction error thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing an 'ID' column, an 'experimental_kcat' column, 
        and model columns named '{model}_kcat' (or matching model names).
    models : list[str]
        List of model identifiers used to locate the relevant columns in `df`.
    threshold_value : float, default 10.0
        The numeric value for the threshold. If threshold_mode is 'percentile', 
        this is the % of the dataset to include. If 'relative_margin', this is 
        the acceptable error percentage (e.g., 10 means 10% error margin).
    subset_type : str, default 'best'
        Must be 'best' (lowest error) or 'worst' (highest error).
    threshold_mode : str, default 'percentile'
        'percentile': takes the top/bottom X% of the dataset based on absolute log10 error.
        'relative_margin': takes predictions within/outside an X% relative error margin.

    Returns
    -------
    dict[str, set]
        Dictionary where keys are model names and values are sets of Reaction IDs.
    """
    
    if subset_type not in ['best', 'worst']:
        raise ValueError("subset_type must be either 'best' or 'worst'")
    if threshold_mode not in ['percentile', 'relative_margin', 'log_margin']:
        raise ValueError("threshold_mode must be 'percentile' or 'relative_margin' or 'log_margin'")
    
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
            
        if threshold_mode == 'percentile':
            errors = np.abs(np.log10(clean_df[mod_col]) - np.log10(clean_df['experimental_kcat']))
            
            if subset_type == 'best':
                threshold = np.percentile(errors, threshold_value)
                subset_mask = errors <= threshold
            else:
                threshold = np.percentile(errors, 100 - threshold_value)
                subset_mask = errors >= threshold
                
        elif threshold_mode == 'relative_margin':
            errors = np.abs(clean_df[mod_col] - clean_df['experimental_kcat']) / clean_df['experimental_kcat']
            fractional_threshold = threshold_value / 100.0
            
            if subset_type == 'best':
                subset_mask = errors <= fractional_threshold
            else:
                subset_mask = errors >= fractional_threshold

        elif threshold_mode == 'log_margin':
            errors = np.abs(np.log10(clean_df[mod_col]) - np.log10(clean_df['experimental_kcat']))
            
            if subset_type == 'best':
                subset_mask = errors <= threshold_value
            else:
                subset_mask = errors >= threshold_value
                
        model_sets[model] = set(clean_df.loc[subset_mask, 'ID'])
        
    return model_sets



def plot_ec_class_enrichment(
    df: pd.DataFrame, 
    model_names: dict[str, str],
    ec_column_name: str = "ec_number",
    threshold_value: float = 10.0, 
    subset_type: str = 'best',
    threshold_mode: str = 'percentile',
    pseudocount: float = 0.1,
    y_limit: Optional[float] = None,
    y_axis_right: bool = False,
    show: bool = True,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    model_colors: Optional[dict[str, str]] = None,
    show_title: bool = True
) -> None:
    """
    Plot the Enzyme Commission (EC) class enrichment for model prediction subsets.

    This function calculates the log2 fold change of EC class distributions within
    a specific performance subset of model predictions compared to the background 
    distribution of the entire dataset. It plots these enrichment scores as a 
    line chart across the 7 primary EC classes.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing prediction IDs and EC numbers.
    model_names : dict[str, str]
        Mapping from model identifier keys to their human-readable display names.
    ec_column_name : str, default 'ec_number'
        The name of the column in the dataframe containing EC numbers.
    threshold_value : float, default 10.0
        The numeric value defining the subset boundary (e.g., percentage or error margin).
    subset_type : str, default 'best'
        Defines the subset extraction logic (e.g., 'best' or 'worst' performing).
    threshold_mode : str, default 'percentile'
        The methodology for applying the threshold. Options include 'percentile', 
        'relative_margin', or 'log_margin'.
    pseudocount : float, default 0.1
        A small constant added to both subset and background distributions prior 
        to log2 fold change calculation to prevent division by zero or log(0).
    y_limit : float, optional
        Explicit symmetric limit for the Y-axis. If None, the limit dynamically 
        scales to 110% of the maximum absolute log2 fold change.
    y_axis_right : bool, default False
        If True, shifts the Y-axis tick labels and spine to the right side.
    show : bool, default True
        If True, displays the plot interactively using plt.show().
    save : bool, default False
        If True, saves the generated figure and its associated legend to disk.
    save_path : str or Path, optional
        Custom directory path or complete file path string to override default save tracking.
    model_colors : dict[str, str], optional
        Mapping from model identifier keys to valid color strings. If provided, 
        dictates the line and marker colors for each model. Falls back to the 
        'husl' palette for missing keys or if None.
    show_title : bool, default True
        If True, renders a dynamically generated title based on the subset criteria.

    Returns
    -------
    None
        Renders or saves the plot configuration directly.

    Raises
    ------
    ValueError
        If no valid EC classes (1-7) are found within the specified dataframe column.
    """
    subsets = get_performance_subsets(
        df, 
        list(model_names.keys()), 
        subset_type=subset_type,
        threshold_value=threshold_value,
        threshold_mode=threshold_mode
    )
    
    ec_name_map = {
        '1': '1: Oxidoreductases',
        '2': '2: Transferases',
        '3': '3: Hydrolases',
        '4': '4: Lyases',
        '5': '5: Isomerases',
        '6': '6: Ligases',
        '7': '7: Translocases'
    }
    
    valid_ec_classes = ['1', '2', '3', '4', '5', '6', '7']
    x_positions = np.arange(len(valid_ec_classes))
    
    bg_ec_series = df[ec_column_name].astype(str).str.split('.').str[0]
    bg_ec_series = bg_ec_series[bg_ec_series.isin(valid_ec_classes)]
    
    if bg_ec_series.empty:
        raise ValueError(f"No valid EC classes (1-7) found in column '{ec_column_name}'.")

    bg_dist = bg_ec_series.value_counts(normalize=True) * 100
    bg_vals = np.array([bg_dist.get(ec, 0.0) for ec in valid_ec_classes])
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.set_style("ticks")
    
    fallback_colors = sns.color_palette("husl", n_colors=len(model_names))
    max_abs_log2fc = 0.0
    
    for idx, (model_key, subset_ids) in enumerate(subsets.items()):
        subset_df = df[df['ID'].isin(subset_ids)]
        
        if subset_df.empty:
            continue
            
        sub_ec_series = subset_df[ec_column_name].astype(str).str.split('.').str[0]
        sub_ec_series = sub_ec_series[sub_ec_series.isin(valid_ec_classes)]
        
        sub_dist = sub_ec_series.value_counts(normalize=True) * 100
        sub_vals = np.array([sub_dist.get(ec, 0.0) for ec in valid_ec_classes])
        
        log2fc = np.log2((sub_vals + pseudocount) / (bg_vals + pseudocount))
        
        current_max_abs = np.max(np.abs(log2fc))
        if current_max_abs > max_abs_log2fc:
            max_abs_log2fc = current_max_abs
            
        clean_name = model_names.get(model_key, model_key)
        
        if model_colors and model_key in model_colors:
            line_color = model_colors[model_key]
        else:
            line_color = fallback_colors[idx]
            
        ax.plot(
            x_positions, 
            log2fc, 
            marker='o', 
            linewidth=2, 
            markersize=7, 
            color=line_color, 
            label=clean_name
        )

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, zorder=1)
    
    final_y_limit = y_limit if y_limit is not None else (max_abs_log2fc * 1.1 if max_abs_log2fc > 0 else 1.0)
    ax.set_ylim(-final_y_limit, final_y_limit)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([ec_name_map[ec] for ec in valid_ec_classes], rotation=45, ha='right')
    
    ax.set_xlabel("EC Class", fontsize=12)
    ax.set_ylabel(r"Enrichment ($\log_2$ Fold Change)", fontsize=12)
    
    if show_title:
        if threshold_mode == 'percentile':
            title_text = f"{int(threshold_value)}% {subset_type} predictions"
        elif threshold_mode == 'relative_margin':
            relation = "within" if subset_type == 'best' else "exceeding"
            title_text = f"Predictions {relation} {threshold_value}% rel error"
        elif threshold_mode == 'log_margin':
            relation = "within" if subset_type == 'best' else "exceeding"
            title_text = f"Predictions {relation} {threshold_value} log10 error"
        else:
            title_text = f"EC Class Enrichment ({subset_type})"
            
        plt.suptitle(title_text, fontsize=16, fontweight='bold', y=1.02)
    
    if y_axis_right:
        sns.despine(left=True, right=False, top=True, bottom=False)
        ax.yaxis.tick_right()                  
        ax.yaxis.set_label_position("right")
        ax.tick_params(
            axis='y',        
            right=True,     
            left=False
        )
    else:
        sns.despine()

    plt.tight_layout()

    if save:
        if threshold_mode == 'percentile':
            file_suffix = f"{int(threshold_value)}pct"
        elif threshold_mode == 'relative_margin':
            file_suffix = f"{int(threshold_value)}pct_rel"
        elif threshold_mode == 'log_margin':
            file_suffix = f"{threshold_value}log"
        else:
            file_suffix = "subset"
            
        save_filename = f"ec_enrichment_{subset_type}_{file_suffix}.png"
        legend_filename = f"ec_enrichment_{subset_type}_{file_suffix}_legend.png"
        
        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / "plots" / "ec_enrichment_plots",
            save_filename
        )
        if save_target is not None:
            plt.savefig(str(save_target), dpi=300, bbox_inches='tight', transparent=False)
            
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig_width = max(3, len(handles) * 1.5)
                fig_leg = plt.figure(figsize=(fig_width, 1))
                
                ax_leg = fig_leg.add_subplot(111)
                ax_leg.axis('off')
                
                ax_leg.legend(
                    handles, 
                    labels, 
                    loc='center', 
                    frameon=False, 
                    title="Model", 
                    ncol=len(handles)
                )
                
                leg_target = save_target.parent / legend_filename
                fig_leg.savefig(str(leg_target), dpi=300, bbox_inches='tight', transparent=True)
                plt.close(fig_leg)

    if show:
        plt.show()
    else:
        plt.close(fig)

        
    
def plot_model_intersection_sets(
    df: pd.DataFrame,
    model_names: dict[str, str],
    threshold_value: float = 10.0,
    threshold_mode: str = "percentile",
    subset_type: str = 'best',
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> tuple[pd.Series, pd.Series]:
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
    save_path : str or Path, optional
        If provided, overrides the save directory or full file path. If a
        directory is provided, the default filename is used. If a filename
        is provided, it is used directly. Ignored when save is False.
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

    model_sets = get_performance_subsets(
        df, 
        list(model_names.keys()), 
        threshold_value=threshold_value, 
        threshold_mode=threshold_mode, 
        subset_type=subset_type
    )

    if not model_sets or all(len(s) == 0 for s in model_sets.values()):
        return pd.Series(dtype=int), pd.Series(dtype=int)

    plot_ready_sets = {}
    for old_name, reaction_set in model_sets.items():
        new_name = model_names.get(old_name, old_name) 
        plot_ready_sets[new_name] = reaction_set
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*downcasting.*", category=FutureWarning)
        upset_data = from_contents(plot_ready_sets)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", 
            message=".*chained assignment.*", 
            category=FutureWarning
        )
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

    if threshold_mode == 'percentile':
        title_text = f"Intersection of {int(threshold_value)}% {subset_type} predictions"

    elif threshold_mode == 'relative_margin':
        relation = "within" if subset_type == 'best' else "exceeding"
        title_text = f"Intersection of predictions {relation} {threshold_value}% relative error"

    elif threshold_mode == 'log_margin':
        relation = "within" if subset_type == 'best' else "exceeding"
        title_text = f"Intersection of predictions {relation} {threshold_value} log10 error"

    plt.suptitle(title_text, fontsize=24, fontweight='bold')
    
    if save:
        if threshold_mode == 'percentile':
            file_suffix = f"{int(threshold_value)}pct"

        elif threshold_mode == 'relative_margin':
            file_suffix = f"{int(threshold_value)}pct_rel"

        elif threshold_mode == 'log_margin':
            file_suffix = f"{threshold_value}log"

        save_filename = f"{subset_type}_{file_suffix}_intersections.png"

        save_target = _resolve_save_target(
            save,
            save_path,
            RESULT_DIR / "plots" / "intersection_plots",
            save_filename
        )
        if save_target is not None:
            plt.savefig(
                str(save_target),
                dpi=300,
                bbox_inches='tight',
                transparent=False
            )
            
    if show:
        plt.show()
    else:
        plt.close(fig)

    set_sizes = pd.Series({name: len(items) for name, items in plot_ready_sets.items()})
    
    return upset.intersections, set_sizes
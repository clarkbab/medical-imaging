import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Literal, Optional, Tuple, Union

from mymi.regions import RegionNames
from mymi.regions.tolerances import get_region_tolerance

DEFAULT_FONT_SIZE = 15
DEFAULT_MAX_NFEV = int(1e6)
DEFAULT_METRIC_LEGEND_LOCS = {
    'dice': 'lower right',
    'hd': 'upper right',
    'hd-95': 'upper right',
    'msd': 'upper right'
}
DEFAULT_METRIC_LABELS = {
    'dice': 'DSC',
    'hd': 'HD [mm]',
    'hd-95': '95HD [mm]',
    'msd': 'MSD [mm]',
}
DEFAULT_METRIC_Y_LIMS = {
    'dice': (0, 1),
    'hd': (0, None),
    'hd-95': (0, None),
    'msd': (0, None)
}
for region in RegionNames:
    tol = get_region_tolerance(region)
    DEFAULT_METRIC_LEGEND_LOCS[f'apl-mm-tol-{tol}'] = 'upper right'
    DEFAULT_METRIC_LEGEND_LOCS[f'dm-surface-dice-tol-{tol}'] = 'lower right'
    DEFAULT_METRIC_LABELS[f'apl-mm-tol-{tol}'] = fr'APL, $\tau$={tol}mm'
    DEFAULT_METRIC_LABELS[f'dm-surface-dice-tol-{tol}'] = fr'Surface DSC, $\tau$={tol}mm'
    DEFAULT_METRIC_Y_LIMS[f'apl-mm-tol-{tol}'] = (0, None)
    DEFAULT_METRIC_Y_LIMS[f'dm-surface-dice-tol-{tol}'] = (0, 1)

DEFAULT_N_SAMPLES = int(1e4)
LOG_SCALE_X_UPPER_LIMS = [100, 150, 200, 300, 400, 600, 800]
LOG_SCALE_X_TICK_LABELS = [5, 10, 20, 50, 100, 200, 400, 800]

def plot_results(
    regions: Union[str, List[str]],
    models: Union[str, List[str]],
    metrics: Union[str, List[str], np.ndarray],
    stats: Union[str, List[str]],
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    fontsize: float = DEFAULT_FONT_SIZE,
    inner_wspace: float = 0.05,
    legend_locs: Optional[Union[str, List[str]]] = None,
    model_labels: Optional[Union[str, List[str]]] = None,
    outer_hspace: float = 0.2,
    outer_wspace: float = 0.2,
    savepath: Optional[str] = None,
    y_lim: bool = True) -> None:
    if type(metrics) == str:
        metrics = np.repeat([[metrics]], len(regions), axis=0)
    elif type(metrics) == list:
        metrics = np.repeat([metrics], len(regions), axis=0)
    n_metrics = metrics.shape[1]
    dfs = [dfs] if type(dfs) == pd.DataFrame else dfs
    dfs = dfs * n_metrics if len(dfs) == 1 else dfs         # Broadcast 'dfs' to 'n_metrics'.
    if legend_locs is not None:
        if type(legend_locs) == str:
            legend_locs = [legend_locs]
        assert len(legend_locs) == n_metrics
    models = [models] if type(models) == str else models
    n_models = len(models)
    if model_labels is not None:
        if type(model_labels) == str:
            model_labels = [model_labels]
        assert len(model_labels) == n_models
    regions = [regions] if type(regions) == str else regions
    n_regions = len(regions)
    stats = [stats] if type(stats) == str else stats
    stats = stats * n_metrics if len(stats) == 1 else stats         # Broadcast 'stats' to 'n_metrics'.

    # Lookup tables.
    # matplotlib.rc('text', usetex=True)
    # matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Create nested subplots.
    fig = plt.figure(constrained_layout=False, figsize=(6 * metrics.shape[1], 6 * n_regions))
    outer_gs = fig.add_gridspec(hspace=outer_hspace, nrows=n_regions, ncols=metrics.shape[1], wspace=outer_wspace)
    for i, region in enumerate(regions):
        for j in range(n_metrics):
            df = dfs[j]
            metric = metrics[i, j]
            stat = stats[j]
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[i, j], width_ratios=[1, 19], wspace=0.05)
            axs = [plt.subplot(cell) for cell in inner_gs]
            y_lim = DEFAULT_METRIC_Y_LIMS[metric] if y_lim else (None, None)
            legend_loc = legend_locs[j] if legend_locs is not None else DEFAULT_METRIC_LEGEND_LOCS[metric]
            plot_bootstrap_fit(region, models, metric, stat, df, axs=axs, fontsize=fontsize, legend_loc=legend_loc, model_labels=model_labels, split=True, wspace=inner_wspace, x_scale='log', y_label=DEFAULT_METRIC_LABELS[metric], y_lim=y_lim)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)

    plt.show()

def plot_bootstrap_fit(
    region: str, 
    models: Union[str, List[str]],
    metric: str,
    stat: str,
    df: pd.DataFrame,
    alpha_ci: float = 0.2,
    alpha_points: float = 1.0,
    axs: Optional[Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]]] = None,
    figsize: Tuple[float, float] = (8, 6),
    fontsize: float = DEFAULT_FONT_SIZE,
    fontweight: Literal['normal', 'bold'] = 'normal',
    index: Optional[int] = None,
    legend_loc: Optional[str] = None,
    model_labels: Optional[Union[str, List[str]]] = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    show_ci: bool = True,
    show_limits: bool = False,
    show_points: bool = True,
    split: bool = True,
    title: str = '',
    wspace: float = 0.05,
    x_scale: str = 'log',
    y_label: str = '',
    y_lim: Optional[Tuple[float, float]] = None):
    colours = sns.color_palette('colorblind')[:len(models)]
    legend_loc = DEFAULT_METRIC_LEGEND_LOCS[metric] if legend_loc is None else legend_loc
    models = [models] if type(models) == str else models
    if model_labels is not None:
        model_labels = [model_labels] if type(model_labels) == str else model_labels
        assert len(model_labels) == len(models)
        
    if axs is None:
        plt.figure(figsize=figsize)
        if split:
            _, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 19]})
        else:
            axs = [plt.gca()]
    
    for ax in axs:
        # Plot data.
        for i, model in enumerate(models):
            colour = colours[i]
            model_label = model_labels[i] if model_labels is not None else model

            if index is not None:
                # Load bootstrapped sample.
                samples, n_trains = load_bootstrap_samples(region, model, metric, stat, include_n_trains=True, n_samples=n_samples)
                sample = samples[index]
                x_raw = np.array([])
                y_raw = np.array([])
                for n_train, n_train_sample in zip(n_trains, sample):
                    x_raw = np.concatenate((x_raw, n_train * np.ones(len(n_train_sample))))
                    y_raw = np.concatenate((y_raw, n_train_sample))

                # Load prediction on bootstrapped sample.
                preds = load_bootstrap_predictions(region, model, metric, stat, n_samples=n_samples)
                pred = preds[index]

                # Plot.
                x = np.linspace(0, len(pred) - 1, num=len(pred))
                ax.plot(x, pred, color=colour, label=model_label)
                if show_points:
                    ax.scatter(x_raw, y_raw, color=colour, marker='o', alpha=alpha_points)
            else:
                # Load raw data and bootstrapped predictions.
                x_raw, y_raw = raw_data(region, model, metric, df)
                preds = load_bootstrap_predictions(region, model, metric, stat, n_samples=n_samples)

                # Calculate means.
                means = preds.mean(axis=0)

                # Plot.
                x = np.linspace(0, len(means) - 1, num=len(means))
                ax.plot(x, means, color=colour, label=model_label)
                if show_ci:
                    low_ci = np.quantile(preds, 0.025, axis=0)
                    high_ci = np.quantile(preds, 0.975, axis=0)
                    ax.fill_between(x, low_ci, high_ci, color=colour, alpha=alpha_ci)
                if show_limits:
                    min = preds.min(axis=0)
                    max = preds.max(axis=0)
                    ax.plot(x, min, c='black', linestyle='--', alpha=0.5)
                    ax.plot(x, max, c='black', linestyle='--', alpha=0.5)
                if show_points:
                    ax.scatter(x_raw, y_raw, color=colour, marker='o', alpha=alpha_points)

    # Set axis scale.
    if split:
        axs[1].set_xscale(x_scale)
    else:
        axs[0].set_xscale(x_scale)

    # Set x tick labels.
    x_upper_lim = None
    if split:
        axs[0].set_xticks([0])
        if x_scale == 'log':
            x_upper_lim = axs[1].get_xlim()[1]      # Record x upper lim as setting ticks overrides this.
            axs[1].set_xticks(LOG_SCALE_X_TICK_LABELS)
            axs[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Set axis limits.
    if split:
        axs[0].set_xlim(-0.5, 0.5)
        axs[1].set_xlim(4.5, x_upper_lim)
        # if x_scale == 'log':
        #     axs[1].set_xlim(4.5, x_upper_lim)
        # else:
        #     axs[1].set_xlim(4.5, None)

    # Set y label.
    axs[0].set_ylabel(y_label, fontsize=fontsize, weight=fontweight)

    # Set y limits.
    axs[0].set_ylim(y_lim)
    if split:
        axs[1].set_ylim(y_lim)

    # Set y tick label fontsize.
    axs[0].tick_params(axis='x', which='major', labelsize=fontsize)
    axs[0].tick_params(axis='y', which='major', labelsize=fontsize)
    if split:
        axs[1].tick_params(axis='x', which='major', labelsize=fontsize)

    # Set title.
    title = title if title else region
    if split:
        axs[1].set_title(title, fontsize=fontsize, weight=fontweight)
    else:
        axs[0].set_title(title, fontsize=fontsize, weight=fontweight)

    # Add legend.
    if split:
        axs[1].legend(fontsize=fontsize, loc=legend_loc)
    else:
        axs[0].legend(fontsize=fontsize, loc=legend_loc)

    if split:
        # Hide axes' spines.
        axs[0].spines['right'].set_visible(False)
        axs[1].spines['left'].set_visible(False)
        axs[1].set_yticks([])

        # Add split between axes.
        plt.subplots_adjust(wspace=wspace)
        d_x_0 = .1
        d_x_1 = .006
        d_y = .03
        kwargs = dict(transform=axs[0].transAxes, color='k', clip_on=False)
        axs[0].plot((1 - (d_x_0 / 2), 1 + (d_x_0 / 2)), (-d_y / 2, d_y / 2), **kwargs)  # bottom-left diagonal
        axs[0].plot((1 - (d_x_0 / 2), 1 + (d_x_0 / 2)), (1 - (d_y / 2), 1 + (d_y / 2)), **kwargs)  # top-left diagonal
        kwargs = dict(transform=axs[1].transAxes, color='k', clip_on=False)
        axs[1].plot((-d_x_1 / 2, d_x_1 / 2), (-d_y / 2, d_y / 2), **kwargs)  # bottom-left diagonal
        axs[1].plot((-d_x_1 / 2, d_x_1 / 2), (1 - (d_y / 2), 1 + (d_y / 2)), **kwargs)  # top-left diagonal

def raw_data(
    region: str,
    model: str,
    metric: str,
    df: pd.DataFrame) -> np.ndarray:
    data_df = df[(df.metric == metric) & (df['model-type'] == model) & (df.region == region)]
    x = data_df['n-train']
    y = data_df['value']
    return x, y

def residuals(f):
    def inner(params, x, y, weights):
        y_pred = f(x, params)
        rs = y - y_pred
        if weights is not None:
            rs *= weights
        return rs
    return inner

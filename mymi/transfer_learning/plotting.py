import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FixedLocator
import numpy as np
import os
import seaborn as sns
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from mymi import config
config.Regions.mode = 1
from mymi.regions import RegionNames
from mymi.regions.tolerances import get_region_tolerance
from mymi.utils import arg_assert_literal, arg_broadcast, arg_to_list

from .differences import load_bootstrap_significant_differences
from .predictions import load_bootstrap_predictions, load_raw_evaluation_data

DEFAULT_FONT_SIZE = 8
DEFAULT_MAX_NFEV = int(1e6)
DEFAULT_METRIC_LEGEND_LOCS = {
    'dice': 'lower right',
    'hd': 'upper right',
    'hd-95': 'upper right',
    'msd': 'upper right'
}
DEFAULT_METRIC_LABELS = {
    'dice': 'DSC',
    'hd': 'HD (mm)',
    'hd-95': '95HD (mm)',
    'msd': 'MSD (mm)',
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
    DEFAULT_METRIC_LABELS[f'apl-mm-tol-{tol}'] = fr'APL, $\tau$={tol}mm (mm)'
    DEFAULT_METRIC_LABELS[f'dm-surface-dice-tol-{tol}'] = fr'Surface DSC, $\tau$={tol}mm'
    DEFAULT_METRIC_Y_LIMS[f'apl-mm-tol-{tol}'] = (0, None)
    DEFAULT_METRIC_Y_LIMS[f'dm-surface-dice-tol-{tol}'] = (0, 1)

DEFAULT_N_SAMPLES = int(1e4)
DEFAULT_N_TRAINS = [5, 10, 20, 50, 100, 200, 400, 800, None]
LOG_SCALE_X_UPPER_LIMS = [100, 150, 200, 300, 400, 600, 800]
LOG_SCALE_X_TICK_LABELS = [5, 10, 20, 50, 100, 200, 400, 800]

def megaplot(
    dataset: Union[str, List[str]],
    region: Union[str, List[str]],
    model: Union[str, List[str]],
    metric: Union[str, List[str], np.ndarray],
    stat: Union[str, List[str]],
    figsize: Optional[Tuple[float, float]] = None,
    fontsize: float = DEFAULT_FONT_SIZE,
    fontsize_tick_label: Optional[float] = None,
    height_ratios: Optional[List[float]] = None,
    hspace_grid: float = 0.25,
    hspace_plot: float = 0.25,
    hspace_plot_xlabel: float = 0.5,
    legend_loc: Optional[Union[str, List[str]]] = None,
    model_label: Optional[Union[str, List[str]]] = None,
    savepath: Optional[str] = None,
    secondary_stat: Optional[Union[str, List[str], np.ndarray]] = None,
    wspace: float = 0.25,
    y_lim: bool = True,
    **kwargs: Dict[str, Any]) -> None:
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)
    n_regions = len(regions)
    if height_ratios is not None:
        assert len(height_ratios) == n_regions
    if type(metric) is str:
        metrics = np.repeat([[metric]], n_regions, axis=0)
    elif type(metric) is list:
        metrics = np.repeat([metric], n_regions, axis=0)
    else:
        metrics = metric
    if type(secondary_stat) is str:
        secondary_stats = np.repeat([[secondary_stat]], n_regions, axis=0)
    elif type(secondary_stat) is list:
        secondary_stats = np.repeat([secondary_stat], n_regions, axis=0)
    else:
        secondary_stats = secondary_stat

    n_metrics = metrics.shape[1]
    legend_locs = arg_to_list(legend_loc, str)
    if legend_locs is not None:
        assert len(legend_locs) == n_metrics
    models = arg_to_list(model, str)
    n_models = len(models)
    model_labels = arg_to_list(model_label, str)
    if model_labels is not None:
        assert len(model_labels) == n_models
    stats = arg_broadcast(stat, n_metrics, str)

    # Lookup tables.
    # matplotlib.rc('text', usetex=True)
    # matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Create main gridspec (size: num OARs x num metrics).
    if figsize is None:
        figsize=(6 * metrics.shape[1], 6 * n_regions)
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = GridSpec(n_regions, metrics.shape[1], figure=fig, height_ratios=height_ratios, hspace=hspace_grid, wspace=wspace)
    for i, region in enumerate(regions):
        for j in range(n_metrics):
            metric = metrics[i, j]
            stat = stats[j]
            sec_stat = secondary_stats[i, j] if secondary_stats is not None else None
            x_label = 'num. institutional samples (n)' if i == len(regions) - 1 else None
            y_lim = DEFAULT_METRIC_Y_LIMS[metric] if y_lim else (None, None)
            legend_loc = legend_locs[j] if legend_locs is not None else DEFAULT_METRIC_LEGEND_LOCS[metric]
            plot_bootstrap_fit(datasets, region, models, metric, stat, fontsize=fontsize, fontsize_y_tick_label=fontsize_y_tick_label, hspace=hspace_plot, hspace_xlabel=hspace_plot_xlabel, legend_loc=legend_loc, model_labels=model_labels, outer_gs=gs[i, j], secondary_stat=sec_stat, split=True, x_label=x_label, x_scale='log', y_label=DEFAULT_METRIC_LABELS[metric], y_lim=y_lim, **kwargs)

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)

    size = fig.get_size_inches() * fig.dpi
    plt.show()

def plot_bootstrap_fit(
    dataset: Union[str, List[str]],
    region: str, 
    model_type: Union[str, List[str]],
    metric: str,
    stat: Literal['mean', 'q1', 'q3'],
    alpha_ci: float = 0.2,
    alpha_points: float = 1.0,
    alpha_secondary: float = 0.5,
    ax: Optional[Union[mpl.axes.Axes, List[mpl.axes.Axes]]] = None,
    diff_hspace: Optional[float] = 0.05,
    diff_marker: str = 's',
    figsize: Tuple[float, float] = (8, 6),
    fontsize: float = DEFAULT_FONT_SIZE,
    fontsize_tick_label: Optional[float] = None,
    fontweight: Literal['normal', 'bold'] = 'normal',
    hspace: float = 0,
    hspace_xlabel: float = 0.2,
    labelpad: float = 0,
    legend_loc: Optional[str] = None,
    linewidth: float = 1,
    model_labels: Optional[Union[str, List[str]]] = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    point_size: float = 1,
    secondary_stat: Optional[str] = None,
    show_ci: bool = True,
    show_diff: bool = True,
    show_legend: bool = True,
    show_limits: bool = False,
    show_points: bool = True,
    show_secondary_stat_ci: bool = False, 
    show_secondary_stat_diff: bool = True,
    split: bool = True,
    split_wspace: Optional[float] = 0.05,
    outer_gs: Optional[mpl.gridspec.SubplotSpec] = None,
    ticklength: float = 1,
    tickpad: float = 2,
    title: str = '',
    titlepad: float = 2,
    x_label: Optional[str] = None,
    x_scale: str = 'log',
    y_label: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = None):
    datasets = arg_to_list(dataset, str)
    model_types = arg_to_list(model_type, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))
    axs = arg_to_list(ax, mpl.axes.Axes)
    fontsize_tick_label = fontsize if fontsize_tick_label is None
    model_colours = sns.color_palette('colorblind')[:len(model_types)]
    legend_loc = DEFAULT_METRIC_LEGEND_LOCS[metric] if legend_loc is None else legend_loc
    if model_labels is not None:
        model_labels = [model_labels] if type(model_labels) == str else model_labels
        assert len(model_labels) == len(model_types)
    if secondary_stat is None:
        show_secondary_stat_diff = False
        
    # Create gridspec. If using 'megaplot' this sits inside the large megaplot gridspec.
    # Size = (2 x 1) if showing significant differences, otherwise (1 x 1). Both primary and secondary
    # stat significant differences share the bottom row.
    gs_size = (2, 1) if show_diff else (1, 1)
    gs_height_ratios = (48 if show_secondary_stat_diff else 49, 2 if show_secondary_stat_diff else 1) if show_diff else None
    hspace = hspace_xlabel if x_label is not None else hspace
    if outer_gs is None:
        # Create standalone gridspec for 'bootstrap fit'.
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(*gs_size, figure=fig, height_ratios=gs_height_ratios, hspace=hspace)
    else:
        # Nest 'bootstrap fit' gridspec in 'megaplot' gridspec.
        fig = outer_gs.get_gridspec().figure
        gs = GridSpecFromSubplotSpec(*gs_size, subplot_spec=outer_gs, height_ratios=gs_height_ratios, hspace=hspace)

    # Create data gridspec.
    # Size = (1 x 2) if splitting, otherwise (1, 1).
    data_gs_size = (1, 2) if split else (1, 1)
    data_gs_width_ratios = (1, 19) if split else None
    data_gs = GridSpecFromSubplotSpec(*data_gs_size, subplot_spec=gs[0, 0], width_ratios=data_gs_width_ratios, wspace=split_wspace)

    # Create difference gridspec.
    if show_diff:
        if split:
            diff_gs_size = (2, 2) if show_secondary_stat_diff else (1, 2)
        else:
            diff_gs_size = (2, 1) if show_secondary_stat_diff else (1, 1)
        diff_gs_hspace = diff_hspace if show_secondary_stat_diff else None
        diff_gs_width_ratios = (1, 19) if split else None
        diff_gs = GridSpecFromSubplotSpec(*diff_gs_size, hspace=diff_gs_hspace, subplot_spec=gs[1, 0], width_ratios=diff_gs_width_ratios, wspace=split_wspace)

    # Create subplot axes.
    # The dimensions of these lists will change depending upon the options:
    #   'split', 'show_diff' and 'show_secondary_stat_diff'. 
    axs = [
        [
            fig.add_subplot(data_gs[0, 0]),
            # Add this axis if we're splitting horizontally. Public results (n=0) will be shown on left split,
            # all other results (n=5,10,...) will be shown on right split.
            *([fig.add_subplot(data_gs[0, 1])] if split else []),
        ],
        # Add this/these axes if we're showing significant differences below the plot. This is another plot
        # that's very thin vertically.
        *([
            [
                # If we're split, don't show significant differences on the left split as it's just the public
                # model results here.
                *([None] if split else []),
                fig.add_subplot(diff_gs[0, 1] if split else diff_gs[0, 0])
            ]
        ] if show_diff else []),
        # Add these axes if we're showing secondary statistic (e.g. Q1/Q3) significant differences.
        *([
            [
                *([None] if split else []),
                fig.add_subplot(diff_gs[1, 1] if split else diff_gs[1, 0])
            ]
        ] if show_diff and show_secondary_stat_diff else [])
    ]
    
    # Plot main data.
    for ax in axs[0]:
        if ax is None:
            continue

        for i, model_type in enumerate(model_types):
            model_colour = model_colours[i]
            model_label = model_labels[i] if model_labels is not None else model_type

            # Load bootstrapped predictions.
            preds, _ = load_bootstrap_predictions(datasets, region, model_type, metric, stat, n_samples=n_samples)

            # Load data for secondary statistic.
            if secondary_stat:
                sec_preds, _ = load_bootstrap_predictions(datasets, region, model_type, metric, secondary_stat, n_samples=n_samples)

            # Plot mean value of 'stat' over all bootstrapped samples (convergent value).
            means = preds.mean(axis=0)
            x = np.linspace(0, len(means) - 1, num=len(means))
            ax.plot(x, means, color=model_colour, label=model_label, linewidth=linewidth)

            # Plot secondary statistic mean values.
            if secondary_stat:
                sec_means = sec_preds.mean(axis=0)
                ax.plot(x, sec_means, color=model_colour, alpha=alpha_secondary, linestyle='--', linewidth=linewidth)

            # Plot secondary statistic 95% CIs.
            if secondary_stat and show_secondary_stat_ci:
                low_ci = np.quantile(sec_preds, 0.025, axis=0)
                high_ci = np.quantile(sec_preds, 0.975, axis=0)
                ax.fill_between(x, low_ci, high_ci, alpha=alpha_secondary * alpha_ci, color=model_colour, linewidth=linewidth)

            # Plot 95% CIs for statistic.
            if show_ci:
                low_ci = np.quantile(preds, 0.025, axis=0)
                high_ci = np.quantile(preds, 0.975, axis=0)
                ax.fill_between(x, low_ci, high_ci, alpha=alpha_ci, color=model_colour, linewidth=linewidth)

            # Plot upper/lower limits for statistic.
            if show_limits:
                min = preds.min(axis=0)
                max = preds.max(axis=0)
                ax.plot(x, min, c='black', linestyle='--', alpha=0.5, linewidth=linewidth)
                ax.plot(x, max, c='black', linestyle='--', alpha=0.5, linewidth=linewidth)

            # Plot original data (before bootstrapping was applied).
            if show_points:
                data, n_trains = load_raw_evaluation_data(datasets, region, model_type, metric)

                if stat == 'mean':
                    data = data.mean(axis=2)
                elif stat == 'q1':
                    data = np.quantile(data, 0.25, axis=2)
                elif stat == 'q3':
                    data = np.quantile(data, 0.75, axis=2)

                x = np.repeat(n_trains, data.shape[1])
                y = data.flatten()
                ax.scatter(x, y, alpha=alpha_points, color=model_colour, marker='o', s=point_size)

    # Plot difference.
    if show_diff:
        assert len(model_types) == 2

        # Load significant difference.
        best_models, _, _ = load_bootstrap_significant_differences(datasets, region, *model_types, metric, stat, n_samples=n_samples)
        
        # Plot model A points.
        diff_ax = axs[1][1] if split else axs[1][0]
        x = np.argwhere(best_models == 0).flatten()
        diff_ax.scatter(x, [0] * len(x), color=model_colours[0], marker=diff_marker, s=point_size)

        # Plot model B points.
        x = np.argwhere(best_models == 1).flatten()
        diff_ax.scatter(x, [0] * len(x), color=model_colours[1], marker=diff_marker, s=point_size)

        if show_secondary_stat_diff:
            # Load significant difference.
            best_models, _, _ = load_bootstrap_significant_differences(datasets, region, *model_types, metric, secondary_stat, n_samples=n_samples)
            
            # Plot model A points.
            diff_ax = axs[2][1] if split else axs[2][0]
            x = np.argwhere(best_models == 0).flatten()
            diff_ax.scatter(x, [0] * len(x), color=model_colours[0], marker=diff_marker, s=point_size)

            # Plot model B points.
            x = np.argwhere(best_models == 1).flatten()
            diff_ax.scatter(x, [0] * len(x), color=model_colours[1], marker=diff_marker, s=point_size)

    # Set axis scale.
    if split:
        axs[0][1].set_xscale(x_scale)

        if show_diff:
            axs[1][1].set_xscale(x_scale)

            if show_secondary_stat_diff:
                axs[2][1].set_xscale(x_scale)
    else:
        axs[0][0].set_xscale(x_scale)

        if show_diff:
            axs[1][0].set_xscale(x_scale)

            if show_secondary_stat_diff:
                axs[2][0].set_xscale(x_scale)

    # Set x tick labels.
    x_upper_lim = None
    if split:
        axs[0][0].set_xticks([0])
        if show_diff:
            axs[1][1].get_xaxis().set_visible(False)
            axs[1][1].get_yaxis().set_visible(False)

            if show_secondary_stat_diff:
                axs[2][1].get_xaxis().set_visible(False)
                axs[2][1].get_yaxis().set_visible(False)

        if x_scale == 'log':
            x_upper_lim = axs[0][1].get_xlim()[1]      # Record x upper lim as setting ticks overrides this.
            axs[0][1].set_xticks(LOG_SCALE_X_TICK_LABELS)
            axs[0][1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    else:
        if show_diff:
            axs[1][0].get_xaxis().set_visible(False)
            axs[1][0].get_yaxis().set_visible(False)

            if show_secondary_stat_diff:
                axs[2][0].get_xaxis().set_visible(False)
                axs[2][0].get_yaxis().set_visible(False)

    # Set axis limits.
    if split:
        axs[0][0].set_xlim(-0.5, 0.5)
        axs[0][1].set_xlim(4.5, x_upper_lim)

        if show_diff:
            axs[1][1].set_xlim(4.5, x_upper_lim)

            if show_secondary_stat_diff:
                axs[2][1].set_xlim(4.5, x_upper_lim)

    # Set x/y label.
    if x_label is not None:
        x_axis = axs[0][1] if split else axs[0][0]
        x_axis.set_xlabel(x_label, fontsize=fontsize, labelpad=labelpad, weight=fontweight)
    if y_label is not None:
        axs[0][0].set_ylabel(y_label, fontsize=fontsize, labelpad=labelpad, weight=fontweight)

    # Set y limits.
    axs[0][0].set_ylim(y_lim)
    if split:
        axs[0][1].set_ylim(y_lim)

    # Set x/y tick label fontsize.
    axs[0][0].tick_params(axis='x', which='major', labelsize=fontsize_tick_label, pad=tickpad)
    axs[0][0].tick_params(axis='y', which='major', labelsize=fontsize_tick_label, pad=tickpad)
    axs[0][0].tick_params(axis='x', which='minor', direction='in')
    if split:
        axs[0][1].tick_params(axis='x', which='major', labelsize=fontsize_tick_label, pad=tickpad)
        axs[0][1].tick_params(axis='x', which='minor', direction='in')

    # Set axis spine/tick linewidths and tick lengths.
    spines = ['top', 'bottom','left','right']
    for spine in spines:
        axs[0][0].spines[spine].set_linewidth(linewidth)
    axs[0][0].tick_params(which='both', length=ticklength, width=linewidth)
    if split:
        for spine in spines:
            axs[0][1].spines[spine].set_linewidth(linewidth)
        axs[0][1].tick_params(which='both', length=ticklength, width=linewidth)

        if show_diff:
            for spine in spines:
                axs[1][1].spines[spine].set_linewidth(linewidth)
            if show_secondary_stat_diff:
                for spine in spines:
                    axs[2][1].spines[spine].set_linewidth(linewidth)
    else:
        if show_diff:
            for spine in spines:
                axs[1][0].spines[spine].set_linewidth(linewidth)
            if show_secondary_stat_diff:
                for spine in spines:
                    axs[2][0].spines[spine].set_linewidth(linewidth)

    # Set title.
    title = title if title else region
    title_axis = axs[0][1] if split else axs[0][0]
    title_axis.set_title(title, fontsize=fontsize, pad=titlepad, weight=fontweight)

    # Add legend.
    if show_legend:
        if split:
            axs[0][1].legend(fontsize=fontsize, loc=legend_loc)
        else:
            axs[0][0].legend(fontsize=fontsize, loc=legend_loc)

    if split:
        # Hide axes' spines.
        axs[0][0].spines['right'].set_visible(False)
        axs[0][1].spines['left'].set_visible(False)
        axs[0][1].set_yticks([])

        # Add split between axes.
        d_x_0 = .1
        d_x_1 = .006
        d_y = .03
        kwargs = dict(transform=axs[0][0].transAxes, color='k', clip_on=False)
        axs[0][0].plot((1 - (d_x_0 / 2), 1 + (d_x_0 / 2)), (-d_y / 2, d_y / 2), linewidth=linewidth, **kwargs)  # bottom-left diagonal
        axs[0][0].plot((1 - (d_x_0 / 2), 1 + (d_x_0 / 2)), (1 - (d_y / 2), 1 + (d_y / 2)), linewidth=linewidth, **kwargs)  # top-left diagonal
        kwargs = dict(transform=axs[0][1].transAxes, color='k', clip_on=False)
        axs[0][1].plot((-d_x_1 / 2, d_x_1 / 2), (-d_y / 2, d_y / 2), linewidth=linewidth, **kwargs)  # bottom-left diagonal
        axs[0][1].plot((-d_x_1 / 2, d_x_1 / 2), (1 - (d_y / 2), 1 + (d_y / 2)), linewidth=linewidth, **kwargs)  # top-left diagonal

    # Apply formatting (e.g 1k, 2k, etc.) if all values are over 1000.
    y_tick_locs = axs[0][0].get_yticks()
    apply_formatting = True
    for y_tick_loc in y_tick_locs:
        if y_tick_loc != 0 and y_tick_loc < 1000:
            apply_formatting = False
    if apply_formatting:
        def format_y_tick(y_tick_loc: float):
            y_tick = int(y_tick_loc / 1000)
            return f'{y_tick}k'
        axs[0][0].yaxis.set_major_locator(FixedLocator(y_tick_locs))
        axs[0][0].set_yticklabels([format_y_tick(y) for y in y_tick_locs])
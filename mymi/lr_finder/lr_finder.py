import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator
import numpy as np
from operator import itemgetter
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple, Union

from mymi import config
from mymi import logging
from mymi.regions import regions_to_list
from mymi.types import PatientRegions
from mymi.utils import arg_to_list

DEFAULT_FONT_SIZE = 8

def load_data(
    model_name: str,
    run_name: str) -> Dict[str, np.ndarray]:
    filepath = os.path.join(config.directories.models, model_name, run_name, 'lr-finder.json')
    if not os.path.exists(filepath):
        raise ValueError(f"Model '{model_name}', run '{run_name}' does not exist. Filepath: {filepath}.")
    return json.load(open(filepath))

def suggestion(
    model_name: str,
    run_name: str,
    skip_end: int = 0,
    skip_start: int = 0,
    smooth: bool = False,
    smooth_kernel_sd: float = 1) -> float:
    # Get loss data.
    results = load_data(model_name, run_name)
    losses = results['loss']

    # Remove 'nan' losses.
    real_idxs = np.argwhere(~np.isnan(losses)).flatten()
    lr = list(np.array(lr)[real_idxs])
    losses = list(np.array(losses)[real_idxs])

    # Smooth the signal.
    if smooth:
        losses = gaussian_filter1d(losses, smooth_kernel_sd)

    # Remove start/end points.
    if skip_end != 0:
        losses = losses[skip_start:-skip_end]
    else:
        losses = losses[skip_start:]

    # Get minimum gradient.
    min_grad_idx = np.gradient(losses).argmin()
    min_grad_idx = skip_start + min_grad_idx

    # Get associated learning rate.
    lr = results['lr'][min_grad_idx]

    return lr

def plot(
    model_name: str,
    run_name: Union[str, List[Union[str, List[str]]]],
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 12),
    fontsize: float = DEFAULT_FONT_SIZE,
    fontsize_label: Optional[float] = None,
    fontsize_tick_label: Optional[float] = None,
    fontsize_legend: Optional[float] = None,
    fontsize_title: Optional[float] = None,
    label: Optional[Union[str, List[str]]] = None,
    legend_bbox: Optional[Tuple[float, float]] = (1, 1),
    region: Optional[PatientRegions] = 'all',
    savepath: Optional[str] = None,
    show: bool = True,
    show_legend: bool = True,
    show_sugg_lr: bool = True,
    show_title: bool = True,
    skip_end: int = 0,
    skip_start: int = 0,
    smooth: bool = False,
    smooth_kernel_sd: float = 1,
    sugg_skip_end: int = 0,
    sugg_skip_start: int = 0,
    vline: Optional[Union[float, List[float]]] = None,
    x_lim: Tuple[Optional[float], Optional[float]] = (None, None),
    y_lim: Tuple[Optional[float], Optional[float]] = (None, None)) -> None:
    regions = regions_to_list(region)
    run_names = arg_to_list(run_name, str)
    labels = arg_to_list(label, str)
    if labels is not None and len(labels) != len(run_names):
        raise ValueError(f"Must pass same number of labels ({len(labels)}) as run_names ({len(run_names)}).")
    if fontsize_label is None:
        fontsize_label = fontsize
    if fontsize_tick_label is None:
        fontsize_tick_label = fontsize
    if fontsize_legend is None:
        fontsize_legend = fontsize
    if fontsize_title is None:
        fontsize_title = fontsize

    if ax is None:
        # Create figure/axes.
        plt.figure(figsize=figsize)
        ax = plt.axes(frameon=False)
        close_figure = True
    else:
        # Assume that parent routine will call 'plt.show()' after
        # all axes are plotted.
        show = False
        close_figure = False

    # Load run data.
    for region in regions:
        for i, run_name in enumerate(run_names):
            # Multiple runs can be specified to average them.
            averaged_runs = arg_to_list(run_name, str)

            # Determine run name.
            if len(averaged_runs) > 1:
                run_name = '/'.join(averaged_runs)

            # Load averaged losses.
            a_losses = []
            a_lrs = []
            for a_run in averaged_runs:
                results = load_data(model_name, a_run)
                lrs = results['lr']
                if skip_end == 0:
                    lrs = lrs[skip_start:]
                else:
                    lrs = lrs[skip_start:-skip_end]
                if len(a_lrs) > 0:
                    if lrs != a_lrs:
                        raise ValueError(f'Averaged runs must be evaluated over same learning rates.')
                a_lrs = lrs
                if region == 'all':
                    losses = results['loss']
                else:
                    losses = results['region-losses'][region]
                if skip_end == 0:
                    losses = losses[skip_start:]
                else:
                    losses = losses[skip_start:-skip_end]
                a_losses.append(losses)
            losses = np.vstack(a_losses).mean(axis=0)

            # Remove 'nan' losses.
            non_nan_idx = np.argwhere(~np.isnan(losses)).flatten()
            lrs = list(np.array(lrs)[non_nan_idx])
            losses = list(np.array(losses)[non_nan_idx])
            
            # Smooth the signal.
            if smooth:
                losses = gaussian_filter1d(losses, smooth_kernel_sd)

            # Remove start/end points.
            if sugg_skip_end != 0:
                sugg_losses  = losses[sugg_skip_start:-sugg_skip_end] 
            else:
                sugg_losses = losses[sugg_skip_start:]

            # Load suggestion.
            min_grad_idx = np.gradient(sugg_losses).argmin()
            min_grad_idx = sugg_skip_start + min_grad_idx
            sugg_lr_val = lrs[min_grad_idx]
            sugg_loss = losses[min_grad_idx]

            # Plot.
            if labels is not None:
                label = labels[i]
            else:
                label = f'{run_name}: {sugg_lr_val:.6f}' if show_sugg_lr else run_name
            if region != 'all':
                label = f"{label} - {region}"
            ax.plot(lrs, losses, label=label)
            if show_sugg_lr:
                ax.scatter(sugg_lr_val, sugg_loss, color='red')

        # Plot vlines.
        if vline is not None:
            vlines = arg_to_list(vline, float)
            for vline in vlines:
                ax.axvline(vline, label='Selected LR', linestyle='--', color='red')

    ax.set_xlabel('Learning rate (LR)', fontsize=fontsize_label)
    ax.set_ylabel('Loss', fontsize=fontsize_label)
    ax.set_xscale('log')

    # Set tick label properties.
    ax.tick_params(axis='both', labelsize=fontsize_tick_label)

    if show_title:
        ax.set_title('LR Find', fontsize=fontsize_title)
    if show_legend:
        ax.legend(bbox_to_anchor=legend_bbox, fontsize=fontsize_legend)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    min_exp = int(np.floor(np.log10(np.min(lrs)))) - 1
    max_exp = int(np.ceil(np.log10(np.max(lrs)))) + 1
    minor_tick_locations = [c * 10 ** e for c in range(1, 10) for e in range(min_exp, max_exp)]
    ax.xaxis.set_minor_locator(FixedLocator(minor_tick_locations))
    ax.grid(axis='both', which='major')
    ax.grid(axis='x', which='minor', linestyle='--')

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show:
        plt.show()

    if close_figure:
        plt.close() 

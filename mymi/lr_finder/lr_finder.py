import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional

from mymi import config

def load_data(
    model_name: str,
    run_name: str) -> Dict[str, np.ndarray]:
    filepath = os.path.join(config.directories.models, model_name, run_name, 'lr-finder.json')
    return json.load(open(filepath))

def suggestion(
    model_name: str,
    run_name: str,
    skip_end: Optional[int] = None,
    skip_start: int = 0,
    smooth: bool = False,
    smooth_kernel_sd: float = 1) -> float:
    # Get loss data.
    results = load_data(model_name, run_name)
    losses = results['loss']

    # Smooth the signal.
    if smooth:
        losses = gaussian_filter1d(losses, smooth_kernel_sd)

    # Remove start/end points.
    if skip_end is not None:
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
    run_name: str,
    skip_end: Optional[int] = None,
    skip_start: int = 0,
    smooth: bool = False,
    smooth_kernel_sd: float = 1) -> None:
    # Load learning rates and losses.
    results = load_data(model_name, run_name)
    lr = results['lr']
    losses = results['loss']
    
    # Smooth the signal.
    if smooth:
        losses = gaussian_filter1d(losses, smooth_kernel_sd)

    # Remove start/end points.
    if skip_end is not None:
        sugg_losses  = losses[skip_start:-skip_end]
    else:
        sugg_losses = losses[skip_start:]

    # Load suggestion.
    min_grad_idx = np.gradient(sugg_losses).argmin()
    min_grad_idx = skip_start + min_grad_idx
    sugg_lr = results['lr'][min_grad_idx]
    sugg_loss = losses[min_grad_idx]

    # Plot.
    plt.plot(lr, losses)
    plt.scatter(sugg_lr, sugg_loss, color='red')
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.xscale('log')
    title = f'Suggested lr={sugg_lr}'
    plt.title(title)
    plt.show()

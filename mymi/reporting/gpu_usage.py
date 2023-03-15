from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import os
import pandas as pd
from time import sleep
from typing import Any, List, Optional, Union

from mymi import config
from mymi import logging
from mymi.utils import append_row, gpu_count, gpu_usage_nvml

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

def record_gpu_usage(
    name: str,
    time: float,
    interval: float) -> None:
    logging.arg_log('Recording GPU usage', ('name', 'time (seconds)', 'interval (seconds)'), (name, time, interval))

    # Create results table.
    cols = {
        'time': str
    }
    for i in range(gpu_count()):
        device_name = f'cuda:{i}'
        cols[f'{device_name}-usage'] = float
    df = pd.DataFrame(columns=cols.keys())

    # Add usage.
    n_intervals = int(np.ceil(time / interval))
    start_time = datetime.now()
    for i in range(n_intervals):
        # Record GPU usage.
        data = {
            'time': (datetime.now() - start_time).total_seconds()
        }
        for j, usage in enumerate(gpu_usage_nvml()):
            device_name = f'cuda:{j}'
            data[f'{device_name}-usage'] = usage
        df = append_row(df, data)

        # Wait for time interval to pass.
        time_passed = (datetime.now() - start_time).total_seconds()
        if time_passed > time:
            break
        time_to_wait = ((i + 1) * interval) - time_passed
        if time_to_wait > 0:
            sleep(time_to_wait)
        elif time_to_wait < 0:
            # Could make the time problem worse by logging this info...
            # logging.warning(f"GPU usage recording took longer than allocated interval '{interval}' (seconds).")
            pass

    # Save results.
    filepath = os.path.join(config.directories.reports, 'gpu-usage', f'{name}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_gpu_usage(name: str) -> pd.DataFrame:
    # Check for 'timing' file - indicates successful training run.
    filepath = os.path.join(config.directories.reports, 'gpu-usage', f'{name}-time.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"Timing file doesn't exist for '{name}'. Training run didn't complete.")
    filepath = os.path.join(config.directories.reports, 'gpu-usage', f'{name}.csv') 
    return pd.read_csv(filepath)

def max_gpu_usage(name: str) -> Union[float, List[float]]:
    # Get max/s.
    df = load_gpu_usage(name)
    n_gpus = len(df.columns) - 1
    maxs = [df[f'cuda:{i}-usage'].max() for i in range(n_gpus)] 
    if len(maxs) == 1:
        return maxs[0]
    else:
        return maxs

def plot_gpu_usage(
    name: str,
    ax: Optional[Axes] = None) -> None:
    show_plot = True if ax is None else False
    if ax is None:
        ax = plt.gca()

    # Plot results.
    df = load_gpu_usage(name)
    x = df['time']
    n_gpus = len(df.columns) - 1
    for i in range(n_gpus):
        device_name = f'cuda:{i}'
        key = f'{device_name}-usage'
        y = df[key]
        ax.plot(x, y, label=key)
        # y = df[f'{device_name}-mem-alloc']
        # ax.plot(x, y, color='g', label='pytorch-alloc')
        # y = df[f'{device_name}-mem-res']
        # ax.plot(x, y, color='b', label='pytorch-res')
    ax.legend()
    ax.set_title(name)
    ax.set_xlabel('time [seconds]')
    ax.set_ylabel('GPU usage [MB]')
    if show_plot:
        plt.show()

from datetime import datetime
import numpy as np
import os
import pandas as pd
from time import sleep

from mymi import config
from mymi import logging
from mymi.utils import append_row, gpu_count, gpu_usage

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
INTERVAL_SECONDS = 0.1

def record_gpu_usage(
    name: str,
    time: int):
    logging.arg_log('Recording GPU usage', ('name', 'time (seconds)'), (name, time))

    # Create results table.
    cols = {
        'datetime': str
    }
    for i in range(gpu_count()):
        cols[f'gpu{i}-usage'] = float
    df = pd.DataFrame(columns=cols.keys())

    # Add usage.
    n_intervals = int(np.ceil(time / INTERVAL_SECONDS))
    start_time = datetime.now()
    for i in range(n_intervals):
        # Record GPU usage.
        data = {
            'datetime': datetime.now().strftime(DATETIME_FORMAT)
        }
        for j, usage in enumerate(gpu_usage()):
            data[f'gpu{j}-usage'] = usage
        df = append_row(df, data)

        # Wait for time interval to pass.
        time_passed = (datetime.now() - start_time).total_seconds()
        time_to_wait = ((i + 1) * INTERVAL_SECONDS) - time_passed
        if time_to_wait > 0:
            sleep(time_to_wait)

    # Save results.
    filepath = os.path.join(config.directories.reports, 'gpu-usage', f'{name}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

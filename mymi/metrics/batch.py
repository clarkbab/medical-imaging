import numpy as np
from typing import *

from mymi import logging

def batch_mean(
    metric: Callable,
    a: np.ndarray,
    b: np.ndarray,
    *metric_args) -> Union[float, Dict[str, float]]:
    assert len(a.shape) == len(b.shape) == 4
    
    if a.shape[0] > a.shape[-1]:
        logging.warning(f"Have you put batch dim in correct axis? Got shape: '{a.shape}'.")

    # Assuming batch dimension is first.
    values = []
    for ai, bi in zip(a, b):
        value = metric(ai, bi, *metric_args)
        values.append(value)

    # Some metrics return dicts. Collapse to mean value by key.
    if isinstance(values[0], dict):
        mean_value = {}
        for v in values:
            for m, mv in v.items():
                if m not in mean_value:
                    mean_value[m] = [mv]
                else:
                    mean_value[m] += [mv]
        for m in mean_value:
            mean_value[m] = np.mean(mean_value[m])
    else:
        mean_value = np.mean(values)

    return mean_value

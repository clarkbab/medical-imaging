import numpy as np
import os
from tqdm import tqdm
from typing import Tuple

from mymi import dataset as ds
from mymi.dataset.other import recreate
from mymi import types

def generate_circles(
    dataset: str,
    size: Tuple[int, int],
    num_samples: str,
    seed: int = 42) -> None:
    # Set random seed for reproducibility.
    np.random.seed(seed)

    # Create dataset.
    set = recreate(dataset)

    for i in tqdm(range(num_samples)):
        # Get sample.
        draw = np.random.uniform(0, np.min(size))
        diameter = int(np.around(draw))
        sample = get_sample(size, diameter)

        # Save sample.
        filepath = os.path.join(set.path, 'data', f'{i}.npz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(filepath, data=sample)

def get_sample(
    size: Tuple[int, int],
    diameter: int) -> np.ndarray:
    centre = tuple(np.floor(np.array(size) / 2).astype(int))
    sample = np.zeros(size, dtype=bool)
    for i in range(size[0]):
        for j in range(size[1]):
            sample[i, j] = _is_circle_pixel((i, j), centre, diameter)
    return sample

def _is_circle_pixel(
    point: types.Point2D,
    centre: types.Point2D,
    diameter: int) -> bool:
    dist = np.sqrt((point[0] - centre[0]) ** 2 + (point[1] - centre[1]) ** 2)
    if dist <= diameter / 2:
        return True
    else:
        return False

import numpy as np
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
from mymi.metrics import dice

def test_dice():
    # Overlapping half-volumes.
    a = _load_asset('label_a')
    b = _load_asset('label_b')
    
    assert dice(a, b) == 1

def _load_asset(filename: str) -> np.ndarray:
    filepath = os.path.join(root_dir, 'test', 'assets', 'metrics', f"{filename}.npz")
    data = np.load(filepath)['data']
    return data

test_dice()

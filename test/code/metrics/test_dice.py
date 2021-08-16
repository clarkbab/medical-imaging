import numpy as np
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
from mymi.metrics import dice

def test_dice():
    # Overlapping half-volumes.
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    a[:5, :, :] = 1
    b[:, :5, :] = 1
    assert dice(a, b) == 0.5

test_dice()

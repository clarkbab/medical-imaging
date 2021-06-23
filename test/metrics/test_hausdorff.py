import numpy as np
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
from mymi.metrics import sitk_hausdorff_distance

def test_sitk_hausdorff_distance():
    # Concentric cubes.
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    a[3:7, 3:7, 3:7] = 1
    b[2:8, 2:8, 2:8] = 1
    spacing = (1, 1, 1)
    assert sitk_hausdorff_distance(a, b, spacing) == np.sqrt(3)

test_sitk_hausdorff_distance()

import nibabel as nib
import numpy as np
from numpy.testing import assert_almost_equal
import os
import sys
from unittest import TestCase

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)
from mymi.metrics import hausdorff_distance, percentile_hausdorff_distance

class TestHausdorff(TestCase):
    def test_hausdorff_distance_patient(self):
        # Overlapping half-volumes.
        a, a_spacing = self._load_asset('label_a')
        b, b_spacing = self._load_asset('label_b')
        assert a_spacing == b_spacing

        hd, avg_hd = hausdorff_distance(a, b, a_spacing)
        assert_almost_equal(hd, 7.109482, decimal=6) 
        # Plastimatch: 7.109482
        assert_almost_equal(avg_hd, 0.312837, decimal=6)          
        # Plastimatch: 0.312982 for average of the average directed HDs - showing that SimpleITK calculates the 
        # average of the directed HDs rather than the max. There is also some small difference in output values.

    def test_hausdorff_distance_sphere(self):
        # Overlapping half-volumes.
        a, a_spacing = self._load_asset('sphere_R_30')
        b, b_spacing = self._load_asset('hollow_sphere_R_30_r_10')
        assert a_spacing == b_spacing

        hd, avg_hd = hausdorff_distance(a, b, a_spacing)
        assert_almost_equal(hd, 10.049875, decimal=6)       
        # Plastimatch: 10.049875 for voxel HD - showing that SimpleITK calculates voxel HD instead of surface HD.
        assert_almost_equal(avg_hd, 0.058864, decimal=6)          
        # Plastimatch: 0.058864 for average of the average directed HDs - showing that SimpleITK calculates the 
        # average of the directed HDs rather than the max. There is also some small difference in output values.

    def test_percentile_hausdorff_distance(self):
        # Overlapping half-volumes.
        p = 100
        a, a_spacing = self._load_asset('sphere_R_30')
        b, b_spacing = self._load_asset('hollow_sphere_R_30_r_10')
        assert a_spacing == b_spacing

        p_hd = percentile_hausdorff_distance(a, b, a_spacing, p)
        print(p_hd)
        assert_almost_equal(p_hd, 1, decimal=6) 

    def _load_asset(
        self,
        filename: str) -> np.ndarray:
        filepath = os.path.join(root_dir, 'test', 'assets', 'metrics', f"{filename}.nii.gz")
        img = nib.load(filepath)
        data = img.get_fdata().astype(bool)
        spacing = (img.affine[0, 0], img.affine[1, 1], img.affine[2, 2]) 
        return data, spacing

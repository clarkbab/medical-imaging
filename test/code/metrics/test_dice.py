import nibabel as nib
import numpy as np
from numpy.testing import assert_almost_equal
import os
import sys
from unittest import TestCase

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)
from mymi.metrics import dice

class TestDice(TestCase):
    def test_dice_patient(self):
        a, _ = self._load_asset('label_a')
        b, _ = self._load_asset('label_b')
        dsc = dice(a, b)
        assert_almost_equal(dsc, 0.810875, decimal=6)
        # Plastimatch: 0.810875

    def test_dice_sphere(self):
        a, _ = self._load_asset('sphere_R_30')
        b, _ = self._load_asset('hollow_sphere_R_30_r_10')
        dsc = dice(a, b)
        assert_almost_equal(dsc, 0.981220, decimal=6)
        # Plastimatch: 0.981220

    def test_dice_cylinder(self):
        a, _ = self._load_asset('cylinder_R_30')
        b, _ = self._load_asset('hollow_cylinder_R_30_r_10')
        dsc = dice(a, b)
        assert_almost_equal(dsc, 0.940200, decimal=6)
        # Plastimatch: 0.940200

    def _load_asset(
        self,
        filename: str) -> np.ndarray:
        filepath = os.path.join(root_dir, 'test', 'assets', 'metrics', f"{filename}.nii.gz")
        img = nib.load(filepath)
        data = img.get_fdata().astype(bool)
        spacing = (img.affine[0, 0], img.affine[1, 1], img.affine[2, 2]) 
        return data, spacing

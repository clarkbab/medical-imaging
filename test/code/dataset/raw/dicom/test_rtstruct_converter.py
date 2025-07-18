import numpy as np
import os
import pydicom as dcm
import sys
from typing import List
import unittest
from unittest import TestCase

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.raw.dicom import ROIData, RtStructConverter
from mymi.regions import to_rgb_255, RegionColours

class TestRtStructConverter(TestCase):
    def test_bidirectional_conversion(self):
        # Load data.
        cts = self._load_cts()
        before = self._load_label()

        # Perform bidirectional conversion.
        rtstruct = RtStructConverter.create_rtstruct(cts)
        roi_data = ROIData(
            colour=list(to_rgb_255(RegionColours.Parotid_L)),
            data=before,
            name='sample'
        )
        RtStructConverter.add_roi_contour(rtstruct, roi_data, cts)
        after = RtStructConverter.get_regions_data(rtstruct, 'sample', cts)

        # Assert that conversion doesn't alter the segmentation.
        np.testing.assert_array_equal(before, after)

    def _load_cts(self) -> List[dcm.dataset.FileDataset]:
        path = os.path.join(root_dir, 'test', 'assets', 'dataset', 'raw', 'dicom', 'ct')
        filepaths = [os.path.join(path, f) for f in os.listdir(path)]
        cts = [dcm.read_file(f) for f in filepaths]
        cts = sorted(cts, key=lambda ct: ct.ImagePositionPatient[2])    # Sort by 'z-position'.
        return cts

    def _load_label(self) -> np.ndarray:
        filepath = os.path.join(root_dir, 'test', 'assets', 'dataset', 'raw', 'dicom', 'label.npz')
        label = np.load(filepath)['data']
        return label

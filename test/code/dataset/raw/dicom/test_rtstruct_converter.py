import numpy as np
import os
import sys
from typing import List
from unittest import TestCase

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.dataset.raw.dicom import ROIData, RTSTRUCTConverter

class TestRTSTRUCTConverter(TestCase):
    def not_test_bidirectional_conversion(self):
        # Load data.
        cts = self._load_cts()
        before = self._load_label()

        # Perform bidirectional conversion.
        cts = []
        rtstruct = RTSTRUCTConverter.create_rtstruct(cts)
        roi_data = ROIData(
            data=before,
            frame_of_reference_uid='UID',
            name='Sample label'
        )
        RTSTRUCTConverter.add_roi(rtstruct, roi_data, cts)
        after = RTSTRUCTConverter.get_roi_data(rtstruct, 'Sphere', cts)

        # Assert that conversion doesn't alter the segmentation.
        np.testing.assert_array_equal(before, after)

    def _load_cts(self) -> List[]

    def _load_label(self) -> np.ndarray:
        filepath = os.path.join(root_dir, 'test', 'assets', 'dataset', 'raw', 'dicom', 'label.npz')
        label = np.load(filepath)['data']
        return label

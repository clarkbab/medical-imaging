import numpy as np
import os
import pydicom as dcm
import sys
from typing import List
from unittest import TestCase

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.dataset.raw.dicom import ROIData, RTSTRUCTConverter
from mymi.regions import to_255, RegionColours

class TestRTSTRUCTConverter(TestCase):
    def test_bidirectional_conversion(self):
        # Load data.
        cts = self._load_cts()
        before = self._load_label()

        # Perform bidirectional conversion.
        rtstruct = RTSTRUCTConverter.create_rtstruct(cts)
        roi_data = ROIData(
            colour=list(to_255(RegionColours.Parotid_L)),
            data=before,
            frame_of_reference_uid='UID',
            name='sample'
        )
        RTSTRUCTConverter.add_roi(rtstruct, roi_data, cts)
        after = RTSTRUCTConverter.get_roi_data(rtstruct, 'sample', cts)

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

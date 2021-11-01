import math
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import List

from mymi import cache
from mymi import config
from mymi import types

from .dicom_series import DICOMModality, DICOMSeries

CLOSENESS_ABS_TOL = 1e-10;

class CTSeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: str):
        self._global_id = f"{study} - {id}"
        self._study = study
        self._id = id
        self._path = os.path.join(study.path, 'ct', id)
        
        # Check that series exists.
        if not os.path.exists(self._path):
            raise ValueError(f"CT series '{self}' not found.")

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.CT

    @property
    def path(self) -> str:
        return self._path

    @property
    def study(self) -> str:
        return self._study
    
    def __str__(self) -> str:
        return self._global_id

    def get_cts(self) -> List[dcm.dataset.FileDataset]:
        # Load CTs.
        ct_paths = [os.path.join(self._path, f) for f in os.listdir(self._path)]
        cts = [dcm.read_file(f) for f in ct_paths]

        # Sort by z-position.
        cts = list(sorted(cts, key=lambda ct: ct.ImagePositionPatient[2]))
        return cts

    def offset(self) -> types.PhysPoint3D:
        cts = self.get_cts()

        # Get offset.
        offset = (
            cts[0].ImagePositionPatient[0],
            cts[0].ImagePositionPatient[1],
            cts[0].ImagePositionPatient[2]
        )
        return offset

    def orientation(self) -> types.ImageSpacing3D:
        cts = self.get_cts()

        # Get the orientation.
        orientation = (
            (
                cts[0].ImageOrientationPatient[0],
                cts[0].ImageOrientationPatient[1],
                cts[0].ImageOrientationPatient[2]
            ),
            (
                cts[0].ImageOrientationPatient[3],
                cts[0].ImageOrientationPatient[4],
                cts[0].ImageOrientationPatient[5]
            )
        )
        return orientation

    def size(self) -> types.ImageSpacing3D:
        cts = self.get_cts()

        # Get size - relies on hierarchy filtering (i.e. removing patients with missing slices).
        size = (
            cts[0].pixel_array.shape[1],
            cts[0].pixel_array.shape[0],
            len(cts)
        )
        return size

    def spacing(self) -> types.ImageSpacing3D:
        cts = self.get_cts()

        # Get spacing - relies on hierarchy filtering (i.e. ensuring consistent voxel spacing).
        spacing = (
            cts[0].PixelSpacing[0],
            cts[0].PixelSpacing[1],
            np.abs(cts[1].ImagePositionPatient[2] - cts[0].ImagePositionPatient[2])
        )
        return spacing

    # @cache.method('_global_id')
    def data(self) -> np.ndarray:
        # Load series CT dicoms.
        cts = self.get_cts()

        # Load CT summary info.
        size = self.size()
        offset = self.offset()
        spacing = self.spacing()
        
        # Create CT data array.
        data = np.zeros(shape=size)
        for ct in cts:
            # Convert to HU. Transpose to (x, y) coordinates, 'pixel_array' returns
            # row-first image data.
            ct_data = np.transpose(ct.pixel_array)
            ct_data = ct.RescaleSlope * ct_data + ct.RescaleIntercept

            # Get z index.
            z_offset =  ct.ImagePositionPatient[2] - offset[2]
            z_idx = int(round(z_offset / spacing[2]))

            # Add data.
            data[:, :, z_idx] = ct_data

        return data

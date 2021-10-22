import math
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import List

from mymi import cache
from mymi import config
from mymi import types

CLOSENESS_ABS_TOL = 1e-10;

class CTSeries:
    def __init__(
        self,
        patient: 'DICOMPatient',
        id: str):
        """
        args:
            patient: the DICOMPatient to which the CT series belongs.
            id: the CT series ID.
        """
        self._global_id = f"{patient} - {id}"
        self._patient = patient
        self._id = id
        self._path = os.path.join(patient.path, 'ct', id)
        
        # Check that series exists.
        if not os.path.exists(self._path):
            raise ValueError(f"CT series '{self}' not found.")

        # Check that DICOMs are present.
        cts = os.listdir(self._path)
        if len(cts) == 0:
            raise ValueError(f"CT series '{self}' empty.")

    def description(self) -> str:
        return self._global_id
    
    def __str__(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    def get_cts(self) -> List[dcm.dataset.FileDataset]:
        # Load CTs.
        ct_paths = [os.path.join(self._path, f) for f in os.listdir(self._path)]
        cts = [dcm.read_file(f) for f in ct_paths]

        # Sort by z-position.
        cts = sorted(cts, key=lambda ct: ct.ImagePositionPatient[2])
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

    @cache.method('_global_id')
    def slice_summary(self) -> pd.DataFrame:
        """
        returns: a table summarising CT slice info.
        """
        # Define dataframe structure.
        cols = {
            'fov-x': float,
            'fov-y': float,
            'hu-max': float,
            'hu-min': float,
            'offset-x': float,
            'offset-y': float,
            'offset-z': float,
            'orientation-row-x': float,
            'orientation-row-y': float,
            'orientation-row-z': float,
            'orientation-col-x': float,
            'orientation-col-y': float,
            'orientation-col-z': float,
            'size-x': int,
            'size-y': int,
            'spacing-x': float,
            'spacing-y': float,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load CT dicoms.
        cts = self.get_cts()

        # Add summary.
        data = {}
        for ct in cts:
            # Add HU stats.
            hus = ct.pixel_array * ct.RescaleSlope + ct.RescaleIntercept
            data['hu-min'] = hus.min()
            data['hu-max'] = hus.max()

            # Add offsets.
            data['offset-x'] = ct.ImagePositionPatient[0]
            data['offset-y'] = ct.ImagePositionPatient[1]
            data['offset-z'] = ct.ImagePositionPatient[2]

            # Add orientation.
            data['orientation-row-x'] = ct.ImageOrientationPatient[0]
            data['orientation-row-y'] = ct.ImageOrientationPatient[1]
            data['orientation-row-z'] = ct.ImageOrientationPatient[2]
            data['orientation-col-x'] = ct.ImageOrientationPatient[3]
            data['orientation-col-y'] = ct.ImageOrientationPatient[4]
            data['orientation-col-z'] = ct.ImageOrientationPatient[5]

            # Add sizes.
            data['size-x'] = ct.pixel_array.shape[1]
            data['size-y'] = ct.pixel_array.shape[0]

            # Add x/y-spacings.
            data['spacing-x'] = ct.PixelSpacing[0]
            data['spacing-y'] = ct.PixelSpacing[1]

            # Add fields-of-view.
            data['fov-x'] = data['size-x'] * data['spacing-x']
            data['fov-y'] = data['size-y'] * data['spacing-y']

            # Add row.
            df = df.append(data, ignore_index=True)

        # Sort columns.
        df = df.reindex(sorted(df.columns), axis=1)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    @cache.method('_global_id')
    def summary(self) -> pd.DataFrame:
        """
        returns: a table summarising CT info.
        """
        # Define dataframe structure.
        cols = {
            'fov-x': float,
            'fov-y': float,
            'fov-z': float,
            'hu-max': float,
            'hu-min': float,
            'offset-x': float,
            'offset-y': float,
            'offset-z': float,
            'orientation-row-x': float,
            'orientation-row-y': float,
            'orientation-row-z': float,
            'orientation-col-x': float,
            'orientation-col-y': float,
            'orientation-col-z': float,
            'size-x': int,
            'size-y': int,
            'size-z': int,
            'spacing-x': float,
            'spacing-y': float,
            'spacing-z': float
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load CT dicoms.
        cts = self.get_cts()

        # Add summary.
        data = {}
        z_offsets = []
        for ct in cts:
            # Add HU stats.
            hus = ct.pixel_array * ct.RescaleSlope + ct.RescaleIntercept
            hu_min = hus.min()
            hu_max = hus.max()
            if 'hu-min' not in data or hu_min < data['hu-min']:
                data['hu-min'] = hu_min
            if 'hu-max' not in data or hu_max > data['hu-max']:
                data['hu-max'] = hu_max

            # Add offsets.
            x_offset = ct.ImagePositionPatient[0]
            y_offset = ct.ImagePositionPatient[1]
            z_offset = ct.ImagePositionPatient[2]
            z_offsets.append(z_offset)
            if 'offset-x' not in data:
                data['offset-x'] = x_offset
            elif x_offset != data['offset-x']:
                raise ValueError(f"Inconsistent 'offset-x' for CT series '{self}'.")
            if 'offset-y' not in data:
                data['offset-y'] = y_offset
            elif y_offset != data['offset-y']:
                raise ValueError(f"Inconsistent 'offset-y' for CT series '{self}'.")
            if 'offset-z' not in data or z_offset < data['offset-z']:
                data['offset-z'] = z_offset

            # Add orientations.
            row_x_orientation = ct.ImageOrientationPatient[0]
            row_y_orientation = ct.ImageOrientationPatient[1]
            row_z_orientation = ct.ImageOrientationPatient[2]
            col_x_orientation = ct.ImageOrientationPatient[3]
            col_y_orientation = ct.ImageOrientationPatient[4]
            col_z_orientation = ct.ImageOrientationPatient[5]
            if 'orientation-row-x' not in data:
                data['orientation-row-x'] = row_x_orientation
            elif not math.isclose(row_x_orientation, 1, abs_tol=CLOSENESS_ABS_TOL):
                raise ValueError(f"Patient 'orientation-row-x' not standard for CT series '{self}'.")
            if 'orientation-row-y' not in data:
                data['orientation-row-y'] = row_y_orientation
            elif not math.isclose(row_y_orientation, 0, abs_tol=CLOSENESS_ABS_TOL):
                raise ValueError(f"Patient 'orientation-row-y' not standard for CT series '{self}'.")
            if 'orientation-row-z' not in data:
                data['orientation-row-z'] = row_z_orientation
            elif not math.isclose(row_z_orientation, 0, abs_tol=CLOSENESS_ABS_TOL):
                raise ValueError(f"Patient 'orientation-row-z' not standard for CT series '{self}'.")
            if 'orientation-col-x' not in data:
                data['orientation-col-x'] = col_x_orientation
            elif not math.isclose(col_x_orientation, 0, abs_tol=CLOSENESS_ABS_TOL):
                raise ValueError(f"Patient 'orientation-col-x' not standard for CT series '{self}'.")
            if 'orientation-col-y' not in data:
                data['orientation-col-y'] = col_y_orientation
            elif not math.isclose(col_y_orientation, 1, abs_tol=CLOSENESS_ABS_TOL):
                raise ValueError(f"Patient 'orientation-col-y' not standard for CT series '{self}'.")
            if 'orientation-col-z' not in data:
                data['orientation-col-z'] = col_z_orientation
            elif not math.isclose(col_z_orientation, 0, abs_tol=CLOSENESS_ABS_TOL):
                raise ValueError(f"Patient 'orientation-col-z' not standard for CT series '{self}'.")

            # Add sizes.
            x_size = ct.pixel_array.shape[1]
            y_size = ct.pixel_array.shape[0]
            if 'size-x' not in data:
                data['size-x'] = x_size
            elif x_size != data['size-x']:
                raise ValueError(f"Inconsistent 'size-x' for CT series '{self}'.")
            if 'size-y' not in data:
                data['size-y'] = y_size
            elif y_size != data['size-y']:
                raise ValueError(f"Inconsistent 'size-y' for CT series '{self}'.")

            # Add x/y-spacings.
            x_spacing = ct.PixelSpacing[0]
            y_spacing = ct.PixelSpacing[1]
            if 'spacing-x' not in data:
                data['spacing-x'] = x_spacing
            elif x_spacing != data['spacing-x']:
                raise ValueError(f"Inconsistent 'spacing-x' for CT series '{self}'.")
            if 'spacing-y' not in data:
                data['spacing-y'] = y_spacing
            elif y_spacing != data['spacing-y']:
                raise ValueError(f"Inconsistent 'spacing-y' for CT series '{self}'.")

        # Add z-spacing. Round z-spacings to 3 d.p. as some of the diffs are whacky like 2.99999809.
        z_spacings = np.unique([round(s, 3) for s in np.diff(sorted(z_offsets))])
        if len(z_spacings) != 1:
            raise ValueError(f"Expected single 'spacing-z', got spacings '{z_spacings}' for CT series '{self}'.")
        z_spacing = z_spacings[0]
        data['spacing-z'] = z_spacing

        # Add fields-of-view.
        x_fov = data['size-x'] * data['spacing-x']
        y_fov = data['size-y'] * data['spacing-y']
        z_fov = np.max(z_offsets) - np.min(z_offsets) + z_spacing   # Assume that FOV includes half-voxel at either end.
        data['fov-x'] = x_fov
        data['fov-y'] = y_fov
        data['fov-z'] = z_fov

        # Add z-size.
        z_size = int(round(z_fov / z_spacing, 0))
        data['size-z'] = z_size

        # Check for missing slices.
        # TODO: Handle missing slices.
        num_missing = z_size - len(cts)
        if num_missing != 0:
            raise ValueError(f"{num_missing} missing slices for CT series '{self}'.")

        # Add row.
        df = df.append(data, ignore_index=True)
        df = df.reindex(sorted(df.columns), axis=1)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    @cache.method('_global_id')
    def data(self) -> np.ndarray:
        """
        returns: a 3D numpy ndarray of CT data in HU.
        """
        # Load patient CT dicoms.
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

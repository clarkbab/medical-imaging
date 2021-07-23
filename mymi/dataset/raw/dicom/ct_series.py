import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import List

from mymi import cache
from mymi import config
from mymi import types

class CTSeries:
    def __init__(
        self,
        dataset: str,
        pat_id: types.PatientID,
        id: str):
        """
        args:
            dataset: the dataset name.
            id: the CT series ID.
        """
        self._dataset = dataset
        self._pat_id = pat_id
        self._id = id
        self._path = os.path.join(config.directories.datasets, 'raw', dataset, 'hierarchical', 'data', pat_id, 'ct', id)
        
        # Check that series exists.
        if not os.path.exists(self._path):
            raise ValueError(f"CT series '{id}' not found for patient '{pat_id}', dataset '{dataset}'.")

        # Check that DICOMs are present.
        cts = os.listdir(self._path)
        if len(cts) == 0:
            raise ValueError(f"CT series '{id}' empty for patient '{pat_id}', dataset '{dataset}'.")

    @property
    def id(self) -> str:
        return self._id

    def get_cts(self) -> List[dcm.dataset.FileDataset]:
        """
        returns: a list of CT DICOM objects.
        """
        # Load CTs.
        ct_paths = [os.path.join(self._path, f) for f in os.listdir(self._path)]
        cts = [dcm.read_file(f) for f in ct_paths]

        # Sort by z-position.
        cts = sorted(cts, key=lambda ct: ct.ImagePositionPatient[2])

        return cts

    def offset(
        self,
        clear_cache: bool = False) -> types.PhysPoint3D:
        """
        returns: the patient offset in physical coordinates.
        kwargs:
            clear_cache: forces the cache to clear.
        """
        # Get the offset.
        offset = tuple(self.summary(clear_cache=clear_cache)[['offset-x', 'offset-y', 'offset-z']].iloc[0])
        return offset

    def size(
        self,
        clear_cache: bool = False) -> types.ImageSpacing3D:
        """
        returns: the CT scan size in physical coordinates.
        kwargs:
            clear_cache: forces the cache to clear.
        """
        # Get the spacing.
        size = tuple(self.summary(clear_cache=clear_cache)[['size-x', 'size-y', 'size-z']].iloc[0])
        return size

    def spacing(
        self,
        clear_cache: bool = False) -> types.ImageSpacing3D:
        """
        returns: the patient spacing in physical coordinates.
        kwargs:
            clear_cache: forces the cache to clear.
        """
        # Get the spacing.
        spacing = tuple(self.summary(clear_cache=clear_cache)[['spacing-x', 'spacing-y', 'spacing-z']].iloc[0])
        return spacing

    @cache.method('_dataset', '_pat_id', '_id')
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

    @cache.method('_dataset', '_pat_id', '_id')
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
            'num-missing': int,
            'offset-x': float,
            'offset-y': float,
            'offset-z': float,
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
                raise ValueError(f"Inconsistent CT 'offset-x' for dataset '{self._dataset}', patient '{self._id}'.")
            if 'offset-y' not in data:
                data['offset-y'] = y_offset
            elif y_offset != data['offset-y']:
                raise ValueError(f"Inconsistent CT 'offset-y' for dataset '{self._dataset}', patient '{self._id}'.")
            if 'offset-z' not in data or z_offset < data['offset-z']:
                data['offset-z'] = z_offset

            # Add sizes.
            x_size = ct.pixel_array.shape[1]
            y_size = ct.pixel_array.shape[0]
            if 'size-x' not in data:
                data['size-x'] = x_size
            elif x_size != data['size-x']:
                raise ValueError(f"Inconsistent CT 'size-x' for dataset '{self._dataset}', patient '{self._id}'.")
            if 'size-y' not in data:
                data['size-y'] = y_size
            elif y_size != data['size-y']:
                raise ValueError(f"Inconsistent CT 'size-y' for dataset '{self._dataset}', patient '{self._id}'.")

            # Add x/y-spacings.
            x_spacing = ct.PixelSpacing[0]
            y_spacing = ct.PixelSpacing[1]
            if 'spacing-x' not in data:
                data['spacing-x'] = x_spacing
            elif x_spacing != data['spacing-x']:
                raise ValueError(f"Inconsistent CT 'spacing-x' for dataset '{self._dataset}', patient '{self._id}'.")
            if 'spacing-y' not in data:
                data['spacing-y'] = y_spacing
            elif y_spacing != data['spacing-y']:
                raise ValueError(f"Inconsistent CT 'spacing-y' for dataset '{self._dataset}', patient '{self._id}'.")

        # Add z-spacing. Round z-spacings to 3 d.p. as some of the diffs are whacky like 2.99999809.
        z_spacing = np.min([round(s, 3) for s in np.diff(sorted(z_offsets))])
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

        # Add num missing slices.
        data['num-missing'] = z_size - len(cts)

        # Add row.
        df = df.append(data, ignore_index=True)
        df = df.reindex(sorted(df.columns), axis=1)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    @cache.method('_dataset', '_pat_id', '_id')
    def data(
        self,
        clear_cache: bool = False) -> np.ndarray:
        """
        returns: a 3D numpy ndarray of CT data in HU.
        kwargs:
            clear_cache: force the cache to clear.
        """
        # Load patient CT dicoms.
        cts = self.get_cts()

        # Load CT summary info.
        size = self.size(clear_cache=True)
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

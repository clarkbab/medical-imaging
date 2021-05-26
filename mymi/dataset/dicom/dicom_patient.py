import numpy as np
import os
import pandas as pd
import pydicom as dcm
from scipy.ndimage import center_of_mass
from typing import *
from rt_utils import RTStructBuilder

from mymi import cache
from mymi.cache import cached_method
from mymi import config
from mymi import logging

from .rtstruct_converter import RTStructConverter

class DicomPatient:
    def __init__(
        self,
        dataset: str,
        id: str,
        ct_from: str = None):
        """
        args:
            dataset: the dataset the patient belongs to, e.g. 'HEAD-NECK-RADIOMICS-HN1'.
            id: the patient ID.
        """
        self._ct_from = ct_from
        self._dataset = dataset
        self._id = id
        self._path = os.path.join(config.directories.datasets, dataset, 'hierarchical', id)

    def _require_ct(fn: Callable) -> Callable:
        """
        returns: a wrapped function that ensures CTs are present.
        args:
            fn: the function to wrap.
        """
        def wrapper(self, *args, **kwargs):
            # Pass query to alternate dataset if required.
            if self._ct_from:
                alt_patient = DicomPatient(self._ct_from, self._id)
                alt_fn = getattr(alt_patient, fn.__name__)
                fn_def = getattr(type(self), fn.__name__)
                if type(fn_def) == property:
                    return alt_fn
                else:
                    return alt_fn()

            # Check CT folder exists.
            cts_path = os.path.join(self._path, 'ct')
            if not os.path.exists(cts_path):
                raise ValueError(f"No CTs found for dataset '{self._dataset}', patient '{self._id}'.")

            # Check that there is at least one CT.
            ct_files = os.listdir(cts_path)
            if len(ct_files) == 0:
                raise ValueError(f"No CTs found for dataset '{self._dataset}', patient '{self._id}'.")
            
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    def ct_from(self) -> str:
        return self._ct_from

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def id(self) -> str:
        return self._id

    @property
    @_require_ct
    def name(self) -> str:
        """
        returns: the patient name.
        """
        # Get patient name.
        cts_path = os.path.join(self._path, 'ct')
        ct_path = os.path.join(cts_path, os.listdir(cts_path)[0])
        ct = dcm.read_file(ct_path)
        name = ct.PatientName
        return name

    @_require_ct
    def get_cts(self) -> Sequence[dcm.dataset.FileDataset]:
        """
        returns: a list of FileDataset objects holding CT info.
        """
        # Load CT dicoms.
        cts_path = os.path.join(self._path, 'ct')
        ct_paths = [os.path.join(cts_path, f) for f in os.listdir(cts_path)]
        cts = [dcm.read_file(f) for f in ct_paths]

        # Sort by z-position.
        cts = sorted(cts, key=lambda ct: ct.ImagePositionPatient[2])

        return cts

    def get_rtstruct(self) -> dcm.dataset.FileDataset:
        """
        returns: a FileDataset object holding RTSTRUCT info.
        """
        # Check number of RTSTRUCTs.
        rtstructs_path = os.path.join(self._path, 'rtstruct')
        rtstruct_paths = [os.path.join(rtstructs_path, f) for f in os.listdir(rtstructs_path)]
        if len(rtstruct_paths) != 1:
            raise ValueError(f"Expected 1 RTSTRUCT dicom for dataset '{self._dataset}', patient '{self._id}', got {len(rtstruct_paths)}.")

        # Load RTSTRUCT.
        rtstruct = dcm.read_file(rtstruct_paths[0])

        return rtstruct

    @_require_ct
    def info(
        self,
        clear_cache: bool = False) -> pd.DataFrame:
        """
        returns: a table of patient info.
        """
        # Define dataframe structure.
        cols = {
            'name': str,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Add data.
        data = {}
        data['name'] = self.name

        # Add row.
        df = df.append(data, ignore_index=True)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    @cached_method('_ct_from', '_dataset', '_id')
    def ct_summary(self) -> pd.DataFrame:
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
                raise ValueError(f"Inconsistent 'offset-x' for dataset '{self._dataset}', patient '{self._id}' CT scans.")
            if 'offset-y' not in data:
                data['offset-y'] = y_offset
            elif y_offset != data['offset-y']:
                raise ValueError(f"Inconsistent 'offset-y' for dataset '{self._dataset}', patient '{self._id}' CT scans.")
            if 'offset-z' not in data or z_offset < data['offset-z']:
                data['offset-z'] = z_offset

            # Add sizes.
            x_size = ct.pixel_array.shape[1]
            y_size = ct.pixel_array.shape[0]
            if 'size-x' not in data:
                data['size-x'] = x_size
            elif x_size != data['size-x']:
                raise ValueError(f"Inconsistent 'size-x' for dataset '{self._dataset}', patient '{self._id}' CT scans.")
            if 'size-y' not in data:
                data['size-y'] = y_size
            elif y_size != data['size-y']:
                raise ValueError(f"Inconsistent 'size-y' for dataset '{self._dataset}', patient '{self._id}' CT scans.")

            # Add x/y-spacings.
            x_spacing = ct.PixelSpacing[0]
            y_spacing = ct.PixelSpacing[1]
            if 'spacing-x' not in data:
                data['spacing-x'] = x_spacing
            elif x_spacing != data['spacing-x']:
                raise ValueError(f"Inconsistent 'spacing-x' for dataset '{self._dataset}', patient '{self._id}' CT scans.")
            if 'spacing-y' not in data:
                data['spacing-y'] = y_spacing
            elif y_spacing != data['spacing-y']:
                raise ValueError(f"Inconsistent 'spacing-y' for dataset '{self._dataset}', patient '{self._id}' CT scans.")

        # Add z-spacing.
        z_spacing = np.min([round(s, 2) for s in np.diff(sorted(z_offsets))])
        data['spacing-z'] = z_spacing

        # Add fields-of-view.
        x_fov = data['size-x'] * data['spacing-x']
        y_fov = data['size-y'] * data['spacing-y']
        z_fov = np.max(z_offsets) - np.min(z_offsets) + z_spacing   # Assume that FOV includes half-voxel at either end.
        data['fov-x'] = x_fov
        data['fov-y'] = y_fov
        data['fov-z'] = z_fov

        # Add z-size.
        z_size = int(round(z_fov / z_spacing, 0) + 1)
        data['size-z'] = z_size

        # Add num missing slices.
        data['num-missing'] = z_size - len(cts)

        # Add row.
        df = df.append(data, ignore_index=True)
        df = df.reindex(sorted(df.columns), axis=1)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    @cached_method('_ct_from', '_dataset', '_id')
    def ct_slice_summary(self) -> pd.DataFrame:
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

    @cached_method('_ct_from', '_dataset', '_id')
    def label_summary(
        self,
        clear_cache: bool = False,
        columns: Union[str, Sequence[str]] = 'all',
        labels: Union[str, Sequence[str]] = 'all') -> pd.DataFrame:
        """
        returns: a DataFrame label summary information.
        kwargs:
            clear_cache: clear the cache.
            labels: the desired labels.
        """
        # Define table structure.
        cols = {
            'label': str,
            'width-mm-x': float,
            'width-mm-y': float,
            'width-mm-z': float,
        }
        cols = dict(filter(self._filterOnDictKeys(columns), cols.items()))
        df = pd.DataFrame(columns=cols.keys())

        # Get label dict.
        label_data = self.label_data(clear_cache=clear_cache, labels=labels)

        # Get voxel spacings.
        summary = self.ct_summary(clear_cache=clear_cache).iloc[0].to_dict()
        spacing = (summary['spacing-x'], summary['spacing-y'], summary['spacing-z'])

        # Add info for each label.
        for name, data in label_data.items():
            # Find centre-of-mass.
            coms = np.round(center_of_mass(data)).astype(int)

            # Find bounding box co-ordinates.
            non_zero = np.argwhere(data != 0)
            mins = non_zero.min(axis=0)
            maxs = non_zero.max(axis=0)
            voxel_widths = maxs - mins

            # Convert voxel widths to millimetres.
            mm_widths = voxel_widths * spacing

            data = {
                'label': name,
                'width-mm-x': mm_widths[0],
                'width-mm-y': mm_widths[1],
                'width-mm-z': mm_widths[2]
            }
            df = df.append(data, ignore_index=True)

        # Set column type.
        df = df.astype(cols)

        # Sort by label.
        df = df.sort_values('label').reset_index(drop=True)

        return df

    @cached_method('_ct_from', '_dataset', '_id')
    def ct_data(self) -> np.ndarray:
        """
        returns: a 3D numpy ndarray of CT data in HU.
        kwargs:
            clear_cache: force the cache to clear.
        """
        # Load patient CT dicoms.
        cts = self.get_cts()
        summary = self.ct_summary().iloc[0].to_dict()
        
        # Create CT data array.
        shape = (int(summary['size-x']), int(summary['size-y']), int(summary['size-z']))
        data = np.zeros(shape=shape)
        for ct in cts:
            # Convert to HU. Transpose to (x, y) coordinates, 'pixel_array' returns
            # row-fist image data.
            ct_data = np.transpose(ct.pixel_array)
            ct_data = ct.RescaleSlope * ct_data + ct.RescaleIntercept

            # Get z index.
            z_offset =  ct.ImagePositionPatient[2] - summary['offset-z']
            z_idx = int(round(z_offset / summary['spacing-z']))

            # Add data.
            data[:, :, z_idx] = ct_data

        return data

    @cached_method('_dataset', '_id')
    def label_names(
        self,
        clear_cache: bool = False) -> pd.DataFrame:
        """
        returns: the patient's label names.
        kwargs:
            clear_cache: force the cache to clear.
        """
        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Get region names.
        names = list(sorted([r.ROIName for r in rtstruct.StructureSetROISequence]))

        # Create dataframe.
        df = pd.DataFrame(names, columns=['label'])

        return df

    @cached_method('_ct_from', '_dataset', '_id')
    def label_data(
        self,
        clear_cache: bool = False,
        labels: Union[str, Sequence[str]] = 'all') -> dict:
        """
        returns: a Dict[str, np.ndarray] of label names and data.
        kwargs:
            clear_cache: force the cache to clear.
            labels: the desired labels.
        """
        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Load ROI names.
        roi_names = RTStructConverter.get_roi_names(rtstruct) 

        # Filter on required labels.
        def fn(name):
            if ((type(labels) == str and (labels == 'all' or name == labels)) or
                ((type(labels) == tuple or type(labels) == list or type(labels) == np.ndarray) and name in labels)):
                return True
            else:
                return False
        roi_names = list(filter(fn, roi_names))

        # Get offset, shape and spacing.
        summary_df = self.ct_summary(clear_cache=clear_cache)
        offset = tuple(summary_df[['offset-x', 'offset-y', 'offset-z']].iloc[0])
        shape = tuple(summary_df[['size-x', 'size-y', 'size-z']].iloc[0])
        spacing = tuple(summary_df[['spacing-x', 'spacing-y', 'spacing-z']].iloc[0])

        # Add ROI data.
        label_dict = {}
        for name in roi_names:
            data = RTStructConverter.get_roi_mask(name, offset, rtstruct, shape, spacing)
            label_dict[name] = data

        return label_dict

    def _filterOnDictKeys(
        self,
        keys: Union[str, Sequence[str]] = 'all') -> Callable[[Tuple[str, Any]], bool]:
        """
        returns: a function that filters out unneeded keys.
        args:
            keys: description of required keys.
        """
        def fn(item: Tuple[str, Any]) -> bool:
            key, _ = item
            if ((isinstance(keys, str) and (keys == 'all' or key == keys)) or
                ((isinstance(keys, list) or isinstance(keys, np.ndarray) or isinstance(keys, tuple)) and key in keys)):
                return True
            else:
                return False
        return fn

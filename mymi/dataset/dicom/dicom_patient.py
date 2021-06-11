from collections import OrderedDict
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from scipy.ndimage import center_of_mass
from typing import *

from mymi.cache import cached_method
from mymi import config
from mymi import types

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

    def _use_internal_regions(fn: Callable) -> Callable:
        """
        returns: a wrapped function that renames DataFrame 'regions' to internal names.
        args:
            fn: the function to wrap.
        """
        def wrapper(self, *args, **kwargs):
            # Determine if internal region names are required.
            use_internal = kwargs.pop('internal_regions', False)

            # Call function.
            result = fn(self, *args, **kwargs)

            if use_internal:
                # Load region map.
                pass 
            else:
                return result

        return wrapper

    @property
    @_require_ct
    def age(self) -> str:
        return getattr(self.get_cts()[0], 'PatientAge', '')

    @property
    @_require_ct
    def birth_date(self) -> str:
        return self.get_cts()[0].PatientBirthDate

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
        return self.get_cts()[0].PatientName

    @property
    @_require_ct
    def sex(self) -> str:
        return self.get_cts()[0].PatientSex

    @property
    @_require_ct
    def size(self) -> str:
        return getattr(self.get_cts()[0], 'PatientSize', '')

    @property
    @_require_ct
    def weight(self) -> str:
        return getattr(self.get_cts()[0], 'PatientWeight', '')

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
            'age': str,
            'birth-date': str,
            'name': str,
            'sex': str,
            'size': str,
            'weight': str
        }
        df = pd.DataFrame(columns=cols.keys())

        # Add data.
        data = {}
        for col in cols.keys():
            col_method = col.replace('-', '_')
            data[col] = getattr(self, col_method)

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

    def ct_offset(
        self,
        clear_cache: bool = False) -> types.Point3D:
        """
        returns: the patient offset in physical coordinates.
        kwargs:
            clear_cache: forces the cache to clear.
        """
        # Get the offset.
        offset = tuple(self.ct_summary(clear_cache=clear_cache)[['offset-x', 'offset-y', 'offset-z']].iloc[0])
        return offset

    def ct_size(
        self,
        clear_cache: bool = False) -> types.Spacing3D:
        """
        returns: the CT scan size in physical coordinates.
        kwargs:
            clear_cache: forces the cache to clear.
        """
        # Get the spacing.
        spacing = tuple(self.ct_summary(clear_cache=clear_cache)[['size-x', 'size-y', 'size-z']].iloc[0])
        return spacing

    def ct_spacing(
        self,
        clear_cache: bool = False) -> types.Spacing3D:
        """
        returns: the patient spacing in physical coordinates.
        kwargs:
            clear_cache: forces the cache to clear.
        """
        # Get the spacing.
        spacing = tuple(self.ct_summary(clear_cache=clear_cache)[['spacing-x', 'spacing-y', 'spacing-z']].iloc[0])
        return spacing

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
    def region_summary(
        self,
        clear_cache: bool = False,
        columns: Union[str, Sequence[str]] = 'all',
        regions: Union[str, Sequence[str]] = 'all') -> pd.DataFrame:
        """
        returns: a DataFrame region summary information.
        kwargs:
            clear_cache: clear the cache.
            regions: the desired regions.
        """
        # Define table structure.
        cols = {
            'region': str,
            'width-mm-x': float,
            'width-mm-y': float,
            'width-mm-z': float,
        }
        cols = dict(filter(self._filterOnDictKeys(columns), cols.items()))
        df = pd.DataFrame(columns=cols.keys())

        # Get region dict.
        region_data = self.region_data(clear_cache=clear_cache, regions=regions)

        # Get voxel spacings.
        summary = self.ct_summary(clear_cache=clear_cache).iloc[0].to_dict()
        spacing = (summary['spacing-x'], summary['spacing-y'], summary['spacing-z'])

        # Add info for each region.
        for name, data in region_data.items():
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
                'region': name,
                'width-mm-x': mm_widths[0],
                'width-mm-y': mm_widths[1],
                'width-mm-z': mm_widths[2]
            }
            df = df.append(data, ignore_index=True)

        # Set column type.
        df = df.astype(cols)

        # Sort by region.
        df = df.sort_values('region').reset_index(drop=True)

        return df

    @cached_method('_ct_from', '_dataset', '_id')
    def ct_data(
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
        size = self.ct_size(clear_cache=True)
        offset = self.ct_offset()
        spacing = self.ct_spacing()
        
        # Create CT data array.
        data = np.zeros(shape=size)
        for ct in cts:
            # Convert to HU. Transpose to (x, y) coordinates, 'pixel_array' returns
            # row-fist image data.
            ct_data = np.transpose(ct.pixel_array)
            ct_data = ct.RescaleSlope * ct_data + ct.RescaleIntercept

            # Get z index.
            z_offset =  ct.ImagePositionPatient[2] - offset[2]
            z_idx = int(round(z_offset / spacing[2]))

            # Add data.
            data[:, :, z_idx] = ct_data

        return data

    @cached_method('_dataset', '_id')
    def region_names(
        self,
        clear_cache: bool = False) -> pd.DataFrame:
        """
        returns: the patient's region names.
        kwargs:
            clear_cache: force the cache to clear.
        """
        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Get region names.
        names = list(sorted([r.ROIName for r in rtstruct.StructureSetROISequence]))

        # Create dataframe.
        df = pd.DataFrame(names, columns=['region'])

        return df

    @cached_method('_ct_from', '_dataset', '_id')
    def region_data(
        self,
        clear_cache: bool = False,
        regions: Union[str, Sequence[str]] = 'all') -> OrderedDict:
        """
        returns: an OrderedDict[str, np.ndarray] of region names and data.
        kwargs:
            clear_cache: force the cache to clear.
            regions: the desired regions.
        """
        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Load ROI names.
        roi_names = RTStructConverter.get_roi_names(rtstruct) 

        # Filter on required regions.
        def fn(name):
            if ((type(regions) == str and (regions == 'all' or name == regions)) or
                ((type(regions) == tuple or type(regions) == list or type(regions) == np.ndarray) and name in regions)):
                return True
            else:
                return False
        roi_names = list(filter(fn, roi_names))

        # Get reference CTs.
        cts = self.get_cts()

        # Add ROI data.
        region_dict = {}
        for name in roi_names:
            # Get binary mask.
            data = RTStructConverter.get_roi_data(rtstruct, name, cts)
            region_dict[name] = data

        # Create ordered dict.
        ordered_dict = OrderedDict((n, region_dict[n]) for n in sorted(roi_names)) 

        return ordered_dict

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

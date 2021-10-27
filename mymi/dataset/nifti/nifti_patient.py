import nibabel as nib
import numpy as np
import os
import pandas as pd
from scipy.ndimage import center_of_mass
from typing import Any, Callable, List, OrderedDict, Tuple, Union

from mymi import cache
from mymi import config
from mymi.regions import is_region
from mymi import types

class NIFTIPatient:
    def __init__(
        self,
        dataset: 'NIFTIDataset',
        id: str):
        self._dataset = dataset
        self._id = id

    def list_regions(
        self,
        whitelist: types.PatientRegions = 'all') -> List[str]:
        path = os.path.join(self._dataset.path, 'data')
        files = os.listdir(path)
        names = []
        for f in files:
            if not is_region(f):
                continue
            region_path = os.path.join(self._dataset.path, 'data', f)
            for r in os.listdir(region_path):
                id = r.replace('.nii.gz', '')
                if id == self._id:
                    names.append(f)

        # Filter on whitelist.
        def filter_fn(region):
            if isinstance(whitelist, str):
                if whitelist == 'all':
                    return True
                else:
                    return region == whitelist
            else:
                if region in whitelist:
                    return True
                else:
                    return False
        names = list(filter(filter_fn, names))

        return names

    def has_region(
        self,
        region: str) -> bool:
        return region in self.list_regions()

    def ct_spacing(self) -> types.ImageSpacing3D:
        path = os.path.join(self._dataset.path, 'data', 'ct', f"{self._id}.nii.gz")
        img = nib.load(path)
        affine = img.affine
        spacing = (abs(affine[0][0]), abs(affine[1][1]), abs(affine[2][2]))
        return spacing

    def ct_offset(self) -> types.Point3D:
        path = os.path.join(self._dataset.path, 'data', 'ct', f"{self._id}.nii.gz")
        img = nib.load(path)
        affine = img.affine
        offset = (affine[0][3], affine[1][3], affine[2][3])
        return offset

    def ct_data(self) -> np.ndarray:
        path = os.path.join(self._dataset.path, 'data', 'ct', f"{self._id}.nii.gz")
        img = nib.load(path)
        data = img.get_data()
        return data

    def ct_size(self) -> np.ndarray:
        return self.ct_data().shape

    def region_data(
        self,
        regions: types.PatientRegions = 'all') -> OrderedDict:
        # Convert regions to list.
        if type(regions) == str:
            if regions == 'all':
                regions = self.list_regions()
            else:
                regions = [regions]

        data = {}
        for region in regions:
            if not is_region(region):
                raise ValueError(f"Requested region '{region}' not a valid internal region.")
            if not self.has_region(region):
                raise ValueError(f"Requested region '{region}' not found for patient '{self._id}', dataset '{self._dataset}'.")
            
            path = os.path.join(self._dataset.path, 'data', region, f"{self._id}.nii.gz")
            img = nib.load(path)
            rdata = img.get_fdata()
            data[region] = rdata
        return data

    @cache.method('_dataset', '_id')
    def ct_summary(self) -> pd.DataFrame:
        cols = {
            'fov-x': float,
            'fov-y': float,
            'fov-z': float,
            'hu-max': float,
            'hu-min': float,
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
        data = {}
        data['hu-min'] = self.ct_data().min()
        data['hu-max'] = self.ct_data().max()
        data['offset-x'] = self.ct_offset()[0]
        data['offset-y'] = self.ct_offset()[1]
        data['offset-z'] = self.ct_offset()[2]
        data['size-x'] = self.ct_data().shape[0]
        data['size-y'] = self.ct_data().shape[1]
        data['size-z'] = self.ct_data().shape[2]
        data['spacing-x'] = self.ct_spacing()[0]
        data['spacing-y'] = self.ct_spacing()[1]
        data['spacing-z'] = self.ct_spacing()[2]
        data['fov-x'] = data['size-x'] * data['spacing-x']
        data['fov-y'] = data['size-y'] * data['spacing-y']
        data['fov-z'] = data['size-z'] * data['spacing-z']

        df = df.append(data, ignore_index=True)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    @cache.method('_dataset', '_id')
    def region_summary(
        self,
        clear_cache: bool = False,
        columns: Union[str, List[str]] = 'all',
        regions: types.PatientRegions = 'all',
        use_mapping: bool = True) -> pd.DataFrame:
        """
        returns: a DataFrame region summary information.
        kwargs:
            clear_cache: clear the cache.
            columns: the columns to return.
            regions: the desired regions.
            use_mapping: use region map if present.
        """
        # Define table structure.
        cols = {
            'region': str,
            'centroid-mm-x': float,
            'centroid-mm-y': float,
            'centroid-mm-z': float,
            'centroid-voxels-x': int,
            'centroid-voxels-y': int,
            'centroid-voxels-z': int,
            'width-mm-x': float,
            'width-mm-y': float,
            'width-mm-z': float,
            'width-voxels-x': int,
            'width-voxels-y': int,
            'width-voxels-z': int,
        }
        cols = dict(filter(self._filter_on_dict_keys(columns, whitelist='region'), cols.items()))
        df = pd.DataFrame(columns=cols.keys())

        # Get region dict.
        region_data = self.region_data(regions=regions)

        # Get voxel offset/spacing.
        offset = self.ct_offset()
        spacing = self.ct_spacing()

        # Add info for each region.
        for name, data in region_data.items():
            # Find centre-of-mass.
            centroid = np.round(center_of_mass(data)).astype(int)

            # Convert COM to millimetres.
            mm_centroid = (centroid * spacing) + offset

            # Find bounding box co-ordinates.
            non_zero = np.argwhere(data != 0)
            mins = non_zero.min(axis=0)
            maxs = non_zero.max(axis=0)
            voxel_widths = maxs - mins

            # Convert voxel widths to millimetres.
            mm_widths = voxel_widths * spacing

            data = {
                'region': name,
                'centroid-mm-x': mm_centroid[0],
                'centroid-mm-y': mm_centroid[1],
                'centroid-mm-z': mm_centroid[2],
                'centroid-voxels-x': centroid[0],
                'centroid-voxels-y': centroid[1],
                'centroid-voxels-z': centroid[2],
                'width-mm-x': mm_widths[0],
                'width-mm-y': mm_widths[1],
                'width-mm-z': mm_widths[2],
                'width-voxels-x': voxel_widths[0],
                'width-voxels-y': voxel_widths[1],
                'width-voxels-z': voxel_widths[2]
            }
            data = dict(filter(self._filter_on_dict_keys(columns, whitelist='region'), data.items()))
            df = df.append(data, ignore_index=True)

        # Set column type.
        df = df.astype(cols)

        # Sort by region.
        df = df.sort_values('region').reset_index(drop=True)

        return df

    def _filter_on_dict_keys(
        self,
        keys: Union[str, List[str]] = 'all',
        whitelist: Union[str, List[str]] = None) -> Callable[[Tuple[str, Any]], bool]:
        """
        returns: a function that filters out unneeded keys.
        kwargs:
            keys: description of required keys.
            whitelist: keys that are never filtered.
        """
        def fn(item: Tuple[str, Any]) -> bool:
            key, _ = item
            # Allow based on whitelist.
            if whitelist is not None:
                if type(whitelist) == str:
                    if key == whitelist:
                        return True
                elif key in whitelist:
                    return True
            
            # Filter based on allowed keys.
            if ((isinstance(keys, str) and (keys == 'all' or key == keys)) or
                ((isinstance(keys, list) or isinstance(keys, np.ndarray) or isinstance(keys, tuple)) and key in keys)):
                return True
            else:
                return False
        return fn
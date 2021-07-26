import nibabel as nib
import numpy as np
import os
import pandas as pd
from typing import List, OrderedDict

from mymi import cache
from mymi import config
from mymi.regions import is_region
from mymi import types

class NIFTIObject:
    def __init__(
        self,
        dataset: str,
        id: str):
        self._dataset = dataset
        self._id = id
        self._path = os.path.join(config.directories.datasets, 'raw', dataset, 'raw')

    def region_names(self) -> List[str]:
        files = os.listdir(self._path)
        names = []
        for f in files:
            if not is_region(f):
                continue
            region_path = os.path.join(self._path, f)
            for r in os.listdir(region_path):
                id = r.replace('.nii.gz', '')
                if id == self._id:
                    names.append(f)
        # Create dataframe.
        df = pd.DataFrame(names, columns=['region'])
        return df

    def has_region(
        self,
        region: str) -> bool:
        return region in list(self.region_names().region)

    def ct_spacing(self) -> types.ImageSpacing3D:
        path = os.path.join(self._path, 'ct', f"{self._id}.nii.gz")
        img = nib.load(path)
        affine = img.affine
        spacing = (abs(affine[0][0]), abs(affine[1][1]), abs(affine[2][2]))
        return spacing

    def ct_offset(self) -> types.Point3D:
        path = os.path.join(self._path, 'ct', f"{self._id}.nii.gz")
        img = nib.load(path)
        affine = img.affine
        offset = (affine[0][3], affine[1][3], affine[2][3])
        return offset

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

    def ct_data(self) -> np.ndarray:
        path = os.path.join(self._path, 'ct', f"{self._id}.nii.gz")
        img = nib.load(path)
        data = img.get_data()
        return data

    def region_data(
        self,
        regions: types.PatientRegions = 'all') -> OrderedDict:
        # Convert regions to list.
        if type(regions) == str:
            if regions == 'all':
                regions = list(self.region_names().region)
            else:
                regions = [regions]

        data = {}
        for region in regions:
            if not is_region(region):
                raise ValueError(f"Requested region '{region}' not a valid internal region.")
            if not self.has_region(region):
                raise ValueError(f"Requested region '{region}' not found for object '{self._id}', dataset '{self._dataset}'.")
            
            path = os.path.join(self._path, region, f"{self._id}.nii.gz")
            img = nib.load(path)
            rdata = img.get_fdata()
            data[region] = rdata
        return data

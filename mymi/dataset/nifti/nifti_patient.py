import nibabel as nib
import numpy as np
import os
import pandas as pd
from scipy.ndimage import center_of_mass
from typing import Any, Callable, List, OrderedDict, Tuple, Union

from mymi import config
from mymi.regions import is_region
from mymi import types

class NIFTIPatient:
    def __init__(
        self,
        dataset: 'NIFTIDataset',
        id: types.PatientID):
        self._global_id = f"{dataset} - {id}"
        self._dataset = dataset
        self._id = str(id)

        # Check that patient ID exists.
        ct_path = os.path.join(dataset.path, 'data', 'ct', f'{id}.nii.gz')
        if not os.path.exists(ct_path):
            raise ValueError(f"Patient '{self}' not found.")
    
    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id

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
            data[region] = rdata.astype(bool)
        return data


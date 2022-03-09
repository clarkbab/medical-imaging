import nibabel as nib
import numpy as np
import os
from typing import Any, List, Optional, OrderedDict

from mymi.regions import is_region
from mymi import types

class NIFTIPatient:
    def __init__(
        self,
        dataset: 'NIFTIDataset',
        id: types.PatientID):
        self._dataset = dataset
        self._id = str(id)
        self._global_id = f"{dataset} - {self._id}"

        # Check that patient ID exists.
        ct_path = os.path.join(dataset.path, 'data', 'ct', f'{self._id}.nii.gz')
        if not os.path.exists(ct_path):
            raise ValueError(f"Patient '{self}' not found.")
    
    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def patient_id(self) -> Optional[str]:
        manifest = self._dataset.anon_manifest
        if manifest is None:
            return None
        pat_df = manifest[manifest['anon-id'] == self._id]['patient-id']
        if len(pat_df) == 0:
            return None
        elif len(pat_df) > 1:
            raise ValueError(f"Patient {self} appears multiple times in anon. manifest.")
        pat_id = pat_df.iloc[0]
        return pat_id

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
        names = list(sorted(names))

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

    @property
    def ct_spacing(self) -> types.ImageSpacing3D:
        path = os.path.join(self._dataset.path, 'data', 'ct', f"{self._id}.nii.gz")
        img = nib.load(path)
        affine = img.affine
        spacing = (abs(affine[0][0]), abs(affine[1][1]), abs(affine[2][2]))
        return spacing

    @property
    def ct_offset(self) -> types.Point3D:
        path = os.path.join(self._dataset.path, 'data', 'ct', f"{self._id}.nii.gz")
        img = nib.load(path)
        affine = img.affine
        offset = (affine[0][3], affine[1][3], affine[2][3])
        return offset

    @property
    def ct_data(self) -> np.ndarray:
        path = os.path.join(self._dataset.path, 'data', 'ct', f"{self._id}.nii.gz")
        img = nib.load(path)
        data = img.get_data()
        return data

    @property
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
            
            path = os.path.join(self._dataset.path, 'data', region, f'{self._id}.nii.gz')
            img = nib.load(path)
            rdata = img.get_fdata()
            data[region] = rdata.astype(bool)
        return data


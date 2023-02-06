import nibabel as nib
import numpy as np
import os
import pandas as pd
from typing import List, Optional, OrderedDict, Tuple

from mymi.regions import is_region
from mymi import types
from mymi.utils import arg_to_list

class NIFTIPatient:
    def __init__(
        self,
        dataset: 'NIFTIDataset',
        id: types.PatientID,
        excluded_regions: Optional[pd.DataFrame] = None):
        self.__dataset = dataset
        self.__excluded_regions = excluded_regions
        self.__id = str(id)
        self.__global_id = f"{dataset} - {self.__id}"

        # Check that patient ID exists.
        self.__path = os.path.join(dataset.path, 'data', 'ct', f'{self.__id}.nii.gz')
        if not os.path.exists(self.__path):
            raise ValueError(f"Patient '{self}' not found.")

    @property
    def ct_data(self) -> np.ndarray:
        path = os.path.join(self.__dataset.path, 'data', 'ct', f"{self.__id}.nii.gz")
        img = nib.load(path)
        data = img.get_data()
        return data

    @property
    def ct_offset(self) -> types.Point3D:
        path = os.path.join(self.__dataset.path, 'data', 'ct', f"{self.__id}.nii.gz")
        img = nib.load(path)
        affine = img.affine
        offset = (affine[0][3], affine[1][3], affine[2][3])
        return offset

    @property
    def ct_size(self) -> np.ndarray:
        return self.ct_data.shape

    @property
    def ct_spacing(self) -> types.ImageSpacing3D:
        path = os.path.join(self.__dataset.path, 'data', 'ct', f"{self.__id}.nii.gz")
        img = nib.load(path)
        affine = img.affine
        spacing = (abs(affine[0][0]), abs(affine[1][1]), abs(affine[2][2]))
        return spacing
    
    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def dose_data(self) -> np.ndarray:
        filepath = os.path.join(self.__dataset.path, 'data', 'dose', f'{self.__id}.nii.gz')
        if not os.path.exists(filepath):
            raise ValueError(f"Dose data not found for patient '{self}'.")
        img = nib.load(filepath)
        data = img.get_fdata()
        return data

    @property
    def id(self) -> str:
        return self.__id

    @property
    def origin(self) -> Tuple[str, str]:
        df = self.__dataset.anon_index
        row = df[df['anon-id'] == self.__id].iloc[0]
        dataset = row['dicom-dataset']
        pat_id = row['patient-id']
        return (dataset, pat_id)

    @property
    def patient_id(self) -> Optional[str]:
        # Get anon manifest.
        manifest = self.__dataset.anon_manifest
        if manifest is None:
            raise ValueError(f"No anon manifest found for dataset '{self.__dataset}'.")

        # Get patient ID.
        manifest = manifest[manifest['anon-id'] == self.__id]
        if len(manifest) == 0:
            raise ValueError(f"No entry for anon patient '{self.__id}' found in anon manifest for dataset '{self.__dataset}'.")
        pat_id = manifest.iloc[0]['patient-id']

        return pat_id

    @property
    def path(self) -> str:
        return self.__path

    def has_region(
        self,
        region: str,
        excluded_regions: bool = False) -> bool:
        return region in self.list_regions(excluded_regions=excluded_regions)

    def list_regions(
        self,
        excluded_regions: bool = False) -> List[str]:
        dirpath = os.path.join(self.__dataset.path, 'data', 'regions')
        folders = os.listdir(dirpath)
        names = []
        for f in folders:
            if not is_region(f):
                continue
            dirpath = os.path.join(self.__dataset.path, 'data', 'regions', f)
            if f'{self.__id}.nii.gz' in os.listdir(dirpath):
                # Check exclusion list.
                if not excluded_regions and self.__excluded_regions is not None:
                    pr_df = self.__excluded_regions[(self.__excluded_regions['patient-id'] == self.__id) & (self.__excluded_regions['region'] == f)]
                    if len(pr_df) == 1:
                        continue
                names.append(f)
        names = list(sorted(names))

        return names

    def region_data(
        self,
        region: types.PatientRegions = 'all') -> OrderedDict:
        if region == 'all':
            regions = self.list_regions()
        else:
            regions = arg_to_list(region, str)

        data = {}
        for region in regions:
            if not is_region(region):
                raise ValueError(f"Requested region '{region}' not a valid internal region.")
            if not self.has_region(region):
                raise ValueError(f"Requested region '{region}' not found for patient '{self.__id}', dataset '{self.__dataset}'.")
            
            path = os.path.join(self.__dataset.path, 'data', 'regions', region, f'{self.__id}.nii.gz')
            img = nib.load(path)
            rdata = img.get_fdata()
            data[region] = rdata.astype(bool)
        return data

    def __str__(self) -> str:
        return self.__global_id

import nibabel as nib
import nrrd
import numpy as np
import os
import pandas as pd
from typing import List, Literal, Optional, OrderedDict, Tuple

from mymi import logging
from mymi.regions import is_region, regions_to_list
from mymi.types import ImageSpacing3D, PatientID, PatientRegions, Point3D
from mymi.utils import arg_to_list

class NRRDPatient:
    def __init__(
        self,
        dataset: 'NRRDDataset',
        id: PatientID,
        excluded_labels: Optional[pd.DataFrame] = None):
        self.__dataset = dataset
        self.__id = str(id)
        self.__excluded_labels = excluded_labels[excluded_labels['patient-id'] == self.__id] if excluded_labels is not None else None
        self.__global_id = f"{dataset} - {self.__id}"

        # Check that patient ID exists.
        self.__path = os.path.join(dataset.path, 'data', 'ct', f'{self.__id}.nrrd')
        if not os.path.exists(self.__path):
            raise ValueError(f"Patient '{self}' not found.")

    @property
    def ct_data(self) -> np.ndarray:
        data, _ = nrrd.read(self.__path)
        return data

    @property
    def ct_offset(self) -> Point3D:
        _, header = nrrd.read(self.__path)
        offset = tuple(header['space origin'])
        return offset

    @property
    def ct_size(self) -> np.ndarray:
        data, header = nrrd.read(self.__path)
        size = data.shape
        assert size == tuple(header['sizes'])
        return size

    @property
    def ct_spacing(self) -> ImageSpacing3D:
        _, header = nrrd.read(self.__path)
        # Assert that there are no off-diagonal entries.
        affine = header['space directions']
        assert affine.sum() == np.diag(affine).sum()
        spacing = (abs(affine[0][0]), abs(affine[1][1]), abs(affine[2][2]))
        return spacing
    
    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> str:
        return self.__id

    @property
    def path(self) -> str:
        return self.__path

    def has_region(
        self,
        regions: PatientRegions,
        labels: Literal['included', 'excluded', 'all'] = 'included') -> bool:
        regions = regions_to_list(regions)
        pat_regions = self.list_regions(labels=labels)
        # Return 'True' if patient has at least one of the requested regions.
        if len(np.intersect1d(regions, pat_regions)) != 0:
            return True
        else:
            return False

    def list_regions(
        self,
        labels: Literal['included', 'excluded', 'all'] = 'included') -> List[str]:
        # Find regions by file names.
        dirpath = os.path.join(self.__dataset.path, 'data', 'regions')
        folders = os.listdir(dirpath)
        regions = []
        for f in folders:
            dirpath = os.path.join(self.__dataset.path, 'data', 'regions', f)
            if f'{self.__id}.nrrd' in os.listdir(dirpath):
                # Apply exclusion criteria.
                if self.__excluded_labels is not None:
                    df = self.__excluded_labels[(self.__excluded_labels['patient-id'] == self.__id) & (self.__excluded_labels['region'] == f)]
                    if labels == 'included' and len(df) >= 1:
                        continue
                    elif labels == 'excluded' and len(df) == 0:
                        continue
                regions.append(f)

        regions = list(sorted(regions))
        return regions

    def region_data(
        self,
        labels: Literal['included', 'excluded', 'all'] = 'included',
        regions: PatientRegions = 'all') -> OrderedDict:
        regions = regions_to_list(regions, literals={ 'all': self.list_regions(labels=labels)})

        data = {}
        for region in regions:
            if not self.has_region(region, labels=labels):
                raise ValueError(f"Requested region '{region}' not found for patient '{self.__id}', dataset '{self.__dataset}'.")
            path = os.path.join(self.__dataset.path, 'data', 'regions', region, f'{self.__id}.nrrd')
            rdata, _ = nrrd.read(path)
            data[region] = rdata.astype(bool)

        return data

    def __str__(self) -> str:
        return self.__global_id

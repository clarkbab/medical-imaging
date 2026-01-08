import numpy as np
import os
from typing import *

from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

from ....dicom import DicomDataset, DicomRtStructSeries
from ....region_map import RegionMap
from .image import NiftiImageSeries

class NiftiRegionsSeries(NiftiImageSeries):
    def __init__(
        self,
        dataset: DatasetID,
        pat: PatientID,
        study: StudyID,
        id: SeriesID,
        index: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None,
        ) -> None:
        super().__init__('regions', dataset, pat, study, id, index=index)
        extensions = ['.nii', '.nii.gz', '.nrrd']
        dirpath = os.path.join(config.directories.datasets, 'nifti', self._dataset_id, 'data', 'patients', self._pat_id, self._study_id, self._modality, self._id)
        if not os.path.exists(dirpath):
            raise ValueError(f"No NiftiRegionsSeries found for study '{self._study_id}'. Dirpath: {dirpath}")
        self.__dirpath = dirpath
        self.__region_map = region_map

    def data(
        self,
        region: Regions = 'all',
        region_ignore_missing: bool = True,
        **kwargs) -> RegionArrays:
        regions = regions_to_list(region, literals={ 'all': self.list_regions })

        rd = {}
        for r in regions:
            if not self.has_region(r):
                if region_ignore_missing:
                    continue
                else:
                    raise ValueError(f'Region {r} not found in image {self.id}.')

            # Load region from disk.
            # If multiple regions have been mapped to the same ID, then get the union of these regions.
            filepaths = self.filepaths(r)
            ds = []
            for f in filepaths:
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    d, _, _ = load_nifti(f)
                elif f.endswith('.nrrd'):
                    d, _, _ = load_nrrd(f)
                else:
                    raise ValueError(f'Unsupported file format: {f}')
                ds.append(d)
            rd[r] = np.sum(ds, axis=0).clip(0, 1).astype(bool)

        return rd

    @property
    def dicom(self) -> DicomRtStructSeries:
        if self._index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self._index[['dataset', 'patient-id', 'study-id', 'series-id', 'modality', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id', 'dicom-series-id']]
        index = index[(index['dataset'] == self._dataset_id) & (index['patient-id'] == self._pat_id) & (index['study-id'] == self._study_id) & (index['series-id'] == self._id) & (index['modality'] == 'regions')].drop_duplicates()
        assert len(index) == 1
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id']).rtstruct_series(row['dicom-series-id'])

    def filepaths(
        self,
        region: RegionID) -> List[FilePath]:
        if not self.has_region(region):
            raise ValueError(f'Region {region} not found in series {self.id}.')
        # Region mapping is many-to-one, so we could get multiple files on disk for the same mapped region.
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        disk_ids = self.__region_map.inv_map_region(region, disk_regions=self.list_regions(use_mapping=False)) if self.__region_map is not None else region
        disk_ids = arg_to_list(disk_ids, RegionID)
        # Check all possible file extensions.
        filepaths = [os.path.join(self.__dirpath, f'{i}{e}') for i in disk_ids for e in image_extensions if os.path.exists(os.path.join(self.__dirpath, f'{i}{e}'))]
        if len(filepaths) == 0:
            raise ValueError(f'No region filespaths found for region {region} in series {self}.')
        return filepaths

    def has_region(
        self,
        regions: RegionIDs,
        any: bool = False,
        **kwargs) -> bool:
        all_ids = self.list_regions(**kwargs)
        regions = arg_to_list(regions, RegionID)
        n_overlap = len(np.intersect1d(regions, all_ids))
        return n_overlap > 0 if any else n_overlap == len(regions)

    @alias_kwargs(('um', 'use_mapping'))
    def list_regions(
        self,
        region: RegionIDs = 'all',
        use_mapping: bool = True) -> List[Region]:
        # Load regions from filenames.
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        ids = os.listdir(self.__dirpath)
        ids = [i.replace(e, '') for i in ids for e in image_extensions if i.endswith(e)]

        # Apply region mapping.
        if use_mapping and self.__region_map is not None:
            ids = [self.__region_map.map_region(i) if self.__region_map is not None else i for i in ids]

        # Filter on 'only'.
        if region != 'all':
            regions = regions_to_list(region)
            ids = [r for r in ids if r in regions]

        # Sort regions.
        ids = list(sorted(ids))

        return ids

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

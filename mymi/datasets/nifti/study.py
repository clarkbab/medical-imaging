import numpy as np
import os
from typing import *

from mymi import config
from mymi.typing import *

from ..dicom import DicomDataset, DicomStudy
from ..mixins import IndexMixin
from ..region_map import RegionMap
from ..study import Study
from .series import *

class NiftiStudy(IndexMixin, Study):
    def __init__(
        self,
        dataset: DatasetID,
        pat: PatientID,
        id: StudyID,
        ct_from: Optional['NiftiStudy'] = None,
        index: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None) -> None:
        super().__init__(dataset, pat, id, ct_from=ct_from, index=index, region_map=region_map)
        self.__path = os.path.join(config.directories.datasets, 'nifti', self._dataset_id, 'data', 'patients', self._pat_id, self._id)
        if not os.path.exists(self.__path):
            raise ValueError(f"No nifti study '{self._id}' found at path: {self.__path}")

    @property
    def dicom(self) -> DicomStudy:
        if self._index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self._index[['dataset', 'patient-id', 'study-id', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id']]
        index = index[(index['dataset'] == self._dataset_id) & (index['patient-id'] == self._pat_id) & (index['study-id'] == self._id)].drop_duplicates()
        assert len(index) == 1
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id'])

    def series(
        self,
        id: NiftiSeriesID,
        modality: NiftiModality) -> Union[NiftiImageSeries, NiftiLandmarksSeries]:
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        if modality == 'ct':
            id = handle_idx_prefix(id, lambda: self.list_series('ct'))
            if self._ct_from is None:
                index = self._index[(self._index['dataset'] == self._dataset_id) & (self._index['patient-id'] == self._pat_id) & (self._index['study-id'] == self._id) & (self._index['series-id'] == id) & (self._index['modality'] == 'ct')].copy() if self._index is not None else None
                return NiftiCtSeries(self._dataset_id, self._pat_id, self._id, id, index=index)
            else:
                return self._ct_from.series(id, modality)
        elif modality == 'dose':
            id = handle_idx_prefix(id, lambda: self.list_series('dose'))
            # Could multiple series have the same series-id? Yeah definitely.
            index = self._index[(self._index['dataset'] == self._dataset_id) & (self._index['patient-id'] == self._pat_id) & (self._index['study-id'] == self._id) & (self._index['series-id'] == id) & (self._index['modality'] == 'dose')].copy() if self._index is not None else None
            return NiftiDoseSeries(self._dataset_id, self._pat_id, self._id, id, index=index)
        elif modality == 'landmarks':
            id = handle_idx_prefix(id, lambda: self.list_series('landmarks'))
            print(id)
            index = self._index[(self._index['dataset'] == self._dataset_id) & (self._index['patient-id'] == self._pat_id) & (self._index['study-id'] == self._id) & (self._index['series-id'] == id) & (self._index['modality'] == 'landmarks')].copy() if self._index is not None else None
            ref_ct = self.default_series('ct')
            ref_dose = self.default_series('dose')
            return NiftiLandmarksSeries(self._dataset_id, self._pat_id, self._id, id, index=index, ref_ct=ref_ct, ref_dose=ref_dose)
        elif modality == 'mr':
            id = handle_idx_prefix(id, lambda: self.list_series('mr'))
            index = self._index[(self._index['dataset'] == self._dataset_id) & (self._index['patient-id'] == self._pat_id) & (self._index['study-id'] == self._id) & (self._index['series-id'] == id) & (self._index['modality'] == 'mr')].copy() if self._index is not None else None
            return NiftiMrSeries(self._dataset_id, self._pat_id, self._id, id, index=index)
        elif modality == 'regions':
            id = handle_idx_prefix(id, lambda: self.list_series('regions'))
            index = self._index[(self._index['dataset'] == self._dataset_id) & (self._index['patient-id'] == self._pat_id) & (self._index['study-id'] == self._id) & (self._index['series-id'] == id) & (self._index['modality'] == 'regions')].copy() if self._index is not None else None
            return NiftiRegionsSeries(self._dataset_id, self._pat_id, self._id, id, index=index, region_map=self._region_map)
        else:
            raise ValueError(f"Unknown NiftiSeries modality '{modality}'.")

    def default_series(
        self,
        modality: NiftiModality) -> Optional[NiftiSeries]:
        serieses = self.list_series(modality)
        if len(serieses) > 1:
            logging.warning(f"More than one '{modality}' series found for '{self}', defaulting to latest.")
        return self.series(serieses[-1], modality) if len(serieses) > 0 else None

    def has_series(
        self,
        id: NiftiSeriesID,
        modality: NiftiModality) -> bool:
        return id in self.list_series(modality)

    def list_series(
        self,
        modality: NiftiModality) -> List[NiftiSeriesID]:
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        if modality == 'ct':
            if self._ct_from is None:
                dirpath = os.path.join(self.__path, 'ct')
                ct_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
                ct_ids = [i.replace(e, '') for i in ct_ids for e in image_extensions if i.endswith(e)]
                return ct_ids
            else:
                return self._ct_from.list_series(modality)
        elif modality == 'dose':
            dirpath = os.path.join(self.__path, 'dose')
            dose_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            dose_ids = [i.replace(e, '') for i in dose_ids for e in image_extensions if i.endswith(e)]
            return dose_ids
        elif modality == 'landmarks':
            dirpath = os.path.join(self.__path, 'landmarks')
            landmarks_ids = list(sorted(f.replace('.csv', '') for f in os.listdir(dirpath))) if os.path.exists(dirpath) else []
            return landmarks_ids
        elif modality == 'mr':
            dirpath = os.path.join(self.__path, 'mr')
            mr_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            mr_ids = [i.replace(e, '') for i in mr_ids for e in image_extensions if i.endswith(e)]
            return mr_ids
        elif modality == 'regions':
            dirpath = os.path.join(self.__path, 'regions')
            regions_ids = list(sorted(os.listdir(dirpath))) if os.path.exists(dirpath) else []
            return regions_ids
        else:
            raise ValueError(f"Unknown modality '{modality}'.")

    @property
    def origin(self) -> Dict[str, str]:
        if self._index is None:
            raise ValueError(f"No 'index.csv' provided for dataset '{self.__patient.dataset}'.")
        info = self._index.iloc[0].to_dict()
        info = {k: info[k] for k in ['dicom-dataset', 'dicom-patient-id', 'dicom-study-id']}
        return info

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add 'list_{mod}_series' methods.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiStudy, f'list_{m}_series', lambda self, m=m: self.list_series(m))

# Add '{mod}_series' methods.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiStudy, f'{m}_series', lambda self, series, m=m: self.series(series, m))
    
# Add 'has_{mod}' properties.
# Note that 'has_landmarks' refers to the landmarks series, whereas 'has_landmark' is used for
# a single landmark ID. Same for regions. Could be confusing.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiStudy, f'has_{m}', property(lambda self, m=m: self.default_series(m) is not None))

# Add 'default_{mod}' properties.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiStudy, f'default_{m}', property(lambda self, m=m: self.default_series(m)))
    
# Add image filepath shortcuts from 'default_series(mod)'
mods = ['ct', 'mr', 'dose']
for m in mods:
    setattr(NiftiStudy, f'{m}_filepath', property(lambda self, m=m: getattr(self.default_series(m), 'filepath') if self.default_series(m) is not None else None))
setattr(NiftiStudy, 'region_filepaths', lambda self, region: self.default_series('regions').filepaths(region) if self.default_series('regions') is not None else None)

# Add image property shortcuts from 'default_series(mod)'.
mods = ['ct', 'mr', 'dose']
props = ['data', 'fov', 'origin', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(NiftiStudy, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(self.default_series(m), p) if self.default_series(m) is not None else None))

# Add landmark/region method shortcuts from 'default_series(mod)'.
mods = ['landmarks', 'regions']
for m in mods:
    setattr(NiftiStudy, f'has_{m[:-1]}', lambda self, *args, m=m, **kwargs: getattr(self.default_series(m), f'has_{m[:-1]}')(*args, **kwargs) if self.default_series(m) is not None else False)
    setattr(NiftiStudy, f'list_{m}', lambda self, *args, m=m, **kwargs: getattr(self.default_series(m), f'list_{m}')(*args, **kwargs) if self.default_series(m) is not None else [])
    setattr(NiftiStudy, f'{m}_data', lambda self, *args, m=m, **kwargs: self.default_series(m).data(*args, **kwargs) if self.default_series(m) is not None else None)

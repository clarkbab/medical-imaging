from typing import *

from mymi.datasets.nifti import NiftiDataset
from mymi.predictions.nifti import load_registration, load_segmenter_predictions
from mymi.typing import *
from mymi.utils import *

from ..plotting import plot_patient_histogram as pph, plot_patient as pp, plot_registration as pr

@delegates(pp)
def plot_patient(dataset, *args, **kwargs) -> None:
    set = NiftiDataset(dataset)
    fns = {
        'ct_series': lambda study, data_id: study.series(data_id, 'ct'),
        'default_dose': lambda study: study.default_dose,
        'default_landmarks': lambda study: study.default_landmarks,
        'default_regions': lambda study: study.default_regions,
        'dose_series': lambda study, series: study.series(series, 'dose'),
        'has_dose': lambda study: study.has_dose,
        'landmarks_data': lambda series, landmark_ids: series.data(landmark=landmark_ids),
        'landmark_series': lambda study, series: study.series(series, 'landmarks'),
        'regions_data': lambda series, region_ids: series.data(region=region_ids),
        'region_series': lambda study, series: study.series(series, 'regions'),
        'study_datetime': lambda study: None,  # Not currently available, might be able to store in nifti metadata: https://pycad.medium.com/store-the-metadata-in-nifti-and-nrrd-8aa4c6d942b5
    }
    pp(set, fns, *args, **kwargs)

@delegates(pph)
def plot_patient_histogram(*args, **kwargs) -> None:
    pph(NiftiDataset, *args, **kwargs)

@delegates(pr)
def plot_registration(*args, **kwargs) -> None:
    fns = {
        'load_registration': load_registration,
    }
    pr(NiftiDataset, fns, *args, **kwargs)

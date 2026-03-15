from mymi.constants import *
from mymi.datasets.dicom import DicomDataset

from ..plotting import plot_patients as pp, plot_patient_histogram as pph  

def plot_patients(dataset, *args, **kwargs) -> None:
    set = DicomDataset(dataset)
    fns = {
        'ct_series': lambda study, series: study.series(series, 'ct'),
        'default_dose': lambda study: study.default_rtdose,
        'default_landmarks': lambda study: study.default_rtstruct,
        'default_regions': lambda study: study.default_rtstruct,
        'dose_series': lambda study, series: study.series(series, 'rtdose'),
        'has_dose': lambda study: study.has_rtdose,
        'landmarks_data': lambda series, landmarks: series.landmarks_data(landmarks=landmarks),
        'landmark_series': lambda study, series: study.series(series, 'rtstruct'),
        'regions_data': lambda series, regions: series.regions_data(regions=region_ids),
        'region_series': lambda study, series: study.series(series, 'rtstruct'),
        'study_datetime': lambda study: study.date,
    }
    pp(set, fns, *args, **kwargs)

def plot_patient_histogram(*args, **kwargs) -> None:
    pph(DicomDataset, *args, **kwargs)

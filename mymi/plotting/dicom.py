from mymi.constants import *
from mymi.datasets.dicom import DicomDataset

from ..plotting import plot_patient as pp, plot_patient_histogram as pph  

def plot_patient(dataset, *args, **kwargs) -> None:
    set = DicomDataset(dataset)
    fns = {
        'ct_series': lambda study, series_id: study.series(series_id, 'ct'),
        'default_dose': lambda study: study.default_rtdose,
        'default_landmarks': lambda study: study.default_rtstruct,
        'default_regions': lambda study: study.default_rtstruct,
        'dose_series': lambda study, series_id: study.series(series_id, 'rtdose'),
        'has_dose': lambda study: study.has_rtdose,
        'landmark_data': lambda series, ids: series.landmark_data(landmark_ids=ids),
        'landmarks_series': lambda study, series_id: study.series(series_id, 'rtstruct'),
        'region_data': lambda series, ids: series.region_data(region_ids=ids),
        'regions_series': lambda study, series_id: study.series(series_id, 'rtstruct'),
        'study_datetime': lambda study: study.date,
    }
    pp(set, fns, *args, **kwargs)

def plot_patient_histogram(*args, **kwargs) -> None:
    pph(DicomDataset, *args, **kwargs)

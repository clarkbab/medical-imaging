from mymi.datasets.dicom import DicomDataset

from ..plotting import plot_patients as pp, plot_patient_histograms as pph  

def plot_patients(*args, **kwargs) -> None:
    dicom_fns = {
        'ct_image': lambda study, series_id: study.series(series_id),
        'dose_image': lambda study: study.default_rtdose,
        'has_dose': lambda study: study.has_rtdose,
    }
    pp(DicomDataset, dicom_fns, *args, **kwargs)

def plot_patient_histograms(*args, **kwargs) -> None:
    pph(DicomDataset, *args, **kwargs)

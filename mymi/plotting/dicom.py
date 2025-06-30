from mymi.datasets.dicom import DicomDataset

from ..plotting import plot_patients as pp, plot_patient_histograms as pph  

def plot_patients(*args, **kwargs) -> None:
    dicom_fns = {
        'default_dose': lambda study: study.default_rtdose,
        'has_dose': lambda study: study.has_rtdose,
        'image': lambda study, id: study.series(id),
    }
    pp(DicomDataset, dicom_fns, *args, **kwargs)

def plot_patient_histograms(*args, **kwargs) -> None:
    pph(DicomDataset, *args, **kwargs)

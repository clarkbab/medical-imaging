from mymi.datasets.dicom import DicomDataset

from ..plotting import plot_patients as pp, plot_patient_histograms as pph  

def plot_patients(*args, **kwargs) -> None:
    pp(DicomDataset, *args, **kwargs)

def plot_patient_histograms(*args, **kwargs) -> None:
    pph(DicomDataset, *args, **kwargs)

from mymi.datasets import NiftiDataset

from ..dicom import convert_to_dicom as convert_to_dicom_base

def convert_to_dicom(
    dataset: str,
    **kwargs) -> None:
    set = NiftiDataset(dataset)
    fns = {
        'has_series': lambda study, id, modality: study.has_series(id, modality),
        'list_series': lambda study, modality: study.list_series(modality),
        'series': lambda study, id, modality: study.data(id, modality),
    }
    convert_to_dicom_base(set, fns, **kwargs)

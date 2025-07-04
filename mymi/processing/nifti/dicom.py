from mymi.datasets import NiftiDataset

from ..dicom import convert_to_dicom as convert_to_dicom_base

def convert_to_dicom(
    dataset: str,
    **kwargs) -> None:
    convert_to_dicom_base(NiftiDataset(dataset), **kwargs)

from mymi.datasets import NiftiDataset

from ...processing import convert_to_dicom as convert_to_dicom_base

def convert_to_dicom(
    dataset: str,
    dest_dataset: str,
    **kwargs) -> None:
    convert_to_dicom_base(NiftiDataset(dataset), dest_dataset, **kwargs)

from typing import Optional

from mymi import types
from mymi.dataset.nifti import NIFTIDataset

from .preprocessing import convert_to_training as ctt

def convert_to_training(
    dataset: str,
    dest_dataset: str,
    dilate_regions: Optional[types.PatientRegions] = None,
    p_test: float = 0.2,
    p_train: float = 0.6,
    p_val: float = 0.2,
    random_seed: int = 42,
    regions: types.PatientRegions = 'all',
    size: Optional[types.ImageSize3D] = None,
    spacing: Optional[types.ImageSpacing3D] = None,
    use_mapping: bool = True):
    set = NIFTIDataset(dataset)
    ctt(set, dest_dataset, dilate_regions=dilate_regions, p_test=p_test, p_train=p_train, p_val=p_val, 
        random_seed=random_seed, regions=regions, size=size, spacing=spacing, use_mapping=use_mapping)

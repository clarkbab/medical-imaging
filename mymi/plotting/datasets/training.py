from typing import *

from mymi.datasets import TrainingDataset
from mymi.typing import *
from mymi.utils import *

from ..plotting import *

def plot_dataset_histogram(
    dataset: str,
    n_samples: Optional[int] = None,
    sample_ids: Optional[PatientIDs] = None,
    split_id: Optional[SplitID] = None,
    **kwargs) -> None:
    set = TrainingDataset(dataset)
    if split_id is None:
        split = set.split(set.list_splits()[0])
    else:
        split = set.split(split_id)
    if n_samples is not None:
        assert sample_ids is None
        sample_ids = split.list_samples()
        sample_ids = sample_ids[:n_samples]
    inputs = [split.sample(s).input for s in sample_ids]
    inputs = np.concatenate([i.flatten() for i in inputs])
    plot_histogram(inputs, **kwargs)

def plot_patients(
    dataset: str,
    sample_idx: str,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    region: Optional[PatientRegions] = None,
    region_label: Optional[Dict[str, str]] = None,     # Gives 'regions' different names to those used for loading the data.
    **kwargs) -> None:
    regions = arg_to_list(region, str)
    region_labels = arg_to_list(region_label, str)

    # Load data.
    set = TrainingDataset(dataset, **kwargs)
    sample = set.sample(sample_idx)
    ct_data = sample.input
    region_data = sample.label(region=regions) if regions is not None else None
    spacing = sample.spacing

    if centre is not None:
        if type(centre) == str:
            if region_data is None or centre not in region_data:
                centre = sample.label(region=centre)[centre]

    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                crop = sample.label(region=crop)[crop]

    if region_labels is not None:
        # Rename 'regions' and 'region_data' keys.
        regions = [region_labels[r] if r in region_labels else r for r in regions]
        for old, new in region_labels.items():
            region_data[new] = region_data.pop(old)

        # Rename 'centre' and 'crop' keys.
        if type(centre) == str and centre in region_labels:
            centre = region_labels[centre] 
        if type(crop) == str and crop in region_labels:
            crop = region_labels[crop]

    # Plot.
    plot_patients_matrix(sample_idx, ct_data.shape, spacing, centre=centre, crop=crop, ct_data=ct_data, region_data=region_data, **kwargs)

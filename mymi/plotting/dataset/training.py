from typing import Dict, Optional, Tuple, Union

from mymi.dataset import TrainingDataset
from mymi.prediction.dataset.training import get_sample_localiser_prediction
from mymi import types
from mymi.utils import arg_to_list

from ..plotting import plot_distribution, plot_localiser_prediction
from ..plotting import plot_patient as plot_patient_base

def plot_patient(
    dataset: str,
    sample_idx: str,
    centre: Optional[str] = None,
    crop: Optional[Union[str, types.Box2D]] = None,
    region: Optional[types.PatientRegions] = None,
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
    plot_patient_base(sample_idx, ct_data.shape, spacing, centre=centre, crop=crop, ct_data=ct_data, region_data=region_data, **kwargs)

def plot_sample_localiser_prediction(
    dataset: str,
    sample_idx: str,
    region: str,
    localiser: types.ModelName,
    **kwargs) -> None:
    # Load data.
    set = TrainingDataset(dataset, **kwargs)
    samples = set.sample(sample_idx)
    input = sample.input
    label = sample.label(region=region)[region]
    spacing = sample.spacing

    # Set truncation if 'SpinalCord'.
    truncate = True if region == 'SpinalCord' else False

    # Make prediction.
    pred = get_sample_localiser_prediction(dataset, sample_idx, localiser, truncate=truncate)
    
    # Plot.
    plot_localiser_prediction(sample_idx, region, input, label, spacing, pred, **kwargs)

def plot_sample_distribution(
    dataset: str,
    sample_idx: int,
    figsize: Tuple[float, float] = (12, 6),
    range: Optional[Tuple[float, float]] = None,
    resolution: float = 10) -> None:
    # Load data.
    set = TrainingDataset(dataset)
    sample = set.sample(sample_idx)
    input = sample.input
    
    # Plot distribution.
    plot_distribution(input, figsize=figsize, range=range, resolution=resolution)

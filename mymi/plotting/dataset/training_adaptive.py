import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Tuple, Union

from mymi.dataset.training_adaptive import TrainingAdaptiveDataset
from mymi.prediction.dataset.training import get_sample_localiser_prediction
from mymi import types
from mymi.utils import arg_to_list

from ..plotter import plot_distribution, plot_localiser_prediction
from ..plotter import plot_region as plot_region_base

def plot_region(
    dataset: str,
    sample_idx: str,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Box2D]] = None,
    figsize: Tuple[float, float] = (12, 6),
    region: Optional[types.PatientRegions] = None,
    region_label: Optional[Dict[str, str]] = None,     # Gives 'regions' different names to those used for loading the data.
    **kwargs) -> None:
    regions = arg_to_list(region, str)
    region_labels = arg_to_list(region_label, str)

    # Load data.
    set = TrainingAdaptiveDataset(dataset)
    all_regions = set.list_regions()
    sample = set.sample(sample_idx)
    ct_data = sample.input
    spacing = sample.spacing
    label = sample.label
    if regions is not None:
        region_data = {}
        for i, r in enumerate(all_regions):
            if r not in regions or not sample.has_region(r):
                continue
            region_data[r] = label[i + 1]
    else:
        region_data = None

    if centre_of is not None:
        if type(centre_of) == str:
            if region_data is None or centre_of not in region_data:
                region_idx = all_regions.index(centre_of) + 1
                centre_of = label[region_idx]

    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                region_idx = all_regions.index(crop) + 1
                crop = label[region_idx]

    if region_labels is not None:
        # Rename 'regions' and 'region_data' keys.
        regions = [region_labels[r] if r in region_labels else r for r in regions]
        for old, new in region_labels.items():
            region_data[new] = region_data.pop(old)

        # Rename 'centre_of' and 'crop' keys.
        if type(centre_of) == str and centre_of in region_labels:
            centre_of = region_labels[centre_of] 
        if type(crop) == str and crop in region_labels:
            crop = region_labels[crop]

    # Plot both scans.
    _, axs = plt.subplots(1, 2, figsize=figsize)
    ct_data_0 = ct_data[0]
    ct_data_1 = ct_data[1]
    if regions is not None:
        region_data_1 = {}
        for i, r in enumerate(all_regions):
            if r not in regions or not sample.has_input_region(r):
                continue
            region_data_1[r] = ct_data[i + 2].astype(np.bool_)
    else:
        region_data_1 = None
    plot_region_base(sample_idx, ct_data_1.shape, spacing, ax=axs[0], centre_of=centre_of, crop=crop, ct_data=ct_data_1, region_data=region_data_1, **kwargs)
    plot_region_base(sample_idx, ct_data_0.shape, spacing, ax=axs[1], centre_of=centre_of, crop=crop, ct_data=ct_data_0, region_data=region_data, **kwargs)

def plot_sample_localiser_prediction(
    dataset: str,
    sample_idx: str,
    region: str,
    localiser: types.ModelName,
    **kwargs) -> None:
    # Load data.
    sample = ds.get(dataset, 'training').sample(sample_idx)
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
    input = ds.get(dataset, 'training').sample(sample_idx).input
    
    # Plot distribution.
    plot_distribution(input, figsize=figsize, range=range, resolution=resolution)

import numpy as np
import re
from typing import Dict, List, Literal, Optional, Union

from mymi.datasets import NrrdDataset
from mymi.gradcam.dataset.nrrd import load_multi_segmenter_heatmap
from mymi import logging
from mymi.predictions.datasets.nrrd import create_segmenter_prediction, load_segmenter_predictions
from mymi.regions import regions_to_list
from mymi.typing import Box2D, ImageSpacing3D, ModelName, PatientRegions
from mymi.utils import arg_broadcast, arg_to_list

from .plotting import *

MODEL_SELECT_PATTERN = r'^model:([0-9]+)$'
MODEL_SELECT_PATTERN_MULTI = r'^model(:([0-9]+))?:([a-zA-Z_]+)$'

def plot_histogram(
    dataset: str,
    n_patients: int = 10) -> None:
    set = NrrdDataset(dataset)
    pat_ids = set.list_patients()
    pat_ids = pat_ids[:n_patients]
    ct_datas = [set.patient(pat_id).ct_data for pat_id in pat_ids]
    plot_histogram_base(ct_datas)

def plot_patients(
    dataset: str,
    pat_id: str,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    labels: Literal['included', 'excluded', 'all'] = 'all',
    regions: Optional[PatientRegions] = None,
    region_labels: Dict[str, str] = {},
    show_dose: bool = False,
    **kwargs) -> None:

    # Load data.
    set = NrrdDataset(dataset)
    pat = set.patient(pat_id)
    ct_data = pat.ct_data
    if regions is not None:
        region_data = pat.region_data(labels=labels, regions=regions)
    else:
        region_data = None
    spacing = pat.ct_spacing
    dose_data = pat.dose_data if show_dose else None

    if centre is not None:
        if type(centre) == str:
            if region_data is None or centre not in region_data:
                centre = pat.region_data(regions=centre)[centre]

    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                crop = pat.region_data(regions=crop)[crop]
    
    # Apply region labels.
    region_data, centre, crop = apply_region_labels(region_labels, region_data, centre, crop)

    # Plot.
    plot_patient_base(pat_id, ct_data.shape, spacing, centre=centre, crop=crop, ct_data=ct_data, dose_data=dose_data, region_data=region_data, **kwargs)

def plot_heatmap(
    dataset: str,
    pat_id: str,
    model: ModelName,
    region: str,
    layer: int,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    **kwargs) -> None:
    # Load data.
    set = NrrdDataset(dataset)
    pat = set.patient(pat_id)
    ct_data = pat.ct_data
    region_data = pat.region_data(region=region)
    spacing = pat.ct_spacing

    # Load heatmap.
    heatmap = load_multi_segmenter_heatmap(dataset, pat_id, model, region, layer)

    if centre is not None:
        centre = pat.region_data(region=centre)[centre]

    if type(crop) == str:
        crop = pat.region_data(region=crop)[crop]
    
    # Plot.
    plot_heatmap_base(heatmap, spacing, centre=centre, crop=crop, ct_data=ct_data, region_data=region_data, **kwargs)

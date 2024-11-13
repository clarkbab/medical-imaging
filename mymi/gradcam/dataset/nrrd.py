import numpy as np
import os
import torch
from typing import List, Union

from mymi import config
from mymi.dataset import NrrdDataset
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.types import ImageSpacing3D, ModelName, PatientIDs, PatientRegions
from mymi.utils import arg_broadcast, arg_to_list

from ..gradcam import get_multi_segmenter_heatmap as get_multi_segmenter_heatmap_base

def create_heatmap(
    dataset: str,
    pat_id: str,
    model: ModelName,
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    region: str,
    layer: Union[str, List[str]],
    layer_spacing: Union[ImageSpacing3D, List[ImageSpacing3D]],
    device: torch.device = torch.device('cpu'),
    **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
    logging.arg_log('Creating heatmap', ('dataset', 'pat_id', 'model', 'model_region', 'model_spacing', 'region', 'layer', 'layer_spacing'), (dataset, pat_id, model, model_region, model_spacing, region, layer, layer_spacing))
    pat_ids = arg_to_list(pat_id, str)
    datasets = arg_broadcast(dataset, pat_ids, str)
    model = replace_ckpt_alias(model)
    layers = arg_to_list(layer, str)

    # Load GPU if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    # Get heatmaps.
    for dataset, pat_id in zip(datasets, pat_ids):
        heatmap = get_heatmap(dataset, pat_id, model, model_region, model_spacing, region, layer, layer_spacing, device=device, **kwargs)

        # Save heatmaps.
        heatmaps = arg_to_list(heatmap, np.ndarray)
        logging.info(f"got {len(heatmap)} heatmaps")
        for layer, heatmap in zip(layers, heatmaps):
            logging.info(f"saving heatmap for layer {layer}")
            filepath = os.path.join(config.directories.heatmaps, dataset, pat_id, *model, f'{region}-layer-{layer}.npz')
            logging.info(f"filepath: {filepath}")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, data=heatmap)

def get_heatmap(
    dataset: str,
    pat_id: str,
    model: ModelName,
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    region: str,
    layer: Union[str, List[str]],
    layer_spacing: Union[ImageSpacing3D, List[ImageSpacing3D]],
    device: torch.device = torch.device('cpu'),
    **kwargs) -> Union[np.ndarray, List[np.ndarray]]:

    # Load patient CT data and spacing.
    set = NrrdDataset(dataset)
    patient = set.patient(pat_id)
    input = patient.ct_data
    input_spacing = patient.ct_spacing
    label = patient.region_data(region=region)[region]

    # Call base method.
    return get_multi_segmenter_heatmap_base(input, input_spacing, label, model, model_region, model_spacing, region, layer, layer_spacing, device=device, **kwargs)

def load_multi_segmenter_heatmap(
    dataset: str,
    pat_id: str,
    model: ModelName,
    region: str,
    layer: Union[str, List[str]]) -> Union[np.array, List[np.ndarray]]:
    model = replace_ckpt_alias(model)
    layers = arg_to_list(layer, str)
    heatmaps = []
    for layer in layers:
        filepath = os.path.join(config.directories.heatmaps, dataset, pat_id, *model, f'{region}-layer-{layer}.npz')
        heatmap = np.load(filepath)['data']
        heatmaps.append(heatmap)

    if len(heatmaps) == 1:
        return heatmaps[0]
    else:
        return heatmaps

import numpy as np
import os
import torch
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm

from mymi import config
from mymi.dataset import NIFTIDataset
from mymi.loaders import MultiLoader
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.models.systems import MultiSegmenter
from mymi.prediction.dataset.nifti import load_localiser_prediction
from mymi.regions import region_to_list
from mymi.types import ImageSpacing3D, Model, ModelName, PatientRegion, PatientRegions
from mymi.utils import arg_broadcast, arg_to_list

from ..gradcam import get_multi_segmenter_heatmap as get_multi_segmenter_heatmap_base

def create_multi_segmenter_heatmap(
    dataset: str,
    pat_id: str,
    model: Union[Model, ModelName],
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    target_region: str,
    layer: Union[str, List[str]],
    layer_spacing: Union[ImageSpacing3D, List[ImageSpacing3D]],
    device: torch.device = torch.device('cpu'),
    **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
    model_name = model if isinstance(model, tuple) else model.name
    logging.arg_log('Creating heatmap', ('dataset', 'pat_id', 'model', 'model_region', 'model_spacing', 'target_region', 'layer', 'layer_spacing'), (dataset, pat_id, model_name, model_region, model_spacing, target_region, layer, layer_spacing))
    pat_ids = arg_to_list(pat_id, (int, str), out_type=str)
    datasets = arg_broadcast(dataset, pat_ids, str)

    # Load GPU if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    # Load model.
    if isinstance(model, tuple):
        logging.info('loading model')
        model = MultiSegmenter.load(model, region=model_region, **kwargs)
        model.eval()
        model.to(device)

    # Get heatmaps.
    for dataset, pat_id in zip(datasets, pat_ids):
        heatmap = get_multi_segmenter_heatmap(dataset, pat_id, model, model_region, model_spacing, target_region, layer, layer_spacing, device=device, **kwargs)

        # Save heatmaps.
        layers = arg_to_list(layer, str)
        heatmaps = arg_to_list(heatmap, np.ndarray)
        logging.info(f"got {len(heatmap)} heatmaps")
        for layer, heatmap in zip(layers, heatmaps):
            logging.info(f"saving heatmap for layer {layer}")
            filepath = os.path.join(config.directories.heatmaps, dataset, pat_id, *model_name, f'{target_region}-layer-{layer}.npz')
            logging.info(f"filepath: {filepath}")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, data=heatmap)

def create_multi_segmenter_heatmaps(
    dataset: Union[str, List[str]],
    model: Union[Model, ModelName],
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    target_region: PatientRegion,
    layer: Union[str, List[str]],
    layer_spacing: Union[ImageSpacing3D, List[ImageSpacing3D]],
    load_all_samples: bool = False,
    n_folds: Optional[int] = None,
    n_pats: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_split_file: bool = False,
    **kwargs: Dict[str, Any]) -> None:
    logging.arg_log('Creating heatmaps', ('dataset', 'model', 'model_region', 'model_spacing', 'target_region', 'layer', 'layer_spacing'), (dataset, model, model_region, model_spacing, target_region, layer, layer_spacing))

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Load model.
    if isinstance(model, tuple):
        logging.info('loading model')
        model = MultiSegmenter.load(model, region=model_region, **kwargs)
        model.eval()
        model.to(device)

    # Create test loader.
    _, _, test_loader = MultiLoader.build_loaders(dataset, load_all_samples=load_all_samples, n_folds=n_folds, region=model_region, test_fold=test_fold, use_split_file=use_loader_split_file) 

    # Make predictions.
    n_pat_count = 0
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            create_multi_segmenter_heatmap(dataset, pat_id, model, model_region, model_spacing, target_region, layer, layer_spacing, device=device, **kwargs)
            n_pat_count += 1
            if n_pats is not None and n_pat_count >= n_pats:
                return

def get_multi_segmenter_heatmap(
    dataset: str,
    pat_id: str,
    model: Union[Model, ModelName],
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    target_region: str,
    layer: Union[str, List[str]],
    layer_spacing: Union[ImageSpacing3D, List[ImageSpacing3D]],
    device: torch.device = torch.device('cpu'),
    use_crop: str = 'brain',
    **kwargs) -> Union[np.ndarray, List[np.ndarray]]:

    # Load model.
    if isinstance(model, tuple):
        logging.info('loading model')
        model = MultiSegmenter.load(model, region=model_region, **kwargs)
        model.eval()
        model.to(device)

    # Load patient CT data and spacing.
    set = NIFTIDataset(dataset)
    patient = set.patient(pat_id)
    input = patient.ct_data
    input_spacing = patient.ct_spacing

    # Get brain label if required.
    if use_crop == 'brain':
        localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
        brain_label = load_localiser_prediction(dataset, pat_id, localiser)
    else:
        brain_label = None

    # Call base method.
    id = f'{dataset}:{pat_id}'
    return get_multi_segmenter_heatmap_base(input, input_spacing, model, model_region, model_spacing, target_region, layer, layer_spacing, brain_label=brain_label, device=device, id=id, **kwargs)

def load_multi_segmenter_heatmap(
    dataset: str,
    pat_id: str,
    model: ModelName,
    target_region: str,
    layer: Union[str, List[str]],
    exists_only: bool = False) -> Union[np.array, List[np.ndarray]]:
    model = replace_ckpt_alias(model)
    layers = arg_to_list(layer, str)
    heatmaps = []
    for layer in layers:
        filepath = os.path.join(config.directories.heatmaps, dataset, pat_id, *model, f'{target_region}-layer-{layer}.npz')
        if not os.path.exists(filepath):
            if exists_only:
                return False
            else:
                raise ValueError(f"Heatmap not found for dataset '{dataset}', patient '{pat_id}', model '{model}', target_region '{target_region}', layer '{layer}'. Filepath: {filepath}")

        if not exists_only:
            heatmap = np.load(filepath)['data']
            heatmaps.append(heatmap)

    if exists_only:
        return True

    if len(heatmaps) == 1:
        return heatmaps[0]
    else:
        return heatmaps

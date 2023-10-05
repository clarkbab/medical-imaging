import numpy as np
import os
import torch
from typing import List, Union

from mymi import config
from mymi.dataset import NRRDDataset
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.models.systems import MultiSegmenter
from mymi.transforms import centre_crop_3D, centre_pad_3D, crop_or_pad_3D, resample_3D
from mymi.types import ImageSpacing3D, ModelName, PatientRegions
from mymi.utils import arg_to_list

def create_heatmap(
    dataset: str,
    pat_id: str,
    model: ModelName,
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    region: str,
    layer: Union[str, List[str]],
    device: torch.device = torch.device('cpu'),
    n_epochs: int = 5000) -> np.array:
    # Get heatmaps.
    layers = arg_to_list(layer, str)
    heatmap = get_heatmap(dataset, pat_id, model, model_region, model_spacing, region, layer, device=device, n_epochs=n_epochs)

    # Save heatmaps.
    heatmaps = arg_to_list(heatmap, np.ndarray)
    for layer, heatmap in zip(layers, heatmaps):
        filepath = os.path.join(config.directories.heatmaps, dataset, pat_id, *model, f'{region}-layer-{layer}.npz')
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
    device: torch.device = torch.device('cpu'),
    n_epochs: int = 5000) -> Union[np.ndarray, List[np.ndarray]]:
    layers = arg_to_list(layer, str)
    model = replace_ckpt_alias(model)
    model_regions = arg_to_list(model_region, str)
    region_channel = model_regions.index(region) + 1
    logging.info(f"Creating heatmap for model '{model}', region '{region}', for layers '{layers}'.")

    # Load model.
    model = MultiSegmenter.load(model, n_epochs=n_epochs, region=model_region)
    model.eval()
    model.to(device)

    # Register hooks.
    activations = {}
    gradients = {}

    def get_activations(layer: str) -> None:
        def hook(model, input, output):
            activations[layer] = output.detach().cpu().numpy()
        return hook

    def get_gradients(layer: str) -> None:
        def hook(model, input, outputs):
            output = outputs[0]
            gradients[layer] = output.detach().cpu().numpy()
        return hook

    # layer_names = ['5', '12', '19', '26', '33']
    layer_spacings = [tuple(np.array(model_spacing) * 2 ** i) for i in range(len(layers))]

    for layer_name in layers:
        layer = model.network.layers._modules[layer_name].layer
        layer.register_forward_hook(get_activations(layer_name))
        layer.register_full_backward_hook(get_gradients(layer_name))

    # Load patient CT data and spacing.
    set = NRRDDataset(dataset)
    patient = set.patient(pat_id)
    input = patient.ct_data
    input_spacing = patient.ct_spacing
    label = patient.region_data(region=region)[region]

    # Resample input to model spacing.
    input_size = input.shape
    input = resample_3D(input, spacing=input_spacing, output_spacing=model_spacing) 
    label = resample_3D(label, spacing=input_spacing, output_spacing=model_spacing) 

    # Apply 'naive' cropping.
    # crop_mm = (320, 520, 730)   # With 60 mm margin (30 mm either end) for each axis.
    crop_mm = (250, 400, 500)   # With 60 mm margin (30 mm either end) for each axis.
    crop = tuple(np.round(np.array(crop_mm) / model_spacing).astype(int))
    resampled_input_size = input.shape
    input = centre_crop_3D(input, crop)
    label = centre_crop_3D(label, crop)

    # Pass image to model.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    pred = model(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Sum all foreground voxels for OAR or interest.
    region_pred = pred[region_channel]
    y = region_pred[label].sum()

    # Perform backward pass.
    y.backward()

    # Get heatmaps.
    heatmaps = []
    for layer_name, spacing in zip(layers, layer_spacings):
        layer_activations = activations[layer_name]
        layer_gradients = gradients[layer_name]

        # Calculate weightings.
        n_channels = layer_activations.shape[1]
        n_elements = layer_activations[0, 0].size
        weights = layer_gradients.mean(axis=(0, 2, 3, 4)) / n_elements
        weights = weights.reshape(1, n_channels, 1, 1, 1)

        # Create heatmap.
        heatmap = (weights * layer_activations).sum(axis=(0, 1))
        heatmap = np.maximum(heatmap, 0)

        # Resample to input spacing.
        heatmap = resample_3D(heatmap, spacing=spacing, output_spacing=model_spacing) 

        # Crop/pad to the resampled size, i.e. before 'naive' cropping.
        heatmap = centre_pad_3D(heatmap, resampled_input_size)

        # Resample to original spacing.
        heatmap = resample_3D(heatmap, spacing=model_spacing, output_spacing=input_spacing)
        # Resampling rounds *up* to nearest number of voxels, cropping may be necessary to obtain original image size.
        crop_box = ((0, 0, 0), input_size)
        heatmap = crop_or_pad_3D(heatmap, crop_box)

        # Save heatmaps. 
        heatmaps.append(heatmap)

    if len(heatmaps) == 0:
        return heatmaps[0]
    else:
        return heatmaps

def load_heatmap(
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

    return heatmaps

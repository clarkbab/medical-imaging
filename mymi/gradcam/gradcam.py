import numpy as np
import os
import torch
from typing import List, Optional, Union

from mymi import config
from mymi.dataset import NRRDDataset
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.models.systems import MultiSegmenter
from mymi.transforms import centre_crop_3D, centre_pad_3D, crop_or_pad_3D, resample_3D
from mymi.types import ImageSpacing3D, ModelName, PatientIDs, PatientRegions
from mymi.utils import arg_to_list

def get_heatmap(
    input: np.ndarray,
    input_spacing: ImageSpacing3D,
    label: np.ndarray,
    model: ModelName,
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    region: str,
    layer: Union[str, List[str]],
    layer_spacing: Union[ImageSpacing3D, List[ImageSpacing3D]],
    device: torch.device = torch.device('cpu'),
    **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
    layers = arg_to_list(layer, str)
    layer_spacings = arg_to_list(layer_spacing, tuple)
    if len(layers) != len(layer_spacings):
        raise ValueError(f"'layer' and 'layer_spacings' must have same number of elements. Got {len(layers)} and {len(layer_spacings)} respectively.")
    model = replace_ckpt_alias(model)
    model_regions = arg_to_list(model_region, str)
    region_channel = model_regions.index(region) + 1

    # Load model.
    logging.info('loading model')
    model = MultiSegmenter.load(model, region=model_region, **kwargs)
    model.eval()
    model.to(device)

    # Register hooks.
    logging.info('registering hooks')
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

    for layer_name in layers:
        layer = model.network.layers._modules[layer_name].layer
        layer.register_forward_hook(get_activations(layer_name))
        layer.register_full_backward_hook(get_gradients(layer_name))

    # Resample input to model spacing.
    logging.info('resampling input')
    input_size = input.shape
    input = resample_3D(input, spacing=input_spacing, output_spacing=model_spacing) 
    label = resample_3D(label, spacing=input_spacing, output_spacing=model_spacing) 

    # Apply 'naive' cropping.
    logging.info('naive cropping')
    # crop_mm = (320, 520, 730)   # With 60 mm margin (30 mm either end) for each axis.
    crop_mm = (250, 400, 500)   # With 60 mm margin (30 mm either end) for each axis.
    crop = tuple(np.round(np.array(crop_mm) / model_spacing).astype(int))
    resampled_input_size = input.shape
    input = centre_crop_3D(input, crop)
    label = centre_crop_3D(label, crop)

    # TODO: remove.
    filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{region}-input.npz')
    np.savez_compressed(filepath, data=input)

    # Pass image to model.
    logging.info('forward pass')
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
    logging.info('backward pass')
    y.backward()

    # Get heatmaps.
    logging.info('creating heatmaps')
    heatmaps = []
    for layer, spacing in zip(layers, layer_spacings):
        layer_activations = activations[layer]
        layer_gradients = gradients[layer]

        # TODO: remove.
        filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{region}-layer-{layer}-activations.npz')
        np.savez_compressed(filepath, data=layer_activations)
        filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{region}-layer-{layer}-gradients.npz')
        np.savez_compressed(filepath, data=layer_gradients)

        # Calculate weightings.
        n_channels = layer_activations.shape[1]
        n_elements = layer_activations[0, 0].size
        weights = layer_gradients.mean(axis=(0, 2, 3, 4)) / n_elements  # Average over all but 'channel' axis.
        weights = weights.reshape(1, n_channels, 1, 1, 1)   # Reshape for broadcasting.

        # Create heatmap.
        heatmap = (weights * layer_activations).sum(axis=(0, 1))    # Apply weighted sum of channels.
        heatmap = np.maximum(heatmap, 0)    # Apply ReLU.

        # TODO: remove.
        filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{region}-layer-{layer}-raw.npz')
        np.savez_compressed(filepath, data=heatmap)

        # Resample to input spacing.
        logging.info(f"resampling from {spacing} to {model_spacing}")
        heatmap = resample_3D(heatmap, spacing=spacing, output_spacing=model_spacing) 

        # TODO: remove.
        filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{region}-layer-{layer}-input.npz')
        np.savez_compressed(filepath, data=heatmap)

        # Crop/pad to the resampled size, i.e. before 'naive' cropping.
        heatmap = centre_pad_3D(heatmap, resampled_input_size)

        # TODO: remove.
        filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{region}-layer-{layer}-input-crop.npz')
        np.savez_compressed(filepath, data=heatmap)

        # Resample to original spacing.
        heatmap = resample_3D(heatmap, spacing=model_spacing, output_spacing=input_spacing)

        # TODO: remove.
        filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{region}-layer-{layer}-native.npz')
        np.savez_compressed(filepath, data=heatmap)

        # Resampling rounds *up* to nearest number of voxels, cropping may be necessary to obtain original image size.
        crop_box = ((0, 0, 0), input_size)
        heatmap = crop_or_pad_3D(heatmap, crop_box)

        # TODO: remove.
        filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{region}-layer-{layer}-native-crop.npz')
        np.savez_compressed(filepath, data=heatmap)

        heatmaps.append(heatmap)

    if len(heatmaps) == 0:
        return heatmaps[0]
    else:
        return heatmaps

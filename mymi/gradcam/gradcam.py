import numpy as np
import os
import torch
from torch.nn.functional import one_hot
from typing import List, Optional, Union
from tqdm import tqdm

from mymi.geometry import get_extent
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.models.systems import MultiSegmenter
from mymi.regions import region_to_list
from mymi.transforms import centre_crop_3D, centre_pad_3D, crop_3D, pad_3D, resample
from mymi.types import ImageSpacing3D, Model, ModelName, PatientRegions
from mymi.utils import arg_to_list

def get_multi_segmenter_heatmap(
    input: np.ndarray,
    input_spacing: ImageSpacing3D,
    model: Union[Model, ModelName],
    model_region: PatientRegions,
    model_spacing: ImageSpacing3D,
    target_region: str,
    layer: Union[str, List[str]],
    layer_spacing: Union[ImageSpacing3D, List[ImageSpacing3D]],
    brain_label: Optional[np.ndarray] = None,
    device: torch.device = torch.device('cpu'),
    heatmap_fill: float = -1,
    save_tmp_files: bool = False,
    use_crop: str = 'brain',
    **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
    layers = arg_to_list(layer, str)
    layer_spacings = arg_to_list(layer_spacing, tuple)
    if len(layers) != len(layer_spacings):
        raise ValueError(f"'layer' and 'layer_spacings' must have same number of elements. Got {len(layers)} and {len(layer_spacings)} respectively.")
    model_regions = region_to_list(model_region)
    target_channel = model_regions.index(target_region) + 1

    # Load model.
    if isinstance(model, tuple):
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
    input = resample(input, spacing=input_spacing, output_spacing=model_spacing) 
    input_size_after_resample = input.shape

    if use_crop == 'naive':
        # Apply 'naive' cropping.
        logging.info('naive cropping')
        # This value used for MICCAI-2015 multi-segmenter only.
        crop_mm = (250, 400, 500)   # With 60 mm margin (30 mm either end) for each axis.
        crop = tuple(np.round(np.array(crop_mm) / model_spacing).astype(int))
        input = centre_crop_3D(input, crop)
    elif use_crop == 'brain':
        assert brain_label is not None
        # Convert to voxel crop.
        # This value used for PMCC-HN-TEST/TRAIN multi-segmenter only.
        crop_mm = (300, 400, 500)
        crop_voxels = tuple((np.array(crop_mm) / np.array(model_spacing)).astype(np.int32))

        # Get brain extent.
        brain_label = resample(brain_label, spacing=input_spacing, output_spacing=model_spacing)
        brain_extent = get_extent(brain_label)

        # Get crop coordinates.
        # Crop origin is centre-of-extent in x/y, and max-extent in z.
        # Cropping boundary extends from origin equally in +/- directions for x/y, and extends
        # in - direction for z.
        p_above_brain = 0.04
        crop_origin = ((brain_extent[0][0] + brain_extent[1][0]) // 2, (brain_extent[0][1] + brain_extent[1][1]) // 2, brain_extent[1][2])
        crop = (
            (int(crop_origin[0] - crop_voxels[0] // 2), int(crop_origin[1] - crop_voxels[1] // 2), int(crop_origin[2] - int(crop_voxels[2] * (1 - p_above_brain)))),
            (int(np.ceil(crop_origin[0] + crop_voxels[0] / 2)), int(np.ceil(crop_origin[1] + crop_voxels[1] / 2)), int(crop_origin[2] + int(crop_voxels[2] * p_above_brain)))
        )

        # Crop input.
        input = crop_3D(input, crop)
    else:
        raise ValueError(f"Unknown 'use_crop' value '{use_crop}'.")

    # TODO: remove.
    if save_tmp_files:
        filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{model.name[1]}-{kwargs["id"]}-{target_region}-input.npz')
        np.savez_compressed(filepath, data=input)

    # Pass image to model.
    logging.info('forward pass')
    input_size_model = input.shape
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    pred = model(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # TODO: remove.
    if save_tmp_files:
        filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{model.name[1]}-{kwargs["id"]}-{target_region}-pred.npz')
        np.savez_compressed(filepath, data=pred.detach().cpu().numpy())

    # Sum all foreground voxels for OAR of interest.
    pred_region = pred[target_channel]
    pred_bin = pred.argmax(dim=0)
    pred_bin = one_hot(pred_bin, num_classes=len(model_regions) + 1)
    pred_bin = pred_bin.moveaxis(-1, 0)
    pred_bin = pred_bin.type(torch.bool)
    pred_region_bin = pred_bin[target_channel]

    # TODO: remove.
    if save_tmp_files:
        filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{model.name[1]}-{kwargs["id"]}-{target_region}-pred-region-bin.npz')
        np.savez_compressed(filepath, data=pred_region_bin.detach().cpu().numpy())
    y = pred_region[pred_region_bin].sum()

    # Perform backward pass.
    logging.info('backward pass')
    y.backward()

    # Get heatmaps.
    logging.info('creating heatmaps')
    heatmaps = []
    for layer, spacing in tqdm(zip(layers, layer_spacings)):
        layer_activations = activations[layer]
        layer_gradients = gradients[layer]

        # TODO: remove.
        if save_tmp_files:
            filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{model.name[1]}-{kwargs["id"]}-{target_region}-layer-{layer}-activations.npz')
            np.savez_compressed(filepath, data=layer_activations)
            filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{model.name[1]}-{kwargs["id"]}-{target_region}-layer-{layer}-gradients.npz')
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
        if save_tmp_files:
            filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{model.name[1]}-{kwargs["id"]}-{target_region}-layer-{layer}-raw.npz')
            np.savez_compressed(filepath, data=heatmap)

        # Resample to model spacing/size - use resampling output grid that mimics a reversal of the max-pooling operation.
        scaling = tuple(np.array(model_spacing) / spacing)
        output_origin = tuple(sp * (s - 1) / 2 for s, sp in zip(scaling, spacing))
        heatmap = resample(heatmap, fill=heatmap_fill, output_origin=output_origin, output_size=input_size_model, output_spacing=model_spacing, spacing=spacing) 

        # TODO: remove.
        if save_tmp_files:
            filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{model.name[1]}-{kwargs["id"]}-{target_region}-layer-{layer}-input.npz')
            np.savez_compressed(filepath, data=heatmap)

        # Reverse the 'naive' or 'brain' cropping.
        if use_crop == 'naive':
            heatmap = centre_pad_3D(heatmap, input_size_after_resample)
        elif use_crop == 'brain':
            pad_min = tuple(-np.array(crop[0]))
            pad_max = np.array(pad_min) + np.array(input_size_after_resample)
            pad = (pad_min, pad_max)
            # Fill with 'heatmap_fill' as we want to know where the heatmap edges are.
            # E.g. if it was cropped in the brain, we can exclude the 'heatmap_fill' values from our mean calculation.
            heatmap = pad_3D(heatmap, pad, fill=heatmap_fill)

        # TODO: remove.
        if save_tmp_files:
            filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{model.name[1]}-{kwargs["id"]}-{target_region}-layer-{layer}-input-size-before-crop.npz')
            np.savez_compressed(filepath, data=heatmap)

        # Resample to original spacing/size.
        heatmap = resample(heatmap, fill=heatmap_fill, output_size=input_size, output_spacing=input_spacing, spacing=model_spacing)

        # TODO: remove.
        if save_tmp_files:
            filepath = os.path.join('/data/gpfs/projects/punim1413/mymi/tmp/heatmaps', f'{model.name[1]}-{kwargs["id"]}-{target_region}-layer-{layer}-native.npz')
            np.savez_compressed(filepath, data=heatmap)

        heatmaps.append(heatmap)

    if len(heatmaps) == 0:
        return heatmaps[0]
    else:
        return heatmaps

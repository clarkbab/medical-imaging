from argparse import ArgumentError
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import center_of_mass as centre_of_mass
import torch
import torchio
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

from ..plotter import plot_localiser_prediction
from mymi import dataset as ds
from mymi.geometry import get_box, get_extent, get_extent_centre
from mymi.prediction.dataset.nifti import get_patient_localiser_prediction, load_patient_localiser_centre, load_patient_localiser_prediction, load_patient_segmenter_prediction
from mymi.regions import get_patch_size
from mymi.transforms import crop_or_pad_2D
from mymi import types

from ..plotter import assert_position, get_aspect_ratio, get_origin, get_slice, plot_box, plot_regions, reverse_box_coords_2D, should_plot_box

def plot_patient_segmenter_prediction(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    aspect: float = None,
    centre_of: Optional[str] = None,
    centre_of_label: bool = True,
    crop: types.Box2D = None,
    crop_margin: float = 40,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    latex: bool = False,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    patch_size: Optional[types.ImageSize3D] = None,
    show_extent: bool = True,
    show_loc_centre: bool = True,
    show_patch: bool = True,
    slice_idx: Optional[int] = None,
    truncate_spine: bool = True,
    view: types.PatientView = 'axial',
    **kwargs: dict) -> None:
    assert_position(centre_of, extent_of, slice_idx)

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Load sample.
    set = ds.get(dataset, 'nifti')
    patient = set.patient(pat_id)
    spacing = patient.ct_spacing()

    # Get localiser centre.
    if region != 'SpinalCord':
        truncate_spine = False
    loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser, truncate_spine=truncate_spine)

    # Load segmenter segmentation.
    pred = load_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter)
    non_empty_pred = False if pred.sum() == 0 else True

    # Centre on OAR if requested.
    if centre_of:
        if centre_of_label:
            # Get centre of label data.
            label = patient.region_data(regions=centre_of)[centre_of]
            com = np.round(centre_of_mass(label)).astype(int)
        else:
            # Get centre of segmentation data.
            com = np.round(centre_of_mass(pred)).astype(int)
        
        # Get 'slice_idx'.
        if view == 'axial':
            slice_idx = com[2]
        elif view == 'coronal':
            slice_idx = com[1]
        elif view == 'sagittal':
            slice_idx = com[0]

    # Plot patient regions.
    plot_patient_regions(dataset, pat_id, aspect=aspect, colours=['gold'], crop=crop, latex=latex, legend=False, legend_loc=legend_loc, regions=region, show=False, slice_idx=slice_idx, view=view, **kwargs)

    # Convert to box crop.
    crop_box = None
    if crop:
        if type(crop) == str:
            # Get crop box from region name.
            label = patient.region_data(regions=region)[region]
            crop_box = _get_region_crop_box(label, crop_margin, spacing, view)
        else:
            # Get crop box from API crop.
            crop_box = _convert_crop_to_box(crop)

    # Get extent.
    extent = get_extent(pred)

    # Plot prediction.
    if non_empty_pred:
        # Get aspect ratio.
        if not aspect:
            aspect = get_aspect_ratio(view, spacing) 

        # Get slice data.
        pred_slice_data = get_slice(pred, slice_idx, view)

        # Crop the image.
        if crop:
            pred_slice_data = crop_or_pad_2D(pred_slice_data, reverse_box_coords_2D(crop_box))

        # Plot prediction.
        colour = 'tomato'
        colours = [(1, 1, 1, 0), colour]
        cmap = ListedColormap(colours)
        plt.imshow(pred_slice_data, aspect=aspect, cmap=cmap, origin=get_origin(view))
        plt.plot(0, 0, c=colour, label='Segmentation')

    # Plot extent.
    if non_empty_pred and show_extent and should_plot_box(extent, view, slice_idx):
        plot_box(extent, view, colour='tomato', crop=crop_box, label='Seg. Extent')

    # Plot localiser centre.
    if non_empty_pred and show_loc_centre:
        if view == 'axial':
            centre = (loc_centre[0], loc_centre[1])
        elif view == 'coronal':
            centre = (loc_centre[0], loc_centre[2])
        elif view == 'sagittal':
            centre = (loc_centre[1], loc_centre[2])
        plt.scatter(*centre, c='royalblue', label='Loc. centre')

    # Plot patch.
    if show_patch:
        patch_size = get_patch_size(region, spacing)
        seg_patch = get_box(loc_centre, patch_size)
        if should_plot_box(seg_patch, view, slice_idx):
            plot_box(seg_patch, view, colour='deepskyblue', crop=crop_box, label='Seg. Patch', linestyle='dashed')

    # Show legend.
    plt_legend = plt.legend(loc=legend_loc, prop={'size': legend_size})
    for l in plt_legend.get_lines():
        l.set_linewidth(8)

    plt.show()

    # Revert latex settings.
    if latex:
        plt.rcParams.update({
            "font.family": rc_params['font.family'],
            'text.usetex': rc_params['text.usetex']
        })

def plot_patient_regions(
    dataset: str,
    pat_id: str,
    regions: types.PatientRegions = 'all',
    **kwargs) -> None:
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(regions=regions)
    spacing = patient.ct_spacing
    
    # Plot.
    plot_regions(pat_id, ct_data, region_data, spacing, regions=regions, **kwargs)

def plot_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4),
    load_prediction: bool = True,
    **kwargs) -> None:
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(regions=region)[region]
    spacing = patient.ct_spacing

    # Load prediction.
    if load_prediction:
        pred = load_patient_localiser_prediction(dataset, pat_id, localiser)
    else:
        # Set truncation if 'SpinalCord'.
        truncate = True if region == 'SpinalCord' else False

        # Make prediction.
        pred = get_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, truncate=truncate)
    
    # Plot.
    plot_localiser_prediction(pat_id, region, ct_data, region_data, spacing, pred, **kwargs)

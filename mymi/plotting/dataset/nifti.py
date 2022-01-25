from argparse import ArgumentError
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import center_of_mass as centre_of_mass
import torch
import torchio
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

from mymi import dataset as ds
from mymi.geometry import get_box, get_extent, get_extent_centre
from mymi import logging
from mymi.prediction.dataset.nifti import get_patient_segmenter_prediction, load_patient_localiser_centre, load_patient_localiser_prediction, load_patient_segmenter_prediction
from mymi.regions import get_patch_size
from mymi.transforms import crop_or_pad_2D
from mymi import types

from ..plotter import assert_position, get_aspect_ratio, get_origin, get_slice, plot_box, plot_regions, reverse_box_coords_2D, should_plot_box

def plot_patient_regions(
    dataset: str,
    pat_id: str,
    alpha: float = 0.2,
    aspect: float = None,
    axis = None,
    cca: bool = False,
    centre_of: Optional[str] = None,
    colours: Optional[List[str]] = None,
    crop: Optional[Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    crop_margin: float = 40,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    figsize: Tuple[int, int] = (12, 12),
    font_size: int = 10,
    latex: bool = False,
    legend: bool = True,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    other_ds: str = None,
    other_regions: Union[str, Sequence[str]] = 'all',
    perimeter: bool = True,
    postproc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    regions: Union[str, Sequence[str]] = 'all',
    show: bool = True,
    show_axis_ticks: bool = True,
    show_axis_xlabel: bool = True,
    show_axis_ylabel: bool = True,
    show_extent: bool = False,
    slice_idx: Optional[int] = None,
    title: Union[bool, str] = True,
    transform: torchio.transforms.Transform = None,
    view: types.PatientView = 'axial',
    window: Tuple[float, float] = (3000, 500)) -> None:
    assert_position(centre_of, extent_of, slice_idx)

    # Create plot figure/axis.
    if not axis:
        plt.figure(figsize=figsize)
        axis = plt.gca()

    # Update font size.
    plt.rcParams.update({
        'font.size': font_size
    })

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Load patient spacing.
    set = ds.get(dataset, 'nifti')
    pat = set.patient(pat_id)
    spacing = pat.ct_spacing()

    # Get slice index if requested OAR centre.
    if centre_of:
        # Get extent centre.
        label = pat.region_data(regions=centre_of)[centre_of]
        if postproc:
            label = postproc(label)
        extent_centre = get_extent_centre(label)

        # Set slice index.
        if view == 'axial':
            slice_idx = extent_centre[2]
        elif view == 'coronal':
            slice_idx = extent_centre[1]
        elif view == 'sagittal':
            slice_idx = extent_centre[0]
    elif extent_of:
        # Get extent.
        eo_region, eo_end = extent_of
        label = pat.region_data(regions=eo_region)[eo_region]
        if postproc:
            label = postproc(label)
        extent = get_extent(label)

        # Set slice index.
        if view == 'axial':
            slice_idx = extent[eo_end][2]
        elif view == 'coronal':
            slice_idx = extent[eo_end][1]
        elif view == 'sagittal':
            slice_idx = extent[eo_end][0]

    # Load CT data.
    ct_data = pat.ct_data()

    # Load region data.
    if regions is not None:
        region_data = pat.region_data(regions=regions)
        if postproc:
            region_data = dict(((r, postproc(d)) for r, d in region_data.items()))

        # Load other regions.
        if other_ds:
            other_ds = DICOMDataset(other_ds) 
            other_region_data = other_ds.patient(pat_id).region_data(clear_cache=clear_cache, regions=other_regions)

            if internal_regions:
                # Map to internal region names.
                other_region_data = dict((_to_internal_region(r, clear_cache=clear_cache), d) for r, d in other_region_data.items())

    # Get slice data.
    ct_slice_data = get_slice(ct_data, slice_idx, view)

    # Convert to box representation.
    crop_box = None
    if crop:
        if type(crop) == str:
            # Get crop box from region name.
            data = region_data[crop]
            crop_box = _get_region_crop_box(data, crop_margin, spacing, view)
        else:
            # Get crop box from API crop.
            crop_box = _convert_crop_to_box(crop)

    # Perform crop.
    if crop_box:
        # Convert crop to 2D box.
        ct_slice_data = crop_or_pad_2D(ct_slice_data, reverse_box_coords_2D(crop_box))

    # Only apply aspect ratio if no transforms are being presented otherwise
    # we might end up with skewed images.
    if not aspect:
        if transform:
            aspect = 1
        else:
            aspect = get_aspect_ratio(view, spacing) 

    # Determine plotting window.
    if window:
        width, level = window
        vmin = level - (width / 2)
        vmax = level + (width / 2)
    else:
        vmin, vmax = ct_data.min(), ct_data.max()

    # Plot CT data.
    axis.imshow(ct_slice_data, cmap='gray', aspect=aspect, interpolation='none', origin=get_origin(view), vmin=vmin, vmax=vmax)

    # Show axis labels.
    if show_axis_xlabel or show_axis_ylabel:
        if view == 'axial':
            spacing_x = spacing[0]
            spacing_y = spacing[1]
        elif view == 'coronal':
            spacing_x = spacing[0]
            spacing_y = spacing[2]
        elif view == 'sagittal':
            spacing_x = spacing[1]
            spacing_y = spacing[2]

        if show_axis_xlabel:
            axis.set_xlabel(f'voxel [@ {spacing_x:.3f} mm spacing]')
        if show_axis_ylabel:
            axis.set_ylabel(f'voxel [@ {spacing_y:.3f} mm spacing]')

    if regions:
        # Plot regions.
        show_legend = plot_regions(region_data, slice_idx, alpha, aspect, latex, perimeter, view, axis=axis, cca=cca, colours=colours, crop=crop_box, show_extent=show_extent)

        if other_ds:
            # Prepend other dataset name.
            other_region_data = dict((f"{r} - {other_ds.name}", d) for r, d in other_region_data.items())
 
            # Plot other regions.
            other_show_legend = plot_regions(other_region_data, slice_idx, alpha, aspect, crop, latex, perimeter, view)

        # Create legend.
        if legend and (show_legend or (other_ds and other_show_legend)): 
            plt_legend = axis.legend(loc=legend_loc, prop={'size': legend_size})
            for l in plt_legend.get_lines():
                l.set_linewidth(8)

    # Show axis markers.
    show_axes = 'on' if show_axis_ticks else 'off'
    axis.axis(show_axes)

    # Determine number of slices.
    if view == 'axial':
        num_slices = ct_data.shape[2]
    elif view == 'coronal':
        num_slices = ct_data.shape[1]
    elif view == 'sagittal':
        num_slices = ct_data.shape[0]

    # Add title.
    if title:
        if isinstance(title, str):
            title_text = title
        else:
            title_text = f"{pat} - {slice_idx}/{num_slices - 1} ({view})"

        # Escape text if using latex.
        if latex:
            title_text = _escape_latex(title_text)

        axis.set_title(title_text)

    if show:
        plt.show()

        # Revert latex settings.
        if latex:
            plt.rcParams.update({
                "font.family": rc_params['font.family'],
                'text.usetex': rc_params['text.usetex']
            })

def plot_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: Tuple[str, str, str],
    aspect: float = None,
    centre_of: Optional[str] = None,
    crop: types.Box2D = None,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    latex: bool = False,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    show_box: bool = False,
    show_centre: bool = True,
    show_extent: bool = False,
    show_patch: bool = False,
    show_seg: bool = False,
    slice_idx: Optional[int] = None,
    truncate_spine: bool = False,
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

    # Centre on OAR if requested.
    if slice_idx is None:
        # Load region data.
        label = patient.region_data(regions=centre_of)[centre_of]
        com = np.round(centre_of_mass(label)).astype(int)
        if view == 'axial':
            slice_idx = com[2]
        elif view == 'coronal':
            slice_idx = com[1]
        elif view == 'sagittal':
            slice_idx = com[0]

    # Plot patient regions.
    plot_patient_regions(dataset, pat_id, aspect=aspect, colours=['gold'], crop=crop, latex=latex, legend=False, legend_loc=legend_loc, regions=region, show=False, show_extent=show_extent, slice_idx=slice_idx, view=view, **kwargs)

    # Load localiser segmentation.
    if region != 'SpinalCord':
        truncate_spine = False
    pred = load_patient_localiser_prediction(dataset, pat_id, localiser, truncate_spine=truncate_spine)
    non_empty_pred = False if pred.sum() == 0 else True

    # Get extent and centre.
    extent = get_extent(pred)
    loc_centre = get_extent_centre(pred)

    # Plot prediction.
    if non_empty_pred and show_seg:
        # Get aspect ratio.
        if not aspect:
            aspect = get_aspect_ratio(view, spacing) 

        # Get slice data.
        pred_slice_data = get_slice(pred, slice_idx, view)

        # Crop the image.
        if crop:
            pred_slice_data = crop_or_pad_2D(pred_slice_data, reverse_box_coords_2D(crop))

        # Plot prediction.
        colour = 'deepskyblue'
        colours = [(1, 1, 1, 0), colour]
        cmap = ListedColormap(colours)
        plt.imshow(pred_slice_data, aspect=aspect, cmap=cmap, origin=get_origin(view))
        plt.plot(0, 0, c=colour, label='Loc. Prediction')

    # Plot localiser bounding box.
    if non_empty_pred and show_box and should_plot_box(extent, view, slice_idx):
        plot_box(extent, view, colour='deepskyblue', crop=crop, label='Loc. Box')

    # Plot localiser centre.
    if non_empty_pred and show_centre:
        if view == 'axial':
            centre = (loc_centre[0], loc_centre[1])
        elif view == 'coronal':
            centre = (loc_centre[0], loc_centre[2])
        elif view == 'sagittal':
            centre = (loc_centre[1], loc_centre[2])
        plt.scatter(*centre, c='royalblue', label='Loc. Centre')

    # Plot second stage patch.
    if non_empty_pred and show_patch:
        size = get_patch_size(region, spacing)
        min, max = get_box(loc_centre, size)

        # Squash min/max to label size.
        min = np.clip(min, a_min=0, a_max=None)
        for i in range(len(max)):
            pred_max = pred.shape[i] - 1
            if max[i] > pred_max: 
                max[i] = pred_max

        if should_plot_box((min, max), view, slice_idx):
            plot_box((min, max), view, colour='tomato', crop=crop, label='Seg. Patch', linestyle='dashed')

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
    truncate_spine: bool = False,
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

def _convert_crop_to_box(crop: Tuple[Tuple[int, int], Tuple[int, int]]) -> types.Box2D:
    return ((crop[0][0], crop[1][0]), (crop[0][1], crop[1][1]))

def _get_region_crop_box(
    data: np.ndarray,
    crop_margin: float,
    spacing: types.ImageSpacing3D,
    view: types.PatientView) -> types.Box2D:
    # Get 3D crop box.
    extent = get_extent(data)

    # Add crop margin.
    crop_margin_vox = tuple(np.ceil(np.array(crop_margin) / spacing).astype(int))
    min, max = extent
    min = tuple(np.array(min) - crop_margin_vox)
    max = tuple(np.array(max) + crop_margin_vox)

    # Select 2D component.
    if view == 'axial':
        min = (min[0], min[1])
        max = (max[0], max[1])
    elif view == 'coronal':
        min = (min[0], min[2])
        max = (max[0], max[2])
    elif view == 'sagittal':
        min = (min[1], min[2])
        max = (max[1], max[2])
    crop = (min, max)
    return crop

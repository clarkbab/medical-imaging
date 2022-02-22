from audioop import reverse
from functools import reduce
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import torch
from torch import nn
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from mymi import dataset
from mymi.geometry import get_box, get_extent, get_extent_centre
from mymi import logging
from mymi.postprocessing import get_largest_cc
from mymi.regions import get_patch_size, is_region, RegionColours
from mymi.transforms import crop_or_pad_2D, crop_or_pad_box, crop_or_pad_point
from mymi import types

def _plot_region_data(
    region_data: Dict[str, np.ndarray],
    slice_idx: int,
    alpha: float,
    aspect: float,
    latex: bool,
    perimeter: bool,
    view: types.PatientView,
    axis = None,
    cca: bool = False,
    connected_extent: bool = False,
    crop: Optional[types.Box2D] = None,
    colours: Optional[List[str]] = None,
    show_extent: bool = False) -> bool:
    """
    effect: adds regions to the plot.
    returns: whether the legend should be shown.
    args:
        region_data: the region data to plot.
        others: see 'plot_patient_regions'.
    """
    if not axis:
        axis = plt.gca()

    # Plot each region.
    show_legend = False
    for i, (region, data) in enumerate(region_data.items()):
        # Get region colour.
        if colours:
            assert len(colours) == len(region_data.keys())
            colour = colours[i]
        else:
            colour = getattr(RegionColours, region)
        cols = [(1.0, 1.0, 1.0, 0), colour]
        cmap = ListedColormap(cols)

        # Convert data to 'imshow' co-ordinate system.
        slice_data = get_slice(data, slice_idx, view)

        # Crop image.
        if crop:
            slice_data = crop_or_pad_2D(slice_data, reverse_box_coords_2D(crop))

        # Plot extent.
        if show_extent:
            extent = get_extent(data)
            if should_plot_box(extent, view, slice_idx):
                show_legend = True
                plot_box_slice(extent, view, colour=colour, crop=crop, label=f'{region} Extent', linestyle='dashed')

        # Plot connected extent.
        if connected_extent:
            extent = get_extent(get_largest_cc(data))
            if should_plot_box(extent, view, slice_idx):
                plot_box_slice(extent, view, colour='b', crop=crop, label=f'{region} conn. extent', linestyle='dashed')

        # Skip region if not present on this slice.
        if slice_data.max() == 0:
            continue
        else:
            show_legend = True

        # Get largest component.
        if cca:
            slice_data = get_largest_cc(slice_data)

        # Plot region.
        axis.imshow(slice_data, alpha=alpha, aspect=aspect, cmap=cmap, interpolation='none', origin=get_origin(view))
        label = _escape_latex(region) if latex else region
        axis.plot(0, 0, c=colour, label=label)
        if perimeter:
            axis.contour(slice_data, colors=[colour], levels=[.5])

        # Set ticks.
        if crop:
            min, max = crop
            width = tuple(np.array(max) - min)
            xticks = np.linspace(0, 10 * np.floor(width[0] / 10), 5).astype(int)
            xtick_labels = xticks + min[0]
            axis.set_xticks(xticks)
            axis.set_xticklabels(xtick_labels)
            yticks = np.linspace(0, 10 * np.floor(width[1] / 10), 5).astype(int)
            ytick_labels = yticks + min[1]
            axis.set_yticks(yticks)
            axis.set_yticklabels(ytick_labels)

    return show_legend

def _to_image_coords(
    data: ndarray,
    view: types.PatientView) -> ndarray:
    """
    returns: data in correct orientation for viewing.
    args:
        data: the data to orient.
        view: the viewing axis.
    """
    # 'plt.imshow' expects (y, x).
    data = np.transpose(data)

    return data

def get_origin(view: types.PatientView) -> str:
    """
    returns: whether to place image origin in lower or upper corner of plot.
    args:
        view: the viewing plane.
    """
    # Get origin.
    if view == 'axial':
        origin = 'upper'
    else:
        origin = 'lower'

    return origin
    
def _to_internal_region(
    region: str,
    clear_cache: bool = False) -> str:
    """
    returns: the internal region name.
    args:
        region: the dataset region name.
    kwargs:
        clear_cache: force the cache to clear.
    """
    # Check if region is an internal name.
    if is_region(region):
        return region

    # Map from dataset name to internal name.
    map_df = dataset.region_map(clear_cache=clear_cache)
    map_dict = dict((r.dataset, r.internal) for _, r in map_df.iterrows())
    if region in map_dict:
        return map_dict[region]

    # Raise an error if we don't know how to translate to the internal name.
    raise ValueError(f"Region '{region}' is neither an internal region, nor listed in the region map, can't create internal name.")

def get_slice(
    data: np.ndarray,
    slice_idx: int,
    view: types.PatientView) -> np.ndarray:
    # Check that slice index isn't too large.
    if view == 'axial' and (slice_idx >= data.shape[2]):
        raise ValueError(f"Slice '{slice_idx}' out of bounds, only '{data.shape[2]}' axial slices.")
    elif view == 'coronal' and (slice_idx >= data.shape[1]):
        raise ValueError(f"Slice '{slice_idx}' out of bounds, only '{data.shape[1]}' coronal slices.")
    elif view == 'sagittal' and (slice_idx >= data.shape[0]):
        raise ValueError(f"Slice '{slice_idx}' out of bounds, only '{data.shape[0]}' sagittal slices.")

    # Find slice in correct plane, x=sagittal, y=coronal, z=axial.
    data_index = (
        slice_idx if view == 'sagittal' else slice(data.shape[0]),
        slice_idx if view == 'coronal' else slice(data.shape[1]),
        slice_idx if view == 'axial' else slice(data.shape[2]),
    )
    slice_data = data[data_index]

    # Convert from our co-ordinate system (frontal, sagittal, longitudinal) to 
    # that required by 'imshow'.
    slice_data = _to_image_coords(slice_data, view)

    return slice_data

def get_aspect_ratio(
    view: types.PatientView,
    spacing: types.ImageSpacing3D) -> float:
    # Get the aspect ratio.
    if view == 'axial':
        aspect = spacing[1] / spacing[0]
    elif view == 'coronal':
        aspect = spacing[2] / spacing[0]
    elif view == 'sagittal':
        aspect = spacing[2] / spacing[1]

    return aspect

def reverse_box_coords_2D(box: types.Box2D) -> types.Box2D:
    """
    returns: a box with (x, y) coordinates reversed.
    args:
        box: the box to swap coordinates for.
    """
    # Reverse coords.
    box = tuple((y, x) for x, y in box)

    return box

def should_plot_box(
    box: types.Box3D,
    view: types.PatientView,
    slice_idx: int) -> bool:
    """
    returns: True if the box should be plotted.
    args:
        bounding_box: the bounding box to plot.
        view: the view direction.
        slice_idx: the index of the slice to plot.
    """
    # Get view bounding box.
    if view == 'axial':
        dim = 2
    elif view == 'coronal':
        dim = 1
    elif view == 'sagittal':
        dim = 0
    min, max = box
    min = min[dim]
    max = max[dim]

    # Return result.
    return slice_idx >= min and slice_idx <= max

def plot_box_slice(
    box: types.Box3D,
    view: types.PatientView,
    colour: str = 'r',
    crop: types.Box2D = None,
    label: str = 'box',
    linestyle: str = 'solid') -> None:
    """
    effect: plots a 2D slice of the box.
    args:
        box: the box to plot.
        view: the view direction.
    kwargs:
        crop: the cropping applied to the image.
    """
    # Compress box to 2D.
    if view == 'axial':
        dims = (0, 1)
    elif view == 'coronal':
        dims = (0, 2)
    elif view == 'sagittal':
        dims = (1, 2)
    min, max = box
    min = np.array(min)[[*dims]]
    max = np.array(max)[[*dims]]
    box_2D = (min, max)

    # Apply crop.
    if crop:
        box_2D = crop_or_pad_box(box_2D, crop)

    # Draw bounding box.
    min, max = box_2D
    min = np.array(min) - .5
    max = np.array(max) + .5
    width = np.array(max) - min
    rect = Rectangle(min, *width, linewidth=1, edgecolor=colour, facecolor='none', linestyle=linestyle)
    ax = plt.gca()
    ax.add_patch(rect)
    plt.plot(0, 0, c=colour, label=label, linestyle=linestyle)

def _escape_latex(text: str) -> str:
    """
    returns: a string with escaped latex special characters.
    args:
        text: the string to escape.
    """
    # Provide map for special characters.
    char_map = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(char_map.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: char_map[match.group()], text)

def assert_position(
    centre_of: Optional[int],
    extent_of: Optional[Tuple[str, bool]],
    slice_idx: Optional[str]):
    if centre_of is None and extent_of is None and slice_idx is None:
        raise ValueError(f"Either 'centre_of', 'extent_of' or 'slice_idx' must be set.")
    elif (centre_of and extent_of) or (centre_of and slice_idx) or (extent_of and slice_idx) or (centre_of and extent_of and slice_idx):
        raise ValueError(f"Only one of 'centre_of', 'extent_of' or 'slice_idx' can be set.")

def plot_regions(
    id: str,
    ct_data: np.ndarray,
    region_data: Dict[str, np.ndarray],
    spacing: types.ImageSpacing3D,
    alpha: float = 0.3,
    aspect: Optional[float] = None,
    axis: Optional[matplotlib.axes.Axes] = None,
    cca: bool = False,
    centre_of: Optional[str] = None,
    colours: Optional[List[str]] = None,
    crop: Optional[Union[str, types.Box2D]] = None,
    crop_margin: float = 40,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    figsize: Tuple[int, int] = (12, 12),
    font_size: int = 10,
    latex: bool = False,
    legend: bool = True,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
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
    if axis is None:
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

    # Get slice index if requested OAR centre.
    if centre_of is not None:
        # Get extent centre.
        label = region_data[centre_of]
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
        label = region_data[eo_region]
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

    # Load region data.
    if regions is not None:
        if postproc:
            region_data = dict(((r, postproc(d)) for r, d in region_data.items()))

    # Get slice data.
    ct_slice_data = get_slice(ct_data, slice_idx, view)

    # Convert to box representation.
    if crop:
        if type(crop) == str:
            # Get crop box from region name.
            data = region_data[crop]
            crop = _get_region_crop(data, crop_margin, spacing, view)

    # Perform crop.
    if crop:
        ct_slice_data = crop_or_pad_2D(ct_slice_data, reverse_box_coords_2D(crop))

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
        show_legend = _plot_region_data(region_data, slice_idx, alpha, aspect, latex, perimeter, view, axis=axis, cca=cca, colours=colours, crop=crop, show_extent=show_extent)

        # Create legend.
        if legend and show_legend:
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
            title_text = f"{id} - {slice_idx}/{num_slices - 1} ({view})"

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

def plot_localiser_prediction(
    id: str,
    region: str,
    ct_data: np.ndarray,
    region_data: np.ndarray,
    spacing: types.ImageSpacing3D, 
    prediction: np.ndarray,
    aspect: float = None,
    centre_of: Optional[str] = None,
    crop: types.Box2D = None,
    extent_of: Optional[Literal[0, 1]] = None,
    latex: bool = False,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    show_label_extent: bool = True,
    show_loc_centre: bool = True,
    show_loc_extent: bool = True,
    show_loc_pred: bool = True,
    show_seg_patch: bool = True,
    slice_idx: Optional[int] = None,
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

    # Load localiser segmentation.
    if prediction.sum() == 0:
        logging.info('Empty prediction')
        empty_pred = True
    else:
        empty_pred = False

    # Centre on OAR if requested.
    if slice_idx is None:
        if centre_of is not None:
            if centre_of == 'loc':
                # Centre of prediction.
                label = prediction
            else:
                # Centre of label.
                label = region_data

            # Get slice index.
            centre = get_extent_centre(label)
            if view == 'axial':
                slice_idx = centre[2]
            elif view == 'coronal':
                slice_idx = centre[1]
            elif view == 'sagittal':
                slice_idx = centre[0]
        elif extent_of is not None:
            # Get extent.
            extent = get_extent(region_data)

            # Set slice index.
            if view == 'axial':
                slice_idx = extent[extent_of][2]
            elif view == 'coronal':
                slice_idx = extent[extent_of][1]
            elif view == 'sagittal':
                slice_idx = extent[extent_of][0]

    # Plot patient regions.
    plot_regions(id, ct_data, { region: region_data }, spacing, aspect=aspect, colours=['gold'], crop=crop, latex=latex, legend=False, legend_loc=legend_loc, regions=region, show=False, show_extent=show_label_extent, slice_idx=slice_idx, view=view, **kwargs)

    # Get extent and centre.
    extent = get_extent(prediction)
    loc_centre = get_extent_centre(prediction)

    # Plot prediction.
    if not empty_pred and show_loc_pred:
        # Get aspect ratio.
        if not aspect:
            aspect = get_aspect_ratio(view, spacing) 

        # Get slice data.
        pred_slice_data = get_slice(prediction, slice_idx, view)

        # Crop the image.
        if crop:
            pred_slice_data = crop_or_pad_2D(pred_slice_data, reverse_box_coords_2D(crop))

        # Plot prediction.
        colour = 'deepskyblue'
        colours = [(1, 1, 1, 0), colour]
        cmap = ListedColormap(colours)
        plt.imshow(pred_slice_data, alpha=.3, aspect=aspect, cmap=cmap, origin=get_origin(view))
        plt.plot(0, 0, c=colour, label='Loc. Prediction')
        plt.contour(pred_slice_data, colors=[colour], levels=[.5])

    # Plot prediction extent.
    if not empty_pred and show_loc_extent:
        if should_plot_box(extent, view, slice_idx):
            plot_box_slice(extent, view, colour='deepskyblue', crop=crop, label='Loc. Box', linestyle='dashed')
        else:
            plt.plot(0, 0, c='deepskyblue', label='Loc. Extent (offscreen)')

    # Plot localiser centre.
    if not empty_pred and show_loc_centre:
        # Get 2D loc centre.
        if view == 'axial':
            centre = (loc_centre[0], loc_centre[1])
            offscreen = False if slice_idx == loc_centre[2] else True
        elif view == 'coronal':
            centre = (loc_centre[0], loc_centre[2])
            offscreen = False if slice_idx == loc_centre[1] else True
        elif view == 'sagittal':
            centre = (loc_centre[1], loc_centre[2])
            offscreen = False if slice_idx == loc_centre[0] else True
            
        # Apply crop.
        centre = crop_or_pad_point(centre, crop)

        if centre:
            plt.scatter(*centre, c='royalblue', label=f"Loc. Centre{' (offscreen)' if offscreen else ''}")
        else:
            plt.plot(0, 0, c='royalblue', label='Loc. Centre (offscreen)')

    # Plot second stage patch.
    if not empty_pred and show_seg_patch:
        size = get_patch_size(region, spacing)
        min, max = get_box(loc_centre, size)

        # Squash min/max to label size.
        min = np.clip(min, a_min=0, a_max=None)
        max = np.clip(max, a_min=None, a_max=prediction.shape)

        if should_plot_box((min, max), view, slice_idx):
            plot_box_slice((min, max), view, colour='tomato', crop=crop, label='Seg. Patch', linestyle='dotted')
        else:
            plt.plot(0, 0, c='tomato', label='Seg. Patch (offscreen)', linestyle='dashed')

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

def _get_region_crop(
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

def plot_distribution(
    data: np.ndarray,
    figsize: Tuple[float, float] = (12, 6),
    range: Optional[Tuple[float, float]] = None,
    resolution: float = 10) -> None:
    # Calculate bin width.
    min = data.min()
    max = data.max()
    num_bins = int(np.ceil((max - min) / resolution))

    # Get limits.
    if range:
        limits = range
    else:
        limits = (min, max)
        
    # Plot histogram.
    plt.figure(figsize=figsize)
    plt.hist(data.flatten(), bins=num_bins, range=range, histtype='step',edgecolor='r',linewidth=3)
    plt.title(f'Hist. of voxel values, range={tuple(np.array(limits).round().astype(int))}')
    plt.xlabel('HU')
    plt.ylabel('Frequency')
    plt.show()

def plot_segmenter_prediction(
    id: str,
    region: str,
    ct_data: np.ndarray,
    loc_centre: types.Point3D,
    region_data: np.ndarray,
    spacing: types.ImageSpacing3D,
    prediction: np.ndarray,
    aspect: float = None,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Box2D]] = None,
    crop_margin: float = 40,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    latex: bool = False,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    show_label_extent: bool = True,
    show_loc_centre: bool = True,
    show_seg_extent: bool = True,
    show_seg_patch: bool = True,
    show_seg_pred: bool = True,
    slice_idx: Optional[int] = None,
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

    # Check for empty pred.
    if prediction.sum() == 0:
        logging.info('Empty prediction')
        empty_pred = True
    else:
        volume_vox = prediction.sum()
        volume_mm3 = volume_vox * reduce(lambda x, y: x * y, spacing)
        logging.info(f"""
Volume (vox): {volume_vox}
Volume (mm^3): {volume_mm3:.2f}""")
        empty_pred = False

    # Centre on OAR if requested.
    if slice_idx is None:
        if centre_of is not None:
            if centre_of == 'seg':
                # Centre of prediction.
                label = prediction
            else:
                # Centre of label.
                label = region_data

            # Get slice index.
            centre = get_extent_centre(label)
            if view == 'axial':
                slice_idx = centre[2]
            elif view == 'coronal':
                slice_idx = centre[1]
            elif view == 'sagittal':
                slice_idx = centre[0]
        elif extent_of is not None:
            # Get extent.
            extent = get_extent(region_data)

            # Set slice index.
            if view == 'axial':
                slice_idx = extent[extent_of][2]
            elif view == 'coronal':
                slice_idx = extent[extent_of][1]
            elif view == 'sagittal':
                slice_idx = extent[extent_of][0]

    # Plot patient regions.
    plot_regions(id, ct_data, { region: region_data }, spacing, aspect=aspect, colours=['gold'], crop=crop, latex=latex, legend=False, legend_loc=legend_loc, regions=region, show=False, show_extent=show_label_extent, slice_idx=slice_idx, view=view, **kwargs)

    if crop:
        if type(crop) == str:
            # Get crop box from region name.
            data = region_data[crop]
            crop = _get_region_crop(data, crop_margin, spacing, view)

    # Get extent and centre.
    extent = get_extent(prediction)

    # Plot prediction.
    if not empty_pred and show_seg_pred:
        # Get aspect ratio.
        if not aspect:
            aspect = get_aspect_ratio(view, spacing) 

        # Get slice data.
        pred_slice_data = get_slice(prediction, slice_idx, view)

        # Crop the image.
        if crop:
            pred_slice_data = crop_or_pad_2D(pred_slice_data, reverse_box_coords_2D(crop))

        # Plot prediction.
        colour = 'tomato'
        colours = [(1, 1, 1, 0), colour]
        cmap = ListedColormap(colours)
        plt.imshow(pred_slice_data, alpha=.3, aspect=aspect, cmap=cmap, origin=get_origin(view))
        plt.plot(0, 0, c=colour, label='Seg. Prediction')
        plt.contour(pred_slice_data, colors=[colour], levels=[.5])

    # Plot segmenter extent.
    if not empty_pred and show_seg_extent:
        if should_plot_box(extent, view, slice_idx):
            plot_box_slice(extent, view, colour='tomato', crop=crop, label='Seg. Extent', linestyle='dashed')
        else:
            plt.plot(0, 0, c='tomato', label='Seg. Extent (offscreen)')

    # Plot localiser centre.
    if not empty_pred and show_loc_centre:
        # Get 2D loc centre.
        if view == 'axial':
            centre = (loc_centre[0], loc_centre[1])
            offscreen = False if slice_idx == loc_centre[2] else True
        elif view == 'coronal':
            centre = (loc_centre[0], loc_centre[2])
            offscreen = False if slice_idx == loc_centre[1] else True
        elif view == 'sagittal':
            centre = (loc_centre[1], loc_centre[2])
            offscreen = False if slice_idx == loc_centre[0] else True
            
        # Apply crop.
        centre = crop_or_pad_point(centre, crop)

        if centre:
            plt.scatter(*centre, c='royalblue', label=f"Loc. Centre{' (offscreen)' if offscreen else ''}")
        else:
            plt.plot(0, 0, c='royalblue', label='Loc. Centre (offscreen)')

    # Plot second stage patch.
    if not empty_pred and show_seg_patch:
        # Get 3D patch - cropped to label size.
        size = get_patch_size(region, spacing)
        patch = get_box(loc_centre, size)
        label_box = ((0, 0, 0), prediction.shape)
        patch = crop_or_pad_box(patch, label_box)

        # Plot box.
        if patch and should_plot_box(patch, view, slice_idx):
            plot_box_slice(patch, view, colour='tomato', crop=crop, label='Seg. Patch', linestyle='dotted')
        else:
            plt.plot(0, 0, c='tomato', label='Seg. Patch (offscreen)', linestyle='dashed')

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

from audioop import reverse
from functools import reduce
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from numpy import ndarray
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns
from statannotations.Annotator import Annotator
import torchio
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
    aspect: Optional[float] = None,
    axis: Optional[matplotlib.axes.Axes] = None,
    cca: bool = False,
    centre_of: Optional[str] = None,
    colours: Optional[List[str]] = None,
    crop: Optional[Union[str, types.Box2D]] = None,
    crop_margin: float = 100,
    dose_alpha: float = 0.3,
    dose_data: Optional[np.ndarray] = None,
    dose_legend_size: float = 0.03,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    figsize: Tuple[int, int] = (8, 8),
    fontsize: int = 10,
    latex: bool = False,
    legend: bool = True,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    perimeter: bool = True,
    postproc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    regions: Union[str, Sequence[str]] = 'all',
    region_alpha: float = 0.3,
    savepath: Optional[str] = None,
    show: bool = True,
    show_axis_labels: bool = True,
    show_axis_ticks: bool = True,
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
    if dose_data is not None:
        dose_slice_data = get_slice(dose_data, slice_idx, view)

    # Convert to box representation.
    if crop:
        if type(crop) == str:
            # Get crop box from region name.
            data = region_data[crop]
            crop = _get_region_crop(data, crop_margin, spacing, view)

    # Perform crop.
    if crop:
        ct_slice_data = crop_or_pad_2D(ct_slice_data, reverse_box_coords_2D(crop))
        if dose_data is not None:
            dose_slice_data = crop_or_pad_2D(dose_slice_data, reverse_box_coords_2D(crop))

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
    if show_axis_labels:
        if view == 'axial':
            spacing_x = spacing[0]
            spacing_y = spacing[1]
        elif view == 'coronal':
            spacing_x = spacing[0]
            spacing_y = spacing[2]
        elif view == 'sagittal':
            spacing_x = spacing[1]
            spacing_y = spacing[2]

        axis.set_xlabel(f'voxel [@ {spacing_x:.3f} mm spacing]')
        axis.set_ylabel(f'voxel [@ {spacing_y:.3f} mm spacing]')

    if regions:
        # Plot regions.
        show_legend = _plot_region_data(region_data, slice_idx, region_alpha, aspect, latex, perimeter, view, axis=axis, cca=cca, colours=colours, crop=crop, show_extent=show_extent)

        # Create legend.
        if legend and show_legend:
            plt_legend = axis.legend(fontsize=fontsize, loc=legend_loc)
            for l in plt_legend.get_lines():
                l.set_linewidth(8)

    # Plot dose data.
    if dose_data is not None:
        axim = axis.imshow(dose_slice_data, alpha=dose_alpha, aspect=aspect, origin=get_origin(view))
        cbar = plt.colorbar(axim, fraction=dose_legend_size)
        print('here')
        print(fontsize)
        cbar.set_label(label='Dose [Gray]', size=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

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
            title_text = f"patient: {id}, slice: {slice_idx}/{num_slices - 1} ({view} view)"

        # Escape text if using latex.
        if latex:
            title_text = _escape_latex(title_text)

        axis.set_title(title_text)

    # Remove whitespace around figure.
    plt.tight_layout()

    # Save plot to disk.
    if savepath is not None:
        plt.savefig(savepath)

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
    fontsize: float = 10,
    latex: bool = False,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
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
        if crop:
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
    plt_legend = plt.legend(fontsize=fontsize, loc=legend_loc)
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
    region_data: np.ndarray,
    spacing: types.ImageSpacing3D,
    preds: Union[np.ndarray, List[np.ndarray]],
    aspect: float = None,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Box2D]] = None,
    crop_margin: float = 100,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    fontsize: float = 10,
    latex: bool = False,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    loc_centre: Optional[Union[types.Point3D, List[types.Point3D]]] = None,
    savepath: Optional[str] = None,
    show_label_extent: bool = True,
    show_loc_centre: bool = True,
    show_pred_extent: bool = True,
    show_pred_patch: bool = True,
    show_pred: bool = True,
    slice_idx: Optional[int] = None,
    pred_labels: Optional[List[str]] = None,
    view: types.PatientView = 'axial',
    **kwargs: dict) -> None:
    palette = [plt.cm.tab20(16), plt.cm.tab20(12), plt.cm.tab20(4)]
    assert_position(centre_of, extent_of, slice_idx)
    # Convert params to lists.
    if type(preds) == np.ndarray:
        preds = [preds]
    else:
        preds = preds
    if pred_labels is None:
        pred_labels = ['Prediction'] * len(preds)
    if loc_centre is not None:
        if type(loc_centre) == types.Point3D:
            loc_centres = [loc_centre]
        else:
            loc_centres = loc_centre

        assert len(loc_centres) == len(preds)

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Print prediction info.
    empty_preds = [False] * len(preds)
    for i, prediction in enumerate(preds):
        if prediction.sum() == 0:
            logging.info(f"Prediction {i} ('{pred_labels[i]}') is empty.")
            empty_preds[i] = True
        else:
            volume_vox = prediction.sum()
            volume_mm3 = volume_vox * reduce(lambda x, y: x * y, spacing)
            logging.info(f"""
    Prediction: {i}
    Volume (vox): {volume_vox}
    Volume (mm^3): {volume_mm3:.2f}""")

    # Centre on OAR if requested.
    if slice_idx is None:
        if centre_of is not None:
            if type(centre_of) == str:
                if centre_of == region:
                    label = region_data
                elif centre_of in pred_labels:
                    label = preds[pred_labels.index(centre_of)]
                else:
                    raise ValueError(f"'centre_of={centre_of}' not valid.")

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

    # Convert crop from 'region' name 
    if type(crop) == str:
        assert crop == region
        crop = _get_region_crop(region_data, crop_margin, spacing, view)

    # Plot patient regions.
    plot_regions(id, ct_data, { region: region_data }, spacing, aspect=aspect, colours=[palette[0]], crop=crop, latex=latex, legend=False, legend_loc=legend_loc, regions=region, show=False, show_extent=show_label_extent, slice_idx=slice_idx, view=view, **kwargs)

    # Get extent and centre.
    extent = get_extent(prediction)

    for i, prediction in enumerate(preds):
        # Get prediction colour.
        colour = palette[i + 1]

        # Plot prediction.
        if not empty_preds[i] and show_pred:
            # Get aspect ratio.
            if not aspect:
                aspect = get_aspect_ratio(view, spacing) 

            # Get slice data.
            pred_slice_data = get_slice(prediction, slice_idx, view)

            # Crop the image.
            if crop:
                pred_slice_data = crop_or_pad_2D(pred_slice_data, reverse_box_coords_2D(crop))

            # Plot prediction.
            colours = [(1, 1, 1, 0), colour]
            cmap = ListedColormap(colours)
            plt.imshow(pred_slice_data, alpha=.3, aspect=aspect, cmap=cmap, origin=get_origin(view))
            plt.plot(0, 0, c=colour, label=pred_labels[i])
            plt.contour(pred_slice_data, colors=[colour], levels=[.5])

        # Plot prediction extent.
        if not empty_preds[i] and show_pred_extent:
            if should_plot_box(extent, view, slice_idx):
                plot_box_slice(extent, view, colour=colour, crop=crop, label=f'{pred_labels[i]} Extent', linestyle='dashed')
            else:
                plt.plot(0, 0, c=colour, label='Pred. Extent (offscreen)')

        # Plot localiser centre.
        if loc_centre is not None and show_loc_centre:
            # Get 2D loc centre.
            loc_centre = loc_centres[i]
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
            if crop:
                centre = crop_or_pad_point(centre, crop)

            if centre:
                plt.scatter(*centre, c='royalblue', label=f"Loc. Centre{' (offscreen)' if offscreen else ''}")
            else:
                plt.plot(0, 0, c='royalblue', label='Loc. Centre (offscreen)')

        # Plot second stage patch.
        if loc_centre is not None and not empty_preds[i] and show_pred_patch:
            # Get 3D patch - cropped to label size.
            size = get_patch_size(region, spacing)
            patch = get_box(loc_centre, size)
            label_box = ((0, 0, 0), prediction.shape)
            patch = crop_or_pad_box(patch, label_box)

            # Plot box.
            if patch and should_plot_box(patch, view, slice_idx):
                plot_box_slice(patch, view, colour=colour, crop=crop, label='Pred. Patch', linestyle='dotted')
            else:
                plt.plot(0, 0, c=colour, label='Pred. Patch (offscreen)', linestyle='dashed')

    # Show legend.
    plt_legend = plt.legend(fontsize=fontsize, loc=legend_loc)
    for l in plt_legend.get_lines():
        l.set_linewidth(8)

    # Remove whitespace around figure.
    plt.tight_layout()

    # Save plot to disk.
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

    # Revert latex settings.
    if latex:
        plt.rcParams.update({
            "font.family": rc_params['font.family'],
            'text.usetex': rc_params['text.usetex']
        })

def plot_dataframe(
    data: pd.DataFrame = None,
    x: str = None,
    y: str = None,
    hue: str = None,
    ylim: Tuple[Optional[int], Optional[int]] = (None, None),
    inner: str ='quartiles',
    annotate_outliers: bool = False,
    annotation_model_offset=35,
    fontsize: float = 10,
    hue_order: Optional[List[str]] = None,
    legend_loc: str = 'best',
    major_tick_freq: Optional[float] = None,
    minor_tick_freq: Optional[float] = None,
    n_col: int = 6,
    offset=5,
    point_size: float = 5,
    point_style: str = 'strip',
    overlap_min_diff=None,
    annotation_overlap_offset=25,
    debug=False,
    row_height: int = 6,
    row_width: int = 18,
    savepath: Optional[str] = None,
    show_points: bool = True,
    show_outliers: bool = True,
    show_stats: bool = False,
    show_x_tick_labels: bool = True,
    stats_index: Optional[List[str]] = None,
    style: Literal['box', 'violin'] = 'box',
    x_label_rot: float = 0,
    ylabel: str = '',
    include_x: Union[str, List[str]] = None,
    exclude_x: Union[str, List[str]] = None):
    df = data

    # Include/exclude.
    if include_x:
        if type(include_x) == str:
            include_x = [include_x]
        df = df[df[x].isin(include_x)]
    if exclude_x:
        if type(exclude_x) == str:
            exclude_x = [exclude_x]
        df = df[~df[x].isin(exclude_x)]
    
    # Add region numbers and outliers.
    df = _add_x_info(df, x, hue)
    df = _add_outlier_info(df, x, y, hue)
    
    # Split data.
    xs = list(sorted(df[x].unique()))
    if hue_order is None:
        hue_order = list(sorted(df[hue].unique())) if hue is not None else None
    num_rows = int(np.ceil(len(xs) / n_col))
    if num_rows > 1:
        _, axs = plt.subplots(num_rows, 1, figsize=(row_width, num_rows * row_height), sharey=True)
    else:
        plt.figure(figsize=(row_width, row_height))
        axs = [plt.gca()]
    for i in range(num_rows):
        # Split data.
        split_xs = xs[i * n_col:(i + 1) * n_col]
        split_df = df[df[x].isin(split_xs)]
        if len(split_df) == 0:
            continue
            
        # Determine category label order.
        x_label = f'{x}_label'
        order = list(sorted(split_df[x_label].unique()))
        if len(xs) < n_col:
            order += [''] * (n_col - len(order))

        # Plot data.
        if style == 'box':
            sns.boxplot(ax=axs[i], data=split_df, x=x_label, y=y, hue=hue, showfliers=False, order=order, hue_order=hue_order)
        elif style == 'violin':
            # Exclude outliers manually.
            split_df = split_df[~split_df.outlier]
            sns.violinplot(ax=axs[i], data=split_df, x=x_label, y=y, hue=hue, inner=inner, split=True, showfliers=False, order=order, hue_order=hue_order)
        else:
            raise ValueError(f"Invalid style {style}, expected 'box' or 'violin'.")

        # Plot points.
        if show_points:
            if point_style == 'strip':
                sns.stripplot(ax=axs[i], data=split_df, x=x_label, y=y, hue=hue, dodge=True, jitter=False, order=order, linewidth=1, label=None, hue_order=hue_order, size=point_size)
            elif point_style == 'swarm':
                sns.swarmplot(ax=axs[i], data=split_df, x=x_label, y=y, hue=hue, dodge=True, order=order, linewidth=1, label=None, hue_order=hue_order, size=point_size)
            else:
                raise ValueError(f"Invalid point style {point_style}, expected 'strip' or 'swarm'.")

            # Plot outliers.
            if show_outliers:
                outlier_df = split_df[split_df.outlier]
                if len(outlier_df) != 0:
                    sns.stripplot(ax=axs[i], data=outlier_df, x=x_label, y=y, hue=hue, dodge=True, jitter=False, edgecolor='white', linewidth=1, order=order, hue_order=hue_order)
                    plt.setp(axs[i].collections, zorder=100, label="")
                    if annotate_outliers:
                        first_region = i * n_col
                        _annotate_outliers(axs[i], outlier_df, f'{x}_num', y, 'patient-id', offset, overlap_min_diff, annotation_overlap_offset, annotation_model_offset, debug, first_region)

        # Plot statistical significance.
        if hue is not None and show_stats:
            if len(hue_order) != 2:
                raise ValueError(f"Hue set must have cardinality 2, got '{len(hue_order)}'.")
            if stats_index is None:
                raise ValueError(f"Please set 'stats_index' to determine sample pairing.")

            # Create pairs to compare. Only works when len(hue_order) == 2.
            pairs = []
            for o in order:
                pair = []
                for h in hue_order:
                    pair.append((o, h))
                pairs.append(tuple(pair))

            # Calculate p-values.
            p_vals = []
            for o in order:
                osplit_df = split_df[split_df[x_label] == o]
                opivot_df = osplit_df.pivot(index=stats_index, columns=[hue], values=[y]).reset_index()
                _, p_val = wilcoxon(opivot_df[y][hue_order[0]], opivot_df[y][hue_order[1]])
                p_vals.append(p_val)

            # Format p-values.
            p_vals = _format_p_values(p_vals) 

            # Remove non-significant pairs.
            tpairs = []
            tp_vals = []
            for pair, p_val in zip(pairs, p_vals):
                if p_val != '':
                    tpairs.append(pair)
                    tp_vals.append(p_val)

            # Annotate figure.
            annotator = Annotator(axs[i], tpairs, data=split_df, x=x_label, y=y, order=order, hue=hue, hue_order=hue_order, verbose=False)
            annotator.set_custom_annotations(tp_vals)
            annotator.annotate()

        # Set y axis major ticks.
        if major_tick_freq is not None:
            major_tick_min = ylim[0]
            if major_tick_min is None:
                major_tick_min = axs[i].get_ylim()[0]
            major_tick_max = ylim[1]
            if major_tick_max is None:
                major_tick_max = axs[i].get_ylim()[1]
            
            # Round range to nearest multiple of 'major_tick_freq'.
            major_tick_min = np.ceil(major_tick_min / major_tick_freq) * major_tick_freq
            major_tick_max = np.floor(major_tick_max / major_tick_freq) * major_tick_freq
            num_major_ticks = int((major_tick_max - major_tick_min) / major_tick_freq) + 1
            major_ticks = np.linspace(major_tick_min, major_tick_max, num_major_ticks)
            integers = True
            for t in major_ticks:
                if not t.is_integer():
                    integers = False
            if integers:
                major_ticks = [int(t) for t in major_ticks]
            major_tick_labels = [str(round(t, 3)) for t in major_ticks]     # Some weird str() conversion without rounding.
            axs[i].set_yticks(major_ticks)
            axs[i].set_yticklabels(major_tick_labels)

        # Set y axis minor ticks.
        if minor_tick_freq is not None:
            minor_tick_min = ylim[0]
            if minor_tick_min is None:
                minor_tick_min = axs[i].get_ylim()[0]
            minor_tick_max = ylim[1]
            if minor_tick_max is None:
                minor_tick_max = axs[i].get_ylim()[1]
            
            # Round range to nearest multiple of 'minor_tick_freq'.
            minor_tick_min = np.ceil(minor_tick_min / minor_tick_freq) * minor_tick_freq
            minor_tick_max = np.floor(minor_tick_max / minor_tick_freq) * minor_tick_freq
            num_minor_ticks = int((minor_tick_max - minor_tick_min) / minor_tick_freq) + 1
            minor_ticks = np.linspace(minor_tick_min, minor_tick_max, num_minor_ticks)
            axs[i].set_yticks(minor_ticks, minor=True)

        # Set y grid lines.
        axs[i].grid(axis='y', linestyle='dashed')
        axs[i].set_axisbelow(True)
          
        # Set axis labels.
        if ylim:
            axs[i].set_ylim(*ylim)
        axs[i].set_xlabel('')
        axs[i].set_ylabel(ylabel, fontsize=fontsize)

        # Set axis tick labels fontsize/rotation.
        if show_x_tick_labels:
            # Rotate x labels.
            axs[i].set_xticklabels(axs[i].get_xticklabels(), fontsize=fontsize, rotation=x_label_rot)
        else:
            axs[i].set_xticklabels([])

        axs[i].tick_params(axis='y', which='major', labelsize=fontsize)
        
        # Set legend location and fix multiple series problem.
        if hue is not None:
            num_hues = len(hue_order)
            handles, labels = axs[i].get_legend_handles_labels()
            handles = handles[:num_hues]
            labels = labels[:num_hues]
            axs[i].legend(handles, labels, fontsize=fontsize, loc=legend_loc)
        else:
            axs[i].legend(fontsize=fontsize, loc=legend_loc)

    # Make sure x-axis tick labels aren't cut.
    plt.tight_layout()

    # Save plot to disk.
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

def _add_x_info(df, x, hue):
    if hue is not None:
        groupby = [hue, x] 
    else:
        groupby = x
    df = df.assign(**{ f'{x}_num': df.groupby(groupby).ngroup() })
    count_map = df.groupby(groupby)['patient-id'].count()
    def x_count_func(row):
        if type(groupby) == list:
            key = tuple(row[groupby])
        else:
            key = row[groupby]
        return count_map[key]
    df = df.assign(**{ f'{x}_count': df.apply(x_count_func, axis=1) })
    def x_label_func(row):
        return f"{row[x]}\n(n={row[f'{x}_count']})"
    df = df.assign(**{ f'{x}_label': df.apply(x_label_func, axis=1) })
    return df

def _add_outlier_info(df, x, y, hue):
    if hue is not None:
        groupby = [hue, x]
    else:
        groupby = x
    q1_map = df.groupby(groupby)[y].quantile(.25)
    q3_map = df.groupby(groupby)[y].quantile(.75)
    def q_func_build(qmap):
        def q_func(row):
            if type(groupby) == list:
                key = tuple(row[groupby])
            else:
                key = row[groupby]
            return qmap[key]
        return q_func
    df = df.assign(q1=df.apply(q_func_build(q1_map), axis=1))
    df = df.assign(q3=df.apply(q_func_build(q3_map), axis=1))
    df = df.assign(iqr=df.q3 - df.q1)
    df = df.assign(outlier_lim_low=df.q1 - 1.5 * df.iqr)
    df = df.assign(outlier_lim_high=df.q3 + 1.5 * df.iqr)
    df = df.assign(outlier=(df[y] < df.outlier_lim_low) | (df[y] > df.outlier_lim_high))
    return df

def _annotate_outliers(ax, data, x, y, label, default_offset, overlap_min_diff, annotation_overlap_offset, annotation_offset, debug, first_region):
    models = None
    if 'model' in data:
        models = data.model.unique()
    offset_transform = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    base_transform = ax.transData
    data = data.sort_values([x, y], ascending=False)
    prev_model = None
    prev_x = None
    prev_y = None
    prev_digits = None
    offset = default_offset
    
    # Print annotations.
    for _, row in data.iterrows():
        if debug:
            print(f'region: {row[x]}')
            print(f'patient: {row[label]}')

        # Get model offset.
        if models is not None:
            if row['model'] == models[0]:
                ann_offset = -annotation_offset
            else:
                ann_offset = annotation_offset
        else:
            ann_offset = 0
        
        # Apply offset to labels when outliers are overlapping.
        if overlap_min_diff is not None:
            # Check that model hasn't changed.
            if prev_model is None or row['model'] == prev_model:
                # Check that region hasn't changed.
                if prev_x is not None and row[x] == prev_x:
                    if debug:
                        print('checking diff')
                    if prev_y is not None:
                        diff = prev_y - row[y]
                        if debug:
                            print(f'diff: {diff}')
                        if diff < overlap_min_diff:
                            if debug:
                                print(f'offsetting point {row[label]}')
                            offset += (annotation_overlap_offset * prev_digits / 3)
                        else:
                            offset = default_offset
                else:
                    offset = default_offset
            else:
                offset = default_offset
                
        # Save previous values.
        if models is not None:
            prev_model = row['model']
        prev_y = row[y]
        prev_x = row[x]
        prev_digits = len(row[label])
        x_val = row[x] - first_region
        ax.text(x_val, row[y], row[label], transform=base_transform + offset_transform(offset) + offset_transform(ann_offset))

def _format_p_values(p_vals: List[float]) -> List[str]:
    f_p_vals = []
    for p_val in p_vals:
        if p_val >= 0.05:
            p_val = ''
        elif p_val >= 0.01:
            p_val = '*'
        elif p_val >= 0.001:
            p_val = '**'
        elif p_val >= 0.0001:
            p_val = '***'
        else:
            p_val = '****'
        f_p_vals.append(p_val)
    return f_p_vals

def plot_dataframe_v2(
    data=None,
    x=None,
    y=None,
    hue=None,
    box_line_colour: str = 'black',
    box_line_width: float = 1,
    fontsize: float = 10,
    hue_order: Optional[List[str]] = None,
    include_x: Optional[Union[str, List[str]]] = None,
    exclude_x: Optional[Union[str, List[str]]] = None,
    legend_loc: str = 'upper right',
    major_tick_freq: Optional[float] = None,
    minor_tick_freq: Optional[float] = None,
    n_col: Optional[int] = None,
    outlier_cols: Optional[Union[str, List[str]]] = None,
    outlier_legend_loc: str = 'upper left',
    point_size: float = 10,
    row_height: float = 6,
    row_width: float = 18,
    savepath: Optional[str] = None,
    show_legend: bool = True,
    show_outliers: bool = False,
    show_x_labels: bool = True,
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
    x_width: float = 0.8,
    xlabel_rot: float = 0,
    ylabel: Optional[str] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = (None, None)):
    if type(include_x) == str:
        include_x = [include_x]
    if type(exclude_x) == str:
        exclude_x = [exclude_x]
        
    # Include/exclude.
    if include_x:
        if type(include_x) == str:
            include_x = [include_x]
        data = data[data[x].isin(include_x)]
    if exclude_x:
        if type(exclude_x) == str:
            exclude_x = [exclude_x]
        data = data[~data[x].isin(exclude_x)]

    # Add outlier data.
    data = _add_outlier_info(data, x, y, hue)

    # Get x values.
    x_vals = list(sorted(data[x].unique()))

    # Get x labels.
    groupby = x if hue is None else [x, hue]
    count_map = data.groupby(groupby)[y].count()
    x_labels = []
    for x_val in x_vals:
        ns = list(count_map.loc[x_val].unique())
        label = f'{x_val}\n(n={ns[0]})' if len(ns) == 1 else x_val
        x_labels.append(label)

    # Create subplots if required.
    if n_col is None:
        n_col = len(x_vals)
    num_rows = int(np.ceil(len(x_vals) / n_col))
    if num_rows > 1:
        _, axs = plt.subplots(num_rows, 1, figsize=(row_width, num_rows * row_height), sharey=True)
    else:
        plt.figure(figsize=(row_width, row_height))
        axs = [plt.gca()]

    # Get hue order/colours.
    if hue is not None:
        if hue_order is None:
            hue_order = list(sorted(data[hue].unique()))
        hue_palette = sns.color_palette('husl', n_colors=len(hue_order))

    # Plot rows.
    for i in range(num_rows):
        # Split data.
        row_x_vals = x_vals[i * n_col:(i + 1) * n_col]
        row_x_labels = x_labels[i * n_col:(i + 1) * n_col]

        for j, row_x_val in enumerate(row_x_vals):
            # Filter out missing hue values.
            if hue is not None:
                hue_order_f = []
                for hue_name in hue_order:
                    hue_data = data[(data[x] == row_x_val) & (data[hue] == hue_name)]
                    if len(hue_data) != 0:
                        hue_order_f.append(hue_name)

            # Calculate hue width.
            hue_width = x_width / len(hue_order_f)

            # Add x positions.
            if hue is not None:
                for k, hue_name in enumerate(hue_order_f):
                    # Add hue x positions.
                    x_pos = j - 0.5 * x_width + (k + 0.5) * hue_width
                    data.loc[(data[x] == row_x_val) & (data[hue] == hue_name), 'x_pos'] = x_pos
            else:
                pass
                
            # Plot boxes.
            if hue is not None:
                box_labels = []
                for j, hue_name in enumerate(hue_order_f):
                    # Get hue data and pos.
                    hue_data = data[(data[x] == row_x_val) & (data[hue] == hue_name)]
                    if len(hue_data) == 0:
                        continue
                    hue_pos = hue_data.iloc[0]['x_pos']

                    # Plot box.
                    bplot = axs[i].boxplot(hue_data[y], boxprops=dict(color=box_line_colour, facecolor=hue_palette[j], linewidth=box_line_width), capprops=dict(color=box_line_colour, linewidth=box_line_width), flierprops=dict(color=box_line_colour, linewidth=box_line_width, marker='D', markeredgecolor=box_line_colour), medianprops=dict(color=box_line_colour, linewidth=box_line_width), patch_artist=True, positions=[hue_pos], showfliers=False, whiskerprops=dict(color=box_line_colour, linewidth=box_line_width), widths=hue_width)
                    box_labels.append((bplot['boxes'][0], hue_name)) 

                # Add legend.
                if show_legend:
                    boxes, labels = list(zip(*box_labels))
                    axs[i].legend(boxes, labels, fontsize=fontsize, loc=legend_loc)
            else:
                pass

            # Plot points.
            if hue is not None:
                for j, hue_name in enumerate(hue_order_f):
                    # Get hue data and pos.
                    hue_data = data[(data[x] == row_x_val) & (data[hue] == hue_name)]
                    if len(hue_data) == 0:
                        continue
                    hue_pos = hue_data.iloc[0]['x_pos']

                    # Plot points.
                    axs[i].scatter(hue_data['x_pos'], hue_data[y], color=hue_palette[j], edgecolors='black', linewidth=0.5, s=point_size, zorder=100)
            else:
                pass

            # Identify outliers - plot line connecting outliers across hue levels.
            if show_outliers and hue is not None:
                # Get column/value pairs to group across hue levels.
                line_ids = data[(data[x] == row_x_val) & data['outlier']][outlier_cols]

                # Drop duplicates.
                line_ids = line_ids.drop_duplicates()

                # Get palette.
                line_palette = sns.color_palette('husl', n_colors=len(line_ids))

                # Plot lines.
                artists = []
                labels = []
                for j, (_, line_id) in enumerate(line_ids.iterrows()):
                    # Get line data.
                    line_data = data[(data[x] == row_x_val)]
                    for k, v in zip(line_ids.columns, line_id):
                        line_data = line_data[line_data[k] == v]
                    line_data = line_data.sort_values('x_pos')
                    x_data = line_data['x_pos'].tolist()
                    y_data = line_data[y].tolist()

                    # Plot line.
                    lines = axs[i].plot(x_data, y_data, color=line_palette[j])

                    # Save line/label for legend.
                    artists.append(lines[0])
                    label = ':'.join(line_id.tolist())
                    labels.append(label)

                # Annotate outlier legend.
                if show_legend:
                    # Save main legend.
                    main_legend = axs[i].get_legend()

                    # Show outlier legend.
                    axs[i].legend(artists, labels, fontsize=fontsize, loc=outlier_legend_loc)

                    # Re-add main legend.
                    axs[i].add_artist(main_legend)
                
        # Set axis ticks and labels.
        axs[i].set_xticks(list(range(len(row_x_labels))))
        axs[i].set_xticklabels(row_x_labels)

        # Set y axis major ticks.
        if major_tick_freq is not None:
            major_tick_min = ylim[0]
            if major_tick_min is None:
                major_tick_min = axs[i].get_ylim()[0]
            major_tick_max = ylim[1]
            if major_tick_max is None:
                major_tick_max = axs[i].get_ylim()[1]
            
            # Round range to nearest multiple of 'major_tick_freq'.
            major_tick_min = np.ceil(major_tick_min / major_tick_freq) * major_tick_freq
            major_tick_max = np.floor(major_tick_max / major_tick_freq) * major_tick_freq
            num_major_ticks = int((major_tick_max - major_tick_min) / major_tick_freq) + 1
            major_ticks = np.linspace(major_tick_min, major_tick_max, num_major_ticks)
            major_tick_labels = [str(round(t, 3)) for t in major_ticks]     # Some weird str() conversion without rounding.
            axs[i].set_yticks(major_ticks)
            axs[i].set_yticklabels(major_tick_labels)

        # Set y axis minor ticks.
        if minor_tick_freq is not None:
            minor_tick_min = ylim[0]
            if minor_tick_min is None:
                minor_tick_min = axs[i].get_ylim()[0]
            minor_tick_max = ylim[1]
            if minor_tick_max is None:
                minor_tick_max = axs[i].get_ylim()[1]
            
            # Round range to nearest multiple of 'minor_tick_freq'.
            minor_tick_min = np.ceil(minor_tick_min / minor_tick_freq) * minor_tick_freq
            minor_tick_max = np.floor(minor_tick_max / minor_tick_freq) * minor_tick_freq
            num_minor_ticks = int((minor_tick_max - minor_tick_min) / minor_tick_freq) + 1
            minor_ticks = np.linspace(minor_tick_min, minor_tick_max, num_minor_ticks)
            axs[i].set_yticks(minor_ticks, minor=True)

        # Set y grid lines.
        axs[i].grid(axis='y', linestyle='dashed')
        axs[i].set_axisbelow(True)

        # Set axis limits.
        xlim = list(xlim)
        if xlim[0] is None:
            xlim[0] = -0.5
        if xlim[1] is None:
            xlim[1] = n_col - 0.5
        axs[i].set_xlim(*xlim)
        axs[i].set_ylim(*ylim)
          
        # Set axis labels.
        axs[i].set_xlabel('')
        if ylabel is None:
            ylabel = ''
        axs[i].set_ylabel(ylabel, fontsize=fontsize)

        # Set axis tick labels fontsize/rotation.
        if show_x_labels:
            axs[i].set_xticklabels(axs[i].get_xticklabels(), fontsize=fontsize, rotation=xlabel_rot)
        else:
            axs[i].set_xticklabels([])

        axs[i].tick_params(axis='y', which='major', labelsize=fontsize)

    # Make sure x-axis tick labels aren't cut.
    plt.tight_layout()

    # Save plot to disk.
    if savepath is not None:
        plt.savefig(savepath)

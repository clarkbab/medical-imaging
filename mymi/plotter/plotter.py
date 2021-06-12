from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import torch
from torch import nn
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import Sequence, Tuple, Union

from mymi import dataset
from mymi.predictions import get_patient_localisation, get_patient_patch_segmentation
from mymi.regions import is_region, RegionColours
from mymi.transforms import crop_or_pad_2D
from mymi import types
from mymi.utils import escape_latex

def plot_ct_distribution(
    bin_width: int = 10,
    clear_cache: bool = False,
    figsize: Tuple[int, int] = (10, 10),
    labels: Union[str, Sequence[str]] = 'all',
    max_bin: int = None,
    min_bin: int = None,
    num_pats: Union[str, int] = 'all',
    pat_ids: Union[str, Sequence[str]] = 'all') -> None:
    """
    effect: plots CT distribution of the dataset.
    kwargs:
        bin_width: the width of the histogram bins.
        clear_cache: forces the cache to clear.
        figsize: the size of the figure.
        labels: include patients with any of the listed labels (behaves like an OR).
        max_bin: the maximum bin to show. 
        min_bin: the minimum bin to show.
        num_pats: the number of patients to include.
        pat_ids: the patients to include.
    """
    # Load CT distribution.
    freqs = dataset.ct_distribution(bin_width=bin_width, clear_cache=clear_cache, labels=labels, num_pats=num_pats, pat_ids=pat_ids)

    # Remove bins we don't want.
    if min_bin or max_bin:
        for b in freqs.keys():
            if (min_bin and b < min_bin) or (max_bin and b > max_bin):
                freqs.pop(b)

    # Plot the histogram.
    plt.figure(figsize=figsize)
    keys = tuple(freqs.keys())
    values = tuple(freqs.values())
    plt.hist(keys[:-1], keys, weights=values[:-1])
    plt.show()

def plot_patient_regions(
    id: str,
    slice_idx: int,
    alpha: float = 0.2,
    aspect: float = None,
    axes: bool = True,
    clear_cache: bool = False,
    crop: types.Box2D = None,
    figsize: Tuple[int, int] = (8, 8),
    font_size: int = 10,
    internal_regions: bool = False,
    latex: bool = False,
    legend: bool = True,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    perimeter: bool = True,
    regions: Union[str, Sequence[str]] = 'all',
    show: bool = True,
    title: Union[bool, str] = True,
    transform: torchio.transforms.Transform = None,
    view: types.PatientView = 'axial',
    window: Tuple[float, float] = None) -> None:
    """
    effect: plots a CT slice with labels.
    args:
        id: the patient ID.
        slice_idx: the slice to plot.
    kwargs:
        alpha: the region alpha.
        aspect: use a hard-coded aspect ratio, useful for viewing transformed images.
        axes: display the axes ticks and labels.
        clear_cache: force cache to clear.
        crop: crop plotting window to this extent.
        figsize: the size of the plot in inches.
        font_size: the size of the font.
        internal_regions: use the internal MYMI region names.
        latex: use latex to display text.
        legend: display the legend.
        legend_loc: the location of the legend.
        perimeter: highlight the perimeter.
        regions: the regions to display.
        show: call 'plt.show'.
        title: turns the title on/off. Can optionally pass a custom title.
        transform: apply the transform before plotting.
        view: the viewing axis.
        window: the HU window to apply.
    """
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
    pat = dataset.patient(id)
    spacing = pat.ct_spacing(clear_cache=clear_cache)

    # Load CT data.
    ct_data = pat.ct_data(clear_cache=clear_cache)

    # Get slice data.
    ct_slice_data = _get_slice_for_plotting(ct_data, slice_idx, view)

    # Perform crop.
    if crop:
        ct_slice_data = crop_or_pad_2D(ct_slice_data, _reverse_box_coords_2D(crop))

    # Load region data.
    if regions:
        region_data = pat.region_data(clear_cache=clear_cache, regions=regions)

        if internal_regions:
            # Map to internal region names.
            region_data = dict((_to_internal_region(r, clear_cache=clear_cache), d) for r, d in region_data.items())

    # Transform data.
    if transform:
        # Add 'batch' dimension.
        ct_data = np.expand_dims(ct_data, axis=0)
        region_data = dict(((n, np.expand_dims(d, axis=0)) for n, d in region_data.items()))

        # Create 'subject'.
        affine = np.array([
            [spacing[0], 0, 0, 0],
            [0, spacing[1], 0, 0],
            [0, 0, spacing[2], 0],
            [0, 0, 0, 1]
        ])
        ct_data = ScalarImage(tensor=ct_data, affine=affine)
        region_data = dict(((n, LabelMap(tensor=d, affine=affine)) for n, d in region_data.items()))

        # Transform CT data.
        subject = Subject(input=ct_data)
        output = transform(subject)

        # Transform region data.
        det_transform = output.get_composed_history()
        region_data = dict(((r, det_transform(Subject(region=d))) for r, d in region_data.items()))

        # Extract results.
        ct_data = output['input'].data.squeeze(0)
        region_data = dict(((n, o['region'].data.squeeze(0)) for n, o in region_data.items()))

    # Only apply aspect ratio if no transforms are being presented otherwise
    # we might end up with skewed images.
    if not aspect:
        if transform:
            aspect = 1
        else:
            aspect = _get_aspect_ratio(id, view) 

    # Determine plotting window.
    if window:
        width, level = window
        vmin = level - (width / 2)
        vmax = level + (width / 2)
    else:
        vmin, vmax = ct_data.min(), ct_data.max()

    # Plot CT data.
    plt.figure(figsize=figsize)
    plt.imshow(ct_slice_data, cmap='gray', aspect=aspect, origin=_get_origin(view), vmin=vmin, vmax=vmax)

    # Add axis labels.
    if axes:
        plt.xlabel('voxel')
        plt.ylabel('voxel')

    if regions:
        # Plot regions.
        if len(region_data) != 0:
            # Create palette if not using internal region colours.
            if not internal_regions:
                palette = plt.cm.tab20

            # Plot each region.
            at_least_one_region = False     # Only show legend if slice has at least one region.
            for i, (region, data) in enumerate(region_data.items()):
                # Convert data to 'imshow' co-ordinate system.
                slice_data = _get_slice_for_plotting(data, slice_idx, view)

                # Crop image.
                if crop:
                    slice_data = crop_or_pad_2D(slice_data, _reverse_box_coords_2D(crop))

                # Skip region if not present on this slice.
                if slice_data.max() == 0:
                    continue
                else:
                    at_least_one_region = True
                
                # Create binary colormap for each region.
                if internal_regions:
                    colour = getattr(RegionColours, region)
                else:
                    colour = palette(i)
                colours = [(1.0, 1.0, 1.0, 0), colour]
                region_cmap = ListedColormap(colours)

                # Plot region.
                plt.imshow(slice_data, alpha=alpha, aspect=aspect, cmap=region_cmap, origin=_get_origin(view))
                label = escape_latex(region) if latex else region
                plt.plot(0, 0, c=colour, label=label)
                if perimeter:
                    plt.contour(slice_data, colors=[colour], levels=[0.5])

            # Create legend.
            if legend and at_least_one_region: 
                plt_legend = plt.legend(loc=legend_loc, prop={'size': legend_size})
                for l in plt_legend.get_lines():
                    l.set_linewidth(8)

    # Show axis markers.
    show_axes = 'on' if axes else 'off'
    plt.axis(show_axes)

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
            title_text = f"{view.capitalize()} slice: {slice_idx}/{num_slices - 1}" 

        # Escape text if using latex.
        if latex:
            title_text = escape_latex(title_text)

        plt.title(title_text)

    if show:
        plt.show()

        # Revert latex settings.
        if latex:
            plt.rcParams.update({
                "font.family": rc_params['font.family'],
                'text.usetex': rc_params['text.usetex']
            })

def plot_patient_localisation(
    id: str,
    slice_idx: int,
    aspect: float = None,
    box_colour: str = 'r',
    crop: types.Box2D = None,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    latex: bool = False,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    localisation: types.Box3D = None,
    localisation_seg: np.ndarray = None,
    localiser: nn.Module = None,
    localiser_size: types.Size3D = None,
    localiser_spacing: types.Spacing3D = None,
    segmentation: bool = False,
    view: types.PatientView = 'axial',
    **kwargs: dict) -> None:
    """
    effect: plots the patient bounding box produced by localiser.
    args:
        id: the patient ID.
        slice_idx: the slice index.
    kwargs:
        aspect: the aspect ratio.
        box_colour: colour of the bounding box.
        clear_cache: force the cache to clear.
        crop: the crop window.
        device: the device to perform network calcs on.
        latex: use latex compiler for text.
        localisation: the 3D box from localisation.
        localisation_seg: the segmentation prediction from localisation.
        localiser: the localiser network.
        localiser_size: the localiser network input size.
        localiser_spacing: the localiser network input voxel spacing.
        segmentation: display the localiser segmentation prediction.
        view: the view plane. 
        **kwargs: all kwargs accepted by 'plot_patient_regions'.
    """
    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Plot patient regions.
    plot_patient_regions(id, slice_idx, aspect=aspect, crop=crop, latex=latex, legend=False, legend_loc=legend_loc, show=False, view=view, **kwargs)

    # Get bounding box (and maybe segmentation).
    if localisation is not None:
        assert localisation_seg is not None
        bounding_box, pred = localisation, localisation_seg
    else:
        assert localiser is not None and localiser_size is not None and localiser_spacing is not None
        bounding_box, pred = get_patient_localisation(id, localiser, localiser_size, localiser_spacing, clear_cache=clear_cache, device=device, return_seg=True)

    # Plot prediction.
    if segmentation:
        # Get aspect ratio.
        if not aspect:
            aspect = _get_aspect_ratio(id, view) 

        # Get slice data.
        pred_slice_data = _get_slice_for_plotting(pred, slice_idx, view)

        # Crop the image.
        if crop:
            pred_slice_data = crop_or_pad_2D(pred_slice_data, _reverse_box_coords_2D(crop))

        # Plot prediction.
        colour = plt.cm.tab20(0)
        colours = [(1, 1, 1, 0), colour]
        cmap = ListedColormap(colours)
        plt.imshow(pred_slice_data, aspect=aspect, cmap=cmap, origin=_get_origin(view))
        plt.plot(0, 0, c=colour, label='Segmentation')

    # Plot bounding box.
    if _should_plot_bounding_box(bounding_box, view, slice_idx):
        _plot_bounding_box(bounding_box, view, box_colour=box_colour, crop=crop, label='Localisation Box')

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

def plot_patient_segmentation(
    id: str,
    slice_idx: int,
    segmenter: nn.Module,
    segmenter_size: types.Size3D,
    segmenter_spacing: types.Spacing3D,
    aspect: float = None,
    crop: types.Box2D = None,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    localiser: nn.Module = None,
    localiser_size: types.Size3D = None,
    localiser_spacing: types.Spacing3D = None,
    localisation_box: bool = None,
    localisation_box_colour: str = 'r',
    segmentation_box_colour: str = 'y',
    segmentation_colour: str = (0.12, 0.47, 0.70, 1.0),
    show_localisation_box: bool = False,
    show_segmentation_box: bool = False,
    view: types.PatientView = 'axial',
    **kwargs: dict) -> None:
    """
    effect: plots the cascader segmentation.
    args:
        id: the patient ID.
        slice_idx: the slice index.
        segmenter: the localiser network.
        segmenter_size: the input size expected by the segmenter.
        segmenter_spacing: the input spacing expected by the segmenter.
    kwargs:
        aspect: the aspect ratio.
        crop: the crop window.
        clear_cache: force the cache to clear.
        device: the device to perform network calcs on.
        legend_loc: the legend location.
        legend_size: the size of the legend.
        localiser: the localiser network.
        localiser_size: the input size of the localiser network.
        localiser_spacing: the voxel spacing for the localiser network.
        localisation_box: the coordinates of the localisation box.
        localisation_box_colour: the colour to use for displaying the localisation box.
        segmentation_box_colour: the colour to use for displaying the segmentation patch box.
        show_localisation_box: display localisation box.
        show_segmentation_box: display segmentation patch box.
        view: the view plane. 
        **kwargs: all kwargs accepted by 'plot_patient_regions'.
    """
    # Plot patient regions.
    plot_patient_regions(id, slice_idx, aspect=aspect, legend=False, legend_loc=legend_loc, show=False, view=view, crop=crop, **kwargs)

    # Get localisation box if not given.
    if not localisation_box:
        localisation_box = get_patient_localisation(id, localiser, localiser_size, localiser_spacing, clear_cache=clear_cache, device=device)

    # Get segmentation prediction.
    seg, seg_patch = get_patient_patch_segmentation(id, localisation_box, segmenter, segmenter_size, segmenter_spacing, clear_cache=clear_cache, device=device, return_patch=True)

    # Get seg slice.
    seg = _get_slice_for_plotting(seg, slice_idx, view)

    # Crop the segmentation.
    if crop:
        seg = crop_or_pad_2D(seg, _reverse_box_coords_2D(crop))

    # Get aspect ratio.
    if not aspect:
        aspect = _get_aspect_ratio(id, view) 

    # Plot segmentation.
    colours = [(1, 1, 1, 0), segmentation_colour]
    cmap = ListedColormap(colours)
    plt.imshow(seg, alpha=0.5, aspect=aspect, cmap=cmap, origin=_get_origin(view))
    plt.plot(0, 0, c=segmentation_colour, label='Segmentation')

    # Plot localisation bounding box.
    if show_localisation_box and _should_plot_bounding_box(localisation_box, view, slice_idx):
        _plot_bounding_box(localisation_box, view, box_colour=localisation_box_colour, crop=crop, label='Localisation Box')

    # Plot segmentation patch.
    if show_segmentation_box and _should_plot_bounding_box(seg_patch, view, slice_idx):
        _plot_bounding_box(seg_patch, view, box_colour=segmentation_box_colour, crop=crop, label='Segmentation Patch')

    # Show legend.
    plt_legend = plt.legend(loc=legend_loc, prop={'size': legend_size})
    for l in plt_legend.get_lines():
        l.set_linewidth(8)

    plt.show()

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

def _get_origin(view: types.PatientView) -> str:
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

def _get_slice_for_plotting(
    data: np.ndarray,
    slice_idx: int,
    view: types.PatientView) -> np.ndarray:
    """
    returns: the slice data ready for plotting.
    args:
        data: the 3D volume.
        slice_idx: the slice index.
        view: the plotting view.
    """
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

def _get_aspect_ratio(
    id: str,
    view: types.PatientView,
    clear_cache: bool = False) -> float:
    """
    returns: the aspect ratio required for the patient view.
    args:
        id: the patient ID.
        view: the view plane.
    kwargs:
        clear_cache: forces the cache to clear.
    """
    # Get patient spacing.
    spacing = dataset.patient(id).ct_spacing(clear_cache=clear_cache)

    # Get the aspect ratio.
    if view == 'axial':
        aspect = spacing[1] / spacing[0]
    elif view == 'coronal':
        aspect = spacing[2] / spacing[0]
    elif view == 'sagittal':
        aspect = spacing[2] / spacing[1]

    return aspect

def _reverse_box_coords_2D(box: types.Box2D) -> types.Box2D:
    """
    returns: a box with (x, y) coordinates reversed.
    args:
        box: the box to swap coordinates for.
    """
    # Reverse coords.
    box = tuple((y, x) for x, y in box)

    return box

def _should_plot_bounding_box(
    bounding_box: types.Box3D,
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
    bbox_min, bbox_max = bounding_box
    bbox_min = bbox_min[dim]
    bbox_max = bbox_max[dim]

    # Return result.
    return slice_idx >= bbox_min and slice_idx < bbox_max

def _plot_bounding_box(
    bounding_box: types.Box3D,
    view: types.PatientView,
    box_colour: str = 'r',
    crop: types.Box2D = None,
    label: str = 'Bounding Box') -> None:
    """
    effect: plots a 2D slice of the bounding box.
    args:
        bounding_box: the bounding box to plot.
        view: the view direction.
    kwargs:
        crop: the cropping applied to the image.
    """
    # Get 2D bounding box.
    if view == 'axial':
        dims = (0, 1)
    elif view == 'coronal':
        dims = (0, 2)
    elif view == 'sagittal':
        dims = (1, 2)
    bbox_min, bbox_max = bounding_box
    bbox_min = np.array(bbox_min)[[*dims]]
    bbox_max = np.array(bbox_max)[[*dims]]
    bbox_width = bbox_max - bbox_min

    # Adjust bounding box for cropped view.
    if crop:
        crop_min, _ = crop
        bbox_min -= crop_min

    # Draw bounding box.
    rect = Rectangle(bbox_min, *bbox_width, linewidth=1, edgecolor=box_colour, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    plt.plot(0, 0, c=box_colour, label=label)

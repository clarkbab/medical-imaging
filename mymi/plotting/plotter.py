from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import torch
from torch import nn
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mymi import dataset
from mymi.geometry import get_extent
from mymi import logging
from mymi.postprocessing import get_largest_cc
from mymi.regions import is_region, RegionColours
from mymi.transforms import crop_or_pad_2D
from mymi import types

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
    colours: Optional[List[str]] = None,
    crop: Optional[Union[types.Box2D, str]] = None,
    crop_margin: int = 20,
    figsize: Tuple[int, int] = (8, 8),
    font_size: int = 10,
    internal_regions: bool = False,
    latex: bool = False,
    legend: bool = True,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    other_ds: str = None,
    other_regions: Union[str, Sequence[str]] = 'all',
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
        other_ds: plot regions from another dataset.
        other_regions: regions to plot from other dataset.
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
    spacing = pat.ct_spacing()

    # Load CT data.
    ct_data = pat.ct_data()

    # Load region data.
    if regions is not None:
        region_data = pat.region_data(regions=regions)

        if internal_regions:
            # Map to internal region names.
            region_data = dict((_to_internal_region(r, clear_cache=clear_cache), d) for r, d in region_data.items())

        # Load other regions.
        if other_ds:
            other_ds = dataset.get(other_ds, 'dicom') 
            other_region_data = other_ds.patient(id).region_data(clear_cache=clear_cache, regions=other_regions)

            if internal_regions:
                # Map to internal region names.
                other_region_data = dict((_to_internal_region(r, clear_cache=clear_cache), d) for r, d in other_region_data.items())

    # Transform data.
    if transform:
        # Add 'batch' dimension.
        ct_data = np.expand_dims(ct_data, axis=0)
        region_data = dict(((n, np.expand_dims(d, axis=0)) for n, d in region_data.items()))
        if other_ds:
            other_region_data = dict(((n, np.expand_dims(d, axis=0)) for n, d in other_region_data.items()))

        # Create 'subject'.
        affine = np.array([
            [spacing[0], 0, 0, 0],
            [0, spacing[1], 0, 0],
            [0, 0, spacing[2], 0],
            [0, 0, 0, 1]
        ])
        ct_data = ScalarImage(tensor=ct_data, affine=affine)
        region_data = dict(((n, LabelMap(tensor=d, affine=affine)) for n, d in region_data.items()))
        if other_ds:
            other_region_data = dict(((n, LabelMap(tensor=d, affine=affine)) for n, d in other_region_data.items()))

        # Transform CT data.
        subject = Subject(input=ct_data)
        output = transform(subject)

        # Transform region data.
        det_transform = output.get_composed_history()
        region_data = dict(((r, det_transform(Subject(region=d))) for r, d in region_data.items()))
        if other_ds:
            other_region_data = dict(((r, det_transform(Subject(region=d))) for r, d in other_region_data.items()))

        # Extract results.
        ct_data = output['input'].data.squeeze(0)
        region_data = dict(((n, o['region'].data.squeeze(0)) for n, o in region_data.items()))
        if other_ds:
            other_region_data = dict(((n, o['region'].data.squeeze(0)) for n, o in other_region_data.items()))

    # Get slice data.
    ct_slice_data = get_slice(ct_data, slice_idx, view)

    # Perform crop.
    if crop:
        if type(crop) == str:
            crop_region_data = region_data[crop]
            extent = get_extent(crop_region_data)
            min, max = extent
            min = tuple(np.array(min) - crop_margin)
            max = tuple(np.array(max) + crop_margin)
            extent_with_margin = (min, max)
            ct_slice_data = crop_or_pad_2D(ct_slice_data, reverse_box_coords_2D(extent_with_margin))
        else:
            ct_slice_data = crop_or_pad_2D(ct_slice_data, reverse_box_coords_2D(crop))

    # Only apply aspect ratio if no transforms are being presented otherwise
    # we might end up with skewed images.
    if not aspect:
        if transform:
            aspect = 1
        else:
            aspect = get_aspect_ratio(id, view, spacing) 

    # Determine plotting window.
    if window:
        width, level = window
        vmin = level - (width / 2)
        vmax = level + (width / 2)
    else:
        vmin, vmax = ct_data.min(), ct_data.max()

    # Plot CT data.
    plt.figure(figsize=figsize)
    plt.imshow(ct_slice_data, cmap='gray', aspect=aspect, origin=get_origin(view), vmin=vmin, vmax=vmax)

    # Add axis labels.
    if axes:
        plt.xlabel('voxel')
        plt.ylabel('voxel')

    if regions:
        # Plot regions.
        show_legend = plot_regions(region_data, slice_idx, alpha, aspect, crop, latex, perimeter, view, colours=colours)

        if other_ds:
            # Prepend other dataset name.
            other_region_data = dict((f"{r} - {other_ds.name}", d) for r, d in other_region_data.items())
 
            # Plot other regions.
            other_show_legend = plot_regions(other_region_data, slice_idx, alpha, aspect, crop, latex, perimeter, view)

        # Create legend.
        if legend and (show_legend or (other_ds and other_show_legend)): 
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
            title_text = _escape_latex(title_text)

        plt.title(title_text)

    if show:
        plt.show()

        # Revert latex settings.
        if latex:
            plt.rcParams.update({
                "font.family": rc_params['font.family'],
                'text.usetex': rc_params['text.usetex']
            })

def plot_regions(
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
                plot_box(extent, view, colour=colour, crop=crop, label=f'{region} extent', linestyle='dashed')

        # Plot connected extent.
        if connected_extent:
            extent = get_extent(get_largest_cc(data))
            if should_plot_box(extent, view, slice_idx):
                plot_box(extent, view, colour='b', crop=crop, label=f'{region} conn. extent', linestyle='dashed')

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
            axis.contour(slice_data, colors=[colour], levels=[0.5])

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
    loc_box: types.Box3D = None,
    loc_seg: np.ndarray = None,
    localiser: nn.Module = None,
    localiser_size: types.ImageSize3D = None,
    localiser_spacing: types.ImageSpacing3D = None,
    show_seg: bool = False,
    view: types.PatientView = 'axial',
    **kwargs: dict) -> None:
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
    if loc_box is not None:
        assert loc_seg is not None
        bounding_box, pred = loc_box, loc_seg
    else:
        assert localiser is not None and localiser_size is not None and localiser_spacing is not None
        bounding_box, pred = get_patient_box(id, localiser, localiser_size, localiser_spacing, clear_cache=clear_cache, device=device, return_seg=True)

    # Plot prediction.
    if show_seg:
        # Get aspect ratio.
        if not aspect:
            aspect = get_aspect_ratio(id, view, spacing) 

        # Get slice data.
        pred_slice_data = get_slice(pred, slice_idx, view)

        # Crop the image.
        if crop:
            pred_slice_data = crop_or_pad_2D(pred_slice_data, reverse_box_coords_2D(crop))

        # Plot prediction.
        colour = plt.cm.tab20(0)
        colours = [(1, 1, 1, 0), colour]
        cmap = ListedColormap(colours)
        plt.imshow(pred_slice_data, aspect=aspect, cmap=cmap, origin=get_origin(view))
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

def plot_localiser_prediction(
    localiser: Tuple[str, str, str],
    pat_id: types.PatientID,
    slice_idx: int,
    aspect: float = None,
    crop: Optional[types.Box2D] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    loc_box_colour: str = 'r',
    show_loc_box: bool = False,
    view: types.PatientView = 'axial',
    **kwargs: dict) -> None:
    # Plot patient regions.
    plot_patient_regions(pat_id, slice_idx, aspect=aspect, legend=False, legend_loc=legend_loc, show=False, view=view, crop=crop, **kwargs)

    # Load prediction.

    # Get localisation box if not given.
    if not loc_box:
        localisation_box = get_patient_box(id, localiser, localiser_size, localiser_spacing, clear_cache=clear_cache, device=device)

    # Get segmentation prediction.
    seg, seg_patch = get_patient_segmentation_patch(id, loc_box, segmenter, segmenter_size, segmenter_spacing, clear_cache=clear_cache, device=device, return_patch=True)

    # Get seg slice.
    seg = get_slice(seg, slice_idx, view)

    # Crop the segmentation.
    if crop:
        seg = crop_or_pad_2D(seg, reverse_box_coords_2D(crop))

    # Get aspect ratio.
    if not aspect:
        aspect = get_aspect_ratio(id, view, spacing) 

    # Plot segmentation.
    colours = [(1, 1, 1, 0), segmentation_colour]
    cmap = ListedColormap(colours)
    plt.imshow(seg, alpha=0.5, aspect=aspect, cmap=cmap, origin=get_origin(view))
    plt.plot(0, 0, c=segmentation_colour, label='Segmentation')

    # Plot localisation bounding box.
    # if show_loc_box and _should_plot_bounding_box(loc_box, view, slice_idx):
    #     _plot_bounding_box(loc_box, view, box_colour=loc_box_colour, crop=crop, label='Localisation Box')

    # Plot segmentation patch.
    if show_seg_box and _should_plot_bounding_box(seg_patch, view, slice_idx):
        _plot_bounding_box(seg_patch, view, box_colour=segmentation_box_colour, crop=crop, label='Segmentation Patch')

    # Show legend.
    plt_legend = plt.legend(loc=legend_loc, prop={'size': legend_size})
    for l in plt_legend.get_lines():
        l.set_linewidth(8)

    plt.show()

def plot_patient_segmentation(
    id: str,
    slice_idx: int,
    segmenter: types.Model,
    segmenter_size: types.ImageSize3D,
    segmenter_spacing: types.ImageSpacing3D,
    aspect: float = None,
    crop: Optional[types.Box2D] = None,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    loc_box: Optional[types.Box3D] = None,
    loc_box_colour: str = 'r',
    localiser: Optional[types.Model] = None,
    localiser_size: Optional[types.ImageSize3D] = None,
    localiser_spacing: Optional[types.ImageSpacing3D] = None,
    segmentation_box_colour: str = 'y',
    segmentation_colour: str = (0.12, 0.47, 0.70, 1.0),
    show_loc_box: bool = False,
    show_seg_box: bool = False,
    view: types.PatientView = 'axial',
    **kwargs: dict) -> None:
    """
    effect: plots the cascader segmentation.
    args:
        id: the patient ID.
        slice_idx: the slice index.
        segmenter: the segmenter network.
        segmenter_size: the input size expected by the segmenter.
        segmenter_spacing: the input spacing expected by the segmenter.
    kwargs:
        aspect: the aspect ratio.
        crop: the crop window.
        clear_cache: force the cache to clear.
        device: the device to perform network calcs on.
        legend_loc: the legend location.
        legend_size: the size of the legend.
        loc_box: the coordinates of the localisation box.
        loc_box_colour: the colour to use for displaying the localisation box.
        localiser: the localiser network.
        localiser_size: the input size of the localiser network.
        localiser_spacing: the voxel spacing for the localiser network.
        segmentation_box_colour: the colour to use for displaying the segmentation patch box.
        show_loc_box: display localisation box.
        show_seg_box: display segmentation patch box.
        view: the view plane. 
        **kwargs: all kwargs accepted by 'plot_patient_regions'.
    """
    # Plot patient regions.
    plot_patient_regions(id, slice_idx, aspect=aspect, legend=False, legend_loc=legend_loc, show=False, view=view, crop=crop, **kwargs)

    # Get localisation box if not given.
    if not loc_box:
        localisation_box = get_patient_box(id, localiser, localiser_size, localiser_spacing, clear_cache=clear_cache, device=device)

    # Get segmentation prediction.
    seg, seg_patch = get_patient_segmentation_patch(id, loc_box, segmenter, segmenter_size, segmenter_spacing, clear_cache=clear_cache, device=device, return_patch=True)

    # Get seg slice.
    seg = get_slice(seg, slice_idx, view)

    # Crop the segmentation.
    if crop:
        seg = crop_or_pad_2D(seg, reverse_box_coords_2D(crop))

    # Get aspect ratio.
    if not aspect:
        aspect = get_aspect_ratio(id, view, spacing) 

    # Plot segmentation.
    colours = [(1, 1, 1, 0), segmentation_colour]
    cmap = ListedColormap(colours)
    plt.imshow(seg, alpha=0.5, aspect=aspect, cmap=cmap, origin=get_origin(view))
    plt.plot(0, 0, c=segmentation_colour, label='Segmentation')

    # Plot localisation bounding box.
    if show_loc_box and _should_plot_bounding_box(loc_box, view, slice_idx):
        _plot_bounding_box(loc_box, view, box_colour=loc_box_colour, crop=crop, label='Localisation Box')

    # Plot segmentation patch.
    if show_seg_box and _should_plot_bounding_box(seg_patch, view, slice_idx):
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

def plot_box(
    box: types.Box3D,
    view: types.PatientView,
    colour: str = 'r',
    crop: types.Box2D = None,
    label: str = 'box',
    linestyle: str = 'solid') -> None:
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
    min, max = box
    min = np.array(min)[[*dims]]
    max = np.array(max)[[*dims]]
    width = max - min

    # Adjust bounding box for cropped view.
    if crop:
        crop_min, _ = crop
        min -= crop_min

    # Draw bounding box.
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

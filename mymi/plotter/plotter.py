from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import re
import torch
from torch import nn
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import Literal, Sequence, Tuple, Union

from mymi import dataset
from mymi.predictions import get_patient_bounding_box
from mymi.regions import is_region, RegionColours

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
    device: nn.Module = None,
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
    view: Literal['axial', 'coronal', 'sagittal'] = 'axial',
    window: Tuple[float, float] = None,
    zoom: Tuple[Tuple[int, int], Tuple[int, int]] = None) -> None:
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
        zoom: zoom to this extent.
    """
    # Get params.
    plt.rcParams.update({
        'font.size': font_size
    })

    # Set latex params.
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Load patient spacing.
    spacing = dataset.patient(id).spacing(clear_cache=clear_cache)

    # Load CT data.
    ct_data = dataset.patient(id).ct_data(clear_cache=clear_cache)

    if regions:
        # Load region data.
        region_data = dataset.patient(id).region_data(clear_cache=clear_cache, regions=regions)

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

    # Get slice data.
    ct_slice_data = _get_slice_data(ct_data, slice_idx, view)

    # Only apply aspect ratio if no transforms are being presented otherwise
    # we might end up with skewed images.
    if not aspect:
        if transform:
            aspect = 1
        else:
            aspect = _get_aspect_ratio(id, view) 

    # Determine plotting window.
    if window:
        vmin, vmax = window
    else:
        vmin, vmax = ct_data.min(), ct_data.max()

    # Plot CT data.
    plt.figure(figsize=figsize)
    if zoom:
        ct_slice_data = _get_zoomed_data(ct_slice_data, zoom)
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
            show_legend = False     # Only show legend if slice has at least one region.
            for i, (region, data) in enumerate(region_data.items()):
                # Convert data to 'imshow' co-ordinate system.
                slice_data = _get_slice_data(data, slice_idx, view)

                # Skip region if not present on this slice.
                if slice_data.max() == 0:
                    continue
                else:
                    show_legend = True
                
                # Create binary colormap for each region.
                if internal_regions:
                    colour = getattr(RegionColours, region)
                else:
                    colour = palette(i)
                colours = [(1.0, 1.0, 1.0, 0), colour]
                region_cmap = ListedColormap(colours)

                # Handle zoom.
                if zoom:
                    slice_data = _get_zoomed_data(slice_data, zoom)

                # Plot region.
                plt.imshow(slice_data, alpha=alpha, aspect=aspect, cmap=region_cmap, origin=_get_origin(view))
                label = _escape_latex(region) if latex else region
                plt.plot(0, 0, c=colour, label=label)
                if perimeter:
                    plt.contour(slice_data, colors=[colour], levels=[0.5])

            # Turn on legend.
            if legend and show_legend: 
                # Get legend props.
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

def plot_patient_bounding_box(
    id: str,
    slice_idx: int,
    localiser: nn.Module,
    box_colour: str = 'y',
    device: torch.device = torch.device('cpu'),
    segmentation: bool = False,
    view: Literal['axial', 'coronal', 'sagittal'] = 'axial',
    zoom: Tuple[Tuple[int, int], Tuple[int, int]] = None,
    **kwargs: dict) -> None:
    """
    effect: plots the patient bounding box produced by localiser.
    args:
        id: the patient ID.
        slice_idx: the slice index.
        localiser: the localiser network.
    kwargs:
        aspect: the aspect ratio.
        device: the device to perform network calcs on.
        segmentation: display the localiser segmentation prediction.
        view: the view plane. 
        zoom: the zoom window.
        **kwargs: all kwargs accepted by 'plot_patient_regions'.
    """
    # Plot patient regions.
    plot_patient_regions(id, slice_idx, show=False, view=view, zoom=zoom, **kwargs)

    # Get bounding box.
    mins, widths, pred = get_patient_bounding_box(id, localiser, device=device, return_prediction=True)

    # Plot prediction.
    if segmentation:
        # Get aspect ratio.
        aspect = _get_aspect_ratio(id, view) 

        # Get slice data.
        pred_slice_data = _get_slice_data(pred, slice_idx, view)

        # Zoom window.
        pred_slice_data = _get_zoomed_data(pred_slice_data, zoom)

        # Plot prediction.
        colours = [(1, 1, 1, 0), plt.cm.tab20(0)]
        cmap = ListedColormap(colours)
        plt.imshow(pred_slice_data, aspect=aspect, cmap=cmap, origin=_get_origin(view))

    # Determine values to use.
    if view == 'axial':
        mins = mins[0], mins[1]
        widths = widths[0], widths[1]
    elif view == 'coronal':
        mins = mins[0], mins[2]
        widths = widths[0], widths[2]
    elif view == 'sagittal':
        mins = mins[1], mins[2]
        widths = widths[1], widths[2]

    # Zoom the rectangle.
    if zoom:
        mins = mins[0] - zoom[0][0], mins[1] - zoom[1][0]

    # Draw rectangle.
    rect = Rectangle(mins, *widths, linewidth=1, edgecolor=box_colour, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)

def _to_image_coords(
    data: ndarray,
    view: Literal['axial', 'coronal', 'sagittal']) -> ndarray:
    """
    returns: data in correct orientation for viewing.
    args:
        data: the data to orient.
        view: the viewing axis.
    """
    # 'plt.imshow' expects (y, x).
    data = np.transpose(data)

    return data

def _get_origin(view: Literal['axial', 'coronal', 'sagittal']) -> str:
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

def _get_slice_data(
    data: np.ndarray,
    slice_idx: int,
    view: Literal['axial', 'coronal', 'sagittal']) -> np.ndarray:
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
    view: Literal['axial', 'coronal', 'sagittal'],
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
    spacing = dataset.patient(id).spacing(clear_cache=clear_cache)

    # Get the aspect ratio.
    if view == 'axial':
        aspect = spacing[1] / spacing[0]
    elif view == 'coronal':
        aspect = spacing[2] / spacing[0]
    elif view == 'sagittal':
        aspect = spacing[2] / spacing[1]

    return aspect

def _get_zoomed_data(
    data: np.ndarray,
    zoom: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    """
    returns: zoomed array.
    args:
        data: the data to zoom.
        zoom: the window to use.
    """
    # Slice data.
    zoom_indices = tuple(slice(z_min, z_max) for z_min, z_max in reversed(zoom))
    data = data[zoom_indices]

    return data

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import torchio
from typing import Optional, Sequence, Tuple, Union

from mymi import dataset as ds
from mymi.prediction.dataset.training import load_localiser_prediction
from mymi.transforms import crop_or_pad_2D
from mymi import types

from ..plotter import get_aspect_ratio, get_origin, get_slice, plot_bounding_box, plot_regions, reverse_box_coords_2D, should_plot_bounding_box

def plot_sample(
    dataset: str,
    partition:  str,
    sample_idx: int,
    alpha: float = 0.2,
    aspect: float = None,
    axes: bool = True,
    centre_of: Optional[str] = None,
    clear_cache: bool = False,
    crop: types.Box2D = None,
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
    regions: types.PatientRegions = 'all',
    show: bool = True,
    slice_idx: Optional[int] = None,
    title: Union[bool, str] = True,
    transform: torchio.transforms.Transform = None,
    view: types.PatientView = 'axial',
    window: Tuple[float, float] = None) -> None:
    # Validate arguments.
    if slice_idx is None and centre_of is None:
        raise ValueError(f"Either 'slice_idx' or 'centre_of' must be set.")

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

    # Load input.
    set = ds.get(dataset, 'training')
    sample = set.partition(partition).sample(sample_idx)
    input = sample.input()

    # Load region data.
    if regions is not None:
        labels = sample.label(regions=regions)

        # # Load other regions.
        # if other_ds:
        #     other_ds = DICOMDataset(other_ds) 
        #     other_region_data = other_ds.patient(id).region_data(clear_cache=clear_cache, regions=other_regions)

    # Get slice index if requested OAR centre.
    if centre_of is not None:
        # Load region data.
        label = sample.label(regions=centre_of)[centre_of]
        com = np.round(center_of_mass(label)).astype(int)
        if view == 'axial':
            slice_idx = com[2]
        elif view == 'coronal':
            slice_idx = com[1]
        elif view == 'sagittal':
            slice_idx = com[0]

    # Get slice data.
    slice_input = get_slice(input, slice_idx, view)

    # Perform crop.
    if crop:
        slice_input = crop_or_pad_2D(slice_input, reverse_box_coords_2D(crop))

    # Determine plotting window.
    if window:
        width, level = window
        vmin = level - (width / 2)
        vmax = level + (width / 2)
    else:
        vmin, vmax = slice_input.min(), slice_input.max()

    # Plot CT data.
    plt.figure(figsize=figsize)
    plt.imshow(slice_input, cmap='gray', origin=get_origin(view), vmin=vmin, vmax=vmax)

    # Add axis labels.
    if axes:
        plt.xlabel('voxel')
        plt.ylabel('voxel')

    if regions:
        # Plot regions.
        aspect = 1
        show_legend = plot_regions(labels, slice_idx, alpha, aspect, crop, latex, perimeter, view) 

        # if other_ds:
        #     # Prepend other dataset name.
        #     other_region_data = dict((f"{r} - {other_ds.name}", d) for r, d in other_region_data.items())
 
        #     # Plot other regions.
        #     other_show_legend = _plot_regions(other_region_data, alpha, aspect, crop, internal_regions, latex, perimeter, slice_idx, view)

        # Create legend.
        # if legend and (show_legend or (other_ds and other_show_legend)): 
        if legend and show_legend:
            plt_legend = plt.legend(loc=legend_loc, prop={'size': legend_size})
            for l in plt_legend.get_lines():
                l.set_linewidth(8)

    # Show axis markers.
    show_axes = 'on' if axes else 'off'
    plt.axis(show_axes)

    # Determine number of slices.
    if view == 'axial':
        num_slices = input.shape[2]
    elif view == 'coronal':
        num_slices = input.shape[1]
    elif view == 'sagittal':
        num_slices = input.shape[0]

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

def plot_localiser_prediction(
    dataset: str,
    partition: str,
    sample_idx: int,
    localiser: Tuple[str, str, str],
    aspect: float = None,
    centre_of: Optional[str] = None,
    crop: Optional[types.Box2D] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    loc_box_colour: str = 'r',
    seg_alpha: float = 1.0, 
    seg_colour: types.Colour = (0.12, 0.47, 0.70),
    show_loc_box: bool = False,
    slice_idx: Optional[int] = None,
    view: types.PatientView = 'axial',
    **kwargs: dict) -> None:
    # Validate arguments.
    if slice_idx is None and centre_of is None:
        raise ValueError(f"Either 'slice_idx' or 'centre_of' must be set.")

    # Load sample.
    set = ds.get(dataset, 'training')
    sample = set.partition(partition).sample(sample_idx)

    # Centre on OAR if requested.
    if slice_idx is None:
        # Load region data.
        label = sample.label(regions=centre_of)[centre_of]
        com = np.round(center_of_mass(label)).astype(int)
        if view == 'axial':
            slice_idx = com[2]
        elif view == 'coronal':
            slice_idx = com[1]
        elif view == 'sagittal':
            slice_idx = com[0]

    # Plot patient regions.
    plot_sample(dataset, partition, sample_idx, aspect=aspect, legend=False, legend_loc=legend_loc, show=False, slice_idx=slice_idx, view=view, crop=crop, **kwargs)

    # Load prediction.
    box, pred = load_localiser_prediction(dataset, partition, sample_idx, localiser, return_seg=True)
    pred_slice = get_slice(pred, slice_idx, view)

    # Crop the segmentation.
    if crop:
        pred_slice = crop_or_pad_2D(pred_slice, reverse_box_coords_2D(crop))

    # Get aspect ratio.
    if not aspect:
        set = ds.get(dataset, 'training')
        spacing = eval(set.params()['spacing'][0])
        aspect = get_aspect_ratio(id, view, spacing) 

    # Plot segmentation.
    colours = [(1, 1, 1, 0), seg_colour]
    cmap = ListedColormap(colours)
    plt.imshow(pred_slice, alpha=seg_alpha, aspect=aspect, cmap=cmap, origin=get_origin(view))
    plt.plot(0, 0, c=seg_colour, label='Segmentation')

    # Plot bounding box.
    if should_plot_bounding_box(box, view, slice_idx):
        plot_bounding_box(box, view, box_colour=loc_box_colour, crop=crop, label='Loc. box')

    # Show legend.
    plt_legend = plt.legend(loc=legend_loc, prop={'size': legend_size})
    for l in plt_legend.get_lines():
        l.set_linewidth(8)

    plt.show()

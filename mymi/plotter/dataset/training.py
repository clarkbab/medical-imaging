from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import torchio
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

from mymi import dataset as ds
from mymi.geometry import get_extent, get_extent_centre
from mymi.prediction.dataset.training import load_localiser_prediction
from mymi.transforms import crop_or_pad_2D
from mymi import types

from ..plotter import assert_position, get_aspect_ratio, get_origin, get_slice, plot_box, plot_regions, reverse_box_coords_2D, should_plot_box

def plot_sample_regions(
    dataset: str,
    partition: str,
    sample_id: str,
    alpha: float = 0.2,
    aspect: float = None,
    axis = None,
    cca: bool = False,
    centre_of: Optional[str] = None,
    colours: Optional[List[str]] = None,
    crop: Optional[Union[str, Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    crop_margin: float = 40,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    figsize: Tuple[int, int] = (8, 8),
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
    window: Tuple[float, float] = None) -> None:
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
    set = ds.get(dataset, 'training')
    sample = set.partition(partition).sample(sample_id)
    spacing = eval(set.params().spacing[0])

    # Get slice index if requested OAR centre.
    if centre_of:
        # Get extent centre.
        label = sample.label(regions=centre_of)[centre_of]
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
        label = sample.label(regions=eo_region)[eo_region]
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
    input = sample.input()

    # Load region data.
    if regions is not None:
        label = sample.label(regions=regions)
        if postproc:
            label = dict(((r, postproc(d)) for r, d in label.items()))

        # Load other regions.
        if other_ds:
            other_ds = DICOMDataset(other_ds) 
            other_region_data = other_ds.patient(sample_id).region_data(clear_cache=clear_cache, regions=other_regions)

            if internal_regions:
                # Map to internal region names.
                other_region_data = dict((_to_internal_region(r, clear_cache=clear_cache), d) for r, d in other_region_data.items())

    # Get slice data.
    input_slice = get_slice(input, slice_idx, view)

    # Convert to box representation.
    if crop:
        # Check if crop is region name.
        if type(crop) == str:
            # Get 3D crop box.
            crop_label = label[crop]
            extent = get_extent(crop_label)

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
        else:
            crop = ((crop[0][0], crop[1][0]), (crop[0][1], crop[1][1]))

    # Perform crop.
    if crop:
        # Convert crop to 2D box.
        input_slice = crop_or_pad_2D(input_slice, reverse_box_coords_2D(crop))

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
        vmin, vmax = input.min(), input.max()

    # Plot CT data.
    axis.imshow(input_slice, cmap='gray', aspect=aspect, interpolation='none', origin=get_origin(view), vmin=vmin, vmax=vmax)

    # Determine voxel spacing per axis.
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
        show_legend = plot_regions(label, slice_idx, alpha, aspect, latex, perimeter, view, axis=axis, cca=cca, colours=colours, crop=crop, show_extent=show_extent)

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
            title_text = f"{sample} - {slice_idx}/{num_slices - 1} ({view})"

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
        aspect = get_aspect_ratio(view, spacing) 

    # Plot segmentation.
    colours = [(1, 1, 1, 0), seg_colour]
    cmap = ListedColormap(colours)
    plt.imshow(pred_slice, alpha=seg_alpha, aspect=aspect, cmap=cmap, origin=get_origin(view))
    plt.plot(0, 0, c=seg_colour, label='Segmentation')

    # Plot bounding box.
    if should_plot_box(box, view, slice_idx):
        plot_box(box, view, box_colour=loc_box_colour, crop=crop, label='Loc. box')

    # Show legend.
    plt_legend = plt.legend(loc=legend_loc, prop={'size': legend_size})
    for l in plt_legend.get_lines():
        l.set_linewidth(8)

    plt.show()

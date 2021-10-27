from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import torchio
from typing import Optional, Sequence, Tuple, Union

from mymi import dataset as ds
from mymi.postprocessing import get_extent, get_extent_centre
from mymi.prediction.dataset.nifti import load_localiser_prediction
from mymi.transforms import crop_or_pad_2D
from mymi import types

from ..plotter import assert_position, get_aspect_ratio, get_origin, get_slice, plot_box, plot_regions, reverse_box_coords_2D, should_plot_box

def plot_patient_regions(
    dataset: str,
    pat_id: str,
    alpha: float = 0.2,
    aspect: float = None,
    axes: bool = True,
    centre_on: Optional[str] = None,
    crop: types.Box2D = None,
    figsize: Tuple[int, int] = (8, 8),
    font_size: int = 10,
    latex: bool = False,
    legend: bool = True,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    other_ds: str = None,
    other_regions: Union[str, Sequence[str]] = 'all',
    perimeter: bool = True,
    regions: Union[str, Sequence[str]] = 'all',
    show: bool = True,
    slice_idx: Optional[int] = None,
    title: Union[bool, str] = True,
    transform: torchio.transforms.Transform = None,
    view: types.PatientView = 'axial',
    window: Tuple[float, float] = None) -> None:
    assert_position(centre_on, slice_idx)

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
    if centre_on:
        # Load region data.
        label = pat.region_data(regions=centre_on)[centre_on]
        com = np.round(center_of_mass(label)).astype(int)
        if view == 'axial':
            slice_idx = com[2]
        elif view == 'coronal':
            slice_idx = com[1]
        elif view == 'sagittal':
            slice_idx = com[0]
    else:
        if not slice_idx: 
            raise ValueError(f"Either 'centre_on' or 'slice_idx' must be given.")

    # Load CT data.
    ct_data = pat.ct_data()

    # Load region data.
    if regions is not None:
        region_data = pat.region_data(regions=regions)

        # Load other regions.
        if other_ds:
            other_ds = DICOMDataset(other_ds) 
            other_region_data = other_ds.patient(pat_id).region_data(clear_cache=clear_cache, regions=other_regions)

            if internal_regions:
                # Map to internal region names.
                other_region_data = dict((_to_internal_region(r, clear_cache=clear_cache), d) for r, d in other_region_data.items())

    # Get slice data.
    ct_slice_data = get_slice(ct_data, slice_idx, view)

    # Perform crop.
    if crop:
        ct_slice_data = crop_or_pad_2D(ct_slice_data, reverse_box_coords_2D(crop))

    # Only apply aspect ratio if no transforms are being presented otherwise
    # we might end up with skewed images.
    if not aspect:
        if transform:
            aspect = 1
        else:
            aspect = get_aspect_ratio(pat_id, view, spacing) 

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

        plt.xlabel(f'voxel [@ {spacing_x:.3f} mm spacing]')
        plt.ylabel(f'voxel [@ {spacing_y:.3f} mm spacing]')

    if regions:
        # Plot regions.
        show_legend = plot_regions(region_data, slice_idx, alpha, aspect, crop, latex, perimeter, view)

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


def plot_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: Tuple[str, str, str],
    aspect: float = None,
    box_colour: str = 'r',
    centre_on: Optional[str] = None,
    crop: types.Box2D = None,
    latex: bool = False,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_size: int = 10,
    loc_box_colour: str = 'r',
    loc_box_margin: Optional[float] = None,
    seg_patch_size: Optional[types.ImageSize3D] = None,
    show_seg: bool = False,
    slice_idx: Optional[int] = None,
    view: types.PatientView = 'axial',
    **kwargs: dict) -> None:
    assert_position(centre_on, slice_idx)

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
        label = patient.region_data(regions=centre_on)[centre_on]
        com = np.round(center_of_mass(label)).astype(int)
        if view == 'axial':
            slice_idx = com[2]
        elif view == 'coronal':
            slice_idx = com[1]
        elif view == 'sagittal':
            slice_idx = com[0]

    # Plot patient regions.
    plot_patient_regions(dataset, pat_id, aspect=aspect, crop=crop, latex=latex, legend=False, legend_loc=legend_loc, show=False, slice_idx=slice_idx, view=view, **kwargs)

    # Load localiser segmentation.
    pred = load_localiser_prediction(dataset, pat_id, localiser)

    # Get extent.
    extent = get_extent(pred)

    # Plot prediction.
    if show_seg:
        # Get aspect ratio.
        if not aspect:
            aspect = get_aspect_ratio(pat_id, view, spacing) 

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
    if should_plot_box(extent, view, slice_idx):
        plot_box(extent, view, colour=box_colour, crop=crop, label='Loc. Box')

    # Plot seg box.
    if seg_patch_size:
        # Get extent centre.
        centre = get_extent_centre(pred)
        low = np.floor(np.array(seg_patch_size) / 2).astype(int)
        high = seg_patch_size - low
        min = np.clip(centre - low, 0, None)
        max = centre + high
        seg_patch = (min, max)
        
        # Plot segmentation patch.
        if should_plot_box(seg_patch, view, slice_idx):
            plot_box(seg_patch, view, colour=box_colour, crop=crop, label='Seg. Patch', linestyle='dashed')

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

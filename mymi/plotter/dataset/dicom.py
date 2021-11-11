from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import torchio
from typing import Optional, Sequence, Tuple, Union

from mymi import dataset as ds
from mymi.postprocessing import get_extent, get_extent_centre
from mymi.transforms import crop_or_pad_2D
from mymi import types

from ..plotter import assert_position, get_aspect_ratio, get_origin, get_slice, plot_box, plot_regions, reverse_box_coords_2D, should_plot_box

def plot_patient_regions(
    dataset: str,
    pat_id: str,
    alpha: float = 0.2,
    aspect: float = None,
    axes: bool = True,
    centre_of: Optional[str] = None,
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
    use_mapping: bool = True,
    view: types.PatientView = 'axial',
    window: Tuple[float, float] = None) -> None:
    assert_position(centre_of, slice_idx)

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
    set = ds.get(dataset, 'dicom')
    pat = set.patient(pat_id)
    spacing = pat.ct_spacing()

    # Get slice index if requested OAR centre.
    if centre_of:
        # Load region data.
        label = pat.region_data(regions=centre_of, use_mapping=use_mapping)[centre_of]
        com = np.round(center_of_mass(label)).astype(int)
        if view == 'axial':
            slice_idx = com[2]
        elif view == 'coronal':
            slice_idx = com[1]
        elif view == 'sagittal':
            slice_idx = com[0]
    else:
        if not slice_idx: 
            raise ValueError(f"Either 'centre_of' or 'slice_idx' must be given.")

    # Load CT data.
    ct_data = pat.ct_data()

    # Load region data.
    if regions is not None:
        region_data = pat.region_data(regions=regions, use_mapping=use_mapping)

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
        plt.xlabel('voxel')
        plt.ylabel('voxel')

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